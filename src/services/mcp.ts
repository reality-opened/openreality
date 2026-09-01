/**
 * `ctx.mcp` — the MCP server capability: owns the @modelcontextprotocol/sdk
 * McpServer and exposes effect-scoped registration. Every tool/resource a
 * plugin contributes is installed through `ctx.effect()` with the SDK handle's
 * `remove()` as the disposer, so unloading a tool-namespace plugin (or the
 * whole app) unregisters its surface — registrations are reversible effects,
 * never global mutations.
 */
import { Context, Service } from 'cordis';
import { z } from 'zod';
import {
  McpServer,
  ResourceTemplate,
  type ReadResourceCallback,
  type ReadResourceTemplateCallback,
  type ResourceMetadata,
  type ToolCallback,
} from '@modelcontextprotocol/sdk/server/mcp.js';
import type {
  AnySchema,
  ZodRawShapeCompat,
} from '@modelcontextprotocol/sdk/server/zod-compat.js';
import type { ToolAnnotations } from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';

declare module 'cordis' {
  interface Context {
    mcp: Mcp;
  }
}

export interface McpConfig {
  /** MCP server name reported to clients. */
  name: string;
  /** MCP server version reported to clients. */
  version: string;
  /** Server-level instructions surfaced to the model. */
  instructions?: string;
}

const configSchema = z.object({
  name: z.string().min(1),
  version: z.string().min(1),
  instructions: z.string().optional(),
});

export interface ToolConfig<
  OutputArgs extends ZodRawShapeCompat | AnySchema,
  InputArgs extends undefined | ZodRawShapeCompat | AnySchema,
> {
  title?: string;
  description?: string;
  inputSchema?: InputArgs;
  outputSchema?: OutputArgs;
  annotations?: ToolAnnotations;
}

export class Mcp extends Service {
  static provide = 'mcp';
  static Config = configSchema;

  readonly server: McpServer;

  constructor(ctx: Context, config: McpConfig) {
    super(ctx, 'mcp');
    this.server = new McpServer(
      { name: config.name, version: config.version },
      config.instructions !== undefined ? { instructions: config.instructions } : {},
    );
    // The SDK initializes the tools/resources capabilities lazily on the FIRST
    // registration and refuses to do so once a transport is connected. Prime
    // both here (register + remove is silent pre-connect) so tool plugins may
    // mount and unmount freely after `connect()` — hot composition changes
    // only ever emit list_changed notifications.
    this.server.registerTool('__cordis_init__', {}, () => ({ content: [] })).remove();
    this.server
      .registerResource('__cordis_init__', 'internal://cordis-init', {}, () => ({ contents: [] }))
      .remove();
  }

  /**
   * Register one tool as a reversible effect on the CALLER's fiber: disposal
   * of the registering plugin removes the tool from the live server.
   * @param ctx - the registering plugin's context (owns the effect).
   */
  tool<
    OutputArgs extends ZodRawShapeCompat | AnySchema,
    InputArgs extends undefined | ZodRawShapeCompat | AnySchema = undefined,
  >(
    ctx: Context,
    name: string,
    config: ToolConfig<OutputArgs, InputArgs>,
    cb: ToolCallback<InputArgs>,
  ): void {
    ctx.effect(() => {
      const handle = this.server.registerTool(name, config, cb);
      return () => handle.remove();
    }, `mcp.tool(${name})`);
  }

  /**
   * Register one resource (fixed URI or template) as a reversible effect on
   * the CALLER's fiber, mirroring `tool()`.
   * @param ctx - the registering plugin's context (owns the effect).
   */
  resource(
    ctx: Context,
    name: string,
    uri: string,
    metadata: ResourceMetadata,
    read: ReadResourceCallback,
  ): void;
  resource(
    ctx: Context,
    name: string,
    template: ResourceTemplate,
    metadata: ResourceMetadata,
    read: ReadResourceTemplateCallback,
  ): void;
  resource(
    ctx: Context,
    name: string,
    uriOrTemplate: string | ResourceTemplate,
    metadata: ResourceMetadata,
    read: ReadResourceCallback | ReadResourceTemplateCallback,
  ): void {
    ctx.effect(() => {
      const handle =
        typeof uriOrTemplate === 'string'
          ? this.server.registerResource(name, uriOrTemplate, metadata, read as ReadResourceCallback)
          : this.server.registerResource(name, uriOrTemplate, metadata, read as ReadResourceTemplateCallback);
      return () => handle.remove();
    }, `mcp.resource(${name})`);
  }

  /** Connect the underlying server to a transport (stdio in the CLI). */
  connect(transport: Transport): Promise<void> {
    return this.server.connect(transport);
  }
}

export default Mcp;
