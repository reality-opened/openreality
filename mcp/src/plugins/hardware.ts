/**
 * <namespace>_* tools: one hardware device (a robot arm, a drone) bridged from
 * the device's own MCP server. createApp mounts this plugin once per device in
 * hardware.json (src/hardware.ts). The fiber starts the device server as a
 * child process over stdio, reads its tool list once, and registers each tool
 * here as `<namespace>_<tool>` (a tool already named `<namespace>_...` keeps its
 * name), plus `<namespace>_guide` carrying the device's own instructions.
 *
 * Calls and results pass through unedited: the device owns its safety checks,
 * so its refusals, images and structured results reach the model verbatim and
 * nothing here retries a call. Disposing the fiber removes the tools and stops
 * the child process.
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import {
  CallToolResultSchema,
  ErrorCode,
  McpError,
  type CallToolResult,
  type Progress,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';
import type { Context } from 'cordis';
import packageJson from '../../package.json';
import { HardwareBridgeError, hardwareDeviceSchema, type HardwareDevice } from '../hardware.js';
import { jsonSchemaToShape } from '../jsonschema.js';
import { textResult, type ProgressExtra } from '../toolkit.js';

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'hardware-bridge';

/** Services required before this plugin activates; device calls never touch the broker. */
export const inject = ['mcp'];

export const Config = hardwareDeviceSchema;

const HEARTBEAT_MS = 10_000;

type CallExtra = ProgressExtra & { readonly signal: AbortSignal };

interface ProgressReport {
  readonly relay: ((progress: Progress) => void) | undefined;
  readonly stop: () => void;
}

/**
 * While a device call runs, tell a caller that asked for progress: relay the
 * device's own progress, and send a heartbeat so the caller's idle timeout does
 * not end a long call (a full arm calibration takes minutes).
 */
function reportProgress(extra: CallExtra, tool: string): ProgressReport {
  const token = extra._meta?.progressToken;
  const send = extra.sendNotification;
  if (token === undefined || !send) return { relay: undefined, stop: () => undefined };
  const started = Date.now();
  let last = 0;
  const notify = (message: string): void => {
    last = Math.max(last + 0.001, (Date.now() - started) / 1000);
    send({ method: 'notifications/progress', params: { progressToken: token, progress: last, message } }).catch(() => {
      // Progress is best-effort; a failed notification must never fail the device call.
    });
  };
  const timer = setInterval(() => notify(`${tool} running (${Math.round((Date.now() - started) / 1000)} s)`), HEARTBEAT_MS);
  return {
    relay: (progress) =>
      notify(progress.message ?? `${tool}: ${progress.progress}${progress.total === undefined ? '' : ` of ${progress.total}`}`),
    stop: () => clearInterval(timer),
  };
}

function hintFor(device: HardwareDevice, code: number): string | undefined {
  switch (code) {
    case ErrorCode.RequestTimeout:
      return `no answer within timeout_s = ${device.timeout_s} s; the device may still be carrying the call out, so check its state before the next motion`;
    case ErrorCode.ConnectionClosed:
      return 'the device server is not running any more; restart openreality-mcp to start it again';
    default:
      return undefined;
  }
}

/** A call that produced no device result (timeout, device server gone, protocol error), as a tool error. */
function failure(device: HardwareDevice, tool: string, err: McpError): CallToolResult {
  const hint = hintFor(device, err.code);
  const error = { code: err.code, message: err.message, ...(err.data === undefined ? {} : { data: err.data }) };
  const body = { device: device.namespace, tool, error, ...(hint === undefined ? {} : { hint }) };
  return { isError: true, content: [{ type: 'text', text: JSON.stringify(body, null, 2) }] };
}

async function listTools(client: Client): Promise<Tool[]> {
  const tools: Tool[] = [];
  let cursor: string | undefined;
  do {
    const page = await client.listTools(cursor === undefined ? undefined : { cursor });
    tools.push(...page.tools);
    cursor = page.nextCursor;
  } while (cursor !== undefined);
  return tools;
}

function guideText(device: HardwareDevice, client: Client, toolMap: readonly string[]): string {
  const server = client.getServerVersion();
  const origin = server ? ` (its MCP server: ${server.name}${server.version ? ` ${server.version}` : ''})` : '';
  return [
    `Device "${device.namespace}"${origin} runs on this machine and is bridged here.`,
    `Its instructions below name its tools without the "${device.namespace}_" prefix; the tool map at the end gives the tool to call here.`,
    '',
    client.getInstructions() ?? '(The device server sent no instructions.)',
    '',
    'Tool map (device name -> tool here):',
    ...toolMap,
  ].join('\n');
}

export async function apply(ctx: Context, device: HardwareDevice): Promise<void> {
  const { mcp } = ctx;
  const ns = device.namespace;
  const client = new Client({ name: `openreality-mcp/${ns}`, version: packageJson.version });
  let connected = true;
  client.onclose = () => {
    connected = false;
  };
  await ctx.effect(async () => {
    const transport = new StdioClientTransport({
      command: device.command,
      args: device.args,
      ...(device.cwd === undefined ? {} : { cwd: device.cwd }),
      env: { ...getDefaultEnvironment(), ...device.env },
      stderr: 'inherit',
    });
    try {
      await client.connect(transport);
    } catch (err) {
      const command = [device.command, ...device.args].join(' ');
      const reason = err instanceof Error ? err.message : String(err);
      throw new HardwareBridgeError(ns, `its server did not start (${command}): ${reason}`, { cause: err });
    }
    return () => client.close();
  }, `hardware(${ns})`);

  const tools = await listTools(client);
  const exposed = (deviceTool: string): string => (deviceTool.startsWith(`${ns}_`) ? deviceTool : `${ns}_${deviceTool}`);
  const guide = `${ns}_guide`;
  const names = [...tools.map((tool) => exposed(tool.name)), guide];
  const clash = names.find((toolName, index) => names.indexOf(toolName) !== index);
  if (clash !== undefined) throw new HardwareBridgeError(ns, `two of its tools would both be called ${clash}`);

  const forward = async (deviceTool: string, args: Record<string, unknown>, extra: CallExtra): Promise<CallToolResult> => {
    const tool = exposed(deviceTool);
    if (!connected) return failure(device, tool, new McpError(ErrorCode.ConnectionClosed, 'Connection closed'));
    const progress = reportProgress(extra, tool);
    try {
      return await client.request(
        { method: 'tools/call', params: { name: deviceTool, arguments: args } },
        CallToolResultSchema,
        { signal: extra.signal, timeout: device.timeout_s * 1000, ...(progress.relay ? { onprogress: progress.relay } : {}) },
      );
    } catch (err) {
      if (err instanceof McpError) return failure(device, tool, err);
      throw err;
    } finally {
      progress.stop();
    }
  };

  const guideBody = guideText(device, client, tools.map((tool) => `- ${tool.name} -> ${exposed(tool.name)}`));
  mcp.tool(
    ctx,
    guide,
    {
      title: `${ns}: device guide`,
      description:
        `Read this before the first ${ns}_* call: the ${ns} device's own operating rules (units, ` +
        'coordinate frames, what it refuses and why), verbatim, and which tool here each of its tool names means.',
      annotations: { readOnlyHint: true },
    },
    () => textResult(guideBody),
  );

  for (const tool of tools) {
    mcp.tool(
      ctx,
      exposed(tool.name),
      {
        title: tool.title,
        description: tool.description,
        inputSchema: jsonSchemaToShape(tool.inputSchema),
        annotations: tool.annotations,
      },
      (args, extra) => forward(tool.name, args, extra),
    );
  }
}
