/**
 * Device argument schemas re-advertised from JSON Schema (src/jsonschema.ts):
 * a bridged tool must advertise the arguments the device declared, accept what
 * the device accepts, and never drop an argument it cannot model. The schemas
 * are the real SO-101 arm server's (test/fixtures/artist-arm-tools.json).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { jsonSchemaToShape } from '../src/jsonschema.js';

interface CapturedTool {
  readonly name: string;
  readonly inputSchema: Record<string, unknown>;
}

const captured: { readonly tools: readonly CapturedTool[] } = JSON.parse(
  readFileSync(new URL('./fixtures/artist-arm-tools.json', import.meta.url), 'utf8'),
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** What a client reads off one argument: its JSON types or allowed values, nullable or not. */
function contract(node: unknown): string {
  if (!isRecord(node)) return 'any';
  if (Array.isArray(node.enum)) return `one of ${node.enum.map(String).sort().join('|')}`;
  const variants = Array.isArray(node.anyOf)
    ? node.anyOf
    : Array.isArray(node.type)
      ? node.type.map((type: unknown) => ({ ...node, type }))
      : undefined;
  if (variants) return variants.map(contract).sort().join(' or ');
  if (node.type === 'array') return `array of ${contract(node.items)}`;
  if (node.type === 'object' && isRecord(node.additionalProperties)) return `map of ${contract(node.additionalProperties)}`;
  return typeof node.type === 'string' ? node.type : 'any';
}

function propertiesOf(schema: unknown): Record<string, unknown> {
  return isRecord(schema) && isRecord(schema.properties) ? schema.properties : {};
}

function requiredOf(schema: unknown): string[] {
  return isRecord(schema) && Array.isArray(schema.required) ? schema.required.map(String).sort() : [];
}

function argsOf(name: string) {
  const tool = captured.tools.find((candidate) => candidate.name === name);
  if (!tool) throw new Error(`no captured tool ${name}`);
  return z.object(jsonSchemaToShape(tool.inputSchema));
}

describe('device argument schemas (JSON Schema to zod)', () => {
  it('re-advertises every arm tool with the same arguments, required set, types and defaults', async () => {
    // Given: every captured arm tool registered from its converted schema
    const server = new McpServer({ name: 'jsonschema-test', version: '0.0.0' });
    for (const { name, inputSchema } of captured.tools) {
      server.registerTool(name, { inputSchema: jsonSchemaToShape(inputSchema) }, () => ({ content: [] }));
    }
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: 'jsonschema-test', version: '0.0.0' });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    // When: a client lists them
    const listed = new Map((await client.listTools()).tools.map((tool) => [tool.name, tool.inputSchema]));
    await client.close();

    // Then: each argument reads the same as on the device
    for (const { name, inputSchema } of captured.tools) {
      const declared = propertiesOf(inputSchema);
      const advertised = propertiesOf(listed.get(name));
      expect(Object.keys(advertised).sort(), name).toEqual(Object.keys(declared).sort());
      expect(requiredOf(listed.get(name)), name).toEqual(requiredOf(inputSchema));
      for (const [key, property] of Object.entries(declared)) {
        const out = advertised[key];
        expect(contract(out), `${name}.${key}`).toBe(contract(property));
        if (isRecord(property) && 'default' in property) {
          expect(isRecord(out) ? out.default : undefined, `${name}.${key} default`).toEqual(property.default);
        }
      }
    }
  });

  it.each([
    ['release_torque', { which: 'all' }, true],
    ['release_torque', { which: 'everything' }, false],
    ['move_joints', { targets: { elbow_flex: -10, gripper: 50 } }, true],
    ['move_joints', { targets: 'nope' }, false],
    ['move_to', { x_mm: 250, y_mm: 0 }, false],
    ['auto_calibration', { enabled: null }, true],
  ] as const)('%s(%j) accepted: %s', (name, args, accepted) => {
    expect(argsOf(name).safeParse(args).success).toBe(accepted);
  });

  it('fills an omitted optional argument with the default the device declared', () => {
    expect(argsOf('look').parse({})).toEqual({ camera: 'gemini', annotate: true, grid: false });
  });

  it('passes an argument it cannot model through unchanged instead of dropping it', () => {
    const args = z.object(
      jsonSchemaToShape({
        type: 'object',
        properties: { pose: { not: { type: 'string' } }, mode: { allOf: [{ type: 'string' }, { minLength: 2 }] } },
        required: ['pose'],
      }),
    );
    expect(args.parse({ pose: { x: 1 }, mode: 'fast' })).toEqual({ pose: { x: 1 }, mode: 'fast' });
  });

  it('resolves a local $defs reference, with the referenced object checks', () => {
    const args = z.object(
      jsonSchemaToShape({
        type: 'object',
        $defs: { Point: { type: 'object', properties: { x_mm: { type: 'number' } }, required: ['x_mm'] } },
        properties: { target: { $ref: '#/$defs/Point' } },
        required: ['target'],
      }),
    );
    expect(args.safeParse({ target: { x_mm: 'far' } }).success).toBe(false);
  });

  it('stops at a self-referencing schema instead of recursing forever', () => {
    const args = z.object(
      jsonSchemaToShape({
        type: 'object',
        $defs: { Node: { type: 'object', properties: { next: { $ref: '#/$defs/Node' } } } },
        properties: { head: { $ref: '#/$defs/Node' } },
      }),
    );
    expect(args.safeParse({ head: { next: { next: {} } } }).success).toBe(true);
  });
});
