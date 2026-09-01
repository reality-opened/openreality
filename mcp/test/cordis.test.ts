/**
 * Cordis lifecycle acceptance: tool namespaces are plugins whose registrations
 * are reversible effects — disposing a fiber removes its tools from the live
 * MCP server, remounting restores them, and a plugin that declares `inject`
 * stays pending until its services exist. A hand-built subset composition is
 * exactly how embedders mount less than the full surface.
 */
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Context } from 'cordis';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { startSimulator, type RunningSimulator } from '../simulator/sim.js';
import { Broker } from '../src/services/broker.js';
import { Mcp } from '../src/services/mcp.js';
import { createApp } from '../src/app.js';
import type { Config } from '../src/config.js';
import * as sceneTools from '../src/plugins/scene.js';
import * as workspaceTools from '../src/plugins/workspace.js';

let sim: RunningSimulator;

function simConfig(): Config {
  const dir = mkdtempSync(join(tmpdir(), 'orl-cordis-'));
  return {
    baseUrl: sim.url,
    loginBaseUrl: sim.url,
    envToken: 'cordis-test-token',
    configDir: dir,
    credentialsPath: join(dir, 'credentials.json'),
    statePath: join(dir, 'state.json'),
    artifactsDir: join(dir, 'artifacts'),
  };
}

async function connectedClient(ctx: Context): Promise<Client> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: 'cordis-lifecycle', version: '0.0.0' });
  await Promise.all([ctx.mcp.connect(serverTransport), client.connect(clientTransport)]);
  return client;
}

async function toolNames(client: Client): Promise<string[]> {
  return (await client.listTools()).tools.map((t) => t.name);
}

beforeAll(async () => {
  sim = await startSimulator({ port: 0 });
});

afterAll(async () => {
  await sim?.close();
});

describe('cordis composition lifecycle', () => {
  it('mounts the full app and serves the complete tool surface', async () => {
    const ctx = await createApp(simConfig());
    const client = await connectedClient(ctx);
    const names = await toolNames(client);
    expect(names.length).toBeGreaterThanOrEqual(40);
    expect(names).toContain('workspace_list_scenes');
    expect(names).toContain('scene_card');
    expect(names).toContain('agent_chat');
    expect(names).toContain('export_prepare');
    const resources = await client.listResources();
    expect(resources.resources.map((r) => r.uri)).toContain('openreality://scenes');
    await client.close();
    await ctx.fiber.dispose();
  });

  it('disposing a namespace fiber unregisters exactly its tools; remounting restores them', async () => {
    const ctx = new Context();
    await Promise.all([
      ctx.plugin(Broker, simConfig()),
      ctx.plugin(Mcp, { name: 'openreality-test', version: '0.0.0' }),
    ]);
    const client = await connectedClient(ctx);

    const sceneFiber = await ctx.plugin(sceneTools);
    await ctx.plugin(workspaceTools);
    const before = await toolNames(client);
    expect(before).toContain('scene_card');
    expect(before).toContain('workspace_list_scenes');

    await sceneFiber.dispose();
    const after = await toolNames(client);
    expect(after).not.toContain('scene_card');
    expect(after).not.toContain('scene_measure_distance');
    expect(after).toContain('workspace_list_scenes'); // the sibling namespace survives

    await ctx.plugin(sceneTools);
    expect(await toolNames(client)).toContain('scene_card');

    await client.close();
    await ctx.fiber.dispose();
  });

  it('a namespace plugin stays pending until its injected services exist', async () => {
    const ctx = new Context();
    const fiber = ctx.plugin(workspaceTools);
    // No broker/mcp mounted yet: the fiber must not activate (load order is
    // expressed through service requirements, not boot sequencing).
    await new Promise((r) => setTimeout(r, 20));
    const ACTIVE = 2; // FiberState.ACTIVE (const enum — value per cordis lib/fiber.d.ts)
    expect(fiber.state).not.toBe(ACTIVE);

    await Promise.all([
      ctx.plugin(Broker, simConfig()),
      ctx.plugin(Mcp, { name: 'openreality-test', version: '0.0.0' }),
    ]);
    await fiber;
    expect(fiber.state).toBe(ACTIVE);
    await ctx.fiber.dispose();
  });

  it('rejects a malformed composition loudly at mount', async () => {
    const ctx = new Context();
    const fiber = ctx.plugin(Broker, { baseUrl: '' } as unknown as Config);
    await expect(fiber.await()).rejects.toThrow();
  });
});
