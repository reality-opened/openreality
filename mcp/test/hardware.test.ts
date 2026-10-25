/**
 * Hardware bridge acceptance (src/hardware.ts, src/plugins/hardware.ts): a
 * device's own MCP server, started as a child process, shows up as
 * `<namespace>_*` tools whose calls and results pass through unedited; a stop
 * reaches the device while a motion call still runs; disposing the fiber ends
 * the device process. The device is test/fixtures/fake-device.mjs, serving the
 * real arm server's captured tool list and instructions.
 */
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import { Context } from 'cordis';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { CallToolResultSchema, ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import { createApp } from '../src/app.js';
import type { Config } from '../src/config.js';
import { Mcp } from '../src/services/mcp.js';
import { HardwareBridgeError, HardwareConfigError, loadHardware, type HardwareDevice } from '../src/hardware.js';
import * as hardwareTools from '../src/plugins/hardware.js';

interface Captured {
  readonly instructions: string;
  readonly tools: readonly { readonly name: string }[];
  readonly replies: { readonly release_torque_unconfirmed: string; readonly look_png_base64: string };
}

const fakeDevice = fileURLToPath(new URL('./fixtures/fake-device.mjs', import.meta.url));
const captured: Captured = JSON.parse(readFileSync(new URL('./fixtures/artist-arm-tools.json', import.meta.url), 'utf8'));

const tempDir = (): string => mkdtempSync(join(tmpdir(), 'orl-hardware-'));

function armDevice(overrides: Partial<HardwareDevice> = {}): HardwareDevice {
  return { namespace: 'arm', command: process.execPath, args: [fakeDevice], env: {}, timeout_s: 30, ...overrides };
}

const contexts: Context[] = [];

afterEach(async () => {
  await Promise.all(contexts.splice(0).map((ctx) => ctx.fiber.dispose()));
});

async function mcpContext(): Promise<Context> {
  const ctx = new Context();
  contexts.push(ctx);
  await ctx.plugin(Mcp, { name: 'openreality-test', version: '0.0.0' });
  return ctx;
}

async function connectedClient(ctx: Context): Promise<Client> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: 'hardware-test', version: '0.0.0' });
  await Promise.all([ctx.mcp.connect(serverTransport), client.connect(clientTransport)]);
  return client;
}

async function bridge(device: HardwareDevice) {
  const ctx = await mcpContext();
  const fiber = await ctx.plugin(hardwareTools, device);
  return { client: await connectedClient(ctx), fiber };
}

function appConfig(): Config {
  const dir = tempDir();
  return {
    baseUrl: 'http://127.0.0.1:9',
    loginBaseUrl: 'http://127.0.0.1:9',
    envToken: 'hardware-test',
    configDir: dir,
    credentialsPath: join(dir, 'credentials.json'),
    statePath: join(dir, 'state.json'),
    artifactsDir: join(dir, 'artifacts'),
  };
}

async function call(client: Client, name: string, args: Record<string, unknown> = {}) {
  return CallToolResultSchema.parse(await client.callTool({ name, arguments: args }));
}

function textOf(result: { readonly content: readonly { readonly type: string; readonly text?: string }[] }): string {
  return result.content.map((part) => part.text ?? '').join('\n');
}

function configError(dir: string, env: NodeJS.ProcessEnv = {}): HardwareConfigError | undefined {
  try {
    loadHardware(dir, env);
  } catch (err) {
    if (err instanceof HardwareConfigError) return err;
    throw err;
  }
  return undefined;
}

function isRunning(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    if (err instanceof Error && 'code' in err && err.code === 'ESRCH') return false;
    throw err;
  }
}

describe('hardware.json', () => {
  it('no file in the config dir means no devices', () => {
    expect(loadHardware(tempDir(), {})).toEqual([]);
  });

  it('a device takes its namespace from its key and defaults for what it leaves out', () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'hardware.json'), JSON.stringify({ devices: { arm: { command: 'python', args: ['server.py'] } } }));
    expect(loadHardware(dir, {})).toEqual([{ namespace: 'arm', command: 'python', args: ['server.py'], env: {}, timeout_s: 900 }]);
  });

  it.each([
    ['text that is not JSON', '{ "devices": '],
    ['a device without a command', JSON.stringify({ devices: { arm: { args: [] } } })],
    ['a namespace with capitals', JSON.stringify({ devices: { Arm: { command: 'python' } } })],
    ['a namespace with an underscore', JSON.stringify({ devices: { my_arm: { command: 'python' } } })],
    ['a namespace the built-in tools use', JSON.stringify({ devices: { scene: { command: 'python' } } })],
  ])('refuses %s, naming the file', (_case, body) => {
    const dir = tempDir();
    const file = join(dir, 'hardware.json');
    writeFileSync(file, body);
    expect(configError(dir)?.path).toBe(file);
  });

  it('OPENREALITY_HARDWARE naming a missing file is an error, not an empty device list', () => {
    const file = join(tempDir(), 'elsewhere.json');
    expect(configError(tempDir(), { OPENREALITY_HARDWARE: file })?.path).toBe(file);
  });
});

describe('hardware bridge', () => {
  it('offers every device tool under its namespace, plus the guide', async () => {
    const { client } = await bridge(armDevice());
    const names = (await client.listTools()).tools.map((tool) => tool.name).sort();
    const expected = captured.tools.map(({ name }) => (name.startsWith('arm_') ? name : `arm_${name}`));
    expect(names).toEqual([...expected, 'arm_guide'].sort());
  });

  it('the guide carries the device instructions verbatim', async () => {
    const { client } = await bridge(armDevice());
    expect(textOf(await call(client, 'arm_guide'))).toContain(captured.instructions);
  });

  it('forwards the arguments, with the defaults the device declared, and returns its result unedited', async () => {
    const { client } = await bridge(armDevice());
    const moved = await call(client, 'arm_move_to', { x_mm: 250, y_mm: -40, z_mm: 120 });
    expect(moved.structuredContent).toEqual({
      status: 'reached',
      tool: 'move_to',
      received: { x_mm: 250, y_mm: -40, z_mm: 120, jaws_down: false, speed_deg_s: 20, tol_mm: 10 },
    });
  });

  it('a device refusal reaches the caller verbatim', async () => {
    const { client } = await bridge(armDevice());
    const refused = await call(client, 'arm_release_torque', { which: 'all' });
    expect(refused.isError).toBe(true);
    expect(refused.content).toEqual([{ type: 'text', text: captured.replies.release_torque_unconfirmed }]);
  });

  it('a camera frame comes back as image content', async () => {
    const { client } = await bridge(armDevice());
    const look = await call(client, 'arm_look');
    expect(look.content).toEqual([{ type: 'image', mimeType: 'image/png', data: captured.replies.look_png_base64 }]);
  });

  it('a stop reaches the device while a motion call is still running on it', async () => {
    // Given: a motion the device reported as running (its progress is relayed to the caller)
    const { client } = await bridge(armDevice());
    let reportMoving: (() => void) | undefined;
    const moving = new Promise<void>((resolve) => {
      reportMoving = resolve;
    });
    const move = client.callTool({ name: 'arm_move_joints', arguments: { targets: { elbow_flex: -60 } } }, undefined, {
      onprogress: () => reportMoving?.(),
    });
    await moving;

    // When: the caller stops it
    const stop = await call(client, 'arm_stop');

    // Then: the stop found the motion, and the motion call ends halted
    expect(stop.structuredContent).toEqual({ stopped: 'true' });
    expect(CallToolResultSchema.parse(await move).structuredContent).toMatchObject({ status: 'halted' });
  });

  it('disposing the fiber removes the device tools and ends the device process', async () => {
    const pidFile = join(tempDir(), 'device.pid');
    const { client, fiber } = await bridge(armDevice({ env: { FAKE_DEVICE_PIDFILE: pidFile } }));
    const pid = Number(readFileSync(pidFile, 'utf8'));

    await fiber.dispose();

    expect((await client.listTools()).tools.filter((tool) => tool.name.startsWith('arm_'))).toEqual([]);
    expect(isRunning(pid)).toBe(false);
  });

  it('answers with a tool error naming the device when its server has died', async () => {
    const pidFile = join(tempDir(), 'device.pid');
    const { client } = await bridge(armDevice({ env: { FAKE_DEVICE_PIDFILE: pidFile } }));
    process.kill(Number(readFileSync(pidFile, 'utf8')));

    const status = await call(client, 'arm_status');

    expect(status.isError).toBe(true);
    expect(JSON.parse(textOf(status))).toMatchObject({
      device: 'arm',
      tool: 'arm_status',
      error: { code: ErrorCode.ConnectionClosed },
    });
  });

  it('a device whose server does not start fails the mount, naming the device', async () => {
    const ctx = await mcpContext();
    const fiber = ctx.plugin(hardwareTools, armDevice({ command: join(tempDir(), 'no-such-device-server') }));
    await expect(fiber.await()).rejects.toBeInstanceOf(HardwareBridgeError);
  });
});

describe('createApp with hardware', () => {
  it('resolves only once the device tools are registered, next to the scene tools', async () => {
    const ctx = await createApp(appConfig(), [armDevice()]);
    contexts.push(ctx);
    const names = (await (await connectedClient(ctx)).listTools()).tools.map((tool) => tool.name);
    expect(names).toEqual(expect.arrayContaining(['scene_card', 'arm_guide', 'arm_status']));
  });

  it('rejects, naming the device, when a configured device does not start', async () => {
    const started = createApp(appConfig(), [armDevice({ command: join(tempDir(), 'no-such-device-server') })]);
    await expect(started).rejects.toBeInstanceOf(HardwareBridgeError);
    await expect(started).rejects.toMatchObject({ namespace: 'arm' });
  });
});
