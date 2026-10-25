/**
 * The BUILT CLI (`dist/cli.js serve`) with a hardware.json in its config dir:
 * device tools appear next to the scene tools and the device gets the env the
 * file gives it; a malformed file stops the server with an error naming the
 * file; closing stdin ends the session, the device server included.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import { CallToolResultSchema, LATEST_PROTOCOL_VERSION } from '@modelcontextprotocol/sdk/types.js';

const pkgDir = dirname(dirname(fileURLToPath(import.meta.url)));
const cliPath = join(pkgDir, 'dist', 'cli.js');
const fakeDevice = join(pkgDir, 'test', 'fixtures', 'fake-device.mjs');

function configDir(hardware: unknown): string {
  const dir = mkdtempSync(join(tmpdir(), 'orl-hardware-cli-'));
  writeFileSync(join(dir, 'hardware.json'), typeof hardware === 'string' ? hardware : JSON.stringify(hardware));
  return dir;
}

function serveEnv(dir: string): Record<string, string> {
  return {
    ...getDefaultEnvironment(),
    OPENREALITY_DIR: dir,
    OPENREALITY_TOKEN: 'hardware-cli-test',
    OPENREALITY_URL: 'http://127.0.0.1:9',
  };
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

let client: Client | undefined;

beforeAll(() => {
  expect(existsSync(cliPath), 'dist/cli.js must exist: npm test builds first').toBe(true);
});

afterEach(async () => {
  await client?.close();
  client = undefined;
});

describe('openreality-mcp serve with hardware.json', () => {
  it('serves the device tools next to the scene tools, with the env the file gives the device', async () => {
    const dir = configDir({
      devices: { arm: { command: process.execPath, args: [fakeDevice], env: { FAKE_DEVICE_TAG: 'bench-7' } } },
    });
    client = new Client({ name: 'hardware-cli-test', version: '0.0.0' });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [cliPath, 'serve'], env: serveEnv(dir) }));

    const names = (await client.listTools()).tools.map((tool) => tool.name);
    const status = CallToolResultSchema.parse(await client.callTool({ name: 'arm_status', arguments: {} }));

    expect(names).toEqual(expect.arrayContaining(['scene_card', 'workspace_list_scenes', 'arm_guide', 'arm_status', 'arm_move_to']));
    expect(status.structuredContent).toMatchObject({ tag: 'bench-7' });
  });

  it('refuses to start on a malformed hardware.json and names the file', async () => {
    const dir = configDir({ devices: { arm: { args: [] } } });
    const child = spawn(process.execPath, [cliPath, 'serve'], { env: serveEnv(dir), stdio: ['pipe', 'ignore', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });

    const code = await new Promise<number | null>((resolve) => child.once('close', resolve));

    expect(code).toBe(1);
    expect(stderr).toContain(join(dir, 'hardware.json'));
  });

  it('exits when the client closes stdin, stopping the device server', async () => {
    // Given: a running server whose device is up (the initialize answer comes after the device started)
    const pidFile = join(mkdtempSync(join(tmpdir(), 'orl-hardware-pid-')), 'device.pid');
    const dir = configDir({
      devices: { arm: { command: process.execPath, args: [fakeDevice], env: { FAKE_DEVICE_PIDFILE: pidFile } } },
    });
    const child = spawn(process.execPath, [cliPath, 'serve'], { env: serveEnv(dir), stdio: ['pipe', 'pipe', 'inherit'] });
    const closed = new Promise<number | null>((resolve) => child.once('close', resolve));
    const answered = new Promise<void>((resolve) => {
      let out = '';
      child.stdout.on('data', (chunk) => {
        out += String(chunk);
        if (out.includes('\n')) resolve();
      });
    });
    const initialize = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { protocolVersion: LATEST_PROTOCOL_VERSION, capabilities: {}, clientInfo: { name: 'cli-test', version: '0.0.0' } },
    };
    child.stdin.write(`${JSON.stringify(initialize)}\n`);
    await answered;
    const devicePid = Number(readFileSync(pidFile, 'utf8'));

    // When: the client ends the session by closing stdin
    child.stdin.end();

    // Then: the server exits on its own, and the device server is gone
    expect(await closed).toBe(0);
    expect(isRunning(devicePid)).toBe(false);
  });
});
