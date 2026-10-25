/**
 * Hardware devices: machines (a robot arm, a drone) that ship their own MCP
 * server. Each device listed here is bridged by src/plugins/hardware.ts.
 *
 * Config file: `<OPENREALITY_DIR>/hardware.json` by default; OPENREALITY_HARDWARE
 * names another file, which must then exist. No default file = no hardware.
 *
 *   { "devices": { "arm": {
 *       "command": "C:/src/artist-agent/.venv/Scripts/python.exe",
 *       "args": ["server.py"], "cwd": "C:/src/artist-agent",
 *       "env": {}, "timeout_s": 900 } } }
 *
 * The key is the device namespace: its tools appear as `<namespace>_<tool>`.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';

const BUILT_IN_NAMESPACES: readonly string[] = ['workspace', 'scene', 'agent', 'export', 'artifact', 'openreality'];

const namespaceSchema = z
  .string()
  .regex(/^[a-z][a-z0-9]{0,15}$/, 'must be 1-16 lowercase letters or digits, starting with a letter')
  .refine((namespace) => !BUILT_IN_NAMESPACES.includes(namespace), (namespace) => ({
    message: `"${namespace}" is the prefix of built-in tools`,
  }));

const deviceFields = {
  command: z.string().min(1),
  args: z.array(z.string()).default([]),
  cwd: z.string().min(1).optional(),
  env: z.record(z.string()).default({}),
  timeout_s: z.number().positive().max(86_400).default(900),
};

/** One bridged device; the plugin config of src/plugins/hardware.ts. */
export const hardwareDeviceSchema = z.object({ namespace: namespaceSchema, ...deviceFields }).strict();

export type HardwareDevice = z.output<typeof hardwareDeviceSchema>;

const hardwareFileSchema = z
  .object({ devices: z.record(namespaceSchema, z.object(deviceFields).strict()) })
  .strict();

/** hardware.json is missing (when named explicitly), not JSON, or not a valid device list. */
export class HardwareConfigError extends Error {
  readonly path: string;

  constructor(path: string, problem: string) {
    super(`hardware config ${path}: ${problem}`);
    this.name = 'HardwareConfigError';
    this.path = path;
  }
}

/** A configured device could not be bridged: its server did not start, or its tool names clash. */
export class HardwareBridgeError extends Error {
  readonly namespace: string;

  constructor(namespace: string, problem: string, options?: ErrorOptions) {
    super(`hardware device "${namespace}": ${problem}`, options);
    this.name = 'HardwareBridgeError';
    this.namespace = namespace;
  }
}

function readConfig(path: string, named: boolean): string | null {
  try {
    return readFileSync(path, 'utf8');
  } catch (err) {
    if (err instanceof Error && 'code' in err && err.code === 'ENOENT') {
      if (named) throw new HardwareConfigError(path, 'file not found (OPENREALITY_HARDWARE names it)');
      return null;
    }
    throw err;
  }
}

/**
 * The devices to bridge, parsed from hardware.json.
 * @param configDir - the openreality state dir (`Config.configDir`).
 * @throws HardwareConfigError naming the file when it is not a valid device list.
 */
export function loadHardware(configDir: string, env: NodeJS.ProcessEnv = process.env): HardwareDevice[] {
  const named = env.OPENREALITY_HARDWARE;
  const path = named || join(configDir, 'hardware.json');
  const text = readConfig(path, Boolean(named));
  if (text === null) return [];
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (err) {
    if (err instanceof SyntaxError) throw new HardwareConfigError(path, `not valid JSON (${err.message})`);
    throw err;
  }
  const parsed = hardwareFileSchema.safeParse(json);
  if (!parsed.success) {
    const problems = parsed.error.issues.map((issue) => `${issue.path.join('.') || '(top level)'}: ${issue.message}`);
    throw new HardwareConfigError(path, problems.join('; '));
  }
  return Object.entries(parsed.data.devices).map(([namespace, device]) => ({ namespace, ...device }));
}
