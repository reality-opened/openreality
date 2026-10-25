/**
 * Application assembly — the cordis composition and nothing else: mount the
 * Broker and Mcp services, then the tool-namespace plugins, then one hardware
 * bridge fiber per configured device (src/hardware.ts). No feature logic
 * lives here; each namespace is an independently mountable plugin, and a
 * composition may load any subset (the lifecycle tests mount one namespace at
 * a time and dispose it again).
 */
import { Context } from 'cordis';
import { loadConfig, type Config } from './config.js';
import type { HardwareDevice } from './hardware.js';
import { Broker } from './services/broker.js';
import { Mcp } from './services/mcp.js';
import * as workspaceTools from './plugins/workspace.js';
import * as sceneTools from './plugins/scene.js';
import * as agentTools from './plugins/agent.js';
import * as exportTools from './plugins/export.js';
import * as sceneResources from './plugins/resources.js';
import * as hardwareTools from './plugins/hardware.js';
import packageJson from '../package.json';

/** Server-level instructions surfaced to MCP clients (workflow + honesty doctrine). */
export const SERVER_INSTRUCTIONS = [
  'Open Reality OS tools. Workflow: workspace_list_scenes (sync) → scene_card (context) →',
  'feature tools (measure/nav/planes/exports are deterministic routes; agent_* runs the',
  "server-side scene agent and spends its bounded LLM budget). Uploads: workspace_upload_*",
  'then workspace_job_wait. Artifacts go to DISK via artifact_fetch — never into context.',
  'HONESTY DOCTRINE (non-negotiable): lengths are metres ONLY when a response says',
  'units:"m" (metric anchor applied) — otherwise say "relative units". Always carry',
  'scale_source / provenance / degraded flags along with any number you report. Objects',
  'are a closed world (scene_list_objects); synthetic views are renders, not photos.',
].join(' ');

/** The full tool surface, as loaded by `createApp` — exported so alternate
 *  compositions (tests, embedders) can mount subsets or supersets. */
export const featurePlugins = [
  workspaceTools,
  sceneTools,
  agentTools,
  exportTools,
  sceneResources,
] as const;

/**
 * Instructions for a composition: the workflow above, plus, when hardware is
 * bridged, which tool prefix drives which device and where its own rules are.
 */
export function serverInstructions(devices: readonly HardwareDevice[]): string {
  if (devices.length === 0) return SERVER_INSTRUCTIONS;
  const prefixes = devices.map(({ namespace }) => `${namespace}_* (read ${namespace}_guide first)`).join(', ');
  return [
    SERVER_INSTRUCTIONS,
    `HARDWARE: real devices on this machine, each bridged from its own MCP server: ${prefixes}.`,
    'A device checks its own limits: report a refusal and its reason, never work around it. A person may be next to it.',
  ].join(' ');
}

/**
 * Compose the full openreality MCP application on a fresh cordis Context.
 * @param config - resolved runtime config (defaults to the environment via `loadConfig`).
 * @param devices - hardware devices to bridge (`loadHardware`); none by default.
 * @returns the root context, with every plugin fiber active.
 */
export async function createApp(
  config: Config = loadConfig(),
  devices: readonly HardwareDevice[] = [],
): Promise<Context> {
  const ctx = new Context();
  const fibers = [
    ctx.plugin(Broker, config),
    ctx.plugin(Mcp, {
      name: 'openreality',
      version: packageJson.version,
      instructions: serverInstructions(devices),
    }),
    ...featurePlugins.map((plugin) => ctx.plugin(plugin)),
  ];
  await Promise.all(fibers);
  // A device fiber starts its device server asynchronously. Mounted once `mcp` exists, awaiting
  // it waits until the device tools are registered (or rejects when the device did not start);
  // awaited while still pending on `mcp`, it would resolve before any of that.
  await Promise.all(devices.map((device) => ctx.plugin(hardwareTools, device)));
  return ctx;
}
