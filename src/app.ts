/**
 * Application assembly — the cordis composition and nothing else: mount the
 * Broker and Mcp services, then the tool-namespace plugins. No feature logic
 * lives here; each namespace is an independently mountable plugin, and a
 * composition may load any subset (the lifecycle tests mount one namespace at
 * a time and dispose it again).
 */
import { Context } from 'cordis';
import { loadConfig, type Config } from './config.js';
import { Broker } from './services/broker.js';
import { Mcp } from './services/mcp.js';
import * as workspaceTools from './plugins/workspace.js';
import * as sceneTools from './plugins/scene.js';
import * as agentTools from './plugins/agent.js';
import * as exportTools from './plugins/export.js';
import * as sceneResources from './plugins/resources.js';
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
 * Compose the full openreality MCP application on a fresh cordis Context.
 * @param config - resolved runtime config (defaults to the environment via `loadConfig`).
 * @returns the root context, with every plugin fiber active.
 */
export async function createApp(config: Config = loadConfig()): Promise<Context> {
  const ctx = new Context();
  const fibers = [
    ctx.plugin(Broker, config),
    ctx.plugin(Mcp, {
      name: 'openreality',
      version: packageJson.version,
      instructions: SERVER_INSTRUCTIONS,
    }),
    ...featurePlugins.map((plugin) => ctx.plugin(plugin)),
  ];
  await Promise.all(fibers);
  return ctx;
}
