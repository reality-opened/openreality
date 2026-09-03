/**
 * agent.* — drive the server-side scene agent (the OS's own orchestrator).
 * The loop, tools, guardrails, and budget live on the server; these tools speak
 * the run/event protocol the OS web client speaks (202 {run_id} + 1 Hz cursor
 * poll of the persisted event log). Replays are $0 re-serves.
 */
import {
  OREOS_REST_PATHS,
  isAgentRunEventsResponse,
  type AgentRunEventsResponse,
  type AgentRunStartResponse,
  type AgentRunsListResponse,
  type OreosAgentEvent,
} from '@reality/protocol';
import { z } from 'zod';
import type { ApiClient } from '../http.js';
import type { Context } from 'cordis';
import { guard, jsonResult, textResult } from '../toolkit.js';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function renderEvent(ev: OreosAgentEvent): string {
  const p = ev.payload as Record<string, unknown>;
  const phase = ev.phase ? ` [${ev.phase}]` : '';
  switch (ev.type) {
    case 'run_meta':
      return `— run meta${phase}: replay=${!!p.replay}${p.model ? ` model=${p.model}` : ''}`;
    case 'agent_thought': {
      const author = (p as { author?: string }).author === 'user' ? 'user' : 'agent';
      return `${author}${phase}: ${String(p.content ?? '')}`;
    }
    case 'agent_state':
      return `state${phase}: ${JSON.stringify(compact(p))}`;
    case 'agent_tool_event':
      return `tool${phase}: ${String(p.tool)} · ${String(p.status)}${p.latency_ms != null ? ` (${p.latency_ms} ms)` : ''}`;
    case 'agent_finding':
      return `finding${phase}: ${String(p.query ?? '')} — ${String(p.description ?? '')} (conf ${p.confidence ?? '?'})`;
    case 'agent_ui_command':
      return `ui${phase}: ${String(p.name ?? '')}`;
    case 'agent_job_event':
      return `job${phase}: ${String(p.job_name ?? p.job_id ?? '')} · ${String(p.status ?? '')}`;
    case 'run_done':
      return `— run done: ${JSON.stringify(compact(p))}`;
    default:
      return `${ev.type}${phase}: ${JSON.stringify(compact(p))}`;
  }
}

function compact(p: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(p)) {
    if (typeof v === 'string' && v.length > 200) out[k] = v.slice(0, 200) + '…';
    else out[k] = v;
  }
  return out;
}

async function drainRun(
  client: ApiClient,
  scanId: string,
  runId: string,
  after: number,
  timeoutS: number,
  pollS: number,
): Promise<{ events: OreosAgentEvent[]; status: string; next: number; error?: string | null; timed_out: boolean }> {
  const deadline = Date.now() + timeoutS * 1000;
  const events: OreosAgentEvent[] = [];
  let cursor = after;
  for (;;) {
    const res = await client.json<AgentRunEventsResponse>(
      'GET',
      OREOS_REST_PATHS.SCENE_AGENT_RUN_EVENTS(scanId, runId, cursor),
    );
    if (!isAgentRunEventsResponse(res)) {
      throw new Error(`Malformed run-events response: ${JSON.stringify(res).slice(0, 300)}`);
    }
    events.push(...res.events);
    cursor = res.next;
    if (res.status !== 'running') {
      return { events, status: res.status, next: cursor, error: res.error ?? null, timed_out: false };
    }
    if (Date.now() >= deadline) {
      return { events, status: res.status, next: cursor, error: res.error ?? null, timed_out: true };
    }
    await sleep(pollS * 1000);
  }
}

function transcriptResult(
  runId: string,
  drained: Awaited<ReturnType<typeof drainRun>>,
): ReturnType<typeof textResult> {
  const lines = drained.events.map(renderEvent);
  const head = `run ${runId} — status: ${drained.status}${drained.timed_out ? ' (poll timed out; poll again with agent_run_events)' : ''}${drained.error ? ` — error: ${drained.error}` : ''}`;
  return textResult([head, '', ...lines, '', `(next cursor: ${drained.next})`].join('\n'));
}

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'agent-tools';

/** Services required before this plugin activates. */
export const inject = ['mcp', 'broker'];

export function apply(ctx: Context): void {
  const { mcp } = ctx;
  const { client } = ctx.broker;
  mcp.tool(ctx, 
    'agent_annotate',
    {
      title: 'Run the annotation agent',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Start the five-phase scene annotation run (survey → labels → description → ' +
        'dimensions → key features) on the server-side scene agent. 202 {run_id}; one active ' +
        'run per scene (409 agent_run_active carries active_run_id — or set attach_if_active). ' +
        'Costs bounded server-side LLM calls. Follow with agent_run_events wait_for_done.',
      inputSchema: {
        scan_id: z.string(),
        attach_if_active: z.boolean().default(false),
      },
    },
    async ({ scan_id, attach_if_active }) =>
      guard(async () =>
        jsonResult(
          await client.json<AgentRunStartResponse>(
            'POST',
            OREOS_REST_PATHS.SCENE_AGENT_ANNOTATE(scan_id),
            { mode: 'full', ...(attach_if_active ? { attach_if_active } : {}) },
          ),
        ),
      ),
  );

  mcp.tool(ctx, 
    'agent_replay',
    {
      title: 'Replay a recorded run',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Re-serve a recorded run\'s persisted event log as a new (replay-badged, $0) run. ' +
        'Use for reviewing what an earlier run found without spending LLM calls.',
      inputSchema: {
        scan_id: z.string(),
        source_run_id: z.string(),
        speed: z.number().positive().optional(),
      },
    },
    async ({ scan_id, source_run_id, speed }) =>
      guard(async () =>
        jsonResult(
          await client.json<AgentRunStartResponse>(
            'POST',
            OREOS_REST_PATHS.SCENE_AGENT_ANNOTATE(scan_id),
            { replay_of: source_run_id, ...(speed ? { speed } : {}) },
          ),
        ),
      ),
  );

  mcp.tool(ctx, 
    'agent_pilot',
    {
      title: 'Run the pilot agent',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Give the pilot agent a free-text instruction (navigate/inspect flows; tools: ' +
        'list_scene_objects, plan_path, measure_distance). 202 {run_id}; poll agent_run_events.',
      inputSchema: {
        scan_id: z.string(),
        instruction: z.string(),
      },
    },
    async ({ scan_id, instruction }) =>
      guard(async () =>
        jsonResult(
          await client.json<AgentRunStartResponse>(
            'POST',
            OREOS_REST_PATHS.SCENE_AGENT_PILOT(scan_id),
            { instruction },
          ),
        ),
      ),
  );

  mcp.tool(ctx, 
    'agent_chat',
    {
      title: 'Chat with the scene agent',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'One chat turn with the scene agent (its own bounded run; the server agent may call ' +
        'its scene tools mid-turn). By default waits for the run to finish and returns the ' +
        'event transcript. The agent\'s numbers carry units/scale_source — repeat them honestly.',
      inputSchema: {
        scan_id: z.string(),
        message: z.string(),
        run_id: z.string().optional().describe('Attach to a prior run\'s context'),
        wait: z.boolean().default(true),
        timeout_s: z.number().min(1).max(1800).default(240),
        poll_s: z.number().min(0.05).max(30).default(1),
      },
    },
    async ({ scan_id, message, run_id, wait, timeout_s, poll_s }) =>
      guard(async () => {
        const start = await client.json<AgentRunStartResponse>(
          'POST',
          OREOS_REST_PATHS.SCENE_AGENT_CHAT(scan_id),
          { message, ...(run_id ? { run_id } : {}) },
        );
        if (!wait) return jsonResult(start);
        const drained = await drainRun(client, scan_id, start.run_id, 0, timeout_s, poll_s);
        return transcriptResult(start.run_id, drained);
      }),
  );

  mcp.tool(ctx, 
    'agent_run_events',
    {
      title: 'Read run events',
      description:
        'Read a run\'s event log from a cursor (GET /runs/<run_id>/events?after=). With ' +
        'wait_for_done, polls at ~1 Hz (the web client\'s cadence) until the run finishes.',
      inputSchema: {
        scan_id: z.string(),
        run_id: z.string(),
        after: z.number().int().min(0).default(0),
        wait_for_done: z.boolean().default(false),
        timeout_s: z.number().min(1).max(1800).default(240),
        poll_s: z.number().min(0.05).max(30).default(1),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, run_id, after, wait_for_done, timeout_s, poll_s }) =>
      guard(async () => {
        if (!wait_for_done) {
          return jsonResult(
            await client.json<AgentRunEventsResponse>(
              'GET',
              OREOS_REST_PATHS.SCENE_AGENT_RUN_EVENTS(scan_id, run_id, after),
            ),
          );
        }
        const drained = await drainRun(client, scan_id, run_id, after, timeout_s, poll_s);
        return transcriptResult(run_id, drained);
      }),
  );

  mcp.tool(ctx, 
    'agent_runs',
    {
      title: 'List agent runs',
      description:
        'The scene\'s run index (newest first) + which run is live: kind, status, cost_usd, ' +
        'llm_calls, findings. Persisted runs can be replayed for $0 with agent_replay.',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<AgentRunsListResponse>(
            'GET',
            OREOS_REST_PATHS.SCENE_AGENT_RUNS(scan_id),
          ),
        ),
      ),
  );
}
