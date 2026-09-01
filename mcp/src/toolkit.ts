/**
 * Shared helpers for tool handlers: result shaping, uniform ApiError surfacing
 * (the server's typed error bodies pass through verbatim — they are contract),
 * and the workspace/scene job poll loops every async feature converges on.
 */
import {
  OREOS_REST_PATHS,
  normalizeOreosJobState,
  type OreosJobStatus,
} from '@reality/protocol';
import { z } from 'zod';
import type { ApiClient } from './http.js';
import { ApiError } from './http.js';

export const vec3 = z.tuple([z.number(), z.number(), z.number()]);

export interface ToolResult {
  [key: string]: unknown;
  content: Array<
    | { type: 'text'; text: string }
    | { type: 'image'; data: string; mimeType: string }
  >;
  isError?: boolean;
}

export function jsonResult(value: unknown): ToolResult {
  return { content: [{ type: 'text', text: JSON.stringify(value, null, 2) }] };
}

export function textResult(text: string): ToolResult {
  return { content: [{ type: 'text', text }] };
}

export function imageResult(bytes: Buffer, mimeType: string, caption?: string): ToolResult {
  const content: ToolResult['content'] = [
    { type: 'image', data: bytes.toString('base64'), mimeType },
  ];
  if (caption) content.push({ type: 'text', text: caption });
  return { content };
}

/** Wrap a handler so ApiErrors surface as structured tool errors, not protocol faults. */
export async function guard(fn: () => Promise<ToolResult>): Promise<ToolResult> {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof ApiError) {
      // A custom message (e.g. http.ts's revoked-API-key hint) is worth surfacing
      // alongside the raw status/body — the default `new Error()` message is just
      // `HTTP ${status}`, which the body already conveys, so only add it when it says
      // something more.
      const message = err.message && err.message !== `HTTP ${err.status}` ? { message: err.message } : {};
      return {
        isError: true,
        content: [
          {
            type: 'text',
            text: JSON.stringify({ http_status: err.status, body: err.body, ...message }, null, 2),
          },
        ],
      };
    }
    return {
      isError: true,
      content: [{ type: 'text', text: err instanceof Error ? err.message : String(err) }],
    };
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export interface PollOutcome {
  /** Null only when the wait timed out before a single successful status read. */
  final: OreosJobStatus | null;
  /** Stage transitions observed, e.g. ["queued", "running:recon", "done"]. */
  trail: string[];
  timed_out: boolean;
}

/** Called once per poll iteration so long waits can surface liveness. */
export type PollTick = (status: OreosJobStatus | null, elapsedS: number) => void | Promise<void>;

/**
 * The shape of the MCP request-handler `extra` this package needs for progress:
 * when the client attached a progressToken to the tool call, each poll tick is
 * reported as a `notifications/progress` — which resets the harness's
 * "no response or progress" idle timeout that otherwise kills waits longer than
 * its window (observed: a healthy 30-min workspace_job_wait aborted client-side).
 * Without a token this returns undefined and the wait behaves as before.
 */
export interface ProgressExtra {
  _meta?: { progressToken?: string | number };
  sendNotification?: (notification: {
    method: 'notifications/progress';
    params: { progressToken: string | number; progress: number; total?: number; message?: string };
  }) => Promise<void>;
}

export function progressTick(extra: ProgressExtra | undefined): PollTick | undefined {
  const token = extra?._meta?.progressToken;
  const send = extra?.sendNotification;
  if (token === undefined || !send) return undefined;
  return async (status, elapsedS) => {
    const stage = status ? (status.stage ?? status.status) : 'polling';
    try {
      await send({
        method: 'notifications/progress',
        params: { progressToken: token, progress: elapsedS, message: `${stage} (${Math.round(elapsedS)}s)` },
      });
    } catch {
      // Progress is best-effort; a notification failure must never kill the wait.
    }
  };
}

/** Transient poll failures tolerated before the wait gives up. One blip in a
 * 30-minute wait used to abort the whole tool call; genuine outages still
 * surface after this many consecutive failures. Semantic 4xx rejections
 * (unknown job, revoked auth) are answers, not blips, and throw immediately. */
const POLL_MAX_CONSECUTIVE_FAILURES = 5;

async function pollJobPath(
  client: ApiClient,
  pathFor: () => string,
  timeoutS: number,
  pollS: number,
  onTick?: PollTick,
): Promise<PollOutcome> {
  const started = Date.now();
  const deadline = started + timeoutS * 1000;
  const trail: string[] = [];
  let last: OreosJobStatus | null = null;
  let consecutiveFailures = 0;
  for (;;) {
    try {
      const status = await client.json<OreosJobStatus>('GET', pathFor());
      consecutiveFailures = 0;
      const state = normalizeOreosJobState(status.status);
      const label = status.stage ? `${state}:${status.stage}` : state;
      if (trail[trail.length - 1] !== label) trail.push(label);
      last = status;
      if (state === 'done' || state === 'error') return { final: status, trail, timed_out: false };
    } catch (err) {
      if (err instanceof ApiError && err.status < 500) throw err;
      consecutiveFailures += 1;
      if (consecutiveFailures >= POLL_MAX_CONSECUTIVE_FAILURES) throw err;
      const note = 'poll-retry';
      if (trail[trail.length - 1] !== note) trail.push(note);
    }
    if (Date.now() >= deadline) {
      return { final: last, trail, timed_out: true };
    }
    await onTick?.(last, (Date.now() - started) / 1000);
    await sleep(pollS * 1000);
  }
}

export function pollWorkspaceJob(
  client: ApiClient,
  jobId: string,
  timeoutS = 900,
  pollS = 2,
  onTick?: PollTick,
): Promise<PollOutcome> {
  return pollJobPath(client, () => OREOS_REST_PATHS.WORKSPACE_JOB(jobId), timeoutS, pollS, onTick);
}

export function pollSceneJob(
  client: ApiClient,
  scanId: string,
  jobId: string,
  timeoutS = 600,
  pollS = 2,
  onTick?: PollTick,
): Promise<PollOutcome> {
  return pollJobPath(client, () => OREOS_REST_PATHS.SCENE_JOB(scanId, jobId), timeoutS, pollS, onTick);
}
