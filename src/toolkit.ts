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
  final: OreosJobStatus;
  /** Stage transitions observed, e.g. ["queued", "running:recon", "done"]. */
  trail: string[];
  timed_out: boolean;
}

async function pollJobPath(
  client: ApiClient,
  pathFor: () => string,
  timeoutS: number,
  pollS: number,
): Promise<PollOutcome> {
  const deadline = Date.now() + timeoutS * 1000;
  const trail: string[] = [];
  let last: OreosJobStatus | null = null;
  for (;;) {
    const status = await client.json<OreosJobStatus>('GET', pathFor());
    const state = normalizeOreosJobState(status.status);
    const label = status.stage ? `${state}:${status.stage}` : state;
    if (trail[trail.length - 1] !== label) trail.push(label);
    last = status;
    if (state === 'done' || state === 'error') return { final: status, trail, timed_out: false };
    if (Date.now() >= deadline) return { final: status, trail, timed_out: true };
    await sleep(pollS * 1000);
  }
}

export function pollWorkspaceJob(
  client: ApiClient,
  jobId: string,
  timeoutS = 900,
  pollS = 2,
): Promise<PollOutcome> {
  return pollJobPath(client, () => OREOS_REST_PATHS.WORKSPACE_JOB(jobId), timeoutS, pollS);
}

export function pollSceneJob(
  client: ApiClient,
  scanId: string,
  jobId: string,
  timeoutS = 600,
  pollS = 2,
): Promise<PollOutcome> {
  return pollJobPath(client, () => OREOS_REST_PATHS.SCENE_JOB(scanId, jobId), timeoutS, pollS);
}
