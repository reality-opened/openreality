/**
 * workspace.* — account-level tools: the sync primitive (list scenes with a
 * local last-seen cursor), the three ingest doors (video / robot recording /
 * splat), and the uniform workspace job poll.
 *
 * Upload contract (routes_ingest.py / routes_recordings.py): raw octet-stream,
 * filename in X-Upload-Filename, optional X-Scene-Label / X-Demo-Source; big
 * splats go through init → chunk × N → finalize. Video cap 1 GB, recording 2 GB
 * (server-enforced; we pre-check to fail fast with a better message).
 */
import { basename } from 'node:path';
import { statSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  OREOS_REST_PATHS,
  REST_PATHS,
  type IngestSplatResponse,
  type IngestVideoResponse,
  type OreosJobStatus,
  type SceneListItem,
} from '@reality/protocol';
import { z } from 'zod';
import type { Context } from 'cordis';
import { guard, jsonResult, pollWorkspaceJob, progressTick } from '../toolkit.js';

const GiB = 1024 * 1024 * 1024;
const INLINE_SPLAT_CAP = 64 * 1024 * 1024; // routes_ingest.py inline cap
const SPLAT_CHUNK = 8 * 1024 * 1024;

interface SyncState {
  last_list_at?: number;
  known_scan_ids?: string[];
}

function readState(path: string): SyncState {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as SyncState;
  } catch {
    return {};
  }
}

function writeState(path: string, state: SyncState): void {
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify(state, null, 2) + '\n');
  } catch {
    // state is a convenience cursor; never fail the tool over it
  }
}

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'workspace-tools';

/** Services required before this plugin activates. */
export const inject = ['mcp', 'broker'];

export function apply(ctx: Context): void {
  const { mcp } = ctx;
  const { client, config } = ctx.broker;
  mcp.tool(ctx, 
    'workspace_list_scenes',
    {
      title: 'List scenes in the account',
      description:
        'List all persisted scenes in the signed-in Open Reality account (newest first). ' +
        'Also reports which scan_ids are NEW since this machine last listed — the account is ' +
        'the sync layer: scans made on the phone or the web appear here. Use scene_card next ' +
        'for anything you want context on.',
      inputSchema: {
        since_epoch_s: z
          .number()
          .optional()
          .describe('Only return scenes created at/after this epoch-seconds timestamp'),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ since_epoch_s }) =>
      guard(async () => {
        const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
        let scenes = Array.isArray(data.scenes) ? data.scenes : [];
        if (since_epoch_s != null) scenes = scenes.filter((s) => s.created_at >= since_epoch_s);
        const state = readState(config.statePath);
        const known = new Set(state.known_scan_ids ?? []);
        const newIds = scenes.map((s) => s.scan_id).filter((id) => !known.has(id));
        writeState(config.statePath, {
          last_list_at: Math.floor(Date.now() / 1000),
          known_scan_ids: [...new Set([...known, ...scenes.map((s) => s.scan_id)])],
        });
        return jsonResult({
          count: scenes.length,
          new_since_last_sync: newIds,
          scenes: scenes.map((s) => ({
            scan_id: s.scan_id,
            created_at: s.created_at,
            room_type: s.room_type,
            summary: s.summary,
            object_count: s.object_count,
            point_count: s.point_count,
            has_splat: s.has_splat ?? false,
            client: s.client ?? null,
            project: s.project ?? null,
          })),
        });
      }),
  );

  mcp.tool(ctx, 
    'workspace_upload_video',
    {
      title: 'Upload a video for reconstruction',
      description:
        'Upload a local video file (.mp4/.mov/.avi/.mkv/.webm, ≤1 GB) into the account and ' +
        'start a reconstruction job. Returns {job_id, scan_id} immediately (202); follow with ' +
        'workspace_job_wait. The scene appears in workspace_list_scenes when the job is done.',
      inputSchema: {
        path: z.string().describe('Absolute path of the local video file'),
        label: z.string().optional().describe('Human label for the scene (X-Scene-Label)'),
        source: z
          .enum(['gemini2', 'depthcam'])
          .optional()
          .describe('Set ONLY for depth-camera footage (X-Demo-Source provenance tag)'),
      },
    },
    async ({ path, label, source }) =>
      guard(async () => {
        const size = statSync(path).size;
        if (size > GiB) {
          throw new Error(`Video is ${size} bytes; the ingest cap is 1 GiB (server-enforced).`);
        }
        const headers: Record<string, string> = { 'X-Upload-Filename': basename(path) };
        if (label) headers['X-Scene-Label'] = label;
        if (source) headers['X-Demo-Source'] = source;
        const res = await client.uploadFile<IngestVideoResponse>(
          OREOS_REST_PATHS.INGEST_VIDEO,
          path,
          headers,
        );
        return jsonResult({ ...res, next: 'workspace_job_wait with this job_id' });
      }),
  );

  mcp.tool(ctx, 
    'workspace_upload_recording',
    {
      title: 'Upload a robot recording',
      description:
        'Upload a robot recording (DimOS memory2 .db, ≤2 GB) into the account. Runs the ' +
        'recordings pipeline (recon → odometry-consistency score → QC gate); the scene persists ' +
        'as source="robot_recording" with the QC report under derived/demo/recordings/. ' +
        'Returns {job_id, scan_id}; follow with workspace_job_wait.',
      inputSchema: {
        path: z.string().describe('Absolute path of the local .db recording'),
      },
    },
    async ({ path }) =>
      guard(async () => {
        const size = statSync(path).size;
        if (size > 2 * GiB) {
          throw new Error(`Recording is ${size} bytes; the ingest cap is 2 GiB (server-enforced).`);
        }
        const res = await client.uploadFile<IngestVideoResponse>(
          OREOS_REST_PATHS.INGEST_RECORDING,
          path,
          { 'X-Upload-Filename': basename(path) },
        );
        return jsonResult({ ...res, next: 'workspace_job_wait with this job_id' });
      }),
  );

  mcp.tool(ctx, 
    'workspace_upload_splat',
    {
      title: 'Import a Gaussian splat',
      description:
        'Import a local .ply/.spz Gaussian splat as a scene (source="imported_splat", honestly ' +
        'degraded report). ≤64 MiB uploads inline and returns {scan_id, gaussian_count} ' +
        'synchronously; larger files go through the chunked lane and return {job_id, scan_id} ' +
        'to poll with workspace_job_wait. Non-gaussian PLYs are refused (422 not_a_gaussian_splat).',
      inputSchema: {
        path: z.string().describe('Absolute path of the local .ply or .spz file'),
        label: z.string().optional().describe('Human label for the scene'),
      },
    },
    async ({ path, label }) =>
      guard(async () => {
        const size = statSync(path).size;
        const baseHeaders: Record<string, string> = { 'X-Upload-Filename': basename(path) };
        if (label) baseHeaders['X-Scene-Label'] = label;
        if (size <= INLINE_SPLAT_CAP) {
          const res = await client.uploadFile<IngestSplatResponse>(
            OREOS_REST_PATHS.INGEST_SPLAT,
            path,
            baseHeaders,
          );
          return jsonResult({ ...res, lane: 'inline' });
        }
        const init = await client.json<{ upload_id: string }>(
          'POST',
          OREOS_REST_PATHS.INGEST_SPLAT_INIT,
          { filename: basename(path), size, label },
        );
        const buf = readFileSync(path);
        let index = 0;
        for (let off = 0; off < buf.byteLength; off += SPLAT_CHUNK, index += 1) {
          await client.rawPost(
            OREOS_REST_PATHS.INGEST_SPLAT_CHUNK(init.upload_id, index),
            buf.subarray(off, Math.min(off + SPLAT_CHUNK, buf.byteLength)),
          );
        }
        const fin = await client.json<IngestVideoResponse>(
          'POST',
          OREOS_REST_PATHS.INGEST_SPLAT_FINALIZE(init.upload_id),
          { chunks: index },
        );
        return jsonResult({ ...fin, lane: 'chunked', next: 'workspace_job_wait with this job_id' });
      }),
  );

  mcp.tool(ctx, 
    'workspace_job_status',
    {
      title: 'Check a workspace job',
      description:
        'One status read of an ingest/export job (GET /api/workspace/jobs/<job_id>): ' +
        '{status: queued|running|done|error, stage?, scan_id?, error?}.',
      inputSchema: { job_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ job_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<OreosJobStatus>('GET', OREOS_REST_PATHS.WORKSPACE_JOB(job_id)),
        ),
      ),
  );

  mcp.tool(ctx, 
    'workspace_job_wait',
    {
      title: 'Wait for a workspace job',
      description:
        'Poll an ingest/export job until it is done or errored (or the timeout passes). ' +
        'Returns the final status plus the stage trail observed.',
      inputSchema: {
        job_id: z.string(),
        timeout_s: z.number().min(1).max(3600).default(900),
        poll_s: z.number().min(0.05).max(60).default(2),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ job_id, timeout_s, poll_s }, extra) =>
      guard(async () =>
        jsonResult(await pollWorkspaceJob(client, job_id, timeout_s, poll_s, progressTick(extra))),
      ),
  );
}
