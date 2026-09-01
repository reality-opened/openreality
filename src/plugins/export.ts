/**
 * export.* + artifact fetch — artifacts out of the account, onto local disk.
 * The to-disk rule: payload bytes NEVER enter context; fetch
 * tools write files and return paths.
 */
import { createWriteStream, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, normalize } from 'node:path';
import {
  OREOS_REST_PATHS,
  REST_PATHS,
  type ExportManifestResponse,
  type OreosPreparedExportResponse,
} from '@reality/protocol';
import { z } from 'zod';
import type { Config } from '../config.js';
import type { Context } from 'cordis';
import { guard, jsonResult, pollWorkspaceJob } from '../toolkit.js';

const exportFormat = z.enum(['openreality', 'groot_lerobot_v2', 'isaac_usd']);

function resolveOut(config: Config, scanId: string, key: string, outPath?: string): string {
  if (outPath) return outPath;
  // derived keys are server-sanitized, but never trust path joins blindly
  const safe = normalize(key).replace(/^(\.\.[/\\])+/, '').replace(/^[/\\]+/, '');
  return join(config.artifactsDir, scanId, safe);
}

function writeOut(path: string, bytes: Buffer): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
}

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'export-tools';

/** Services required before this plugin activates. */
export const inject = ['mcp', 'broker'];

export function apply(ctx: Context): void {
  const { mcp } = ctx;
  const { client, config } = ctx.broker;
  mcp.tool(ctx, 
    'export_manifest',
    {
      title: 'Export dry-run manifest',
      description:
        'What an export WOULD contain (GET /demo/export/manifest): file tree + sizes, ' +
        'complete/absent components, and whether the zip route would work. No build, no writes. ' +
        'Formats: openreality | groot_lerobot_v2 | isaac_usd (isaac is metric-gated — needs an anchor).',
      inputSchema: {
        scan_id: z.string(),
        format: exportFormat,
        source: z.string().optional().describe("'original' (default) or a derived/... key"),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, format, source }) =>
      guard(async () =>
        jsonResult(
          await client.json<ExportManifestResponse>(
            'GET',
            OREOS_REST_PATHS.SCENE_EXPORT_MANIFEST(scan_id, format, source),
          ),
        ),
      ),
  );

  mcp.tool(ctx, 
    'export_prepare',
    {
      title: 'Build an export',
      description:
        'Build the export zip as a background job (POST /demo/export/prepare → 202 {job_id}; ' +
        '409 export_job_active when one is running). With wait=true (default) polls the job and ' +
        'returns the prepared artifact (including download_path). Then artifact_fetch it.',
      inputSchema: {
        scan_id: z.string(),
        format: exportFormat,
        source: z.string().optional(),
        wait: z.boolean().default(true),
        timeout_s: z.number().min(1).max(3600).default(1200),
        poll_s: z.number().min(0.05).max(60).default(2),
      },
    },
    async ({ scan_id, format, source, wait, timeout_s, poll_s }) =>
      guard(async () => {
        const started = await client.json<Record<string, unknown>>(
          'POST',
          OREOS_REST_PATHS.SCENE_EXPORT_PREPARE(scan_id),
          { format, ...(source ? { source } : {}) },
        );
        const jobId = typeof started.job_id === 'string' ? started.job_id : null;
        if (!wait || !jobId) return jsonResult(started);
        const outcome = await pollWorkspaceJob(client, jobId, timeout_s, poll_s);
        const prepared = await client.json<OreosPreparedExportResponse>(
          'GET',
          OREOS_REST_PATHS.SCENE_EXPORT_PREPARED(scan_id, format, source),
        );
        return jsonResult({ job: outcome, prepared });
      }),
  );

  mcp.tool(ctx, 
    'export_status',
    {
      title: 'Prepared-export status',
      description:
        'What is prepared for scene+format (GET /demo/export/prepared). Always 200 with ' +
        '{status: ready|running|error|none}; ready carries the artifact + download_path.',
      inputSchema: {
        scan_id: z.string(),
        format: exportFormat,
        source: z.string().optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, format, source }) =>
      guard(async () =>
        jsonResult(
          await client.json<OreosPreparedExportResponse>(
            'GET',
            OREOS_REST_PATHS.SCENE_EXPORT_PREPARED(scan_id, format, source),
          ),
        ),
      ),
  );

  mcp.tool(ctx, 
    'artifact_fetch',
    {
      title: 'Fetch a derived artifact to disk',
      description:
        'Download any derived/... artifact (export zips, LOD levels, agent run logs, QC ' +
        'reports, masks) to a local file and return the path — bytes NEVER go into context. ' +
        'Default target: <artifacts_dir>/<scan_id>/<key>.',
      inputSchema: {
        scan_id: z.string(),
        derived_key: z.string().describe('A derived/... key (from manifests, pointers, job results)'),
        out_path: z.string().optional().describe('Absolute path override for the output file'),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, derived_key, out_path }) =>
      guard(async () => {
        if (out_path && !isAbsolute(out_path)) throw new Error('out_path must be absolute');
        const { bytes, contentType } = await client.bytes(
          REST_PATHS.SCENE_DERIVED(scan_id, derived_key),
        );
        const target = resolveOut(config, scan_id, derived_key, out_path);
        writeOut(target, bytes);
        return jsonResult({ path: target, bytes: bytes.byteLength, content_type: contentType });
      }),
  );

  mcp.tool(ctx, 
    'artifact_fetch_splat',
    {
      title: 'Fetch the scene splat to disk',
      description:
        'Download the scene\'s original splat.ply to a local file and return the path. For a ' +
        'render-sized variant prefer the LOD levels (scene_lod → artifact_fetch of a level key).',
      inputSchema: {
        scan_id: z.string(),
        out_path: z.string().optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, out_path }) =>
      guard(async () => {
        if (out_path && !isAbsolute(out_path)) throw new Error('out_path must be absolute');
        const { bytes, contentType } = await client.bytes(REST_PATHS.SCENE_SPLAT_PLY(scan_id));
        const target = out_path ?? join(config.artifactsDir, scan_id, 'splat.ply');
        writeOut(target, bytes);
        return jsonResult({ path: target, bytes: bytes.byteLength, content_type: contentType });
      }),
  );

  mcp.tool(ctx, 
    'artifact_fetch_cloud',
    {
      title: 'Fetch the scene point cloud to disk',
      description: 'Download the scene\'s cloud.ply to a local file and return the path.',
      inputSchema: {
        scan_id: z.string(),
        out_path: z.string().optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, out_path }) =>
      guard(async () => {
        if (out_path && !isAbsolute(out_path)) throw new Error('out_path must be absolute');
        const { bytes, contentType } = await client.bytes(REST_PATHS.SCENE_CLOUD_PLY(scan_id));
        const target = out_path ?? join(config.artifactsDir, scan_id, 'cloud.ply');
        writeOut(target, bytes);
        return jsonResult({ path: target, bytes: bytes.byteLength, content_type: contentType });
      }),
  );
}
