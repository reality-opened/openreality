/**
 * export.* + artifact fetch — artifacts out of the account, onto local disk.
 * The to-disk rule: payload bytes NEVER enter context; fetch
 * tools write files and return paths.
 */
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
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
import { ApiError } from '../http.js';
import { guard, jsonResult, pollWorkspaceJob, progressTick, type ToolResult } from '../toolkit.js';
import { extractZip } from '../unzip.js';

const exportFormat = z.enum(['openreality', 'groot_lerobot_v2', 'isaac_usd', 'mujoco_mjcf']);

const MUJOCO = 'mujoco_mjcf' as const;

function errorResult(value: unknown): ToolResult {
  return { ...jsonResult(value), isError: true };
}

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
        'Formats: openreality | groot_lerobot_v2 | isaac_usd | mujoco_mjcf (the simulator formats ' +
        'isaac_usd / mujoco_mjcf are metric-gated — they need an anchor).',
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
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Build the export zip as a background job (POST /demo/export/prepare → 202 {job_id}; ' +
        '409 export_job_active when one is running). With wait=true (default) polls the job and ' +
        'returns the prepared artifact (including download_path). Then artifact_fetch it. ' +
        'For MuJoCo prefer export_mujoco, which also downloads and unpacks the scene.',
      inputSchema: {
        scan_id: z.string(),
        format: exportFormat,
        source: z.string().optional(),
        wait: z.boolean().default(true),
        timeout_s: z.number().min(1).max(3600).default(1200),
        poll_s: z.number().min(0.05).max(60).default(2),
      },
    },
    async ({ scan_id, format, source, wait, timeout_s, poll_s }, extra) =>
      guard(async () => {
        const started = await client.json<Record<string, unknown>>(
          'POST',
          OREOS_REST_PATHS.SCENE_EXPORT_PREPARE(scan_id),
          { format, ...(source ? { source } : {}) },
        );
        const jobId = typeof started.job_id === 'string' ? started.job_id : null;
        if (!wait || !jobId) return jsonResult(started);
        const outcome = await pollWorkspaceJob(client, jobId, timeout_s, poll_s, progressTick(extra));
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
    'export_mujoco',
    {
      title: 'Export the scene to MuJoCo',
      description:
        'One call from scene to a loadable MuJoCo model: builds (or reuses a fresh) mujoco_mjcf ' +
        'export, downloads it and unpacks it to disk, and returns the path to scene.xml — load ' +
        'it with mujoco.MjModel.from_xml_path(path) or `python -m mujoco.viewer --mjcf=<path>` ' +
        '(needs mujoco>=3.3). METRIC-GATED: an un-anchored scene is refused with 409 ' +
        'metric_scale_required — run scene_anchor first. Collision is primitives only (a floor ' +
        'plane + one upright box per detected object); the scan mesh is visual-only. A fresh ' +
        'build meshes the scene on a 64 GB container and takes minutes. Bytes never enter context.',
      inputSchema: {
        scan_id: z.string(),
        source: z.string().optional().describe("'original' (default) or a derived/... key"),
        out_dir: z
          .string()
          .optional()
          .describe('Absolute directory to unpack into (default <artifacts_dir>); the scene lands in <out_dir>/<scan_id>/mujoco/'),
        force: z.boolean().default(false).describe('Rebuild even when a fresh export is already prepared'),
        timeout_s: z.number().min(1).max(3600).default(1800),
        poll_s: z.number().min(0.05).max(60).default(2),
      },
      annotations: { readOnlyHint: false, destructiveHint: false },
    },
    async ({ scan_id, source, out_dir, force, timeout_s, poll_s }, extra) =>
      guard(async () => {
        if (out_dir && !isAbsolute(out_dir)) throw new Error('out_dir must be absolute');

        let started: Record<string, unknown>;
        try {
          started = await client.json<Record<string, unknown>>(
            'POST',
            OREOS_REST_PATHS.SCENE_EXPORT_PREPARE(scan_id),
            { format: MUJOCO, ...(source ? { source } : {}), ...(force ? { force: true } : {}) },
          );
        } catch (err) {
          const code = err instanceof ApiError ? (err.body as { error?: unknown } | null)?.error : null;
          if (err instanceof ApiError && code === 'metric_scale_required') {
            return errorResult({
              http_status: err.status,
              body: err.body,
              next_step:
                'Anchor the scene first: scene_anchor with two points and their real distance ' +
                '(measure a known length — a door, a table edge), then call export_mujoco again.',
            });
          }
          throw err;
        }

        const jobId = typeof started.job_id === 'string' ? started.job_id : null;
        const job = jobId
          ? await pollWorkspaceJob(client, jobId, timeout_s, poll_s, progressTick(extra))
          : null;
        const prepared = await client.json<OreosPreparedExportResponse>(
          'GET',
          OREOS_REST_PATHS.SCENE_EXPORT_PREPARED(scan_id, MUJOCO, source),
        );
        const artifact = prepared.status === 'ready' ? prepared.artifact : null;
        if (!artifact?.key) {
          return errorResult({
            error: job?.timed_out ? 'export_timed_out' : 'export_not_ready',
            message: job?.timed_out
              ? `the MuJoCo build is still running after ${timeout_s}s; check export_status later or call export_mujoco again`
              : 'the MuJoCo export did not produce an artifact',
            job,
            prepared,
          });
        }

        const { bytes } = await client.bytes(REST_PATHS.SCENE_DERIVED(scan_id, artifact.key));
        const zipPath = resolveOut(config, scan_id, artifact.key);
        writeOut(zipPath, bytes);
        const destRoot = out_dir ?? config.artifactsDir;
        const written = extractZip(bytes, destRoot);
        const mujocoDir = join(destRoot, scan_id, 'mujoco');
        const sceneXml = join(mujocoDir, 'scene.xml');
        const manifestPath = join(mujocoDir, 'manifest.json');
        if (!existsSync(sceneXml) || !existsSync(manifestPath)) {
          return errorResult({
            error: 'unexpected_archive',
            message: `the archive has no ${scan_id}/mujoco/scene.xml + manifest.json`,
            zip_path: zipPath,
            files: written,
          });
        }
        const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
        const objects = Array.isArray(manifest.objects)
          ? (manifest.objects as Array<Record<string, unknown>>).map((o) => ({
              geom: o.geom,
              label: o.label,
              fit: o.fit,
            }))
          : [];

        return jsonResult({
          scene_xml: sceneXml,
          mujoco_dir: mujocoDir,
          zip_path: zipPath,
          files: written.length,
          load: {
            python: `mujoco.MjModel.from_xml_path(${JSON.stringify(sceneXml)})`,
            viewer: `python -m mujoco.viewer --mjcf=${sceneXml}`,
            replay_scan_camera: `python ${join(mujocoDir, 'replay.py')}  (macOS: mjpython)`,
            requires: 'mujoco>=3.3',
          },
          // The tree's own manifest, passed through: alignment carries metric / scale /
          // scale_source / gravity_aligned — the honesty fields.
          alignment: manifest.alignment,
          geometry: manifest.geometry,
          camera: manifest.camera,
          objects,
          collision_note: manifest.collision_note,
          notes: manifest.notes,
          validated_with_mujoco: manifest.validated_with_mujoco,
          artifact: {
            key: artifact.key,
            bytes: artifact.bytes,
            source_key: artifact.source_key,
            built_at: artifact.built_at,
            stale: prepared.stale ?? false,
            decimated: artifact.decimated ?? false,
            original_points: artifact.original_points ?? null,
            exported_points: artifact.exported_points ?? null,
          },
          job: job ?? { reused: true, message: 'a fresh export was already prepared (pass force to rebuild)' },
        });
      }),
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
