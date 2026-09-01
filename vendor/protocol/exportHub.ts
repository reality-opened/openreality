/**
 * Export hub — the real, LIVE contract for the three robot-training export formats a
 * finished scan can produce (see `platform/contracts/export-format.md` and
 * `server/docs/dataset-export.md` §4/§5, `server/docs/isaac-export.md`):
 *
 *   1. `openreality`      — the OpenReality native tree (trajectory + video + cloud +
 *                            grounding sidecar). The format we own; source of truth.
 *   2. `groot_lerobot_v2` — a thin transcode of (1) into the GR00T-LeRobot v2 layout.
 *   3. `isaac_usd`        — an Isaac Sim/Lab USD scene (metric + gravity-aligned).
 *
 * `GET /api/scenes/<scan_id>/export?format=<format>` is now a real, deployed broker route
 * (`server/app.py::export_scene_route`) — GPU-free, no live Solver:
 *
 *   - `openreality` / `groot_lerobot_v2` -> 200, `application/zip`, streamed as an attachment
 *     named `scan-<scan_id>-<format>.zip`. There is no JSON success envelope — the response
 *     body IS the zip.
 *   - `isaac_usd` -> now a REAL, wired export (`server/app.py::_export_isaac_usd`): the same
 *     GPU-free broker route authors an Isaac Sim/Lab USD tree (`scene.usd` + `trajectory.usd` +
 *     `manifest.json`) from the persisted geometry and streams it zipped as
 *     `scan-<scan_id>-isaac_usd.zip`. It is GATED on a metric scale — it needs either an explicit
 *     `?scale=` OR a Metric anchor (Refine stage) whose calibrated (already-metric) geometry it
 *     exports. Without one -> 409 `{error:"metric_scale_required"}`. `EXPORT_FORMATS` marks it
 *     `serverAvailable: true` + `requiresMetricAnchor: true` so the UI enables it only once an
 *     anchor exists (and routes the operator to Refine otherwise). The heavy USD deps
 *     (`usd-core`/`open3d`) may not be enabled in a given deployment yet -> 501
 *     `{error:"isaac_unavailable"}` (the UI shows an honest "not enabled here" message).
 *   - Other error statuses: 400 `{error:"invalid_format"|"invalid_request", message}`, 404
 *     `{error:"not_found"}` (no such scan for this owner) / `{error:"no_trajectory", message}`
 *     (scan predates per-keyframe trajectory persistence) / `{error:"no_points"}` (no cloud to
 *     build from) / `{error:"unknown_derived_key"}` (bogus `?source=`), 409
 *     `{error:"metric_scale_required"}` (isaac, no scale/anchor), 501
 *     `{error:"isaac_unavailable"}` (isaac deps not installed), 500
 *     `{error:"export_failed", message}`.
 */

export type ExportFormat = 'openreality' | 'groot_lerobot_v2' | 'isaac_usd';

export interface ExportFormatDescriptor {
  format: ExportFormat;
  label: string;
  /** One-line, honest description of what the tree/file contains. */
  summary: string;
  /** Doc path (relative to the `server` repo root) with the full spec. */
  docs: string;
  /** Whether `GET /api/scenes/<id>/export?format=<format>` is a wired route that can produce this
   *  format. All three are `true` now (isaac_usd was un-501'd). A `true` here does NOT mean
   *  "clickable unconditionally" — `isaac_usd` also carries `requiresMetricAnchor` (a client-side
   *  precondition) and can still 501 at runtime if the deployment lacks the USD deps. */
  serverAvailable: boolean;
  /** `true` for a format that the server GATES on a metric scale (only `isaac_usd`): the UI must
   *  enable it only once a Metric anchor has been applied (`derived_latest.kind === 'anchor'`),
   *  and route the operator to Refine otherwise. Mirrors the server's 409 `metric_scale_required`
   *  gate so the button is never a guaranteed error. */
  requiresMetricAnchor?: boolean;
  /** Shown only when `serverAvailable` is false — why, honestly, without inventing a promise.
   *  (No format sets this today; kept for a future format the route can't yet produce.) */
  unavailableReason?: string;
}

/** Static catalog — the three real formats. Order matches the handoff spec's write-up. */
export const EXPORT_FORMATS: readonly ExportFormatDescriptor[] = [
  {
    format: 'openreality',
    label: 'OpenReality native',
    summary: 'Trajectory + ego-view video + world cloud + grounding sidecar (the format we own).',
    docs: 'server/docs/dataset-export.md#4-the-openreality-export-format-the-format-we-own--full-spec',
    serverAvailable: true,
  },
  {
    format: 'groot_lerobot_v2',
    label: 'GR00T-LeRobot v2',
    summary: 'Thin transcode of the OpenReality tree into the GR00T-LeRobot v2 layout (state/action/video + modality.json).',
    docs: 'server/docs/dataset-export.md#5-the-groot-lerobot-v2-converter--target-after-the-openreality-tree-exists',
    serverAvailable: true,
  },
  {
    format: 'isaac_usd',
    label: 'Isaac Sim / Isaac Lab USD',
    summary: 'Metric, gravity-aligned USD scene (scene.usd + trajectory.usd + manifest.json) for Isaac Sim/Lab. Needs a metric scale — run Refine → Metric anchor first.',
    docs: 'server/docs/isaac-export.md',
    serverAvailable: true,
    requiresMetricAnchor: true,
  },
] as const;

/** The `?source=` value that forces the ORIGINAL geometry even when a scan has a persisted
 *  "latest calibrated" pointer (mirrors the server's `EXPORT_SOURCE_ORIGINAL`). */
export const EXPORT_SOURCE_ORIGINAL = 'original' as const;

/** `GET /api/scenes/<scan_id>/export?format=<format>` — the real, deployed broker route. On
 *  success the response body is the zip itself (`application/zip`), not JSON.
 *
 *  `source` is the OPTIONAL geometry selector (product-workflow Refine → Export): pass a
 *  `derived/...` key (from an anchor/clamp response or `PersistedScene.derived_latest.source_key`)
 *  to export the calibrated/clamped result, or `EXPORT_SOURCE_ORIGINAL` to force the original.
 *  Omit it to let the server default (persisted pointer if any, else original). Precedence on the
 *  server: explicit `source` > persisted pointer > original. An unknown/foreign `source` key is a
 *  404 (`unknown_derived_key`). */
export const SCENE_EXPORT_PATH = (
  scanId: string,
  format: ExportFormat,
  source?: string,
): string => {
  const base = `/api/scenes/${encodeURIComponent(scanId)}/export?format=${encodeURIComponent(format)}`;
  return source ? `${base}&source=${encodeURIComponent(source)}` : base;
};

/** The filename the server sets via `Content-Disposition` on a successful export download.
 *  Prefer reading the real header when available; this is the documented fallback/expected
 *  shape (`scan-<scan_id>-<format>.zip`). */
export const sceneExportDownloadFilename = (scanId: string, format: ExportFormat): string =>
  `scan-${scanId}-${format}.zip`;

/** Error codes the real route returns (see file header for the status -> code mapping).
 *  `unknown_derived_key` (404) is returned when a `?source=` selector names a derived artifact
 *  that isn't a real derived key of this scan. `metric_scale_required` (409) and
 *  `isaac_unavailable` (501) are isaac_usd-specific: no metric scale / the deployment lacks the
 *  USD deps. `not_implemented` is retired — isaac_usd is now a real, gated export. */
export type ExportErrorCode =
  | 'invalid_format'
  | 'invalid_request'
  | 'not_found'
  | 'no_trajectory'
  | 'no_points'
  | 'export_failed'
  | 'unknown_derived_key'
  | 'metric_scale_required'
  | 'isaac_unavailable';

/** JSON error body shape for every non-2xx response from the export route. */
export interface SceneExportErrorResponse {
  error: ExportErrorCode;
  message?: string;
}

function isExportErrorCode(value: unknown): value is ExportErrorCode {
  return (
    value === 'invalid_format' ||
    value === 'invalid_request' ||
    value === 'not_found' ||
    value === 'no_trajectory' ||
    value === 'no_points' ||
    value === 'export_failed' ||
    value === 'unknown_derived_key' ||
    value === 'metric_scale_required' ||
    value === 'isaac_unavailable'
  );
}

/** True for the isaac_usd-specific runtime failure where the deployment hasn't enabled the heavy
 *  USD deps (`usd-core`/`open3d`). The UI surfaces this as an honest "not enabled here" state
 *  rather than a generic failure. */
export function isIsaacUnavailable(body: unknown): boolean {
  return isSceneExportErrorResponse(body) && body.error === 'isaac_unavailable';
}

/** True for the isaac_usd gate: the scan has no metric scale yet (run Refine → Metric anchor). */
export function isMetricScaleRequired(body: unknown): boolean {
  return isSceneExportErrorResponse(body) && body.error === 'metric_scale_required';
}

/** Hand-written guard (never throws) — validates a `SceneExportErrorResponse` shape
 *  defensively, since it's parsed from a live HTTP error body. */
export function isSceneExportErrorResponse(value: unknown): value is SceneExportErrorResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return isExportErrorCode(o.error) && (o.message === undefined || typeof o.message === 'string');
}
