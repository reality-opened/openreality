/**
 * Demo-surface contract (2026-07 end-of-July demo build) — REST types + UICommand payload
 * shapes for the founder-only `demo.html` workspace. Additive module: nothing here changes
 * the product contract; the demo server blueprint (`server/server/demo/`) mirrors these.
 *
 * Coordinate/units doctrine (platform `docs/demo-2026-07/WORLD-TRANSFORM-CONTRACT.md`):
 *   - All payload coordinates are WORLD frame (SLAM gauge, un-rotated, un-offset).
 *   - Every length-carrying payload states `units: 'm' | 'relative'` + a `units_basis`;
 *     the UI unit chip is driven by these fields, never inferred.
 *   - OBBs are `{center, extents (FULL lengths), rotation (3x3, columns = axes)}` —
 *     identical to ./objectLayer's ObjectLayerOBB convention.
 */

import type { ObjectLayerOBB, ObjectQualityTier } from './objectLayer';
import type { ExportFormat } from './exportHub';
import type { SceneMetrics, SceneObjectInstance, SceneSyntheticViewRef } from './types';

// ── Shared scalars ──

/** `[x, y, z]` in the scene's WORLD frame (SLAM units unless stated otherwise). */
export type OreosVec3 = [number, number, number];

/** Unit tag carried by every length-bearing demo payload. `'m'` is legal ONLY when the value
 *  came through a metric anchor (or depth-anchor) scale factor. */
export type OreosUnits = 'm' | 'relative';

/** Honesty envelope carried in every generated artifact's meta.json (object-layer pattern). */
export interface OreosHonestyEnvelope {
  provenance: string;
  generated: boolean;
  generator?: 'sam3d' | 'trellis' | 'flux-kontext' | 'planner' | 'recolor' | string;
  quality?: ObjectQualityTier;
  caveats?: string[];
  inputs?: Record<string, unknown>;
}

// ── REST paths (demo blueprint routes; same builder style as ./rest REST_PATHS) ──

export const OREOS_REST_PATHS = {
  INGEST_VIDEO: '/api/workspace/ingest/video',
  /** Robot recording (.db) — the merged Oreos recordings lane; 202 {job_id, scan_id},
   *  polled through WORKSPACE_JOB exactly like a video recon job. */
  INGEST_RECORDING: '/api/workspace/ingest/recording',
  /** Single-request splat import. Small files only — see INGEST_SPLAT_INIT. */
  INGEST_SPLAT: '/api/workspace/ingest/splat',
  /**
   * Chunked splat import. A single-request upload has to finish inside the
   * platform's hard 150-second HTTP request timeout, so anything bigger than the
   * server's inline cap (64 MiB) must go through init → chunk × N → finalize,
   * which has no time limit. `finalize` answers 202 {job_id, scan_id}; poll it
   * with WORKSPACE_JOB exactly like a video recon job.
   */
  INGEST_SPLAT_INIT: '/api/workspace/ingest/splat/init',
  INGEST_SPLAT_CHUNK: (uploadId: string, index: number) =>
    `/api/workspace/ingest/splat/${encodeURIComponent(uploadId)}/chunk/${index}`,
  INGEST_SPLAT_FINALIZE: (uploadId: string) =>
    `/api/workspace/ingest/splat/${encodeURIComponent(uploadId)}/finalize`,
  INGEST_SPLAT_CANCEL: (uploadId: string) =>
    `/api/workspace/ingest/splat/${encodeURIComponent(uploadId)}`,
  WORKSPACE_JOB: (jobId: string) => `/api/workspace/jobs/${encodeURIComponent(jobId)}`,
  SCENE_JOB: (scanId: string, jobId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/jobs/${encodeURIComponent(jobId)}`,
  SCENE_SEGMENT: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/segment`,
  SCENE_OBJECT_COMPLETE: (scanId: string, uid: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/objects/${encodeURIComponent(uid)}/complete`,
  SCENE_OBJECT_VARIANTS: (scanId: string, uid: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/objects/${encodeURIComponent(uid)}/variants`,
  SCENE_NAV_PLAN: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/nav/plan`,
  SCENE_PLANES: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/planes`,
  SCENE_MEASURE: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/measure`,
  SCENE_DEMO_DOC: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/demo/doc`,
  SCENE_AGENT_ANNOTATE: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/agent/annotate`,
  SCENE_AGENT_PILOT: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/agent/pilot`,
  SCENE_AGENT_RUN_EVENTS: (scanId: string, runId: string, after?: number) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/agent/runs/${encodeURIComponent(runId)}/events` +
    (after != null ? `?after=${after}` : ''),
  /** Run index for the scene (W2 as-built): `GET → {runs[], active_run_id}`. */
  SCENE_AGENT_RUNS: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/agent/runs`,
  /** Chat turn (its own run): `POST {message, run_id?} → 202 {run_id}`. */
  SCENE_AGENT_CHAT: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/agent/chat`,
  /** F6 floor patch under a hidden object (W5 as-built, `routes_planes.py`). */
  SCENE_OBJECT_FLOOR_PATCH: (scanId: string, uid: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/objects/${encodeURIComponent(uid)}/floor_patch`,
  /** Level-of-detail index (`GET`) / generation (`POST`), `routes_lod.py`. The LOD
   *  binaries themselves ride the existing derived route, `SCENE_DERIVED`. */
  SCENE_LOD: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/lod`,
  /** Export manifest viewer route (W6 as-built, `routes_export.py`): JSON tree +
   *  info/modality — no zip. `source` selects 'original' or a derived/... key. */
  SCENE_EXPORT_MANIFEST: (scanId: string, format: ExportFormat, source?: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/export/manifest?format=${encodeURIComponent(format)}` +
    (source ? `&source=${encodeURIComponent(source)}` : ''),
  /** Prepared export (`routes_export_job.py`): `POST` spawns the build as a background
   *  job — 202 {job_id}, polled through the EXISTING WORKSPACE_JOB route. Exists because
   *  the synchronous export deflates the whole tree before sending a byte (45 s on a
   *  1.39 GB scene) and Modal kills a web request at 150 s. */
  SCENE_EXPORT_PREPARE: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/export/prepare`,
  /** What is prepared for this scene+format. Always 200 with an explicit `status`, so the
   *  panel never has to interpret a 404 (same shape as SCENE_LOD). */
  SCENE_EXPORT_PREPARED: (scanId: string, format: ExportFormat, source?: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/demo/export/prepared?format=${encodeURIComponent(format)}` +
    (source ? `&source=${encodeURIComponent(source)}` : ''),
  /** Synthetic views — renders of an imported splat registered as evidence
   *  (`POST` to register, `GET` for the manifest). See SceneSyntheticViewRef. */
  SCENE_SYNTHETIC_VIEWS: (scanId: string) =>
    `/api/demo/scene/${encodeURIComponent(scanId)}/synthetic_views`,
  /** One synthetic view's PNG bytes (owner-authed). */
  SCENE_SYNTHETIC_VIEW_PNG: (scanId: string, viewId: string) =>
    `/api/demo/scene/${encodeURIComponent(scanId)}/synthetic_views/${encodeURIComponent(viewId)}.png`,
  /** Ground frame: `POST` fits the floor plane and writes it onto facts.metrics,
   *  `GET` reads back what is recorded. See server/demo/ground_frame.py. */
  SCENE_GROUND_FRAME: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/ground_frame`,
  /** Geometry objects for an imported splat (`routes_sam3d.py`): `GET` the last run's
   *  provenance doc, `POST` to cluster the scan's geometry into `facts.objects`. */
  SCENE_IMPORTED_OBJECTS: (scanId: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/imported_objects`,
} as const;

/** Derived key of the shell's single manifest index (read via ./rest SCENE_DERIVED). */
export const OREOS_MANIFEST_KEY = 'derived/demo/manifest.json' as const;

// ── Ingest: video upload → recon job ──

/** `POST /api/workspace/ingest/video` — raw octet-stream body; filename in the
 *  `X-Upload-Filename` header (optional label in `X-Scene-Label`). Responds 202. */
export interface IngestVideoResponse {
  job_id: string;
  scan_id: string;
}

/** Pipeline stage of a demo recon job (upload → recon → report → persist). */
export type OreosJobStage = 'upload' | 'recon' | 'report' | 'persist';

/** Tolerant union — the recon path says 'error', the gen-asset/ingest paths say 'failed'
 *  for the same terminal state. Compare via {@link normalizeOreosJobState}, never both
 *  spellings by hand. */
export type OreosJobState = 'queued' | 'running' | 'done' | 'error' | 'failed';

/** Canonical job state: collapses the legacy 'failed' spelling onto 'error'; anything
 *  unrecognized also reads as 'error' (fail-visible, never fail-silent). */
export function normalizeOreosJobState(state: string): 'queued' | 'running' | 'done' | 'error' {
  if (state === 'queued' || state === 'running' || state === 'done') return state;
  return 'error';
}

/** Envelope from `GET /api/workspace/jobs/<job_id>` and `GET /api/scenes/<id>/jobs/<job_id>`
 *  (both poll at ~1 Hz). `stage` is recon-pipeline-specific; gen-asset jobs omit it. */
/** A built export sitting on the volume, ready to stream. */
export interface OreosPreparedExportArtifact {
  format: ExportFormat;
  /** Derived key of the zip. Prefer `download_path` over rebuilding this by hand. */
  key: string;
  source_key: string;
  /** Compressed size — what the browser will actually transfer. */
  bytes: number;
  /** Uncompressed tree size, for an honest "what is in here" line. */
  tree_bytes: number;
  file_count: number;
  built_at?: number;
  /** Seconds since it was built, so the panel can say "prepared 3 min ago" without
   *  trusting the client's clock against the server's. */
  age_s?: number | null;
  /** Wall-clock seconds the build took (measured 48.8 s for the 1.39 GB canonical scene). */
  build_seconds?: number;
  /** Ready-made URL for the existing derived streaming route. */
  download_path: string;
  /** Set when the builder had to subsample before it could work — Isaac meshes at most
   *  ~5M points, so a large capture is exported from a uniform subsample. Both counts are
   *  carried so nobody reads the export as the scene's real density. */
  decimated?: boolean;
  original_points?: number | null;
  exported_points?: number | null;
  /** Filename to save as — matches `sceneExportDownloadFilename`. */
  download_filename?: string;
}

/** `GET …/export/prepared`. Always 200; `status` carries the meaning. */
export interface OreosPreparedExportResponse {
  status: 'ready' | 'running' | 'error' | 'none';
  artifact?: OreosPreparedExportArtifact;
  /** True when the slot predates the scene's current geometry — still served, but the
   *  panel offers a rebuild. Deleting the operator's only copy is the worse failure. */
  stale?: boolean;
  /** Why it is stale, in words the panel can show: different geometry, or an age. */
  stale_reason?: string | null;
  job_id?: string;
  stage?: string;
  error?: string;
  /** Uncompressed tree size, published by the running job the moment the tree exists —
   *  so a wait stops being abstract before the zip is finished. */
  tree_bytes?: number;
  zip_bytes?: number;
}

export interface OreosJobStatus {
  job_id: string;
  status: OreosJobState;
  stage?: OreosJobStage;
  /** 0–1 coarse progress when the job reports one. */
  progress?: number;
  /** Set when status = 'done' and the job yields a payload (shape is per-job). */
  result?: Record<string, unknown>;
  /** Set when status = 'error'. */
  error?: string;
  scan_id?: string;
  created_at?: number;
  updated_at?: number;
  /** Export jobs publish measured sizes at stage boundaries (`modal_demo_export.py`):
   *  the uncompressed tree once it is written, the zip once it is deflated. Absent on
   *  other job kinds and before the stage that produces them — never defaulted to 0,
   *  because "0 bytes" and "not measured yet" are different things. */
  tree_bytes?: number;
  zip_bytes?: number;
  file_count?: number;
}

export function isOreosJobStatus(value: unknown): value is OreosJobStatus {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.job_id === 'string' &&
    (o.status === 'queued' ||
      o.status === 'running' ||
      o.status === 'done' ||
      o.status === 'error' ||
      o.status === 'failed')
  );
}

// ── Ingest: splat import ──

/** `POST /api/workspace/ingest/splat` — raw .ply/.spz stream, `X-Upload-Filename` header.
 *  Synchronous on the broker; responds 201 with the persisted scene. Gaussian-schema
 *  failures are 422 `{error: "not_a_gaussian_splat"}`. */
export interface IngestSplatResponse {
  scan_id: string;
  gaussian_count: number;
}

export function isIngestSplatResponse(value: unknown): value is IngestSplatResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return typeof o.scan_id === 'string' && typeof o.gaussian_count === 'number';
}

// ── Agent runs (REST run + 1s poll; replay is a re-serve of the persisted event log) ──

/** `POST .../demo/agent/annotate` body. As-built (W2) the route drives on `mode` /
 *  `replay_of`; `goal` is the legacy Day-1 contract shape and stays accepted. Exactly one
 *  of `mode` | `replay_of` | `goal` is meaningful per request. */
export interface AgentAnnotateRequest {
  /** Legacy free-text goal (Day-1 walking skeleton). */
  goal?: string;
  /** As-built annotate trigger: the full five-phase annotation run. */
  mode?: 'full';
  /** Server-paced re-serve of a recorded run's event log (structural REPLAY badge). */
  replay_of?: string;
  /** Replay pacing multiplier (1 = recorded pacing). */
  speed?: number;
  /** Idempotent start: when a run is already live on this scene, attach to it (202
   *  `{run_id, attached: true}`) instead of 409. Omit it to keep the honest 409 — a
   *  second operator/tab must be told the scene is busy, not silently joined. Never
   *  starts a second run either way. */
  attach_if_active?: boolean;
}

/** `POST .../demo/agent/annotate` / `POST .../demo/agent/pilot {instruction}` → 202.
 *  One active run per scene — a second start is 409 `{error: 'agent_run_active',
 *  active_run_id}` (clients offer a "watch active run" affordance). */
export interface AgentRunStartResponse {
  run_id: string;
  /** True when `attach_if_active` matched a live run: `run_id` is that EXISTING run,
   *  nothing new was started (and nothing new was spent). */
  attached?: boolean;
  /** Set on a `replay_of` start. */
  replay?: boolean;
  source_run_id?: string;
}

/** `POST .../demo/agent/chat` — a chat turn as its own bounded run. User turns echo back
 *  as `agent_thought` events with `author: 'user'`. */
export interface AgentChatRequest {
  message: string;
  /** Attach the turn to a prior run's context (defaults to a fresh context). */
  run_id?: string;
}

/** One row of `GET .../demo/agent/runs` (W2 as-built run index; newest-first). */
export interface AgentRunListItem {
  run_id: string;
  /** `'replay'` = a server-paced re-serve of another run's log (it also has
   *  `replay: true` and a `source_run_id`). */
  kind: 'annotate' | 'pilot' | 'chat' | 'replay';
  status: OreosAgentRunStatus;
  replay: boolean;
  started_at: number;
  n_events: number;
  cost_usd: number | null;
  /** Epoch seconds; null while the run is still going. */
  finished_at?: number | null;
  /** Why a `status: 'error'` run failed — rendered verbatim, never invented. */
  error?: string | null;
  /** Recorded in the run's `run_done` event (null for runs that never reached it). */
  llm_calls?: number | null;
  findings_emitted?: number | null;
  /** The run this one replays, when `kind: 'replay'`. */
  source_run_id?: string | null;
  /** The run's event log survives a broker restart under
   *  `derived/demo/agent_runs/<run_id>/events.json`. */
  persisted?: boolean;
}

/** `GET .../demo/agent/runs` → the scene's run index + which run (if any) is live. */
export interface AgentRunsListResponse {
  runs: AgentRunListItem[];
  active_run_id?: string | null;
}

/** One event in a demo agent run. `seq` is a per-run monotonic cursor (poll with
 *  `?after=<seq>`); `type` uses the existing agent event vocabulary (thought / tool /
 *  finding / ui_command / run_done ...) so the stock feed components render it. */
export interface OreosAgentEvent {
  seq: number;
  /** Epoch seconds. */
  ts: number;
  type: string;
  payload: Record<string, unknown>;
  /** Annotation-run phase this event belongs to (W2 as-built:
   *  survey → labels → description → dimensions → key_features). */
  phase?: string;
}

/** Metadata for a run; persisted alongside the event log. `replay: true` marks a re-served
 *  historical run — the client MUST render the structural REPLAY badge (no off switch). */
export interface OreosAgentRunMeta {
  run_id: string;
  replay: boolean;
  started_at?: number;
  model?: string;
  /** Actual LLM spend for the run, when known (replays are $0). */
  cost_usd?: number;
}

export type OreosAgentRunStatus = 'running' | 'done' | 'error';

/** `GET .../demo/agent/runs/<run_id>/events?after=<seq>` → the events after the cursor plus
 *  run status and the next cursor to poll from. */
export interface AgentRunEventsResponse {
  events: OreosAgentEvent[];
  status: OreosAgentRunStatus;
  /** Pass as the next `?after=`. */
  next: number;
  run_meta?: OreosAgentRunMeta;
  /** Present (null when healthy) so `status: 'error'` can be shown WITH its reason.
   *  Rendered verbatim — the client never paraphrases or invents a cause. */
  error?: string | null;
  /** Set by `?replay=1` (the stored log re-served as a replay stream). */
  replay?: boolean;
}

export function isAgentRunEventsResponse(value: unknown): value is AgentRunEventsResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    Array.isArray(o.events) &&
    (o.status === 'running' || o.status === 'done' || o.status === 'error') &&
    typeof o.next === 'number'
  );
}

// ── Measurement (F7) ──

export type MeasureKind = 'distance' | 'angle';

/** `POST /api/scenes/<id>/measure`. 2 world points for distance, 3 for angle (vertex is the
 *  2nd point). Points are WORLD frame, SLAM units. */
export interface MeasureRequest {
  kind: MeasureKind;
  points_world: OreosVec3[];
}

/** `value` is in `units` (`'m'` only when a scale factor was applied — then `scale_factor` +
 *  `scale_source` say which anchor). Angles additionally carry `degrees`. The client NEVER
 *  renders an "m" glyph for `units: 'relative'` (provenance chip enforces). */
export interface MeasureResponse {
  kind: MeasureKind;
  value: number;
  units: OreosUnits;
  scale_factor?: number | null;
  /** e.g. `anchor:derived/anchor/<stamp>` | `'none'`. */
  scale_source: string;
  degrees?: number;
}

export function isMeasureResponse(value: unknown): value is MeasureResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    (o.kind === 'distance' || o.kind === 'angle') &&
    typeof o.value === 'number' &&
    (o.units === 'm' || o.units === 'relative') &&
    typeof o.scale_source === 'string'
  );
}

// ── Pathfinding (F4) ──

/** Goal for a nav plan: a clicked world point, or a registry object uid whose OBB center is
 *  the target. Exactly one of the two. */
export type NavGoal = { point_world: OreosVec3; object_uid?: never } | { object_uid: string; point_world?: never };

/** Up-axis override: a signed canonical axis (the UI flip control) or an explicit
 *  world-frame vector. */
export type NavUpAxis = '+y' | '-y' | '+z' | '-z';
export type NavUpOverride = NavUpAxis | OreosVec3;

export interface NavPlanParams {
  clearance?: number;
  band_lo?: number;
  band_hi?: number;
  eye_height?: number;
  /** Units per second along the resampled path. */
  speed?: number;
  fps?: number;
  up_override?: NavUpOverride;
}

/** `POST /api/scenes/<id>/nav/plan`. Errors: 422 `{error: 'no_geometry'|'no_floor'|
 *  'unreachable_goal'}` (unreachable includes a nearest-reachable suggestion when one exists). */
export interface NavPlanRequest {
  goal: NavGoal;
  start?: OreosVec3;
  params?: NavPlanParams;
}

/** One robot-height camera pose along the planned path (WORLD frame, OpenCV c2w). */
export interface NavPose {
  /** 4x4 row-major c2w matrix. */
  c2w: number[][];
  /** Seconds from path start. */
  t: number;
}

export interface NavPlanResponse {
  path_id: string;
  waypoints_world: OreosVec3[];
  poses: NavPose[];
  floor?: { point: OreosVec3; normal: OreosVec3; up?: OreosVec3; up_source: string };
  grid?: {
    n_free: number;
    n_components: number;
    /** Cells in the largest connected component actually planned over. */
    n_navigable?: number;
    cell_size: number;
    /** 'm' | 'relative' — mirrors the plan's `units` for the grid raster. */
    cell_size_units?: string;
  };
  units: OreosUnits;
  /** e.g. `anchor:<derived key>` | `capture_height_fraction` | `extent_fraction`. */
  units_basis: string;
  /** Path summary (as-built): lengths in `units`, snaps are start/goal substitutions. */
  stats?: {
    path_length: number;
    duration_s: number;
    n_frames: number;
    goal_snap: number;
    start_snap: number;
  };
  /** Human-readable planner notes (substitutions, heuristic up, band clamps …). */
  notes?: string[];
  /** Whether this plan re-rastered the occupancy grid or served the cached one. */
  cache?: { source: 'built' | 'cache'; grid_rebuilt: boolean };
  /** Fixed honesty line: "Planned in scanned free space … not certified navigation." */
  provenance: string;
  /**
   * Derived key of the persisted path doc (waypoints + per-frame poses), or `null`
   * when the store write failed — the plan itself still stands.
   */
  doc_key?: string | null;
  /** Derived key of the planner debug dump when requested with `?debug=1`. */
  debug_key?: string;
}

export function isNavPlanResponse(value: unknown): value is NavPlanResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.path_id === 'string' &&
    Array.isArray(o.waypoints_world) &&
    Array.isArray(o.poses) &&
    (o.units === 'm' || o.units === 'relative') &&
    typeof o.units_basis === 'string' &&
    typeof o.provenance === 'string'
  );
}

// ── Segmentation (F3) — response of POST /api/scenes/<id>/segment ──

/** Which prompt-view the segmentation ran on (honesty ladder). */
export type SegmentPromptSource = 'detection_keyframe' | 'indexed_keyframe' | 'rendered_view';

export interface SegmentResponse {
  /** Registry uid, `sel:<stamp>`. */
  uid: string;
  obb: ObjectLayerOBB;
  n_points: number;
  /** Derived key of the persisted mask.png. */
  mask_key: string;
  prompt_source: SegmentPromptSource;
  label?: string;
}

// ── Planes (F5) — response of POST /api/scenes/<id>/planes (W5 as-built wire shape) ──

/** One RANSAC plane candidate. Axes are WORLD frame; `uv_bounds` are the inlier extents
 *  along `u_axis`/`v_axis` about `point`; `thickness` is the inlier slab thickness. */
export interface PlaneCandidate {
  id: string;
  kind: 'floor' | 'wall';
  point: OreosVec3;
  normal: OreosVec3;
  u_axis: OreosVec3;
  v_axis: OreosVec3;
  uv_bounds: [[number, number], [number, number]];
  thickness: number;
  inlier_count: number;
}

export interface PlanesResponse {
  planes: PlaneCandidate[];
  /** World up used to split floor/wall (pose-derived, else heuristic). */
  up?: OreosVec3 | null;
  up_source?: string | null;
  provenance?: string | null;
  /** True when served from the per-scene plane cache. */
  cached?: boolean;
}

// ── Floor patch (F6) — response of POST /api/scenes/<id>/objects/<uid>/floor_patch ──

/** Synthesized floor-colored gaussians behind a hidden object (always `generated: true`
 *  in the honesty envelope at `meta_key`). */
export interface FloorPatchResponse {
  /** Derived key of the generated patch.ply. */
  patch_key: string;
  /** Derived key of the honesty-envelope meta.json. */
  meta_key: string;
  n_gaussians: number;
  units?: OreosUnits;
  units_basis?: string;
  meta?: Record<string, unknown>;
}

// ── Imported-splat objects (W-B) — /api/scenes/<id>/imported_objects ──

/** How many labels the synthetic-view pass produced, and why it produced fewer than one
 *  per object. `note` is present exactly when no label could be made honestly. */
export interface ImportedObjectsLabelSummary {
  mode: 'auto' | 'geometry';
  attempted?: number;
  labelled?: number;
  declined?: number;
  errors?: number;
  camera_convention?: string;
  view_in_frame_frac?: number;
  note?: string;
}

/** Response of `POST /api/scenes/<id>/imported_objects`. `objects` is written straight
 *  into `facts.objects`, so it is the SAME shape a captured scene persists. */
export interface ImportedObjectsResponse {
  scan_id: string;
  run_id: string;
  count: number;
  /** Objects from a previous geometry run that this one replaced. */
  replaced: number;
  objects: SceneObjectInstance[];
  labels: ImportedObjectsLabelSummary;
  provenance: string;
  caveats: string[];
  up: number[];
  up_source: string;
  diagnostics: Record<string, unknown>;
  rejected: Record<string, number>;
  derived_key: string;
}

/** Response of `GET /api/scenes/<id>/imported_objects` — always 200. */
export interface ImportedObjectsStatus {
  scan_id: string;
  status: 'none' | 'ready';
  object_count?: number;
  reason?: string;
  /** Whether the scan has geometry to cluster at all. */
  eligible?: boolean;
  run?: Record<string, unknown>;
}

// ── Export manifest (W6) — GET /api/scenes/<id>/demo/export/manifest?format=… ──

export interface ManifestTreeEntry {
  /** Zip-entry path (`<scan_id>/…` or `<scan_id>_groot_lerobot_v2/…`) — `unzip -l` parity. */
  path: string;
  size: number;
}

export interface ManifestAbsentEntry {
  component: string;
  reason: string;
}

export interface ExportManifestResponse {
  scan_id: string;
  format: ExportFormat;
  /** Geometry selector the builders read: 'original' or a derived/... key. */
  source_key: string;
  complete: boolean;
  absent: ManifestAbsentEntry[];
  tree: ManifestTreeEntry[];
  file_count: number;
  total_bytes: number;
  /** meta/info.json (openreality/groot) or the tree's own isaac/manifest.json (isaac). */
  info: Record<string, unknown> | null;
  /** openreality only: meta/episode.json. */
  episode: Record<string, unknown> | null;
  /** groot only. */
  modality: Record<string, unknown> | null;
  episodes: Array<Record<string, unknown>> | null;
  tasks: Array<Record<string, unknown>> | null;
  /** Whether the existing zip route would produce a download for this scan+format. */
  zip_available: boolean;
  zip_blocked_reason: string | null;
  /** isaac only: {scale, scale_source, anchor_scale}. */
  scale?: Record<string, unknown> | null;
}

export function isExportManifestResponse(value: unknown): value is ExportManifestResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.scan_id === 'string' &&
    typeof o.format === 'string' &&
    typeof o.complete === 'boolean' &&
    Array.isArray(o.absent) &&
    Array.isArray(o.tree) &&
    typeof o.total_bytes === 'number'
  );
}

// ── Demo manifest (derived/demo/manifest.json — the shell's single index) ──

export interface OreosManifestRunRef {
  run_id: string;
  kind: 'annotate' | 'pilot';
  created_at: number;
  /** Derived key of the flushed event log. */
  events_key: string;
}

export interface OreosManifestDocRef {
  /** Derived key of the doc (e.g. `derived/demo/edits/<stamp>/edit.json`). */
  key: string;
  created_at: number;
  label?: string;
}

/** The single demo index the shell fetches at workspace load (server updates it under a
 *  lock on every demo write). All arrays newest-last; absent file = empty manifest. */
export interface OreosManifest {
  version: number;
  /** Epoch seconds of the last write. */
  updated_at: number;
  /** Derived key of annotations v2 doc, when F2 has run. */
  annotations_key?: string | null;
  agent_runs: OreosManifestRunRef[];
  edits: OreosManifestDocRef[];
  paths: OreosManifestDocRef[];
  variations: OreosManifestDocRef[];
  measurements: OreosManifestDocRef[];
}

/** An empty manifest for scenes with no demo artifacts yet (404 on the manifest key). */
export function emptyOreosManifest(): OreosManifest {
  return {
    version: 1,
    updated_at: 0,
    annotations_key: null,
    agent_runs: [],
    edits: [],
    paths: [],
    variations: [],
    measurements: [],
  };
}

export function isOreosManifest(value: unknown): value is OreosManifest {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.version === 'number' &&
    Array.isArray(o.agent_runs) &&
    Array.isArray(o.edits) &&
    Array.isArray(o.paths) &&
    Array.isArray(o.variations) &&
    Array.isArray(o.measurements)
  );
}

/** `PUT /api/scenes/<id>/demo/doc` — generic guarded doc writer (namespace-restricted to
 *  `derived/demo/...` suffixes so W4/W5/W6 need no server merges). */
export interface OreosDocPutRequest {
  /** Suffix under `derived/demo/`, e.g. `edits/<stamp>/edit.json`. */
  key_suffix: string;
  json: Record<string, unknown>;
}

export interface OreosDocPutResponse {
  /** Full derived key the doc landed at. */
  key: string;
}

// ── Demo UICommand payloads ──
// The names below extend AgentUICommand['name'] (see ./types). `args` stays
// `Record<string, unknown>` on the wire; these are the typed shapes the demo shell
// narrows to. All coordinates WORLD frame.

/** `drive_path` — animate the robot along a planned path (F4 agent pilot). */
export interface DrivePathArgs {
  path_id: string;
  /** Inline waypoints for feeds that missed the plan response; else looked up by path_id. */
  waypoints_world?: OreosVec3[];
  fps?: number;
}

/** `recolor_region` — apply a live recolor edit to a scoped region (F5). */
export interface RecolorRegionArgs {
  scope:
    | { kind: 'obb'; obb: ObjectLayerOBB }
    | { kind: 'plane'; plane_id: string }
    | { kind: 'object'; object_uid: string };
  /** 0–1 floats. */
  rgba: [number, number, number, number];
  /** 0–1 blend toward the target color (1 = full recolor); default is a tint. */
  blend?: number;
}

/** `open_export` — open the export panel on a format tab. */
export interface OpenExportArgs {
  format: 'openreality' | 'groot_lerobot_v2' | 'isaac_usd';
}

/** `focus_object` — fly the camera to a registry object. */
export interface FocusObjectArgs {
  /** `det:<idx>` | `sel:<stamp>` | `layer:<id>`. */
  object_uid: string;
  distance?: number;
}

/** `show_measurement` — render a measurement the agent computed (numbers from code). */
export interface ShowMeasurementArgs {
  kind: MeasureKind;
  points_world: OreosVec3[];
  value: number;
  units: OreosUnits;
  degrees?: number;
  label?: string;
}

/** `show_dimension_labels` — toggle per-object OBB dimension labels. */
export interface ShowDimensionLabelsArgs {
  visible: boolean;
  /** Restrict to specific objects; absent = all labeled objects. */
  object_uids?: string[];
}

/** Typed args by demo command name (narrow an AgentUICommand's `args` via this map). */
export interface OreosUICommandArgs {
  drive_path: DrivePathArgs;
  recolor_region: RecolorRegionArgs;
  open_export: OpenExportArgs;
  focus_object: FocusObjectArgs;
  show_measurement: ShowMeasurementArgs;
  show_dimension_labels: ShowDimensionLabelsArgs;
}

export type OreosUICommandName = keyof OreosUICommandArgs;

export const OREOS_UI_COMMAND_NAMES: readonly OreosUICommandName[] = [
  'drive_path',
  'recolor_region',
  'open_export',
  'focus_object',
  'show_measurement',
  'show_dimension_labels',
] as const;

export function isOreosUICommandName(name: string): name is OreosUICommandName {
  return (OREOS_UI_COMMAND_NAMES as readonly string[]).includes(name);
}

// ── Level of detail (LOD) ────────────────────────────────────────────────────
//
// Persisted splats are far too large to render as exported (the canonical demo
// scene is 18.85M gaussians / 1.19 GiB; a real founder upload was 63.3M / 4.01
// GiB). The server writes decimated variants under `derived/demo/lod/` plus an
// index describing them; the client loads one of those, never the raw splat.ply,
// and must label which one it is showing.

/** One decimated level. `key` is a PLY; `spz_key` (when present) is the same
 *  level as a Spark-native `.spz` — ~9-15x smaller and what clients should load. */
export interface OreosLodLevel {
  /** Short token, e.g. `"2000k"`. */
  name: string;
  /** Requested gaussian budget. */
  budget: number;
  /** Gaussians actually written (voxel selection lands near, not exactly, on budget). */
  gaussians: number;
  /** Derived key of the PLY variant (relative to `derived/`). */
  key: string;
  /** Byte size of the PLY variant. */
  bytes: number;
  /** Derived key of the `.spz` variant, when encoding succeeded. */
  spz_key?: string;
  /** Byte size of the `.spz` variant. */
  spz_bytes?: number;
  /** Why the `.spz` variant is absent, when it is. */
  spz_error?: string;
  /** Source-size / this-size, for the honest "N x smaller" line. */
  size_reduction?: number;
  spz_size_reduction?: number;
  /** True when per-gaussian scale was floored to the decimation spacing. */
  scale_floor_applied?: boolean;
}

/** Full-detail transport: the ENTIRE reconstruction, compressed but not decimated.
 *  Absent (or carrying `unavailable_reason`) when the source is too large to render. */
export interface OreosLodFullDetail {
  key?: string;
  gaussians?: number;
  bytes?: number;
  size_reduction?: number;
  unavailable_reason?: 'too_many_gaussians' | string;
  ceiling?: number;
  error?: string;
}

/** `derived/demo/lod/index.json`. */
export interface OreosLodIndex {
  version: number;
  scan_id: string;
  source: { key: string; gaussians: number; bytes: number };
  /** Budget of the level a client should load by default. */
  default_level: number | null;
  levels: OreosLodLevel[];
  full_detail?: OreosLodFullDetail;
  provenance: string;
  generated: boolean;
  generator: string;
  caveats: string[];
  normals_dropped?: boolean;
  build_seconds?: number;
}

export type OreosLodStatus = 'ready' | 'none' | 'running' | 'error' | 'not_needed';

/** `GET /api/scenes/<id>/lod`. Always 200 — read `status`, never infer from a 404. */
export interface OreosLodResponse {
  scan_id: string;
  status: OreosLodStatus;
  index: OreosLodIndex | null;
  source_gaussians?: number | null;
  default_level_target?: number;
  job_id?: string;
  stage?: string;
  error?: string;
  reason?: string;
}

// ── Synthetic views + ground frame (imported-splat parity) ──
//
// A splat imported from Splatica / SuperSplat / Scaniverse has no capture video, so it
// has no keyframes and no gravity. These two payloads are how it gets both back without
// anything pretending to be a photograph: renders of the splat itself, registered as
// evidence in their own field, and a floor plane fitted to the geometry.
// Server: `server/demo/synthetic_views.py`, `server/demo/ground_frame.py`.

/** One view in a `POST …/synthetic_views` body. Poses are WORLD frame — the client
 *  MUST invert the display rotation and scene_center offset before sending (see
 *  WORLD-TRANSFORM-CONTRACT.md); the quaternion is three.js camera convention. */
export interface SyntheticViewUpload {
  /** PNG bytes, base64, NO `data:` prefix (the route refuses one). */
  image_b64: string;
  position: [number, number, number];
  /** `[x, y, z, w]` — three.js order. */
  quaternion: [number, number, number, number];
  fov_y_deg: number;
  width: number;
  height: number;
  label?: string;
}

export interface SyntheticViewsRequest {
  /** Clear the scene's prior view set instead of appending to it. */
  replace?: boolean;
  views: SyntheticViewUpload[];
}

export interface SyntheticViewsResponse {
  views: Array<{ view_id: string; index: number }>;
  count: number;
  replaced?: boolean;
  provenance: string;
  provenance_detail: string;
}

export interface SyntheticViewsListResponse {
  views: Array<SceneSyntheticViewRef & { url: string }>;
  count: number;
  provenance: string;
  provenance_detail: string;
}

/** The derived floor/ceiling frame. Heights are signed distances along `up_axis` in the
 *  world frame, in the scene's own units. A weak fit reports `vertical_axis_known:
 *  false` with a `note` and null numbers — read the flag, never the numbers alone. */
export interface GroundFrame {
  vertical_axis_known: boolean;
  derivation: string | null;
  up_axis: number[] | null;
  floor_height: number | null;
  ceiling_height: number | null;
  room_height: number | null;
  floor_extent: number[] | null;
  floor_area: number | null;
  note?: string | null;
  units: OreosUnits;
  units_basis: string;
  /** Diagnostics present on a freshly computed frame (POST), absent on a read-back. */
  floor_inliers?: number;
  floor_inlier_ratio?: number;
  up_source?: string;
}

export interface GroundFrameResponse {
  scan_id: string;
  frame: GroundFrame;
  computed: boolean;
  persisted?: boolean;
  metrics?: SceneMetrics;
  metrics_patch?: Record<string, unknown>;
}
