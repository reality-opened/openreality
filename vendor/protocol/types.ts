/**
 * Type definitions for the Open Reality client/server contract.
 */

import type { OreosUnits } from './oreos';
import type { ObjectLayerManifest, ObjectLayerOBB } from './objectLayer';
import type { DerivedLatestPointer } from './rest';

export interface SLAMUpdate {
  frame_id: number;
  num_submaps: number;
  num_loops: number;
  /** Binary-encoded point positions: base64(float32 N x 3 flat). Preferred over `points`. */
  points_b64?: string;
  /** Binary-encoded point colors: base64(uint8 N x 3 flat, values 0-255). Preferred over `colors`. */
  colors_b64?: string;
  /** Legacy JSON arrays, empty when binary fields are present. */
  points: number[][];
  colors: number[][];
  camera_positions: number[][];
  camera_rotations: number[][][];
  scene_center: number[];
  n_points: number;
  n_cameras: number;
  resolved_beacons?: ResolvedBeacon[];
  detections?: DetectionResult[];
  active_queries?: string[];
  /** Update type: 'full' replaces entire scene, 'incremental' adds a single submap. */
  type?: 'full' | 'incremental';
  /** Submap ID for incremental updates. */
  submap_id?: number;
}

export interface ResolvedBeacon {
  beacon_id: number;
  x: number;
  y: number;
  z: number;
}

export interface SLAMStats {
  frames: number;
  submaps: number;
  loops: number;
  points: number;
  cameras: number;
  fps: number;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ConnectionStatus {
  state: ConnectionState;
  message: string;
  latency?: number;
}

export interface ControlsConfig {
  showCameras: boolean;
  showPoints: boolean;
  showGrid: boolean;
  showAxes: boolean;
  pointSize: number;
  cameraSize: number;
  followCamera: boolean;
  flipY: boolean;
  showDetectionBoxes: boolean;
  showDetectionLabels: boolean;
}

export interface BoundingBox3D {
  center: number[];
  extent: number[];
  rotation: number[][];
  corners: number[][];
}

/**
 * Real 2D SAM detection geometry, additive/optional on live detections + persisted evidence.
 * See `platform/contracts/export-format.md` § "Grounding evidence: persisted 2D detection
 * geometry". Both fields are optional — older payloads that omit them are unaffected, and a
 * consumer that ignores them sees byte-identical behavior to before.
 */

/** `[x0, y0, x1, y1]` in pixels on the source keyframe. */
export type Box2D = number[];

/**
 * Compact run-length-encoded mask, row-major runs, alternating, starting with a `False`
 * run (mirrors `server/server/export/mask_rle.py`). Typed + guarded here; decoding to a
 * canvas overlay is not yet implemented anywhere in this app — see WORKFLOW-GAPS.md.
 */
export interface MaskRLE {
  size: [number, number];
  counts: number[];
  order: 'C';
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Hand-written guard (never throws) for the optional `box_2d` field. */
export function isBox2D(value: unknown): value is Box2D {
  return Array.isArray(value) && value.length === 4 && value.every(isFiniteNumber);
}

/** Hand-written guard (never throws) for the optional `mask_rle` field. */
export function isMaskRle(value: unknown): value is MaskRLE {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    Array.isArray(o.size) &&
    o.size.length === 2 &&
    o.size.every(isFiniteNumber) &&
    Array.isArray(o.counts) &&
    o.counts.every(isFiniteNumber) &&
    o.order === 'C'
  );
}

export interface DetectionResult {
  success: boolean;
  query: string;
  bounding_box?: BoundingBox3D;
  confidence?: number;
  keyframe_image?: string;
  mask_image?: string;
  matched_submap?: number;
  matched_frame?: number;
  query_time_ms?: number;
  error?: string;
  description?: ObjectDescription;  // fine-grained identification (filled on click / finalize)
  /** Real SAM 2D box on the source keyframe, when the server persisted one. See Box2D above. */
  box_2d?: Box2D;
  /** Real SAM mask on the source keyframe, when the server persisted one. See MaskRLE above. */
  mask_rle?: MaskRLE;
}

/**
 * One reverse-image-search candidate (e.g. Google Lens via SerpApi)
 */
export interface ProductMatch {
  title: string;
  source: string;
  link: string;
  price: string;
}

/**
 * Fine-grained, evidence-grounded identification of a detected object
 * ("laptop" -> "Apple MacBook Pro 14\", Space Black, likely M3").
 */
export interface ObjectDescription {
  title: string;
  category: string;
  brand: string;
  model: string;
  color: string;
  materials: string[];
  visible_text: string[];
  distinguishing_features: string[];
  confidence: number;        // calibrated 0..1 for the brand/model guess
  uncertainty_note: string;
  matches: ProductMatch[];
  model_name: string;        // VLM model id that authored this
  grounded_by_search: boolean;
}

export interface BatchDetectionResponse {
  success: boolean;
  results: DetectionResult[];
  total_queries: number;
  error?: string;
}

export interface DetectionPartialResult {
  detections: DetectionResult[];
  active_queries: string[];
  is_final: boolean;
}

export interface DetectionPreview {
  query: string;
  submap_id: number;
  frame_idx: number;
  keyframe_image?: string;
  mask_image?: string;
  error?: string;
}

/**
 * Request payload for a fine-grained detection description ("click-in").
 *
 * Mirrors exactly what `handle_describe_detection` reads off the wire
 * (`server/server/app.py:4213-4238`): `submap_id`/`frame_idx` are int-coerced
 * (app.py:4233-4234 — anything non-numeric answers `detection_description` with
 * `{error: 'Invalid submap/frame'}`), and `query` is stringified, trimmed and truncated to
 * `MAX_QUERY_LEN` (app.py:4238). Emitted by the viewer at
 * `web/apps/webserver/src/services/SLAMConnection.ts:256`.
 */
export interface DetectionDescriptionRequest {
  submap_id: number;
  frame_idx: number;
  query: string;
}

/**
 * Fine-grained description returned for a specific detection click ("click-in")
 *
 * Every field is optional because the server has two shapes on this one event
 * (`server/server/app.py:4225-4275`): the success emit carries
 * `{submap_id, frame_idx, query, description}` (app.py:4243-4248 cached / 4267-4272 fresh),
 * while every failure path emits `{error}` alone — `enrichment_disabled` (app.py:4225),
 * `Invalid submap/frame` (app.py:4236), `crop_failed` (app.py:4256), `enrich_failed`
 * (app.py:4265), or `str(e)` (app.py:4275).
 */
export interface DetectionDescriptionResult {
  submap_id?: number;
  frame_idx?: number;
  query?: string;
  description?: ObjectDescription;
  error?: string;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Debug detection: `debug_detect` → `debug_detect_results`
 *
 * The detection-debug page (`web/apps/webserver/src/detection-debug.ts`) and the summary
 * page's debug drawer (`web/apps/webserver/src/summary.ts`) run the PRODUCTION
 * CLIP → SAM → 3D-box → dedup pipeline and get rich per-frame diagnostics back. The types
 * below mirror the dict built by `StreamingSLAM.debug_detect_full`
 * (`server/server/streaming_slam.py:1200-1407`) and emitted by the `debug_detect` handler
 * (`server/server/app.py:4299-4324`).
 * ───────────────────────────────────────────────────────────────────────────── */

/**
 * Request payload for `debug_detect` (read at `server/server/app.py:4305-4308`).
 */
export interface DebugDetectRequest {
  /**
   * Free-text queries to run the full pipeline for. An empty list short-circuits to
   * `debug_detect_results` `{error: 'No queries provided'}` (server/server/app.py:4310-4312).
   * Read at server/server/app.py:4305.
   */
  queries: string[];
  /**
   * Per-query CLIP cosine threshold; the `default` key is the fallback for any query
   * without its own entry, and 0.2 is used when neither is present
   * (server/server/streaming_slam.py:1249). Read at server/server/app.py:4306.
   */
  clip_thresholds?: Record<string, number>;
  /**
   * Per-query SAM score threshold, same `default` fallback, 0.3 when absent
   * (server/server/streaming_slam.py:1250). Read at server/server/app.py:4307.
   */
  sam_thresholds?: Record<string, number>;
  /**
   * How many CLIP-ranked frames per query get a SAM pass. Omitted (or `<= 0`) falls back
   * to the server's `SAM_TOP_K_FRAMES` (server/server/streaming_slam.py:1215) — the summary
   * page omits it (`web/apps/webserver/src/summary.ts:664-668`), the debug page sends it
   * (`web/apps/webserver/src/detection-debug.ts:166-171`). Read at server/server/app.py:4308.
   */
  top_k?: number;
}

/**
 * One final (post-dedup) or pre-dedup debug detection.
 *
 * Built at `server/server/streaming_slam.py:1365-1374`; the post-dedup list is filtered by
 * `ObjectDetector.deduplicate_detections` (`core/vggt_slam/object_detector.py:158-186`).
 * Deliberately NOT `DetectionResult`: this path always populates the box/score fields and
 * adds the two per-stage scores, so the shape is stated exactly rather than inherited as
 * optional.
 */
export interface DebugDetection {
  /** Always `true` on this path — dedup drops anything else. streaming_slam.py:1366. */
  success: boolean;
  /** The query this detection answers. streaming_slam.py:1367. */
  query: string;
  /**
   * Scene-centered OBB from `ObjectDetector.compute_3d_bbox`
   * (`core/vggt_slam/object_detector.py:104-155`). Never null here — only masks with
   * `has_3d_box` reach this list. streaming_slam.py:1368.
   */
  bounding_box: BoundingBox3D;
  /** Combined CLIP × SAM score — the value dedup ranks on. streaming_slam.py:1369. */
  confidence: number;
  /** Submap the evidence keyframe belongs to. streaming_slam.py:1370. */
  matched_submap: number;
  /** Frame index within that submap. streaming_slam.py:1371. */
  matched_frame: number;
  /** CLIP cosine similarity for (frame, query). streaming_slam.py:1372. */
  clip_score: number;
  /** SAM segmentation score for the mask. streaming_slam.py:1373. */
  sam_score: number;
}

/**
 * One SAM mask proposal on one debug frame.
 *
 * Built at `server/server/streaming_slam.py:1323-1334`; `dedup_kept` is back-filled after
 * dedup at streaming_slam.py:1380-1384. Nullable fields are `null` (not absent) on the
 * wire — Python emits the key with a `None` value in every case.
 */
export interface DebugSamMaskDiag {
  /** Raw SAM segmentation score. streaming_slam.py:1324. */
  score: number;
  /** CLIP similarity of the parent frame for this query. streaming_slam.py:1325. */
  clip_score: number;
  /** `score * clip_score` — the ranking value. streaming_slam.py:1326. */
  combined_score: number;
  /** SAM 2D box `[x0, y0, x1, y1]` on the source keyframe. streaming_slam.py:1327. */
  box_2d: Box2D;
  /**
   * Base64 PNG mask overlay, or `null` when overlay rendering was skipped or threw
   * (streaming_slam.py:1316-1321). streaming_slam.py:1328.
   */
  mask_image: string | null;
  /** Whether `score` cleared `sam_threshold_used`. streaming_slam.py:1329. */
  above_sam_threshold: boolean;
  /** The SAM threshold actually applied to this query. streaming_slam.py:1330. */
  sam_threshold_used: number;
  /** Whether a 3D OBB could be lifted from this mask. streaming_slam.py:1331. */
  has_3d_box: boolean;
  /** The lifted OBB, or `null` when below threshold / not liftable. streaming_slam.py:1332. */
  bbox_3d: BoundingBox3D | null;
  /**
   * Dedup outcome: `true`/`false` once dedup ran, and `null` for masks that never entered
   * dedup (no 3D box). streaming_slam.py:1333 (init) / 1380-1384 (fill).
   */
  dedup_kept: boolean | null;
}

/**
 * Per-(submap, frame, query) diagnostics — one row of the debug page's grid.
 *
 * Built at `server/server/streaming_slam.py:1342-1359`. Present only when the server ran
 * with `include_frames=True`, which is the only way the socket handler calls it
 * (`server/server/app.py:4317-4320` → `debug_detect_full` default, streaming_slam.py:1200).
 */
export interface DebugFrameDiag {
  /** streaming_slam.py:1343. */
  submap_id: number;
  /** streaming_slam.py:1344. */
  frame_idx: number;
  /** streaming_slam.py:1345. */
  query: string;
  /** CLIP cosine similarity for this (frame, query). streaming_slam.py:1346. */
  clip_similarity: number;
  /** 1-based rank of this frame among all frames for the query. streaming_slam.py:1347. */
  clip_rank: number;
  /** Whether `clip_similarity` cleared the CLIP threshold. streaming_slam.py:1348. */
  above_threshold: boolean;
  /** Whether the frame made the top-K SAM shortlist. streaming_slam.py:1349. */
  in_top_k: boolean;
  /** Above threshold but outside top-K, so SAM never ran. streaming_slam.py:1350. */
  sam_skipped: boolean;
  /** streaming_slam.py:1351. */
  clip_threshold_used: number;
  /** streaming_slam.py:1352. */
  sam_threshold_used: number;
  /** The effective top-K after the `<= 0`/omitted fallback. streaming_slam.py:1353. */
  top_k_used: number;
  /** Base64 JPEG, 120px-tall thumbnail; `null` if it could not be produced. streaming_slam.py:1354. */
  thumbnail: string | null;
  /** `"{w}×{h}"` of the source keyframe; `null` alongside a null thumbnail. streaming_slam.py:1355. */
  resolution: string | null;
  /** SAM proposals for this frame — empty when SAM was skipped. streaming_slam.py:1356. */
  sam_masks: DebugSamMaskDiag[];
  /** Message from a SAM failure on this frame, else `null`. streaming_slam.py:1357 / 1338-1339. */
  sam_error: string | null;
  /** Raw detections from this frame before cross-frame dedup. streaming_slam.py:1358 / 1391-1393. */
  detections_before_dedup: DebugDetection[];
}

/**
 * Successful `debug_detect_results` payload — the full diagnostics dict returned by
 * `StreamingSLAM.debug_detect_full` (`server/server/streaming_slam.py:1396-1407`, and the
 * empty-map early return at streaming_slam.py:1224-1230, which is the same shape with empty
 * lists). Every field below is always present on a successful run.
 *
 * `error` is declared `never` so the union in {@link DebugDetectResults} narrows on a plain
 * `if (data.error)` check — the shape the two client pages already use.
 */
export interface DebugDetectResponse {
  /** Echo of the requested queries. streaming_slam.py:1397. */
  queries: string[];
  /** Echo of the applied CLIP thresholds (`{}` when none were sent). streaming_slam.py:1398. */
  clip_thresholds: Record<string, number>;
  /** Echo of the applied SAM thresholds (`{}` when none were sent). streaming_slam.py:1399. */
  sam_thresholds: Record<string, number>;
  /** The effective top-K after the server-side fallback. streaming_slam.py:1400. */
  top_k: number;
  /** Per-frame diagnostics; `[]` when the map has no submaps. streaming_slam.py:1401. */
  frames: DebugFrameDiag[];
  /** Detection count before dedup. streaming_slam.py:1402. */
  raw_detection_count: number;
  /** Detection count after dedup — i.e. `detections.length`. streaming_slam.py:1403. */
  deduped_detection_count: number;
  /** The final, deduped detections. streaming_slam.py:1404. */
  detections: DebugDetection[];
  /** Total (frame, query) combos scanned. streaming_slam.py:1405. */
  total_frames_scanned: number;
  /** Server-side wall time for the whole pipeline. streaming_slam.py:1406. */
  query_time_ms: number;
  /** Never set on a successful run — see {@link DebugDetectError}. */
  error?: never;
}

/**
 * Failed `debug_detect_results` payload. The handler emits `{error}` and NOTHING else:
 * `'SLAM not initialized'` (`server/server/app.py:4302`), `'No queries provided'`
 * (app.py:4311), or `str(e)` from the pipeline (app.py:4324).
 */
export interface DebugDetectError {
  error: string;
}

/** What actually arrives on `debug_detect_results` — success diagnostics or an error-only body. */
export type DebugDetectResults = DebugDetectResponse | DebugDetectError;

export interface AgentThought {
  id: string;
  timestamp: number;
  type: 'observation' | 'thinking' | 'chat_response' | 'error' | 'action';
  content: string;
  keyframe_b64?: string;
  subagent?: string;
  phase?: string;
  confidence?: number;
  attachments?: AgentAttachment[];
}

export interface AgentAction {
  id: string;
  timestamp: number;
  action: string;
  details: string;
  mission_id?: number;
  queries?: string[];
}

export interface AgentFinding {
  id: string;
  timestamp: number;
  query: string;
  description: string;
  confidence: number;
  position?: number[];
  mission_id?: number;
  evidence?: Record<string, unknown>;
}

export interface MissionState {
  id: number;
  category: string;
  goal: string;
  queries: string[];
  found: string[];
  status: 'active' | 'recovering' | 'completed' | 'stalled';
  confidence: number;
  findings_count: number;
  last_progress_ts?: number;
  stall_reason?: string | null;
  stall_count?: number;
}

export interface AgentTaskState {
  id: string;
  timestamp: number;
  task_type: 'orchestrator' | 'subagent' | 'mission' | 'tool_batch';
  name: string;
  status: 'started' | 'heartbeat' | 'succeeded' | 'failed' | 'timed_out' | 'stalled' | 'resumed';
  mission_id?: number;
}

export interface AgentTaskEvent extends AgentTaskState {
  latency_ms?: number;
  details?: string;
  error?: string;
}

export interface AgentState {
  enabled: boolean;
  scene_description: string;
  room_type: string;
  missions: MissionState[];
  active_queries: string[];
  discovered_objects: string[];
  current_goal: string | null;
  submaps_processed: number;
  coverage_estimate: number;
  health?: 'ok' | 'degraded' | 'disabled';
  degraded_mode?: boolean;
  runtime_v2_enabled?: boolean;
  active_tasks?: AgentTaskState[];
  pending_jobs?: AgentJobEvent[];
  running_jobs?: AgentJobEvent[];
  last_job_errors?: string[];
  orchestrator_busy?: boolean;
}

export interface AgentAttachment {
  kind: 'keyframe' | 'mask' | 'other';
  image_b64?: string;
  label?: string;
}

export interface AgentUICommand {
  id: string;
  name:
    | 'focus_detection'
    | 'set_detection_queries'
    | 'add_detection_query'
    | 'remove_detection_query'
    | 'show_waypoint'
    | 'show_path'
    | 'show_toast'
    | 'open_detection_preview'
    // Demo-surface commands (2026-07 demo build; additive — the frozen live-agent's Python
    // Literal union is NEVER edited to match; only the demo's persisted-scene agent emits
    // these). Typed args shapes live in ./oreos (OreosUICommandArgs).
    | 'drive_path'
    | 'recolor_region'
    | 'open_export'
    | 'focus_object'
    | 'show_measurement'
    | 'show_dimension_labels';
  args: Record<string, unknown>;
  mission_id?: number;
  ttl_ms?: number;
}

export interface AgentUIResult {
  id: string;
  status: 'ok' | 'error' | 'ignored' | 'timeout';
  result?: Record<string, unknown>;
  error?: string;
}

export interface AgentToolEvent {
  id: string;
  tool: string;
  status: 'started' | 'succeeded' | 'failed';
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  latency_ms?: number;
}

export interface AgentJobEvent {
  id?: string;
  job_id: string;
  job_name: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'timed_out' | 'canceled' | 'not_found';
  mission_id?: number;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  created_at?: number;
  started_at?: number;
  finished_at?: number;
}

export interface PlanObject {
  name: string;
  confidence: number;
}

export interface PlanResponse {
  objects: PlanObject[];
  justification: string;
  waypoints: number[][];
}

export interface SceneEvidenceRef {
  submap_id: number;
  frame_idx: number;
  /**
   * Resolved global keyframe index, when the server included it (grounding/objects.jsonl).
   *
   * Populated ONLY by the dataset-export grounding path (`server/server/export/grounding.py`);
   * it is never present on live `scene_report_update` / `scene_report_ready` payloads. Live
   * consumers must key evidence off `submap_id` + `frame_idx` and treat this as absent.
   */
  global_frame_index?: number;
  /** Real SAM 2D box on this evidence keyframe, when the server persisted one. */
  box_2d?: Box2D;
  /** Real SAM mask on this evidence keyframe, when the server persisted one. */
  mask_rle?: MaskRLE;
}

export interface SceneMetrics {
  dimensions: number[];
  bbox_min: number[];
  bbox_max: number[];
  trajectory_length: number;
  num_submaps: number;
  num_keyframes: number;
  point_count: number;
  vertical_axis_known: boolean;
  // --- ground frame (server/demo/ground_frame.py). Additive; absent on older records.
  // Present ONLY when a plane fit succeeded well enough to publish: a weak fit records
  // `vertical_axis_known: false` + `derivation` + `ground_frame_note` and NO heights, so
  // there is never a plausible-looking number sitting next to a false flag. Heights are
  // signed distances along `up_axis` in the world frame, in the scene's own units —
  // metres only via the metric anchor, like every other length here.
  /** Unit world-frame vertical, derived from the fitted floor normal. */
  up_axis?: number[];
  floor_height?: number;
  ceiling_height?: number;
  /** `ceiling_height - floor_height` — the floor-to-ceiling height. */
  room_height?: number;
  /** `[width, depth]` of the floor footprint in the floor plane's own basis. */
  floor_extent?: number[];
  /** Occupancy-based floor area — NOT `width * depth` (an L-shaped room is not a box). */
  floor_area?: number;
  /** How the frame was derived, e.g. `"plane-fit"`. */
  derivation?: string;
  /** Why a weak or partial fit did not publish numbers. */
  ground_frame_note?: string;
}

/**
 * Provenance envelope for an object an IMPORTED splat got from GEOMETRY.
 *
 * An imported splat has no capture video, so it has no detections. Its objects come from
 * clustering the gaussian centres once the floor, walls and ceiling are removed
 * (`server/demo/imported_objects.py`). That is not the same act as detection: nothing ran
 * a detector, nothing read a photograph, and the parent's `confidence` is 0.0 rather than
 * an invented score. This block is how the UI says so.
 *
 * `obb` is the tight gravity-aware box; the parent's `center`/`extent` stay the world AABB
 * so every consumer that predates this field keeps working unchanged.
 *
 * Additive — absent on every captured detection.
 */
export interface ImportedDetection {
  method: 'geometry_cluster';
  /** Human-readable, drives the panel's provenance chip. */
  provenance: string;
  /** `geometry` = measured size only; `synthetic_view` = a VLM named it from a RENDER. */
  label_source: 'geometry' | 'synthetic_view';
  /** The measured-size label, kept even after a view label replaces the display name. */
  geometry_label: string;
  obb?: ObjectLayerOBB;
  /** Facts read off the fitted planes, e.g. "on the floor", "against a wall". */
  placement: string[];
  support_points: number;
  support_voxels: number;
  /** Points per occupied voxel — how solid this cluster is next to the rest of the scene. */
  density?: number;
  units: OreosUnits;
  units_basis: string;
  up_source: string;
  caveats: string[];
  label_model?: string;
  label_confidence?: number;
  view_id?: string;
}

export interface SceneObjectInstance {
  query: string;
  center: number[];
  extent: number[];
  confidence: number;
  evidence: SceneEvidenceRef[];
  /** Present only on geometry-derived objects of an imported splat. See above. */
  imported_detection?: ImportedDetection;
  description?: ObjectDescription;  // fine-grained identification (top-N enriched at finalize)
  // Human review edits from the product-workflow Detect stage — persisted via
  // PATCH /api/scenes/<id>/objects/<object_id> (see rest.ts). Both are additive + optional:
  // older records / unedited objects simply omit them.
  /** Operator relabel — recorded ALONGSIDE `query` (the model's original detection), never
   *  replacing it. Undefined until a human relabels. */
  human_label?: string;
  /** Operator dismissed this detection in the Detect review. Undefined/false until dismissed;
   *  the object is kept in the inventory (annotation, not deletion), just filtered from view. */
  dismissed?: boolean;
}

export interface SceneSpatialRelation {
  a: string;
  b: string;
  distance: number;
  relation: string;
}

export interface SceneFacts {
  metrics: SceneMetrics;
  objects: SceneObjectInstance[];
  object_counts: Record<string, number>;
  relations: SceneSpatialRelation[];
  coverage_estimate: number;
  scene_center: number[];
  units_note: string;
}

export interface SceneReportObject {
  name: string;
  location: string;
  evidence: SceneEvidenceRef[];
}

export interface SceneReport {
  summary: string;
  room_type: string;
  objects: SceneReportObject[];
  observations: string[];
  coverage_note: string;
  facts: SceneFacts;
  model: string;
  degraded: boolean;
  progressive?: boolean;
  final?: boolean;
  generated_at: number;
  error?: string;
}

export interface SceneKeyframeRef {
  submap_id: number;
  frame_idx: number;
  blob_key: string;
}

/**
 * One SYNTHETIC VIEW: a render of the scene's own gaussian splat, registered as evidence
 * so the keyframe-shaped pipelines (VLM annotation, SAM prompts, mask back-projection)
 * work on an imported splat that has no capture video.
 *
 * It is a SIBLING of `SceneKeyframeRef`, never a substitute — the two live in different
 * record fields and must stay distinguishable forever. A claim grounded on one of these
 * carries the `synthetic view` provenance chip, never `photo`/`keyframe`.
 * See `server/demo/synthetic_views.py` and IMPORTED-SPLAT-CONTRACT.md.
 */
export interface SceneSyntheticViewRef {
  view_id: string;
  /** Capture order within the scene's view set. */
  index: number;
  /** World-frame camera position. */
  position: number[];
  /** World-frame camera orientation, three.js convention (+x right, +y up, −z forward). */
  quaternion: number[];
  fov_y_deg: number;
  width: number;
  height: number;
  /** 4x4 OpenCV camera-to-world (the `trajectory.npz` convention), derived server-side. */
  c2w?: number[][];
  /** `[fx, fy, cx, cy]` for this view's pixel grid. */
  intrinsics?: number[];
  created_at?: number;
  /** Fixed copy: `"synthetic view"`. */
  provenance?: string;
  /** Fixed copy: `"rendered from the imported splat"`. */
  provenance_detail?: string;
  label?: string | null;
}

export interface SceneListItem {
  scan_id: string;
  created_at: number;
  room_type: string;
  summary: string;
  object_count: number;
  point_count: number;
  thumbnail: SceneKeyframeRef | null;
  /** Whether a 3DGS splat.ply artifact is stored for this scan (embed delivery). */
  has_splat?: boolean;
  /** Optional grouping tags (e.g. a pilot's client/project). */
  client?: string | null;
  project?: string | null;
}

/**
 * How a persisted scene came to exist (2026-07 demo build; additive `source` kwarg on the
 * server's `save_scene`). Absent on older records — readers must treat undefined as
 * `"recon_video"` (the only path that existed before).
 */
export type PersistedSceneSource =
  | 'recon_video'
  | 'imported_splat'
  /** Orbbec Gemini 2 specifically. */
  | 'recon_gemini2'
  /** Any other depth-sensing camera (Kinect, Xtion, RealSense, a public RGB-D dataset).
   *  Same RGB-only pipeline as `recon_gemini2` — the distinction exists because naming a
   *  device we did not record on would be a false provenance claim. */
  | 'recon_depthcam'
  /** A robot recording (DimOS memory2 .db) through the merged Oreos recordings lane:
   *  camera-only recon on the oreos-recon app + Sim(3) consistency vs the robot's own
   *  odometry + QC gate; scene persists degraded (no detections, no splat, preview cloud,
   *  zero intrinsics — all declared in the report). */
  | 'robot_recording';

export interface PersistedScene {
  scan_id: string;
  created_at: number;
  report: SceneReport;
  facts: SceneFacts;
  keyframes: SceneKeyframeRef[];
  points_key: string | null;
  point_count: number;
  scene_center: number[];
  /** Ingest provenance (demo build). Additive — absent means `"recon_video"`. */
  source?: PersistedSceneSource;
  /** Whether a 3DGS splat.ply artifact is available at SCENE_SPLAT_PLY (embed delivery). */
  has_splat?: boolean;
  /** The scan's "latest calibrated" derived artifact (set by the Refine anchor/clamp routes),
   *  or null/absent until one runs. Lets the Refine/Export stages offer + fetch the
   *  calibrated/clamped result. See ./rest DerivedLatestPointer. Additive — safe to omit. */
  derived_latest?: DerivedLatestPointer | null;
  /**
   * Optional AI-completed object layer (generated 3D assets placed in the scan). Present only
   * when the server bundles one into the embed export; the viewer renders it behind a feature
   * flag. See ./objectLayer and the object-layer embed spec. Additive — safe to omit.
   */
  object_layer?: ObjectLayerManifest;
  /**
   * Renders of this scene's own splat, registered as evidence (see SceneSyntheticViewRef).
   * A SEPARATE field from `keyframes` on purpose and forever. Additive — absent means the
   * scene has none, which for a captured scan is the normal state.
   */
  synthetic_views?: SceneSyntheticViewRef[];
  /** Optional grouping tags (e.g. a pilot's client="efficura", project="chorus"). */
  client?: string | null;
  project?: string | null;
}

export interface SceneQAResponse {
  answer: string;
  model: string;
  degraded: boolean;
  focus: { name: string; center: number[] } | null;
  evidence: SceneKeyframeRef[];
}
