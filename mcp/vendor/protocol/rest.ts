import type { SceneObjectInstance } from './types';
import { API_KEY_PREFIX } from './auth';

export const REST_PATHS = {
  HEALTH: '/health',
  SESSION: '/session',
  SESSION_STATUS: '/session/status',
  AUTH_SESSION: '/auth/session',
  AUTH_QR_TOKEN: '/auth/qr-token',
  SESSION_REFRESH: '/api/session/refresh',
  // Durable API keys (`ork_...` bearers). See the "API keys" section below.
  API_KEYS: '/api/keys',
  API_KEY: (keyId: string) => `/api/keys/${encodeURIComponent(keyId)}`,
  SCANS_CONSUME: '/api/scans/consume',
  ASSISTANT_CHAT: '/api/assistant/chat',
  PLAN: '/api/plan',
  RESET: '/reset',
  DEMO_VIDEOS: '/api/demo/videos',
  DEMO_START: '/api/demo/start',
  DEMO_STOP: '/api/demo/stop',
  DEMO_STATUS: '/api/demo/status',
  DEMO_VIDEO: '/api/demo/video',
  SCENES: '/api/scenes',
  SCENE_QA: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/qa`,
  SCENE_DETAIL: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}`,
  SCENE_POINTS: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/points`,
  SCENE_CLOUD_PLY: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/cloud.ply`,
  SCENE_KEYFRAME: (scanId: string, blobKey: string) =>
    `/api/scenes/${encodeURIComponent(scanId)}/keyframes/${encodeURIComponent(blobKey)}`,
  // 3DGS splat artifact (server mirrors core's --export_splat output).
  SCENE_SPLAT_PLY: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/splat.ply`,
  // Mint a read-only, single-scan share token (owner-authed). See SHARE types below.
  SCENE_SHARE: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/share`,
  // Owner-only read of what every minted share link was used for (opens, visits, returns,
  // Q&A questions), per link. A share token cannot reach it. See SHARE ACCESS types below.
  SCENE_SHARE_ACCESS: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/share/access`,
  // Metric-anchor calibration (owner-authed, non-destructive). See ANCHOR types below.
  SCENE_ANCHOR: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/anchor`,
  // Auto-clamp 3DGS scale outliers (owner-authed, non-destructive). See CLAMP types below.
  SCENE_CLAMP: (scanId: string) => `/api/scenes/${encodeURIComponent(scanId)}/clamp`,
  // Persist a per-object review edit (relabel/dismiss) from the Detect stage (owner-authed,
  // non-destructive). `objectId` is the 0-based index into the scan's `facts.objects`. See
  // OBJECT EDIT types below.
  SCENE_OBJECT: (scanId: string, objectId: number) =>
    `/api/scenes/${encodeURIComponent(scanId)}/objects/${objectId}`,
  // Read-only stream of a NON-DESTRUCTIVE derived artifact (a calibrated/clamped cloud or splat
  // the anchor/clamp routes produced), so the workflow viewer can render a true before/after.
  // `derivedKey` is a `derived/...` key returned by anchor/clamp; a leading `derived/` is stripped
  // (the route namespace already IS `/derived/`). Owner-authed; header or ?auth_token= (the latter
  // so the Gaussian-splat URL loader, which can't set headers, can fetch a clamped splat).
  SCENE_DERIVED: (scanId: string, derivedKey: string) => {
    const rel = derivedKey.replace(/^derived\//, '');
    const encoded = rel.split('/').map(encodeURIComponent).join('/');
    return `/api/scenes/${encodeURIComponent(scanId)}/derived/${encoded}`;
  },
} as const;

// ── Refine: metric anchor + auto-clamp ──
// Both are real, deployed, owner-authed `server/app.py` routes (product-workflow Refine
// stage). Both are NON-DESTRUCTIVE: they never touch a scan's original `cloud.npz` /
// `splat.ply` / `trajectory.npz` — every output is a NEW `derived/...` artifact key. The
// derived artifact can now be fetched back (REST_PATHS.SCENE_DERIVED) to re-render a true
// before/after in the viewer, and exported (SCENE_EXPORT_PATH's `source` selector). On success
// each route also persists a "latest calibrated" pointer (DerivedLatestPointer below) onto the
// scene record — surfaced on GET /api/scenes/<id> as `derived_latest`.

/** Which Refine action produced a derived artifact group. */
export type DerivedArtifactKind = 'anchor' | 'clamp';

/** The scan's "latest calibrated" derived-artifact pointer, set by the anchor/clamp routes and
 *  returned on `PersistedScene.derived_latest`. A group's per-type keys are `null` when that
 *  action didn't produce that type (a clamp group carries only `splat_key`). `source_key` is the
 *  representative key to hand to `SCENE_EXPORT_PATH`'s `source` selector / `SCENE_DERIVED` — the
 *  export reads the whole group from it. Metadata-only + non-destructive on the server side. */
export interface DerivedLatestPointer {
  kind: DerivedArtifactKind;
  cloud_key: string | null;
  trajectory_key: string | null;
  splat_key: string | null;
  source_key: string;
  /** ISO-8601 UTC timestamp of the Refine action. */
  applied_at: string;
  /** Metres-per-SLAM-unit, carried on an `anchor` pointer only (else absent/null). Provenance for
   *  the Isaac USD export's metric gate — the anchor's calibrated geometry is already metric, so
   *  this is displayed, not re-applied. Additive/optional; the guard doesn't require it. */
  scale_factor?: number | null;
}

/** Hand-written guard (never throws) — validates a `DerivedLatestPointer` shape defensively,
 *  since it's parsed from the live scene-detail response. */
export function isDerivedLatestPointer(value: unknown): value is DerivedLatestPointer {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  const nullableKey = (v: unknown): boolean => v === null || typeof v === 'string';
  return (
    (o.kind === 'anchor' || o.kind === 'clamp') &&
    typeof o.source_key === 'string' &&
    o.source_key.length > 0 &&
    nullableKey(o.cloud_key) &&
    nullableKey(o.trajectory_key) &&
    nullableKey(o.splat_key)
  );
}

/** Generic error body shape shared by the anchor + clamp routes (and most `/api/scenes/*`
 *  routes): `{error: "<code>", message?: "<human readable>"}`. */
export interface ApiErrorResponse {
  error: string;
  message?: string;
}

export function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return typeof o.error === 'string' && (o.message === undefined || typeof o.message === 'string');
}

/** Request body for POST /api/scenes/<id>/anchor. Two points already picked in the scene's
 *  current (up-to-scale) world frame, plus the real-world distance between them, in metres. */
export interface AnchorSceneRequest {
  point_a: [number, number, number];
  point_b: [number, number, number];
  distance_m: number;
  /** `'job'` → the route computes + persists the scale instantly (202) and materializes
   *  the calibrated `derived/anchor/*` copies on a background job (`AnchorSceneJobResponse`).
   *  Omitted/anything else → the historical synchronous form (`AnchorSceneResponse`). */
  materialize?: 'job' | 'sync';
}

/** Response from POST /api/scenes/<id>/anchor. `scale_factor` is metres-per-SLAM-unit
 *  (`distance_m / measured_distance`); the `calibrated_*_key` fields are NEW `derived/anchor/
 *  <stamp>/...` artifacts (trajectory/splat keys are `null` when the scan has none to
 *  calibrate — the cloud key is always present). */
export interface AnchorSceneResponse {
  scale_factor: number;
  measured_distance: number;
  distance_m: number;
  /** ISO-8601 UTC timestamp. */
  applied_at: string;
  calibrated_cloud_key: string;
  calibrated_trajectory_key: string | null;
  calibrated_splat_key: string | null;
  /** The exact operator-picked segment, before calibration (SLAM units). */
  gauge_span_before: number;
  /** The same segment, after calibration — equal to `distance_m` by construction (metres). */
  gauge_span_after_m: number;
  /** Whole-cloud bounding-box diagonal, before calibration (SLAM units). */
  cloud_extent_before: number;
  /** Whole-cloud bounding-box diagonal, after calibration (metres). */
  cloud_extent_after_m: number;
}

/** Response from POST /api/scenes/<id>/anchor with `materialize:'job'` (HTTP 202): the
 *  scale is applied — `derived_latest` already carries `kind:'anchor'` + `scale_factor`,
 *  so metres are live — while the calibrated `derived/anchor/<stamp>/...` copies build on
 *  a background job. Poll `WORKSPACE_JOB(job_id)`; the pointer gains its keys (and the
 *  Isaac export gate opens) when the job finishes. No extent fields: computing them needs
 *  the full cloud, which is exactly the work this form defers. */
export interface AnchorSceneJobResponse {
  scale_factor: number;
  measured_distance: number;
  distance_m: number;
  /** ISO-8601 UTC timestamp. */
  applied_at: string;
  pending: true;
  job_id: string;
  /** The exact operator-picked segment, before calibration (SLAM units). */
  gauge_span_before: number;
  /** The same segment, after calibration — equal to `distance_m` by construction (metres). */
  gauge_span_after_m: number;
}

/** Hand-written guard (never throws) for the async anchor response. */
export function isAnchorSceneJobResponse(value: unknown): value is AnchorSceneJobResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    o.pending === true &&
    typeof o.job_id === 'string' &&
    typeof o.scale_factor === 'number' &&
    typeof o.measured_distance === 'number' &&
    typeof o.distance_m === 'number' &&
    typeof o.applied_at === 'string' &&
    typeof o.gauge_span_before === 'number' &&
    typeof o.gauge_span_after_m === 'number'
  );
}

/** Hand-written guard (never throws) — validates an `AnchorSceneResponse` shape defensively,
 *  since it's parsed from a live HTTP response. */
export function isAnchorSceneResponse(value: unknown): value is AnchorSceneResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.scale_factor === 'number' &&
    typeof o.measured_distance === 'number' &&
    typeof o.distance_m === 'number' &&
    typeof o.applied_at === 'string' &&
    typeof o.calibrated_cloud_key === 'string' &&
    typeof o.gauge_span_before === 'number' &&
    typeof o.gauge_span_after_m === 'number'
  );
}

/** Request body for POST /api/scenes/<id>/clamp. `percentile` is optional; the server default
 *  (`DEFAULT_SCALE_CLAMP_PERCENTILE`) is 99.0 when omitted. */
export interface ClampSceneRequest {
  percentile?: number;
}

/** Response from POST /api/scenes/<id>/clamp. `cloud_key` and `derived_splat_key` are the SAME
 *  value (the derived splat's key, under `derived/clamp/<stamp>/splat.ply`) — `cloud_key` is
 *  named to match the platform-agreed contract; despite the name it does not hold a point
 *  cloud. */
export interface ClampSceneResponse {
  clamped_gaussian_count: number;
  scale_clamp_percentile: number;
  cloud_key: string;
  derived_splat_key: string;
  max_scale_before: number;
  max_scale_after: number;
  gaussian_count: number;
}

/** Hand-written guard (never throws) — validates a `ClampSceneResponse` shape defensively,
 *  since it's parsed from a live HTTP response. */
export function isClampSceneResponse(value: unknown): value is ClampSceneResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.clamped_gaussian_count === 'number' &&
    typeof o.scale_clamp_percentile === 'number' &&
    typeof o.cloud_key === 'string' &&
    typeof o.derived_splat_key === 'string' &&
    typeof o.max_scale_before === 'number' &&
    typeof o.max_scale_after === 'number' &&
    typeof o.gaussian_count === 'number'
  );
}

// ── Detect: per-object review edits (relabel / dismiss) ──
// PATCH /api/scenes/<id>/objects/<object_id> is a real, owner-authed `server/app.py` route
// (product-workflow Detect stage). `object_id` is the 0-based index into the scan's
// `facts.objects` inventory. NON-DESTRUCTIVE: a relabel records the operator's `label` in the
// object's `human_label` field ALONGSIDE the model's original `query` (never overwritten);
// `dismissed: true` hides the instance from the reviewed inventory without deleting it.

/** Request body for PATCH /api/scenes/<id>/objects/<object_id>. At least one field must be
 *  present; the two are independent (a dismiss never clobbers a prior relabel, or vice versa). */
export interface ObjectEditRequest {
  /** Operator relabel — stored alongside the model's original `query`. */
  label?: string;
  /** Hide this detection from the reviewed inventory (kept, not deleted). */
  dismissed?: boolean;
}

/** Response from PATCH /api/scenes/<id>/objects/<object_id> — the updated object (carrying the
 *  original `query` plus the persisted `human_label` / `dismissed`). */
export interface ObjectEditResponse {
  scan_id: string;
  object_id: number;
  object: SceneObjectInstance;
}

/** Hand-written guard (never throws) — validates an `ObjectEditResponse` shape defensively,
 *  since it's parsed from a live HTTP response. */
export function isObjectEditResponse(value: unknown): value is ObjectEditResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.scan_id === 'string' &&
    typeof o.object_id === 'number' &&
    !!o.object &&
    typeof o.object === 'object' &&
    typeof (o.object as Record<string, unknown>).query === 'string'
  );
}

// ── API keys: durable `ork_...` bearers ──
// POST /api/keys mints a new key from a Clerk JWT or broker session token bearer (never
// from another API key — the server 403s that attempt as `api_key_cannot_mint`, an
// ApiErrorResponse). GET lists every key on the account (revoked included, newest
// first). DELETE revokes one; idempotent (revoking twice is a no-op success), and 404s
// an unknown or cross-user key id. The minted secret is returned exactly once, from the
// POST response's `key` field — afterward only its `prefix` (first 10 chars) is visible.

/** One key's account-facing summary — never carries the secret. */
export interface ApiKeySummary {
  key_id: string;
  name: string;
  prefix: string;
  /** Epoch seconds. */
  created_at: number;
  /** Epoch seconds, or `null` if this key has never authenticated a request. */
  last_used_at: number | null;
  /** Epoch seconds, or `null` while the key is still active. */
  revoked_at: number | null;
}

/** Request body for POST /api/keys. */
export interface MintApiKeyRequest {
  /** Human label, e.g. `claude-mcp@<hostname>`. Server caps this at 64 chars. */
  name?: string;
}

/** Response from POST /api/keys. `key` is the full `ork_...` secret — shown only this
 *  once; store it immediately, the server never returns it again. */
export interface MintApiKeyResponse {
  key: string;
  key_id: string;
  name: string;
  prefix: string;
  /** Epoch seconds. */
  created_at: number;
}

/** Response from GET /api/keys. */
export interface ListApiKeysResponse {
  keys: ApiKeySummary[];
}

/** Response from DELETE /api/keys/<key_id>. */
export interface RevokeApiKeyResponse {
  key_id: string;
  /** Epoch seconds. */
  revoked_at: number;
}

/** Hand-written guard (never throws) — validates an `ApiKeySummary` shape defensively,
 *  since it's parsed from a live HTTP response. */
export function isApiKeySummary(value: unknown): value is ApiKeySummary {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  const nullableNumber = (v: unknown): boolean => v === null || typeof v === 'number';
  return (
    typeof o.key_id === 'string' &&
    typeof o.name === 'string' &&
    typeof o.prefix === 'string' &&
    typeof o.created_at === 'number' &&
    nullableNumber(o.last_used_at) &&
    nullableNumber(o.revoked_at)
  );
}

/** Hand-written guard (never throws) — validates a `MintApiKeyResponse` shape defensively,
 *  including that `key` actually carries the durable-key bearer prefix. */
export function isMintApiKeyResponse(value: unknown): value is MintApiKeyResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.key === 'string' &&
    o.key.startsWith(API_KEY_PREFIX) &&
    typeof o.key_id === 'string' &&
    typeof o.name === 'string' &&
    typeof o.prefix === 'string' &&
    typeof o.created_at === 'number'
  );
}

/** Hand-written guard (never throws) — validates a `ListApiKeysResponse` shape defensively,
 *  checking every entry with `isApiKeySummary`. */
export function isListApiKeysResponse(value: unknown): value is ListApiKeysResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return Array.isArray(o.keys) && o.keys.every(isApiKeySummary);
}

/** Hand-written guard (never throws) — validates a `RevokeApiKeyResponse` shape defensively. */
export function isRevokeApiKeyResponse(value: unknown): value is RevokeApiKeyResponse {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return typeof o.key_id === 'string' && typeof o.revoked_at === 'number';
}

// Embed delivery — a finished scan living in a third party's dashboard (the
// Efficura/Labrador pilot) as an <iframe>, authorized by a read-only share token (no
// Clerk login). This is the source of truth the server's Python side mirrors. See
// platform/contracts/embed-delivery.md.
export const SHARE_QUERY_PARAM = 'share_token' as const;
export const SHARE_TOKEN_SCOPE = 'scene:read' as const;

/** Build the embed page URL for a scan + share token. The token rides the URL **hash**
 *  (not the query) so it isn't sent on the document request / logged. */
export function buildEmbedUrl(origin: string, scanId: string, shareToken: string): string {
  return `${origin.replace(/\/$/, '')}/embed.html#scan_id=${encodeURIComponent(scanId)}&${SHARE_QUERY_PARAM}=${encodeURIComponent(shareToken)}`;
}

/** Request body for POST /api/scenes/<id>/share (owner-authed mint). */
export interface ShareSceneRequest {
  /** Token lifetime in seconds; server default is 30 days when omitted. */
  ttl_seconds?: number;
}

/** Response from POST /api/scenes/<id>/share. */
export interface ShareSceneResponse {
  share_token: string;
  /** `<broker-origin>/embed.html#scan_id=<id>&share_token=<jwt>` — drop into an <iframe>. */
  embed_url: string;
  scan_id: string;
  /** Epoch seconds at which the token expires. */
  expires_at: number;
  /**
   * `sha256(token)[:12]` — unique per minted link, reveals nothing about the token. Record it
   * next to the person the link is sent to; GET …/share/access reports per `access_id`.
   * Absent on brokers that predate the access log.
   */
  access_id?: string;
}

// ── Share-link access log (measurement) ──
// Every share-authorised request is recorded server-side under the scene's derived tree,
// keyed by the link's access_id. The owner reads the fold via GET /api/scenes/<id>/share/access.
// Definitions (server/share_access.py): a new VISIT starts after 30 min without requests;
// RETURNED = a visit that began >= 24 h after first_seen.

export interface ShareAccessQuestion {
  ts: number;
  question: string;
}

export interface ShareAccessProspect {
  access_id: string;
  /** Token iat/exp (epoch seconds) as carried on the events, when known. */
  iat: number | null;
  exp: number | null;
  opened: true;
  requests: number;
  visits: number;
  first_seen: number;
  last_seen: number;
  returned: boolean;
  returned_at: number | null;
  /** Distinct (hashed ip, hashed ua) pairs seen on this link. */
  devices: number;
  questions: ShareAccessQuestion[];
  /** Request count per server endpoint name. */
  endpoints: Record<string, number>;
  visit_starts: number[];
}

export interface ShareAccessEvent {
  ts: number;
  access_id: string;
  iat: number | null;
  exp: number | null;
  endpoint: string;
  method: string;
  ip: string | null;
  ua: string | null;
  question?: string;
}

/** Response from GET /api/scenes/<id>/share/access (owner-only). */
export interface ShareAccessResponse {
  scan_id: string;
  generated_at: number;
  definitions: Record<string, string>;
  prospects: ShareAccessProspect[];
  event_count: number;
  /** Present only with `?events=1`. */
  events?: ShareAccessEvent[];
}

// ── Building / project-level delivery ──
// Operator mints a project token via POST /api/projects/share (Clerk-authed). The
// resulting building_embed_url is dropped into the client's dashboard. It hosts
// embed.html in an inner <iframe> and lets the viewer flip between all scenes in the
// project (each with its own per-scene share token). This is the source of truth the
// server's Python side mirrors. See platform/contracts/embed-delivery.md.

/** POST /api/projects/share — mint a project-level building embed token (Clerk-authed). */
export const PROJECTS_SHARE = '/api/projects/share' as const;

/** Build the API path for GET /api/projects/manifest (authed by project Bearer token). */
export const PROJECTS_MANIFEST = (client: string, project: string): string =>
  `/api/projects/manifest?client=${encodeURIComponent(client)}&project=${encodeURIComponent(project)}`;

/** The hash parameter name that carries the project token in a building embed URL.
 *  It rides the fragment so it isn't sent on the document request / logged. */
export const PROJECT_TOKEN_PARAM = 'token' as const;

/** Build the building embed page URL for a client + project + project token.
 *  The token rides the URL **hash** (not the query) so it is not sent on the
 *  document request / logged. */
export function buildBuildingUrl(
  origin: string,
  client: string,
  project: string,
  projectToken: string,
): string {
  return (
    `${origin.replace(/\/$/, '')}/building.html` +
    `#client=${encodeURIComponent(client)}` +
    `&project=${encodeURIComponent(project)}` +
    `&token=${encodeURIComponent(projectToken)}`
  );
}

/** Request body for POST /api/projects/share (Clerk-authed mint). */
export interface ShareProjectRequest {
  client: string;
  project: string;
  /** Token lifetime in seconds; server default is 30 days when omitted. */
  ttl_seconds?: number;
}

/** Response from POST /api/projects/share. */
export interface ShareProjectResponse {
  project_token: string;
  client: string;
  project: string;
  /** Unix epoch seconds at which the token expires. */
  expires_at: number;
  /** `<origin>/building.html#client=<c>&project=<p>&token=<project_token>` — drop into an <iframe>. */
  building_embed_url: string;
}

/** One scene entry in a BuildingManifest. Carries its own per-scene share token + embed URL. */
export interface BuildingManifestScene {
  scan_id: string;
  label: string;
  room_type: string;
  /** Unix epoch seconds when this scan was created. */
  created_at: number;
  has_splat: boolean;
  share_token: string;
  /** Ready-to-use embed.html URL for this scene (= embed.html#scan_id=<id>&share_token=<tok>). */
  embed_url: string;
}

/** Building manifest returned by GET /api/projects/manifest — all scenes for a client/project. */
export interface BuildingManifest {
  client: string;
  project: string;
  scenes: BuildingManifestScene[];
}

export const CONNECT_ERROR_CODES = {
  SESSION_BUSY: 'session_busy',
  AT_CAPACITY: 'at_capacity',
  NO_SCANS_REMAINING: 'no_scans_remaining',
} as const;

export type ConnectErrorCode = typeof CONNECT_ERROR_CODES[keyof typeof CONNECT_ERROR_CODES];

export interface ModalGpuSession {
  session_id: string;
  url: string | null;
  status: string;
  expires_at?: number;
}

export interface HealthResponse {
  status: string;
  gpu?: boolean;
  gpu_name?: string;
}

export interface ScanConsumeResult {
  remaining: number | null;
  unlimited?: boolean;
}
