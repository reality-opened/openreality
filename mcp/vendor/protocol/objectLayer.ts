/**
 * Object Layer — the AI-completed 3D "object layer" that overlays a finished scan.
 *
 * A scan is a point cloud / splat plus detected object boxes (see `SceneFacts.objects`).
 * The *object layer* goes one step further: for a subset of detected objects we generate a
 * full 3D asset (SAM 3D → a `.glb`) and place it back in the scan at its measured pose, with
 * an honesty envelope — a per-object quality tier, a provenance string, and any caveats. The
 * viewer renders the generated asset in place with a fade-in, falling back to a wireframe box
 * for low-quality objects.
 *
 * This is an ADDITIVE cross-repo contract: the server bundles an `ObjectLayerManifest` into an
 * embed export (see the object-layer embed spec) and the webserver viewer consumes it. Field
 * names are snake_case for the fields that cross the wire (mirrors the Python server and the
 * existing `BuildingManifest` / `PersistedScene` shapes in this package).
 *
 * Geometry conventions match `BoundingBox3D` in ./types:
 *   - `center`   : object centre in the SLAM world frame, [x, y, z]
 *   - `extents`  : full edge lengths of the oriented box, [x, y, z] (NOT half-extents)
 *   - `rotation` : 3×3, columns are the box's local axes expressed in the world frame
 * Generated `.glb` assets are baked in the SAME world frame (vertices already world-placed), so
 * the viewer only has to offset them by the scan's `scene_center` — exactly like the boxes.
 */

/** Per-object honesty tier. `low` renders box-only (no asset), even if an asset_url is present. */
export type ObjectQualityTier = 'good' | 'usable' | 'low';

/** Current manifest schema version. Bump on any breaking shape change. */
export const OBJECT_LAYER_SCHEMA_VERSION = 1;

/** Default global disclaimer shown under the object-layer report cards. */
export const DEFAULT_OBJECT_LAYER_DISCLAIMER =
  'Generated assets complete unseen geometry; envelope-verified, functional parts unverified.';

/** Oriented bounding box, world frame. Mirrors `BoundingBox3D` (center/extent/rotation). */
export interface ObjectLayerOBB {
  /** Centre in the SLAM world frame, [x, y, z]. */
  center: number[];
  /** Full edge lengths [x, y, z] (twice the half-extents). */
  extents: number[];
  /** 3×3 rotation; columns are the box's local axes in the world frame. */
  rotation: number[][];
}

/** One generated object in the layer. */
export interface ObjectLayerItem {
  /** Stable id, unique within the manifest (e.g. "office_chair"). */
  id: string;
  /** Human label / best-guess category (e.g. "Office chair"). */
  label: string;
  /** Measured oriented box, world frame. */
  obb: ObjectLayerOBB;
  /** Honesty tier. `low` ⇒ the viewer shows the box only. */
  quality: ObjectQualityTier;
  /** URL of the generated `.glb` (world-baked). Optional; omitted/ignored for `low`. */
  asset_url?: string;
  /** Honesty string, e.g. "AI-completed — envelope-verified". */
  provenance: string;
  /** Zero or more caveat lines shown on the card (e.g. "legs genuinely occluded"). */
  caveats: string[];
  /** Where the object was best seen, e.g. "submap 00225 · frame 7". */
  source_frame?: string;
  /** Submap id the object belongs to (e.g. "00225"). */
  submap?: string;
}

/** The object layer bundled with an embed export. */
export interface ObjectLayerManifest {
  /** Schema version — see OBJECT_LAYER_SCHEMA_VERSION. */
  version: number;
  /** Scan this layer belongs to (matches PersistedScene.scan_id). */
  scan_id?: string;
  /** ISO-8601 generation timestamp. */
  generated_at?: string;
  /** Free-text description of the coordinate frame (e.g. "global up-to-scale world; not metric"). */
  frame?: string;
  /** Global disclaimer shown once under the cards. */
  disclaimer?: string;
  /** The generated objects. */
  objects: ObjectLayerItem[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation — hand-written guards, no dependency. Contract: never throw; the
// parse* entry point returns null on any malformed input (mirrors decodeJwtPayload).
// ─────────────────────────────────────────────────────────────────────────────

function isFiniteNumberArray(value: unknown, len?: number): value is number[] {
  return (
    Array.isArray(value) &&
    (len === undefined || value.length === len) &&
    value.length > 0 &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  );
}

function is3x3(value: unknown): value is number[][] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((row) => isFiniteNumberArray(row, 3))
  );
}

function isQualityTier(value: unknown): value is ObjectQualityTier {
  return value === 'good' || value === 'usable' || value === 'low';
}

export function isObjectLayerOBB(value: unknown): value is ObjectLayerOBB {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return isFiniteNumberArray(o.center, 3) && isFiniteNumberArray(o.extents, 3) && is3x3(o.rotation);
}

export function isObjectLayerItem(value: unknown): value is ObjectLayerItem {
  if (!value || typeof value !== 'object') return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.id === 'string' &&
    o.id.length > 0 &&
    typeof o.label === 'string' &&
    isObjectLayerOBB(o.obb) &&
    isQualityTier(o.quality) &&
    typeof o.provenance === 'string' &&
    Array.isArray(o.caveats) &&
    o.caveats.every((c) => typeof c === 'string') &&
    (o.asset_url === undefined || typeof o.asset_url === 'string') &&
    (o.source_frame === undefined || typeof o.source_frame === 'string') &&
    (o.submap === undefined || typeof o.submap === 'string')
  );
}

export function isObjectLayerManifest(value: unknown): value is ObjectLayerManifest {
  if (!value || typeof value !== 'object') return false;
  const m = value as Record<string, unknown>;
  return (
    typeof m.version === 'number' &&
    Array.isArray(m.objects) &&
    m.objects.every(isObjectLayerItem)
  );
}

/**
 * Parse an untrusted JSON value into an ObjectLayerManifest. Returns null on any malformed
 * input; never throws. Use this at the trust boundary (fetch response, embed bundle).
 */
export function parseObjectLayerManifest(raw: unknown): ObjectLayerManifest | null {
  try {
    const value = typeof raw === 'string' ? JSON.parse(raw) : raw;
    return isObjectLayerManifest(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * Coerce an arbitrary quality label into a tier. Accepts the EXP-19/EXP-20 upper-case
 * vocabulary ("GOOD" / "USABLE" / "LOW" / "STILL-CHIMERA" / "NO-GO" …). Anything unrecognized
 * or unshippable falls back to `low` (box-only) — the conservative, honesty-preserving default.
 */
export function normalizeQualityTier(raw: unknown): ObjectQualityTier {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === 'good') return 'good';
  if (s === 'usable') return 'usable';
  return 'low';
}

// ─────────────────────────────────────────────────────────────────────────────
// Converter — EXP-20 scene_inventory.json → ObjectLayerManifest.
//
// The inventory shape below is the CONTRACT this converter assumes for EXP-20's
// results_fullscene/scene_inventory.json. It is intentionally tolerant of the field-name
// variants seen across EXP-3/EXP-19/EXP-20 outputs: label vs label_guess; a string `asset_path`/`glb`
// vs the EXP-20 `asset` dict (demo_glb preferred); full `extents` vs `extent`; a columns-are-axes
// `rotation` vs the inventory's ROW-major `axes_rows` (transposed here). Reconciled against the real
// EXP-20 emitter on 2026-07-07 — see the object-layer embed spec.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Asset paths as emitted by EXP-20 (a dict of per-variant paths), or a single string for older runs.
 * Any field may be null when that variant wasn't produced (e.g. LOW-tier objects have no `demo_glb`).
 */
export interface SceneInventoryAsset {
  /** Raw generated GLB (may sit in a local/eye frame). */
  glb?: string | null;
  ply?: string | null;
  render_eye?: string | null;
  /** World-baked, global-frame GLB — the one the viewer should place. Preferred over `glb`. */
  demo_glb?: string | null;
}

/** One object as emitted by EXP-20's scene_inventory.json (assumed/tolerant shape). */
export interface SceneInventoryObject {
  id: string;
  label?: string;
  label_guess?: string;
  submap?: string;
  best_frame?: number | string;
  source_frame?: string;
  /**
   * World-frame OBB. `extents`/`extent` are FULL edge lengths (preferred; `half_extent` is half of
   * these). `axes_rows` gives the box axes as ROWS (one world axis per row); our OBB.rotation is
   * columns-are-axes, so `axes_rows` is transposed. A pre-transposed `rotation` wins if present.
   */
  world_obb?: {
    center?: number[];
    extents?: number[];
    extent?: number[];
    half_extent?: number[];
    rotation?: number[][];
    axes_rows?: number[][];
  };
  /** Quality tier as produced by the eval pass ("GOOD"/"USABLE"/"LOW"/…). */
  quality?: string;
  quality_tier?: string;
  /** Generated glb path(s): a single string (older runs) or the EXP-20 `asset` dict. */
  asset_path?: string;
  asset?: string | SceneInventoryAsset;
  glb?: string;
  provenance?: string;
  caveats?: string[];
  caveat?: string;
  /**
   * Internal pipeline commentary (EXP-20 `notes`). NOT customer-facing and deliberately NOT mapped
   * to `caveats` — customer-facing caveats must be curated at regen time (see object-layer spec).
   */
  notes?: string;
}

/** EXP-20 scene_inventory.json (assumed/tolerant shape). */
export interface SceneInventory {
  version?: number;
  scan_id?: string;
  generated_at?: string;
  frame?: string;
  disclaimer?: string;
  objects: SceneInventoryObject[];
}

export interface ConvertSceneInventoryOptions {
  /**
   * Base URL prepended to each object's relative asset path to make it fetchable by the viewer
   * (e.g. the embed's per-scan asset root). Ignored for already-absolute paths/URLs.
   */
  assetBaseUrl?: string;
  /** Provenance string used when an object doesn't carry its own. */
  defaultProvenance?: string;
  /** Global disclaimer used when the inventory doesn't carry its own. */
  disclaimer?: string;
}

const DEFAULT_PROVENANCE = 'AI-completed — envelope-verified';

function isAbsoluteUrl(path: string): boolean {
  return /^([a-z]+:)?\/\//i.test(path) || path.startsWith('/');
}

function joinUrl(base: string, path: string): string {
  if (!base) return path;
  const b = base.endsWith('/') ? base.slice(0, -1) : base;
  const p = path.startsWith('/') ? path.slice(1) : path;
  return `${b}/${p}`;
}

/** Resolve an object's asset path to a fetchable URL. Absolute paths/URLs pass through. */
export function resolveAssetUrl(path: string | undefined, assetBaseUrl?: string): string | undefined {
  if (!path) return undefined;
  if (isAbsoluteUrl(path) || !assetBaseUrl) return path;
  return joinUrl(assetBaseUrl, path);
}

/** Transpose a 3×3 matrix (swap rows↔columns). */
function transpose3x3(m: number[][]): number[][] {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

function inventoryOBB(o: SceneInventoryObject): ObjectLayerOBB | null {
  const w = o.world_obb;
  if (!w) return null;
  const center = w.center;
  // `extents`/`extent` are the FULL edge lengths (what ObjectLayerOBB wants). `half_extent` is a
  // separate half-size field in the inventory and is deliberately NOT used here.
  const extents = w.extents ?? w.extent;
  // Our OBB.rotation is columns-are-axes. The inventory emits `axes_rows` (world axes as ROWS), so
  // transpose it. A pre-transposed `rotation` (already columns-are-axes) wins if present.
  const rotation = w.rotation ?? (is3x3(w.axes_rows) ? transpose3x3(w.axes_rows) : undefined);
  if (!isFiniteNumberArray(center, 3) || !isFiniteNumberArray(extents, 3) || !is3x3(rotation)) {
    return null;
  }
  return { center, extents, rotation };
}

/**
 * Pick a fetchable GLB path from an object's asset field. Accepts a single string (older runs) or
 * the EXP-20 `asset` dict, in which case the world-baked `demo_glb` is preferred over the raw `glb`.
 * Returns undefined when no GLB is available — the object is still placed box-only, never skipped.
 */
function inventoryAssetPath(o: SceneInventoryObject): string | undefined {
  const candidates: unknown[] = [o.asset_path];
  const a = o.asset;
  if (typeof a === 'string') candidates.push(a);
  else if (a && typeof a === 'object') candidates.push(a.demo_glb, a.glb);
  candidates.push(o.glb);
  return candidates.find((c): c is string => typeof c === 'string' && c.length > 0);
}

function inventorySourceFrame(o: SceneInventoryObject): string | undefined {
  if (o.source_frame) return o.source_frame;
  if (o.submap && o.best_frame != null) return `submap ${o.submap} · frame ${o.best_frame}`;
  if (o.submap) return `submap ${o.submap}`;
  return undefined;
}

/**
 * Convert an EXP-20 scene_inventory.json object into ObjectLayerManifest. Objects missing a
 * usable world OBB are skipped (they can't be placed). Quality tiers are normalized; caveats are
 * coerced to an array; asset paths are resolved against `assetBaseUrl`.
 */
export function convertSceneInventory(
  raw: SceneInventory,
  opts: ConvertSceneInventoryOptions = {},
): ObjectLayerManifest {
  const defaultProvenance = opts.defaultProvenance ?? DEFAULT_PROVENANCE;
  const objects: ObjectLayerItem[] = [];

  for (const o of raw.objects ?? []) {
    const obb = inventoryOBB(o);
    if (!o || typeof o.id !== 'string' || !obb) continue;

    const caveats = o.caveats
      ? o.caveats.filter((c): c is string => typeof c === 'string' && c.length > 0)
      : o.caveat
        ? [o.caveat]
        : [];

    objects.push({
      id: o.id,
      label: o.label ?? o.label_guess ?? o.id,
      obb,
      quality: normalizeQualityTier(o.quality ?? o.quality_tier),
      asset_url: resolveAssetUrl(inventoryAssetPath(o), opts.assetBaseUrl),
      provenance: o.provenance ?? defaultProvenance,
      caveats,
      source_frame: inventorySourceFrame(o),
      submap: o.submap,
    });
  }

  return {
    version: raw.version ?? OBJECT_LAYER_SCHEMA_VERSION,
    scan_id: raw.scan_id,
    generated_at: raw.generated_at,
    frame: raw.frame,
    disclaimer: raw.disclaimer ?? opts.disclaimer ?? DEFAULT_OBJECT_LAYER_DISCLAIMER,
    objects,
  };
}
