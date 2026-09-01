"""Objects for IMPORTED splats — boxes from geometry, labels from renders (W-B).

An imported splat (Splatica / SuperSplat / Scaniverse) has no capture video, so it
persists with ``keyframes == []`` and ``facts.objects == []``: the Objects tab is empty
and everything downstream of object selection (nav "Go to", swap, SAM-3D completion) is
dead. See ``docs/demo-2026-07/IMPORTED-SPLAT-CONTRACT.md``.

The boxes here come from GEOMETRY, never from images — that is what makes them work with
no capture at all:

  voxel-downsample the cloud
    -> fit + remove the floor (pathplan RANSAC, the contract's floor source)
    -> remove the ceiling slab and the wall planes (``planes.detect_walls``)
    -> connected components over what is left, at voxel resolution
    -> physical-plausibility filters
    -> gravity-aware OBB per surviving cluster (``segment_geometry.pca_obb``)

HONESTY. A geometric cluster is NOT a "detected object" in the captured-scene sense.
Nothing here has seen a photograph, nothing ran a detector, and there is no detector
confidence to report — so ``confidence`` stays 0.0 and every emitted object carries
``imported_detection.provenance`` saying exactly what produced it. The geometry-only
label states the cluster's own measured size ("object 3 — 0.82 x 0.61 x 1.10") plus
placement facts read off the floor/wall planes; it never guesses a name.

Labels are the one part that needs pixels. When workstream A's synthetic views exist
(renders of this splat at known world-frame poses), ``label_clusters_from_views``
projects each cluster into its best-facing view, crops it and asks a VLM for a name.
That label is provenance-chipped ``synthetic view`` — inferred from a render, never read
off a photograph — and it upgrades the label only; the box is still geometry.

All coordinates are WORLD frame (WORLD-TRANSFORM-CONTRACT.md: un-rotated, un-offset).
All lengths are SLAM/import units (``units: "relative"``); metres exist only through the
metric anchor, which is a display concern.

Dependency policy matches ``segment_geometry``: numpy at module scope, scipy imported
lazily inside the one function that needs it, so the module imports on any broker.
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from server.oreos import planes as planes_mod
from server.oreos import segment_geometry as sg

# ---------------------------------------------------------------------------
# tuning — every number is relative to the cloud's own bbox diagonal, because an
# imported splat's units are arbitrary (an absolute threshold is meaningless here,
# the same reasoning as planes.default_thresh).
# ---------------------------------------------------------------------------

#: Cap on points fed to the pipeline. The founder's scene is 8.49M gaussians; the broker
#: has ~4 GB total and this runs synchronously inside a request. Deliberately EQUAL to
#: ``segment_geometry.MAX_LIFT_POINTS``, which is the cap the route's shared cloud loader
#: (``routes_sam3d._lift_cloud``) already applies: any other value would make the working
#: set differ between the route and any harness that hands over the raw cloud, so what got
#: verified would not be what ships.
MAX_WORK_POINTS = sg.MAX_LIFT_POINTS

#: Voxel edge as a fraction of the (outlier-robust) bbox diagonal. 1/220 puts a
#: ~17-unit room on a ~7.7 cm grid — fine enough to separate a chair from a table,
#: coarse enough that a 3M-point cloud collapses to a few 10^5 voxels.
VOXEL_FRAC = 1.0 / 220.0

#: Chebyshev radius (in voxels) that still counts as "connected". 1 = plain
#: 26-connectivity. Raising it bridges holes but merges neighbouring furniture.
LINK_RADIUS = 1

#: Plane inlier band, as a fraction of the diagonal (planes.default_thresh default).
PLANE_THRESH_FRAC = 0.01

#: A height bin holding at least this fraction of the fullest bin, high in the scene,
#: is a ceiling slab rather than clutter.
CEILING_PEAK_FRAC = 0.25

#: A wall candidate must cover at least this fraction of diagonal-squared to count as
#: room structure. ``planes.detect_walls`` is deliberately generous — it exists to offer
#: a human hover-tint chips — so on a cluttered scene its later rounds return the flat
#: FACE of a bookcase and oblique phantom slabs through open floor. Deleting those from
#: the cloud eats real objects; keeping them makes every cluster claim "against a wall".
WALL_MIN_AREA_FRAC = 0.03

#: Wall planes to look for. ``planes.MAX_WALL_PLANES`` is 6, sized for a chip list a human
#: reads. Measured on the founder's 8.49M-gaussian import, six rounds found six planes of
#: ONE parallel family and never reached the perpendicular walls, which then bridged every
#: piece of furniture into one un-clusterable blob. Removal needs every wall, not the six
#: most prominent.
WALL_MAX_PLANES = 16

#: A cluster's points-per-voxel, as a fraction of the scene's median, below which it is
#: floater noise rather than a surface. Measured on the founder's import: real structure
#: sits at 8-17 points per voxel and every stray blob at 1.1-1.7, against a scene median
#: this filter reads at run time (splat density is scene-specific, so the bar is too).
MIN_DENSITY_FRAC = 0.25

#: Share of POINTS the occupancy denoise is allowed to discard. See ``denoise_grid``.
DENOISE_MAX_POINT_LOSS = 0.15

#: Never demand more than this many points in a voxel — a genuinely sparse capture must
#: not be denoised into nothing.
DENOISE_MAX_THRESHOLD = 32

#: Plausibility band for a real object's longest edge, as a fraction of the diagonal.
#: Below: speckle. Above: unremoved structure (a wall run, the rest of the room).
MIN_EXTENT_FRAC = 0.012
MAX_EXTENT_FRAC = 0.35

#: A cluster needs this many occupied voxels to be worth a box at all.
MIN_VOXELS = 24

#: A cluster spanning more than this share of the floor-to-ceiling height is architecture
#: — a partition, a door frame, a column. Real furniture does not reach the ceiling, and
#: on a multi-room floor plan the interior walls that no plane fit caught come out
#: exactly here (measured: three full-height 3.0-tall "objects" on the founder's import).
MAX_HEIGHT_FRAC = 0.7

#: A cluster thinner than this many voxels along its shortest axis is a sheet, which at
#: this resolution means an unremoved slice of floor/wall/ceiling rather than an object.
MIN_THICKNESS_VOXELS = 1.5

#: Hard cap on emitted objects (largest support first) — the Objects list is a short
#: scroll box and a wall of fragments reads as noise, not as detection.
MAX_OBJECTS = 40

#: Points outside the robust bbox grown by this factor are dropped before voxelizing.
#: Imported splats routinely carry a handful of far-flung floater gaussians; left in,
#: they set the voxel size for the whole scene.
OUTLIER_BOX_GROW = 1.25
OUTLIER_PCTL = (0.5, 99.5)

#: Labelling model roster (vision). Same env-override shape as the agent's roster.
DEFAULT_LABEL_MODEL = "google/gemini-3-flash-preview"
DEFAULT_LABEL_FALLBACKS = ["openai/gpt-5.6-luna", "anthropic/claude-haiku-4.5"]

#: A view whose frame holds less than this fraction of the scene's points is not a
#: usable view of this scene — see ``pick_view_convention``.
VIEW_SANITY_MIN_INSIDE_FRAC = 0.05

#: Padding around a cluster's projected box before the crop goes to the VLM, as a
#: fraction of the box's own size. Context helps the model; a bare silhouette does not.
CROP_PAD_FRAC = 0.18
CROP_MIN_PX = 48

PROVENANCE_GEOMETRY = "geometric cluster (imported splat)"
PROVENANCE_VIEW = "geometric cluster, labelled from a synthetic view"

CAVEAT_GEOMETRY = (
    "Box from geometry alone — a cluster of gaussians separated from the floor, walls "
    "and ceiling. No detector ran and no image was read, so this is not a recognised "
    "object and carries no detection confidence."
)
CAVEAT_VIEW_LABEL = (
    "Name inferred from a RENDER of this splat, not from a photograph — the scene was "
    "never captured on camera."
)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectParams:
    """Everything tunable, so a route can pass overrides and the run doc can record
    exactly what produced a given set of boxes."""

    max_work_points: int = MAX_WORK_POINTS
    voxel_frac: float = VOXEL_FRAC
    link_radius: int = LINK_RADIUS
    plane_thresh_frac: float = PLANE_THRESH_FRAC
    ceiling_peak_frac: float = CEILING_PEAK_FRAC
    min_extent_frac: float = MIN_EXTENT_FRAC
    max_extent_frac: float = MAX_EXTENT_FRAC
    min_voxels: int = MIN_VOXELS
    min_thickness_voxels: float = MIN_THICKNESS_VOXELS
    max_height_frac: float = MAX_HEIGHT_FRAC
    min_density_frac: float = MIN_DENSITY_FRAC
    denoise_max_point_loss: float = DENOISE_MAX_POINT_LOSS
    wall_min_area_frac: float = WALL_MIN_AREA_FRAC
    wall_max_planes: int = WALL_MAX_PLANES
    max_objects: int = MAX_OBJECTS
    detect_walls: bool = True
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_work_points": int(self.max_work_points),
            "voxel_frac": float(self.voxel_frac),
            "link_radius": int(self.link_radius),
            "plane_thresh_frac": float(self.plane_thresh_frac),
            "ceiling_peak_frac": float(self.ceiling_peak_frac),
            "min_extent_frac": float(self.min_extent_frac),
            "max_extent_frac": float(self.max_extent_frac),
            "min_voxels": int(self.min_voxels),
            "min_thickness_voxels": float(self.min_thickness_voxels),
            "max_height_frac": float(self.max_height_frac),
            "min_density_frac": float(self.min_density_frac),
            "denoise_max_point_loss": float(self.denoise_max_point_loss),
            "wall_min_area_frac": float(self.wall_min_area_frac),
            "wall_max_planes": int(self.wall_max_planes),
            "max_objects": int(self.max_objects),
            "detect_walls": bool(self.detect_walls),
            "seed": int(self.seed),
        }


@dataclass
class Cluster:
    """One surviving connected component, in world coordinates."""

    index: int
    n_voxels: int
    n_points: int
    #: World AABB — what ``facts.objects`` persists (``center`` + ``extent``), so every
    #: existing consumer that ignores rotation still gets a valid box.
    aabb_center: list[float]
    aabb_extent: list[float]
    #: Gravity-aware OBB — the tighter box the viewer draws (ObjectLayerOBB shape).
    obb_center: list[float]
    obb_extents: list[float]
    obb_rotation: list[list[float]]
    #: Heights of the cluster's bottom/top above the fitted floor, along up.
    floor_gap: float
    height: float
    #: Distance to the nearest detected wall plane, or None when no wall was found.
    wall_distance: Optional[float]
    placement: list[str]
    label: str
    #: Points per occupied voxel — how solid this cluster is next to the rest of the scene.
    density: float = 0.0
    label_source: str = "geometry"
    label_confidence: Optional[float] = None
    label_model: Optional[str] = None
    label_view_id: Optional[str] = None
    #: Subsampled member points, kept for the labelling pass. Never serialized.
    points: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class DetectionResult:
    clusters: list[Cluster]
    up: list[float]
    up_source: str
    floor: Optional[dict[str, Any]]
    walls: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    rejected: dict[str, int]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return a / n


def _robust_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(points, OUTLIER_PCTL[0], axis=0)
    hi = np.percentile(points, OUTLIER_PCTL[1], axis=0)
    mid = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-9) * OUTLIER_BOX_GROW
    return mid - half, mid + half


def format_extent(extents: Any) -> str:
    """``0.82 x 0.61 x 1.10``, longest edge first. No unit glyph: an imported splat is
    unanchored, and the honesty doctrine forbids an "m" the geometry cannot support."""
    vals = sorted((abs(float(v)) for v in np.asarray(extents).reshape(3)), reverse=True)
    return " × ".join(f"{v:.2f}" for v in vals)


def geometry_label(index: int, extents: Any) -> str:
    """The geometry-only label: what the cluster IS measurably, never a guessed name."""
    return f"object {int(index)} — {format_extent(extents)}"


def yaw_obb(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gravity-constrained OBB whose yaw is the MINIMUM-AREA enclosing rectangle of the
    horizontal footprint. ``(center, extents FULL, R columns=axes)``, up is the last axis
    — same contract as ``segment_geometry.pca_obb``.

    Why not that function's in-plane PCA: furniture footprints are near-symmetric, and on
    a symmetric footprint the covariance has no dominant in-plane direction, so PCA picks
    an arbitrary diagonal. Measured on the synthetic room fixture, that inflated a
    1.40 x 0.90 table to a 1.56 x 0.86 box — a box visibly larger and more rotated than
    the thing it encloses, which on a demo reads as the detector being wrong. Rotating
    calipers over the 2D convex hull is the standard answer and is exact.

    Falls back to ``pca_obb`` when the footprint is degenerate (collinear / too few
    points) — hull construction is the only thing here that can fail.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    u = _unit(up)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(ref, u))
    e2 = np.cross(u, e1)

    xy = np.stack([P @ e1, P @ e2], axis=1)
    try:
        from scipy.spatial import ConvexHull  # lazy: same policy as segment_geometry

        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        return sg.pca_obb(P, up=u)
    if hull.shape[0] < 2:
        return sg.pca_obb(P, up=u)

    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    edges = edges[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    if edges.shape[0] == 0:
        return sg.pca_obb(P, up=u)

    # One rotation per hull edge; the min-area rectangle is always flush with one of them.
    proj_a = hull @ edges.T                       # (H, E) along each candidate axis
    proj_b = hull @ np.stack([-edges[:, 1], edges[:, 0]], axis=1).T
    widths = proj_a.max(axis=0) - proj_a.min(axis=0)
    heights = proj_b.max(axis=0) - proj_b.min(axis=0)
    best = int(np.argmin(widths * heights))
    a1 = e1 * edges[best, 0] + e2 * edges[best, 1]
    a2 = np.cross(u, a1)

    R = np.stack([a1, a2, u], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    local = P @ R
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    return R @ ((lo + hi) / 2.0), hi - lo, R


# ---------------------------------------------------------------------------
# stage 1 — decimate + voxelize
# ---------------------------------------------------------------------------


def voxelize(points: np.ndarray, voxel: float) -> dict[str, Any]:
    """Occupied-voxel view of a cloud.

    Returns ``{ijk (M,3) int64, centroids (M,3), counts (M,), point_voxel (N,) int64,
    dims (3,), origin (3,)}`` where ``point_voxel[i]`` indexes the voxel row that point
    ``i`` fell in. Deterministic, order-independent, numpy-only.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    voxel = max(float(voxel), 1e-12)
    origin = pts.min(axis=0)
    ijk = np.floor((pts - origin) / voxel).astype(np.int64)
    np.maximum(ijk, 0, out=ijk)
    dims = ijk.max(axis=0) + 1
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    uniq, inverse, counts = np.unique(lin, return_inverse=True, return_counts=True)
    m = uniq.shape[0]
    # Per-voxel centroid via three weighted bincounts — np.add.at is an order of
    # magnitude slower at 10^6 points and this runs inside a request.
    cent = np.empty((m, 3), dtype=np.float64)
    for axis in range(3):
        cent[:, axis] = np.bincount(inverse, weights=pts[:, axis], minlength=m) / counts
    vk = np.empty((m, 3), dtype=np.int64)
    vk[:, 0] = uniq // (dims[1] * dims[2])
    rem = uniq % (dims[1] * dims[2])
    vk[:, 1] = rem // dims[2]
    vk[:, 2] = rem % dims[2]
    return {
        "ijk": vk,
        "centroids": cent,
        "counts": counts,
        "point_voxel": inverse,
        "dims": dims,
        "origin": origin,
    }


def denoise_threshold(counts: np.ndarray, max_point_loss: float) -> int:
    """Smallest points-per-voxel a voxel must hold to count as surface.

    Chosen so that discarding everything below it loses at most ``max_point_loss`` of the
    POINTS. That asymmetry is the whole trick: floater fog is a huge number of voxels
    holding one or two gaussians each, so a threshold that deletes most of the VOXELS
    costs almost none of the geometry.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return 1
    total = float(counts.sum())
    hist = np.bincount(counts)
    lost = np.cumsum(hist * np.arange(hist.size))  # points lost by dropping count <= k
    budget = max_point_loss * total
    allowed = np.flatnonzero(lost <= budget)
    top = int(allowed[-1]) if allowed.size else 0
    return int(min(max(top + 1, 1), DENOISE_MAX_THRESHOLD))


def denoise_grid(grid: dict[str, Any], max_point_loss: float) -> dict[str, Any]:
    """Drop under-occupied voxels from a grid, keeping the point->voxel map consistent.

    Runs BEFORE any plane fit. Measured on the founder's 8.49M-gaussian import: the raw
    grid was 210,762 occupied voxels at a MEDIAN of 2 points each — a fog of stray
    gaussians filling the whole bounding box. RANSAC will happily fit plane after plane
    through fog (12 near-parallel "walls" striping the room), and the fog also bridges
    every piece of furniture into one component. Denoising first is what makes the rest
    of the pipeline mean anything on a real export.
    """
    counts = grid["counts"]
    thresh = denoise_threshold(counts, max_point_loss)
    dense = counts >= thresh
    kept = np.flatnonzero(dense)
    remap = np.full(counts.shape[0], -1, dtype=np.int64)
    remap[kept] = np.arange(kept.size)
    point_voxel = remap[grid["point_voxel"]]
    return {
        "ijk": grid["ijk"][kept],
        "centroids": grid["centroids"][kept],
        "counts": counts[kept],
        "point_voxel": point_voxel,          # -1 for points in dropped voxels
        "dims": grid["dims"],
        "origin": grid["origin"],
        "threshold": int(thresh),
        "dropped_voxels": int(counts.shape[0] - kept.size),
        "dropped_points": int(counts.sum() - counts[kept].sum()),
    }


# ---------------------------------------------------------------------------
# stage 2 — structure removal
# ---------------------------------------------------------------------------


def fit_floor_plane(
    points: np.ndarray, up: np.ndarray, *, thresh: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """``(point, normal-oriented-along-up, source)`` for the dominant horizontal plane.

    Primary source is the pathplan RANSAC fit the world-transform contract names as THE
    floor source. Imported splats are exactly the case that fit can fail on (arbitrary
    gauge, no trajectory to seed gravity), so a percentile fallback keeps the pipeline
    alive with its source recorded rather than silently pretending a fit happened.
    """
    up = _unit(up)
    try:
        plane = planes_mod.fit_floor(points, up, thresh=float(thresh), seed=int(seed))
        p0 = np.asarray(plane.point, dtype=np.float64).reshape(3)
        n = _unit(np.asarray(plane.normal, dtype=np.float64))
        if float(n @ up) < 0:
            n = -n
        return p0, n, "pathplan_ransac"
    except (planes_mod.PathplanUnavailable, ValueError, RuntimeError) as exc:
        print(f"[demo.imported_objects] floor RANSAC unavailable ({exc}) — percentile fallback")

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    h = pts @ up
    floor_h = float(np.percentile(h, 1.0))
    return up * floor_h, up, "height_percentile"


def _wall_area(wall: dict[str, Any]) -> float:
    """In-plane area of a wall candidate's percentile-clipped inlier footprint."""
    (u0, u1), (v0, v1) = wall["uv_bounds"]
    return float(max(u1 - u0, 0.0) * max(v1 - v0, 0.0))


def _ceiling_height(heights: np.ndarray, *, band: float, peak_frac: float) -> Optional[float]:
    """Height of the ceiling slab above the floor, or None when the scene has no
    horizontal structure up top (an open capture, a single wall of a room).

    A ceiling is a dense horizontal band near the ceiling: the HIGHEST histogram bin
    still holding ``peak_frac`` of the fullest bin's population.
    """
    if heights.size < 64:
        return None
    hi = float(np.percentile(heights, 99.5))
    lo = float(np.percentile(heights, 0.5))
    if not (hi > lo + band):
        return None
    bins = int(np.clip((hi - lo) / max(band, 1e-9), 8, 400))
    hist, edges = np.histogram(heights, bins=bins, range=(lo, hi))
    peak = float(hist.max())
    if peak <= 0:
        return None
    dense = np.flatnonzero(hist >= peak_frac * peak)
    if dense.size == 0:
        return None
    top = int(dense[-1])
    ceiling = float(0.5 * (edges[top] + edges[top + 1]))
    # A "ceiling" at the very top of an open scene is just the highest clutter; only
    # call it structure when it sits clearly above the bulk of the geometry.
    if ceiling < lo + 3.0 * band:
        return None
    return ceiling


def remove_structure(
    centroids: np.ndarray,
    up: np.ndarray,
    diag: float,
    params: DetectParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boolean keep-mask over ``centroids`` with floor / below-floor / ceiling / walls
    removed, plus the planes that did the removing (for placement facts + the run doc)."""
    up = _unit(up)
    band = max(params.plane_thresh_frac * diag, 1e-9)

    floor_point, floor_normal, floor_source = fit_floor_plane(
        centroids, up, thresh=band, seed=params.seed
    )
    heights = (centroids - floor_point) @ floor_normal

    # 2x the fit band on removal, for the same reason detect_walls removes at 2x thresh:
    # a RANSAC plane's ragged fringe survives a 1x cut and then clusters as a "floor-level"
    # or "ceiling-level" object. Measured on the founder's import: six thin slabs hanging
    # just under the ceiling, every one of them the ceiling's own bottom edge.
    keep = heights > 2.0 * band  # removes the floor slab AND everything under it
    n_floor = int((~keep).sum())

    ceiling = _ceiling_height(heights[keep], band=band, peak_frac=params.ceiling_peak_frac)
    n_ceiling = 0
    if ceiling is not None:
        above = keep & (heights >= ceiling - 2.0 * band)
        n_ceiling = int(above.sum())
        keep &= ~above

    walls: list[dict[str, Any]] = []
    n_wall = 0
    n_wall_rejected = 0
    if params.detect_walls and int(keep.sum()) >= 64:
        candidates = planes_mod.detect_walls(
            centroids[keep], up, thresh=band, seed=params.seed,
            max_planes=params.wall_max_planes,
        )
        min_area = params.wall_min_area_frac * diag * diag
        walls = [w for w in candidates if _wall_area(w) >= min_area]
        n_wall_rejected = len(candidates) - len(walls)
        for wall in walls:
            wp = np.asarray(wall["point"], dtype=np.float64)
            wn = np.asarray(wall["normal"], dtype=np.float64)
            on_wall = keep & (np.abs((centroids - wp) @ wn) < float(wall["thickness"]))
            n_wall += int(on_wall.sum())
            keep &= ~on_wall

    info = {
        "floor": {
            "point": [float(v) for v in floor_point],
            "normal": [float(v) for v in floor_normal],
            "source": floor_source,
            "band": float(band),
        },
        "ceiling_height": None if ceiling is None else float(ceiling),
        "walls": walls,
        "heights": heights,
        "removed": {
            "floor_and_below": n_floor,
            "ceiling": n_ceiling,
            "walls": n_wall,
        },
        "wall_candidates_rejected": n_wall_rejected,
    }
    return keep, info


# ---------------------------------------------------------------------------
# stage 3 — connected components over the surviving voxels
# ---------------------------------------------------------------------------


def _neighbour_offsets(radius: int) -> np.ndarray:
    """Half the Chebyshev-``radius`` neighbourhood — one offset per undirected edge."""
    r = max(1, int(radius))
    rng = range(-r, r + 1)
    offs = [
        (dx, dy, dz)
        for dx in rng
        for dy in rng
        for dz in rng
        if (dx, dy, dz) > (0, 0, 0)
    ]
    return np.asarray(offs, dtype=np.int64)


def cluster_voxels(ijk: np.ndarray, dims: np.ndarray, radius: int = LINK_RADIUS) -> np.ndarray:
    """Connected-component label per occupied voxel (Chebyshev ``radius`` connectivity).

    This is euclidean clustering done on the grid: linking voxels within ``radius`` is
    DBSCAN with ``eps = radius * voxel`` and the min-points test already paid by voxel
    occupancy — but it is fully vectorised (searchsorted per neighbour offset, one
    sparse connected-components call) instead of a per-point neighbourhood query, which
    is what makes 10^5 clusters-worth of geometry tractable inside a request.
    """
    from scipy.sparse import coo_matrix  # lazy: keeps the module import broker-light
    from scipy.sparse.csgraph import connected_components

    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    m = ijk.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.int64)
    dims = np.asarray(dims, dtype=np.int64).reshape(3)

    def pack(a: np.ndarray) -> np.ndarray:
        return (a[:, 0] * dims[1] + a[:, 1]) * dims[2] + a[:, 2]

    keys = pack(ijk)
    order = np.argsort(keys)
    sorted_keys = keys[order]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for off in _neighbour_offsets(radius):
        nbr = ijk + off
        valid = np.all((nbr >= 0) & (nbr < dims), axis=1)
        if not valid.any():
            continue
        src = np.flatnonzero(valid)
        nkeys = pack(nbr[src])
        pos = np.searchsorted(sorted_keys, nkeys)
        inside = pos < sorted_keys.shape[0]
        if not inside.any():
            continue
        src = src[inside]
        pos = pos[inside]
        hit = sorted_keys[pos] == nkeys[inside]
        if not hit.any():
            continue
        rows.append(src[hit])
        cols.append(order[pos[hit]])

    if not rows:
        return np.arange(m, dtype=np.int64)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix((np.ones(r.shape[0], dtype=np.int8), (r, c)), shape=(m, m))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# stage 4 — clusters -> plausible objects with boxes + placement facts
# ---------------------------------------------------------------------------


def _nearest_wall_distance(cluster: np.ndarray, walls: list[dict[str, Any]]) -> Optional[float]:
    """Distance to the nearest wall SURFACE, or None when no wall was detected.

    A plane is infinite; a wall is not. Without the in-plane bounds test every cluster in
    the room comes out "against a wall" the moment a wall plane happens to graze it —
    measured on the synthetic room fixture, where all four clusters (including a table in
    open floor) claimed wall contact. So a point only counts as near a wall when it also
    projects inside that wall's own uv footprint.
    """
    best: Optional[float] = None
    for wall in walls:
        p0 = np.asarray(wall["point"], dtype=np.float64)
        n = np.asarray(wall["normal"], dtype=np.float64)
        u = np.asarray(wall["u_axis"], dtype=np.float64)
        v = np.asarray(wall["v_axis"], dtype=np.float64)
        (u0, u1), (v0, v1) = wall["uv_bounds"]
        rel = cluster - p0
        on_face = (
            (rel @ u >= u0) & (rel @ u <= u1) & (rel @ v >= v0) & (rel @ v <= v1)
        )
        if not on_face.any():
            continue
        d = float(np.abs(rel[on_face] @ n).min())
        best = d if best is None else min(best, d)
    return best


def _placement_facts(
    floor_gap: float,
    height: float,
    wall_distance: Optional[float],
    band: float,
    ceiling_gap: Optional[float],
) -> list[str]:
    facts: list[str] = []
    if floor_gap <= 2.0 * band:
        facts.append("on the floor")
    else:
        facts.append(f"off the floor — bottom {floor_gap:.2f} above it")
    if wall_distance is not None and wall_distance <= 2.0 * band:
        facts.append("against a wall")
    if ceiling_gap is not None and ceiling_gap <= 2.0 * band:
        facts.append("reaches the ceiling")
    facts.append(f"{height:.2f} tall")
    return facts


def detect_objects(
    points: np.ndarray,
    *,
    up: Optional[Any] = None,
    params: Optional[DetectParams] = None,
    keep_points: bool = True,
) -> DetectionResult:
    """The whole pipeline: world-frame cloud in, plausible object clusters out.

    ``up`` may be a world-frame 3-vector (workstream A's ground frame, or a caller
    override); when absent the ``planes.estimate_up`` heuristic runs and the source is
    recorded honestly as such.
    """
    params = params or DetectParams()
    # Decimate BEFORE widening to float64: the founder's cloud is 8.49M points, and
    # promoting all of it first costs 200 MB of a 4 GB broker for points we then throw away.
    src = np.asarray(points).reshape(-1, 3)
    n_source = int(src.shape[0])
    pts = np.asarray(src[sg.stride_subsample(n_source, params.max_work_points)], dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < params.min_voxels:
        raise ValueError("too few finite points to cluster")

    lo, hi = _robust_box(pts)
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    n_outliers = int((~inside).sum())
    if inside.any():
        pts = pts[inside]
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not (diag > 0):
        raise ValueError("degenerate cloud (zero extent)")
    voxel = max(diag * params.voxel_frac, 1e-9)

    raw_grid = voxelize(pts, voxel)
    grid = denoise_grid(raw_grid, params.denoise_max_point_loss)
    centroids = grid["centroids"]
    if centroids.shape[0] < params.min_voxels:
        raise ValueError("cloud is too sparse to cluster after denoising")

    if up is not None:
        up_vec, up_source = _unit(up), "provided"
    else:
        up_vec, up_source = planes_mod.estimate_up(centroids)

    keep, structure = remove_structure(centroids, up_vec, diag, params)
    heights = structure["heights"]
    band = float(structure["floor"]["band"])
    ceiling = structure["ceiling_height"]

    kept_idx = np.flatnonzero(keep)
    rejected = {"floor_or_structure_voxels": int(centroids.shape[0] - kept_idx.size)}
    clusters: list[Cluster] = []
    if kept_idx.size == 0:
        return DetectionResult(
            clusters=[],
            up=[float(v) for v in up_vec],
            up_source=up_source,
            floor=structure["floor"],
            walls=structure["walls"],
            diagnostics=_diagnostics(n_source, pts.shape[0], voxel, diag, raw_grid, grid,
                                     kept_idx.size, n_outliers, structure, params, 0),
            rejected=rejected,
        )

    labels = cluster_voxels(grid["ijk"][kept_idx], grid["dims"], params.link_radius)
    n_components = int(labels.max()) + 1 if labels.size else 0

    # Voxel -> component for EVERY voxel (-1 = removed structure), then point -> component.
    voxel_comp = np.full(centroids.shape[0], -1, dtype=np.int64)
    voxel_comp[kept_idx] = labels
    pv = grid["point_voxel"]
    # Points whose voxel the denoise dropped belong to no component (pv == -1); without
    # the guard they would silently wrap onto the LAST voxel's component.
    point_comp = np.where(pv >= 0, voxel_comp[np.maximum(pv, 0)], -1)
    order = np.argsort(point_comp, kind="stable")
    sorted_comp = point_comp[order]
    starts = np.searchsorted(sorted_comp, np.arange(n_components), side="left")
    ends = np.searchsorted(sorted_comp, np.arange(n_components), side="right")

    voxel_counts = np.bincount(labels, minlength=n_components)
    walls = structure["walls"]
    floor_point = np.asarray(structure["floor"]["point"], dtype=np.float64)
    floor_normal = np.asarray(structure["floor"]["normal"], dtype=np.float64)

    # Splat density is a property of the capture, not a constant, so the floater bar is
    # read off this scene: the median points-per-occupied-voxel over the whole cloud.
    scene_density = float(np.median(grid["counts"]))
    min_density = params.min_density_frac * scene_density

    # Room height, when a ceiling was found: the yardstick for "this is architecture".
    room_height = None if ceiling is None else float(ceiling)

    reject_counts = {
        "too_few_voxels": 0, "too_small": 0, "too_large": 0, "sheet": 0, "too_sparse": 0,
        "floor_to_ceiling": 0,
    }
    candidates: list[dict[str, Any]] = []
    for comp in np.argsort(-voxel_counts):
        n_vox = int(voxel_counts[comp])
        if n_vox < params.min_voxels:
            reject_counts["too_few_voxels"] += 1
            continue
        member = order[starts[comp] : ends[comp]]
        if member.size == 0:
            reject_counts["too_few_voxels"] += 1
            continue
        cp = pts[member]
        density = float(member.size) / n_vox
        if density < min_density:
            reject_counts["too_sparse"] += 1
            continue
        longest = float((cp.max(axis=0) - cp.min(axis=0)).max())
        if longest < params.min_extent_frac * diag:
            reject_counts["too_small"] += 1
            continue
        if longest > params.max_extent_frac * diag:
            reject_counts["too_large"] += 1
            continue
        if room_height is not None:
            h = (cp - floor_point) @ floor_normal
            if float(h.max() - h.min()) > params.max_height_frac * room_height:
                reject_counts["floor_to_ceiling"] += 1
                continue
        # The OBB fit is the only O(n) step per cluster; cap it so one huge component
        # cannot dominate the request.
        fit_pts = cp[sg.stride_subsample(cp.shape[0], 200_000)]
        # Thickness is measured in the cluster's OWN frame (unconstrained PCA), not in
        # the yaw-aligned box: a sheet tilted out of horizontal has a large vertical
        # extent under a gravity-constrained fit and would sail through the filter.
        if float(np.min(sg.pca_obb(fit_pts)[1])) < params.min_thickness_voxels * voxel:
            reject_counts["sheet"] += 1
            continue
        obb_c, obb_e, obb_r = yaw_obb(fit_pts, up_vec)
        candidates.append({
            "points": cp, "fit": fit_pts, "n_vox": n_vox, "density": density,
            "obb": (obb_c, obb_e, obb_r),
        })

    for rank, cand in enumerate(candidates[: params.max_objects], start=1):
        cp = cand["points"]
        obb_c, obb_e, obb_r = cand["obb"]
        aabb_lo = cp.min(axis=0)
        aabb_hi = cp.max(axis=0)
        h = (cp - floor_point) @ floor_normal
        floor_gap = float(h.min())
        top = float(h.max())
        wall_distance = _nearest_wall_distance(cp, walls)
        ceiling_gap = None if ceiling is None else float(ceiling - top)
        placement = _placement_facts(floor_gap, top - floor_gap, wall_distance, band, ceiling_gap)
        clusters.append(
            Cluster(
                index=rank,
                n_voxels=int(cand["n_vox"]),
                n_points=int(cp.shape[0]),
                aabb_center=[float(v) for v in (aabb_lo + aabb_hi) / 2.0],
                aabb_extent=[float(v) for v in (aabb_hi - aabb_lo)],
                obb_center=[float(v) for v in obb_c],
                obb_extents=[float(v) for v in obb_e],
                obb_rotation=[[float(v) for v in row] for row in obb_r],
                floor_gap=floor_gap,
                height=float(top - floor_gap),
                wall_distance=wall_distance,
                placement=placement,
                label=geometry_label(rank, obb_e),
                density=float(cand["density"]),
                points=cand["fit"] if keep_points else None,
            )
        )

    rejected.update(reject_counts)
    rejected["over_max_objects"] = max(0, len(candidates) - params.max_objects)
    return DetectionResult(
        clusters=clusters,
        up=[float(v) for v in up_vec],
        up_source=up_source,
        floor=structure["floor"],
        walls=structure["walls"],
        diagnostics=_diagnostics(n_source, pts.shape[0], voxel, diag, raw_grid, grid,
                                 kept_idx.size, n_outliers, structure, params, n_components),
        rejected=rejected,
    )


def _diagnostics(
    n_source: int,
    n_worked: int,
    voxel: float,
    diag: float,
    raw_grid: dict[str, Any],
    grid: dict[str, Any],
    n_kept_voxels: int,
    n_outliers: int,
    structure: dict[str, Any],
    params: DetectParams,
    n_components: int,
) -> dict[str, Any]:
    return {
        "source_points": int(n_source),
        "worked_points": int(n_worked),
        "outliers_dropped": int(n_outliers),
        "voxel_size": float(voxel),
        "bbox_diagonal": float(diag),
        "occupied_voxels": int(raw_grid["centroids"].shape[0]),
        "denoise": {
            "min_points_per_voxel": int(grid["threshold"]),
            "voxels_dropped": int(grid["dropped_voxels"]),
            "points_dropped": int(grid["dropped_points"]),
            "surface_voxels": int(grid["centroids"].shape[0]),
        },
        "scene_density": round(float(np.median(grid["counts"])), 2),
        "object_voxels": int(n_kept_voxels),
        "components": int(n_components),
        "removed_voxels": dict(structure["removed"]),
        "floor_source": structure["floor"]["source"],
        "ceiling_found": structure["ceiling_height"] is not None,
        "walls_found": len(structure["walls"]),
        "wall_candidates_rejected": int(structure.get("wall_candidates_rejected", 0)),
        "params": params.to_dict(),
    }


# ---------------------------------------------------------------------------
# stage 5 — the persisted shape (facts.objects, byte-identical to a captured scene's)
# ---------------------------------------------------------------------------


def to_facts_objects(result: DetectionResult, *, units_basis: str = "slam_world_units") -> list[dict[str, Any]]:
    """``facts.objects`` entries in the EXISTING ``ObjectInstance`` shape.

    ``center``/``extent`` are the world AABB — exactly what a captured scene's detections
    are — so the Objects list, nav "Go to", swap and export all work with no change. The
    tighter gravity-aware OBB and the whole honesty envelope ride in the additive
    ``imported_detection`` field, which older consumers simply ignore.

    ``confidence`` stays 0.0 on purpose: nothing scored these. Inventing a number here is
    the one thing that would turn an honest geometric cluster into a fake detection.
    """
    out: list[dict[str, Any]] = []
    for c in result.clusters:
        out.append(
            {
                "query": c.label,
                "center": list(c.aabb_center),
                "extent": list(c.aabb_extent),
                "confidence": 0.0,
                "evidence": [],
                "imported_detection": {
                    "method": "geometry_cluster",
                    "provenance": PROVENANCE_VIEW if c.label_source == "synthetic_view" else PROVENANCE_GEOMETRY,
                    "label_source": c.label_source,
                    "geometry_label": geometry_label(c.index, c.obb_extents),
                    "obb": sg.obb_to_dict(
                        np.asarray(c.obb_center),
                        np.asarray(c.obb_extents),
                        np.asarray(c.obb_rotation),
                    ),
                    "placement": list(c.placement),
                    "support_points": int(c.n_points),
                    "support_voxels": int(c.n_voxels),
                    "density": round(float(c.density), 2),
                    "units": "relative",
                    "units_basis": units_basis,
                    "up_source": result.up_source,
                    "caveats": (
                        [CAVEAT_GEOMETRY, CAVEAT_VIEW_LABEL]
                        if c.label_source == "synthetic_view"
                        else [CAVEAT_GEOMETRY]
                    ),
                    **({"label_model": c.label_model} if c.label_model else {}),
                    **({"label_confidence": c.label_confidence} if c.label_confidence is not None else {}),
                    **({"view_id": c.label_view_id} if c.label_view_id else {}),
                },
            }
        )
    return out


def run_document(
    result: DetectionResult,
    *,
    scan_id: str,
    run_id: str,
    created_at: str,
    parent_artifact: str,
    label_mode: str,
    label_note: Optional[str] = None,
) -> dict[str, Any]:
    """The derived provenance doc. Carries the transform contract's staleness stamps
    (``scan_id``/``parent_artifact``/``created_at``/``run_id``) so a panel can badge boxes
    computed against geometry that has since been re-anchored."""
    return {
        "scan_id": scan_id,
        "run_id": run_id,
        "created_at": created_at,
        "parent_artifact": parent_artifact,
        "generator": "imported_objects.detect_objects",
        "generated": False,
        "method": "geometry_cluster",
        "provenance": PROVENANCE_GEOMETRY,
        "label_mode": label_mode,
        "label_note": label_note,
        "object_count": len(result.clusters),
        "up": list(result.up),
        "up_source": result.up_source,
        "floor": result.floor,
        "walls": [
            {k: w[k] for k in ("id", "kind", "point", "normal", "inlier_count", "thickness")}
            for w in result.walls
        ],
        "diagnostics": result.diagnostics,
        "rejected": result.rejected,
        "caveats": [CAVEAT_GEOMETRY],
        "units": "relative",
        "units_basis": "slam_world_units",
    }


# ---------------------------------------------------------------------------
# stage 6 — labels from workstream A's synthetic views (upgrade path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticView:
    """One registered render, per IMPORTED-SPLAT-CONTRACT.md Interface 1.

    ``c2w``/``intrinsics`` are the OpenCV camera geometry ``synthetic_views.py`` already
    derived from the posted three.js pose and stored on the record. When they are present
    they WIN: converting the pose twice, in two modules, is exactly how a convention
    drifts, and the writer's conversion is the authoritative one.
    """

    view_id: str
    index: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # x, y, z, w
    fov_y_deg: float
    width: int
    height: int
    c2w: Optional[list[list[float]]] = None
    intrinsics: Optional[list[float]] = None

    @staticmethod
    def from_wire(doc: dict[str, Any]) -> "SyntheticView":
        pos = [float(v) for v in doc["position"]]
        quat = [float(v) for v in doc["quaternion"]]
        c2w = doc.get("c2w")
        intr = doc.get("intrinsics")
        return SyntheticView(
            view_id=str(doc.get("view_id") or doc.get("id") or ""),
            index=int(doc.get("index", 0)),
            position=(pos[0], pos[1], pos[2]),
            quaternion=(quat[0], quat[1], quat[2], quat[3]),
            fov_y_deg=float(doc.get("fov_y_deg", 60.0)),
            width=int(doc.get("width", 0)),
            height=int(doc.get("height", 0)),
            c2w=[[float(v) for v in row] for row in c2w] if c2w else None,
            intrinsics=[float(v) for v in intr] if intr else None,
        )


def _rotation_from_quaternion(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


#: three.js cameras look down -z with +y up; OpenCV cameras look down +z with +y down.
#: The two differ by a 180-degree roll about x, which is exactly the flip that lands
#: every back-projection upside down and behind the camera when guessed wrong.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def view_c2w(view: SyntheticView, convention: str = "opengl") -> np.ndarray:
    """4x4 OpenCV camera-to-world for a view.

    Returns the view's STORED ``c2w`` when it has one — the writer already did this
    conversion and re-deriving it here would be a second, independently-drifting copy of
    the convention. ``convention`` only applies to the fallback, and says which camera
    basis the quaternion is expressed in: ``"opengl"`` for a three.js camera (the client
    that captures these), ``"opencv"`` for the trajectory convention.
    """
    if view.c2w is not None:
        return np.asarray(view.c2w, dtype=np.float64).reshape(4, 4)
    R = _rotation_from_quaternion(view.quaternion)
    if convention == "opengl":
        R = R @ _GL_TO_CV
    elif convention != "opencv":
        raise ValueError(f"unknown camera convention {convention!r}")
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = np.asarray(view.position, dtype=np.float64)
    return c2w


def view_intrinsics(view: SyntheticView) -> np.ndarray:
    """``[fx, fy, cx, cy]`` — stored when the view carries them, else from the vertical
    FOV + pixel size (square pixels)."""
    if view.intrinsics is not None:
        return np.asarray(view.intrinsics, dtype=np.float64).reshape(-1)[:4]
    h = max(int(view.height), 1)
    w = max(int(view.width), 1)
    f = (h / 2.0) / math.tan(math.radians(max(view.fov_y_deg, 1e-3)) / 2.0)
    return np.array([f, f, w / 2.0, h / 2.0], dtype=np.float64)


def project_points(points: np.ndarray, c2w: np.ndarray, intr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> ``(uv (N,2), z (N,))`` in a view's pixel grid, OpenCV convention."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R_wc = c2w[:3, :3].T
    t_wc = -R_wc @ c2w[:3, 3]
    pc = P @ R_wc.T + t_wc
    z = pc[:, 2]
    safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
    u = intr[0] * pc[:, 0] / safe + intr[2]
    v = intr[1] * pc[:, 1] / safe + intr[3]
    return np.stack([u, v], axis=1), z


def pick_view_convention(points: np.ndarray, views: list[SyntheticView]) -> tuple[str, float]:
    """Which camera basis the views' quaternions are in, decided by measurement.

    The contract fixes the FRAME (world) but not the camera BASIS, and getting it wrong
    puts every crop on the wrong pixels — silently. So probe: project a sample of the
    scene into every view under both conventions and take whichever actually sees the
    scene. Returns ``(convention, in_frame_fraction)``; the caller refuses to label from
    views when the fraction is below :data:`VIEW_SANITY_MIN_INSIDE_FRAC`, rather than
    cropping confidently from the wrong place.
    """
    sample = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sample = sample[sg.stride_subsample(sample.shape[0], 20_000)]
    # Views that carry the writer's own OpenCV c2w have no ambiguity left to resolve;
    # only the sanity number is still worth measuring.
    conventions = ("stored",) if all(v.c2w is not None for v in views) else ("opengl", "opencv")
    best = (conventions[0], -1.0)
    for convention in conventions:
        seen = 0.0
        for view in views:
            try:
                uv, z = project_points(sample, view_c2w(view, convention), view_intrinsics(view))
            except ValueError:
                continue
            ok = (
                (z > 1e-6)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < view.width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < view.height)
            )
            seen = max(seen, float(ok.mean()))
        if seen > best[1]:
            best = (convention, seen)
    return best


def cluster_in_view(
    cluster_points: np.ndarray, view: SyntheticView, convention: str
) -> Optional[dict[str, Any]]:
    """Where a cluster lands in one view: ``{bbox_px, inside_frac, area_px, depth}``,
    or None when it is behind the camera / entirely out of frame."""
    uv, z = project_points(cluster_points, view_c2w(view, convention), view_intrinsics(view))
    front = z > 1e-6
    if not front.any():
        return None
    uvf = uv[front]
    inside = (
        (uvf[:, 0] >= 0) & (uvf[:, 0] < view.width) & (uvf[:, 1] >= 0) & (uvf[:, 1] < view.height)
    )
    if not inside.any():
        return None
    vis = uvf[inside]
    x0, y0 = vis.min(axis=0)
    x1, y1 = vis.max(axis=0)
    return {
        "bbox_px": [float(x0), float(y0), float(x1), float(y1)],
        "inside_frac": float(inside.mean()),
        "area_px": float(max(x1 - x0, 1.0) * max(y1 - y0, 1.0)),
        "depth": float(np.median(z[front])),
    }


def best_view_for(
    cluster_points: np.ndarray, views: list[SyntheticView], convention: str
) -> Optional[tuple[SyntheticView, dict[str, Any]]]:
    """The view that sees the most of this cluster, biggest on screen. Score is
    ``in-frame fraction x projected area`` — a view that only clips the corner of an
    object loses to one that frames it, even if it is closer."""
    best: Optional[tuple[float, SyntheticView, dict[str, Any]]] = None
    for view in views:
        hit = cluster_in_view(cluster_points, view, convention)
        if hit is None:
            continue
        score = hit["inside_frac"] * hit["area_px"]
        if best is None or score > best[0]:
            best = (score, view, hit)
    if best is None:
        return None
    return best[1], best[2]


def crop_jpeg_b64(
    png_bytes: bytes, bbox_px: list[float], width: int, height: int, *, pad_frac: float = CROP_PAD_FRAC
) -> str:
    """Padded crop of a view around a projected box, JPEG base64 (the shape
    ``OpenRouterClient`` puts on the wire)."""
    from PIL import Image  # lazy: Pillow is a broker dep but not needed to import this

    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    pad_x = max((x1 - x0) * pad_frac, CROP_MIN_PX / 2.0)
    pad_y = max((y1 - y0) * pad_frac, CROP_MIN_PX / 2.0)
    box = (
        int(max(0, math.floor(x0 - pad_x))),
        int(max(0, math.floor(y0 - pad_y))),
        int(min(width, math.ceil(x1 + pad_x))),
        int(min(height, math.ceil(y1 + pad_y))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("degenerate crop box")
    with Image.open(io.BytesIO(png_bytes)) as img:
        crop = img.convert("RGB").crop(box)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


LABEL_SYSTEM_PROMPT = (
    "You name objects in crops taken from a RENDER of a 3D gaussian-splat scan. "
    "The crop is centred on one object whose 3D extent was measured geometrically. "
    "Answer with the common name of that object in one to three words. "
    "If the crop is too blurry, too partial or too ambiguous to name, answer exactly "
    "'unknown' — a wrong name is far worse than no name. "
    'Reply as JSON: {"label": "...", "confidence": 0.0-1.0}.'
)


def label_from_crop(crop_b64: str, client: Any, *, size_hint: str = "") -> tuple[Optional[str], Optional[float], str]:
    """``(label|None, confidence|None, model)`` for one crop. ``None`` label means the
    model declined — the caller keeps the geometry-only label rather than inventing one."""
    prompt = "Name the object at the centre of this crop."
    if size_hint:
        prompt += f" Its measured extent is {size_hint} (scene units, not metres)."
    parsed, response = client.chat_json(
        system_prompt=LABEL_SYSTEM_PROMPT,
        user_prompt=prompt,
        images_b64=[crop_b64],
        temperature=0.1,
        max_tokens=120,
    )
    raw = str(parsed.get("label") or "").strip()
    if not raw or raw.lower() in ("unknown", "unclear", "n/a", "none"):
        return None, None, response.model
    conf = parsed.get("confidence")
    try:
        conf_f: Optional[float] = float(conf)
    except (TypeError, ValueError):
        conf_f = None
    return raw[:64], conf_f, response.model


def label_clusters_from_views(
    result: DetectionResult,
    views: list[SyntheticView],
    load_view: Callable[[SyntheticView], Optional[bytes]],
    client: Any,
    *,
    max_labels: int = MAX_OBJECTS,
) -> dict[str, Any]:
    """Upgrade geometry-only labels to VLM names read off synthetic views, in place.

    Every failure mode degrades to the geometry label with a reason recorded — a scene
    where the views do not see the objects must end up looking like a geometry-only run,
    not like a run that invented names.
    """
    summary: dict[str, Any] = {"attempted": 0, "labelled": 0, "declined": 0, "errors": 0}
    if not views or client is None or not result.clusters:
        summary["note"] = "no synthetic views registered for this scene"
        return summary

    all_points = np.concatenate([c.points for c in result.clusters if c.points is not None])
    convention, seen_frac = pick_view_convention(all_points, views)
    summary["camera_convention"] = convention
    summary["view_in_frame_frac"] = round(seen_frac, 4)
    if seen_frac < VIEW_SANITY_MIN_INSIDE_FRAC:
        summary["note"] = (
            f"synthetic views do not see this scene (best in-frame fraction {seen_frac:.3f} "
            f"under either camera convention) — refusing to crop from the wrong pixels"
        )
        return summary

    cache: dict[str, Optional[bytes]] = {}
    for cluster in result.clusters[:max_labels]:
        if cluster.points is None:
            continue
        summary["attempted"] += 1
        picked = best_view_for(cluster.points, views, convention)
        if picked is None:
            summary["declined"] += 1
            continue
        view, hit = picked
        try:
            if view.view_id not in cache:
                cache[view.view_id] = load_view(view)
            png = cache[view.view_id]
            if not png:
                summary["errors"] += 1
                continue
            crop = crop_jpeg_b64(png, hit["bbox_px"], view.width, view.height)
            label, confidence, model = label_from_crop(
                crop, client, size_hint=format_extent(cluster.obb_extents)
            )
        except Exception as exc:  # provider/network/decode variance — never fatal
            print(f"[demo.imported_objects] label failed for object {cluster.index}: {exc}")
            summary["errors"] += 1
            continue
        if not label:
            summary["declined"] += 1
            continue
        cluster.label = label
        cluster.label_source = "synthetic_view"
        cluster.label_confidence = confidence
        cluster.label_model = model
        cluster.label_view_id = view.view_id
        summary["labelled"] += 1
    return summary


# ---------------------------------------------------------------------------
# label-model roster (mirrors persisted_agent's env-override shape)
# ---------------------------------------------------------------------------


def label_model_roster() -> tuple[str, list[str]]:
    model = os.environ.get("DEMO_IMPORTED_LABEL_MODEL", DEFAULT_LABEL_MODEL).strip()
    raw = os.environ.get("DEMO_IMPORTED_LABEL_FALLBACKS", "")
    fallbacks = (
        [item.strip() for item in raw.split(",") if item.strip()]
        if raw.strip()
        else list(DEFAULT_LABEL_FALLBACKS)
    )
    return model or DEFAULT_LABEL_MODEL, fallbacks


def make_label_client() -> Optional[Any]:
    """A vision-capable OpenRouter client, or ``None`` when no key is mounted (the
    geometry-only path then stands on its own — it is never mocked)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or os.environ.get("SCENE_AGENT_DISABLE_LLM", "").strip() == "1":
        return None
    try:
        from server.billing.usage import named_tally
        from server.llm.openrouter_client import OpenRouterClient

        model, fallbacks = label_model_roster()
        return OpenRouterClient(
            api_key=api_key,
            primary_model=model,
            fallback_models=fallbacks,
            timeout=30.0,
            app_name="OpenReality Imported Objects",
            max_retries=1,
            usage_sink=named_tally("imported_objects_label"),
        )
    except Exception as exc:
        print(f"[demo.imported_objects] label client init failed: {exc}")
        return None


__all__ = [
    "Cluster",
    "DetectParams",
    "DetectionResult",
    "SyntheticView",
    "best_view_for",
    "cluster_in_view",
    "cluster_voxels",
    "crop_jpeg_b64",
    "detect_objects",
    "fit_floor_plane",
    "format_extent",
    "geometry_label",
    "label_clusters_from_views",
    "label_from_crop",
    "label_model_roster",
    "make_label_client",
    "pick_view_convention",
    "project_points",
    "remove_structure",
    "run_document",
    "to_facts_objects",
    "view_c2w",
    "view_intrinsics",
    "voxelize",
]
