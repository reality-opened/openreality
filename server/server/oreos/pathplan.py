"""F4 path planning on persisted scenes — the EXP-25 trajectory engine, ported.

Origin: ``experiments/exp25_recon_gym/scripts/traj/sample_trajectories.py`` (the
version that survived the EXP-25 mechanical dry-runs). Ported function-by-function;
every geometric stage and every hard-won fix is preserved verbatim, with its
original comment lesson attached:

  - ``estimate_up_axis`` / ``estimate_up_from_poses``  — the up-vector ladder incl.
    the SIGN trap (real VGGT worlds inherit the first camera's OpenCV frame, up is
    typically -y and tilted by the capturer's frame-0 pitch).
  - ``fit_floor_plane``          — lowest-density-SHELF seeding (not argmax: the desk-plane
    lesson), RANSAC, SVD refine, then full-cloud iterative refit with a SHRINKING
    threshold (the tilted-world lesson: a residual 2-3° tilt leaks the dense floor
    into the robot-body band far from the plane anchor).
  - ``build_occupancy_grid``     — robot-body height band, known-area mask ("we never
    scanned there" ≠ "free"), clearance inflation.
  - ``largest_free_component``   — planning restricted to the largest connected free
    component (the 5/8-unroutable lesson: floater obstacles sever free space into
    islands and "nearest free cell" lands on unreachable pockets).
  - ``astar`` → ``shortcut_path`` → ``moving_average_smooth`` → ``resample_by_arclength``
    → ``heading_rate_limited`` → ``camera_rotation`` — the full path→robot-eye-pose
    chain, incl. the deterministic A* tie-break counter and the smooth-clips-corner
    fallback.

What changed for the demo (and why — see the port-fidelity notes in each spot):

  - EXP-25's *driver* (random far-start sampling, target permutation, JSONL objects,
    argparse) is experiment harness, not engine — replaced by request-driven
    start/goal resolution. Every geometric stage is untouched.
  - ``estimate_up_from_poses`` takes a ``(N,4,4)`` pose array (``trajectory.npz``
    shape) instead of a poses JSON file; warnings become returned notes.
  - The camera-below-floor sanity check (EXP-25 printed a warning) now *fails* the
    fit with ``no_floor`` per features.md F4's fallback ladder ("camera-below-floor
    sanity trip → 'floor not found — set up axis' + override").
  - New: deterministic voxel-grid downsample (18.8M-point clouds → ≤1.5M before
    planning; no open3d on the broker), the units ladder, grid caching, goal
    snapping with honest substitution notes, and the top-down debug render.

Frames & units (docs/demo-2026-07/WORLD-TRANSFORM-CONTRACT.md — authoritative):

  - ALL inputs and outputs are WORLD-frame (the persisted scene's SLAM world,
    un-rotated, un-offset). Poses are OpenCV c2w (x right, y down, z forward).
  - Native lengths are SLAM units. Every payload states ``units: "m" | "relative"``
    plus ``units_basis``:
      anchored scene            → "m",        basis ``anchor:<derived key>``
        (lengths × ``derived_latest.scale_factor`` = metres; geometry itself stays
        in the original SLAM gauge so overlays line up with the displayed splat —
        the F7 measurement pattern)
      unanchored, capture poses → "relative", basis ``capture_height_fraction``
        (lengths ÷ median capture-camera height above the fitted floor)
      unanchored, no poses      → "relative", basis ``extent_fraction``
        (lengths ÷ robust vertical extent of the cloud)
    Metric *defaults* (clearance 0.3 m, band 0.2–1.5 m, …) are converted into SLAM
    units through the basis via a nominal capture height (1.4 m) / room height
    (2.5 m). The nominals shape DEFAULTS only — they never appear in any output
    value or claim. Request params arrive in the scene's active units (the same
    units the response reports).
  - shell.md §3 sketched ``units:"slam"|"metres"`` for the path doc; the newer
    transform contract mandates ``"m"|"relative"`` + ``units_basis`` — the contract
    wins (recorded here so the divergence is deliberate, not drift).

Determinism: the ONLY stochastic stage is the floor-fit RANSAC, seeded via
``np.random.SeedSequence(seed)`` exactly like EXP-25 (A* uses an insertion counter
tie-break; everything downstream is arithmetic). Same seed + params + geometry →
identical waypoints.

Pure numpy/scipy (+ stdlib); matplotlib only inside the debug render, imported
lazily. No flask here — ``routes_nav.py`` owns HTTP; this module is directly
callable (that is also how the real-scene verification harness drives it).
"""

from __future__ import annotations

import heapq
import io
import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------------------------
# Constants (BUILD-PLAN honesty doctrine + derived-key namespace)
# --------------------------------------------------------------------------------------------

PROVENANCE_LINE = (
    "Planned in scanned free space of the point cloud — "
    "planner visualization, not certified navigation"
)

ENGINE_NAME = "exp25-port"

# Derived-key namespace (relative to the scene's derived/ root — store keys drop the prefix).
GRID_CACHE_RELATIVE_KEY = "demo/nav/grid_cache.npz"
PATH_DOC_RELATIVE_PREFIX = "demo/nav/paths"
CLOUD_FROM_SPLAT_RELATIVE_KEY = "demo/nav/cloud_from_splat.npz"

# Voxel-downsample cap before planning (18.8M-point canonical cloud → broker-safe).
MAX_PLAN_POINTS = 1_500_000

# Metric-equivalent parameter defaults — EXP-25's CLI defaults, verbatim. Interpreted
# through the units ladder (see module docstring): on an anchored scene these ARE metres;
# otherwise they are shaped into SLAM units via the nominal constants below.
DEFAULTS_M = {
    "cell_size": 0.05,
    "band_lo": 0.2,
    "band_hi": 1.5,
    "clearance": 0.3,
    "floor_fill_radius": 0.15,
    "eye_height": 1.25,
    "speed": 0.6,  # per second
    "smooth_window": 1.0,
    "ransac_thresh": 0.03,
    "known_height_max": 3.0,
    # How far from an OBSTRUCTED goal the robot may park (see classify_goal_cell). Goals
    # that are objects land inside their own occupied volume, so some standoff is always
    # required; 3 m covers "stand beside the desk / in front of the monitor" while still
    # refusing to drag the robot across the building for a ceiling fixture on another floor.
    # Measured on canonical-office-loop 2026-07-31: the 11 previously-422'd object goals
    # need 0.63–2.81 m of standoff, so 3.0 clears all of them.
    "approach_radius": 3.0,
}
DEFAULT_FPS = 10.0
DEFAULT_MAX_ANG_VEL_DEG = 90.0
DEFAULT_SEED = 0

# Nominal metric assumptions used ONLY to shape parameter defaults on unanchored scenes
# (never emitted as an output value — the outputs stay fractions per the units ladder).
NOMINAL_CAPTURE_HEIGHT_M = 1.4  # handheld phone capture height
NOMINAL_VERTICAL_EXTENT_M = 2.5  # indoor floor-to-ceiling span

# Floor-fit internals (EXP-25 defaults, not exposed on the request).
FLOOR_LOW_QUANTILE = 0.4
FLOOR_RANSAC_ITERS = 300
FLOOR_PEAK_FRAC = 0.3
FLOOR_MIN_CANDIDATES = 50

# Safety rails on request params (the founder's sliders, W2's tool calls).
MAX_FPS = 60.0
MAX_FRAMES = 5000

_GRID_PARAM_KEYS = ("cell_size", "band_lo", "band_hi", "clearance", "floor_fill_radius", "known_height_max")


class NavError(Exception):
    """A planner failure with a wire-ready error code + honest detail.

    ``code`` ∈ {"no_geometry", "no_floor", "unreachable_goal", "bad_request"} —
    the F4 error vocabulary (+ bad_request for malformed inputs). ``extra`` is
    merged into the JSON error payload (e.g. the nearest-reachable suggestion).
    """

    def __init__(self, code: str, detail: str, extra: Optional[dict[str, Any]] = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.extra = dict(extra or {})


def wrap_to_pi(angle):
    """Wrap an angle (radians) to ``(-pi, pi]``. (EXP-25 ``_common.wrap_to_pi``, verbatim.)"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------------------------
# Deterministic voxel downsample (demo addition — 18.8M points don't fit a broker plan loop)
# --------------------------------------------------------------------------------------------

def voxel_downsample(points: np.ndarray, max_points: int = MAX_PLAN_POINTS) -> tuple[np.ndarray, float]:
    """Grid-hash downsample to ``<= max_points``, keeping the FIRST point per voxel.

    Deterministic (no RNG): voxel ids are exact int64 linear indices (no hash
    collisions), ``np.unique(..., return_index=True)`` picks each voxel's first
    point in input order, and the kept indices are re-sorted so output order is a
    stable subsequence of the input. Returns ``(points_ds, voxel_size)`` with
    ``voxel_size == 0.0`` when no downsampling was needed. numpy-only (no open3d
    on the broker).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n <= max_points:
        return pts, 0.0
    lo = pts.min(axis=0)
    extent = pts.max(axis=0) - lo
    diag = float(np.linalg.norm(extent))
    if diag <= 0.0:  # fully degenerate cloud — nothing meaningful to thin
        return pts[:max_points], 0.0
    voxel = diag / 1500.0
    for _ in range(40):
        dims = np.maximum((extent / voxel).astype(np.int64) + 1, 1)
        ids = np.floor((pts - lo) / voxel).astype(np.int64)
        np.clip(ids, 0, dims - 1, out=ids)
        lin = (ids[:, 0] * dims[1] + ids[:, 1]) * dims[2] + ids[:, 2]
        _, first_idx = np.unique(lin, return_index=True)
        if first_idx.shape[0] <= max_points:
            return pts[np.sort(first_idx)], float(voxel)
        voxel *= 1.4
    # Unreachable in practice (voxel grows geometrically); keep a hard fallback anyway.
    return pts[np.sort(first_idx)[:max_points]], float(voxel)


# --------------------------------------------------------------------------------------------
# Up-vector ladder (EXP-25 port; sign trap honored)
# --------------------------------------------------------------------------------------------

def estimate_up_axis(points: np.ndarray) -> int:
    """Autodetect the up axis as **y (1) or z (2)**, never x.

    Heuristic: a scanned room's vertical extent (ceiling height, ~2-3 m) is almost always
    smaller than its horizontal footprint, so pick whichever of y/z has the smaller range.
    This fails for scenes narrower/shorter than they are tall (a stairwell, a narrow closet
    scanned floor-to-ceiling) — override with ``up_override`` in that case.

    NOTE: this picks the AXIS only, not the SIGN. Real VGGT-SLAM worlds inherit the first
    camera's OpenCV frame (x right, y DOWN, z forward), so "up" is typically **-y** there —
    the synthetic fixture (z-up, +z) hid this. Sign comes from ``estimate_up_from_poses``
    (gravity from camera down-columns, preferred — recon scenes have ``trajectory.npz``)
    or defaults to +1 with a note. Found+fixed in the EXP-25 mechanical dry-run on scene
    smoke01, where the unsigned (+y) assumption made the floor fit lock onto a mid-height
    density band.
    """
    extent = points.max(axis=0) - points.min(axis=0)
    return 1 if extent[1] <= extent[2] else 2


def estimate_up_from_poses(poses: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    """Estimate the world up VECTOR from camera poses (gravity heuristic).

    In OpenCV camera convention the camera's +y column of ``c2w[:3, :3]`` points DOWN.
    Averaged over a whole handheld capture (yaw varies, pitch roughly constant), the mean
    camera-down direction approximates world gravity; up = -gravity. Returned as a full
    unit VECTOR (not snapped to an axis) because real VGGT-SLAM worlds inherit the FIRST
    camera's frame — any pitch the capturer held at frame 0 tilts the whole world, so the
    floor is generally NOT perpendicular to any coordinate axis (measured ~14 deg tilt on
    the EXP-25 smoke01 office scene; axis-aligned height histograms smear the floor into
    invisibility there).

    Port note: EXP-25 read a poses JSON and printed the x-dominant warning; here the input
    is the persisted ``trajectory.npz`` pose block ``(N,4,4)`` and the warning is returned
    as a note (second tuple member, ``None`` when unremarkable).
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    downs = poses[:, :3, 1]
    g = downs.mean(axis=0)
    norm = float(np.linalg.norm(g))
    if norm < 1e-9:
        raise NavError("no_floor", "degenerate capture poses (zero mean camera-down); pass up_override")
    g = g / norm
    up_vec = -g
    note = None
    if abs(up_vec[0]) > max(abs(up_vec[1]), abs(up_vec[2])):
        note = (
            f"gravity estimate {(-up_vec).round(3).tolist()} is x-dominant — unusual world frame; "
            "check the debug plot"
        )
    return up_vec, note


_UP_OVERRIDE_SPEC = {
    "y": (1, 1.0), "z": (2, 1.0),  # bare forms mean '+' (EXP-25 CLI legacy)
    "+y": (1, 1.0), "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0),
    "+x": (0, 1.0), "-x": (0, -1.0),  # demo addition: imported splats can be anything
}


def resolve_up_vector(
    points: np.ndarray,
    poses: Optional[np.ndarray],
    up_override: Any = None,
) -> tuple[np.ndarray, str, list[str]]:
    """The F4 up-vector source ladder → ``(unit up vector, up_source, notes)``.

    Priority (WORLD-TRANSFORM-CONTRACT "Up vector & floor"):
      1. explicit ``up_override`` — a signed axis string (``"+y"|"-y"|"+z"|"-z"|"+x"|"-x"``,
         bare ``"y"/"z"`` = '+') or a world-frame 3-vector; the UI flip control / import
         wizard sends this. ``up_source = "override:<spec>"``.
      2. pose-derived gravity when capture poses exist. ``up_source = "poses_gravity"``.
      3. EXP-25 extent heuristic, sign UNVERIFIED (+1 assumed) — with the loud note.
         ``up_source = "extent_heuristic_unsigned"``.
    """
    notes: list[str] = []
    if up_override is not None:
        if isinstance(up_override, str):
            spec = _UP_OVERRIDE_SPEC.get(up_override.strip().lower())
            if spec is None:
                raise NavError(
                    "bad_request",
                    f"up_override {up_override!r} not understood — use one of "
                    f"{sorted(_UP_OVERRIDE_SPEC)} or a [x,y,z] vector",
                )
            axis, sign = spec
            up = sign * np.eye(3)[axis]
            return up, f"override:{up_override.strip().lower()}", notes
        try:
            vec = np.asarray(up_override, dtype=np.float64).reshape(3)
        except Exception:
            raise NavError("bad_request", "up_override must be an axis string or a 3-vector")
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(vec).all() or norm < 1e-9:
            raise NavError("bad_request", "up_override vector must be finite and non-zero")
        vec = vec / norm
        return vec, f"override:vec[{','.join(f'{v:.3f}' for v in vec)}]", notes
    if poses is not None and np.asarray(poses).size:
        up_vec, note = estimate_up_from_poses(poses)
        if note:
            notes.append(note)
        return up_vec, "poses_gravity", notes
    up_idx = estimate_up_axis(points)
    up_vec = np.eye(3)[up_idx]
    notes.append(
        "no capture poses — up-axis SIGN defaults to +1 (extent heuristic). Real VGGT-SLAM "
        "worlds are typically y-DOWN (tilted by the first camera's pitch); pass up_override "
        "if the floor lands wrong."
    )
    return up_vec, "extent_heuristic_unsigned", notes


# --------------------------------------------------------------------------------------------
# Floor plane estimation (EXP-25 port, verbatim math)
# --------------------------------------------------------------------------------------------

@dataclass
class FloorPlane:
    point: np.ndarray  # (3,) a point on the plane (centroid of inlier floor points)
    normal: np.ndarray  # (3,) unit normal, oriented to point "up"
    u_axis: np.ndarray  # (3,) unit in-plane basis vector
    v_axis: np.ndarray  # (3,) unit in-plane basis vector, normal x u_axis


def fit_floor_plane(
    points: np.ndarray,
    up_vec: np.ndarray,
    rng: np.random.Generator,
    low_quantile: float = FLOOR_LOW_QUANTILE,
    ransac_iters: int = FLOOR_RANSAC_ITERS,
    ransac_thresh: float = 0.03,
    min_candidate_points: int = FLOOR_MIN_CANDIDATES,
    peak_frac: float = FLOOR_PEAK_FRAC,
) -> FloorPlane:
    """RANSAC-fit the floor plane from the LOWEST significant height-density shelf.

    ``up_vec`` is a unit world-up VECTOR (gravity-derived via ``estimate_up_from_poses``,
    or a signed coordinate axis when no poses are available). Heights are ``points @ up_vec``.

    1. Restrict to the bottom ``low_quantile`` fraction of points by height.
    2. Seed on the LOWEST density shelf, not the global mode: walking bins from the bottom,
       take the first run of >=2 consecutive bins whose count reaches ``peak_frac`` of the
       candidate histogram's max. WHY not argmax: in furniture-dense scans (offices) the
       densest low band is routinely the DESK/TABLE plane, not the floor — the global mode
       seeded the fit onto desks on EXP-25's smoke01 scene. The ``peak_frac`` + 2-consecutive-
       bins rule still rejects sparse below-floor floater tails the way the old mode rule did.
    3. RANSAC: repeatedly fit a plane to 3 random candidate points, keep the plane with the
       most inliers (within ``ransac_thresh`` of the plane).
    4. Refine the winning plane with a least-squares (SVD) fit on its inlier set.
    """
    up_vec = np.asarray(up_vec, dtype=np.float64)
    up_vec = up_vec / np.linalg.norm(up_vec)
    up_vals = points @ up_vec
    low_thresh = np.quantile(up_vals, low_quantile)
    candidate_mask = up_vals <= low_thresh
    candidates = points[candidate_mask]
    candidate_up_vals = up_vals[candidate_mask]
    if candidates.shape[0] < min_candidate_points:
        candidates = points  # degenerate scene (near-flat); fall back to everything
        candidate_up_vals = up_vals

    hist, edges = np.histogram(candidate_up_vals, bins=50)
    thresh = max(peak_frac * hist.max(), 1.0)
    peak_bin = None
    for b in range(len(hist) - 1):
        if hist[b] >= thresh and hist[b + 1] >= thresh:
            peak_bin = b
            break
    if peak_bin is None:  # no 2-consecutive-bin shelf; fall back to the global mode
        peak_bin = int(np.argmax(hist))
    bin_width = edges[1] - edges[0]
    mode_center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    band_half = max(bin_width * 3.0, 1e-6)
    floor_pts = candidates[np.abs(candidate_up_vals - mode_center) <= band_half]
    if floor_pts.shape[0] < min_candidate_points:
        floor_pts = candidates  # sparse floor sampling; widen to all low-band candidates
    if floor_pts.shape[0] < 3:
        raise NavError("no_floor", "not enough points near the floor to fit a plane")

    best_inliers = None
    best_count = -1
    n_floor = floor_pts.shape[0]
    for _ in range(ransac_iters):
        idx = rng.choice(n_floor, size=3, replace=False)
        p0, p1, p2 = floor_pts[idx]
        e1, e2 = p1 - p0, p2 - p0
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue  # degenerate (near-collinear) triple
        normal = normal / norm_len
        dist = np.abs((floor_pts - p0) @ normal)
        inlier_mask = dist < ransac_thresh
        count = int(inlier_mask.sum())
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
    if best_inliers is None or best_count < 3:
        raise NavError("no_floor", "RANSAC floor-plane fit failed to find any consistent plane")

    inliers = floor_pts[best_inliers]
    centroid = inliers.mean(axis=0)
    centered = inliers - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    if float(normal @ up_vec) < 0:
        normal = -normal

    # Full-cloud iterative refinement with shrinking threshold. WHY: the RANSAC seed slab
    # only covers the floor's level-set strip when the scene is tilted vs `up_vec`, and a
    # small residual tilt (~2-3 deg) leaks the DENSE floor into the robot-body occupancy
    # band far from the plane anchor (measured on EXP-25 smoke01: free space collapsed to
    # 545 cells; after this refinement the capture-camera height spread dropped from
    # std 0.049 to 0.022 scene units). Each round re-selects inliers over the WHOLE cloud
    # and re-fits; the shrinking threshold keeps furniture planes from bleeding in.
    for thr in (ransac_thresh, ransac_thresh * 2.0 / 3.0, ransac_thresh * 0.4):
        d = (points - centroid) @ normal
        m = np.abs(d) < thr
        if int(m.sum()) < max(min_candidate_points, 3):
            break
        sel = points[m]
        centroid = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
        cand_normal = vt[-1] / np.linalg.norm(vt[-1])
        if float(cand_normal @ up_vec) < 0:
            cand_normal = -cand_normal
        normal = cand_normal

    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ ref)) > 0.9:  # near-x-up world (noted upstream); use y as ref
        ref = np.array([0.0, 1.0, 0.0])
    u_axis = ref - np.dot(ref, normal) * normal
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    return FloorPlane(point=centroid, normal=normal, u_axis=u_axis, v_axis=v_axis)


def project_to_plane(points: np.ndarray, plane: FloorPlane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, height) of ``points`` in the floor-plane basis; height = signed dist along normal."""
    rel = points - plane.point
    u = rel @ plane.u_axis
    v = rel @ plane.v_axis
    h = rel @ plane.normal
    return u, v, h


def plane_to_world(plane: FloorPlane, u: np.ndarray, v: np.ndarray, height: float) -> np.ndarray:
    """Inverse of ``project_to_plane`` for a batch of (u, v) at a fixed height above the floor."""
    return (
        plane.point[None, :]
        + u[:, None] * plane.u_axis[None, :]
        + v[:, None] * plane.v_axis[None, :]
        + height * plane.normal[None, :]
    )


# --------------------------------------------------------------------------------------------
# Occupancy grid (EXP-25 port, verbatim)
# --------------------------------------------------------------------------------------------

@dataclass
class OccupancyGrid:
    free: np.ndarray  # (nu, nv) bool — planable (known floor area, not inflated-obstacle)
    occ: np.ndarray  # (nu, nv) bool — raw (uninflated) obstacle cells, for debug plotting
    known: np.ndarray  # (nu, nv) bool — any scanned geometry nearby, for debug plotting
    u_min: float
    v_min: float
    cell_size: float
    nu: int
    nv: int

    def cell_of(self, u: float, v: float) -> tuple[int, int]:
        iu = int(math.floor((u - self.u_min) / self.cell_size))
        iv = int(math.floor((v - self.v_min) / self.cell_size))
        return iu, iv

    def center_of(self, iu: int, iv: int) -> tuple[float, float]:
        return (
            self.u_min + (iu + 0.5) * self.cell_size,
            self.v_min + (iv + 0.5) * self.cell_size,
        )

    def in_bounds(self, iu: int, iv: int) -> bool:
        return 0 <= iu < self.nu and 0 <= iv < self.nv


def _disk_structure(radius_cells: int) -> np.ndarray:
    r = max(radius_cells, 0)
    if r == 0:
        return np.array([[True]])
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def build_occupancy_grid(
    u: np.ndarray,
    v: np.ndarray,
    height: np.ndarray,
    cell_size: float,
    band_lo: float,
    band_hi: float,
    clearance: float,
    floor_fill_radius: float,
    known_height_max: float = 3.0,
) -> OccupancyGrid:
    """Rasterize a 2D occupancy grid in the floor-plane (u, v) frame.

    - ``occ``: cells containing >=1 point whose height above the floor falls in
      ``[band_lo, band_hi]`` (robot-body band) — these are obstacles.
    - ``known``: cells containing >=1 point of any height below ``known_height_max`` (walls,
      floor, furniture), dilated by ``floor_fill_radius`` to bridge gaps in a sparse SLAM point
      cloud. Cells outside this mask are "we never scanned there", not "free space" — without
      it A* could route through the void outside a wall where no points exist at all.
    - ``free = known AND NOT dilate(occ, clearance)``.
    """
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    margin = cell_size * 4
    u_min -= margin
    v_min -= margin
    nu = int(math.ceil((u_max - u_min + margin) / cell_size)) + 1
    nv = int(math.ceil((v_max - v_min + margin) / cell_size)) + 1

    def to_cells(uu, vv):
        iu = np.floor((uu - u_min) / cell_size).astype(np.int64)
        iv = np.floor((vv - v_min) / cell_size).astype(np.int64)
        valid = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        return iu[valid], iv[valid]

    occ = np.zeros((nu, nv), dtype=bool)
    band_mask = (height >= band_lo) & (height <= band_hi)
    iu, iv = to_cells(u[band_mask], v[band_mask])
    occ[iu, iv] = True

    known = np.zeros((nu, nv), dtype=bool)
    known_mask = height <= known_height_max
    iu, iv = to_cells(u[known_mask], v[known_mask])
    known[iu, iv] = True
    fill_r = max(int(round(floor_fill_radius / cell_size)), 0)
    if fill_r > 0:
        known = binary_dilation(known, structure=_disk_structure(fill_r))

    clear_r = max(int(round(clearance / cell_size)), 0)
    inflated_occ = binary_dilation(occ, structure=_disk_structure(clear_r)) if clear_r > 0 else occ

    free = known & ~inflated_occ
    return OccupancyGrid(
        free=free, occ=occ, known=known, u_min=u_min, v_min=v_min,
        cell_size=cell_size, nu=nu, nv=nv,
    )


def largest_free_component(free: np.ndarray) -> tuple[np.ndarray, int]:
    """Restrict planning to the LARGEST connected free component → ``(navigable, n_components)``.

    WHY (EXP-25, verbatim lesson): on real feed-forward splats, floater/mush obstacles sever
    the free space into islands; "nearest free cell to the target" can then land on an
    unreachable pocket and A* fails for every start (observed on EXP-25 smoke01: 5/8
    trajectories unroutable before this). Starts and goals both come from one component, so
    a path always exists. The trade-off — a goal cell possibly farther from the requested
    point — is surfaced honestly via the snap/substitution notes instead of silently.
    """
    comp, n_comp = cc_label(free, structure=np.ones((3, 3), dtype=bool))
    if n_comp <= 1:
        return free, int(n_comp)
    sizes = np.bincount(comp.ravel())
    sizes[0] = 0  # background
    keep = int(np.argmax(sizes))
    return comp == keep, int(n_comp)


# --------------------------------------------------------------------------------------------
# A* + path shaping (EXP-25 port, verbatim)
# --------------------------------------------------------------------------------------------

_NEIGHBORS = [
    (dx, dy, math.sqrt(dx * dx + dy * dy))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if not (dx == 0 and dy == 0)
]


def astar(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """8-connected A* over a boolean free-space grid. Returns a list of (iu, iv) or None."""
    if not free[start] or not free[goal]:
        return None
    if start == goal:
        return [start]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    counter = 0  # deterministic tie-break, independent of Python's hash/insertion order
    open_heap = [(heuristic(start, goal), counter, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed = set()
    nu, nv = free.shape

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        cu, cv = current
        for dx, dy, step_cost in _NEIGHBORS:
            nxt = (cu + dx, cv + dy)
            if not (0 <= nxt[0] < nu and 0 <= nxt[1] < nv):
                continue
            if not free[nxt]:
                continue
            if nxt in closed:
                continue
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative
                came_from[nxt] = current
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(nxt, goal), counter, nxt))
    return None


def _line_clear(grid: OccupancyGrid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Sample the segment a->b (in world/plane units) at half-cell steps; all samples must be free."""
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(int(math.ceil(dist / (grid.cell_size * 0.5))), 1)
    for k in range(steps + 1):
        t = k / steps
        u = a[0] + t * (b[0] - a[0])
        v = a[1] + t * (b[1] - a[1])
        iu, iv = grid.cell_of(u, v)
        if not grid.in_bounds(iu, iv) or not grid.free[iu, iv]:
            return False
    return True


def shortcut_path(grid: OccupancyGrid, waypoints_uv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Greedy iterative shortcutting: from each kept waypoint, jump to the farthest waypoint
    reachable by a collision-free straight line, dropping everything in between."""
    if len(waypoints_uv) <= 2:
        return list(waypoints_uv)
    result = [waypoints_uv[0]]
    i = 0
    n = len(waypoints_uv)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _line_clear(grid, waypoints_uv[i], waypoints_uv[j]):
            j -= 1
        result.append(waypoints_uv[j])
        i = j
    return result


def moving_average_smooth(
    grid: OccupancyGrid,
    waypoints_uv: list[tuple[float, float]],
    resample_step: float,
    window: int,
) -> list[tuple[float, float]]:
    """Densify the polyline at ``resample_step`` then apply a moving-average filter to round
    corners. Falls back to the un-smoothed polyline if smoothing would clip through an
    inflated obstacle (can happen right at a tight corner)."""
    pts = np.asarray(waypoints_uv, dtype=np.float64)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seglen.sum())
    if total < 1e-9:
        return list(waypoints_uv)
    n_samples = max(int(math.ceil(total / resample_step)) + 1, 2)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    sample_s = np.linspace(0.0, total, n_samples)
    dense_u = np.interp(sample_s, cum, pts[:, 0])
    dense_v = np.interp(sample_s, cum, pts[:, 1])

    if window >= 3 and n_samples >= window:
        kernel = np.ones(window) / window
        pad = window // 2
        padded_u = np.pad(dense_u, pad, mode="edge")
        padded_v = np.pad(dense_v, pad, mode="edge")
        smooth_u = np.convolve(padded_u, kernel, mode="valid")[: len(dense_u)]
        smooth_v = np.convolve(padded_v, kernel, mode="valid")[: len(dense_v)]
        # Endpoints must stay exactly at start/goal.
        smooth_u[0], smooth_v[0] = dense_u[0], dense_v[0]
        smooth_u[-1], smooth_v[-1] = dense_u[-1], dense_v[-1]
    else:
        smooth_u, smooth_v = dense_u, dense_v

    safe = all(
        grid.in_bounds(*grid.cell_of(uu, vv)) and grid.free[grid.cell_of(uu, vv)]
        for uu, vv in zip(smooth_u, smooth_v)
    )
    if not safe:
        return list(zip(dense_u.tolist(), dense_v.tolist()))
    return list(zip(smooth_u.tolist(), smooth_v.tolist()))


def resample_by_arclength(path_uv: list[tuple[float, float]], step: float, n_frames_min: int = 2) -> np.ndarray:
    """Resample a dense polyline at fixed arc-length ``step``; returns ``(n, 2)``."""
    pts = np.asarray(path_uv, dtype=np.float64)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seglen.sum())
    n_frames = max(int(round(total / step)) + 1, n_frames_min)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    sample_s = np.linspace(0.0, total, n_frames)
    out_u = np.interp(sample_s, cum, pts[:, 0])
    out_v = np.interp(sample_s, cum, pts[:, 1])
    return np.stack([out_u, out_v], axis=1)


# --------------------------------------------------------------------------------------------
# Camera synthesis (EXP-25 port, verbatim)
# --------------------------------------------------------------------------------------------

def heading_rate_limited(raw_theta: np.ndarray, max_step: float) -> np.ndarray:
    """Rate-limit a sequence of raw headings (radians) to at most ``max_step`` change per frame."""
    out = np.empty_like(raw_theta)
    out[0] = raw_theta[0]
    for k in range(1, len(raw_theta)):
        diff = wrap_to_pi(raw_theta[k] - out[k - 1])
        diff = np.clip(diff, -max_step, max_step)
        out[k] = out[k - 1] + diff
    return out


def camera_rotation(forward_world: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """OpenCV-convention camera-to-world rotation (columns = [right, down, forward])."""
    f = forward_world / np.linalg.norm(forward_world)
    right = np.cross(f, up_world)
    right = right / np.linalg.norm(right)
    down = np.cross(f, right)
    down = down / np.linalg.norm(down)
    return np.stack([right, down, f], axis=1)  # (3,3), columns are camera axes in world frame


def rotation_to_quat(rot: np.ndarray) -> list[float]:
    """Rotation matrix → unit quaternion ``[w, x, y, z]`` (Shepperd's method; for the path
    doc's compact ``frames[{t,pos,quat}]`` shape — the route response keeps full c2w)."""
    m = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return [float(v) for v in q]


# --------------------------------------------------------------------------------------------
# Units ladder (WORLD-TRANSFORM-CONTRACT "Units & scale")
# --------------------------------------------------------------------------------------------

@dataclass
class UnitsInfo:
    units: str  # "m" | "relative"
    units_basis: str  # "anchor:<key>" | "capture_height_fraction" | "extent_fraction"
    report_per_slam: float  # multiply a SLAM length by this to get a reported value
    m_equiv_per_slam: float  # metric-equivalent scale used ONLY to shape parameter defaults
    note: Optional[str] = None


def resolve_units(
    anchor_scale_factor: Optional[float],
    anchor_key: Optional[str],
    capture_height_slam: Optional[float],
    vextent_slam: float,
) -> UnitsInfo:
    """The three-rung units ladder (module docstring). ``capture_height_slam`` is the median
    capture-camera height above the fitted floor (positive; ``None`` when no poses), and
    ``vextent_slam`` the robust vertical extent of the cloud — both computed per scene."""
    if anchor_scale_factor and anchor_scale_factor > 0:
        return UnitsInfo(
            units="m",
            units_basis=f"anchor:{anchor_key or 'derived_latest'}",
            report_per_slam=float(anchor_scale_factor),
            m_equiv_per_slam=float(anchor_scale_factor),
            note=None,
        )
    if capture_height_slam and capture_height_slam > 0:
        return UnitsInfo(
            units="relative",
            units_basis="capture_height_fraction",
            report_per_slam=1.0 / float(capture_height_slam),
            m_equiv_per_slam=NOMINAL_CAPTURE_HEIGHT_M / float(capture_height_slam),
            note=(
                "lengths are fractions of the median capture-camera height above the floor "
                "(scaled to capture height — relative units)"
            ),
        )
    vextent_slam = max(float(vextent_slam), 1e-9)
    return UnitsInfo(
        units="relative",
        units_basis="extent_fraction",
        report_per_slam=1.0 / vextent_slam,
        m_equiv_per_slam=NOMINAL_VERTICAL_EXTENT_M / vextent_slam,
        note="lengths are fractions of the scene's vertical extent (relative units)",
    )


def resolve_params(
    body_params: Optional[dict[str, Any]],
    units_info: UnitsInfo,
) -> tuple[dict[str, float], dict[str, float], Any]:
    """Merge request ``params`` over the metric defaults → ``(params_slam, params_units, up_override)``.

    Request lengths arrive in the scene's ACTIVE units (the same units the response
    reports); omitted lengths default to the EXP-25 metric values shaped through
    ``m_equiv_per_slam``. ``fps`` / ``max_ang_vel_deg`` / ``seed`` are absolute.
    Unknown keys are rejected (400) so a typo'd slider or W2 tool call fails loudly
    instead of silently planning with a default.
    """
    params = dict(body_params or {})
    if not isinstance(params, dict):
        raise NavError("bad_request", "params must be an object")
    allowed = {
        "clearance", "band_lo", "band_hi", "eye_height", "speed", "fps",
        "up_override", "seed", "max_ang_vel_deg", "approach_radius",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise NavError("bad_request", f"unknown params {unknown}; allowed: {sorted(allowed)}")

    up_override = params.pop("up_override", None)

    def _num(key: str, default: float, *, minimum: Optional[float] = None,
             maximum: Optional[float] = None, positive: bool = False) -> float:
        val = params.get(key, default)
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise NavError("bad_request", f"param {key!r} must be a number")
        if not math.isfinite(val):
            raise NavError("bad_request", f"param {key!r} must be finite")
        if positive and val <= 0:
            raise NavError("bad_request", f"param {key!r} must be > 0")
        if minimum is not None and val < minimum:
            raise NavError("bad_request", f"param {key!r} must be >= {minimum}")
        if maximum is not None and val > maximum:
            raise NavError("bad_request", f"param {key!r} must be <= {maximum}")
        return val

    slam_per_unit = 1.0 / units_info.report_per_slam  # SLAM length of one reported unit
    slam_per_m_equiv = 1.0 / units_info.m_equiv_per_slam  # SLAM length of one (equiv-)metre

    def _length(key: str, *, minimum_units: Optional[float] = None, positive: bool = False) -> tuple[float, float]:
        """→ (value in reported units, value in SLAM units); default = metric default shaped."""
        if key in params:
            val_units = _num(key, 0.0, minimum=minimum_units, positive=positive)
            return val_units, val_units * slam_per_unit
        val_slam = DEFAULTS_M[key] * slam_per_m_equiv
        return val_slam * units_info.report_per_slam, val_slam

    clearance_u, clearance_s = _length("clearance", minimum_units=0.0)
    band_lo_u, band_lo_s = _length("band_lo", minimum_units=0.0)
    band_hi_u, band_hi_s = _length("band_hi", positive=True)
    eye_u, eye_s = _length("eye_height", positive=True)
    speed_u, speed_s = _length("speed", positive=True)
    approach_u, approach_s = _length("approach_radius", minimum_units=0.0)
    if band_hi_s <= band_lo_s:
        raise NavError("bad_request", "band_hi must be > band_lo")
    fps = _num("fps", DEFAULT_FPS, positive=True, maximum=MAX_FPS)
    max_ang = _num("max_ang_vel_deg", DEFAULT_MAX_ANG_VEL_DEG, positive=True)
    seed = params.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise NavError("bad_request", "param 'seed' must be a non-negative integer")

    # Internal (non-request) lengths, shaped from the metric defaults.
    params_slam = {
        "cell_size": DEFAULTS_M["cell_size"] * slam_per_m_equiv,
        "band_lo": band_lo_s,
        "band_hi": band_hi_s,
        "clearance": clearance_s,
        "floor_fill_radius": DEFAULTS_M["floor_fill_radius"] * slam_per_m_equiv,
        "known_height_max": DEFAULTS_M["known_height_max"] * slam_per_m_equiv,
        "eye_height": eye_s,
        "speed": speed_s,
        "smooth_window": DEFAULTS_M["smooth_window"] * slam_per_m_equiv,
        "approach_radius": approach_s,
        "fps": fps,
        "max_ang_vel_deg": max_ang,
        "seed": int(seed),
    }
    params_units = {
        "clearance": clearance_u,
        "band_lo": band_lo_u,
        "band_hi": band_hi_u,
        "eye_height": eye_u,
        "speed": speed_u,
        "approach_radius": approach_u,
        "fps": fps,
        "max_ang_vel_deg": max_ang,
        "seed": int(seed),
    }
    return params_slam, params_units, up_override


# --------------------------------------------------------------------------------------------
# NavContext: the cached plane + grid bundle
# --------------------------------------------------------------------------------------------

CACHE_VERSION = 1


@dataclass
class NavContext:
    """Everything the planner needs that is worth caching per scene: the downsampled
    cloud, the fitted floor, the occupancy grid + largest navigable component, the
    up ladder outcome, and the scalars the units ladder derives from geometry.

    Units are deliberately NOT baked in: they are resolved fresh on every plan call
    from the scene record (an anchor applied after the cache was built must flip the
    reported units without a refit — geometry is anchor-invariant because planning
    always runs on the ORIGINAL world-frame cloud)."""

    points_ds: np.ndarray  # (M,3) float64 — downsampled world-frame cloud
    voxel_size: float
    n_points_full: int
    plane: FloorPlane
    up_vec: np.ndarray
    up_source: str
    grid: OccupancyGrid
    navigable: np.ndarray  # (nu,nv) bool — largest connected free component
    n_components: int
    grid_params_slam: dict[str, float]  # the SLAM-unit params the grid was built with
    seed: int
    capture_height_slam: Optional[float]  # median capture-cam height above floor (poses only)
    vextent_slam: float  # robust vertical extent along up
    parent_artifact: str  # "cloud.npz" | "splat.ply" — geometry the fit ran against
    last_cam_world: Optional[list[float]] = None  # last capture-cam position (default start)
    notes: list[str] = field(default_factory=list)
    # lazily built planning aids (never serialized)
    _nav_ij: Optional[np.ndarray] = None
    _nav_uv: Optional[np.ndarray] = None
    _nav_tree: Optional[cKDTree] = None
    _free_ij: Optional[np.ndarray] = None
    _free_uv: Optional[np.ndarray] = None
    _free_tree: Optional[cKDTree] = None

    def nav_lookup(self) -> tuple[np.ndarray, np.ndarray, cKDTree]:
        if self._nav_tree is None:
            ij = np.argwhere(self.navigable)
            uv = np.stack(
                [
                    self.grid.u_min + (ij[:, 0] + 0.5) * self.grid.cell_size,
                    self.grid.v_min + (ij[:, 1] + 0.5) * self.grid.cell_size,
                ],
                axis=1,
            )
            self._nav_ij, self._nav_uv, self._nav_tree = ij, uv, cKDTree(uv)
        return self._nav_ij, self._nav_uv, self._nav_tree

    def free_lookup(self) -> tuple[np.ndarray, np.ndarray, cKDTree]:
        if self._free_tree is None:
            ij = np.argwhere(self.grid.free)
            uv = np.stack(
                [
                    self.grid.u_min + (ij[:, 0] + 0.5) * self.grid.cell_size,
                    self.grid.v_min + (ij[:, 1] + 0.5) * self.grid.cell_size,
                ],
                axis=1,
            )
            self._free_ij, self._free_uv, self._free_tree = ij, uv, cKDTree(uv)
        return self._free_ij, self._free_uv, self._free_tree


def _grid_params_match(a: dict[str, float], b: dict[str, float]) -> bool:
    """Grid-shaping params equal within float slop (cache-validity test)."""
    for key in _GRID_PARAM_KEYS:
        va, vb = float(a.get(key, -1.0)), float(b.get(key, -2.0))
        if not math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def _robust_vertical_extent(points: np.ndarray, up_vec: np.ndarray) -> float:
    """p2–p98 span of heights along up — floater-robust vertical extent for the units ladder."""
    h = points @ (np.asarray(up_vec, dtype=np.float64) / np.linalg.norm(up_vec))
    lo, hi = np.percentile(h, [2.0, 98.0])
    return float(max(hi - lo, 1e-9))


@dataclass
class GeomFit:
    """Stage-1 output: everything up to (and including) the floor fit, no grid yet.

    Two-stage build rationale: the floor fit needs a scale-sensitive RANSAC threshold,
    but the capture-height units basis needs the FITTED floor — circular. The fit
    therefore uses a provisional scale (real anchor scale when anchored, else the
    nominal-room-height extent prior), and the definitive units ladder + grid params
    are resolved AFTER the fit (stage 2, ``attach_grid``). The provisional scale is a
    robustness knob for the fit only — never an output value."""

    points_ds: np.ndarray
    voxel_size: float
    n_points_full: int
    plane: FloorPlane
    up_vec: np.ndarray
    up_source: str
    seed: int
    capture_height_slam: Optional[float]
    vextent_slam: float
    parent_artifact: str
    last_cam_world: Optional[list[float]]
    notes: list[str]


def fit_context_geometry(
    points_full: np.ndarray,
    poses: Optional[np.ndarray],
    *,
    seed: int = DEFAULT_SEED,
    up_override: Any = None,
    m_equiv_per_slam: Optional[float] = None,
    parent_artifact: str = "cloud.npz",
) -> GeomFit:
    """Stage 1: downsample → up ladder → floor fit → camera-below-floor sanity.

    ``m_equiv_per_slam`` (metres per SLAM unit) shapes the fit's RANSAC threshold: pass
    the anchor's real ``scale_factor`` when the scene is anchored; ``None`` derives a
    provisional scale from the cloud's vertical extent vs the nominal room height.
    Raises ``NavError`` (``no_geometry`` / ``no_floor`` / ``bad_request``).
    """
    points_full = np.asarray(points_full, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points_full).all(axis=1)
    if not finite.all():
        points_full = points_full[finite]
    if points_full.shape[0] < FLOOR_MIN_CANDIDATES:
        raise NavError("no_geometry", f"scene cloud has only {points_full.shape[0]} finite points")

    points, voxel = voxel_downsample(points_full, MAX_PLAN_POINTS)

    up_vec, up_source, notes = resolve_up_vector(points, poses, up_override)
    # Pre-fit vertical extent along the LADDER up — good enough to scale the fit's
    # RANSAC threshold (a robustness knob), but NOT the units basis: on a tilted world
    # an axis-override up is degrees away from true vertical, and the building's
    # horizontal span leaks into this measurement (sin(tilt) × length). Measured on
    # canonical-office-loop: 19° tilt inflated the "vertical" extent ~2.6× and shrank
    # every defaulted parameter with it (0.3 m clearance became corridor-sealing).
    # The definitive extent is recomputed along the FITTED normal below.
    vextent_prefit = _robust_vertical_extent(points, up_vec)

    if m_equiv_per_slam is None or m_equiv_per_slam <= 0:
        m_equiv_per_slam = NOMINAL_VERTICAL_EXTENT_M / vextent_prefit
    ransac_thresh_slam = DEFAULTS_M["ransac_thresh"] / m_equiv_per_slam
    notes.append(
        f"floor-fit RANSAC threshold {ransac_thresh_slam:.5f} scene units "
        f"(0.03 m-equivalent at provisional {m_equiv_per_slam:.4f} m-equiv/unit)"
    )

    plane_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(1)[0])
    plane = fit_floor_plane(points, up_vec, plane_rng, ransac_thresh=ransac_thresh_slam)

    # Definitive vertical extent: along the fitted floor normal (true vertical), not the
    # ladder's up guess — the tilted-world lesson applied to the units ladder.
    vextent_slam = _robust_vertical_extent(points, plane.normal)
    if abs(vextent_slam - vextent_prefit) > 0.05 * max(vextent_prefit, 1e-9):
        notes.append(
            f"vertical extent re-measured along the fitted floor normal: {vextent_slam:.4f} "
            f"scene units (pre-fit up-axis estimate {vextent_prefit:.4f} — axis/normal tilt)"
        )

    # Capture-camera height sanity (EXP-25 printed this; F4's fallback ladder promotes it to a
    # hard no_floor with the up_override hint — a floor above the cameras is never right).
    capture_height_slam: Optional[float] = None
    last_cam_world: Optional[list[float]] = None
    if poses is not None and np.asarray(poses).size:
        cam_pos = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)[:, :3, 3]
        cam_h = (cam_pos - plane.point) @ plane.normal
        med = float(np.median(cam_h))
        if med <= 0:
            raise NavError(
                "no_floor",
                "capture cameras are AT/BELOW the fitted floor — floor detection is likely "
                "wrong (floor not found — set up axis via params.up_override, or check the "
                "debug plot)",
                extra={"median_capture_height": med, "up_source": up_source},
            )
        capture_height_slam = med
        last_cam_world = [float(x) for x in cam_pos[-1]]
        notes.append(
            f"capture-camera height above fitted floor: median={med:.3f} "
            f"min={float(cam_h.min()):.3f} max={float(cam_h.max()):.3f} scene units"
        )

    return GeomFit(
        points_ds=points,
        voxel_size=voxel,
        n_points_full=int(points_full.shape[0]),
        plane=plane,
        up_vec=np.asarray(up_vec, dtype=np.float64),
        up_source=up_source,
        seed=int(seed),
        capture_height_slam=capture_height_slam,
        vextent_slam=vextent_slam,
        parent_artifact=parent_artifact,
        last_cam_world=last_cam_world,
        notes=notes,
    )


def _rasterize(
    points_ds: np.ndarray,
    plane: FloorPlane,
    params_slam: dict[str, float],
    notes: list[str],
) -> tuple[OccupancyGrid, np.ndarray, int, dict[str, float]]:
    """Grid + largest component with the F4 zero-free fallback (one labeled retry at half
    clearance, then honest failure — no plan can ever succeed on an empty grid)."""
    u, v, h = project_to_plane(points_ds, plane)
    grid_kwargs = {k: float(params_slam[k]) for k in _GRID_PARAM_KEYS}
    grid = build_occupancy_grid(u, v, h, **grid_kwargs)
    effective = dict(grid_kwargs)
    if int(grid.free.sum()) == 0:
        retry_clearance = grid_kwargs["clearance"] * 0.5
        grid = build_occupancy_grid(u, v, h, **{**grid_kwargs, "clearance": retry_clearance})
        if int(grid.free.sum()) == 0:
            raise NavError(
                "no_floor",
                "zero free cells in the occupancy grid even after an automatic retry at half "
                "clearance — the scan may not include walkable floor at this scale",
            )
        effective["clearance"] = retry_clearance
        notes.append(
            f"original clearance left zero free cells; auto-retried once at half clearance "
            f"({retry_clearance:.4f} scene units) — labeled, per the F4 fallback ladder"
        )
    navigable, n_components = largest_free_component(grid.free)
    if n_components > 1:
        notes.append(
            f"free space has {n_components} connected components; planning restricted to the "
            f"largest ({int(navigable.sum())} of {int(grid.free.sum())} cells)"
        )
    return grid, navigable, n_components, effective


def attach_grid(geom: GeomFit, params_slam: dict[str, float]) -> NavContext:
    """Stage 2: rasterize the occupancy grid over a stage-1 fit → full ``NavContext``."""
    notes = list(geom.notes)
    grid, navigable, n_components, effective = _rasterize(
        geom.points_ds, geom.plane, params_slam, notes
    )
    return NavContext(
        points_ds=geom.points_ds,
        voxel_size=geom.voxel_size,
        n_points_full=geom.n_points_full,
        plane=geom.plane,
        up_vec=geom.up_vec,
        up_source=geom.up_source,
        grid=grid,
        navigable=navigable,
        n_components=n_components,
        grid_params_slam={k: float(effective[k]) for k in _GRID_PARAM_KEYS},
        seed=geom.seed,
        capture_height_slam=geom.capture_height_slam,
        vextent_slam=geom.vextent_slam,
        parent_artifact=geom.parent_artifact,
        last_cam_world=geom.last_cam_world,
        notes=notes,
    )


def rebuild_grid(ctx: NavContext, params_slam: dict[str, float]) -> NavContext:
    """Re-rasterize with new grid params, reusing the cached downsampled cloud + floor
    plane (the expensive stages). Returns a NEW context (grid-note lines refreshed)."""
    notes = [
        n for n in ctx.notes
        if not n.startswith("free space has ") and not n.startswith("original clearance left ")
    ]
    grid, navigable, n_components, effective = _rasterize(
        ctx.points_ds, ctx.plane, params_slam, notes
    )
    return NavContext(
        points_ds=ctx.points_ds,
        voxel_size=ctx.voxel_size,
        n_points_full=ctx.n_points_full,
        plane=ctx.plane,
        up_vec=ctx.up_vec,
        up_source=ctx.up_source,
        grid=grid,
        navigable=navigable,
        n_components=n_components,
        grid_params_slam={k: float(effective[k]) for k in _GRID_PARAM_KEYS},
        seed=ctx.seed,
        capture_height_slam=ctx.capture_height_slam,
        vextent_slam=ctx.vextent_slam,
        parent_artifact=ctx.parent_artifact,
        last_cam_world=ctx.last_cam_world,
        notes=notes,
    )


def build_nav_context(
    points_full: np.ndarray,
    poses: Optional[np.ndarray],
    body_params: Optional[dict[str, Any]] = None,
    anchor_scale_factor: Optional[float] = None,
    anchor_key: Optional[str] = None,
    parent_artifact: str = "cloud.npz",
) -> tuple[NavContext, UnitsInfo, dict[str, float], dict[str, float]]:
    """One-call convenience wiring both stages + the units ladder (tests, the verification
    harness, and the route's cold-build path all use this):
    ``→ (ctx, units_info, params_slam, params_units)``."""
    raw = dict(body_params or {})
    up_override = raw.get("up_override")
    seed = raw.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise NavError("bad_request", "param 'seed' must be a non-negative integer")
    geom = fit_context_geometry(
        points_full,
        poses,
        seed=seed,
        up_override=up_override,
        m_equiv_per_slam=anchor_scale_factor,
        parent_artifact=parent_artifact,
    )
    units_info = resolve_units(
        anchor_scale_factor, anchor_key, geom.capture_height_slam, geom.vextent_slam
    )
    params_slam, params_units, _ = resolve_params(raw, units_info)
    ctx = attach_grid(geom, params_slam)
    return ctx, units_info, params_slam, params_units


# --------------------------------------------------------------------------------------------
# Grid-cache (de)serialization — derived/demo/nav/grid_cache.npz
# --------------------------------------------------------------------------------------------

def serialize_nav_context(ctx: NavContext, scan_id: str) -> bytes:
    """NavContext → compressed npz bytes for ``save_derived_artifact``. Numeric arrays are
    stored natively; everything scalar/stringy rides in one ``meta_json`` entry."""
    meta = {
        "version": CACHE_VERSION,
        "engine": ENGINE_NAME,
        "scan_id": str(scan_id),
        "parent_artifact": ctx.parent_artifact,
        "voxel_size": ctx.voxel_size,
        "n_points_full": ctx.n_points_full,
        "up_vec": [float(x) for x in ctx.up_vec],
        "up_source": ctx.up_source,
        "plane": {
            "point": [float(x) for x in ctx.plane.point],
            "normal": [float(x) for x in ctx.plane.normal],
            "u_axis": [float(x) for x in ctx.plane.u_axis],
            "v_axis": [float(x) for x in ctx.plane.v_axis],
        },
        "grid": {
            "u_min": ctx.grid.u_min,
            "v_min": ctx.grid.v_min,
            "cell_size": ctx.grid.cell_size,
            "nu": ctx.grid.nu,
            "nv": ctx.grid.nv,
        },
        "grid_params_slam": ctx.grid_params_slam,
        "seed": ctx.seed,
        "n_components": ctx.n_components,
        "capture_height_slam": ctx.capture_height_slam,
        "vextent_slam": ctx.vextent_slam,
        "last_cam_world": ctx.last_cam_world,
        "notes": ctx.notes,
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        points_ds=ctx.points_ds.astype(np.float32),
        free=ctx.grid.free,
        occ=ctx.grid.occ,
        known=ctx.grid.known,
        navigable=ctx.navigable,
    )
    return buf.getvalue()


def deserialize_nav_context(blob: bytes) -> Optional[NavContext]:
    """npz bytes → NavContext, or ``None`` on any mismatch/corruption (a bad cache degrades
    to 'recompute', never to a 500 — same posture as the manifest reader)."""
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta_json"].tobytes()).decode("utf-8"))
            if int(meta.get("version", -1)) != CACHE_VERSION:
                return None
            plane = FloorPlane(
                point=np.asarray(meta["plane"]["point"], dtype=np.float64),
                normal=np.asarray(meta["plane"]["normal"], dtype=np.float64),
                u_axis=np.asarray(meta["plane"]["u_axis"], dtype=np.float64),
                v_axis=np.asarray(meta["plane"]["v_axis"], dtype=np.float64),
            )
            g = meta["grid"]
            grid = OccupancyGrid(
                free=np.asarray(data["free"], dtype=bool),
                occ=np.asarray(data["occ"], dtype=bool),
                known=np.asarray(data["known"], dtype=bool),
                u_min=float(g["u_min"]),
                v_min=float(g["v_min"]),
                cell_size=float(g["cell_size"]),
                nu=int(g["nu"]),
                nv=int(g["nv"]),
            )
            if grid.free.shape != (grid.nu, grid.nv):
                return None
            return NavContext(
                points_ds=np.asarray(data["points_ds"], dtype=np.float64),
                voxel_size=float(meta["voxel_size"]),
                n_points_full=int(meta["n_points_full"]),
                plane=plane,
                up_vec=np.asarray(meta["up_vec"], dtype=np.float64),
                up_source=str(meta["up_source"]),
                grid=grid,
                navigable=np.asarray(data["navigable"], dtype=bool),
                n_components=int(meta["n_components"]),
                grid_params_slam={k: float(v) for k, v in meta["grid_params_slam"].items()},
                seed=int(meta["seed"]),
                capture_height_slam=(
                    float(meta["capture_height_slam"])
                    if meta.get("capture_height_slam") is not None
                    else None
                ),
                vextent_slam=float(meta["vextent_slam"]),
                parent_artifact=str(meta.get("parent_artifact", "cloud.npz")),
                last_cam_world=(
                    [float(x) for x in meta["last_cam_world"]]
                    if meta.get("last_cam_world") is not None
                    else None
                ),
                notes=[str(n) for n in meta.get("notes", [])],
            )
    except Exception:
        return None


def context_reusable(
    ctx: NavContext,
    seed: int,
    up_override: Any,
    has_poses: bool,
) -> bool:
    """Is a cached context still valid for this request (grid params checked separately)?

    Invalidated when: the seed changed (the floor RANSAC is the one seeded stage — a new
    seed must be allowed to produce its own fit); an up_override is requested that differs
    from the cached source; or the cache was built WITHOUT poses (heuristic up) and the
    scene now has a trajectory (W1 lands trajectory re-persisting mid-build — the better
    up source must win as soon as it exists)."""
    if int(seed) != ctx.seed:
        return False
    if up_override is not None:
        if isinstance(up_override, str):
            want = f"override:{up_override.strip().lower()}"
        else:
            try:
                vec = np.asarray(up_override, dtype=np.float64).reshape(3)
                vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
                want = f"override:vec[{','.join(f'{v:.3f}' for v in vec)}]"
            except Exception:
                return False
        return ctx.up_source == want
    if ctx.up_source.startswith("override:"):
        return False  # cache was built with an override the request no longer asks for
    if ctx.up_source == "extent_heuristic_unsigned" and has_poses:
        return False  # trajectory appeared since the cache was built — upgrade the ladder
    return True


# --------------------------------------------------------------------------------------------
# Planning (request-driven start/goal — the demo replacement for EXP-25's random driver)
# --------------------------------------------------------------------------------------------

@dataclass
class PlanResult:
    path_uv: np.ndarray  # (n,2) resampled floor-frame path (one row per frame)
    waypoints_world: np.ndarray  # (n,3) world, ON the floor plane (client lifts for display)
    cam_positions: np.ndarray  # (n,3) world, at eye height
    c2ws: np.ndarray  # (n,4,4) OpenCV camera-to-world per frame
    times: np.ndarray  # (n,) seconds
    start_uv: tuple[float, float]
    goal_uv: tuple[float, float]
    start_world: np.ndarray  # (3,) snapped start on the floor
    goal_world: np.ndarray  # (3,) snapped goal on the floor
    start_snap_slam: float  # distance from requested start (floor-projected) to snapped cell
    goal_snap_slam: float
    path_length_slam: float
    goal_mode: str = "direct"  # "direct" | "approach" — see classify_goal_cell
    goal_requested_world: Optional[np.ndarray] = None  # (3,) the point the caller actually asked for
    approach_radius_slam: float = 0.0  # the radius the approach was allowed to use
    notes: list[str] = field(default_factory=list)


def _snap_to_navigable(
    ctx: NavContext, u: float, v: float
) -> tuple[tuple[int, int], tuple[float, float], float]:
    """Nearest navigable cell to (u, v) → (cell_ij, cell_center_uv, snap_distance_slam)."""
    nav_ij, nav_uv, nav_tree = ctx.nav_lookup()
    dist, idx = nav_tree.query([u, v])
    ij = (int(nav_ij[idx][0]), int(nav_ij[idx][1]))
    uv = (float(nav_uv[idx][0]), float(nav_uv[idx][1]))
    return ij, uv, float(dist)


# Goal-cell classes (see classify_goal_cell).
GOAL_FREE_NAVIGABLE = "free_navigable"
GOAL_FREE_POCKET = "free_pocket"
GOAL_OBSTRUCTED = "obstructed"


def classify_goal_cell(ctx: NavContext, u: float, v: float) -> str:
    """Which *kind* of place did the caller point at? → one of the ``GOAL_*`` classes.

    This replaces the original "is the goal's nearest FREE cell navigable?" test, which
    asked the wrong question and was the cause of the F4 422 storm (measured 2026-07-31:
    11/95 object goals on canonical-office-loop, 133/184 on the founder's 63.3M-point
    scene, all HTTP 422 ``unreachable_goal``).

    Why it was wrong: an OBJECT goal is the centre of a *thing*, so it lands inside that
    thing's own occupied volume — never in free space. "Nearest free cell" is then an
    essentially arbitrary neighbouring cell, and objects are precisely what carve small
    severed free pockets (the void under a desk, the gap between a monitor and the wall,
    the well of an office chair). Whether the arbitrary nearest cell fell in such a pocket
    or on open floor was a coin flip — measured on canonical-office-loop, 83 of the 84
    *succeeding* object goals were also inside occupied cells and merely got lucky.

    The distinction that actually matters:

      ``free_navigable`` the caller pointed at open, reachable floor → plan straight there.
      ``free_pocket``    the caller pointed at open floor that is severed from the navigable
                         component (scan gap / floater-severed island). This is the genuine
                         unreachable case EXP-25's largest-connected-component lesson is
                         about, and it still fails honestly with a nearest-reachable hint.
      ``obstructed``     the caller pointed at a thing (or unscanned space). Not a
                         reachability question at all — an *approach* question: a robot
                         parks next to the thing. Handled by ``plan_route``'s approach path.
    """
    iu, iv = ctx.grid.cell_of(u, v)
    if not ctx.grid.in_bounds(iu, iv):
        return GOAL_OBSTRUCTED  # outside the scanned grid — approach rules (and the radius) apply
    if not bool(ctx.grid.free[iu, iv]):
        return GOAL_OBSTRUCTED
    return GOAL_FREE_NAVIGABLE if bool(ctx.navigable[iu, iv]) else GOAL_FREE_POCKET


def plan_route(
    ctx: NavContext,
    goal_world: np.ndarray,
    start_world: Optional[np.ndarray],
    params_slam: dict[str, float],
    *,
    goal_footprint_slam: float = 0.0,
) -> PlanResult:
    """A* → shortcut → moving-average smooth → arc-length resample → rate-limited headings →
    robot-height OpenCV pose synthesis. World-frame in, world-frame out.

    ``start_world=None`` → default start (features.md: "camera-below default" — the route
    layer passes the last capture-camera position; a scene with no poses gets the navigable
    component's most central cell, computed here as the cell nearest the component's uv
    centroid).

    ``goal_footprint_slam`` widens the approach radius by the goal object's own half-size
    (a desk needs more standoff than a mug); the route layer derives it from the detected
    object's ``extent``. See ``classify_goal_cell`` for the three goal classes and why an
    obstructed goal is an *approach* problem rather than a reachability failure.
    """
    notes: list[str] = []
    goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
    gu, gv, _gh = project_to_plane(goal_world[None, :], ctx.plane)
    goal_class = classify_goal_cell(ctx, float(gu[0]), float(gv[0]))
    goal_ij, goal_uv, goal_snap = _snap_to_navigable(ctx, float(gu[0]), float(gv[0]))

    def _nearest_reachable_extra() -> dict[str, Any]:
        suggestion = plane_to_world(
            ctx.plane, np.array([goal_uv[0]]), np.array([goal_uv[1]]), 0.0
        )[0]
        return {
            "nearest_reachable": {"point_world": [float(x) for x in suggestion]},
            "snap_distance_slam": float(goal_snap),
            "goal_class": goal_class,
        }

    # The approach budget: how far from the requested goal the robot may park. Scaled by the
    # goal's own footprint so "go to the desk" tolerates the desk's half-width.
    approach_radius = float(params_slam.get("approach_radius", 0.0)) + max(
        float(goal_footprint_slam), 0.0
    )
    goal_mode = "direct"

    if goal_class == GOAL_FREE_POCKET:
        # EXP-25's largest-connected-component lesson, unchanged: the caller pointed at open
        # floor that is genuinely severed from the navigable area. Honest failure + a hint.
        raise NavError(
            "unreachable_goal",
            "goal lies in a free-space pocket disconnected from the navigable area "
            "(scan gap or floater-severed island) — nearest reachable point suggested",
            extra=_nearest_reachable_extra(),
        )

    if goal_class == GOAL_OBSTRUCTED:
        # The caller pointed at a thing (or at unscanned space). A robot does not stand
        # inside the thing — it parks next to it. Approach if there is reachable floor
        # within the budget; otherwise the goal really is out of reach, and we say so.
        goal_mode = "approach"
        if goal_snap > approach_radius:
            raise NavError(
                "unreachable_goal",
                "no reachable floor within the approach radius of this goal — the goal sits "
                "inside scanned geometry (or in unscanned space) and the nearest navigable "
                "floor is further than the robot may park; nearest reachable point suggested",
                extra={**_nearest_reachable_extra(), "approach_radius_slam": approach_radius},
            )
        notes.append(
            f"goal is inside scanned geometry (not open floor); planned an APPROACH to the "
            f"nearest reachable floor {goal_snap:.3f} scene units away — a robot parks "
            f"beside the target rather than inside it"
        )
    elif goal_snap > 1.5 * ctx.grid.cell_size:
        notes.append(
            "goal point was not on reachable free floor; substituted the nearest reachable "
            f"cell ({goal_snap:.3f} scene units away)"
        )

    if start_world is not None:
        start_world = np.asarray(start_world, dtype=np.float64).reshape(3)
        su, sv, _sh = project_to_plane(start_world[None, :], ctx.plane)
        start_ij, start_uv, start_snap = _snap_to_navigable(ctx, float(su[0]), float(sv[0]))
        if start_snap > 1.5 * ctx.grid.cell_size:
            notes.append(
                "start point was not on reachable free floor; substituted the nearest "
                f"reachable cell ({start_snap:.3f} scene units away)"
            )
    else:
        _nav_ij, nav_uv, _tree = ctx.nav_lookup()
        centroid = nav_uv.mean(axis=0)
        start_ij, start_uv, _snap = _snap_to_navigable(ctx, float(centroid[0]), float(centroid[1]))
        start_snap = 0.0
        notes.append("no start given and no capture poses — started from the free area's center")

    path_cells = astar(ctx.grid.free, start_ij, goal_ij)
    if path_cells is None:
        # Belt-and-braces: within one 8-connected component this cannot happen, but honesty
        # beats a stack trace if it ever does.
        raise NavError(
            "unreachable_goal",
            "no collision-free path found between start and goal",
            extra={"start_cell": list(start_ij), "goal_cell": list(goal_ij)},
        )

    waypoints_uv = [ctx.grid.center_of(iu, iv) for iu, iv in path_cells]
    # Snap endpoints to the exact snapped uv (cell centers already, but keep explicit).
    short = shortcut_path(ctx.grid, waypoints_uv)
    resample_step_smooth = ctx.grid.cell_size
    # moving-average window, expressed in scene units via smooth_window (the EXP-25 default
    # is "~1 m" in a metric scene — the units ladder scales it with the other length params
    # in scale-normalized scenes, else the window covers several real metres of path)
    window = max(int(round(params_slam["smooth_window"] / resample_step_smooth)), 3)
    smoothed = moving_average_smooth(ctx.grid, short, resample_step_smooth, window)

    frame_step = params_slam["speed"] / params_slam["fps"]
    uv_path = resample_by_arclength(smoothed, frame_step)
    n_frames = uv_path.shape[0]
    if n_frames > MAX_FRAMES:
        raise NavError(
            "bad_request",
            f"path would need {n_frames} frames at this speed/fps (cap {MAX_FRAMES}) — "
            "raise speed, lower fps, or pick a closer goal",
        )

    du = np.gradient(uv_path[:, 0])
    dv = np.gradient(uv_path[:, 1])
    raw_theta = np.arctan2(dv, du)
    max_step = math.radians(params_slam["max_ang_vel_deg"]) / params_slam["fps"]
    theta = heading_rate_limited(raw_theta, max_step)

    cam_pos = plane_to_world(ctx.plane, uv_path[:, 0], uv_path[:, 1], params_slam["eye_height"])
    floor_pos = plane_to_world(ctx.plane, uv_path[:, 0], uv_path[:, 1], 0.0)

    c2ws = np.empty((n_frames, 4, 4), dtype=np.float64)
    for i in range(n_frames):
        forward_world = math.cos(theta[i]) * ctx.plane.u_axis + math.sin(theta[i]) * ctx.plane.v_axis
        rot = camera_rotation(forward_world, ctx.plane.normal)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = cam_pos[i]
        c2ws[i] = c2w

    seglen = np.linalg.norm(np.diff(uv_path, axis=0), axis=1)
    return PlanResult(
        path_uv=uv_path,
        waypoints_world=floor_pos,
        cam_positions=cam_pos,
        c2ws=c2ws,
        times=np.arange(n_frames, dtype=np.float64) / params_slam["fps"],
        start_uv=start_uv,
        goal_uv=goal_uv,
        start_world=plane_to_world(ctx.plane, np.array([start_uv[0]]), np.array([start_uv[1]]), 0.0)[0],
        goal_world=plane_to_world(ctx.plane, np.array([goal_uv[0]]), np.array([goal_uv[1]]), 0.0)[0],
        start_snap_slam=start_snap,
        goal_snap_slam=goal_snap,
        path_length_slam=float(seglen.sum()),
        goal_mode=goal_mode,
        goal_requested_world=goal_world.copy(),
        approach_radius_slam=float(approach_radius),
        notes=notes,
    )


# --------------------------------------------------------------------------------------------
# Debug render (EXP-25 ``_render_debug_plot``, extended with grid masks + components)
# --------------------------------------------------------------------------------------------

def render_debug_png(
    ctx: NavContext,
    result: Optional[PlanResult] = None,
    title: str = "",
    objects: Optional[list[dict[str, Any]]] = None,
) -> bytes:
    """Top-down occupancy render → PNG bytes. matplotlib imported lazily (Agg) so the
    planner itself never needs it. Layers: scene points (gray), known mask (paper),
    free space (green tint), non-navigable free islands (orange), obstacle cells
    (firebrick), floor inliers (blue tint), path + start/goal markers, object stars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = ctx.grid
    u, v, h = project_to_plane(ctx.points_ds, ctx.plane)
    fig, ax = plt.subplots(figsize=(10, 10))

    extent = [
        grid.u_min,
        grid.u_min + grid.nu * grid.cell_size,
        grid.v_min,
        grid.v_min + grid.nv * grid.cell_size,
    ]
    # Masks are (nu, nv) indexed [iu, iv] — transpose so u runs along x. origin="lower"
    # keeps v increasing upward.
    base = np.zeros((grid.nv, grid.nu, 4))
    base[grid.known.T] = (0.93, 0.91, 0.88, 1.0)  # known-but-not-free: scanned paper
    base[grid.free.T] = (0.85, 0.93, 0.82, 1.0)  # free
    stranded = grid.free & ~ctx.navigable
    base[stranded.T] = (0.98, 0.85, 0.62, 1.0)  # free but unreachable islands
    base[grid.occ.T] = (0.70, 0.13, 0.13, 1.0)  # raw obstacle cells
    ax.imshow(base, origin="lower", extent=extent, interpolation="nearest", zorder=0)

    # Subsample the raw cloud for plot speed (EXP-25 rule).
    if len(u) > 60000:
        idx = np.random.default_rng(0).choice(len(u), 60000, replace=False)
        u_plot, v_plot, h_plot = u[idx], v[idx], h[idx]
    else:
        u_plot, v_plot, h_plot = u, v, h
    ax.scatter(u_plot, v_plot, s=0.4, c="dimgray", linewidths=0, alpha=0.25, zorder=1,
               label="scene points")
    inlier = np.abs(h_plot) < ctx.grid_params_slam.get("cell_size", grid.cell_size)
    ax.scatter(u_plot[inlier], v_plot[inlier], s=0.4, c="steelblue", linewidths=0, alpha=0.35,
               zorder=2, label="floor inliers")

    for obj in objects or []:
        center = np.asarray(obj.get("center"), dtype=np.float64).reshape(1, 3)
        ou, ov, _ = project_to_plane(center, ctx.plane)
        ax.scatter(ou, ov, marker="*", s=140, c="gold", edgecolors="black", zorder=5)
        label = obj.get("label")
        if label:
            ax.annotate(str(label), (ou[0], ov[0]), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")

    if result is not None:
        ax.plot(result.path_uv[:, 0], result.path_uv[:, 1], "-", linewidth=1.6,
                color="tab:blue", alpha=0.95, zorder=6, label="path")
        ax.scatter([result.start_uv[0]], [result.start_uv[1]], marker="o", s=45,
                   color="green", zorder=7, label="start")
        ax.scatter([result.goal_uv[0]], [result.goal_uv[1]], marker="x", s=60,
                   color="red", zorder=7, label="goal")

    ax.set_aspect("equal")
    ax.set_xlabel("u (scene units)")
    ax.set_ylabel("v (scene units)")
    head = title or "nav debug"
    ax.set_title(
        f"{head}\nfree={int(grid.free.sum())} cells, components={ctx.n_components}, "
        f"navigable={int(ctx.navigable.sum())}, cell={grid.cell_size:.4f}, up={ctx.up_source}"
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=2.0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    plt.close(fig)
    return buf.getvalue()
