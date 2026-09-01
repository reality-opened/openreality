"""Plane candidates + floor-patch synthesis for the demo restyle/swap features (W5).

Geometry only — no Flask. Routes live in ``routes_planes.py``.

Three jobs:

1. ``estimate_up`` — world-up with the WORLD-TRANSFORM-CONTRACT source ladder:
   pose-derived gravity (mean camera-down of the trajectory's OpenCV c2w
   matrices) when a trajectory exists, else the floor-plane heuristic from
   ``server/export/isaac/align.py`` (imported, not re-implemented), with a
   caller ``up_override`` escape hatch. Every result carries ``up_source``.

2. ``detect_walls`` — sequential RANSAC for near-vertical planes, patterned on
   ``export/isaac/align.py::fit_plane_ransac`` (relative threshold from the
   bbox diagonal, deterministic subsample cap, SVD refit on the best inlier
   set, full-cloud recheck) with the F5 wall constraint ``|n . up| < 0.2``
   enforced at hypothesis time and re-checked after refit. At most 6 planes.
   Candidates are HUMAN-CONFIRMED client-side (hover-tint chips) — this module
   never auto-picks a wall.

3. ``synthesize_floor_patch`` — the F6 hole-fill: sample floor-inlier
   color/height in an annulus around an OBB footprint and synthesize ~2k
   jittered floor-colored gaussians in ``splat_io`` field-dict form
   (``serialize_splat_ply``-ready). Always labeled generated content.

The FLOOR plane itself is NOT fit here. Per the world-transform contract the
floor comes from W4's exp25 port (``server/oreos/pathplan.py``, branch
feat/demo-pathplan, building concurrently); ``load_pathplan()`` imports it
lazily and callers answer 503 ``pathplan_not_merged`` while it is absent so
this branch stays green standalone. Seam (exp25 ``sample_trajectories.py``):

    pathplan.fit_floor_plane(points, up_vec, rng, ..., ransac_thresh=...)
        -> FloorPlane(point, normal, u_axis, v_axis)

All coordinates are WORLD frame (un-rotated, un-offset); all lengths are SLAM
units (``units: "relative"``, ``units_basis: "slam_world_units"``) — metres
exist only through the metric anchor, which is a display concern.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np

_EPS = 1e-9

# 3DGS SH DC coefficient: rgb = 0.5 + _SH_C0 * f_dc  (splat_io convention).
_SH_C0 = 0.28209479177387814

# F5 spec: a wall hypothesis must satisfy |normal . up| < this.
WALL_VERTICAL_TOL = 0.2

# F5 spec: at most this many wall candidates.
MAX_WALL_PLANES = 6


# ---------------------------------------------------------------------------
# small helpers (align.py idioms)
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("zero-length vector")
    return v / n


def _bbox_diagonal(points: np.ndarray) -> float:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _finite_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    return pts if finite.all() else pts[finite]


def default_thresh(points: np.ndarray, thresh_frac: float = 0.01) -> float:
    """The relative plane-inlier threshold: ``thresh_frac`` of the bbox diagonal
    (the SLAM cloud is up-to-scale, so an absolute threshold is meaningless —
    same reasoning as ``align.fit_plane_ransac``)."""
    pts = _finite_points(points)
    if pts.shape[0] < 2:
        return _EPS
    return max(thresh_frac * _bbox_diagonal(pts), _EPS)


def downsample_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    max_points: int = 400_000,
    seed: int = 0,
):
    """Deterministic random subsample so multi-million-point clouds stay
    broker-friendly. Colors (if given) are subsampled with the same index."""
    pts = np.asarray(points)
    n = pts.shape[0]
    if n <= max_points:
        return (pts, colors) if colors is not None else (pts, None)
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    return (pts[idx], colors[idx] if colors is not None else None)


# ---------------------------------------------------------------------------
# pathplan seam (W4)
# ---------------------------------------------------------------------------


class PathplanUnavailable(RuntimeError):
    """W4's ``server/oreos/pathplan.py`` has not merged yet — the floor plane
    (and therefore every route needing it) answers 503 ``pathplan_not_merged``."""


def load_pathplan():
    """The W4 pathplan module, or raise :class:`PathplanUnavailable`.

    ``importlib.import_module`` (not ``from server.oreos import pathplan``) so
    tests can inject a fake via ``sys.modules['server.oreos.pathplan']``."""
    try:
        return importlib.import_module("server.oreos.pathplan")
    except ImportError as exc:
        raise PathplanUnavailable(str(exc)) from exc


def fit_floor(points: np.ndarray, up_vec: np.ndarray, *, thresh: float, seed: int = 0):
    """Fit the floor through the W4 seam -> the pathplan ``FloorPlane`` object
    (``point``/``normal``/``u_axis``/``v_axis`` attributes, exp25 shape).

    Raises :class:`PathplanUnavailable` when W4 hasn't merged, and re-raises the
    port's ``ValueError`` on degenerate geometry (route maps it to 422
    ``no_floor``)."""
    pathplan = load_pathplan()
    rng = np.random.default_rng(seed)
    return pathplan.fit_floor_plane(points, up_vec, rng, ransac_thresh=float(thresh))


# ---------------------------------------------------------------------------
# up vector
# ---------------------------------------------------------------------------


def estimate_up(
    points: np.ndarray,
    poses: Optional[np.ndarray] = None,
    up_override: Optional[Any] = None,
) -> tuple[np.ndarray, str]:
    """World-up unit vector + its source, per the contract's source priority.

    ``up_override`` (any 3-vector) -> ``"user_override"``;
    ``poses`` (N,4,4) OpenCV c2w -> mean camera-down, ``up = -mean(c2w[:,:3,1])``
    -> ``"poses"``; else the align.py dominant-plane heuristic -> ``"heuristic"``.
    """
    if up_override is not None:
        arr = np.asarray(up_override, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3 or not np.isfinite(arr).all():
            raise ValueError("up_override must be a finite 3-vector")
        return _unit(arr), "user_override"

    if poses is not None:
        p = np.asarray(poses, dtype=np.float64)
        if p.ndim == 3 and p.shape[1:] == (4, 4) and p.shape[0] >= 1:
            downs = p[:, :3, 1]  # OpenCV c2w: camera +y column points world-DOWN
            g = downs.mean(axis=0)
            if float(np.linalg.norm(g)) > _EPS:
                return _unit(-g), "poses"

    from server.export.isaac.align import estimate_gravity  # numpy-only, broker-safe

    up, _mask, _source = estimate_gravity(np.asarray(points, dtype=np.float64))
    return _unit(up), "heuristic"


# ---------------------------------------------------------------------------
# wall candidates
# ---------------------------------------------------------------------------


def _plane_basis_for_wall(normal: np.ndarray, up_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane (u, v) basis for a near-vertical plane: ``u`` horizontal along
    the wall, ``v`` the in-plane vertical (up projected into the plane), so
    ``uv_bounds`` read as (width, height) and ``[u, v, n]`` is right-handed."""
    u = np.cross(up_vec, normal)
    nu = float(np.linalg.norm(u))
    if nu < _EPS:  # degenerate: plane ~horizontal (caller filters these out)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(normal @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(ref, normal)
        nu = float(np.linalg.norm(u))
    u = u / nu
    v = np.cross(normal, u)
    return u, v / np.linalg.norm(v)


def _occupancy_fill(inliers: np.ndarray, point: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> float:
    """Fraction of occupied cells in a count-adaptive uv grid over the inlier
    bbox — the "filled rectangle vs thin bands" wall discriminator (see
    ``detect_walls``). Grid resolution scales with the inlier count so sparse
    real walls aren't starved (>= ~6 expected points per cell when filled)."""
    rel = inliers - point
    u = rel @ u_axis
    v = rel @ v_axis
    n = u.shape[0]
    if n < 8:
        return 0.0
    if (u.max() - u.min()) < _EPS or (v.max() - v.min()) < _EPS:
        return 0.0
    bins = int(np.clip(np.sqrt(n / 6.0), 6, 24))
    hist, _, _ = np.histogram2d(u, v, bins=bins)
    return float((hist > 0).mean())


def _uv_bounds(
    inliers: np.ndarray,
    point: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> list[list[float]]:
    """Percentile-clipped in-plane extent of the inlier set (resists the few
    stragglers RANSAC always admits at a relative threshold)."""
    rel = inliers - point
    u = rel @ u_axis
    v = rel @ v_axis
    return [
        [float(np.percentile(u, lo_pct)), float(np.percentile(u, hi_pct))],
        [float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))],
    ]


def _candidate(
    plane_id: str,
    kind: str,
    point: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    uv_bounds: list[list[float]],
    thickness: float,
    inlier_count: int,
) -> dict[str, Any]:
    return {
        "id": plane_id,
        "kind": kind,
        "point": [float(x) for x in point],
        "normal": [float(x) for x in normal],
        "u_axis": [float(x) for x in u_axis],
        "v_axis": [float(x) for x in v_axis],
        "uv_bounds": uv_bounds,
        "thickness": float(thickness),
        "inlier_count": int(inlier_count),
    }


def detect_walls(
    points: np.ndarray,
    up_vec: np.ndarray,
    *,
    max_planes: int = MAX_WALL_PLANES,
    thresh: Optional[float] = None,
    thresh_frac: float = 0.01,
    n_iter: int = 600,
    seed: int = 0,
    max_ransac_points: int = 100_000,
    vertical_tol: float = WALL_VERTICAL_TOL,
    min_inliers: Optional[int] = None,
    min_aspect: float = 0.12,
    min_fill: float = 0.5,
) -> list[dict[str, Any]]:
    """Sequential RANSAC wall candidates -> list of candidate dicts (see
    ``_candidate``), strongest (most inliers) first, ids ``wall_0..``.

    Per round: hypothesis planes from 3 random points are REJECTED unless
    ``|normal . up| < vertical_tol`` (so the dominant floor/desk planes can
    never win a wall round); the best hypothesis is SVD-refit on its inlier set
    and re-scored against the full working cloud (align.py pattern, incl. the
    ``full_matrices=False`` and non-finite traps); accepted inliers are removed
    (at 2x thresh, so a wall's fringe can't re-seed round k+1) and the loop
    continues until ``max_planes``, an under-count round, or exhaustion.

    ``thresh`` defaults to ``thresh_frac`` x the FULL cloud's bbox diagonal,
    computed ONCE — never per-round (the residual bbox shrinks as walls are
    removed, which would silently tighten later rounds). ``min_inliers``
    defaults to ``max(200, 0.5% of the working cloud)``.

    Two phantom filters (both measured on the synthetic-room fixture, where a
    vertical plane slicing obliquely through the table slab + floor + noise
    out-counts nothing real but still clears ``min_inliers``):

    - ``min_aspect``: the refit inlier set's singular-value ratio ``s1/s0``
      must show 2D extension — rejects pure quasi-1D stripes.
    - ``min_fill``: 2D occupancy fill of the inlier footprint (fraction of
      occupied cells in a count-adaptive uv grid over the inlier bbox). A real
      wall is a FILLED rectangle of points (measured ~0.9+ on the fixture); a
      phantom slab through table+floor+noise is thin bands plus scatter
      (~0.3). Absolute and per-candidate, so unevenly captured scenes (dense
      near wall, sparse far wall) cannot knock out real walls the way a
      relative density test would.

    Candidates are exactly that — the client shows hover-tint chips and a human
    confirms. This function never auto-applies anything.
    """
    pts = _finite_points(points)
    n_total = pts.shape[0]
    if n_total < 3:
        return []
    up = _unit(up_vec)
    if thresh is None:
        thresh = max(thresh_frac * _bbox_diagonal(pts), _EPS)
    thresh = float(thresh)
    if min_inliers is None:
        min_inliers = max(200, int(0.005 * n_total))

    rng = np.random.default_rng(seed)
    working = np.ones(n_total, dtype=bool)
    walls: list[dict[str, Any]] = []
    max_rounds = max_planes * 3  # bounded even when stripe/constraint rejections burn rounds

    for _round in range(max_rounds):
        if len(walls) >= max_planes:
            break
        work_idx = np.flatnonzero(working)
        work = pts[work_idx]
        m_work = work.shape[0]
        if m_work < max(3, min_inliers):
            break

        # Deterministic subsample cap for the O(N)-per-iteration scoring loop
        # (align.py scale-safety pattern; refit still uses the full working set).
        if m_work > max_ransac_points:
            sub = np.random.default_rng(0).choice(m_work, size=max_ransac_points, replace=False)
            scan = work[sub]
        else:
            scan = work
        m_scan = scan.shape[0]

        best_normal: Optional[np.ndarray] = None
        best_d = 0.0
        best_count = 0
        for _ in range(int(n_iter)):
            idx = rng.choice(m_scan, size=3, replace=False)
            p0, p1, p2 = scan[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            nn = float(np.linalg.norm(normal))
            if nn < _EPS:
                continue  # degenerate (collinear) triple
            normal = normal / nn
            if abs(float(normal @ up)) >= vertical_tol:
                continue  # not wall-like — the F5 constraint, applied at hypothesis time
            d = -float(normal @ p0)
            dist = np.abs(scan @ normal + d)
            count = int((dist < thresh).sum())
            if count > best_count:
                best_count = count
                best_normal = normal
                best_d = d

        if best_normal is None or best_count < 3:
            break  # no vertical structure left

        # SVD refit on the FULL working set's inliers (not the scoring subsample).
        inlier_mask = np.abs(work @ best_normal + best_d) < thresh
        inliers = work[inlier_mask]
        if inliers.shape[0] < 3:
            break
        centroid = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vh[-1] / np.linalg.norm(vh[-1])
        d = -float(normal @ centroid)
        refit_mask = np.abs(work @ normal + d) < thresh
        refit_inliers = work[refit_mask]
        count = int(refit_mask.sum())

        # Remove the plane's slab either way (2x thresh: fringe suppression) so a
        # rejected candidate can't win every remaining round.
        remove_mask = np.abs(work @ normal + d) < (2.0 * thresh)
        working[work_idx[remove_mask]] = False

        if count < min_inliers:
            break  # remaining structure is below the candidate bar
        if abs(float(normal @ up)) >= vertical_tol:
            continue  # refit drifted out of the wall band — drop, keep scanning
        svals = np.linalg.svd(refit_inliers - refit_inliers.mean(axis=0), compute_uv=False)
        if float(svals[0]) < _EPS or float(svals[1] / svals[0]) < min_aspect:
            continue  # quasi-1D stripe (oblique cut through horizontal structure) — not a wall
        u_axis, v_axis = _plane_basis_for_wall(normal, up)
        if _occupancy_fill(refit_inliers, centroid, u_axis, v_axis) < min_fill:
            continue  # bands + scatter, not a filled surface — phantom slab, not a wall

        # Orient the normal toward the interior (the side holding more points).
        side = np.sign((pts - centroid) @ normal).sum()
        if side < 0:
            normal = -normal
        u_axis, v_axis = _plane_basis_for_wall(normal, up)
        walls.append(
            _candidate(
                plane_id="",  # ids assigned after the strongest-first sort
                kind="wall",
                point=centroid,
                normal=normal,
                u_axis=u_axis,
                v_axis=v_axis,
                uv_bounds=_uv_bounds(refit_inliers, centroid, u_axis, v_axis),
                thickness=2.0 * thresh,
                inlier_count=count,
            )
        )

    walls.sort(key=lambda w: -w["inlier_count"])
    for i, wall in enumerate(walls):
        wall["id"] = f"wall_{i}"
    return walls


def floor_candidate(points: np.ndarray, floor_plane: Any, thresh: float) -> dict[str, Any]:
    """The pathplan floor as a candidate dict (same shape as walls, ``id:
    "floor"``): inliers re-selected at ``thresh`` for the count + uv bounds,
    basis taken from the FloorPlane verbatim (never re-fit — contract rule)."""
    pts = _finite_points(points)
    point = np.asarray(floor_plane.point, dtype=np.float64).reshape(3)
    normal = _unit(np.asarray(floor_plane.normal, dtype=np.float64))
    u_axis = _unit(np.asarray(floor_plane.u_axis, dtype=np.float64))
    v_axis = _unit(np.asarray(floor_plane.v_axis, dtype=np.float64))
    dist = np.abs((pts - point) @ normal)
    inlier_mask = dist < float(thresh)
    inliers = pts[inlier_mask] if int(inlier_mask.sum()) >= 3 else pts
    return _candidate(
        plane_id="floor",
        kind="floor",
        point=point,
        normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        uv_bounds=_uv_bounds(inliers, point, u_axis, v_axis),
        thickness=2.0 * float(thresh),
        inlier_count=int(inlier_mask.sum()),
    )


# ---------------------------------------------------------------------------
# floor patch (F6 hole-fill)
# ---------------------------------------------------------------------------


class InsufficientFloorContext(ValueError):
    """Not enough floor inliers around the footprint to sample colors from."""


def _quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z) — the splat.ply rot_0..3 order."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def obb_corners(center: np.ndarray, extents: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The 8 world-frame corners of an OBB in the contract convention
    (``extents`` FULL lengths, ``rotation`` columns = axes). ``(8, 3)``."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.asarray(extents, dtype=np.float64).reshape(3) / 2.0
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    return c + (signs * h) @ R.T


def synthesize_floor_patch(
    positions: np.ndarray,
    colors01: np.ndarray,
    floor_plane: Any,
    obb_center: np.ndarray,
    obb_extents: np.ndarray,
    obb_rotation: np.ndarray,
    *,
    n_gaussians: int = 2000,
    seed: int = 0,
    annulus_margin: float = 0.35,
    height_band: Optional[float] = None,
    min_context_points: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Synthesize a floor patch under a (hidden) object's OBB footprint.

    -> ``(fields, info)`` where ``fields`` is a ``splat_io`` field dict
    (``serialize_splat_ply``-ready, colors SH-DC encoded) and ``info`` carries
    the numbers for the honesty envelope (annulus stats, footprint uv).

    Method (features.md F6 "floor patch"): project the OBB footprint into the
    floor plane's (u, v) basis; collect real floor inliers (``|height| <
    height_band``) in an annulus ``annulus_margin`` x footprint-size around it
    (widening once, then falling back to all floor inliers if the annulus is
    empty — e.g. the object sat in a corner); sample ``n_gaussians`` uniform
    positions in the footprint with small height jitter; color each from its
    nearest annulus neighbors (k=4 mean + tiny jitter — keeps the real floor's
    color gradient); orient flat discs in the floor plane. The patch NEVER
    claims to be captured background: callers must persist it with
    ``generated: true`` and the fixed caveat copy.
    """
    from scipy.spatial import cKDTree  # broker-safe (scipy is a base dep)

    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors01, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError("positions/colors length mismatch")
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]
    if pts.shape[0] < min_context_points:
        raise InsufficientFloorContext("cloud too small to sample floor context")

    point = np.asarray(floor_plane.point, dtype=np.float64).reshape(3)
    normal = _unit(np.asarray(floor_plane.normal, dtype=np.float64))
    u_axis = _unit(np.asarray(floor_plane.u_axis, dtype=np.float64))
    v_axis = _unit(np.asarray(floor_plane.v_axis, dtype=np.float64))

    if height_band is None:
        height_band = max(0.01 * _bbox_diagonal(pts), _EPS)
    height_band = float(height_band)

    # Footprint rect in floor uv.
    corners = obb_corners(obb_center, obb_extents, obb_rotation)
    rel_c = corners - point
    cu, cv = rel_c @ u_axis, rel_c @ v_axis
    u0, u1 = float(cu.min()), float(cu.max())
    v0, v1 = float(cv.min()), float(cv.max())
    du, dv = max(u1 - u0, _EPS), max(v1 - v0, _EPS)

    # Floor inliers in (u, v, h).
    rel = pts - point
    h = rel @ normal
    floor_mask = np.abs(h) < height_band
    fu = (rel @ u_axis)[floor_mask]
    fv = (rel @ v_axis)[floor_mask]
    fh = h[floor_mask]
    fcol = cols[floor_mask]
    if fu.shape[0] < min_context_points:
        raise InsufficientFloorContext(
            f"only {fu.shape[0]} floor inliers within height band {height_band:.4g}"
        )

    def annulus_mask(margin_frac: float) -> np.ndarray:
        m = margin_frac * max(du, dv)
        inside_outer = (fu >= u0 - m) & (fu <= u1 + m) & (fv >= v0 - m) & (fv <= v1 + m)
        inside_foot = (fu >= u0) & (fu <= u1) & (fv >= v0) & (fv <= v1)
        return inside_outer & ~inside_foot

    ann = annulus_mask(annulus_margin)
    widened = False
    if int(ann.sum()) < min_context_points:
        ann = annulus_mask(annulus_margin * 3.0)
        widened = True
    fallback_all_floor = False
    if int(ann.sum()) < min_context_points:
        ann = np.ones(fu.shape[0], dtype=bool)  # corner case: sample the whole floor
        fallback_all_floor = True

    au, av, ah, acol = fu[ann], fv[ann], fh[ann], fcol[ann]

    n = int(n_gaussians)
    if n < 1:
        raise ValueError("n_gaussians must be >= 1")
    rng = np.random.default_rng(seed)
    su = rng.uniform(u0, u1, size=n)
    sv = rng.uniform(v0, v1, size=n)

    tree = cKDTree(np.stack([au, av], axis=1))
    k = min(4, au.shape[0])
    _dists, nn_idx = tree.query(np.stack([su, sv], axis=1), k=k)
    if k == 1:
        nn_idx = nn_idx[:, None]
    sample_col = acol[nn_idx].mean(axis=1)
    sample_col = np.clip(sample_col + rng.normal(0.0, 0.02, size=sample_col.shape), 0.0, 1.0)
    # Follow the real floor's local height (annulus mean) + a small jitter, so a
    # tilted-vs-plane floor region doesn't get a visibly offset patch.
    sample_h = ah[nn_idx].mean(axis=1) + rng.normal(0.0, 0.15 * height_band, size=n)

    world = (
        point[None, :]
        + su[:, None] * u_axis[None, :]
        + sv[:, None] * v_axis[None, :]
        + sample_h[:, None] * normal[None, :]
    )

    # Flat discs: in-plane sigma ~ sampling cell (with overlap), thin along the normal.
    cell = float(np.sqrt((du * dv) / n))
    sigma_uv = max(0.9 * cell, _EPS)
    sigma_n = max(0.15 * cell, _EPS)
    quat = _quat_wxyz_from_matrix(np.stack([u_axis, v_axis, normal], axis=1))
    opacity_logit = float(np.log(0.9 / 0.1))

    f_dc = (sample_col - 0.5) / _SH_C0
    zeros = np.zeros(n, dtype=np.float32)
    fields: dict[str, np.ndarray] = {
        "x": world[:, 0].astype(np.float32),
        "y": world[:, 1].astype(np.float32),
        "z": world[:, 2].astype(np.float32),
        "nx": zeros,
        "ny": zeros,
        "nz": zeros,
        "f_dc_0": f_dc[:, 0].astype(np.float32),
        "f_dc_1": f_dc[:, 1].astype(np.float32),
        "f_dc_2": f_dc[:, 2].astype(np.float32),
        "opacity": np.full(n, opacity_logit, dtype=np.float32),
        "scale_0": np.full(n, np.log(sigma_uv), dtype=np.float32),
        "scale_1": np.full(n, np.log(sigma_uv), dtype=np.float32),
        "scale_2": np.full(n, np.log(sigma_n), dtype=np.float32),
        "rot_0": np.full(n, quat[0], dtype=np.float32),
        "rot_1": np.full(n, quat[1], dtype=np.float32),
        "rot_2": np.full(n, quat[2], dtype=np.float32),
        "rot_3": np.full(n, quat[3], dtype=np.float32),
    }
    info: dict[str, Any] = {
        "n_gaussians": n,
        "n_annulus_points": int(au.shape[0]),
        "annulus_widened": widened,
        "annulus_fallback_all_floor": fallback_all_floor,
        "footprint_uv": [[u0, u1], [v0, v1]],
        "height_band": height_band,
        "cell_size": cell,
    }
    return fields, info
