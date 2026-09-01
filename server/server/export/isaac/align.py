"""Metric + gravity alignment: SLAM world -> Isaac world (metric, +Z up, floor at z=0).

The SLAM world frame is **up-to-scale** and **not gravity-aligned** (``docs/gotchas.md``); Isaac
Sim / Isaac Lab needs a metric, Z-up stage with the floor at the origin. This module:

1. estimates the gravity (up) vector from the dominant floor plane (RANSAC + SVD refit),
   sign-disambiguated so it points toward the cameras (which sit above the floor);
2. resolves a metric scale from a user-supplied factor or a reference object's known size;
3. composes the 4x4 ``T_align`` that maps ``p_isaac = T_align @ [p_slam, 1]``.

**Pure numpy/scipy** (no open3d/pxr) so it is fully unit-testable in any env. The geometric
risk lives here — the dominant plane could be a wall/table rather than the floor, and the
sign could flip — so every estimate is reported with a confidence + provenance in the
returned :class:`AlignmentResult` (surfaced in ``isaac/manifest.json``), and an ``up_override``
escape hatch is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _bbox_diagonal(points: np.ndarray) -> float:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def aabb_corners(center: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box given center + full extent (size). ``(8,3)``."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.asarray(extent, dtype=np.float64).reshape(3) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    return c + signs * h


# ---------------------------------------------------------------------------
# plane fitting
# ---------------------------------------------------------------------------

def fit_plane_ransac(
    points: np.ndarray,
    *,
    thresh: Optional[float] = None,
    thresh_frac: float = 0.01,
    n_iter: int = 1000,
    seed: int = 0,
    max_ransac_points: int = 100_000,
) -> tuple:
    """RANSAC-fit the dominant plane -> ``(normal (3,), d, inlier_mask (N,))``.

    Plane is ``normal . x + d = 0`` with ``||normal|| = 1``. ``thresh`` defaults to
    ``thresh_frac`` of the cloud's bbox diagonal (the SLAM cloud is up-to-scale, so an
    absolute metric threshold is meaningless). The best inlier set is refit via SVD.

    ``inlier_mask`` is always returned over the **original** ``points`` (length ``N``);
    non-finite rows are ``False`` (they are dropped before fitting so a single inf/nan can't
    corrupt the SVD refit — see below).

    **Scale safety.** On multi-million-point clouds, re-scoring every point on each of
    ``n_iter`` iterations is billions of point-plane evaluations (timeout risk for the Isaac
    export). When more than ``max_ransac_points`` finite points are present, the hypothesis +
    inlier-scoring loop runs on a fixed, deterministically-seeded random subsample of that
    size; the winning plane is then refit (SVD) against the **full** finite cloud, so the
    returned plane still reflects every point. For ``N <= max_ransac_points`` (and an
    all-finite cloud) the fit is bit-for-bit the pre-cap behavior.
    """
    pts_all = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_all = pts_all.shape[0]

    # (1) Drop non-finite rows up front: one inf/nan point otherwise dominates the SVD refit
    # and blows the plane up. The returned mask stays full-length (False at the dropped rows)
    # so callers indexing the *original* cloud (estimate_gravity / compute_alignment) still work.
    finite = np.isfinite(pts_all).all(axis=1)
    pts = pts_all[finite] if not finite.all() else pts_all
    n = pts.shape[0]
    if n < 3:
        # Degenerate input (empty / <3 finite points): documented fallback = +Z, no inliers.
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n_all, dtype=bool)

    if thresh is None:
        diag = _bbox_diagonal(pts)
        thresh = max(thresh_frac * diag, _EPS)

    # (2) Cap the O(N) per-iteration scoring work with a deterministic subsample. A separate,
    # locally-seeded RNG picks it (does NOT reseed global numpy state); the RANSAC hypothesis
    # RNG below still uses ``seed``. For a small cloud we score everything, unchanged.
    if n > max_ransac_points:
        sub_idx = np.random.default_rng(0).choice(n, size=max_ransac_points, replace=False)
        scan = pts[sub_idx]
    else:
        scan = pts
    m = scan.shape[0]

    rng = np.random.default_rng(seed)
    best_normal: Optional[np.ndarray] = None
    best_d = 0.0
    best_count = 0

    for _ in range(int(n_iter)):
        idx = rng.choice(m, size=3, replace=False)
        p0, p1, p2 = scan[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < _EPS:  # degenerate (collinear) sample
            continue
        normal = normal / nn
        d = -float(normal @ p0)
        dist = np.abs(scan @ normal + d)
        count = int((dist < thresh).sum())
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d

    if best_normal is None or best_count < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n_all, dtype=bool)

    # SVD refit on the best inlier set (more accurate than the 3-point seed). Inliers are
    # gathered from the FULL finite cloud — not the subsample — so the plane reflects all points.
    full_inlier = np.abs(pts @ best_normal + best_d) < thresh
    inliers = pts[full_inlier]
    if inliers.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n_all, dtype=bool)
    centroid = inliers.mean(axis=0)
    # full_matrices=False keeps U at (N,3) instead of (N,N) — the latter is a ~137 TiB
    # allocation for a multi-million-point cloud. Vh stays (3,3) either way, so the
    # plane normal is still its last row.
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = _unit(vh[-1])  # smallest singular direction = plane normal
    d = -float(normal @ centroid)
    # Recompute inliers against the refined plane (full finite cloud), then lift the mask back
    # onto the original (pre-filter) indexing so the returned mask is length N.
    mask = np.abs(pts @ normal + d) < thresh
    if finite.all():
        full_mask = mask
    else:
        full_mask = np.zeros(n_all, dtype=bool)
        full_mask[finite] = mask
    return normal, d, full_mask


# ---------------------------------------------------------------------------
# gravity / up vector
# ---------------------------------------------------------------------------

def estimate_gravity(
    points: np.ndarray,
    cam_centers: Optional[np.ndarray] = None,
    *,
    up_override: Optional[np.ndarray] = None,
    **ransac_kwargs,
) -> tuple:
    """Estimate the SLAM-frame up (gravity) unit vector -> ``(up (3,), inlier_mask, source)``.

    With ``up_override`` set, that direction is used verbatim (``source="user_override"``,
    no inlier mask). Otherwise the dominant plane is taken as the floor and its normal is
    oriented to point *up*: toward the camera centroid if ``cam_centers`` is given (cameras
    sit above the floor), else toward the side holding most of the cloud.
    """
    if up_override is not None:
        return _unit(up_override), None, "user_override"

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normal, d, mask = fit_plane_ransac(pts, **ransac_kwargs)

    # Orient the normal so it points "up".
    if cam_centers is not None and len(cam_centers) > 0:
        cams = np.asarray(cam_centers, dtype=np.float64).reshape(-1, 3)
        plane_pt = pts[mask].mean(axis=0) if mask.any() else pts.mean(axis=0)
        if normal @ (cams.mean(axis=0) - plane_pt) < 0:
            normal = -normal
    else:
        # Most scene points sit above the floor -> up is the majority side.
        signed = pts @ normal + d
        if float(np.sign(signed).sum()) < 0:
            normal = -normal

    return _unit(normal), mask, "floor_plane"


def rotation_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix ``R`` such that ``R @ unit(a) == unit(b)`` (3x3)."""
    a = _unit(a)
    b = _unit(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-8:  # already aligned
        return np.eye(3)
    if c < -1.0 + 1e-8:  # anti-parallel: 180° about any axis ⟂ a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _unit(axis)
        # Rodrigues with theta = pi  ->  R = 2 axis axis^T - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ---------------------------------------------------------------------------
# metric scale
# ---------------------------------------------------------------------------

def resolve_scale(
    scale: Optional[float] = None,
    ref_extent_slam: Optional[float] = None,
    ref_meters: Optional[float] = None,
    ref_query: Optional[str] = None,
) -> tuple:
    """Resolve the SLAM->meters scale -> ``(scale, source, metric)``.

    Priority: an explicit ``scale`` factor; else back it out from a reference object's known
    real size (``ref_meters / ref_extent_slam``); else fall back to ``1.0`` and ``metric=False``
    (the export is then honestly flagged as still up-to-scale).
    """
    if scale is not None and float(scale) > 0:
        return float(scale), "user:factor", True
    if (
        ref_extent_slam is not None
        and ref_meters is not None
        and float(ref_extent_slam) > _EPS
        and float(ref_meters) > 0
    ):
        s = float(ref_meters) / float(ref_extent_slam)
        tag = f"user:ref_object:{ref_query}" if ref_query else "user:ref_object"
        return s, tag, True
    return 1.0, "none", False


# ---------------------------------------------------------------------------
# full alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Output of :func:`compute_alignment` — the transform plus its provenance."""

    transform: np.ndarray  # 4x4: p_isaac = transform @ [p_slam, 1]
    rotation: np.ndarray  # 3x3 orthonormal (gravity rotation, no scale)
    scale: float
    scale_source: str
    metric: bool
    up_vector_slam: np.ndarray
    gravity_aligned: bool
    gravity_source: str
    floor_inlier_fraction: float
    recentered_xy: bool = True

    @property
    def translation(self) -> np.ndarray:
        return self.transform[:3, 3].copy()


def compute_alignment(
    positions: np.ndarray,
    cam_centers: Optional[np.ndarray] = None,
    *,
    scale: Optional[float] = None,
    ref_box: Optional[tuple] = None,  # (center (3,), extent (3,)) in SLAM world, gravity height
    ref_meters: Optional[float] = None,
    ref_query: Optional[str] = None,
    up_override: Optional[np.ndarray] = None,
    recenter_xy: bool = True,
    **ransac_kwargs,
) -> AlignmentResult:
    """Estimate gravity + scale and compose ``T_align`` (SLAM world -> Isaac world).

    ``ref_box`` is a reference object's SLAM-frame AABB; its **height along the recovered up
    axis** (not a raw extent component — the box is axis-aligned in the SLAM frame, not the
    gravity frame) is paired with ``ref_meters`` to back out scale.
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]

    up_unit, inlier_mask, gravity_source = estimate_gravity(
        pts, cam_centers, up_override=up_override, **ransac_kwargs
    )
    rot = rotation_align_vectors(up_unit, np.array([0.0, 0.0, 1.0]))

    # Reference-object height must be measured in the gravity-aligned frame.
    ref_extent_slam = None
    if ref_box is not None:
        center, extent = ref_box
        rotated_corners = (rot @ aabb_corners(center, extent).T).T
        ref_extent_slam = float(rotated_corners[:, 2].max() - rotated_corners[:, 2].min())

    scale_val, scale_source, metric = resolve_scale(
        scale, ref_extent_slam, ref_meters, ref_query
    )

    # M maps SLAM -> rotated+scaled (pre-translation) coordinates.
    M = scale_val * rot
    q = (M @ pts.T).T if n else pts.reshape(0, 3)

    # Drop the floor to z = 0 (median over floor inliers if we have them, else the min z).
    if inlier_mask is not None and bool(inlier_mask.any()):
        floor_z = float(np.median(q[inlier_mask, 2]))
        inlier_frac = float(inlier_mask.sum()) / float(max(n, 1))
    elif n:
        floor_z = float(q[:, 2].min())
        inlier_frac = 0.0
    else:
        floor_z = 0.0
        inlier_frac = 0.0

    t = np.zeros(3, dtype=np.float64)
    t[2] = -floor_z
    if recenter_xy and n:
        t[0:2] = -q[:, 0:2].mean(axis=0)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = M
    T[:3, 3] = t

    return AlignmentResult(
        transform=T,
        rotation=rot,
        scale=scale_val,
        scale_source=scale_source,
        metric=metric,
        up_vector_slam=up_unit,
        gravity_aligned=(gravity_source == "user_override") or (inlier_frac > 0.0),
        gravity_source=gravity_source,
        floor_inlier_fraction=inlier_frac,
        recentered_xy=bool(recenter_xy and n),
    )


# ---------------------------------------------------------------------------
# applying the transform
# ---------------------------------------------------------------------------

def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``T`` to ``(N,3)`` points -> ``(N,3)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def transform_camera_pose(
    result: "AlignmentResult", pose_cam2world: np.ndarray
) -> np.ndarray:
    """Map a cam->world pose into the Isaac frame, keeping the rotation orthonormal.

    The camera *position* gets the full affine (scale + rotate + translate); the camera
    *orientation* gets only the gravity rotation (scaling a rotation would shear it).
    """
    P = np.asarray(pose_cam2world, dtype=np.float64).reshape(4, 4)
    R_c = P[:3, :3]
    t_c = P[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = result.rotation @ R_c
    out[:3, 3] = result.scale * (result.rotation @ t_c) + result.translation
    return out
