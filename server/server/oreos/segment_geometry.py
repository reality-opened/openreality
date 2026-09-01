"""W3 segmentation geometry — mask/box → 3D gaussian subset → OBB → asset fit.

Spec: platform docs/demo-2026-07/design/features.md §0.2 (mask lift) + F3
(fit-to-OBB + EXP-20 quality gate), WORLD-TRANSFORM-CONTRACT.md (all
coordinates world frame; OBB = ``{center, extents FULL lengths, rotation
3x3 columns=axes}`` — the ObjectLayerOBB convention).

Owned by W3 (routes_sam3d.py). Deliberately SEPARATE from W2's minimal
``geometry.py``/``measure.py`` (file-separation rule) — nothing here is
imported by other workstreams' modules.

Dependency policy: numpy everywhere; scipy imported lazily INSIDE the two
functions that need it (``binary_dilation`` / ``cKDTree``) so the fit+gate
half of the module also runs on the Modal SAM-3D app without scipy. The OBB
math is a spec-read reimplementation — core/ is never imported (BUILD-PLAN
"never touches core/").
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# §0.2 defaults (load-bearing numbers from the spec — change only with the spec).
MASK_DILATE_PX = 3
ZBUF_TAU = 0.08            # relative z-band — robust across unanchored gauges
ZBUF_DOWNSCALE = 4         # z-buffer at H/4 x W/4
KDTREE_K = 8
KDTREE_SIGMA = 2.0
MAX_LIFT_POINTS = 2_000_000  # stride-subsample above this (spec: <1 s at <=2M pts)

# F3 quality gate (EXP-20 numbers).
GATE_CENTROID_FRAC = 0.20      # centroid offset <= 20% of OBB diagonal
GATE_VOL_RATIO = (0.2, 3.0)    # fitted-volume / OBB-volume band

# F3 fallback: fewer lifted points than this => OBB-only low tier, completion off.
MIN_LIFT_POINTS = 50


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _as_np(a: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def _right_handed(R: np.ndarray) -> np.ndarray:
    """Flip the last column if needed so det(R) = +1 (columns stay unit axes)."""
    R = R.copy()
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def obb_to_dict(center: np.ndarray, extents: np.ndarray, R: np.ndarray) -> dict:
    """Wire shape (ObjectLayerOBB): center, extents FULL lengths, rotation columns=axes."""
    return {
        "center": [float(v) for v in np.asarray(center).reshape(3)],
        "extents": [float(v) for v in np.asarray(extents).reshape(3)],
        "rotation": [[float(v) for v in row] for row in np.asarray(R).reshape(3, 3)],
    }


def stride_subsample(n: int, cap: int = MAX_LIFT_POINTS) -> np.ndarray:
    """Deterministic index subsample: every k-th point so at most ``cap`` survive."""
    if n <= cap:
        return np.arange(n)
    step = int(np.ceil(n / cap))
    return np.arange(0, n, step)


# ---------------------------------------------------------------------------
# §0.2 mask lift: 2D mask -> world-point subset
# ---------------------------------------------------------------------------

def lift_mask_to_points(
    points_world: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    c2w: Any,
    *,
    dilate_px: int = MASK_DILATE_PX,
    tau: float = ZBUF_TAU,
    zbuf_downscale: int = ZBUF_DOWNSCALE,
    max_points: int = MAX_LIFT_POINTS,
) -> np.ndarray:
    """Indices (into ``points_world``) of points that project inside the dilated
    mask AND survive the z-buffer occlusion test (§0.2 steps 1–3).

    ``points_world``: (N,3) world-frame positions (cloud.npz or splat means).
    ``mask``: (H,W) bool on the prompt image. ``intrinsics``: [fx, fy, cx, cy]
    on that image's pixel grid. ``c2w``: 4x4 OpenCV camera-to-world (the
    trajectory.npz convention — x right, y down, z forward).
    """
    P = _as_np(points_world, np.float32).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics).reshape(-1)[:4]]
    c2w = _as_np(c2w).reshape(4, 4)

    sub = stride_subsample(P.shape[0], max_points)
    Ps = P[sub]

    # 1. world -> camera (OpenCV), keep points in front of the camera.
    R_wc = c2w[:3, :3].T
    t_wc = -R_wc @ c2w[:3, 3]
    pc = Ps @ R_wc.T + t_wc
    z = pc[:, 2]
    front = z > 1e-6
    if not front.any():
        return np.empty(0, dtype=np.int64)

    # 2. project; keep in-image points inside the mask dilated ``dilate_px``.
    u = fx * pc[:, 0] / np.where(front, z, 1.0) + cx
    v = fy * pc[:, 1] / np.where(front, z, 1.0) + cy
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    in_img = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not in_img.any():
        return np.empty(0, dtype=np.int64)

    if dilate_px > 0:
        from scipy.ndimage import binary_dilation  # lazy: broker-safe numpy+scipy only

        mask_d = binary_dilation(mask, iterations=int(dilate_px))
    else:
        mask_d = mask

    in_mask = np.zeros(Ps.shape[0], dtype=bool)
    in_mask[in_img] = mask_d[vi[in_img], ui[in_img]]

    # 3. occlusion test (load-bearing): z-buffer from ALL projected points at
    # (H/zbuf_downscale x W/zbuf_downscale); keep candidates with z <= zbuf*(1+tau).
    s = max(1, int(zbuf_downscale))
    Hs, Ws = (H + s - 1) // s, (W + s - 1) // s
    zbuf = np.full((Hs, Ws), np.inf, dtype=np.float32)
    cell_v = vi[in_img] // s
    cell_u = ui[in_img] // s
    np.minimum.at(zbuf, (cell_v, cell_u), z[in_img].astype(np.float32))

    visible = np.zeros(Ps.shape[0], dtype=bool)
    cand = in_mask
    if cand.any():
        cz = zbuf[vi[cand] // s, ui[cand] // s]
        visible[cand] = z[cand] <= cz * (1.0 + float(tau))

    return sub[np.flatnonzero(visible)]


def filter_outliers_kdtree(
    points: np.ndarray, k: int = KDTREE_K, sigma: float = KDTREE_SIGMA
) -> np.ndarray:
    """§0.2 step 4: cKDTree k-mean-neighbor-distance filter at ``sigma`` std.
    Returns indices (into ``points``) of the inliers. No open3d on the broker."""
    P = _as_np(points, np.float64).reshape(-1, 3)
    n = P.shape[0]
    if n <= k + 1:
        return np.arange(n)
    from scipy.spatial import cKDTree  # lazy: broker-safe

    dists, _ = cKDTree(P).query(P, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    keep = mean_d <= mean_d.mean() + sigma * mean_d.std()
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# §0.2 step 5: gravity-aware PCA OBB (spec-read reimplementation; core/ never imported)
# ---------------------------------------------------------------------------

def pca_obb(
    points: np.ndarray, up: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oriented bounding box of a point set: ``(center, extents FULL, R columns=axes)``.

    ``up`` given → gravity-constrained: one axis snapped to ``up``, in-plane PCA
    for the other two. ``up`` None → plain 3D PCA. Axes ordered by descending
    extent; right-handed."""
    P = _as_np(points, np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        raise ValueError("pca_obb needs at least one point")
    c0 = P.mean(axis=0)
    Q = P - c0

    if up is not None:
        u = _as_np(up).reshape(3)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            return pca_obb(P, None)
        u = u / nu
        # in-plane PCA: project out the up component, eigen in the plane.
        Qp = Q - np.outer(Q @ u, u)
        cov = Qp.T @ Qp
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        a1 = evecs[:, order[0]]
        a1 = a1 - (a1 @ u) * u
        n1 = np.linalg.norm(a1)
        if n1 < 1e-9:  # degenerate (all points on the up axis)
            a1 = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            a1 = a1 - (a1 @ u) * u
            a1 /= np.linalg.norm(a1)
        else:
            a1 = a1 / n1
        a2 = np.cross(u, a1)
        R = np.stack([a1, a2, u], axis=1)  # columns = axes; up is LAST column
    else:
        if P.shape[0] < 3:
            R = np.eye(3)
        else:
            cov = Q.T @ Q
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            R = evecs[:, order]
    R = _right_handed(R)

    local = Q @ R  # (p-c0) in box axes (columns=axes => local = R^T (p-c0) = (p-c0) @ R)
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    extents = hi - lo
    center = c0 + R @ ((lo + hi) / 2.0)
    return center, extents, R


def points_in_obb(
    points: np.ndarray,
    center: Any,
    extents: Any,
    rotation: Optional[Any] = None,
    margin: float = 0.0,
) -> np.ndarray:
    """Indices of ``points`` inside the OBB, extents FULL lengths, optionally grown
    by ``margin`` (fraction per axis). ``rotation`` None → axis-aligned box (the
    facts.objects detection shape)."""
    P = _as_np(points, np.float64).reshape(-1, 3)
    c = _as_np(center).reshape(3)
    e = _as_np(extents).reshape(3) * (1.0 + float(margin))
    half = np.maximum(e / 2.0, 1e-9)
    if rotation is None:
        local = P - c
    else:
        R = _as_np(rotation).reshape(3, 3)
        local = (P - c) @ R
    keep = np.all(np.abs(local) <= half, axis=1)
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# F3 completion: fit the generated asset into OUR selection OBB + EXP-20 gate
# ---------------------------------------------------------------------------

def _axes_by_extent(extents: np.ndarray) -> np.ndarray:
    """Axis indices ordered by descending extent."""
    return np.argsort(np.asarray(extents))[::-1]


def fit_asset_to_obb(
    asset_points: np.ndarray,
    obb_center: Any,
    obb_extents: Any,
    obb_rotation: Any,
    up: Optional[Any] = None,
) -> dict:
    """Rigid+uniform-scale placement of a generated asset into the selection OBB
    (features.md F3: "scale/pose from OUR selection OBB — uniform scale + yaw-align
    + floor snap — not the model's l2c").

    - Axis correspondence: asset local axes are matched to OBB axes by descending
      extent (deterministic; the yaw-align of the largest horizontal axis falls out
      of this when ``up`` is one of the OBB axes).
    - Uniform scale = min per-axis ratio (NEVER non-uniform — it would distort
      gaussian covariances).
    - ``up`` given → bottom-snap along up to the OBB bottom face ("floor snap when
      floor known"); else centered placement (recorded caveat).

    Returns ``{transform (4x4 row-major world), scale, rotation, translation,
    fitted_extents, caveats[]}``.
    """
    A = _as_np(asset_points, np.float64).reshape(-1, 3)
    if A.shape[0] == 0:
        raise ValueError("fit_asset_to_obb needs asset points")
    c_t = _as_np(obb_center).reshape(3)
    e_t = np.maximum(_as_np(obb_extents).reshape(3), 1e-9)
    R_t = _as_np(obb_rotation).reshape(3, 3)
    caveats: list[str] = []

    # Asset local box: PCA-free — SAM 3D assets come out roughly canonical, so use
    # the axis-aligned bbox of the local frame (stable across seeds).
    lo = A.min(axis=0)
    hi = A.max(axis=0)
    e_a = np.maximum(hi - lo, 1e-9)
    c_a = (lo + hi) / 2.0

    # Match axes by descending extent: asset axis order -> obb axis order.
    t_order = _axes_by_extent(e_t)
    a_order = _axes_by_extent(e_a)
    # Build R_map with columns: world direction for each ASSET axis.
    R_map = np.zeros((3, 3))
    for rank in range(3):
        R_map[:, a_order[rank]] = R_t[:, t_order[rank]]
    R_map = _right_handed(R_map)

    # Uniform scale: min per-matched-axis ratio.
    ratios = e_t[t_order] / e_a[a_order]
    scale = float(ratios.min())

    # Place asset bbox center at the OBB center...
    t_vec = c_t - scale * (R_map @ c_a)

    # ...then bottom-snap along up when known.
    if up is not None:
        u = _as_np(up).reshape(3)
        nu = np.linalg.norm(u)
        if nu > 1e-9:
            u = u / nu
            # bottom of target obb along up: center - (extent along u)/2
            half_t = float(np.abs((R_t * e_t[None, :] / 2.0).T @ u).sum())
            # transformed asset extent along up
            half_a = float(np.abs((R_map * (scale * e_a)[None, :] / 2.0).T @ u).sum())
            t_vec = t_vec - u * (half_t - half_a)  # drop so bottoms align
        else:
            caveats.append("up vector degenerate — centered placement, no floor snap")
    else:
        caveats.append("no up vector for this scene — centered placement, no floor snap")

    T = np.eye(4)
    T[:3, :3] = scale * R_map
    T[:3, 3] = t_vec
    return {
        "transform": [[float(v) for v in row] for row in T],
        "scale": scale,
        "rotation": [[float(v) for v in row] for row in R_map],
        "translation": [float(v) for v in t_vec],
        "fitted_extents": [float(v) for v in (scale * e_a)],
        "caveats": caveats,
    }


def apply_transform(points: np.ndarray, transform: Any) -> np.ndarray:
    """Apply a 4x4 row-major transform (as produced by fit_asset_to_obb) to (N,3) points."""
    P = _as_np(points, np.float64).reshape(-1, 3)
    T = _as_np(transform).reshape(4, 4)
    return P @ T[:3, :3].T + T[:3, 3]


def quality_gate(
    fitted_points: np.ndarray,
    obb_center: Any,
    obb_extents: Any,
    obb_rotation: Any,
) -> dict:
    """EXP-20 honesty gate on a fitted asset vs its selection OBB.

    GOOD  = centroid offset <= 20% of the OBB diagonal AND volume ratio in [0.2, 3.0]
    USABLE = exactly one of the two fails
    LOW   = both fail (client renders box-only)

    Volume ratio compares the fitted asset's bbox volume (in OBB axes) to the OBB
    volume. Returns ``{tier, centroid_offset_frac, vol_ratio, checks{...}}``.
    """
    P = _as_np(fitted_points, np.float64).reshape(-1, 3)
    c_t = _as_np(obb_center).reshape(3)
    e_t = np.maximum(_as_np(obb_extents).reshape(3), 1e-9)
    R_t = _as_np(obb_rotation).reshape(3, 3)

    diag = float(np.linalg.norm(e_t))
    centroid = P.mean(axis=0)
    off = float(np.linalg.norm(centroid - c_t))
    off_frac = off / diag if diag > 0 else np.inf

    local = (P - c_t) @ R_t
    fitted_e = np.maximum(local.max(axis=0) - local.min(axis=0), 1e-12)
    vol_ratio = float(np.prod(fitted_e) / np.prod(e_t))

    centroid_ok = off_frac <= GATE_CENTROID_FRAC
    vol_ok = GATE_VOL_RATIO[0] <= vol_ratio <= GATE_VOL_RATIO[1]
    tier = "good" if (centroid_ok and vol_ok) else ("usable" if (centroid_ok or vol_ok) else "low")
    return {
        "tier": tier,
        "centroid_offset_frac": round(off_frac, 4),
        "vol_ratio": round(vol_ratio, 4),
        "checks": {
            "centroid_within_20pct_diag": bool(centroid_ok),
            "vol_ratio_in_band": bool(vol_ok),
        },
    }


# ---------------------------------------------------------------------------
# Minimal gaussian-splat PLY I/O (positions only) — enough to fit/gate a SAM-3D
# asset without importing scene_report.splat_io (whose reader wants the full
# product schema; SAM-3D PLYs are plain INRIA layout).
# ---------------------------------------------------------------------------

def read_ply_positions(data: bytes) -> np.ndarray:
    """(N,3) float positions from a binary_little_endian PLY with all-float vertex
    properties (the INRIA gaussian layout SAM 3D / TRELLIS emit). Raises ValueError
    on anything else — callers surface an honest error, never a silent guess."""
    head_end = data.find(b"end_header\n")
    if head_end < 0:
        raise ValueError("not a PLY (no end_header)")
    header = data[:head_end].decode("ascii", errors="replace")
    body = data[head_end + len(b"end_header\n"):]
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("PLY is not binary_little_endian")
    n = None
    props: list[str] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] not in ("float", "float32"):
                raise ValueError(f"non-float vertex property {parts[2]!r}")
            props.append(parts[2])
    if n is None or not props:
        raise ValueError("PLY has no vertex element")
    for ax in ("x", "y", "z"):
        if ax not in props:
            raise ValueError(f"PLY vertex missing {ax!r}")
    stride = len(props)
    arr = np.frombuffer(body, dtype="<f4", count=n * stride).reshape(n, stride)
    ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
    return arr[:, [ix, iy, iz]].astype(np.float64)
