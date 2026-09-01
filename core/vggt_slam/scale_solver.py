import numpy as np

def debug_visualize(pcd1_points, pcd2_points):
    import open3d as o3d  # lazy: keeps this module importable without open3d (CI / CPU tests)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

def estimate_scale_pairwise(X, Y, DEBUG=False):
    assert X.shape == Y.shape
    x_dists = np.linalg.norm(X, axis=1)
    y_dists = np.linalg.norm(Y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scales = y_dists / x_dists

    # A point at the origin gives 0/0 -> NaN (or |X|=0 < |Y| -> inf), and np.median
    # propagates a SINGLE NaN to the whole result. That NaN scale becomes a NaN
    # homography and silently poisons the gtsam graph — the run keeps going and the
    # damage only surfaces (if ever) far downstream. Note scale_dispersion_info below
    # already filters to finite ratios, so the diagnostics looked clean while the
    # scale itself was NaN. Filter here too, and refuse loudly if nothing survives:
    # a stopped run beats a silently wrong map.
    finite_scales = scales[np.isfinite(scales)]
    if finite_scales.size == 0:
        raise ValueError(
            f"estimate_scale_pairwise: no finite scale ratios among {scales.size} point "
            "pair(s) — every |X| was zero or non-finite, so the median scale is undefined. "
            "This usually means the overlap mask selected degenerate or empty geometry."
        )
    scale = float(np.median(finite_scales))

    if DEBUG:
        debug_visualize(X*scale, Y)

    # Second return element is now an info dict (was None). It carries the scale-estimate
    # DISPERSION — a robust rel-IQR of the per-point norm ratios — used by the opt-in junction
    # weighting (vggt_slam.junction_weight) to soften ill-determined junctions. Callers that only
    # want the scale keep using [0]; the change is additive.
    info = scale_dispersion_info(scales, scale)
    return scale, info


def scale_dispersion_info(scales, scale=None):
    """Robust spread of the per-point scale ratios: rel-IQR = (p75 - p25) / median.

    Kept separate (pure numpy) so it is unit-testable without the geometry above. NaN-safe;
    returns ``{"dispersion", "n"}``.
    """
    s = np.asarray(scales, float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        return {"dispersion": float("nan"), "n": 0}
    med = float(np.median(s)) if scale is None else float(scale)
    if med == 0:
        return {"dispersion": float("nan"), "n": n}
    q1, q3 = np.percentile(s, [25, 75])
    return {"dispersion": float((q3 - q1) / abs(med)), "n": n}