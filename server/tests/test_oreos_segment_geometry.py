"""W3 segment_geometry tests — synthetic scenes, GPU-free (numpy + scipy only).

Covers the §0.2 mask-lift ladder (projection → dilated-mask test → z-buffer
occlusion at tau=0.08 → cKDTree outlier filter → gravity-aware PCA OBB), the
F3 fit-to-OBB placement, and the EXP-20 quality gate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from server.oreos import segment_geometry as sg


# ---------------------------------------------------------------------------
# synthetic camera rig: OpenCV convention, camera at origin looking +z
# ---------------------------------------------------------------------------

H, W = 64, 64
INTR = [100.0, 100.0, 32.0, 32.0]  # fx fy cx cy
C2W = np.eye(4)  # camera frame == world frame


def _grid(z: float, half: float, n: int = 11) -> np.ndarray:
    """n x n grid of points on the plane at depth z spanning +-half in x/y."""
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def _center_mask(px: int = 20) -> np.ndarray:
    """Square mask of +-px around the principal point."""
    m = np.zeros((H, W), dtype=bool)
    m[32 - px : 32 + px, 32 - px : 32 + px] = True
    return m


# ---------------------------------------------------------------------------
# lift: projection + mask + z-buffer occlusion
# ---------------------------------------------------------------------------


def test_lift_selects_masked_points_and_occludes_background():
    target = _grid(z=2.0, half=0.3)          # projects to u,v in [17,47] — inside mask
    behind = _grid(z=4.0, half=0.6)          # same pixels, twice the depth — occluded
    outside = _grid(z=2.0, half=0.3) + np.array([1.2, 0.0, 0.0])  # out of mask/image
    pts = np.concatenate([target, behind, outside])

    idx = sg.lift_mask_to_points(pts, _center_mask(), INTR, C2W)
    got = set(idx.tolist())

    n_t = len(target)
    assert set(range(n_t)) <= got, "every visible target point survives the lift"
    assert not (got & set(range(n_t, n_t + len(behind)))), "occluded background is culled"
    assert not (got & set(range(n_t + len(behind), len(pts)))), "out-of-mask points are culled"


def test_lift_zbuffer_band_keeps_near_surface():
    """Points within tau of the front surface stay; beyond the band they are culled."""
    target = _grid(z=2.0, half=0.3)
    near = _grid(z=2.10, half=0.3)   # 5% behind — inside the 8% band
    far = _grid(z=2.40, half=0.3)    # 20% behind — outside the band
    pts = np.concatenate([target, near, far])
    idx = set(sg.lift_mask_to_points(pts, _center_mask(), INTR, C2W).tolist())
    n = len(target)
    assert set(range(n)) <= idx
    assert set(range(n, 2 * n)) <= idx, "points within tau of the surface survive"
    assert not (idx & set(range(2 * n, 3 * n))), "points beyond tau are occluded"


def test_lift_mask_dilation_recovers_edge_points():
    """A point projecting 2 px outside the mask edge is kept at dilate=3, culled at 0."""
    # mask right edge at u=51 (exclusive); point at u=53 -> x = (53.5-32)/100*2
    pt = np.array([[(53.5 - 32.0) / 100.0 * 2.0, 0.0, 2.0]])
    kept = sg.lift_mask_to_points(pt, _center_mask(), INTR, C2W, dilate_px=3)
    culled = sg.lift_mask_to_points(pt, _center_mask(), INTR, C2W, dilate_px=0)
    assert len(kept) == 1
    assert len(culled) == 0


def test_lift_behind_camera_and_empty():
    pts = _grid(z=-2.0, half=0.3)  # entirely behind the camera
    assert len(sg.lift_mask_to_points(pts, _center_mask(), INTR, C2W)) == 0


def test_lift_respects_nontrivial_c2w():
    """Moving the camera back 1 unit shifts depths but not the selection."""
    c2w = np.eye(4)
    c2w[2, 3] = -1.0  # camera at z=-1 looking +z
    target = _grid(z=2.0, half=0.3)  # depth 3 in this camera
    idx = sg.lift_mask_to_points(target, _center_mask(px=26), INTR, c2w)
    assert len(idx) == len(target)


# ---------------------------------------------------------------------------
# outlier filter
# ---------------------------------------------------------------------------


def test_kdtree_filter_drops_far_outliers():
    rng = np.random.default_rng(7)
    cluster = rng.normal(0, 0.05, size=(400, 3))
    outliers = np.array([[5.0, 5.0, 5.0], [-6.0, 2.0, 1.0], [0.0, 9.0, -3.0]])
    pts = np.concatenate([cluster, outliers])
    keep = set(sg.filter_outliers_kdtree(pts).tolist())
    assert len(keep & {400, 401, 402}) == 0, "far outliers dropped"
    assert len(keep) >= 380, "the cluster body survives"


def test_kdtree_filter_tiny_input_passthrough():
    pts = np.zeros((4, 3))
    assert len(sg.filter_outliers_kdtree(pts, k=8)) == 4


# ---------------------------------------------------------------------------
# PCA OBB (plain + gravity-constrained)
# ---------------------------------------------------------------------------


def _box_points(extents, R=np.eye(3), center=np.zeros(3), n=9):
    ex = np.asarray(extents) / 2.0
    g = np.stack(
        np.meshgrid(*[np.linspace(-h, h, n) for h in ex], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return g @ np.asarray(R).T + center


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_pca_obb_recovers_rotated_box():
    true_e = np.array([2.0, 1.0, 0.5])
    R = _rot_z(0.6)
    center = np.array([3.0, -1.0, 2.0])
    pts = _box_points(true_e, R, center)
    c, e, Rr = sg.pca_obb(pts)
    assert np.allclose(c, center, atol=1e-6)
    assert np.allclose(sorted(e, reverse=True), sorted(true_e, reverse=True), atol=1e-6)
    assert np.isclose(abs(np.linalg.det(Rr)), 1.0, atol=1e-9)
    assert np.linalg.det(Rr) > 0, "right-handed axes"


def test_pca_obb_gravity_constrained_snaps_axis_to_up():
    up = np.array([0.0, 0.0, 1.0])
    # box yawed in-plane AND sheared points would tilt plain PCA; gravity mode may not.
    pts = _box_points([2.0, 1.0, 0.6], _rot_z(0.8), center=[1.0, 2.0, 0.3])
    c, e, R = sg.pca_obb(pts, up=up)
    # one axis is exactly +-up
    dots = np.abs(R.T @ up)
    assert np.isclose(dots.max(), 1.0, atol=1e-9)
    up_axis = int(np.argmax(dots))
    assert np.isclose(e[up_axis], 0.6, atol=1e-6), "extent along up = true height"
    # in-plane extents recovered despite yaw
    inplane = sorted(np.delete(e, up_axis), reverse=True)
    assert np.allclose(inplane, [2.0, 1.0], atol=1e-6)


def test_pca_obb_wire_convention_columns_are_axes():
    pts = _box_points([2.0, 1.0, 0.5], _rot_z(0.3))
    c, e, R = sg.pca_obb(pts)
    local = (pts - c) @ R
    spans = local.max(0) - local.min(0)
    assert np.allclose(spans, e, atol=1e-9), "extents are FULL lengths in column axes"


def test_points_in_obb_axis_aligned_and_margin():
    pts = np.array([[0.0, 0, 0], [0.6, 0, 0], [1.4, 0, 0]])
    inside = sg.points_in_obb(pts, [0, 0, 0], [1.0, 1.0, 1.0])
    assert inside.tolist() == [0]
    grown = sg.points_in_obb(pts, [0, 0, 0], [1.0, 1.0, 1.0], margin=0.4)
    assert grown.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# fit-to-OBB + quality gate
# ---------------------------------------------------------------------------


def test_fit_asset_uniform_scale_and_containment():
    asset = _box_points([1.0, 1.0, 1.0])  # unit cube, centered
    R_t = _rot_z(0.5)
    fit = sg.fit_asset_to_obb(asset, [5.0, 0.0, 1.0], [2.0, 1.0, 0.5], R_t)
    assert np.isclose(fit["scale"], 0.5), "uniform scale = min per-axis ratio"
    placed = sg.apply_transform(asset, fit["transform"])
    local = (placed - np.array([5.0, 0.0, 1.0])) @ R_t
    assert np.all(np.abs(local) <= np.array([2.0, 1.0, 0.5]) / 2 + 1e-9), "fits inside the OBB"
    assert any("no up vector" in c for c in fit["caveats"])


def test_fit_asset_bottom_snap_with_up():
    asset = _box_points([1.0, 1.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    center, extents = [0.0, 0.0, 2.0], [2.0, 2.0, 1.0]
    fit = sg.fit_asset_to_obb(asset, center, extents, np.eye(3), up=up)
    placed = sg.apply_transform(asset, fit["transform"])
    obb_bottom = 2.0 - 0.5
    assert np.isclose(placed[:, 2].min(), obb_bottom, atol=1e-9), "asset bottom on OBB bottom"
    assert fit["caveats"] == []


def test_quality_gate_tiers():
    obb_c, obb_e, obb_R = np.zeros(3), np.array([2.0, 1.0, 0.5]), np.eye(3)
    good = _box_points(obb_e * 0.95)
    assert sg.quality_gate(good, obb_c, obb_e, obb_R)["tier"] == "good"

    # centered but far too small a volume -> exactly one check fails -> usable
    small = _box_points(obb_e * 0.2)  # vol ratio 0.008
    g = sg.quality_gate(small, obb_c, obb_e, obb_R)
    assert g["tier"] == "usable"
    assert g["checks"]["centroid_within_20pct_diag"] and not g["checks"]["vol_ratio_in_band"]

    # small AND far off-center -> both fail -> low (box-only)
    off = small + np.array([3.0, 0.0, 0.0])
    assert sg.quality_gate(off, obb_c, obb_e, obb_R)["tier"] == "low"


# ---------------------------------------------------------------------------
# PLY positions reader (SAM-3D / TRELLIS asset layout)
# ---------------------------------------------------------------------------


def _tiny_ply(points: np.ndarray, extra_props=("f_dc_0",)) -> bytes:
    props = ["x", "y", "z", *extra_props]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    ).encode("ascii")
    body = np.zeros((len(points), len(props)), dtype="<f4")
    body[:, :3] = points
    return header + body.tobytes()


def test_read_ply_positions_roundtrip():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    out = sg.read_ply_positions(_tiny_ply(pts))
    assert np.allclose(out, pts)


def test_read_ply_positions_rejects_ascii():
    with pytest.raises(ValueError):
        sg.read_ply_positions(b"ply\nformat ascii 1.0\nend_header\n")


def test_stride_subsample_cap():
    idx = sg.stride_subsample(10, cap=3)
    assert len(idx) <= 3 + 1 and idx[0] == 0
    assert len(sg.stride_subsample(3, cap=10)) == 3
