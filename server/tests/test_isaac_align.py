"""Unit tests for server/export/isaac/align.py — gravity + scale + alignment transform.

Pure numpy/scipy (no open3d/pxr), so these run in any env. The geometric correctness of the
Isaac export lives here, so the synthetic scenes mirror the real failure modes: a tilted,
arbitrarily-scaled SLAM frame whose floor must be recovered, oriented up via the cameras, and
mapped to a metric Z-up stage with the floor at z=0.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from server.export.isaac import align


def _tilt_and_scale(rot, k, P):
    """Map an ideal metric scene into a tilted, k-times-bigger SLAM frame."""
    return (rot @ np.asarray(P, dtype=float).T).T * k


def test_fit_plane_ransac_recovers_known_plane():
    rng = np.random.default_rng(0)
    normal_gt = align._unit([0.2, -0.3, 1.0])
    # Random points on the plane normal·x = 5, plus small noise.
    basis = np.linalg.svd(normal_gt.reshape(1, 3))[2][1:]  # two in-plane directions
    uv = rng.uniform(-3, 3, size=(1500, 2))
    pts = uv @ basis + 5.0 * normal_gt
    pts += rng.normal(scale=1e-3, size=pts.shape)

    normal, d, mask = align.fit_plane_ransac(pts, seed=1)
    # Normal parallel to ground truth (up to sign).
    assert abs(abs(float(normal @ normal_gt)) - 1.0) < 1e-3
    assert mask.mean() > 0.95


def test_estimate_gravity_points_up_toward_cameras():
    rng = np.random.default_rng(2)
    rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    k = 3.7
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(2000, 3))
    floor[:, 2] = 0.0
    cams = np.array([[0, 0, 1.5], [1, 0, 1.5], [-1, 0, 1.5]], dtype=float)

    slam_floor = _tilt_and_scale(rot, k, floor)
    slam_cams = _tilt_and_scale(rot, k, cams)

    up, mask, source = align.estimate_gravity(slam_floor, slam_cams)
    up_true = align._unit(rot @ np.array([0, 0, 1.0]))
    assert source == "floor_plane"
    assert float(up @ up_true) > 0.99  # correct axis AND correct (upward) sign


def test_estimate_gravity_override():
    up, mask, source = align.estimate_gravity(
        np.zeros((10, 3)), up_override=[0, 0, 5.0]
    )
    assert source == "user_override"
    assert np.allclose(up, [0, 0, 1.0])
    assert mask is None


@pytest.mark.parametrize(
    "a,b",
    [([0, 0, 1], [0, 0, 1]), ([0, 0, 1], [1, 0, 0]), ([0, 0, 1], [0, 0, -1]),
     ([0.2, -0.3, 1.0], [0, 0, 1])],
)
def test_rotation_align_vectors(a, b):
    R = align.rotation_align_vectors(a, b)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)  # orthonormal
    assert abs(np.linalg.det(R) - 1.0) < 1e-9  # proper rotation
    assert np.allclose(R @ align._unit(a), align._unit(b), atol=1e-8)


def test_resolve_scale_priority():
    assert align.resolve_scale(scale=2.0) == (2.0, "user:factor", True)
    s, src, metric = align.resolve_scale(ref_extent_slam=1.18, ref_meters=2.0, ref_query="door")
    assert abs(s - 2.0 / 1.18) < 1e-9 and metric and "door" in src
    assert align.resolve_scale() == (1.0, "none", False)


def test_compute_alignment_floor_to_origin_metric_up():
    rng = np.random.default_rng(3)
    rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    k = 3.7  # SLAM units per meter
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(3000, 3))
    floor[:, 2] = 0.0
    floor[:, 2] += rng.normal(scale=1e-3, size=floor.shape[0])
    above = rng.uniform([-1, -1, 0.2], [1, 1, 2.0], size=(800, 3))
    cams = np.array([[0, 0, 1.5], [1, 0, 1.5], [-1, 0, 1.5]], dtype=float)

    slam_pts = _tilt_and_scale(rot, k, np.vstack([floor, above]))
    slam_cams = _tilt_and_scale(rot, k, cams)

    res = align.compute_alignment(slam_pts, slam_cams, scale=1.0 / k)

    assert res.metric and res.gravity_aligned and res.scale_source == "user:factor"
    # Recovered up maps to +Z.
    up_true = align._unit(rot @ np.array([0, 0, 1.0]))
    assert np.allclose(res.rotation @ up_true, [0, 0, 1.0], atol=1e-2)

    q = align.apply_transform(res.transform, slam_pts)
    q_floor, q_above = q[: len(floor)], q[len(floor):]
    assert abs(np.median(q_floor[:, 2])) < 1e-2  # floor dropped to z=0
    assert q_above[:, 2].min() > 0.05  # objects sit above the floor
    # Metric height recovered: tallest object near its true 2.0 m.
    assert abs(q_above[:, 2].max() - above[:, 2].max()) < 0.1
    # XY recentered.
    assert np.allclose(q[:, :2].mean(axis=0), [0, 0], atol=1e-6)


def test_compute_alignment_ref_object_scale():
    rng = np.random.default_rng(4)
    k = 5.0
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(3000, 3))
    floor[:, 2] = 0.0
    cams = np.array([[0, 0, 1.5]], dtype=float)
    slam_floor = floor * k  # axis-aligned (no tilt) so the box height is exact
    slam_cams = cams * k

    # A "door" 2.0 m tall in reality; in SLAM units that is 2.0*k tall.
    ref_center = np.array([0.0, 0.0, 1.0]) * k
    ref_extent = np.array([0.9, 0.1, 2.0]) * k
    res = align.compute_alignment(
        slam_floor, slam_cams, ref_box=(ref_center, ref_extent),
        ref_meters=2.0, ref_query="door",
    )
    assert res.metric and "door" in res.scale_source
    assert abs(res.scale - 1.0 / k) < 1e-6  # back-out recovers the true metric scale


def test_transform_camera_pose_consistency():
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(2000, 3))
    pts[:, 2] = np.abs(pts[:, 2]) * 0.01  # near-floor cluster for a stable plane
    res = align.compute_alignment(pts, scale=2.0)

    # A camera pose: orthonormal rotation + position.
    Rc = Rotation.from_euler("xyz", [0.5, 0.1, -0.3]).as_matrix()
    tc = np.array([0.3, -0.4, 0.8])
    pose = np.eye(4)
    pose[:3, :3] = Rc
    pose[:3, 3] = tc

    out = align.transform_camera_pose(res, pose)
    # Position matches the affine point transform; rotation stays orthonormal.
    assert np.allclose(out[:3, 3], align.apply_transform(res.transform, tc[None])[0], atol=1e-9)
    assert np.allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-9)
