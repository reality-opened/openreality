"""Tests for the strict-gated camera-space object-tracks builder (Track-On-R seam).

Synthetic ground-truth fixtures, GPU-free (numpy only; cv2 needed just for the seeder test,
which skips when unavailable). Covers the EXP-15 invariants:

- exact camera-space lift + lazy world compose (per-frame-own-pose) on static geometry;
- STRICT C1 visibility gating: occluded frames are explicit ``null`` gaps, never positions;
- confidence-masked depth: visible-but-unconfident frames become gaps too;
- serialization: strict JSON, companion ``poses.jsonl``, no collision with the eager
  ``objects_tracks.jsonl`` sidecar.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from server.export.dynamics import (
    CAMERA_TRACKS_SCHEMA_VERSION,
    backproject_camera,
    build_camera_space_tracks,
    robust_depth_at,
    write_camera_space_tracks,
)
from server.export.tracker_trackonr import (
    assemble_point_objects_2d,
    fps_select,
    seed_query_points,
)


# ---------------------------------------------------------------------------
# Synthetic scene fixture: moving pinhole camera over static points + one mover.
# ---------------------------------------------------------------------------

T, H, W = 10, 96, 128
FX = FY = 100.0
CX, CY = W / 2.0, H / 2.0

STATIC_W = np.array(
    [[0.3, 0.1, 3.0], [-0.4, 0.2, 4.0], [0.1, -0.3, 3.5], [0.5, 0.4, 5.0]]
)
MOVER0 = np.array([0.0, 0.0, 3.0])
MOVER_VEL = np.array([0.15, 0.0, 0.0])


def _scene():
    """Returns (poses (T,4,4), K (T,3,3), depth (T,H,W), conf (T,H,W), tracks (N,T,2),
    world_gt (N,) static + mover). Depth stamped in a window around each projection."""
    K = np.tile(np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]]), (T, 1, 1))
    poses = np.zeros((T, 4, 4))
    for t in range(T):
        ang = 0.04 * t
        Rz = np.array(
            [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        )
        poses[t, :3, :3] = Rz
        poses[t, :3, 3] = [0.08 * t, 0.02 * t, 0.0]
        poses[t, 3, 3] = 1.0

    depth = np.full((T, H, W), np.nan, np.float32)
    conf = np.full((T, H, W), 50.0, np.float32)  # uniformly confident by default

    n_static = STATIC_W.shape[0]
    N = n_static + 1
    tracks = np.zeros((N, T, 2))

    def project(pw, t):
        w2c = np.linalg.inv(poses[t])
        pc = (w2c[:3, :3] @ pw) + w2c[:3, 3]
        return FX * pc[0] / pc[2] + CX, FY * pc[1] / pc[2] + CY, pc[2]

    def stamp(u, v, z, t):
        ui, vi = int(round(u)), int(round(v))
        depth[t, max(0, vi - 3): vi + 4, max(0, ui - 3): ui + 4] = z

    for t in range(T):
        for i, pw in enumerate(STATIC_W):
            u, v, z = project(pw, t)
            tracks[i, t] = [u, v]
            stamp(u, v, z, t)
        u, v, z = project(MOVER0 + MOVER_VEL * t, t)
        tracks[n_static, t] = [u, v]
        stamp(u, v, z, t)
    return poses, K, depth, conf, tracks


def _objects(tracks, vis=None):
    N, Tn = tracks.shape[0], tracks.shape[1]
    if vis is None:
        vis = np.ones((N, Tn), bool)
    prov = ["shi_tomasi"] * (N - 1) + ["mover"]
    return assemble_point_objects_2d(prov, list(range(Tn)), tracks, vis)


IDENTITY_MAP = {}  # frames are already global ints


# ---------------------------------------------------------------------------
# lift math
# ---------------------------------------------------------------------------

def test_backproject_camera_exact():
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
    p = np.array([0.4, -0.2, 2.5])
    u = FX * p[0] / p[2] + CX
    v = FY * p[1] / p[2] + CY
    got = backproject_camera(u, v, p[2], K)
    assert np.allclose(got, p, atol=1e-12)
    # 4-vector intrinsics too
    got4 = backproject_camera(u, v, p[2], np.array([FX, FY, CX, CY]))
    assert np.allclose(got4, p, atol=1e-12)


def test_robust_depth_confidence_mask():
    depth = np.full((H, W), 2.0, np.float32)
    conf = np.full((H, W), 10.0, np.float32)
    d, ok, mc = robust_depth_at(depth, conf, 40, 40, conf_thresh=5.0)
    assert ok and d == pytest.approx(2.0) and mc == pytest.approx(10.0)
    # all-rejected window -> gap
    d, ok, _ = robust_depth_at(depth, conf, 40, 40, conf_thresh=99.0)
    assert not ok and np.isnan(d)
    # out-of-bounds pixel -> gap
    d, ok, _ = robust_depth_at(depth, conf, -5, 40, conf_thresh=5.0)
    assert not ok
    # median ignores low-confidence outlier pixels (poison only the window's inner 3x3,
    # leaving the confident 5x5 ring at the true depth)
    depth2 = depth.copy()
    conf2 = conf.copy()
    depth2[39:42, 39:42] = 9.0
    conf2[39:42, 39:42] = 1.0
    d, ok, _ = robust_depth_at(depth2, conf2, 40, 40, conf_thresh=5.0)
    assert ok and d == pytest.approx(2.0)


def test_static_points_lift_exactly_and_spread_zero():
    poses, K, depth, conf, tracks = _scene()
    out = build_camera_space_tracks(
        _objects(tracks), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    recs = out["records"]
    assert len(recs) == 5
    for i, rec in enumerate(recs[:4]):
        assert rec["n_liftable"] == T
        # camera-space xyz composes (with each frame's OWN pose) back to the true world point
        for t, xyz in enumerate(rec["frame_xyz_cam"]):
            P = poses[t]
            world = P[:3, :3] @ np.array(xyz) + P[:3, 3]
            assert np.allclose(world, STATIC_W[i], atol=1e-6)
        assert rec["world_rms_spread"] < 1e-6
        assert rec["motion"] == "static"
    # scene scale = camera path length
    expect_scale = float(
        np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
    )
    assert out["scene_scale"] == pytest.approx(expect_scale)


def test_mover_excursion_and_label():
    poses, K, depth, conf, tracks = _scene()
    out = build_camera_space_tracks(
        _objects(tracks), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    mover = out["records"][-1]
    assert mover["query"] == "mover"
    expect = np.linalg.norm(MOVER_VEL) * (T - 1)
    assert mover["max_displacement"] == pytest.approx(expect, abs=0.02)
    assert mover["motion"] == "moving"


# ---------------------------------------------------------------------------
# the C1 strict gate
# ---------------------------------------------------------------------------

def test_strict_visibility_gate_occluded_is_null_never_position():
    poses, K, depth, conf, tracks = _scene()
    vis = np.ones((5, T), bool)
    occ = [2, 3, 7]
    vis[0, occ] = False
    out = build_camera_space_tracks(
        _objects(tracks, vis), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    rec = out["records"][0]
    for t in range(T):
        if t in occ:
            assert rec["frame_visibility"][t] is False
            assert rec["frame_uv"][t] is None          # not even the 2D position leaks
            assert rec["frame_xyz_cam"][t] is None     # explicit gap, never a position
            assert rec["frame_liftable"][t] is False
            assert rec["frame_depth_conf"][t] is None
        else:
            assert rec["frame_visibility"][t] is True
            assert rec["frame_xyz_cam"][t] is not None
    assert rec["n_visible"] == T - len(occ)
    assert rec["n_liftable"] == T - len(occ)


def test_depth_reject_is_gap_but_stays_visible():
    poses, K, depth, conf, tracks = _scene()
    # kill confidence around point 1's pixel on frame 4 -> visible but not liftable
    u, v = tracks[1, 4]
    ui, vi = int(round(u)), int(round(v))
    conf = conf.copy()
    conf[4, max(0, vi - 3): vi + 4, max(0, ui - 3): ui + 4] = 0.1
    out = build_camera_space_tracks(
        _objects(tracks), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    rec = out["records"][1]
    assert rec["frame_visibility"][4] is True    # the tracker did see it
    assert rec["frame_liftable"][4] is False     # but depth failed the mask
    assert rec["frame_xyz_cam"][4] is None       # => explicit gap
    assert rec["frame_uv"][4] is not None        # 2D is kept (it IS a real observation)
    assert rec["n_liftable"] == T - 1


def test_no_liftable_frames_means_unknown_motion_no_3d():
    poses, K, depth, conf, tracks = _scene()
    conf = np.zeros_like(conf)  # nothing clears any threshold
    out = build_camera_space_tracks(
        _objects(tracks), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    for rec in out["records"]:
        assert rec["n_liftable"] == 0
        assert all(x is None for x in rec["frame_xyz_cam"])
        assert rec["motion"] == "unknown"
        assert rec["world_rms_spread"] is None


def test_conf_threshold_from_percentile():
    poses, K, depth, conf, tracks = _scene()
    conf = conf.copy()
    conf[:, :, : W // 2] = 10.0  # bimodal confidences
    out = build_camera_space_tracks(
        _objects(tracks), poses, K, depth, conf, IDENTITY_MAP, conf_percentile=30.0
    )
    assert out["conf_thresh"] == pytest.approx(float(np.percentile(conf, 30.0)))


def test_index_map_resolution_and_unresolvable_dropped():
    poses, K, depth, conf, tracks = _scene()
    index_map = {(0, i): i for i in range(T - 1)}  # last frame unresolvable
    objs = _objects(tracks)
    for o in objs:
        o["frames"] = [(0, i) for i in range(T)]  # submap-local keys
    out = build_camera_space_tracks(
        objs, poses, K, depth, conf, index_map, conf_thresh=5.0
    )
    rec = out["records"][0]
    assert rec["frames"] == list(range(T - 1))  # unresolvable key dropped gracefully
    assert rec["n_liftable"] == T - 1


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

def test_write_camera_space_tracks_files_and_strict_json(tmp_path):
    poses, K, depth, conf, tracks = _scene()
    vis = np.ones((5, T), bool)
    vis[0, 3] = False
    out = build_camera_space_tracks(
        _objects(tracks, vis), poses, K, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    res = write_camera_space_tracks(out["records"], poses, K, (H, W), str(tmp_path))

    assert os.path.isfile(res["tracks_path"])
    assert os.path.isfile(res["poses_path"])
    assert res["n_tracks"] == 5 and res["n_pose_rows"] == T
    # the eager ObjectTrack sidecar file is NOT touched by this producer
    assert not os.path.exists(os.path.join(str(tmp_path), "dynamics", "objects_tracks.jsonl"))

    def _no_nan(x):
        raise AssertionError(f"non-strict JSON constant leaked: {x}")

    lines = open(res["tracks_path"]).read().splitlines()
    assert len(lines) == 5
    recs = [json.loads(ln, parse_constant=_no_nan) for ln in lines]
    assert recs[0]["schema_version"] == CAMERA_TRACKS_SCHEMA_VERSION
    assert recs[0]["frame_xyz_cam"][3] is None  # the occluded gap round-trips as null

    pose_rows = [json.loads(ln, parse_constant=_no_nan)
                 for ln in open(res["poses_path"]).read().splitlines()]
    assert [r["frame_index"] for r in pose_rows] == list(range(T))
    assert np.asarray(pose_rows[0]["cam_to_world"]).shape == (4, 4)
    assert np.asarray(pose_rows[0]["intrinsic"]).shape == (3, 3)
    assert pose_rows[0]["image_hw"] == [H, W]

    # consumer-side lazy lift from the two FILES alone reproduces the static world point
    pose_by_frame = {r["frame_index"]: np.asarray(r["cam_to_world"]) for r in pose_rows}
    rec = recs[1]
    lifted = []
    for t, gi in enumerate(rec["frames"]):
        if rec["frame_xyz_cam"][t] is None:
            continue
        P = pose_by_frame[gi]
        lifted.append(P[:3, :3] @ np.asarray(rec["frame_xyz_cam"][t]) + P[:3, 3])
    lifted = np.stack(lifted)
    assert np.allclose(lifted, STATIC_W[1], atol=1e-6)


def test_intrinsics_4vector_serialized_as_3x3(tmp_path):
    poses, K, depth, conf, tracks = _scene()
    K4 = [np.array([FX, FY, CX, CY])] * T  # trajectory-convention intrinsics
    out = build_camera_space_tracks(
        _objects(tracks), poses, K4, depth, conf, IDENTITY_MAP, conf_thresh=5.0
    )
    res = write_camera_space_tracks(out["records"], poses, K4, (H, W), str(tmp_path))
    row = json.loads(open(res["poses_path"]).readline())
    K3 = np.asarray(row["intrinsic"])
    assert K3.shape == (3, 3)
    assert K3[0, 0] == FX and K3[1, 1] == FY and K3[0, 2] == CX and K3[1, 2] == CY


# ---------------------------------------------------------------------------
# producer helpers
# ---------------------------------------------------------------------------

def test_assemble_point_objects_2d_shapes_and_passthrough():
    tracks = np.zeros((3, 4, 2))
    tracks[1, 2] = [10.5, 20.5]
    vis = np.ones((3, 4), bool)
    vis[1, 2] = False
    objs = assemble_point_objects_2d(["a", "b", "c"], [7, 8, 9, 10], tracks, vis)
    assert len(objs) == 3
    assert objs[0]["id"] == 0 and objs[0]["query"] == "a"
    assert objs[1]["frames"] == [7, 8, 9, 10]
    # occluded frame's 2D position is passed through UNTOUCHED (the builder gates it)
    assert objs[1]["centroids"][2] == (10.5, 20.5)
    assert objs[1]["visibility"][2] is False
    assert objs[1]["confidence"][2] == 0.0


def test_assemble_point_objects_2d_validation():
    with pytest.raises(ValueError):
        assemble_point_objects_2d(["a"], [0, 1], np.zeros((1, 2, 3)), np.ones((1, 2), bool))
    with pytest.raises(ValueError):
        assemble_point_objects_2d(["a"], [0], np.zeros((1, 2, 2)), np.ones((1, 2), bool))
    with pytest.raises(ValueError):
        assemble_point_objects_2d(["a", "b"], [0, 1], np.zeros((1, 2, 2)), np.ones((1, 2), bool))


def test_fps_select_spread_and_small_input():
    px = np.array([[0, 0], [100, 0], [0, 100], [1, 1], [2, 2]])
    sel = fps_select(px, 3)
    assert len(sel) == 3
    assert {0, 1, 2} <= set(sel.tolist())  # the far corners are picked
    assert len(fps_select(px[:2], 5)) == 2  # M <= k returns all


def test_seed_query_points_extras_first_then_corners():
    cv2 = pytest.importorskip("cv2")  # noqa: F841 — seeder needs cv2
    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, (H, W), np.uint8)).astype(np.uint8)
    extras = np.array([[30.0, 30.0], [-5.0, 10.0]])  # second is out of bounds
    q, prov = seed_query_points(img, n_total=8, extra_pixels=extras,
                                extra_provenance="person_kp")
    assert q.shape[1] == 3 and np.all(q[:, 0] == 0.0)
    assert prov[0] == "person_kp" and prov.count("person_kp") == 1  # OOB extra dropped
    assert all(p in ("person_kp", "shi_tomasi") for p in prov)
    assert len(q) <= 8
