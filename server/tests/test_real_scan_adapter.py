"""CPU tests for the real-export -> dynamics-producer adapter (GPU-free, no torch/modal).

Builds a **realistic OpenReality export tree** on disk with the *exact* on-disk schema a real
scan writes (``server/export/writer.py``): ``data/trajectory.parquet`` (per-keyframe
cam->world state + intrinsics), ``videos/ego_view.mp4`` (per-keyframe RGB), ``grounding/
objects.jsonl`` (3D object box + evidence keyframe), ``clouds/cloud.npz`` (fused world cloud).
It deliberately does NOT use ``vggt_slam`` -- the same reason CI auto-skips the live export
path -- so it exercises the adapter's parsing/projection/depth logic against real fields.

Then it drives the full CPU chain: parse -> project the object's 3D world box into its evidence
keyframe -> render per-frame depth from the cloud -> (FAKE tracker) assemble ``objects_2d`` ->
``build_object_tracks`` -> ``write_object_tracks_jsonl``, asserting the ``objects_2d`` shape and
that project/unproject round-trips.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from server.export import real_scan_adapter as rsa
from server.export.dynamics import (
    build_object_tracks,
    unproject_to_world,
    write_object_tracks_jsonl,
)
from server.export.tracker_bootstapir import (
    assemble_objects_2d,
    sample_query_points_in_mask,
)
from server.scene_report.schemas import ObjectTrack

# fixture geometry: identity-rotation cam->world poses sliding along world x, looking down +z.
N_FRAMES = 6
H, W = 48, 64
FX = FY = 300.0
CX, CY = W / 2.0, H / 2.0
K = [FX, FY, CX, CY]
BOX_CENTER = [0.3, 0.0, 3.0]
BOX_EXTENT = [0.5, 0.5, 0.5]
EVIDENCE_GI = 2


def _cam_center(i: int) -> np.ndarray:
    return np.array([0.1 * i, 0.0, 0.0], dtype=np.float64)


def _write_fixture_export(base: str) -> str:
    """Write a real-schema export tree; return its base dir. numpy/pandas/cv2 only."""
    import cv2
    import pandas as pd

    for sub in ("data", "videos", "grounding", "clouds", "meta"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # --- data/trajectory.parquet : state = [tx,ty,tz, qx,qy,qz,qw] cam->world (R=I) ---
    records = []
    for i in range(N_FRAMES):
        t = _cam_center(i)
        state = [float(t[0]), float(t[1]), float(t[2]), 0.0, 0.0, 0.0, 1.0]
        records.append(
            {
                "index": i,
                "frame_index": i,
                "episode_index": 0,
                "timestamp": float(i) / 10.0,
                "source_frame_id": float(i),
                "observation.state": state,
                "action": [0.0] * 6,
                "observation.intrinsics": [FX, FY, CX, CY],
            }
        )
    pd.DataFrame.from_records(records).to_parquet(
        os.path.join(base, "data", "trajectory.parquet"), index=False
    )

    # --- videos/ego_view.mp4 : N frames of real content (gradient) ---
    vpath = os.path.join(base, "videos", "ego_view.mp4")
    writer = None
    for code in ("avc1", "mp4v", "H264", "h264"):
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*code), 10.0, (W, H))
        if writer.isOpened():
            break
        writer.release()
    if writer is None or not writer.isOpened():
        pytest.skip("no mp4 VideoWriter available in this env")
    for i in range(N_FRAMES):
        gx = np.linspace(0, 255, W, dtype=np.uint8)[None, :].repeat(H, 0)
        img = np.stack([gx, gx, np.full_like(gx, i * 30)], axis=-1)
        writer.write(img)
    writer.release()

    # --- grounding/objects.jsonl : one object with a 3D world box + evidence keyframe ---
    obj_line = {
        "object_id": 0,
        "query": "box",
        "world_center": BOX_CENTER,
        "world_extent": BOX_EXTENT,
        "confidence": 0.9,
        "description": None,
        "evidence": [
            {"submap_id": 0, "frame_idx": EVIDENCE_GI, "global_frame_index": EVIDENCE_GI}
        ],
        "relations": [],
    }
    with open(os.path.join(base, "grounding", "objects.jsonl"), "w") as fh:
        fh.write(json.dumps(obj_line) + "\n")

    # --- clouds/cloud.npz : a dense grid of world points around the object box ---
    gx = np.linspace(-0.5, 1.1, 12)
    gy = np.linspace(-0.5, 0.5, 8)
    gz = np.linspace(2.5, 3.5, 8)
    xs, ys, zs = np.meshgrid(gx, gy, gz, indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)
    colors = np.full((positions.shape[0], 3), 128, np.uint8)
    np.savez_compressed(
        os.path.join(base, "clouds", "cloud.npz"), positions=positions, colors=colors
    )
    return base


@pytest.fixture()
def export_dir(tmp_path):
    return _write_fixture_export(str(tmp_path / "scan"))


# ---------------------------------------------------------------------------
# parsing: trajectory / frames / objects / cloud
# ---------------------------------------------------------------------------

def test_load_trajectory_reconstructs_poses_and_intrinsics(export_dir):
    traj = rsa.load_trajectory(export_dir)
    assert traj["n"] == N_FRAMES
    assert sorted(traj["poses"].keys()) == list(range(N_FRAMES))
    for i in range(N_FRAMES):
        pose = traj["poses"][i]
        assert pose.shape == (4, 4)
        # identity rotation, translation = camera center along x
        assert np.allclose(pose[:3, :3], np.eye(3))
        assert np.allclose(pose[:3, 3], _cam_center(i))
        assert np.allclose(traj["intrinsics"][i], [FX, FY, CX, CY])


def test_load_frames_count_and_shape(export_dir):
    frames = rsa.load_frames(export_dir)
    # mp4 codecs can drop/dup at most a frame; require the full set within tolerance.
    assert N_FRAMES - 1 <= len(frames) <= N_FRAMES
    assert frames[0].shape == (H, W, 3)
    assert frames[0].dtype == np.uint8


def test_load_objects_carries_box_and_evidence(export_dir):
    objs = rsa.load_objects(export_dir)
    assert len(objs) == 1
    o = objs[0]
    assert o["object_id"] == 0 and o["query"] == "box"
    assert np.allclose(o["world_center"], BOX_CENTER)
    assert np.allclose(o["world_extent"], BOX_EXTENT)
    assert o["evidence_frames"] == [EVIDENCE_GI]


def test_load_cloud(export_dir):
    positions, colors = rsa.load_cloud(export_dir)
    assert positions.shape[1] == 3 and positions.shape[0] > 0
    assert colors.shape == positions.shape


# ---------------------------------------------------------------------------
# geometry: project 3D box -> 2D box, and project/unproject round-trip
# ---------------------------------------------------------------------------

def test_project_world_box_to_image_in_bounds(export_dir):
    traj = rsa.load_trajectory(export_dir)
    box = rsa.project_world_box_to_image(
        BOX_CENTER, BOX_EXTENT, traj["poses"][EVIDENCE_GI], K, H, W
    )
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= W - 1
    assert 0 <= y0 < y1 <= H - 1
    # the box center must project inside the 2D box
    uv, _z, valid = rsa.project_world_points(np.asarray([BOX_CENTER]), traj["poses"][EVIDENCE_GI], K)
    assert valid[0]
    assert x0 <= uv[0, 0] <= x1 and y0 <= uv[0, 1] <= y1


def test_project_unproject_roundtrip(export_dir):
    traj = rsa.load_trajectory(export_dir)
    pose = traj["poses"][EVIDENCE_GI]
    pw = np.asarray(BOX_CENTER, dtype=np.float64)
    uv, z, valid = rsa.project_world_points(pw[None, :], pose, K)
    assert valid[0]
    back = unproject_to_world(uv[0, 0], uv[0, 1], z[0], K, pose)
    assert np.allclose(back, pw, atol=1e-6)


# ---------------------------------------------------------------------------
# depth: cloud z-buffer is dense + scale-consistent (unprojects near a cloud point)
# ---------------------------------------------------------------------------

def test_render_cloud_depth_dense_and_scale_consistent(export_dir):
    traj = rsa.load_trajectory(export_dir)
    positions, _ = rsa.load_cloud(export_dir)
    depth = rsa.render_cloud_depth(positions, traj["poses"][EVIDENCE_GI], K, H, W)
    assert depth is not None
    assert depth.shape == (H, W)
    assert np.all(np.isfinite(depth))  # NN-filled -> no holes
    # depth is world scale: cloud z spans ~2.5..3.5 in front of a z=0 camera.
    assert 2.0 < float(np.median(depth)) < 4.0


def test_render_cloud_depth_stack_length(export_dir):
    traj = rsa.load_trajectory(export_dir)
    positions, _ = rsa.load_cloud(export_dir)
    frames_idx = list(range(N_FRAMES))
    stack = rsa.render_cloud_depth_stack(positions, traj["poses"], traj["intrinsics"], frames_idx, H, W)
    assert len(stack) == N_FRAMES
    assert all(d.shape == (H, W) for d in stack)


# ---------------------------------------------------------------------------
# end-to-end CPU: fixture export -> objects_2d -> ObjectTrack jsonl (FAKE tracker)
# ---------------------------------------------------------------------------

def _fake_tracks_from_mask(mask, frames_idx, depth_stack, k=16):
    """Deterministic stand-in for BootsTAPIR: seed points in the mask, hold them fixed
    across frames (a static object). Returns (frame_indices, tracks(T,N,2), vis(T,N))."""
    pts = sample_query_points_in_mask(mask, k, np.random.default_rng(0))
    n = pts.shape[0]
    T = len(frames_idx)
    tracks = np.broadcast_to(pts[None, :, :], (T, n, 2)).astype(np.float64).copy()
    vis = np.ones((T, n), dtype=bool)
    return tracks, vis


# ---------------------------------------------------------------------------
# prefer-persisted seeding: object_seed(...) precedence + load_objects geometry
# ---------------------------------------------------------------------------

def _identity_pose(gi):
    p = np.eye(4, dtype=np.float64)
    p[:3, 3] = _cam_center(gi)
    return p


def _seed_ctx():
    return {EVIDENCE_GI: _identity_pose(EVIDENCE_GI)}, {EVIDENCE_GI: np.asarray(K, dtype=np.float64)}


def test_object_seed_prefers_persisted_mask_and_skips_sam2():
    from server.export.mask_rle import mask_to_rle

    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 15:35] = True
    obj = {
        "world_center": BOX_CENTER, "world_extent": BOX_EXTENT,
        "evidence": [{"global_frame_index": EVIDENCE_GI, "box_2d": None, "mask_rle": mask_to_rle(mask)}],
    }
    poses, intr = _seed_ctx()
    seed = rsa.object_seed(obj, EVIDENCE_GI, poses, intr, H, W)
    assert seed["source"] == "persisted_mask"
    assert seed["mask"] is not None and np.array_equal(seed["mask"], mask)  # used directly, no SAM2
    # box is the persisted mask's tight bbox
    assert seed["box"] == (15.0, 10.0, 34.0, 19.0)


def test_object_seed_prefers_persisted_box_over_projection():
    persisted = [1.0, 2.0, 40.0, 30.0]
    obj = {
        "world_center": BOX_CENTER, "world_extent": BOX_EXTENT,
        "evidence": [{"global_frame_index": EVIDENCE_GI, "box_2d": persisted, "mask_rle": None}],
    }
    poses, intr = _seed_ctx()
    projected = rsa.project_world_box_to_image(BOX_CENTER, BOX_EXTENT, poses[EVIDENCE_GI], K, H, W)
    seed = rsa.object_seed(obj, EVIDENCE_GI, poses, intr, H, W)
    assert seed["source"] == "persisted_box"
    assert seed["mask"] is None
    assert seed["box"] == (1.0, 2.0, 40.0, 30.0)  # the persisted box, clamped to the image
    assert seed["box"] != projected  # NOT the coarser 3D-box projection


def test_object_seed_falls_back_to_projected_box_when_absent():
    obj = {
        "world_center": BOX_CENTER, "world_extent": BOX_EXTENT,
        "evidence": [{"global_frame_index": EVIDENCE_GI, "box_2d": None, "mask_rle": None}],
    }
    poses, intr = _seed_ctx()
    seed = rsa.object_seed(obj, EVIDENCE_GI, poses, intr, H, W)
    assert seed["source"] == "projected_box"
    assert seed["mask"] is None
    assert seed["box"] == rsa.project_world_box_to_image(
        BOX_CENTER, BOX_EXTENT, poses[EVIDENCE_GI], K, H, W
    )


def test_load_objects_carries_persisted_evidence_geometry(tmp_path):
    base = str(tmp_path / "scan")
    os.makedirs(os.path.join(base, "grounding"), exist_ok=True)
    rle = {"size": [H, W], "counts": [3, 4, H * W - 7], "order": "C"}
    line = {
        "object_id": 0, "query": "box",
        "world_center": BOX_CENTER, "world_extent": BOX_EXTENT, "confidence": 0.9,
        "evidence": [
            {"submap_id": 0, "frame_idx": EVIDENCE_GI, "global_frame_index": EVIDENCE_GI,
             "box_2d": [1.0, 2.0, 3.0, 4.0], "mask_rle": rle},
        ],
    }
    with open(os.path.join(base, "grounding", "objects.jsonl"), "w") as fh:
        fh.write(json.dumps(line) + "\n")
    objs = rsa.load_objects(base)
    assert objs[0]["evidence_frames"] == [EVIDENCE_GI]  # backward-compat field kept
    ev = objs[0]["evidence"][0]
    assert ev["global_frame_index"] == EVIDENCE_GI
    assert ev["box_2d"] == [1.0, 2.0, 3.0, 4.0]
    assert ev["mask_rle"] == rle


def test_fixture_export_to_object_tracks_cpu(export_dir, tmp_path):
    traj = rsa.load_trajectory(export_dir)
    frames = rsa.load_frames(export_dir)
    objs = rsa.load_objects(export_dir)
    positions, _ = rsa.load_cloud(export_dir)
    assert objs, "fixture must yield at least one object"

    obj = objs[0]
    qf = obj["evidence_frames"][0]
    # seed region: project the REAL 3D world box into the evidence keyframe (CPU fallback mask)
    box = rsa.project_world_box_to_image(
        obj["world_center"], obj["world_extent"], traj["poses"][qf], K, H, W
    )
    assert box is not None
    mask = rsa.box_to_mask(box, H, W)
    assert mask.sum() > 0

    # track causally forward from the evidence keyframe (matches BootsTAPIR's causal pass)
    frames_idx = list(range(qf, min(len(frames), traj["n"])))
    depth_stack = rsa.render_cloud_depth_stack(
        positions, traj["poses"], traj["intrinsics"], frames_idx, H, W
    )
    tracks, vis = _fake_tracks_from_mask(mask, frames_idx, depth_stack)

    objects_2d = assemble_objects_2d(
        object_ids=[obj["object_id"]],
        queries=[obj["query"]],
        frame_indices=frames_idx,
        point_ranges=[(0, tracks.shape[1])],
        tracks=tracks,
        visibles=vis,
        depth_maps=depth_stack,
    )
    # exactly the keys build_object_tracks consumes
    assert set(objects_2d[0]) == {
        "id", "query", "frames", "centroids", "depths", "visibility", "confidence"
    }
    assert objects_2d[0]["frames"] == frames_idx
    assert all(np.isfinite(d) for d in objects_2d[0]["depths"])  # real cloud depth

    # frames are plain global ints -> index_map can be empty
    result = build_object_tracks(objects_2d, traj["poses"], traj["intrinsics"], {})
    assert len(result) == 1 and isinstance(result[0], ObjectTrack)
    tr = result[0]
    assert tr.query == "box"
    assert tr.frames == frames_idx
    assert all(np.all(np.isfinite(c)) for c in tr.frame_centers)
    # the lifted world center should sit near the real 3D box (up-to-scale, coarse mask seed)
    lifted = np.asarray(tr.center, dtype=np.float64)
    assert np.linalg.norm(lifted - np.asarray(BOX_CENTER)) < 1.5

    n = write_object_tracks_jsonl(result, str(tmp_path / "out"))
    assert n == 1
    out = os.path.join(str(tmp_path / "out"), "dynamics", "objects_tracks.jsonl")
    rec = json.loads(open(out).readline())
    for key in ("query", "object_id", "frames", "frame_centers", "motion", "max_displacement"):
        assert key in rec
