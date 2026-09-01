"""Dynamics-export tests: per-frame world lift + motion label + jsonl + non-breaking wiring.

Mirrors ``test_export_grounding.py`` (real pydantic schemas, duck-typed, no GPU / no tracker).
The producer is tracker-agnostic: these tests feed it already-computed 2D tracks + depth +
poses, exactly as a permissive tracker (e.g. BootsTAPIR) would.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from server.export.dynamics import (
    build_object_tracks,
    unproject_to_world,
    write_object_tracks_jsonl,
)
from server.scene_report.schemas import ObjectTrack


# intrinsics in the trajectory convention: [fx, fy, cx, cy]
K = [600.0, 600.0, 320.0, 240.0]


def _pose(R=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


# ---------------------------------------------------------------------------
# (a) geometry: unproject a camera-frame point to the correct world coord
# ---------------------------------------------------------------------------

def test_unproject_principal_point_uses_pose_translation():
    # A pixel at the principal point unprojects to (0, 0, depth) in camera frame; with an
    # identity-rotation cam->world pose the world point is just t + [0,0,depth].
    world = unproject_to_world(320.0, 240.0, 5.0, K, _pose(t=(10.0, 20.0, 30.0)))
    assert world == pytest.approx([10.0, 20.0, 35.0])


def test_unproject_offaxis_pixel_scales_by_depth_over_focal():
    # u = cx + 60, depth 5, fx 600 -> x = 60/600 * 5 = 0.5; y unchanged; z = depth.
    world = unproject_to_world(380.0, 240.0, 5.0, K, _pose(t=(10.0, 20.0, 30.0)))
    assert world == pytest.approx([10.5, 20.0, 35.0])


def test_build_object_tracks_lifts_point_with_known_pose():
    # single object, single frame; principal-point centroid, depth 5, pose t=(1,2,3).
    index_map = {(0, 0): 0}
    poses = {0: _pose(t=(1.0, 2.0, 3.0))}
    intr = {0: K}
    objs = [
        {
            "id": 7,
            "query": "cat",
            "frames": [(0, 0)],  # (submap_id, local) resolved through index_map
            "centroids": [(320.0, 240.0)],
            "depths": [5.0],
            "visibility": [True],
            "confidence": [0.9],
        }
    ]
    tracks = build_object_tracks(objs, poses, intr, index_map)
    assert len(tracks) == 1
    tr = tracks[0]
    assert isinstance(tr, ObjectTrack)
    assert tr.object_id == 7 and tr.query == "cat"
    assert tr.frames == [0]
    assert tr.frame_centers[0] == pytest.approx([1.0, 2.0, 8.0])  # t + [0,0,5]
    assert tr.center == pytest.approx([1.0, 2.0, 8.0])  # time-average over the one visible frame
    assert tr.confidence == pytest.approx(0.9)
    assert tr.first_frame == 0 and tr.last_frame == 0


# ---------------------------------------------------------------------------
# (b) motion label: static vs translating object
# ---------------------------------------------------------------------------

def _two_frame_object(oid, query, centroids):
    return {
        "id": oid,
        "query": query,
        "frames": [0, 1],  # global indices directly
        "centroids": centroids,
        "depths": [5.0, 5.0],
        "visibility": [True, True],
        "confidence": [1.0, 1.0],
    }


def test_motion_label_static_vs_moving():
    index_map = {(0, 0): 0, (0, 1): 1}
    poses = {0: _pose(), 1: _pose()}  # identity cam->world for both frames
    intr = {0: K, 1: K}

    # static: identical centroid both frames -> zero displacement.
    static = _two_frame_object(0, "plant", [(320.0, 240.0), (320.0, 240.0)])
    # moving: centroid slides by 600 px -> x moves 600/600*5 = 5 world units.
    moving = _two_frame_object(1, "person", [(320.0, 240.0), (920.0, 240.0)])

    tracks = build_object_tracks([static, moving], poses, intr, index_map, move_threshold=0.05)
    by_q = {t.query: t for t in tracks}

    assert by_q["plant"].motion == "static"
    assert by_q["plant"].max_displacement == pytest.approx(0.0)

    assert by_q["person"].motion == "moving"
    assert by_q["person"].max_displacement == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# (c) write_object_tracks_jsonl -> valid jsonl with the ObjectTrack fields
# ---------------------------------------------------------------------------

def test_write_object_tracks_jsonl(tmp_path):
    index_map = {(0, 0): 0, (0, 1): 1}
    poses = {0: _pose(), 1: _pose(t=(1.0, 0.0, 0.0))}
    intr = {0: K, 1: K}
    objs = [_two_frame_object(3, "dog", [(320.0, 240.0), (320.0, 240.0)])]
    tracks = build_object_tracks(objs, poses, intr, index_map)

    n = write_object_tracks_jsonl(tracks, str(tmp_path))
    assert n == 1

    out = os.path.join(str(tmp_path), "dynamics", "objects_tracks.jsonl")
    assert os.path.isfile(out)  # dynamics/ subdir created

    lines = [json.loads(l) for l in open(out)]
    assert len(lines) == 1
    rec = lines[0]
    for key in (
        "query",
        "object_id",
        "center",
        "extent",
        "confidence",
        "frames",
        "frame_centers",
        "frame_extents",
        "frame_visibility",
        "frame_confidence",
        "first_frame",
        "last_frame",
        "motion",
        "max_displacement",
    ):
        assert key in rec, f"missing ObjectTrack field {key}"
    assert rec["object_id"] == 3
    assert rec["query"] == "dog"
    assert rec["frames"] == [0, 1]
    assert len(rec["frame_centers"]) == 2
    assert rec["motion"] in ("static", "moving")


def test_unresolved_and_missing_pose_frames_dropped():
    # frame key (9, 9) is not in index_map; global 1 has no pose -> both dropped gracefully.
    index_map = {(0, 0): 0, (0, 1): 1}
    poses = {0: _pose(t=(0.0, 0.0, 0.0))}  # no pose for global 1
    intr = {0: K, 1: K}
    objs = [
        {
            "id": 0,
            "query": "ball",
            "frames": [(0, 0), (0, 1), (9, 9)],
            "centroids": [(320.0, 240.0), (320.0, 240.0), (320.0, 240.0)],
            "depths": [5.0, 5.0, 5.0],
        }
    ]
    tracks = build_object_tracks(objs, poses, intr, index_map)
    assert tracks[0].frames == [0]  # only the frame with a resolvable pose survives


# ---------------------------------------------------------------------------
# (d) non-breaking: export_scene without object_tracks writes no dynamics/ dir
# ---------------------------------------------------------------------------

def _fixture_solver_and_report():
    """Same fake solver/report shape as test_export_writer (needs openreality-core)."""
    from export_fakes import FakeMap, FakeSolver, FakeSubmap, make_frames, rot
    from server.scene_report.schemas import (
        EvidenceRef,
        ObjectInstance,
        SceneFacts,
        SceneMetrics,
        SceneReport,
    )

    g = np.linspace(0, 2, 5)
    xs, ys, zs = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    cols = np.full((pts.shape[0], 3), 128, dtype=np.uint8)
    s0 = FakeSubmap(
        0,
        cameras=[(np.eye(3), [0.2, 0.2, 0.2]), (rot(rz=0.2), [1.0, 0.2, 0.2])],
        frame_ids=[0.0, 1.0],
        frames=make_frames(2),
        points=pts,
        colors=cols,
    )
    s1 = FakeSubmap(
        2,
        cameras=[(np.eye(3), [1.5, 0.2, 0.2])],
        frame_ids=[2.0],
        frames=make_frames(1),
        points=pts,
        colors=cols,
    )
    solver = FakeSolver(FakeMap([s0, s1]))
    facts = SceneFacts(
        metrics=SceneMetrics(num_submaps=2, num_keyframes=3, trajectory_length=2.0),
        objects=[
            ObjectInstance(
                query="laptop",
                center=[1.0, 1.0, 1.0],
                extent=[0.4, 0.3, 0.02],
                confidence=0.8,
                evidence=[EvidenceRef(submap_id=0, frame_idx=0)],
            )
        ],
        object_counts={"laptop": 1},
    )
    report = SceneReport(summary="A small office.", room_type="office", facts=facts)
    return solver, report


# export_scene -> extract_trajectory -> vggt_slam.slam_utils.decompose_camera.
_STATIC_FILES = (
    "meta/info.json",
    "meta/episode.json",
    "meta/scene_graph.json",
    "meta/report.json",
    "meta/annotations.jsonl",
    "data/trajectory.parquet",
    "videos/ego_view.mp4",
    "clouds/cloud.npz",
    "grounding/objects.jsonl",
)


def test_export_scene_without_tracks_has_no_dynamics_dir(tmp_path):
    pytest.importorskip(
        "vggt_slam.slam_utils",
        reason="openreality-core not installed — export_scene needs decompose_camera.",
    )
    from server.export.writer import export_scene

    solver, report = _fixture_solver_and_report()
    try:
        base = export_scene(
            solver, detections=[], report=report, out_dir=str(tmp_path), scan_id="nodyn"
        )
    except RuntimeError as exc:
        if "VideoWriter" in str(exc):
            pytest.skip(f"no mp4 VideoWriter in this env: {exc}")
        raise

    # the static tree is intact ...
    for rel in _STATIC_FILES:
        assert os.path.isfile(os.path.join(base, rel)), f"missing {rel}"
    # ... and no dynamics/ sidecar exists by default (byte-identical static export).
    assert not os.path.exists(os.path.join(base, "dynamics"))


def test_export_scene_with_tracks_writes_dynamics_sidecar(tmp_path):
    pytest.importorskip(
        "vggt_slam.slam_utils",
        reason="openreality-core not installed — export_scene needs decompose_camera.",
    )
    from server.export.writer import export_scene

    solver, report = _fixture_solver_and_report()
    track = ObjectTrack(
        query="person",
        object_id=0,
        center=[1.0, 1.0, 1.0],
        frames=[0, 1],
        frame_centers=[[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]],
        frame_visibility=[True, True],
        frame_confidence=[0.9, 0.9],
        first_frame=0,
        last_frame=1,
        motion="moving",
        max_displacement=0.2,
    )
    try:
        base = export_scene(
            solver,
            detections=[],
            report=report,
            out_dir=str(tmp_path),
            scan_id="withdyn",
            object_tracks=[track],
        )
    except RuntimeError as exc:
        if "VideoWriter" in str(exc):
            pytest.skip(f"no mp4 VideoWriter in this env: {exc}")
        raise

    # static tree still present + the optional dynamics sidecar now exists.
    for rel in _STATIC_FILES:
        assert os.path.isfile(os.path.join(base, rel)), f"missing {rel}"
    dyn = os.path.join(base, "dynamics", "objects_tracks.jsonl")
    assert os.path.isfile(dyn)
    rec = json.loads(open(dyn).readline())
    assert rec["query"] == "person" and rec["motion"] == "moving"
