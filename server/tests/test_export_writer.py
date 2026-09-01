"""Stage 1-2 end-to-end test: export_scene assembles the full OpenReality tree (with fakes).

See docs/dataset-export.md §8 Stage 1-2 / §9. Asserts the trajectory dataset + grounding
sidecar are produced and internally consistent (frame<->row count, evidence join, object
centers inside the cloud AABB). Skips if no mp4 VideoWriter is available.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

# export_scene -> extract_trajectory -> vggt_slam.slam_utils.decompose_camera.
# Skip this whole module when openreality-core is not installed.
# Add the GH_PAT secret at:
#   https://github.com/reality-opened/server/settings/secrets/actions
pytest.importorskip(
    "vggt_slam.slam_utils",
    reason=(
        "openreality-core not installed — export_scene requires "
        "vggt_slam.slam_utils.decompose_camera.  Set GH_PAT to enable."
    ),
)

from export_fakes import FakeMap, FakeSolver, FakeSubmap, make_frames, rot

from server.export.writer import export_scene
from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    SceneFacts,
    SceneMetrics,
    SceneReport,
)


def _cloud_points():
    # a dense-ish cube spanning [0,2]^3 so object centers sit inside the AABB
    g = np.linspace(0, 2, 5)
    xs, ys, zs = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    cols = np.full((pts.shape[0], 3), 128, dtype=np.uint8)
    return pts, cols


def _solver():
    pts, cols = _cloud_points()
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
    return FakeSolver(FakeMap([s0, s1]))


def _report():
    facts = SceneFacts(
        metrics=SceneMetrics(num_submaps=2, num_keyframes=3, trajectory_length=2.0),
        objects=[
            ObjectInstance(
                query="laptop",
                center=[1.0, 1.0, 1.0],  # inside the [0,2]^3 cloud AABB
                extent=[0.4, 0.3, 0.02],
                confidence=0.8,
                evidence=[EvidenceRef(submap_id=0, frame_idx=0)],
            )
        ],
        object_counts={"laptop": 1},
    )
    return SceneReport(summary="A small office.", room_type="office", facts=facts)


def test_export_scene_full_tree(tmp_path):
    solver = _solver()
    try:
        base = export_scene(
            solver, detections=[], report=_report(), out_dir=str(tmp_path), scan_id="scanX"
        )
    except RuntimeError as exc:
        if "VideoWriter" in str(exc):
            pytest.skip(f"no mp4 VideoWriter in this env: {exc}")
        raise

    assert base == os.path.join(str(tmp_path), "scanX")
    # required tree
    for rel in (
        "meta/info.json",
        "meta/episode.json",
        "meta/scene_graph.json",
        "meta/report.json",
        "meta/annotations.jsonl",
        "data/trajectory.parquet",
        "videos/ego_view.mp4",
        "clouds/cloud.npz",
        "grounding/objects.jsonl",
    ):
        assert os.path.isfile(os.path.join(base, rel)), f"missing {rel}"

    # info.json declares the non-metric caveats (non-negotiable, §0)
    info = json.load(open(os.path.join(base, "meta", "info.json")))
    assert info["up_to_scale"] is True
    assert info["gravity_aligned"] is False
    assert info["scan_id"] == "scanX"
    assert info["features"]["observation.state"]["shape"] == [7]

    # episode summary
    ep = json.load(open(os.path.join(base, "meta", "episode.json")))
    assert ep["num_keyframes"] == 3
    assert ep["room_type"] == "office"
    assert ep["num_objects"] == 1

    # trajectory.parquet: N rows, last action zero, all finite
    import pandas as pd

    df = pd.read_parquet(os.path.join(base, "data", "trajectory.parquet"))
    assert len(df) == 3
    assert np.asarray(df["action"].iloc[-1]) == pytest.approx(np.zeros(6))
    assert np.all(np.isfinite(np.vstack(df["observation.state"].to_numpy())))

    # cloud.npz loads and object center is inside its AABB (shared world frame, §9.2)
    with np.load(os.path.join(base, "clouds", "cloud.npz")) as cz:
        positions = cz["positions"]
    assert positions.shape[1] == 3 and positions.shape[0] > 0
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    obj = json.loads(open(os.path.join(base, "grounding", "objects.jsonl")).readline())
    center = np.asarray(obj["world_center"])
    assert np.all(center >= lo - 1e-6) and np.all(center <= hi + 1e-6)
    assert obj["evidence"][0]["global_frame_index"] == 0  # (submap 0, frame 0) -> 0

    # annotation windows all within [0, N)
    n = len(df)
    for line in open(os.path.join(base, "meta", "annotations.jsonl")):
        ann = json.loads(line)
        assert 0 <= ann["frame_start"] <= ann["frame_end"] < n


def test_export_scene_survives_missing_report(tmp_path):
    """No report (offline main.py path): trajectory dataset still produced; facts derived."""
    solver = _solver()
    try:
        base = export_scene(
            solver, detections=[], report=None, out_dir=str(tmp_path), scan_id="noreport"
        )
    except RuntimeError as exc:
        if "VideoWriter" in str(exc):
            pytest.skip(f"no mp4 VideoWriter in this env: {exc}")
        raise
    assert os.path.isfile(os.path.join(base, "data", "trajectory.parquet"))
    assert os.path.isfile(os.path.join(base, "clouds", "cloud.npz"))
    # scene_graph derived from the solver even without a report
    assert os.path.isfile(os.path.join(base, "meta", "scene_graph.json"))
