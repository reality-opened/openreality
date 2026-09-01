"""Stage-3 tests: OpenReality tree -> GR00T-LeRobot v2 invariants.

See docs/dataset-export.md §5 / Stage 3. We can't load real GR00T here (no GPU/package), so
we assert the structural contract the doc lists: modality index ranges == feature widths,
tasks.jsonl indices referenced by the annotation columns all exist, and the v2 layout.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

# Most tests call export_scene -> extract_trajectory ->
# vggt_slam.slam_utils.decompose_camera.  Skip this whole module when
# openreality-core is not installed.
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

from server.export.to_lerobot import build_modality_json, convert_to_groot_lerobot
from server.export.writer import export_scene
from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport


def test_modality_index_ranges_match_feature_widths():
    m = build_modality_json()
    assert m["state"]["camera_pose"]["end"] - m["state"]["camera_pose"]["start"] == 7
    assert m["action"]["ego_motion"]["end"] - m["action"]["ego_motion"]["start"] == 6
    assert m["video"]["ego_view"]["original_key"] == "observation.images.ego_view"


def _make_export(tmp_path):
    s0 = FakeSubmap(
        0,
        cameras=[(np.eye(3), [0, 0, 0]), (rot(rz=0.2), [1, 0, 0])],
        frame_ids=[0.0, 1.0],
        frames=make_frames(2),
        points=np.zeros((4, 3)),
        colors=np.zeros((4, 3), np.uint8),
    )
    solver = FakeSolver(FakeMap([s0]))
    report = SceneReport(
        summary="A tidy office.", room_type="office", facts=SceneFacts(metrics=SceneMetrics())
    )
    return export_scene(
        solver, detections=[], report=report, out_dir=str(tmp_path), scan_id="scanY"
    )


def test_convert_to_groot_lerobot_layout_and_indices(tmp_path):
    try:
        export_dir = _make_export(tmp_path / "export")
    except RuntimeError as exc:
        if "VideoWriter" in str(exc):
            pytest.skip(f"no mp4 VideoWriter in this env: {exc}")
        raise

    out = str(tmp_path / "groot")
    convert_to_groot_lerobot(export_dir, out)

    # v2 on-disk layout
    for rel in (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
    ):
        assert os.path.isfile(os.path.join(out, rel)), f"missing {rel}"

    info = json.load(open(os.path.join(out, "meta", "info.json")))
    assert info["codebase_version"] == "v2.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2

    # tasks.jsonl indices referenced by the annotation columns all exist
    import pandas as pd

    df = pd.read_parquet(os.path.join(out, "data", "chunk-000", "episode_000000.parquet"))
    assert len(df) == 2
    for col in ("observation.state", "action", "task_index",
                "annotation.human.action.task_description", "annotation.scene.caption"):
        assert col in df.columns
    assert len(df["observation.state"].iloc[0]) == 7
    assert len(df["action"].iloc[0]) == 6

    tasks = [json.loads(l) for l in open(os.path.join(out, "meta", "tasks.jsonl"))]
    valid_ids = {t["task_index"] for t in tasks}
    for col in ("task_index", "annotation.human.action.task_description",
                "annotation.scene.caption"):
        assert set(int(v) for v in df[col]).issubset(valid_ids)

    # the scene caption made it into the task table
    assert any(t["task"] == "A tidy office." for t in tasks)

    episodes = [json.loads(l) for l in open(os.path.join(out, "meta", "episodes.jsonl"))]
    assert episodes[0]["episode_index"] == 0 and episodes[0]["length"] == 2
