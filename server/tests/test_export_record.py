"""Stage-4 tests: GPU-free export from a persisted record (poses match a live export).

See docs/dataset-export.md §8 Stage 4. Verifies (a) ``trajectory.npz`` round-trips through
``ModalScenePersistence``, (b) ``build_rows_from_trajectory`` reproduces the live Stage-1
rows, and (c) ``export_from_record`` writes a ``trajectory.parquet`` matching a live export.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

# build_trajectory_arrays and extract_trajectory both call
# vggt_slam.slam_utils.decompose_camera at runtime.
# Skip this whole module when openreality-core is not installed.
# Add the GH_PAT secret at:
#   https://github.com/reality-opened/server/settings/secrets/actions
pytest.importorskip(
    "vggt_slam.slam_utils",
    reason=(
        "openreality-core not installed — build_trajectory_arrays/extract_trajectory "
        "require vggt_slam.slam_utils.decompose_camera.  Set GH_PAT to enable."
    ),
)

from export_fakes import FakeMap, FakeSolver, FakeSubmap, rot

from server.export.record import (
    build_rows_from_trajectory,
    build_trajectory_arrays,
    load_record_from_store,
)
from server.export.trajectory import extract_trajectory
from server.export.writer import export_from_record
from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport
from server.scene_report.store import ModalScenePersistence


def _solver():
    s0 = FakeSubmap(
        0,
        cameras=[(np.eye(3), [0.0, 0.0, 0.0]), (rot(rz=0.3), [1.0, 0.0, 0.0])],
        frame_ids=[0.0, 1.0],
    )
    s2 = FakeSubmap(2, cameras=[(np.eye(3), [2.0, 0.0, 0.0])], frame_ids=[10.0])
    return FakeSolver(FakeMap([s0, s2]))


def test_build_rows_from_trajectory_matches_live():
    solver = _solver()
    live_rows, _ = extract_trajectory(solver)
    traj = build_trajectory_arrays(solver)
    assert traj["poses"].shape == (3, 4, 4)
    assert traj["intrinsics"].shape == (3, 4)

    rec_rows = build_rows_from_trajectory(traj)
    assert len(rec_rows) == len(live_rows)
    for live, rec in zip(live_rows, rec_rows):
        assert rec["observation.state"] == pytest.approx(live["observation.state"], abs=1e-4)
        assert rec["action"] == pytest.approx(live["action"], abs=1e-4)
        assert float(rec["source_frame_id"]) == pytest.approx(float(live["source_frame_id"]))


def test_export_from_record_parquet_matches_live(tmp_path):
    solver = _solver()

    # live Stage-1 trajectory
    import pandas as pd

    from server.export.trajectory import write_trajectory_parquet

    live_rows, _ = extract_trajectory(solver)
    live_path = str(tmp_path / "live.parquet")
    write_trajectory_parquet(live_rows, live_path)
    live_df = pd.read_parquet(live_path)

    # record export (no Solver in the call)
    record = {
        "scan_id": "scanZ",
        "trajectory": build_trajectory_arrays(solver),
        "facts": SceneFacts(metrics=SceneMetrics(num_keyframes=3)).model_dump(),
        "report": SceneReport(summary="rec", room_type="lab").model_dump(),
        "cloud": (np.zeros((3, 3), np.float32), np.zeros((3, 3), np.uint8)),
        "keyframes": [],
    }
    base = export_from_record(record, str(tmp_path / "out"))
    rec_df = pd.read_parquet(os.path.join(base, "data", "trajectory.parquet"))

    assert len(rec_df) == len(live_df)
    live_state = np.vstack(live_df["observation.state"].to_numpy())
    rec_state = np.vstack(rec_df["observation.state"].to_numpy())
    assert rec_state == pytest.approx(live_state, abs=1e-4)
    live_action = np.vstack(live_df["action"].to_numpy())
    rec_action = np.vstack(rec_df["action"].to_numpy())
    assert rec_action == pytest.approx(live_action, abs=1e-4)

    # sidecar JSON from the persisted dicts
    assert os.path.isfile(os.path.join(base, "meta", "scene_graph.json"))
    assert json.load(open(os.path.join(base, "meta", "report.json")))["room_type"] == "lab"
    with np.load(os.path.join(base, "clouds", "cloud.npz")) as cz:
        assert cz["positions"].shape == (3, 3)


def test_store_persists_and_reads_trajectory(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path))
    solver = _solver()
    traj = build_trajectory_arrays(solver)
    report = SceneReport(summary="s", room_type="office", facts=SceneFacts())

    store.save_scene("user-a", "scan1", report, SceneFacts(), trajectory=traj)

    rec = store.get_scene("user-a", "scan1")
    assert rec["trajectory_key"] == "trajectory.npz"
    assert rec["trajectory_count"] == 3

    loaded = store.get_trajectory("user-a", "scan1")
    assert loaded["poses"].shape == (3, 4, 4)
    assert loaded["poses"] == pytest.approx(traj["poses"], abs=1e-4)
    assert loaded["source_frame_id"] == pytest.approx([0.0, 1.0, 10.0])

    # end-to-end: assemble a normalized record off the store and export it GPU-free
    normalized = load_record_from_store(store, "user-a", "scan1")
    assert normalized is not None
    base = export_from_record(normalized, str(tmp_path / "out"))
    import pandas as pd

    df = pd.read_parquet(os.path.join(base, "data", "trajectory.parquet"))
    assert len(df) == 3


def test_store_without_trajectory_has_no_key(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene("u", "s", SceneReport(), SceneFacts())
    rec = store.get_scene("u", "s")
    assert rec.get("trajectory_key") is None
    assert store.get_trajectory("u", "s") is None
    assert load_record_from_store(store, "u", "s") is None
