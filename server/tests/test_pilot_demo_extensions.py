"""W1 pilot-extension tests (docs/demo-2026-07): ``--persist-trajectory`` +
``--demo-index`` in ``server.reconstruct_pilot`` — GPU-free.

Import pattern mirrors tests/test_pilot_reconstruct.py: ``load_app_module`` stubs
flask/torch/etc., then ``server.reconstruct_pilot`` is imported fresh so its
module-level ``from server import app`` binds the stubbed module. The extensions
are exercised at the seam they live on (the persistence wrapper around a real
``ModalScenePersistence``), plus a fake-solver run of the trajectory/index build.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import numpy as np

from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport
from server.scene_report.store import ModalScenePersistence

from conftest import load_app_module


def _load_recon(monkeypatch):
    load_app_module(monkeypatch)
    sys.modules.pop("server.reconstruct_pilot", None)
    return importlib.import_module("server.reconstruct_pilot")


def _report():
    return SceneReport(summary="s", room_type="office")


def _facts():
    return SceneFacts(metrics=SceneMetrics(num_submaps=2))


def _traj(n=3):
    poses = np.stack([np.eye(4, dtype=np.float32) * 1.0 for _ in range(n)])
    for i in range(n):
        poses[i, 0, 3] = float(i)  # distinct translations so rows are tellable-apart
        poses[i, 3, 3] = 1.0
    intrinsics = np.tile(
        np.asarray([500.0, 501.0, 320.0, 240.0], dtype=np.float32), (n, 1)
    )
    intrinsics[:, 0] += np.arange(n, dtype=np.float32)
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "source_frame_id": np.arange(n, dtype=np.float32),
    }


def _keyframes():
    return [
        {"submap_id": 0, "frame_idx": 0, "image_b64": "aGk="},
        {"submap_id": 0, "frame_idx": 2, "image_b64": "aGk="},
        {"submap_id": 9, "frame_idx": 1, "image_b64": "aGk="},   # no traj row (LC submap)
        {"submap_id": 1, "frame_idx": 0},                        # no image → store skips it
    ]


_INDEX_MAP = {(0, 0): 0, (0, 2): 1, (1, 0): 2}


# ---------------------------------------------------------------------------
# build_frames_index (pure)
# ---------------------------------------------------------------------------


def test_build_frames_index_entries_and_envelope(monkeypatch):
    recon = _load_recon(monkeypatch)
    doc = recon.build_frames_index(
        _keyframes(), _traj(), _INDEX_MAP, "scan-x", "trajectory.npz"
    )
    # WORLD-TRANSFORM-CONTRACT doc envelope
    assert doc["version"] == 1
    assert doc["scan_id"] == "scan-x"
    assert doc["parent_artifact"] == "trajectory.npz"
    assert doc["created_at"] > 0
    assert "run_id" in doc

    frames = doc["frames"]
    assert doc["count"] == len(frames) == 3  # image-less keyframe skipped

    by_key = {f["blob_key"]: f for f in frames}
    assert set(by_key) == {"0_0.jpg", "0_2.jpg", "9_1.jpg"}

    f00 = by_key["0_0.jpg"]
    assert f00["submap_id"] == 0 and f00["frame_idx"] == 0
    assert f00["traj_row"] == 0
    assert f00["c2w"][0][3] == 0.0
    assert f00["intrinsics"] == [500.0, 501.0, 320.0, 240.0]

    f02 = by_key["0_2.jpg"]
    assert f02["traj_row"] == 1
    assert f02["c2w"][0][3] == 1.0  # row 1's pose, not row 2's
    assert f02["intrinsics"][0] == 501.0

    # keyframe with no trajectory row keeps the blob mapping, nulls the pose
    f91 = by_key["9_1.jpg"]
    assert f91["traj_row"] is None and f91["c2w"] is None and f91["intrinsics"] is None

    json.dumps(doc)  # must be wire-clean


def test_build_frames_index_without_trajectory(monkeypatch):
    recon = _load_recon(monkeypatch)
    doc = recon.build_frames_index(_keyframes(), None, None, "scan-y", None)
    assert doc["count"] == 3
    assert all(f["traj_row"] is None and f["c2w"] is None for f in doc["frames"])
    assert doc["parent_artifact"] is None


def test_build_frames_index_empty_inputs(monkeypatch):
    recon = _load_recon(monkeypatch)
    doc = recon.build_frames_index(None, None, None, "scan-z", None)
    assert doc["count"] == 0 and doc["frames"] == []


# ---------------------------------------------------------------------------
# _wrap_demo_persistence / _DemoExtensionsPersistence
# ---------------------------------------------------------------------------


def test_wrap_is_identity_when_flags_off(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    assert recon._wrap_demo_persistence(store, False, False) is store


def test_wrapper_threads_trajectory_and_writes_index(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    wrapper = recon._wrap_demo_persistence(store, True, True)
    assert wrapper is not store
    wrapper._build_trajectory_and_index = lambda: (_traj(), dict(_INDEX_MAP))

    scan = wrapper.save_scene(
        "user-a", "scan1", _report(), _facts(),
        keyframes_b64=_keyframes(),
        points=(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.uint8)),
        label="canonical",
    )
    assert scan == "scan1"

    record = store.get_scene("user-a", "scan1")
    assert record["trajectory_key"] == "trajectory.npz"
    assert record["trajectory_count"] == 3
    assert record["label"] == "canonical"

    # the persisted trajectory round-trips through the store's reader
    traj = store.get_trajectory("user-a", "scan1")
    assert traj["poses"].shape == (3, 4, 4)
    np.testing.assert_allclose(traj["poses"][1, 0, 3], 1.0)

    raw = store.get_derived_artifact("user-a", "scan1", "derived/demo/frames_index.json")
    assert raw is not None
    doc = json.loads(raw.decode("utf-8"))
    assert doc["scan_id"] == "scan1"
    assert doc["parent_artifact"] == "trajectory.npz"
    assert doc["count"] == 3
    assert doc["frames"][0]["blob_key"] == "0_0.jpg"


def test_wrapper_index_only_leaves_trajectory_unpersisted(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    wrapper = recon._wrap_demo_persistence(store, True, False)
    wrapper._build_trajectory_and_index = lambda: (_traj(), dict(_INDEX_MAP))

    wrapper.save_scene("user-a", "scan2", _report(), _facts(), keyframes_b64=_keyframes())
    record = store.get_scene("user-a", "scan2")
    assert record["trajectory_key"] is None  # --demo-index alone doesn't persist npz
    doc = json.loads(
        store.get_derived_artifact("user-a", "scan2", "derived/demo/frames_index.json")
    )
    assert doc["parent_artifact"] == "cloud.npz"
    # ...but the index still carries the poses (computed live from the solver)
    assert doc["frames"][0]["c2w"] is not None


def test_wrapper_degrades_when_no_solver(monkeypatch, tmp_path):
    """No live solver (or an empty map) → scene persists exactly as before, and the
    frames index is still written with null poses."""
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    wrapper = recon._wrap_demo_persistence(store, True, True)
    monkeypatch.setattr(recon.app_module, "slam_processor", None, raising=False)

    wrapper.save_scene("user-a", "scan3", _report(), _facts(), keyframes_b64=_keyframes())
    record = store.get_scene("user-a", "scan3")
    assert record is not None
    assert record["trajectory_key"] is None
    doc = json.loads(
        store.get_derived_artifact("user-a", "scan3", "derived/demo/frames_index.json")
    )
    assert doc["count"] == 3
    assert all(f["c2w"] is None for f in doc["frames"])


def test_wrapper_index_write_failure_never_aborts_persist(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    wrapper = recon._wrap_demo_persistence(store, True, True)
    wrapper._build_trajectory_and_index = lambda: (_traj(), dict(_INDEX_MAP))

    def _boom(*a, **k):
        raise OSError("volume hiccup")

    monkeypatch.setattr(store, "save_derived_artifact", _boom)
    scan = wrapper.save_scene("user-a", "scan4", _report(), _facts(),
                              keyframes_b64=_keyframes())
    assert scan == "scan4"
    assert store.get_scene("user-a", "scan4") is not None


def test_wrapper_forwards_reads_to_inner(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    wrapper = recon._wrap_demo_persistence(store, True, True)
    wrapper._build_trajectory_and_index = lambda: (None, None)
    wrapper.save_scene("user-a", "scan5", _report(), _facts())
    # __getattr__ forwarding: reads go straight to the wrapped store
    assert wrapper.get_scene("user-a", "scan5")["scan_id"] == "scan5"
    assert wrapper.list_scenes("user-a")[0]["scan_id"] == "scan5"


# ---------------------------------------------------------------------------
# _build_trajectory_and_index against a fake solver (the real export-helper path)
# ---------------------------------------------------------------------------


class _FakeSubmap:
    def __init__(self, sid, n, lc=False, base=0.0):
        self._sid, self._n, self._lc, self._base = sid, n, lc, base

    def get_id(self):
        return self._sid

    def get_lc_status(self):
        return self._lc

    def get_all_poses_world(self, graph, give_camera_mat=True):
        out = []
        for i in range(self._n):
            m = np.eye(4)
            m[0, 3] = self._base + i
            out.append(m)
        return np.asarray(out)

    def get_frame_ids(self):
        return [self._base + i for i in range(self._n)]


class _FakeMap:
    def __init__(self, submaps):
        self._submaps = submaps

    def ordered_submaps_by_key(self):
        return list(self._submaps)


def test_build_trajectory_and_index_from_fake_solver(monkeypatch, tmp_path):
    recon = _load_recon(monkeypatch)

    # decompose_camera: identity K, pass the pose through (P = [R|t] here)
    slam_utils = types.ModuleType("vggt_slam.slam_utils")

    def _decompose(P):
        P = np.asarray(P, dtype=np.float64)
        return np.eye(3), P[:3, :3], P[:3, 3], 1.0

    slam_utils.decompose_camera = _decompose
    monkeypatch.setitem(sys.modules, "vggt_slam.slam_utils", slam_utils)

    solver = types.SimpleNamespace(
        graph=None,
        map=_FakeMap([
            _FakeSubmap(0, 2, base=0.0),
            _FakeSubmap(1, 1, lc=True, base=100.0),  # loop-closure → skipped
            _FakeSubmap(2, 2, base=10.0),
        ]),
    )
    monkeypatch.setattr(
        recon.app_module, "slam_processor",
        types.SimpleNamespace(solver=solver), raising=False,
    )

    wrapper = recon._wrap_demo_persistence(
        ModalScenePersistence({}, str(tmp_path)), True, True
    )
    traj, index_map = wrapper._build_trajectory_and_index()
    assert traj["poses"].shape == (4, 4, 4)  # 2 + 2, LC submap skipped
    assert list(traj["source_frame_id"]) == [0.0, 1.0, 10.0, 11.0]
    assert index_map == {(0, 0): 0, (0, 1): 1, (2, 0): 2, (2, 1): 3}
    np.testing.assert_allclose(traj["poses"][3, 0, 3], 11.0)
