"""Contract tests for ModalScenePersistence (Phase 4 durable store) + cloud_io.

Uses a plain dict as the injected KV store (the modal.Dict stand-in) and a
``tmp_path`` as the blob root, so it runs with no Modal/GPU. Verifies the
save -> list -> get round-trip, per-user scoping, index ordering/dedup, the
full-fidelity world-frame point-cloud artifact (``cloud.npz``) + legacy
``points.bin`` back-compat, unlimited-by-default retention, the PLY/downsample
helpers, and delete.
"""

from __future__ import annotations

import base64
import os

import numpy as np

from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    ReportObject,
    SceneFacts,
    SceneMetrics,
    SceneReport,
)
from server.scene_report.store import ModalScenePersistence
from server.scene_report.cloud_io import build_ply_bytes, downsample_cloud


def _store(tmp_path):
    return ModalScenePersistence({}, str(tmp_path))


def _report(room="office", summary="a room"):
    return SceneReport(
        summary=summary,
        room_type=room,
        objects=[ReportObject(name="chair")],
        facts=SceneFacts(metrics=SceneMetrics(num_submaps=2)),
    )


def _facts():
    return SceneFacts(
        metrics=SceneMetrics(num_submaps=2, num_keyframes=8),
        objects=[
            ObjectInstance(query="chair", center=[1, 0, 0], confidence=0.9,
                           evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),
        ],
        object_counts={"chair": 1},
    )


def _keyframes():
    img = base64.b64encode(b"\xff\xd8jpeg-bytes").decode("ascii")
    return [{"submap_id": 0, "frame_idx": 2, "image_b64": img}]


def test_save_list_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.save_scene("user-a", "scan1", _report(), _facts(), keyframes_b64=_keyframes())

    listed = store.list_scenes("user-a")
    assert len(listed) == 1
    item = listed[0]
    assert item["scan_id"] == "scan1"
    assert item["room_type"] == "office"
    assert item["object_count"] == 1
    assert item["thumbnail"]["blob_key"] == "0_2.jpg"

    record = store.get_scene("user-a", "scan1")
    assert record["report"]["room_type"] == "office"
    assert record["facts"]["objects"][0]["query"] == "chair"
    assert record["keyframes"][0]["blob_key"] == "0_2.jpg"

    blob = store.get_keyframe("user-a", "scan1", "0_2.jpg")
    assert blob == b"\xff\xd8jpeg-bytes"


def test_object_description_survives_roundtrip(tmp_path):
    from server.scene_report.schemas import ObjectDescription

    store = _store(tmp_path)
    facts = SceneFacts(
        metrics=SceneMetrics(num_submaps=1),
        objects=[
            ObjectInstance(
                query="laptop", center=[1, 0, 0], confidence=0.9,
                evidence=[EvidenceRef(submap_id=0, frame_idx=1)],
                description=ObjectDescription(
                    title='Apple MacBook Pro 14"', brand="Apple", model="MacBook Pro M3",
                    confidence=0.6, model_name="google/gemini-3-flash-preview",
                ),
            ),
        ],
    )
    store.save_scene("u", "s", _report(), facts)
    record = store.get_scene("u", "s")
    desc = record["facts"]["objects"][0]["description"]
    assert desc["brand"] == "Apple"
    assert desc["model"] == "MacBook Pro M3"
    assert desc["model_name"] == "google/gemini-3-flash-preview"


def test_keyframe_key_is_sanitized(tmp_path):
    store = _store(tmp_path)
    store.save_scene("user-a", "scan1", _report(), _facts(), keyframes_b64=_keyframes())
    # Path traversal / non-keyframe keys are rejected by the strict pattern.
    assert store.get_keyframe("user-a", "scan1", "../../etc/passwd") is None
    assert store.get_keyframe("user-a", "scan1", "cloud.npz") is None


def test_point_cloud_artifact_is_full_world_frame(tmp_path):
    store = _store(tmp_path)
    positions = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    store.save_scene("u", "s", _report(), _facts(), points=(positions, colors))

    record = store.get_scene("u", "s")
    assert record["point_count"] == 3            # full count, no cap/stride
    assert record["points_key"] == "cloud.npz"
    # scene_center is the mean (x=2) — used by readers for display recenter / Q&A focus.
    assert record["scene_center"][0] == 2.0

    # ...but the stored artifact stays in the WORLD frame (un-recentered, lossless).
    got = store.get_cloud("u", "s")
    assert got is not None
    world_pos, world_col = got
    np.testing.assert_allclose(world_pos, positions, atol=1e-6)
    np.testing.assert_array_equal(world_col, colors)


def test_get_cloud_reconstructs_legacy_points_bin(tmp_path):
    """Pre-migration scans stored a recentered+capped points.bin; get_cloud must read
    them back in the world frame by adding the stored scene_center."""
    store = _store(tmp_path)
    world = np.array([[1, 2, 3], [3, 2, 1], [5, 5, 5]], dtype=np.float32)
    colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    center = world.mean(axis=0).astype(np.float32)
    recentered = (world - center).astype(np.float32)

    scene_dir = store._scene_dir("u", "legacy")
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "points.bin"), "wb") as fh:
        fh.write(recentered.tobytes())
        fh.write(colors.tobytes())
    store._store[store._scene_key("u", "legacy")] = {
        "scan_id": "legacy", "user_id": "u",
        "points_key": "points.bin", "point_count": 3,
        "scene_center": [float(v) for v in center],
    }

    got = store.get_cloud("u", "legacy")
    assert got is not None
    world_pos, world_col = got
    np.testing.assert_allclose(world_pos, world, atol=1e-5)
    np.testing.assert_array_equal(world_col, colors)


def test_unlimited_retention_keeps_all_by_default(tmp_path):
    store = _store(tmp_path)  # default max_scans_per_user=0 → unlimited (keep every artifact)
    for i in range(5):
        store.save_scene("u", f"s{i}", _report(), _facts())
    assert len(store.list_scenes("u")) == 5


def test_positive_cap_prunes_oldest(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path), max_scans_per_user=2)
    for i in range(4):
        store.save_scene("u", f"s{i}", _report(), _facts())
    assert [s["scan_id"] for s in store.list_scenes("u")] == ["s3", "s2"]  # newest two kept


def test_per_user_scoping(tmp_path):
    store = _store(tmp_path)
    store.save_scene("user-a", "scan1", _report(room="office"), _facts())
    store.save_scene("user-b", "scan2", _report(room="kitchen"), _facts())

    assert [s["scan_id"] for s in store.list_scenes("user-a")] == ["scan1"]
    assert [s["scan_id"] for s in store.list_scenes("user-b")] == ["scan2"]
    assert store.get_scene("user-a", "scan2") is None  # cannot read another user's scan


def test_index_newest_first_and_dedup(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "old", _report(summary="old"), _facts())
    store.save_scene("u", "new", _report(summary="new"), _facts())
    assert [s["scan_id"] for s in store.list_scenes("u")] == ["new", "old"]

    # Re-saving an existing scan_id dedups (no duplicate row) and moves it to front.
    store.save_scene("u", "old", _report(summary="old-v2"), _facts())
    listed = store.list_scenes("u")
    assert [s["scan_id"] for s in listed] == ["old", "new"]
    assert listed[0]["summary"] == "old-v2"


def test_delete_scene(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts(), keyframes_b64=_keyframes())
    store.delete_scene("u", "s")
    assert store.list_scenes("u") == []
    assert store.get_scene("u", "s") is None
    assert store.get_keyframe("u", "s", "0_2.jpg") is None


def test_save_scene_overwrite_clears_stale_keyframes(tmp_path):
    """Re-persisting a scan_id (e.g. re-running a pilot capture) must not leave the
    PREVIOUS run's keyframe JPEGs sitting in keyframes/ alongside the new ones —
    only the record's manifest points at the current set, so anything else would be
    an orphaned-but-present blob (observed for the "finc" pilot scan)."""
    store = _store(tmp_path)
    kf_dir = os.path.join(str(tmp_path), "user-a", "finc", "keyframes")

    first = [
        {"submap_id": 0, "frame_idx": 1,
         "image_b64": base64.b64encode(b"\xff\xd8first-a").decode("ascii")},
        {"submap_id": 0, "frame_idx": 2,
         "image_b64": base64.b64encode(b"\xff\xd8first-b").decode("ascii")},
    ]
    store.save_scene("user-a", "finc", _report(), _facts(), keyframes_b64=first)
    assert sorted(os.listdir(kf_dir)) == ["0_1.jpg", "0_2.jpg"]

    # A blob that doesn't match the keyframe naming must survive the cleanup — the
    # fix is scoped strictly to _KEYFRAME_BLOB matches, never other files.
    unrelated_path = os.path.join(kf_dir, "notes.txt")
    with open(unrelated_path, "w") as fh:
        fh.write("not a keyframe blob")

    # Re-run the same scan_id with a different, smaller keyframe set (a re-capture
    # producing fewer/renamed submaps).
    second = [
        {"submap_id": 1, "frame_idx": 5,
         "image_b64": base64.b64encode(b"\xff\xd8second").decode("ascii")},
    ]
    store.save_scene("user-a", "finc", _report(), _facts(), keyframes_b64=second)

    # Only the CURRENT run's keyframe remains on disk; the unrelated file is untouched.
    assert sorted(os.listdir(kf_dir)) == ["1_5.jpg", "notes.txt"]
    assert os.path.exists(unrelated_path)

    record = store.get_scene("user-a", "finc")
    assert [kf["blob_key"] for kf in record["keyframes"]] == ["1_5.jpg"]
    assert store.get_keyframe("user-a", "finc", "0_1.jpg") is None
    assert store.get_keyframe("user-a", "finc", "0_2.jpg") is None
    assert store.get_keyframe("user-a", "finc", "1_5.jpg") == b"\xff\xd8second"


# ── splat.ply artifact (product-workflow API: anchor/clamp read this) ──


def test_save_scene_with_splat_bytes_roundtrips(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts(), splat_bytes=b"PLYDATA")

    assert store.get_scene("u", "s")["splat_key"] == "splat.ply"
    assert store.list_scenes("u")[0]["has_splat"] is True
    assert store.get_splat("u", "s") == b"PLYDATA"

    path = store.get_splat_path("u", "s")
    assert path is not None and path.endswith("splat.ply")
    assert open(path, "rb").read() == b"PLYDATA"


def test_scene_without_splat_has_no_splat(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts())
    assert store.get_scene("u", "s")["splat_key"] is None
    assert store.list_scenes("u")[0]["has_splat"] is False
    assert store.get_splat("u", "s") is None
    assert store.get_splat_path("u", "s") is None


# ── derived (non-destructive) artifacts (product-workflow API: anchor/clamp write these) ──


def test_save_derived_artifact_roundtrips_and_leaves_originals_untouched(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts(), splat_bytes=b"ORIGINAL")

    key = store.save_derived_artifact("u", "s", "clamp/t1/splat.ply", b"CLAMPED")
    assert key == "derived/clamp/t1/splat.ply"
    assert store.get_derived_artifact("u", "s", key) == b"CLAMPED"
    path = store.get_derived_artifact_path("u", "s", key)
    assert path is not None and os.path.isfile(path)

    # the original splat is completely unaffected by writing a derived artifact
    assert store.get_splat("u", "s") == b"ORIGINAL"


def test_get_derived_artifact_missing_key_returns_none(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts())
    assert store.get_derived_artifact_path("u", "s", "derived/nope/x.ply") is None
    assert store.get_derived_artifact("u", "s", "derived/nope/x.ply") is None
    # not even prefixed with "derived/" -> rejected outright, no accidental scene-dir read
    assert store.get_derived_artifact_path("u", "s", "cloud.npz") is None


def test_save_derived_artifact_sanitizes_path_traversal(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "s", _report(), _facts())

    key = store.save_derived_artifact("u", "s", "../../../etc/passwd", b"pwned")
    # every path segment is sanitized independently -- the result can never escape
    # this scan's derived/ dir, regardless of how hostile relative_key is.
    resolved = os.path.realpath(store.get_derived_artifact_path("u", "s", key))
    scene_root = os.path.realpath(os.path.join(str(tmp_path), "u", "s"))
    assert resolved.startswith(scene_root + os.sep)
    assert store.get_derived_artifact("u", "s", key) == b"pwned"


# ── cloud_io helpers (dependency-light, broker-side) ──

def test_build_ply_bytes_layout_and_roundtrip():
    positions = np.array([[0, 0, 0], [1.5, -2.0, 3.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 128, 64]], dtype=np.uint8)
    blob = build_ply_bytes(positions, colors)

    marker = b"end_header\n"
    head_end = blob.index(marker) + len(marker)
    header = blob[:head_end].decode("ascii")
    assert "format binary_little_endian 1.0" in header
    assert "element vertex 2" in header

    body = blob[head_end:]
    assert len(body) == 2 * 15  # 3*float32 + 3*uint8, no padding
    vtx = np.frombuffer(
        body,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    np.testing.assert_allclose(
        np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1), positions, atol=1e-6)
    np.testing.assert_array_equal(
        np.stack([vtx["red"], vtx["green"], vtx["blue"]], axis=1), colors)


def test_downsample_cloud_caps_and_passthrough():
    positions = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.arange(30, dtype=np.uint8).reshape(10, 3)

    dp, dc = downsample_cloud(positions, colors, 4)
    assert dp.shape == (4, 3) and dc.shape == (4, 3)
    # Subset of the originals (deterministic), positions/colors stay paired.
    for row_p, row_c in zip(dp, dc):
        idx = int(row_p[0]) // 3
        np.testing.assert_array_equal(row_p, positions[idx])
        np.testing.assert_array_equal(row_c, colors[idx])

    # max_points <= 0 or >= N → unchanged.
    np.testing.assert_array_equal(downsample_cloud(positions, colors, 0)[0], positions)
    np.testing.assert_array_equal(downsample_cloud(positions, colors, 999)[0], positions)


# -- splat.ply artifact + client/project tags (embed-delivery additions) ----------

def test_splat_artifact_roundtrip(tmp_path):
    store = _store(tmp_path)
    splat = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
    store.save_scene("user-a", "scanS", _report(), _facts(), splat_bytes=splat)

    # round-trips byte-for-byte
    assert store.get_splat("user-a", "scanS") == splat
    # surfaced on the record + list summary
    assert store.get_scene("user-a", "scanS")["splat_key"] == "splat.ply"
    assert store.list_scenes("user-a")[0]["has_splat"] is True
    # per-user scoped — another user can't read it
    assert store.get_splat("user-b", "scanS") is None

    # get_splat_path streams the same artifact from disk (used by the broker to avoid loading
    # a hundreds-of-MB splat fully into RAM); points at a real file with the same bytes.
    p = store.get_splat_path("user-a", "scanS")
    assert p is not None and os.path.isfile(p)
    with open(p, "rb") as fh:
        assert fh.read() == splat
    # same per-user scoping as get_splat
    assert store.get_splat_path("user-b", "scanS") is None


def test_splat_absent_is_none_and_backward_compatible(tmp_path):
    store = _store(tmp_path)
    # a scan saved WITHOUT a splat (the common/legacy case) has no artifact
    store.save_scene("user-a", "scanP", _report(), _facts(), keyframes_b64=_keyframes())
    assert store.get_splat("user-a", "scanP") is None
    rec = store.get_scene("user-a", "scanP")
    assert rec["splat_key"] is None
    assert store.list_scenes("user-a")[0]["has_splat"] is False
    # tags default to None when not provided
    assert rec["client"] is None and rec["project"] is None


def test_client_project_tags_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.save_scene(
        "user-a", "scanC", _report(), _facts(),
        client="efficura", project="chorus",
    )
    rec = store.get_scene("user-a", "scanC")
    assert rec["client"] == "efficura" and rec["project"] == "chorus"
    summary = store.list_scenes("user-a")[0]
    assert summary["client"] == "efficura" and summary["project"] == "chorus"
