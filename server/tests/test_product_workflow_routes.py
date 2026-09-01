"""Product-workflow API tests: GPU-free dataset export, metric anchor, auto-clamp.

See docs/dataset-export.md (export), server/scene_report/splat_io.py (anchor/clamp math
-- self-contained ports of unmerged `core` (reality-opened/core) branches, read there as a
SPEC and never imported at runtime: `vggt_slam.metric_absolute` /
`vggt_slam.splat_export`). Exercises server.app's underlying helper functions through the
conftest `load_app_module` stub (no Flask/GPU/network) -- the same pattern
tests/test_scene_qa.py uses. Deliberately builds trajectory/cloud/splat fixtures by hand
(no FakeSolver / vggt_slam.slam_utils.decompose_camera), so -- unlike
tests/test_export_record.py -- these tests do NOT skip when openreality-core isn't
installed; they always run.
"""

from __future__ import annotations

import io
import os
import types
import zipfile

import numpy as np
import pytest

from conftest import load_app_module

from server.export.record import EXPORT_SOURCE_ORIGINAL, load_record_from_store
from server.scene_report.cloud_io import build_ply_bytes, read_ply_cloud
from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport
from server.scene_report.store import ModalScenePersistence
from server.scene_report.splat_io import (
    DEFAULT_SCALE_CLAMP_PERCENTILE,
    SCALE_FIELDS,
    clamp_scales_percentile,
    clamp_splat_fields,
    read_splat_ply,
    scale_splat_fields,
    serialize_splat_ply,
)


# ---------------------------------------------------------------------------
# fixtures -- hand-built, no live/fake Solver, no vggt_slam
# ---------------------------------------------------------------------------


def _synthetic_trajectory(n=4):
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for i in range(n):
        poses[i, :3, 3] = [float(i), 0.5 * i, 0.0]
    intrinsics = np.tile(np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32), (n, 1))
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "source_frame_id": np.arange(n, dtype=np.float32),
    }


def _synthetic_cloud(n=50, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n, 3)).astype(np.float32) * 2.0
    colors = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return positions, colors


def _synthetic_record(scan_id="scan-synthetic-001", n_traj=4, n_points=50):
    return {
        "scan_id": scan_id,
        "trajectory": _synthetic_trajectory(n_traj),
        "facts": SceneFacts(metrics=SceneMetrics(num_keyframes=n_traj)).model_dump(),
        "report": SceneReport(summary="a synthetic dev scan", room_type="lab").model_dump(),
        "cloud": _synthetic_cloud(n_points),
        "keyframes": [],  # Stage-4 record path: video is skipped, matches load_record_from_store
    }


def _synthetic_splat_fields(n=30, seed=0, spike=False):
    """A hand-built splat field dict (the shape read_splat_ply/serialize_splat_ply move) --
    independent of any live SLAM/gsplat pipeline. `spike=True` pushes a few gaussians'
    linear scale to a large outlier value (the EXP-7 'spike tail' this module's clamp is
    for)."""
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    scale_linear = np.full((n, 3), 0.01, dtype=np.float32)
    if spike:
        scale_linear[0, :] = 50.0  # a single blown-out gaussian (EXP-7: top-~1% tail)
    scale_log = np.log(scale_linear).astype(np.float32)
    fields = {
        "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
        "f_dc_0": colors[:, 0], "f_dc_1": colors[:, 1], "f_dc_2": colors[:, 2],
        "opacity": np.full(n, 2.0, np.float32),
        "scale_0": scale_log[:, 0], "scale_1": scale_log[:, 1], "scale_2": scale_log[:, 2],
        "rot_0": np.ones(n, np.float32), "rot_1": np.zeros(n, np.float32),
        "rot_2": np.zeros(n, np.float32), "rot_3": np.zeros(n, np.float32),
    }
    return fields


def _store_with_scene(tmp_path, user="user-a", scan="scan1", with_splat=True, with_trajectory=True):
    store = ModalScenePersistence({}, str(tmp_path))
    positions, colors = _synthetic_cloud(60, seed=3)
    splat_bytes = serialize_splat_ply(_synthetic_splat_fields(24, seed=4, spike=True)) if with_splat else None
    store.save_scene(
        user,
        scan,
        SceneReport(summary="s", room_type="office"),
        SceneFacts(),
        points=(positions, colors),
        trajectory=_synthetic_trajectory(5) if with_trajectory else None,
        splat_bytes=splat_bytes,
    )
    return store


# ---------------------------------------------------------------------------
# splat_io: PLY read/write round trip + clamp math (exact port of core's function)
# ---------------------------------------------------------------------------


def test_read_splat_ply_parses_a_hand_built_file_independent_of_the_writer(tmp_path):
    """Build the binary PLY bytes by hand (not via serialize_splat_ply) so the read path
    is verified against an independent construction of the exact 3DGS schema."""
    n = 3
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(n)
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"
    dtype = np.dtype([(name, "<f4") for name in names])
    verts = np.zeros(n, dtype=dtype)
    verts["x"] = [1.0, 2.0, 3.0]
    verts["y"] = [4.0, 5.0, 6.0]
    verts["z"] = [7.0, 8.0, 9.0]
    verts["scale_0"] = [-1.0, -2.0, -3.0]
    verts["scale_1"] = [-1.0, -2.0, -3.0]
    verts["scale_2"] = [-1.0, -2.0, -3.0]
    path = tmp_path / "manual.ply"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes(order="C"))

    fields = read_splat_ply(str(path))
    assert list(fields.keys()) == names
    np.testing.assert_allclose(fields["x"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(fields["scale_0"], [-1.0, -2.0, -3.0])


def test_write_then_read_splat_ply_roundtrip(tmp_path):
    fields = _synthetic_splat_fields(10, seed=1)
    data = serialize_splat_ply(fields)
    path = tmp_path / "rt.ply"
    with open(path, "wb") as f:
        f.write(data)
    back = read_splat_ply(str(path))
    assert list(back.keys()) == list(fields.keys())  # property order preserved
    for name in fields:
        np.testing.assert_allclose(back[name], fields[name], atol=1e-6)


def test_clamp_scales_percentile_matches_manual_numpy_percentile():
    rng = np.random.default_rng(0)
    scales = rng.uniform(0.001, 1.0, size=(200, 3)).astype(np.float32)
    scales[0, :] = 500.0  # obvious outlier

    clamped, info = clamp_scales_percentile(scales, 99.0)

    expected_clamp = np.percentile(scales[np.isfinite(scales)], 99.0)
    assert info["clamp_value"] == pytest.approx(float(expected_clamp), rel=1e-5)
    assert clamped.max() <= expected_clamp + 1e-4
    assert info["num_clamped"] >= 1  # the outlier row got pulled down
    # every non-outlier row is untouched
    untouched = ~(scales > expected_clamp).any(axis=1)
    np.testing.assert_allclose(clamped[untouched], scales[untouched])


def test_clamp_scales_percentile_noop_at_100():
    scales = np.array([[1.0, 2.0, 3.0], [400.0, 1.0, 1.0]], dtype=np.float32)
    clamped, info = clamp_scales_percentile(scales, 100.0)
    np.testing.assert_allclose(clamped, scales)
    assert info["num_clamped"] == 0


def test_clamp_splat_fields_despikes_and_preserves_other_properties():
    # n=100 keeps the single spiked gaussian (3 of 300 scale values) cleanly above the
    # p99 threshold -- EXP-7's "top-1% tail is entirely the spike" assumption.
    fields = _synthetic_splat_fields(100, seed=2, spike=True)
    clamped, info = clamp_splat_fields(fields, DEFAULT_SCALE_CLAMP_PERCENTILE)

    assert info["clamped_gaussian_count"] >= 1
    assert info["max_scale_after"] < info["max_scale_before"]  # despiked
    assert info["gaussian_count"] == 100
    assert info["scale_clamp_percentile"] == 99.0

    # every non-scale property is byte-identical -- clamp touches ONLY scale_0..2
    for name in fields:
        if name in SCALE_FIELDS:
            continue
        np.testing.assert_array_equal(clamped[name], fields[name])
    # property order preserved (matters for write_splat_ply/serialize_splat_ply)
    assert list(clamped.keys()) == list(fields.keys())

    # clamp changes no COUNT -- same N in, same N out (contract: no removed_point_count)
    for name in fields:
        assert clamped[name].shape == fields[name].shape

    # input is never mutated
    assert not np.array_equal(fields["scale_0"], clamped["scale_0"]) or info["clamped_gaussian_count"] == 0


def test_clamp_splat_fields_missing_scale_field_raises():
    fields = {"x": np.zeros(3, np.float32)}
    with pytest.raises(ValueError):
        clamp_splat_fields(fields, 99.0)


def test_scale_splat_fields_scales_positions_and_log_scales_additively():
    fields = _synthetic_splat_fields(5, seed=5)
    k = 2.5
    scaled = scale_splat_fields(fields, k)

    np.testing.assert_allclose(scaled["x"], fields["x"] * k, rtol=1e-5)
    np.testing.assert_allclose(scaled["y"], fields["y"] * k, rtol=1e-5)
    np.testing.assert_allclose(scaled["z"], fields["z"] * k, rtol=1e-5)
    # log(s*k) == log(s) + log(k)
    np.testing.assert_allclose(scaled["scale_0"], fields["scale_0"] + np.log(k), rtol=1e-5)
    # linear scale actually multiplies by k
    np.testing.assert_allclose(np.exp(scaled["scale_0"]), np.exp(fields["scale_0"]) * k, rtol=1e-4)
    # untouched properties unchanged
    np.testing.assert_array_equal(scaled["opacity"], fields["opacity"])
    # input never mutated
    assert not np.shares_memory(scaled["x"], fields["x"])


def test_scale_splat_fields_rejects_non_positive_factor():
    fields = _synthetic_splat_fields(3, seed=6)
    with pytest.raises(ValueError):
        scale_splat_fields(fields, 0.0)
    with pytest.raises(ValueError):
        scale_splat_fields(fields, -1.0)


# ---------------------------------------------------------------------------
# server.app helpers: export
# ---------------------------------------------------------------------------


def test_build_export_zip_openreality_tree(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    record = _synthetic_record()

    data = app_mod._build_export_zip(record, "openreality")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("meta/info.json") for n in names)
        assert any(n.endswith("meta/episode.json") for n in names)
        assert any(n.endswith("data/trajectory.parquet") for n in names)
        assert any(n.endswith("clouds/cloud.npz") for n in names)
        # the zip's top-level entry is the scan_id (nice UX for `unzip`)
        assert all(n.startswith(record["scan_id"]) for n in names)


def test_build_export_zip_groot_lerobot_v2_tree(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    record = _synthetic_record(scan_id="scan-groot-002")

    data = app_mod._build_export_zip(record, "groot_lerobot_v2")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("meta/modality.json") for n in names)
        assert any(n.endswith("meta/info.json") for n in names)
        assert any(n.endswith("data/chunk-000/episode_000000.parquet") for n in names)


def test_export_formats_tuple_excludes_isaac_usd():
    from server.app import _EXPORT_FORMATS

    assert "isaac_usd" not in _EXPORT_FORMATS
    assert set(_EXPORT_FORMATS) == {"openreality", "groot_lerobot_v2"}


# ---------------------------------------------------------------------------
# server.app helpers: metric anchor
# ---------------------------------------------------------------------------


def test_scale_factor_from_known_distance_matches_division(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._scale_factor_from_known_distance(2.0, 4.0) == pytest.approx(0.5)


def test_scale_factor_from_known_distance_rejects_degenerate_measured(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    with pytest.raises(ValueError):
        app_mod._scale_factor_from_known_distance(2.0, 1e-9)


def test_scale_factor_from_known_distance_rejects_non_positive_target(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    with pytest.raises(ValueError):
        app_mod._scale_factor_from_known_distance(-1.0, 1.0)


def test_apply_metric_anchor_scales_cloud_and_writes_derived_key_only(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)

    original_cloud_before = store.get_cloud("user-a", "scan1")
    original_bytes = open(os.path.join(str(tmp_path), "user-a", "scan1", "cloud.npz"), "rb").read()

    point_a, point_b = [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]  # measured = 2.0 SLAM-units
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", point_a, point_b, 1.0)

    assert result["measured_distance"] == pytest.approx(2.0)
    assert result["scale_factor"] == pytest.approx(0.5)  # 1.0m / 2.0 units
    assert result["distance_m"] == 1.0
    assert result["calibrated_cloud_key"].startswith("derived/anchor/")
    assert result["gauge_span_after_m"] == pytest.approx(1.0, rel=1e-4)
    assert result["cloud_extent_after_m"] == pytest.approx(result["cloud_extent_before"] * 0.5, rel=1e-4)

    # ORIGINAL cloud.npz is byte-for-byte untouched
    with_new_bytes = open(os.path.join(str(tmp_path), "user-a", "scan1", "cloud.npz"), "rb").read()
    assert with_new_bytes == original_bytes
    original_cloud_after = store.get_cloud("user-a", "scan1")
    np.testing.assert_array_equal(original_cloud_before[0], original_cloud_after[0])

    # the DERIVED cloud is retrievable and actually scaled by 0.5
    derived_path = store.get_derived_artifact_path("user-a", "scan1", result["calibrated_cloud_key"])
    assert derived_path is not None and os.path.isfile(derived_path)
    from server.scene_report.cloud_io import build_ply_bytes
    expected = build_ply_bytes(original_cloud_before[0] * 0.5, original_cloud_before[1])
    assert open(derived_path, "rb").read() == expected

    # trajectory was present -> a calibrated derived trajectory was also written
    assert result["calibrated_trajectory_key"] is not None
    traj_path = store.get_derived_artifact_path("user-a", "scan1", result["calibrated_trajectory_key"])
    with np.load(traj_path) as npz:
        # translation scaled by 0.5 (poses[1] == [1.0, 0.5, 0.0] before), rotation untouched
        np.testing.assert_allclose(npz["poses"][1, :3, 3], np.array([0.5, 0.25, 0.0]), atol=1e-4)

    # no splat on this scene -> no derived splat key
    assert result["calibrated_splat_key"] is None


def test_apply_metric_anchor_scales_splat_when_present(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=False)
    original_splat_bytes = open(store.get_splat_path("user-a", "scan1"), "rb").read()

    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [1, 0, 0], 4.0)
    assert result["calibrated_splat_key"] is not None

    # original splat.ply untouched
    assert open(store.get_splat_path("user-a", "scan1"), "rb").read() == original_splat_bytes

    derived_path = store.get_derived_artifact_path("user-a", "scan1", result["calibrated_splat_key"])
    scaled_fields = read_splat_ply(derived_path)
    original_fields = read_splat_ply(store.get_splat_path("user-a", "scan1"))
    k = result["scale_factor"]
    np.testing.assert_allclose(scaled_fields["x"], original_fields["x"] * k, rtol=1e-4)
    np.testing.assert_allclose(
        np.exp(scaled_fields["scale_0"]), np.exp(original_fields["scale_0"]) * k, rtol=1e-3
    )


def test_apply_metric_anchor_raises_no_geometry_when_scan_has_no_cloud(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene("u", "s", SceneReport(), SceneFacts())  # no points=

    with pytest.raises(KeyError):
        app_mod._apply_metric_anchor(store, "u", "s", [0, 0, 0], [1, 0, 0], 1.0)


def test_apply_metric_anchor_rejects_malformed_point(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path)
    with pytest.raises(ValueError):
        app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0], [1, 0, 0], 1.0)  # only 2 coords


# ---------------------------------------------------------------------------
# async anchor split (materialize:'job'): instant scale + pending pointer
# ---------------------------------------------------------------------------


def test_compute_anchor_scale_matches_sync_apply(monkeypatch, tmp_path):
    """The instant half must produce byte-identical numbers to the one-shot apply."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=False)

    measured, factor = app_mod._anchor_impl.compute_anchor_scale([0, 0, 0], [2, 0, 0], 1.0)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    assert measured == pytest.approx(result["measured_distance"])
    assert factor == pytest.approx(result["scale_factor"])


def test_compute_anchor_scale_validates_before_any_work(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    with pytest.raises(ValueError):
        app_mod._anchor_impl.compute_anchor_scale([0, 0], [1, 0, 0], 1.0)  # malformed point
    with pytest.raises(ValueError):
        app_mod._anchor_impl.compute_anchor_scale([0, 0, 0], [0, 0, 0], 1.0)  # degenerate pair
    with pytest.raises(ValueError):
        app_mod._anchor_impl.compute_anchor_scale([0, 0, 0], [1, 0, 0], -2.0)  # bad distance


def test_pending_anchor_pointer_is_metric_metadata_with_no_keys(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    ptr = app_mod._anchor_impl.pending_anchor_pointer(0.5, "2026-08-04T00:00:00+00:00", "job123")
    assert ptr["kind"] == "anchor"
    assert ptr["pending"] is True
    assert ptr["scale_factor"] == pytest.approx(0.5)
    assert ptr["applied_at"] == "2026-08-04T00:00:00+00:00"
    assert ptr["materializing_job_id"] == "job123"
    # every artifact key stays None until the job upgrades the pointer
    for key in ("cloud_key", "trajectory_key", "splat_key", "source_key"):
        assert ptr[key] is None


def test_pending_anchor_pointer_keeps_isaac_gate_closed(monkeypatch):
    """A pending pointer means metres-by-factor only — no calibrated geometry exists yet,
    so the Isaac export must keep refusing (it requires a truthy source_key)."""
    app_mod = load_app_module(monkeypatch)
    ptr = app_mod._anchor_impl.pending_anchor_pointer(0.5, "2026-08-04T00:00:00+00:00")
    record = {"derived_latest": ptr}
    assert app_mod._resolve_isaac_scale(record, None, None) is None


def test_materialize_writes_same_artifacts_the_sync_apply_wrote(monkeypatch, tmp_path):
    """The job's half, given the precomputed factor, writes an equivalent calibrated cloud."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    original_cloud = store.get_cloud("user-a", "scan1")

    artifacts = app_mod._anchor_impl.materialize_anchor_artifacts(
        store, "user-a", "scan1", 0.5
    )
    assert artifacts["calibrated_cloud_key"].startswith("derived/anchor/")
    assert artifacts["calibrated_trajectory_key"] is not None
    assert artifacts["calibrated_splat_key"] is None
    assert artifacts["cloud_extent_after_m"] == pytest.approx(
        artifacts["cloud_extent_before"] * 0.5, rel=1e-4
    )
    derived_path = store.get_derived_artifact_path(
        "user-a", "scan1", artifacts["calibrated_cloud_key"]
    )
    from server.scene_report.cloud_io import build_ply_bytes

    assert open(derived_path, "rb").read() == build_ply_bytes(
        original_cloud[0] * 0.5, original_cloud[1]
    )


# ---------------------------------------------------------------------------
# non-destructiveness across BOTH product-workflow mutators, on one shared scene
# ---------------------------------------------------------------------------


def test_anchor_and_clamp_together_never_touch_original_assets(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    cloud_path = os.path.join(str(tmp_path), "user-a", "scan1", "cloud.npz")
    splat_path = store.get_splat_path("user-a", "scan1")
    cloud_before, splat_before = open(cloud_path, "rb").read(), open(splat_path, "rb").read()

    app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [1, 0, 0], 3.0)
    clamped_fields, info = clamp_splat_fields(read_splat_ply(store.get_splat_path("user-a", "scan1")))
    store.save_derived_artifact("user-a", "scan1", "clamp/t/splat.ply", serialize_splat_ply(clamped_fields))

    assert open(cloud_path, "rb").read() == cloud_before
    assert open(splat_path, "rb").read() == splat_before


# ---------------------------------------------------------------------------
# P1: consume derived artifacts on EXPORT (the bug: anchor/clamp wrote derived
# keys nothing ever read back). cloud_io PLY reader + selector/pointer precedence.
# ---------------------------------------------------------------------------


def test_read_ply_cloud_roundtrips_build_ply_bytes():
    positions = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [200, 100, 0]], dtype=np.uint8)
    back_pos, back_col = read_ply_cloud(build_ply_bytes(positions, colors))
    np.testing.assert_array_equal(back_pos, positions)
    np.testing.assert_array_equal(back_col, colors)


def test_read_ply_cloud_handles_empty_and_rejects_garbage():
    pos, col = read_ply_cloud(build_ply_bytes(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)))
    assert pos.shape == (0, 3) and col.shape == (0, 3)
    with pytest.raises(ValueError):
        read_ply_cloud(b"not a ply at all")


def _fake_request(monkeypatch, app_mod, *, args=None, claims=None, json_body=None):
    """Minimal fake ``request`` for driving a route body directly (mirrors test_project_share)."""
    req = types.SimpleNamespace(
        args=dict(args or {}),
        headers={},
        cookies={},
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: dict(claims or {})},
        get_json=lambda *a, **k: dict(json_body or {}),
    )
    monkeypatch.setattr(app_mod, "request", req)
    return req


def test_plain_export_unchanged_when_no_derived_exists(monkeypatch, tmp_path):
    """(a) No selector, no persisted pointer -> ORIGINAL geometry, and NO splat in the tree
    (the record path never exported a splat before this change; keep it byte-identical)."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    orig_cloud = store.get_cloud("user-a", "scan1")

    normalized = load_record_from_store(store, "user-a", "scan1")  # no source
    assert "splat" not in normalized  # original splat is intentionally NOT added to a plain export
    np.testing.assert_array_equal(normalized["cloud"][0], orig_cloud[0])

    data = app_mod._build_export_zip(normalized, "openreality")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("clouds/cloud.npz") for n in names)
        assert not any(n.endswith("clouds/splat.ply") for n in names)  # unchanged: no splat


def test_anchor_then_export_yields_calibrated_geometry(monkeypatch, tmp_path):
    """(b) Selecting the anchor's derived cloud key -> the export's cloud.npz + splat.ply carry
    the CALIBRATED geometry (positions/splat scaled), and the trajectory is the calibrated one."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    orig_cloud = store.get_cloud("user-a", "scan1")

    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    k = result["scale_factor"]  # 0.5
    assert k == pytest.approx(0.5)

    normalized = load_record_from_store(store, "user-a", "scan1", source=result["calibrated_cloud_key"])
    # cloud read back from the derived PLY == original * scale
    np.testing.assert_allclose(normalized["cloud"][0], orig_cloud[0] * k, rtol=1e-6)
    # a derived splat path is carried onto the record
    assert isinstance(normalized["splat"], str) and normalized["splat"].endswith("splat.ply")

    data = app_mod._build_export_zip(normalized, "openreality")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open(next(n for n in zf.namelist() if n.endswith("clouds/cloud.npz"))) as fh:
            with np.load(io.BytesIO(fh.read())) as npz:
                np.testing.assert_allclose(npz["positions"], orig_cloud[0] * k, rtol=1e-6)
        splat_name = next(n for n in zf.namelist() if n.endswith("clouds/splat.ply"))
        exported_splat = zf.read(splat_name)
    # the exported splat is byte-identical to the derived (calibrated) splat, not the original
    derived_splat_path = store.get_derived_artifact_path("user-a", "scan1", result["calibrated_splat_key"])
    assert exported_splat == open(derived_splat_path, "rb").read()
    assert exported_splat != open(store.get_splat_path("user-a", "scan1"), "rb").read()


def test_clamp_then_export_yields_clamped_splat(monkeypatch, tmp_path):
    """(c) Selecting a clamp derived splat key -> the export's splat.ply is the CLAMPED splat,
    while cloud/trajectory stay original (a clamp group carries only a splat)."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    orig_cloud = store.get_cloud("user-a", "scan1")
    clamped_fields, _info = clamp_splat_fields(read_splat_ply(store.get_splat_path("user-a", "scan1")))
    clamped_bytes = serialize_splat_ply(clamped_fields)
    clamp_key = store.save_derived_artifact("user-a", "scan1", "clamp/s1/splat.ply", clamped_bytes)

    normalized = load_record_from_store(store, "user-a", "scan1", source=clamp_key)
    np.testing.assert_array_equal(normalized["cloud"][0], orig_cloud[0])  # cloud unchanged
    assert normalized["splat"].endswith("splat.ply")

    data = app_mod._build_export_zip(normalized, "openreality")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        splat_name = next(n for n in zf.namelist() if n.endswith("clouds/splat.ply"))
        assert zf.read(splat_name) == clamped_bytes
        # cloud still the original
        with zf.open(next(n for n in zf.namelist() if n.endswith("clouds/cloud.npz"))) as fh:
            with np.load(io.BytesIO(fh.read())) as npz:
                np.testing.assert_array_equal(npz["positions"], orig_cloud[0])


def test_derived_source_groot_export_still_builds(monkeypatch, tmp_path):
    """A derived-source export in the GR00T-LeRobot format must still transcode — the extra
    clouds/splat.ply the derived path adds to the tree is ignored by the converter."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    normalized = load_record_from_store(store, "user-a", "scan1", source=result["calibrated_cloud_key"])
    data = app_mod._build_export_zip(normalized, "groot_lerobot_v2")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("meta/modality.json") for n in names)
        assert any(n.endswith("data/chunk-000/episode_000000.parquet") for n in names)


def test_source_original_forces_original_even_with_pointer(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    orig_cloud = store.get_cloud("user-a", "scan1")
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    store.set_derived_pointer(
        "user-a", "scan1", {"kind": "anchor", "source_key": result["calibrated_cloud_key"]}
    )
    # default (pointer) -> calibrated
    np.testing.assert_allclose(
        load_record_from_store(store, "user-a", "scan1")["cloud"][0], orig_cloud[0] * 0.5, rtol=1e-6
    )
    # explicit ?source=original -> back to the original
    np.testing.assert_array_equal(
        load_record_from_store(store, "user-a", "scan1", source=EXPORT_SOURCE_ORIGINAL)["cloud"][0],
        orig_cloud[0],
    )


def test_stale_pointer_degrades_to_original(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    orig_cloud = store.get_cloud("user-a", "scan1")
    store.set_derived_pointer(
        "user-a", "scan1", {"kind": "anchor", "source_key": "derived/anchor/gone/cloud.ply"}
    )
    # a pointer whose files no longer exist must not error — fall back to the original geometry
    np.testing.assert_array_equal(
        load_record_from_store(store, "user-a", "scan1")["cloud"][0], orig_cloud[0]
    )


def test_set_derived_pointer_is_metadata_only(tmp_path):
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    cloud_before = open(os.path.join(str(tmp_path), "user-a", "scan1", "cloud.npz"), "rb").read()
    assert store.set_derived_pointer("user-a", "scan1", {"kind": "clamp", "source_key": "derived/clamp/x/splat.ply"})
    rec = store.get_scene("user-a", "scan1")
    assert rec["derived_latest"]["source_key"] == "derived/clamp/x/splat.ply"
    # original blob untouched; clearing works; unknown scan returns False
    assert open(os.path.join(str(tmp_path), "user-a", "scan1", "cloud.npz"), "rb").read() == cloud_before
    assert store.set_derived_pointer("user-a", "scan1", None)
    assert "derived_latest" not in store.get_scene("user-a", "scan1")
    assert store.set_derived_pointer("user-a", "nope", {"kind": "clamp", "source_key": "k"}) is False


# ---------------------------------------------------------------------------
# route-level: export ?source= validation + the new derived read route
# ---------------------------------------------------------------------------


def test_export_route_rejects_unknown_derived_key(monkeypatch, tmp_path):
    """(d) A bogus/foreign derived key is a hard 404 (unknown_derived_key), never a silent
    fall-through to the original and never a path traversal."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(
        monkeypatch, app_mod,
        args={"source": "derived/anchor/does-not-exist/cloud.ply"},
        claims={"sub": "user-a"},
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 404
    assert out[0]["args"][0]["error"] == "unknown_derived_key"


def test_export_route_rejects_traversal_source(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(
        monkeypatch, app_mod,
        args={"source": "derived/../../etc/passwd"},
        claims={"sub": "user-a"},
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 404


def test_export_route_streams_zip_from_disk_and_cleans_up(monkeypatch, tmp_path):
    """(demo reconciliation f / BUILD-PLAN R8) The zip route hands ``send_file``
    a file PATH — the archive streams from disk, never through an in-RAM
    BytesIO (an anchored-source canonical export is ~1.4 GB on a 4 GB broker) —
    and its ``after_this_request`` callback removes the temp file. Small-scene
    behavior is unchanged: the response is the same valid export-tree zip."""
    import sys as _sys

    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(monkeypatch, app_mod, claims={"sub": "user-a"})

    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, dict) and "file" in out, out  # FakeFlask.send_file dict
    path = out["file"][0]
    assert isinstance(path, str) and path.endswith(".zip") and os.path.isfile(path)
    assert out["kwargs"]["as_attachment"] is True
    assert out["kwargs"]["download_name"] == "scan-scan1-openreality.zip"
    assert out["kwargs"]["mimetype"] == "application/zip"
    with zipfile.ZipFile(path) as zf:  # unchanged small-scene behavior: real tree
        names = set(zf.namelist())
    assert any(n.endswith("meta/info.json") for n in names)
    assert any(n.endswith("clouds/cloud.npz") for n in names)

    # the registered after-request callback removes the on-disk temp zip
    callbacks = _sys.modules["flask"].after_this_request_callbacks
    assert callbacks, "export route registered no after_this_request cleanup"
    sentinel = object()
    assert callbacks[-1](sentinel) is sentinel  # passes the response through
    assert not os.path.exists(path)


def test_anchor_route_persists_pointer_and_default_export_uses_it(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    orig_cloud = store.get_cloud("user-a", "scan1")
    _fake_request(
        monkeypatch, app_mod, claims={"sub": "user-a"},
        json_body={"point_a": [0, 0, 0], "point_b": [2, 0, 0], "distance_m": 1.0},
    )
    out = app_mod.anchor_scene_route("scan1")
    result = out["args"][0]  # FakeFlask.jsonify wraps positional args
    assert result["calibrated_cloud_key"].startswith("derived/anchor/")

    rec = store.get_scene("user-a", "scan1")
    assert rec["derived_latest"]["kind"] == "anchor"
    assert rec["derived_latest"]["source_key"] == result["calibrated_cloud_key"]

    # default (no ?source=) export now reads the calibrated group via the persisted pointer
    normalized = load_record_from_store(store, "user-a", "scan1")
    np.testing.assert_allclose(normalized["cloud"][0], orig_cloud[0] * 0.5, rtol=1e-6)


def test_derived_read_route_streams_owned_artifact(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    # the route takes the key WITHOUT the "derived/" prefix
    rel = result["calibrated_cloud_key"][len("derived/"):]

    _fake_request(monkeypatch, app_mod, claims={"sub": "user-a"})
    out = app_mod.get_scene_derived_route("scan1", rel)
    # FakeFlask.send_file returns {"file": args, "kwargs": kwargs}
    path = out["file"][0]
    assert path.endswith("cloud.ply") and os.path.isfile(path)
    expected = store.get_derived_artifact_path("user-a", "scan1", result["calibrated_cloud_key"])
    assert path == expected


def test_derived_read_route_tolerates_full_prefixed_key(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    _fake_request(monkeypatch, app_mod, claims={"sub": "user-a"})
    # passing the full "derived/..." key (not just the relative tail) still resolves
    out = app_mod.get_scene_derived_route("scan1", result["calibrated_cloud_key"])
    assert out["file"][0].endswith("cloud.ply")


def test_derived_read_route_rejects_foreign_key(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(monkeypatch, app_mod, claims={"sub": "user-a"})
    out = app_mod.get_scene_derived_route("scan1", "anchor/not-mine/cloud.ply")
    assert isinstance(out, tuple) and out[1] == 404


def test_derived_read_route_requires_auth(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(monkeypatch, app_mod, claims={})  # no sub -> unauthenticated
    out = app_mod.get_scene_derived_route("scan1", "anchor/whatever/cloud.ply")
    assert isinstance(out, tuple) and out[1] in (401, 403)
