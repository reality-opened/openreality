"""Splat-import core tests — the module the broker route and the Modal job share.

Focus is on the two things the route tests can't see: that the memory-lean
column read agrees exactly with the generic reader it replaces, and that
persisting from a PATH (rather than from bytes in RAM) produces a byte-identical
artifact. Both are load-bearing for multi-GB imports.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask")

from server.oreos import splat_import
from server.scene_report.splat_io import read_splat_ply, serialize_splat_ply
from server.scene_report.store import ModalScenePersistence

_PROPS = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def _splat_bytes(n=64, seed=3):
    rng = np.random.default_rng(seed)
    fields = {name: rng.normal(size=n).astype(np.float32) for name in _PROPS}
    return serialize_splat_ply(fields)


def _write(tmp_path, data, name="scan.ply"):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_read_columns_matches_the_generic_reader(tmp_path):
    """read_columns() must be a pure optimization of read_splat_ply(): identical
    values, just without materializing 62 properties twice."""
    path = _write(tmp_path, _splat_bytes(n=256))
    reference = read_splat_ply(path)
    header = splat_import.inspect_splat(path)
    wanted = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")
    columns = splat_import.read_columns(path, header, wanted)
    assert set(columns) == set(wanted)
    for name in wanted:
        np.testing.assert_array_equal(columns[name], reference[name])
        assert columns[name].dtype == np.float32
        assert columns[name].flags["C_CONTIGUOUS"]


def test_header_reports_geometry_without_reading_the_body(tmp_path):
    data = _splat_bytes(n=1000)
    path = _write(tmp_path, data)
    header = splat_import.inspect_splat(path)
    assert header.count == 1000
    assert header.names == list(_PROPS)
    assert header.itemsize == 4 * len(_PROPS)
    assert header.data_offset + header.body_bytes == len(data)


def test_truncation_is_detected_and_quantified(tmp_path):
    data = _splat_bytes(n=1000)
    path = _write(tmp_path, data[: len(data) // 3])
    with pytest.raises(splat_import.SplatRejected) as exc:
        splat_import.inspect_splat(path)
    assert exc.value.error == "truncated_splat"
    assert exc.value.status == 422
    assert exc.value.extra["actual_bytes"] < exc.value.extra["expected_bytes"]


def test_gaussian_cap_rejects_before_any_body_read(tmp_path):
    path = _write(tmp_path, _splat_bytes(n=100))
    with pytest.raises(splat_import.SplatRejected) as exc:
        splat_import.inspect_splat(path, max_gaussians=10)
    assert exc.value.error == "too_many_gaussians"
    assert exc.value.extra == {"gaussian_count": 100, "limit": 10}
    assert "Decimate" in exc.value.detail


def test_import_persists_the_artifact_byte_verbatim_from_disk(tmp_path):
    """splat_path= must produce exactly the uploaded bytes — the whole point is
    that a multi-GB file is copied through a buffer, never held in RAM."""
    data = _splat_bytes(n=512)
    path = _write(tmp_path, data, name="lobby.ply")
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))

    result = splat_import.import_splat(store, "user-x", path, "lobby.ply")
    assert result["gaussian_count"] == 512
    assert result["point_count"] == 512

    assert store.get_splat("user-x", result["scan_id"]) == data
    record = store.get_scene("user-x", result["scan_id"])
    assert record["source"] == "imported_splat"
    assert record["splat_key"] == "splat.ply"
    assert record["report"]["degraded"] is True
    assert record["keyframes"] == []
    assert record["trajectory_key"] is None


def test_import_reports_stages_in_order(tmp_path):
    path = _write(tmp_path, _splat_bytes(n=32))
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    seen: list[str] = []
    splat_import.import_splat(store, "user-x", path, "a.ply", on_stage=seen.append)
    # ground_frame runs LAST, after the scene is persisted: it patches facts.metrics on
    # the saved record, and a 32-point fixture has no floor to find — the stage still
    # reports (best-effort, never fatal), which is the behaviour this pins.
    assert seen == ["validate", "parse", "persist", "ground_frame"]


def test_synthesized_cloud_uses_the_dc_sh_decode(tmp_path):
    n = 8
    rng = np.random.default_rng(11)
    fields = {name: np.zeros(n, np.float32) for name in _PROPS}
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    f_dc = rng.normal(scale=0.6, size=(n, 3)).astype(np.float32)
    fields["x"], fields["y"], fields["z"] = positions.T
    for i in range(3):
        fields[f"f_dc_{i}"] = f_dc[:, i]
    path = _write(tmp_path, serialize_splat_ply(fields))
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))

    result = splat_import.import_splat(store, "user-x", path, "a.ply")
    got_pos, got_col = store.get_cloud("user-x", result["scan_id"])
    np.testing.assert_allclose(got_pos, positions, atol=1e-6)
    expected = np.round(
        np.clip(0.5 + splat_import.SH_C0 * f_dc, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    np.testing.assert_array_equal(got_col, expected)


def test_render_advisory_only_fires_above_the_budget(monkeypatch):
    monkeypatch.setattr(splat_import, "RENDER_ADVISORY_GAUSSIANS", 1_000)
    assert splat_import.render_advisory(999) is None
    assert splat_import.render_advisory(1_000) is None
    advisory = splat_import.render_advisory(5_000)
    assert advisory["gaussian_count"] == 5_000
    assert advisory["comfortable_budget"] == 1_000
    assert "5,000" in advisory["detail"]


def test_limits_are_internally_consistent():
    """A guard on the numbers themselves: the inline cap must be far below the
    chunked cap, and one chunk must be smaller than the inline cap (otherwise a
    'too big for one request' file could not be chunked either)."""
    assert splat_import.CHUNK_BYTES <= splat_import.INLINE_MAX_BYTES
    assert splat_import.INLINE_MAX_BYTES < splat_import.CHUNKED_MAX_BYTES
    assert splat_import.RENDER_ADVISORY_GAUSSIANS < splat_import.MAX_GAUSSIANS
