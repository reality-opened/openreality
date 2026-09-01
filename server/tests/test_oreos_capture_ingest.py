"""LiDAR capture-session ingest lane (EXP-45 metric grounding): zip validation,
safe extraction, the persistence bridge, and the init/chunk/finalize routes.

Layered like the recordings/splat suites this mirrors:
  - zip_validate / session_assembly: pure stdlib, no fixtures needed.
  - persist_scene: synthetic metric-layer outputs (numpy arrays + plain dicts) ->
    asserted ``save_scene``/derived-artifact shape — no Modal, no GPU, no core.
  - routes: real Flask + test_client on a freshly imported ``server.oreos``
    blueprint (test_oreos_ingest_routes.py's harness), spawner injected.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from server.oreos.capture.persist_scene import build_capture_scene_payload, persist_capture_scene
from server.oreos.capture.session_assembly import extract_capture_session
from server.oreos.capture.zip_validate import CaptureZipRejected, validate_capture_zip
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_zip(path: Path, entries: dict, *, extra_members: list = None) -> Path:
    """Write a store-only zip at ``path`` with ``entries`` (name -> bytes), plus
    any raw ``ZipInfo`` members in ``extra_members`` (for zip-slip cases where the
    name itself must bypass normal path handling)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        for info, data in extra_members or []:
            zf.writestr(info, data)
    return path


def _valid_session_entries(**overrides) -> dict:
    entries = {
        "odometry.csv": b"timestamp,frame,x,y,z,qx,qy,qz,qw\n0.0,0,0,0,0,0,0,0,1\n",
        "rgb.mp4": b"\x00fake-mp4-bytes\x00" * 8,
        "depth/000000.png": b"\x89PNGfakedepth",
        "confidence/000000.png": b"\x89PNGfakeconf",
        "meta.json": b'{"rgb_width": 1920, "rgb_height": 1440}',
        "imu_raw_gyro.csv": b"timestamp,sequence,rx_radps,ry_radps,rz_radps\n",
    }
    entries.update(overrides)
    return entries


# ---------------------------------------------------------------------------
# zip_validate
# ---------------------------------------------------------------------------


def test_validate_happy_path(tmp_path):
    zp = _write_zip(tmp_path / "session.zip", _valid_session_entries())
    info = validate_capture_zip(str(zp))
    assert "odometry.csv" in info.names
    assert "rgb.mp4" in info.names
    assert info.n_entries == len(_valid_session_entries())
    assert info.total_uncompressed > 0


@pytest.mark.parametrize(
    "missing_key,expect_in_missing",
    [
        ("odometry.csv", "odometry.csv"),
        ("rgb.mp4", "rgb.mp4"),
    ],
)
def test_validate_missing_top_level_entry(tmp_path, missing_key, expect_in_missing):
    entries = _valid_session_entries()
    del entries[missing_key]
    zp = _write_zip(tmp_path / "session.zip", entries)
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    exc = exc_info.value
    assert exc.error == "missing_required_entries"
    assert exc.status == 400
    assert expect_in_missing in exc.extra["missing"]


def test_validate_missing_depth_prefix(tmp_path):
    entries = _valid_session_entries()
    del entries["depth/000000.png"]
    zp = _write_zip(tmp_path / "session.zip", entries)
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert "depth/" in exc_info.value.extra["missing"]


def test_validate_missing_confidence_prefix(tmp_path):
    entries = _valid_session_entries()
    del entries["confidence/000000.png"]
    zp = _write_zip(tmp_path / "session.zip", entries)
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert "confidence/" in exc_info.value.extra["missing"]


def test_validate_rejects_dotdot_traversal(tmp_path):
    entries = _valid_session_entries()
    entries["../../etc/passwd"] = b"pwned"
    zp = _write_zip(tmp_path / "session.zip", entries)
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert exc_info.value.error == "zip_slip"
    assert exc_info.value.status == 400


def test_validate_rejects_nested_dotdot(tmp_path):
    entries = _valid_session_entries()
    entries["depth/../../../etc/passwd"] = b"pwned"
    zp = _write_zip(tmp_path / "session.zip", entries)
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert exc_info.value.error == "zip_slip"


def test_validate_rejects_absolute_path_entry(tmp_path):
    entries = _valid_session_entries()
    zp = tmp_path / "session.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        info = zipfile.ZipInfo("/etc/passwd")
        zf.writestr(info, b"pwned")
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert exc_info.value.error == "zip_slip"


def test_validate_not_a_zip(tmp_path):
    p = tmp_path / "not_a_zip.zip"
    p.write_bytes(b"this is definitely not a zip file")
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(p))
    assert exc_info.value.error == "not_a_zip"
    assert exc_info.value.status == 400


def test_validate_zip_slip_checked_before_missing_entries(tmp_path):
    """A hostile entry is refused even in an otherwise-empty/invalid zip — the
    zip-slip check must never be skipped because required entries are absent."""
    zp = tmp_path / "session.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("../evil.txt", b"pwned")
    with pytest.raises(CaptureZipRejected) as exc_info:
        validate_capture_zip(str(zp))
    assert exc_info.value.error == "zip_slip"


# ---------------------------------------------------------------------------
# session_assembly
# ---------------------------------------------------------------------------


def test_extract_preserves_session_layout(tmp_path):
    zp = _write_zip(tmp_path / "session.zip", _valid_session_entries())
    dest = tmp_path / "extracted"
    out = extract_capture_session(str(zp), str(dest))
    assert out == str(dest)
    assert (dest / "odometry.csv").is_file()
    assert (dest / "rgb.mp4").is_file()
    assert (dest / "depth" / "000000.png").is_file()
    assert (dest / "confidence" / "000000.png").is_file()
    assert (dest / "meta.json").read_text().startswith("{")


def test_extract_rejects_zip_slip_directory_only_entry(tmp_path):
    """A malicious entry disguised as a DIRECTORY (trailing slash) gets the exact
    same zip-slip check as a file entry — it must not get a free pass just because
    nothing is written directly to it."""
    zp = tmp_path / "session.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("odometry.csv", b"ok")
        zf.writestr("../../evil_dir/", b"")
    dest = tmp_path / "sandbox" / "extracted"
    with pytest.raises(CaptureZipRejected) as exc_info:
        extract_capture_session(str(zp), str(dest))
    assert exc_info.value.error == "zip_slip"
    assert not (tmp_path / "evil_dir").exists()


def test_extract_rejects_zip_slip_even_without_prior_validation(tmp_path):
    """Defense in depth: extraction re-checks zip-slip even if a caller skipped
    validate_capture_zip. Nothing must land outside dest_dir."""
    zp = tmp_path / "session.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("odometry.csv", b"ok")
        zf.writestr("../../escaped.txt", b"pwned")
    dest = tmp_path / "sandbox" / "extracted"
    with pytest.raises(CaptureZipRejected) as exc_info:
        extract_capture_session(str(zp), str(dest))
    assert exc_info.value.error == "zip_slip"
    # nothing escaped: no file lands outside the sandbox root's parent
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists()


# ---------------------------------------------------------------------------
# persist_scene
# ---------------------------------------------------------------------------


def _synthetic_trajectory_rows(n=6):
    rows = []
    for i in range(n):
        t = 1000.0 + i * 0.5
        rows.append([t, 0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float64)


def _synthetic_metric_report(**overrides):
    report = {
        "gauges": 3,
        "scale_priors": 3,
        "junction_rotations": 2,
        "junctions_skipped_chain_break": 0,
        "vision_consistency_edges": 2,
        "vision_vs_arkit_junction_rot_deg": [1.2, 0.8],
        "final_error": 0.0042,
        "scales": {0: 0.51, 1: 0.52, 2: 0.50},
    }
    report.update(overrides)
    return report


def test_build_capture_scene_payload_shape():
    rows = _synthetic_trajectory_rows()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 1]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    payload = build_capture_scene_payload(
        rows, points, colors, _synthetic_metric_report(),
        session_label="walk1.zip", frames_total=120, frames_kept=48,
        submap_count=3, loop_closures=1,
    )
    assert payload["report"].degraded is True
    assert "LiDAR-grounded" in payload["report"].summary
    assert "48/120" in payload["report"].summary
    facts = payload["facts"]
    assert facts.metrics.point_count == 3
    assert facts.metrics.vertical_axis_known is True
    assert facts.metrics.up_axis == [0.0, 1.0, 0.0]
    assert "metres" in facts.units_note
    traj = payload["trajectory"]
    assert traj["poses"].shape == (6, 4, 4)
    assert traj["intrinsics"].shape == (6, 4) and float(traj["intrinsics"].sum()) == 0.0
    assert traj["source_frame_id"].dtype == np.float32
    assert traj["source_frame_id"][0] == 0.0
    pts, cols = payload["points"]
    assert pts.shape == (3, 3) and cols.shape == (3, 3)


def test_build_capture_scene_payload_handles_no_points():
    rows = _synthetic_trajectory_rows()
    payload = build_capture_scene_payload(
        rows, None, None, _synthetic_metric_report(),
        session_label="walk1.zip", frames_total=10, frames_kept=5,
        submap_count=1, loop_closures=0,
    )
    assert payload["points"] is None
    assert payload["facts"].metrics.point_count == 0
    # vertical axis is still known — it comes from the ARKit gravity anchor, not
    # from a point-cloud plane fit
    assert payload["facts"].metrics.vertical_axis_known is True


def test_build_capture_scene_payload_refuses_short_trajectory():
    with pytest.raises(ValueError):
        build_capture_scene_payload(
            _synthetic_trajectory_rows(n=1), None, None, _synthetic_metric_report(),
            session_label="x", frames_total=1, frames_kept=1, submap_count=1, loop_closures=0,
        )


def test_persist_writes_scene_and_metric_derived_artifacts(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    rows = _synthetic_trajectory_rows()
    points = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    colors = np.full((50, 3), 128, dtype=np.uint8)
    gauges = {"0": {"scale": 0.51, "matrix": np.eye(4).tolist()}}
    telemetry = {"keyframe_gate": "vio", "lidar_anchor": True, "submap_ratios": []}

    result = persist_capture_scene(
        store, "user-a", "scan-lidar-1",
        trajectory_rows=rows, points=points, colors=colors,
        metric_report=_synthetic_metric_report(), gauges=gauges, anchor_telemetry=telemetry,
        session_label="walk1.zip", frames_total=120, frames_kept=48,
        submap_count=3, loop_closures=1,
    )

    record = store.get_scene("user-a", "scan-lidar-1")
    assert record is not None
    assert record.get("source") == "recon_lidar"
    assert record.get("trajectory_key")
    assert record.get("points_key")
    assert result["scan_id"] == "scan-lidar-1"
    derived = result["derived"]
    assert set(derived) == {
        "metric_trajectory_tum.txt", "metric_gauges.json", "anchor_telemetry.json",
    }
    for key in derived.values():
        assert key.startswith("derived/demo/capture/")
        path = store.get_derived_artifact_path("user-a", "scan-lidar-1", key)
        assert path and os.path.isfile(path)

    gauges_path = store.get_derived_artifact_path(
        "user-a", "scan-lidar-1", derived["metric_gauges.json"]
    )
    saved = json.loads(Path(gauges_path).read_text())
    assert saved["report"]["gauges"] == 3
    assert saved["gauges"] == gauges

    tum_path = store.get_derived_artifact_path(
        "user-a", "scan-lidar-1", derived["metric_trajectory_tum.txt"]
    )
    tum_lines = Path(tum_path).read_text().strip().splitlines()
    assert len(tum_lines) == rows.shape[0]
    assert len(tum_lines[0].split()) == 8


def test_persist_uses_label_or_falls_back_to_session_label(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    rows = _synthetic_trajectory_rows()
    persist_capture_scene(
        store, "user-a", "scan-lidar-2",
        trajectory_rows=rows, points=None, colors=None,
        metric_report=_synthetic_metric_report(), gauges={}, anchor_telemetry={},
        session_label="living_room.zip", frames_total=10, frames_kept=5,
        submap_count=1, loop_closures=0, label="Living room walk",
    )
    record = store.get_scene("user-a", "scan-lidar-2")
    assert record["label"] == "Living room walk"


# ---------------------------------------------------------------------------
# routes: init / chunk / finalize
# ---------------------------------------------------------------------------

flask = pytest.importorskip("flask")


def _fresh_demo_package():
    """(Re)import ``server.oreos`` under the CURRENTLY active flask module — the
    same dance test_oreos_ingest_routes.py uses, protecting ``recordings`` and
    ``capture`` (pure-python, no flask dependency) from being needlessly reloaded."""
    for name in [
        m
        for m in list(sys.modules)
        if m == "server.oreos"
        or (
            m.startswith("server.oreos.")
            and not m.startswith("server.oreos.recordings")
            and not m.startswith("server.oreos.capture")
        )
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def capture_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(
        _scene_persistence=store,
        _auth_user_id=lambda: "user-a",
    )
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    jobs = sys.modules["server.oreos.jobs"]
    jobs.configure_jobs_store({})
    routes = sys.modules["server.oreos.routes_capture"]
    spawned: list[dict] = []
    routes.configure_capture_spawner(lambda **kw: spawned.append(kw))
    yield types.SimpleNamespace(
        client=app.test_client(),
        store=store,
        stub=stub,
        jobs=jobs,
        routes=routes,
        spawned=spawned,
        blob_root=str(tmp_path),
    )
    routes.configure_capture_spawner(None)
    routes._capture_uploads.clear()
    jobs.configure_jobs_store(None)


def _zip_bytes(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _init(h, body_bytes: bytes, filename="session1.zip", **overrides):
    body = {"filename": filename, "size_bytes": len(body_bytes)}
    body.update(overrides)
    return h.client.post(
        "/api/workspace/ingest/capture/init",
        data=json.dumps(body),
        content_type="application/json",
    )


def _send_all_chunks(h, upload_id, data: bytes, chunk_bytes: int, skip=()):
    total = (len(data) + chunk_bytes - 1) // chunk_bytes
    resps = []
    for i in range(total):
        if i in skip:
            continue
        piece = data[i * chunk_bytes : (i + 1) * chunk_bytes]
        resps.append(
            h.client.post(
                f"/api/workspace/ingest/capture/chunk/{i}",
                data=piece,
                content_type="application/octet-stream",
                headers={"X-Upload-Id": upload_id},
            )
        )
    return resps


def _finalize(h, upload_id):
    return h.client.post(
        "/api/workspace/ingest/capture/finalize", headers={"X-Upload-Id": upload_id}
    )


def test_capture_happy_path_init_chunk_finalize_spawns_job(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 64)
    data = _zip_bytes(_valid_session_entries())

    init = _init(h, data)
    assert init.status_code == 200, init.get_json()
    meta = init.get_json()
    assert meta["chunk_bytes"] == 64
    assert meta["total_chunks"] == (len(data) + 63) // 64
    assert "max_bytes" in meta

    for resp in _send_all_chunks(h, meta["upload_id"], data, 64):
        assert resp.status_code == 200, resp.get_json()

    fin = _finalize(h, meta["upload_id"])
    assert fin.status_code == 202, fin.get_json()
    payload = fin.get_json()
    assert set(payload) == {"job_id", "scan_id"}

    # queued job record exists immediately, pollable via the existing jobs route
    record = h.jobs._jobs_store[payload["job_id"]]
    assert record["status"] == "queued"
    assert record["kind"] == "capture_lidar"
    assert record["scan_id"] == payload["scan_id"]

    poll = h.client.get(f"/api/workspace/jobs/{payload['job_id']}")
    assert poll.status_code == 200
    assert poll.get_json()["scan_id"] == payload["scan_id"]

    assert len(h.spawned) == 1
    spawned = h.spawned[0]
    assert spawned["scan_id"] == payload["scan_id"]
    assert spawned["session_label"] == "session1.zip"
    staged = os.path.join(h.blob_root, spawned["upload_rel_path"])
    with open(staged, "rb") as fh:
        assert fh.read() == data


def test_capture_init_requires_zip_filename(capture_app):
    resp = _init(capture_app, b"x" * 10, filename="session1.tar")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_filename"


def test_capture_init_requires_size_bytes(capture_app):
    resp = capture_app.client.post(
        "/api/workspace/ingest/capture/init",
        data=json.dumps({"filename": "s.zip"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_size_bytes"


def test_capture_init_over_cap_413(capture_app, monkeypatch):
    monkeypatch.setattr(capture_app.routes, "CAPTURE_UPLOAD_MAX_BYTES", 1024)
    resp = _init(capture_app, b"x" * 2048)
    assert resp.status_code == 413
    assert resp.get_json()["error"] == "too_large"


def test_capture_finalize_missing_odometry_csv_400_no_spawn(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 4096)
    entries = _valid_session_entries()
    del entries["odometry.csv"]
    data = _zip_bytes(entries)
    meta = _init(h, data).get_json()
    _send_all_chunks(h, meta["upload_id"], data, 4096)
    resp = _finalize(h, meta["upload_id"])
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "missing_required_entries"
    assert "odometry.csv" in payload["missing"]
    assert h.spawned == []
    assert h.store.list_scenes("user-a") == []
    # staged bytes released, not left pinning the volume
    uploads = os.path.join(h.blob_root, "user-a", "_uploads")
    assert not os.path.isdir(uploads) or os.listdir(uploads) == []


def test_capture_finalize_zip_slip_400_no_spawn(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 4096)
    entries = _valid_session_entries()
    entries["../../evil.txt"] = b"pwned"
    data = _zip_bytes(entries)
    meta = _init(h, data).get_json()
    _send_all_chunks(h, meta["upload_id"], data, 4096)
    resp = _finalize(h, meta["upload_id"])
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "zip_slip"
    assert h.spawned == []


def test_capture_chunk_size_mismatch_400(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 512)
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    resp = h.client.post(
        "/api/workspace/ingest/capture/chunk/0",
        data=data[:100],
        content_type="application/octet-stream",
        headers={"X-Upload-Id": meta["upload_id"]},
    )
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "chunk_size_mismatch"
    assert payload["expected_bytes"] == 512 and payload["received_bytes"] == 100


def test_capture_chunk_oversize_body_400(capture_app, monkeypatch):
    """An over-large single chunk (more bytes than the declared chunk size) is
    exactly the same 'expected vs received' mismatch — no special-casing needed,
    but explicitly exercised since "oversize chunk handling" is a named case."""
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 64)
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    oversized = data[:64] + b"\x00" * 200  # way more than the 64-byte chunk size
    resp = h.client.post(
        "/api/workspace/ingest/capture/chunk/0",
        data=oversized,
        content_type="application/octet-stream",
        headers={"X-Upload-Id": meta["upload_id"]},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "chunk_size_mismatch"


def test_capture_chunk_bad_index_400(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 512)
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    resp = h.client.post(
        "/api/workspace/ingest/capture/chunk/999",
        data=b"x",
        content_type="application/octet-stream",
        headers={"X-Upload-Id": meta["upload_id"]},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_chunk_index"


def test_capture_finalize_incomplete_upload_409(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 512)
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    _send_all_chunks(h, meta["upload_id"], data, 512, skip={0})
    resp = _finalize(h, meta["upload_id"])
    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["error"] == "incomplete_upload"
    assert payload["missing_chunks"] == [0]
    assert h.spawned == []


def test_capture_unknown_upload_id_404(capture_app):
    resp = capture_app.client.post(
        "/api/workspace/ingest/capture/finalize", headers={"X-Upload-Id": "nope"}
    )
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "upload_not_found"


def test_capture_missing_upload_id_header_400(capture_app):
    resp = capture_app.client.post("/api/workspace/ingest/capture/finalize")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_upload_id"


def test_capture_another_users_upload_is_invisible(capture_app, monkeypatch):
    h = capture_app
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    h.stub._auth_user_id = lambda: "user-b"
    resp = h.client.post(
        f"/api/workspace/ingest/capture/chunk/0",
        data=data,
        content_type="application/octet-stream",
        headers={"X-Upload-Id": meta["upload_id"]},
    )
    assert resp.status_code == 404


def test_capture_routes_require_auth(capture_app):
    h = capture_app
    h.stub._auth_user_id = lambda: None
    assert _init(h, b"x" * 10).status_code == 401
    assert (
        h.client.post(
            "/api/workspace/ingest/capture/chunk/0",
            data=b"x",
            headers={"X-Upload-Id": "whatever"},
        ).status_code
        == 401
    )
    assert (
        h.client.post(
            "/api/workspace/ingest/capture/finalize", headers={"X-Upload-Id": "whatever"}
        ).status_code
        == 401
    )


def test_capture_cancel_releases_staged_bytes(capture_app, monkeypatch):
    h = capture_app
    monkeypatch.setattr(h.routes, "CAPTURE_CHUNK_BYTES", 512)
    data = _zip_bytes(_valid_session_entries())
    meta = _init(h, data).get_json()
    resp = h.client.delete(f"/api/workspace/ingest/capture/{meta['upload_id']}")
    assert resp.status_code == 200
    assert resp.get_json()["cancelled"] is True
    uploads = os.path.join(h.blob_root, "user-a", "_uploads")
    assert not os.path.isdir(uploads) or os.listdir(uploads) == []
    # a chunk against the cancelled upload now 404s
    resp2 = h.client.post(
        "/api/workspace/ingest/capture/chunk/0",
        data=data[:512],
        content_type="application/octet-stream",
        headers={"X-Upload-Id": meta["upload_id"]},
    )
    assert resp2.status_code == 404
