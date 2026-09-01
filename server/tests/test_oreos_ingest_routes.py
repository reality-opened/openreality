"""W1 ingest-route tests (docs/demo-2026-07 design/shell.md §4a/b) — GPU-free.

Same harness as tests/test_demo_routes.py (W0): real Flask + ``test_client`` on a
freshly imported ``server.oreos`` blueprint, with ``server.app`` stubbed via
``sys.modules`` (auth helper + a real ``ModalScenePersistence`` on tmp_path), the
jobs store swapped for a plain dict, and the recon-job spawner swapped for a
recording fake — no modal, no GPU, nothing heavy imported.
"""

from __future__ import annotations

import importlib
import os
import io
import sys
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.scene_report.splat_io import read_splat_ply, serialize_splat_ply
from server.scene_report.store import ModalScenePersistence


def _fresh_demo_package():
    """(Re)import ``server.oreos`` under the CURRENTLY active flask module (a
    sibling test may have cached it under conftest's fake flask)."""
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def ingest_app(monkeypatch, tmp_path):
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
    routes = sys.modules["server.oreos.routes_ingest"]
    spawned: list[dict] = []
    routes.configure_recon_spawner(lambda **kw: spawned.append(kw))
    splat_spawned: list[dict] = []
    routes.configure_splat_spawner(lambda **kw: splat_spawned.append(kw))
    yield types.SimpleNamespace(
        client=app.test_client(),
        store=store,
        stub=stub,
        jobs=jobs,
        routes=routes,
        spawned=spawned,
        splat_spawned=splat_spawned,
        blob_root=str(tmp_path),
    )
    routes.configure_recon_spawner(None)
    routes.configure_splat_spawner(None)
    routes._splat_uploads.clear()
    jobs.configure_jobs_store(None)


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/video
# ---------------------------------------------------------------------------


def _post_video(h, data=b"\x00fake-mp4-bytes\x00" * 64, filename="office loop.mp4", **headers):
    hdrs = {"X-Upload-Filename": filename, **headers}
    return h.client.post(
        "/api/workspace/ingest/video",
        data=data,
        headers=hdrs,
        content_type="application/octet-stream",
    )


def test_video_happy_path_streams_spawns_and_queues(ingest_app, tmp_path):
    h = ingest_app
    body = b"\x00fake-mp4-bytes\x00" * 64
    resp = _post_video(h, data=body, **{"X-Scene-Label": "My Office"})
    assert resp.status_code == 202, resp.get_json()
    payload = resp.get_json()
    assert set(payload) == {"job_id", "scan_id"}

    # exactly one spawn, with the staged path + identity threaded through
    assert len(h.spawned) == 1
    kw = h.spawned[0]
    assert kw["job_id"] == payload["job_id"]
    assert kw["scan_id"] == payload["scan_id"]
    assert kw["user_id"] == "user-a"
    assert kw["label"] == "My Office"
    assert kw["source"] == "recon_video"

    # streamed write landed under <safe_user>/_uploads/<upload_id>/<sanitized name>
    rel = kw["upload_rel_path"]
    parts = rel.split("/")
    assert parts[0] == "user-a" and parts[1] == "_uploads"
    assert parts[3] == "office_loop.mp4"  # space sanitized, suffix kept
    staged = tmp_path / rel
    assert staged.read_bytes() == body

    # queued marker present + owner-scoped readable via the jobs route
    job = h.client.get(f"/api/workspace/jobs/{payload['job_id']}").get_json()
    assert job["status"] == "queued"
    assert job["stage"] == "upload"
    assert job["scan_id"] == payload["scan_id"]
    assert job["upload_bytes"] == len(body)


@pytest.mark.parametrize("filename", ["clip.mkv", "clip.webm", "clip.MOV"])
def test_video_extra_suffixes_accepted(ingest_app, filename):
    resp = _post_video(ingest_app, filename=filename)
    assert resp.status_code == 202, (filename, resp.get_json())


def test_video_bad_suffix_400_no_spawn(ingest_app, tmp_path):
    resp = _post_video(ingest_app, filename="notes.txt")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unsupported_video_type"
    assert ingest_app.spawned == []
    assert not (tmp_path / "user-a" / "_uploads").exists()


def test_video_missing_filename_400(ingest_app):
    resp = ingest_app.client.post(
        "/api/workspace/ingest/video", data=b"x", content_type="application/octet-stream"
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_filename"


def test_video_unauthed_401(ingest_app):
    ingest_app.stub._auth_user_id = lambda: None
    resp = _post_video(ingest_app)
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "invalid_token"}
    assert ingest_app.spawned == []


def test_video_empty_body_400(ingest_app, tmp_path):
    resp = _post_video(ingest_app, data=b"")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "empty_upload"
    assert ingest_app.spawned == []
    # partial dir cleaned up
    uploads = tmp_path / "user-a" / "_uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_video_cap_413_declared_length(ingest_app, monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_app.routes, "VIDEO_UPLOAD_MAX_BYTES", 100)
    resp = _post_video(ingest_app, data=b"x" * 101)
    assert resp.status_code == 413
    body = resp.get_json()
    assert body["error"] == "too_large" and body["limit_bytes"] == 100
    assert ingest_app.spawned == []
    uploads = tmp_path / "user-a" / "_uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_video_cap_413_mid_stream_partial_deleted(ingest_app, monkeypatch, tmp_path):
    """No declared Content-Length (chunked-style input stream) → the cap must trip
    WHILE reading and the partial file must be removed."""
    monkeypatch.setattr(ingest_app.routes, "VIDEO_UPLOAD_MAX_BYTES", 100)
    monkeypatch.setattr(ingest_app.routes, "_STREAM_CHUNK", 32)
    resp = ingest_app.client.post(
        "/api/workspace/ingest/video",
        input_stream=io.BytesIO(b"y" * 500),
        headers={"X-Upload-Filename": "big.mp4"},
        content_type="application/octet-stream",
    )
    assert resp.status_code == 413
    assert ingest_app.spawned == []
    uploads = tmp_path / "user-a" / "_uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_video_gemini_source_header(ingest_app):
    resp = _post_video(ingest_app, **{"X-Demo-Source": "gemini2"})
    assert resp.status_code == 202
    assert ingest_app.spawned[0]["source"] == "recon_gemini2"


def test_video_depthcam_source_header(ingest_app):
    """A depth camera that is NOT a Gemini 2 gets its own source.

    The value is a provenance claim about which hardware shot the footage, so a Kinect /
    Xtion / public RGB-D sequence must not be persisted as "recon_gemini2" — that would
    name a device that never recorded it. Same RGB pipeline either way."""
    resp = _post_video(ingest_app, **{"X-Demo-Source": "depthcam"})
    assert resp.status_code == 202
    assert ingest_app.spawned[0]["source"] == "recon_depthcam"


def test_video_unknown_source_400(ingest_app):
    resp = _post_video(ingest_app, **{"X-Demo-Source": "realsense"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unsupported_source"
    assert ingest_app.spawned == []


def test_video_spawn_failure_502_cleans_up_and_marks_failed(ingest_app, tmp_path):
    def _boom(**kw):
        raise RuntimeError("modal down")

    ingest_app.routes.configure_recon_spawner(_boom)
    resp = _post_video(ingest_app)
    assert resp.status_code == 502
    assert resp.get_json()["error"] == "spawn_failed"
    # staged upload removed so the volume doesn't leak dead uploads
    uploads = tmp_path / "user-a" / "_uploads"
    assert not uploads.exists() or not any(
        p for d in uploads.iterdir() for p in d.iterdir()
    )
    # job record flipped to failed (still pollable — the UI shows an honest error)
    records = [r for r in ingest_app.jobs._jobs_store.values()]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "spawn_failed"


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/splat
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814

# The 17-property 3DGS schema splat_io documents (header order preserved).
_GAUSSIAN_PROPS = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def _gaussian_ply(n=5, positions=None, f_dc=None) -> bytes:
    rng = np.random.default_rng(7)
    if positions is None:
        positions = rng.normal(size=(n, 3)).astype(np.float32)
    if f_dc is None:
        f_dc = rng.normal(scale=0.8, size=(n, 3)).astype(np.float32)
    fields = {}
    for name in _GAUSSIAN_PROPS:
        fields[name] = np.zeros(n, dtype=np.float32)
    fields["x"], fields["y"], fields["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    fields["opacity"] = np.full(n, 2.0, dtype=np.float32)
    fields["rot_0"] = np.ones(n, dtype=np.float32)
    return serialize_splat_ply(fields)


def _post_splat(h, data, filename="scan.ply", **headers):
    hdrs = {"X-Upload-Filename": filename, **headers}
    return h.client.post(
        "/api/workspace/ingest/splat",
        data=data,
        headers=hdrs,
        content_type="application/octet-stream",
    )


def test_splat_happy_path_persists_scene(ingest_app):
    h = ingest_app
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32
    )
    f_dc = np.array(
        [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [3.0, 3.0, 3.0]], dtype=np.float32
    )
    body = _gaussian_ply(n=3, positions=positions, f_dc=f_dc)
    resp = _post_splat(h, body, filename="gallery scan.ply")
    assert resp.status_code == 201, resp.get_json()
    payload = resp.get_json()
    assert payload["gaussian_count"] == 3
    scan_id = payload["scan_id"]

    record = h.store.get_scene("user-a", scan_id)
    assert record is not None
    assert record["source"] == "imported_splat"
    assert record["label"] == "gallery scan.ply"
    assert record["splat_key"] == "splat.ply"
    assert record["keyframes"] == []
    assert record["trajectory_key"] is None
    assert record["point_count"] == 3

    # splat artifact byte-verbatim (Spark renders exactly what was uploaded)
    assert h.store.get_splat("user-a", scan_id) == body

    # synthesized center cloud: means + DC-SH colors, clamped
    cloud = h.store.get_cloud("user-a", scan_id)
    assert cloud is not None
    got_pos, got_col = cloud
    np.testing.assert_allclose(got_pos, positions, atol=1e-6)
    expect_col = np.round(np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(got_col, expect_col)

    # honest degraded report + geometry-only facts
    assert record["report"]["degraded"] is True
    assert record["report"]["objects"] == []
    metrics = record["facts"]["metrics"]
    assert metrics["point_count"] == 3
    assert metrics["bbox_min"] == [-1.0, -2.0, -3.0]
    assert metrics["bbox_max"] == [1.0, 2.0, 3.0]
    assert metrics["vertical_axis_known"] is False
    assert record["facts"]["objects"] == []

    # the scene lists immediately (broker picker shows it)
    scans = h.store.list_scenes("user-a")
    assert scans[0]["scan_id"] == scan_id
    assert scans[0]["has_splat"] is True


def test_splat_label_header_wins_over_filename(ingest_app):
    resp = _post_splat(
        ingest_app, _gaussian_ply(), **{"X-Scene-Label": "Lobby splat"}
    )
    assert resp.status_code == 201
    record = ingest_app.store.get_scene("user-a", resp.get_json()["scan_id"])
    assert record["label"] == "Lobby splat"


def test_splat_plain_point_cloud_422(ingest_app):
    fields = {
        "x": np.zeros(4, np.float32),
        "y": np.zeros(4, np.float32),
        "z": np.zeros(4, np.float32),
    }
    resp = _post_splat(ingest_app, serialize_splat_ply(fields))
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "not_a_gaussian_splat"
    assert "f_dc_0" in body["missing"]
    assert ingest_app.store.list_scenes("user-a") == []


def test_splat_garbage_bytes_422(ingest_app):
    resp = _post_splat(ingest_app, b"this is not a ply file at all")
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_a_gaussian_splat"


def test_splat_ascii_ply_422(ingest_app):
    body = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0.0\n"
    resp = _post_splat(ingest_app, body)
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_a_gaussian_splat"


def test_splat_gaussian_cap_422(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_MAX_GAUSSIANS", 3)
    resp = _post_splat(ingest_app, _gaussian_ply(n=5))
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "too_many_gaussians"
    assert body["gaussian_count"] == 5 and body["limit"] == 3
    assert ingest_app.store.list_scenes("user-a") == []


def test_splat_byte_cap_413(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_UPLOAD_MAX_BYTES", 64)
    resp = _post_splat(ingest_app, _gaussian_ply(n=50))
    assert resp.status_code == 413
    assert resp.get_json()["error"] == "too_large"


def test_splat_spz_imports_a_real_scene(ingest_app):
    """.spz is no longer deferred: a real Spark-encoded fixture imports like a
    .ply (decoder verified against Spark's SpzWriter in tests/test_demo_spz.py)."""
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "spz", "truth_sh3.spz")
    with open(fixture, "rb") as fh:
        body = fh.read()
    resp = _post_splat(ingest_app, body, filename="polycam capture.spz")
    assert resp.status_code == 201, resp.get_json()
    payload = resp.get_json()
    assert payload["gaussian_count"] == 128

    record = ingest_app.store.get_scene("user-a", payload["scan_id"])
    assert record["source"] == "imported_splat"
    assert record["label"] == "polycam capture.spz"
    # The persisted artifact is a real 3DGS .ply — Spark loads splat.ply, not .spz.
    stored = ingest_app.store.get_splat("user-a", payload["scan_id"])
    assert stored.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert record["point_count"] == 128


def test_splat_unreadable_spz_422(ingest_app):
    resp = _post_splat(ingest_app, b"not-actually-an-spz", filename="scan.spz")
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "not_a_gaussian_splat"
    assert "could not be decoded" in body["detail"]


def test_splat_unknown_suffix_400_names_what_is_accepted(ingest_app):
    resp = _post_splat(ingest_app, b"xx", filename="scan.splat")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "unsupported_splat_type"
    assert set(body["allowed"]) == {".ply", ".spz"}


def test_splat_over_inline_cap_413_points_at_the_chunked_route(ingest_app, monkeypatch):
    """The founder's failure mode, made self-explaining: too big for one request
    must say so AND say what to do — never spin into the 150 s platform timeout."""
    monkeypatch.setattr(ingest_app.routes, "SPLAT_UPLOAD_MAX_BYTES", 64)
    resp = _post_splat(ingest_app, _gaussian_ply(n=50))
    assert resp.status_code == 413
    body = resp.get_json()
    assert body["error"] == "too_large"
    assert body["use_chunked_upload"] is True
    assert body["chunked_init_path"] == "/api/workspace/ingest/splat/init"
    assert "150-second" in body["detail"]


def test_splat_truncated_body_422_says_so(ingest_app):
    """A half-arrived file gets a specific diagnosis, not a generic parse error."""
    full = _gaussian_ply(n=200)
    resp = _post_splat(ingest_app, full[: len(full) // 2])
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "truncated_splat"
    assert "stops early" in body["detail"]
    assert body["actual_bytes"] < body["expected_bytes"]


# ---------------------------------------------------------------------------
# Chunked splat import: init → chunk × N → finalize
# ---------------------------------------------------------------------------


def _init_chunked(h, body, filename="big.ply", chunk_bytes=None, **headers):
    if chunk_bytes is not None:
        h.routes.SPLAT_CHUNK_BYTES = chunk_bytes
    hdrs = {
        "X-Upload-Filename": filename,
        "X-Upload-Bytes": str(len(body)),
        **headers,
    }
    return h.client.post("/api/workspace/ingest/splat/init", headers=hdrs)


def _send_chunks(h, upload_id, body, chunk_bytes, skip=()):
    sent = []
    total = (len(body) + chunk_bytes - 1) // chunk_bytes
    for i in range(total):
        if i in skip:
            continue
        piece = body[i * chunk_bytes : (i + 1) * chunk_bytes]
        sent.append(
            h.client.post(
                f"/api/workspace/ingest/splat/{upload_id}/chunk/{i}",
                data=piece,
                content_type="application/octet-stream",
            )
        )
    return sent


def test_chunked_import_spawns_a_job(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 512)
    body = _gaussian_ply(n=300)
    init = _init_chunked(ingest_app, body, **{"X-Scene-Label": "Splatica lobby"})
    assert init.status_code == 200, init.get_json()
    meta = init.get_json()
    assert meta["chunk_bytes"] == 512
    assert meta["total_chunks"] == (len(body) + 511) // 512
    assert meta["max_gaussians"] == ingest_app.routes.SPLAT_MAX_GAUSSIANS

    for resp in _send_chunks(ingest_app, meta["upload_id"], body, 512):
        assert resp.status_code == 200, resp.get_json()
    last = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    )
    assert last.status_code == 202, last.get_json()
    payload = last.get_json()
    assert payload["gaussian_count"] == 300
    assert payload["job_id"] and payload["scan_id"]

    # a queued job record exists from the moment of the 202 (pollable immediately)
    record = ingest_app.jobs._jobs_store[payload["job_id"]]
    assert record["status"] == "queued"
    assert record["kind"] == "splat_import"
    assert record["scan_id"] == payload["scan_id"]

    # the job was spawned on the STAGED bytes, which are byte-identical
    assert len(ingest_app.splat_spawned) == 1
    spawned = ingest_app.splat_spawned[0]
    assert spawned["scan_id"] == payload["scan_id"]
    assert spawned["label"] == "Splatica lobby"
    staged = os.path.join(ingest_app.blob_root, spawned["upload_rel_path"])
    with open(staged, "rb") as fh:
        assert fh.read() == body


def test_chunks_may_arrive_out_of_order(ingest_app, monkeypatch):
    """Seek-writes at index*chunk_bytes — order-independent and retryable."""
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 256)
    body = _gaussian_ply(n=200)
    meta = _init_chunked(ingest_app, body).get_json()
    total = meta["total_chunks"]
    order = list(reversed(range(total)))
    for i in order:
        piece = body[i * 256 : (i + 1) * 256]
        resp = ingest_app.client.post(
            f"/api/workspace/ingest/splat/{meta['upload_id']}/chunk/{i}",
            data=piece, content_type="application/octet-stream",
        )
        assert resp.status_code == 200
    # re-send one chunk: idempotent, must not corrupt or double-count
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/chunk/0",
        data=body[:256], content_type="application/octet-stream",
    )
    assert resp.get_json()["chunks_received"] == total
    assert ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    ).status_code == 202
    staged = os.path.join(ingest_app.blob_root,
                          ingest_app.splat_spawned[0]["upload_rel_path"])
    with open(staged, "rb") as fh:
        assert fh.read() == body


def test_chunk_size_mismatch_400(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 512)
    body = _gaussian_ply(n=300)
    meta = _init_chunked(ingest_app, body).get_json()
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/chunk/0",
        data=body[:100], content_type="application/octet-stream",
    )
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "chunk_size_mismatch"
    assert payload["expected_bytes"] == 512 and payload["received_bytes"] == 100


def test_chunk_index_out_of_range_400(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 512)
    meta = _init_chunked(ingest_app, _gaussian_ply(n=300)).get_json()
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/chunk/999",
        data=b"x", content_type="application/octet-stream",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_chunk_index"


def test_finalize_missing_chunks_409_lists_them(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 512)
    body = _gaussian_ply(n=300)
    meta = _init_chunked(ingest_app, body).get_json()
    _send_chunks(ingest_app, meta["upload_id"], body, 512, skip={1})
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    )
    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["error"] == "incomplete_upload"
    assert payload["missing_chunks"] == [1]
    assert ingest_app.splat_spawned == []


def test_finalize_rejects_a_plain_point_cloud_without_spawning(ingest_app, monkeypatch):
    """The broker's cheap header gate: a bad file must cost zero Modal spend."""
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 4096)
    fields = {a: np.zeros(64, np.float32) for a in ("x", "y", "z")}
    body = serialize_splat_ply(fields)
    meta = _init_chunked(ingest_app, body).get_json()
    _send_chunks(ingest_app, meta["upload_id"], body, 4096)
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_a_gaussian_splat"
    assert ingest_app.splat_spawned == []
    assert ingest_app.store.list_scenes("user-a") == []
    # staged bytes are released, not left pinning the volume
    uploads = os.path.join(ingest_app.blob_root, "user-a", "_uploads")
    assert not os.path.isdir(uploads) or os.listdir(uploads) == []


def test_finalize_rejects_over_gaussian_cap_without_spawning(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 4096)
    monkeypatch.setattr(ingest_app.routes, "SPLAT_MAX_GAUSSIANS", 10)
    body = _gaussian_ply(n=64)
    meta = _init_chunked(ingest_app, body).get_json()
    _send_chunks(ingest_app, meta["upload_id"], body, 4096)
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    )
    assert resp.status_code == 422
    payload = resp.get_json()
    assert payload["error"] == "too_many_gaussians"
    assert payload["gaussian_count"] == 64 and payload["limit"] == 10
    assert "Decimate" in payload["detail"]
    assert ingest_app.splat_spawned == []


def test_init_over_total_cap_413(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNKED_MAX_BYTES", 1024)
    resp = ingest_app.client.post(
        "/api/workspace/ingest/splat/init",
        headers={"X-Upload-Filename": "huge.ply", "X-Upload-Bytes": "999999999"},
    )
    assert resp.status_code == 413
    assert resp.get_json()["error"] == "too_large"


def test_init_requires_declared_size(ingest_app):
    resp = ingest_app.client.post(
        "/api/workspace/ingest/splat/init", headers={"X-Upload-Filename": "a.ply"}
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_upload_bytes"


def test_unknown_upload_id_404_explains_itself(ingest_app):
    resp = ingest_app.client.post(
        "/api/workspace/ingest/splat/deadbeef/chunk/0",
        data=b"x", content_type="application/octet-stream",
    )
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["error"] == "upload_not_found"
    assert "server restarted" in body["detail"]


def test_another_users_upload_is_invisible(ingest_app):
    body = _gaussian_ply(n=8)
    meta = _init_chunked(ingest_app, body).get_json()
    ingest_app.stub._auth_user_id = lambda: "user-b"
    resp = ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    )
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "upload_not_found"


def test_cancel_releases_staged_bytes(ingest_app, monkeypatch):
    monkeypatch.setattr(ingest_app.routes, "SPLAT_CHUNK_BYTES", 512)
    body = _gaussian_ply(n=300)
    meta = _init_chunked(ingest_app, body).get_json()
    _send_chunks(ingest_app, meta["upload_id"], body, 512, skip={0})
    resp = ingest_app.client.delete(f"/api/workspace/ingest/splat/{meta['upload_id']}")
    assert resp.status_code == 200 and resp.get_json()["cancelled"] is True
    uploads = os.path.join(ingest_app.blob_root, "user-a", "_uploads")
    assert not os.path.isdir(uploads) or os.listdir(uploads) == []
    assert ingest_app.client.post(
        f"/api/workspace/ingest/splat/{meta['upload_id']}/finalize"
    ).status_code == 404


def test_chunked_routes_require_auth(ingest_app):
    ingest_app.stub._auth_user_id = lambda: None
    for call in (
        lambda: ingest_app.client.post(
            "/api/workspace/ingest/splat/init",
            headers={"X-Upload-Filename": "a.ply", "X-Upload-Bytes": "10"}),
        lambda: ingest_app.client.post("/api/workspace/ingest/splat/x/chunk/0", data=b"x"),
        lambda: ingest_app.client.post("/api/workspace/ingest/splat/x/finalize"),
        lambda: ingest_app.client.delete("/api/workspace/ingest/splat/x"),
    ):
        assert call().status_code == 401


def test_render_advisory_rides_along_when_the_splat_is_heavy(ingest_app, monkeypatch):
    """Honest, non-blocking: big imports still succeed but say what it will cost."""
    monkeypatch.setattr(ingest_app.routes.splat_import, "RENDER_ADVISORY_GAUSSIANS", 2)
    resp = _post_splat(ingest_app, _gaussian_ply(n=9))
    assert resp.status_code == 201
    advisory = resp.get_json()["render_advisory"]
    assert advisory["gaussian_count"] == 9
    assert advisory["comfortable_budget"] == 2
    assert "renders smoothly" in advisory["detail"]


def test_splat_empty_vertex_count_422(ingest_app):
    fields = {name: np.zeros(0, np.float32) for name in _GAUSSIAN_PROPS}
    resp = _post_splat(ingest_app, serialize_splat_ply(fields))
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_a_gaussian_splat"


def test_splat_nonfinite_means_filtered_from_cloud(ingest_app):
    positions = np.array(
        [[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
    )
    body = _gaussian_ply(n=3, positions=positions)
    resp = _post_splat(ingest_app, body)
    assert resp.status_code == 201
    payload = resp.get_json()
    assert payload["gaussian_count"] == 3  # splat artifact keeps every gaussian
    record = ingest_app.store.get_scene("user-a", payload["scan_id"])
    assert record["point_count"] == 2  # cloud drops the non-finite mean
    # scene_center stayed finite (a NaN here would poison display recentering)
    assert all(np.isfinite(v) for v in record["scene_center"])


def test_splat_unauthed_401(ingest_app):
    ingest_app.stub._auth_user_id = lambda: None
    resp = _post_splat(ingest_app, _gaussian_ply())
    assert resp.status_code == 401


def test_splat_missing_filename_400(ingest_app):
    resp = ingest_app.client.post(
        "/api/workspace/ingest/splat", data=b"x", content_type="application/octet-stream"
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing_filename"


def test_splat_fixture_roundtrips_through_splat_io(tmp_path):
    """Sanity: the fixture built with serialize_splat_ply reads back through
    read_splat_ply with the schema fields intact (guards the fixture itself)."""
    path = tmp_path / "fixture.ply"
    path.write_bytes(_gaussian_ply(n=4))
    fields = read_splat_ply(str(path))
    for name in ("x", "f_dc_0", "opacity", "scale_0", "rot_0"):
        assert name in fields and fields[name].shape == (4,)


# ---------------------------------------------------------------------------
# GET /api/scenes/<scan_id>/demo/trajectory.npz (W1 — gemini-glue anchor step)
# ---------------------------------------------------------------------------


def _minimal_scene_kwargs():
    from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport

    return dict(
        report=SceneReport(summary="s", room_type="office"),
        facts=SceneFacts(metrics=SceneMetrics(num_submaps=1)),
    )


def _save_scene_with_trajectory(store, scan_id="scan-t", n=5):
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n, dtype=np.float32)  # distinct translations
    traj = {
        "poses": poses,
        "intrinsics": np.tile(
            np.array([500.0, 500.0, 320.0, 240.0], np.float32), (n, 1)
        ),
        "source_frame_id": np.arange(n, dtype=np.float32) * 3.0,
    }
    kw = _minimal_scene_kwargs()
    store.save_scene("user-a", scan_id, kw["report"], kw["facts"], trajectory=traj)
    return traj


def test_trajectory_npz_roundtrip(ingest_app):
    traj = _save_scene_with_trajectory(ingest_app.store)
    resp = ingest_app.client.get("/api/scenes/scan-t/demo/trajectory.npz")
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert set(data.files) == {"poses", "intrinsics", "source_frame_id"}
        np.testing.assert_array_equal(data["poses"], traj["poses"])
        np.testing.assert_array_equal(data["intrinsics"], traj["intrinsics"])
        np.testing.assert_array_equal(
            data["source_frame_id"], traj["source_frame_id"]
        )


def test_trajectory_npz_404_when_absent(ingest_app):
    kw = _minimal_scene_kwargs()
    ingest_app.store.save_scene("user-a", "scan-nt", kw["report"], kw["facts"])
    resp = ingest_app.client.get("/api/scenes/scan-nt/demo/trajectory.npz")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "no_trajectory"


def test_trajectory_npz_404_unknown_scan(ingest_app):
    resp = ingest_app.client.get("/api/scenes/nope/demo/trajectory.npz")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "not_found"


def test_trajectory_npz_401_unauthed(ingest_app):
    _save_scene_with_trajectory(ingest_app.store, scan_id="scan-a")
    ingest_app.stub._auth_user_id = lambda: None
    resp = ingest_app.client.get("/api/scenes/scan-a/demo/trajectory.npz")
    assert resp.status_code == 401
