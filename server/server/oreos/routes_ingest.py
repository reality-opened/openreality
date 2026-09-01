"""Ingest routes (W1, docs/demo-2026-07 design/shell.md §4a/b).

Two ways a scene enters the demo store:

``POST /api/workspace/ingest/video`` — raw ``application/octet-stream`` (NOT
multipart: the broker has ~4 GB RAM, so the body is **streamed to the scenes
volume in chunks**, never buffered whole), headers ``X-Upload-Filename``
(+ optional ``X-Scene-Label``, ``X-Demo-Source: gemini2``). Staged under
``<safe_user>/_uploads/<upload_id>/<filename>``, then
``modal_oreos_ingest.demo_recon_job`` is spawned on it (A10G pilot recon) and a
``queued`` marker lands in the demo-jobs store so ``GET
/api/workspace/jobs/<job_id>`` answers from the moment of the 202.

Splat import comes in TWO shapes, because one synchronous request cannot carry a
big splat (see the 2026-07-30 diagnosis in ``server/oreos/splat_import.py``:
Modal enforces a hard **150-second HTTP request timeout** on every web function,
so the founder's ~2 GB upload was killed mid-body and produced nothing at all —
no log line, no staged file, no scene, just a spinner then a generic network
error):

``POST /api/workspace/ingest/splat`` — the small-file fast path, unchanged
contract: raw stream, validate, persist, ``201 {scan_id, gaussian_count}``. Now
capped at ``SPLAT_UPLOAD_MAX_BYTES`` (64 MiB — the 150 s ceiling expressed in
bytes with a slow-connection safety factor). Over that it answers 413 naming
the chunked route instead of letting the request die as a timeout.

``POST /api/workspace/ingest/splat/init`` → ``.../<upload_id>/chunk/<index>`` →
``.../<upload_id>/finalize`` — the large-file path. Each chunk is its own small
request (16 MiB, tens of seconds even on a bad connection), seek-written at
``index * chunk_bytes`` into a pre-sized file staged on the scenes volume, so
chunks are order-independent and safely retryable. ``finalize`` re-checks the
PLY/SPZ header on the broker (~1 KB read — a bad file is rejected *before* a job
is spawned), then spawns ``demo_splat_import_job`` and answers ``202 {job_id,
scan_id}``. The client polls the same ``GET /api/workspace/jobs/<job_id>`` the
video path uses, and gets a real stage timeline instead of a spinner.

Both shapes converge on ``server/oreos/splat_import.py``, which validates,
**synthesizes the center point cloud** (means + DC-SH colors) so points/anchor/
planner light up, and persists a real ``PersistedScene`` with an honest
``degraded`` report, geometry-only facts, and ``source="imported_splat"``.

``.spz`` is now accepted on both paths (``server/oreos/spz.py``, a numpy port of
Spark's own ``SpzReader``, verified against ``SpzWriter`` ground truth) — it is
decoded to a 3DGS ``.ply`` and imported identically.

Auth: owner-only via ``auth_user_id()`` (see ``server/oreos/__init__``).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import jobs as demo_jobs
from server.oreos import splat_import

# The one dispatch rule for what counts as a video lives in server/ingest/video.py
# (import-light: stdlib only). The demo route additionally accepts .mkv/.webm
# (design/shell.md §4a) — phone/screen-recorder containers ffmpeg decodes fine.
from server.ingest.video import VIDEO_SUFFIXES

DEMO_VIDEO_SUFFIXES = tuple(VIDEO_SUFFIXES) + (".mkv", ".webm")
SPLAT_SUFFIXES = (".ply", ".spz")

# Streaming caps, enforced WHILE reading (a too-large body 413s mid-stream and the
# partial file is deleted — the broker never holds more than one chunk in RAM).
VIDEO_UPLOAD_MAX_BYTES = int(os.environ.get("DEMO_VIDEO_MAX_BYTES", str(1 * 1024**3)))  # 1 GB
# Inline (single-request) splat cap. NOT a statement about how big a splat may be
# — see SPLAT_CHUNKED_MAX_BYTES — but about what fits in one 150 s Modal request.
SPLAT_UPLOAD_MAX_BYTES = splat_import.INLINE_MAX_BYTES
SPLAT_CHUNKED_MAX_BYTES = splat_import.CHUNKED_MAX_BYTES
SPLAT_CHUNK_BYTES = splat_import.CHUNK_BYTES
SPLAT_MAX_GAUSSIANS = splat_import.MAX_GAUSSIANS
_STREAM_CHUNK = 4 * 1024 * 1024  # 4 MB read granularity

# Staged chunked uploads older than this are swept (a browser tab closed
# mid-upload must not pin GBs on the scenes volume forever).
SPLAT_UPLOAD_TTL_S = int(os.environ.get("DEMO_SPLAT_UPLOAD_TTL_S", str(6 * 3600)))

# ``X-Demo-Source`` header → persisted ``source`` stamp. Only known capture
# provenances are accepted (a caller must not stamp arbitrary provenance strings).
# X-Demo-Source -> persisted scene source. Every one of these runs the SAME RGB pipeline;
# the value records WHICH CAMERA the footage came off, which is a provenance claim and so
# must not be guessed. "depthcam" exists for depth cameras that are not a Gemini 2 (a
# Kinect, an Xtion, a public RGB-D dataset) — labelling those "gemini2" would name hardware
# that never recorded them.
_VIDEO_SOURCES = {
    None: "recon_video",
    "": "recon_video",
    "gemini2": "recon_gemini2",
    "depthcam": "recon_depthcam",
}

# Upload staging area on the scenes volume — one layout shared with
# modal_oreos_ingest._save_upload / demo_recon_job's containment check.
_UPLOADS_DIRNAME = "_uploads"

# Filename sanitation identical to modal_oreos_ingest._save_upload so the two
# staging writers can never diverge on layout.
_FNAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_filename(value: str, fallback: str) -> str:
    name = os.path.basename(str(value or ""))
    name = _FNAME_UNSAFE.sub("_", name)[:128]
    return name or fallback


# ---------------------------------------------------------------------------
# recon-job spawner (injectable so GPU-free tests never import modal)
# ---------------------------------------------------------------------------

_recon_spawner: Optional[Callable[..., Any]] = None


def configure_recon_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    """Inject the job spawner (tests pass a recording fake; ``None`` restores the
    real ``modal.Function.from_name`` lookup)."""
    global _recon_spawner
    _recon_spawner = spawner


def _spawn_recon_job(**kwargs: Any) -> None:
    """Spawn ``demo_recon_job`` on the deployed app (fire-and-forget; status flows
    through the demo-jobs Dict). Lazy modal import — the broker container has modal
    by construction; tests inject a fake via :func:`configure_recon_spawner`."""
    if _recon_spawner is not None:
        _recon_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("vggt-slam-streaming", "demo_recon_job").spawn(**kwargs)


# ---------------------------------------------------------------------------
# shared streaming helper
# ---------------------------------------------------------------------------


class _TooLarge(Exception):
    pass


def _stream_request_to(path: str, max_bytes: int) -> int:
    """Stream the raw request body to ``path`` in chunks, enforcing ``max_bytes``
    WHILE reading. Returns the byte count; raises ``_TooLarge`` (caller deletes the
    partial file and answers 413). A declared oversized ``Content-Length`` short-
    circuits before any read."""
    declared = request.content_length
    if declared is not None and declared > max_bytes:
        raise _TooLarge()
    total = 0
    with open(path, "wb") as fh:
        while True:
            chunk = request.stream.read(_STREAM_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise _TooLarge()
            fh.write(chunk)
    return total


def _publish_job(record: dict[str, Any]) -> bool:
    """Best-effort write into the demo-jobs store (modal.Dict on the broker; a plain
    dict in tests). Job status is telemetry — a store hiccup must not fail the 202."""
    store = demo_jobs._get_jobs_store()
    if store is None:
        return False
    try:
        store[record["job_id"]] = record
        return True
    except Exception as exc:
        print(f"[demo.ingest] job-status publish failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/video
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/workspace/ingest/video", methods=["POST"])
def ingest_video():
    """Raw video stream → scenes-volume ``_uploads/`` staging → ``demo_recon_job``
    spawn → ``202 {job_id, scan_id}`` (poll ``GET /api/workspace/jobs/<job_id>``)."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    filename = request.headers.get("X-Upload-Filename")
    if not filename:
        return jsonify({"error": "missing_filename",
                        "detail": "set X-Upload-Filename to the video's filename"}), 400
    suffix = os.path.splitext(str(filename))[1].lower()
    if suffix not in DEMO_VIDEO_SUFFIXES:
        return jsonify({
            "error": "unsupported_video_type",
            "suffix": suffix,
            "allowed": sorted(DEMO_VIDEO_SUFFIXES),
        }), 400

    demo_source = (request.headers.get("X-Demo-Source") or "").strip().lower() or None
    if demo_source not in _VIDEO_SOURCES:
        return jsonify({"error": "unsupported_source",
                        "allowed": ["gemini2", "depthcam"]}), 400
    source = _VIDEO_SOURCES[demo_source]
    label = request.headers.get("X-Scene-Label") or None

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503

    # Stage under the scenes volume root the persistence is mounted on. Reaching the
    # store's private ``_blob_root``/``_commit_blobs`` keeps the layout + commit
    # semantics owned by ONE class (store.py, W0) instead of duplicating volume
    # wiring here; the demo blueprint is same-repo code (cf. app.py's own use of
    # persistence internals' conventions).
    from server.scene_report.store import _safe_component

    upload_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    scan_id = uuid.uuid4().hex
    fname = _safe_filename(filename, fallback="upload.mp4")
    rel = "/".join([_safe_component(user_id), _UPLOADS_DIRNAME, upload_id, fname])
    dest = os.path.join(persistence._blob_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        total = _stream_request_to(dest, VIDEO_UPLOAD_MAX_BYTES)
    except _TooLarge:
        _cleanup_upload_dir(dest)
        return jsonify({"error": "too_large",
                        "limit_bytes": VIDEO_UPLOAD_MAX_BYTES}), 413
    except Exception as exc:
        print(f"[demo.ingest] upload stream failed: {exc}")
        _cleanup_upload_dir(dest)
        return jsonify({"error": "upload_failed"}), 500
    if total == 0:
        _cleanup_upload_dir(dest)
        return jsonify({"error": "empty_upload"}), 400
    persistence._commit_blobs()

    # Queued marker BEFORE the spawn so the jobs route answers from the moment the
    # client sees the 202 (mirrors modal_oreos_ingest.ingest's ordering).
    _publish_job(demo_jobs.job_status_record(
        job_id, user_id, status="queued", stage="upload", scan_id=scan_id,
        upload_bytes=total,
    ))

    try:
        _spawn_recon_job(
            job_id=job_id,
            user_id=user_id,
            upload_rel_path=rel,
            scan_id=scan_id,
            label=label,
            source=source,
        )
    except Exception as exc:
        print(f"[demo.ingest] demo_recon_job spawn failed: {exc}")
        _cleanup_upload_dir(dest)
        persistence._commit_blobs()
        _publish_job(demo_jobs.job_status_record(
            job_id, user_id, status="failed", stage="upload", scan_id=scan_id,
            error="spawn_failed",
        ))
        return jsonify({"error": "spawn_failed"}), 502

    return jsonify({"job_id": job_id, "scan_id": scan_id}), 202


def _cleanup_upload_dir(dest: str) -> None:
    """Remove a staged upload file + its per-upload dir (best-effort)."""
    try:
        if os.path.isfile(dest):
            os.remove(dest)
        parent = os.path.dirname(dest)
        if os.path.basename(os.path.dirname(parent)) == _UPLOADS_DIRNAME:
            os.rmdir(parent)  # only ever the empty per-upload_id dir
    except OSError:
        pass


# ---------------------------------------------------------------------------
# splat-import job spawner (injectable so GPU-free tests never import modal)
# ---------------------------------------------------------------------------

_splat_spawner: Optional[Callable[..., Any]] = None


def configure_splat_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    """Inject the splat-import job spawner (tests pass a recording fake; ``None``
    restores the real ``modal.Function.from_name`` lookup)."""
    global _splat_spawner
    _splat_spawner = spawner


def _spawn_splat_job(**kwargs: Any) -> None:
    """Spawn ``demo_splat_import_job`` on the deployed app (fire-and-forget;
    status flows through the demo-jobs Dict)."""
    if _splat_spawner is not None:
        _splat_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("vggt-slam-streaming", "demo_splat_import_job").spawn(**kwargs)


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/splat  (inline fast path, small files)
# ---------------------------------------------------------------------------


def _splat_suffix_or_error(filename: Optional[str]):
    """(suffix, None) or (None, (json, status)) — the shared filename gate."""
    if not filename:
        return None, (jsonify({
            "error": "missing_filename",
            "detail": "set X-Upload-Filename to the splat's filename",
        }), 400)
    suffix = os.path.splitext(str(filename))[1].lower()
    if suffix not in SPLAT_SUFFIXES:
        return None, (jsonify({
            "error": "unsupported_splat_type",
            "suffix": suffix,
            "allowed": list(SPLAT_SUFFIXES),
            "detail": (
                f"'{suffix or filename}' is not a gaussian-splat file. Accepted: "
                ".ply (3DGS binary little-endian) and .spz."
            ),
        }), 400)
    return suffix, None


def _rejection_response(exc: splat_import.SplatRejected):
    return jsonify(exc.payload()), exc.status


@oreos_bp.route("/api/workspace/ingest/splat", methods=["POST"])
def ingest_splat():
    """Small gaussian splat → persisted scene in one request.

    ``201 {scan_id, gaussian_count}`` on success; honest 4xx otherwise. Anything
    over ``SPLAT_UPLOAD_MAX_BYTES`` is 413'd **with instructions** rather than
    allowed to run into Modal's 150 s request ceiling and die as a timeout."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    filename = request.headers.get("X-Upload-Filename")
    suffix, error = _splat_suffix_or_error(filename)
    if error:
        return error

    # Refuse an oversized body BEFORE reading it, pointing at the route that can
    # actually carry it. This is the founder's failure mode, made self-explaining.
    declared = request.content_length
    if declared is not None and declared > SPLAT_UPLOAD_MAX_BYTES:
        return _too_large_for_inline(declared)

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503

    # Stream to a broker-local temp file first (never in RAM); all validation reads
    # from disk. The temp file is always deleted, success or not.
    fd, tmp_path = tempfile.mkstemp(prefix="demo_splat_", suffix=suffix)
    os.close(fd)
    ply_path = tmp_path
    converted: Optional[str] = None
    try:
        try:
            total = _stream_request_to(tmp_path, SPLAT_UPLOAD_MAX_BYTES)
        except _TooLarge:
            return _too_large_for_inline(declared)
        if total == 0:
            return jsonify({"error": "empty_upload",
                            "detail": "the request body was empty"}), 400
        if suffix == ".spz":
            ply_path = converted = tmp_path + ".ply"
            try:
                _convert_spz(tmp_path, ply_path)
            except splat_import.SplatRejected as exc:
                return _rejection_response(exc)
        try:
            result = splat_import.import_splat(
                persistence, user_id, ply_path, filename,
                label=request.headers.get("X-Scene-Label") or None,
                max_gaussians=SPLAT_MAX_GAUSSIANS,
            )
        except splat_import.SplatRejected as exc:
            return _rejection_response(exc)
        return jsonify(result), 201
    except Exception as exc:
        print(f"[demo.ingest] splat import failed: {exc}")
        return jsonify({"error": "splat_import_failed",
                        "detail": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        for path in (tmp_path, converted):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def _too_large_for_inline(declared: Optional[int]):
    """The honest 413: say how big it is, what the one-shot limit is, and which
    route to use instead. Never a bare 'too_large'."""
    size_note = f"{declared / 1024**2:.0f} MB " if declared else ""
    return jsonify({
        "error": "too_large",
        "limit_bytes": SPLAT_UPLOAD_MAX_BYTES,
        "max_bytes": SPLAT_CHUNKED_MAX_BYTES,
        "use_chunked_upload": True,
        "chunked_init_path": "/api/workspace/ingest/splat/init",
        "detail": (
            f"this splat ({size_note}) is over the {SPLAT_UPLOAD_MAX_BYTES // 1024**2} MB "
            "single-request limit — one upload has to finish inside the platform's "
            "150-second request timeout. Use the chunked import "
            "(POST /api/workspace/ingest/splat/init), which has no time limit and "
            f"accepts up to {SPLAT_CHUNKED_MAX_BYTES // 1024**3} GB."
        ),
    }), 413


def _convert_spz(spz_path: str, ply_path: str) -> dict[str, Any]:
    """Decode a staged ``.spz`` into a 3DGS ``.ply`` in place, translating
    decoder errors into the same honest rejections the PLY path raises."""
    from server.oreos import spz as spz_mod

    try:
        return spz_mod.spz_to_ply_file(spz_path, ply_path, max_gaussians=SPLAT_MAX_GAUSSIANS)
    except spz_mod.SpzError as exc:
        message = str(exc)
        if "import limit" in message:
            raise splat_import.SplatRejected(
                "too_many_gaussians", 422,
                f"{message}. Decimate the splat before importing.",
                limit=SPLAT_MAX_GAUSSIANS,
            ) from exc
        raise splat_import.SplatRejected(
            "not_a_gaussian_splat", 422,
            f"this .spz could not be decoded: {message}",
        ) from exc


# ---------------------------------------------------------------------------
# Chunked splat import: init → chunk × N → finalize
#
# In-process staging state. Safe because the broker runs max_containers=1
# (modal_streaming.py) — every chunk of an upload lands in the same container.
# If that container recycles mid-upload the state is gone and the next chunk
# 404s with an explicit "the server restarted, please start the upload again",
# which is a far better outcome than silently writing into a file nobody will
# finalize.
# ---------------------------------------------------------------------------

_splat_uploads: dict[str, dict[str, Any]] = {}
_splat_uploads_lock = threading.Lock()


def _sweep_stale_uploads(persistence: Any) -> None:
    """Drop staging state + bytes for uploads abandoned longer than the TTL."""
    cutoff = time.time() - SPLAT_UPLOAD_TTL_S
    with _splat_uploads_lock:
        stale = [k for k, v in _splat_uploads.items() if v["created_at"] < cutoff]
        records = [_splat_uploads.pop(k) for k in stale]
    for record in records:
        _discard_upload_dir(record["dest"])
    if records:
        print(f"[demo.ingest] swept {len(records)} stale splat upload(s)")


def _discard_upload_dir(dest: str) -> None:
    """Remove a staged upload's whole ``<upload_id>`` dir (best-effort)."""
    parent = os.path.dirname(dest)
    if os.path.basename(os.path.dirname(parent)) == _UPLOADS_DIRNAME:
        shutil.rmtree(parent, ignore_errors=True)


def _owned_upload(upload_id: str, user_id: str):
    """(record, None) or (None, 404 envelope). Another user's upload and an
    unknown id are indistinguishable from outside."""
    with _splat_uploads_lock:
        record = _splat_uploads.get(str(upload_id))
    if record is None or record["user_id"] != user_id:
        return None, (jsonify({
            "error": "upload_not_found",
            "upload_id": str(upload_id),
            "detail": (
                "no staged upload with that id — it either finished, was cancelled, "
                "expired, or the server restarted mid-upload. Start the import again."
            ),
        }), 404)
    return record, None


@oreos_bp.route("/api/workspace/ingest/splat/init", methods=["POST"])
def ingest_splat_init():
    """Open a chunked splat upload.

    Headers: ``X-Upload-Filename`` (.ply/.spz), ``X-Upload-Bytes`` (total size,
    required — it lets us refuse an impossible upload before a single byte
    moves), optional ``X-Scene-Label``. → ``200 {upload_id, chunk_bytes,
    total_chunks, max_bytes, max_gaussians}``."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    filename = request.headers.get("X-Upload-Filename")
    suffix, error = _splat_suffix_or_error(filename)
    if error:
        return error

    raw_total = request.headers.get("X-Upload-Bytes")
    try:
        total_bytes = int(str(raw_total))
    except (TypeError, ValueError):
        return jsonify({
            "error": "missing_upload_bytes",
            "detail": "set X-Upload-Bytes to the splat's total size in bytes",
        }), 400
    if total_bytes <= 0:
        return jsonify({"error": "empty_upload",
                        "detail": "X-Upload-Bytes must be positive"}), 400
    if total_bytes > SPLAT_CHUNKED_MAX_BYTES:
        return jsonify({
            "error": "too_large",
            "limit_bytes": SPLAT_CHUNKED_MAX_BYTES,
            "upload_bytes": total_bytes,
            "detail": (
                f"this splat is {total_bytes / 1024**3:.1f} GB — over the "
                f"{SPLAT_CHUNKED_MAX_BYTES // 1024**3} GB import limit. Decimate it "
                "first; the demo viewer cannot render a splat that size anyway."
            ),
        }), 413

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503
    _sweep_stale_uploads(persistence)

    from server.scene_report.store import _safe_component

    upload_id = uuid.uuid4().hex
    fname = _safe_filename(filename, fallback=f"splat{suffix}")
    rel = "/".join([_safe_component(user_id), _UPLOADS_DIRNAME, upload_id, fname])
    dest = os.path.join(persistence._blob_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # Pre-size the file so chunks can be seek-written in any order.
        with open(dest, "wb") as fh:
            fh.truncate(total_bytes)
    except OSError as exc:
        print(f"[demo.ingest] could not stage splat upload: {exc}")
        _discard_upload_dir(dest)
        return jsonify({"error": "staging_failed",
                        "detail": f"could not reserve space on the scenes volume: {exc}"}), 507

    chunk_bytes = SPLAT_CHUNK_BYTES
    total_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes
    with _splat_uploads_lock:
        _splat_uploads[upload_id] = {
            "upload_id": upload_id,
            "user_id": user_id,
            "filename": str(filename),
            "suffix": suffix,
            "label": request.headers.get("X-Scene-Label") or None,
            "total_bytes": total_bytes,
            "chunk_bytes": chunk_bytes,
            "total_chunks": int(total_chunks),
            "received": set(),
            "rel": rel,
            "dest": dest,
            "created_at": time.time(),
        }
    print(f"[demo.ingest] chunked splat upload {upload_id} opened for {user_id}: "
          f"{filename!r} {total_bytes} bytes in {total_chunks} chunks")
    return jsonify({
        "upload_id": upload_id,
        "chunk_bytes": chunk_bytes,
        "total_chunks": int(total_chunks),
        "max_bytes": SPLAT_CHUNKED_MAX_BYTES,
        "max_gaussians": SPLAT_MAX_GAUSSIANS,
    }), 200


@oreos_bp.route("/api/workspace/ingest/splat/<upload_id>/chunk/<int:index>", methods=["POST"])
def ingest_splat_chunk(upload_id: str, index: int):
    """Accept one chunk, seek-written at ``index * chunk_bytes``. Idempotent: a
    retry of the same index simply rewrites the same range."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    record, error = _owned_upload(upload_id, user_id)
    if error:
        return error

    total_chunks = record["total_chunks"]
    if index < 0 or index >= total_chunks:
        return jsonify({
            "error": "bad_chunk_index",
            "index": index, "total_chunks": total_chunks,
            "detail": f"chunk index must be in [0, {total_chunks - 1}]",
        }), 400

    offset = index * record["chunk_bytes"]
    expected = min(record["chunk_bytes"], record["total_bytes"] - offset)
    body = request.get_data(cache=False)
    if len(body) != expected:
        return jsonify({
            "error": "chunk_size_mismatch",
            "index": index, "expected_bytes": int(expected), "received_bytes": len(body),
            "detail": (
                f"chunk {index} should be {expected} bytes but {len(body)} arrived — "
                "the upload was cut short or the chunk size does not match the one "
                "returned by init."
            ),
        }), 400

    try:
        with open(record["dest"], "r+b") as fh:
            fh.seek(offset)
            fh.write(body)
    except OSError as exc:
        print(f"[demo.ingest] chunk write failed ({upload_id} #{index}): {exc}")
        return jsonify({"error": "chunk_write_failed",
                        "detail": f"could not stage chunk {index}: {exc}"}), 500

    with _splat_uploads_lock:
        live = _splat_uploads.get(upload_id)
        if live is None:  # cancelled/swept while this chunk was in flight
            return jsonify({
                "error": "upload_not_found", "upload_id": upload_id,
                "detail": "the upload was cancelled while this chunk was in flight",
            }), 404
        live["received"].add(index)
        received = len(live["received"])
    return jsonify({
        "upload_id": upload_id,
        "index": index,
        "chunks_received": received,
        "total_chunks": total_chunks,
        "received_bytes": int(min(received * record["chunk_bytes"], record["total_bytes"])),
    }), 200


@oreos_bp.route("/api/workspace/ingest/splat/<upload_id>/finalize", methods=["POST"])
def ingest_splat_finalize(upload_id: str):
    """Close a chunked upload: verify completeness, validate the header on the
    broker (cheap), spawn the import job → ``202 {job_id, scan_id}``."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    record, error = _owned_upload(upload_id, user_id)
    if error:
        return error

    missing = sorted(set(range(record["total_chunks"])) - record["received"])
    if missing:
        return jsonify({
            "error": "incomplete_upload",
            "missing_chunks": missing[:64],
            "missing_count": len(missing),
            "total_chunks": record["total_chunks"],
            "detail": (
                f"{len(missing)} of {record['total_chunks']} chunks never arrived — "
                "re-send the listed indices, then finalize again."
            ),
        }), 409

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503

    dest = record["dest"]
    actual = os.path.getsize(dest) if os.path.exists(dest) else 0
    if actual != record["total_bytes"]:
        _drop_upload(upload_id)
        return jsonify({
            "error": "size_mismatch",
            "expected_bytes": record["total_bytes"], "actual_bytes": int(actual),
            "detail": ("the staged file is not the size announced at init — the upload "
                       "was corrupted. Please start it again."),
        }), 409

    # Cheap header gate on the broker: reject a bad file for FREE instead of
    # paying for a Modal job that would only fail.
    try:
        if record["suffix"] == ".spz":
            from server.oreos import spz as spz_mod

            try:
                header = spz_mod.read_spz_header(dest)
            except spz_mod.SpzError as exc:
                raise splat_import.SplatRejected(
                    "not_a_gaussian_splat", 422,
                    f"this .spz could not be read: {exc}",
                ) from exc
            gaussian_count = header.num_splats
            if gaussian_count > SPLAT_MAX_GAUSSIANS:
                raise splat_import.SplatRejected(
                    "too_many_gaussians", 422,
                    f"{gaussian_count:,} gaussians is over the {SPLAT_MAX_GAUSSIANS:,} "
                    "import limit. Decimate the splat before importing.",
                    gaussian_count=int(gaussian_count), limit=SPLAT_MAX_GAUSSIANS,
                )
        else:
            gaussian_count = splat_import.inspect_splat(
                dest, max_gaussians=SPLAT_MAX_GAUSSIANS
            ).count
    except splat_import.SplatRejected as exc:
        _drop_upload(upload_id)
        return _rejection_response(exc)

    persistence._commit_blobs()

    job_id = uuid.uuid4().hex
    scan_id = uuid.uuid4().hex
    _publish_job(demo_jobs.job_status_record(
        job_id, user_id, status="queued", stage="upload", scan_id=scan_id,
        kind="splat_import", upload_bytes=record["total_bytes"],
        gaussian_count=int(gaussian_count),
    ))
    try:
        _spawn_splat_job(
            job_id=job_id,
            user_id=user_id,
            upload_rel_path=record["rel"],
            scan_id=scan_id,
            label=record["label"],
            filename=record["filename"],
        )
    except Exception as exc:
        print(f"[demo.ingest] demo_splat_import_job spawn failed: {exc}")
        _drop_upload(upload_id)
        _publish_job(demo_jobs.job_status_record(
            job_id, user_id, status="failed", stage="upload", scan_id=scan_id,
            kind="splat_import", error="spawn_failed",
        ))
        return jsonify({"error": "spawn_failed",
                        "detail": "the import worker could not be started"}), 502

    # State only; the bytes now belong to the job, which deletes them when done.
    with _splat_uploads_lock:
        _splat_uploads.pop(upload_id, None)
    advisory = splat_import.render_advisory(int(gaussian_count))
    payload = {"job_id": job_id, "scan_id": scan_id, "gaussian_count": int(gaussian_count)}
    if advisory:
        payload["render_advisory"] = advisory
    return jsonify(payload), 202


@oreos_bp.route("/api/workspace/ingest/splat/<upload_id>", methods=["DELETE"])
def ingest_splat_cancel(upload_id: str):
    """Cancel a staged upload and free its bytes (the browser's beforeunload /
    an explicit cancel button)."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    record, error = _owned_upload(upload_id, user_id)
    if error:
        return error
    _drop_upload(upload_id)
    return jsonify({"upload_id": upload_id, "cancelled": True}), 200


def _drop_upload(upload_id: str) -> None:
    """Forget a staged upload and delete its bytes."""
    with _splat_uploads_lock:
        record = _splat_uploads.pop(str(upload_id), None)
    if record:
        _discard_upload_dir(record["dest"])

# ---------------------------------------------------------------------------
# GET /api/scenes/<scan_id>/demo/trajectory.npz
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/scenes/<scan_id>/demo/trajectory.npz", methods=["GET"])
def get_scene_trajectory_npz(scan_id):
    """Stream the scan's ORIGINAL per-keyframe trajectory block
    (``{poses (N,4,4), intrinsics (N,4), source_frame_id (N,)}``) re-serialized
    as an ``.npz`` — read-only, owner-authed.

    Consumer: ``scripts/demo_ingest_gemini.py anchor`` (design/shell.md §4c
    steps 3–5) needs ``source_frame_id`` to bridge sampled keyframes back to
    the capture's depth PNGs via ``frame_map.json``. No pre-existing route
    serves the scene-root ``trajectory.npz`` (the derived GET route serves
    only ``derived/…`` keys, and the export zip carries it only inside a
    pandas-parquet tree), so this is the minimal additive read path. World
    frame, OpenCV c2w, SLAM units — per WORLD-TRANSFORM-CONTRACT."""
    import io as _io

    import numpy as np
    from flask import send_file

    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503
    record = persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    traj = persistence.get_trajectory(user_id, scan_id)
    if traj is None:
        return jsonify({
            "error": "no_trajectory",
            "detail": ("scan has no persisted per-keyframe trajectory "
                       "(pilot-persisted scenes predate --persist-trajectory; "
                       "demo ingests persist it by default)"),
        }), 404
    buf = _io.BytesIO()
    np.savez_compressed(
        buf,
        poses=np.asarray(traj["poses"], dtype=np.float32),
        intrinsics=np.asarray(traj["intrinsics"], dtype=np.float32),
        source_frame_id=np.asarray(traj["source_frame_id"], dtype=np.float32),
    )
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=False,
        download_name=f"{scan_id}-trajectory.npz",
    )
