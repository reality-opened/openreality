"""LiDAR capture-session ingest — an iPhone uploads a zipped ``CaptureSession``.

``POST /api/workspace/ingest/capture/init`` -> ``/chunk/<index>`` -> ``/finalize``:
the same staged-chunked-upload MECHANICS as the splat lane's large-file path
(``server/oreos/routes_ingest.py`` — pre-sized file on the scenes volume,
seek-written chunks so they are order-independent and safely retryable, in-process
staging state because the broker runs ``max_containers=1``), but with a DIFFERENT
wire shape: this is a fixed contract the iOS client is built against in parallel
(see the module docstring deviations note below), so the paths/headers/response
keys here are NOT the splat lane's — only the underlying streaming/staging
mechanics are borrowed.

Wire contract (fixed):
  ``POST /api/workspace/ingest/capture/init``
      body (JSON) ``{"filename": "<session_id>.zip", "size_bytes": N}``
      -> ``200 {"upload_id", "chunk_bytes", "total_chunks", "max_bytes"}``
  ``POST /api/workspace/ingest/capture/chunk/<index>``
      header ``X-Upload-Id``; raw ``application/octet-stream`` body (one chunk)
      -> ``200 {"upload_id", "index", "chunks_received", "total_chunks",
                "received_bytes"}``
  ``POST /api/workspace/ingest/capture/finalize``
      header ``X-Upload-Id``
      -> ``202 {"job_id", "scan_id"}`` (poll ``GET /api/workspace/jobs/<job_id>``)

DEVIATIONS from the splat lane's mechanics (flagged per the build brief):
  - splat's ``init`` reads ``X-Upload-Filename``/``X-Upload-Bytes`` HEADERS; this
    lane's ``init`` reads a JSON BODY (``filename``/``size_bytes``) instead — the
    wire contract given to the iOS client specifies a JSON body explicitly.
  - splat's ``chunk``/``finalize`` carry ``upload_id`` as a URL PATH SEGMENT
    (``/splat/<upload_id>/chunk/<index>``); this lane carries it as the
    ``X-Upload-Id`` HEADER instead, with ``upload_id`` absent from the URL
    (``/capture/chunk/<index>``, ``/capture/finalize``) — again per the fixed
    contract.
  Everything else — pre-sized file + seek-written chunks, in-process
  ``_capture_uploads`` dict guarded by a lock, ``X-Upload-Id`` ownership check,
  stale-upload TTL sweep, cheap validation on the broker before a job is spawned —
  mirrors the splat lane's chunked path exactly.

``finalize`` validates the staged zip with
:func:`server.oreos.capture.zip_validate.validate_capture_zip` (required entries:
``odometry.csv``, ``rgb.mp4``, ``depth/``, ``confidence/``; zip-slip guard) —
cheap (central-directory read only), so a bad zip costs zero Modal spend and
answers a clear ``400`` instead of failing inside the job. On success it spawns
the standalone ``lidar-recon`` Modal app's ``lidar_recon_job``
(``server/oreos/capture/modal_recon.py``), which runs VGGT-SLAM with the LiDAR
metric anchor and persists a ``source="recon_lidar"`` scene.

Auth: owner-only via ``auth_user_id()`` (see ``server/oreos/__init__``).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import jobs as demo_jobs
from server.oreos.capture.zip_validate import CaptureZipRejected, validate_capture_zip
from server.oreos.routes_ingest import (
    _UPLOADS_DIRNAME,
    _publish_job,
    _safe_filename,
)

CAPTURE_SUFFIX = ".zip"

# Chunked-upload sizing. A capture session is dominated by rgb.mp4 + depth/confidence
# PNGs; 6 GiB mirrors the splat lane's chunked ceiling (server/oreos/splat_import.py),
# comfortably above any realistic handheld walk.
CAPTURE_CHUNK_BYTES = int(os.environ.get("CAPTURE_CHUNK_BYTES", str(16 * 1024**2)))
CAPTURE_UPLOAD_MAX_BYTES = int(os.environ.get("CAPTURE_UPLOAD_MAX_BYTES", str(6 * 1024**3)))

# Staged chunked uploads older than this are swept (a closed app mid-upload must not
# pin GBs on the scenes volume forever) — same doctrine as the splat lane.
CAPTURE_UPLOAD_TTL_S = int(os.environ.get("CAPTURE_UPLOAD_TTL_S", str(6 * 3600)))

# ---------------------------------------------------------------------------
# recon-job spawner (injectable so GPU-free tests never import modal)
# ---------------------------------------------------------------------------

_capture_spawner: Optional[Callable[..., Any]] = None


def configure_capture_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    """Inject the ``lidar_recon_job`` spawner (tests pass a recording fake;
    ``None`` restores the real ``modal.Function.from_name`` lookup)."""
    global _capture_spawner
    _capture_spawner = spawner


def _spawn_capture_job(**kwargs: Any) -> None:
    if _capture_spawner is not None:
        _capture_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("lidar-recon", "lidar_recon_job").spawn(**kwargs)


# ---------------------------------------------------------------------------
# In-process staging state (routes_ingest's chunked-splat pattern, own dict).
# Safe because the broker runs max_containers=1 — every chunk of an upload lands
# in the same container. A container recycle mid-upload loses the state; the next
# chunk/finalize 404s with an explicit "start again" rather than silently orphaning
# bytes nobody will finalize.
# ---------------------------------------------------------------------------

_capture_uploads: dict[str, dict[str, Any]] = {}
_capture_uploads_lock = threading.Lock()


def _discard_upload_dir(dest: str) -> None:
    parent = os.path.dirname(dest)
    if os.path.basename(os.path.dirname(parent)) == _UPLOADS_DIRNAME:
        import shutil

        shutil.rmtree(parent, ignore_errors=True)


def _drop_upload(upload_id: str) -> None:
    with _capture_uploads_lock:
        record = _capture_uploads.pop(str(upload_id), None)
    if record:
        _discard_upload_dir(record["dest"])


def _sweep_stale_uploads() -> None:
    cutoff = time.time() - CAPTURE_UPLOAD_TTL_S
    with _capture_uploads_lock:
        stale = [k for k, v in _capture_uploads.items() if v["created_at"] < cutoff]
        records = [_capture_uploads.pop(k) for k in stale]
    for record in records:
        _discard_upload_dir(record["dest"])
    if records:
        print(f"[oreos.capture] swept {len(records)} stale capture upload(s)")


def _owned_capture_upload(upload_id: str, user_id: str):
    """(record, None) or (None, 404 envelope). Another user's upload and an
    unknown id are indistinguishable from outside."""
    with _capture_uploads_lock:
        record = _capture_uploads.get(str(upload_id))
    if record is None or record["user_id"] != user_id:
        return None, (
            jsonify(
                {
                    "error": "upload_not_found",
                    "upload_id": str(upload_id),
                    "detail": (
                        "no staged capture upload with that id — it either finished, "
                        "was cancelled, expired, or the server restarted mid-upload. "
                        "Start the upload again."
                    ),
                }
            ),
            404,
        )
    return record, None


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/capture/init
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/workspace/ingest/capture/init", methods=["POST"])
def ingest_capture_init():
    """Open a chunked capture-session-zip upload.

    Body (JSON): ``{"filename": "<session_id>.zip", "size_bytes": N}`` ->
    ``200 {upload_id, chunk_bytes, total_chunks, max_bytes}``."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    body = request.get_json(silent=True) or {}
    filename = body.get("filename")
    if not filename or not str(filename).lower().endswith(CAPTURE_SUFFIX):
        return (
            jsonify(
                {
                    "error": "missing_filename",
                    "detail": (
                        "set \"filename\" to the capture session's zip name "
                        f"(must end in {CAPTURE_SUFFIX!r})"
                    ),
                }
            ),
            400,
        )

    raw_size = body.get("size_bytes")
    try:
        total_bytes = int(raw_size)
    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "error": "missing_size_bytes",
                    "detail": "set \"size_bytes\" to the zip's total size in bytes",
                }
            ),
            400,
        )
    if total_bytes <= 0:
        return jsonify({"error": "empty_upload", "detail": "size_bytes must be positive"}), 400
    if total_bytes > CAPTURE_UPLOAD_MAX_BYTES:
        return (
            jsonify(
                {
                    "error": "too_large",
                    "limit_bytes": CAPTURE_UPLOAD_MAX_BYTES,
                    "upload_bytes": total_bytes,
                    "detail": (
                        f"this capture session is {total_bytes / 1024**3:.1f} GB — over "
                        f"the {CAPTURE_UPLOAD_MAX_BYTES // 1024**3} GB import limit."
                    ),
                }
            ),
            413,
        )

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503
    _sweep_stale_uploads()

    from server.scene_report.store import _safe_component

    upload_id = uuid.uuid4().hex
    fname = _safe_filename(filename, fallback=f"session{CAPTURE_SUFFIX}")
    rel = "/".join([_safe_component(user_id), _UPLOADS_DIRNAME, upload_id, fname])
    dest = os.path.join(persistence._blob_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # Pre-size the file so chunks can be seek-written in any order.
        with open(dest, "wb") as fh:
            fh.truncate(total_bytes)
    except OSError as exc:
        print(f"[oreos.capture] could not stage capture upload: {exc}")
        _discard_upload_dir(dest)
        return (
            jsonify(
                {
                    "error": "staging_failed",
                    "detail": f"could not reserve space on the scenes volume: {exc}",
                }
            ),
            507,
        )

    chunk_bytes = CAPTURE_CHUNK_BYTES
    total_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes
    with _capture_uploads_lock:
        _capture_uploads[upload_id] = {
            "upload_id": upload_id,
            "user_id": user_id,
            "filename": str(filename),
            "total_bytes": total_bytes,
            "chunk_bytes": chunk_bytes,
            "total_chunks": int(total_chunks),
            "received": set(),
            "rel": rel,
            "dest": dest,
            "created_at": time.time(),
        }
    print(
        f"[oreos.capture] chunked capture upload {upload_id} opened for {user_id}: "
        f"{filename!r} {total_bytes} bytes in {total_chunks} chunks"
    )
    return (
        jsonify(
            {
                "upload_id": upload_id,
                "chunk_bytes": chunk_bytes,
                "total_chunks": int(total_chunks),
                "max_bytes": CAPTURE_UPLOAD_MAX_BYTES,
            }
        ),
        200,
    )


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/capture/chunk/<index>
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/workspace/ingest/capture/chunk/<int:index>", methods=["POST"])
def ingest_capture_chunk(index: int):
    """Accept one chunk, seek-written at ``index * chunk_bytes``. Idempotent: a
    retry of the same index simply rewrites the same range. Upload identified by
    the ``X-Upload-Id`` header (NOT a URL path segment — see the wire-contract
    deviation note in the module docstring)."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    upload_id = request.headers.get("X-Upload-Id")
    if not upload_id:
        return (
            jsonify(
                {"error": "missing_upload_id", "detail": "set the X-Upload-Id header"}
            ),
            400,
        )
    record, error = _owned_capture_upload(upload_id, user_id)
    if error:
        return error

    total_chunks = record["total_chunks"]
    if index < 0 or index >= total_chunks:
        return (
            jsonify(
                {
                    "error": "bad_chunk_index",
                    "index": index,
                    "total_chunks": total_chunks,
                    "detail": f"chunk index must be in [0, {total_chunks - 1}]",
                }
            ),
            400,
        )

    offset = index * record["chunk_bytes"]
    expected = min(record["chunk_bytes"], record["total_bytes"] - offset)
    body = request.get_data(cache=False)
    if len(body) != expected:
        return (
            jsonify(
                {
                    "error": "chunk_size_mismatch",
                    "index": index,
                    "expected_bytes": int(expected),
                    "received_bytes": len(body),
                    "detail": (
                        f"chunk {index} should be {expected} bytes but {len(body)} "
                        "arrived — the upload was cut short or the chunk size does "
                        "not match the one returned by init."
                    ),
                }
            ),
            400,
        )

    try:
        with open(record["dest"], "r+b") as fh:
            fh.seek(offset)
            fh.write(body)
    except OSError as exc:
        print(f"[oreos.capture] chunk write failed ({upload_id} #{index}): {exc}")
        return (
            jsonify(
                {"error": "chunk_write_failed", "detail": f"could not stage chunk {index}: {exc}"}
            ),
            500,
        )

    with _capture_uploads_lock:
        live = _capture_uploads.get(upload_id)
        if live is None:  # cancelled/swept while this chunk was in flight
            return (
                jsonify(
                    {
                        "error": "upload_not_found",
                        "upload_id": upload_id,
                        "detail": "the upload was cancelled while this chunk was in flight",
                    }
                ),
                404,
            )
        live["received"].add(index)
        received = len(live["received"])
    return (
        jsonify(
            {
                "upload_id": upload_id,
                "index": index,
                "chunks_received": received,
                "total_chunks": total_chunks,
                "received_bytes": int(min(received * record["chunk_bytes"], record["total_bytes"])),
            }
        ),
        200,
    )


# ---------------------------------------------------------------------------
# POST /api/workspace/ingest/capture/finalize
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/workspace/ingest/capture/finalize", methods=["POST"])
def ingest_capture_finalize():
    """Close a chunked capture upload: verify completeness, validate the zip on
    the broker (cheap — central directory only), spawn ``lidar_recon_job`` ->
    ``202 {job_id, scan_id}``. Upload identified by the ``X-Upload-Id`` header."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401

    upload_id = request.headers.get("X-Upload-Id")
    if not upload_id:
        return (
            jsonify(
                {"error": "missing_upload_id", "detail": "set the X-Upload-Id header"}
            ),
            400,
        )
    record, error = _owned_capture_upload(upload_id, user_id)
    if error:
        return error

    missing = sorted(set(range(record["total_chunks"])) - record["received"])
    if missing:
        return (
            jsonify(
                {
                    "error": "incomplete_upload",
                    "missing_chunks": missing[:64],
                    "missing_count": len(missing),
                    "total_chunks": record["total_chunks"],
                    "detail": (
                        f"{len(missing)} of {record['total_chunks']} chunks never "
                        "arrived — re-send the listed indices, then finalize again."
                    ),
                }
            ),
            409,
        )

    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "persistence_unavailable"}), 503

    dest = record["dest"]
    actual = os.path.getsize(dest) if os.path.exists(dest) else 0
    if actual != record["total_bytes"]:
        _drop_upload(upload_id)
        return (
            jsonify(
                {
                    "error": "size_mismatch",
                    "expected_bytes": record["total_bytes"],
                    "actual_bytes": int(actual),
                    "detail": (
                        "the staged file is not the size announced at init — the "
                        "upload was corrupted. Please start it again."
                    ),
                }
            ),
            409,
        )

    # Cheap gate on the broker: reject a bad zip for FREE instead of paying for a
    # Modal job that would only fail (routes_ingest's finalize-header-check doctrine).
    try:
        validate_capture_zip(dest)
    except CaptureZipRejected as exc:
        _drop_upload(upload_id)
        return jsonify(exc.payload()), exc.status

    persistence._commit_blobs()

    job_id = uuid.uuid4().hex
    scan_id = uuid.uuid4().hex
    _publish_job(
        demo_jobs.job_status_record(
            job_id,
            user_id,
            status="queued",
            stage="upload",
            scan_id=scan_id,
            kind="capture_lidar",
            upload_bytes=record["total_bytes"],
        )
    )
    try:
        _spawn_capture_job(
            job_id=job_id,
            user_id=str(user_id),
            scan_id=scan_id,
            upload_rel_path=record["rel"],
            session_label=record["filename"],
        )
    except Exception as exc:  # noqa: BLE001 — spawn boundary
        print(f"[oreos.capture] lidar_recon_job spawn failed: {exc}")
        _drop_upload(upload_id)
        _publish_job(
            demo_jobs.job_status_record(
                job_id,
                user_id,
                status="failed",
                stage="upload",
                scan_id=scan_id,
                kind="capture_lidar",
                error="spawn_failed",
            )
        )
        return (
            jsonify({"error": "spawn_failed", "detail": "the recon worker could not be started"}),
            502,
        )

    # State only; the bytes now belong to the job, which deletes them when done.
    with _capture_uploads_lock:
        _capture_uploads.pop(upload_id, None)
    return jsonify({"job_id": job_id, "scan_id": scan_id}), 202


@oreos_bp.route("/api/workspace/ingest/capture/<upload_id>", methods=["DELETE"])
def ingest_capture_cancel(upload_id: str):
    """Cancel a staged capture upload and free its bytes. Not part of the fixed
    wire contract (the iOS client doesn't need it to work), additive — mirrors the
    splat lane's cancel route for the same operational reason (a closed app leaves
    no orphaned GBs)."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    record, error = _owned_capture_upload(upload_id, user_id)
    if error:
        return error
    _drop_upload(upload_id)
    return jsonify({"upload_id": upload_id, "cancelled": True}), 200
