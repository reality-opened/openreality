"""Robot-recording ingest — the merged recordings pipeline's front door in the OS.

``POST /api/workspace/ingest/recording``: a DimOS memory2 ``.db`` recording streams to
the scenes volume's ``_uploads/`` staging (routes_ingest's exact pattern), then
``oreos_recording_job`` runs the recordings pipeline end-to-end on Modal — extract →
recon on the SEPARATE ``oreos-recon`` A100 app (different VGGT backbone by design; see
``recordings/modal_image.py``) → consistency + QC gate → persisted
``source="robot_recording"`` scene with the QC report attached as
``derived/demo/recordings/*`` artifacts. Poll ``GET /api/workspace/jobs/<job_id>``.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import jobs as demo_jobs
from server.oreos.routes_ingest import (
    _TooLarge,
    _UPLOADS_DIRNAME,
    _cleanup_upload_dir,
    _publish_job,
    _safe_filename,
    _stream_request_to,
)

RECORDING_SUFFIXES = (".db",)
RECORDING_UPLOAD_MAX_BYTES = int(os.environ.get("DEMO_RECORDING_MAX_BYTES", str(2 * 1024**3)))

# Injectable spawner (routes_lod.py pattern — tests pass a recording fake).
_recording_spawner: Optional[Callable[..., Any]] = None


def configure_recording_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    global _recording_spawner
    _recording_spawner = spawner


def _spawn_recording_job(**kwargs: Any) -> None:
    if _recording_spawner is not None:
        _recording_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("vggt-slam-streaming", "oreos_recording_job").spawn(**kwargs)


@oreos_bp.route("/api/workspace/ingest/recording", methods=["POST"])
def ingest_recording_route():
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    persistence = scene_persistence()
    if persistence is None:
        return jsonify({"error": "no_scene_store"}), 503

    raw_name = (request.headers.get("X-Upload-Filename") or "recording.db").strip()
    if not raw_name.lower().endswith(RECORDING_SUFFIXES):
        return (
            jsonify(
                {
                    "error": "unsupported_recording",
                    "message": f"expected a DimOS memory2 .db recording, got {raw_name!r}",
                }
            ),
            415,
        )

    from server.scene_report.store import _safe_component

    upload_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    scan_id = uuid.uuid4().hex
    fname = _safe_filename(raw_name, fallback="recording.db")
    rel = "/".join([_safe_component(user_id), _UPLOADS_DIRNAME, upload_id, fname])
    dest = os.path.join(persistence._blob_root, rel)  # noqa: SLF001 — routes_ingest convention
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        total = _stream_request_to(dest, RECORDING_UPLOAD_MAX_BYTES)
    except _TooLarge:
        _cleanup_upload_dir(dest)
        return jsonify({"error": "too_large", "limit_bytes": RECORDING_UPLOAD_MAX_BYTES}), 413
    except Exception as exc:  # noqa: BLE001 — upload boundary
        print(f"[oreos.recordings] upload stream failed: {exc}")
        _cleanup_upload_dir(dest)
        return jsonify({"error": "upload_failed"}), 500
    if total == 0:
        _cleanup_upload_dir(dest)
        return jsonify({"error": "empty_upload"}), 400
    persistence._commit_blobs()  # noqa: SLF001

    _publish_job(
        demo_jobs.job_status_record(
            job_id,
            user_id,
            status="queued",
            stage="upload",
            scan_id=scan_id,
            kind="recording",
            upload_bytes=total,
        )
    )
    try:
        _spawn_recording_job(
            job_id=job_id,
            user_id=str(user_id),
            scan_id=scan_id,
            upload_rel_path=rel,
            recording_name=fname,
        )
    except Exception as exc:  # noqa: BLE001 — spawn boundary
        print(f"[oreos.recordings] spawn failed: {exc}")
        return jsonify({"error": "spawn_failed", "message": "could not start the recording job"}), 503
    return jsonify({"job_id": job_id, "scan_id": scan_id}), 202
