"""Demo job-status store + polling route (docs/demo-2026-07 design/shell.md §4a).

Write side: ``modal_oreos_ingest.demo_recon_job`` publishes stage-boundary status
records into the ``vggt-slam-demo-jobs`` ``modal.Dict`` (mirroring
``modal_streaming.py``'s ``worker_status_dict`` publish pattern, :347-354). It
builds the records with :func:`job_status_record` below, so this module is the
single source of truth for the shape.

Read side (this module, runs on the CPU broker): ``GET
/api/workspace/jobs/<job_id>`` returns the caller's record verbatim, or the 404
JSON envelope ``{"error": "job_not_found", "job_id": ...}`` when the job is
unknown — including when it belongs to another user (existence is not revealed).

Record shape::

    {job_id, user_id, scan_id, status: queued|running|done|failed,
     stage: upload|recon|persist|done, updated_at, error?, ...extra}

Wire vocabulary: @reality/protocol ``DemoJobStatus`` is ``queued | running |
done | error``. Ingest writers historically publish ``failed`` (kept as-stored
for back-compat with records already in the modal.Dict); the route translates
``failed`` → ``error`` in responses so every polled job speaks the protocol
vocabulary regardless of which writer produced it.

``stage`` is the coarse pipeline position the UI shows. NOTE: the design doc
sketched ``upload|recon|report|persist``; the ``report`` boundary is not
observable without modifying ``reconstruct_pilot`` (out of W0's additive-only
scope), so the report phase is folded into ``recon`` — recorded as a deviation
in BUILD-LOG.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from flask import jsonify

from server.oreos import auth_user_id, oreos_bp

# The named modal.Dict the ingest job writes into. Env-overridable so a staging
# deploy can point at its own dict without a code change.
DEMO_JOBS_DICT_NAME = "vggt-slam-demo-jobs"

# Lazily bound: a modal.Dict on the broker, or whatever store (e.g. a plain
# dict) tests inject via configure_jobs_store(). Only needs ``.get(key)``.
_jobs_store: Any = None


def configure_jobs_store(store: Any) -> None:
    """Inject the job-status store (tests pass a plain dict; ``None`` resets to
    the lazy modal.Dict binding)."""
    global _jobs_store
    _jobs_store = store


def _get_jobs_store() -> Any:
    """The injected store, else a lazily created handle on the named modal.Dict.
    Returns ``None`` when modal is unavailable/unauthenticated (local dev) — the
    route answers 503 rather than crashing."""
    global _jobs_store
    if _jobs_store is not None:
        return _jobs_store
    try:
        import modal

        _jobs_store = modal.Dict.from_name(
            os.environ.get("DEMO_JOBS_DICT", DEMO_JOBS_DICT_NAME),
            create_if_missing=True,
        )
    except Exception as exc:  # no creds / no modal — leave unbound, report 503
        print(f"[demo.jobs] jobs store unavailable: {exc}")
        return None
    return _jobs_store


def job_status_record(
    job_id: Any,
    user_id: Any,
    *,
    status: str,
    stage: str,
    scan_id: Optional[Any] = None,
    **extra: Any,
) -> dict[str, Any]:
    """The canonical job-status record. Writers (``modal_oreos_ingest``) build
    every publish through this function; the route returns it verbatim, so the
    two sides can never drift."""
    record: dict[str, Any] = {
        "job_id": str(job_id),
        "user_id": str(user_id),
        "status": str(status),
        "stage": str(stage),
        "scan_id": str(scan_id) if scan_id else None,
        "updated_at": time.time(),
    }
    record.update(extra)
    return record


@oreos_bp.route("/api/workspace/jobs/<job_id>", methods=["GET"])
def get_demo_job(job_id: str):
    """Poll one ingest job's status. Owner-scoped: an unknown job id and another
    user's job return the identical 404 envelope."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    store = _get_jobs_store()
    if store is None:
        return jsonify({"error": "jobs_store_unavailable"}), 503
    try:
        record = store.get(str(job_id))
    except Exception as exc:
        print(f"[demo.jobs] job read failed {job_id!r}: {exc}")
        record = None
    if not isinstance(record, dict) or str(record.get("user_id") or "") != user_id:
        return jsonify({"error": "job_not_found", "job_id": str(job_id)}), 404
    if record.get("status") == "failed":  # stored legacy value → protocol vocab
        record = {**record, "status": "error"}
    return jsonify(record)


# ---------------------------------------------------------------------------
# W3 addition (append-only): broker-local gen-asset jobs + the scene-scoped
# polling route (design/features.md §0.4 "Job pattern: long ops -> 202 {job_id};
# GET /api/scenes/<id>/jobs/<job_id> -> {status, result?, error?}; broker
# thread + dict; client polls 1 Hz").
#
# Distinct from the ingest jobs above by design: ingest jobs are written by a
# SEPARATE Modal GPU function into a modal.Dict; gen-asset jobs (SAM-3D
# completion, TRELLIS variants) run in a thread of THIS broker process (the
# heavy lifting happens on the remote Modal SAM-3D app / fal.ai — the thread
# only orchestrates), so a plain in-process dict is the right store. Durable
# outputs are persisted as derived artifacts by the job body; the in-memory
# record is only the polling envelope, and a broker restart honestly 404s
# in-flight jobs (the client surfaces "job lost").
#
# Status vocabulary follows @reality/protocol DemoJobStatus:
# queued | running | done | error.
# ---------------------------------------------------------------------------

import threading
import uuid

_scene_jobs: dict[str, dict[str, Any]] = {}
_scene_jobs_lock = threading.Lock()


def create_scene_job(user_id: Any, scan_id: Any, kind: str, **extra: Any) -> str:
    """Register a queued gen-asset job; returns its ``job_id``.

    Also reserves credits for it. The hold is keyed on the job id, so the settle
    in ``run_scene_job`` finds it without any extra plumbing, and a crashed
    broker leaves a hold the reaper can close.
    """
    job_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {
        "job_id": job_id,
        "user_id": str(user_id),
        "scan_id": str(scan_id),
        "kind": str(kind),
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    record.update(extra)

    from server.billing import meter, prices

    record["hold_id"] = meter.open_hold(
        str(user_id), "gen_asset", job_id, prices.CREDITS_GEN_ASSET
    )

    with _scene_jobs_lock:
        _scene_jobs[job_id] = record
    return job_id


def update_scene_job(job_id: str, **fields: Any) -> None:
    with _scene_jobs_lock:
        record = _scene_jobs.get(job_id)
        if record is None:
            return
        record.update(fields)
        record["updated_at"] = time.time()


def get_scene_job_record(job_id: str) -> Optional[dict[str, Any]]:
    with _scene_jobs_lock:
        record = _scene_jobs.get(job_id)
        return dict(record) if record else None


def any_active_scene_job(user_id: Any, scan_id: Any, kind_prefix: str = "") -> Optional[str]:
    """job_id of a queued/running job for this scene (optionally by kind prefix)
    — the idempotency guard: one active gen job per object (409 otherwise)."""
    with _scene_jobs_lock:
        for job_id, record in _scene_jobs.items():
            if (
                record.get("user_id") == str(user_id)
                and record.get("scan_id") == str(scan_id)
                and record.get("status") in ("queued", "running")
                and str(record.get("kind", "")).startswith(kind_prefix)
            ):
                return job_id
    return None


def run_scene_job(job_id: str, body: Any) -> None:
    """Run ``body()`` on a daemon thread; its return dict becomes ``result``,
    an exception becomes the honest ``error`` string (type + message)."""

    def _runner() -> None:
        from server.billing import meter, prices

        record = get_scene_job_record(job_id) or {}
        hold_id = record.get("hold_id")

        update_scene_job(job_id, status="running")
        try:
            result = body()
            update_scene_job(job_id, status="done", result=result)
            meter.settle(hold_id, prices.CREDITS_GEN_ASSET, basis="estimate")
        except Exception as exc:  # noqa: BLE001 — the job envelope carries the reason
            print(f"[demo.jobs] scene job {job_id} failed: {type(exc).__name__}: {exc}")
            update_scene_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
            # Our fault, our cost. Partial-refund logic is not worth maintaining
            # forever for a few cents.
            meter.void(hold_id, f"{type(exc).__name__}: {exc}"[:200])

    threading.Thread(target=_runner, name=f"demo-job-{job_id}", daemon=True).start()


@oreos_bp.route("/api/scenes/<scan_id>/jobs/<job_id>", methods=["GET"])
def get_scene_gen_job(scan_id: str, job_id: str):
    """Poll one gen-asset job (SAM-3D completion / TRELLIS variant). Owner- and
    scene-scoped; unknown, foreign-user and foreign-scene ids all return the
    identical 404 envelope."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    record = get_scene_job_record(str(job_id))
    if (
        record is None
        or str(record.get("user_id") or "") != user_id
        or str(record.get("scan_id") or "") != str(scan_id)
    ):
        return jsonify({"error": "job_not_found", "job_id": str(job_id)}), 404
    return jsonify(record)
