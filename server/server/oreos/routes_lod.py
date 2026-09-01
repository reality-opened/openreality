"""Demo LOD routes: serve and generate level-of-detail splat artifacts.

  GET  /api/scenes/<scan_id>/lod
      The LOD index for a scene. Always answers 200 with an explicit ``status``
      so the client never has to interpret a 404:

        {"status": "ready",    "index": {...}}          # artifacts on the volume
        {"status": "none",     "index": null, ...}      # never generated
        {"status": "running",  "job_id": "...", ...}    # generation in flight
        {"status": "error",    "job_id": "...", "error": "..."}
        {"status": "not_needed","index": null, ...}     # source already small

  POST /api/scenes/<scan_id>/lod
      Spawn generation on the Modal side-function ``demo_lod_job``. Returns
      ``202 {job_id, status}``; poll the existing ``GET /api/workspace/jobs/<id>``.
      Returns ``409 lod_job_active`` (with the live ``job_id``) when one is
      already running for this scene, so a double-click cannot fan out GPU-free
      but still billable containers. Body (all optional):
      ``{levels?: [int], method?: str, force?: bool}``.

The LOD *files themselves* are served by the EXISTING derived GET route
(``GET /api/scenes/<id>/derived/<key>``), which already does owner auth, path
containment and Range — so this module adds zero new binary read plumbing, per
the shell design's "reuse the derived route" rule.

Both routes are owner-only. The broker never decimates: it only reads a small
JSON index and spawns a container (see ``modal_oreos_lod.py`` for why).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import lod as lod_mod

# ---------------------------------------------------------------------------
# job spawning (injectable for tests — routes_ingest.py pattern)
# ---------------------------------------------------------------------------

_lod_spawner: Optional[Callable[..., Any]] = None


def configure_lod_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    """Inject the job spawner (tests pass a recording fake; ``None`` restores the
    real ``modal.Function.from_name`` lookup)."""
    global _lod_spawner
    _lod_spawner = spawner


def _spawn_lod_job(**kwargs: Any) -> None:
    """Spawn ``demo_lod_job`` on the deployed app (fire-and-forget; status flows
    through the shared demo-jobs Dict)."""
    if _lod_spawner is not None:
        _lod_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("vggt-slam-streaming", "demo_lod_job").spawn(**kwargs)


# In-process record of the job we last spawned per scene, so a second POST can
# answer 409 with a real job id. Safe for the demo topology (broker runs
# max_containers=1, single operator) — same reasoning as routes_planes' cache.
_ACTIVE: dict[tuple[str, str], str] = {}
_ACTIVE_LOCK = threading.Lock()


def _jobs_store():
    """The demo-jobs Dict, or ``None`` when unreachable (status degrades, never 500s)."""
    try:
        from server.oreos.jobs import _get_jobs_store

        return _get_jobs_store()
    except Exception:
        return None


def _job_record(job_id: str) -> Optional[dict]:
    store = _jobs_store()
    if store is None:
        return None
    try:
        rec = store.get(job_id)
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def _read_index(persistence, user_id: str, scan_id: str) -> Optional[dict]:
    """Read ``derived/demo/lod/index.json`` off the volume, or ``None``."""
    try:
        path = persistence.get_derived_artifact_path(
            user_id, scan_id, f"derived/{lod_mod.LOD_INDEX_KEY}"
        )
    except Exception:
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8"))
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


def _scene_or_error(scan_id: str):
    """``(user_id, persistence, record, None)`` or ``(None, None, None, response)``."""
    user_id = auth_user_id()
    if not user_id:
        return None, None, None, (jsonify({"error": "unauthorized"}), 401)
    persistence = scene_persistence()
    if persistence is None:
        return None, None, None, (
            jsonify({"error": "no_scene_store", "message": "server has no persisted scene store"}),
            503,
        )
    record = persistence.get_scene(user_id, scan_id)
    if not record:
        return None, None, None, (jsonify({"error": "not_found", "scan_id": scan_id}), 404)
    return user_id, persistence, record, None


@oreos_bp.route("/api/scenes/<scan_id>/lod", methods=["GET"])
def get_scene_lod_route(scan_id: str):
    user_id, persistence, record, err = _scene_or_error(scan_id)
    if err:
        return err

    index = _read_index(persistence, user_id, scan_id)
    source_count = int(record.get("point_count") or 0)
    payload: dict[str, Any] = {
        "scan_id": scan_id,
        "source_gaussians": source_count or None,
        "default_level_target": lod_mod.DEFAULT_LEVEL,
    }

    if index:
        payload["status"] = "ready"
        payload["index"] = index
        return jsonify(payload), 200

    payload["index"] = None
    with _ACTIVE_LOCK:
        job_id = _ACTIVE.get((user_id, scan_id))
    if job_id:
        rec = _job_record(job_id)
        status = (rec or {}).get("status")
        if status in ("queued", "running") or (rec is None):
            payload.update({"status": "running", "job_id": job_id,
                            "stage": (rec or {}).get("stage")})
            return jsonify(payload), 200
        if status in ("failed", "error"):
            payload.update({"status": "error", "job_id": job_id,
                            "error": (rec or {}).get("error")})
            return jsonify(payload), 200
        # status == done but no index on the volume: the volume view may be stale
        # in this container. Report ready-pending rather than inventing a state.
        payload.update({"status": "running", "job_id": job_id, "stage": "committing"})
        return jsonify(payload), 200

    payload["status"] = "none"
    payload["reason"] = "no LOD artifacts generated for this scene yet"
    return jsonify(payload), 200


@oreos_bp.route("/api/scenes/<scan_id>/lod", methods=["POST"])
def post_scene_lod_route(scan_id: str):
    user_id, persistence, record, err = _scene_or_error(scan_id)
    if err:
        return err

    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))

    if not force and _read_index(persistence, user_id, scan_id):
        return jsonify({"status": "ready", "scan_id": scan_id,
                        "message": "LOD artifacts already exist; pass force to rebuild"}), 200

    with _ACTIVE_LOCK:
        existing = _ACTIVE.get((user_id, scan_id))
    if existing:
        rec = _job_record(existing)
        if rec is None or rec.get("status") in ("queued", "running"):
            return jsonify({
                "error": "lod_job_active", "job_id": existing, "scan_id": scan_id,
                "message": "an LOD job is already running for this scene",
            }), 409

    levels = body.get("levels")
    if levels is not None:
        try:
            levels = [int(x) for x in levels]
        except (TypeError, ValueError):
            return jsonify({"error": "bad_request", "message": "levels must be a list of ints"}), 400

    method = str(body.get("method") or "auto")
    if method not in ("auto", "voxel_importance", "voxel_first", "uniform", "importance"):
        return jsonify({"error": "bad_request", "message": f"unknown method {method!r}"}), 400

    job_id = uuid.uuid4().hex[:12]
    try:
        _spawn_lod_job(
            job_id=job_id, user_id=user_id, scan_id=scan_id,
            levels=levels, method=method, force=force,
        )
    except Exception as exc:
        return jsonify({
            "error": "spawn_failed", "scan_id": scan_id,
            "message": f"{type(exc).__name__}: {exc}"[:300],
        }), 502

    with _ACTIVE_LOCK:
        _ACTIVE[(user_id, scan_id)] = job_id

    store = _jobs_store()
    if store is not None:
        try:
            from server.oreos.jobs import job_status_record

            store[job_id] = job_status_record(
                job_id, user_id, status="queued", stage="queued", scan_id=scan_id, kind="lod"
            )
        except Exception:
            pass

    return jsonify({"job_id": job_id, "status": "queued", "scan_id": scan_id,
                    "queued_at": time.time()}), 202
