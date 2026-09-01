"""Prepared-export routes: build a scene's zip as a background job, then download a
finished file.

  POST /api/scenes/<scan_id>/demo/export/prepare
      Spawn ``demo_export_job`` on the Modal side-function. ``202 {job_id, status}``;
      poll the EXISTING ``GET /api/workspace/jobs/<job_id>``. Body (all optional):
      ``{format?, source?, force?}``. ``409 export_job_active`` (carrying the live
      ``job_id``) when one is already running for this scene+format, so a double-click
      cannot fan out two containers both deflating 1.39 GB onto the same slot.

  GET  /api/scenes/<scan_id>/demo/export/prepared?format=…[&source=…]
      Always 200 with an explicit ``status`` (the ``routes_lod.py`` shape), so the panel
      never has to interpret a 404:

        {"status": "ready",   "artifact": {...}, "stale": false}
        {"status": "running", "job_id": "...", "stage": "compress", ...}
        {"status": "error",   "job_id": "...", "error": "..."}
        {"status": "none"}

``prepare_export_for_scene(user_id, scan_id, body, store=…) -> (payload, status)`` is the
Flask-free seam behind the POST (the ``routes_nav.plan_path_for_scene`` pattern). The agent's
``export.build`` tool calls it directly, so the scene has exactly ONE export scheduler and
one 409 guard — an agent build and an operator click cannot fan out two containers.

The ZIP ITSELF is served by the EXISTING derived GET route
(``GET /api/scenes/<id>/derived/<key>``) — the same proven path the LOD levels ride. It
already does owner auth, path containment and Range, and because the file is finished on
the volume its first byte leaves immediately: no build happens inside the request.
``artifact.download_path`` hands the client that URL so no caller hand-builds it.

WHY EVERY EXPORT GOES THROUGH THE JOB, INCLUDING SMALL ONES
-----------------------------------------------------------
There is no size threshold and no fast path. A threshold is a number that has to be right
on demo morning against an unknown scene, and being wrong in the cheap direction is
exactly the failure we are fixing (45 s of silence, then a killed request). Uniform
behaviour means the panel has ONE flow to render and one to test, the operator always
gets a progress bar, and a small scene's job simply finishes in seconds. The synchronous
``GET /api/scenes/<id>/export`` route is untouched for scripts and existing callers.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import export_artifacts as ea
from server.export.record import EXPORT_SOURCE_ORIGINAL, resolved_source_key

# ---------------------------------------------------------------------------
# job spawning (injectable for tests — the routes_lod.py / routes_ingest.py pattern)
# ---------------------------------------------------------------------------

_export_spawner: Optional[Callable[..., Any]] = None


def configure_export_spawner(spawner: Optional[Callable[..., Any]]) -> None:
    """Inject the job spawner (tests pass a recording fake; ``None`` restores the real
    ``modal.Function.from_name`` lookup)."""
    global _export_spawner
    _export_spawner = spawner


def _spawn_export_job(**kwargs: Any) -> None:
    """Spawn the export job on the deployed app (fire-and-forget; status flows through the
    shared demo-jobs Dict, which the workspace job route already serves).

    The FORMAT picks the container. Isaac authors a Poisson mesh — measured 378 s on 5M
    points, and loading a 63M-point scene needs tens of GB — so it runs on a 64 GB twin,
    while a LeRobot zip does not pay for that. Same body either way: `demo_isaac_export_job`
    delegates to `demo_export_job.local()`."""
    if _export_spawner is not None:
        _export_spawner(**kwargs)
        return
    import modal

    from server.oreos import export_artifacts as ea

    fn = (
        "demo_isaac_export_job"
        if str(kwargs.get("export_format") or "") in ea.HEAVY_JOB_FORMATS
        else "demo_export_job"
    )
    modal.Function.from_name("vggt-slam-streaming", fn).spawn(**kwargs)


# In-process record of the job last spawned per (scene, format), so a second POST can
# answer 409 with a real job id. Safe for the demo topology (the broker runs
# max_containers=1, single operator) — same reasoning as routes_lod's ``_ACTIVE``.
_ACTIVE: dict[tuple[str, str, str], str] = {}
_ACTIVE_LOCK = threading.Lock()


def _jobs_store():
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


def _scene_or_error(scan_id: str):
    """``(user_id, persistence, record, None)`` or ``(None, None, None, response)``."""
    user_id = auth_user_id()
    if not user_id:
        return None, None, None, (jsonify({"error": "invalid_token"}), 401)
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


def _format_error(raw: Any) -> tuple[Optional[str], Optional[dict]]:
    """``(format, None)`` or ``(None, error-payload)`` — the Flask-free core, so the
    ``prepare_export_for_scene`` seam and the routes cannot drift on the message."""
    fmt = (str(raw).strip() if raw is not None else "") or "openreality"
    if fmt not in ea.EXPORT_JOB_FORMATS:
        return None, {
            "error": "invalid_format",
            "message": (
                f"unknown format {fmt!r}; the prepared-export job builds "
                f"{list(ea.EXPORT_JOB_FORMATS)}"
            ),
        }
    return fmt, None


def _validated_format(raw: Any) -> tuple[Optional[str], Optional[tuple]]:
    fmt, err = _format_error(raw)
    if err is not None:
        return None, (jsonify(err), 400)
    return fmt, None


def _source_error(
    persistence: Any, user_id: str, scan_id: str, source: Any
) -> Optional[dict]:
    """Mirror the zip route: an explicit but bogus/foreign ``source`` is a hard 404, never
    a silent degrade to the original. Flask-free core (see ``_format_error``)."""
    if source is None or str(source).strip() in ("", EXPORT_SOURCE_ORIGINAL):
        return None
    get_derived_path = getattr(persistence, "get_derived_artifact_path", None)
    resolved = (
        get_derived_path(user_id, scan_id, str(source).strip())
        if callable(get_derived_path)
        else None
    )
    if not resolved:
        return {
            "error": "unknown_derived_key",
            "message": (
                f"no derived artifact {str(source).strip()!r} for this scan; pass a "
                "derived key returned by the anchor/clamp routes, or omit source"
            ),
        }
    return None


def _validated_source(persistence: Any, user_id: str, scan_id: str, source: Any):
    err = _source_error(persistence, user_id, scan_id, source)
    return None if err is None else (jsonify(err), 404)


def _artifact_payload(persistence: Any, user_id: str, scan_id: str, slot: dict) -> Optional[dict]:
    """A slot dict enriched for the wire, or ``None`` when the index names a zip that is
    no longer on the volume (index and blobs are separate writes; trust the blob)."""
    key = str(slot.get("key") or "")
    try:
        path = persistence.get_derived_artifact_path(user_id, scan_id, key)
    except Exception:
        path = None
    if not path or not os.path.isfile(path):
        return None
    on_disk = os.path.getsize(path)
    built_at = float(slot.get("built_at") or 0.0)
    return {
        **slot,
        # The index records what the job wrote; ``bytes`` is re-stat'ed here so the
        # number driving the download progress bar is the file that will actually be
        # sent, not a remembered one.
        "bytes": int(on_disk),
        "age_s": round(time.time() - built_at, 1) if built_at else None,
        # The derived route's namespace already IS ``/derived/``, so the key's own
        # ``derived/`` prefix comes off — same normalization as @reality/protocol's
        # REST_PATHS.SCENE_DERIVED, so client and server produce the identical URL.
        "download_path": f"/api/scenes/{scan_id}/derived/{key.split('derived/', 1)[-1]}",
        # Matches @reality/protocol's sceneExportDownloadFilename.
        "download_filename": f"scan-{scan_id}-{slot.get('format')}.zip",
    }


@oreos_bp.route("/api/scenes/<scan_id>/demo/export/prepared", methods=["GET"])
def get_prepared_export_route(scan_id: str):
    """Status of this scene's prepared zip for one format. Read-only; cheap (reads a
    small JSON index and stats one file — it never runs a builder)."""
    user_id, persistence, record, err = _scene_or_error(scan_id)
    if err:
        return err

    export_format, ferr = _validated_format(request.args.get("format"))
    if ferr:
        return ferr

    source = request.args.get("source")
    serr = _validated_source(persistence, user_id, scan_id, source)
    if serr:
        return serr

    wanted_source = resolved_source_key(record, source)
    payload: dict[str, Any] = {
        "scan_id": scan_id,
        "format": export_format,
        "source_key": wanted_source,
        "artifact": None,
    }

    index = ea.read_index(persistence, user_id, scan_id)
    slot = ea.live_slot(index, export_format)
    artifact = (
        _artifact_payload(persistence, user_id, scan_id, slot) if isinstance(slot, dict) else None
    )

    with _ACTIVE_LOCK:
        job_id = _ACTIVE.get((user_id, scan_id, export_format))
    job = _job_record(job_id) if job_id else None
    status = (job or {}).get("status") if job_id else None

    # A job in flight wins the headline even when a previous artifact exists — the
    # operator asked for a rebuild and wants to watch it. The old artifact still rides
    # along so the panel can keep offering the last good download meanwhile.
    if job_id and (status in ("queued", "running") or job is None):
        payload.update(
            {
                "status": "running",
                "job_id": job_id,
                "stage": (job or {}).get("stage"),
                "tree_bytes": (job or {}).get("tree_bytes"),
                "zip_bytes": (job or {}).get("zip_bytes"),
                "artifact": artifact,
            }
        )
        return jsonify(payload), 200

    if artifact:
        stale, reason = ea.staleness(slot, wanted_source)
        payload.update(
            {"status": "ready", "artifact": artifact, "stale": stale, "stale_reason": reason}
        )
        return jsonify(payload), 200

    if job_id and status in ("failed", "error"):
        payload.update({"status": "error", "job_id": job_id, "error": (job or {}).get("error")})
        return jsonify(payload), 200

    if job_id and status == "done":
        # Job finished but no blob is visible here: this container's volume view can lag a
        # write made in another container. Report it as still committing rather than
        # inventing "ready" over a file we cannot stat.
        payload.update({"status": "running", "job_id": job_id, "stage": "committing"})
        return jsonify(payload), 200

    payload.update({"status": "none", "reason": "no export has been prepared for this scene yet"})
    return jsonify(payload), 200


def prepare_export_for_scene(
    user_id: str,
    scan_id: str,
    body: Optional[dict] = None,
    *,
    store: Any = None,
) -> tuple[dict, int]:
    """Spawn (or short-circuit) one prepared-export build → ``(json-able payload, status)``.

    THE non-Flask seam behind ``POST /api/scenes/<id>/demo/export/prepare`` — same shape
    and the same reason as ``routes_nav.plan_path_for_scene``: the route below is a thin
    wrapper over it, and the agent's ``export.build`` tool calls it directly. There is
    therefore exactly ONE export scheduler in the process — the 409 ``export_job_active``
    guard, the "already prepared and fresh" short-circuit and the ``_ACTIVE`` bookkeeping
    are shared, so an agent-issued build and an operator's click cannot fan out two
    containers onto the same slot.

    Body (all optional): ``{format?, source?, force?}``. ``store`` defaults to the live
    ``scene_persistence()``. Statuses: 200 already-ready · 202 queued · 400 invalid_format
    · 404 not_found/unknown_derived_key · 409 export_job_active · 502 spawn_failed · 503
    no_scene_store. Auth is the CALLER's job (the route does it; the agent runs as the
    scene's owner).
    """
    persistence = store if store is not None else scene_persistence()
    if persistence is None:
        return (
            {"error": "no_scene_store", "message": "server has no persisted scene store"},
            503,
        )
    record = persistence.get_scene(user_id, scan_id)
    if not record:
        return {"error": "not_found", "scan_id": scan_id}, 404

    body = body if isinstance(body, dict) else {}
    export_format, ferr = _format_error(body.get("format"))
    if ferr:
        return ferr, 400

    source = body.get("source")
    serr = _source_error(persistence, user_id, scan_id, source)
    if serr:
        return serr, 404

    force = bool(body.get("force"))
    wanted_source = resolved_source_key(record, source)

    # Already prepared, fresh, same geometry → don't spend 45 s of container time
    # rebuilding a byte-identical archive. ``force`` is the operator's override.
    if not force:
        index = ea.read_index(persistence, user_id, scan_id)
        slot = ea.live_slot(index, export_format)
        if isinstance(slot, dict):
            artifact = _artifact_payload(persistence, user_id, scan_id, slot)
            stale, reason = ea.staleness(slot, wanted_source)
            if artifact and not stale:
                return (
                    {
                        "status": "ready",
                        "scan_id": scan_id,
                        "format": export_format,
                        "artifact": artifact,
                        "message": "a current export is already prepared; pass force to rebuild",
                    },
                    200,
                )
            if artifact and stale:
                print(f"[demo-export] rebuilding {scan_id}/{export_format}: {reason}")

    with _ACTIVE_LOCK:
        existing = _ACTIVE.get((user_id, scan_id, export_format))
    if existing:
        rec = _job_record(existing)
        if rec is None or rec.get("status") in ("queued", "running"):
            return (
                {
                    "error": "export_job_active",
                    "job_id": existing,
                    "scan_id": scan_id,
                    "format": export_format,
                    "message": "an export is already being prepared for this scene",
                },
                409,
            )

    job_id = uuid.uuid4().hex[:12]
    try:
        _spawn_export_job(
            job_id=job_id,
            user_id=user_id,
            scan_id=scan_id,
            export_format=export_format,
            source=str(source).strip() if source else None,
        )
    except Exception as exc:
        return (
            {
                "error": "spawn_failed",
                "scan_id": scan_id,
                "message": f"{type(exc).__name__}: {exc}"[:300],
            },
            502,
        )

    with _ACTIVE_LOCK:
        _ACTIVE[(user_id, scan_id, export_format)] = job_id

    # Seed the polling record here rather than waiting for the container to boot: the
    # panel starts polling immediately and a 404 in that window reads as "job lost".
    jobs_store = _jobs_store()
    if jobs_store is not None:
        try:
            from server.oreos.jobs import job_status_record

            jobs_store[job_id] = job_status_record(
                job_id,
                user_id,
                status="queued",
                stage="queued",
                scan_id=scan_id,
                kind="export",
                export_format=export_format,
            )
        except Exception:
            pass

    return (
        {
            "job_id": job_id,
            "status": "queued",
            "scan_id": scan_id,
            "format": export_format,
            "source_key": wanted_source,
            # The EXISTING workspace-jobs poll route; handed back so no caller (panel or
            # agent) hand-builds the URL.
            "poll_route": f"/api/workspace/jobs/{job_id}",
            "queued_at": time.time(),
        },
        202,
    )


@oreos_bp.route("/api/scenes/<scan_id>/demo/export/prepare", methods=["POST"])
def post_prepare_export_route(scan_id: str):
    """Spawn the background build. See the module docstring for the response contract.

    A thin wrapper over :func:`prepare_export_for_scene` — auth here, everything else
    there, so the agent's ``export.build`` tool rides the identical path."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    payload, status = prepare_export_for_scene(
        user_id, scan_id, request.get_json(silent=True) or {}
    )
    return jsonify(payload), status
