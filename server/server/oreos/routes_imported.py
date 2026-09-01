"""Imported-splat parity routes: synthetic views + ground frame.

  POST /api/demo/scene/<scan_id>/synthetic_views
      Register N renders of this scene's splat as synthetic views (body per
      IMPORTED-SPLAT-CONTRACT.md: ``{replace?: bool, views: [{image_b64, position,
      quaternion, fov_y_deg, width, height}]}``, poses in the WORLD frame). PNGs land in
      the derived blob namespace ``derived/demo/synthetic_views/<view_id>.png``; the
      scene record gains a ``synthetic_views`` manifest (metadata only, never bytes).
      ``replace: true`` clears the prior set for this scan.

  GET  /api/demo/scene/<scan_id>/synthetic_views
      The manifest + a fetchable ``url`` per view.

  GET  /api/demo/scene/<scan_id>/synthetic_views/<view_id>.png
      One view's bytes (owner-authed, like the keyframe route).

  POST /api/demo/scene/<scan_id>/synthetic_views/render
      Render a ring HEADLESSLY and register it through the same path the POST above
      uses. This is the whole point of ``splat_render.py``: until it existed the only
      producer of synthetic views was a human with the scene open in a browser, so an
      imported scene could not be given annotation evidence without one. A server render
      is NOT better evidence than a browser render — both are renders, both carry the
      ``synthetic view`` chip; the record simply notes which renderer drew the pixels.
      202 ``{job_id}``; poll ``GET /api/scenes/<id>/jobs/<job_id>``.

  POST /api/scenes/<scan_id>/ground_frame
      Fit the dominant horizontal plane and write the derived frame onto
      ``facts.metrics`` (``up_axis``/``floor_height``/``ceiling_height``/
      ``floor_extent``/``vertical_axis_known``/``derivation``). This exists as a route,
      not only as an import-time step, because the founder's scene is ALREADY imported:
      a 2 GB upload must never have to be re-sent to learn which way is up.

  GET  /api/scenes/<scan_id>/ground_frame
      What is currently recorded (no computation, no side effects).

Every route is owner-only (the ``server.oreos`` package docstring explains the gate).
Synthetic views are never written into ``keyframes`` — see ``synthetic_views.py`` for
why that separation is structural rather than cosmetic.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from flask import jsonify, request, send_file

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import ground_frame as ground_frame_mod
from server.oreos import jobs as demo_jobs
from server.oreos import planes as planes_mod
from server.oreos import synthetic_views as sv

# ---------------------------------------------------------------------------
# server-side renderer binding (injectable — routes_sam3d's _complete_fn pattern)
# ---------------------------------------------------------------------------

#: The Modal app that owns the rasterizer. Its OWN app, so deploying it never touches the
#: streaming deploy; the broker reaches it by name exactly as it reaches ``demo-sam3d``.
DEMO_RENDER_APP = os.environ.get("DEMO_RENDER_APP", "demo-splat-render")

_render_ring_fn: Any = None


def configure_render_ring_fn(fn: Any) -> None:
    """Inject the ring-render function (tests pass a fake with ``.remote``; ``None``
    resets to the lazy ``modal.Function.from_name`` binding)."""
    global _render_ring_fn
    _render_ring_fn = fn


def _get_render_ring_fn() -> Any:
    global _render_ring_fn
    if _render_ring_fn is not None:
        return _render_ring_fn
    import modal  # broker has modal; tests always inject

    _render_ring_fn = modal.Function.from_name(DEMO_RENDER_APP, "render_ring")
    return _render_ring_fn


def _gate(scan_id: str):
    """Shared owner+scene gate -> ``(user_id, store, record, error_response)``."""
    user_id = auth_user_id()
    if not user_id:
        return None, None, None, (jsonify({"error": "invalid_token"}), 401)
    store = scene_persistence()
    record = store.get_scene(user_id, scan_id) if store is not None else None
    if not record:
        return None, None, None, (jsonify({"error": "not_found"}), 404)
    return user_id, store, record, None


def _view_url(scan_id: str, view_id: str) -> str:
    return f"/api/demo/scene/{scan_id}/synthetic_views/{view_id}.png"


# ── synthetic views ──


@oreos_bp.route("/api/demo/scene/<scan_id>/synthetic_views", methods=["POST"])
def post_synthetic_views(scan_id: str):
    """Register rendered views for this scene (see module docstring)."""
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "expected a JSON object"}), 400

    try:
        parsed = sv.parse_views(body.get("views"))
    except sv.SyntheticViewError as exc:
        return jsonify(exc.payload()), exc.status

    replace = bool(body.get("replace"))
    existing = [] if replace else sv.manifest(record)
    if len(existing) + len(parsed) > sv.MAX_VIEWS:
        return (
            jsonify(
                {
                    "error": "too_many_views",
                    "detail": (
                        f"this scene already holds {len(existing)} synthetic views; adding "
                        f"{len(parsed)} would exceed the {sv.MAX_VIEWS} limit. POST with "
                        "{'replace': true} to swap the set instead of appending."
                    ),
                    "limit": sv.MAX_VIEWS,
                }
            ),
            400,
        )

    try:
        stored = sv.persist_views(store, user_id, scan_id, parsed, existing=existing)
    except sv.SyntheticViewError as exc:
        print(f"[demo.synthetic_views] store failed for {scan_id}: {exc.detail}")
        return jsonify(exc.payload()), exc.status

    return (
        jsonify(
            {
                "views": [{"view_id": m["view_id"], "index": m["index"]} for m in stored],
                "count": len(existing) + len(stored),
                "replaced": replace,
                "provenance": sv.SYNTHETIC_VIEW_PROVENANCE,
                "provenance_detail": sv.SYNTHETIC_VIEW_DETAIL,
            }
        ),
        200,
    )


@oreos_bp.route("/api/demo/scene/<scan_id>/synthetic_views", methods=["GET"])
def get_synthetic_views(scan_id: str):
    """This scene's synthetic-view manifest (metadata + fetch urls)."""
    _user_id, _store, record, err = _gate(scan_id)
    if err:
        return err
    views = sv.manifest(record)
    return jsonify(
        {
            "views": [sv.public_view(m, _view_url(scan_id, str(m["view_id"]))) for m in views],
            "count": len(views),
            "provenance": sv.SYNTHETIC_VIEW_PROVENANCE,
            "provenance_detail": sv.SYNTHETIC_VIEW_DETAIL,
        }
    )


@oreos_bp.route("/api/demo/scene/<scan_id>/synthetic_views/render", methods=["POST"])
def post_render_synthetic_views(scan_id: str):
    """Render a ring of views for this scene on the server and register them.

    Body (all optional): ``{replace?: bool, budget?: int, sh?: bool, ring_count?: int,
    width?: int, height?: int}``. Answers ``202 {job_id}`` — a ring is ~10 frames plus a
    one-off decode, comfortably longer than Modal's 150 s HTTP ceiling, and the splat
    import already learned that lesson the expensive way.

    The broker never rasterizes: it is 1 CPU / 4 GB and serves every request. This blocks
    a job thread on a remote call, the same shape ``routes_sam3d.complete_object`` uses."""
    user_id, _store, _record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "body must be a JSON object"}), 400

    def _int(name: str, lo: int, hi: int) -> Optional[int]:
        raw = body.get(name)
        if raw is None:
            return None
        value = int(raw)
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be within [{lo}, {hi}], got {value}")
        return value

    try:
        kwargs = {
            "budget": _int("budget", 10_000, 8_000_000),
            "ring_count": _int("ring_count", 1, sv.MAX_VIEWS),
            "elevated_count": _int("elevated_count", 0, sv.MAX_VIEWS),
            "width": _int("width", sv.MIN_DIMENSION, sv.MAX_DIMENSION),
            "height": _int("height", sv.MIN_DIMENSION, sv.MAX_DIMENSION),
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": "bad_request", "detail": str(exc)}), 400
    kwargs["want_sh"] = bool(body.get("sh", True))
    kwargs["replace"] = bool(body.get("replace", True))

    active = demo_jobs.any_active_scene_job(user_id, scan_id, kind_prefix="render_views")
    if active:
        return jsonify({"error": "job_conflict", "job_id": active}), 409

    job_id = demo_jobs.create_scene_job(
        user_id,
        scan_id,
        "render_views",
        note="rendering views on the server — a cold GPU container takes 30–90 s",
    )

    def _run() -> dict[str, Any]:
        out = _get_render_ring_fn().remote(user_id, scan_id, **kwargs)
        if not isinstance(out, dict) or not out.get("ok"):
            detail = (out or {}).get("detail") if isinstance(out, dict) else None
            raise RuntimeError(
                f"server-side render failed: {(out or {}).get('error') if isinstance(out, dict) else out}"
                f"{f' — {detail}' if detail else ''}"
            )
        # The log is for the container; the client gets the numbers it can act on.
        out.pop("log", None)
        out.pop("images", None)
        return out

    demo_jobs.run_scene_job(job_id, _run)
    return jsonify({"job_id": job_id, "scan_id": scan_id, "queued_at": time.time()}), 202


@oreos_bp.route("/api/demo/scene/<scan_id>/synthetic_views/<view_id>.png", methods=["GET"])
def get_synthetic_view_png(scan_id: str, view_id: str):
    """One view's PNG bytes. Served from the manifest's recorded key, never from a key
    built out of the URL — a forged ``view_id`` can only ever miss."""
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    match = next((m for m in sv.manifest(record) if str(m.get("view_id")) == view_id), None)
    if match is None:
        return jsonify({"error": "not_found"}), 404
    path = store.get_derived_artifact_path(user_id, scan_id, str(match.get("blob_key")))
    if not path:
        return jsonify({"error": "not_found"}), 404
    response = send_file(path, mimetype="image/png", conditional=True)
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


# ── ground frame ──


def _recorded_frame(record: dict[str, Any]) -> dict[str, Any]:
    metrics = ((record.get("facts") or {}).get("metrics")) or {}
    return {
        "vertical_axis_known": bool(metrics.get("vertical_axis_known", False)),
        "derivation": metrics.get("derivation"),
        "up_axis": metrics.get("up_axis"),
        "floor_height": metrics.get("floor_height"),
        "ceiling_height": metrics.get("ceiling_height"),
        "room_height": metrics.get("room_height"),
        "floor_extent": metrics.get("floor_extent"),
        "floor_area": metrics.get("floor_area"),
        "note": metrics.get("ground_frame_note"),
        "units": "relative",
        "units_basis": "slam_world_units",
    }


@oreos_bp.route("/api/scenes/<scan_id>/ground_frame", methods=["GET"])
def get_ground_frame(scan_id: str):
    """What ``facts.metrics`` currently records — no fit, no writes."""
    _user_id, _store, record, err = _gate(scan_id)
    if err:
        return err
    return jsonify({"scan_id": scan_id, "frame": _recorded_frame(record), "computed": False})


@oreos_bp.route("/api/scenes/<scan_id>/ground_frame", methods=["POST"])
def post_ground_frame(scan_id: str):
    """(Re)compute the ground frame and persist it onto ``facts.metrics``.

    Body (all optional): ``{up_override?: [3], max_points?: int, dry_run?: bool}``.
    ``dry_run`` returns the fit without writing — how you inspect a weak fit before
    letting it change what the scene claims."""
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "body must be a JSON object"}), 400

    max_points = body.get("max_points", 400_000)
    if not isinstance(max_points, int) or not (10_000 <= max_points <= 2_000_000):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "detail": "max_points must be an int in [10000, 2000000]",
                }
            ),
            400,
        )

    cloud = store.get_cloud(user_id, scan_id)
    if cloud is None:
        return jsonify({"error": "no_geometry", "detail": "scan has no stored point cloud"}), 422
    positions, _colors = cloud

    trajectory = store.get_trajectory(user_id, scan_id)
    poses = trajectory.get("poses") if isinstance(trajectory, dict) else None
    try:
        frame = ground_frame_mod.compute_ground_frame(
            positions, poses=poses, up_override=body.get("up_override"), max_points=max_points
        )
    except planes_mod.PathplanUnavailable as exc:
        return jsonify({"error": "pathplan_not_merged", "detail": str(exc)}), 503
    except ground_frame_mod.GroundFrameError as exc:
        return jsonify({"error": exc.code, "detail": exc.detail}), 422
    except ValueError as exc:
        return jsonify({"error": "bad_request", "detail": str(exc)}), 400

    patch = ground_frame_mod.metrics_patch(frame)
    if body.get("dry_run"):
        return jsonify({"scan_id": scan_id, "frame": frame, "metrics_patch": patch, "computed": True, "persisted": False})

    try:
        metrics = store.update_scene_metrics(user_id, scan_id, patch)
    except KeyError:
        return jsonify({"error": "not_found"}), 404

    return jsonify(
        {
            "scan_id": scan_id,
            "frame": frame,
            "metrics": metrics,
            "computed": True,
            "persisted": True,
            "parent_artifact": record.get("points_key"),
            "created_at": time.time(),
        }
    )
