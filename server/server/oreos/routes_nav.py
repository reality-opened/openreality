"""F4 nav routes: ``POST /api/scenes/<scan_id>/nav/plan`` (W4, docs/demo-2026-07).

HTTP glue over the pure planner in ``server/oreos/pathplan.py`` (the EXP-25 port —
read its module docstring for frames/units/determinism). This module owns:

  - geometry loading (persisted ``cloud.npz``; imported splats fall back to the
    splat means, cached at ``derived/demo/nav/cloud_from_splat.npz`` — the F3
    geometry pipeline shares that cache, features.md §0.2);
  - the per-scene context cache ladder: in-process dict → the durable
    ``derived/demo/nav/grid_cache.npz`` → cold compute (5–15 s once; <1 s after);
  - goal resolution ({point_world} | {object_uid "det:<idx>"} → facts.objects);
  - the path doc (``derived/demo/nav/paths/<pid>.json``, shell.md §3 shape +
    transform-contract stamps) indexed into the demo manifest through
    ``routes_docs``'s OWN lock + helpers (one lock per process — never a second
    read-modify-write path);
  - the ``?debug=1`` top-down occupancy PNG at
    ``derived/demo/nav/paths/<pid>_debug.png`` (founder eyeball loop);
  - the agent tool seam: ``plan_path_for_scene(user_id, scan_id, body)`` is the
    exact function W2's ``plan_path`` tool calls — same contract as the route,
    no flask objects in or out.

Auth mirrors the package posture (see ``server/oreos/__init__``): the app-level
hook has already vetted the bearer token; handlers re-check ``auth_user_id()``.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos.pathplan import (
    CLOUD_FROM_SPLAT_RELATIVE_KEY,
    DEFAULT_SEED,
    ENGINE_NAME,
    GRID_CACHE_RELATIVE_KEY,
    NavContext,
    NavError,
    PROVENANCE_LINE,
    PlanResult,
    UnitsInfo,
    _grid_params_match,
    build_nav_context,
    context_reusable,
    deserialize_nav_context,
    plan_route,
    rebuild_grid,
    render_debug_png,
    resolve_params,
    resolve_units,
    rotation_to_quat,
    serialize_nav_context,
)

# Shared manifest machinery — the ONE process-wide lock + read/index helpers that
# routes_docs' PUT writer uses; importing them (same package) keeps every
# manifest read-modify-write serialized through a single lock instance.
from server.oreos.routes_docs import (  # noqa: F401  (shared-lock import is the point)
    _MANIFEST_LOCK,
    _MANIFEST_RELATIVE_KEY,
    _index_into_manifest,
    _read_manifest,
)

# ---------------------------------------------------------------------------
# In-process context cache (broker deploys max_containers=1; threads share it)
# ---------------------------------------------------------------------------

_CTX_LOCK = threading.Lock()
_CTX_CACHE: dict[tuple[str, str], NavContext] = {}

_STATUS_FOR_CODE = {
    "bad_request": 400,
    "unknown_object": 404,
    "no_geometry": 404,
    "no_floor": 422,
    "unreachable_goal": 422,
}


def reset_nav_cache() -> None:
    """Drop every in-process NavContext (tests; and a lever for ops if a scene's
    geometry is ever re-persisted in place)."""
    with _CTX_LOCK:
        _CTX_CACHE.clear()


# ---------------------------------------------------------------------------
# Scene-record helpers
# ---------------------------------------------------------------------------

def _resolve_anchor(record: dict[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """The scene's metric anchor, if any: ``derived_latest`` of kind "anchor" with a
    positive ``scale_factor`` (metres per SLAM unit) → ``(scale_factor, source_key)``."""
    pointer = record.get("derived_latest")
    if not isinstance(pointer, dict) or pointer.get("kind") != "anchor":
        return None, None
    try:
        scale = float(pointer.get("scale_factor") or 0.0)
    except (TypeError, ValueError):
        return None, None
    if scale <= 0 or not np.isfinite(scale):
        return None, None
    source = pointer.get("source_key")
    return scale, (str(source) if source else None)


def _load_geometry(store: Any, user_id: str, scan_id: str) -> tuple[np.ndarray, str]:
    """World-frame planning cloud → ``(positions (N,3) float64, parent_artifact)``.

    Ladder: persisted ``cloud.npz`` → cached splat-means cloud
    (``derived/demo/nav/cloud_from_splat.npz``) → splat.ply means (read once via
    ``splat_io.read_splat_ply``, then cached for F3/F4 reuse). ``no_geometry`` if
    the scene has none of these."""
    cloud = store.get_cloud(user_id, scan_id)
    if cloud is not None:
        positions, _colors = cloud
        return np.asarray(positions, dtype=np.float64).reshape(-1, 3), "cloud.npz"

    derived_key = f"derived/{CLOUD_FROM_SPLAT_RELATIVE_KEY}"
    cached_path = store.get_derived_artifact_path(user_id, scan_id, derived_key)
    if cached_path:
        try:
            with np.load(cached_path, allow_pickle=False) as data:
                return np.asarray(data["positions"], dtype=np.float64).reshape(-1, 3), "splat.ply"
        except Exception as exc:  # corrupt cache degrades to re-derive, never a 500
            print(f"[demo.nav] unreadable cloud_from_splat cache for {scan_id}: {exc}")

    splat_path = None
    if hasattr(store, "get_splat_path"):
        splat_path = store.get_splat_path(user_id, scan_id)
    if splat_path:
        from server.scene_report.splat_io import read_splat_ply  # lazy: numpy-only module

        fields = read_splat_ply(splat_path)
        if not all(k in fields for k in ("x", "y", "z")):
            raise NavError("no_geometry", "stored splat.ply lacks x/y/z means")
        positions = np.stack(
            [fields["x"], fields["y"], fields["z"]], axis=1
        ).astype(np.float32)
        try:
            import io as _io

            buf = _io.BytesIO()
            np.savez_compressed(buf, positions=positions)
            store.save_derived_artifact(
                user_id, scan_id, CLOUD_FROM_SPLAT_RELATIVE_KEY, buf.getvalue()
            )
        except Exception as exc:  # best-effort cache — planning proceeds regardless
            print(f"[demo.nav] cloud_from_splat cache write failed for {scan_id}: {exc}")
        return positions.astype(np.float64), "splat.ply"

    raise NavError(
        "no_geometry",
        "scene has no stored point cloud and no splat to derive one from",
    )


def _load_poses(store: Any, user_id: str, scan_id: str) -> Optional[np.ndarray]:
    """Capture-camera c2w poses ``(N,4,4)`` from ``trajectory.npz``, or ``None``.
    (W1 is re-persisting trajectories for the canonical scene — a missing block
    downgrades the up ladder honestly rather than failing.)"""
    try:
        block = store.get_trajectory(user_id, scan_id)
    except Exception as exc:
        print(f"[demo.nav] trajectory read failed for {scan_id}: {exc}")
        return None
    if not block:
        return None
    poses = np.asarray(block.get("poses"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not poses.size:
        return None
    return poses


def _object_footprint_slam(obj: dict[str, Any]) -> float:
    """Half-diagonal of a detected object's ``extent`` (FULL lengths, world axes) → the extra
    standoff the planner may use when approaching it.

    An object goal is the object's CENTRE, which by construction lies inside the object's own
    occupied volume — so the planner always has to park some distance away. How far depends on
    how big the thing is: a desk needs more room than a mug. Missing/malformed extents degrade
    to 0.0 (the flat approach radius still applies)."""
    extent = obj.get("extent")
    try:
        vec = np.asarray(extent, dtype=np.float64).reshape(3)
    except Exception:
        return 0.0
    if not np.isfinite(vec).all():
        return 0.0
    return float(0.5 * np.linalg.norm(np.abs(vec)))


def _resolve_goal(
    record: dict[str, Any], body: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], float]:
    """``goal.point_world`` | ``goal.object_uid`` ("det:<idx>", §0.2 identity scheme) →
    ``(world-frame goal point, doc/goal metadata, object footprint radius in SLAM units)``.
    NavError(bad_request|unknown_object)."""
    goal = body.get("goal")
    if not isinstance(goal, dict):
        raise NavError("bad_request", "expected goal:{point_world:[x,y,z]} or goal:{object_uid}")

    uid = goal.get("object_uid")
    if uid is not None:
        if not isinstance(uid, str) or not uid.startswith("det:"):
            raise NavError(
                "bad_request",
                "object_uid goals support the 'det:<idx>' scheme (facts.objects index); "
                "sel:/layer: goals arrive with the F3 selection store",
            )
        try:
            idx = int(uid.split(":", 1)[1])
        except ValueError:
            raise NavError("bad_request", f"malformed object_uid {uid!r}")
        objects = (record.get("facts") or {}).get("objects") or []
        if idx < 0 or idx >= len(objects):
            raise NavError(
                "unknown_object",
                f"{uid} does not exist (scene has {len(objects)} detected objects)",
            )
        obj = objects[idx] or {}
        center = obj.get("center")
        try:
            point = np.asarray(center, dtype=np.float64).reshape(3)
        except Exception:
            raise NavError("unknown_object", f"{uid} has no usable world center")
        if not np.isfinite(point).all():
            raise NavError("unknown_object", f"{uid} has a non-finite world center")
        label = obj.get("human_label") or obj.get("query") or uid
        footprint = _object_footprint_slam(obj)
        return (
            point,
            {
                "object_uid": uid,
                "label": str(label),
                "point_world": [float(x) for x in point],
                "footprint_radius_slam": footprint,
            },
            footprint,
        )

    point_raw = goal.get("point_world")
    try:
        point = np.asarray(point_raw, dtype=np.float64).reshape(3)
    except Exception:
        raise NavError("bad_request", "goal.point_world must be [x, y, z]")
    if not np.isfinite(point).all():
        raise NavError("bad_request", "goal.point_world must be finite")
    return (
        point,
        {"object_uid": None, "label": None, "point_world": [float(x) for x in point]},
        0.0,
    )


def _resolve_start(body: dict[str, Any], ctx: NavContext) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Optional explicit start ({point_world} or bare [x,y,z]); default = the last
    capture-camera position projected to the floor ("camera-below", features.md F4);
    scenes with no poses fall through to plan_route's free-area-center default."""
    start = body.get("start")
    if start is not None:
        raw = start.get("point_world") if isinstance(start, dict) else start
        try:
            point = np.asarray(raw, dtype=np.float64).reshape(3)
        except Exception:
            raise NavError("bad_request", "start must be {point_world:[x,y,z]} or [x,y,z]")
        if not np.isfinite(point).all():
            raise NavError("bad_request", "start point must be finite")
        return point, "request"
    if ctx.last_cam_world is not None:
        return np.asarray(ctx.last_cam_world, dtype=np.float64), "camera_below"
    return None, None


# ---------------------------------------------------------------------------
# Context cache ladder
# ---------------------------------------------------------------------------

def _get_context(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    body_params: Optional[dict[str, Any]],
    anchor_scale: Optional[float],
    anchor_key: Optional[str],
) -> tuple[NavContext, UnitsInfo, dict[str, float], dict[str, float], dict[str, Any]]:
    """memory → volume npz → compute; grid-only rebuild when just the grid params moved.
    → ``(ctx, units_info, params_slam, params_units, cache_info)``."""
    raw = dict(body_params or {})
    seed = raw.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise NavError("bad_request", "param 'seed' must be a non-negative integer")
    up_override = raw.get("up_override")
    has_poses = bool(record.get("trajectory_key"))

    cache_source = None
    with _CTX_LOCK:
        ctx = _CTX_CACHE.get((user_id, scan_id))
    if ctx is not None and context_reusable(ctx, seed, up_override, has_poses):
        cache_source = "memory"
    else:
        ctx = None

    if ctx is None:
        blob = store.get_derived_artifact(user_id, scan_id, f"derived/{GRID_CACHE_RELATIVE_KEY}")
        if blob:
            cached = deserialize_nav_context(blob)
            if cached is not None and context_reusable(cached, seed, up_override, has_poses):
                ctx = cached
                cache_source = "volume"

    grid_rebuilt = False
    if ctx is not None:
        units_info = resolve_units(
            anchor_scale, anchor_key, ctx.capture_height_slam, ctx.vextent_slam
        )
        params_slam, params_units, _ = resolve_params(raw, units_info)
        if not _grid_params_match(ctx.grid_params_slam, params_slam):
            ctx = rebuild_grid(ctx, params_slam)
            grid_rebuilt = True
    else:
        cache_source = "computed"
        points, parent_artifact = _load_geometry(store, user_id, scan_id)
        poses = _load_poses(store, user_id, scan_id) if has_poses else None
        if has_poses and poses is None:
            # Metadata promised a trajectory the volume could not deliver — degrade the
            # ladder honestly (the up-source note says so) instead of 500ing.
            has_poses = False
        ctx, units_info, params_slam, params_units = build_nav_context(
            points,
            poses,
            body_params=raw,
            anchor_scale_factor=anchor_scale,
            anchor_key=anchor_key,
            parent_artifact=parent_artifact,
        )

    if cache_source == "computed" or grid_rebuilt:
        try:
            store.save_derived_artifact(
                user_id, scan_id, GRID_CACHE_RELATIVE_KEY, serialize_nav_context(ctx, scan_id)
            )
        except Exception as exc:  # cache is an accelerator, never a correctness gate
            print(f"[demo.nav] grid cache write failed for {scan_id}: {exc}")

    with _CTX_LOCK:
        _CTX_CACHE[(user_id, scan_id)] = ctx

    return ctx, units_info, params_slam, params_units, {
        "source": cache_source,
        "grid_rebuilt": grid_rebuilt,
    }


# ---------------------------------------------------------------------------
# Doc + response assembly
# ---------------------------------------------------------------------------

def _new_path_id() -> str:
    import os

    return f"p{int(time.time() * 1000):x}{os.urandom(2).hex()}"


def _build_payload_and_doc(
    scan_id: str,
    pid: str,
    ctx: NavContext,
    units_info: UnitsInfo,
    params_slam: dict[str, float],
    params_units: dict[str, float],
    result: PlanResult,
    goal_info: dict[str, Any],
    start_source: Optional[str],
    cache_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """→ (route/tool response payload, path doc for derived/demo/nav/paths/<pid>.json).

    The doc follows shell.md §3 (planner block, waypoints, frames[{t,pos,quat}]) with the
    transform contract's units vocabulary + staleness stamps; the response additionally
    carries full c2w matrices so the client never reassembles rotations from quats."""
    rep = units_info.report_per_slam
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    notes = list(ctx.notes) + list(result.notes)
    if units_info.note:
        notes.append(units_info.note)

    floor_block = {
        "point": [float(x) for x in ctx.plane.point],
        "normal": [float(x) for x in ctx.plane.normal],
        "up": [float(x) for x in ctx.up_vec],
        "up_source": ctx.up_source,
    }
    grid_block = {
        "n_free": int(ctx.grid.free.sum()),
        "n_components": int(ctx.n_components),
        "n_navigable": int(ctx.navigable.sum()),
        "cell_size": float(ctx.grid.cell_size),  # scene (SLAM) units — a world-geometry quantity
        "cell_size_units": float(ctx.grid.cell_size * rep),  # in the reported units
    }
    stats_block = {
        "path_length": float(result.path_length_slam * rep),
        "path_length_slam": float(result.path_length_slam),
        "duration_s": float(result.times[-1]) if result.times.size else 0.0,
        "n_frames": int(result.c2ws.shape[0]),
        "goal_snap": float(result.goal_snap_slam * rep),
        "start_snap": float(result.start_snap_slam * rep),
    }
    # Did the robot drive TO the clicked point, or park BESIDE it? Object goals are always
    # the latter (the centre of a thing is inside the thing) — the client renders the
    # standoff honestly rather than pretending the robot reached the object itself.
    approach_block = {
        "mode": result.goal_mode,  # "direct" | "approach"
        "standoff": float(result.goal_snap_slam * rep),
        "approach_radius": float(result.approach_radius_slam * rep),
        "requested_point_world": (
            [float(x) for x in result.goal_requested_world]
            if result.goal_requested_world is not None
            else None
        ),
    }

    payload = {
        "path_id": pid,
        "scan_id": str(scan_id),
        "parent_artifact": ctx.parent_artifact,
        "created_at": created_at,
        "waypoints_world": [[float(v) for v in row] for row in result.waypoints_world],
        "poses": [
            {"c2w": [[float(v) for v in r] for r in c2w], "t": float(t)}
            for c2w, t in zip(result.c2ws, result.times)
        ],
        "floor": floor_block,
        "grid": grid_block,
        "units": units_info.units,
        "units_basis": units_info.units_basis,
        "stats": stats_block,
        "start_world": [float(x) for x in result.start_world],
        "start_source": start_source or "free_area_center",
        "goal_world": [float(x) for x in result.goal_world],
        "goal_requested": goal_info,
        "goal_approach": approach_block,
        "params": params_units,
        "notes": notes,
        "cache": cache_info,
        "provenance": PROVENANCE_LINE,
    }

    label = goal_info.get("label") or "point goal"
    doc = {
        "version": 1,
        "label": f"path to {label}",
        "scan_id": str(scan_id),
        "parent_artifact": ctx.parent_artifact,
        "created_at": created_at,
        "run_id": None,
        "path_id": pid,
        "planner": {
            "engine": ENGINE_NAME,
            "seed": int(params_slam["seed"]),
            "params": params_units,
            "params_slam": {k: float(v) for k, v in params_slam.items()},
            "up": {"vector": floor_block["up"], "source": ctx.up_source},
            "floor_plane": {"point": floor_block["point"], "normal": floor_block["normal"]},
            "notes": notes,
        },
        "start": {"point_world": payload["start_world"], "source": payload["start_source"]},
        "goal": goal_info,
        "goal_approach": approach_block,
        "waypoints": payload["waypoints_world"],
        "frames": [
            {
                "t": float(t),
                "pos": [float(v) for v in c2w[:3, 3]],
                "quat": rotation_to_quat(c2w[:3, :3]),  # [w,x,y,z], OpenCV c2w
            }
            for c2w, t in zip(result.c2ws, result.times)
        ],
        "units": units_info.units,
        "units_basis": units_info.units_basis,
        "stats": stats_block,
        "grid": grid_block,
        "provenance": PROVENANCE_LINE,
    }
    return payload, doc


def _persist_path_doc(store: Any, user_id: str, scan_id: str, pid: str, doc: dict[str, Any]) -> Optional[str]:
    """Write the path doc + index it into the demo manifest (routes_docs' lock/helpers —
    exactly the PUT writer's flow). Best-effort: a store hiccup degrades to doc_key=None
    with a note, never a failed plan."""
    suffix = f"nav/paths/{pid}.json"
    try:
        with _MANIFEST_LOCK:
            derived_key = store.save_derived_artifact(
                user_id,
                scan_id,
                f"demo/{suffix}",
                json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            manifest = _read_manifest(store, user_id, scan_id)
            _index_into_manifest(manifest, suffix, derived_key)
            store.save_derived_artifact(
                user_id,
                scan_id,
                _MANIFEST_RELATIVE_KEY,
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        return derived_key
    except Exception as exc:
        print(f"[demo.nav] path doc persistence failed for {scan_id}/{pid}: {exc}")
        return None


# ---------------------------------------------------------------------------
# The plan entry point (route handler + W2 tool seam)
# ---------------------------------------------------------------------------

def plan_path_for_scene(
    user_id: str,
    scan_id: str,
    body: dict[str, Any],
    *,
    debug: bool = False,
    store: Any = None,
) -> tuple[dict[str, Any], int]:
    """Plan one path on a persisted scene → ``(json-able payload, http-ish status)``.

    THE seam for W2's ``plan_path`` agent tool (and the route handler below):

        payload, status = plan_path_for_scene(
            user_id, scan_id,
            {"goal": {"object_uid": "det:3"} | {"point_world": [x, y, z]},
             "start": {"point_world": [x, y, z]}?,          # optional
             "params": {"clearance"|"band_lo"|"band_hi"|"eye_height"|"speed":  <scene units>,
                        "fps": Hz, "max_ang_vel_deg": deg/s, "seed": int,
                        "up_override": "+y"|"-y"|"+z"|"-z"|[x,y,z]}?},
            debug=False)
        # status 200 → the F4 response contract; else {"error": code, "detail", ...}
        # codes: bad_request 400 · unknown_object/no_geometry 404 · no_floor/unreachable_goal 422

    No flask involved; ``store`` defaults to the live ``scene_persistence()``.
    """
    store = store if store is not None else scene_persistence()
    if store is None:
        return {"error": "no_geometry", "detail": "no durable scene store configured"}, 404
    record = store.get_scene(user_id, scan_id)
    if not record:
        return {"error": "not_found"}, 404
    if not isinstance(body, dict):
        return {"error": "bad_request", "detail": "expected a JSON object body"}, 400

    try:
        goal_world, goal_info, goal_footprint = _resolve_goal(record, body)
        anchor_scale, anchor_key = _resolve_anchor(record)
        body_params = body.get("params")
        if body_params is not None and not isinstance(body_params, dict):
            raise NavError("bad_request", "params must be an object")
        ctx, units_info, params_slam, params_units, cache_info = _get_context(
            store, user_id, scan_id, record, body_params, anchor_scale, anchor_key
        )
        start_world, start_source = _resolve_start(body, ctx)
        result = plan_route(
            ctx, goal_world, start_world, params_slam, goal_footprint_slam=goal_footprint
        )
    except NavError as err:
        payload = {"error": err.code, "detail": err.detail}
        payload.update(err.extra)
        return payload, _STATUS_FOR_CODE.get(err.code, 400)

    pid = _new_path_id()
    payload, doc = _build_payload_and_doc(
        scan_id, pid, ctx, units_info, params_slam, params_units,
        result, goal_info, start_source, cache_info,
    )
    doc_key = _persist_path_doc(store, user_id, scan_id, pid, doc)
    payload["doc_key"] = doc_key
    if doc_key is None:
        payload["notes"].append("path doc persistence failed — this plan is not replayable")

    if debug:
        try:
            objects = [
                {"center": o.get("center"), "label": o.get("human_label") or o.get("query")}
                for o in ((record.get("facts") or {}).get("objects") or [])
                if isinstance(o, dict) and isinstance(o.get("center"), (list, tuple))
                and len(o.get("center")) == 3
            ]
            png = render_debug_png(
                ctx, result, title=f"{scan_id} · {pid} · {units_info.units_basis}",
                objects=objects,
            )
            payload["debug_key"] = store.save_derived_artifact(
                user_id, scan_id, f"demo/nav/paths/{pid}_debug.png", png
            )
        except Exception as exc:  # debug render is a diagnostic, never a failure mode
            print(f"[demo.nav] debug render failed for {scan_id}/{pid}: {exc}")
            payload["debug_key"] = None
            payload["notes"].append(f"debug render unavailable: {exc}")

    return payload, 200


# W4: nav routes -------------------------------------------------------------


@oreos_bp.route("/api/scenes/<scan_id>/nav/plan", methods=["POST"])
def nav_plan(scan_id: str):
    """F4: plan a robot path in the scene's scanned free space (BUILD-PLAN "Routes")."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "expected a JSON object body"}), 400
    debug = request.args.get("debug") == "1"
    payload, status = plan_path_for_scene(user_id, scan_id, body, debug=debug)
    return jsonify(payload), status
