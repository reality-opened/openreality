"""W5 demo routes: plane candidates (F5 restyle) + floor patch (F6 swap).

  POST /api/scenes/<scan_id>/planes
      Floor + wall plane candidates for the restyle surface picker. Floor via
      W4's pathplan seam (``planes.fit_floor``); walls via the constrained
      sequential RANSAC in ``server/oreos/planes.py``. Body (all optional):
      ``{force?: bool, up_override?: [3], max_walls?: int<=6}``. Answers 503
      ``pathplan_not_merged`` until W4's ``demo/pathplan.py`` lands (branch
      feat/demo-pathplan) — the client treats that as "detect unavailable".
      Results are cached in-process per scene (recompute with ``force``).

  POST /api/scenes/<scan_id>/objects/<uid>/floor_patch
      F6 hole-fill: synthesize ~2k floor-colored gaussians under a hidden
      object's OBB footprint (features.md F6 ladder step 2) and persist them as
      ``derived/demo/objects/<uid>/floor_patch/patch.ply`` (+ ``meta.json``
      honesty envelope, ``generated: true``). Body: ``{obb: {center, extents,
      rotation}, n_gaussians?, seed?}`` — the OBB comes from the client's hide
      region (the swap panel owns inflation). Sync, well under the 2 s budget
      at the 8k-gaussian cap. Served back by the existing derived GET route;
      idempotent per uid (re-patching overwrites, never duplicates).

Both routes answer in WORLD coordinates and SLAM units (``units:
"relative"``, ``units_basis: "slam_world_units"``) per the world-transform
contract; the metric anchor is a display concern.

Lives in W5's file (not W3's ``routes_sam3d.py``) by the build plan's file-
ownership split, even though the floor-patch path nests under ``objects/``.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import planes as planes_mod
from server.oreos.planes import InsufficientFloorContext, PathplanUnavailable

_FIXED_PROVENANCE = (
    "RANSAC plane candidates on the reconstruction point cloud — "
    "human-confirmed before any edit applies"
)

_FLOOR_PATCH_CAVEAT = (
    "Synthesized floor-colored gaussians under the removed object — "
    "the background behind the original was never captured."
)

_MAX_PATCH_GAUSSIANS = 8000
_UID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")

# In-process plane cache: (user_id, scan_id) -> response dict. Safe for the
# demo topology (broker max_containers=1, one operator, many threads) — same
# reasoning as routes_docs' manifest lock.
_PLANES_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_PLANES_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pathplan_503():
    return (
        jsonify(
            {
                "error": "pathplan_not_merged",
                "detail": (
                    "the floor plane comes from W4's server/oreos/pathplan.py "
                    "(branch feat/demo-pathplan); this route activates when it merges"
                ),
            }
        ),
        503,
    )


def _auth_and_scene(scan_id: str):
    """-> (user_id, store, record, error_response). Standard demo-route gate."""
    user_id = auth_user_id()
    if not user_id:
        return None, None, None, (jsonify({"error": "invalid_token"}), 401)
    store = scene_persistence()
    record = store.get_scene(user_id, scan_id) if store is not None else None
    if not record:
        return None, None, None, (jsonify({"error": "not_found"}), 404)
    return user_id, store, record, None


def _parse_obb(raw: Any) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Validate the contract OBB {center[3], extents[3] FULL, rotation 3x3
    columns=axes} -> (center, extents, rotation) or None."""
    if not isinstance(raw, dict):
        return None
    try:
        center = np.asarray(raw.get("center"), dtype=np.float64).reshape(3)
        extents = np.asarray(raw.get("extents"), dtype=np.float64).reshape(3)
        rotation = np.asarray(raw.get("rotation"), dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(center).all() and np.isfinite(extents).all() and np.isfinite(rotation).all()):
        return None
    if (extents <= 0).any():
        return None
    # Columns must be ~orthonormal (a rotation, possibly with tiny float drift).
    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-2):
        return None
    return center, extents, rotation


@oreos_bp.route("/api/scenes/<scan_id>/planes", methods=["POST"])
def detect_planes(scan_id: str):
    """Floor + wall candidates for the F5 surface picker (see module docstring)."""
    user_id, store, record, err = _auth_and_scene(scan_id)
    if err:
        return err

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "body must be a JSON object"}), 400
    force = bool(body.get("force"))

    cache_key = (user_id, scan_id)
    if not force:
        with _PLANES_LOCK:
            cached = _PLANES_CACHE.get(cache_key)
        if cached is not None:
            return jsonify({**cached, "cached": True})

    cloud = store.get_cloud(user_id, scan_id)
    if cloud is None:
        return jsonify({"error": "no_geometry", "detail": "scan has no stored point cloud"}), 422

    positions, _colors = cloud
    points, _ = planes_mod.downsample_points(positions, None)

    max_walls = body.get("max_walls", planes_mod.MAX_WALL_PLANES)
    if not isinstance(max_walls, int) or not (1 <= max_walls <= planes_mod.MAX_WALL_PLANES):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "detail": f"max_walls must be an int in [1, {planes_mod.MAX_WALL_PLANES}]",
                }
            ),
            400,
        )

    trajectory = store.get_trajectory(user_id, scan_id)
    poses = trajectory.get("poses") if isinstance(trajectory, dict) else None
    try:
        up, up_source = planes_mod.estimate_up(points, poses=poses, up_override=body.get("up_override"))
    except ValueError as exc:
        return jsonify({"error": "bad_request", "detail": str(exc)}), 400

    thresh = planes_mod.default_thresh(points)
    try:
        floor_plane = planes_mod.fit_floor(points, up, thresh=thresh)
    except PathplanUnavailable:
        return _pathplan_503()
    except ValueError as exc:
        return jsonify({"error": "no_floor", "detail": str(exc)}), 422

    floor = planes_mod.floor_candidate(points, floor_plane, thresh)
    # The refit floor normal is the best up estimate — constrain walls against it.
    wall_up = np.asarray(floor["normal"], dtype=np.float64)
    walls = planes_mod.detect_walls(points, wall_up, max_planes=max_walls, thresh=thresh)

    response = {
        "planes": [floor] + walls,
        "up": [float(x) for x in up],
        "up_source": up_source,
        "thresh": float(thresh),
        "units": "relative",
        "units_basis": "slam_world_units",
        "provenance": _FIXED_PROVENANCE,
        "scan_id": scan_id,
        "parent_artifact": record.get("points_key"),
        "created_at": _now_iso(),
    }
    with _PLANES_LOCK:
        _PLANES_CACHE[cache_key] = response
    return jsonify({**response, "cached": False})


@oreos_bp.route("/api/scenes/<scan_id>/objects/<uid>/floor_patch", methods=["POST"])
def floor_patch(scan_id: str, uid: str):
    """Synthesize + persist the F6 floor patch for one object (module docstring)."""
    user_id, store, record, err = _auth_and_scene(scan_id)
    if err:
        return err

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "body must be a JSON object"}), 400
    parsed = _parse_obb(body.get("obb"))
    if parsed is None:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "detail": (
                        "obb required: {center:[3], extents:[3] full lengths > 0, "
                        "rotation: 3x3 columns=axes} in world frame"
                    ),
                }
            ),
            400,
        )
    obb_center, obb_extents, obb_rotation = parsed

    n_gaussians = body.get("n_gaussians", 2000)
    if not isinstance(n_gaussians, int) or not (1 <= n_gaussians <= _MAX_PATCH_GAUSSIANS):
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "detail": f"n_gaussians must be an int in [1, {_MAX_PATCH_GAUSSIANS}]",
                }
            ),
            400,
        )
    seed = body.get("seed", 0)
    if not isinstance(seed, int):
        return jsonify({"error": "bad_request", "detail": "seed must be an int"}), 400

    safe_uid = _UID_SAFE_RE.sub("-", uid)[:64]
    if not safe_uid or safe_uid in (".", ".."):
        return jsonify({"error": "bad_request", "detail": "invalid object uid"}), 400

    cloud = store.get_cloud(user_id, scan_id)
    if cloud is None:
        return jsonify({"error": "no_geometry", "detail": "scan has no stored point cloud"}), 422
    positions, colors = cloud
    positions, colors = planes_mod.downsample_points(positions, colors)
    colors01 = np.asarray(colors, dtype=np.float64) / 255.0

    trajectory = store.get_trajectory(user_id, scan_id)
    poses = trajectory.get("poses") if isinstance(trajectory, dict) else None
    try:
        up, up_source = planes_mod.estimate_up(positions, poses=poses, up_override=body.get("up_override"))
    except ValueError as exc:
        return jsonify({"error": "bad_request", "detail": str(exc)}), 400

    thresh = planes_mod.default_thresh(positions)
    try:
        floor_plane = planes_mod.fit_floor(positions, up, thresh=thresh)
    except PathplanUnavailable:
        return _pathplan_503()
    except ValueError as exc:
        return jsonify({"error": "no_floor", "detail": str(exc)}), 422

    try:
        fields, info = planes_mod.synthesize_floor_patch(
            positions,
            colors01,
            floor_plane,
            obb_center,
            obb_extents,
            obb_rotation,
            n_gaussians=n_gaussians,
            seed=seed,
        )
    except InsufficientFloorContext as exc:
        return jsonify({"error": "insufficient_floor_context", "detail": str(exc)}), 422

    from server.scene_report.splat_io import serialize_splat_ply

    ply_bytes = serialize_splat_ply(fields)
    created_at = _now_iso()
    meta = {
        # Honesty envelope (object-layer pattern) — the client's chip copy keys
        # off generator == "floor_patch" ("generated floor patch").
        "provenance": "generated floor patch",
        "generated": True,
        "generator": "floor_patch",
        "quality": "usable",
        "caveats": [_FLOOR_PATCH_CAVEAT],
        "inputs": {
            "object_uid": uid,
            "obb": {
                "center": [float(x) for x in obb_center],
                "extents": [float(x) for x in obb_extents],
                "rotation": [[float(v) for v in row] for row in obb_rotation],
            },
            "up_source": up_source,
            "seed": seed,
            **info,
        },
        # Versioning stamps (world-transform contract).
        "scan_id": scan_id,
        "parent_artifact": record.get("points_key"),
        "created_at": created_at,
        "units": "relative",
        "units_basis": "slam_world_units",
    }
    import json as _json

    base = f"demo/objects/{safe_uid}/floor_patch"
    patch_key = store.save_derived_artifact(user_id, scan_id, f"{base}/patch.ply", ply_bytes)
    meta_key = store.save_derived_artifact(
        user_id,
        scan_id,
        f"{base}/meta.json",
        _json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return jsonify(
        {
            "patch_key": patch_key,
            "meta_key": meta_key,
            "n_gaussians": info["n_gaussians"],
            "n_annulus_points": info["n_annulus_points"],
            "footprint_uv": info["footprint_uv"],
            "units": "relative",
            "units_basis": "slam_world_units",
            "meta": meta,
        }
    )
