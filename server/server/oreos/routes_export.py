"""W6 — export-surface manifest route (demo blueprint).

``GET /api/scenes/<scan_id>/demo/export/manifest?format=openreality|groot_lerobot_v2|isaac_usd``

A DRY-RUN of the existing zip export (``server/app.py::export_scene_route``): it runs the
SAME builders (``export_from_record`` / ``convert_to_groot_lerobot`` /
``export_isaac_from_record``) into a ``tempfile.TemporaryDirectory`` and returns a JSON
manifest of the tree — ``{tree: [{path, size}], total_bytes, info, modality?, episodes?,
tasks?, …}`` — WITHOUT building or buffering a zip. The ``tree`` paths equal the zip's
entry names (``<scan_id>/…`` / ``<scan_id>_groot_lerobot_v2/…``), so ``unzip -l`` of a
downloaded export matches the manifest byte-for-byte on sizes.

Contract vs the zip route (deliberate, minimal divergence):

  * SAME auth (owner-only), SAME ``?source=`` validation (404 ``unknown_derived_key``),
    SAME 400 ``invalid_format``, and for ``isaac_usd`` the SAME gate order and bodies —
    409 ``metric_scale_required`` → 501 ``isaac_unavailable`` → 404 ``no_trajectory`` /
    ``no_points`` — reusing the app's own gate helpers (``_resolve_isaac_scale`` etc.)
    through the lazy ``app_module()`` indirection so the gate logic lives in ONE place.
  * ONE divergence: where the zip route 404s ``no_trajectory`` for
    openreality/groot_lerobot_v2 (today's canonical pilot-persisted scenes:
    ``trajectory_key=None``), the manifest DEGRADES HONESTLY instead — 200 with
    ``complete: false`` and ``absent: [{component, reason}]`` naming exactly which
    components are missing and why, plus ``zip_available: false`` /
    ``zip_blocked_reason: "no_trajectory"`` so the client can explain the Download
    button's behaviour without trying it.

Builders are absence-tolerant by design (verified: empty trajectory → 0-row parquet;
missing cloud/splat → files skipped) — only the record LOADER gates on trajectory, which
is why the degraded path assembles a partial record itself (mirroring
``load_record_from_store`` minus that gate). ``convert_to_groot_lerobot`` KeyErrors on a
0-row trajectory parquet (no columns): caught and reported as an ``absent`` entry, never
a 500.

Memory note (R8): unlike the zip route (which reads the whole archive into RAM before
``send_file(BytesIO(...))``), this route never zips — peak footprint is the export tree
on the broker's disk (tempdir), and the response is metadata-sized JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

from flask import jsonify, request

from server.oreos import app_module, auth_user_id, oreos_bp, scene_persistence
from server.export.record import (
    EXPORT_SOURCE_ORIGINAL,
    _resolve_derived_group,
    load_record_from_store,
    resolved_source_key,
)
from server.export.to_lerobot import convert_to_groot_lerobot
from server.export.writer import export_from_record

MANIFEST_FORMATS = ("openreality", "groot_lerobot_v2", "isaac_usd")

# Reasons are user-facing panel copy — keep them precise and honest.
_REASON_NO_TRAJECTORY = (
    "scan has no persisted per-keyframe trajectory (trajectory_key=None — pilot-persisted "
    "scenes predate the Stage-4 trajectory persist); the export carries 0 poses and the "
    "zip route returns 404 no_trajectory"
)
_REASON_NO_VIDEO = (
    "persisted-record exports carry no keyframe images, so videos/ego_view.mp4 is never "
    "written on this path (the stored keyframes are a sparse evidence subset, not a video "
    "stream)"
)
_REASON_NO_CLOUD = "scan has no stored point cloud (cloud.npz absent)"
_REASON_SPLAT_BY_DESIGN = (
    "the original splat is intentionally not exported — a splat rides only when an "
    "anchor/clamp derived splat is selected (?source= or the derived_latest pointer)"
)
_REASON_GROOT_BLOCKED = (
    "GR00T-LeRobot conversion needs at least one trajectory row (ego-motion IS the action "
    "channel); it fails on a 0-pose export"
)


def _tree_of(root_dir: str, base_name: str) -> tuple[list[dict[str, Any]], int]:
    """Walk ``root_dir`` and return (sorted [{path, size}], total_bytes) with paths
    prefixed ``<base_name>/…`` — exactly the zip's entry names."""
    entries: list[dict[str, Any]] = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root_dir)
            size = int(os.path.getsize(full))
            total += size
            entries.append({"path": f"{base_name}/{rel.replace(os.sep, '/')}", "size": size})
    entries.sort(key=lambda e: e["path"])
    return entries, total


def _read_json_file(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_jsonl_file(path: str) -> Optional[list]:
    try:
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    except Exception:
        return None


# Moved to server.export.record so the background export job can label geometry the same
# way without importing a Flask route module. Re-bound here under its original name for
# every existing caller in this file.
_resolved_source_key = resolved_source_key


def _degraded_record_from_store(
    persistence: Any, user_id: str, scan_id: str, rec: dict[str, Any], source: Any
) -> dict[str, Any]:
    """Assemble a partial ``export_from_record`` input for a scan WITHOUT a persisted
    trajectory. Mirrors ``load_record_from_store`` (same derived-group precedence via its
    ``_resolve_derived_group``) with the trajectory gate removed: ``trajectory`` is empty
    → the builders emit a 0-row parquet, honestly reported in ``absent``."""
    group = _resolve_derived_group(persistence, user_id, scan_id, rec, source)

    normalized: dict[str, Any] = {
        "scan_id": rec.get("scan_id", scan_id),
        "trajectory": {},
        "facts": rec.get("facts"),
        "report": rec.get("report"),
        "keyframes": [],
    }

    cloud = None
    if group.get("cloud"):
        try:
            from server.scene_report.cloud_io import read_ply_cloud

            with open(group["cloud"], "rb") as fh:
                cloud = read_ply_cloud(fh.read())
        except Exception as exc:  # stale derived key → fall back to the original
            print(f"[demo-export] derived cloud read failed {group['cloud']}: {exc}")
            cloud = None
    if cloud is None:
        cloud = persistence.get_cloud(user_id, scan_id)
    if cloud is not None:
        normalized["cloud"] = cloud

    if group.get("splat"):
        normalized["splat"] = group["splat"]

    return normalized


def _absent_for_record(normalized: dict[str, Any], trajectory_missing: bool) -> list[dict[str, str]]:
    """Honest component report for an openreality/groot tree built from ``normalized``."""
    absent: list[dict[str, str]] = []
    if trajectory_missing:
        absent.append({"component": "trajectory", "reason": _REASON_NO_TRAJECTORY})
    # The record path never carries keyframe images → no ego-view video, ever (today).
    absent.append({"component": "videos/ego_view.mp4", "reason": _REASON_NO_VIDEO})
    if normalized.get("cloud") is None:
        absent.append({"component": "clouds/cloud.npz", "reason": _REASON_NO_CLOUD})
    if normalized.get("splat") is None:
        absent.append({"component": "clouds/splat.ply", "reason": _REASON_SPLAT_BY_DESIGN})
    if normalized.get("facts") is None:
        absent.append({"component": "meta/scene_graph.json", "reason": "scan record has no facts"})
    if normalized.get("report") is None:
        absent.append({"component": "meta/report.json", "reason": "scan record has no report"})
    return absent


def _record_manifest(
    scan_id: str,
    export_format: str,
    source_key: str,
    normalized: dict[str, Any],
    trajectory_missing: bool,
) -> tuple[dict[str, Any], int]:
    """Build the openreality/groot tree in a tempdir and shape the manifest response."""
    absent = _absent_for_record(normalized, trajectory_missing)
    body: dict[str, Any] = {
        "scan_id": scan_id,
        "format": export_format,
        "source_key": source_key,
        "tree": [],
        "file_count": 0,
        "total_bytes": 0,
        "info": None,
        "episode": None,
        "modality": None,
        "episodes": None,
        "tasks": None,
        "zip_available": not trajectory_missing,
        "zip_blocked_reason": "no_trajectory" if trajectory_missing else None,
    }

    with tempfile.TemporaryDirectory(prefix="openreality_manifest_") as tmp:
        base = export_from_record(normalized, os.path.join(tmp, "openreality"))

        if export_format == "openreality":
            tree_root, base_name = base, os.path.basename(base)
            body["info"] = _read_json_file(os.path.join(base, "meta", "info.json"))
            body["episode"] = _read_json_file(os.path.join(base, "meta", "episode.json"))
        else:  # groot_lerobot_v2
            groot_dir = os.path.join(tmp, f"{os.path.basename(base)}_groot_lerobot_v2")
            try:
                convert_to_groot_lerobot(base, groot_dir)
            except Exception as exc:
                # 0-pose exports cannot transcode (ego-motion IS the action channel) —
                # report, never 500. The openreality base still built; its manifest is
                # available under format=openreality.
                absent.append(
                    {
                        "component": "groot_lerobot_v2",
                        "reason": f"{_REASON_GROOT_BLOCKED} ({type(exc).__name__})",
                    }
                )
                body["absent"] = absent
                body["complete"] = False
                body["zip_available"] = False
                body["zip_blocked_reason"] = body["zip_blocked_reason"] or "export_failed"
                return body, 200
            tree_root, base_name = groot_dir, os.path.basename(groot_dir)
            body["info"] = _read_json_file(os.path.join(groot_dir, "meta", "info.json"))
            body["modality"] = _read_json_file(os.path.join(groot_dir, "meta", "modality.json"))
            body["episodes"] = _read_jsonl_file(os.path.join(groot_dir, "meta", "episodes.jsonl"))
            body["tasks"] = _read_jsonl_file(os.path.join(groot_dir, "meta", "tasks.jsonl"))
            if not os.path.isfile(
                os.path.join(groot_dir, "videos", "chunk-000", "observation.images.ego_view", "episode_000000.mp4")
            ):
                absent.append(
                    {
                        "component": "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
                        "reason": (
                            "modality.json declares video.ego_view but the source tree has no "
                            "ego_view.mp4 to transcode (see videos/ego_view.mp4 reason)"
                        ),
                    }
                )

        tree, total = _tree_of(tree_root, base_name)
        body["tree"] = tree
        body["file_count"] = len(tree)
        body["total_bytes"] = total

    body["absent"] = absent
    body["complete"] = len(absent) == 0
    return body, 200


def _isaac_manifest(
    app: Any,
    persistence: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    source: Any,
) -> tuple[Any, int]:
    """isaac_usd manifest — SAME gates, order and JSON bodies as the zip route's
    ``_export_isaac_usd`` (gate helpers reused from ``server.app``, never re-implemented),
    then the tree authored WITHOUT zipping."""
    try:
        scale_arg = app._parse_isaac_scale_arg(request.args.get("scale"))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    resolved = app._resolve_isaac_scale(record, source, scale_arg)
    if resolved is None:
        return (
            jsonify(
                {
                    "error": "metric_scale_required",
                    "message": (
                        "Run Metric anchor first — Isaac needs a metric scale. "
                        "Alternatively pass ?scale=<metres-per-SLAM-unit>."
                    ),
                }
            ),
            409,
        )
    isaac_source, isaac_scale, scale_source_label, anchor_scale = resolved

    missing = app._isaac_dependency_status()
    if missing:
        return (
            jsonify(
                {
                    "error": "isaac_unavailable",
                    "message": (
                        f"Isaac USD export needs {', '.join(missing)} which "
                        f"{'is' if len(missing) == 1 else 'are'} not installed in this deployment."
                    ),
                }
            ),
            501,
        )

    normalized = load_record_from_store(persistence, user_id, scan_id, source=isaac_source)
    if normalized is None:
        return (
            jsonify(
                {
                    "error": "no_trajectory",
                    "message": (
                        "scan has no persisted per-keyframe trajectory (pre-Stage-4 scan); "
                        "cannot export Isaac USD GPU-free"
                    ),
                }
            ),
            404,
        )
    cloud = normalized.get("cloud")
    if not cloud or cloud[0] is None or len(cloud[0]) == 0:
        return (
            jsonify(
                {
                    "error": "no_points",
                    "message": "scan has no stored point cloud to build an Isaac scene from",
                }
            ),
            404,
        )

    try:
        from server.export.isaac.writer import export_isaac_from_record

        with tempfile.TemporaryDirectory(prefix="openreality_isaac_manifest_") as tmp:
            isaac_dir = export_isaac_from_record(
                normalized,
                tmp,
                scan_id=scan_id,
                scale=isaac_scale,
                scale_source=scale_source_label,
                anchor_scale=anchor_scale,
                require_metric=True,
            )
            if not isaac_dir:
                raise RuntimeError("record had no geometry to author an Isaac scene from")
            scan_root = os.path.join(tmp, scan_id)
            tree, total = _tree_of(scan_root, scan_id)
            info = _read_json_file(os.path.join(scan_root, "isaac", "manifest.json"))
    except ImportError as exc:
        print(f"[demo-export] isaac manifest dep import failed: {exc}")
        return (
            jsonify(
                {
                    "error": "isaac_unavailable",
                    "message": f"Isaac USD dependency unavailable at runtime: {exc}",
                }
            ),
            501,
        )
    except Exception as exc:
        print(f"[demo-export] scan {scan_id} isaac manifest failed: {exc}")
        return jsonify({"error": "export_failed", "message": str(exc)}), 500

    return (
        jsonify(
            {
                "scan_id": scan_id,
                "format": "isaac_usd",
                "source_key": isaac_source,
                "complete": True,
                "absent": [],
                "tree": tree,
                "file_count": len(tree),
                "total_bytes": total,
                # NOTE: for isaac, ``info`` is the tree's own isaac/manifest.json
                # (IsaacManifest — alignment/geometry provenance), distinct from this
                # HTTP response, which is the export-surface manifest.
                "info": info,
                "episode": None,
                "modality": None,
                "episodes": None,
                "tasks": None,
                "zip_available": True,
                "zip_blocked_reason": None,
                "scale": {
                    "scale": isaac_scale,
                    "scale_source": scale_source_label,
                    "anchor_scale": anchor_scale,
                },
            }
        ),
        200,
    )


@oreos_bp.route("/api/scenes/<scan_id>/demo/export/manifest", methods=["GET"])
def get_export_manifest(scan_id: str):
    """See module docstring. Read-only; never writes to the scene store."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    store = scene_persistence()
    if store is None:
        return jsonify({"error": "not_found"}), 404
    record = store.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    # Explicit-source validation — identical to the zip route: a bogus/foreign derived key
    # is a hard 404, never a silent fallback.
    source = request.args.get("source")
    if source is not None and str(source).strip() not in ("", EXPORT_SOURCE_ORIGINAL):
        get_derived_path = getattr(store, "get_derived_artifact_path", None)
        resolved = (
            get_derived_path(user_id, scan_id, str(source).strip())
            if callable(get_derived_path)
            else None
        )
        if not resolved:
            return (
                jsonify(
                    {
                        "error": "unknown_derived_key",
                        "message": (
                            f"no derived artifact {str(source).strip()!r} for this scan; pass a "
                            "derived key returned by the anchor/clamp routes, or omit ?source="
                        ),
                    }
                ),
                404,
            )

    export_format = (request.args.get("format") or "openreality").strip()
    if export_format not in MANIFEST_FORMATS:
        return (
            jsonify(
                {
                    "error": "invalid_format",
                    "message": (
                        f"unknown format {export_format!r}; expected one of "
                        f"{list(MANIFEST_FORMATS)}"
                    ),
                }
            ),
            400,
        )

    if export_format == "isaac_usd":
        resp, status = _isaac_manifest(app_module(), store, user_id, scan_id, record, source)
        return resp, status

    normalized = load_record_from_store(store, user_id, scan_id, source=source)
    trajectory_missing = normalized is None
    if trajectory_missing:
        normalized = _degraded_record_from_store(store, user_id, scan_id, record, source)

    try:
        body, status = _record_manifest(
            scan_id,
            export_format,
            _resolved_source_key(record, source),
            normalized,
            trajectory_missing,
        )
    except Exception as exc:
        print(f"[demo-export] scan {scan_id} manifest failed ({export_format}): {exc}")
        return jsonify({"error": "export_failed", "message": str(exc)}), 500

    return jsonify(body), status
