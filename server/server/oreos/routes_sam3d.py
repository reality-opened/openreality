"""Segmentation / SAM-3D / variations routes — W3 (docs/demo-2026-07, F3).

Prompt-view ladder (design/features.md F3, honesty ladder a -> b -> c):

  (a) DETECTION EVIDENCE — live with NO generative model: every persisted
      ``facts.objects`` entry carries one evidence item ``{submap_id, frame_idx,
      box_2d, mask_rle}`` (verified on demo-ops/canonical-office-loop:66/66).
      When the scene also stores camera geometry for that keyframe
      (``derived/demo/frames_index.json``, §0.3) the mask is LIFTED to 3D per
      §0.2 (project -> dilated-mask test -> z-buffer occlusion tau=0.08 ->
      cKDTree filter -> gravity-aware PCA OBB). Scenes without a frames index
      (the canonical scene has ``trajectory_key=None``) fall back to the
      detection's own 3D box: points-in-box subset -> outlier filter -> PCA
      OBB, honestly labeled "box-selected (coarse)".
  (b) BEST INDEXED KEYFRAME — needs the frames index AND SAM 3 via fal.ai;
      answers ``503 fal_key_missing`` when the key is not mounted.
  (c) CLIENT-RENDERED VIEW — the request carries ``{view: {image_b64, c2w,
      intrinsics}, point_px}``; SAM 3 via fal.ai segments the render, the mask
      is lifted with the PROVIDED camera geometry (works on every scene).
      ``503 fal_key_missing`` when the key is not mounted; provenance "from
      rendered view" once live.

fal is LIVE as of 2026-07-31 (contracts verified against the real endpoints —
see ``genai/fal_client.py``). It is mounted only on deploys made with
the ``fal-secret`` Modal secret; without it both fal paths degrade honestly to 503.

Completion runs on the self-hosted ``modal_oreos_sam3d`` app (A100-40G,
``modal.Function.from_name`` — the dynamics-sidecar pattern), then the asset
is fitted to OUR selection OBB (uniform scale + extent-matched axes + bottom
snap when up is known — never the model's l2c) and EXP-20 quality-gated.

Variations are TRELLIS via fal.ai. **TRELLIS returns a textured GLB mesh, not a
gaussian PLY** (verified live; our blind implementation guessed
``model_gaussian`` and would have failed every call). The raw GLB is stored as
the provenance record and ``genai.mesh_to_splat`` converts it to a splat PLY so
the fit/gate math and the viewer's gaussian path work unchanged. fal's TRELLIS
is image-conditioned ONLY — there is no text mode.

Storage (§0.1, all non-destructive derived artifacts)::

    derived/demo/objects/<uid_key>/select.json | mask.png | view.png
    derived/demo/objects/<uid_key>/completed/{asset.ply, asset.glb, meta.json}
    derived/demo/objects/<uid_key>/variants/<vid>/{asset.ply, asset.glb, meta.json}
    derived/demo/objects/index.json                (reload-restore index)

``<uid_key>`` is the registry uid with ``:`` -> ``_`` (the store's derived-key
charset has no colon). Every meta carries the honesty envelope ``{provenance,
generated, generator, quality, caveats[], inputs{}}`` + the transform-contract
stamps ``{scan_id, parent_artifact, created_at}``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import jobs as demo_jobs
from server.oreos import segment_geometry as sg
from server.oreos.genai import fal_client
from server.oreos.routes_docs import (
    _MANIFEST_LOCK,
    _MANIFEST_RELATIVE_KEY,
    _index_into_manifest,
    _read_manifest,
)
from server.export.mask_rle import rle_area, rle_to_mask

OBJECTS_INDEX_KEY = "derived/demo/objects/index.json"
FRAMES_INDEX_KEY = "derived/demo/frames_index.json"
NAV_CACHE_KEY = "derived/demo/nav/grid_cache.npz"

# A100-40GB on-demand list price — meta cost estimates only, never billed math.
A100_USD_PER_HR = 2.10

_FIXED_COMPLETION_CAVEAT = "Generated asset completes unseen geometry — unseen parts are invented"

# ---------------------------------------------------------------------------
# injectable seams (tests / verify harness)
# ---------------------------------------------------------------------------

_complete_fn: Any = None  # injected modal Function stand-in with .remote()
_render_object_fn: Any = None


def configure_complete_fn(fn: Any) -> None:
    """Inject the SAM-3D completion function (tests pass a fake with
    ``.remote``; ``None`` resets to the lazy modal.Function.from_name binding)."""
    global _complete_fn
    _complete_fn = fn


def _get_complete_fn() -> Any:
    global _complete_fn
    if _complete_fn is not None:
        return _complete_fn
    import modal  # broker has modal; tests always inject

    _complete_fn = modal.Function.from_name(
        os.environ.get("DEMO_SAM3D_APP", "demo-sam3d"), "complete_object"
    )
    return _complete_fn


def configure_render_object_fn(fn: Any) -> None:
    """Inject the server-side object-view renderer (``modal_oreos_render.render_object_view``)."""
    global _render_object_fn
    _render_object_fn = fn


def _get_render_object_fn() -> Any:
    global _render_object_fn
    if _render_object_fn is not None:
        return _render_object_fn
    import modal

    _render_object_fn = modal.Function.from_name(
        os.environ.get("DEMO_RENDER_APP", "demo-splat-render"), "render_object_view"
    )
    return _render_object_fn


# ---------------------------------------------------------------------------
# scene access helpers
# ---------------------------------------------------------------------------


def _uid_key(uid: str) -> str:
    """Registry uid -> derived-key segment (``det:0`` -> ``det_0``)."""
    return str(uid).replace(":", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_scene(scan_id: str):
    """(store, record, user_id) or a flask error response tuple."""
    user_id = auth_user_id()
    if not user_id:
        return None, (jsonify({"error": "invalid_token"}), 401)
    store = scene_persistence()
    if store is None:
        return None, (jsonify({"error": "not_found"}), 404)
    record = store.get_scene(user_id, scan_id)
    if not record:
        return None, (jsonify({"error": "not_found"}), 404)
    return (store, record, user_id), None


def _facts_objects(record: dict[str, Any]) -> list[dict[str, Any]]:
    facts = record.get("facts") or {}
    objs = facts.get("objects") or []
    return [o for o in objs if isinstance(o, dict)]


def _keyframe_blobs(record: dict[str, Any]) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for kf in record.get("keyframes") or []:
        try:
            out[(int(kf["submap_id"]), int(kf["frame_idx"]))] = str(kf["blob_key"])
        except Exception:
            continue
    return out


def _derived_json(store: Any, user_id: str, scan_id: str, key: str) -> Optional[dict]:
    raw = store.get_derived_artifact(user_id, scan_id, key)
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _frames_index(store: Any, user_id: str, scan_id: str) -> dict[tuple[int, int], dict]:
    """§0.3 keyframe->pose index, ``{(submap, frame): {c2w, intrinsics}}``.
    Empty for scenes recorded without ``--demo-index`` (per-feature fallback)."""
    raw = store.get_derived_artifact(user_id, scan_id, FRAMES_INDEX_KEY)
    if not raw:
        return {}
    try:
        entries = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    if isinstance(entries, dict):
        entries = entries.get("frames") or []
    out: dict[tuple[int, int], dict] = {}
    for e in entries if isinstance(entries, list) else []:
        try:
            key = (int(e["submap_id"]), int(e["frame_idx"]))
            c2w = np.asarray(e["c2w"], dtype=np.float64).reshape(4, 4)
            intr = np.asarray(e["intrinsics"], dtype=np.float64).reshape(-1)[:4]
            out[key] = {"c2w": c2w, "intrinsics": intr}
        except Exception:
            continue
    return out


def _scene_up(store: Any, user_id: str, scan_id: str) -> tuple[Optional[np.ndarray], str]:
    """Up-vector source priority (WORLD-TRANSFORM-CONTRACT): W4's nav cache,
    else pose-derived gravity from trajectory.npz, else None."""
    raw = store.get_derived_artifact(user_id, scan_id, NAV_CACHE_KEY)
    if raw:
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                for key in ("up", "up_vector", "floor_normal", "plane_normal"):
                    if key in data:
                        up = np.asarray(data[key], dtype=np.float64).reshape(-1)[:3]
                        n = np.linalg.norm(up)
                        if n > 1e-9:
                            return up / n, "nav_cache"
        except Exception as exc:
            print(f"[demo.sam3d] nav cache unreadable: {exc}")
    try:
        traj = store.get_trajectory(user_id, scan_id)
    except Exception:
        traj = None
    if traj and traj.get("poses") is not None:
        poses = np.asarray(traj["poses"], dtype=np.float64).reshape(-1, 4, 4)
        if poses.shape[0] > 0:
            # OpenCV c2w: camera +y points DOWN in the image -> world down is the
            # mean of the rotation's second column; up is its negation.
            down = poses[:, :3, 1].mean(axis=0)
            n = np.linalg.norm(down)
            if n > 1e-9:
                return -down / n, "pose_gravity"
    return None, "none"


# lift working set: stride-subsampled world points, cached per scene (<=2 scenes)
_cloud_cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}


def _lift_cloud(store: Any, user_id: str, scan_id: str) -> Optional[tuple[np.ndarray, int]]:
    """(points (M,3) float32 world, stride) with M <= MAX_LIFT_POINTS, or None."""
    key = (str(user_id), str(scan_id))
    if key in _cloud_cache:
        return _cloud_cache[key]
    cloud = store.get_cloud(user_id, scan_id)
    if cloud is None:
        return None
    positions = np.asarray(cloud[0], dtype=np.float32).reshape(-1, 3)
    if positions.shape[0] == 0:
        return None
    sub = sg.stride_subsample(positions.shape[0], sg.MAX_LIFT_POINTS)
    stride = int(np.ceil(positions.shape[0] / max(1, len(sub))))
    pts = positions[sub]
    while len(_cloud_cache) >= 2:  # tiny LRU — broker has 4 GB total
        _cloud_cache.pop(next(iter(_cloud_cache)))
    _cloud_cache[key] = (pts, stride)
    return pts, stride


def _mask_from_evidence(ev: dict[str, Any]) -> Optional[np.ndarray]:
    """Bool mask from an evidence item: mask_rle preferred, else box_2d rect."""
    mask = rle_to_mask(ev.get("mask_rle"))
    if mask is not None:
        return mask
    box = ev.get("box_2d")
    size = (ev.get("mask_rle") or {}).get("size")
    if box and size and len(size) == 2:
        h, w = int(size[0]), int(size[1])
        m = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        m[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
        return m
    return None


def _mask_png(mask: np.ndarray) -> bytes:
    from PIL import Image

    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode_image_b64(image_b64: str) -> bytes:
    """Raw bytes from a base64 string or data URI."""
    payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    return base64.b64decode(payload)


def _persist_object_doc(
    store: Any, user_id: str, scan_id: str, key_suffix: str, doc: dict[str, Any]
) -> str:
    """Write one JSON doc under derived/demo/ and index it into the manifest
    (routes_docs' lock + helpers — ONE lock for every manifest RMW)."""
    doc_bytes = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
    with _MANIFEST_LOCK:
        derived_key = store.save_derived_artifact(user_id, scan_id, f"demo/{key_suffix}", doc_bytes)
        manifest = _read_manifest(store, user_id, scan_id)
        _index_into_manifest(manifest, key_suffix, derived_key)
        store.save_derived_artifact(
            user_id,
            scan_id,
            _MANIFEST_RELATIVE_KEY,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return derived_key


def _update_objects_index(
    store: Any, user_id: str, scan_id: str, uid: str, patch: dict[str, Any]
) -> None:
    """RMW the small reload-restore index at derived/demo/objects/index.json."""
    with _MANIFEST_LOCK:
        index = _derived_json(store, user_id, scan_id, OBJECTS_INDEX_KEY) or {
            "version": 1,
            "objects": {},
        }
        objects = index.get("objects")
        if not isinstance(objects, dict):
            objects = {}
        entry = objects.get(uid) if isinstance(objects.get(uid), dict) else {}
        entry.update(patch)
        entry["updated_at"] = _now_iso()
        objects[uid] = entry
        index["objects"] = objects
        index["updated_at"] = _now_iso()
        store.save_derived_artifact(
            user_id,
            scan_id,
            "demo/objects/index.json",
            json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8"),
        )


# ---------------------------------------------------------------------------
# selection core (shared by the segment route and det-direct completion)
# ---------------------------------------------------------------------------


def _select_from_detection(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    det_index: int,
) -> dict[str, Any]:
    """Path (a): detection evidence -> 3D subset -> OBB. Returns the selection
    payload dict (never a flask response). Raises ValueError on bad input."""
    objects = _facts_objects(record)
    if not (0 <= det_index < len(objects)):
        raise ValueError(f"detection index {det_index} out of range (0..{len(objects) - 1})")
    det = objects[det_index]
    center = np.asarray(det.get("center"), dtype=np.float64).reshape(3)
    extent = np.asarray(det.get("extent") or [0.1, 0.1, 0.1], dtype=np.float64).reshape(3)
    label = str(det.get("human_label") or det.get("query") or f"object {det_index}")
    evidence = (det.get("evidence") or [{}])[0]
    ev_key = None
    if evidence.get("submap_id") is not None and evidence.get("frame_idx") is not None:
        ev_key = (int(evidence["submap_id"]), int(evidence["frame_idx"]))

    up, up_source = _scene_up(store, user_id, scan_id)
    cloud = _lift_cloud(store, user_id, scan_id)
    caveats: list[str] = []
    if up is None:
        caveats.append("no up vector for this scene — OBB not gravity-aligned")

    method = "detection_box"
    idxs = np.empty(0, dtype=np.int64)
    if cloud is not None:
        pts, stride = cloud
        frames = _frames_index(store, user_id, scan_id)
        geom = frames.get(ev_key) if ev_key else None
        mask = _mask_from_evidence(evidence) if evidence else None
        if geom is not None and mask is not None:
            idxs = sg.lift_mask_to_points(pts, mask, geom["intrinsics"], geom["c2w"])
            method = "mask_lift"
        if method == "detection_box" or len(idxs) < sg.MIN_LIFT_POINTS:
            if method == "mask_lift":
                caveats.append("mask lift too sparse — fell back to the detection 3D box")
                method = "detection_box"
            idxs = sg.points_in_obb(pts, center, extent, None, margin=0.25)
            if len(idxs) < sg.MIN_LIFT_POINTS:
                idxs = sg.points_in_obb(pts, center, extent, None, margin=1.0)
                if len(idxs) >= sg.MIN_LIFT_POINTS:
                    caveats.append("detection box grown 2x to reach enough points")

    if cloud is not None and len(idxs) >= sg.MIN_LIFT_POINTS:
        pts, stride = cloud
        keep = sg.filter_outliers_kdtree(pts[idxs])
        sel_pts = pts[idxs][keep]
        c, e, R = sg.pca_obb(sel_pts, up=up)
        obb = sg.obb_to_dict(c, e, R)
        n_points = int(len(sel_pts)) * stride  # approx full-cloud count (stride-scaled)
        quality = "good" if method == "mask_lift" else "usable"
    else:
        # too sparse (or no cloud): OBB-only low tier, completion disabled
        obb = sg.obb_to_dict(center, extent, np.eye(3))
        n_points = int(len(idxs)) if cloud is not None else 0
        quality = "low"
        caveats.append("selection too sparse — detection box only (completion disabled)")

    if method == "mask_lift":
        provenance = "persisted SAM mask lifted to 3D (z-buffer occlusion)"
    else:
        provenance = "box-selected (coarse) — detection 3D box subset (no camera index for this scene)"

    return {
        "det_index": det_index,
        "label": label,
        "evidence": evidence,
        "evidence_key": ev_key,
        "obb": obb,
        "n_points": n_points,
        "method": method,
        "quality": quality,
        "caveats": caveats,
        "up_source": up_source,
        "prompt_source": "detection_keyframe",
    }


def _resolve_detection_near(record: dict[str, Any], point_world: np.ndarray) -> Optional[int]:
    """Nearest detection whose (margin-grown) box contains the click, else the
    nearest center within 1.5x its own max extent. None when nothing is close."""
    objects = _facts_objects(record)
    best: tuple[float, int] | None = None
    for i, det in enumerate(objects):
        try:
            c = np.asarray(det.get("center"), dtype=np.float64).reshape(3)
            e = np.asarray(det.get("extent") or [0.1, 0.1, 0.1], dtype=np.float64).reshape(3)
        except Exception:
            continue
        d = float(np.linalg.norm(point_world - c))
        inside = np.all(np.abs(point_world - c) <= e / 2.0 * 1.25)
        radius = max(float(np.linalg.norm(e)) * 1.5, 1e-6)
        if inside or d <= radius:
            score = d / radius if not inside else d * 0.1  # containment wins strongly
            if best is None or score < best[0]:
                best = (score, i)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# POST /api/scenes/<scan_id>/segment
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/scenes/<scan_id>/segment", methods=["POST"])
def segment_object(scan_id: str):
    """Click -> mask -> 3D selection (prompt ladder a/b/c; see module docstring).

    Body (exactly one prompt form):
      {"object_ref": {"uid": "det:<idx>"}}                       -> path (a)
      {"point_world": [x, y, z]}                                 -> resolve det (a), else (b)/(c)
      {"view": {"image_b64", "c2w", "intrinsics"}, "point_px"}   -> path (c), SAM via fal
    """
    ctx, err = _load_scene(scan_id)
    if err:
        return err
    store, record, user_id = ctx
    body = request.get_json(silent=True) or {}

    # ---- path (c): client-rendered view + click px --------------------------
    if isinstance(body.get("view"), dict):
        return _segment_view(store, record, user_id, scan_id, body)

    # ---- resolve a detection: explicit object_ref or a world-point click ----
    det_index: Optional[int] = None
    if isinstance(body.get("object_ref"), dict):
        uid = str(body["object_ref"].get("uid") or "")
        if uid.startswith("det:"):
            try:
                det_index = int(uid.split(":", 1)[1])
            except ValueError:
                det_index = None
        if det_index is None:
            return jsonify({"error": "bad_request", "message": "object_ref.uid must be det:<idx>"}), 422
    elif body.get("point_world") is not None:
        try:
            p = np.asarray(body["point_world"], dtype=np.float64).reshape(3)
        except Exception:
            return jsonify({"error": "bad_request", "message": "point_world must be [x,y,z]"}), 422
        if not np.all(np.isfinite(p)):
            return jsonify({"error": "bad_request", "message": "point_world must be finite"}), 422
        det_index = _resolve_detection_near(record, p)
        if det_index is None:
            # path (b): needs the frames index AND the fal key. Honest ladder:
            if not fal_client.fal_api_key():
                return (
                    jsonify(
                        {
                            "error": fal_client.ERROR_FAL_KEY_MISSING,
                            "message": "no detection near the click; SAM 3 prompt paths need the "
                            "fal.ai key (create the fal-secret Modal secret). Path (a): click on a "
                            "detected object.",
                        }
                    ),
                    503,
                )
            if not _frames_index(store, user_id, scan_id):
                return (
                    jsonify(
                        {
                            "error": "no_object_near_point",
                            "message": "no detection near the click and no keyframe index for "
                            "this scene — send a rendered view (path c)",
                        }
                    ),
                    404,
                )
            return (
                jsonify({"error": "not_implemented", "message": "prompt path (b) lands with the keyed smoke"}),
                501,
            )
    else:
        return (
            jsonify({"error": "bad_request", "message": "expected object_ref, point_world or view"}),
            422,
        )

    try:
        sel = _select_from_detection(store, user_id, scan_id, record, det_index)
    except ValueError as exc:
        return jsonify({"error": "bad_request", "message": str(exc)}), 422

    uid = f"sel:{int(time.time() * 1000):x}"
    ukey = _uid_key(uid)

    # persist the evidence mask (when present) for the keyframe overlay
    mask_key = ""
    ev = sel["evidence"]
    mask = _mask_from_evidence(ev) if ev else None
    if mask is not None:
        mask_key = store.save_derived_artifact(
            user_id, scan_id, f"demo/objects/{ukey}/mask.png", _mask_png(mask)
        )

    kf_blobs = _keyframe_blobs(record)
    ev_blob = kf_blobs.get(sel["evidence_key"]) if sel["evidence_key"] else None

    select_doc = {
        "uid": uid,
        "scan_id": scan_id,
        "created_at": _now_iso(),
        "origin": "click_segment",
        "prompt_source": sel["prompt_source"],
        "method": sel["method"],
        "label": sel["label"],
        "source_detection": {
            "index": sel["det_index"],
            "evidence": {
                "submap_id": (sel["evidence_key"] or (None, None))[0],
                "frame_idx": (sel["evidence_key"] or (None, None))[1],
                "keyframe_blob_key": ev_blob,
                "mask_area_px": rle_area(ev.get("mask_rle")) if ev else 0,
            },
        },
        "obb": sel["obb"],
        "n_points": sel["n_points"],
        "up_source": sel["up_source"],
        "mask_key": mask_key or None,
        "parent_artifact": record.get("points_key") or record.get("splat_key"),
        "honesty": {
            "provenance": sel["method"]
            and (
                "persisted SAM mask lifted to 3D (z-buffer occlusion)"
                if sel["method"] == "mask_lift"
                else "box-selected (coarse)"
            ),
            "generated": False,
            "quality": sel["quality"],
            "caveats": sel["caveats"],
        },
    }
    _persist_object_doc(store, user_id, scan_id, f"objects/{ukey}/select.json", select_doc)
    _update_objects_index(
        store,
        user_id,
        scan_id,
        uid,
        {
            "label": sel["label"],
            "select_key": f"derived/demo/objects/{ukey}/select.json",
            "mask_key": mask_key or None,
            "det_index": sel["det_index"],
            "quality": sel["quality"],
        },
    )

    return jsonify(
        {
            # SegmentResponse (protocol demo.ts) ...
            "uid": uid,
            "obb": sel["obb"],
            "n_points": sel["n_points"],
            "mask_key": mask_key,
            "prompt_source": sel["prompt_source"],
            "label": sel["label"],
            # ... + additive detail the panel reads defensively:
            "det_index": sel["det_index"],
            "method": sel["method"],
            "quality": sel["quality"],
            "caveats": sel["caveats"],
            "up_source": sel["up_source"],
            "evidence": {
                "submap_id": (sel["evidence_key"] or (None, None))[0],
                "frame_idx": (sel["evidence_key"] or (None, None))[1],
                "keyframe_blob_key": ev_blob,
                "box_2d": ev.get("box_2d") if ev else None,
                "mask_rle": ev.get("mask_rle") if ev else None,
            },
        }
    )


def _segment_view(store, record, user_id: str, scan_id: str, body: dict[str, Any]):
    """Prompt path (c): SAM 3 on a client-rendered view, lift with the provided
    camera. Keyless -> 503 fal_key_missing (the full plumbing stays in place)."""
    view = body.get("view") or {}
    point_px = body.get("point_px")
    try:
        image_b64 = str(view["image_b64"])
        c2w = np.asarray(view["c2w"], dtype=np.float64).reshape(4, 4)
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64).reshape(-1)[:4]
        px = (float(point_px[0]), float(point_px[1]))
    except Exception:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "view needs {image_b64, c2w 4x4, intrinsics [fx,fy,cx,cy]} + point_px [x,y]",
                }
            ),
            422,
        )

    try:
        fal_client.require_key()
    except fal_client.FalKeyMissing:
        return (
            jsonify(
                {
                    "error": fal_client.ERROR_FAL_KEY_MISSING,
                    "message": (
                        "SAM 3 click-segmentation needs the fal.ai key — this deployment "
                        "was made before the fal-secret Modal secret existed, so it is not mounted"
                    ),
                }
            ),
            503,
        )

    data_uri = (
        image_b64 if image_b64.startswith("data:") else f"data:image/png;base64,{image_b64}"
    )
    try:
        result = fal_client.segment_sam3(data_uri, point_px=px)
        mask = _extract_fal_mask(result)
    except fal_client.FalAccountError as exc:
        # Billing/key problem, not a model failure — say so, with fal's own words.
        return (
            jsonify(
                {
                    "error": "fal_account_error",
                    "message": f"fal rejected the key: {exc.detail or exc}",
                    "fal_status": exc.status,
                }
            ),
            502,
        )
    except fal_client.FalError as exc:
        return (
            jsonify(
                {
                    "error": "sam3_failed",
                    "message": str(exc)[:400],
                    "fal_status": exc.status,
                    "fal_detail": exc.detail,
                }
            ),
            502,
        )
    if mask is None:
        return (
            jsonify(
                {
                    "error": "sam3_empty",
                    "message": (
                        "SAM 3 returned no usable mask for that click — the result carried "
                        f"keys {sorted(result.keys())} with no decodable non-empty mask"
                    ),
                }
            ),
            502,
        )

    cloud = _lift_cloud(store, user_id, scan_id)
    if cloud is None:
        return jsonify({"error": "no_geometry", "message": "scene has no stored point cloud"}), 422
    pts, stride = cloud
    up, up_source = _scene_up(store, user_id, scan_id)
    idxs = sg.lift_mask_to_points(pts, mask, intrinsics, c2w)
    caveats = ["segmented on a client-rendered view — lower completion quality expected"]
    if up is None:
        caveats.append("no up vector for this scene — OBB not gravity-aligned")
    if len(idxs) < sg.MIN_LIFT_POINTS:
        return (
            jsonify(
                {
                    "error": "selection_too_sparse",
                    "message": f"mask lifted to only {len(idxs)} points (<{sg.MIN_LIFT_POINTS})",
                }
            ),
            422,
        )
    keep = sg.filter_outliers_kdtree(pts[idxs])
    sel_pts = pts[idxs][keep]
    c, e, R = sg.pca_obb(sel_pts, up=up)
    obb = sg.obb_to_dict(c, e, R)

    uid = f"sel:{int(time.time() * 1000):x}"
    ukey = _uid_key(uid)
    mask_key = store.save_derived_artifact(
        user_id, scan_id, f"demo/objects/{ukey}/mask.png", _mask_png(mask)
    )
    store.save_derived_artifact(
        user_id, scan_id, f"demo/objects/{ukey}/view.png", _decode_image_b64(image_b64)
    )
    label = str(body.get("label") or "selection")
    select_doc = {
        "uid": uid,
        "scan_id": scan_id,
        "created_at": _now_iso(),
        "origin": "click_segment",
        "prompt_source": "rendered_view",
        "method": "mask_lift",
        "label": label,
        "obb": obb,
        "n_points": int(len(sel_pts)) * stride,
        "up_source": up_source,
        "mask_key": mask_key,
        "view_key": f"derived/demo/objects/{ukey}/view.png",
        "parent_artifact": record.get("points_key") or record.get("splat_key"),
        "honesty": {
            "provenance": "from rendered view",
            "generated": False,
            "quality": "usable",
            "caveats": caveats,
        },
    }
    _persist_object_doc(store, user_id, scan_id, f"objects/{ukey}/select.json", select_doc)
    _update_objects_index(
        store, user_id, scan_id, uid,
        {"label": label, "select_key": f"derived/demo/objects/{ukey}/select.json",
         "mask_key": mask_key, "quality": "usable"},
    )
    return jsonify(
        {
            "uid": uid,
            "obb": obb,
            "n_points": int(len(sel_pts)) * stride,
            "mask_key": mask_key,
            "prompt_source": "rendered_view",
            "label": label,
            "method": "mask_lift",
            "quality": "usable",
            "caveats": caveats,
            "up_source": up_source,
        }
    )


def _extract_fal_mask(result: dict[str, Any]) -> Optional[np.ndarray]:
    """Bool mask from a fal SAM-3 result.

    VERIFIED shape (``fal-ai/sam-3/image``, 2026-07-31)::

        {"masks":    [{"url", "width", "height", "content_type": "image/png"}],
         "image":    {"url", ...},          # primary preview
         "scores":   [0.988...],            # when include_scores
         "boxes":    [[cx, cy, w, h] | null],   # NORMALISED cxcywh, when available
         "metadata": [{"index", "score", "box"}]}

    ``masks[i].url`` is an ``https://…fal.media/…`` URL normally and an inline
    ``data:image/png;base64,…`` URI under ``sync_mode`` — :func:`fal_client.download`
    handles both. The PNG is mode-L, values {0, 255}, at the FULL input-image
    resolution (not cropped), origin top-left — the same grid our renders and
    ``lift_mask_to_points`` use, so no resize or flip is needed.

    ``masks`` is preferred over ``image``: with ``apply_mask=False`` they agree,
    but with ``apply_mask=True`` ``image`` is the RGB with the mask burned in,
    which would binarise into garbage. The legacy speculative keys are kept as a
    last-resort fallback so an endpoint revision degrades rather than breaks."""
    from PIL import Image

    def _decode(data: bytes) -> Optional[np.ndarray]:
        try:
            mask = np.asarray(Image.open(io.BytesIO(data)).convert("L")) > 127
        except Exception:
            return None
        return mask if mask.any() else None

    def _from(cand: Any) -> Optional[np.ndarray]:
        if isinstance(cand, dict):
            b64 = cand.get("b64") or cand.get("base64") or cand.get("image_b64")
            if isinstance(b64, str):
                got = _decode(_decode_image_b64(b64))
                if got is not None:
                    return got
            url = cand.get("url")
            if isinstance(url, str):
                try:
                    return _decode(fal_client.download(url))
                except fal_client.FalError:
                    return None
        elif isinstance(cand, str):
            if cand.startswith("http") or cand.startswith("data:"):
                try:
                    return _decode(fal_client.download(cand))
                except fal_client.FalError:
                    return None
            return _decode(_decode_image_b64(cand))
        return None

    # Verified primary path: the highest-scoring entry of `masks`.
    masks = result.get("masks")
    if isinstance(masks, list) and masks:
        order = list(range(len(masks)))
        scores = result.get("scores")
        if isinstance(scores, list) and len(scores) == len(masks):
            order.sort(key=lambda i: (scores[i] is not None, scores[i] or 0.0), reverse=True)
        for i in order:
            got = _from(masks[i])
            if got is not None:
                return got

    for key in ("mask", "individual_masks", "combined_mask", "image"):
        v = result.get(key)
        for cand in v if isinstance(v, list) else [v]:
            if cand is None:
                continue
            got = _from(cand)
            if got is not None:
                return got
    return None


# ---------------------------------------------------------------------------
# POST /api/scenes/<scan_id>/objects/<uid>/complete
# ---------------------------------------------------------------------------


#: Fixed copy for a completion prompted by a render rather than by a photograph. It is
#: the SAME honesty rule synthetic views carry (IMPORTED-SPLAT-CONTRACT.md): a render is
#: real evidence about the splat's geometry and appearance, and it is never a photo.
RENDERED_PROMPT_CAVEAT = (
    "Prompted from a synthetic view — a render of this scene's own splat, not a "
    "photograph; no camera ever saw this object from this angle"
)


def _rendered_completion_inputs(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    uid: str,
    det_index: int,
    up: Optional[np.ndarray],
    up_source: str,
) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, int]]]:
    """Completion inputs for an object with no evidence keyframe, from a server render.

    The picture is a render of the scene framing this object's OBB; the mask is the
    object's OWN gaussians, rendered alone and then cut back wherever nearer geometry
    occludes them (``splat_render.visibility``). Neither is a photograph and neither
    pretends to be — ``source`` and the caveat both say so, and they ride all the way
    onto the completed asset's ``meta.json``.

    Every failure here degrades to the SAME ``not_completable`` shape the keyframe path
    uses, carrying the renderer's own reason. That matters more than it looks: the
    alternative to "this object is hidden behind other geometry at every angle tried" is
    a completion built on a mask over a wall, which is invention wearing measurement's
    clothes."""
    try:
        sel = _select_from_detection(store, user_id, scan_id, record, det_index)
    except ValueError as exc:
        return None, (jsonify({"error": "unknown_object", "uid": uid, "message": str(exc)}), 404)
    if sel["quality"] == "low":
        return None, (
            jsonify(
                {
                    "error": "not_completable",
                    "message": "selection too sparse — completion disabled (box-only)",
                }
            ),
            422,
        )

    try:
        out = _get_render_object_fn().remote(user_id, scan_id, sel["obb"])
    except Exception as exc:  # app not deployed, modal unreachable, container crash
        print(f"[demo.sam3d] server render unavailable: {type(exc).__name__}: {exc}")
        return None, (
            jsonify(
                {
                    "error": "not_completable",
                    "message": (
                        "this object has no persisted evidence keyframe, and the "
                        f"server-side renderer that would stand in for one is unreachable "
                        f"({type(exc).__name__}). Deploy the "
                        f"{os.environ.get('DEMO_RENDER_APP', 'demo-splat-render')} app, or "
                        "capture views from the viewer."
                    ),
                }
            ),
            422,
        )
    if not isinstance(out, dict) or not out.get("ok"):
        detail = (out or {}).get("detail") if isinstance(out, dict) else str(out)[:200]
        return None, (
            jsonify(
                {
                    "error": "not_completable",
                    "message": f"no usable rendered view of this object: {detail}",
                    "render_error": (out or {}).get("error") if isinstance(out, dict) else None,
                    "render_diagnostics": (out or {}).get("diagnostics") if isinstance(out, dict) else None,
                }
            ),
            422,
        )

    # Persist what the model was shown, beside the mask, exactly like prompt path (c).
    # A generated asset whose prompt image cannot be produced again is not reviewable.
    ukey = _uid_key(uid)
    view_key = store.save_derived_artifact(
        user_id, scan_id, f"demo/objects/{ukey}/view.png", out["image_png"]
    )
    store.save_derived_artifact(
        user_id, scan_id, f"demo/objects/{ukey}/mask.png", out["mask_png"]
    )
    return (
        {
            "image_bytes": out["image_png"],
            "mask_bytes": out["mask_png"],
            "obb": sel["obb"],
            "label": sel["label"],
            "n_points": sel["n_points"],
            "source": "server_rendered_view",
            "evidence_blob": None,
            "view_key": view_key,
            "render": {
                "view": out.get("view"),
                "chosen": out.get("chosen"),
                "artifact": (out.get("source") or {}).get("artifact"),
                "sh_degree": (out.get("source") or {}).get("sh_degree"),
                "provenance": out.get("provenance"),
                "caveats": out.get("caveats"),
            },
            "extra_caveats": [RENDERED_PROMPT_CAVEAT, *(out.get("caveats") or [])],
            "up": up,
            "up_source": up_source,
        },
        None,
    )


def _completion_inputs(
    store: Any, user_id: str, scan_id: str, record: dict[str, Any], uid: str
) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, int]]]:
    """(inputs, error) — inputs = {image_bytes, mask_bytes, obb, label, source,
    n_points, up} for a completable object; error = (json, status) otherwise."""
    ukey = _uid_key(uid)
    up, up_source = _scene_up(store, user_id, scan_id)

    if uid.startswith("sel:"):
        doc = _derived_json(store, user_id, scan_id, f"derived/demo/objects/{ukey}/select.json")
        if not doc:
            return None, (jsonify({"error": "unknown_object", "uid": uid}), 404)
        obb = doc.get("obb")
        label = str(doc.get("label") or uid)
        n_points = int(doc.get("n_points") or 0)
        if str((doc.get("honesty") or {}).get("quality")) == "low":
            return None, (
                jsonify(
                    {
                        "error": "not_completable",
                        "message": "selection is low tier (too sparse) — completion disabled",
                    }
                ),
                422,
            )
        if doc.get("view_key"):  # path (c) selection: stored render + SAM mask
            image_bytes = store.get_derived_artifact(user_id, scan_id, str(doc["view_key"]))
            mask_bytes = store.get_derived_artifact(user_id, scan_id, str(doc.get("mask_key")))
            if image_bytes and mask_bytes:
                return (
                    {
                        "image_bytes": image_bytes,
                        "mask_bytes": mask_bytes,
                        "obb": obb,
                        "label": label,
                        "n_points": n_points,
                        "source": "rendered_view",
                        "up": up,
                        "up_source": up_source,
                    },
                    None,
                )
        det_index = ((doc.get("source_detection") or {}).get("index"))
        if det_index is None:
            return None, (
                jsonify(
                    {"error": "not_completable", "message": "selection has no prompt image stored"}
                ),
                422,
            )
        uid_for_evidence = f"det:{int(det_index)}"
    elif uid.startswith("det:"):
        uid_for_evidence = uid
        obb = None
        label = None
        n_points = 0
    else:
        return None, (jsonify({"error": "unknown_object", "uid": uid}), 404)

    # detection evidence path: keyframe JPEG + persisted RLE mask
    try:
        det_index = int(uid_for_evidence.split(":", 1)[1])
    except ValueError:
        return None, (jsonify({"error": "unknown_object", "uid": uid}), 404)
    objects = _facts_objects(record)
    if not (0 <= det_index < len(objects)):
        return None, (jsonify({"error": "unknown_object", "uid": uid}), 404)
    det = objects[det_index]
    ev = (det.get("evidence") or [{}])[0]
    ev_key = None
    if ev.get("submap_id") is not None and ev.get("frame_idx") is not None:
        ev_key = (int(ev["submap_id"]), int(ev["frame_idx"]))
    blob = _keyframe_blobs(record).get(ev_key) if ev_key else None
    if not blob:
        # No photograph of this object exists — not for want of looking, the scene has
        # none (an import has no capture video at all, and only 23/66 canonical objects
        # carry a persisted evidence keyframe). Render one instead. This is the path the
        # old refusal text pointed at and that did not exist.
        return _rendered_completion_inputs(
            store, user_id, scan_id, record, uid, det_index, up, up_source
        )
    image_bytes = store.get_keyframe(user_id, scan_id, blob)
    if not image_bytes:
        return None, (
            jsonify({"error": "not_completable", "message": f"keyframe blob {blob} unreadable"}),
            422,
        )
    mask = _mask_from_evidence(ev)
    if mask is None:
        return None, (
            jsonify({"error": "not_completable", "message": "no persisted mask/box for this object"}),
            422,
        )
    # scale the mask to the keyframe pixel grid when they differ (defensive)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            iw, ih = img.size
        if (ih, iw) != mask.shape:
            mimg = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (iw, ih), Image.NEAREST
            )
            mask = np.asarray(mimg) > 127
    except Exception as exc:
        print(f"[demo.sam3d] mask/keyframe size check failed: {exc}")

    if obb is None:  # det-direct completion: refine the box subset now
        sel = _select_from_detection(store, user_id, scan_id, record, det_index)
        if sel["quality"] == "low":
            return None, (
                jsonify(
                    {
                        "error": "not_completable",
                        "message": "selection too sparse — completion disabled (box-only)",
                    }
                ),
                422,
            )
        obb = sel["obb"]
        label = sel["label"]
        n_points = sel["n_points"]

    return (
        {
            "image_bytes": image_bytes,
            "mask_bytes": _mask_png(mask),
            "obb": obb,
            "label": label or uid,
            "n_points": n_points,
            "source": "detection_keyframe",
            "evidence_blob": blob,
            "up": up,
            "up_source": up_source,
        },
        None,
    )


@oreos_bp.route("/api/scenes/<scan_id>/objects/<uid>/complete", methods=["POST"])
def complete_object(scan_id: str, uid: str):
    """SAM 3D Objects completion -> fit to OUR selection OBB -> EXP-20 gate.
    202 {job_id}; poll GET /api/scenes/<id>/jobs/<job_id> (1 Hz)."""
    ctx, err = _load_scene(scan_id)
    if err:
        return err
    store, record, user_id = ctx
    body = request.get_json(silent=True) or {}
    seed = int(body.get("seed") or 42)

    inputs, ierr = _completion_inputs(store, user_id, scan_id, record, uid)
    if ierr:
        return ierr

    active = demo_jobs.any_active_scene_job(user_id, scan_id, kind_prefix=f"complete:{uid}")
    if active:
        return jsonify({"error": "job_conflict", "job_id": active}), 409

    ukey = _uid_key(uid)
    job_id = demo_jobs.create_scene_job(
        user_id,
        scan_id,
        f"complete:{uid}",
        note="SAM 3D completion — cold GPU starts take 60–120 s (warming GPU)",
    )

    def _run() -> dict[str, Any]:
        t0 = time.time()
        fn = _get_complete_fn()
        out = fn.remote(
            image_png=inputs["image_bytes"], mask_png=inputs["mask_bytes"], seed=seed
        )
        gpu_meta = out.get("meta") or {}
        asset_ply = out.get("asset_ply")
        if not asset_ply:
            raise RuntimeError(f"SAM-3D returned no asset: {gpu_meta.get('error') or 'unknown'}")

        obb = inputs["obb"]
        asset_pts = sg.read_ply_positions(asset_ply)
        fit = sg.fit_asset_to_obb(
            asset_pts, obb["center"], obb["extents"], obb["rotation"], up=inputs["up"]
        )
        placed = sg.apply_transform(asset_pts, fit["transform"])
        gate = sg.quality_gate(placed, obb["center"], obb["extents"], obb["rotation"])

        asset_key = store.save_derived_artifact(
            user_id, scan_id, f"demo/objects/{ukey}/completed/asset.ply", asset_ply
        )
        glb_key = None
        if out.get("asset_glb"):
            glb_key = store.save_derived_artifact(
                user_id, scan_id, f"demo/objects/{ukey}/completed/asset.glb", out["asset_glb"]
            )
        total_s = time.time() - t0
        gpu_s = float(gpu_meta.get("seconds") or 0.0)
        meta = {
            "uid": uid,
            "scan_id": scan_id,
            "created_at": _now_iso(),
            "parent_artifact": record.get("points_key") or record.get("splat_key"),
            "provenance": "AI-completed (SAM 3D)",
            "generated": True,
            "generator": "sam3d",
            "quality": gate["tier"],
            "caveats": [
                _FIXED_COMPLETION_CAVEAT,
                *fit["caveats"],
                # When the prompt was a render rather than a photograph, that fact travels
                # with the asset — a reader of this file must never have to infer it.
                *(inputs.get("extra_caveats") or []),
            ],
            "inputs": {
                "label": inputs["label"],
                "prompt_source": inputs["source"],
                "evidence_blob": inputs.get("evidence_blob"),
                "prompt_view_key": inputs.get("view_key"),
                "render": inputs.get("render"),
                "seed": seed,
                "n_selection_points": inputs["n_points"],
                "up_source": inputs["up_source"],
            },
            "fit": fit,
            "gate": gate,
            "asset": {
                "ply_key": asset_key,
                "glb_key": glb_key,
                "n_gaussians": int(gpu_meta.get("n_gaussians") or len(asset_pts)),
            },
            "timings": {"gpu_seconds": gpu_s, "total_seconds": round(total_s, 2)},
            "cost_estimate_usd": round(gpu_s / 3600.0 * A100_USD_PER_HR, 4),
        }
        _persist_object_doc(
            store, user_id, scan_id, f"objects/{ukey}/completed/meta.json", meta
        )
        _update_objects_index(
            store,
            user_id,
            scan_id,
            uid,
            {
                "label": inputs["label"],
                "completed_key": asset_key,
                "completed_meta_key": f"derived/demo/objects/{ukey}/completed/meta.json",
                "quality": gate["tier"],
            },
        )
        return {
            "asset_key": asset_key,
            "glb_key": glb_key,
            "meta_key": f"derived/demo/objects/{ukey}/completed/meta.json",
            "transform": fit["transform"],
            "quality": gate["tier"],
            "gate": gate,
            "caveats": meta["caveats"],
            "n_gaussians": meta["asset"]["n_gaussians"],
            "seconds": round(total_s, 2),
        }

    demo_jobs.run_scene_job(job_id, _run)
    return jsonify({"job_id": job_id}), 202


# ---------------------------------------------------------------------------
# POST /api/scenes/<scan_id>/objects/<uid>/variants
# ---------------------------------------------------------------------------


@oreos_bp.route("/api/scenes/<scan_id>/objects/<uid>/variants", methods=["POST"])
def object_variants(scan_id: str, uid: str):
    """TRELLIS variations via fal.ai. 202 {job_id}; 503 fal_key_missing when the
    key is not mounted (create the fal-secret Modal secret, then redeploy).

    ``mode`` must be ``"image"``: fal's TRELLIS endpoint is image-conditioned
    only — its input schema has no ``prompt`` field, and because fal silently
    ignores unknown keys a text request would have burned an A100 run and
    returned a 3D model of the *conditioning image*, not of the words. We reject
    it explicitly instead of pretending."""
    ctx, err = _load_scene(scan_id)
    if err:
        return err
    store, record, user_id = ctx
    body = request.get_json(silent=True) or {}
    mode = str(body.get("mode") or "image")
    if mode == "text":
        return (
            jsonify(
                {
                    "error": "text_mode_unsupported",
                    "message": (
                        "fal's TRELLIS (fal-ai/trellis) is image-to-3D only — it has no text "
                        "prompt input. Generate a variation from the object's evidence crop "
                        "(mode 'image'), optionally with your own reference_image_b64."
                    ),
                }
            ),
            422,
        )
    if mode != "image":
        return jsonify({"error": "bad_request", "message": "mode must be 'image'"}), 422

    if not fal_client.fal_api_key():
        return (
            jsonify(
                {
                    "error": fal_client.ERROR_FAL_KEY_MISSING,
                    "message": (
                        "TRELLIS variations need the fal.ai key — this deployment was made "
                        "before the fal-secret Modal secret existed, so it is not mounted"
                    ),
                }
            ),
            503,
        )

    inputs, ierr = _completion_inputs(store, user_id, scan_id, record, uid)
    if ierr and mode == "image" and not body.get("reference_image_b64"):
        return ierr  # image mode needs SOME conditioning image
    obb = (inputs or {}).get("obb")
    if obb is None:
        # variants can still fit to a sel: doc's obb even when not completable
        ukey = _uid_key(uid)
        doc = _derived_json(store, user_id, scan_id, f"derived/demo/objects/{ukey}/select.json")
        if doc and doc.get("obb"):
            obb = doc["obb"]
        else:
            return jsonify({"error": "unknown_object", "uid": uid}), 404

    active = demo_jobs.any_active_scene_job(user_id, scan_id, kind_prefix=f"variants:{uid}")
    if active:
        return jsonify({"error": "job_conflict", "job_id": active}), 409

    ukey = _uid_key(uid)
    seed = int(body.get("seed") or 42)
    reference_b64 = body.get("reference_image_b64")
    up = (inputs or {}).get("up")
    job_id = demo_jobs.create_scene_job(user_id, scan_id, f"variants:{uid}")

    def _run() -> dict[str, Any]:
        import uuid as _uuid

        from server.oreos.genai import mesh_to_splat

        if reference_b64:
            data_uri = (
                reference_b64
                if str(reference_b64).startswith("data:")
                else f"data:image/png;base64,{reference_b64}"
            )
            conditioning = "reference_image"
        else:
            # RAW evidence crop located by the mask — see _evidence_crop for why
            # this must not be composited onto a flat background
            data_uri = "data:image/png;base64," + base64.b64encode(
                _evidence_crop(inputs["image_bytes"], inputs["mask_bytes"])
            ).decode("ascii")
            conditioning = "evidence_crop"
        result = fal_client.trellis_generate(image_b64_data_uri=data_uri, seed=seed)

        raw_bytes, kind = _extract_trellis_asset(result)
        if not raw_bytes:
            raise RuntimeError(
                "TRELLIS returned no downloadable artifact "
                f"(result keys: {sorted(result.keys())})"
            )

        vid = _uuid.uuid4().hex[:8]
        conversion: dict[str, Any] = {}
        glb_key = None
        if kind == "glb":
            # The real fal contract: a textured GLB. Keep the generator's own
            # output verbatim as the provenance record, and derive the splat the
            # fit/gate math and the viewer both speak.
            glb_key = store.save_derived_artifact(
                user_id, scan_id, f"demo/objects/{ukey}/variants/{vid}/asset.glb", raw_bytes
            )
            ply_bytes, conversion = mesh_to_splat.glb_to_splat_ply(raw_bytes, seed=seed)
        else:
            ply_bytes = raw_bytes
            conversion = {"source_format": "ply", "caveats": []}

        asset_pts = sg.read_ply_positions(ply_bytes)
        fit = sg.fit_asset_to_obb(asset_pts, obb["center"], obb["extents"], obb["rotation"], up=up)
        placed = sg.apply_transform(asset_pts, fit["transform"])
        gate = sg.quality_gate(placed, obb["center"], obb["extents"], obb["rotation"])
        asset_key = store.save_derived_artifact(
            user_id, scan_id, f"demo/objects/{ukey}/variants/{vid}/asset.ply", ply_bytes
        )
        caveats = ["Generated variation — not captured geometry"]
        if kind == "glb":
            caveats.append(
                "TRELLIS produced a textured mesh — the rendered splat is a surface "
                "resampling of that mesh, not native gaussians"
            )
        caveats.extend(conversion.get("caveats") or [])
        caveats.extend(fit.get("caveats") or [])
        meta = {
            "uid": uid,
            "variant_id": vid,
            "scan_id": scan_id,
            "created_at": _now_iso(),
            "parent_artifact": record.get("points_key") or record.get("splat_key"),
            "provenance": "AI-generated (TRELLIS)",
            "generated": True,
            "generator": "trellis",
            "quality": gate["tier"],
            "caveats": caveats,
            "inputs": {"mode": mode, "conditioning": conditioning, "seed": seed},
            "fit": fit,
            "gate": gate,
            "timings": result.get("timings"),
            "conversion": conversion,
            "asset": {
                "ply_key": asset_key,
                "glb_key": glb_key,
                "source_format": kind,
                "n_gaussians": int(len(asset_pts)),
            },
        }
        _persist_object_doc(
            store, user_id, scan_id, f"objects/{ukey}/variants/{vid}/meta.json", meta
        )
        index_patch = {
            "variants": {
                vid: {
                    "asset_key": asset_key,
                    "glb_key": glb_key,
                    "meta_key": f"derived/demo/objects/{ukey}/variants/{vid}/meta.json",
                }
            }
        }
        # merge (don't clobber) the per-uid variants map
        with _MANIFEST_LOCK:
            index = _derived_json(store, user_id, scan_id, OBJECTS_INDEX_KEY) or {
                "version": 1,
                "objects": {},
            }
            entry = index.get("objects", {}).get(uid) or {}
            variants = entry.get("variants") or {}
            variants.update(index_patch["variants"])
            entry["variants"] = variants
            entry["updated_at"] = _now_iso()
            index.setdefault("objects", {})[uid] = entry
            store.save_derived_artifact(
                user_id, scan_id, "demo/objects/index.json",
                json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        return {
            "variant_id": vid,
            "asset_key": asset_key,
            "glb_key": glb_key,
            "meta_key": f"derived/demo/objects/{ukey}/variants/{vid}/meta.json",
            "transform": fit["transform"],
            "quality": gate["tier"],
            "gate": gate,
            "caveats": caveats,
            "n_gaussians": int(len(asset_pts)),
        }

    demo_jobs.run_scene_job(job_id, _run)
    return jsonify({"job_id": job_id}), 202


TRELLIS_CROP_PAD_PX = 16


def _evidence_crop(image_bytes: bytes, mask_png: bytes, pad: int = TRELLIS_CROP_PAD_PX) -> bytes:
    """The RAW photo cropped to the mask's bounding box — TRELLIS conditioning.

    ⚠️ **Do not composite the object onto a flat background here.** Measured on
    the real endpoint 2026-07-31 (det:5 'office chair', identical seed):

      * raw crop, unmasked          -> a clean chair (backrest, arms, castors)
      * masked onto WHITE  (+16 px) -> a correct chair embedded in a 1.0 x 1.0
                                       square SLAB — the flat background is
                                       reconstructed as geometry
      * masked onto MID-GREY        -> same slab
      * masked, transparent RGBA    -> same slab (fal flattens alpha first)

    TRELLIS already runs its own background removal; handing it a synthetic flat
    region defeats that and it dutifully models the plane. So the mask is used
    only to LOCATE the crop — never to paint it. (The mask still does all the
    real work elsewhere: it is what the OBB is fitted from.)"""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img, Image.open(io.BytesIO(mask_png)) as m:
        rgb = img.convert("RGB")
        mask = m.convert("L").resize(rgb.size, Image.NEAREST)
        bbox = mask.getbbox()
        if bbox:
            rgb = rgb.crop(
                (
                    max(0, bbox[0] - pad),
                    max(0, bbox[1] - pad),
                    min(rgb.size[0], bbox[2] + pad),
                    min(rgb.size[1], bbox[3] + pad),
                )
            )
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def _extract_trellis_asset(result: dict[str, Any]) -> tuple[Optional[bytes], Optional[str]]:
    """``(bytes, kind)`` for the TRELLIS artifact; ``kind`` is ``"glb"`` or ``"ply"``.

    VERIFIED shape (``fal-ai/trellis``, 2026-07-31)::

        {"model_mesh": {"url", "content_type": "application/octet-stream",
                        "file_name": "model.glb", "file_size": 689344},
         "timings":    {"prepare", "generation", "export"}}

    The artifact is glTF-binary (``glTF`` magic) — a textured mesh, ~1.4k verts /
    2.4k tris for a small object. There is NO ``model_gaussian`` key; the blind
    implementation looked for one and required ``data[:3] == b"ply"``, so it
    would have discarded every real result and raised "returned no gaussian PLY".

    We sniff the magic rather than trust the extension, and still accept a PLY in
    case fal adds a native-gaussian output later."""
    for key in ("model_mesh", "model_gaussian", "gaussian_ply", "model", "files", "outputs"):
        v = result.get(key)
        for cand in v if isinstance(v, list) else [v]:
            if cand is None:
                continue
            url = (
                cand.get("url")
                if isinstance(cand, dict)
                else (cand if isinstance(cand, str) else None)
            )
            if not isinstance(url, str):
                continue
            data = fal_client.download(url)
            if data[:4] == b"glTF":
                return data, "glb"
            if data[:3] == b"ply":
                return data, "ply"
    return None, None


# ---------------------------------------------------------------------------
# Imported-splat objects (W-B) — IMPORTED-SPLAT-CONTRACT.md
#
#   POST /api/scenes/<scan_id>/imported_objects   run detection, persist the inventory
#   GET  /api/scenes/<scan_id>/imported_objects   the last run's provenance doc
#
# Appended as its own block so the parallel workstream merges stay friction-free.
# The detection itself lives in ``imported_objects.py``; this is plumbing + honesty.
# ---------------------------------------------------------------------------

IMPORTED_RUN_KEY = "derived/demo/objects/imported_run.json"
IMPORTED_RUN_SUFFIX = "objects/imported_run.json"

#: How workstream A's synthetic-view PNGs are fetched. Injected by tests and by the
#: integration once A's storage lands; the default below covers the two shapes the
#: contract admits (a store accessor, or a key on the view record).
_view_loader: Any = None


def configure_synthetic_view_loader(fn: Any) -> None:
    """Inject ``(store, user_id, scan_id, view_dict) -> bytes|None``. ``None`` restores
    the default resolver."""
    global _view_loader
    _view_loader = fn


def _load_synthetic_view(store: Any, user_id: str, scan_id: str, view: dict[str, Any]) -> Optional[bytes]:
    if _view_loader is not None:
        return _view_loader(store, user_id, scan_id, view)
    getter = getattr(store, "get_synthetic_view", None)
    if callable(getter):
        try:
            return getter(user_id, scan_id, str(view.get("view_id") or ""))
        except Exception as exc:
            print(f"[demo.imported_objects] synthetic view read failed: {exc}")
            return None
    key = view.get("derived_key") or view.get("blob_key")
    if isinstance(key, str) and key.startswith("derived/"):
        return store.get_derived_artifact(user_id, scan_id, key)
    return None


def _synthetic_views(record: dict[str, Any]) -> list[dict[str, Any]]:
    """A's registered views off the scene record, or [] while that half is unbuilt."""
    views = record.get("synthetic_views")
    if not isinstance(views, list):
        return []
    return [v for v in views if isinstance(v, dict) and v.get("position") and v.get("quaternion")]


def _ground_frame_up(record: dict[str, Any]) -> Optional[list[float]]:
    """A's ground-frame up axis when it has run, else None. Feature-detected rather than
    depended on: B's boxes must stand up on their own (contract: "B depends on A only for
    labels")."""
    metrics = ((record.get("facts") or {}).get("metrics")) or {}
    if not metrics.get("vertical_axis_known"):
        return None
    up = metrics.get("up_axis")
    if isinstance(up, (list, tuple)) and len(up) == 3:
        try:
            vals = [float(v) for v in up]
        except (TypeError, ValueError):
            return None
        if any(abs(v) > 1e-9 for v in vals):
            return vals
    return None


@oreos_bp.route("/api/scenes/<scan_id>/imported_objects", methods=["GET"])
def get_imported_objects_route(scan_id: str):
    ctx, err = _load_scene(scan_id)
    if err:
        return err
    store, record, user_id = ctx
    doc = _derived_json(store, user_id, scan_id, IMPORTED_RUN_KEY)
    if not doc:
        return jsonify({
            "scan_id": scan_id,
            "status": "none",
            "object_count": 0,
            "reason": "no geometry detection has run on this scene yet",
            "eligible": bool(record.get("points_key")),
        }), 200
    return jsonify({"scan_id": scan_id, "status": "ready", "run": doc}), 200


@oreos_bp.route("/api/scenes/<scan_id>/imported_objects", methods=["POST"])
def post_imported_objects_route(scan_id: str):
    """Cluster the scan's geometry into objects and persist them as ``facts.objects``.

    The whole point is schema parity: the inventory written here is the SAME shape a
    captured scene's detections are, so the Objects list, selection sync, nav "Go to",
    swap and completion all work on an imported splat with no client special-casing. What
    differs is provenance, and that rides in each object's ``imported_detection`` block.
    """
    ctx, err = _load_scene(scan_id)
    if err:
        return err
    store, record, user_id = ctx

    from server.oreos import imported_objects as imp

    body = request.get_json(silent=True) or {}
    label_mode = str(body.get("labels") or "auto").lower()
    if label_mode not in ("auto", "geometry"):
        return jsonify({"error": "bad_request", "message": "labels must be 'auto' or 'geometry'"}), 400

    existing = _facts_objects(record)
    captured = [o for o in existing if not o.get("imported_detection")]
    if captured:
        return jsonify({
            "error": "has_detections",
            "scan_id": scan_id,
            "object_count": len(existing),
            "message": (
                f"this scan already has {len(captured)} objects from a capture — geometry "
                "clustering is only for imported splats with no detected inventory"
            ),
        }), 409

    loaded = _lift_cloud(store, user_id, scan_id)
    if loaded is None:
        return jsonify({
            "error": "no_geometry",
            "scan_id": scan_id,
            "message": "this scan has no stored point cloud to cluster",
        }), 422

    points, stride = loaded
    params = _imported_params(body)
    up = body.get("up") or _ground_frame_up(record)
    try:
        result = imp.detect_objects(points, up=up, params=params)
    except ValueError as exc:
        return jsonify({"error": "no_geometry", "scan_id": scan_id, "message": str(exc)}), 422

    label_summary: dict[str, Any] = {"mode": label_mode}
    if label_mode == "auto":
        label_summary.update(_label_from_synthetic_views(store, record, user_id, scan_id, result, imp))
    else:
        label_summary["note"] = "geometry-only requested"

    objects = imp.to_facts_objects(result)
    try:
        written = store.replace_geometry_objects(user_id, scan_id, objects)
    except KeyError:
        return jsonify({"error": "not_found"}), 404
    except ValueError as exc:
        return jsonify({"error": "has_detections", "scan_id": scan_id, "message": str(exc)}), 409

    run_id = uuid.uuid4().hex[:12]
    doc = imp.run_document(
        result,
        scan_id=scan_id,
        run_id=run_id,
        created_at=_now_iso(),
        parent_artifact=str(record.get("points_key") or "cloud.npz"),
        label_mode=label_mode,
        label_note=label_summary.get("note"),
    )
    doc["labels"] = label_summary
    doc["cloud"] = {"source_points": int(record.get("point_count") or 0), "stride": int(stride)}
    doc["replaced"] = int(written.get("replaced", 0))
    derived_key = _persist_object_doc(store, user_id, scan_id, IMPORTED_RUN_SUFFIX, doc)

    return jsonify({
        "scan_id": scan_id,
        "run_id": run_id,
        "count": len(objects),
        "replaced": doc["replaced"],
        "objects": objects,
        "labels": label_summary,
        "provenance": imp.PROVENANCE_GEOMETRY,
        "caveats": [imp.CAVEAT_GEOMETRY],
        "up": result.up,
        "up_source": result.up_source,
        "diagnostics": result.diagnostics,
        "rejected": result.rejected,
        "derived_key": derived_key,
    }), 200


def _imported_params(body: dict[str, Any]):
    """Caller overrides on top of the module defaults — every one of them lands in the
    run doc, so a set of boxes can always be traced back to the numbers that made it."""
    import dataclasses

    from server.oreos.imported_objects import DetectParams

    base = DetectParams()
    fields: dict[str, Any] = {}
    for name, cast in (
        ("voxel_frac", float),
        ("link_radius", int),
        ("min_voxels", int),
        ("max_objects", int),
        ("min_extent_frac", float),
        ("max_extent_frac", float),
        ("detect_walls", bool),
    ):
        if body.get(name) is not None:
            try:
                fields[name] = cast(body[name])
            except (TypeError, ValueError):
                continue
    return dataclasses.replace(base, **fields) if fields else base


def _label_from_synthetic_views(
    store: Any, record: dict[str, Any], user_id: str, scan_id: str, result: Any, imp: Any
) -> dict[str, Any]:
    """Upgrade the geometry labels from A's synthetic views, or say why not.

    Every branch that cannot produce a real label leaves the geometry label in place —
    an imported splat with no views must look like a geometry-only run, never like one
    that guessed names.
    """
    raw_views = _synthetic_views(record)
    if not raw_views:
        return {"note": "no synthetic views registered for this scene — geometry labels only"}
    client = imp.make_label_client()
    if client is None:
        return {"note": "no OPENROUTER_API_KEY mounted — geometry labels only"}
    try:
        views = [imp.SyntheticView.from_wire(v) for v in raw_views]
    except (KeyError, TypeError, ValueError) as exc:
        return {"note": f"synthetic view records unreadable ({exc}) — geometry labels only"}
    by_id = {v.view_id: raw for v, raw in zip(views, raw_views)}
    return imp.label_clusters_from_views(
        result,
        views,
        lambda view: _load_synthetic_view(store, user_id, scan_id, by_id.get(view.view_id, {})),
        client,
    )
