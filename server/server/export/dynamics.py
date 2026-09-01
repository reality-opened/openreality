"""Dynamics exporter: dynamic object *tracks* lifted into the SLAM world frame.

The temporal sibling of ``grounding.py``. Where ``grounding.py`` emits one static
``ObjectInstance`` per detected object, this emits one :class:`ObjectTrack` per *moving*
object -- a per-keyframe 3D center followed across the scan -- into a ``dynamics/`` sidecar
alongside the existing static export. It is **additive and default-off**: nothing here runs
unless the caller passes already-computed 2D tracks, so the static trajectory + grounding
outputs stay byte-identical.

Design invariants (locked -- see ``contracts/export-format.md`` "Dynamic objects"):

- **Per-frame pose lift, never cross-frame fusion.** Each frame's object 3D center is the
  object's 2D-track centroid unprojected with a robust (mask-region median) depth and pushed
  to world by *that frame's own* VGGT cam->world pose. The dynamics + static producers share
  the ``index_map`` **clock** (frame ordering) only -- not a shared metric frame.
- **Tracker-agnostic.** This module consumes ALREADY-COMPUTED 2D tracks + per-frame depth +
  visibility (``objects_2d``). It imports **no tracker** and no GPU/torch. The seam where a
  permissively-licensed tracker (e.g. BootsTAPIR / Apache-2.0) plugs in is the ``objects_2d``
  argument -- see ``build_object_tracks`` and the contract. (CoTracker3 is CC-BY-NC and cannot
  ship; do not add it here.)
- **Up-to-scale + approximate on movers.** The SLAM world frame is up-to-scale / not metric /
  not gravity-aligned, and VGGT depth weakens on dynamic content
  (``experiments/e2_track3d/RESULTS.md``). ``frame_centers`` is an estimate; label it as such.

GPU-free / numpy-only, mirroring the rest of ``server/export/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from server.scene_report.schemas import ObjectTrack


DEFAULT_MOVE_THRESHOLD = 0.05  # up-to-scale world units; > this excursion => "moving"


# ---------------------------------------------------------------------------
# small helpers (mirror grounding.py)
# ---------------------------------------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _resolve_frame(frame_key: Any, index_map: dict) -> Optional[int]:
    """Object frame key -> global keyframe index (the shared clock), or ``None``.

    A key may already be a global int, or a ``(submap_id, local_idx)`` pair to be resolved
    through ``index_map`` exactly like an ``EvidenceRef`` (§7.1). Unresolvable keys are
    dropped gracefully (e.g. a track frame that landed on a loop-closure submap).
    """
    if frame_key is None:
        return None
    if isinstance(frame_key, (list, tuple)) and len(frame_key) == 2:
        try:
            return index_map.get((int(frame_key[0]), int(frame_key[1])))
        except (TypeError, ValueError):
            return None
    try:
        return int(frame_key)
    except (TypeError, ValueError):
        return None


def _lookup(seq: Any, gi: int) -> Optional[np.ndarray]:
    """Index a per-frame pose/intrinsics container (dict keyed by global idx OR sequence)."""
    if seq is None:
        return None
    try:
        if isinstance(seq, dict):
            val = seq.get(gi)
        else:
            if gi < 0 or gi >= len(seq):
                return None
            val = seq[gi]
    except (TypeError, KeyError, IndexError):
        return None
    return None if val is None else np.asarray(val, dtype=np.float64)


def _intrinsics_params(K: np.ndarray) -> tuple:
    """Accept a 4-vector ``[fx,fy,cx,cy]`` (trajectory convention) or a 3x3 K -> fx,fy,cx,cy."""
    arr = np.asarray(K, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    K3 = arr.reshape(3, 3)
    return float(K3[0, 0]), float(K3[1, 1]), float(K3[0, 2]), float(K3[1, 2])


def unproject_to_world(
    u: float, v: float, depth: float, K: np.ndarray, pose_c2w: np.ndarray
) -> np.ndarray:
    """Pinhole-unproject pixel ``(u,v)`` at ``depth`` -> camera point -> world.

    ``K`` is ``[fx,fy,cx,cy]`` or 3x3; ``pose_c2w`` is a 4x4 cam->world matrix (``t`` is the
    camera center in world, matching ``trajectory._decompose`` / ``decompose_camera``). Camera
    looks down +z. Returns a world-frame ``(3,)`` point.
    """
    fx, fy, cx, cy = _intrinsics_params(K)
    d = float(depth)
    x = (float(u) - cx) / fx * d
    y = (float(v) - cy) / fy * d
    p_cam = np.array([x, y, d, 1.0], dtype=np.float64)
    p_world = np.asarray(pose_c2w, dtype=np.float64).reshape(4, 4) @ p_cam
    return p_world[:3]


# ---------------------------------------------------------------------------
# build tracks
# ---------------------------------------------------------------------------

def build_object_tracks(
    objects_2d: list,
    poses_c2w: Any,
    intrinsics: Any,
    index_map: dict,
    *,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> list:
    """Lift already-computed 2D object tracks to world-frame :class:`ObjectTrack`s.

    Pure / numpy; **tracker-agnostic**. This is the integration seam: a permissively-licensed
    2D point/mask tracker (e.g. BootsTAPIR) produces ``objects_2d``; nothing here depends on
    which tracker did.

    ``objects_2d``: list of per-object dicts (or attr objects) with::

        {
          "id": int,                       # object id
          "query": str,                    # open-set label / prompt
          "frames": [f0, f1, ...],         # per-frame keys: global int OR (submap_id, local)
          "centroids": [(u, v), ...],      # 2D track centroid per frame (pixels)
          "depths": [d, ...],              # robust mask-region depth per frame (median VGGT/DepthPro)
          "visibility": [bool, ...],       # optional; default True
          "confidence": [c, ...],          # optional; default 1.0
          "extents": [(ex,ey,ez), ...],    # optional per-frame world extent; default zeros
          "description": ObjectDescription # optional
        }

    ``poses_c2w`` / ``intrinsics``: per-**global-frame-index** containers (dict keyed by global
    index, or a sequence indexed by it). Each pose is 4x4 cam->world; each intrinsic is
    ``[fx,fy,cx,cy]`` or 3x3 K. ``index_map`` (``{(submap_id, local) -> global_idx}``) is the
    shared clock used to resolve/validate frame keys.

    For each object/frame: unproject the centroid with the frame depth+intrinsics to a camera
    point, then lift to world via that frame's own pose. Aggregate to a time-averaged summary
    (mirroring ``ObjectInstance``) plus the per-frame track; ``motion`` is ``"moving"`` iff the
    largest world-frame excursion from the first center exceeds ``move_threshold``.
    """
    tracks: list = []
    for obj in objects_2d or []:
        oid = _get(obj, "id", 0)
        query = _get(obj, "query", "") or ""
        frame_keys = list(_get(obj, "frames", []) or [])
        centroids = list(_get(obj, "centroids", []) or [])
        depths = list(_get(obj, "depths", []) or [])
        vis_in = list(_get(obj, "visibility", []) or [])
        conf_in = list(_get(obj, "confidence", []) or [])
        ext_in = list(_get(obj, "extents", []) or [])

        # collect per-frame samples in global-frame order
        samples: list = []  # (global_idx, world_center(3,), extent(3,), visible, conf)
        for i, fkey in enumerate(frame_keys):
            gi = _resolve_frame(fkey, index_map)
            if gi is None:
                continue
            pose = _lookup(poses_c2w, gi)
            K = _lookup(intrinsics, gi)
            if pose is None or K is None:
                continue
            if i >= len(centroids) or i >= len(depths):
                continue
            u, v = centroids[i]
            depth = depths[i]
            if depth is None or not np.isfinite(float(depth)):
                continue
            center = unproject_to_world(u, v, depth, K, pose)
            if not np.all(np.isfinite(center)):
                continue
            visible = bool(vis_in[i]) if i < len(vis_in) else True
            conf = float(conf_in[i]) if i < len(conf_in) else 1.0
            if i < len(ext_in) and ext_in[i] is not None:
                extent = [float(x) for x in np.asarray(ext_in[i], dtype=np.float64).reshape(-1)[:3]]
                if len(extent) < 3:
                    extent = extent + [0.0] * (3 - len(extent))
            else:
                extent = [0.0, 0.0, 0.0]
            samples.append((int(gi), center, extent, visible, conf))

        samples.sort(key=lambda s: s[0])
        tracks.append(_aggregate(oid, query, obj, samples, move_threshold))
    return tracks


def _aggregate(
    oid: Any, query: str, obj: Any, samples: list, move_threshold: float
) -> ObjectTrack:
    frames = [int(s[0]) for s in samples]
    frame_centers = [[float(c) for c in s[1]] for s in samples]
    frame_extents = [[float(e) for e in s[2]] for s in samples]
    frame_visibility = [bool(s[3]) for s in samples]
    frame_confidence = [float(s[4]) for s in samples]

    # time-averaged summary from VISIBLE frames (fall back to all if none flagged visible).
    vis_idx = [i for i, ok in enumerate(frame_visibility) if ok] or list(range(len(samples)))
    if vis_idx:
        centers_arr = np.asarray([frame_centers[i] for i in vis_idx], dtype=np.float64)
        avg_center = centers_arr.mean(axis=0).tolist()
        exts = np.asarray([frame_extents[i] for i in vis_idx], dtype=np.float64)
        avg_extent = exts.mean(axis=0).tolist() if exts.size else []
        avg_conf = float(np.mean([frame_confidence[i] for i in vis_idx]))
    else:
        avg_center, avg_extent, avg_conf = [], [], 0.0

    # motion: largest world-frame excursion from the first center (up-to-scale).
    max_disp = 0.0
    if len(frame_centers) >= 2:
        c0 = np.asarray(frame_centers[0], dtype=np.float64)
        max_disp = float(
            max(np.linalg.norm(np.asarray(c, dtype=np.float64) - c0) for c in frame_centers)
        )
    motion = "moving" if max_disp > float(move_threshold) else "static"

    try:
        object_id = int(oid)
    except (TypeError, ValueError):
        object_id = 0

    return ObjectTrack(
        query=str(query),
        object_id=object_id,
        center=[float(x) for x in avg_center],
        extent=[float(x) for x in avg_extent],
        confidence=avg_conf,
        evidence=_get(obj, "evidence", []) or [],
        description=_get(obj, "description", None),
        frames=frames,
        frame_centers=frame_centers,
        frame_extents=frame_extents,
        frame_visibility=frame_visibility,
        frame_confidence=frame_confidence,
        first_frame=frames[0] if frames else -1,
        last_frame=frames[-1] if frames else -1,
        motion=motion,
        max_displacement=max_disp,
    )


# ---------------------------------------------------------------------------
# write sidecar (mirror grounding.write_objects_jsonl)
# ---------------------------------------------------------------------------

def write_object_tracks_jsonl(tracks: list, out_dir: str) -> int:
    """``<out_dir>/dynamics/objects_tracks.jsonl`` -- one :class:`ObjectTrack` per line.

    Creates the ``dynamics/`` subdir. Returns the number of tracks written. Called only when
    the caller supplies tracks, so absence of this file is the default (static export intact).
    """
    dyn_dir = os.path.join(out_dir, "dynamics")
    os.makedirs(dyn_dir, exist_ok=True)
    path = os.path.join(dyn_dir, "objects_tracks.jsonl")
    written = 0
    with open(path, "w") as fh:
        for track in tracks or []:
            payload = track.model_dump() if hasattr(track, "model_dump") else dict(track)
            fh.write(json.dumps(payload, default=_json_default) + "\n")
            written += 1
    return written


# ===========================================================================
# Camera-space object tracks (Track-On-R / strict-gated variant).
#
# The second, STRICTER consumer of the same tracker-agnostic ``objects_2d`` seam. Where
# :func:`build_object_tracks` eagerly lifts every finite-depth sample to a WORLD
# ``frame_centers`` row, this variant implements the EXP-15 design
# (``platform:experiments/research/2026-07-06-exp15-trackonr-dynamics.md``):
#
#   * **STRICT visibility gating (C1).** A position is emitted ONLY on frames the tracker
#     flags visible; occluded frames become explicit ``null`` gaps — never positions.
#     (EXP-15 proved occluded Track-On-R positions wander; an ungated consumer exports
#     garbage.)
#   * **Confidence-masked depth lift.** Depth is sampled from the per-frame SLAM depth map
#     with a window median over pixels whose ``depth_conf`` clears a per-run percentile
#     threshold; failing the mask is an explicit gap too (``frame_liftable=False``).
#   * **Camera-space positions + per-frame poses.** ``frame_xyz_cam`` is CAMERA-space; the
#     per-frame ``cam_to_world`` / ``intrinsic`` ship once in a companion ``poses.jsonl``
#     (the "share the clock, not the metric frame" invariant taken literally — the consumer
#     composes with the frame's OWN pose, lazily). A world lift is still done internally,
#     per frame, for the ``world_rms_spread`` sanity diagnostics only.
#
# Additive + default-off like everything here: nothing calls this unless the caller does.
# GPU-free / numpy-only. Field map vs :class:`ObjectTrack` documented in EXP-15 §2
# (``frame_xyz_cam`` replaces ``frame_centers``; ``frame_liftable``/``frame_uv``/
# ``frame_depth_conf`` are new; gaps are ``null``).
# ===========================================================================

CAMERA_TRACKS_SCHEMA_VERSION = "openreality-dynamics-objects-cam/0.1"
DEFAULT_DEPTH_CONF_PERCENTILE = 30.0  # reject the lowest-confidence 30% (EXP-15 recipe)


def robust_depth_at(
    depth_t: np.ndarray,
    conf_t: np.ndarray,
    u: float,
    v: float,
    conf_thresh: float,
    win: int = 2,
) -> tuple:
    """Confidence-masked window-median SLAM depth at pixel ``(u, v)``.

    Samples a ``(2*win+1)^2`` window of ``depth_t``/``conf_t`` (both ``(H, W)``), keeps
    pixels with ``conf >= conf_thresh`` and finite positive depth, and returns
    ``(median_depth, ok, mean_conf_of_kept)``. ``ok=False`` (depth ``nan``) when the pixel is
    out of bounds or no window pixel clears the mask — the caller records an explicit gap.
    Ported from the validated EXP-15 lift (``exp15_common.robust_depth``).
    """
    H, W = depth_t.shape
    ui, vi = int(round(float(u))), int(round(float(v)))
    if not (0 <= ui < W and 0 <= vi < H):
        return float("nan"), False, 0.0
    u0, u1 = max(0, ui - win), min(W, ui + win + 1)
    v0, v1 = max(0, vi - win), min(H, vi + win + 1)
    dpatch = np.asarray(depth_t[v0:v1, u0:u1], dtype=np.float64)
    cpatch = np.asarray(conf_t[v0:v1, u0:u1], dtype=np.float64)
    keep = (cpatch >= conf_thresh) & np.isfinite(dpatch) & (dpatch > 0)
    if not keep.any():
        return float("nan"), False, 0.0
    return float(np.median(dpatch[keep])), True, float(cpatch[keep].mean())


def backproject_camera(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """Pinhole back-projection to CAMERA space (no pose compose): ``(u,v,d) -> (X,Y,Z)``.

    ``K`` is ``[fx,fy,cx,cy]`` or 3x3. OpenCV axes (+X right, +Y down, +Z forward)."""
    fx, fy, cx, cy = _intrinsics_params(K)
    d = float(depth)
    return np.array([(float(u) - cx) / fx * d, (float(v) - cy) / fy * d, d], dtype=np.float64)


def _lookup_map(seq: Any, gi: int) -> Optional[np.ndarray]:
    """Index a per-frame MAP container (depth / conf) without dtype coercion."""
    if seq is None:
        return None
    try:
        if isinstance(seq, dict):
            val = seq.get(gi)
        else:
            if gi < 0 or gi >= len(seq):
                return None
            val = seq[gi]
    except (TypeError, KeyError, IndexError):
        return None
    return None if val is None else np.asarray(val)


def _scene_scale(poses_c2w: Any, global_frames: list) -> float:
    """Camera-path length (sum of consecutive camera-center distances) over the sorted
    global frames — EXP-15's scene-scale normalizer for the spread diagnostics."""
    centers = []
    for gi in sorted(set(int(g) for g in global_frames)):
        pose = _lookup(poses_c2w, gi)
        if pose is not None:
            centers.append(np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, 3])
    if len(centers) < 2:
        return 0.0
    c = np.stack(centers)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def build_camera_space_tracks(
    objects_2d: list,
    poses_c2w: Any,
    intrinsics: Any,
    depth_maps: Any,
    depth_conf_maps: Any,
    index_map: dict,
    *,
    conf_percentile: float = DEFAULT_DEPTH_CONF_PERCENTILE,
    conf_thresh: Optional[float] = None,
    depth_win: int = 2,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> dict:
    """Strict-gated, camera-space lift of already-computed 2D tracks (the EXP-15 recipe).

    Same tracker-agnostic seam as :func:`build_object_tracks` — consumes ``objects_2d``
    entries ``{id, query, frames, centroids, visibility[, confidence]}`` (a per-POINT track
    from e.g. Track-On-R; ``depths`` from the seam are IGNORED here — depth is re-sampled
    from the maps under the confidence mask). ``centroids`` must be pixels in the SAME native
    raster as ``depth_maps`` (no rescale). ``poses_c2w``/``intrinsics``/``depth_maps``/
    ``depth_conf_maps`` are per-global-frame containers (dict keyed by global index, or
    sequence); ``index_map`` resolves ``(submap_id, local)`` keys exactly like
    :func:`build_object_tracks`.

    Gate policy (per frame, in order):
      1. tracker says NOT visible          -> ``frame_visibility=False``, uv/xyz ``null``.
      2. visible but depth fails the mask  -> ``frame_visibility=True``,
         ``frame_liftable=False``, uv kept, xyz ``null``.
      3. visible + confident depth         -> ``frame_liftable=True``, camera-space xyz.

    ``conf_thresh`` overrides the per-run percentile threshold (else it is
    ``np.percentile(all depth_conf, conf_percentile)`` over the frames actually used).

    Returns ``{"records": [...], "conf_thresh": float, "scene_scale": float}`` — records are
    JSONL-ready dicts (gaps ``null``, strict-JSON safe).
    """
    # -- resolve every object's frame keys once; collect the global-frame universe ---------
    resolved: list = []  # (obj, [(i_local, gi), ...])
    universe: list = []
    for obj in objects_2d or []:
        frame_keys = list(_get(obj, "frames", []) or [])
        pairs = []
        for i, fkey in enumerate(frame_keys):
            gi = _resolve_frame(fkey, index_map)
            if gi is None:
                continue
            pairs.append((i, int(gi)))
        pairs.sort(key=lambda p: p[1])
        resolved.append((obj, pairs))
        universe.extend(gi for _, gi in pairs)

    # -- per-run confidence threshold (percentile over the conf maps actually referenced) --
    if conf_thresh is None:
        samples = []
        for gi in sorted(set(universe)):
            c = _lookup_map(depth_conf_maps, gi)
            if c is not None:
                samples.append(np.asarray(c, dtype=np.float64).ravel())
        conf_thresh = (
            float(np.percentile(np.concatenate(samples), conf_percentile)) if samples else 0.0
        )
    conf_thresh = float(conf_thresh)

    scene_scale = _scene_scale(poses_c2w, universe)

    records: list = []
    for obj, pairs in resolved:
        oid = _get(obj, "id", 0)
        query = _get(obj, "query", "") or ""
        centroids = list(_get(obj, "centroids", []) or [])
        vis_in = list(_get(obj, "visibility", []) or [])
        conf_in = list(_get(obj, "confidence", []) or [])

        frames: list = []
        frame_visibility: list = []
        frame_liftable: list = []
        frame_uv: list = []
        frame_xyz_cam: list = []
        frame_depth_conf: list = []
        frame_track_conf: list = []
        world_pts: list = []  # internal, for the sanity diagnostics only

        for i, gi in pairs:
            if i >= len(centroids):
                continue
            pose = _lookup(poses_c2w, gi)
            K = _lookup(intrinsics, gi)
            if pose is None or K is None:
                continue
            visible = bool(vis_in[i]) if i < len(vis_in) else True
            tconf = float(conf_in[i]) if i < len(conf_in) else 1.0

            frames.append(int(gi))
            frame_visibility.append(visible)
            frame_track_conf.append(tconf)

            if not visible:
                # C1 STRICT: occluded => explicit gap, never a position.
                frame_liftable.append(False)
                frame_uv.append(None)
                frame_xyz_cam.append(None)
                frame_depth_conf.append(None)
                continue

            u, v = centroids[i]
            frame_uv.append([float(u), float(v)])
            dmap = _lookup_map(depth_maps, gi)
            cmap = _lookup_map(depth_conf_maps, gi)
            d, ok, mean_conf = float("nan"), False, 0.0
            if dmap is not None and cmap is not None:
                d, ok, mean_conf = robust_depth_at(dmap, cmap, u, v, conf_thresh, win=depth_win)
            if not ok:
                # visible but depth failed the mask => gap (no fabricated position).
                frame_liftable.append(False)
                frame_xyz_cam.append(None)
                frame_depth_conf.append(None)
                continue
            cam = backproject_camera(u, v, d, K)
            if not np.all(np.isfinite(cam)):
                frame_liftable.append(False)
                frame_xyz_cam.append(None)
                frame_depth_conf.append(None)
                continue
            frame_liftable.append(True)
            frame_xyz_cam.append([float(x) for x in cam])
            frame_depth_conf.append(float(mean_conf))
            P = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            world_pts.append(P[:3, :3] @ cam + P[:3, 3])  # that frame's OWN pose (lazy-lift dual)

        # -- sanity diagnostics (world-frame, per-frame-own-pose lifts; up-to-scale) --------
        n_lift = len(world_pts)
        if n_lift >= 2:
            W = np.stack(world_pts)
            centroid = W.mean(axis=0)
            dists = np.linalg.norm(W - centroid, axis=1)
            rms = float(np.sqrt((dists**2).mean()))
            max_exc = float(dists.max())
            disp = float(np.linalg.norm(W - W[0], axis=1).max())
            motion = "moving" if disp > float(move_threshold) else "static"
        else:
            rms = max_exc = disp = None
            motion = "unknown"

        vis_frames = [frames[j] for j in range(len(frames)) if frame_visibility[j]]
        try:
            object_id = int(oid)
        except (TypeError, ValueError):
            object_id = 0
        records.append(
            {
                "schema_version": CAMERA_TRACKS_SCHEMA_VERSION,
                "object_id": object_id,
                "query": str(query),
                "motion": motion,
                "frames": frames,  # the SHARED CLOCK (global keyframe indices)
                "frame_visibility": frame_visibility,  # tracker-visible (the C1 gate)
                "frame_liftable": frame_liftable,  # visible AND depth cleared the mask
                "frame_uv": frame_uv,  # native-raster px; null when occluded
                "frame_xyz_cam": frame_xyz_cam,  # CAMERA-space 3D; null on any gap
                "frame_depth_conf": frame_depth_conf,  # window-mean conf of kept pixels
                "frame_confidence": frame_track_conf,  # tracker confidence passthrough
                "first_frame": (min(vis_frames) if vis_frames else -1),
                "last_frame": (max(vis_frames) if vis_frames else -1),
                "n_visible": int(sum(frame_visibility)),
                "n_liftable": int(sum(frame_liftable)),
                "max_displacement": disp,  # up-to-scale world excursion (diagnostic)
                "world_rms_spread": rms,
                "world_max_excursion": max_exc,
                "world_rms_spread_pct_scene": (
                    round(rms / scene_scale * 100.0, 4)
                    if (rms is not None and scene_scale)
                    else None
                ),
                "up_to_scale": True,
                "units_note": (
                    "frame_xyz_cam are CAMERA-space SLAM units (up-to-scale); compose with the "
                    "frame's own cam_to_world from poses.jsonl to reach world. Gaps are null — "
                    "occluded frames never carry positions."
                ),
            }
        )
    return {"records": records, "conf_thresh": conf_thresh, "scene_scale": scene_scale}


def write_camera_space_tracks(
    records: list,
    poses_c2w: Any,
    intrinsics: Any,
    image_hw: tuple,
    out_dir: str,
) -> dict:
    """Write ``<out_dir>/dynamics/{objects_tracks_cam.jsonl, poses.jsonl}``.

    ``objects_tracks_cam.jsonl`` — one strict-gated camera-space record per line (from
    :func:`build_camera_space_tracks`). ``poses.jsonl`` — the shared per-frame camera table
    (``frame_index``, ``cam_to_world`` 4x4, ``intrinsic`` 3x3, ``image_hw``), one row per
    global frame referenced by any record. Both strict JSON (``allow_nan=False`` — gaps are
    ``null`` by construction, so a NaN reaching serialization is a bug and fails loud).

    Deliberately a DIFFERENT file from ``objects_tracks.jsonl`` (the eager world-lift
    :class:`ObjectTrack` sidecar) so the two producers never clobber each other. Additive +
    default-off. Returns ``{"tracks_path", "poses_path", "n_tracks", "n_pose_rows"}``.
    """
    dyn_dir = os.path.join(out_dir, "dynamics")
    os.makedirs(dyn_dir, exist_ok=True)
    tracks_path = os.path.join(dyn_dir, "objects_tracks_cam.jsonl")
    poses_path = os.path.join(dyn_dir, "poses.jsonl")

    with open(tracks_path, "w") as fh:
        for rec in records or []:
            fh.write(json.dumps(rec, allow_nan=False) + "\n")

    all_frames = sorted({int(f) for rec in records or [] for f in rec.get("frames", [])})
    n_rows = 0
    with open(poses_path, "w") as fh:
        for gi in all_frames:
            pose = _lookup(poses_c2w, gi)
            K = _lookup(intrinsics, gi)
            if pose is None or K is None:
                continue
            Kf = np.asarray(K, dtype=np.float64).reshape(-1)
            if Kf.size == 4:
                fx, fy, cx, cy = Kf
                K3 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            else:
                K3 = Kf.reshape(3, 3)
            fh.write(
                json.dumps(
                    {
                        "frame_index": gi,
                        "cam_to_world": np.asarray(pose, dtype=np.float64)
                        .reshape(4, 4)
                        .tolist(),
                        "intrinsic": K3.tolist(),
                        "image_hw": [int(image_hw[0]), int(image_hw[1])],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            n_rows += 1
    return {
        "tracks_path": tracks_path,
        "poses_path": poses_path,
        "n_tracks": len(records or []),
        "n_pose_rows": n_rows,
    }
