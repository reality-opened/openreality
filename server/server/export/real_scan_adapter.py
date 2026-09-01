"""Real-export -> dynamics-producer input adapter (GPU-free core).

Assembles the inputs the BootsTAPIR dynamics producer needs
(:func:`server.export.dynamics.build_object_tracks`) **from a real OpenReality export tree**
written by ``server/export/writer.py``. This is the seam that replaces the synthetic smoke
inputs in ``modal_dynamics_export.py`` with what an actual scan persisted.

What a real export genuinely contains (and this module reads directly):

- ``data/trajectory.parquet``  -> per-keyframe **cam->world pose** (``observation.state`` =
  ``[tx,ty,tz,qx,qy,qz,qw]``, ``trajectory._pose_to_state``) + **intrinsics**
  (``observation.intrinsics`` = ``[fx,fy,cx,cy]``). Row ``i`` == global keyframe index ``i``.
- ``videos/ego_view.mp4``       -> the per-keyframe **RGB frames**, frame ``i`` == row ``i`` (§7.4).
- ``grounding/objects.jsonl``   -> per **object**: ``world_center`` (3), ``world_extent`` (3),
  ``query``, and ``evidence[*].global_frame_index`` (the keyframe the object was detected on).
- ``clouds/cloud.npz``          -> the fused **world-frame point cloud** (positions + colors).

What a real export does **NOT** contain (the honest gaps this adapter must derive):

- **Per-frame object masks.** SAM3 masks live only in ``StreamingSLAM._sam_cache`` during a
  scan; ``_dedup_and_store`` drops ``mask_2d``/``box_2d`` (``mask_image=None``) before export.
  There is no 2D region on disk -- only the 3D world box + the evidence keyframe index. So a
  seed region is **derived** by projecting the object's 3D world box into its evidence
  keyframe (:func:`project_world_box_to_image`) and, on the GPU side, box-prompting **SAM2**
  (Apache-2.0) for a real mask. The rectangular projected box (:func:`box_to_mask`) is the
  GPU-free fallback seed (clearly a coarser seed, not a fabricated mask).
- **Per-frame depth.** Only the fused world cloud is persisted. Recomputing raw VGGT depth
  would be in VGGT's *own* per-image frame/scale (SPARK quirk, ``core/modal_app.py``), which
  is inconsistent with the export's SLAM-world poses. The scale-consistent real depth source
  is the fused cloud itself: :func:`render_cloud_depth` z-buffers the world cloud into each
  keyframe (world scale, same frame as the poses) so the unprojection round-trips.

numpy/pandas/cv2 only -- no torch, no modal. The GPU orchestration (SAM2 masks + BootsTAPIR
tracking) that consumes these arrays lives in ``modal_dynamics_export.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# trajectory.parquet -> poses_c2w + intrinsics (per global keyframe index)
# ---------------------------------------------------------------------------

def _state_to_pose_c2w(state: Any) -> np.ndarray:
    """``[tx,ty,tz,qx,qy,qz,qw]`` (cam->world, ``trajectory._pose_to_state``) -> 4x4 cam->world."""
    from scipy.spatial.transform import Rotation

    s = np.asarray(state, dtype=np.float64).reshape(-1)
    t = s[:3]
    quat = s[3:7]  # xyzw, matching Rotation.as_quat()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = t
    return T


def load_trajectory(export_dir: str) -> dict:
    """Read ``data/trajectory.parquet`` -> ``{poses, intrinsics, source_frame_id, n}``.

    ``poses`` / ``intrinsics`` are dicts keyed by the **global keyframe index** (the parquet
    ``index`` column, which is the shared clock ``build_object_tracks`` looks poses up by):
    ``poses[gi]`` is a 4x4 cam->world matrix; ``intrinsics[gi]`` is ``[fx,fy,cx,cy]``.
    """
    import pandas as pd

    path = os.path.join(export_dir, "data", "trajectory.parquet")
    df = pd.read_parquet(path)
    poses: dict = {}
    intrinsics: dict = {}
    source_ids: dict = {}
    for _, row in df.iterrows():
        gi = int(row["index"])
        poses[gi] = _state_to_pose_c2w(row["observation.state"])
        intrinsics[gi] = np.asarray(row["observation.intrinsics"], dtype=np.float64).reshape(-1)
        source_ids[gi] = float(row["source_frame_id"]) if "source_frame_id" in row else float(gi)
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "source_frame_id": source_ids,
        "n": int(len(poses)),
    }


# ---------------------------------------------------------------------------
# videos/ego_view.mp4 -> per-keyframe frames (frame i == global keyframe index i)
# ---------------------------------------------------------------------------

def load_frames(export_dir: str, *, as_bgr: bool = True) -> list:
    """Decode ``videos/ego_view.mp4`` -> list of ``(H, W, 3)`` uint8 frames in global order.

    Frame ``i`` is global keyframe index ``i`` (§7.4 guarantees mp4 frame ``i`` == trajectory
    row ``i``). Returned **BGR** by default -- the convention ``tracks_from_masks`` expects
    (its ``preprocess_frame`` flips BGR->RGB internally). Pass ``as_bgr=False`` for RGB.
    """
    import cv2

    path = os.path.join(export_dir, "videos", "ego_view.mp4")
    cap = cv2.VideoCapture(path)
    frames: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame if as_bgr else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# grounding/objects.jsonl -> per-object 3D box + evidence keyframe(s)
# ---------------------------------------------------------------------------

def load_objects(export_dir: str) -> list:
    """Read ``grounding/objects.jsonl`` -> per-object dicts the adapter can seed from.

    Each returned dict::

        {"object_id", "query", "world_center" (3,), "world_extent" (3,),
         "confidence", "evidence_frames": [global_frame_index, ...],
         "evidence": [{"global_frame_index", "box_2d"|None, "mask_rle"|None}, ...]}

    ``evidence_frames`` are the resolved global keyframe indices the object was detected on
    (its seed keyframe(s)); objects with no resolvable evidence are still returned (empty list)
    so the caller can decide to skip them.

    ``evidence`` mirrors those keyframes but *also* carries the **real per-frame SAM detection
    geometry** when the scan persisted it: ``box_2d`` (``[x0,y0,x1,y1]`` pixels) and/or
    ``mask_rle`` (``server/export/mask_rle.py``). Both are optional -- older exports simply
    have them ``None`` and the caller falls back to projecting the 3D box (see
    :func:`object_seed`). ``contracts/export-format.md`` documents these fields.
    """
    path = os.path.join(export_dir, "grounding", "objects.jsonl")
    objects: list = []
    if not os.path.exists(path):
        return objects
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ev_frames = []
            evidence = []
            for ev in rec.get("evidence", []) or []:
                gi = ev.get("global_frame_index")
                if gi is None:
                    continue
                gi = int(gi)
                ev_frames.append(gi)
                box = ev.get("box_2d")
                evidence.append(
                    {
                        "global_frame_index": gi,
                        "box_2d": [float(v) for v in box]
                        if isinstance(box, (list, tuple)) and len(box) == 4
                        else None,
                        "mask_rle": ev.get("mask_rle")
                        if isinstance(ev.get("mask_rle"), dict)
                        else None,
                    }
                )
            objects.append(
                {
                    "object_id": int(rec.get("object_id", 0)),
                    "query": str(rec.get("query", "") or ""),
                    "world_center": np.asarray(
                        rec.get("world_center") or [], dtype=np.float64
                    ).reshape(-1),
                    "world_extent": np.asarray(
                        rec.get("world_extent") or [], dtype=np.float64
                    ).reshape(-1),
                    "confidence": float(rec.get("confidence", 0.0) or 0.0),
                    "evidence_frames": ev_frames,
                    "evidence": evidence,
                }
            )
    return objects


# ---------------------------------------------------------------------------
# clouds/cloud.npz -> fused world-frame cloud
# ---------------------------------------------------------------------------

def load_cloud(export_dir: str) -> tuple:
    """Read the fused ``clouds/cloud.npz`` -> ``(positions (N,3) f64, colors (N,3) u8))``."""
    path = os.path.join(export_dir, "clouds", "cloud.npz")
    if not os.path.exists(path):
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)
    z = np.load(path)
    positions = np.asarray(z["positions"], dtype=np.float64).reshape(-1, 3)
    colors = (
        np.asarray(z["colors"]).reshape(-1, 3)
        if "colors" in z
        else np.zeros((positions.shape[0], 3), np.uint8)
    )
    return positions, colors


# ---------------------------------------------------------------------------
# geometry: 3D world box -> 2D image box  (exact inverse of dynamics.unproject_to_world)
# ---------------------------------------------------------------------------

def _intrinsics_params(K: Any) -> tuple:
    arr = np.asarray(K, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    K3 = arr.reshape(3, 3)
    return float(K3[0, 0]), float(K3[1, 1]), float(K3[0, 2]), float(K3[1, 2])


def project_world_points(pts_world: np.ndarray, pose_c2w: np.ndarray, K: Any) -> tuple:
    """Project world points -> ``(uv (M,2), z (M,), valid (M,) bool)``.

    Exact inverse of ``dynamics.unproject_to_world``: ``p_cam = inv(pose_c2w) @ [Pw,1]``,
    ``u = Xc/Zc*fx + cx``, ``v = Yc/Zc*fy + cy``; ``valid`` iff the point is in front
    (``Zc > 0``). Camera looks down +z, matching the export pose convention.
    """
    fx, fy, cx, cy = _intrinsics_params(K)
    P = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    w2c = np.linalg.inv(np.asarray(pose_c2w, dtype=np.float64).reshape(4, 4))
    hom = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)  # (M,4)
    cam = (w2c @ hom.T).T[:, :3]  # (M,3)
    z = cam[:, 2]
    valid = z > 1e-6
    zc = np.where(valid, z, 1.0)
    u = cam[:, 0] / zc * fx + cx
    v = cam[:, 1] / zc * fy + cy
    return np.stack([u, v], axis=1), z, valid


def _box_corners(center: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """8 corners of an axis-aligned world box (``extent`` is the full side length)."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.abs(np.asarray(extent, dtype=np.float64).reshape(3)) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
    )
    return c[None, :] + signs * h[None, :]


def project_world_box_to_image(
    center: Any, extent: Any, pose_c2w: Any, K: Any, height: int, width: int
) -> Optional[tuple]:
    """Project a 3D world box into a keyframe -> 2D pixel box ``(x0, y0, x1, y1)`` or ``None``.

    Projects the 8 axis-aligned corners with :func:`project_world_points`, keeps those in
    front of the camera, and returns the axis-aligned pixel bbox clipped to the image. Returns
    ``None`` if no corner is in front of the camera (object behind the evidence view -- should
    not happen for a real detection keyframe, but handled gracefully).
    """
    corners = _box_corners(center, extent)
    uv, _z, valid = project_world_points(corners, pose_c2w, K)
    uv = uv[valid]
    if uv.shape[0] == 0:
        return None
    x0 = float(np.clip(np.floor(uv[:, 0].min()), 0, width - 1))
    y0 = float(np.clip(np.floor(uv[:, 1].min()), 0, height - 1))
    x1 = float(np.clip(np.ceil(uv[:, 0].max()), 0, width - 1))
    y1 = float(np.clip(np.ceil(uv[:, 1].max()), 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def box_to_mask(box: tuple, height: int, width: int) -> np.ndarray:
    """Rectangular boolean ``(H, W)`` mask from a pixel box ``(x0,y0,x1,y1)``.

    The GPU-free **fallback** seed when SAM2 is unavailable: a coarse region derived from the
    real projected 3D box (not a fabricated mask). BootsTAPIR seeds points inside it.
    """
    m = np.zeros((int(height), int(width)), dtype=bool)
    if box is None:
        return m
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    m[y0 : y1 + 1, x0 : x1 + 1] = True
    return m


# ---------------------------------------------------------------------------
# seed region for one object on its evidence keyframe -- PREFER persisted inputs
# ---------------------------------------------------------------------------

def _evidence_at(obj: dict, gi: int) -> dict:
    """The per-evidence dict (from :func:`load_objects`) for global keyframe ``gi``, or ``{}``."""
    for ev in obj.get("evidence") or []:
        if int(ev.get("global_frame_index", -1)) == int(gi):
            return ev
    return {}


def _clip_box(box: Any, height: int, width: int) -> Optional[tuple]:
    """Clamp a ``[x0,y0,x1,y1]`` box into the image; ``None`` if degenerate."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0 = float(np.clip(x0, 0, width - 1))
    y0 = float(np.clip(y0, 0, height - 1))
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def object_seed(
    obj: dict, gi: int, poses: dict, intrinsics: dict, height: int, width: int
) -> dict:
    """Best seed for one object on evidence keyframe ``gi`` -- **prefer persisted inputs**.

    Precedence (most→least accurate), reporting ``source`` so the caller can log it honestly:

    1. ``"persisted_mask"`` -- the scan persisted a real SAM ``mask_rle`` for this frame: decode
       and use it **directly** (``mask`` set, ``box`` = its tight bbox). The GPU caller SKIPS SAM2.
    2. ``"persisted_box"`` -- the scan persisted the real 2D detection ``box_2d``: seed SAM2 with
       **that** (tighter than the projected 3D box). ``box`` set, ``mask`` = ``None``.
    3. ``"projected_box"`` -- neither persisted (older export): fall back to projecting the 3D
       world box into the keyframe (:func:`project_world_box_to_image`). ``box`` set (or ``None``
       if the object is behind the view), ``mask`` = ``None``.

    Returns ``{"box": (x0,y0,x1,y1)|None, "mask": ndarray|None, "source": str}``.
    """
    ev = _evidence_at(obj, gi)

    # 1. persisted mask -> use directly.
    rle = ev.get("mask_rle")
    if isinstance(rle, dict):
        try:
            from server.export.mask_rle import rle_to_mask

            mask = rle_to_mask(rle)
        except Exception:
            mask = None
        if mask is not None and mask.shape == (int(height), int(width)) and bool(mask.any()):
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            return {"box": box, "mask": mask, "source": "persisted_mask"}

    # 2. persisted 2D detection box -> seed from it (tighter than the projected 3D box).
    box = _clip_box(ev.get("box_2d"), height, width)
    if box is not None:
        return {"box": box, "mask": None, "source": "persisted_box"}

    # 3. fall back: project the object's 3D world box into this keyframe.
    box = project_world_box_to_image(
        obj.get("world_center"), obj.get("world_extent"), poses[gi], intrinsics[gi], height, width
    )
    return {"box": box, "mask": None, "source": "projected_box"}


# ---------------------------------------------------------------------------
# per-frame depth from the fused world cloud (scale-consistent with the poses)
# ---------------------------------------------------------------------------

def render_cloud_depth(
    positions: np.ndarray,
    pose_c2w: Any,
    K: Any,
    height: int,
    width: int,
    *,
    fill: bool = True,
) -> Optional[np.ndarray]:
    """Z-buffer the fused world cloud into one keyframe -> dense ``(H, W)`` depth (world scale).

    Projects every world point into the camera, keeps in-front + in-bounds points, and keeps
    the **nearest** ``Zc`` per pixel (a real, up-to-scale depth in the *same* frame as the
    export poses -- so ``unproject_to_world`` round-trips). Empty pixels are nearest-neighbour
    filled (``fill=True``, needs scipy) so BootsTAPIR-tracked points always sample a finite
    depth. Returns ``None`` if no cloud point projects into the frame.
    """
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        return None
    uv, z, valid = project_world_points(P, pose_c2w, K)
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    inb = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inb):
        return None
    u, v, zc = u[inb], v[inb], z[inb]
    depth = np.full((int(height), int(width)), np.inf, dtype=np.float64)
    # nearest-z wins per pixel (sort far->near so the nearest write lands last)
    order = np.argsort(-zc)
    depth[v[order], u[order]] = zc[order]
    depth[~np.isfinite(depth)] = np.nan
    if fill and np.any(np.isnan(depth)):
        depth = _fill_nan_nearest(depth)
    return depth


def _fill_nan_nearest(depth: np.ndarray) -> np.ndarray:
    """Nearest-valid fill of NaN holes (scipy EDT); no-op if fully NaN or scipy missing."""
    invalid = np.isnan(depth)
    if not np.any(invalid) or np.all(invalid):
        return depth
    try:
        from scipy import ndimage

        _dist, (iy, ix) = ndimage.distance_transform_edt(
            invalid, return_distances=True, return_indices=True
        )
        return depth[iy, ix]
    except Exception:
        # scipy unavailable: fill with the global median so sampling stays finite.
        med = float(np.nanmedian(depth))
        out = depth.copy()
        out[invalid] = med
        return out


def render_cloud_depth_stack(
    positions: np.ndarray,
    poses: dict,
    intrinsics: dict,
    frame_indices: list,
    height: int,
    width: int,
    *,
    fill: bool = True,
) -> list:
    """Per-frame :func:`render_cloud_depth` over ``frame_indices`` -> list of ``(H,W)`` depths.

    A missing/empty projection yields a constant-median fallback plane so the stack length
    always matches ``frame_indices`` (the depth list ``tracks_from_masks`` indexes by frame).
    """
    depths: list = []
    for gi in frame_indices:
        d = render_cloud_depth(positions, poses[gi], intrinsics[gi], height, width, fill=fill)
        if d is None:
            d = np.full((int(height), int(width)), np.nan, dtype=np.float64)
        depths.append(d)
    return depths
