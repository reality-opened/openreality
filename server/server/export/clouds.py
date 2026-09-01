"""Point-cloud artifacts: full world-frame cloud (+ optional per-keyframe clouds).

See ``docs/dataset-export.md`` §4 (clouds/). Stage 1 (full cloud) / Stage 5 (per-frame).

Reuse:
- Full cloud: ``gather_full_cloud`` is THE gather walk -- ``StreamingSLAM.gather_world_point_cloud()``
  (``server/streaming_slam.py``) delegates to it, and the offline ``main.py`` path (a bare
  ``Solver``, no ``StreamingSLAM``) calls it directly, so both stay in lock-step. It excludes
  loop-closure re-observation submaps and each non-first submap's carried-over overlap
  frame(s) -- see its docstring (window-quality study,
  ``experiments/research/2026-07-03-omega-full-quality.md`` §2/§6, platform repo).
- Per-frame: ``Submap.get_points_list_in_world_frame(graph)`` (``vggt_slam/submap.py:174``).

The cloud is world-frame and so are ``SceneFacts.objects[*].center`` -- they must stay aligned.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _overlap_point_count(submap: Any, graph: Any, overlap_frames: int) -> int:
    """Count confidence-filtered world points contributed by ``submap``'s first
    ``overlap_frames`` frame(s).

    Those points are the ones duplicated from the immediately preceding non-LC submap's
    carried-over overlap frame(s) -- see ``gather_full_cloud``. Uses the per-frame accessor
    (``Submap.get_points_list_in_world_frame``, also used by ``write_frame_clouds`` below) so
    the count stays in lock-step with ``get_points_in_world_frame``'s own per-frame confidence
    filter and frame-major stacking order: frame 0's (then frame 1's, ...) points are always
    the leading entries of the flat arrays ``get_points_in_world_frame``/``get_points_colors``
    return, so this count is exactly how many leading rows to drop.
    """
    try:
        _points, _frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
    except Exception:
        return 0
    total = 0
    for mask in conf_masks[:overlap_frames]:
        if mask is None:
            continue
        try:
            total += int(np.count_nonzero(mask))
        except Exception:
            continue
    return total


def gather_full_cloud(solver: Any, overlap_frames: int = 1) -> tuple:
    """Aggregate the **complete** world-frame cloud -> ``(positions (N,3) f32, colors (N,3) u8)``.

    Lossless in resolution (no stride / voxel / cap) but EXCLUDES two sources of duplicated
    geometry identified by the window-quality study
    (``experiments/research/2026-07-03-omega-full-quality.md`` §2/§6, platform repo):

    - **Loop-closure (LC) re-observation submaps** (``submap.get_lc_status()`` True). A LC
      submap exists only to anchor a loop-closure edge in the pose graph
      (``Solver.add_points`` in ``vggt_slam/solver.py``) -- its 2 frames are a copy of a query
      frame + the retrieved frame, already covered by their own "regular" submaps. Counting
      them into the fused cloud doubles geometry exactly at revisited places.
    - **The shared overlap frame(s)** at the start of every regular (non-LC) submap after the
      first. The streaming design carries the last ``overlap_frames`` frame(s) of a submap
      forward as the first frame(s) of the next one (to compute the SL(4) junction homography
      -- ``StreamingSLAM.overlapping_window_size`` / ``process_submap``'s
      ``image_names_subset[-overlapping_window_size:]``), so those frames' points would
      otherwise be unprojected and counted twice: once as the tail of the earlier submap, once
      as the head of the later one.

    ``overlap_frames`` should match the caller's overlap window size. Default 1 matches
    ``StreamingSLAM``'s ``overlapping_window_size`` and core's
    ``--overlapping_window_size`` (the only supported value today, per ``core/main.py``'s
    help text) -- correct for both the live/pilot ``StreamingSLAM`` path and the offline
    bare-``Solver`` ``main.py`` path.

    Known gap (not fixed here): core's own ``vggt_slam.splat_export.gather_world_cloud`` walks
    ``ordered_submaps_by_key()`` with neither exclusion, so the 3DGS splat export still carries
    both duplication sources -- a core-side follow-up.
    """
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    graph = getattr(solver, "graph", None)
    smap = getattr(solver, "map", None)
    if smap is None:
        return empty
    try:
        submaps = list(smap.get_submaps())
    except Exception:
        return empty

    regular = []
    for submap in submaps:
        if submap is None:
            continue
        try:
            is_lc = submap.get_lc_status()
        except Exception:
            is_lc = False
        if not is_lc:
            regular.append(submap)
    try:
        regular.sort(key=lambda s: s.get_id())
    except Exception:
        pass
    first_id = regular[0].get_id() if regular else None

    all_pts: list = []
    all_cols: list = []
    for submap in regular:
        try:
            pts = submap.get_points_in_world_frame(graph)
            cols = submap.get_points_colors()
        except Exception:
            continue
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        cols = np.asarray(cols).reshape(-1, 3)
        n = min(pts.shape[0], cols.shape[0])
        if n == 0:
            continue
        pts, cols = pts[:n], cols[:n]

        skip = 0
        if overlap_frames > 0 and submap.get_id() != first_id:
            skip = min(_overlap_point_count(submap, graph, overlap_frames), n)
        if skip:
            pts, cols = pts[skip:], cols[skip:]
        if pts.shape[0] == 0:
            continue
        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        return empty
    positions = np.vstack(all_pts).astype(np.float32)
    colors = np.vstack(all_cols)
    # get_points_colors() is normally [0,255]; tolerate a [0,1] source too.
    if colors.size and float(colors.max()) <= 1.0 + 1e-6:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    return positions, colors


def write_full_cloud(solver: Any, path: str, overlap_frames: int = 1) -> int:
    """Write the complete world-frame cloud to ``path`` (``cloud.npz``: positions + colors).

    Return the point count N.
    """
    positions, colors = gather_full_cloud(solver, overlap_frames=overlap_frames)
    np.savez_compressed(path, positions=positions, colors=colors)
    return int(positions.shape[0])


def write_cloud_array(positions: Any, colors: Any, path: str) -> int:
    """Write an already-gathered ``(positions, colors)`` pair to ``cloud.npz`` (Stage 4 reuse)."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = int(min(positions.shape[0], colors.shape[0]))
    np.savez_compressed(path, positions=positions[:n], colors=colors[:n])
    return n


def write_frame_clouds(solver: Any, index_map: dict, out_dir: str) -> int:
    """OPTIONAL per-keyframe ``clouds/frames/<global_index>.npz`` (points + conf mask).

    Off by default (large) -- gated by ``writer.export_scene(include_frame_clouds=...)``.
    Returns the number of per-frame clouds written.
    """
    os.makedirs(out_dir, exist_ok=True)
    graph = getattr(solver, "graph", None)
    smap = getattr(solver, "map", None)
    if smap is None:
        return 0
    written = 0
    for submap in smap.ordered_submaps_by_key():
        try:
            if submap.get_lc_status():
                continue
        except Exception:
            pass
        sid = int(submap.get_id())
        try:
            point_list, _frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
        except Exception:
            continue
        for local in range(len(point_list)):
            gi = index_map.get((sid, local))
            if gi is None:
                continue
            try:
                np.savez_compressed(
                    os.path.join(out_dir, f"{gi}.npz"),
                    points=np.asarray(point_list[local], dtype=np.float32),
                    mask=np.asarray(conf_masks[local]),
                )
                written += 1
            except Exception as exc:
                print(f"[export] WARN frame cloud write failed (global {gi}): {exc}")
    return written
