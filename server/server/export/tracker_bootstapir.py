"""BootsTAPIR tracker producer — GPU-free core.

Produces the ``objects_2d`` list that :func:`server.export.dynamics.build_object_tracks`
consumes, from an object *mask* per keyframe + per-point 2D tracks + per-frame depth. The
GPU/torch half (loading BootsTAPIR and running its causal point tracker) lives in the
standalone Modal app ``modal_dynamics_export.py``; the two pure, numpy-only, unit-testable
pieces live here:

- :func:`sample_query_points_in_mask` — sample up-to-K seed points *inside* an object mask
  on its keyframe (BootsTAPIR is seeded with these query points).
- :func:`assemble_objects_2d` — aggregate the resulting per-point tracks + visibility + depth
  into the ``objects_2d`` dict shape ``build_object_tracks`` expects (per object per frame:
  centroid = median of visible points, depth = median depth at those points, visibility =
  fraction visible, confidence).

numpy-only (no torch, no modal, no cv2), matching the ``server/export/`` invariant so it
unit-tests on the CPU broker / CI with fake tracks. BootsTAPIR is Apache-2.0
(``google-deepmind/tapnet``), so this producer is *shippable* — unlike CoTracker3 (CC-BY-NC).

Coordinate convention (matches ``build_object_tracks`` + BootsTAPIR's ``TapirInference``):
query points and tracks are ``[x, y]`` = ``[col, row]`` in image pixels.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def sample_query_points_in_mask(
    mask: Any, k: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample up to ``k`` seed points strictly INSIDE ``mask`` -> ``(n, 2)`` float32 ``[x, y]``.

    ``mask`` is a 2D boolean / 0-1 array ``(H, W)``. Points are returned in ``[col=x, row=y]``
    pixel order (the BootsTAPIR / ``build_object_tracks`` convention). Every returned point is
    guaranteed to land on a True mask pixel, so ``mask[int(y), int(x)]`` is always True.

    Fewer than ``k`` points come back when the mask has fewer than ``k`` interior pixels (then
    *all* of them are returned); an empty mask returns ``(0, 2)``. Deterministic given ``rng``
    (a ``numpy.random.Generator``; defaults to ``default_rng(0)``).
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {m.shape!r}")
    ys, xs = np.nonzero(m)
    n = int(ys.shape[0])
    k = int(k)
    if n == 0 or k <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(0)
    if n <= k:
        sel = np.arange(n)
    else:
        sel = rng.choice(n, size=k, replace=False)
    # column 0 = x (col), column 1 = y (row)
    return np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)


def _depth_at(depth_map: np.ndarray, pts_xy: np.ndarray) -> float:
    """Median depth of ``depth_map`` sampled at ``pts_xy`` (``[x, y]``), clamped to bounds."""
    dm = np.asarray(depth_map, dtype=np.float64)
    h, w = dm.shape[:2]
    xs = np.clip(np.rint(pts_xy[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(pts_xy[:, 1]).astype(np.int64), 0, h - 1)
    return float(np.median(dm[ys, xs]))


def assemble_objects_2d(
    object_ids: Sequence[int],
    queries: Sequence[str],
    frame_indices: Sequence[int],
    point_ranges: Sequence[tuple],
    tracks: Any,
    visibles: Any,
    depth_maps: Any,
    *,
    min_visible_frac: float = 0.5,
) -> list:
    """Aggregate per-point BootsTAPIR tracks -> the ``objects_2d`` list for build_object_tracks.

    Parameters mirror what the GPU producer collects after one causal BootsTAPIR pass:

    - ``object_ids`` / ``queries``: length ``M`` — one entry per tracked object.
    - ``frame_indices``: length ``T`` — the *global keyframe index* (the shared clock) for each
      tracked frame, in order. These become each object's ``frames`` list.
    - ``point_ranges``: length ``M`` — ``(start, end)`` column slice into the ``N`` points axis
      identifying which seed points belong to each object.
    - ``tracks``: ``(T, N, 2)`` float, ``[x, y]`` pixel coords per point per frame.
    - ``visibles``: ``(T, N)`` bool, BootsTAPIR per-point visibility.
    - ``depth_maps``: per-frame depth — a list of ``T`` ``(H, W)`` arrays, a single ``(H, W)``
      array (broadcast to all frames), or a ``(T, H, W)`` array.

    Per object per frame: ``centroid`` = median of the object's *visible* tracked points
    (falls back to *all* its points when none are visible that frame); ``depth`` = median of
    the frame depth sampled at those points (clamped to image bounds); ``visibility`` = the
    visible fraction ``>= min_visible_frac`` as a bool; ``confidence`` = the visible fraction
    in ``[0, 1]``.

    Returns a list of ``M`` dicts with EXACTLY the keys ``build_object_tracks`` reads:
    ``{id, query, frames, centroids, depths, visibility, confidence}``.
    """
    tr = np.asarray(tracks, dtype=np.float64)
    vs = np.asarray(visibles, dtype=bool)
    T = len(frame_indices)
    if tr.ndim != 3 or tr.shape[2] != 2:
        raise ValueError(f"tracks must be (T, N, 2); got shape {tr.shape!r}")
    if tr.shape[0] != T or vs.shape[0] != T:
        raise ValueError(
            f"tracks/visibles frame count {tr.shape[0]}/{vs.shape[0]} != "
            f"len(frame_indices) {T}"
        )

    # normalise depth_maps -> list of T 2D arrays
    if isinstance(depth_maps, (list, tuple)):
        depth_list = [np.asarray(d, dtype=np.float64) for d in depth_maps]
    else:
        arr = np.asarray(depth_maps, dtype=np.float64)
        depth_list = [arr] * T if arr.ndim == 2 else [arr[f] for f in range(T)]

    objects_2d: list = []
    for oid, query, (start, end) in zip(object_ids, queries, point_ranges):
        frames: list = []
        centroids: list = []
        depths: list = []
        visibility: list = []
        confidence: list = []
        for f in range(T):
            pts = tr[f, start:end]  # (k, 2)
            vis = vs[f, start:end]  # (k,)
            if pts.shape[0] == 0:
                continue
            sel = pts[vis] if bool(np.any(vis)) else pts
            u, v = np.median(sel, axis=0)
            frac = float(np.mean(vis)) if vis.size else 0.0
            frames.append(int(frame_indices[f]))
            centroids.append((float(u), float(v)))
            depths.append(_depth_at(depth_list[f], sel))
            visibility.append(bool(frac >= min_visible_frac))
            confidence.append(frac)
        objects_2d.append(
            {
                "id": int(oid),
                "query": str(query),
                "frames": frames,
                "centroids": centroids,
                "depths": depths,
                "visibility": visibility,
                "confidence": confidence,
            }
        )
    return objects_2d
