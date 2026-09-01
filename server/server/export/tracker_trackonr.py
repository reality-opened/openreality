"""Track-On-R tracker producer — GPU-free core.

Produces the ``objects_2d`` list that :func:`server.export.dynamics.build_camera_space_tracks`
(strict-gated, camera-space) consumes, from Track-On-R per-**point** online tracks +
visibility. The GPU/torch half (loading Track-On-R and running its causal tracker) lives in
the standalone Modal app ``modal_trackonr_dynamics.py``; the pure, numpy-only, unit-testable
pieces live here — mirroring ``tracker_bootstapir.py`` (the eager-world-lift sibling).

- :func:`fps_select` — farthest-point-sample k pixel coords for spatial spread.
- :func:`seed_query_points` — pick ~N trackable query pixels on a seed frame (Shi-Tomasi
  structure corners + optional caller-provided pixels, e.g. person keypoints or SAM-mask
  interior points). cv2 is imported lazily (only this function needs it).
- :func:`assemble_point_objects_2d` — per-point Track-On-R tracks + visibility -> the
  ``objects_2d`` dict shape the camera-space builder expects (one entry per POINT; Track-On-R
  emits point tracks, not object masks — grouping points into per-object tracks via SAM masks
  is the documented follow-up, see EXP-15 §2 deviation 2).

License posture (EXP-15 §6, C2 memo): Track-On-R code + ``track_on_r.pt`` are MIT; the gated
DINOv3 *backbone* stays server-side (never redistributed) and its License carries **no
outputs/derivative-data clause** — the exported track DATA is unencumbered.

Coordinate convention (matches ``build_camera_space_tracks`` + Track-On-R): query points and
tracks are ``[x, y]`` = ``[col, row]`` pixels in the NATIVE raster fed to the tracker (the
same raster as the SLAM ``depth``/``depth_conf`` maps — no rescale anywhere).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def fps_select(pixels: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sample ``k`` indices from ``(M, 2)`` pixel coords for spatial spread.

    Deterministic (starts at index 0). Returns all indices when ``M <= k``."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    M = pixels.shape[0]
    if M <= int(k):
        return np.arange(M)
    chosen = [0]
    d = np.linalg.norm(pixels - pixels[0], axis=1)
    for _ in range(int(k) - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(pixels - pixels[i], axis=1))
    return np.array(chosen)


def seed_query_points(
    frame0: np.ndarray,
    n_total: int = 20,
    extra_pixels: Optional[np.ndarray] = None,
    extra_provenance: str = "seed_px",
    min_dist_px: float = 12.0,
) -> tuple:
    """Pick ~``n_total`` trackable query pixels on a seed frame.

    ``extra_pixels`` (``(P, 2)`` ``[x, y]``, e.g. SAM-mask interior points or person
    keypoints) are taken first (in-bounds only), then Shi-Tomasi corners fill the rest,
    skipping any candidate within ``min_dist_px`` of an already-chosen point. Seeding is
    deliberately label-free — motion classification happens downstream.

    Returns ``(queries (n, 3) float32 [t=0, x, y], provenance list[str])``.
    """
    import cv2  # lazy: only the seeder needs it (the Modal image ships opencv-headless)

    frame0 = np.asarray(frame0)
    g = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY) if frame0.ndim == 3 else frame0
    H, W = g.shape[:2]

    pts: list = []
    prov: list = []
    if extra_pixels is not None and len(extra_pixels):
        for (x, y) in np.asarray(extra_pixels, dtype=np.float64).reshape(-1, 2):
            if 2 <= x < W - 2 and 2 <= y < H - 2:
                pts.append((float(x), float(y)))
                prov.append(extra_provenance)

    n_corners = max(int(n_total) - len(pts), 0)
    corners: list = []
    if n_corners > 0:
        c = cv2.goodFeaturesToTrack(
            g, maxCorners=n_corners * 3, qualityLevel=0.01,
            minDistance=max(12, W // 22), blockSize=7,
        )
        corners = [] if c is None else [tuple(map(float, p.ravel())) for p in c]
    md2 = float(min_dist_px) ** 2
    for p in corners:
        if len(pts) >= int(n_total):
            break
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > md2 for q in pts):
            pts.append(p)
            prov.append("shi_tomasi")
    pts = pts[: int(n_total)]
    prov = prov[: int(n_total)]
    q = np.array([[0.0, x, y] for (x, y) in pts], dtype=np.float32)
    return q, prov


def assemble_point_objects_2d(
    provenance: Sequence[str],
    frame_indices: Sequence[int],
    tracks: np.ndarray,
    visibles: np.ndarray,
    point_ids: Optional[Sequence[int]] = None,
) -> list:
    """Per-point Track-On-R tracks -> the ``objects_2d`` list for the camera-space builder.

    - ``provenance``: length ``N`` — seed provenance / label per point (becomes ``query``).
    - ``frame_indices``: length ``T`` — the *global keyframe index* (shared clock) of each
      tracked frame, in order.
    - ``tracks``: ``(N, T, 2)`` float ``[x, y]`` px per point per frame (native raster).
    - ``visibles``: ``(N, T)`` bool — Track-On-R's visibility flags. Passed through
      UNTOUCHED: the strict C1 gate happens in :func:`build_camera_space_tracks`, and the 2D
      position on an occluded frame is still recorded here (the builder nulls it).
    - ``point_ids``: optional explicit ids (default ``0..N-1``).

    Returns a list of ``N`` dicts with exactly the seam keys
    ``{id, query, frames, centroids, visibility, confidence}``.
    """
    tr = np.asarray(tracks, dtype=np.float64)
    vs = np.asarray(visibles, dtype=bool)
    if tr.ndim != 3 or tr.shape[2] != 2:
        raise ValueError(f"tracks must be (N, T, 2); got shape {tr.shape!r}")
    N, T = tr.shape[0], tr.shape[1]
    if vs.shape != (N, T):
        raise ValueError(f"visibles shape {vs.shape!r} != (N, T) = {(N, T)!r}")
    if len(frame_indices) != T:
        raise ValueError(f"len(frame_indices) {len(frame_indices)} != T {T}")
    if len(provenance) != N:
        raise ValueError(f"len(provenance) {len(provenance)} != N {N}")
    ids = list(point_ids) if point_ids is not None else list(range(N))
    if len(ids) != N:
        raise ValueError(f"len(point_ids) {len(ids)} != N {N}")

    frames = [int(f) for f in frame_indices]
    objects_2d: list = []
    for n in range(N):
        objects_2d.append(
            {
                "id": int(ids[n]),
                "query": str(provenance[n]),
                "frames": list(frames),
                "centroids": [(float(tr[n, t, 0]), float(tr[n, t, 1])) for t in range(T)],
                "visibility": [bool(vs[n, t]) for t in range(T)],
                "confidence": [1.0 if vs[n, t] else 0.0 for t in range(T)],
            }
        )
    return objects_2d
