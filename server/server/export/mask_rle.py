"""Compact, dependency-free run-length codec for 2D boolean object masks.

Used to persist the per-frame SAM object masks that ``StreamingSLAM`` already computes
(``_sam_cache`` entries carry ``mask_2d``) through the export contract so the dynamics
producer (``server/export/real_scan_adapter.py``) can seed BootsTAPIR directly instead of
re-deriving a coarse region from the projected 3D box. See ``contracts/export-format.md``.

Format (a small JSON-friendly dict, deliberately NOT tied to pycocotools so it adds no dep)::

    {"size": [H, W], "counts": [c0, c1, ...], "order": "C"}

``counts`` are the lengths of alternating runs in **row-major (C) order**, starting with a
run of ``False``; if the mask starts with ``True`` the first count is ``0``. This mirrors the
COCO RLE idea but in C order and with plain Python ints so it round-trips through ``json``.

numpy only. Everything is best-effort and guarded: a mask that cannot be encoded yields
``None`` (the caller simply persists no mask and the consumer falls back), so this never
breaks the live streaming pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _to_bool_2d(mask: Any) -> Optional[np.ndarray]:
    """Coerce a mask (numpy array, torch tensor, list) to a 2D bool ndarray, or ``None``."""
    if mask is None:
        return None
    # torch tensor (or anything with .detach()/.cpu()) -> numpy, without importing torch.
    if hasattr(mask, "detach"):
        try:
            mask = mask.detach()
        except Exception:
            pass
    if hasattr(mask, "cpu"):
        try:
            mask = mask.cpu()
        except Exception:
            pass
    try:
        arr = np.asarray(mask)
    except Exception:
        return None
    # Drop a leading/trailing singleton channel or batch dim (e.g. (1,H,W) or
    # (H,W,1)) WITHOUT collapsing a genuine 2D mask that has a size-1 side
    # (e.g. (1,N) or (N,1)) — np.squeeze would over-collapse the latter.
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr.astype(bool)


def mask_to_rle(mask: Any) -> Optional[dict]:
    """Encode a 2D boolean mask -> RLE dict (``{"size","counts","order"}``) or ``None``.

    Returns ``None`` for anything that is not a genuine 2D mask so callers can guard with a
    plain truthiness check and simply persist nothing.
    """
    arr = _to_bool_2d(mask)
    if arr is None:
        return None
    h, w = int(arr.shape[0]), int(arr.shape[1])
    flat = arr.ravel(order="C")
    if flat.size == 0:
        return {"size": [h, w], "counts": [], "order": "C"}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if bool(flat[0]):  # mask starts with True -> lead with an empty False run
        counts = [0] + counts
    return {"size": [h, w], "counts": [int(c) for c in counts], "order": "C"}


def rle_to_mask(rle: Any) -> Optional[np.ndarray]:
    """Decode an RLE dict (as produced by :func:`mask_to_rle`) -> 2D bool ndarray, or ``None``."""
    if not isinstance(rle, dict):
        return None
    size = rle.get("size")
    counts = rle.get("counts")
    if not size or counts is None or len(size) != 2:
        return None
    try:
        h, w = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for c in counts:
        try:
            c = int(c)
        except (TypeError, ValueError):
            return None
        if value and c > 0:
            flat[pos : pos + c] = True
        pos += c
        value = not value
    if pos != h * w:
        return None
    return flat.reshape(h, w, order="C")


def rle_area(rle: Any) -> int:
    """Number of set pixels in an RLE dict (cheap; no full decode)."""
    if not isinstance(rle, dict):
        return 0
    counts = rle.get("counts") or []
    # runs alternate starting with False: odd-indexed runs are the True runs.
    return int(sum(int(c) for i, c in enumerate(counts) if i % 2 == 1))
