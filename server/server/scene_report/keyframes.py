"""Keyframe selection + encoding for the scene report VLM call.

Selection (``choose_frame_refs``) is pure and import-light so it stays testable;
encoding (``encode_frames``) lazily imports the heavy ``ObjectDetector`` and is
best-effort (any unreadable frame is skipped).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from server.scene_report.schemas import EvidenceRef, ObjectInstance

# Capture-coverage keyframe budget: how many report keyframes to aim for when falling back
# to "spread across the capture" selection (see ``_spread_frame_positions``), decoupled from
# submap/window count. Env-configurable (``SCENE_REPORT_KEYFRAMES``) so it can be tuned
# without a code change; existing callers (``SceneReportBuilder``, ``server/app.py``) pass an
# explicit ``max_frames`` sourced from their own env vars (``SCENE_REPORT_MAX_KEYFRAMES`` /
# ``SCENE_REPORT_PROGRESSIVE_KEYFRAMES``) and are unaffected by this default.
_DEFAULT_SCENE_REPORT_KEYFRAMES = 12


def _default_keyframe_budget() -> int:
    try:
        return max(1, int(os.environ.get("SCENE_REPORT_KEYFRAMES", str(_DEFAULT_SCENE_REPORT_KEYFRAMES))))
    except (TypeError, ValueError):
        return _DEFAULT_SCENE_REPORT_KEYFRAMES


def choose_frame_refs(
    objects: list[ObjectInstance],
    solver,
    max_frames: int | None = None,
    prefer_submap_ids: list[int] | None = None,
) -> list[EvidenceRef]:
    """Pick a bounded, de-duplicated set of (submap, frame) refs.

    Priority: any ``prefer_submap_ids`` first (used by the live/progressive report
    to front-load the newest area), then one frame per detected instance (highest
    confidence first), then enough frames -- spread evenly across the WHOLE capture
    timeline, not just one per submap -- to fill the remaining budget (a
    capture-coverage target, see ``_spread_frame_positions``).

    ``max_frames`` defaults to the ``SCENE_REPORT_KEYFRAMES`` env var (default 12) when not
    given explicitly.
    """
    if max_frames is None:
        max_frames = _default_keyframe_budget()
    seen: set[tuple[int, int]] = set()
    refs: list[EvidenceRef] = []

    def _add(submap_id: int, frame_idx: int) -> None:
        key = (int(submap_id), int(frame_idx))
        if key in seen or key[1] < 0 or key[0] < 0:
            return
        seen.add(key)
        refs.append(EvidenceRef(submap_id=key[0], frame_idx=key[1]))

    for sid in prefer_submap_ids or []:
        try:
            _add(sid, 0)
        except (TypeError, ValueError):
            continue
        if len(refs) >= max_frames:
            return refs

    for obj in sorted(objects, key=lambda o: o.confidence, reverse=True):
        for ev in obj.evidence:
            _add(ev.submap_id, ev.frame_idx)
            if len(refs) >= max_frames:
                return refs

    remaining = max_frames - len(refs)
    if remaining > 0:
        for submap_id, frame_idx in _spread_frame_positions(solver, remaining):
            _add(submap_id, frame_idx)
            if len(refs) >= max_frames:
                break
    return refs[:max_frames]


def _spread_frame_positions(solver, n: int) -> list[tuple[int, int]]:
    """Up to ``n`` ``(submap_id, frame_idx)`` positions evenly spaced across the WHOLE
    capture timeline: every frame of every non-loop-closure submap, in submap-id (capture)
    order, treated as one global sequence.

    Replaces the old "one representative frame per submap" fallback, which capped total
    keyframes at the submap count -- fine at small submaps (~15 windows at ``submap_size=8``
    for a 60s capture) but thinning report coverage once windows got wider (~4-7 windows at
    ``submap_size=32`` for the same capture, see
    ``experiments/research/2026-07-03-omega-full-quality.md`` in the platform repo). This
    targets the requested budget regardless of how many submaps the capture produced.
    """
    smap = getattr(solver, "map", None)
    if smap is None or n <= 0:
        return []
    try:
        submaps = [s for s in smap.ordered_submaps_by_key() if not s.get_lc_status()]
    except Exception:
        return []

    counts: list[int] = []
    for s in submaps:
        try:
            count = len(s.get_frame_ids())
        except Exception:
            count = 0
        counts.append(max(0, count))
    total = sum(counts)
    if total <= 0:
        return []

    def _resolve(global_idx: int) -> tuple[int, int] | None:
        left = global_idx
        for submap, count in zip(submaps, counts):
            if left < count:
                return submap.get_id(), left
            left -= count
        return None

    k = min(n, total)
    if k <= 1:
        raw_positions = [0]
    else:
        raw_positions = [round(i * (total - 1) / (k - 1)) for i in range(k)]

    out: list[tuple[int, int]] = []
    seen_global: set[int] = set()
    for g in raw_positions:
        if g in seen_global:
            continue
        seen_global.add(g)
        resolved = _resolve(g)
        if resolved is not None:
            out.append(resolved)
    return out


def encode_frames(solver, refs: list[EvidenceRef]) -> list[dict[str, Any]]:
    """Turn refs into ``[{submap_id, frame_idx, image_b64}]`` (best effort)."""
    out: list[dict[str, Any]] = []
    smap = getattr(solver, "map", None)
    if smap is None:
        return out
    try:
        from vggt_slam.object_detector import ObjectDetector
    except Exception:
        return out
    for ref in refs:
        try:
            submap = smap.get_submap(ref.submap_id)
            if submap is None:
                continue
            frame_tensor = submap.get_frame_at_index(ref.frame_idx)
            frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            b64 = ObjectDetector.image_to_base64(frame_np)
            out.append(
                {"submap_id": ref.submap_id, "frame_idx": ref.frame_idx, "image_b64": b64}
            )
        except Exception:
            continue
    return out
