"""Minimal deterministic geometry over the persisted scene record (W2 scope).

The "numbers from code" half of the scene agent's guardrail #2: every measurable
quantity an agent utters is computed here (or in ``measure.py``) from the persisted
record — never taken from LLM output. Pure functions over the record dict; no flask,
no store, no LLM.

Scope note: this is the MINIMAL module the agent needs (per-object extents, room bbox,
point/object distances, uid resolution). The fuller demo geometry module (mask lift,
z-buffer occlusion, gravity-constrained OBB fitting for click-segmentation) arrives
with W3/W5 per features.md and extends this file additively.

Conventions (WORLD-TRANSFORM-CONTRACT.md):
  * object identity: ``det:<idx>`` = 0-based index into ``facts.objects`` (the list is
    written once at persist and never reordered, so the index is a stable handle);
  * ``human_label`` wins over the model's ``query`` in ALL prose (the raw ``query`` is
    never overwritten and stays available for audit);
  * lengths are SLAM units — callers pair them with a ``MetricState`` for wording.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

DET_PREFIX = "det:"


def object_uid(index: int) -> str:
    """Registry uid for a facts.objects entry: ``det:<idx>``."""
    return f"{DET_PREFIX}{int(index)}"


def _facts(record: Any) -> dict[str, Any]:
    facts = record.get("facts") if isinstance(record, dict) else None
    return facts if isinstance(facts, dict) else {}


def iter_objects(
    record: Any, include_dismissed: bool = False
) -> list[tuple[str, dict[str, Any]]]:
    """``[(uid, object_dict), ...]`` over ``facts.objects``, dismissed hidden by default.

    The uid is the object's index in the FULL list (dismissed included), so uids stay
    stable regardless of the filter."""
    out: list[tuple[str, dict[str, Any]]] = []
    for idx, obj in enumerate(_facts(record).get("objects") or []):
        if not isinstance(obj, dict):
            continue
        if not include_dismissed and obj.get("dismissed"):
            continue
        out.append((object_uid(idx), obj))
    return out


def known_uids(record: Any, include_dismissed: bool = False) -> set[str]:
    """The closed-world uid set — the ONLY object ids agent findings may cite."""
    return {uid for uid, _ in iter_objects(record, include_dismissed=include_dismissed)}


def object_label(obj: dict[str, Any]) -> str:
    """Display label with ``human_label`` precedence (guardrail #4)."""
    human = obj.get("human_label")
    if isinstance(human, str) and human.strip():
        return human.strip()
    return str(obj.get("query") or "object").strip() or "object"


def object_center(obj: dict[str, Any]) -> Optional[np.ndarray]:
    """World-frame center ``(3,)`` or ``None`` when absent/malformed."""
    center = obj.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return None
    arr = np.asarray(center, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def object_extents(obj: dict[str, Any]) -> Optional[np.ndarray]:
    """FULL side lengths ``(3,)`` (SLAM units) or ``None``. ``facts.objects[].extent``
    already stores full lengths (ObjectInstance.extent)."""
    extent = obj.get("extent")
    if not isinstance(extent, (list, tuple)) or len(extent) != 3:
        return None
    arr = np.asarray(extent, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        return None
    if float(arr.max()) <= 0:
        return None
    return arr


def resolve_object(record: Any, ref: Any) -> Optional[tuple[str, dict[str, Any]]]:
    """Resolve an object reference to ``(uid, obj)`` or ``None``.

    Accepts a ``det:<idx>`` uid, a bare integer index, or a label (matched
    case-insensitively against ``human_label`` then ``query``; ties broken by
    confidence). Dismissed objects resolve ONLY by explicit uid/index — label lookup
    respects the reviewed inventory."""
    objects = _facts(record).get("objects") or []

    def _by_index(idx: int) -> Optional[tuple[str, dict[str, Any]]]:
        if 0 <= idx < len(objects) and isinstance(objects[idx], dict):
            return object_uid(idx), objects[idx]
        return None

    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return _by_index(ref)
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    if not text:
        return None
    if text.startswith(DET_PREFIX):
        try:
            return _by_index(int(text[len(DET_PREFIX):]))
        except ValueError:
            return None
    if text.isdigit():
        return _by_index(int(text))
    # label lookup over the non-dismissed inventory
    needle = text.lower()
    best: Optional[tuple[str, dict[str, Any]]] = None
    best_conf = -1.0
    for uid, obj in iter_objects(record):
        label = object_label(obj).lower()
        query = str(obj.get("query") or "").strip().lower()
        if needle == label or needle == query or needle in label or needle in query:
            conf = float(obj.get("confidence", 0.0) or 0.0)
            exact = needle in (label, query)
            score = conf + (10.0 if exact else 0.0)
            if score > best_conf:
                best_conf = score
                best = (uid, obj)
    return best


def room_bbox(record: Any) -> Optional[dict[str, Any]]:
    """Room AABB from ``facts.metrics`` — ``{bbox_min, bbox_max, dimensions}`` (SLAM
    units, dimensions sorted desc as persisted) or ``None`` when the metrics are absent
    or degenerate. ``vertical_axis_known`` rides along so callers can gate height talk
    (VGGT world is not gravity-aligned; no floor-fit provenance here → no height claims)."""
    metrics = _facts(record).get("metrics")
    if not isinstance(metrics, dict):
        return None
    bbox_min = metrics.get("bbox_min")
    bbox_max = metrics.get("bbox_max")
    dims = metrics.get("dimensions")
    if (
        not isinstance(bbox_min, (list, tuple))
        or not isinstance(bbox_max, (list, tuple))
        or len(bbox_min) != 3
        or len(bbox_max) != 3
    ):
        return None
    lo = np.asarray(bbox_min, dtype=np.float64)
    hi = np.asarray(bbox_max, dtype=np.float64)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    extents = hi - lo
    if float(np.max(np.abs(extents))) <= 0:
        return None
    if not (
        isinstance(dims, (list, tuple))
        and len(dims) == 3
        and all(isinstance(d, (int, float)) and np.isfinite(d) for d in dims)
    ):
        dims = sorted((abs(float(v)) for v in extents), reverse=True)
    return {
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "dimensions": [float(v) for v in dims],
        "vertical_axis_known": bool(metrics.get("vertical_axis_known", False)),
        "ground_frame": ground_frame(record),
    }


def ground_frame(record: Any) -> Optional[dict[str, Any]]:
    """The derived floor/ceiling frame from ``facts.metrics``, or ``None``.

    ``None`` covers three cases that are the same to a caller: the fit never ran, it ran
    and was too weak to publish (``ground_frame.metrics_patch`` writes no heights in that
    case, by design), or the record predates the field. In all three the caller must fall
    back to raw AABB extents and keep saying "not gravity-aligned" — this function is
    what lets height talk be gated on the presence of evidence rather than on a boolean
    sitting next to a plausible-looking number."""
    metrics = _facts(record).get("metrics")
    if not isinstance(metrics, dict) or not metrics.get("vertical_axis_known"):
        return None
    up = metrics.get("up_axis")
    floor_height = metrics.get("floor_height")
    if not isinstance(up, (list, tuple)) or len(up) != 3 or not isinstance(floor_height, (int, float)):
        return None
    axis = np.asarray(up, dtype=np.float64)
    if not np.all(np.isfinite(axis)) or float(np.linalg.norm(axis)) <= 0:
        return None

    def _num(key: str) -> Optional[float]:
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None

    extent = metrics.get("floor_extent")
    return {
        "up_axis": [float(v) for v in axis],
        "floor_height": float(floor_height),
        "ceiling_height": _num("ceiling_height"),
        "room_height": _num("room_height"),
        "floor_extent": [float(v) for v in extent]
        if isinstance(extent, (list, tuple)) and len(extent) == 2
        else None,
        "floor_area": _num("floor_area"),
        "derivation": str(metrics.get("derivation") or "plane-fit"),
    }


def point_distance(a: Any, b: Any) -> float:
    """Plain world-frame point-to-point distance (SLAM units). Raises ``ValueError`` on
    malformed input (callers at the tool boundary catch and rephrase)."""
    pa = np.asarray(a, dtype=np.float64).reshape(3)
    pb = np.asarray(b, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        raise ValueError("points must be finite [x, y, z]")
    return float(np.linalg.norm(pa - pb))


def scene_stats(record: Any) -> dict[str, Any]:
    """The survey-phase numbers, computed from the record (never the LLM): counts,
    coverage, extents. Missing pieces come back as 0/None rather than raising."""
    facts = _facts(record)
    metrics = facts.get("metrics") if isinstance(facts.get("metrics"), dict) else {}
    objects = [o for o in (facts.get("objects") or []) if isinstance(o, dict)]
    dismissed = sum(1 for o in objects if o.get("dismissed"))
    keyframes = record.get("keyframes") if isinstance(record, dict) else None
    synthetic = record.get("synthetic_views") if isinstance(record, dict) else None
    bbox = room_bbox(record)
    return {
        "object_count": len(objects),
        "dismissed_count": dismissed,
        "active_object_count": len(objects) - dismissed,
        "object_types": len(facts.get("object_counts") or {}),
        "num_submaps": int(metrics.get("num_submaps", 0) or 0),
        "num_keyframes": int(metrics.get("num_keyframes", 0) or 0),
        "stored_keyframes": len(keyframes) if isinstance(keyframes, list) else 0,
        # Renders of the scene's own splat, counted SEPARATELY from keyframes forever
        # (server/oreos/synthetic_views.py) — never summed into stored_keyframes.
        "synthetic_view_count": len(synthetic) if isinstance(synthetic, list) else 0,
        "point_count": int(record.get("point_count", 0) or 0) if isinstance(record, dict) else 0,
        "coverage_estimate": float(facts.get("coverage_estimate", 0.0) or 0.0),
        "trajectory_count": int(record.get("trajectory_count", 0) or 0)
        if isinstance(record, dict)
        else 0,
        "has_trajectory": bool(record.get("trajectory_key")) if isinstance(record, dict) else False,
        "has_splat": bool(record.get("splat_key")) if isinstance(record, dict) else False,
        "room_dimensions": bbox["dimensions"] if bbox else None,
        "vertical_axis_known": bool(bbox["vertical_axis_known"]) if bbox else False,
        "ground_frame": bbox["ground_frame"] if bbox else None,
    }
