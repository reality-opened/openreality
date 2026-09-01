"""Grounding sidecar: scene graph, report, per-object grounding, localized annotations.

See ``docs/dataset-export.md`` §4.3-4.5. Stage 2.

This is the differentiated, action-free V+L product. Everything is grounded: every
``EvidenceRef`` (``server/scene_report/schemas.py:15``) is resolved to a global keyframe
index through the ``index_map`` from ``trajectory.build_index_map`` (§7.1); refs that miss
(e.g. they point at a loop-closure submap) are dropped gracefully.

Reuse the scene-report layer (no re-derivation): ``SceneFacts`` / ``SceneReport``
(``server/scene_report/schemas.py``), produced by ``SceneFeatureExtractor.extract``
(``server/scene_report/features.py:47``) and ``SceneReportBuilder.build_final``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# small access helpers (tolerate both pydantic models and plain dicts)
# ---------------------------------------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_dump(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _dump_json(obj: Any, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(_model_dump(obj), fh, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _resolve_evidence(ref: Any, index_map: dict) -> Optional[int]:
    """``EvidenceRef`` -> global keyframe index, or ``None`` if it does not resolve (§7.1)."""
    if ref is None:
        return None
    try:
        sid = int(_get(ref, "submap_id"))
        fidx = int(_get(ref, "frame_idx"))
    except (TypeError, ValueError):
        return None
    return index_map.get((sid, fidx))


# ---------------------------------------------------------------------------
# §4.3 scene graph + report
# ---------------------------------------------------------------------------

def write_scene_graph(facts: Any, path: str) -> None:
    """``meta/scene_graph.json`` -- ``SceneFacts.model_dump()`` (world-frame object centers)."""
    _dump_json(facts, path)


def write_report(report: Any, path: str) -> None:
    """``meta/report.json`` -- ``SceneReport.model_dump()``."""
    _dump_json(report, path)


# ---------------------------------------------------------------------------
# §4.5 grounding/objects.jsonl
# ---------------------------------------------------------------------------

def _box_lookup(detections: list) -> dict:
    """``(submap_id, frame_idx, query) -> box_2d`` from any detections that carry one."""
    out: dict = {}
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        box = det.get("box_2d")
        if box is None:
            continue
        try:
            key = (
                int(det.get("matched_submap", -1)),
                int(det.get("matched_frame", -1)),
                str(det.get("query", "")).strip().lower(),
            )
        except (TypeError, ValueError):
            continue
        out[key] = list(box)
    return out


def write_objects_jsonl(
    facts: Any,
    detections: list,
    index_map: dict,
    path: str,
) -> int:
    """``grounding/objects.jsonl`` -- per-object grounding tuples (§4.5).

    One line per object: world 3D center/extent, confidence, fine-grained description,
    evidence frames (with resolved ``global_frame_index`` + optional 2D box), and
    relations to other objects. Returns the number of objects written.
    """
    objects = _get(facts, "objects", []) or []
    relations = _get(facts, "relations", []) or []
    boxes = _box_lookup(detections)

    # query -> object ids, for resolving query-keyed relations to concrete object ids.
    ids_by_query: dict = {}
    for i, obj in enumerate(objects):
        ids_by_query.setdefault(_get(obj, "query"), []).append(i)

    written = 0
    with open(path, "w") as fh:
        for oid, obj in enumerate(objects):
            query = _get(obj, "query")
            evidence_out = []
            for ev in _get(obj, "evidence", []) or []:
                gi = _resolve_evidence(ev, index_map)
                if gi is None:
                    continue
                sid = int(_get(ev, "submap_id"))
                fidx = int(_get(ev, "frame_idx"))
                entry = {"submap_id": sid, "frame_idx": fidx, "global_frame_index": gi}
                # Prefer the real 2D detection geometry carried on the EvidenceRef itself
                # (persisted from the live scan); fall back to joining it off the detections
                # list. Both are additive/optional -- absent both, the entry is byte-identical
                # to the pre-persistence output.
                box = _get(ev, "box_2d")
                if box is None:
                    box = boxes.get((sid, fidx, str(query).strip().lower()))
                if box is not None:
                    entry["box_2d"] = list(box)
                mask_rle = _get(ev, "mask_rle")
                if mask_rle is not None:
                    entry["mask_rle"] = mask_rle
                evidence_out.append(entry)

            rels_out = []
            for rel in relations:
                ra, rb = _get(rel, "a"), _get(rel, "b")
                if ra == query:
                    other_q = rb
                elif rb == query:
                    other_q = ra
                else:
                    continue
                for other_id in ids_by_query.get(other_q, []):
                    if other_id == oid:
                        continue
                    rels_out.append(
                        {
                            "other_id": other_id,
                            "distance": float(_get(rel, "distance", 0.0) or 0.0),
                            "relation": _get(rel, "relation", ""),
                        }
                    )

            line = {
                "object_id": oid,
                "query": query,
                "world_center": _as_list(_get(obj, "center")),
                "world_extent": _as_list(_get(obj, "extent")),
                "confidence": float(_get(obj, "confidence", 0.0) or 0.0),
                "description": _model_dump(_get(obj, "description")) or None,
                "evidence": evidence_out,
                "relations": rels_out,
            }
            fh.write(json.dumps(line, default=_json_default) + "\n")
            written += 1
    return written


def _as_list(value: Any) -> list:
    if value is None:
        return []
    try:
        return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# §4.4 meta/annotations.jsonl -- temporally-localized language
# ---------------------------------------------------------------------------

def _num_frames(index_map: dict) -> int:
    return (max(index_map.values()) + 1) if index_map else 0


def _clamp_window(center: int, radius: int, n: int) -> tuple:
    start = max(0, center - radius)
    end = min(n - 1, center + radius)
    return start, end


def _object_caption_text(obj: Any) -> str:
    desc = _get(obj, "description")
    if desc is not None:
        title = _get(desc, "title")
        if title:
            return str(title)
        brand = (_get(desc, "brand") or "").strip()
        model = (_get(desc, "model") or "").strip()
        if brand or model:
            return (brand + " " + model).strip()
        category = _get(desc, "category")
        if category:
            return str(category)
    return str(_get(obj, "query", "") or "object")


def _first_evidence_frame(obj: Any, index_map: dict) -> Optional[tuple]:
    """First resolvable ``(EvidenceRef, global_index)`` for an object, or ``None``."""
    for ev in _get(obj, "evidence", []) or []:
        gi = _resolve_evidence(ev, index_map)
        if gi is not None:
            return ev, gi
    return None


def write_annotations_jsonl(
    facts: Any,
    report: Any,
    index_map: dict,
    path: str,
    window: int = 3,
) -> int:
    """``meta/annotations.jsonl`` -- temporally-localized language (§4.4).

    Channels: ``scene.caption`` (whole episode), ``object.caption`` (windowed ±``window``
    around each object's evidence frame), ``spatial.relation`` (windowed around the two
    objects' evidence). Every window satisfies ``[frame_start, frame_end] ⊆ [0, N)``.
    Returns the number of annotation lines written.
    """
    n = _num_frames(index_map)
    lines: list = []
    if n <= 0:
        _write_jsonl(lines, path)
        return 0

    objects = _get(facts, "objects", []) or []

    # scene.caption -- whole episode, from the report summary.
    summary = (_get(report, "summary", "") or "").strip() if report is not None else ""
    if summary:
        lines.append(
            {
                "frame_start": 0,
                "frame_end": n - 1,
                "channel": "scene.caption",
                "text": summary,
                "evidence": None,
            }
        )

    # object.caption -- windowed around each object's first resolvable evidence frame.
    obj_frame: dict = {}  # object index -> (ref, global_index)
    for i, obj in enumerate(objects):
        hit = _first_evidence_frame(obj, index_map)
        if hit is None:
            continue
        ref, gi = hit
        obj_frame[i] = (ref, gi)
        start, end = _clamp_window(gi, window, n)
        lines.append(
            {
                "frame_start": start,
                "frame_end": end,
                "channel": "object.caption",
                "text": _object_caption_text(obj),
                "evidence": {
                    "submap_id": int(_get(ref, "submap_id")),
                    "frame_idx": int(_get(ref, "frame_idx")),
                },
            }
        )

    # spatial.relation -- windowed around the union of the two objects' evidence frames.
    query_to_idx: dict = {}
    for i, obj in enumerate(objects):
        query_to_idx.setdefault(_get(obj, "query"), i)  # first instance per query
    for rel in _get(facts, "relations", []) or []:
        ia = query_to_idx.get(_get(rel, "a"))
        ib = query_to_idx.get(_get(rel, "b"))
        if ia is None or ib is None:
            continue
        if ia not in obj_frame or ib not in obj_frame:
            continue
        ref_a, ga = obj_frame[ia]
        _ref_b, gb = obj_frame[ib]
        lo, hi = (ga, gb) if ga <= gb else (gb, ga)
        start = max(0, lo - window)
        end = min(n - 1, hi + window)
        lines.append(
            {
                "frame_start": start,
                "frame_end": end,
                "channel": "spatial.relation",
                "text": f"the {_get(rel, 'a')} is {_get(rel, 'relation')} the {_get(rel, 'b')}",
                "evidence": {
                    "submap_id": int(_get(ref_a, "submap_id")),
                    "frame_idx": int(_get(ref_a, "frame_idx")),
                },
            }
        )

    _write_jsonl(lines, path)
    return len(lines)


def _write_jsonl(lines: list, path: str) -> None:
    with open(path, "w") as fh:
        for line in lines:
            fh.write(json.dumps(line, default=_json_default) + "\n")
