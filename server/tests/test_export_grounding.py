"""Stage-2 grounding tests: evidence join + temporally-localized annotation windows.

See docs/dataset-export.md §4.4-4.5, §7.1. Duck-typed (real pydantic schemas, no GPU).
"""

from __future__ import annotations

import json

from server.export.grounding import (
    _resolve_evidence,
    write_annotations_jsonl,
    write_objects_jsonl,
)
from server.scene_report.schemas import (
    EvidenceRef,
    ObjectDescription,
    ObjectInstance,
    SceneFacts,
    SceneReport,
    SpatialRelation,
)


def _index_map():
    # 5 keyframes: submap 0 frames 0..2 -> 0..2 ; submap 2 frames 0..1 -> 3..4
    return {(0, 0): 0, (0, 1): 1, (0, 2): 2, (2, 0): 3, (2, 1): 4}


def _facts():
    return SceneFacts(
        objects=[
            ObjectInstance(
                query="laptop",
                center=[1.0, 0.0, 0.0],
                extent=[0.5, 0.3, 0.02],
                confidence=0.8,
                evidence=[EvidenceRef(submap_id=0, frame_idx=1)],
                description=ObjectDescription(title='MacBook Pro 14"', category="laptop"),
            ),
            ObjectInstance(
                query="monitor",
                center=[1.5, 0.0, 0.0],
                extent=[0.6, 0.4, 0.05],
                confidence=0.7,
                evidence=[EvidenceRef(submap_id=2, frame_idx=0)],
            ),
            ObjectInstance(
                query="ghost",  # evidence points at a missing/LC frame -> must drop gracefully
                center=[9.0, 9.0, 9.0],
                confidence=0.4,
                evidence=[EvidenceRef(submap_id=99, frame_idx=0)],
            ),
        ],
        relations=[SpatialRelation(a="laptop", b="monitor", distance=0.5, relation="near")],
    )


def test_resolve_evidence_hit_and_miss():
    im = _index_map()
    assert _resolve_evidence(EvidenceRef(submap_id=0, frame_idx=1), im) == 1
    assert _resolve_evidence(EvidenceRef(submap_id=99, frame_idx=0), im) is None


def test_objects_jsonl_evidence_join_and_relations(tmp_path):
    path = str(tmp_path / "objects.jsonl")
    n = write_objects_jsonl(_facts(), detections=[], index_map=_index_map(), path=path)
    assert n == 3
    lines = [json.loads(l) for l in open(path)]

    laptop = lines[0]
    assert laptop["query"] == "laptop"
    assert laptop["evidence"][0]["global_frame_index"] == 1
    assert laptop["world_center"] == [1.0, 0.0, 0.0]
    # relation resolves to the monitor's object id
    assert {r["other_id"] for r in laptop["relations"]} == {1}
    assert laptop["relations"][0]["relation"] == "near"
    assert laptop["description"]["category"] == "laptop"

    # ghost object's unresolved evidence is dropped, leaving an empty evidence list
    ghost = lines[2]
    assert ghost["query"] == "ghost"
    assert ghost["evidence"] == []

    # every emitted global_frame_index is within [0, N)
    N = max(_index_map().values()) + 1
    for line in lines:
        for ev in line["evidence"]:
            assert 0 <= ev["global_frame_index"] < N


def test_objects_jsonl_includes_box_2d_when_detection_has_one(tmp_path):
    path = str(tmp_path / "objects.jsonl")
    detections = [
        {"query": "laptop", "matched_submap": 0, "matched_frame": 1, "box_2d": [10, 20, 30, 40]}
    ]
    write_objects_jsonl(_facts(), detections=detections, index_map=_index_map(), path=path)
    laptop = json.loads(open(path).readline())
    assert laptop["evidence"][0]["box_2d"] == [10, 20, 30, 40]


def test_objects_jsonl_prefers_box_2d_carried_on_evidence(tmp_path):
    # An EvidenceRef that carries the real persisted box_2d/mask_rle should be emitted directly,
    # without needing the detections list (the live dynamics-export path).
    rle = {"size": [4, 4], "counts": [5, 6, 5], "order": "C"}
    facts = SceneFacts(
        objects=[
            ObjectInstance(
                query="laptop",
                center=[1.0, 0.0, 0.0],
                extent=[0.5, 0.3, 0.02],
                confidence=0.8,
                evidence=[
                    EvidenceRef(submap_id=0, frame_idx=1, box_2d=[1.0, 2.0, 3.0, 4.0], mask_rle=rle)
                ],
            )
        ]
    )
    path = str(tmp_path / "objects.jsonl")
    write_objects_jsonl(facts, detections=[], index_map=_index_map(), path=path)
    laptop = json.loads(open(path).readline())
    ev = laptop["evidence"][0]
    assert ev["box_2d"] == [1.0, 2.0, 3.0, 4.0]
    assert ev["mask_rle"] == rle


def test_objects_jsonl_backward_compatible_without_geometry(tmp_path):
    # No detections, no evidence box/mask -> the object line must be byte-identical to the
    # pre-persistence output: existing fields present, no box_2d/mask_rle keys sneak in.
    path = str(tmp_path / "objects.jsonl")
    write_objects_jsonl(_facts(), detections=[], index_map=_index_map(), path=path)
    laptop = json.loads(open(path).readline())
    expected_keys = {
        "object_id", "query", "world_center", "world_extent",
        "confidence", "description", "evidence", "relations",
    }
    assert set(laptop) == expected_keys
    assert set(laptop["evidence"][0]) == {"submap_id", "frame_idx", "global_frame_index"}
    assert "box_2d" not in laptop["evidence"][0]
    assert "mask_rle" not in laptop["evidence"][0]


def test_annotations_windows_within_bounds(tmp_path):
    path = str(tmp_path / "annotations.jsonl")
    report = SceneReport(summary="A home office with a laptop and monitor.")
    n = write_annotations_jsonl(_facts(), report, _index_map(), path, window=3)
    lines = [json.loads(l) for l in open(path)]
    assert n == len(lines)

    N = max(_index_map().values()) + 1
    channels = {l["channel"] for l in lines}
    assert "scene.caption" in channels
    assert "object.caption" in channels
    assert "spatial.relation" in channels

    for line in lines:
        assert 0 <= line["frame_start"] <= line["frame_end"] < N

    # scene.caption spans the whole episode
    scene = next(l for l in lines if l["channel"] == "scene.caption")
    assert scene["frame_start"] == 0 and scene["frame_end"] == N - 1
    assert scene["text"] == "A home office with a laptop and monitor."

    # object.caption for the laptop is windowed ±3 around its evidence frame (global index 1)
    laptop_cap = next(
        l for l in lines if l["channel"] == "object.caption" and "MacBook" in l["text"]
    )
    assert laptop_cap["frame_start"] == 0  # max(0, 1-3)
    assert laptop_cap["frame_end"] == 4  # min(N-1, 1+3)
    assert laptop_cap["evidence"] == {"submap_id": 0, "frame_idx": 1}


def test_annotations_empty_index_map_is_safe(tmp_path):
    path = str(tmp_path / "annotations.jsonl")
    n = write_annotations_jsonl(_facts(), SceneReport(summary="x"), {}, path)
    assert n == 0
    assert open(path).read() == ""
