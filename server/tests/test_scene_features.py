"""Contract tests for SceneFeatureExtractor (no GPU / model weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

from server.scene_report.features import SceneFeatureExtractor


def _pose(tx, ty, tz):
    p = np.eye(4)
    p[:3, 3] = [tx, ty, tz]
    return p


class FakeSubmap:
    def __init__(self, sid, points=None, poses=None, lc=False):
        self._id = sid
        self._points = None if points is None else np.asarray(points, dtype=float)
        self._poses = None if poses is None else np.asarray(poses, dtype=float)
        self._lc = lc

    def get_id(self):
        return self._id

    def get_lc_status(self):
        return self._lc

    def get_points_in_world_frame(self, graph):
        return self._points

    def get_all_poses_world(self, graph):
        return self._poses


class FakeMap:
    def __init__(self, submaps):
        self._submaps = {s.get_id(): s for s in submaps}

    def ordered_submaps_by_key(self):
        for k in sorted(self._submaps):
            yield self._submaps[k]

    def get_submap(self, sid):
        return self._submaps.get(sid)


class FakeSolver:
    def __init__(self, smap):
        self.map = smap
        self.graph = object()


def _box_points():
    # corners of a [0,4] x [0,3] x [0,2] box
    xs, ys, zs = np.meshgrid([0, 4], [0, 3], [0, 2], indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(float)


def test_metrics_dimensions_trajectory_and_counts():
    submap = FakeSubmap(
        0,
        points=_box_points(),
        poses=np.stack([_pose(0, 0, 0), _pose(1, 0, 0), _pose(2, 0, 0)]),
    )
    facts = SceneFeatureExtractor(percentile=0.0).extract(
        FakeSolver(FakeMap([submap])), detections=[], scene_center=[0, 0, 0]
    )
    assert facts.metrics.num_submaps == 1
    assert facts.metrics.num_keyframes == 3
    # percentile=0 -> exact AABB extents, sorted descending
    assert facts.metrics.dimensions == pytest.approx([4.0, 3.0, 2.0])
    # camera moved 0->1->2 along x
    assert facts.metrics.trajectory_length == pytest.approx(2.0)
    assert facts.metrics.point_count == 8


def test_loop_closure_submap_excluded():
    real = FakeSubmap(0, points=_box_points(), poses=np.stack([_pose(0, 0, 0)]))
    lc = FakeSubmap(1, points=_box_points() + 100.0, poses=np.stack([_pose(50, 0, 0)]), lc=True)
    facts = SceneFeatureExtractor(percentile=0.0).extract(
        FakeSolver(FakeMap([real, lc])), detections=[], scene_center=[0, 0, 0]
    )
    assert facts.metrics.num_submaps == 1  # LC submap skipped
    # bbox must not be inflated by the far-away LC points
    assert max(facts.metrics.bbox_max) <= 4.0 + 1e-6


def test_objects_reanchored_to_world_frame():
    det = {
        "query": "Chair",
        "confidence": 0.8,
        "matched_submap": 2,
        "matched_frame": 5,
        "bounding_box": {"center": [1.0, 0.0, 0.0], "extent": [0.5, 0.5, 0.5]},
    }
    facts = SceneFeatureExtractor().extract(
        FakeSolver(FakeMap([])), detections=[det], scene_center=[10.0, 0.0, 0.0]
    )
    assert len(facts.objects) == 1
    obj = facts.objects[0]
    assert obj.query == "chair"  # normalized
    # recentered center 1.0 + scene_center 10.0 -> world 11.0
    assert obj.center == pytest.approx([11.0, 0.0, 0.0])
    assert obj.evidence[0].submap_id == 2 and obj.evidence[0].frame_idx == 5
    assert facts.object_counts["chair"] == 1


def test_relations_qualified_and_cross_query_only():
    dets = [
        {"query": "tv", "confidence": 0.9, "matched_submap": 0, "matched_frame": 0,
         "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [1, 1, 1]}},
        {"query": "sofa", "confidence": 0.9, "matched_submap": 0, "matched_frame": 1,
         "bounding_box": {"center": [1.0, 0.0, 0.0], "extent": [1, 1, 1]}},
        {"query": "lamp", "confidence": 0.9, "matched_submap": 0, "matched_frame": 2,
         "bounding_box": {"center": [10.0, 0.0, 0.0], "extent": [1, 1, 1]}},
    ]
    facts = SceneFeatureExtractor(near_threshold=1.5, medium_threshold=3.0).extract(
        FakeSolver(FakeMap([])), detections=dets, scene_center=[0, 0, 0]
    )
    rel = {(r.a, r.b): r for r in facts.relations}
    assert rel[("tv", "sofa")].relation == "near"
    assert rel[("tv", "sofa")].distance == pytest.approx(1.0)
    assert rel[("sofa", "lamp")].relation == "far"
    # sorted ascending by distance
    dists = [r.distance for r in facts.relations]
    assert dists == sorted(dists)


def test_empty_scene_is_safe():
    facts = SceneFeatureExtractor().extract(FakeSolver(FakeMap([])), detections=[], scene_center=None)
    assert facts.metrics.num_submaps == 0
    assert facts.objects == []
    assert facts.relations == []
    assert facts.scene_center == [0.0, 0.0, 0.0]


def test_detection_description_flows_into_object_instance():
    det = {
        "query": "laptop",
        "confidence": 0.8,
        "matched_submap": 0,
        "matched_frame": 1,
        "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [0.5, 0.5, 0.5]},
        "description": {
            "title": "Apple MacBook Pro 14\"",
            "brand": "Apple",
            "model": "MacBook Pro M3",
            "confidence": 0.6,
        },
    }
    facts = SceneFeatureExtractor().extract(
        FakeSolver(FakeMap([])), detections=[det], scene_center=[0, 0, 0]
    )
    obj = facts.objects[0]
    assert obj.description is not None
    assert obj.description.brand == "Apple"
    assert obj.description.model == "MacBook Pro M3"


def test_bad_description_is_dropped_not_fatal():
    det = {
        "query": "chair",
        "confidence": 0.5,
        "matched_submap": 0,
        "matched_frame": 0,
        "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [1, 1, 1]},
        "description": "not-a-dict",
    }
    facts = SceneFeatureExtractor().extract(
        FakeSolver(FakeMap([])), detections=[det], scene_center=[0, 0, 0]
    )
    assert facts.objects[0].description is None
