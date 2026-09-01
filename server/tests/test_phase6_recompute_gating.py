import queue
import threading

import numpy as np
import pytest

from conftest import load_streaming_slam_module
from test_phase1_latency import FakeGraph, FakeTimer


class DetectionSubmap:
    def __init__(self, submap_id):
        self.submap_id = submap_id

    def get_id(self):
        return self.submap_id


class DetectionMap:
    def __init__(self, submaps):
        self.submaps = submaps

    def get_submaps(self):
        return self.submaps

    def get_latest_submap(self):
        return self.submaps[-1]

    def get_num_submaps(self):
        return len(self.submaps)

    def get_largest_key(self, ignore_loop_closure_submaps=False):
        return self.submaps[-1].get_id()


class DetectionSolver:
    def __init__(self, detected_loops=None):
        self.vggt_timer = FakeTimer()
        self.loop_closure_timer = FakeTimer()
        self.clip_timer = FakeTimer()
        self.graph = FakeGraph()
        self.map = DetectionMap([DetectionSubmap(1), DetectionSubmap(2)])
        self.detected_loops = detected_loops or []

    def run_predictions(self, image_names, model, max_loops):
        return {"detected_loops": self.detected_loops}

    def add_points(self, predictions):
        self.predictions = predictions


def make_detection_slam(streaming_slam_mod, detected_loops=None):
    slam = streaming_slam_mod.StreamingSLAM.__new__(streaming_slam_mod.StreamingSLAM)
    slam.solver = DetectionSolver(detected_loops=detected_loops)
    slam.model = object()
    slam.max_loops = 1
    slam.overlapping_window_size = 1
    slam.frame_count = 2
    slam._detection_lock = threading.Lock()
    slam.object_enricher = None
    slam._description_cache = {}
    slam._description_lock = threading.Lock()
    slam.active_queries = ["chair"]
    slam.accumulated_detections = []
    slam._sam_cache = {
        (1, 0, "chair"): [
            {
                "bbox_3d": {"center": [0, 0, 0]},
                "seg_score": 0.9,
                "clip_score": 0.8,
            }
        ]
    }
    slam.result_queue = queue.Queue(maxsize=10)
    slam.result_ready_event = None
    slam.event_loop = None
    slam.spatial_agent = None
    slam._last_stream_data = None
    slam.image_names_subset = []
    slam.image_io_times_subset = []
    slam.extract_stream_data_full = lambda: {
        "n_points": 1,
        "n_cameras": 1,
        "num_submaps": 2,
    }
    slam.extract_stream_data_incremental = lambda submap_id: {
        "submap_id": submap_id,
        "n_points": 1,
        "n_cameras": 1,
        "num_submaps": 2,
    }
    return slam


def test_steady_state_detection_skips_global_bbox_recompute(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)
    called_submaps = []
    recompute_calls = []

    slam._run_clip_sam_on_submap = lambda submap, queries: called_submaps.append(submap.get_id()) or []
    slam._recompute_all_bboxes = lambda: recompute_calls.append(True)

    slam._detect_after_submap_update(loop_closure_detected=False)

    assert recompute_calls == []
    assert slam.solver.map.get_latest_submap().get_id() in called_submaps


def test_loop_closure_detection_triggers_global_bbox_recompute(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)
    called_submaps = []
    recompute_calls = []

    slam._run_clip_sam_on_submap = lambda submap, queries: called_submaps.append(submap.get_id()) or []
    slam._recompute_all_bboxes = lambda: recompute_calls.append(True)

    slam._detect_after_submap_update(loop_closure_detected=True)

    assert recompute_calls == [True]
    assert slam.solver.map.get_latest_submap().get_id() in called_submaps


@pytest.mark.parametrize(
    ("detected_loops", "expected_recompute_calls"),
    [
        ([object()], 1),
        ([], 0),
    ],
)
def test_process_submap_passes_loop_closure_flag_to_detection(
    tmp_path,
    monkeypatch,
    detected_loops,
    expected_recompute_calls,
):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod, detected_loops=detected_loops)
    image_path = tmp_path / "frame_000001.jpg"
    image_path.write_bytes(b"jpeg")
    slam.image_names_subset = [str(image_path)]
    slam.image_io_times_subset = [0.001]
    recompute_calls = []

    slam._run_clip_sam_on_submap = lambda submap, queries: []
    slam._recompute_all_bboxes = lambda: recompute_calls.append(True)

    slam.process_submap()

    assert len(recompute_calls) == expected_recompute_calls
