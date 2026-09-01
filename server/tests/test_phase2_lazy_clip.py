import threading

import numpy as np

from conftest import load_streaming_slam_module
from test_phase1_latency import FakeSolver


class SemanticSubmap:
    def __init__(self, submap_id, image_names, semantic_vectors=None):
        self.submap_id = submap_id
        self.img_names = image_names
        self.semantic_vectors = semantic_vectors if semantic_vectors is not None else []
        self.set_vectors_calls = []

    def get_id(self):
        return self.submap_id

    def get_all_semantic_vectors(self):
        return self.semantic_vectors

    def set_all_semantic_vectors(self, semantic_vectors):
        self.semantic_vectors = semantic_vectors
        self.set_vectors_calls.append(semantic_vectors)


class ClipSolver(FakeSolver):
    def __init__(self, submaps):
        super().__init__()
        self.clip_model = None
        self.clip_preprocess = None
        self.submaps = submaps
        self.set_clip_calls = []
        self.reset_called = False

        class Map:
            def __init__(self, submaps):
                self.submaps = submaps

            def get_submaps(self):
                return self.submaps

            def get_num_submaps(self):
                return len(self.submaps)

        self.map = Map(submaps)

    def set_clip_model(self, clip_model, clip_preprocess):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.set_clip_calls.append((clip_model, clip_preprocess))

    def reset(self):
        self.reset_called = True


def make_slam(streaming_slam_mod, submaps):
    slam = streaming_slam_mod.StreamingSLAM.__new__(streaming_slam_mod.StreamingSLAM)
    slam.device = "cpu"
    slam.solver = ClipSolver(submaps)
    slam.object_detector = type(
        "Detector",
        (),
        {"clip_model": "clip-model", "clip_preprocess": "clip-preprocess"},
    )()
    slam.active_queries = []
    slam.accumulated_detections = []
    slam._detection_lock = threading.Lock()
    slam._sam_cache = {}
    slam.object_enricher = None
    slam._description_cache = {}
    slam._description_lock = threading.Lock()
    slam.frame_count = 0
    slam.image_names_subset = []
    slam.image_io_times_subset = []
    slam.pending_beacons = []
    slam.resolved_beacons = []
    slam.latest_scene_center = np.zeros(3)
    slam._last_stream_data = None
    slam._submap_cache = {}
    slam._scene_center = np.zeros(3)
    slam.temp_dir = "/tmp/reality-opened-test"
    slam.spatial_agent = None
    return slam


def test_set_detection_queries_attaches_clip_and_backfills_existing_submaps(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    submap_a = SemanticSubmap(2, ["frame_000002.jpg", "frame_000003.jpg"])
    existing = np.ones((1, 4), dtype=np.float32)
    submap_b = SemanticSubmap(1, ["frame_000001.jpg"], semantic_vectors=existing)
    slam = make_slam(streaming_slam_mod, [submap_a, submap_b])
    calls = []

    def fake_compute(model, preprocess, image_names, device="cuda", **kwargs):
        calls.append((model, preprocess, list(image_names), device))
        return np.full((len(image_names), 4), 0.5, dtype=np.float32)

    monkeypatch.setattr(streaming_slam_mod, "compute_image_embeddings", fake_compute)

    slam.set_detection_queries([" chair "])

    assert slam.active_queries == ["chair"]
    assert slam.solver.set_clip_calls == [("clip-model", "clip-preprocess")]
    assert calls == [
        (
            "clip-model",
            "clip-preprocess",
            ["frame_000002.jpg", "frame_000003.jpg"],
            "cpu",
        )
    ]
    assert submap_a.semantic_vectors.shape == (2, 4)
    assert submap_b.semantic_vectors is existing


def test_set_detection_queries_detaches_clip_when_queries_clear(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_slam(streaming_slam_mod, [])
    slam.solver.clip_model = "clip-model"
    slam.solver.clip_preprocess = "clip-preprocess"
    slam.active_queries = ["chair"]
    slam.accumulated_detections = [{"query": "chair"}]
    slam._sam_cache = {(1, 0, "chair"): ["mask"]}

    slam.set_detection_queries([])

    assert slam.solver.clip_model is None
    assert slam.solver.clip_preprocess is None
    assert slam.accumulated_detections == []
    assert slam._sam_cache == {}


def test_progressive_query_paths_sync_clip(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    submap = SemanticSubmap(1, ["frame_000001.jpg"])
    slam = make_slam(streaming_slam_mod, [submap])
    monkeypatch.setattr(
        streaming_slam_mod,
        "compute_image_embeddings",
        lambda *args, **kwargs: np.ones((1, 3), dtype=np.float32),
    )

    partials = list(slam.add_query_progressive("chair"))
    assert slam.solver.clip_model == "clip-model"
    assert partials[-1]["is_final"] is True

    partials = list(slam.run_detection_progressive([]))
    assert slam.solver.clip_model is None
    assert slam.solver.clip_preprocess is None
    assert partials == [{"detections": [], "is_final": True}]


def test_remove_query_detaches_clip_when_last_query_removed(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_slam(streaming_slam_mod, [])
    slam.active_queries = ["chair"]
    slam.solver.clip_model = "clip-model"
    slam.solver.clip_preprocess = "clip-preprocess"

    slam.remove_query("chair")

    assert slam.active_queries == []
    assert slam.solver.clip_model is None
    assert slam.solver.clip_preprocess is None


def test_soft_reset_does_not_reattach_clip(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_slam(streaming_slam_mod, [])
    slam.solver.clip_model = "clip-model"
    slam.solver.clip_preprocess = "clip-preprocess"
    slam.is_running = False
    slam.stop = lambda: None
    slam.start = lambda: None
    monkeypatch.setattr(streaming_slam_mod.tempfile, "mkdtemp", lambda: "/tmp/reality-opened-reset")

    slam.soft_reset()

    assert slam.solver.reset_called
    assert slam.solver.clip_model is None
    assert slam.solver.clip_preprocess is None
