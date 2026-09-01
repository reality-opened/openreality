import asyncio
import base64
import queue
import threading
import time

import cv2
import numpy as np

from conftest import load_app_module, load_streaming_slam_module


class FakeTimer:
    def __init__(self):
        self.total_time = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += time.perf_counter() - self.start


class FakeSubmap:
    def __init__(self, submap_id):
        self.submap_id = submap_id

    def get_id(self):
        return self.submap_id


class FakeMap:
    def __init__(self):
        self.latest = FakeSubmap(7)

    def get_latest_submap(self):
        return self.latest

    def get_num_submaps(self):
        return 1

    def get_largest_key(self, ignore_loop_closure_submaps=False):
        return self.latest.get_id()


class FakeGraph:
    def __init__(self):
        self.optimize_called = False

    def optimize(self):
        self.optimize_called = True

    def get_num_loops(self):
        return 0


class FakeFlowTracker:
    def compute_disparity(self, frame, min_disparity):
        return True


class FakeSolver:
    def __init__(self):
        self.vggt_timer = FakeTimer()
        self.loop_closure_timer = FakeTimer()
        self.clip_timer = FakeTimer()
        self.graph = FakeGraph()
        self.map = FakeMap()
        self.flow_tracker = FakeFlowTracker()

    def run_predictions(self, image_names, model, max_loops):
        self.vggt_timer.total_time += 0.011
        self.loop_closure_timer.total_time += 0.022
        self.clip_timer.total_time += 0.033
        return {"detected_loops": []}

    def add_points(self, predictions):
        self.added_predictions = predictions


def make_slam(streaming_slam_mod):
    slam = streaming_slam_mod.StreamingSLAM.__new__(streaming_slam_mod.StreamingSLAM)
    slam.solver = FakeSolver()
    slam.model = object()
    slam.max_loops = 1
    slam.overlapping_window_size = 1
    slam.min_disparity = 30.0
    slam.frame_count = 5
    slam._detection_lock = threading.Lock()
    slam.active_queries = []
    slam.accumulated_detections = []
    slam.result_queue = queue.Queue(maxsize=10)
    slam.result_ready_event = None
    slam.event_loop = None
    slam.spatial_agent = None
    slam._last_stream_data = None
    slam.extract_stream_data_incremental = lambda submap_id: {
        "submap_id": submap_id,
        "n_points": 3,
        "n_cameras": 2,
        "num_submaps": 1,
    }
    slam.extract_stream_data_full = lambda: {"n_points": 3, "n_cameras": 2, "num_submaps": 1}
    return slam


def test_process_submap_logs_phase1_latency_and_stamps_result(tmp_path, capsys, monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    image_path = tmp_path / "frame_000001.jpg"
    image_path.write_bytes(b"jpeg")
    slam = make_slam(streaming_slam_mod)
    slam.image_names_subset = [str(image_path)]
    slam.image_io_times_subset = [0.040]

    slam.process_submap()

    out = capsys.readouterr().out
    assert "[latency] submap=7" in out
    assert "io=40ms" in out
    assert "vggt=11ms" in out
    assert "retrieval=22ms" in out
    assert "clip=33ms" in out
    assert "solver=" in out
    assert "extract=" in out
    assert "detect=0ms" in out
    assert "gpu_busy_pct=" in out
    assert slam.solver.graph.optimize_called

    result = slam.result_queue.get_nowait()
    assert isinstance(result["_put_time"], float)
    assert result["detections"] == []
    assert result["active_queries"] == []


def test_process_loop_records_keyframe_io_time(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_slam(streaming_slam_mod)
    slam.frame_queue = queue.Queue()
    slam.image_names_subset = []
    slam.image_io_times_subset = []
    slam.submap_size = 0
    slam.is_running = True
    slam._stop_event = threading.Event()
    slam._save_frame = lambda frame: "frame_000001.jpg"

    def stop_after_submap():
        slam.is_running = False

    slam.process_submap = stop_after_submap
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok
    slam.frame_queue.put({"image": base64.b64encode(jpeg.tobytes()).decode("ascii")})

    slam.process_loop()

    assert slam.image_names_subset == ["frame_000001.jpg"]
    assert len(slam.image_io_times_subset) == 1
    assert slam.image_io_times_subset[0] >= 0.0


def test_stream_results_logs_broadcast_lag_and_removes_internal_timestamp(
    capsys,
    monkeypatch,
    ):
    app_mod = load_app_module(monkeypatch)
    broadcasted = []
    done = None

    async def fake_broadcast(result):
        broadcasted.append(dict(result))
        done.set()

    app_mod._broadcast_slam_update = fake_broadcast
    app_mod._schedule_agent_cycles_for_result = lambda result: None
    app_mod.client_connected.set()
    app_mod.result_queue.put(
        {
            "_put_time": time.perf_counter() - 0.010,
            "n_points": 1,
            "n_cameras": 1,
            "num_submaps": 1,
        }
    )

    async def run_once():
        nonlocal done
        done = asyncio.Event()
        app_mod._result_ready = asyncio.Event()
        app_mod._result_ready.set()
        task = asyncio.create_task(app_mod.stream_results())
        await asyncio.wait_for(done.wait(), timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_once())

    out = capsys.readouterr().out
    assert "[latency] broadcast_lag=" in out
    assert broadcasted == [{"n_points": 1, "n_cameras": 1, "num_submaps": 1}]
