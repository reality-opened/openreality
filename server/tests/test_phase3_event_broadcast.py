import asyncio
import queue
import types

from conftest import load_app_module, load_streaming_slam_module
from test_phase1_latency import make_slam


class FakeLoop:
    def __init__(self):
        self.callbacks = []

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)
        callback()


class FakeEvent:
    def __init__(self):
        self.set_count = 0

    def set(self):
        self.set_count += 1


def test_result_queue_put_wakes_result_ready_event(tmp_path, monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    image_path = tmp_path / "frame_000001.jpg"
    image_path.write_bytes(b"jpeg")
    slam = make_slam(streaming_slam_mod)
    slam.image_names_subset = [str(image_path)]
    slam.image_io_times_subset = [0.001]
    slam.event_loop = FakeLoop()
    slam.result_ready_event = FakeEvent()

    slam.process_submap()

    assert len(slam.event_loop.callbacks) == 1
    assert slam.result_ready_event.set_count == 1
    assert slam.result_queue.qsize() == 1


def test_stream_results_waits_for_event_and_drains_queue(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    broadcasted = []
    done = None

    async def fake_broadcast(result):
        broadcasted.append(dict(result))
        if len(broadcasted) == 2:
            done.set()

    app_mod._broadcast_slam_update = fake_broadcast
    app_mod._schedule_agent_cycles_for_result = lambda result: None
    app_mod.client_connected.set()
    app_mod.result_queue.put({"n_points": 1, "n_cameras": 1, "num_submaps": 1})
    app_mod.result_queue.put({"n_points": 2, "n_cameras": 2, "num_submaps": 2})

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

    assert broadcasted == [
        {"n_points": 1, "n_cameras": 1, "num_submaps": 1},
        {"n_points": 2, "n_cameras": 2, "num_submaps": 2},
    ]
    assert app_mod.result_queue.empty()


def test_handle_connect_creates_event_and_passes_it_to_slam_processor(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    processor = types.SimpleNamespace(event_loop=None, result_ready_event=None)
    app_mod.slam_processor = processor
    app_mod._ensure_session = lambda sid, *args, **kwargs: types.SimpleNamespace(agent=None)
    app_mod._verify_socket_auth = lambda auth, environ: {"sub": "user-1"}

    async def _claim_ok(user_id, sid):
        return True

    app_mod._claim_user_session = _claim_ok

    async def fake_stream_results():
        await asyncio.sleep(60)

    app_mod.stream_results = fake_stream_results

    async def run_connect():
        await app_mod.handle_connect("sid-1", {"token": "t"}, {})
        assert app_mod._result_ready is not None
        assert processor.event_loop is asyncio.get_running_loop()
        assert processor.result_ready_event is app_mod._result_ready
        app_mod._stream_task.cancel()
        try:
            await app_mod._stream_task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_connect())
