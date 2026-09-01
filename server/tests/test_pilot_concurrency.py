"""Concurrency-hardening tests for the offline per-scene pilot harness (Codex HIGH-1/2).

These pin the *ordering* that prevents a per-scene reset/finalize from racing an in-flight
VGGT pass: ``reset_for_next_scan`` and ``finalize_scan_blocking`` must wait for the SLAM
processor to go idle BEFORE they wipe (``soft_reset``/``solver.reset``) or flush-stop it.
They also unit-test ``StreamingSLAM.wait_until_idle`` directly (queue-drained AND no submap
in flight). No GPU: the orchestration test uses a recording fake; the idle test binds the
real method to a lightweight fake ``self``. The real GPU-side isolation is only validated by
an actual run — these cover the wrapper logic (the part we can pin deterministically).
"""

from __future__ import annotations

import threading
import time
import types

from conftest import load_app_module, load_streaming_slam_module


class _RecordingSlam:
    """Records the order of the lifecycle calls so a test can assert idle-before-mutate."""

    def __init__(self, idle=True):
        self.calls: list[str] = []
        self._idle = idle
        self.solver = types.SimpleNamespace(reset=lambda: self.calls.append("solver.reset"))
        self.stop_kwargs = None

    def wait_until_idle(self, timeout: float = 120.0) -> bool:
        self.calls.append("wait_until_idle")
        return self._idle

    def soft_reset(self):
        self.calls.append("soft_reset")

    def set_detection_queries(self, q):
        self.calls.append("set_detection_queries")

    def stop(self, flush=False, join_timeout=2.0):
        self.calls.append("stop")
        self.stop_kwargs = {"flush": flush, "join_timeout": join_timeout}

    def finalize_detection_state(self):
        self.calls.append("finalize_detection_state")


def test_reset_for_next_scan_waits_for_idle_before_wipe(monkeypatch):
    app = load_app_module(monkeypatch)
    slam = _RecordingSlam()
    monkeypatch.setattr(app, "slam_processor", slam)
    monkeypatch.setattr(app, "_clear_queues", lambda: slam.calls.append("_clear_queues"))
    monkeypatch.setattr(app, "_clear_scene_report", lambda: None)

    app.reset_for_next_scan()

    # The world-frame wipe (soft_reset) must come AFTER we've confirmed the loop is idle.
    assert "wait_until_idle" in slam.calls
    assert slam.calls.index("wait_until_idle") < slam.calls.index("soft_reset")


def test_finalize_scan_blocking_waits_for_idle_before_stop(monkeypatch):
    app = load_app_module(monkeypatch)
    slam = _RecordingSlam()
    monkeypatch.setattr(app, "slam_processor", slam)
    monkeypatch.setattr(app, "_wait_for_frame_queue_drain", lambda: None)
    monkeypatch.setattr(app, "_build_and_emit_scene_report", lambda sid: None)

    app.finalize_scan_blocking(wait_for_drain=True)

    assert slam.calls.index("wait_until_idle") < slam.calls.index("stop")
    # the offline path uses a generous join so a long VGGT pass fully finishes
    assert slam.stop_kwargs["flush"] is True
    assert slam.stop_kwargs["join_timeout"] >= 60.0


def test_pilot_lock_blocks_concurrent_runs(monkeypatch):
    app = load_app_module(monkeypatch)
    # Holding the pilot lock means a second reconstruct run in the same process is refused.
    assert app._pilot_lock.acquire(blocking=False) is True
    try:
        assert app._pilot_lock.acquire(blocking=False) is False
    finally:
        app._pilot_lock.release()


# -- StreamingSLAM.wait_until_idle logic (real method, lightweight fake self) -------

class _FakeQueue:
    def __init__(self, empty=True):
        self._empty = empty

    def empty(self):
        return self._empty


def _idle_self(streaming_mod, *, processing: bool, queue_empty: bool):
    obj = types.SimpleNamespace()
    obj._processing = threading.Event()
    if processing:
        obj._processing.set()
    obj.frame_queue = _FakeQueue(empty=queue_empty)
    obj._process_thread = None
    # bind the real method
    obj.wait_until_idle = streaming_mod.StreamingSLAM.wait_until_idle.__get__(obj)
    return obj


def test_wait_until_idle_true_when_drained_and_not_processing(monkeypatch):
    mod = load_streaming_slam_module(monkeypatch)
    obj = _idle_self(mod, processing=False, queue_empty=True)
    assert obj.wait_until_idle(timeout=1.0) is True


def test_wait_until_idle_times_out_while_processing(monkeypatch):
    mod = load_streaming_slam_module(monkeypatch)
    obj = _idle_self(mod, processing=True, queue_empty=True)
    t0 = time.time()
    assert obj.wait_until_idle(timeout=0.2) is False  # never idle → times out
    assert time.time() - t0 >= 0.2


def test_wait_until_idle_times_out_with_pending_frames(monkeypatch):
    mod = load_streaming_slam_module(monkeypatch)
    obj = _idle_self(mod, processing=False, queue_empty=False)  # frames still queued
    assert obj.wait_until_idle(timeout=0.2) is False


def test_wait_until_idle_becomes_true_when_processing_clears(monkeypatch):
    mod = load_streaming_slam_module(monkeypatch)
    obj = _idle_self(mod, processing=True, queue_empty=True)
    # clear the processing flag shortly after the wait starts → it should then return True
    threading.Timer(0.1, obj._processing.clear).start()
    assert obj.wait_until_idle(timeout=2.0) is True
