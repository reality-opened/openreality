"""VideoFeeder frame_queue backpressure policy (realtime vs offline/batch).

``VideoFeeder._feed_loop`` pushes decoded frames into the module-global ``frame_queue``.
When the queue is full, the *live* path (a real camera, or a viewer watching a demo
video in real time) must never backpressure the producer — it drops the frame
(pre-existing behavior, unchanged). The *offline/batch* path (the pilot reconstruction
harness, ``reconstruct_pilot.py``) has no live camera/viewer to protect, so losing a
keyframe there is a silent, nondeterministic quality hit on long captures (observed
2/360 dropped on a 180s capture) — it should block for room instead.

These tests fake out ``cv2.VideoCapture``/``cv2.imencode`` (no real video decoding) and
the module-global ``frame_queue`` (a tiny fake queue that can simulate transient or
permanent fullness), then drive the real ``VideoFeeder._feed_loop`` synchronously.
"""

import queue
import threading
import time

from conftest import load_app_module


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in: yields ``n_frames`` placeholder frames."""

    def __init__(self, n_frames=3, fps=2.0):
        self._left = n_frames
        self._n_frames = n_frames
        self._fps = fps

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._n_frames
        if prop == cv2.CAP_PROP_ORIENTATION_META:
            return 0
        return 0

    def read(self):
        if self._left <= 0:
            return False, None
        self._left -= 1
        return True, "fake-frame"

    def release(self):
        pass


class _FakeEncodedBuf:
    def tobytes(self):
        return b"fake-jpeg-bytes"


def _install_fake_video(app, monkeypatch, n_frames=3):
    monkeypatch.setattr(app.cv2, "VideoCapture", lambda path: _FakeCapture(n_frames=n_frames))
    monkeypatch.setattr(app.cv2, "imencode", lambda ext, frame, params: (True, _FakeEncodedBuf()))
    # No SLAM processor to auto-start; keep _feed_loop from touching it.
    monkeypatch.setattr(app, "slam_processor", None)


def _write_dummy_video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"\x00\x01\x02\x03")  # non-empty, not a git-lfs pointer
    return str(path)


class _AlwaysFullQueue:
    """Every put() raises queue.Full — simulates a consumer that never catches up."""

    def __init__(self):
        self.put_calls = 0

    def put(self, item, timeout=None):
        self.put_calls += 1
        raise queue.Full()


class _FullThenOkQueue:
    """The first ``full_for`` put() calls raise queue.Full, then puts succeed —
    simulates a consumer that is briefly behind but drains in time."""

    def __init__(self, full_for):
        self.full_for = full_for
        self.put_calls = 0
        self.items = []

    def put(self, item, timeout=None):
        self.put_calls += 1
        if self.put_calls <= self.full_for:
            raise queue.Full()
        self.items.append(item)


def test_realtime_default_drops_frames_when_queue_stays_full(monkeypatch, tmp_path, capsys):
    """Unchanged pre-existing behavior: a live/demo feed (realtime=True, the default)
    drops a frame rather than blocking when the queue never drains."""
    app = load_app_module(monkeypatch)
    _install_fake_video(app, monkeypatch, n_frames=3)
    fake_q = _AlwaysFullQueue()
    monkeypatch.setattr(app, "frame_queue", fake_q)

    feeder = app.VideoFeeder(_write_dummy_video(tmp_path), fast=True, target_fps=2.0)
    assert feeder.realtime is True
    feeder._feed_loop()

    # All 3 frames were attempted and all dropped (queue never had room).
    assert fake_q.put_calls == 3
    out = capsys.readouterr().out
    assert "dropping frame" in out


def test_offline_mode_blocks_instead_of_dropping_on_transient_full_queue(
    monkeypatch, tmp_path, capsys
):
    """realtime=False (the pilot harness): a transiently-full queue must not cost a
    keyframe — the feeder retries until there's room, so every frame is eventually
    delivered."""
    app = load_app_module(monkeypatch)
    _install_fake_video(app, monkeypatch, n_frames=3)
    fake_q = _FullThenOkQueue(full_for=2)
    monkeypatch.setattr(app, "frame_queue", fake_q)

    feeder = app.VideoFeeder(
        _write_dummy_video(tmp_path), fast=True, target_fps=2.0, realtime=False
    )
    feeder._feed_loop()

    # No frame was dropped: all 3 frames made it into the queue despite the first two
    # put() attempts finding it full.
    assert len(fake_q.items) == 3
    assert fake_q.put_calls > 3  # proves it retried rather than succeeding on try #1
    out = capsys.readouterr().out
    assert "dropping frame" not in out


def test_offline_mode_logs_once_when_a_block_exceeds_five_seconds(monkeypatch, tmp_path, capsys):
    """The block-not-drop retry logs (once) if waiting for room exceeds ~5s — visibility
    into a slow SLAM loop, not treated as a failure."""
    app = load_app_module(monkeypatch)
    _install_fake_video(app, monkeypatch, n_frames=1)
    fake_q = _FullThenOkQueue(full_for=1)
    monkeypatch.setattr(app, "frame_queue", fake_q)

    # time.time() call order in _feed_loop: t0, the frame's `data["timestamp"]`,
    # wait_start, then one `waited` check for the one failed put() this fake queue
    # produces (full_for=1). Feed exactly 6s for that last call to cross the ~5s
    # logging threshold once.
    ticks = iter([100.0, 100.0, 100.0, 106.0])
    monkeypatch.setattr(app.time, "time", lambda: next(ticks, 106.0))

    feeder = app.VideoFeeder(
        _write_dummy_video(tmp_path), fast=True, target_fps=2.0, realtime=False
    )
    feeder._feed_loop()

    out = capsys.readouterr().out
    assert out.count("blocking to feed frame") == 1
    assert "dropping frame" not in out


def test_offline_mode_stop_interrupts_a_persistently_full_queue(monkeypatch, tmp_path):
    """A block in offline mode must still be interruptible: stop() (checked every ~1s
    of retrying) unblocks the feed loop within a bounded time instead of hanging
    forever, so it can never wedge finalize/teardown."""
    app = load_app_module(monkeypatch)
    _install_fake_video(app, monkeypatch, n_frames=3)

    class _AlwaysFullSlow(_AlwaysFullQueue):
        def put(self, item, timeout=None):
            time.sleep(0.01)  # loosely mimic a real Queue's bounded blocking wait
            return super().put(item, timeout=timeout)

    fake_q = _AlwaysFullSlow()
    monkeypatch.setattr(app, "frame_queue", fake_q)

    feeder = app.VideoFeeder(
        _write_dummy_video(tmp_path), fast=True, target_fps=2.0, realtime=False
    )

    t = threading.Thread(target=feeder._feed_loop, daemon=True)
    t.start()
    time.sleep(0.1)
    feeder.stop()  # sets _stop_event; the retry loop notices within its next attempt
    t.join(timeout=3.0)

    assert not t.is_alive(), "offline block did not unblock on stop() — deadlock risk"
    # Nothing was ever fed (queue never had room) and nothing was dropped either.
    assert fake_q.put_calls > 0
