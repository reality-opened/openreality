"""Live-node lifecycle tests — the 2026-07-24 audit findings, in CI.

The live node shipped with ZERO tests; every defect the audit found (L1-L8) was
in code no test ever ran. ``session.py`` (pure state) and ``asgi_app.py`` (the
connection lifecycle) are modal-free precisely so these can run GPU-free and
Modal-free, driving the REAL ASGI app through fake ASGI channels against a fake
node — the same code path the container serves.

Coverage map:
  L1 stall watchdog        TestStallWatchdog, TestSessionStall
  L2 results-cache LRU     TestResultsCache, TestSessionResume
  L3 queue backpressure    TestQueueBackpressure
  L4 receiver awaited      TestReceiverShutdown
  L5 snapshot-only health  TestSnapshot, TestHealthEndpoint
  L6 session bleed guard   TestBleedGuard, TestSessionBleedOverWire
  L7 chunk-id advance      TestChunkOrdering
  (L8 Assembler caps live in tests/test_live_node_protocol.py)
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from server.oreos.recordings.live_node import asgi_app, protocol as proto, session as sess


# -- pure state ----------------------------------------------------------------


class TestResultsCache:
    def test_lru_eviction_keeps_the_recent_tail(self):
        c = sess.ResultsCache(maxsize=3)
        for i in range(5):
            c.put(i, {"chunk_id": i})
        assert c.keys() == [2, 3, 4]
        assert c.evictions == 2
        assert c.get(0) is None
        assert c.get(4) == {"chunk_id": 4}

    def test_a_hit_marks_the_entry_recent(self):
        c = sess.ResultsCache(maxsize=3)
        for i in range(3):
            c.put(i, {"chunk_id": i})
        assert c.get(0) is not None  # 0 becomes the most recent
        c.put(3, {"chunk_id": 3})
        assert 0 in c and 1 not in c  # 1 was the LRU, not 0

    def test_clear_and_len(self):
        c = sess.ResultsCache(maxsize=2)
        c.put(1, {})
        assert len(c) == 1
        c.clear()
        assert len(c) == 0 and c.get(1) is None

    def test_maxsize_must_be_positive(self):
        with pytest.raises(ValueError):
            sess.ResultsCache(maxsize=0)

    def test_env_knob(self, monkeypatch):
        monkeypatch.setenv(sess.ENV_RESULTS_CACHE_MAX, "7")
        assert sess.ResultsCache().maxsize == 7
        monkeypatch.setenv(sess.ENV_RESULTS_CACHE_MAX, "not-a-number")
        assert sess.ResultsCache().maxsize == sess.DEFAULT_RESULTS_CACHE_MAX
        monkeypatch.setenv(sess.ENV_RESULTS_CACHE_MAX, "0")  # below minimum
        assert sess.ResultsCache().maxsize == sess.DEFAULT_RESULTS_CACHE_MAX

    def test_state_default_cap_is_bounded(self):
        """L2: the old cache was a plain dict — unbounded by construction."""
        st = sess.LiveSessionState(results_cache_max_=4)
        for i in range(100):
            st.remember_result(i, {"chunk_id": i, "poses_tum_lines": ["x"] * 1000})
        assert len(st.results) == 4
        assert st.snapshot()["results_cached"] == 4


class TestChunkOrdering:
    def test_in_order_stream_is_never_flagged(self):
        st = sess.LiveSessionState()
        for i in range(4):
            assert st.begin_chunk(i) is False
            st.finish_chunk(i, ok=True, n_submaps=i + 1, n_loops=0)
        assert st.next_chunk_id == 4 and st.chunks_done == 4

    def test_failed_chunk_still_advances(self):
        """L7: one error used to mark every later chunk out_of_order forever."""
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=1, n_loops=0)
        assert st.begin_chunk(1) is False
        st.finish_chunk(1, ok=False, error="RuntimeError: boom")
        assert st.next_chunk_id == 2
        assert st.begin_chunk(2) is False  # would have been True before the fix
        st.finish_chunk(2, ok=True, n_submaps=2, n_loops=0)
        assert st.chunks_done == 2 and st.chunks_failed == 1

    def test_genuine_out_of_order_still_detected(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=1, n_loops=0)
        assert st.begin_chunk(5) is True

    def test_reset_zeroes_everything(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=3, n_loops=1)
        st.remember_result(0, {"chunk_id": 0})
        st.reset(ts=123.0)
        snap = st.snapshot()
        assert st.next_chunk_id == 0 and st.chunks_done == 0
        assert snap["n_submaps_total"] == 0 and snap["results_cached"] == 0
        assert snap["has_solver_state"] is False and snap["last_update_ts"] == 123.0


class TestSnapshot:
    def test_worker_publishes_counts_health_reads_them(self):
        """L5: /health must never touch live Solver containers."""
        st = sess.LiveSessionState()
        assert st.snapshot()["n_submaps_total"] == 0
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=2, n_loops=1, ts=10.0)
        snap = st.snapshot()
        assert snap["n_submaps_total"] == 2 and snap["loop_closures_total"] == 1
        assert snap["last_chunk_id"] == 0 and snap["last_chunk_error"] is None
        assert snap["chunks_done"] == 1 and snap["has_solver_state"] is True

    def test_unreadable_counts_keep_the_last_known_value(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=4, n_loops=2)
        st.begin_chunk(1)
        st.finish_chunk(1, ok=False, n_submaps=None, n_loops=None, error="boom")
        snap = st.snapshot()
        assert snap["n_submaps_total"] == 4  # NOT the old flaky -1
        assert snap["last_chunk_error"] == "boom"

    def test_snapshot_is_a_copy(self):
        st = sess.LiveSessionState()
        st.snapshot()["n_submaps_total"] = 99
        assert st.snapshot()["n_submaps_total"] == 0

    def test_snapshot_is_never_half_written(self):
        """Concurrent readers see a complete old snapshot or a complete new one."""
        st = sess.LiveSessionState()
        stop = threading.Event()
        seen: list[dict] = []

        def reader():
            while not stop.is_set():
                s = st.snapshot()
                # chunks_done and next_chunk_id are published together
                seen.append(s)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        for i in range(200):
            st.begin_chunk(i)
            st.finish_chunk(i, ok=True, n_submaps=i + 1, n_loops=0)
        stop.set()
        t.join(timeout=5)
        assert seen, "reader never sampled"
        for s in seen:
            assert s["next_chunk_id"] == s["chunks_done"]  # invariant, all-success stream


class TestBleedGuard:
    def test_fresh_node_accepts_chunk_zero(self):
        st = sess.LiveSessionState()
        assert st.first_chunk_refusal(0) is None

    def test_refuses_a_new_stream_onto_stale_state(self):
        """L6: appending a second recording to the first one's map is never right."""
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=1, n_loops=0)
        msg = st.first_chunk_refusal(0)
        assert msg is not None
        assert sess.ERR_SESSION_STATE_PRESENT in msg
        assert "/reset" in msg

    def test_resume_of_a_cached_chunk_is_allowed(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=1, n_loops=0)
        st.remember_result(0, {"chunk_id": 0})
        assert st.first_chunk_refusal(0) is None

    def test_resume_at_a_later_chunk_is_allowed(self):
        st = sess.LiveSessionState()
        for i in range(3):
            st.begin_chunk(i)
            st.finish_chunk(i, ok=True, n_submaps=i + 1, n_loops=0)
        assert st.first_chunk_refusal(3) is None

    def test_after_reset_a_new_stream_is_accepted(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=True, n_submaps=1, n_loops=0)
        st.reset()
        assert st.first_chunk_refusal(0) is None

    def test_failed_first_chunk_still_counts_as_state(self):
        st = sess.LiveSessionState()
        st.begin_chunk(0)
        st.finish_chunk(0, ok=False, error="boom")
        assert st.first_chunk_refusal(0) is not None


class TestStallWatchdog:
    def test_disabled_waits_forever(self):
        wd = sess.StallWatchdog(timeout_s=0, now=0.0)
        assert wd.enabled is False
        assert wd.wait_timeout(1000.0) is None
        assert wd.is_stalled(1e9) is False

    def test_counts_down_from_the_last_activity(self):
        wd = sess.StallWatchdog(timeout_s=120.0, now=0.0)
        assert wd.wait_timeout(0.0) == pytest.approx(120.0)
        assert wd.wait_timeout(100.0) == pytest.approx(20.0)
        assert wd.is_stalled(119.0) is False
        assert wd.is_stalled(121.0) is True

    def test_touch_resets_the_countdown(self):
        wd = sess.StallWatchdog(timeout_s=10.0, now=0.0)
        wd.touch(9.0)
        assert wd.is_stalled(15.0) is False
        assert wd.wait_timeout(9.0) == pytest.approx(10.0)
        assert wd.touches == 1

    def test_wait_timeout_never_zero_or_negative(self):
        wd = sess.StallWatchdog(timeout_s=10.0, now=0.0)
        assert wd.wait_timeout(1000.0) == pytest.approx(sess.StallWatchdog.MIN_WAIT_S)

    def test_env_knob(self, monkeypatch):
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "5.5")
        assert sess.StallWatchdog().timeout_s == pytest.approx(5.5)
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "off")
        assert sess.StallWatchdog().timeout_s == pytest.approx(sess.DEFAULT_IDLE_TIMEOUT_S)
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "-1")  # explicit disable
        assert sess.StallWatchdog().enabled is False

    def test_report_shape(self):
        wd = sess.StallWatchdog(timeout_s=2.0, now=0.0)
        r = wd.report(3.0)
        assert r == {"idle_s": 3.0, "timeout_s": 2.0, "touches": 0}


class TestQueueKnob:
    def test_default_is_bounded(self):
        assert sess.uplink_queue_max() == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(sess.ENV_UPLINK_QUEUE_MAX, "5")
        assert sess.uplink_queue_max() == 5
        monkeypatch.setenv(sess.ENV_UPLINK_QUEUE_MAX, "0")
        assert sess.uplink_queue_max() == sess.DEFAULT_UPLINK_QUEUE_MAX


# -- ASGI lifecycle (fake node, fake channels) ---------------------------------


class FakeNode:
    """The node interface asgi_app drives, with no GPU and no Modal."""

    def __init__(self, gpu_delay_s: float = 0.0, fail_chunks: set | None = None):
        self.state = sess.LiveSessionState()
        self.session_lock = threading.Lock()
        self.boot_s = 1.0
        self.gpu_delay_s = gpu_delay_s
        self.fail_chunks = fail_chunks or set()
        self.processed: list[int] = []
        self.resets = 0

    def status(self) -> dict:
        return {"ok": True, "session_active": self.session_lock.locked(),
                "core_source": self.core_source(), **self.state.snapshot()}

    def core_source(self) -> dict:
        return {"mode": "tag", "tag": "v0.0.0-test"}

    def reset_sync(self) -> dict:
        self.resets += 1
        self.state.reset()
        return {"type": proto.T_RESET_OK, "ts": 0.0}

    def process_chunk_sync(self, chunk_id: int, names: list, blobs: list) -> dict:
        import time as _t

        out_of_order = self.state.begin_chunk(chunk_id)
        self.processed.append(chunk_id)
        if self.gpu_delay_s:
            _t.sleep(self.gpu_delay_s)
        failed = chunk_id in self.fail_chunks
        result = {"type": proto.T_RESULT, "chunk_id": chunk_id, "n_frames": len(blobs),
                  "out_of_order": out_of_order}
        if failed:
            result["error"] = "RuntimeError: synthetic"
        self.state.finish_chunk(chunk_id, ok=not failed,
                                n_submaps=None if failed else len(self.processed),
                                n_loops=0, error=result.get("error"))
        return result


class WSChannels:
    """Fake ASGI websocket channels: a scripted inbound queue + a sent-frame log."""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed_code = None
        self.consumed = 0  # messages handed to the app (backpressure probe)

    async def receive(self) -> dict:
        msg = await self.inbox.get()
        self.consumed += 1
        return msg

    async def send(self, message: dict) -> None:
        self.sent.append(message)
        if message["type"] == "websocket.close":
            self.closed_code = message.get("code")

    # -- scripting helpers
    def connect(self):
        self.inbox.put_nowait({"type": "websocket.connect"})

    def message(self, header: dict, blobs=()):
        for frame in proto.iter_wire_frames(proto.encode_message(header, blobs)):
            self.inbox.put_nowait({"type": "websocket.bytes", "bytes": frame})

    def raw(self, frame: bytes):
        self.inbox.put_nowait({"type": "websocket.bytes", "bytes": frame})

    def disconnect(self, code: int = 1000):
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": code})

    def headers(self) -> list[dict]:
        """Decoded (and reassembled) headers of everything the app sent."""
        asm = proto.Assembler()  # fresh: headers() may be called repeatedly
        out = []
        for m in self.sent:
            if m["type"] != "websocket.send":
                continue
            complete = asm.feed(m["bytes"])
            if complete is not None:
                out.append(complete[0])
        return out

    def types(self) -> list[str]:
        return [h.get("type") for h in self.headers()]


def ws_scope(path: str = "/ws") -> dict:
    return {"type": "websocket", "path": path}


async def run_ws(node, ch: WSChannels, timeout: float = 5.0) -> None:
    app = asgi_app.build_asgi(node)
    await asyncio.wait_for(app(ws_scope(), ch.receive, ch.send), timeout)


@pytest.mark.asyncio
class TestSessionHappyPath:
    async def test_hello_chunks_and_bye(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        for i in range(3):
            ch.message({"type": proto.T_CHUNK, "chunk_id": i, "names": [f"{i}.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        types = ch.types()
        assert types[0] == proto.T_HELLO
        assert types.count(proto.T_RESULT) == 3
        assert types[-1] == proto.T_BYE
        assert node.processed == [0, 1, 2]
        assert ch.closed_code == 1000
        assert not node.session_lock.locked()

    async def test_timing_split_is_attached(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"],
                    "client_enqueue_ts": 42.0}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        result = [h for h in ch.headers() if h.get("type") == proto.T_RESULT][0]
        for key in ("server_recv_ts", "server_arrive_ts", "server_start_ts",
                    "server_done_ts", "server_dispatch_overhead_s",
                    "server_queue_wait_s", "arrived_before_prev_done"):
            assert key in result, key
        assert result["client_enqueue_ts_echo"] == 42.0

    async def test_reset_message_clears_state(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_RESET})
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        assert node.resets == 1
        assert proto.T_RESET_OK in ch.types()
        assert node.processed == [0, 0]  # accepted again after the reset

    async def test_second_connection_gets_busy(self):
        node = FakeNode()
        node.session_lock.acquire()
        ch = WSChannels()
        ch.connect()
        await run_ws(node, ch)
        assert ch.types() == [proto.T_BUSY]
        assert ch.closed_code == 1013

    async def test_unknown_route_is_rejected(self):
        node = FakeNode()
        app = asgi_app.build_asgi(node)
        ch = WSChannels()
        ch.connect()
        await asyncio.wait_for(app(ws_scope("/nope"), ch.receive, ch.send), 5)
        assert ch.types() == [proto.T_ERROR]
        assert ch.closed_code == 1008
        assert not node.session_lock.locked()


@pytest.mark.asyncio
class TestSessionStall:
    async def test_half_open_connection_releases_the_session(self, monkeypatch):
        """L1: no disconnect ever arrives; the worker must not hold the lock."""
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "0.15")
        node = FakeNode()
        ch = WSChannels()
        ch.connect()  # ...and then nothing, ever: a half-open TCP connection
        await run_ws(node, ch, timeout=5.0)
        assert ch.closed_code == 1001
        assert not node.session_lock.locked(), "session lock leaked for the full timeout"
        errors = [h for h in ch.headers() if h.get("type") == proto.T_ERROR]
        assert errors and "idle timeout" in errors[0]["error"]

    async def test_traffic_keeps_the_session_alive(self, monkeypatch):
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "0.4")
        node = FakeNode()
        ch = WSChannels()
        ch.connect()

        async def feed():
            for i in range(4):
                await asyncio.sleep(0.1)
                ch.message({"type": proto.T_CHUNK, "chunk_id": i, "names": [f"{i}.jpg"]}, [b"x"])
            await asyncio.sleep(0.05)
            ch.message({"type": proto.T_END})

        feeder = asyncio.create_task(feed())
        await run_ws(node, ch, timeout=5.0)
        await feeder
        assert node.processed == [0, 1, 2, 3]
        assert ch.closed_code == 1000  # bye, not a stall

    async def test_slow_gpu_chunk_is_not_a_stall(self, monkeypatch):
        """The watchdog is touched on completion, so a long chunk is not a stall."""
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "0.3")
        node = FakeNode(gpu_delay_s=0.5)
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch, timeout=5.0)
        assert ch.closed_code == 1000
        assert node.processed == [0]

    async def test_state_survives_a_stall_for_resume(self, monkeypatch):
        monkeypatch.setenv(sess.ENV_IDLE_TIMEOUT, "0.15")
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        await run_ws(node, ch, timeout=5.0)
        assert ch.closed_code == 1001
        assert node.state.snapshot()["chunks_done"] == 1
        assert node.state.cached_result(0) is not None


@pytest.mark.asyncio
class TestSessionResume:
    async def test_resent_chunk_is_served_from_cache(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        assert node.processed == [0], "duplicate chunk must never re-run the GPU"
        results = [h for h in ch.headers() if h.get("type") == proto.T_RESULT]
        assert len(results) == 2 and results[1]["served_from_cache"] is True

    async def test_failed_chunk_is_not_cached(self):
        node = FakeNode(fail_chunks={0})
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        assert node.processed == [0, 0]  # retried, not served from cache


@pytest.mark.asyncio
class TestSessionBleedOverWire:
    async def test_fresh_stream_onto_stale_state_is_refused(self):
        """L6, end to end: refuse instead of appending to the previous map."""
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.disconnect()
        await run_ws(node, ch)
        node.state.results.clear()  # the resume window has expired (LRU/reset)

        ch2 = WSChannels()
        ch2.connect()
        ch2.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch2.message({"type": proto.T_CHUNK, "chunk_id": 1, "names": ["1.jpg"]}, [b"x"])
        await run_ws(node, ch2)
        assert node.processed == [0], "the second session's frames were appended"
        errs = [h for h in ch2.headers() if h.get("type") == proto.T_ERROR]
        assert errs and errs[0]["code"] == sess.ERR_SESSION_STATE_PRESENT
        assert ch2.closed_code == 1008
        assert not node.session_lock.locked()

    async def test_reset_first_then_a_new_stream_is_accepted(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.disconnect()
        await run_ws(node, ch)

        ch2 = WSChannels()
        ch2.connect()
        ch2.message({"type": proto.T_RESET})
        ch2.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch2.message({"type": proto.T_END})
        await run_ws(node, ch2)
        assert node.processed == [0, 0]
        assert ch2.closed_code == 1000

    async def test_resume_after_a_drop_is_accepted(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch.disconnect()
        await run_ws(node, ch)

        ch2 = WSChannels()  # client resends chunk 0 (never saw the result)
        ch2.connect()
        ch2.message({"type": proto.T_CHUNK, "chunk_id": 0, "names": ["0.jpg"]}, [b"x"])
        ch2.message({"type": proto.T_CHUNK, "chunk_id": 1, "names": ["1.jpg"]}, [b"x"])
        ch2.message({"type": proto.T_END})
        await run_ws(node, ch2)
        assert node.processed == [0, 1]
        assert ch2.closed_code == 1000


@pytest.mark.asyncio
class TestQueueBackpressure:
    async def test_receiver_is_throttled_to_the_queue_bound(self, monkeypatch):
        """L3: the queue was unbounded, so a fast peer buffered the whole stream."""
        monkeypatch.setenv(sess.ENV_UPLINK_QUEUE_MAX, "2")
        node = FakeNode(gpu_delay_s=0.15)
        ch = WSChannels()
        ch.connect()
        for i in range(8):
            ch.message({"type": proto.T_CHUNK, "chunk_id": i, "names": [f"{i}.jpg"]}, [b"x"])
        ch.message({"type": proto.T_END})

        async def sample_during_first_chunk() -> int:
            await asyncio.sleep(0.05)
            return ch.consumed

        probe = asyncio.create_task(sample_during_first_chunk())
        await run_ws(node, ch, timeout=10)
        early = await probe
        # While chunk 0 is on the GPU the receiver may only be 1 (connect) +
        # 1 (in the worker) + maxsize (queued) + 1 (parked in put) messages
        # ahead. Unbounded, it would have swallowed all 10 by now.
        assert early <= 6, f"receiver read {early} messages without backpressure"
        assert node.processed == [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
class TestReceiverShutdown:
    async def test_receiver_is_cancelled_and_awaited(self):
        """L4: the receiver used to survive the session that spawned it."""
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_END})
        before = len(asyncio.all_tasks())
        await run_ws(node, ch)
        await asyncio.sleep(0)  # a leaked task would still be pending here
        assert len(asyncio.all_tasks()) <= before
        assert not node.session_lock.locked()

    async def test_protocol_error_closes_the_session(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.raw(b"\x00\x00\x00\x05notjson-and-more")
        await run_ws(node, ch)
        errs = [h for h in ch.headers() if h.get("type") == proto.T_ERROR]
        assert errs and "protocol" in errs[0]["error"]
        assert ch.closed_code == 1002
        assert not node.session_lock.locked()

    async def test_bad_chunk_header_closes_the_session(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": proto.T_CHUNK, "names": ["0.jpg"]}, [b"x"])  # no chunk_id
        await run_ws(node, ch)
        assert ch.closed_code == 1002
        assert node.processed == []
        assert not node.session_lock.locked()

    async def test_unknown_message_type_is_reported_not_fatal(self):
        node = FakeNode()
        ch = WSChannels()
        ch.connect()
        ch.message({"type": "nonsense"})
        ch.message({"type": proto.T_END})
        await run_ws(node, ch)
        assert proto.T_ERROR in ch.types()
        assert ch.closed_code == 1000


# -- HTTP side -----------------------------------------------------------------


class HTTPChannels:
    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.inbox.put_nowait({"type": "http.request", "more_body": False})
        self.start = None
        self.body = b""

    async def receive(self):
        return await self.inbox.get()

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.start = message
        else:
            self.body += message.get("body", b"")

    def json(self):
        import json

        return json.loads(self.body)


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_reads_the_snapshot(self):
        node = FakeNode()
        node.state.begin_chunk(0)
        node.state.finish_chunk(0, ok=True, n_submaps=3, n_loops=1)
        app = asgi_app.build_asgi(node)
        ch = HTTPChannels()
        await app({"type": "http", "path": "/health", "method": "GET"}, ch.receive, ch.send)
        assert ch.start["status"] == 200
        body = ch.json()
        assert body["n_submaps_total"] == 3 and body["loop_closures_total"] == 1
        assert body["session_active"] is False

    async def test_health_answers_while_a_session_holds_the_lock(self):
        """L5: /health must not block on — or race — the streaming session."""
        node = FakeNode()
        node.session_lock.acquire()
        try:
            app = asgi_app.build_asgi(node)
            ch = HTTPChannels()
            await asyncio.wait_for(
                app({"type": "http", "path": "/health", "method": "GET"},
                    ch.receive, ch.send),
                timeout=2,
            )
            assert ch.json()["session_active"] is True
        finally:
            node.session_lock.release()

    async def test_reset_409s_while_a_session_is_active(self):
        node = FakeNode()
        node.session_lock.acquire()
        try:
            app = asgi_app.build_asgi(node)
            ch = HTTPChannels()
            await app({"type": "http", "path": "/reset", "method": "POST"},
                      ch.receive, ch.send)
            assert ch.start["status"] == 409
            assert node.resets == 0
        finally:
            node.session_lock.release()

    async def test_reset_runs_when_idle(self):
        node = FakeNode()
        node.state.begin_chunk(0)
        node.state.finish_chunk(0, ok=True, n_submaps=2, n_loops=0)
        app = asgi_app.build_asgi(node)
        ch = HTTPChannels()
        await app({"type": "http", "path": "/reset", "method": "POST"}, ch.receive, ch.send)
        assert ch.start["status"] == 200 and node.resets == 1
        assert node.state.snapshot()["n_submaps_total"] == 0
        assert not node.session_lock.locked()

    async def test_unknown_route_404s(self):
        app = asgi_app.build_asgi(FakeNode())
        ch = HTTPChannels()
        await app({"type": "http", "path": "/nope", "method": "GET"}, ch.receive, ch.send)
        assert ch.start["status"] == 404

    async def test_lifespan_startup_and_shutdown(self):
        app = asgi_app.build_asgi(FakeNode())
        inbox: asyncio.Queue = asyncio.Queue()
        inbox.put_nowait({"type": "lifespan.startup"})
        inbox.put_nowait({"type": "lifespan.shutdown"})
        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(app({"type": "lifespan"}, inbox.get, send), 5)
        assert [m["type"] for m in sent] == [
            "lifespan.startup.complete", "lifespan.shutdown.complete"
        ]
