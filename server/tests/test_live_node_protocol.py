"""Hermetic tests for the live-node wire protocol + client (Task #5).

No GPU, no Modal, no real network: the mock-server integration test runs a
thread-based WebSocket server on 127.0.0.1 speaking the real protocol module
(the SAME Assembler/encode path the Modal container runs). Covers: message
encode/decode round-trips, fragmentation/reassembly under the 2 MiB cap,
chunker consistency with the probe driver, client timing math, and a full
client-vs-fake-GPU run producing the results-file schema.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from server.oreos.recordings.live_node import client as cl
from server.oreos.recordings.live_node import protocol as proto
from server.oreos.recordings.live_probe import driver as drv


# -- wire protocol: encode/decode ---------------------------------------------


class TestEncodeDecode:
    def test_round_trip_header_and_blobs(self):
        header = {"type": proto.T_CHUNK, "chunk_id": 3, "names": ["100.000001.jpg", "å∂.jpg"]}
        blobs = [b"\xff\xd8jpeg-one", b"", b"\x00" * 1000]
        out_header, out_blobs = proto.decode_message(proto.encode_message(header, blobs))
        assert out_header["type"] == proto.T_CHUNK
        assert out_header["chunk_id"] == 3
        assert out_header["names"] == header["names"]
        assert out_blobs == blobs

    def test_no_blobs(self):
        header, blobs = proto.decode_message(proto.encode_message({"type": proto.T_RESET}))
        assert header["type"] == proto.T_RESET
        assert blobs == []

    def test_truncated_and_garbage_raise(self):
        msg = proto.encode_message({"type": proto.T_RESULT, "chunk_id": 0}, [b"abcdef"])
        with pytest.raises(proto.ProtocolError):
            proto.decode_message(msg[:-3])  # truncated blob
        with pytest.raises(proto.ProtocolError):
            proto.decode_message(b"\x00\x00\x00\x05notjson-and-more")
        with pytest.raises(proto.ProtocolError):
            proto.decode_message(b"\x00\x01")  # shorter than length prefix
        with pytest.raises(proto.ProtocolError):
            proto.decode_message(msg + b"trailing")

    def test_header_must_be_object(self):
        bad = b"\x00\x00\x00\x02[]"
        with pytest.raises(proto.ProtocolError):
            proto.decode_message(bad)


# -- fragmentation under the Modal 2 MiB message cap ---------------------------


class TestFragmentation:
    def test_small_message_passes_through_untouched(self):
        msg = proto.encode_message({"type": proto.T_HELLO}, [b"x" * 100])
        frames = proto.iter_wire_frames(msg, max_frame=10_000)
        assert frames == [msg]

    def test_oversized_message_splits_and_reassembles_exactly(self):
        blobs = [bytes([i % 256]) * 40_000 for i in range(8)]  # ~320 KB
        msg = proto.encode_message({"type": proto.T_CHUNK, "chunk_id": 7, "names": ["a"] * 8}, blobs)
        frames = proto.iter_wire_frames(msg, max_frame=50_000)
        assert len(frames) > 1
        assert all(len(f) <= 50_000 for f in frames)
        asm = proto.Assembler()
        results = [asm.feed(f) for f in frames]
        assert all(r is None for r in results[:-1])
        header, out_blobs = results[-1]
        assert header["chunk_id"] == 7
        assert out_blobs == blobs

    def test_assembler_passthrough_between_part_sequences(self):
        asm = proto.Assembler()
        big = proto.encode_message({"type": proto.T_CHUNK, "chunk_id": 0}, [b"z" * 30_000])
        for f in proto.iter_wire_frames(big, max_frame=9_000):
            last = asm.feed(f)
        assert last is not None and last[0]["chunk_id"] == 0
        small = proto.encode_message({"type": proto.T_RESULT, "chunk_id": 0})
        header, _ = asm.feed(small)
        assert header["type"] == proto.T_RESULT

    def test_out_of_order_part_raises(self):
        big = proto.encode_message({"type": proto.T_CHUNK, "chunk_id": 0}, [b"z" * 30_000])
        frames = proto.iter_wire_frames(big, max_frame=9_000)
        asm = proto.Assembler()
        asm.feed(frames[0])
        with pytest.raises(proto.ProtocolError):
            asm.feed(frames[2])  # skipped seq 1

    def test_interleaved_complete_message_raises(self):
        big = proto.encode_message({"type": proto.T_CHUNK, "chunk_id": 0}, [b"z" * 30_000])
        frames = proto.iter_wire_frames(big, max_frame=9_000)
        asm = proto.Assembler()
        asm.feed(frames[0])
        with pytest.raises(proto.ProtocolError):
            asm.feed(proto.encode_message({"type": proto.T_RESET}))


# -- Assembler accumulation caps (2026-07-24 audit, finding L8) ----------------
#
# A peer that sends `part` frames and never sets `last` used to grow the
# assembler's buffer without bound — an OOM of the single GPU container from
# anyone who can open the socket.


class TestAssemblerCaps:
    def part(self, seq: int, payload: bytes, last: bool = False) -> bytes:
        return proto.encode_message({"type": proto.T_PART, "seq": seq, "last": last}, [payload])

    def test_defaults_are_generous_enough_for_real_traffic(self):
        # A 17-frame go2 chunk is ~1.6 MB — one frame, no fragmentation at all.
        assert proto.MAX_ASSEMBLED_BYTES >= 100 * proto.MAX_WS_MESSAGE_BYTES
        assert proto.MAX_ASSEMBLED_PARTS >= 256

    def test_byte_cap_fails_the_assembly(self):
        asm = proto.Assembler(max_bytes=1000, max_parts=1000)
        assert asm.feed(self.part(0, b"a" * 600)) is None
        with pytest.raises(proto.ProtocolError, match="exceeds 1000 bytes"):
            asm.feed(self.part(1, b"a" * 600))

    def test_part_count_cap_fails_the_assembly(self):
        asm = proto.Assembler(max_bytes=10**9, max_parts=3)
        for seq in range(3):
            assert asm.feed(self.part(seq, b"a")) is None
        with pytest.raises(proto.ProtocolError, match="exceeds 3 fragments"):
            asm.feed(self.part(3, b"a"))

    def test_a_capped_assembler_still_accepts_messages_under_the_cap(self):
        asm = proto.Assembler(max_bytes=1000, max_parts=10)
        payload = proto.encode_message({"type": proto.T_CHUNK, "chunk_id": 7}, [b"z" * 300])
        frames = proto.iter_wire_frames(payload, max_frame=_PART_FRAME)
        out = None
        for f in frames:
            out = asm.feed(f)
        assert out is not None and out[0]["chunk_id"] == 7

    def test_failed_assembly_drops_the_buffer(self):
        """A blown cap must not leave the partial message wired into the next one."""
        asm = proto.Assembler(max_bytes=1000, max_parts=1000)
        asm.feed(self.part(0, b"a" * 900))
        with pytest.raises(proto.ProtocolError):
            asm.feed(self.part(1, b"a" * 900))
        # the assembler is usable again, starting from seq 0
        assert asm.feed(proto.encode_message({"type": proto.T_RESET}))[0]["type"] == proto.T_RESET


_PART_FRAME = 400  # small enough to force fragmentation in the test above


# -- chunker: the client's plan IS the probe driver's --------------------------


class TestChunkerConsistencyWithProbe:
    def names(self, n):
        return [f"{100.0 + i * 0.14:.6f}.jpg" for i in range(n)]

    def test_client_plan_matches_probe_chunker(self):
        frames = self.names(428)
        assert cl.build_chunk_plan(frames, chunk_size=16) == drv.chunk_frames(frames, 16)

    def test_go2_short_shape(self):
        # The real load: 428 frames, 16+1 chunks -> 27 chunks, tail of 12.
        chunks = cl.build_chunk_plan(self.names(428), chunk_size=16)
        assert len(chunks) == 27
        assert [len(c) for c in chunks[:-1]] == [17] * 26
        assert len(chunks[-1]) == 12
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt[0] == prev[-1], "1-frame overlap must seed the next chunk"

    def test_client_replay_loop_uses_same_boundaries(self, tmp_path):
        # End-to-end check (via the mock server below) is in TestMockServerRun;
        # here: the pure boundary math for a non-trivial tail.
        chunks = cl.build_chunk_plan(self.names(10), chunk_size=4)
        assert [len(c) for c in chunks] == [5, 5, 2]


# -- client timing math --------------------------------------------------------


def make_meta(chunk_id=0, ready=1000.0, send=1000.1, sent_done=1000.5):
    return {
        "chunk_id": chunk_id,
        "n_frames": 17,
        "first_frame_ts": 50.0,
        "last_frame_ts": 52.24,
        "chunk_ready_wall": ready,
        "client_send_ts": send,
        "client_sent_done_ts": sent_done,
        "chunk_bytes": 12345,
    }


def make_result(chunk_id=0, gpu_ms=1200.0, **kw):
    base = {
        "type": proto.T_RESULT,
        "chunk_id": chunk_id,
        "n_frames": 17,
        "gpu_ms": gpu_ms,
        "extract_ms": 30.0,
        "n_poses": 17,
        "poses_tum_lines": ["50.000000 0 0 0 0 0 0 1", "52.240000 1 0 0 0 0 0 1"],
        "cloud_summary": {"n_points": 5000, "bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1]},
        "server_recv_ts": 2000.0,
        "server_arrive_ts": 2000.0,
        "server_start_ts": 2000.01,
        "server_done_ts": 2001.25,
        "server_dispatch_overhead_s": 0.01,
        "server_idle_before_s": 0.9,
        "server_queue_wait_s": 0.01,
        "arrived_before_prev_done": False,
        "out_of_order": False,
        "n_submaps_total": 1,
        "loop_closures_total": 0,
    }
    base.update(kw)
    return base


class TestRowMath:
    def test_round_trip_and_lag(self):
        # replay_t0 = 998.0, first src ts 50, last frame ts 52.24 -> that frame
        # "happened" at wall 1000.24; result received at 1002.0 -> lag 1.76 s.
        row = cl.build_row(make_meta(), make_result(), recv_ts=1002.0, replay_t0=998.0, first_src_ts=50.0)
        assert row["round_trip_s"] == pytest.approx(1002.0 - 1000.1)
        assert row["lag_behind_clock_s"] == pytest.approx(1.76)
        assert row["lag_behind_clock_s"] == pytest.approx(
            drv.lag_behind_clock(1002.0, 998.0, 50.0, 52.24)
        )
        assert row["dispatch_delay_s"] == pytest.approx(0.1)
        assert row["send_duration_s"] == pytest.approx(0.4)
        assert row["gpu_s"] == pytest.approx(1.2)
        assert row["extract_s"] == pytest.approx(0.03)

    def test_probe_schema_fields_present(self):
        row = cl.build_row(make_meta(), make_result(), 1002.0, 998.0, 50.0)
        probe_row_keys = {
            "chunk_id", "n_frames", "first_frame_ts", "last_frame_ts",
            "chunk_ready_offset_s", "client_send_ts", "client_recv_ts",
            "dispatch_delay_s", "round_trip_s", "gpu_s", "n_poses",
            "server_recv_ts", "server_done_ts", "lag_behind_clock_s",
            "out_of_order", "n_submaps_total", "loop_closures_total",
        }
        assert probe_row_keys <= set(row)
        # live-node additions
        for key in ("server_dispatch_overhead_s", "server_idle_before_s",
                    "send_duration_s", "cloud_summary", "poses_tum_first"):
            assert key in row

    def test_pose_evidence_compacted(self):
        row = cl.build_row(make_meta(), make_result(), 1002.0, 998.0, 50.0)
        assert row["poses_tum_first"].startswith("50.000000")
        assert row["poses_tum_last"].startswith("52.240000")

    def test_summary_extension(self):
        rows = []
        for i in range(6):
            meta = make_meta(chunk_id=i, ready=1000.0 + 2.24 * i, send=1000.1 + 2.24 * i,
                             sent_done=1000.9 + 2.24 * i)
            meta["first_frame_ts"] = 50.0 + 2.24 * i
            meta["last_frame_ts"] = 52.24 + 2.24 * i
            res = make_result(chunk_id=i, server_dispatch_overhead_s=0.005 + 0.001 * i)
            rows.append(cl.build_row(meta, res, 1002.0 + 2.24 * i, 998.0, 50.0))
        summary = cl.extend_summary(drv.compute_summary(rows, 7.14), rows)
        assert summary["transport"] == "websocket"
        assert summary["server_dispatch_overhead_s"]["p50"] < 0.02
        assert "architecture_note" in summary
        # the probe's summary keys survive untouched
        for key in ("n_chunks", "round_trip_s", "gpu_s", "lag_bounded",
                    "lag_slope_s_per_chunk", "verdict_note"):
            assert key in summary

    def test_reencode_passthrough_when_disabled(self):
        blob = b"\xff\xd8not-really-a-jpeg"
        out, did = cl.reencode_jpeg(blob, 0)
        assert out is blob and did is False

    def test_reencode_never_grows(self):
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)  # noise: q30 shrinks
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        blob = buf.tobytes()
        out, _ = cl.reencode_jpeg(blob, 30)
        assert len(out) <= len(blob)
        out_high, did_high = cl.reencode_jpeg(out, 95)  # would grow -> must keep original
        assert did_high is False and out_high == out

    def test_http_to_ws(self):
        assert cl.http_to_ws("https://x--y.modal.run") == "wss://x--y.modal.run/ws"
        assert cl.http_to_ws("https://x--y.modal.run/ws") == "wss://x--y.modal.run/ws"
        assert cl.http_to_ws("http://127.0.0.1:8123") == "ws://127.0.0.1:8123/ws"
        assert cl.http_to_ws("ws://127.0.0.1:8123/ws") == "ws://127.0.0.1:8123/ws"


# -- mock-server integration ---------------------------------------------------
#
# Only THIS section needs the `websockets` package. It used to be guarded by a
# module-scope `pytest.importorskip("websockets")`, which silently skipped the
# whole file — including every pure test above (encode/decode, fragmentation,
# chunker consistency, row math), none of which touch a socket. The guard is
# scoped to the integration class now, so the wire-format tests always run.
# `websockets` is a real runtime dependency of server/oreos/recordings/live_node/client.py;
# it is declared in requirements.txt and installed by CI.

try:  # cheap capability probe — the actual import happens in FakeLiveNode.__init__
    from websockets.sync.server import serve as _serve  # noqa: F401

    _HAVE_WEBSOCKETS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_WEBSOCKETS = False

requires_websockets = pytest.mark.skipif(
    not _HAVE_WEBSOCKETS,
    reason="websockets not installed — wire-protocol tests above still run",
)


class FakeLiveNode:
    """Thread-based WebSocket server speaking the real protocol module.

    Mirrors the Modal worker's contract: hello on accept, FIFO chunk processing
    with a fake GPU sleep, result per chunk with the server-side timing split,
    reset_ok / bye handling. Uses proto.Assembler — the same reassembly code
    path the real container runs.
    """

    def __init__(self, gpu_sleep_s: float = 0.02, kill_after_results: int | None = None):
        from websockets.sync.server import serve

        self.gpu_sleep_s = gpu_sleep_s
        self.kill_after_results = kill_after_results  # per connection; None = never
        self.reset_calls = 0
        self.chunks_seen: list[dict] = []
        self.cache: dict = {}  # chunk_id -> result, survives reconnects (like the node)
        self._server = serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def _handler(self, ws) -> None:
        # Mirrors the real node's split: the receiver stamps arrive_ts the
        # moment a message is off the socket while a worker thread runs the
        # fake GPU FIFO — so queued-chunk attribution behaves like production.
        import queue as _queue

        asm = proto.Assembler()
        q: _queue.Queue = _queue.Queue()
        _STOP = object()

        def worker() -> None:
            prev_done = None
            next_chunk_id = 0
            results_sent = 0
            while True:
                item = q.get()
                if item is _STOP:
                    return
                header, blobs, arrive_ts = item
                kind = header.get("type")
                if kind == proto.T_RESET:
                    self.reset_calls += 1
                    next_chunk_id = 0
                    prev_done = None
                    self.cache.clear()
                    ws.send(proto.encode_message({"type": proto.T_RESET_OK, "ts": time.time()}))
                elif kind == proto.T_END:
                    ws.send(proto.encode_message({"type": proto.T_BYE, "server_ts": time.time()}))
                    return
                elif kind == proto.T_CHUNK and header["chunk_id"] in self.cache:
                    # resume path: duplicate chunk served from cache, no GPU
                    cached = dict(self.cache[header["chunk_id"]])
                    cached["served_from_cache"] = True
                    ws.send(proto.encode_message(cached))
                elif kind == proto.T_CHUNK:
                    start = time.time()
                    time.sleep(self.gpu_sleep_s)
                    done = time.time()
                    self.chunks_seen.append(
                        {"chunk_id": header["chunk_id"], "names": header["names"],
                         "blob_lens": [len(b) for b in blobs]}
                    )
                    result = {
                        "type": proto.T_RESULT,
                        "chunk_id": header["chunk_id"],
                        "n_frames": len(blobs),
                        "n_poses": len(blobs),
                        "gpu_ms": self.gpu_sleep_s * 1000.0,
                        "extract_ms": 1.0,
                        "poses_tum_lines": [f"{n.rsplit('.', 1)[0]} 0 0 0 0 0 0 1"
                                            for n in header["names"]],
                        "cloud_summary": {"n_points": 100 * len(blobs)},
                        "server_recv_ts": arrive_ts,
                        "server_arrive_ts": arrive_ts,
                        "server_start_ts": start,
                        "server_done_ts": done,
                        "server_dispatch_overhead_s": round(
                            start - (arrive_ts if prev_done is None else max(arrive_ts, prev_done)),
                            4,
                        ),
                        "server_idle_before_s": round(start - prev_done, 4) if prev_done else None,
                        "server_queue_wait_s": round(start - arrive_ts, 4),
                        "arrived_before_prev_done": prev_done is not None and arrive_ts <= prev_done,
                        "out_of_order": header["chunk_id"] != next_chunk_id,
                        "n_submaps_total": header["chunk_id"] + 1,
                        "loop_closures_total": 0,
                        "container_boot_s": 0.0,
                    }
                    next_chunk_id = header["chunk_id"] + 1
                    prev_done = done
                    self.cache[header["chunk_id"]] = result
                    ws.send(proto.encode_message(result))
                    results_sent += 1
                    if self.kill_after_results and results_sent >= self.kill_after_results:
                        ws.close()  # drop the connection mid-stream: client must resume
                        return

        ws.send(proto.encode_message({"type": proto.T_HELLO, "server_ts": time.time(),
                                      "container_boot_s": 0.0}))
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        try:
            for raw in ws:
                complete = asm.feed(raw)
                if complete is None:
                    continue
                header, blobs = complete
                q.put((header, blobs, time.time()))
        finally:
            q.put(_STOP)
            worker_thread.join(timeout=5)

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


FRAME_SPACING_S = 0.02  # 50 fps synthetic replay: fast tests, real clock math


def make_session(tmp_path: Path, n_frames: int, t0: float = 100.0,
                 frame_bytes: int = 200) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(n_frames):
        (frames_dir / f"{t0 + i * FRAME_SPACING_S:.6f}.jpg").write_bytes(
            b"\xff\xd8" + bytes([i % 256]) * frame_bytes
        )
    return frames_dir


@requires_websockets
class TestMockServerRun:
    def run_client(self, tmp_path, n_frames=9, chunk_size=3, gpu_sleep=0.02,
                   kill_after_results=None, **kw):
        server = FakeLiveNode(gpu_sleep_s=gpu_sleep, kill_after_results=kill_after_results)
        frames_dir = make_session(tmp_path, n_frames)
        out_path = tmp_path / "out" / "live_node_results.json"
        kw.setdefault("reset_first", True)
        kw.setdefault("reset_via", "ws")  # the mock is WS-only (no HTTP routes)
        client = cl.OreosLiveClient(
            url=server.url, chunk_size=chunk_size, label="test",
            max_chunk_wait_s=10.0, open_timeout_s=5.0, proxy=None, **kw,
        )
        try:
            doc = client.run(frames_dir=frames_dir, out_path=out_path)
        finally:
            server.close()
        return doc, out_path, server

    def test_end_to_end_rows_and_ordering(self, tmp_path):
        doc, _, server = self.run_client(tmp_path)
        rows = doc["chunks"]
        assert [r["chunk_id"] for r in rows] == [0, 1, 2]
        assert [r["n_frames"] for r in rows] == [4, 4, 3]
        assert all(r["n_poses"] == r["n_frames"] for r in rows)
        assert not any(r["out_of_order"] for r in rows)
        # the server saw the SAME chunk plan the probe driver would build
        frame_names = [f"{100.0 + i * FRAME_SPACING_S:.6f}.jpg" for i in range(9)]
        assert [c["names"] for c in server.chunks_seen] == drv.chunk_frames(frame_names, 3)

    def test_reset_sent_once_and_skippable(self, tmp_path):
        _, _, server = self.run_client(tmp_path / "a", reset_first=True)
        assert server.reset_calls == 1
        _, _, server = self.run_client(tmp_path / "b", reset_first=False)
        assert server.reset_calls == 0

    def test_results_file_schema_matches_probe_plus_extras(self, tmp_path):
        _, out_path, _ = self.run_client(tmp_path)
        doc = json.loads(out_path.read_text())
        assert set(doc) == {"meta", "chunks", "summary"}
        assert doc["meta"]["transport"] == "websocket"
        for key in ("label", "frames_dir", "n_frames", "chunk_size", "replay_t0",
                    "jpeg_quality", "bytes_frames_sent"):
            assert key in doc["meta"], f"meta missing {key}"
        probe_summary_keys = {
            "n_chunks", "round_trip_s", "gpu_s", "max_lag_s", "final_lag_s",
            "replay_rate_fps", "verdict_note", "lag_slope_s_per_chunk", "lag_bounded",
        }
        assert probe_summary_keys <= set(doc["summary"])
        assert doc["summary"]["transport"] == "websocket"
        assert doc["summary"]["n_chunks"] == 3

    def test_clock_math_consistency(self, tmp_path):
        doc, _, _ = self.run_client(tmp_path, gpu_sleep=0.05)
        for r in doc["chunks"]:
            assert r["round_trip_s"] >= 0.05 - 1e-6  # covers the fake GPU sleep
            reconstructed = (r["client_recv_ts"] - doc["meta"]["replay_t0"]) - (
                r["last_frame_ts"] - doc["chunks"][0]["first_frame_ts"]
            )
            assert r["lag_behind_clock_s"] == pytest.approx(reconstructed, abs=0.01)
            assert r["client_recv_ts"] >= r["client_send_ts"]
            assert r["dispatch_delay_s"] >= 0
            assert r["send_duration_s"] is not None and r["send_duration_s"] >= 0

    def test_replay_clock_not_blocked_by_slow_server(self, tmp_path):
        # fake GPU (0.15 s) slower than the 3-frame chunk period (0.06 s): the
        # replay clock must not stall, and lag must grow monotonically.
        doc, _, _ = self.run_client(tmp_path, n_frames=13, chunk_size=3, gpu_sleep=0.15)
        rows = doc["chunks"]
        assert len(rows) == 4
        lags = [r["lag_behind_clock_s"] for r in rows]
        assert lags == sorted(lags)
        assert lags[-1] > lags[0] + 0.2
        # queued chunks (arrived while GPU busy) must show ~zero dispatch overhead:
        queued = [r for r in rows if r["arrived_before_prev_done"]]
        assert queued, "with a slow GPU, later chunks must have been queued"
        for r in queued:
            assert r["server_dispatch_overhead_s"] < 0.05

    def test_fragmented_chunks_survive(self, tmp_path):
        # frames big enough that each chunk message exceeds max_frame_bytes ->
        # the client fragments; the fake server's Assembler must reassemble.
        server = FakeLiveNode()
        frames_dir = make_session(tmp_path, 6, frame_bytes=6_000)
        client = cl.OreosLiveClient(
            url=server.url, chunk_size=2, label="frag", max_chunk_wait_s=10.0,
            open_timeout_s=5.0, proxy=None, max_frame_bytes=5_000, reset_via="ws",
        )
        try:
            doc = client.run(frames_dir, tmp_path / "r.json")
        finally:
            server.close()
        assert doc["summary"]["n_chunks"] == 3  # 6 frames, 2+1 chunks with overlap
        for c, seen in zip(doc["chunks"], server.chunks_seen):
            assert c["n_frames"] == len(seen["blob_lens"])
            assert all(b >= 6_000 for b in seen["blob_lens"])

    def test_remote_error_aborts(self, tmp_path):
        class FailingNode(FakeLiveNode):
            def _handler(self, ws):
                asm = proto.Assembler()
                ws.send(proto.encode_message({"type": proto.T_HELLO}))
                for raw in ws:
                    complete = asm.feed(raw)
                    if complete is None:
                        continue
                    header, _ = complete
                    if header.get("type") == proto.T_RESET:
                        ws.send(proto.encode_message({"type": proto.T_RESET_OK}))
                    elif header.get("type") == proto.T_CHUNK:
                        ws.send(proto.encode_message({
                            "type": proto.T_RESULT, "chunk_id": header["chunk_id"],
                            "error": "CUDA OOM", "traceback": "fake",
                        }))
                        return

        server = FailingNode()
        frames_dir = make_session(tmp_path, 5)
        client = cl.OreosLiveClient(url=server.url, chunk_size=3, max_chunk_wait_s=10.0,
                                    open_timeout_s=5.0, proxy=None, reset_via="ws")
        try:
            with pytest.raises(cl.LiveNodeError, match="CUDA OOM"):
                client.run(frames_dir, tmp_path / "r.json")
        finally:
            server.close()

    def test_reconnect_resume_completes_all_chunks(self, tmp_path):
        # The server drops the connection after every 2 results (the pathology
        # this network exhibits). The client must reconnect WITHOUT resetting,
        # resend unacked chunks, and still deliver every row exactly once.
        doc, _, server = self.run_client(
            tmp_path, n_frames=13, chunk_size=3, kill_after_results=2
        )
        rows = doc["chunks"]
        assert [r["chunk_id"] for r in rows] == [0, 1, 2, 3]
        assert doc["meta"]["n_reconnects"] >= 1
        assert server.reset_calls == 1  # reconnects must NOT reset SLAM state
        # any chunk resent after a drop must have been served from cache or
        # processed exactly once — chunks_seen holds only real GPU processings:
        assert sorted({c["chunk_id"] for c in server.chunks_seen}) == [0, 1, 2, 3]
        assert len(server.chunks_seen) == 4, "duplicates must be served from cache, not re-run"

    def test_busy_node_aborts_cleanly(self, tmp_path):
        class BusyNode(FakeLiveNode):
            def _handler(self, ws):
                ws.send(proto.encode_message({"type": proto.T_BUSY}))
                ws.close(1013)

        server = BusyNode()
        client = cl.OreosLiveClient(url=server.url, open_timeout_s=5.0, proxy=None)
        try:
            with pytest.raises(cl.LiveNodeError, match="busy"):
                client.connect()
        finally:
            server.close()
