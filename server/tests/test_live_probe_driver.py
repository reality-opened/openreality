"""Hermetic tests for the live-probe driver's chunking + clock math.

No GPU, no modal, no network: the remote call is mocked with a fake sleeping
function behind ``SerialFunctionTransport`` (same serial-FIFO contract as the
single warm Modal container). Covers: chunk boundaries with overlap, lag
computation correctness, and the results-file schema.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from server.oreos.recordings.live_probe import driver as drv


# -- fixtures -----------------------------------------------------------------

FRAME_SPACING_S = 0.02  # 50 fps synthetic replay: fast tests, real clock math


def make_session(tmp_path: Path, n_frames: int, t0: float = 100.0, spacing: float = FRAME_SPACING_S) -> Path:
    """A miniature session: timestamp-named jpgs like the real recorder writes."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(n_frames):
        (frames_dir / f"{t0 + i * spacing:.6f}.jpg").write_bytes(b"\xff\xd8fakejpeg%d" % i)
    return frames_dir


def fake_gpu(service_time_s: float):
    """Mock of LiveProbe.process_chunk: sleeps like a GPU, returns the real schema."""

    def process(chunk_id: int, names: list, jpegs: list) -> dict:
        recv = time.time()
        time.sleep(service_time_s)
        return {
            "chunk_id": chunk_id,
            "n_frames": len(names),
            "n_poses": len(names),
            "gpu_ms": service_time_s * 1000.0,
            "server_recv_ts": recv,
            "server_done_ts": time.time(),
            "out_of_order": False,
            "n_submaps_total": chunk_id + 1,
            "loop_closures_total": 0,
        }

    return process


# -- chunk boundaries with overlap --------------------------------------------


def names(n):
    return [f"{100.0 + i * 0.1:.6f}.jpg" for i in range(n)]


class TestChunkFrames:
    def test_overlap_invariant_and_sizes(self):
        chunks = drv.chunk_frames(names(10), chunk_size=4)
        # fires at frames 0-4 and 4-8; frame 9 rides out with the overlap frame
        assert [len(c) for c in chunks] == [5, 5, 2]
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt[0] == prev[-1], "next chunk must start with previous chunk's last frame"

    def test_every_frame_appears_exactly_once_as_new(self):
        frames = names(41)
        chunks = drv.chunk_frames(frames, chunk_size=7)
        new_frames = list(chunks[0]) + [f for c in chunks[1:] for f in c[1:]]
        assert new_frames == frames

    def test_exact_boundary_no_phantom_tail_chunk(self):
        # 9 frames, size 4: fires at frame 4 and frame 8 (which is also the last
        # frame) — the lone overlap frame must NOT become a third chunk.
        chunks = drv.chunk_frames(names(9), chunk_size=4)
        assert [len(c) for c in chunks] == [5, 5]

    def test_short_stream_single_chunk(self):
        chunks = drv.chunk_frames(names(3), chunk_size=16)
        assert [len(c) for c in chunks] == [3]

    def test_go2_short_shape(self):
        # The real probe load: 428 frames, 16+1 chunks -> 27 chunks, tail of 12.
        chunks = drv.chunk_frames(names(428), chunk_size=16)
        assert len(chunks) == 27
        assert [len(c) for c in chunks[:-1]] == [17] * 26
        assert len(chunks[-1]) == 12


# -- clock math ---------------------------------------------------------------


class TestClockMath:
    def test_parse_frame_ts(self):
        assert drv.parse_frame_ts("1778055838.694958.jpg") == pytest.approx(1778055838.694958)
        assert drv.parse_frame_ts(Path("/x/y/42.jpg")) == 42.0
        with pytest.raises(ValueError):
            drv.parse_frame_ts("noframe.jpg")

    def test_lag_behind_clock(self):
        # Replay started at wall 1000; chunk's last frame has source ts 55 in a
        # session starting at 50 -> that frame "happened" at wall 1005. A result
        # received at wall 1011.5 is 6.5 s behind the clock.
        lag = drv.lag_behind_clock(
            recv_wall_ts=1011.5, replay_t0=1000.0, first_src_ts=50.0, last_src_ts=55.0
        )
        assert lag == pytest.approx(6.5)

    def test_lag_can_be_reconstructed_from_row_fields(self):
        # lag == (recv - replay_t0) - (last_frame_ts - first_frame_ts): the row
        # stores every term, so the JSON is self-auditing.
        recv, t0, f0, fl = 2000.75, 1990.0, 10.0, 12.5
        assert drv.lag_behind_clock(recv, t0, f0, fl) == pytest.approx((recv - t0) - (fl - f0))

    def test_percentile_linear_interpolation(self):
        assert drv.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
        assert drv.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)  # unsorted in
        assert drv.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
        assert drv.percentile([7], 95) == 7
        assert math.isnan(drv.percentile([], 50))

    def test_linear_slope(self):
        assert drv.linear_slope([0, 1, 2, 3], [1.0, 1.5, 2.0, 2.5]) == pytest.approx(0.5)
        assert drv.linear_slope([0, 1, 2], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
        assert drv.linear_slope([1], [5.0]) == 0.0


# -- summary math: bounded vs growing lag -------------------------------------


def make_row(chunk_id, last_frame_ts, lag, rt=1.0, gpu=0.8, first_frame_ts=None):
    return {
        "chunk_id": chunk_id,
        "n_frames": 17,
        "first_frame_ts": first_frame_ts if first_frame_ts is not None else last_frame_ts - 2.2,
        "last_frame_ts": last_frame_ts,
        "round_trip_s": rt,
        "gpu_s": gpu,
        "lag_behind_clock_s": lag,
    }


class TestSummary:
    def test_bounded_lag(self):
        rows = [make_row(i, 10.0 + 2.2 * i, lag=1.5 + 0.01 * (i % 2)) for i in range(8)]
        s = drv.compute_summary(rows, replay_rate_fps=7.2)
        assert s["lag_bounded"] is True
        assert abs(s["lag_slope_s_per_chunk"]) < 0.05
        assert "BOUNDED" in s["verdict_note"]
        assert s["chunk_period_s"] == pytest.approx(2.2)

    def test_growing_lag(self):
        # GPU 1 s/chunk slower than real time -> lag climbs linearly.
        rows = [make_row(i, 10.0 + 2.2 * i, lag=1.0 + 1.0 * i) for i in range(8)]
        s = drv.compute_summary(rows, replay_rate_fps=7.2)
        assert s["lag_bounded"] is False
        assert s["lag_slope_s_per_chunk"] == pytest.approx(1.0, abs=0.01)
        assert "GROWING" in s["verdict_note"]
        assert s["max_lag_s"] == pytest.approx(8.0)
        assert s["final_lag_s"] == pytest.approx(8.0)

    def test_steady_state_excludes_cold_first_chunk(self):
        # A huge cold first chunk must not flip a bounded steady state to growing.
        rows = [make_row(0, 10.0, lag=60.0)] + [
            make_row(i, 10.0 + 2.2 * i, lag=1.5) for i in range(1, 8)
        ]
        s = drv.compute_summary(rows, replay_rate_fps=7.2)
        assert s["lag_bounded"] is True
        assert s["max_lag_s"] == pytest.approx(60.0)  # still reported honestly

    def test_percentiles_over_rows(self):
        rows = [make_row(i, 10.0 + 2.2 * i, lag=1.0, rt=1.0 + i, gpu=0.5 + i) for i in range(4)]
        s = drv.compute_summary(rows, replay_rate_fps=7.2)
        assert s["round_trip_s"]["p50"] == pytest.approx(2.5)
        assert s["gpu_s"]["p50"] == pytest.approx(2.0)
        assert s["round_trip_s"]["first"] == pytest.approx(1.0)
        assert s["n_chunks"] == 4

    def test_empty_rows(self):
        s = drv.compute_summary([], replay_rate_fps=7.2)
        assert s["n_chunks"] == 0
        assert "no chunks" in s["verdict_note"]


# -- end-to-end replay with a fake sleeping GPU -------------------------------


class TestReplayDriverEndToEnd:
    def run_driver(self, tmp_path, n_frames=9, chunk_size=3, service=0.05, **kw):
        frames_dir = make_session(tmp_path, n_frames)
        out_path = tmp_path / "out" / "live_probe_results.json"
        transport = drv.SerialFunctionTransport(fake_gpu(service))
        driver = ReplayDriverFactory(frames_dir, out_path, transport, chunk_size, **kw)
        try:
            doc = driver.run()
        finally:
            transport.close()
        return doc, out_path, transport

    def test_rows_ordered_and_chunked(self, tmp_path):
        doc, _, _ = self.run_driver(tmp_path)
        rows = doc["chunks"]
        assert [r["chunk_id"] for r in rows] == [0, 1, 2]
        assert [r["n_frames"] for r in rows] == [4, 4, 3]
        assert all(r["n_poses"] == r["n_frames"] for r in rows)

    def test_clock_math_consistency(self, tmp_path):
        doc, _, _ = self.run_driver(tmp_path, service=0.05)
        for r in doc["chunks"]:
            # round trip covers the fake GPU's sleep, with modest scheduling slop
            assert 0.05 <= r["round_trip_s"] < 0.6
            # lag = time from the chunk's last frame on the replay clock to recv:
            # at least the service time, and self-consistent with the row's terms
            assert r["lag_behind_clock_s"] >= 0.05 - 1e-6
            reconstructed = (r["client_recv_ts"] - doc["meta"]["replay_t0"]) - (
                r["last_frame_ts"] - doc["chunks"][0]["first_frame_ts"]
            )
            assert r["lag_behind_clock_s"] == pytest.approx(reconstructed, abs=0.01)
            assert r["client_recv_ts"] >= r["client_send_ts"]
            assert r["dispatch_delay_s"] >= 0

    def test_replay_clock_not_blocked_by_slow_gpu(self, tmp_path):
        # GPU (0.15 s/chunk) slower than the chunk period (3 frames * 0.02 s):
        # replay must still finish on schedule and lag must GROW monotonically.
        doc, _, _ = self.run_driver(tmp_path, n_frames=13, chunk_size=3, service=0.15)
        rows = doc["chunks"]
        assert len(rows) == 4
        lags = [r["lag_behind_clock_s"] for r in rows]
        assert lags == sorted(lags)
        assert lags[-1] > lags[0] + 0.2  # fell ~3 service times behind

    def test_reset_called_unless_skipped(self, tmp_path):
        _, _, transport = self.run_driver(tmp_path / "a", reset_first=True)
        assert transport.reset_calls == 1
        _, _, transport = self.run_driver(tmp_path / "b", reset_first=False)
        assert transport.reset_calls == 0

    def test_results_file_schema(self, tmp_path):
        _, out_path, _ = self.run_driver(tmp_path)
        assert out_path.exists()
        doc = json.loads(out_path.read_text())
        assert set(doc) == {"meta", "chunks", "summary"}

        meta = doc["meta"]
        for key in ("label", "frames_dir", "n_frames", "chunk_size", "replay_t0", "transport"):
            assert key in meta, f"meta missing {key}"

        row_keys = {
            "chunk_id", "n_frames", "first_frame_ts", "last_frame_ts",
            "client_send_ts", "client_recv_ts", "round_trip_s", "gpu_s",
            "server_recv_ts", "server_done_ts", "lag_behind_clock_s", "n_poses",
        }
        for row in doc["chunks"]:
            assert row_keys <= set(row), f"row missing {row_keys - set(row)}"

        summary = doc["summary"]
        for key in (
            "n_chunks", "round_trip_s", "gpu_s", "max_lag_s", "final_lag_s",
            "replay_rate_fps", "verdict_note", "lag_slope_s_per_chunk", "lag_bounded",
        ):
            assert key in summary, f"summary missing {key}"
        for pct in ("p50", "p95"):
            assert pct in summary["round_trip_s"]
            assert pct in summary["gpu_s"]
        assert summary["n_chunks"] == 3
        assert summary["replay_rate_fps"] == pytest.approx(50.0, rel=0.01)
        assert isinstance(summary["verdict_note"], str) and summary["verdict_note"]

    def test_remote_error_aborts(self, tmp_path):
        def failing(chunk_id, names, jpegs):
            return {"chunk_id": chunk_id, "error": "CUDA OOM", "server_recv_ts": 0, "server_done_ts": 0}

        frames_dir = make_session(tmp_path, 5)
        transport = drv.SerialFunctionTransport(failing)
        driver = ReplayDriverFactory(frames_dir, tmp_path / "r.json", transport, chunk_size=3)
        try:
            with pytest.raises(drv.ProbeError, match="CUDA OOM"):
                driver.run()
        finally:
            transport.close()


def ReplayDriverFactory(frames_dir, out_path, transport, chunk_size, **kw):
    kw.setdefault("reset_first", False)
    return drv.ReplayDriver(
        frames_dir=frames_dir,
        out_path=out_path,
        transport=transport,
        chunk_size=chunk_size,
        label="test",
        poll_timeout_s=0.05,
        max_chunk_wait_s=10.0,
        **kw,
    )
