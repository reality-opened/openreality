"""Local replay driver for the Oreos live probe.

Answers the Phase-2 question: if a robot streams keyframes at their recorded rate
(~7.2 fps for go2_short), how far behind the replay clock does a remote GPU fall
per 16-frame chunk?

Mechanics
  * Frames are replayed at their RECORDED rate: filenames are timestamps
    (``1778055838.694958.jpg``); frame *i* "arrives" at wall time
    ``replay_t0 + (ts_i - ts_0)``. The replay clock NEVER blocks on the GPU.
  * Chunking mirrors ``server/oreos/recordings/modal_recon.py``'s submap loop: a buffer
    accumulates frames; at ``chunk_size + 1`` frames the chunk fires and the last
    frame is retained as the 1-frame overlap seeding the next chunk.
  * Transport: ``.spawn`` per chunk from a dispatcher thread (the replay clock is
    never blocked), results collected in order by a poller thread using
    ``FunctionCall.get(timeout=POLL)`` — short polls survive the flaky local
    proxy where one long-lived stream would not. Cost of this choice: a dropped
    poll re-poll adds up to ~one proxy RTT to the measured round trip, and
    ``round_trip_s`` includes argument upload + queue wait on the single serial
    container + result download. ``gpu_s`` (measured server-side) is transport-free.

Per chunk: ``client_send_ts``, ``client_recv_ts``, ``round_trip_s``, ``gpu_s``,
``lag_behind_clock_s`` (completion wall time vs the replay-clock time of the
chunk's LAST frame). Summary answers the steady-state question: does lag grow
linearly (GPU slower than real time) or stay bounded?

Usage (deploy modal_live.py first — see its docstring):

    python server/oreos/recordings/live_probe/driver.py \
        --frames server/sessions/go2_short/frames \
        --out    server/sessions/go2_short/live_probe \
        --label  warm

The cold measurement is just the first run against a scaled-to-zero app
(``--label cold --skip-reset --out-name live_probe_results_cold.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "oreos-live-probe"
CLS_NAME = "LiveProbe"
DEFAULT_CHUNK_SIZE = 16
DEFAULT_POLL_TIMEOUT_S = 2.0
DEFAULT_MAX_CHUNK_WAIT_S = 900.0
RESULTS_BASENAME = "live_probe_results.json"

# Slope (s of lag per chunk) below which steady-state lag counts as bounded:
# generous enough to absorb poll jitter, far below any real falling-behind rate
# (a GPU 10% slower than real time on a ~2.2s chunk period slips ~0.22 s/chunk).
LAG_BOUNDED_SLOPE_S_PER_CHUNK = 0.1


class ProbeError(RuntimeError):
    """Fatal probe failure (remote error or chunk deadline exceeded)."""


# -- pure helpers (unit-tested; no modal, no threads) -------------------------

_TS_RE = re.compile(r"\d+(?:\.\d+)?(?=\.[^.]+$)")


def parse_frame_ts(path: str | Path) -> float:
    """Timestamp encoded in a frame filename (mirrors core's sort_images_by_number)."""
    m = _TS_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"no timestamp in frame name: {path}")
    return float(m.group())


def list_frames(frames_dir: str | Path) -> list[Path]:
    frames = sorted(Path(frames_dir).glob("*.jpg"), key=parse_frame_ts)
    if not frames:
        raise FileNotFoundError(f"no *.jpg frames under {frames_dir}")
    return frames


def chunk_frames(paths: list, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[list]:
    """Split an ordered frame list into submap chunks with a 1-frame overlap.

    Mirrors modal_recon's loop: fire at ``chunk_size + 1`` buffered frames, keep
    the last frame as the seed of the next chunk; whatever remains at the end
    fires as a final (shorter) chunk. Invariant: ``chunks[i+1][0] is chunks[i][-1]``.
    """
    chunks: list[list] = []
    buf: list = []
    for i, p in enumerate(paths):
        buf.append(p)
        if len(buf) == chunk_size + 1 or i == len(paths) - 1:
            # A boundary fire and end-of-stream coinciding emit ONE chunk (the
            # append above guarantees the buffer holds a new frame here).
            chunks.append(buf)
            buf = buf[-1:]
    return chunks


def lag_behind_clock(
    recv_wall_ts: float, replay_t0: float, first_src_ts: float, last_src_ts: float
) -> float:
    """Seconds between a chunk's completion and its last frame on the replay clock.

    The chunk's last frame "happened" at ``replay_t0 + (last_src_ts - first_src_ts)``;
    lag is how long after that moment the poses came back.
    """
    return recv_wall_ts - (replay_t0 + (last_src_ts - first_src_ts))


def percentile(values: list, q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear'), stdlib-only."""
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(vals[int(k)])
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def linear_slope(xs: list, ys: list) -> float:
    """Least-squares slope of ys over xs; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def compute_summary(rows: list, replay_rate_fps: float) -> dict:
    """Aggregate per-chunk rows into the probe's headline numbers."""
    if not rows:
        return {"n_chunks": 0, "verdict_note": "no chunks completed", "replay_rate_fps": replay_rate_fps}

    rts = [r["round_trip_s"] for r in rows]
    gpus = [r["gpu_s"] for r in rows if r.get("gpu_s") is not None]
    lags = [r["lag_behind_clock_s"] for r in rows]

    # Chunk period on the replay clock (spacing of chunk-final frames).
    ends = [r["last_frame_ts"] for r in rows]
    diffs = [b - a for a, b in zip(ends, ends[1:])]
    chunk_period_s = percentile(diffs, 50) if diffs else (ends[0] - rows[0]["first_frame_ts"])

    # Steady state: exclude the first chunk (cold / pipeline fill) when possible.
    steady = rows[1:] if len(rows) >= 4 else rows
    slope = linear_slope([r["chunk_id"] for r in steady], [r["lag_behind_clock_s"] for r in steady])
    bounded = slope <= LAG_BOUNDED_SLOPE_S_PER_CHUNK

    gpu_p50 = percentile(gpus, 50) if gpus else float("nan")
    occupancy = gpu_p50 / chunk_period_s if chunk_period_s else float("nan")
    verdict_note = (
        f"p50 GPU {gpu_p50:.2f}s per {chunk_period_s:.2f}s chunk period "
        f"(occupancy {occupancy:.2f}x realtime); steady-state lag slope "
        f"{slope:+.3f} s/chunk -> "
        + (
            "BOUNDED (GPU keeps up with the replay clock)"
            if bounded
            else "GROWING linearly (GPU/transport slower than real time)"
        )
        + f"; first-chunk round trip {rts[0]:.2f}s, final lag {lags[-1]:.2f}s, max lag {max(lags):.2f}s."
    )

    return {
        "n_chunks": len(rows),
        "replay_rate_fps": round(replay_rate_fps, 3),
        "chunk_period_s": round(chunk_period_s, 3),
        "round_trip_s": {
            "p50": round(percentile(rts, 50), 3),
            "p95": round(percentile(rts, 95), 3),
            "max": round(max(rts), 3),
            "first": round(rts[0], 3),
        },
        "gpu_s": {
            "p50": round(percentile(gpus, 50), 3) if gpus else None,
            "p95": round(percentile(gpus, 95), 3) if gpus else None,
            "max": round(max(gpus), 3) if gpus else None,
            "first": round(gpus[0], 3) if gpus else None,
        },
        "max_lag_s": round(max(lags), 3),
        "final_lag_s": round(lags[-1], 3),
        "first_chunk_round_trip_s": round(rts[0], 3),
        "lag_slope_s_per_chunk": round(slope, 4),
        "lag_bounded": bounded,
        "gpu_occupancy_x_realtime": round(occupancy, 3) if occupancy == occupancy else None,
        "verdict_note": verdict_note,
    }


# -- transports ---------------------------------------------------------------


@dataclass
class _FnHandle:
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: BaseException | None = None


class SerialFunctionTransport:
    """Executes a plain callable serially on ONE worker thread.

    Mirrors the remote contract (single warm container, FIFO inputs) without
    modal — the test seam, and a local smoke path. ``process_fn(chunk_id, names,
    jpegs) -> dict`` may sleep to simulate GPU time.
    """

    note = "in-process serial worker (no network); round_trip_s ~= service time + queue wait"

    def __init__(self, process_fn):
        self._process_fn = process_fn
        self._q: queue.Queue = queue.Queue()
        self.reset_calls = 0
        self._worker = threading.Thread(target=self._loop, name="fn-transport", daemon=True)
        self._worker.start()

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            handle, chunk_id, names, jpegs = item
            try:
                handle.result = self._process_fn(chunk_id, names, jpegs)
            except BaseException as e:  # noqa: BLE001 — surfaced via wait()
                handle.error = e
            handle.done.set()

    def submit(self, chunk_id: int, names: list, jpegs: list) -> _FnHandle:
        handle = _FnHandle()
        self._q.put((handle, chunk_id, names, jpegs))
        return handle

    def wait(self, handle: _FnHandle, timeout: float) -> dict | None:
        if not handle.done.wait(timeout):
            return None
        if handle.error is not None:
            raise ProbeError(f"transport process_fn failed: {handle.error}") from handle.error
        return handle.result

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=5)


class ModalTransport:
    """spawn/poll against the deployed ``oreos-live-probe`` app (lazy modal import)."""

    def __init__(
        self,
        app_name: str = APP_NAME,
        cls_name: str = CLS_NAME,
        poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    ):
        import modal  # lazy: tests never import modal

        probe_cls = modal.Cls.from_name(app_name, cls_name)
        self._probe = probe_cls()
        self.poll_timeout_s = poll_timeout_s
        self.note = (
            f".spawn per chunk + FunctionCall.get(timeout={poll_timeout_s}s) polling via local "
            "proxy (short polls survive proxy drops; one long-lived stream would not). "
            "round_trip_s includes arg upload + FIFO queue wait on the single serial "
            "container + result poll granularity (~one proxy RTT worst case per re-poll); "
            "gpu_s is pure server-side model+solver time."
        )

    def submit(self, chunk_id: int, names: list, jpegs: list):
        return self._probe.process_chunk.spawn(jpegs=jpegs, names=names, chunk_id=chunk_id)

    def wait(self, handle, timeout: float) -> dict | None:
        try:
            return handle.get(timeout=timeout)
        except Exception as e:  # modal.exception.TimeoutError subclasses builtins
            if isinstance(e, TimeoutError) or type(e).__name__ == "TimeoutError":
                return None
            raise

    def reset(self) -> dict:
        return self._probe.reset.remote()

    def close(self) -> None:
        pass


# -- the replay driver --------------------------------------------------------


class ReplayDriver:
    """Replays a session's frames at their recorded rate and probes the GPU per chunk."""

    def __init__(
        self,
        frames_dir: str | Path,
        out_path: str | Path,
        transport,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        label: str = "live",
        reset_first: bool = True,
        poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        max_chunk_wait_s: float = DEFAULT_MAX_CHUNK_WAIT_S,
    ):
        self.frames = list_frames(frames_dir)
        self.frames_dir = str(frames_dir)
        self.out_path = Path(out_path)
        self.transport = transport
        self.chunk_size = chunk_size
        self.label = label
        self.reset_first = reset_first
        self.poll_timeout_s = poll_timeout_s
        self.max_chunk_wait_s = max_chunk_wait_s

        self._ts = [parse_frame_ts(p) for p in self.frames]
        self._chunks_q: queue.Queue = queue.Queue()
        self._pending_q: queue.Queue = queue.Queue()
        self.rows: list[dict] = []
        self._collector_error: BaseException | None = None
        self.replay_t0: float | None = None

    # dispatcher: chunk-ready -> spawn (never on the replay thread)
    def _dispatch_loop(self):
        while True:
            item = self._chunks_q.get()
            if item is None:
                self._pending_q.put(None)
                return
            chunk_id, names, jpegs, meta = item
            meta["client_send_ts"] = time.time()
            handle = self.transport.submit(chunk_id, names, jpegs)
            self._pending_q.put((handle, meta))

    # collector: poll results IN ORDER (server is serial FIFO anyway)
    def _collect_loop(self):
        try:
            while True:
                item = self._pending_q.get()
                if item is None:
                    return
                handle, meta = item
                deadline = time.time() + self.max_chunk_wait_s
                result = None
                while result is None:
                    if time.time() > deadline:
                        raise ProbeError(
                            f"chunk {meta['chunk_id']} no result within {self.max_chunk_wait_s}s"
                        )
                    result = self.transport.wait(handle, self.poll_timeout_s)
                recv_ts = time.time()
                if result.get("error"):
                    raise ProbeError(
                        f"remote solver failed on chunk {meta['chunk_id']}: {result['error']}\n"
                        f"{result.get('traceback', '')}"
                    )
                self.rows.append(self._build_row(meta, result, recv_ts))
        except BaseException as e:  # noqa: BLE001 — re-raised on the main thread
            self._collector_error = e

    def _build_row(self, meta: dict, result: dict, recv_ts: float) -> dict:
        gpu_ms = result.get("gpu_ms")
        # Provenance is per-container, not per-chunk: keep it once, in meta.
        if getattr(self, "_core_source", None) is None:
            self._core_source = result.get("core_source")
        return {
            "chunk_id": meta["chunk_id"],
            "n_frames": meta["n_frames"],
            "first_frame_ts": meta["first_frame_ts"],
            "last_frame_ts": meta["last_frame_ts"],
            "chunk_ready_offset_s": round(meta["chunk_ready_wall"] - self.replay_t0, 3),
            "client_send_ts": meta["client_send_ts"],
            "client_recv_ts": recv_ts,
            "dispatch_delay_s": round(meta["client_send_ts"] - meta["chunk_ready_wall"], 3),
            "round_trip_s": round(recv_ts - meta["client_send_ts"], 3),
            "gpu_s": round(gpu_ms / 1000.0, 3) if gpu_ms is not None else None,
            "n_poses": result.get("n_poses"),
            "server_recv_ts": result.get("server_recv_ts"),
            "server_done_ts": result.get("server_done_ts"),
            "lag_behind_clock_s": round(
                lag_behind_clock(recv_ts, self.replay_t0, self._ts[0], meta["last_frame_ts"]),
                3,
            ),
            "out_of_order": result.get("out_of_order", False),
            "n_submaps_total": result.get("n_submaps_total"),
            "loop_closures_total": result.get("loop_closures_total"),
        }

    def run(self) -> dict:
        core_source = None
        if self.reset_first:
            out = self.transport.reset()
            # The remote reset echoes the image's core-source stamp; keep it so
            # the results file can name the core that produced these numbers.
            if isinstance(out, dict):
                core_source = out.get("core_source")

        dispatcher = threading.Thread(target=self._dispatch_loop, name="dispatcher", daemon=True)
        collector = threading.Thread(target=self._collect_loop, name="collector", daemon=True)
        dispatcher.start()
        collector.start()

        # -- replay loop: the clock. Only sleeps + local file reads + queue puts.
        self.replay_t0 = time.time()
        buf_names: list[str] = []
        buf_bytes: list[bytes] = []
        chunk_id = 0
        n = len(self.frames)
        for i, (path, ts) in enumerate(zip(self.frames, self._ts)):
            target = self.replay_t0 + (ts - self._ts[0])
            delay = target - time.time()
            if delay > 0:
                time.sleep(delay)
            buf_names.append(path.name)
            buf_bytes.append(path.read_bytes())
            if len(buf_names) == self.chunk_size + 1 or i == n - 1:
                meta = {
                    "chunk_id": chunk_id,
                    "n_frames": len(buf_names),
                    "first_frame_ts": parse_frame_ts(buf_names[0]),
                    "last_frame_ts": parse_frame_ts(buf_names[-1]),
                    "chunk_ready_wall": time.time(),
                }
                self._chunks_q.put((chunk_id, list(buf_names), list(buf_bytes), meta))
                buf_names, buf_bytes = buf_names[-1:], buf_bytes[-1:]
                chunk_id += 1

        self._chunks_q.put(None)
        dispatcher.join()
        collector.join()
        if self._collector_error is not None:
            raise self._collector_error

        duration = self._ts[-1] - self._ts[0]
        replay_rate_fps = (n - 1) / duration if duration > 0 else float("nan")
        doc = {
            "meta": {
                "label": self.label,
                "frames_dir": self.frames_dir,
                "n_frames": n,
                "chunk_size": self.chunk_size,
                "session_duration_s": round(duration, 3),
                "replay_t0": self.replay_t0,
                "reset_first": self.reset_first,
                "transport": getattr(self.transport, "note", type(self.transport).__name__),
                "app": f"{APP_NAME}/{CLS_NAME}",
                "core_source": core_source or getattr(self, "_core_source", None),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            "chunks": self.rows,
            "summary": compute_summary(self.rows, replay_rate_fps),
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(doc, f, indent=2)
        return doc


# -- CLI ----------------------------------------------------------------------


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", required=True, help="session frames dir (timestamp-named jpgs)")
    ap.add_argument("--out", required=True, help="output dir (or a .json path)")
    ap.add_argument("--out-name", default=RESULTS_BASENAME, help="filename when --out is a dir")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--label", default="live", help="run label recorded in results meta")
    ap.add_argument(
        "--skip-reset",
        action="store_true",
        help="do not call reset() first (use for the cold run against a scaled-to-zero app)",
    )
    ap.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_S)
    ap.add_argument("--max-chunk-wait", type=float, default=DEFAULT_MAX_CHUNK_WAIT_S)
    ap.add_argument("--app-name", default=APP_NAME)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out_path = out if out.suffix == ".json" else out / args.out_name

    transport = ModalTransport(app_name=args.app_name, poll_timeout_s=args.poll_timeout)
    driver = ReplayDriver(
        frames_dir=args.frames,
        out_path=out_path,
        transport=transport,
        chunk_size=args.chunk_size,
        label=args.label,
        reset_first=not args.skip_reset,
        poll_timeout_s=args.poll_timeout,
        max_chunk_wait_s=args.max_chunk_wait,
    )
    doc = driver.run()
    print(json.dumps(doc["summary"], indent=2))
    print(f"wrote {out_path}")
    return doc


if __name__ == "__main__":
    main()
