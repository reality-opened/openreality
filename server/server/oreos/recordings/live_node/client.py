"""OreosLiveClient: stream a recorded session into the live node over ONE WebSocket.

The measurement counterpart of ``server/oreos/recordings/live_probe/driver.py``, with the
transport the probe's verdict demanded: chunks flow over one persistent
connection into the warm container — no per-chunk RPC, no Modal per-call
overhead. Results stream back full duplex the moment each chunk completes.

Mechanics (mirrors the probe driver wherever comparability matters):

  * Frames replay at their RECORDED rate (filenames are timestamps); the replay
    clock NEVER blocks on the network or the GPU.
  * Chunking is the probe's own ``chunk_frames`` (16+1 with 1-frame overlap) —
    imported, not copied, so the two runs are chunk-for-chunk identical.
  * Threads: a REPLAY thread only keeps the clock and publishes finished chunks;
    a per-connection SENDER thread writes them to the socket; a per-connection
    READER thread receives results and builds timing rows; the main thread is a
    connection SUPERVISOR.
  * ``client_send_ts`` is stamped when the sender FIRST starts writing the chunk
    to the socket (the probe stamped just before ``.spawn``, which uploads args —
    same semantics), so ``round_trip_s`` is directly comparable to the probe's.

Reconnect-and-resume: this dev box's proxy/route kills long-lived WebSocket
tunnels non-deterministically within seconds (measured against a public WS echo
too — a property of THIS network, not of Modal or the architecture; it is the
same pathology the probe driver documented when it chose short polls). The
supervisor therefore reconnects on any drop and resends unacked chunks; the
server keeps SLAM state + a per-chunk result cache across connections, so
resends are idempotent (served from cache, never reprocessed). On a sane robot
network the run is one connection end to end; here ``meta.n_reconnects``
records how hostile the path was. Client keepalive pings are DISABLED: with a
saturated uplink the ping queues behind megabytes of chunk bytes and websockets
self-terminates on pong timeout (observed) — liveness is the watchdog's job.

Optional ``--jpeg-quality N`` re-encodes frames before the clock starts and
keeps each frame's SMALLER encoding (never silently ships a bigger file).
Measured on go2_short: the recorded 720p jpegs are already compressed harder
than q80 — re-encoding at 80 GROWS them 1.23x, so the default is passthrough.

Output: same ``{meta, chunks, summary}`` schema as ``live_probe_results.json``
(``compute_summary`` is imported from the probe driver) with
``meta.transport = "websocket"`` plus the server-side timing split
(``server_dispatch_overhead_s`` etc.) that makes the architecture verdict
independent of this network's uplink.

Usage (deploy modal_stream.py first — see its docstring):

    python server/oreos/recordings/live_node/client.py \
        --frames sessions/go2_short/frames \
        --out    sessions/go2_short/live_node \
        --label  warm [--url https://...modal.run]

Without ``--url`` the endpoint is resolved from the deployed Modal app.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from server.oreos.recordings.live_node import protocol as proto
from server.oreos.recordings.live_probe.driver import (
    DEFAULT_CHUNK_SIZE,
    chunk_frames,
    compute_summary,
    lag_behind_clock,
    list_frames,
    parse_frame_ts,
    percentile,
)

APP_NAME = "oreos-live-node"
CLS_NAME = "LiveNode"
RESULTS_BASENAME = "live_node_results.json"
DEFAULT_MAX_CHUNK_WAIT_S = 120.0
DEFAULT_OPEN_TIMEOUT_S = 300.0  # first connect may ride a cold container boot
MAX_RECONNECTS = 50
MAX_CONSECUTIVE_CONNECT_FAILURES = 6


class LiveNodeError(RuntimeError):
    """Fatal client failure (remote error, exhausted reconnects, stalled stream)."""


class BusyError(LiveNodeError):
    """The node is serving another session (retryable shortly after a drop)."""


# -- pure helpers (unit-tested; no websockets import) --------------------------


def http_to_ws(url: str) -> str:
    """Map the deployed endpoint URL to its WebSocket route."""
    url = url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    if not url.endswith("/ws"):
        url += "/ws"
    return url


def build_chunk_plan(frames: list, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[list]:
    """The client's chunking IS the probe's: same fire boundary, same overlap."""
    return chunk_frames(frames, chunk_size=chunk_size)


def reencode_jpeg(blob: bytes, quality: int) -> tuple[bytes, bool]:
    """Re-encode a jpeg at ``quality``; keep whichever encoding is SMALLER.

    Returns ``(bytes, reencoded)``. quality <= 0 means passthrough. Never
    silently degrades AND grows: a re-encode that comes out larger is dropped.
    """
    if quality <= 0:
        return blob, False
    import cv2  # lazy: hermetic tests don't need it
    import numpy as np

    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return blob, False
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or len(buf) >= len(blob):
        return blob, False
    return buf.tobytes(), True


def build_row(meta: dict, result: dict, recv_ts: float, replay_t0: float, first_src_ts: float) -> dict:
    """One per-chunk timing row — probe schema + the live-node timing split."""
    gpu_ms = result.get("gpu_ms")
    extract_ms = result.get("extract_ms")
    sent_done = meta.get("client_sent_done_ts")
    tum = result.get("poses_tum_lines") or []
    return {
        # -- probe-comparable fields (identical names + semantics) ------------
        "chunk_id": meta["chunk_id"],
        "n_frames": meta["n_frames"],
        "first_frame_ts": meta["first_frame_ts"],
        "last_frame_ts": meta["last_frame_ts"],
        "chunk_ready_offset_s": round(meta["chunk_ready_wall"] - replay_t0, 3),
        "client_send_ts": meta["client_send_ts"],
        "client_recv_ts": recv_ts,
        "dispatch_delay_s": round(meta["client_send_ts"] - meta["chunk_ready_wall"], 3),
        "round_trip_s": round(recv_ts - meta["client_send_ts"], 3),
        "gpu_s": round(gpu_ms / 1000.0, 3) if gpu_ms is not None else None,
        "n_poses": result.get("n_poses"),
        "server_recv_ts": result.get("server_recv_ts"),
        "server_done_ts": result.get("server_done_ts"),
        "lag_behind_clock_s": round(
            lag_behind_clock(recv_ts, replay_t0, first_src_ts, meta["last_frame_ts"]), 3
        ),
        "out_of_order": result.get("out_of_order", False),
        "n_submaps_total": result.get("n_submaps_total"),
        "loop_closures_total": result.get("loop_closures_total"),
        # -- live-node additions ----------------------------------------------
        "send_duration_s": round(sent_done - meta["client_send_ts"], 3) if sent_done else None,
        "chunk_bytes_sent": meta.get("chunk_bytes"),
        "resends": meta.get("resends", 0),
        "served_from_cache": bool(result.get("served_from_cache", False)),
        "extract_s": round(extract_ms / 1000.0, 3) if extract_ms is not None else None,
        "server_start_ts": result.get("server_start_ts"),
        "server_dispatch_overhead_s": result.get("server_dispatch_overhead_s"),
        "server_idle_before_s": result.get("server_idle_before_s"),
        "server_queue_wait_s": result.get("server_queue_wait_s"),
        "arrived_before_prev_done": result.get("arrived_before_prev_done"),
        "submap_lc_status": result.get("submap_lc_status"),
        "cloud_summary": result.get("cloud_summary"),
        "poses_tum_first": tum[0] if tum else None,
        "poses_tum_last": tum[-1] if tum else None,
    }


def extend_summary(summary: dict, rows: list) -> dict:
    """Live-node summary extras: the transport-independent architecture split."""
    summary = dict(summary)
    summary["transport"] = "websocket"
    steady = rows[1:] if len(rows) >= 4 else rows

    def pct_block(key):
        vals = [r[key] for r in steady if r.get(key) is not None]
        if not vals:
            return None
        return {
            "p50": round(percentile(vals, 50), 4),
            "p95": round(percentile(vals, 95), 4),
            "max": round(max(vals), 4),
        }

    summary["server_dispatch_overhead_s"] = pct_block("server_dispatch_overhead_s")
    summary["server_idle_before_s"] = pct_block("server_idle_before_s")
    summary["send_duration_s"] = pct_block("send_duration_s")
    summary["n_chunks_resent"] = sum(1 for r in rows if r.get("resends", 0) > 0)
    overhead = summary["server_dispatch_overhead_s"]
    if overhead is not None:
        summary["architecture_note"] = (
            f"server-side dispatch overhead p50 {overhead['p50']:.3f}s per chunk "
            "(probe spawn-per-chunk path, measured from live_probe_results.json "
            "server timestamps: p50 1.94s, range 0.9-3.6s inter-chunk idle). This is "
            "measured on the container clock between 'chunk fully received AND GPU free' "
            "and 'GPU starts' — independent of the client uplink, so it carries the "
            "architecture verdict even if this network's uplink still grows the lag."
        )
    return summary


# -- the client ----------------------------------------------------------------


class OreosLiveClient:
    """Replays a session's frames at recorded rate into one live-node WebSocket,
    reconnecting-and-resuming if the network drops the tunnel."""

    def __init__(
        self,
        url: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        label: str = "live",
        reset_first: bool = True,
        reset_via: str = "http",  # "http": POST /reset before connecting (robust on
        # flaky tunnels); "ws": in-band fire-and-forget reset message (used by tests)
        jpeg_quality: int = 0,
        max_chunk_wait_s: float = DEFAULT_MAX_CHUNK_WAIT_S,
        open_timeout_s: float = DEFAULT_OPEN_TIMEOUT_S,
        proxy=True,  # websockets semantics: True = env proxies, None = direct, str = explicit
        max_frame_bytes: int = proto.MAX_WS_MESSAGE_BYTES,
        max_reconnects: int = MAX_RECONNECTS,
    ):
        self.http_url = url.rstrip("/").removesuffix("/ws")
        self.ws_url = http_to_ws(url)
        self.chunk_size = chunk_size
        self.label = label
        self.reset_first = reset_first
        self.reset_via = reset_via
        self.jpeg_quality = jpeg_quality
        self.max_chunk_wait_s = max_chunk_wait_s
        self.open_timeout_s = open_timeout_s
        self.proxy = proxy
        self.max_frame_bytes = max_frame_bytes
        self.max_reconnects = max_reconnects

        self._hello: dict | None = None
        # chunk store: replay thread appends; senders read; reader acks.
        self._chunks: list[tuple[dict, list[bytes]]] = []  # (meta, wire_frames)
        self._chunks_cond = threading.Condition()
        self._n_chunks_planned: int | None = None
        self._replay_done = threading.Event()
        self._acked: set[int] = set()
        self.rows: list[dict] = []
        self._state_lock = threading.Lock()
        self._done = threading.Event()
        self._fatal: str | None = None
        self._last_progress = time.time()
        self.n_reconnects = 0
        self.replay_t0: float | None = None
        self._first_src_ts: float | None = None

    # -- out-of-band reset (HTTP survives where a quiet WS tunnel dies) --------

    def reset_http(self, attempts: int = 3) -> dict:
        import urllib.request

        last: Exception | None = None
        for i in range(attempts):
            try:
                req = urllib.request.Request(self.http_url + "/reset", method="POST", data=b"")
                with urllib.request.urlopen(req, timeout=self.open_timeout_s) as resp:
                    return json.loads(resp.read())
            except Exception as e:  # noqa: BLE001 — retry then surface
                last = e
                time.sleep(2 * (i + 1))
        raise LiveNodeError(f"HTTP reset failed after {attempts} attempts: {last}")

    # -- one connection attempt -------------------------------------------------

    def _connect_ws(self, send_reset: bool):
        """Open a socket, wait for hello, optionally fire in-band reset."""
        from websockets.sync.client import connect as ws_connect

        ws = ws_connect(
            self.ws_url,
            open_timeout=self.open_timeout_s,
            max_size=8 * 2**20,
            close_timeout=5,
            proxy=self.proxy,
            # NO keepalive: with a saturated uplink the ping queues behind chunk
            # bytes and websockets kills the connection on pong timeout (observed).
            ping_interval=None,
        )
        try:
            assembler = proto.Assembler()
            deadline = time.time() + self.open_timeout_s
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise LiveNodeError(f"timed out waiting for hello from {self.ws_url}")
                complete = assembler.feed(ws.recv(timeout=remaining))
                if complete is None:
                    continue
                header, _ = complete
                if header.get("type") == proto.T_BUSY:
                    raise BusyError("live node is busy with another streaming session")
                if header.get("type") == proto.T_HELLO:
                    self._hello = header
                    break
            if send_reset:
                # Fire-and-forget: the server's FIFO queue guarantees the reset
                # lands before chunk 0. Waiting for reset_ok would leave the
                # tunnel quiet for the ~2.5s solver re-init — a measured kill
                # window on this network.
                ws.send(proto.encode_message({"type": proto.T_RESET}))
        except BaseException:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        return ws

    # -- public one-shot connect (handshake check; used by tests) --------------

    def connect(self):
        """Open one connection and complete the hello handshake (no streaming)."""
        ws = self._connect_ws(send_reset=False)
        self._ws_probe = ws
        return self._hello

    # -- per-connection worker threads -----------------------------------------

    def _sender_loop(self, ws, conn_dead: threading.Event) -> None:
        try:
            i = 0
            while not conn_dead.is_set() and not self._done.is_set():
                with self._chunks_cond:
                    have = len(self._chunks)
                if i < have:
                    with self._state_lock:
                        skip = self._chunks[i][0]["chunk_id"] in self._acked
                    if not skip:
                        meta, wire = self._chunks[i]
                        if "client_send_ts" not in meta:
                            meta["client_send_ts"] = time.time()  # FIRST attempt only
                        else:
                            meta["resends"] = meta.get("resends", 0) + 1
                        for frame in wire:
                            ws.send(frame)
                        meta["client_sent_done_ts"] = time.time()
                    i += 1
                elif self._replay_done.is_set():
                    ws.send(proto.encode_message({"type": proto.T_END}))
                    return
                else:
                    with self._chunks_cond:
                        self._chunks_cond.wait(timeout=0.2)
        except BaseException:  # noqa: BLE001 — connection died; supervisor resumes
            conn_dead.set()

    def _reader_loop(self, ws, conn_dead: threading.Event) -> None:
        assembler = proto.Assembler()
        try:
            while not conn_dead.is_set() and not self._done.is_set():
                complete = assembler.feed(ws.recv())
                self._last_progress = time.time()
                if complete is None:
                    continue
                header, _ = complete
                kind = header.get("type")
                if kind == proto.T_RESULT:
                    recv_ts = time.time()
                    if header.get("error"):
                        self._abort(
                            f"remote solver failed on chunk {header.get('chunk_id')}: "
                            f"{header['error']}\n{header.get('traceback', '')}"
                        )
                        return
                    cid = header.get("chunk_id")
                    with self._state_lock:
                        if cid in self._acked:
                            continue  # duplicate after a resume race
                        meta = next(
                            (m for m, _ in self._chunks if m["chunk_id"] == cid), None
                        )
                        if meta is None:
                            self._abort(f"result for unknown chunk {cid}")
                            return
                        self._acked.add(cid)
                        self.rows.append(
                            build_row(meta, header, recv_ts, self.replay_t0, self._first_src_ts)
                        )
                elif kind == proto.T_BYE:
                    self._done.set()
                    return
                elif kind == proto.T_ERROR:
                    self._abort(f"live node error: {header.get('error')}")
                    return
                elif kind == proto.T_BUSY:
                    conn_dead.set()  # stale session lock server-side; retry
                    return
                # reset_ok etc.: informational, ignore
        except BaseException:  # noqa: BLE001 — connection died; supervisor resumes
            conn_dead.set()

    def _abort(self, msg: str) -> None:
        if self._fatal is None:
            self._fatal = msg
        self._done.set()

    def _complete(self) -> bool:
        if self._n_chunks_planned is None or not self._replay_done.is_set():
            return False
        with self._state_lock:
            return len(self._acked) >= self._n_chunks_planned

    # -- replay thread: the clock ----------------------------------------------

    def _replay_loop(self, plan: list, wires: list) -> None:
        """Stamp chunk_ready at recorded-rate boundaries; publish to the store.

        All heavy work (file reads, re-encode, wire encoding) happened before
        the clock started — this loop only sleeps and publishes.
        """
        for (meta, _), wire in zip(plan, wires):
            target = self.replay_t0 + (meta["last_frame_ts"] - self._first_src_ts)
            delay = target - time.time()
            if delay > 0:
                time.sleep(delay)
            meta["chunk_ready_wall"] = time.time()
            with self._chunks_cond:
                self._chunks.append((meta, wire))
                self._chunks_cond.notify_all()
        self._replay_done.set()
        with self._chunks_cond:
            self._chunks_cond.notify_all()

    # -- the run ----------------------------------------------------------------

    def run(self, frames_dir: str | Path, out_path: str | Path) -> dict:
        frames = list_frames(frames_dir)
        ts = [parse_frame_ts(p) for p in frames]
        self._first_src_ts = ts[0]

        # Pre-load (+ optionally re-encode) and pre-encode ALL wire frames
        # before the clock starts: the replay loop must only sleep + publish.
        payloads: list[bytes] = []
        n_reencoded = 0
        bytes_original = 0
        for p in frames:
            blob = p.read_bytes()
            bytes_original += len(blob)
            blob, did = reencode_jpeg(blob, self.jpeg_quality)
            n_reencoded += did
            payloads.append(blob)
        bytes_sent_frames = sum(len(b) for b in payloads)

        index = {p.name: i for i, p in enumerate(frames)}
        plan: list[tuple[dict, list[bytes]]] = []
        wires: list[list[bytes]] = []
        for chunk_id, chunk in enumerate(build_chunk_plan(frames, self.chunk_size)):
            names = [p.name for p in chunk]
            blobs = [payloads[index[n]] for n in names]
            meta = {
                "chunk_id": chunk_id,
                "n_frames": len(names),
                "first_frame_ts": parse_frame_ts(names[0]),
                "last_frame_ts": parse_frame_ts(names[-1]),
            }
            message = proto.encode_message(
                {
                    "type": proto.T_CHUNK,
                    "chunk_id": chunk_id,
                    "names": names,
                    "client_enqueue_ts": None,  # stamped server-side comparisons unused
                },
                blobs,
            )
            wire = proto.iter_wire_frames(message, self.max_frame_bytes)
            meta["chunk_bytes"] = sum(len(f) for f in wire)
            plan.append((meta, []))
            wires.append(wire)
        self._n_chunks_planned = len(plan)

        # Out-of-band reset, then first connection BEFORE the clock starts
        # (mirrors the probe: reset -> replay; connect time is not lag).
        if self.reset_first and self.reset_via == "http":
            self.reset_http()
        ws = self._connect_ws(send_reset=self.reset_first and self.reset_via == "ws")

        self.replay_t0 = time.time()
        replay = threading.Thread(
            target=self._replay_loop, args=(plan, wires), name="replay", daemon=True
        )
        replay.start()

        # -- supervisor: keep exactly one sender+reader pair on a live socket.
        consecutive_failures = 0
        try:
            while not self._done.is_set():
                conn_dead = threading.Event()
                sender = threading.Thread(
                    target=self._sender_loop, args=(ws, conn_dead), name="sender", daemon=True
                )
                reader = threading.Thread(
                    target=self._reader_loop, args=(ws, conn_dead), name="reader", daemon=True
                )
                sender.start()
                reader.start()
                while not self._done.is_set() and not conn_dead.is_set():
                    if self._complete():
                        self._done.set()
                        break
                    if time.time() - self._last_progress > self.max_chunk_wait_s:
                        conn_dead.set()  # zombie tunnel: force a reconnect
                        break
                    time.sleep(0.25)
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass
                sender.join(timeout=10)
                reader.join(timeout=10)
                if self._done.is_set():
                    break
                # -- resume: reconnect (NEVER reset — SLAM state must survive).
                while not self._done.is_set():
                    if self.n_reconnects >= self.max_reconnects:
                        self._abort(f"gave up after {self.n_reconnects} reconnects")
                        break
                    if consecutive_failures >= MAX_CONSECUTIVE_CONNECT_FAILURES:
                        self._abort(
                            f"{consecutive_failures} consecutive connect failures "
                            f"({len(self._acked)}/{self._n_chunks_planned} chunks acked)"
                        )
                        break
                    try:
                        self.n_reconnects += 1
                        ws = self._connect_ws(send_reset=False)
                        consecutive_failures = 0
                        self._last_progress = time.time()
                        break
                    except BusyError:
                        consecutive_failures += 1
                        time.sleep(2.0)  # server session lock still draining
                    except Exception:  # noqa: BLE001
                        consecutive_failures += 1
                        time.sleep(min(1.0 * consecutive_failures, 5.0))
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        replay.join(timeout=30)
        if self._fatal is not None:
            raise LiveNodeError(self._fatal)
        if not self._complete():
            with self._state_lock:
                missing = sorted(
                    set(range(self._n_chunks_planned)) - {r["chunk_id"] for r in self.rows}
                )
            raise LiveNodeError(f"run ended but chunks missing results: {missing}")

        with self._state_lock:
            rows = sorted(self.rows, key=lambda r: r["chunk_id"])
        n = len(frames)
        duration = ts[-1] - ts[0]
        replay_rate_fps = (n - 1) / duration if duration > 0 else float("nan")
        doc = {
            "meta": {
                "label": self.label,
                "frames_dir": str(frames_dir),
                "n_frames": n,
                "chunk_size": self.chunk_size,
                "session_duration_s": round(duration, 3),
                "replay_t0": self.replay_t0,
                "reset_first": self.reset_first,
                "transport": "websocket",
                "transport_note": (
                    "persistent WebSocket into the warm container "
                    f"({self.ws_url}); chunks stream full duplex (uplink of chunk k+1 "
                    "overlaps GPU on chunk k), results return the moment each chunk "
                    "completes — no per-chunk RPC. On tunnel drops (this network kills "
                    "long-lived streams; measured against a public WS echo too) the "
                    "client reconnects and resends unacked chunks; the server serves "
                    "duplicates from its result cache without reprocessing. "
                    "round_trip_s = FIRST send attempt to result received (outages "
                    "included — honest for lag); gpu_s is pure server-side time."
                ),
                "n_reconnects": self.n_reconnects,
                "app": f"{APP_NAME}/{CLS_NAME}",
                "jpeg_quality": self.jpeg_quality,
                "n_frames_reencoded": n_reencoded,
                "bytes_frames_original": bytes_original,
                "bytes_frames_sent": bytes_sent_frames,
                "max_ws_frame_bytes": self.max_frame_bytes,
                "container_boot_s": (self._hello or {}).get("container_boot_s"),
                # Which core the node's image installed (build-time stamp echoed
                # in `hello`) — a measurement nobody can attribute is not a
                # measurement. See server/oreos/recordings/modal_image.py.
                "core_source": (self._hello or {}).get("core_source"),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            "chunks": rows,
            "summary": extend_summary(compute_summary(rows, replay_rate_fps), rows),
        }
        doc["summary"]["n_reconnects"] = self.n_reconnects
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        return doc


# -- CLI ------------------------------------------------------------------------


def resolve_url(app_name: str = APP_NAME, cls_name: str = CLS_NAME) -> str:
    """Look up the deployed web endpoint's URL via Modal (lazy import)."""
    import modal  # lazy: tests never import modal

    fn = getattr(modal.Cls.from_name(app_name, cls_name)(), "web")
    url = fn.get_web_url()
    if not url:
        raise LiveNodeError(f"could not resolve web URL for {app_name}/{cls_name}")
    return url


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
        help="do not reset first (continue on the node's existing SLAM state)",
    )
    ap.add_argument("--max-chunk-wait", type=float, default=DEFAULT_MAX_CHUNK_WAIT_S)
    ap.add_argument("--url", default=None, help="endpoint URL; default: resolve from Modal")
    ap.add_argument("--app-name", default=APP_NAME)
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=0,
        help="re-encode frames at this jpeg quality when it SHRINKS them (0 = passthrough; "
        "note: go2_short's recorded jpegs are already smaller than a q80 re-encode)",
    )
    ap.add_argument(
        "--no-proxy", action="store_true", help="bypass env proxies (direct connection)"
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out_path = out if out.suffix == ".json" else out / args.out_name
    url = args.url or resolve_url(app_name=args.app_name)

    client = OreosLiveClient(
        url=url,
        chunk_size=args.chunk_size,
        label=args.label,
        reset_first=not args.skip_reset,
        jpeg_quality=args.jpeg_quality,
        max_chunk_wait_s=args.max_chunk_wait,
        proxy=None if args.no_proxy else True,
    )
    doc = client.run(frames_dir=args.frames, out_path=out_path)
    print(json.dumps(doc["summary"], indent=2))
    print(f"wrote {out_path}")
    return doc


if __name__ == "__main__":
    main()
