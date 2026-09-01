"""The live node's ASGI app — ``/health``, ``/reset`` (HTTP) and ``/ws`` (WebSocket).

Framework-free and **modal-free**: it drives a "node" object (in production the
warm ``LiveNode`` GPU container from ``modal_stream.py``, in tests a fake) through
a small, explicit interface, so the connection lifecycle the 2026-07-24 audit
found broken is exercisable in CI without Modal or a GPU.

The node interface (see ``modal_stream.LiveNode``):

    node.session_lock                 threading.Lock — one streaming session/reset
    node.state                        session.LiveSessionState (snapshot + cache)
    node.boot_s                       float, model+solver load time
    node.status()                     dict for /health, snapshot-only (never live
                                      solver objects — that was audit finding L5)
    node.reset_sync()                 sync; clears Solver + session state
    node.process_chunk_sync(id, names, blobs)   sync; one GPU cycle -> result dict
    node.core_source()                dict; which core the image installed, so a
                                      client's results file can name it

Audit findings fixed in this file (see ``session.py`` for the state half):

  L1  the worker's ``await q.get()`` had no timeout. A half-open TCP connection
      never enqueues a disconnect, so the worker blocked forever holding
      ``session_lock`` — for the container's full ``timeout=3600`` — and with
      ``max_containers=1`` every later connection got ``busy`` and every
      ``/reset`` 409'd for an hour. The wait is now bounded by
      :class:`session.StallWatchdog` (``OREOS_LIVE_IDLE_TIMEOUT_S``, default
      120 s); on a stall the socket is closed, the queue drained and the session
      released cleanly, with the stall logged.
  L3  the receiver->worker queue was unbounded despite the docstring promising
      "uplink of chunk k+1 overlaps GPU on chunk k". It is now
      ``Queue(maxsize=OREOS_LIVE_UPLINK_QUEUE_MAX)`` (default 2): real
      backpressure, bounded memory, same overlap.
  L4  ``recv_task.cancel()`` was never awaited, so the receiver could still be
      running (and swallowing its own exceptions) after ``session_lock`` was
      released. It is now cancelled AND awaited, with any non-cancellation
      exception logged.
  L6  a NEW connection whose first chunk is ``chunk_id=0`` while the node still
      holds a previous session's SLAM state is refused with a protocol error
      (``session_state_present``) instead of silently appending its frames to the
      stale map. A resume (chunk already in the results cache) is still served.
"""

from __future__ import annotations

import asyncio
import json as _json
import time

try:  # local import (deploy-time, tests): the server repo package
    from . import protocol as proto
    from . import session as sess
except ImportError:  # remote container: package mounted flat at /root/oreos_live_node
    import oreos_live_node.protocol as proto  # type: ignore[no-redef]
    import oreos_live_node.session as sess  # type: ignore[no-redef]

#: How long to wait for a courtesy close frame on a socket we believe is dead.
CLOSE_TIMEOUT_S = 5.0


def build_asgi(node):
    """Raw ASGI app around a warm node instance."""

    async def send_json(send, status: int, obj: dict) -> None:
        body = _json.dumps(obj, indent=2).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def handle_http(scope, receive, send) -> None:
        # Drain the (empty) request body so the connection is well-behaved.
        while True:
            msg = await receive()
            if msg["type"] != "http.request" or not msg.get("more_body"):
                break
        path, method = scope["path"], scope["method"]
        if path == "/health" and method in ("GET", "HEAD"):
            await send_json(send, 200, node.status())
        elif path == "/reset" and method == "POST":
            if not node.session_lock.acquire(blocking=False):
                await send_json(send, 409, {"error": "a streaming session is active"})
                return
            try:
                out = await asyncio.to_thread(node.reset_sync)
                await send_json(send, 200, {"reset": True, **out})
            finally:
                node.session_lock.release()
        else:
            await send_json(send, 404, {"error": f"no route {method} {path}"})

    async def ws_send(send, header: dict, blobs: list = ()) -> None:
        for frame in proto.iter_wire_frames(proto.encode_message(header, blobs)):
            await send({"type": "websocket.send", "bytes": frame})

    async def close_quietly(send, code: int, header: dict | None = None) -> None:
        """Best-effort "goodbye + close" on a socket that may already be dead.

        Bounded by :data:`CLOSE_TIMEOUT_S` and never raises: this runs on the
        stall path, where the whole point is that the peer is unreachable.
        """
        try:
            async with asyncio.timeout(CLOSE_TIMEOUT_S):
                if header is not None:
                    await ws_send(send, header)
                await send({"type": "websocket.close", "code": code})
        except BaseException as e:  # noqa: BLE001 — the peer is presumed gone
            print(f"[live-node] close on a dead socket: {type(e).__name__}: {e}")

    async def handle_ws(scope, receive, send) -> None:
        msg = await receive()
        if msg["type"] != "websocket.connect":
            return
        await send({"type": "websocket.accept"})
        if scope["path"] not in ("/ws", "/"):
            await ws_send(send, {"type": proto.T_ERROR, "error": f"no ws route {scope['path']}"})
            await send({"type": "websocket.close", "code": 1008})
            return
        if not node.session_lock.acquire(blocking=False):
            await ws_send(send, {"type": proto.T_BUSY, "note": "another session is streaming"})
            await send({"type": "websocket.close", "code": 1013})
            return
        try:
            reason = await _session(scope, receive, send)
            print(f"[live-node] session ended: {reason}")
        finally:
            node.session_lock.release()

    async def _session(scope, receive, send) -> str:
        """One streaming session: receiver task + single GPU worker task."""
        snap = node.state.snapshot()
        await ws_send(
            send,
            {
                "type": proto.T_HELLO,
                "server_ts": time.time(),
                "container_boot_s": round(node.boot_s, 2),
                "next_chunk_id": snap["next_chunk_id"],
                "n_submaps_total_at_connect": snap["n_submaps_total"],
                "has_solver_state": snap["has_solver_state"],
                "idle_timeout_s": sess.idle_timeout_s(),
                "core_source": node.core_source(),
            },
        )
        q: asyncio.Queue = asyncio.Queue(maxsize=sess.uplink_queue_max())
        assembler = proto.Assembler()

        async def receiver() -> None:
            """Drain the socket -> queue. Never blocks on the GPU: uplink of
            chunk k+1 proceeds while the worker runs chunk k (full duplex), up to
            the queue's maxsize — past that the peer is throttled on purpose."""
            try:
                while True:
                    msg = await receive()
                    if msg["type"] == "websocket.disconnect":
                        print(f"[live-node] receiver: disconnect code={msg.get('code')}")
                        await q.put(("disconnect", None, None))
                        return
                    data = msg.get("bytes")
                    if data is None:  # text frames are not part of the protocol
                        continue
                    try:
                        complete = assembler.feed(data)
                    except proto.ProtocolError as e:
                        print(f"[live-node] receiver: protocol error: {e}")
                        await q.put(("protocol_error", str(e), None))
                        return
                    if complete is None:
                        continue
                    header, blobs = complete
                    await q.put((header.get("type"), header, blobs, time.time()))
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001 — log; worker unblocks via queue
                print(f"[live-node] receiver crashed: {type(e).__name__}: {e}")
                await q.put(("disconnect", None, None))
                raise

        async def worker() -> str:
            prev_done_ts = None
            first_chunk = True
            watchdog = sess.StallWatchdog(now=time.time())
            while True:
                try:
                    item = await asyncio.wait_for(
                        q.get(), timeout=watchdog.wait_timeout(time.time())
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    # L1: a half-open connection enqueues nothing, ever. Give the
                    # session back instead of pinning the only container.
                    report = watchdog.report(time.time())
                    print(
                        f"[live-node] STALL: no client traffic for {report['idle_s']}s "
                        f"(> {report['timeout_s']}s idle timeout, "
                        f"{report['touches']} messages this session) — closing the "
                        f"socket and releasing the session. SLAM state is kept: "
                        f"reconnect and resend unacked chunks to resume."
                    )
                    await close_quietly(
                        send,
                        1001,
                        {"type": proto.T_ERROR, "error": f"idle timeout: {report}"},
                    )
                    return "stalled"
                watchdog.touch(time.time())
                kind = item[0]
                if kind == "disconnect":
                    return "disconnect"
                if kind == "protocol_error":
                    await ws_send(send, {"type": proto.T_ERROR, "error": f"protocol: {item[1]}"})
                    await send({"type": "websocket.close", "code": 1002})
                    return "protocol_error"
                _, header, blobs, arrive_ts = item
                if kind == proto.T_RESET:
                    out = await asyncio.to_thread(node.reset_sync)
                    prev_done_ts = None
                    first_chunk = True
                    await ws_send(send, out)
                elif kind == proto.T_END:
                    await ws_send(
                        send,
                        {
                            "type": proto.T_BYE,
                            "chunks_done": node.state.chunks_done,
                            "server_ts": time.time(),
                        },
                    )
                    await send({"type": "websocket.close", "code": 1000})
                    return "bye"
                elif kind == proto.T_CHUNK:
                    try:
                        chunk_id = int(header["chunk_id"])
                    except (KeyError, TypeError, ValueError) as e:
                        await ws_send(
                            send,
                            {"type": proto.T_ERROR, "error": f"protocol: bad chunk header: {e}"},
                        )
                        await send({"type": "websocket.close", "code": 1002})
                        return "protocol_error"
                    cached = node.state.cached_result(chunk_id)
                    if cached is not None:
                        # Resent chunk after a dropped connection: idempotent —
                        # serve the original result, never reprocess. Timing
                        # fields describe the ORIGINAL processing.
                        served = dict(cached)
                        served["served_from_cache"] = True
                        first_chunk = False
                        await ws_send(send, served)
                        watchdog.touch(time.time())
                        continue
                    if first_chunk:
                        # L6: never append a fresh recording to a stale map.
                        refusal = node.state.first_chunk_refusal(chunk_id)
                        if refusal is not None:
                            print(f"[live-node] refusing session: {refusal}")
                            await ws_send(
                                send,
                                {
                                    "type": proto.T_ERROR,
                                    "code": sess.ERR_SESSION_STATE_PRESENT,
                                    "error": refusal,
                                },
                            )
                            await send({"type": "websocket.close", "code": 1008})
                            return sess.ERR_SESSION_STATE_PRESENT
                    first_chunk = False
                    start_ts = time.time()
                    # THE architecture number: dead time between "chunk available
                    # AND GPU free" and "GPU starts" (probe's spawn path: 1.0-1.4s).
                    base = arrive_ts if prev_done_ts is None else max(arrive_ts, prev_done_ts)
                    result = await asyncio.to_thread(
                        node.process_chunk_sync,
                        chunk_id,
                        list(header.get("names", [])),
                        blobs,
                    )
                    done_ts = time.time()
                    result.update(
                        server_recv_ts=arrive_ts,  # probe-schema name: chunk fully received
                        server_arrive_ts=arrive_ts,
                        server_start_ts=start_ts,
                        server_done_ts=done_ts,
                        server_dispatch_overhead_s=round(start_ts - base, 4),
                        server_idle_before_s=(
                            round(start_ts - prev_done_ts, 4) if prev_done_ts is not None else None
                        ),
                        server_queue_wait_s=round(start_ts - arrive_ts, 4),
                        arrived_before_prev_done=(
                            prev_done_ts is not None and arrive_ts <= prev_done_ts
                        ),
                        client_enqueue_ts_echo=header.get("client_enqueue_ts"),
                    )
                    prev_done_ts = done_ts
                    if "error" not in result:
                        node.state.remember_result(chunk_id, result)
                    await ws_send(send, result)
                    watchdog.touch(time.time())
                else:
                    await ws_send(
                        send, {"type": proto.T_ERROR, "error": f"unknown message type {kind!r}"}
                    )

        recv_task = asyncio.create_task(receiver())
        try:
            return await worker()
        except BaseException as e:  # noqa: BLE001 — make session failures visible in logs
            import traceback

            print(f"[live-node] worker crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        finally:
            # L4: cancel AND await. Otherwise the receiver can still be running
            # (and its exceptions silently discarded) after session_lock is
            # released — i.e. reading the socket of a session that is over.
            recv_task.cancel()
            while True:  # unblock a receiver parked on a full queue, then let it die
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            outcome = await asyncio.gather(recv_task, return_exceptions=True)
            exc = outcome[0] if outcome else None
            if isinstance(exc, BaseException) and not isinstance(exc, asyncio.CancelledError):
                print(f"[live-node] receiver ended with {type(exc).__name__}: {exc}")

    async def asgi(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            await handle_http(scope, receive, send)
        elif scope["type"] == "websocket":
            await handle_ws(scope, receive, send)

    return asgi
