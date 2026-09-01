"""Pure session/lifecycle state for the Oreos live node — no modal, no GPU, no I/O.

Extracted from ``modal_stream.py`` by the 2026-07-24 live-node audit. Everything
here is plain Python so the lifecycle rules that broke in the audit are testable
in CI (``tests/test_live_node_session.py``); ``modal_stream.py`` owns only the
GPU/Modal half and wires these objects in.

What lives here and which audit finding it answers:

  * :class:`ResultsCache`  — L2: the chunk-resume cache was an unbounded dict of
    full ``poses_tum_lines`` payloads, cleared only by an explicit reset while
    the container stays warm for ``scaledown_window`` seconds. Now LRU-capped
    (``OREOS_LIVE_RESULTS_CACHE_MAX``, default 64), which preserves the
    behaviour that matters — a reconnecting client resends only its most recent
    unacked chunks — and drops the ancient tail that nobody can ask for.
  * :class:`LiveSessionState` — L5/L6/L7:
      - L5 (/health data race): ``@modal.concurrent`` lets ``/health`` run while
        the worker thread mutates the Solver. Reading ``ordered_submaps_by_key``
        / ``get_num_loops`` from another thread mid-``add_points`` is a genuine
        race that the old blanket ``except`` degraded to a flaky ``-1``. The
        worker now publishes plain snapshot fields after each chunk and
        ``/health`` reads ONLY the snapshot: no live solver object is ever
        touched off the worker thread.
      - L6 (cross-session bleed): solver state survives a disconnect on purpose
        (resume). But a *new* client starting at ``chunk_id=0`` while state
        exists used to be merely flagged ``out_of_order`` and have its frames
        APPENDED to the previous client's map — silently wrong geometry.
        :meth:`LiveSessionState.first_chunk_refusal` refuses that connection and
        tells the operator to ``POST /reset``.
      - L7 (sticky out-of-order): ``_next_chunk_id`` advanced only on the success
        path, so one failed chunk marked every later chunk out-of-order forever.
        :meth:`LiveSessionState.finish_chunk` advances on both paths while
        keeping genuine out-of-order detection (the comparison happens in
        :meth:`begin_chunk`, before processing).
  * :class:`StallWatchdog` — L1: the worker's ``await q.get()`` never timed out.
    A half-open TCP connection enqueues no disconnect, so the worker blocked
    holding the session lock for the full ``timeout=3600``; with
    ``max_containers=1`` every later connection got ``busy`` and every ``/reset``
    409'd for an hour. The watchdog bounds that wait
    (``OREOS_LIVE_IDLE_TIMEOUT_S``, default 120 s).

Knobs (all read from the process env at session start, so they can be set on the
Modal image/function without a code change):

    OREOS_LIVE_IDLE_TIMEOUT_S    float, default 120.0; <= 0 disables the watchdog
    OREOS_LIVE_RESULTS_CACHE_MAX int,   default 64,  minimum 1
    OREOS_LIVE_UPLINK_QUEUE_MAX  int,   default 2,   minimum 1
"""

from __future__ import annotations

import os
from collections import OrderedDict

# -- env knobs -----------------------------------------------------------------

ENV_IDLE_TIMEOUT = "OREOS_LIVE_IDLE_TIMEOUT_S"
ENV_RESULTS_CACHE_MAX = "OREOS_LIVE_RESULTS_CACHE_MAX"
ENV_UPLINK_QUEUE_MAX = "OREOS_LIVE_UPLINK_QUEUE_MAX"

DEFAULT_IDLE_TIMEOUT_S = 120.0
DEFAULT_RESULTS_CACHE_MAX = 64
DEFAULT_UPLINK_QUEUE_MAX = 2

#: Error code sent to a client whose fresh stream would append to a stale map.
ERR_SESSION_STATE_PRESENT = "session_state_present"


def _env_number(name: str, default, cast, minimum=None, environ: dict | None = None):
    """Parse a numeric env knob; a bad value logs and falls back (never raises)."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = cast(str(raw).strip())
    except (TypeError, ValueError):
        print(f"[live-node] ignoring {name}={raw!r} (not a {cast.__name__}); using {default}")
        return default
    if minimum is not None and value < minimum:
        print(f"[live-node] ignoring {name}={raw!r} (< {minimum}); using {default}")
        return default
    return value


def idle_timeout_s(environ: dict | None = None) -> float:
    """Stall-watchdog timeout in seconds; ``<= 0`` disables the watchdog."""
    return _env_number(ENV_IDLE_TIMEOUT, DEFAULT_IDLE_TIMEOUT_S, float, None, environ)


def results_cache_max(environ: dict | None = None) -> int:
    return _env_number(ENV_RESULTS_CACHE_MAX, DEFAULT_RESULTS_CACHE_MAX, int, 1, environ)


def uplink_queue_max(environ: dict | None = None) -> int:
    """Max chunks buffered between the receiver and the GPU worker (backpressure).

    2 keeps the documented overlap — uplink of chunk k+1 proceeds while the GPU
    runs chunk k — without letting an unthrottled peer buffer the whole stream
    (each item holds a chunk's jpeg bytes, ~1.6 MB for a 17-frame go2 chunk).
    """
    return _env_number(ENV_UPLINK_QUEUE_MAX, DEFAULT_UPLINK_QUEUE_MAX, int, 1, environ)


# -- results cache (L2) --------------------------------------------------------


class ResultsCache:
    """Bounded LRU of ``chunk_id -> result`` for idempotent chunk resends.

    Both a hit and a store mark an entry as recently used, so the entries a
    reconnecting client can still ask for (its most recent, unacked chunks) are
    the last to be evicted.
    """

    def __init__(self, maxsize: int | None = None) -> None:
        self.maxsize = int(maxsize if maxsize is not None else results_cache_max())
        if self.maxsize < 1:
            raise ValueError(f"ResultsCache maxsize must be >= 1, got {self.maxsize}")
        self._items: OrderedDict = OrderedDict()
        self.evictions = 0

    def __contains__(self, chunk_id) -> bool:
        return chunk_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def keys(self):
        return list(self._items.keys())

    def get(self, chunk_id):
        """Return the cached result (marking it recent) or ``None``."""
        if chunk_id not in self._items:
            return None
        self._items.move_to_end(chunk_id)
        return self._items[chunk_id]

    def put(self, chunk_id, result) -> None:
        self._items[chunk_id] = result
        self._items.move_to_end(chunk_id)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._items.clear()


# -- session state (L5 / L6 / L7) ----------------------------------------------


class LiveSessionState:
    """Chunk bookkeeping + the health snapshot the worker publishes.

    Every method here is called from the GPU worker thread (or, for
    :meth:`snapshot`, from any thread). The snapshot is rebuilt as a fresh plain
    dict on each publish, so a concurrent reader can only ever see a complete
    older snapshot — never a half-mutated solver.
    """

    def __init__(self, results_cache_max_: int | None = None) -> None:
        self.results = ResultsCache(results_cache_max_)
        self.next_chunk_id = 0
        self.chunks_started = 0
        self.chunks_done = 0
        self.chunks_failed = 0
        self.n_submaps_total = 0
        self.loop_closures_total = 0
        self._snapshot = self._empty_snapshot()

    # -- snapshot ------------------------------------------------------------

    def _empty_snapshot(self) -> dict:
        return {
            "n_submaps_total": 0,
            "loop_closures_total": 0,
            "chunks_done": 0,
            "chunks_failed": 0,
            "next_chunk_id": 0,
            "last_chunk_id": None,
            "last_chunk_error": None,
            "last_update_ts": None,
            "results_cached": 0,
            "has_solver_state": False,
        }

    def _publish(self, **fields) -> None:
        """Rebuild + rebind the snapshot. Unnamed fields keep their last value.

        ``None`` for ``n_submaps``/``n_loops`` means "could not be read this
        time" (e.g. the chunk raised before the counts were taken) and keeps the
        previous number rather than inventing a ``-1``.
        """
        n_submaps = fields.pop("n_submaps", None)
        n_loops = fields.pop("n_loops", None)
        if n_submaps is not None:
            self.n_submaps_total = int(n_submaps)
        if n_loops is not None:
            self.loop_closures_total = int(n_loops)
        snap = dict(self._snapshot)
        snap.update(fields)
        snap["n_submaps_total"] = self.n_submaps_total
        snap["loop_closures_total"] = self.loop_closures_total
        snap["chunks_done"] = self.chunks_done
        snap["chunks_failed"] = self.chunks_failed
        snap["next_chunk_id"] = self.next_chunk_id
        snap["results_cached"] = len(self.results)
        snap["has_solver_state"] = self.has_solver_state
        self._snapshot = snap  # single atomic rebind — readers never see a partial dict

    def snapshot(self) -> dict:
        """A consistent copy of the last published state (safe from any thread)."""
        return dict(self._snapshot)

    @property
    def has_solver_state(self) -> bool:
        """True once any chunk has touched the Solver (success OR failure).

        A failed chunk can still have mutated the map before it raised, so
        ``chunks_started`` — not ``chunks_done`` — is the honest test.
        """
        return self.chunks_started > 0 or self.n_submaps_total > 0

    # -- chunk lifecycle ------------------------------------------------------

    def begin_chunk(self, chunk_id: int) -> bool:
        """Register the start of a chunk; returns whether it arrived out of order."""
        out_of_order = int(chunk_id) != self.next_chunk_id
        self.chunks_started += 1
        return out_of_order

    def finish_chunk(self, chunk_id: int, *, ok: bool, n_submaps=None, n_loops=None,
                     error: str | None = None, ts: float | None = None) -> None:
        """Close out a chunk and publish the snapshot.

        ``next_chunk_id`` advances on BOTH paths (L7): a chunk that raised is
        still consumed, so the stream's next expected id is ``chunk_id + 1``.
        Genuine out-of-order detection is unaffected because it is decided in
        :meth:`begin_chunk`, before the work runs.
        """
        self.next_chunk_id = int(chunk_id) + 1
        if ok:
            self.chunks_done += 1
        else:
            self.chunks_failed += 1
        self._publish(n_submaps=n_submaps, n_loops=n_loops, last_chunk_id=int(chunk_id),
                      last_chunk_error=error, last_update_ts=ts)

    def remember_result(self, chunk_id: int, result: dict) -> None:
        self.results.put(chunk_id, result)
        self._publish()  # refresh results_cached only

    def cached_result(self, chunk_id: int):
        return self.results.get(chunk_id)

    def reset(self, ts: float | None = None) -> None:
        self.next_chunk_id = 0
        self.chunks_started = 0
        self.chunks_done = 0
        self.chunks_failed = 0
        self.n_submaps_total = 0
        self.loop_closures_total = 0
        self.results.clear()
        snap = self._empty_snapshot()
        snap["last_update_ts"] = ts
        self._snapshot = snap

    # -- cross-session bleed guard (L6) --------------------------------------

    def first_chunk_refusal(self, chunk_id: int) -> str | None:
        """Reason to refuse this connection's FIRST chunk, or ``None`` to accept.

        Refused only for the unambiguous bleed case: a stream that starts at
        ``chunk_id=0`` (i.e. a NEW recording, not a resume) while this container
        still holds a previous session's SLAM state. Appending those frames to
        the old map is never correct — the two recordings share no gauge — and
        the old code merely flagged ``out_of_order`` and appended anyway.

        A resume is explicitly NOT refused: a reconnecting client resends chunks
        it never got results for, and if chunk 0 is still in the results cache it
        is served idempotently from there without touching the Solver.
        """
        if int(chunk_id) != 0:
            return None
        if not self.has_solver_state:
            return None
        if chunk_id in self.results:  # resume: served from cache, no solver mutation
            return None
        return (
            f"{ERR_SESSION_STATE_PRESENT}: this node still holds SLAM state from an "
            f"earlier session ({self.chunks_started} chunks started, "
            f"{self.n_submaps_total} submaps). A new stream starting at chunk_id=0 "
            f"would append its frames to that map, which is never correct — the two "
            f"recordings share no gauge. POST /reset (or send a `reset` message before "
            f"the first chunk) to start a fresh session."
        )


# -- stall watchdog (L1) -------------------------------------------------------


class StallWatchdog:
    """Idle-timeout decision logic for the session worker's queue wait.

    ``touch`` is called on every sign of life (session start, each dequeued
    message, each result sent) — including the completion of a long GPU chunk, so
    a slow chunk can never be mistaken for a dead peer.
    """

    #: Never ask the event loop for a zero/negative timeout.
    MIN_WAIT_S = 0.001

    def __init__(self, timeout_s: float | None = None, now: float = 0.0) -> None:
        self.timeout_s = float(idle_timeout_s() if timeout_s is None else timeout_s)
        self.last_activity = float(now)
        self.touches = 0

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0

    def touch(self, now: float) -> None:
        self.last_activity = float(now)
        self.touches += 1

    def idle_s(self, now: float) -> float:
        return max(0.0, float(now) - self.last_activity)

    def remaining_s(self, now: float) -> float:
        return self.timeout_s - self.idle_s(now)

    def wait_timeout(self, now: float) -> float | None:
        """Timeout to hand ``asyncio.wait_for``; ``None`` means wait forever."""
        if not self.enabled:
            return None
        return max(self.MIN_WAIT_S, self.remaining_s(now))

    def is_stalled(self, now: float) -> bool:
        return self.enabled and self.remaining_s(now) <= 0

    def report(self, now: float) -> dict:
        return {
            "idle_s": round(self.idle_s(now), 2),
            "timeout_s": self.timeout_s,
            "touches": self.touches,
        }
