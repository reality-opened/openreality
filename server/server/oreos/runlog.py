"""Agent-run event log: in-memory ring + persisted replayable artifact + registry.

Transport backbone for the persisted-scene agent (BUILD-PLAN decision 6): REST run +
1 s polling. A run's thread appends events to (i) the run's in-memory ring — the live
``GET …/events?after=<seq>`` reader (safe in-process: the broker deploys with
``max_containers=1``) — and (ii) a JSON artifact flushed through the store to
``derived/demo/agent_runs/<run_id>/events.json`` (buffered: every ``FLUSH_EVERY_EVENTS``
events or ``FLUSH_EVERY_S`` seconds, plus once at the end), so replay/resume survive a
broker restart and every live run is automatically its own recording.

Replay is free and deterministic, two ways:
  * ``POST …/annotate {replay_of}`` → a *replay run*: a playback thread re-emits the
    stored log into a fresh run with the original inter-event deltas clamped to
    ``[MIN_REPLAY_DELTA_S, MAX_REPLAY_DELTA_S] / speed`` — the unmodified 1 s-polling
    client renders it exactly like a live run. Zero LLM cost, byte-identical payloads.
  * ``GET …/events?replay=1[&speed=S]`` → the stored log served in one response, each
    event annotated ``delay_ms`` (same clamp) for client-side pacing.

REPLAY invariant (honesty doctrine): every replayed stream BEGINS with
``run_meta{replay: true, recorded_at, source_run_id}`` — the client keys its structural
REPLAY badge off it and there is no off switch here to disable that marker.

One active run per scene: ``RunRegistry.start_run`` raises :class:`ActiveRunError`
(→ 409 ``agent_run_active``) while the scene has a run in status ``running``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable, Optional

from server.oreos.schemas import RUN_EVENT_TYPES

# Flush cadence (agent-exports.md: "2 s/20-event buffer + at end").
FLUSH_EVERY_EVENTS = 20
FLUSH_EVERY_S = 2.0

# Replay inter-event delta clamp (agent-exports.md: "original deltas clamped
# [120 ms, 2.5 s]/speed").
MIN_REPLAY_DELTA_S = 0.12
MAX_REPLAY_DELTA_S = 2.5

# Run status vocabulary — protocol demo.ts DemoAgentRunStatus.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

EVENTS_DOC_VERSION = 1


class ActiveRunError(RuntimeError):
    """Another run is active on this scene (→ 409 agent_run_active)."""

    def __init__(self, active_run_id: str):
        super().__init__(f"agent run {active_run_id} is active on this scene")
        self.active_run_id = active_run_id


def events_key_suffix(run_id: str) -> str:
    """Key suffix under ``derived/demo/`` for a run's event log."""
    return f"agent_runs/{run_id}/events.json"


def clamp_replay_delta(delta_s: float, speed: float = 1.0) -> float:
    """Original inter-event gap → replay gap: clamped to the honest-pacing window and
    divided by ``speed`` (≥ 1e-3, so a hostile 0 can't stall a playback forever)."""
    try:
        delta = float(delta_s)
    except (TypeError, ValueError):
        delta = MIN_REPLAY_DELTA_S
    if not (delta == delta and delta >= 0):  # NaN / negative
        delta = MIN_REPLAY_DELTA_S
    clamped = min(MAX_REPLAY_DELTA_S, max(MIN_REPLAY_DELTA_S, delta))
    return clamped / max(1e-3, float(speed))


class AgentRun:
    """One agent run: metadata + the monotonic event ring + flush plumbing.

    ``flusher`` is a callable ``(run) -> None`` that persists ``run.to_doc()`` (injected
    by the routes so this module needs no store handle of its own). Flush failures are
    swallowed after logging — a persistence hiccup must never kill a live run mid-take;
    the final ``finish()`` flush retries regardless.
    """

    def __init__(
        self,
        run_id: str,
        user_id: str,
        scan_id: str,
        mode: str,
        replay: bool = False,
        source_run_id: Optional[str] = None,
        flusher: Optional[Callable[["AgentRun"], None]] = None,
        max_wall_s: Optional[float] = None,
    ):
        self.run_id = str(run_id)
        self.user_id = str(user_id)
        self.scan_id = str(scan_id)
        self.mode = str(mode)
        self.replay = bool(replay)
        self.source_run_id = source_run_id
        self.status = STATUS_RUNNING
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.deadline = self.created_at + max_wall_s if max_wall_s else None
        self.thread: Optional[threading.Thread] = None
        self._flusher = flusher
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_seq = 0
        self._unflushed = 0
        self._last_flush = 0.0
        self._indexed = False  # manifest ref written yet?

    # -- emit side (run thread) ---------------------------------------

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        phase: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append one event (monotonic ``seq``, epoch ``ts``) and maybe flush."""
        if event_type not in RUN_EVENT_TYPES:
            raise ValueError(f"unknown run event type {event_type!r}")
        with self._lock:
            event = {
                "seq": self._next_seq,
                "ts": time.time(),
                "type": event_type,
                "phase": phase,
                "payload": payload if isinstance(payload, dict) else {"value": payload},
            }
            self._next_seq += 1
            self._events.append(event)
            self._unflushed += 1
            due = (
                self._unflushed >= FLUSH_EVERY_EVENTS
                or (time.time() - self._last_flush) >= FLUSH_EVERY_S
            )
        if due:
            self._flush()
        return event

    def check_deadline(self) -> None:
        """Raise ``TimeoutError`` once the run's wall-clock budget is spent (guardrail:
        5-min wall). Called by the agent between steps — never mid-event."""
        if self.deadline is not None and time.time() > self.deadline:
            raise TimeoutError(
                f"run wall-clock budget exceeded ({int(self.deadline - self.created_at)}s)"
            )

    def finish(self, status: str = STATUS_DONE, error: Optional[str] = None) -> None:
        with self._lock:
            if self.status != STATUS_RUNNING:
                return
            self.status = STATUS_ERROR if status == STATUS_ERROR else STATUS_DONE
            self.error = error
            self.finished_at = time.time()
        self._flush()

    def _flush(self) -> None:
        if self._flusher is None:
            with self._lock:
                self._unflushed = 0
                self._last_flush = time.time()
            return
        try:
            self._flusher(self)
            with self._lock:
                self._unflushed = 0
                self._last_flush = time.time()
        except Exception as exc:  # never kill a live run over persistence
            print(f"[demo.runlog] flush failed for run {self.run_id}: {exc}")

    # -- read side (poll route) ---------------------------------------

    def events_after(self, after: int) -> tuple[list[dict[str, Any]], str, int]:
        """``(events with seq > after, status, next)`` for the poll response.

        ``next`` is the HIGHEST seq known (-1 when empty) — per protocol demo.ts it is
        passed back verbatim as the next ``?after=``, so it must be the last delivered
        cursor, not the count (count would skip one event per poll)."""
        with self._lock:
            events = [dict(e) for e in self._events if e["seq"] > after]
            return events, self.status, self._next_seq - 1

    def snapshot_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]

    # -- persistence shapes -------------------------------------------

    def to_doc(self) -> dict[str, Any]:
        """The persisted events artifact (``derived/demo/agent_runs/<id>/events.json``)."""
        with self._lock:
            return {
                "version": EVENTS_DOC_VERSION,
                "run_id": self.run_id,
                "scan_id": self.scan_id,
                "mode": self.mode,
                "status": self.status,
                "error": self.error,
                "replay": self.replay,
                "source_run_id": self.source_run_id,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "events": [dict(e) for e in self._events],
            }

    def run_ref(self, events_key: str) -> dict[str, Any]:
        """Manifest entry (protocol demo.ts DemoManifestRunRef + status extras)."""
        return {
            "run_id": self.run_id,
            "kind": self.mode,
            "created_at": self.created_at,
            "events_key": events_key,
            "replay": self.replay,
        }

    def summary(self) -> dict[str, Any]:
        """Row for ``GET …/agent/runs``."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "mode": self.mode,
                "status": self.status,
                "error": self.error,
                "replay": self.replay,
                "source_run_id": self.source_run_id,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "event_count": self._next_seq,
            }


class RunRegistry:
    """In-process registry of runs, keyed by run_id, with the one-active-run gate.

    Process-local by design: the demo broker deploys ``max_containers=1`` with a single
    founder operator (same argument as routes_docs' manifest lock). A restart empties
    it — the poll route then falls back to the persisted events artifact.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, AgentRun] = {}
        self._scene_order: dict[tuple[str, str], list[str]] = {}

    def start_run(
        self,
        user_id: str,
        scan_id: str,
        mode: str,
        replay: bool = False,
        source_run_id: Optional[str] = None,
        flusher: Optional[Callable[[AgentRun], None]] = None,
        max_wall_s: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> AgentRun:
        key = (str(user_id), str(scan_id))
        with self._lock:
            active = self._active_locked(key)
            if active is not None:
                raise ActiveRunError(active.run_id)
            run = AgentRun(
                run_id=run_id or uuid.uuid4().hex[:16],
                user_id=user_id,
                scan_id=scan_id,
                mode=mode,
                replay=replay,
                source_run_id=source_run_id,
                flusher=flusher,
                max_wall_s=max_wall_s,
            )
            self._runs[run.run_id] = run
            self._scene_order.setdefault(key, []).append(run.run_id)
            return run

    def _active_locked(self, key: tuple[str, str]) -> Optional[AgentRun]:
        for rid in reversed(self._scene_order.get(key, [])):
            run = self._runs.get(rid)
            if run is not None and run.status == STATUS_RUNNING:
                return run
        return None

    def active_run(self, user_id: str, scan_id: str) -> Optional[AgentRun]:
        with self._lock:
            return self._active_locked((str(user_id), str(scan_id)))

    def get(self, run_id: str) -> Optional[AgentRun]:
        with self._lock:
            return self._runs.get(str(run_id))

    def get_for_scene(self, user_id: str, scan_id: str, run_id: str) -> Optional[AgentRun]:
        """Owner-scoped lookup: the run must belong to THIS user's THIS scene."""
        run = self.get(run_id)
        if run is None or run.user_id != str(user_id) or run.scan_id != str(scan_id):
            return None
        return run

    def latest_for_scene(self, user_id: str, scan_id: str) -> Optional[AgentRun]:
        key = (str(user_id), str(scan_id))
        with self._lock:
            for rid in reversed(self._scene_order.get(key, [])):
                run = self._runs.get(rid)
                if run is not None:
                    return run
        return None

    def list_for_scene(self, user_id: str, scan_id: str) -> list[dict[str, Any]]:
        key = (str(user_id), str(scan_id))
        with self._lock:
            rids = list(self._scene_order.get(key, []))
        return [self._runs[r].summary() for r in rids if r in self._runs]

    def reset(self) -> None:
        """Tests only."""
        with self._lock:
            self._runs.clear()
            self._scene_order.clear()


# The process-wide registry the routes use (module-level like routes_docs' lock).
REGISTRY = RunRegistry()


# ── Persistence glue (store side of the flush) ──


def make_flusher(store: Any, user_id: str, scan_id: str) -> Callable[[AgentRun], None]:
    """Build a run flusher over ``ModalScenePersistence``: write the events doc under
    ``derived/demo/agent_runs/<run_id>/events.json`` and (once) index a rich run ref
    into the demo manifest — through routes_docs' lock + reader so this writer and the
    PUT-doc route can never interleave a manifest read-modify-write.

    NOTE (recorded delta vs W0's generic writer): the PUT-doc route indexes plain
    derived-key strings into ``manifest.agent_runs``; the protocol contract
    (``demo.ts`` DemoManifestRunRef) wants ``{run_id, kind, created_at, events_key}``
    refs. This flusher writes the rich shape. Readers here tolerate both.
    """
    # Imported here (not module top) so runlog stays importable without flask.
    from server.oreos import routes_docs

    def _flush(run: AgentRun) -> None:
        doc = run.to_doc()
        payload = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        suffix = events_key_suffix(run.run_id)
        with routes_docs._MANIFEST_LOCK:
            derived_key = store.save_derived_artifact(
                user_id, scan_id, f"demo/{suffix}", payload
            )
            if not run._indexed:
                manifest = routes_docs._read_manifest(store, user_id, scan_id)
                entries = manifest.get("agent_runs")
                if not isinstance(entries, list):
                    entries = []
                if not any(
                    isinstance(e, dict) and e.get("run_id") == run.run_id for e in entries
                ):
                    entries.append(run.run_ref(derived_key))
                manifest["agent_runs"] = entries
                docs = manifest.get("docs")
                if not isinstance(docs, list):
                    docs = []
                if derived_key not in docs:
                    docs.append(derived_key)
                manifest["docs"] = docs
                manifest["updated_at"] = doc_updated_stamp()
                store.save_derived_artifact(
                    user_id,
                    scan_id,
                    routes_docs._MANIFEST_RELATIVE_KEY,
                    json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                )
                run._indexed = True

    return _flush


def doc_updated_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_run_doc(store: Any, user_id: str, scan_id: str, run_id: str) -> Optional[dict[str, Any]]:
    """Read a persisted events doc back, or ``None``. ``run_id`` is sanitized by the
    store's per-segment cleaning; a foreign/unknown id is simply absent."""
    raw = store.get_derived_artifact(
        user_id, scan_id, f"derived/demo/{events_key_suffix(run_id)}"
    )
    if not raw:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(f"[demo.runlog] unreadable events doc for run {run_id}: {exc}")
        return None
    return doc if isinstance(doc, dict) else None


def list_persisted_runs(store: Any, user_id: str, scan_id: str) -> list[dict[str, Any]]:
    """Run refs recorded in the demo manifest (rich dicts preferred; plain string keys
    from the generic PUT-doc writer are normalized best-effort)."""
    from server.oreos import routes_docs

    with routes_docs._MANIFEST_LOCK:
        manifest = routes_docs._read_manifest(store, user_id, scan_id)
    out: list[dict[str, Any]] = []
    for entry in manifest.get("agent_runs") or []:
        if isinstance(entry, dict) and entry.get("run_id"):
            out.append(dict(entry))
        elif isinstance(entry, str):
            # derived/demo/agent_runs/<run_id>/events.json → <run_id>
            parts = entry.split("/")
            if len(parts) >= 2 and parts[-1] == "events.json":
                out.append({"run_id": parts[-2], "events_key": entry})
    return out


# ── Replay ──


def replay_view(
    doc: dict[str, Any], after: int = -1, speed: float = 1.0
) -> tuple[list[dict[str, Any]], str, int]:
    """The stored log as a replay stream (for ``GET …/events?replay=1``): events after
    the cursor, re-sequenced 0..N with fresh monotonic ``seq``, each annotated
    ``delay_ms`` (clamped original delta / speed), and the REPLAY-invariant
    ``run_meta{replay: true, recorded_at, source_run_id}`` as the FIRST event of the
    stream (seq 0). Returns ``(events, status, next)``."""
    source_events = [e for e in (doc.get("events") or []) if isinstance(e, dict)]
    stream: list[dict[str, Any]] = []

    meta_payload: dict[str, Any] = {}
    for e in source_events:
        if e.get("type") == "run_meta" and isinstance(e.get("payload"), dict):
            meta_payload = dict(e["payload"])
            break
    meta_payload.update(
        {
            "replay": True,
            "recorded_at": doc.get("created_at"),
            "source_run_id": doc.get("run_id"),
        }
    )
    stream.append(
        {
            "seq": 0,
            "ts": time.time(),
            "type": "run_meta",
            "phase": None,
            "payload": meta_payload,
            "delay_ms": 0,
        }
    )

    prev_ts: Optional[float] = None
    seq = 1
    for e in source_events:
        if e.get("type") == "run_meta":
            prev_ts = e.get("ts", prev_ts)
            continue  # replaced by the replay-marked meta above
        ts = e.get("ts")
        delta = 0.0 if prev_ts is None or ts is None else max(0.0, float(ts) - float(prev_ts))
        prev_ts = ts if ts is not None else prev_ts
        stream.append(
            {
                "seq": seq,
                "ts": time.time(),
                "type": e.get("type"),
                "phase": e.get("phase"),
                "payload": e.get("payload") or {},
                "delay_ms": int(clamp_replay_delta(delta, speed) * 1000),
            }
        )
        seq += 1

    events = [e for e in stream if e["seq"] > after]
    status = doc.get("status") if doc.get("status") in (STATUS_DONE, STATUS_ERROR) else STATUS_DONE
    # next = highest seq of the FULL stream (pass-back-verbatim cursor semantics).
    return events, str(status), stream[-1]["seq"] if stream else -1


def play_replay_into(run: AgentRun, doc: dict[str, Any], speed: float = 1.0) -> None:
    """Playback-thread body for a replay run started via ``{replay_of}``: re-emit the
    stored log into ``run`` with clamped sleeps, replay-marked run_meta first. Payloads
    are byte-identical to the recording (annotations replay deterministically)."""
    events, _status, _next = replay_view(doc, after=-1, speed=speed)
    try:
        for event in events:
            delay_ms = int(event.get("delay_ms") or 0)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            run.emit(event["type"], event.get("payload") or {}, phase=event.get("phase"))
        run.finish(STATUS_DONE)
    except Exception as exc:
        print(f"[demo.runlog] replay of {doc.get('run_id')} failed: {exc}")
        run.finish(STATUS_ERROR, error=str(exc))
