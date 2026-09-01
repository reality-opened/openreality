"""Share-link access log — the behavioural measurement layer for customer discovery.

Every share token is minted once, for one prospect. Whenever a request authorised by a
share token reaches a scene-read endpoint, this module records a compact event under the
scene's derived tree (``derived/share_access/events.json``), keyed by a short hash of the
token (``access_id``) so a link can be tied to the person it was sent to without ever
storing the token itself. Share-token Q&A records the question text (capped), because
what a prospect *asks* the scene is the clearest signal of the job they actually have.

Privacy: no raw IP or user agent is stored — both are salted-hashed, and only to tell
one device from another within a single prospect's history.

Read side: :meth:`ShareAccessLog.summary` folds the events into per-prospect facts that
map onto the evidence ladder used in ``open-reality-gtm/customer-discovery-2026-08.md``:

* ``opened``       — any request at all (the link was looked at)
* ``visits``       — sessions: a new visit starts after ``VISIT_GAP_S`` of silence
* ``returned``     — a visit began ≥ ``RETURN_GAP_S`` after the first one (L5 evidence)
* ``questions``    — the Q&A questions, in order

Storage: in-process buffer per scene (safe: the broker deploys with ``max_containers=1``,
same invariant ``oreos/runlog.py`` relies on) flushed through the store's
``save_derived_artifact`` every ``FLUSH_EVERY_EVENTS`` events or ``FLUSH_EVERY_S`` seconds,
and on every *first* event of a visit so an "opened" signal is never lost to a restart.
Recording must never affect the request it observes: every public entry point swallows
its own errors.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Optional

DERIVED_KEY = "share_access/events.json"
FLUSH_EVERY_EVENTS = 10
FLUSH_EVERY_S = 5.0
VISIT_GAP_S = 30 * 60          # silence that separates two visits
RETURN_GAP_S = 24 * 3600       # a visit this long after the first counts as "returned"
QUESTION_MAX_CHARS = 500
MAX_EVENTS_PER_SCENE = 20_000  # hard cap so a scraped link can't grow the artifact unbounded
_ID_SALT = "open-reality-share-access-v1"


def access_id_for_token(token: str) -> str:
    """Stable 12-hex id for a share token — unique per mint, reveals nothing about it."""
    return hashlib.sha256((_ID_SALT + ":" + str(token)).encode("utf-8")).hexdigest()[:12]


def _short_hash(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256((_ID_SALT + "|" + str(value)).encode("utf-8")).hexdigest()[:8]


class ShareAccessLog:
    """Per-scene share-link access events, persisted as a derived artifact."""

    def __init__(self, persistence: Any, *, clock=time.time) -> None:
        self._store = persistence
        self._clock = clock
        self._lock = threading.Lock()
        # (user_id, scan_id) -> {"events": [...], "dirty": int, "last_flush": float, "loaded": bool}
        self._scenes: dict[tuple[str, str], dict[str, Any]] = {}

    # -- recording ---------------------------------------------------------------------

    def record(
        self,
        *,
        user_id: str,
        scan_id: str,
        token: str,
        token_iat: Optional[int],
        token_exp: Optional[int],
        endpoint: str,
        method: str = "GET",
        question: Optional[str] = None,
        remote_addr: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Append one event. Never raises — a logging failure must not fail the request."""
        try:
            event = {
                "ts": round(float(self._clock()), 3),
                "access_id": access_id_for_token(token),
                "iat": int(token_iat) if token_iat is not None else None,
                "exp": int(token_exp) if token_exp is not None else None,
                "endpoint": str(endpoint or ""),
                "method": str(method or "GET").upper(),
                "ip": _short_hash(remote_addr),
                "ua": _short_hash(user_agent),
            }
            if question:
                event["question"] = str(question)[:QUESTION_MAX_CHARS]
            with self._lock:
                state = self._state(user_id, scan_id)
                events = state["events"]
                visit_start = self._starts_visit(events, event)
                events.append(event)
                if len(events) > MAX_EVENTS_PER_SCENE:
                    del events[: len(events) - MAX_EVENTS_PER_SCENE]
                state["dirty"] += 1
                due = (
                    visit_start
                    or bool(question)
                    or state["dirty"] >= FLUSH_EVERY_EVENTS
                    or (event["ts"] - state["last_flush"]) >= FLUSH_EVERY_S
                )
                if due:
                    self._flush_locked(user_id, scan_id, state)
            return event
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[share_access] record failed ({scan_id}): {exc}")
            return None

    def flush_all(self) -> None:
        with self._lock:
            for (user_id, scan_id), state in self._scenes.items():
                if state["dirty"]:
                    self._flush_locked(user_id, scan_id, state)

    # -- reading -----------------------------------------------------------------------

    def events(self, user_id: str, scan_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._state(user_id, scan_id)["events"])

    def summary(self, user_id: str, scan_id: str, *, now: Optional[float] = None) -> dict[str, Any]:
        """Fold the events into per-prospect facts (see module docstring)."""
        events = self.events(user_id, scan_id)
        now_ts = float(now if now is not None else self._clock())
        by_id: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            by_id.setdefault(str(ev.get("access_id")), []).append(ev)

        prospects = []
        for access_id, evs in by_id.items():
            evs.sort(key=lambda e: float(e.get("ts", 0.0)))
            visits = _split_visits(evs)
            first_ts = float(evs[0]["ts"])
            last_ts = float(evs[-1]["ts"])
            returned_at = None
            for visit in visits[1:]:
                if float(visit[0]["ts"]) - first_ts >= RETURN_GAP_S:
                    returned_at = float(visit[0]["ts"])
                    break
            questions = [
                {"ts": float(e["ts"]), "question": e["question"]}
                for e in evs if e.get("question")
            ]
            endpoints: dict[str, int] = {}
            for e in evs:
                endpoints[e.get("endpoint", "")] = endpoints.get(e.get("endpoint", ""), 0) + 1
            prospects.append(
                {
                    "access_id": access_id,
                    "iat": evs[0].get("iat"),
                    "exp": evs[0].get("exp"),
                    "opened": True,
                    "requests": len(evs),
                    "visits": len(visits),
                    "first_seen": first_ts,
                    "last_seen": last_ts,
                    "returned": returned_at is not None,
                    "returned_at": returned_at,
                    "devices": len({(e.get("ip"), e.get("ua")) for e in evs}),
                    "questions": questions,
                    "endpoints": endpoints,
                    "visit_starts": [float(v[0]["ts"]) for v in visits],
                }
            )
        prospects.sort(key=lambda p: p["first_seen"])
        return {
            "scan_id": str(scan_id),
            "generated_at": now_ts,
            "definitions": {
                "visit": f"a new visit starts after {VISIT_GAP_S} s without requests",
                "returned": f"a visit beginning >= {RETURN_GAP_S} s after first_seen",
                "access_id": "sha256(token)[:12] — unique per minted share link; the token itself is never stored",
            },
            "prospects": prospects,
            "event_count": len(events),
        }

    # -- internals ---------------------------------------------------------------------

    def _state(self, user_id: str, scan_id: str) -> dict[str, Any]:
        key = (str(user_id), str(scan_id))
        state = self._scenes.get(key)
        if state is None:
            state = {"events": self._load(*key), "dirty": 0, "last_flush": float(self._clock()), "loaded": True}
            self._scenes[key] = state
        return state

    def _load(self, user_id: str, scan_id: str) -> list[dict[str, Any]]:
        try:
            raw = self._store.get_derived_artifact(user_id, scan_id, "derived/" + DERIVED_KEY)
            if not raw:
                return []
            data = json.loads(raw.decode("utf-8"))
            events = data.get("events") if isinstance(data, dict) else data
            return [e for e in (events or []) if isinstance(e, dict)]
        except Exception as exc:
            print(f"[share_access] load failed ({scan_id}): {exc}")
            return []

    def _flush_locked(self, user_id: str, scan_id: str, state: dict[str, Any]) -> None:
        try:
            payload = json.dumps({"version": 1, "events": state["events"]}, separators=(",", ":")).encode("utf-8")
            self._store.save_derived_artifact(user_id, scan_id, DERIVED_KEY, payload)
            state["dirty"] = 0
            state["last_flush"] = float(self._clock())
        except Exception as exc:
            print(f"[share_access] flush failed ({scan_id}): {exc}")

    @staticmethod
    def _starts_visit(events: list[dict[str, Any]], event: dict[str, Any]) -> bool:
        prior = [e for e in events if e.get("access_id") == event["access_id"]]
        if not prior:
            return True
        return float(event["ts"]) - float(prior[-1].get("ts", 0.0)) >= VISIT_GAP_S


def _split_visits(evs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    visits: list[list[dict[str, Any]]] = []
    for ev in evs:
        if visits and float(ev["ts"]) - float(visits[-1][-1]["ts"]) < VISIT_GAP_S:
            visits[-1].append(ev)
        else:
            visits.append([ev])
    return visits
