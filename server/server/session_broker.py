"""Dedicated GPU session allocation for the Modal broker endpoint."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


class SessionCapacityError(RuntimeError):
    """Raised when all configured dedicated GPU worker slots are in use."""


class SessionWorkerError(RuntimeError):
    """Raised when a spawned GPU worker fails before exposing an endpoint."""


class SessionBroker:
    """Allocate at most one dedicated worker endpoint per authenticated user."""

    TERMINAL_STATES = frozenset({"failed", "stopped", "expired"})

    def __init__(
        self,
        registry: Any,
        worker_status: Any,
        spawn_worker: Callable[[str, str], Any],
        *,
        max_sessions: int,
        session_lifetime_sec: int,
        endpoint_wait_sec: float = 20.0,
        poll_interval_sec: float = 0.2,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self._registry = registry
        self._worker_status = worker_status
        self._spawn_worker = spawn_worker
        self._max_sessions = max_sessions
        self._session_lifetime_sec = session_lifetime_sec
        self._endpoint_wait_sec = endpoint_wait_sec
        self._poll_interval_sec = poll_interval_sec
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()

    def allocate(self, user_id: str) -> dict[str, Any]:
        """Return the user's current worker, or reserve and spawn one."""
        normalized_user_id = str(user_id).strip()
        if not normalized_user_id:
            raise ValueError("user_id is required")

        with self._lock:
            self._prune_inactive_locked()
            record = self._registry.get(normalized_user_id)
            if not self._is_live_record(record):
                active_count = sum(
                    1
                    for _, active_record in self._registry.items()
                    if self._is_live_record(active_record)
                )
                if active_count >= self._max_sessions:
                    raise SessionCapacityError("at_capacity")

                session_id = uuid.uuid4().hex
                expires_at = self._now() + self._session_lifetime_sec
                record = {
                    "session_id": session_id,
                    "user_id": normalized_user_id,
                    "status": "starting",
                    "expires_at": expires_at,
                }
                self._registry[normalized_user_id] = record
                self._worker_status[session_id] = {
                    "session_id": session_id,
                    "status": "starting",
                    "expires_at": expires_at,
                }
                try:
                    self._spawn_worker(session_id, normalized_user_id)
                except Exception as exc:
                    self._remove_record_locked(normalized_user_id, session_id)
                    raise SessionWorkerError("worker_spawn_failed") from exc

        return self._wait_for_endpoint(normalized_user_id, str(record["session_id"]))

    def status(self, user_id: str) -> dict[str, Any] | None:
        """Return the user's live allocation, dropping stopped or stale entries."""
        normalized_user_id = str(user_id).strip()
        with self._lock:
            self._prune_inactive_locked()
            record = self._registry.get(normalized_user_id)
            if not self._is_live_record(record):
                return None
            return self._payload_locked(normalized_user_id, record)

    def _wait_for_endpoint(self, user_id: str, session_id: str) -> dict[str, Any]:
        deadline = self._now() + self._endpoint_wait_sec
        while True:
            with self._lock:
                record = self._registry.get(user_id)
                if not self._is_live_record(record):
                    raise SessionWorkerError("worker_unavailable")
                payload = self._payload_locked(user_id, record)
                if payload.get("url") or payload["status"] in self.TERMINAL_STATES:
                    if payload["status"] in self.TERMINAL_STATES:
                        self._remove_record_locked(user_id, session_id)
                        raise SessionWorkerError("worker_unavailable")
                    return payload
            if self._now() >= deadline:
                return payload
            self._sleep(self._poll_interval_sec)

    def _payload_locked(self, user_id: str, record: dict[str, Any]) -> dict[str, Any]:
        session_id = str(record["session_id"])
        status = self._worker_status.get(session_id)
        status = status if isinstance(status, dict) else {}
        state = str(status.get("status") or record.get("status") or "starting")
        url = status.get("url") or record.get("url")
        if url or state != record.get("status"):
            updated = dict(record)
            updated["status"] = state
            if url:
                updated["url"] = url
            self._registry[user_id] = updated
            record = updated
        return {
            "session_id": session_id,
            "url": url,
            "status": state,
            "expires_at": record.get("expires_at"),
        }

    def _is_live_record(self, record: Any) -> bool:
        if not isinstance(record, dict) or not record.get("session_id"):
            return False
        try:
            if float(record.get("expires_at", 0)) <= self._now():
                return False
        except (TypeError, ValueError):
            return False
        status = self._worker_status.get(str(record["session_id"]))
        if isinstance(status, dict) and status.get("status") in self.TERMINAL_STATES:
            return False
        return True

    def _prune_inactive_locked(self) -> None:
        for user_id, record in list(self._registry.items()):
            if not self._is_live_record(record):
                session_id = record.get("session_id") if isinstance(record, dict) else None
                self._remove_record_locked(str(user_id), str(session_id or ""))

    def _remove_record_locked(self, user_id: str, session_id: str) -> None:
        current = self._registry.get(user_id)
        if isinstance(current, dict) and str(current.get("session_id")) != session_id:
            return
        self._registry.pop(user_id, None)
        if session_id:
            self._worker_status.pop(session_id, None)
