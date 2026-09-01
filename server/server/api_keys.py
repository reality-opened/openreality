"""First-class API keys for the OpenReality broker (Bearer ``ork_...``).

A key is a durable, revocable secret for programmatic ``/api/*`` access (MCP
clients, CI) — unlike the broker session JWT it survives restarts and can be
revoked instantly, and unlike a Clerk JWT it never expires on its own. The raw
key string is returned exactly once at mint; only its SHA-256 hex lives at rest,
so neither the KV store nor a listing can ever leak the secret.

``ApiKeyRegistry`` follows the repo's injected-store duck-typing convention
(see ``server/scene_report/store.py``): the ``store`` only needs dict-like
``get``/``__setitem__`` (both ``modal.Dict`` and a plain ``dict`` satisfy this)
and is wired by ``modal_streaming.py`` in production (a shared ``modal.Dict``
across the broker + GPU workers) or a plain ``dict`` in local runs/tests. The
module is deliberately free of ``import modal`` and of any ``server.app``
import, so it stays unit-testable with stdlib only.

KV layout (all values JSON-able):

- ``hash:<sha256hex>`` → ``key_id`` (constant-size lookup of a presented key)
- ``key:<key_id>``     → record dict (see ``mint``; never contains the raw key)
- ``index:<user_id>``  → list of the user's key_ids, newest first

Hot-path behavior: ``verify`` keeps a small in-process TTL cache keyed by the
key's hash so per-request auth doesn't hit the (remote, in production) KV, and
stamps ``last_used_at`` with at most one KV write per
``LAST_USED_WRITE_INTERVAL_S``. ``revoke`` drops the local cache entry
immediately; other *processes* observe a revocation once their cache entry
expires (≤ ``VERIFY_CACHE_TTL_S`` seconds).
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from typing import Any, Optional

# Wire contract v1 (platform contract — the TS protocol package + MCP client are
# built against exactly these): key = "ork_" + token_urlsafe(32); display prefix
# is the first 10 chars of the full key; key_id is token_hex(8).
API_KEY_PREFIX = "ork_"
API_KEY_DISPLAY_PREFIX_CHARS = 10
API_KEY_NAME_MAX_CHARS = 64
API_KEY_DEFAULT_NAME = "api-key"
# v1 keys carry the whole account's authority (same scope as a session token).
# A list so a future version can mint narrower keys without a layout change.
API_KEY_SCOPES = ["account"]

# Throttle on the last_used_at KV write (per key), and TTL on the in-process
# verify cache. The cache TTL is also the cross-process revocation lag bound.
LAST_USED_WRITE_INTERVAL_S = 300.0
VERIFY_CACHE_TTL_S = 60.0


def sha256_key_hash(full_key: str) -> str:
    """The at-rest identity of a key: SHA-256 hex of the full ``ork_...`` string."""
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def normalize_key_name(raw: Any) -> str:
    """Strip + cap a caller-supplied key name; empty/non-string → the default."""
    name = raw.strip() if isinstance(raw, str) else ""
    return name[:API_KEY_NAME_MAX_CHARS] or API_KEY_DEFAULT_NAME


class ApiKeyRegistry:
    """Mint/verify/list/revoke first-class API keys over an injected KV store.

    Thread-safe: one process-wide lock guards both the KV writes and the local
    verify cache, so an in-process revoke can never interleave with a verify's
    throttled ``last_used_at`` write-back and resurrect a revoked record.
    """

    def __init__(
        self,
        store: Any,
        *,
        last_used_write_interval_s: float = LAST_USED_WRITE_INTERVAL_S,
        verify_cache_ttl_s: float = VERIFY_CACHE_TTL_S,
    ):
        self._store = store
        self._lock = threading.Lock()
        self._last_used_write_interval_s = float(last_used_write_interval_s)
        self._verify_cache_ttl_s = float(verify_cache_ttl_s)
        # sha256hex -> (record copy, cached_at). Only successful verifies are
        # cached (never misses/revocations), so garbage tokens can't grow it.
        self._verify_cache: dict[str, tuple[dict[str, Any], float]] = {}
        # Clock seam for tests (throttle/TTL behavior without sleeping).
        self._now = time.time

    # -- KV keys -------------------------------------------------------------

    @staticmethod
    def _hash_key(digest: str) -> str:
        return f"hash:{digest}"

    @staticmethod
    def _record_key(key_id: str) -> str:
        return f"key:{key_id}"

    @staticmethod
    def _index_key(user_id: str) -> str:
        return f"index:{user_id}"

    # -- operations ----------------------------------------------------------

    def mint(
        self,
        user_id: str,
        name: Any = None,
        claims_snapshot: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create a key for ``user_id``. Returns ``(full_key, record)`` — the ONLY
        time the full key exists outside the caller's hands; at rest we keep its
        SHA-256 only. ``claims_snapshot`` carries identity-adjacent claims copied
        onto the record (the same ``tier`` + scans-quota fields the broker session
        token snapshots); it can never override the record's own fields."""
        user_id = str(user_id).strip()
        full_key = API_KEY_PREFIX + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        digest = sha256_key_hash(full_key)
        with self._lock:
            now = int(self._now())
            record: dict[str, Any] = {
                "key_id": key_id,
                "user_id": user_id,
                "name": normalize_key_name(name),
                "prefix": full_key[:API_KEY_DISPLAY_PREFIX_CHARS],
                "created_at": now,
                "last_used_at": None,
                "revoked_at": None,
                "scopes": list(API_KEY_SCOPES),
            }
            for claim, value in dict(claims_snapshot or {}).items():
                if claim not in record:
                    record[claim] = value
            # Record before hash mapping, so a concurrent verify can never find
            # the hash and miss the record.
            self._store[self._record_key(key_id)] = dict(record)
            self._store[self._hash_key(digest)] = key_id
            existing = self._store.get(self._index_key(user_id))
            ids = [str(i) for i in existing] if isinstance(existing, list) else []
            self._store[self._index_key(user_id)] = [key_id] + [
                i for i in ids if i != key_id
            ]
        return full_key, dict(record)

    def verify(self, full_key: Any) -> Optional[dict[str, Any]]:
        """Resolve a presented key to its record, or ``None`` for anything that
        must not authenticate (wrong shape, unknown, revoked). Serves hot paths
        from the in-process cache (TTL ``verify_cache_ttl_s``) and stamps
        ``last_used_at`` with at most one KV write per
        ``last_used_write_interval_s``; when that write is due, the record is
        re-read from the KV first so a revocation written by another process is
        honored rather than clobbered."""
        if not isinstance(full_key, str) or not full_key.startswith(API_KEY_PREFIX):
            return None
        digest = sha256_key_hash(full_key)
        with self._lock:
            now = self._now()
            cached = self._verify_cache.get(digest)
            if cached is not None and now - cached[1] < self._verify_cache_ttl_s:
                record, cached_at = dict(cached[0]), cached[1]
            else:
                key_id = self._store.get(self._hash_key(digest))
                raw = (
                    self._store.get(self._record_key(str(key_id))) if key_id else None
                )
                if not isinstance(raw, dict):
                    self._verify_cache.pop(digest, None)
                    return None
                record, cached_at = dict(raw), now
            if record.get("revoked_at") is not None:
                self._verify_cache.pop(digest, None)
                return None

            last = record.get("last_used_at")
            if last is None or now - float(last) >= self._last_used_write_interval_s:
                # Write due — re-read first (cross-process revocation / freshness).
                fresh = self._store.get(self._record_key(str(record.get("key_id"))))
                if isinstance(fresh, dict):
                    record, cached_at = dict(fresh), now
                if record.get("revoked_at") is not None:
                    self._verify_cache.pop(digest, None)
                    return None
                record["last_used_at"] = int(now)
                self._store[self._record_key(str(record["key_id"]))] = dict(record)

            self._verify_cache[digest] = (dict(record), cached_at)
            return dict(record)

    def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """The user's key records, newest first, revoked included. Records only —
        raw secrets are unrecoverable by design."""
        user_id = str(user_id).strip()
        with self._lock:
            ids = self._store.get(self._index_key(user_id))
            records: list[dict[str, Any]] = []
            for key_id in ids if isinstance(ids, list) else []:
                raw = self._store.get(self._record_key(str(key_id)))
                if isinstance(raw, dict) and str(raw.get("user_id", "")) == user_id:
                    records.append(dict(raw))
            return records

    def revoke(self, user_id: str, key_id: str) -> Optional[dict[str, Any]]:
        """Revoke ``key_id`` if it belongs to ``user_id``. Idempotent (a repeat
        revoke returns the record with the ORIGINAL ``revoked_at``); returns
        ``None`` for an unknown key_id or another user's key — indistinguishable
        on purpose so key_ids can't be probed across accounts. Drops the local
        verify-cache entry immediately."""
        user_id = str(user_id).strip()
        key_id = str(key_id).strip()
        with self._lock:
            raw = self._store.get(self._record_key(key_id))
            if not isinstance(raw, dict) or str(raw.get("user_id", "")) != user_id:
                return None
            record = dict(raw)
            if record.get("revoked_at") is None:
                record["revoked_at"] = int(self._now())
                self._store[self._record_key(key_id)] = dict(record)
            for digest, (cached_record, _cached_at) in list(self._verify_cache.items()):
                if cached_record.get("key_id") == key_id:
                    self._verify_cache.pop(digest, None)
            return dict(record)
