"""ApiKeyRegistry unit tests (server/api_keys.py).

First-class ``ork_`` API keys: the raw key exists only at mint time (SHA-256 at
rest), verification is cached in-process (TTL) with a throttled ``last_used_at``
KV write, and revocation is immediate locally + bounded by the cache TTL for
other processes. The registry is stdlib-only, so these tests run against a plain
dict store (the same duck-typed shape ``modal.Dict`` satisfies in production).
"""

from __future__ import annotations

import hashlib

from server.api_keys import (
    API_KEY_DEFAULT_NAME,
    API_KEY_PREFIX,
    ApiKeyRegistry,
    sha256_key_hash,
)


class CountingStore(dict):
    """Plain-dict store that counts KV traffic (cache/throttle assertions)."""

    def __init__(self):
        super().__init__()
        self.gets = 0
        self.sets = 0

    def get(self, key, default=None):
        self.gets += 1
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.sets += 1
        super().__setitem__(key, value)


def _registry(store=None, **kwargs):
    registry = ApiKeyRegistry(store if store is not None else {}, **kwargs)
    clock = {"t": 1_000_000.0}
    registry._now = lambda: clock["t"]  # deterministic clock seam
    return registry, clock


# -- mint --------------------------------------------------------------------------


def test_mint_returns_contract_shape():
    registry, _ = _registry()
    full_key, record = registry.mint("user_1", "ci bot", {"tier": "pro"})

    assert full_key.startswith(API_KEY_PREFIX)
    assert record["prefix"] == full_key[:10]
    assert len(record["key_id"]) == 16  # token_hex(8)
    int(record["key_id"], 16)  # hex
    assert record["user_id"] == "user_1"
    assert record["name"] == "ci bot"
    assert isinstance(record["created_at"], int)
    assert record["last_used_at"] is None
    assert record["revoked_at"] is None
    assert record["scopes"] == ["account"]
    assert record["tier"] == "pro"


def test_mint_stores_hash_never_the_raw_key():
    store = {}
    registry, _ = _registry(store)
    full_key, record = registry.mint("user_1", "k")

    digest = hashlib.sha256(full_key.encode("utf-8")).hexdigest()
    assert sha256_key_hash(full_key) == digest
    assert store[f"hash:{digest}"] == record["key_id"]
    assert store["index:user_1"] == [record["key_id"]]
    # The raw secret appears NOWHERE at rest — not in any key or value — and the
    # returned record carries only the 10-char display prefix.
    assert full_key not in repr(store)
    assert full_key not in repr(sorted(record.items(), key=str))


def test_mint_name_normalization():
    registry, _ = _registry()
    assert registry.mint("u", None)[1]["name"] == API_KEY_DEFAULT_NAME
    assert registry.mint("u", "   ")[1]["name"] == API_KEY_DEFAULT_NAME
    assert registry.mint("u", 42)[1]["name"] == API_KEY_DEFAULT_NAME
    assert registry.mint("u", "  spaced  ")[1]["name"] == "spaced"
    assert registry.mint("u", "x" * 100)[1]["name"] == "x" * 64


def test_mint_snapshot_copied_but_cannot_override_record_fields():
    registry, _ = _registry()
    _, record = registry.mint(
        "user_1",
        "k",
        {
            "tier": "approved",
            "scansRemaining": 3,
            # Hostile/accidental collisions must not clobber the record's own fields.
            "user_id": "someone_else",
            "revoked_at": 123,
            "scopes": ["admin"],
        },
    )
    assert record["tier"] == "approved"
    assert record["scansRemaining"] == 3
    assert record["user_id"] == "user_1"
    assert record["revoked_at"] is None
    assert record["scopes"] == ["account"]


# -- verify ------------------------------------------------------------------------


def test_verify_roundtrip():
    registry, _ = _registry()
    full_key, record = registry.mint("user_1", "k", {"tier": "pro", "scansRemaining": 5})
    seen = registry.verify(full_key)
    assert seen is not None
    assert seen["key_id"] == record["key_id"]
    assert seen["user_id"] == "user_1"
    assert seen["tier"] == "pro"
    assert seen["scansRemaining"] == 5


def test_verify_rejects_unknown_and_malformed():
    registry, _ = _registry()
    registry.mint("user_1", "k")
    assert registry.verify(API_KEY_PREFIX + "not-a-real-key") is None
    assert registry.verify("sk-something-else") is None
    assert registry.verify("") is None
    assert registry.verify(None) is None
    assert registry.verify(12345) is None


def test_verify_returns_none_for_revoked():
    registry, _ = _registry()
    full_key, record = registry.mint("user_1", "k")
    assert registry.verify(full_key) is not None
    registry.revoke("user_1", record["key_id"])
    assert registry.verify(full_key) is None


# -- revoke ------------------------------------------------------------------------


def test_revoke_idempotent_returns_original_revoked_at():
    registry, clock = _registry()
    _, record = registry.mint("user_1", "k")

    first = registry.revoke("user_1", record["key_id"])
    assert first["revoked_at"] == int(clock["t"])

    clock["t"] += 1000
    second = registry.revoke("user_1", record["key_id"])
    assert second["revoked_at"] == first["revoked_at"]  # unchanged


def test_revoke_unknown_or_cross_user_returns_none():
    registry, _ = _registry()
    _, record = registry.mint("user_1", "k")
    assert registry.revoke("user_1", "deadbeefdeadbeef") is None
    # Another user's key is indistinguishable from an unknown one.
    assert registry.revoke("user_2", record["key_id"]) is None
    # ...and the key is untouched for its owner.
    assert registry.list_for_user("user_1")[0]["revoked_at"] is None


def test_revoke_invalidates_verify_cache_immediately():
    registry, _ = _registry()
    full_key, record = registry.mint("user_1", "k")
    assert registry.verify(full_key) is not None  # populates the cache
    registry.revoke("user_1", record["key_id"])
    # Well inside the 60s cache TTL — the local entry must already be gone.
    assert registry.verify(full_key) is None


# -- list --------------------------------------------------------------------------


def test_list_newest_first_includes_revoked_and_is_user_scoped():
    registry, clock = _registry()
    _, first = registry.mint("user_1", "first")
    clock["t"] += 10
    _, second = registry.mint("user_1", "second")
    clock["t"] += 10
    _, third = registry.mint("user_1", "third")
    registry.mint("user_2", "other-user")
    registry.revoke("user_1", second["key_id"])

    records = registry.list_for_user("user_1")
    assert [r["key_id"] for r in records] == [
        third["key_id"], second["key_id"], first["key_id"],
    ]
    assert records[1]["revoked_at"] is not None  # revoked keys stay listed
    assert all(r["user_id"] == "user_1" for r in records)
    assert [r["key_id"] for r in registry.list_for_user("user_3")] == []


# -- last_used throttle + verify cache ---------------------------------------------


def test_last_used_written_at_most_once_per_interval():
    store = CountingStore()
    registry, clock = _registry(store)
    full_key, record = registry.mint("user_1", "k")
    key_kv = f"key:{record['key_id']}"
    sets_after_mint = store.sets

    registry.verify(full_key)  # first use → stamp
    assert store[key_kv]["last_used_at"] == int(clock["t"])
    assert store.sets == sets_after_mint + 1

    clock["t"] += 100  # inside the 300s throttle (cache already expired at 60s)
    registry.verify(full_key)
    assert store[key_kv]["last_used_at"] == int(clock["t"]) - 100  # NOT rewritten
    assert store.sets == sets_after_mint + 1

    clock["t"] += 201  # 301s since the stamp → write due
    registry.verify(full_key)
    assert store[key_kv]["last_used_at"] == int(clock["t"])
    assert store.sets == sets_after_mint + 2


def test_verify_cache_serves_hot_path_with_zero_kv_traffic():
    store = CountingStore()
    registry, clock = _registry(store)
    full_key, _ = registry.mint("user_1", "k")

    registry.verify(full_key)  # cache miss: KV reads + last_used write
    gets, sets = store.gets, store.sets

    clock["t"] += 10  # inside cache TTL and throttle
    assert registry.verify(full_key) is not None
    assert (store.gets, store.sets) == (gets, sets)  # pure in-process hit


def test_cache_hit_survives_store_loss_within_ttl():
    """The cache really is the source on the hot path: KV contents are not consulted."""
    store = {}
    registry, clock = _registry(store)
    full_key, _ = registry.mint("user_1", "k")
    assert registry.verify(full_key) is not None
    store.clear()  # simulate the KV being unreachable/emptied
    clock["t"] += 10
    assert registry.verify(full_key) is not None  # cached
    clock["t"] += 61  # TTL expired → KV consulted → gone
    assert registry.verify(full_key) is None


def test_cross_process_revocation_seen_after_cache_ttl():
    """A revoke written by ANOTHER process (store mutated underneath us) is honored
    once the local cache entry expires — the documented ≤60s staleness bound."""
    store = {}
    registry, clock = _registry(store)
    full_key, record = registry.mint("user_1", "k")
    assert registry.verify(full_key) is not None

    kv = dict(store[f"key:{record['key_id']}"])
    kv["revoked_at"] = int(clock["t"])
    store[f"key:{record['key_id']}"] = kv  # not via registry.revoke → no cache drop

    clock["t"] += 30
    assert registry.verify(full_key) is not None  # stale-but-bounded window
    clock["t"] += 31  # past the 60s TTL
    assert registry.verify(full_key) is None


def test_due_last_used_write_rereads_and_honors_external_revocation():
    """When the throttled last_used write IS due, verify re-reads the record first, so
    it can't clobber (un-revoke) a revocation written by another process."""
    store = {}
    registry, clock = _registry(store)
    full_key, record = registry.mint("user_1", "k")
    assert registry.verify(full_key) is not None

    kv = dict(store[f"key:{record['key_id']}"])
    kv["revoked_at"] = int(clock["t"])
    store[f"key:{record['key_id']}"] = kv  # external revoke

    clock["t"] += 301  # cache expired AND write due
    assert registry.verify(full_key) is None
    assert store[f"key:{record['key_id']}"]["revoked_at"] == kv["revoked_at"]
