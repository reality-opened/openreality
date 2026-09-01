"""Single-occupancy session gate (cross-user isolation fix).

The streaming container holds one global SLAM map and Modal runs at most one
container, so two *different* users must not hold a session concurrently or they
share one reconstruction. Same-user sids (laptop viewer + phone sender) are OK.
"""

import asyncio
import time
import types

from conftest import load_app_module


def _fresh_app(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    app_mod._active_session_store = None
    app_mod._local_active_sessions.clear()
    return app_mod


def test_entry_has_live_sid_handles_both_shapes(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    now = time.time()
    assert app_mod._entry_has_live_sid({"sids": {"s": now + 100}}, now) is True
    assert app_mod._entry_has_live_sid({"sids": {"s": now - 100}}, now) is False
    # Legacy single-sid entry shape.
    assert app_mod._entry_has_live_sid({"sid": "s", "expires_at": now + 100}, now) is True
    assert app_mod._entry_has_live_sid({"sid": "s", "expires_at": now - 100}, now) is False
    # Garbage timestamps never count as live.
    assert app_mod._entry_has_live_sid({"sids": {"s": "nope"}}, now) is False


def test_same_user_second_sid_allowed(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    assert app_mod._claim_user_session_sync("user-1", "sid-a") is True
    assert app_mod._claim_user_session_sync("user-1", "sid-b") is True
    store = app_mod._get_active_session_store()
    assert set(store["user-1"]["sids"]) == {"sid-a", "sid-b"}


def test_different_user_rejected_while_active(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    assert app_mod._claim_user_session_sync("user-1", "sid-a") is True
    assert app_mod._claim_user_session_sync("user-2", "sid-b") is False
    store = app_mod._get_active_session_store()
    assert "user-2" not in store  # rejected claim leaves no trace


def test_slot_frees_after_release(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    assert app_mod._claim_user_session_sync("user-1", "sid-a") is True
    app_mod._release_user_session_sync("user-1", "sid-a")
    assert app_mod._claim_user_session_sync("user-2", "sid-b") is True


def test_expired_entry_does_not_block_new_user(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    store = app_mod._get_active_session_store()
    stale = time.time() - 1.0
    store["user-1"] = {"sids": {"sid-old": stale}, "expires_at": stale}
    assert app_mod._claim_user_session_sync("user-2", "sid-b") is True


def test_handle_connect_rejects_when_slot_busy(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    app_mod.slam_processor = types.SimpleNamespace(event_loop=None, result_ready_event=None)
    app_mod._verify_socket_auth = lambda auth, environ: {"sub": "user-2"}

    async def _claim_busy(user_id, sid):
        return False

    app_mod._claim_user_session = _claim_busy

    async def run():
        try:
            await app_mod.handle_connect("sid-x", {"token": "t"}, {})
        except app_mod.SocketIOConnectionRefused as exc:
            return exc
        return None

    exc = asyncio.run(run())
    assert exc is not None


def test_dedicated_worker_rejects_user_other_than_assigned_owner(monkeypatch):
    app_mod = _fresh_app(monkeypatch)
    app_mod.configure_worker_runtime("user-1")
    app_mod.slam_processor = types.SimpleNamespace(event_loop=None, result_ready_event=None)
    app_mod._verify_socket_auth = lambda auth, environ: {"sub": "user-2"}

    async def run():
        try:
            await app_mod.handle_connect("sid-x", {"token": "t"}, {})
        except app_mod.SocketIOConnectionRefused as exc:
            return exc
        return None

    assert asyncio.run(run()) is not None
