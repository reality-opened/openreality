"""API-key auth + /api/keys route tests (wire contract v1).

Follows the repo's app-test convention (see tests/test_session_token.py /
test_share_token.py): ``server.app`` is loaded via ``conftest.load_app_module``
with flask/jwt faked, route handlers are invoked directly against a fake
``request``, and the faked ``jsonify`` returns ``{"args": (payload,), ...}``.
The registry itself is the REAL ``server.api_keys.ApiKeyRegistry`` over a plain
dict (the production duck-type), so these tests cover the full bearer path:
``require_modal_auth`` → ``_verify_http_token`` → ``_verify_any_token`` →
``_verify_api_key`` → registry, plus the three routes.
"""

from __future__ import annotations

import types

import pytest

from conftest import load_app_module
from server.api_keys import ApiKeyRegistry


def _fake_request(
    monkeypatch,
    app_mod,
    *,
    method="GET",
    path="/api/keys",
    bearer=None,
    claims=None,
    json_body=None,
    endpoint="create_api_key_route",
):
    """Install a minimal fake ``request`` for the before_request cascade + routes."""
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    environ = {}
    if claims is not None:
        environ[app_mod.AUTH_CLAIMS_ENV_KEY] = claims
    req = types.SimpleNamespace(
        method=method,
        path=path,
        headers=headers,
        args={},
        cookies={},
        url_rule=types.SimpleNamespace(endpoint=endpoint),
        view_args={},
        environ=environ,
        get_json=lambda silent=False: json_body,
    )
    monkeypatch.setattr(app_mod, "request", req)
    return req


def _payload(resp):
    body = resp[0] if isinstance(resp, tuple) else resp
    return body["args"][0]  # conftest's fake jsonify shape


def _status(resp):
    return resp[1] if isinstance(resp, tuple) else 200


def _registry(app_mod):
    registry = ApiKeyRegistry({})
    app_mod.configure_api_key_registry(registry)
    return registry


# -- route registration ------------------------------------------------------------


def test_key_routes_registered_under_api(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    routes = {(args[0], tuple(kwargs.get("methods", ())))
              for args, kwargs, _fn in app_mod.app.routes}
    assert ("/api/keys", ("POST",)) in routes
    assert ("/api/keys", ("GET",)) in routes
    assert ("/api/keys/<key_id>", ("DELETE",)) in routes
    # /api/* is inside the auth'd prefix, so require_modal_auth guards all three.
    assert app_mod._requires_http_auth("/api/keys")


# -- _verify_api_key / _verify_any_token dispatch ----------------------------------


def test_verify_any_token_routes_ork_prefix_before_jwt_parsing(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, _ = registry.mint("user_1", "k", {"tier": "pro", "scansRemaining": 4})

    def _boom(_token):
        raise AssertionError("ork_ bearer must never reach JWT header parsing")

    monkeypatch.setattr(app_mod.jwt, "get_unverified_header", _boom)
    claims = app_mod._verify_any_token(full_key)
    assert claims == {
        "iss": app_mod.API_KEY_ISSUER,
        "sub": "user_1",
        "tier": "pro",
        app_mod.SCANS_CLAIM: 4,
        "auth_kind": "api_key",
        "key_id": registry.list_for_user("user_1")[0]["key_id"],
    }


def test_verify_api_key_without_registry_is_invalid_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)  # registry never configured
    with pytest.raises(app_mod.AuthError) as excinfo:
        app_mod._verify_api_key("ork_whatever")
    assert str(excinfo.value) == "invalid_token"
    assert excinfo.value.status_code == 401
    # Same via the dispatcher (the ork_ prefix must not fall through to Clerk).
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_any_token("ork_whatever")


def test_unknown_and_revoked_keys_are_invalid_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, record = registry.mint("user_1", "k")
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_api_key("ork_unknown-key")
    registry.revoke("user_1", record["key_id"])
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_api_key(full_key)


# -- end-to-end bearer auth through require_modal_auth -----------------------------


def test_ork_bearer_authenticates_an_api_route_end_to_end(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, record = registry.mint("user_1", "mcp", {"tier": "approved"})

    _fake_request(monkeypatch, app_mod, method="GET", path="/api/scenes",
                  bearer=full_key, endpoint="list_scenes_route")
    assert app_mod.require_modal_auth() is None  # authorized

    claims = app_mod.request.environ[app_mod.AUTH_CLAIMS_ENV_KEY]
    assert claims["sub"] == "user_1"
    assert claims["auth_kind"] == "api_key"
    assert claims["key_id"] == record["key_id"]

    # ...and the route it was aimed at serves under those claims.
    assert _payload(app_mod.list_scenes_route()) == {"scenes": []}


def test_revoked_ork_bearer_is_401_invalid_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, record = registry.mint("user_1", "k", {"tier": "approved"})
    registry.revoke("user_1", record["key_id"])

    _fake_request(monkeypatch, app_mod, method="GET", path="/api/scenes",
                  bearer=full_key, endpoint="list_scenes_route")
    resp = app_mod.require_modal_auth()
    assert _status(resp) == 401
    assert _payload(resp) == {"error": "invalid_token"}


def test_ork_bearer_without_registry_is_401(monkeypatch):
    app_mod = load_app_module(monkeypatch)  # registry stays None
    _fake_request(monkeypatch, app_mod, method="GET", path="/api/scenes",
                  bearer="ork_some-key", endpoint="list_scenes_route")
    resp = app_mod.require_modal_auth()
    assert _status(resp) == 401
    assert _payload(resp) == {"error": "invalid_token"}


# -- POST /api/keys (mint) ---------------------------------------------------------


def test_mint_with_session_token_bearer_end_to_end(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    token, _ = app_mod._issue_session_token(
        {"sub": "user_1", "tier": "pro", "scansRemaining": 5}
    )
    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  bearer=token, json_body={"name": "  ci bot  "})
    assert app_mod.require_modal_auth() is None  # session bearer accepted

    resp = app_mod.create_api_key_route()
    assert _status(resp) == 200
    body = _payload(resp)
    assert set(body) == {"key", "key_id", "name", "prefix", "created_at"}
    assert body["key"].startswith("ork_")
    assert body["prefix"] == body["key"][:10]
    assert body["name"] == "ci bot"  # stripped
    assert isinstance(body["created_at"], int)

    # The minted key snapshots identity + tier only — NOT scansRemaining. A quota
    # snapshot inside a durable credential is a standing grant (the defect the billing
    # line removed from session tokens); balance is read live from the ledger, and the
    # verify path's missing claim falls back to the advisory DEFAULT_SCANS gate.
    record = registry.verify(body["key"])
    assert record["user_id"] == "user_1"
    assert record["tier"] == "pro"
    assert record.get(app_mod.SCANS_CLAIM) is None


def test_mint_defaults_name_when_body_absent(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _registry(app_mod)
    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  claims={"sub": "user_1", "tier": "pro"}, json_body=None)
    body = _payload(app_mod.create_api_key_route())
    assert body["name"] == "api-key"

    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  claims={"sub": "user_1"}, json_body={"name": "x" * 100})
    assert _payload(app_mod.create_api_key_route())["name"] == "x" * 64


def test_api_key_cannot_mint_api_key(monkeypatch):
    """A leaked key must not self-propagate: an ork_ bearer gets 403 on the mint."""
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, _ = registry.mint("user_1", "k", {"tier": "approved"})

    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  bearer=full_key, json_body={"name": "sneaky"})
    assert app_mod.require_modal_auth() is None  # the key IS valid auth...
    resp = app_mod.create_api_key_route()  # ...but may not mint
    assert _status(resp) == 403
    assert _payload(resp) == {"error": "api_key_cannot_mint"}
    assert len(registry.list_for_user("user_1")) == 1  # nothing minted


def test_mint_without_registry_is_unavailable(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  claims={"sub": "user_1"}, json_body={})
    resp = app_mod.create_api_key_route()
    assert _status(resp) == 503
    assert _payload(resp) == {"error": "api_keys_unavailable"}


def test_mint_requires_identity(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _registry(app_mod)
    _fake_request(monkeypatch, app_mod, method="POST", path="/api/keys",
                  claims={"sub": "  "}, json_body={})
    resp = app_mod.create_api_key_route()
    assert _status(resp) == 401


# -- GET /api/keys (list) ----------------------------------------------------------


def test_list_masks_secrets_newest_first_and_is_caller_scoped(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    clock = {"t": 1_000_000.0}
    registry._now = lambda: clock["t"]
    first_key, first = registry.mint("user_1", "first", {"tier": "pro"})
    clock["t"] += 10
    second_key, second = registry.mint("user_1", "second", {"tier": "pro"})
    registry.mint("user_2", "not-yours")
    registry.revoke("user_1", first["key_id"])

    _fake_request(monkeypatch, app_mod, method="GET", path="/api/keys",
                  claims={"sub": "user_1"})
    body = _payload(app_mod.list_api_keys_route())
    assert [k["key_id"] for k in body["keys"]] == [second["key_id"], first["key_id"]]
    for entry in body["keys"]:
        # Exactly the contract's fields — no raw key, no hash, no tier/scopes/user_id.
        assert set(entry) == {
            "key_id", "name", "prefix", "created_at", "last_used_at", "revoked_at",
        }
    assert body["keys"][0]["revoked_at"] is None
    assert body["keys"][1]["revoked_at"] is not None  # revoked keys stay listed
    assert first_key not in repr(body) and second_key not in repr(body)


def test_list_without_registry_serves_empty(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _fake_request(monkeypatch, app_mod, method="GET", path="/api/keys",
                  claims={"sub": "user_1"})
    assert _payload(app_mod.list_api_keys_route()) == {"keys": []}


# -- DELETE /api/keys/<key_id> (revoke) --------------------------------------------


def test_delete_revokes_and_is_idempotent(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    full_key, record = registry.mint("user_1", "k", {"tier": "approved"})

    _fake_request(monkeypatch, app_mod, method="DELETE",
                  path=f"/api/keys/{record['key_id']}", claims={"sub": "user_1"})
    resp = app_mod.revoke_api_key_route(record["key_id"])
    assert _status(resp) == 200
    body = _payload(resp)
    assert body["key_id"] == record["key_id"]
    assert isinstance(body["revoked_at"], int)
    assert registry.verify(full_key) is None  # revocation bites immediately

    again = _payload(app_mod.revoke_api_key_route(record["key_id"]))
    assert again["revoked_at"] == body["revoked_at"]  # idempotent, original timestamp


def test_delete_unknown_or_cross_user_is_404(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    registry = _registry(app_mod)
    _, theirs = registry.mint("user_2", "theirs")

    _fake_request(monkeypatch, app_mod, method="DELETE",
                  path="/api/keys/x", claims={"sub": "user_1"})
    resp = app_mod.revoke_api_key_route("deadbeefdeadbeef")
    assert _status(resp) == 404
    assert _payload(resp) == {"error": "not_found"}

    # Another user's key_id: same 404, and their key survives untouched.
    resp = app_mod.revoke_api_key_route(theirs["key_id"])
    assert _status(resp) == 404
    assert registry.list_for_user("user_2")[0]["revoked_at"] is None
