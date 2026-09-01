"""Read-only scene share-token tests (embed delivery).

A share token is a single-scan, read-only capability (no Clerk login) used to embed a
finished scan in a third party's dashboard. It is an HS256 JWT signed with the same secret
as the durable session token but under a distinct issuer + a ``scope``/``scan_id``, so it
can never be confused with a session token and is authorized only for the scene-read
endpoints of its one scan. These tests exercise mint/verify and the before_request scope
enforcement (right scan passes; wrong scan, the list, DELETE, and the mint route are all
rejected; expired rejected). ``jwt`` is faked by conftest (``load_app_module``) — this
covers the wrapper logic, not real crypto. See platform/contracts/embed-delivery.md.
"""

from __future__ import annotations

import time
import types

import pytest

from conftest import load_app_module


def _fake_request(monkeypatch, app_mod, *, endpoint, scan_id=None, bearer=None, query=None):
    """Install a minimal fake ``request`` for the before_request helpers + silence the
    worker-activity side effect."""
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    req = types.SimpleNamespace(
        headers=headers,
        args=dict(query or {}),
        cookies={},
        url_rule=types.SimpleNamespace(endpoint=endpoint),
        view_args={"scan_id": scan_id} if scan_id is not None else {},
        environ={},
    )
    monkeypatch.setattr(app_mod, "request", req)
    monkeypatch.setattr(app_mod, "_note_worker_activity", lambda: None)
    return req


# -- mint / verify -----------------------------------------------------------------

def test_issue_and_verify_share_token_roundtrip(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, expires_at = app_mod._issue_share_token("scan_1", "owner_42")
    assert expires_at > time.time()

    claims = app_mod._verify_share_token(token)
    assert claims["scan_id"] == "scan_1"
    assert claims["sub"] == "owner_42"
    assert claims["scope"] == app_mod.SHARE_TOKEN_SCOPE
    assert claims["iss"] == app_mod.SHARE_TOKEN_ISSUER


def test_share_token_default_ttl_and_override(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _, exp_default = app_mod._issue_share_token("s", "o")
    assert exp_default >= time.time() + app_mod.SHARE_TOKEN_TTL_SEC_DEFAULT - 5
    _, exp_short = app_mod._issue_share_token("s", "o", ttl_seconds=60)
    assert exp_short <= time.time() + 61


def test_verify_share_token_rejects_session_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A durable session token is HS256 with the same secret but a different issuer —
    # the share verifier must reject it (issuer mismatch), so the two never cross over.
    session_token, _ = app_mod._issue_session_token({"sub": "user_1"})
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_share_token(session_token)


def test_expired_share_token_rejected(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    expired = app_mod.jwt.encode(
        {
            "iss": app_mod.SHARE_TOKEN_ISSUER,
            "scope": app_mod.SHARE_TOKEN_SCOPE,
            "scan_id": "scan_x",
            "sub": "owner",
            "iat": 0,
            "exp": int(time.time()) - 10,
        },
        app_mod.SESSION_TOKEN_SECRET,
        algorithm=app_mod.SESSION_TOKEN_ALG,
    )
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_share_token(expired)


def test_verify_share_token_rejects_wrong_scope(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    bad = app_mod.jwt.encode(
        {
            "iss": app_mod.SHARE_TOKEN_ISSUER,
            "scope": "scene:write",  # not the read scope
            "scan_id": "scan_x",
            "sub": "owner",
            "iat": 0,
            "exp": int(time.time()) + 60,
        },
        app_mod.SESSION_TOKEN_SECRET,
        algorithm=app_mod.SESSION_TOKEN_ALG,
    )
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_share_token(bad)


# -- before_request scope enforcement ----------------------------------------------

def test_share_token_authorizes_matching_scan_read(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_ok", "owner_9")
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route",
                  scan_id="scan_ok", bearer=token)

    assert app_mod._try_share_token_auth() is True
    # carries the OWNER's id so user-scoped store lookups resolve, flagged read-only
    claims = app_mod.request.environ[app_mod.AUTH_CLAIMS_ENV_KEY]
    assert claims["sub"] == "owner_9"
    assert claims["share"] is True
    assert claims["scan_id"] == "scan_ok"


def test_share_token_via_query_param(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_q", "owner_q")
    # img/blob/.ply GETs can't set a header — the token rides ?share_token=
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_splat_ply_route",
                  scan_id="scan_q", query={"share_token": token})
    assert app_mod._try_share_token_auth() is True


def test_share_token_rejects_wrong_scan(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_a", "owner")
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route",
                  scan_id="scan_b", bearer=token)  # token is for scan_a
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 403


@pytest.mark.parametrize("endpoint", [
    "list_scenes_route",        # the scenes list
    "delete_scene_route",       # DELETE
    "share_scene_route",        # the mint route itself (no token re-mints)
    "demo.post_scene_lod_route",  # the LOD build — billable compute, owner-only
    "share_access_route",       # the owner's per-link access report
])
def test_share_token_rejects_disallowed_endpoints(monkeypatch, endpoint):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_a", "owner")
    _fake_request(monkeypatch, app_mod, endpoint=endpoint,
                  scan_id="scan_a", bearer=token)
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 403


def test_no_share_token_falls_through(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # No bearer / no ?share_token → not a share request; normal auth handles it.
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route", scan_id="scan_a")
    assert app_mod._try_share_token_auth() is False


def test_clerk_bearer_falls_through_to_normal_auth(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A normal Clerk JWT on a scene route is NOT a share token → fall through (False),
    # so the normal Clerk/session path authenticates it as before.
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route",
                  scan_id="scan_a", bearer="clerk-rs256-token")
    assert app_mod._try_share_token_auth() is False


# -- red-team hardening: TTL clamp, query-token strict reject, max_exp cap ----------

def test_share_token_ttl_clamped_to_max(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A huge ttl_seconds must be clamped to SHARE_TOKEN_TTL_MAX, not honored.
    _, exp = app_mod._issue_share_token("s", "o", ttl_seconds=10**12)
    assert exp <= int(time.time()) + app_mod.SHARE_TOKEN_TTL_MAX + 2


def test_share_token_ttl_floor(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _, exp = app_mod._issue_share_token("s", "o", ttl_seconds=0)
    assert exp >= int(time.time())  # clamped to >= now+1, never already-expired
    _, exp_neg = app_mod._issue_share_token("s", "o", ttl_seconds=-5)
    assert exp_neg >= int(time.time())


def test_share_token_max_exp_caps_expiry(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    cap = int(time.time()) + 100
    _, exp = app_mod._issue_share_token("s", "o", ttl_seconds=10_000_000, max_exp=cap)
    assert exp == cap  # never outlives the cap (the project token's exp)


def test_invalid_share_query_token_rejected_not_fallthrough(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # An expired/forged ?share_token= must 401, NOT fall through to cookie/session auth.
    expired = app_mod.jwt.encode(
        {"iss": app_mod.SHARE_TOKEN_ISSUER, "scope": app_mod.SHARE_TOKEN_SCOPE,
         "scan_id": "s", "sub": "o", "iat": 0, "exp": int(time.time()) - 10},
        app_mod.SESSION_TOKEN_SECRET, algorithm=app_mod.SESSION_TOKEN_ALG,
    )
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route",
                  scan_id="s", query={"share_token": expired})
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 401


def test_bearer_non_share_token_still_falls_through(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A Bearer that isn't a share token stays ambiguous → fall through (False), so normal
    # Clerk/session auth still handles it (regression guard for the query-vs-bearer split).
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route",
                  scan_id="s", bearer="clerk-rs256-token")
    assert app_mod._try_share_token_auth() is False


# -- fast-preview (LOD) delivery ---------------------------------------------------

def test_share_token_reads_lod_index(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_l", "owner_l")
    # The embed asks which decimated levels exist before fetching one (GET only).
    # Blueprint-qualified name — what Flask actually reports for an oreos_bp route
    # (test_oreos_lod.py::test_lod_get_endpoint_name_is_share_allowlisted pins it).
    _fake_request(monkeypatch, app_mod, endpoint="demo.get_scene_lod_route",
                  scan_id="scan_l", bearer=token)
    assert app_mod._try_share_token_auth() is True
    assert app_mod.request.environ[app_mod.AUTH_CLAIMS_ENV_KEY]["sub"] == "owner_l"


def test_bare_lod_function_name_is_not_an_endpoint(monkeypatch):
    # Regression guard for the 2026-08-24 deploy: the allow-list held the bare function
    # name, which is not what Flask reports for a Blueprint route, so every share embed
    # 403'd on /lod while the unit test (fake request, bare name) stayed green.
    app_mod = load_app_module(monkeypatch)
    assert "get_scene_lod_route" not in app_mod.SHARE_TOKEN_ALLOWED_ENDPOINTS
    token, _ = app_mod._issue_share_token("scan_l", "owner_l")
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_lod_route",
                  scan_id="scan_l", bearer=token)
    with pytest.raises(app_mod.AuthError):
        app_mod._try_share_token_auth()


@pytest.mark.parametrize("derived_key", [
    "demo/lod/splat_2000k.spz",
    "demo/lod/splat_2000k.ply",
    "demo/lod/full.spz",
    "demo/lod/index.json",
    "derived/demo/lod/splat_600k.spz",   # leading derived/ is tolerated by the route
])
def test_share_token_reads_lod_artifacts(monkeypatch, derived_key):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_l", "owner_l")
    req = _fake_request(monkeypatch, app_mod, endpoint="get_scene_derived_route",
                        scan_id="scan_l", query={"share_token": token})
    req.view_args["derived_key"] = derived_key
    assert app_mod._try_share_token_auth() is True


@pytest.mark.parametrize("derived_key", [
    "anchor/2026-08-01/splat.ply",       # metric-anchor derivative
    "clamp/2026-08-01/splat.ply",        # clamp derivative
    "demo/manifest.json",                # the OS workspace manifest
    "demo/agent/runs/abc/run.json",      # agent runs
    "share_access/events.json",          # the access log itself
    "demo/lod/../../cloud.ply",          # traversal dressed as an LOD key
    "demo/lod",                          # the directory, not an artifact
    "",
])
def test_share_token_rejects_non_lod_derived_keys(monkeypatch, derived_key):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_l", "owner_l")
    req = _fake_request(monkeypatch, app_mod, endpoint="get_scene_derived_route",
                        scan_id="scan_l", bearer=token)
    req.view_args["derived_key"] = derived_key
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 403


def test_share_token_lod_artifact_wrong_scan_rejected(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scan_l", "owner_l")
    req = _fake_request(monkeypatch, app_mod, endpoint="get_scene_derived_route",
                        scan_id="scan_other", bearer=token)
    req.view_args["derived_key"] = "demo/lod/splat_2000k.spz"
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 403
