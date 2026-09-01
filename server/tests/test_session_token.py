"""Durable broker session-token tests.

The revisit page trades its short-lived, unrefreshable Clerk JWT (carried in the URL
hash) for a broker-signed session token so history reads + grounded Q&A keep
authenticating after that JWT expires. These tests exercise mint/verify, the alg-based
dispatch between Clerk (RS256) and session (HS256) tokens, the HTTP accept path, and
expiry/issuer rejection. ``jwt`` is faked by conftest (see ``load_app_module``), so this
covers the wrapper logic, not real crypto.
"""

from __future__ import annotations

import time
import types

import pytest

from conftest import load_app_module


def test_issue_and_verify_roundtrip(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    clerk_claims = {"sub": "user_123", "tier": "approved", "scansRemaining": 5}

    token, expires_at = app_mod._issue_session_token(clerk_claims)
    assert expires_at > time.time()

    claims = app_mod._verify_session_token(token)
    # Identity survives, under our own issuer.
    assert claims["sub"] == "user_123"
    assert claims["tier"] == "approved"
    assert claims["iss"] == app_mod.SESSION_TOKEN_ISSUER

    # ...but the QUOTA does NOT. This token lives for 12 hours; snapshotting a
    # balance into it made it a 12-hour grant, so a user at zero could keep
    # allocating GPU sessions all day. Balance is read live from the ledger now,
    # and the authoritative gate is the hold taken at dispatch.
    assert app_mod.SCANS_CLAIM not in claims


def test_verify_any_token_routes_by_alg(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # HS256 session token -> session verifier (sub from our payload).
    token, _ = app_mod._issue_session_token({"sub": "user_abc"})
    assert app_mod._verify_any_token(token)["sub"] == "user_abc"
    # Anything else -> Clerk verifier (the stub's fixed verified user).
    assert app_mod._verify_any_token("clerk-rs256-token")["sub"] == "test-user"


def test_http_token_accepts_session_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_session_token({"sub": "user_xyz", "tier": "approved"})
    monkeypatch.setattr(
        app_mod,
        "request",
        types.SimpleNamespace(
            headers={"Authorization": f"Bearer {token}"}, args={}, cookies={}
        ),
    )
    # The durable token authenticates an /api/ call exactly like a Clerk JWT would.
    assert app_mod._verify_http_token()["sub"] == "user_xyz"


def test_expired_session_token_rejected(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "SESSION_TOKEN_TTL_SEC", -10)  # mint already-expired
    token, _ = app_mod._issue_session_token({"sub": "user_1"})
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_session_token(token)


def test_foreign_issuer_rejected(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    forged = app_mod.jwt.encode(
        {"iss": "someone-else", "sub": "x", "exp": 9_999_999_999}, "k", algorithm="HS256"
    )
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_session_token(forged)
