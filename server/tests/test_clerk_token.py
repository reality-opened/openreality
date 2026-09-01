"""Clerk JWT verification: the `azp` (authorized party) contract.

Clerk derives `azp` from the browser Origin, so tokens minted for **native mobile
apps carry no `azp` claim at all** (and Clerk refuses it as a custom template claim).
Requiring it therefore 401'd every native call before any value was compared.
Absent `azp` is accepted; a present `azp` must still be in the `CLERK_JWT_AZP`
allow-list, so browser tokens are unaffected.

``jwt`` is faked by conftest (see ``load_app_module``); the helper below re-fakes
``jwt.decode`` so PyJWT's ``require`` option is honoured the way the real library
does — a missing required claim raises before the allow-list check runs.
"""

from __future__ import annotations

import pytest

from conftest import load_app_module


def _fake_clerk_decode(monkeypatch, app_mod, claims):
    """Make the Clerk (RS256) decode path return ``claims``, enforcing ``require``."""

    def _decode(token, key, **kwargs):
        for required in kwargs.get("options", {}).get("require", []):
            if required not in claims:
                raise app_mod.jwt.PyJWTError(f"Token is missing the '{required}' claim")
        return dict(claims)

    monkeypatch.setattr(app_mod.jwt, "decode", _decode)


def test_token_without_azp_is_accepted(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A native-app token: valid in every other respect, simply has no azp.
    _fake_clerk_decode(
        monkeypatch,
        app_mod,
        {"sub": "user_native", "iss": "https://clerk.test", "exp": 9_999_999_999},
    )
    assert app_mod._verify_clerk_token("clerk-rs256-token")["sub"] == "user_native"


def test_token_with_unlisted_azp_is_rejected(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _fake_clerk_decode(
        monkeypatch,
        app_mod,
        {
            "sub": "user_browser",
            "iss": "https://clerk.test",
            "exp": 9_999_999_999,
            "azp": "https://evil.test",
        },
    )
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_clerk_token("clerk-rs256-token")


def test_token_with_allowlisted_azp_is_accepted(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # conftest sets CLERK_JWT_AZP=https://app.test
    _fake_clerk_decode(
        monkeypatch,
        app_mod,
        {
            "sub": "user_browser",
            "iss": "https://clerk.test",
            "exp": 9_999_999_999,
            "azp": "https://app.test",
        },
    )
    assert app_mod._verify_clerk_token("clerk-rs256-token")["sub"] == "user_browser"
