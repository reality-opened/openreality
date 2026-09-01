"""Worker scan-quota gate (replaces the old binary approved-tier gate).

The worker reads the ``scansRemaining`` JWT claim as best-effort,
defence-in-depth enforcement; the authoritative balance is spent by the landing
``/api/scans/consume`` route against live Clerk metadata.
"""

import pytest

from conftest import load_app_module


def test_approved_tier_is_unlimited(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._scans_remaining({"tier": "approved"}) is None
    # Unlimited never raises, even with a zero count alongside.
    app_mod._require_scan_access({"tier": "approved", "scansRemaining": 0})


def test_missing_claim_defaults_to_two(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._scans_remaining({}) == app_mod.DEFAULT_SCANS
    assert app_mod._scans_remaining({"scansRemaining": ""}) == app_mod.DEFAULT_SCANS
    app_mod._require_scan_access({})  # never-seeded new user is allowed


def test_string_and_int_counts_are_parsed(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._scans_remaining({"scansRemaining": 1}) == 1
    assert app_mod._scans_remaining({"scansRemaining": "3"}) == 3
    app_mod._require_scan_access({"scansRemaining": 1})


def test_zero_scans_is_forbidden(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    with pytest.raises(app_mod.AuthError) as excinfo:
        app_mod._require_scan_access({"scansRemaining": 0})
    assert excinfo.value.status_code == 403
    with pytest.raises(app_mod.AuthError):
        app_mod._require_scan_access({"scansRemaining": "0"})
