"""The safety contract every dispatch point relies on.

Two properties are load-bearing and easy to regress:

  1. With no ledger configured, nothing raises and nothing blocks. This is what
     lets the billing layer ship before Neon exists and survive a Neon outage.
  2. In shadow mode an insufficient balance is recorded but never refuses work,
     and flipping BILLING_ENFORCE is the ONLY difference between the measuring
     system and the charging system — which is what makes the two weeks of
     shadow data a valid basis for setting prices.
"""

from __future__ import annotations

import os

import pytest

from tests.test_credit_ledger import _test_url  # shared skip logic


@pytest.fixture()
def unconfigured(monkeypatch):
    from server.billing import db as billing_db

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("BILLING_ENFORCE", raising=False)
    billing_db.reset_pool_for_tests()
    from server.billing import meter

    yield meter
    billing_db.reset_pool_for_tests()


@pytest.fixture()
def broke(monkeypatch):
    """A real ledger holding an account with no credits."""
    url = _test_url()
    if url is None:
        pytest.skip("no test Postgres (set BILLING_TEST_DATABASE_URL)")

    monkeypatch.setenv("DATABASE_URL", url)
    from server.billing import db as billing_db
    from server.billing import ledger, meter

    billing_db.reset_pool_for_tests()
    billing_db.apply_migrations()
    billing_db.call(
        "TRUNCATE credit_ledger, credit_holds, credit_accounts CASCADE", fetch="none"
    )
    ledger.ensure_account("broke", seed=0)
    yield meter
    billing_db.reset_pool_for_tests()


# ── 1. no ledger = no-op, never an exception ───────────────────────────────────


def test_every_entry_point_degrades_to_a_noop(unconfigured):
    meter = unconfigured
    assert meter.enabled() is False

    assert meter.open_hold("u", "session", "s1", 100) is None
    assert meter.topup(None, 25) is True
    assert meter.balance("u") is None
    assert meter.charge("u", 10, "scan", "s1", "idem") is None
    # settle/void on a null hold are silent no-ops, not errors
    meter.settle(None, 10)
    meter.void(None, "reason")
    meter.attach_modal_call(None, "fc-1")
    meter.ensure_account("u")


def test_metered_context_still_runs_the_work_without_a_ledger(unconfigured):
    meter = unconfigured
    ran = []
    with meter.metered("u", "gen_asset", "g1", 16) as rec:
        ran.append(True)
        rec.record(16, usd=0.04, basis="fal_flat")
    assert ran == [True]


def test_metered_context_propagates_the_real_error(unconfigured):
    meter = unconfigured
    with pytest.raises(ValueError, match="boom"):
        with meter.metered("u", "gen_asset", "g1", 16):
            raise ValueError("boom")


# ── 2. shadow mode records but never refuses ───────────────────────────────────


def test_shadow_mode_does_not_refuse_a_broke_user(broke, monkeypatch):
    monkeypatch.delenv("BILLING_ENFORCE", raising=False)
    meter = broke
    assert meter.enforcing() is False

    # Would be refused under enforcement; here it proceeds unbilled.
    assert meter.open_hold("broke", "session", "s1", 10_000) is None
    assert meter.charge("broke", 10_000, "scan", "s1", "k1") is None
    assert meter.topup(None, 10_000) is True


def test_enforcing_refuses_the_same_calls(broke, monkeypatch):
    monkeypatch.setenv("BILLING_ENFORCE", "1")
    meter = broke
    assert meter.enforcing() is True

    with pytest.raises(meter.OutOfCredits):
        meter.open_hold("broke", "session", "s1", 10_000)
    with pytest.raises(meter.OutOfCredits):
        meter.charge("broke", 10_000, "scan", "s1", "k1")


def test_enforcing_topup_returns_false_to_stop_a_session(broke, monkeypatch):
    """The live session's out-of-credits signal. It must be a return value, not
    an exception — it is read inside the lifecycle loop."""
    from server.billing import ledger

    monkeypatch.setenv("BILLING_ENFORCE", "1")
    meter = broke
    ledger.grant("broke", 30, "grant", idem="topup-test")
    h = meter.open_hold("broke", "session", "s2", 25)
    assert h is not None
    assert meter.topup(h, 5) is True     # 30 - 25 - 5 = 0 left
    assert meter.topup(h, 25) is False   # cannot cover the next minute


def test_shadow_topup_never_stops_a_session(broke, monkeypatch):
    from server.billing import ledger

    monkeypatch.delenv("BILLING_ENFORCE", raising=False)
    meter = broke
    ledger.grant("broke", 30, "grant", idem="topup-test-2")
    h = meter.open_hold("broke", "session", "s3", 25)
    assert meter.topup(h, 10_000) is True


# ── the enforcement flag itself ────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("on", True), ("yes", True),
    ("0", False), ("false", False), ("off", False), ("", False), ("maybe", False),
])
def test_enforce_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("BILLING_ENFORCE", value)
    from server.billing import meter

    assert meter.enforcing() is expected


def test_enforcement_is_off_by_default(monkeypatch):
    """Shipping dark is the default. If this ever flips by accident, users get
    refused with prices that were never reconciled against an invoice."""
    monkeypatch.delenv("BILLING_ENFORCE", raising=False)
    from server.billing import meter

    assert meter.enforcing() is False
