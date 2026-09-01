"""Credit ledger — the money rules.

Runs against a real Postgres because the properties under test (row locking,
unique-index idempotency, clamping) are properties of the database, not of the
Python wrapper. Point ``BILLING_TEST_DATABASE_URL`` at a scratch database; the
suite skips cleanly when there isn't one, so CI without Postgres stays green.

    createdb openreality_billing_test
    BILLING_TEST_DATABASE_URL=postgresql:///openreality_billing_test pytest tests/test_credit_ledger.py
"""

from __future__ import annotations

import os
import threading

import pytest

DEFAULT_TEST_URL = "postgresql:///openreality_billing_test"


def _test_url() -> str | None:
    url = os.environ.get("BILLING_TEST_DATABASE_URL", "").strip()
    if url:
        return url
    # Convenience for local dev: try the conventional scratch database, but only
    # if psycopg is installed and it actually accepts a connection.
    try:
        import psycopg

        with psycopg.connect(DEFAULT_TEST_URL, connect_timeout=2):
            return DEFAULT_TEST_URL
    except Exception:
        return None


@pytest.fixture()
def ledger(monkeypatch):
    url = _test_url()
    if url is None:
        pytest.skip("no test Postgres (set BILLING_TEST_DATABASE_URL)")

    monkeypatch.setenv("DATABASE_URL", url)

    from server.billing import db as billing_db
    from server.billing import ledger as led

    billing_db.reset_pool_for_tests()
    billing_db.apply_migrations()
    billing_db.call(
        "TRUNCATE credit_ledger, credit_holds, credit_accounts CASCADE", fetch="none"
    )
    yield led
    billing_db.reset_pool_for_tests()


# ── the core guarantee ─────────────────────────────────────────────────────────


def test_concurrent_holds_against_one_credit_admit_exactly_one(ledger):
    """THE test. The old Clerk quota was a read-modify-write: two tabs both read
    1, both wrote 0, two scans ran and one was charged — and because each computed
    next = remaining - 1 from its own stale read there was no negative residue, so
    it was undetectable forever. Under a row lock exactly one caller may win."""
    ledger.ensure_account("racer", seed=1)

    results: list[str] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(16)

    def attempt(i: int) -> None:
        barrier.wait()  # maximise the overlap
        try:
            ledger.hold("racer", 1, "recon_job", f"job-{i}", 60)
            results.append(f"job-{i}")
        except ledger.InsufficientCredits as exc:
            errors.append(exc)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1, f"expected exactly one winner, got {results}"
    assert len(errors) == 15
    assert ledger.available("racer") == 0


def test_concurrent_spends_cannot_oversell(ledger):
    """Same guarantee on the immediate-spend path (/api/scans/consume)."""
    ledger.ensure_account("spender", seed=3)

    ok: list[int] = []
    barrier = threading.Barrier(12)

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            ledger.spend("spender", 1, "scan", f"s-{i}", f"idem-{i}")
            ok.append(i)
        except ledger.InsufficientCredits:
            pass

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ok) == 3
    assert ledger.available("spender") == 0


# ── holds ──────────────────────────────────────────────────────────────────────


def test_hold_reserves_then_settle_charges_actual(ledger):
    ledger.ensure_account("u", seed=200)
    h = ledger.hold("u", 60, "recon_job", "job-1", 3600)
    assert ledger.available("u") == 140  # reserved, not yet spent

    ledger.settle(h.hold_id, 45, usd=0.169, basis="modal_rate_estimate")
    assert ledger.available("u") == 155  # charged the real cost, hold released


def test_settle_is_idempotent_under_modal_retry(ledger):
    """Modal re-runs the body of a preempted call with the same hold_id. We burn
    two containers but must charge once."""
    ledger.ensure_account("u", seed=100)
    h = ledger.hold("u", 20, "recon_job", "job-1", 3600)

    first = ledger.settle(h.hold_id, 20)
    second = ledger.settle(h.hold_id, 20)
    assert first == second == 80


def test_void_releases_without_charging(ledger):
    ledger.ensure_account("u", seed=100)
    h = ledger.hold("u", 40, "recon_job", "job-1", 3600)
    assert ledger.void(h.hold_id, "job crashed") == 100
    assert ledger.void(h.hold_id, "again") == 100  # idempotent


def test_duplicate_ref_returns_the_same_hold(ledger):
    """The durable replacement for `any_active_scene_job`, which is in-process and
    lost on restart. Also what contains the hold storm from the client's
    waitForReadyModalGpuSession retry loop."""
    ledger.ensure_account("u", seed=100)
    a = ledger.hold("u", 40, "export_job", "e1", 3600)
    b = ledger.hold("u", 40, "export_job", "e1", 3600)
    assert a.hold_id == b.hold_id
    assert ledger.available("u") == 60  # reserved once, not twice


def test_hold_refuses_when_short(ledger):
    ledger.ensure_account("u", seed=10)
    with pytest.raises(ledger.InsufficientCredits):
        ledger.hold("u", 999, "recon_job", "job-1", 3600)
    assert ledger.available("u") == 10  # nothing reserved on refusal


def test_topup_meters_a_running_session(ledger):
    """Live sessions are metered per minute rather than reserved for the full
    hour — an hour of A100 held up front would lock out every small balance."""
    ledger.ensure_account("u", seed=100)
    h = ledger.hold("u", 25, "session", "sess-1", 7200)
    ledger.topup_hold(h.hold_id, 25)
    assert ledger.available("u") == 50

    with pytest.raises(ledger.InsufficientCredits):
        ledger.topup_hold(h.hold_id, 500)  # the signal to end the session

    ledger.settle(h.hold_id, 50)
    assert ledger.available("u") == 50


# ── settle must never fail ─────────────────────────────────────────────────────


def test_settle_clamps_instead_of_failing(ledger):
    """A settle that raised would mean work done, provider paid, and no record.
    It clamps at the overdraft floor and books the shortfall as queryable bad
    debt instead."""
    ledger.ensure_account("u", seed=5)
    h = ledger.hold("u", 5, "session", "sess-1", 3600)

    ledger.settle(h.hold_id, 500)
    assert ledger.available("u") == -100  # the default overdraft floor

    rows = [r for r in ledger.history("u") if r["kind"] == "settle"]
    assert rows[0]["delta"] == -105
    assert rows[0]["metadata"]["uncollected_credits"] == 395
    assert rows[0]["metadata"]["amount_requested"] == 500


# ── unlimited accounts ─────────────────────────────────────────────────────────


def test_unlimited_is_never_refused_but_still_ledgered(ledger):
    """`tier: "approved"` accounts are where most of our cost data comes from.
    They must never be blocked, and must never stop producing measurements."""
    ledger.ensure_account("admin", seed=0, unlimited=True)
    h = ledger.hold("admin", 100_000, "session", "sess-1", 3600)
    ledger.settle(h.hold_id, 100_000, usd=3.40, basis="modal_rate_estimate")

    assert ledger.available("admin") == 0  # balance untouched
    row = [r for r in ledger.history("admin") if r["kind"] == "settle"][0]
    assert row["delta"] == 0
    assert row["usd_cost"] == 3.40
    assert row["metadata"]["unlimited"] is True


# ── grants, idempotency, subscriptions ─────────────────────────────────────────


def test_grant_is_idempotent_on_stripe_event_id(ledger):
    """Stripe redelivers webhooks. Same event id must grant once."""
    ledger.ensure_account("u")
    ledger.grant("u", 5000, "purchase", idem="evt_123")
    ledger.grant("u", 5000, "purchase", idem="evt_123")
    assert ledger.available("u") == 5000


def test_spend_is_idempotent_on_its_key(ledger):
    """A lost response on retry must not charge twice."""
    ledger.ensure_account("u", seed=100)
    ledger.spend("u", 10, "scan", "scan-1", "req-abc")
    ledger.spend("u", 10, "scan", "scan-1", "req-abc")
    assert ledger.available("u") == 90


def test_ensure_account_seeds_once(ledger):
    assert ledger.ensure_account("u", seed=200) == 200
    assert ledger.ensure_account("u", seed=200) == 200  # not 400


def test_subscription_credits_are_spent_before_purchased(ledger):
    """Subscription credits expire and purchased ones don't, so spending the
    perishable bucket first is what the user would choose."""
    import datetime as dt

    ledger.ensure_account("u", seed=100)  # purchased
    ledger.grant(
        "u", 50, "subscription_grant", idem="sub-1",
        expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30),
    )
    assert ledger.available("u") == 150

    ledger.spend("u", 50, "scan", "s1", "k1")
    acct = ledger.account("u")
    assert acct["balance_subscription"] == 0
    assert acct["balance_purchased"] == 100  # untouched


def test_expired_subscription_credits_are_not_spendable(ledger):
    import datetime as dt

    ledger.ensure_account("u", seed=10)
    ledger.grant(
        "u", 500, "subscription_grant", idem="sub-old",
        expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1),
    )
    assert ledger.available("u") == 10  # the 500 is gone, not spendable

    with pytest.raises(ledger.InsufficientCredits):
        ledger.spend("u", 100, "scan", "s1", "k1")


def test_expiry_keeps_the_append_only_invariant(ledger):
    """Zeroing an expired bucket writes its own ledger row, so balance still
    equals the sum of deltas."""
    import datetime as dt

    ledger.ensure_account("u", seed=10)
    ledger.grant(
        "u", 500, "subscription_grant", idem="sub-old",
        expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1),
    )
    ledger.spend("u", 1, "scan", "s1", "k1")  # triggers lazy expiry
    assert ledger.invariant_violations() == []


# ── reaper ─────────────────────────────────────────────────────────────────────


def test_expired_holds_are_reported_for_the_reaper(ledger):
    """A SIGKILLed container settles nothing. The reaper is the only backstop."""
    ledger.ensure_account("u", seed=100)
    h = ledger.hold("u", 30, "recon_job", "job-1", ttl_s=-1)  # already expired
    ledger.set_modal_call_id(h.hold_id, "fc-abc123")

    stale = ledger.expired_holds()
    assert [s["hold_id"] for s in stale] == [h.hold_id]
    assert stale[0]["modal_call_id"] == "fc-abc123"
    assert stale[0]["amount_est"] == 30

    ledger.void(h.hold_id, "reaped: never started")
    assert ledger.expired_holds() == []


def test_settled_holds_are_not_reaped(ledger):
    ledger.ensure_account("u", seed=100)
    h = ledger.hold("u", 30, "recon_job", "job-1", ttl_s=-1)
    ledger.settle(h.hold_id, 30)
    assert ledger.expired_holds() == []


# ── invariant ──────────────────────────────────────────────────────────────────


def test_invariant_holds_across_a_full_lifecycle(ledger):
    ledger.ensure_account("u", seed=500)
    ledger.grant("u", 1000, "purchase", idem="evt_1")
    h1 = ledger.hold("u", 100, "session", "s1", 3600)
    ledger.topup_hold(h1.hold_id, 50)
    ledger.settle(h1.hold_id, 120, usd=0.4, basis="modal_rate_estimate")
    h2 = ledger.hold("u", 60, "export_job", "e1", 3600)
    ledger.void(h2.hold_id, "failed")
    ledger.spend("u", 25, "scan", "sc1", "k1")
    ledger.grant("u", -200, "refund", idem="evt_refund")

    assert ledger.invariant_violations() == []
    assert ledger.available("u") == 500 + 1000 - 120 - 25 - 200


# ── unconfigured ───────────────────────────────────────────────────────────────


def test_ledger_absent_raises_a_distinct_error(monkeypatch):
    """Shipping dark must not crash. 'No ledger' and 'the ledger said no' are
    opposite situations and callers branch on the difference."""
    from server.billing import db as billing_db
    from server.billing import ledger as led

    monkeypatch.delenv("DATABASE_URL", raising=False)
    billing_db.reset_pool_for_tests()
    try:
        with pytest.raises(led.LedgerUnavailable):
            led.available("nobody")
    finally:
        billing_db.reset_pool_for_tests()
