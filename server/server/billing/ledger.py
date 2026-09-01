"""Typed Python wrapper over the credit ledger's SQL functions.

Deliberately thin. Every rule — what "available" means, spend order, clamping,
idempotency — lives in ``migrations/0002_credit_functions.sql`` so the Next.js
caller and this one cannot drift. If you find yourself adding an ``if`` here that
decides whether money moves, it belongs in the SQL instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from server.billing.db import LedgerUnavailable, call, is_configured

__all__ = [
    "Hold",
    "InsufficientCredits",
    "LedgerError",
    "LedgerUnavailable",
    "NoAccount",
    "available",
    "ensure_account",
    "expired_holds",
    "grant",
    "hold",
    "is_configured",
    "set_modal_call_id",
    "settle",
    "spend",
    "topup_hold",
    "void",
]


class LedgerError(RuntimeError):
    """The ledger rejected the operation."""


class InsufficientCredits(LedgerError):
    """Not enough credits. This is a normal, expected outcome — surface it to the
    user as 402, never as a 500."""


class NoAccount(LedgerError):
    """No credit account for this user yet. Call ``ensure_account`` first."""


# SQLSTATEs raised by the plpgsql functions.
_ERRORS = {
    "P0001": LedgerError,          # invalid_amount
    "P0002": NoAccount,            # no_account
    "P0003": InsufficientCredits,  # insufficient_credits
    "P0004": LedgerError,          # no_hold
    "P0005": LedgerError,          # hold_not_open
}


@dataclass(frozen=True)
class Hold:
    hold_id: str
    available: int


def _translate(exc: Exception) -> Exception:
    """Map a database error onto the typed exception the callers branch on."""
    sqlstate = getattr(exc, "sqlstate", None)
    cls = _ERRORS.get(sqlstate or "")
    if cls is None:
        return exc
    message = getattr(getattr(exc, "diag", None), "message_primary", None) or str(exc)
    return cls(message)


def _call(sql: str, params: tuple = (), *, fetch: str = "one"):
    try:
        return call(sql, params, fetch=fetch)
    except LedgerUnavailable:
        raise
    except Exception as exc:
        raise _translate(exc) from exc


# ── reads ──────────────────────────────────────────────────────────────────────

def available(user_id: str) -> int:
    row = _call("SELECT credit_available(%s::text)", (user_id,))
    return int(row[0] or 0) if row else 0


def account(user_id: str) -> Optional[dict[str, Any]]:
    row = _call(
        """
        SELECT balance_purchased, balance_subscription, subscription_expires_at,
               held, overdraft_limit, unlimited, stripe_customer_id
          FROM credit_accounts WHERE user_id = %s
        """,
        (user_id,),
    )
    if row is None:
        return None
    return {
        "balance_purchased": int(row[0]),
        "balance_subscription": int(row[1]),
        "subscription_expires_at": row[2],
        "held": int(row[3]),
        "overdraft_limit": int(row[4]),
        "unlimited": bool(row[5]),
        "stripe_customer_id": row[6],
        "available": available(user_id),
    }


def history(user_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    rows = _call(
        """
        SELECT created_at, kind, delta, ref_type, ref_id, usd_cost, cost_basis, metadata
          FROM credit_ledger
         WHERE user_id = %s
         ORDER BY created_at DESC, id DESC
         LIMIT %s OFFSET %s
        """,
        (user_id, limit, offset),
        fetch="all",
    )
    return [
        {
            "at": r[0].isoformat() if r[0] else None,
            "kind": r[1],
            "delta": int(r[2]),
            "ref_type": r[3],
            "ref_id": r[4],
            # usd_cost is our COGS, not the user's price — never expose it on a
            # user-facing route.
            "usd_cost": float(r[5]) if r[5] is not None else None,
            "cost_basis": r[6],
            "metadata": r[7] or {},
        }
        for r in (rows or [])
    ]


# ── writes ─────────────────────────────────────────────────────────────────────

def ensure_account(user_id: str, seed: int = 0, unlimited: bool = False) -> int:
    """Idempotent. ``seed`` is granted only when the account is first created."""
    row = _call("SELECT credit_ensure_account(%s::text, %s::bigint, %s::boolean)", (user_id, seed, unlimited))
    return int(row[0] or 0) if row else 0


def hold(
    user_id: str,
    amount: int,
    ref_type: str,
    ref_id: str,
    ttl_s: int,
    idem: Optional[str] = None,
) -> Hold:
    """Reserve credits for work of unknown final cost.

    Raises ``InsufficientCredits``. Idempotent on ``(ref_type, ref_id)`` — an
    already-open hold for the same reference comes back unchanged, which is what
    makes a Modal retry or a client reconnect storm harmless.
    """
    row = _call(
        "SELECT hold_id, available FROM credit_hold("
        "%s::text, %s::bigint, %s::text, %s::text, %s::int, %s::text)",
        (user_id, amount, ref_type, ref_id, ttl_s, idem),
    )
    return Hold(hold_id=str(row[0]), available=int(row[1]))


def topup_hold(hold_id: str, extra: int) -> int:
    """Extend an open hold — the per-minute metering step for live sessions.
    Raises ``InsufficientCredits`` when the next block cannot be covered."""
    row = _call("SELECT credit_topup_hold(%s::uuid, %s::bigint)", (hold_id, extra))
    return int(row[0] or 0) if row else 0


def settle(
    hold_id: str,
    actual: int,
    usd: Optional[float] = None,
    basis: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Close a hold at its true cost. Idempotent, and never raises for want of
    funds — it clamps and books the shortfall as uncollected."""
    row = _call(
        "SELECT credit_settle(%s::uuid, %s::bigint, %s::numeric, %s::text, %s::jsonb)",
        (hold_id, actual, usd, basis, json.dumps(metadata or {})),
    )
    return int(row[0] or 0) if row else 0


def void(hold_id: str, reason: Optional[str] = None) -> int:
    """Release a hold without charging. Idempotent."""
    row = _call("SELECT credit_void(%s::uuid, %s::text)", (hold_id, reason))
    return int(row[0] or 0) if row else 0


def spend(user_id: str, amount: int, ref_type: str, ref_id: str, idem: str) -> int:
    """Charge a known, fixed cost immediately. Raises ``InsufficientCredits``.

    ``idem`` is required, not optional: without it a lost HTTP response on retry
    charges twice with no concurrency involved at all.
    """
    row = _call(
        "SELECT credit_spend(%s::text, %s::bigint, %s::text, %s::text, %s::text)",
        (user_id, amount, ref_type, ref_id, idem),
    )
    return int(row[0] or 0) if row else 0


def grant(
    user_id: str,
    amount: int,
    kind: str,
    idem: str,
    ref_type: Optional[str] = None,
    ref_id: Optional[str] = None,
    expires_at=None,
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Credits in. ``expires_at`` routes to the subscription bucket.

    For Stripe, ``idem`` MUST be the event id — Stripe redelivers webhooks.
    """
    row = _call(
        "SELECT credit_grant("
        "%s::text, %s::bigint, %s::text, %s::text, %s::text, %s::text, "
        "%s::timestamptz, %s::jsonb)",
        (user_id, amount, kind, idem, ref_type, ref_id, expires_at,
         json.dumps(metadata or {})),
    )
    return int(row[0] or 0) if row else 0


def set_modal_call_id(hold_id: str, call_id: str) -> None:
    """Record the Modal FunctionCall id so the reaper can tell a dead container
    from a slow one instead of guessing."""
    _call(
        "UPDATE credit_holds SET modal_call_id = %s::text WHERE id = %s::uuid",
        (call_id, hold_id),
        fetch="none",
    )


def expired_holds(grace_s: int = 0) -> list[dict[str, Any]]:
    rows = _call("SELECT * FROM credit_expire_holds(%s::int)", (grace_s,), fetch="all")
    return [
        {
            "hold_id": str(r[0]),
            "user_id": r[1],
            "ref_type": r[2],
            "ref_id": r[3],
            "amount_est": int(r[4]),
            "modal_call_id": r[5],
        }
        for r in (rows or [])
    ]


def invariant_violations() -> list[dict[str, Any]]:
    """Accounts whose balance disagrees with their own ledger. Always empty."""
    rows = _call("SELECT * FROM credit_assert_invariant()", fetch="all")
    return [
        {"user_id": r[0], "balance": int(r[1]), "ledger_sum": int(r[2])}
        for r in (rows or [])
    ]
