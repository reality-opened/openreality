"""The seam every dispatch point calls. Safe by construction.

Two rules govern everything here:

1. **A billing fault must never break the product.** If the ledger is
   unconfigured or unreachable, every function degrades to a no-op and logs. The
   whole layer can be deployed before Neon exists, and if Neon has an outage the
   scanner keeps working. We would rather lose a charge than lose a customer's
   scan.

2. **Shadow mode is one flag, not a separate code path.** With
   ``BILLING_ENFORCE`` off (the default) an insufficient balance is logged and
   the work proceeds. Every hold, top-up and settle still runs, so the ledger
   fills with real data whose accuracy can be checked against real invoices
   before a single user is ever refused. Flipping the flag is the only
   difference between the measuring system and the charging system — which is
   what makes the measurement trustworthy.

Import this, not ``ledger``, from product code.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

from server.billing import ledger, prices


def enforcing() -> bool:
    """Whether an insufficient balance actually refuses work."""
    return os.environ.get("BILLING_ENFORCE", "").strip().lower() in {"1", "true", "on", "yes"}


def enabled() -> bool:
    return ledger.is_configured()


class OutOfCredits(RuntimeError):
    """Raised only when enforcing. Callers map it to HTTP 402."""


def _log(msg: str) -> None:
    print(f"[billing] {msg}")


def ensure_account(user_id: str, unlimited: bool = False) -> None:
    """Idempotent; seeds the free allowance on first sight of a user."""
    if not user_id:
        return
    try:
        ledger.ensure_account(user_id, seed=prices.SIGNUP_CREDITS, unlimited=unlimited)
    except ledger.LedgerUnavailable:
        pass
    except Exception as exc:
        _log(f"ensure_account({user_id}) failed: {exc}")


def open_hold(
    user_id: str,
    ref_type: str,
    ref_id: str,
    credits: int,
    ttl_s: int = prices.HOLD_TTL_JOB_S,
) -> Optional[str]:
    """Reserve credits before dispatching work.

    Returns a hold id, or None when the ledger is unavailable (proceed unbilled)
    or when the balance is short and we are not enforcing. Raises ``OutOfCredits``
    only when enforcing — that is the one case where the caller must not dispatch.
    """
    if not user_id:
        return None
    try:
        ensure_account(user_id)
        h = ledger.hold(user_id, credits, ref_type, ref_id, ttl_s)
        return h.hold_id
    except ledger.InsufficientCredits:
        if enforcing():
            raise OutOfCredits(f"{ref_type} needs {credits} credits") from None
        _log(f"SHADOW would refuse {ref_type}:{ref_id} for {user_id} ({credits} credits)")
        return None
    except ledger.LedgerUnavailable:
        return None
    except Exception as exc:
        _log(f"open_hold({ref_type}:{ref_id}) failed: {exc}")
        return None


def topup(hold_id: Optional[str], credits: int) -> bool:
    """Extend a running hold. False means "stop the work" — but only when
    enforcing; in shadow mode it always returns True so nothing is interrupted."""
    if not hold_id:
        return True
    try:
        ledger.topup_hold(hold_id, credits)
        return True
    except ledger.InsufficientCredits:
        if enforcing():
            return False
        _log(f"SHADOW would stop hold {hold_id} (out of credits)")
        return True
    except ledger.LedgerUnavailable:
        return True
    except Exception as exc:
        _log(f"topup({hold_id}) failed: {exc}")
        return True


def settle(
    hold_id: Optional[str],
    credits: int,
    usd: Optional[float] = None,
    basis: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Close a hold at its true cost. Best-effort and idempotent — safe to call
    from a Modal container that may be retried."""
    if not hold_id:
        return
    try:
        ledger.settle(hold_id, max(0, int(credits)), usd, basis, metadata)
    except ledger.LedgerUnavailable:
        pass
    except Exception as exc:
        _log(f"settle({hold_id}) failed: {exc}")


def void(hold_id: Optional[str], reason: str = "") -> None:
    """Release a hold without charging — our fault, so we eat it."""
    if not hold_id:
        return
    try:
        ledger.void(hold_id, reason)
    except ledger.LedgerUnavailable:
        pass
    except Exception as exc:
        _log(f"void({hold_id}) failed: {exc}")


def attach_modal_call(hold_id: Optional[str], call_id: Optional[str]) -> None:
    """Record the spawned call id so the reaper can distinguish a dead container
    from a slow one."""
    if not hold_id or not call_id:
        return
    try:
        ledger.set_modal_call_id(hold_id, call_id)
    except ledger.LedgerUnavailable:
        pass
    except Exception as exc:
        _log(f"attach_modal_call({hold_id}) failed: {exc}")


def charge(user_id: str, credits: int, ref_type: str, ref_id: str, idem: str) -> Optional[int]:
    """Immediate, known-cost spend. Returns the remaining balance, or None when
    unbilled. Raises ``OutOfCredits`` when enforcing and short."""
    if not user_id:
        return None
    try:
        ensure_account(user_id)
        return ledger.spend(user_id, credits, ref_type, ref_id, idem)
    except ledger.InsufficientCredits:
        if enforcing():
            raise OutOfCredits(f"{ref_type} needs {credits} credits") from None
        _log(f"SHADOW would refuse spend {ref_type}:{ref_id} for {user_id}")
        return None
    except ledger.LedgerUnavailable:
        return None
    except Exception as exc:
        _log(f"charge({ref_type}:{ref_id}) failed: {exc}")
        return None


def balance(user_id: str) -> Optional[int]:
    """Spendable credits, or None when there is no ledger to ask."""
    if not user_id:
        return None
    try:
        return ledger.available(user_id)
    except ledger.LedgerUnavailable:
        return None
    except Exception as exc:
        _log(f"balance({user_id}) failed: {exc}")
        return None


@contextmanager
def metered(
    user_id: str,
    ref_type: str,
    ref_id: str,
    estimate: int,
    ttl_s: int = prices.HOLD_TTL_JOB_S,
):
    """Hold → run → settle, for work that begins and ends in this process.

    Yields a small recorder; call ``record(credits, usd=…, basis=…)`` to set the
    true cost. An exception voids the hold rather than charging for work that
    failed on our side.
    """
    hold_id = open_hold(user_id, ref_type, ref_id, estimate, ttl_s)
    outcome: dict[str, Any] = {"credits": estimate, "usd": None, "basis": None, "metadata": {}}

    class _Recorder:
        def record(self, credits: int, usd: Optional[float] = None,
                   basis: Optional[str] = None, **metadata: Any) -> None:
            outcome["credits"] = credits
            outcome["usd"] = usd
            outcome["basis"] = basis
            outcome["metadata"].update(metadata)

    try:
        yield _Recorder()
    except BaseException as exc:
        void(hold_id, f"{type(exc).__name__}: {exc}"[:200])
        raise
    else:
        settle(hold_id, outcome["credits"], outcome["usd"], outcome["basis"], outcome["metadata"])
