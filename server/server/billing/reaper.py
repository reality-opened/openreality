"""Closes holds whose owner died without settling.

Every other settle path runs *inside* the container doing the work. That covers
exceptions, but not SIGKILL and not Modal preemption — in those cases nothing
in-process runs and the hold would sit open forever, permanently reserving
credits the user never spent. This is the only backstop for that, which is why
it runs on the broker: `min_containers=1`, `timeout=86400`, the one process in
the system that is always warm.

Classification, not guesswork: an expired hold carrying a `modal_call_id` is
cross-checked against Modal itself.

  * call finished (or the id is unknown/gone) → the work was done, or at least
    started and billed to us by Modal → **settle at the estimate**.
  * call never started / not found              → **void**.

When Modal cannot be consulted at all we default to VOID, because charging for
work we cannot prove happened is the worse error.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from server.billing import ledger, meter

REAP_INTERVAL_S = float(os.environ.get("BILLING_REAP_INTERVAL_S", "60"))
# Grace beyond expires_at before a hold is considered orphaned. A hold reaped out
# from under a still-running job would double-charge when the job finally settles,
# so this errs long.
REAP_GRACE_S = int(os.environ.get("BILLING_REAP_GRACE_S", "300"))

_started = False
_lock = threading.Lock()


def _modal_call_finished(call_id: str) -> Optional[bool]:
    """True/False if we can tell, None if we cannot reach Modal."""
    try:
        import modal

        fc = modal.FunctionCall.from_id(call_id)
        try:
            fc.get(timeout=0)
            return True  # completed
        except TimeoutError:
            return False  # still running — not orphaned at all
        except Exception:
            # Terminal failure of the call itself: it ran and died.
            return True
    except Exception:
        return None


def reap_once() -> dict[str, int]:
    """One pass. Returns counts by action, for the log line."""
    counts = {"settled": 0, "voided": 0, "skipped": 0}
    try:
        stale = ledger.expired_holds(REAP_GRACE_S)
    except ledger.LedgerUnavailable:
        return counts
    except Exception as exc:
        print(f"[billing.reaper] listing expired holds failed: {exc}")
        return counts

    for h in stale:
        call_id = h.get("modal_call_id")
        finished = _modal_call_finished(call_id) if call_id else None

        if finished is False:
            counts["skipped"] += 1  # genuinely still running
            continue

        if finished is True:
            # It ran; Modal billed us for the compute whether or not it produced
            # anything. Charge the estimate rather than nothing.
            meter.settle(
                h["hold_id"],
                h["amount_est"],
                basis="estimate",
                metadata={"reaped": True, "reason": "container died after starting"},
            )
            counts["settled"] += 1
        else:
            meter.void(h["hold_id"], "reaped: no evidence the work ran")
            counts["voided"] += 1

    if any(counts.values()):
        print(
            f"[billing.reaper] settled={counts['settled']} voided={counts['voided']} "
            f"skipped={counts['skipped']}"
        )
    return counts


def _loop() -> None:
    while True:
        try:
            reap_once()
        except Exception as exc:  # never let the reaper die
            print(f"[billing.reaper] pass failed: {exc}")
        time.sleep(REAP_INTERVAL_S)


def start() -> bool:
    """Start the daemon once per process. No-op without a ledger."""
    global _started
    if not ledger.is_configured():
        return False
    with _lock:
        if _started:
            return True
        _started = True
    threading.Thread(target=_loop, name="billing-reaper", daemon=True).start()
    print(f"[billing.reaper] started (every {REAP_INTERVAL_S:.0f}s, grace {REAP_GRACE_S}s)")
    return True
