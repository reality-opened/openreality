"""Cost accounting for OpenReality.

Phase 0 of the credit-billing plan: make cost *visible* before anything is
charged. This package owns the two things that must never be duplicated —
what a unit of work costs us (``prices``) and what a run actually consumed
(``usage``).

Nothing here refuses work or touches a balance. That arrives with the ledger.
"""

from server.billing.usage import UsageTally

__all__ = ["UsageTally"]
