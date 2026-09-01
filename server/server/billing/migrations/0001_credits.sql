-- Credit ledger schema.
--
-- Owned by `server` rather than the landing app because the concurrency tests
-- that matter (N racing holds against one credit) run in the Python suite, and
-- because this is where the money is actually spent — the Modal containers that
-- hold and settle. The Next.js side never needs the DDL, only the functions in
-- 0002.
--
-- Three tables and one rule: the ledger is APPEND-ONLY. A refund is a new row,
-- never an UPDATE of the row it reverses. The invariant asserted in
-- `credit_assert_invariant` is:
--
--     balance_purchased + balance_subscription == SUM(credit_ledger.delta)
--
-- which is why hold and void rows carry delta = 0 — they move money between
-- "available" and "held", not into or out of the account.

CREATE TABLE IF NOT EXISTS credit_accounts (
  user_id                 TEXT PRIMARY KEY,            -- Clerk claims.sub
  balance_purchased       BIGINT  NOT NULL DEFAULT 0,  -- never expires
  balance_subscription    BIGINT  NOT NULL DEFAULT 0,  -- expires with the period
  subscription_expires_at TIMESTAMPTZ,
  held                    BIGINT  NOT NULL DEFAULT 0,  -- sum of open holds
  overdraft_limit         BIGINT  NOT NULL DEFAULT 100,
  -- Successor to Clerk's `tier: "approved"`. Unlimited accounts are never
  -- refused, but they STILL write ledger rows (delta 0, real usd_cost) because
  -- admin accounts are where most of our cost data comes from.
  unlimited               BOOLEAN NOT NULL DEFAULT FALSE,
  stripe_customer_id      TEXT UNIQUE,
  version                 BIGINT  NOT NULL DEFAULT 0,
  created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- NOT `>= 0`. A settle may legitimately exceed its hold (a job overran) and a
  -- refund clawback may exceed the balance. A settle must never fail — a failed
  -- settle means work done for free with no record — so it clamps at this floor
  -- and books the shortfall to metadata.uncollected_credits instead.
  CONSTRAINT balance_floor CHECK
    (balance_purchased + balance_subscription >= -overdraft_limit),
  CONSTRAINT held_nonneg CHECK (held >= 0),
  CONSTRAINT overdraft_nonneg CHECK (overdraft_limit >= 0)
);

CREATE TABLE IF NOT EXISTS credit_holds (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        TEXT   NOT NULL REFERENCES credit_accounts(user_id) ON DELETE CASCADE,
  amount_est     BIGINT NOT NULL CHECK (amount_est >= 0),
  amount_settled BIGINT,
  status         TEXT   NOT NULL DEFAULT 'open'
                 CHECK (status IN ('open', 'settled', 'voided', 'expired')),
  -- session | recon_job | export_job | recording_job | agent_run | gen_asset
  ref_type       TEXT   NOT NULL,
  ref_id         TEXT   NOT NULL,
  -- modal.FunctionCall.object_id, so the reaper can turn "this hold is stale"
  -- into "this call is genuinely dead" rather than guessing.
  modal_call_id  TEXT,
  reason         TEXT,
  expires_at     TIMESTAMPTZ NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  settled_at     TIMESTAMPTZ
);

-- The durable replacement for the in-process guards that die with the container
-- (`any_active_scene_job` in oreos/jobs.py, `_ACTIVE` in routes_export_job.py).
-- Also what contains a hold storm from the retry loop in the web client's
-- waitForReadyModalGpuSession.
CREATE UNIQUE INDEX IF NOT EXISTS credit_holds_ref
  ON credit_holds (ref_type, ref_id) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS credit_holds_open_expiry
  ON credit_holds (expires_at) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS credit_holds_user
  ON credit_holds (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS credit_ledger (
  id              BIGSERIAL PRIMARY KEY,
  user_id         TEXT   NOT NULL REFERENCES credit_accounts(user_id) ON DELETE CASCADE,
  kind            TEXT   NOT NULL CHECK (kind IN (
                    'migration', 'grant', 'purchase', 'subscription_grant',
                    'hold', 'settle', 'void', 'spend', 'refund', 'adjustment',
                    'expire'
                  )),
  delta           BIGINT NOT NULL,  -- signed credits; hold/void rows are 0
  hold_id         UUID REFERENCES credit_holds(id) ON DELETE SET NULL,
  ref_type        TEXT,
  ref_id          TEXT,
  idempotency_key TEXT   NOT NULL,
  -- Measured provider cost, when known. Carried from the phase-0 instrumentation
  -- so reconciliation against a real invoice is a query, not archaeology.
  usd_cost        NUMERIC(12,6),
  -- openrouter_usage | openrouter_usage_partial | unreported | modal_rate_estimate
  -- | modal_rate_reconciled | fal_flat | estimate
  cost_basis      TEXT,
  metadata        JSONB  NOT NULL DEFAULT '{}',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Stripe redelivers webhooks; Modal retries preempted calls; the client retries
-- a lost response. Every write path passes an idempotency key and this index is
-- what makes the retry a no-op instead of a double charge.
CREATE UNIQUE INDEX IF NOT EXISTS credit_ledger_idem
  ON credit_ledger (user_id, idempotency_key);
CREATE INDEX IF NOT EXISTS credit_ledger_user_time
  ON credit_ledger (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS credit_ledger_ref
  ON credit_ledger (ref_type, ref_id);
-- Reconciliation: sum usd_cost by surface over a window.
CREATE INDEX IF NOT EXISTS credit_ledger_cost_window
  ON credit_ledger (created_at) WHERE usd_cost IS NOT NULL;
