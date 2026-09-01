-- The money rules. These live in SQL, not in application code, because there are
-- two callers in two languages (Next.js on Vercel, Flask/Modal in Python) and a
-- rule implemented twice is a rule that will eventually disagree with itself.
--
-- Every function below takes its account row lock FIRST and in the same order, so
-- concurrent callers serialise rather than deadlock. The read and the write of a
-- balance happen inside one statement under that lock, which is what closes the
-- double-spend the old Clerk read-modify-write had.
--
-- Two behaviours that look inconsistent but are deliberate:
--   * credit_hold / credit_spend FAIL CLOSED  — refusing work is the whole point.
--   * credit_settle NEVER FAILS               — a failed settle means we did the
--     work, paid the provider, and have no record of it. It clamps at the
--     overdraft floor and books the shortfall as uncollected instead.

-- ── internal helpers ───────────────────────────────────────────────────────────

-- Zero an expired subscription bucket and record it, so the append-only
-- invariant (balance == SUM(delta)) still holds afterwards. Assumes the caller
-- already holds the account row lock.
CREATE OR REPLACE FUNCTION _credit_expire_subscription(p_user_id TEXT)
RETURNS VOID AS $$
DECLARE
  v_bal BIGINT;
  v_exp TIMESTAMPTZ;
BEGIN
  SELECT balance_subscription, subscription_expires_at
    INTO v_bal, v_exp
    FROM credit_accounts WHERE user_id = p_user_id;

  IF v_exp IS NOT NULL AND v_exp <= now() AND v_bal <> 0 THEN
    UPDATE credit_accounts
       SET balance_subscription = 0,
           subscription_expires_at = NULL,
           version = version + 1,
           updated_at = now()
     WHERE user_id = p_user_id;

    INSERT INTO credit_ledger (user_id, kind, delta, idempotency_key, metadata)
    VALUES (p_user_id, 'expire', -v_bal,
            'expire:' || p_user_id || ':' || extract(epoch from v_exp)::bigint,
            jsonb_build_object('expired_at', v_exp))
    ON CONFLICT (user_id, idempotency_key) DO NOTHING;
  END IF;
END;
$$ LANGUAGE plpgsql;

-- Spendable right now: both buckets minus what open holds have reserved. An
-- expired subscription bucket is excluded even before _credit_expire_subscription
-- has zeroed it, so a stale row can never be spent.
CREATE OR REPLACE FUNCTION credit_available(p_user_id TEXT)
RETURNS BIGINT AS $$
  SELECT COALESCE(
    balance_purchased
      + CASE WHEN subscription_expires_at IS NULL OR subscription_expires_at > now()
             THEN balance_subscription ELSE 0 END
      - held,
    0)
  FROM credit_accounts WHERE user_id = p_user_id;
$$ LANGUAGE sql STABLE;

-- Deduct `p_amount`, subscription bucket first (it expires; purchased credits do
-- not, so spending them last is what the user would choose). Assumes the account
-- row lock is held. Returns credits actually deducted.
CREATE OR REPLACE FUNCTION _credit_deduct(p_user_id TEXT, p_amount BIGINT)
RETURNS BIGINT AS $$
DECLARE
  v_sub  BIGINT;
  v_take BIGINT;
BEGIN
  IF p_amount <= 0 THEN
    RETURN 0;
  END IF;

  SELECT GREATEST(balance_subscription, 0) INTO v_sub
    FROM credit_accounts WHERE user_id = p_user_id;

  v_take := LEAST(v_sub, p_amount);

  UPDATE credit_accounts
     SET balance_subscription = balance_subscription - v_take,
         balance_purchased    = balance_purchased - (p_amount - v_take),
         version = version + 1,
         updated_at = now()
   WHERE user_id = p_user_id;

  RETURN p_amount;
END;
$$ LANGUAGE plpgsql;

-- ── accounts ───────────────────────────────────────────────────────────────────

-- Idempotent. `p_seed` is granted only on first creation, so calling this on
-- every request (or from a Clerk user.created webhook that fires twice) cannot
-- mint credits twice.
CREATE OR REPLACE FUNCTION credit_ensure_account(
  p_user_id   TEXT,
  p_seed      BIGINT  DEFAULT 0,
  p_unlimited BOOLEAN DEFAULT FALSE
) RETURNS BIGINT AS $$
DECLARE
  v_created BOOLEAN := FALSE;
BEGIN
  INSERT INTO credit_accounts (user_id, unlimited)
  VALUES (p_user_id, p_unlimited)
  ON CONFLICT (user_id) DO NOTHING;

  GET DIAGNOSTICS v_created = ROW_COUNT;

  IF v_created AND p_seed <> 0 THEN
    UPDATE credit_accounts
       SET balance_purchased = balance_purchased + p_seed,
           version = version + 1,
           updated_at = now()
     WHERE user_id = p_user_id;

    INSERT INTO credit_ledger (user_id, kind, delta, idempotency_key, metadata)
    VALUES (p_user_id, 'grant', p_seed, 'seed:' || p_user_id,
            jsonb_build_object('note', 'free allowance on account creation'))
    ON CONFLICT (user_id, idempotency_key) DO NOTHING;
  END IF;

  RETURN credit_available(p_user_id);
END;
$$ LANGUAGE plpgsql;

-- ── holds ──────────────────────────────────────────────────────────────────────

-- Reserve credits for work whose true cost is not known until it finishes.
-- Raises `insufficient_credits` when the balance will not cover the estimate.
--
-- Idempotent on (ref_type, ref_id): an already-open hold for the same reference
-- is returned unchanged rather than duplicated. That is what makes a Modal retry
-- or a client reconnect storm safe, and it is enforced by a unique index as well
-- as by this branch.
CREATE OR REPLACE FUNCTION credit_hold(
  p_user_id  TEXT,
  p_amount   BIGINT,
  p_ref_type TEXT,
  p_ref_id   TEXT,
  p_ttl_s    INT,
  p_idem     TEXT DEFAULT NULL
) RETURNS TABLE (hold_id UUID, available BIGINT) AS $$
DECLARE
  v_unlimited BOOLEAN;
  v_avail     BIGINT;
  v_existing  UUID;
  v_id        UUID;
BEGIN
  IF p_amount < 0 THEN
    RAISE EXCEPTION 'invalid_amount' USING ERRCODE = 'P0001';
  END IF;

  SELECT unlimited INTO v_unlimited
    FROM credit_accounts WHERE user_id = p_user_id FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'no_account' USING ERRCODE = 'P0002';
  END IF;

  PERFORM _credit_expire_subscription(p_user_id);

  SELECT id INTO v_existing FROM credit_holds
   WHERE ref_type = p_ref_type AND ref_id = p_ref_id AND status = 'open';

  IF v_existing IS NOT NULL THEN
    RETURN QUERY SELECT v_existing, credit_available(p_user_id);
    RETURN;
  END IF;

  v_avail := credit_available(p_user_id);

  IF NOT v_unlimited AND v_avail < p_amount THEN
    RAISE EXCEPTION 'insufficient_credits' USING ERRCODE = 'P0003';
  END IF;

  INSERT INTO credit_holds (user_id, amount_est, ref_type, ref_id, expires_at)
  VALUES (p_user_id, p_amount, p_ref_type, p_ref_id,
          now() + make_interval(secs => p_ttl_s))
  RETURNING id INTO v_id;

  UPDATE credit_accounts
     SET held = held + p_amount, version = version + 1, updated_at = now()
   WHERE user_id = p_user_id;

  -- delta 0: a hold moves credits from available to held, it does not leave the
  -- account. This is what keeps balance == SUM(delta) true.
  INSERT INTO credit_ledger (user_id, kind, delta, hold_id, ref_type, ref_id, idempotency_key)
  VALUES (p_user_id, 'hold', 0, v_id, p_ref_type, p_ref_id,
          COALESCE(p_idem, 'hold:' || v_id::text))
  ON CONFLICT (user_id, idempotency_key) DO NOTHING;

  RETURN QUERY SELECT v_id, credit_available(p_user_id);
END;
$$ LANGUAGE plpgsql;

-- Extend an open hold. This is how a live session is metered: a small opening
-- block, then one top-up per minute from the loop that already ticks. Holding a
-- full hour of A100 up front would lock out every user with a small balance.
-- Raises `insufficient_credits` when the next minute cannot be covered, which is
-- the signal for the session to end itself.
CREATE OR REPLACE FUNCTION credit_topup_hold(p_hold_id UUID, p_extra BIGINT)
RETURNS BIGINT AS $$
DECLARE
  v_user      TEXT;
  v_status    TEXT;
  v_unlimited BOOLEAN;
BEGIN
  SELECT user_id, status INTO v_user, v_status
    FROM credit_holds WHERE id = p_hold_id FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'no_hold' USING ERRCODE = 'P0004';
  END IF;
  IF v_status <> 'open' THEN
    RAISE EXCEPTION 'hold_not_open' USING ERRCODE = 'P0005';
  END IF;

  SELECT unlimited INTO v_unlimited
    FROM credit_accounts WHERE user_id = v_user FOR UPDATE;

  PERFORM _credit_expire_subscription(v_user);

  IF NOT v_unlimited AND credit_available(v_user) < p_extra THEN
    RAISE EXCEPTION 'insufficient_credits' USING ERRCODE = 'P0003';
  END IF;

  UPDATE credit_holds SET amount_est = amount_est + p_extra WHERE id = p_hold_id;
  UPDATE credit_accounts
     SET held = held + p_extra, version = version + 1, updated_at = now()
   WHERE user_id = v_user;

  RETURN credit_available(v_user);
END;
$$ LANGUAGE plpgsql;

-- Close a hold at its true cost. IDEMPOTENT by construction: the `status = 'open'`
-- guard means a second settle (Modal retrying a preempted call, say) affects zero
-- rows and simply reports the current balance.
--
-- NEVER RAISES on insufficient funds. It clamps at the overdraft floor and writes
-- the shortfall to metadata.uncollected_credits, which gives a queryable bad-debt
-- figure instead of an exception at the one moment we cannot afford one.
CREATE OR REPLACE FUNCTION credit_settle(
  p_hold_id  UUID,
  p_actual   BIGINT,
  p_usd      NUMERIC DEFAULT NULL,
  p_basis    TEXT    DEFAULT NULL,
  p_metadata JSONB   DEFAULT '{}'
) RETURNS BIGINT AS $$
DECLARE
  v_user        TEXT;
  v_status      TEXT;
  v_est         BIGINT;
  v_ref_type    TEXT;
  v_ref_id      TEXT;
  v_unlimited   BOOLEAN;
  v_total       BIGINT;
  v_floor       BIGINT;
  v_chargeable  BIGINT;
  v_charged     BIGINT := 0;
  v_uncollected BIGINT := 0;
BEGIN
  SELECT user_id, status, amount_est, ref_type, ref_id
    INTO v_user, v_status, v_est, v_ref_type, v_ref_id
    FROM credit_holds WHERE id = p_hold_id FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'no_hold' USING ERRCODE = 'P0004';
  END IF;

  IF v_status <> 'open' THEN
    RETURN credit_available(v_user);  -- already settled/voided: no-op
  END IF;

  SELECT unlimited, balance_purchased + balance_subscription, -overdraft_limit
    INTO v_unlimited, v_total, v_floor
    FROM credit_accounts WHERE user_id = v_user FOR UPDATE;

  UPDATE credit_accounts
     SET held = GREATEST(held - v_est, 0), version = version + 1, updated_at = now()
   WHERE user_id = v_user;

  IF NOT v_unlimited AND p_actual > 0 THEN
    v_chargeable := GREATEST(v_total - v_floor, 0);
    v_charged    := LEAST(p_actual, v_chargeable);
    v_uncollected := p_actual - v_charged;
    PERFORM _credit_deduct(v_user, v_charged);
  END IF;

  UPDATE credit_holds
     SET status = 'settled', amount_settled = p_actual, settled_at = now()
   WHERE id = p_hold_id;

  INSERT INTO credit_ledger (
    user_id, kind, delta, hold_id, ref_type, ref_id,
    idempotency_key, usd_cost, cost_basis, metadata
  ) VALUES (
    v_user, 'settle', -v_charged, p_hold_id, v_ref_type, v_ref_id,
    'settle:' || p_hold_id::text, p_usd, p_basis,
    COALESCE(p_metadata, '{}'::jsonb)
      || jsonb_build_object('amount_requested', p_actual)
      || CASE WHEN v_uncollected > 0
              THEN jsonb_build_object('uncollected_credits', v_uncollected)
              ELSE '{}'::jsonb END
      || CASE WHEN v_unlimited
              THEN jsonb_build_object('unlimited', TRUE)
              ELSE '{}'::jsonb END
  ) ON CONFLICT (user_id, idempotency_key) DO NOTHING;

  RETURN credit_available(v_user);
END;
$$ LANGUAGE plpgsql;

-- Release a hold without charging. Idempotent, same guard as settle. Used when a
-- job fails on our side — we eat the cost; partial-refund logic is not worth
-- maintaining forever for a fraction of a cent.
CREATE OR REPLACE FUNCTION credit_void(p_hold_id UUID, p_reason TEXT DEFAULT NULL)
RETURNS BIGINT AS $$
DECLARE
  v_user     TEXT;
  v_status   TEXT;
  v_est      BIGINT;
  v_ref_type TEXT;
  v_ref_id   TEXT;
BEGIN
  SELECT user_id, status, amount_est, ref_type, ref_id
    INTO v_user, v_status, v_est, v_ref_type, v_ref_id
    FROM credit_holds WHERE id = p_hold_id FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'no_hold' USING ERRCODE = 'P0004';
  END IF;

  IF v_status <> 'open' THEN
    RETURN credit_available(v_user);
  END IF;

  UPDATE credit_accounts
     SET held = GREATEST(held - v_est, 0), version = version + 1, updated_at = now()
   WHERE user_id = v_user;

  UPDATE credit_holds
     SET status = 'voided', reason = p_reason, settled_at = now()
   WHERE id = p_hold_id;

  INSERT INTO credit_ledger (
    user_id, kind, delta, hold_id, ref_type, ref_id, idempotency_key, metadata
  ) VALUES (
    v_user, 'void', 0, p_hold_id, v_ref_type, v_ref_id,
    'void:' || p_hold_id::text, jsonb_build_object('reason', p_reason)
  ) ON CONFLICT (user_id, idempotency_key) DO NOTHING;

  RETURN credit_available(v_user);
END;
$$ LANGUAGE plpgsql;

-- ── immediate spend / grant ────────────────────────────────────────────────────

-- Fixed-price work whose cost is known up front — no hold needed. This is what
-- /api/scans/consume becomes. Raises `insufficient_credits`.
--
-- The idempotency key is load-bearing: without it a lost HTTP response on retry
-- charges twice with no concurrency involved at all.
CREATE OR REPLACE FUNCTION credit_spend(
  p_user_id  TEXT,
  p_amount   BIGINT,
  p_ref_type TEXT,
  p_ref_id   TEXT,
  p_idem     TEXT
) RETURNS BIGINT AS $$
DECLARE
  v_unlimited BOOLEAN;
  v_seen      BIGINT;
BEGIN
  IF p_amount < 0 THEN
    RAISE EXCEPTION 'invalid_amount' USING ERRCODE = 'P0001';
  END IF;

  SELECT unlimited INTO v_unlimited
    FROM credit_accounts WHERE user_id = p_user_id FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'no_account' USING ERRCODE = 'P0002';
  END IF;

  SELECT 1 INTO v_seen FROM credit_ledger
   WHERE user_id = p_user_id AND idempotency_key = p_idem;

  IF FOUND THEN
    RETURN credit_available(p_user_id);  -- replay: already charged
  END IF;

  PERFORM _credit_expire_subscription(p_user_id);

  IF NOT v_unlimited AND credit_available(p_user_id) < p_amount THEN
    RAISE EXCEPTION 'insufficient_credits' USING ERRCODE = 'P0003';
  END IF;

  IF NOT v_unlimited THEN
    PERFORM _credit_deduct(p_user_id, p_amount);
  END IF;

  INSERT INTO credit_ledger (user_id, kind, delta, ref_type, ref_id, idempotency_key, metadata)
  VALUES (p_user_id, 'spend',
          CASE WHEN v_unlimited THEN 0 ELSE -p_amount END,
          p_ref_type, p_ref_id, p_idem,
          CASE WHEN v_unlimited THEN jsonb_build_object('unlimited', TRUE) ELSE '{}'::jsonb END)
  ON CONFLICT (user_id, idempotency_key) DO NOTHING;

  RETURN credit_available(p_user_id);
END;
$$ LANGUAGE plpgsql;

-- Credits in: purchases, subscription grants, refunds, manual adjustments.
-- `p_expires_at` non-null routes to the subscription bucket and sets its expiry;
-- null routes to purchased credits, which never expire.
--
-- Stripe redelivers webhooks, so p_idem MUST be the Stripe event id.
CREATE OR REPLACE FUNCTION credit_grant(
  p_user_id    TEXT,
  p_amount     BIGINT,
  p_kind       TEXT,
  p_idem       TEXT,
  p_ref_type   TEXT        DEFAULT NULL,
  p_ref_id     TEXT        DEFAULT NULL,
  p_expires_at TIMESTAMPTZ DEFAULT NULL,
  p_metadata   JSONB       DEFAULT '{}'
) RETURNS BIGINT AS $$
DECLARE
  v_seen BIGINT;
BEGIN
  PERFORM credit_ensure_account(p_user_id, 0, FALSE);

  PERFORM 1 FROM credit_accounts WHERE user_id = p_user_id FOR UPDATE;

  SELECT 1 INTO v_seen FROM credit_ledger
   WHERE user_id = p_user_id AND idempotency_key = p_idem;

  IF FOUND THEN
    RETURN credit_available(p_user_id);  -- webhook redelivery
  END IF;

  IF p_expires_at IS NOT NULL THEN
    -- Subscription bucket. A new period replaces the old expiry rather than
    -- extending it; rollover is deliberately NOT supported (see billing.md).
    UPDATE credit_accounts
       SET balance_subscription = balance_subscription + p_amount,
           subscription_expires_at = p_expires_at,
           version = version + 1, updated_at = now()
     WHERE user_id = p_user_id;
  ELSE
    UPDATE credit_accounts
       SET balance_purchased = balance_purchased + p_amount,
           version = version + 1, updated_at = now()
     WHERE user_id = p_user_id;
  END IF;

  INSERT INTO credit_ledger (user_id, kind, delta, ref_type, ref_id, idempotency_key, metadata)
  VALUES (p_user_id, p_kind, p_amount, p_ref_type, p_ref_id, p_idem, COALESCE(p_metadata, '{}'::jsonb))
  ON CONFLICT (user_id, idempotency_key) DO NOTHING;

  RETURN credit_available(p_user_id);
END;
$$ LANGUAGE plpgsql;

-- ── reaper ─────────────────────────────────────────────────────────────────────

-- Expired open holds. The container that should have settled them was killed
-- (SIGKILL, preemption) so nothing in-process ran. Policy:
--   'void'                — the job never started; release the credits.
--   'settle_at_estimate'  — the job ran; we paid Modal for that compute.
-- The caller decides per hold by cross-checking modal_call_id against Modal, and
-- passes only the ids it has classified.
CREATE OR REPLACE FUNCTION credit_expire_holds(p_grace_s INT DEFAULT 0)
RETURNS TABLE (id UUID, user_id TEXT, ref_type TEXT, ref_id TEXT,
               amount_est BIGINT, modal_call_id TEXT) AS $$
  SELECT h.id, h.user_id, h.ref_type, h.ref_id, h.amount_est, h.modal_call_id
    FROM credit_holds h
   WHERE h.status = 'open'
     AND h.expires_at + make_interval(secs => p_grace_s) < now()
   ORDER BY h.expires_at
   LIMIT 200;
$$ LANGUAGE sql STABLE;

-- ── audit ──────────────────────────────────────────────────────────────────────

-- Every account whose balance disagrees with its own ledger. Should always be
-- empty; run it nightly. Hold and void rows carry delta 0 precisely so this holds.
CREATE OR REPLACE FUNCTION credit_assert_invariant()
RETURNS TABLE (user_id TEXT, balance BIGINT, ledger_sum BIGINT) AS $$
  SELECT a.user_id,
         a.balance_purchased + a.balance_subscription,
         COALESCE(SUM(l.delta), 0)
    FROM credit_accounts a
    LEFT JOIN credit_ledger l ON l.user_id = a.user_id
   GROUP BY a.user_id, a.balance_purchased, a.balance_subscription
  HAVING a.balance_purchased + a.balance_subscription <> COALESCE(SUM(l.delta), 0);
$$ LANGUAGE sql STABLE;
