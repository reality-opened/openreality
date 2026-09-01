"""What a unit of work costs us. ONE source of truth.

Before this file the same GPU rate table was copy-pasted into experiment scripts
and had already drifted: A100-80GB is ``2.50`` in
``experiments/exp29_splat_throughput/modal_exp29_remeasure.py`` and ``3.40`` in
``experiments/exp25_recon_gym/scripts/gopt/run_gopt.py`` — a 36% spread on our
single largest cost line, feeding every ``usd_est`` in every ledger in the repo.

⚠️  RATES ARE NOT YET RECONCILED AGAINST A REAL INVOICE.
    ``RATES_RECONCILED`` stays False until someone pulls an actual Modal
    statement and replaces the table below. Until then these are *estimates* and
    every derived figure must be reported as such — that is what ``cost_basis``
    is for. Do not put a price in front of a customer off an unreconciled rate.

Where a rate is disputed we take the HIGHER figure. Under-stating our own COGS
is the expensive direction of the error.
"""

from __future__ import annotations

# Set to True only when the table below has been checked against a real Modal
# invoice. Phase 2 of the billing plan gates on this.
RATES_RECONCILED = False

# Modal on-demand GPU rates, USD per hour. [estimate — see warning above]
GPU_USD_PER_HOUR: dict[str, float] = {
    "H100": 3.95,
    "A100-80GB": 3.40,  # disputed 2.50 vs 3.40 — taking the higher
    "A100-40GB": 2.10,
    "L40S": 1.95,
    "A10G": 1.10,
    "L4": 0.80,
    "T4": 0.59,
}

# CPU-only Modal containers. Rates vary with the cpu=/memory= request, so this is
# a coarse placeholder used only to keep CPU jobs from reading as free.
CPU_USD_PER_HOUR_PER_CORE = 0.036  # [assumption — reconcile]

# Third-party per-call rates.
# fal SAM-3 is the one number here that is genuinely MEASURED: fal's own billing
# metadata returned "Your request will cost $0.005 per request" on a live run
# (server/docs/demo-2026-07/evidence/fal/README.md).
FAL_SAM3_USD_PER_REQUEST = 0.005  # [measured]
# TRELLIS bills per GPU-second on an A100 with no flat rate published; measured
# model time was 15.6–24.5s across 8 live runs, so a variation lands around 2–5¢.
# The exact $/GPU-s is NOT machine-readable — confirm on the fal dashboard.
FAL_TRELLIS_USD_ESTIMATE = 0.035  # [estimate — midpoint of a measured range]

# Measured reference points, kept here so drift is visible when someone re-runs
# them. Source: docs/demo-2026-07/BUILD-LOG.md, one 47s clip, back to back.
# NOTE these are ~1.5–2x the $0.05–0.10/video-minute the GTM capability profile
# assumes; the profile understates reality and should be corrected from here.
MEASURED_RECON_USD_PER_VIDEO_MINUTE = {
    "A10G": 0.152,  # $0.119 for a 47s clip
    "A100-40GB": 0.216,  # $0.169 for a 47s clip
}


def gpu_usd(gpu: str, seconds: float) -> float:
    """Cost of ``seconds`` on ``gpu``. Unknown GPU → 0.0 rather than a guess;
    callers pair this with a ``cost_basis`` so an unpriced run is visibly
    unpriced instead of silently free."""
    rate = GPU_USD_PER_HOUR.get(gpu)
    if rate is None or seconds <= 0:
        return 0.0
    return (float(seconds) / 3600.0) * rate


def cpu_usd(cores: float, seconds: float) -> float:
    """Cost of a CPU-only container run."""
    if cores <= 0 or seconds <= 0:
        return 0.0
    return (float(seconds) / 3600.0) * float(cores) * CPU_USD_PER_HOUR_PER_CORE


def cost_basis_for(gpu: str) -> str:
    """The honesty flag that must travel with any figure derived from this table."""
    if not RATES_RECONCILED:
        return "modal_rate_estimate"
    if gpu not in GPU_USD_PER_HOUR:
        return "unreported"
    return "modal_rate_reconciled"


# ── credit prices ──────────────────────────────────────────────────────────────
#
# Denomination: 1 scan = 100 credits, 1 credit ≈ $0.01 retail. Deliberately NOT
# 1 credit = 1 scan — that resolution cannot bill a 40-second session or a
# $0.0146 agent run, and both are things we now meter.
#
# Hybrid model: FLAT for the things a user thinks about before clicking, METERED
# for the open-ended tail where cost genuinely varies by an order of magnitude.
#
# ⚠️  Every number below is derived from the UNRECONCILED rate table above. They
#     are placeholders with the right shape, not prices. Phase 2 sets them from a
#     real invoice; `RATES_RECONCILED` gates that.

CREDITS_PER_SCAN = 100
FREE_SCANS_ON_SIGNUP = 2
SIGNUP_CREDITS = CREDITS_PER_SCAN * FREE_SCANS_ON_SIGNUP  # 200

# Markup applied to measured COGS to get a retail credit price.
MARKUP = 4.0

# Flat prices.
CREDITS_VIDEO_RECON_PER_MINUTE = 60      # ~$0.15 COGS/min measured
CREDITS_EXPORT = {
    "lerobot": 20,
    "groot": 20,
    "isaac_usd": 60,                     # 8CPU/64GB, ~380s Poisson mesh
}
CREDITS_GEN_ASSET = 16                   # fal SAM-3 ($0.005) + TRELLIS (~$0.035)
CREDITS_RECORDING_INGEST = 150           # A100-80GB recon, 2h ceiling
CREDITS_LIDAR_RECON = 150                # A100-80GB VGGT-SLAM + metric gauge layer, 2h ceiling

# Metered.
CREDITS_SESSION_OPENING_BLOCK = 125      # 5 minutes up front
CREDITS_SESSION_PER_MINUTE = 25          # ~$0.057/min COGS at $3.40/hr
CREDITS_PER_USD_OF_LLM = 400             # agent runs, billed on real usage.cost

# How long a hold may sit open before the reaper considers it orphaned. Generous
# on purpose: a hold that outlives its job costs the user nothing until settled,
# but a hold reaped out from under a running job double-charges them.
HOLD_TTL_SESSION_S = 7200
HOLD_TTL_JOB_S = 10800


def credits_for_llm_usd(usd: float) -> int:
    """Agent spend → credits, rounded up so a sub-credit run is never free."""
    if not usd or usd <= 0:
        return 0
    import math

    return max(1, math.ceil(float(usd) * CREDITS_PER_USD_OF_LLM))


def credits_for_session_seconds(seconds: float) -> int:
    """Live session minutes, rounded up to the minute — matches how the session
    is metered while running, so the settle agrees with the top-ups."""
    if not seconds or seconds <= 0:
        return 0
    import math

    return math.ceil(seconds / 60.0) * CREDITS_SESSION_PER_MINUTE


def credits_for_video_seconds(seconds: float) -> int:
    """Video recon, priced on the input's duration (what the user can see) rather
    than on GPU time (which they cannot)."""
    if not seconds or seconds <= 0:
        return CREDITS_VIDEO_RECON_PER_MINUTE
    import math

    return max(1, math.ceil(seconds / 60.0) * CREDITS_VIDEO_RECON_PER_MINUTE)
