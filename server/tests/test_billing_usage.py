"""Cost accounting (phase 0 of credit billing).

These tests deliberately avoid importing ``openai`` so they run everywhere the
rest of the GPU-free suite runs. The honesty rule is the thing under test: an
unpriced call must never read as a free call.
"""

from types import SimpleNamespace

from server.billing import prices
from server.billing.usage import UsageTally, all_summaries, named_tally


def test_priced_call_records_real_cost():
    tally = UsageTally()
    tally.record(
        "anthropic/claude-sonnet-5",
        SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=0.0012, model_extra=None),
    )
    summary = tally.summary()
    assert summary["llm_calls"] == 1
    assert summary["prompt_tokens"] == 100
    assert summary["completion_tokens"] == 20
    assert summary["cost_usd"] == 0.0012
    assert summary["cost_basis"] == "openrouter_usage"
    assert summary["by_model"] == {"anthropic/claude-sonnet-5": 1}


def test_unpriced_call_is_never_counted_as_free():
    tally = UsageTally()
    tally.record("m", SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=None, model_extra=None))
    summary = tally.summary()
    assert summary["cost_usd"] is None
    assert summary["cost_basis"] == "unreported"
    # tokens still counted — we know the call happened, just not what it cost
    assert summary["llm_calls"] == 1
    assert summary["prompt_tokens"] == 1


def test_partial_pricing_is_flagged():
    """A run where only some calls came back priced must say so, rather than
    reporting the priced subset as if it were the whole run's cost."""
    tally = UsageTally()
    tally.record("a", SimpleNamespace(prompt_tokens=10, completion_tokens=2, cost=0.001, model_extra=None))
    tally.record("b", None)
    summary = tally.summary()
    assert summary["cost_basis"] == "openrouter_usage_partial"
    assert summary["cost_usd"] == 0.001
    assert summary["llm_calls"] == 2


def test_dict_shaped_usage_is_accepted():
    """OpenRouter returns usage as an object, but a dict shape shows up through
    mocks and older SDK paths — both must record."""
    tally = UsageTally()
    tally.record("m", {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.5})
    summary = tally.summary()
    assert summary["prompt_tokens"] == 7
    assert summary["completion_tokens"] == 3
    assert summary["cost_usd"] == 0.5


def test_named_tally_is_a_process_singleton():
    a = named_tally("test_surface_singleton")
    b = named_tally("test_surface_singleton")
    assert a is b
    a.record("m", {"cost": 0.25, "prompt_tokens": 1, "completion_tokens": 1})
    assert named_tally("test_surface_singleton").summary()["cost_usd"] == 0.25


def test_all_summaries_skips_surfaces_that_never_called():
    named_tally("test_surface_silent")
    used = named_tally("test_surface_used")
    used.record("m", {"cost": 0.1, "prompt_tokens": 1, "completion_tokens": 1})
    summaries = all_summaries()
    assert "test_surface_used" in summaries
    assert "test_surface_silent" not in summaries


def test_gpu_cost_uses_the_single_rate_table():
    assert prices.gpu_usd("A100-80GB", 3600) == prices.GPU_USD_PER_HOUR["A100-80GB"]
    assert prices.gpu_usd("A10G", 1800) == prices.GPU_USD_PER_HOUR["A10G"] / 2


def test_unknown_gpu_costs_zero_rather_than_guessing():
    """An unknown GPU must not silently borrow another GPU's rate. Callers pair
    this with ``cost_basis_for`` so the zero is visibly unpriced."""
    assert prices.gpu_usd("B200", 3600) == 0.0
    assert prices.gpu_usd("A10G", 0) == 0.0
    assert prices.gpu_usd("A10G", -5) == 0.0


def test_rates_are_flagged_unreconciled_until_an_invoice_is_checked():
    """Guards the pricing gate: no customer-facing price may be set off this
    table while the flag is False. Flip it only after reconciling a real
    Modal invoice, and this test's expectation changes with it."""
    assert prices.RATES_RECONCILED is False
    assert prices.cost_basis_for("A100-80GB") == "modal_rate_estimate"


def test_disputed_a100_rate_takes_the_higher_figure():
    """exp29 recorded $2.50/hr and exp25 recorded $3.40/hr for the same GPU.
    Under-stating our own COGS is the expensive direction of the error."""
    assert prices.GPU_USD_PER_HOUR["A100-80GB"] == 3.40


def test_measured_recon_cost_exceeds_the_gtm_estimate():
    """The GTM capability profile assumes $0.05-0.10 per video-minute. The only
    real production measurement is well above that ceiling; this test exists so
    that if someone 'corrects' the table back down to the estimate, it fails."""
    assert prices.MEASURED_RECON_USD_PER_VIDEO_MINUTE["A10G"] > 0.10
    assert prices.MEASURED_RECON_USD_PER_VIDEO_MINUTE["A100-40GB"] > 0.10
