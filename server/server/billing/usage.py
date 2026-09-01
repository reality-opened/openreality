"""Real provider-reported usage accounting.

Moved here from ``server/oreos/persisted_agent.py`` so the *live* spatial agent
and the persisted scene agent share one implementation. They previously did
not: ``persisted_agent`` asked OpenRouter for usage and recorded real dollars,
while ``spatial_agent`` — which issues far more calls, continuously, for a whole
session — read nothing at all and its spend was invisible.

The honesty rule this class exists to enforce: an unpriced call is never
silently counted as free. ``cost_usd`` is the sum of what the provider actually
reported and ``cost_basis`` says how much of the run that covers.
"""

from __future__ import annotations

import threading
from typing import Any


class UsageTally:
    """Accumulates real usage reported by OpenRouter (``usage: {include: true}`` →
    ``usage.cost`` in USD credits). When the provider reports no usage the tally still
    counts calls; ``cost_usd`` stays the sum of what was actually reported and
    ``cost_basis`` says how much of the run that covers — never a fabricated number."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.priced_calls = 0
        self.by_model: dict[str, int] = {}

    def record(self, model: str, usage: Any) -> None:
        with self._lock:
            self.calls += 1
            self.by_model[model] = self.by_model.get(model, 0) + 1
            if usage is None:
                return
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            cost = getattr(usage, "cost", None)
            if cost is None:
                extra = getattr(usage, "model_extra", None)
                if isinstance(extra, dict):
                    cost = extra.get("cost")
                elif isinstance(usage, dict):
                    prompt = usage.get("prompt_tokens")
                    completion = usage.get("completion_tokens")
                    cost = usage.get("cost")
            if isinstance(prompt, (int, float)):
                self.prompt_tokens += int(prompt)
            if isinstance(completion, (int, float)):
                self.completion_tokens += int(completion)
            if isinstance(cost, (int, float)):
                self.cost_usd += float(cost)
                self.priced_calls += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            if self.priced_calls == self.calls and self.calls > 0:
                basis = "openrouter_usage"
            elif self.priced_calls > 0:
                basis = "openrouter_usage_partial"
            else:
                basis = "unreported"
            return {
                "llm_calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6) if self.priced_calls else None,
                "cost_basis": basis,
                "by_model": dict(self.by_model),
            }


# ── process-level tallies, one per LLM surface ──
#
# Session-scoped work (the spatial agent) owns a private tally so its cost can be
# attributed to one scan. The rest of the LLM clients in this process are lazy
# module singletons — scene report, plan, summary assistant, object enricher — and
# each gets a named tally here so no surface's spend is invisible. Named rather
# than one global so the Phase-0 cost report can say WHICH surface the money went to.

_REGISTRY: dict[str, UsageTally] = {}
_REGISTRY_LOCK = threading.Lock()


def named_tally(name: str) -> UsageTally:
    """Get (or create) the process-wide tally for an LLM surface."""
    with _REGISTRY_LOCK:
        tally = _REGISTRY.get(name)
        if tally is None:
            tally = UsageTally()
            _REGISTRY[name] = tally
        return tally


def all_summaries() -> dict[str, dict[str, Any]]:
    """Every named surface's totals, for the teardown cost record."""
    with _REGISTRY_LOCK:
        items = list(_REGISTRY.items())
    return {name: tally.summary() for name, tally in items if tally.calls > 0}


def log_all(prefix: str = "") -> None:
    """Print one ``[cost]`` line per surface that made a call. Modal's log stream is
    where this gets mined until the ledger lands."""
    for name, summary in all_summaries().items():
        print(
            f"[cost] {prefix}surface={name} llm_calls={summary['llm_calls']} "
            f"prompt_tokens={summary['prompt_tokens']} "
            f"completion_tokens={summary['completion_tokens']} "
            f"cost_usd={summary['cost_usd']} basis={summary['cost_basis']} "
            f"by_model={summary['by_model']}"
        )
