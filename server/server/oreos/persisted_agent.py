"""Persisted-scene agent — lean bounded tool loop over the persisted record (W2).

BUILD-PLAN decision 7: a NEW service that wraps, never trusts, the frozen live
``spatial_agent.py``. Its whole world model is the persisted scene record
(facts / report / keyframes / derived_latest); it runs on the CPU broker with no
Solver, no socket, no GPU. Events stream through ``runlog.AgentRun`` (REST run +
1 s polling; see routes_agent.py).

LLM roles (env-configured; slugs verified against the live OpenRouter model list
2026-07-31, and each role chosen from a measured A/B on the canonical scene —
see docs/demo-2026-07/BUILD-LOG.md "SOTA model + GPU pass"):

  pilot     SCENE_AGENT_PILOT_MODEL      default anthropic/claude-sonnet-5
            fallbacks SCENE_AGENT_PILOT_FALLBACKS
            default anthropic/claude-sonnet-4.6,google/gemini-3-flash-preview
  annotator SCENE_AGENT_ANNOTATOR_MODEL  default google/gemini-3-flash-preview
            fallbacks SCENE_AGENT_ANNOTATOR_FALLBACKS
            default openai/gpt-5.6-luna,anthropic/claude-haiku-4.5
  narrator  SCENE_AGENT_NARRATOR_MODEL   default anthropic/claude-opus-5
            fallbacks SCENE_AGENT_NARRATOR_FALLBACKS
            default anthropic/claude-sonnet-5,google/gemini-3-flash-preview
            (built-in fact-only fallback so the run never stalls on an outage)

Output-token budgets (raise these before blaming a model for "bad" output — too
small a budget truncates the JSON mid-string and the phase degrades SILENTLY to
its fact-only fallback):
  SCENE_AGENT_NARRATOR_MAX_TOKENS   default 2000
  SCENE_AGENT_ANNOTATOR_MAX_TOKENS  default 1400

Guardrails (agent-exports.md, enforced structurally — never trusted to the LLM):
  1. Closed-world inventory — prompts receive only the persisted inventory; a
     validator drops findings citing unknown object ids / evidence refs.
  2. Numbers from code — every measurable quantity is computed by ``geometry.py`` /
     ``measure.py``; LLM text carrying unit-bearing digits is dropped/replaced.
  3. Metric gating — metres ONLY when ``derived_latest.kind == "anchor"``; otherwise
     "relative units (uncalibrated)". Every numeric emission carries
     ``metric_state`` + ``units``. No height claims without floor-fit provenance.
  4. Confidence propagation — low-confidence labels read "possible <label>";
     ``human_label`` wins in all prose.
  5. Non-destructive — nothing here mutates a scene. The one ACTING tool
     (``export.build``) starts the existing prepared-export job, which writes only
     under ``derived/``, into the one slot per (scan, format) that job already
     owns; everything else in this module reads. Persistence is the run log.
  6. Per-run budgets — ≤ SCENE_AGENT_MAX_LLM_CALLS LLM calls (default 40), wall
     clock ≤ SCENE_AGENT_MAX_WALL_S (default 300 s), tool allowlist
     SCENE_AGENT_TOOLS (csv; default all), and — for tools declared ``effect="spend"``
     — ≤ SCENE_AGENT_MAX_SPEND_USD of committed container time (default 1.0). There
     is deliberately NO per-action confirmation, so that ceiling is the only brake:
     see ``tool_kernel.DEFAULT_MAX_SPEND_USD``.

Tool layer (Phase 2): tools are registered on a ``tool_kernel.ToolKernel``, which
wraps the live agent's ``ToolRegistry`` with a declared result shape, an effect
(read/annotate/mutate/spend), an execution mode (sync/job), and a projector mapping
a tool RESULT to UI commands. The kernel's policy executor runs in front of every
mutate/spend handler: idempotency (a retry replays, it does not re-spawn), the spend
ceiling, and structured ``precondition_failed`` refusals the agent can both explain
and repair.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Optional

from server.oreos import geometry
from server.oreos import runlog
from server.oreos import tool_kernel
from server.oreos.measure import MeasureError, MetricState, measure_angle, measure_distance, resolve_metric
from server.oreos.runlog import REGISTRY, AgentRun, make_flusher
from server.oreos.schemas import (
    DemoUICommand,
    DescribeSceneArgs,
    EstimateDimensionsArgs,
    ExportBuildArgs,
    ExportBuildResult,
    ExportNarrateArgs,
    KeyFeaturesArgs,
    LabelObjectsArgs,
    ListSceneObjectsArgs,
    MeasureAngleArgs,
    MeasureDistanceArgs,
    PlanPathArgs,
)
from server.oreos.tool_kernel import ToolKernel, precondition_failure

# ── Role defaults (verified live against GET openrouter.ai/api/v1/models 2026-07-31) ──
#
# Chosen PER ROLE from a measured A/B on the canonical scene, not "biggest everywhere"
# (that arm was strictly worst: slowest, ~6x the cost, and ZERO usable annotations).
# See docs/demo-2026-07/BUILD-LOG.md "SOTA model + GPU pass" for the tables.
#
#   pilot     — decides tool calls. Sonnet 5 is the newer Sonnet and CHEAPER than 4.6
#               ($2/$10 vs $3/$15 per Mtok), ran 2-3x faster than Opus 5 (9-27 s vs
#               35-43 s per turn, on stage that is the wait the room watches), and made
#               zero malformed calls where 4.6 retried a failing plan_path twice.
#               HONEST CAVEAT (weakest of the three role comparisons): Opus 5 had a
#               HIGHER tool-invocation rate — it actually called measure_distance /
#               plan_path and recovered after a tool-side failure, where both Sonnets
#               often answered from list_scene_objects alone. Part of that gap is my
#               probe's fault (one instruction referenced "the door", which is not in
#               the closed-world inventory, so declining to plan is arguably the
#               correct grounded behaviour). If a take needs visible tool use more than
#               it needs latency, set SCENE_AGENT_PILOT_MODEL=anthropic/claude-opus-5.
#   annotator — 2 batched VISION calls per run carrying SCENE_AGENT_ANNOTATOR_KEYFRAMES
#               images each, so this role sets the run's latency floor. gemini-3-flash
#               measured FASTEST (~3.6 s/call vs 6.7 s luna, 10.6 s opus), most reliable
#               (4/4 parses), and currently bills $0 on this account. Deliberately NOT
#               upgraded: the evidence did not support it. gpt-5.6-luna is the SOTA
#               alternative (better spatial phrasing, weaker standalone pin labels) and
#               sits first in the fallback chain.
#   narrator  — exactly ONE call per run and its prose is what the founder reads on
#               screen, so this is the one role where flagship rates are rational.
#               Opus 5 produced the richest grounded paragraphs; Sonnet 5 backs it up.
#
# NOTE (measured): claude-sonnet-4.6 CANNOT serve the narrator role — it emits invalid
# JSON on the real 95-object prompt even with a large token budget, so the old default
# silently fell back to the persisted report on every run. Do not put it back.
DEFAULT_PILOT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_PILOT_FALLBACKS = ["anthropic/claude-sonnet-4.6", "google/gemini-3-flash-preview"]
DEFAULT_ANNOTATOR_MODEL = "google/gemini-3-flash-preview"
DEFAULT_ANNOTATOR_FALLBACKS = ["openai/gpt-5.6-luna", "anthropic/claude-haiku-4.5"]
DEFAULT_NARRATOR_MODEL = "anthropic/claude-opus-5"
DEFAULT_NARRATOR_FALLBACKS = ["anthropic/claude-sonnet-5", "google/gemini-3-flash-preview"]

# Output-token budgets. These are NOT cosmetic: the shipped 600/700 values truncated the
# real canonical prompt mid-JSON (finish_reason="length"), the parse then failed, and the
# phase silently degraded to its fact-only fallback — which is why every roster produced
# byte-identical description prose before this was raised. Measured need on the canonical
# scene: narrator 473-904 completion tokens, annotator ~250-600.
DEFAULT_NARRATOR_MAX_TOKENS = 2000
DEFAULT_ANNOTATOR_MAX_TOKENS = 1400

DEFAULT_MAX_LLM_CALLS = 40
DEFAULT_MAX_WALL_S = 300.0
DEFAULT_PILOT_MAX_STEPS = 6
# A tool-shaped chat turn needs list -> measure -> final, so 3 left ZERO slack: one
# JSON-shape retry consumed the last step and the turn ended on "Step budget reached"
# instead of an answer. Each step is one small call (~$0.00002), so the headroom is
# cheap and the failure it prevents is the user-visible one.
DEFAULT_CHAT_MAX_STEPS = 5
CONFIDENCE_POSSIBLE_THRESHOLD = 0.35

# Unit-bearing-digit detector (guardrail 2): LLM prose may never carry its own
# metric quantities. "in" is deliberately absent (false-positive prone).
_METRIC_CLAIM_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|km|m|metres?|meters?|ft|feet|foot|inch(?:es)?|°|degrees?)\b",
    re.IGNORECASE,
)

# Chat-intent heuristic: these route a chat turn through the tool loop instead of
# plain grounded QA.
_TOOL_INTENT_RE = re.compile(
    r"\b(measure|distance|far|apart|angle|path|route|navigate|drive|export|lerobot|isaac|"
    r"dimension|size|wide|width|long|length|tall|height|big)\b",
    re.IGNORECASE,
)

# Test/DI hook: module-level factory ``(role, model, fallbacks, tally) -> client|None``.
LLM_CLIENT_FACTORY: Optional[Callable[..., Any]] = None

# Handle on server/agent/tool_registry.py (reused UNCHANGED per the design doc). The
# loader itself now lives in tool_kernel so this module and the kernel share ONE module
# object — ToolExecutionError has to be a single class or the ``except`` below misses
# refusals raised inside the kernel.
def _tool_registry_module():
    return tool_kernel.tool_registry_module()


def _tool_error() -> type[Exception]:
    return tool_kernel.tool_execution_error()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def role_models() -> dict[str, dict[str, Any]]:
    """The env-resolved model roster (also what the frozen-agent smoke greps)."""
    pilot = os.environ.get("SCENE_AGENT_PILOT_MODEL", DEFAULT_PILOT_MODEL).strip()
    annotator = os.environ.get("SCENE_AGENT_ANNOTATOR_MODEL", DEFAULT_ANNOTATOR_MODEL).strip()
    # The narrator is its own role with its own default (it used to inherit the pilot's
    # model, which coupled prose quality to tool-calling choice — two different jobs).
    narrator = os.environ.get("SCENE_AGENT_NARRATOR_MODEL", DEFAULT_NARRATOR_MODEL).strip()
    return {
        "pilot": {
            "model": pilot,
            "fallbacks": _env_csv("SCENE_AGENT_PILOT_FALLBACKS", DEFAULT_PILOT_FALLBACKS),
        },
        "annotator": {
            "model": annotator,
            "fallbacks": _env_csv(
                "SCENE_AGENT_ANNOTATOR_FALLBACKS", DEFAULT_ANNOTATOR_FALLBACKS
            ),
        },
        "narrator": {
            "model": narrator,
            "fallbacks": _env_csv(
                "SCENE_AGENT_NARRATOR_FALLBACKS", DEFAULT_NARRATOR_FALLBACKS
            ),
        },
    }


# What degraded, in the words the fallback prose is allowed to use. "LLM unavailable"
# is a claim about an outage; a model that answered in prose, or a QA path that never
# reached a model, are different failures and saying otherwise sends whoever reads the
# transcript to the wrong place.
_FALLBACK_CAUSES = {
    "llm_unavailable": "LLM unavailable",
    "bad_format": "model answered but broke format",
    "call_failed": "model call failed",
    "qa_unavailable": "grounded QA unavailable",
}


# ── LLM plumbing: budget, usage tally, per-role roster ──


def _is_json_shape_error(exc: BaseException) -> bool:
    """Did the model reply in the wrong SHAPE (prose instead of JSON), rather than fail?

    `chat_json` raises a bare RuntimeError("No JSON object found in LLM response") when
    its brace-matcher finds nothing, and json.JSONDecodeError when the blob is malformed.
    Both mean the provider answered fine and the model forgot the contract — a retryable
    slip, not an outage.
    """
    if isinstance(exc, json.JSONDecodeError):
        return True
    text = str(exc).lower()
    return "no json object found" in text or "expecting" in text and "delimiter" in text


class BudgetExceededError(RuntimeError):
    """The per-run LLM-call budget is spent — callers degrade to fact-only output."""


class LLMUnavailableError(RuntimeError):
    """No client for this role (no key / disabled) — callers degrade to fact-only."""


def is_format_error(exc: BaseException) -> bool:
    """True for the provider client's ``LLMFormatError`` — the model answered, but not
    as JSON.

    Matched by name (the same trick ``openrouter_client._is_permanent_model_error``
    uses on provider exceptions) so this module stays importable without the provider
    SDK: the roster imports ``OpenRouterClient`` lazily precisely so a GPU-free, key-free
    environment can still run the agent.
    """
    return type(exc).__name__ == "LLMFormatError" and isinstance(
        getattr(exc, "content", None), str
    )


# ``UsageTally`` now lives in ``server.billing.usage`` so the live spatial agent —
# which issues far more calls than this one — shares the same accounting instead of
# discarding its usage. Re-exported here because this module was its original home.
from server.billing.usage import UsageTally  # noqa: E402


def _make_openrouter_client(model: str, fallbacks: list[str], tally: UsageTally):
    """Real client factory. The base client asks OpenRouter to include usage
    accounting whenever a sink is attached, so this no longer needs a subclass.
    Lazy import so the oreos package (and its tests) never require the ``openai``
    package unless a real LLM call is about to happen."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    from server.llm.openrouter_client import OpenRouterClient

    return OpenRouterClient(
        api_key=api_key,
        primary_model=model,
        fallback_models=fallbacks,
        timeout=30.0,
        app_name="OpenReality Scene Agent",
        max_retries=1,
        usage_sink=tally,
    )


def _default_llm_factory(role: str, model: str, fallbacks: list[str], tally: UsageTally):
    if os.environ.get("SCENE_AGENT_DISABLE_LLM", "").strip() == "1":
        return None
    try:
        return _make_openrouter_client(model, fallbacks, tally)
    except Exception as exc:
        print(f"[demo.agent] LLM client init failed for role {role}: {exc}")
        return None


class LLMRoster:
    """Per-role lazy clients + the shared call budget. ``chat_json(role, …)`` is the
    only way agent code talks to a model, so counting/budgeting can't be bypassed."""

    def __init__(
        self,
        budget_calls: int,
        tally: UsageTally,
        factory: Optional[Callable[..., Any]] = None,
    ):
        self._roles = role_models()
        self._tally = tally
        self._factory = factory or LLM_CLIENT_FACTORY or _default_llm_factory
        self._clients: dict[str, Any] = {}
        self._budget_calls = int(budget_calls)
        self._used = 0
        self._lock = threading.Lock()

    @property
    def calls_used(self) -> int:
        with self._lock:
            return self._used

    def _spend(self) -> None:
        with self._lock:
            if self._used >= self._budget_calls:
                raise BudgetExceededError(
                    f"per-run LLM budget ({self._budget_calls} calls) exhausted"
                )
            self._used += 1

    def client(self, role: str):
        if role not in self._clients:
            cfg = self._roles.get(role) or {}
            try:
                self._clients[role] = self._factory(
                    role, cfg.get("model", ""), list(cfg.get("fallbacks") or []), self._tally
                )
            except Exception as exc:
                print(f"[demo.agent] LLM factory failed for role {role}: {exc}")
                self._clients[role] = None
        return self._clients[role]

    def chat_json(self, role: str, **kwargs) -> tuple[dict[str, Any], Any]:
        client = self.client(role)
        if client is None:
            raise LLMUnavailableError(f"no LLM client for role {role!r}")
        self._spend()
        return client.chat_json(**kwargs)

    def roster_payload(self) -> dict[str, Any]:
        return {role: cfg["model"] for role, cfg in self._roles.items()}


# ── Guardrail helpers ──


def strip_metric_claims(text: Any) -> Optional[str]:
    """Guardrail 2 enforcement for prose: returns the text unchanged when it carries
    no unit-bearing digits, else ``None`` (caller drops it / uses the computed
    template). Replacement-not-mutation: we never rewrite model text into different
    model text."""
    if not isinstance(text, str) or not text.strip():
        return None
    return None if _METRIC_CLAIM_RE.search(text) else text.strip()


def format_length(value_slam: float, metric: MetricState) -> str:
    """A length for prose — computed by code, unit-gated (guardrail 2+3)."""
    if metric.anchored and metric.scale_factor:
        return f"{value_slam * metric.scale_factor:.2f} m"
    return f"{value_slam:.2f} units (relative, uncalibrated)"


class FindingValidator:
    """Closed-world validator (guardrail 1): findings must cite object ids from the
    persisted inventory and evidence refs that exist on the record."""

    def __init__(self, record: dict[str, Any]):
        self.known_uids = geometry.known_uids(record, include_dismissed=True)
        refs: set[tuple[int, int]] = set()
        for kf in record.get("keyframes") or []:
            if isinstance(kf, dict):
                try:
                    refs.add((int(kf.get("submap_id")), int(kf.get("frame_idx"))))
                except (TypeError, ValueError):
                    continue
        facts = record.get("facts") or {}
        for obj in facts.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            for ev in obj.get("evidence") or []:
                if isinstance(ev, dict):
                    try:
                        refs.add((int(ev.get("submap_id")), int(ev.get("frame_idx"))))
                    except (TypeError, ValueError):
                        continue
        self.known_refs = refs
        self.dropped = 0

    def object_ok(self, object_uid: Any) -> bool:
        return isinstance(object_uid, str) and object_uid in self.known_uids

    def refs_ok(self, evidence: Any) -> bool:
        if not evidence:
            return True  # citing nothing is allowed; citing a fake ref is not
        if not isinstance(evidence, list):
            return False
        for ev in evidence:
            if not isinstance(ev, dict):
                return False
            try:
                key = (int(ev.get("submap_id")), int(ev.get("frame_idx")))
            except (TypeError, ValueError):
                return False
            if key not in self.known_refs:
                return False
        return True

    def admit(self, object_uid: Any, evidence: Any = None) -> bool:
        if self.object_ok(object_uid) and self.refs_ok(evidence):
            return True
        self.dropped += 1
        return False


# ── Projectors: tool RESULT → UI commands ──
#
# Guardrail 2 in one place. Every argument these emit is read off the tool's own result
# dict, which was computed by code — the LLM never supplies a coordinate, a length or a
# format. Callables rather than static per-tool lists because the mapping is conditional:
# plan_path emits TWO commands and only when the planner returned waypoints, and the
# export tools emit a repair toast only when a gate blocked them.


def _project_measure_distance(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "show_measurement",
            {
                "kind": "distance",
                "points_world": result.get("points_world") or [],
                "value": result.get("value"),
                "units": result.get("units"),
                "label": result.get("label") or result.get("display"),
            },
        )
    ]


def _project_measure_angle(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "show_measurement",
            {
                "kind": "angle",
                "points_world": result.get("points_world") or [],
                "value": result.get("value"),
                "units": result.get("units"),
                "degrees": result.get("degrees"),
                "label": result.get("label"),
            },
        )
    ]


def _project_plan_path(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Two commands from one result — draw it, then offer to drive it — and none at all
    when the planner came back empty (an unreachable goal must not leave a ghost path)."""
    waypoints = result.get("waypoints_world") or result.get("waypoints") or []
    if not waypoints:
        return []
    path_id = str(result.get("path_id") or f"agent-path-{uuid.uuid4().hex[:8]}")
    return [
        ("show_path", {"points": waypoints, "path_id": path_id}),
        ("drive_path", {"path_id": path_id, "waypoints_world": waypoints}),
    ]


def _project_export_narrate(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    commands: list[tuple[str, dict[str, Any]]] = [
        ("open_export", {"format": result.get("format")})
    ]
    if result.get("anchor_cta"):
        commands.append(("show_toast", {"message": f"{result.get('reason')}", "level": "info"}))
    return commands


# ── export.build: cost model, projection, job event ──
#
# Budget figures, not bills. They exist so the spend ceiling means something, and they are
# deliberate OVER-estimates so the ceiling errs toward refusing: measured container time
# (dataset zip ~45 s on cpu=1/4 GB; Isaac's Poisson mesh 378-382 s on 8 CPU/64 GB) priced
# at roughly Modal's CPU+memory rates and then rounded up hard. If real accounting ever
# lands, replace these with it — do not tune them to make a run fit.
EXPORT_BUILD_COST_USD = {
    "openreality": 0.01,
    "groot_lerobot_v2": 0.01,
    "isaac_usd": 0.30,
}
DEFAULT_EXPORT_BUILD_COST_USD = 0.05


def _isaac_gate_resolver() -> Optional[Callable[..., Any]]:
    """``server.app._resolve_isaac_scale`` — the LIVE Isaac gate, reached through the
    package's lazy ``app_module()`` indirection (the same one ``routes_export.py`` uses,
    so the gate logic lives in exactly one place). ``None`` when it is not reachable."""
    try:
        from server.oreos import app_module

        resolve = getattr(app_module(), "_resolve_isaac_scale", None)
    except Exception:
        return None
    return resolve if callable(resolve) else None


def _export_build_cost(args: ExportBuildArgs) -> float:
    return float(EXPORT_BUILD_COST_USD.get(str(args.format), DEFAULT_EXPORT_BUILD_COST_USD))


def _export_build_settle(result: dict[str, Any], reserved: float) -> float:
    """Book the reservation ONLY when a container was actually spawned. An already-ready
    archive, or a handle onto a build that was already running, cost nothing new — and
    charging for them would burn the run's ceiling on work it did not cause."""
    return reserved if result.get("status") == "queued" else 0.0


def _project_export_build(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Open the export panel on the right tab either way; add a repair toast when a gate
    refused, so the operator sees WHY nothing is building."""
    commands: list[tuple[str, dict[str, Any]]] = [
        ("open_export", {"format": result.get("format")})
    ]
    if result.get("error") or result.get("status") == "blocked":
        message = str(
            result.get("reason") or result.get("message") or "Export could not be started"
        )
        repair = result.get("repair_tool")
        if repair:
            message = f"{message} (repair: {repair})"
        commands.append(("show_toast", {"message": message, "level": "info"}))
    return commands


def _export_build_job_event(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The ``agent_job_event`` that lets the client's run coordinator keep watching this
    build AFTER the agent run ends — the whole point of returning a handle instead of
    blocking. ``active`` carries a handle too (someone else's build, same archive)."""
    job_id = result.get("job_id")
    if not job_id or result.get("status") not in ("queued", "active"):
        return None
    return {
        "job_id": str(job_id),
        "job_name": "demo_export_job",
        "kind": "export",
        "status": "queued" if result.get("status") == "queued" else "running",
        "format": result.get("format"),
        "poll_route": result.get("poll_route") or f"/api/workspace/jobs/{job_id}",
        "estimated_cost_usd": result.get("estimated_cost_usd"),
    }


# ── The agent ──


class PersistedSceneAgent:
    """One run's agent instance over one persisted scene record."""

    def __init__(
        self,
        store: Any,
        user_id: str,
        scan_id: str,
        record: dict[str, Any],
        run: AgentRun,
        llm_factory: Optional[Callable[..., Any]] = None,
    ):
        self.store = store
        self.user_id = str(user_id)
        self.scan_id = str(scan_id)
        self.record = record if isinstance(record, dict) else {}
        self.run = run
        self.tally = UsageTally()
        self.llm = LLMRoster(
            budget_calls=_env_int("SCENE_AGENT_MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS),
            tally=self.tally,
            factory=llm_factory,
        )
        self.metric = resolve_metric(self.record, self._load_depth_provenance())
        self.validator = FindingValidator(self.record)
        self.findings_emitted = 0
        self._tool_allowlist = self._resolve_tool_allowlist()
        self.registry = self._build_tool_registry()

    # -- infrastructure ------------------------------------------------

    def _load_depth_provenance(self) -> Optional[dict[str, Any]]:
        try:
            raw = self.store.get_derived_artifact(
                self.user_id, self.scan_id, "derived/demo/metric/provenance.json"
            )
        except Exception:
            return None
        if not raw:
            return None
        try:
            doc = json.loads(raw.decode("utf-8"))
            return doc if isinstance(doc, dict) else None
        except Exception:
            return None

    def _resolve_tool_allowlist(self) -> Optional[set[str]]:
        raw = os.environ.get("SCENE_AGENT_TOOLS", "").strip()
        if not raw:
            return None  # all tools
        return {name.strip() for name in raw.split(",") if name.strip()}

    def _event_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _emit_thought(
        self,
        content: str,
        phase: Optional[str],
        thought_type: str = "observation",
        **extra: Any,
    ) -> None:
        payload = {
            "id": self._event_id(),
            "timestamp": time.time(),
            "type": thought_type,
            "content": content,
            "phase": phase,
        }
        payload.update(extra)
        self.run.emit("agent_thought", payload, phase=phase)

    def _emit_finding(
        self,
        query: str,
        description: str,
        confidence: float,
        phase: Optional[str],
        object_uid: Optional[str] = None,
        position: Optional[list[float]] = None,
        evidence: Optional[list[dict[str, Any]]] = None,
        **extra: Any,
    ) -> bool:
        """Emit one finding THROUGH the closed-world validator. Findings tied to an
        object must cite a known uid; free findings (room-level) pass ``object_uid=None``
        but their evidence refs are still checked."""
        if object_uid is not None and not self.validator.admit(object_uid, evidence):
            return False
        if object_uid is None and not self.validator.refs_ok(evidence):
            self.validator.dropped += 1
            return False
        payload = {
            "id": self._event_id(),
            "timestamp": time.time(),
            "query": query,
            "description": description,
            "confidence": round(float(confidence), 3),
            "position": position,
            "object_uid": object_uid,
            "evidence": {"refs": evidence or []},
        }
        payload.update(extra)
        self.run.emit("agent_finding", payload, phase=phase)
        self.findings_emitted += 1
        return True

    def _emit_ui(self, name: str, args: dict[str, Any], phase: Optional[str]) -> None:
        """UI commands are validated against the demo SUPERSET schema (never the frozen
        live union) before they hit the feed."""
        command = DemoUICommand(id=self._event_id(), name=name, args=args)
        self.run.emit("agent_ui_command", command.model_dump(), phase=phase)

    def _emit_tool_event(
        self,
        tool: str,
        status: str,
        phase: Optional[str],
        event_id: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[int] = None,
    ) -> str:
        eid = event_id or self._event_id()
        self.run.emit(
            "agent_tool_event",
            {
                "id": eid,
                "tool": tool,
                "status": status,
                "args": args,
                "result": result,
                "error": error,
                "latency_ms": latency_ms,
            },
            phase=phase,
        )
        return eid

    def _emit_state(self, phase: Optional[str], **overrides: Any) -> None:
        labels = [geometry.object_label(o) for _uid, o in geometry.iter_objects(self.record)]
        stats = geometry.scene_stats(self.record)
        report = self.record.get("report") or {}
        payload = {
            "enabled": True,
            "scene_description": str(report.get("summary", "")),
            "room_type": str(report.get("room_type", "unknown")),
            "missions": [],
            "active_queries": [],
            "discovered_objects": sorted(set(labels)),
            "current_goal": None,
            "submaps_processed": stats["num_submaps"],
            "coverage_estimate": stats["coverage_estimate"],
        }
        payload.update(overrides)
        self.run.emit("agent_state", payload, phase=phase)

    # -- evidence imagery (keyframes, else synthetic views) ------------
    #
    # Two sources, never merged. A captured keyframe is a photograph of a real room; a
    # synthetic view is a render of the scene's own splat. They are equally usable as
    # VLM input and NOT equally strong as evidence, so the record keeps them in separate
    # fields (``synthetic_views.py``) and everything below carries the distinction
    # through to the emitted claim's ``provenance``. An imported splat has no keyframes
    # at all, which is the whole reason the fallback exists.

    def _keyframe_manifest(self) -> list[dict[str, Any]]:
        return [k for k in (self.record.get("keyframes") or []) if isinstance(k, dict)]

    def _synthetic_manifest(self) -> list[dict[str, Any]]:
        from server.oreos import synthetic_views

        return synthetic_views.manifest(self.record)

    def evidence_source(self) -> str:
        """``"keyframe"`` | ``"synthetic_view"`` | ``"none"`` — what this run can look at.

        Keyframes win whenever they exist: a scan with real capture imagery must never
        silently downgrade to renders of its own reconstruction."""
        if self._keyframe_manifest():
            return "keyframe"
        if self._synthetic_manifest():
            return "synthetic_view"
        return "none"

    def _load_synthetic_b64(self, blob_key: str) -> Optional[str]:
        try:
            blob = self.store.get_derived_artifact(self.user_id, self.scan_id, blob_key)
        except Exception:
            return None
        if not blob:
            return None
        import base64

        return base64.b64encode(blob).decode("ascii")

    def _select_synthetic_frames(self, k: int) -> list[dict[str, Any]]:
        """Up to ``k`` synthetic views in capture order -> ``[{view_id, index,
        image_b64}]``.

        Capture order, not "best first": the client captures a ring plus elevated
        three-quarters, so walking it in order gives the VLM an even sweep of the room
        rather than ``k`` neighbouring views of one wall."""
        frames: list[dict[str, Any]] = []
        for meta in sorted(self._synthetic_manifest(), key=lambda m: int(m.get("index", 0) or 0)):
            if len(frames) >= k:
                break
            b64 = self._load_synthetic_b64(str(meta.get("blob_key") or ""))
            if b64:
                frames.append(
                    {
                        "view_id": str(meta.get("view_id")),
                        "index": int(meta.get("index", len(frames)) or 0),
                        "image_b64": b64,
                    }
                )
        return frames

    def _select_evidence_frames(self, k: int) -> tuple[list[dict[str, Any]], str]:
        """``(frames, source)`` — the images this phase may reason over.

        Keyframe frames keep their ``{submap_id, frame_idx}`` refs (the closed-world
        validator's evidence vocabulary); synthetic frames carry ``{view_id, index}``
        instead, and callers must NOT pass those to the keyframe validator — a synthetic
        view is deliberately not a citable keyframe ref."""
        frames = self._select_annotator_frames(k)
        if frames:
            return frames, "keyframe"
        return self._select_synthetic_frames(k), "synthetic_view"

    def _load_keyframe_b64(self, blob_key: str) -> Optional[str]:
        try:
            blob = self.store.get_keyframe(self.user_id, self.scan_id, blob_key)
        except Exception:
            return None
        if not blob:
            return None
        import base64

        return base64.b64encode(blob).decode("ascii")

    def _select_annotator_frames(self, k: int) -> list[dict[str, Any]]:
        """Up to ``k`` keyframes: evidence frames of the highest-confidence objects
        first, then a spread over the stored manifest. Returns
        ``[{submap_id, frame_idx, image_b64}]`` (loadable frames only)."""
        manifest = self._keyframe_manifest()
        by_ref = {}
        for kf in manifest:
            try:
                by_ref[(int(kf.get("submap_id")), int(kf.get("frame_idx")))] = kf
            except (TypeError, ValueError):
                continue
        ordered: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        objs = sorted(
            (o for _uid, o in geometry.iter_objects(self.record)),
            key=lambda o: float(o.get("confidence", 0.0) or 0.0),
            reverse=True,
        )
        for obj in objs:
            for ev in obj.get("evidence") or []:
                if not isinstance(ev, dict):
                    continue
                try:
                    ref = (int(ev.get("submap_id")), int(ev.get("frame_idx")))
                except (TypeError, ValueError):
                    continue
                if ref in by_ref and ref not in seen:
                    seen.add(ref)
                    ordered.append(ref)
        for kf in manifest:
            try:
                ref = (int(kf.get("submap_id")), int(kf.get("frame_idx")))
            except (TypeError, ValueError):
                continue
            if ref not in seen:
                seen.add(ref)
                ordered.append(ref)
        frames: list[dict[str, Any]] = []
        for ref in ordered:
            if len(frames) >= k:
                break
            blob_key = by_ref.get(ref, {}).get("blob_key")
            if not blob_key:
                continue
            b64 = self._load_keyframe_b64(str(blob_key))
            if b64:
                frames.append({"submap_id": ref[0], "frame_idx": ref[1], "image_b64": b64})
        return frames

    # -- tool registry -------------------------------------------------

    def _build_tool_registry(self):
        """The run's :class:`ToolKernel` — the live ``ToolRegistry`` plus each tool's
        declared effect/execution, its result shape, and the projector that turns its
        RESULT into UI commands (never the LLM's text)."""
        registry = ToolKernel(user_id=self.user_id, scan_id=self.scan_id, max_workers=2)

        def _allow(tool_name: str) -> bool:
            return self._tool_allowlist is None or tool_name in self._tool_allowlist

        def _add(name, description, args_model, handler, aliases=(), **spec):
            registry.register_tool(
                name,
                description,
                args_model,
                handler,
                aliases=tuple(aliases),
                allow=_allow,
                **spec,
            )

        _add(
            "list_scene_objects",
            "List the persisted, reviewed object inventory of this scene (closed world: "
            "these uids are the ONLY objects that exist).",
            ListSceneObjectsArgs,
            self._tool_list_scene_objects,
            effect="read",
        )
        _add(
            "measure_distance",
            "Distance between two endpoints — each an object (uid det:<idx> or label; its "
            "detected center is used) or a world point [x,y,z]. Numbers computed by code; "
            "metres only when the scene is metric-anchored.",
            MeasureDistanceArgs,
            self._tool_measure_distance,
            aliases=("measure.distance",),
            effect="read",
            projector=_project_measure_distance,
        )
        _add(
            "measure_angle",
            "Angle (degrees) at the middle of three world points.",
            MeasureAngleArgs,
            self._tool_measure_angle,
            aliases=("measure.angle",),
            effect="read",
            projector=_project_measure_angle,
        )
        _add(
            "plan_path",
            "Plan a robot path in scanned free space to a goal object (det:<idx>/label) "
            "or world point — planner visualization, not certified navigation. On "
            "scenes without pose gravity an up-axis override is applied and noted. "
            "(Reports itself unavailable when the W4 planner is not in this build.)",
            PlanPathArgs,
            self._tool_plan_path,
            aliases=("path.plan",),
            effect="read",
            projector=_project_plan_path,
        )
        _add(
            "annotate.label_objects",
            "The grounded per-object label inventory (labels phase payload).",
            LabelObjectsArgs,
            self._tool_label_objects,
            effect="annotate",
        )
        _add(
            "annotate.describe_scene",
            "Grounded scene description (persisted report / narrator).",
            DescribeSceneArgs,
            self._tool_describe_scene,
            effect="annotate",
        )
        _add(
            "annotate.estimate_dimensions",
            "Room + object dimensions computed from persisted geometry, metric-gated.",
            EstimateDimensionsArgs,
            self._tool_estimate_dimensions,
            effect="annotate",
        )
        _add(
            "annotate.key_features",
            "Key features of the space, grounded in the detected inventory.",
            KeyFeaturesArgs,
            self._tool_key_features,
            effect="annotate",
        )
        _add(
            "export.lerobot",
            "Narrate the scene's LeRobot export route (scene/observation bundle, "
            "LeRobot v2). Presence/eligibility only — does not build the zip; call "
            "export.build to actually build it.",
            ExportNarrateArgs,
            self._tool_export_lerobot,
            effect="read",
            projector=_project_export_narrate,
        )
        _add(
            "export.isaac",
            "Narrate the scene's Isaac Sim USD export route, honoring the metric gate. "
            "Presence/eligibility only — does not build the zip; call export.build to "
            "actually build it.",
            ExportNarrateArgs,
            self._tool_export_isaac,
            effect="read",
            projector=_project_export_narrate,
        )
        _add(
            "export.build",
            "BUILD this scene's export archive (openreality | groot_lerobot_v2 | "
            "isaac_usd). Costs container time and returns IMMEDIATELY with a job handle "
            "{job_id, status, poll_route} — the archive is not finished when this "
            "returns, so report the handle and move on; never wait or re-call to poll. "
            "isaac_usd requires a metric anchor and is refused without one.",
            ExportBuildArgs,
            self._tool_export_build,
            effect="spend",
            execution="job",
            result_model=ExportBuildResult,
            projector=_project_export_build,
            project_errors=True,
            precondition=self._export_build_precondition,
            cost_estimator=_export_build_cost,
            settle_cost=_export_build_settle,
            job_event=_export_build_job_event,
        )
        return registry

    # -- tool handlers -------------------------------------------------

    def _tool_list_scene_objects(self, args: ListSceneObjectsArgs) -> dict[str, Any]:
        """The closed-world inventory.

        COMPACT BY DEFAULT. Every row used to carry `center` and `extent` — six floats,
        roughly two thirds of the row — and on a 95-object scene that pushed the payload
        past the prompt budget, so most of the inventory never reached the model. Those
        coordinates were never needed here: `measure_distance` and `plan_path` take an
        object REF (uid or label) and resolve its centre server-side via
        `_resolve_endpoint`, precisely so that geometry stays in code. Pass
        `detail="full"` when a caller genuinely wants the boxes.

        `total_objects` is the TRUE count and is always the count of everything in the
        scene, never the count of what fitted. An agent that can only see part of the
        inventory has to be able to tell that it is looking at part of the inventory.
        """
        full = getattr(args, "detail", "compact") == "full"
        all_objects = list(
            geometry.iter_objects(self.record, include_dismissed=args.include_dismissed)
        )
        rows = []
        for uid, obj in all_objects[: args.max_results]:
            row: dict[str, Any] = {
                "uid": uid,
                "label": geometry.object_label(obj),
                "confidence": round(float(obj.get("confidence", 0.0) or 0.0), 3),
            }
            if obj.get("human_label"):
                row["human_label"] = obj.get("human_label")
            if obj.get("dismissed"):
                row["dismissed"] = True
            if full:
                center = geometry.object_center(obj)
                extent = geometry.object_extents(obj)
                row["query"] = obj.get("query")
                row["center"] = [float(v) for v in center] if center is not None else None
                row["extent"] = [float(v) for v in extent] if extent is not None else None
            rows.append(row)

        payload: dict[str, Any] = {
            "objects": rows,
            "count": len(rows),
            "total_objects": len(all_objects),
            "closed_world": True,
            "units": self.metric.units,
            "units_basis": self.metric.units_basis,
        }
        if len(rows) < len(all_objects):
            payload["_truncated"] = {
                "field": "objects",
                "shown": len(rows),
                "total": len(all_objects),
                "note": (
                    f"Only {len(rows)} of {len(all_objects)} objects are listed here. The "
                    f"rest EXIST — do not report them as absent. Raise max_results to see more."
                ),
            }
        return payload

    def _resolve_endpoint(self, endpoint) -> tuple[list[float], dict[str, Any]]:
        ToolExecutionError = _tool_error()

        if endpoint.point is not None:
            return [float(v) for v in endpoint.point], {"kind": "point"}
        if endpoint.object:
            resolved = geometry.resolve_object(self.record, endpoint.object)
            if resolved is None:
                raise ToolExecutionError(
                    f"unknown object {endpoint.object!r} — cite a uid/label from "
                    "list_scene_objects (closed world)"
                )
            uid, obj = resolved
            center = geometry.object_center(obj)
            if center is None:
                raise ToolExecutionError(f"object {uid} has no persisted center")
            return (
                [float(v) for v in center],
                {"kind": "object", "uid": uid, "label": geometry.object_label(obj)},
            )
        raise ToolExecutionError("endpoint needs 'object' or 'point'")

    def _tool_measure_distance(self, args: MeasureDistanceArgs) -> dict[str, Any]:
        ToolExecutionError = _tool_error()

        point_a, meta_a = self._resolve_endpoint(args.a)
        point_b, meta_b = self._resolve_endpoint(args.b)
        try:
            result = measure_distance([point_a, point_b], self.metric)
        except MeasureError as exc:
            raise ToolExecutionError(str(exc)) from exc
        result["endpoints"] = {"a": meta_a, "b": meta_b}
        result["points_world"] = [point_a, point_b]
        result["display"] = format_length(result["value_slam"], self.metric)
        if args.label:
            result["label"] = args.label
        return result

    def _tool_measure_angle(self, args: MeasureAngleArgs) -> dict[str, Any]:
        ToolExecutionError = _tool_error()

        try:
            result = measure_angle(args.points_world, self.metric)
        except MeasureError as exc:
            raise ToolExecutionError(str(exc)) from exc
        result["points_world"] = args.points_world
        if args.label:
            result["label"] = args.label
        return result

    def _tool_plan_path(self, args: PlanPathArgs) -> dict[str, Any]:
        """W4 seam: calls ``routes_nav.plan_path_for_scene`` (feat/demo-pathplan) —
        the same implementation behind ``POST /api/scenes/<id>/nav/plan`` — when that
        module exists in the build. The import stays OPTIONAL so this branch is green
        standalone; the real planner lights up at integration with no W2 change.

        Up-axis rule: scenes without a persisted trajectory have no pose-derived
        gravity, so the planner gets ``params.up_override`` (default ``"-y"``, env
        ``SCENE_AGENT_NAV_UP_OVERRIDE``; explicit tool arg wins) and the result
        carries the honest "manual up-axis override" note. Scenes WITH a trajectory
        pass nothing — pose gravity takes over (W1 re-persist)."""
        ToolExecutionError = _tool_error()

        try:
            from server.oreos import routes_nav  # type: ignore[attr-defined]  # W4 module
        except ImportError:
            routes_nav = None
        planner = getattr(routes_nav, "plan_path_for_scene", None) if routes_nav else None
        if not callable(planner):
            raise ToolExecutionError(
                "plan_path is not available in this build: the W4 planner "
                "(server/oreos/routes_nav.py, branch feat/demo-pathplan) is not merged "
                "here yet. Say so honestly — do not invent a path."
            )

        if args.goal_object:
            resolved = geometry.resolve_object(self.record, args.goal_object)
            if resolved is None:
                raise ToolExecutionError(
                    f"unknown goal object {args.goal_object!r} — cite a det:<idx> uid "
                    "or label from list_scene_objects (only det:<idx> goals resolve)"
                )
            goal: dict[str, Any] = {"object_uid": resolved[0]}
        elif args.goal_point is not None:
            goal = {"point_world": [float(v) for v in args.goal_point]}
        else:
            raise ToolExecutionError("plan_path needs goal_object or goal_point")

        body: dict[str, Any] = {"goal": goal}
        up_note: Optional[str] = None
        override = (
            args.up_override
            or os.environ.get("SCENE_AGENT_NAV_UP_OVERRIDE", "").strip()
            or None
        )
        if override is None and not geometry.scene_stats(self.record)["has_trajectory"]:
            override = "-y"
        if override:
            body["params"] = {"up_override": override}
            up_note = (
                f"using manual up-axis override ({override}) — this scene has no "
                "pose-derived gravity yet"
            )

        payload, status = planner(self.user_id, self.scan_id, body, store=self.store)
        if status == 200 and isinstance(payload, dict):
            result = dict(payload)
            if up_note:
                result["notes"] = list(result.get("notes") or []) + [up_note]
            return result

        err = payload.get("error") if isinstance(payload, dict) else None
        if err == "unreachable_goal":
            nearest = None
            if isinstance(payload, dict):
                nearest = (payload.get("nearest_reachable") or {}).get("point_world")
            raise ToolExecutionError(
                "goal unreachable in scanned free space"
                + (
                    f"; nearest reachable point_world={nearest} — you may retry "
                    f"plan_path with goal_point={nearest}"
                    if nearest
                    else ""
                )
            )
        detail = json.dumps(payload, default=str)[:400] if payload else ""
        raise ToolExecutionError(
            f"plan_path failed ({err or 'error'}, HTTP {status}): {detail}"
        )

    def _tool_label_objects(self, args: LabelObjectsArgs) -> dict[str, Any]:
        listing = self._tool_list_scene_objects(
            ListSceneObjectsArgs(max_results=min(args.max_objects, 500))
        )
        labels = []
        for row in listing["objects"]:
            conf = row["confidence"]
            label = row["label"]
            phrased = label if conf >= CONFIDENCE_POSSIBLE_THRESHOLD else f"possible {label}"
            labels.append({**row, "phrased_label": phrased})
        return {"labels": labels, "count": len(labels), "closed_world": True}

    def _tool_describe_scene(self, args: DescribeSceneArgs) -> dict[str, Any]:
        return self._build_description(max_paragraphs=args.max_paragraphs)

    def _tool_estimate_dimensions(self, args: EstimateDimensionsArgs) -> dict[str, Any]:
        return self._build_dimensions(object_uids=args.object_uids or None)

    def _tool_key_features(self, args: KeyFeaturesArgs) -> dict[str, Any]:
        return self._build_key_features(max_features=args.max_features)

    def _tool_export_lerobot(self, args: ExportNarrateArgs) -> dict[str, Any]:
        stats = geometry.scene_stats(self.record)
        available = stats["has_trajectory"]
        return {
            "format": "groot_lerobot_v2",
            "route": f"/api/scenes/{self.scan_id}/export?format=groot_lerobot_v2",
            "available": available,
            "reason": None
            if available
            else "scan has no persisted per-keyframe trajectory (pre-Stage-4 scan or "
            "imported splat) — the GPU-free export route will answer no_trajectory",
            "manifest_probe": "unavailable — export ?manifest=1 is not built yet; this is "
            "route-presence narration only",
            "copy": (
                "Scene/observation bundle in LeRobot v2 (GR00T N1.5 loads v2; ecosystem "
                "mainline is v3; pin lerobot==0.5.1). The episode encodes the capture "
                "trajectory (camera ego-motion) through a static scene — ego-motion is the "
                "action channel; no robot manipulation is claimed."
            ),
            "metric_state": "metric_anchored" if self.metric.anchored else "up_to_scale",
        }

    def _tool_export_isaac(self, args: ExportNarrateArgs) -> dict[str, Any]:
        stats = geometry.scene_stats(self.record)
        gate_ok = self.metric.anchored
        available = gate_ok and stats["has_trajectory"]
        if gate_ok:
            reason = (
                None
                if stats["has_trajectory"]
                else "scan has no persisted trajectory — Isaac export needs Stage-4 geometry"
            )
        else:
            reason = (
                "metric_scale_required: the Isaac route answers 409 until a metric anchor "
                "is set — run Metric anchor first (or pass ?scale=)"
            )
        return {
            "format": "isaac_usd",
            "route": f"/api/scenes/{self.scan_id}/export?format=isaac_usd",
            "available": available,
            "metric_gate": "passed" if gate_ok else "blocked",
            "reason": reason,
            "anchor_cta": None if gate_ok else "Anchor to metres",
            "scale_source": self.metric.scale_source,
            "note": "requires usd-core/open3d in the deployment image (501 isaac_unavailable "
            "otherwise); semantics attach to proxy meshes, not splats",
            "metric_state": "metric_anchored" if self.metric.anchored else "up_to_scale",
        }

    # -- export.build (the first ACTING tool) --------------------------

    def _export_build_precondition(self, args: ExportBuildArgs) -> Optional[dict[str, Any]]:
        """The Isaac metric gate, evaluated BEFORE anything is spawned or charged.

        The gate itself is not re-implemented here: it is ``server.app``'s own
        ``_resolve_isaac_scale`` — the exact function the zip route and the manifest route
        gate on (they reach it through the same lazy ``app_module()`` indirection). It
        returns ``None`` when the scan has no metric scale, which is the route's 409
        ``metric_scale_required``.

        When the helper is not reachable at all (a deployment or harness whose
        ``server.app`` does not expose it), this FAILS CLOSED. Spawning a $-costing Isaac
        build whose gate we could not evaluate is the one outcome worth refusing over.
        """
        if str(args.format) != "isaac_usd":
            return None
        resolve = _isaac_gate_resolver()
        if resolve is None:
            return precondition_failure(
                "metric_anchor",
                "the Isaac metric gate (server.app._resolve_isaac_scale) is not available "
                "in this build, so an Isaac export cannot be authorised here",
                repair_tool=None,
                format="isaac_usd",
            )
        if resolve(self.record, args.source, None) is None:
            return precondition_failure(
                "metric_anchor",
                "Isaac USD export needs a metric anchor — the export route answers 409 "
                "metric_scale_required until one is set. Run Metric anchor on this scene "
                "first, then ask again.",
                repair_tool="export.isaac",
                format="isaac_usd",
                metric_state="metric_anchored" if self.metric.anchored else "up_to_scale",
                scale_source=self.metric.scale_source,
            )
        return None

    def _tool_export_build(self, args: ExportBuildArgs) -> dict[str, Any]:
        """Start the prepared-export build and return its HANDLE.

        Rides the SAME scheduler as the panel's Prepare button
        (``routes_export_job.prepare_export_for_scene``) — one 409 guard, one ``_ACTIVE``
        table, one slot per (scene, format). It must never block: an Isaac export measured
        382 s against the broker's hard 150 s web-request timeout, and an LLM polling for
        minutes burns tokens while holding the scene's only run slot. So the job's own
        status flows through the existing ``GET /api/workspace/jobs/<job_id>`` route, and
        the client's run coordinator picks it up from the ``agent_job_event``.
        """
        from server.oreos.routes_export_job import prepare_export_for_scene

        payload, status = prepare_export_for_scene(
            self.user_id,
            self.scan_id,
            {"format": args.format, "source": args.source},
            store=self.store,
        )
        estimate = _export_build_cost(args)
        base: dict[str, Any] = {
            "format": str(payload.get("format") or args.format),
            "http_status": int(status),
            "estimated_cost_usd": estimate,
        }

        if status == 202:
            job_id = str(payload.get("job_id") or "")
            return {
                **base,
                "status": "queued",
                "job_id": job_id,
                "poll_route": str(payload.get("poll_route") or f"/api/workspace/jobs/{job_id}"),
                "source_key": payload.get("source_key"),
                "message": "build started; it finishes on its own — do not wait for it",
            }

        if status == 409 and payload.get("job_id"):
            # A build for this scene+format was already running. That is a HANDLE, not an
            # error: hand back the same job id rather than starting a duplicate container.
            job_id = str(payload["job_id"])
            return {
                **base,
                "status": "active",
                "job_id": job_id,
                "poll_route": f"/api/workspace/jobs/{job_id}",
                "message": str(payload.get("message") or "an export is already being prepared"),
            }

        if status == 200 and isinstance(payload.get("artifact"), dict):
            artifact = payload["artifact"]
            return {
                **base,
                "status": "ready",
                "download_path": artifact.get("download_path"),
                "bytes": artifact.get("bytes"),
                "source_key": artifact.get("source_key"),
                "message": str(payload.get("message") or "a current export is already prepared"),
            }

        # Everything else the shared route can answer (400 invalid_format, 404
        # not_found/unknown_derived_key, 502 spawn_failed, 503 no_scene_store) becomes the
        # one structured refusal shape, so the pilot explains it the same way every time.
        return {
            **base,
            "status": "blocked",
            "estimated_cost_usd": None,
            **precondition_failure(
                str(payload.get("error") or "export_unavailable"),
                str(payload.get("message") or payload.get("error") or "export could not be started"),
                repair_tool=None,
            ),
        }

    # -- content builders (numbers from code) --------------------------

    def _build_description(self, max_paragraphs: int = 3) -> dict[str, Any]:
        """Scene description: persisted report first, optional narrator final pass,
        fact-only fallback. Narrator text carrying unit-bearing digits is discarded
        in favor of the grounded report (guardrail 2)."""
        report = self.record.get("report") or {}
        base_summary = str(report.get("summary", "")).strip()
        observations = [
            str(o).strip() for o in (report.get("observations") or []) if str(o).strip()
        ]
        room_type = str(report.get("room_type", "unknown"))

        paragraphs: list[str] = []
        source = "persisted_report"
        degraded = bool(report.get("degraded", False))
        evidence = self.evidence_source()
        evidence_line = {
            "keyframe": "photographic keyframes from the capture",
            "synthetic_view": (
                "synthetic views rendered from this scene's own gaussian splat — there is "
                "no capture video, so never describe anything here as photographed or "
                "filmed; the geometry is real, the imagery is a render of it"
            ),
            "none": "no imagery at all — geometry and the object inventory only",
        }[evidence]
        try:
            facts = self.record.get("facts") or {}
            inventory = [
                {"uid": uid, "label": geometry.object_label(obj)}
                for uid, obj in geometry.iter_objects(self.record)
            ]
            parsed, response = self.llm.chat_json(
                "narrator",
                system_prompt=(
                    "You narrate a 3D scene annotation pass for a spatial intelligence "
                    "system, from STRUCTURED FACTS and a prior grounded report. Write "
                    f"{max_paragraphs} short paragraphs (JSON list). Ground every claim in "
                    "the provided inventory/report; NEVER introduce objects not listed; "
                    "NEVER state numeric lengths, areas, or dimensions — geometry is "
                    "announced separately by measurement tools. Coordinates are up-to-scale."
                    ' Output strict JSON only: {"paragraphs": ["..."]}'
                ),
                user_prompt=(
                    f"ROOM TYPE: {room_type}\n"
                    f"PRIOR REPORT SUMMARY: {base_summary}\n"
                    f"PRIOR OBSERVATIONS: {json.dumps(observations[:8])}\n"
                    f"OBJECT INVENTORY (closed world): {json.dumps(inventory[:80])}\n"
                    f"OBJECT COUNTS: {json.dumps(facts.get('object_counts') or {})}\n"
                    # Stated so the prose cannot call a render a photograph. It is the
                    # narrator's only signal about what this scene is made of.
                    f"EVIDENCE AVAILABLE: {evidence_line}\n"
                ),
                temperature=0.4,
                max_tokens=_env_int(
                    "SCENE_AGENT_NARRATOR_MAX_TOKENS", DEFAULT_NARRATOR_MAX_TOKENS
                ),
            )
            candidate = [
                strip_metric_claims(p)
                for p in (parsed.get("paragraphs") or [])
                if isinstance(p, str)
            ]
            candidate = [p for p in candidate if p]
            if candidate:
                paragraphs = candidate[:max_paragraphs]
                source = f"narrator:{getattr(response, 'model', '')}"
                degraded = bool(getattr(response, "degraded", False))
        except (LLMUnavailableError, BudgetExceededError) as exc:
            print(f"[demo.agent] narrator unavailable ({exc}); using persisted report")
        except Exception as exc:
            print(f"[demo.agent] narrator failed ({exc}); using persisted report")

        if not paragraphs:
            paragraphs = [p for p in ([base_summary] + observations[:2]) if p]
            if not paragraphs:
                stats = geometry.scene_stats(self.record)
                paragraphs = [
                    f"Persisted scene with {stats['active_object_count']} detected objects "
                    f"across {stats['num_submaps']} submaps. No stored report; fact-only "
                    "description."
                ]
                source = "fact_fallback"
                degraded = True

        return {
            "paragraphs": paragraphs,
            "room_type": room_type,
            "source": source,
            "degraded": degraded,
            "provenance": evidence,
        }

    def _build_dimensions(
        self, object_uids: Optional[list[str]] = None, max_objects: int = 8
    ) -> dict[str, Any]:
        """Room + per-object dimension payload. EVERY number computed here from the
        record; metric-gated wording.

        Height talk is gated on a DERIVED ground frame, not on hope: without one (a raw
        VGGT world, or an imported splat whose plane fit was too weak to publish) the
        room block stays raw AABB extents and says out loud that none of them is a
        floor-to-ceiling height. With one, the floor area and ceiling height are real
        derived quantities and are reported as such, still scaled by the same metric
        gate as every other length."""
        metric = self.metric
        room = geometry.room_bbox(self.record)
        room_out = None
        if room is not None:
            dims = room["dimensions"]
            scale = metric.scale_factor if metric.anchored else 1.0
            scaled = [d * scale for d in dims]
            room_out = {
                "dimensions": [round(float(d), 3) for d in scaled],
                "dimensions_slam": [round(float(d), 4) for d in dims],
                "units": metric.units,
                "units_basis": metric.units_basis,
                "metric_state": "metric_anchored" if metric.anchored else "up_to_scale",
                "vertical_axis_known": room["vertical_axis_known"],
                "note": (
                    "principal AABB extents in the SLAM world frame — axes are not "
                    "gravity-aligned, so none of these is a floor-to-ceiling height"
                ),
            }
            frame = room.get("ground_frame")
            if frame:
                room_out["derivation"] = frame["derivation"]
                room_out["note"] = (
                    "floor plane fitted to the reconstruction; heights and footprint are "
                    "measured along the fitted vertical axis"
                )
                if frame.get("floor_extent"):
                    room_out["floor_extent"] = [
                        round(float(v) * scale, 3) for v in frame["floor_extent"]
                    ]
                if frame.get("floor_area") is not None:
                    # Area scales with the SQUARE of a length factor — the one place in
                    # this file where applying `scale` once would be silently wrong.
                    room_out["floor_area"] = round(float(frame["floor_area"]) * scale * scale, 3)
                    room_out["floor_area_units"] = "m2" if metric.anchored else "relative2"
                if frame.get("room_height") is not None:
                    room_out["ceiling_height"] = round(float(frame["room_height"]) * scale, 3)

        rows = []
        if object_uids:
            pool = []
            for ref in object_uids:
                resolved = geometry.resolve_object(self.record, ref)
                if resolved is not None:
                    pool.append(resolved)
        else:
            pool = sorted(
                geometry.iter_objects(self.record),
                key=lambda pair: float(pair[1].get("confidence", 0.0) or 0.0),
                reverse=True,
            )[:max_objects]
        for uid, obj in pool:
            extent = geometry.object_extents(obj)
            if extent is None:
                continue
            scaled = extent * metric.scale_factor if metric.anchored else extent
            rows.append(
                {
                    "object_uid": uid,
                    "label": geometry.object_label(obj),
                    "confidence": round(float(obj.get("confidence", 0.0) or 0.0), 3),
                    "extents": [round(float(v), 3) for v in scaled],
                    "extents_slam": [round(float(v), 4) for v in extent],
                    "longest_span": round(float(max(scaled)), 3),
                    "units": metric.units,
                    "units_basis": metric.units_basis,
                    "metric_state": "metric_anchored" if metric.anchored else "up_to_scale",
                }
            )
        return {
            "room": room_out,
            "objects": rows,
            "units": metric.units,
            "units_basis": metric.units_basis,
            "wording": metric.wording,
            "metric_state": "metric_anchored" if metric.anchored else "up_to_scale",
            "anchor_cta": None if metric.anchored else "Anchor to metres",
        }

    def _build_key_features(self, max_features: int = 5) -> dict[str, Any]:
        """Key features via the annotator VLM over the run's evidence imagery, prompt
        closed over the detected inventory; claims must cite known object uids (validator
        drops the rest). Fact-only fallback: top object types by count.

        Two things change when the imagery is SYNTHETIC (an imported splat, which has no
        keyframes at all):

        * the model is told what it is looking at — a render of a reconstruction, not a
          photograph — because "describe this photo" invites reconstruction artifacts to
          be narrated as real objects;
        * evidence is cited by ``view_id``, not by ``{submap_id, frame_idx}``. A synthetic
          view is deliberately NOT a citable keyframe ref, so those refs never reach the
          closed-world keyframe validator and can never be mistaken for capture evidence
          downstream.

        When the inventory is EMPTY (the normal state of a freshly imported splat, before
        object detection has run) features become room-level observations grounded on the
        views alone: uid-free, view-cited, and carrying the synthetic-view provenance. The
        alternative — dropping every claim for lack of a uid to cite — would leave the
        imported scene with a silent annotation pass, which is exactly the degradation
        this workstream exists to remove."""
        inventory = [
            {"uid": uid, "label": geometry.object_label(obj)}
            for uid, obj in geometry.iter_objects(self.record)
        ]
        features: list[dict[str, Any]] = []
        source = "fact_fallback"
        provenance = "none"
        try:
            frames, provenance = self._select_evidence_frames(
                _env_int("SCENE_AGENT_ANNOTATOR_KEYFRAMES", 6)
            )
            synthetic = provenance == "synthetic_view"
            if synthetic:
                frame_index = [{"view_id": f["view_id"], "index": f["index"]} for f in frames]
                imagery_line = (
                    "IMAGES are SYNTHETIC VIEWS rendered from this scene's own gaussian "
                    "splat — not photographs. Describe the space they show; do not "
                    "describe rendering artifacts (holes, floaters, blur) as objects.\n"
                    f"VIEWS attached in this order: {json.dumps(frame_index)}"
                )
                cite_line = (
                    'cite the views that support it as "view_ids": ["<view_id>"]'
                    + (
                        ' and the object uids as "object_uids"'
                        if inventory
                        else ' (this scene has no detected object inventory yet, so omit "object_uids")'
                    )
                )
            else:
                frame_index = [
                    {"submap_id": f["submap_id"], "frame_idx": f["frame_idx"]} for f in frames
                ]
                imagery_line = f"KEYFRAMES attached in this order: {json.dumps(frame_index)}"
                cite_line = (
                    'cite the object uids as "object_uids" and optionally the keyframes as '
                    '"keyframes": [{"submap_id": 0, "frame_idx": 0}]'
                )
            parsed, response = self.llm.chat_json(
                "annotator",
                system_prompt=(
                    "You identify the key features of a scanned 3D space from its detected "
                    "object inventory and a few images of it. CLOSED WORLD: you may only "
                    "cite object uids from the inventory — never introduce objects. Never "
                    f"state numeric lengths or dimensions. For each feature, {cite_line}. "
                    'Output strict JSON only: {"features": [{"claim": "...", '
                    '"object_uids": [], "view_ids": [], "keyframes": []}]}'
                ),
                user_prompt=(
                    f"OBJECT INVENTORY (closed world): {json.dumps(inventory[:80])}\n"
                    f"{imagery_line}\n"
                    f"Return up to {max_features} features."
                ),
                images_b64=[f["image_b64"] for f in frames] or None,
                temperature=0.3,
                max_tokens=_env_int(
                    "SCENE_AGENT_ANNOTATOR_MAX_TOKENS", DEFAULT_ANNOTATOR_MAX_TOKENS
                ),
            )
            known_views = {f["view_id"] for f in frames} if synthetic else set()
            for raw in (parsed.get("features") or [])[: max_features * 2]:
                if not isinstance(raw, dict):
                    continue
                claim = strip_metric_claims(raw.get("claim"))
                uids = raw.get("object_uids") or []
                refs = raw.get("keyframes")
                if not claim or not isinstance(uids, list):
                    self.validator.dropped += 1
                    continue
                # A uid-free claim is admissible ONLY when there is no inventory to cite.
                if not uids and inventory:
                    self.validator.dropped += 1
                    continue
                if not all(self.validator.object_ok(u) for u in uids):
                    self.validator.dropped += 1
                    continue
                view_ids = [
                    str(v) for v in (raw.get("view_ids") or []) if str(v) in known_views
                ]
                if not synthetic and not self.validator.refs_ok(refs):
                    self.validator.dropped += 1
                    continue
                features.append(
                    {
                        "claim": claim,
                        "object_uids": uids,
                        # Keyframe refs exist only on the captured path; a synthetic run
                        # cites views instead, and never both.
                        "keyframes": [] if synthetic else (refs or []),
                        "view_ids": view_ids,
                        "provenance": provenance,
                    }
                )
                if len(features) >= max_features:
                    break
            if features:
                source = f"annotator:{getattr(response, 'model', '')}"
        except (LLMUnavailableError, BudgetExceededError) as exc:
            print(f"[demo.agent] key-features annotator unavailable ({exc}); fact fallback")
        except Exception as exc:
            print(f"[demo.agent] key-features annotator failed ({exc}); fact fallback")

        if not features:
            facts = self.record.get("facts") or {}
            counts = facts.get("object_counts") or {}
            label_to_uid: dict[str, str] = {}
            for uid, obj in geometry.iter_objects(self.record):
                label_to_uid.setdefault(str(obj.get("query") or "").strip().lower(), uid)
            for name, count in sorted(counts.items(), key=lambda kv: -int(kv[1] or 0))[
                :max_features
            ]:
                uid = label_to_uid.get(str(name).strip().lower())
                if uid is None:
                    continue
                features.append(
                    {
                        "claim": f"{count}× {name} detected in the inventory",
                        "object_uids": [uid],
                        "keyframes": [],
                        "view_ids": [],
                        "provenance": "record",
                    }
                )
        return {"features": features, "source": source, "provenance": provenance}

    def _annotator_notes(self, top_k: int) -> tuple[dict[str, str], str]:
        """Optional labels-phase enrichment: ONE batched annotator call phrasing a short
        note per top object, grounded on the run's evidence imagery. Unknown uids and
        metric-claiming notes are dropped (guardrails 1+2).

        Returns ``(notes, provenance)`` so the labels phase can chip each enriched label
        with where the phrasing came from — a note written off a render must not read
        like a note written off a photograph."""
        pool = sorted(
            geometry.iter_objects(self.record),
            key=lambda pair: float(pair[1].get("confidence", 0.0) or 0.0),
            reverse=True,
        )[:top_k]
        if not pool:
            return {}, "none"
        frames, provenance = self._select_evidence_frames(
            _env_int("SCENE_AGENT_ANNOTATOR_KEYFRAMES", 6)
        )
        if not frames:
            return {}, "none"
        imagery = (
            "SYNTHETIC VIEWS rendered from this scene's own gaussian splat (not "
            "photographs — do not describe rendering artifacts as objects)"
            if provenance == "synthetic_view"
            else "keyframes of a 3D scan"
        )
        targets = [
            {"uid": uid, "label": geometry.object_label(obj)} for uid, obj in pool
        ]
        try:
            parsed, _response = self.llm.chat_json(
                "annotator",
                system_prompt=(
                    f"You verify detected objects in {imagery}. For each "
                    "target uid, if visible, write ONE short phrase about its appearance/"
                    "placement. CLOSED WORLD: only the given uids; never new objects; no "
                    "numeric sizes. Output strict JSON only: "
                    '{"notes": [{"object_id": "det:0", "note": "..."}]}'
                ),
                user_prompt=f"TARGETS: {json.dumps(targets)}",
                images_b64=[f["image_b64"] for f in frames] or None,
                temperature=0.3,
                max_tokens=_env_int(
                    "SCENE_AGENT_ANNOTATOR_MAX_TOKENS", DEFAULT_ANNOTATOR_MAX_TOKENS
                ),
            )
        except (LLMUnavailableError, BudgetExceededError):
            return {}, provenance
        except Exception as exc:
            print(f"[demo.agent] annotator notes failed: {exc}")
            return {}, provenance
        notes: dict[str, str] = {}
        for raw in parsed.get("notes") or []:
            if not isinstance(raw, dict):
                continue
            uid = raw.get("object_id")
            note = strip_metric_claims(raw.get("note"))
            if not note or not self.validator.admit(uid):
                continue
            notes[str(uid)] = note
        return notes, provenance

    # -- the annotate run ---------------------------------------------

    def run_annotate(self, mode: str = "full") -> None:
        """The F2 spectacle: survey → labels → description → dimensions →
        key_features → final state + run_done. ``mode="fact_only"`` skips every LLM
        call (free, deterministic — the walking-skeleton path)."""
        run = self.run
        fact_only = mode == "fact_only" or os.environ.get(
            "SCENE_AGENT_DISABLE_LLM", ""
        ).strip() == "1"
        stats = geometry.scene_stats(self.record)
        label = self.record.get("label") or self.scan_id
        run.emit(
            "run_meta",
            {
                "run_id": run.run_id,
                "scan_id": self.scan_id,
                "mode": run.mode,
                "annotate_mode": mode,
                "replay": False,
                "started_at": run.created_at,
                "models": self.llm.roster_payload(),
                "llm_enabled": not fact_only,
                "scene": {
                    "label": label,
                    "source": self.record.get("source"),
                    "object_count": stats["object_count"],
                    "stored_keyframes": stats["stored_keyframes"],
                    # Additive: what this run can actually look at, so the panel's
                    # close-out ("0 findings from N keyframes") stays truthful on a
                    # scene whose imagery is synthetic.
                    "synthetic_views": stats["synthetic_view_count"],
                    "evidence_source": self.evidence_source(),
                },
                "metric": self.metric.to_payload(),
            },
        )
        try:
            self._phase_survey(stats)
            run.check_deadline()
            self._phase_labels(fact_only)
            run.check_deadline()
            self._phase_description(fact_only)
            run.check_deadline()
            self._phase_dimensions()
            run.check_deadline()
            self._phase_key_features(fact_only)
            self._finish_ok()
        except TimeoutError as exc:
            self._emit_thought(
                f"Stopping early: {exc}. Emitted results so far stand.", None, "error"
            )
            self._finish_error(f"wall_clock: {exc}")
        except Exception as exc:
            print(f"[demo.agent] annotate run {run.run_id} failed: {exc}")
            self._emit_thought(f"Run failed: {exc}", None, "error")
            self._finish_error(str(exc))

    def _phase_survey(self, stats: dict[str, Any]) -> None:
        label = self.record.get("label") or self.scan_id
        source = self.evidence_source()
        if source == "synthetic_view":
            # State the substitution out loud, in the feed the founder is watching.
            # A run that quietly reasons over renders while the operator assumes photos
            # is the failure this whole workstream is guarding against.
            imagery = (
                f"{stats['synthetic_view_count']} synthetic views rendered from the "
                "imported splat (no capture video exists for this scene)"
            )
        elif source == "keyframe":
            imagery = f"{stats['stored_keyframes']} stored keyframes"
        else:
            imagery = "no stored imagery — findings will cite geometry only"
        self._emit_thought(
            f"Surveying persisted scene '{label}': {stats['point_count']:,} world points "
            f"across {stats['num_submaps']} submaps, {imagery}.",
            "survey",
            evidence_source=source,
        )
        self._emit_thought(
            f"Detected inventory: {stats['active_object_count']} objects across "
            f"{stats['object_types']} types"
            + (
                f" ({stats['dismissed_count']} dismissed by review)."
                if stats["dismissed_count"]
                else "."
            ),
            "survey",
        )
        coverage = stats["coverage_estimate"]
        self._emit_thought(
            (f"Coverage estimate {coverage:.0%}. " if coverage else "")
            + f"Scale state: {self.metric.wording}."
            + (
                ""
                if self.metric.anchored
                else " Lengths stay in relative units until a metric anchor is set."
            ),
            "survey",
        )

    def _phase_labels(self, fact_only: bool) -> None:
        notes: dict[str, str] = {}
        notes_provenance = "none"
        if not fact_only:
            notes, notes_provenance = self._annotator_notes(
                _env_int("SCENE_AGENT_ANNOTATOR_OBJECTS", 10)
            )
        for uid, obj in geometry.iter_objects(self.record):
            confidence = float(obj.get("confidence", 0.0) or 0.0)
            label = geometry.object_label(obj)
            phrased = (
                label if confidence >= CONFIDENCE_POSSIBLE_THRESHOLD else f"possible {label}"
            )
            description = phrased
            if obj.get("human_label"):
                description += " (operator-confirmed label)"
            note = notes.get(uid)
            if note:
                description += f" — {note}"
            center = geometry.object_center(obj)
            evidence = [
                {"submap_id": ev.get("submap_id"), "frame_idx": ev.get("frame_idx")}
                for ev in (obj.get("evidence") or [])
                if isinstance(ev, dict)
            ][:3]
            emitted = self._emit_finding(
                query=label,
                description=description,
                confidence=confidence,
                phase="labels",
                object_uid=uid,
                position=[float(v) for v in center] if center is not None else None,
                evidence=evidence,
                label_source="human" if obj.get("human_label") else "model",
                # Only the ENRICHED phrasing came from imagery; a bare label came from
                # the record, so an un-noted object must not claim a view grounded it.
                **({"provenance": notes_provenance} if note else {}),
            )
            if emitted and center is not None:
                self._emit_ui("focus_object", {"object_uid": uid}, "labels")

    def _phase_description(self, fact_only: bool) -> None:
        if fact_only:
            report = self.record.get("report") or {}
            description = {
                "paragraphs": [
                    p
                    for p in [str(report.get("summary", "")).strip()]
                    + [str(o).strip() for o in (report.get("observations") or [])[:2]]
                    if p
                ],
                "room_type": str(report.get("room_type", "unknown")),
                "source": "persisted_report",
                "degraded": True,
            }
            if not description["paragraphs"]:
                description["paragraphs"] = ["No stored report; fact-only description."]
            description["provenance"] = "record"
        else:
            description = self._build_description()
        for paragraph in description["paragraphs"]:
            self._emit_thought(
                paragraph, "description", "observation", provenance=description["provenance"]
            )
        self._emit_state(
            "description",
            scene_description=" ".join(description["paragraphs"]),
            room_type=description["room_type"],
            description_source=description["source"],
            degraded_mode=description["degraded"],
            evidence_source=self.evidence_source(),
        )

    def _phase_dimensions(self) -> None:
        dims = self._build_dimensions()
        room = dims.get("room")
        if room:
            d = room["dimensions"]
            unit_word = "m" if room["units"] == "m" else "units (relative, uncalibrated)"
            self._emit_finding(
                query="room extents",
                description=(
                    f"Principal extents ≈ {d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} {unit_word}. "
                    f"{room['note']}."
                ),
                confidence=1.0,
                phase="dimensions",
                object_uid=None,
                units=room["units"],
                units_basis=room["units_basis"],
                metric_state=room["metric_state"],
                values=room["dimensions"],
            )
            if "floor_area" in room or "ceiling_height" in room:
                parts = []
                if "floor_area" in room:
                    area_unit = "m²" if room["units"] == "m" else "square units (relative)"
                    parts.append(f"floor area ≈ {room['floor_area']:.2f} {area_unit}")
                if "floor_extent" in room:
                    fe = room["floor_extent"]
                    parts.append(f"footprint ≈ {fe[0]:.2f} × {fe[1]:.2f} {unit_word}")
                if "ceiling_height" in room:
                    parts.append(f"floor-to-ceiling ≈ {room['ceiling_height']:.2f} {unit_word}")
                self._emit_finding(
                    query="room dimensions",
                    description="; ".join(parts).capitalize() + f". {room['note']}.",
                    confidence=0.9,
                    phase="dimensions",
                    object_uid=None,
                    units=room["units"],
                    units_basis=room["units_basis"],
                    metric_state=room["metric_state"],
                    derivation=room.get("derivation"),
                )
        labeled_uids: list[str] = []
        for row in dims["objects"]:
            self._emit_finding(
                query=row["label"],
                description=(
                    f"{row['label']}: longest span "
                    + format_length(row["extents_slam"][int(_argmax(row['extents_slam']))], self.metric)
                ),
                confidence=row["confidence"],
                phase="dimensions",
                object_uid=row["object_uid"],
                units=row["units"],
                units_basis=row["units_basis"],
                metric_state=row["metric_state"],
                extents=row["extents"],
            )
            labeled_uids.append(row["object_uid"])
        if labeled_uids:
            self._emit_ui(
                "show_dimension_labels",
                {"visible": True, "object_uids": labeled_uids},
                "dimensions",
            )
        if not self.metric.anchored:
            self._emit_thought(
                "All lengths above are relative units (uncalibrated). Set a metric anchor "
                "(two points + a known distance) to read metres.",
                "dimensions",
            )

    def _phase_key_features(self, fact_only: bool) -> None:
        if fact_only:
            payload = None
            try:
                payload = self._fact_only_key_features()
            except Exception:
                payload = {"features": [], "source": "fact_fallback"}
        else:
            payload = self._build_key_features()
        for feature in payload["features"]:
            self._emit_finding(
                query="key feature",
                description=feature["claim"],
                confidence=0.8,
                phase="key_features",
                object_uid=feature["object_uids"][0] if feature["object_uids"] else None,
                evidence=feature.get("keyframes") or [],
                object_uids=feature["object_uids"],
                source=payload["source"],
                provenance=feature.get("provenance", "record"),
                view_ids=feature.get("view_ids") or [],
            )

    def _fact_only_key_features(self) -> dict[str, Any]:
        saved_llm = self.llm

        class _NoLLM:
            def chat_json(self, *a, **k):
                raise LLMUnavailableError("fact_only mode")

        try:
            self.llm = _NoLLM()  # type: ignore[assignment]
            return self._build_key_features()
        finally:
            self.llm = saved_llm

    def _run_done_payload(self, status: str, error: Optional[str] = None) -> dict[str, Any]:
        payload = {
            "run_id": self.run.run_id,
            "status": status,
            "duration_s": round(time.time() - self.run.created_at, 2),
            "findings_emitted": self.findings_emitted,
            "findings_dropped": self.validator.dropped,
            **self.tally.summary(),
        }
        # Attempted calls (budget spends) — covers mock/unpriced providers where the
        # usage tally saw nothing; real runs report both.
        payload["llm_calls_attempted"] = self.llm.calls_used
        if error is not None:
            payload["error"] = error
        return payload

    def _settle_run(self, payload: dict[str, Any]) -> None:
        """Charge an agent run on what it actually cost.

        Metered, not flat: the measured spread across model arms is 7.7x
        ($0.0146 to $0.1123 on the same scene), so a flat price would either
        overcharge every cheap run or lose money on every expensive one. There is
        no hold — the run is short and already bounded by
        SCENE_AGENT_MAX_SPEND_USD, so a single idempotent charge at the end is
        both simpler and more accurate than reserving a guess up front.

        An unpriced run (mock provider, or a model that reported no usage) has
        cost_usd None and is charged nothing, rather than charged a made-up
        number.
        """
        try:
            from server.billing import meter, prices

            usd = payload.get("cost_usd")
            if not usd:
                return
            meter.charge(
                self.user_id,
                prices.credits_for_llm_usd(usd),
                "agent_run",
                self.run.run_id,
                idem=f"agent_run:{self.run.run_id}",
            )
        except Exception as exc:  # billing must never fail a finished run
            print(f"[billing] agent run settle failed: {exc}")

    def _finish_ok(self) -> None:
        self._emit_state(None)
        payload = self._run_done_payload("done")
        self.run.emit("run_done", payload)
        self._settle_run(payload)
        self.run.finish(runlog.STATUS_DONE)

    def _finish_error(self, error: str) -> None:
        payload = self._run_done_payload("error", error=error)
        self.run.emit("run_done", payload)
        # Charged even on error: the LLM calls were made and billed to us. The
        # ceiling caps the exposure.
        self._settle_run(payload)
        self.run.finish(runlog.STATUS_ERROR, error=error)

    # -- tool loop (pilot + tool-intent chat) --------------------------

    def _execute_tool(self, name: str, args: dict[str, Any], phase: str) -> tuple[bool, Any]:
        """Run one kernel tool with started/succeeded/failed events + the code-driven UI
        side effects.

        The UI side effects come from the tool spec's PROJECTOR, applied to the tool's
        own result — so every argument that reaches the viewport was computed by code.
        A structured refusal (spend ceiling, a gate) is a failure that still carries its
        payload to the pilot, and — for tools that ask for it — still projects, because
        "nothing is building, here is why" is exactly what the operator needs to see."""
        ToolExecutionError = _tool_error()

        eid = self._emit_tool_event(name, "started", phase, args=args)
        t0 = time.time()
        try:
            result = self.registry.execute(name, args, timeout_s=20.0)
        except ToolExecutionError as exc:
            payload = getattr(exc, "payload", None)
            spec = self.registry.spec(name)
            self._emit_tool_event(
                name,
                "failed",
                phase,
                event_id=eid,
                args=args,
                result=payload if isinstance(payload, dict) else None,
                error=str(exc),
                latency_ms=int((time.time() - t0) * 1000),
            )
            if isinstance(payload, dict) and spec is not None and spec.project_errors:
                projected = {"format": (args or {}).get("format"), **payload}
                for ui_name, ui_args in spec.project(projected):
                    self._emit_ui(ui_name, ui_args, phase)
            return False, payload if isinstance(payload, dict) else str(exc)
        self._emit_tool_event(
            name,
            "succeeded",
            phase,
            event_id=eid,
            args=args,
            result=result,
            latency_ms=int((time.time() - t0) * 1000),
        )
        for ui_name, ui_args in self.registry.project(name, result):
            self._emit_ui(ui_name, ui_args, phase)
        job_event = self.registry.job_event_for(name, result)
        if job_event is not None:
            self._emit_job_event(name, job_event, phase)
        return True, result

    def _emit_job_event(self, tool: str, payload: dict[str, Any], phase: Optional[str]) -> None:
        """Announce a background job the agent started, carrying ``job_id``.

        This is what makes a non-blocking acting tool honest: the run can end long before
        the build does, and the client's run coordinator keeps watching the job through
        the existing ``GET /api/workspace/jobs/<job_id>`` route on the strength of this
        event."""
        self.run.emit(
            "agent_job_event",
            {
                "id": self._event_id(),
                "timestamp": time.time(),
                "tool": tool,
                "scan_id": self.scan_id,
                **payload,
            },
            phase=phase,
        )

    def _tool_loop(
        self,
        instruction: str,
        phase: str,
        max_steps: int,
        allowed_tools: Optional[list[str]] = None,
    ) -> None:
        """Bounded pilot loop: LLM picks tools as strict JSON; code executes + emits.
        Fallback chain per the roster (pilot → fallbacks); with no client at all we
        answer tool-free from facts, honestly."""
        tool_specs = [
            t
            for t in self.registry.list_tools()
            if allowed_tools is None or t["name"] in allowed_tools
        ]
        history: list[dict[str, str]] = []
        system_prompt = (
            "You pilot a 3D scene workspace over a PERSISTED reconstruction. You interact "
            "ONLY through the tools below; the world is closed to the persisted inventory. "
            "Never invent objects, coordinates, or measurements — numbers come from tool "
            "results. If a tool fails, say so honestly and adapt. Each tool declares an "
            "'effect' and an 'execution': effect 'spend' costs real money, so call it only "
            "when the instruction actually asked for it; execution 'job' returns a HANDLE "
            "({job_id, status, poll_route}) and the work finishes on its own — report the "
            "handle and move on, never wait for it and never re-call the tool to poll. A "
            "result carrying error 'precondition_failed' means nothing was started: say "
            "what is missing and offer its 'repair_tool' if there is one. Respond with "
            'strict JSON only: {"thought": "...", "action": {"tool": "name", "args": {…}} '
            '| null, "final": "closing message" | null}. Use exactly one of action/final '
            "per turn.\n"
            f"TOOLS: {json.dumps(tool_specs)}"
        )
        user_prompt = f"INSTRUCTION: {instruction}"
        final_message: Optional[str] = None
        #: One repair attempt per turn. A second shape miss is a real problem, not a slip.
        json_repair_used = False
        tools_run: list[str] = []
        for _step in range(max_steps):
            self.run.check_deadline()
            try:
                parsed, _response = self.llm.chat_json(
                    "pilot",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    history=history,
                    temperature=0.2,
                    max_tokens=700,
                )
            except (LLMUnavailableError, BudgetExceededError) as exc:
                self._emit_thought(f"LLM unavailable: {exc}", phase, "error")
                final_message = self._tool_free_answer(
                    instruction, reason=str(exc), cause="llm_unavailable", tools_run=tools_run
                )
                break
            except Exception as exc:
                # A SHAPE miss is not an outage. Two rungs, cheapest salvage first:
                # a typed LLMFormatError carries the completion we already paid for, so
                # keep its prose outright (the metric gate below still polices any
                # numbers in it) — retrying would spend another call to reproduce an
                # answer we are holding. Only a shape slip with nothing salvageable
                # (bare "No JSON object found" / decode errors) gets the one explicit
                # retry nudge — the loop contract is strict JSON and the model simply
                # forgot it mid-conversation.
                fmt = is_format_error(exc)
                if not fmt and not json_repair_used and _is_json_shape_error(exc):
                    json_repair_used = True
                    self._emit_thought(
                        "That reply was not in the required JSON shape — retrying once.",
                        phase,
                        "thinking",
                    )
                    user_prompt = (
                        f"{user_prompt}\n\nYour previous reply was not valid JSON. Reply "
                        'with STRICT JSON ONLY, no prose, no code fence: '
                        '{"thought": "...", "action": {"tool": "name", "args": {}} | null, '
                        '"final": "closing message" | null}'
                    )
                    continue
                self._emit_thought(
                    f"{'Model broke JSON format' if fmt else 'LLM call failed'}: {exc}",
                    phase,
                    "error",
                )
                final_message = strip_metric_claims(getattr(exc, "content", "")) if fmt else None
                if final_message is None:
                    final_message = self._tool_free_answer(
                        instruction,
                        reason=str(exc),
                        cause="bad_format" if fmt else "call_failed",
                        tools_run=tools_run,
                    )
                break
            history.append({"role": "user", "content": user_prompt})
            thought = parsed.get("thought")
            if isinstance(thought, str) and thought.strip():
                self._emit_thought(thought.strip(), phase, "thinking")
            action = parsed.get("action")
            if isinstance(action, dict) and action.get("tool"):
                name = str(action.get("tool"))
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                if allowed_tools is not None and name not in allowed_tools:
                    ok, outcome = False, f"tool {name!r} is not allowed for this turn"
                    self._emit_tool_event(name, "failed", phase, args=args, error=outcome)
                else:
                    ok, outcome = self._execute_tool(name, args, phase)
                if ok and name not in tools_run:
                    tools_run.append(name)
                # Shrinks structurally and says so in-band. The old `[:4000]` cut the
                # 95-object inventory mid-JSON at ~20 rows with nothing to signal it, so
                # the agent denied that real objects existed. See tool_kernel.present_result.
                outcome_json = tool_kernel.present_result(outcome)
                history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"thought": thought, "action": action}),
                    }
                )
                user_prompt = (
                    f"TOOL RESULT ({name}, {'ok' if ok else 'FAILED'}): {outcome_json}\n"
                    "Continue. Remember: numbers only from tool results."
                )
                continue
            final = parsed.get("final")
            final_message = (
                final.strip() if isinstance(final, str) and final.strip() else None
            ) or "Done."
            break
        else:
            final_message = final_message or (
                "Step budget reached — stopping here. Results emitted so far stand."
            )
        if final_message is None:
            final_message = "Done."
        cleaned = strip_metric_claims(final_message)
        if cleaned is None:
            # The model tried to utter its own numbers — replace with the honest note.
            cleaned = (
                "Finished — see the measurement events above for the computed values "
                "(model prose withheld: it contained uncomputed quantities)."
            )
        self._emit_thought(cleaned, phase, "chat_response")

    def _tool_free_answer(
        self,
        question: str,
        reason: str = "",
        *,
        cause: str = "llm_unavailable",
        tools_run: Optional[list[str]] = None,
    ) -> str:
        """Grounded, fact-only fallback answer (last rung of every chat/pilot chain).

        ``cause`` names what actually degraded — "LLM unavailable" is a claim about an
        outage and must not be printed for a model that answered badly, or for a
        grounded-QA path that never reached a model at all. ``tools_run`` are the tools
        that already succeeded this turn: their results are emitted as events carrying
        units and scale source, so the fallback points at them rather than pretending
        the turn produced nothing. The full ``reason`` goes to the transcript as an
        error event — this prose only carries the label.
        """
        stats = geometry.scene_stats(self.record)
        report = self.record.get("report") or {}
        top = [
            geometry.object_label(o)
            for _uid, o in sorted(
                geometry.iter_objects(self.record),
                key=lambda pair: float(pair[1].get("confidence", 0.0) or 0.0),
                reverse=True,
            )[:6]
        ]
        note = f" ({_FALLBACK_CAUSES.get(cause, cause)}; fact-only answer)" if reason else ""
        # Never drop computed work on the floor: if tools ran, their emitted results are
        # the answer to the question that was asked, and prose that ignores them reads as
        # "nothing was found" when in fact everything was.
        computed = (
            f"Computed this turn by {', '.join(tools_run)} — see the tool events above "
            f"for those values with their units and scale source. "
            if tools_run
            else ""
        )
        scale = (
            f"Scale: {self.metric.wording} (scale source: {self.metric.scale_source})."
            if self.metric.anchored
            else f"Scale: {self.metric.wording} — this scene has no metric anchor."
        )
        return (
            f"{computed}This {report.get('room_type', 'space')} scan holds "
            f"{stats['active_object_count']} detected objects across "
            f"{stats['object_types']} types; notable: {', '.join(top) if top else 'none'}. "
            f"{scale}{note}"
        )

    # -- chat ----------------------------------------------------------

    def run_chat_turn(self, message: str, history: Optional[list[dict[str, str]]] = None) -> None:
        run = self.run
        run.emit(
            "run_meta",
            {
                "run_id": run.run_id,
                "scan_id": self.scan_id,
                "mode": run.mode,
                "replay": False,
                "started_at": run.created_at,
                "models": self.llm.roster_payload(),
                "metric": self.metric.to_payload(),
            },
        )
        self._emit_thought(message, "chat", "observation", author="user")
        try:
            if _TOOL_INTENT_RE.search(message):
                self._tool_loop(
                    message,
                    phase="chat",
                    max_steps=_env_int("SCENE_AGENT_CHAT_MAX_STEPS", DEFAULT_CHAT_MAX_STEPS),
                    allowed_tools=[
                        "list_scene_objects",
                        "measure_distance",
                        "measure_angle",
                        "plan_path",
                        "export.lerobot",
                        "export.isaac",
                        "annotate.estimate_dimensions",
                    ],
                )
            else:
                self._grounded_qa_turn(message, history or [])
            self._finish_ok()
        except TimeoutError as exc:
            self._emit_thought(f"Stopping early: {exc}", "chat", "error")
            self._finish_error(f"wall_clock: {exc}")
        except Exception as exc:
            print(f"[demo.agent] chat turn failed: {exc}")
            self._emit_thought(f"Chat turn failed: {exc}", "chat", "error")
            self._emit_thought(
                self._tool_free_answer(message, reason=str(exc), cause="call_failed"),
                "chat",
                "chat_response",
            )
            self._finish_ok()

    def _grounded_qa_turn(self, message: str, history: list[dict[str, str]]) -> None:
        """Non-tool answers route through the existing grounded-QA core
        (``server.app._answer_scene_question`` — citation-bearing, evidence-selecting),
        with the fact-only fallback when the app/LLM side is unavailable."""
        answer: Optional[dict[str, Any]] = None
        try:
            from server.oreos import app_module

            qa = getattr(app_module(), "_answer_scene_question", None)
            if callable(qa):
                answer = qa(self.user_id, self.scan_id, self.record, message, history)
        except Exception as exc:
            print(f"[demo.agent] grounded QA unavailable ({exc}); fact fallback")
        if not isinstance(answer, dict) or not answer.get("answer"):
            self._emit_thought(
                self._tool_free_answer(
                    message, reason="qa_unavailable", cause="qa_unavailable"
                ),
                "chat",
                "chat_response",
            )
            return
        self._emit_thought(
            str(answer.get("answer")),
            "chat",
            "chat_response",
            model=answer.get("model"),
            degraded=answer.get("degraded"),
            evidence=answer.get("evidence"),
        )
        focus = answer.get("focus")
        if isinstance(focus, dict) and focus.get("name"):
            resolved = geometry.resolve_object(self.record, str(focus["name"]))
            if resolved is not None:
                self._emit_ui("focus_object", {"object_uid": resolved[0]}, "chat")

    # -- pilot ---------------------------------------------------------

    def run_pilot(self, instruction: str) -> None:
        run = self.run
        run.emit(
            "run_meta",
            {
                "run_id": run.run_id,
                "scan_id": self.scan_id,
                "mode": run.mode,
                "replay": False,
                "started_at": run.created_at,
                "models": self.llm.roster_payload(),
                "instruction": instruction,
                "metric": self.metric.to_payload(),
            },
        )
        self._emit_thought(instruction, "pilot", "observation", author="user")
        try:
            self._tool_loop(
                instruction,
                phase="pilot",
                max_steps=_env_int("SCENE_AGENT_PILOT_MAX_STEPS", DEFAULT_PILOT_MAX_STEPS),
                allowed_tools=["list_scene_objects", "plan_path", "measure_distance"],
            )
            self._finish_ok()
        except TimeoutError as exc:
            self._emit_thought(f"Stopping early: {exc}", "pilot", "error")
            self._finish_error(f"wall_clock: {exc}")
        except Exception as exc:
            print(f"[demo.agent] pilot run failed: {exc}")
            self._emit_thought(f"Pilot run failed: {exc}", "pilot", "error")
            self._finish_error(str(exc))


def _argmax(values: list[float]) -> int:
    best, best_i = None, 0
    for i, v in enumerate(values):
        if best is None or v > best:
            best, best_i = v, i
    return best_i


# ── Run starters (called by routes_agent; each spawns the run thread) ──


def _max_wall_s() -> float:
    return _env_float("SCENE_AGENT_MAX_WALL_S", DEFAULT_MAX_WALL_S)


def _spawn(run: AgentRun, body: Callable[[], None]) -> AgentRun:
    thread = threading.Thread(target=body, name=f"demo-agent-{run.run_id}", daemon=True)
    run.thread = thread
    thread.start()
    return run


def start_annotate_run(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    mode: str = "full",
    llm_factory: Optional[Callable[..., Any]] = None,
) -> AgentRun:
    run = REGISTRY.start_run(
        user_id,
        scan_id,
        mode="annotate",
        flusher=make_flusher(store, user_id, scan_id),
        max_wall_s=_max_wall_s(),
    )
    agent = PersistedSceneAgent(store, user_id, scan_id, record, run, llm_factory)
    return _spawn(run, lambda: agent.run_annotate(mode=mode))


def start_replay_run(
    store: Any,
    user_id: str,
    scan_id: str,
    source_doc: dict[str, Any],
    speed: float = 1.0,
) -> AgentRun:
    run = REGISTRY.start_run(
        user_id,
        scan_id,
        mode="replay",
        replay=True,
        source_run_id=str(source_doc.get("run_id") or ""),
        flusher=make_flusher(store, user_id, scan_id),
        max_wall_s=None,  # pacing is clamped, not budgeted
    )
    return _spawn(run, lambda: runlog.play_replay_into(run, source_doc, speed=speed))


def start_chat_run(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    message: str,
    history: Optional[list[dict[str, str]]] = None,
    llm_factory: Optional[Callable[..., Any]] = None,
) -> AgentRun:
    run = REGISTRY.start_run(
        user_id,
        scan_id,
        mode="chat",
        flusher=make_flusher(store, user_id, scan_id),
        max_wall_s=_max_wall_s(),
    )
    agent = PersistedSceneAgent(store, user_id, scan_id, record, run, llm_factory)
    return _spawn(run, lambda: agent.run_chat_turn(message, history))


def start_pilot_run(
    store: Any,
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    instruction: str,
    llm_factory: Optional[Callable[..., Any]] = None,
) -> AgentRun:
    run = REGISTRY.start_run(
        user_id,
        scan_id,
        mode="pilot",
        flusher=make_flusher(store, user_id, scan_id),
        max_wall_s=_max_wall_s(),
    )
    agent = PersistedSceneAgent(store, user_id, scan_id, record, run, llm_factory)
    return _spawn(run, lambda: agent.run_pilot(instruction))
