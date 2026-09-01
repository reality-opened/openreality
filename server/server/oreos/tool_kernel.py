"""Tool kernel — typed results, declared effects and a policy gate in front of the
persisted-scene agent's tools (Phase 2 of the agent tool layer).

This module **extends** ``server/agent/tool_registry.py``; it does not replace it. The
live agent's ``ToolRegistry`` keeps owning what it already does well — pydantic argument
validation, the worker pool, the execution timeout — and every tool still goes through it.
What this module adds is the part a registry of ``(name, args_model, handler)`` triples
cannot express:

  ``result_model``  the shape the handler MUST return. Until now a handler could return
                    anything at all and the loop would serialize it straight into the
                    pilot's context; a typo'd key became a silent behaviour change three
                    turns later. A result that does not match its declared shape now
                    fails LOUDLY, at the call site, with the validation error attached.

  ``effect``        ``read`` | ``annotate`` | ``mutate`` | ``spend``. This is the axis the
                    policy executor gates on: ``read``/``annotate`` tools are free to
                    re-run, ``mutate``/``spend`` tools are not.

  ``execution``     ``sync`` | ``job``. A ``job`` tool returns a HANDLE immediately and
                    never blocks on the work: the Modal broker kills a web request at
                    150 s and an Isaac export measured 382 s, so a tool that waited would
                    be killed mid-build — and an LLM polling for minutes burns tokens and
                    holds the scene's one run slot the whole time.

  ``projector``     result -> ``[(ui_command_name, args), …]``. Deliberately a CALLABLE and
                    not a static list, because the mapping is genuinely conditional:
                    ``plan_path`` emits two commands (``show_path`` + ``drive_path``) from
                    one result and only when the planner returned waypoints, and the
                    export tools emit a repair toast only when a gate blocked them. UI
                    args are read off the tool RESULT — never off LLM text.

Nothing here writes to the scene: handlers remain non-destructive and, where they persist
at all, only under ``derived/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ValidationError

# ── vocabulary ─────────────────────────────────────────────────────────────────

ToolEffect = Literal["read", "annotate", "mutate", "spend"]
ToolExecution = Literal["sync", "job"]

#: Effects whose handler changes something outside this process. Only these are
#: idempotency-guarded and budget-checked; a read tool re-running is free and correct.
GUARDED_EFFECTS: tuple[str, ...] = ("mutate", "spend")

#: Per-run ceiling on money a single agent run may commit, in USD.
#:
#: THIS CEILING IS THE ONLY BRAKE. The founder deliberately chose autonomy over a
#: per-action confirmation dialog: the agent spawns GPU/CPU jobs without asking. That is
#: the right demo experience and the wrong safety story unless *something* bounds it, and
#: this number is that something — it is what stands between a mis-parsed instruction
#: ("prepare every export for every format, twice") and a real bill. Raise it knowingly.
DEFAULT_MAX_SPEND_USD = 1.0
MAX_SPEND_ENV = "SCENE_AGENT_MAX_SPEND_USD"

DEFAULT_TOOL_TIMEOUT_S = 20.0

#: Marker added to a result served from the idempotency cache, so neither the pilot nor a
#: reader of the run log can mistake a replay for a second real execution.
REPLAY_KEY = "idempotent_replay"


# ── the live registry module (shared handle, single class identity) ────────────

_TOOL_REGISTRY_MOD: Any = None


def tool_registry_module():
    """The live agent's ``tool_registry`` module WITHOUT triggering
    ``server.agent.__init__`` when core is absent.

    The package init pulls the live-agent runtime chain (``agent/runtime.py`` →
    ``vggt_slam``), which exists on the deployed image (core is pip-installed) but not in
    GPU-free test/local envs. ``tool_registry.py`` itself is dependency-light (pydantic +
    stdlib), so when the normal import fails we load the SAME file by path — the module is
    still imported unchanged, just without the heavy siblings.

    Moved here from ``persisted_agent`` so this module and the agent share ONE module
    object: ``ToolExecutionError`` must be one class, or the agent's ``except`` clause
    would not catch an error raised in here.
    """
    global _TOOL_REGISTRY_MOD
    if _TOOL_REGISTRY_MOD is not None:
        return _TOOL_REGISTRY_MOD
    import sys

    # Already-loaded wins, under EITHER name. Reusing the by-path module too keeps
    # ``ToolExecutionError`` one class across a reload of this module (the demo tests
    # re-import the whole ``server.oreos`` package between cases; without this, a refusal
    # raised by the reloaded kernel would not be an instance of the class an older
    # reference still holds, and ``except`` clauses would quietly stop matching).
    if "server.agent.tool_registry" in sys.modules:
        _TOOL_REGISTRY_MOD = sys.modules["server.agent.tool_registry"]
        return _TOOL_REGISTRY_MOD
    if "_demo_tool_registry" in sys.modules:
        _TOOL_REGISTRY_MOD = sys.modules["_demo_tool_registry"]
        return _TOOL_REGISTRY_MOD
    try:
        import importlib

        _TOOL_REGISTRY_MOD = importlib.import_module("server.agent.tool_registry")
        return _TOOL_REGISTRY_MOD
    except Exception:
        pass

    # ``server.agent.__init__`` fails on the heavy runtime chain — but it fails AFTER
    # ``from .runtime import …`` has already pulled this submodule in, so the real module
    # is usually sitting in sys.modules by now. Take it: path-loading a second copy would
    # mint a SECOND ToolExecutionError class, and whichever call happened first would win
    # a coin toss over whether the agent's ``except`` clause matches.
    recovered = sys.modules.get("server.agent.tool_registry")
    if recovered is not None:
        _TOOL_REGISTRY_MOD = recovered
        return _TOOL_REGISTRY_MOD

    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent",
        "tool_registry.py",
    )
    spec = importlib.util.spec_from_file_location("_demo_tool_registry", path)
    module = importlib.util.module_from_spec(spec)
    # Must be registered BEFORE exec_module (dataclasses looks itself up via
    # sys.modules[cls.__module__] at class-creation time).
    sys.modules["_demo_tool_registry"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _TOOL_REGISTRY_MOD = module
    return _TOOL_REGISTRY_MOD


def tool_execution_error() -> type[Exception]:
    return tool_registry_module().ToolExecutionError


_POLICY_ERROR_CLS: Any = None


def tool_policy_error() -> type[Exception]:
    """``ToolPolicyError`` — a ``ToolExecutionError`` carrying a structured ``payload``.

    Built lazily and subclassed off ``tool_execution_error()`` on purpose: the agent
    catches the class returned by that loader, so a refusal raised here has to be an
    instance of that exact class object (a same-named class from a second import of the
    file would sail straight past the handler and kill the run)."""
    global _POLICY_ERROR_CLS
    if _POLICY_ERROR_CLS is None:
        base = tool_execution_error()

        class ToolPolicyError(base):  # type: ignore[misc,valid-type]
            """Refused before the handler ran. ``payload`` is the structured reason."""

            def __init__(self, payload: dict[str, Any]):
                self.payload: dict[str, Any] = dict(payload)
                super().__init__(
                    str(payload.get("reason") or payload.get("error") or "tool refused")
                )

        _POLICY_ERROR_CLS = ToolPolicyError
    return _POLICY_ERROR_CLS


def raise_policy(payload: dict[str, Any]) -> None:
    raise tool_policy_error()(payload)  # type: ignore[operator]


def precondition_failure(
    capability: str,
    reason: str,
    repair_tool: Optional[str] = None,
    **extra: Any,
) -> dict[str, Any]:
    """The ONE structured refusal shape.

    It carries a ``repair_tool`` because a refusal the agent can only apologise for is a
    dead end — with it the agent can both explain and offer the fix ("that needs a metric
    anchor — want me to set one?")."""
    payload: dict[str, Any] = {
        "error": "precondition_failed",
        "capability": str(capability),
        "reason": str(reason),
        "repair_tool": str(repair_tool) if repair_tool else None,
    }
    payload.update(extra)
    return payload


# ── presenting a result to the model ───────────────────────────────────────────


#: Character budget for one tool result in the loop's prompt. Generous enough that a
#: whole scene inventory fits in the compact projection (95 objects ≈ 6 KB); the shrink
#: path below exists for the cases that still do not.
TOOL_RESULT_CHAR_BUDGET = 12000


def present_result(result: Any, *, budget: int = TOOL_RESULT_CHAR_BUDGET) -> str:
    """Serialize a tool result for the model, shrinking it HONESTLY if it is too large.

    THE BUG THIS EXISTS FOR (found live, 2026-08-04). The loop used to do
    ``json.dumps(result)[:4000] + "…(truncated)"``. On the canonical office scene that cut
    the 95-object inventory to about 20 rows — mid-string, mid-object, producing JSON the
    model cannot parse but happily pattern-matches over. Asked to measure to a "coat rack"
    (``det:36`` and ``det:55``, both real, both rendered on screen), the agent answered
    that no such object exists and that it would not invent one. Impeccable reasoning over
    a silently amputated list.

    The failure was not "too few objects" — it was that nothing in the payload said the
    list had been cut, so the agent could not know to say "I can only see 20 of 95".
    Truncation that a reader cannot detect is worse than a smaller budget: it converts a
    capacity limit into a confident false denial, which the closed-world guardrail then
    dresses up as integrity.

    So: shrink STRUCTURALLY (drop whole items from the longest list, never split a value)
    and DECLARE it in-band, as data the model reads like any other field.
    """
    encoded = json.dumps(result, default=str)
    if len(encoded) <= budget or not isinstance(result, dict):
        return encoded

    # Shrink the longest list field, keeping the payload valid JSON throughout.
    lists = [(k, v) for k, v in result.items() if isinstance(v, list) and v]
    if lists:
        key, items = max(lists, key=lambda kv: len(kv[1]))
        total = len(items)
        kept = list(items)
        while kept:
            kept.pop()
            trimmed = {
                **result,
                key: kept,
                "_truncated": {
                    "field": key,
                    "shown": len(kept),
                    "total": total,
                    "note": (
                        f"Only {len(kept)} of {total} {key} fit in this result. The rest "
                        f"EXIST — do not report them as absent. Narrow the query or raise "
                        f"max_results to see more."
                    ),
                },
            }
            encoded = json.dumps(trimmed, default=str)
            if len(encoded) <= budget:
                return encoded

    # No list to shrink (or shrinking was not enough): refuse rather than hand over a
    # fragment. An unparseable partial is what caused the false denial in the first place.
    return json.dumps(
        {
            "error": "result_too_large",
            "note": (
                "This tool's result did not fit the prompt budget and was NOT truncated, "
                "because a partial result reads as a complete one. Re-run with a narrower "
                "scope. Do not conclude anything about what the result would have said."
            ),
            "bytes": len(json.dumps(result, default=str)),
        }
    )


# ── declared result shapes ─────────────────────────────────────────────────────


class ToolResultModel(BaseModel):
    """Base for a tool's declared RESULT shape.

    ``extra="forbid"`` is the point of the exercise: an undeclared key means the handler
    and its contract have drifted, and the drift surfaces here rather than as strange LLM
    behaviour later. ``latency_ms`` is declared because ``ToolRegistry.execute`` stamps it
    onto every result after the handler returns."""

    model_config = {"extra": "forbid"}

    latency_ms: Optional[int] = None
    idempotent_replay: Optional[bool] = None


# ── the spec ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    """One tool: the live ``ToolDefinition`` plus everything the loop needs to know
    about it that a ``(name, args, handler)`` triple cannot say."""

    definition: Any  # tool_registry.ToolDefinition
    effect: ToolEffect = "read"
    execution: ToolExecution = "sync"
    result_model: Optional[type[BaseModel]] = None
    #: result -> [(ui_command_name, args), …]
    projector: Optional[Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]]]] = None
    #: Project the structured refusal too (the gate-blocked export wants its toast).
    project_errors: bool = False
    #: validated-args -> structured refusal, or ``None`` to proceed. Runs OUTSIDE the
    #: worker pool so a refusal can never be mistaken for a handler crash.
    precondition: Optional[Callable[[Any], Optional[dict[str, Any]]]] = None
    estimated_cost_usd: float = 0.0
    cost_estimator: Optional[Callable[[Any], float]] = None
    #: (result, reserved_estimate) -> the amount actually booked. Lets a tool that
    #: short-circuited (nothing spawned) release its reservation instead of charging.
    settle_cost: Optional[Callable[[dict[str, Any], float], float]] = None
    #: result -> an ``agent_job_event`` payload (or ``None``), for ``execution="job"``.
    job_event: Optional[Callable[[dict[str, Any]], Optional[dict[str, Any]]]] = None

    @property
    def name(self) -> str:
        return str(self.definition.name)

    @property
    def guarded(self) -> bool:
        return self.effect in GUARDED_EFFECTS

    def estimate(self, model: Any) -> float:
        if self.cost_estimator is not None:
            return max(0.0, float(self.cost_estimator(model)))
        return max(0.0, float(self.estimated_cost_usd))

    def settle(self, result: dict[str, Any], reserved: float) -> float:
        if self.settle_cost is None:
            return reserved
        return max(0.0, float(self.settle_cost(result, reserved)))

    def validate_result(self, result: Any) -> None:
        """Raise ``ToolExecutionError`` when the handler's return does not match the
        declared shape. Validation NEVER rewrites the result — the caller keeps the
        handler's own dict, so nothing is silently coerced or dropped."""
        if self.result_model is None:
            return
        try:
            self.result_model.model_validate(result)
        except ValidationError as exc:
            raise tool_execution_error()(
                f"Tool {self.name!r} returned a result that does not match "
                f"{self.result_model.__name__}: {exc}"
            ) from exc

    def project(self, result: Any) -> list[tuple[str, dict[str, Any]]]:
        if self.projector is None or not isinstance(result, dict):
            return []
        commands = self.projector(result) or []
        return [(str(n), dict(a or {})) for n, a in commands]


# ── idempotency ledger ─────────────────────────────────────────────────────────
#
# Process-wide (not per-kernel) on purpose: the retry that has to be caught usually
# arrives on a LATER agent run, after the operator or the pilot re-issues an instruction
# whose first attempt appeared to fail.
_IDEMPOTENCY: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_IDEMPOTENCY_LOCK = threading.Lock()


def reset_idempotency_cache() -> None:
    """Drop every remembered result (tests; a fresh broker starts empty anyway)."""
    with _IDEMPOTENCY_LOCK:
        _IDEMPOTENCY.clear()


def args_digest(args: dict[str, Any]) -> str:
    """A stable digest of a call's arguments — the DEFAULT idempotency key.

    Falling back to the arguments matters more than accepting an explicit key: the LLM
    does not mint idempotency keys, it simply calls the same tool with the same arguments
    again, and that is exactly the call that must not spawn a second job."""
    try:
        blob = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        blob = repr(args)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# ── the kernel ─────────────────────────────────────────────────────────────────


class ToolKernel:
    """Holds the specs, exposes the registry surface the agent loop already consumes, and
    runs the policy executor in front of every guarded handler."""

    def __init__(
        self,
        *,
        user_id: Any,
        scan_id: Any,
        max_workers: int = 2,
        default_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    ):
        tr = tool_registry_module()
        self._registry = tr.ToolRegistry(max_workers=max_workers)
        self._tool_definition = tr.ToolDefinition
        self._specs: dict[str, ToolSpec] = {}
        self.user_id = str(user_id)
        self.scan_id = str(scan_id)
        self.default_timeout_s = float(default_timeout_s)
        self._spend_usd = 0.0
        self._spend_lock = threading.Lock()

    # -- registration --------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec
        self._registry.register(spec.definition)

    def register_tool(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[Any], dict[str, Any]],
        *,
        aliases: tuple[str, ...] = (),
        allow: Optional[Callable[[str], bool]] = None,
        **spec_kwargs: Any,
    ) -> None:
        """Register ``name`` (and any aliases) as one spec each. ``allow`` is the agent's
        ``SCENE_AGENT_TOOLS`` allowlist predicate, applied per NAME so an alias can be
        allowlisted independently — the pre-existing behaviour."""
        for tool_name in (name, *aliases):
            if allow is not None and not allow(tool_name):
                continue
            self.register(
                ToolSpec(
                    definition=self._tool_definition(
                        name=tool_name,
                        description=description,
                        args_model=args_model,
                        handler=handler,
                    ),
                    **spec_kwargs,
                )
            )

    # -- introspection -------------------------------------------------

    def spec(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """The registry's own listing (``name`` / ``description`` / ``args_schema`` — the
        shape the pilot loop already consumes), plus the declared ``effect`` and
        ``execution`` so the pilot knows which calls cost money and which return a handle
        rather than an answer."""
        rows = self._registry.list_tools()
        for row in rows:
            spec = self._specs.get(row.get("name"))
            if spec is not None:
                row["effect"] = spec.effect
                row["execution"] = spec.execution
        return rows

    def project(self, name: str, result: Any) -> list[tuple[str, dict[str, Any]]]:
        spec = self._specs.get(name)
        return spec.project(result) if spec is not None else []

    def job_event_for(self, name: str, result: Any) -> Optional[dict[str, Any]]:
        spec = self._specs.get(name)
        if spec is None or spec.job_event is None or not isinstance(result, dict):
            return None
        payload = spec.job_event(result)
        return dict(payload) if isinstance(payload, dict) else None

    # -- budget --------------------------------------------------------

    @property
    def spend_usd(self) -> float:
        with self._spend_lock:
            return round(self._spend_usd, 6)

    @property
    def max_spend_usd(self) -> float:
        raw = os.environ.get(MAX_SPEND_ENV, "").strip()
        if not raw:
            return DEFAULT_MAX_SPEND_USD
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return DEFAULT_MAX_SPEND_USD

    # -- execution -----------------------------------------------------

    def execute(
        self,
        name: str,
        args: Optional[dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run one tool through the policy executor and the live registry.

        Argument-compatible with ``ToolRegistry.execute`` so existing callers are
        unchanged."""
        spec = self._specs.get(name)
        if spec is None:
            # Same wording the registry uses, so the agent's existing failure copy and
            # its tests keep reading the same.
            raise tool_execution_error()(f"Unknown tool: {name}")

        call_args = dict(args or {})
        reserved = 0.0
        cache_key: Optional[tuple[str, str, str, str]] = None

        if spec.guarded:
            model = self._validate_args(spec, call_args)
            cache_key = (
                self.user_id,
                self.scan_id,
                name,
                str(idempotency_key or args_digest(call_args)),
            )
            cached = self._cached(cache_key)
            if cached is not None:
                return cached
            failure = spec.precondition(model) if spec.precondition is not None else None
            if failure:
                raise_policy(failure)
            if spec.effect == "spend":
                reserved = self._reserve(spec, model)

        result = self._registry.execute(
            name, call_args, timeout_s=float(timeout_s or self.default_timeout_s)
        )
        spec.validate_result(result)

        if spec.effect == "spend" and reserved:
            self._settle(spec, result, reserved)
        if cache_key is not None:
            self._remember(cache_key, result)
        return result

    # -- policy executor internals -------------------------------------

    def _validate_args(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        """Validate a guarded call's args HERE as well as in the registry: the
        precondition and the cost estimator both read the typed model, and both run before
        the handler is ever submitted to the pool."""
        try:
            return spec.definition.args_model.model_validate(args)
        except ValidationError as exc:
            raise tool_execution_error()(
                f"Invalid args for tool '{spec.name}': {exc}"
            ) from exc

    def _cached(self, key: tuple[str, str, str, str]) -> Optional[dict[str, Any]]:
        """A previously returned result for this exact call, or ``None``.

        WHY THIS IS LOAD-BEARING, not a nicety: ``ToolRegistry.execute`` waits with
        ``future.result(timeout=…)``. On timeout that RAISES — but it does not cancel the
        worker thread, which keeps running to completion. So the first attempt at a spend
        tool can time out at the loop's 20 s while its handler goes on to spawn a GPU job
        perfectly successfully; the pilot sees a failure and retries; without this cache
        the retry spawns a SECOND container and the scene is billed twice for one
        instruction. Returning the first attempt's result also hands the pilot the job_id
        it thought it had lost."""
        with _IDEMPOTENCY_LOCK:
            hit = _IDEMPOTENCY.get(key)
        if hit is None:
            return None
        replay = dict(hit)
        replay[REPLAY_KEY] = True
        return replay

    def _remember(self, key: tuple[str, str, str, str], result: dict[str, Any]) -> None:
        with _IDEMPOTENCY_LOCK:
            _IDEMPOTENCY[key] = dict(result)

    def _reserve(self, spec: ToolSpec, model: Any) -> float:
        """Check the run's spend ceiling and reserve this call's estimate.

        Reserved BEFORE the handler runs, because refusing after the job is spawned is not
        a refusal. See ``DEFAULT_MAX_SPEND_USD``: with per-action confirmation deliberately
        off, this check is the only thing between a mis-parsed instruction and real
        money."""
        estimate = spec.estimate(model)
        ceiling = self.max_spend_usd
        with self._spend_lock:
            projected = self._spend_usd + estimate
            if projected > ceiling + 1e-9:
                raise_policy(
                    precondition_failure(
                        "spend_budget",
                        (
                            f"this run has committed ${self._spend_usd:.2f} and {spec.name} "
                            f"would add ${estimate:.2f}, over the ${ceiling:.2f} per-run "
                            f"ceiling ({MAX_SPEND_ENV}). Nothing was started."
                        ),
                        repair_tool=None,
                        spent_usd=round(self._spend_usd, 4),
                        estimate_usd=round(estimate, 4),
                        ceiling_usd=round(ceiling, 4),
                        tool=spec.name,
                    )
                )
            self._spend_usd = projected
        return estimate

    def _settle(self, spec: ToolSpec, result: dict[str, Any], reserved: float) -> None:
        booked = spec.settle(result if isinstance(result, dict) else {}, reserved)
        if abs(booked - reserved) < 1e-9:
            return
        with self._spend_lock:
            self._spend_usd = max(0.0, self._spend_usd - reserved + booked)

    # -- lifecycle -----------------------------------------------------

    def shutdown(self) -> None:
        self._registry.shutdown()
