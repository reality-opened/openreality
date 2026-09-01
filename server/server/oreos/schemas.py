"""Demo-side agent schemas — the SUPERSET UICommand model + run-event envelope.

The live agent's ``server/agent/schemas.py`` carries a strict ``Literal`` union of
UICommand names that is FROZEN (agent-exports.md: "do not edit the live agent's strict
Literal UICommand union"). The demo scene agent therefore validates its UI commands
against this superset model instead: the 8 live names PLUS the six demo commands the
protocol contract added additively (``web/packages/protocol/types.ts`` AgentUICommand +
``demo.ts`` DemoUICommandArgs). Payload arg shapes mirror ``demo.ts``.

Event envelope: one run event is ``{seq, ts, type, phase, payload}`` — ``seq`` a per-run
monotonic cursor, ``ts`` epoch seconds, ``type`` the existing socket vocabulary
(``agent_thought`` / ``agent_finding`` / ``agent_state`` / ``agent_tool_event`` /
``agent_job_event`` / ``agent_ui_command``) plus the run-level ``run_meta`` /
``run_done``, ``phase`` the annotate-run phase strip key.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from server.oreos.tool_kernel import ToolResultModel


class DemoStrictModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }


# ── Event vocabulary ──

# Existing socket names (web/packages/protocol/events.ts) the stock feed renders …
AGENT_EVENT_TYPES = (
    "agent_thought",
    "agent_finding",
    "agent_state",
    "agent_tool_event",
    "agent_job_event",
    "agent_ui_command",
)
# … plus the run-level bookends (protocol demo.ts DemoAgentRunMeta / run_done).
RUN_EVENT_TYPES = AGENT_EVENT_TYPES + ("run_meta", "run_done")

# Annotate-run phase strip (agent-exports.md "Run phasing"), in emit order.
RUN_PHASES = (
    "survey",
    "labels",
    "description",
    "dimensions",
    "key_features",
)


class RunEvent(DemoStrictModel):
    """One event on a run's feed (protocol ``demo.ts`` DemoAgentEvent + ``phase``)."""

    seq: int = Field(ge=0)
    ts: float
    type: Literal[
        "agent_thought",
        "agent_finding",
        "agent_state",
        "agent_tool_event",
        "agent_job_event",
        "agent_ui_command",
        "run_meta",
        "run_done",
    ]
    phase: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


# ── UI commands (superset — live union NEVER edited) ──

DEMO_UI_COMMAND_NAMES = (
    # live vocabulary (agent/schemas.py AgentUICommand)
    "focus_detection",
    "set_detection_queries",
    "add_detection_query",
    "remove_detection_query",
    "show_waypoint",
    "show_path",
    "show_toast",
    "open_detection_preview",
    # demo additions (protocol types.ts + demo.ts, additive contract PR)
    "drive_path",
    "recolor_region",
    "open_export",
    "focus_object",
    "show_measurement",
    "show_dimension_labels",
)


class DemoUICommand(DemoStrictModel):
    """Superset of the live ``AgentUICommand`` — same envelope, wider name union."""

    id: str = Field(min_length=4, max_length=64)
    name: Literal[
        "focus_detection",
        "set_detection_queries",
        "add_detection_query",
        "remove_detection_query",
        "show_waypoint",
        "show_path",
        "show_toast",
        "open_detection_preview",
        "drive_path",
        "recolor_region",
        "open_export",
        "focus_object",
        "show_measurement",
        "show_dimension_labels",
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    mission_id: Optional[int] = None
    ttl_ms: Optional[int] = Field(default=8000, ge=100, le=120000)


# ── Tool args (persisted-scene agent tool registry) ──


class ListSceneObjectsArgs(DemoStrictModel):
    include_dismissed: bool = False
    max_results: int = Field(default=100, ge=1, le=500)
    #: `compact` (uid/label/confidence) is the default because centres and extents are two
    #: thirds of a row and are not needed to REASON about the inventory — measure/plan
    #: resolve an object by ref and compute geometry in code. `full` adds the boxes back.
    detail: Literal["compact", "full"] = "compact"


class ObjectOrPoint(DemoStrictModel):
    """Exactly one of ``object`` (uid ``det:<idx>`` / index / label → OBB center) or
    ``point`` (world ``[x,y,z]``). Validated by the tool handler, not here, so the
    error surfaces as a graceful tool failure the agent can narrate."""

    object: Optional[str] = Field(default=None, max_length=160)
    point: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)


class MeasureDistanceArgs(DemoStrictModel):
    """Two endpoints, each an object reference or a world point (features.md F7:
    "points or object uids → centers")."""

    a: ObjectOrPoint
    b: ObjectOrPoint
    label: Optional[str] = Field(default=None, max_length=120)


class MeasureAngleArgs(DemoStrictModel):
    points_world: list[list[float]] = Field(min_length=3, max_length=3)
    label: Optional[str] = Field(default=None, max_length=120)


class PlanPathArgs(DemoStrictModel):
    """Goal = object reference or world point. Wired to W4's
    ``routes_nav.plan_path_for_scene`` when that module is present in the build;
    otherwise the handler answers a structured tool error the pilot narrates
    honestly. ``up_override`` (e.g. ``"-y"``) forces the planner's up axis on
    scenes without pose-derived gravity."""

    goal_object: Optional[str] = Field(default=None, max_length=160)
    goal_point: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    up_override: Optional[str] = Field(default=None, max_length=8)


class DescribeSceneArgs(DemoStrictModel):
    max_paragraphs: int = Field(default=3, ge=1, le=6)


class LabelObjectsArgs(DemoStrictModel):
    max_objects: int = Field(default=200, ge=1, le=500)


class EstimateDimensionsArgs(DemoStrictModel):
    object_uids: list[str] = Field(default_factory=list, max_length=64)


class KeyFeaturesArgs(DemoStrictModel):
    max_features: int = Field(default=5, ge=1, le=10)


class ExportNarrateArgs(DemoStrictModel):
    """No args — which export a tool narrates is fixed by the tool name
    (``export.lerobot`` / ``export.isaac``)."""


# ── export.build (the acting export tool — Phase 2) ──
#
# The narrate tools above answer "could this scene export?"; this one actually starts the
# build. Kept as a SEPARATE tool rather than a flag on the narrators so the effect is
# legible in the registry: narration is ``read``, this is ``spend`` + ``job``.

#: The formats the prepared-export job can build. Mirrors
#: ``export_artifacts.EXPORT_JOB_FORMATS`` so the pilot gets a real JSON-schema enum;
#: tests assert the two stay identical (a new format must be added in both places).
ExportBuildFormat = Literal["openreality", "groot_lerobot_v2", "isaac_usd"]


class ExportBuildArgs(DemoStrictModel):
    """Which export to BUILD. ``source`` names an alternate geometry (a derived
    anchor/clamp key); omit it for the scan's own geometry."""

    format: ExportBuildFormat
    source: Optional[str] = Field(default=None, max_length=200)


class ExportBuildResult(ToolResultModel):
    """What ``export.build`` returns — a HANDLE, never a finished archive.

    ``status`` is the whole contract:
      ``queued``   a container was spawned; watch ``job_id`` at ``poll_route``.
      ``active``   a build for this scene+format was ALREADY running; same handle, no
                   second container (the route's 409 ``export_job_active``, read as a
                   handle rather than an error because that is what it is).
      ``ready``    a current archive already exists; ``download_path`` serves it.
      ``blocked``  a gate refused before anything was started; the structured
                   ``capability``/``reason``/``repair_tool`` say what would unblock it.

    ``estimated_cost_usd`` is the budget figure the spend ceiling reserved — an
    order-of-magnitude container-time estimate, NOT a bill.
    """

    status: Literal["queued", "active", "ready", "blocked"]
    format: str
    job_id: Optional[str] = None
    poll_route: Optional[str] = None
    source_key: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    download_path: Optional[str] = None
    bytes: Optional[int] = None
    message: Optional[str] = None
    # Structured refusal fields (present only when status == "blocked").
    error: Optional[str] = None
    capability: Optional[str] = None
    reason: Optional[str] = None
    repair_tool: Optional[str] = None
    #: The seam's HTTP-ish status, so a reader of the run log can see exactly which
    #: branch of the shared route the agent hit.
    http_status: Optional[int] = None
