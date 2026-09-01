"""Tool-kernel tests (Phase 2 of the agent tool layer) — GPU-free, Modal-free.

Two halves:

  * the KERNEL in isolation (``server/oreos/tool_kernel.py``): declared result shapes,
    the idempotency ledger, the per-run spend ceiling, and projectors;
  * ``export.build`` on a real ``PersistedSceneAgent`` — the first tool that acts rather
    than narrates — over the same harness ``tests/test_oreos_agent.py`` uses (real Flask
    blueprint, real ``ModalScenePersistence`` on tmp_path, ``server.app`` stubbed, the
    export spawner swapped for a recorder).

What these lock down, in the order the money depends on them:
  * a handler whose return drifts from its declared shape fails LOUDLY, at the call site;
  * a retried spend call replays the first result and does NOT spawn a second container
    (``ToolRegistry.execute`` times out without cancelling its worker — the retry is the
    realistic path to a double charge);
  * a run that would cross ``SCENE_AGENT_MAX_SPEND_USD`` is refused BEFORE the spawn;
  * a blocked Isaac build still opens the export panel and says why, and a started build
    announces a ``job_id`` the client can keep watching after the run ends.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import types
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.oreos import export_artifacts as ea
from server.oreos import tool_kernel as tk
from server.oreos.schemas import DemoStrictModel, ExportBuildFormat, ToolResultModel
from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    SceneFacts,
    SceneMetrics,
    SceneReport,
)
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# kernel unit harness
# ---------------------------------------------------------------------------


class _Args(DemoStrictModel):
    n: int = 1


class _GoodResult(ToolResultModel):
    value: int


@pytest.fixture(autouse=True)
def _clean_idempotency():
    tk.reset_idempotency_cache()
    yield
    tk.reset_idempotency_cache()


def _kernel(**kwargs):
    return tk.ToolKernel(user_id="user-a", scan_id="scan-1", max_workers=2, **kwargs)


# ---------------------------------------------------------------------------
# result-model validation
# ---------------------------------------------------------------------------


def test_result_model_rejects_a_bad_shape_loudly():
    """A handler whose return drifts from its declared shape must fail at the call site.
    Silently forwarding it would put junk in the pilot's context and surface as strange
    behaviour several turns later, with nothing pointing at the real cause."""
    kernel = _kernel()
    kernel.register_tool(
        "good", "returns the declared shape", _Args,
        lambda a: {"value": a.n}, result_model=_GoodResult,
    )
    kernel.register_tool(
        "bad", "returns a drifted shape", _Args,
        lambda a: {"valeu": a.n}, result_model=_GoodResult,
    )

    assert kernel.execute("good", {"n": 3})["value"] == 3

    with pytest.raises(tk.tool_execution_error()) as exc:
        kernel.execute("bad", {"n": 3})
    message = str(exc.value)
    assert "does not match _GoodResult" in message
    # The validation error itself rides along, so the drift is diagnosable from the log.
    assert "valeu" in message and "value" in message


def test_result_validation_never_rewrites_the_result():
    """Validation is a gate, not a filter: the handler's own dict comes back, so no field
    is silently coerced or dropped on the way to the pilot."""

    class _Loose(ToolResultModel):
        value: int
        note: Optional[str] = None

    kernel = _kernel()
    handed_back = {"value": 1, "note": "kept"}
    kernel.register_tool(
        "t", "d", _Args, lambda a: dict(handed_back), result_model=_Loose
    )
    result = kernel.execute("t", {})
    assert result["note"] == "kept"
    assert result["value"] == 1


def test_unvalidated_tools_are_unaffected():
    """result_model is opt-in — the existing narrate/measure tools declare none and keep
    returning whatever they return."""
    kernel = _kernel()
    kernel.register_tool("t", "d", _Args, lambda a: {"anything": [1, 2, 3]})
    assert kernel.execute("t", {})["anything"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_idempotent_replay_does_not_re_run_the_handler():
    """THE double-charge guard. Same tool, same args, twice: the handler runs ONCE and the
    second call replays the first result (job id included)."""
    calls: list[int] = []

    def handler(args: _Args) -> dict:
        calls.append(args.n)
        return {"job_id": f"job-{len(calls)}", "status": "queued"}

    kernel = _kernel()
    kernel.register_tool("spawn", "d", _Args, handler, effect="spend", execution="job")

    first = kernel.execute("spawn", {"n": 1})
    second = kernel.execute("spawn", {"n": 1})

    assert calls == [1], "the handler must not run a second time"
    assert second["job_id"] == first["job_id"] == "job-1"
    assert first.get(tk.REPLAY_KEY) is None
    assert second[tk.REPLAY_KEY] is True, "a replay must be labelled, never disguised"


def test_idempotency_is_keyed_on_the_arguments_by_default():
    """The LLM does not mint idempotency keys — it re-calls with the same args. Different
    args are a different call and DO run."""
    calls: list[int] = []
    kernel = _kernel()
    kernel.register_tool(
        "spawn", "d", _Args,
        lambda a: (calls.append(a.n), {"job_id": f"job-{a.n}"})[1],
        effect="spend",
    )
    kernel.execute("spawn", {"n": 1})
    kernel.execute("spawn", {"n": 2})
    kernel.execute("spawn", {"n": 1})
    assert calls == [1, 2]


def test_explicit_idempotency_key_overrides_the_args_digest():
    calls: list[int] = []
    kernel = _kernel()
    kernel.register_tool(
        "spawn", "d", _Args,
        lambda a: (calls.append(a.n), {"job_id": "j"})[1],
        effect="spend",
    )
    kernel.execute("spawn", {"n": 1}, idempotency_key="op-7")
    kernel.execute("spawn", {"n": 2}, idempotency_key="op-7")
    assert calls == [1], "one logical operation, one execution"


def test_read_tools_are_not_idempotency_guarded():
    """A read must be free to re-run — the scene changes under it and a cached answer
    would be a lie."""
    calls: list[int] = []
    kernel = _kernel()
    kernel.register_tool(
        "look", "d", _Args, lambda a: (calls.append(a.n), {"ok": True})[1], effect="read"
    )
    kernel.execute("look", {"n": 1})
    kernel.execute("look", {"n": 1})
    assert calls == [1, 1]


def test_idempotency_is_scoped_per_scene():
    calls: list[str] = []

    def handler(args: _Args) -> dict:
        calls.append("ran")
        return {"job_id": "j"}

    a = tk.ToolKernel(user_id="user-a", scan_id="scan-1")
    b = tk.ToolKernel(user_id="user-a", scan_id="scan-2")
    for kernel in (a, b):
        kernel.register_tool("spawn", "d", _Args, handler, effect="spend")
    a.execute("spawn", {"n": 1})
    b.execute("spawn", {"n": 1})
    assert calls == ["ran", "ran"], "another scene's build is a different build"


# ---------------------------------------------------------------------------
# spend ceiling
# ---------------------------------------------------------------------------


def test_spend_ceiling_refuses_before_the_handler_spawns(monkeypatch):
    """Refusing after the container is up is not refusing. The handler must never run."""
    monkeypatch.setenv(tk.MAX_SPEND_ENV, "0.5")
    calls: list[str] = []

    def handler(args: _Args) -> dict:
        calls.append("spawned")
        return {"job_id": "j"}

    kernel = _kernel()
    kernel.register_tool(
        "build", "d", _Args, handler, effect="spend", execution="job",
        estimated_cost_usd=0.30,
    )

    kernel.execute("build", {"n": 1})
    assert calls == ["spawned"] and kernel.spend_usd == pytest.approx(0.30)

    with pytest.raises(tk.tool_execution_error()) as exc:
        kernel.execute("build", {"n": 2})  # 0.30 + 0.30 > 0.5

    payload = getattr(exc.value, "payload", None)
    assert isinstance(payload, dict)
    assert payload["error"] == "precondition_failed"
    assert payload["capability"] == "spend_budget"
    assert payload["ceiling_usd"] == pytest.approx(0.5)
    assert payload["estimate_usd"] == pytest.approx(0.30)
    assert "Nothing was started" in payload["reason"]
    assert calls == ["spawned"], "the refused call must not have spawned anything"
    assert kernel.spend_usd == pytest.approx(0.30), "a refusal does not consume budget"


def test_spend_ceiling_default_and_env_override(monkeypatch):
    monkeypatch.delenv(tk.MAX_SPEND_ENV, raising=False)
    assert _kernel().max_spend_usd == pytest.approx(tk.DEFAULT_MAX_SPEND_USD)
    monkeypatch.setenv(tk.MAX_SPEND_ENV, "2.5")
    assert _kernel().max_spend_usd == pytest.approx(2.5)
    monkeypatch.setenv(tk.MAX_SPEND_ENV, "not-a-number")
    assert _kernel().max_spend_usd == pytest.approx(tk.DEFAULT_MAX_SPEND_USD)


def test_settle_releases_the_reservation_when_nothing_spawned(monkeypatch):
    """A short-circuit ("already prepared") costs nothing new, so it must not burn the
    run's ceiling on work it did not cause."""
    monkeypatch.setenv(tk.MAX_SPEND_ENV, "1.0")
    kernel = _kernel()
    kernel.register_tool(
        "build", "d", _Args, lambda a: {"status": "ready"}, effect="spend",
        estimated_cost_usd=0.4,
        settle_cost=lambda result, reserved: reserved if result.get("status") == "queued" else 0.0,
    )
    kernel.execute("build", {"n": 1})
    assert kernel.spend_usd == pytest.approx(0.0)


def test_precondition_refusal_skips_the_handler():
    calls: list[str] = []
    kernel = _kernel()
    kernel.register_tool(
        "build", "d", _Args, lambda a: (calls.append("ran"), {"ok": True})[1],
        effect="spend",
        precondition=lambda args: tk.precondition_failure(
            "metric_anchor", "needs metres", repair_tool="export.isaac"
        ),
    )
    with pytest.raises(tk.tool_execution_error()) as exc:
        kernel.execute("build", {})
    payload = exc.value.payload
    assert payload["capability"] == "metric_anchor"
    assert payload["repair_tool"] == "export.isaac"
    assert calls == []


def test_policy_error_is_catchable_as_a_tool_execution_error():
    """The agent's ``except _tool_error()`` has to catch a refusal — a same-named class
    from a second import of tool_registry.py would sail past it and kill the run."""
    assert issubclass(tk.tool_policy_error(), tk.tool_execution_error())


# ---------------------------------------------------------------------------
# projectors
# ---------------------------------------------------------------------------


def test_projector_emits_two_commands_from_one_result():
    """plan_path is why the projector is a callable and not a static list: one result,
    two commands, and the arguments are read off the RESULT."""
    from server.oreos.persisted_agent import _project_plan_path

    commands = _project_plan_path(
        {"path_id": "p1", "waypoints_world": [[0, 0, 0], [1, 0, 0]]}
    )
    assert [name for name, _ in commands] == ["show_path", "drive_path"]
    assert commands[0][1] == {"points": [[0, 0, 0], [1, 0, 0]], "path_id": "p1"}
    assert commands[1][1] == {"path_id": "p1", "waypoints_world": [[0, 0, 0], [1, 0, 0]]}


def test_projector_emits_nothing_when_the_planner_returned_no_waypoints():
    from server.oreos.persisted_agent import _project_plan_path

    assert _project_plan_path({"path_id": "p1", "waypoints_world": []}) == []


def test_export_build_projector_is_conditional():
    from server.oreos.persisted_agent import _project_export_build

    started = _project_export_build({"format": "openreality", "status": "queued"})
    assert [n for n, _ in started] == ["open_export"]

    blocked = _project_export_build(
        {
            "format": "isaac_usd",
            "status": "blocked",
            "error": "precondition_failed",
            "reason": "needs a metric anchor",
            "repair_tool": "export.isaac",
        }
    )
    assert [n for n, _ in blocked] == ["open_export", "show_toast"]
    assert blocked[0][1] == {"format": "isaac_usd"}
    assert "metric anchor" in blocked[1][1]["message"]
    assert "export.isaac" in blocked[1][1]["message"]


def test_kernel_project_uses_the_registered_spec():
    kernel = _kernel()
    kernel.register_tool(
        "t", "d", _Args, lambda a: {"v": 2},
        projector=lambda result: [("show_toast", {"message": str(result["v"])})],
    )
    assert kernel.project("t", {"v": 2}) == [("show_toast", {"message": "2"})]
    assert kernel.project("unregistered", {}) == []


def test_list_tools_keeps_the_shape_the_loop_consumes():
    kernel = _kernel()
    kernel.register_tool("t", "a description", _Args, lambda a: {}, effect="spend",
                         execution="job")
    row = kernel.list_tools()[0]
    assert row["name"] == "t"
    assert row["description"] == "a description"
    assert "properties" in row["args_schema"]
    # additive metadata: the pilot should know which calls cost money
    assert row["effect"] == "spend" and row["execution"] == "job"


# ---------------------------------------------------------------------------
# export.build on a real agent
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [
        m
        for m in list(sys.modules)
        if m == "server.oreos"
        or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


def _save_scene(store, user="user-a", scan="scan-1", anchored=False):
    objects = [
        ObjectInstance(
            query="desk",
            center=[0.0, 0.0, 0.0],
            extent=[2.0, 1.0, 1.0],
            confidence=0.9,
            evidence=[EvidenceRef(submap_id=0, frame_idx=0)],
        )
    ]
    facts = SceneFacts(
        metrics=SceneMetrics(
            dimensions=[5.0, 4.0, 2.0],
            bbox_min=[-1.0, -1.0, 0.0],
            bbox_max=[4.0, 3.0, 2.0],
            num_submaps=1,
            num_keyframes=2,
            point_count=1000,
        ),
        objects=objects,
        object_counts={"desk": 1},
        coverage_estimate=0.5,
    )
    report = SceneReport(summary="An office.", room_type="office", observations=[])
    store.save_scene(
        user,
        scan,
        report,
        facts,
        keyframes_b64=[
            {"submap_id": 0, "frame_idx": 0, "image_b64": base64.b64encode(b"j").decode()}
        ],
        trajectory={
            "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            "intrinsics": np.ones((2, 4), dtype=np.float32),
            "source_frame_id": np.arange(2, dtype=np.float32),
        },
        label="fixture-office",
        source="recon_video",
    )
    if anchored:
        store.set_derived_pointer(
            user,
            scan,
            {
                "kind": "anchor",
                "cloud_key": "derived/anchor/t/cloud.ply",
                "source_key": "derived/anchor/t/cloud.ply",
                "scale_factor": 2.0,
                "applied_at": "2026-07-30T00:00:00+00:00",
            },
        )


@pytest.fixture()
def build_app(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SCENE_AGENT_TOOLS", raising=False)
    monkeypatch.delenv(tk.MAX_SPEND_ENV, raising=False)
    demo_pkg = _fresh_demo_package()
    agent_mod = importlib.import_module("server.oreos.persisted_agent")
    runlog_mod = importlib.import_module("server.oreos.runlog")

    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    jobs = sys.modules["server.oreos.jobs"]
    jobs_store: dict = {}
    jobs.configure_jobs_store(jobs_store)
    routes = sys.modules["server.oreos.routes_export_job"]
    spawned: list[dict] = []
    routes.configure_export_spawner(lambda **kw: spawned.append(kw))

    _save_scene(store)
    _save_scene(store, scan="scan-anchored", anchored=True)

    def agent_for(scan="scan-1", kind="pilot"):
        record = store.get_scene("user-a", scan)
        run = runlog_mod.AgentRun(f"t-{scan}-{len(spawned)}", "user-a", scan, kind)
        return agent_mod.PersistedSceneAgent(store, "user-a", scan, record, run), run

    yield SimpleNamespace(
        store=store,
        stub=stub,
        agent=agent_mod,
        runlog=runlog_mod,
        routes=routes,
        spawned=spawned,
        jobs_store=jobs_store,
        agent_for=agent_for,
        client=app.test_client(),
    )
    routes.configure_export_spawner(None)
    jobs.configure_jobs_store(None)
    runlog_mod.REGISTRY.reset()
    agent_mod.LLM_CLIENT_FACTORY = None


def test_export_build_formats_track_the_prepared_job_formats():
    """The pilot gets a JSON-schema enum; a new prepared format must be added in both
    places or this fails rather than silently going uncallable."""
    from typing import get_args

    assert set(get_args(ExportBuildFormat)) == set(ea.EXPORT_JOB_FORMATS)


def test_export_build_returns_a_handle_immediately(build_app):
    """It must not block: an Isaac export measured 382 s against a 150 s request timeout,
    and a polling LLM holds the scene's only run slot the whole time."""
    agent, run = build_app.agent_for()
    result = agent.registry.execute("export.build", {"format": "openreality"})

    assert result["status"] == "queued"
    assert result["job_id"]
    assert result["poll_route"] == f"/api/workspace/jobs/{result['job_id']}"
    assert result["format"] == "openreality"
    assert result["estimated_cost_usd"] > 0
    assert "download_path" not in result or result["download_path"] is None

    # It rode the EXISTING scheduler — one spawn, on the real spawner seam.
    assert len(build_app.spawned) == 1
    assert build_app.spawned[0]["export_format"] == "openreality"
    assert build_app.spawned[0]["scan_id"] == "scan-1"
    # …and seeded the existing workspace-jobs record the client already polls.
    assert build_app.jobs_store[result["job_id"]]["kind"] == "export"


def test_export_build_retry_does_not_spawn_a_second_container(build_app):
    """The realistic double-charge: the loop's future.result() times out without
    cancelling the worker, the pilot retries the identical call. One container only."""
    agent, _run = build_app.agent_for()
    first = agent.registry.execute("export.build", {"format": "openreality"})
    second = agent.registry.execute("export.build", {"format": "openreality"})
    assert second["job_id"] == first["job_id"]
    assert second[tk.REPLAY_KEY] is True
    assert len(build_app.spawned) == 1


def test_export_build_isaac_is_refused_without_a_metric_anchor(build_app):
    """Guardrail 3 with money attached: no metric anchor, no Isaac build, nothing
    spawned — and a repair the agent can offer."""
    build_app.stub._resolve_isaac_scale = lambda record, source, scale: None
    agent, _run = build_app.agent_for()

    with pytest.raises(tk.tool_execution_error()) as exc:
        agent.registry.execute("export.build", {"format": "isaac_usd"})

    payload = exc.value.payload
    assert payload["error"] == "precondition_failed"
    assert payload["capability"] == "metric_anchor"
    assert payload["repair_tool"] == "export.isaac"
    assert "metric anchor" in payload["reason"]
    assert build_app.spawned == []


def test_export_build_isaac_fails_closed_when_the_gate_is_unreachable(build_app):
    """No gate helper, no authorisation. Refusing a build we could not gate is the right
    failure; spawning it and hoping is not."""
    assert not hasattr(build_app.stub, "_resolve_isaac_scale")
    agent, _run = build_app.agent_for()
    with pytest.raises(tk.tool_execution_error()) as exc:
        agent.registry.execute("export.build", {"format": "isaac_usd"})
    assert exc.value.payload["capability"] == "metric_anchor"
    assert build_app.spawned == []


def test_export_build_isaac_proceeds_once_the_gate_passes(build_app):
    build_app.stub._resolve_isaac_scale = lambda record, source, scale: (
        "derived/anchor/t/cloud.ply",
        1.0,
        "anchor:prescaled",
        2.0,
    )
    agent, _run = build_app.agent_for(scan="scan-anchored")
    result = agent.registry.execute("export.build", {"format": "isaac_usd"})
    assert result["status"] == "queued"
    assert build_app.spawned[0]["export_format"] == "isaac_usd"


def test_export_build_spend_ceiling_refuses_the_run_that_would_cross_it(build_app, monkeypatch):
    """With per-action confirmation deliberately off, this ceiling is the only brake
    between a mis-parsed instruction and a real bill."""
    monkeypatch.setenv(tk.MAX_SPEND_ENV, "0.02")
    build_app.stub._resolve_isaac_scale = lambda record, source, scale: (
        "derived/anchor/t/cloud.ply", 1.0, "anchor:prescaled", 2.0,
    )
    agent, _run = build_app.agent_for(scan="scan-anchored")

    ok = agent.registry.execute("export.build", {"format": "openreality"})
    assert ok["status"] == "queued"

    with pytest.raises(tk.tool_execution_error()) as exc:
        agent.registry.execute("export.build", {"format": "isaac_usd"})  # 0.30 > 0.02
    payload = exc.value.payload
    assert payload["capability"] == "spend_budget"
    assert [s["export_format"] for s in build_app.spawned] == ["openreality"]


def test_export_build_emits_ui_and_a_job_event_through_the_loop(build_app):
    """The full ``_execute_tool`` path: open the panel from the RESULT, and announce the
    job_id so the client keeps watching after the run ends."""
    agent, run = build_app.agent_for()
    ok, result = agent._execute_tool("export.build", {"format": "groot_lerobot_v2"}, "chat")
    assert ok and result["status"] == "queued"

    events = run.snapshot_events()
    ui = [e["payload"] for e in events if e["type"] == "agent_ui_command"]
    assert [c["name"] for c in ui] == ["open_export"]
    assert ui[0]["args"]["format"] == "groot_lerobot_v2"

    jobs = [e["payload"] for e in events if e["type"] == "agent_job_event"]
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == result["job_id"]
    assert jobs[0]["kind"] == "export"
    assert jobs[0]["poll_route"] == f"/api/workspace/jobs/{result['job_id']}"
    assert jobs[0]["scan_id"] == "scan-1"
    assert jobs[0]["tool"] == "export.build"


def test_blocked_export_build_still_opens_the_panel_and_says_why(build_app):
    """A refusal the operator cannot see is indistinguishable from a hang."""
    build_app.stub._resolve_isaac_scale = lambda record, source, scale: None
    agent, run = build_app.agent_for()
    ok, outcome = agent._execute_tool("export.build", {"format": "isaac_usd"}, "chat")

    assert ok is False
    assert isinstance(outcome, dict) and outcome["error"] == "precondition_failed"
    assert outcome["repair_tool"] == "export.isaac"

    events = run.snapshot_events()
    ui = [e["payload"] for e in events if e["type"] == "agent_ui_command"]
    assert [c["name"] for c in ui] == ["open_export", "show_toast"]
    assert ui[0]["args"]["format"] == "isaac_usd"
    assert "metric anchor" in ui[1]["args"]["message"]

    tool_events = [
        e["payload"] for e in events if e["type"] == "agent_tool_event"
    ]
    failed = [t for t in tool_events if t["status"] == "failed"]
    assert failed and failed[0]["result"]["capability"] == "metric_anchor"
    assert [e for e in events if e["type"] == "agent_job_event"] == []
    assert build_app.spawned == []


def test_export_build_reports_an_already_prepared_archive_without_spending(
    build_app, tmp_path
):
    """A current archive is handed back as a download, not rebuilt — and the reservation
    is released, so the ceiling is not burned on work that did not happen."""
    slot_bytes = b"ZIPBYTES"
    blob = tmp_path / "openreality.src.zip"
    blob.write_bytes(slot_bytes)
    build_app.store.save_derived_artifact_file(
        "user-a", "scan-1", ea.zip_relative_key("openreality"), str(blob), move=True
    )
    slot = ea.slot_record(
        export_format="openreality",
        source_key="original",
        zip_bytes=len(slot_bytes),
        tree_bytes=len(slot_bytes) * 2,
        file_count=3,
        build_seconds=45.1,
    )
    index = ea.merge_slot(ea.read_index(build_app.store, "user-a", "scan-1"), slot)
    build_app.store.save_derived_artifact(
        "user-a", "scan-1", ea.INDEX_RELATIVE_KEY, ea.index_json_bytes(index)
    )

    agent, _run = build_app.agent_for()
    result = agent.registry.execute("export.build", {"format": "openreality"})
    assert result["status"] == "ready"
    assert result["download_path"].endswith("/derived/demo/export/openreality.zip")
    assert result["bytes"] == len(slot_bytes)
    assert build_app.spawned == [], "a fresh archive must not be rebuilt"
    assert agent.registry.spend_usd == pytest.approx(0.0)


def test_export_build_result_shape_is_enforced(build_app):
    """The declared ExportBuildResult is live on the real tool, not just in the tests."""
    from server.oreos.schemas import ExportBuildResult

    agent, _run = build_app.agent_for()
    spec = agent.registry.spec("export.build")
    assert spec.result_model is ExportBuildResult
    assert spec.effect == "spend" and spec.execution == "job"
    with pytest.raises(tk.tool_execution_error()):
        spec.validate_result({"status": "queued", "format": "openreality", "bogus": 1})


def test_export_narration_tools_are_unchanged(build_app):
    """Additive only: the narrate tools keep their behaviour and their UI projection."""
    agent, run = build_app.agent_for()
    ok, result = agent._execute_tool("export.isaac", {}, "chat")
    assert ok and result["metric_gate"] == "blocked"
    ui = [e["payload"] for e in run.snapshot_events() if e["type"] == "agent_ui_command"]
    assert [c["name"] for c in ui] == ["open_export", "show_toast"]
    assert build_app.spawned == [], "narration never builds"


def test_prepare_export_seam_matches_the_route(build_app):
    """One scheduler: the seam the agent calls and the route the panel posts to answer the
    same thing, and the route's 409 guard sees the agent's job."""
    from server.oreos.routes_export_job import prepare_export_for_scene

    payload, status = prepare_export_for_scene(
        "user-a", "scan-1", {"format": "openreality"}, store=build_app.store
    )
    assert status == 202 and payload["status"] == "queued"

    # The panel's POST now collides with the agent's in-flight build instead of
    # fanning out a second container.
    build_app.jobs_store[payload["job_id"]] = {
        "job_id": payload["job_id"], "user_id": "user-a", "status": "running", "stage": "build",
    }
    r = build_app.client.post("/api/scenes/scan-1/demo/export/prepare", json={})
    assert r.status_code == 409
    assert r.get_json()["job_id"] == payload["job_id"]


def test_prepare_export_seam_reports_route_errors_as_payloads(build_app):
    from server.oreos.routes_export_job import prepare_export_for_scene

    payload, status = prepare_export_for_scene(
        "user-a", "scan-1", {"format": "nonsense"}, store=build_app.store
    )
    assert status == 400 and payload["error"] == "invalid_format"

    payload, status = prepare_export_for_scene(
        "user-a", "no-such-scan", {}, store=build_app.store
    )
    assert status == 404 and payload["error"] == "not_found"


# ── presenting a result to the model ───────────────────────────────────────────
#
# Regression tests for a defect found on the LIVE deployment (2026-08-04): asked to
# measure to a "coat rack", the agent replied that no such object existed and that it
# would not invent one — while det:36 and det:55, both labelled "coat rack" and both
# rendered on screen, sat past a silent cut in the inventory it had been handed.


def test_present_result_keeps_a_whole_scene_inventory():
    """95 compact objects must arrive INTACT — this is the live failure."""
    from server.oreos.tool_kernel import present_result

    objects = [
        {"uid": f"det:{i}", "label": "coat rack" if i in (36, 55) else "cubicle partition",
         "confidence": 0.171}
        for i in range(95)
    ]
    out = present_result({"objects": objects, "count": 95, "total_objects": 95})

    assert json.loads(out)["count"] == 95
    assert "_truncated" not in out
    # The specific objects the deployed agent denied the existence of.
    assert '"det:36"' in out and '"det:55"' in out


def test_present_result_shrinks_structurally_and_says_so():
    """When it must cut, the payload stays parseable AND announces the cut."""
    from server.oreos.tool_kernel import present_result

    objects = [{"uid": f"det:{i}", "label": "x" * 300} for i in range(200)]
    parsed = json.loads(present_result({"objects": objects}))  # must be valid JSON

    trunc = parsed["_truncated"]
    assert trunc["total"] == 200
    assert trunc["shown"] == len(parsed["objects"]) < 200
    # The agent must be told the rest EXIST, or it will report them absent — which is
    # exactly what happened in production.
    assert "EXIST" in trunc["note"]


def test_present_result_refuses_rather_than_hand_over_a_fragment():
    """No list to shrink → refuse. A partial reads as a complete one."""
    from server.oreos.tool_kernel import present_result

    parsed = json.loads(present_result({"blob": "y" * 50_000}))
    assert parsed["error"] == "result_too_large"
    assert "y" * 100 not in json.dumps(parsed)  # no fragment of the payload leaked


def test_old_truncation_produced_invalid_json():
    """Characterises the defect: the previous `[:4000]` cut mid-value."""
    objects = [
        {"uid": f"det:{i}", "label": "possible coat rack", "query": "coat rack",
         "human_label": None, "confidence": 0.171,
         "center": [-0.199, -0.328, 1.727], "extent": [0.4, 1.2, 0.35], "dismissed": False}
        for i in range(95)
    ]
    encoded = json.dumps({"objects": objects, "count": 95})
    assert len(encoded) > 4000
    with pytest.raises(json.JSONDecodeError):
        json.loads(encoded[:4000] + "…(truncated)")


# ── degradation honesty ────────────────────────────────────────────────────────
#
# Found live: every fallback said "(LLM unavailable; fact-only answer)", including the
# common case where the provider answered perfectly and the model merely replied in prose
# instead of JSON. That message sent debugging at a missing API key that was never missing.


def test_fallback_cause_labels_stay_honest():
    # The mechanism moved with the main merge: call sites now pass an explicit
    # ``cause`` (rendered via _FALLBACK_CAUSES) instead of parsing the reason string
    # with _degradation_note. The honesty contract is unchanged: a model that
    # answered badly must never be reported as an outage.
    from server.oreos.persisted_agent import _FALLBACK_CAUSES, is_format_error

    # Only a genuinely absent client may claim an LLM outage.
    assert _FALLBACK_CAUSES["llm_unavailable"] == "LLM unavailable"
    assert _FALLBACK_CAUSES["bad_format"] == "model answered but broke format"
    assert _FALLBACK_CAUSES["call_failed"] == "model call failed"

    # The classifier that routes to bad_format: a typed LLMFormatError carrying the
    # prose completion is a format slip, not a failure; a transport error is not.
    class LLMFormatError(Exception):
        pass

    exc = LLMFormatError("model replied in prose")
    exc.content = "the prose answer"
    assert is_format_error(exc)
    assert not is_format_error(RuntimeError("connection reset by peer"))


def test_json_shape_errors_are_told_apart_from_real_failures():
    from server.oreos.persisted_agent import _is_json_shape_error

    assert _is_json_shape_error(RuntimeError("No JSON object found in LLM response"))
    assert _is_json_shape_error(json.JSONDecodeError("Expecting value", "", 0))
    # A transport failure is NOT a retryable shape slip — retrying would waste spend.
    assert not _is_json_shape_error(RuntimeError("connection reset by peer"))
    assert not _is_json_shape_error(TimeoutError("deadline"))
