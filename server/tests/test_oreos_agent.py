"""Scene-agent tests (W2, docs/demo-2026-07 agent-exports.md Part 1) — GPU-free.

Harness = W0's test_demo_routes pattern: REAL Flask + test_client over the demo
blueprint, ``server.app`` stubbed via ``sys.modules`` (auth helper + a real
``ModalScenePersistence`` on tmp_path). LLM = scripted mocks injected through
``persisted_agent.LLM_CLIENT_FACTORY`` — no network, no key.

Covers: run/poll lifecycle (phases in order, monotonic seq, run_meta/run_done),
one-active-run 409, replay (byte-identical + structural replay flag, both the
replay-run path and ``?replay=1``), closed-world validator drops, numbers-from-code
prose gate, metric gating (relative ↔ metres wording/values), human-label +
confidence propagation, chat (grounded-QA + tool-intent), pilot loop with the W4
plan_path seam, LLM budget degradation, and the OpenRouter client usage wiring.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    SceneFacts,
    SceneMetrics,
    SceneReport,
)
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [
        m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def demo_app(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SCENE_AGENT_DISABLE_LLM", raising=False)
    demo_pkg = _fresh_demo_package()
    agent_mod = importlib.import_module("server.oreos.persisted_agent")
    runlog_mod = importlib.import_module("server.oreos.runlog")

    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(
        _scene_persistence=store,
        _auth_user_id=lambda: "user-a",
    )
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    yield SimpleNamespace(
        client=app.test_client(),
        store=store,
        stub=stub,
        agent=agent_mod,
        runlog=runlog_mod,
    )
    runlog_mod.REGISTRY.reset()
    agent_mod.LLM_CLIENT_FACTORY = None


def _save_scene(
    store,
    user="user-a",
    scan="scan1",
    anchored=False,
    with_trajectory=True,
    with_keyframes=True,
):
    objects = [
        ObjectInstance(
            query="desk",
            center=[0.0, 0.0, 0.0],
            extent=[2.0, 1.0, 1.0],
            confidence=0.9,
            evidence=[EvidenceRef(submap_id=0, frame_idx=0)],
        ),
        ObjectInstance(
            query="chair",
            center=[3.0, 4.0, 0.0],
            extent=[0.5, 0.5, 1.0],
            confidence=0.8,
            evidence=[EvidenceRef(submap_id=0, frame_idx=1)],
        ),
        ObjectInstance(
            query="plant", center=[1.0, 1.0, 0.0], extent=[0.3, 0.3, 0.6], confidence=0.2
        ),
        ObjectInstance(
            query="lamp",
            center=[2.0, 0.0, 0.0],
            extent=[0.2, 0.2, 0.5],
            confidence=0.7,
            human_label="floor lamp",
        ),
        ObjectInstance(
            query="box", center=[0.0, 2.0, 0.0], extent=[0.4, 0.4, 0.4], confidence=0.6
        ),
    ]
    objects[4].dismissed = True
    facts = SceneFacts(
        metrics=SceneMetrics(
            dimensions=[5.0, 4.0, 2.0],
            bbox_min=[-1.0, -1.0, 0.0],
            bbox_max=[4.0, 3.0, 2.0],
            num_submaps=3,
            num_keyframes=12,
            point_count=50_000,
        ),
        objects=objects,
        object_counts={"desk": 1, "chair": 1, "plant": 1, "lamp": 1, "box": 1},
        coverage_estimate=0.7,
    )
    report = SceneReport(
        summary="An office with a desk and chairs.",
        room_type="office",
        observations=["The desk faces the window wall.", "Two seating positions."],
    )
    keyframes_b64 = (
        [
            {"submap_id": 0, "frame_idx": 0, "image_b64": base64.b64encode(b"jpeg0").decode()},
            {"submap_id": 0, "frame_idx": 1, "image_b64": base64.b64encode(b"jpeg1").decode()},
        ]
        if with_keyframes
        else None
    )
    trajectory = (
        {
            "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            "intrinsics": np.ones((2, 4), dtype=np.float32),
            "source_frame_id": np.arange(2, dtype=np.float32),
        }
        if with_trajectory
        else None
    )
    store.save_scene(
        user,
        scan,
        report,
        facts,
        keyframes_b64=keyframes_b64,
        trajectory=trajectory,
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


class ScriptedLLM:
    """chat_json pops scripted parsed dicts in order (or raises a scripted error).
    ``gate`` (a threading.Event) blocks the FIRST call until set — for 409 tests."""

    def __init__(self, script, gate=None, model="mock-model"):
        self.script = list(script)
        self.calls: list[dict] = []
        self.gate = gate
        self.model = model
        self._first = True

    def chat_json(self, **kwargs):
        if self.gate is not None and self._first:
            self._first = False
            assert self.gate.wait(timeout=15), "test gate never released"
        self.calls.append(kwargs)
        if not self.script:
            raise RuntimeError("mock LLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, SimpleNamespace(model=self.model, degraded=False)


def _factory_for(by_role):
    def factory(role, model, fallbacks, tally):
        return by_role.get(role)

    return factory


def _wait_run(env, run_id, timeout=20.0):
    run = env.runlog.REGISTRY.get(run_id)
    assert run is not None, f"run {run_id} not in registry"
    if run.thread is not None:
        run.thread.join(timeout=timeout)
    assert run.status != "running", "run did not finish in time"
    return run


def _events_of(env, scan, run_id, **params):
    resp = env.client.get(
        f"/api/scenes/{scan}/demo/agent/runs/{run_id}/events",
        query_string=params or None,
    )
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


# ---------------------------------------------------------------------------
# annotate lifecycle
# ---------------------------------------------------------------------------


def _standard_annotate_mocks():
    annotator = ScriptedLLM(
        [
            {
                "notes": [
                    {"object_id": "det:0", "note": "wooden surface"},
                    {"object_id": "det:999", "note": "fabricated object"},
                ]
            },
            {
                "features": [
                    {
                        "claim": "Desk-centric workspace",
                        "object_uids": ["det:0"],
                        "keyframes": [{"submap_id": 0, "frame_idx": 0}],
                    },
                    {"claim": "Ghost claim", "object_uids": ["det:42"], "keyframes": []},
                ]
            },
        ]
    )
    narrator = ScriptedLLM(
        [
            {
                "paragraphs": [
                    "A compact office arranged around the desk.",
                    "The room is 3 m wide.",  # metric claim → must be dropped
                ]
            }
        ]
    )
    return {"annotator": annotator, "narrator": narrator}


def test_annotate_full_lifecycle(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for(_standard_annotate_mocks())

    resp = env.client.post("/api/scenes/scan1/demo/agent/annotate", json={"mode": "full"})
    assert resp.status_code == 202
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)

    body = _events_of(env, "scan1", run_id, after=-1)
    assert body["status"] == "done"
    events = body["events"]
    # next = LAST seq (protocol demo.ts: "Pass as the next ?after=") — passing it
    # back verbatim must yield zero new events, never skip one.
    assert body["next"] == len(events) - 1
    assert _events_of(env, "scan1", run_id, after=body["next"])["events"] == []

    # monotonic seq from 0
    assert [e["seq"] for e in events] == list(range(len(events)))
    # run_meta first, run_done last
    assert events[0]["type"] == "run_meta"
    assert events[0]["payload"]["replay"] is False
    assert events[0]["payload"]["models"]["pilot"] == "anthropic/claude-sonnet-5"
    assert events[0]["payload"]["models"]["annotator"] == "google/gemini-3-flash-preview"
    assert events[0]["payload"]["models"]["narrator"] == "anthropic/claude-opus-5"
    assert events[-1]["type"] == "run_done"
    done = events[-1]["payload"]
    assert done["status"] == "done"
    assert done["llm_calls_attempted"] == 3  # notes + narrator + key_features

    # phases appear in spec order
    phase_order = []
    for e in events:
        p = e.get("phase")
        if p and p not in phase_order:
            phase_order.append(p)
    assert phase_order == ["survey", "labels", "description", "dimensions", "key_features"]

    # findings honour the closed world + label rules
    findings = [e["payload"] for e in events if e["type"] == "agent_finding"]
    by_uid = {}
    for f in findings:
        by_uid.setdefault(f.get("object_uid"), []).append(f)
    assert "det:999" not in by_uid
    assert "det:4" not in by_uid  # dismissed object never labeled
    desk_labels = [f for f in by_uid["det:0"] if f.get("label_source")]
    assert any("wooden surface" in f["description"] for f in desk_labels)
    plant = [f for f in by_uid["det:2"] if f.get("label_source")][0]
    assert plant["description"].startswith("possible plant")
    lamp = [f for f in by_uid["det:3"] if f.get("label_source")][0]
    assert lamp["query"] == "floor lamp"  # human_label precedence
    assert lamp["label_source"] == "human"

    # description: narrator paragraph kept, metric-claiming paragraph dropped
    thoughts = [e["payload"]["content"] for e in events if e["type"] == "agent_thought"]
    assert any("compact office arranged around the desk" in t for t in thoughts)
    assert not any("3 m wide" in t for t in thoughts)

    # key features: grounded claim kept, ghost claim dropped
    feature_findings = [f for f in findings if f.get("query") == "key feature"]
    assert any("Desk-centric workspace" in f["description"] for f in feature_findings)
    assert not any("Ghost claim" in f["description"] for f in feature_findings)
    assert done["findings_dropped"] >= 2

    # dimensions: relative wording (unanchored), UI label toggle emitted
    dim_findings = [
        e["payload"]
        for e in events
        if e["type"] == "agent_finding" and e.get("phase") == "dimensions"
    ]
    assert dim_findings and all(f["units"] == "relative" for f in dim_findings)
    room = [f for f in dim_findings if f["query"] == "room extents"][0]
    assert "units (relative, uncalibrated)" in room["description"]
    ui = [e["payload"] for e in events if e["type"] == "agent_ui_command"]
    assert any(u["name"] == "show_dimension_labels" for u in ui)
    assert any(u["name"] == "focus_object" and u["args"]["object_uid"] == "det:0" for u in ui)

    # persisted artifact + manifest rich ref
    raw = env.store.get_derived_artifact(
        "user-a", "scan1", f"derived/demo/agent_runs/{run_id}/events.json"
    )
    doc = json.loads(raw.decode("utf-8"))
    assert doc["run_id"] == run_id
    assert doc["status"] == "done"
    assert len(doc["events"]) == len(events)
    manifest = env.client.get("/api/scenes/scan1/demo/manifest").get_json()
    refs = [r for r in manifest["agent_runs"] if isinstance(r, dict)]
    assert any(
        r["run_id"] == run_id
        and r["kind"] == "annotate"
        and r["events_key"] == f"derived/demo/agent_runs/{run_id}/events.json"
        for r in refs
    )


def test_annotate_fact_only_needs_no_llm(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})  # every role → None

    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    )
    assert resp.status_code == 202
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)
    body = _events_of(env, "scan1", run_id)
    assert body["status"] == "done"
    types_seen = {e["type"] for e in body["events"]}
    assert {"run_meta", "agent_thought", "agent_finding", "agent_state", "run_done"} <= types_seen
    done = body["events"][-1]["payload"]
    assert done["llm_calls_attempted"] == 0
    assert done["cost_usd"] is None


def test_annotate_metric_gating_anchored(demo_app):
    env = demo_app
    _save_scene(env.store, scan="scan-m", anchored=True)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})

    resp = env.client.post(
        "/api/scenes/scan-m/demo/agent/annotate", json={"mode": "fact_only"}
    )
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan-m", run_id)["events"]
    assert events[0]["payload"]["metric"]["anchored"] is True
    dim_findings = [
        e["payload"]
        for e in events
        if e["type"] == "agent_finding" and e.get("phase") == "dimensions"
    ]
    room = [f for f in dim_findings if f["query"] == "room extents"][0]
    assert room["units"] == "m"
    assert room["metric_state"] == "metric_anchored"
    # dims [5,4,2] × scale 2.0 → [10, 8, 4]
    assert room["values"] == [10.0, 8.0, 4.0]
    assert "10.00 × 8.00 × 4.00 m" in room["description"]
    desk = [f for f in dim_findings if f.get("object_uid") == "det:0"][0]
    assert desk["units"] == "m"
    assert desk["extents"] == [4.0, 2.0, 2.0]
    assert "4.00 m" in desk["description"]


def test_second_run_409_while_active(demo_app):
    env = demo_app
    _save_scene(env.store)
    gate = threading.Event()
    mocks = _standard_annotate_mocks()
    mocks["annotator"].gate = gate
    env.agent.LLM_CLIENT_FACTORY = _factory_for(mocks)

    first = env.client.post("/api/scenes/scan1/demo/agent/annotate", json={"mode": "full"})
    assert first.status_code == 202
    run_id = first.get_json()["run_id"]
    try:
        second = env.client.post("/api/scenes/scan1/demo/agent/annotate", json={})
        assert second.status_code == 409
        body = second.get_json()
        assert body["error"] == "agent_run_active"
        assert body["active_run_id"] == run_id
        # pilot + chat + replay starts are gated by the same active-run rule
        assert (
            env.client.post(
                "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "go"}
            ).status_code
            == 409
        )
        assert (
            env.client.post(
                "/api/scenes/scan1/demo/agent/chat", json={"message": "hello there"}
            ).status_code
            == 409
        )
    finally:
        gate.set()
    _wait_run(env, run_id)
    third = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    )
    assert third.status_code == 202
    _wait_run(env, third.get_json()["run_id"])


def test_runs_listing(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    r1 = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    ).get_json()["run_id"]
    _wait_run(env, r1)
    listing = env.client.get("/api/scenes/scan1/demo/agent/runs").get_json()
    assert listing["active_run_id"] is None
    rows = {r["run_id"]: r for r in listing["runs"]}
    assert rows[r1]["status"] == "done"
    assert rows[r1]["mode"] == "annotate"
    assert rows[r1]["persisted"] is True


def test_events_survive_registry_restart(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)
    live = _events_of(env, "scan1", run_id)

    env.runlog.REGISTRY.reset()  # simulate broker restart
    stored = _events_of(env, "scan1", run_id, after=-1)
    assert stored["status"] == "done"
    assert [e["seq"] for e in stored["events"]] == [e["seq"] for e in live["events"]]
    assert stored["next"] == live["next"]
    # cursor still honoured from the stored log; pass-back-verbatim yields nothing new
    tail = _events_of(env, "scan1", run_id, after=stored["next"] - 1)
    assert len(tail["events"]) == 1
    assert _events_of(env, "scan1", run_id, after=stored["next"])["events"] == []


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_run_byte_identical_with_flag(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    src_id = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    ).get_json()["run_id"]
    _wait_run(env, src_id)
    src_events = _events_of(env, "scan1", src_id)["events"]
    src_doc = json.loads(
        env.store.get_derived_artifact(
            "user-a", "scan1", f"derived/demo/agent_runs/{src_id}/events.json"
        ).decode("utf-8")
    )

    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate",
        json={"replay_of": src_id, "speed": 200},
    )
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["replay"] is True
    replay_id = body["run_id"]
    assert replay_id != src_id
    _wait_run(env, replay_id, timeout=30.0)

    replay_events = _events_of(env, "scan1", replay_id)["events"]
    # REPLAY invariant: first event is run_meta with the structural flag
    meta = replay_events[0]
    assert meta["type"] == "run_meta"
    assert meta["payload"]["replay"] is True
    assert meta["payload"]["source_run_id"] == src_id
    assert meta["payload"]["recorded_at"] == src_doc["created_at"]
    # payloads byte-identical to the recording (minus the replaced run_meta)
    src_rest = [e["payload"] for e in src_events if e["type"] != "run_meta"]
    replay_rest = [e["payload"] for e in replay_events if e["type"] != "run_meta"]
    assert json.dumps(replay_rest, sort_keys=True) == json.dumps(src_rest, sort_keys=True)
    # the replay run persists as its own artifact, marked replay
    replay_doc = json.loads(
        env.store.get_derived_artifact(
            "user-a", "scan1", f"derived/demo/agent_runs/{replay_id}/events.json"
        ).decode("utf-8")
    )
    assert replay_doc["replay"] is True
    assert replay_doc["source_run_id"] == src_id


def test_replay_query_param_view(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    src_id = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
    ).get_json()["run_id"]
    _wait_run(env, src_id)
    src_events = _events_of(env, "scan1", src_id)["events"]

    view = _events_of(env, "scan1", src_id, replay=1, speed=2)
    assert view["replay"] is True
    assert view["status"] == "done"
    events = view["events"]
    assert events[0]["type"] == "run_meta"
    assert events[0]["payload"]["replay"] is True
    assert events[0]["payload"]["source_run_id"] == src_id
    # delay_ms annotated and clamped to [120, 2500]/speed=2 → [60, 1250]
    for e in events[1:]:
        assert 60 <= e["delay_ms"] <= 1250
    src_rest = [e["payload"] for e in src_events if e["type"] != "run_meta"]
    view_rest = [e["payload"] for e in events if e["type"] != "run_meta"]
    assert json.dumps(view_rest, sort_keys=True) == json.dumps(src_rest, sort_keys=True)
    # cursor works on the replay stream too (next = last seq, pass-back-verbatim)
    assert view["next"] == len(events) - 1
    tail = _events_of(env, "scan1", src_id, replay=1, after=view["next"] - 1)
    assert len(tail["events"]) == 1
    assert _events_of(env, "scan1", src_id, replay=1, after=view["next"])["events"] == []


def test_replay_of_unknown_run_404(demo_app):
    env = demo_app
    _save_scene(env.store)
    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"replay_of": "nope"}
    )
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "unknown_run"


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def test_chat_grounded_qa_path(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    qa_calls = []

    def fake_qa(user_id, scan_id, record, question, history):
        qa_calls.append((user_id, scan_id, question, list(history)))
        return {
            "answer": "The desk sits by the window.",
            "model": "mock-qa",
            "degraded": False,
            "focus": {"name": "desk"},
            "evidence": [],
        }

    env.stub._answer_scene_question = fake_qa

    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/chat", json={"message": "what do you notice about this office?"}
    )
    assert resp.status_code == 202
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    assert qa_calls and qa_calls[0][2] == "what do you notice about this office?"
    thoughts = [e["payload"] for e in events if e["type"] == "agent_thought"]
    assert any(t.get("author") == "user" for t in thoughts)  # echoed user turn
    answers = [t for t in thoughts if t.get("type") == "chat_response"]
    assert answers and answers[0]["content"] == "The desk sits by the window."
    ui = [e["payload"] for e in events if e["type"] == "agent_ui_command"]
    assert any(u["name"] == "focus_object" and u["args"]["object_uid"] == "det:0" for u in ui)
    assert events[-1]["type"] == "run_done"


def test_chat_tool_intent_measures_with_code_numbers(demo_app):
    env = demo_app
    _save_scene(env.store)
    pilot = ScriptedLLM(
        [
            {
                "thought": "Measuring chair to desk.",
                "action": {
                    "tool": "measure_distance",
                    "args": {"a": {"object": "chair"}, "b": {"object": "desk"}},
                },
                "final": None,
            },
            {"thought": None, "action": None, "final": "The measurement is shown above."},
        ]
    )
    env.agent.LLM_CLIENT_FACTORY = _factory_for({"pilot": pilot})

    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/chat",
        json={"message": "How far is the chair from the desk?"},
    )
    assert resp.status_code == 202
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    tool_events = [e["payload"] for e in events if e["type"] == "agent_tool_event"]
    ok = [t for t in tool_events if t["status"] == "succeeded"]
    assert ok and ok[0]["tool"] == "measure_distance"
    assert ok[0]["result"]["value"] == 5.0  # |(3,4,0)| — computed, not model-invented
    assert ok[0]["result"]["units"] == "relative"
    ui = [e["payload"] for e in events if e["type"] == "agent_ui_command"]
    sm = [u for u in ui if u["name"] == "show_measurement"]
    assert sm and sm[0]["args"]["value"] == 5.0 and sm[0]["args"]["units"] == "relative"
    finals = [
        e["payload"]["content"]
        for e in events
        if e["type"] == "agent_thought" and e["payload"].get("type") == "chat_response"
    ]
    assert finals[-1] == "The measurement is shown above."


def test_chat_requires_message(demo_app):
    env = demo_app
    _save_scene(env.store)
    assert env.client.post("/api/scenes/scan1/demo/agent/chat", json={}).status_code == 400


def test_chat_continues_finished_run_history(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    env.stub._answer_scene_question = lambda *a: {
        "answer": "First answer.",
        "model": "m",
        "degraded": False,
        "focus": {},
        "evidence": [],
    }
    r1 = env.client.post(
        "/api/scenes/scan1/demo/agent/chat", json={"message": "hello office"}
    ).get_json()["run_id"]
    _wait_run(env, r1)

    seen_history = {}

    def qa2(user_id, scan_id, record, question, history):
        seen_history["history"] = list(history)
        return {"answer": "Second answer.", "model": "m", "degraded": False, "focus": {}, "evidence": []}

    env.stub._answer_scene_question = qa2
    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/chat",
        json={"message": "and the ceiling?", "run_id": r1},
    )
    assert resp.status_code == 202
    assert resp.get_json()["continues_run"] == r1
    _wait_run(env, resp.get_json()["run_id"])
    roles = [(h["role"], h["content"]) for h in seen_history["history"]]
    assert ("user", "hello office") in roles
    assert ("assistant", "First answer.") in roles


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def test_pilot_loop_plan_path_failure_degrades_gracefully(demo_app):
    """Pre-integration this asserted the 'W4 not merged' stub error. With
    routes_nav merged the seam is LIVE: on this synthetic scene (no stored
    point cloud) the real route answers no_geometry — the pilot must surface
    that honest tool failure, invent no path, and still finish the run."""
    env = demo_app
    _save_scene(env.store)
    pilot = ScriptedLLM(
        [
            {
                "thought": "Checking the inventory first.",
                "action": {"tool": "list_scene_objects", "args": {}},
                "final": None,
            },
            {
                "thought": "Planning a path to the desk.",
                "action": {"tool": "plan_path", "args": {"goal_object": "desk"}},
                "final": None,
            },
            {
                "thought": None,
                "action": None,
                "final": "Path planning is not available yet; the inventory is listed.",
            },
        ]
    )
    env.agent.LLM_CLIENT_FACTORY = _factory_for({"pilot": pilot})

    resp = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "drive to the desk"}
    )
    assert resp.status_code == 202
    run_id = resp.get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    tool_events = [e["payload"] for e in events if e["type"] == "agent_tool_event"]
    listed = [t for t in tool_events if t["tool"] == "list_scene_objects" and t["status"] == "succeeded"]
    assert listed and listed[0]["result"]["closed_world"] is True
    assert listed[0]["result"]["count"] == 4  # dismissed box excluded
    failed = [t for t in tool_events if t["tool"] == "plan_path" and t["status"] == "failed"]
    assert failed and "no_geometry" in failed[0]["error"]  # live routes_nav error, not a stub
    ui_names = {e["payload"]["name"] for e in events if e["type"] == "agent_ui_command"}
    assert "show_path" not in ui_names  # no path invented from a failed plan
    assert events[-1]["type"] == "run_done"
    assert events[-1]["payload"]["status"] == "done"


class _FakeFormatError(RuntimeError):
    """Stand-in for ``openrouter_client.LLMFormatError``.

    The real class cannot be imported here — it lives beside the provider SDK, which
    this GPU-free suite deliberately does without — and ``persisted_agent`` matches it
    by name for exactly that reason. ``test_chat_json_format_error_carries_completion``
    covers the real class where the SDK is installed.
    """

    __name__ = "LLMFormatError"

    def __init__(self, message, content):
        super().__init__(message)
        self.content = content


_FakeFormatError.__name__ = "LLMFormatError"


def _responses_of(env, scan, run_id):
    return [
        e["payload"]["content"]
        for e in _events_of(env, scan, run_id)["events"]
        if e["type"] == "agent_thought" and e["payload"].get("type") == "chat_response"
    ]


def _errors_of(env, scan, run_id):
    return [
        e["payload"]["content"]
        for e in _events_of(env, scan, run_id)["events"]
        if e["type"] == "agent_thought" and e["payload"].get("type") == "error"
    ]


def test_prose_final_turn_is_kept_not_called_an_outage(demo_app):
    """A model that answers in prose instead of JSON broke FORMAT, not availability.
    The completion is already paid for, so its prose stands as the answer (the metric
    gate still polices numbers) and the run must not report an outage."""
    env = demo_app
    _save_scene(env.store)
    pilot = ScriptedLLM(
        [
            {
                "thought": "Checking the inventory first.",
                "action": {"tool": "list_scene_objects", "args": {}},
                "final": None,
            },
            _FakeFormatError(
                "No JSON object found in LLM response",
                "The inventory is listed above; nothing else was measurable.",
            ),
        ]
    )
    env.agent.LLM_CLIENT_FACTORY = _factory_for({"pilot": pilot})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "measure the room"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)

    answers = _responses_of(env, "scan1", run_id)
    assert answers and answers[-1] == "The inventory is listed above; nothing else was measurable."
    assert "LLM unavailable" not in answers[-1]
    assert "fact-only answer" not in answers[-1]
    # The real reason reaches the transcript, so this is diagnosable after the fact.
    errors = _errors_of(env, "scan1", run_id)
    assert any("broke JSON format" in e and "No JSON object found" in e for e in errors)


def test_format_error_fallback_names_the_tools_that_ran(demo_app):
    """When the prose carries its own numbers the gate drops it (guardrail 2) — but the
    fallback must still point at the work the turn actually did, and must not blame an
    outage that did not happen."""
    env = demo_app
    _save_scene(env.store, anchored=True)
    pilot = ScriptedLLM(
        [
            {
                "thought": "Checking the inventory first.",
                "action": {"tool": "list_scene_objects", "args": {}},
                "final": None,
            },
            _FakeFormatError(
                "No JSON object found in LLM response",
                "The desk is 1.20 m across.",  # unit-bearing prose → dropped by the gate
            ),
        ]
    )
    env.agent.LLM_CLIENT_FACTORY = _factory_for({"pilot": pilot})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "measure the desk"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)

    answer = _responses_of(env, "scan1", run_id)[-1]
    assert "1.20 m" not in answer  # the gate still holds
    assert "list_scene_objects" in answer  # the computed work is not thrown away
    assert "model answered but broke format" in answer
    assert "LLM unavailable" not in answer
    assert "scale source: anchor:" in answer  # provenance rides along (honesty doctrine)


def test_real_outage_still_reports_an_outage(demo_app):
    """The counterpart guard: with no client at all, 'LLM unavailable' is the truth and
    must survive the wording split."""
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "measure the room"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)

    answer = _responses_of(env, "scan1", run_id)[-1]
    assert "LLM unavailable; fact-only answer" in answer
    assert "no metric anchor" in answer  # unanchored scene says so instead of a source
    assert any("LLM unavailable" in e for e in _errors_of(env, "scan1", run_id))


def test_pilot_requires_instruction(demo_app):
    env = demo_app
    _save_scene(env.store)
    assert env.client.post("/api/scenes/scan1/demo/agent/pilot", json={}).status_code == 400


def test_pilot_no_llm_falls_back_to_fact_answer(demo_app):
    env = demo_app
    _save_scene(env.store)
    env.agent.LLM_CLIENT_FACTORY = _factory_for({})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "measure the room"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    assert events[-1]["payload"]["status"] == "done"
    answers = [
        e["payload"]["content"]
        for e in events
        if e["type"] == "agent_thought" and e["payload"].get("type") == "chat_response"
    ]
    assert answers and "fact-only answer" in answers[-1]


# ---------------------------------------------------------------------------
# plan_path → routes_nav seam (W4 contract, verified against a scripted module)
# ---------------------------------------------------------------------------


def _install_fake_routes_nav(monkeypatch, impl):
    """Inject a fake ``server.oreos.routes_nav`` exposing W4's seam signature
    ``plan_path_for_scene(user_id, scan_id, body, *, debug=False, store=None)``."""
    mod = types.ModuleType("server.oreos.routes_nav")
    mod.plan_path_for_scene = impl
    monkeypatch.setitem(sys.modules, "server.oreos.routes_nav", mod)
    demo_pkg = sys.modules["server.oreos"]
    monkeypatch.setattr(demo_pkg, "routes_nav", mod, raising=False)


def _agent_with_run(env, scan):
    record = env.store.get_scene("user-a", scan)
    run = env.runlog.AgentRun("t-" + scan, "user-a", scan, "pilot")
    return env.agent.PersistedSceneAgent(env.store, "user-a", scan, record, run), run


def test_plan_path_seam_success_with_up_override(demo_app, monkeypatch):
    """No-trajectory scene (today's canonical): body carries params.up_override='-y',
    the result notes the manual override honestly, and show_path/drive_path UI come
    from the PLANNER payload."""
    env = demo_app
    _save_scene(env.store, scan="scan-nav", with_trajectory=False)
    seen = {}

    def fake_plan(user_id, scan_id, body, *, debug=False, store=None):
        seen.update({"user_id": user_id, "scan_id": scan_id, "body": body, "store": store})
        return (
            {
                "path_id": "p1",
                "waypoints_world": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                "poses": [{"c2w": [[1, 0, 0, 0]] * 4, "t": 0.0}],
                "units": "relative",
                "units_basis": "extent_fraction",
                "notes": ["floor from heuristic"],
                "provenance": "planned in scanned free space — planner visualization, "
                "not certified navigation",
            },
            200,
        )

    _install_fake_routes_nav(monkeypatch, fake_plan)
    agent, run = _agent_with_run(env, "scan-nav")
    ok, result = agent._execute_tool("plan_path", {"goal_object": "desk"}, "pilot")
    assert ok is True
    assert seen["body"]["goal"] == {"object_uid": "det:0"}
    assert seen["body"]["params"] == {"up_override": "-y"}
    assert seen["store"] is env.store
    assert any("manual up-axis override" in n for n in result["notes"])
    ui = [e["payload"] for e in run.snapshot_events() if e["type"] == "agent_ui_command"]
    show = [u for u in ui if u["name"] == "show_path"]
    drive = [u for u in ui if u["name"] == "drive_path"]
    assert show and show[0]["args"]["points"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert show[0]["args"]["path_id"] == "p1"
    assert drive and drive[0]["args"]["path_id"] == "p1"


def test_plan_path_seam_no_override_with_trajectory(demo_app, monkeypatch):
    env = demo_app
    _save_scene(env.store, scan="scan-nav2", with_trajectory=True)
    seen = {}

    def fake_plan(user_id, scan_id, body, *, debug=False, store=None):
        seen["body"] = body
        return {"path_id": "p2", "waypoints_world": [[0, 0, 0]], "poses": [],
                "units": "relative", "units_basis": "extent_fraction", "notes": [],
                "provenance": "planner visualization"}, 200

    _install_fake_routes_nav(monkeypatch, fake_plan)
    agent, _run = _agent_with_run(env, "scan-nav2")
    result = agent.registry.execute(
        "plan_path", {"goal_point": [1.0, 2.0, 3.0]}
    )
    assert seen["body"]["goal"] == {"point_world": [1.0, 2.0, 3.0]}
    assert "params" not in seen["body"]  # pose gravity takes over
    assert not any("up-axis" in n for n in result.get("notes", []))


def test_plan_path_seam_unreachable_carries_retry_hint(demo_app, monkeypatch):
    env = demo_app
    _save_scene(env.store, scan="scan-nav3", with_trajectory=False)

    def fake_plan(user_id, scan_id, body, *, debug=False, store=None):
        return {"error": "unreachable_goal",
                "nearest_reachable": {"point_world": [1.5, 0.0, 2.0]}}, 422

    _install_fake_routes_nav(monkeypatch, fake_plan)
    agent, run = _agent_with_run(env, "scan-nav3")
    ok, error = agent._execute_tool("plan_path", {"goal_object": "chair"}, "pilot")
    assert ok is False
    assert "unreachable" in error
    assert "[1.5, 0.0, 2.0]" in error and "retry" in error
    ui_names = {e["payload"]["name"] for e in run.snapshot_events() if e["type"] == "agent_ui_command"}
    assert "show_path" not in ui_names


def test_plan_path_seam_404_errors_surface(demo_app, monkeypatch):
    env = demo_app
    _save_scene(env.store, scan="scan-nav4")

    def fake_plan(user_id, scan_id, body, *, debug=False, store=None):
        return {"error": "no_geometry", "message": "scan has no stored cloud"}, 404

    _install_fake_routes_nav(monkeypatch, fake_plan)
    agent, _run = _agent_with_run(env, "scan-nav4")
    tr = env.agent._tool_registry_module()
    with pytest.raises(tr.ToolExecutionError) as exc:
        agent.registry.execute("plan_path", {"goal_point": [0.0, 0.0, 0.0]})
    assert "no_geometry" in str(exc.value)
    assert "HTTP 404" in str(exc.value)


# ---------------------------------------------------------------------------
# budget + roster + client wiring
# ---------------------------------------------------------------------------


def test_llm_budget_degrades_not_dies(demo_app, monkeypatch):
    env = demo_app
    _save_scene(env.store)
    monkeypatch.setenv("SCENE_AGENT_MAX_LLM_CALLS", "1")
    env.agent.LLM_CLIENT_FACTORY = _factory_for(_standard_annotate_mocks())
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"mode": "full"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    done = events[-1]["payload"]
    assert done["status"] == "done"  # degraded, not dead
    assert done["llm_calls_attempted"] == 1
    # description fell back to the persisted report
    thoughts = [e["payload"]["content"] for e in events if e["type"] == "agent_thought"]
    assert any("An office with a desk and chairs." in t for t in thoughts)


def test_role_models_env_overrides(monkeypatch):
    from server.oreos import persisted_agent as pa

    monkeypatch.delenv("SCENE_AGENT_PILOT_MODEL", raising=False)
    monkeypatch.delenv("SCENE_AGENT_ANNOTATOR_MODEL", raising=False)
    monkeypatch.delenv("SCENE_AGENT_NARRATOR_MODEL", raising=False)
    monkeypatch.delenv("SCENE_AGENT_NARRATOR_FALLBACKS", raising=False)
    roles = pa.role_models()
    assert roles["pilot"]["model"] == "anthropic/claude-sonnet-5"
    assert roles["annotator"]["model"] == "google/gemini-3-flash-preview"
    assert roles["annotator"]["fallbacks"] == [
        "openai/gpt-5.6-luna",
        "anthropic/claude-haiku-4.5",
    ]
    # The narrator is its own role now — it no longer inherits the pilot's model, so
    # changing the pilot must NOT silently change who writes the on-screen prose.
    assert roles["narrator"]["model"] == "anthropic/claude-opus-5"
    assert roles["narrator"]["fallbacks"] == [
        "anthropic/claude-sonnet-5",
        "google/gemini-3-flash-preview",
    ]
    monkeypatch.setenv("SCENE_AGENT_PILOT_MODEL", "anthropic/claude-opus-5")
    monkeypatch.setenv("SCENE_AGENT_PILOT_FALLBACKS", "a/b, c/d")
    roles = pa.role_models()
    assert roles["pilot"]["model"] == "anthropic/claude-opus-5"
    assert roles["pilot"]["fallbacks"] == ["a/b", "c/d"]
    assert roles["narrator"]["model"] == "anthropic/claude-opus-5"
    assert roles["narrator"]["fallbacks"] != ["a/b", "c/d"]
    monkeypatch.setenv("SCENE_AGENT_NARRATOR_MODEL", "x/y")
    monkeypatch.setenv("SCENE_AGENT_NARRATOR_FALLBACKS", "e/f")
    roles = pa.role_models()
    assert roles["narrator"]["model"] == "x/y"
    assert roles["narrator"]["fallbacks"] == ["e/f"]


def test_usage_tally_and_openrouter_wiring(monkeypatch):
    """DemoRouterClient asks OpenRouter for usage accounting and records REAL cost
    into the tally (no fabricated estimates)."""
    pytest.importorskip("openai")
    from server.oreos import persisted_agent as pa

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-wiring-test")
    tally = pa.UsageTally()
    client = pa._make_openrouter_client("anthropic/claude-sonnet-4.6", [], tally)
    assert client is not None

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20, cost=0.0012, model_extra=None
            ),
        )

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    parsed, response = client.chat_json(system_prompt="s", user_prompt="u")
    assert parsed == {"ok": True}
    assert captured["extra_body"] == {"usage": {"include": True}}
    assert captured["model"] == "anthropic/claude-sonnet-4.6"
    summary = tally.summary()
    assert summary["llm_calls"] == 1
    assert summary["prompt_tokens"] == 100
    assert summary["cost_usd"] == pytest.approx(0.0012)
    assert summary["cost_basis"] == "openrouter_usage"

    # unpriced usage stays honest: no cost claimed
    t2 = pa.UsageTally()
    t2.record("m", SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=None, model_extra=None))
    s2 = t2.summary()
    assert s2["cost_usd"] is None
    assert s2["cost_basis"] == "unreported"


def test_chat_json_format_error_carries_completion(monkeypatch):
    """A prose reply must raise the typed LLMFormatError carrying the raw completion —
    that is what lets a caller keep the answer instead of discarding a paid-for turn.
    It subclasses RuntimeError so existing `except RuntimeError` handlers still catch it,
    and its class name is what ``persisted_agent.is_format_error`` matches."""
    pytest.importorskip("openai")
    from server.llm.openrouter_client import LLMFormatError
    from server.oreos import persisted_agent as pa

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-wiring-test")
    client = pa._make_openrouter_client("anthropic/claude-sonnet-4.6", [], pa.UsageTally())

    def fake_create(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Sure — the desk is the widest object.")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, cost=0.0, model_extra=None
            ),
        )

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    with pytest.raises(LLMFormatError) as caught:
        client.chat_json(system_prompt="s", user_prompt="u")
    assert caught.value.content == "Sure — the desk is the widest object."
    assert isinstance(caught.value, RuntimeError)
    assert pa.is_format_error(caught.value)
    # An outage is a different thing and must not match.
    assert not pa.is_format_error(pa.LLMUnavailableError("no client"))


def test_tool_allowlist_env(demo_app, monkeypatch):
    env = demo_app
    _save_scene(env.store)
    monkeypatch.setenv("SCENE_AGENT_TOOLS", "list_scene_objects")
    pilot = ScriptedLLM(
        [
            {
                "thought": "Trying a disallowed tool.",
                "action": {
                    "tool": "measure_distance",
                    "args": {"a": {"object": "chair"}, "b": {"object": "desk"}},
                },
                "final": None,
            },
            {"thought": None, "action": None, "final": "Blocked, as configured."},
        ]
    )
    env.agent.LLM_CLIENT_FACTORY = _factory_for({"pilot": pilot})
    run_id = env.client.post(
        "/api/scenes/scan1/demo/agent/pilot", json={"instruction": "measure the distance"}
    ).get_json()["run_id"]
    _wait_run(env, run_id)
    events = _events_of(env, "scan1", run_id)["events"]
    failed = [
        e["payload"]
        for e in events
        if e["type"] == "agent_tool_event" and e["payload"]["status"] == "failed"
    ]
    assert failed and failed[0]["tool"] == "measure_distance"
    assert "Unknown tool" in failed[0]["error"]


# ---------------------------------------------------------------------------
# export narration tools (grounded, route-presence only)
# ---------------------------------------------------------------------------


def test_export_tools_narrate_gates(demo_app):
    env = demo_app
    _save_scene(env.store, scan="scan-t")  # trajectory, unanchored
    _save_scene(env.store, scan="scan-nt", with_trajectory=False)
    _save_scene(env.store, scan="scan-a", anchored=True)
    agent_mod = env.agent
    runlog_mod = env.runlog

    def make_agent(scan):
        record = env.store.get_scene("user-a", scan)
        run = runlog_mod.AgentRun("t" + scan, "user-a", scan, "chat")
        return agent_mod.PersistedSceneAgent(env.store, "user-a", scan, record, run)

    a = make_agent("scan-t")
    lr = a.registry.execute("export.lerobot", {})
    assert lr["available"] is True
    assert "manifest_probe" in lr and "not built yet" in lr["manifest_probe"]
    assert "LeRobot v2" in lr["copy"] and "no robot manipulation" in lr["copy"]
    isaac = a.registry.execute("export.isaac", {})
    assert isaac["available"] is False
    assert isaac["metric_gate"] == "blocked"
    assert "metric_scale_required" in isaac["reason"]
    assert isaac["anchor_cta"] == "Anchor to metres"

    b = make_agent("scan-nt")
    assert b.registry.execute("export.lerobot", {})["available"] is False

    c = make_agent("scan-a")
    isaac_ok = c.registry.execute("export.isaac", {})
    assert isaac_ok["metric_gate"] == "passed"
    assert isaac_ok["available"] is True
    assert isaac_ok["scale_source"] == "anchor:derived/anchor/t/cloud.ply"


# ---------------------------------------------------------------------------
# auth / scene guards on every route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/api/scenes/scan1/demo/agent/annotate", {}),
        ("GET", "/api/scenes/scan1/demo/agent/runs", None),
        ("GET", "/api/scenes/scan1/demo/agent/runs/r1/events", None),
        ("POST", "/api/scenes/scan1/demo/agent/chat", {"message": "hello"}),
        ("POST", "/api/scenes/scan1/demo/agent/pilot", {"instruction": "go"}),
        ("POST", "/api/scenes/scan1/measure", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]}),
    ],
)
def test_agent_routes_auth_and_scene_guards(demo_app, method, path, body):
    env = demo_app
    # unauthenticated → 401
    env.stub._auth_user_id = lambda: None
    resp = env.client.open(path, method=method, json=body)
    assert resp.status_code == 401
    # authenticated but unknown scene → 404
    env.stub._auth_user_id = lambda: "user-a"
    resp = env.client.open(path, method=method, json=body)
    assert resp.status_code == 404


def test_unknown_run_events_404(demo_app):
    env = demo_app
    _save_scene(env.store)
    resp = env.client.get("/api/scenes/scan1/demo/agent/runs/ghost/events")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "unknown_run"
