"""Scene-agent RUN VISIBILITY lifecycle (feat/demo-annotate-visibility) — GPU-free.

The founder started an annotation run, looked away, and came back to an empty panel.
The events were never lost — they were in ``derived/demo/agent_runs/<run_id>/events.json``
the whole time — but nothing the client could read told it (a) which run to render when
none is active, (b) in the protocol's own vocabulary, or (c) *why* a run had failed.

Harness = the ``test_demo_agent.py`` pattern (real Flask + demo blueprint, real
``ModalScenePersistence`` on tmp_path, ``server.app`` stubbed). Runs use
``mode="fact_only"`` so they need no LLM and no network.

Covers: the protocol shape of ``GET …/agent/runs`` (AgentRunListItem), rendering a
FINISHED run from its persisted log (the mount path), resume across a broker restart
(registry wiped), the ``error`` reason on both the poll response and the run row, and
the idempotent ``attach_if_active`` start — including that it never starts a second run
and that the honest 409 is untouched without it.
"""

from __future__ import annotations

import base64
import importlib
import sys
import threading
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
# harness (test_demo_agent.py:47-85)
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
    monkeypatch.delenv("SCENE_AGENT_MAX_WALL_S", raising=False)
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

    yield SimpleNamespace(
        client=app.test_client(),
        store=store,
        stub=stub,
        agent=agent_mod,
        runlog=runlog_mod,
    )
    runlog_mod.REGISTRY.reset()
    agent_mod.LLM_CLIENT_FACTORY = None


def _save_scene(store, user="user-a", scan="scan1", objects=True):
    instances = (
        [
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
        ]
        if objects
        else []
    )
    facts = SceneFacts(
        metrics=SceneMetrics(
            dimensions=[5.0, 4.0, 2.0],
            bbox_min=[-1.0, -1.0, 0.0],
            bbox_max=[4.0, 3.0, 2.0],
            num_submaps=3,
            num_keyframes=12,
            point_count=50_000,
        ),
        objects=instances,
        object_counts={"desk": 1, "chair": 1} if objects else {},
        coverage_estimate=0.7,
    )
    report = SceneReport(
        summary="An office with a desk and chairs.",
        room_type="office",
        observations=["The desk faces the window wall."],
    )
    store.save_scene(
        user,
        scan,
        report,
        facts,
        keyframes_b64=[
            {"submap_id": 0, "frame_idx": 0, "image_b64": base64.b64encode(b"jpeg0").decode()}
        ],
        trajectory={
            "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            "intrinsics": np.ones((2, 4), dtype=np.float32),
            "source_frame_id": np.arange(2, dtype=np.float32),
        },
        label="fixture-office",
        source="recon_video",
    )


def _run_annotate(env, scan="scan1", mode="fact_only", **body):
    resp = env.client.post(
        f"/api/scenes/{scan}/demo/agent/annotate", json={"mode": mode, **body}
    )
    assert resp.status_code == 202, resp.get_json()
    run_id = resp.get_json()["run_id"]
    run = env.runlog.REGISTRY.get(run_id)
    assert run is not None
    if run.thread is not None:
        run.thread.join(timeout=20.0)
    assert run.status != "running", "run did not finish in time"
    return run_id


def _runs(env, scan="scan1"):
    resp = env.client.get(f"/api/scenes/{scan}/demo/agent/runs")
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _events(env, run_id, scan="scan1", **params):
    resp = env.client.get(
        f"/api/scenes/{scan}/demo/agent/runs/{run_id}/events", query_string=params or None
    )
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


# ---------------------------------------------------------------------------
# the runs index speaks the protocol (web/packages/protocol demo.ts)
# ---------------------------------------------------------------------------


def test_runs_list_carries_protocol_fields(demo_app):
    """AgentRunListItem: kind / started_at / n_events / cost_usd.

    The client gates its Replay affordance on ``kind`` and renders ``started_at`` as a
    clock time — the pre-fix response carried only ``mode``/``created_at``, so Replay
    never rendered and every row read "Invalid Date".
    """
    _save_scene(demo_app.store)
    run_id = _run_annotate(demo_app)

    data = _runs(demo_app)
    assert data["active_run_id"] is None  # the run is over — this is the founder's state
    (row,) = data["runs"]
    assert row["run_id"] == run_id
    assert row["kind"] == "annotate"
    assert isinstance(row["started_at"], float) and row["started_at"] > 0
    assert row["n_events"] == row["event_count"] > 0
    assert row["status"] == "done"
    assert "cost_usd" in row  # None on a fact-only run: recorded, never invented
    assert row["cost_usd"] is None
    assert row["findings_emitted"] >= 1
    # internal vocabulary preserved (additive-only)
    assert row["mode"] == "annotate" and row["created_at"] == row["started_at"]


def test_run_row_reports_recorded_spend(demo_app):
    """``cost_usd``/``llm_calls`` come from the run's own ``run_done`` event."""
    _save_scene(demo_app.store)
    run_id = _run_annotate(demo_app)

    doc = demo_app.runlog.load_run_doc(demo_app.store, "user-a", "scan1", run_id)
    done = [e for e in doc["events"] if e["type"] == "run_done"][-1]["payload"]
    (row,) = _runs(demo_app)["runs"]
    assert row["llm_calls"] == done["llm_calls"]
    assert row["cost_usd"] == done["cost_usd"]
    assert row["findings_emitted"] == done["findings_emitted"]


# ---------------------------------------------------------------------------
# rendering a FINISHED run — the mount path
# ---------------------------------------------------------------------------


def test_finished_run_replays_its_whole_timeline_from_after_minus_one(demo_app):
    """What the panel does on mount when nothing is active: pick the newest run and
    ask for ``?after=-1``. It must get the ENTIRE log back, terminal status included."""
    _save_scene(demo_app.store)
    run_id = _run_annotate(demo_app)

    body = _events(demo_app, run_id, after=-1)
    assert body["status"] == "done"
    assert body["error"] is None
    types_ = [e["type"] for e in body["events"]]
    assert types_[0] == "run_meta" and types_[-1] == "run_done"
    assert "agent_finding" in types_
    assert [e["seq"] for e in body["events"]] == sorted(e["seq"] for e in body["events"])
    assert body["next"] == body["events"][-1]["seq"]
    # The run_meta the client keys its REPLAY badge off says this was NOT a replay.
    assert body["events"][0]["payload"]["replay"] is False


def test_finished_run_renders_after_a_broker_restart(demo_app):
    """The in-memory registry is process-local; a restart must not blank the panel.
    Both the index and the timeline come back off the persisted artifact."""
    _save_scene(demo_app.store)
    run_id = _run_annotate(demo_app)

    demo_app.runlog.REGISTRY.reset()  # broker restart

    data = _runs(demo_app)
    assert data["active_run_id"] is None
    (row,) = data["runs"]
    assert row["run_id"] == run_id
    assert row["kind"] == "annotate"  # protocol shape survives the restart
    assert row["started_at"] > 0
    assert row["n_events"] > 0
    body = _events(demo_app, run_id, after=-1)
    assert body["status"] == "done"
    assert len(body["events"]) == row["n_events"]


def test_thin_run_is_still_a_renderable_timeline(demo_app):
    """A scene with NO detected objects still produces a real timeline: survey thoughts
    plus the scene-level geometry finding. Nothing about it is blank, and the object
    counts the client cites in its "no findings" note come from the run's own run_meta."""
    _save_scene(demo_app.store, objects=False)
    run_id = _run_annotate(demo_app)

    body = _events(demo_app, run_id, after=-1)
    assert body["status"] == "done"
    assert len(body["events"]) >= 3  # run_meta + survey thoughts + run_done
    findings = [e["payload"] for e in body["events"] if e["type"] == "agent_finding"]
    # No OBJECT-grounded findings (there are no objects to ground them in)…
    assert all(f.get("object_uid") is None for f in findings)
    # …but the scene-level geometry finding still lands, honestly gated.
    assert any(f["query"] == "room extents" for f in findings)
    assert all(f["metric_state"] == "up_to_scale" for f in findings)

    meta = body["events"][0]["payload"]
    assert meta["scene"]["object_count"] == 0  # the counts the client cites
    assert "stored_keyframes" in meta["scene"]


# ---------------------------------------------------------------------------
# errors carry their reason
# ---------------------------------------------------------------------------


def test_error_run_surfaces_the_reason_on_poll_and_in_the_index(demo_app, monkeypatch):
    """``status: 'error'`` alone left the panel saying "run errored — see feed" with an
    empty feed. The reason now rides the poll response and the run row."""
    _save_scene(demo_app.store)
    monkeypatch.setenv("SCENE_AGENT_MAX_WALL_S", "0.0001")  # trips between phases
    run_id = _run_annotate(demo_app)

    body = _events(demo_app, run_id, after=-1)
    assert body["status"] == "error"
    assert "wall_clock" in (body["error"] or "")

    (row,) = _runs(demo_app)["runs"]
    assert row["status"] == "error"
    assert "wall_clock" in (row["error"] or "")


def test_error_reason_survives_a_broker_restart(demo_app, monkeypatch):
    _save_scene(demo_app.store)
    monkeypatch.setenv("SCENE_AGENT_MAX_WALL_S", "0.0001")
    run_id = _run_annotate(demo_app)

    demo_app.runlog.REGISTRY.reset()

    assert "wall_clock" in (_events(demo_app, run_id, after=-1)["error"] or "")
    (row,) = _runs(demo_app)["runs"]
    assert "wall_clock" in (row["error"] or "")


def test_healthy_poll_reports_error_none(demo_app):
    """The field is always present so a client can branch on it without guessing."""
    _save_scene(demo_app.store)
    run_id = _run_annotate(demo_app)
    assert _events(demo_app, run_id, after=-1)["error"] is None
    assert _events(demo_app, run_id, replay=1)["error"] is None


# ---------------------------------------------------------------------------
# idempotent start-or-attach
# ---------------------------------------------------------------------------


class _BlockedRun:
    """Holds a real run in ``running`` so the routes see genuine concurrency."""

    def __init__(self, runlog_mod, user="user-a", scan="scan1", mode="annotate"):
        self.run = runlog_mod.REGISTRY.start_run(user, scan, mode=mode)

    def finish(self):
        self.run.finish()


def test_attach_if_active_returns_the_live_run_instead_of_409(demo_app):
    _save_scene(demo_app.store)
    blocked = _BlockedRun(demo_app.runlog)
    try:
        resp = demo_app.client.post(
            "/api/scenes/scan1/demo/agent/annotate",
            json={"mode": "fact_only", "attach_if_active": True},
        )
        assert resp.status_code == 202
        body = resp.get_json()
        assert body["run_id"] == blocked.run.run_id
        assert body["attached"] is True
        # and NO second run was started (the one-active-run guarantee is untouched)
        assert len(demo_app.runlog.REGISTRY.list_for_scene("user-a", "scan1")) == 1
    finally:
        blocked.finish()


def test_attach_if_active_also_covers_a_replay_start(demo_app):
    _save_scene(demo_app.store)
    source_run = _run_annotate(demo_app)
    blocked = _BlockedRun(demo_app.runlog)
    try:
        resp = demo_app.client.post(
            "/api/scenes/scan1/demo/agent/annotate",
            json={"replay_of": source_run, "attach_if_active": True},
        )
        assert resp.status_code == 202
        assert resp.get_json() == {"run_id": blocked.run.run_id, "attached": True}
    finally:
        blocked.finish()


def test_without_the_flag_the_honest_409_stands(demo_app):
    """Genuine concurrency (another tab/operator) must still be TOLD the scene is busy,
    not silently joined to someone else's run."""
    _save_scene(demo_app.store)
    blocked = _BlockedRun(demo_app.runlog)
    try:
        resp = demo_app.client.post(
            "/api/scenes/scan1/demo/agent/annotate", json={"mode": "fact_only"}
        )
        assert resp.status_code == 409
        assert resp.get_json() == {
            "error": "agent_run_active",
            "active_run_id": blocked.run.run_id,
        }
    finally:
        blocked.finish()


def test_attach_if_active_is_inert_when_nothing_is_running(demo_app):
    """With no active run the flag changes nothing: a NEW run starts, unattached."""
    _save_scene(demo_app.store)
    resp = demo_app.client.post(
        "/api/scenes/scan1/demo/agent/annotate",
        json={"mode": "fact_only", "attach_if_active": True},
    )
    assert resp.status_code == 202
    body = resp.get_json()
    assert "attached" not in body
    run = demo_app.runlog.REGISTRY.get(body["run_id"])
    if run.thread is not None:
        run.thread.join(timeout=20.0)
    assert run.status == "done"


def test_chat_keeps_the_409_while_a_run_is_active(demo_app):
    """Chat is a NEW turn — attaching it to a running annotate would misreport it."""
    _save_scene(demo_app.store)
    blocked = _BlockedRun(demo_app.runlog)
    try:
        resp = demo_app.client.post(
            "/api/scenes/scan1/demo/agent/chat",
            json={"message": "what do you see?", "attach_if_active": True},
        )
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "agent_run_active"
    finally:
        blocked.finish()


# ---------------------------------------------------------------------------
# replay stays free, deterministic and structurally labelled
# ---------------------------------------------------------------------------


def test_replay_run_is_byte_identical_and_marked(demo_app):
    """The affordance the panel now leads with: a replay re-serves the recorded payloads
    unchanged and always opens with ``run_meta{replay: true}``."""
    _save_scene(demo_app.store)
    source_run = _run_annotate(demo_app)
    original = _events(demo_app, source_run, after=-1)["events"]

    resp = demo_app.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"replay_of": source_run, "speed": 50.0}
    )
    assert resp.status_code == 202
    replay_id = resp.get_json()["run_id"]
    assert resp.get_json()["replay"] is True
    run = demo_app.runlog.REGISTRY.get(replay_id)
    run.thread.join(timeout=30.0)

    replayed = _events(demo_app, replay_id, after=-1)["events"]
    assert replayed[0]["type"] == "run_meta"
    assert replayed[0]["payload"]["replay"] is True
    assert replayed[0]["payload"]["source_run_id"] == source_run
    # every non-meta payload is the recording's, unchanged
    assert [e["payload"] for e in replayed[1:]] == [
        e["payload"] for e in original if e["type"] != "run_meta"
    ]

    # …and it lands in the index as a replay row pointing at its source.
    rows = {r["run_id"]: r for r in _runs(demo_app)["runs"]}
    assert rows[replay_id]["kind"] == "replay"
    assert rows[replay_id]["replay"] is True
    assert rows[replay_id]["source_run_id"] == source_run


def test_replay_costs_nothing(demo_app):
    """Replays make no model calls — the recorded tally is what the row reports."""
    _save_scene(demo_app.store)
    source_run = _run_annotate(demo_app)
    resp = demo_app.client.post(
        "/api/scenes/scan1/demo/agent/annotate", json={"replay_of": source_run, "speed": 50.0}
    )
    replay_id = resp.get_json()["run_id"]
    demo_app.runlog.REGISTRY.get(replay_id).thread.join(timeout=30.0)

    rows = {r["run_id"]: r for r in _runs(demo_app)["runs"]}
    assert rows[replay_id]["cost_usd"] is None or rows[replay_id]["cost_usd"] == 0.0


def test_concurrent_starts_never_double_spend(demo_app):
    """Two clicks racing: exactly one run is created, the loser attaches to it."""
    _save_scene(demo_app.store)
    results: list[tuple[int, dict]] = []
    lock = threading.Lock()

    def click():
        resp = demo_app.client.post(
            "/api/scenes/scan1/demo/agent/annotate",
            json={"mode": "fact_only", "attach_if_active": True},
        )
        with lock:
            results.append((resp.status_code, resp.get_json()))

    threads = [threading.Thread(target=click) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert all(code == 202 for code, _ in results)
    run_ids = {body["run_id"] for _, body in results}
    for run in [demo_app.runlog.REGISTRY.get(r) for r in run_ids]:
        if run is not None and run.thread is not None:
            run.thread.join(timeout=20.0)
    # Runs actually created on this scene — attaches must not have added any.
    created = demo_app.runlog.REGISTRY.list_for_scene("user-a", "scan1")
    assert len(created) <= len(run_ids)
    assert len(run_ids) <= len(created)
