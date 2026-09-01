"""Demo blueprint tests (W0, docs/demo-2026-07) — GPU-free.

Two harnesses:

1. REAL Flask + ``test_client`` for the blueprint itself. ``server.oreos``
   imports only flask/stdlib, and its handlers reach ``server.app`` lazily
   (the reconstruct_pilot.py:59 pattern), so a stub ``server.app`` module —
   auth helper + a real ``ModalScenePersistence`` on tmp_path — is injected via
   ``sys.modules`` and nothing GPU-flavored is ever imported.

2. ``conftest.load_app_module`` (FakeFlask harness, the
   test_product_workflow_routes pattern) for the ``server.app`` surface: the
   register_blueprint one-liner and ``get_scene_route``'s ``source`` field.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

flask = pytest.importorskip("flask")

from conftest import load_app_module

from server.scene_report.schemas import SceneFacts, SceneReport
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# harness 1: real Flask app + stubbed server.app
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    """(Re)import ``server.oreos`` under the CURRENTLY active flask module. A
    sibling test may have cached it under conftest's fake flask (or vice
    versa), so the cached package is dropped first."""
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def demo_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(
        _scene_persistence=store,
        _auth_user_id=lambda: "user-a",
    )
    # The handlers' lazy `from server import app as app_module` must resolve to
    # the stub — patch both the submodule entry and the package attribute.
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    demo_jobs = sys.modules["server.oreos.jobs"]
    demo_jobs.configure_jobs_store({})  # deterministic in-process store
    yield app.test_client(), store, stub, demo_jobs
    demo_jobs.configure_jobs_store(None)  # back to the lazy modal binding


def _save_demo_scene(store, user="user-a", scan="scan1", **kwargs):
    store.save_scene(
        user,
        scan,
        SceneReport(summary="s", room_type="office"),
        SceneFacts(),
        **kwargs,
    )


# -- registration -----------------------------------------------------------


def test_blueprint_registers_every_demo_rule(demo_app):
    client, _store, _stub, _jobs = demo_app
    rules = {str(r) for r in client.application.url_map.iter_rules()}
    for expected in [
        "/api/scenes/<scan_id>/demo/manifest",
        "/api/scenes/<scan_id>/demo/doc",
        "/api/workspace/jobs/<job_id>",
        "/api/workspace/ingest/video",
        "/api/workspace/ingest/splat",
        "/api/scenes/<scan_id>/demo/agent/annotate",
        "/api/scenes/<scan_id>/demo/agent/runs",
        "/api/scenes/<scan_id>/demo/agent/runs/<run_id>/events",
        "/api/scenes/<scan_id>/demo/agent/chat",
        "/api/scenes/<scan_id>/demo/agent/pilot",
        "/api/scenes/<scan_id>/segment",
        "/api/scenes/<scan_id>/objects/<uid>/complete",
        "/api/scenes/<scan_id>/objects/<uid>/variants",
    ]:
        assert expected in rules, f"missing rule {expected}"


def test_app_py_registers_the_blueprint(monkeypatch):
    """server.app's module-level one-liner registers oreos_bp (FakeFlask records it)."""
    app_module = load_app_module(monkeypatch)
    assert len(app_module.app.registered_blueprints) == 1


# -- manifest read ----------------------------------------------------------


def test_manifest_empty_default(demo_app):
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)
    resp = client.get("/api/scenes/scan1/demo/manifest")
    assert resp.status_code == 200
    manifest = resp.get_json()
    assert manifest == {
        "version": 1,
        "updated_at": None,
        "annotations_key": None,
        "agent_runs": [],
        "edits": [],
        "paths": [],
        "variations": [],
        "measurements": [],
        "docs": [],
    }


def test_manifest_unknown_scene_404(demo_app):
    client, _store, _stub, _jobs = demo_app
    resp = client.get("/api/scenes/nope/demo/manifest")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "not_found"}


def test_manifest_unauthed_401(demo_app):
    client, store, stub, _jobs = demo_app
    _save_demo_scene(store)
    stub._auth_user_id = lambda: None
    resp = client.get("/api/scenes/scan1/demo/manifest")
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "invalid_token"}


def test_manifest_not_leaked_across_users(demo_app):
    """user-b's identical scan_id must not read user-a's manifest/scene."""
    client, store, stub, _jobs = demo_app
    _save_demo_scene(store, user="user-a", scan="scan1")
    stub._auth_user_id = lambda: "user-b"
    resp = client.get("/api/scenes/scan1/demo/manifest")
    assert resp.status_code == 404


# -- doc writer -------------------------------------------------------------


def test_put_doc_writes_and_indexes_manifest(demo_app):
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)

    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "annotations/v2.json", "json": {"version": 2, "labels": []}},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["key"] == "derived/demo/annotations/v2.json"

    raw = store.get_derived_artifact("user-a", "scan1", "derived/demo/annotations/v2.json")
    assert json.loads(raw.decode("utf-8")) == {"version": 2, "labels": []}

    # a second doc lands in its typed collection
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "paths/p1.json", "json": {"waypoints": []}},
    )
    assert resp.status_code == 200

    manifest = client.get("/api/scenes/scan1/demo/manifest").get_json()
    assert manifest["annotations_key"] == "derived/demo/annotations/v2.json"
    assert manifest["paths"] == ["derived/demo/paths/p1.json"]
    assert set(manifest["docs"]) == {
        "derived/demo/annotations/v2.json",
        "derived/demo/paths/p1.json",
    }
    assert manifest["updated_at"]  # stamped

    # idempotent re-PUT does not duplicate index entries
    client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "paths/p1.json", "json": {"waypoints": [1]}},
    )
    manifest = client.get("/api/scenes/scan1/demo/manifest").get_json()
    assert manifest["paths"] == ["derived/demo/paths/p1.json"]


@pytest.mark.parametrize(
    "bad_suffix",
    [
        "../evil.json",            # escape up out of demo/
        "a/../../evil.json",       # nested escape
        "/etc/passwd.json",        # absolute
        "a//b.json",               # empty segment
        "a\\b.json",               # backslash
        "",                        # empty
        "..",                      # bare traversal
        "manifest.json",           # reserved (writer-maintained)
        "notes.txt",               # not a .json doc
        "a/./b.json",              # dot segment
        ".json",                   # extension only
        "sp ace.json",             # charset violation (space)
        "x" * 250 + ".json",       # oversized
    ],
)
def test_put_doc_guard_rejects(demo_app, bad_suffix):
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": bad_suffix, "json": {"x": 1}},
    )
    assert resp.status_code == 422, bad_suffix
    assert resp.get_json()["error"] == "invalid_key"
    # nothing escaped the demo namespace
    assert store.get_derived_artifact("user-a", "scan1", "derived/evil.json") is None


def test_put_doc_non_string_key_suffix_rejected(demo_app):
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": ["a.json"], "json": {}},
    )
    assert resp.status_code == 422


def test_put_doc_bad_body_400(demo_app):
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)
    assert (
        client.put("/api/scenes/scan1/demo/doc", json={"key_suffix": "a.json"}).status_code
        == 400
    )  # no json member
    assert (
        client.put(
            "/api/scenes/scan1/demo/doc", data="[]", content_type="application/json"
        ).status_code
        == 400
    )  # non-dict body


def test_put_doc_agent_runs_key_indexes_rich_run_ref(demo_app):
    """agent_runs/* docs index as protocol DemoManifestRunRef dicts — the same
    shape runlog's flusher writes — not plain key strings (integration
    reconciliation: one manifest.agent_runs shape from both writers)."""
    client, store, _stub, _jobs = demo_app
    _save_demo_scene(store)

    events_doc = {
        "version": 1,
        "run_id": "r-abc",
        "mode": "annotate",
        "created_at": 1753.5,
        "replay": False,
        "events": [],
    }
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "agent_runs/r-abc/events.json", "json": events_doc},
    )
    assert resp.status_code == 200
    manifest = resp.get_json()["manifest"]
    assert manifest["agent_runs"] == [
        {
            "run_id": "r-abc",
            "kind": "annotate",
            "created_at": 1753.5,
            "events_key": "derived/demo/agent_runs/r-abc/events.json",
            "replay": False,
        }
    ]
    # the generic docs index still records the key string
    assert "derived/demo/agent_runs/r-abc/events.json" in manifest["docs"]

    # idempotent re-PUT: deduped by run_id, no second entry
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "agent_runs/r-abc/events.json", "json": events_doc},
    )
    assert len(resp.get_json()["manifest"]["agent_runs"]) == 1

    # a doc without W2's fields still yields a well-formed minimal ref
    resp = client.put(
        "/api/scenes/scan1/demo/doc",
        json={"key_suffix": "agent_runs/r-bare/events.json", "json": {"events": []}},
    )
    refs = resp.get_json()["manifest"]["agent_runs"]
    assert len(refs) == 2
    bare = refs[1]
    assert bare["run_id"] == "r-bare"
    assert bare["events_key"] == "derived/demo/agent_runs/r-bare/events.json"
    assert bare["created_at"] > 0  # defaulted to now

    # W2's reader consumes the rich entries directly
    runlog = sys.modules["server.oreos.runlog"]
    listed = runlog.list_persisted_runs(store, "user-a", "scan1")
    assert [r["run_id"] for r in listed] == ["r-abc", "r-bare"]


def test_put_doc_unknown_scene_404(demo_app):
    client, _store, _stub, _jobs = demo_app
    resp = client.put(
        "/api/scenes/nope/demo/doc", json={"key_suffix": "a.json", "json": {}}
    )
    assert resp.status_code == 404


# -- jobs route -------------------------------------------------------------


def test_jobs_unknown_id_404_envelope(demo_app):
    client, _store, _stub, _jobs = demo_app
    resp = client.get("/api/workspace/jobs/nope")
    assert resp.status_code == 404
    assert resp.is_json
    assert resp.get_json() == {"error": "job_not_found", "job_id": "nope"}


def test_jobs_returns_own_record_hides_others(demo_app):
    client, _store, _stub, jobs = demo_app
    mine = jobs.job_status_record("j1", "user-a", status="running", stage="recon", scan_id="s1")
    theirs = jobs.job_status_record("j2", "user-b", status="done", stage="done", scan_id="s2")
    jobs.configure_jobs_store({"j1": mine, "j2": theirs})

    resp = client.get("/api/workspace/jobs/j1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["job_id"] == "j1"
    assert body["status"] == "running"
    assert body["stage"] == "recon"
    assert body["scan_id"] == "s1"
    assert body["updated_at"] > 0

    # another user's job → identical 404 (existence not revealed)
    resp = client.get("/api/workspace/jobs/j2")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "job_not_found", "job_id": "j2"}


def test_jobs_failed_status_translates_to_protocol_error(demo_app):
    """Ingest writers store ``failed``; the route answers protocol ``error``
    (@reality/protocol DemoJobStatus) while the stored record keeps ``failed``
    for back-compat with records already in the modal.Dict."""
    client, _store, _stub, jobs = demo_app
    stored = jobs.job_status_record(
        "j1", "user-a", status="failed", stage="upload", scan_id="s1", error="spawn_failed"
    )
    backing = {"j1": stored}
    jobs.configure_jobs_store(backing)

    body = client.get("/api/workspace/jobs/j1").get_json()
    assert body["status"] == "error"
    assert body["error"] == "spawn_failed"
    assert backing["j1"]["status"] == "failed"  # store untouched

    # non-failed statuses pass through verbatim
    jobs.configure_jobs_store(
        {"j2": jobs.job_status_record("j2", "user-a", status="done", stage="done")}
    )
    assert client.get("/api/workspace/jobs/j2").get_json()["status"] == "done"


def test_jobs_store_unavailable_503(demo_app, monkeypatch):
    client, _store, _stub, jobs = demo_app
    monkeypatch.setattr(jobs, "_get_jobs_store", lambda: None)
    resp = client.get("/api/workspace/jobs/j1")
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "jobs_store_unavailable"


# -- 501 stubs --------------------------------------------------------------
#
# Retired at demo integration (integ/demo-2026-07): every W0 stub row was
# replaced by a live route, each covered by its own suite —
#   ingest video/splat        → tests/test_demo_ingest_routes.py   (W1)
#   agent annotate/runs/…     → tests/test_demo_agent.py           (W2)
#   segment/complete/variants → tests/test_demo_sam3d_routes.py    (W3)
# No demo route answers 501 any more, so the parametrized stub test is gone.


# ---------------------------------------------------------------------------
# harness 2: server.app surface (FakeFlask) — save_scene source + scene route
# ---------------------------------------------------------------------------


def test_save_scene_source_kwarg_persists(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path))
    _save_demo_scene(store, scan="with-source", source="recon_video")
    _save_demo_scene(store, scan="without-source")
    assert store.get_scene("user-a", "with-source")["source"] == "recon_video"
    assert store.get_scene("user-a", "without-source")["source"] is None


def test_get_scene_route_surfaces_source(monkeypatch, tmp_path):
    app_module = load_app_module(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    _save_demo_scene(store, scan="scan-src", source="imported_splat")
    app_module.configure_scene_persistence(store)
    monkeypatch.setattr(
        app_module.request,
        "environ",
        {app_module.AUTH_CLAIMS_ENV_KEY: {"sub": "user-a"}},
        raising=False,
    )
    payload = app_module.get_scene_route("scan-src")["args"][0]  # fake jsonify
    assert payload["source"] == "imported_splat"
    assert payload["scan_id"] == "scan-src"

    # pre-demo records (no source key) surface None, not a KeyError
    _save_demo_scene(store, scan="scan-old")
    record = store.get_scene("user-a", "scan-old")
    record.pop("source")
    store._store[store._scene_key("user-a", "scan-old")] = record
    payload = app_module.get_scene_route("scan-old")["args"][0]
    assert payload["source"] is None
