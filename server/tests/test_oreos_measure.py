"""F7 measurement route + demo/measure.py math tests (W2) — GPU-free.

Same harness as test_demo_agent: real Flask over the demo blueprint, stubbed
``server.app``. Covers the protocol ``MeasureResponse`` shape, the metric state
machine (relative ↔ anchored ↔ depth-anchored wording), degenerate/malformed
inputs, the collinear-angle case, and the persisted audit doc.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

flask = pytest.importorskip("flask")

from server.scene_report.schemas import SceneFacts, SceneReport
from server.scene_report.store import ModalScenePersistence


def _fresh_demo_package():
    for name in [
        m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def measure_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
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
        client=app.test_client(), store=store, stub=stub, tmp_path=tmp_path
    )


def _save_scene(store, scan="scan1", anchored=False, scale=2.0):
    store.save_scene(
        "user-a", scan, SceneReport(summary="s", room_type="office"), SceneFacts()
    )
    if anchored:
        store.set_derived_pointer(
            "user-a",
            scan,
            {
                "kind": "anchor",
                "cloud_key": "derived/anchor/t/cloud.ply",
                "source_key": "derived/anchor/t/cloud.ply",
                "scale_factor": scale,
                "applied_at": "2026-07-30T00:00:00+00:00",
            },
        )


def _post(env, scan, body):
    return env.client.post(f"/api/scenes/{scan}/measure", json=body)


# -- distance ---------------------------------------------------------------


def test_distance_relative(measure_app):
    env = measure_app
    _save_scene(env.store)
    resp = _post(env, "scan1", {"kind": "distance", "points_world": [[0, 0, 0], [3, 4, 0]]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["kind"] == "distance"
    assert body["value"] == pytest.approx(5.0)
    assert body["units"] == "relative"
    assert body["scale_factor"] is None
    assert body["scale_source"] == "none"
    assert body["value_slam"] == pytest.approx(5.0)
    assert body["metric_state"] == "up_to_scale"
    assert "degrees" not in body


def test_distance_anchored_scales_to_metres(measure_app):
    env = measure_app
    _save_scene(env.store, scan="scan-a", anchored=True, scale=2.0)
    resp = _post(env, "scan-a", {"kind": "distance", "points_world": [[0, 0, 0], [3, 4, 0]]})
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["value"] == pytest.approx(10.0)
    assert body["units"] == "m"
    assert body["scale_factor"] == pytest.approx(2.0)
    assert body["scale_source"] == "anchor:derived/anchor/t/cloud.ply"
    assert body["value_slam"] == pytest.approx(5.0)
    assert body["metric_state"] == "metric_anchored"


def test_non_anchor_pointer_stays_relative(measure_app):
    """A clamp pointer (or malformed factor) must NOT unlock metres."""
    env = measure_app
    _save_scene(env.store, scan="scan-c")
    env.store.set_derived_pointer(
        "user-a",
        "scan-c",
        {"kind": "clamp", "source_key": "derived/clamp/x/splat.ply", "scale_factor": 3.0},
    )
    body = _post(
        env, "scan-c", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]}
    ).get_json()
    assert body["units"] == "relative"

    env.store.set_derived_pointer(
        "user-a",
        "scan-c",
        {"kind": "anchor", "source_key": "derived/anchor/x/cloud.ply", "scale_factor": -1.0},
    )
    body = _post(
        env, "scan-c", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]}
    ).get_json()
    assert body["units"] == "relative"


def test_distance_degenerate_422(measure_app):
    env = measure_app
    _save_scene(env.store)
    resp = _post(
        env, "scan1", {"kind": "distance", "points_world": [[1, 1, 1], [1, 1, 1]]}
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "degenerate_points"


# -- angle ------------------------------------------------------------------


def test_angle_right_angle(measure_app):
    env = measure_app
    _save_scene(env.store)
    resp = _post(
        env,
        "scan1",
        {"kind": "angle", "points_world": [[1, 0, 0], [0, 0, 0], [0, 1, 0]]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["kind"] == "angle"
    assert body["degrees"] == pytest.approx(90.0)
    assert body["value"] == pytest.approx(90.0)
    assert body["units"] == "relative"


def test_angle_collinear_is_valid_180(measure_app):
    env = measure_app
    _save_scene(env.store)
    body = _post(
        env,
        "scan1",
        {"kind": "angle", "points_world": [[-1, 0, 0], [0, 0, 0], [5, 0, 0]]},
    ).get_json()
    assert body["degrees"] == pytest.approx(180.0)


def test_angle_scale_invariant_under_anchor(measure_app):
    env = measure_app
    _save_scene(env.store, scan="scan-a", anchored=True, scale=7.5)
    body = _post(
        env,
        "scan-a",
        {"kind": "angle", "points_world": [[1, 0, 0], [0, 0, 0], [0, 1, 0]]},
    ).get_json()
    assert body["degrees"] == pytest.approx(90.0)  # never scaled
    assert body["units"] == "m"  # chip reflects scene metric state


def test_angle_degenerate_vertex_422(measure_app):
    env = measure_app
    _save_scene(env.store)
    resp = _post(
        env,
        "scan1",
        {"kind": "angle", "points_world": [[0, 0, 0], [0, 0, 0], [1, 0, 0]]},
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "degenerate_points"


# -- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected_error",
    [
        ({"kind": "distance", "points_world": [[0, 0, 0]]}, "invalid_points"),
        ({"kind": "distance", "points_world": [[0, 0, 0], [1, 0]]}, "invalid_points"),
        ({"kind": "angle", "points_world": [[0, 0, 0], [1, 0, 0]]}, "invalid_points"),
        ({"kind": "distance", "points_world": [[0, 0, 0], ["x", 0, 0]]}, "invalid_points"),
        (
            {"kind": "distance", "points_world": [[0, 0, 0], [float("nan")] * 3]},
            "invalid_points",
        ),
        ({"kind": "distance", "points_world": "nope"}, "invalid_points"),
        ({"kind": "volume", "points_world": [[0, 0, 0], [1, 0, 0]]}, "invalid_kind"),
        ({"points_world": [[0, 0, 0], [1, 0, 0]]}, "invalid_kind"),
    ],
)
def test_measure_bad_requests_400(measure_app, body, expected_error):
    env = measure_app
    _save_scene(env.store)
    # NaN can't ride JSON strictly; build the payload via data= for that case
    resp = env.client.post(
        "/api/scenes/scan1/measure",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 400, resp.get_json()
    assert resp.get_json()["error"] == expected_error


def test_measure_non_dict_body_400(measure_app):
    env = measure_app
    _save_scene(env.store)
    resp = env.client.post(
        "/api/scenes/scan1/measure", data="[]", content_type="application/json"
    )
    assert resp.status_code == 400


def test_measure_unknown_scene_404_and_unauth_401(measure_app):
    env = measure_app
    resp = _post(env, "ghost", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]})
    assert resp.status_code == 404
    env.stub._auth_user_id = lambda: None
    resp = _post(env, "ghost", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]})
    assert resp.status_code == 401


def test_measure_cross_user_isolation(measure_app):
    env = measure_app
    _save_scene(env.store)
    env.stub._auth_user_id = lambda: "user-b"
    resp = _post(env, "scan1", {"kind": "distance", "points_world": [[0, 0, 0], [1, 0, 0]]})
    assert resp.status_code == 404


# -- audit + provenance -----------------------------------------------------


def test_measure_persists_audit_doc(measure_app):
    env = measure_app
    _save_scene(env.store)
    _post(env, "scan1", {"kind": "distance", "points_world": [[0, 0, 0], [3, 4, 0]]})
    pattern = os.path.join(
        str(env.tmp_path), "user-a", "scan1", "derived", "demo", "measurements", "*", "measure.json"
    )
    matches = glob.glob(pattern)
    assert len(matches) == 1
    audit = json.loads(open(matches[0]).read())
    assert audit["result"]["value"] == pytest.approx(5.0)
    assert audit["request"]["kind"] == "distance"


def test_depth_provenance_changes_wording_only(measure_app):
    """A Gemini-2 depth anchor (provenance doc present) keeps the math identical but
    the resolved wording becomes 'depth-scaled estimate' (CLAIM-LEDGER row)."""
    env = measure_app
    _save_scene(env.store, scan="scan-d", anchored=True, scale=2.0)
    env.store.save_derived_artifact(
        "user-a",
        "scan-d",
        "demo/metric/provenance.json",
        json.dumps({"method": "gemini2_depth_ratio", "cov": 0.1}).encode(),
    )
    body = _post(
        env, "scan-d", {"kind": "distance", "points_world": [[0, 0, 0], [3, 4, 0]]}
    ).get_json()
    assert body["value"] == pytest.approx(10.0)  # math unchanged
    assert body["units"] == "m"

    from server.oreos import measure as m

    record = env.store.get_scene("user-a", "scan-d")
    state = m.resolve_metric(record, {"method": "gemini2_depth_ratio"})
    assert state.wording == "depth-scaled estimate (metres)"
    assert m.resolve_metric(record).wording == "metres"
