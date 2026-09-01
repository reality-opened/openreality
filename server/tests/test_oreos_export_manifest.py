"""W6 — demo export-manifest route tests (GPU-free).

Harness: the test_demo_routes.py pattern — REAL Flask + ``test_client`` on the demo
blueprint, with ``server.app`` stubbed (auth + a real ``ModalScenePersistence`` on
tmp_path + minimal-but-faithful copies of the isaac gate helpers the route reaches via
the lazy ``app_module()`` indirection; the REAL helpers are covered by
``test_isaac_export_route.py``).

Covers, per the W6 brief:
  * tree matches the builders' output (manifest tree == an independent
    ``export_from_record`` run, path+size for path+size);
  * honest degradation when the scan has no persisted trajectory (the canonical
    demo-ops scene's real state) — 200 + ``absent`` reasons, never a 500/404;
  * auth (401), unknown scan (404), unknown format (400), unknown ?source= (404);
  * isaac gate parity: 409 metric_scale_required / 501 isaac_unavailable / 400 bad
    ?scale=, and (when usd-core+open3d are installed) a real isaac tree manifest.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from server.scene_report.schemas import SceneFacts, SceneReport
from server.scene_report.store import ModalScenePersistence

MANIFEST_URL = "/api/scenes/{scan}/demo/export/manifest"


# ---------------------------------------------------------------------------
# harness — real Flask + stubbed server.app (test_demo_routes.py pattern)
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


def _stub_parse_isaac_scale_arg(raw):
    if raw is None or str(raw).strip() == "":
        return None
    val = float(raw)  # ValueError propagates -> 400 at the route
    if not (math.isfinite(val) and val > 0):
        raise ValueError(f"'scale' must be a positive finite number, got {raw!r}")
    return val


def _stub_resolve_isaac_scale(record, source, scale_arg):
    if scale_arg is not None:
        return ("original", scale_arg, "user:factor", None)
    s = str(source).strip() if source is not None else ""
    if s and s != "original" and s.startswith("derived/anchor/"):
        return (s, 1.0, "anchor:prescaled", None)
    pointer = record.get("derived_latest") if isinstance(record, dict) else None
    if isinstance(pointer, dict) and pointer.get("kind") == "anchor" and pointer.get("source_key"):
        return (str(pointer["source_key"]), 1.0, "anchor:prescaled", pointer.get("scale_factor"))
    return None


@pytest.fixture()
def demo_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(
        _scene_persistence=store,
        _auth_user_id=lambda: "user-a",
        _parse_isaac_scale_arg=_stub_parse_isaac_scale_arg,
        _resolve_isaac_scale=_stub_resolve_isaac_scale,
        _isaac_dependency_status=lambda: [],
    )
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)
    yield app.test_client(), store, stub


# ---------------------------------------------------------------------------
# scene fixtures (test_product_workflow_routes.py builders)
# ---------------------------------------------------------------------------


def _synthetic_trajectory(n=5):
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for i in range(n):
        poses[i, :3, 3] = [float(i), 0.5 * i, 0.0]
    intrinsics = np.tile(np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32), (n, 1))
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "source_frame_id": np.arange(n, dtype=np.float32),
    }


def _synthetic_cloud(n=60, seed=3):
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n, 3)).astype(np.float32) * 2.0
    colors = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return positions, colors


def _save_scene(store, user="user-a", scan="scan1", with_trajectory=True, with_points=True):
    store.save_scene(
        user,
        scan,
        SceneReport(summary="synthetic office scan", room_type="office"),
        SceneFacts(),
        points=_synthetic_cloud() if with_points else None,
        trajectory=_synthetic_trajectory() if with_trajectory else None,
    )


def _paths(body):
    return {e["path"] for e in body["tree"]}


# ---------------------------------------------------------------------------
# registration + auth + validation
# ---------------------------------------------------------------------------


def test_manifest_rule_registered(demo_app):
    client, _store, _stub = demo_app
    rules = {str(r) for r in client.application.url_map.iter_rules()}
    assert "/api/scenes/<scan_id>/demo/export/manifest" in rules


def test_unauthed_is_401(demo_app):
    client, store, stub = demo_app
    _save_scene(store)
    stub._auth_user_id = lambda: None
    res = client.get(MANIFEST_URL.format(scan="scan1"))
    assert res.status_code == 401
    assert res.get_json()["error"] == "invalid_token"


def test_unknown_scan_is_404(demo_app):
    client, _store, _stub = demo_app
    res = client.get(MANIFEST_URL.format(scan="nope"))
    assert res.status_code == 404
    assert res.get_json()["error"] == "not_found"


def test_unknown_format_is_400(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=tarball")
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"] == "invalid_format"
    assert "openreality" in body["message"] and "isaac_usd" in body["message"]


def test_unknown_source_is_404_unknown_derived_key(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(
        MANIFEST_URL.format(scan="scan1") + "?format=openreality&source=derived/anchor/x/cloud.ply"
    )
    assert res.status_code == 404
    assert res.get_json()["error"] == "unknown_derived_key"


# ---------------------------------------------------------------------------
# openreality / groot manifests — tree matches the builders' output
# ---------------------------------------------------------------------------


def test_openreality_manifest_matches_builder_tree(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=openreality")
    assert res.status_code == 200
    body = res.get_json()

    # Independent build of the SAME record through the SAME builder — path+size parity.
    from server.export.record import load_record_from_store
    from server.export.writer import export_from_record

    normalized = load_record_from_store(store, "user-a", "scan1")
    assert normalized is not None
    with tempfile.TemporaryDirectory() as tmp:
        base = export_from_record(normalized, os.path.join(tmp, "openreality"))
        expected = {}
        for dirpath, _d, files in os.walk(base):
            for fn in files:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, base).replace(os.sep, "/")
                expected[f"{os.path.basename(base)}/{rel}"] = os.path.getsize(full)

    got = {e["path"]: e["size"] for e in body["tree"]}
    assert got == expected
    assert body["total_bytes"] == sum(expected.values())
    assert body["file_count"] == len(expected)

    assert "scan1/meta/info.json" in got
    assert "scan1/data/trajectory.parquet" in got
    assert "scan1/clouds/cloud.npz" in got
    assert body["info"] is not None and body["info"].get("scan_id") == "scan1"
    assert body["zip_available"] is True
    assert body["zip_blocked_reason"] is None
    # Record-path exports NEVER carry the ego-view video (structural, honest):
    absent_components = {a["component"] for a in body["absent"]}
    assert "videos/ego_view.mp4" in absent_components
    assert body["complete"] is False  # because of the structural absences above


def test_groot_manifest_has_lerobot_meta(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=groot_lerobot_v2")
    assert res.status_code == 200
    body = res.get_json()
    paths = _paths(body)
    base = "scan1_groot_lerobot_v2"
    assert f"{base}/meta/info.json" in paths
    assert f"{base}/meta/modality.json" in paths
    assert f"{base}/meta/episodes.jsonl" in paths
    assert f"{base}/meta/tasks.jsonl" in paths
    assert f"{base}/data/chunk-000/episode_000000.parquet" in paths

    assert body["modality"] is not None
    assert "state" in body["modality"] and "action" in body["modality"]
    assert isinstance(body["episodes"], list) and body["episodes"][0]["length"] == 5
    assert isinstance(body["tasks"], list) and len(body["tasks"]) >= 1
    assert body["info"]["total_frames"] == 5
    assert body["zip_available"] is True


# ---------------------------------------------------------------------------
# honest degradation — no trajectory (the canonical scene's real state)
# ---------------------------------------------------------------------------


def test_openreality_degrades_honestly_without_trajectory(demo_app):
    client, store, _stub = demo_app
    _save_scene(store, with_trajectory=False)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=openreality")
    assert res.status_code == 200  # never a 404/500 — the zip route 404s here
    body = res.get_json()
    assert body["complete"] is False
    components = {a["component"]: a["reason"] for a in body["absent"]}
    assert "trajectory" in components
    assert "trajectory_key=None" in components["trajectory"]
    assert body["zip_available"] is False
    assert body["zip_blocked_reason"] == "no_trajectory"
    # The partial tree still materializes what CAN be built:
    paths = _paths(body)
    assert "scan1/meta/info.json" in paths
    assert "scan1/clouds/cloud.npz" in paths
    assert "scan1/data/trajectory.parquet" in paths  # 0-row parquet, honestly listed


def test_groot_degrades_honestly_without_trajectory(demo_app):
    client, store, _stub = demo_app
    _save_scene(store, with_trajectory=False)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=groot_lerobot_v2")
    assert res.status_code == 200
    body = res.get_json()
    assert body["complete"] is False
    components = {a["component"] for a in body["absent"]}
    assert "trajectory" in components
    assert "groot_lerobot_v2" in components  # conversion blocked on a 0-pose export
    assert body["tree"] == []
    assert body["zip_available"] is False


def test_degraded_scene_still_404s_on_zip_route_semantics_not_manifest(demo_app):
    """Documents the ONE deliberate divergence: same scan, zip semantics vs manifest.
    (The zip route lives in server.app — here we assert only the manifest half; the
    zip route's 404 no_trajectory is covered by test_product_workflow_routes.py.)"""
    client, store, _stub = demo_app
    _save_scene(store, with_trajectory=False)
    res = client.get(MANIFEST_URL.format(scan="scan1"))
    assert res.status_code == 200
    assert res.get_json()["zip_blocked_reason"] == "no_trajectory"


# ---------------------------------------------------------------------------
# isaac gate parity
# ---------------------------------------------------------------------------


def test_isaac_unanchored_is_409_metric_scale_required(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=isaac_usd")
    assert res.status_code == 409
    body = res.get_json()
    assert body["error"] == "metric_scale_required"
    assert "Metric anchor" in body["message"]


def test_isaac_bad_scale_is_400(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=isaac_usd&scale=-3")
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_request"


def test_isaac_missing_deps_is_501(demo_app):
    client, store, stub = demo_app
    _save_scene(store)
    stub._isaac_dependency_status = lambda: ["usd-core"]
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=isaac_usd&scale=2.0")
    assert res.status_code == 501
    body = res.get_json()
    assert body["error"] == "isaac_unavailable"
    assert "usd-core" in body["message"]


def test_isaac_no_points_is_404(demo_app):
    client, store, _stub = demo_app
    _save_scene(store, with_points=False)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=isaac_usd&scale=2.0")
    assert res.status_code == 404
    assert res.get_json()["error"] == "no_points"


_ISAAC_DEPS_MISSING = [
    name
    for spec, name in ((importlib.util.find_spec("pxr"), "usd-core"), (importlib.util.find_spec("open3d"), "open3d"))
    if spec is None
]


@pytest.mark.skipif(bool(_ISAAC_DEPS_MISSING), reason=f"isaac deps not installed: {_ISAAC_DEPS_MISSING}")
def test_isaac_manifest_tree_with_explicit_scale(demo_app):
    client, store, _stub = demo_app
    _save_scene(store)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=isaac_usd&scale=2.0")
    assert res.status_code == 200
    body = res.get_json()
    paths = _paths(body)
    assert "scan1/isaac/scene.usd" in paths
    assert "scan1/isaac/trajectory.usd" in paths
    assert "scan1/isaac/manifest.json" in paths
    assert body["complete"] is True and body["absent"] == []
    # ``info`` is the tree's own IsaacManifest (alignment provenance), parsed:
    assert body["info"]["alignment"]["scale_source"] == "user:factor"
    assert body["scale"]["scale"] == 2.0
    assert body["total_bytes"] == sum(e["size"] for e in body["tree"])


# ---------------------------------------------------------------------------
# response is JSON-serializable end-to-end (jsonify already proved it; belt+braces on
# the degraded body, whose reasons are hand-built strings)
# ---------------------------------------------------------------------------


def test_degraded_body_roundtrips_json(demo_app):
    client, store, _stub = demo_app
    _save_scene(store, with_trajectory=False)
    res = client.get(MANIFEST_URL.format(scan="scan1") + "?format=groot_lerobot_v2")
    assert json.loads(json.dumps(res.get_json()))["format"] == "groot_lerobot_v2"
