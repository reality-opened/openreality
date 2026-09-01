"""W5 demo planes tests (docs/demo-2026-07 F5/F6) — GPU-free.

Covers ``server/oreos/planes.py`` (constrained wall RANSAC, up-vector ladder,
floor-patch synthesis) and ``server/oreos/routes_planes.py`` (the /planes and
/objects/<uid>/floor_patch routes) on the test_demo_routes harness: real Flask
test_client, stubbed ``server.app`` (auth + a real ``ModalScenePersistence`` on
tmp_path).

The W4 pathplan seam is exercised BOTH ways:
- a fake ``server.oreos.pathplan`` module (exp25 ``fit_floor_plane`` signature)
  injected via ``sys.modules`` proves the success path + the exact call shape;
- its absence (the true state of this branch until feat/demo-pathplan merges)
  proves the 503 ``pathplan_not_merged`` fallback.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.scene_report.schemas import SceneFacts, SceneReport
from server.scene_report.store import ModalScenePersistence
from server.scene_report.splat_io import read_splat_ply

# ---------------------------------------------------------------------------
# harness (test_demo_routes pattern)
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    """(Re)import ``server.oreos`` under the CURRENTLY active flask module (a
    sibling test may have cached it under conftest's fake flask). Also drops
    any injected fake ``server.oreos.pathplan`` from a previous test."""
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
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)
    yield app.test_client(), store, stub


@pytest.fixture()
def fake_pathplan(monkeypatch):
    """Inject a fake ``server.oreos.pathplan`` with the exp25 seam surface:
    ``fit_floor_plane(points, up_vec, rng, ..., ransac_thresh=...) ->
    FloorPlane(point, normal, u_axis, v_axis)``. Records its calls so the seam
    shape is asserted, not assumed. Request AFTER ``demo_app`` in test params —
    the demo_app fixture pops ``server.oreos.*`` from ``sys.modules``."""
    mod = types.ModuleType("server.oreos.pathplan")
    calls: list[dict] = []

    class FloorPlane:
        def __init__(self, point, normal, u_axis, v_axis):
            self.point = point
            self.normal = normal
            self.u_axis = u_axis
            self.v_axis = v_axis

    def fit_floor_plane(points, up_vec, rng, **kwargs):
        calls.append({"n_points": int(np.asarray(points).shape[0]), "rng": rng, "kwargs": kwargs})
        up = np.asarray(up_vec, dtype=np.float64)
        up = up / np.linalg.norm(up)
        heights = np.asarray(points, dtype=np.float64) @ up
        floor_h = float(np.quantile(heights, 0.05))
        near = np.abs(heights - floor_h) < max(0.05, float(kwargs.get("ransac_thresh", 0.03)))
        pts = np.asarray(points, dtype=np.float64)
        point = pts[near].mean(axis=0) if int(near.sum()) >= 3 else pts.mean(axis=0)
        point = point - float((point @ up) - floor_h) * up  # snap onto the shelf height
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(up @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u_axis = ref - float(ref @ up) * up
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(up, u_axis)
        return FloorPlane(point=point, normal=up, u_axis=u_axis, v_axis=v_axis)

    mod.FloorPlane = FloorPlane
    mod.fit_floor_plane = fit_floor_plane
    mod.calls = calls
    monkeypatch.setitem(sys.modules, "server.oreos.pathplan", mod)
    return mod


# ---------------------------------------------------------------------------
# synthetic room fixture: floor (z=0) + two walls (x=0, y=4) + a table slab
# (z=0.8, horizontal — must never read as a wall) + uniform noise. Up = +z.
# ---------------------------------------------------------------------------


def _room_cloud(seed=0, n_floor=4000, n_wall_x=1800, n_wall_y=1800, n_table=700, n_noise=150):
    rng = np.random.default_rng(seed)
    floor = np.stack(
        [rng.uniform(0, 6, n_floor), rng.uniform(0, 4, n_floor), rng.normal(0, 0.008, n_floor)], axis=1
    )
    wall_x = np.stack(
        [rng.normal(0, 0.008, n_wall_x), rng.uniform(0, 4, n_wall_x), rng.uniform(0, 2.5, n_wall_x)],
        axis=1,
    )
    wall_y = np.stack(
        [rng.uniform(0, 6, n_wall_y), rng.normal(4, 0.008, n_wall_y), rng.uniform(0, 2.5, n_wall_y)],
        axis=1,
    )
    table = np.stack(
        [rng.uniform(2, 3.2, n_table), rng.uniform(1, 1.8, n_table), rng.normal(0.8, 0.008, n_table)],
        axis=1,
    )
    noise = np.stack(
        [rng.uniform(0, 6, n_noise), rng.uniform(0, 4, n_noise), rng.uniform(0, 2.5, n_noise)], axis=1
    )
    positions = np.concatenate([floor, wall_x, wall_y, table, noise]).astype(np.float32)
    colors = np.concatenate(
        [
            np.tile(np.array([60, 160, 60], np.uint8), (n_floor, 1)),  # green floor
            np.tile(np.array([150, 150, 150], np.uint8), (n_wall_x, 1)),
            np.tile(np.array([150, 150, 150], np.uint8), (n_wall_y, 1)),
            np.tile(np.array([120, 80, 40], np.uint8), (n_table, 1)),
            np.asarray(np.random.default_rng(seed + 1).integers(0, 255, (n_noise, 3)), np.uint8),
        ]
    )
    return positions, colors


def _fake_floor_plane():
    return types.SimpleNamespace(
        point=np.array([3.0, 2.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
    )


def _identity_obb(center=(1.5, 2.0, 0.3), extents=(0.8, 0.6, 0.6)):
    return {
        "center": list(center),
        "extents": list(extents),
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _save_scene(store, user="user-a", scan="scan1", **kwargs):
    store.save_scene(
        user,
        scan,
        SceneReport(summary="s", room_type="office"),
        SceneFacts(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# unit: detect_walls
# ---------------------------------------------------------------------------


def _planes_module():
    _fresh_demo_package()
    return importlib.import_module("server.oreos.planes")


def test_detect_walls_finds_both_walls_and_only_walls():
    planes = _planes_module()
    positions, _ = _room_cloud()
    up = np.array([0.0, 0.0, 1.0])
    walls = planes.detect_walls(positions, up, seed=0)

    assert len(walls) == 2, [w["inlier_count"] for w in walls]
    assert [w["id"] for w in walls] == ["wall_0", "wall_1"]
    # strongest-first ordering
    assert walls[0]["inlier_count"] >= walls[1]["inlier_count"]
    for w in walls:
        n = np.asarray(w["normal"])
        assert abs(float(n @ up)) < planes.WALL_VERTICAL_TOL  # the F5 constraint
        assert w["inlier_count"] > 1000
        assert w["thickness"] > 0
    # the two walls are the x=0 and y=4 planes (order free, sign free)
    axes = sorted(int(np.argmax(np.abs(np.asarray(w["normal"])))) for w in walls)
    assert axes == [0, 1]
    for w in walls:
        n = np.abs(np.asarray(w["normal"]))
        assert float(n.max()) > 0.98  # tight normals on a clean fixture
        # v (in-plane vertical) spans roughly the 2.5-unit wall height
        (v_lo, v_hi) = w["uv_bounds"][1]
        assert (v_hi - v_lo) > 2.0


def test_detect_walls_never_returns_the_table_or_floor():
    planes = _planes_module()
    positions, _ = _room_cloud()
    up = np.array([0.0, 0.0, 1.0])
    walls = planes.detect_walls(positions, up, seed=0)
    for w in walls:
        # a horizontal surface (floor z=0, table z=0.8) would have |n.up| ~ 1
        assert abs(float(np.asarray(w["normal"]) @ up)) < 0.2


def test_detect_walls_respects_max_planes_and_degenerate_input():
    planes = _planes_module()
    positions, _ = _room_cloud()
    up = np.array([0.0, 0.0, 1.0])
    assert len(planes.detect_walls(positions, up, max_planes=1, seed=0)) == 1
    assert planes.detect_walls(np.zeros((2, 3)), up) == []
    line = np.stack([np.linspace(0, 1, 500), np.zeros(500), np.zeros(500)], axis=1)
    assert planes.detect_walls(line, up, min_inliers=10) == []  # collinear: no plane


# ---------------------------------------------------------------------------
# unit: estimate_up
# ---------------------------------------------------------------------------


def test_estimate_up_pose_derived():
    planes = _planes_module()
    # c2w with camera +y column = [0,0,-1] (camera down = world -z) -> up = +z
    c2w = np.eye(4)
    c2w[:3, 0] = [1.0, 0.0, 0.0]
    c2w[:3, 1] = [0.0, 0.0, -1.0]
    c2w[:3, 2] = [0.0, 1.0, 0.0]
    poses = np.stack([c2w] * 5)
    up, source = planes.estimate_up(np.zeros((10, 3)), poses=poses)
    assert source == "poses"
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9)


def test_estimate_up_override_and_validation():
    planes = _planes_module()
    up, source = planes.estimate_up(np.zeros((10, 3)), up_override=[0.0, 0.0, 2.0])
    assert source == "user_override"
    assert np.allclose(up, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        planes.estimate_up(np.zeros((10, 3)), up_override=[1.0, 2.0])
    with pytest.raises(ValueError):
        planes.estimate_up(np.zeros((10, 3)), up_override=[np.nan, 0.0, 1.0])


def test_estimate_up_heuristic_from_dominant_floor():
    planes = _planes_module()
    positions, _ = _room_cloud()
    up, source = planes.estimate_up(positions, poses=None)
    assert source == "heuristic"
    assert float(up @ np.array([0.0, 0.0, 1.0])) > 0.9  # floor-dominant room, cloud above


# ---------------------------------------------------------------------------
# unit: floor candidate + floor patch synthesis
# ---------------------------------------------------------------------------


def test_floor_candidate_covers_the_floor():
    planes = _planes_module()
    positions, _ = _room_cloud()
    cand = planes.floor_candidate(positions, _fake_floor_plane(), thresh=0.05)
    assert cand["id"] == "floor" and cand["kind"] == "floor"
    assert cand["inlier_count"] > 3000
    (u_lo, u_hi), (v_lo, v_hi) = cand["uv_bounds"]
    assert (u_hi - u_lo) > 5.0 and (v_hi - v_lo) > 3.0  # ~6x4 floor


def test_synthesize_floor_patch_fields_and_colors():
    planes = _planes_module()
    positions, colors = _room_cloud()
    obb = _identity_obb()
    fields, info = planes.synthesize_floor_patch(
        positions,
        colors.astype(np.float64) / 255.0,
        _fake_floor_plane(),
        np.asarray(obb["center"]),
        np.asarray(obb["extents"]),
        np.asarray(obb["rotation"]),
        n_gaussians=1500,
        seed=0,
    )
    expected_keys = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    assert list(fields.keys()) == expected_keys
    assert all(fields[k].shape == (1500,) and fields[k].dtype == np.float32 for k in expected_keys)

    # positions: inside the footprint rect, on the floor
    assert float(np.abs(fields["z"]).max()) < 0.1
    assert fields["x"].min() >= 1.1 - 1e-3 and fields["x"].max() <= 1.9 + 1e-3
    assert fields["y"].min() >= 1.7 - 1e-3 and fields["y"].max() <= 2.3 + 1e-3

    # colors: SH-DC-encoded green (the floor color), not gray/brown
    sh_c0 = 0.28209479177387814
    r = 0.5 + sh_c0 * float(fields["f_dc_0"].mean())
    g = 0.5 + sh_c0 * float(fields["f_dc_1"].mean())
    b = 0.5 + sh_c0 * float(fields["f_dc_2"].mean())
    assert g > r + 0.15 and g > b + 0.15
    assert abs(g - 160 / 255) < 0.1

    # quaternion is unit, scales are log-space (thin along the normal)
    q = np.array([float(fields[f"rot_{i}"][0]) for i in range(4)])
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-6
    assert float(fields["scale_2"][0]) < float(fields["scale_0"][0])  # normal thinner than in-plane

    assert info["n_gaussians"] == 1500
    assert info["n_annulus_points"] >= 20
    assert not info["annulus_fallback_all_floor"]


def test_synthesize_floor_patch_insufficient_context():
    planes = _planes_module()
    rng = np.random.default_rng(0)
    # a blob far above the floor plane: zero floor inliers within the band
    positions = rng.normal(size=(200, 3)) * 0.1 + np.array([0.0, 0.0, 5.0])
    colors = np.full((200, 3), 0.5)
    obb = _identity_obb()
    with pytest.raises(planes.InsufficientFloorContext):
        planes.synthesize_floor_patch(
            positions,
            colors,
            _fake_floor_plane(),
            np.asarray(obb["center"]),
            np.asarray(obb["extents"]),
            np.asarray(obb["rotation"]),
        )


# ---------------------------------------------------------------------------
# routes: registration + /planes
# ---------------------------------------------------------------------------


def test_w5_routes_registered(demo_app):
    client, _store, _stub = demo_app
    rules = {str(r) for r in client.application.url_map.iter_rules()}
    assert "/api/scenes/<scan_id>/planes" in rules
    assert "/api/scenes/<scan_id>/objects/<uid>/floor_patch" in rules


def test_planes_route_full_flow_and_cache(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())

    resp = client.post("/api/scenes/scan1/planes")
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["cached"] is False
    assert body["planes"][0]["id"] == "floor"
    walls = [p for p in body["planes"] if p["kind"] == "wall"]
    assert len(walls) == 2
    assert body["up_source"] == "heuristic"  # no trajectory saved
    assert body["units"] == "relative" and body["units_basis"] == "slam_world_units"
    assert body["parent_artifact"] == "cloud.npz"
    assert "human-confirmed" in body["provenance"]
    for plane in body["planes"]:
        for key in ("id", "kind", "point", "normal", "u_axis", "v_axis", "uv_bounds", "thickness", "inlier_count"):
            assert key in plane, key

    # the seam was called with the exp25 signature (rng Generator + ransac_thresh kwarg)
    assert fake_pathplan.calls, "pathplan.fit_floor_plane was never called"
    call = fake_pathplan.calls[0]
    assert isinstance(call["rng"], np.random.Generator)
    assert "ransac_thresh" in call["kwargs"]
    assert call["kwargs"]["ransac_thresh"] == pytest.approx(body["thresh"])

    # cache: second call is served from memory, force recomputes
    assert client.post("/api/scenes/scan1/planes").get_json()["cached"] is True
    assert client.post("/api/scenes/scan1/planes", json={"force": True}).get_json()["cached"] is False


def test_planes_route_pathplan_missing_503(demo_app, monkeypatch):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())
    # pathplan IS merged now — simulate its absence: a None sys.modules entry
    # makes importlib.import_module raise ImportError, the real degradation path.
    monkeypatch.setitem(sys.modules, "server.oreos.pathplan", None)
    resp = client.post("/api/scenes/scan1/planes")
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "pathplan_not_merged"


def test_planes_route_no_cloud_422(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store)  # no points
    resp = client.post("/api/scenes/scan1/planes")
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "no_geometry"


def test_planes_route_auth_and_scene_gates(demo_app):
    client, store, stub = demo_app
    resp = client.post("/api/scenes/nope/planes")
    assert resp.status_code == 404
    _save_scene(store)
    stub._auth_user_id = lambda: None
    resp = client.post("/api/scenes/scan1/planes")
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "invalid_token"}


def test_planes_route_bad_params_400(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())
    resp = client.post("/api/scenes/scan1/planes", json={"up_override": [1.0, 2.0]})
    assert resp.status_code == 400
    resp = client.post("/api/scenes/scan1/planes", json={"max_walls": 99})
    assert resp.status_code == 400


def test_planes_route_up_override_reported(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())
    resp = client.post(
        "/api/scenes/scan1/planes", json={"up_override": [0.0, 0.0, 1.0], "force": True}
    )
    assert resp.status_code == 200
    assert resp.get_json()["up_source"] == "user_override"


# ---------------------------------------------------------------------------
# routes: /objects/<uid>/floor_patch
# ---------------------------------------------------------------------------


def test_floor_patch_route_writes_artifacts(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())

    resp = client.post(
        "/api/scenes/scan1/objects/sel:123/floor_patch", json={"obb": _identity_obb()}
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    # uid colon sanitized into the derived key, honesty envelope present
    assert body["patch_key"] == "derived/demo/objects/sel-123/floor_patch/patch.ply"
    assert body["meta_key"] == "derived/demo/objects/sel-123/floor_patch/meta.json"
    assert body["n_gaussians"] == 2000
    meta = body["meta"]
    assert meta["generated"] is True
    assert meta["generator"] == "floor_patch"
    assert meta["provenance"] == "generated floor patch"
    assert any("never captured" in c for c in meta["caveats"])
    assert meta["inputs"]["object_uid"] == "sel:123"
    assert meta["scan_id"] == "scan1" and meta["parent_artifact"] == "cloud.npz"

    # the written PLY is a readable gaussian splat with the full field schema
    path = store.get_derived_artifact_path("user-a", "scan1", body["patch_key"])
    assert path is not None
    fields = read_splat_ply(path)
    assert fields["x"].shape == (2000,)
    for key in ("f_dc_0", "opacity", "scale_0", "rot_0"):
        assert key in fields

    # idempotent: re-patching the same uid overwrites the same key
    resp2 = client.post(
        "/api/scenes/scan1/objects/sel:123/floor_patch",
        json={"obb": _identity_obb(), "n_gaussians": 500},
    )
    assert resp2.status_code == 200
    assert resp2.get_json()["patch_key"] == body["patch_key"]
    fields2 = read_splat_ply(store.get_derived_artifact_path("user-a", "scan1", body["patch_key"]))
    assert fields2["x"].shape == (500,)


def test_floor_patch_route_validation(demo_app, fake_pathplan):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())
    # missing / malformed obb
    assert client.post("/api/scenes/scan1/objects/u1/floor_patch").status_code == 400
    bad = _identity_obb()
    bad["extents"] = [0.0, 0.6, 0.6]
    assert (
        client.post("/api/scenes/scan1/objects/u1/floor_patch", json={"obb": bad}).status_code == 400
    )
    skewed = _identity_obb()
    skewed["rotation"] = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert (
        client.post("/api/scenes/scan1/objects/u1/floor_patch", json={"obb": skewed}).status_code
        == 400
    )
    # n_gaussians out of range
    assert (
        client.post(
            "/api/scenes/scan1/objects/u1/floor_patch",
            json={"obb": _identity_obb(), "n_gaussians": 0},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/scenes/scan1/objects/u1/floor_patch",
            json={"obb": _identity_obb(), "n_gaussians": 999_999},
        ).status_code
        == 400
    )


def test_floor_patch_route_pathplan_missing_503(demo_app, monkeypatch):
    client, store, _stub = demo_app
    _save_scene(store, points=_room_cloud())
    # simulate W4 absence post-merge (see test_planes_route_pathplan_missing_503)
    monkeypatch.setitem(sys.modules, "server.oreos.pathplan", None)
    resp = client.post(
        "/api/scenes/scan1/objects/u1/floor_patch", json={"obb": _identity_obb()}
    )
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "pathplan_not_merged"
