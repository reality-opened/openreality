"""W-B imported-splat objects — GPU-free, no network.

Two halves:

* the pure pipeline (``server/oreos/imported_objects.py``) on a synthetic room whose
  ground truth is known exactly — a floor, a ceiling, four walls, and three objects of
  stated size at stated places;
* the routes + persistence, on the ``test_demo_sam3d_routes.py`` harness (real Flask,
  real ``ModalScenePersistence`` on tmp_path, stubbed ``server.app``).

The synthetic-view labelling is exercised with a MOCK client throughout: workstream A's
endpoint may not exist yet, so this codes to IMPORTED-SPLAT-CONTRACT.md Interface 1 and
proves the seams, including the projection round-trip the contract calls its
highest-risk line.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import sys
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("scipy")
PIL_Image = pytest.importorskip("PIL.Image")

from server.oreos import imported_objects as imp
from server.oreos import segment_geometry as sg
from server.scene_report.schemas import ObjectInstance
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# synthetic room — 10 x 8 footprint, 3 tall, y up. Ground truth lives here.
# ---------------------------------------------------------------------------

TABLE = {"center": [1.0, 0.4, 0.5], "size": [1.4, 0.8, 0.9]}
CHAIR = {"center": [-1.5, 0.45, -1.0], "size": [0.5, 0.9, 0.5]}
SHELF = {"center": [-4.6, 1.0, 2.0], "size": [0.4, 2.0, 1.2]}  # flush to the x = -5 wall


def _plane(rng, n, p0, u, v, su, sv):
    a = rng.uniform(-su, su, n)
    b = rng.uniform(-sv, sv, n)
    return np.asarray(p0, float) + np.outer(a, u) + np.outer(b, v)


def _box_surface(rng, n, center, size):
    c = np.asarray(center, float)
    half = np.asarray(size, float) / 2.0
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            q = rng.uniform(-1, 1, (n // 6, 3)) * half
            q[:, axis] = sign * half[axis]
            faces.append(c + q)
    return np.concatenate(faces)


def _room(seed=0):
    rng = np.random.default_rng(seed)
    x, y, z = np.eye(3)
    parts = [
        _plane(rng, 120_000, [0, 0, 0], x, z, 5, 4),        # floor
        _plane(rng, 80_000, [0, 3, 0], x, z, 5, 4),         # ceiling
        _plane(rng, 60_000, [-5, 1.5, 0], y, z, 1.5, 4),
        _plane(rng, 60_000, [5, 1.5, 0], y, z, 1.5, 4),
        _plane(rng, 60_000, [0, 1.5, -4], x, y, 5, 1.5),
        _plane(rng, 60_000, [0, 1.5, 4], x, y, 5, 1.5),
        _box_surface(rng, 30_000, TABLE["center"], TABLE["size"]),
        _box_surface(rng, 20_000, CHAIR["center"], CHAIR["size"]),
        _box_surface(rng, 20_000, SHELF["center"], SHELF["size"]),
    ]
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture(scope="module")
def room():
    return _room()


@pytest.fixture(scope="module")
def detected(room):
    return imp.detect_objects(room, up=[0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def test_finds_the_three_objects_and_nothing_else(detected):
    assert len(detected.clusters) == 3
    # Ground-truth footprints, longest edge first (chair 0.5x0.9x0.5, table 1.4x0.8x0.9,
    # shelf 0.4x2.0x1.2). Heights come out short because the floor band — removed at 2x
    # for fringe suppression — eats the bottom of anything standing on it, and the shelf
    # loses its wall-side face to the wall band.
    got = sorted(tuple(round(v, 1) for v in sorted(c.obb_extents, reverse=True))
                 for c in detected.clusters)
    assert got == [(0.7, 0.5, 0.5), (1.4, 0.9, 0.6), (1.8, 1.2, 0.4)]


def test_structure_is_removed_not_clustered(detected):
    d = detected.diagnostics
    assert d["floor_source"] == "pathplan_ransac"
    assert d["ceiling_found"] is True
    assert d["walls_found"] == 4                 # the room's four, and only those
    assert d["wall_candidates_rejected"] >= 1    # a bookcase face + an oblique phantom
    removed = d["removed_voxels"]
    assert removed["floor_and_below"] > 0 and removed["ceiling"] > 0 and removed["walls"] > 0
    # Whatever survives is a tiny fraction of the room — objects, not surfaces.
    assert d["object_voxels"] < 0.1 * d["occupied_voxels"]


def test_placement_facts_are_specific(detected):
    by_size = sorted(detected.clusters, key=lambda c: -max(c.obb_extents))
    shelf, table, chair = by_size[0], by_size[1], by_size[2]
    assert "against a wall" in shelf.placement
    # A table in open floor must NOT claim wall contact: wall planes are infinite and
    # the bounded-footprint test is the only thing stopping "against a wall" everywhere.
    assert "against a wall" not in table.placement
    assert "against a wall" not in chair.placement
    for c in detected.clusters:
        assert "on the floor" in c.placement


def test_sheets_are_rejected():
    """A slab that survives plane removal is leftover surface, not an object. Here a
    tilted sheet floats clear of the floor, so only the thickness filter can reject it."""
    rng = np.random.default_rng(5)
    x, y, z = np.eye(3)
    tilted = _plane(rng, 40_000, [2.5, 1.2, 0.0], (x + 0.25 * y) / np.linalg.norm(x + 0.25 * y), z, 1.2, 1.2)
    cloud = np.concatenate([
        _plane(rng, 120_000, [0, 0, 0], x, z, 5, 4),
        _box_surface(rng, 30_000, TABLE["center"], TABLE["size"]),
        tilted,
    ]).astype(np.float32)

    # Ceiling detection off: the sheet is the densest horizontal band in this fixture,
    # so it would be removed as a ceiling before the thickness filter ever sees it.
    res = imp.detect_objects(
        cloud, up=[0.0, 1.0, 0.0],
        params=imp.DetectParams(detect_walls=False, ceiling_peak_frac=1.01),
    )
    assert res.rejected["sheet"] >= 1
    assert len(res.clusters) == 1                      # the table, and only the table
    assert sorted(round(float(v), 1) for v in res.clusters[0].obb_extents) == [0.6, 0.9, 1.4]


def test_up_vector_is_estimated_when_not_supplied(room):
    res = imp.detect_objects(room)
    assert res.up_source == "heuristic"
    assert abs(float(np.asarray(res.up) @ np.array([0.0, 1.0, 0.0]))) > 0.99
    assert len(res.clusters) == 3


def test_yaw_obb_is_tight_on_a_symmetric_footprint():
    """The reason this module does not use ``pca_obb``: a symmetric footprint has no
    dominant in-plane axis, so PCA picks a diagonal and inflates the box."""
    rng = np.random.default_rng(1)
    pts = _box_surface(rng, 30_000, [0, 0.4, 0], [1.4, 0.8, 0.9])
    up = np.array([0.0, 1.0, 0.0])
    _, yaw_e, yaw_r = imp.yaw_obb(pts, up)
    _, pca_e, _ = sg.pca_obb(pts, up=up)
    assert float(np.prod(yaw_e)) <= float(np.prod(pca_e)) + 1e-9
    assert sorted(round(float(v), 2) for v in yaw_e) == [0.8, 0.9, 1.4]
    assert np.allclose(yaw_r.T @ yaw_r, np.eye(3), atol=1e-9)


def test_outliers_do_not_set_the_voxel_size(room):
    """One stray gaussian 100x away must not collapse the whole scene into six voxels."""
    strays = np.array([[900.0, 900.0, 900.0], [-900.0, -900.0, -900.0]], dtype=np.float32)
    res = imp.detect_objects(np.concatenate([room, strays]), up=[0.0, 1.0, 0.0])
    assert res.diagnostics["outliers_dropped"] == 2
    assert len(res.clusters) == 3


def test_geometry_label_states_measured_size_only():
    label = imp.geometry_label(3, [0.61, 1.10, 0.82])
    assert label == "object 3 — 1.10 × 0.82 × 0.61"
    assert "m" not in label.replace("object", "")  # never a metre glyph on an unanchored scan


def test_facts_objects_validate_against_the_persisted_schema(detected):
    objects = imp.to_facts_objects(detected)
    assert len(objects) == 3
    for raw in objects:
        obj = ObjectInstance.model_validate(raw)   # the SAME model a capture persists
        assert len(obj.center) == 3 and len(obj.extent) == 3
        assert obj.confidence == 0.0               # nothing scored these; never invent one
        assert obj.evidence == []
        det = obj.imported_detection
        assert det is not None
        assert det.method == "geometry_cluster"
        assert det.label_source == "geometry"
        assert det.units == "relative"
        assert det.obb is not None and len(det.obb.rotation) == 3
        assert any("No detector ran" in c for c in det.caveats)
    # The AABB really does contain the OBB, so consumers that ignore rotation are correct.
    for raw in objects:
        det = raw["imported_detection"]
        half = np.asarray(raw["extent"]) / 2.0 + 1e-6
        corners = _obb_corners(det["obb"])
        assert np.all(np.abs(corners - np.asarray(raw["center"])) <= half)


def _obb_corners(obb):
    c = np.asarray(obb["center"], float)
    e = np.asarray(obb["extents"], float)
    R = np.asarray(obb["rotation"], float)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return c + (signs * e / 2.0) @ R.T


def test_run_document_carries_the_staleness_stamps(detected):
    doc = imp.run_document(
        detected,
        scan_id="scanX",
        run_id="r1",
        created_at="2026-07-31T00:00:00+00:00",
        parent_artifact="cloud.npz",
        label_mode="geometry",
    )
    for key in ("scan_id", "run_id", "created_at", "parent_artifact"):
        assert doc[key]
    assert doc["generated"] is False
    assert doc["object_count"] == 3
    assert doc["units"] == "relative"
    json.dumps(doc)  # must be JSON-serializable as persisted


# ---------------------------------------------------------------------------
# synthetic views (contract Interface 1) — the projection round-trip
# ---------------------------------------------------------------------------


def _view(view_id="v0", index=0, position=(0.0, 0.0, 5.0), quat=(0.0, 0.0, 0.0, 1.0)):
    return imp.SyntheticView(
        view_id=view_id, index=index, position=position, quaternion=quat,
        fov_y_deg=60.0, width=320, height=240,
    )


def test_known_point_lands_on_the_right_pixel():
    """The contract's highest-risk line, pinned. An identity-quaternion three.js camera
    at +5z looks down -z with +y up, so the origin lands on the principal point, world
    +x is screen-right and world +y is screen-UP (smaller v)."""
    view = _view()
    intr = imp.view_intrinsics(view)
    c2w = imp.view_c2w(view, "opengl")
    uv, z = imp.project_points(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), c2w, intr
    )
    assert np.all(z > 0)
    assert np.allclose(uv[0], [view.width / 2, view.height / 2])
    assert uv[1][0] > uv[0][0] and np.isclose(uv[1][1], uv[0][1])
    assert uv[2][1] < uv[0][1] and np.isclose(uv[2][0], uv[0][0])


def test_opencv_convention_is_the_mirror_image():
    """Reading the wrong basis flips the image about the principal point AND puts the
    scene behind the camera — the silent mislanding the convention probe exists to stop."""
    view = _view()
    probe = np.array([[1.0, 1.0, 0.0]])
    uv_gl, z_gl = imp.project_points(probe, imp.view_c2w(view, "opengl"), imp.view_intrinsics(view))
    uv_cv, z_cv = imp.project_points(probe, imp.view_c2w(view, "opencv"), imp.view_intrinsics(view))
    assert z_gl[0] > 0 and z_cv[0] < 0
    # The y-flip and the z-flip cancel in v, so only u mirrors — an object crops from
    # the wrong side of the frame while looking perfectly plausible.
    cx, cy = view.width / 2, view.height / 2
    assert np.isclose(uv_gl[0][0] - cx, -(uv_cv[0][0] - cx))
    assert np.isclose(uv_gl[0][1], uv_cv[0][1])


def test_stored_c2w_wins_over_re_deriving_it():
    """`synthetic_views.py` converts the posted three.js pose to an OpenCV c2w once and
    stores it. Consuming that instead of redoing the conversion is the whole point of it
    being on the record — and the two must agree, or one of them is wrong."""
    view = _view()
    derived = imp.view_c2w(view, "opengl")
    stored = imp.SyntheticView.from_wire({
        "view_id": "sv0", "index": 0, "position": list(view.position),
        "quaternion": list(view.quaternion), "fov_y_deg": view.fov_y_deg,
        "width": view.width, "height": view.height,
        "c2w": [[float(v) for v in row] for row in derived],
        "intrinsics": [111.0, 111.0, 160.0, 120.0],
    })
    assert np.allclose(imp.view_c2w(stored, "opencv"), derived)   # convention ignored
    assert np.allclose(imp.view_intrinsics(stored), [111.0, 111.0, 160.0, 120.0])
    # Nothing left to probe once the writer's own conversion is on the record.
    convention, _ = imp.pick_view_convention(np.zeros((4, 3)), [stored])
    assert convention == "stored"


def test_view_convention_is_measured_not_assumed(room):
    """Views authored by a three.js camera must be read as three.js cameras."""
    convention, seen = imp.pick_view_convention(room, [_view(position=(0.0, 1.5, 12.0))])
    assert convention == "opengl"
    assert seen > imp.VIEW_SANITY_MIN_INSIDE_FRAC


def test_labels_refuse_when_views_do_not_see_the_scene(detected):
    """A view pointing into empty space must not produce confident crops."""
    # 90 degrees of yaw at 400 units puts the scene off-axis under BOTH conventions,
    # which is the only situation where neither reading is usable.
    away = _view(position=(0.0, 1.5, 400.0), quat=(0.0, 0.7071068, 0.0, 0.7071068))
    summary = imp.label_clusters_from_views(detected, [away], lambda v: b"", _FakeVLM("desk"))
    assert summary["labelled"] == 0
    assert "do not see this scene" in summary["note"]
    assert all(c.label_source == "geometry" for c in detected.clusters)


class _FakeVLM:
    """Mock OpenRouter client — same ``chat_json`` shape, no network."""

    def __init__(self, label, confidence=0.8, model="fake/vlm"):
        self.label = label
        self.confidence = confidence
        self.model = model
        self.calls = []

    def chat_json(self, *, system_prompt, user_prompt, images_b64, temperature, max_tokens):
        self.calls.append({"prompt": user_prompt, "images": len(images_b64 or [])})
        payload = {"label": self.label, "confidence": self.confidence}
        return payload, types.SimpleNamespace(content="", model=self.model, degraded=False)


def _png(width=320, height=240):
    buf = io.BytesIO()
    PIL_Image.new("RGB", (width, height), (120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def test_view_labels_upgrade_the_geometry_label(room):
    result = imp.detect_objects(room, up=[0.0, 1.0, 0.0])
    vlm = _FakeVLM("office chair")
    views = [_view(position=(0.0, 1.5, 12.0)), _view("v1", 1, position=(12.0, 1.5, 0.0),
                                                    quat=(0.0, 0.7071068, 0.0, 0.7071068))]
    summary = imp.label_clusters_from_views(result, views, lambda v: _png(), vlm)

    assert summary["labelled"] == 3
    assert summary["camera_convention"] == "opengl"
    assert vlm.calls and vlm.calls[0]["images"] == 1
    for c in result.clusters:
        assert c.label == "office chair"
        assert c.label_source == "synthetic_view"
        assert c.label_model == "fake/vlm"
        assert c.label_view_id in ("v0", "v1")

    objects = imp.to_facts_objects(result)
    for raw in objects:
        det = raw["imported_detection"]
        assert det["provenance"] == imp.PROVENANCE_VIEW
        assert det["label_source"] == "synthetic_view"
        # The measured-size label survives the upgrade — the size claim is still ours.
        assert det["geometry_label"].startswith("object ")
        assert any("not from a photograph" in c for c in det["caveats"])


def test_declined_labels_keep_the_geometry_label(room):
    result = imp.detect_objects(room, up=[0.0, 1.0, 0.0])
    before = [c.label for c in result.clusters]
    summary = imp.label_clusters_from_views(
        result, [_view(position=(0.0, 1.5, 12.0))], lambda v: _png(), _FakeVLM("unknown")
    )
    assert summary["labelled"] == 0
    assert summary["declined"] == 3
    assert [c.label for c in result.clusters] == before
    assert all(c.label_source == "geometry" for c in result.clusters)


def test_label_errors_never_abort_the_run(room):
    result = imp.detect_objects(room, up=[0.0, 1.0, 0.0])

    def boom(view):
        raise RuntimeError("blob store down")

    summary = imp.label_clusters_from_views(
        result, [_view(position=(0.0, 1.5, 12.0))], boom, _FakeVLM("desk")
    )
    assert summary["errors"] == 3 and summary["labelled"] == 0
    assert all(c.label_source == "geometry" for c in result.clusters)


def test_crop_is_padded_and_clipped_to_the_frame():
    crop_b64 = imp.crop_jpeg_b64(_png(), [10.0, 10.0, 40.0, 30.0], 320, 240)
    raw = base64.b64decode(crop_b64)
    with PIL_Image.open(io.BytesIO(raw)) as img:
        assert img.format == "JPEG"
        assert img.width > 30 and img.height > 20   # padding was applied
        assert img.width <= 320 and img.height <= 240


# ---------------------------------------------------------------------------
# store guard
# ---------------------------------------------------------------------------


def test_store_refuses_to_overwrite_captured_detections(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene("u", "captured", {"summary": ""},
                     {"objects": [{"query": "desk", "center": [0, 0, 0], "extent": [1, 1, 1]}]})
    with pytest.raises(ValueError, match="came from a capture"):
        store.replace_geometry_objects("u", "captured", [])
    assert store.get_scene("u", "captured")["facts"]["objects"][0]["query"] == "desk"


def test_store_replaces_its_own_geometry_objects(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene("u", "imported", {"summary": ""}, {"objects": []}, source="imported_splat")
    geo = [{"query": "object 1 — 1.00 × 1.00 × 1.00", "center": [0, 0, 0], "extent": [1, 1, 1],
            "confidence": 0.0, "evidence": [], "imported_detection": {"method": "geometry_cluster"}}]
    assert store.replace_geometry_objects("u", "imported", geo) == {"replaced": 0, "count": 1}
    assert store.replace_geometry_objects("u", "imported", geo * 2) == {"replaced": 1, "count": 2}
    # The picker's count follows, so an imported scan stops advertising zero objects.
    assert store.list_scenes("u")[0]["object_count"] == 2


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def imported_app(monkeypatch, tmp_path, room):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    colors = np.full((len(room), 3), 128, dtype=np.uint8)
    store.save_scene("user-a", "imported", {"summary": "imported", "room_type": "unknown"},
                     {"objects": [], "metrics": {"vertical_axis_known": False}},
                     keyframes_b64=None, points=(room, colors), source="imported_splat")
    store.save_scene("user-a", "captured", {"summary": "captured"},
                     {"objects": [{"query": "desk", "center": [0, 0, 0], "extent": [1, 1, 1]}]},
                     points=(room, colors), source="recon_video")

    routes = sys.modules["server.oreos.routes_sam3d"]
    routes._cloud_cache.clear()
    # The fixture reloads server.oreos, so the routes hold a DIFFERENT module object than
    # this file's top-level `imp`. Hand back the reloaded one — patching the wrong copy
    # is silent and the test just quietly exercises the unpatched path.
    reloaded = importlib.import_module("server.oreos.imported_objects")
    yield app.test_client(), store, routes, reloaded
    routes.configure_synthetic_view_loader(None)
    routes._cloud_cache.clear()


def test_get_reports_none_before_a_run(imported_app):
    client, _store, _routes, _imp = imported_app
    body = client.get("/api/scenes/imported/imported_objects").get_json()
    assert body["status"] == "none" and body["object_count"] == 0
    assert body["eligible"] is True


def test_post_detects_and_persists_in_the_captured_schema(imported_app):
    client, store, _routes, _imp = imported_app
    resp = client.post("/api/scenes/imported/imported_objects", json={"labels": "geometry"})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["count"] == 3
    assert body["provenance"] == imp.PROVENANCE_GEOMETRY
    assert body["labels"]["note"] == "geometry-only requested"
    assert body["derived_key"].startswith("derived/demo/objects/")

    persisted = store.get_scene("user-a", "imported")["facts"]["objects"]
    assert len(persisted) == 3
    # The Objects tab reads facts.objects; parity with a capture is the whole point.
    for obj in persisted:
        ObjectInstance.model_validate(obj)
        assert obj["imported_detection"]["provenance"] == imp.PROVENANCE_GEOMETRY
    run = json.loads(store.get_derived_artifact("user-a", "imported", routes_run_key()))
    assert run["object_count"] == 3
    assert run["cloud"]["source_points"] == len(store.get_cloud("user-a", "imported")[0])


def routes_run_key():
    return sys.modules["server.oreos.routes_sam3d"].IMPORTED_RUN_KEY


def test_get_returns_the_run_after_a_post(imported_app):
    client, _store, _routes, _imp = imported_app
    client.post("/api/scenes/imported/imported_objects", json={"labels": "geometry"})
    body = client.get("/api/scenes/imported/imported_objects").get_json()
    assert body["status"] == "ready"
    assert body["run"]["object_count"] == 3
    assert body["run"]["diagnostics"]["walls_found"] == 4


def test_rerun_replaces_rather_than_appends(imported_app):
    client, store, _routes, _imp = imported_app
    client.post("/api/scenes/imported/imported_objects", json={"labels": "geometry"})
    second = client.post("/api/scenes/imported/imported_objects", json={"labels": "geometry"})
    assert second.status_code == 200
    assert second.get_json()["replaced"] == 3
    assert len(store.get_scene("user-a", "imported")["facts"]["objects"]) == 3


def test_post_refuses_a_scan_with_real_detections(imported_app):
    client, store, _routes, _imp = imported_app
    resp = client.post("/api/scenes/captured/imported_objects", json={})
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "has_detections"
    assert store.get_scene("user-a", "captured")["facts"]["objects"][0]["query"] == "desk"


def test_post_422s_without_geometry(imported_app, tmp_path):
    client, store, _routes, _imp = imported_app
    store.save_scene("user-a", "bare", {"summary": ""}, {"objects": []}, source="imported_splat")
    resp = client.post("/api/scenes/bare/imported_objects", json={})
    assert resp.status_code == 422 and resp.get_json()["error"] == "no_geometry"


def test_post_rejects_an_unknown_label_mode(imported_app):
    client, _store, _routes, _imp = imported_app
    resp = client.post("/api/scenes/imported/imported_objects", json={"labels": "vibes"})
    assert resp.status_code == 400


def test_params_are_overridable_and_recorded(imported_app):
    client, _store, _routes, _imp = imported_app
    body = client.post(
        "/api/scenes/imported/imported_objects",
        json={"labels": "geometry", "max_objects": 2, "min_voxels": 40},
    ).get_json()
    assert body["count"] == 2
    params = body["diagnostics"]["params"]
    assert params["max_objects"] == 2 and params["min_voxels"] == 40


def test_auto_labels_degrade_honestly_without_views(imported_app):
    client, _store, _routes, _imp = imported_app
    body = client.post("/api/scenes/imported/imported_objects", json={"labels": "auto"}).get_json()
    assert "no synthetic views registered" in body["labels"]["note"]
    assert all(o["imported_detection"]["label_source"] == "geometry" for o in body["objects"])


def test_synthetic_views_on_the_record_drive_the_label_pass(imported_app, monkeypatch):
    """Codes to contract Interface 1: A's views ride on the scene record, bytes come
    through the injectable loader. Proves the seam without A's endpoint existing."""
    client, store, routes, live_imp = imported_app
    record = store.get_scene("user-a", "imported")
    record["synthetic_views"] = [{
        "view_id": "sv0", "index": 0, "position": [0.0, 1.5, 12.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0], "fov_y_deg": 60.0, "width": 320, "height": 240,
        "derived_key": "derived/demo/views/sv0.png",
    }]
    store._store["user-a:imported"] = record
    routes.configure_synthetic_view_loader(lambda *a: _png())
    monkeypatch.setattr(live_imp, "make_label_client", lambda: _FakeVLM("bookcase"))

    body = client.post("/api/scenes/imported/imported_objects", json={"labels": "auto"}).get_json()
    assert body["labels"]["labelled"] == 3
    assert body["labels"]["camera_convention"] == "opengl"
    for obj in body["objects"]:
        assert obj["query"] == "bookcase"
        det = obj["imported_detection"]
        assert det["label_source"] == "synthetic_view"
        assert det["view_id"] == "sv0"
        assert det["provenance"] == imp.PROVENANCE_VIEW


def test_ground_frame_up_is_used_when_workstream_a_has_run(imported_app):
    """B reads A's up axis when it exists and stands alone when it does not."""
    client, store, _routes, _imp = imported_app
    record = store.get_scene("user-a", "imported")
    record["facts"]["metrics"] = {"vertical_axis_known": True, "up_axis": [0.0, 1.0, 0.0]}
    store._store["user-a:imported"] = record
    body = client.post("/api/scenes/imported/imported_objects", json={"labels": "geometry"}).get_json()
    assert body["up_source"] == "provided"
    assert body["up"] == [0.0, 1.0, 0.0]
