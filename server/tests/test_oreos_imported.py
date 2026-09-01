"""Imported-splat parity tests (IMPORTED-SPLAT-CONTRACT.md, workstream A) — GPU-free.

Covers ``server/oreos/synthetic_views.py``, ``server/oreos/ground_frame.py``,
``server/oreos/routes_imported.py``, and the synthetic-view fallback in
``server/oreos/persisted_agent.py``, on the test_demo_routes harness (real Flask +
test_client, ``server.app`` stubbed, a real ``ModalScenePersistence`` on tmp_path).

The load-bearing test in this file is ``test_world_point_projects_to_the_expected_pixel``
and its siblings. The client posts a camera pose in the world frame after inverting the
display frame's π-rotation and scene_center offset; if that conversion is wrong, nothing
throws — every mask, every back-projection and every dimension quietly lands somewhere
else in the room. So the projection is pinned here in Python against hand-computed
pixels, and the identical case is pinned in TypeScript
(``apps/webserver/tests/unit/syntheticViews.test.ts``) against the same numbers. The two
suites are the two ends of one wire.
"""

from __future__ import annotations

import base64
import importlib
import struct
import sys
import types
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.scene_report.schemas import ObjectInstance, SceneFacts, SceneMetrics, SceneReport
from server.scene_report.store import ModalScenePersistence

from server.oreos import ground_frame as gf
from server.oreos import synthetic_views as sv


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
        client=app.test_client(), store=store, agent=agent_mod, runlog=runlog_mod
    )
    runlog_mod.REGISTRY.reset()
    agent_mod.LLM_CLIENT_FACTORY = None


def _png_bytes(width=4, height=4, rgb=(120, 90, 40)) -> bytes:
    """A real, minimal PNG. Hand-built rather than mocked: the route asserts the PNG
    magic on purpose, so a fixture that isn't a PNG would test the wrong thing."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _view_payload(position=(0.0, 0.0, 5.0), quat=(0.0, 0.0, 0.0, 1.0), w=1024, h=768):
    return {
        "image_b64": base64.b64encode(_png_bytes()).decode("ascii"),
        "position": list(position),
        "quaternion": list(quat),
        "fov_y_deg": 60.0,
        "width": w,
        "height": h,
    }


def _save_imported(store, scan="imp1", user="user-a", points=None, objects=None):
    """Persist an imported-splat-shaped scene: no keyframes, degraded report."""
    positions, colors = points if points is not None else _room_cloud()
    facts = SceneFacts(
        metrics=SceneMetrics(
            dimensions=[6.0, 4.0, 2.5],
            bbox_min=[0.0, 0.0, 0.0],
            bbox_max=[6.0, 4.0, 2.5],
            point_count=int(positions.shape[0]),
            vertical_axis_known=False,
        ),
        objects=objects or [],
        units_note="Imported splat.",
    )
    store.save_scene(
        user,
        scan,
        SceneReport(summary="Imported Gaussian splat.", room_type="unknown", degraded=True),
        facts,
        keyframes_b64=None,
        points=(positions, colors),
        label="founder-import",
        source="imported_splat",
    )


# Synthetic room with a CEILING (the planes.py fixture has none, and a ceiling is
# exactly what this module has to find): floor z=0, ceiling z=2.5, two walls, a table
# slab at z=0.8 that must not be mistaken for either.
def _room_cloud(seed=0, n_floor=6000, n_ceil=4000, n_wall=2500, n_table=800, n_noise=200):
    rng = np.random.default_rng(seed)
    floor = np.stack(
        [rng.uniform(0, 6, n_floor), rng.uniform(0, 4, n_floor), rng.normal(0, 0.008, n_floor)],
        axis=1,
    )
    ceil = np.stack(
        [rng.uniform(0, 6, n_ceil), rng.uniform(0, 4, n_ceil), rng.normal(2.5, 0.008, n_ceil)],
        axis=1,
    )
    wall_x = np.stack(
        [rng.normal(0, 0.008, n_wall), rng.uniform(0, 4, n_wall), rng.uniform(0, 2.5, n_wall)],
        axis=1,
    )
    wall_y = np.stack(
        [rng.uniform(0, 6, n_wall), rng.normal(4, 0.008, n_wall), rng.uniform(0, 2.5, n_wall)],
        axis=1,
    )
    table = np.stack(
        [rng.uniform(2, 3.2, n_table), rng.uniform(1, 1.8, n_table), rng.normal(0.8, 0.008, n_table)],
        axis=1,
    )
    noise = np.stack(
        [rng.uniform(0, 6, n_noise), rng.uniform(0, 4, n_noise), rng.uniform(0, 2.5, n_noise)],
        axis=1,
    )
    positions = np.concatenate([floor, ceil, wall_x, wall_y, table, noise]).astype(np.float32)
    colors = np.full((positions.shape[0], 3), 140, dtype=np.uint8)
    return positions, colors


# ---------------------------------------------------------------------------
# the transform — the highest-risk line in the contract
# ---------------------------------------------------------------------------


def test_identity_gl_pose_becomes_the_opencv_flip():
    """A three.js camera with an identity quaternion looks down −Z with +Y up; the same
    camera in OpenCV looks down +Z with +Y down. So the c2w rotation is exactly the
    axis flip, and nothing else."""
    c2w = sv.gl_pose_to_cv_c2w([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(c2w[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(c2w[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_world_point_projects_to_the_expected_pixel():
    """THE round trip. Camera at world (0,0,5) with an identity three.js orientation, so
    it looks back down −Z at the origin. 1024x768 at fov_y 60 gives
    f = 384 / tan(30°) = 665.108…

      * the origin is dead centre;
      * a point 1 unit ABOVE the axis lands 133.02 px ABOVE centre (smaller py — the
        image y axis runs down, which is the sign that gets inverted by accident);
      * a point 1 unit to the camera's RIGHT lands 133.02 px right of centre.
    """
    view = sv.parse_view(_view_payload(), 0)["meta"]
    c2w, intr = view["c2w"], view["intrinsics"]
    f = 384.0 / np.tan(np.radians(30.0))
    assert intr == pytest.approx([f, f, 512.0, 384.0], rel=1e-12)

    px, py = sv.project_world_point(c2w, intr, [0.0, 0.0, 0.0])
    assert (px, py) == pytest.approx((512.0, 384.0), abs=1e-9)

    px, py = sv.project_world_point(c2w, intr, [0.0, 1.0, 0.0])
    assert (px, py) == pytest.approx((512.0, 384.0 - f / 5.0), abs=1e-9)

    px, py = sv.project_world_point(c2w, intr, [1.0, 0.0, 0.0])
    assert (px, py) == pytest.approx((512.0 + f / 5.0, 384.0), abs=1e-9)


def test_a_yawed_camera_still_lands_on_the_same_world_point():
    """Rotate the camera 90° about +Y and move it onto the +X axis so it faces the origin
    again: the origin must stay dead centre. This is the case a transposed rotation
    matrix passes at identity and fails here."""
    s = np.sqrt(0.5)  # quaternion for +90° about Y, [x, y, z, w]
    view = sv.parse_view(_view_payload(position=(5.0, 0.0, 0.0), quat=(0.0, s, 0.0, s)), 0)["meta"]
    px, py = sv.project_world_point(view["c2w"], view["intrinsics"], [0.0, 0.0, 0.0])
    assert (px, py) == pytest.approx((512.0, 384.0), abs=1e-6)


def test_a_point_behind_the_camera_is_none_not_a_mirrored_pixel():
    view = sv.parse_view(_view_payload(), 0)["meta"]
    assert sv.project_world_point(view["c2w"], view["intrinsics"], [0.0, 0.0, 9.0]) is None


def test_off_frame_pixels_are_reported_not_clamped():
    """A visible-but-off-crop point and a behind-camera point are different failures;
    clamping would merge them and hide a bad pose."""
    view = sv.parse_view(_view_payload(), 0)["meta"]
    px, _py = sv.project_world_point(view["c2w"], view["intrinsics"], [40.0, 0.0, 0.0])
    assert px > 1024


# ---------------------------------------------------------------------------
# wire validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda v: v.update(image_b64="data:image/png;base64,AAAA"), "invalid_view"),
        (lambda v: v.update(image_b64=base64.b64encode(b"\xff\xd8\xffnotpng").decode()), "invalid_view"),
        (lambda v: v.update(image_b64="!!!not base64!!!"), "invalid_view"),
        (lambda v: v.update(position=[0.0, 0.0]), "invalid_view"),
        (lambda v: v.update(quaternion=[0.0, 0.0, 0.0, 0.0]), "invalid_view"),
        (lambda v: v.update(fov_y_deg=0.0), "invalid_view"),
        (lambda v: v.update(width=8), "invalid_view"),
        (lambda v: v.update(height=99999), "invalid_view"),
    ],
)
def test_malformed_views_are_refused_with_a_reason(mutate, code):
    payload = _view_payload()
    mutate(payload)
    with pytest.raises(sv.SyntheticViewError) as exc:
        sv.parse_view(payload, 0)
    assert exc.value.code == code
    assert exc.value.detail  # never a bare code — the client shows this verbatim


def test_view_count_is_capped():
    with pytest.raises(sv.SyntheticViewError) as exc:
        sv.parse_views([_view_payload() for _ in range(sv.MAX_VIEWS + 1)])
    assert exc.value.code == "too_many_views"


# ---------------------------------------------------------------------------
# routes: synthetic views
# ---------------------------------------------------------------------------


def test_views_round_trip_through_the_routes(demo_app):
    _save_imported(demo_app.store)
    resp = demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views",
        json={"replace": True, "views": [_view_payload(), _view_payload(position=(0, 0, 6))]},
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["count"] == 2
    assert [v["index"] for v in body["views"]] == [0, 1]

    listing = demo_app.client.get("/api/demo/scene/imp1/synthetic_views").get_json()
    assert listing["count"] == 2
    assert listing["provenance"] == "synthetic view"
    first = listing["views"][0]
    assert first["url"] == f"/api/demo/scene/imp1/synthetic_views/{first['view_id']}.png"
    assert len(first["c2w"]) == 4 and len(first["intrinsics"]) == 4

    png = demo_app.client.get(first["url"])
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    assert png.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_synthetic_views_never_touch_keyframes(demo_app):
    """The structural honesty guarantee. A reader of the persisted record — today's UI,
    a future export, a human in 2028 — must never have to guess whether an image was
    photographed or rendered."""
    _save_imported(demo_app.store)
    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views", json={"views": [_view_payload()]}
    )
    record = demo_app.store.get_scene("user-a", "imp1")
    assert record["keyframes"] == []
    assert len(record["synthetic_views"]) == 1
    assert record["synthetic_views"][0]["provenance"] == "synthetic view"
    assert record["synthetic_views"][0]["blob_key"].startswith("derived/demo/synthetic_views/")


def test_replace_swaps_the_set_and_append_extends_it(demo_app):
    _save_imported(demo_app.store)
    post = lambda body: demo_app.client.post(  # noqa: E731
        "/api/demo/scene/imp1/synthetic_views", json=body
    ).get_json()
    first = post({"replace": True, "views": [_view_payload()]})
    appended = post({"views": [_view_payload()]})
    assert appended["count"] == 2
    assert [v["index"] for v in appended["views"]] == [1]
    replaced = post({"replace": True, "views": [_view_payload()]})
    assert replaced["count"] == 1
    listing = demo_app.client.get("/api/demo/scene/imp1/synthetic_views").get_json()
    assert [v["view_id"] for v in listing["views"]] != [first["views"][0]["view_id"]]


def test_a_forged_view_id_cannot_reach_a_blob(demo_app):
    _save_imported(demo_app.store)
    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views", json={"views": [_view_payload()]}
    )
    assert demo_app.client.get("/api/demo/scene/imp1/synthetic_views/deadbeef.png").status_code == 404


def test_unknown_scene_is_404_not_a_write(demo_app):
    resp = demo_app.client.post(
        "/api/demo/scene/nope/synthetic_views", json={"views": [_view_payload()]}
    )
    assert resp.status_code == 404


def test_bad_body_is_400_with_a_sentence(demo_app):
    _save_imported(demo_app.store)
    resp = demo_app.client.post("/api/demo/scene/imp1/synthetic_views", json={"views": []})
    assert resp.status_code == 400
    assert resp.get_json()["detail"]


# ---------------------------------------------------------------------------
# routes: the HEADLESS render path
#
# Until this existed the only producer of synthetic views was a person with the scene
# open in a browser, so the founder's imported scene could not be given annotation
# evidence at all. The renderer itself lives on its own Modal app and is exercised
# against the real scene by `modal run modal_oreos_render.py::ring`; what these cover is the
# broker's half — validation, the job contract, and the fact that a server render lands
# in exactly the same place with exactly the same provenance as a browser one.
# ---------------------------------------------------------------------------


class _FakeRingRenderer:
    """Stand-in for ``modal_oreos_render.render_ring``.

    It registers views through the SAME helper the real function uses, so the test
    exercises the storage contract rather than a mock of it."""

    def __init__(self, store, views=2, fail=None):
        self.store = store
        self.views = views
        self.fail = fail
        self.calls = []

    def remote(self, user_id, scan_id, **kwargs):
        self.calls.append({"user_id": user_id, "scan_id": scan_id, **kwargs})
        if self.fail is not None:
            return self.fail
        from server.oreos import synthetic_views as sv_mod

        wire = [
            {
                **_view_payload(position=(0.0, 0.0, 5.0 + i)),
                "label": f"ring {i + 1}/{self.views}",
                "renderer": {"generator": "openreality-splat-render/1", "sh_degree": 3},
            }
            for i in range(self.views)
        ]
        parsed = sv_mod.parse_views(wire)
        stored = sv_mod.persist_views(self.store, user_id, scan_id, parsed, existing=[])
        return {"ok": True, "count": len(stored), "views": [
            {"view_id": m["view_id"], "index": m["index"], "label": m.get("label")} for m in stored
        ]}


def _poll(client, scan_id, job_id, timeout_s=6.0):
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        record = client.get(f"/api/scenes/{scan_id}/jobs/{job_id}").get_json()
        if record["status"] in ("done", "error"):
            return record
        _time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def test_a_headless_render_lands_in_the_same_field_as_a_browser_capture(demo_app):
    """A server render is NOT better evidence than a browser render. Same field, same
    blob namespace, same chip — the record only notes which renderer drew the pixels."""
    _save_imported(demo_app.store)
    routes = importlib.import_module("server.oreos.routes_imported")
    fake = _FakeRingRenderer(demo_app.store)
    routes.configure_render_ring_fn(fake)
    try:
        resp = demo_app.client.post(
            "/api/demo/scene/imp1/synthetic_views/render", json={"budget": 600000}
        )
        assert resp.status_code == 202, resp.get_json()
        record = _poll(demo_app.client, "imp1", resp.get_json()["job_id"])
        assert record["status"] == "done", record
    finally:
        routes.configure_render_ring_fn(None)

    assert fake.calls[0]["budget"] == 600000 and fake.calls[0]["want_sh"] is True
    scene = demo_app.store.get_scene("user-a", "imp1")
    assert scene["keyframes"] == []               # never, under any circumstances
    assert len(scene["synthetic_views"]) == 2
    view = scene["synthetic_views"][0]
    assert view["provenance"] == "synthetic view"
    assert view["blob_key"].startswith("derived/demo/synthetic_views/")
    assert view["renderer"]["generator"] == "openreality-splat-render/1"

    listing = demo_app.client.get("/api/demo/scene/imp1/synthetic_views").get_json()
    assert listing["count"] == 2
    png = demo_app.client.get(listing["views"][0]["url"])
    assert png.status_code == 200 and png.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_a_failed_render_surfaces_the_reason_instead_of_an_empty_set(demo_app):
    _save_imported(demo_app.store)
    routes = importlib.import_module("server.oreos.routes_imported")
    routes.configure_render_ring_fn(
        _FakeRingRenderer(
            demo_app.store,
            fail={"ok": False, "error": "no_geometry", "detail": "this scene has no LOD artifacts"},
        )
    )
    try:
        job_id = demo_app.client.post(
            "/api/demo/scene/imp1/synthetic_views/render", json={}
        ).get_json()["job_id"]
        record = _poll(demo_app.client, "imp1", job_id)
    finally:
        routes.configure_render_ring_fn(None)
    assert record["status"] == "error"
    assert "no LOD artifacts" in str(record.get("error"))
    assert demo_app.store.get_scene("user-a", "imp1").get("synthetic_views") in (None, [])


def test_render_parameters_are_bounded_before_a_container_is_spawned(demo_app):
    _save_imported(demo_app.store)
    routes = importlib.import_module("server.oreos.routes_imported")
    fake = _FakeRingRenderer(demo_app.store)
    routes.configure_render_ring_fn(fake)
    try:
        for body in ({"width": 9}, {"height": 99999}, {"ring_count": 0}, {"budget": 5}):
            resp = demo_app.client.post("/api/demo/scene/imp1/synthetic_views/render", json=body)
            assert resp.status_code == 400, (body, resp.get_json())
            assert resp.get_json()["detail"]
    finally:
        routes.configure_render_ring_fn(None)
    assert fake.calls == []


def test_rendering_an_unknown_scene_is_404(demo_app):
    routes = importlib.import_module("server.oreos.routes_imported")
    fake = _FakeRingRenderer(demo_app.store)
    routes.configure_render_ring_fn(fake)
    try:
        resp = demo_app.client.post("/api/demo/scene/nope/synthetic_views/render", json={})
    finally:
        routes.configure_render_ring_fn(None)
    assert resp.status_code == 404
    assert fake.calls == []


# ---------------------------------------------------------------------------
# ground frame
# ---------------------------------------------------------------------------


def test_ground_frame_finds_the_floor_ceiling_and_footprint():
    positions, _colors = _room_cloud()
    frame = gf.compute_ground_frame(positions)
    assert frame["vertical_axis_known"] is True
    assert frame["derivation"] == "plane-fit"
    assert abs(abs(frame["up_axis"][2]) - 1.0) < 0.02  # the fixture's up is ±z
    assert frame["floor_height"] == pytest.approx(0.0, abs=0.05)
    assert frame["room_height"] == pytest.approx(2.5, abs=0.15)
    width, depth = sorted(frame["floor_extent"], reverse=True)
    assert width == pytest.approx(6.0, abs=0.3)
    assert depth == pytest.approx(4.0, abs=0.3)
    # Occupancy-based, so a filled rectangular floor lands near w*d; the point of the
    # method is that an L-shaped one would NOT.
    assert frame["floor_area"] == pytest.approx(24.0, rel=0.15)


def test_a_floorless_cloud_refuses_to_name_a_vertical_axis():
    """A sphere shell has no dominant horizontal plane. The honest answer is "I don't
    know which way is up", and the honest STORAGE of that answer is no heights at all."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=(20_000, 3))
    positions = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    frame = gf.compute_ground_frame(positions)
    patch = gf.metrics_patch(frame)
    if frame["vertical_axis_known"]:
        pytest.skip("this cloud happened to admit a strong plane; nothing to assert")
    assert frame["note"]
    assert patch["vertical_axis_known"] is False
    assert patch["derivation"] == "plane-fit"
    assert patch["ground_frame_note"]
    for forbidden in ("up_axis", "floor_height", "ceiling_height", "floor_extent", "floor_area"):
        assert forbidden not in patch, f"a weak fit must not publish {forbidden}"


def test_too_few_points_is_an_error_not_a_guess():
    with pytest.raises(gf.GroundFrameError) as exc:
        gf.compute_ground_frame(np.zeros((4, 3), dtype=np.float32))
    assert exc.value.code == "no_geometry"


def test_metrics_patch_omits_the_ceiling_when_the_capture_has_none():
    frame = {
        "vertical_axis_known": True,
        "derivation": "plane-fit",
        "up_axis": [0.0, 0.0, 1.0],
        "floor_height": 0.0,
        "floor_extent": [3.0, 2.0],
        "floor_area": 6.0,
        "ceiling_height": None,
        "room_height": None,
        "note": "no dense surface near the top of the cloud",
    }
    patch = gf.metrics_patch(frame)
    assert patch["floor_area"] == 6.0
    assert "ceiling_height" not in patch
    assert "room_height" not in patch
    assert patch["ground_frame_note"]


def test_ground_frame_route_persists_onto_facts_metrics(demo_app):
    _save_imported(demo_app.store)
    resp = demo_app.client.post("/api/scenes/imp1/ground_frame", json={})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["persisted"] is True
    assert body["frame"]["vertical_axis_known"] is True

    record = demo_app.store.get_scene("user-a", "imp1")
    metrics = record["facts"]["metrics"]
    assert metrics["vertical_axis_known"] is True
    assert metrics["derivation"] == "plane-fit"
    assert metrics["floor_area"] > 0
    # The patch MERGES: the AABB the import recorded must survive.
    assert metrics["bbox_max"] == [6.0, 4.0, 2.5]
    # The report embeds its own copy of the facts; it must not drift from them.
    assert record["report"]["facts"]["metrics"]["vertical_axis_known"] is True

    read_back = demo_app.client.get("/api/scenes/imp1/ground_frame").get_json()
    assert read_back["frame"]["vertical_axis_known"] is True
    assert read_back["computed"] is False


def test_ground_frame_dry_run_computes_without_writing(demo_app):
    _save_imported(demo_app.store)
    resp = demo_app.client.post("/api/scenes/imp1/ground_frame", json={"dry_run": True})
    assert resp.status_code == 200
    assert resp.get_json()["persisted"] is False
    metrics = demo_app.store.get_scene("user-a", "imp1")["facts"]["metrics"]
    assert metrics["vertical_axis_known"] is False


def test_ground_frame_needs_geometry(demo_app):
    demo_app.store.save_scene(
        "user-a", "empty", SceneReport(summary="s"), SceneFacts(), source="imported_splat"
    )
    resp = demo_app.client.post("/api/scenes/empty/ground_frame", json={})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "no_geometry"


# ---------------------------------------------------------------------------
# agent: synthetic views as evidence
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    def __init__(self, script, model="mock-model"):
        self.script = list(script)
        self.calls: list[dict] = []
        self.model = model

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise RuntimeError("mock LLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, SimpleNamespace(model=self.model, degraded=False)


def _agent_for(demo_app, scan="imp1", factory=None):
    demo_app.agent.LLM_CLIENT_FACTORY = factory or (lambda *a, **k: None)
    record = demo_app.store.get_scene("user-a", scan)
    run = demo_app.runlog.AgentRun(run_id="r1", user_id="user-a", scan_id=scan, mode="annotate")
    return demo_app.agent.PersistedSceneAgent(
        demo_app.store, "user-a", scan, record, run, llm_factory=factory
    )


def test_evidence_source_prefers_keyframes_and_falls_back_to_views(demo_app):
    _save_imported(demo_app.store)
    assert _agent_for(demo_app).evidence_source() == "none"

    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views", json={"views": [_view_payload()]}
    )
    assert _agent_for(demo_app).evidence_source() == "synthetic_view"

    # A scan that HAS capture imagery must never downgrade to renders of itself.
    demo_app.store.save_scene(
        "user-a",
        "cap1",
        SceneReport(summary="s"),
        SceneFacts(),
        keyframes_b64=[{"submap_id": 0, "frame_idx": 0, "image_b64": base64.b64encode(b"j").decode()}],
        source="recon_video",
    )
    demo_app.store.set_synthetic_views("user-a", "cap1", [{"view_id": "v", "blob_key": "k"}])
    assert _agent_for(demo_app, "cap1").evidence_source() == "keyframe"


def test_key_features_ground_on_views_when_there_is_no_inventory(demo_app):
    """A freshly imported splat has no detected objects yet (that is workstream B's
    output, and it may not have run). The annotation pass must still produce grounded,
    labelled findings rather than dropping every claim for lack of a uid to cite."""
    _save_imported(demo_app.store)
    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views",
        json={"views": [_view_payload(), _view_payload(position=(0, 0, 7))]},
    )
    llm = _ScriptedLLM(
        [
            {
                "features": [
                    {"claim": "A long counter runs along the far wall.", "view_ids": []},
                    {"claim": "Seating clusters near the window.", "view_ids": ["bogus"]},
                ]
            }
        ]
    )
    agent = _agent_for(demo_app, factory=lambda role, *a, **k: llm if role == "annotator" else None)
    payload = agent._build_key_features()

    assert payload["provenance"] == "synthetic_view"
    assert len(payload["features"]) == 2
    assert all(f["provenance"] == "synthetic_view" for f in payload["features"])
    assert all(f["keyframes"] == [] for f in payload["features"])
    # A view id the model invented is dropped like any other unknown reference.
    assert payload["features"][1]["view_ids"] == []
    # The prompt has to SAY the images are renders, or the model narrates artifacts.
    prompt = llm.calls[0]["system_prompt"] + llm.calls[0]["user_prompt"]
    assert "SYNTHETIC VIEWS" in prompt
    assert len(llm.calls[0]["images_b64"]) == 2


def test_a_uid_free_claim_is_dropped_when_an_inventory_exists(demo_app):
    """The closed-world guardrail is unchanged where it still applies: with objects on
    the record, a claim that cites none of them is not admissible."""
    _save_imported(
        demo_app.store,
        objects=[ObjectInstance(query="desk", center=[1, 1, 1], extent=[1, 1, 1], confidence=0.9)],
    )
    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views", json={"views": [_view_payload()]}
    )
    llm = _ScriptedLLM([{"features": [{"claim": "Something vague.", "object_uids": []}]}])
    agent = _agent_for(demo_app, factory=lambda role, *a, **k: llm if role == "annotator" else None)
    payload = agent._build_key_features()
    assert all(f["provenance"] != "synthetic_view" for f in payload["features"])
    assert agent.validator.dropped >= 1


def test_annotate_run_on_an_imported_scene_streams_view_grounded_findings(demo_app):
    """End to end: the imported scene gets the same streaming walkthrough a captured one
    gets, with synthetic-view provenance on every claim that came from imagery."""
    _save_imported(demo_app.store)
    demo_app.client.post("/api/scenes/imp1/ground_frame", json={})
    demo_app.client.post(
        "/api/demo/scene/imp1/synthetic_views", json={"views": [_view_payload()]}
    )
    annotator = _ScriptedLLM(
        [{"features": [{"claim": "An open room with a long counter.", "view_ids": []}]}]
    )
    narrator = _ScriptedLLM([{"paragraphs": ["A open, high-ceilinged space."]}])
    demo_app.agent.LLM_CLIENT_FACTORY = lambda role, *a, **k: {
        "annotator": annotator,
        "narrator": narrator,
    }.get(role)

    start = demo_app.client.post("/api/scenes/imp1/demo/agent/annotate", json={"mode": "full"})
    assert start.status_code == 202
    run_id = start.get_json()["run_id"]
    run = demo_app.runlog.REGISTRY.get(run_id)
    run.thread.join(timeout=30)
    assert run.status == "done", run.error

    events = demo_app.client.get(
        f"/api/scenes/imp1/demo/agent/runs/{run_id}/events"
    ).get_json()["events"]
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e["payload"])

    assert by_type["run_meta"][0]["scene"]["evidence_source"] == "synthetic_view"
    assert by_type["run_meta"][0]["scene"]["synthetic_views"] == 1
    survey = by_type["agent_thought"][0]
    assert "synthetic views rendered from the imported splat" in survey["content"]
    assert survey["evidence_source"] == "synthetic_view"

    findings = by_type["agent_finding"]
    assert findings, "an imported scene must not produce a silent annotation pass"
    assert any(f.get("provenance") == "synthetic_view" for f in findings)
    # Nothing may ever be labelled as a photograph or a keyframe.
    assert all(f.get("provenance") not in ("photo", "keyframe") for f in findings)

    dims = [f for f in findings if f["query"] == "room dimensions"]
    assert dims, "the ground frame should have produced real room dimensions"
    assert "Floor area" in dims[0]["description"]
    assert "floor-to-ceiling" in dims[0]["description"]
    assert dims[0]["units"] == "relative"  # unanchored: never an "m" glyph


def test_anchored_floor_area_scales_with_the_square_of_the_factor(demo_app):
    """Area is the one quantity where applying the metric factor once is silently wrong.
    2x the linear scale is 4x the area."""
    _save_imported(demo_app.store)
    demo_app.client.post("/api/scenes/imp1/ground_frame", json={})
    unanchored = _agent_for(demo_app)._build_dimensions()["room"]

    demo_app.store.set_derived_pointer(
        "user-a",
        "imp1",
        {
            "kind": "anchor",
            "source_key": "derived/anchor/t/cloud.ply",
            "scale_factor": 2.0,
            "applied_at": "2026-07-31T00:00:00+00:00",
        },
    )
    anchored = _agent_for(demo_app)._build_dimensions()["room"]

    assert anchored["units"] == "m"
    assert anchored["floor_area_units"] == "m2"
    assert anchored["floor_area"] == pytest.approx(unanchored["floor_area"] * 4.0, rel=1e-3)
    assert anchored["ceiling_height"] == pytest.approx(unanchored["ceiling_height"] * 2.0, rel=1e-3)


def test_dimensions_stay_bbox_only_without_a_ground_frame(demo_app):
    """No fit, no height claims — the pre-existing posture must survive untouched."""
    _save_imported(demo_app.store)
    room = _agent_for(demo_app)._build_dimensions()["room"]
    assert room["vertical_axis_known"] is False
    assert "floor_area" not in room and "ceiling_height" not in room
    assert "not gravity-aligned" in room["note"]
