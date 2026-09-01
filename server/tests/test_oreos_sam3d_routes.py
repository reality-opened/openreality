"""W3 routes_sam3d tests — GPU-free, no network.

Harness = test_demo_routes.py pattern: fresh ``server.oreos`` package, REAL
Flask + test_client, stubbed ``server.app`` (auth + a real
``ModalScenePersistence`` on tmp_path). The modal completion function and the
fal HTTP session are injected fakes; keyless behavior is exercised by deleting
``FAL_API_KEY``.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import sys
import time
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("scipy")
PIL_Image = pytest.importorskip("PIL.Image")

from server.export.mask_rle import mask_to_rle
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# synthetic scene: a compact object cluster in front of a camera at the origin
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 32, 16
INTR = [100.0, 100.0, 16.0, 8.0]
DET_CENTER = [0.0, 0.0, 2.0]
DET_EXTENT = [0.25, 0.25, 0.25]


def _cluster_points(rng, n=600):
    """Object cluster (inside the detection box) + background sheet behind it."""
    obj = rng.uniform(-0.1, 0.1, size=(n, 3)) + np.asarray(DET_CENTER)
    bg = rng.uniform(-0.6, 0.6, size=(n, 3)) + np.asarray([0.0, 0.0, 4.0])
    return np.concatenate([obj, bg]).astype(np.float32)


def _evidence_mask():
    m = np.zeros((IMG_H, IMG_W), dtype=bool)
    m[2:14, 8:24] = True  # covers the cluster's projection (u in [11,21], v in [3,13])
    return m


def _tiny_jpeg_b64():
    img = PIL_Image.new("RGB", (IMG_W, IMG_H), (90, 120, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tiny_asset_ply(extent=0.2, n=7):
    """Binary INRIA-style PLY: a cube of gaussian centers (all-float props)."""
    xs = np.linspace(-extent / 2, extent / 2, n)
    g = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(g)}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    ).encode("ascii")
    body = np.zeros((len(g), len(props)), dtype="<f4")
    body[:, :3] = g
    return header + body.tobytes()


def _facts():
    return {
        "objects": [
            {
                "query": "crate",
                "human_label": "wooden crate",
                "center": DET_CENTER,
                "extent": DET_EXTENT,
                "confidence": 0.5,
                "dismissed": False,
                "evidence": [
                    {
                        "submap_id": 0,
                        "frame_idx": 1,
                        "box_2d": [8.0, 2.0, 24.0, 14.0],
                        "mask_rle": mask_to_rle(_evidence_mask()),
                    }
                ],
            },
            {
                "query": "far shelf",
                "human_label": None,
                "center": [4.0, 4.0, 4.0],
                "extent": [0.2, 0.2, 0.2],
                "confidence": 0.3,
                "dismissed": False,
                "evidence": [
                    {  # evidence frame that is NOT persisted as a keyframe
                        "submap_id": 9,
                        "frame_idx": 9,
                        "box_2d": [0.0, 0.0, 4.0, 4.0],
                        "mask_rle": mask_to_rle(np.ones((IMG_H, IMG_W), dtype=bool)),
                    }
                ],
            },
            {
                # Real geometry (the same cluster as det:0) but NO persisted evidence
                # keyframe — the exact shape every object on an imported splat has, and
                # the only one that reaches the server-rendered-view path. det:1 cannot:
                # it is out in empty space and is refused earlier, as "too sparse".
                "query": "shelf unit",
                "human_label": "shelf unit",
                "center": DET_CENTER,
                "extent": DET_EXTENT,
                "confidence": 0.4,
                "dismissed": False,
                "evidence": [{"submap_id": 7, "frame_idx": 7}],
            },
        ]
    }


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeCompleteFn:
    """Stand-in for the modal Function: returns a tiny cube asset."""

    def __init__(self, gate=None):
        self.calls = []
        self.gate = gate  # optional threading.Event to block on (409 test)

    def remote(self, image_png=None, mask_png=None, seed=42):
        self.calls.append({"image": len(image_png or b""), "mask": len(mask_png or b""), "seed": seed})
        if self.gate is not None:
            self.gate.wait(timeout=5)
        return {
            "asset_ply": _tiny_asset_ply(),
            "asset_glb": b"glTF-fake",
            "meta": {"seconds": 9.5, "n_gaussians": 343},
        }


class _Resp:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = json.dumps(self._payload) if payload else ""

    def json(self):
        return self._payload


class _FakeFalSession:
    """Queue-API fake mirroring fal's REAL recorded behaviour (verified 2026-07-31).

    The load-bearing detail: fal's submit body echoes ``status_url``/``response_url``
    rooted at the OWNER/APP pair only, DROPPING any endpoint subpath — a submit to
    ``…/fal-ai/sam-3/image`` polls at ``…/fal-ai/sam-3/requests/<id>/status``. This
    fake reproduces that and then 405s any URL that still carries the subpath, so a
    regression back to URL-construction fails the suite instead of production.
    """

    def __init__(self, result, artifacts=None):
        self.result = result
        self.artifacts = artifacts or {}
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        app_id = url.split("queue.fal.run/", 1)[-1]
        root = "/".join([p for p in app_id.split("/") if p][:2])
        base = f"https://queue.fal.run/{root}/requests/req-1"
        return _Resp(
            200,
            {
                "status": "IN_QUEUE",
                "request_id": "req-1",
                "response_url": base,
                "status_url": f"{base}/status",
                "cancel_url": f"{base}/cancel",
            },
        )

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        if url in self.artifacts:
            return _Resp(200, content=self.artifacts[url])
        if "/requests/" in url:
            app_path = url.split("queue.fal.run/", 1)[-1].split("/requests/", 1)[0]
            if len([p for p in app_path.split("/") if p]) > 2:
                # what the real API does to a constructed subpath polling URL
                return _Resp(405, None)
        if url.endswith("/status"):
            return _Resp(200, {"status": "COMPLETED", "request_id": "req-1"})
        return _Resp(200, self.result)


def _sam3_result(mask_png_bytes, url="https://v3b.fal.media/files/b/fake/mask.png"):
    """The real ``fal-ai/sam-3/image`` output envelope."""
    return {
        "image": {"url": url, "width": IMG_W, "height": IMG_H, "content_type": "image/png"},
        "masks": [{"url": url, "width": IMG_W, "height": IMG_H, "content_type": "image/png"}],
        "metadata": [{"index": 0, "score": 0.988, "box": None}],
        "scores": [0.988],
        "boxes": [None],
    }


def _trellis_result(url="https://v3b.fal.media/files/b/fake/model.glb"):
    """The real ``fal-ai/trellis`` output envelope — a GLB, no gaussian key."""
    return {
        "model_mesh": {
            "url": url,
            "content_type": "application/octet-stream",
            "file_name": "model.glb",
            "file_size": 689344,
        },
        "timings": {"prepare": 3.6e-05, "generation": 3.63, "export": 19.97},
    }


def _glb_bytes(extent=0.4):
    """A real (tiny) binary GLB, built offline with trimesh — no network."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(extent, extent, extent))
    return mesh.export(file_type="glb")


# ---------------------------------------------------------------------------
# fixture: fresh demo package on a real Flask app + seeded scene
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def sam3d_app(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    rng = np.random.default_rng(3)
    pts = _cluster_points(rng)
    colors = np.full((len(pts), 3), 128, dtype=np.uint8)
    store.save_scene(
        "user-a",
        "scan1",
        {"summary": "s", "room_type": "office"},
        _facts(),
        keyframes_b64=[{"submap_id": 0, "frame_idx": 1, "image_b64": _tiny_jpeg_b64()}],
        points=(pts, colors),
        splat_bytes=b"ply\nfake",
        source="recon_video",
    )

    routes = sys.modules["server.oreos.routes_sam3d"]
    routes._cloud_cache.clear()
    fal = sys.modules["server.oreos.genai.fal_client"]
    yield app.test_client(), store, routes, fal
    routes.configure_complete_fn(None)
    routes.configure_render_object_fn(None)
    fal.configure_session(None)


def _poll_job(client, scan_id, job_id, timeout_s=6.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/scenes/{scan_id}/jobs/{job_id}")
        assert resp.status_code == 200, resp.get_json()
        record = resp.get_json()
        if record["status"] in ("done", "error"):
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


# ---------------------------------------------------------------------------
# segment — path (a)
# ---------------------------------------------------------------------------


def test_segment_object_ref_detection_box(sam3d_app):
    client, store, routes, _fal = sam3d_app
    resp = client.post("/api/scenes/scan1/segment", json={"object_ref": {"uid": "det:0"}})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["uid"].startswith("sel:")
    assert body["prompt_source"] == "detection_keyframe"
    assert body["method"] == "detection_box"  # no frames index on this scene
    assert body["label"] == "wooden crate"
    assert body["n_points"] >= 50
    obb = body["obb"]
    assert len(obb["center"]) == 3 and len(obb["extents"]) == 3
    # refined OBB reflects the 0.2-span cluster (PCA axes may rotate, so the
    # bound is the cluster diagonal), centered on it — not the raw det box.
    assert max(obb["extents"]) <= 0.2 * np.sqrt(3) + 1e-6
    assert np.allclose(obb["center"], DET_CENTER, atol=0.05)
    assert body["mask_key"].startswith("derived/demo/objects/")
    assert body["evidence"]["keyframe_blob_key"] == "0_1.jpg"
    assert body["evidence"]["mask_rle"]["size"] == [IMG_H, IMG_W]
    # persisted artifacts
    ukey = body["uid"].replace(":", "_")
    assert store.get_derived_artifact("user-a", "scan1", f"derived/demo/objects/{ukey}/select.json")
    assert store.get_derived_artifact("user-a", "scan1", body["mask_key"])
    index = json.loads(
        store.get_derived_artifact("user-a", "scan1", "derived/demo/objects/index.json")
    )
    assert body["uid"] in index["objects"]
    manifest = json.loads(
        store.get_derived_artifact("user-a", "scan1", "derived/demo/manifest.json")
    )
    assert any(ukey in key for key in manifest["variations"])


def test_segment_point_world_resolves_detection(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/segment", json={"point_world": [0.05, -0.02, 2.02]})
    assert resp.status_code == 200
    assert resp.get_json()["det_index"] == 0


def test_segment_point_world_miss_is_keyless_503(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/segment", json={"point_world": [50.0, 50.0, 50.0]})
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "fal_key_missing"


def test_segment_mask_lift_with_frames_index(sam3d_app):
    client, store, *_ = sam3d_app
    frames = [
        {
            "blob_key": "0_1.jpg",
            "submap_id": 0,
            "frame_idx": 1,
            "c2w": np.eye(4).tolist(),
            "intrinsics": INTR,
        }
    ]
    store.save_derived_artifact(
        "user-a", "scan1", "demo/frames_index.json", json.dumps(frames).encode()
    )
    resp = client.post("/api/scenes/scan1/segment", json={"object_ref": {"uid": "det:0"}})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["method"] == "mask_lift"
    assert body["quality"] == "good"
    # z-buffer culled the background sheet: OBB depth stays near the cluster's 0.2
    assert body["obb"]["extents"][2] < 0.5


def test_segment_bad_request_shapes(sam3d_app):
    client, *_ = sam3d_app
    assert client.post("/api/scenes/scan1/segment", json={}).status_code == 422
    assert (
        client.post("/api/scenes/scan1/segment", json={"object_ref": {"uid": "layer:x"}}).status_code
        == 422
    )
    assert (
        client.post("/api/scenes/scan1/segment", json={"point_world": [1, 2]}).status_code == 422
    )
    assert client.post("/api/scenes/missing/segment", json={}).status_code == 404


# ---------------------------------------------------------------------------
# segment — path (c)
# ---------------------------------------------------------------------------


def _view_body():
    img = PIL_Image.new("RGB", (IMG_W, IMG_H), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "view": {
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "c2w": np.eye(4).tolist(),
            "intrinsics": INTR,
        },
        "point_px": [16, 8],
    }


def test_segment_view_keyless_503(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/segment", json=_view_body())
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "fal_key_missing"


def _mask_png_bytes():
    buf = io.BytesIO()
    PIL_Image.fromarray((_evidence_mask() * 255).astype(np.uint8), mode="L").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def test_segment_view_with_key_runs_sam_and_lifts(sam3d_app, monkeypatch):
    """Path (c) against the REAL SAM 3 response envelope (masks[] of signed URLs)."""
    client, store, routes, fal = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    url = "https://v3b.fal.media/files/b/fake/mask.png"
    session = _FakeFalSession(_sam3_result(None, url=url), artifacts={url: _mask_png_bytes()})
    fal.configure_session(session)

    resp = client.post("/api/scenes/scan1/segment", json=_view_body())
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["prompt_source"] == "rendered_view"
    assert body["n_points"] >= 50
    ukey = body["uid"].replace(":", "_")
    assert store.get_derived_artifact("user-a", "scan1", f"derived/demo/objects/{ukey}/view.png")
    assert store.get_derived_artifact("user-a", "scan1", body["mask_key"])
    assert any("rendered view" in c for c in body["caveats"])


def test_segment_view_sends_the_verified_sam3_payload(sam3d_app, monkeypatch):
    """Locks the request shape fal actually accepts (this is what was wrong)."""
    client, _store, _routes, fal = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    url = "https://v3b.fal.media/files/b/fake/mask.png"
    session = _FakeFalSession(_sam3_result(None, url=url), artifacts={url: _mask_png_bytes()})
    fal.configure_session(session)

    assert client.post("/api/scenes/scan1/segment", json=_view_body()).status_code == 200

    submit_url, payload = session.posts[0]
    # the endpoint carries the /image subpath ...
    assert submit_url == "https://queue.fal.run/fal-ai/sam-3/image"
    # ... but polling drops it (a constructed subpath URL 405s in the fake)
    assert any(u.startswith("https://queue.fal.run/fal-ai/sam-3/requests/") for u in session.gets)
    assert not any("/sam-3/image/requests/" in u for u in session.gets)

    assert payload["image_url"].startswith("data:image/png;base64,")
    # typed pixel-coordinate arrays, NOT the old prompts:[{type:"point"}] guess
    assert "prompts" not in payload
    assert payload["point_prompts"] == [{"x": 16, "y": 8, "label": 1}]
    assert all(isinstance(p["x"], int) and isinstance(p["y"], int) for p in payload["point_prompts"])
    # the empty prompt is what suppresses SAM 3's server-side default of "wheel"
    assert payload["prompt"] == ""
    assert payload["apply_mask"] is False


def test_segment_view_box_prompt_payload(sam3d_app, monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    _client, _store, _routes, fal = sam3d_app
    session = _FakeFalSession(_sam3_result(None), artifacts={})
    fal.configure_session(session)
    fal.segment_sam3("data:image/png;base64,AAA", box_px=(10.4, 20.6, 3.2, 40.0))
    _url, payload = session.posts[0]
    assert "point_prompts" not in payload
    # integer pixel corners, normalised to min/max regardless of argument order
    assert payload["box_prompts"] == [{"x_min": 3, "y_min": 21, "x_max": 10, "y_max": 40}]


def test_segment_view_surfaces_fal_account_error(sam3d_app, monkeypatch):
    """A billing/key rejection must reach the client with fal's own words."""
    client, _store, _routes, fal = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")

    class _Locked(_FakeFalSession):
        def post(self, url, json=None, headers=None, timeout=None):
            self.posts.append((url, json))
            return _Resp(403, {"detail": "User is locked. Reason: Exhausted balance."})

    fal.configure_session(_Locked({}))
    resp = client.post("/api/scenes/scan1/segment", json=_view_body())
    assert resp.status_code == 502
    body = resp.get_json()
    assert body["error"] == "fal_account_error"
    assert body["fal_status"] == 403
    assert "Exhausted balance" in body["message"]


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------


def test_complete_det_end_to_end(sam3d_app):
    client, store, routes, _fal = sam3d_app
    fake = _FakeCompleteFn()
    routes.configure_complete_fn(fake)

    resp = client.post("/api/scenes/scan1/objects/det:0/complete", json={"seed": 7})
    assert resp.status_code == 202, resp.get_json()
    job_id = resp.get_json()["job_id"]
    record = _poll_job(client, "scan1", job_id)
    assert record["status"] == "done", record
    result = record["result"]
    assert result["quality"] in ("good", "usable")
    assert result["asset_key"] == "derived/demo/objects/det_0/completed/asset.ply"
    assert result["glb_key"] == "derived/demo/objects/det_0/completed/asset.glb"
    assert np.asarray(result["transform"]).shape == (4, 4)
    assert fake.calls and fake.calls[0]["seed"] == 7
    assert fake.calls[0]["image"] > 0 and fake.calls[0]["mask"] > 0

    meta = json.loads(
        store.get_derived_artifact(
            "user-a", "scan1", "derived/demo/objects/det_0/completed/meta.json"
        )
    )
    assert meta["generated"] is True and meta["generator"] == "sam3d"
    assert meta["provenance"] == "AI-completed (SAM 3D)"
    assert any("unseen parts are invented" in c for c in meta["caveats"])
    assert meta["gate"]["tier"] == result["quality"]
    assert meta["timings"]["gpu_seconds"] == 9.5
    assert store.get_derived_artifact("user-a", "scan1", result["asset_key"])

    index = json.loads(
        store.get_derived_artifact("user-a", "scan1", "derived/demo/objects/index.json")
    )
    assert index["objects"]["det:0"]["completed_key"] == result["asset_key"]


def test_complete_conflict_409_while_running(sam3d_app):
    import threading

    client, _store, routes, _fal = sam3d_app
    gate = threading.Event()
    routes.configure_complete_fn(_FakeCompleteFn(gate=gate))
    first = client.post("/api/scenes/scan1/objects/det:0/complete", json={})
    assert first.status_code == 202
    second = client.post("/api/scenes/scan1/objects/det:0/complete", json={})
    assert second.status_code == 409
    assert second.get_json()["job_id"] == first.get_json()["job_id"]
    gate.set()
    _poll_job(client, "scan1", first.get_json()["job_id"])


def test_complete_unpersisted_evidence_keyframe_422(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/objects/det:1/complete", json={})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_completable"


# ---------------------------------------------------------------------------
# complete — the server-rendered-view path (no evidence keyframe exists at all)
# ---------------------------------------------------------------------------


def _render_png(w=IMG_W, h=IMG_H, colour=(70, 90, 110)):
    buf = io.BytesIO()
    PIL_Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _render_mask_png():
    buf = io.BytesIO()
    PIL_Image.fromarray((_evidence_mask() * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


class _FakeRenderObjectFn:
    """Stand-in for ``modal_oreos_render.render_object_view``."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def remote(self, user_id, scan_id, obb, **kwargs):
        self.calls.append({"user_id": user_id, "scan_id": scan_id, "obb": obb, **kwargs})
        if self.result is not None:
            return self.result
        return {
            "ok": True,
            "image_png": _render_png(),
            "mask_png": _render_mask_png(),
            "view": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "fov_y_deg": 55.0,
                     "width": IMG_W, "height": IMG_H, "c2w": np.eye(4).tolist(), "intrinsics": INTR},
            "chosen": {"visible_pixels": 4321, "visible_fraction": 0.92},
            "source": {"artifact": "derived/demo/lod/full.spz", "sh_degree": 3},
            "provenance": "synthetic view",
            "caveats": ["Rendered on the server by our own rasterizer"],
        }


def test_completion_falls_back_to_a_server_render_when_no_keyframe_exists(sam3d_app):
    """The whole point. An imported splat has no capture video, so no object has an
    evidence keyframe and completion used to refuse every one of them."""
    client, store, routes, _fal = sam3d_app
    render = _FakeRenderObjectFn()
    routes.configure_render_object_fn(render)
    routes.configure_complete_fn(_FakeCompleteFn())

    resp = client.post("/api/scenes/scan1/objects/det:2/complete", json={})
    assert resp.status_code == 202, resp.get_json()
    record = _poll_job(client, "scan1", resp.get_json()["job_id"])
    assert record["status"] == "done", record

    assert render.calls and render.calls[0]["scan_id"] == "scan1"
    # the renderer is handed OUR selection OBB, in the contract's convention
    obb = render.calls[0]["obb"]
    assert set(obb) >= {"center", "extents", "rotation"}

    meta = json.loads(
        store.get_derived_artifact(
            "user-a", "scan1", "derived/demo/objects/det_2/completed/meta.json"
        )
    )
    assert meta["inputs"]["prompt_source"] == "server_rendered_view"
    assert meta["inputs"]["render"]["sh_degree"] == 3
    assert meta["inputs"]["evidence_blob"] is None
    # provenance survives onto the asset: this was never a photograph
    assert any("synthetic view" in c for c in meta["caveats"])
    # and the exact image the model was shown is persisted, so it stays reviewable
    assert store.get_derived_artifact("user-a", "scan1", meta["inputs"]["prompt_view_key"])


def test_an_object_nothing_can_see_is_refused_with_the_renderers_own_reason(sam3d_app):
    """Degrade with a reason, never fabricate. A completion built on a mask over the wall
    in front of the object would be invention wearing measurement's clothes."""
    client, _store, routes, _fal = sam3d_app
    routes.configure_render_object_fn(
        _FakeRenderObjectFn(
            {
                "ok": False,
                "error": "object_not_visible",
                "detail": "the best of 3 rendered viewpoints shows only 12 unoccluded pixels",
                "diagnostics": [{"label": "object view 1/3", "visible_pixels": 12}],
            }
        )
    )
    resp = client.post("/api/scenes/scan1/objects/det:2/complete", json={})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "not_completable"
    assert "12 unoccluded pixels" in body["message"]
    assert body["render_error"] == "object_not_visible"
    assert body["render_diagnostics"]


def test_an_unreachable_renderer_refuses_and_names_the_app(sam3d_app):
    """No renderer deployed is a deployment fact, and the message has to say which app is
    missing — 'not completable' with no next action is how a demo stalls on camera."""

    class _Boom:
        def remote(self, *a, **kw):
            raise RuntimeError("app demo-splat-render not found")

    client, _store, routes, _fal = sam3d_app
    routes.configure_render_object_fn(_Boom())
    resp = client.post("/api/scenes/scan1/objects/det:2/complete", json={})
    assert resp.status_code == 422
    message = resp.get_json()["message"]
    assert "demo-splat-render" in message and "capture views from the viewer" in message


def test_a_too_sparse_object_is_refused_before_a_render_is_attempted(sam3d_app):
    """det:1 sits in empty space. Rendering a view of nothing costs a GPU container and
    tells us what the point cloud already did."""
    client, _store, routes, _fal = sam3d_app
    render = _FakeRenderObjectFn()
    routes.configure_render_object_fn(render)
    resp = client.post("/api/scenes/scan1/objects/det:1/complete", json={})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "not_completable"
    assert render.calls == []


def test_complete_unknown_uid_404(sam3d_app):
    client, *_ = sam3d_app
    assert client.post("/api/scenes/scan1/objects/det:99/complete", json={}).status_code == 404
    assert client.post("/api/scenes/scan1/objects/sel:none/complete", json={}).status_code == 404


def test_complete_sel_uid_after_segment(sam3d_app):
    client, _store, routes, _fal = sam3d_app
    routes.configure_complete_fn(_FakeCompleteFn())
    seg = client.post("/api/scenes/scan1/segment", json={"object_ref": {"uid": "det:0"}}).get_json()
    resp = client.post(f"/api/scenes/scan1/objects/{seg['uid']}/complete", json={})
    assert resp.status_code == 202
    record = _poll_job(client, "scan1", resp.get_json()["job_id"])
    assert record["status"] == "done"
    ukey = seg["uid"].replace(":", "_")
    assert record["result"]["asset_key"] == f"derived/demo/objects/{ukey}/completed/asset.ply"


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def test_variants_keyless_503(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/objects/det:0/variants", json={"mode": "image"})
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "fal_key_missing"


def test_variants_with_key_full_plumbing(sam3d_app, monkeypatch):
    """End-to-end against the REAL TRELLIS envelope: model_mesh -> GLB -> splat PLY."""
    pytest.importorskip("trimesh")
    client, store, routes, fal = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    glb_url = "https://v3b.fal.media/files/b/fake/model.glb"
    session = _FakeFalSession(_trellis_result(glb_url), artifacts={glb_url: _glb_bytes()})
    fal.configure_session(session)

    resp = client.post("/api/scenes/scan1/objects/det:0/variants", json={"mode": "image"})
    assert resp.status_code == 202, resp.get_json()
    record = _poll_job(client, "scan1", resp.get_json()["job_id"])
    assert record["status"] == "done", record
    result = record["result"]

    # the outbound payload fal actually accepts: image_url only, never a prompt
    _url, payload = session.posts[0]
    assert _url == "https://queue.fal.run/fal-ai/trellis"
    assert payload["image_url"].startswith("data:image/png;base64,")
    assert "prompt" not in payload
    assert payload["seed"] == 42

    # BOTH artifacts land: the generator's own GLB + the derived splat
    assert result["asset_key"] == f"derived/demo/objects/det_0/variants/{result['variant_id']}/asset.ply"
    assert result["glb_key"] == f"derived/demo/objects/det_0/variants/{result['variant_id']}/asset.glb"
    glb = store.get_derived_artifact("user-a", "scan1", result["glb_key"])
    assert glb[:4] == b"glTF"
    ply = store.get_derived_artifact("user-a", "scan1", result["asset_key"])
    assert ply[:3] == b"ply"
    assert result["n_gaussians"] > 1000

    meta = json.loads(store.get_derived_artifact("user-a", "scan1", result["meta_key"]))
    assert meta["generator"] == "trellis" and meta["generated"] is True
    assert meta["provenance"] == "AI-generated (TRELLIS)"
    assert meta["inputs"]["conditioning"] == "evidence_crop"
    assert meta["asset"]["source_format"] == "glb"
    assert meta["conversion"]["mesh_faces"] > 0
    assert meta["timings"]["generation"] == 3.63
    # the mesh-not-gaussians fact is stated, not hidden
    assert any("textured mesh" in c for c in meta["caveats"])

    index = json.loads(
        store.get_derived_artifact("user-a", "scan1", "derived/demo/objects/index.json")
    )
    assert result["variant_id"] in index["objects"]["det:0"]["variants"]


def test_variants_text_mode_is_rejected(sam3d_app, monkeypatch):
    """fal's TRELLIS has no text input — we refuse rather than silently ignore it."""
    client, *_ = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    resp = client.post(
        "/api/scenes/scan1/objects/det:0/variants", json={"mode": "text", "prompt": "red crate"}
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "text_mode_unsupported"


def test_variants_native_ply_still_supported(sam3d_app, monkeypatch):
    """If fal ever ships a native-gaussian output, take it without conversion."""
    client, store, routes, fal = sam3d_app
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    ply_url = "https://v3b.fal.media/files/b/fake/model.ply"
    fal.configure_session(
        _FakeFalSession(
            {"model_gaussian": {"url": ply_url}}, artifacts={ply_url: _tiny_asset_ply()}
        )
    )
    resp = client.post("/api/scenes/scan1/objects/det:0/variants", json={"mode": "image"})
    assert resp.status_code == 202, resp.get_json()
    record = _poll_job(client, "scan1", resp.get_json()["job_id"])
    assert record["status"] == "done", record
    meta = json.loads(store.get_derived_artifact("user-a", "scan1", record["result"]["meta_key"]))
    assert meta["asset"]["source_format"] == "ply"
    assert record["result"]["glb_key"] is None


def test_evidence_crop_does_not_composite_the_background(sam3d_app):
    """Measured on the real endpoint: masking the object onto ANY flat colour
    makes TRELLIS reconstruct that flat region as a slab. The crop must keep the
    original pixels and use the mask only to locate the box."""
    _client, _store, routes, _fal = sam3d_app
    img = PIL_Image.new("RGB", (40, 40), (17, 200, 90))  # distinctive background
    img.paste(PIL_Image.new("RGB", (10, 10), (250, 0, 0)), (15, 15))
    ibuf = io.BytesIO()
    img.save(ibuf, format="PNG")
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[15:25, 15:25] = 255
    mbuf = io.BytesIO()
    PIL_Image.fromarray(mask, mode="L").save(mbuf, format="PNG")

    out = routes._evidence_crop(ibuf.getvalue(), mbuf.getvalue(), pad=5)
    crop = np.asarray(PIL_Image.open(io.BytesIO(out)).convert("RGB"))
    assert crop.shape[:2] == (20, 20)  # mask bbox (10px) + 5px pad each side
    # the padding ring is the ORIGINAL background, not white/grey/transparent
    assert tuple(crop[0, 0]) == (17, 200, 90)
    assert tuple(crop[10, 10]) == (250, 0, 0)
    assert not (crop == 255).all(axis=2).any()


def test_variants_bad_mode_422(sam3d_app):
    client, *_ = sam3d_app
    resp = client.post("/api/scenes/scan1/objects/det:0/variants", json={"mode": "voxel"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# scene jobs route
# ---------------------------------------------------------------------------


def test_scene_job_unknown_404_and_wrong_scene(sam3d_app):
    client, _store, routes, _fal = sam3d_app
    assert client.get("/api/scenes/scan1/jobs/nope").status_code == 404
    routes.configure_complete_fn(_FakeCompleteFn())
    job_id = client.post("/api/scenes/scan1/objects/det:0/complete", json={}).get_json()["job_id"]
    # same job id under a different scene id -> identical 404 envelope
    assert client.get(f"/api/scenes/other/jobs/{job_id}").status_code == 404
    _poll_job(client, "scan1", job_id)
