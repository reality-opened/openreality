"""fal.ai client + GLB->splat conversion tests — GPU-free, NO NETWORK.

Every response body here is a *recording* of what fal really returned on
2026-07-31 (see ``server/oreos/genai/fal_client.py`` for the verified contracts).
The point of this file is to pin the handful of shapes the blind implementation
got wrong, so they cannot silently regress:

  * the SAM 3 endpoint carries a ``/image`` subpath but its polling URLs DO NOT
  * SAM 3 takes ``point_prompts``/``box_prompts``, not a tagged ``prompts`` list
  * SAM 3's ``prompt`` defaults to "wheel" server-side, so we send ""
  * TRELLIS returns ``model_mesh`` (a GLB) and has no text input at all
  * artifact URLs may be inline ``data:`` URIs under ``sync_mode``
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from server.oreos.genai import fal_client


class _Resp:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Records calls; replays queued responses."""

    def __init__(self, submit=None, statuses=None, result=None, artifacts=None):
        self.submit = submit
        self.statuses = list(statuses or [{"status": "COMPLETED"}])
        self.result = result if result is not None else {}
        self.artifacts = artifacts or {}
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        return _Resp(200, self.submit)

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        if url in self.artifacts:
            return _Resp(200, content=self.artifacts[url])
        if url.endswith("/status"):
            return _Resp(200, self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])
        return _Resp(200, self.result)


@pytest.fixture(autouse=True)
def _key_and_session(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    yield
    fal_client.configure_session(None)


# --------------------------------------------------------------------------
# queue transport
# --------------------------------------------------------------------------

REAL_SUBMIT = {
    "status": "IN_QUEUE",
    "request_id": "019fb903-22c6-78e0-aeb6-88730a7ce53a",
    # NOTE: no "/image" — this is exactly what fal returned for a submit to
    # https://queue.fal.run/fal-ai/sam-3/image
    "response_url": "https://queue.fal.run/fal-ai/sam-3/requests/019fb903-22c6-78e0-aeb6-88730a7ce53a",
    "status_url": "https://queue.fal.run/fal-ai/sam-3/requests/019fb903-22c6-78e0-aeb6-88730a7ce53a/status",
    "cancel_url": "https://queue.fal.run/fal-ai/sam-3/requests/019fb903-22c6-78e0-aeb6-88730a7ce53a/cancel",
    "queue_position": 0,
}


def test_wait_follows_the_urls_fal_returns_not_constructed_ones():
    """The bug that would have broken every SAM 3 call: a constructed
    .../sam-3/image/requests/<id>/status URL returns HTTP 405."""
    session = _Session(submit=REAL_SUBMIT, result={"masks": []})
    fal_client.configure_session(session)

    submitted = fal_client.submit("fal-ai/sam-3/image", {"image_url": "data:,x"})
    fal_client.wait(submitted, sleep=lambda _s: None)

    assert session.posts[0][0] == "https://queue.fal.run/fal-ai/sam-3/image"
    assert submitted["status_url"] == REAL_SUBMIT["status_url"]
    assert all("/sam-3/image/requests/" not in u for u in session.gets)


def test_submit_falls_back_to_owner_app_root_when_urls_absent():
    session = _Session(submit={"request_id": "req-1"})
    fal_client.configure_session(session)
    submitted = fal_client.submit("fal-ai/sam-3/image", {})
    assert submitted["status_url"] == "https://queue.fal.run/fal-ai/sam-3/requests/req-1/status"
    assert submitted["response_url"] == "https://queue.fal.run/fal-ai/sam-3/requests/req-1"


@pytest.mark.parametrize(
    "app_id,expected",
    [
        ("fal-ai/sam-3/image", "fal-ai/sam-3"),
        ("fal-ai/sam-3/video", "fal-ai/sam-3"),
        ("fal-ai/trellis", "fal-ai/trellis"),
        ("solo", "solo"),
    ],
)
def test_queue_root(app_id, expected):
    assert fal_client._queue_root(app_id) == expected


def test_auth_header_is_key_scheme():
    session = _Session(submit={"request_id": "r"})
    fal_client.configure_session(session)
    fal_client.submit("fal-ai/trellis", {})
    assert session.posts[0][2]["Authorization"] == "Key test-key"


def test_missing_key_raises_before_any_network(monkeypatch):
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    session = _Session(submit={"request_id": "r"})
    fal_client.configure_session(session)
    with pytest.raises(fal_client.FalKeyMissing):
        fal_client.run("fal-ai/trellis", {})
    assert session.posts == []


def test_exhausted_balance_403_becomes_a_specific_account_error():
    """Recorded live: fal answers 403 {"detail": "User is locked. Reason:
    Exhausted balance..."} — a founder billing action, not a model failure."""

    class _Locked(_Session):
        def post(self, url, json=None, headers=None, timeout=None):
            self.posts.append((url, json, headers))
            return _Resp(
                403,
                {"detail": "User is locked. Reason: Exhausted balance. Top up your balance at fal.ai/dashboard/billing."},
            )

    fal_client.configure_session(_Locked())
    with pytest.raises(fal_client.FalAccountError) as excinfo:
        fal_client.submit("fal-ai/trellis", {})
    assert excinfo.value.status == 403
    assert "Exhausted balance" in str(excinfo.value.detail)


def test_model_error_keeps_fal_detail_and_is_not_an_account_error():
    class _Boom(_Session):
        def post(self, url, json=None, headers=None, timeout=None):
            self.posts.append((url, json, headers))
            return _Resp(422, {"detail": [{"loc": ["body", "image_url"], "msg": "field required"}]})

    fal_client.configure_session(_Boom())
    with pytest.raises(fal_client.FalError) as excinfo:
        fal_client.submit("fal-ai/trellis", {})
    assert not isinstance(excinfo.value, fal_client.FalAccountError)
    assert excinfo.value.status == 422
    assert "field required" in str(excinfo.value)


def test_failed_status_raises_rather_than_returning_a_result():
    session = _Session(submit=REAL_SUBMIT, statuses=[{"status": "FAILED", "error": "oom"}])
    fal_client.configure_session(session)
    with pytest.raises(fal_client.FalError, match="FAILED"):
        fal_client.wait(fal_client.submit("fal-ai/trellis", {}), sleep=lambda _s: None)


def test_download_handles_data_uri_and_http():
    png = b"\x89PNG\r\n\x1a\n-fake"
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    session = _Session(artifacts={"https://v3b.fal.media/f/m.png": png})
    fal_client.configure_session(session)
    assert fal_client.download(uri) == png  # sync_mode inline
    assert session.gets == []  # no network for a data URI
    assert fal_client.download("https://v3b.fal.media/f/m.png") == png


# --------------------------------------------------------------------------
# payload shapes
# --------------------------------------------------------------------------


def test_sam3_default_app_is_the_image_subpath_endpoint():
    assert fal_client.DEFAULT_SAM3_APP == "fal-ai/sam-3/image"


def test_sam3_point_payload_matches_the_verified_schema():
    session = _Session(submit=REAL_SUBMIT, result={"masks": []})
    fal_client.configure_session(session)
    fal_client.segment_sam3("data:image/png;base64,AAA", point_px=(120.6, 44.2))
    _url, payload, _h = session.posts[0]
    assert payload["point_prompts"] == [{"x": 121, "y": 44, "label": 1}]
    assert payload["prompt"] == ""  # suppresses the server-side "wheel" default
    assert payload["apply_mask"] is False
    assert payload["include_scores"] is True
    assert "prompts" not in payload and "box_prompts" not in payload


def test_sam3_requires_a_prompt_geometry():
    with pytest.raises(ValueError):
        fal_client.segment_sam3("data:image/png;base64,AAA")


def test_trellis_never_sends_a_prompt_and_requires_an_image():
    session = _Session(
        submit={"request_id": "r", "status_url": "https://queue.fal.run/fal-ai/trellis/requests/r/status",
                "response_url": "https://queue.fal.run/fal-ai/trellis/requests/r"},
        result={"model_mesh": {"url": "u"}},
    )
    fal_client.configure_session(session)
    fal_client.trellis_generate(image_b64_data_uri="data:image/png;base64,AAA", seed=7)
    _url, payload, _h = session.posts[0]
    assert payload == {"image_url": "data:image/png;base64,AAA", "seed": 7}
    with pytest.raises(ValueError, match="image-to-3D only"):
        fal_client.trellis_generate(image_b64_data_uri="")


# --------------------------------------------------------------------------
# GLB -> splat PLY
# --------------------------------------------------------------------------


def test_glb_to_splat_ply_roundtrips_through_every_downstream_reader(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from server.oreos import segment_geometry as sg
    from server.oreos.genai import mesh_to_splat
    from server.scene_report.splat_io import read_splat_ply

    glb = trimesh.creation.box(extents=(0.4, 0.8, 0.4)).export(file_type="glb")
    ply, meta = mesh_to_splat.glb_to_splat_ply(glb, target_points=5000, seed=1)

    assert ply[:3] == b"ply"
    assert meta["source_format"] == "glb"
    assert meta["n_gaussians"] == 5000
    assert meta["mesh_faces"] == 12

    # 1. the fit/gate reader
    pts = sg.read_ply_positions(ply)
    assert pts.shape == (5000, 3)
    assert np.allclose(pts.max(axis=0), [0.2, 0.4, 0.2], atol=0.02)

    # 2. the product-workflow splat reader
    path = tmp_path / "asset.ply"
    path.write_bytes(ply)
    fields = read_splat_ply(str(path))
    for name in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"):
        assert name in fields
    assert np.allclose(fields["rot_0"], 1.0)
    assert float(fields["opacity"][0]) > 0  # logit(0.99)

    # 3. it actually fits to an OBB and passes the gate
    fit = sg.fit_asset_to_obb(pts, [1.0, 2.0, 3.0], [0.4, 0.8, 0.4], np.eye(3).tolist())
    placed = sg.apply_transform(pts, fit["transform"])
    gate = sg.quality_gate(placed, [1.0, 2.0, 3.0], [0.4, 0.8, 0.4], np.eye(3).tolist())
    assert gate["tier"] in ("good", "usable")


def test_glb_conversion_rejects_a_non_glb_artifact():
    from server.oreos.genai import mesh_to_splat

    with pytest.raises(mesh_to_splat.MeshConversionError, match="not a binary GLB"):
        mesh_to_splat.glb_to_splat_ply(b"ply\nformat binary_little_endian 1.0\n")
    with pytest.raises(mesh_to_splat.MeshConversionError):
        mesh_to_splat.glb_to_splat_ply(b"")


def _backdrop_glb(trimesh, obj=None):
    """A TRELLIS-shaped artifact: an object sitting on a zero-thickness,
    double-sided, full-footprint ground quad (the real signature, measured)."""
    import numpy as np

    if obj is None:
        obj = trimesh.creation.box(extents=(0.24, 0.42, 0.17))
        for _ in range(3):  # enough faces to clear the surviving-geometry guard
            obj = obj.subdivide()
    obj.apply_translation([0, 0.21, 0])
    # a coincident double-sided sheet, finely tessellated so face centroids cover
    # the whole footprint the way the real 1100-face quad does
    n = 12
    g = np.linspace(-0.5, 0.5, n)
    gx, gz = np.meshgrid(g, g, indexing="ij")
    verts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [[a, b, d], [a, d, c], [d, b, a], [c, d, a]]  # both windings
    ground = trimesh.Trimesh(vertices=verts, faces=np.array(tris), process=False)
    return trimesh.util.concatenate([obj, ground]).export(file_type="glb")


def test_backdrop_ground_plane_is_stripped():
    """Without this the plane dominates the bbox and fit_asset_to_obb aligns the
    BACKDROP instead of the object — the asset lands sideways in the scene."""
    trimesh = pytest.importorskip("trimesh")
    from server.oreos.genai import mesh_to_splat

    _ply, meta = mesh_to_splat.glb_to_splat_ply(_backdrop_glb(trimesh), target_points=8000, seed=0)
    assert meta["backdrop"]["removed_planes"] == 1
    assert meta["backdrop"]["removed_area_frac"] > 0.5
    assert meta["mesh_faces"] < meta["mesh_faces_raw"]
    assert any("floor/backdrop" in c for c in meta["caveats"])
    # the surviving object is the box, so its bbox is the box's, not the quad's
    span = np.asarray(meta["bbox_max"]) - np.asarray(meta["bbox_min"])
    assert span[0] < 0.5 and span[2] < 0.5
    assert np.argmax(span) == 1  # tallest axis is up, as a standing object should be


@pytest.mark.parametrize(
    "name,build",
    [
        ("thin whiteboard", lambda t: t.creation.box(extents=(1.0, 0.02, 1.4))),
        ("plain cube", lambda t: t.creation.box(extents=(1.0, 1.0, 1.0))),
        ("sphere", lambda t: t.creation.icosphere(subdivisions=3)),
    ],
)
def test_real_flat_and_solid_objects_are_never_stripped(name, build):
    """These scans are full of whiteboards and posters — a genuinely flat OBJECT
    must survive. It has thickness and is not a coincident double-sided sheet."""
    trimesh = pytest.importorskip("trimesh")
    from server.oreos.genai import mesh_to_splat

    _ply, meta = mesh_to_splat.glb_to_splat_ply(
        build(trimesh).export(file_type="glb"), target_points=4000, seed=0
    )
    assert meta["backdrop"]["removed_planes"] == 0, name
    assert meta["mesh_faces"] == meta["mesh_faces_raw"], name


def test_glb_conversion_scale_tracks_sample_density():
    trimesh = pytest.importorskip("trimesh")
    from server.oreos.genai import mesh_to_splat

    glb = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(file_type="glb")
    _ply_sparse, sparse = mesh_to_splat.glb_to_splat_ply(glb, target_points=1000, seed=0)
    _ply_dense, dense = mesh_to_splat.glb_to_splat_ply(glb, target_points=100_000, seed=0)
    # denser sampling => smaller per-gaussian radius, so the surface stays crisp
    assert dense["gaussian_scale"] < sparse["gaussian_scale"]
    assert sparse["surface_area"] == pytest.approx(6.0, rel=1e-3)
