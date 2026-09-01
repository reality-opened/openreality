"""Object-layer contract tests: protocol mirror, inventory ingest/curation, durable store,
and the embed serving path (SCENE_DETAIL manifest + share-token-gated GLB endpoint).

All fixtures are SYNTHETIC — a hand-written manifest + junk GLB bytes. No Finc-derived data
(coordinates, real GLBs, real inventory) is used or committed. The app-route tests reuse the
``load_app_module`` harness (conftest fakes Flask/jwt/etc.), calling the route functions
directly the way ``test_share_token`` / ``test_scene_qa`` do.
"""

from __future__ import annotations

import types

import pytest

from conftest import load_app_module

from server.scene_report.object_layer import (
    DEFAULT_OBJECT_LAYER_DISCLAIMER,
    ObjectLayerManifest,
    normalize_quality_tier,
    parse_object_layer_manifest,
)
from server.scene_report.object_layer_ingest import (
    convert_scene_inventory,
    curate_manifest,
    transpose_3x3,
)
from server.scene_report.store import ModalScenePersistence


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
GLB_BYTES = b"glTF\x02\x00\x00\x00synthetic-binary-glb-bytes"


def _obb(center=(0.0, 0.0, 0.0)):
    return {"center": list(center), "extents": [1.0, 2.0, 3.0], "rotation": IDENTITY}


def _manifest_dict(scan_id="scanZ"):
    """A wire-shape manifest: one mesh object (widget) + one box-only object (marker)."""
    return {
        "version": 1,
        "scan_id": scan_id,
        "frame": "synthetic up-to-scale world",
        "disclaimer": "test disclaimer",
        "objects": [
            {
                "id": "widget",
                "label": "Widget",
                "obb": _obb((1.0, 0.0, 0.0)),
                "quality": "good",
                "provenance": "AI-completed — envelope-verified",
                "caveats": [],
                "asset_url": f"/api/scenes/{scan_id}/objects/widget.glb",
            },
            {
                "id": "marker",
                "label": "Marker",
                "obb": _obb((0.0, 1.0, 0.0)),
                "quality": "low",
                "provenance": "detected only",
                "caveats": ["box only"],
                # no asset_url → box-only
            },
        ],
    }


# ── protocol mirror (object_layer.py) ─────────────────────────────────────────

def test_manifest_roundtrips_and_omits_none_asset_url():
    m = ObjectLayerManifest.model_validate(_manifest_dict())
    assert m.version == 1 and len(m.objects) == 2
    dumped = m.model_dump(exclude_none=True)
    widget, marker = dumped["objects"]
    # CRITICAL: box-only object must OMIT asset_url (the TS guard rejects asset_url: null).
    assert "asset_url" in widget
    assert "asset_url" not in marker


def test_manifest_default_disclaimer():
    m = ObjectLayerManifest.model_validate({"version": 1, "objects": []})
    assert m.disclaimer == DEFAULT_OBJECT_LAYER_DISCLAIMER


@pytest.mark.parametrize("bad", [
    {"center": [0, 0], "extents": [1, 1, 1], "rotation": IDENTITY},          # center len 2
    {"center": [0, 0, 0], "extents": [1, 1], "rotation": IDENTITY},          # extents len 2
    {"center": [0, 0, 0], "extents": [1, 1, 1], "rotation": [[1, 0], [0, 1]]},  # not 3x3
    {"center": [0, 0, float("nan")], "extents": [1, 1, 1], "rotation": IDENTITY},  # non-finite
])
def test_obb_validation_rejects_malformed(bad):
    obj = {"id": "x", "label": "X", "obb": bad, "quality": "good",
           "provenance": "p", "caveats": []}
    assert parse_object_layer_manifest({"version": 1, "objects": [obj]}) is None


def test_normalize_quality_tier():
    assert normalize_quality_tier("GOOD") == "good"
    assert normalize_quality_tier("Usable") == "usable"
    for junk in ("LOW", "STILL-CHIMERA", "NO-GO", "", None, 123):
        assert normalize_quality_tier(junk) == "low"


def test_parse_never_throws():
    assert parse_object_layer_manifest(None) is None
    assert parse_object_layer_manifest("not a dict") is None
    assert parse_object_layer_manifest({"objects": [{"id": 1}]}) is None  # bad item
    assert parse_object_layer_manifest(_manifest_dict()) is not None


# ── inventory ingest + curation (object_layer_ingest.py) ──────────────────────

def _inventory():
    """Synthetic EXP-20-shaped inventory: row-major axes_rows + FULL extent, upper-case tiers,
    internal notes (must NOT surface), one object with no world_obb (must be skipped)."""
    axes_rows = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # a 90° yaw, rows=axes
    return {
        "frame": "synthetic world",
        "generated_utc": "2026-07-07T00:00:00Z",
        "provenance_note": "generative — envelope-verified",
        "objects": [
            {
                "id": "chair", "label_guess": "office chair", "submap": "00042", "best_frame": 7,
                "quality_tier": "GOOD", "notes": "internal: v2 tighter than v1",
                "world_obb": {"center": [1, 2, 3], "extent": [0.4, 0.6, 0.8],
                              "half_extent": [0.2, 0.3, 0.4], "axes_rows": axes_rows},
            },
            {
                "id": "screen", "label_guess": "monitor", "submap": "00042",
                "quality_tier": "LOW", "notes": "internal: box-only",
                "world_obb": {"center": [4, 5, 6], "extent": [0.1, 0.1, 0.1], "axes_rows": IDENTITY},
            },
            {"id": "ghost", "label_guess": "no box", "quality_tier": "GOOD"},  # no world_obb → skip
        ],
    }


def test_transpose_3x3():
    assert transpose_3x3([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]


def test_convert_uses_full_extent_and_transposes_axes():
    m = convert_scene_inventory(_inventory(), scan_id="s")
    assert [o.id for o in m.objects] == ["chair", "screen"]  # ghost (no obb) skipped
    chair = m.objects[0]
    # extents == FULL extent, i.e. 2 × half_extent (pins the half-extent regression).
    assert chair.obb.extents == [0.4, 0.6, 0.8]
    # rotation is columns-are-axes = transpose(axes_rows): row0 of axes_rows becomes col0.
    assert [chair.obb.rotation[r][0] for r in range(3)] == [0.0, 1.0, 0.0]
    assert chair.quality == "good" and m.objects[1].quality == "low"
    assert chair.source_frame == "submap 00042 · frame 7"
    assert m.objects[1].source_frame == "submap 00042"  # no best_frame
    assert m.frame == "synthetic world" and m.generated_at == "2026-07-07T00:00:00Z"
    assert m.scan_id == "s"


def test_convert_does_not_surface_notes():
    m = convert_scene_inventory(_inventory())
    assert all(o.caveats == [] for o in m.objects)  # internal notes NEVER become caveats


def test_curate_excludes_and_caveats_and_box_only():
    m = ObjectLayerManifest.model_validate(_manifest_dict())
    cur = curate_manifest(
        m,
        exclude_ids={"nonexistent"},
        caveats={"widget": "curated widget caveat"},
        box_only_caveat="detected only, box shown",
    )
    widget = next(o for o in cur.objects if o.id == "widget")
    marker = next(o for o in cur.objects if o.id == "marker")
    assert widget.caveats == ["curated widget caveat"]
    # marker already had its own caveat → box-only default does NOT clobber it.
    assert marker.caveats == ["box only"]
    # input manifest untouched (curate returns a copy).
    assert m.objects[0].caveats == []


def test_curate_exclude_and_include_allowlist():
    m = ObjectLayerManifest.model_validate(_manifest_dict())
    assert [o.id for o in curate_manifest(m, exclude_ids={"marker"}).objects] == ["widget"]
    assert [o.id for o in curate_manifest(m, include_ids={"marker"}).objects] == ["marker"]
    assert curate_manifest(m, drop_box_only=True).objects[-1].id == "widget"  # marker dropped


def test_curate_box_only_caveat_applied_when_empty():
    d = _manifest_dict()
    d["objects"][1]["caveats"] = []  # marker with no caveat
    m = ObjectLayerManifest.model_validate(d)
    cur = curate_manifest(m, box_only_caveat="box only default")
    assert next(o for o in cur.objects if o.id == "marker").caveats == ["box only default"]


# ── durable store (store.py) ──────────────────────────────────────────────────

def _store(tmp_path):
    return ModalScenePersistence({}, str(tmp_path))


def _write_glb(tmp_path, name="widget.glb"):
    p = tmp_path / name
    p.write_bytes(GLB_BYTES)
    return str(p)


def test_attach_and_get_object_layer_and_asset(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "scanZ", report={}, facts={})
    src = _write_glb(tmp_path)
    store.attach_object_layer("u", "scanZ", _manifest_dict(), {"widget": src})

    layer = store.get_object_layer("u", "scanZ")
    assert layer is not None and layer["scan_id"] == "scanZ"
    assert {o["id"] for o in layer["objects"]} == {"widget", "marker"}

    # widget has an on-disk GLB; marker (box-only) does not.
    path = store.get_object_asset_path("u", "scanZ", "widget")
    assert path is not None
    assert store.get_object_asset("u", "scanZ", "widget") == GLB_BYTES
    assert store.get_object_asset_path("u", "scanZ", "marker") is None  # box-only, no file
    # also surfaced on the record / SCENE_DETAIL source
    assert store.get_scene("u", "scanZ")["object_layer"]["version"] == 1


def test_object_asset_rejects_unsafe_and_unknown_ids(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "scanZ", report={}, facts={})
    store.attach_object_layer("u", "scanZ", _manifest_dict(), {"widget": _write_glb(tmp_path)})
    assert store.get_object_asset_path("u", "scanZ", "../../etc/passwd") is None  # traversal
    assert store.get_object_asset_path("u", "scanZ", "not_in_manifest") is None   # unknown id
    # per-user scoping — another user can't reach it
    assert store.get_object_asset_path("other", "scanZ", "widget") is None
    assert store.get_object_layer("other", "scanZ") is None


def test_layerless_scene_unaffected(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "plain", report={}, facts={})
    assert store.get_object_layer("u", "plain") is None
    assert store.get_object_asset_path("u", "plain", "widget") is None
    # backward-compatible: the record carries object_layer=None
    assert store.get_scene("u", "plain")["object_layer"] is None


def test_attach_to_missing_scan_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.attach_object_layer("u", "ghost", _manifest_dict(), {})


def test_attach_malformed_manifest_raises(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "scanZ", report={}, facts={})
    with pytest.raises(ValueError):
        store.attach_object_layer("u", "scanZ", {"objects": [{"id": 1}]}, {})


def test_reattach_prunes_stale_glbs(tmp_path):
    import os
    store = _store(tmp_path)
    store.save_scene("u", "scanZ", report={}, facts={})
    store.attach_object_layer("u", "scanZ", _manifest_dict(), {"widget": _write_glb(tmp_path)})
    obj_dir = os.path.join(store._scene_dir("u", "scanZ"), "objects")
    assert os.listdir(obj_dir) == ["widget.glb"]

    # Re-attach a manifest that renames the mesh object → the old GLB must be pruned.
    d = _manifest_dict()
    d["objects"][0]["id"] = "gadget"
    d["objects"][0]["asset_url"] = "/api/scenes/scanZ/objects/gadget.glb"
    store.attach_object_layer("u", "scanZ", d, {"gadget": _write_glb(tmp_path, "gadget.glb")})
    assert sorted(os.listdir(obj_dir)) == ["gadget.glb"]  # widget.glb gone


def test_save_scene_with_object_layer_kwarg(tmp_path):
    store = _store(tmp_path)
    store.save_scene("u", "scanZ", report={}, facts={}, object_layer=_manifest_dict())
    layer = store.get_object_layer("u", "scanZ")
    assert layer is not None and len(layer["objects"]) == 2


# ── embed serving path (app.py routes) ────────────────────────────────────────

def _load_app_with_scene(monkeypatch, tmp_path, *, with_layer=True, user="owner"):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene(user, "scanZ", report={"room_type": "office"}, facts={})
    if with_layer:
        src = tmp_path / "widget.glb"
        src.write_bytes(GLB_BYTES)
        store.attach_object_layer(user, "scanZ", _manifest_dict(), {"widget": str(src)})
    app_mod.configure_scene_persistence(store)
    return app_mod


def _fake_authed_request(app_mod, monkeypatch, user="owner"):
    req = types.SimpleNamespace(
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: {"sub": user, "share": True}},
        args={},
    )
    monkeypatch.setattr(app_mod, "request", req)
    return req


def test_new_endpoint_is_share_token_allowed(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert "get_scene_object_asset_route" in app_mod.SHARE_TOKEN_ALLOWED_ENDPOINTS


def test_share_token_authorizes_object_asset_endpoint(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_share_token("scanZ", "owner")
    req = types.SimpleNamespace(
        headers={"Authorization": f"Bearer {token}"},
        args={},
        url_rule=types.SimpleNamespace(endpoint="get_scene_object_asset_route"),
        view_args={"scan_id": "scanZ", "object_id": "widget"},
        environ={},
    )
    monkeypatch.setattr(app_mod, "request", req)
    monkeypatch.setattr(app_mod, "_note_worker_activity", lambda: None)
    assert app_mod._try_share_token_auth() is True
    # a token for a DIFFERENT scan is rejected on this endpoint
    req.view_args["scan_id"] = "other"
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_share_token_auth()
    assert exc.value.status_code == 403


def test_scene_detail_carries_object_layer(monkeypatch, tmp_path):
    app_mod = _load_app_with_scene(monkeypatch, tmp_path, with_layer=True)
    _fake_authed_request(app_mod, monkeypatch)
    result = app_mod.get_scene_route("scanZ")
    payload = result["args"][0]  # fake jsonify → {"args": (payload,), "kwargs": {}}
    assert "object_layer" in payload
    assert payload["object_layer"]["scan_id"] == "scanZ"
    assert {o["id"] for o in payload["object_layer"]["objects"]} == {"widget", "marker"}


def test_scene_detail_omits_layer_when_absent(monkeypatch, tmp_path):
    app_mod = _load_app_with_scene(monkeypatch, tmp_path, with_layer=False)
    _fake_authed_request(app_mod, monkeypatch)
    payload = app_mod.get_scene_route("scanZ")["args"][0]
    assert "object_layer" not in payload  # backward-compatible: key omitted entirely


def test_object_asset_route_streams_glb_under_token(monkeypatch, tmp_path):
    app_mod = _load_app_with_scene(monkeypatch, tmp_path, with_layer=True)
    _fake_authed_request(app_mod, monkeypatch)
    # valid object → send_file (fake returns a dict carrying the file arg + kwargs)
    result = app_mod.get_scene_object_asset_route("scanZ", "widget")
    assert result["kwargs"]["mimetype"] == "model/gltf-binary"
    assert result["file"][0].endswith("widget.glb")

    # unknown object → 404
    missing = app_mod.get_scene_object_asset_route("scanZ", "nope")
    assert missing[1] == 404
    # box-only object (no GLB) → 404
    box = app_mod.get_scene_object_asset_route("scanZ", "marker")
    assert box[1] == 404
    # unknown scan → 404
    noscene = app_mod.get_scene_object_asset_route("ghost", "widget")
    assert noscene[1] == 404


def test_object_asset_route_requires_auth(monkeypatch, tmp_path):
    app_mod = _load_app_with_scene(monkeypatch, tmp_path, with_layer=True)
    # No auth claims in the environ → _auth_user_id() is None → 401 (before any store read).
    monkeypatch.setattr(app_mod, "request", types.SimpleNamespace(environ={}, args={}))
    result = app_mod.get_scene_object_asset_route("scanZ", "widget")
    assert result[1] == 401
