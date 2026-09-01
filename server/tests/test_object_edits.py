"""Object-review edit persistence (product-workflow Detect stage).

Covers the store's ``update_object_edit`` (relabel preserves the model's original ``query``;
dismiss persists; unknown id / missing scan raise) and the
``PATCH /api/scenes/<id>/objects/<object_id>`` route helpers (auth required; body validation;
unknown id -> 404). Follows tests/test_product_workflow_routes.py: exercises the store
directly and the app helpers through the conftest ``load_app_module`` stub (no Flask/GPU/
network), so these tests always run -- they never touch ``vggt_slam`` / ``decompose_camera``.
"""

from __future__ import annotations

import types

import pytest

from conftest import load_app_module

from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    SceneFacts,
    SceneReport,
)
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _facts_with_objects():
    return SceneFacts(
        objects=[
            ObjectInstance(
                query="office chair",
                center=[0.0, 0.0, 0.0],
                extent=[0.5, 0.5, 1.0],
                confidence=0.9,
                evidence=[EvidenceRef(submap_id=0, frame_idx=1)],
            ),
            ObjectInstance(
                query="potted plant",
                center=[1.0, 0.0, 0.0],
                extent=[0.3, 0.3, 0.6],
                confidence=0.4,
            ),
        ]
    )


def _store_with_objects(tmp_path, user="user-a", scan="scan1"):
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene(user, scan, SceneReport(summary="s", room_type="office"), _facts_with_objects())
    return store


# ---------------------------------------------------------------------------
# store: update_object_edit
# ---------------------------------------------------------------------------


def test_relabel_records_human_label_and_preserves_original_query(tmp_path):
    store = _store_with_objects(tmp_path)

    obj = store.update_object_edit("user-a", "scan1", 0, label="standing lamp")
    assert obj["human_label"] == "standing lamp"
    assert obj["query"] == "office chair"  # original detection NEVER overwritten
    assert obj.get("dismissed") in (None, False)

    # persisted: a fresh read sees the edit, with the original query intact
    reread = store.get_scene("user-a", "scan1")["facts"]["objects"]
    assert reread[0]["human_label"] == "standing lamp"
    assert reread[0]["query"] == "office chair"
    # the sibling object is untouched
    assert "human_label" not in reread[1] or reread[1]["human_label"] is None
    assert reread[1]["query"] == "potted plant"


def test_dismiss_persists_without_erasing_the_detection(tmp_path):
    store = _store_with_objects(tmp_path)

    store.update_object_edit("user-a", "scan1", 1, dismissed=True)

    reread = store.get_scene("user-a", "scan1")["facts"]["objects"]
    assert reread[1]["dismissed"] is True
    assert reread[1]["query"] == "potted plant"  # detection preserved
    # the inventory list keeps BOTH objects -- dismiss is an annotation, not a delete
    assert len(reread) == 2
    assert reread[0].get("dismissed") in (None, False)


def test_relabel_then_dismiss_are_independent(tmp_path):
    store = _store_with_objects(tmp_path)
    store.update_object_edit("user-a", "scan1", 0, label="beanbag")
    store.update_object_edit("user-a", "scan1", 0, dismissed=True)

    obj = store.get_scene("user-a", "scan1")["facts"]["objects"][0]
    assert obj["human_label"] == "beanbag"  # dismiss did not clobber the relabel
    assert obj["dismissed"] is True

    # a later relabel of the same object leaves dismissed as-is
    store.update_object_edit("user-a", "scan1", 0, label="beanbag chair")
    obj = store.get_scene("user-a", "scan1")["facts"]["objects"][0]
    assert obj["human_label"] == "beanbag chair"
    assert obj["dismissed"] is True


def test_unknown_object_id_raises_index_error(tmp_path):
    store = _store_with_objects(tmp_path)
    with pytest.raises(IndexError):
        store.update_object_edit("user-a", "scan1", 99, dismissed=True)
    with pytest.raises(IndexError):
        store.update_object_edit("user-a", "scan1", -1, dismissed=True)
    # a bool is not a valid index even though bool subclasses int
    with pytest.raises(IndexError):
        store.update_object_edit("user-a", "scan1", True, dismissed=True)  # type: ignore[arg-type]


def test_missing_scan_raises_key_error(tmp_path):
    store = _store_with_objects(tmp_path)
    with pytest.raises(KeyError):
        store.update_object_edit("user-a", "does-not-exist", 0, label="x")
    # a scan the caller doesn't own is invisible -> not_found for them too
    with pytest.raises(KeyError):
        store.update_object_edit("other-user", "scan1", 0, label="x")


# ---------------------------------------------------------------------------
# app helper: _coerce_object_edit (body validation)
# ---------------------------------------------------------------------------


def test_coerce_object_edit_accepts_label_and_dismissed(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._coerce_object_edit({"label": "sofa"}) == {"label": "sofa"}
    assert app_mod._coerce_object_edit({"dismissed": True}) == {"dismissed": True}
    assert app_mod._coerce_object_edit({"label": "sofa", "dismissed": False}) == {
        "label": "sofa",
        "dismissed": False,
    }


def test_coerce_object_edit_rejects_bad_bodies(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    for bad in (None, [], "hi", 3):
        with pytest.raises(ValueError):
            app_mod._coerce_object_edit(bad)
    with pytest.raises(ValueError):  # empty edit
        app_mod._coerce_object_edit({})
    with pytest.raises(ValueError):  # wrong-typed label
        app_mod._coerce_object_edit({"label": 5})
    with pytest.raises(ValueError):  # wrong-typed dismissed
        app_mod._coerce_object_edit({"dismissed": "yes"})


# ---------------------------------------------------------------------------
# route: patch_scene_object_route (auth + wiring)
# ---------------------------------------------------------------------------


def _fake_request(monkeypatch, app_mod, body, claims=None):
    """Point app.py's module-level ``request`` at a stub carrying this JSON body + auth
    claims (``_auth_user_id`` reads ``request.environ[AUTH_CLAIMS_ENV_KEY]``)."""
    environ = {app_mod.AUTH_CLAIMS_ENV_KEY: claims} if claims is not None else {}
    req = types.SimpleNamespace(
        environ=environ,
        get_json=lambda silent=False: body,
        args={},
    )
    monkeypatch.setattr(app_mod, "request", req)


def test_route_requires_auth(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "_scene_persistence", _store_with_objects(tmp_path))
    _fake_request(monkeypatch, app_mod, {"label": "x"}, claims=None)  # no auth claims

    result = app_mod.patch_scene_object_route("scan1", 0)
    # _auth_response(AuthError("invalid_token")) -> (jsonify(...), 401) via the stubs
    assert isinstance(result, tuple) and result[1] == 401


def test_route_relabel_persists_and_returns_object(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_objects(tmp_path)
    monkeypatch.setattr(app_mod, "_scene_persistence", store)
    _fake_request(monkeypatch, app_mod, {"label": "standing lamp"}, claims={"sub": "user-a"})

    result = app_mod.patch_scene_object_route("scan1", 0)
    # jsonify stub returns {"args": (payload,), "kwargs": {}}
    payload = result["args"][0]
    assert payload["object_id"] == 0
    assert payload["object"]["human_label"] == "standing lamp"
    assert payload["object"]["query"] == "office chair"
    # actually persisted in the store
    assert store.get_scene("user-a", "scan1")["facts"]["objects"][0]["human_label"] == "standing lamp"


def test_route_unknown_object_id_is_404(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "_scene_persistence", _store_with_objects(tmp_path))
    _fake_request(monkeypatch, app_mod, {"dismissed": True}, claims={"sub": "user-a"})

    result = app_mod.patch_scene_object_route("scan1", 42)
    assert isinstance(result, tuple) and result[1] == 404
    assert result[0]["args"][0]["error"] == "unknown_object"


def test_route_missing_scene_is_404(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "_scene_persistence", _store_with_objects(tmp_path))
    _fake_request(monkeypatch, app_mod, {"label": "x"}, claims={"sub": "user-a"})

    result = app_mod.patch_scene_object_route("nope", 0)
    assert isinstance(result, tuple) and result[1] == 404


def test_route_bad_body_is_400(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "_scene_persistence", _store_with_objects(tmp_path))
    _fake_request(monkeypatch, app_mod, {}, claims={"sub": "user-a"})  # empty edit

    result = app_mod.patch_scene_object_route("scan1", 0)
    assert isinstance(result, tuple) and result[1] == 400
    assert result[0]["args"][0]["error"] == "invalid_request"
