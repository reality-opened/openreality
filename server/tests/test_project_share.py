"""Building (project) share-token tests — share a whole client/project as one unit.

A project token is a read-only *building* capability (no Clerk login): an HS256 JWT signed
with the same secret as the session + share tokens, but under a distinct issuer
(``open-reality-project``) + a ``project:read`` scope and ``client``/``project`` claims. It
authorizes EXACTLY ONE endpoint — the building manifest (``GET /api/projects/manifest``) —
and only for its own client+project; it is rejected (403) everywhere else. The manifest then
mints FRESH per-scene share tokens, so per-scene reads keep using single-scan share tokens
(the project token never reads a scene directly).

These tests cover mint/verify roundtrip, issuer/scope isolation from session + scene-share
tokens, the before_request authorization (manifest yes / scene-read + mismatch no), and that
the manifest yields one entry per matching scene each carrying a per-scene share token.
``jwt`` is faked by conftest (``load_app_module``) — this covers the wrapper logic, not real
crypto. Style mirrors tests/test_share_token.py + tests/test_pilot_reconstruct.py. See
platform/contracts/embed-delivery.md.
"""

from __future__ import annotations

import time
import types

import pytest

from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport
from server.scene_report.store import ModalScenePersistence

from conftest import load_app_module


def _fake_request(
    monkeypatch, app_mod, *, endpoint, bearer=None, query=None, scheme="https",
    host="broker.test", environ=None,
):
    """Install a minimal fake ``request`` for the before_request helpers + route bodies, and
    silence the worker-activity side effect."""
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    req = types.SimpleNamespace(
        headers=headers,
        args=dict(query or {}),
        cookies={},
        url_rule=types.SimpleNamespace(endpoint=endpoint),
        view_args={},
        environ=dict(environ or {}),
        scheme=scheme,
        host=host,
        get_json=lambda *a, **k: dict(query or {}),
    )
    monkeypatch.setattr(app_mod, "request", req)
    monkeypatch.setattr(app_mod, "_note_worker_activity", lambda: None)
    return req


def _report(room_type="living_room"):
    return SceneReport(
        summary="a staged room",
        room_type=room_type,
        objects=[],
        facts=SceneFacts(metrics=SceneMetrics(num_submaps=2)),
    )


def _facts():
    return SceneFacts(metrics=SceneMetrics(num_submaps=2, num_keyframes=6), objects=[])


# -- mint / verify -----------------------------------------------------------------

def test_issue_and_verify_project_token_roundtrip(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, expires_at = app_mod._issue_project_token("efficura", "chorus", "owner_42")
    assert expires_at > time.time()

    claims = app_mod._verify_project_token(token)
    assert claims["scope"] == app_mod.PROJECT_TOKEN_SCOPE
    assert claims["iss"] == app_mod.PROJECT_TOKEN_ISSUER
    assert claims["client"] == "efficura"
    assert claims["project"] == "chorus"
    assert claims["sub"] == "owner_42"


def test_project_token_default_ttl_and_override(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _, exp_default = app_mod._issue_project_token("efficura", "chorus", "o")
    assert exp_default >= time.time() + app_mod.PROJECT_TOKEN_TTL_SEC_DEFAULT - 5
    _, exp_short = app_mod._issue_project_token("efficura", "chorus", "o", ttl_seconds=60)
    assert exp_short <= time.time() + 61


def test_verify_project_token_rejects_session_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A durable session token is HS256 with the same secret but a different issuer — the
    # project verifier must reject it (issuer mismatch).
    session_token, _ = app_mod._issue_session_token({"sub": "user_1"})
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_project_token(session_token)


def test_verify_project_token_rejects_scene_share_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A single-scene share token shares the secret but a different issuer + scope — rejected,
    # so a scene token can never be used as a building token (or vice versa).
    share_token, _ = app_mod._issue_share_token("scan_1", "owner_9")
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_project_token(share_token)


def test_project_token_not_mistaken_for_share_token(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # The inverse: the scene-share verifier must reject a project token (issuer mismatch),
    # so _try_share_token_auth falls through for project tokens.
    project_token, _ = app_mod._issue_project_token("efficura", "chorus", "owner")
    with pytest.raises(app_mod.AuthError):
        app_mod._verify_share_token(project_token)


# -- before_request authorization --------------------------------------------------

def test_project_token_authorizes_manifest(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_project_token("efficura", "chorus", "owner_9")
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route", bearer=token,
        query={"client": "efficura", "project": "chorus"},
    )
    assert app_mod._try_project_token_auth() is True
    claims = app_mod.request.environ[app_mod.AUTH_CLAIMS_ENV_KEY]
    assert claims["sub"] == "owner_9"
    assert claims["scope"] == app_mod.PROJECT_TOKEN_SCOPE
    assert claims["client"] == "efficura" and claims["project"] == "chorus"
    assert claims["project_token"] is True


def test_project_token_rejected_on_scene_read_endpoint(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_project_token("efficura", "chorus", "owner")
    # A project token presented to a per-scene read endpoint is forbidden — the project token
    # authorizes the manifest ONLY (per-scene reads use per-scene share tokens).
    _fake_request(
        monkeypatch, app_mod, endpoint="get_scene_route", bearer=token,
        query={"client": "efficura", "project": "chorus"},
    )
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_project_token_auth()
    assert exc.value.status_code == 403


def test_project_token_rejected_on_mismatched_client_project(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    token, _ = app_mod._issue_project_token("efficura", "chorus", "owner")
    # Right endpoint, wrong query target → forbidden (the token is scoped to its own building).
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route", bearer=token,
        query={"client": "efficura", "project": "other"},
    )
    with pytest.raises(app_mod.AuthError) as exc:
        app_mod._try_project_token_auth()
    assert exc.value.status_code == 403


def test_no_project_token_falls_through(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route",
        query={"client": "efficura", "project": "chorus"},
    )
    assert app_mod._try_project_token_auth() is False


def test_session_token_does_not_authorize_manifest(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # A Clerk/session bearer on the manifest endpoint is NOT a project token → fall through
    # (the normal path then runs, and the route body rejects a non-project scope).
    session_token, _ = app_mod._issue_session_token({"sub": "owner"})
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route", bearer=session_token,
        query={"client": "efficura", "project": "chorus"},
    )
    assert app_mod._try_project_token_auth() is False


# -- manifest route body -----------------------------------------------------------

def _seed_scenes(app_mod, store, user_id="efficura-ops"):
    """Persist 3 chorus scenes (living_room, unknown, kitchen) + 1 unrelated scene."""
    app_mod.configure_scene_persistence(store)
    # Saved oldest→newest; list_scenes returns newest first.
    store.save_scene(user_id, "chorus-1", _report("living_room"), _facts(),
                     client="efficura", project="chorus")
    store.save_scene(user_id, "chorus-2", _report("unknown"), _facts(),
                     client="efficura", project="chorus")
    store.save_scene(user_id, "chorus-3", _report("kitchen"), _facts(),
                     client="efficura", project="chorus")
    # A different project for the same owner must NOT appear in the chorus manifest.
    store.save_scene(user_id, "other-1", _report("office"), _facts(),
                     client="efficura", project="harmony")


def test_manifest_returns_one_entry_per_matching_scene(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence(store={}, blob_root=str(tmp_path))
    _seed_scenes(app_mod, store)

    claims = {
        "sub": "efficura-ops",
        "scope": app_mod.PROJECT_TOKEN_SCOPE,
        "client": "efficura",
        "project": "chorus",
        "project_token": True,
    }
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route",
        query={"client": "efficura", "project": "chorus"},
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: claims},
    )

    out = app_mod.project_manifest_route()
    payload = out["args"][0]  # FakeFlask.jsonify wraps positional args
    assert payload["client"] == "efficura" and payload["project"] == "chorus"

    scenes = payload["scenes"]
    assert len(scenes) == 3  # the 3 chorus scenes only, not the harmony one
    by_id = {s["scan_id"]: s for s in scenes}
    assert set(by_id) == {"chorus-1", "chorus-2", "chorus-3"}

    # Each scene carries a fresh per-scene share token (the embed credential) + embed url.
    for sid, scene in by_id.items():
        assert scene["share_token"].startswith("hs256.")  # conftest's fake HS256 encoding
        verified = app_mod._verify_share_token(scene["share_token"])
        assert verified["scan_id"] == sid
        assert verified["sub"] == "efficura-ops"
        assert scene["embed_url"].endswith(scene["share_token"])
        assert f"scan_id={sid}" in scene["embed_url"]

    # Label = room_type capitalized, else "Scene N" for unknown/missing.
    assert by_id["chorus-1"]["label"] == "Living_room"
    assert by_id["chorus-3"]["label"] == "Kitchen"
    assert by_id["chorus-2"]["label"].startswith("Scene ")  # unknown → positional fallback


def test_manifest_rejects_non_project_scope(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence(store={}, blob_root=str(tmp_path))
    _seed_scenes(app_mod, store)
    # A request that reached the route with normal (Clerk/session) claims — no project scope —
    # must be rejected: the manifest is project-token-only.
    claims = {"sub": "efficura-ops", "scope": app_mod.SHARE_TOKEN_SCOPE}
    _fake_request(
        monkeypatch, app_mod, endpoint="project_manifest_route",
        query={"client": "efficura", "project": "chorus"},
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: claims},
    )
    body, status = app_mod.project_manifest_route()
    assert status == 403


# -- mint route body ---------------------------------------------------------------

def test_share_project_route_mints_for_owned_project(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence(store={}, blob_root=str(tmp_path))
    _seed_scenes(app_mod, store)

    claims = {"sub": "efficura-ops"}
    _fake_request(
        monkeypatch, app_mod, endpoint="share_project_route",
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: claims},
        query={"client": "efficura", "project": "chorus"},  # doubles as get_json body
    )
    out = app_mod.share_project_route()
    payload = out["args"][0]
    assert payload["client"] == "efficura" and payload["project"] == "chorus"
    assert payload["expires_at"] > time.time()
    # The minted token verifies as a project token for this building.
    verified = app_mod._verify_project_token(payload["project_token"])
    assert verified["client"] == "efficura" and verified["project"] == "chorus"
    assert verified["sub"] == "efficura-ops"
    # building_embed_url targets building.html with urlencoded client/project/token (the
    # fake JWT carries base64 '=' padding, which must be percent-encoded in the URL).
    from urllib.parse import quote
    url = payload["building_embed_url"]
    assert url.startswith("https://broker.test/building.html#")
    assert "client=efficura" in url and "project=chorus" in url
    assert f"token={quote(payload['project_token'], safe='')}" in url


def test_share_project_route_404_without_matching_scene(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = ModalScenePersistence(store={}, blob_root=str(tmp_path))
    _seed_scenes(app_mod, store)

    claims = {"sub": "efficura-ops"}
    _fake_request(
        monkeypatch, app_mod, endpoint="share_project_route",
        environ={app_mod.AUTH_CLAIMS_ENV_KEY: claims},
        query={"client": "efficura", "project": "nope"},  # owner has no scenes here
    )
    body, status = app_mod.share_project_route()
    assert status == 404
