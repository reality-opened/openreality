"""Self-host mode tests (GPU-free).

Covers the two seams the self-host fork stands on:

1. ``OPENREALITY_AUTH=local`` in ``server.app`` — boots without any Clerk
   configuration, the static bearer authorizes (constant-time), everything else
   still 401s, and the local claims can bootstrap session tokens / API-key
   mints exactly like a verified Clerk JWT.
2. ``server.selfhost`` — the local stores, the job runner, upload containment,
   and the composed wiring (spawner → runner thread → jobs store) end to end
   on the failure path that needs no SLAM stack.

``server.app`` is loaded via ``conftest.load_app_module`` (flask/jwt/torch
faked), the repo's app-test convention.
"""

from __future__ import annotations

import threading
import time

import pytest

from conftest import load_app_module

LOCAL_TOKEN = "selfhost-secret-token-0123456789"


def _load_local_app(monkeypatch, token: str = LOCAL_TOKEN):
    monkeypatch.setenv("OPENREALITY_AUTH", "local")
    monkeypatch.setenv("OPENREALITY_LOCAL_TOKEN", token)
    return load_app_module(monkeypatch)


# ---------------------------------------------------------------------------
# OPENREALITY_AUTH=local (server.app)
# ---------------------------------------------------------------------------


def test_local_mode_boots_without_clerk_trust_anchor(monkeypatch):
    app_mod = _load_local_app(monkeypatch)
    assert app_mod.AUTH_MODE == "local"
    # No Clerk trust anchor exists: constants cleared, JWKS never required.
    assert app_mod.CLERK_JWKS_URL is None
    assert app_mod.CLERK_JWT_ISSUER is None
    assert app_mod.CLERK_JWT_ALLOWED_AZP == frozenset()


def test_local_token_authorizes_and_everything_else_401s(monkeypatch):
    app_mod = _load_local_app(monkeypatch)

    claims = app_mod._verify_any_token(LOCAL_TOKEN)
    assert claims["sub"] == "local"
    assert claims["iss"] == app_mod.LOCAL_AUTH_ISSUER
    assert claims["tier"] == app_mod.APPROVED_TIER
    # No auth_kind marker: the /api/keys mint route (which refuses
    # key-mints-key by auth_kind) accepts the local bearer as a bootstrap.
    assert "auth_kind" not in claims

    with pytest.raises(app_mod.AuthError):
        app_mod._verify_any_token("not-the-token-but-equally-long")
    with pytest.raises(app_mod.AuthError):
        # ork_ prefix with no registry configured: rejected, never crashes.
        app_mod._verify_any_token("ork_0123456789abcdef")


def test_local_claims_bootstrap_a_session_token(monkeypatch):
    app_mod = _load_local_app(monkeypatch)
    token, expires_at = app_mod._issue_session_token(app_mod._verify_any_token(LOCAL_TOKEN))
    assert expires_at > time.time()
    claims = app_mod._verify_any_token(token)
    assert claims["sub"] == "local"


def test_local_mode_requires_a_real_token(monkeypatch):
    monkeypatch.setenv("OPENREALITY_AUTH", "local")
    monkeypatch.setenv("OPENREALITY_LOCAL_TOKEN", "short")
    with pytest.raises(RuntimeError, match="OPENREALITY_LOCAL_TOKEN"):
        load_app_module(monkeypatch)


def test_unknown_auth_mode_fails_loud(monkeypatch):
    monkeypatch.setenv("OPENREALITY_AUTH", "none")
    with pytest.raises(RuntimeError, match="OPENREALITY_AUTH"):
        load_app_module(monkeypatch)


def test_clerk_mode_stays_the_default(monkeypatch):
    monkeypatch.delenv("OPENREALITY_AUTH", raising=False)
    monkeypatch.delenv("OPENREALITY_LOCAL_TOKEN", raising=False)
    app_mod = load_app_module(monkeypatch)
    assert app_mod.AUTH_MODE == "clerk"
    assert app_mod.CLERK_JWKS_URL  # the fixture's fake env, required as before
    # The local branch is inert in clerk mode — an empty LOCAL_AUTH_TOKEN can
    # never match a bearer (compare_digest of "" vs a non-empty token).
    assert app_mod.LOCAL_AUTH_TOKEN == ""


# ---------------------------------------------------------------------------
# server.selfhost primitives
# ---------------------------------------------------------------------------


def test_filebacked_dict_roundtrip(tmp_path):
    from server.selfhost import FileBackedDict

    d = FileBackedDict(str(tmp_path / "kv"))
    d["user:index"] = ["a", "b"]
    d["sha:0f/slash:colon"] = {"n": 1}
    assert d["user:index"] == ["a", "b"]
    assert d.get("sha:0f/slash:colon") == {"n": 1}
    assert d.get("missing") is None
    assert "user:index" in d and "missing" not in d
    assert sorted(d.keys()) == sorted(["user:index", "sha:0f/slash:colon"])

    # Durability: a fresh handle over the same root sees the data.
    d2 = FileBackedDict(str(tmp_path / "kv"))
    assert d2["user:index"] == ["a", "b"]

    assert d.pop("user:index") == ["a", "b"]
    assert d.get("user:index") is None
    assert d.pop("user:index", "gone") == "gone"


def test_local_runner_runs_jobs_and_survives_errors():
    from server.selfhost import LocalJobRunner

    runner = LocalJobRunner()
    ran = []
    done = threading.Event()

    def boom(**_kw):
        ran.append("boom")
        raise RuntimeError("job crash must not kill the worker")

    def ok(**_kw):
        ran.append("ok")
        done.set()

    runner.submit(boom)
    runner.submit(ok)
    assert done.wait(timeout=5), "runner thread died after the crashing job"
    assert ran == ["boom", "ok"]


def test_contain_upload_rejects_escapes(tmp_path):
    from server.selfhost import _contain_upload

    root = str(tmp_path)
    staged = tmp_path / "user" / "_uploads" / "u1" / "clip.mp4"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x")
    assert _contain_upload(root, "user/_uploads/u1/clip.mp4") == str(staged.resolve())

    for bad in ("../outside.mp4", "user/plain/clip.mp4", "/etc/passwd"):
        with pytest.raises(ValueError):
            _contain_upload(root, bad)


def test_recording_job_answers_honestly():
    from server.selfhost import run_recording_job

    jobs: dict = {}
    run_recording_job(jobs, job_id="j1", user_id="u", scan_id="s",
                      upload_rel_path="x", recording_name="r.db")
    record = jobs["j1"]
    assert record["status"] == "failed"
    assert record["error"] == "recording_ingest_not_supported_selfhost"


# ---------------------------------------------------------------------------
# Composed wiring: route spawner → runner thread → jobs store
# ---------------------------------------------------------------------------


def test_build_selfhost_app_wires_the_job_lane(monkeypatch, tmp_path):
    app_mod = _load_local_app(monkeypatch)

    from server.oreos import jobs as demo_jobs
    from server.oreos import routes_ingest
    from server.selfhost import LocalJobRunner, build_selfhost_app

    runner = LocalJobRunner()
    asgi = build_selfhost_app(str(tmp_path / "data"), runner=runner)
    assert asgi is app_mod.asgi_application

    # The jobs store is the injected plain dict, not a lazy modal binding.
    store = demo_jobs._get_jobs_store()
    assert isinstance(store, dict)

    # A recon spawn with a bogus staged path runs on the worker thread and
    # publishes an honest failure — the exact wire the ingest route drives.
    routes_ingest._spawn_recon_job(
        job_id="job-esc", user_id="u", upload_rel_path="../escape.mp4",
        scan_id="s", label=None, source="recon_video",
    )
    deadline = time.time() + 5
    while time.time() < deadline and "job-esc" not in store:
        time.sleep(0.02)
    record = store.get("job-esc")
    assert record and record["status"] == "failed"
    assert record["error"] == "invalid_upload_path"

    # API keys ride the file-backed registry end to end: mint via the module
    # under test, then verify a bearer against it through the app's auth path.
    registry = app_mod._api_key_registry
    full_key, _record = registry.mint(
        "local", name="selfhost-test",
        claims_snapshot={"tier": app_mod.APPROVED_TIER},
    )
    claims = app_mod._verify_any_token(full_key)
    assert claims["sub"] == "local"
    assert claims["auth_kind"] == "api_key"
