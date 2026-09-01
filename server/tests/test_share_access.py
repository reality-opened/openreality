"""Share-link access log — the behavioural measurement layer for customer discovery.

Covers ``server/share_access.py`` (events, visits, returned, questions, persistence
round-trip, never-raises) and the app hooks: a share-authorised read records an event,
share-authorised Q&A records the question, the mint route's ``access_id`` matches the one
the hook records, and a share token can NOT read the access route.
"""

from __future__ import annotations

import json
import types

import pytest

from conftest import load_app_module
from server.share_access import (
    DERIVED_KEY,
    RETURN_GAP_S,
    VISIT_GAP_S,
    ShareAccessLog,
    access_id_for_token,
)


class FakeStore:
    """Minimal derived-artifact store: bytes by (user, scan, key)."""

    def __init__(self):
        self.blobs: dict[tuple[str, str, str], bytes] = {}
        self.saves = 0

    def save_derived_artifact(self, user_id, scan_id, relative_key, data):
        self.blobs[(user_id, scan_id, "derived/" + relative_key)] = bytes(data)
        self.saves += 1
        return "derived/" + relative_key

    def get_derived_artifact(self, user_id, scan_id, derived_key):
        return self.blobs.get((user_id, scan_id, derived_key))


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _log(clock=None):
    store = FakeStore()
    clock = clock or Clock()
    return ShareAccessLog(store, clock=clock), store, clock


# -- module ------------------------------------------------------------------------

def test_access_id_is_stable_unique_and_opaque():
    a = access_id_for_token("tok-A")
    assert a == access_id_for_token("tok-A")
    assert a != access_id_for_token("tok-B")
    assert len(a) == 12 and "tok" not in a


def test_first_event_flushes_immediately_and_round_trips():
    log, store, clock = _log()
    ev = log.record(user_id="u", scan_id="s", token="tok", token_iat=1, token_exp=2,
                    endpoint="get_scene_route", remote_addr="1.2.3.4", user_agent="UA")
    assert ev["access_id"] == access_id_for_token("tok")
    assert store.saves == 1, "an 'opened' signal must hit disk on the first event"
    # no raw PII on disk
    raw = store.blobs[("u", "s", "derived/" + DERIVED_KEY)].decode()
    assert "1.2.3.4" not in raw and "UA" not in raw and "tok" not in raw
    # a fresh instance reloads it from the store
    fresh = ShareAccessLog(store, clock=clock)
    assert [e["access_id"] for e in fresh.events("u", "s")] == [ev["access_id"]]


def test_visits_and_returned_fold_correctly():
    log, store, clock = _log()
    t0 = clock.t
    kw = dict(user_id="u", scan_id="s", token="tok", token_iat=1, token_exp=2)
    log.record(endpoint="get_scene_route", **kw)
    clock.t = t0 + 60
    log.record(endpoint="get_scene_points_route", **kw)          # same visit
    clock.t = t0 + VISIT_GAP_S + 120
    log.record(endpoint="get_scene_route", **kw)                 # visit 2 (same day)
    clock.t = t0 + RETURN_GAP_S + 600
    log.record(endpoint="get_scene_route", **kw)                 # visit 3 → returned
    log.record(endpoint="scene_qa_route", method="POST", question="how wide is the hallway?", **kw)

    summary = log.summary("u", "s", now=clock.t)
    assert summary["event_count"] == 5
    (p,) = summary["prospects"]
    assert p["access_id"] == access_id_for_token("tok")
    assert p["opened"] is True and p["requests"] == 5
    assert p["visits"] == 3
    assert p["first_seen"] == t0 and p["last_seen"] == clock.t
    assert p["returned"] is True and p["returned_at"] == t0 + RETURN_GAP_S + 600
    assert [q["question"] for q in p["questions"]] == ["how wide is the hallway?"]
    assert p["endpoints"]["get_scene_route"] == 3


def test_two_links_are_two_prospects_and_same_day_revisit_is_not_returned():
    log, store, clock = _log()
    t0 = clock.t
    log.record(user_id="u", scan_id="s", token="tok-A", token_iat=1, token_exp=2, endpoint="get_scene_route")
    clock.t = t0 + VISIT_GAP_S + 1
    log.record(user_id="u", scan_id="s", token="tok-A", token_iat=1, token_exp=2, endpoint="get_scene_route")
    log.record(user_id="u", scan_id="s", token="tok-B", token_iat=5, token_exp=6, endpoint="get_scene_route")
    ps = {p["access_id"]: p for p in log.summary("u", "s")["prospects"]}
    assert len(ps) == 2
    a = ps[access_id_for_token("tok-A")]
    assert a["visits"] == 2 and a["returned"] is False
    assert ps[access_id_for_token("tok-B")]["visits"] == 1


def test_question_is_capped_and_flushed():
    log, store, clock = _log()
    log.record(user_id="u", scan_id="s", token="t", token_iat=1, token_exp=2,
               endpoint="scene_qa_route", method="POST", question="x" * 5000)
    (ev,) = log.events("u", "s")
    assert len(ev["question"]) == 500
    assert store.saves == 1


def test_record_never_raises_when_store_breaks():
    class Broken(FakeStore):
        def save_derived_artifact(self, *a, **k):
            raise RuntimeError("disk gone")

        def get_derived_artifact(self, *a, **k):
            raise RuntimeError("disk gone")

    log = ShareAccessLog(Broken(), clock=Clock())
    ev = log.record(user_id="u", scan_id="s", token="t", token_iat=1, token_exp=2, endpoint="e")
    assert ev is not None  # in-memory event still produced
    assert log.summary("u", "s")["event_count"] == 1


# -- app hooks ---------------------------------------------------------------------

def _fake_request(monkeypatch, app_mod, *, endpoint, scan_id, bearer=None, query=None, method="GET"):
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    headers["User-Agent"] = "pytest"
    req = types.SimpleNamespace(
        headers=headers,
        args=dict(query or {}),
        cookies={},
        url_rule=types.SimpleNamespace(endpoint=endpoint),
        view_args={"scan_id": scan_id},
        environ={},
        method=method,
        remote_addr="127.0.0.1",
    )
    monkeypatch.setattr(app_mod, "request", req)
    monkeypatch.setattr(app_mod, "_note_worker_activity", lambda: None)
    return req


def _wire(app_mod):
    store = FakeStore()
    app_mod.configure_scene_persistence(store)
    assert app_mod._share_access is not None
    return store


def test_share_authorised_read_is_recorded_with_access_id(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    store = _wire(app_mod)
    token, _ = app_mod._issue_share_token("scan_ok", "owner_9")
    req = _fake_request(monkeypatch, app_mod, endpoint="get_scene_route", scan_id="scan_ok", bearer=token)
    assert app_mod._try_share_token_auth() is True
    claims = req.environ[app_mod.AUTH_CLAIMS_ENV_KEY]
    assert claims["access_id"] == access_id_for_token(token)
    assert req.environ[app_mod.SHARE_TOKEN_ENV_KEY] == token
    events = app_mod._share_access.events("owner_9", "scan_ok")
    assert len(events) == 1 and events[0]["endpoint"] == "get_scene_route"
    assert events[0]["access_id"] == access_id_for_token(token)
    assert store.saves == 1  # first open persisted immediately


def test_qa_route_records_the_question_not_the_auth_hook(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _wire(app_mod)
    token, _ = app_mod._issue_share_token("scan_q", "owner_q")
    req = _fake_request(monkeypatch, app_mod, endpoint="scene_qa_route", scan_id="scan_q",
                        bearer=token, method="POST")
    assert app_mod._try_share_token_auth() is True
    assert app_mod._share_access.events("owner_q", "scan_q") == []  # deferred to the route
    app_mod._record_share_access(req.environ[app_mod.AUTH_CLAIMS_ENV_KEY], token,
                                 question="where is the water heater?")
    (ev,) = app_mod._share_access.events("owner_q", "scan_q")
    assert ev["question"] == "where is the water heater?" and ev["method"] == "POST"


def test_share_token_cannot_read_the_access_route(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _wire(app_mod)
    token, _ = app_mod._issue_share_token("scan_ok", "owner_9")
    _fake_request(monkeypatch, app_mod, endpoint="share_access_route", scan_id="scan_ok", bearer=token)
    with pytest.raises(app_mod.AuthError):
        app_mod._try_share_token_auth()


def test_no_log_when_persistence_absent(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    app_mod.configure_scene_persistence(None)
    token, _ = app_mod._issue_share_token("scan_ok", "owner_9")
    _fake_request(monkeypatch, app_mod, endpoint="get_scene_route", scan_id="scan_ok", bearer=token)
    assert app_mod._try_share_token_auth() is True  # logging absence never blocks auth
