"""Grounded scene Q&A tests (Phase 5).

Exercises ``server.app``'s QA grounding helpers through the conftest
``load_app_module`` stub (no Flask/GPU/network). Focuses on the bits with real
logic: question→object relevance ranking and validating the LLM's ``focus``
target against the persisted inventory (never trusting an invented coordinate).
"""

from __future__ import annotations

import base64
import types

import pytest

from conftest import load_app_module


def _objects():
    return [
        {"query": "keys", "center": [1.0, 0.5, 2.0], "confidence": 0.8,
         "evidence": [{"submap_id": 0, "frame_idx": 3}]},
        {"query": "coffee mug", "center": [3.0, 0.4, 1.0], "confidence": 0.9,
         "evidence": [{"submap_id": 1, "frame_idx": 0}]},
        {"query": "backpack", "center": [0.0, 0.0, 5.0], "confidence": 0.6, "evidence": []},
    ]


def _record():
    return {
        "scan_id": "scan1",
        "created_at": 1.0,
        "report": {"summary": "a desk area", "observations": ["two items on the desk"]},
        "facts": {"objects": _objects(), "relations": []},
        "keyframes": [
            {"submap_id": 0, "frame_idx": 3, "blob_key": "0_3.jpg"},
            {"submap_id": 1, "frame_idx": 0, "blob_key": "1_0.jpg"},
        ],
        "points_key": "points.bin",
        "point_count": 0,
        "scene_center": [0.0, 0.0, 0.0],
    }


class _FakePersistence:
    def __init__(self, record):
        self._record = record

    def get_scene(self, user_id, scan_id):
        return self._record

    def get_keyframe(self, user_id, scan_id, blob_key):
        return b"\xff\xd8" + blob_key.encode()


class _Resp:
    def __init__(self, model="fake/model", degraded=False):
        self.model = model
        self.degraded = degraded


class _FakeClient:
    """Captures the prompt and returns a canned grounded answer. ``evidence`` is the
    keyframe(s) the model 'cites' — overridable so tests can prove the shown images
    follow the model's choice rather than a fixed top-3."""

    def __init__(self, focus_name="keys", evidence=None):
        self.focus_name = focus_name
        self.evidence = (
            evidence if evidence is not None else [{"submap_id": 0, "frame_idx": 3}]
        )
        self.last = {}

    def chat_json(self, system_prompt, user_prompt, images_b64=None, **kwargs):
        self.last = {"user": user_prompt, "images": list(images_b64 or [])}
        return (
            {
                "answer": "Your keys are on the desk near the mug.",
                "focus": {"name": self.focus_name},
                "evidence": self.evidence,
            },
            _Resp(),
        )


def test_relevant_objects_ranks_by_question_overlap(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    ranked = app_mod._relevant_scene_objects(_objects(), "where are my keys?")
    assert ranked[0]["query"] == "keys"  # keyword overlap beats higher-confidence mug


def test_relevant_objects_falls_back_to_confidence(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    ranked = app_mod._relevant_scene_objects(_objects(), "what is in here?")
    assert ranked[0]["query"] == "coffee mug"  # no overlap -> top confidence


def test_answer_validates_focus_against_inventory(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient(focus_name="keys")
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    monkeypatch.setattr(app_mod, "_scene_persistence", _FakePersistence(_record()))

    out = app_mod._answer_scene_question("u", "scan1", _record(), "where are my keys?", [])

    assert "keys" in out["answer"].lower()
    # focus center comes from the stored inventory, not the LLM payload.
    assert out["focus"]["name"] == "keys"
    assert out["focus"]["center"] == [1.0, 0.5, 2.0]
    # evidence keyframes are resolved + at least one image attached for grounding.
    assert any(e["blob_key"] == "0_3.jpg" for e in out["evidence"])
    assert len(client.last["images"]) >= 1


def test_answer_drops_invented_focus(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient(focus_name="teleporter")  # not in inventory
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    monkeypatch.setattr(app_mod, "_scene_persistence", _FakePersistence(_record()))

    out = app_mod._answer_scene_question("u", "scan1", _record(), "where is the teleporter?", [])
    assert out["focus"] is None  # never fabricate a location not in the facts


def test_answer_attaches_decoded_keyframes(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    monkeypatch.setattr(app_mod, "_scene_persistence", _FakePersistence(_record()))

    app_mod._answer_scene_question("u", "scan1", _record(), "where are my keys?", [])
    # The fake persistence returns raw bytes; the helper must base64-encode them.
    for img in client.last["images"]:
        base64.b64decode(img)  # round-trips without error


# ── Shared grounded core + live summary chat (convergence) ──


def _facts_with_metrics():
    return {
        "objects": _objects(),
        "relations": [{"a": "keys", "b": "coffee mug", "distance": 2.0, "relation": "near"}],
        "metrics": {"dimensions": [4.0, 3.0, 2.5]},
    }


def _report_dict():
    return {
        "summary": "a small desk area",
        "room_type": "office",
        "observations": ["keys and a mug share the desk"],
        "facts": _facts_with_metrics(),
    }


def test_grounded_core_grounds_prompt_in_3d_facts(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient(focus_name="keys")
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)

    out = app_mod._grounded_scene_answer(
        _facts_with_metrics(), _report_dict(), "where are my keys?", []
    )
    prompt = client.last["user"]
    # The spatial grounding the summary chat used to lack now reaches the model:
    # world-frame coordinates, pairwise relations, and room extents.
    assert "OBJECT INVENTORY" in prompt and "1.0" in prompt  # keys' world-frame center
    assert "RELATIONS" in prompt and "near" in prompt
    assert "ROOM SIZE" in prompt and "4.0" in prompt
    # focus center comes from our stored inventory, not the model's payload.
    assert out["focus"]["center"] == [1.0, 0.5, 2.0]
    # No image_loader -> text-grounded path: no keyframes attached.
    assert out["evidence"] == []
    assert not client.last["images"]


def test_grounded_core_uses_image_loader_when_provided(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)

    refs = [{"submap_id": 0, "frame_idx": 3, "blob_key": "0_3.jpg"}]
    out = app_mod._grounded_scene_answer(
        _facts_with_metrics(),
        _report_dict(),
        "where are my keys?",
        [],
        lambda relevant: (["aGVsbG8="], refs),
    )
    assert out["evidence"] == refs
    assert client.last["images"] == ["aGVsbG8="]


def _call_assistant_chat(app_mod, monkeypatch, payload):
    monkeypatch.setattr(
        app_mod, "request", types.SimpleNamespace(get_json=lambda: payload)
    )
    resp = app_mod.assistant_chat()
    if isinstance(resp, tuple):  # (body, status) on error
        resp = resp[0]
    # conftest's FakeFlask.jsonify wraps the dict as {"args": (dict,), ...}.
    return resp["args"][0]


def test_assistant_chat_degrades_without_report(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "_last_scene_report", None)
    monkeypatch.setattr(app_mod, "_progressive_report", None)

    out = _call_assistant_chat(app_mod, monkeypatch, {"message": "where are my keys?"})
    assert out["degraded"] is True
    assert out["focus"] is None and out["evidence"] == []


def test_assistant_chat_grounds_in_cached_report(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient(focus_name="keys")
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    monkeypatch.setattr(app_mod, "_last_scene_report", _report_dict())

    out = _call_assistant_chat(app_mod, monkeypatch, {"message": "where are my keys?"})
    # Grounded server-side from the worker's cached facts, not client-sent context.
    assert "1.0" in client.last["user"]
    assert out["focus"]["center"] == [1.0, 0.5, 2.0]


def test_assistant_chat_falls_back_to_progressive_report(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient(focus_name="keys")
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    # Mid-scan: no final report yet, but a live progressive one exists.
    monkeypatch.setattr(app_mod, "_last_scene_report", None)
    monkeypatch.setattr(
        app_mod,
        "_progressive_report",
        types.SimpleNamespace(model_dump=lambda: _report_dict()),
    )

    out = _call_assistant_chat(app_mod, monkeypatch, {"message": "where are my keys?"})
    assert out["focus"]["name"] == "keys"


# ── Cited evidence tracks the model's choice (the "same 3 pictures" fix) ──


def test_select_evidence_honors_model_order_and_falls_back(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    resolved = [
        {"submap_id": 0, "frame_idx": 3, "blob_key": "0_3.jpg"},
        {"submap_id": 1, "frame_idx": 0, "blob_key": "1_0.jpg"},
    ]
    # The model's cited frame is surfaced, even when it isn't the ranker's first.
    picked = app_mod._select_evidence([{"submap_id": 1, "frame_idx": 0}], resolved)
    assert [r["blob_key"] for r in picked] == ["1_0.jpg"]
    # An invented / unknown ref is dropped, and with nothing valid left we fall back
    # to the resolved order so the UI never shows zero evidence when frames exist.
    fb = app_mod._select_evidence([{"submap_id": 9, "frame_idx": 9}], resolved)
    assert [r["blob_key"] for r in fb] == ["0_3.jpg", "1_0.jpg"]
    # Evidence never carries internal fields like "shows" to the client.
    assert all(set(r) == {"submap_id", "frame_idx", "blob_key"} for r in picked)


def test_answer_evidence_tracks_model_citation(monkeypatch):
    """The reported bug: every question returned the same first keyframe. Now the
    surfaced image follows the frame the model actually cites."""
    app_mod = load_app_module(monkeypatch)
    # Model answers about the mug and cites the mug's keyframe (1, 0), not keys' (0, 3).
    client = _FakeClient(focus_name="coffee mug", evidence=[{"submap_id": 1, "frame_idx": 0}])
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    monkeypatch.setattr(app_mod, "_scene_persistence", _FakePersistence(_record()))

    out = app_mod._answer_scene_question("u", "scan1", _record(), "where is the mug?", [])
    assert [e["blob_key"] for e in out["evidence"]] == ["1_0.jpg"]
    # Both candidate keyframes were attached for grounding, but only the cited one shows.
    assert len(client.last["images"]) == 2


def test_candidate_keyframes_dedupes_and_caps(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "SCENE_QA_CANDIDATE_IMAGES", 1)
    relevant = [
        {"query": "keys", "evidence": [{"submap_id": 0, "frame_idx": 3}]},
        {"query": "mug", "evidence": [{"submap_id": 1, "frame_idx": 0}]},
        {"query": "dupe", "evidence": [{"submap_id": 0, "frame_idx": 3}]},
        {"query": "empty", "evidence": []},
    ]
    cands = app_mod._candidate_keyframes(relevant)
    assert cands == [{"submap_id": 0, "frame_idx": 3, "shows": "keys"}]  # capped at 1


# ── Strengthened relevance: stem / substring / relation+report context ──


def test_relevant_objects_matches_via_substring(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = [
        {"query": "whiteboard", "confidence": 0.4, "evidence": []},
        {"query": "lamp", "confidence": 0.95, "evidence": []},
    ]
    # "board" is a substring of "whiteboard" -> beats the higher-confidence lamp.
    ranked = app_mod._relevant_scene_objects(objs, "what's written on the board?")
    assert ranked[0]["query"] == "whiteboard"


def test_relevant_objects_matches_via_stemming(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = [
        {"query": "potted plant", "confidence": 0.3, "evidence": []},
        {"query": "lamp", "confidence": 0.95, "evidence": []},
    ]
    # "plants" stems to "plant" -> matches "potted plant" despite lower confidence.
    ranked = app_mod._relevant_scene_objects(objs, "are there any plants?")
    assert ranked[0]["query"] == "potted plant"


def test_relevant_objects_uses_relation_context(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = [
        {"query": "remote", "confidence": 0.3, "evidence": []},
        {"query": "lamp", "confidence": 0.9, "evidence": []},
    ]
    relations = [{"a": "remote", "b": "television", "relation": "near", "distance": 0.5}]
    # The question names 'television' (not in the inventory); the relation lifts 'remote'.
    ranked = app_mod._relevant_scene_objects(
        objs, "what is near the television?", relations=relations
    )
    assert ranked[0]["query"] == "remote"


def test_relevant_objects_uses_report_context(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = [
        {"query": "mug", "confidence": 0.3, "evidence": []},
        {"query": "lamp", "confidence": 0.9, "evidence": []},
    ]
    report = {"observations": ["the mug is chipped and ceramic"]}
    # 'ceramic' isn't in any name, but the report ties it to the mug.
    ranked = app_mod._relevant_scene_objects(
        objs, "is anything ceramic?", report=report
    )
    assert ranked[0]["query"] == "mug"


# ── Embedding similarity tie-break (Feature 2) ──


def test_relevant_objects_embedding_tiebreak_overrides_confidence(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = _objects()  # keys 0.8, coffee mug 0.9, backpack 0.6
    # No lexical hit -> ranking is pure tie-break. Embedding sim favors the backpack.
    sims = {id(objs[0]): 0.10, id(objs[1]): 0.20, id(objs[2]): 0.99}
    ranked = app_mod._relevant_scene_objects(
        objs, "somewhere to stash my gear", sim_by_id=sims
    )
    assert ranked[0]["query"] == "backpack"  # beats top-confidence mug via similarity


def test_embedding_similarities_success_and_degrade(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    objs = _objects()

    class _EmbClient:
        def embed(self, texts, model):
            # question + 3 objects; question aligns with the first object only.
            assert len(texts) == 4
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: _EmbClient())
    sims = app_mod._embedding_similarities("q", objs, ["t0", "t1", "t2"])
    assert sims[id(objs[0])] > sims[id(objs[1])]  # cos(q, obj0)=1 > cos(q, obj1)=0

    class _BoomClient:
        def embed(self, texts, model):
            raise RuntimeError("no embeddings here")

    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: _BoomClient())
    assert app_mod._embedding_similarities("q", objs, ["t0", "t1", "t2"]) is None


def test_grounded_core_embed_tiebreak_reorders_inventory(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    monkeypatch.setattr(app_mod, "SCENE_QA_EMBED_TIEBREAK", True)
    client = _FakeClient(focus_name="backpack")
    # facts objects order: keys, coffee mug, backpack. Make the question embed nearest
    # the backpack; with no lexical hit, the inventory should lead with backpack.
    client.embed = lambda texts, model: [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)

    app_mod._grounded_scene_answer(
        _facts_with_metrics(), _report_dict(), "where can i stash my things?", []
    )
    prompt = client.last["user"]
    assert prompt.index('"name": "backpack"') < prompt.index('"name": "keys"')


# -- metric grounding (anchor + ground frame) --------------------------------------

def _anchored_record(*, scale=2.0, vertical=True, ceiling=True):
    rec = _record()
    rec["derived_latest"] = {
        "kind": "anchor",
        "scale_factor": scale,
        "source_key": "derived/anchor/123_abc",
    }
    rec["facts"]["relations"] = [{"a": "keys", "b": "coffee mug", "relation": "near", "distance": 1.5}]
    rec["facts"]["metrics"] = {
        "dimensions": [3.0, 2.0, 1.0],
        "vertical_axis_known": vertical,
        "floor_extent": [2.5, 2.0],
        "floor_area": 4.0,
        "room_height": 1.2 if ceiling else None,
    }
    return rec


def _ask(monkeypatch, record, question="how wide is the room?"):
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    app_mod.configure_scene_persistence(_FakePersistence(record))
    captured = {}
    orig = client.chat_json

    def chat_json(system_prompt, user_prompt, **kw):
        captured["system"] = system_prompt
        return orig(system_prompt, user_prompt, **kw)

    client.chat_json = chat_json
    app_mod._answer_scene_question("u", "scan1", record, question, [])
    return captured["system"], client.last["user"]


def test_qa_unanchored_scene_is_told_relative_units_and_forbids_metres(monkeypatch):
    system, user = _ask(monkeypatch, _record())
    assert "RELATIVE UNITS" in system and "never assert metric distances" in system
    assert "UNITS: relative units (uncalibrated)" in user
    assert "metric anchor applied" not in user
    # raw coordinates pass through unscaled
    assert '"center": [1.0, 0.5, 2.0]' in user
    assert "floor frame not fitted" in user


def test_qa_anchored_scene_gets_metres_and_scaled_lengths(monkeypatch):
    system, user = _ask(monkeypatch, _anchored_record(scale=2.0))
    assert "METRIC ANCHOR is applied" in system and "MAY state sizes and distances in metres" in system
    assert "UNITS: metres — metric anchor applied (metres; scale_source=anchor:derived/anchor/123_abc)" in user
    # centres, relation distances, room dims, floor extent all × scale
    assert '"center": [2.0, 1.0, 4.0]' in user
    assert '"distance": 3.0' in user
    assert "bounding box [6.0, 4.0, 2.0] m" in user
    assert "floor extent [5.0, 4.0] m" in user
    # area scales with the square of the factor
    assert "floor area 16.0 m²" in user
    assert "floor-to-ceiling 2.4 m" in user
    assert "Only state a floor-to-ceiling height if ROOM SIZE lists one" in system


def test_qa_anchored_without_ceiling_says_so(monkeypatch):
    system, user = _ask(monkeypatch, _anchored_record(ceiling=False))
    assert "ceiling not captured" in user
    assert "floor-to-ceiling 2.4" not in user


def test_qa_anchored_without_floor_frame_forbids_height(monkeypatch):
    system, user = _ask(monkeypatch, _anchored_record(vertical=False))
    assert "Do not assert a floor-to-ceiling height" in system
    assert "floor frame not fitted" in user


@pytest.mark.parametrize("pointer", [
    {"kind": "clamp", "scale_factor": 2.0, "source_key": "derived/clamp/x"},
    {"kind": "anchor", "scale_factor": 0.0, "source_key": "derived/anchor/x"},
    {"kind": "anchor", "scale_factor": "nan", "source_key": "derived/anchor/x"},
    {"kind": "anchor", "scale_factor": 2.0},
])
def test_qa_bad_or_non_anchor_pointers_stay_relative(monkeypatch, pointer):
    rec = _anchored_record()
    rec["derived_latest"] = pointer
    system, user = _ask(monkeypatch, rec)
    assert "RELATIVE UNITS" in system
    assert '"center": [1.0, 0.5, 2.0]' in user


def test_grounded_core_without_record_is_relative(monkeypatch):
    """The live summary chat passes no record — it must keep the relative wording."""
    app_mod = load_app_module(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(app_mod, "_get_assistant_client", lambda: client)
    app_mod._grounded_scene_answer({"objects": _objects(), "relations": []}, {"summary": "x"}, "where?", [])
    assert "UNITS: relative units (uncalibrated)" in client.last["user"]
