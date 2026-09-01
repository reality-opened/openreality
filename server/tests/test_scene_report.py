"""Contract tests for SceneReportBuilder + keyframe selection (no LLM/GPU)."""

from __future__ import annotations

import pytest

from server.scene_report.keyframes import choose_frame_refs, encode_frames
from server.scene_report.report import SceneReportBuilder
from server.scene_report.schemas import (
    EvidenceRef,
    ObjectInstance,
    ReportObject,
    SceneFacts,
    SceneMetrics,
    SceneReport,
)


class _Resp:
    def __init__(self, model="fake/model", degraded=False):
        self.model = model
        self.degraded = degraded


class FakeLLM:
    """chat_json that returns a canned grounded report."""

    def __init__(self, payload=None, raise_exc=None):
        self.payload = payload
        self.raise_exc = raise_exc
        self.calls = []

    def chat_json(self, system_prompt, user_prompt, images_b64=None, **kwargs):
        self.calls.append(
            {"images": list(images_b64 or []), "user": user_prompt, "kwargs": dict(kwargs)}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.payload, _Resp()


class FakeMap:
    def __init__(self):
        self._submaps = {}

    def ordered_submaps_by_key(self):
        return iter([])

    def get_submap(self, sid):
        return None


class FakeSolver:
    def __init__(self):
        self.map = FakeMap()
        self.graph = object()


class _CaptureSubmap:
    """Minimal submap for exercising the capture-coverage spread tier: just enough of the
    real ``Submap`` surface (``get_id``/``get_lc_status``/``get_frame_ids``) for
    ``_spread_frame_positions`` -- no points/poses needed since that tier never touches them."""

    def __init__(self, sid, n_frames, lc=False):
        self._id = sid
        self._n_frames = n_frames
        self._lc = lc

    def get_id(self):
        return self._id

    def get_lc_status(self):
        return self._lc

    def get_frame_ids(self):
        return list(range(self._n_frames))


class _CaptureMap:
    def __init__(self, submaps):
        self._submaps = list(submaps)

    def ordered_submaps_by_key(self):
        return iter(sorted(self._submaps, key=lambda s: s.get_id()))

    def get_submap(self, sid):
        for s in self._submaps:
            if s.get_id() == sid:
                return s
        return None


class _CaptureSolver:
    def __init__(self, submaps):
        self.map = _CaptureMap(submaps)
        self.graph = object()


def _facts_with_objects():
    return SceneFacts(
        metrics=SceneMetrics(num_submaps=3, num_keyframes=12),
        objects=[
            ObjectInstance(query="chair", center=[1, 0, 0], confidence=0.9,
                           evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),
            ObjectInstance(query="desk", center=[2, 0, 0], confidence=0.7,
                           evidence=[EvidenceRef(submap_id=1, frame_idx=0)]),
        ],
        object_counts={"chair": 1, "desk": 1},
    )


def test_build_final_parses_grounded_report():
    payload = {
        "summary": "A small office.",
        "room_type": "Office",
        "objects": [
            {"name": "chair", "location": "by the desk", "evidence": [{"submap_id": 0, "frame_idx": 2}]},
            {"name": "desk", "location": "against the wall", "evidence": []},
        ],
        "observations": ["The chair sits ~1 unit from the desk."],
        "coverage_note": "Far corner not scanned.",
    }
    llm = FakeLLM(payload=payload)
    report = SceneReportBuilder(llm, max_keyframes=12).build_final(FakeSolver(), _facts_with_objects())

    assert isinstance(report, SceneReport)
    assert report.summary == "A small office."
    assert report.room_type == "office"  # normalized
    assert report.degraded is False
    assert report.model == "fake/model"
    assert [o.name for o in report.objects] == ["chair", "desk"]
    assert report.objects[0].evidence[0].submap_id == 0
    # facts are attached to the report for downstream grounding
    assert report.facts.metrics.num_submaps == 3
    assert len(llm.calls) == 1


def test_fallback_when_no_llm_client():
    report = SceneReportBuilder(None).build_final(FakeSolver(), _facts_with_objects())
    assert report.degraded is True
    assert sorted(o.name for o in report.objects) == ["chair", "desk"]
    assert report.facts.metrics.num_keyframes == 12


def test_fallback_on_llm_error():
    llm = FakeLLM(raise_exc=RuntimeError("boom"))
    report = SceneReportBuilder(llm).build_final(FakeSolver(), _facts_with_objects())
    assert report.degraded is True
    assert report.observations  # carries the "model unavailable" note


def test_bad_llm_output_falls_back():
    llm = FakeLLM(payload=["not", "a", "dict"])
    report = SceneReportBuilder(llm).build_final(FakeSolver(), _facts_with_objects())
    assert report.degraded is True


def test_choose_frame_refs_prioritizes_evidence_and_dedups():
    objects = [
        ObjectInstance(query="a", center=[0, 0, 0], confidence=0.9,
                       evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),
        ObjectInstance(query="b", center=[0, 0, 0], confidence=0.8,
                       evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),  # duplicate ref
        ObjectInstance(query="c", center=[0, 0, 0], confidence=0.7,
                       evidence=[EvidenceRef(submap_id=1, frame_idx=4)]),
    ]
    refs = choose_frame_refs(objects, FakeSolver(), max_frames=12)
    pairs = [(r.submap_id, r.frame_idx) for r in refs]
    assert pairs == [(0, 2), (1, 4)]  # deduped, highest-confidence first


def test_choose_frame_refs_respects_budget():
    objects = [
        ObjectInstance(query=f"q{i}", center=[0, 0, 0], confidence=1.0,
                       evidence=[EvidenceRef(submap_id=i, frame_idx=0)])
        for i in range(20)
    ]
    refs = choose_frame_refs(objects, FakeSolver(), max_frames=5)
    assert len(refs) == 5


def test_encode_frames_is_safe_without_real_submaps():
    refs = [EvidenceRef(submap_id=0, frame_idx=0)]
    assert encode_frames(FakeSolver(), refs) == []


# -- progressive / incremental report (Phase 3) -----------------------------


def _previous_report():
    return SceneReport(
        summary="Partial scan so far.",
        room_type="office",
        objects=[ReportObject(name="chair", location="on the left")],
        observations=["one chair seen"],
        facts=_facts_with_objects(),
        progressive=True,
        final=False,
    )


def test_build_incremental_marks_progressive_and_parses():
    payload = {
        "summary": "Office, partially scanned.",
        "room_type": "Office",
        "objects": [
            {"name": "chair", "location": "by the desk", "evidence": [{"submap_id": 0, "frame_idx": 2}]}
        ],
        "observations": ["A chair sits near a desk."],
        "coverage_note": "Far side not yet scanned.",
    }
    llm = FakeLLM(payload=payload)
    report = SceneReportBuilder(llm).build_incremental(FakeSolver(), _facts_with_objects())

    assert report.progressive is True
    assert report.final is False
    assert report.degraded is False
    assert report.summary == "Office, partially scanned."
    assert report.room_type == "office"  # normalized
    assert [o.name for o in report.objects] == ["chair"]
    assert report.facts.metrics.num_submaps == 3
    assert len(llm.calls) == 1


def test_build_incremental_fallback_without_llm():
    report = SceneReportBuilder(None).build_incremental(FakeSolver(), _facts_with_objects())
    assert report.progressive is True
    assert report.final is False
    assert report.degraded is True
    assert "in progress" in report.summary.lower()
    # with no prior report, falls back to the fact-only inventory
    assert sorted(o.name for o in report.objects) == ["chair", "desk"]


def test_build_incremental_fallback_keeps_previous_prose():
    prev = _previous_report()
    report = SceneReportBuilder(None).build_incremental(
        FakeSolver(), _facts_with_objects(), previous=prev
    )
    assert report.progressive is True
    assert report.final is False
    assert report.degraded is True
    # keeps the prior prose rather than regressing to the fact-only inventory
    assert report.summary == prev.summary
    assert [o.name for o in report.objects] == ["chair"]
    assert report.facts.metrics.num_submaps == 3


def test_build_incremental_folds_previous_into_prompt():
    llm = FakeLLM(payload={"summary": "x", "objects": []})
    prev = _previous_report()
    SceneReportBuilder(llm).build_incremental(FakeSolver(), _facts_with_objects(), previous=prev)
    assert len(llm.calls) == 1
    # the running report is handed to the model so it can revise rather than restart
    assert "Partial scan so far." in llm.calls[0]["user"]


def test_choose_frame_refs_prefers_given_submaps():
    objects = [
        ObjectInstance(query="a", center=[0, 0, 0], confidence=0.9,
                       evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),
    ]
    refs = choose_frame_refs(objects, FakeSolver(), max_frames=5, prefer_submap_ids=[7])
    pairs = [(r.submap_id, r.frame_idx) for r in refs]
    assert pairs[0] == (7, 0)  # preferred (newest) submap front-loaded
    assert (0, 2) in pairs  # object evidence still included


# -- capture-coverage keyframe budget (decoupled from window/submap count) -----------------


def test_choose_frame_refs_many_narrow_windows_hits_budget():
    """~15 windows x 9 frames (ss=8-style capture) -- budget should still be met, not capped
    at "one frame per submap" (which would also happen to equal 12 here only by coincidence;
    the point is this no longer depends on submap count at all -- see the few-windows case)."""
    submaps = [_CaptureSubmap(sid=i * 9, n_frames=9) for i in range(15)]
    refs = choose_frame_refs([], _CaptureSolver(submaps), max_frames=12)
    assert len(refs) == 12


def test_choose_frame_refs_few_wide_windows_hits_budget():
    """~4 windows x 33 frames (ss=32-style capture, same total capture length as the
    many-narrow-windows case). The OLD "one frame per submap" fallback would cap this at 4
    refs; the capture-coverage spread must still reach the budget."""
    submaps = [_CaptureSubmap(sid=i * 33, n_frames=33) for i in range(4)]
    refs = choose_frame_refs([], _CaptureSolver(submaps), max_frames=12)
    assert len(refs) == 12


def test_choose_frame_refs_spread_excludes_lc_submaps():
    submaps = [
        _CaptureSubmap(sid=0, n_frames=9),
        _CaptureSubmap(sid=9, n_frames=2, lc=True),
        _CaptureSubmap(sid=11, n_frames=9),
    ]
    refs = choose_frame_refs([], _CaptureSolver(submaps), max_frames=12)
    assert all(r.submap_id != 9 for r in refs)


def test_choose_frame_refs_spread_valid_submap_and_frame_idx():
    """Every produced ref must resolve to a real (submap_id, frame_idx) pair a consumer
    (encode_frames) can look up -- frame_idx within that submap's own frame count."""
    submaps = [_CaptureSubmap(sid=0, n_frames=5), _CaptureSubmap(sid=5, n_frames=20)]
    by_id = {s.get_id(): s._n_frames for s in submaps}
    refs = choose_frame_refs([], _CaptureSolver(submaps), max_frames=12)
    assert len(refs) == 12
    for r in refs:
        assert r.submap_id in by_id
        assert 0 <= r.frame_idx < by_id[r.submap_id]


def test_choose_frame_refs_spread_fills_remaining_budget_after_evidence():
    """Evidence-tier refs count against the budget; the spread tier only fills what's left."""
    objects = [
        ObjectInstance(query="a", center=[0, 0, 0], confidence=0.9,
                       evidence=[EvidenceRef(submap_id=0, frame_idx=2)]),
        ObjectInstance(query="b", center=[0, 0, 0], confidence=0.8,
                       evidence=[EvidenceRef(submap_id=5, frame_idx=1)]),
    ]
    submaps = [_CaptureSubmap(sid=0, n_frames=9), _CaptureSubmap(sid=9, n_frames=9)]
    refs = choose_frame_refs(objects, _CaptureSolver(submaps), max_frames=6)
    assert len(refs) == 6
    pairs = [(r.submap_id, r.frame_idx) for r in refs]
    assert (0, 2) in pairs and (5, 1) in pairs


def test_choose_frame_refs_uses_env_default_when_max_frames_omitted(monkeypatch):
    monkeypatch.setenv("SCENE_REPORT_KEYFRAMES", "5")
    submaps = [_CaptureSubmap(sid=i * 9, n_frames=9) for i in range(15)]
    refs = choose_frame_refs([], _CaptureSolver(submaps))
    assert len(refs) == 5


def test_choose_frame_refs_default_budget_is_12_without_env(monkeypatch):
    monkeypatch.delenv("SCENE_REPORT_KEYFRAMES", raising=False)
    submaps = [_CaptureSubmap(sid=i * 9, n_frames=9) for i in range(15)]
    refs = choose_frame_refs([], _CaptureSolver(submaps))
    assert len(refs) == 12


# -- final-report object budget + token knobs (SCENE_REPORT_OBJECT_BUDGET / _MAX_TOKENS) --


def _minimal_payload():
    return {"summary": "s", "room_type": "office", "objects": [], "observations": [],
            "coverage_note": ""}


def test_object_budget_env_adds_prompt_instruction(monkeypatch):
    monkeypatch.setenv("SCENE_REPORT_OBJECT_BUDGET", "20")
    llm = FakeLLM(payload=_minimal_payload())
    SceneReportBuilder(llm, max_keyframes=2).build_final(FakeSolver(), _facts_with_objects())
    assert "20 most notable DISTINCT object types" in llm.calls[0]["user"]


def test_no_object_budget_keeps_curated_prompt(monkeypatch):
    monkeypatch.delenv("SCENE_REPORT_OBJECT_BUDGET", raising=False)
    llm = FakeLLM(payload=_minimal_payload())
    SceneReportBuilder(llm, max_keyframes=2).build_final(FakeSolver(), _facts_with_objects())
    assert "DISTINCT object types" not in llm.calls[0]["user"]


def test_report_max_tokens_env_controls_final_call(monkeypatch):
    monkeypatch.setenv("SCENE_REPORT_MAX_TOKENS", "2400")
    llm = FakeLLM(payload=_minimal_payload())
    SceneReportBuilder(llm, max_keyframes=2).build_final(FakeSolver(), _facts_with_objects())
    assert llm.calls[0]["kwargs"]["max_tokens"] == 2400
    monkeypatch.delenv("SCENE_REPORT_MAX_TOKENS", raising=False)
    llm2 = FakeLLM(payload=_minimal_payload())
    SceneReportBuilder(llm2, max_keyframes=2).build_final(FakeSolver(), _facts_with_objects())
    assert llm2.calls[0]["kwargs"]["max_tokens"] == 1200
