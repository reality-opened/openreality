"""Dispatch/gating tests for the in-scan progressive scene report (Phase 3).

Exercises ``server.app`` glue — the finalize-suppression invariant and the
throttle — via the conftest ``load_app_module`` stub (no Flask/GPU/LLM)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import types

from conftest import load_app_module

from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport


class _ImmediateExecutor:
    """Runs ``submit`` synchronously so ``loop.run_in_executor`` is deterministic."""

    def submit(self, fn, *args, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut


def _install_build_fakes(app_mod, num_submaps: int = 2) -> dict:
    """Wire fake slam_processor/extractor/builder so the build path can run."""
    app_mod._scene_report_finalized = False
    app_mod._progressive_report = None
    app_mod._progressive_report_inflight = False

    app_mod.slam_processor = types.SimpleNamespace(
        _detection_lock=threading.Lock(),
        accumulated_detections=[],
        latest_scene_center=[0.0, 0.0, 0.0],
        solver=object(),  # no `.map` -> _latest_submap_id() falls back to None
    )

    facts = SceneFacts(metrics=SceneMetrics(num_submaps=num_submaps, num_keyframes=8))
    app_mod._scene_feature_extractor = types.SimpleNamespace(
        extract=lambda solver, detections, center: facts
    )

    tracker = {"calls": 0, "previous": "unset"}

    class _FakeBuilder:
        def build_incremental(self, solver, facts, scene_center=None, previous=None, **kwargs):
            tracker["calls"] += 1
            tracker["previous"] = previous
            return SceneReport(summary="live", progressive=True, final=False, facts=facts)

    app_mod._scene_report_builder = _FakeBuilder()
    return tracker


def test_progressive_build_stores_report_when_not_finalized(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    tracker = _install_build_fakes(app_mod)

    app_mod._build_and_emit_progressive_report()

    assert tracker["calls"] == 1
    assert app_mod._progressive_report is not None
    assert app_mod._progressive_report.summary == "live"
    assert app_mod._progressive_report.progressive is True
    # inflight is always released in the finally
    assert app_mod._progressive_report_inflight is False


def test_progressive_build_passes_previous_for_folding(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    tracker = _install_build_fakes(app_mod)
    prior = SceneReport(summary="earlier", progressive=True, final=False)
    app_mod._progressive_report = prior

    app_mod._build_and_emit_progressive_report()

    assert tracker["previous"] is prior  # running report is folded into the next build


def test_progressive_build_suppressed_once_finalized(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    tracker = _install_build_fakes(app_mod)
    app_mod._scene_report_finalized = True  # final report already started

    app_mod._build_and_emit_progressive_report()

    assert tracker["calls"] == 0  # builder never runs
    assert app_mod._progressive_report is None
    assert app_mod._progressive_report_inflight is False


def test_progressive_build_skips_empty_scan(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    _install_build_fakes(app_mod, num_submaps=0)

    app_mod._build_and_emit_progressive_report()

    assert app_mod._progressive_report is None  # nothing scanned yet


def test_maybe_dispatch_progressive_throttle_and_gates(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    calls: list[int] = []
    app_mod._build_and_emit_progressive_report = lambda: calls.append(1)
    app_mod._agent_executor = _ImmediateExecutor()
    app_mod.slam_processor = types.SimpleNamespace()  # not None
    app_mod.SCENE_REPORT_PROGRESSIVE = True
    app_mod.SCENE_REPORT_PROGRESSIVE_EVERY_N = 2
    app_mod.SCENE_REPORT_PROGRESSIVE_MIN_INTERVAL_S = 12.0
    app_mod._scene_report_finalized = False
    app_mod._progressive_report_inflight = False
    app_mod._progressive_report_last_submaps = 0
    app_mod._progressive_report_last_ts = 0.0

    async def run():
        # 1 submap: fewer than EVERY_N new submaps -> no dispatch
        app_mod._maybe_dispatch_progressive_report({"num_submaps": 1})
        assert calls == []

        # 2 submaps + interval elapsed (last_ts == 0) -> dispatch
        app_mod._maybe_dispatch_progressive_report({"num_submaps": 2})
        assert calls == [1]
        assert app_mod._progressive_report_inflight is True
        assert app_mod._progressive_report_last_submaps == 2

        # a build is in flight (stub didn't clear it) -> skip even with more submaps
        app_mod._maybe_dispatch_progressive_report({"num_submaps": 6})
        assert calls == [1]

        # finalized gate beats the throttle entirely
        app_mod._progressive_report_inflight = False
        app_mod._progressive_report_last_ts = 0.0
        app_mod._scene_report_finalized = True
        app_mod._maybe_dispatch_progressive_report({"num_submaps": 20})
        assert calls == [1]

    asyncio.run(run())
