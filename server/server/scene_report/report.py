"""LLM synthesis of a grounded scene report from SceneFacts + keyframes.

The builder takes an injected LLM client (``OpenRouterClient`` shape: a
``chat_json(system_prompt, user_prompt, images_b64, ...)`` returning
``(dict, response)``). If the client is ``None`` or the call fails, it returns a
fact-only fallback report so the user always gets something.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from server.scene_report.keyframes import choose_frame_refs, encode_frames
from server.scene_report.schemas import EvidenceRef, ReportObject, SceneFacts, SceneReport


def _report_object_budget() -> int:
    """``SCENE_REPORT_OBJECT_BUDGET`` (default 0 = curated highlights, the LLM's editorial
    choice). >0 instructs the final report to cover that many distinct grounded object
    types with counts — the grounded inventory in ``facts`` can far exceed what the LLM
    volunteers on its own (observed: 103 grounded instances → 6 highlight objects)."""
    try:
        return max(0, int(os.environ.get("SCENE_REPORT_OBJECT_BUDGET", "0")))
    except (TypeError, ValueError):
        return 0


def _report_max_tokens(default: int = 1200) -> int:
    """``SCENE_REPORT_MAX_TOKENS`` for the FINAL report call (default 1200). A larger
    object budget needs room; the incremental/live call keeps its own smaller budget."""
    try:
        return max(200, int(os.environ.get("SCENE_REPORT_MAX_TOKENS", str(default))))
    except (TypeError, ValueError):
        return default


class SceneReportBuilder:
    def __init__(self, llm_client, max_keyframes: int = 12):
        self.llm = llm_client
        self.max_keyframes = int(max_keyframes)

    def build_final(self, solver, facts: SceneFacts, scene_center=None) -> SceneReport:
        refs = choose_frame_refs(facts.objects, solver, self.max_keyframes)
        frames = encode_frames(solver, refs)
        images_b64 = [f["image_b64"] for f in frames if f.get("image_b64")]

        if self.llm is None:
            return self._fallback_report(facts, error="no LLM client configured")

        system_prompt, user_prompt = self._build_prompts(facts, frames)
        try:
            parsed, response = self.llm.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=images_b64,
                temperature=0.3,
                max_tokens=_report_max_tokens(),
            )
            return self._to_report(
                parsed,
                facts,
                model=getattr(response, "model", ""),
                degraded=bool(getattr(response, "degraded", False)),
            )
        except Exception as e:
            return self._fallback_report(facts, error=str(e))

    def build_incremental(
        self,
        solver,
        facts: SceneFacts,
        scene_center=None,
        previous: Optional[SceneReport] = None,
        max_keyframes: Optional[int] = None,
        prefer_submap_ids: Optional[list[int]] = None,
    ) -> SceneReport:
        """Refresh a *running* report mid-scan, folding the latest facts and a small,
        newest-biased set of keyframes into ``previous`` instead of re-summarizing
        from scratch.

        Decoupled from the autonomous mission loop: reads only the reconstruction
        (``facts`` + ``solver`` keyframes), so it works in reconstruction-only mode.
        Best-effort — never raises; falls back to a fact-refreshed progressive report.
        """
        budget = self.max_keyframes if max_keyframes is None else max(1, int(max_keyframes))
        refs = choose_frame_refs(facts.objects, solver, budget, prefer_submap_ids=prefer_submap_ids)
        frames = encode_frames(solver, refs)
        images_b64 = [f["image_b64"] for f in frames if f.get("image_b64")]

        if self.llm is None:
            return self._progressive_fallback(facts, previous)

        system_prompt, user_prompt = self._build_incremental_prompts(facts, frames, previous)
        try:
            parsed, response = self.llm.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=images_b64,
                temperature=0.3,
                max_tokens=900,
            )
            report = self._to_report(
                parsed,
                facts,
                model=getattr(response, "model", ""),
                degraded=bool(getattr(response, "degraded", False)),
            )
        except Exception as e:
            return self._progressive_fallback(facts, previous, error=str(e))

        report.progressive = True
        report.final = False
        return report

    # -- prompting ----------------------------------------------------

    def _build_prompts(self, facts: SceneFacts, frames: list[dict[str, Any]]) -> tuple[str, str]:
        system_prompt = (
            "You are a spatial intelligence reporter for a 3D scanning system. "
            "You are given STRUCTURED 3D FACTS derived from a SLAM reconstruction "
            "(object locations, sizes, pairwise distances, room extents, camera path) "
            "plus a few keyframes. Write a detailed report of the space. Ground every "
            "claim in the provided facts or visible keyframe content; never invent "
            "objects or measurements. Sizes and coordinates are up-to-scale, not "
            "guaranteed metric — phrase them as approximate/relative. Favor observations "
            "that require the 3D layout (relative positions, counts, how the space is "
            "organized), not things obvious from a single photo. Output strict JSON only."
        )
        frame_index = [
            {"submap_id": f.get("submap_id"), "frame_idx": f.get("frame_idx")} for f in frames
        ]
        object_budget = _report_object_budget()
        budget_line = ""
        if object_budget:
            budget_line = (
                f'\nIn "objects", cover the {object_budget} most notable DISTINCT object types '
                "grounded in the facts (object_counts is the authoritative list) — when a type "
                "has multiple instances, state the approximate count in its location text. Do "
                "not invent types absent from the facts.\n"
            )
        user_prompt = (
            "STRUCTURED FACTS (JSON):\n"
            f"{facts.model_dump_json()}\n\n"
            f"KEYFRAMES attached in this order: {json.dumps(frame_index)}\n"
            f"{budget_line}\n"
            "Return JSON exactly in this shape:\n"
            "{"
            '"summary":"2-4 sentence overview of the space",'
            '"room_type":"office|kitchen|bedroom|bathroom|hallway|living room|outdoor|other",'
            '"objects":[{"name":"...","location":"where it is in the space",'
            '"evidence":[{"submap_id":0,"frame_idx":0}]}],'
            '"observations":["fact-grounded observations a person could not get from one photo"],'
            '"coverage_note":"what was or was not well covered"'
            "}"
        )
        return system_prompt, user_prompt

    def _build_incremental_prompts(
        self, facts: SceneFacts, frames: list[dict[str, Any]], previous: Optional[SceneReport]
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a spatial intelligence reporter for a 3D scanning system, writing "
            "a LIVE report while the scan is still in progress. You are given the report "
            "SO FAR (may be null), updated STRUCTURED 3D FACTS from the reconstruction, "
            "and a few keyframes from the newest area. REVISE and EXTEND the running "
            "report to reflect what has been added — do not restate unchanged content "
            "verbatim, and do not drop previously reported objects unless the new facts "
            "contradict them. Ground every claim in the facts or visible keyframes; never "
            "invent objects or measurements. Sizes and coordinates are up-to-scale, not "
            "guaranteed metric — phrase them as approximate/relative. The scan is ongoing, "
            "so frame coverage as partial. Output strict JSON only."
        )
        prev_json = "null"
        if previous is not None:
            prev_json = json.dumps(
                {
                    "summary": previous.summary,
                    "room_type": previous.room_type,
                    "objects": [{"name": o.name, "location": o.location} for o in previous.objects],
                    "observations": previous.observations,
                }
            )
        frame_index = [
            {"submap_id": f.get("submap_id"), "frame_idx": f.get("frame_idx")} for f in frames
        ]
        user_prompt = (
            "REPORT SO FAR (JSON or null):\n"
            f"{prev_json}\n\n"
            "UPDATED STRUCTURED FACTS (JSON):\n"
            f"{facts.model_dump_json()}\n\n"
            f"NEW KEYFRAMES attached in this order: {json.dumps(frame_index)}\n\n"
            "Return JSON exactly in this shape:\n"
            "{"
            '"summary":"2-4 sentence overview of the space scanned so far",'
            '"room_type":"office|kitchen|bedroom|bathroom|hallway|living room|outdoor|other",'
            '"objects":[{"name":"...","location":"where it is in the space",'
            '"evidence":[{"submap_id":0,"frame_idx":0}]}],'
            '"observations":["fact-grounded observations a person could not get from one photo"],'
            '"coverage_note":"what has and has not been covered yet"'
            "}"
        )
        return system_prompt, user_prompt

    # -- parsing ------------------------------------------------------

    def _to_report(self, parsed, facts: SceneFacts, model: str = "", degraded: bool = False) -> SceneReport:
        if not isinstance(parsed, dict):
            return self._fallback_report(facts, error="non-dict LLM output")

        objects: list[ReportObject] = []
        for o in parsed.get("objects", []) or []:
            if not isinstance(o, dict):
                continue
            name = str(o.get("name", "")).strip()
            if not name:
                continue
            objects.append(
                ReportObject(
                    name=name,
                    location=str(o.get("location", "")).strip(),
                    evidence=self._parse_evidence(o.get("evidence")),
                )
            )

        observations = [str(x).strip() for x in (parsed.get("observations") or []) if str(x).strip()]
        room_type = str(parsed.get("room_type", "unknown")).strip().lower() or "unknown"
        return SceneReport(
            summary=str(parsed.get("summary", "")).strip(),
            room_type=room_type,
            objects=objects,
            observations=observations,
            coverage_note=str(parsed.get("coverage_note", "")).strip(),
            facts=facts,
            model=model,
            degraded=degraded,
        )

    @staticmethod
    def _parse_evidence(raw) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for e in raw or []:
            if not isinstance(e, dict):
                continue
            try:
                refs.append(
                    EvidenceRef(submap_id=int(e["submap_id"]), frame_idx=int(e["frame_idx"]))
                )
            except Exception:
                continue
        return refs

    def _fallback_report(self, facts: SceneFacts, error: str = "") -> SceneReport:
        names = sorted(facts.object_counts.keys())
        summary = (
            f"Scanned a space spanning {len(facts.objects)} detected object instance(s) "
            f"across {facts.metrics.num_submaps} submap(s)."
        )
        observations = ["Report model unavailable; showing a fact-only summary."] if error else []
        return SceneReport(
            summary=summary,
            room_type="unknown",
            objects=[ReportObject(name=n) for n in names],
            observations=observations,
            coverage_note="",
            facts=facts,
            model="",
            degraded=True,
        )

    def _progressive_fallback(
        self, facts: SceneFacts, previous: Optional[SceneReport], error: str = ""
    ) -> SceneReport:
        """Live-report fallback when no LLM is available or the call fails. Keeps the
        prior prose (so the report never regresses) but refreshes the live facts; with
        no prior report, emits a minimal fact-only progressive report."""
        if previous is not None:
            report = previous.model_copy(deep=True)
            report.facts = facts
            report.degraded = True
        else:
            report = self._fallback_report(facts, error=error)
            report.summary = (
                f"Scan in progress — {len(facts.objects)} object instance(s) across "
                f"{facts.metrics.num_submaps} submap(s) so far."
            )
        report.progressive = True
        report.final = False
        return report
