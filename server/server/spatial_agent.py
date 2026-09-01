"""SpatialAgent: autonomous spatial intelligence with OpenRouter-backed subagents."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

try:
    from server.agent import AgentRuntime
    from server.agent.schemas import ToolCall
except Exception:  # pragma: no cover - optional dependency path
    AgentRuntime = None
    ToolCall = None
from server.agent.scene_index import SceneIndex
from server.billing.usage import named_tally
from server.llm import OpenRouterClient


@dataclass
class Mission:
    id: int
    category: str
    goal: str
    queries: list[str]
    found: set[str] = field(default_factory=set)
    status: str = "active"  # active | recovering | completed | stalled
    findings: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    submaps_since_finding: int = 0
    last_progress_ts: float = field(default_factory=time.time)
    stall_reason: Optional[str] = None
    stall_count: int = 0


class SpatialAgent:
    """Session-scoped autonomous agent for SLAM scene exploration."""

    def __init__(
        self,
        streaming_slam,
        emit_fn: Callable[[str, dict[str, Any]], None],
        openrouter_api_key: str,
        session_id: str = "global",
        on_queries_changed: Optional[Callable[[str, list[str]], None]] = None,
        on_query_persisted: Optional[Callable[[str], None]] = None,
    ):
        self.slam = streaming_slam
        self.emit = emit_fn
        self.session_id = session_id
        self.on_queries_changed = on_queries_changed
        self.on_query_persisted = on_query_persisted

        # LLM config
        orch_model = os.environ.get(
            "SPATIAL_ORCH_MODEL", "google/gemini-3-flash-preview"
        )
        sub_model = os.environ.get(
            "SPATIAL_SUBAGENT_MODEL", "google/gemini-3-flash-preview"
        )
        orch_fallbacks = self._parse_csv_env(
            "SPATIAL_ORCH_FALLBACKS",
            "google/gemini-3-flash-preview,openai/gpt-4o-mini",
        )
        sub_fallbacks = self._parse_csv_env(
            "SPATIAL_SUBAGENT_FALLBACKS",
            "google/gemini-3-flash-preview,openai/gpt-4o-mini",
        )

        # Cost accounting for the whole session. The observe/act cycle below fires up
        # to six vision calls per submap for as long as the session runs, which made
        # this the largest LLM cost in the product and the only one nobody could see —
        # the client used to discard ``response.usage`` entirely. Both clients share
        # one tally so ``usage_summary()`` is the session total. Registered by name
        # rather than held privately so the worker's teardown log still reports it
        # if the container dies before ``shutdown()`` runs.
        self.usage = named_tally(f"spatial_agent:{session_id}")

        self.orchestrator_client = OpenRouterClient(
            api_key=openrouter_api_key,
            primary_model=orch_model,
            fallback_models=orch_fallbacks,
            timeout=float(os.environ.get("SPATIAL_LLM_TIMEOUT_S", "20")),
            max_retries=int(os.environ.get("SPATIAL_LLM_RETRIES", "2")),
            usage_sink=self.usage,
        )
        self.subagent_client = OpenRouterClient(
            api_key=openrouter_api_key,
            primary_model=sub_model,
            fallback_models=sub_fallbacks,
            timeout=float(os.environ.get("SPATIAL_LLM_TIMEOUT_S", "20")),
            max_retries=int(os.environ.get("SPATIAL_LLM_RETRIES", "2")),
            usage_sink=self.usage,
        )

        # Guardrails
        self.max_missions = int(os.environ.get("SPATIAL_MAX_MISSIONS", "8"))
        self.max_active_queries = int(os.environ.get("SPATIAL_MAX_ACTIVE_QUERIES", "12"))
        self.max_keyframes_per_analysis = int(os.environ.get("SPATIAL_MAX_KEYFRAMES", "4"))
        self.query_update_cooldown_s = float(os.environ.get("SPATIAL_QUERY_COOLDOWN_S", "2.0"))
        self.cycle_min_interval_s = float(os.environ.get("SPATIAL_CYCLE_MIN_INTERVAL_S", "1.2"))
        self.auto_complete_submap_gap = int(os.environ.get("SPATIAL_AUTOCOMPLETE_GAP", "6"))
        self.stall_submap_gap = int(os.environ.get("SPATIAL_STALL_SUBMAP_GAP", "3"))
        self.max_tool_calls_per_cycle = int(os.environ.get("SPATIAL_MAX_TOOL_CALLS_PER_CYCLE", "4"))
        self.tool_timeout_s = float(os.environ.get("SPATIAL_TOOL_TIMEOUT_S", "10.0"))
        self.task_heartbeat_s = float(os.environ.get("SPATIAL_TASK_HEARTBEAT_S", "2.0"))
        self.chat_timeout_s = float(os.environ.get("SPATIAL_CHAT_TIMEOUT_S", "25.0"))
        self.deep_scan_timeout_s = float(os.environ.get("SPATIAL_DEEP_SCAN_TIMEOUT_S", "60.0"))
        self.max_deep_scan_workers = int(os.environ.get("SPATIAL_DEEP_SCAN_WORKERS", "2"))
        self.max_pending_jobs = int(os.environ.get("SPATIAL_MAX_PENDING_JOBS", "8"))
        self.runtime_v2_enabled = os.environ.get("AGENT_RUNTIME_V2_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        if AgentRuntime is None:
            self.runtime_v2_enabled = False
            print("SpatialAgent runtime v2 disabled (missing optional dependencies)")

        # Scene understanding
        self.scene_description = ""
        self.room_type = "unknown"
        self.spatial_layout = ""
        self.scene_history: list[dict[str, Any]] = []

        # Mission tracking
        self.missions: dict[int, Mission] = {}
        self.next_mission_id = 1
        self.all_active_queries: list[str] = []

        # Detection tracking
        self.discovered_objects: dict[str, list[dict[str, Any]]] = {}
        self.previous_detection_keys: set[tuple[str, int, int]] = set()
        self.scene_index = SceneIndex(max_per_query=24)

        # User interaction
        self.current_goal: Optional[str] = None
        self.chat_history: list[dict[str, str]] = []

        # Visual context memory (bounded session cache)
        self.visual_memory: deque[dict[str, Any]] = deque(maxlen=24)

        # Runtime
        self.enabled = True
        self._processing = False
        self._last_query_sync_ts = 0.0
        self._last_cycle_ts = 0.0
        self._submap_count = 0
        self._coverage_estimate = 0.0
        self._active_tasks: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._recent_job_errors: deque[str] = deque(maxlen=8)

        # Tool metadata used for tool-call prompting.
        self._internal_tool_specs: list[dict[str, Any]] = [
            {
                "name": "get_visual_context_summary",
                "description": "Return compact structured scene context for observe phase.",
                "args_schema": {
                    "type": "object",
                    "properties": {
                        "max_items": {"type": "integer", "minimum": 1, "maximum": 24},
                        "include_missions": {"type": "boolean"},
                    },
                },
            },
            {
                "name": "request_visual_context_images",
                "description": "Return selected keyframe bundle (base64) for optional deeper visual review.",
                "args_schema": {
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "minimum": 1, "maximum": 6},
                        "purpose": {"type": "string"},
                        "query": {"type": "string"},
                    },
                },
            },
            {
                "name": "search_scene_index",
                "description": "Fast search over already discovered detections. Can auto-schedule deep scan on miss.",
                "args_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 24},
                        "auto_deep_scan_on_miss": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "add_detection_object",
                "description": (
                    "Add a single object query to the live detection worker pool. "
                    "Detection results stream to the viewer as submaps are scanned. "
                    "Call this once per object — do NOT batch multiple objects in one call."
                ),
                "args_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "remove_detection_object",
                "description": (
                    "Remove a single object from the active detection pool and clear its "
                    "bounding boxes from the viewer."
                ),
                "args_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_job_status",
                "description": "Get status/result for an async job.",
                "args_schema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
            {
                "name": "cancel_job",
                "description": "Request cancellation of an async job.",
                "args_schema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
        ]
        self._observe_tool_names = {
            "get_visual_context_summary",
            "request_visual_context_images",
            "search_scene_index",
            "get_job_status",
        }

        # Thread safety
        self._lock = threading.Lock()
        self._task_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._job_executor = ThreadPoolExecutor(max_workers=max(1, self.max_deep_scan_workers))
        self._runtime = None
        if self.runtime_v2_enabled and AgentRuntime is not None:
            self._runtime = AgentRuntime(
                session_id=self.session_id,
                streaming_slam=self.slam,
                emit_event=self.emit,
                max_workers=4,
                on_query_added=self.on_query_persisted,
            )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def on_submap_processed(self, submap_id: int, detections: Optional[list[dict[str, Any]]] = None):
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_cycle_ts < self.cycle_min_interval_s:
            return

        with self._lock:
            if self._processing:
                return
            self._processing = True
            self._last_cycle_ts = now

        try:
            self._submap_count = max(self._submap_count, submap_id + 1)
            self._refresh_job_states()
            keyframes_b64 = self._extract_keyframes_b64(submap_id)
            if keyframes_b64:
                self._remember_keyframes(submap_id, keyframes_b64)

            if detections is None:
                with self.slam._detection_lock:
                    detections = list(self.slam.accumulated_detections)

            new_detections = self._diff_detections(detections or [])
            if new_detections:
                self._route_detections(new_detections)

            # Subagents in parallel with tracked lifecycle events.
            scene_task = self._start_task("subagent", "scene_analyzer")
            spotter_task = self._start_task("subagent", "object_spotter")
            scene_future = self._executor.submit(self._run_scene_analyzer, keyframes_b64)
            spotter_future = self._executor.submit(self._run_object_spotter, keyframes_b64)

            scene_result = self._safe_future_result(
                scene_future,
                "scene_analyzer",
                task_id=scene_task["id"],
                started_at=scene_task["started_at"],
            )
            spotter_result = self._safe_future_result(
                spotter_future,
                "object_spotter",
                task_id=spotter_task["id"],
                started_at=spotter_task["started_at"],
            )

            layout_result = None
            if self._submap_count % 4 == 0:
                layout_result = self._run_tracked_task(
                    task_type="subagent",
                    name="layout_mapper",
                    fn=lambda: self._run_layout_mapper(keyframes_b64),
                    timeout_s=14.0,
                )
                if layout_result and layout_result.get("layout_description"):
                    self.spatial_layout = layout_result["layout_description"]

            self._bootstrap_missions_from_spotter(spotter_result)

            orchestrator_result = self._run_tracked_task(
                task_type="orchestrator",
                name="cycle_orchestrator",
                fn=lambda: self._run_orchestrator(
                    keyframes_b64=keyframes_b64,
                    scene_result=scene_result,
                    spotter_result=spotter_result,
                    layout_result=layout_result,
                    new_detections=new_detections,
                ),
                timeout_s=max(8.0, self.chat_timeout_s),
            )
            if orchestrator_result:
                self._apply_orchestrator_result(orchestrator_result)
            self._update_mission_stall_states()

            # Coverage estimate: mostly based on explored submaps and mission completion.
            completed = sum(1 for m in self.missions.values() if m.status == "completed")
            total = max(len(self.missions), 1)
            mission_ratio = completed / total
            self._coverage_estimate = min(1.0, (self._submap_count * 0.08) + (mission_ratio * 0.25))

            self._emit_state()

        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] cycle error: {e}")
            self._emit_thought(
                "Agent hit a transient issue and will continue scanning.",
                thought_type="error",
                subagent="orchestrator",
            )
        finally:
            with self._lock:
                self._processing = False

    # ------------------------------------------------------------------
    # Orchestrator + subagents
    # ------------------------------------------------------------------

    def _start_task(self, task_type: str, name: str, mission_id: Optional[int] = None) -> dict[str, Any]:
        task_id = str(uuid.uuid4())[:12]
        started_at = time.time()
        task = {
            "id": task_id,
            "timestamp": started_at,
            "task_type": task_type,
            "name": name,
            "status": "started",
            "mission_id": mission_id,
        }
        with self._task_lock:
            self._active_tasks[task_id] = task
        self._emit_task_event(task)
        return {"id": task_id, "started_at": started_at}

    def _update_active_task(self, task_id: str, status: str):
        with self._task_lock:
            task = self._active_tasks.get(task_id)
            if task is not None:
                task["status"] = status

    def _finish_task(
        self,
        task_id: str,
        task_type: str,
        name: str,
        status: str,
        started_at: float,
        mission_id: Optional[int] = None,
        details: Optional[str] = None,
        error: Optional[str] = None,
    ):
        with self._task_lock:
            self._active_tasks.pop(task_id, None)
        payload: dict[str, Any] = {
            "id": task_id,
            "timestamp": time.time(),
            "task_type": task_type,
            "name": name,
            "status": status,
            "latency_ms": int((time.time() - started_at) * 1000),
        }
        if mission_id is not None:
            payload["mission_id"] = mission_id
        if details:
            payload["details"] = details
        if error:
            payload["error"] = error
        self._emit_task_event(payload)

    def _run_tracked_task(
        self,
        task_type: str,
        name: str,
        fn: Callable[[], Any],
        timeout_s: float = 12.0,
        mission_id: Optional[int] = None,
    ):
        task = self._start_task(task_type, name, mission_id=mission_id)
        task_id = task["id"]
        started_at = float(task["started_at"])
        future = self._executor.submit(fn)
        return self._safe_future_result(
            future,
            name,
            timeout=timeout_s,
            task_id=task_id,
            started_at=started_at,
            task_type=task_type,
            mission_id=mission_id,
        )

    def _run_scene_analyzer(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are a scene analysis subagent for a live 3D SLAM system. "
            "Output strict JSON only."
        )
        user_prompt = (
            "Analyze the provided keyframes. Return JSON: "
            "{\"description\":\"1-2 concise sentences\","
            "\"room_type\":\"office|kitchen|bedroom|bathroom|hallway|outdoor|other\","
            "\"notable_features\":[\"...\"]}"
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.2,
                max_tokens=384,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] scene analyzer error: {e}")
            return None

    def _run_object_spotter(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are an object spotting subagent for a real-world mapping system. "
            "List only concrete visible objects. Output strict JSON only."
        )
        goal_context = f"User goal: {self.current_goal}\n\n" if self.current_goal else ""
        user_prompt = (
            f"{goal_context}"
            "Return JSON: {\"objects\":[{\"name\":\"object name\","
            "\"count_estimate\":1,\"location_hint\":\"where\"}]}."
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.2,
                max_tokens=448,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] object spotter error: {e}")
            return None

    def _run_layout_mapper(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are a spatial layout subagent. Infer rough layout relationships from keyframes. "
            "Output strict JSON only."
        )
        user_prompt = (
            "Return JSON: {\"layout_description\":\"...\","
            "\"spatial_relationships\":[\"item A is near item B\"],"
            "\"room_dimensions_estimate\":\"rough dimensions\"}."
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.25,
                max_tokens=512,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] layout mapper error: {e}")
            return None

    def _run_orchestrator(
        self,
        keyframes_b64: list[str],
        scene_result: Optional[dict[str, Any]],
        spotter_result: Optional[dict[str, Any]],
        layout_result: Optional[dict[str, Any]],
        new_detections: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        context = self._build_orchestrator_context(
            scene_result=scene_result,
            spotter_result=spotter_result,
            layout_result=layout_result,
            new_detections=new_detections,
        )

        observe_result = self._run_orchestrator_observe(
            context=context,
            keyframes_b64=keyframes_b64,
        )
        observe_calls = []
        if isinstance(observe_result, dict):
            observe_calls = observe_result.get("observe_tool_calls", [])
            if isinstance(observe_result.get("narrative"), str) and observe_result["narrative"].strip():
                self._emit_thought(
                    str(observe_result["narrative"]).strip(),
                    thought_type="thinking",
                    subagent="orchestrator",
                    keyframe_b64=self._latest_keyframe_b64(),
                )
        observe_outcomes = self._execute_tool_calls(observe_calls if isinstance(observe_calls, list) else [], phase="observe")

        return self._run_orchestrator_act(
            context=context,
            observe_result=observe_result or {},
            observe_tool_outcomes=observe_outcomes,
        )

    def _build_orchestrator_context(
        self,
        scene_result: Optional[dict[str, Any]],
        spotter_result: Optional[dict[str, Any]],
        layout_result: Optional[dict[str, Any]],
        new_detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        missions_summary = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
                "confidence": m.confidence,
                "stall_reason": m.stall_reason,
            }
            for m in self.missions.values()
        ]
        return {
            "session_id": self.session_id,
            "submaps_processed": self._submap_count,
            "room_type": self.room_type,
            "scene_description": self.scene_description,
            "spatial_layout": self.spatial_layout,
            "goal": self.current_goal or "autonomous exploration",
            "missions": missions_summary,
            "objects_discovered": sorted(self.discovered_objects.keys()),
            "new_detections": new_detections[:12],
            "scene_analyzer": scene_result if scene_result else "unavailable",
            "object_spotter": spotter_result if spotter_result else "unavailable",
            "layout_mapper": layout_result if layout_result else "not run",
            "scene_index": self.scene_index.summary(max_queries=8),
        }

    def _list_available_tools(self, observe_only: bool = False) -> list[dict[str, Any]]:
        internal = []
        for spec in self._internal_tool_specs:
            if observe_only and spec.get("name") not in self._observe_tool_names:
                continue
            internal.append(spec)
        runtime_tools = self._runtime.list_tools() if (self.runtime_v2_enabled and self._runtime is not None) else []
        return internal + runtime_tools

    def _compact_tool_outcomes_for_prompt(self, outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in outcomes[:8]:
            if not isinstance(item, dict):
                continue
            payload: dict[str, Any] = {
                "tool": item.get("tool"),
                "ok": bool(item.get("ok")),
            }
            if item.get("error"):
                payload["error"] = str(item.get("error"))[:200]
            data = item.get("data", {})
            if isinstance(data, dict):
                reduced: dict[str, Any] = {}
                for key, value in data.items():
                    if key == "images" and isinstance(value, list):
                        reduced["image_count"] = len(value)
                        if value:
                            first = value[0]
                            if isinstance(first, dict):
                                reduced["first_image_meta"] = {
                                    "submap_id": first.get("submap_id"),
                                    "timestamp": first.get("timestamp"),
                                }
                        continue
                    if key in {"detections", "matches"} and isinstance(value, list):
                        reduced[key] = value[:3]
                        reduced[f"{key}_count"] = len(value)
                        continue
                    if isinstance(value, str):
                        reduced[key] = value[:240]
                    elif isinstance(value, list) and len(value) > 12:
                        reduced[key] = value[:12]
                    else:
                        reduced[key] = value
                payload["data"] = reduced
            compact.append(payload)
        return compact

    def _images_from_tool_outcomes(self, outcomes: list[dict[str, Any]], max_images: int = 2) -> list[str]:
        images: list[str] = []
        for item in outcomes:
            if len(images) >= max_images:
                break
            if not isinstance(item, dict):
                continue
            data = item.get("data", {})
            if not isinstance(data, dict):
                continue
            bundle = data.get("images", [])
            if not isinstance(bundle, list):
                continue
            for row in bundle:
                if len(images) >= max_images:
                    break
                if not isinstance(row, dict):
                    continue
                img = row.get("image_b64")
                if isinstance(img, str) and img:
                    images.append(img)
        return images

    def _run_orchestrator_observe(
        self,
        context: dict[str, Any],
        keyframes_b64: list[str],
    ) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are the observe-phase planner for an autonomous spatial intelligence system. "
            "Only request tools that gather context. Do not create or complete missions in this phase. "
            "Output strict JSON only."
        )
        user_prompt = (
            "Observe context:\n"
            f"{json.dumps(context)}\n"
            f"- Available observe tools: {json.dumps(self._list_available_tools(observe_only=True))}\n"
            "\nReturn JSON exactly in this shape:\n"
            "{"
            "\"narrative\":\"1-2 concise lines about what context is missing\","
            "\"observe_tool_calls\":[{\"name\":\"tool_name\",\"args\":{}}]"
            "}"
        )
        try:
            parsed, response = self.orchestrator_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[:2],
                temperature=0.25,
                max_tokens=512,
            )
            if response.degraded:
                self._emit_thought(
                    f"LLM fallback active ({response.model}) — observe phase continuing.",
                    thought_type="thinking",
                    subagent="orchestrator",
                )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] observe-phase error: {e}")
            return None

    def _run_orchestrator_act(
        self,
        context: dict[str, Any],
        observe_result: dict[str, Any],
        observe_tool_outcomes: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are the act-phase orchestrator for an autonomous spatial intelligence system. "
            "Use observed tool outputs first, then update missions and choose practical tool calls. "
            "Prefer fast `search_scene_index` before expensive scan tools. Output strict JSON only."
        )
        user_prompt = (
            "Act context:\n"
            f"- Base context: {json.dumps(context)}\n"
            f"- Observe output: {json.dumps(observe_result)}\n"
            f"- Observe tool outcomes: {json.dumps(self._compact_tool_outcomes_for_prompt(observe_tool_outcomes))}\n"
            f"- Attached observe images: {len(self._images_from_tool_outcomes(observe_tool_outcomes, max_images=2))}\n"
            f"- Available tools: {json.dumps(self._list_available_tools(observe_only=False))}\n"
            "\nReturn JSON exactly in this shape:\n"
            "{"
            "\"narrative\":\"1-3 concise sentences\","
            "\"scene_update\":\"updated scene summary\","
            "\"room_type\":\"office|kitchen|bedroom|bathroom|hallway|outdoor|other\","
            "\"new_missions\":[{\"category\":\"Category\",\"goal\":\"Goal\",\"queries\":[\"q1\",\"q2\"]}],"
            "\"complete_missions\":[1],"
            "\"add_queries_to_mission\":{\"2\":[\"q3\"]},"
            "\"coverage_estimate\":0.0,"
            "\"tool_calls\":[{\"name\":\"tool_name\",\"args\":{}}]"
            "}"
        )
        try:
            parsed, response = self.orchestrator_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=self._images_from_tool_outcomes(observe_tool_outcomes, max_images=2),
                temperature=0.35,
                max_tokens=1024,
            )
            if response.degraded:
                self._emit_thought(
                    f"LLM fallback active ({response.model}) — act phase continuing.",
                    thought_type="thinking",
                    subagent="orchestrator",
                )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] act-phase error: {e}")
            return None

    def _bootstrap_missions_from_spotter(self, spotter_result: Optional[dict[str, Any]]):
        if not spotter_result or len(self.missions) >= 2:
            return

        objects = spotter_result.get("objects", [])
        candidate_queries: list[str] = []
        for obj in objects:
            name = str(obj.get("name", "")).strip().lower()
            if not name:
                continue
            if name not in candidate_queries:
                candidate_queries.append(name)
            if len(candidate_queries) >= 4:
                break

        if not candidate_queries:
            return

        self._create_mission(
            category="Autonomous Recon",
            goal="Catalog key objects in the current environment",
            queries=candidate_queries,
        )

    def _apply_orchestrator_result(self, result: dict[str, Any]):
        scene_update = result.get("scene_update")
        if isinstance(scene_update, str) and scene_update.strip():
            self.scene_description = scene_update.strip()

        room_type = result.get("room_type")
        if isinstance(room_type, str) and room_type.strip():
            self.room_type = room_type.strip().lower()

        narrative = result.get("narrative")
        if isinstance(narrative, str) and narrative.strip():
            self._emit_thought(
                narrative.strip(),
                thought_type="observation",
                subagent="orchestrator",
                keyframe_b64=self._latest_keyframe_b64(),
            )

        new_missions = result.get("new_missions", [])
        if isinstance(new_missions, list):
            for mission_data in new_missions:
                if not isinstance(mission_data, dict):
                    continue
                queries = mission_data.get("queries", [])
                if not isinstance(queries, list):
                    continue
                self._create_mission(
                    category=str(mission_data.get("category", "General")).strip() or "General",
                    goal=str(mission_data.get("goal", "Explore")).strip() or "Explore",
                    queries=[str(q) for q in queries],
                )

        complete = result.get("complete_missions", [])
        if isinstance(complete, list):
            for mid in complete:
                try:
                    mission_id = int(mid)
                except Exception:
                    continue
                mission = self.missions.get(mission_id)
                if mission is None:
                    continue
                mission.status = "completed"
                self._emit_action(
                    "mission_completed",
                    f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                    {"mission_id": mission_id},
                )

        add_queries = result.get("add_queries_to_mission", {})
        if isinstance(add_queries, dict):
            for mid_str, query_list in add_queries.items():
                if not isinstance(query_list, list):
                    continue
                try:
                    mission_id = int(mid_str)
                except Exception:
                    continue
                mission = self.missions.get(mission_id)
                if mission is None or mission.status not in {"active", "recovering"}:
                    continue
                for query in query_list:
                    q = str(query).strip().lower()
                    if q and q not in mission.queries:
                        mission.queries.append(q)

        cov = result.get("coverage_estimate")
        if isinstance(cov, (int, float)):
            self._coverage_estimate = min(max(float(cov), 0.0), 1.0)

        tool_calls = result.get("tool_calls", [])
        if isinstance(tool_calls, list) and tool_calls:
            self._execute_tool_calls(tool_calls)

        self._sync_detection_queries()

        self.scene_history.append(
            {
                "submap_count": self._submap_count,
                "scene_description": self.scene_description,
                "room_type": self.room_type,
                "timestamp": time.time(),
            }
        )
        self.scene_history = self.scene_history[-20:]

    def _run_single_tool(self, tool_name: str, tool_args: dict[str, Any], mission_id: Optional[int] = None) -> dict[str, Any]:
        task = self._start_task("tool_batch", tool_name, mission_id=mission_id)
        if self._is_internal_tool(tool_name):
            result = self._run_internal_tool(tool_name, tool_args, mission_id=mission_id)
        elif self.runtime_v2_enabled and self._runtime is not None:
            result = self._runtime.execute_tool(
                tool_name,
                tool_args,
                timeout_s=self.tool_timeout_s,
            )
        else:
            result = {"ok": False, "tool": tool_name, "error": "runtime_disabled"}
        if result.get("ok", False):
            self._finish_task(
                task_id=task["id"],
                task_type="tool_batch",
                name=tool_name,
                status="succeeded",
                started_at=float(task["started_at"]),
                mission_id=mission_id,
            )
        else:
            self._finish_task(
                task_id=task["id"],
                task_type="tool_batch",
                name=tool_name,
                status="failed",
                started_at=float(task["started_at"]),
                mission_id=mission_id,
                error=str(result.get("error", "tool_failed")),
            )
        return result

    def _execute_tool_calls(self, tool_calls: list[Any], phase: str = "act"):
        outcomes: list[dict[str, Any]] = []
        executed = 0
        for call in tool_calls:
            if executed >= self.max_tool_calls_per_cycle:
                break
            if not isinstance(call, dict):
                continue

            if ToolCall is not None:
                try:
                    parsed = ToolCall.model_validate(call)
                    tool_name = parsed.name
                    tool_args = parsed.args
                except Exception:
                    continue
            else:
                tool_name = str(call.get("name", "")).strip()
                tool_args = call.get("args", {})
                if not tool_name or not isinstance(tool_args, dict):
                    continue

            mission_id = None
            if isinstance(tool_args.get("mission_id"), int):
                mission_id = int(tool_args["mission_id"])
            elif isinstance(tool_args.get("query"), str):
                mission_id = self._find_mission_for_query(str(tool_args["query"]).strip().lower())

            result = self._run_single_tool(tool_name, tool_args, mission_id=mission_id)
            executed += 1
            outcomes.append(result)

            if not result.get("ok", False):
                self._emit_thought(
                    f"Tool failed: {tool_name}",
                    thought_type="error",
                    subagent="orchestrator",
                )
                continue

            data = result.get("data", {})
            if tool_name == "search_objects":
                detections = data.get("detections", [])
                if isinstance(detections, list) and detections:
                    self._route_detections(detections)
                    for mission in self.missions.values():
                        if mission.status == "completed":
                            continue
                        if any(str(d.get("query", "")).strip().lower() in mission.queries for d in detections):
                            self._mark_mission_progress(mission, reason="tool:search_objects")
            elif tool_name == "locate_object_3d" and data.get("found"):
                center = data.get("center")
                query = data.get("query")
                if isinstance(center, list) and len(center) == 3:
                    self._run_single_tool(
                        "focus_detection_ui",
                        {
                            "query": query,
                            "submap_id": data.get("matched_submap"),
                            "frame_idx": data.get("matched_frame"),
                            "center": center,
                        },
                    )
                for mission in self.missions.values():
                    if mission.status == "completed":
                        continue
                        if query and query in mission.queries:
                            self._mark_mission_progress(mission, reason="tool:locate_object_3d")
            elif tool_name == "search_scene_index":
                if not data.get("found") and data.get("scheduled_deep_scan_job_id"):
                    self._emit_action(
                        "deep_scan_scheduled",
                        f"Scheduled deep scan for {data.get('query')}",
                        {"job_id": data.get("scheduled_deep_scan_job_id")},
                    )
            elif tool_name == "add_detection_object" and data.get("success"):
                query = data.get("query", "")
                # Add to mission tracking and sync queries — this triggers on_queries_changed
                # which spawns the per-query worker in app.py
                self._sync_detection_queries_add(query)
                self._emit_action(
                    "detection_object_added",
                    f"Added detection target: {query}",
                    {"query": query, "status": data.get("status", "queued")},
                )
            elif tool_name == "remove_detection_object" and data.get("success"):
                query = data.get("query", "")
                self._sync_detection_queries_remove(query)
                self._emit_action(
                    "detection_object_removed",
                    f"Removed detection target: {query}",
                    {"query": query},
                )
        return outcomes

    def _is_internal_tool(self, tool_name: str) -> bool:
        return any(spec.get("name") == tool_name for spec in self._internal_tool_specs)

    def _run_internal_tool(self, tool_name: str, tool_args: dict[str, Any], mission_id: Optional[int]) -> dict[str, Any]:
        event_id = str(uuid.uuid4())[:12]
        started = time.time()
        self._emit_tool_event_payload({"id": event_id, "tool": tool_name, "status": "started", "args": tool_args})
        try:
            if tool_name == "get_visual_context_summary":
                payload = self._tool_get_visual_context_summary(tool_args)
            elif tool_name == "request_visual_context_images":
                payload = self._tool_request_visual_context_images(tool_args)
            elif tool_name == "search_scene_index":
                payload = self._tool_search_scene_index(tool_args, mission_id=mission_id)
            elif tool_name == "add_detection_object":
                payload = self._tool_add_detection_object(tool_args)
            elif tool_name == "remove_detection_object":
                payload = self._tool_remove_detection_object(tool_args)
            elif tool_name == "get_job_status":
                payload = self._tool_get_job_status(tool_args)
            elif tool_name == "cancel_job":
                payload = self._tool_cancel_job(tool_args)
            else:
                raise RuntimeError(f"Unknown internal tool: {tool_name}")

            elapsed = int((time.time() - started) * 1000)
            self._emit_tool_event_payload(
                {
                    "id": event_id,
                    "tool": tool_name,
                    "status": "succeeded",
                    "result": payload,
                    "latency_ms": elapsed,
                }
            )
            return {"ok": True, "tool": tool_name, "data": payload, "latency_ms": elapsed}
        except Exception as exc:
            elapsed = int((time.time() - started) * 1000)
            self._emit_tool_event_payload(
                {
                    "id": event_id,
                    "tool": tool_name,
                    "status": "failed",
                    "error": str(exc),
                    "latency_ms": elapsed,
                }
            )
            return {"ok": False, "tool": tool_name, "error": str(exc), "latency_ms": elapsed}

    def _tool_get_visual_context_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        max_items = max(1, min(24, int(args.get("max_items", 8))))
        include_missions = bool(args.get("include_missions", True))
        unresolved_queries: list[str] = []
        missions = []
        if include_missions:
            for mission in self.missions.values():
                missions.append(
                    {
                        "id": mission.id,
                        "goal": mission.goal,
                        "status": mission.status,
                        "queries": mission.queries,
                        "found": sorted(mission.found),
                        "stall_reason": mission.stall_reason,
                    }
                )
                if mission.status in {"active", "recovering"}:
                    for q in mission.queries:
                        if q not in mission.found and q not in unresolved_queries:
                            unresolved_queries.append(q)

        recent_observations = []
        for item in list(self.visual_memory)[-max_items:]:
            recent_observations.append(
                {
                    "submap_id": item.get("submap_id"),
                    "timestamp": item.get("timestamp"),
                }
            )
        with self._job_lock:
            running_jobs = sum(1 for j in self._jobs.values() if j.get("status") == "running")
            queued_jobs = sum(1 for j in self._jobs.values() if j.get("status") == "queued")

        return {
            "session_id": self.session_id,
            "room_type": self.room_type,
            "scene_description": self.scene_description,
            "spatial_layout": self.spatial_layout,
            "submaps_processed": self._submap_count,
            "coverage_estimate": round(float(self._coverage_estimate), 3),
            "active_queries": list(self.all_active_queries),
            "unresolved_queries": unresolved_queries[:max_items],
            "discovered_objects": sorted(self.discovered_objects.keys())[:max_items],
            "scene_index": self.scene_index.summary(max_queries=max_items),
            "job_queue": {"running": running_jobs, "queued": queued_jobs},
            "recent_observations": recent_observations,
            "missions": missions[:max_items] if include_missions else [],
            "explanation_for_model": (
                "Use unresolved_queries and scene_index first. "
                "Request visual images only when ambiguity remains."
            ),
        }

    def _tool_request_visual_context_images(self, args: dict[str, Any]) -> dict[str, Any]:
        k = max(1, min(6, int(args.get("k", 2))))
        purpose = str(args.get("purpose", "") or "").strip()
        query = str(args.get("query", "") or "").strip().lower()

        preferred_submaps: set[int] = set()
        if query:
            for match in self.scene_index.search(query, max_results=8):
                preferred_submaps.add(int(match.get("matched_submap", -1)))

        selected: list[dict[str, Any]] = []
        seen = set()
        for item in reversed(self.visual_memory):
            img = item.get("image_b64")
            submap_id = int(item.get("submap_id", -1))
            if not img or img in seen:
                continue
            if preferred_submaps and submap_id not in preferred_submaps:
                continue
            selected.append(
                {
                    "submap_id": submap_id,
                    "timestamp": item.get("timestamp"),
                    "image_b64": img,
                }
            )
            seen.add(img)
            if len(selected) >= k:
                break

        if not selected:
            for item in reversed(self.visual_memory):
                img = item.get("image_b64")
                if not img or img in seen:
                    continue
                selected.append(
                    {
                        "submap_id": int(item.get("submap_id", -1)),
                        "timestamp": item.get("timestamp"),
                        "image_b64": img,
                    }
                )
                seen.add(img)
                if len(selected) >= k:
                    break

        return {
            "bundle_id": str(uuid.uuid4())[:12],
            "purpose": purpose or None,
            "query": query or None,
            "images": selected,
            "count": len(selected),
            "explanation_for_model": (
                "These are recent keyframes selected for extra context. "
                "Use them to refine query updates and scan priorities."
            ),
        }

    def _tool_search_scene_index(self, args: dict[str, Any], mission_id: Optional[int]) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip().lower()
        if not query:
            raise RuntimeError("query is required")
        max_results = max(1, min(24, int(args.get("max_results", 8))))
        auto_deep_scan = bool(args.get("auto_deep_scan_on_miss", True))
        matches = self.scene_index.search(query, max_results=max_results)
        payload = {
            "query": query,
            "found": len(matches) > 0,
            "matches": matches,
            "count": len(matches),
        }
        if not matches and auto_deep_scan:
            scheduled = self._enqueue_deep_scan([query], mission_id=mission_id, top_k=3)
            if scheduled is not None:
                payload["scheduled_deep_scan_job_id"] = scheduled["job_id"]
        return payload

    def _tool_add_detection_object(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip().lower()
        if not query:
            raise RuntimeError("query is required")
        with self.slam._detection_lock:
            already_active = query in self.slam.active_queries
        if already_active:
            return {"success": True, "query": query, "status": "already_active"}
        return {"success": True, "query": query, "status": "queued"}

    def _tool_remove_detection_object(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip().lower()
        if not query:
            raise RuntimeError("query is required")
        self.slam.remove_query(query)
        return {"success": True, "query": query, "removed": True}

    def _sync_detection_queries_add(self, query: str):
        """Add a query to all_active_queries and notify app.py via on_queries_changed."""
        if not query:
            return
        with self._lock:
            if query not in self.all_active_queries:
                self.all_active_queries.append(query)
        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, list(self.all_active_queries))
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] query add callback error: {e}")

    def _sync_detection_queries_remove(self, query: str):
        """Remove a query from all_active_queries and notify app.py via on_queries_changed."""
        if not query:
            return
        with self._lock:
            if query in self.all_active_queries:
                self.all_active_queries.remove(query)
        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, list(self.all_active_queries))
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] query remove callback error: {e}")

    def _tool_start_deep_scan(self, args: dict[str, Any], mission_id: Optional[int]) -> dict[str, Any]:
        raw_queries = args.get("queries", [])
        queries = self._normalize_queries_for_tools(raw_queries if isinstance(raw_queries, list) else [raw_queries])
        if not queries:
            raise RuntimeError("queries cannot be empty")
        resolved_mission_id = mission_id
        if args.get("mission_id") is not None:
            try:
                resolved_mission_id = int(args.get("mission_id"))
            except Exception:
                pass
        top_k = max(1, min(8, int(args.get("top_k", 3))))
        scheduled = self._enqueue_deep_scan(queries, mission_id=resolved_mission_id, top_k=top_k)
        if scheduled is None:
            raise RuntimeError("job_queue_full")
        return {
            "job_id": scheduled["job_id"],
            "status": scheduled["status"],
            "queries": queries,
            "mission_id": resolved_mission_id,
            "top_k": top_k,
        }

    def _tool_get_job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = str(args.get("job_id", "") or "").strip()
        if not job_id:
            raise RuntimeError("job_id is required")
        self._refresh_job_states()
        with self._job_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"job_id": job_id, "status": "not_found"}
        return self._serialize_job(job, include_result=True)

    def _tool_cancel_job(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = str(args.get("job_id", "") or "").strip()
        if not job_id:
            raise RuntimeError("job_id is required")
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "not_found"}
            future = job.get("future")
            status = str(job.get("status"))
            if status in {"succeeded", "failed", "timed_out", "canceled"}:
                return {"job_id": job_id, "status": status, "canceled": False}
            canceled = bool(future.cancel()) if future is not None else False
            job["status"] = "canceled" if canceled else "running"
            job["finished_at"] = time.time() if canceled else None
        if canceled:
            self._emit_job_event(job)
        return {"job_id": job_id, "status": job["status"], "canceled": canceled}

    def _normalize_queries_for_tools(self, raw: list[Any], max_count: int = 12) -> list[str]:
        out: list[str] = []
        for item in raw:
            q = str(item or "").strip().lower()
            if not q:
                continue
            if len(q) > 120:
                q = q[:120]
            if q not in out:
                out.append(q)
            if len(out) >= max_count:
                break
        return out

    def _enqueue_deep_scan(self, queries: list[str], mission_id: Optional[int], top_k: int) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._job_lock:
            active_jobs = [j for j in self._jobs.values() if j.get("status") in {"queued", "running"}]
            if len(active_jobs) >= self.max_pending_jobs:
                return None

            normalized = tuple(sorted(queries))
            for job in active_jobs:
                if job.get("name") == "deep_scan" and tuple(job.get("query_signature", ())) == normalized:
                    return {"job_id": job["job_id"], "status": job["status"], "deduped": True}

            job_id = str(uuid.uuid4())[:12]
            job = {
                "job_id": job_id,
                "name": "deep_scan",
                "status": "queued",
                "args": {"queries": list(queries), "top_k": int(top_k)},
                "query_signature": normalized,
                "mission_id": mission_id,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "timeout_s": float(self.deep_scan_timeout_s),
                "future": None,
            }
            self._jobs[job_id] = job
            self._emit_job_event(job)

            future = self._job_executor.submit(self._deep_scan_worker, list(queries), int(top_k))
            job["future"] = future
            job["status"] = "running"
            job["started_at"] = time.time()
            self._emit_job_event(job)
            future.add_done_callback(lambda fut, jid=job_id: self._on_deep_scan_done(jid, fut))
            return {"job_id": job_id, "status": "running"}

    def _deep_scan_worker(self, queries: list[str], top_k: int) -> dict[str, Any]:
        report = self.slam.debug_detect_full(queries, top_k=top_k, include_frames=False)
        detections = report.get("detections", []) if isinstance(report, dict) else []
        return {
            "queries": list(queries),
            "count": len(detections) if isinstance(detections, list) else 0,
            "detections": detections if isinstance(detections, list) else [],
            "query_time_ms": int(report.get("query_time_ms", 0)) if isinstance(report, dict) else 0,
        }

    def _on_deep_scan_done(self, job_id: str, future: concurrent.futures.Future):
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if str(job.get("status")) == "timed_out":
                return
            job["finished_at"] = time.time()

        try:
            result = future.result()
            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "succeeded"
                job["result"] = result
            self._emit_job_event(job)
            self._executor.submit(self._apply_deep_scan_result, job_id, result)
        except concurrent.futures.CancelledError:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "canceled"
                job["error"] = "job canceled"
            self._emit_job_event(job)
        except Exception as exc:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "failed"
                job["error"] = str(exc)
                self._recent_job_errors.append(str(exc))
            self._emit_job_event(job)

    def _apply_deep_scan_result(self, job_id: str, result: dict[str, Any]):
        detections = result.get("detections", []) if isinstance(result, dict) else []
        if isinstance(detections, list) and detections:
            self._route_detections(detections)
            self._emit_action(
                "deep_scan_completed",
                f"Deep scan {job_id} found {len(detections)} detections.",
                {"job_id": job_id},
            )
        else:
            self._emit_action(
                "deep_scan_completed",
                f"Deep scan {job_id} completed with no detections.",
                {"job_id": job_id},
            )
        self._emit_state()

    def _refresh_job_states(self):
        now = time.time()
        timed_out: list[dict[str, Any]] = []
        with self._job_lock:
            for job in self._jobs.values():
                if job.get("status") != "running":
                    continue
                started = job.get("started_at")
                timeout_s = float(job.get("timeout_s", self.deep_scan_timeout_s))
                if started is None or (now - float(started)) < timeout_s:
                    continue
                future = job.get("future")
                if future is not None:
                    future.cancel()
                job["status"] = "timed_out"
                job["finished_at"] = now
                job["error"] = f"timed out after {int(timeout_s)}s"
                timed_out.append(dict(job))
            if len(self._jobs) > 128:
                terminal = sorted(
                    [j for j in self._jobs.values() if j.get("status") in {"succeeded", "failed", "timed_out", "canceled"}],
                    key=lambda x: float(x.get("finished_at") or x.get("created_at") or 0.0),
                )
                for old in terminal[: max(0, len(self._jobs) - 96)]:
                    self._jobs.pop(str(old.get("job_id")), None)
        for job in timed_out:
            self._recent_job_errors.append(str(job.get("error")))
            self._emit_job_event(job)

    def _serialize_job(self, job: dict[str, Any], include_result: bool = False) -> dict[str, Any]:
        payload = {
            "job_id": job.get("job_id"),
            "job_name": job.get("name"),
            "status": job.get("status"),
            "mission_id": job.get("mission_id"),
            "args": job.get("args"),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }
        if include_result:
            payload["result"] = job.get("result")
        else:
            result = job.get("result")
            if isinstance(result, dict):
                payload["result"] = {k: result.get(k) for k in ("count", "queries", "query_time_ms")}
        return payload

    def _emit_job_event(self, job: dict[str, Any]):
        payload = self._serialize_job(job, include_result=False)
        payload["id"] = str(uuid.uuid4())[:12]
        try:
            self.emit("agent_job_event", payload)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit job event error: {e}")

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def _parse_query_candidates(self, text: str) -> list[str]:
        cleaned = re.sub(r"\b(and|then|also)\b", ",", text.lower())
        parts = [part.strip(" .,!?:;") for part in cleaned.split(",")]
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            tokens = [tok for tok in part.split() if tok not in {"a", "an", "the", "for", "to"}]
            query = " ".join(tokens).strip()
            if query and query not in out:
                out.append(query[:120])
        return out[:6]

    def _try_tool_first_chat(self, message: str) -> Optional[str]:
        lower = message.strip().lower()
        if not lower:
            return None

        if any(phrase in lower for phrase in ("what are you doing", "status", "missions", "progress")):
            active = sum(1 for m in self.missions.values() if m.status in {"active", "recovering"})
            stalled = sum(1 for m in self.missions.values() if m.status == "stalled")
            completed = sum(1 for m in self.missions.values() if m.status == "completed")
            return (
                f"I am tracking {active} active missions, {stalled} stalled, and {completed} completed. "
                f"Current targets: {', '.join(self.all_active_queries) if self.all_active_queries else 'none'}."
            )

        if any(phrase in lower for phrase in ("scan for", "look for", "track", "find")):
            source = lower
            for prefix in ("scan for", "look for", "track", "find"):
                if prefix in source:
                    source = source.split(prefix, 1)[1]
                    break
            queries = self._parse_query_candidates(source)
            if queries:
                self._create_mission(
                    category="User Request",
                    goal=f"Find: {', '.join(queries)}",
                    queries=queries,
                )
                if self._runtime is not None:
                    self._run_single_tool("set_detection_queries_ui", {"queries": queries})
                self._sync_detection_queries()
                return f"Started searching for {', '.join(queries)}."

        if any(phrase in lower for phrase in ("where is", "locate", "focus on", "show")):
            query_source = lower
            for prefix in ("where is", "locate", "focus on", "show"):
                if prefix in query_source:
                    query_source = query_source.split(prefix, 1)[1]
                    break
            queries = self._parse_query_candidates(query_source)
            if queries:
                query = queries[0]
                locate = self._run_single_tool("locate_object_3d", {"query": query})
                if locate.get("ok") and locate.get("data", {}).get("found"):
                    return f"I located {query} and focused it in the UI."
                search = self._run_single_tool("search_scene_index", {"query": query, "max_results": 8})
                data = search.get("data", {})
                matches = data.get("matches", []) if isinstance(data, dict) else []
                if isinstance(matches, list) and matches:
                    return f"I found likely matches for {query} in the indexed scene data."
                return f"I queued a deeper scan for {query}; I will update you when it completes."

        return None

    def handle_user_message(self, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return ""

        self.chat_history.append({"role": "user", "content": message})
        self.chat_history = self.chat_history[-20:]

        chat_task = self._start_task("orchestrator", "chat_request")
        missions_summary = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
            }
            for m in self.missions.values()
        ]

        system_prompt = (
            "You are the session-specific Spatial Intelligence Agent for a live 3D scan. "
            "Be concise, practical, and mission-driven. Output strict JSON only."
        )
        user_prompt = (
            "State:\n"
            f"- Room type: {self.room_type}\n"
            f"- Scene: {self.scene_description}\n"
            f"- Layout: {self.spatial_layout}\n"
            f"- Objects: {json.dumps(list(self.discovered_objects.keys()))}\n"
            f"- Missions: {json.dumps(missions_summary)}\n"
            f"- Scene index: {json.dumps(self.scene_index.summary(max_queries=8))}\n"
            f"- Available tools: {json.dumps(self._list_available_tools(observe_only=False))}\n"
            f"- Goal: {self.current_goal or 'none'}\n"
            f"- Recent chat: {json.dumps(self.chat_history[-6:])}\n"
            f"\nUser message: {message}\n\n"
            "Return JSON:\n"
            "{"
            "\"response\":\"assistant reply\","
            "\"new_missions\":[{\"category\":\"...\",\"goal\":\"...\",\"queries\":[\"q1\"]}],"
            "\"remove_queries\":[\"q\"],"
            "\"set_goal\":null,"
            "\"tool_calls\":[{\"name\":\"tool_name\",\"args\":{}}]"
            "}"
        )

        try:
            fast_reply = self._try_tool_first_chat(message)
            if fast_reply is not None:
                self.chat_history.append({"role": "assistant", "content": fast_reply})
                self.chat_history = self.chat_history[-20:]
                self._emit_thought(
                    fast_reply,
                    thought_type="chat_response",
                    subagent="orchestrator",
                    keyframe_b64=self._latest_keyframe_b64(),
                )
                self._finish_task(
                    task_id=chat_task["id"],
                    task_type="orchestrator",
                    name="chat_request",
                    status="succeeded",
                    started_at=float(chat_task["started_at"]),
                    details="tool_first_path",
                )
                self._emit_state()
                return fast_reply

            parsed, _ = self.orchestrator_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=self._recent_context_images(max_images=2),
                temperature=0.45,
                max_tokens=768,
            )

            reply = str(parsed.get("response", "I am continuing to scan and organize the scene.")).strip()

            for mission_data in parsed.get("new_missions", []):
                if not isinstance(mission_data, dict):
                    continue
                self._create_mission(
                    category=str(mission_data.get("category", "User Request")),
                    goal=str(mission_data.get("goal", message)),
                    queries=[str(q) for q in mission_data.get("queries", [])],
                )

            for q in parsed.get("remove_queries", []):
                q_lower = str(q).strip().lower()
                if not q_lower:
                    continue
                for mission in self.missions.values():
                    if q_lower in mission.queries:
                        mission.queries.remove(q_lower)

            set_goal = parsed.get("set_goal")
            if isinstance(set_goal, str) and set_goal.strip():
                self.current_goal = set_goal.strip()

            tool_calls = parsed.get("tool_calls", [])
            if isinstance(tool_calls, list) and tool_calls:
                self._execute_tool_calls(tool_calls)

            self._sync_detection_queries()
            self.chat_history.append({"role": "assistant", "content": reply})
            self.chat_history = self.chat_history[-20:]

            self._emit_thought(
                reply,
                thought_type="chat_response",
                subagent="orchestrator",
                keyframe_b64=self._latest_keyframe_b64(),
            )
            self._finish_task(
                task_id=chat_task["id"],
                task_type="orchestrator",
                name="chat_request",
                status="succeeded",
                started_at=float(chat_task["started_at"]),
                details="llm_path",
            )
            self._emit_state()
            return reply

        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] chat error: {e}")
            fallback = "I hit a temporary model issue, but I am still scanning and updating missions."
            self.chat_history.append({"role": "assistant", "content": fallback})
            self.chat_history = self.chat_history[-20:]
            self._emit_thought(fallback, thought_type="error", subagent="orchestrator")
            self._finish_task(
                task_id=chat_task["id"],
                task_type="orchestrator",
                name="chat_request",
                status="failed",
                started_at=float(chat_task["started_at"]),
                error=str(e),
            )
            return fallback

    def set_initial_context(self, goal: str, initial_queries: list[str]) -> None:
        self.current_goal = (goal or "").strip() or None

        # Build startup thought
        parts = []
        if self.current_goal:
            parts.append(f"My goal is to {self.current_goal}.")
        if initial_queries:
            q_str = ", ".join(initial_queries)
            parts.append(f"I'll actively scan the scene and search for: {q_str}.")
        if not parts:
            parts.append("I'm ready to explore the scene and report what I find.")
        self._emit_thought(" ".join(parts), thought_type="thinking")

        if self.current_goal:
            self._emit_action("goal_updated", f"Goal: {self.current_goal}")

        if initial_queries:
            self._create_mission(
                category="user_specified",
                goal=f"Locate user-specified objects: {', '.join(initial_queries)}",
                queries=list(initial_queries),
            )
            self._sync_detection_queries()

        self._emit_state()

    def set_goal(self, goal: str) -> None:
        self.set_initial_context(goal, [])

    # ------------------------------------------------------------------
    # Detection routing and query sync
    # ------------------------------------------------------------------

    def _diff_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new: list[dict[str, Any]] = []
        current_keys: set[tuple[str, int, int]] = set()

        for det in detections:
            query = str(det.get("query", "")).lower()
            submap = int(det.get("matched_submap", -1))
            frame = int(det.get("matched_frame", -1))
            key = (query, submap, frame)
            current_keys.add(key)
            if key not in self.previous_detection_keys:
                new.append(det)

        self.previous_detection_keys = current_keys
        return new

    def _mark_mission_progress(self, mission: Mission, reason: str):
        mission.submaps_since_finding = 0
        mission.last_progress_ts = time.time()
        if mission.status in {"stalled", "recovering"}:
            mission.status = "active"
            mission.stall_reason = None
            self._emit_task_event(
                {
                    "id": str(uuid.uuid4())[:12],
                    "timestamp": time.time(),
                    "task_type": "mission",
                    "name": "mission_status",
                    "status": "resumed",
                    "mission_id": mission.id,
                    "details": reason,
                }
            )
            self._emit_action("mission_resumed", f"Resumed: {mission.category}", {"mission_id": mission.id})

    def _stall_mission(self, mission: Mission, reason: str):
        if mission.status != "completed":
            mission.status = "stalled"
            mission.stall_reason = reason
            mission.stall_count += 1
            self._emit_task_event(
                {
                    "id": str(uuid.uuid4())[:12],
                    "timestamp": time.time(),
                    "task_type": "mission",
                    "name": "mission_status",
                    "status": "stalled",
                    "mission_id": mission.id,
                    "details": reason,
                }
            )
            self._emit_action(
                "mission_stalled",
                f"Stalled: {mission.category} ({reason})",
                {"mission_id": mission.id},
            )

    def _attempt_mission_recovery(self, mission: Mission, reason: str):
        if mission.status == "completed":
            return
        missing = [q for q in mission.queries if q not in mission.found]
        if not missing:
            return
        rewritten = self._rewrite_queries_for_recovery(missing)
        added = 0
        for q in rewritten:
            if q not in mission.queries:
                mission.queries.append(q)
                added += 1
        mission.status = "recovering"
        mission.stall_reason = reason
        mission.stall_count += 1
        self._emit_task_event(
            {
                "id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "task_type": "mission",
                "name": "mission_status",
                "status": "stalled",
                "mission_id": mission.id,
                "details": f"{reason}; rewritten={added}",
            }
        )
        self._emit_action(
            "mission_recovering",
            f"Recovering: {mission.category} ({reason})",
            {"mission_id": mission.id, "added_queries": rewritten[:6]},
        )
        self._enqueue_deep_scan(missing[:4], mission_id=mission.id, top_k=3)

    @staticmethod
    def _rewrite_queries_for_recovery(queries: list[str]) -> list[str]:
        rewritten: list[str] = []
        alias_map: dict[str, list[str]] = {
            "sofa": ["couch"],
            "couch": ["sofa"],
            "tv": ["television", "monitor"],
            "trash can": ["bin", "garbage can"],
            "fridge": ["refrigerator"],
            "sink": ["faucet"],
            "microwave": ["oven"],
            "chair": ["seat"],
        }
        for q in queries:
            qn = str(q).strip().lower()
            if not qn:
                continue
            variants = [f"{qn} object", f"{qn} item"]
            for alias in alias_map.get(qn, []):
                variants.append(alias)
                variants.append(f"{alias} object")
            for v in variants:
                if v not in rewritten:
                    rewritten.append(v[:120])
        return rewritten[:8]

    def _update_mission_stall_states(self):
        changed = False
        for mission in self.missions.values():
            if mission.status not in {"active", "recovering"}:
                continue
            if mission.submaps_since_finding < self.stall_submap_gap:
                continue
            if len(mission.found) == len(mission.queries) and mission.queries:
                mission.status = "completed"
                self._emit_action(
                    "mission_completed",
                    f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                    {"mission_id": mission.id},
                )
                changed = True
                continue
            self._attempt_mission_recovery(
                mission,
                reason=f"no_progress_for_{mission.submaps_since_finding}_submaps",
            )
            changed = True
        if changed:
            self._sync_detection_queries(force=True)

    def _route_detections(self, new_detections: list[dict[str, Any]]):
        if new_detections:
            self.scene_index.ingest(new_detections)
        for det in new_detections:
            query = str(det.get("query", "")).strip().lower()
            if not query:
                continue

            confidence = float(det.get("confidence", 0.0) or 0.0)
            self.discovered_objects.setdefault(query, []).append(det)

            already_found = any(
                query in mission.found
                for mission in self.missions.values()
                if query in mission.queries
            )

            for mission in self.missions.values():
                if mission.status == "completed":
                    continue
                if query in mission.queries:
                    if query in mission.found:
                        continue  # already reported; skip duplicate emission
                    mission.found.add(query)
                    mission.findings.append(det)
                    mission.confidence = len(mission.found) / max(1, len(mission.queries))
                    self._mark_mission_progress(mission, reason=f"detection:{query}")
                    if mission.queries and len(mission.found) >= len(mission.queries):
                        mission.status = "completed"
                        self._emit_action(
                            "mission_completed",
                            f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                            {"mission_id": mission.id},
                        )

            if not already_found:
                evidence = {
                    "matched_submap": det.get("matched_submap"),
                    "matched_frame": det.get("matched_frame"),
                }
                self._emit_finding(
                    query=query,
                    description=f"Found {query} ({confidence:.0%})",
                    confidence=confidence,
                    position=det.get("bounding_box", {}).get("center") if det.get("bounding_box") else None,
                    mission_id=self._find_mission_for_query(query),
                    evidence=evidence,
                )

        for mission in self.missions.values():
            if mission.status == "completed":
                continue

            has_new = any(str(d.get("query", "")).strip().lower() in mission.queries for d in new_detections)
            if not has_new:
                mission.submaps_since_finding += 1

            if (
                mission.status == "active"
                and mission.submaps_since_finding >= self.auto_complete_submap_gap
                and len(mission.found) > 0
                and mission.confidence >= 0.5
            ):
                mission.status = "completed"
                self._emit_action(
                    "mission_completed",
                    f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                    {"mission_id": mission.id},
                )
        self._update_mission_stall_states()

    def _sync_detection_queries(self, force: bool = False):
        active_queries: list[str] = []
        for mission in self.missions.values():
            if mission.status not in {"active", "recovering"}:
                continue
            for q in mission.queries:
                q_norm = q.strip().lower()
                if q_norm and q_norm not in active_queries:
                    active_queries.append(q_norm)

        active_queries = active_queries[: self.max_active_queries]
        if active_queries == self.all_active_queries:
            return

        now = time.time()
        if (not force) and (now - self._last_query_sync_ts < self.query_update_cooldown_s):
            return

        self._last_query_sync_ts = now
        self.all_active_queries = active_queries

        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, list(self.all_active_queries))
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] query callback error: {e}")

        details = ", ".join(self.all_active_queries) if self.all_active_queries else "none"
        self._emit_action(
            "queries_updated",
            f"Searching for: {details}",
            {"queries": list(self.all_active_queries)},
        )

    # ------------------------------------------------------------------
    # State + reset
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        self._refresh_job_states()
        missions_list = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
                "confidence": m.confidence,
                "findings_count": len(m.findings),
                "last_progress_ts": m.last_progress_ts,
                "stall_reason": m.stall_reason,
                "stall_count": m.stall_count,
            }
            for m in self.missions.values()
        ]

        with self._task_lock:
            active_tasks = list(self._active_tasks.values())
        with self._job_lock:
            all_jobs = [self._serialize_job(j, include_result=False) for j in self._jobs.values()]
        pending_jobs = [j for j in all_jobs if j.get("status") == "queued"]
        running_jobs = [j for j in all_jobs if j.get("status") == "running"]
        return {
            "enabled": self.enabled,
            "scene_description": self.scene_description,
            "room_type": self.room_type,
            "missions": missions_list,
            "active_queries": list(self.all_active_queries),
            "discovered_objects": sorted(self.discovered_objects.keys()),
            "current_goal": self.current_goal,
            "submaps_processed": self._submap_count,
            "coverage_estimate": min(max(self._coverage_estimate, 0.0), 1.0),
            "health": "degraded" if self.orchestrator_client.degraded_mode else "ok",
            "degraded_mode": self.orchestrator_client.degraded_mode,
            "runtime_v2_enabled": self.runtime_v2_enabled,
            "active_tasks": active_tasks,
            "pending_jobs": pending_jobs,
            "running_jobs": running_jobs,
            "last_job_errors": list(self._recent_job_errors),
            "orchestrator_busy": any(
                task.get("task_type") == "orchestrator" for task in active_tasks
            ),
        }

    def reset(self):
        with self._lock:
            self.scene_description = ""
            self.room_type = "unknown"
            self.spatial_layout = ""
            self.scene_history.clear()
            self.missions.clear()
            self.next_mission_id = 1
            self.all_active_queries.clear()
            self.discovered_objects.clear()
            self.scene_index.clear()
            self.previous_detection_keys.clear()
            self.current_goal = None
            self.chat_history.clear()
            self.visual_memory.clear()
            self._processing = False
            self._submap_count = 0
            self._coverage_estimate = 0.0
            self._last_query_sync_ts = 0.0
            self._last_cycle_ts = 0.0
            with self._task_lock:
                self._active_tasks.clear()
            with self._job_lock:
                for job in self._jobs.values():
                    future = job.get("future")
                    if future is not None:
                        future.cancel()
                self._jobs.clear()
            self._recent_job_errors.clear()

        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, [])
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] reset callback error: {e}")

    def usage_summary(self) -> dict[str, Any]:
        """Session LLM usage: call count, tokens, and real USD as reported by
        OpenRouter. ``cost_basis`` says how much of the total is actually priced —
        an unreported call is never counted as free."""
        return self.usage.summary()

    def shutdown(self):
        self.enabled = False
        # Print before tearing down: this line is the per-session LLM cost record
        # until the ledger lands, and Modal's log stream is where it gets mined.
        try:
            summary = self.usage.summary()
            print(
                f"[cost] session={self.session_id} llm_calls={summary['llm_calls']} "
                f"prompt_tokens={summary['prompt_tokens']} "
                f"completion_tokens={summary['completion_tokens']} "
                f"cost_usd={summary['cost_usd']} basis={summary['cost_basis']} "
                f"by_model={summary['by_model']}"
            )
        except Exception as e:  # pragma: no cover - accounting must never break teardown
            print(f"SpatialAgent[{self.session_id}] usage summary error: {e}")
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:
                pass
        self._job_executor.shutdown(wait=False, cancel_futures=True)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit_thought(
        self,
        content: str,
        thought_type: str = "observation",
        subagent: Optional[str] = None,
        confidence: Optional[float] = None,
        keyframe_b64: Optional[str] = None,
    ):
        data: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "type": thought_type,
            "content": content,
        }
        if subagent:
            data["subagent"] = subagent
        if confidence is not None:
            data["confidence"] = float(confidence)
        if keyframe_b64:
            data["keyframe_b64"] = keyframe_b64
            data["attachments"] = [{"kind": "keyframe", "image_b64": keyframe_b64}]
        try:
            self.emit("agent_thought", data)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit thought error: {e}")

    def _emit_action(self, action: str, details: str, extra: Optional[dict[str, Any]] = None):
        data: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "action": action,
            "details": details,
        }
        if extra:
            data.update(extra)
        try:
            self.emit("agent_action", data)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit action error: {e}")

    def _emit_task_event(self, payload: dict[str, Any]):
        try:
            self.emit("agent_task_event", payload)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit task event error: {e}")

    def _emit_tool_event_payload(self, payload: dict[str, Any]):
        try:
            self.emit("agent_tool_event", payload)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit tool event error: {e}")

    def _emit_finding(
        self,
        query: str,
        description: str,
        confidence: float,
        position: Optional[list] = None,
        mission_id: Optional[int] = None,
        evidence: Optional[dict[str, Any]] = None,
    ):
        data: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "query": query,
            "description": description,
            "confidence": confidence,
            "position": position,
            "mission_id": mission_id,
        }
        if evidence:
            data["evidence"] = evidence
        try:
            self.emit("agent_finding", data)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit finding error: {e}")

    def _emit_state(self):
        try:
            self.emit("agent_state", self.get_state())
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit state error: {e}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _create_mission(self, category: str, goal: str, queries: list[str]):
        normalized = []
        for query in queries:
            q = str(query).strip().lower()
            if q and q not in normalized:
                normalized.append(q)

        if not normalized:
            return

        if len(self.missions) >= self.max_missions:
            return

        mission = Mission(
            id=self.next_mission_id,
            category=category,
            goal=goal,
            queries=normalized,
        )
        self.missions[mission.id] = mission
        self.next_mission_id += 1

        self._emit_action(
            "mission_created",
            f"New mission: {mission.category} - {mission.goal}",
            {"mission_id": mission.id, "queries": mission.queries},
        )

    def _extract_keyframes_b64(self, submap_id: int) -> list[str]:
        try:
            submap = None
            graph_map = self.slam.solver.map
            try:
                submap = graph_map.get_submap(int(submap_id))
            except Exception:
                submap = None
            if submap is None:
                try:
                    submap = graph_map.get_latest_submap(ignore_loop_closure_submaps=True)
                except Exception:
                    submap = None
            if submap is None:
                try:
                    submap = graph_map.get_latest_submap()
                except Exception:
                    submap = None
            if submap is None:
                return []

            all_frames = submap.get_all_frames()
            if all_frames is None:
                return []

            num_frames = int(all_frames.shape[0]) if hasattr(all_frames, "shape") else len(all_frames)
            if num_frames <= 0:
                return []

            target_keyframes = min(max(1, self.max_keyframes_per_analysis), num_frames)
            if num_frames <= target_keyframes:
                indices = list(range(num_frames))
            else:
                step = max(1, num_frames // target_keyframes)
                indices = list(range(0, num_frames, step))[:target_keyframes]
                if indices[-1] != num_frames - 1:
                    indices[-1] = num_frames - 1
            keyframes: list[str] = []

            for idx in indices:
                try:
                    frame_tensor = submap.get_frame_at_index(idx)
                    frame_np = (
                        frame_tensor.detach()
                        .cpu()
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)

                    h, w = frame_np.shape[:2]
                    if w > 640:
                        scale = 640.0 / float(w)
                        frame_np = cv2.resize(frame_np, (640, int(h * scale)))

                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    ok, jpeg_buf = cv2.imencode(
                        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
                    )
                    if ok:
                        keyframes.append(base64.b64encode(jpeg_buf.tobytes()).decode("ascii"))
                except Exception as e:
                    print(f"SpatialAgent[{self.session_id}] keyframe extraction error idx={idx}: {e}")

            return keyframes
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] keyframe extraction error: {e}")
            return []

    def _remember_keyframes(self, submap_id: int, keyframes_b64: list[str]):
        ts = time.time()
        for b64 in keyframes_b64:
            self.visual_memory.append(
                {
                    "timestamp": ts,
                    "submap_id": submap_id,
                    "image_b64": b64,
                }
            )

    def _recent_context_images(self, max_images: int = 4) -> list[str]:
        images: list[str] = []
        seen = set()
        for item in reversed(self.visual_memory):
            img = item.get("image_b64")
            if not img or img in seen:
                continue
            images.append(img)
            seen.add(img)
            if len(images) >= max_images:
                break
        return images

    def _latest_keyframe_b64(self) -> Optional[str]:
        if not self.visual_memory:
            return None
        latest = self.visual_memory[-1]
        img = latest.get("image_b64")
        return img if isinstance(img, str) else None

    def _find_mission_for_query(self, query: str) -> Optional[int]:
        for mission in self.missions.values():
            if query in mission.queries:
                return mission.id
        return None

    @staticmethod
    def _parse_csv_env(env_name: str, default: str) -> list[str]:
        raw = os.environ.get(env_name, default)
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _safe_future_result(
        self,
        future,
        task_name: str,
        timeout: float = 12.0,
        task_id: Optional[str] = None,
        started_at: Optional[float] = None,
        task_type: str = "subagent",
        mission_id: Optional[int] = None,
    ):
        t0 = started_at or time.time()
        wait_slice = min(0.5, max(0.1, float(timeout)))
        next_heartbeat = time.time() + self.task_heartbeat_s
        try:
            while True:
                remaining = timeout - (time.time() - t0)
                if remaining <= 0:
                    raise FuturesTimeout()
                try:
                    result = future.result(timeout=min(wait_slice, max(0.01, remaining)))
                    if task_id is not None:
                        self._finish_task(
                            task_id=task_id,
                            task_type=task_type,
                            name=task_name,
                            status="succeeded",
                            started_at=t0,
                            mission_id=mission_id,
                        )
                    return result
                except FuturesTimeout:
                    if task_id is not None and time.time() >= next_heartbeat:
                        self._update_active_task(task_id, "heartbeat")
                        self._emit_task_event(
                            {
                                "id": task_id,
                                "timestamp": time.time(),
                                "task_type": task_type,
                                "name": task_name,
                                "status": "heartbeat",
                                "mission_id": mission_id,
                            }
                        )
                        next_heartbeat = time.time() + self.task_heartbeat_s
        except FuturesTimeout:
            print(f"SpatialAgent task timed out: {task_name}")
            if task_id is not None:
                self._finish_task(
                    task_id=task_id,
                    task_type=task_type,
                    name=task_name,
                    status="timed_out",
                    started_at=t0,
                    mission_id=mission_id,
                    error=f"{task_name} timed out",
                )
            return None
        except Exception as e:
            print(f"SpatialAgent task error ({task_name}): {e}")
            if task_id is not None:
                self._finish_task(
                    task_id=task_id,
                    task_type=task_type,
                    name=task_name,
                    status="failed",
                    started_at=t0,
                    mission_id=mission_id,
                    error=str(e),
                )
            return None
