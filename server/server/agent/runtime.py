"""Session-scoped tool runtime used by the SpatialAgent orchestrator."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from server.agent.schemas import (
    AddDetectionObjectArgs,
    AgentToolEvent,
    GetSceneSnapshotArgs,
    InferSpatialRelationsArgs,
    InspectDetectionArgs,
    LocateObject3DArgs,
    OpenDetectionPreviewUIArgs,
    ProposeNextScanFocusArgs,
    RemoveDetectionObjectArgs,
    SearchObjectsArgs,
    SetDetectionQueriesUIArgs,
    FocusDetectionUIArgs,
    ShowWaypointUIArgs,
    ShowPathUIArgs,
    ShowToastUIArgs,
)
from server.agent.tool_registry import ToolDefinition, ToolExecutionError, ToolRegistry
from server.agent.tools.ui_tools import UITools
from server.agent.tools.vggt_tools import VGGTTools


class AgentRuntime:
    """Validated tool execution + event emission for one agent session."""

    def __init__(
        self,
        session_id: str,
        streaming_slam,
        emit_event: Callable[[str, dict[str, Any]], None],
        max_workers: int = 4,
        on_query_added=None,
    ):
        self.session_id = session_id
        self.slam = streaming_slam
        self.emit_event = emit_event

        self.registry = ToolRegistry(max_workers=max_workers)
        self.vggt = VGGTTools(streaming_slam, on_query_added=on_query_added)
        self.ui = UITools(emit_ui_command=lambda cmd: self.emit_event("agent_ui_command", cmd))

        self._register_tools()

    def _register_tools(self):
        # VGGT-backed tools
        self.registry.register(
            ToolDefinition(
                name="get_scene_snapshot",
                description="Return map/submap/detection summary from VGGT-SLAM state.",
                args_model=GetSceneSnapshotArgs,
                handler=self.vggt.get_scene_snapshot,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="search_objects",
                description="Run CLIP+SAM object search over current map and return detections.",
                args_model=SearchObjectsArgs,
                handler=self.vggt.search_objects,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="inspect_detection",
                description="Fetch keyframe + segmentation + optional 3D bbox for one frame/query.",
                args_model=InspectDetectionArgs,
                handler=self.vggt.inspect_detection,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="locate_object_3d",
                description="Locate best 3D detection for a query from deduped detections.",
                args_model=LocateObject3DArgs,
                handler=self.vggt.locate_object_3d,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="infer_spatial_relations",
                description="Infer pairwise spatial relations between detected objects.",
                args_model=InferSpatialRelationsArgs,
                handler=self.vggt.infer_spatial_relations,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="propose_next_scan_focus",
                description="Suggest next scan targets based on unresolved/low-confidence detections.",
                args_model=ProposeNextScanFocusArgs,
                handler=self.vggt.propose_next_scan_focus,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="add_detection_object",
                description=(
                    "Add a single object query to the live detection worker pool. "
                    "Results stream to the viewer as submaps are scanned."
                ),
                args_model=AddDetectionObjectArgs,
                handler=self.vggt.add_detection_object,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="remove_detection_object",
                description="Remove a single object from the active detection pool and clear its bounding boxes.",
                args_model=RemoveDetectionObjectArgs,
                handler=self.vggt.remove_detection_object,
            )
        )

        # UI command tools
        self.registry.register(
            ToolDefinition(
                name="set_detection_queries_ui",
                description="Update detection target list on UI.",
                args_model=SetDetectionQueriesUIArgs,
                handler=self.ui.set_detection_queries_ui,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="focus_detection_ui",
                description="Focus camera/view on a detected object in the UI.",
                args_model=FocusDetectionUIArgs,
                handler=self.ui.focus_detection_ui,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="show_waypoint_ui",
                description="Render waypoint marker in UI.",
                args_model=ShowWaypointUIArgs,
                handler=self.ui.show_waypoint_ui,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="show_path_ui",
                description="Render suggested path polyline in UI.",
                args_model=ShowPathUIArgs,
                handler=self.ui.show_path_ui,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="show_toast_ui",
                description="Show a toast notification in UI.",
                args_model=ShowToastUIArgs,
                handler=self.ui.show_toast_ui,
            )
        )
        self.registry.register(
            ToolDefinition(
                name="open_detection_preview_ui",
                description="Open detection preview modal for a specific submap/frame/query.",
                args_model=OpenDetectionPreviewUIArgs,
                handler=self.ui.open_detection_preview_ui,
            )
        )

    def execute_tool(self, tool_name: str, args: dict[str, Any], timeout_s: float = 10.0) -> dict[str, Any]:
        event_id = str(uuid.uuid4())[:12]
        self._emit_tool_event(
            AgentToolEvent(
                id=event_id,
                tool=tool_name,
                status="started",
                args=args,
            )
        )

        t0 = time.time()
        try:
            result = self.registry.execute(tool_name, args, timeout_s=timeout_s)
            elapsed = int((time.time() - t0) * 1000)
            self._emit_tool_event(
                AgentToolEvent(
                    id=event_id,
                    tool=tool_name,
                    status="succeeded",
                    result=result,
                    latency_ms=elapsed,
                )
            )
            return {
                "ok": True,
                "tool": tool_name,
                "data": result,
                "latency_ms": elapsed,
            }
        except ToolExecutionError as exc:
            elapsed = int((time.time() - t0) * 1000)
            self._emit_tool_event(
                AgentToolEvent(
                    id=event_id,
                    tool=tool_name,
                    status="failed",
                    error=str(exc),
                    latency_ms=elapsed,
                )
            )
            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
                "latency_ms": elapsed,
            }

    def list_tools(self) -> list[dict[str, Any]]:
        return self.registry.list_tools()

    def close(self):
        self.registry.shutdown()

    def _emit_tool_event(self, event: AgentToolEvent):
        self.emit_event("agent_tool_event", event.model_dump())


__all__ = ["AgentRuntime", "ToolExecutionError"]
