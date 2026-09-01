"""Typed schemas for agent tool calls and UI command exchange."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class StrictModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }


# ------------------------------------------------------------------
# Runtime envelopes
# ------------------------------------------------------------------


class ToolCall(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    args: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(StrictModel):
    ok: bool
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class AgentUICommand(StrictModel):
    id: str = Field(min_length=4, max_length=64)
    name: Literal[
        "focus_detection",
        "set_detection_queries",
        "add_detection_query",
        "remove_detection_query",
        "show_waypoint",
        "show_path",
        "show_toast",
        "open_detection_preview",
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    mission_id: Optional[int] = None
    ttl_ms: Optional[int] = Field(default=8000, ge=100, le=120000)


class AgentUIResult(StrictModel):
    id: str = Field(min_length=4, max_length=64)
    status: Literal["ok", "error", "ignored", "timeout"]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class AgentToolEvent(StrictModel):
    id: str = Field(min_length=4, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    status: Literal["started", "succeeded", "failed"]
    args: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class AgentJobEvent(StrictModel):
    id: str = Field(min_length=4, max_length=64)
    job_name: str = Field(min_length=1, max_length=128)
    status: Literal["queued", "running", "succeeded", "failed", "timed_out", "canceled"]
    mission_id: Optional[int] = None
    args: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


# ------------------------------------------------------------------
# VGGT-backed tool args
# ------------------------------------------------------------------


class GetSceneSnapshotArgs(StrictModel):
    include_detections: bool = True


class SearchObjectsArgs(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=16)
    max_results: int = Field(default=40, ge=1, le=200)


class InspectDetectionArgs(StrictModel):
    submap_id: int = Field(ge=0)
    frame_idx: int = Field(ge=0)
    query: str = Field(min_length=1, max_length=120)


class LocateObject3DArgs(StrictModel):
    query: str = Field(min_length=1, max_length=120)


class InferSpatialRelationsArgs(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=16)


class ProposeNextScanFocusArgs(StrictModel):
    goal: Optional[str] = Field(default=None, max_length=300)
    max_queries: int = Field(default=4, ge=1, le=10)


class GetVisualContextSummaryArgs(StrictModel):
    max_items: int = Field(default=8, ge=1, le=24)
    include_missions: bool = True


class RequestVisualContextImagesArgs(StrictModel):
    k: int = Field(default=2, ge=1, le=6)
    purpose: Optional[str] = Field(default=None, max_length=180)
    query: Optional[str] = Field(default=None, max_length=120)


class SearchSceneIndexArgs(StrictModel):
    query: str = Field(min_length=1, max_length=120)
    max_results: int = Field(default=8, ge=1, le=24)
    auto_deep_scan_on_miss: bool = True


class AddDetectionObjectArgs(StrictModel):
    query: str = Field(min_length=1, max_length=120)


class RemoveDetectionObjectArgs(StrictModel):
    query: str = Field(min_length=1, max_length=120)


class StartDeepScanArgs(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=12)
    mission_id: Optional[int] = Field(default=None, ge=1)
    top_k: int = Field(default=3, ge=1, le=8)


class GetJobStatusArgs(StrictModel):
    job_id: str = Field(min_length=6, max_length=64)


class CancelJobArgs(StrictModel):
    job_id: str = Field(min_length=6, max_length=64)


# ------------------------------------------------------------------
# UI tool args
# ------------------------------------------------------------------


class SetDetectionQueriesUIArgs(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=16)


class FocusDetectionUIArgs(StrictModel):
    query: Optional[str] = Field(default=None, max_length=120)
    submap_id: Optional[int] = Field(default=None, ge=0)
    frame_idx: Optional[int] = Field(default=None, ge=0)
    center: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)


class ShowWaypointUIArgs(StrictModel):
    waypoint_id: str = Field(min_length=1, max_length=64)
    position: list[float] = Field(min_length=3, max_length=3)
    label: Optional[str] = Field(default=None, max_length=120)


class ShowPathUIArgs(StrictModel):
    points: list[list[float]] = Field(default_factory=list, max_length=200)
    path_id: str = Field(default="agent-path", min_length=1, max_length=64)


class ShowToastUIArgs(StrictModel):
    message: str = Field(min_length=1, max_length=280)
    level: Literal["info", "success", "error"] = "info"


class OpenDetectionPreviewUIArgs(StrictModel):
    submap_id: int = Field(ge=0)
    frame_idx: int = Field(ge=0)
    query: str = Field(min_length=1, max_length=120)
