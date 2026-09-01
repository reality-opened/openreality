"""UI command tools for hybrid agent runtime."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from server.agent.schemas import AgentUICommand


class UITools:
    def __init__(
        self,
        emit_ui_command: Callable[[dict[str, Any]], None],
        default_ttl_ms: int = 8000,
    ):
        self._emit_ui_command = emit_ui_command
        self._default_ttl_ms = int(default_ttl_ms)

    def _emit(self, name: str, args: dict[str, Any], mission_id: Optional[int] = None) -> dict[str, Any]:
        cmd = AgentUICommand(
            id=str(uuid.uuid4())[:12],
            name=name,
            args=args,
            mission_id=mission_id,
            ttl_ms=self._default_ttl_ms,
        )
        payload = cmd.model_dump()
        payload["timestamp"] = time.time()
        self._emit_ui_command(payload)
        return {"command_id": cmd.id, "command": cmd.name}

    def set_detection_queries_ui(self, args) -> dict[str, Any]:
        queries = [str(q).strip().lower() for q in args.queries if str(q).strip()]
        return self._emit("set_detection_queries", {"queries": queries})

    def focus_detection_ui(self, args) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if args.query:
            payload["query"] = str(args.query).strip().lower()
        if args.submap_id is not None:
            payload["submap_id"] = int(args.submap_id)
        if args.frame_idx is not None:
            payload["frame_idx"] = int(args.frame_idx)
        if args.center is not None:
            payload["center"] = [float(v) for v in args.center]
        return self._emit("focus_detection", payload)

    def show_waypoint_ui(self, args) -> dict[str, Any]:
        return self._emit(
            "show_waypoint",
            {
                "waypoint_id": str(args.waypoint_id),
                "position": [float(v) for v in args.position],
                "label": args.label,
            },
        )

    def show_path_ui(self, args) -> dict[str, Any]:
        return self._emit(
            "show_path",
            {
                "path_id": str(args.path_id),
                "points": [[float(v) for v in pt] for pt in args.points],
            },
        )

    def show_toast_ui(self, args) -> dict[str, Any]:
        return self._emit(
            "show_toast",
            {
                "message": str(args.message),
                "level": str(args.level),
            },
        )

    def open_detection_preview_ui(self, args) -> dict[str, Any]:
        return self._emit(
            "open_detection_preview",
            {
                "submap_id": int(args.submap_id),
                "frame_idx": int(args.frame_idx),
                "query": str(args.query).strip().lower(),
            },
        )
