"""Tool registry with strict argument validation and execution timeouts."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails validation or execution."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]


class ToolRegistry:
    def __init__(self, max_workers: int = 4):
        self._tools: dict[str, ToolDefinition] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.args_model.model_json_schema(),
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, args: dict[str, Any], timeout_s: float = 10.0) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolExecutionError(f"Unknown tool: {name}")

        try:
            model = spec.args_model.model_validate(args or {})
        except ValidationError as exc:
            raise ToolExecutionError(f"Invalid args for tool '{name}': {exc}") from exc

        t0 = time.time()
        future = self._executor.submit(spec.handler, model)
        try:
            result = future.result(timeout=max(0.1, float(timeout_s)))
        except concurrent.futures.TimeoutError as exc:
            raise ToolExecutionError(f"Tool '{name}' timed out") from exc
        except Exception as exc:  # pragma: no cover - runtime variance
            raise ToolExecutionError(f"Tool '{name}' failed: {exc}") from exc

        if not isinstance(result, dict):
            result = {"value": result}

        result.setdefault("latency_ms", int((time.time() - t0) * 1000))
        return result

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
