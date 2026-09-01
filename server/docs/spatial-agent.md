# Spatial Agent System

> The autonomous spatial intelligence layer that runs alongside the SLAM loop in [server/app.py](streaming-server.md). Sessions are scoped per-connection.

**Key files:**
- `server/spatial_agent.py` — `SpatialAgent` orchestrates exploration with a multi-mission tracker (`Mission` dataclass), stall detection, and adaptive query generation. Uses OpenRouter.
- `server/agent/runtime.py` — `AgentRuntime` registers VGGT + UI tools, validates args with Pydantic, executes them via a thread pool, and emits SocketIO events.
- `server/agent/schemas.py` — Pydantic schemas: `GetSceneSnapshotArgs`, `SearchObjectsArgs`, `LocateObject3DArgs`, `AgentUICommand`, `ToolCall`, etc.
- `server/agent/tool_registry.py` — `ToolRegistry` + `ToolDefinition`; thread-pool executor with timeouts.
- `server/agent/scene_index.py` — `SceneIndex`, a thread-safe deduplicating in-memory detection cache (`ingest`, `search`, `best_recent`).
- `server/agent/tools/vggt_tools.py` — SLAM-backed tools (scene snapshot, object search, 3D detection, spatial relations).
- `server/agent/tools/ui_tools.py` — UI commands (waypoints, beacons, detection previews, toasts).
- `server/llm/openrouter_client.py` — `OpenRouterClient` with retries, fallback chains, JSON parsing.

**Frontend surfaces:** `plan.html`/`plan.ts` for mission planning, `summary.html`/`summary.ts` for detection results, plus `AgentPanel` in `index.html`. See [frontend.md](frontend.md).

**Config env vars:**
- `SPATIAL_ORCH_MODEL` — orchestrator LLM (default `google/gemini-3-flash-preview`).
- `SPATIAL_SUBAGENT_MODEL` — subagent LLM (default `google/gemini-3-flash-preview`).
- `SPATIAL_ORCH_FALLBACKS` — CSV fallback models (default `google/gemini-3-flash-preview,openai/gpt-4o-mini`).
- `SPATIAL_SUBAGENT_FALLBACKS` — CSV fallback models for subagent.
- `CORS_ALLOWED_ORIGINS` — CORS policy (default `*`).
