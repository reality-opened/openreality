# Streaming / Web Server Mode (`server/`)

> The Flask + python-socketio ASGI server (`server/app.py`) and the frame-by-frame SLAM wrapper (`server/streaming_slam.py`).
> Deployed per user on Modal (see [modal-deployment.md](modal-deployment.md)). Hosts the [spatial agent](spatial-agent.md) and [scene report](scene-report.md) systems.

## `server/app.py` — Flask + python-socketio ASGI server (~2500 lines)

Main entry point for the streaming SLAM. Exposes both HTTP routes and SocketIO events; integrates Clerk auth (scan-quota gated — see [access-control.md](access-control.md)), spatial agents, demo-video playback, and an assistant chat endpoint.

**HTTP routes:**
- `GET /health` — returns `{status, gpu, gpu_name}` on an allocated worker; used by the landing dashboard to gate the "Launch Demo" button.
- `POST /session` / `GET /session/status` — broker-only routes that allocate or report the authenticated user's dedicated GPU worker.
- `POST /auth/session` — verifies a Clerk JWT, enforces the caller's scan quota (`_require_scan_access`) + worker-owner check, and sets the `slam_session` cookie. See [access-control.md](access-control.md).
- `GET /auth/qr-token` — issues a short-lived bearer for the phone-sender QR handshake (same scan-quota + owner checks).
- `POST /api/session/refresh` — exchanges a verified Clerk JWT for a **durable broker session token** (HS256, independent expiry) accepted by `_verify_any_token` on every HTTP route; the revisit page mints one on load so its history reads + Q&A outlive the static hash JWT. See [access-control.md](access-control.md) / [gotchas.md](gotchas.md).
- `POST /reset` — soft reset: clears SLAM data without unloading VGGT/CLIP; emits `slam_reset`.
- `GET /api/demo/videos` — lists pre-uploaded demo clips (Modal Volume `vggt-slam-demo-videos`).
- `POST /api/demo/start` / `POST /api/demo/stop` — start/stop server-side `VideoFeeder` that pushes frames into the SLAM pipeline.
- `GET /api/demo/status` — current demo state (running, current `video_id`).
- `GET /api/demo/video` — stream a demo video back to a viewer client.
- `POST /api/plan` — natural-language mission plan via Gemini (`gemini-1.5-flash`); falls back to keyword extraction on error.
- `POST /api/assistant/chat` — grounded "talk to your scan" chat for the summary page. Grounds in the worker's cached scene report (`_last_scene_report`, final or in-scan progressive) via the shared `_grounded_scene_answer` core — the **same** grounding as the revisit `…/qa` route — not client-supplied context. See [scene-report.md](scene-report.md).
- Scene-history routes (`GET/DELETE /api/scenes…`, `POST /api/scenes/<id>/qa`) — see [scene-report.md](scene-report.md).

**SocketIO events (server → client):** `slam_update`, `global_map`, `slam_reset`, `slam_stopped`, `beacon_queued`, `detection_preview`, `scene_report_update` (live in-scan report) / `scene_report_ready` (end-of-scan report; see [scene-report.md](scene-report.md)), plus agent-specific events emitted by `AgentRuntime` (mission updates, tool calls, waypoints).

**SocketIO events (client → server):** `frame` (base64 JPEG; auto-starts SLAM on first frame), `stop_slam`, `set_detection_queries`, `get_detection_preview`, `place_beacon`, `clear_beacons`, `get_global_map`, `get_scene_report` (fetch the cached end-of-scan report), plus agent commands (start/stop missions, manual tool invocations).

**How a scan ends (two paths, both persist):**
- **Clean stop** — `stop_slam` → flush-stop → drain the frame backlog (`wait_for_drain=True`; a live 10 fps sender leaves a deep queue whose race once killed every native scan's persist) → build the final report → **persist durably → then emit `scene_report_ready`** (the native app holds its "Saving" state until that event and immediately lists the library, so the scene row must already exist — see [scene-report.md](scene-report.md)).
- **Abandoned scan (salvage)** — a last client that vanishes without `stop_slam` (dead transport, killed app, locked phone) schedules `_salvage_abandoned_scan`: after `SCAN_SALVAGE_GRACE_S` (default 15 s) with no reconnect, the worker runs the same finalize+persist as a clean stop (`finalize_scan_blocking`). A reconnect or `/reset` voids the pending salvage (generation counter, thread-safe). Demo playback keeps abandon-on-close. Only the un-flushed partial submap is lost; every completed submap persists. The last disconnect deliberately does **not** wipe `accumulated_detections` or the CLIP/SAM cache anymore — the salvage needs the objects, and reconnect re-scans run warm; a fresh claim still wipes via `/reset`.

**Live detection emit budget:** `detection_partial` / `slam_update` payloads are copied through `_wire_detections`, which drops `mask_rle` (the dominant field — O(2·H) run-length ints per detection — that **no live client decodes**; exports and persistence keep it on the stored dicts). Intermediate `detection_partial`s are paced per sid to one per `DETECTION_PARTIAL_MIN_INTERVAL_S` (default 0.3 s); every query worker's final always emits. Rationale: each per-query worker yields the full accumulated detection superset once **per submap**, so N queries × M submaps of masked detections used to serialize thousands of payloads inline on the event loop, starving socket.io pings until live phones timed out mid-scan (measured 2026-08-13).

**`VideoFeeder` class:** reads from a video file and pushes frames into `frame_queue` for offline testing. Supports FPS throttling (`--video-fps`) or fast-forward (`--fast`).

**SSL:** local mode loads `server/webserver/server.cert` + `server.key` for HTTPS (phone camera access via WebRTC requires it). Modal deployment skips SSL because the Modal tunnel provides HTTPS.

```bash
# Local: live camera
python -m server.app --port 5000

# Local: replay a video for testing
python -m server.app --video /path/to/video.mp4 --video-fps 2 --submap-size 8
```

## `server/streaming_slam.py` — `StreamingSLAM` (~1150 lines)

Wraps `Solver` for frame-by-frame streaming (no Viser, `skip_viewer=True`).

**Initialization:** loads VGGT-1B + `ObjectDetector` (PE-Core CLIP + SAM3). Sets CLIP model on `Solver` via `solver.set_clip_model()` so `run_predictions()` can compute per-frame semantic embeddings. ObjectDetector uses `torch.autocast` for mixed precision (see [testing.md](testing.md) latency notes).

**Processing loop** (`process_loop()`):
1. Reads base64 JPEG frames from `frame_queue`.
2. Optical-flow disparity gating skips low-motion frames.
3. Keyframes written to a `tempfile.mkdtemp()` directory.
4. When `submap_size + 1` keyframes accumulate, calls `process_submap()`.

**Submap processing** (`process_submap()`):
1. `solver.run_predictions()` → VGGT inference.
2. `solver.add_points()` → adds to map.
3. `solver.graph.optimize()` → GTSAM pose graph optimization.
4. `extract_stream_data()` → collects points/colors/camera poses, recenters around scene mean, resolves pending beacons.
5. `_detect_after_submap_update()` → CLIP+SAM3 on latest submap if queries active.
6. Pushes result dict to `result_queue`; keeps last 1 frame as overlap window.

**Object detection** (cache-aware):
- CLIP embeddings fetched via `submap.get_all_semantic_vectors()`.
- SAM3 runs only on frames with CLIP similarity above threshold.
- Cache key: `(submap_id, frame_idx, query)` guarded by `_detection_lock`.
- 3D bboxes via `ObjectDetector.compute_3d_bbox()` using masked point cloud.
- `_dedup_and_store()` deduplicates overlapping 3D boxes across submaps.

**Soft reset** (`soft_reset()`): clears all SLAM state and temp files without reloading VGGT or CLIP models; re-attaches CLIP model to solver after `solver.reset()`.
