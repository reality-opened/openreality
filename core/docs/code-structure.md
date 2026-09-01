# Code Structure — File Map

> "Where does X live?" — a file-by-file map of every top-level file and package, plus the root reference docs.
> For how the pieces interact at runtime, see [overview.md](overview.md).

## Top-level files

| File | Purpose |
|------|---------|
| `main.py` | Standalone SLAM demo entry point |
| `modal_app.py` | Modal batch job — uploads images to A100, runs SLAM, downloads poses/point clouds |
| `modal_streaming.py` | Modal persistent ASGI server — builds frontend at image-build time, serves WebSocket + HTTP |
| `visualize_results.py` | Offline point-cloud viewer (`.npz` → Viser) |
| `setup.py` | Package metadata (`vggt_slam` v2.0.0) |
| `setup.sh` / `dev_setup.sh` | Install deps, clone third-party repos |
| `launch_modal.sh` / `tmux_setup.sh` | Convenience launchers |
| `requirements.txt` | Python deps (torch 2.3.1, gtsam-develop, viser 0.2.23, Flask, python-socketio, openai, anthropic, PyJWT[crypto], etc.) |

## `vggt_slam/` — Core SLAM module

| File | Key class / function | Role |
|------|---------------------|------|
| `solver.py` | `Solver` | Central coordinator; `run_predictions()` runs VGGT inference; `reset()` reinitializes without reloading models; `DEBUG` flag gates matplotlib/open3d plots |
| `map.py` | `GraphMap` | Dict of `Submap` keyed by submap ID; `retrieve_best_semantic_frame()` for CLIP cosine search |
| `submap.py` | `Submap` | Per-submap bundle (images, poses, world points, colors, confidences, CLIP vectors); `get_points_in_mask()` projects 2D masks to 3D |
| `graph.py` | `PoseGraph` | GTSAM SL(4) factor graph; `add_homography()`, `add_between_factor()`, `optimize()` (Levenberg-Marquardt) |
| `frame_overlap.py` | `FrameTracker` | Lucas-Kanade optical flow keyframe selector |
| `loop_closure.py` | `ImageRetrieval`, `LoopMatchQueue` | SALAD descriptor model; `find_loop_closures()` returns top-K matches |
| `object_detector.py` | `ObjectDetector` | PE-Core CLIP + SAM3 wrapper with `torch.autocast` mixed precision; `encode_text()`, `segment()`, `compute_3d_bbox()`, `deduplicate_detections()` |
| `scale_solver.py` | `estimate_scale_pairwise()` | Median depth-ratio scale estimation |
| `splat_export.py` | `export_splat()`, `point_cloud_to_splat_ply()` | 3DGS `splat.ply` export: deterministic CPU baseline (one gaussian per world point) + optional guarded gsplat GPU refinement |
| `metric_anchor.py` | `MetricScaleAnchor`, `combine_scale()` | EXP-12 per-submap metric scale anchor (default OFF): gated ratio-form scale from a monocular metric-depth model, to remove cross-submap scale drift |
| `junction_weight.py` | `junction_covariance_multiplier()` | Junction-confidence weighting (default OFF): scales each cross-submap factor's covariance by measured junction quality so weak junctions get looser factors |
| `hygiene.py` | RobustVGGT-F scoring | Offline wrong-place gate for scan windows — training-free per-frame outlier score read off stock VGGT patch features; not in the live loop |
| `slam_utils.py` | `Accumulator` + helpers | `compute_image_embeddings()`, `cosine_similarity()`, `sort_images_by_number()`, `Accumulator` timer |
| `viewer.py` | `Viewer` | Viser server — point clouds + frustums; `visualize_frames()`, `run_walkthrough()` |

## `server/` — Streaming server + agent system

| File / Dir | Key class | Role |
|------------|-----------|------|
| `app.py` | Flask + SocketIO ASGI | WebSocket + HTTP server; Clerk JWT verification; session store via `modal.Dict`; demo-video playback; `VideoFeeder`; spatial agent session management |
| `streaming_slam.py` | `StreamingSLAM` | Frame-by-frame `Solver` wrapper; `add_frame()`, `process_submap()`, `extract_stream_data()`, `soft_reset()`, `set_active_queries()` |
| `spatial_agent.py` | `SpatialAgent`, `Mission` | Autonomous exploration orchestrator using OpenRouter (Gemini 3 Flash Preview by default); multi-mission tracking, stall detection, adaptive query generation |
| `llm/openrouter_client.py` | `OpenRouterClient` | Resilient LLM client with fallback chains, JSON parsing, timeout/retry |
| `agent/runtime.py` | `AgentRuntime` | Session-scoped tool executor; validated args; SocketIO event emission |
| `agent/schemas.py` | Pydantic models | Tool arg/return schemas |
| `agent/scene_index.py` | `SceneIndex` | Thread-safe deduplicating detection cache |
| `agent/tool_registry.py` | `ToolRegistry`, `ToolDefinition` | Thread-pool executor with Pydantic validation and timeouts |
| `agent/tools/vggt_tools.py` | `VGGTTools` | SLAM-backed tools (`get_scene_snapshot`, `search_objects`, `locate_object_3d`, `infer_spatial_relations`) |
| `agent/tools/ui_tools.py` | `UITools` | UI commands (`focus_detection_ui`, `show_waypoint_ui`, `show_path_ui`, `show_toast_ui`) |
| `scene_report/` | `SceneFeatureExtractor`, `SceneReportBuilder` | Live (`scene_report_update`) + end-of-scan (`scene_report_ready`) 3D facts + grounded report; world-frame re-anchoring (see [scene-report.md](scene-report.md)) |
| `demo_videos/` | static | Pre-recorded demo clips (Git LFS); uploaded to Modal Volume via `modal run modal_streaming.py` |

## `server/webserver/src/` — TypeScript SPA

| Dir / File | Role |
|------------|------|
| `main.ts`, `landing.ts`, `plan.ts`, `summary.ts`, `sender.ts`, `detection-debug.ts` | Vite entry points (one per HTML page) |
| `services/SLAMConnection.ts` | Socket.IO client — emits `frame`, receives `slam_update`/`global_map` |
| `services/SceneManager.ts` | Three.js scene state — `updateMap()`, `updateDetections()`, `focusOnDetection()` |
| `services/DedalusAPI.ts` | HTTP client for agent REST endpoints |
| `components/UIManager.ts` | Panels, toasts, detection/mission UI rendering |
| `components/AgentPanel.ts` | Agent control + mission status UI |
| `components/HeroScene.ts`, `MeshBackground.ts` | Landing visuals |
| `components/onboarding/` | `HelpButton.ts`, `TourController.ts`, `steps.ts` — driver.js tour |
| `utils/auth.ts`, `utils/navigation.ts` | Clerk token plumbing (mirrors `landing/app/utils/`) |
| `styles/` | `agent.css`, `dashboard.css`, `glass-ui.css`, `landing.css`, `plan.css`, `driver-theme.css` |

## `landing/` — Next.js 15 + Clerk frontend (Vercel)

| Dir / File | Role |
|------------|------|
| `app/page.tsx` | Renders `LandingExperience` (hero, sign-in, demo carousel) |
| `app/dashboard/page.tsx` | Authenticated dashboard polling Modal `/health`, launching the SLAM session |
| `app/components/` | `LandingExperience.tsx`, `HeroScene.tsx`, `DemoCarousel.tsx` |
| `app/onboarding/` | driver.js tour (`OnboardingTour.tsx`, `useOnboarding.ts`, `steps.ts`, `HelpButton.tsx`) |
| `app/api/onboarding/complete/` | API route persisting onboarding completion |
| `app/utils/navigation.ts` | Clerk token retrieval, Modal session creation, URL token hashing |
| `middleware.ts` | Clerk auth middleware protecting `/dashboard(.*)` |
| `e2e/` | Playwright end-to-end tests |
| `test/` | Vitest unit tests |

## `evals/` — Benchmarks

| File | Role |
|------|------|
| `eval_tum.sh` | Batch TUM RGB-D — set `abs_dir` to dataset location |
| `eval_7scenes.sh` | Batch 7-Scenes evaluation |
| `process_logs_tum.py` | Convert SLAM poses to TUM format; compute ATE/RTE |

## `third_party/` — External models (installed by `setup.sh`)

| Dir | Model | Used for |
|-----|-------|---------|
| `vggt/` | VGGT_SPARK | Dense depth + SL(4) pose prediction |
| `salad/` | DINO-Salad | Loop closure image descriptors |
| `perception_models/` | PE-Core CLIP | Semantic embeddings for open-set detection |
| `sam3/` | SAM3 | Instance segmentation for 3D bounding boxes |

## Reference docs (root)

| File | What's in it |
|------|--------------|
| `latency_blog.md` | Narrative writeup of the latency reduction work (the "phase 4-9" effort) — useful background when touching the streaming path or detection cache |
| `latency_handoff.md` | Engineering handoff covering the latency optimization branch state |
| `scene_report_plan.md` | Phased plan for the scene report + grounded Q&A work (Phases 1–3 implemented: 3D feature extractor + end-of-scan report + progressive in-scan report; 4–5 deferred) |
| `fixes.md` / `h1_dashboard_allocation_fix.md` / `per_user_gpu_brief.md` | Open security/robustness items + the per-user GPU broker/worker design (see [modal-deployment.md](modal-deployment.md)) |
| `README.md` | User-facing intro |
| `architecture_v2.svg` | Diagram referenced from the README |

Other root markdown files (`yc_chat_history.md`, `reducing_latency_chat_history.md`) are conversation logs — not authoritative; ignore them when looking for current architecture.
