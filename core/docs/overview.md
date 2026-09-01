# Overview & Architecture

> System overview, the SLAM pipeline, and how the core classes fit together.
> For the file-by-file map, see [code-structure.md](code-structure.md). For data shapes/conventions, see [conventions.md](conventions.md).

## Project Overview

Open Reality is a real-time dense feed-forward monocular SLAM system built on **VGGT-SLAM 2.0**. It uses the VGGT model for depth/pose prediction, optimizes on the SL(4) manifold via GTSAM, and supports loop closure, open-set 3D object detection (CLIP + SAM3), and autonomous spatial agents.

The product surface today has three layers:

1. **Core SLAM library** (`vggt_slam/`) — runnable standalone (`main.py`) or via batch Modal job (`modal_app.py`).
2. **Streaming server** (`server/`) — Flask + python-socketio ASGI app that streams frames in and broadcasts SLAM updates; hosts the spatial agent system. Deployed per user on Modal GPU workers via `modal_streaming.py`.
3. **Two frontends**:
   - `landing/` — Next.js 15 + Clerk app deployed on Vercel; handles auth, dashboard, onboarding, and warms up the Modal GPU.
   - `server/webserver/` — Vite + TypeScript SPA (Three.js) bundled into the Modal image; serves the live SLAM viewer, phone sender, plan, and summary pages.

## Architecture

**Pipeline flow** (orchestrated in `main.py` for offline, `StreamingSLAM.process_loop()` for streaming):
1. **Keyframe selection** — `FrameTracker` (`frame_overlap.py`) uses Lucas-Kanade optical flow to skip frames without enough motion.
2. **VGGT inference** — `Solver.run_predictions()` calls the VGGT model for dense depth, confidence maps, camera poses, and intrinsics for each submap batch.
3. **Loop closure detection** — `ImageRetrieval` (`loop_closure.py`) uses DINO-Salad descriptors; loop closure rejected if VGGT's `image_match_ratio < 0.85`.
4. **Scale estimation** — `estimate_scale_pairwise()` (`scale_solver.py`) computes scale via median depth ratios of overlapping points.
5. **Submap construction** — `Submap` (`submap.py`) bundles frames, poses, 3D points, colors, confidences; stored in `GraphMap` (`map.py`).
6. **Pose graph optimization** — `PoseGraph` (`graph.py`) GTSAM SL(4) manifold with intra-submap, inter-submap, and loop closure constraints.
7. **Visualization / streaming** — local: `Viewer` (`viewer.py`) → Viser on :8080. Streaming: `extract_stream_data()` → SocketIO `slam_update` → Three.js scene in browser.

**Key class relationships:**
- `Solver` is the central coordinator — owns `GraphMap`, `PoseGraph`, `ImageRetrieval`, `FrameTracker`, and optionally `Viewer`.
- `Solver.reset()` resets SLAM state without reloading models (used by streaming `soft_reset()`).
- `GraphMap` holds a dict of `Submap` keyed by submap ID; tracks `non_lc_submap_ids` separately.
- `PoseGraph` wraps GTSAM; node IDs are `submap_id + frame_index_within_submap` (submap_id is pre-offset for global uniqueness).
- `Submap` stores per-frame data (images, points, poses as 4×4 matrices); LC submaps have `is_lc_submap=True`.
- `ObjectDetector` wraps PE-Core CLIP + SAM3 for open-set 3D bounding box detection (uses `torch.autocast` for mixed precision).

**Third-party dependencies** (in `third_party/`, installed by `setup.sh` / inside the Modal image):
- `vggt/` — VGGT_SPARK: monocular dense depth/pose model.
- `salad/` — DINO-Salad: image descriptor model for loop closure.
- `perception_models/` — Facebook Perception Encoder CLIP.
- `sam3/` — SAM3 segmentation model (decord is stubbed inside Modal — no Python 3.11 wheel; we don't use video paths).
