# Conventions & Code Style

> Data shapes/dtypes, naming conventions, and code style. Read this before adding or changing data that flows through the SLAM pipeline.
> Cross-cutting pitfalls live in [gotchas.md](gotchas.md).

## Data conventions

| Data | Shape / dtype |
|------|--------------|
| Model input images | `(B, 3, H, W)` torch `float` `[0,1]`, cast to `bfloat16` |
| `world_points` | `(S, H, W, 3)` numpy `float32`, world frame |
| `colors` | `(S, H, W, 3)` numpy `uint8` `[0,255]` |
| `depth_conf` | `(S, H, W)` numpy `float32` `[0,1]` |
| Poses (cam-to-world) | `(S, 4, 4)` numpy `float32`, SL(4) homography |
| `extrinsic` (world-to-cam) | `(S, 3, 4)` numpy `float32` |
| `proj_mats` (K padded) | `(S, 4, 4)` numpy `float32` |

- `conf_threshold` is a **percentile** (not a fraction): filters the bottom N% lowest-confidence points.
- Retrieval vectors use **L2 distance** (SALAD); semantic vectors use **cosine similarity** (CLIP).
- GTSAM node IDs: `node_id = submap_id + frame_index_within_submap`; use `X(node_id)` symbols.
- `pred` dict from `Solver.run_predictions()` keys: `images`, `extrinsic`, `intrinsic`, `depth`, `depth_conf`, `detected_loops`; LC data uses `_lc`-suffixed keys.
- `slam_utils.py` exports `Accumulator` timer (`with vggt_timer: ...`) used throughout the streaming path for `[latency]` log lines.
- Pose output format (TUM): `timestamp tx ty tz qx qy qz qw`, written by `GraphMap.write_poses_to_file()`.

## Code style

- `snake_case` throughout Python; no abstract base classes; duck-typed component interfaces.
- Type hints are minimal; short single-line docstrings on key public methods only.
- `DEBUG = False` module-level flag in `solver.py` gates matplotlib/open3d debug plots.
- Frontend: ES modules, TypeScript strict, Three.js for 3D, driver.js for tours. Avoid adding new rendering libs.
