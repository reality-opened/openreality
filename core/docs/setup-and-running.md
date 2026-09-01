# Setup, Standalone SLAM & Evaluation

> How to install the environment, run SLAM standalone, collect custom data, and run the benchmark evals.
> For cloud (Modal) runs, see [modal-deployment.md](modal-deployment.md). For the streaming server, see [streaming-server.md](streaming-server.md).

## Setup & Installation

```bash
conda create -n vggt-slam python=3.11
conda activate vggt-slam
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs pip requirements, clones four third-party repos into `third_party/` (Salad, VGGT_SPARK, Perception Encoder, SAM3), and installs the main package in editable mode. `dev_setup.sh` is the more complete variant used during active development.

**First-run model downloads** (happen automatically, require network/disk space):
- VGGT-1B weights from HuggingFace (~4 GB) → cached in `~/.cache/torch/hub/checkpoints/model.pt`
- DINO-Salad checkpoint (~350 MB) → cached in `~/.cache/torch/hub/checkpoints/dino_salad.ckpt`
- DINOv2 backbone (~350 MB) → pulled via `torch.hub`
- PE-Core-L14-336 CLIP model (if `--run_os` or open-set detection is enabled) → downloaded from HuggingFace

## Running Standalone SLAM

```bash
# Basic run with visualization (Viser UI on http://localhost:8080)
python main.py --image_folder /path/to/images --max_loops 1 --vis_map

# With open-set object detection (requires Perception Encoder + SAM3)
python main.py --image_folder /path/to/images --max_loops 1 --vis_map --run_os

# Quick test with bundled data
unzip office_loop.zip
python main.py --image_folder office_loop --max_loops 1 --vis_map

# Log poses and dense point clouds to disk
python main.py --image_folder office_loop --max_loops 1 --log_results --log_path poses.txt

# Offline visualization of saved results
python visualize_results.py
```

Key arguments: `--submap_size` (frames per submap, default 16), `--min_disparity` (keyframe threshold, default 50), `--conf_threshold` (filter low-confidence points %, default 25), `--lc_thres` (loop closure similarity threshold, default 0.95), `--vis_voxel_size` (downsample for visualization).

**Limitations:** `--max_loops` only supports 0 or 1; `--overlapping_window_size` only supports 1.

> **`lc_thres` has two different defaults.** The 0.95 above is the **CLI** default that
> `main.py` passes down. A directly constructed `Solver()` (what `server` and the eval
> harnesses use) defaults to **0.80** — more permissive, more loop closures. Pass
> `lc_thres` explicitly if the value matters to your run.

### Custom Data Collection

```bash
mkdir /path/to/img_folder
ffmpeg -i /path/to/video.MOV -vf "fps=10" /path/to/img_folder/frame_%04d.jpg
```

Use horizontal videos to avoid cropping. Images are sorted by the numeric value in their filename.

## Evaluation

```bash
# TUM RGB-D benchmark
./evals/eval_tum.sh 32       # argument is submap_size
python evals/process_logs_tum.py --submap_size 32

# 7-Scenes benchmark
./evals/eval_7scenes.sh 32
```

Set `abs_dir` in the eval shell scripts to your MASt3R-SLAM dataset download location.

There is no SLAM-correctness test suite. Validate behavioural changes by running against `office_loop/` sample data or demo videos in `server/demo_videos/`. The pytest suite under `tests/` covers latency-related contracts, not SLAM accuracy (see [testing.md](testing.md)).
