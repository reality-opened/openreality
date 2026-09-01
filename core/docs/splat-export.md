# 3D Gaussian Splatting (3DGS) export

> Turn the SLAM result (a world-frame colored point cloud) into a standard 3DGS
> `splat.ply` that opens in any 3DGS viewer. Module: [`vggt_slam/splat_export.py`](../vggt_slam/splat_export.py).
> CLI flag on `main.py`: `--export_splat <path>` (+ optional `--export_splat_refine`).

This is the first geometry producer in `core` beyond the raw point cloud
(`GraphMap.write_points_to_file`, Open3D `.pcd`/`.ply`). It exists because a
real-estate pilot's contract promises a **3DGS asset produced by our pipeline**.

## Two levels

| Level | GPU? | Deps | Determinism | What it does |
|-------|------|------|-------------|--------------|
| **Baseline** (default) | no | numpy (+ optional scipy for spacing) | fully deterministic | One gaussian per world point. The supported, contract-grade path. |
| **Refinement** (opt-in) | yes | `gsplat` (optional extra), `torch` | not deterministic | Photometric optimization of the baseline init against posed keyframes. |

The baseline is the deliverable. Refinement is best-effort: if `gsplat` / CUDA /
posed keyframes are unavailable it is caught and the baseline file is kept.

## Usage

```bash
# Baseline (CPU, no training, no extra deps):
python main.py --image_folder examples/kitchen/images/ --export_splat out/splat.ply

# With optional GPU refinement (needs `pip install -e ".[splat]"` and a CUDA box):
python main.py --image_folder examples/kitchen/images/ \
    --export_splat out/splat.ply --export_splat_refine
```

`<path>` is the output `.ply` file; its parent directory is created if missing.
The flag is entirely off the default path — runs only when passed, after SLAM
finishes (mirrors `--export_dataset` / `--export_isaac`). Existing flows are
byte-for-byte unchanged when it is absent.

Programmatic entry point:

```python
from vggt_slam.splat_export import export_splat
export_splat(solver, "out/splat.ply", refine=False)   # uses solver.map + solver.graph
```

## PLY layout

Binary little-endian, every property `float32`, the de-facto 3DGS vertex schema
(17 fields, in this order):

```
x, y, z,  nx, ny, nz,
f_dc_0, f_dc_1, f_dc_2,          # SH degree-0 DC term, from RGB
opacity,                          # inverse-sigmoid (logit) of alpha
scale_0, scale_1, scale_2,        # log of the per-axis gaussian scale
rot_0, rot_1, rot_2, rot_3        # quaternion (w, x, y, z)
```

### Point -> gaussian field math (baseline)

- **Position** `x,y,z`: the world point, straight from
  `Submap.get_points_in_world_frame(graph)` (confidence-filtered), aggregated over
  all **regular** submaps (see below) roughly like `GraphMap.write_points_to_file`.

  `gather_world_cloud` (unlike `write_points_to_file`) excludes two sources of
  duplicated geometry, mirroring the fix already shipped server-side
  (`server.export.clouds.gather_full_cloud`; window-quality study,
  `experiments/research/2026-07-03-omega-full-quality.md` §2/§6, platform repo):
  loop-closure re-observation submaps (`submap.get_lc_status()`), and each
  non-first submap's leading `overlap_frames` frame(s), which duplicate the
  carried-over overlap frame(s) shared with the previous submap
  (`--overlapping_window_size`, default/only-supported value `1`). `export_splat`
  takes an `overlap_frames` kwarg (default `1`) forwarded straight through; `main.py`
  passes `args.overlapping_window_size`.
- **Color** `f_dc_*`: RGB in `[0,1]` stored as the SH DC coefficient
  `f_dc = (c - 0.5) / C0`, `C0 = 0.28209479177387814`. Viewers invert it as
  `rgb = f_dc * C0 + 0.5`. Inputs in `[0,255]` are auto-normalized.
- **Opacity**: a constant alpha (default `0.9`) stored as its logit
  `log(a / (1 - a))` (3DGS applies a sigmoid at render time).
- **Scale** `scale_*`: an isotropic per-point scale derived from local spacing —
  mean distance to the `k` nearest neighbours (`scipy.cKDTree`), falling back to a
  bbox-diagonal / cube-root(N) heuristic if scipy is missing or a point is
  degenerate — stored as `log(scale)`.
- **Rotation** `rot_*`: identity quaternion `(1, 0, 0, 0)` (isotropic gaussians).
- **Normals** `nx,ny,nz`: zeros (3DGS ignores them; present for schema
  compatibility).

### p99 scale-tail clamp (default ON)

Before storing, per-axis scales are clamped down to a high percentile of the
(flattened, finite) scale distribution — `clamp_scales_percentile(scales, p=99)`,
applied inside `write_splat_ply` so **both** the baseline and the refined write go
through it. EXP-7 (`experiments/research/2026-07-05-exp7-b3seg-probe.md`, platform
repo) showed the refined/composed splat's off-trajectory render "spike catastrophe"
is *entirely* the top-1% per-gaussian scale tail: clamping every axis to the p99
value turns those renders readable with no re-export. It is a **near-no-op on the
baseline** (its isotropic k-NN scales already sit at a capped max), so default
behavior of the shipped baseline path is effectively unchanged.

- Off: `--no_splat_scale_clamp` on `main.py`, or `clamp_scales=False` in the API.
- Tune: `--splat_scale_clamp_pct <p>` / `scale_clamp_percentile=<p>` (default `99`).
- When any gaussian is clamped, `export_splat` logs the clamp value + affected count.

## Optional gsplat refinement

`refine_with_gsplat(...)` seeds gaussians from the baseline init and runs an Adam
photometric optimization (`gsplat.rasterization` + MSE vs. the keyframe image)
over the posed keyframes, then overwrites the `.ply` with the refined splats.

Posed keyframes are reachable from the solver/map and gathered by
`gather_posed_keyframes`:

- images: `Submap.get_all_frames()` (the preprocessed `(S,3,H,W)` keyframe tensor),
- intrinsics `K`: `Submap.proj_mats[i][:3,:3]` (proj_mats is the K-padded matrix,
  per `docs/conventions.md`),
- world-to-camera viewmat: `inv(Submap.get_all_poses_world(graph)[i])`
  (`get_all_poses_world` returns cam-to-world, as consumed by `Viewer`).

Everything torch/gsplat-related is **lazily imported** and the whole call is
wrapped in `try/except` by `export_splat`, so:

- importing `vggt_slam.splat_export` on a CPU / no-gsplat box always works;
- a missing dep, no CUDA, or empty keyframe set raises and the baseline file is
  kept untouched.

`gsplat` is an **optional extra**, not a hard dependency: `pip install -e ".[splat]"`
(declared in `setup.py` `extras_require["splat"]`; see also the note in
`requirements.txt`).

> ⚠️ The refinement path is **not GPU-validated in CI** — the tests cover the
> baseline serialization + field math and the graceful CPU fallback only. Treat
> the baseline as the supported export.

## Tests

`tests/test_splat_export.py` (GPU-free): PLY header/field/byte-layout checks, a
serialization round-trip, the point->gaussian field math on a tiny synthetic
cloud, fake-solver integration, the refine-falls-back-to-baseline guarantee, and
`gather_world_cloud`'s LC-exclusion + overlap-frame dedup (fake per-frame
submaps mirroring `server/tests/test_gather_dedup.py`'s patterns). The gsplat
test is `pytest.importorskip` + CUDA-gated.

```bash
python -c "import vggt_slam.splat_export"          # import smoke test
pytest tests/test_splat_export.py -v               # 24 pass, 1 skip on CPU
```
