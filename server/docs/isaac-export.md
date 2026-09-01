# Isaac Sim / Isaac Lab USD Export

> Turn a finished VGGT-SLAM scan into a **metric, gravity-aligned, Z-up USD scene** that loads
> directly in NVIDIA **Isaac Sim / Isaac Lab** — a robot can stand on the scanned floor and
> collide with the geometry. This is the sibling of the LeRobot/GR00T trajectory export; it is
> the [dataset-export.md](dataset-export.md) §5 "metric + gravity alignment" item, finally
> built, plus a USD writer.
>
> Cross-refs: the geometry it reuses [dataset-export.md](dataset-export.md) ·
> up-to-scale/gravity caveat [gotchas.md](gotchas.md) · where things live
> [code-structure.md](code-structure.md) · data shapes [conventions.md](conventions.md).

**Status:** implemented on branch `isaac-usd-export` (off `robot`). Stages 1–2 (scene + camera
path) done; **Stage 3 (Isaac Lab task scaffold) is not built.** Unit tests green; real-SLAM E2E
and a human Isaac Sim load are the open verification rows (see §6).

**Now also wired into the product-workflow Export stage** as a real broker route
(`GET /api/scenes/<id>/export?format=isaac_usd`) — GPU-free, from the persisted export record (no
live Solver), gated on a metric scale. See §3 "Via the product-workflow Export stage" and the
deploy note there (the Modal image must add `usd-core` + `open3d`).

---

## 0. Isaac GR00T vs Isaac Sim/Lab — what this is

NVIDIA ships two different "Isaac" things and the export targets are different:

| Target | What it is | Our exporter |
|--------|-----------|--------------|
| **Isaac GR00T** | a robot foundation **VLA model** you fine-tune on trajectories | `server/export/` → GR00T-LeRobot v2 ([dataset-export.md](dataset-export.md)) |
| **Isaac Sim / Isaac Lab** | a USD-based **simulator / RL framework** you load a *scene* into | **this doc** → `server/export/isaac/` → `scene.usd` |

If you want to *train a policy on the camera trajectory*, use the GR00T export. If you want to
*drop the scanned room into a simulator as an environment*, use this one.

---

## 1. The two hard problems (why this is net-new)

The geometry was already there — the export package hands us a world-frame dense cloud
(`server/export/clouds.py`) and ordered keyframe poses + intrinsics
(`server/export/trajectory.py`). Two things were missing, both called out as un-built in
[dataset-export.md](dataset-export.md) §10 and [gotchas.md](gotchas.md):

1. **The SLAM world frame is not gravity-aligned.** Isaac needs +Z up with the floor as the
   ground. We estimate the gravity (up) vector from the **dominant floor plane** (RANSAC + SVD
   refit), oriented up toward the cameras (which sit above the floor).
2. **The SLAM world frame is up-to-scale, not metric.** There is *no* metric anchor in the
   system (`vggt_slam/scale_solver.py` only does relative submap ratios). We take a metric
   scale from the **user** — a direct factor, or a reference object's known real size.

Plus a third, mechanical one: **USD's unauthored defaults are wrong for robotics**
(`metersPerUnit` falls back to `0.01` cm, `upAxis` to `Y`). The writer authors
`metersPerUnit=1.0` + `upAxis="Z"` explicitly.

Everything else (`AlignmentResult`) is reported with provenance + confidence into
`manifest.json`, because the floor-plane estimate *can* be wrong (a wall/table picked instead
of the floor, or a flipped sign) and downstream must know how much to trust it.

---

## 2. What it writes

```
export/<scan_id>/isaac/
  scene.usd        # /World/Environment : UsdGeom.Mesh + vertex colors + UsdPhysics collision
                   #   (or /World/Cloud : UsdGeom.Points, when mesh reconstruction is off/failed)
  trajectory.usd   # /World/Camera : an animated UsdGeomCamera following the scan keyframes
  manifest.json    # IsaacManifest: T_align, scale provenance, metric / gravity_aligned flags
```

- **Stage:** `upAxis="Z"`, `metersPerUnit=1.0`, default prim `/World`.
- **Mesh:** Poisson reconstruction (open3d), vertex-colored, with a **static full-triangle-mesh
  collider** (`UsdPhysics.CollisionAPI` + `MeshCollisionAPI` `approximation="none"`; no
  `RigidBodyAPI` → the environment is fixed, not dynamic). PhysX-specific tuning (contact
  offsets etc.) is left to the Isaac Lab side.
- **Camera:** the CV camera convention (x-right, y-down, **z-forward**) is flipped to the USD
  camera convention (x-right, y-up, **z-back**) via `diag(1,-1,-1,1)` so the animated camera
  actually points where the scan camera pointed. This is the easiest thing to get silently
  wrong — the unit test reads the camera world position/forward back and checks it.

---

## 3. Running it

### Locally (offline `main.py`, needs a GPU for SLAM)

```bash
python main.py --image_folder examples/kitchen/images \
    --export_isaac /tmp/isaac --isaac_scale 0.25
```

| Flag | Meaning |
|------|---------|
| `--export_isaac <dir>` | write `scene.usd`/`trajectory.usd`/`manifest.json` under `<dir>/<scan>/isaac/` |
| `--isaac_scale <f>` | SLAM-units → meters factor. **Omit and the scene stays up-to-scale** (`metric=false`). |
| `--isaac_ref_object <q>` + `--isaac_ref_meters <m>` | back the scale out of a detected object's known real size (e.g. `door` `2.0`); needs `--run_os` so the scene graph exists |
| `--isaac_no_mesh` | skip reconstruction, write a points-only scene (no collision) |

### On Modal (real SLAM on a clip; mesh runs for real — Linux has open3d)

```bash
modal run modal_data_export_test.py::download_models                       # one-time
modal run modal_data_export_test.py --demo-video office_loop.mp4 \
    --export-isaac --isaac-scale 0.25
```

Downloads the tree to `./export_results/isaac/<scan>/`.

### Validate any export

```bash
python scripts/inspect_isaac.py --isaac ./export_results/isaac/<scan>
```

Reopens the USD with `pxr` and runs PASS/FAIL invariant checks (Z-up, meters, mesh/collider or
points, camera sample count == keyframes, transform shape, metric flags). **Paste its output to
verify** without shipping the binaries around.

### Via the product-workflow Export stage (broker route, GPU-free)

`GET /api/scenes/<scan_id>/export?format=isaac_usd` — the same owner-authed broker route that
serves the OpenReality / GR00T-LeRobot exports (`server/app.py::export_scene_route` →
`_export_isaac_usd`). It reads the **persisted export record** (world-frame cloud + keyframe
poses/intrinsics — the same geometry the GR00T/LeRobot export consumes), not a live Solver, and
authors the USD tree via `export_isaac_from_record`. No GPU.

| Query param | Meaning |
|-------------|---------|
| `format=isaac_usd` | select the Isaac USD export |
| `scale=<f>` | optional — SLAM-units→metres factor, applied to the ORIGINAL geometry (`scale_source="user:factor"`). Mirrors the CLI `--isaac_scale`. |
| `source=<derived_key>` | optional geometry selector (shared with the other formats). Pass a Metric-anchor key (`derived/anchor/<stamp>/cloud.ply`) to export the **already-metric** calibrated geometry (the route applies scale `1.0` — never double-scales). |

**Metric-scale gate (never emits an up-to-scale USD).** The route requires a metric scale from
*either*:
1. an explicit `?scale=<f>`, **or**
2. a persisted **Metric anchor** (`derived_latest` of kind `"anchor"`, from `POST
   .../anchor`) — the route consumes its calibrated (already-metric) geometry at scale `1.0` and
   records the anchor's `scale_factor` as provenance in `manifest.json`.

With neither → **409 `{error:"metric_scale_required"}`** (`"Run Metric anchor first — Isaac needs
a metric scale."`). Gravity alignment needs no input — the writer estimates it from the floor
plane.

**Success:** `200 application/zip`, streamed as `scan-<scan_id>-isaac_usd.zip`, containing
`<scan_id>/isaac/{scene.usd, trajectory.usd, manifest.json}` (same tree the CLI writes).

**Error codes:** `409 metric_scale_required` (no scale/anchor) · `501 isaac_unavailable`
(`usd-core`/`open3d` not installed in this deployment — see the deploy note below) · `400
invalid_request` (bad `?scale=`) / `invalid_format` · `404 not_found` / `no_trajectory` (pre-Stage-4
scan) / `no_points` / `unknown_derived_key` (bogus `?source=`) · `500 export_failed`.

> **Deploy note — the broker image needs `usd-core` (and already has `open3d`).** The route runs
> on the always-on CPU **broker** (`modal_streaming.py`, the shared `image`). Two heavy deps author
> the USD: `usd-core` (`pxr`, the USD stages) and `open3d` (the Poisson collider mesh). Current
> state of that image:
>   - `open3d` — **already installed** (`modal_streaming.py` `.pip_install(... "open3d" ...)`).
>   - `usd-core` — **added on this branch** to the same `.pip_install(...)` block (search
>     `[isaac]`). It is in `requirements.txt` too, but the Modal image pins its deps explicitly, so
>     it had to be added there. Linux wheels exist for both; `usd-core` is authoring-only (no Isaac
>     Sim runtime).
>
> Both import **lazily** inside the Isaac path and are pre-checked, so a bare/older image returns a
> clean **501 `isaac_unavailable`** (listing which dep is missing) — it never crashes the server.
> The route only actually produces USD once the image is rebuilt with `usd-core`; until then it
> honestly 501s.

---

## 4. Setting the metric scale

There is no automatic metric scale. Pick one:

- **Direct factor** (`--isaac_scale`): you measured something in the scene and know
  SLAM-units → meters. Simplest, deterministic.
- **Reference object** (`--isaac_ref_object door --isaac_ref_meters 2.0` + `--run_os`): the
  exporter finds that object in the scene graph, measures its height **along the recovered up
  axis** (the AABB is axis-aligned in the SLAM frame, so it is rotated into the gravity frame
  first), and divides. Approximate — only as good as the object's true size and detection box.
- **Neither:** the export is honestly flagged `metric=false` with a `notes` line; the geometry
  is Z-up and gravity-aligned but still up-to-scale. Useful for visualization, not for
  affordance-accurate physics.

A future automatic option (a monocular metric-depth model, e.g. UniDepth/Metric3D) is noted in
§7 but not built.

---

## 5. Code map (`server/export/isaac/`)

GPU-free / duck-typed like `server/export/`. `align.py` is **pure numpy/scipy** (fully
unit-testable anywhere); the heavy deps are lazy: `open3d` (Poisson) in `mesh.py`, `pxr`
(`usd-core`) in `usd_writer.py`.

| File | Responsibility | Key entry points |
|------|----------------|------------------|
| `align.py` | gravity + scale + transform (the geometric core) | `fit_plane_ransac`, `estimate_gravity`, `resolve_scale`, `compute_alignment` → `AlignmentResult`, `apply_transform`, `transform_camera_pose` |
| `mesh.py` | cloud → collidable surface (lazy open3d) | `reconstruct_mesh` (Poisson / ball-pivoting) |
| `usd_writer.py` | author the stages (lazy pxr) | `write_scene_usd`, `write_trajectory_usd` (+ the CV→USD flip), `_gf_matrix` |
| `schema.py` | manifest pydantic | `IsaacManifest`, `AlignmentInfo`, `GeometryInfo` |
| `writer.py` | orchestrator | `export_isaac(solver, out_dir, scale=…, ref_object=…)` |

**Other touched files:** `main.py` (`--export_isaac` + scale flags), `requirements.txt`
(`usd-core`), `modal_data_export_test.py` (`--export-isaac`/`--isaac-scale`),
`scripts/inspect_isaac.py`, `scripts/make_isaac_fixture.py`.

**Tests** (`tests/test_isaac_*.py`): `align` (numpy-only — tilted-floor gravity recovery, scale
resolution, floor→z=0, camera-pose consistency), `usd` (reopen with pxr; Z-up/meters, mesh +
collider, the camera flip), `writer` (full fake-solver E2E + manifest). The pxr/open3d tests
`importorskip` so the suite stays green without those installed.

---

## 6. Verification status

| Check | Status |
|-------|--------|
| `align.py` geometry (gravity/scale/transform) unit tests | ✅ numpy-only, run anywhere |
| `usd_writer` conventions + camera flip + collider (reopen with pxr) | ✅ with `usd-core` |
| `export_isaac` full tree + manifest (fake solver) | ✅ |
| `make_isaac_fixture.py` → `inspect_isaac.py` 16 invariants | ✅ on synthetic |
| **Broker route** metric-scale gate (409) + format routing + lazy-dep 501 | ✅ `tests/test_isaac_export_route.py` (dep-free, always runs) |
| **`export_isaac_from_record`** real USD from a persisted record (points-only + anchor-prescaled provenance) | ✅ with `usd-core` (`importorskip`; verified locally) |
| **Route → real zip end-to-end** (`?format=isaac_usd`, mesh + collider, metric manifest) | ✅ with `usd-core`+`open3d` (`importorskip`; verified locally, incl. anchor `scale=1.0` no-double-scale) |
| **Real SLAM geometry** on a real clip (Modal) end-to-end through the route | ❌ NOT YET (needs the broker image rebuilt with `usd-core`, run on a real persisted scan) |
| **Load `scene.usd` in Isaac Sim** (robot collides with the floor) | ❌ NOT YET (manual, needs an Isaac Sim install) |

The wiring, gates, and USD authoring are covered by tests (the `importorskip` rows run wherever
`usd-core`/`open3d` are installed — they were exercised locally against real `usd-core` 25.x +
`open3d` 0.19). The two ❌ rows still need a real Modal run (broker image rebuilt with `usd-core`)
+ a human Isaac Sim load — unchanged from before, now reachable through the product-workflow
Export button instead of only the CLI.

---

## 7. Not built (future)

- **Stage 3 — Isaac Lab task scaffold.** A runnable `InteractiveSceneCfg` that loads `scene.usd`
  as a static asset + spawns a robot + ground plane. Needs an Isaac Lab runtime to validate.
- **Automatic metric scale** via a monocular metric-depth model (UniDepth/Metric3D/Depth Pro).
- **IMU gravity.** The phone already captures `DeviceOrientation` (a gravity reference) in
  `server/webserver/src/sender.ts` but never sends it to the server. Plumbing it through would
  give a true gravity vector instead of the floor-plane estimate.
- **Convex decomposition** colliders (faster physics than the full tri-mesh) and PhysX material
  tuning — better done on the Isaac Lab side.
