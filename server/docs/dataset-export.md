# Dataset Export — Robot-Training Data (LeRobot / GR00T) + Grounding Sidecar

> **Status:** Stages 1–4 **implemented** (Stage 5 is stretch/future). §0–§10 below remain the
> design spec / rationale; **§11 is the as-built map** — read it for what actually shipped,
> the deviations from spec, and how to run the GPU test harness. The spec is still written to
> be executed cold: read top to bottom, follow the external links in §1, respect the
> correctness traps in §7.
>
> Cross-refs: data shapes [conventions.md](conventions.md) · non-obvious traps
> [gotchas.md](gotchas.md) · the semantic layer this builds on [scene-report.md](scene-report.md)
> · where things live [code-structure.md](code-structure.md).

---

## 0. Purpose & strategic framing

Open Reality scans produce two things that, **together**, make a uniquely valuable
robot-training asset:

1. A **camera trajectory** through a real space — RGB keyframes each with a known 6-DoF
   pose, plus dense depth/point clouds and intrinsics.
2. A **grounded, evidence-cited semantic layer** — open-set 3D object detections,
   fine-grained object descriptions, a spatial scene graph, and an LLM-authored report.
   **Every claim carries an `EvidenceRef` back to a specific keyframe.**

Why this matters for robot learning: modern robot policies are **VLAs**
(vision-language-action). VLA training has *two* data regimes — (a) scarce, expensive
**action-labeled manipulation demos**, and (b) **action-free vision-language/video data**
that gives the model its semantic grounding and generalization. Our scans are *premium
regime (b)*: egocentric video + metric-relative 3D + a grounded object graph + hierarchical
language. That is differentiated against Ego4D (no 3D, no object graph), against sim (no
real-world semantic long tail), and against raw LLM-captioned video (ungrounded /
hallucinated — ours is auditable against pixels + 3D).

**We are not selling manipulation demos.** We are producing the grounded-semantic substrate
that makes a VLA's vision-language half spatially competent — plus a camera/navigation
trajectory that is *convertible* to robot-demo formats.

### What we are building

A **self-owned, versioned "OpenReality export" format** (a format *we* control — research
first, not bound to any external trainer), with **two coupled outputs**:

- **Trajectory dataset** — RGB video + camera-pose *state* + ego-motion *action* (parquet +
  mp4). Convertible to **GR00T-LeRobot v2** for fine-tuning Isaac GR00T as a camera/nav
  embodiment.
- **Grounding sidecar** — scene graph + per-object 3D grounding + **temporally-localized**
  language annotations, all carrying evidence. This is the differentiated product and is
  first-class, not an afterthought.

### Non-negotiable caveat (read [gotchas.md](gotchas.md))

The SLAM world frame is **up-to-scale, not metric, and not gravity-aligned**. Every
exported coordinate is *relative*. The format **must** declare this (`up_to_scale: true`,
`gravity_aligned: false`) so downstream never assumes metric/affordance-ready geometry.

---

## 1. Required background reading (external — the format target evolves, verify it)

Do not rely on memory for the LeRobot/GR00T APIs — they change. Read these before coding §5:

- **LeRobot v3.0 format (current upstream):** <https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3>
  and the announcement <https://huggingface.co/blog/lerobot-datasets-v3>. v3 stores *many
  episodes per* parquet/mp4 file; requires `dataset.finalize()` before push.
- **LeRobot dataset API (create/add_frame/save_episode):** <https://huggingface.co/docs/lerobot/main/en/lerobot-dataset>
- **GR00T LeRobot schema (the format GR00T actually consumes):**
  <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md>
  — **Key fact: GR00T uses LeRobot _v2_, not v3.** It is "LeRobot v2 + one extra file:
  `meta/modality.json`." If you produce v3, convert it down with NVIDIA's
  `scripts/lerobot_conversion/convert_v3_to_v2.py`.
- **GR00T new-embodiment fine-tuning:** <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md>
  — our data loads under `EmbodimentTag.NEW_EMBODIMENT` (no built-in embodiment fits a camera).
- Optional reference example of a from-scratch features dict: <https://docs.phospho.ai/learn/lerobot-dataset>

**GR00T-LeRobot v2 on-disk layout** (target of §5), per the NVIDIA doc:
```
meta/
  ├─ info.json          # LeRobot dataset info
  ├─ episodes.jsonl     # one line per episode: {episode_index, tasks, length}
  ├─ tasks.jsonl        # language strings ↔ integer index
  └─ modality.json      # GR00T-specific (see §5)
data/chunk-000/episode_000000.parquet     # one parquet per episode
videos/chunk-000/<video_key>/episode_000000.mp4
```

---

## 2. What the scan currently produces (data inventory)

All of this already exists. **Reuse it; do not re-derive geometry.** Shapes are also in
[conventions.md](conventions.md).

| Data | Shape / dtype | Where it lives |
|------|---------------|----------------|
| RGB keyframes | JPGs on disk `frame_NNNNNN.jpg`; or tensor `(3,H,W)` float `[0,1]` | `StreamingSLAM._save_frame` (`server/streaming_slam.py:1337`); `Submap.img_names`; `Submap.get_frame_at_index(i)` |
| Camera poses (cam→world) | `(S,4,4)` float32 | `Submap.get_all_poses_world(graph)` (`vggt_slam/submap.py:114`); `GraphMap.get_all_cam_matricies(graph, give_camera_mat)` (`vggt_slam/map.py:125`) |
| Camera/projection matrices (for intrinsics) | `(S,3,4)`/`(S,4,4)` | same accessors with `give_camera_mat=True`, then `decompose_camera` |
| Intrinsics K | derived | `decompose_camera(proj[:3,:])` (`vggt_slam/slam_utils.py`); `Submap.proj_mats` |
| Depth confidence | `(S,H,W)` float32 `[0,1]` | `Submap.conf` / `Submap.conf_masks` |
| Dense per-frame clouds (world) | `(S,H,W,3)` float32 + conf mask | `Submap.get_points_list_in_world_frame(graph)` (`vggt_slam/submap.py:174`) |
| Full aggregated cloud (world) | `(N,3)` f32 pos + `(N,3)` u8 color | `StreamingSLAM.gather_world_point_cloud()` (`server/streaming_slam.py:344`) |
| Frame timestamps/ids | float, parsed from filename | `Submap.get_frame_ids()` (`vggt_slam/submap.py:166`) |

**Semantic layer** (see [scene-report.md](scene-report.md); schemas in
`server/scene_report/schemas.py`):

| Artifact | Type | Where |
|----------|------|-------|
| Open-set 3D detections | dicts: `query`, `bounding_box{center,extent}`, `matched_submap`, `matched_frame`, `confidence`, `description` | built in `StreamingSLAM` (`server/streaming_slam.py:920-930`, `enrich_top_n` `:1051`) |
| Scene graph (objects + relations + metrics) | `SceneFacts` | `SceneFeatureExtractor.extract(solver, detections, scene_center)` (`server/scene_report/features.py:47`) |
| Object instance | `ObjectInstance` (`query`, world `center`, `extent`, `confidence`, `evidence: [EvidenceRef]`, `description: ObjectDescription`) | `schemas.py:75` |
| Fine-grained ID | `ObjectDescription` (`category`, `brand`, `model`, `color`, `visible_text`, `confidence`, …) | `schemas.py:46` |
| Spatial relations | `SpatialRelation` (`a`, `b`, `distance`, `relation∈{near,medium,far}`) | `schemas.py:87` |
| Evidence pointer | `EvidenceRef` (`submap_id`, `frame_idx`) | `schemas.py:15` |
| LLM report | `SceneReport` (`summary`, `room_type`, `objects:[ReportObject]`, `observations`, …) | `schemas.py:114`; built by `SceneReportBuilder.build_final`; cached at scan-end as `_last_scene_report` |
| Persisted scan record | report + facts + keyframe JPEGs + full `cloud.npz` | `server/scene_report/store.py` (`save_scene` `:135`) |

---

## 3. Conceptual mapping (camera = the agent)

| LeRobot/GR00T key | dtype / shape | Source |
|-------------------|---------------|--------|
| `observation.images.ego_view` | video, `(H,W,3)` u8 | the keyframe RGB |
| `observation.state` | float32 `(7,)` = `[tx,ty,tz,qx,qy,qz,qw]` | camera world pose (the exact row `write_poses_to_file` already emits) |
| `action` | float32 `(6,)` = `[Δtrans(3), Δrotvec(3)]` | ego-motion to the next keyframe (§7) |
| `observation.intrinsics` *(extra)* | float32 `(4,)` = `[fx,fy,cx,cy]` | `decompose_camera` → K |
| `task` / `annotation.*` | string ↔ int | scene report / object captions (§4.4) |

LeRobot **auto-adds** `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` —
you supply only declared features + task/annotations.

**1 scan session = 1 episode.**

---

## 4. The OpenReality export format (the format we own) — full spec

One directory per scan. **This is the source of truth**; the GR00T converter (§5) is a thin
transcode of this tree.

```
export/<scan_id>/
  meta/
    info.json          # schema + feature spec + frame conventions + caveats
    episode.json       # episode-level summary
    scene_graph.json   # SceneFacts (objects, relations, metrics) as JSON
    report.json        # SceneReport as JSON
    annotations.jsonl  # temporally-localized language (one JSON object per line)
  data/
    trajectory.parquet # per-keyframe rows
  videos/
    ego_view.mp4       # keyframe RGB stream, frame i == trajectory row i
  clouds/
    cloud.npz          # full world-frame cloud: positions (N,3) f32, colors (N,3) u8
    frames/<frame_index>.npz   # OPTIONAL per-keyframe cloud + conf mask
  grounding/
    objects.jsonl      # per-object grounding tuples (one JSON object per line)
```

### 4.1 `data/trajectory.parquet`

One row per keyframe, ordered by global frame index. Columns:

| Column | Type | Notes |
|--------|------|-------|
| `index` | int64 | global row index, 0..N-1 |
| `frame_index` | int64 | == `index` for a single-episode file |
| `episode_index` | int64 | 0 |
| `timestamp` | float32 | **synthetic uniform**: `frame_index / fps` (default `fps=10`) — see §7.3 |
| `source_frame_id` | float32 | the real `Submap.get_frame_ids()` value (true, non-uniform timing) |
| `observation.state` | list<float32>[7] | `[tx,ty,tz,qx,qy,qz,qw]` (quaternion xyzw) |
| `action` | list<float32>[6] | `[Δtrans(3), Δrotvec(3)]`; last row = zeros |
| `observation.intrinsics` | list<float32>[4] | `[fx,fy,cx,cy]` |

### 4.2 `meta/info.json`

```json
{
  "schema_version": "openreality-export/0.1",
  "scan_id": "<id>",
  "fps": 10,
  "coordinate_frame": "slam_world",
  "up_to_scale": true,
  "gravity_aligned": false,
  "metric": false,
  "units_note": "Coordinates/sizes are up-to-scale (SLAM world frame); not metric.",
  "features": {
    "observation.images.ego_view": {"dtype": "video", "shape": [H, W, 3], "names": ["height","width","channel"]},
    "observation.state":      {"dtype": "float32", "shape": [7], "names": ["tx","ty","tz","qx","qy","qz","qw"]},
    "action":                 {"dtype": "float32", "shape": [6], "names": ["dx","dy","dz","rx","ry","rz"]},
    "observation.intrinsics": {"dtype": "float32", "shape": [4], "names": ["fx","fy","cx","cy"]}
  }
}
```

### 4.3 `meta/scene_graph.json` & `meta/report.json`

Direct `model_dump()` of `SceneFacts` and `SceneReport` (`server/scene_report/schemas.py`).
`SceneFacts.objects[*].center` is **world-frame**; `cloud.npz` is **world-frame**; they
share a frame so object centers index into the cloud directly.

### 4.4 `meta/annotations.jsonl` — temporally-localized language (the key new work)

The scene report is *scene-level*; VLA conditioning wants *frame-windowed* language. Each
annotation line is scoped to a frame window resolved from an `EvidenceRef` via the
`(submap_id, frame_idx) → global_index` map (§7.1):

```json
{"frame_start": 12, "frame_end": 18, "channel": "object.caption",
 "text": "a MacBook Pro 14\" on the desk", "evidence": {"submap_id": 1, "frame_idx": 4}}
{"frame_start": 12, "frame_end": 30, "channel": "scene.caption",
 "text": "A home office with a desk, monitor, and chair.", "evidence": null}
{"frame_start": 20, "frame_end": 26, "channel": "spatial.relation",
 "text": "the keyboard is near the monitor", "evidence": {"submap_id": 2, "frame_idx": 1}}
```

Channels to emit (v1): `scene.caption` (whole episode, from `SceneReport.summary`),
`object.caption` (per `ObjectInstance`, windowed around its evidence frame ±k),
`spatial.relation` (per `SpatialRelation`, windowed around the union of the two objects'
evidence frames). Window radius `k` is a config (default 3).

### 4.5 `grounding/objects.jsonl`

One line per object — the `(frame, phrase, 2D box, 3D center)` grounding tuple set that
spatial-VLA training is starved for:

```json
{"object_id": 0, "query": "laptop", "world_center": [..3..], "world_extent": [..3..],
 "confidence": 0.82, "description": { ...ObjectDescription... },
 "evidence": [{"submap_id": 1, "frame_idx": 4, "global_frame_index": 14, "box_2d": [x0,y0,x1,y1]}],
 "relations": [{"other_id": 3, "distance": 0.7, "relation": "near"}]}
```

`box_2d` is the SAM 2D box cached per detection (the `box_2d` used by
`StreamingSLAM.make_detection_crop`, `server/streaming_slam.py:987`); include it when
available, else omit.

---

## 5. The GR00T-LeRobot v2 converter (§ target after the OpenReality tree exists)

`to_lerobot.py` transcodes `export/<scan_id>/` → the GR00T-LeRobot **v2** layout from §1.
Most fields map 1:1; the GR00T-specific piece is `meta/modality.json`:

```json
{
  "state":  {"camera_pose": {"start": 0, "end": 7}},
  "action": {"ego_motion":  {"start": 0, "end": 6}},
  "video":  {"ego_view": {"original_key": "observation.images.ego_view"}},
  "annotation": {"human.action.task_description": {}, "scene.caption": {}}
}
```

- Per-frame parquet rows carry `observation.state`, `action`, `timestamp`, `frame_index`,
  `episode_index`, `index`, `task_index`, and an `annotation.human.action.task_description`
  **int column** (index into `meta/tasks.jsonl`). GR00T's loader requires the dedicated
  `annotation.*` column, not just `task_index`.
- Load/verify with `LeRobotSingleDataset(path, modality_configs, EmbodimentTag.NEW_EMBODIMENT)`.
- **Decision:** emit v2 directly (simplest — one parquet + one mp4 per episode). Do **not**
  emit v3 then downconvert unless a future upstream need forces v3.

---

## 6. Reuse map (call these — do not reinvent)

| Need | Reuse |
|------|-------|
| Per-keyframe poses + skip LC submaps + pair with frame_ids | **mirror `GraphMap.write_poses_to_file` exactly** (`vggt_slam/map.py:134-162`) — it is the canonical pose↔frame_id↔order alignment |
| All camera matrices in order | `GraphMap.get_all_cam_matricies(graph, give_camera_mat=True)` (`map.py:125`) |
| Decompose projection → K, R, t | `decompose_camera` (`vggt_slam/slam_utils.py`; used at `map.py:149`) |
| Rotation matrix → quaternion (xyzw) / rotvec | `scipy.spatial.transform.Rotation` (already imported `map.py:5`): `.as_quat()`, `.as_rotvec()` |
| Keyframe tensor → RGB uint8 bytes | `encode_frames` pattern (`server/scene_report/keyframes.py:71`): `submap.get_frame_at_index(i).cpu().permute(1,2,0).numpy()*255` |
| Full world cloud | `StreamingSLAM.gather_world_point_cloud()` (`server/streaming_slam.py:344`) |
| Per-keyframe cloud + conf mask | `Submap.get_points_list_in_world_frame(graph)` (`vggt_slam/submap.py:174`) |
| Scene graph | `SceneFeatureExtractor.extract(solver, detections, scene_center)` (`server/scene_report/features.py:47`) |
| Final report | scan-end `_last_scene_report` (built by `SceneReportBuilder.build_final`) |
| EvidenceRef → keyframe image | `smap.get_submap(ref.submap_id).get_frame_at_index(ref.frame_idx)` |
| Persistence (Stage 4) | `store.py:save_scene` (`server/scene_report/store.py:135`) |

The new `server/export/` package must stay **GPU-free and duck-typed** (numpy/scipy/pydantic
only), mirroring `server/scene_report/` so it is unit-testable with fakes and can run on the
CPU broker.

---

## 7. Correctness-critical details (the traps)

### 7.1 The `(submap_id, frame_idx) → global_index` map

Build it during the **same ordered walk** used for poses so it can never drift:
```
global_index = 0
index_map = {}                      # (submap.get_id(), local_pose_idx) -> global_index
for submap in map.ordered_submaps_by_key():
    if submap.get_lc_status():      # skip loop-closure submaps (duplicates)
        continue
    n = len(submap.get_all_poses_world(graph))      # == number of camera mats for this submap
    for local in range(n):
        index_map[(submap.get_id(), local)] = global_index
        global_index += 1
```
`EvidenceRef.submap_id` is `submap.get_id()`; `EvidenceRef.frame_idx` is the local index
used by `get_frame_at_index` / `get_all_poses_world`. Resolve every evidence ref through
`index_map`; if a ref misses (e.g. it points at an LC submap), drop it gracefully.

### 7.2 frame_ids vs poses length (assert it)

`write_poses_to_file` zips `submap.get_frame_ids()` with the per-submap poses assuming equal
length. `get_frame_ids()` "does not include loop-closure frames." **Assert
`len(frame_ids) == len(poses)` per non-LC submap**; if it ever fails, fall back to a
synthetic `source_frame_id = global_index` and log loudly rather than mis-pairing.

### 7.3 Action math & timeline

```
# poses are 4x4 cam->world built from decompose_camera (R, t) per keyframe, in global order
T_t, T_t1 = poses[i], poses[i+1]
T_rel = np.linalg.inv(T_t) @ T_t1          # motion expressed in frame t
dtrans = T_rel[:3, 3]
drot   = Rotation.from_matrix(T_rel[:3, :3]).as_rotvec()
action[i] = np.concatenate([dtrans, drot]).astype(np.float32)
action[N-1] = np.zeros(6, np.float32)
```
Verify `t` from `decompose_camera` is the **camera position in world** (it is what
`write_poses_to_file` writes as TUM xyz). **Timeline:** keyframes are motion-gated (not fixed
fps). Use a synthetic uniform `timestamp = frame_index / fps` so LeRobot's timestamp↔fps
tolerance check passes; keep true timing in `source_frame_id`.

### 7.4 Frame ↔ video alignment

`videos/ego_view.mp4` frame `i` **must** equal `trajectory.parquet` row `i`. Encode frames in
the exact global order from §7.1. Do not let any frame dropping/reordering desync them.

---

## 8. Implementation stages (each has a verifiable goal)

> New package: `server/export/`. First/primary driver is the **offline batch path**
> (`main.py`) where a full `Solver` is in hand — simplest and most complete. Streaming /
> broker paths come later.

### Stage 0 — Scaffolding & schema
- **Create:** `server/export/__init__.py`, `server/export/schema.py` (export-manifest
  pydantic models + `SCHEMA_VERSION`; builds `info.json`/`episode.json`).
- **Verify:** `python -c "import server.export.schema"` works; a unit test round-trips
  `info.json`/`episode.json` and asserts `up_to_scale/gravity_aligned` are present.

### Stage 1 — Trajectory + video + cloud (the core dataset)
- **Create:** `server/export/trajectory.py` (build rows + `index_map`, §7.1/7.3),
  `server/export/video.py` (encode `ego_view.mp4`, §7.4), `server/export/clouds.py` (reuse
  `gather_world_point_cloud`; optional per-frame clouds), `server/export/writer.py`
  (`export_scene(solver, detections, report, out_dir, fps=10)` orchestrator).
- **Wire:** add `--export_dataset <dir>` to `main.py`; after SLAM finishes, call
  `export_scene(...)`.
- **Verify (E2E):** `python main.py --image_folder examples/kitchen/images --export_dataset /tmp/ds`
  produces `/tmp/ds/<scan>/` with: `trajectory.parquet` (N rows, all `state`/`action`
  finite, `action[-1]` is zeros), `ego_view.mp4` (N frames, same order), `cloud.npz`
  loadable. Assert `N == len(get_all_cam_matricies(...))`.
- **Unit:** pose↔state round-trip (state→R,t→pose reproduces input); action recovers a known
  relative transform on a synthetic 3-pose fixture.

### Stage 2 — Grounding sidecar + temporally-localized annotations
- **Create:** `server/export/grounding.py` — emit `scene_graph.json`, `report.json`,
  `grounding/objects.jsonl`, and `annotations.jsonl` (§4.4), resolving every `EvidenceRef`
  through `index_map`.
- **Reuse:** `SceneFeatureExtractor.extract` for facts; scan-end `_last_scene_report`.
- **Verify:** every `grounding/objects.jsonl` evidence `global_frame_index` is in
  `[0, N)` and indexes a real `trajectory.parquet` row + `mp4` frame; every
  `annotations.jsonl` window `[frame_start, frame_end] ⊆ [0, N)`; object `world_center`
  values fall within the `cloud.npz` AABB.
- **Unit:** duck-typed fake solver/facts (mirror `tests/test_scene_features.py`) → assert
  annotation windows and the evidence join.

### Stage 3 — GR00T-LeRobot v2 converter
- **Create:** `server/export/to_lerobot.py` (OpenReality tree → GR00T-LeRobot v2 + §5
  `modality.json`); CLI `scripts/export_to_groot.py`.
- **Verify (E2E):** run on a Stage-1/2 export; load with GR00T
  `LeRobotSingleDataset(path, modality_configs, EmbodimentTag.NEW_EMBODIMENT)` (in a venv per
  §1) and assert a sample has `observation.state (7)`, `action (6)`,
  `observation.images.ego_view (C,H,W)`, and a resolvable
  `annotation.human.action.task_description`. Also load plain with `lerobot`'s
  `LeRobotDataset` if feasible.
- **Unit:** `modality.json` index ranges equal feature widths; `tasks.jsonl` indices
  referenced by the annotation column all exist.

### Stage 4 — GPU-free offline export from persisted scenes
- **Why:** the dashboard/broker should export scan *history* without booting a GPU. The
  persisted record (`store.py`) already has facts + report + keyframe JPEGs + full cloud, but
  **lacks per-keyframe poses + intrinsics**.
- **Change:** extend `ModalScenePersistence.save_scene` (and `InMemoryScenePersistence`) to
  persist a small `trajectory.npz` (poses `(N,4,4)` + intrinsics `(N,4)` + `source_frame_id`)
  — cheap (~N×20 floats). Update the persist call site (`_persist_scene_report`).
- **Create:** an `export_from_record(record, out_dir)` path in `writer.py` that reads the
  persisted record instead of a live `Solver`; add a broker route (e.g.
  `GET /api/scenes/<id>/export`) returning a zipped export.
- **Verify:** export a previously-persisted scan with no GPU/Solver in process; diff its
  `trajectory.parquet` against a live Stage-1 export of the same scan (poses match to fp
  tolerance).

### Stage 5 — Stretch / future (do not block earlier stages)
- Batch-generate **grounded spatial-QA** pairs into `grounding/spatial_qa.jsonl` reusing the
  `_grounded_scene_answer` core (`server/scene_report/`), each with a 3D `focus` point + cited
  frames.
- **Metric + gravity alignment**: upgrade grounding from *relative* → *metric* (requires
  scale + up-vector estimation) so coordinates become affordance-usable. Track separately.
  → **Built** for the Isaac Sim/Lab USD target (floor-plane gravity + user-supplied scale):
  see [isaac-export.md](isaac-export.md) (`server/export/isaac/`). The *grounding* layer here is
  still relative; the Isaac scene is the metric/gravity-aligned consumer.

---

## 9. End-to-end verification (acceptance for the workstream)

1. `python main.py --image_folder examples/kitchen/images --export_dataset /tmp/ds` (Stage 1–2).
2. Load `trajectory.parquet` + `cloud.npz`; assert object centers in `scene_graph.json` and
   the cloud share the world frame (centers inside the cloud AABB).
3. Spot-check 3 `annotations.jsonl` lines against the cited `mp4` frames (open them; the text
   should describe what's visible).
4. Run `to_lerobot.py`; load with GR00T `LeRobotSingleDataset(..., NEW_EMBODIMENT)` — it parses
   and yields the declared shapes.
5. `pytest tests/test_export_*.py` green.

---

## 10. Decisions already made (don't re-litigate) & open questions

**Decided:** (a) build *both* outputs — core trajectory dataset **and** grounding sidecar;
(b) optimize for a **format we control** (research-first), GR00T-convertible but not
GR00T-bound; (c) emit **GR00T-LeRobot v2** (not v3) for the converter; (d) state = pose 7-vec,
action = ego-motion 6-vec, camera/`NEW_EMBODIMENT`; (e) synthetic uniform timeline + keep true
`source_frame_id`.

**Open (flag to product before/while building):** state rotation as quaternion vs. 6D
rotation (quaternion v1; 6D is more learning-friendly — revisit if we train policies);
whether to ship per-frame clouds (`clouds/frames/`) by default (off v1 — large); window
radius `k` for annotations; whether depth gets its own `observation.depth.*` channel (deferred
— lossy in mp4).

---

## 11. As-built (what shipped) — implementation map

Stages 1–4 are implemented; Stage 5 is not. This section is the source of truth for the
**delivered code** and where it deviates from §1–§10. All of `server/export/` is GPU-free /
duck-typed (numpy / scipy / pydantic / pandas / pyarrow / cv2); heavy imports (torch, VGGT,
`decompose_camera`) are lazy, inside functions.

### 11.1 File map (`server/export/`)

| File | Responsibility | Key entry points |
|------|----------------|------------------|
| `schema.py` | Stage 0 manifests | `build_info`, `build_episode_summary`, `standard_features`, `SCHEMA_VERSION` |
| `trajectory.py` | §4.1/§7.1–7.3 rows | `_ordered_keyframes` (the **single canonical walk**), `build_index_map`, `extract_trajectory`, `_pose_to_state`, `_relative_action`, `write_trajectory_parquet` |
| `video.py` | §7.4 `ego_view.mp4` | `write_ego_view_video`, `frame_to_rgb_uint8` (torch **or** numpy CHW/HWC) |
| `clouds.py` | §4 clouds/ | `gather_full_cloud`, `write_full_cloud`, `write_cloud_array` (Stage-4 reuse), `write_frame_clouds` |
| `grounding.py` | §4.3–4.5 sidecar | `write_scene_graph`, `write_report`, `write_objects_jsonl`, `write_annotations_jsonl`, `_resolve_evidence` |
| `writer.py` | orchestrators | `export_scene` (live Solver, Stage 1–2), `export_from_record` (Stage 4) |
| `record.py` | **new (not in scaffold)** Stage-4 glue | `build_trajectory_arrays` (run on GPU worker), `build_rows_from_trajectory`, `load_record_from_store` |
| `to_lerobot.py` | Stage 3 | `convert_to_groot_lerobot`, `build_modality_json` |
| `__init__.py` | lazy public proxies | `export_scene`, `export_from_record`, `convert_to_groot_lerobot`, `SCHEMA_VERSION` |

**Other touched files:**
- `main.py` — added `--export_dataset <dir>` + `--export_fps` (default 10); calls `export_scene` after SLAM, before the interactive `--run_os` loop (which never returns).
- `scripts/export_to_groot.py` — CLI wrapper for the Stage-3 converter.
- `server/scene_report/store.py` — `save_scene(..., trajectory=...)` persists a compact
  `trajectory.npz` (`poses (N,4,4)`, `intrinsics (N,4)`, `source_frame_id (N,)`); added
  `get_trajectory(user_id, scan_id)` reader + `trajectory_key`/`trajectory_count` on the record.
- `modal_data_export_test.py` — **standalone throwaway Modal app** (`data-export-test`): runs
  SLAM + `export_scene` (+ optional GR00T convert) on an A100, seeded from an image folder or a
  `server/demo_videos/` clip; the `verify_groot` function loads the result with the real
  GR00T loader on a Linux/CPU image (`--verify-groot-load`). See §11.4.

**Local test helpers (no GPU / no Modal):**
- `scripts/make_local_export_fixture.py` — runs the **real** `export_scene` +
  `convert_to_groot_lerobot` on a tiny synthetic scan (real projection matrices, gradient
  frames, cube cloud) → a full OpenReality + GR00T tree on disk. For testing the converter /
  loader without SLAM.
- `scripts/inspect_export.py` — dumps every meta file + parquet schema/first-row + mp4 frame
  count + cloud shape, and runs **19 PASS/FAIL invariant checks** over an OpenReality and/or
  GR00T tree (`--export <dir> --groot <dir>`). The artifact to paste when verifying an export.
- `scripts/load_groot.py` — standalone `LeRobotSingleDataset(..., NEW_EMBODIMENT)` load for a
  **local** GR00T tree; run it in a GR00T env (native arm64 mac or Linux), not this repo's env.

### 11.2 Deviations / decisions made while building

- **mp4 codec is probed, not fixed.** OpenCV's writer backend is platform-specific (macOS =
  AVFoundation, which opens `avc1` but **not** `mp4v`; Linux/Modal = FFMPEG, usually `mp4v`).
  `video.py` tries `("avc1", "mp4v", "H264", "h264")` and keeps the first that `isOpened()`;
  raises a clear `RuntimeError` if none open. `export_scene` hard-asserts
  `frames_written == len(rows)` so a desync (§7.4) fails loudly instead of shipping.
- **Stage-4 record path emits `scene_graph.json` + `report.json` only** — not the
  temporally-windowed `annotations.jsonl` / `grounding/objects.jsonl`. The persisted record
  keeps facts/report but **not** the `(submap_id, frame_idx) → global_index` map, which the
  windowing needs; the live `export_scene` path is the full-fidelity sidecar.
- **`export_scene(report=None)` is supported** (the offline `main.py` path passes no report):
  `_resolve_facts` derives `SceneFacts` from the live `Solver` via `SceneFeatureExtractor`
  when the report is absent or carries no objects.
- **Grounding sidecar is best-effort** — wrapped in try/except inside `export_scene` so a
  semantic-layer failure never drops the (already-written) trajectory dataset.
- **`relations` in `objects.jsonl`** are resolved from the query-keyed `SpatialRelation`s to
  concrete `other_id`s by matching object queries; evidence refs that don't resolve through
  `index_map` (e.g. LC submaps) are dropped, leaving an empty `evidence` list rather than failing.

### 11.3 Tests (26, all green; full suite 133)

`tests/export_fakes.py` builds **real** camera projection matrices (so the production
`decompose_camera` round-trips) and stubs `matplotlib` only when it's genuinely unimportable
(numpy-2 ABI breakage in this env) — a healthy env keeps the real module.

| Test file | Covers |
|-----------|--------|
| `test_export_schema.py` | Stage 0 manifests + caveats (pre-existing) |
| `test_export_trajectory.py` | pose↔state round-trip, ego-motion action, `index_map` LC-skip + ordering, parquet |
| `test_export_video.py` | `frame_to_rgb_uint8`, frame-count == row-count (skips if no mp4 writer) |
| `test_export_grounding.py` | evidence join, `global_frame_index ∈ [0,N)`, annotation windows clamped, box_2d |
| `test_export_writer.py` | full-tree E2E, object center ∈ cloud AABB, `report=None` path |
| `test_export_to_lerobot.py` | v2 layout, modality index ranges == feature widths, `tasks.jsonl` indices valid |
| `test_export_record.py` | record rows match live rows (fp tol), `store` trajectory round-trip, GPU-free export |

Run: `pytest tests/test_export_*.py -q`. **Not verifiable in this env (need GPU/weights or
the `gr00t` package):** the real `main.py --export_dataset` E2E (§9.1) and loading with GR00T
`LeRobotSingleDataset(..., NEW_EMBODIMENT)` (§9.4) — close those with §11.4.

### 11.4 GPU test harness — `modal_data_export_test.py`

A standalone, throwaway Modal app (`modal.App("data-export-test")`) that runs the **real**
inference path independent of the production streaming/batch apps. Mirrors `modal_app.py`'s
known-good SLAM image + adds the export deps (`pydantic`/`pandas`/`pyarrow`) and the
`server/{__init__,export,scene_report}` dirs (`/root/project` on `sys.path`). It reuses the
shared `vggt-slam-models` (weights) and `vggt-slam-data` (I/O) Volumes.

```bash
modal run modal_data_export_test.py::download_models               # one-time weight cache
modal run modal_data_export_test.py --demo-video office_loop.mp4 --to-lerobot
modal run modal_data_export_test.py --image-folder ./my_frames     # or a local frame folder
```

- **Seed from a `server/demo_videos/` clip** (`--demo-video`): the entrypoint LFS-checks +
  uploads it; the remote decodes it into numbered `frame_NNNNNN.jpg` (subsampling matches
  `server/app.py` `VideoFeeder`: `skip = round(video_fps / extract_fps)`), then runs the
  `main.py` SLAM loop + `export_scene` (+ optional `convert_to_groot_lerobot`). Prefer
  `office_loop.mp4` — it's a loop, so it actually exercises the LC-skip / index-map alignment.
- Knobs: `--extract-fps` (decode rate, default 4), `--rotate {0,90,180,270}`, `--max-frames`
  (cost cap), plus the usual `--submap-size`/`--max-loops`/`--export-fps`.
- Downloads the whole export tree to `./export_results/exports/<scan>/` (+ `groot/<scan>/`).
- Writes seed/frames/exports under the shared `vggt-slam-data` Volume — point at a
  `-test`-suffixed Volume if you need isolation.

**§9.4 loader check — needs a native arm64 / Linux Python.** GR00T pins `torch==2.7.1`,
which ships wheels for `manylinux_*` and `macosx_11_0_arm64` but **not** `macosx_*_x86_64`.
So the gotcha is an **x86_64 Python toolchain** — e.g. an Intel Anaconda (`~/opt/anaconda3`)
or a Rosetta shell — even on Apple-Silicon hardware: `pip`/`uv`/conda then fail the torch
resolve with `macosx_..._x86_64 has no wheels`. Check with `uname -m` (shell) +
`python -c "import platform;print(platform.machine())"` (env); fix by creating a **native
arm64** env (miniforge-arm64 or a uv-managed python), or just load on Linux:

- `modal run modal_data_export_test.py --demo-video office_loop.mp4 --verify-groot-load`
  runs SLAM → export → convert → **loads the tree with `LeRobotSingleDataset(...,
  NEW_EMBODIMENT)`** in the app's `verify_groot` function (a separate Linux/CPU image that
  `pip install -e .`'s Isaac-GR00T). Prints the dataset length + first-sample key→shape map.
- Locally (native arm64 mac or Linux): `pip install -e .` Isaac-GR00T, then
  `scripts/load_groot.py --dataset <groot_dir>`. **Skip the separate flash-attn install** on
  mac — it's CUDA-only and the data loader doesn't need it.
- Format-only checks need no GR00T at all: `scripts/make_local_export_fixture.py` (real export
  code, synthetic scan, no GPU) + `scripts/inspect_export.py` (dumps every meta file + 19
  PASS/FAIL invariant checks over the OpenReality + GR00T trees).
