# DIMOS-NODE — diff-plan: EXP-36 `OreosVisionNode` replay demo → live mode

Precise upgrade plan for
`platform/experiments/exp36_oreos_dimos_spike/scripts/oreos_vision_node.py`
(results-playback DimOS module) to a **live** mode driven by this package's
`OreosLiveClient` against the deployed `oreos-live-node` Modal app. This is a
plan, not an integration: **the exp36 scripts are not modified** — live mode
would land as a new module/config in a future exp or in `server/` proper, with
the replay-playback path kept intact for offline demos.

## What stays exactly as-is (results playback → shared by live mode)

| Piece | Why it survives |
|---|---|
| Typed ports `color_image: In[Image]`, `odom: In[PoseStamped]`, `oreos_cloud: Out[PointCloud2]`, `oreos_pose: Out[PoseStamped]` | The DimOS-facing contract is unchanged — that pluggability is the whole point of EXP-36 Phase B. |
| rerun logging (`rrd_path`, `rr_image_every`), odometry overlay | Pure visualization; live results log the same way. |
| `PointCloud2` construction from points+colors, `PoseStamped` publish | Same message assembly, different data source. |
| `world_frame` config, blueprint/autoconnect wiring | Untouched. |

## What changes

### 1. Data source: staged `submaps/*.npz` → per-chunk WebSocket results

* **Delete** (in the live subclass/mode, not from the file): `_load_results()`'s
  npz staging, the `emit_at <= ts` playback trigger in `_on_image`, and the
  pre-loaded `_pose_ts`/`_pose_mats` arrays.
* **Add**: an `OreosLiveClient`-based connection (`server/oreos/recordings/live_node/client.py`)
  opened in `start()`:
  * `connect()` + initial `reset` (fresh SLAM session per module start);
  * frames arriving on `color_image` are jpeg-encoded (`cv2.imencode`, the
    inverse of `Image.to_opencv`) and appended to a chunk buffer using the SAME
    16+1-overlap boundary as `build_chunk_plan` (imported — never re-implement);
  * on chunk boundary: `protocol.encode_message` + `iter_wire_frames` → sender
    queue. The node does NOT replay from disk, so `OreosLiveClient.run()`'s
    replay clock is unused — factor its send/receive threads out or drive the
    client's `_send_q`/`_pending` directly (small refactor: extract a
    `stream_chunk(names, blobs)` method from `run()`; the threads already
    support it).
* **Result handling** (reader-thread callback instead of replay-clock trigger):
  each `result` message carries `poses_tum_lines` (world-frame, sign-repaired)
  and `cloud_summary`. Publish immediately on arrival:
  * `oreos_pose`: parse the newest TUM line → `PoseStamped` (same quaternion
    order — TUM is `x y z qx qy qz qw`, matching the existing
    `Rotation.as_quat()` usage);
  * `oreos_cloud`: today's wire result carries a cloud *summary* only. For live
    clouds, extend `modal_stream.py`'s result payload with an optional
    downsampled point blob (voxel-downsampled float32 xyz+rgb, the
    `modal_recon.py` npz recipe) behind a `send_points: true` handshake flag —
    the protocol's blob framing already supports it (that's what `blob_lens`
    is for; fragmentation handles size). Until then, live mode publishes poses
    at chunk rate and clouds stay a playback-only feature.

### 2. Threading model

Replay node: everything happens on the `color_image` subscriber callback under
one lock. Live node:

* **subscriber thread** (DimOS-owned): jpeg-encode + buffer append + boundary
  check only — never blocks on the network (encode is ~4 ms/frame at 720p).
* **sender thread** (from `OreosLiveClient`): drains the chunk queue into the
  WebSocket; socket backpressure lands here, not on the subscriber.
* **reader thread** (from `OreosLiveClient`): receives results, publishes
  `oreos_pose`/`oreos_cloud`, logs rerun. Publishing from a non-subscriber
  thread is fine for DimOS `Out` ports (LCM publish is thread-safe), but keep
  the existing `self._lock` around rerun calls — rerun's `set_time` is
  per-thread otherwise.
* **lifecycle**: `stop()` (new rpc) sends `end`, waits for `bye` (bounded),
  closes. A dropped connection sets a `degraded` flag and retries with
  exponential backoff; on reconnect send `reset` ONLY if the solver-side state
  is stale (server keeps SLAM state across connections by design).

### 3. Config additions (`OreosVisionNodeConfig`)

```python
live_url: str = ""        # wss endpoint; "" = replay-playback mode (current behavior)
chunk_size: int = 16      # must match server-side submap semantics (16+1 overlap)
jpeg_quality: int = 0     # 0 = passthrough; see client.py — re-encode only helps
                          # if the camera emits lightly-compressed jpegs
```

`results_dir`/`align_json` stay for playback mode. In live mode `align_json`
has no meaning (no GT to Umeyama against) — the Sim(3) block is skipped, poses
publish in OpenReality's own world frame, and downstream consumers align via
their own anchor (the EXP-36 alignment was PREREG-registered gauge-fixing for
scoring, not a runtime feature).

### 4. Latency semantics to surface honestly

* Publish each pose with `ts = source frame timestamp` (as today) — consumers
  see SLAM results arrive `lag_behind_clock_s` late (measured here:
  `sessions/go2_short/live_node/live_node_results.json`). Do not backdate-hide
  the lag: expose `oreos_lag_s` as a module stat so a planner can gate on it.
* The chunk cadence bounds pose freshness at ~2.2 s + round-trip; that is the
  measured product envelope from this task, not a bug in the node.

## Explicit non-goals of this plan

* No onboard GPU claim — the node still calls a remote A100.
* No change to exp36's `oreos_replay_blueprint.py` / recorded-results demos.
* No multi-robot fan-in: `oreos-live-node` is `max_containers=1`, one session
  at a time; a fleet needs per-robot containers (Modal `max_containers` lift +
  session routing) — separate task.
