"""
StreamingSLAM — wraps the VGGT-SLAM 2.0 Solver for real-time streaming.

Uses VGGT-SLAM 2.0 APIs exclusively:
  - submap.get_points_in_world_frame(graph)
  - submap.get_all_poses_world(graph)
  - submap.get_points_colors()
  - submap.get_all_semantic_vectors()
  - submap.get_points_in_mask(frame_idx, mask, graph)
  - solver.run_predictions(image_names, model, max_loops)
"""

import cv2
import numpy as np
import torch
import base64
import threading
import tempfile
import os
import time
import queue
from PIL import Image

from vggt_slam.solver import Solver
from vggt_slam.object_detector import ObjectDetector
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import compute_image_embeddings


def _detection_env_defaults() -> tuple[float, float, int]:
    """Read the detection-recall env knobs, defaulting to the historical hardcoded values
    (0.15 / 0.80 / 1) when unset -- so this commit adds knobs, not new defaults.

    Style-matched to ``modal_streaming.py``'s ``SUBMAP_SIZE`` pattern
    (``int(os.environ.get("SUBMAP_SIZE", "16"))``), read here (rather than threaded through
    ``app.initialize()``) so BOTH the live worker (``modal_streaming.py``) and the offline
    pilot harness (``reconstruct_pilot.py`` / ``modal_reconstruct_pilot.py``) pick them up --
    both ultimately construct a ``StreamingSLAM`` via ``app.initialize()``, which doesn't
    thread detection knobs through as constructor kwargs.

    Split out from ``StreamingSLAM.__init__`` so it's unit-testable without constructing a
    full ``StreamingSLAM`` (which loads the VGGT model + SAM3/CLIP ``ObjectDetector`` -- GPU,
    heavy, not available in the GPU-free test suite).

    Returns ``(clip_threshold, sam_threshold, frames_per_window)``:
    - ``DETECTION_CLIP_THRESHOLD`` -> the CLIP similarity gate for a candidate frame/query.
    - ``DETECTION_SAM_THRESHOLD`` -> the SAM segmentation-score gate for a detected mask.
    - ``DETECTION_FRAMES_PER_WINDOW`` -> how many top-CLIP-ranked frames per submap get a SAM
      pass per query (``SAM_TOP_K_FRAMES``); more frames = more grounding chances.
    """
    clip_threshold = float(os.environ.get("DETECTION_CLIP_THRESHOLD", "0.15"))
    sam_threshold = float(os.environ.get("DETECTION_SAM_THRESHOLD", "0.80"))
    frames_per_window = int(os.environ.get("DETECTION_FRAMES_PER_WINDOW", "1"))
    return clip_threshold, sam_threshold, frames_per_window


class StreamingSLAM:
    def __init__(self,
                 submap_size=8,
                 min_disparity=30.0,
                 conf_threshold=25.0,
                 vis_stride=4,
                 lc_thres=0.95):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Object detector (PE-Core CLIP + SAM3)
        print("Loading ObjectDetector (PE-Core CLIP + SAM3)...")
        self.object_detector = ObjectDetector(device=self.device)
        print("ObjectDetector loaded!")

        # Solver — skip viser viewer since we stream to the web frontend
        self.solver = Solver(
            init_conf_threshold=conf_threshold,
            lc_thres=lc_thres,
            skip_viewer=True,
        )

        # VGGT model (VGGT-Omega backbone via the vggt-omega-slam fork).
        # Weights are pre-fetched into the torch hub cache by download_models().
        print("Loading VGGT model...")
        self.model = VGGT()
        _ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", "vggt_omega_1b_512.pt")
        self.model.load_state_dict(torch.load(_ckpt, map_location="cpu"))
        self.model.eval()
        # Weights stay fp32 (unlike SPARK): Omega autocasts its aggregator to
        # bf16 internally, but its heads run with autocast disabled and cast
        # their inputs to fp32 — bf16 weights make F.layer_norm throw a
        # Float/BFloat16 mismatch in the camera head.
        self.model = self.model.to(self.device)
        print("VGGT (Omega) model loaded!")

        # Configuration
        self.submap_size = submap_size
        self.overlapping_window_size = 1
        self.min_disparity = min_disparity
        self.max_loops = 1
        self.vis_stride = vis_stride

        # State
        self.frame_count = 0
        self.image_names_subset = []
        self.image_io_times_subset = []
        self.temp_dir = tempfile.mkdtemp()
        self.is_running = False
        self._stop_event = threading.Event()
        self._process_thread: threading.Thread | None = None
        # Set for the duration of a submap pass (VGGT + graph opt). Lets the offline
        # per-scene harness wait for an in-flight pass to finish before it finalizes or
        # resets the solver, so a reset can't race a running VGGT pass on shared state.
        self._processing = threading.Event()

        # Beacon state
        self.pending_beacons = []
        self.resolved_beacons = []
        self.latest_scene_center = np.zeros(3)

        # Detection state
        self.active_queries = []
        self.accumulated_detections = []
        self._detection_lock = threading.Lock()
        self._sam_cache = {}

        # Detection thresholds + frames-per-window — env-configurable (see
        # _detection_env_defaults' docstring for the env vars + why they're read here rather
        # than threaded through app.initialize()). Defaults unchanged (0.15 / 0.80 / 1).
        _clip_thresh, _sam_thresh, _frames_per_window = _detection_env_defaults()
        self.detection_clip_thresholds = {"default": _clip_thresh}
        self.detection_sam_thresholds = {"default": _sam_thresh}
        # Shadows the SAM_TOP_K_FRAMES class default with an instance attribute.
        self.SAM_TOP_K_FRAMES = _frames_per_window

        # Fine-grained object enrichment (VLM captioner; injected via set_object_enricher,
        # None in plain local runs). Descriptions are cached per (submap, frame, query) so a
        # click, a re-click, and the finalize top-N pass all share a single LLM call.
        self.object_enricher = None
        self._description_cache = {}  # (submap_id, frame_idx, query) -> ObjectDescription dict
        self._description_lock = threading.Lock()

        # Cached last stream data (avoid redundant extract_stream_data calls)
        self._last_stream_data: dict | None = None

        # Incremental extraction cache
        self._submap_cache: dict[int, dict] = {}
        self._scene_center: np.ndarray = np.zeros(3)

        # External queues (set by app.py)
        self.frame_queue = None
        self.result_queue = None
        self.result_ready_event = None
        self.event_loop = None

        # Spatial agent (set by app.py when session-scoped agent runtime is enabled)
        self.spatial_agent = None

        print(f"Temp directory: {self.temp_dir}")

    def start(self):
        if self._process_thread is not None and self._process_thread.is_alive():
            return
        self.is_running = True
        self._stop_event.clear()
        self._process_thread = threading.Thread(
            target=self.process_loop,
            daemon=True,
            name="StreamingSLAMLoop",
        )
        self._process_thread.start()
        print("SLAM processing loop started")

    def wait_until_idle(self, timeout: float = 120.0) -> bool:
        """Block until the processing loop is idle — the frame queue is drained AND no
        submap pass is in flight — or ``timeout`` elapses. Used by the offline per-scene
        harness so a finalize/reset never races a running VGGT pass on shared solver state.
        Returns ``True`` if idle was reached, ``False`` on timeout. Must not be called from
        the processing thread itself (it would deadlock waiting on its own pass)."""
        if threading.current_thread() is self._process_thread:
            return not self._processing.is_set()
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            q = self.frame_queue
            q_empty = q is None or q.empty()
            if q_empty and not self._processing.is_set():
                return True
            time.sleep(0.05)
        q = self.frame_queue
        return (q is None or q.empty()) and not self._processing.is_set()

    def stop(self, flush=False, join_timeout=2.0):
        """Stop the processing loop.

        flush=True builds a final submap from any buffered keyframes that never
        filled a full batch (e.g. a short or low-motion clip) instead of
        discarding them. The flush runs only after the loop thread has fully
        exited, so it never races the loop on shared SLAM state. Because it may
        run VGGT inference, callers should invoke a flush stop off the event
        loop (e.g. on the GPU executor).

        join_timeout bounds the wait for the loop thread to exit. The live path uses
        the short default (2s) for snappy teardown; the offline per-scene harness passes
        a generous timeout so a long in-flight VGGT pass fully finishes before any
        finalize/reset touches shared solver state.
        """
        self.is_running = False
        self._stop_event.set()
        on_process_thread = threading.current_thread() is self._process_thread

        # Normal stop: drop the partial batch immediately so a teardown can't
        # trigger a submap if the loop is still mid-iteration. Flush stop: keep
        # it so we can build a final submap once the loop thread has exited.
        if not flush:
            self.image_names_subset.clear()
            self.image_io_times_subset.clear()

        thread = self._process_thread
        if thread is not None and thread.is_alive() and not on_process_thread:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                print(f"[streaming_slam] WARNING: processing thread still alive after "
                      f"{join_timeout}s join — proceeding may race shared SLAM state")

        if flush:
            # Only safe to touch shared SLAM state once the loop thread is gone.
            if not on_process_thread and (thread is None or not thread.is_alive()):
                self.flush_partial_submap()
            self.image_names_subset.clear()
            self.image_io_times_subset.clear()

        self._process_thread = None
        print("SLAM processing loop stopped")

    def flush_partial_submap(self):
        """Build a final submap from leftover buffered keyframes.

        The processing loop only builds a submap once the batch reaches
        submap_size + overlapping_window_size frames, so a stream that ends
        early (a short or low-motion clip) would otherwise discard everything it
        buffered and produce no points or detections at all. VGGT runs fine on
        fewer than submap_size frames, so flush whatever is buffered.

        Must run with the processing loop stopped — process_submap() mutates
        shared SLAM state that the loop also touches.
        """
        n = len(self.image_names_subset)
        # n <= overlapping_window_size means only carried-over overlap frame(s)
        # remain (already in a prior submap); n < 2 is too few to reconstruct.
        if n <= self.overlapping_window_size or n < 2:
            return
        print(f"Flushing partial submap with {n} frames...")
        self.process_submap()

    # ------------------------------------------------------------------
    # Frame processing loop
    # ------------------------------------------------------------------

    def process_loop(self):
        """Main processing loop — reads frames, checks disparity, triggers submap processing."""
        while self.is_running and not self._stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)

                io_start = time.perf_counter()
                img_bytes = base64.b64decode(frame_data['image'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Failed to decode frame")
                    continue

                # Check disparity via optical flow
                enough_disparity = self.solver.flow_tracker.compute_disparity(
                    frame, self.min_disparity
                )

                if enough_disparity:
                    frame_path = self._save_frame(frame)
                    self.image_names_subset.append(frame_path)
                    self.image_io_times_subset.append(time.perf_counter() - io_start)
                    self.frame_count += 1
                    print(f"Keyframe {self.frame_count} added (subset size: {len(self.image_names_subset)})")

                # Process submap when batch is full
                if len(self.image_names_subset) == self.submap_size + self.overlapping_window_size:
                    print(f"Processing submap with {len(self.image_names_subset)} frames...")
                    self.process_submap()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()

    def process_submap(self):
        """Process a submap batch — run VGGT, optimize graph, extract data, detect objects."""
        # Best-effort "busy" signal for the offline harness's wait_until_idle(). Guarded with
        # getattr so process_submap still runs on a minimally-constructed test double (several
        # unit tests build StreamingSLAM via __new__ without __init__).
        _processing = getattr(self, "_processing", None)
        if _processing is not None:
            _processing.set()
        try:
            process_start = time.perf_counter()
            io_time = sum(self.image_io_times_subset)
            vggt_start = self.solver.vggt_timer.total_time
            retrieval_start = self.solver.loop_closure_timer.total_time
            clip_start = self.solver.clip_timer.total_time

            # Guard: skip if any saved images are missing (e.g. after a reset mid-batch)
            valid_pairs = [
                (p, io_t)
                for p, io_t in zip(self.image_names_subset, self.image_io_times_subset)
                if os.path.exists(p)
            ]
            valid_names = [p for p, _ in valid_pairs]
            if not valid_names:
                print("process_submap: all image files missing, skipping")
                self.image_names_subset = []
                self.image_io_times_subset = []
                return
            self.image_names_subset = valid_names
            self.image_io_times_subset = [io_t for _, io_t in valid_pairs]
            io_time = sum(self.image_io_times_subset)

            # 1. Run predictions (uses stored clip model via solver)
            predictions = self.solver.run_predictions(
                self.image_names_subset,
                self.model,
                self.max_loops,
            )

            # 2. Add points to map
            solver_start = time.perf_counter()
            self.solver.add_points(predictions)

            # 3. Optimize graph
            self.solver.graph.optimize()
            solver_time = time.perf_counter() - solver_start

            # 4. Check for loop closures
            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if loop_closure_detected:
                print("Loop closure detected!")

            # 5. Extract data for streaming — incremental on normal updates, full on loop closure
            extract_start = time.perf_counter()
            if loop_closure_detected:
                stream_data = self.extract_stream_data_full()
            else:
                latest = self.solver.map.get_latest_submap()
                stream_data = self.extract_stream_data_incremental(latest.get_id())
            self._last_stream_data = stream_data
            extract_time = time.perf_counter() - extract_start

            # 6. Run object detection if queries are active
            detect_time = 0.0
            detect_breakdown = {"sam_ms": 0.0, "bbox_ms": 0.0, "recompute_ms": 0.0, "dedup_ms": 0.0}
            with self._detection_lock:
                has_queries = len(self.active_queries) > 0
            if has_queries:
                detect_start = time.perf_counter()
                detect_breakdown = self._detect_after_submap_update(loop_closure_detected=loop_closure_detected)
                detect_time = time.perf_counter() - detect_start
                with self._detection_lock:
                    n_det = len(self.accumulated_detections)
                print(f"Detection: {n_det} detections in {detect_time*1000:.0f}ms")

            # Add detections to stream data
            with self._detection_lock:
                stream_data['detections'] = list(self.accumulated_detections)
                stream_data['active_queries'] = list(self.active_queries)

            # 7. Send to result queue
            if stream_data and self.result_queue is not None and not self.result_queue.full():
                stream_data['_put_time'] = time.perf_counter()
                self.result_queue.put(stream_data)
                if self.event_loop is not None and self.result_ready_event is not None:
                    self.event_loop.call_soon_threadsafe(self.result_ready_event.set)

            # 7b. Trigger spatial agent analysis in background
            if self.spatial_agent is not None:
                current_submap_id = self.solver.map.get_num_submaps() - 1
                threading.Thread(
                    target=self.spatial_agent.on_submap_processed,
                    args=(current_submap_id,),
                    daemon=True,
                ).start()

            # 8. Keep overlapping frames
            self.image_names_subset = self.image_names_subset[-self.overlapping_window_size:]
            self.image_io_times_subset = self.image_io_times_subset[-self.overlapping_window_size:]

            print(f"Submap processed. Total submaps: {self.solver.map.get_num_submaps()}, "
                  f"Loop closures: {self.solver.graph.get_num_loops()}")
            vggt_time = self.solver.vggt_timer.total_time - vggt_start
            retrieval_time = self.solver.loop_closure_timer.total_time - retrieval_start
            clip_time = self.solver.clip_timer.total_time - clip_start
            total_time = io_time + (time.perf_counter() - process_start)
            gpu_time = vggt_time + retrieval_time + clip_time + detect_time
            gpu_busy_pct = min(100.0, (gpu_time / total_time) * 100.0) if total_time > 0 else 0.0
            submap_id = self.solver.map.get_largest_key(ignore_loop_closure_submaps=True)
            print(
                f"[latency] submap={submap_id} "
                f"io={io_time*1000:.0f}ms "
                f"vggt={vggt_time*1000:.0f}ms "
                f"retrieval={retrieval_time*1000:.0f}ms "
                f"clip={clip_time*1000:.0f}ms "
                f"solver={solver_time*1000:.0f}ms "
                f"extract={extract_time*1000:.0f}ms "
                f"detect={detect_time*1000:.0f}ms "
                f"(sam={detect_breakdown['sam_ms']:.0f}ms "
                f"bbox={detect_breakdown['bbox_ms']:.0f}ms "
                f"recompute={detect_breakdown['recompute_ms']:.0f}ms "
                f"dedup={detect_breakdown['dedup_ms']:.0f}ms) "
                f"total={total_time*1000:.0f}ms "
                f"gpu_busy_pct={gpu_busy_pct:.1f}"
            )

        except Exception as e:
            print(f"Submap processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if _processing is not None:
                _processing.clear()

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_one_submap(self, submap) -> dict:
        """Extract points, colors, and camera poses from a single submap."""
        pts = submap.get_points_in_world_frame(self.solver.graph)
        cols = submap.get_points_colors()
        poses = submap.get_all_poses_world(self.solver.graph)

        if self.vis_stride > 1 and pts is not None and len(pts) > 0:
            pts = pts[::self.vis_stride]
            cols = cols[::self.vis_stride]

        if cols is not None and len(cols) > 0 and cols.max() > 1.0:
            cols = cols / 255.0

        cam_positions = [p[:3, 3].tolist() for p in poses]
        cam_rotations = [p[:3, :3].tolist() for p in poses]

        return {
            'submap_id': submap.get_id(),
            'points': pts,
            'colors': cols,
            'cam_positions': cam_positions,
            'cam_rotations': cam_rotations,
        }

    def gather_world_point_cloud(self):
        """Aggregate the **complete** world-frame point cloud across all submaps for
        durable persistence — the full-fidelity *artifact* (e.g. for evidence/export),
        not the display view. Uses the graph-aware accessors (same pairing as
        ``_extract_one_submap`` / ``Map.write_points_to_file``) so the cloud is in the
        SLAM **world frame**, aligned with the world-frame object centers in
        ``SceneFacts`` so Q&A focus targets land correctly.

        Deliberately lossless w.r.t. the reconstruction: **no stride, no voxel, no
        cap** — ``float32`` positions and the native ``uint8`` colors are exactly what
        VGGT produced. Delegates to ``server.export.clouds.gather_full_cloud`` (the
        same gather the dataset-export path uses, so both stay in lock-step), which
        excludes loop-closure re-observation submaps and each non-first submap's
        carried-over overlap frame(s) — both were previously double-counted here (the
        window-quality study, ``experiments/research/2026-07-03-omega-full-quality.md``
        §2/§6 in the platform repo, measured ~2x duplicated points in stitched clouds).
        Display downsampling happens later, on read, in the broker route. Returns
        ``(positions (N,3) float32, colors (N,3) uint8)`` in the world frame
        (un-recentered)."""
        from server.export.clouds import gather_full_cloud
        return gather_full_cloud(self.solver, overlap_frames=self.overlapping_window_size)

    def extract_stream_data_full(self) -> dict:
        """Re-extract all submaps. Called on loop closure or get_global_map."""
        try:
            num_submaps = self.solver.map.get_num_submaps()
            if num_submaps == 0:
                self._submap_cache.clear()
                return self._empty_data()

            self._submap_cache.clear()
            for submap in self.solver.map.get_submaps():
                entry = self._extract_one_submap(submap)
                self._submap_cache[entry['submap_id']] = entry

            return self._build_full_payload()

        except Exception as e:
            print(f"Extract data error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_data()

    def extract_stream_data_incremental(self, new_submap_id: int) -> dict:
        """Extract only the new submap. O(1) submap extraction."""
        try:
            submap = self.solver.map.get_submap(new_submap_id)
            entry = self._extract_one_submap(submap)
            self._submap_cache[new_submap_id] = entry
            return self._build_incremental_payload(entry)

        except Exception as e:
            print(f"Incremental extract error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_data()

    def extract_stream_data(self):
        """Backward-compatible alias — full extraction."""
        return self.extract_stream_data_full()

    def _build_full_payload(self) -> dict:
        """Build a full payload from all cached submap entries."""
        all_pts, all_cols, all_cam_pos, all_cam_rot = [], [], [], []
        for e in self._submap_cache.values():
            if e['points'] is not None and len(e['points']) > 0:
                all_pts.append(e['points'])
                all_cols.append(e['colors'])
            all_cam_pos.extend(e['cam_positions'])
            all_cam_rot.extend(e['cam_rotations'])

        if all_pts:
            pts = np.vstack(all_pts)
            cols = np.vstack(all_cols)
            center = np.mean(pts, axis=0)
            self._scene_center = center
            self.latest_scene_center = center
            pts = pts - center
        else:
            pts = np.zeros((0, 3))
            cols = np.zeros((0, 3))
            center = np.zeros(3)

        cam_arr = np.array(all_cam_pos) - center if all_cam_pos else np.array([])

        # Resolve pending beacons
        self._resolve_beacons(cam_arr)

        n_points = len(pts)
        n_cameras = len(cam_arr) if isinstance(cam_arr, np.ndarray) and cam_arr.ndim == 2 else 0

        pts_f32 = pts.astype(np.float32)
        cols_u8 = (cols * 255).clip(0, 255).astype(np.uint8)
        points_b64 = base64.b64encode(pts_f32.tobytes()).decode('ascii')
        colors_b64 = base64.b64encode(cols_u8.tobytes()).decode('ascii')

        return {
            'type': 'full',
            'frame_id': self.frame_count,
            'num_submaps': len(self._submap_cache),
            'num_loops': self.solver.graph.get_num_loops(),
            'points_b64': points_b64,
            'colors_b64': colors_b64,
            'points': [],
            'colors': [],
            'camera_positions': cam_arr.tolist() if n_cameras > 0 else [],
            'camera_rotations': all_cam_rot,
            'scene_center': center.tolist(),
            'n_points': n_points,
            'n_cameras': n_cameras,
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

    def _build_incremental_payload(self, entry: dict) -> dict:
        """Build an incremental payload for a single new submap."""
        pts, cols = entry['points'], entry['colors']
        if pts is not None and len(pts) > 0:
            # Use the current scene center for recentering (computed on last full extraction)
            pts_f32 = (pts - self._scene_center).astype(np.float32)
            cols_u8 = (cols * 255).clip(0, 255).astype(np.uint8)
            # Recenter camera positions too
            cam_pos = (np.array(entry['cam_positions']) - self._scene_center).tolist()
        else:
            pts_f32 = np.zeros((0, 3), dtype=np.float32)
            cols_u8 = np.zeros((0, 3), dtype=np.uint8)
            cam_pos = entry['cam_positions']

        points_b64 = base64.b64encode(pts_f32.tobytes()).decode('ascii')
        colors_b64 = base64.b64encode(cols_u8.tobytes()).decode('ascii')

        return {
            'type': 'incremental',
            'submap_id': entry['submap_id'],
            'frame_id': self.frame_count,
            'num_submaps': self.solver.map.get_num_submaps(),
            'num_loops': self.solver.graph.get_num_loops(),
            'points_b64': points_b64,
            'colors_b64': colors_b64,
            'points': [],
            'colors': [],
            'camera_positions': cam_pos,
            'camera_rotations': entry['cam_rotations'],
            'scene_center': self._scene_center.tolist(),
            'n_points': len(pts_f32),
            'n_cameras': len(entry['cam_positions']),
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

    def _resolve_beacons(self, all_cam_positions):
        """Resolve pending beacons to 3D positions using camera positions."""
        if not isinstance(all_cam_positions, np.ndarray) or all_cam_positions.ndim != 2:
            return
        if len(all_cam_positions) == 0 or len(self.pending_beacons) == 0:
            return

        n_cameras = len(all_cam_positions)
        still_pending = []
        for beacon in self.pending_beacons:
            cam_idx = min(beacon['frame_number'] - 1, n_cameras - 1)
            cam_idx = max(cam_idx, 0)
            if cam_idx < n_cameras:
                pos = all_cam_positions[cam_idx]
                resolved = {
                    'beacon_id': beacon['beacon_id'],
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'z': float(pos[2]),
                }
                self.resolved_beacons.append(resolved)
                print(f"Beacon {beacon['beacon_id']} resolved at camera {cam_idx}")
            else:
                still_pending.append(beacon)
        self.pending_beacons = still_pending

    def _empty_data(self):
        return {
            'frame_id': self.frame_count,
            'num_submaps': 0,
            'num_loops': 0,
            'points_b64': '',
            'colors_b64': '',
            'points': [],
            'colors': [],
            'camera_positions': [],
            'camera_rotations': [],
            'scene_center': [0, 0, 0],
            'n_points': 0,
            'n_cameras': 0,
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

    def _cache_get(self, key):
        with self._detection_lock:
            return self._sam_cache.get(key)

    def _cache_set(self, key, value):
        with self._detection_lock:
            self._sam_cache[key] = value

    def _cache_delete_removed_queries(self, removed_queries):
        if not removed_queries:
            return
        with self._detection_lock:
            for k in [k for k in self._sam_cache if k[2] in removed_queries]:
                del self._sam_cache[k]
        with self._description_lock:
            for k in [k for k in self._description_cache if k[2] in removed_queries]:
                del self._description_cache[k]

    def _cache_clear(self):
        with self._detection_lock:
            self._sam_cache = {}
        with self._description_lock:
            self._description_cache = {}

    def _cache_items_snapshot(self):
        with self._detection_lock:
            return list(self._sam_cache.items())

    # ------------------------------------------------------------------
    # Object detection (cache-aware)
    # ------------------------------------------------------------------

    def _sync_clip_model_for_queries(self):
        has_queries = len(self.active_queries) > 0
        if has_queries and self.solver.clip_model is None:
            self.solver.set_clip_model(
                self.object_detector.clip_model,
                self.object_detector.clip_preprocess,
            )
            self._backfill_semantic_vectors()
        elif not has_queries and self.solver.clip_model is not None:
            self.solver.clip_model = None
            self.solver.clip_preprocess = None

    def _backfill_semantic_vectors(self):
        for submap in self._sorted_submaps():
            clip_embs = submap.get_all_semantic_vectors()
            if clip_embs is not None and len(clip_embs) > 0:
                continue
            image_names = list(getattr(submap, "img_names", []))
            if not image_names:
                continue
            with self.solver.clip_timer:
                image_embs = compute_image_embeddings(
                    self.solver.clip_model,
                    self.solver.clip_preprocess,
                    image_names,
                    device=self.device,
                )
            submap.set_all_semantic_vectors(image_embs)

    def set_detection_queries(self, queries):
        """Set detection queries. Cache-aware: handles add/remove of queries."""
        with self._detection_lock:
            old_queries = set(self.active_queries)
            self.active_queries = [q.strip() for q in queries if q.strip()]
            new_queries = set(self.active_queries)
        self._sync_clip_model_for_queries()

        if len(self.active_queries) == 0:
            self._cache_clear()
            with self._detection_lock:
                self.accumulated_detections = []
            return

        # Purge cache for removed queries
        removed = old_queries - new_queries
        self._cache_delete_removed_queries(removed)

        # Run CLIP+SAM on all existing submaps for new queries
        added = new_queries - old_queries
        if added and self.solver.map.get_num_submaps() > 0:
            t0 = time.time()
            added_list = sorted(added)
            self._reconcile_detection_state(added_list, recompute_all_bboxes=True)
            print(f"New queries {added_list}: SAM on all submaps in {(time.time()-t0)*1000:.0f}ms")
        else:
            self._dedup_and_store()

    def run_detection_progressive(self, queries):
        """Generator: run CLIP+SAM submap-by-submap, yield partial detections."""
        with self._detection_lock:
            old_queries = set(self.active_queries)
            self.active_queries = [q.strip() for q in queries if q.strip()]
            new_queries = set(self.active_queries)
        self._sync_clip_model_for_queries()

        if not self.active_queries:
            self._cache_clear()
            with self._detection_lock:
                self.accumulated_detections = []
            yield {'detections': [], 'is_final': True}
            return

        # Purge cache for removed queries
        removed = old_queries - new_queries
        self._cache_delete_removed_queries(removed)

        added = sorted(new_queries - old_queries)
        all_submaps = self._sorted_submaps()

        if not added or not all_submaps:
            self._dedup_and_store()
            with self._detection_lock:
                yield {'detections': list(self.accumulated_detections), 'is_final': True}
            return

        for i, submap in enumerate(all_submaps):
            self._reconcile_detection_state(
                added,
                submaps=[submap],
                recompute_all_bboxes=False,
            )
            with self._detection_lock:
                yield {
                    'detections': list(self.accumulated_detections),
                    'is_final': i == len(all_submaps) - 1,
                }

    def remove_query(self, query: str):
        """Remove a single query, purge its cache entries, rebuild detections without rerunning."""
        with self._detection_lock:
            self.active_queries = [q for q in self.active_queries if q != query]
        self._sync_clip_model_for_queries()
        self._cache_delete_removed_queries({query})
        self._dedup_and_store()

    def add_query_progressive(self, query: str):
        """Add a single query and scan submaps for it progressively, yielding partial results.

        Unlike run_detection_progressive, this method only adds/scans the given query without
        touching any other active queries — safe to call concurrently for different queries.
        """
        with self._detection_lock:
            if query not in self.active_queries:
                self.active_queries.append(query)
        self._sync_clip_model_for_queries()
        all_submaps = self._sorted_submaps()
        if not all_submaps:
            self._dedup_and_store()
            with self._detection_lock:
                yield {'detections': list(self.accumulated_detections), 'is_final': True}
            return
        for i, submap in enumerate(all_submaps):
            self._reconcile_detection_state([query], submaps=[submap], recompute_all_bboxes=False)
            with self._detection_lock:
                yield {
                    'detections': list(self.accumulated_detections),
                    'is_final': i == len(all_submaps) - 1,
                }

    # Maximum number of frames per submap to run SAM on, chosen by CLIP similarity rank.
    # Only the top-K most semantically matching frames are segmented, so SAM focuses
    # on clear, well-composed views rather than every frame that barely clears the threshold.
    # Class-level default; __init__ shadows this per-instance with DETECTION_FRAMES_PER_WINDOW
    # (env-configurable — more frames scanned per window means more grounding chances for the
    # detector, at the cost of more SAM calls per submap).
    SAM_TOP_K_FRAMES = 1

    # Persist the per-frame SAM mask (as a compact RLE) alongside the 2D box on each exported
    # detection so the dynamics producer can seed BootsTAPIR directly (see
    # ``contracts/export-format.md``). The 2D box is always carried (tiny, high-value); the mask
    # RLE is larger, so this gate lets an operator drop it if the live-broadcast payload matters
    # (detections ride ``stream_data['detections']`` every SLAM update). Default on: persist it.
    EXPORT_PERSIST_MASK_RLE = True

    def _run_clip_sam_on_submap(self, submap, queries):
        """Run CLIP matching + SAM on unprocessed (submap, frame, query) combos.

        Uses submap.get_all_semantic_vectors() for CLIP embeddings (VGGT-SLAM 2.0 API).
        Only the top SAM_TOP_K_FRAMES frames by CLIP similarity are segmented, which
        focuses SAM on the clearest, most semantically relevant views of the object.
        Returns list of cache keys that got new mask entries.
        """
        od = self.object_detector
        submap_id = submap.get_id()

        # VGGT-SLAM 2.0 API: get_all_semantic_vectors()
        clip_embs = submap.get_all_semantic_vectors()
        if clip_embs is None or len(clip_embs) == 0:
            return []

        # Convert to tensor if numpy
        if isinstance(clip_embs, np.ndarray):
            clip_embs = torch.from_numpy(clip_embs)

        new_mask_keys = []

        for query in queries:
            query = query.strip()
            if not query:
                continue

            ct = self.detection_clip_thresholds
            clip_thresh = ct.get(query, ct.get("default", 0.2))
            st = self.detection_sam_thresholds
            sam_thresh = st.get(query, st.get("default", 0.0))

            text_emb = od.encode_text_vector(query)
            sims = clip_embs @ text_emb  # (S,)

            last_orig = submap.get_last_non_loop_frame_index()
            if last_orig is None or last_orig < 0:
                last_orig = sims.shape[0] - 1

            # Mark all frames as processed (empty) first, then overwrite for top-K
            candidate_frames = []
            for frame_idx in range(last_orig + 1):
                cache_key = (submap_id, frame_idx, query)
                if self._cache_get(cache_key) is not None:
                    continue
                sim_val = sims[frame_idx].item()
                if sim_val < clip_thresh:
                    self._cache_set(cache_key, [])
                else:
                    candidate_frames.append((sim_val, frame_idx))

            # Sort candidates by CLIP similarity descending; only run SAM on top-K
            candidate_frames.sort(key=lambda x: x[0], reverse=True)
            top_frames = candidate_frames[:self.SAM_TOP_K_FRAMES]
            skipped_frames = candidate_frames[self.SAM_TOP_K_FRAMES:]

            # Mark frames outside top-K as empty so they aren't reconsidered
            for _, frame_idx in skipped_frames:
                self._cache_set((submap_id, frame_idx, query), [])

            for sim_val, frame_idx in top_frames:
                cache_key = (submap_id, frame_idx, query)
                try:
                    frame_tensor = submap.get_frame_at_index(frame_idx)
                    frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)

                    seg_results = od.segment_all(frame_pil, query)
                    passed = []
                    for mask_2d, box_2d, seg_score in seg_results:
                        if seg_score >= sam_thresh:
                            passed.append({
                                'mask_2d': mask_2d,
                                'box_2d': box_2d,
                                'seg_score': float(seg_score),
                                'clip_score': float(sim_val),
                                'bbox_3d': None,
                            })
                    self._cache_set(cache_key, passed)
                    if passed:
                        new_mask_keys.append(cache_key)
                except Exception as e:
                    print(f"  SAM error submap {submap_id} frame {frame_idx} query '{query}': {e}")
                    self._cache_set(cache_key, [])

        return new_mask_keys

    def _compute_bboxes_for_keys(self, cache_keys):
        """Compute 3D bounding boxes for cached mask entries."""
        od = self.object_detector
        scene_center = self.latest_scene_center
        for key in cache_keys:
            submap_id, frame_idx, _query = key
            masks = self._cache_get(key) or []
            if not masks:
                continue
            submap = self.solver.map.get_submap(submap_id)
            if submap is None:
                continue
            for entry in masks:
                entry['bbox_3d'] = od.compute_3d_bbox(
                    submap, frame_idx, entry['mask_2d'],
                    self.solver.graph, scene_center
                )
            self._cache_set(key, masks)

    def _sorted_submaps(self, submaps=None):
        """Return submaps in deterministic key order for consistent reconciliation."""
        if submaps is None:
            items = list(self.solver.map.get_submaps())
        else:
            items = list(submaps)
        return sorted(items, key=lambda s: s.get_id())

    def _reconcile_detection_state(self, queries, submaps=None, recompute_all_bboxes=True) -> dict[str, float]:
        """Apply CLIP+SAM for queries and refresh deduplicated detection state."""
        # Returned (not stored on self) so concurrent agent-thread callers don't race on shared state.
        breakdown = {"sam_ms": 0.0, "bbox_ms": 0.0, "recompute_ms": 0.0, "dedup_ms": 0.0}
        if not queries:
            t0 = time.perf_counter()
            self._dedup_and_store()
            breakdown["dedup_ms"] = (time.perf_counter() - t0) * 1000
            return breakdown

        new_mask_keys = []
        t0 = time.perf_counter()
        for submap in self._sorted_submaps(submaps):
            try:
                new_mask_keys.extend(self._run_clip_sam_on_submap(submap, queries))
            except Exception as e:
                print(f"  Detection error on submap {submap.get_id()}: {e}")
        breakdown["sam_ms"] = (time.perf_counter() - t0) * 1000

        if new_mask_keys:
            t0 = time.perf_counter()
            self._compute_bboxes_for_keys(new_mask_keys)
            breakdown["bbox_ms"] = (time.perf_counter() - t0) * 1000
        if recompute_all_bboxes:
            t0 = time.perf_counter()
            self._recompute_all_bboxes()
            breakdown["recompute_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._dedup_and_store()
        breakdown["dedup_ms"] = (time.perf_counter() - t0) * 1000
        return breakdown

    def _recompute_all_bboxes(self):
        """Recompute ALL 3D bboxes from cached SAM masks (after graph optimization)."""
        od = self.object_detector
        scene_center = self.latest_scene_center
        for key, masks in self._cache_items_snapshot():
            if not masks:
                continue
            submap_id, frame_idx, _query = key
            submap = self.solver.map.get_submap(submap_id)
            if submap is None:
                continue
            for entry in masks:
                entry['bbox_3d'] = od.compute_3d_bbox(
                    submap, frame_idx, entry['mask_2d'],
                    self.solver.graph, scene_center
                )
            self._cache_set(key, masks)

    @staticmethod
    def _detection_box_2d(box_2d):
        """Coerce a SAM ``box_2d`` (``[x0,y0,x1,y1]``) to plain floats, or ``None``. Guarded so a
        malformed box never breaks the detection dict that feeds the live stream + export."""
        if box_2d is None:
            return None
        try:
            box = [float(v) for v in box_2d]
        except (TypeError, ValueError):
            return None
        return box if len(box) == 4 else None

    def _detection_mask_rle(self, mask_2d):
        """Encode a SAM ``mask_2d`` as a compact RLE dict, or ``None``. Gated by
        ``EXPORT_PERSIST_MASK_RLE`` and fully guarded (encoding failures yield ``None`` so the
        live pipeline is never affected). See ``server/export/mask_rle.py``."""
        if not self.EXPORT_PERSIST_MASK_RLE or mask_2d is None:
            return None
        try:
            from server.export.mask_rle import mask_to_rle
            return mask_to_rle(mask_2d)
        except Exception:
            return None

    def _dedup_and_store(self):
        """Build detection list from cache, dedup, store.

        Confidence used for ranking is a combined CLIP × SAM score so that the
        deduplication prefers frames that are both semantically clear (high CLIP)
        and well-segmented (high SAM), not just whichever had the biggest blob.
        """
        raw = []
        for (submap_id, frame_idx, query), masks in self._cache_items_snapshot():
            for entry in masks:
                bbox = entry.get('bbox_3d')
                if bbox is None:
                    continue
                clip_score = entry.get('clip_score', 1.0)
                seg_score = entry['seg_score']
                combined = clip_score * seg_score
                raw.append({
                    "success": True,
                    "query": query,
                    "bounding_box": bbox,
                    "confidence": combined,
                    "keyframe_image": None,
                    "mask_image": None,
                    # Real per-frame SAM detection geometry (additive, optional): the 2D pixel box
                    # is always carried; the mask RLE is gated (EXPORT_PERSIST_MASK_RLE). Both were
                    # previously dropped -- persisting them lets the dynamics producer seed
                    # BootsTAPIR from the real detection. See contracts/export-format.md. These
                    # ride the deduped detection dict, which is also socket-broadcast (§stream);
                    # keep them optional so web/protocol consumers that ignore them are unaffected.
                    "box_2d": self._detection_box_2d(entry.get('box_2d')),
                    "mask_rle": self._detection_mask_rle(entry.get('mask_2d')),
                    "matched_submap": int(submap_id),
                    "matched_frame": int(frame_idx),
                    "query_time_ms": 0,
                    "error": None,
                    # Fine-grained label, if this object was already enriched (click / top-N).
                    "description": self.get_cached_description(submap_id, frame_idx, query),
                })
        deduped = ObjectDetector.deduplicate_detections(raw)
        with self._detection_lock:
            self.accumulated_detections = deduped

    def _detect_after_submap_update(self, loop_closure_detected: bool = False) -> dict[str, float]:
        """Run detection after a new submap is added."""
        with self._detection_lock:
            queries = list(self.active_queries)
        if not queries:
            return {"sam_ms": 0.0, "bbox_ms": 0.0, "recompute_ms": 0.0, "dedup_ms": 0.0}
        # Pose graph only shifts old submaps on loop closure; skip global bbox recompute otherwise.
        return self._reconcile_detection_state(queries, recompute_all_bboxes=loop_closure_detected)

    def finalize_detection_state(self):
        """Force a final all-submap reconciliation for active queries."""
        with self._detection_lock:
            queries = list(self.active_queries)
        if not queries:
            self._dedup_and_store()
            return []
        return self._reconcile_detection_state(queries, recompute_all_bboxes=True)

    # ------------------------------------------------------------------
    # Fine-grained object enrichment (VLM captioner, cache-aware)
    # ------------------------------------------------------------------

    def set_object_enricher(self, enricher):
        """Inject the ObjectEnricher (mirrors the app-level configure_* setters)."""
        self.object_enricher = enricher

    def get_cached_description(self, submap_id, frame_idx, query):
        """Return the cached ObjectDescription dict for a detection, or None."""
        key = (int(submap_id), int(frame_idx), str(query).strip())
        with self._description_lock:
            return self._description_cache.get(key)

    def _set_cached_description(self, key, desc_dict):
        with self._description_lock:
            self._description_cache[key] = desc_dict

    def _best_box_for(self, submap_id, frame_idx, query):
        """Highest CLIP×SAM box_2d among cached SAM masks for this (submap, frame, query)."""
        entries = self._cache_get((int(submap_id), int(frame_idx), str(query).strip())) or []
        best, best_score = None, -1.0
        for e in entries:
            if e.get("box_2d") is None:
                continue
            score = float(e.get("seg_score", 0.0)) * float(e.get("clip_score", 1.0))
            if score > best_score:
                best, best_score = e.get("box_2d"), score
        return best

    def make_detection_crop(self, submap_id, frame_idx, query):
        """Return (crop_b64, crop_bytes) for the detected object, or (None, None).

        Reads the keyframe tensor + the cached SAM box and crops tight around the object
        (falling back to the full frame if no box). This touches GPU tensors, so call it on
        the GPU executor; the LLM/network step (enrich_from_crop) runs off it.
        """
        try:
            submap = self.solver.map.get_submap(int(submap_id))
            if submap is None:
                return None, None
            frame_tensor = submap.get_frame_at_index(int(frame_idx))
            frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            box = self._best_box_for(submap_id, frame_idx, query)
            if box is not None:
                from server.scene_report.object_enricher import crop_from_box
                crop = crop_from_box(frame_np, box)
            else:
                crop = frame_np
            img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, None
            crop_bytes = buf.tobytes()
            return base64.b64encode(crop_bytes).decode("utf-8"), crop_bytes
        except Exception as e:
            print(f"make_detection_crop error (submap {submap_id} frame {frame_idx}): {e}")
            return None, None

    def enrich_from_crop(self, submap_id, frame_idx, query, crop_b64, crop_url=None):
        """Run the enricher on a precomputed crop and cache the result.

        Pure LLM/network — call off the GPU executor. Returns the ObjectDescription dict,
        or None if there is no enricher / no crop. Cached re-clicks are free.
        """
        if self.object_enricher is None or not crop_b64:
            return None
        key = (int(submap_id), int(frame_idx), str(query).strip())
        with self._description_lock:
            cached = self._description_cache.get(key)
        if cached is not None:
            return cached
        try:
            desc = self.object_enricher.describe(crop_b64, key[2], crop_url=crop_url)
            desc_dict = desc.model_dump()
        except Exception as e:
            print(f"enrich_from_crop error: {e}")
            return None
        self._set_cached_description(key, desc_dict)
        return desc_dict

    def enrich_detection(self, submap_id, frame_idx, query, crop_url=None):
        """Crop + enrich in one call (used by the finalize top-N pass, where the GPU loop
        is idle). Cache-checked; returns the ObjectDescription dict or None."""
        if self.object_enricher is None:
            return None
        cached = self.get_cached_description(submap_id, frame_idx, query)
        if cached is not None:
            return cached
        crop_b64, _ = self.make_detection_crop(submap_id, frame_idx, query)
        if not crop_b64:
            return None
        return self.enrich_from_crop(submap_id, frame_idx, query, crop_b64, crop_url=crop_url)

    def enrich_top_n(self, detections, n):
        """Fill ``det['description']`` for the top-N deduped detections by confidence.

        Best-effort and idempotent (skips already-described / cached objects). Safe to call
        at finalize on the agent executor; mutates the passed detection dicts in place.
        """
        if self.object_enricher is None or not detections or n <= 0:
            return
        ranked = sorted(
            (d for d in detections if d.get("success") and d.get("bounding_box")),
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )[:n]
        for det in ranked:
            if det.get("description"):
                continue
            desc = self.enrich_detection(
                det.get("matched_submap", -1),
                det.get("matched_frame", -1),
                det.get("query", ""),
            )
            if desc:
                det["description"] = desc

    # ------------------------------------------------------------------
    # Debug detection — full pipeline with rich per-frame diagnostics
    # ------------------------------------------------------------------

    def debug_detect_full(self, queries, clip_thresholds=None, sam_thresholds=None, top_k=None, include_frames=True):
        """Run the full detection pipeline and return rich per-frame diagnostics.

        Does NOT touch _sam_cache or accumulated_detections — purely diagnostic.
        Respects top_k frame selection and combined CLIP×SAM ranking exactly as
        production does, so the debug page reflects what production would pick.

        Returns a dict matching the DebugDetectResponse type expected by the frontend.
        """
        import time as _time
        t0 = _time.time()

        od = self.object_detector
        clip_thresh_map = clip_thresholds or {}
        sam_thresh_map = sam_thresholds or {}
        effective_top_k = top_k if (top_k is not None and top_k > 0) else self.SAM_TOP_K_FRAMES

        all_frames_diag = []
        total_frame_count = 0
        # Maps (submap_id, frame_idx, query) -> mask diag list for dedup
        key_to_masks = {}

        all_submaps = list(self.solver.map.get_submaps())
        if not all_submaps:
            return {
                'queries': queries, 'clip_thresholds': clip_thresh_map,
                'sam_thresholds': sam_thresh_map, 'top_k': effective_top_k,
                'frames': [], 'raw_detection_count': 0,
                'deduped_detection_count': 0, 'detections': [],
                'total_frames_scanned': 0, 'query_time_ms': 0,
            }

        for submap in all_submaps:
            submap_id = submap.get_id()
            clip_embs = submap.get_all_semantic_vectors()
            if clip_embs is None or len(clip_embs) == 0:
                continue
            if isinstance(clip_embs, np.ndarray):
                clip_embs = torch.from_numpy(clip_embs)

            last_orig = submap.get_last_non_loop_frame_index()
            if last_orig is None or last_orig < 0:
                last_orig = clip_embs.shape[0] - 1

            for query in queries:
                query = query.strip()
                if not query:
                    continue

                clip_thresh = clip_thresh_map.get(query, clip_thresh_map.get('default', 0.2))
                sam_thresh = sam_thresh_map.get(query, sam_thresh_map.get('default', 0.3))

                text_emb = od.encode_text_vector(query)
                sims = clip_embs @ text_emb  # (S,)

                # Rank all frames by CLIP similarity
                scored = []
                for frame_idx in range(last_orig + 1):
                    scored.append((sims[frame_idx].item(), frame_idx))
                scored.sort(key=lambda x: x[0], reverse=True)

                candidates_above = [(s, fi) for s, fi in scored if s >= clip_thresh]
                top_k_set = {fi for _, fi in candidates_above[:effective_top_k]}
                clip_rank_map = {fi: rank + 1 for rank, (_, fi) in enumerate(scored)}

                for frame_idx in range(last_orig + 1):
                    total_frame_count += 1
                    sim_val = sims[frame_idx].item()
                    above = sim_val >= clip_thresh
                    in_top_k = frame_idx in top_k_set
                    rank = clip_rank_map.get(frame_idx, frame_idx + 1)

                    # Thumbnail for all frames (cheap)
                    thumbnail = None
                    resolution = None
                    if include_frames:
                        try:
                            frame_tensor = submap.get_frame_at_index(frame_idx)
                            frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            h, w = frame_np.shape[:2]
                            resolution = f"{w}×{h}"
                            thumb_h = 120
                            thumb_w = int(w * thumb_h / h)
                            thumb = cv2.resize(frame_np, (thumb_w, thumb_h))
                            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                            _, buf = cv2.imencode('.jpg', thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            thumbnail = base64.b64encode(buf.tobytes()).decode('ascii')
                        except Exception:
                            pass

                    sam_masks_diag = []
                    sam_error = None
                    sam_skipped = above and not in_top_k

                    if in_top_k and above:
                        try:
                            frame_tensor = submap.get_frame_at_index(frame_idx)
                            frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            frame_pil = Image.fromarray(frame_np)
                            seg_results = od.segment_all(frame_pil, query)

                            for mask_2d, box_2d, seg_score in seg_results:
                                above_sam = float(seg_score) >= sam_thresh
                                combined = sim_val * float(seg_score)
                                bbox_3d = None
                                has_3d_box = False
                                if above_sam:
                                    try:
                                        bbox_3d = od.compute_3d_bbox(
                                            submap, frame_idx, mask_2d,
                                            self.solver.graph, self.latest_scene_center
                                        )
                                        has_3d_box = bbox_3d is not None
                                    except Exception:
                                        pass

                                mask_image = None
                                if include_frames:
                                    try:
                                        mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, mask_2d)
                                    except Exception:
                                        pass

                                mask_entry = {
                                    'score': float(seg_score),
                                    'clip_score': float(sim_val),
                                    'combined_score': combined,
                                    'box_2d': [float(v) for v in box_2d],
                                    'mask_image': mask_image,
                                    'above_sam_threshold': above_sam,
                                    'sam_threshold_used': sam_thresh,
                                    'has_3d_box': has_3d_box,
                                    'bbox_3d': bbox_3d,
                                    'dedup_kept': None,
                                }
                                sam_masks_diag.append(mask_entry)
                                if above_sam and has_3d_box:
                                    key_to_masks.setdefault((submap_id, frame_idx, query), []).append(mask_entry)
                        except Exception as e:
                            sam_error = str(e)

                    if include_frames:
                        all_frames_diag.append({
                            'submap_id': submap_id,
                            'frame_idx': frame_idx,
                            'query': query,
                            'clip_similarity': float(sim_val),
                            'clip_rank': rank,
                            'above_threshold': above,
                            'in_top_k': in_top_k,
                            'sam_skipped': sam_skipped,
                            'clip_threshold_used': clip_thresh,
                            'sam_threshold_used': sam_thresh,
                            'top_k_used': effective_top_k,
                            'thumbnail': thumbnail,
                            'resolution': resolution,
                            'sam_masks': sam_masks_diag,
                            'sam_error': sam_error,
                            'detections_before_dedup': [],
                        })

        # Build raw detections and deduplicate using combined score
        raw_detections = []
        for (submap_id, frame_idx, query), masks in key_to_masks.items():
            for entry in masks:
                raw_detections.append({
                    'success': True,
                    'query': query,
                    'bounding_box': entry['bbox_3d'],
                    'confidence': entry['combined_score'],
                    'matched_submap': int(submap_id),
                    'matched_frame': int(frame_idx),
                    'clip_score': entry['clip_score'],
                    'sam_score': entry['score'],
                })

        deduped = ObjectDetector.deduplicate_detections(raw_detections)
        kept_keys = {(d['matched_submap'], d['matched_frame'], d['query']) for d in deduped}

        # Mark dedup_kept on mask entries
        for (submap_id, frame_idx, query), masks in key_to_masks.items():
            kept = (submap_id, frame_idx, query) in kept_keys
            for m in masks:
                if m['has_3d_box']:
                    m['dedup_kept'] = kept

        # Populate detections_before_dedup on frame diag entries
        if include_frames:
            raw_by_key = {}
            for r in raw_detections:
                raw_by_key.setdefault((r['matched_submap'], r['matched_frame'], r['query']), []).append(r)
            for fd in all_frames_diag:
                k = (fd['submap_id'], fd['frame_idx'], fd['query'])
                fd['detections_before_dedup'] = raw_by_key.get(k, [])

        elapsed_ms = int((_time.time() - t0) * 1000)
        return {
            'queries': queries,
            'clip_thresholds': clip_thresh_map,
            'sam_thresholds': sam_thresh_map,
            'top_k': effective_top_k,
            'frames': all_frames_diag if include_frames else [],
            'raw_detection_count': len(raw_detections),
            'deduped_detection_count': len(deduped),
            'detections': deduped,
            'total_frames_scanned': total_frame_count,
            'query_time_ms': elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def soft_reset(self):
        """Reset SLAM state without reloading models."""
        print("Performing soft reset...")
        was_running = self.is_running
        self.stop()

        self.frame_count = 0
        self.image_names_subset.clear()
        self.image_io_times_subset.clear()
        self.pending_beacons.clear()
        self.resolved_beacons.clear()
        self.latest_scene_center = np.zeros(3)

        with self._detection_lock:
            self.active_queries = []
            self.accumulated_detections = []
        self._cache_clear()
        self._last_stream_data = None
        self._submap_cache.clear()
        self._scene_center = np.zeros(3)

        self.solver.reset()
        self.solver.clip_model = None
        self.solver.clip_preprocess = None

        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.temp_dir = tempfile.mkdtemp()

        # Reset spatial agent if present
        if self.spatial_agent is not None:
            self.spatial_agent.reset()

        print(f"Soft reset complete. New temp dir: {self.temp_dir}")

        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_frame(self, frame):
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
