"""Offline multi-video reconstruction harness — two grouping models for pilot deliveries.

Both modes feed walkthrough clips into the streaming SLAM + finalize path offline, persist
durable scans (full cloud + optional 3DGS ``splat.ply``), and tag them with
``client``/``project`` for embed delivery. They differ in how the clips are grouped:

* ``--mode single`` (default, backward-compatible): feed EVERY video into ONE live
  ``StreamingSLAM`` session and finalize **once** after the last clip. Chaining clips
  through one ``Solver`` accumulates submaps and lets loop closure stitch them into a single
  world frame → ONE persisted scan. Use this when the clips are one *continuous / overlapping*
  capture of one space (the operator then mints a per-scene share token and embeds it).

* ``--mode per-scene``: reconstruct EACH video as its OWN scan — one ``finalize`` + persist
  per clip, each with its own ``scan_id``, all sharing the ``client``/``project`` tags. Use
  this when the clips are *distinct, non-overlapping locations* that CANNOT fuse into one
  world frame — e.g. a real-estate pilot's 5 clips are 5 different units
  + floors with no visual overlap. The result is a **building** of N scenes delivered as a
  tour: mint a building (project) token via ``POST /api/projects/share`` →
  ``building_embed_url`` → ``GET /api/projects/manifest`` (which hands out per-scene share
  tokens). See ``docs/pilot-reconstruction.md``.

Per-scene isolation reuses ONE ``StreamingSLAM`` session and calls the production
``soft_reset()`` path (``app.reset_for_next_scan()``) between clips. ``soft_reset()`` wipes
the world frame (``solver.reset()``) while keeping the VGGT model loaded, so N clips become
N isolated scans without paying the GPU model load per clip.

Run locally (CPU will be slow / toy-sized; real reconstruction needs a GPU)::

    # one continuous capture → ONE scan
    python -m server.reconstruct_pilot --mode single \
        --videos a.mp4 b.mp4 c.mp4 --user-id efficura-ops \
        --client efficura --project chorus --blob-root ./pilot_scans

    # 5 distinct units → 5 scans, one building
    python -m server.reconstruct_pilot --mode per-scene \
        --videos unit1.mp4 unit2.mp4 unit3.mp4 unit4.mp4 unit5.mp4 \
        --user-id efficura-ops --client efficura --project chorus \
        --scan-id chorus --blob-root ./pilot_scans

On Modal GPU, run the same module inside a worker image (the ``ModalScenePersistence``
gets a ``modal.Dict`` + Volume mount instead of the local dict/dir) — see
``docs/pilot-reconstruction.md``.

What is **NOT** validated here without a real GPU: the actual SLAM reconstruction, the
3DGS splat export/refinement, and (single mode) cross-video loop closure. The harness wiring
(sequential feed → finalize → persist with tags + splat; per-scene reset + one persist per
clip) is what this module owns and tests cover.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Optional

from server import app as app_module
from server.api_keys import ApiKeyRegistry


# Recall-tuned detection defaults for the OFFLINE PILOT PATH ONLY. Two independent GPU A/B
# captures (Finc, Chorus) validated that DETECTION_SAM_THRESHOLD=0.6 / DETECTION_CLIP_THRESHOLD=
# 0.10 / DETECTION_FRAMES_PER_WINDOW=3 grounds ~3x more of the LLM-discovered objects than the
# historical hardcoded values (0.80/0.15/1): Finc 10/12 vs 4/12, Chorus 6/12 vs 2/12, zero
# hallucinated junk, ~+30s wall time. See docs/pilot-reconstruction.md ("Detection defaults").
#
# server/streaming_slam.py's _detection_env_defaults() keeps ITS OWN fallback (0.15/0.80/1)
# unchanged -- that function also serves the LIVE streaming path (modal_streaming.py), which
# this pilot-only tuning must not touch. Instead, the pilot entrypoint seeds these as env
# defaults (via os.environ.setdefault, so an explicit operator override -- or one forwarded from
# modal_reconstruct_pilot.py's main() -- still wins) before app_module.initialize() constructs
# the StreamingSLAM that reads the env.
_PILOT_DETECTION_DEFAULTS = {
    "DETECTION_CLIP_THRESHOLD": "0.10",
    "DETECTION_SAM_THRESHOLD": "0.6",
    "DETECTION_FRAMES_PER_WINDOW": "3",
    # Offline batch can afford a patient report call: the live default (30s, read at
    # SceneReportBuilder construction via SCENE_REPORT_TIMEOUT_S in app.py) times out on
    # final-report calls carrying ~12 keyframe images + a raised SCENE_REPORT_MAX_TOKENS,
    # silently degrading the persisted report to the fact-only fallback.
    "SCENE_REPORT_TIMEOUT_S": "90",
}


def _apply_pilot_detection_defaults() -> None:
    """Seed the recall-tuned DETECTION_* env vars for this pilot run, without stomping an
    explicit override already present in the environment (``os.environ.setdefault`` is a no-op
    when the key is set). Split out from ``reconstruct_pilot`` so it's unit-testable, and called
    before ``app_module.initialize()`` -- see the module-level comment on
    ``_PILOT_DETECTION_DEFAULTS`` for why it must run before that call."""
    for key, value in _PILOT_DETECTION_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _persistence_for(blob_root: str):
    """A durable ``ModalScenePersistence`` backed by a plain dict (KV) + a local dir
    (blobs). In production ``modal_streaming.py`` injects a ``modal.Dict`` + Volume mount
    with ``commit``/``reload`` callables; here the local filesystem is the store so the
    harness runs standalone (and the persisted scan can be inspected on disk)."""
    from server.scene_report.store import ModalScenePersistence

    os.makedirs(blob_root, exist_ok=True)
    # Unlimited retention (default) — keep the full-fidelity pilot artifact forever.
    return ModalScenePersistence(store={}, blob_root=blob_root)


# ---------------------------------------------------------------------------
# Demo-build extensions (docs/demo-2026-07, W1) — flag-gated, default OFF, so the
# product pilot path is byte-identical unless --demo-index / --persist-trajectory
# is passed. Implemented as a persistence WRAPPER (the same interception seam
# modal_oreos_ingest.demo_recon_job uses to stamp ``source=``) so ``server/app.py``'s
# persist path needs no edits.
# ---------------------------------------------------------------------------


def build_frames_index(
    keyframes_b64: Optional[list],
    traj: Optional[dict],
    index_map: Optional[dict],
    scan_id: str,
    parent_artifact: Optional[str],
) -> dict:
    """The ``derived/demo/frames_index.json`` doc (features.md §0.3): one entry per
    persisted report keyframe, joining it back to its trajectory row + camera.

    Pure/testable: inputs are the ``save_scene`` ``keyframes_b64`` list, the
    ``{poses, intrinsics, source_frame_id}`` block from
    ``server.export.record.build_trajectory_arrays``, and
    ``server.export.trajectory.build_index_map``'s ``(submap_id, local_idx) → row``.

    Entries carry ``{blob_key, submap_id, frame_idx, traj_row, c2w, intrinsics}``;
    a keyframe with no trajectory row (loop-closure submap, or no trajectory at
    all) keeps its blob mapping with ``traj_row/c2w/intrinsics`` null — consumers
    (F3 prompt-view ladder b) fall back per-feature. The doc is wrapped in the
    WORLD-TRANSFORM-CONTRACT versioning envelope (``scan_id``/``parent_artifact``/
    ``created_at``/``run_id``); coordinates are WORLD-frame OpenCV c2w, per the
    same contract.
    """
    import numpy as np

    poses = intrinsics = None
    if traj is not None:
        poses = np.asarray(traj.get("poses"), dtype=np.float64).reshape(-1, 4, 4)
        intrinsics = np.asarray(traj.get("intrinsics"), dtype=np.float64).reshape(-1, 4)

    frames: list[dict] = []
    for kf in keyframes_b64 or []:
        if not isinstance(kf, dict) or not kf.get("image_b64"):
            continue  # the store skips these too — keep blob_key alignment exact
        sm = int(kf.get("submap_id", -1))
        fr = int(kf.get("frame_idx", -1))
        row = None
        if index_map is not None:
            row = index_map.get((sm, fr))
        entry: dict = {
            "blob_key": f"{sm}_{fr}.jpg",
            "submap_id": sm,
            "frame_idx": fr,
            "traj_row": None if row is None else int(row),
            "c2w": None,
            "intrinsics": None,
        }
        if row is not None and poses is not None and 0 <= int(row) < poses.shape[0]:
            entry["c2w"] = [[float(v) for v in r] for r in poses[int(row)]]
            entry["intrinsics"] = [float(v) for v in intrinsics[int(row)]]
        frames.append(entry)

    return {
        "version": 1,
        "scan_id": str(scan_id),
        "created_at": time.time(),
        "run_id": None,  # not an agent run; stamped for the shared doc envelope
        "parent_artifact": parent_artifact,
        "count": len(frames),
        "frames": frames,
    }


class _DemoExtensionsPersistence:
    """Wraps any scene persistence; on ``save_scene`` it (a) threads the live
    solver's ``{poses, intrinsics, source_frame_id}`` block in as ``trajectory=``
    (``--persist-trajectory`` — closes the pilot-path gap where scenes persisted
    ``trajectory_key=None``, starving LeRobot export + pose-derived up-vectors) and
    (b) writes ``derived/demo/frames_index.json`` (``--demo-index``). Everything
    else forwards untouched; both steps are best-effort and can never abort the
    scene persist."""

    def __init__(self, inner, *, demo_index: bool, persist_trajectory: bool):
        self._inner = inner
        self._demo_index = bool(demo_index)
        self._persist_trajectory = bool(persist_trajectory)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # -- solver access (patchable in tests) ---------------------------------

    @staticmethod
    def _live_solver():
        return getattr(app_module.slam_processor, "solver", None)

    def _build_trajectory_and_index(self):
        """``(traj, index_map)`` from the live solver via the Stage-4 export
        helpers (same ordered walk on both, so they can never disagree), or
        ``(None, None)`` when there is no solver / empty map / import failure."""
        try:
            solver = self._live_solver()
            if solver is None:
                return None, None
            from server.export.record import build_trajectory_arrays
            from server.export.trajectory import build_index_map

            traj = build_trajectory_arrays(solver)
            if traj["poses"].shape[0] == 0:
                return None, None
            return traj, build_index_map(solver)
        except Exception as exc:
            print(f"[pilot] demo trajectory build failed (degrading): {exc}")
            return None, None

    def save_scene(self, *args, **kwargs):
        user_id = args[0] if len(args) > 0 else kwargs.get("user_id")
        scan_id = args[1] if len(args) > 1 else kwargs.get("scan_id")

        traj, index_map = self._build_trajectory_and_index()
        if self._persist_trajectory and traj is not None and kwargs.get("trajectory") is None:
            kwargs["trajectory"] = traj

        result = self._inner.save_scene(*args, **kwargs)

        if self._demo_index and user_id is not None and scan_id is not None:
            try:
                import json

                parent = "trajectory.npz" if kwargs.get("trajectory") is not None else "cloud.npz"
                doc = build_frames_index(
                    kwargs.get("keyframes_b64"), traj, index_map, scan_id, parent
                )
                key = self._inner.save_derived_artifact(
                    user_id, scan_id,
                    "demo/frames_index.json",
                    json.dumps(doc).encode("utf-8"),
                )
                print(f"[pilot] wrote {key} ({doc['count']} keyframe entries)")
            except Exception as exc:  # never abort the persist over the index
                print(f"[pilot] frames-index write failed (degrading): {exc}")
        return result


def _wrap_demo_persistence(persistence, demo_index: bool, persist_trajectory: bool):
    """The identity when both flags are off (product path untouched), else the
    demo-extensions wrapper. Split out for unit tests."""
    if not (demo_index or persist_trajectory):
        return persistence
    return _DemoExtensionsPersistence(
        persistence, demo_index=demo_index, persist_trajectory=persist_trajectory
    )


def _feed_one_video(video_path: str, extract_fps: float, fast: bool) -> None:
    """Push one video's frames into the shared SLAM session and block until the clip is
    fully fed + the frame queue drains. Reuses ``VideoFeeder`` but with **no
    ``on_complete``**, so the per-video end-of-scan finalize never fires — finalize is
    deferred to one call after the last clip."""
    feeder = app_module.VideoFeeder(
        video_path,
        fast=fast,
        target_fps=extract_fps,
        on_complete=None,  # suppress per-video finalize; we finalize once at the end
        # Offline batch harness — no live camera/viewer to backpressure, so a full
        # frame_queue should block the feed rather than silently drop a keyframe.
        realtime=False,
    )
    feeder.start()
    # VideoFeeder._feed_loop runs on its own thread and auto-starts the SLAM loop on the
    # first frame; wait for that thread to finish pushing this clip's frames.
    if feeder._thread is not None:
        feeder._thread.join()
    # Let the SLAM loop catch up on this clip before the next one is fed, so submaps are
    # built in capture order (one continuous trajectory across clips).
    app_module._wait_for_frame_queue_drain()


def reconstruct_pilot(
    videos: list[str],
    user_id: str = "local",
    client: Optional[str] = None,
    project: Optional[str] = None,
    export_splat: bool = True,
    extract_fps: float = 2.0,
    fast: bool = True,
    blob_root: str = "./pilot_scans",
    submap_size: int = 32,
    scan_id: Optional[str] = None,
    mode: str = "single",
    persistence: Optional[object] = None,
    detect_objects: bool = True,
    labels: Optional[list[str]] = None,
    demo_index: bool = False,
    persist_trajectory: bool = False,
):
    """Drive the offline reconstruction end to end and return the persisted scan id(s).

    Shared setup (both modes): wire the worker runtime → user scoping, the durable store, the
    pilot tags + splat toggle, and bring up ONE ``StreamingSLAM`` session (loads the GPU model
    once). Then dispatch on ``mode``:

    * ``"single"`` → feed every video into the one session, finalize + persist **once**.
      Returns the single ``scan_id`` (str).
    * ``"per-scene"`` → reconstruct each video as its own scan (reset between clips), one
      persist per clip. Returns the list of ``scan_id``\\ s.

    ``labels`` (optional) is aligned to ``videos`` order: ``labels[i]`` is the human space name
    persisted with clip ``i`` (per-scene mode) or, in single mode, ``labels[0]`` names the one
    fused scan. Supplied by the captures manifest so the building tour shows real space names
    with no separate label step. Missing/short → those scans persist unlabeled (the old default).

    ``demo_index`` / ``persist_trajectory`` (demo build, default OFF → product path
    byte-identical): thread the live solver's per-keyframe trajectory into ``save_scene``
    and/or write ``derived/demo/frames_index.json`` after it — see
    ``_DemoExtensionsPersistence``.
    """
    # Seed the recall-tuned detection defaults for THIS pilot run (env-unset vars only; an
    # explicit override is left untouched). Safe to do before validation -- it only touches
    # os.environ, no SLAM/GPU state yet -- and must land before app_module.initialize() below
    # constructs the StreamingSLAM that reads these vars.
    _apply_pilot_detection_defaults()

    # Validate here (not only in CLI main()) so the programmatic entrypoint can't init SLAM
    # and return a scan id with nothing persisted.
    if not videos:
        raise SystemExit("no videos provided to reconstruct_pilot()")
    missing = [v for v in videos if not os.path.isfile(v)]
    if missing:
        raise SystemExit(f"Video file(s) not found: {', '.join(missing)}")
    if mode not in ("single", "per-scene"):
        raise SystemExit(f"unknown --mode {mode!r} (expected 'single' or 'per-scene')")
    if labels and len(labels) > len(videos):
        raise SystemExit(
            f"got {len(labels)} labels for {len(videos)} video(s) — labels align to videos order"
        )

    # The harness mutates app-global scan-id/persistence config + drives the shared SLAM
    # session, so two concurrent runs in one process would corrupt each other. Take the
    # non-reentrant pilot lock for the whole run (it's an offline batch tool — a busy lock
    # means a second harness run was started in the same process).
    if not app_module._pilot_lock.acquire(blocking=False):
        raise SystemExit("another pilot reconstruction is already running in this process")
    try:
        # Wire the worker runtime → user scoping, the durable store, and the pilot knobs
        # (tags + splat export) that _persist_scene_report threads into save_scene.
        app_module.configure_worker_runtime(user_id)
        # On Modal the entrypoint injects the SHARED ModalScenePersistence (the broker's
        # modal.Dict + Volume) so the scans this run produces are visible to the always-on
        # broker. Standalone/local runs fall back to a dict + local-dir store under blob_root.
        # Demo flags (default OFF) wrap it — never replace it — with the extensions shim.
        app_module.configure_scene_persistence(
            _wrap_demo_persistence(
                persistence or _persistence_for(blob_root), demo_index, persist_trajectory
            )
        )
        # Local-dev parity with the Modal wiring: a plain-dict API-key registry so any
        # ork_-bearer / /api/keys path exercised offline behaves like production. Not
        # durable on purpose — the harness is a batch tool; keys die with the process.
        app_module.configure_api_key_registry(ApiKeyRegistry({}))
        app_module.configure_pilot_persistence(
            client=client, project=project, export_splat=export_splat,
            detect_objects=detect_objects,
        )

        # Bring up the SLAM processor (no static frontend, no socket server). This loads the
        # GPU model — on a CPU-only box it will be slow/degraded; the reconstruction quality
        # is only meaningful on a GPU. One session serves every clip in both modes (per-scene
        # soft-resets it between clips rather than reloading the model).
        app_module.initialize(submap_size=submap_size)

        if mode == "per-scene":
            return _reconstruct_per_scene(
                videos, user_id, client, project, export_splat,
                extract_fps, fast, blob_root, scan_id, labels,
            )
        return _reconstruct_single(
            videos, user_id, client, project, export_splat,
            extract_fps, fast, blob_root, scan_id, labels,
        )
    finally:
        app_module._pilot_lock.release()


def _reconstruct_single(
    videos, user_id, client, project, export_splat, extract_fps, fast, blob_root, scan_id,
    labels=None,
) -> str:
    """``--mode single``: feed every clip into one world frame, finalize + persist once."""
    # One scan_id for the whole multi-video session, so all clips land in one record.
    scan_id = scan_id or uuid.uuid4().hex
    app_module._clear_scene_report()
    # Adopt this id for the scan the first fed frame auto-starts: the video feed path calls
    # _begin_new_scan(), which mints a random hash unless _pilot_scan_id is pre-set. Without
    # this, the scan persists under a hash and the get_scene(scan_id) check below false-warns.
    app_module._pilot_scan_id = scan_id
    app_module._current_scan_id = scan_id
    # The one fused scan takes the first label (if any); consumed at persist.
    app_module._pilot_label = labels[0] if labels else None

    print("=" * 60)
    print(f"Pilot reconstruction [single]: {len(videos)} video(s) → one scan {scan_id}")
    print(f"  user_id={user_id} client={client} project={project} "
          f"export_splat={export_splat} extract_fps={extract_fps}")
    print("=" * 60)

    t0 = time.time()
    for i, video in enumerate(videos, 1):
        print(f"[pilot] feeding video {i}/{len(videos)}: {video}")
        _feed_one_video(video, extract_fps=extract_fps, fast=fast)

    print(f"[pilot] all {len(videos)} videos fed in {time.time() - t0:.1f}s — finalizing once")
    app_module.finalize_scan_blocking(wait_for_drain=True)

    record = app_module._scene_persistence.get_scene(user_id, scan_id)
    if record is None:
        print("[pilot] WARNING: no scene persisted (empty reconstruction?)")
    else:
        print(f"[pilot] persisted scan {scan_id}: "
              f"points={record.get('point_count')} "
              f"has_splat={bool(record.get('splat_key'))} "
              f"client={record.get('client')} project={record.get('project')}")
        print(f"[pilot] blobs under: {blob_root}")
        print(f"[pilot] mint a share token: POST /api/scenes/{scan_id}/share (owner-authed)")
    return scan_id


def _reconstruct_per_scene(
    videos, user_id, client, project, export_splat, extract_fps, fast, blob_root, base_scan_id,
    labels=None,
) -> list[str]:
    """``--mode per-scene``: each clip → its OWN scan (one finalize + persist per clip), all
    sharing the ``client``/``project`` tags. The session is soft-reset between clips
    (``app.reset_for_next_scan()``) so each clip reconstructs in a clean world frame while the
    VGGT model stays loaded. Per-scene ids derive from ``--scan-id`` (``<base>-<n>``) or a
    fresh uuid per clip when unset. Returns the list of persisted scan ids (a building)."""
    print("=" * 60)
    print(f"Pilot reconstruction [per-scene]: {len(videos)} video(s) → {len(videos)} scan(s)")
    print(f"  user_id={user_id} client={client} project={project} "
          f"export_splat={export_splat} extract_fps={extract_fps}")
    print("=" * 60)

    scan_ids: list[str] = []
    t0 = time.time()
    for i, video in enumerate(videos):
        # Wipe the world frame between clips (keeps the model loaded). Skipped for the first
        # clip — initialize() already gave us a fresh session.
        if i > 0:
            app_module.reset_for_next_scan()
        scene_id = f"{base_scan_id}-{i + 1}" if base_scan_id else uuid.uuid4().hex
        app_module._clear_scene_report()
        # Adopt this id for the scan the first fed frame auto-starts (_begin_new_scan would
        # otherwise mint a random hash, so the clip would persist under the wrong id and the
        # get_scene(scene_id) check below would false-warn "persisted nothing").
        app_module._pilot_scan_id = scene_id
        app_module._current_scan_id = scene_id
        # Per-clip human label (aligned to videos order), consumed at persist → save_scene.
        label = labels[i] if labels and i < len(labels) else None
        app_module._pilot_label = label

        print(f"[pilot] scene {i + 1}/{len(videos)} → scan {scene_id}"
              f"{f' [{label}]' if label else ''}: {video}")
        _feed_one_video(video, extract_fps=extract_fps, fast=fast)
        app_module.finalize_scan_blocking(wait_for_drain=True)

        record = app_module._scene_persistence.get_scene(user_id, scene_id)
        if record is None:
            print(f"[pilot] WARNING: scene {i + 1} persisted nothing (empty reconstruction?)")
        else:
            print(f"[pilot] persisted scan {scene_id}: "
                  f"points={record.get('point_count')} "
                  f"has_splat={bool(record.get('splat_key'))} "
                  f"client={record.get('client')} project={record.get('project')}")
        scan_ids.append(scene_id)

    print(f"[pilot] all {len(videos)} scenes done in {time.time() - t0:.1f}s "
          f"→ {len(scan_ids)} scan(s) under {blob_root}")
    print(f"[pilot] share the building: POST /api/projects/share "
          f"{{client={client!r}, project={project!r}}} → building_embed_url → "
          f"GET /api/projects/manifest")
    return scan_ids


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct ONE persisted scan from multiple videos (pilot harness)."
    )
    # Accept both repeated --video and a multi-valued --videos, then merge in order.
    parser.add_argument(
        "--videos", nargs="+", default=[], metavar="VIDEO",
        help="Two or more video files, fed sequentially into one scan.",
    )
    parser.add_argument(
        "--video", action="append", default=[], dest="video_repeated", metavar="VIDEO",
        help="A video file (repeatable). Combine with or instead of --videos.",
    )
    parser.add_argument(
        "--mode", choices=["single", "per-scene"], default="single",
        help="single (default) = one continuous/overlapping capture → ONE fused scan; "
             "per-scene = distinct non-overlapping locations → one scan per video, grouped "
             "as a building (e.g. the Chorus pilot's 5 units).",
    )
    parser.add_argument("--user-id", default="local", help="Owner user id for the scan.")
    parser.add_argument("--client", default=None, help="Grouping tag, e.g. 'efficura'.")
    parser.add_argument("--project", default=None, help="Grouping tag, e.g. 'chorus'.")
    splat = parser.add_mutually_exclusive_group()
    splat.add_argument(
        "--export-splat", dest="export_splat", action="store_true", default=True,
        help="Export + persist a 3DGS splat.ply (default on).",
    )
    splat.add_argument(
        "--no-export-splat", dest="export_splat", action="store_false",
        help="Skip the 3DGS splat export.",
    )
    detect = parser.add_mutually_exclusive_group()
    detect.add_argument(
        "--detect-objects", dest="detect_objects", action="store_true", default=True,
        help="Agentic object-detection pass at finalize: auto-discover objects + locate them "
             "in 3D so facts.objects (the viewer's bounding boxes) is populated (default on).",
    )
    detect.add_argument(
        "--no-detect-objects", dest="detect_objects", action="store_false",
        help="Skip object detection (faster; scans will have no 3D bounding boxes).",
    )
    parser.add_argument(
        "--extract-fps", type=float, default=2.0,
        help="Effective FPS to sample from each video (default 2).",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Feed at real-time pace instead of as fast as possible (offline default).",
    )
    parser.add_argument("--submap-size", type=int, default=32, help="Frames per submap.")
    parser.add_argument(
        "--blob-root", default="./pilot_scans",
        help="Local directory for persisted scan blobs (cloud.npz, splat.ply, keyframes).",
    )
    parser.add_argument("--scan-id", default=None, help="Override the generated scan_id.")
    parser.add_argument(
        "--demo-index", action="store_true", default=False,
        help="Demo build (docs/demo-2026-07): after save_scene, write "
             "derived/demo/frames_index.json mapping report keyframes to trajectory "
             "rows + cameras (features.md §0.3). Default off — product path untouched.",
    )
    parser.add_argument(
        "--persist-trajectory", action="store_true", default=False,
        help="Demo build: thread the live solver's per-keyframe {poses, intrinsics, "
             "source_frame_id} into save_scene so the record gets trajectory.npz "
             "(GPU-free LeRobot export + pose-derived up-vector). Default off.",
    )
    parser.add_argument(
        "--label", action="append", default=[], dest="labels", metavar="LABEL",
        help="Human space name for a scan, repeatable + aligned to video order "
             "(1st --label → 1st video's scan, etc.). per-scene: one per clip; "
             "single: only the first is used. Usually supplied via the captures manifest "
             "(see modal_reconstruct_pilot.py) rather than by hand.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    videos = list(args.videos) + list(args.video_repeated)
    if not videos:
        print("error: provide at least one video via --videos or --video", file=sys.stderr)
        return 2
    reconstruct_pilot(
        videos=videos,
        user_id=args.user_id,
        client=args.client,
        project=args.project,
        export_splat=args.export_splat,
        detect_objects=args.detect_objects,
        extract_fps=args.extract_fps,
        fast=not args.realtime,
        blob_root=args.blob_root,
        submap_size=args.submap_size,
        scan_id=args.scan_id,
        mode=args.mode,
        labels=list(args.labels) or None,
        demo_index=args.demo_index,
        persist_trajectory=args.persist_trajectory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
