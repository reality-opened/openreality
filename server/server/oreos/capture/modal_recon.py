"""Standalone Modal app for the LiDAR capture-session recon lane (EXP-45).

App: ``lidar-recon`` — modeled on ``server/oreos/recordings/modal_recon.py``'s
standalone ``oreos-recon`` app (its own image, its own GPU function, no dependency
on the ``vggt-slam-streaming`` broker app's huge image). Unlike that lane's
CPU-orchestrator + separate-GPU-app split (``oreos_recording_job`` on the broker
app hands frames to the standalone ``oreos-recon`` app and persists on return),
THIS lane does everything in ONE function: unzip -> ``CaptureSession`` -> extract
``rgb.mp4`` -> VIO keyframe gate -> VGGT-SLAM with the LiDAR metric anchor -> metric
gauge layer -> persist. There is no second app to hand frames to, so one A100
container runs start to finish.

Deploy: ``modal deploy server/oreos/capture/modal_recon.py``

Critical dependency: the deployed server's core pin (``v2.2.0``) does NOT contain
the LiDAR machinery (``vggt_slam.capture_session``, ``lidar_anchor``,
``metric_layer``, ``offline_fit`` — all EXP-45, unreleased). This image
pip-installs core from the feature branch by SHA instead of the release tag::

    openreality-core @ git+https://github.com/reality-opened/core@e3ee6fb

That SHA is ``feat/lidar-imu-grounding``'s Phase-2 tip (capture session + LiDAR
anchor + metric gauge layer + vision-consistency edges + the ARKit/CV convention
fix). Bump ``CORE_REF`` below once this work lands on a release tag, and this lane
can move to the tag-pinned ``OREOS_CORE_SOURCE`` convention
``server/oreos/recordings/modal_image.py`` uses for its siblings.

Scenes/jobs stores: this app is standalone (not attached to ``vggt-slam-streaming``
via a shared ``app`` object), but it references the SAME NAMED ``vggt-slam-scenes``
Dict + Volume and the SAME NAMED ``vggt-slam-demo-jobs`` Dict the broker reads —
Modal Dicts/Volumes are named singletons shared across every app that references
them, so a scene persisted here lands in the byte-identical store a scene persisted
by ``demo_recon_job``/``oreos_recording_job`` would. That is what makes "the scene
appears in the workspace library exactly like other ingest sources" true without
the broker app needing to know this lane exists.

The pure logic this job calls into (zip validation, safe extraction, the
``save_scene``-shaped persistence bridge) lives in sibling modules with no
flask/modal/core import at module scope, so it is unit-testable without a GPU —
see ``zip_validate.py`` / ``session_assembly.py`` / ``persist_scene.py`` and
``tests/test_oreos_capture_ingest.py``. This file itself (image build + the SLAM
loop) cannot be unit-tested (needs a real GPU + gtsam + the pinned core checkout).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("lidar-recon")

# --- core provenance: pinned by SHA, not by the release tag (see module docstring) ---
CORE_REPO = "https://github.com/reality-opened/core"
CORE_REF = os.environ.get("LIDAR_RECON_CORE_REF", "e3ee6fb")

# This python package's root ("<repo>/server/"), resolved from this file's own path
# so the image build works regardless of the deploying shell's CWD (unlike
# modal_streaming.py's `add_local_dir("server", ...)`, which is CWD-relative and
# only correct when `modal deploy` is run from the repo root).
_THIS_FILE = Path(__file__).resolve()
# Image-definition-time only: on the deploying machine this file sits at
# <repo>/server/oreos/capture/modal_recon.py; INSIDE the container Modal mounts
# it at /root/modal_recon.py (2 parents), where parents[2] raises IndexError and
# the value is never needed (the image is already built). Guard with is_local().
_SERVER_PKG_DIR = _THIS_FILE.parents[2] if modal.is_local() else None  # .../capture -> .../oreos -> .../server

# Same third-party chain as server/oreos/recordings/modal_image.py's Oreos workers:
# the MIT-SPARK VGGT_SPARK backbone (loads VGGT-1B weights), deliberately NOT
# vggt-omega — that fork is the live-streaming product's backbone, claims the same
# `vggt` import name, and swapping it in here would silently break the weight load
# (see that module's docstring for the full reasoning; this lane inherits it).
_APT = ("libgl1-mesa-glx", "libglib2.0-0", "git", "cmake", "build-essential")
_PIP = (
    "torch==2.3.1", "torchvision==0.18.1", "numpy", "Pillow", "open3d",
    "opencv-python", "trimesh", "huggingface_hub", "einops", "safetensors",
    "gtsam-develop", "scipy", "pytorch_metric_learning", "pytorch-lightning",
    "viser==0.2.23", "matplotlib", "gradio", "termcolor", "tqdm", "omegaconf",
    "requests", "lz4", "ftfy", "regex", "imageio", "pyyaml",
    # This job's own persistence/billing/job-status glue needs these — NOT part
    # of core. `import server.oreos.jobs` (used for job_status_record) forces
    # Python to execute server/oreos/__init__.py first, which imports EVERY
    # route module in that package; traced module-level imports across all of
    # them (jobs/routes_agent/routes_docs/routes_ingest/routes_sam3d/routes_nav/
    # routes_planes/routes_export/routes_export_job/routes_lod/routes_imported/
    # routes_recordings/routes_capture + their own server.export.*/
    # server.scene_report.*/server.oreos.* imports) land on exactly:
    # flask, numpy, scipy (already above), pydantic. psycopg is the ledger's own
    # lazy import (server/billing/db.py) — the 2026-08-05 shadow-mode outage was
    # exactly THIS class of bug (psycopg missing from an image's pip list), so
    # it is listed explicitly rather than left to degrade silently.
    "flask", "pydantic>=2.8", "psycopg[binary,pool]>=3.2",
)
_SALAD = (
    "git clone https://github.com/Dominic101/salad.git /root/third_party/salad"
    " && pip install -e /root/third_party/salad"
)
_VGGT_SPARK = (
    "git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /root/third_party/vggt"
    " && pip install -e /root/third_party/vggt"
)

_CORE_INSTALL_HINT = (
    f"core {CORE_REF} could not be installed from {CORE_REPO}. Check that the "
    "'github-token' Modal secret exists and has read access to reality-opened/core, "
    f"and that {CORE_REF} is still a valid ref on that repo."
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(*_APT)
    .pip_install(*_PIP)
    .run_commands(_SALAD)
    .run_commands(_VGGT_SPARK)
    .run_commands(
        # Same guard as modal_streaming.py / modal_image.py: an empty/misnamed
        # secret otherwise bakes an EMPTY credential into ~/.gitconfig and only
        # surfaces later as GitHub's opaque "Invalid username or token".
        'test -n "${GITHUB_TOKEN}" || { echo "FATAL: github-token secret has no'
        ' GITHUB_TOKEN value"; exit 1; }',
        'git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/"'
        '.insteadOf "https://github.com/"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .run_commands(
        f"pip install 'openreality-core @ git+{CORE_REPO}@{CORE_REF}'"
        f' || {{ echo "FATAL: {_CORE_INSTALL_HINT}"; exit 1; }}',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({"LIDAR_RECON_CORE_REF": CORE_REF})
    # Mount last (no copy — nothing runs after this): this job's own package code.
    .add_local_dir(str(_SERVER_PKG_DIR), remote_path="/root/project/server")
)

# Named singletons shared with the broker app (modal_streaming.py) — see the module
# docstring for why referencing them by name (not by importing that app's objects)
# is sufficient for scenes persisted here to land in the same store.
model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)
scene_dict = modal.Dict.from_name("vggt-slam-scenes", create_if_missing=True)
scene_vol = modal.Volume.from_name("vggt-slam-scenes", create_if_missing=True)
demo_jobs_dict = modal.Dict.from_name(
    os.environ.get("DEMO_JOBS_DICT", "vggt-slam-demo-jobs"), create_if_missing=True
)

CACHE_PATH = "/root/.cache/torch/hub"
SCENES_PATH = "/root/scenes"
_UPLOADS_DIRNAME = "_uploads"

# Runtime secrets: billing is best-effort by construction (server/billing/meter.py
# degrades to a no-op if unconfigured/unreachable) but attached for parity with
# every other ingest job so this lane's GPU spend lands in the same ledger.
_RUNTIME_SECRETS = [modal.Secret.from_name("billing-secret", required_keys=["DATABASE_URL"])]


def _ensure_models() -> None:
    import torch

    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            os.path.join(ckpt_dir, "model.pt"),
        )
    if not os.path.exists(os.path.join(ckpt_dir, "dino_salad.ckpt")):
        torch.hub.download_url_to_file(
            "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            os.path.join(ckpt_dir, "dino_salad.ckpt"),
        )
    if not os.path.exists(os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")):
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model_cache.commit()


def _compose_metric_points_with_colors(submaps, layer, stride: int = 8):
    """``vggt_slam.metric_layer.compose_metric_points`` re-implemented with colors.

    Core's ``compose_metric_points`` (positions only) can't be extended from here
    without editing core, so this mirrors its exact masking/gauge-composition logic
    — same per-frame confidence mask, same stride, same gauge action — and carries
    each submap's ``.colors`` array (the same one ``Submap.get_points_colors()``
    reads) alongside the positions. Returns ``(positions (N,3) float32, colors
    (N,3) uint8)``; both arrays are empty (not None) when no submap has any point
    surviving the confidence mask, so downstream code never has to special-case
    "zero" vs "absent"."""
    import numpy as np

    all_pts, all_cols = [], []
    for sm in submaps:
        G = layer.gauge_matrix(sm.submap_id)
        thr = sm.get_conf_threshold()
        colors_sm = getattr(sm, "colors", None)
        for k in range(len(sm.pointclouds)):
            pts = np.asarray(sm.pointclouds[k], float)[::stride, ::stride]
            conf = np.asarray(sm.get_conf_masks_frame(k), float)[::stride, ::stride]
            mask = (conf > thr) & np.isfinite(pts).all(axis=-1)
            if not mask.any():
                continue
            cam_to_local = np.linalg.inv(sm.poses[k])
            p_local = pts[mask] @ cam_to_local[:3, :3].T + cam_to_local[:3, 3]
            world_pts = p_local @ G[:3, :3].T + G[:3, 3]
            all_pts.append(world_pts)
            if colors_sm is not None:
                cols = np.asarray(colors_sm[k], float)[::stride, ::stride][mask]
                if cols.size and cols.max() <= 1.0 + 1e-6:
                    cols = cols * 255.0
                all_cols.append(np.clip(cols, 0, 255).astype(np.uint8))
            else:
                all_cols.append(np.full((world_pts.shape[0], 3), 128, dtype=np.uint8))
    positions = np.concatenate(all_pts).astype(np.float32) if all_pts else np.empty((0, 3), np.float32)
    colors = np.concatenate(all_cols) if all_cols else np.empty((0, 3), np.uint8)
    return positions, colors


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_PATH: model_cache, SCENES_PATH: scene_vol},
    secrets=_RUNTIME_SECRETS,
    timeout=7200,
    max_containers=1,  # spend guard: one capture recon at a time
)
def lidar_recon_job(
    job_id: str,
    user_id: str,
    scan_id: str,
    upload_rel_path: str,
    session_label: str = None,
    submap_size: int = 16,
    max_loops: int = 1,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    lidar_max_depth: float = 4.5,
    lidar_min_conf: int = 2,
    lidar_keyframes: int = 5,
    point_stride: int = 8,
) -> dict:
    """Unzip the staged capture session, run VGGT-SLAM with the LiDAR metric
    anchor, build the metric gauge layer, and persist a ``source="recon_lidar"``
    scene.

    Mirrors core's ``modal_app.py::run_slam`` capture-session lane (frame
    extraction, keyframe selection, the submap-batched VGGT+graph-optimize loop),
    but with the keyframe gate FIXED to ``"vio"`` (``select_keyframes`` — pose-delta
    selection, no per-frame RAFT decode) rather than exposed as a flag: this lane
    always has real ARKit poses to gate on, so the cheap path is always available
    and there is no reason to pay for optical-flow gating here. No live Viser
    tunnel (``skip_viewer=True``) — this is a background batch job, not an
    interactive session."""
    import shutil
    import sys
    import time
    import traceback

    sys.path.insert(0, "/root/project")

    from server.oreos.capture.persist_scene import persist_capture_scene
    from server.oreos.capture.session_assembly import extract_capture_session
    from server.oreos.capture.zip_validate import CaptureZipRejected, validate_capture_zip
    from server.oreos.jobs import job_status_record
    from server.scene_report.store import ModalScenePersistence

    from server.billing import meter as _meter
    from server.billing import prices as _prices

    started = time.time()

    def publish(status: str, stage: str, **extra) -> None:
        try:
            demo_jobs_dict[str(job_id)] = job_status_record(
                job_id, user_id, status=status, stage=stage, scan_id=scan_id,
                kind="capture_lidar", **extra,
            )
        except Exception as exc:  # status is telemetry — never kill the job over it
            print(f"[lidar-recon] status publish failed ({stage}): {exc}")

    _hold_id = _meter.open_hold(
        user_id, "capture_lidar_job", str(job_id),
        _prices.CREDITS_LIDAR_RECON, ttl_s=_prices.HOLD_TTL_JOB_S,
    )

    # Resolve + contain the staged zip (demo_recon_job / oreos_recording_job's exact
    # containment rule): it must live inside the _uploads staging area of the
    # scenes volume — a forged/mangled rel path must never reach another blob.
    scene_vol.reload()
    rel = os.path.normpath(str(upload_rel_path)).lstrip("/")
    parts = rel.split(os.sep)
    zip_path = os.path.realpath(os.path.join(SCENES_PATH, rel))
    volume_root = os.path.realpath(SCENES_PATH)
    if (
        ".." in parts
        or _UPLOADS_DIRNAME not in parts
        or not zip_path.startswith(volume_root + os.sep)
    ):
        publish("failed", "upload", error="invalid_upload_path")
        _meter.void(_hold_id, "invalid_upload_path")
        return {"status": "error", "error": "invalid_upload_path"}
    if not os.path.isfile(zip_path):
        publish("failed", "upload", error="upload_not_found")
        _meter.void(_hold_id, "upload_not_found")
        return {"status": "error", "error": "upload_not_found"}

    session_dir = f"/tmp/capture_{scan_id[:12]}"

    try:
        publish("running", "extract")
        # Re-validate defensively even though the broker already did (finalize's
        # cheap header gate must never be the ONLY line of defense the job trusts).
        validate_capture_zip(zip_path)
        extract_capture_session(zip_path, session_dir)

        publish("running", "slam")
        result = _run_slam_and_metric_layer(
            session_dir=session_dir,
            submap_size=submap_size,
            max_loops=max_loops,
            conf_threshold=conf_threshold,
            lc_thres=lc_thres,
            lidar_max_depth=lidar_max_depth,
            lidar_min_conf=lidar_min_conf,
            lidar_keyframes=lidar_keyframes,
            point_stride=point_stride,
            models_ready_fn=_ensure_models,
        )

        publish("running", "persist")
        persistence = ModalScenePersistence(
            scene_dict, SCENES_PATH, commit_fn=scene_vol.commit, reload_fn=scene_vol.reload,
        )
        persisted = persist_capture_scene(
            persistence, user_id, scan_id,
            trajectory_rows=result["trajectory_rows"],
            points=result["points"],
            colors=result["colors"],
            metric_report=result["metric_report"],
            gauges=result["gauges"],
            anchor_telemetry=result["anchor_telemetry"],
            session_label=session_label or os.path.basename(zip_path),
            frames_total=result["frames_total"],
            frames_kept=result["frames_kept"],
            submap_count=result["submap_count"],
            loop_closures=result["loop_closures"],
        )

        # Success: drop the staged upload (demo_recon_job / oreos_recording_job
        # convention) and the local extraction scratch dir.
        upload_dir = os.path.dirname(zip_path)
        if os.path.basename(os.path.dirname(upload_dir)) == _UPLOADS_DIRNAME:
            shutil.rmtree(upload_dir, ignore_errors=True)
        scene_vol.commit()
        shutil.rmtree(session_dir, ignore_errors=True)

        elapsed = round(time.time() - started, 1)
        publish(
            "done", "done",
            elapsed_s=elapsed,
            derived=persisted["derived"],
            frames_total=result["frames_total"],
            frames_kept=result["frames_kept"],
            submaps=result["submap_count"],
            loop_closures=result["loop_closures"],
        )
        _meter.settle(
            _hold_id,
            _prices.CREDITS_LIDAR_RECON,
            usd=_prices.gpu_usd("A100-80GB", elapsed),
            basis=_prices.cost_basis_for("A100-80GB"),
            metadata={"elapsed_s": elapsed, "scan_id": scan_id},
        )
        print(f"[lidar-recon] job {job_id} done -> scan {scan_id} in {elapsed:.0f}s")
        return {"status": "done", "job_id": str(job_id), "scan_id": scan_id, "elapsed_s": elapsed}
    except CaptureZipRejected as exc:
        publish("failed", "extract", error=exc.error, detail=exc.detail)
        _meter.void(_hold_id, f"{exc.error}: {exc.detail}"[:200])
        shutil.rmtree(session_dir, ignore_errors=True)
        return {"status": "error", "error": exc.error, "detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 — job boundary
        traceback.print_exc()
        publish("failed", "error", error=f"{type(exc).__name__}: {exc}",
                elapsed_s=round(time.time() - started, 1))
        _meter.void(_hold_id, f"{type(exc).__name__}: {exc}"[:200])
        shutil.rmtree(session_dir, ignore_errors=True)
        return {"status": "error", "error": str(exc)}


def _run_slam_and_metric_layer(
    *,
    session_dir: str,
    submap_size: int,
    max_loops: int,
    conf_threshold: float,
    lc_thres: float,
    lidar_max_depth: float,
    lidar_min_conf: int,
    lidar_keyframes: int,
    point_stride: int,
    models_ready_fn,
) -> dict:
    """The GPU-only half of the job: everything from ``CaptureSession(dir)`` to the
    composed metric trajectory/points, no persistence. Split out of
    ``lidar_recon_job`` purely for readability — it still needs core/gtsam/torch
    and cannot run outside the container."""
    import glob

    import torch

    import vggt_slam.slam_utils as utils
    from vggt.models.vggt import VGGT
    from vggt_slam.capture_session import CaptureSession, frame_index_from_name, select_keyframes
    from vggt_slam.metric_layer import build_from_capture, compose_metric_trajectory, load_solved_poses
    from vggt_slam.solver import Solver

    models_ready_fn()

    session = CaptureSession(session_dir)

    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
        skip_viewer=True,
        capture_session=session,
        lidar_anchor=True,
        lidar_max_depth=lidar_max_depth,
        lidar_min_conf=lidar_min_conf,
        lidar_keyframes=lidar_keyframes,
    )

    model = VGGT()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        )
    )
    model.eval().to(torch.bfloat16).to("cuda")

    # mirrors core's modal_app.py::run_slam capture-session lane
    abs_image_folder = solver.capture_session.rgb_dir() or solver.capture_session.extract_frames()
    image_names = [
        f for f in glob.glob(os.path.join(abs_image_folder, "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    frames_total = len(image_names)
    if frames_total == 0:
        raise RuntimeError(f"no frames extracted from rgb.mp4 under {abs_image_folder}")

    # EXP-45 VIO keyframe gate (always on for this lane — see the job docstring):
    # select keyframes from ARKit pose deltas, free and parallax-aware. Selection
    # only; VIO never enters the SL(4) graph (the EXP-45 double-counting rule).
    keep = set(select_keyframes(solver.capture_session))
    image_names = [n for n in image_names if frame_index_from_name(n) in keep]
    frames_kept = len(image_names)
    if frames_kept == 0:
        raise RuntimeError("VIO keyframe gate kept zero frames")

    image_names_subset: list = []
    n_submaps = 0
    overlap = 1
    for image_name in image_names:
        image_names_subset.append(image_name)
        if len(image_names_subset) == submap_size + overlap or image_name == image_names[-1]:
            n_submaps += 1
            predictions = solver.run_predictions(image_names_subset, model, max_loops, None, None)
            solver.add_points(predictions)
            solver.graph.optimize()
            image_names_subset = image_names_subset[-overlap:]

    submaps = [sm for sm in solver.map.ordered_submaps_by_key() if not sm.get_lc_status()]
    if not submaps:
        raise RuntimeError("SLAM produced no non-loop-closure submaps")

    poses_txt = os.path.join(session_dir, "poses.txt")
    solver.map.write_poses_to_file(poses_txt, solver.graph, kitti_format=False)
    vision_poses = load_solved_poses(poses_txt)

    ratios = {sm.get_id(): sm.metric_scale for sm in submaps if sm.metric_scale is not None}
    if not ratios:
        raise RuntimeError(
            "no submap has a usable LiDAR scale ratio — every submap's "
            "LidarMetricAnchor.compute_submap_ratio failed its gate"
        )

    layer, metric_report = build_from_capture(submaps, session, ratios, vision_poses=vision_poses)

    traj_tum_path = os.path.join(session_dir, "metric_trajectory_tum.txt")
    trajectory_rows = compose_metric_trajectory(submaps, layer, out_tum=traj_tum_path)
    if trajectory_rows.shape[0] < 2:
        raise RuntimeError(f"metric trajectory has {trajectory_rows.shape[0]} pose(s); need >= 2")

    positions, colors = _compose_metric_points_with_colors(submaps, layer, stride=point_stride)

    gauges = {
        str(sid): {"scale": layer.scale(sid), "matrix": layer.gauge_matrix(sid).tolist()}
        for sid in sorted(layer.nodes)
    }
    loop_closures = int(solver.graph.get_num_loops())
    anchor_telemetry = {
        "core_source": {"repo": CORE_REPO, "ref": CORE_REF},
        "keyframe_gate": "vio",
        "lidar_anchor": True,
        "frames_total": frames_total,
        "frames_kept": frames_kept,
        "submaps": len(submaps),
        "loop_closures": loop_closures,
        "submap_ratios": [
            {
                "submap_id": sm.get_id(),
                "r_i": (sm.metric_scale or {}).get("r_i"),
                "within_cov": (sm.metric_scale or {}).get("within_cov"),
                "n_frames": (sm.metric_scale or {}).get("n_frames"),
                "ok": bool((sm.metric_scale or {}).get("ok")),
                "reason": (sm.metric_scale or {}).get("reason"),
            }
            for sm in submaps
        ],
        "junction_rotations": metric_report.get("junction_rotations"),
        "junctions_skipped_chain_break": metric_report.get("junctions_skipped_chain_break"),
        "vision_consistency_edges": metric_report.get("vision_consistency_edges"),
        "vision_vs_arkit_junction_rot_deg": metric_report.get("vision_vs_arkit_junction_rot_deg"),
        "final_error": metric_report.get("final_error"),
        "scales": metric_report.get("scales"),
    }

    return {
        "trajectory_rows": trajectory_rows,
        "points": positions,
        "colors": colors,
        "metric_report": metric_report,
        "gauges": gauges,
        "anchor_telemetry": anchor_telemetry,
        "frames_total": frames_total,
        "frames_kept": frames_kept,
        "submap_count": len(submaps),
        "loop_closures": loop_closures,
    }


# ---------------------------------------------------------------------------
# CLI bootstrap: upload a LOCAL capture-session zip and run the job — the manual
# twin of POST /api/workspace/ingest/capture/{init,chunk,finalize}, used for
# backfill and for exercising the whole pipeline before an iOS build exists.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes={SCENES_PATH: scene_vol}, timeout=1200)
def _stage_capture_upload(user_id: str, upload_id: str, filename: str, data: bytes) -> str:
    import re

    def _safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:128] or "_"

    fname = os.path.basename(str(filename)) or "session.zip"
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", fname)[:128]
    rel = "/".join([_safe(user_id), _UPLOADS_DIRNAME, _safe(upload_id), fname])
    dest = os.path.join(SCENES_PATH, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(data)
    scene_vol.commit()
    print(f"[lidar-recon] staged upload {rel} ({len(data) / 1024 / 1024:.1f} MB)")
    return rel


@app.local_entrypoint()
def run(zip_path: str, user_id: str = "capture-ops", scan_id: str = None, detach: bool = False):
    """``modal run server/oreos/capture/modal_recon.py --zip-path <session>.zip``

    Uploads a LOCAL capture-session zip and runs ``lidar_recon_job`` on it.
    ``--detach`` spawns and returns immediately (poll ``GET
    /api/workspace/jobs/<job_id>``); default blocks until the scene is persisted.
    """
    import uuid

    from server.oreos.jobs import job_status_record  # local shell: server/ is the cwd

    if not os.path.isfile(zip_path):
        raise SystemExit(f"zip not found: {zip_path}")
    job_id = uuid.uuid4().hex
    upload_id = uuid.uuid4().hex
    scan_id = scan_id or uuid.uuid4().hex

    with open(zip_path, "rb") as fh:
        data = fh.read()
    print(f"Staging {zip_path} ({len(data) / 1024 / 1024:.1f} MB) as job {job_id} -> scan {scan_id}")
    rel = _stage_capture_upload.remote(user_id, upload_id, os.path.basename(zip_path), data)

    try:
        demo_jobs_dict[job_id] = job_status_record(
            job_id, user_id, status="queued", stage="upload", scan_id=scan_id, kind="capture_lidar",
        )
    except Exception as exc:
        print(f"[lidar-recon] queued-status publish failed: {exc}")

    kwargs = dict(
        job_id=job_id, user_id=user_id, scan_id=scan_id, upload_rel_path=rel,
        session_label=os.path.basename(zip_path),
    )
    if detach:
        lidar_recon_job.spawn(**kwargs)
        print(f"Spawned. Poll: GET /api/workspace/jobs/{job_id}")
    else:
        result = lidar_recon_job.remote(**kwargs)
        print(f"Done -> {result}")
