"""Oreos recon worker (Modal): camera-only VGGT-SLAM over a session's frames.

Product distillation of the EXP-36 harness
(platform/experiments/exp36_oreos_dimos_spike/scripts/modal_exp36_slam.py), keeping
every hardening that experiment paid for:
  * poses are written FIRST, via a self-written TUM writer; improper rotations
    (SL(4) decomposition can emit det(R) = -1) are COUNTED from the raw camera
    matrices — core repairs them now, so counting our own repair would count
    zero forever and quietly kill the QC signal (see server/oreos/recordings/pose_qc.py);
  * every export step is non-fatal and empty-safe;
  * per-submap npz bundles are exported for replay-node playback;
  * frames may arrive as frames/ or a single frames.tar (bulk uploads die on flaky
    egress; single-file tar + server-side untar survives).

App: oreos-recon · sessions volume: oreos-sessions · model cache: vggt-slam-models
(shared with the streaming product — VGGT-1B/salad/DINOv2 already cached there).

The image (and the core it installs) comes from ``server/oreos/recordings/modal_image.py``:
``OREOS_CORE_SOURCE=tag`` (default) pins core, ``=local`` ships the sibling
checkout. Either way the source is stamped into ``recon_summary.json`` under
``core_source`` — read that before trusting a number.

Usage (spawn-and-exit; never hold a long client connection):
    modal run --detach server/oreos/recordings/modal_recon.py --session <name> [--submap-size 16]
"""

import os
from pathlib import Path

import modal

try:  # local import (deploy-time, tests): the server repo package
    from server.oreos.recordings import modal_image as oreos_image, pose_qc
except ImportError:  # remote container: mounted flat at /root/oreos_*.py
    import oreos_modal_image as oreos_image  # type: ignore[no-redef]
    import oreos_pose_qc as pose_qc  # type: ignore[no-redef]

app = modal.App("oreos-recon")

# Image recipe + core source: server/oreos/recordings/modal_image.py (shared by all three
# Oreos workers). OREOS_CORE_SOURCE=tag (default) installs the pinned core tag;
# =local mounts the sibling checkout, as this file used to do unconditionally.
image = oreos_image.build_oreos_image(
    local_files=((Path(pose_qc.__file__).resolve(), "/root/oreos_pose_qc.py"),),
)

model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)
sessions_vol = modal.Volume.from_name("oreos-sessions", create_if_missing=True)

CACHE_PATH = "/root/.cache/torch/hub"
DATA_PATH = "/root/sessions"


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


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_PATH: model_cache, DATA_PATH: sessions_vol},
    timeout=7200,
)
def run_recon(
    session: str,
    submap_size: int = 16,
    max_loops: int = 1,
    min_disparity: float = 50.0,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    voxel: float = 0.05,
) -> dict:
    import glob
    import json
    import tarfile
    import time

    import cv2
    import numpy as np
    import open3d as o3d
    import torch
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm

    import vggt_slam.slam_utils as utils
    from vggt.models.vggt import VGGT
    from vggt_slam.solver import Solver

    _ensure_models()

    sess_dir = os.path.join(DATA_PATH, session)
    frames_dir = os.path.join(sess_dir, "frames")
    tar_path = os.path.join(sess_dir, "frames.tar")
    if not os.path.isdir(frames_dir) and os.path.exists(tar_path):
        with tarfile.open(tar_path) as tf:
            # filter="data" rejects absolute/../ member paths, links, devices and
            # setuid bits (CVE-2007-4559 class). The tar is written by our own
            # upload stage, but it round-trips through a shared Modal volume, and
            # the image pins py3.11 so the default is still the unsafe one and no
            # DeprecationWarning fires to flag it.
            tf.extractall(sess_dir, filter="data")
        sessions_vol.commit()

    image_names = utils.sort_images_by_number(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not image_names:
        return {"error": f"no frames under {frames_dir}"}

    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
        skip_viewer=True,
    )
    model = VGGT()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        )
    )
    model.eval().to(torch.bfloat16).to("cuda")

    subset: list = []
    n_kept = n_submaps = 0
    solver_error = None
    t0 = time.time()
    try:
        for image_name in tqdm(image_names):
            img = cv2.imread(image_name)
            if solver.flow_tracker.compute_disparity(img, min_disparity, False):
                subset.append(image_name)
                n_kept += 1
            if len(subset) == submap_size + 1 or image_name == image_names[-1]:
                if not subset:
                    continue
                n_submaps += 1
                predictions = solver.run_predictions(subset, model, max_loops, None, None)
                solver.add_points(predictions)
                solver.graph.optimize()
                subset = subset[-1:]
    except Exception as e:  # noqa: BLE001 — EXP-36 failure mode #2: report, keep partials
        solver_error = f"{type(e).__name__}: {e}"

    total_time = time.time() - t0
    results_dir = os.path.join(sess_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Poses first. The improper-rotation event is counted from the RAW camera
    # matrices (see server/oreos/recordings/pose_qc.py): core repairs det(R) = -1 inside
    # decompose_camera now, so counting our own repair would count zero forever
    # and silently kill the QC signal it feeds. No local repair remains — only a
    # tripwire on core's guarantee.
    export_errors: list[str] = []
    n_repairs = 0
    n_nonfinite = 0
    repairs_known = True
    pose_tripwire = None
    tum_lines: list[str] = []
    bundles = []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        raw = pose_qc.submap_improper_stats(submap, solver.graph)
        if raw["available"]:
            n_repairs += raw["n_improper"]
            n_nonfinite += raw["n_nonfinite"]
        else:  # never report a fake 0 — that is how this signal died before
            repairs_known = False
            export_errors.append(f"submap {submap.get_id()}: {raw['note']}")
        try:
            fixed = pose_qc.proper_rigid_poses(
                submap.get_all_poses_world(solver.graph), where=f"submap {submap.get_id()}"
            )
        except pose_qc.ImproperRotationError as e:
            # Refuse to write plausible-looking wrong quaternions; the QC gate
            # turns solver_error into an OUT_OF_CLASS refusal with this reason.
            pose_tripwire = f"{type(e).__name__}: {e}"
            export_errors.append(pose_tripwire)
            break
        bundles.append((submap, fixed))
        for fid, Tn in zip(submap.get_frame_ids(), fixed):
            q = Rotation.from_matrix(Tn[:3, :3]).as_quat()
            t_ = Tn[:3, 3]
            tum_lines.append(
                f"{float(fid):.6f} {t_[0]:.6f} {t_[1]:.6f} {t_[2]:.6f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"
            )
    with open(os.path.join(results_dir, "est_tum.txt"), "w") as f:
        f.write("\n".join(tum_lines) + "\n")
    sessions_vol.commit()

    submap_dir = os.path.join(results_dir, "submaps")
    os.makedirs(submap_dir, exist_ok=True)
    all_pts, all_cols = [], []
    for submap, fixed in bundles:
        try:
            pts = submap.get_points_in_world_frame(solver.graph).reshape(-1, 3)
            colors = np.asarray(submap.get_points_colors()).reshape(-1, 3)
            if len(pts) == 0:
                export_errors.append(f"submap {submap.get_id()}: empty")
                continue
            if colors.size and colors.max() > 1.0:
                colors = colors / 255.0
            sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            if len(colors) == len(pts):
                sp.colors = o3d.utility.Vector3dVector(colors)
            sp = sp.voxel_down_sample(voxel)
            np.savez_compressed(
                os.path.join(submap_dir, f"submap_{submap.get_id():04d}.npz"),
                frame_ts=np.asarray(submap.get_frame_ids(), dtype=np.float64),
                poses_world=np.asarray(fixed),
                points=np.asarray(sp.points, dtype=np.float32),
                colors=(np.asarray(sp.colors) * 255).astype(np.uint8)
                if sp.has_colors()
                else np.zeros((len(sp.points), 3), dtype=np.uint8),
            )
            all_pts.append(np.asarray(sp.points))
            all_cols.append(np.asarray(sp.colors) if sp.has_colors() else None)
        except Exception as e:  # noqa: BLE001
            export_errors.append(f"submap {submap.get_id()}: {type(e).__name__}: {e}")

    try:
        if all_pts:
            merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(all_pts)))
            cols = [c for c in all_cols if c is not None]
            if len(cols) == len(all_pts):
                merged.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
            o3d.io.write_point_cloud(
                os.path.join(results_dir, "map_preview.ply"), merged.voxel_down_sample(voxel)
            )
    except Exception as e:  # noqa: BLE001
        export_errors.append(f"map_preview: {type(e).__name__}: {e}")

    summary = {
        "session": session,
        # Which core actually produced these numbers (build-time stamp; see
        # server/oreos/recordings/modal_image.py). Without it a recon is unattributable:
        # the image used to ship the operator's working tree, silently.
        "core_source": oreos_image.core_provenance_from_env(),
        "frames_total": len(image_names),
        "frames_kept": n_kept,
        "submaps": n_submaps,
        "loop_closures": solver.graph.get_num_loops(),
        # RAW improper-rotation count: poses whose camera matrix had det(R) < 0
        # BEFORE any repair, wherever the repair happens (core does it now). Feeds
        # sign_repair_rate in server/qc/confidence.py, whose EXP-36 calibration
        # therefore still means what it meant. None = could not be measured.
        "n_pose_sign_repairs": n_repairs if repairs_known else None,
        "n_pose_nonfinite_dets": n_nonfinite,
        "n_poses_written": len(tum_lines),
        "solver_error": solver_error or pose_tripwire,
        "pose_tripwire": pose_tripwire,
        "export_errors": export_errors,
        "gpu_seconds": round(total_time, 2),
        "config": {
            "submap_size": submap_size, "max_loops": max_loops,
            "min_disparity": min_disparity, "conf_threshold": conf_threshold,
            "lc_thres": lc_thres,
        },
    }
    with open(os.path.join(results_dir, "recon_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sessions_vol.commit()
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(session: str, submap_size: int = 16, min_disparity: float = 50.0):
    call = run_recon.spawn(
        session=session, submap_size=submap_size, min_disparity=min_disparity
    )
    print(f"SPAWNED recon {session}: {call.object_id}")
