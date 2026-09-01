"""``dynamics/human/`` sidecar: world-frame human motion from SAM 3D Body (MHR).

Offline **post-processing** stage (opt-in, additive, default-off — like the ObjectTrack
``dynamics/`` sidecar): given a completed scan's posed frames + per-frame SLAM depth +
intrinsics, plus per-frame SAM 3D Body (MHR) outputs, it anchors the body into the SLAM
world frame and writes ``dynamics/human/{human_tracks.jsonl, mhr_params.npz}``.

This module is the **CPU/GPU-free heart**: pure numpy, no torch, no ``vggt_slam`` import, so
it unit-tests with synthetic fixtures on the CI broker. The GPU half (SAM 3D Body inference)
lives in ``server/modal_dynamics_human.py``; the opt-in local entry point is
``scripts/export_dynamics_human.py``.

Ported verbatim from the validated EXP-2 harness
(``platform:experiments/exp2_sam3dbody_motion/anchor.py`` + ``run_pipeline.py``; findings
``platform:experiments/research/2026-07-05-exp2-sam3dbody-motion.md``). Schema:
``docs/dynamics-human-export.md`` (mirrors the ObjectTrack invariant — *share the clock, not
the metric frame*).

THE ONE FACT THAT MAKES THIS EASY: both cameras use the SAME OpenCV convention (+X right,
+Y down, +Z forward). SAM 3D Body ``pred_keypoints_3d`` are ROOT-RELATIVE METERS in the
OpenCV camera frame; VGGT ``extrinsic`` is world->cam, also OpenCV. So there is NO
inter-camera axis flip. Anchoring = (i) resolve monocular scale λ, (ii) place the root at
metric depth from SLAM (3DB's own ``tz`` is discarded — it rides on an assumed focal),
(iii) rigidly lift camera->world with the SLAM ``c2w`` pose.

WORLD-FRAME CAVEATS (ride WITH the data, never silently applied):
  * up-to-scale: ``joints_world`` are SLAM units, metric only up to one global λ
    (``lambda_slam_per_m``). Divide by λ for meters.
  * not gravity-aligned: the SLAM world == frame-0's camera frame, so world +Y ≈ DOWN.
    ``gravity_world`` is a heuristic (mean camera +Y); a floor-plane fit is the proper fix.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable, Optional

import numpy as np

# --- sidecar identity ------------------------------------------------------
DYNAMICS_HUMAN_SCHEMA_VERSION = "openreality-dynamics-human/0.1"
BODY_MODEL = "mhr"            # Meta Human Rig, as output by SAM 3D Body
SKELETON = "mhr70"           # the 70-keypoint set of pred_keypoints_3d
NUM_JOINTS = 70

# --- mhr70 skeleton metadata (sam_3d_body/metadata/mhr70.py). Root = hip midpoint. --------
ROOT_IDX = (9, 10)           # left_hip, right_hip -> pelvis = midpoint (SAM3DBody.pelvis_idx)

# mhr70 body edges (verified against sam_3d_body/metadata/mhr70.py): used only for the
# bone-length-stability diagnostic, never for the anchoring itself (which is uniform over 70).
BODY_EDGES = (
    (5, 6), (5, 9), (6, 10), (9, 10),          # torso
    (5, 7), (7, 62), (6, 8), (8, 41),          # arms (elbow->wrist)
    (9, 11), (11, 13), (10, 12), (12, 14),     # legs
    (0, 5), (0, 6),                            # head
)

# MHR parameter arrays serialized into mhr_params.npz (SAM 3D Body output names, verbatim).
# Only those actually present in the per-frame body dicts are written; all are optional.
# ``pred_keypoints_3d`` is renamed ``pred_keypoints_3d_cam`` on write (root-relative meters).
MHR_PARAM_KEYS = (
    "pred_pose_raw",          # (266,) full raw MHR pose vector
    "global_rot",             # (3,)   body global orientation, axis-angle, CAMERA frame
    "body_pose_params",       # (133,)
    "hand_pose_params",       # (108,)
    "shape_params",           # (45,)  identity/shape (should be ~const per person)
    "scale_params",           # (28,)  per-part scale
    "expr_params",            # (72,)  face/expression
    "mhr_model_params",       # (204,) packed MHR model params
    "pred_cam_t",             # (3,)   3DB's own perspective t (tx,ty ok-ish, tz NOT) — provenance
)


# ===========================================================================
# Camera projection helpers (OpenCV pinhole). K: (3,3) pixels.
# ===========================================================================
def root_of(kpt3d: np.ndarray) -> np.ndarray:
    """Pelvis = midpoint of the two hip keypoints. kpt3d: (...,70,3) -> (...,3)."""
    return 0.5 * (kpt3d[..., ROOT_IDX[0], :] + kpt3d[..., ROOT_IDX[1], :])


def backproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels + depth -> camera-frame XYZ (OpenCV). uv:(N,2) px, z:(N,) or scalar -> (N,3)."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    z = np.broadcast_to(np.asarray(z, float).reshape(-1), (uv.shape[0],))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * z
    y = (uv[:, 1] - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def project(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame XYZ -> pixels (OpenCV). xyz:(N,3) -> (N,2)."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.clip(xyz[:, 2], 1e-6, None)
    return np.stack([fx * xyz[:, 0] / z + cx, fy * xyz[:, 1] / z + cy], axis=-1)


def sample_depth(depth: np.ndarray, conf: np.ndarray, uv, win: int = 9,
                 conf_pct: float = 40.0) -> float:
    """Robust SLAM depth at a pixel: median over a (2*win+1)^2 window, keeping only pixels
    whose confidence is above the local ``conf_pct`` percentile. depth/conf: (H,W). Returns
    ``nan`` if the window is empty/degenerate (mirrors ObjectTrack's mask-region depth idea).
    Sampling ONLY the person's root pixel is deliberate: per-limb VGGT depth on a *mover* is
    never trusted (EXP-2/E2 lesson)."""
    if not np.isfinite(np.asarray(uv, dtype=float)).all():
        return float("nan")  # no detection on this frame
    H, W = depth.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    u0, u1 = max(0, u - win), min(W, u + win + 1)
    v0, v1 = max(0, v - win), min(H, v + win + 1)
    if u0 >= u1 or v0 >= v1:
        return float("nan")
    d = depth[v0:v1, u0:u1].astype(np.float64).ravel()
    c = conf[v0:v1, u0:u1].astype(np.float64).ravel()
    good = np.isfinite(d) & (d > 0)
    d, c = d[good], c[good]
    if d.size == 0:
        return float("nan")
    thr = np.percentile(c, conf_pct)
    keep = c >= thr
    d = d[keep] if keep.any() else d
    return float(np.median(d))


# ===========================================================================
# Scale λ (SLAM units per meter): similar triangles at the root depth.
# ===========================================================================
def estimate_scale_frame(kpt3d_rr: np.ndarray, kpt2d_px: np.ndarray, z_root: float,
                         K: np.ndarray, min_off_m: float = 0.15) -> float:
    """λ from one frame: image-plane joint offsets in SLAM units (Δpx·z_root/f) vs the same
    offsets in meters from 3DB (root-relative XY). Median over joints ≥ ``min_off_m`` off-root.
    Needs ONLY the root depth (production-safe on a dynamic person)."""
    root2d = 0.5 * (kpt2d_px[ROOT_IDX[0]] + kpt2d_px[ROOT_IDX[1]])
    fx, fy = K[0, 0], K[1, 1]
    du = (kpt2d_px[:, 0] - root2d[0]) * z_root / fx
    dv = (kpt2d_px[:, 1] - root2d[1]) * z_root / fy
    slam_off = np.hypot(du, dv)                                  # SLAM-unit image-plane offset
    rr = kpt3d_rr - root_of(kpt3d_rr)
    meter_off = np.hypot(rr[:, 0], rr[:, 1])                     # metric image-plane offset
    m = meter_off > min_off_m
    if m.sum() < 3:
        return float("nan")
    return float(np.median(slam_off[m] / meter_off[m]))


# ===========================================================================
# Per-frame anchoring: 3DB camera-frame joints (SLAM-scaled) -> world.
# ===========================================================================
def anchor_frame(kpt3d_rr: np.ndarray, kpt2d_px: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                 depth: np.ndarray, conf: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Return (joints_world (70,3) in SLAM units, z_root). Steps:
       1. root pixel -> robust SLAM depth z_root  (OVERRIDES 3DB's tz).
       2. root_cam = backproject(root_px, z_root)  (SLAM units).
       3. joints_cam = root_cam + λ * root_relative_joints   (meters -> SLAM units).
       4. joints_world = R_c2w @ joints_cam + t_c2w   (OpenCV both sides, no axis flip)."""
    root2d = 0.5 * (kpt2d_px[ROOT_IDX[0]] + kpt2d_px[ROOT_IDX[1]])
    z_root = sample_depth(depth, conf, root2d)
    if not np.isfinite(z_root):
        return np.full((kpt3d_rr.shape[0], 3), np.nan), np.nan
    root_cam = backproject(root2d[None], z_root, K)[0]           # (3,)
    rr = kpt3d_rr - root_of(kpt3d_rr)                            # ensure root-centered
    joints_cam = root_cam[None] + lam * rr                       # (70,3) SLAM units
    R, t = c2w[:3, :3], c2w[:3, 3]
    joints_world = joints_cam @ R.T + t[None]
    return joints_world, z_root


def anchor_sequence(poses: np.ndarray, intrinsics: np.ndarray, depth: np.ndarray,
                    depth_conf: np.ndarray, per_frame_body: list[dict],
                    lam: Optional[float] = None) -> dict:
    """Anchor a whole clip into the SLAM world frame.

    poses ``(S,4,4)`` cam->world · intrinsics ``(S,3,3)`` px · depth/depth_conf ``(S,H,W)`` ·
    ``per_frame_body`` list (len S) of dicts with ``pred_keypoints_3d`` (70,3 root-rel meters)
    and ``pred_keypoints_2d`` (70,2 px in the SLAM-preprocessed frame). Gap frames (no
    detection) carry NaN keypoints and anchor to a NaN (masked-out) row.

    Returns ``{world_raw, lam, lam_stats, z_root, valid}``. A single **global** λ is used
    (median of per-frame estimates) so the whole track shares one scale.
    """
    c2w = np.asarray(poses, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float32)
    conf = np.asarray(depth_conf, dtype=np.float32)
    S = len(per_frame_body)
    if S == 0:
        return {"world_raw": np.zeros((0, NUM_JOINTS, 3)), "lam": 1.0,
                "lam_stats": {"lam": 1.0, "lam_per_frame_std": float("nan"),
                              "lam_cv": float("nan"), "n_frames_used": 0},
                "z_root": np.zeros(0), "valid": np.zeros(0, dtype=bool)}
    J = int(np.asarray(per_frame_body[0]["pred_keypoints_3d"]).shape[0])

    # --- global λ (median of per-frame estimates) ---
    if lam is None:
        lam_f: list[float] = []
        for i in range(S):
            d = per_frame_body[i]
            k2 = np.asarray(d["pred_keypoints_2d"], dtype=np.float64)
            k3 = np.asarray(d["pred_keypoints_3d"], dtype=np.float64)
            root2d = 0.5 * (k2[ROOT_IDX[0]] + k2[ROOT_IDX[1]])
            z = sample_depth(depth[i], conf[i], root2d)
            if np.isfinite(z):
                lam_f.append(estimate_scale_frame(k3, k2, z, K[i]))
        lam_f = [x for x in lam_f if np.isfinite(x)]
        lam = float(np.median(lam_f)) if lam_f else 1.0
        lam_stats = {
            "lam": lam,
            "lam_per_frame_std": float(np.std(lam_f)) if lam_f else float("nan"),
            "lam_cv": float(np.std(lam_f) / (np.mean(lam_f) + 1e-9)) if lam_f else float("nan"),
            "n_frames_used": len(lam_f),
        }
    else:
        lam = float(lam)
        lam_stats = {"lam": lam, "lam_per_frame_std": 0.0, "lam_cv": 0.0, "n_frames_used": S}

    world = np.full((S, J, 3), np.nan)
    z_root = np.full(S, np.nan)
    for i in range(S):
        d = per_frame_body[i]
        k3 = np.asarray(d["pred_keypoints_3d"], dtype=np.float64)
        k2 = np.asarray(d["pred_keypoints_2d"], dtype=np.float64)
        world[i], z_root[i] = anchor_frame(k3, k2, c2w[i], K[i], depth[i], conf[i], lam)
    valid = np.isfinite(world).all(axis=(1, 2))
    return {"world_raw": world, "lam": lam, "lam_stats": lam_stats,
            "z_root": z_root, "valid": valid}


# ===========================================================================
# Temporal smoothing: 1€ (OneEuro) filter per joint per world axis.
# ===========================================================================
class OneEuro:
    """1€ (OneEuro) low-lag jitter filter (Casiez et al. 2012). Standard for pose streams."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.3,
                 d_cutoff: float = 1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def smooth_oneeuro(world: np.ndarray, fps: float, min_cutoff: float = 0.8,
                   beta: float = 0.4) -> np.ndarray:
    """world: (S,J,3) with possible NaN (gap) rows. Linearly interpolates gaps so the filter
    sees a continuous stream, then 1€-filters each (joint,axis). Returns (S,J,3).

    NOTE: interpolated rows are FABRICATED — every fabricated row is flagged by
    ``frame_visibility=False`` in the sidecar. This function never invents a detection; it
    only conditions a continuous signal for the low-lag filter (see the v1 gap policy in
    ``build_human_track``)."""
    S, J, C = world.shape
    if S == 0:
        return world.copy()
    out = world.copy()
    t = np.arange(S)
    for j in range(J):
        for c in range(C):
            col = out[:, j, c]
            good = np.isfinite(col)
            if good.sum() < 2:
                continue
            col[~good] = np.interp(t[~good], t[good], col[good])
            out[:, j, c] = col
    sm = np.empty_like(out)
    for j in range(J):
        for c in range(C):
            f = OneEuro(freq=fps, min_cutoff=min_cutoff, beta=beta)
            sm[:, j, c] = [f(v) for v in out[:, j, c]]
    return sm


# ===========================================================================
# Smoothness / scale / gravity diagnostics.
# ===========================================================================
def mean_jerk(world: np.ndarray, fps: float) -> float:
    """Mean per-joint jerk magnitude (3rd time-derivative, units/s^3). Lower = smoother."""
    if world.shape[0] < 4:
        return float("nan")
    v = np.diff(world, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    jk = np.diff(a, axis=0) * fps
    return float(np.nanmean(np.linalg.norm(jk, axis=-1)))


def bone_length_cv(world: np.ndarray) -> dict:
    """World-frame bone lengths should be constant over time; their CV measures combined
    anchoring+HMR noise (a warp check)."""
    cvs = []
    for a, b in BODY_EDGES:
        if a >= world.shape[1] or b >= world.shape[1]:
            continue
        L = np.linalg.norm(world[:, a] - world[:, b], axis=-1)
        L = L[np.isfinite(L)]
        if len(L) > 3 and L.mean() > 1e-6:
            cvs.append(L.std() / L.mean())
    if not cvs:
        return {"bone_length_cv_mean": float("nan"), "bone_length_cv_max": float("nan")}
    return {"bone_length_cv_mean": float(np.mean(cvs)),
            "bone_length_cv_max": float(np.max(cvs))}


def estimate_gravity_down(poses: np.ndarray) -> np.ndarray:
    """Heuristic world 'down' = mean of each camera's +Y axis in world (cameras are ~upright
    on a handheld walkthrough, and OpenCV camera +Y is image-down ≈ gravity-down). NOT a true
    gravity solve — a display/alignment hint. Returns unit (3,). Floor-plane fit is the fix."""
    c2w = np.asarray(poses, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[0] == 0:
        return np.array([0.0, 1.0, 0.0])
    ydirs = c2w[:, :3, 1]                       # camera +Y in world, per frame
    d = ydirs.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-9)


# ===========================================================================
# Serialization to the dynamics/human/ sidecar.
# ===========================================================================
def _finite_scalar(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite_vec_or_none(a: Any) -> Optional[list]:
    """A finite list, or None if any entry is non-finite / missing (honest gap)."""
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr.tolist()


def build_human_track(anchored: dict, per_frame_body: list[dict], frames: list[int],
                      fps: float, person_id: int = 0, smooth: bool = True,
                      min_cutoff: float = 0.8, beta: float = 0.4,
                      gravity_world: Optional[np.ndarray] = None,
                      poses: Optional[np.ndarray] = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Assemble one ``HumanTrack`` JSONL line (dict) + return (line, world_raw, world_sm).

    v1 GAP POLICY (no interpolation fabrication):
      * ``joints_world[i]`` / ``root_world[i]`` are ``null`` on frames that did not anchor
        (no detection, or root-depth sampling failed) — never fabricated.
      * ``frame_visibility[i]`` is False on those frames.
      * ``joints_world_smoothed`` (optional) IS gap-filled to feed the 1€ filter, but every
        filled row is exactly the ``frame_visibility=False`` set, so consumers can drop them.
    """
    world_raw = anchored["world_raw"]                 # (S,J,3), NaN on gaps
    valid = anchored["valid"]                          # (S,)
    lam = anchored["lam"]
    S = world_raw.shape[0]

    world_sm = smooth_oneeuro(world_raw, fps=fps, min_cutoff=min_cutoff, beta=beta) \
        if (smooth and S > 0) else None

    if gravity_world is None and poses is not None:
        gravity_world = estimate_gravity_down(poses)
    grav = None if gravity_world is None else np.asarray(gravity_world, dtype=np.float64)

    root_raw = root_of(world_raw) if S > 0 else np.zeros((0, 3))

    joints_world = [_finite_vec_or_none(world_raw[i]) if valid[i] else None for i in range(S)]
    root_world = [_finite_vec_or_none(root_raw[i]) if valid[i] else None for i in range(S)]
    frame_visibility = [bool(valid[i]) for i in range(S)]

    # detector-score proxy (3DB has no calibrated per-joint confidence). Use a per-frame
    # ``det_score`` if the body results carry one, else null.
    frame_confidence = [_finite_scalar(per_frame_body[i].get("det_score")) for i in range(S)]
    bbox_2d = [_finite_vec_or_none(per_frame_body[i].get("bbox")) for i in range(S)]

    valid_frames = [int(frames[i]) for i in range(S) if valid[i]]
    line = {
        "schema_version": DYNAMICS_HUMAN_SCHEMA_VERSION,
        "person_id": int(person_id),
        "body_model": BODY_MODEL,
        "skeleton": SKELETON,
        "num_joints": int(world_raw.shape[1]) if S > 0 else NUM_JOINTS,
        "frames": [int(f) for f in frames],               # the SHARED CLOCK (global keyframe idx)
        "joints_world": joints_world,                      # (M,J,3) SLAM units, null on gaps
        "root_world": root_world,                          # (M,3), null on gaps
        "frame_visibility": frame_visibility,              # (M,) anchored on this keyframe?
        "frame_confidence": frame_confidence,              # (M,) detector-score proxy, null if none
        "bbox_2d": bbox_2d,                                # (M,4) [x0,y0,x1,y1] px, null on gaps
        "lambda_slam_per_m": _finite_scalar(lam),          # SLAM units per meter (see Scale)
        "lambda_stats": {k: _finite_scalar(v) if k != "n_frames_used" else int(v)
                         for k, v in anchored["lam_stats"].items()},
        "gravity_world": None if grav is None else _finite_vec_or_none(grav),
        "smoothing": ({"method": "one_euro", "min_cutoff": float(min_cutoff),
                       "beta": float(beta), "fps": float(fps)} if world_sm is not None else None),
        "first_frame": (min(valid_frames) if valid_frames else -1),
        "last_frame": (max(valid_frames) if valid_frames else -1),
        "mhr_ref": f"p{int(person_id)}",
        "up_to_scale": True,
        "gravity_aligned": False,
        "units_note": (
            "joints_world are SLAM world units (up-to-scale); divide by lambda_slam_per_m for "
            "meters. World is NOT gravity-aligned (+Y ≈ down); use gravity_world to align."
        ),
    }
    if world_sm is not None:
        line["joints_world_smoothed"] = [_finite_vec_or_none(world_sm[i]) for i in range(S)]
    return line, world_raw, (world_sm if world_sm is not None else world_raw)


def _mhr_params_npz(per_frame_body: list[dict], person_id: int) -> dict:
    """Pack per-frame MHR parameter arrays -> ``{p<id>/<name>: (M,...) f32}`` (NaN on gaps).

    Only keys actually present across the frames are emitted. ``pred_keypoints_3d`` is stored
    as ``pred_keypoints_3d_cam`` (root-relative meters, camera frame) so consumers can
    re-anchor with their own scale."""
    S = len(per_frame_body)
    pref = f"p{int(person_id)}"
    out: dict = {}

    # keypoints_3d_cam (kept under a clarifying name)
    if S and per_frame_body[0].get("pred_keypoints_3d") is not None:
        J = int(np.asarray(per_frame_body[0]["pred_keypoints_3d"]).shape[0])
        arr = np.full((S, J, 3), np.nan, dtype=np.float32)
        for i in range(S):
            v = per_frame_body[i].get("pred_keypoints_3d")
            if v is not None:
                arr[i] = np.asarray(v, dtype=np.float32).reshape(J, 3)
        out[f"{pref}/pred_keypoints_3d_cam"] = arr

    for key in MHR_PARAM_KEYS:
        present = [np.asarray(d[key], dtype=np.float32).reshape(-1)
                   for d in per_frame_body if d.get(key) is not None]
        if not present:
            continue
        dim = present[0].shape[0]
        arr = np.full((S, dim), np.nan, dtype=np.float32)
        for i in range(S):
            v = per_frame_body[i].get(key)
            if v is not None:
                vv = np.asarray(v, dtype=np.float32).reshape(-1)
                if vv.shape[0] == dim:
                    arr[i] = vv
        out[f"{pref}/{key}"] = arr
    return out


def write_human_sidecar(
    out_dir: str,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    per_frame_body: list[dict],
    frames: Optional[Iterable[int]] = None,
    fps: float = 10.0,
    person_id: int = 0,
    lam: Optional[float] = None,
    smooth: bool = True,
    min_cutoff: float = 0.8,
    beta: float = 0.4,
    gravity_world: Optional[np.ndarray] = None,
) -> dict:
    """Write ``<out_dir>/dynamics/human/{human_tracks.jsonl, mhr_params.npz}`` (v1).

    ``out_dir`` is the scan export base (``export/<scan_id>/``). Additive + default-off: only
    called when human-motion results are supplied, so a static-only export stays untouched.

    v1 SCOPE (documented in ``docs/dynamics-human-export.md``):
      * **single-person** — ``per_frame_body`` is the ONE tracked person per frame (largest
        box, chosen upstream); one JSONL line is written.
      * **gap entries, no fabrication** — undetected/unanchored frames get ``null`` joints and
        ``frame_visibility=False`` (see ``build_human_track``).
      * **MHR + world joints only** (SAM License, commercial-clean); MHR→SMPL-X is a
        documented consumer-side conversion, never applied here.

    Returns a summary dict (paths + counts + λ + jerk metrics).
    """
    S = len(per_frame_body)
    if frames is None:
        frames = list(range(S))
    frames = [int(f) for f in frames]
    if len(frames) != S:
        raise ValueError(f"frames/per_frame_body length mismatch: {len(frames)} vs {S}")

    poses = np.asarray(poses, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    for name, arr, want in (("poses", poses, (S, 4, 4)), ("intrinsics", intrinsics, (S, 3, 3))):
        if arr.shape != want:
            raise ValueError(f"{name} shape {arr.shape} != expected {want}")

    anchored = anchor_sequence(poses, intrinsics, depth, depth_conf, per_frame_body, lam=lam)

    if gravity_world is None:
        gravity_world = estimate_gravity_down(poses)
    line, world_raw, world_sm = build_human_track(
        anchored, per_frame_body, frames, fps=fps, person_id=person_id, smooth=smooth,
        min_cutoff=min_cutoff, beta=beta, gravity_world=gravity_world,
    )

    human_dir = os.path.join(out_dir, "dynamics", "human")
    os.makedirs(human_dir, exist_ok=True)
    jsonl_path = os.path.join(human_dir, "human_tracks.jsonl")
    npz_path = os.path.join(human_dir, "mhr_params.npz")

    with open(jsonl_path, "w") as fh:
        fh.write(json.dumps(line, allow_nan=False) + "\n")   # allow_nan=False: fail loud on NaN

    npz_arrays = _mhr_params_npz(per_frame_body, person_id)
    np.savez_compressed(npz_path, **npz_arrays)

    valid = anchored["valid"]
    return {
        "human_tracks_path": jsonl_path,
        "mhr_params_path": npz_path,
        "frames": S,
        "frames_anchored": int(valid.sum()),
        "person_ids": [int(person_id)],
        "lambda_slam_per_m": _finite_scalar(anchored["lam"]),
        "lambda_stats": anchored["lam_stats"],
        "gravity_world": _finite_vec_or_none(gravity_world),
        "jerk_raw": mean_jerk(world_raw[:, :17], fps),
        "jerk_smoothed": (mean_jerk(world_sm[:, :17], fps) if smooth else None),
        "bone_length_cv": bone_length_cv(world_raw),
        "z_root_range": [_finite_scalar(np.nanmin(anchored["z_root"])),
                         _finite_scalar(np.nanmax(anchored["z_root"]))],
        "schema_version": DYNAMICS_HUMAN_SCHEMA_VERSION,
    }


# ===========================================================================
# Body-results adapter: (S-major dict of arrays) -> list[per-frame dict].
# ===========================================================================
def body_results_to_per_frame(data: Any) -> list[dict]:
    """Turn a SAM-3D-Body results npz/dict (keys are S-major arrays, produced by
    ``modal_dynamics_human.py``) into the per-frame list ``anchor_sequence`` consumes.

    Gap frames (no detection) keep their slot with NaN keypoints, so the shared clock is
    preserved. ``pred_keypoints_3d``/``pred_keypoints_2d`` are required; all MHR params are
    optional and passed through when present."""
    kp3 = np.asarray(data["pred_keypoints_3d"])
    S = int(kp3.shape[0])
    passthrough = ("pred_keypoints_3d", "pred_keypoints_2d", "bbox", "det_score") + MHR_PARAM_KEYS
    frames: list[dict] = []
    for i in range(S):
        d: dict = {}
        for k in passthrough:
            if k in data:
                try:
                    d[k] = np.asarray(data[k])[i]
                except (IndexError, KeyError):
                    continue
        frames.append(d)
    return frames


# ===========================================================================
# Self-test: synthetic camera + STATIC world skeleton -> anchor back. Ground-truth check of
# the convention + scale math, independent of any real weights or SLAM run. Mirrors EXP-2's
# anchor.py::_selftest. (The unit tests in tests/test_export_dynamics_human.py assert this.)
# ===========================================================================
def synthetic_fixture(seed: int = 0, S: int = 40, lam_true: float = 1.7,
                      joint_noise_m: float = 0.01):
    """Build (poses, intrinsics, depth, depth_conf, per_frame_body, world_gt) for a STATIC
    person seen by a moving synthetic camera, in the documented 3DB output format.
    ``joint_noise_m`` = per-joint gaussian noise (meters) on the root-relative keypoints;
    set 0.0 for an exact round-trip (e.g. the no-axis-flip reprojection check)."""
    rng = np.random.default_rng(seed)
    J = NUM_JOINTS
    H, W = 416, 624
    fx = fy = 380.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], float)
    Ks = np.repeat(K[None], S, axis=0)

    # crude but structured upright rest skeleton (root at 0; +Y down => head negative y).
    rest = rng.normal(0, 0.25, (J, 3)).astype(float)
    rest[:, 1] -= 0.6
    rest = rest - root_of(rest)

    world_gt = (lam_true * rest) + np.array([0.4, -0.2, 3.0])       # STATIC in world
    poses = np.zeros((S, 4, 4))
    depth = np.zeros((S, H, W), np.float32)
    conf = np.ones((S, H, W), np.float32)
    per_frame = []
    for i in range(S):
        ang = 0.15 * np.sin(2 * np.pi * i / S)
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        cpos = np.array([0.6 * np.sin(2 * np.pi * i / S), 0.05 * i, 0.0])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = cpos
        poses[i] = M
        w2c = np.linalg.inv(M)
        cam = world_gt @ w2c[:3, :3].T + w2c[:3, 3]
        uv = project(cam, K)
        root_cam = 0.5 * (cam[ROOT_IDX[0]] + cam[ROOT_IDX[1]])
        depth[i, :] = root_cam[2]
        rr_m = (cam - root_cam) / lam_true
        if joint_noise_m:
            rr_m += rng.normal(0, joint_noise_m, rr_m.shape)        # joint noise (meters)
        cam_t = root_cam.copy()
        cam_t[2] *= 0.7                                             # WRONG monocular tz
        bb = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
        per_frame.append({"pred_keypoints_3d": rr_m.astype(np.float32),
                          "pred_keypoints_2d": uv.astype(np.float32),
                          "pred_cam_t": cam_t.astype(np.float32),
                          "bbox": bb.astype(np.float32), "det_score": np.float32(0.99)})
    return poses, Ks, depth, conf, per_frame, np.repeat(world_gt[None], S, axis=0)


def _selftest(seed: int = 0) -> bool:
    lam_true = 1.7
    poses, Ks, depth, conf, per_frame, world_gt_seq = synthetic_fixture(seed, lam_true=lam_true)
    res = anchor_sequence(poses, Ks, depth, conf, per_frame)
    world = res["world_raw"]
    err = np.linalg.norm(world - world_gt_seq, axis=-1)
    traj_std = np.nanstd(world, axis=0).mean()
    print(f"[selftest] λ_true={lam_true}  λ_est={res['lam']:.3f}  cv={res['lam_stats']['lam_cv']:.3f}")
    print(f"[selftest] mean world joint err = {np.nanmean(err):.4f} u  (height ~{lam_true*1.6:.2f} u)")
    print(f"[selftest] static-person world-traj std (want ~0) = {traj_std:.4f} u")
    sm = smooth_oneeuro(world, fps=10.0)
    print(f"[selftest] jerk raw={mean_jerk(world,10.0):.4f}  smoothed={mean_jerk(sm,10.0):.4f} u/s^3")
    ok = (abs(res["lam"] - lam_true) / lam_true < 0.1) and (np.nanmean(err) < 0.1 * lam_true * 1.6)
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(0 if _selftest() else 1)
