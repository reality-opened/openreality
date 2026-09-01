"""TUM RGB-D benchmark adapter — the licence-clean depth source with mocap ground truth.

Feeds ``rgbd_dataset_freiburg1_room`` (CC BY 4.0, https://cvg.cit.tum.de/rgbd/dataset/) into
the sensor-agnostic ratio core in ``depth_ratio.py``. It exists because the depth→metric-scale
path had never been run against real depth data with an independent answer to check it
against; TUM supplies both (16-bit registered depth PNGs, and a 100 Hz motion-capture
trajectory in metres).

Everything here is pure numpy + stdlib and unit-tested — the Modal entrypoint
(``modal_tum_depth_anchor.py``) is the only thing that knows about scenes, volumes or clouds.

Dataset facts this module encodes (from the TUM file-format page, looked up — not assumed):

  * ``rgb.txt`` / ``depth.txt`` are ``<timestamp> <path>`` indices of two SEPARATE streams.
    The RGB and depth frames are captured at slightly different instants and there is no
    1:1 row correspondence — they must be associated by nearest timestamp
    (:func:`associate_nearest`). fr1/room: 1362 RGB, 1360 depth, worst pairing gap 18 ms.
  * The depth images ARE spatially pre-registered: "the depth images ... are reprojected
    into the frame of the color camera, ... 1:1 correspondence between pixels in the depth
    map and the color image". So no depth←colour rotation is applied (unlike the Gemini 2
    d2c case) and the COLOUR camera model is the depth image's camera model.
  * ``5000`` depth units per metre → ``MM_PER_UNIT = 0.2`` for ``block_median_depth``.
  * Two published intrinsic sets, and TUM explicitly recommends the ROS default one for the
    raw (still distorted) pre-registered frames: "We recommend to use the ROS default
    parameter set (i.e., without undistortion), as undistortion of the pre-registered depth
    images is not trivial."
  * ``groundtruth.txt`` is ``<timestamp> tx ty tz qx qy qz qw`` in metres, ~100 Hz, and its
    time span is WIDER than the sensor streams (fr1/room: 48.90 s of mocap around 45.37 s
    of RGB) — a length computed over the whole file is not the published figure.
"""

from __future__ import annotations

import numpy as np

#: TUM depth PNGs store 5000 integer units per metre.
DEPTH_UNITS_PER_METRE = 5000.0
#: ``depth_ratio.block_median_depth`` wants millimetres per raw unit.
MM_PER_UNIT = 1000.0 / DEPTH_UNITS_PER_METRE  # 0.2

#: TUM's own recommendation for the raw, still-distorted, pre-registered fr1 frames.
TUM_ROS_DEFAULT = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}
#: The fr1 calibrated set — correct for UNDISTORTED frames, which these are not.
TUM_FR1_RGB = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3}
#: fr1 radial/tangential coefficients, recorded for provenance; no undistortion is applied.
TUM_FR1_DISTORTION = (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)

RGB_SIZE = (640, 480)  # (width, height) of both the RGB and the registered depth PNGs

#: Published dataset statistic for fr1/room — the number the whole exercise is checked
#: against. https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
PUBLISHED_TRAJECTORY_LENGTH_M = 15.989


# ---------------------------------------------------------------------------
# file parsing
# ---------------------------------------------------------------------------


def read_index(path) -> tuple[np.ndarray, list[str]]:
    """``rgb.txt`` / ``depth.txt`` → ``(timestamps (N,) float64, relative paths)``.
    ``#`` comment lines and blanks are skipped; row order is preserved (it is the frame
    order the mp4 was assembled in, and therefore the mp4 frame index)."""
    ts: list[float] = []
    files: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ts.append(float(parts[0]))
            files.append(parts[1])
    return np.asarray(ts, dtype=np.float64), files


def read_groundtruth(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``groundtruth.txt`` → ``(t (N,), positions (N,3) metres, quaternions (N,4) xyzw)``."""
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            rows.append([float(v) for v in parts])
    if not rows:
        raise ValueError(f"no ground-truth rows in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


# ---------------------------------------------------------------------------
# association + interpolation
# ---------------------------------------------------------------------------


def associate_nearest(query_ts, target_ts, max_diff: float = 0.02):
    """Nearest-timestamp association (the TUM ``associate.py`` rule, 20 ms default).

    Returns ``(index (N,) int64, gap (N,) float64)``; entries whose nearest target is
    further than ``max_diff`` get index ``-1``. The RGB and depth streams are separate
    captures — this is the ONLY correct way to pair them, and a row-index pairing would be
    silently off by one from the first dropped frame onward.
    """
    q = np.asarray(query_ts, dtype=np.float64).reshape(-1)
    t = np.asarray(target_ts, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return np.full(q.shape, -1, dtype=np.int64), np.full(q.shape, np.inf)
    right = np.clip(np.searchsorted(t, q), 1, t.size - 1)
    left = right - 1
    pick = np.where(np.abs(t[left] - q) <= np.abs(t[right] - q), left, right)
    gap = np.abs(t[pick] - q)
    pick = np.where(gap <= max_diff, pick, -1)
    return pick.astype(np.int64), gap


def interpolate_positions(gt_ts, gt_pos, query_ts) -> np.ndarray:
    """Linearly interpolate the mocap positions at arbitrary times ``(M, 3)``.

    Queries outside the mocap span clamp to the endpoints (``np.interp`` behaviour); the
    caller is responsible for checking coverage — :func:`covers` exists for that.
    """
    gt_ts = np.asarray(gt_ts, dtype=np.float64).reshape(-1)
    gt_pos = np.asarray(gt_pos, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(query_ts, dtype=np.float64).reshape(-1)
    return np.stack([np.interp(q, gt_ts, gt_pos[:, i]) for i in range(3)], axis=1)


def covers(gt_ts, query_ts) -> bool:
    """True iff every query time lies inside the mocap span (no clamped extrapolation)."""
    gt_ts = np.asarray(gt_ts, dtype=np.float64)
    q = np.asarray(query_ts, dtype=np.float64)
    return bool(q.size and gt_ts.min() <= q.min() and q.max() <= gt_ts.max())


# ---------------------------------------------------------------------------
# trajectory metrics
# ---------------------------------------------------------------------------


def polyline_length(positions) -> float:
    """Sum of consecutive segment lengths — a trajectory's length AT THE SAMPLING GIVEN.

    Sampling matters and is not a detail: the same fr1/room mocap path measures 16.00 m
    summed at 100 Hz and 15.50 m summed at 2 Hz. Any comparison between two trajectories'
    lengths is only meaningful when both are sampled at the same instants.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def umeyama_sim3(src, dst) -> dict:
    """Least-squares similarity (scale, rotation, translation) mapping ``src`` onto ``dst``.

    Umeyama (1991) with scale — the standard trajectory-alignment estimator. ``src`` is our
    SLAM-unit positions, ``dst`` the mocap metres, so ``scale`` comes out as
    METRES PER SLAM UNIT: an estimate of the same quantity the depth ratio measures, from a
    completely independent source. Returns ``{scale, R, t, rmse, n}`` where ``rmse`` is the
    post-alignment ATE RMSE in metres.
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if s.shape != d.shape or s.shape[0] < 3:
        raise ValueError(f"need >=3 matched 3-D pairs, got {s.shape} vs {d.shape}")
    n = s.shape[0]
    mu_s, mu_d = s.mean(axis=0), d.mean(axis=0)
    sc, dc = s - mu_s, d - mu_d
    sigma = (dc.T @ sc) / n
    u, dvals, vt = np.linalg.svd(sigma)
    w = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # keep it a rotation, never a reflection
        w[2, 2] = -1.0
    rot = u @ w @ vt
    var_src = float((sc * sc).sum() / n)
    if var_src <= 0:
        raise ValueError("degenerate source trajectory (zero variance)")
    scale = float(np.trace(np.diag(dvals) @ w) / var_src)
    trans = mu_d - scale * (rot @ mu_s)
    resid = d - (scale * (s @ rot.T) + trans)
    rmse = float(np.sqrt((resid * resid).sum(axis=1).mean()))
    return {"scale": scale, "R": rot, "t": trans, "rmse": rmse, "n": int(n)}


# ---------------------------------------------------------------------------
# camera model
# ---------------------------------------------------------------------------


def rescale_intrinsics(fx, fy, cx, cy, from_size, to_size) -> dict:
    """Move a pinhole model between two grids related by a pure (possibly anisotropic) RESIZE.

    Needed because the reconstruction did not run on the original 640×480 frames: VGGT-Omega's
    ``balanced`` loader resizes to the patch-multiple grid nearest ``512²`` in area, which for
    a 640×480 input is 592×448 — a 0.925 × 0.9333 anisotropic scale, no crop. The no-crop half
    of that is CHECKABLE rather than assumed: a crop would push the predicted principal point
    away from the grid centre, and on this scene it sits within 1 px of it.

    Returns a ``depth_ratio.pair_ratio`` camera dict for ``to_size``.
    """
    sx = float(to_size[0]) / float(from_size[0])
    sy = float(to_size[1]) / float(from_size[1])
    return {
        "fx": float(fx) * sx, "fy": float(fy) * sy,
        "cx": float(cx) * sx, "cy": float(cy) * sy,
        "width": int(to_size[0]), "height": int(to_size[1]),
        "rotation": None,  # TUM depth is already in the colour camera's frame
    }


def balanced_grid(width: int, height: int, resolution: int = 512, patch: int = 16):
    """VGGT-Omega ``mode="balanced"`` output grid for a ``width``×``height`` input.

    The loader targets ``resolution²`` pixels at the source aspect, snapped to ``patch``
    multiples. Reproduced here (rather than imported) because ``vggt_omega`` is a GPU-image
    dependency and this has to run in a numpy-only container; it is cross-checked against the
    persisted intrinsics, whose principal points sit at the returned grid's centre.
    """
    scale = float(resolution) / float(np.sqrt(width * height))
    gw = max(patch, int(round(width * scale / patch)) * patch)
    gh = max(patch, int(round(height * scale / patch)) * patch)
    return gw, gh


def camera_from_intrinsics_row(row, recon_size, to_size=RGB_SIZE) -> dict:
    """A persisted ``[fx, fy, cx, cy]`` trajectory/frames-index row → a depth-grid camera.

    These are VGGT's OWN per-frame predictions, on the reconstruction grid. Preferring them
    over TUM's published calibration is deliberate: the world cloud was built by unprojecting
    with exactly these numbers, so re-projecting with them puts each point back on the pixel
    whose image content produced it — which is the pixel whose hardware depth we pair it with.
    Projecting with the true calibration instead would shift content radially by the
    prediction error and pair depths across that shift.
    """
    fx, fy, cx, cy = (float(v) for v in list(row)[:4])
    return rescale_intrinsics(fx, fy, cx, cy, recon_size, to_size)


def published_camera(which: str = "ros_default", to_size=RGB_SIZE) -> dict:
    """TUM's published pinhole model as a camera dict — the sensitivity cross-check for
    :func:`camera_from_intrinsics_row`. ``which`` is ``"ros_default"`` or ``"fr1"``."""
    table = {"ros_default": TUM_ROS_DEFAULT, "fr1": TUM_FR1_RGB}
    if which not in table:
        raise ValueError(f"unknown intrinsics set {which!r}; choices: {sorted(table)}")
    k = table[which]
    return {**k, "width": int(to_size[0]), "height": int(to_size[1]), "rotation": None}
