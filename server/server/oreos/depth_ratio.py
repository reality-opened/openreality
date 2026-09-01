"""Depth → metric-scale ratio core, sensor-agnostic (WORLD-TRANSFORM-CONTRACT "Units & scale").

ONE implementation of "how many metres is one SLAM unit, measured against a hardware depth
sensor". Extracted verbatim from ``scripts/demo_ingest_gemini.py`` (which was written for an
Orbbec Gemini 2 capture and is still its only CLI) so a second depth source — the TUM RGB-D
benchmark, see ``modal_tum_depth_anchor.py`` — reuses the same math instead of forking it.
Nothing here knows about a capture layout, a broker, HTTP, or a scan id.

The method, unchanged from the Gemini glue:

  1. z-buffer-project the persisted WORLD cloud into a keyframe's ``c2w`` pose on that
     keyframe's own pixel grid, giving per-pixel SLAM depth (SLAM units);
  2. block-median the hardware depth PNG onto the same coarse grid, in metres;
  3. hand both to core's ``vggt_slam.metric_anchor.frame_ratio`` (EXP-12, pure numpy) for a
     robust per-frame ``metres per SLAM unit``;
  4. median across keyframes, gated on ``CoV > DEFAULT_WITHIN_COV`` (0.15) — over the gate the
     caller must REFUSE to auto-anchor rather than apply a number it cannot stand behind.

Step 3 is core's, not ours: the ratio is never re-derived here. Steps 1/2/4 are.

Dependency posture: numpy at module level, nothing else — the same posture as core's
``metric_anchor``, so this module is importable from a light Modal image, from CI, and by
file-path load from the Gemini CLI's dependency-thin capture venv. ``cv2`` is imported lazily
inside :func:`read_depth_png` only.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

#: Fewer usable per-frame ratios than this and the median is not a measurement.
MIN_RATIO_FRAMES = 3
#: Principal-axis fit for the anchor point pair runs on at most this many points.
ANCHOR_CLOUD_SUBSAMPLE = 200_000
#: NCC acceptance threshold when matching a stored keyframe back to a source frame.
MATCH_MIN_CORR = 0.55


# ---------------------------------------------------------------------------
# core's metric_anchor, without depending on how core got installed
# ---------------------------------------------------------------------------


def load_metric_anchor_module():
    """The core pure-numpy anchor fns, install-first with a file-load fallback.

    Ladder: (1) installed ``openreality-core`` package (``vggt_slam/__init__`` is empty, so
    this pulls numpy only — verified against core@v2.2.2); (2) ``CORE_METRIC_ANCHOR_PY`` env,
    or a sibling ``core/vggt_slam/metric_anchor.py`` under any ancestor directory of this file
    (the platform super-repo checkout) loaded directly by file — that dodges packaging
    entirely, and the module imports only numpy at top level; (3) an actionable error.
    Returns the module.

    The ancestor WALK (rather than a fixed ``parents[N]``) is deliberate: this module is
    imported from ``server/oreos/``, from ``server/scripts/`` via re-export, and from a Modal
    container where the tree is grafted at ``/root/project`` — one hard-coded depth cannot be
    right in all three.
    """
    try:
        import vggt_slam.metric_anchor as ma  # type: ignore

        return ma
    except ImportError:
        pass
    candidates = []
    env_path = os.environ.get("CORE_METRIC_ANCHOR_PY")
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "core" / "vggt_slam" / "metric_anchor.py")
    for cand in candidates:
        if cand.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("_depth_ratio_metric_anchor", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # numpy-only at module level
            return mod
    raise ImportError(
        "vggt_slam.metric_anchor unavailable: pip install "
        '"openreality-core @ git+https://github.com/reality-opened/core@v2.2.2" '
        "(needs GH auth), or set CORE_METRIC_ANCHOR_PY to a core checkout's "
        "vggt_slam/metric_anchor.py"
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def parse_cloud_ply(data: bytes):
    """Positions from a binary-little-endian PLY (the exact schema the server's
    ``build_ply_bytes`` emits: float32 x/y/z + uchar rgb; tolerant of extra
    properties via the header's declared order). Returns ``(N, 3) float64``."""
    header_end = data.find(b"end_header\n")
    if not data.startswith(b"ply") or header_end < 0:
        raise ValueError("not a PLY file")
    header = data[:header_end].decode("ascii", "replace").splitlines()
    if not any(l.strip() == "format binary_little_endian 1.0" for l in header):
        raise ValueError("unsupported PLY format (binary_little_endian 1.0 only)")
    count = None
    fields: list[tuple[str, str]] = []
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
    }
    for line in header:
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property":
            if parts[1] not in type_map:
                raise ValueError(f"unsupported PLY property type {parts[1]!r}")
            fields.append((parts[2], type_map[parts[1]]))
    if count is None or not fields:
        raise ValueError("PLY header missing vertex element/properties")
    for axis in ("x", "y", "z"):
        if axis not in [n for n, _ in fields]:
            raise ValueError(f"PLY lacks '{axis}' property")
    dtype = np.dtype(fields)
    body = data[header_end + len(b"end_header\n"):]
    need = count * dtype.itemsize
    if len(body) < need:
        raise ValueError(f"PLY body truncated ({len(body)} < {need} bytes)")
    vertex = np.frombuffer(body[:need], dtype=dtype)
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def zbuffer_project(points_cam, fx: float, fy: float, cx: float, cy: float,
                    width: int, height: int, downsample: int):
    """Nearest-depth rasterization of CAMERA-frame points (OpenCV convention,
    +z forward) onto a ``downsample``-coarsened pixel grid. Returns
    ``(depth (H', W') float64 with NaN holes, valid bool mask)``."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    gw = width // downsample
    gh = height // downsample
    depth = np.full((gh, gw), np.inf, dtype=np.float64)
    z = pts[:, 2]
    front = z > 1e-9
    if front.any():
        pts = pts[front]
        z = z[front]
        u = np.floor((fx * pts[:, 0] / z + cx) / downsample).astype(np.int64)
        v = np.floor((fy * pts[:, 1] / z + cy) / downsample).astype(np.int64)
        ok = (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        if ok.any():
            flat = v[ok] * gw + u[ok]
            np.minimum.at(depth.reshape(-1), flat, z[ok])
    valid = np.isfinite(depth)
    depth = np.where(valid, depth, np.nan)
    return depth, valid


def block_median_depth(depth_raw_u16, downsample: int, mm_per_unit: float):
    """Hardware depth (uint16, 0 = invalid) → block-median METRES on the same
    coarse grid as :func:`zbuffer_project` (crop to grid multiples, per-block
    nanmedian). Invalid blocks are NaN.

    ``mm_per_unit`` converts the sensor's raw integer unit to millimetres — 1.0 for a
    millimetre sensor (Gemini 2), 0.2 for TUM's 5000-units-per-metre PNGs.
    """
    d = np.asarray(depth_raw_u16)
    gh, gw = d.shape[0] // downsample, d.shape[1] // downsample
    d = d[: gh * downsample, : gw * downsample].astype(np.float64)
    d[d == 0] = np.nan
    blocks = d.reshape(gh, downsample, gw, downsample).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(gh, gw, downsample * downsample)
    import warnings

    with warnings.catch_warnings():  # all-NaN blocks are expected, not a bug
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(blocks, axis=2)
    return med * (float(mm_per_unit) / 1000.0)


def pair_ratio(points_world, c2w, camera: dict, depth_raw_u16, ma, *,
               downsample: int, max_depth_m: float, mm_per_unit: float):
    """One keyframe's metres-per-SLAM-unit, or ``None`` when too few shared pixels.

    ``camera`` is the pixel grid the DEPTH image lives in:
    ``{fx, fy, cx, cy, width, height, rotation}`` where ``rotation`` is an optional
    depth←colour rotation (Gemini 2's d2c; ``None`` for a pre-registered sensor like TUM's
    Kinect, whose depth is already reprojected into the colour frame).

    This is steps 1–3 of the module method in one call; the ratio itself is core's
    ``frame_ratio`` and is not re-derived here.
    """
    c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    w2c = np.linalg.inv(c2w)
    pts_cam = np.asarray(points_world, dtype=np.float64).reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
    rotation = camera.get("rotation")
    if rotation is not None:
        pts_cam = pts_cam @ np.asarray(rotation, dtype=np.float64)  # X_d = R^T · X_c → x·R
    slam_depth, slam_valid = zbuffer_project(
        pts_cam, float(camera["fx"]), float(camera["fy"]), float(camera["cx"]),
        float(camera["cy"]), int(camera["width"]), int(camera["height"]), int(downsample))
    metric_m = block_median_depth(depth_raw_u16, int(downsample), mm_per_unit)
    return ma.frame_ratio(slam_depth, slam_valid, metric_m, max_depth=float(max_depth_m))


def pick_anchor_points(points):
    """Two well-separated REAL cloud points: the extremes of the projection on
    the cloud's principal axis (covariance eigenvector, on a subsample for
    speed). Returns ``(a, b, separation)`` as float64 arrays."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("cloud has fewer than 2 points")
    sub = pts
    if pts.shape[0] > ANCHOR_CLOUD_SUBSAMPLE:
        step = pts.shape[0] // ANCHOR_CLOUD_SUBSAMPLE
        sub = pts[::step]
    centered = sub - sub.mean(axis=0)
    cov3 = centered.T @ centered
    _w, vecs = np.linalg.eigh(cov3)
    axis = vecs[:, -1]  # largest-eigenvalue direction
    proj = centered @ axis
    a = sub[int(np.argmin(proj))]
    b = sub[int(np.argmax(proj))]
    sep = float(np.linalg.norm(a - b))
    if sep <= 0:
        raise ValueError("degenerate cloud: zero principal-axis extent")
    return a, b, sep


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def ratio_gate(ratios: list[float], cov_fn, cov_max: float) -> dict:
    """Median-ratio + CoV gate (pure; unit-tested without network/depth).
    Returns ``{ok, ratio, cov, n}`` — ``ok=False`` means REFUSE auto-anchor."""
    arr = [float(r) for r in ratios if math.isfinite(r) and r > 0]
    if len(arr) < MIN_RATIO_FRAMES:
        return {"ok": False, "ratio": None, "cov": None, "n": len(arr),
                "reason": f"only {len(arr)} usable frame ratios "
                          f"(< {MIN_RATIO_FRAMES})"}
    c = float(cov_fn(arr))
    med = float(np.median(np.asarray(arr)))
    if not math.isfinite(c) or c > cov_max:
        return {"ok": False, "ratio": med, "cov": c, "n": len(arr),
                "reason": f"CoV {c:.3f} > {cov_max:g} gate"}
    return {"ok": True, "ratio": med, "cov": c, "n": len(arr)}


# ---------------------------------------------------------------------------
# keyframe ↔ source-frame matching (alignment is MEASURED, never assumed)
# ---------------------------------------------------------------------------


def ncc(a, b) -> float:
    """Pearson correlation of two equal-shaped grayscale patches (NaN-safe)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def gray_thumb(image, size: int = 64):
    """A ``size``×``size`` float64 grayscale thumbnail — the NCC comparison unit.
    Accepts BGR/RGB (H,W,3) or already-gray (H,W)."""
    import cv2  # lazy: only the matching path needs an image codec

    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)


def best_ncc_match(query_thumb, candidate_thumbs, indices=None,
                   min_corr: float = MATCH_MIN_CORR):
    """Exhaustive best NCC match of one keyframe thumbnail against candidates.

    ``candidate_thumbs`` is an ``(N, S, S)`` stack (or a sequence of ``(S, S)`` arrays);
    ``indices`` maps candidate position → the caller's own frame id (default ``range(N)``).
    Returns ``(index, corr)`` or ``None`` when nothing correlates at least ``min_corr`` —
    an unmatched keyframe must be DROPPED, never guessed.
    """
    cands = np.asarray(candidate_thumbs, dtype=np.float64)
    if cands.ndim != 3 or cands.shape[0] == 0:
        return None
    q = np.asarray(query_thumb, dtype=np.float64).reshape(-1)
    q = q - q.mean()
    qn = float(np.sqrt((q * q).sum()))
    flat = cands.reshape(cands.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.sqrt((flat * flat).sum(axis=1))
    denom = norms * qn
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, (flat @ q) / denom, 0.0)
    pos = int(np.argmax(corr))
    best = float(corr[pos])
    if not math.isfinite(best) or best < min_corr:
        return None
    idx = int(indices[pos]) if indices is not None else pos
    return idx, best


def fed_index_for_admitted(fids, anchors, n_fed: int | None = None):
    """Which FED frame each ADMITTED SLAM keyframe was, from a few verified pairs.

    ``trajectory.npz``'s ``source_frame_id`` counts frames the disparity gate ADMITTED
    (``streaming_slam.process_loop`` only increments ``frame_count`` when optical-flow
    disparity clears ``min_disparity``), while the video feeder feeds frame ``fed × skip``
    of the source. So ``fed = fid + delta(fid)`` where ``delta`` counts rejections so far:
    non-negative, non-decreasing, and unknown a priori.

    ``anchors`` is a sequence of ``(fid, fed)`` pairs established by image matching. Between
    two anchors with the SAME delta, monotonicity forces every delta in between to be that
    value — exact, not interpolated. Across a delta step the rows in the gap are genuinely
    ambiguous; they get the lower bound and are flagged.

    Returns ``(fed (N,) int64, exact (N,) bool)`` for ``fids``.
    """
    f = np.asarray(fids, dtype=np.int64).reshape(-1)
    pairs = sorted({(int(a), int(b)) for a, b in anchors})
    if not pairs:
        raise ValueError("need at least one verified (fid, fed) anchor pair")
    a_fid = np.array([p[0] for p in pairs], dtype=np.int64)
    a_delta = np.array([p[1] - p[0] for p in pairs], dtype=np.int64)
    if np.any(np.diff(a_delta) < 0):
        raise ValueError(f"anchor deltas are not non-decreasing: {a_delta.tolist()} — the "
                         "keyframe matching is inconsistent with the admission model")
    out = np.empty(f.shape, dtype=np.int64)
    exact = np.zeros(f.shape, dtype=bool)
    for i, fid in enumerate(f):
        right = int(np.searchsorted(a_fid, fid, side="left"))
        left = right - 1
        lo = int(a_delta[left]) if left >= 0 else 0
        hi = int(a_delta[right]) if right < a_delta.size else int(a_delta[-1])
        if right < a_fid.size and a_fid[right] == fid:
            lo = hi = int(a_delta[right])
        out[i] = fid + lo
        exact[i] = lo == hi
    if n_fed is not None:
        out = np.clip(out, 0, int(n_fed) - 1)
    return out, exact


def read_depth_png(path):
    """A 16-bit single-channel depth PNG as a uint16 array (cv2 IMREAD_UNCHANGED)."""
    import cv2  # lazy: anchor-only dependency

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable depth PNG {path}")
    if img.dtype != np.uint16 or img.ndim != 2:
        raise ValueError(f"{path} is not a 16-bit single-channel depth PNG "
                         f"(dtype={img.dtype}, ndim={img.ndim})")
    return img
