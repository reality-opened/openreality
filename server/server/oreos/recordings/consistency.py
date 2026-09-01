"""Self-contained trajectory consistency scoring (numpy-only, no evo).

Compares the reconstruction's trajectory against the recording's own odometry
reference (when one exists): nearest-timestamp association, Umeyama Sim(3)
alignment, ATE as metres AND as % of extent / % of path, plus the scale-drift
indicators EXP-36 showed matter (windowed scale CV, step statistics).

This is the PRODUCT scorer — deliberately dependency-light. The experiment-grade
frozen scorer (evo-based, platform/experiments/slam_bench/metrics.py) remains the
authority for preregistered claims; numbers here are for operator perspective.
Ported from EXP-36's analyze_cell.py (validated against the frozen scorer there).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ts, positions Nx3, quats Nx4 xyzw), sorted by timestamp; comments skipped.

    Sorting is NOT cosmetic: associate() below binary-searches the reference
    timestamps with np.searchsorted, which silently returns nonsense on unsorted
    input — mis-paired samples then flow into umeyama(), whose scale factor gates
    the metric Isaac export. Nothing upstream guarantees order: est_tum.txt is
    written submap-by-submap by modal_recon.py, and a loop closure can reorder
    frames. server/qc/confidence.py already argsorts; this makes the two readers
    agree.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    if not rows:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4))
    a = np.asarray(rows)
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def associate(ts_a: np.ndarray, ts_b: np.ndarray, tol: float = 0.05):
    """Nearest-timestamp pairing. REQUIRES ts_b sorted ascending (np.searchsorted);
    load_tum() guarantees that for both trajectories it returns."""
    ia, ib = [], []
    for i, t in enumerate(ts_a):
        j = int(np.searchsorted(ts_b, t))
        best, bestd = None, tol
        for k in (j - 1, j):
            if 0 <= k < len(ts_b) and abs(ts_b[k] - t) <= bestd:
                best, bestd = k, abs(ts_b[k] - t)
        if best is not None:
            ia.append(i)
            ib.append(best)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) src->dst minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def path_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def score_consistency(
    est_tum: str | Path,
    reference_tum: str | Path | None,
    out_json: str | Path | None = None,
    window: int = 20,
) -> dict:
    """Score est vs reference; degrades gracefully when no/degenerate reference."""
    ts_e, p_e, _ = load_tum(est_tum)
    out: dict = {"n_est": int(len(ts_e)), "reference_available": False}

    ref_ok = False
    if reference_tum and Path(reference_tum).exists():
        ts_g, p_g, _ = load_tum(reference_tum)
        spread = float(np.ptp(p_g, axis=0).max()) if len(p_g) else 0.0
        out["n_reference"] = int(len(ts_g))
        out["reference_spread_m"] = round(spread, 4)
        ref_ok = len(p_g) >= 10 and spread > 0.05

    if ref_ok:
        ie, ig = associate(ts_e, ts_g)
        out["n_matched"] = int(len(ie))
        if len(ie) >= 10:
            src, dst = p_e[ie], p_g[ig]
            try:
                s, R, t = umeyama(src, dst)
            except np.linalg.LinAlgError:
                out["warning"] = "alignment degenerate"
            else:
                aligned = (s * (R @ src.T)).T + t
                resid = np.linalg.norm(aligned - dst, axis=1)
                extent = float(
                    max(np.ptp(dst[:, 0]), np.ptp(dst[:, 1]), np.ptp(dst[:, 2]))
                )
                gt_path = path_length(p_g)
                ate = float(np.sqrt((resid**2).mean()))
                out.update(
                    reference_available=True,
                    ate_rmse_m=round(ate, 4),
                    ate_median_m=round(float(np.median(resid)), 4),
                    extent_m=round(extent, 3),
                    gt_path_length_m=round(gt_path, 3),
                    ate_pct_extent=round(100 * ate / extent, 2) if extent else None,
                    ate_pct_path=round(100 * ate / gt_path, 2) if gt_path else None,
                    umeyama_scale=round(s, 4),
                    aligned_positions=aligned[:: max(1, len(aligned) // 2000)].round(4).tolist(),
                    reference_positions=dst[:: max(1, len(dst) // 2000)].round(4).tolist(),
                )
                # windowed scale CV — EXP-36's headline drift indicator
                if len(src) > window:
                    scales = []
                    for i in range(0, len(src) - window + 1, max(1, window // 2)):
                        try:
                            sw, _, _ = umeyama(src[i : i + window], dst[i : i + window])
                            scales.append(sw)
                        except np.linalg.LinAlgError:
                            continue
                    if scales:
                        arr = np.asarray(scales)
                        m = float(arr.mean())
                        out["windowed_scale_cv"] = (
                            round(float(arr.std() / abs(m)), 4) if m else None
                        )

    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=2))
    return out
