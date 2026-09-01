"""GT-free scan-confidence scoring.

EXP-36 measured a reconstruction failure class (long, loopy, wide-FOV robot
tours) whose signature is *long-horizon scale/orientation drift with sound local
geometry*. Nothing in the live pipeline caught it — every run "succeeded" and
produced artifacts. This module is the productized lesson: score every
reconstruction from signals that need NO ground truth, and let callers gate
exports/reports on the result.

Signals (each validated to discriminate on EXP-36 cells; see tests):
  step_trend_ratio    median per-frame translation step, last fifth of the
                      trajectory over first fifth. Progressive monocular scale
                      collapse drives this far below 1 (bigoffice ~0.03).
  step_cv             coefficient of variation of per-frame steps — chaotic
                      scale handling inflates this even when the trend is flat.
  speed_jump_rate     fraction of steps > `jump_factor` x rolling median —
                      graph "teleports" from bad junctions/loop factors.
  sign_repair_rate    improper rotations / poses (SL(4) decomposition
                      degeneracy; EXP-36: 15% on the collapsed cell, 0 elsewhere).
                      SEMANTICS (fixed 2026-07-24): this is the RAW
                      improper-rotation rate — poses whose camera matrix had
                      det(R) < 0 before ANY repair — independent of which layer
                      repairs it. The name is historical: the producers used to
                      count their own local repair, which went vacuously 0 once
                      core started repairing inside decompose_camera (3ea5a03),
                      silently killing this signal. Counting the event instead of
                      the reaction is what keeps the EXP-36 calibration below
                      meaningful. Producers pass None when they could not
                      measure it; the gate then skips the signal rather than
                      scoring a fake 0. See server/oreos/recordings/pose_qc.py.
  keep_ratio          keyframes kept / frames offered (starved input signal,
                      e.g. dark or motionless footage; drone cell kept 6%).

Thresholds are INITIAL, chosen to separate EXP-36's measured negatives from
clean synthetic/handheld-class trajectories. They are conservative in the
"review" direction: unknown patterns should land in NEEDS_REVIEW, not
HIGH_CONFIDENCE. Recalibrate with real pilot positives before hard-gating.

KNOWN LIMITATION (measured, regression-tested): these signals detect
*catastrophic* failure shapes (progressive scale collapse, teleports, decomposition
degeneracy) but NOT moderate uniform long-horizon drift — EXP-36's china_office
cell drifted 27.7%-of-extent vs the robot's odometry yet scores high_confidence
here, because uniform drift leaves per-step statistics clean. Catching that class
GT-free needs v2 signals (loop-closure return-gap, per-submap VGGT scale-chain
consistency). Do not present high_confidence as "verified accurate"; it means
"no known failure signature detected".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class ConfidenceLevel(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    NEEDS_REVIEW = "needs_review"
    OUT_OF_CLASS = "out_of_class"


@dataclass
class ScanConfidence:
    level: ConfidenceLevel
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def export_allowed(self) -> bool:
        """Hard gate for scale-dependent exports (Isaac, metric claims)."""
        return self.level is not ConfidenceLevel.OUT_OF_CLASS


# Initial thresholds (see module docstring). Names appear in `reasons` verbatim.
THRESHOLDS = {
    "step_trend_ratio_low": 0.4,  # last/first-fifth median step below this => collapse
    "step_trend_ratio_high": 2.5,  # ...above this => inflation
    "step_cv_review": 1.2,
    "step_cv_fail": 2.5,
    "speed_jump_rate_review": 0.02,
    "speed_jump_rate_fail": 0.08,
    # Calibrated on EXP-36 against the pre-3ea5a03 core; still valid because the
    # producers now count the raw event, not their own repair (see docstring).
    "sign_repair_rate_review": 0.01,
    "sign_repair_rate_fail": 0.05,
    "keep_ratio_review": 0.15,
    "min_poses": 30,
}


def load_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(ts, positions Nx3) from a TUM file, comments skipped.

    The Nx3 shape is part of the contract and must hold for N == 0 too: a bare
    ``np.asarray([])`` is 1-D, and _trajectory_metrics' ``norm(..., axis=1)`` then
    raises AxisError instead of returning a refusal. That crash is reachable in
    production — modal_recon.py writes results/est_tum.txt unconditionally, so a
    recon that produced no poses leaves a zero-row file, and the QC stage blew up
    on it *before* writing confidence.json, report.html or EXPORT_REFUSED.md. The
    gate failed open in precisely the "recon produced nothing" case.
    """
    ts, pos = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = line.split()
        if len(v) >= 8:
            ts.append(float(v[0]))
            pos.append([float(v[1]), float(v[2]), float(v[3])])
    if not ts:
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(ts, dtype=float), np.asarray(pos, dtype=float).reshape(-1, 3)


def _trajectory_metrics(ts: np.ndarray, pos: np.ndarray, jump_factor: float = 8.0) -> dict:
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)  # belt-and-braces: never 1-D
    ts = np.asarray(ts, dtype=float).reshape(-1)
    if len(ts) < 2:  # 0 or 1 poses: no steps exist, so no statistic does either
        return {"n_poses": int(len(ts)), "insufficient": True}
    order = np.argsort(ts)
    ts, pos = ts[order], pos[order]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    steps = steps[np.isfinite(steps)]
    n = len(steps)
    if n < 10:
        return {"n_poses": len(ts), "insufficient": True}

    fifth = max(5, n // 5)
    first = float(np.median(steps[:fifth]))
    last = float(np.median(steps[-fifth:]))
    trend = last / first if first > 1e-12 else math.inf

    mean = float(steps.mean())
    cv = float(steps.std() / mean) if mean > 1e-12 else math.inf

    # rolling-median jump detection (window 15, centered)
    k = 15
    pad = np.pad(steps, (k // 2, k // 2), mode="edge")
    roll = np.array([np.median(pad[i : i + k]) for i in range(n)])
    jumps = steps > jump_factor * np.maximum(roll, 1e-12)
    jump_rate = float(jumps.mean())

    return {
        "n_poses": int(len(ts)),
        "span_s": float(ts[-1] - ts[0]),
        "step_trend_ratio": trend,
        "step_cv": cv,
        "speed_jump_rate": jump_rate,
        "insufficient": False,
    }


def compute_confidence(
    est_tum_path: str | Path,
    run_stats: dict | None = None,
) -> ScanConfidence:
    """Score a reconstruction from its trajectory + optional solver stats.

    run_stats (all optional): n_pose_sign_repairs, n_poses_written,
    frames_kept, frames_total — the harness/streaming layer exports these.
    """
    t = THRESHOLDS
    ts, pos = load_tum(est_tum_path)
    m = _trajectory_metrics(ts, pos)
    reasons: list[str] = []
    review = False
    fail = False

    n_poses = int(m.get("n_poses", 0))
    if m.get("insufficient") or n_poses < t["min_poses"]:
        reason = (
            "no poses in the trajectory file — reconstruction produced nothing"
            if n_poses == 0
            else f"too few poses ({n_poses} < {t['min_poses']})"
        )
        return ScanConfidence(ConfidenceLevel.OUT_OF_CLASS, [reason], m)

    if m["step_trend_ratio"] < t["step_trend_ratio_low"] or m["step_trend_ratio"] > t[
        "step_trend_ratio_high"
    ]:
        fail = True
        reasons.append(
            f"scale drift: step_trend_ratio {m['step_trend_ratio']:.3f} outside "
            f"[{t['step_trend_ratio_low']}, {t['step_trend_ratio_high']}]"
        )
    if m["step_cv"] > t["step_cv_fail"]:
        fail = True
        reasons.append(f"step_cv {m['step_cv']:.2f} > {t['step_cv_fail']}")
    elif m["step_cv"] > t["step_cv_review"]:
        review = True
        reasons.append(f"step_cv {m['step_cv']:.2f} > {t['step_cv_review']}")

    if m["speed_jump_rate"] > t["speed_jump_rate_fail"]:
        fail = True
        reasons.append(f"speed_jump_rate {m['speed_jump_rate']:.3f} > {t['speed_jump_rate_fail']}")
    elif m["speed_jump_rate"] > t["speed_jump_rate_review"]:
        review = True
        reasons.append(
            f"speed_jump_rate {m['speed_jump_rate']:.3f} > {t['speed_jump_rate_review']}"
        )

    if run_stats:
        repairs = run_stats.get("n_pose_sign_repairs")
        n_poses = run_stats.get("n_poses_written") or m["n_poses"]
        if repairs is not None and n_poses:
            rate = repairs / n_poses
            m["sign_repair_rate"] = rate
            if rate > t["sign_repair_rate_fail"]:
                fail = True
                reasons.append(f"sign_repair_rate {rate:.3f} > {t['sign_repair_rate_fail']}")
            elif rate > t["sign_repair_rate_review"]:
                review = True
                reasons.append(f"sign_repair_rate {rate:.3f} > {t['sign_repair_rate_review']}")
        kept, total = run_stats.get("frames_kept"), run_stats.get("frames_total")
        if kept is not None and total:
            keep = kept / total
            m["keep_ratio"] = keep
            if keep < t["keep_ratio_review"]:
                review = True
                reasons.append(f"keep_ratio {keep:.3f} < {t['keep_ratio_review']}")

    level = (
        ConfidenceLevel.OUT_OF_CLASS
        if fail
        else ConfidenceLevel.NEEDS_REVIEW
        if review
        else ConfidenceLevel.HIGH_CONFIDENCE
    )
    return ScanConfidence(level, reasons, m)
