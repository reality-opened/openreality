"""Tests for server.qc.confidence.

Two layers: synthetic trajectories with known pathologies (hermetic), and real
EXP-36 outputs as regression fixtures — the collapsed bigoffice cell and the
drifting hk_village1 cell must NEVER score high_confidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from server.qc.confidence import (
    ConfidenceLevel,
    THRESHOLDS,
    compute_confidence,
)

FIXTURES = Path(__file__).parent / "fixtures" / "exp36"


def write_tum(path: Path, ts: np.ndarray, pos: np.ndarray) -> Path:
    lines = [
        f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 1" for t, p in zip(ts, pos)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def circle_trajectory(n: int = 300, radius: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0, 60, n)
    ang = np.linspace(0, 2 * np.pi, n)
    pos = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)
    return ts, pos


class TestSynthetic:
    def test_clean_circle_high_confidence(self, tmp_path: Path) -> None:
        ts, pos = circle_trajectory()
        c = compute_confidence(write_tum(tmp_path / "a.txt", ts, pos))
        assert c.level is ConfidenceLevel.HIGH_CONFIDENCE
        assert c.reasons == []
        assert c.export_allowed

    def test_scale_collapse_out_of_class(self, tmp_path: Path) -> None:
        # per-step displacement decays 30x over the run — monocular collapse shape
        n = 300
        ts = np.linspace(0, 60, n)
        decay = np.geomspace(1.0, 1 / 30.0, n - 1)
        steps = 0.05 * decay
        x = np.concatenate([[0], np.cumsum(steps)])
        pos = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
        c = compute_confidence(write_tum(tmp_path / "b.txt", ts, pos))
        assert c.level is ConfidenceLevel.OUT_OF_CLASS
        assert any("scale drift" in r for r in c.reasons)
        assert not c.export_allowed

    def test_teleports_flagged(self, tmp_path: Path) -> None:
        ts, pos = circle_trajectory()
        pos = pos.copy()
        rng = np.random.default_rng(0)
        for i in rng.choice(np.arange(50, 250), size=12, replace=False):
            pos[i:] += np.array([3.0, -2.0, 0.0])  # graph "teleport" every hit
        c = compute_confidence(write_tum(tmp_path / "c.txt", ts, pos))
        assert c.level is not ConfidenceLevel.HIGH_CONFIDENCE
        assert any("speed_jump_rate" in r for r in c.reasons)

    def test_sign_repairs_gate(self, tmp_path: Path) -> None:
        ts, pos = circle_trajectory()
        p = write_tum(tmp_path / "d.txt", ts, pos)
        stats = {"n_pose_sign_repairs": 50, "n_poses_written": 300}
        c = compute_confidence(p, run_stats=stats)
        assert c.level is ConfidenceLevel.OUT_OF_CLASS
        assert any("sign_repair_rate" in r for r in c.reasons)

    def test_unmeasurable_sign_repairs_skip_the_signal(self, tmp_path: Path) -> None:
        """Producers pass None when the raw camera matrices were unreadable.

        The gate must then say nothing about sign_repair_rate rather than score a
        fake 0 — reporting 0 for "we could not measure it" is exactly how this
        signal died once already (see server/oreos/recordings/pose_qc.py).
        """
        ts, pos = circle_trajectory()
        p = write_tum(tmp_path / "d_none.txt", ts, pos)
        c = compute_confidence(
            p, run_stats={"n_pose_sign_repairs": None, "n_poses_written": 300}
        )
        assert "sign_repair_rate" not in c.metrics
        assert not any("sign_repair_rate" in r for r in c.reasons)

    def test_starved_input_reviews(self, tmp_path: Path) -> None:
        ts, pos = circle_trajectory()
        p = write_tum(tmp_path / "e.txt", ts, pos)
        c = compute_confidence(p, run_stats={"frames_kept": 38, "frames_total": 599})
        assert c.level is ConfidenceLevel.NEEDS_REVIEW
        assert any("keep_ratio" in r for r in c.reasons)

    def test_too_few_poses(self, tmp_path: Path) -> None:
        ts, pos = circle_trajectory(n=10)
        c = compute_confidence(write_tum(tmp_path / "f.txt", ts, pos))
        assert c.level is ConfidenceLevel.OUT_OF_CLASS


class TestDegenerateTrajectoryFiles:
    """The gate must REFUSE a pose-less reconstruction, never crash on it.

    server/oreos/recordings/modal_recon.py writes results/est_tum.txt unconditionally, so a
    run whose solver died mid-way leaves a zero-row (or comment-only) file. That
    used to raise numpy.AxisError inside _trajectory_metrics, which aborted the
    pipeline *before* confidence.json / report.html / EXPORT_REFUSED.md existed —
    the gate failing open in exactly the "recon produced nothing" case.
    """

    def _refusal(self, path: Path):
        c = compute_confidence(path)
        assert c.level is ConfidenceLevel.OUT_OF_CLASS
        assert not c.export_allowed
        assert c.reasons, "a refusal must carry a reason"
        return c

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.txt"
        p.write_text("")
        c = self._refusal(p)
        assert c.metrics.get("n_poses") == 0
        assert any("produced nothing" in r for r in c.reasons)

    def test_comment_only_file(self, tmp_path: Path) -> None:
        p = tmp_path / "comments.txt"
        p.write_text("# TUM: ts tx ty tz qx qy qz qw\n#\n\n   \n")
        c = self._refusal(p)
        assert c.metrics.get("n_poses") == 0

    def test_no_row_parses_to_a_pose(self, tmp_path: Path) -> None:
        # n=0 after parsing: rows exist but none carries the 8 TUM fields
        p = tmp_path / "short_rows.txt"
        p.write_text("1.0 0.0 0.0 0.0\n2.0 0.1 0.0 0.0\n3.0 0.2\n")
        c = self._refusal(p)
        assert c.metrics.get("n_poses") == 0

    def test_single_pose_file(self, tmp_path: Path) -> None:
        p = tmp_path / "one.txt"
        p.write_text("1.0 0.0 0.0 0.0 0 0 0 1\n")
        c = self._refusal(p)
        assert c.metrics.get("n_poses") == 1

    def test_load_tum_shape_is_nx3_even_when_empty(self, tmp_path: Path) -> None:
        from server.qc.confidence import load_tum

        p = tmp_path / "empty.txt"
        p.write_text("")
        ts, pos = load_tum(p)
        assert ts.shape == (0,)
        assert pos.shape == (0, 3), "the Nx3 contract must hold at N=0"

    def test_pipeline_qc_stage_refuses_instead_of_aborting(self, tmp_path: Path) -> None:
        """End-to-end on the real fail-open path: an empty est_tum.txt must leave
        a confidence.json + EXPORT_REFUSED.md behind, not a PipelineError."""
        from server.oreos.recordings.pipeline import OreosPipeline

        session = tmp_path / "dead_recon"
        (session / "results").mkdir(parents=True)
        (session / "results" / "est_tum.txt").write_text("")

        p = OreosPipeline("unused.db", session)
        qc = p.stage_qc()
        assert qc.status == "refused"
        conf = json.loads((session / "confidence.json").read_text())
        assert conf["level"] == "out_of_class"
        assert conf["export_allowed"] is False
        assert conf["reasons"]

        exp = p.stage_export()
        assert exp.status == "refused"
        assert (session / "EXPORT_REFUSED.md").exists()


class TestExp36Regression:
    """Real measured negatives: these cells failed EXP-36's bars at 21-29% of
    extent. The QC layer's whole reason to exist is that they never pass."""

    def test_bigoffice_collapse_detected(self) -> None:
        stats = json.loads((FIXTURES / "go2_bigoffice_stats.json").read_text())
        c = compute_confidence(FIXTURES / "go2_bigoffice_est_tum.txt", run_stats=stats)
        assert c.level is ConfidenceLevel.OUT_OF_CLASS
        assert not c.export_allowed

    def test_hk_village1_not_high_confidence(self) -> None:
        c = compute_confidence(FIXTURES / "hk_village1_est_tum.txt")
        assert c.level is not ConfidenceLevel.HIGH_CONFIDENCE

    def test_thresholds_documented(self) -> None:
        # every threshold key used in reasons exists and is finite
        for k, v in THRESHOLDS.items():
            assert np.isfinite(v), k


class TestKnownLimitations:
    """Documented boundary of the GT-free signals (see module docstring)."""

    def test_moderate_uniform_drift_is_a_known_false_negative(self) -> None:
        # china_office failed EXP-36 at 27.7% of extent vs the robot's odometry,
        # but uniform drift leaves per-step statistics clean, so v1 signals pass
        # it. This test PINS the limitation: if it starts failing, either a v2
        # signal landed (update the docstring + this test) or thresholds moved
        # (justify against pilot positives).
        c = compute_confidence(FIXTURES / "go2_china_office_est_tum.txt")
        assert c.level is ConfidenceLevel.HIGH_CONFIDENCE


class TestQcCli:
    """server/qc/cli.py — the shell-facing gate (exit 0/1/2)."""

    def _run(self, monkeypatch, argv: list[str]) -> tuple[int, dict]:
        import sys

        from server.qc import cli as qc_cli

        printed: list[str] = []
        monkeypatch.setattr(sys, "argv", ["server.qc.cli", *argv])
        monkeypatch.setattr("builtins.print", lambda *a, **kw: printed.append(str(a[0])))
        with pytest.raises(SystemExit) as exc:
            qc_cli.main()
        return int(exc.value.code), json.loads(printed[-1])

    def test_exit_codes(self, tmp_path: Path, monkeypatch) -> None:
        ts, pos = circle_trajectory()
        good = write_tum(tmp_path / "good.txt", ts, pos)
        code, doc = self._run(monkeypatch, [str(good)])
        assert code == 0 and doc["level"] == "high_confidence"

        empty = tmp_path / "empty.txt"
        empty.write_text("")
        code, doc = self._run(monkeypatch, [str(empty)])
        assert code == 2 and doc["level"] == "out_of_class"
        assert doc["export_allowed"] is False

    def test_stats_file_handle_is_closed(self, tmp_path: Path, monkeypatch) -> None:
        """`json.load(open(...))` never closed the handle — CPython reaped it and
        raised ResourceWarning, and on a long-lived caller it is a real fd leak."""
        import warnings

        ts, pos = circle_trajectory()
        traj = write_tum(tmp_path / "t.txt", ts, pos)
        stats = tmp_path / "run_summary.json"
        stats.write_text(json.dumps({"n_pose_sign_repairs": 0, "n_poses_written": 300}))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            code, _doc = self._run(monkeypatch, [str(traj), "--stats", str(stats)])

        assert code == 0
        leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert not leaks, f"unclosed file handle(s): {[str(w.message) for w in leaks]}"
