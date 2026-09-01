"""Tests for server.oreos.recordings.pipeline (+ the server.oreos.recordings.cli entrypoints).

Hermetic: no network, no GPU, no modal account. Real stages (ingest / score / qc /
export) run for real against tiny fixtures; Modal-touching stages (upload / recon /
fetch) and the report stage are mocked — see module docstring in pipeline.py for the
stage list and what each one is allowed to touch.

Reuses `make_fixture_db` from test_ingest_dimos.py (a miniature DimOS memory2 .db)
rather than re-deriving the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# `modal` is NOT in CI's curated GPU-free dependency subset. A bare
# `import modal` here is a *collection* error, which pytest treats as fatal for
# the whole run — one missing optional dep took the entire suite down. Skip the
# module instead (same style as the vggt_slam/pxr guards elsewhere in tests/).
modal = pytest.importorskip("modal", reason="modal not installed (GPU-free CI subset)")

from server.oreos.recordings import cli as cli_mod  # noqa: E402
from server.oreos.recordings.pipeline import OreosPipeline, PipelineError, StageResult  # noqa: E402
from test_ingest_dimos import make_fixture_db  # noqa: E402


# -- shared helpers -----------------------------------------------------------


def circle_trajectory(n: int = 60, radius: float = 5.0):
    """Same shape as test_qc_confidence.py's circle_trajectory: a clean, closed
    loop with no scale drift or teleports — should score high_confidence and
    align to itself with ~0 ATE."""
    import numpy as np

    ts = np.linspace(0, 60, n)
    ang = np.linspace(0, 2 * np.pi, n)
    pos = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)
    return ts, pos


def write_tum(path: Path, ts, pos) -> Path:
    lines = [f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 1" for t, p in zip(ts, pos)]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    p = tmp_path / "mini.db"
    make_fixture_db(p, n_frames=12, fps=15.0)
    return p


def read_log(session_dir: Path) -> list[dict]:
    text = (session_dir / "pipeline_log.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class FakeFunctionCall:
    """Stand-in for modal.FunctionCall.from_id(...) -> obj.get(timeout=...)."""

    def __init__(self, summary: dict):
        self._summary = summary

    def get(self, timeout=0):
        return self._summary


class FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


# -- 1. ingest (real) -----------------------------------------------------


def test_ingest_stage_real(fixture_db: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(fixture_db, session_dir, fps=15.0)

    res = p.stage_ingest()

    assert isinstance(res, StageResult)
    assert res.stage == "ingest"
    assert res.status == "ok"
    assert res.detail["n_frames"] == 12
    manifest_path = session_dir / "ingest_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_frames"] == 12
    frames = sorted((session_dir / "frames").glob("*.jpg"))
    assert len(frames) == 12
    log = read_log(session_dir)
    assert len(log) == 1
    assert log[0]["stage"] == "ingest"
    assert log[0]["status"] == "ok"


# -- 2. idempotency ---------------------------------------------------------


def test_stage_skip_idempotency(fixture_db: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(fixture_db, session_dir, fps=15.0)

    first = p.stage_ingest()
    second = p.stage_ingest()

    assert first.status == "ok"
    assert second.status == "skipped"
    assert second.detail.get("marker") == "ingest_manifest.json"
    log = read_log(session_dir)
    assert len(log) == 2
    assert [entry["status"] for entry in log] == ["ok", "skipped"]


# -- 3/4. recon (mocked subprocess + modal.FunctionCall) --------------------


def test_recon_poll_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(tmp_path / "unused.db", session_dir, session_name="sess1")

    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        return FakeCompletedProcess(stdout="SPAWNED recon x: fc-123\n")

    summary = {"submaps": 4, "loop_closures": 6, "solver_error": None, "gpu_seconds": 88.5}
    from_id_calls = []

    def fake_from_id(function_call_id, client=None):
        from_id_calls.append(function_call_id)
        return FakeFunctionCall(summary)

    monkeypatch.setattr("server.oreos.recordings.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(fake_from_id))

    res = p.stage_recon(submap_size=16, min_disparity=50.0)

    assert res.status == "ok"
    assert res.detail["submaps"] == 4
    assert res.detail["loop_closures"] == 6
    assert res.detail["fc_id"] == "fc-123"
    assert from_id_calls == ["fc-123"]

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "modal"
    assert cmd[1:3] == ["run", "--detach"]
    assert cmd[3] == str(p._modal_recon_script)
    assert "--session" in cmd and cmd[cmd.index("--session") + 1] == "sess1"
    assert "--submap-size" in cmd and cmd[cmd.index("--submap-size") + 1] == "16"
    assert "--min-disparity" in cmd and cmd[cmd.index("--min-disparity") + 1] == "50.0"

    recon_call = json.loads((session_dir / "recon_call.json").read_text())
    assert recon_call == {"fc_id": "fc-123"}


def test_recon_spawn_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(tmp_path / "unused.db", session_dir, session_name="sess1")

    def fake_run(cmd, capture_output, text, timeout):
        return FakeCompletedProcess(stdout="modal: some unrelated log line\n", stderr="")

    monkeypatch.setattr("server.oreos.recordings.pipeline.subprocess.run", fake_run)

    with pytest.raises(PipelineError) as excinfo:
        p.stage_recon()

    assert excinfo.value.stage_result.stage == "recon"
    log = read_log(session_dir)
    assert len(log) == 1
    assert log[0]["stage"] == "recon"
    assert log[0]["status"] == "error"
    assert "spawn failed" in log[0]["detail"]["error"]
    # no recon_call.json since the spawn never produced an fc_id
    assert not (session_dir / "recon_call.json").exists()


# -- 5. score + qc (real) -----------------------------------------------


def test_score_and_qc_real(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(tmp_path / "unused.db", session_dir, session_name="sess1")
    (session_dir / "results").mkdir(parents=True)

    ts, pos = circle_trajectory()
    write_tum(session_dir / "results" / "est_tum.txt", ts, pos)
    write_tum(session_dir / "reference.txt", ts, pos)  # same circle -> ~0 ATE

    score_res = p.stage_score()
    assert score_res.status == "ok"
    assert score_res.detail["reference_available"] is True
    assert score_res.detail["ate_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    consistency = json.loads((session_dir / "consistency.json").read_text())
    assert consistency["reference_available"] is True
    assert consistency["ate_rmse_m"] == pytest.approx(0.0, abs=1e-6)

    qc_res = p.stage_qc()
    assert qc_res.status == "ok"
    confidence = json.loads((session_dir / "confidence.json").read_text())
    assert confidence["level"] == "high_confidence"
    assert confidence["export_allowed"] is True
    assert confidence["reasons"] == []


# -- 6. qc refuses on solver error, without touching est_tum.txt --------


def test_qc_refuses_on_solver_error(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(tmp_path / "unused.db", session_dir, session_name="sess1")
    results = session_dir / "results"
    results.mkdir(parents=True)
    # deliberately NOT writing est_tum.txt: stage_qc must refuse before it would
    # ever need to read it.
    (results / "recon_summary.json").write_text(
        json.dumps({"solver_error": "gtsam LM did not converge", "n_poses_written": 0})
    )

    res = p.stage_qc()

    assert res.status == "refused"
    assert not (results / "est_tum.txt").exists()
    confidence = json.loads((session_dir / "confidence.json").read_text())
    assert confidence["level"] == "out_of_class"
    assert confidence["export_allowed"] is False
    assert any("solver runtime failure" in r for r in confidence["reasons"])
    assert any("gtsam LM did not converge" in r for r in confidence["reasons"])


# -- 7. export gate ----------------------------------------------------------


def test_export_gate(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(tmp_path / "unused.db", session_dir, session_name="sess1")

    (session_dir / "confidence.json").write_text(json.dumps({
        "level": "out_of_class",
        "export_allowed": False,
        "reasons": ["scale drift: step_trend_ratio 0.1 outside [0.4, 2.5]", "too few poses"],
    }))

    refused_res = p.stage_export()
    assert refused_res.status == "refused"
    refused_md = (session_dir / "EXPORT_REFUSED.md").read_text()
    assert "out_of_class" in refused_md
    assert "scale drift: step_trend_ratio 0.1 outside [0.4, 2.5]" in refused_md
    assert "too few poses" in refused_md
    assert not (session_dir / "bundle").exists()
    # a refusal must NOT write the done-marker: the export stays re-evaluable
    assert not (session_dir / ".export_done").exists()

    # confidence now allows export; minimal artifacts present
    (session_dir / "confidence.json").write_text(json.dumps({
        "level": "high_confidence",
        "export_allowed": True,
        "reasons": [],
    }))
    (session_dir / "results").mkdir(exist_ok=True)
    (session_dir / "results" / "est_tum.txt").write_text("0.0 0 0 0 0 0 0 1\n")
    (session_dir / "consistency.json").write_text(json.dumps({"reference_available": False}))
    (session_dir / "ingest_manifest.json").write_text(json.dumps({"n_frames": 12}))

    # no force needed: refusal left no marker, so the retry re-evaluates naturally
    export_res = p.stage_export()

    assert export_res.status == "ok"
    bundle = session_dir / "bundle"
    assert bundle.is_dir()
    assert (bundle / "est_tum.txt").exists()
    assert (bundle / "consistency.json").exists()
    assert (bundle / "ingest_manifest.json").exists()
    assert (bundle / "confidence.json").exists()
    bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert bundle_manifest["session"] == "sess1"
    assert bundle_manifest["kind"] == "oreos-scene-bundle-v0"
    assert bundle_manifest["confidence"] == "high_confidence"


# -- 8. every stage call appends exactly one log line -----------------------


def test_pipeline_log_grows(fixture_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    p = OreosPipeline(fixture_db, session_dir, fps=15.0)

    p.stage_ingest()
    log = read_log(session_dir)
    assert len(log) == 1 and log[-1]["status"] == "ok"

    p.stage_ingest()  # no force -> skipped, still logs
    log = read_log(session_dir)
    assert len(log) == 2 and log[-1]["status"] == "skipped"

    p.stage_ingest(force=True)  # forced re-run -> ok, logs again
    log = read_log(session_dir)
    assert len(log) == 3 and log[-1]["status"] == "ok"

    def fake_run_fail(cmd, capture_output, text, timeout):
        return FakeCompletedProcess(stdout="no spawn marker here\n")

    monkeypatch.setattr("server.oreos.recordings.pipeline.subprocess.run", fake_run_fail)
    with pytest.raises(PipelineError):
        p.stage_recon()  # errors -> still exactly one more log line
    log = read_log(session_dir)
    assert len(log) == 4 and log[-1]["status"] == "error" and log[-1]["stage"] == "recon"


# -- 9. cli ------------------------------------------------------------------


def test_cli_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def make_stub(stage_name: str):
        def stub(self, force=False, **kwargs):
            calls.append(stage_name)
            return StageResult(stage_name, "ok", 0.01, {})
        return stub

    for name in ("ingest", "upload", "recon", "fetch", "score", "qc", "report", "export"):
        monkeypatch.setattr(OreosPipeline, f"stage_{name}", make_stub(name))

    db = tmp_path / "rec.db"
    db.write_text("stub-recording")
    out_dir = tmp_path / "out"

    rc = cli_mod.main(["run", str(db), "--out", str(out_dir)])

    assert rc == 0
    assert calls == ["ingest", "upload", "recon", "fetch", "score", "qc", "report", "export"]

    # restore the real stage_qc (undo the stubs above) before exercising the
    # `qc` subcommand for real against a refused session.
    monkeypatch.undo()

    # `qc` subcommand: session with a refused confidence must exit 2
    session_dir = tmp_path / "qc_session"
    results = session_dir / "results"
    results.mkdir(parents=True)
    (results / "recon_summary.json").write_text(json.dumps({"solver_error": "diverged"}))

    rc_qc = cli_mod.main(["qc", str(session_dir)])

    assert rc_qc == 2
    confidence = json.loads((session_dir / "confidence.json").read_text())
    assert confidence["export_allowed"] is False
