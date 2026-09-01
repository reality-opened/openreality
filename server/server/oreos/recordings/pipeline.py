"""Oreos pipeline orchestrator: recording -> gated, human-readable scene results.

Stages (each idempotent; completed stages are skipped unless force=True):
  ingest   recording.db -> frames/ + reference.txt + ingest_manifest.json  (local)
  upload   frames -> single frames.tar on the oreos-sessions Modal volume  (tar survives
           flaky egress where multi-file puts die — EXP-36 lesson)
  recon    spawn server/oreos/recordings/modal_recon.py on an A100; poll with short calls only
  fetch    est_tum.txt + recon_summary.json + map_preview.ply back to the session dir
  score    consistency vs the recording's own odometry (numpy scorer; no evo)
  qc       ScanConfidence gate (server.qc) — the verdict every later stage obeys
  report   self-contained report.html (server.oreos.recordings.report)
  export   scene bundle ONLY if qc allows; otherwise EXPORT_REFUSED.md with reasons

Every stage appends to pipeline_log.jsonl (stage, status, seconds, detail) — the
report renders this so an operator can see exactly what happened and why.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StageResult:
    stage: str
    status: str  # ok | skipped | refused | error
    seconds: float
    detail: dict

    def as_dict(self) -> dict:
        return {"stage": self.stage, "status": self.status,
                "seconds": round(self.seconds, 2), "detail": self.detail}


class OreosPipeline:
    def __init__(
        self,
        recording: str | Path,
        session_dir: str | Path,
        session_name: str | None = None,
        fps: float = 10.0,
        modal_bin: str = "modal",
        poll_interval: float = 30.0,
        recon_timeout_s: float = 3600.0,
    ) -> None:
        self.recording = Path(recording)
        self.session_dir = Path(session_dir)
        self.session = session_name or self.recording.stem
        self.fps = fps
        self.modal_bin = modal_bin
        self.poll_interval = poll_interval
        self.recon_timeout_s = recon_timeout_s
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.session_dir / "pipeline_log.jsonl"
        self._modal_recon_script = Path(__file__).parent / "modal_recon.py"

    # -- plumbing -----------------------------------------------------------

    def _log(self, res: StageResult) -> StageResult:
        entry = {"ts": time.time(), **res.as_dict()}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[oreos:{res.stage}] {res.status} ({res.seconds:.1f}s) "
              + (json.dumps(res.detail)[:160] if res.detail else ""))
        return res

    def _stage(self, name, done_marker: Path, fn, force: bool) -> StageResult:
        if done_marker.exists() and not force:
            return self._log(StageResult(name, "skipped", 0.0, {"marker": str(done_marker.name)}))
        t0 = time.time()
        try:
            detail = fn() or {}
            status = detail.pop("_status", "ok")
            return self._log(StageResult(name, status, time.time() - t0, detail))
        except Exception as e:  # noqa: BLE001 — log + surface, never half-die silently
            res = self._log(
                StageResult(name, "error", time.time() - t0, {"error": f"{type(e).__name__}: {e}"})
            )
            raise PipelineError(res) from e

    # -- stages -------------------------------------------------------------

    def stage_ingest(self, force: bool = False) -> StageResult:
        # One dispatch rule, shared with server/ingest/cli.py: VIDEO_SUFFIXES is
        # the single definition (a second copy here drifted from it once already).
        from server.ingest.video import VIDEO_SUFFIXES

        if self.recording.suffix.lower() in VIDEO_SUFFIXES:
            from server.ingest.video import extract_frames_from_video as extract_frames
        else:
            from server.ingest.dimos_db import extract_frames

        def fn():
            manifest = extract_frames(self.recording, self.session_dir, fps=self.fps)
            return {"n_frames": manifest["n_frames"],
                    "reference_usable": manifest.get("reference_usable"),
                    "odom_stream": manifest.get("odom_stream")}

        return self._stage("ingest", self.session_dir / "ingest_manifest.json", fn, force)

    def stage_upload(self, force: bool = False) -> StageResult:
        tar_local = self.session_dir / "frames.tar"

        def fn():
            if not tar_local.exists() or force:
                with tarfile.open(tar_local, "w") as tf:
                    tf.add(self.session_dir / "frames", arcname="frames")
            import modal

            vol = modal.Volume.from_name("oreos-sessions", create_if_missing=True)
            last_err = None
            for attempt in range(8):
                try:
                    with vol.batch_upload(force=True) as batch:
                        batch.put_file(str(tar_local), f"{self.session}/frames.tar")
                    return {"bytes": tar_local.stat().st_size, "attempts": attempt + 1}
                except Exception as e:  # noqa: BLE001 — flaky egress; retry
                    last_err = e
                    time.sleep(10)
            raise RuntimeError(f"upload failed after 8 attempts: {last_err}")

        return self._stage_with_marker("upload", fn, force)

    def _stage_with_marker(self, name: str, fn, force: bool) -> StageResult:
        marker = self.session_dir / f".{name}_done"
        def wrapped():
            detail = fn() or {}
            marker.write_text(json.dumps({"ts": time.time()}))
            return detail
        return self._stage(name, marker, wrapped, force)

    def stage_recon(self, force: bool = False, submap_size: int = 16,
                    min_disparity: float = 50.0) -> StageResult:
        def fn():
            proc = subprocess.run(
                [self.modal_bin, "run", "--detach", str(self._modal_recon_script),
                 "--session", self.session, "--submap-size", str(submap_size),
                 "--min-disparity", str(min_disparity)],
                capture_output=True, text=True, timeout=600,
            )
            fc_id = None
            for line in (proc.stdout + proc.stderr).splitlines():
                if line.startswith("SPAWNED recon"):
                    fc_id = line.rsplit(": ", 1)[-1].strip()
            if not fc_id:
                raise RuntimeError(f"spawn failed: {proc.stdout[-300:]} {proc.stderr[-300:]}")
            (self.session_dir / "recon_call.json").write_text(json.dumps({"fc_id": fc_id}))

            import modal

            deadline = time.time() + self.recon_timeout_s
            while time.time() < deadline:
                try:
                    summary = modal.FunctionCall.from_id(fc_id).get(timeout=0)
                    if summary.get("error"):
                        raise RuntimeError(summary["error"])
                    return {"fc_id": fc_id,
                            "submaps": summary.get("submaps"),
                            "loop_closures": summary.get("loop_closures"),
                            "solver_error": summary.get("solver_error"),
                            "gpu_seconds": summary.get("gpu_seconds")}
                except TimeoutError:
                    time.sleep(self.poll_interval)
            raise RuntimeError(f"recon timed out after {self.recon_timeout_s}s (call {fc_id})")

        return self._stage_with_marker("recon", fn, force)

    def stage_fetch(self, force: bool = False, with_submaps: bool = False) -> StageResult:
        def fn():
            import modal

            vol = modal.Volume.from_name("oreos-sessions")
            results = self.session_dir / "results"
            results.mkdir(exist_ok=True)
            wanted = ["est_tum.txt", "recon_summary.json", "map_preview.ply"]
            fetched = []
            for name in wanted:
                remote = f"{self.session}/results/{name}"
                try:
                    data = b"".join(vol.read_file(remote))
                except Exception:  # noqa: BLE001 — optional artifacts may not exist
                    continue
                (results / name).write_bytes(data)
                fetched.append(name)
            if with_submaps:
                sub = results / "submaps"
                sub.mkdir(exist_ok=True)
                try:
                    for entry in vol.listdir(f"{self.session}/results/submaps"):
                        data = b"".join(vol.read_file(entry.path))
                        (sub / Path(entry.path).name).write_bytes(data)
                        fetched.append(f"submaps/{Path(entry.path).name}")
                except Exception:  # noqa: BLE001
                    pass
            if "est_tum.txt" not in fetched:
                raise RuntimeError("est_tum.txt not on volume — recon produced no poses")
            return {"fetched": fetched}

        return self._stage_with_marker("fetch", fn, force)

    def stage_score(self, force: bool = False) -> StageResult:
        from server.oreos.recordings.consistency import score_consistency

        def fn():
            ref = self.session_dir / "reference.txt"
            out = score_consistency(
                self.session_dir / "results" / "est_tum.txt",
                ref if ref.exists() else None,
                out_json=self.session_dir / "consistency.json",
            )
            return {k: out.get(k) for k in
                    ("reference_available", "ate_rmse_m", "ate_pct_extent",
                     "ate_pct_path", "windowed_scale_cv")}

        return self._stage("score", self.session_dir / "consistency.json", fn, force)

    def stage_qc(self, force: bool = False) -> StageResult:
        from server.qc.confidence import compute_confidence

        def fn():
            stats = {}
            rs = self.session_dir / "results" / "recon_summary.json"
            if rs.exists():
                s = json.loads(rs.read_text())
                stats = {k: s.get(k) for k in
                         ("n_pose_sign_repairs", "n_poses_written",
                          "frames_kept", "frames_total")}
                if s.get("solver_error"):
                    payload = {
                        "level": "out_of_class",
                        "export_allowed": False,
                        "reasons": [f"solver runtime failure: {s['solver_error']}"],
                        "metrics": stats,
                    }
                    (self.session_dir / "confidence.json").write_text(
                        json.dumps(payload, indent=2))
                    return {"_status": "refused", **payload}
            c = compute_confidence(
                self.session_dir / "results" / "est_tum.txt", run_stats=stats)
            payload = {"level": c.level.value, "export_allowed": c.export_allowed,
                       "reasons": c.reasons, "metrics": c.metrics}
            (self.session_dir / "confidence.json").write_text(json.dumps(payload, indent=2))
            return {"_status": "ok" if c.export_allowed else "refused", **payload}

        return self._stage("qc", self.session_dir / "confidence.json", fn, force)

    def stage_report(self, force: bool = False) -> StageResult:
        from server.oreos.recordings.report import generate_report

        def fn():
            out = generate_report(self.session_dir)
            return {"report": str(out)}

        return self._stage("report", self.session_dir / "report.html", fn, force)

    def stage_export(self, force: bool = False) -> StageResult:
        # Marker is written ONLY on a successful (allowed) export: a refusal must
        # stay re-evaluable — re-running the gate after fixes should be able to
        # export without forcing a full pipeline re-run.
        marker = self.session_dir / ".export_done"

        def fn():
            conf = json.loads((self.session_dir / "confidence.json").read_text())
            if not conf.get("export_allowed", False):
                refused = self.session_dir / "EXPORT_REFUSED.md"
                refused.write_text(
                    "# Export refused by QC gate\n\n"
                    f"Confidence level: **{conf.get('level')}**\n\nReasons:\n"
                    + "\n".join(f"- {r}" for r in conf.get("reasons", []))
                    + "\n\nSee report.html for full context. Nothing ships un-gated.\n"
                )
                return {"_status": "refused", "reasons": conf.get("reasons", [])}
            bundle = self.session_dir / "bundle"
            bundle.mkdir(exist_ok=True)
            import shutil

            for name in ("results/est_tum.txt", "results/map_preview.ply",
                         "consistency.json", "confidence.json", "ingest_manifest.json"):
                src = self.session_dir / name
                if src.exists():
                    shutil.copy2(src, bundle / Path(name).name)
            # Real training-data targets (gate already passed above). build_exports
            # never raises for data problems: per-target status + reasons land in the
            # bundle manifest so a target failure can't kill the gated bundle.
            from server.oreos.recordings.export_targets import build_exports

            exports = build_exports(self.session_dir, targets=["lerobot"], out_dir=bundle)
            (bundle / "bundle_manifest.json").write_text(json.dumps({
                "session": self.session,
                "kind": "oreos-scene-bundle-v0",
                "note": ("Gated scene bundle: trajectory + cloud + provenance copies, "
                         "plus training-data targets (openreality/ = OpenReality export "
                         "tree, lerobot/ = GR00T-LeRobot v2, isaac/ optional). "
                         "Per-target status + reasons under 'targets'."),
                "confidence": conf.get("level"),
                "targets": exports.get("targets", {}),
                "caveats": exports.get("caveats", []),
            }, indent=2))
            marker.write_text(json.dumps({"ts": time.time()}))
            return {"bundle": str(bundle),
                    "targets": {k: v.get("status")
                                for k, v in exports.get("targets", {}).items()}}

        return self._stage("export", marker, fn, force)

    # -- one-shot -----------------------------------------------------------

    def run_all(self, force: bool = False, with_submaps: bool = False) -> list[StageResult]:
        results = [
            self.stage_ingest(force),
            self.stage_upload(force),
            self.stage_recon(force),
            self.stage_fetch(force, with_submaps=with_submaps),
            self.stage_score(force),
            self.stage_qc(force),
            self.stage_report(force),
            self.stage_export(force),
        ]
        return results


class PipelineError(RuntimeError):
    def __init__(self, stage_result: StageResult) -> None:
        super().__init__(f"stage {stage_result.stage}: {stage_result.detail}")
        self.stage_result = stage_result
