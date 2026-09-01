"""Oreos CLI — the end-to-end walkthrough entrypoint.

  python -m server.oreos.recordings run <recording.db> --out sessions/<name>
      [--name <session>] [--fps 10] [--force] [--with-submaps]
      [--submap-size 16] [--min-disparity 50]

  python -m server.oreos.recordings report <session_dir>       # (re)build report.html only
  python -m server.oreos.recordings qc <session_dir>           # re-run the gate only

Output tree (the operator's mental model):
  <session_dir>/
    ingest_manifest.json   what the recording contained
    frames/ frames.tar     extracted RGB (uploaded as one tar)
    results/               est_tum.txt, recon_summary.json, map_preview.ply[, submaps/]
    consistency.json       vs the robot's own odometry (NOT ground truth)
    confidence.json        the QC verdict that gates everything downstream
    report.html            open this — the whole story on one page
    bundle/ | EXPORT_REFUSED.md
    pipeline_log.jsonl     stage-by-stage record
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from server.oreos.recordings.pipeline import OreosPipeline, PipelineError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oreos", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="full pipeline on a recording")
    run.add_argument("recording", help="DimOS memory2 .db recording, or a plain video (.mp4/.mov/.avi)")
    run.add_argument("--out", required=True, help="session output dir")
    run.add_argument("--name", default=None, help="session name (default: recording stem)")
    run.add_argument("--fps", type=float, default=10.0)
    run.add_argument("--force", action="store_true", help="re-run completed stages")
    run.add_argument("--with-submaps", action="store_true",
                     help="also fetch per-submap bundles (for the DimOS replay node)")
    run.add_argument("--submap-size", type=int, default=16)
    run.add_argument("--min-disparity", type=float, default=50.0)

    rep = sub.add_parser("report", help="rebuild report.html for a session dir")
    rep.add_argument("session_dir")

    qc = sub.add_parser("qc", help="re-run the QC gate for a session dir")
    qc.add_argument("session_dir")

    exp = sub.add_parser("export", help="re-attempt the gated export for a session dir")
    exp.add_argument("session_dir")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        p = OreosPipeline(args.recording, args.out, session_name=args.name, fps=args.fps)
        try:
            results = [
                p.stage_ingest(args.force),
                p.stage_upload(args.force),
                p.stage_recon(args.force, submap_size=args.submap_size,
                              min_disparity=args.min_disparity),
                p.stage_fetch(args.force, with_submaps=args.with_submaps),
                p.stage_score(args.force),
                p.stage_qc(args.force),
                p.stage_report(args.force),
                p.stage_export(args.force),
            ]
        except PipelineError as e:
            print(f"\nPIPELINE STOPPED at {e.stage_result.stage}: {e.stage_result.detail}")
            return 1
        gate = next((r for r in results if r.stage == "qc"), None)
        print(f"\nDone. Open {Path(args.out) / 'report.html'}")
        if gate and gate.status == "refused":
            print("QC verdict: REFUSED — see EXPORT_REFUSED.md (this is the gate working).")
        return 0

    if args.cmd == "report":
        from server.oreos.recordings.report import generate_report

        out = generate_report(Path(args.session_dir))
        print(out)
        return 0

    if args.cmd == "qc":
        p = OreosPipeline("unused.db", args.session_dir)
        res = p.stage_qc(force=True)
        return 0 if res.status == "ok" else 2

    if args.cmd == "export":
        p = OreosPipeline("unused.db", args.session_dir)
        res = p.stage_export()
        # "skipped" means the .export_done marker is already there — this session
        # exported successfully on an earlier run. That is an idempotent no-op
        # success, not a failure; returning 2 for it made retry loops and CI
        # wrappers treat a completed export as an error. Only "refused" (the QC
        # gate said no) and "error" are non-zero.
        return 0 if res.status in ("ok", "skipped") else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
