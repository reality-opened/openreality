"""CLI: score a reconstruction's confidence (GT-free).

  python -m server.qc.cli <est_tum.txt> [--stats run_summary.json]

Prints the ScanConfidence as JSON and exits 0 (high_confidence),
1 (needs_review), or 2 (out_of_class) so shell pipelines can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys

from server.qc.confidence import ConfidenceLevel, compute_confidence

_EXIT = {
    ConfidenceLevel.HIGH_CONFIDENCE: 0,
    ConfidenceLevel.NEEDS_REVIEW: 1,
    ConfidenceLevel.OUT_OF_CLASS: 2,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("est_tum")
    ap.add_argument("--stats", help="run_summary.json with solver stats")
    args = ap.parse_args()
    stats = None
    if args.stats:
        with open(args.stats) as f:  # `json.load(open(...))` leaked the handle
            stats = json.load(f)
    c = compute_confidence(args.est_tum, run_stats=stats)
    print(
        json.dumps(
            {"level": c.level.value, "export_allowed": c.export_allowed,
             "reasons": c.reasons, "metrics": c.metrics},
            indent=2,
        )
    )
    sys.exit(_EXIT[c.level])


if __name__ == "__main__":
    main()
