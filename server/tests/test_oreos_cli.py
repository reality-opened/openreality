"""Exit-code contract for `python -m server.oreos.recordings` (server/oreos/recordings/cli.py).

Lives apart from test_oreos_pipeline.py on purpose: that module importorskips
`modal`, so it never runs in the GPU-free CI subset — and the CLI's exit-code
mapping is exactly the sort of shell-facing contract CI must keep honest. Nothing
here touches Modal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.oreos.recordings import cli as cli_mod

# the export stage builds real training-data targets from the gated bundle
pytest.importorskip("cv2")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _exportable_session(root: Path) -> Path:
    d = root / "sess"
    (d / "results").mkdir(parents=True)
    (d / "confidence.json").write_text(json.dumps({
        "level": "high_confidence", "export_allowed": True, "reasons": [],
    }))
    (d / "results" / "est_tum.txt").write_text("0.0 0 0 0 0 0 0 1\n")
    (d / "consistency.json").write_text(json.dumps({"reference_available": False}))
    (d / "ingest_manifest.json").write_text(json.dumps({"n_frames": 12}))
    return d


def _refused_session(root: Path) -> Path:
    d = root / "refused"
    d.mkdir(parents=True)
    (d / "confidence.json").write_text(json.dumps({
        "level": "out_of_class", "export_allowed": False,
        "reasons": ["scale drift: step_trend_ratio 0.1 outside [0.4, 2.5]"],
    }))
    return d


def test_export_is_idempotent_and_exits_zero_when_already_done(tmp_path: Path) -> None:
    """An already-exported session reports status "skipped", which used to map to
    exit code 2 — so re-running `oreos export` (a retry loop, a CI wrapper, a
    second operator) reported failure for a session that had exported fine. A
    completed export is a no-op success."""
    d = _exportable_session(tmp_path)

    first = cli_mod.main(["export", str(d)])
    assert first == 0
    assert (d / ".export_done").exists()
    assert (d / "bundle" / "bundle_manifest.json").exists()

    second = cli_mod.main(["export", str(d)])  # marker present -> "skipped"
    assert second == 0, "an already-exported session is success, not failure"


def test_export_refused_by_the_gate_still_exits_two(tmp_path: Path) -> None:
    d = _refused_session(tmp_path)

    rc = cli_mod.main(["export", str(d)])

    assert rc == 2
    assert (d / "EXPORT_REFUSED.md").exists()
    assert not (d / "bundle").exists()
    # a refusal leaves no marker, so it stays re-evaluable (and keeps exiting 2)
    assert not (d / ".export_done").exists()
    assert cli_mod.main(["export", str(d)]) == 2


def test_export_error_exits_two(tmp_path: Path) -> None:
    """A genuinely broken session (no confidence.json at all) must not be
    mistaken for the benign "skipped" case."""
    from server.oreos.recordings.pipeline import PipelineError

    d = tmp_path / "no_gate"
    d.mkdir()

    with pytest.raises(PipelineError):
        cli_mod.main(["export", str(d)])
