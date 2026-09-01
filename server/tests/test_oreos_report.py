"""Tests for server.oreos.recordings.report.generate_report.

Three scenarios:
  * full session   — every input present; the page must carry the level string,
                     both plot data URIs, the mandatory odometry caveat, and every
                     reason verbatim.
  * empty session  — an empty dir must still produce report.html, degrading to
                     "not available" everywhere and never raising.
  * refused session — out_of_class + a solver runtime error; the page must show
                     the red band and surface the solver error text.
  * no-matplotlib  — the page must still render (figures degrade to a note).

Optional deps: only the full-session case needs matplotlib (it asserts the two
baked PNGs), so that one test is skip-gated; the other cases exercise the
figure-free paths and must run in the GPU-free CI subset, which ships neither
matplotlib nor open3d. The PLY is written with open3d when it is importable (the
real read path) and otherwise by hand as ascii, which exercises report.py's
documented hand-rolled parser fallback — either way a real cloud is read.
"""

from __future__ import annotations

import html as htmllib
import json
from pathlib import Path

import numpy as np
import pytest

from server.oreos.recordings import report as report_mod
from server.oreos.recordings.report import CAVEAT, PLOTS_UNAVAILABLE_NOTE, generate_report

# Ask the module itself whether it can plot rather than probing for the package:
# tests/export_fakes.py may already have parked a *stub* matplotlib in sys.modules
# (see the leak note there), which importlib.util.find_spec chokes on.
requires_matplotlib = pytest.mark.skipif(
    report_mod._pyplot() is None,
    reason="matplotlib unavailable (GPU-free CI subset); figures degrade to a note",
)


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _write_tum(path: Path, n: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ang = np.linspace(0, 2 * np.pi, n)
    lines = []
    for i, a in enumerate(ang):
        x, y, z = 3.0 * np.cos(a), 3.0 * np.sin(a), 0.1 * i
        lines.append(f"{i * 0.1:.6f} {x:.6f} {y:.6f} {z:.6f} 0 0 0 1")
    path.write_text("# TUM: ts tx ty tz qx qy qz qw\n" + "\n".join(lines) + "\n")


def _write_ply(path: Path, n: int = 400) -> None:
    """Small cloud: open3d (binary_little_endian) when available — the real read
    path — else a hand-written ascii PLY, which report.py's fallback parser reads.
    open3d is not in the GPU-free CI dep subset, so this must not hard-require it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.0, 2.0, size=(n, 3))
    try:
        import open3d as o3d
    except ImportError:
        body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
        path.write_text(
            "ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n" + body + "\n"
        )
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(path), pcd)


def _full_session(root: Path) -> tuple[Path, list[str]]:
    """Build a complete session dir; returns (dir, reasons that must appear)."""
    d = root / "kitchen_tour_01"
    d.mkdir(parents=True)
    (d / "results").mkdir()

    aligned = [[float(x), float(x * 0.5), 0.0] for x in range(0, 12)]
    reference = [[float(x) + 0.05, float(x * 0.5) - 0.03, 0.0] for x in range(0, 12)]

    (d / "ingest_manifest.json").write_text(json.dumps({
        "source": "rec_kitchen.db", "n_frames": 240, "span_s": 61.4,
        "odom_stream": "/odom", "n_reference_poses": 300,
        "reference_spread_m": 4.2, "reference_usable": True,
        "streams": [{"name": "/cam0", "codec": "h264", "rows": 240, "span_s": 61.4}],
    }))
    (d / "results" / "recon_summary.json").write_text(json.dumps({
        "frames_total": 240, "frames_kept": 180, "submaps": 6, "loop_closures": 2,
        "n_pose_sign_repairs": 0, "n_poses_written": 180, "solver_error": None,
        "export_errors": [], "gpu_seconds": 142.7, "config": {"submap_size": 16},
        "core_source": {"mode": "tag", "tag": "v2.2.0"},
    }))
    (d / "consistency.json").write_text(json.dumps({
        "reference_available": True, "ate_rmse_m": 0.084, "ate_median_m": 0.071,
        "extent_m": 4.2, "gt_path_length_m": 11.9, "ate_pct_extent": 2.0,
        "ate_pct_path": 0.71, "umeyama_scale": 0.973, "windowed_scale_cv": 0.061,
        "aligned_positions": aligned, "reference_positions": reference,
    }))
    reasons = ["keep_ratio 0.150 < 0.15", "step_cv 1.34 > 1.2"]
    (d / "confidence.json").write_text(json.dumps({
        "level": "needs_review", "export_allowed": True, "reasons": reasons,
        "metrics": {"step_cv": 1.34, "keep_ratio": 0.15},
    }))
    _write_tum(d / "results" / "est_tum.txt")
    _write_ply(d / "results" / "map_preview.ply")
    (d / "pipeline_log.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"ts": 1.0, "stage": "ingest", "status": "ok", "seconds": 3.2,
         "detail": {"n_frames": 240, "reference_usable": True}},
        {"ts": 2.0, "stage": "recon", "status": "ok", "seconds": 142.7,
         "detail": {"submaps": 6, "loop_closures": 2}},
        {"ts": 3.0, "stage": "score", "status": "ok", "seconds": 0.4,
         "detail": {"ate_rmse_m": 0.084}},
        {"ts": 4.0, "stage": "qc", "status": "refused", "seconds": 0.1,
         "detail": {"level": "needs_review"}},
    ]) + "\n")
    return d, reasons


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@requires_matplotlib
def test_full_session_report(tmp_path: Path) -> None:
    d, reasons = _full_session(tmp_path)

    out = generate_report(d)

    assert out == d / "report.html"
    assert out.exists()
    html = out.read_text()

    # confidence level in words + machine level string both present
    assert "Needs review" in html
    # export decision surfaced
    assert "export allowed: yes" in html

    # both matplotlib figures embedded as base64 PNG data URIs (trajectory + map)
    assert html.count("data:image/png;base64,") == 2

    # mandatory odometry caveat, verbatim
    assert CAVEAT in html

    # every reason string, verbatim — reasons contain '<'/'>' which are correctly
    # HTML-escaped in the page, so check against an unescaped view of the content
    decoded = htmllib.unescape(html)
    for r in reasons:
        assert r in decoded

    # metrics that must be visible
    assert "180 / 240" in html          # frames kept / total
    assert "142.7" in html              # gpu seconds
    # self-contained: no external network references
    assert "http://" not in html and "https://" not in html


def test_empty_session_degrades(tmp_path: Path) -> None:
    d = tmp_path / "empty_session"
    d.mkdir()

    out = generate_report(d)

    assert out.exists()
    html = out.read_text()
    # never crashes, still names itself and shows the not-yet-gated banner
    assert "empty_session" in html
    assert "Not yet gated" in html
    # graceful "not available" notes where inputs are missing
    assert "not available" in html.lower()
    # the caveat is mandatory even with no consistency data
    assert CAVEAT in html
    # no figures could be produced, so no embedded images
    assert "data:image/png;base64," not in html


def test_refused_session_shows_red_band_and_solver_error(tmp_path: Path) -> None:
    d = tmp_path / "collapsed_tour"
    d.mkdir()
    (d / "results").mkdir()

    solver_msg = "cholesky factorization failed: matrix not positive definite"
    (d / "results" / "recon_summary.json").write_text(json.dumps({
        "frames_total": 300, "frames_kept": 40, "submaps": 9, "loop_closures": 0,
        "n_pose_sign_repairs": 45, "n_poses_written": 300, "solver_error": solver_msg,
        "export_errors": ["est_tum write skipped"], "gpu_seconds": 88.0, "config": {},
    }))
    reason = f"solver runtime failure: {solver_msg}"
    (d / "confidence.json").write_text(json.dumps({
        "level": "out_of_class", "export_allowed": False, "reasons": [reason],
        "metrics": {},
    }))
    (d / "pipeline_log.jsonl").write_text(json.dumps({
        "ts": 1.0, "stage": "qc", "status": "refused", "seconds": 0.1,
        "detail": {"error": "solver"},
    }) + "\n")

    out = generate_report(d)

    assert out.exists()
    html = out.read_text()
    # out_of_class -> red band class present, level word shown
    assert "band-ooc" in html
    assert "Out of class" in html
    assert "export allowed: no" in html
    # the solver error text is surfaced (both in the reason and the recon table)
    assert solver_msg in html
    assert reason in html


def test_report_renders_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    """A matplotlib-free environment must still get the whole page.

    `server.oreos.recordings.report` used to `import matplotlib; matplotlib.use("Agg")` at
    module scope, which made *importing* it fatal wherever matplotlib is absent
    (the GPU-free CI subset) — a collection error that took the entire suite
    down. The import is lazy now; the figures degrade to a note and every other
    section renders normally.
    """
    d, reasons = _full_session(tmp_path)
    # simulate "matplotlib not installed" by poisoning the cached lazy accessor
    monkeypatch.setattr(report_mod, "_PYPLOT_TRIED", True)
    monkeypatch.setattr(report_mod, "_PYPLOT", None)

    out = generate_report(d)

    html = out.read_text()
    assert "data:image/png;base64," not in html      # no figures
    assert PLOTS_UNAVAILABLE_NOTE in html            # ...and we say why
    # everything that does not need matplotlib is still there
    assert "Needs review" in html
    assert CAVEAT in html
    assert "180 / 240" in html
    decoded = htmllib.unescape(html)
    for r in reasons:
        assert r in decoded


def test_report_module_imports_without_matplotlib() -> None:
    """Guard the actual regression: importing the module must not touch matplotlib.

    Re-import `server.oreos.recordings.report` with matplotlib blocked at the finder level;
    a module-level `import matplotlib` (or an attribute touch on a stub, which is
    what tests/export_fakes.py leaves in sys.modules) would raise here.
    """
    import importlib
    import sys

    class _Block:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "matplotlib":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Block()
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "matplotlib"}
    for k in saved:
        del sys.modules[k]
    sys.modules.pop("server.oreos.recordings.report", None)
    sys.meta_path.insert(0, blocker)
    try:
        mod = importlib.import_module("server.oreos.recordings.report")
        assert mod._pyplot() is None  # degrades instead of raising
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        sys.modules.pop("server.oreos.recordings.report", None)
        importlib.import_module("server.oreos.recordings.report")


# ---------------------------------------------------------------------------
# core-source provenance row (2026-07-24: images used to ship the operator's
# working tree silently; the report must name what actually ran)
# ---------------------------------------------------------------------------

class TestCoreSourceRow:
    def render(self, prov):
        from server.oreos.recordings.report import _core_source_html

        return _core_source_html(prov)

    def test_tag_is_rendered_as_ok(self):
        html = self.render({"mode": "tag", "tag": "v2.2.0"})
        assert "v2.2.0" in html and "ok-note" in html

    def test_clean_local_checkout_names_the_sha(self):
        html = self.render({"mode": "local", "sha": "a" * 40, "dirty": False})
        assert "local aaaaaaaaa" in html and "err" not in html

    def test_dirty_local_checkout_is_flagged(self):
        html = self.render({"mode": "local", "sha": "b" * 40, "dirty": True,
                            "dirty_files": 3})
        assert "err" in html and "uncommitted" in html and "3" in html

    def test_unknown_dirty_state_is_flagged(self):
        html = self.render({"mode": "local", "sha": "c" * 40, "dirty": None})
        assert "err" in html and "unknown" in html

    def test_missing_provenance_degrades(self):
        assert "—" in self.render(None)
        assert "—" in self.render({})

    def test_row_appears_in_the_recon_table(self):
        from server.oreos.recordings.report import _recon_table_html

        html = _recon_table_html({"submaps": 3,
                                  "core_source": {"mode": "tag", "tag": "v2.2.0"}})
        assert "core source" in html and "v2.2.0" in html
