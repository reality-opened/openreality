"""Oreos session report — one self-contained HTML page telling the whole story.

`generate_report(session_dir)` reads whatever the pipeline has produced so far
(ingest manifest, recon summary, consistency scoring, the QC verdict, the TUM
trajectory, an optional map preview cloud, and the stage-by-stage log) and
renders `report.html`: a confidence banner (the hero), a reference-vs-estimate
trajectory overlay, a top-down map glance, the recon + consistency metric tables
(with the mandatory "reference is odometry, NOT ground truth" caveat), and the
pipeline timeline.

Every input is OPTIONAL. Nothing here raises on a missing/partial/corrupt file —
each section degrades to a calm "not available" note so an operator can open the
page after any stage and see exactly how far the run got. The output is a single
file with inline CSS and base64-embedded PNGs: no external requests, no JS
frameworks, readable in both light and dark (prefers-color-scheme).

Dependency budget: stdlib + numpy, plus two OPTIONAL extras, both imported lazily
inside the functions that need them so importing this module never requires them:
matplotlib (Agg) draws the two figures — without it the page still renders, with
the figure blocks degrading to a "figures omitted" note; open3d reads the PLY when
present (a hand-rolled PLY parser is the fallback, so the map glance only vanishes
if neither can read the cloud). Keeping matplotlib lazy is deliberate: a bare
module-level `import matplotlib; matplotlib.use("Agg")` made merely *importing*
this module fatal in the GPU-free test/CI environment, which does not ship it.

Design note: the two matplotlib figures follow the platform data-viz palette —
the trajectory panel is an *emphasis* form (the estimate is the subject in the
blue accent; the odometry reference is context in neutral gray), and the map
glance is a *sequential* height encoding on the blue ramp. Figures render on a
transparent background with mode-invariant muted axis ink so a single baked PNG
reads on either page theme.
"""

from __future__ import annotations

import base64
import html
import io
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# -- palette (platform data-viz reference instance) -------------------------
# Marks chosen to read on BOTH a light (#fcfcfb) and a dark (#1a1a19) surface,
# since one baked PNG serves both page themes.
ACCENT = "#2a78d6"        # blue — the reconstruction (the subject)
REFERENCE_INK = "#898781"  # muted, mode-invariant — the odometry reference (context)
AXIS_INK = "#898781"       # muted axis/label ink (same value in both modes)
GRID_INK = "#b8b7b1"
# Blue sequential ramp, narrowed to a mid-band so neither end vanishes on either
# surface (height "glance", not a precise readout). Built lazily by _height_cmap().
HEIGHT_RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#184f95"]

PLOTS_UNAVAILABLE_NOTE = (
    "Figures omitted — matplotlib is not installed in this environment. "
    "Every other section below is unaffected."
)


# -- lazy matplotlib --------------------------------------------------------
# matplotlib is optional (see module docstring). Import it INSIDE the plotting
# helpers, select the headless Agg backend there, and cache the result, so that
# `import server.oreos.recordings.report` costs nothing and works in a matplotlib-free env.

_PYPLOT: object | None = None
_PYPLOT_TRIED = False


def _pyplot():
    """`matplotlib.pyplot` on the Agg backend, or None when unavailable."""
    global _PYPLOT, _PYPLOT_TRIED
    if _PYPLOT_TRIED:
        return _PYPLOT
    _PYPLOT_TRIED = True
    try:
        import matplotlib  # noqa: PLC0415 — optional, deliberately lazy

        matplotlib.use("Agg")  # headless: no display, deterministic PNG bytes
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — absent or broken install: degrade, never raise
        _PYPLOT = None
    else:
        _PYPLOT = plt
    return _PYPLOT


def _height_cmap():
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    return LinearSegmentedColormap.from_list("oreos_height", HEIGHT_RAMP)

LEVEL_META = {
    "high_confidence": ("High confidence", "band-high"),
    "needs_review": ("Needs review", "band-review"),
    "out_of_class": ("Out of class", "band-ooc"),
}
STATUS_DOT = {  # pipeline stage status -> css dot class
    "ok": "dot-ok",
    "skipped": "dot-skip",
    "refused": "dot-refused",
    "error": "dot-error",
}

CAVEAT = (
    "Reference is the robot's own odometry — NOT ground truth. "
    "Sim(3)-aligned; scale is relative."
)
PROVENANCE = (
    "EXP-36-calibrated QC; see experiments/exp36_oreos_dimos_spike "
    "+ 2026-07-24-oreos-on-dimos-plan.md"
)


# ===========================================================================
# tiny robust readers (every one returns a safe default, never raises)
# ===========================================================================

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — missing/corrupt is expected; degrade
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001 — skip a bad line, keep the rest
                continue
    except Exception:  # noqa: BLE001
        return []
    return rows


def _load_tum_positions(path: Path) -> np.ndarray:
    """Nx3 positions from a TUM file (ts tx ty tz qx qy qz qw); comments skipped."""
    pos: list[list[float]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split()
            if len(v) >= 4:
                try:
                    pos.append([float(v[1]), float(v[2]), float(v[3])])
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001
        return np.zeros((0, 3))
    return np.asarray(pos, dtype=float) if pos else np.zeros((0, 3))


# -- PLY reading: open3d if importable, else a small hand-rolled parser -----

_PLY_STRUCT = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _read_ply_manual(path: Path) -> np.ndarray:
    """Parse xyz from an ascii or binary_little_endian PLY. Best-effort; [] on doubt."""
    try:
        raw = path.read_bytes()
    except Exception:  # noqa: BLE001
        return np.zeros((0, 3))
    end = raw.find(b"end_header")
    if end < 0:
        return np.zeros((0, 3))
    header_end = raw.find(b"\n", end) + 1
    header = raw[:header_end].decode("ascii", "replace")
    fmt = "ascii"
    n_vert = 0
    props: list[tuple[str, str]] = []  # (type, name), in file order for `vertex`
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                n_vert = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3:
            props.append((parts[-2], parts[-1]))
    names = [p[1] for p in props]
    if not ({"x", "y", "z"} <= set(names)) or n_vert <= 0:
        return np.zeros((0, 3))
    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")

    if fmt.startswith("ascii"):
        vals: list[list[float]] = []
        body = raw[header_end:].decode("ascii", "replace").split()
        stride = len(props)
        for r in range(n_vert):
            base = r * stride
            if base + max(xi, yi, zi) >= len(body):
                break
            try:
                vals.append([float(body[base + xi]), float(body[base + yi]),
                             float(body[base + zi])])
            except ValueError:
                continue
        return np.asarray(vals, dtype=float) if vals else np.zeros((0, 3))

    # binary_little_endian (big-endian PLYs are rare in this pipeline; skip them)
    if "little_endian" not in fmt:
        return np.zeros((0, 3))
    codes = []
    for ptype, _name in props:
        sc = _PLY_STRUCT.get(ptype)
        if sc is None:
            return np.zeros((0, 3))  # a list/unknown property -> bail to a skip
        codes.append(sc)
    rec_fmt = "<" + "".join(c for c, _ in codes)
    rec_size = sum(sz for _, sz in codes)
    body = raw[header_end:]
    if len(body) < rec_size * n_vert:
        n_vert = len(body) // rec_size if rec_size else 0
    out = np.empty((n_vert, 3), dtype=float)
    off = 0
    for r in range(n_vert):
        rec = struct.unpack_from(rec_fmt, body, off)
        out[r] = (rec[xi], rec[yi], rec[zi])
        off += rec_size
    return out


def _read_ply(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    try:
        import open3d as o3d  # noqa: PLC0415 — optional, guarded
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=float)
        if len(pts):
            return pts
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 — corrupt cloud / reader hiccup
        pass
    return _read_ply_manual(path)


# ===========================================================================
# figures -> base64 PNG data URIs
# ===========================================================================

def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    _pyplot().close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def _style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_INK)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=AXIS_INK, labelsize=8, length=3)
    ax.xaxis.label.set_color(AXIS_INK)
    ax.yaxis.label.set_color(AXIS_INK)
    ax.title.set_color(AXIS_INK)
    ax.grid(True, color=GRID_INK, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def _trajectory_uri(consistency: dict | None, est_pos: np.ndarray) -> str | None:
    """Top-down overlay: reference (gray) vs aligned estimate (blue accent).

    Falls back to the estimate alone when no usable reference exists.
    """
    ref = None
    aligned = None
    if consistency and consistency.get("reference_available"):
        rp = consistency.get("reference_positions")
        ap = consistency.get("aligned_positions")
        if rp:
            ref = np.asarray(rp, dtype=float)
        if ap:
            aligned = np.asarray(ap, dtype=float)

    have_overlay = ref is not None and len(ref) and aligned is not None and len(aligned)
    if not have_overlay and (est_pos is None or len(est_pos) == 0):
        return None
    plt = _pyplot()
    if plt is None:  # matplotlib absent — caller renders the "figures omitted" note
        return None

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if have_overlay:
        ax.plot(ref[:, 0], ref[:, 1], color=REFERENCE_INK, linewidth=1.6,
                label="odometry reference", zorder=2)
        ax.plot(aligned[:, 0], aligned[:, 1], color=ACCENT, linewidth=2.0,
                label="reconstruction (aligned)", zorder=3)
        ax.scatter([aligned[0, 0]], [aligned[0, 1]], s=26, color=ACCENT,
                   edgecolor="none", zorder=4)
        ax.set_title("Trajectory overlay (top-down, x–y)", fontsize=9, loc="left")
        leg = ax.legend(loc="best", fontsize=8, frameon=False)
        for txt in leg.get_texts():
            txt.set_color(AXIS_INK)
    else:
        ax.plot(est_pos[:, 0], est_pos[:, 1], color=ACCENT, linewidth=2.0,
                label="estimated trajectory", zorder=3)
        ax.scatter([est_pos[0, 0]], [est_pos[0, 1]], s=26, color=ACCENT,
                   edgecolor="none", zorder=4)
        ax.set_title("Estimated trajectory (top-down, x–y) — no reference",
                     fontsize=9, loc="left")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    _style_axes(ax)
    return _fig_to_uri(fig)


def _map_uri(points: np.ndarray) -> str | None:
    """Top-down scatter of the map preview cloud, colored by height (z)."""
    if points is None or len(points) == 0:
        return None
    plt = _pyplot()
    if plt is None:  # matplotlib absent — caller renders the "figures omitted" note
        return None
    pts = points
    if len(pts) > 50_000:
        idx = np.linspace(0, len(pts) - 1, 50_000).astype(int)
        pts = pts[idx]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sc = ax.scatter(x, y, c=z, cmap=_height_cmap(), s=2, linewidths=0, alpha=0.85)
    ax.set_title("Map glance (top-down, colored by height)", fontsize=9, loc="left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    _style_axes(ax)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("height z (m)", color=AXIS_INK, fontsize=8)
    cbar.ax.tick_params(colors=AXIS_INK, labelsize=7)
    cbar.outline.set_edgecolor(AXIS_INK)
    return _fig_to_uri(fig)


# ===========================================================================
# HTML fragments
# ===========================================================================

def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v, suffix: str = "", nd: int = 3) -> str:
    if v is None:
        return "<span class='na'>—</span>"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{_esc(v)}{suffix}"


def _banner_html(conf: dict | None) -> str:
    if not conf:
        return (
            "<section class='banner band-none'>"
            "<div class='band-level'>Not yet gated</div>"
            "<p class='band-sub'>No QC verdict on disk yet "
            "(<code>confidence.json</code> missing). Run the qc stage.</p>"
            "</section>"
        )
    level = conf.get("level", "")
    word, cls = LEVEL_META.get(level, (level or "unknown", "band-none"))
    allowed = conf.get("export_allowed")
    chip_cls = "chip-yes" if allowed else "chip-no"
    chip_txt = "yes" if allowed else "no"
    reasons = conf.get("reasons") or []
    if reasons:
        items = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
        reasons_html = f"<ul class='reasons'>{items}</ul>"
    else:
        reasons_html = ("<p class='band-sub'>No known failure signature detected. "
                        "This is not a verified-accurate claim — it means no gate "
                        "tripped.</p>")
    return (
        f"<section class='banner {cls}'>"
        f"<div class='band-head'>"
        f"<div class='band-level'>{_esc(word)}</div>"
        f"<div class='chip {chip_cls}'>export allowed: {chip_txt}</div>"
        f"</div>"
        f"{reasons_html}"
        f"</section>"
    )


def _kv_rows(rows: list[tuple[str, str]]) -> str:
    return "".join(
        f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>" for k, v in rows
    )


def _core_source_html(prov: dict | None) -> str:
    """Which core produced this recon (build-time stamp in recon_summary.json).

    A reconstruction whose SLAM library cannot be named is not attributable — and
    until 2026-07-24 every Oreos image silently shipped the operator's working
    tree. A dirty local checkout is therefore rendered as a warning, not a note.
    """
    if not isinstance(prov, dict) or not prov:
        return "<span class='na'>—</span>"
    mode = prov.get("mode")
    if mode == "tag":
        return f"<span class='ok-note'>tag {_esc(prov.get('tag'))}</span>"
    if mode == "local":
        sha = prov.get("sha")
        short = _esc(sha[:9]) if isinstance(sha, str) else "unknown sha"
        if prov.get("dirty"):
            return (f"<span class='err'>local {short} + uncommitted changes "
                    f"({_esc(prov.get('dirty_files'))} files)</span>")
        if prov.get("dirty") is None:
            return f"<span class='err'>local {short} (dirty state unknown)</span>"
        return f"local {short}"
    return _esc(prov.get("note") or mode or "unknown")


def _recon_table_html(recon: dict | None) -> str:
    if not recon:
        return ("<h2>Reconstruction</h2>"
                "<p class='na'>recon_summary.json not available.</p>")
    kept = recon.get("frames_kept")
    total = recon.get("frames_total")
    frames = (f"{_esc(kept)} / {_esc(total)}"
              if kept is not None or total is not None
              else "<span class='na'>—</span>")
    solver = recon.get("solver_error")
    solver_cell = (f"<span class='err'>{_esc(solver)}</span>" if solver
                   else "<span class='ok-note'>none</span>")
    exp_errs = recon.get("export_errors") or []
    exp_cell = (f"<span class='err'>{len(exp_errs)}</span>" if exp_errs
                else "0")
    rows = [
        ("core source", _core_source_html(recon.get("core_source"))),
        ("frames kept / total", frames),
        ("submaps", _fmt(recon.get("submaps"))),
        ("loop closures", _fmt(recon.get("loop_closures"))),
        ("poses written", _fmt(recon.get("n_poses_written"))),
        ("pose sign repairs", _fmt(recon.get("n_pose_sign_repairs"))),
        ("gpu seconds", _fmt(recon.get("gpu_seconds"), " s", nd=1)),
        ("solver error", solver_cell),
        ("export errors", exp_cell),
    ]
    return f"<h2>Reconstruction</h2><table class='metrics'>{_kv_rows(rows)}</table>"


def _consistency_table_html(cons: dict | None) -> str:
    caveat = f"<p class='caveat'>{CAVEAT}</p>"  # trusted constant; render verbatim
    if not cons:
        return ("<h2>Consistency vs odometry</h2>"
                "<p class='na'>consistency.json not available.</p>" + caveat)
    if not cons.get("reference_available"):
        return ("<h2>Consistency vs odometry</h2>"
                "<p class='na'>No usable odometry reference — trajectory "
                "consistency was not scored for this session.</p>" + caveat)
    ate = cons.get("ate_rmse_m")
    ate_med = cons.get("ate_median_m")
    ate_cell = "<span class='na'>—</span>"
    if ate is not None:
        ate_cell = f"{ate:.3f} m (rmse)"
        if ate_med is not None:
            ate_cell += f" &middot; {ate_med:.3f} m (median)"
    rows = [
        ("ATE", ate_cell),
        ("ATE % of extent", _fmt(cons.get("ate_pct_extent"), " %", nd=2)),
        ("ATE % of path", _fmt(cons.get("ate_pct_path"), " %", nd=2)),
        ("extent", _fmt(cons.get("extent_m"), " m", nd=2)),
        ("path length", _fmt(cons.get("gt_path_length_m"), " m", nd=2)),
        ("Umeyama scale", _fmt(cons.get("umeyama_scale"), nd=4)),
        ("windowed scale CV", _fmt(cons.get("windowed_scale_cv"), nd=4)),
    ]
    return (f"<h2>Consistency vs odometry</h2>"
            f"<table class='metrics'>{_kv_rows(rows)}</table>{caveat}")


def _timeline_html(log_rows: list[dict]) -> str:
    if not log_rows:
        return ("<h2>Pipeline timeline</h2>"
                "<p class='na'>pipeline_log.jsonl not available.</p>")
    body = []
    for e in log_rows:
        status = str(e.get("status", ""))
        dot = STATUS_DOT.get(status, "dot-skip")
        detail = e.get("detail")
        if isinstance(detail, dict) and detail:
            snippet = ", ".join(f"{k}={_short(v)}" for k, v in list(detail.items())[:3])
        elif detail:
            snippet = _short(detail)
        else:
            snippet = ""
        secs = e.get("seconds")
        secs_txt = f"{secs:.1f}" if isinstance(secs, (int, float)) else _esc(secs)
        body.append(
            "<tr>"
            f"<td class='stage'>{_esc(e.get('stage'))}</td>"
            f"<td><span class='dot {dot}'></span>{_esc(status)}</td>"
            f"<td class='num'>{secs_txt}</td>"
            f"<td class='detail'>{_esc(snippet)}</td>"
            "</tr>"
        )
    return (
        "<h2>Pipeline timeline</h2>"
        "<table class='timeline'><thead><tr>"
        "<th>stage</th><th>status</th><th>seconds</th><th>detail</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _short(v, limit: int = 60) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _overview_html(manifest: dict | None) -> str:
    if not manifest:
        return ""
    span = manifest.get("span_s")
    span_txt = f"{span:.1f} s" if isinstance(span, (int, float)) else "—"
    ref_usable = manifest.get("reference_usable")
    ref_txt = ("usable" if ref_usable else "unusable") if ref_usable is not None else "—"
    streams = manifest.get("streams") or []
    bits = [
        ("source", _esc(manifest.get("source") or "—")),
        ("frames", _esc(manifest.get("n_frames") if manifest.get("n_frames") is not None else "—")),
        ("span", span_txt),
        ("odom stream", _esc(manifest.get("odom_stream") or "—")),
        ("reference poses", _esc(manifest.get("n_reference_poses")
                                 if manifest.get("n_reference_poses") is not None else "—")),
        ("reference", ref_txt),
        ("streams", str(len(streams))),
    ]
    chips = "".join(f"<span class='meta'><b>{k}</b>&nbsp;{v}</span>" for k, v in bits)
    return f"<div class='overview'>{chips}</div>"


def _figure_block(uri: str | None, title: str, missing: str) -> str:
    if uri:
        return (f"<figure class='plot'><figcaption>{_esc(title)}</figcaption>"
                f"<img alt='{_esc(title)}' src='{uri}'></figure>")
    return (f"<figure class='plot'><figcaption>{_esc(title)}</figcaption>"
            f"<p class='na'>{_esc(missing)}</p></figure>")


# ===========================================================================
# stylesheet
# ===========================================================================

_CSS = """
:root {
  --page: #f9f9f7; --surface: #ffffff; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --hair: #e1e0d9; --hair2: rgba(11,11,11,0.10);
  --accent: #2a78d6; --err: #c0271f; --ok: #0a7a0a;
  --band-high-bg: #127a12; --band-high-tx: #ffffff;
  --band-review-bg: #f2ac12; --band-review-tx: #241c00;
  --band-ooc-bg: #c0271f; --band-ooc-tx: #ffffff;
  --band-none-bg: #6b6a66; --band-none-tx: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --hair: #2c2c2a; --hair2: rgba(255,255,255,0.10);
    --accent: #3987e5; --err: #f07a72; --ok: #4fbf4f;
    --band-high-bg: #157a15; --band-review-bg: #e0a015;
    --band-ooc-bg: #cf3b34; --band-none-bg: #55544f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 900px; margin: 0 auto; padding: 32px 22px 60px; }
header.top { border-bottom: 1px solid var(--hair); padding-bottom: 16px; margin-bottom: 22px; }
.eyebrow { color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
h1 { font-size: 26px; margin: 4px 0 2px; font-weight: 650; }
.gen { color: var(--ink2); font-size: 13px; }
.overview { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.meta {
  font-size: 12.5px; color: var(--ink2); background: var(--surface);
  border: 1px solid var(--hair2); border-radius: 6px; padding: 3px 9px;
}
.meta b { color: var(--muted); font-weight: 600; }
section, .section { margin: 26px 0; }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em;
     color: var(--ink2); margin: 0 0 12px; font-weight: 600; }

/* confidence banner — the hero */
.banner { border-radius: 12px; padding: 20px 22px; margin: 20px 0 30px;
  box-shadow: 0 1px 0 var(--hair2); }
.band-high { background: var(--band-high-bg); color: var(--band-high-tx); }
.band-review { background: var(--band-review-bg); color: var(--band-review-tx); }
.band-ooc { background: var(--band-ooc-bg); color: var(--band-ooc-tx); }
.band-none { background: var(--band-none-bg); color: var(--band-none-tx); }
.band-head { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; }
.band-level { font-size: 27px; font-weight: 700; letter-spacing: -.01em; }
.chip { font-size: 13px; font-weight: 600; padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.22); border: 1px solid rgba(255,255,255,.35); }
.band-review .chip { background: rgba(0,0,0,.14); border-color: rgba(0,0,0,.28); }
.chip-no { text-decoration: none; }
.band-sub { margin: 12px 0 0; font-size: 13.5px; opacity: .92; }
.band-sub code { font-size: 12.5px; }
ul.reasons { margin: 14px 0 0; padding-left: 20px; }
ul.reasons li { margin: 4px 0; font-size: 14px; font-weight: 500; }

/* figures */
.plots { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 720px) { .plots { grid-template-columns: 1fr; } }
figure.plot { margin: 0; background: var(--surface); border: 1px solid var(--hair);
  border-radius: 10px; padding: 12px; }
figure.plot figcaption { font-size: 12.5px; color: var(--ink2); margin-bottom: 8px; }
figure.plot img { display: block; width: 100%; height: auto; }

/* tables */
.tables { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 720px) { .tables { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.metrics th, table.metrics td { text-align: left; padding: 7px 6px;
  border-bottom: 1px solid var(--hair); vertical-align: top; }
table.metrics th { color: var(--ink2); font-weight: 500; width: 52%; }
table.metrics td { font-variant-numeric: tabular-nums; }
.caveat { margin: 12px 2px 0; font-size: 12.5px; color: var(--ink2);
  border-left: 3px solid var(--muted); padding-left: 10px; }
.na { color: var(--muted); }
.err { color: var(--err); font-weight: 600; }
.ok-note { color: var(--muted); }

/* timeline */
table.timeline { border: 1px solid var(--hair); border-radius: 8px; overflow: hidden; }
table.timeline th, table.timeline td { text-align: left; padding: 8px 10px;
  border-bottom: 1px solid var(--hair); }
table.timeline thead th { font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--muted); font-weight: 600; }
table.timeline .stage { font-weight: 600; }
table.timeline .num { font-variant-numeric: tabular-nums; color: var(--ink2); }
table.timeline .detail { color: var(--ink2); font-size: 12.5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 7px; vertical-align: middle; }
.dot-ok { background: #0ca30c; } .dot-skip { background: var(--muted); }
.dot-refused { background: #fab219; } .dot-error { background: #d03b3b; }

footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--hair);
  color: var(--muted); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


# ===========================================================================
# entry point
# ===========================================================================

def generate_report(session_dir: str | Path) -> Path:
    """Render <session_dir>/report.html from whatever the pipeline produced.

    Reads (all optional): ingest_manifest.json, results/recon_summary.json,
    consistency.json, confidence.json, results/est_tum.txt, results/map_preview.ply,
    pipeline_log.jsonl. Never raises on missing/partial inputs. Returns the path
    to the written report.html.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(session_dir / "ingest_manifest.json")
    recon = _read_json(session_dir / "results" / "recon_summary.json")
    cons = _read_json(session_dir / "consistency.json")
    conf = _read_json(session_dir / "confidence.json")
    log_rows = _read_jsonl(session_dir / "pipeline_log.jsonl")
    est_pos = _load_tum_positions(session_dir / "results" / "est_tum.txt")
    map_pts = _read_ply(session_dir / "results" / "map_preview.ply")

    traj_uri = _trajectory_uri(cons, est_pos)
    map_uri = _map_uri(map_pts)

    # Say *why* a figure is absent: no data, or no plotting library at all.
    no_traj = "No trajectory available (est_tum.txt / consistency missing)."
    no_map = "No map preview available (map_preview.ply missing or unreadable)."
    if _pyplot() is None:
        no_traj = no_map = PLOTS_UNAVAILABLE_NOTE

    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_name = session_dir.name or "session"

    body = f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Oreos — camera-only scene pipeline</div>
  <h1>{_esc(session_name)}</h1>
  <div class="gen">Report generated {_esc(gen_time)}</div>
  {_overview_html(manifest)}
</header>

{_banner_html(conf)}

<section class="section">
  <h2>Trajectory &amp; map</h2>
  <div class="plots">
    {_figure_block(traj_uri, "Trajectory overlay (top-down)", no_traj)}
    {_figure_block(map_uri, "Map glance (top-down, by height)", no_map)}
  </div>
</section>

<section class="section">
  <div class="tables">
    <div>{_recon_table_html(recon)}</div>
    <div>{_consistency_table_html(cons)}</div>
  </div>
</section>

<section class="section">
  {_timeline_html(log_rows)}
</section>

<footer>{_esc(PROVENANCE)}</footer>
</div>"""

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Oreos report — {_esc(session_name)}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )

    out = session_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
