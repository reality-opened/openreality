#!/usr/bin/env python3
"""Render the synchronized 3D panel for the Codex demo.

With ``--splat``, the panel reads positions and colors from an actual ASCII or
binary little-endian PLY and draws the measured endpoints on that geometry. If
no PLY is supplied, it draws the deterministic simulator fixture and labels it
as such. This is a point-splat preview, not a photorealistic Gaussian raster.

Example (from docs/demo):

  python3 render_scene_panel.py --duration 78 --fps 12 \
      --t-cloud 16 --t-measure 30 --t-metric 52 --t-export 68 \
      --splat /path/to/splat.ply --terminal terminal.mp4 \
      --out panel.mp4 --composite demo.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BG = "#071c15"
GOLD = "#d4a017"
GOLD_BRIGHT = "#f0c75e"
CORAL = "#e07070"
GREEN = "#66c294"
INK = "#e8e3d5"
SH_C0 = 0.28209479177387814

# Simulator scene facts (mcp/simulator/fixture.ts). Coordinates are y-up.
DEFAULT_POINT_A = np.array([1.2, 0.38, 0.4])
DEFAULT_POINT_B = np.array([0.3, 0.45, 1.1])
DESK_EXTENT = np.array([1.4, 0.75, 0.7])
CHAIR_EXTENT = np.array([0.6, 0.95, 0.6])
FIXTURE_MIN = np.array([-1.9, -0.1, -1.4])
FIXTURE_MAX = np.array([2.3, 2.5, 1.7])

PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def box_cloud(rng, center, extent, n, jitter=0.02):
    """Generate points on an axis-aligned fixture object."""
    pts = []
    for _ in range(n):
        face = rng.integers(0, 3)
        sign = rng.choice([-1.0, 1.0])
        p = rng.uniform(-0.5, 0.5, 3)
        p[face] = 0.5 * sign
        pts.append(center + p * extent + rng.normal(0, jitter, 3))
    return np.array(pts)


def build_fixture_cloud(seed=44):
    """Build the explicit fallback used by the offline simulator take."""
    rng = np.random.default_rng(seed)
    parts, colors, sizes = [], [], []

    floor = rng.uniform(0, 1, (2600, 3))
    floor = FIXTURE_MIN + floor * (FIXTURE_MAX - FIXTURE_MIN)
    floor[:, 1] = rng.normal(0.0, 0.015, len(floor))
    parts.append(floor)
    colors.append([GOLD] * len(floor))
    sizes.append(1.1)

    for fixed, val in ((2, FIXTURE_MIN[2]), (0, FIXTURE_MAX[0])):
        wall = rng.uniform(0, 1, (1500, 3))
        wall = FIXTURE_MIN + wall * (FIXTURE_MAX - FIXTURE_MIN)
        wall[:, fixed] = val + rng.normal(0, 0.015, len(wall))
        parts.append(wall)
        colors.append(["#9c8a4a"] * len(wall))
        sizes.append(1.0)

    for center, extent, color, count in (
        (DEFAULT_POINT_A, DESK_EXTENT, GOLD_BRIGHT, 1500),
        (DEFAULT_POINT_B, CHAIR_EXTENT, CORAL, 1100),
        (np.array([1.25, 0.95, 0.35]), np.array([0.62, 0.4, 0.08]), INK, 500),
    ):
        box = box_cloud(rng, center, extent, count)
        parts.append(box)
        colors.append([color] * len(box))
        sizes.append(2.2)

    pts = np.concatenate(parts)
    cols = np.concatenate(colors)
    size = np.concatenate([np.full(len(part), s) for part, s in zip(parts, sizes)])
    order = rng.permutation(len(pts))
    return pts[order], cols[order], size[order]


def _read_ply_header(fh):
    if fh.readline().strip() != b"ply":
        raise ValueError("not a PLY file")

    fmt = None
    vertex_count = None
    vertex_props = []
    in_vertices = False
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        text = line.decode("ascii", errors="strict").strip()
        fields = text.split()
        if not fields or fields[0] in {"comment", "obj_info"}:
            continue
        if fields[0] == "format":
            fmt = fields[1]
        elif fields[0] == "element":
            in_vertices = fields[1] == "vertex"
            if in_vertices:
                vertex_count = int(fields[2])
        elif fields[0] == "property" and in_vertices:
            if fields[1] == "list":
                raise ValueError("list properties are unsupported in the vertex element")
            vertex_props.append((fields[2], fields[1]))
        elif fields[0] == "end_header":
            break

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"unsupported PLY format: {fmt}")
    if vertex_count is None or not vertex_props:
        raise ValueError("PLY has no vertex element")
    return fmt, vertex_count, vertex_props, fh.tell()


def _sample_indices(count, limit, seed):
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=limit, replace=False))


def load_ply_cloud(path, max_points=50_000, seed=44):
    """Load a bounded, deterministic sample from a point-cloud or 3DGS PLY."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    path = Path(path)
    with path.open("rb") as fh:
        fmt, count, props, data_offset = _read_ply_header(fh)
        if count < 1:
            raise ValueError("PLY contains no vertices")
        indices = _sample_indices(count, max_points, seed)
        names = [name for name, _ in props]
        prop_types = dict(props)

        if fmt == "binary_little_endian":
            try:
                dtype = np.dtype([(name, "<" + PLY_DTYPES[kind]) for name, kind in props])
            except KeyError as exc:
                raise ValueError(f"unsupported PLY scalar type: {exc.args[0]}") from exc
            mapped = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(count,))
            rows = mapped[indices]

            def column(name):
                return np.asarray(rows[name], dtype=np.float64)

        else:
            selected = []
            wanted = iter(indices.tolist())
            target = next(wanted, None)
            for row_index, raw in enumerate(fh):
                if target is None:
                    break
                if row_index == target:
                    selected.append([float(value) for value in raw.split()])
                    target = next(wanted, None)
            rows = np.asarray(selected, dtype=np.float64)
            if len(rows) != len(indices):
                raise ValueError(f"PLY ended after {len(rows)} sampled vertices; expected {len(indices)}")
            positions = {name: i for i, name in enumerate(names)}

            def column(name):
                return rows[:, positions[name]]

    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex element must contain x, y, and z")
    points = np.column_stack([column("x"), column("y"), column("z")])

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([column("red"), column("green"), column("blue")])
        color_types = {prop_types[name] for name in ("red", "green", "blue")}
        if color_types.issubset({"uchar", "uint8"}):
            rgb = rgb / 255.0
        elif color_types.issubset({"ushort", "uint16"}):
            rgb = rgb / 65535.0
        elif color_types.issubset({"float", "float32", "double", "float64"}):
            if np.nanmin(rgb) < 0.0 or np.nanmax(rgb) > 1.0:
                raise ValueError("floating-point PLY RGB properties must be normalized to 0..1")
        else:
            raise ValueError(f"unsupported PLY RGB property types: {sorted(color_types)}")
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([column("f_dc_0"), column("f_dc_1"), column("f_dc_2")])
        rgb = 0.5 + SH_C0 * sh
    else:
        rgb = np.tile(np.array([0.83, 0.63, 0.09]), (len(points), 1))
    rgb = np.clip(np.nan_to_num(rgb, nan=0.5), 0.0, 1.0)

    if "opacity" in names:
        logits = np.clip(column("opacity"), -10.0, 10.0)
        alpha = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.18, 1.0)
        colors = np.column_stack([rgb, alpha])
    else:
        colors = rgb

    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in names]
    if scale_names:
        scale = np.exp(np.clip(np.mean([column(name) for name in scale_names], axis=0), -12, 4))
        lo, hi = np.nanpercentile(scale, [5, 95])
        sizes = 0.6 + 5.4 * np.clip((scale - lo) / max(hi - lo, 1e-12), 0, 1)
    else:
        sizes = np.full(len(points), 1.2)

    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        raise ValueError("PLY contains no finite points")
    return points[finite], colors[finite], sizes[finite], count


def fixture_trajectory():
    t = np.linspace(0, 2 * np.pi, 160)
    x = 0.2 + 1.5 * np.cos(t)
    z = 0.15 + 1.05 * np.sin(t)
    y = 1.45 + 0.12 * np.sin(2 * t)
    return np.stack([x, y, z], axis=1)


def view_bounds(points, point_a, point_b):
    cloud_min, cloud_max = np.nanpercentile(points, [0.5, 99.5], axis=0)
    low = np.minimum(cloud_min, np.minimum(point_a, point_b))
    high = np.maximum(cloud_max, np.maximum(point_a, point_b))
    span = np.maximum(high - low, 1e-3)
    return low - span * 0.06, high + span * 0.06


def render(args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            'rendering requires compatible packages: python3 -m pip install "numpy<2" "matplotlib>=3.8"'
        ) from exc

    point_a = np.asarray(args.point_a, dtype=float)
    point_b = np.asarray(args.point_b, dtype=float)
    relative_distance = float(np.linalg.norm(point_a - point_b))

    if args.splat:
        points, colors, sizes, source_count = load_ply_cloud(
            args.splat, max_points=args.max_points, seed=args.seed
        )
        subtitle = f"point-splat preview · {Path(args.splat).name} · {len(points):,}/{source_count:,} points"
        is_fixture = False
    else:
        points, colors, sizes = build_fixture_cloud(args.seed)
        subtitle = "offline simulator fixture · not a reconstructed splat"
        is_fixture = True

    bounds_min, bounds_max = view_bounds(points, point_a, point_b)
    trajectory = fixture_trajectory()
    total = int(round(args.duration * args.fps))
    tmp = tempfile.mkdtemp(prefix="openreality-panel-")
    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)
    fig.patch.set_facecolor(BG)

    try:
        for frame in range(total):
            t = frame / args.fps
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(BG)
            ax.set_axis_off()
            ax.set_box_aspect((bounds_max - bounds_min)[[0, 2, 1]])
            ax.view_init(elev=args.elevation, azim=args.azimuth + 14 * np.sin(2 * np.pi * t / 48))
            ax.set_xlim(bounds_min[0], bounds_max[0])
            ax.set_ylim(bounds_min[2], bounds_max[2])
            ax.set_zlim(bounds_min[1], bounds_max[1])

            if t < args.t_cloud - 6:
                fraction = 0.0
            else:
                fraction = min(1.0, max(0.03, (t - (args.t_cloud - 6)) / 6.0))
            visible = int(len(points) * fraction)
            if visible:
                ax.scatter(
                    points[:visible, 0], points[:visible, 2], points[:visible, 1],
                    c=colors[:visible], s=sizes[:visible], depthshade=False, linewidths=0,
                )

            label = None
            if t < args.t_cloud:
                status, status_color = "reconstructing…", INK
            elif t < args.t_measure:
                status, status_color = f"scan {args.scan_id} persisted", GREEN
            elif t < args.t_metric:
                status, status_color = f"measuring {args.point_a_label} → {args.point_b_label}", GOLD_BRIGHT
                label = (f"{relative_distance:.2f} relative units", "no metric anchor yet: NOT metres", CORAL)
            elif t < args.t_export:
                status, status_color = "anchored with one real distance", GREEN
                label = (f"{args.metric_distance:.2f} m", 'units: "m" · scale from your calibration', GREEN)
            else:
                status, status_color = "exporting LeRobot / GR00T dataset", GOLD_BRIGHT
                label = (f"{args.metric_distance:.2f} m", "trajectory + cloud + splat → zip", GREEN)

            if t >= args.t_measure and visible:
                segment = np.stack([point_a, point_b])
                line_color = CORAL if t < args.t_metric else GREEN
                ax.plot(segment[:, 0], segment[:, 2], segment[:, 1], color=line_color, linewidth=3.0)
                ax.scatter(
                    segment[:, 0], segment[:, 2], segment[:, 1],
                    color=line_color, edgecolors=INK, linewidths=0.5, s=52, depthshade=False,
                )
                ax.text(*point_a[[0, 2, 1]], f"  {args.point_a_label}", color=INK, fontsize=8)
                ax.text(*point_b[[0, 2, 1]], f"  {args.point_b_label}", color=INK, fontsize=8)

            if t >= args.t_export and is_fixture:
                ax.plot(
                    trajectory[:, 0], trajectory[:, 2], trajectory[:, 1],
                    color=CORAL, linewidth=1.2, linestyle=(0, (1, 3)),
                )

            fig.text(0.05, 0.955, "the scene being measured", color=INK, fontsize=13, family="monospace")
            fig.text(0.05, 0.915, subtitle, color=GOLD, fontsize=8, family="monospace")
            fig.text(0.05, 0.075, status, color=status_color, fontsize=11, family="monospace")
            if label:
                big, small, color = label
                fig.text(0.95, 0.86, big, color=color, fontsize=22, family="monospace", ha="right", weight="bold")
                fig.text(0.95, 0.815, small, color=INK, fontsize=9, family="monospace", ha="right")

            fig.savefig(os.path.join(tmp, f"f_{frame:05d}.png"), facecolor=BG)

        subprocess.run(
            [
                args.ffmpeg, "-y", "-v", "error", "-framerate", str(args.fps),
                "-i", os.path.join(tmp, "f_%05d.png"), "-pix_fmt", "yuv420p", args.out,
            ],
            check=True,
        )
    finally:
        plt.close(fig)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"wrote {args.out} ({total} frames @ {args.fps} fps)")
    if args.composite:
        if not args.terminal:
            raise ValueError("--composite requires --terminal")
        subprocess.run(
            [
                args.ffmpeg, "-y", "-v", "error", "-i", args.terminal, "-i", args.out,
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
                "-map", "0:a?", "-shortest", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", args.composite,
            ],
            check=True,
        )
        print(f"wrote {args.composite} (Codex + scene)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--t-cloud", type=float, required=True)
    parser.add_argument("--t-measure", type=float, required=True)
    parser.add_argument("--t-metric", type=float, required=True)
    parser.add_argument("--t-export", type=float, required=True)
    parser.add_argument("--splat", help="actual point-cloud or 3DGS PLY to preview")
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--point-a", type=float, nargs=3, default=DEFAULT_POINT_A)
    parser.add_argument("--point-b", type=float, nargs=3, default=DEFAULT_POINT_B)
    parser.add_argument("--point-a-label", default="desk")
    parser.add_argument("--point-b-label", default="chair")
    parser.add_argument("--metric-distance", type=float, default=1.6)
    parser.add_argument("--scan-id", default="sim-scan-01")
    parser.add_argument("--elevation", type=float, default=18)
    parser.add_argument("--azimuth", type=float, default=32)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out", default="panel.mp4")
    parser.add_argument("--terminal", help="VHS terminal MP4 to place on the left")
    parser.add_argument("--composite", help="output MP4 for the side-by-side composite")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
