#!/usr/bin/env python3
"""Render the 3D side panel for the CLI demo composite.

Draws THE scene the demo actually measures: the simulator's ingested-scene
geometry (mcp/simulator/fixture.ts) as a synthesized point cloud, with the
desk-to-chair measurement highlighted. Annotations flip in sync with the
terminal beats, driven by the timestamps you pass in (read them off the
rendered terminal take):

  --t-cloud    reconstruction finished (cloud fades in)
  --t-measure  first measurement shown (gold line + "relative units" label)
  --t-metric   calibration applied (label flips to metres)
  --t-export   export beat (trajectory + bounding box appear)

Usage (from docs/demo, matplotlib + numpy + ffmpeg needed):

  python3 render_scene_panel.py --duration 78 --fps 12 --out panel.mp4 \
      --t-cloud 16 --t-measure 30 --t-metric 52 --t-export 68

Then composite side by side:

  ffmpeg -i demo.mp4 -i panel.mp4 -filter_complex \
    "[0:v][1:v]hstack=inputs=2" demo-composite.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Brand palette (docs/assets/hero.svg)
BG = "#071c15"
GOLD = "#d4a017"
GOLD_BRIGHT = "#f0c75e"
CORAL = "#e07070"
GREEN = "#66c294"
INK = "#e8e3d5"

# Simulator ingested-scene geometry (mcp/simulator/fixture.ts) — the scan the
# demo measures. Coordinates are the fixture's own; y is up.
DESK_C, DESK_E = np.array([1.2, 0.38, 0.4]), np.array([1.4, 0.75, 0.7])
CHAIR_C, CHAIR_E = np.array([0.3, 0.45, 1.1]), np.array([0.6, 0.95, 0.6])
BBOX_MIN, BBOX_MAX = np.array([-1.9, -0.1, -1.4]), np.array([2.3, 2.5, 1.7])
REL_DIST = float(np.linalg.norm(DESK_C - CHAIR_C))  # 1.14 relative units


def box_cloud(rng, center, extent, n, jitter=0.02):
    """Points on the surface of an axis-aligned box (a detected object)."""
    pts = []
    for _ in range(n):
        face = rng.integers(0, 3)
        sign = rng.choice([-1.0, 1.0])
        p = rng.uniform(-0.5, 0.5, 3)
        p[face] = 0.5 * sign
        pts.append(center + p * extent + rng.normal(0, jitter, 3))
    return np.array(pts)


def build_cloud(seed=44):
    rng = np.random.default_rng(seed)
    parts, colors, sizes = [], [], []

    floor = rng.uniform(0, 1, (2600, 3))
    floor = BBOX_MIN + floor * (BBOX_MAX - BBOX_MIN)
    floor[:, 1] = rng.normal(0.0, 0.015, len(floor))
    parts.append(floor); colors.append([GOLD] * len(floor)); sizes.append(1.1)

    for (fixed, val) in ((2, BBOX_MIN[2]), (0, BBOX_MAX[0])):  # two walls
        w = rng.uniform(0, 1, (1500, 3))
        w = BBOX_MIN + w * (BBOX_MAX - BBOX_MIN)
        w[:, fixed] = val + rng.normal(0, 0.015, len(w))
        parts.append(w); colors.append(["#9c8a4a"] * len(w)); sizes.append(1.0)

    for c, e, col, n in (
        (DESK_C, DESK_E, GOLD_BRIGHT, 1500),
        (CHAIR_C, CHAIR_E, CORAL, 1100),
        (np.array([1.25, 0.95, 0.35]), np.array([0.62, 0.4, 0.08]), INK, 500),
    ):
        b = box_cloud(rng, c, e, n)
        parts.append(b); colors.append([col] * len(b)); sizes.append(2.2)

    pts = np.concatenate(parts)
    cols = np.concatenate(colors)
    size = np.concatenate([np.full(len(p), s) for p, s in zip(parts, sizes)])
    order = rng.permutation(len(pts))  # so partial reveals look scan-like
    return pts[order], cols[order], size[order]


def trajectory():
    t = np.linspace(0, 2 * np.pi, 160)
    x = 0.2 + 1.5 * np.cos(t)
    z = 0.15 + 1.05 * np.sin(t)
    y = 1.45 + 0.12 * np.sin(2 * t)
    return np.stack([x, y, z], axis=1)


def render(args):
    pts, cols, size = build_cloud()
    traj = trajectory()
    total = int(round(args.duration * args.fps))
    tmp = tempfile.mkdtemp(prefix="panel_")

    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)
    fig.patch.set_facecolor(BG)

    for f in range(total):
        t = f / args.fps
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.set_box_aspect((BBOX_MAX - BBOX_MIN)[[0, 2, 1]])
        ax.view_init(elev=18, azim=32 + 14 * np.sin(2 * np.pi * t / 48))
        ax.set_xlim(BBOX_MIN[0], BBOX_MAX[0])
        ax.set_ylim(BBOX_MIN[2], BBOX_MAX[2])
        ax.set_zlim(BBOX_MIN[1], BBOX_MAX[1])

        # Cloud reveal: nothing before upload, sweeps in around t_cloud.
        if t < args.t_cloud - 6:
            frac = 0.0
        else:
            frac = min(1.0, max(0.03, (t - (args.t_cloud - 6)) / 6.0))
        n = int(len(pts) * frac)
        if n:
            ax.scatter(pts[:n, 0], pts[:n, 2], pts[:n, 1], c=cols[:n],
                       s=size[:n], depthshade=False, linewidths=0)

        title = "the scene being measured"
        sub = "(simulator fixture geometry)"
        label = None
        if t < args.t_cloud:
            status, scol = "reconstructing…", INK
        elif t < args.t_measure:
            status, scol = "scan sim-scan-01 persisted", GREEN
        elif t < args.t_metric:
            status, scol = "measuring desk → chair", GOLD_BRIGHT
            label = (f"{REL_DIST:.2f} relative units", "no metric anchor yet: NOT metres", CORAL)
        elif t < args.t_export:
            status, scol = "anchored with one real distance", GREEN
            label = ("1.60 m", 'units: "m" · scale from your calibration', GREEN)
        else:
            status, scol = "exporting LeRobot / GR00T dataset", GOLD_BRIGHT
            label = ("1.60 m", "trajectory + cloud + splat → zip", GREEN)

        if t >= args.t_measure and n:
            seg = np.stack([DESK_C, CHAIR_C])
            lc = CORAL if t < args.t_metric else GREEN
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=lc, linewidth=2.4)
            ax.scatter(seg[:, 0], seg[:, 2], seg[:, 1], color=lc, s=42,
                       depthshade=False)

        if t >= args.t_export:
            ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color=CORAL,
                    linewidth=1.2, linestyle=(0, (1, 3)))

        fig.text(0.05, 0.955, title, color=INK, fontsize=13, family="monospace")
        fig.text(0.05, 0.915, sub, color=GOLD, fontsize=9, family="monospace")
        fig.text(0.05, 0.075, status, color=scol, fontsize=11, family="monospace")
        if label:
            big, small, col = label
            fig.text(0.95, 0.86, big, color=col, fontsize=22, family="monospace",
                     ha="right", weight="bold")
            fig.text(0.95, 0.815, small, color=INK, fontsize=9,
                     family="monospace", ha="right")

        fig.savefig(os.path.join(tmp, f"f_{f:05d}.png"), facecolor=BG)

    subprocess.run([
        args.ffmpeg, "-y", "-v", "error", "-framerate", str(args.fps),
        "-i", os.path.join(tmp, "f_%05d.png"),
        "-pix_fmt", "yuv420p", args.out,
    ], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out} ({total} frames @ {args.fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=700)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--t-cloud", type=float, required=True)
    ap.add_argument("--t-measure", type=float, required=True)
    ap.add_argument("--t-metric", type=float, required=True)
    ap.add_argument("--t-export", type=float, required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--out", default="panel.mp4")
    render(ap.parse_args())
