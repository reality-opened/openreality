"""End-to-end test for server/export/isaac/writer.export_isaac (duck-typed fake solver).

Exercises the full orchestrator: gather cloud -> extract poses -> align -> author USD +
manifest. open3d is typically absent in this env, so this validates the points-only fallback
path (mesh reconstruction is verified on Modal/Linux). Skips if usd-core (pxr) is missing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tests.export_fakes as fakes  # also stubs a broken matplotlib

pytest.importorskip("pxr", reason="usd-core (pxr) not installed")

from pxr import Usd, UsdGeom  # noqa: E402

from server.export.isaac import writer as isaac_writer  # noqa: E402


def _fake_scan(scale_k=4.0):
    """A fake scan: a dominant floor + a few above points + 3 cameras above the floor."""
    rng = np.random.default_rng(0)
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(900, 3))
    floor[:, 2] = 0.0
    above = rng.uniform([-1, -1, 0.3], [1, 1, 1.8], size=(120, 3))
    pts = np.vstack([floor, above]) * scale_k  # up-to-scale SLAM cloud (axis-aligned)
    cols = rng.integers(0, 255, size=(len(pts), 3), dtype=np.uint8)

    # 3 cameras above the floor, looking around.
    cams = [
        (fakes.rot(0, 0, 0), np.array([0, 0, 1.5]) * scale_k),
        (fakes.rot(0, 0.3, 0), np.array([0.5, 0, 1.5]) * scale_k),
        (fakes.rot(0, -0.3, 0), np.array([-0.5, 0, 1.5]) * scale_k),
    ]
    submap = fakes.FakeSubmap(0, cams, points=pts, colors=cols)
    return fakes.FakeSolver(fakes.FakeMap([submap])), len(pts)


def test_export_isaac_full_tree(tmp_path):
    solver, n_points = _fake_scan(scale_k=4.0)
    isaac_dir = isaac_writer.export_isaac(
        solver, str(tmp_path), scan_id="fixture", scale=1.0 / 4.0, fps=10.0
    )
    assert isaac_dir is not None
    base = tmp_path / "fixture" / "isaac"
    assert (base / "scene.usd").exists()
    assert (base / "trajectory.usd").exists()
    assert (base / "manifest.json").exists()

    manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["alignment"]["metric"] is True
    assert manifest["alignment"]["scale_source"] == "user:factor"
    assert manifest["alignment"]["gravity_aligned"] is True
    assert manifest["num_keyframes"] == 3
    # No open3d in this env -> points-only fallback scene.
    assert manifest["geometry"]["representation"] == "points"
    assert manifest["geometry"]["num_points"] == n_points

    # The authored stages are valid and Z-up / metric.
    scene = Usd.Stage.Open(str(base / "scene.usd"))
    assert UsdGeom.GetStageUpAxis(scene) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(scene) == 1.0
    assert scene.GetPrimAtPath("/World/Cloud").IsA(UsdGeom.Points)

    traj = Usd.Stage.Open(str(base / "trajectory.usd"))
    cam = traj.GetPrimAtPath("/World/Camera")
    assert cam.IsA(UsdGeom.Camera)
    op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]
    assert len(op.GetTimeSamples()) == 3

    # Floor inliers were found (gravity from the floor plane, not a fallback).
    assert manifest["alignment"]["floor_inlier_fraction"] > 0.5


def test_export_isaac_unresolved_scale_is_flagged(tmp_path):
    solver, _ = _fake_scan(scale_k=4.0)
    isaac_writer.export_isaac(solver, str(tmp_path), scan_id="noscale", fps=10.0)
    manifest = json.loads((tmp_path / "noscale" / "isaac" / "manifest.json").read_text())
    assert manifest["alignment"]["metric"] is False
    assert manifest["alignment"]["scale_source"] == "none"
    assert "up-to-scale" in manifest["notes"]
