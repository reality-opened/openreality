"""Unit tests for server/export/isaac/usd_writer.py.

Author each USD stage, reopen it with ``pxr.Usd.Stage.Open``, and assert the Isaac
conventions hold: Z-up + meters, a vertex-colored mesh with a static collider, and — the
trap — an animated camera whose world position/forward match the original CV camera (the
CV->USD axis flip applied correctly).

Skipped automatically if ``usd-core`` (``pxr``) is not installed (the package authors lazily;
real authoring is also exercised on Modal/Linux).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pxr", reason="usd-core (pxr) not installed")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from server.export.isaac import usd_writer  # noqa: E402


def _quad_mesh():
    verts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    vcolors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], np.uint8)
    return verts, faces, vcolors


def test_scene_usd_mesh_conventions_and_collision(tmp_path):
    out = str(tmp_path / "scene.usd")
    info = usd_writer.write_scene_usd(out, mesh=_quad_mesh(), with_collision=True)

    assert info.representation == "mesh"
    assert info.num_vertices == 4 and info.num_faces == 2 and info.has_collision

    stage = Usd.Stage.Open(out)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/World"

    prim = stage.GetPrimAtPath("/World/Environment")
    assert prim and prim.IsA(UsdGeom.Mesh)
    mesh = UsdGeom.Mesh(prim)
    assert len(mesh.GetPointsAttr().Get()) == 4
    assert len(mesh.GetFaceVertexIndicesAttr().Get()) == 6
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
    # vertex colors
    dc = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    assert dc and len(dc.Get()) == 4
    # static full-mesh collider
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)  # environment is fixed, not dynamic


def test_scene_usd_points_fallback(tmp_path):
    out = str(tmp_path / "scene.usd")
    pts = np.random.default_rng(0).normal(size=(50, 3))
    cols = np.random.default_rng(1).integers(0, 255, size=(50, 3), dtype=np.uint8)
    info = usd_writer.write_scene_usd(out, points=pts, point_colors=cols)

    assert info.representation == "points"
    assert info.num_points == 50 and not info.has_collision
    assert info.reconstruction == "none(points)"

    stage = Usd.Stage.Open(out)
    prim = stage.GetPrimAtPath("/World/Cloud")
    assert prim and prim.IsA(UsdGeom.Points)
    assert len(UsdGeom.Points(prim).GetPointsAttr().Get()) == 50


def test_trajectory_usd_camera_flip(tmp_path):
    out = str(tmp_path / "trajectory.usd")
    rng = np.random.default_rng(7)
    poses = []
    for i in range(5):
        Rc = Rotation.from_euler("xyz", rng.uniform(-1, 1, 3)).as_matrix()
        tc = rng.uniform(-3, 3, 3)
        P = np.eye(4)
        P[:3, :3] = Rc
        P[:3, 3] = tc
        poses.append(P)
    poses = np.stack(poses)

    n = usd_writer.write_trajectory_usd(out, poses, fps=10.0)
    assert n == 5

    stage = Usd.Stage.Open(out)
    assert stage.GetTimeCodesPerSecond() == 10.0
    assert stage.GetEndTimeCode() == 4
    cam = stage.GetPrimAtPath("/World/Camera")
    assert cam and cam.IsA(UsdGeom.Camera)
    op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]

    for i, P in enumerate(poses):
        gf = op.Get(Usd.TimeCode(float(i)))
        world_from_usdcam = np.array(gf).T  # undo the row-vector transpose
        Rc, tc = P[:3, :3], P[:3, 3]
        # camera world position is unchanged by the flip
        assert np.allclose(world_from_usdcam[:3, 3], tc, atol=1e-5)
        # USD camera looks down local -Z; that must equal the CV camera forward (+Z)
        usd_forward = world_from_usdcam[:3, :3] @ np.array([0, 0, -1.0])
        assert np.allclose(usd_forward, Rc @ np.array([0, 0, 1.0]), atol=1e-5)
        # USD camera up is local +Y; CV camera up is -Y
        usd_up = world_from_usdcam[:3, :3] @ np.array([0, 1.0, 0])
        assert np.allclose(usd_up, Rc @ np.array([0, -1.0, 0]), atol=1e-5)


def test_trajectory_intrinsics_focal_length(tmp_path):
    out = str(tmp_path / "trajectory.usd")
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    intr = np.tile([500.0, 500.0, 320.0, 240.0], (3, 1))
    usd_writer.write_trajectory_usd(out, poses, intrinsics=intr, fps=10.0, image_hw=(480, 640))

    stage = Usd.Stage.Open(out)
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    assert cam.GetFocalLengthAttr().Get() > 0
    assert cam.GetHorizontalApertureAttr().Get() > 0
