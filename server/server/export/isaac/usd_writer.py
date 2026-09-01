"""Author the Isaac-ready USD stages (``scene.usd`` + ``trajectory.usd``).

Uses OpenUSD's ``pxr`` (the ``usd-core`` pip wheel) — **authoring only**, no Isaac Sim runtime
needed. ``pxr`` is imported lazily so importing this module is cheap and the rest of the
export package stays usable without USD installed.

Two conventions are non-negotiable for Isaac Sim / Isaac Lab and are the whole reason a custom
writer exists (USD's *unauthored* defaults are wrong for robotics):

- **Stage:** ``upAxis="Z"`` and ``metersPerUnit=1.0`` are authored explicitly (defaults would
  be Y-up / centimeters).
- **Camera:** computer-vision cameras look down **+Z** with **+Y down**; a USD ``Camera`` looks
  down **-Z** with **+Y up**. ``trajectory.usd`` applies the ``diag(1,-1,-1,1)`` flip so the
  animated camera actually points where the scan camera pointed. This is the easiest thing to
  get silently wrong, so the unit test reads the camera world position back and checks it.

USD also uses **row-vector** math (``p' = p * M``), so a standard numpy column-translation 4x4
must be transposed before it goes into a ``Gf.Matrix4d`` (see :func:`_gf_matrix`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from server.export.isaac.schema import GeometryInfo

# computer-vision cam (x right, y down, z forward) -> USD cam (x right, y up, z back)
_CV_TO_USD_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _import_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "usd-core (pxr) is required to author Isaac USD scenes (`pip install usd-core`)."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _gf_matrix(mat: np.ndarray):
    """numpy 4x4 (column-translation) -> ``Gf.Matrix4d`` (USD row-vector convention)."""
    Gf = _import_pxr()[0]
    m = np.asarray(mat, dtype=np.float64).reshape(4, 4).T  # transpose: USD is row-vector
    return Gf.Matrix4d(*m.flatten().tolist())


def _vec3f_array(Vt, Gf, arr: np.ndarray):
    """``(N,3)`` numpy -> ``Vt.Vec3fArray`` (numpy scalars don't bind to ``Gf.Vec3f``)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1, 3))
    try:
        return Vt.Vec3fArray.FromNumpy(a)
    except Exception:  # older pxr without FromNumpy
        return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in a])


def _define_isaac_stage(out_path: str):
    """Create a new stage with the Isaac robotics convention (Z-up, meters) + a /World prim."""
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt = _import_pxr()
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage, (Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt)


# ---------------------------------------------------------------------------
# scene.usd  (environment geometry + collision)
# ---------------------------------------------------------------------------

def write_scene_usd(
    out_path: str,
    *,
    mesh: Optional[tuple] = None,  # (verts (V,3), faces (F,3), vcolors (V,3) u8|None)
    points: Optional[np.ndarray] = None,  # (P,3)
    point_colors: Optional[np.ndarray] = None,  # (P,3) u8
    with_collision: bool = True,
    reconstruction: str = "poisson",
) -> GeometryInfo:
    """Write the environment ``scene.usd``; return a :class:`GeometryInfo` describing it.

    A ``mesh`` becomes ``/World/Environment`` (``UsdGeom.Mesh``) with vertex colors and, when
    ``with_collision``, a static full-triangle-mesh collider (``UsdPhysics.CollisionAPI`` +
    ``MeshCollisionAPI`` approximation="none"; no ``RigidBodyAPI`` -> the environment is fixed).
    A ``points`` cloud (optional, or the fallback when there is no mesh) becomes
    ``/World/Cloud`` (``UsdGeom.Points``) as visual reference only.
    """
    stage, (Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt) = _define_isaac_stage(out_path)

    info = GeometryInfo(reconstruction=reconstruction)
    has_mesh = mesh is not None and len(mesh[0]) > 0 and len(mesh[1]) > 0

    if has_mesh:
        verts, faces, vcolors = mesh
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        m = UsdGeom.Mesh.Define(stage, "/World/Environment")
        m.CreatePointsAttr(_vec3f_array(Vt, Gf, verts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.reshape(-1)]))
        if vcolors is not None and len(vcolors) == len(verts):
            primvar = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )
            primvar.Set(_vec3f_array(Vt, Gf, np.asarray(vcolors, dtype=np.float32) / 255.0))
        if with_collision:
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.none)  # full triangle mesh
            info.has_collision = True
        info.representation = "mesh"
        info.num_vertices = int(len(verts))
        info.num_faces = int(len(faces))

    if points is not None and len(points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pgeom = UsdGeom.Points.Define(stage, "/World/Cloud")
        pgeom.CreatePointsAttr(_vec3f_array(Vt, Gf, pts))
        pgeom.CreateWidthsAttr(Vt.FloatArray([0.01] * len(pts)))
        if point_colors is not None and len(point_colors) == len(pts):
            primvar = UsdGeom.PrimvarsAPI(pgeom).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )
            primvar.Set(_vec3f_array(Vt, Gf, np.asarray(point_colors, dtype=np.float32) / 255.0))
        info.num_points = int(len(pts))
        info.representation = "mesh+points" if has_mesh else "points"
        if not has_mesh:
            info.reconstruction = "none(points)"

    stage.GetRootLayer().Save()
    return info


# ---------------------------------------------------------------------------
# trajectory.usd  (animated camera following the scan)
# ---------------------------------------------------------------------------

def write_trajectory_usd(
    out_path: str,
    poses_isaac: np.ndarray,  # (N,4,4) cam->world, already in the Isaac frame
    *,
    intrinsics: Optional[np.ndarray] = None,  # (N,4) [fx,fy,cx,cy]
    fps: float = 10.0,
    image_hw: Optional[tuple] = None,  # (H, W)
) -> int:
    """Write ``trajectory.usd`` — a ``UsdGeomCamera`` animated over the keyframe poses.

    Each pose is right-multiplied by the CV->USD camera flip so the USD camera looks where the
    scan camera looked. Returns the number of time samples written.
    """
    poses = np.asarray(poses_isaac, dtype=np.float64).reshape(-1, 4, 4)
    n = poses.shape[0]
    stage, (Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt) = _define_isaac_stage(out_path)

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(n - 1, 0))
    stage.SetTimeCodesPerSecond(float(fps))

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1.0e6))

    # Intrinsics -> focal length / aperture (first keyframe; aperture is an arbitrary gauge,
    # focal length scales with it to preserve fx/width). Best-effort, only if we know H,W.
    if intrinsics is not None and len(intrinsics) and image_hw and all(image_hw):
        fx, fy, cx, cy = np.asarray(intrinsics[0], dtype=np.float64)
        H, W = image_hw
        h_aperture = 20.955  # mm, a conventional gauge
        v_aperture = h_aperture * (float(H) / float(W))
        cam.CreateHorizontalApertureAttr(float(h_aperture))
        cam.CreateVerticalApertureAttr(float(v_aperture))
        cam.CreateFocalLengthAttr(float(fx / float(W) * h_aperture))
        # principal-point offset (mm)
        cam.CreateHorizontalApertureOffsetAttr(float((cx - W / 2.0) / W * h_aperture))
        cam.CreateVerticalApertureOffsetAttr(float((cy - H / 2.0) / H * v_aperture))

    op = UsdGeom.Xformable(cam).AddTransformOp()
    for i in range(n):
        world_from_usdcam = poses[i] @ _CV_TO_USD_CAMERA
        op.Set(_gf_matrix(world_from_usdcam), Usd.TimeCode(float(i)))

    stage.GetRootLayer().Save()
    return n
