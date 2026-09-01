"""Point cloud -> watertight-ish surface mesh for the Isaac scene collider.

Isaac Sim/Lab needs a *surface* for a robot to stand/collide on; the SLAM dense cloud is just
points. We reconstruct a colored triangle mesh with open3d (Poisson by default, with a
ball-pivoting fallback) and trim the low-density Poisson "balloon" outside the real support.

``open3d`` is imported lazily (it is a heavy, optional dep — already in ``requirements.txt``).
Reconstruction quality is the main risk of the mesh path; callers (``writer.export_isaac``)
treat a failure here as best-effort and fall back to a points-only scene.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _import_open3d():
    try:
        import open3d as o3d  # noqa: WPS433 (lazy heavy import)
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "open3d is required for Isaac mesh reconstruction "
            "(`pip install open3d`); the writer falls back to a points-only scene."
        ) from exc
    return o3d


def reconstruct_mesh(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    method: str = "poisson",
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    normal_k: int = 30,
    decimate_target: Optional[int] = None,
) -> tuple:
    """Reconstruct a colored surface mesh -> ``(verts (V,3) f64, faces (F,3) i64, vcolors (V,3) u8|None)``.

    ``method`` is ``"poisson"`` (default) or ``"ball_pivoting"``. Normals are estimated +
    consistently oriented first (Poisson needs oriented normals). Low-density Poisson
    vertices (below ``density_quantile``) are dropped — those are the extrapolated balloon
    outside the scanned support.
    """
    o3d = _import_open3d()

    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        raise ValueError(f"need >=4 points to reconstruct a mesh, got {pts.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None and len(colors):
        col = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if col.max() > 1.0 + 1e-6:
            col = col / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(col, 0.0, 1.0))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    )
    pcd.orient_normals_consistent_tangent_plane(normal_k)

    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        densities = np.asarray(densities)
        if densities.size:
            thr = np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(densities < thr)
    elif method == "ball_pivoting":
        dists = pcd.compute_nearest_neighbor_distance()
        avg = float(np.mean(dists)) if len(dists) else 1.0
        radii = o3d.utility.DoubleVector([avg * r for r in (1.0, 2.0, 4.0)])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    else:
        raise ValueError(f"unknown reconstruction method: {method!r}")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if decimate_target:
        mesh = mesh.simplify_quadric_decimation(int(decimate_target))
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    vcolors = None
    if mesh.has_vertex_colors():
        vcolors = np.clip(
            np.asarray(mesh.vertex_colors) * 255.0, 0, 255
        ).astype(np.uint8)
    return verts, faces, vcolors
