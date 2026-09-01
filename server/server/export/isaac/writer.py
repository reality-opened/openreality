"""Orchestrator: a finished SLAM scan -> ``export/<scan_id>/isaac/`` (Isaac Sim/Lab USD).

Pulls the world-frame cloud (``server/export/clouds``) and the ordered keyframe poses +
intrinsics (``server/export/trajectory``) from a live ``Solver``, estimates the metric +
gravity alignment (``align``), reconstructs a collidable mesh (``mesh`` — best-effort, falls
back to a points-only scene), and authors ``scene.usd`` + ``trajectory.usd`` + ``manifest.json``
(``usd_writer``). GPU-free / duck-typed like the rest of ``server/export/``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional

import numpy as np

from server.export.isaac import align as align_mod
from server.export.isaac import mesh as mesh_mod
from server.export.isaac import usd_writer
from server.export.isaac.schema import (
    AlignmentInfo,
    IsaacManifest,
)
from server.export.schema import DEFAULT_FPS


def _derive_scan_id(provided: Optional[str]) -> str:
    return str(provided) if provided else f"scan_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _poses_and_intrinsics(rows: list) -> tuple:
    """Reconstruct ``(poses_slam (N,4,4), cam_centers (N,3), intrinsics (N,4))`` from rows."""
    from scipy.spatial.transform import Rotation

    poses, centers, intr = [], [], []
    for r in rows:
        st = np.asarray(r["observation.state"], dtype=np.float64)
        t, quat = st[:3], st[3:7]
        P = np.eye(4, dtype=np.float64)
        P[:3, :3] = Rotation.from_quat(quat).as_matrix()  # xyzw, matches trajectory.py
        P[:3, 3] = t
        poses.append(P)
        centers.append(t)
        intr.append(np.asarray(r["observation.intrinsics"], dtype=np.float64))
    n = len(rows)
    return (
        np.stack(poses) if n else np.zeros((0, 4, 4)),
        np.stack(centers) if n else np.zeros((0, 3)),
        np.stack(intr) if n else np.zeros((0, 4)),
    )


def _find_ref_box(solver, ref_object, detections, report) -> Optional[tuple]:
    """Look up a reference object's SLAM-frame ``(center, extent)`` from the scene graph."""
    try:
        from server.export.writer import _resolve_facts

        facts = _resolve_facts(solver, detections or [], report)
    except Exception as exc:
        print(f"[isaac] WARN could not resolve scene facts for ref object: {exc}")
        return None
    for obj in getattr(facts, "objects", []) or []:
        q = getattr(obj, "query", None) or getattr(
            getattr(obj, "description", None), "category", None
        )
        if q and str(ref_object).lower() in str(q).lower():
            try:
                center = np.asarray(getattr(obj, "center"), dtype=np.float64).reshape(3)
                extent = np.asarray(getattr(obj, "extent"), dtype=np.float64).reshape(3)
                return center, extent
            except Exception:
                return None
    print(f"[isaac] WARN ref object {ref_object!r} not found in scene graph; "
          "scale will be unresolved (export stays up-to-scale).")
    return None


def _author_isaac_tree(
    isaac_dir: str,
    scan_id: str,
    positions: Any,
    colors: Any,
    poses_slam: np.ndarray,
    cam_centers: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    num_keyframes: int,
    image_hw: Optional[tuple],
    *,
    fps: float,
    scale: Optional[float],
    ref_box: Optional[tuple] = None,
    ref_meters: Optional[float] = None,
    ref_query: Optional[str] = None,
    up_override: Optional[Any] = None,
    recenter_xy: bool = True,
    reconstruct: bool = True,
    reconstruction_method: str = "poisson",
    poisson_depth: int = 9,
    decimate_target: Optional[int] = None,
    include_points: bool = False,
    scale_source: Optional[str] = None,
    metric_note: Optional[str] = None,
    require_metric: bool = False,
) -> str:
    """Shared geometry -> USD core for both the live-``Solver`` and persisted-record entry points.

    Takes geometry already gathered (world-frame cloud + keyframe poses/intrinsics) and authors
    ``scene.usd`` + ``trajectory.usd`` + ``manifest.json`` under ``isaac_dir``. ``align.py`` runs
    on pure numpy; ``mesh``/``usd_writer`` lazy-import ``open3d``/``pxr`` (a missing ``pxr``
    surfaces as an ``ImportError`` here — the record path maps that to a 501). ``scale_source`` /
    ``metric_note`` let a caller stamp the manifest with a provenance the raw ``resolve_scale``
    can't know (e.g. the scale came pre-applied from the Refine Metric-anchor calibration).
    ``require_metric`` refuses to write an up-to-scale scene (defense-in-depth for the gated route).
    """
    align_res = align_mod.compute_alignment(
        positions,
        cam_centers if cam_centers is not None and len(cam_centers) else None,
        scale=scale,
        ref_box=ref_box,
        ref_meters=ref_meters,
        ref_query=ref_query,
        up_override=up_override,
        recenter_xy=recenter_xy,
    )
    if require_metric and not align_res.metric:
        raise ValueError(
            "refusing to author an up-to-scale Isaac scene: no metric scale resolved "
            "(pass a scale factor or a Metric-anchor-calibrated source)"
        )

    positions_isaac = align_mod.apply_transform(align_res.transform, positions)

    # Reconstruct a collidable mesh; on any failure fall back to a points-only scene.
    mesh_tuple = None
    reconstruction = "none(points)"
    if reconstruct:
        try:
            mesh_tuple = mesh_mod.reconstruct_mesh(
                positions_isaac,
                colors,
                method=reconstruction_method,
                poisson_depth=poisson_depth,
                decimate_target=decimate_target,
            )
            reconstruction = reconstruction_method
            if mesh_tuple is None or len(mesh_tuple[0]) == 0:
                print("[isaac] WARN reconstruction produced no mesh; using points-only scene.")
                mesh_tuple = None
                reconstruction = "none(points)"
        except Exception as exc:
            # Best-effort: any reconstruction failure (incl. open3d absent) degrades to a
            # points-only scene. The gated broker route pre-checks open3d/pxr and 501s
            # ("isaac_unavailable") before ever reaching here, so a route export that gets this
            # far always has both deps; this fallback is for the offline CLI path.
            print(f"[isaac] WARN mesh reconstruction failed ({exc}); using points-only scene.")
            mesh_tuple = None
            reconstruction = "none(points)"

    want_points = include_points or mesh_tuple is None
    geom_info = usd_writer.write_scene_usd(
        os.path.join(isaac_dir, "scene.usd"),
        mesh=mesh_tuple,
        points=positions_isaac if want_points else None,
        point_colors=colors if want_points else None,
        with_collision=True,
        reconstruction=reconstruction,
    )

    poses_isaac = (
        np.stack([align_mod.transform_camera_pose(align_res, P) for P in poses_slam])
        if len(poses_slam)
        else np.zeros((0, 4, 4))
    )
    usd_writer.write_trajectory_usd(
        os.path.join(isaac_dir, "trajectory.usd"),
        poses_isaac,
        intrinsics=intrinsics if intrinsics is not None and len(intrinsics) else None,
        fps=fps,
        image_hw=image_hw,
    )

    if align_res.metric:
        notes = metric_note or ""
    else:
        notes = (
            "Scale unresolved (no scale factor / ref object): geometry is still up-to-scale. "
            "Provide a scale to make the stage metric."
        )
    manifest = IsaacManifest(
        scan_id=scan_id,
        fps=float(fps),
        num_keyframes=int(num_keyframes),
        alignment=AlignmentInfo(
            transform=align_res.transform.tolist(),
            scale=align_res.scale,
            scale_source=scale_source or align_res.scale_source,
            metric=align_res.metric,
            gravity_aligned=align_res.gravity_aligned,
            up_vector_slam=align_res.up_vector_slam.tolist(),
            floor_inlier_fraction=align_res.floor_inlier_fraction,
            gravity_source=align_res.gravity_source,
            recentered_xy=align_res.recentered_xy,
        ),
        geometry=geom_info,
        notes=notes,
    )
    with open(os.path.join(isaac_dir, "manifest.json"), "w") as fh:
        json.dump(manifest.model_dump(), fh, indent=2)

    print(
        f"[isaac] wrote {isaac_dir} "
        f"(metric={align_res.metric}, scale={align_res.scale:.4g} "
        f"[{scale_source or align_res.scale_source}], "
        f"gravity_aligned={align_res.gravity_aligned}, "
        f"geometry={geom_info.representation}, collision={geom_info.has_collision})"
    )
    return isaac_dir


def export_isaac(
    solver: Any,
    out_dir: str,
    *,
    scan_id: Optional[str] = None,
    fps: float = DEFAULT_FPS,
    scale: Optional[float] = None,
    ref_object: Optional[str] = None,
    ref_meters: Optional[float] = None,
    up_override: Optional[Any] = None,
    detections: Optional[list] = None,
    report: Any = None,
    reconstruct: bool = True,
    reconstruction_method: str = "poisson",
    poisson_depth: int = 9,
    decimate_target: Optional[int] = None,
    include_points: bool = False,
    recenter_xy: bool = True,
) -> Optional[str]:
    """Write ``<out_dir>/<scan_id>/isaac/{scene.usd, trajectory.usd, manifest.json}`` from a live
    ``Solver`` (offline ``main.py`` / Modal GPU path).

    Returns the ``isaac/`` directory, or ``None`` if the scan had no geometry. Metric scale
    comes from ``scale`` or a ``ref_object``+``ref_meters`` back-out; without either the export
    is honestly flagged ``metric=False`` (still up-to-scale).
    """
    from server.export import clouds as clouds_mod
    from server.export import trajectory as trajectory_mod

    scan_id = _derive_scan_id(scan_id)
    isaac_dir = os.path.join(out_dir, scan_id, "isaac")
    os.makedirs(isaac_dir, exist_ok=True)

    positions, colors = clouds_mod.gather_full_cloud(solver)
    if positions is None or len(positions) == 0:
        print("[isaac] WARN no world-frame cloud; skipping Isaac export.")
        return None

    rows, index_map = trajectory_mod.extract_trajectory(solver, fps=fps)
    poses_slam, cam_centers, intrinsics = _poses_and_intrinsics(rows)

    ref_box = _find_ref_box(solver, ref_object, detections, report) if ref_object else None

    image_hw = None
    try:
        from server.export.writer import _first_frame_hw

        image_hw = _first_frame_hw(solver, index_map)
    except Exception:
        image_hw = None

    return _author_isaac_tree(
        isaac_dir,
        scan_id,
        positions,
        colors,
        poses_slam,
        cam_centers,
        intrinsics,
        len(rows),
        image_hw,
        fps=fps,
        scale=scale,
        ref_box=ref_box,
        ref_meters=ref_meters,
        ref_query=ref_object,
        up_override=up_override,
        recenter_xy=recenter_xy,
        reconstruct=reconstruct,
        reconstruction_method=reconstruction_method,
        poisson_depth=poisson_depth,
        decimate_target=decimate_target,
        include_points=include_points,
    )


def export_isaac_from_record(
    record: dict,
    out_dir: str,
    *,
    scan_id: Optional[str] = None,
    fps: float = DEFAULT_FPS,
    scale: Optional[float] = None,
    scale_source: Optional[str] = None,
    anchor_scale: Optional[float] = None,
    require_metric: bool = True,
    reconstruct: bool = True,
    reconstruction_method: str = "poisson",
    poisson_depth: int = 9,
    decimate_target: Optional[int] = None,
    include_points: bool = False,
    recenter_xy: bool = True,
) -> Optional[str]:
    """Write the Isaac USD tree from a **persisted Stage-4 record** — no live ``Solver`` / GPU.

    Consumes the SAME geometry the GR00T/LeRobot export reads: the world-frame dense cloud
    (``record["cloud"]`` -> ``(positions, colors)``) and ordered keyframe poses + intrinsics
    (``record["trajectory"]`` -> ``{poses (N,4,4) cam->world, intrinsics (N,4)}``), exactly as
    assembled by ``server.export.record.load_record_from_store``. The metric scale is supplied by
    the caller (the broker route gates on it): ``scale`` is the SLAM-units->metres factor to apply
    (``1.0`` when ``record`` already carries anchor-calibrated, already-metric geometry). Returns
    the ``isaac/`` directory, or ``None`` if the record has no cloud.

    ``anchor_scale`` is stamped into the manifest note for provenance when the scale came from a
    Metric-anchor calibration (``scale_source="anchor:prescaled"``). GPU-free; ``pxr``/``open3d``
    stay lazy in ``usd_writer``/``mesh``.
    """
    scan_id = _derive_scan_id(scan_id)
    cloud = record.get("cloud")
    if not cloud or cloud[0] is None or len(cloud[0]) == 0:
        print("[isaac] WARN record has no world-frame cloud; skipping Isaac export.")
        return None
    positions, colors = cloud[0], cloud[1]

    traj = record.get("trajectory") or {}
    poses_slam = np.asarray(traj.get("poses"), dtype=np.float64).reshape(-1, 4, 4) if traj.get(
        "poses"
    ) is not None else np.zeros((0, 4, 4))
    cam_centers = poses_slam[:, :3, 3] if len(poses_slam) else None
    intrinsics = (
        np.asarray(traj.get("intrinsics"), dtype=np.float64).reshape(-1, 4)
        if traj.get("intrinsics") is not None
        else None
    )

    metric_note = None
    if scale_source == "anchor:prescaled":
        if anchor_scale:
            metric_note = (
                f"Metric scale from the Refine Metric-anchor calibration "
                f"({float(anchor_scale):.4g} m per SLAM unit), pre-applied to the exported geometry."
            )
        else:
            metric_note = (
                "Metric scale from the Refine Metric-anchor calibration, pre-applied to the "
                "exported geometry."
            )

    isaac_dir = os.path.join(out_dir, scan_id, "isaac")
    os.makedirs(isaac_dir, exist_ok=True)
    return _author_isaac_tree(
        isaac_dir,
        scan_id,
        positions,
        colors,
        poses_slam,
        cam_centers,
        intrinsics,
        int(len(poses_slam)),
        None,  # no source video frames on the record path -> best-effort camera aperture is skipped
        fps=fps,
        scale=scale,
        scale_source=scale_source,
        metric_note=metric_note,
        require_metric=require_metric,
        recenter_xy=recenter_xy,
        reconstruct=reconstruct,
        reconstruction_method=reconstruction_method,
        poisson_depth=poisson_depth,
        decimate_target=decimate_target,
        include_points=include_points,
    )
