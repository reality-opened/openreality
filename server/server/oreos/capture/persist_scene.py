"""Bridge: a finished LiDAR-grounded capture-session recon -> a persisted OS scene.

Unlike the robot-recording lane (``server/oreos/recordings/persist_scene.py``),
whose poses/points are up-to-scale until a later Refine step, a capture-session
recon is GROUNDED AT SOLVE TIME by real iPhone LiDAR depth (EXP-45's Sim(3) gauge
sidecar over VGGT-SLAM submaps — core's ``vggt_slam.metric_layer`` /
``lidar_anchor``): the persisted trajectory and cloud are already in real metres,
in a gravity-aligned world (ARKit's world +Y is up; the gauge layer copies that
basis, only rescaling it — see ``metric_layer.build_from_capture``). Persisted with
``source="recon_lidar"``:

  - report/facts are DEGRADED + geometry-only (this job runs no live agent, no
    object detections) — same doctrine as every other ingest source
    (``splat_import.build_scene_payload`` / ``recordings.persist_scene`` are the
    direct analogs);
  - ``facts.metrics.vertical_axis_known=True`` / ``up_axis=[0,1,0]`` — a claim
    ordinary VGGT-SLAM output cannot make (its world frame is NOT gravity-aligned)
    but this lane can, because the anchor's world basis IS ARKit's gravity frame;
  - the metric-layer sidecars (``metric_trajectory_tum.txt``, ``metric_gauges.json``,
    ``anchor_telemetry.json`` — a kill-experiment-shaped per-submap/junction
    summary, PLAN §4's report shape, WITHOUT re-running the expensive diagnostic
    full-Sim(3) fit that experiment reserves for offline analysis) land as
    ``derived/demo/capture/<stamp>/*`` artifacts (the historical ``demo`` key
    namespace token is a persisted contract — see platform/CLAUDE.md).

No flask, no modal, no core/gtsam import at module scope: every function here takes
already-computed numpy arrays and plain dicts, so this module is importable and
unit-testable without a GPU, without ``core``, and without the SLAM stack. The GPU
job (``server/oreos/capture/modal_recon.py``) does the SLAM + metric-layer math and
calls straight into ``persist_capture_scene`` with the results.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

import numpy as np


def build_capture_scene_payload(
    trajectory_rows: Any,
    points: Optional[Any],
    colors: Optional[Any],
    metric_report: dict[str, Any],
    *,
    session_label: str,
    frames_total: int,
    frames_kept: int,
    submap_count: int,
    loop_closures: int,
) -> dict[str, Any]:
    """Everything ``save_scene`` needs, built from already-composed metric outputs.

    ``trajectory_rows``: array-like ``(N, 8)`` = ``[timestamp, x, y, z, qx, qy, qz,
    qw]`` — the exact shape ``vggt_slam.metric_layer.compose_metric_trajectory``
    returns; already metric + gravity-aligned. ``points``/``colors``: ``(M, 3)``
    float / ``(M, 3)`` uint8 world-frame cloud, or both ``None`` if the recon
    produced no usable points (facts then report ``point_count=0`` rather than
    silently omitting the scene). ``metric_report``: the report dict
    ``vggt_slam.metric_layer.build_from_capture`` returns alongside the optimized
    gauge layer (``gauges``, ``scale_priors``, ``junction_rotations``,
    ``vision_consistency_edges``, ``final_error``, ...).

    Raises ``ValueError`` if ``trajectory_rows`` has fewer than 2 rows — a scene
    without a real trajectory is not persistable (mirrors
    ``recordings.build_recording_scene_payload``'s same refusal)."""
    from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport

    rows = np.asarray(trajectory_rows, dtype=np.float64).reshape(-1, 8)
    if rows.shape[0] < 2:
        raise ValueError(f"metric trajectory has {rows.shape[0]} pose(s); need >= 2")

    from server.oreos.recordings.export_targets import _pose_matrices

    poses = _pose_matrices(rows[:, 1:4], rows[:, 4:8]).astype(np.float32)
    t0 = float(rows[0, 0])
    trajectory = {
        "poses": poses,
        # Real per-frame intrinsics exist on the CaptureSession (intrinsics.csv /
        # camera_matrix.csv) but core's compose_metric_trajectory does not thread
        # per-row intrinsics through its return value, and reconstructing this
        # array's row order independently (it is time-sorted across ALL submaps)
        # would duplicate core's own sort — left as zeros, declared, same
        # convention robot recordings use for the identical reason.
        "intrinsics": np.zeros((rows.shape[0], 4), dtype=np.float32),
        "source_frame_id": (rows[:, 0] - t0).astype(np.float32),
    }

    has_points = points is not None and colors is not None and len(points) > 0
    if has_points:
        positions = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors_arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        extents = (bbox_max - bbox_min).astype(np.float64)
        metrics = SceneMetrics(
            dimensions=sorted((float(v) for v in extents), reverse=True),
            bbox_min=[float(v) for v in bbox_min],
            bbox_max=[float(v) for v in bbox_max],
            num_submaps=int(submap_count),
            num_keyframes=int(rows.shape[0]),
            point_count=int(positions.shape[0]),
            # ARKit's world frame is gravity-aligned with +Y up, and the metric
            # gauge layer copies that basis verbatim (only rescaling it) — so
            # unlike ordinary VGGT-SLAM output, this world's vertical axis is a
            # known fact, not a plane-fit guess.
            vertical_axis_known=True,
            up_axis=[0.0, 1.0, 0.0],
            derivation="lidar_gravity_anchor",
        )
        scene_center = [float(v) for v in positions.mean(axis=0)]
    else:
        colors_arr = None
        metrics = SceneMetrics(
            point_count=0,
            num_submaps=int(submap_count),
            num_keyframes=int(rows.shape[0]),
            vertical_axis_known=True,
            up_axis=[0.0, 1.0, 0.0],
            derivation="lidar_gravity_anchor",
        )
        scene_center = [0.0, 0.0, 0.0]

    n_scale = int(metric_report.get("scale_priors", 0) or 0)
    n_gauges = int(metric_report.get("gauges", 0) or 0)
    n_vis = int(metric_report.get("vision_consistency_edges", 0) or 0)
    final_error = metric_report.get("final_error")

    scale_note = f"Metric scale from {n_scale}/{n_gauges} LiDAR-anchored gauge(s)"
    if n_vis:
        scale_note += f", {n_vis} vision-consistency junction(s)"
    if isinstance(final_error, (int, float)):
        scale_note += f" (gauge-layer error {final_error:.4g})"
    scale_note += "."

    facts = SceneFacts(
        metrics=metrics,
        scene_center=scene_center,
        units_note=(
            "Metric, gravity-aligned world grounded by the phone's LiDAR depth "
            "(Sim(3) gauge layer over VGGT-SLAM submaps) — coordinates are in real "
            "metres, not just up-to-scale."
        ),
    )
    report = SceneReport(
        summary=(
            f"LiDAR-grounded capture '{session_label}' reconstructed with VGGT-SLAM "
            f"({frames_kept}/{frames_total} keyframes, {submap_count} submap(s), "
            f"{loop_closures} loop closure(s)). {scale_note} The cloud shown is the "
            "metric-composed submap union; no object detections exist yet — "
            "annotation runs on the geometry and real capture keyframes."
        ),
        room_type="unknown",
        coverage_note="Handheld iPhone LiDAR capture — coverage follows the walked path.",
        facts=facts,
        degraded=True,
    )

    return {
        "report": report,
        "facts": facts,
        "points": (positions, colors_arr) if has_points else None,
        "trajectory": trajectory,
    }


def persist_capture_scene(
    persistence: Any,
    user_id: str,
    scan_id: str,
    *,
    trajectory_rows: Any,
    points: Optional[Any],
    colors: Optional[Any],
    metric_report: dict[str, Any],
    gauges: dict[str, Any],
    anchor_telemetry: dict[str, Any],
    session_label: str,
    frames_total: int,
    frames_kept: int,
    submap_count: int,
    loop_closures: int,
    label: Optional[str] = None,
) -> dict[str, Any]:
    """Persist the scene (``source="recon_lidar"``) + the metric-layer sidecars.

    Sidecars land under ``derived/demo/capture/<stamp>/``:

      - ``metric_trajectory_tum.txt``  the composed metric trajectory, TUM format
        (regenerated from ``trajectory_rows`` — byte-identical to what
        ``compose_metric_trajectory(..., out_tum=...)`` would have written)
      - ``metric_gauges.json``         ``{"report": metric_report, "gauges": gauges}``
        — the optimized per-submap Sim(3) gauges + ``build_from_capture``'s report
      - ``anchor_telemetry.json``      ``anchor_telemetry`` verbatim (kill-experiment
        -shaped per-submap LiDAR-ratio + junction telemetry)

    Best-effort on the sidecars (a write failure never aborts the scene save — same
    doctrine as ``recordings.persist_recording_scene``). Returns
    ``{"scan_id", "derived": {name: derived_key}}``."""
    payload = build_capture_scene_payload(
        trajectory_rows, points, colors, metric_report,
        session_label=session_label, frames_total=frames_total, frames_kept=frames_kept,
        submap_count=submap_count, loop_closures=loop_closures,
    )
    persistence.save_scene(
        user_id,
        scan_id,
        payload["report"],
        payload["facts"],
        points=payload["points"],
        trajectory=payload["trajectory"],
        label=label or session_label,
        source="recon_lidar",
    )

    rows = np.asarray(trajectory_rows, dtype=np.float64).reshape(-1, 8)
    tum_lines = [" ".join(f"{v:.9f}" for v in row) for row in rows]
    artifacts: dict[str, bytes] = {
        "metric_trajectory_tum.txt": ("\n".join(tum_lines) + "\n").encode("utf-8"),
        "metric_gauges.json": json.dumps(
            {"report": metric_report, "gauges": gauges}, indent=2, default=str
        ).encode("utf-8"),
        "anchor_telemetry.json": json.dumps(
            anchor_telemetry, indent=2, default=str
        ).encode("utf-8"),
    }

    stamp = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    derived: dict[str, str] = {}
    for name, data in artifacts.items():
        try:
            key = persistence.save_derived_artifact(
                user_id, scan_id, f"demo/capture/{stamp}/{name}", data
            )
            derived[name] = key
        except Exception:
            continue

    return {"scan_id": scan_id, "derived": derived}
