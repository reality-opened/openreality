"""Metric-anchor application — the ONE implementation of "make this scan metric".

Lifted verbatim out of ``server/app.py`` (the ``POST /api/scenes/<id>/anchor`` route body)
so it can run in two places that must never disagree:

  * the Flask route, unchanged — ``app.py`` imports these names and its handler is untouched;
  * a plain Modal container with no Flask, no auth and no browser
    (``modal_tum_depth_anchor.py``). A Clerk token lives 60 seconds, so anything that must
    be verified server-side has to reach the scene Dict + Volume directly — the same reason
    ``scripts/backfill_imported_scene.py`` and ``modal_oreos_render.py`` exist.

Nothing here imports flask, and the only server-side dependencies are the two sibling I/O
modules and a persistence object (duck-typed: ``get_cloud``, ``get_trajectory``,
``get_splat_path``, ``save_derived_artifact``, optionally ``set_derived_pointer``).

Doctrine unchanged (WORLD-TRANSFORM-CONTRACT "Units & scale"): NON-DESTRUCTIVE — every
output is a new ``derived/anchor/<stamp>/...`` artifact and ``cloud.npz`` / ``splat.ply`` /
``trajectory.npz`` are never touched, so live pilot embeds keep shipping from the originals.
"""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from server.scene_report.cloud_io import build_ply_bytes
from server.scene_report.splat_io import (
    read_splat_ply,
    scale_splat_fields,
    serialize_splat_ply,
)

# Reject a degenerate (near-coincident) point pair — a port of core's
# ``vggt_slam.metric_absolute.MIN_DISTANCE_UNITS``, mirrored in ``server.oreos.measure``.
MIN_ANCHOR_DISTANCE_UNITS = 1e-6


def validate_anchor_point(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{name}' must be [x, y, z], got {value!r}")
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must contain finite numbers, got {value!r}")
    return arr


def scale_factor_from_known_distance(distance_m: float, measured: float) -> float:
    """metres per SLAM-unit = distance_m / measured. Port of
    ``vggt_slam.metric_absolute.scale_from_known_distance`` (core ``feat/metric-absolute``,
    read as a spec)."""
    if not np.isfinite(measured) or measured < MIN_ANCHOR_DISTANCE_UNITS:
        raise ValueError(
            f"measured point-pair distance {measured:.3g} is degenerate "
            f"(< {MIN_ANCHOR_DISTANCE_UNITS:.0e}); pick two points that are clearly apart"
        )
    if not (np.isfinite(distance_m) and distance_m > 0):
        raise ValueError(f"'distance_m' must be a positive finite number, got {distance_m!r}")
    factor = float(distance_m) / float(measured)
    if not (np.isfinite(factor) and factor > 0):
        raise ValueError("non-finite/non-positive scale factor")
    return factor


def cloud_extent(positions: np.ndarray) -> float:
    """Bounding-box diagonal — a single stable 'how big does this scene currently read'
    number for the before/after signal. 0.0 for an empty/degenerate cloud."""
    if positions.shape[0] == 0:
        return 0.0
    finite = positions[np.isfinite(positions).all(axis=1)]
    if finite.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))


def derived_pointer(
    kind: str,
    *,
    cloud_key: Optional[str] = None,
    trajectory_key: Optional[str] = None,
    splat_key: Optional[str] = None,
    applied_at: Optional[str] = None,
    scale_factor: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Build the scene record's "latest calibrated" pointer for a Refine action, or ``None`` if
    the action produced no derived artifact. ``source_key`` is the representative key an export's
    ``?source=`` selector uses (the cloud when present, else the splat) — export reads the whole
    derived group from it, so the single key stands in for the trio. ``scale_factor``
    (metres-per-SLAM-unit) is carried on an ``anchor`` pointer so the Isaac USD export can gate on
    a metric scale and record its provenance (the anchor's calibrated geometry is already metric,
    so ``scale_factor`` is provenance, not a factor the export re-applies)."""
    source_key = cloud_key or splat_key or trajectory_key
    if not source_key:
        return None
    return {
        "kind": kind,
        "cloud_key": cloud_key,
        "trajectory_key": trajectory_key,
        "splat_key": splat_key,
        "source_key": source_key,
        "applied_at": applied_at or datetime.now(timezone.utc).isoformat(),
        "scale_factor": scale_factor,
    }


def persist_derived_pointer(
    persistence: Any, user_id: str, scan_id: str, pointer: Optional[dict[str, Any]]
) -> None:
    """Best-effort record of the latest calibrated pointer (metadata-only, non-destructive). A
    persistence backend without ``set_derived_pointer`` (or a KV hiccup) is swallowed — the
    derived artifact itself is already written and the explicit ``?source=`` selector still works,
    so a missing pointer only means default (no-selector) export won't auto-pick it up."""
    if pointer is None:
        return
    setter = getattr(persistence, "set_derived_pointer", None)
    if not callable(setter):
        return
    try:
        setter(user_id, scan_id, pointer)
    except Exception as exc:
        print(f"[derived] latest-calibrated pointer persist failed for {scan_id}: {exc}")


def compute_anchor_scale(point_a: Any, point_b: Any, distance_m: float) -> tuple[float, float]:
    """The instant half of an anchor: ``(measured, scale_factor)`` from the two picked
    points and the known real distance. Pure arithmetic — raises ``ValueError`` exactly
    like the full apply on bad input, so a route can validate BEFORE spawning any work."""
    pa = validate_anchor_point(point_a, "point_a")
    pb = validate_anchor_point(point_b, "point_b")
    measured = float(np.linalg.norm(pa - pb))
    scale_factor = scale_factor_from_known_distance(float(distance_m), measured)
    return measured, scale_factor


def pending_anchor_pointer(
    scale_factor: float, applied_at: str, job_id: Optional[str] = None
) -> dict[str, Any]:
    """A ``derived_latest`` anchor pointer whose artifacts are still materializing.

    Deliberately bypasses ``derived_pointer``'s no-source-key rule: metadata (kind +
    ``scale_factor`` + ``applied_at``) is what flips the UI to metres instantly; every
    key stays ``None`` until the background job upgrades the pointer. All consumers are
    None-safe by prior contract: the Isaac export gate requires a truthy ``source_key``
    (stays gated), the viewer's direct-load falls back to original-gauge geometry
    (consistent with original-gauge overlays), and the client's capabilities only need
    ``kind == 'anchor'`` + ``scale_factor``."""
    return {
        "kind": "anchor",
        "cloud_key": None,
        "trajectory_key": None,
        "splat_key": None,
        "source_key": None,
        "applied_at": applied_at,
        "scale_factor": scale_factor,
        "pending": True,
        **({"materializing_job_id": job_id} if job_id else {}),
    }


def materialize_anchor_artifacts(
    persistence: Any,
    user_id: str,
    scan_id: str,
    scale_factor: float,
    *,
    stamp: Optional[str] = None,
) -> dict[str, Any]:
    """The heavy half of an anchor: write the calibrated ``derived/anchor/<stamp>/...``
    copies (cloud always; trajectory/splat when present). On big scenes this reads and
    rewrites the full splat (GB-class) — never run it inside a request; the broker is
    1 CPU / 4 GB (see ``modal_oreos_anchor.py``).

    Raises ``KeyError("no_geometry")`` when the scan has no stored cloud to calibrate.
    NEVER mutates ``cloud.npz``/``splat.ply``/``trajectory.npz``.
    """
    cloud = persistence.get_cloud(user_id, scan_id)
    if cloud is None or cloud[0].shape[0] == 0:
        raise KeyError("no_geometry")
    positions, colors = cloud
    extent_before = cloud_extent(np.asarray(positions))
    calibrated_positions = np.asarray(positions, dtype=np.float32) * np.float32(scale_factor)
    extent_after_m = extent_before * scale_factor

    stamp = stamp or f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    calibrated_cloud_key = persistence.save_derived_artifact(
        user_id, scan_id, f"anchor/{stamp}/cloud.ply",
        build_ply_bytes(calibrated_positions, colors),
    )

    calibrated_trajectory_key = None
    traj = persistence.get_trajectory(user_id, scan_id)
    if traj is not None:
        poses = np.asarray(traj.get("poses"), dtype=np.float32).copy()
        if poses.size:
            poses[:, :3, 3] *= np.float32(scale_factor)  # translate only; rotation is scale-free
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            poses=poses,
            intrinsics=np.asarray(traj.get("intrinsics"), dtype=np.float32),
            source_frame_id=np.asarray(traj.get("source_frame_id"), dtype=np.float32),
        )
        calibrated_trajectory_key = persistence.save_derived_artifact(
            user_id, scan_id, f"anchor/{stamp}/trajectory.npz", buf.getvalue()
        )

    calibrated_splat_key = None
    get_splat_path = getattr(persistence, "get_splat_path", None)
    splat_path = get_splat_path(user_id, scan_id) if callable(get_splat_path) else None
    if splat_path:
        scaled_fields = scale_splat_fields(read_splat_ply(splat_path), scale_factor)
        calibrated_splat_key = persistence.save_derived_artifact(
            user_id, scan_id, f"anchor/{stamp}/splat.ply", serialize_splat_ply(scaled_fields)
        )

    return {
        "calibrated_cloud_key": calibrated_cloud_key,
        "calibrated_trajectory_key": calibrated_trajectory_key,
        "calibrated_splat_key": calibrated_splat_key,
        "cloud_extent_before": extent_before,
        "cloud_extent_after_m": extent_after_m,
    }


def apply_metric_anchor(
    persistence: Any,
    user_id: str,
    scan_id: str,
    point_a: Any,
    point_b: Any,
    distance_m: float,
) -> dict[str, Any]:
    """Core logic behind ``POST /api/scenes/<scan_id>/anchor`` (synchronous form —
    ``compute_anchor_scale`` + ``materialize_anchor_artifacts`` in one call; the async
    route splits them across the request and a background job).

    Raises ``ValueError`` on bad input (→ 400 at the route) or ``KeyError("no_geometry")``
    when the scan has no stored cloud to calibrate (→ 404). NEVER mutates
    ``cloud.npz``/``splat.ply``/``trajectory.npz`` — every output is a NEW
    ``derived/anchor/<stamp>/...`` artifact.
    """
    measured, scale_factor = compute_anchor_scale(point_a, point_b, distance_m)
    artifacts = materialize_anchor_artifacts(persistence, user_id, scan_id, scale_factor)

    return {
        "scale_factor": scale_factor,
        "measured_distance": measured,
        "distance_m": float(distance_m),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "calibrated_cloud_key": artifacts["calibrated_cloud_key"],
        "calibrated_trajectory_key": artifacts["calibrated_trajectory_key"],
        "calibrated_splat_key": artifacts["calibrated_splat_key"],
        # Human-readable before/after: the exact segment the operator measured (SLAM-unit ->
        # metres, true by construction) and the scene's overall extent under the same gauge.
        "gauge_span_before": measured,
        "gauge_span_after_m": measured * scale_factor,
        "cloud_extent_before": artifacts["cloud_extent_before"],
        "cloud_extent_after_m": artifacts["cloud_extent_after_m"],
    }
