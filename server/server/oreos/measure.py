"""Deterministic measurement math + metric-state resolution (features.md F7, server side).

ONE implementation shared by ``POST /api/scenes/<id>/measure`` (routes_agent.py) and the
scene agent's ``measure.*`` tools (persisted_agent.py), so the route and the agent can
never disagree on a number. Pure numpy — no flask, no store access; callers pass the
persisted scene record (for the metric state) and world-frame points.

Doctrine (WORLD-TRANSFORM-CONTRACT.md):
  * inputs are WORLD-frame SLAM-unit points;
  * metres exist ONLY via the metric anchor — ``record.derived_latest.kind == "anchor"``
    carries ``scale_factor`` (metres per SLAM unit);
  * every length-bearing result states ``units: "m" | "relative"`` + ``scale_source``;
  * the "m" glyph is client-side forbidden for ``relative`` (chip enforces) — the server
    never pre-formats a unit string onto a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Reject a degenerate (near-coincident) point pair — matches
# ``server.app._MIN_ANCHOR_DISTANCE_UNITS`` (itself a port of core's
# ``vggt_slam.metric_absolute.MIN_DISTANCE_UNITS``).
MIN_DISTANCE_UNITS = 1e-6

MEASURE_KINDS = ("distance", "angle")


class MeasureError(ValueError):
    """Invalid measurement input. ``code`` is the wire error id (→ 4xx at the route)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MetricState:
    """The scene's resolved scale state — the single source every emitter cites.

    ``wording`` is the exact honesty phrase prose must use for lengths
    (CLAIM-LEDGER: unanchored numbers are "relative units (uncalibrated)"; anchored are
    metres; depth-anchored metres additionally read "depth-scaled estimate").
    """

    anchored: bool
    scale_factor: Optional[float]  # metres per SLAM unit, when anchored
    scale_source: str  # "anchor:<derived key>" | "none"
    units: str  # "m" | "relative"
    units_basis: str  # WORLD-TRANSFORM units_basis value ("anchor:<key>" | "slam")
    wording: str  # "metres" | "depth-scaled estimate (metres)" | "relative units (uncalibrated)"

    def to_payload(self) -> dict[str, Any]:
        return {
            "anchored": self.anchored,
            "scale_factor": self.scale_factor,
            "scale_source": self.scale_source,
            "units": self.units,
            "units_basis": self.units_basis,
            "wording": self.wording,
        }


_UNANCHORED = MetricState(
    anchored=False,
    scale_factor=None,
    scale_source="none",
    units="relative",
    units_basis="slam",
    wording="relative units (uncalibrated)",
)


def resolve_metric(record: Any, depth_provenance: Optional[dict] = None) -> MetricState:
    """Resolve the scene's metric state from its persisted record.

    Metres require the record's ``derived_latest`` pointer of ``kind == "anchor"`` with a
    positive finite ``scale_factor`` (written by ``POST /anchor`` — see
    ``server.app._derived_pointer``). Anything else — no pointer, a clamp pointer, a
    poisoned factor — resolves to relative units. ``depth_provenance`` is the parsed
    ``derived/demo/metric/provenance.json`` doc when the caller has read one (Gemini-2
    depth-ratio anchors); it only changes the honesty *wording*, never the math.
    """
    pointer = record.get("derived_latest") if isinstance(record, dict) else None
    if not isinstance(pointer, dict) or pointer.get("kind") != "anchor":
        return _UNANCHORED
    try:
        factor = float(pointer.get("scale_factor"))
    except (TypeError, ValueError):
        return _UNANCHORED
    if not (np.isfinite(factor) and factor > 0):
        return _UNANCHORED
    source_key = str(pointer.get("source_key") or "").strip()
    if not source_key:
        return _UNANCHORED
    scale_source = f"anchor:{source_key}"
    wording = "metres"
    if isinstance(depth_provenance, dict) and depth_provenance.get("method"):
        wording = "depth-scaled estimate (metres)"
    return MetricState(
        anchored=True,
        scale_factor=factor,
        scale_source=scale_source,
        units="m",
        units_basis=scale_source,
        wording=wording,
    )


def validate_points(points_world: Any, expected: int, kind: str) -> np.ndarray:
    """Validate ``points_world`` as exactly ``expected`` finite ``[x, y, z]`` triples
    (the ``server.app._validate_anchor_point`` posture). Raises :class:`MeasureError`."""
    if not isinstance(points_world, (list, tuple)):
        raise MeasureError(
            "invalid_points", f"'points_world' must be a list of [x,y,z] points, got {type(points_world).__name__}"
        )
    if len(points_world) != expected:
        raise MeasureError(
            "invalid_points",
            f"'{kind}' needs exactly {expected} points_world entries, got {len(points_world)}",
        )
    rows = []
    for i, value in enumerate(points_world):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise MeasureError(
                "invalid_points", f"points_world[{i}] must be [x, y, z], got {value!r}"
            )
        try:
            row = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise MeasureError(
                "invalid_points", f"points_world[{i}] must be numeric, got {value!r}"
            ) from None
        if row.shape != (3,) or not np.all(np.isfinite(row)):
            raise MeasureError(
                "invalid_points", f"points_world[{i}] must contain finite numbers, got {value!r}"
            )
        rows.append(row)
    return np.stack(rows, axis=0)


def measure_distance(points_world: Any, metric: MetricState) -> dict[str, Any]:
    """2 world points → the F7 ``MeasureResponse`` dict (protocol ``demo.ts``).

    ``value`` is metres when anchored (``units: "m"``), else raw SLAM units
    (``units: "relative"``). ``value_slam`` (additive) always carries the raw
    SLAM-unit distance for the audit trail."""
    pts = validate_points(points_world, 2, "distance")
    raw = float(np.linalg.norm(pts[0] - pts[1]))
    if not np.isfinite(raw) or raw < MIN_DISTANCE_UNITS:
        raise MeasureError(
            "degenerate_points",
            f"point pair distance {raw:.3g} is degenerate (< {MIN_DISTANCE_UNITS:.0e}); "
            "pick two points that are clearly apart",
        )
    value = raw * metric.scale_factor if metric.anchored else raw
    return {
        "kind": "distance",
        "value": float(value),
        "units": metric.units,
        "scale_factor": metric.scale_factor,
        "scale_source": metric.scale_source,
        "value_slam": raw,
        "metric_state": "metric_anchored" if metric.anchored else "up_to_scale",
    }


def measure_angle(points_world: Any, metric: MetricState) -> dict[str, Any]:
    """3 world points → angle at the SECOND point (the vertex), in degrees.

    Angles are scale-invariant, so ``degrees`` never depends on the anchor; the
    ``units``/``scale_source`` fields still report the scene's metric state (they drive
    the provenance chip, not the number). Collinear points are a valid 0°/180° answer —
    only a zero-length arm is degenerate."""
    pts = validate_points(points_world, 3, "angle")
    v1 = pts[0] - pts[1]
    v2 = pts[2] - pts[1]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < MIN_DISTANCE_UNITS or n2 < MIN_DISTANCE_UNITS:
        raise MeasureError(
            "degenerate_points",
            "angle needs the vertex (2nd point) clearly apart from both endpoints",
        )
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    degrees = float(np.degrees(np.arccos(cosine)))
    return {
        "kind": "angle",
        "value": degrees,
        "units": metric.units,
        "scale_factor": metric.scale_factor,
        "scale_source": metric.scale_source,
        "degrees": degrees,
        "metric_state": "metric_anchored" if metric.anchored else "up_to_scale",
    }


def measure(kind: Any, points_world: Any, metric: MetricState) -> dict[str, Any]:
    """Dispatch by ``kind`` (the route/tool entrypoint)."""
    if kind == "distance":
        return measure_distance(points_world, metric)
    if kind == "angle":
        return measure_angle(points_world, metric)
    raise MeasureError(
        "invalid_kind", f"kind must be one of {list(MEASURE_KINDS)}, got {kind!r}"
    )
