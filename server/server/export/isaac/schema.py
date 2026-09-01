"""Isaac/USD export manifest — ``isaac/manifest.json``.

Declarative record of *how* the SLAM scan was placed into the Isaac world: the alignment
transform, where the metric scale came from, and whether the result is actually metric /
gravity-aligned. Downstream (and a human loading the USD in Isaac Sim) reads this to know how
much to trust the geometry — mirroring the up-to-scale caveats in ``server/export/schema.py``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

ISAAC_SCHEMA_VERSION = "openreality-isaac/0.1"

# Isaac Sim / Isaac Lab robotics convention (USD's unauthored fallbacks are cm + Y-up).
ISAAC_UP_AXIS = "Z"
ISAAC_METERS_PER_UNIT = 1.0


class AlignmentInfo(BaseModel):
    """How SLAM world was mapped to the Isaac world frame."""

    # 4x4 row-major: p_isaac = T_align @ [p_slam, 1]  (includes scale, rotation, translation).
    transform: list[list[float]] = Field(default_factory=list)
    scale: float = 1.0
    scale_source: str = "none"  # "user:factor" | "user:ref_object:<q>" | "none"
    metric: bool = False
    up_axis: str = ISAAC_UP_AXIS
    meters_per_unit: float = ISAAC_METERS_PER_UNIT
    gravity_aligned: bool = False
    # Estimated SLAM-frame up (gravity) unit vector before alignment.
    up_vector_slam: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    # Fraction of cloud points on the floor plane — a rough confidence in the gravity estimate.
    floor_inlier_fraction: float = 0.0
    gravity_source: str = "floor_plane"  # "floor_plane" | "user_override"
    recentered_xy: bool = True


class GeometryInfo(BaseModel):
    """What geometry the ``scene.usd`` contains."""

    representation: str = "mesh"  # "mesh" | "points" | "mesh+points"
    num_vertices: int = 0
    num_faces: int = 0
    num_points: int = 0
    has_collision: bool = False
    reconstruction: str = "poisson"  # "poisson" | "tsdf" | "none(points)"


class IsaacManifest(BaseModel):
    """``isaac/manifest.json`` — the full provenance of one Isaac/USD export."""

    schema_version: str = ISAAC_SCHEMA_VERSION
    scan_id: str = ""
    fps: float = 10.0
    num_keyframes: int = 0
    scene_usd: str = "scene.usd"
    trajectory_usd: str = "trajectory.usd"
    alignment: AlignmentInfo = Field(default_factory=AlignmentInfo)
    geometry: GeometryInfo = Field(default_factory=GeometryInfo)
    notes: str = ""
