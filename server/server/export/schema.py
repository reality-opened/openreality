"""Export-manifest schema — ``meta/info.json`` and ``meta/episode.json``.

See ``docs/dataset-export.md`` §4.2. These models are declarative (the substance of Stage 0);
they intentionally carry the up-to-scale / not-gravity-aligned caveats so downstream never
assumes metric geometry.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "openreality-export/0.1"

# Synthetic uniform timeline default (keyframes are motion-gated, not fixed fps — see §7.3).
DEFAULT_FPS = 10.0


class FeatureSpec(BaseModel):
    """One declared feature (mirrors the LeRobot features-dict entry)."""

    dtype: str  # "video" | "float32" | ...
    shape: list[int]
    names: list[str] = Field(default_factory=list)


class ExportInfo(BaseModel):
    """``meta/info.json`` — schema + feature spec + coordinate-frame caveats."""

    schema_version: str = SCHEMA_VERSION
    scan_id: str = ""
    fps: float = DEFAULT_FPS
    coordinate_frame: str = "slam_world"
    # The VGGT world frame is up-to-scale and not gravity-aligned (see docs/gotchas.md).
    up_to_scale: bool = True
    gravity_aligned: bool = False
    metric: bool = False
    units_note: str = (
        "Coordinates/sizes are up-to-scale (SLAM world frame); not guaranteed metric."
    )
    features: dict[str, FeatureSpec] = Field(default_factory=dict)


class EpisodeSummary(BaseModel):
    """``meta/episode.json`` — episode-level rollup of one scan."""

    scan_id: str = ""
    num_keyframes: int = 0
    trajectory_length: float = 0.0
    room_type: str = "unknown"
    num_objects: int = 0


def standard_features(height: int, width: int) -> dict[str, FeatureSpec]:
    """The v1 feature set: ego-view video + pose state + ego-motion action + intrinsics (§3)."""
    return {
        "observation.images.ego_view": FeatureSpec(
            dtype="video", shape=[height, width, 3], names=["height", "width", "channel"]
        ),
        "observation.state": FeatureSpec(
            dtype="float32", shape=[7], names=["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        ),
        "action": FeatureSpec(
            dtype="float32", shape=[6], names=["dx", "dy", "dz", "rx", "ry", "rz"]
        ),
        "observation.intrinsics": FeatureSpec(
            dtype="float32", shape=[4], names=["fx", "fy", "cx", "cy"]
        ),
    }


def build_info(scan_id: str, height: int, width: int, fps: float = DEFAULT_FPS) -> ExportInfo:
    """Assemble ``meta/info.json`` for one scan."""
    return ExportInfo(
        scan_id=scan_id, fps=fps, features=standard_features(height, width)
    )


def build_episode_summary(
    scan_id: str,
    num_keyframes: int,
    trajectory_length: float = 0.0,
    room_type: str = "unknown",
    num_objects: int = 0,
) -> EpisodeSummary:
    return EpisodeSummary(
        scan_id=scan_id,
        num_keyframes=num_keyframes,
        trajectory_length=trajectory_length,
        room_type=room_type,
        num_objects=num_objects,
    )
