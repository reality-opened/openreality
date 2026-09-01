"""Python mirror of the ``@reality/protocol`` object-layer wire contract.

Source of truth is the TypeScript in ``web/packages/protocol/objectLayer.ts`` — this module
mirrors it so the server can emit/serve the same snake_case shapes (exactly as
``server/scene_report/schemas.py`` mirrors the report/facts wire types). The object layer is
the AI-completed 3D layer that overlays a finished scan: for a subset of detected objects we
generate a world-baked ``.glb`` and place it back at its measured pose with an honesty
envelope (per-object quality tier, provenance, caveats). The webserver embed viewer consumes
an ``ObjectLayerManifest`` attached to ``PersistedScene.object_layer``.

Geometry conventions (must match ``BoundingBox3D`` / the TS contract):
  - ``center``   : object centre in the SLAM world frame, ``[x, y, z]``.
  - ``extents``  : FULL edge lengths of the oriented box (NOT half-extents).
  - ``rotation`` : 3x3, **columns are the box's local axes** expressed in the world frame.
Generated ``.glb`` assets are baked in the same world frame, so the viewer only offsets them
by the scan's ``scene_center`` (exactly like the detection boxes).

This is an ADDITIVE contract — nothing here changes existing shapes.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# Keep these in lockstep with objectLayer.ts.
OBJECT_LAYER_SCHEMA_VERSION = 1
DEFAULT_OBJECT_LAYER_DISCLAIMER = (
    "Generated assets complete unseen geometry; envelope-verified, functional parts unverified."
)
DEFAULT_PROVENANCE = "AI-completed — envelope-verified"

ObjectQualityTier = Literal["good", "usable", "low"]


def _all_finite(values: list[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


class ObjectLayerOBB(BaseModel):
    """Oriented bounding box, world frame. Mirrors ``ObjectLayerOBB`` / ``BoundingBox3D``."""

    center: list[float]
    extents: list[float]  # FULL edge lengths [x, y, z]
    rotation: list[list[float]]  # 3x3, columns = box axes in world

    @field_validator("center", "extents")
    @classmethod
    def _vec3(cls, v: list[float]) -> list[float]:
        if len(v) != 3 or not _all_finite(v):
            raise ValueError("must be 3 finite numbers")
        return [float(x) for x in v]

    @field_validator("rotation")
    @classmethod
    def _mat3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 or not _all_finite(row) for row in v):
            raise ValueError("must be a 3x3 matrix of finite numbers")
        return [[float(x) for x in row] for row in v]


class ObjectLayerItem(BaseModel):
    """One generated object in the layer. ``low`` quality renders box-only in the viewer."""

    id: str = Field(min_length=1)
    label: str
    obb: ObjectLayerOBB
    quality: ObjectQualityTier
    provenance: str = DEFAULT_PROVENANCE
    caveats: list[str] = Field(default_factory=list)
    # Optional; omitted/ignored for ``low`` (box-only). URL of the world-baked ``.glb``.
    asset_url: Optional[str] = None
    source_frame: Optional[str] = None
    submap: Optional[str] = None


class ObjectLayerManifest(BaseModel):
    """The object layer bundled with an embed export (``PersistedScene.object_layer``)."""

    version: int = OBJECT_LAYER_SCHEMA_VERSION
    objects: list[ObjectLayerItem] = Field(default_factory=list)
    scan_id: Optional[str] = None
    generated_at: Optional[str] = None
    frame: Optional[str] = None
    disclaimer: Optional[str] = DEFAULT_OBJECT_LAYER_DISCLAIMER


def normalize_quality_tier(raw: Any) -> ObjectQualityTier:
    """Coerce an arbitrary quality label into a tier. Accepts the EXP-19/EXP-20 upper-case
    vocabulary ("GOOD"/"USABLE"/"LOW"/"STILL-CHIMERA"/"NO-GO"/…). Anything unrecognized or
    unshippable falls back to ``low`` (box-only) — the conservative, honesty-preserving default.
    Mirrors ``normalizeQualityTier`` in objectLayer.ts."""
    s = str(raw if raw is not None else "").strip().lower()
    if s == "good":
        return "good"
    if s == "usable":
        return "usable"
    return "low"


def parse_object_layer_manifest(raw: Any) -> Optional[ObjectLayerManifest]:
    """Parse an untrusted value (dict) into an ``ObjectLayerManifest``. Returns ``None`` on any
    malformed input; never raises. Use at the trust boundary. Mirrors the never-throw
    ``parseObjectLayerManifest`` in objectLayer.ts."""
    if raw is None:
        return None
    try:
        if isinstance(raw, ObjectLayerManifest):
            return raw
        return ObjectLayerManifest.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return None
