"""Offline ingest: EXP-20 ``scene_inventory.json`` -> ``ObjectLayerManifest`` + curation.

Used by ``scripts/attach_object_layer.py`` (the embed-regen tool). Kept out of the runtime
request path — this is import-time-cheap (pydantic only) but conceptually "tooling".

Two stages, deliberately separated so each is testable in isolation:
  1. ``convert_scene_inventory`` — geometry + metadata only. Transposes the inventory's
     ROW-major ``world_obb.axes_rows`` into our COLUMNS-are-axes ``rotation``; takes the FULL
     ``extent`` (ignores ``half_extent``); maps quality/provenance/source_frame. It does NOT
     resolve asset files or copy inventory ``notes`` into caveats (notes are internal —
     customer caveats are curated at regen time). ``asset_url`` is left ``None`` here; asset
     resolution is the attach step's job (file presence on disk is authoritative).
  2. ``curate_manifest`` — apply the regen curation pass: include/exclude ids, per-id curated
     caveats, a box-only caveat for objects with no mesh, optional drop-box-only.

Mirrors ``convertSceneInventory`` in ``web/packages/protocol/objectLayer.ts`` on the geometry,
but intentionally defers asset resolution to the attach step (the browser harness resolves
asset URLs eagerly; the server resolves them from the world-baked GLB dir).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from server.scene_report.object_layer import (
    DEFAULT_OBJECT_LAYER_DISCLAIMER,
    DEFAULT_PROVENANCE,
    OBJECT_LAYER_SCHEMA_VERSION,
    ObjectLayerItem,
    ObjectLayerManifest,
    ObjectLayerOBB,
    normalize_quality_tier,
)


def transpose_3x3(m: Sequence[Sequence[float]]) -> list[list[float]]:
    """Swap rows<->columns of a 3x3. The inventory emits ``axes_rows`` (one world axis per ROW);
    ``ObjectLayerOBB.rotation`` is columns-are-axes, so the transpose is mandatory."""
    return [[float(m[r][c]) for r in range(3)] for c in range(3)]


def _is_3x3(v: Any) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in v)
    )


def _inventory_obb(world_obb: Optional[Mapping[str, Any]]) -> Optional[ObjectLayerOBB]:
    """Build an ``ObjectLayerOBB`` from an inventory ``world_obb``, or ``None`` if unplaceable.

    - ``center``   : taken as-is (world frame).
    - ``extents``  : FULL edge lengths — prefer ``extents``/``extent`` (== 2x ``half_extent`` in
      the real data); ``half_extent`` is deliberately NOT used.
    - ``rotation`` : a pre-transposed ``rotation`` (columns-are-axes) wins; else transpose the
      ROW-major ``axes_rows``.
    """
    if not isinstance(world_obb, Mapping):
        return None
    center = world_obb.get("center")
    extents = world_obb.get("extents")
    if extents is None:
        extents = world_obb.get("extent")
    rotation = world_obb.get("rotation")
    if not _is_3x3(rotation):
        axes_rows = world_obb.get("axes_rows")
        rotation = transpose_3x3(axes_rows) if _is_3x3(axes_rows) else None
    try:
        return ObjectLayerOBB(center=center, extents=extents, rotation=rotation)
    except Exception:
        return None


def _source_frame(obj: Mapping[str, Any]) -> Optional[str]:
    if obj.get("source_frame"):
        return str(obj["source_frame"])
    submap = obj.get("submap")
    best_frame = obj.get("best_frame")
    if submap is not None and best_frame is not None:
        return f"submap {submap} · frame {best_frame}"
    if submap is not None:
        return f"submap {submap}"
    return None


def convert_scene_inventory(
    inventory: Mapping[str, Any],
    *,
    scan_id: Optional[str] = None,
    default_provenance: Optional[str] = None,
    disclaimer: Optional[str] = None,
) -> ObjectLayerManifest:
    """Convert an EXP-20 ``scene_inventory.json`` (already parsed) into an ``ObjectLayerManifest``.

    Objects without a placeable ``world_obb`` are skipped (they can't be positioned). Quality is
    normalized to good/usable/low. ``asset_url`` is left unset (resolved at attach time); caveats
    are left empty (inventory ``notes`` are internal and NOT surfaced — curate them explicitly).
    """
    prov_default = default_provenance or str(inventory.get("provenance_note") or DEFAULT_PROVENANCE)
    items: list[ObjectLayerItem] = []
    for obj in inventory.get("objects") or []:
        if not isinstance(obj, Mapping):
            continue
        oid = obj.get("id")
        if not isinstance(oid, str) or not oid:
            continue
        obb = _inventory_obb(obj.get("world_obb"))
        if obb is None:
            continue
        items.append(
            ObjectLayerItem(
                id=oid,
                label=str(obj.get("label") or obj.get("label_guess") or oid),
                obb=obb,
                quality=normalize_quality_tier(obj.get("quality_tier") or obj.get("quality")),
                provenance=str(obj.get("provenance") or prov_default),
                caveats=[],  # notes are internal; caveats are curated at regen time
                asset_url=None,  # resolved by the attach step
                source_frame=_source_frame(obj),
                submap=str(obj["submap"]) if obj.get("submap") is not None else None,
            )
        )
    return ObjectLayerManifest(
        version=int(inventory.get("version") or OBJECT_LAYER_SCHEMA_VERSION),
        objects=items,
        scan_id=scan_id or (str(inventory["scan_id"]) if inventory.get("scan_id") else None),
        generated_at=(
            str(inventory.get("generated_at") or inventory.get("generated_utc"))
            if (inventory.get("generated_at") or inventory.get("generated_utc"))
            else None
        ),
        frame=str(inventory["frame"]) if inventory.get("frame") else None,
        disclaimer=str(
            inventory.get("disclaimer") or disclaimer or DEFAULT_OBJECT_LAYER_DISCLAIMER
        ),
    )


CaveatSpec = Union[str, Sequence[str]]


def curate_manifest(
    manifest: ObjectLayerManifest,
    *,
    exclude_ids: Iterable[str] = (),
    include_ids: Iterable[str] = (),
    caveats: Optional[Mapping[str, CaveatSpec]] = None,
    box_only_caveat: Optional[str] = None,
    drop_box_only: bool = False,
) -> ObjectLayerManifest:
    """Apply the regen curation pass and return a new manifest (input untouched).

    - ``include_ids`` (allowlist): keep only these ids; ``exclude_ids`` still removes from it.
    - ``caveats``: id -> curated customer-facing caveat(s); replaces that item's caveats.
    - ``box_only_caveat``: applied to kept items with no ``asset_url`` (mesh absent) that have no
      explicit curated caveat.
    - ``drop_box_only``: drop objects with no ``asset_url`` entirely instead of keeping the box.
    """
    exclude = {str(x) for x in exclude_ids}
    include = {str(x) for x in include_ids}
    caveat_map = dict(caveats or {})
    kept: list[ObjectLayerItem] = []
    for item in manifest.objects:
        if include and item.id not in include:
            continue
        if item.id in exclude:
            continue
        new = item.model_copy(deep=True)
        if item.id in caveat_map:
            spec = caveat_map[item.id]
            new.caveats = [spec] if isinstance(spec, str) else [str(c) for c in spec]
        if not new.asset_url:  # box-only (no placeable mesh)
            if drop_box_only:
                continue
            if box_only_caveat and not new.caveats:
                new.caveats = [box_only_caveat]
        kept.append(new)
    return manifest.model_copy(update={"objects": kept})
