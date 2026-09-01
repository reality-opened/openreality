"""Pydantic schemas for the scene report subsystem.

Everything the LLM produces is grounded: ``ObjectInstance``/``ReportObject`` carry
``EvidenceRef``s pointing back at the keyframe(s) that support the claim.
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    """Pointer to the keyframe that supports a fact.

    ``box_2d`` / ``mask_rle`` are **additive, optional** and default ``None`` so every existing
    consumer is untouched. When present they carry the *real* per-frame SAM detection geometry
    that ``StreamingSLAM`` computed on this evidence keyframe (2D pixel box ``[x0,y0,x1,y1]`` and,
    optionally, the mask as a compact run-length dict — see ``server/export/mask_rle.py``). This
    lets the dynamics producer seed BootsTAPIR from the persisted detection instead of
    re-deriving a region from the projected 3D box. See ``contracts/export-format.md``.
    """

    submap_id: int
    frame_idx: int
    box_2d: Optional[list[float]] = None  # [x0, y0, x1, y1] pixels, in the evidence keyframe
    mask_rle: Optional[dict] = None  # compact RLE (server/export/mask_rle.py), optional


class SceneMetrics(BaseModel):
    """Coarse, up-to-scale geometry of the scanned space (SLAM world frame)."""

    dimensions: list[float] = Field(default_factory=list)  # AABB extents, sorted desc
    bbox_min: list[float] = Field(default_factory=list)
    bbox_max: list[float] = Field(default_factory=list)
    trajectory_length: float = 0.0
    num_submaps: int = 0
    num_keyframes: int = 0
    point_count: int = 0
    # VGGT world frame is not gravity-aligned, so we cannot reliably name a
    # vertical axis / floor-to-ceiling height until something derives one.
    vertical_axis_known: bool = False
    # --- ground frame (server/oreos/ground_frame.py; additive, backward-compatible) ---
    # Present once a plane fit has succeeded WELL ENOUGH to publish: a weak fit records
    # only ``vertical_axis_known=False`` + ``derivation`` + ``ground_frame_note``, so a
    # consumer can never read a floor height that was not actually supported. Heights are
    # signed distances along ``up_axis`` in the world frame (``ceiling_height -
    # floor_height`` is the room height), in SLAM units like every other metric here.
    up_axis: Optional[list[float]] = None
    floor_height: Optional[float] = None
    ceiling_height: Optional[float] = None
    room_height: Optional[float] = None
    floor_extent: Optional[list[float]] = None  # [width, depth] of the floor footprint
    floor_area: Optional[float] = None  # occupancy-based, NOT width x depth
    derivation: Optional[str] = None  # "plane-fit"
    ground_frame_note: Optional[str] = None  # why a weak/absent fit was not published


class ProductMatch(BaseModel):
    """One candidate product hit from reverse image search (e.g. Google Lens)."""

    title: str = ""
    source: str = ""
    link: str = ""
    price: str = ""


class ObjectDescription(BaseModel):
    """Fine-grained, evidence-grounded identification of a detected object.

    Produced by ``ObjectEnricher`` from a crop of the object's best keyframe (a VLM
    captioning pass, optionally grounded by reverse image search). ``category`` is the
    always-safe generic label; ``brand``/``model``/``color`` are the fine-grained guess
    that ``confidence`` + ``uncertainty_note`` qualify. ``visible_text`` is what the model
    could literally read (logos/labels), distinguished from inferred fields.
    """

    # ``model`` (brand model) and ``model_name`` (the VLM id) intentionally live in the
    # ``model_`` namespace pydantic reserves; opt out of the protection so they're allowed.
    model_config = {"protected_namespaces": ()}

    title: str = ""  # headline, e.g. "Apple MacBook Pro 14\""
    category: str = ""  # always-safe generic label, e.g. "laptop"
    brand: str = ""
    model: str = ""
    color: str = ""
    materials: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)  # what it could READ (logos/labels)
    distinguishing_features: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # calibrated 0..1
    uncertainty_note: str = ""
    matches: list[ProductMatch] = Field(default_factory=list)  # reverse-image-search hits
    model_name: str = ""  # VLM model id that authored this
    grounded_by_search: bool = False


class ImportedObjectOBB(BaseModel):
    """``ObjectLayerOBB`` on the wire: ``extents`` are FULL edge lengths and ``rotation``
    columns are the box axes, world frame (WORLD-TRANSFORM-CONTRACT.md)."""

    center: list[float]
    extents: list[float]
    rotation: list[list[float]]


class ImportedDetection(BaseModel):
    """Provenance envelope for an object found by GEOMETRY on an imported splat.

    An imported splat has no capture video, so its objects come from clustering the
    gaussian centres after the floor/walls/ceiling are removed — see
    ``server/oreos/imported_objects.py``. That is emphatically NOT the same act as a
    captured scene's detection: no detector ran, no photograph was read, and the parent
    ``ObjectInstance.confidence`` is therefore 0.0 rather than an invented score. This
    block is what lets the UI say so instead of dressing a cluster up as a detection.

    ``obb`` is the tight gravity-aware box; the parent's ``center``/``extent`` stay the
    world AABB so every consumer that predates this field keeps working unchanged.
    """

    method: str = "geometry_cluster"
    provenance: str = ""
    #: ``"geometry"`` (measured size only) or ``"synthetic_view"`` (a VLM named it from a
    #: render of this splat — never from a photograph).
    label_source: str = "geometry"
    #: The measured-size label, kept even after a view label replaces the display name.
    geometry_label: str = ""
    obb: Optional[ImportedObjectOBB] = None
    #: Facts read off the fitted floor/wall planes, e.g. "on the floor", "against a wall".
    placement: list[str] = Field(default_factory=list)
    support_points: int = 0
    support_voxels: int = 0
    units: str = "relative"
    units_basis: str = "slam_world_units"
    up_source: str = ""
    caveats: list[str] = Field(default_factory=list)
    label_model: Optional[str] = None
    label_confidence: Optional[float] = None
    view_id: Optional[str] = None


class ObjectInstance(BaseModel):
    """One detected object instance, located in the SLAM world frame."""

    query: str
    center: list[float]  # (3,) world frame
    extent: list[float] = Field(default_factory=list)  # (3,)
    confidence: float = 0.0
    evidence: list[EvidenceRef] = Field(default_factory=list)
    # Fine-grained identification (filled lazily on click + for the top-N at finalize).
    description: Optional[ObjectDescription] = None
    # --- human review edits (product-workflow Detect stage; additive, backward-compatible) ---
    # A relabel records the operator's correction in ``human_label`` ALONGSIDE the model's
    # original ``query`` (which is NEVER overwritten), so the edit is auditable and the raw
    # detection is never lost. ``dismissed`` marks the instance as hidden from the reviewed
    # inventory without deleting it. Both default to the unedited value, so every existing
    # record and consumer is untouched. Persisted via
    # ``PATCH /api/scenes/<scan_id>/objects/<object_id>`` where ``object_id`` is the 0-based
    # index into this scan's ``SceneFacts.objects`` list.
    human_label: Optional[str] = None
    dismissed: bool = False
    # --- imported-splat geometry detection (additive, backward-compatible) ---
    # Present ONLY on objects produced by ``server/oreos/imported_objects.py``; absent on
    # every real detection, which is exactly what the store's replace guard keys off.
    imported_detection: Optional[ImportedDetection] = None


class ObjectTrack(BaseModel):
    """One *dynamic* object followed across keyframes, in the SLAM world frame.

    The temporal counterpart of ``ObjectInstance``: it mirrors the static fields
    (``query``/``center``/``extent``/``confidence``/``evidence``/``description``) so existing
    static consumers keep working on the time-averaged summary, and *adds* a per-frame track.

    Frame lift invariant (read ``docs/dataset-export.md`` + ``contracts/export-format.md``):
    each ``frame_centers[i]`` is the object's 2D-track centroid unprojected with a robust
    mask-region depth and lifted to world **per-frame via that frame's own VGGT cam->world
    pose** — never via cross-frame consistency. The dynamics + static producers therefore
    share the ``index_map`` clock (frame ordering) but NOT a fused metric frame.

    Caveats: coordinates are **up-to-scale** (SLAM world frame; not metric, not
    gravity-aligned), and the per-frame 3D center is **approximate on fast movers** — VGGT
    depth weakens on dynamic content (see ``experiments/e2_track3d/RESULTS.md``); treat
    ``frame_centers`` as an up-to-scale estimate, strongest when ``frame_visibility`` is True.
    """

    query: str
    object_id: int
    # --- time-averaged summary (mirrors ObjectInstance; static consumers use these) ---
    center: list[float] = Field(default_factory=list)  # (3,) world frame, averaged over visible frames
    extent: list[float] = Field(default_factory=list)  # (3,)
    confidence: float = 0.0
    evidence: list[EvidenceRef] = Field(default_factory=list)
    description: Optional[ObjectDescription] = None
    # --- per-frame temporal track (all lists share one index i, aligned to ``frames``) ---
    frames: list[int] = Field(default_factory=list)  # global keyframe indices (the shared clock)
    frame_centers: list[list[float]] = Field(default_factory=list)  # (M,3) world, per-frame-pose lift
    frame_extents: list[list[float]] = Field(default_factory=list)  # (M,3)
    frame_visibility: list[bool] = Field(default_factory=list)  # (M,)
    frame_confidence: list[float] = Field(default_factory=list)  # (M,)
    first_frame: int = -1  # min(frames), -1 if empty
    last_frame: int = -1  # max(frames), -1 if empty
    motion: str = "static"  # "static" | "moving"
    max_displacement: float = 0.0  # largest world-frame excursion from the first center (up-to-scale)


class SpatialRelation(BaseModel):
    a: str
    b: str
    distance: float
    relation: str  # near | medium | far


class SceneFacts(BaseModel):
    """Structured, evidence-grounded facts derived from the reconstruction."""

    metrics: SceneMetrics = Field(default_factory=SceneMetrics)
    objects: list[ObjectInstance] = Field(default_factory=list)
    object_counts: dict[str, int] = Field(default_factory=dict)
    relations: list[SpatialRelation] = Field(default_factory=list)
    coverage_estimate: float = 0.0
    scene_center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    units_note: str = (
        "Coordinates and sizes are up-to-scale (SLAM world frame); not guaranteed metric."
    )


class ReportObject(BaseModel):
    name: str
    location: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)


class SceneReport(BaseModel):
    """LLM-authored, fact-grounded report of the scanned space."""

    summary: str = ""
    room_type: str = "unknown"
    objects: list[ReportObject] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    coverage_note: str = ""
    facts: SceneFacts = Field(default_factory=SceneFacts)
    model: str = ""
    degraded: bool = False
    # Progressive (in-scan) reports are refreshed mid-scan and superseded by the
    # final report when the scan stops. The end-of-scan report is final.
    progressive: bool = False
    final: bool = True
    generated_at: float = Field(default_factory=time.time)
