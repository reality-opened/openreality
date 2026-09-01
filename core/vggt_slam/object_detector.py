"""
Open-set object detection wrapper for VGGT-SLAM 2.0.

Composes existing PE-Core CLIP + SAM3 functions into a single class.
Uses VGGT-SLAM 2.0 APIs (submap.get_points_in_mask(frame_idx, mask, graph),
submap.get_all_semantic_vectors(), etc.).
"""

import base64
import os
import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image

from vggt_slam.slam_utils import compute_text_embeddings, compute_obb_from_points, overlay_masks


# ---------------------------------------------------------------------------
# Inventory fusion (dedup-before-persistence) — EXP-46 gap G-C
# ---------------------------------------------------------------------------
# Emergency rollback switch. Fusion is on by default; set
# OPENREALITY_OBJECT_FUSION=0/false/no/off before process startup to restore the
# preceding exact-query overlap dedup without rolling back the core release.
_FUSION_ENV = "OPENREALITY_OBJECT_FUSION"
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}

# The shipped dedup dropped one detection per overlapping pair of detections that
# shared an IDENTICAL query string. Two failure modes followed, both measured on
# real product data:
#   * fragments of ONE object persisted as many objects (a scan of a single
#     L-shaped desk carried 10 "desk" detections), because a pair only collapsed
#     when the two boxes actually overlapped — fragments sitting slightly apart
#     survived as separate "objects";
#   * anything the detector was asked for under two spellings ("shelf" vs
#     "shelves") never met at all, because raw query strings were compared.
# So the inventory could not answer "how many X are in this scene" — the counting
# questions EXP-46 needs (PILOT-LOG L9/L10).
#
# The fusion below is the algorithm validated harness-side first, in
# platform/experiments/exp_46_3dbench/shakedown/dedup_inventory.py (measurements in
# shakedown/results/dedup_report.md): scene0663 34 -> 23 detections (desk 10 -> 4,
# which matches the real L-desk; keyboard 6 -> 2) and canonical-office-loop 95 -> 79
# — with "trash can" 13 -> 13 deliberately NOT merged, because that corridor really
# does hold that many bins. Keeping genuinely separate instances separate is the
# binding design constraint, which is why the merge test is SIZE-RELATIVE and there
# is no metre threshold anywhere in this file.

# Irregular plurals the naive fold below gets wrong. Kept deliberately small: a
# missed fold only costs a merge, whereas a wrong fold could cost a real instance.
_PLURAL_EXCEPTIONS = {"shelves": "shelf", "boxes": "box"}

# Two same-label detections are the same object when their centers sit closer than
# this fraction of the mean of the two boxes' longest sides. Size-relative BY
# DESIGN: scenes without a metric anchor are in arbitrary reconstruction units (the
# units-honesty doctrine), so an absolute threshold would be meaningless on them and
# would silently change meaning the moment a scene did get anchored.
_MERGE_CENTER_FRACTION = 0.6

# A fused cluster whose members span more than this multiple of the representative
# box's longest side is still fused, but flagged: that is the shape an over-merge
# would take, so it should be reviewable instead of invisible.
_REVIEW_SPREAD_FACTOR = 1.5

# Floor for confidence weights, so a cluster of all-zero-confidence detections still
# produces a finite centroid instead of a divide-by-zero.
_MIN_CONFIDENCE_WEIGHT = 1e-6

# Corner order matches compute_3d_bbox() above: bottom face CCW, then top face.
_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)


def normalize_object_label(label):
    """Fold a detector query/label into a clustering key.

    Lowercase, whitespace-collapsed, naive singular ("Office Chairs" -> "office
    chair", "shelves" -> "shelf"). Conservative by construction: the fold is only the
    GATE for clustering — the geometry test still has to agree — so an imperfect fold
    ("glasses" -> "glasse") can only cost a merge, never invent one. Two distinct
    classes would have to differ by exactly a trailing "s" AND be spatially
    co-located to fuse wrongly.
    """
    text = " ".join(str(label if label is not None else "").lower().split())
    if text in _PLURAL_EXCEPTIONS:
        return _PLURAL_EXCEPTIONS[text]
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def _detection_confidence(det):
    """Confidence as a finite float; anything missing/unparseable ranks as 0.0."""
    try:
        conf = float(det.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return conf if np.isfinite(conf) else 0.0


def _detection_box(det):
    """``(center, extent)`` as finite float(3,) arrays, or ``None`` if unusable.

    Reads ``center``/``extent`` as an AXIS-ALIGNED box and IGNORES the ``rotation``
    the OBB carries — the same simplification the shipped overlap test made, kept on
    purpose. Ignoring rotation treats each OBB as an axis-aligned box of the same
    dimensions about the same center: neither a strict superset nor subset of the
    true oriented box, but rotation-invariant, cheap, and identical to the behavior
    already in production. The distance arm of the merge test (which does most of the
    merging) is orientation-free regardless, so the simplification can only matter for
    boxes that are near-touching AND strongly rotated relative to each other — and
    there the size-relative distance arm dominates the decision anyway. Upgrading to a
    separating-axis test on the real rotations is a measurable change and should be
    justified by measurement (EXP-46 P3), not assumed.
    """
    box = det.get("bounding_box")
    if not isinstance(box, dict):
        return None
    try:
        center = np.asarray(box["center"], dtype=float).reshape(3)
        extent = np.abs(np.asarray(box["extent"], dtype=float).reshape(3))
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(center)) and np.all(np.isfinite(extent))):
        return None
    return center, extent


def _is_same_object(box_a, box_b):
    """True when two same-label boxes should fuse: size-relative center proximity OR
    axis-aligned overlap. No absolute/metre thresholds."""
    center_a, extent_a = box_a
    center_b, extent_b = box_b
    reach = _MERGE_CENTER_FRACTION * (float(extent_a.max()) + float(extent_b.max())) / 2.0
    if float(np.linalg.norm(center_a - center_b)) < reach:
        return True
    return bool(np.all(np.abs(center_a - center_b) <= (extent_a + extent_b) / 2.0))


def _fusion_enabled():
    """Read the rollback switch at call time so workers and tests can override it."""
    return os.getenv(_FUSION_ENV, "1").strip().lower() not in _FALSE_ENV_VALUES


def _legacy_deduplicate_detections(detections):
    """The pre-v2.2.2 exact-query overlap behavior used by the kill switch."""
    keep = []
    for det in sorted(detections, key=_detection_confidence, reverse=True):
        if not isinstance(det, dict) or not det.get("success") or not det.get("bounding_box"):
            continue
        box = _detection_box(det)
        if box is None:
            # The legacy implementation attempted the raw arrays here and could
            # crash. Keep malformed-but-declared detections isolated instead.
            keep.append(det)
            continue
        center, extent = box
        is_dup = False
        for kept in keep:
            if kept.get("query") != det.get("query"):
                continue
            kept_box = _detection_box(kept)
            if kept_box is None:
                continue
            kept_center, kept_extent = kept_box
            if np.all(np.abs(center - kept_center) < (extent + kept_extent) / 2.0):
                is_dup = True
                break
        if not is_dup:
            keep.append(det)
    return keep


def _member_provenance(det):
    """The identity of one fused member, so a merged count stays auditable."""
    return {
        "query": det.get("query"),
        "confidence": _detection_confidence(det),
        "matched_submap": det.get("matched_submap"),
        "matched_frame": det.get("matched_frame"),
    }


def _fuse_cluster(label, members):
    """Collapse one cluster into a single detection dict.

    The representative (highest-confidence member, ``members[0]``) is COPIED, so the
    result keeps every key the pipeline puts on a detection — ``query``,
    ``matched_submap``/``matched_frame`` (the diagnostic path re-keys deduped
    detections by those), ``box_2d``, ``mask_rle``, ``description`` — and still
    points at a real frame. Only geometry, confidence and the new bookkeeping keys
    are rewritten.
    """
    rep, rep_box = members[0]
    fused = dict(rep)
    fused["label_norm"] = label
    # Fusion is a heuristic, so any COUNT derived from this inventory is provisional
    # and every downstream surface that counts objects must say so.
    fused["provisional_count"] = True
    dedup = {
        "merged": len(members) > 1,
        "n_members": len(members),
        "members": [_member_provenance(det) for det, _ in members],
        "review_flag": False,
        "rule": "same normalized label AND (center distance < "
                f"{_MERGE_CENTER_FRACTION} * mean(max extent) OR aabb overlap)",
    }
    fused["dedup"] = dedup
    if len(members) == 1 or rep_box is None:
        # Nothing fused: leave the representative's box exactly as computed (an
        # untouched OBB, rotation included).
        return fused

    centers = np.array([box[0] for _, box in members], dtype=float)
    extents = np.array([box[1] for _, box in members], dtype=float)
    weights = np.array([_detection_confidence(det) for det, _ in members], dtype=float)
    weights = np.where(weights > 0.0, weights, _MIN_CONFIDENCE_WEIGHT)

    center = (centers * weights[:, None]).sum(axis=0) / weights.sum()
    union_min = (centers - extents / 2.0).min(axis=0)
    union_max = (centers + extents / 2.0).max(axis=0)
    extent = union_max - union_min

    # Merge-safety: members spread over more than _REVIEW_SPREAD_FACTOR box-lengths
    # are the signature of an over-merge. Still fused (the threshold decided that),
    # but surfaced so a human can check rather than discover it in a wrong count.
    spread = float(max(
        np.linalg.norm(a - b) for a in centers for b in centers
    )) if len(centers) > 1 else 0.0
    rep_length = float(rep_box[1].max())
    dedup["review_flag"] = bool(rep_length > 0.0 and spread > _REVIEW_SPREAD_FACTOR * rep_length)
    dedup["member_spread"] = spread
    # The exact union span, for consumers that need the true bounds: the fused box
    # below is centered on the confidence-weighted centroid (the better position
    # estimate) with the union's SIZE, so it is not bit-identical to that span.
    dedup["union_min"] = union_min.tolist()
    dedup["union_max"] = union_max.tolist()

    box = dict(rep["bounding_box"])
    box["center"] = center.tolist()
    box["extent"] = extent.tolist()
    # The fused box is axis-aligned (a union of axis-aligned bounds), so the stored
    # rotation must become identity rather than inherit the representative's OBB
    # rotation, which no longer describes this box.
    box["rotation"] = np.eye(3).tolist()
    box["corners"] = (center + _CORNER_SIGNS * (extent / 2.0)).tolist()
    fused["bounding_box"] = box
    fused["confidence"] = max(_detection_confidence(det) for det, _ in members)
    return fused


class ObjectDetector:
    """
    Encapsulates PE-Core CLIP + SAM3 for open-set 3D object detection.

    Usage:
        od = ObjectDetector(device="cuda")
        text_emb = od.encode_text("chair")
        masks, boxes, scores = od.segment(image_pil, "chair")
        bbox = od.compute_3d_bbox(submap, frame_idx, mask, graph, scene_center)
    """

    def __init__(self, device="cuda", clip_model_name="PE-Core-L14-336",
                 sam3_confidence_threshold=0.30):
        self.device = device

        # Load PE-Core CLIP
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        self.clip_model = pe.CLIP.from_config(clip_model_name, pretrained=True)
        self.clip_model.eval()
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = pe_transforms.get_text_tokenizer(self.clip_model.context_length)
        self.clip_preprocess = pe_transforms.get_image_transform(self.clip_model.image_size)

        # Load SAM3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(sam3_model, confidence_threshold=sam3_confidence_threshold)

    def encode_text(self, query):
        """Encode a text query into a CLIP embedding. Returns (1, D) numpy array."""
        return compute_text_embeddings(self.clip_model, self.clip_tokenizer, query)

    @torch.no_grad()
    def encode_text_vector(self, query):
        """Encode a text query into a (D,) CPU tensor for dot-product matching."""
        text_tokens = self.clip_tokenizer([query]).to(self.device)
        text_emb = self.clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.float().cpu().squeeze(0)

    def segment(self, image_pil, query):
        """Run SAM3 text-prompted segmentation on an image.

        Args:
            image_pil: PIL Image
            query: text prompt string

        Returns:
            (masks, boxes, scores) tensors, or (None, None, None) if nothing detected.
        """
        # SAM3 unconditionally casts backbone features to bf16 in sam3_image.py and
        # relies on outer autocast for LayerNorm fp32 output before the FFN's
        # `enabled=False` block. Match the canonical notebook usage.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            inference_state = self.sam3_processor.set_image(image_pil)
            output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=query)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or len(masks) == 0:
            return None, None, None
        return masks, boxes, scores

    def segment_all(self, image_pil, query):
        """Run SAM3 and return list of (mask_2d, box_2d, score) tuples."""
        masks, boxes, scores = self.segment(image_pil, query)
        if masks is None:
            return []
        results = []
        for i in range(len(scores)):
            mask_2d = masks[i, 0].cpu().numpy() if masks.dim() == 4 else masks[i].cpu().numpy()
            box_2d = boxes[i].cpu().numpy()
            score = scores[i].item()
            results.append((mask_2d, box_2d, score))
        return results

    def compute_3d_bbox(self, submap, frame_idx, mask, graph, scene_center):
        """Compute a 3D oriented bounding box from a 2D mask.

        Uses submap.get_points_in_mask(frame_idx, mask, graph) to get world-frame points,
        then computes OBB via compute_obb_from_points().

        Args:
            submap: Submap object
            frame_idx: frame index within the submap
            mask: 2D boolean mask (H, W)
            graph: PoseGraph instance
            scene_center: (3,) array for recentering

        Returns:
            dict with center, extent, rotation, corners (all recentered), or None.
        """
        points_world = submap.get_points_in_mask(frame_idx, mask, graph)
        if points_world is None or len(points_world) < 10:
            return None

        # Recenter to match viewer coordinates
        points_recentered = points_world - scene_center

        # Remove outliers
        if len(points_recentered) > 50:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_recentered)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            points_recentered = np.asarray(pcd.points)

        if len(points_recentered) < 10:
            return None

        try:
            center, extent, rotation = compute_obb_from_points(points_recentered)
        except ValueError:
            return None

        # Compute 8 corner points
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
        ])
        corners_world = (rotation @ corners_local.T).T + center

        return {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "rotation": rotation.tolist(),
            "corners": corners_world.tolist(),
        }

    normalize_label = staticmethod(normalize_object_label)

    @staticmethod
    def deduplicate_detections(detections):
        """Fuse duplicate and fragment detections into one detection per object instance.

        Signature and return shape are unchanged — a list of detection dicts, only the
        ones carrying ``success`` and a ``bounding_box`` — so no caller changes. What
        changed is that a cluster now collapses into ONE fused detection instead of
        "keep the best, drop the rest", and that clustering is by normalized label
        rather than by raw query string. See the module block comment for the measured
        motivation (EXP-46 gap G-C, PILOT-LOG L9/L10).

        Clustering is one greedy confidence-descending pass. A detection joins a
        cluster when its NORMALIZED label matches the cluster's (so "shelf" and
        "shelves" meet, which raw-string comparison never did) AND it is spatially the
        same object as that cluster's representative, i.e. either

          * their 3D centers are closer than ``0.6 x`` the mean of the two boxes'
            longest sides (size-relative — there is no metre threshold anywhere), or
          * their axis-aligned bounds overlap.

        Candidates are tested against the cluster REPRESENTATIVE only (never against
        the growing fused box), so clusters cannot chain their way across a room.

        The fused detection is the representative's dict with a confidence-weighted
        centroid, the UNION of the member bounds as extent, identity rotation (the
        union is axis-aligned), max member confidence, and a ``dedup`` block holding
        member provenance, the member spread and a ``review_flag`` for suspiciously
        spread clusters. Every returned detection carries ``provisional_count: True``:
        fusion is a heuristic, so counts taken off this inventory are provisional and
        downstream surfaces must present them that way.

        Example — one desk detected twice, once under a plural query (the exact shape
        that used to persist as two objects)::

            a = {"success": True, "query": "desk",  "confidence": 0.8,
                 "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [1.0, 1.0, 1.0]}}
            b = {"success": True, "query": "desks", "confidence": 0.2,
                 "bounding_box": {"center": [0.5, 0.0, 0.0], "extent": [1.0, 1.0, 1.0]}}

        Old behavior: TWO detections — raw queries "desk" != "desks" were never
        compared, so nothing merged.
        New behavior: ONE detection — ``query`` "desk" (the 0.8 member represents the
        cluster), ``bounding_box.center`` ``[0.1, 0.0, 0.0]`` (weighted 0.8/0.2 toward
        it), ``bounding_box.extent`` ``[1.5, 1.0, 1.0]`` (union of the two boxes),
        ``confidence`` 0.8, ``dedup.n_members`` 2, ``dedup.review_flag`` False,
        ``provisional_count`` True.
        """
        if not _fusion_enabled():
            return _legacy_deduplicate_detections(detections)

        usable = [
            det for det in detections
            if isinstance(det, dict) and det.get("success") and det.get("bounding_box")
        ]
        if not usable:
            return []

        clusters = []  # [(label, [(det, box_or_None), ...])]
        for det in sorted(usable, key=_detection_confidence, reverse=True):
            label = normalize_object_label(det.get("query") or det.get("label"))
            box = _detection_box(det)
            placed = False
            if box is not None:
                for cluster_label, members in clusters:
                    rep_box = members[0][1]
                    if cluster_label != label or rep_box is None:
                        continue
                    if _is_same_object(box, rep_box):
                        members.append((det, box))
                        placed = True
                        break
            if not placed:
                # A detection whose box is unusable becomes its own cluster and is
                # returned untouched — never silently dropped (that would change
                # counts) and never merged on geometry we could not read.
                clusters.append((label, [(det, box)]))

        return [_fuse_cluster(label, members) for label, members in clusters]

    @staticmethod
    def image_to_base64(image_np):
        """Convert RGB numpy image to base64-encoded JPEG string."""
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def mask_overlay_to_base64(image_np, mask):
        """Create a mask overlay visualization and return as base64 PNG."""
        overlay = image_np.copy()
        color = np.array([0, 255, 100], dtype=np.uint8)
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 100), 2)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        return base64.b64encode(buffer).decode('utf-8')
