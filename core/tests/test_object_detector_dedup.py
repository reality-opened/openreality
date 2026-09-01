"""CPU-only tests for inventory fusion (vggt_slam.object_detector.deduplicate_detections).

Covers the EXP-46 gap-G-C fix: cluster same-NORMALIZED-label detections, merge on a
size-relative center distance OR axis-aligned overlap, fuse into one detection with a
confidence-weighted centroid + union bounds, and mark counts provisional. The CLIP/SAM3
model loading in ObjectDetector.__init__ is never touched here (it is lazy and
GPU-bound), so this file runs on a CPU box with neither torch nor open3d installed.

Run:  pytest core/tests/test_object_detector_dedup.py -v
"""

import sys
import types

import numpy as np
import pytest

# object_detector imports torch/cv2/open3d/torchvision at module top for the detector
# itself; the dedup path under test is pure numpy. Stub ONLY the heavy deps that are
# genuinely absent, so this runs in GPU-free CI without clobbering a real install.
# House pattern — see test_decompose_camera.py and the conftest stubs.
for _name in ("torch", "torchvision", "torchvision.transforms", "cv2", "open3d"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)
# The attributes below are filled in on whatever module object is present, gated on the
# attribute rather than on "did *this* file create the stub" — a sibling test module
# (test_decompose_camera / test_normalize_to_sl4) may already have installed a barer
# torch stub, in which case the whole suite fails at import in file order. A real torch
# has all of these, so nothing here can clobber an actual install.
if not hasattr(sys.modules["torchvision"], "transforms"):
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
if not hasattr(sys.modules["torch"], "Tensor"):
    # scipy's array-API compat probes any importable torch for Tensor — keep it happy.
    sys.modules["torch"].Tensor = type("Tensor", (), {})
if not hasattr(sys.modules["torch"], "no_grad"):
    # `@torch.no_grad()` decorates a method in the class body, so it must exist at import.
    sys.modules["torch"].no_grad = lambda: (lambda fn: fn)

from vggt_slam.object_detector import (  # noqa: E402
    ObjectDetector,
    normalize_object_label,
)

dedup = ObjectDetector.deduplicate_detections


def det(query, center, extent=(1.0, 1.0, 1.0), confidence=0.5, **extra):
    """A detection dict shaped like the ones streaming_slam builds."""
    d = {
        "success": True,
        "query": query,
        "confidence": confidence,
        "bounding_box": {
            "center": list(center),
            "extent": list(extent),
            "rotation": np.eye(3).tolist(),
            "corners": [],
        },
        "matched_submap": 0,
        "matched_frame": 0,
    }
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# normalize_object_label — the clustering key
# ---------------------------------------------------------------------------
def test_label_fold_lowercases_collapses_and_singularizes():
    assert normalize_object_label("Office Chairs") == "office chair"
    assert normalize_object_label("  Trash   Can ") == "trash can"
    assert normalize_object_label("chairs") == normalize_object_label("chair") == "chair"
    assert normalize_object_label("shelves") == normalize_object_label("shelf") == "shelf"
    assert normalize_object_label("boxes") == normalize_object_label("box") == "box"


def test_label_fold_leaves_double_s_and_short_words_alone():
    # "glass" must not become "glas", "bus" must not become "bu".
    assert normalize_object_label("glass") == "glass"
    assert normalize_object_label("bus") == "bus"


def test_label_fold_handles_missing_labels():
    assert normalize_object_label(None) == ""
    assert normalize_object_label("") == ""


# ---------------------------------------------------------------------------
# The docstring example: one desk, detected twice, once under a plural query
# ---------------------------------------------------------------------------
def test_docstring_example_fuses_desk_and_desks_into_one():
    a = det("desk", (0.0, 0.0, 0.0), confidence=0.8)
    b = det("desks", (0.5, 0.0, 0.0), confidence=0.2)

    out = dedup([a, b])

    assert len(out) == 1                      # old behavior kept BOTH (raw strings differed)
    fused = out[0]
    assert fused["query"] == "desk"           # the 0.8 member represents the cluster
    assert fused["label_norm"] == "desk"
    assert fused["bounding_box"]["center"] == [0.1, 0.0, 0.0]     # weighted 0.8/0.2
    assert fused["bounding_box"]["extent"] == [1.5, 1.0, 1.0]     # union of the two boxes
    assert fused["confidence"] == 0.8                             # max, not blended
    assert fused["provisional_count"] is True
    assert fused["dedup"]["merged"] is True
    assert fused["dedup"]["n_members"] == 2
    assert fused["dedup"]["review_flag"] is False
    assert [m["query"] for m in fused["dedup"]["members"]] == ["desk", "desks"]


def test_fused_box_is_axis_aligned_and_self_consistent():
    out = dedup([det("desk", (0.0, 0.0, 0.0), confidence=0.8),
                 det("desk", (0.5, 0.0, 0.0), confidence=0.2)])
    box = out[0]["bounding_box"]
    # The union of axis-aligned bounds has no orientation left to inherit.
    assert np.allclose(box["rotation"], np.eye(3))
    corners = np.asarray(box["corners"])
    assert corners.shape == (8, 3)
    center = np.asarray(box["center"])
    half = np.asarray(box["extent"]) / 2.0
    assert np.allclose(corners.min(axis=0), center - half)
    assert np.allclose(corners.max(axis=0), center + half)
    # The true union span is still available for consumers that need it.
    assert out[0]["dedup"]["union_min"] == [-0.5, -0.5, -0.5]
    assert out[0]["dedup"]["union_max"] == [1.0, 0.5, 0.5]


def test_singleton_keeps_its_obb_untouched_but_is_still_annotated():
    rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    lone = det("chair", (2.0, 0.0, 0.0), confidence=0.4)
    lone["bounding_box"]["rotation"] = rotation
    lone["bounding_box"]["corners"] = [[1.0] * 3] * 8

    out = dedup([lone])

    assert len(out) == 1
    # No fusion happened, so the oriented box survives exactly as computed.
    assert out[0]["bounding_box"]["rotation"] == rotation
    assert out[0]["bounding_box"]["corners"] == [[1.0] * 3] * 8
    assert out[0]["confidence"] == 0.4
    # ...but the count is still provisional and the provenance block is present.
    assert out[0]["provisional_count"] is True
    assert out[0]["dedup"]["merged"] is False
    assert out[0]["dedup"]["n_members"] == 1


# ---------------------------------------------------------------------------
# Merge safety — the constraint that matters more than the merging
# ---------------------------------------------------------------------------
def test_genuinely_separate_instances_survive():
    """The canonical-office-loop case: a corridor really does hold many bins
    (dedup_report.md — trash can 13 -> 13). Separated instances must not fuse."""
    bins = [det("trash can", (i * 3.0, 0.0, 0.0), confidence=0.5 - 0.1 * i) for i in range(3)]
    assert len(dedup(bins)) == 3


def test_different_labels_never_fuse_even_when_co_located():
    out = dedup([det("chair", (0.0, 0.0, 0.0), confidence=0.9),
                 det("table", (0.0, 0.0, 0.0), confidence=0.8)])
    assert len(out) == 2
    assert sorted(o["label_norm"] for o in out) == ["chair", "table"]


def test_merge_decision_is_scale_invariant_no_absolute_threshold():
    """Requirement: no metre/absolute thresholds — unanchored scenes are in arbitrary
    reconstruction units, so scaling a whole scene must not change any decision."""
    def scene(scale):
        return [det("desk", (0.0, 0.0, 0.0), extent=(scale,) * 3, confidence=0.8),
                det("desk", (0.5 * scale, 0.0, 0.0), extent=(scale,) * 3, confidence=0.2),
                det("desk", (9.0 * scale, 0.0, 0.0), extent=(scale,) * 3, confidence=0.7)]

    for scale in (0.01, 1.0, 100.0):
        out = dedup(scene(scale))
        assert len(out) == 2, f"scale {scale} changed the merge decision"
        merged = [o for o in out if o["dedup"]["merged"]][0]
        # Geometry scales with the scene, the *decision* does not.
        assert np.allclose(merged["bounding_box"]["extent"], [1.5 * scale, scale, scale])


def test_overlapping_boxes_fuse_even_when_centers_are_far_apart():
    """The OR arm: a long fragment whose center sits beyond the distance threshold but
    whose bounds still overlap the representative is the same object."""
    small = det("desk", (0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), confidence=0.9)
    long_ = det("desk", (5.0, 0.0, 0.0), extent=(10.0, 1.0, 1.0), confidence=0.3)
    # distance arm: 5.0 >= 0.6 * (1 + 10) / 2 = 3.3  -> no; overlap arm: 5.0 <= 5.5 -> yes.
    out = dedup([small, long_])
    assert len(out) == 1
    assert out[0]["dedup"]["n_members"] == 2


def test_spread_cluster_is_fused_but_flagged_for_review():
    rep = det("desk", (0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), confidence=0.9)
    far = det("desk", (2.9, 0.0, 0.0), extent=(10.0, 1.0, 1.0), confidence=0.3)
    out = dedup([rep, far])
    assert len(out) == 1
    assert out[0]["dedup"]["merged"] is True
    assert out[0]["dedup"]["review_flag"] is True     # 2.9 > 1.5 x the rep's longest side
    assert out[0]["dedup"]["member_spread"] == pytest.approx(2.9)


def test_candidates_are_tested_against_the_representative_not_the_growing_box():
    """No chaining: clusters must not walk across a scene by re-measuring against their
    own (widened) fused box."""
    a = det("desk", (0.00, 0.0, 0.0), confidence=0.9)
    b = det("desk", (0.55, 0.0, 0.0), confidence=0.5)   # fuses with a
    c = det("desk", (1.05, 0.0, 0.0), confidence=0.4)   # would fuse with the a+b box
    out = dedup([a, b, c])
    assert len(out) == 2
    assert sorted(o["dedup"]["n_members"] for o in out) == [1, 2]


# ---------------------------------------------------------------------------
# Contract preservation — callers must not need changing
# ---------------------------------------------------------------------------
def test_fused_detection_keeps_the_representative_identity_and_payload_keys():
    """streaming_slam re-keys deduped detections by (matched_submap, matched_frame,
    query) and reads box_2d/mask_rle/description off them, so the survivor must stay a
    real member with all of its payload."""
    rep = det("keyboard", (0.0, 0.0, 0.0), confidence=0.9,
              matched_submap=3, matched_frame=7,
              box_2d=[1.0, 2.0, 3.0, 4.0], mask_rle={"size": [4, 4], "counts": "abc"},
              description="black mechanical keyboard", keyframe_image=None, error=None)
    other = det("keyboards", (0.3, 0.0, 0.0), confidence=0.2,
                matched_submap=5, matched_frame=11)

    fused = dedup([rep, other])[0]

    assert (fused["matched_submap"], fused["matched_frame"]) == (3, 7)
    assert fused["query"] == "keyboard"
    assert fused["box_2d"] == [1.0, 2.0, 3.0, 4.0]
    assert fused["mask_rle"] == {"size": [4, 4], "counts": "abc"}
    assert fused["description"] == "black mechanical keyboard"
    assert fused["success"] is True
    # The dropped member is still auditable rather than lost.
    assert {(m["matched_submap"], m["matched_frame"]) for m in fused["dedup"]["members"]} == {
        (3, 7), (5, 11),
    }


def test_unsuccessful_and_boxless_detections_are_still_dropped():
    out = dedup([
        {"success": False, "query": "chair", "confidence": 0.9, "bounding_box": {}},
        {"success": True, "query": "chair", "confidence": 0.8},          # no box
        det("chair", (0.0, 0.0, 0.0), confidence=0.7),
    ])
    assert len(out) == 1
    assert out[0]["confidence"] == 0.7


def test_returns_empty_list_for_empty_input():
    assert dedup([]) == []


def test_fusion_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENREALITY_OBJECT_FUSION", raising=False)
    a = det("desk", (0.0, 0.0, 0.0), confidence=0.8)
    b = det("desks", (0.4, 0.0, 0.0), confidence=0.2)

    out = dedup([a, b])

    assert len(out) == 1
    assert out[0]["dedup"]["merged"] is True


@pytest.mark.parametrize("disabled_value", ["0", "false", "FALSE", "no", "off"])
def test_kill_switch_restores_exact_query_overlap_dedup(monkeypatch, disabled_value):
    monkeypatch.setenv("OPENREALITY_OBJECT_FUSION", disabled_value)
    exact_a = det("desk", (0.0, 0.0, 0.0), confidence=0.8)
    exact_b = det("desk", (0.4, 0.0, 0.0), confidence=0.2)
    plural = det("desks", (0.3, 0.0, 0.0), confidence=0.6)

    out = dedup([exact_a, exact_b, plural])

    assert len(out) == 2
    assert {item["query"] for item in out} == {"desk", "desks"}
    assert all("dedup" not in item for item in out)


def test_does_not_mutate_its_inputs():
    a = det("desk", (0.0, 0.0, 0.0), confidence=0.8)
    b = det("desk", (0.4, 0.0, 0.0), confidence=0.2)
    before = (dict(a), dict(a["bounding_box"]), dict(b), dict(b["bounding_box"]))

    dedup([a, b])

    assert (a, a["bounding_box"], b, b["bounding_box"]) == before


# ---------------------------------------------------------------------------
# Degenerate inputs must degrade, never crash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("box", [
    {"center": [0.0, 0.0], "extent": [1.0, 1.0, 1.0]},          # wrong arity
    {"center": [0.0, 0.0, 0.0], "extent": [float("nan")] * 3},  # non-finite
    {"center": [0.0, 0.0, 0.0]},                                # no extent
    {"center": "nope", "extent": [1.0, 1.0, 1.0]},              # unparseable
])
def test_unusable_geometry_is_kept_as_its_own_object_not_dropped(box):
    bad = {"success": True, "query": "chair", "confidence": 0.9, "bounding_box": box}
    good = det("chair", (0.0, 0.0, 0.0), confidence=0.4)

    out = dedup([bad, good])

    # Dropping it would silently change the count; merging it would use geometry we
    # could not read. It stays its own untouched object.
    assert len(out) == 2
    assert [o["dedup"]["n_members"] for o in out] == [1, 1]


def test_zero_confidence_cluster_still_yields_a_finite_centroid():
    out = dedup([det("desk", (0.0, 0.0, 0.0), confidence=0.0),
                 det("desk", (0.4, 0.0, 0.0), confidence=0.0)])
    assert len(out) == 1
    assert out[0]["bounding_box"]["center"] == pytest.approx([0.2, 0.0, 0.0])
    assert out[0]["confidence"] == 0.0


def test_missing_confidence_is_treated_as_zero_and_ranks_last():
    scored = det("desk", (0.0, 0.0, 0.0), confidence=0.6)
    unscored = det("desk", (0.3, 0.0, 0.0))
    del unscored["confidence"]
    fused = dedup([unscored, scored])[0]
    assert fused["confidence"] == 0.6
    assert fused["dedup"]["members"][0]["confidence"] == 0.6   # scored member represents
