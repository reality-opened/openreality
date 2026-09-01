"""CPU tests for persisting the real per-frame SAM detection geometry (box_2d + mask_rle).

The live scan already computes a 2D SAM box + mask per object; the streaming pipeline used to
null them before export, forcing the dynamics producer to re-derive a seed region. These tests
pin the additive, backward-compatible plumbing that now carries them through:

  * ``server/export/mask_rle.py``            -- the compact RLE codec (round-trips).
  * ``StreamingSLAM._dedup_and_store``       -- keeps box_2d/mask_rle on the exported detection.
  * ``SceneFeatureExtractor._build_objects`` -- copies them onto ``EvidenceRef``.

No GPU / no ``vggt_slam`` weights: the streaming test uses the conftest stub loader (which fakes
``vggt_slam``/torch), the rest is pure. See ``contracts/export-format.md``.
"""

from __future__ import annotations

import threading

import numpy as np

from conftest import load_streaming_slam_module

from server.export.mask_rle import mask_to_rle, rle_area, rle_to_mask
from server.scene_report.features import SceneFeatureExtractor
from export_fakes import FakeMap, FakeSolver


# ---------------------------------------------------------------------------
# (codec) mask_rle round-trips, incl. the awkward starts-with-True + empty cases
# ---------------------------------------------------------------------------

def _roundtrip(mask):
    rle = mask_to_rle(mask)
    assert rle is not None
    back = rle_to_mask(rle)
    assert back is not None
    assert np.array_equal(back, np.asarray(mask, dtype=bool))
    return rle


def test_mask_rle_roundtrips_various_masks():
    rng = np.random.default_rng(0)
    m = rng.random((7, 11)) > 0.5
    rle = _roundtrip(m)
    assert rle["size"] == [7, 11]
    assert rle_area(rle) == int(m.sum())

    # all-False, all-True, and a mask whose very first pixel is set (leading 0-run).
    _roundtrip(np.zeros((4, 5), dtype=bool))
    _roundtrip(np.ones((4, 5), dtype=bool))
    starts_true = np.zeros((3, 3), dtype=bool)
    starts_true[0, 0] = True
    rle = _roundtrip(starts_true)
    assert rle["counts"][0] == 0  # leading empty False run so runs still start with False


def test_mask_rle_rejects_non_masks():
    assert mask_to_rle(None) is None
    assert mask_to_rle(np.zeros((2, 2, 3))) is None  # not 2D
    assert rle_to_mask(None) is None
    assert rle_to_mask({"size": [2, 2], "counts": [1]}) is None  # counts don't fill H*W


# ---------------------------------------------------------------------------
# (a) box_2d + mask_rle survive _dedup_and_store into accumulated_detections
# ---------------------------------------------------------------------------

def _make_dedup_slam(streaming_slam_mod, cache):
    slam = streaming_slam_mod.StreamingSLAM.__new__(streaming_slam_mod.StreamingSLAM)
    slam._detection_lock = threading.Lock()
    slam._sam_cache = cache
    slam.accumulated_detections = []
    slam.get_cached_description = lambda *a, **k: None
    return slam


def test_dedup_and_store_carries_box_and_mask(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)  # stubs vggt_slam + torch
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    cache = {
        (1, 0, "chair"): [
            {
                "bbox_3d": {"center": [0.0, 0.0, 0.0], "extent": [1, 1, 1]},
                "seg_score": 0.9,
                "clip_score": 0.8,
                "box_2d": [10.0, 20.0, 30.0, 40.0],
                "mask_2d": mask,
            }
        ]
    }
    slam = _make_dedup_slam(streaming_slam_mod, cache)
    slam._dedup_and_store()

    assert len(slam.accumulated_detections) == 1
    det = slam.accumulated_detections[0]
    # backward-compatible fields still present + unchanged
    assert det["query"] == "chair"
    assert det["matched_submap"] == 1 and det["matched_frame"] == 0
    assert det["mask_image"] is None
    # NEW: the real 2D detection geometry is now carried through
    assert det["box_2d"] == [10.0, 20.0, 30.0, 40.0]
    assert np.array_equal(rle_to_mask(det["mask_rle"]), mask)


def test_dedup_and_store_mask_gate_off_still_keeps_box(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    cache = {
        (2, 3, "lamp"): [
            {
                "bbox_3d": {"center": [1.0, 0.0, 0.0]},
                "seg_score": 0.5,
                "clip_score": 0.5,
                "box_2d": [1.0, 2.0, 3.0, 4.0],
                "mask_2d": np.ones((3, 3), dtype=bool),
            }
        ]
    }
    slam = _make_dedup_slam(streaming_slam_mod, cache)
    slam.EXPORT_PERSIST_MASK_RLE = False  # operator opts out of per-frame mask payload
    slam._dedup_and_store()
    det = slam.accumulated_detections[0]
    assert det["box_2d"] == [1.0, 2.0, 3.0, 4.0]  # box always carried (tiny, high-value)
    assert det["mask_rle"] is None  # mask suppressed


def test_dedup_and_store_tolerates_missing_geometry(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    cache = {(0, 0, "old"): [{"bbox_3d": {"center": [0, 0, 0]}, "seg_score": 0.7}]}
    slam = _make_dedup_slam(streaming_slam_mod, cache)
    slam._dedup_and_store()  # no box_2d / mask_2d present -> None, not a crash
    det = slam.accumulated_detections[0]
    assert det["box_2d"] is None and det["mask_rle"] is None


# ---------------------------------------------------------------------------
# (a') the geometry flows detection-dict -> EvidenceRef (SceneFeatureExtractor)
# ---------------------------------------------------------------------------

def test_evidence_ref_carries_box_and_mask_from_detection():
    rle = {"size": [4, 4], "counts": [5, 6, 5], "order": "C"}
    det = {
        "query": "laptop",
        "confidence": 0.8,
        "matched_submap": 0,
        "matched_frame": 1,
        "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [0.5, 0.5, 0.5]},
        "box_2d": [10.0, 20.0, 30.0, 40.0],
        "mask_rle": rle,
    }
    facts = SceneFeatureExtractor().extract(
        FakeSolver(FakeMap([])), detections=[det], scene_center=[0, 0, 0]
    )
    ev = facts.objects[0].evidence[0]
    assert ev.submap_id == 0 and ev.frame_idx == 1
    assert ev.box_2d == [10.0, 20.0, 30.0, 40.0]
    assert ev.mask_rle == rle


def test_evidence_ref_defaults_none_when_detection_lacks_geometry():
    det = {
        "query": "chair",
        "confidence": 0.5,
        "matched_submap": 0,
        "matched_frame": 0,
        "bounding_box": {"center": [0.0, 0.0, 0.0], "extent": [1, 1, 1]},
    }
    facts = SceneFeatureExtractor().extract(
        FakeSolver(FakeMap([])), detections=[det], scene_center=[0, 0, 0]
    )
    ev = facts.objects[0].evidence[0]
    assert ev.box_2d is None and ev.mask_rle is None  # additive/optional -> unchanged behavior
