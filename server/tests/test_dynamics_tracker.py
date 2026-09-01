"""CPU unit tests for the BootsTAPIR tracker producer (GPU-free, no torch, no modal).

Covers the two pure pieces of ``server/export/tracker_bootstapir.py``:

  (a) mask -> query-point seeding: sampled points always land INSIDE the mask, respect ``k``,
      and are deterministic.
  (b) per-point-tracks -> ``objects_2d`` assembly: centroid = median of visible points (with
      the all-points fallback), depth = median depth at those points, visibility bool via the
      fraction threshold, confidence = the fraction.

Then an end-to-end shape check: the assembled ``objects_2d`` feeds straight into the
ALREADY-BUILT ``build_object_tracks`` + ``write_object_tracks_jsonl`` and yields valid
``ObjectTrack`` jsonl — proving the producer's output shape matches the exporter's input.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from server.export.dynamics import build_object_tracks, write_object_tracks_jsonl
from server.export.tracker_bootstapir import (
    assemble_objects_2d,
    sample_query_points_in_mask,
)
from server.scene_report.schemas import ObjectTrack


# ---------------------------------------------------------------------------
# (a) mask -> query-point seeding
# ---------------------------------------------------------------------------

def _box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_seed_points_land_inside_mask():
    mask = _box_mask(100, 120, 30, 70, 40, 90)  # 40x50 = 2000 interior px
    pts = sample_query_points_in_mask(mask, 32, np.random.default_rng(0))
    assert pts.shape == (32, 2)
    for x, y in pts:
        # [x, y] = [col, row]; every point must sit on a True mask pixel
        assert mask[int(round(y)), int(round(x))], f"point ({x},{y}) outside mask"


def test_seed_respects_available_count_and_dtype():
    # only 6 interior pixels; asking for 50 returns exactly those 6 (all inside)
    mask = _box_mask(20, 20, 5, 7, 8, 11)  # 2 rows x 3 cols = 6 px
    pts = sample_query_points_in_mask(mask, 50, np.random.default_rng(1))
    assert pts.shape == (6, 2)
    assert pts.dtype == np.float32
    seen = {(int(x), int(y)) for x, y in pts}
    assert len(seen) == 6  # all distinct interior pixels
    for x, y in pts:
        assert mask[int(y), int(x)]


def test_seed_empty_mask_and_deterministic():
    assert sample_query_points_in_mask(np.zeros((10, 10), bool), 8).shape == (0, 2)
    assert sample_query_points_in_mask(np.ones((10, 10), bool), 0).shape == (0, 2)
    mask = _box_mask(64, 64, 10, 50, 10, 50)
    a = sample_query_points_in_mask(mask, 16, np.random.default_rng(7))
    b = sample_query_points_in_mask(mask, 16, np.random.default_rng(7))
    assert np.array_equal(a, b)  # same seed -> identical sample


# ---------------------------------------------------------------------------
# (b) per-point tracks -> objects_2d assembly (FAKE tracks; no model)
# ---------------------------------------------------------------------------

def _fake_two_object_tracks():
    """T=2 frames, N=4 points, 2 objects (2 points each). Depth = row index (y)."""
    # object A: columns 0..2 ; object B: columns 2..4
    tracks = np.zeros((2, 4, 2), dtype=np.float64)
    # --- frame 0 ---
    tracks[0, 0] = (10, 10)   # A p0
    tracks[0, 1] = (20, 20)   # A p1   -> A median (15,15)
    tracks[0, 2] = (100, 50)  # B p0
    tracks[0, 3] = (100, 70)  # B p1   -> B median (100,60)
    # --- frame 1 ---
    tracks[1, 0] = (12, 10)   # A p0 (visible)
    tracks[1, 1] = (22, 20)   # A p1 (occluded) -> A uses visible only -> (12,10)
    tracks[1, 2] = (110, 50)  # B p0 (occluded)
    tracks[1, 3] = (110, 70)  # B p1 (occluded) -> B none visible -> all -> (110,60)
    visibles = np.array(
        [
            [True, True, True, True],    # frame 0: all visible
            [True, False, False, False],  # frame 1: only A p0 visible
        ],
        dtype=bool,
    )
    depth = np.tile(np.arange(200, dtype=np.float64)[:, None], (1, 200))  # depth[y,x] = y
    return tracks, visibles, depth


def test_assemble_centroid_depth_visibility_confidence():
    tracks, visibles, depth = _fake_two_object_tracks()
    objs = assemble_objects_2d(
        object_ids=[7, 9],
        queries=["cat", "dog"],
        frame_indices=[0, 1],
        point_ranges=[(0, 2), (2, 4)],
        tracks=tracks,
        visibles=visibles,
        depth_maps=depth,  # single (H,W) -> broadcast to both frames
        min_visible_frac=0.5,
    )
    assert [o["id"] for o in objs] == [7, 9]
    a, b = objs

    # object A (id 7)
    assert a["query"] == "cat"
    assert a["frames"] == [0, 1]
    assert a["centroids"][0] == pytest.approx((15.0, 15.0))  # median of both visible pts
    assert a["centroids"][1] == pytest.approx((12.0, 10.0))  # only p0 visible -> its coord
    # depth = median of depth[y,x] at the (visible) points; depth == y
    assert a["depths"][0] == pytest.approx(15.0)  # median(10,20)
    assert a["depths"][1] == pytest.approx(10.0)  # only p0 (y=10)
    assert a["visibility"] == [True, True]        # frac 1.0 and 0.5 both >= 0.5
    assert a["confidence"] == pytest.approx([1.0, 0.5])

    # object B (id 9): frame 1 has NO visible points -> falls back to all points
    assert b["centroids"][0] == pytest.approx((100.0, 60.0))
    assert b["centroids"][1] == pytest.approx((110.0, 60.0))  # median of both (occluded) pts
    assert b["visibility"] == [True, False]       # frac 1.0 then 0.0
    assert b["confidence"] == pytest.approx([1.0, 0.0])


def test_assemble_per_frame_depth_list():
    """A list of per-frame depth maps is honoured frame-by-frame."""
    tracks, visibles, _ = _fake_two_object_tracks()
    d0 = np.full((200, 200), 4.0)
    d1 = np.full((200, 200), 9.0)
    objs = assemble_objects_2d(
        [0], ["x"], [0, 1], [(0, 2)], tracks, visibles, [d0, d1]
    )
    assert objs[0]["depths"] == pytest.approx([4.0, 9.0])


# ---------------------------------------------------------------------------
# end-to-end: assembled objects_2d -> build_object_tracks -> jsonl (shape match)
# ---------------------------------------------------------------------------

def test_objects_2d_feeds_build_object_tracks(tmp_path):
    tracks, visibles, depth = _fake_two_object_tracks()
    objects_2d = assemble_objects_2d(
        [7, 9], ["cat", "dog"], [0, 1], [(0, 2), (2, 4)], tracks, visibles, depth
    )

    # synthetic poses (identity) + intrinsics keyed by the plain global frame index;
    # index_map empty because objects_2d["frames"] are already global ints.
    K = [200.0, 200.0, 100.0, 100.0]
    poses = {0: np.eye(4), 1: np.eye(4)}
    intr = {0: K, 1: K}

    result = build_object_tracks(objects_2d, poses, intr, {})
    assert len(result) == 2
    assert all(isinstance(t, ObjectTrack) for t in result)
    by_id = {t.object_id: t for t in result}
    assert by_id[7].query == "cat"
    assert by_id[7].frames == [0, 1]
    assert len(by_id[7].frame_centers) == 2
    assert all(np.all(np.isfinite(c)) for c in by_id[7].frame_centers)

    n = write_object_tracks_jsonl(result, str(tmp_path))
    assert n == 2
    out = os.path.join(str(tmp_path), "dynamics", "objects_tracks.jsonl")
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) == 2
    for rec in recs:
        for key in (
            "query", "object_id", "center", "frames", "frame_centers",
            "frame_visibility", "frame_confidence", "motion", "max_displacement",
        ):
            assert key in rec, f"missing ObjectTrack field {key}"
