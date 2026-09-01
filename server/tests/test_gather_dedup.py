"""Coverage for the fused-cloud dedup fix (window-quality study,
``experiments/research/2026-07-03-omega-full-quality.md`` §2/§6 in the platform repo).

``server.export.clouds.gather_full_cloud`` (shared by ``StreamingSLAM.gather_world_point_cloud``
and the offline dataset-export path) used to walk every submap unconditionally, which
double-counted two kinds of points:

- loop-closure (LC) re-observation submaps, which re-add points already covered by their
  target submap;
- the shared overlap frame(s) at the start of every non-first "regular" submap, which
  duplicate the last ``overlap_frames`` frame(s) of the immediately preceding submap (the
  streaming design carries these frames forward to compute the SL(4) junction homography).

Fake submaps here mirror the real ``vggt_slam.submap.Submap`` per-frame semantics closely
enough to unit-test the exclusion logic CPU-only, with no ``vggt_slam``/GPU dependency (mirrors
the fake-submap style in ``tests/export_fakes.py`` / ``tests/test_scene_features.py``, but
per-frame-list based so a submap's total point count can be decomposed frame-by-frame).
"""

from __future__ import annotations

import numpy as np

from server.export.clouds import gather_full_cloud


class FakeSubmap:
    """Duck-typed stand-in for ``vggt_slam.submap.Submap``.

    ``frames_points``/``frames_colors`` are one entry per LOCAL frame index -- already
    "confidence-filtered" (every point kept), matching the shape the real per-frame
    accessors return post-filter. ``get_points_in_world_frame``/``get_points_colors``
    vstack them frame-major (index 0 first), exactly like the real ``Submap``, so the leading
    rows of the returned arrays are always frame 0's, then frame 1's, etc.
    """

    def __init__(self, sid, frames_points, frames_colors=None, lc=False):
        self._id = sid
        self._lc = lc
        self._frames_points = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in frames_points]
        if frames_colors is None:
            frames_colors = [
                np.tile(np.array([sid % 256, i % 256, 0], dtype=np.uint8), (len(p), 1))
                for i, p in enumerate(self._frames_points)
            ]
        self._frames_colors = [np.asarray(c) for c in frames_colors]

    def get_id(self):
        return self._id

    def get_lc_status(self):
        return self._lc

    def get_points_in_world_frame(self, graph):
        if not self._frames_points:
            return np.zeros((0, 3))
        return np.vstack(self._frames_points)

    def get_points_colors(self):
        if not self._frames_colors:
            return np.zeros((0, 3))
        return np.vstack(self._frames_colors)

    def get_points_list_in_world_frame(self, graph):
        masks = [np.ones(len(p), dtype=bool) for p in self._frames_points]
        frame_ids = list(range(len(self._frames_points)))
        return list(self._frames_points), frame_ids, masks


class FakeMap:
    def __init__(self, submaps):
        self._submaps = list(submaps)

    def get_submaps(self):
        return list(self._submaps)


class FakeSolver:
    def __init__(self, submaps):
        self.map = FakeMap(submaps)
        self.graph = object()


def _pts(n, base=0.0):
    """``n`` distinct 3D points so vstack/slice bugs show up as wrong values, not just counts."""
    return (np.arange(n, dtype=np.float64).reshape(-1, 1) + base) * np.ones((1, 3))


def test_gather_full_cloud_single_submap_keeps_everything():
    """A single (first, and only) regular submap has no predecessor to duplicate — none of
    its points should be dropped even though it's technically "non-first-eligible-for-skip"."""
    sm = FakeSubmap(0, frames_points=[_pts(3), _pts(2, base=100)])
    positions, colors = gather_full_cloud(FakeSolver([sm]), overlap_frames=1)
    assert positions.shape[0] == 5
    assert colors.shape[0] == 5


def test_gather_full_cloud_excludes_lc_submaps():
    """LC re-observation submaps must contribute zero points to the fused cloud."""
    regular = FakeSubmap(0, frames_points=[_pts(4)])
    lc = FakeSubmap(9, frames_points=[_pts(2, base=500), _pts(2, base=600)], lc=True)
    positions, _colors = gather_full_cloud(FakeSolver([regular, lc]), overlap_frames=1)
    assert positions.shape[0] == 4  # LC's 4 points excluded entirely
    assert not np.any(positions >= 500)  # sanity: none of the LC-only values leaked in


def test_gather_full_cloud_skips_later_submaps_overlap_frame():
    """submap1's frame 0 duplicates submap0's last frame (the carried-over overlap frame) --
    it must be counted once, not twice."""
    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)  # the overlap frame: submap0's last == submap1's first
    frame_c = _pts(4, base=200)

    submap0 = FakeSubmap(0, frames_points=[frame_a, frame_b_shared])
    submap1 = FakeSubmap(5, frames_points=[frame_b_shared, frame_c])  # id > submap0's -> "later"

    positions, colors = gather_full_cloud(FakeSolver([submap0, submap1]), overlap_frames=1)

    # 3 (frame_a) + 2 (frame_b, once) + 4 (frame_c) = 9, NOT 11.
    assert positions.shape[0] == 9
    assert colors.shape[0] == 9
    # The shared frame's points appear exactly once (from submap0), not twice.
    shared_row = frame_b_shared[0]
    assert int(np.sum(np.all(positions == shared_row, axis=1))) == 1


def test_gather_full_cloud_respects_submap_order_not_dict_order():
    """Exclusion must key off submap ID order (the chain order), not iteration order."""
    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)
    frame_c = _pts(4, base=200)
    submap0 = FakeSubmap(0, frames_points=[frame_a, frame_b_shared])
    submap1 = FakeSubmap(5, frames_points=[frame_b_shared, frame_c])

    # Feed the map in reverse order -- the fix must still treat submap0 (min id) as "first".
    positions, _colors = gather_full_cloud(FakeSolver([submap1, submap0]), overlap_frames=1)
    assert positions.shape[0] == 9


def test_gather_full_cloud_many_submaps_chain_and_lc_mixed():
    """A longer chain (mirrors a real multi-window scan) with an LC submap interleaved:
    total unique points = sum of each submap's own frames minus the shared overlap frames,
    with the LC submap contributing nothing."""
    f0 = [_pts(3, base=0), _pts(2, base=50)]  # submap0: 5 points, last frame (2) is shared
    f1 = [f0[-1], _pts(2, base=60), _pts(2, base=70)]  # submap1: first frame shared with submap0
    f2 = [f1[-1], _pts(3, base=80)]  # submap2: first frame shared with submap1

    submap0 = FakeSubmap(0, frames_points=f0)
    submap1 = FakeSubmap(5, frames_points=f1)
    submap2 = FakeSubmap(10, frames_points=f2)
    lc = FakeSubmap(12, frames_points=[_pts(2, base=900), _pts(2, base=910)], lc=True)

    positions, colors = gather_full_cloud(
        FakeSolver([submap0, lc, submap1, submap2]), overlap_frames=1
    )
    # unique: submap0 (3+2) + submap1 (2+2, skip its shared first frame) + submap2 (3, skip
    # its shared first frame) = 5 + 4 + 3 = 12
    assert positions.shape[0] == 12
    assert colors.shape[0] == 12


def test_gather_full_cloud_overlap_frames_zero_disables_skip():
    """``overlap_frames=0`` is an explicit opt-out -- every frame counts (LC exclusion still
    applies), matching the pre-fix behavior for callers that pass it explicitly."""
    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)
    frame_c = _pts(4, base=200)
    submap0 = FakeSubmap(0, frames_points=[frame_a, frame_b_shared])
    submap1 = FakeSubmap(5, frames_points=[frame_b_shared, frame_c])

    positions, _colors = gather_full_cloud(FakeSolver([submap0, submap1]), overlap_frames=0)
    assert positions.shape[0] == 11  # 3 + 2 + 2 + 4, shared frame double-counted


def test_gather_full_cloud_empty_map_returns_empty():
    positions, colors = gather_full_cloud(FakeSolver([]), overlap_frames=1)
    assert positions.shape == (0, 3)
    assert colors.shape == (0, 3)


def test_gather_full_cloud_all_lc_returns_empty():
    lc = FakeSubmap(0, frames_points=[_pts(3)], lc=True)
    positions, _colors = gather_full_cloud(FakeSolver([lc]), overlap_frames=1)
    assert positions.shape[0] == 0


# -- StreamingSLAM.gather_world_point_cloud delegates to gather_full_cloud -------------------


def test_streaming_slam_gather_world_point_cloud_delegates(monkeypatch):
    from conftest import load_streaming_slam_module

    streaming_slam_mod = load_streaming_slam_module(monkeypatch)

    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)
    frame_c = _pts(4, base=200)
    submap0 = FakeSubmap(0, frames_points=[frame_a, frame_b_shared])
    submap1 = FakeSubmap(5, frames_points=[frame_b_shared, frame_c])
    lc = FakeSubmap(7, frames_points=[_pts(2, base=900)], lc=True)

    slam = streaming_slam_mod.StreamingSLAM.__new__(streaming_slam_mod.StreamingSLAM)
    slam.solver = FakeSolver([submap0, lc, submap1])
    slam.overlapping_window_size = 1

    positions, colors = slam.gather_world_point_cloud()
    assert positions.shape[0] == 9  # same as the direct gather_full_cloud test above
    assert colors.shape[0] == 9
