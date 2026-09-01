"""GPU-free tests for the 3DGS splat export (vggt_slam.splat_export).

Covers the deterministic baseline only — PLY serialization round-trip + the
point->gaussian field math on a tiny synthetic cloud. The optional gsplat GPU
refinement is exercised only behind ``pytest.importorskip`` / CUDA guards, so the
whole file runs on a CPU box with neither torch nor gsplat installed.

Run:  pytest core/tests/test_splat_export.py -v
"""

import struct

import numpy as np
import pytest

# conftest.py puts the repo root on sys.path.
from vggt_slam import splat_export as se


# --------------------------------------------------------------------------- #
# field math
# --------------------------------------------------------------------------- #
def test_rgb_sh_dc_roundtrip():
    rgb = np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]], dtype=np.float32)
    f_dc = se.rgb_to_sh_dc(rgb)
    # known mapping: f_dc = (c - 0.5) / C0
    assert np.allclose(f_dc, (rgb - 0.5) / se.SH_C0, atol=1e-6)
    # mid-grey (0.5) maps to exactly 0
    assert np.allclose(se.rgb_to_sh_dc([0.5, 0.5, 0.5]), 0.0, atol=1e-6)
    # inverse recovers the input
    assert np.allclose(se.sh_dc_to_rgb(f_dc), rgb, atol=1e-5)


def test_inverse_sigmoid_is_logit():
    # inverse_sigmoid(0.9) == log(0.9/0.1) == logit(0.9)
    assert np.isclose(float(se.inverse_sigmoid(0.9)), np.log(0.9 / 0.1), atol=1e-6)
    # sigmoid(inverse_sigmoid(x)) == x for a few values
    for x in (0.1, 0.5, 0.9):
        logit = float(se.inverse_sigmoid(x))
        assert np.isclose(1.0 / (1.0 + np.exp(-logit)), x, atol=1e-6)
    # clamped at the extremes (no inf / nan)
    assert np.isfinite(float(se.inverse_sigmoid(0.0)))
    assert np.isfinite(float(se.inverse_sigmoid(1.0)))


def test_estimate_point_scales_unit_grid():
    # This asserts the exact cKDTree nearest-neighbour result. Without scipy,
    # estimate_point_scales deliberately falls back to a bbox-derived global spacing
    # (it "never raises on a CPU box"), which is not the edge length — so skip rather
    # than fail on a minimal install. CI installs scipy, so CI exercises this.
    pytest.importorskip("scipy.spatial")

    # 8 corners of a unit cube -> nearest-neighbour spacing ~= 1.0 (edge length)
    pts = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float32,
    )
    scales = se.estimate_point_scales(pts, k=1)
    assert scales.shape == (8,)
    assert scales.dtype == np.float32
    assert np.all(scales > 0.0)
    assert np.all(np.isfinite(scales))
    # k=1 nearest neighbour of every cube corner is an adjacent corner at distance 1
    assert np.allclose(scales, 1.0, atol=1e-4)


def test_estimate_point_scales_edge_cases():
    assert se.estimate_point_scales(np.zeros((0, 3))).shape == (0,)
    one = se.estimate_point_scales(np.array([[1.0, 2.0, 3.0]]))
    assert one.shape == (1,) and np.isfinite(one[0]) and one[0] > 0.0


# --------------------------------------------------------------------------- #
# p99 scale-tail clamp (EXP-7): pulls down the spike tail, near-no-op otherwise
# --------------------------------------------------------------------------- #
def test_clamp_scales_percentile_pulls_down_tail():
    # 99 tame scales at 0.01 + one giant spike at 100 -> p99 clamp removes the spike.
    scales = np.full((100, 3), 0.01, dtype=np.float32)
    scales[0] = 100.0  # a spike on all three axes
    clamped, info = se.clamp_scales_percentile(scales, percentile=99.0)
    assert info["num_clamped"] == 1          # exactly the spike row
    assert info["clamp_value"] is not None
    assert float(clamped.max()) <= info["clamp_value"] + 1e-6
    assert np.isclose(float(clamped.max()), info["clamp_value"], rtol=1e-5)  # spike -> p99
    assert float(clamped.max()) < 2.0        # the 100.0 spike is gone (p99 ~= 1.0)
    # the tame rows are untouched
    assert np.allclose(clamped[1:], 0.01, atol=1e-6)


def test_clamp_scales_percentile_noop_on_uniform_and_edges():
    # uniform scales: p99 ~= max, nothing pulled down
    uniform = np.full((50, 3), 0.05, dtype=np.float32)
    clamped, info = se.clamp_scales_percentile(uniform, percentile=99.0)
    assert info["num_clamped"] == 0
    assert np.allclose(clamped, uniform, atol=1e-7)
    # empty input passes through
    empty, einfo = se.clamp_scales_percentile(np.zeros((0, 3), dtype=np.float32))
    assert empty.shape == (0, 3) and einfo["num_clamped"] == 0
    # percentile >= 100 (or None) disables the clamp entirely
    spiky = np.array([[0.01, 0.01, 0.01], [9.0, 9.0, 9.0]], dtype=np.float32)
    for p in (100.0, None):
        out, oinfo = se.clamp_scales_percentile(spiky, percentile=p)
        assert oinfo["num_clamped"] == 0 and np.allclose(out, spiky)


def test_write_splat_ply_clamp_default_on_vs_off(tmp_path):
    # A cloud with one giant per-row scale spike; default-ON clamp must pull it down,
    # clamp_scales=False must store it verbatim.
    positions = np.random.RandomState(0).randn(200, 3).astype(np.float32)
    colors = np.full((200, 3), 0.5, dtype=np.float32)
    scales = np.full((200,), 0.02, dtype=np.float32)
    scales[0] = 50.0  # spike

    on = tmp_path / "on.ply"
    se.write_splat_ply(str(on), positions, colors, scales=scales)  # clamp default ON
    got_on = se.read_splat_ply(str(on))
    assert float(np.exp(got_on["scale_0"]).max()) < 50.0  # spike clamped away

    off = tmp_path / "off.ply"
    se.write_splat_ply(str(off), positions, colors, scales=scales, clamp_scales=False)
    got_off = se.read_splat_ply(str(off))
    assert np.isclose(float(np.exp(got_off["scale_0"]).max()), 50.0, rtol=1e-4)  # verbatim


# --------------------------------------------------------------------------- #
# PLY serialization round-trip
# --------------------------------------------------------------------------- #
def _tiny_cloud():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    return positions, colors


def test_write_splat_ply_header_and_fields(tmp_path):
    positions, colors = _tiny_cloud()
    out = tmp_path / "splat.ply"
    n = se.write_splat_ply(
        str(out), positions, colors, scales=0.01, opacities=se.DEFAULT_OPACITY
    )
    assert n == 4

    raw = out.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")

    assert header.startswith("ply\n")
    assert "format binary_little_endian 1.0\n" in header
    assert "element vertex 4\n" in header
    for name in se.SPLAT_PLY_FIELDS:
        assert f"property float {name}\n" in header
    # exactly the 17 canonical 3DGS fields
    assert len(se.SPLAT_PLY_FIELDS) == 17
    assert header.count("property float ") == 17

    # binary payload is exactly N * 17 * 4 bytes
    payload = raw[header_end:]
    assert len(payload) == 4 * 17 * 4

    # first vertex's first float is x of point 0 == 0.0, little-endian float32
    first_x = struct.unpack("<f", payload[:4])[0]
    assert np.isclose(first_x, 0.0)


def test_write_splat_ply_roundtrip_values(tmp_path):
    positions, colors = _tiny_cloud()
    out = tmp_path / "splat.ply"
    scale = 0.05
    opacity = se.DEFAULT_OPACITY
    se.write_splat_ply(str(out), positions, colors, scales=scale, opacities=opacity)

    got = se.read_splat_ply(str(out))

    # positions preserved
    assert np.allclose(np.stack([got["x"], got["y"], got["z"]], axis=1), positions, atol=1e-6)
    # normals default to zero
    assert np.allclose(np.stack([got["nx"], got["ny"], got["nz"]], axis=1), 0.0)
    # colors stored as SH DC term
    f_dc = np.stack([got["f_dc_0"], got["f_dc_1"], got["f_dc_2"]], axis=1)
    assert np.allclose(f_dc, se.rgb_to_sh_dc(colors), atol=1e-5)
    assert np.allclose(se.sh_dc_to_rgb(f_dc), colors, atol=1e-4)
    # opacity stored as logit of 0.9
    assert np.allclose(got["opacity"], se.inverse_sigmoid(opacity), atol=1e-5)
    # scale stored as log(scale) on all three axes (isotropic)
    for k in ("scale_0", "scale_1", "scale_2"):
        assert np.allclose(got[k], np.log(scale), atol=1e-5)
    # rotation is the identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    assert np.allclose(got["rot_0"], 1.0, atol=1e-6)
    assert np.allclose(np.stack([got["rot_1"], got["rot_2"], got["rot_3"]], axis=1), 0.0)


def test_colors_uint8_autonormalized(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    colors_255 = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.float32)
    out = tmp_path / "splat.ply"
    se.write_splat_ply(str(out), positions, colors_255, scales=0.01)
    got = se.read_splat_ply(str(out))
    f_dc = np.stack([got["f_dc_0"], got["f_dc_1"], got["f_dc_2"]], axis=1)
    assert np.allclose(se.sh_dc_to_rgb(f_dc), colors_255 / 255.0, atol=1e-4)


def test_quaternions_normalized(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    quats = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # non-unit
    out = tmp_path / "splat.ply"
    se.write_splat_ply(str(out), positions, colors, scales=0.01, quats=quats)
    got = se.read_splat_ply(str(out))
    q = np.array([got["rot_0"][0], got["rot_1"][0], got["rot_2"][0], got["rot_3"][0]])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_length_mismatch_raises(tmp_path):
    positions = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        se.write_splat_ply(str(tmp_path / "x.ply"), positions, colors)


def test_point_cloud_to_splat_ply_estimates_scale(tmp_path):
    # 8 corners of a unit cube; no explicit scale -> estimated from spacing (~1.0)
    pts = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float32,
    )
    cols = np.tile([0.3, 0.6, 0.9], (8, 1)).astype(np.float32)
    out = tmp_path / "splat.ply"
    n = se.point_cloud_to_splat_ply(pts, cols, str(out))
    assert n == 8
    got = se.read_splat_ply(str(out))
    # recovered linear scales are positive and finite
    recovered = np.exp(got["scale_0"])
    assert np.all(recovered > 0.0) and np.all(np.isfinite(recovered))
    assert np.allclose(recovered, 1.0, atol=0.25)


# --------------------------------------------------------------------------- #
# integration against a fake solver/map (CPU, deterministic) + graceful fallback
# --------------------------------------------------------------------------- #
class _FakeSubmap:
    def __init__(self, points, colors, lc=False, frames=None):
        self._points = points
        self._colors = colors
        self._lc = lc
        self._frames = frames

    def get_points_in_world_frame(self, graph):
        return self._points

    def get_points_colors(self):
        return self._colors

    def get_lc_status(self):
        return self._lc

    def get_all_frames(self):
        return self._frames


class _FakeMap:
    def __init__(self, submaps):
        self._submaps = submaps

    def ordered_submaps_by_key(self):
        return iter(self._submaps)


class _FakeSolver:
    def __init__(self, submaps):
        self.map = _FakeMap(submaps)
        self.graph = object()


def test_gather_world_cloud_concatenates_submaps():
    s0 = _FakeSubmap(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32),
    )
    s1 = _FakeSubmap(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0, 0, 255]], dtype=np.float32),
    )
    pos, col = se.gather_world_cloud(_FakeMap([s0, s1]), graph=object())
    assert pos.shape == (3, 3)
    assert col.shape == (3, 3)
    # uint8-range colors auto-normalized to [0, 1]
    assert float(col.max()) <= 1.0
    # ``_FakeSubmap`` has no per-frame accessor, so the new overlap-dedup logic
    # gracefully no-ops (falls back to counting every point) -- this pre-existing
    # test's totals are unaffected by the dedup fix below.


# --------------------------------------------------------------------------- #
# gather_world_cloud dedup: exclude LC submaps + overlap-frame duplicates
# (mirrors the server-side fix, server/tests/test_gather_dedup.py, platform repo)
# --------------------------------------------------------------------------- #
class _FrameSubmap:
    """Duck-typed stand-in for ``vggt_slam.submap.Submap`` with real per-frame
    semantics, needed to exercise the overlap-frame dedup in
    ``gather_world_cloud`` (``_FakeSubmap`` above only stores a pre-flattened
    cloud, so it can't express "submap1's frame 0 == submap0's last frame").

    ``frames_points``/``frames_colors`` are one entry per LOCAL frame index,
    already confidence-filtered (every point kept). ``get_points_in_world_frame``/
    ``get_points_colors`` vstack them frame-major (frame 0 first), exactly like
    the real ``Submap``, so the leading rows are always frame 0's, then frame 1's.
    """

    def __init__(self, frames_points, frames_colors=None, lc=False):
        self._lc = lc
        self._frames_points = [
            np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in frames_points
        ]
        if frames_colors is None:
            frames_colors = [
                np.tile(np.array([100, 150, 200], dtype=np.uint8), (len(p), 1))
                for p in self._frames_points
            ]
        self._frames_colors = [np.asarray(c) for c in frames_colors]

    def get_lc_status(self):
        return self._lc

    def get_points_in_world_frame(self, graph):
        if not self._frames_points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(self._frames_points)

    def get_points_colors(self):
        if not self._frames_colors:
            return np.zeros((0, 3))
        return np.vstack(self._frames_colors)

    def get_points_list_in_world_frame(self, graph):
        masks = [np.ones(len(p), dtype=bool) for p in self._frames_points]
        frame_ids = list(range(len(self._frames_points)))
        return list(self._frames_points), frame_ids, masks


def _pts(n, base=0.0):
    """``n`` distinct 3D points so vstack/slice bugs show up as wrong values,
    not just counts."""
    return (np.arange(n, dtype=np.float64).reshape(-1, 1) + base) * np.ones((1, 3))


def test_gather_world_cloud_single_submap_keeps_everything():
    """A single (first, and only) regular submap has no predecessor to
    duplicate -- none of its points should be dropped."""
    sm = _FrameSubmap(frames_points=[_pts(3), _pts(2, base=100)])
    positions, colors = se.gather_world_cloud(_FakeMap([sm]), graph=object(), overlap_frames=1)
    assert positions.shape[0] == 5
    assert colors.shape[0] == 5


def test_gather_world_cloud_excludes_lc_submaps():
    """LC re-observation submaps must contribute zero points to the fused cloud."""
    regular = _FrameSubmap(frames_points=[_pts(4)])
    lc = _FrameSubmap(frames_points=[_pts(2, base=500), _pts(2, base=600)], lc=True)
    positions, _colors = se.gather_world_cloud(
        _FakeMap([regular, lc]), graph=object(), overlap_frames=1
    )
    assert positions.shape[0] == 4  # LC's 4 points excluded entirely
    assert not np.any(positions >= 500)  # sanity: none of the LC-only values leaked in


def test_gather_world_cloud_skips_later_submaps_overlap_frame():
    """submap1's frame 0 duplicates submap0's last frame (the carried-over
    overlap frame) -- it must be counted once, not twice."""
    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)  # overlap frame: submap0's last == submap1's first
    frame_c = _pts(4, base=200)

    submap0 = _FrameSubmap(frames_points=[frame_a, frame_b_shared])
    submap1 = _FrameSubmap(frames_points=[frame_b_shared, frame_c])

    positions, colors = se.gather_world_cloud(
        _FakeMap([submap0, submap1]), graph=object(), overlap_frames=1
    )

    # 3 (frame_a) + 2 (frame_b, once) + 4 (frame_c) = 9, NOT 11.
    assert positions.shape[0] == 9
    assert colors.shape[0] == 9
    # The shared frame's points appear exactly once (from submap0), not twice.
    shared_row = frame_b_shared[0]
    assert int(np.sum(np.all(positions == shared_row, axis=1))) == 1


def test_gather_world_cloud_many_submaps_chain_and_lc_mixed():
    """A longer chain (mirrors a real multi-window scan) with an LC submap
    interleaved: total unique points = sum of each submap's own frames minus the
    shared overlap frames, with the LC submap contributing nothing."""
    f0 = [_pts(3, base=0), _pts(2, base=50)]  # submap0: 5 points, last frame shared
    f1 = [f0[-1], _pts(2, base=60), _pts(2, base=70)]  # first frame shared with submap0
    f2 = [f1[-1], _pts(3, base=80)]  # first frame shared with submap1

    submap0 = _FrameSubmap(frames_points=f0)
    submap1 = _FrameSubmap(frames_points=f1)
    submap2 = _FrameSubmap(frames_points=f2)
    lc = _FrameSubmap(frames_points=[_pts(2, base=900), _pts(2, base=910)], lc=True)

    positions, colors = se.gather_world_cloud(
        _FakeMap([submap0, lc, submap1, submap2]), graph=object(), overlap_frames=1
    )
    # unique: submap0 (3+2) + submap1 (2+2, skip its shared first frame) + submap2
    # (3, skip its shared first frame) = 5 + 4 + 3 = 12
    assert positions.shape[0] == 12
    assert colors.shape[0] == 12


def test_gather_world_cloud_overlap_frames_zero_disables_skip():
    """``overlap_frames=0`` is an explicit opt-out -- every frame counts (LC
    exclusion still applies), matching the pre-fix behavior for callers that pass
    it explicitly."""
    frame_a = _pts(3, base=0)
    frame_b_shared = _pts(2, base=100)
    frame_c = _pts(4, base=200)
    submap0 = _FrameSubmap(frames_points=[frame_a, frame_b_shared])
    submap1 = _FrameSubmap(frames_points=[frame_b_shared, frame_c])

    positions, _colors = se.gather_world_cloud(
        _FakeMap([submap0, submap1]), graph=object(), overlap_frames=0
    )
    assert positions.shape[0] == 11  # 3 + 2 + 2 + 4, shared frame double-counted


def test_gather_world_cloud_all_lc_returns_empty():
    lc = _FrameSubmap(frames_points=[_pts(3)], lc=True)
    positions, _colors = se.gather_world_cloud(_FakeMap([lc]), graph=object(), overlap_frames=1)
    assert positions.shape[0] == 0


def test_export_splat_writes_baseline(tmp_path):
    s0 = _FakeSubmap(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    solver = _FakeSolver([s0])
    out = tmp_path / "nested" / "splat.ply"  # parent dir must be created
    result = se.export_splat(solver, str(out), refine=False)
    assert result == str(out)
    assert out.exists()
    got = se.read_splat_ply(str(out))
    assert got["x"].shape == (3,)


def test_export_splat_refine_falls_back_to_baseline(tmp_path):
    # refine=True must NEVER break the baseline deliverable: with no GPU/gsplat the
    # guarded refinement raises and export_splat keeps the baseline file.
    s0 = _FakeSubmap(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], dtype=np.float32),
    )
    solver = _FakeSolver([s0])
    out = tmp_path / "splat.ply"
    result = se.export_splat(solver, str(out), refine=True)
    assert result == str(out)
    assert out.exists()
    got = se.read_splat_ply(str(out))
    assert got["x"].shape == (2,)


def test_export_splat_empty_cloud_returns_none(tmp_path):
    solver = _FakeSolver([])  # no submaps -> nothing to export
    out = tmp_path / "splat.ply"
    assert se.export_splat(solver, str(out), refine=False) is None
    assert not out.exists()


# --------------------------------------------------------------------------- #
# optional GPU refinement — guarded, only runs where torch+gsplat+CUDA exist
# --------------------------------------------------------------------------- #
def test_refine_with_gsplat_guarded():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device for gsplat refinement")
    # We only assert the function is importable/guarded here; a full photometric
    # optimization needs real posed keyframes and is not run in CI.
    assert callable(se.refine_with_gsplat)


# --------------------------------------------------------------------------- #
# non-finite filtering + color robustness (red-team hardening)
# --------------------------------------------------------------------------- #
def test_write_splat_ply_drops_nonfinite_rows(tmp_path):
    # 4 points, 2 poisoned (NaN position, Inf color) → only 2 finite rows written.
    positions = np.array(
        [[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.inf, 0.0, 0.0], [0.2, 0.4, 0.6]],
        dtype=np.float32,
    )
    out = tmp_path / "splat.ply"
    n = se.write_splat_ply(str(out), positions, colors, scales=0.01)
    assert n == 2
    got = se.read_splat_ply(str(out))
    assert got["x"].shape == (2,)
    # every written value is finite
    for k in se.SPLAT_PLY_FIELDS:
        assert np.all(np.isfinite(got[k]))
    # the surviving positions are the two finite ones
    xyz = np.stack([got["x"], got["y"], got["z"]], axis=1)
    assert np.allclose(xyz, [[0, 0, 0], [2, 2, 2]], atol=1e-6)


def test_write_splat_ply_filters_nonfinite_perrow_scales_quats(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    colors = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], dtype=np.float32)
    scales = np.array([0.01, np.nan], dtype=np.float32)  # 2nd row poisoned
    n = se.write_splat_ply(str(tmp_path / "x.ply"), positions, colors, scales=scales)
    assert n == 1  # the NaN-scale row dropped


def test_export_splat_all_nonfinite_returns_none(tmp_path):
    s0 = _FakeSubmap(
        np.array([[np.nan, np.nan, np.nan], [np.inf, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    out = tmp_path / "splat.ply"
    assert se.export_splat(_FakeSolver([s0]), str(out), refine=False) is None
    assert not out.exists()  # no empty splat left behind


def test_normalized_floats_with_stray_over_one_not_crushed(tmp_path):
    # A normalized float cloud whose max is 1.001 must NOT be divided by 255.
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[0.8, 0.6, 1.001], [0.5, 0.5, 0.5]], dtype=np.float32)
    out = tmp_path / "x.ply"
    se.write_splat_ply(str(out), positions, colors, scales=0.01)
    got = se.read_splat_ply(str(out))
    rgb = se.sh_dc_to_rgb(np.stack([got["f_dc_0"], got["f_dc_1"], got["f_dc_2"]], axis=1))
    # clipped to 1.0, NOT crushed to ~0.004 (which a /255 misfire would produce)
    assert rgb[0, 2] > 0.9
    assert np.allclose(rgb[1], [0.5, 0.5, 0.5], atol=1e-3)
