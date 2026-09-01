"""CPU-only tests for the EXP-12 metric scale anchor (vggt_slam.metric_anchor).

Covers the pure-numpy ratio + 4-tier sanity-gate math and the per-submap extraction path
(``compute_submap_ratio``) with an injected depth function — so no torch, cv2, huggingface_hub,
or Depth-Anything-V2 weights are needed. The real GPU model load is lazy and NOT exercised here
(same posture as test_hygiene.py); it is verified on Modal (see the research doc).

Run:  pytest core/tests/test_metric_anchor.py -v
"""

import numpy as np
import pytest

# conftest.py puts the repo root on sys.path.
from vggt_slam import metric_anchor as ma


# ---------------------------------------------------------------------------
# frame_ratio — tier-1 hygiene + robust per-frame ratio
# ---------------------------------------------------------------------------
def test_frame_ratio_exact_on_clean_input():
    rng = np.random.RandomState(0)
    slam = rng.uniform(0.5, 3.0, (100, 100)).astype(np.float64)
    R = 2.5
    metric = R * slam
    valid = np.ones_like(slam, bool)
    fr = ma.frame_ratio(slam, valid, metric, max_depth=20.0, min_valid=200)
    assert fr is not None
    assert abs(fr["ratio"] - R) < 1e-9
    assert fr["n"] == slam.size
    assert fr["rel_iqr"] < 1e-9  # constant ratio -> zero spread


def test_frame_ratio_drops_capped_and_invalid_pixels():
    slam = np.full((50, 50), 1.0)
    metric = np.full((50, 50), 5.0)
    # push a block to the depth cap (>= 0.98*20 = 19.6) — must be dropped
    metric[:10, :] = 19.9
    # mark a block low-confidence (invalid) — must be dropped
    valid = np.ones_like(slam, bool)
    valid[40:, :] = False
    fr = ma.frame_ratio(slam, valid, metric, max_depth=20.0, min_valid=100)
    assert fr is not None
    # only the clean rows [10:40] survive -> ratio 5.0, n = 30*50
    assert abs(fr["ratio"] - 5.0) < 1e-9
    assert fr["n"] == 30 * 50


def test_frame_ratio_returns_none_below_min_valid():
    slam = np.ones((10, 10))
    metric = 3.0 * slam
    valid = np.zeros_like(slam, bool)
    valid[0, :5] = True  # only 5 valid pixels
    assert ma.frame_ratio(slam, valid, metric, min_valid=200) is None


def test_frame_ratio_ignores_nonpositive_and_nonfinite():
    slam = np.ones((20, 20))
    slam[0, 0] = 0.0       # non-positive slam -> dropped
    metric = 4.0 * np.ones((20, 20))
    metric[1, 1] = np.nan  # non-finite metric -> dropped
    metric[2, 2] = -1.0    # non-positive metric -> dropped
    valid = np.ones_like(slam, bool)
    fr = ma.frame_ratio(slam, valid, metric, min_valid=100)
    assert abs(fr["ratio"] - 4.0) < 1e-9
    assert fr["n"] == 20 * 20 - 3


# ---------------------------------------------------------------------------
# submap_ratio — aggregation + reliability
# ---------------------------------------------------------------------------
def _fr(ratio, n=1000, rel_iqr=0.03):
    return {"ratio": ratio, "rel_iqr": rel_iqr, "n": n, "slam_med": 1.0, "metric_med_m": ratio}


def test_submap_ratio_median_and_cov():
    out = ma.submap_ratio([_fr(4.0), _fr(5.0), _fr(6.0)])
    assert out["ok"] is True
    assert abs(out["r_i"] - 5.0) < 1e-9
    assert out["n_frames"] == 3
    assert out["within_cov"] == pytest.approx(np.std([4, 5, 6]) / 5.0)


def test_submap_ratio_single_frame_is_failsafe_unreliable():
    # one valid frame -> we cannot judge consistency -> within_cov = +inf (tier-2 will skip)
    out = ma.submap_ratio([_fr(5.0)])
    assert out["ok"] is True
    assert out["r_i"] == 5.0
    assert not np.isfinite(out["within_cov"])


def test_submap_ratio_no_valid_frames():
    out = ma.submap_ratio([None, None])
    assert out["ok"] is False
    assert out["r_i"] is None
    assert not np.isfinite(out["within_cov"])


# ---------------------------------------------------------------------------
# combine_scale — the 4-tier sanity gate
# ---------------------------------------------------------------------------
def test_combine_scale_agree_applies_metric_ratio():
    curr = ma.submap_ratio([_fr(4.0), _fr(4.0), _fr(4.0)])   # cov 0
    prev = ma.submap_ratio([_fr(5.0), _fr(5.0), _fr(5.0)])   # cov 0
    # s_metric = 4/5 = 0.8; geometry agrees (0.82) -> apply the anchor
    s, dec = ma.combine_scale(0.82, curr, prev, n_overlap=5000)
    assert dec["applied"] and dec["reason"] == "agree"
    assert abs(s - 0.8) < 1e-9
    assert abs(dec["s_metric"] - 0.8) < 1e-9


def test_combine_scale_tier2_high_cov_keeps_geometry():
    curr = ma.submap_ratio([_fr(4.0), _fr(6.0), _fr(4.0)])   # CoV ~0.2 >= 0.15 -> unreliable
    prev = ma.submap_ratio([_fr(5.0), _fr(5.0), _fr(5.0)])
    s, dec = ma.combine_scale(0.82, curr, prev, n_overlap=5000)
    assert not dec["applied"] and dec["reason"] == "within_submap_cov_high"
    assert s == 0.82


def test_combine_scale_disagree_rich_overlap_keeps_geometry_flags_untrusted():
    curr = ma.submap_ratio([_fr(10.0)] * 3)
    prev = ma.submap_ratio([_fr(5.0)] * 3)   # s_metric = 2.0
    # geometry says 0.8 -> |ln(2.0/0.8)| = 0.916 > 0.5 gate; overlap rich -> anchor suspect
    s, dec = ma.combine_scale(0.8, curr, prev, n_overlap=5000, thin_overlap=400)
    assert not dec["applied"] and dec["reason"] == "disagree_rich_overlap"
    assert dec["untrusted"] is True
    assert s == 0.8


def test_combine_scale_disagree_thin_overlap_prefers_anchor():
    curr = ma.submap_ratio([_fr(10.0)] * 3)
    prev = ma.submap_ratio([_fr(5.0)] * 3)   # s_metric = 2.0
    # same disagreement but overlap is thin -> geometry is the suspect -> prefer anchor
    s, dec = ma.combine_scale(0.8, curr, prev, n_overlap=100, thin_overlap=400)
    assert dec["applied"] and dec["reason"] == "disagree_thin_overlap"
    assert dec["degenerate"] is True
    assert abs(s - 2.0) < 1e-9


def test_combine_scale_anchor_unavailable_keeps_geometry():
    prev = ma.submap_ratio([_fr(5.0)] * 3)
    s, dec = ma.combine_scale(1.23, None, prev, n_overlap=5000)
    assert not dec["applied"] and dec["reason"] == "anchor_unavailable"
    assert s == 1.23
    # a not-ok submap is also unavailable
    s2, dec2 = ma.combine_scale(1.23, ma.submap_ratio([None]), prev, n_overlap=5000)
    assert not dec2["applied"] and dec2["reason"] == "anchor_unavailable"
    assert s2 == 1.23


def test_combine_scale_degenerate_geometry_prefers_anchor():
    curr = ma.submap_ratio([_fr(4.0)] * 3)
    prev = ma.submap_ratio([_fr(5.0)] * 3)
    s, dec = ma.combine_scale(0.0, curr, prev, n_overlap=5000)  # geom scale collapsed
    assert dec["applied"] and dec["reason"] == "geom_degenerate"
    assert abs(s - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# evenly_spaced / cov
# ---------------------------------------------------------------------------
def test_evenly_spaced():
    assert ma.evenly_spaced(16, 3) == [0, 8, 15]
    assert ma.evenly_spaced(17, 3) == [0, 8, 16]
    assert ma.evenly_spaced(5, 1) == [2]
    assert ma.evenly_spaced(2, 5) == [0, 1]  # k >= n


def test_cov_edge_cases():
    assert np.isnan(ma.cov([]))
    assert ma.cov([2, 2, 2]) == 0.0
    assert ma.cov([1, 2, 3]) == pytest.approx(np.std([1, 2, 3]) / 2.0)


# ---------------------------------------------------------------------------
# MetricScaleAnchor.compute_submap_ratio — full extraction path, injected model
# ---------------------------------------------------------------------------
class _FakeSubmap:
    """Minimal stand-in exposing exactly what compute_submap_ratio reads."""
    def __init__(self, S=6, H=40, W=40, ratio=3.0, conf_threshold=0.3, is_lc=False):
        rng = np.random.RandomState(1)
        self.is_lc_submap = is_lc
        self._last = S - 1
        # camera-frame point clouds: z (channel 2) is the per-pixel SLAM depth
        z = rng.uniform(0.5, 4.0, (S, H, W)).astype(np.float64)
        self.pointclouds = np.concatenate([np.zeros((S, H, W, 2)), z[..., None]], axis=-1)
        self._z = z
        self.colors = (rng.rand(S, H, W, 3) * 255).astype(np.uint8)
        self._conf = np.full((S, H, W), 0.9)  # all high-conf
        self._conf_threshold = conf_threshold
        self._ratio = ratio

    def get_last_non_loop_frame_index(self):
        return self._last

    def get_conf_threshold(self):
        return self._conf_threshold

    def get_conf_masks_frame(self, i):
        return self._conf[i]

    def get_frame_at_index(self, i):
        return np.transpose(self.colors[i].astype(np.float64) / 255.0, (2, 0, 1))


def test_compute_submap_ratio_recovers_injected_ratio():
    sm = _FakeSubmap(ratio=3.0)
    # injected metric depth = 3.0 * slam camera-z for whichever keyframe RGB is passed, matched
    # by color-array identity (the anchor reads submap.colors[fi] as the model input).
    color_to_z = {sm.colors[i].tobytes(): sm._z[i] for i in range(len(sm._z))}

    def infer(rgb):
        z = color_to_z[np.ascontiguousarray(rgb).astype(np.uint8).tobytes()]
        return 3.0 * z

    anchor = ma.MetricScaleAnchor(keyframes=3, min_valid=100, infer_fn=infer)
    out = anchor.compute_submap_ratio(sm)
    assert out["ok"] is True
    assert out["n_frames"] == 3
    assert abs(out["r_i"] - 3.0) < 1e-9
    assert out["within_cov"] == pytest.approx(0.0, abs=1e-9)
    assert out["keyframes"] == [0, 2, 5]  # evenly_spaced(6, 3)


def test_compute_submap_ratio_skips_lc_submap():
    sm = _FakeSubmap(is_lc=True)
    anchor = ma.MetricScaleAnchor(keyframes=3, infer_fn=lambda rgb: np.ones(rgb.shape[:2]))
    out = anchor.compute_submap_ratio(sm)
    assert out["ok"] is False
    assert out.get("reason") == "lc_submap"


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        ma.MetricScaleAnchor(model_name="not-a-real-model")
