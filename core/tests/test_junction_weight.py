"""CPU-only tests for junction-confidence weighting (vggt_slam.junction_weight) and the
scale-dispersion diagnostics added to vggt_slam.scale_solver.

Pure numpy: no torch / gtsam / open3d needed (scale_solver's open3d import is lazy). The
graph-level effect (a loosened junction letting a good loop-closure constraint win) is verified
synthetically through the real gtsam PoseGraph on Modal — see the research doc — because gtsam's
SL4 build is not available on this CPU box.

Run:  pytest core/tests/test_junction_weight.py -v
"""

import numpy as np
import pytest

# conftest.py puts the repo root on sys.path.
from vggt_slam import junction_weight as jw
from vggt_slam.scale_solver import estimate_scale_pairwise, scale_dispersion_info


# ---------------------------------------------------------------------------
# junction_quality — signal combination
# ---------------------------------------------------------------------------
def test_no_signals_is_neutral():
    # absence of evidence never loosens a junction
    assert jw.junction_quality() == 1.0
    assert jw.junction_covariance_multiplier() == 1.0
    assert jw.sigma_multiplier() == 1.0


def test_perfect_signals_are_neutral():
    q = jw.junction_quality(dispersion=0.0, n_overlap=jw.COVERAGE_REF * 10,
                            image_match_ratio=1.0)
    assert q == pytest.approx(1.0)
    assert jw.junction_covariance_multiplier(dispersion=0.0, n_overlap=jw.COVERAGE_REF * 10,
                                             image_match_ratio=1.0) == pytest.approx(1.0)


def test_quality_monotone_in_each_signal():
    # dispersion: more spread -> lower quality
    qs = [jw.junction_quality(dispersion=d) for d in [0.0, 0.1, 0.3, 0.6, 1.5]]
    assert all(a >= b for a, b in zip(qs, qs[1:]))
    # coverage: fewer overlap pixels -> lower quality
    qs = [jw.junction_quality(n_overlap=n) for n in [5000, 2000, 800, 200, 50]]
    assert all(a >= b for a, b in zip(qs, qs[1:]))
    # match ratio: lower ratio -> lower quality
    qs = [jw.junction_quality(image_match_ratio=r) for r in [1.0, 0.95, 0.9, 0.85, 0.8]]
    assert all(a >= b for a, b in zip(qs, qs[1:]))


def test_cov_multiplier_monotone_and_bounded():
    ms = [jw.junction_covariance_multiplier(dispersion=d) for d in [0.0, 0.2, 0.5, 1.0, 10.0]]
    assert all(b >= a for a, b in zip(ms, ms[1:]))  # looser as dispersion grows
    for m in ms:
        assert 1.0 <= m <= jw.MAX_COV_MULT
    # a hopeless junction saturates near (never beyond) MAX_COV_MULT
    worst = jw.junction_covariance_multiplier(dispersion=100.0, n_overlap=0,
                                              image_match_ratio=0.0)
    assert worst == pytest.approx(jw.MAX_COV_MULT, rel=0.05)


def test_sigma_is_sqrt_of_covariance():
    m = jw.junction_covariance_multiplier(dispersion=0.4, n_overlap=500)
    s = jw.sigma_multiplier(dispersion=0.4, n_overlap=500)
    assert s == pytest.approx(np.sqrt(m))


def test_geometric_mean_combination():
    # one strong + one weak signal must land strictly between the two single-signal qualities
    q_disp = jw.junction_quality(dispersion=0.6)         # weak
    q_cov = jw.junction_quality(n_overlap=jw.COVERAGE_REF)  # strong (=1)
    q_both = jw.junction_quality(dispersion=0.6, n_overlap=jw.COVERAGE_REF)
    assert q_disp < q_both <= q_cov
    assert q_both == pytest.approx(np.sqrt(q_disp * q_cov))


def test_nonfinite_signals_ignored():
    assert jw.junction_quality(dispersion=float("nan")) == 1.0
    assert jw.junction_quality(image_match_ratio=float("nan")) == 1.0
    # a finite signal still counts next to an ignored one
    q = jw.junction_quality(dispersion=float("nan"), n_overlap=100)
    assert q == pytest.approx(jw._q_coverage(100))


def test_match_ratio_ramp_endpoints():
    assert jw._q_match(jw.MATCH_MIN) == 0.0
    assert jw._q_match(jw.MATCH_HI) == 1.0
    assert jw._q_match(0.5 * (jw.MATCH_MIN + jw.MATCH_HI)) == pytest.approx(0.5)
    assert jw._q_match(0.0) == 0.0 and jw._q_match(1.0) == 1.0  # clamped


# ---------------------------------------------------------------------------
# scale_solver — dispersion diagnostics (additive second return element)
# ---------------------------------------------------------------------------
def test_estimate_scale_pairwise_scale_unchanged_and_info_added():
    rng = np.random.RandomState(0)
    X = rng.uniform(0.5, 2.0, (500, 3))
    Y = 3.0 * X
    scale, info = estimate_scale_pairwise(X, Y)
    assert scale == pytest.approx(3.0)               # the scale math is untouched
    assert isinstance(info, dict)
    assert info["n"] == 500
    assert info["dispersion"] == pytest.approx(0.0, abs=1e-12)  # exact ratio -> zero spread


def test_dispersion_grows_with_noise():
    rng = np.random.RandomState(1)
    X = rng.uniform(0.5, 2.0, (2000, 3))
    disps = []
    for noise in [0.0, 0.05, 0.2]:
        Y = 3.0 * X * (1.0 + noise * rng.standard_normal((2000, 1)))
        _, info = estimate_scale_pairwise(X, Y)
        disps.append(info["dispersion"])
    assert disps[0] < disps[1] < disps[2]


def test_scale_dispersion_info_edge_cases():
    assert scale_dispersion_info([])["n"] == 0
    assert np.isnan(scale_dispersion_info([])["dispersion"])
    out = scale_dispersion_info([2.0, 2.0, 2.0])
    assert out["n"] == 3 and out["dispersion"] == 0.0
    # non-finite ratios are dropped, not propagated
    out = scale_dispersion_info([1.0, np.nan, np.inf, 1.0])
    assert out["n"] == 2 and np.isfinite(out["dispersion"])
    assert np.isnan(scale_dispersion_info([0.0, 0.0])["dispersion"])  # zero median -> nan


# ---------------------------------------------------------------------------
# end-to-end mapping sanity on realistic magnitudes
# ---------------------------------------------------------------------------
def test_realistic_regimes():
    # strong junction (rich overlap, tight ratios): essentially unweighted
    m_strong = jw.junction_covariance_multiplier(dispersion=0.05, n_overlap=5000)
    assert m_strong < 1.6
    # weak junction (thin overlap, wild ratios): substantially loosened
    m_weak = jw.junction_covariance_multiplier(dispersion=0.8, n_overlap=150)
    assert m_weak > 10.0
    assert m_weak / m_strong > 8.0
