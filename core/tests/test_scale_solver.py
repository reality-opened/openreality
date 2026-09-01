"""Regression: estimate_scale_pairwise must never return a NaN scale.

A point at the origin makes the per-point ratio |Y|/|X| a 0/0 NaN, and np.median
propagates a single NaN to the entire result. That NaN scale became a NaN homography
and silently poisoned the gtsam graph — the run continued and the corruption surfaced
(if ever) far downstream. Deceptively, `scale_dispersion_info` already filtered to
finite ratios, so the diagnostics printed alongside the scale looked perfectly healthy.

The fix takes the median over finite ratios only, and raises when none are finite.
"""

import numpy as np
import pytest

from vggt_slam.scale_solver import estimate_scale_pairwise, scale_dispersion_info


def _pairs(x_norms, scale):
    """Build (X, Y) point pairs along +x with the requested norms and a true scale."""
    X = np.zeros((len(x_norms), 3), dtype=float)
    X[:, 0] = x_norms
    return X, X * scale


def test_origin_point_does_not_poison_the_median():
    # 20 clean pairs at true scale 2.0 plus one degenerate pair at the origin (0/0 -> NaN).
    X, Y = _pairs(np.linspace(1.0, 5.0, 20), 2.0)
    X = np.vstack([X, np.zeros(3)])
    Y = np.vstack([Y, np.zeros(3)])

    scale, info = estimate_scale_pairwise(X, Y)

    assert np.isfinite(scale), "a single origin point must not NaN the whole scale"
    assert scale == pytest.approx(2.0)
    # the finite filter drops exactly the degenerate pair
    assert info["n"] == 20


def test_infinite_ratio_is_filtered_too():
    # |X| = 0 with |Y| > 0 gives +inf rather than NaN — also non-finite, also excluded.
    X, Y = _pairs(np.linspace(1.0, 5.0, 9), 3.0)
    X = np.vstack([X, np.zeros(3)])
    Y = np.vstack([Y, np.array([1.0, 0.0, 0.0])])

    scale, info = estimate_scale_pairwise(X, Y)

    assert scale == pytest.approx(3.0)
    assert info["n"] == 9


def test_all_degenerate_raises_instead_of_returning_nan():
    X = np.zeros((5, 3))
    Y = np.zeros((5, 3))

    with pytest.raises(ValueError, match="no finite scale ratios"):
        estimate_scale_pairwise(X, Y)


def test_clean_input_is_unchanged():
    X, Y = _pairs(np.linspace(0.5, 4.0, 33), 0.25)

    scale, info = estimate_scale_pairwise(X, Y)

    assert scale == pytest.approx(0.25)
    assert info["n"] == 33
    assert info["dispersion"] == pytest.approx(0.0, abs=1e-12)


def test_dispersion_info_still_reports_on_partly_degenerate_input():
    # Guard the asymmetry that hid the bug: dispersion counts only the finite ratios,
    # and now the scale is computed from that same finite subset.
    scales = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
    info = scale_dispersion_info(scales)
    assert info["n"] == 3
    assert np.isfinite(info["dispersion"])
