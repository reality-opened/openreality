"""Regression: normalize_to_sl4 must refuse what it cannot normalize.

Two silent failures used to escape this helper:

* **det < 0** — ``det ** (1/4)`` on a negative numpy float is NaN, so the caller received
  an all-NaN 4x4 and no exception. No real scaling can rescue a negative determinant
  anyway: H/s has det(H)/s**4 and s**4 > 0 for every real s.
* **det ~ 0** — the singularity check was exact equality (``det == 0``), so det = 1e-300
  passed and produced a ~1e75 blow-up instead of an error.
"""

import sys
import types

import numpy as np
import pytest

# slam_utils imports torch/torchvision at module top for unrelated helpers; the function
# under test is pure numpy. Stub ONLY genuinely-absent modules so a real installed torch
# is never clobbered (house pattern — see conftest stubs / test_decompose_camera.py).
_STUBBED = []
for _name in ("torch", "torchvision", "torchvision.transforms"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)
            _STUBBED.append(_name)
if "torchvision.transforms" in _STUBBED:
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
if "torch" in _STUBBED:
    # scipy's array-API compat probes any importable torch for Tensor — keep it happy
    sys.modules["torch"].Tensor = type("Tensor", (), {})

from vggt_slam.slam_utils import normalize_to_sl4  # noqa: E402


# --------------------------------------------------------------------------- #
# valid input still normalizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(10))
def test_valid_matrix_normalizes_to_unit_determinant(seed):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((4, 4))
    if np.linalg.det(H) < 0:  # only positive-determinant matrices are normalizable
        H[:, [0, 1]] = H[:, [1, 0]]

    out = normalize_to_sl4(H)

    assert np.linalg.det(out) == pytest.approx(1.0, rel=1e-9)
    # normalization is a pure positive rescale: direction is preserved
    ratio = out / H
    assert np.allclose(ratio, ratio.flat[0])
    assert ratio.flat[0] > 0


def test_identity_is_a_fixed_point():
    assert np.allclose(normalize_to_sl4(np.eye(4)), np.eye(4))


def test_result_is_invariant_to_input_scaling():
    # H and cH are the same projective object, so both must normalize to the same SL(4).
    rng = np.random.default_rng(3)
    H = rng.standard_normal((4, 4))
    if np.linalg.det(H) < 0:
        H[:, [0, 1]] = H[:, [1, 0]]

    assert np.allclose(normalize_to_sl4(H), normalize_to_sl4(H * 1e-6), atol=1e-9)
    assert np.allclose(normalize_to_sl4(H), normalize_to_sl4(H * 1e6), atol=1e-9)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_negative_determinant_raises_rather_than_returning_nan():
    H = np.diag([1.0, 1.0, 1.0, -1.0])  # det = -1
    assert np.linalg.det(H) < 0

    with pytest.raises(ValueError, match="negative determinant"):
        normalize_to_sl4(H)


@pytest.mark.parametrize("tiny", [0.0, 1e-300, 1e-18])
def test_near_zero_determinant_raises(tiny):
    H = np.diag([1.0, 1.0, 1.0, tiny])

    with pytest.raises(ValueError, match="singular"):
        normalize_to_sl4(H)


def test_exactly_singular_matrix_raises():
    H = np.ones((4, 4))  # rank 1
    with pytest.raises(ValueError, match="singular"):
        normalize_to_sl4(H)


def test_all_zero_matrix_raises():
    with pytest.raises(ValueError, match="all-zero or non-finite"):
        normalize_to_sl4(np.zeros((4, 4)))


def test_non_finite_matrix_raises():
    H = np.eye(4)
    H[0, 0] = np.nan
    with pytest.raises(ValueError, match="all-zero or non-finite"):
        normalize_to_sl4(H)


def test_small_magnitude_matrix_is_not_falsely_called_singular():
    # det(1e-4 * I) = 1e-16 in absolute terms, but the matrix is perfectly well
    # conditioned — the relative determinant test must let it through.
    H = np.eye(4) * 1e-4
    out = normalize_to_sl4(H)
    assert np.linalg.det(out) == pytest.approx(1.0, rel=1e-9)
