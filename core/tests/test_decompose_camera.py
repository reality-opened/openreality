"""Regression: decompose_camera must always return a PROPER rotation.

EXP-36 (platform/experiments/exp36_oreos_dimos_spike) hit det(R) = -1 on
319/2074 poses of a real sequence, crashing write_poses_to_file inside
scipy's Rotation.from_matrix. Root cause: RQ diagonal sign fixes toggle the
determinant and nothing restored properness. The fix decomposes -P (the same
projective camera) whenever det goes negative.
"""

import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# slam_utils imports torch/torchvision/PIL/matplotlib at module top for unrelated
# helpers; decompose_camera itself is pure numpy/scipy. Stub ONLY the heavy deps that
# are genuinely absent so this regression runs in GPU-free CI without clobbering a real
# installed torch (the unconditional attribute writes below used to overwrite
# torch.Tensor even when torch was present). House pattern — see conftest stubs.
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

from vggt_slam.slam_utils import decompose_camera  # noqa: E402


def make_P(K: np.ndarray, R: np.ndarray, t: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """P = sign * K [R | t] — sign=-1 exercises the projective sign ambiguity."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return sign * (K @ Rt)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("seed", range(20))
def test_rotation_always_proper(seed: int, sign: float) -> None:
    rng = np.random.default_rng(seed)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])
    R_true = Rotation.random(random_state=int(rng.integers(0, 2**31))).as_matrix()
    t_true = rng.standard_normal(3)
    P = make_P(K, R_true, t_true, sign=sign)

    K_out, R_out, t_out, scale = decompose_camera(P, no_inverse=True)

    assert np.linalg.det(R_out) == pytest.approx(1.0, abs=1e-9)
    # quaternion conversion (the EXP-36 crash site) must succeed
    Rotation.from_matrix(R_out)
    # and the decomposition still reproduces the camera: K R == +/- normalized M
    M = P[:3, :3] if P.shape[0] == 3 else (P / P[-1, -1])[:3, :3]
    recon = (K_out * scale) @ R_out
    assert np.allclose(recon, M, atol=1e-6) or np.allclose(recon, -M, atol=1e-6)
    assert K_out[0, 0] > 0 and K_out[1, 1] > 0 and K_out[2, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_world_pose_branch_proper(sign: float) -> None:
    rng = np.random.default_rng(7)
    K = np.diag([600.0, 600.0, 1.0])
    K[0, 2], K[1, 2] = 310.0, 250.0
    R_true = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    t_true = np.array([1.0, -2.0, 0.5])
    P = make_P(K, R_true, t_true, sign=sign)

    _, R_out, t_out, _ = decompose_camera(P)  # no_inverse=False: cam-to-world

    assert np.linalg.det(R_out) == pytest.approx(1.0, abs=1e-9)
    Rotation.from_matrix(R_out)
    # cam-to-world of K[R|t]: rotation transposes, center = -R^T t — sign-invariant
    assert np.allclose(R_out, R_true.T, atol=1e-8)
    assert np.allclose(t_out, -R_true.T @ t_true, atol=1e-8)


# --------------------------------------------------------------------------- #
# 4x4 dehomogenization guard
#
# The 4x4 branch divides by P[3,3]. An SL(4) submap pose can drive that entry to ~0
# (projectively singular — the camera centre lands on the plane at infinity), which
# produced inf/NaN and then an opaque scipy.linalg.rq / LAPACK crash naming neither
# the bad pose nor the cause. Reached from map.write_poses_to_file, the same callsite
# the improper-rotation fix above was written for.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, 1e-18, -1e-20])
def test_singular_ideal_plane_4x4_raises_clearly(bad):
    P = np.eye(4)
    P[0, 0], P[1, 1] = 500.0, 480.0
    P[3, 3] = bad

    with pytest.raises(ValueError, match="dehomogenize"):
        decompose_camera(P)


def test_4x4_guard_message_names_the_likely_cause():
    P = np.eye(4)
    P[3, 3] = 0.0
    with pytest.raises(ValueError, match="SL\\(4\\) submap pose"):
        decompose_camera(P)


def test_guard_is_relative_to_matrix_scale():
    # A uniformly tiny camera is still a valid projective camera: P and cP are the
    # same object, so an absolute threshold on P[3,3] would reject it wrongly.
    K = np.array([[500.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])
    R_true = Rotation.from_euler("xyz", [0.2, 0.4, -0.7]).as_matrix()
    t_true = np.array([0.3, 0.2, 1.5])
    P4 = np.eye(4)
    P4[:3, :] = K @ np.hstack([R_true, t_true.reshape(3, 1)])

    _, R_out, t_out, _ = decompose_camera(P4 * 1e-15)

    assert np.linalg.det(R_out) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(R_out, R_true.T, atol=1e-6)


def test_valid_4x4_still_decomposes():
    K = np.array([[600.0, 0.0, 310.0], [0.0, 600.0, 250.0], [0.0, 0.0, 1.0]])
    R_true = Rotation.from_euler("xyz", [0.1, -0.5, 1.2]).as_matrix()
    t_true = np.array([-0.4, 1.1, 2.0])
    P4 = np.eye(4)
    P4[:3, :] = K @ np.hstack([R_true, t_true.reshape(3, 1)])

    _, R_out, t_out, _ = decompose_camera(P4)

    assert np.allclose(R_out, R_true.T, atol=1e-8)
    assert np.allclose(t_out, -R_true.T @ t_true, atol=1e-8)
