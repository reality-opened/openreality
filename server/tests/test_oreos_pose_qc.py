"""Fix-independent improper-rotation counting (server/oreos/recordings/pose_qc.py).

The regression this guards is a DEAD SIGNAL, not a crash: modal_recon and the
live node used to count how often *their own* copy of core's det(R) = -1 repair
fired. core 3ea5a03 moved that repair inside decompose_camera, so from the next
image build onward the local copy could never fire, `n_pose_sign_repairs` would
be 0 on every recording, `sign_repair_rate` (thresholds 0.01/0.05, calibrated on
EXP-36) would read "clean" forever, and nothing would warn.

So the tests here assert the counting is done on the RAW camera matrix — the
event — and that it agrees with what core's decompose_camera repairs internally
(that last test needs `vggt_slam` and is skipped in the GPU-free CI subset, like
the other core-dependent export tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from server.oreos.recordings import pose_qc


# -- fixtures: honest camera matrices -----------------------------------------

K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 240.0],
              [0.0, 0.0, 1.0]])


def rot(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return Rz @ Ry


def camera_mat(R: np.ndarray, t=(0.1, -0.2, 1.5)) -> np.ndarray:
    """4x4 projection matrix in the shape ``get_all_poses_world(give_camera_mat=True)``
    returns: ``P = K [R | t]`` stacked on a homogeneous row, dehomogenised."""
    P = np.eye(4)
    P[:3, :3] = K @ R
    P[:3, 3] = K @ np.asarray(t, dtype=float)
    return P


def improper(R: np.ndarray) -> np.ndarray:
    """The det(R) = -1 twin of R — what an SL(4) decomposition can hand back."""
    return -R


class FakeSubmap:
    def __init__(self, cams, sid=0, raises=None, honors_flag=True):
        self._cams = np.asarray(cams, dtype=float)
        self._sid = sid
        self._raises = raises
        self._honors_flag = honors_flag

    def get_id(self):
        return self._sid

    def get_all_poses_world(self, graph, give_camera_mat=False):
        if self._raises is not None:
            raise self._raises
        if not give_camera_mat or not self._honors_flag:
            # what a caller gets WITHOUT the raw matrices: already-repaired poses
            out = []
            for P in self._cams:
                R = P[:3, :3] / np.linalg.norm(P[:3, :3], axis=0)
                if np.linalg.det(R) < 0:
                    R = -R
                T = np.eye(4)
                T[:3, :3] = R
                out.append(T)
            return np.stack(out)
        return self._cams


# -- counting ------------------------------------------------------------------


class TestImproperRotationStats:
    def test_all_proper(self):
        cams = [camera_mat(rot(a)) for a in (0.0, 0.4, 1.1)]
        s = pose_qc.improper_rotation_stats(cams)
        assert s == {"n_poses": 3, "n_improper": 0, "n_nonfinite": 0}

    def test_counts_the_raw_improper_ones(self):
        cams = [camera_mat(rot(0.0)), camera_mat(improper(rot(0.3))),
                camera_mat(rot(0.6)), camera_mat(improper(rot(0.9)))]
        s = pose_qc.improper_rotation_stats(cams)
        assert s["n_poses"] == 4 and s["n_improper"] == 2 and s["n_nonfinite"] == 0

    def test_intrinsics_never_flip_the_sign(self):
        """det(K) > 0 after core's positive-diagonal normalisation, so the raw
        3x3 determinant IS the rotation's determinant sign."""
        for f in (1.0, 500.0, 1e4):
            k = np.diag([f, f * 1.3, 1.0])
            k[0, 2], k[1, 2] = 300.0, 200.0  # principal point: no effect on det
            P = np.eye(4)
            P[:3, :3] = k @ improper(rot(0.7))
            assert pose_qc.improper_rotation_stats([P])["n_improper"] == 1

    def test_nonfinite_is_reported_not_swallowed(self):
        """A projectively-singular SL(4) pose (EXP-36 failure mode #2) must not
        be quietly counted as proper."""
        bad = camera_mat(rot(0.2))
        bad[:3, :3] = np.inf
        s = pose_qc.improper_rotation_stats([camera_mat(rot(0.0)), bad])
        assert s["n_nonfinite"] == 1 and s["n_improper"] == 0

    def test_empty_and_none(self):
        assert pose_qc.improper_rotation_stats(None)["n_poses"] == 0
        assert pose_qc.improper_rotation_stats(np.zeros((0, 4, 4)))["n_poses"] == 0

    def test_accepts_a_single_matrix_and_3x4(self):
        assert pose_qc.improper_rotation_stats(camera_mat(rot(0.1)))["n_poses"] == 1
        assert pose_qc.improper_rotation_stats(
            [camera_mat(improper(rot(0.1)))[:3, :]])["n_improper"] == 1


class TestSubmapStats:
    def test_reads_the_raw_camera_matrices(self):
        sm = FakeSubmap([camera_mat(rot(0.0)), camera_mat(improper(rot(0.5)))])
        s = pose_qc.submap_improper_stats(sm, graph=None)
        assert s["available"] is True and s["n_improper"] == 1

    def test_counting_survives_core_repairing_first(self):
        """THE regression: core repairs internally, so a count taken from the
        REPAIRED poses is 0 while the raw count still sees the event."""
        cams = [camera_mat(improper(rot(a))) for a in (0.0, 0.4, 0.8)]
        sm = FakeSubmap(cams)
        repaired = sm.get_all_poses_world(None)  # what core hands the worker
        assert all(np.linalg.det(T[:3, :3]) > 0 for T in repaired)  # nothing to repair
        assert pose_qc.submap_improper_stats(sm, None)["n_improper"] == 3

    def test_unavailable_is_flagged_not_zeroed(self):
        sm = FakeSubmap([camera_mat(rot(0.0))], raises=TypeError("no give_camera_mat"))
        s = pose_qc.submap_improper_stats(sm, None)
        assert s["available"] is False and "note" in s

    def test_unusable_shape_is_flagged(self):
        class Weird:
            def get_all_poses_world(self, graph, give_camera_mat=False):
                return "not a matrix"

        s = pose_qc.submap_improper_stats(Weird(), None)
        assert s["available"] is False


# -- the tripwire --------------------------------------------------------------


class TestProperRigidPoses:
    def test_passes_proper_poses_through_orthonormalised(self):
        T = np.eye(4)
        T[:3, :3] = rot(0.3) @ (np.eye(3) + 1e-9)  # RQ output: orthogonal to float
        T[:3, 3] = [1.0, 2.0, 3.0]
        out = pose_qc.proper_rigid_poses([T], where="unit")
        assert len(out) == 1
        R = out[0][:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(out[0][:3, 3], [1.0, 2.0, 3.0])

    def test_raises_on_an_improper_pose_naming_where(self):
        T = np.eye(4)
        T[:3, :3] = improper(rot(0.3))
        with pytest.raises(pose_qc.ImproperRotationError) as e:
            pose_qc.proper_rigid_poses([np.eye(4), T], where="submap 7")
        msg = str(e.value)
        assert "submap 7" in msg and "pose 1" in msg and "3ea5a03" in msg

    def test_raises_on_a_degenerate_pose(self):
        T = np.eye(4)
        T[:3, :3] = np.zeros((3, 3))
        with pytest.raises(pose_qc.ImproperRotationError):
            pose_qc.proper_rigid_poses([T])

    def test_is_an_assertion_error_so_callers_can_stay_non_fatal(self):
        assert issubclass(pose_qc.ImproperRotationError, AssertionError)

    def test_empty(self):
        assert pose_qc.proper_rigid_poses(None) == []
        assert pose_qc.proper_rigid_poses(np.zeros((0, 4, 4))) == []


# -- agreement with the core the images actually ship --------------------------


class TestAgreesWithCore:
    """The raw-determinant predicate must be exactly what core repairs.

    Skipped without `vggt_slam` (the GPU-free CI subset), same as the other
    core-dependent tests in this repo.
    """

    @pytest.fixture(autouse=True)
    def _core(self):
        slam_utils = pytest.importorskip(
            "vggt_slam.slam_utils",
            reason="openreality-core not installed (GH_PAT unset in CI)",
        )
        self.decompose = slam_utils.decompose_camera

    def test_raw_det_sign_predicts_cores_repair(self):
        for yaw in (0.0, 0.7, 2.1):
            proper_P = camera_mat(rot(yaw))
            improper_P = camera_mat(improper(rot(yaw)))

            assert pose_qc.improper_rotation_stats([proper_P])["n_improper"] == 0
            assert pose_qc.improper_rotation_stats([improper_P])["n_improper"] == 1

            # core repairs both to a PROPER rotation — which is exactly why the
            # old "count our own repair" scheme now counts zero.
            for P in (proper_P, improper_P):
                _K, R, _t, _s = self.decompose(P)
                assert np.linalg.det(R) > 0

    def test_core_folds_exactly_the_sign_we_count(self):
        """P and -P are the same projective camera; one of them is improper.

        Feeding the 3x4 form (no dehomogenisation, so the sign survives), core
        must return the SAME (R, t) for both — i.e. it folds the sign into (R, t)
        precisely when our raw determinant is negative. That is the equivalence
        the manifest key rests on.
        """
        P = camera_mat(rot(0.9))[:3, :]
        assert pose_qc.improper_rotation_stats([P])["n_improper"] == 0
        assert pose_qc.improper_rotation_stats([-P])["n_improper"] == 1
        _K1, R1, t1, _s1 = self.decompose(P)
        _K2, R2, t2, _s2 = self.decompose(-P)
        assert np.allclose(R1, R2, atol=1e-9)
        assert np.allclose(t1, t2, atol=1e-6)

    def test_core_repaired_poses_pass_the_tripwire(self):
        P = camera_mat(improper(rot(1.0)))
        _K, R, t, _s = self.decompose(P)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        out = pose_qc.proper_rigid_poses([T], where="core output")
        assert np.linalg.det(out[0][:3, :3]) == pytest.approx(1.0)
