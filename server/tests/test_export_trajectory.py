"""Stage-1 geometry tests: pose<->state round-trip, ego-motion action, index_map ordering.

See docs/dataset-export.md §7.1-7.3, Stage 1. Uses real ``decompose_camera`` via the
``export_fakes`` projection-matrix builder (no GPU).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# extract_trajectory calls vggt_slam.slam_utils.decompose_camera at runtime.
# Skip this whole module when openreality-core is not installed (i.e. GH_PAT
# secret not yet configured in CI).  See scripts/fetch_demo_videos.sh and
# .github/workflows/ci.yml for setup instructions.
pytest.importorskip(
    "vggt_slam.slam_utils",
    reason=(
        "openreality-core not installed — vggt_slam.slam_utils.decompose_camera "
        "is required by extract_trajectory.  Add the GH_PAT secret at "
        "https://github.com/reality-opened/server/settings/secrets/actions to enable."
    ),
)

from export_fakes import FakeMap, FakeSolver, FakeSubmap, rot

from server.export.trajectory import (
    _pose_to_state,
    _relative_action,
    build_index_map,
    extract_trajectory,
    write_trajectory_parquet,
)


def _scene():
    """Two non-LC submaps (2 + 1 keyframes) around a skipped loop-closure submap."""
    s0 = FakeSubmap(
        0,
        cameras=[(np.eye(3), [0.0, 0.0, 0.0]), (rot(rz=0.3), [1.0, 0.0, 0.0])],
        frame_ids=[0.0, 1.0],
    )
    lc = FakeSubmap(1, cameras=[(np.eye(3), [50.0, 0.0, 0.0])], frame_ids=[99.0], lc=True)
    s2 = FakeSubmap(2, cameras=[(np.eye(3), [2.0, 0.0, 0.0])], frame_ids=[10.0])
    return FakeSolver(FakeMap([s0, lc, s2]))


def test_index_map_skips_lc_and_is_globally_ordered():
    index_map = build_index_map(_scene())
    assert index_map == {(0, 0): 0, (0, 1): 1, (2, 0): 2}


def test_extract_trajectory_state_intrinsics_and_source_ids():
    rows, index_map = extract_trajectory(_scene(), fps=10.0)
    assert len(rows) == 3
    assert index_map == {(0, 0): 0, (0, 1): 1, (2, 0): 2}

    # translations recover the chosen world camera centers
    assert rows[0]["observation.state"][:3] == pytest.approx([0, 0, 0], abs=1e-5)
    assert rows[1]["observation.state"][:3] == pytest.approx([1, 0, 0], abs=1e-5)
    assert rows[2]["observation.state"][:3] == pytest.approx([2, 0, 0], abs=1e-5)

    # rotation round-trips: quat(state) == R_cw
    R1 = Rotation.from_quat(rows[1]["observation.state"][3:]).as_matrix()
    assert R1 == pytest.approx(rot(rz=0.3), abs=1e-5)

    # intrinsics [fx,fy,cx,cy] from DEFAULT_K
    assert rows[0]["observation.intrinsics"] == pytest.approx([600, 600, 320, 240], abs=1e-3)

    # synthetic uniform timeline + true source ids preserved
    assert [float(r["timestamp"]) for r in rows] == pytest.approx([0.0, 0.1, 0.2])
    assert [float(r["source_frame_id"]) for r in rows] == pytest.approx([0.0, 1.0, 10.0])

    # all states/actions finite
    for r in rows:
        assert np.all(np.isfinite(r["observation.state"]))
        assert np.all(np.isfinite(r["action"]))


def test_action_recovers_relative_transform_and_last_is_zero():
    rows, _ = extract_trajectory(_scene())
    # last action is zeros (no successor)
    assert rows[-1]["action"] == pytest.approx(np.zeros(6), abs=0)

    # action[0]: from pose0 (I, origin) to pose1 (Rz(0.3), [1,0,0]) -> dtrans=[1,0,0], drot=Rz(0.3)
    a0 = rows[0]["action"]
    assert a0[:3] == pytest.approx([1, 0, 0], abs=1e-5)
    assert Rotation.from_rotvec(a0[3:]).as_matrix() == pytest.approx(rot(rz=0.3), abs=1e-5)


def test_relative_action_on_synthetic_poses():
    def pose(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    Ta = pose(rot(rz=0.2), [1.0, 2.0, 3.0])
    Tb = pose(rot(rz=0.5), [1.5, 2.0, 3.0])
    action = _relative_action(Ta, Tb)
    expected = np.linalg.inv(Ta) @ Tb
    assert action[:3] == pytest.approx(expected[:3, 3], abs=1e-6)
    assert Rotation.from_rotvec(action[3:]).as_matrix() == pytest.approx(
        expected[:3, :3], abs=1e-6
    )


def test_pose_to_state_quaternion_is_xyzw():
    R_cw = rot(rx=0.1, ry=0.2, rz=0.3)
    state = _pose_to_state(R_cw, [4.0, 5.0, 6.0])
    assert state.shape == (7,)
    assert state[:3] == pytest.approx([4, 5, 6])
    # last component is the scalar w for a small rotation (xyzw convention)
    assert Rotation.from_quat(state[3:]).as_matrix() == pytest.approx(R_cw, abs=1e-6)


def test_write_trajectory_parquet_roundtrip(tmp_path):
    import pandas as pd

    rows, _ = extract_trajectory(_scene())
    path = str(tmp_path / "trajectory.parquet")
    write_trajectory_parquet(rows, path)
    df = pd.read_parquet(path)
    assert len(df) == 3
    assert list(df["index"]) == [0, 1, 2]
    assert len(df["observation.state"].iloc[0]) == 7
    assert len(df["action"].iloc[0]) == 6
    assert len(df["observation.intrinsics"].iloc[0]) == 4
    assert np.asarray(df["action"].iloc[-1]) == pytest.approx(np.zeros(6))
