"""Per-keyframe trajectory rows: pose *state* + ego-motion *action* + intrinsics.

See ``docs/dataset-export.md`` §4.1, §7.1-7.4. Stage 1.

CORRECTNESS TRAPS (read the spec):
- Mirror ``GraphMap.write_poses_to_file`` (``vggt_slam/map.py:134``) exactly for the
  pose<->frame_id<->ordering alignment; skip loop-closure submaps (``submap.get_lc_status()``).
- Build the ``(submap_id, local_pose_idx) -> global_index`` map on the SAME walk (§7.1).
- Assert ``len(frame_ids) == len(poses)`` per non-LC submap (§7.2).
- Synthetic uniform ``timestamp = frame_index / fps``; keep true timing in ``source_frame_id`` (§7.3).

GPU-free / duck-typed: ``solver`` only needs ``.graph`` + ``.map``; ``.map`` needs
``ordered_submaps_by_key()`` / ``get_submap``; each submap needs ``get_id()``,
``get_lc_status()``, ``get_all_poses_world(graph, give_camera_mat=True)`` and
``get_frame_ids()``. Heavy imports stay local.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np

from server.export.schema import DEFAULT_FPS


def _decompose(pose_mat: np.ndarray):
    """``decompose_camera(P)`` -> ``(K, R_cam2world, t_world, scale)`` (lazy import)."""
    from vggt_slam.slam_utils import decompose_camera

    return decompose_camera(np.asarray(pose_mat, dtype=np.float64))


def _ordered_keyframes(solver: Any) -> Iterator[tuple]:
    """Canonical single walk over keyframes in **global order**.

    Yields ``(submap_id, local_pose_idx, pose_mat_4x4, frame_id_or_None)`` for every
    non-loop-closure keyframe, mirroring ``GraphMap.get_all_cam_matricies`` +
    ``write_poses_to_file`` so the ordering can never drift from the poses/frame_ids.

    Per §7.2, if ``len(frame_ids) != len(poses)`` for a submap we do not mis-pair: we
    yield ``None`` frame ids for that submap (the caller substitutes a synthetic
    ``source_frame_id``) and log loudly.
    """
    graph = getattr(solver, "graph", None)
    smap = getattr(solver, "map", None)
    if smap is None:
        return
    for submap in smap.ordered_submaps_by_key():
        try:
            if submap.get_lc_status():
                continue
        except Exception:
            pass
        poses = np.asarray(
            submap.get_all_poses_world(graph, give_camera_mat=True), dtype=np.float64
        )
        if poses.ndim != 3 or poses.shape[0] == 0:
            continue
        n = int(poses.shape[0])
        try:
            frame_ids = submap.get_frame_ids()
        except Exception:
            frame_ids = None
        if frame_ids is None or len(frame_ids) != n:
            print(
                f"[export] WARN frame_ids/poses length mismatch for submap "
                f"{submap.get_id()} (poses={n}, frame_ids="
                f"{None if frame_ids is None else len(frame_ids)}); falling back to "
                f"synthetic source_frame_id."
            )
            frame_ids = [None] * n
        sid = submap.get_id()
        for local in range(n):
            yield sid, local, poses[local], frame_ids[local]


def build_index_map(solver: Any) -> dict:
    """``(submap.get_id(), local_pose_idx) -> global keyframe index`` over non-LC submaps.

    Same ordered walk as ``extract_trajectory`` (§7.1) so the two can never disagree.
    """
    index_map: dict = {}
    gi = 0
    for sid, local, _pose, _fid in _ordered_keyframes(solver):
        index_map[(int(sid), int(local))] = gi
        gi += 1
    return index_map


def extract_trajectory(solver: Any, fps: float = DEFAULT_FPS) -> tuple:
    """Walk the map -> ``(rows, index_map)``.

    Each row carries ``index``, ``frame_index``, ``episode_index``, ``timestamp``,
    ``source_frame_id``, ``observation.state`` (7), ``action`` (6),
    ``observation.intrinsics`` (4). Reuses ``decompose_camera`` for (K, R, t) per keyframe
    and the scipy ``Rotation`` quaternion/rotvec helpers, exactly like
    ``GraphMap.write_poses_to_file``.
    """
    entries = list(_ordered_keyframes(solver))
    n = len(entries)

    index_map: dict = {}
    poses_4x4: list = []
    states: list = []
    intrinsics: list = []
    source_ids: list = []

    for gi, (sid, local, pose_mat, fid) in enumerate(entries):
        index_map[(int(sid), int(local))] = gi
        K, rot, trans, _scale = _decompose(pose_mat)
        # cam->world 4x4 pose, matching ``Submap.get_all_poses_world`` (give_camera_mat=False)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = trans
        poses_4x4.append(T)
        states.append(_pose_to_state(rot, trans))
        intrinsics.append(
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        )
        source_ids.append(float(fid) if fid is not None else float(gi))

    rows: list = []
    for i in range(n):
        if i < n - 1:
            action = _relative_action(poses_4x4[i], poses_4x4[i + 1])
        else:
            action = np.zeros(6, dtype=np.float32)
        rows.append(
            {
                "index": int(i),
                "frame_index": int(i),
                "episode_index": 0,
                "timestamp": np.float32(i / float(fps)),
                "source_frame_id": np.float32(source_ids[i]),
                "observation.state": states[i].astype(np.float32),
                "action": action.astype(np.float32),
                "observation.intrinsics": intrinsics[i].astype(np.float32),
            }
        )
    return rows, index_map


def _pose_to_state(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(R, t) cam->world -> ``[tx,ty,tz,qx,qy,qz,qw]`` (quaternion xyzw).

    ``Rotation.from_matrix(rot).as_quat()`` returns xyzw, matching the convention in
    ``GraphMap.write_poses_to_file`` (``map.py:160``).
    """
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()  # xyzw
    t = np.asarray(trans, dtype=np.float64).reshape(3)
    return np.concatenate([t, quat]).astype(np.float32)


def _relative_action(pose_t: np.ndarray, pose_t1: np.ndarray) -> np.ndarray:
    """Ego-motion ``[Δtrans(3), Δrotvec(3)]`` from two 4×4 cam->world poses (§7.3).

    ``T_rel = inv(pose_t) @ pose_t1`` expresses the motion in frame ``t``.
    """
    from scipy.spatial.transform import Rotation

    T_rel = np.linalg.inv(np.asarray(pose_t, dtype=np.float64)) @ np.asarray(
        pose_t1, dtype=np.float64
    )
    dtrans = T_rel[:3, 3]
    drot = Rotation.from_matrix(T_rel[:3, :3]).as_rotvec()
    return np.concatenate([dtrans, drot]).astype(np.float32)


# Columns whose values are per-keyframe vectors (stored as parquet list columns).
_VECTOR_COLUMNS = ("observation.state", "action", "observation.intrinsics")


def write_trajectory_parquet(rows: list, path: str) -> None:
    """Write ``data/trajectory.parquet`` (one row per keyframe, list-typed vector columns)."""
    import pandas as pd

    records = []
    for r in rows:
        rec = {
            "index": int(r["index"]),
            "frame_index": int(r["frame_index"]),
            "episode_index": int(r["episode_index"]),
            "timestamp": float(r["timestamp"]),
            "source_frame_id": float(r["source_frame_id"]),
        }
        for col in _VECTOR_COLUMNS:
            rec[col] = np.asarray(r[col], dtype=np.float32).tolist()
        records.append(rec)
    pd.DataFrame.from_records(records).to_parquet(path, index=False)
