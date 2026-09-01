"""Stage-4 helpers: rebuild trajectory rows from a persisted record (no live ``Solver``).

See ``docs/dataset-export.md`` §8 Stage 4. The persisted record stores per-keyframe
``poses (N,4,4)`` (cam->world) + ``intrinsics (N,4)`` (``[fx,fy,cx,cy]``) +
``source_frame_id (N,)`` -- everything needed to re-emit ``trajectory.parquet`` on the
GPU-free broker. State/action math matches ``trajectory.extract_trajectory`` exactly so a
record export is bit-comparable (to fp tolerance) with a live export of the same scan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from server.export.schema import DEFAULT_FPS
from server.export.trajectory import _pose_to_state, _relative_action
from server.scene_report.cloud_io import read_ply_cloud


def build_trajectory_arrays(solver: Any) -> dict:
    """Extract the compact ``{poses, intrinsics, source_frame_id}`` block for persistence.

    Run once on the GPU worker at scan end (a live ``Solver`` in hand) and stash in the
    scene record so the broker can later export without a GPU. Cheap (~N×20 floats).
    """
    from server.export.trajectory import _decompose, _ordered_keyframes

    poses: list = []
    intrinsics: list = []
    source_ids: list = []
    for gi, (_sid, _local, pose_mat, fid) in enumerate(_ordered_keyframes(solver)):
        K, rot, trans, _scale = _decompose(pose_mat)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = trans
        poses.append(T)
        intrinsics.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        source_ids.append(float(fid) if fid is not None else float(gi))
    return {
        "poses": np.asarray(poses, dtype=np.float32).reshape(-1, 4, 4),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32).reshape(-1, 4),
        "source_frame_id": np.asarray(source_ids, dtype=np.float32).reshape(-1),
    }


def build_rows_from_trajectory(traj: Any, fps: float = DEFAULT_FPS) -> list:
    """``{poses, intrinsics, source_frame_id}`` -> trajectory rows (same schema as Stage 1)."""
    poses = np.asarray(_get(traj, "poses", []), dtype=np.float64).reshape(-1, 4, 4)
    n = int(poses.shape[0])
    if n == 0:
        return []
    intrinsics = np.asarray(_get(traj, "intrinsics", []), dtype=np.float32).reshape(-1, 4)
    source_ids = np.asarray(_get(traj, "source_frame_id", []), dtype=np.float32).reshape(-1)

    rows: list = []
    for i in range(n):
        rot = poses[i, :3, :3]
        trans = poses[i, :3, 3]
        action = (
            _relative_action(poses[i], poses[i + 1]) if i < n - 1 else np.zeros(6, np.float32)
        )
        intr = intrinsics[i] if i < intrinsics.shape[0] else np.zeros(4, np.float32)
        sid = float(source_ids[i]) if i < source_ids.shape[0] else float(i)
        rows.append(
            {
                "index": int(i),
                "frame_index": int(i),
                "episode_index": 0,
                "timestamp": np.float32(i / float(fps)),
                "source_frame_id": np.float32(sid),
                "observation.state": _pose_to_state(rot, trans),
                "action": action.astype(np.float32),
                "observation.intrinsics": np.asarray(intr, dtype=np.float32),
            }
        )
    return rows


# Derived-artifact filenames written by the product-workflow anchor/clamp routes under a
# ``derived/<kind>/<stamp>/`` group (server/app.py). Any subset may exist: anchor writes all
# three; auto-clamp writes only ``splat.ply``. Filenames must match those routes exactly.
_DERIVED_CLOUD = "cloud.ply"
_DERIVED_TRAJECTORY = "trajectory.npz"
_DERIVED_SPLAT = "splat.ply"

# The explicit "give me the ORIGINAL geometry" selector value (overrides any persisted pointer).
EXPORT_SOURCE_ORIGINAL = "original"


def _resolve_derived_group(persistence: Any, user_id: str, scan_id: str, rec: Any, source: Any) -> dict:
    """Resolve which derived artifact group (if any) an export should read, and return the
    on-disk path for each artifact TYPE it provides (missing ones are ``None`` -> caller falls
    back to the original).

    Precedence — EXPLICIT selector > persisted "latest calibrated" pointer > original:
      * ``source`` is a derived key (e.g. ``derived/anchor/<stamp>/cloud.ply``) -> use that
        key's group (its sibling ``cloud.ply``/``trajectory.npz``/``splat.ply``).
      * ``source`` is ``"original"`` / ``""`` -> force the original (return an empty group).
      * ``source`` is ``None`` -> fall back to the record's ``derived_latest`` pointer if present.

    A ``source_key`` (explicit or from the pointer) that no longer resolves to a real file under
    THIS scan's ``derived/`` dir yields an empty group -> graceful degrade to the original. The
    route validates an *explicit* selector up front so a bogus key is a hard 404 there instead."""
    empty = {"cloud": None, "trajectory": None, "splat": None}
    get_path = getattr(persistence, "get_derived_artifact_path", None)
    if not callable(get_path):
        return empty

    if source is not None:
        s = str(source).strip()
        if s == "" or s == EXPORT_SOURCE_ORIGINAL:
            return empty
        source_key = s
    else:
        pointer = rec.get("derived_latest") if isinstance(rec, dict) else None
        source_key = pointer.get("source_key") if isinstance(pointer, dict) else None
        if not source_key:
            return empty

    if not get_path(user_id, scan_id, str(source_key)):
        return empty  # stale/invalid group -> original
    group = str(source_key).rsplit("/", 1)[0] if "/" in str(source_key) else str(source_key)
    return {
        "cloud": get_path(user_id, scan_id, f"{group}/{_DERIVED_CLOUD}"),
        "trajectory": get_path(user_id, scan_id, f"{group}/{_DERIVED_TRAJECTORY}"),
        "splat": get_path(user_id, scan_id, f"{group}/{_DERIVED_SPLAT}"),
    }


def _load_trajectory_npz(path: str):
    """Load a derived ``trajectory.npz`` (same ``{poses, intrinsics, source_frame_id}`` shape as
    ``ModalScenePersistence.get_trajectory``). ``None`` on any read error."""
    try:
        with np.load(path) as data:
            return {
                "poses": np.asarray(data["poses"], dtype=np.float32).reshape(-1, 4, 4),
                "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32).reshape(-1, 4),
                "source_frame_id": np.asarray(data["source_frame_id"], dtype=np.float32).reshape(-1),
            }
    except Exception as exc:
        print(f"[export] derived trajectory read failed {path}: {exc}")
        return None


def resolved_source_key(record: dict[str, Any], source: Any) -> str:
    """The geometry selector the builders will actually read — the zip route's precedence
    (explicit ``source`` > the scan's persisted "latest calibrated" pointer > original),
    named rather than re-derived.

    Lives here beside ``load_record_from_store`` (which applies the same precedence
    internally) so the broker routes, the export-surface manifest and the background
    export job all label geometry identically. The job in particular must not import a
    Flask route module just to answer this."""
    if source is not None:
        s = str(source).strip()
        if s and s != EXPORT_SOURCE_ORIGINAL:
            return s
        return EXPORT_SOURCE_ORIGINAL
    pointer = record.get("derived_latest") if isinstance(record, dict) else None
    if isinstance(pointer, dict) and pointer.get("source_key"):
        return str(pointer["source_key"])
    return EXPORT_SOURCE_ORIGINAL


def load_record_from_store(
    persistence: Any, user_id: str, scan_id: str, source: Any = None
) -> dict:
    """Assemble the normalized ``export_from_record`` input from a persisted scene.

    Reads facts/report from the metadata record, the per-keyframe ``trajectory.npz``, and
    the full ``cloud.npz`` off the durable store -- no GPU / ``Solver``. Returns ``None`` if
    the scan has no persisted trajectory (older scans pre-Stage-4).

    ``source`` selects the geometry (precedence EXPLICIT > persisted pointer > original — see
    ``_resolve_derived_group``). When a derived group is selected, the calibrated/clamped
    ``cloud.ply`` / ``trajectory.npz`` / ``splat.ply`` under it REPLACE the originals for the
    types it provides; any type it lacks (auto-clamp writes only a splat) still comes from the
    original. NON-DESTRUCTIVE: only READS the ``derived/...`` artifacts; the originals are never
    touched. The splat is carried on the record only when a derived splat is selected, so a plain
    (no-source) export is byte-for-byte unchanged.
    """
    rec = persistence.get_scene(user_id, scan_id)
    if not rec:
        return None

    group = _resolve_derived_group(persistence, user_id, scan_id, rec, source)

    traj = None
    if group.get("trajectory"):
        traj = _load_trajectory_npz(group["trajectory"])
    if traj is None:
        traj = persistence.get_trajectory(user_id, scan_id)
    if traj is None:
        return None

    normalized = {
        "scan_id": rec.get("scan_id", scan_id),
        "trajectory": traj,
        "facts": rec.get("facts"),
        "report": rec.get("report"),
        "keyframes": [],  # persisted JPEGs are a sparse evidence subset; video is skipped
    }

    cloud = None
    if group.get("cloud"):
        try:
            with open(group["cloud"], "rb") as fh:
                cloud = read_ply_cloud(fh.read())
        except Exception as exc:
            print(f"[export] derived cloud read failed {group['cloud']}: {exc}")
            cloud = None
    if cloud is None:
        cloud = persistence.get_cloud(user_id, scan_id)
    if cloud is not None:
        normalized["cloud"] = cloud

    # 3DGS splat: carried ONLY when a derived splat was selected (anchor/clamp). The original
    # splat is intentionally NOT added to a plain export, keeping that path byte-identical.
    if group.get("splat"):
        normalized["splat"] = group["splat"]

    return normalized


def _get(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
