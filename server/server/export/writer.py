"""Orchestrator: assemble a complete ``export/<scan_id>/`` tree.

See ``docs/dataset-export.md`` §4, §8. Stage 1 (live ``Solver``) / Stage 4 (persisted record).

Two entry points:
- ``export_scene`` -- from a live ``Solver`` at scan end (full geometry; the offline
  ``main.py --export_dataset`` path and the streaming scan-end hook).
- ``export_from_record`` -- GPU-free, from a persisted scene record on the broker (Stage 4;
  uses the per-keyframe poses + intrinsics persisted by ``store.save_scene``).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Any, Optional

import numpy as np

from server.export import clouds as clouds_mod
from server.export import dynamics as dynamics_mod
from server.export import grounding as grounding_mod
from server.export import trajectory as trajectory_mod
from server.export import video as video_mod
from server.export.schema import (
    DEFAULT_FPS,
    build_episode_summary,
    build_info,
)


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------

def _make_tree(out_dir: str, scan_id: str) -> dict:
    base = os.path.join(out_dir, scan_id)
    paths = {
        "base": base,
        "meta": os.path.join(base, "meta"),
        "data": os.path.join(base, "data"),
        "videos": os.path.join(base, "videos"),
        "clouds": os.path.join(base, "clouds"),
        "grounding": os.path.join(base, "grounding"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _derive_scan_id(provided: Optional[str]) -> str:
    if provided:
        return str(provided)
    return f"scan_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _dump_json(obj: Any, path: str) -> None:
    payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------
# live Solver path (Stage 1-2)
# ---------------------------------------------------------------------------

def export_scene(
    solver: Any,
    detections: list,
    report: Any,
    out_dir: str,
    scan_id: Optional[str] = None,
    fps: float = DEFAULT_FPS,
    *,
    include_frame_clouds: bool = False,
    object_tracks: Optional[list] = None,
) -> str:
    """Write the full OpenReality export for one scan; return the scan export dir.

    Order: build ``index_map`` + trajectory rows -> ``meta/info.json`` + ``episode.json`` ->
    ``data/trajectory.parquet`` -> ``videos/ego_view.mp4`` -> ``clouds/cloud.npz`` ->
    grounding sidecar. The sidecar is best-effort -- a semantic-layer failure must not drop
    the trajectory dataset.

    ``object_tracks`` (optional): a list of ``ObjectTrack``s (dynamic objects lifted to world
    per-frame; see ``export/dynamics.py``). When provided, a ``dynamics/objects_tracks.jsonl``
    sidecar is written; when ``None`` (the default) nothing dynamic is emitted and the export
    is byte-identical to the static-only path.
    """
    scan_id = _derive_scan_id(scan_id)
    paths = _make_tree(out_dir, scan_id)

    rows, index_map = trajectory_mod.extract_trajectory(solver, fps=fps)
    n = len(rows)

    height, width = _first_frame_hw(solver, index_map)

    facts = _resolve_facts(solver, detections, report)

    info = build_info(scan_id, height, width, fps)
    _dump_json(info, os.path.join(paths["meta"], "info.json"))
    episode = build_episode_summary(
        scan_id,
        num_keyframes=n,
        trajectory_length=_trajectory_length(facts),
        room_type=str(_attr(report, "room_type", "unknown")),
        num_objects=len(_attr(facts, "objects", []) or []),
    )
    _dump_json(episode, os.path.join(paths["meta"], "episode.json"))

    # --- core dataset (must succeed) -----------------------------------
    trajectory_mod.write_trajectory_parquet(
        rows, os.path.join(paths["data"], "trajectory.parquet")
    )
    frames_written = video_mod.write_ego_view_video(
        solver, index_map, os.path.join(paths["videos"], "ego_view.mp4"), fps=fps
    )
    if frames_written != n:
        raise RuntimeError(
            f"video/trajectory desync: wrote {frames_written} frames for {n} rows (§7.4)"
        )
    clouds_mod.write_full_cloud(solver, os.path.join(paths["clouds"], "cloud.npz"))
    if include_frame_clouds:
        clouds_mod.write_frame_clouds(
            solver, index_map, os.path.join(paths["clouds"], "frames")
        )

    # --- grounding sidecar (best-effort) -------------------------------
    try:
        _write_sidecar(facts, report, detections, index_map, paths)
    except Exception as exc:
        print(f"[export] WARN grounding sidecar failed (trajectory dataset kept): {exc}")

    # --- dynamics sidecar (optional, default-off, best-effort) ---------
    # Only touches disk when the caller supplies already-lifted object tracks; otherwise the
    # export tree is unchanged (no dynamics/ dir). See export/dynamics.py.
    if object_tracks:
        try:
            dynamics_mod.write_object_tracks_jsonl(object_tracks, paths["base"])
        except Exception as exc:
            print(f"[export] WARN dynamics sidecar failed (trajectory dataset kept): {exc}")

    return paths["base"]


def _write_sidecar(
    facts: Any, report: Any, detections: list, index_map: dict, paths: dict
) -> None:
    grounding_mod.write_scene_graph(facts, os.path.join(paths["meta"], "scene_graph.json"))
    if report is not None:
        grounding_mod.write_report(report, os.path.join(paths["meta"], "report.json"))
    grounding_mod.write_objects_jsonl(
        facts, detections or [], index_map, os.path.join(paths["grounding"], "objects.jsonl")
    )
    grounding_mod.write_annotations_jsonl(
        facts, report, index_map, os.path.join(paths["meta"], "annotations.jsonl")
    )


def _resolve_facts(solver: Any, detections: list, report: Any) -> Any:
    """Prefer the report's embedded ``SceneFacts``; otherwise derive from the solver."""
    facts = _attr(report, "facts", None)
    if facts is not None and (_attr(facts, "objects", None) is not None):
        return facts
    try:
        from server.scene_report.features import SceneFeatureExtractor

        center = _attr(facts, "scene_center", None) or [0.0, 0.0, 0.0]
        return SceneFeatureExtractor().extract(solver, detections or [], scene_center=center)
    except Exception as exc:
        print(f"[export] WARN could not derive scene facts: {exc}")
        return facts


def _first_frame_hw(solver: Any, index_map: dict) -> tuple:
    if not index_map:
        return 0, 0
    (sid, local), _gi = min(index_map.items(), key=lambda kv: kv[1])
    try:
        submap = solver.map.get_submap(int(sid))
        rgb = video_mod.frame_to_rgb_uint8(submap.get_frame_at_index(int(local)))
        return int(rgb.shape[0]), int(rgb.shape[1])
    except Exception:
        return 0, 0


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _trajectory_length(facts: Any) -> float:
    metrics = _attr(facts, "metrics", None)
    val = _attr(metrics, "trajectory_length", 0.0)
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# persisted-record path (Stage 4, GPU-free)
# ---------------------------------------------------------------------------

def export_from_record(record: Any, out_dir: str) -> str:
    """GPU-free export from a persisted scene record (Stage 4).

    ``record`` is the dict persisted by ``ModalScenePersistence.save_scene`` plus the
    Stage-4 ``trajectory`` block (per-keyframe ``poses (N,4,4)``, ``intrinsics (N,4)``,
    ``source_frame_id (N,)``) and resolved keyframe images / cloud. No live ``Solver``.

    Expected ``record`` keys:
      - ``scan_id``: str
      - ``trajectory``: ``{poses, intrinsics, source_frame_id}`` (arrays / lists)
      - ``keyframes``: ``[{global_index, image}]`` where ``image`` is HWC uint8 (optional)
      - ``cloud``: ``(positions, colors)`` (optional)
      - ``facts`` / ``report``: dicts (optional)
    """
    from server.export.record import build_rows_from_trajectory

    scan_id = _derive_scan_id(_attr(record, "scan_id", None))
    paths = _make_tree(out_dir, scan_id)

    traj = _attr(record, "trajectory", {}) or {}
    fps = float(_attr(record, "fps", DEFAULT_FPS) or DEFAULT_FPS)
    rows = build_rows_from_trajectory(traj, fps=fps)
    n = len(rows)

    keyframes = _attr(record, "keyframes", []) or []
    height, width = _record_frame_hw(keyframes)

    info = build_info(scan_id, height, width, fps)
    _dump_json(info, os.path.join(paths["meta"], "info.json"))
    facts = _attr(record, "facts", None)
    report = _attr(record, "report", None)
    episode = build_episode_summary(
        scan_id,
        num_keyframes=n,
        trajectory_length=_trajectory_length(facts),
        room_type=str(_attr(report, "room_type", "unknown")),
        num_objects=len(_attr(facts, "objects", []) or []),
    )
    _dump_json(episode, os.path.join(paths["meta"], "episode.json"))

    trajectory_mod.write_trajectory_parquet(
        rows, os.path.join(paths["data"], "trajectory.parquet")
    )
    frames_written = _write_record_video(
        keyframes, n, os.path.join(paths["videos"], "ego_view.mp4"), fps
    )
    if keyframes and frames_written != n:
        print(
            f"[export] WARN record video has {frames_written} frames for {n} rows "
            f"(persisted keyframes may be a sparse subset)."
        )

    cloud = _attr(record, "cloud", None)
    if cloud is not None:
        positions, colors = cloud
        clouds_mod.write_cloud_array(
            positions, colors, os.path.join(paths["clouds"], "cloud.npz")
        )

    # Optional 3DGS splat: carried on the record ONLY when a calibrated/clamped derived splat
    # was selected for this export (server.export.record.load_record_from_store). A plain export
    # never sets this, so its tree is byte-identical to before. ``splat`` is either raw PLY bytes
    # or an on-disk path to a derived ``splat.ply``; either way it's COPIED verbatim into the
    # tree (never re-serialized) so a clamped/scaled splat lands exactly as produced.
    splat = _attr(record, "splat", None)
    if splat is not None:
        splat_dst = os.path.join(paths["clouds"], "splat.ply")
        try:
            if isinstance(splat, (bytes, bytearray)):
                with open(splat_dst, "wb") as fh:
                    fh.write(splat)
            else:
                shutil.copyfile(str(splat), splat_dst)
        except Exception as exc:
            print(f"[export] WARN record splat write failed: {exc}")

    # The persisted record keeps facts/report but not the (submap_id, frame_idx)->global
    # index map, so the record path emits the scene-graph + report JSON only; the
    # temporally-windowed annotations / objects.jsonl come from the live ``export_scene``.
    try:
        if facts is not None:
            grounding_mod.write_scene_graph(
                facts, os.path.join(paths["meta"], "scene_graph.json")
            )
        if report is not None:
            grounding_mod.write_report(report, os.path.join(paths["meta"], "report.json"))
    except Exception as exc:
        print(f"[export] WARN record sidecar failed: {exc}")

    return paths["base"]


def _record_frame_hw(keyframes: list) -> tuple:
    for kf in keyframes:
        img = _attr(kf, "image", None)
        if img is None:
            continue
        arr = video_mod.frame_to_rgb_uint8(img)
        return int(arr.shape[0]), int(arr.shape[1])
    return 0, 0


def _write_record_video(keyframes: list, n: int, path: str, fps: float) -> int:
    """Encode persisted keyframes (carrying a ``global_index``) in order."""
    indexed = []
    for kf in keyframes:
        gi = _attr(kf, "global_index", None)
        img = _attr(kf, "image", None)
        if gi is None or img is None:
            continue
        indexed.append((int(gi), img))
    if not indexed:
        return 0
    indexed.sort(key=lambda t: t[0])
    fake_map = {(0, gi): gi for gi, _img in indexed}

    class _Submap:
        def __init__(self, frames):
            self._frames = frames

        def get_frame_at_index(self, local):
            return self._frames[local]

    class _Map:
        def __init__(self, frames):
            self._sm = _Submap(frames)

        def get_submap(self, _sid):
            return self._sm

    class _Solver:
        def __init__(self, frames):
            self.map = _Map(frames)

    frames = {gi: img for gi, img in indexed}
    return video_mod.write_ego_view_video(_Solver(frames), fake_map, path, fps=fps)
