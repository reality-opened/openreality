"""Transcode an OpenReality export tree -> GR00T-LeRobot **v2** + ``meta/modality.json``.

See ``docs/dataset-export.md`` §1, §5. Stage 3.

GR00T consumes LeRobot **v2** (one parquet + one mp4 per episode) plus the GR00T-specific
``meta/modality.json``. The v2 on-disk layout is::

    meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4

Verify the target layout against the live NVIDIA doc (§1) before trusting it downstream --
it evolves. Load/verify with ``LeRobotSingleDataset(path, modality_configs,
EmbodimentTag.NEW_EMBODIMENT)``.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Optional

import numpy as np

LEROBOT_V2_VERSION = "v2.0"
_VIDEO_KEY = "observation.images.ego_view"
_TASK_ANNOTATION = "annotation.human.action.task_description"
_SCENE_ANNOTATION = "annotation.scene.caption"

# Feature widths -- the source of truth the modality.json index ranges must match (§5 unit test).
_STATE_DIM = 7
_ACTION_DIM = 6


def build_modality_json() -> dict:
    """The GR00T ``meta/modality.json`` for our camera/nav embodiment (§5).

    Index ranges MUST equal the feature widths (asserted in tests).
    """
    return {
        "state": {"camera_pose": {"start": 0, "end": _STATE_DIM}},
        "action": {"ego_motion": {"start": 0, "end": _ACTION_DIM}},
        "video": {"ego_view": {"original_key": _VIDEO_KEY}},
        "annotation": {
            "human.action.task_description": {},
            "scene.caption": {},
        },
    }


def _read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _read_jsonl(path: str) -> list:
    out: list = []
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _scene_caption(export_dir: str) -> str:
    """The episode's scene caption: report summary, else the ``scene.caption`` annotation."""
    report_path = os.path.join(export_dir, "meta", "report.json")
    if os.path.isfile(report_path):
        summary = (_read_json(report_path).get("summary") or "").strip()
        if summary:
            return summary
    for ann in _read_jsonl(os.path.join(export_dir, "meta", "annotations.jsonl")):
        if ann.get("channel") == "scene.caption" and (ann.get("text") or "").strip():
            return ann["text"].strip()
    return ""


def _task_description(export_dir: str, scene_caption: str) -> str:
    """The per-frame task string. v1: the scene caption, falling back to a generic nav task."""
    return scene_caption or "Navigate through the scanned environment."


def _ego_view_hw(info: dict) -> tuple:
    feat = (info.get("features") or {}).get(_VIDEO_KEY) or {}
    shape = feat.get("shape") or []
    if len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def _lerobot_v2_info(
    info: dict, n_frames: int, n_tasks: int, fps: float, height: int, width: int
) -> dict:
    """A LeRobot v2 ``meta/info.json`` describing our single-episode camera embodiment."""
    return {
        "codebase_version": LEROBOT_V2_VERSION,
        "robot_type": "openreality_camera",
        "total_episodes": 1,
        "total_frames": int(n_frames),
        "total_tasks": int(n_tasks),
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": float(fps),
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [_STATE_DIM],
                "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
            },
            "action": {
                "dtype": "float32",
                "shape": [_ACTION_DIM],
                "names": ["dx", "dy", "dz", "rx", "ry", "rz"],
            },
            _VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            _TASK_ANNOTATION: {"dtype": "int64", "shape": [1], "names": None},
            _SCENE_ANNOTATION: {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def convert_to_groot_lerobot(export_dir: str, out_dir: str) -> str:
    """Convert ``export/<scan_id>/`` -> the GR00T-LeRobot v2 dataset at ``out_dir``. Return ``out_dir``."""
    import pandas as pd

    info = _read_json(os.path.join(export_dir, "meta", "info.json"))
    fps = float(info.get("fps", 10.0))
    height, width = _ego_view_hw(info)

    traj = pd.read_parquet(os.path.join(export_dir, "data", "trajectory.parquet"))
    n = len(traj)

    scene_caption = _scene_caption(export_dir)
    task_desc = _task_description(export_dir, scene_caption)

    # tasks.jsonl: dedupe language strings -> integer index.
    tasks: list = []
    task_index: dict = {}

    def _task_id(text: str) -> int:
        text = text or ""
        if text not in task_index:
            task_index[text] = len(tasks)
            tasks.append(text)
        return task_index[text]

    desc_id = _task_id(task_desc)
    scene_id = _task_id(scene_caption or task_desc)

    # layout
    meta_dir = os.path.join(out_dir, "meta")
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    video_dir = os.path.join(out_dir, "videos", "chunk-000", _VIDEO_KEY)
    for d in (meta_dir, data_dir, video_dir):
        os.makedirs(d, exist_ok=True)

    # per-frame parquet
    out = pd.DataFrame(
        {
            "observation.state": [
                np.asarray(v, dtype=np.float32).tolist() for v in traj["observation.state"]
            ],
            "action": [np.asarray(v, dtype=np.float32).tolist() for v in traj["action"]],
            "timestamp": traj["timestamp"].astype("float32"),
            "frame_index": traj["frame_index"].astype("int64"),
            "episode_index": np.zeros(n, dtype="int64"),
            "index": traj["index"].astype("int64"),
            "task_index": np.full(n, desc_id, dtype="int64"),
            _TASK_ANNOTATION: np.full(n, desc_id, dtype="int64"),
            _SCENE_ANNOTATION: np.full(n, scene_id, dtype="int64"),
        }
    )
    out.to_parquet(os.path.join(data_dir, "episode_000000.parquet"), index=False)

    # video: copy mp4 across (frame i == row i already guaranteed upstream, §7.4)
    src_mp4 = os.path.join(export_dir, "videos", "ego_view.mp4")
    if os.path.isfile(src_mp4):
        shutil.copyfile(src_mp4, os.path.join(video_dir, "episode_000000.mp4"))

    # meta files
    _write_json(os.path.join(meta_dir, "info.json"), _lerobot_v2_info(info, n, len(tasks), fps, height, width))
    _write_jsonl(
        os.path.join(meta_dir, "episodes.jsonl"),
        [{"episode_index": 0, "tasks": [task_desc], "length": int(n)}],
    )
    _write_jsonl(
        os.path.join(meta_dir, "tasks.jsonl"),
        [{"task_index": i, "task": t} for i, t in enumerate(tasks)],
    )
    _write_json(os.path.join(meta_dir, "modality.json"), build_modality_json())

    return out_dir


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
