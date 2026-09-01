"""Encode the keyframe RGB stream -> ``videos/ego_view.mp4``.

See ``docs/dataset-export.md`` §4 (videos/) and §7.4. Stage 1.

CRITICAL (§7.4): mp4 frame ``i`` MUST equal ``trajectory.parquet`` row ``i``. Frames are
encoded in the exact global order from ``trajectory.build_index_map`` -- never dropped or
reordered. If a keyframe cannot be read we raise rather than silently desync the streams.

Reuse the ``encode_frames`` tensor->image pattern (``server/scene_report/keyframes.py:71``):
``submap.get_frame_at_index(i).cpu().permute(1,2,0).numpy() * 255`` -> uint8 HWC. A numpy
fallback keeps this duck-typed / unit-testable without torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from server.export.schema import DEFAULT_FPS

# mp4-compatible fourccs to try in order. Codec/container support is platform-dependent:
# Linux OpenCV (FFMPEG) usually has ``mp4v``; macOS (AVFoundation) has ``avc1`` but not
# ``mp4v``. Try each and keep the first that actually opens.
_MP4_FOURCCS = ("avc1", "mp4v", "H264", "h264")


def frame_to_rgb_uint8(frame: Any) -> np.ndarray:
    """Coerce a keyframe (torch CHW tensor or numpy CHW/HWC array) to ``(H,W,3)`` uint8 RGB."""
    if hasattr(frame, "cpu") and hasattr(frame, "permute"):
        arr = frame.cpu().permute(1, 2, 0).numpy()  # CHW -> HWC
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def _open_writer(path: str, fps: float, width: int, height: int):
    import cv2

    last_err: Any = None
    for code in _MP4_FOURCCS:
        try:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
            if writer.isOpened():
                return writer
            writer.release()
        except Exception as exc:  # pragma: no cover - platform dependent
            last_err = exc
    raise RuntimeError(
        f"Could not open an mp4 VideoWriter for {path!r} (tried {', '.join(_MP4_FOURCCS)}). "
        f"Last error: {last_err}"
    )


def write_ego_view_video(
    solver: Any,
    index_map: dict,
    path: str,
    fps: float = DEFAULT_FPS,
) -> int:
    """Encode keyframes (in global index order) to ``path``; return the frame count written.

    The returned count must equal ``len(index_map)`` and the trajectory row count (§7.4).
    """
    import cv2

    ordered = sorted(index_map.items(), key=lambda kv: kv[1])  # by global index
    smap = getattr(solver, "map", None)
    if smap is None:
        raise ValueError("solver has no .map")

    writer = None
    width = height = None
    count = 0
    try:
        for (sid, local), gi in ordered:
            submap = smap.get_submap(int(sid))
            if submap is None:
                raise ValueError(f"submap {sid} missing while encoding video (global {gi})")
            rgb = frame_to_rgb_uint8(submap.get_frame_at_index(int(local)))
            h, w = rgb.shape[:2]
            if writer is None:
                width, height = w, h
                writer = _open_writer(path, fps, width, height)
            elif (w, h) != (width, height):
                rgb = _resize(rgb, width, height)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return count


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
