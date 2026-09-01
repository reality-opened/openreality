"""Plain-video ingest adapter (.mp4/.mov/.avi) — phone captures -> pipeline inputs.

Mirrors the output contract of `server.ingest.dimos_db.extract_frames` exactly:
timestamp-named JPEGs under <out_dir>/frames/ plus <out_dir>/ingest_manifest.json
with the same keys, so every downstream stage (upload/recon/score/qc/report/export)
consumes a video session without knowing the source was not a robot recording.
Differences inherent to plain video, encoded in the manifest rather than hidden:
no odometry stream exists, so `odom_stream` is null, `n_reference_poses` is 0 and
`reference_usable` is false (no reference.txt is written — consistency scoring
degrades gracefully, exactly as for a .db with no usable odometry).

Timestamps: frames are named <ts:.6f>.jpg where ts = (t0 or 0.0) + frame_pts and
frame_pts comes from the container via cv2.CAP_PROP_POS_MSEC (read after each
decode, i.e. the pts of the frame just decoded — verified against this build,
OpenCV 5.0/FFmpeg). Some containers/backends return 0 or non-increasing values
mid-stream; those fall back to last_pts + 1/native_fps (monotonic fallback), so
frame names are always strictly increasing and numeric sort == time sort.

Rotation metadata: phones store portrait video as landscape frames + a rotation
tag. OpenCV's FFmpeg backend auto-applies that rotation since 4.5 (the
CAP_PROP_ORIENTATION_AUTO property, present and defaulting to on in this build),
so decoded frames arrive display-oriented; we still set the property explicitly
in case a backend defaults it off. Frames are written at native decoded
resolution — portrait stays portrait, nothing is resized.
"""

from __future__ import annotations

import json
from pathlib import Path

from server.ingest.frame_names import unique_frame_name

# The one dispatch rule: server/oreos/recordings/pipeline.py and server/ingest/cli.py both
# route a recording to this adapter iff its suffix is in here. Keep it here, not
# duplicated at the call sites.
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi")

_FPS_TOL = 1e-4  # same fp-noise tolerance as dimos_db.iter_jpeg_frames


def _next_pts(raw_ms: float, prev_pts: float | None, fallback_dt: float) -> float:
    """Frame pts in seconds from a raw CAP_PROP_POS_MSEC reading, with the
    monotonic fallback: a missing (<=0 for a non-first frame) or non-increasing
    reading is replaced by prev_pts + fallback_dt. Pure function (unit-tested)."""
    pts = raw_ms / 1000.0
    if prev_pts is None:
        return max(pts, 0.0)
    if pts <= prev_pts:
        return prev_pts + fallback_dt
    return pts


def extract_frames_from_video(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 10.0,
    t0: float | None = None,
) -> dict:
    """Materialize a plain video as pipeline inputs: frames/<ts>.jpg + manifest.

    fps is the *target* extraction rate: frames closer than 1/fps to the last
    kept frame are skipped (fps=0 keeps every frame). t0 offsets all timestamps
    (e.g. the capture's wall-clock start); default 0.0 keeps container time.

    Returns the manifest dict (also written to <out_dir>/ingest_manifest.json).
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {video_path}")
    try:
        # Belt-and-braces: ask the backend to apply rotation metadata (default on
        # for FFmpeg since OpenCV 4.5; portrait phone video decodes portrait).
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        fallback_dt = 1.0 / native_fps if native_fps and native_fps > 0 else 1.0 / 30.0
        min_dt = 1.0 / fps if fps else 0.0
        base = t0 if t0 is not None else 0.0

        n_read = 0
        n_kept = 0
        pts = None
        last_kept = float("-inf")
        pts_first = pts_last = None
        stream_pts_first = None
        width = height = 0
        # pts are strictly increasing, but two of them can still round to the same
        # `%.6f` name; without this set the second write silently clobbered the
        # first while n_kept counted both (see server/ingest/frame_names.py).
        written: set[str] = set()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = _next_pts(cap.get(cv2.CAP_PROP_POS_MSEC), pts, fallback_dt)
            n_read += 1
            stream_pts_first = stream_pts_first if stream_pts_first is not None else pts
            if pts - last_kept < min_dt - _FPS_TOL:
                continue
            height, width = frame.shape[:2]  # native decoded (display-oriented)
            ts = base + pts
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok_enc:
                raise RuntimeError(f"JPEG encode failed at pts {pts:.3f}s")
            (frames_dir / unique_frame_name(ts, written)).write_bytes(buf.tobytes())
            n_kept += 1
            pts_first = pts_first if pts_first is not None else pts
            pts_last = pts
            last_kept = pts
    finally:
        cap.release()

    span = round(pts_last - pts_first, 3) if n_kept else 0.0
    manifest = {
        "source": video_path.name,
        "format": "video",
        "camera_stream": "video",
        "odom_stream": None,
        "fps_target": fps,
        "n_frames": n_kept,
        "span_s": span,
        "n_reference_poses": 0,
        "reference_spread_m": 0,
        "reference_usable": False,
        "streams": [
            {
                "name": "video",
                "codec": video_path.suffix.lstrip(".").lower(),
                "rows": n_read,
                "span_s": round(pts - stream_pts_first, 3) if n_read else 0.0,
                "width": width,
                "height": height,
                "fps_native": round(native_fps, 3) if native_fps else None,
            }
        ],
    }
    (out / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
