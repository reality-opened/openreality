"""CLI: materialize a recording as pipeline inputs.

  python -m server.ingest.cli <recording.db|video.mp4> --out <dir> [--fps 10]
        [--camera-stream color_image]

Accepts either a DimOS memory2 `.db` recording or a plain video (.mp4/.mov/.avi)
and dispatches on the suffix exactly as `server.oreos.recordings.pipeline` does — the two
must agree, so both read `server.ingest.video.VIDEO_SUFFIXES`.

Writes frames/<ts>.jpg + reference.txt (TUM, if an odom stream exists) +
ingest_manifest.json, then prints the manifest. Recon and QC are separate
steps: feed frames/ to the reconstruction pipeline, then score the resulting
trajectory with `python -m server.qc.cli`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from server.ingest.video import VIDEO_SUFFIXES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "recording",
        help="DimOS memory2 .db recording, or a plain video (.mp4/.mov/.avi)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument(
        "--camera-stream",
        default="color_image",
        help="camera stream name (.db recordings only; ignored for video)",
    )
    args = ap.parse_args()

    if Path(args.recording).suffix.lower() in VIDEO_SUFFIXES:
        # A video has no streams and no odometry; --camera-stream does not apply.
        from server.ingest.video import extract_frames_from_video

        manifest = extract_frames_from_video(args.recording, args.out, fps=args.fps)
    else:
        from server.ingest.dimos_db import extract_frames

        manifest = extract_frames(
            args.recording, args.out, fps=args.fps, camera_stream=args.camera_stream
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
