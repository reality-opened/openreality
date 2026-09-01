"""Dependency-free reader for DimOS "memory2" SQLite recordings (`.db`).

Schema (reverse-engineered and verified bit-exact against the dimos decoder in
EXP-36 — platform/experiments/exp36_oreos_dimos_spike/):

  _streams(name, config)        config JSON: payload_module, codec_id ("jpeg"/"lcm"...)
  <stream>(id, ts, value, pose_x..pose_qw, tags)
                                every observation carries a world-pose anchor in the
                                pose_* columns (their tf lookup at record time)
  <stream>_blob(id, data)       large payloads; codec "jpeg" wraps a plain JPEG
                                inside a small LCM header — the JPEG is located by
                                its SOI magic (\\xff\\xd8\\xff) and runs to the blob end

Two verified facts this module relies on (test-guarded):
  * for pose-typed streams (e.g. `odom`) the pose_* columns equal the decoded
    message payload exactly, so no LCM decoding is needed for trajectories;
  * for "jpeg"-codec streams the blob's JPEG payload is a self-contained image
    readable by cv2.imdecode, so frames pass through without re-encoding.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from server.ingest.frame_names import unique_frame_name

_JPEG_SOI = b"\xff\xd8\xff"

# SQLite cannot parameterise identifiers, so table names have to be interpolated
# into the SQL text — which makes them an injection surface. They are not
# trusted input: a stream name comes from the recording file's own `_streams`
# table (a third-party .db an operator was handed) or straight off the CLI. A
# name containing `"` closes the quoted identifier and everything after it is
# executed. Allowlist the characters real dimos stream names actually use
# (`color_image`, `go2_odom`, `/camera/color`, `depth.raw`, `cam-0`) and reject
# the rest with a clear error rather than trying to escape.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_/.-]+$")


def _quote_identifier(name: str) -> str:
    """Validate a table/stream name and return it double-quoted for SQL."""
    if not isinstance(name, str) or not _SAFE_IDENTIFIER.match(name):
        raise ValueError(
            f"unsafe SQL identifier {name!r}: stream/table names must match "
            f"{_SAFE_IDENTIFIER.pattern} (letters, digits, _ / . -)"
        )
    return f'"{name}"'


@dataclass(frozen=True)
class StreamInfo:
    name: str
    payload_module: str
    codec_id: str
    n_rows: int
    ts_min: float | None
    ts_max: float | None

    @property
    def span_s(self) -> float:
        if self.ts_min is None or self.ts_max is None:
            return 0.0
        return self.ts_max - self.ts_min


@dataclass(frozen=True)
class Observation:
    ts: float
    pose: tuple[float, float, float, float, float, float, float] | None
    """World-pose anchor (tx, ty, tz, qx, qy, qz, qw) or None if unset."""
    payload: bytes | None
    """Raw blob payload (None when the observation carried no blob)."""


def _anchor_is_informative(pose: tuple[float, ...] | None, eps: float = 1e-9) -> bool:
    """Does this world-pose anchor carry any actual tracking information?

    False for an unset anchor (NULL columns -> None), for an all-zero row, and for
    "at the origin, unrotated" (zero translation + identity or all-zero quaternion)
    — the shapes the zeroed-anchor recording eras write. True as soon as the robot
    is somewhere else OR oriented somehow, so a stationary-but-tracked robot is not
    mistaken for an unset anchor.
    """
    if pose is None:
        return False
    tx, ty, tz, qx, qy, qz, qw = pose
    if any(abs(v) > eps for v in (tx, ty, tz)):
        return True
    if all(abs(v) <= eps for v in (qx, qy, qz)):
        # (0,0,0,0) = never written; (0,0,0,±1) = identity rotation
        return False
    return True


class DimosDbReader:
    """Read-only access to a memory2 recording. Not thread-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "DimosDbReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- discovery ----------------------------------------------------------

    def streams(self) -> list[StreamInfo]:
        out = []
        for name, config in self._db.execute("SELECT name, config FROM _streams"):
            cfg = json.loads(config)
            try:
                n, lo, hi = self._db.execute(
                    f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM {_quote_identifier(name)}"
                ).fetchone()
            except sqlite3.OperationalError:
                n, lo, hi = 0, None, None
            out.append(
                StreamInfo(
                    name=name,
                    payload_module=cfg.get("payload_module", ""),
                    codec_id=cfg.get("codec_id", ""),
                    n_rows=n,
                    ts_min=lo,
                    ts_max=hi,
                )
            )
        return out

    def stream_names(self) -> list[str]:
        return [s.name for s in self.streams()]

    def find_stream(self, *candidates: str) -> str:
        """First existing stream among candidates (recording eras rename streams,
        e.g. odom vs go2_odom — same fallback dimos itself uses)."""
        names = set(self.stream_names())
        for c in candidates:
            if c in names:
                return c
        raise KeyError(f"none of {candidates!r} in {sorted(names)}")

    # -- iteration ----------------------------------------------------------

    def iter_observations(self, stream: str) -> Iterator[Observation]:
        table = _quote_identifier(stream)
        blob_table = _quote_identifier(f"{stream}_blob")
        q = (
            "SELECT o.ts, o.pose_x, o.pose_y, o.pose_z, o.pose_qx, o.pose_qy,"
            f" o.pose_qz, o.pose_qw, b.data FROM {table} o"
            f" LEFT JOIN {blob_table} b ON b.id = o.id ORDER BY o.ts"
        )
        for ts, px, py, pz, qx, qy, qz, qw, blob in self._db.execute(q):
            pose = None
            if px is not None:
                pose = (px, py, pz, qx, qy, qz, qw)
            yield Observation(ts=float(ts), pose=pose, payload=blob)

    def iter_jpeg_frames(
        self, stream: str = "color_image", min_dt: float = 0.0
    ) -> Iterator[tuple[float, bytes]]:
        """Yield (ts, jpeg_bytes); frames closer than min_dt to the last yielded
        frame are skipped. The JPEG payload is passed through unmodified.

        The 1e-4 s tolerance matters: recordings carry epoch-scale float64
        timestamps (~1e9 s), where subtraction noise reaches ~4e-7 s — a frame
        recorded exactly min_dt after the last must not be dropped by fp error.
        """
        last = float("-inf")
        for obs in self.iter_observations(stream):
            if obs.payload is None or obs.ts - last < min_dt - 1e-4:
                continue
            i = obs.payload.find(_JPEG_SOI)
            if i < 0:
                continue
            yield obs.ts, obs.payload[i:]
            last = obs.ts

    def trajectory_tum(self, stream: str) -> list[str]:
        """Trajectory of a pose stream as TUM lines (ts tx ty tz qx qy qz qw).

        Primary source: the world-pose anchor columns. Recording-era fallback:
        some eras leave the anchors uninformative and carry odometry only in the
        LCM payload — whose last 56 bytes are ">7d" = (x,y,z,qx,qy,qz,qw),
        verified bit-exact against the dimos decoder (EXP-36 extension sweep).

        "Uninformative" is deliberately broader than "all zero", and the test is
        deliberately narrower than "translation is zero" — both directions were
        wrong before:
          * NULL pose columns yield ``Observation.pose is None``. Those rows used
            to be dropped *before* the fallback check, so a NULL-pose era left
            `rows` empty, the fallback never ran, and the recording reported no
            odometry at all even though the payload carried it.
          * the check looked only at translation, so a robot that rotated in place
            but never translated (a perfectly well-tracked stationary recording)
            tripped the fallback and had its good anchors thrown away. The full
            pose is considered now: a zero translation with a real rotation is
            information, a zero/identity rotation with it is not.
        """
        import struct

        rows: list[tuple[float, tuple[float, ...] | None, bytes | None]] = []
        anchors_informative = False
        for obs in self.iter_observations(stream):
            rows.append((obs.ts, obs.pose, obs.payload))
            if _anchor_is_informative(obs.pose):
                anchors_informative = True

        posed: list[tuple[float, tuple[float, ...]]] = []
        if not anchors_informative:
            posed = [
                (ts, struct.unpack(">7d", payload[-56:]))
                for ts, _pose, payload in rows
                if payload and len(payload) >= 56
            ]
        if not posed:
            posed = [(ts, pose) for ts, pose, _ in rows if pose is not None]
        return [
            f"{ts:.6f} " + " ".join(f"{v:.6f}" for v in pose) for ts, pose in posed
        ]


def extract_frames(
    db_path: str | Path,
    out_dir: str | Path,
    fps: float = 10.0,
    camera_stream: str = "color_image",
    odom_candidates: tuple[str, ...] = ("go2_odom", "odom"),
) -> dict:
    """Materialize a recording as pipeline inputs: frames/<ts>.jpg (payload
    passthrough, no re-encode) + reference.txt (TUM, base frame) + manifest.

    Returns the manifest dict (also written to <out_dir>/ingest_manifest.json).
    """
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with DimosDbReader(db_path) as r:
        streams = r.streams()
        n_frames = 0
        ts_first = ts_last = None
        # Sub-microsecond-apart observations render to the same `%.6f` name; without
        # this set the second silently overwrote the first while n_frames counted
        # both. See server/ingest/frame_names.py for why the fix is a µs nudge and
        # not a filename suffix.
        written: set[str] = set()
        for ts, jpeg in r.iter_jpeg_frames(camera_stream, min_dt=1.0 / fps if fps else 0.0):
            (frames_dir / unique_frame_name(ts, written)).write_bytes(jpeg)
            n_frames += 1
            ts_first = ts_first if ts_first is not None else ts
            ts_last = ts

        ref_lines: list[str] = []
        odom_stream = None
        try:
            odom_stream = r.find_stream(*odom_candidates)
            ref_lines = r.trajectory_tum(odom_stream)
        except KeyError:
            pass
        ref_spread = 0.0
        if ref_lines:
            xs, ys, zs = [], [], []
            for line in ref_lines:
                v = line.split()
                xs.append(float(v[1])); ys.append(float(v[2])); zs.append(float(v[3]))
            ref_spread = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            (out / "reference.txt").write_text(
                "# odometry of stream "
                f"'{odom_stream}' (robot's own estimator, NOT ground truth)\n"
                + "\n".join(ref_lines)
                + "\n"
            )

    manifest = {
        "source": str(Path(db_path).name),
        "format": "dimos-memory2-sqlite",
        "camera_stream": camera_stream,
        "odom_stream": odom_stream,
        "fps_target": fps,
        "n_frames": n_frames,
        "span_s": round(ts_last - ts_first, 3) if n_frames else 0.0,
        "n_reference_poses": len(ref_lines),
        "reference_spread_m": round(ref_spread, 4),
        "reference_usable": bool(ref_lines) and ref_spread > 0.05,
        "streams": [
            {"name": s.name, "codec": s.codec_id, "rows": s.n_rows, "span_s": round(s.span_s, 3)}
            for s in streams
        ],
    }
    (out / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
