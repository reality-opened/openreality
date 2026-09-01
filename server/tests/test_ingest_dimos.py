"""Tests for server.ingest.dimos_db — hermetic: builds a miniature memory2 .db
with the exact schema observed in real DimOS recordings (EXP-36-verified)."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from server.ingest.dimos_db import DimosDbReader, extract_frames

cv2 = pytest.importorskip("cv2")


def _tiny_jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def make_fixture_db(path: Path, n_frames: int = 12, fps: float = 15.0,
                    zero_anchors: bool = False, anchors: str | None = None) -> None:
    """Real recordings wrap the JPEG in a small LCM header; emulate with an
    8-byte fingerprint prefix (the reader locates the JPEG by SOI magic).

    `anchors` picks the world-pose-anchor era written into the odom columns
    (`zero_anchors=True` is the legacy spelling of `anchors="zero"`):

      "pose"      the anchors equal the payload pose — the healthy modern era.
      "zero"      all seven columns zeroed except qw=1; odometry lives only in
                  the LCM payload tail (last 56 bytes = ">7d" pose).
      "null"      the columns are SQL NULL — same era, different encoding. The
                  payload is still the only source of odometry.
      "rotating"  zero translation but a REAL (non-identity) rotation: a robot
                  that turns in place while perfectly tracked. The anchors are
                  good and must be used, so the fixture makes the payload
                  disagree (a far-away pose) to prove which source won.
    """
    mode = anchors or ("zero" if zero_anchors else "pose")
    assert mode in {"pose", "zero", "null", "rotating"}
    db = sqlite3.connect(path)
    cur = db.cursor()
    cur.execute("CREATE TABLE _streams (name TEXT, config TEXT)")
    for name, codec, mod in [
        ("color_image", "jpeg", "dimos.msgs.sensor_msgs.Image.Image"),
        ("odom", "lcm", "dimos.msgs.geometry_msgs.PoseStamped.PoseStamped"),
    ]:
        cur.execute(
            "INSERT INTO _streams VALUES (?, ?)",
            (name, json.dumps({"payload_module": mod, "codec_id": codec})),
        )
        cur.execute(
            f'CREATE TABLE "{name}" (id INTEGER PRIMARY KEY, ts REAL, value BLOB,'
            " pose_x REAL, pose_y REAL, pose_z REAL,"
            " pose_qx REAL, pose_qy REAL, pose_qz REAL, pose_qw REAL, tags TEXT)"
        )
        cur.execute(f'CREATE TABLE "{name}_blob" (id INTEGER PRIMARY KEY, data BLOB)')

    t0 = 1700000000.0
    for i in range(n_frames):
        ts = t0 + i / fps
        x = 0.1 * i
        cur.execute(
            'INSERT INTO "color_image" (id, ts, pose_x, pose_y, pose_z,'
            " pose_qx, pose_qy, pose_qz, pose_qw) VALUES (?,?,?,?,?,?,?,?,?)",
            (i + 1, ts, x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
        )
        wrapped = struct.pack(">q", 0x535CFACE1F4F5717) + _tiny_jpeg(i)
        cur.execute('INSERT INTO "color_image_blob" VALUES (?, ?)', (i + 1, wrapped))

    for i in range(n_frames * 2):  # odom at 2x camera rate
        ts = t0 + i / (2 * fps)
        pose = (0.05 * i, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0)
        if mode == "zero":
            anchor: tuple = (0.0,) * 6 + (1.0,)
        elif mode == "null":
            anchor = (None,) * 7
        elif mode == "rotating":
            # in place, yawing: translation stays 0, quaternion sweeps
            yaw = 0.02 * i
            anchor = (0.0, 0.0, 0.0, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
            pose = (99.0, 99.0, 99.0, 0.0, 0.0, 0.0, 1.0)  # payload must NOT win
        else:
            anchor = pose
        cur.execute(
            'INSERT INTO "odom" (id, ts, pose_x, pose_y, pose_z,'
            " pose_qx, pose_qy, pose_qz, pose_qw) VALUES (?,?,?,?,?,?,?,?,?)",
            (i + 1, ts, *anchor),
        )
        # payload: fake LCM header + trailing ">7d" pose (the real layout's tail)
        blob = struct.pack(">q", 0x1234) + b"hdr" + struct.pack(">7d", *pose)
        cur.execute('INSERT INTO "odom_blob" VALUES (?, ?)', (i + 1, blob))
    db.commit()
    db.close()


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    p = tmp_path / "mini.db"
    make_fixture_db(p)
    return p


def test_streams_discovery(fixture_db: Path) -> None:
    with DimosDbReader(fixture_db) as r:
        infos = {s.name: s for s in r.streams()}
    assert infos["color_image"].codec_id == "jpeg"
    assert infos["color_image"].n_rows == 12
    assert infos["odom"].n_rows == 24
    assert infos["odom"].span_s > 0


def test_find_stream_fallback(fixture_db: Path) -> None:
    with DimosDbReader(fixture_db) as r:
        assert r.find_stream("go2_odom", "odom") == "odom"
        with pytest.raises(KeyError):
            r.find_stream("nope", "nada")


def test_jpeg_frames_decode_and_min_dt(fixture_db: Path) -> None:
    with DimosDbReader(fixture_db) as r:
        frames = list(r.iter_jpeg_frames("color_image"))
        assert len(frames) == 12
        for ts, jpeg in frames:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert img is not None and img.shape == (24, 32, 3)
        # min_dt at half the camera rate keeps every other frame
        thinned = list(r.iter_jpeg_frames("color_image", min_dt=2 / 15.0))
        assert len(thinned) == 6


def test_trajectory_tum_pose_columns(fixture_db: Path) -> None:
    with DimosDbReader(fixture_db) as r:
        lines = r.trajectory_tum("odom")
    assert len(lines) == 24
    first = lines[0].split()
    assert len(first) == 8
    assert float(first[1]) == pytest.approx(0.0)
    assert float(lines[-1].split()[1]) == pytest.approx(0.05 * 23)


def test_extract_frames_end_to_end(fixture_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "ingested"
    manifest = extract_frames(fixture_db, out, fps=15.0)
    frames = sorted((out / "frames").glob("*.jpg"))
    assert manifest["n_frames"] == len(frames) == 12
    assert manifest["odom_stream"] == "odom"
    assert manifest["n_reference_poses"] == 24
    # frames are timestamp-named => numerically sorted == time-sorted
    stems = [float(f.stem) for f in frames]
    assert stems == sorted(stems)
    # payload passthrough: bytes on disk are the original JPEG, not a re-encode
    img = cv2.imdecode(np.fromfile(frames[0], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    ref = (out / "reference.txt").read_text()
    assert "NOT ground truth" in ref
    manifest_disk = json.loads((out / "ingest_manifest.json").read_text())
    assert manifest_disk["format"] == "dimos-memory2-sqlite"


def test_zero_anchor_era_payload_fallback(tmp_path: Path) -> None:
    """Recording era with zeroed anchor columns: odometry must come from the
    LCM payload tail (EXP-36 extension-sweep discovery)."""
    p = tmp_path / "zeroed.db"
    make_fixture_db(p, zero_anchors=True)
    with DimosDbReader(p) as r:
        lines = r.trajectory_tum("odom")
    assert len(lines) == 24
    assert float(lines[-1].split()[1]) == pytest.approx(0.05 * 23)  # from payload
    out = tmp_path / "ingested_zeroed"
    manifest = extract_frames(p, out, fps=15.0)
    assert manifest["reference_usable"] is True
    assert manifest["reference_spread_m"] == pytest.approx(0.05 * 23, abs=1e-3)


def test_reference_usability_flag(fixture_db: Path, tmp_path: Path) -> None:
    manifest = extract_frames(fixture_db, tmp_path / "m", fps=15.0)
    assert manifest["reference_usable"] is True
    assert manifest["reference_spread_m"] > 0.5


def test_real_exp36_reference_note() -> None:
    """The EXP-36 provenance note in the module docstring stays load-bearing:
    pose columns were verified bit-exact against the dimos decoder there."""
    from server.ingest import dimos_db

    assert "bit-exact" in (dimos_db.__doc__ or "")


# -- world-pose anchor eras (B7) ----------------------------------------------


def test_null_anchor_era_payload_fallback(tmp_path: Path) -> None:
    """NULL pose columns are the same "anchors carry nothing" era as all-zero.

    iter_observations yields pose=None for NULL columns, and trajectory_tum used
    to `continue` past those rows *before* deciding whether to fall back — so the
    row list was empty, the fallback never ran, and a perfectly good odometry
    payload was reported as "no reference at all".
    """
    p = tmp_path / "nulls.db"
    make_fixture_db(p, anchors="null")

    with DimosDbReader(p) as r:
        obs = list(r.iter_observations("odom"))
        assert all(o.pose is None for o in obs), "fixture must really write NULLs"
        lines = r.trajectory_tum("odom")

    assert len(lines) == 24, "the payload tail must be decoded, not dropped"
    assert float(lines[-1].split()[1]) == pytest.approx(0.05 * 23)

    out = tmp_path / "ingested_null"
    manifest = extract_frames(p, out, fps=15.0)
    assert manifest["n_reference_poses"] == 24
    assert manifest["reference_usable"] is True
    assert manifest["reference_spread_m"] == pytest.approx(0.05 * 23, abs=1e-3)


def test_stationary_but_tracked_robot_keeps_its_anchors(tmp_path: Path) -> None:
    """A robot that yaws in place has zero translation and PERFECTLY GOOD anchors.

    The fallback used to test translation only (`obs.pose[:3]`), so it fired here
    and threw the real anchors away in favour of the payload. The full pose is
    considered now: zero translation + a non-identity rotation is information.
    """
    p = tmp_path / "spin.db"
    make_fixture_db(p, anchors="rotating")

    with DimosDbReader(p) as r:
        lines = r.trajectory_tum("odom")

    assert len(lines) == 24
    for line in lines:
        v = [float(x) for x in line.split()]
        assert v[1:4] == [0.0, 0.0, 0.0], "translation must stay at the origin"
        assert v[1] != 99.0, "the payload pose must NOT have been substituted"
    # the yaw actually varies -> the anchors were used, not blanked
    qz = [float(line.split()[6]) for line in lines]
    assert max(qz) > min(qz)


def test_all_zero_anchors_still_use_the_payload(tmp_path: Path) -> None:
    """Regression guard for the era the fallback was written for (commit 8378788):
    zero translation AND identity rotation carries no information, so the payload
    must still win."""
    p = tmp_path / "zeros.db"
    make_fixture_db(p, anchors="zero")
    with DimosDbReader(p) as r:
        lines = r.trajectory_tum("odom")
    assert float(lines[-1].split()[1]) == pytest.approx(0.05 * 23)


def test_anchor_informativeness_rules() -> None:
    from server.ingest.dimos_db import _anchor_is_informative

    assert _anchor_is_informative(None) is False                       # NULL columns
    assert _anchor_is_informative((0, 0, 0, 0, 0, 0, 0)) is False       # never written
    assert _anchor_is_informative((0, 0, 0, 0, 0, 0, 1)) is False       # identity
    assert _anchor_is_informative((0, 0, 0, 0, 0, 0, -1)) is False      # identity (-w)
    assert _anchor_is_informative((0.1, 0, 0, 0, 0, 0, 1)) is True      # translated
    assert _anchor_is_informative((0, 0, 0, 0, 0, 0.38, 0.92)) is True  # yawed in place


# -- frame-name collisions (B6) ------------------------------------------------


def test_duplicate_timestamps_do_not_overwrite_frames(tmp_path: Path) -> None:
    """Two observations under a microsecond apart render to the same `%.6f` name.

    The second used to overwrite the first while n_frames counted both, so the
    manifest claimed more frames than existed — surfacing much later as
    export_targets' frame/trajectory desync error, which blames the wrong stage.
    """
    p = tmp_path / "dupes.db"
    make_fixture_db(p, n_frames=4, fps=15.0)

    # rewrite the camera timestamps so rows 1&2 and 3&4 collide at 1e-6 precision
    db = sqlite3.connect(p)
    base = 1700000000.0
    for row_id, ts in [(1, base), (2, base + 1e-9), (3, base + 5.0), (4, base + 5.0)]:
        db.execute('UPDATE "color_image" SET ts = ? WHERE id = ?', (ts, row_id))
    db.commit()
    db.close()

    out = tmp_path / "ingested_dupes"
    manifest = extract_frames(p, out, fps=0)  # fps=0 -> keep every frame

    files = sorted((out / "frames").glob("*.jpg"))
    assert manifest["n_frames"] == len(files) == 4, "count must equal files on disk"
    assert len({f.name for f in files}) == 4
    # names stay plain parseable decimals (core's sort_images_by_number,
    # export_targets' float(stem) and live_probe's regex all depend on it)
    stems = [float(f.stem) for f in files]
    assert stems == sorted(stems)
    # the nudge is exactly one microsecond in the NAME (the float round-trip of an
    # epoch-scale timestamp cannot resolve 1e-6 — one float64 ULP here is ~2.4e-7)
    def _micros(stem: str) -> int:
        whole, frac = stem.split(".")
        return int(whole) * 1_000_000 + int(frac)

    names = [f.stem for f in files]
    assert _micros(names[1]) - _micros(names[0]) == 1
    # and the bytes of every original frame survived (no clobber)
    assert len({f.read_bytes() for f in files}) == 4


def test_unique_frame_name_is_deterministic_and_ordered() -> None:
    from server.ingest.frame_names import unique_frame_name

    taken: set[str] = set()
    names = [unique_frame_name(12.5, taken) for _ in range(3)]
    assert names == ["12.500000.jpg", "12.500001.jpg", "12.500002.jpg"]
    assert [float(Path(n).stem) for n in names] == sorted(
        float(Path(n).stem) for n in names
    )
    # same inputs, fresh set -> same outputs
    assert [unique_frame_name(12.5, set()) for _ in range(1)] == ["12.500000.jpg"]


# -- SQL identifier validation (B8) --------------------------------------------


def test_malicious_stream_name_is_rejected_not_interpolated(tmp_path: Path) -> None:
    """Table/stream names are f-string-interpolated into SQL (sqlite cannot
    parameterise identifiers) and are NOT trusted: they come from the recording's
    own _streams table or straight off the CLI. A name containing a double quote
    closes the identifier and everything after it executes."""
    p = tmp_path / "evil.db"
    make_fixture_db(p)

    db = sqlite3.connect(p)
    db.execute("CREATE TABLE victim (x INTEGER)")
    db.execute("INSERT INTO victim VALUES (1)")
    evil = 'odom" ; DROP TABLE victim ; SELECT * FROM "odom'
    db.execute(
        "INSERT INTO _streams VALUES (?, ?)",
        (evil, json.dumps({"payload_module": "x", "codec_id": "lcm"})),
    )
    db.commit()
    db.close()

    with DimosDbReader(p) as r:
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            list(r.iter_observations(evil))
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            r.streams()  # the poisoned _streams row is refused on discovery too

    check = sqlite3.connect(p)
    assert check.execute("SELECT COUNT(*) FROM victim").fetchone()[0] == 1
    check.close()


@pytest.mark.parametrize(
    "name", ["color_image", "go2_odom", "/camera/color", "depth.raw", "cam-0"]
)
def test_real_stream_names_are_accepted(name: str) -> None:
    from server.ingest.dimos_db import _quote_identifier

    assert _quote_identifier(name) == f'"{name}"'


@pytest.mark.parametrize(
    # the double quote is the one that actually breaks out of a quoted identifier;
    # the rest are refused because the allowlist is deliberately narrow
    "name", ['a"b', "a;b", "a b", "", "a'b", "a\nb", "a`b", "a\\b", "a\x00b"]
)
def test_unsafe_stream_names_are_refused(name: str) -> None:
    from server.ingest.dimos_db import _quote_identifier

    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        _quote_identifier(name)
