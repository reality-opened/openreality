"""Tests for server.ingest.video — hermetic: synthesizes a tiny mp4 with
cv2.VideoWriter (known per-frame content) and checks the extract contract
mirrors server.ingest.dimos_db.extract_frames (frames/<ts>.jpg + manifest),
plus the suffix dispatch in server.oreos.recordings.pipeline.stage_ingest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from server.ingest.video import _next_pts, extract_frames_from_video
from server.oreos.recordings.pipeline import OreosPipeline
from test_ingest_dimos import make_fixture_db

cv2 = pytest.importorskip("cv2")

N_FRAMES = 20
NATIVE_FPS = 12.0


def make_fixture_video(path: Path, n_frames: int = N_FRAMES, fps: float = NATIVE_FPS,
                       size: tuple[int, int] = (64, 48)) -> None:
    """Tiny mp4: frame i is a solid gray level 10*i+5 (known, order-checkable)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened(), "mp4v VideoWriter unavailable in this cv2 build"
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 10 * i + 5, np.uint8)
        w.write(img)
    w.release()


@pytest.fixture()
def fixture_video(tmp_path: Path) -> Path:
    p = tmp_path / "tour.mp4"
    make_fixture_video(p)
    return p


def test_extract_all_frames_contract(fixture_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "ingested"
    # target fps above native -> every frame kept
    manifest = extract_frames_from_video(fixture_video, out, fps=30.0)

    frames = sorted((out / "frames").glob("*.jpg"))
    assert manifest["n_frames"] == len(frames) == N_FRAMES
    # timestamp-named => numeric sort == time sort, strictly increasing
    stems = [float(f.stem) for f in frames]
    assert stems == sorted(stems) and len(set(stems)) == N_FRAMES
    # t0=None -> container time starting at 0
    assert stems[0] == pytest.approx(0.0, abs=1e-6)
    assert stems[-1] == pytest.approx((N_FRAMES - 1) / NATIVE_FPS, abs=0.02)
    # span mirrors dimos: last kept - first kept
    assert manifest["span_s"] == pytest.approx((N_FRAMES - 1) / NATIVE_FPS, abs=0.02)
    # known content survives the decode->JPEG round trip in order
    means = []
    for f in frames:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (48, 64, 3)
        means.append(float(img.mean()))
    assert means == sorted(means)  # gray level 10*i+5 is increasing
    assert means[0] == pytest.approx(5, abs=4) and means[-1] == pytest.approx(195, abs=6)


def test_manifest_mirrors_dimos_contract(fixture_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "ingested"
    manifest = extract_frames_from_video(fixture_video, out, fps=30.0)

    disk = json.loads((out / "ingest_manifest.json").read_text())
    assert disk == manifest
    assert manifest["format"] == "video"
    assert manifest["source"] == "tour.mp4"
    assert manifest["fps_target"] == 30.0
    # no odometry exists in plain video -> reference fields honest, not absent
    assert manifest["odom_stream"] is None
    assert manifest["n_reference_poses"] == 0
    assert manifest["reference_spread_m"] == 0
    assert manifest["reference_usable"] is False
    assert not (out / "reference.txt").exists()
    (stream,) = manifest["streams"]
    assert stream["name"] == "video"
    assert stream["rows"] == N_FRAMES
    assert stream["fps_native"] == pytest.approx(NATIVE_FPS, abs=0.1)
    assert stream["width"] == 64 and stream["height"] == 48
    # exact same top-level keys as the dimos manifest (contract lock)
    assert set(manifest) == {
        "source", "format", "camera_stream", "odom_stream", "fps_target",
        "n_frames", "span_s", "n_reference_poses", "reference_spread_m",
        "reference_usable", "streams",
    }


def test_fps_thinning(fixture_video: Path, tmp_path: Path) -> None:
    # native 12 fps, target 6 -> every other frame; boundary frames must not be
    # dropped by fp noise (same 1e-4 tolerance as the dimos reader)
    manifest = extract_frames_from_video(fixture_video, tmp_path / "half", fps=6.0)
    assert manifest["n_frames"] == N_FRAMES // 2
    # fps=0 -> keep everything
    manifest0 = extract_frames_from_video(fixture_video, tmp_path / "all", fps=0.0)
    assert manifest0["n_frames"] == N_FRAMES


def test_t0_offset(fixture_video: Path, tmp_path: Path) -> None:
    t0 = 1700000000.0
    extract_frames_from_video(fixture_video, tmp_path / "off", fps=30.0, t0=t0)
    stems = sorted(float(f.stem) for f in (tmp_path / "off" / "frames").glob("*.jpg"))
    assert stems[0] == pytest.approx(t0, abs=1e-3)
    assert stems[-1] == pytest.approx(t0 + (N_FRAMES - 1) / NATIVE_FPS, abs=0.02)


def test_portrait_native_resolution(tmp_path: Path) -> None:
    """Portrait (taller-than-wide) video keeps its native resolution, unresized.
    (Rotation *metadata* can't be synthesized via VideoWriter; the module relies
    on cv2's auto-rotation — property verified present — and this test locks the
    no-resize half of the behavior with truly portrait-shaped frames.)"""
    p = tmp_path / "portrait.mp4"
    make_fixture_video(p, size=(48, 64))  # width < height
    out = tmp_path / "ingested"
    extract_frames_from_video(p, out, fps=30.0)
    f = sorted((out / "frames").glob("*.jpg"))[0]
    img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (64, 48, 3)  # portrait preserved, no resize


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_frames_from_video(tmp_path / "nope.mp4", tmp_path / "out")


def test_next_pts_monotonic_fallback() -> None:
    """Container pts is trusted while increasing; 0/stuck readings fall back to
    prev + 1/native_fps so frame names stay strictly increasing."""
    dt = 1.0 / 30.0
    assert _next_pts(0.0, None, dt) == 0.0
    assert _next_pts(100.0, 0.0, dt) == pytest.approx(0.1)
    # backend returns 0 mid-stream -> monotonic fallback
    assert _next_pts(0.0, 0.1, dt) == pytest.approx(0.1 + dt)
    # non-increasing reading -> fallback
    assert _next_pts(50.0, 0.1, dt) == pytest.approx(0.1 + dt)
    # recovery: once pts moves ahead again it is trusted
    assert _next_pts(200.0, 0.1 + dt, dt) == pytest.approx(0.2)


# -- pipeline dispatch --------------------------------------------------------


def test_stage_ingest_dispatch_video(fixture_video: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session_vid"
    p = OreosPipeline(fixture_video, session_dir, fps=30.0)
    res = p.stage_ingest()
    assert res.status == "ok"
    assert res.detail["n_frames"] == N_FRAMES
    assert res.detail["odom_stream"] is None
    assert res.detail["reference_usable"] is False
    manifest = json.loads((session_dir / "ingest_manifest.json").read_text())
    assert manifest["format"] == "video"
    assert len(list((session_dir / "frames").glob("*.jpg"))) == N_FRAMES


@pytest.mark.parametrize("suffix", [".mov", ".MOV", ".avi"])
def test_stage_ingest_dispatch_other_video_suffixes(
    fixture_video: Path, tmp_path: Path, suffix: str
) -> None:
    # dispatch is by suffix (case-insensitive); the bytes are the same mp4
    renamed = tmp_path / f"clip{suffix}"
    renamed.write_bytes(fixture_video.read_bytes())
    session_dir = tmp_path / f"session{suffix}"
    res = OreosPipeline(renamed, session_dir, fps=30.0).stage_ingest()
    assert res.status == "ok"
    manifest = json.loads((session_dir / "ingest_manifest.json").read_text())
    assert manifest["format"] == "video"


def test_stage_ingest_dispatch_db(tmp_path: Path) -> None:
    db = tmp_path / "mini.db"
    make_fixture_db(db, n_frames=12, fps=15.0)
    session_dir = tmp_path / "session_db"
    res = OreosPipeline(db, session_dir, fps=15.0).stage_ingest()
    assert res.status == "ok"
    assert res.detail["n_frames"] == 12
    assert res.detail["odom_stream"] == "odom"
    manifest = json.loads((session_dir / "ingest_manifest.json").read_text())
    assert manifest["format"] == "dimos-memory2-sqlite"


# -- ingest CLI dispatch (must match the pipeline's) ---------------------------


def _run_ingest_cli(monkeypatch, argv: list[str]) -> dict:
    import sys

    from server.ingest import cli as ingest_cli

    printed: list[str] = []
    monkeypatch.setattr(sys, "argv", ["server.ingest.cli", *argv])
    monkeypatch.setattr("builtins.print", lambda *a, **kw: printed.append(str(a[0])))
    ingest_cli.main()
    return json.loads(printed[-1])


def test_ingest_cli_dispatches_video_by_suffix(
    fixture_video: Path, tmp_path: Path, monkeypatch
) -> None:
    """`python -m server.ingest.cli tour.mp4` used to hand the mp4 straight to the
    sqlite reader and die with `sqlite3.DatabaseError: file is not a database`,
    even though server.oreos.recordings.pipeline already dispatched the same argument by
    suffix. The CLI now shares that one rule (video.VIDEO_SUFFIXES)."""
    out = tmp_path / "cli_video"
    manifest = _run_ingest_cli(
        monkeypatch, [str(fixture_video), "--out", str(out), "--fps", "30"]
    )
    assert manifest["format"] == "video"
    assert manifest["n_frames"] == N_FRAMES
    assert len(list((out / "frames").glob("*.jpg"))) == N_FRAMES


def test_ingest_cli_still_dispatches_db(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mini.db"
    make_fixture_db(db, n_frames=12, fps=15.0)
    out = tmp_path / "cli_db"
    manifest = _run_ingest_cli(
        monkeypatch, [str(db), "--out", str(out), "--fps", "15"]
    )
    assert manifest["format"] == "dimos-memory2-sqlite"
    assert manifest["n_frames"] == 12


def test_pipeline_and_cli_share_one_suffix_table() -> None:
    """B5: the suffix tuple was duplicated in pipeline.py while the canonical
    VIDEO_SUFFIXES sat unused in video.py. Both call sites read it now."""
    import inspect

    from server.ingest import cli as ingest_cli
    from server.ingest.video import VIDEO_SUFFIXES
    from server.oreos.recordings import pipeline as pipeline_mod

    assert VIDEO_SUFFIXES == (".mp4", ".mov", ".avi")
    for mod in (pipeline_mod, ingest_cli):
        src = inspect.getsource(mod)
        assert "VIDEO_SUFFIXES" in src
        assert '".mp4", ".mov", ".avi"' not in src, f"{mod.__name__} re-lists the suffixes"


# -- frame-name collisions -----------------------------------------------------


def test_video_frames_written_equals_count(tmp_path: Path, monkeypatch) -> None:
    """B6: two frames whose pts round to the same microsecond must both land on
    disk, so n_frames never exceeds the files (which trips export_targets'
    desync check much later with a misleading message)."""
    from server.ingest import video as video_mod

    out = tmp_path / "collide"
    vid = tmp_path / "clip.mp4"
    make_fixture_video(vid, n_frames=6)

    # force every decoded frame onto the same timestamp
    monkeypatch.setattr(video_mod, "_next_pts", lambda raw_ms, prev, dt: 1.0)

    manifest = video_mod.extract_frames_from_video(vid, out, fps=0)

    files = sorted((out / "frames").glob("*.jpg"))
    assert manifest["n_frames"] == len(files) == 6
    assert len({f.name for f in files}) == 6
    # every name still parses as a float and sorts in write order
    stems = [float(f.stem) for f in files]
    assert stems == sorted(stems)
    assert stems[0] == pytest.approx(1.0)
