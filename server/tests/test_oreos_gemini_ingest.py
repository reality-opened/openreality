"""W1 gemini-glue tests (docs/demo-2026-07 design/shell.md §4c) — GPU-free.

Covers ``scripts/demo_ingest_gemini.py``:

  * frame_map correctness on a synthetic capture dir (numeric ts ordering,
    fps resolution ladder, single-frame refusal, depth manifest);
  * the ffmpeg invocation shape (recorded fake ``run``, no real ffmpeg);
  * the upload/poll client against a real localhost mock broker (headers,
    streamed body, job-stage polling, receipt idempotency, failure paths);
  * ratio math + CoV gate on synthetic depth — unit-level, no network: the
    z-buffer rasterizer, block-median depth, grid selection, anchor-point
    picking, and (when a core checkout/install is reachable) exact-ratio
    recovery through the REAL ``vggt_slam.metric_anchor.frame_ratio``;
  * the full ``anchor`` flow end-to-end against the mock broker (fetches →
    verified ratio → gate → POST /anchor payload → provenance PUT), plus the
    CoV-refusal path with the manual-anchor instruction.

Core-dependent tests skip cleanly when neither an installed core nor the
platform-tree ``core/`` checkout is available (CI clones server standalone) —
same posture as the export tests' ``pytest.importorskip`` pattern.
"""

from __future__ import annotations

import importlib.util
import json
import re
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# module under test (scripts/ is not a package — load by file path)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_ingest_gemini.py"
_spec = importlib.util.spec_from_file_location("demo_ingest_gemini", _SCRIPT)
gem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gem)


@pytest.fixture(scope="module")
def ma():
    """The real core ``metric_anchor`` module via the script's own ladder, or
    skip (CI without core)."""
    try:
        return gem.load_metric_anchor_module()
    except ImportError as exc:
        pytest.skip(f"core metric_anchor unavailable: {exc}")


# ---------------------------------------------------------------------------
# synthetic capture dirs
# ---------------------------------------------------------------------------

_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcffff3f030005fe02fea731818d0000000049454e44ae426082"
)


def _make_capture(tmp_path: Path, stems: list[str], *, depth_for=None,
                  meta=None, intrinsics="default") -> Path:
    cap = tmp_path / "captures" / "scene1" / "gemini"
    (cap / "color").mkdir(parents=True)
    for s in stems:
        (cap / "color" / f"{s}.png").write_bytes(_PNG_1x1)
    if depth_for is not None:
        (cap / "depth").mkdir()
        for s in depth_for:
            (cap / "depth" / f"{s}.png").write_bytes(_PNG_1x1 * 2)
    if intrinsics == "default":
        intrinsics = {"color": {"width": 1920, "height": 1080, "fx": 1000.0,
                                "fy": 1000.0, "cx": 960.0, "cy": 540.0},
                      "depth": {"width": 1280, "height": 800, "fx": 614.0,
                                "fy": 614.0, "cx": 640.0, "cy": 400.0},
                      "depth_scale_mm_per_unit": 1.0}
    if intrinsics is not None:
        (cap / "intrinsics.json").write_text(json.dumps(intrinsics))
    if meta is not None:
        (cap / "capture_meta.json").write_text(json.dumps(meta))
    return cap


_SWEEP_META = {"capture": {"mode": "sweep", "seconds": 10.0, "n_frames": 150,
                           "dropped_frames": 0, "started_utc": "t"},
               "device": {"name": "Orbbec Gemini 2", "serial": "S",
                          "firmware": "F"}}


# ---------------------------------------------------------------------------
# frame_map + capture discovery
# ---------------------------------------------------------------------------


def test_numeric_ts_sort_beats_lexicographic():
    stems = ["1000.000001", "999.000001", "1000.000000"]
    assert gem.numeric_ts_sort(stems) == ["999.000001", "1000.000000",
                                          "1000.000001"]
    assert sorted(stems) != gem.numeric_ts_sort(stems)  # the trap is real


def test_discover_capture_sorts_and_names_scene(tmp_path):
    cap = _make_capture(tmp_path, ["1000.000001", "999.000001"])
    c = gem.discover_capture(cap)
    assert c["scene"] == "scene1"
    assert c["ts"] == ["999.000001", "1000.000001"]
    assert c["intrinsics"]["depth"]["width"] == 1280


def test_discover_capture_rejects_non_capture_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        gem.discover_capture(tmp_path / "empty")


def test_fps_resolution_ladder():
    assert gem.resolve_fps(_SWEEP_META, None) == (15.0, "capture_meta.capture.n_frames/seconds")
    explicit = {"capture": {"fps": 12.5, "seconds": 10.0, "n_frames": 150}}
    assert gem.resolve_fps(explicit, None) == (12.5, "capture_meta.capture.fps")
    assert gem.resolve_fps(explicit, 24.0) == (24.0, "--fps")
    snapshot = {"capture": {"mode": "snapshot", "seconds": 0.0, "n_frames": 1}}
    assert gem.resolve_fps(snapshot, None) == (gem.DEFAULT_FPS, "default")
    assert gem.resolve_fps(None, None) == (gem.DEFAULT_FPS, "default")


def test_frame_map_shape():
    ts = ["1.000000", "2.000000"]
    fm = gem.build_frame_map("s", Path("/cap"), ts, 15.0, "default")
    assert fm["kind"] == "gemini2_frame_map"
    assert fm["frame_count"] == 2 and fm["ts"] == ts
    assert "source_frame_id" in fm["note"]


def test_depth_manifest_counts_and_missing(tmp_path):
    stems = [f"{i}.000000" for i in range(1, 11)]
    cap = _make_capture(tmp_path, stems, depth_for=stems[:7])
    c = gem.discover_capture(cap)
    doc = gem.build_depth_manifest(c["depth_dir"], c["ts"], max_entries=4)
    assert doc["color_frames"] == 10 and doc["depth_present"] == 7
    assert doc["missing_for_color"] == [f"{i}.000000" for i in (8, 9, 10)]
    assert doc["entries"], "subsample entries expected"
    assert all(e["bytes"] > 0 for e in doc["entries"])
    # last present frame always included
    assert doc["entries"][-1]["ts"] == "7.000000"


# ---------------------------------------------------------------------------
# assemble — ffmpeg invocation shape (mocked subprocess) + refusals
# ---------------------------------------------------------------------------


class _FakeRun:
    """Records the command and materializes the ffmpeg output file."""

    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"fake-mp4")
        return types.SimpleNamespace(returncode=0, stderr="")


@pytest.fixture()
def no_real_ffmpeg(monkeypatch):
    monkeypatch.setattr(gem.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(gem, "probe_frame_count", lambda p: None)


def test_assemble_ffmpeg_shape_and_outputs(tmp_path, no_real_ffmpeg):
    stems = [f"{1000 + i}.000000" for i in range(40)]
    cap = _make_capture(tmp_path, stems, depth_for=stems, meta=_SWEEP_META)
    c = gem.discover_capture(cap)
    fps, src = gem.resolve_fps(c["capture_meta"], None)
    out = cap / "ingest"
    fake = _FakeRun()
    summary = gem.assemble(c, out, fps, src, allow_single=False, run=fake)

    assert len(fake.cmds) == 1
    cmd = fake.cmds[0]
    assert cmd[0] == "ffmpeg" and cmd[-1] == str(out / "capture.mp4")
    assert cmd[cmd.index("-framerate") + 1] == "15"
    assert cmd[cmd.index("-i") + 1].endswith("%06d.png")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"

    fm = json.loads((out / "frame_map.json").read_text())
    assert fm["ts"] == stems and fm["fps"] == 15.0
    assert (out / "intrinsics.json").is_file()
    dm = json.loads((out / "depth_manifest.json").read_text())
    assert dm["depth_present"] == 40
    assert summary["frames"] == 40 and summary["depth_present"] == 40


def test_assemble_stages_frames_in_numeric_order(tmp_path, no_real_ffmpeg,
                                                 monkeypatch):
    # 999.x must stage BEFORE 1000.x — capture the symlink targets in order.
    stems = ["1000.000001", "999.000001"]
    cap = _make_capture(tmp_path, stems)
    c = gem.discover_capture(cap)
    staged = []
    real_symlink = gem.os.symlink

    def spy_symlink(src, dst):
        staged.append((Path(dst).name, Path(src).name))
        real_symlink(src, dst)

    monkeypatch.setattr(gem.os, "symlink", spy_symlink)
    gem.assemble(c, cap / "ingest", 15.0, "default", allow_single=False,
                 run=_FakeRun())
    assert staged == [("000000.png", "999.000001.png"),
                      ("000001.png", "1000.000001.png")]


def test_assemble_refuses_single_frame_honestly(tmp_path, no_real_ffmpeg,
                                                capsys):
    cap = _make_capture(tmp_path, ["1785387732.612805"])
    c = gem.discover_capture(cap)
    with pytest.raises(SystemExit) as exc:
        gem.assemble(c, cap / "ingest", 15.0, "default", allow_single=False,
                     run=_FakeRun())
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "single frame" in err and "sweep" in err
    assert not (cap / "ingest" / "capture.mp4").exists()


def test_assemble_allow_single_overrides(tmp_path, no_real_ffmpeg):
    cap = _make_capture(tmp_path, ["1785387732.612805"])
    c = gem.discover_capture(cap)
    summary = gem.assemble(c, cap / "ingest", 15.0, "default",
                           allow_single=True, run=_FakeRun())
    assert summary["frames"] == 1
    assert (cap / "ingest" / "frame_map.json").is_file()


def test_assemble_refuses_on_probe_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(gem.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(gem, "probe_frame_count", lambda p: 999)  # wrong
    stems = [f"{i}.000000" for i in range(1, 41)]
    cap = _make_capture(tmp_path, stems)
    c = gem.discover_capture(cap)
    with pytest.raises(SystemExit):
        gem.assemble(c, cap / "ingest", 15.0, "default", allow_single=False,
                     run=_FakeRun())


# ---------------------------------------------------------------------------
# mock broker (real localhost HTTP server — the client speaks stdlib urllib)
# ---------------------------------------------------------------------------


class _MockBroker:
    """Programmable endpoints + request capture. ``routes`` maps
    ``(method, path)`` → (status, bytes|dict) or a callable(handler, body)."""

    def __init__(self):
        self.routes = {}
        self.requests = []  # (method, path, headers-dict, body-bytes)

    def start(self):
        broker = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                broker.requests.append(
                    (method, self.path, dict(self.headers), body))
                route = broker.routes.get((method, self.path))
                if route is None:
                    payload = json.dumps({"error": "not_found"}).encode()
                    status = 404
                elif callable(route):
                    status, payload = route(self, body)
                    if isinstance(payload, dict):
                        payload = json.dumps(payload).encode()
                else:
                    status, payload = route
                    if isinstance(payload, dict):
                        payload = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

            def do_PUT(self):
                self._serve("PUT")

            def log_message(self, *a):  # keep pytest output clean
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def broker():
    b = _MockBroker()
    b.url = b.start()
    yield b
    b.stop()


@pytest.fixture()
def fast_poll(monkeypatch):
    monkeypatch.setattr(gem.time, "sleep", lambda s: None)


def _bundle(tmp_path: Path, n=40) -> Path:
    stems = [f"{1000 + i}.000000" for i in range(n)]
    cap = _make_capture(tmp_path, stems, depth_for=stems, meta=_SWEEP_META)
    out = cap / "ingest"
    out.mkdir()
    (out / "capture.mp4").write_bytes(b"\x00mp4-bytes\x00" * 128)
    fm = gem.build_frame_map("scene1", cap, stems, 15.0, "default")
    (out / "frame_map.json").write_text(json.dumps(fm))
    (out / "intrinsics.json").write_text((cap / "intrinsics.json").read_text())
    return out


def test_upload_headers_body_poll_and_receipt(tmp_path, broker, fast_poll):
    out = _bundle(tmp_path)
    polls = {"n": 0}

    def jobs_route(handler, body):
        polls["n"] += 1
        seq = [("queued", "upload"), ("running", "recon"), ("done", "done")]
        status, stage = seq[min(polls["n"] - 1, len(seq) - 1)]
        return 200, {"job_id": "J1", "scan_id": "S1", "status": status,
                     "stage": stage, "elapsed_s": 42, "point_count": 5,
                     "has_splat": True, "has_trajectory": True,
                     "keyframe_count": 8}

    broker.routes[("POST", "/api/workspace/ingest/video")] = (
        202, {"job_id": "J1", "scan_id": "S1"})
    broker.routes[("GET", "/api/workspace/jobs/J1")] = jobs_route

    rc = gem.main(["upload", "--bundle-dir", str(out), "--broker", broker.url,
                   "--token", "tok-123", "--timeout-mins", "1"])
    assert rc == 0

    method, path, headers, body = broker.requests[0]
    assert (method, path) == ("POST", "/api/workspace/ingest/video")
    assert headers["X-Demo-Source"] == "gemini2"
    assert headers["X-Upload-Filename"] == "scene1_gemini.mp4"
    assert headers["X-Scene-Label"] == "scene1 (Gemini 2)"
    assert headers["Authorization"] == "Bearer tok-123"
    assert headers["Content-Type"] == "application/octet-stream"
    assert body == (out / "capture.mp4").read_bytes()
    assert int(headers["Content-Length"]) == len(body)
    assert polls["n"] >= 3

    receipt = json.loads((out / "upload_receipt.json").read_text())
    assert receipt["scan_id"] == "S1" and receipt["source"] == "recon_gemini2"
    assert receipt["job_final"]["status"] == "done"


def test_upload_prints_scan_id(tmp_path, broker, fast_poll, capsys):
    out = _bundle(tmp_path)
    broker.routes[("POST", "/api/workspace/ingest/video")] = (
        202, {"job_id": "J2", "scan_id": "SCAN-PRINTED"})
    broker.routes[("GET", "/api/workspace/jobs/J2")] = (
        200, {"status": "done", "stage": "done"})
    gem.main(["upload", "--bundle-dir", str(out), "--broker", broker.url,
              "--token", "t"])
    assert "scan_id: SCAN-PRINTED" in capsys.readouterr().out


def test_upload_failed_job_exits_nonzero_no_receipt(tmp_path, broker,
                                                    fast_poll):
    out = _bundle(tmp_path)
    broker.routes[("POST", "/api/workspace/ingest/video")] = (
        202, {"job_id": "J3", "scan_id": "S3"})
    broker.routes[("GET", "/api/workspace/jobs/J3")] = (
        200, {"status": "failed", "stage": "recon", "error": "boom"})
    with pytest.raises(SystemExit) as exc:
        gem.main(["upload", "--bundle-dir", str(out), "--broker", broker.url,
                  "--token", "t"])
    assert exc.value.code == 1
    assert not (out / "upload_receipt.json").exists()


def test_upload_rejected_402x_is_fatal(tmp_path, broker, fast_poll):
    out = _bundle(tmp_path)
    broker.routes[("POST", "/api/workspace/ingest/video")] = (
        400, {"error": "unsupported_video_type"})
    with pytest.raises(SystemExit):
        gem.main(["upload", "--bundle-dir", str(out), "--broker", broker.url,
                  "--token", "t"])


def test_upload_receipt_idempotency(tmp_path, broker, fast_poll):
    out = _bundle(tmp_path)
    (out / "upload_receipt.json").write_text(json.dumps(
        {"scan_id": "OLD", "job_id": "OLDJOB", "uploaded_at": "then"}))
    with pytest.raises(SystemExit):
        gem.main(["upload", "--bundle-dir", str(out), "--broker", broker.url,
                  "--token", "t"])
    assert broker.requests == []  # refused before any network traffic


def test_upload_requires_token(tmp_path, monkeypatch):
    out = _bundle(tmp_path)
    monkeypatch.delenv(gem.TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit):
        gem.main(["upload", "--bundle-dir", str(out), "--broker",
                  "http://127.0.0.1:1"])


def test_poll_job_timeout(monkeypatch, broker):
    broker.routes[("GET", "/api/workspace/jobs/JT")] = (
        200, {"status": "running", "stage": "recon"})
    clock = {"t": 0.0}

    def fake_clock():
        clock["t"] += 30.0
        return clock["t"]

    with pytest.raises(SystemExit):
        gem.poll_job(broker.url, "t", "JT", timeout_s=60.0,
                     sleep=lambda s: None, clock=fake_clock)


# ---------------------------------------------------------------------------
# ratio math — unit-level, no network
# ---------------------------------------------------------------------------


def test_zbuffer_nearest_wins_and_bounds():
    # two points project to the same pixel; the nearer must win
    pts = np.array([
        [0.0, 0.0, 2.0],    # center pixel, z=2
        [0.0, 0.0, 5.0],    # same pixel, farther
        [0.0, 0.0, -1.0],   # behind camera → dropped
        [100.0, 0.0, 1.0],  # projects far out of bounds → dropped
    ])
    depth, valid = gem.zbuffer_project(pts, fx=100, fy=100, cx=32, cy=32,
                                       width=64, height=64, downsample=1)
    assert valid[32, 32] and depth[32, 32] == pytest.approx(2.0)
    assert valid.sum() == 1  # nothing else landed


def test_zbuffer_downsample_grid_shape():
    pts = np.array([[0.0, 0.0, 3.0]])
    depth, valid = gem.zbuffer_project(pts, 100, 100, 160, 100, 320, 200, 4)
    assert depth.shape == (50, 80)
    assert valid[25, 40] and depth[25, 40] == pytest.approx(3.0)


def test_block_median_depth_zeros_are_invalid():
    img = np.zeros((8, 8), np.uint16)
    img[0:4, 0:4] = 2000  # top-left block: 2000 mm = 2 m
    # top-right block: half zeros, half 1000 → median of valid = 1 m
    img[0:4, 4:6] = 1000
    m = gem.block_median_depth(img, 4, 1.0)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(2.0)
    assert m[0, 1] == pytest.approx(1.0)
    assert np.isnan(m[1, 0]) and np.isnan(m[1, 1])


def test_block_median_respects_mm_per_unit():
    img = np.full((4, 4), 1000, np.uint16)
    assert gem.block_median_depth(img, 4, 2.0)[0, 0] == pytest.approx(2.0)


def test_choose_gemini_grid_branches():
    intr = {"color": {"width": 1920, "height": 1080, "fx": 1030.0, "fy": 1030.0,
                      "cx": 962.0, "cy": 538.0},
            "depth": {"width": 1280, "height": 800, "fx": 614.0, "fy": 614.0,
                      "cx": 647.0, "cy": 405.0},
            "d2c_extrinsics": {"rot": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               "trans": [-13.9, -0.1, -1.9]}}
    g = gem.choose_gemini_grid(intr, (800, 1280))
    assert g["grid"] == "depth" and g["fx"] == 614.0
    assert g["rotation"] is not None and g["rotation"].shape == (3, 3)
    g = gem.choose_gemini_grid(intr, (1080, 1920))
    assert g["grid"] == "color" and g["fx"] == 1030.0 and g["rotation"] is None
    with pytest.raises(ValueError):
        gem.choose_gemini_grid(intr, (600, 800))


def test_parse_cloud_ply_roundtrips_server_format():
    from server.scene_report.cloud_io import build_ply_bytes

    positions = np.array([[0.5, -1.0, 2.0], [3.0, 4.0, 5.0]], np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    pts = gem.parse_cloud_ply(build_ply_bytes(positions, colors))
    np.testing.assert_allclose(pts, positions.astype(np.float64))


def test_parse_cloud_ply_rejects_garbage():
    with pytest.raises(ValueError):
        gem.parse_cloud_ply(b"not a ply at all")


def test_pick_anchor_points_are_real_extremes():
    rng = np.random.default_rng(7)
    line = np.stack([np.linspace(0, 10, 500),
                     rng.normal(0, 0.1, 500),
                     rng.normal(0, 0.1, 500)], axis=1)
    a, b, sep = gem.pick_anchor_points(line)
    assert sep == pytest.approx(10.0, abs=0.5)
    # both are members of the cloud
    assert any(np.allclose(a, p) for p in line[[0, -1]])
    assert any(np.allclose(b, p) for p in line[[0, -1]])
    with pytest.raises(ValueError):
        gem.pick_anchor_points(np.zeros((1, 3)))


def test_ncc_extremes():
    a = np.arange(64.0).reshape(8, 8)
    assert gem.ncc(a, a) == pytest.approx(1.0)
    assert gem.ncc(a, -a) == pytest.approx(-1.0)
    assert gem.ncc(a, np.zeros_like(a)) == 0.0


def test_ratio_gate_pass_and_refusals(ma):
    good = [0.40, 0.41, 0.39, 0.405, 0.395]
    g = gem.ratio_gate(good, ma.cov, 0.15)
    assert g["ok"] and g["ratio"] == pytest.approx(0.40, abs=0.01)
    assert g["cov"] < 0.05

    scattered = [0.30, 0.50, 0.90, 1.40, 0.20]
    g = gem.ratio_gate(scattered, ma.cov, 0.15)
    assert not g["ok"] and "CoV" in g["reason"]

    g = gem.ratio_gate([0.4, 0.4], ma.cov, 0.15)  # < MIN_RATIO_FRAMES
    assert not g["ok"] and "usable frame ratios" in g["reason"]

    g = gem.ratio_gate([float("nan"), -1.0], ma.cov, 0.15)
    assert not g["ok"]


def test_manual_anchor_instructions_content():
    msg = gem.manual_anchor_instructions("SCAN9", "https://b", "CoV 0.3 > 0.15")
    assert "REFUSED" in msg and "SCAN9" in msg
    assert "Measure panel" in msg and "/api/scenes/SCAN9/anchor" in msg
    assert "distance_m" in msg


# ---------------------------------------------------------------------------
# synthetic-scene helpers for the ratio-recovery + e2e anchor tests
# ---------------------------------------------------------------------------

_W, _H = 320, 200
_FX = _FY = 260.0
_CX, _CY = _W / 2.0, _H / 2.0
TRUE_RATIO = 0.37  # metres per SLAM unit


def _synthetic_cloud():
    """One SLAM-unit world point per full-res pixel of the synthetic camera at
    identity pose — z varies smoothly so the ratio test isn't degenerate."""
    u, v = np.meshgrid(np.arange(_W) + 0.5, np.arange(_H) + 0.5)
    z = 2.0 + 0.3 * np.sin(u / 40.0) + 0.2 * np.cos(v / 30.0)
    x = (u - _CX) / _FX * z
    y = (v - _CY) / _FY * z
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _depth_png_for(points_world, ratio, noise=0.0, seed=0):
    """The hardware depth PNG (uint16 mm) this cloud would produce at identity
    pose under a given metres-per-unit ratio."""
    depth, _valid = gem.zbuffer_project(points_world, _FX, _FY, _CX, _CY,
                                        _W, _H, 1)
    metres = np.nan_to_num(depth, nan=0.0) * ratio
    if noise:
        rng = np.random.default_rng(seed)
        metres *= 1.0 + rng.normal(0.0, noise, metres.shape)
    return np.clip(metres * 1000.0, 0, 65535).astype(np.uint16)


def test_frame_ratio_recovers_synthetic_scale(ma):
    """The real core ``frame_ratio`` on my raster output recovers a known
    metres-per-unit factor to <1%."""
    cloud = _synthetic_cloud()
    ds = 4
    slam_depth, slam_valid = gem.zbuffer_project(cloud, _FX, _FY, _CX, _CY,
                                                 _W, _H, ds)
    metric_m = gem.block_median_depth(_depth_png_for(cloud, TRUE_RATIO), ds, 1.0)
    diag = ma.frame_ratio(slam_depth, slam_valid, metric_m, max_depth=10.0)
    assert diag is not None and diag["n"] >= 200
    assert diag["ratio"] == pytest.approx(TRUE_RATIO, rel=0.01)


# ---------------------------------------------------------------------------
# anchor — full flow against the mock broker (cv2 required for depth PNGs)
# ---------------------------------------------------------------------------


def _npz_bytes(**arrays) -> bytes:
    import io as _io

    buf = _io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _anchor_fixture(tmp_path, broker, *, per_frame_ratios, scan="S1"):
    """A synthetic capture + bundle + fully-programmed mock broker for the
    anchor flow: K keyframes at identity pose, fids 0..K-1 (skip=8 under
    fps=15/extract=2), per-frame depth PNGs at the given ratios."""
    cv2 = pytest.importorskip("cv2")
    from server.scene_report.cloud_io import build_ply_bytes

    k = len(per_frame_ratios)
    n_frames = 40
    stems = [f"{2000 + i}.000000" for i in range(n_frames)]
    cap = _make_capture(
        tmp_path, stems, meta=_SWEEP_META,
        intrinsics={"color": {"width": 1920, "height": 1080, "fx": 1030.0,
                              "fy": 1030.0, "cx": 960.0, "cy": 540.0},
                    "depth": {"width": _W, "height": _H, "fx": _FX, "fy": _FY,
                              "cx": _CX, "cy": _CY},
                    "d2c_extrinsics": {"rot": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                       "trans": [-13.9, -0.1, -1.9]},
                    "depth_scale_mm_per_unit": 1.0})
    cloud = _synthetic_cloud()
    skip = 8  # round(15 / 2.0)
    (cap / "depth").mkdir()
    for i, r in enumerate(per_frame_ratios):
        png = _depth_png_for(cloud, r, noise=0.002, seed=i)
        assert cv2.imwrite(str(cap / "depth" / f"{stems[i * skip]}.png"), png)

    out = cap / "ingest"
    out.mkdir()
    fm = gem.build_frame_map("scene1", cap, stems, 15.0, "default")
    (out / "frame_map.json").write_text(json.dumps(fm))
    (out / "intrinsics.json").write_text((cap / "intrinsics.json").read_text())
    (out / "upload_receipt.json").write_text(json.dumps({"scan_id": scan}))

    eye = np.eye(4)
    entries = [{"blob_key": f"0_{i}.jpg", "submap_id": 0, "frame_idx": i,
                "traj_row": i, "c2w": eye.tolist(),
                "intrinsics": [_FX, _FY, _CX, _CY]} for i in range(k)]
    traj = _npz_bytes(
        poses=np.tile(np.eye(4, dtype=np.float32), (k, 1, 1)),
        intrinsics=np.tile(np.array([_FX, _FY, _CX, _CY], np.float32), (k, 1)),
        source_frame_id=np.arange(k, dtype=np.float32),
    )
    positions32 = cloud.astype(np.float32)
    colors = np.full((cloud.shape[0], 3), 128, np.uint8)

    captured = {}

    def anchor_route(handler, body):
        req = json.loads(body)
        captured["anchor_body"] = req
        pa = np.asarray(req["point_a"], float)
        pb = np.asarray(req["point_b"], float)
        measured = float(np.linalg.norm(pa - pb))
        scale = req["distance_m"] / measured
        return 200, {"scale_factor": scale, "measured_distance": measured,
                     "distance_m": req["distance_m"],
                     "calibrated_cloud_key": "derived/anchor/1_ab/cloud.ply",
                     "cloud_extent_before": 4.0,
                     "cloud_extent_after_m": 4.0 * scale}

    def doc_route(handler, body):
        captured["doc_body"] = json.loads(body)
        return 200, {"ok": True, "key": "derived/demo/metric/provenance.json"}

    broker.routes[("GET", f"/api/scenes/{scan}")] = (
        200, {"scan_id": scan, "source": "recon_gemini2",
              "derived_latest": None})
    broker.routes[("GET", f"/api/scenes/{scan}/derived/demo/frames_index.json")] = (
        200, {"version": 1, "count": k, "frames": entries})
    broker.routes[("GET", f"/api/scenes/{scan}/demo/trajectory.npz")] = (200, traj)
    broker.routes[("GET", f"/api/scenes/{scan}/cloud.ply")] = (
        200, build_ply_bytes(positions32, colors))
    broker.routes[("POST", f"/api/scenes/{scan}/anchor")] = anchor_route
    broker.routes[("PUT", f"/api/scenes/{scan}/demo/doc")] = doc_route
    return out, captured


def _run_anchor(out, broker, *extra):
    return gem.main(["anchor", "--bundle-dir", str(out), "--broker",
                     broker.url, "--token", "tok", "--keyframes", "4",
                     "--no-verify-frames", *extra])


def test_anchor_e2e_applies_ratio_and_provenance(tmp_path, broker, ma):
    out, captured = _anchor_fixture(
        tmp_path, broker, per_frame_ratios=[TRUE_RATIO] * 4)
    rc = _run_anchor(out, broker)
    assert rc == 0

    body = captured["anchor_body"]
    pa = np.asarray(body["point_a"])
    pb = np.asarray(body["point_b"])
    implied = body["distance_m"] / float(np.linalg.norm(pa - pb))
    assert implied == pytest.approx(TRUE_RATIO, rel=0.01)

    doc = captured["doc_body"]
    assert doc["key_suffix"] == "metric/provenance.json"
    prov = doc["json"]
    assert prov["method"] == "gemini2_depth_ratio"
    assert prov["note"] == gem.PROVENANCE_NOTE
    assert prov["claim_copy"] == gem.CLAIM_COPY
    assert prov["display_hint"] == "depth-scaled estimate"
    assert prov["units"] == "m"
    assert prov["units_basis"] == "anchor:derived/anchor/1_ab/cloud.ply"
    assert prov["frames_used"] >= 3
    assert prov["cov"] <= 0.15
    assert prov["ratio_m_per_slam_unit"] == pytest.approx(TRUE_RATIO, rel=0.01)
    assert prov["projection"]["d2c_translation_ignored"] is True
    assert prov["sensor"]["name"] == "Orbbec Gemini 2"
    # local provenance copy for the capture notebook
    assert (out / "metric_provenance.json").is_file()


def test_anchor_e2e_cov_gate_refuses_with_instructions(tmp_path, broker, ma,
                                                       capsys):
    out, captured = _anchor_fixture(
        tmp_path, broker, per_frame_ratios=[0.25, 0.40, 0.60, 0.95])
    rc = _run_anchor(out, broker)
    assert rc == 3
    err = capsys.readouterr().err
    assert "REFUSED auto-anchor" in err and "manual two-point anchor" in err
    assert "anchor_body" not in captured  # nothing was applied
    assert "doc_body" not in captured


def test_anchor_dry_run_never_writes(tmp_path, broker, ma):
    out, captured = _anchor_fixture(
        tmp_path, broker, per_frame_ratios=[TRUE_RATIO] * 4)
    rc = _run_anchor(out, broker, "--dry-run")
    assert rc == 0
    assert "anchor_body" not in captured and "doc_body" not in captured


def test_anchor_refuses_wrong_source_without_force(tmp_path, broker, ma):
    out, _captured = _anchor_fixture(
        tmp_path, broker, per_frame_ratios=[TRUE_RATIO] * 4, scan="S2")
    broker.routes[("GET", "/api/scenes/S2")] = (
        200, {"scan_id": "S2", "source": "recon_video", "derived_latest": None})
    with pytest.raises(SystemExit):
        gem.main(["anchor", "--bundle-dir", str(out), "--broker", broker.url,
                  "--token", "t", "--scan-id", "S2", "--no-verify-frames"])


def test_anchor_refuses_already_anchored_without_force(tmp_path, broker, ma):
    out, _captured = _anchor_fixture(
        tmp_path, broker, per_frame_ratios=[TRUE_RATIO] * 4, scan="S3")
    broker.routes[("GET", "/api/scenes/S3")] = (
        200, {"scan_id": "S3", "source": "recon_gemini2",
              "derived_latest": {"kind": "anchor", "scale_factor": 0.5}})
    with pytest.raises(SystemExit):
        gem.main(["anchor", "--bundle-dir", str(out), "--broker", broker.url,
                  "--token", "t", "--scan-id", "S3", "--no-verify-frames"])


def test_keyframe_offset_matching(tmp_path):
    """The verification matcher finds the true capture frame around a drifted
    guess and rejects unrelated content."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    color_dir = tmp_path / "color"
    color_dir.mkdir()
    ts = [f"{3000 + i}.000000" for i in range(30)]
    frames = []
    for i, stem in enumerate(ts):
        img = (rng.uniform(0, 255, (48, 64, 3))).astype(np.uint8)
        frames.append(img)
        assert cv2.imwrite(str(color_dir / f"{stem}.png"), img)

    hit = gem.match_keyframe_offset(frames[17], ts, color_dir, guess=14,
                                    window=5)
    assert hit is not None and hit[0] == 17 and hit[1] > 0.9

    unrelated = (rng.uniform(0, 255, (48, 64, 3))).astype(np.uint8)
    assert gem.match_keyframe_offset(unrelated, ts, color_dir, guess=14,
                                     window=5) is None
