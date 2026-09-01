#!/usr/bin/env python3
"""Gemini-2 capture → demo scene glue (docs/demo-2026-07 design/shell.md §4c, W1).

Founder-run CLI bridging the EXP-39 capture contract
(``experiments/exp39_depth_recon_metric/scripts/capture_gemini.py``:
``captures/<scene>/gemini/{color,depth}/<ts>.png`` + ``intrinsics.json`` +
``capture_meta.json``) into the deployed demo broker. Three subcommands, run in
order on the capture Mac:

``assemble --capture-dir captures/<scene>/gemini``
    Sorted color PNGs → ffmpeg → ``ingest/capture.mp4`` at a fixed fps
    (``capture_meta.json`` capture.fps if present, else derived from a sweep's
    ``n_frames/seconds``, else 15), plus ``ingest/frame_map.json`` (mp4 frame
    index → original ``<ts>`` — the ``trajectory.npz.source_frame_id`` bridge
    back to the depth PNGs), a copy of ``intrinsics.json``, and
    ``ingest/depth_manifest.json`` (subsampled depth-PNG inventory). The depth
    PNGs are NEVER uploaded; they stay local for the ``anchor`` step.

    Single-frame captures (a ``snapshot``) are REFUSED with an honest message:
    one frame cannot be reconstructed — a real sweep is needed. Decision
    documented here per the build brief; ``--allow-single`` overrides for
    plumbing dry-runs only (the resulting scene would be garbage and is
    labeled accordingly).

``upload --bundle-dir captures/<scene>/gemini/ingest``
    Streams ``capture.mp4`` to the DEPLOYED broker's
    ``POST /api/workspace/ingest/video`` with ``X-Demo-Source: gemini2``
    (→ ``source="recon_gemini2"``) and ``X-Scene-Label``, then polls
    ``GET /api/workspace/jobs/<job_id>`` until done and prints the scan_id.
    Auth: ``--token`` or ``DEMO_AUTH_TOKEN`` (same env demo_smoke.sh uses).
    Writes ``ingest/upload_receipt.json``; re-running refuses without
    ``--re-upload`` (upload idempotency per WORLD-TRANSFORM-CONTRACT).

``anchor --bundle-dir captures/<scene>/gemini/ingest [--scan-id …]``
    STRETCH (shell.md §4c steps 3–5): depth-ratio auto-anchor. Fetches the
    persisted scene's ``derived/demo/frames_index.json`` + trajectory
    (``GET /api/scenes/<id>/demo/trajectory.npz``) + world cloud
    (``GET /api/scenes/<id>/cloud.ply``), samples K≈8 persisted keyframes,
    z-buffer-projects the world cloud into each keyframe's pose on the Gemini
    pixel grid for per-pixel SLAM depth, pairs it with the hardware depth PNG,
    and computes per-frame metres-per-SLAM-unit ratios with the pure-numpy
    ``vggt_slam.metric_anchor`` fns from core@v2.2.2 (imported install-first,
    else file-loaded from a sibling core checkout — the module is import-clean
    with numpy only, verified). ``CoV > 0.15`` (core ``DEFAULT_WITHIN_COV``)
    → REFUSES to auto-anchor and prints the manual two-point instruction.
    Otherwise applies the median ratio through the EXISTING
    ``POST /api/scenes/<id>/anchor`` route (two well-separated real cloud
    points, ``distance_m = |a−b|·ratio`` → ``scale_factor == ratio``) and
    writes ``derived/demo/metric/provenance.json`` via
    ``PUT /api/scenes/<id>/demo/doc``.

Frame↔depth alignment is MEASURED, not assumed: ``source_frame_id`` counts
disparity-ADMITTED keyframes (``streaming_slam.py`` gates every fed frame on
optical-flow disparity before ``frame_count`` increments), so
``fid × skip`` (``skip = round(capture_fps / extract_fps)``, demo_recon_job
``extract_fps=2.0``) is only an initial guess. Each sampled keyframe's stored
JPEG (``GET /api/scenes/<id>/keyframes/<blob_key>``) is correlation-matched
against the local color PNGs in a ±search window around the guess (chained so
cumulative admission drift is followed); frames that fail to match are
dropped, and too few matches refuses the auto-anchor honestly.

Documented approximations (recorded in the provenance doc):
  * The d2c extrinsic translation (~14 mm baseline) is IGNORED when projecting
    into the depth grid — it cannot be composed with SLAM-unit geometry before
    the scale is known, and its effect on a median ratio at metre-range depths
    is ≲1%. The d2c rotation (≈identity) IS applied.
  * No lens-distortion model is applied when rasterizing (the depth sensor's
    stored coefficients are zero; color-grid mode notes the omission).

Honesty doctrine (CLAIM-LEDGER row "Depth camera (Gemini 2) input"): the fixed
copy is "Gemini 2 RGB capture reconstructed through the existing pipeline;
depth used experimentally to establish metric scale." — never "RGB-D
reconstruction". Anchored-via-depth numbers render as "depth-scaled estimate".
Both strings are embedded verbatim in the provenance doc.

Dependencies: stdlib for assemble/upload (plus the ``ffmpeg`` binary);
``numpy`` (+ ``cv2`` for PNG/JPEG decode) only for ``anchor``. HTTP is stdlib
``urllib`` on purpose — the exp39 capture venv (``.venv-mac``) has no
``requests``.

Ratio math lives in ``server/oreos/depth_ratio.py``, not here: it was lifted out
of this file so a second depth source (the TUM RGB-D benchmark — see
``modal_tum_depth_anchor.py``) could reuse it rather than fork it. This module
still owns every Gemini-specific decision (capture layout, d2c grid choice,
fid×skip guessing, the broker conversation) and re-exports the shared names so
its own CLI and test suite are unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:  # anchor-only dependency; assemble/upload stay stdlib-pure
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in numpy-less envs
    np = None


def _load_depth_ratio_core():
    """``server/oreos/depth_ratio.py`` loaded BY FILE PATH, not by package import.

    ``import server.oreos.depth_ratio`` would execute ``server/oreos/__init__.py``, which
    registers the Flask blueprint and imports every route module in the repo — flask,
    openai and opencv dragged into a capture-Mac venv that has none of them, to run a
    z-buffer. The target module is numpy-only at module level (cv2 is lazy inside it), so
    a direct file load is both sufficient and honest about the dependency. Same ladder
    idea as :func:`load_metric_anchor_module` below, and the same reasoning as
    ``scripts/backfill_imported_scene.py``'s stub-package trick.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "server" / "oreos" / "depth_ratio.py"
    if not path.is_file():  # pragma: no cover - a broken checkout, not a runtime state
        raise ImportError(f"depth-ratio core not found at {path}")
    spec = importlib.util.spec_from_file_location("_gemini_depth_ratio", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if np is not None:
    _dr = _load_depth_ratio_core()
    # Re-exported so this module's public surface (and its test suite) is unchanged by
    # the move. These are THE shared implementations — never shadow one locally.
    load_metric_anchor_module = _dr.load_metric_anchor_module
    parse_cloud_ply = _dr.parse_cloud_ply
    zbuffer_project = _dr.zbuffer_project
    block_median_depth = _dr.block_median_depth
    pair_ratio = _dr.pair_ratio
    pick_anchor_points = _dr.pick_anchor_points
    ratio_gate = _dr.ratio_gate
    ncc = _dr.ncc
    MIN_RATIO_FRAMES = _dr.MIN_RATIO_FRAMES
    MATCH_MIN_CORR = _dr.MATCH_MIN_CORR
    ANCHOR_CLOUD_SUBSAMPLE = _dr.ANCHOR_CLOUD_SUBSAMPLE
    _read_depth_png = _dr.read_depth_png
    _gray_small = _dr.gray_thumb
else:  # pragma: no cover - numpy-less env: assemble/upload only
    # The shared core imports numpy at module level, so it cannot be loaded at all here.
    # These names exist purely so this module still IMPORTS without numpy — its documented
    # posture is that assemble/upload are stdlib-pure. Every one of them is anchor-only, and
    # ``cmd_anchor`` refuses on ``np is None`` before any of them is read.
    _dr = None
    load_metric_anchor_module = parse_cloud_ply = zbuffer_project = None
    block_median_depth = pair_ratio = pick_anchor_points = ratio_gate = ncc = None
    _read_depth_png = _gray_small = None
    MIN_RATIO_FRAMES = MATCH_MIN_CORR = ANCHOR_CLOUD_SUBSAMPLE = None

# Deployed broker default — same default + env names as scripts/demo_smoke.sh.
DEFAULT_BROKER = "https://galois77777--vggt-slam-streaming-web.modal.run"
BROKER_ENV = "DEMO_BROKER_URL"
TOKEN_ENV = "DEMO_AUTH_TOKEN"

DEFAULT_FPS = 15.0            # shell.md §4c: fixed fps fallback
DEFAULT_EXTRACT_FPS = 2.0     # demo_recon_job / reconstruct_pilot default
DEFAULT_KEYFRAMES = 8         # K sampled keyframes for the depth ratio
DEFAULT_RASTER_DOWNSAMPLE = 4  # z-buffer + depth block-median grid divisor
DEFAULT_MAX_DEPTH_M = 10.0    # Gemini 2 usable indoor range; beyond is noise
DEFAULT_MATCH_WINDOW = 12     # ± raw-frame search window for keyframe matching
# MATCH_MIN_CORR / MIN_RATIO_FRAMES / ANCHOR_CLOUD_SUBSAMPLE are the shared core's
# (re-exported above) — a second copy here could drift from the gate that actually runs.

# <ts> filename stems: epoch seconds, fixed 6 decimals (capture_gemini.py contract).
TS_RE = re.compile(r"^\d{1,20}\.\d{6}$")

INGEST_DIRNAME = "ingest"
FRAME_MAP_NAME = "frame_map.json"
DEPTH_MANIFEST_NAME = "depth_manifest.json"
INTRINSICS_NAME = "intrinsics.json"
CAPTURE_META_NAME = "capture_meta.json"
VIDEO_NAME = "capture.mp4"
RECEIPT_NAME = "upload_receipt.json"

# CLAIM-LEDGER / shell.md §4c verbatim strings (do not edit here — edit the
# ledger first; these must stay byte-identical to the approved copy).
PROVENANCE_NOTE = (
    "Metric scale from hardware-aligned depth; geometry is camera-only "
    "reconstruction. No depth-accuracy claims."
)
CLAIM_COPY = (
    "Gemini 2 RGB capture reconstructed through the existing pipeline; "
    "depth used experimentally to establish metric scale."
)


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 1) -> "None":
    eprint(f"ERROR: {msg}")
    raise SystemExit(code)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# HTTP (stdlib urllib; Bearer auth; JSON + bytes + streamed-file helpers)
# ---------------------------------------------------------------------------


def _request(
    method: str,
    url: str,
    token: str,
    *,
    data=None,
    headers: dict | None = None,
    timeout: float = 120.0,
):
    """One HTTP round-trip. Returns ``(status, body_bytes, content_type)``;
    HTTP error statuses are returned (not raised) so callers can read the JSON
    error envelope. Network-level failures raise ``urllib.error.URLError``."""
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a JSON body
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def api_json(method: str, url: str, token: str, *, body: dict | None = None,
             timeout: float = 120.0) -> tuple[int, dict]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    status, raw, _ctype = _request(method, url, token, data=data,
                                   headers=headers, timeout=timeout)
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
    except (ValueError, UnicodeDecodeError):
        parsed = {"_raw": raw[:200].decode("utf-8", "replace")}
    return status, parsed


def api_bytes(url: str, token: str, *, timeout: float = 600.0) -> tuple[int, bytes]:
    status, raw, _ctype = _request("GET", url, token, timeout=timeout)
    return status, raw


def post_file_stream(url: str, token: str, path: Path, headers: dict,
                     timeout: float = 1800.0) -> tuple[int, dict]:
    """POST a file as a streamed ``application/octet-stream`` body with an
    explicit Content-Length (the ingest route 413s oversized bodies early)."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        hdrs = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            **headers,
        }
        status, raw, _ctype = _request("POST", url, token, data=fh,
                                       headers=hdrs, timeout=timeout)
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
    except (ValueError, UnicodeDecodeError):
        parsed = {"_raw": raw[:200].decode("utf-8", "replace")}
    return status, parsed


def resolve_broker(arg_value: str | None) -> str:
    return (arg_value or os.environ.get(BROKER_ENV) or DEFAULT_BROKER).rstrip("/")


def resolve_token(arg_value: str | None) -> str:
    token = arg_value or os.environ.get(TOKEN_ENV)
    if not token:
        die(f"no auth token: pass --token or set {TOKEN_ENV} "
            "(owner Clerk JWT or broker session token)")
    return token


# ---------------------------------------------------------------------------
# Capture-dir model (EXP-39 contract)
# ---------------------------------------------------------------------------


def numeric_ts_sort(stems: list[str]) -> list[str]:
    """Sort ``<ts>`` stems by NUMERIC value while returning the EXACT original
    strings (identity matters: the stem IS the filename; float round-tripping
    could lose it). Lexicographic order breaks when the integer-second digit
    count differs, so an explicit numeric key is used."""
    return sorted(stems, key=lambda s: float(s))


def discover_capture(capture_dir: Path) -> dict:
    """Validate + load a ``captures/<scene>/gemini`` dir per the EXP-39 layout.
    Returns ``{scene, color_dir, depth_dir, ts (sorted stems), intrinsics,
    capture_meta}``. Missing color frames are fatal; missing depth/intrinsics
    degrade with a warning (assemble+upload work; anchor will refuse)."""
    capture_dir = capture_dir.resolve()
    if not capture_dir.is_dir():
        die(f"capture dir not found: {capture_dir}")
    color_dir = capture_dir / "color"
    depth_dir = capture_dir / "depth"
    if not color_dir.is_dir():
        die(f"no color/ dir under {capture_dir} — is this a "
            "captures/<scene>/gemini directory (capture_gemini.py layout)?")

    stems = []
    for p in color_dir.iterdir():
        if p.suffix.lower() == ".png" and TS_RE.match(p.stem):
            stems.append(p.stem)
    if not stems:
        die(f"no <epoch>.<6-decimals>.png color frames in {color_dir}")
    ts = numeric_ts_sort(stems)

    intrinsics = None
    intr_path = capture_dir / INTRINSICS_NAME
    if intr_path.is_file():
        intrinsics = json.loads(intr_path.read_text())
    else:
        eprint(f"WARNING: {intr_path} missing — anchor step will refuse")

    capture_meta = None
    meta_path = capture_dir / CAPTURE_META_NAME
    if meta_path.is_file():
        capture_meta = json.loads(meta_path.read_text())

    # scene name: captures/<scene>/gemini → <scene>
    scene = capture_dir.parent.name if capture_dir.name == "gemini" else capture_dir.name

    return {
        "scene": scene,
        "capture_dir": capture_dir,
        "color_dir": color_dir,
        "depth_dir": depth_dir if depth_dir.is_dir() else None,
        "ts": ts,
        "intrinsics": intrinsics,
        "capture_meta": capture_meta,
    }


def resolve_fps(capture_meta: dict | None, override: float | None) -> tuple[float, str]:
    """Fixed assembly fps (shell.md §4c): explicit override wins; then an
    explicit ``capture.fps`` key (future-proof — today's capture_gemini.py does
    not write one); then a sweep's measured ``n_frames/seconds``; else 15."""
    if override is not None:
        if override <= 0:
            die(f"--fps must be positive (got {override})")
        return float(override), "--fps"
    cap = (capture_meta or {}).get("capture") or {}
    fps = cap.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps), "capture_meta.capture.fps"
    seconds = cap.get("seconds")
    n_frames = cap.get("n_frames")
    if (
        isinstance(seconds, (int, float)) and seconds > 0
        and isinstance(n_frames, (int, float)) and n_frames > 1
    ):
        return round(float(n_frames) / float(seconds), 3), "capture_meta.capture.n_frames/seconds"
    return DEFAULT_FPS, "default"


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def build_frame_map(scene: str, capture_dir: Path, ts: list[str], fps: float,
                    fps_source: str) -> dict:
    return {
        "version": 1,
        "kind": "gemini2_frame_map",
        "scene": scene,
        "capture_dir": str(capture_dir),
        "video": VIDEO_NAME,
        "fps": fps,
        "fps_source": fps_source,
        "frame_count": len(ts),
        "ts": list(ts),
        "created_at": iso_now(),
        "note": (
            "mp4 frame i (0-based) was assembled from color/<ts[i]>.png. "
            "trajectory.npz source_frame_id counts disparity-ADMITTED SLAM "
            "keyframes of the frames fed at extract_fps (demo_recon_job "
            f"default {DEFAULT_EXTRACT_FPS}); the anchor step verifies the "
            "fid*skip guess by image-matching stored keyframes."
        ),
    }


def build_depth_manifest(depth_dir: Path | None, ts: list[str],
                         max_entries: int = 64) -> dict:
    """Subsampled inventory of the depth PNGs matching the color timeline —
    the anchor step's local lookup is by ``depth/<ts>.png`` directly; this doc
    records coverage so a mismatched capture is visible at assemble time."""
    doc: dict = {
        "version": 1,
        "kind": "gemini2_depth_manifest",
        "depth_dir": str(depth_dir) if depth_dir else None,
        "color_frames": len(ts),
        "depth_present": 0,
        "missing_for_color": [],
        "stride": None,
        "entries": [],
        "created_at": iso_now(),
    }
    if depth_dir is None:
        doc["missing_for_color"] = ts[:10]
        return doc
    present = []
    missing = []
    for stem in ts:
        p = depth_dir / f"{stem}.png"
        if p.is_file():
            present.append((stem, p))
        else:
            missing.append(stem)
    doc["depth_present"] = len(present)
    doc["missing_for_color"] = missing[:10]  # sample, not the full list
    if present:
        stride = max(1, math.ceil(len(present) / max_entries))
        doc["stride"] = stride
        picked = present[::stride]
        if present[-1] not in picked:
            picked.append(present[-1])
        doc["entries"] = [
            {"ts": stem, "file": f"depth/{stem}.png", "bytes": p.stat().st_size}
            for stem, p in picked
        ]
    return doc


def ffmpeg_command(staging_pattern: str, fps: float, out_path: Path) -> list[str]:
    """The fixed ffmpeg invocation (split out so tests can assert its shape).
    libx264 + yuv420p decodes cleanly in the server's cv2 feed path; crf 18
    keeps SLAM-relevant detail; the scale filter defensively evens dimensions."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}",
        "-i", staging_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]


def probe_frame_count(video_path: Path) -> int | None:
    """Container frame count via ffprobe (packet count — exact for our
    image-sequence encodes), or ``None`` when ffprobe is unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def assemble(capture: dict, out_dir: Path, fps: float, fps_source: str,
             allow_single: bool, run=subprocess.run) -> dict:
    """Steps: stage numbered symlinks → ffmpeg → verify frame count → write
    frame_map + intrinsics copy + depth manifest. Returns a summary dict.
    ``run`` is injectable for tests (recording fake)."""
    ts = capture["ts"]
    if len(ts) == 1 and not allow_single:
        die(
            "capture has a single frame (a snapshot) — one frame cannot be "
            "reconstructed. Need a real sweep: re-capture with\n"
            "  sudo .venv-mac/bin/python scripts/capture_gemini.py sweep "
            "--scene <scene> --seconds 75\n"
            "(--allow-single forces assembly for plumbing dry-runs only; the "
            "resulting scene would be garbage)",
            code=2,
        )
    if len(ts) < 30:
        eprint(f"WARNING: only {len(ts)} frames — a very short sweep; "
               "reconstruction quality will be poor")

    if shutil.which("ffmpeg") is None:
        die("ffmpeg not found on PATH (brew install ffmpeg)")

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / VIDEO_NAME

    with tempfile.TemporaryDirectory(prefix="gemini_assemble_") as staging:
        staging_dir = Path(staging)
        for i, stem in enumerate(ts):
            src = capture["color_dir"] / f"{stem}.png"
            dst = staging_dir / f"{i:06d}.png"
            try:
                os.symlink(src, dst)
            except OSError:  # e.g. FAT volume — fall back to copying
                shutil.copyfile(src, dst)
        cmd = ffmpeg_command(str(staging_dir / "%06d.png"), fps, video_path)
        proc = run(cmd, capture_output=True, text=True)
        if getattr(proc, "returncode", 1) != 0:
            die(f"ffmpeg failed ({proc.returncode}):\n{getattr(proc, 'stderr', '')}")

    if not video_path.is_file() or video_path.stat().st_size == 0:
        die(f"ffmpeg produced no output at {video_path}")

    probed = probe_frame_count(video_path)
    if probed is not None and probed != len(ts):
        die(f"assembled mp4 has {probed} frames but {len(ts)} were staged — "
            "frame_map would be wrong; refusing")
    if probed is None:
        eprint("WARNING: ffprobe unavailable — frame count not verified")

    frame_map = build_frame_map(capture["scene"], capture["capture_dir"], ts,
                                fps, fps_source)
    (out_dir / FRAME_MAP_NAME).write_text(json.dumps(frame_map, indent=2))

    if capture["intrinsics"] is not None:
        (out_dir / INTRINSICS_NAME).write_text(
            json.dumps(capture["intrinsics"], indent=2))

    manifest = build_depth_manifest(capture["depth_dir"], ts)
    (out_dir / DEPTH_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    return {
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
        "frames": len(ts),
        "fps": fps,
        "fps_source": fps_source,
        "depth_present": manifest["depth_present"],
        "probed_frames": probed,
    }


def cmd_assemble(args) -> int:
    capture = discover_capture(Path(args.capture_dir))
    fps, fps_source = resolve_fps(capture["capture_meta"], args.fps)
    out_dir = Path(args.out) if args.out else capture["capture_dir"] / INGEST_DIRNAME
    summary = assemble(capture, out_dir, fps, fps_source, args.allow_single)
    print(json.dumps({"assembled": summary, "bundle_dir": str(out_dir)}, indent=2))
    print(f"\nOK: {summary['frames']} frames @ {fps:g} fps ({fps_source}) → "
          f"{summary['video']} ({summary['video_bytes'] / 1e6:.1f} MB)")
    print(f"Next: {sys.argv[0]} upload --bundle-dir {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _ascii_header(value: str, fallback: str) -> str:
    """Header values must be latin-1-safe; scene labels are best-effort ASCII."""
    cleaned = value.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or fallback


def poll_job(broker: str, token: str, job_id: str, *, timeout_s: float,
             poll_s: float = 10.0, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Poll ``GET /api/workspace/jobs/<id>`` until done/error/timeout. Prints
    stage transitions. Returns the final record; raises SystemExit on job
    error or timeout. Transient network errors are retried (5 strikes)."""
    url = f"{broker}/api/workspace/jobs/{job_id}"
    started = clock()
    last_stage = None
    net_errors = 0
    while True:
        try:
            status, record = api_json("GET", url, token, timeout=30.0)
            net_errors = 0
        except urllib.error.URLError as exc:
            net_errors += 1
            if net_errors >= 5:
                die(f"job polling failed 5 times in a row: {exc}")
            sleep(poll_s)
            continue
        if status == 404:
            die(f"job {job_id} unknown to the broker (404) — was the upload "
                "accepted by a different deployment?")
        if status != 200:
            die(f"job poll HTTP {status}: {record}")
        stage = record.get("stage")
        state = record.get("status")
        if stage != last_stage:
            elapsed = clock() - started
            print(f"  [{elapsed:6.0f}s] stage={stage} status={state}")
            last_stage = stage
        if state in ("done",):
            return record
        if state in ("error", "failed"):
            die(f"recon job failed at stage {stage!r}: "
                f"{record.get('error') or 'unknown error'}")
        if clock() - started > timeout_s:
            die(f"recon job still {state}/{stage} after {timeout_s / 60:.0f} min "
                f"— keep polling manually: GET {url}")
        sleep(poll_s)


def cmd_upload(args) -> int:
    bundle_dir = Path(args.bundle_dir).resolve()
    video_path = bundle_dir / VIDEO_NAME
    frame_map_path = bundle_dir / FRAME_MAP_NAME
    if not video_path.is_file():
        die(f"no {VIDEO_NAME} in {bundle_dir} — run assemble first")
    if not frame_map_path.is_file():
        die(f"no {FRAME_MAP_NAME} in {bundle_dir} — run assemble first")
    frame_map = json.loads(frame_map_path.read_text())

    receipt_path = bundle_dir / RECEIPT_NAME
    if receipt_path.is_file() and not args.re_upload:
        old = json.loads(receipt_path.read_text())
        die(f"already uploaded as scan {old.get('scan_id')} (job "
            f"{old.get('job_id')}, {old.get('uploaded_at')}). Pass "
            "--re-upload to force a duplicate scene.")

    broker = resolve_broker(args.broker)
    token = resolve_token(args.token)
    scene = frame_map.get("scene") or bundle_dir.parent.parent.name
    label = args.label or f"{scene} (Gemini 2)"

    headers = {
        "X-Upload-Filename": f"{_ascii_header(scene, 'gemini_capture')}_gemini.mp4",
        "X-Demo-Source": "gemini2",
        "X-Scene-Label": _ascii_header(label, "Gemini 2 capture"),
    }
    size_mb = video_path.stat().st_size / 1e6
    print(f"Uploading {video_path.name} ({size_mb:.1f} MB) → "
          f"{broker}/api/workspace/ingest/video  [X-Demo-Source: gemini2]")
    status, payload = post_file_stream(
        f"{broker}/api/workspace/ingest/video", token, video_path, headers)
    if status != 202:
        die(f"upload rejected: HTTP {status} {payload}")
    job_id = payload.get("job_id")
    scan_id = payload.get("scan_id")
    print(f"Accepted: job_id={job_id} scan_id={scan_id} — polling "
          f"(~10 min per 3-min clip; recon on A10G)")

    record = poll_job(broker, token, job_id,
                      timeout_s=float(args.timeout_mins) * 60.0)

    receipt = {
        "version": 1,
        "kind": "gemini2_upload_receipt",
        "broker": broker,
        "job_id": job_id,
        "scan_id": scan_id,
        "label": label,
        "source": "recon_gemini2",
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
        "frame_count": frame_map.get("frame_count"),
        "uploaded_at": iso_now(),
        "job_final": {k: record.get(k) for k in
                      ("status", "stage", "elapsed_s", "point_count",
                       "has_splat", "has_trajectory", "keyframe_count")},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nDONE — scan_id: {scan_id}")
    print(f"Receipt: {receipt_path}")
    print(f"Next (stretch): {sys.argv[0]} anchor --bundle-dir {bundle_dir}")
    return 0


# ---------------------------------------------------------------------------
# anchor (stretch) — depth-ratio auto-anchor
# ---------------------------------------------------------------------------


def choose_gemini_grid(intrinsics: dict, depth_shape: tuple[int, int]) -> dict:
    """Which camera model the depth PNG lives in (see module docstring):
    color-resolution PNG → software-aligned into the COLOR camera (color K,
    no extra rotation); depth-resolution PNG → the depth sensor's own grid
    (depth K + d2c-rotation-only transform; ~14 mm translation ignored).
    Anything else refuses honestly."""
    h, w = int(depth_shape[0]), int(depth_shape[1])
    color = intrinsics.get("color") or {}
    depth = intrinsics.get("depth") or {}
    if (h, w) == (int(color.get("height", -1)), int(color.get("width", -1))):
        return {
            "grid": "color", "width": w, "height": h,
            "fx": float(color["fx"]), "fy": float(color["fy"]),
            "cx": float(color["cx"]), "cy": float(color["cy"]),
            "rotation": None,
        }
    if (h, w) == (int(depth.get("height", -1)), int(depth.get("width", -1))):
        rot = None
        d2c = intrinsics.get("d2c_extrinsics") or {}
        r = d2c.get("rot")
        if isinstance(r, list) and len(r) == 9 and np is not None:
            rot = np.asarray(r, dtype=np.float64).reshape(3, 3)
        return {
            "grid": "depth", "width": w, "height": h,
            "fx": float(depth["fx"]), "fy": float(depth["fy"]),
            "cx": float(depth["cx"]), "cy": float(depth["cy"]),
            "rotation": rot,  # X_depth = R^T · X_color (translation ignored)
        }
    raise ValueError(
        f"depth PNG is {w}x{h} but intrinsics describe color "
        f"{color.get('width')}x{color.get('height')} / depth "
        f"{depth.get('width')}x{depth.get('height')} — cannot pick a camera "
        "model; re-check the capture")


def match_keyframe_offset(kf_bgr, ts: list[str], color_dir: Path, guess: int,
                          window: int, min_corr: float | None = None):
    """Find the raw capture frame best matching a stored SLAM keyframe JPEG
    by NCC over ``guess ± window``. Returns ``(raw_index, corr)`` or ``None``
    when nothing correlates ≥ ``min_corr`` (caller drops the frame).

    ``min_corr`` defaults to the shared core's threshold, resolved at CALL time rather
    than as a def-time default: the constant only exists once numpy is importable, and
    binding it in the signature would break importing this module without numpy."""
    import cv2

    min_corr = MATCH_MIN_CORR if min_corr is None else min_corr
    kf = _gray_small(kf_bgr)
    best = (None, -2.0)
    lo = max(0, guess - window)
    hi = min(len(ts) - 1, guess + window)
    for idx in range(lo, hi + 1):
        img = cv2.imread(str(color_dir / f"{ts[idx]}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        c = ncc(kf, _gray_small(img))
        if c > best[1]:
            best = (idx, c)
    if best[0] is None or best[1] < min_corr:
        return None
    return best


def manual_anchor_instructions(scan_id: str, broker: str, reason: str) -> str:
    return (
        f"REFUSED auto-anchor for scan {scan_id}: {reason}.\n"
        "Fall back to the manual two-point anchor (demo-safe, still honest):\n"
        "  1. Open the demo workspace for this scan → Measure panel.\n"
        "  2. Pick two points spanning a known real dimension (doorway width,\n"
        "     desk edge, printed tag sheet).\n"
        "  3. Enter the true distance in metres — the UI POSTs the existing\n"
        f"     {broker}/api/scenes/{scan_id}/anchor route.\n"
        "  (curl equivalent: POST that route with\n"
        '   {"point_a": [x,y,z], "point_b": [x,y,z], "distance_m": D})\n'
        "The scene stays fully demoable; scale provenance will read "
        '"manual two-point anchor" instead of "depth-scaled estimate".'
    )


def cmd_anchor(args) -> int:  # noqa: C901 — one linear founder-facing flow
    if np is None:
        die("numpy is required for the anchor step (pip install numpy)")
    bundle_dir = Path(args.bundle_dir).resolve()
    frame_map_path = bundle_dir / FRAME_MAP_NAME
    if not frame_map_path.is_file():
        die(f"no {FRAME_MAP_NAME} in {bundle_dir} — run assemble first")
    frame_map = json.loads(frame_map_path.read_text())
    ts = frame_map["ts"]
    intr_path = bundle_dir / INTRINSICS_NAME
    if not intr_path.is_file():
        die(f"no {INTRINSICS_NAME} in {bundle_dir} — capture lacked intrinsics; "
            "auto-anchor impossible (use the manual two-point anchor)")
    intrinsics = json.loads(intr_path.read_text())
    depth_dir = Path(frame_map["capture_dir"]) / "depth"
    if not depth_dir.is_dir():
        die(f"no depth dir at {depth_dir} — auto-anchor impossible")

    scan_id = args.scan_id
    if not scan_id:
        receipt_path = bundle_dir / RECEIPT_NAME
        if not receipt_path.is_file():
            die("no --scan-id and no upload_receipt.json — run upload first "
                "or pass --scan-id")
        scan_id = json.loads(receipt_path.read_text()).get("scan_id")
    broker = resolve_broker(args.broker)
    token = resolve_token(args.token)

    ma = load_metric_anchor_module()
    cov_max = float(args.cov_max if args.cov_max is not None
                    else ma.DEFAULT_WITHIN_COV)

    # --- scene record + guards --------------------------------------------
    status, record = api_json("GET", f"{broker}/api/scenes/{scan_id}", token)
    if status != 200:
        die(f"scene {scan_id} fetch failed: HTTP {status} {record}")
    source = record.get("source")
    if source != "recon_gemini2" and not args.force:
        die(f"scene {scan_id} has source={source!r}, not recon_gemini2 — "
            "is this really the Gemini capture's scan? (--force to override)")
    derived_latest = record.get("derived_latest") or {}
    if derived_latest.get("kind") == "anchor" and not args.force:
        die(f"scene {scan_id} already has a metric anchor "
            f"(scale_factor={derived_latest.get('scale_factor')}) — "
            "--force to re-anchor from depth")

    # --- fetch frames_index + trajectory + cloud ---------------------------
    status, fidx = api_json(
        "GET", f"{broker}/api/scenes/{scan_id}/derived/demo/frames_index.json",
        token)
    if status != 200:
        die(f"frames_index fetch failed (HTTP {status}) — scene predates "
            "--demo-index recon? Auto-anchor needs it; use the manual anchor.")
    entries = [e for e in fidx.get("frames", [])
               if e.get("traj_row") is not None and e.get("c2w")]
    if not entries:
        die("frames_index has no keyframes with trajectory rows — use the "
            "manual anchor")

    status, traj_bytes = api_bytes(
        f"{broker}/api/scenes/{scan_id}/demo/trajectory.npz", token)
    if status != 200:
        die(f"trajectory fetch failed (HTTP {status}): the deployed broker "
            "may predate the demo/trajectory.npz route — redeploy, or use "
            "the manual anchor")
    with np.load(io.BytesIO(traj_bytes)) as data:
        source_frame_id = np.asarray(data["source_frame_id"],
                                     dtype=np.float64).reshape(-1)

    print(f"Fetching world cloud (cloud.ply) for {scan_id} …")
    status, ply_bytes = api_bytes(f"{broker}/api/scenes/{scan_id}/cloud.ply",
                                  token)
    if status != 200:
        die(f"cloud.ply fetch failed: HTTP {status}")
    points_world = parse_cloud_ply(ply_bytes)
    print(f"  {points_world.shape[0]} world points, "
          f"{len(entries)} indexed keyframes, {len(ts)} capture frames")

    # --- fed-index → raw-frame skip (initial guess; verified per frame) ----
    fps = float(frame_map["fps"])
    extract_fps = float(args.extract_fps)
    skip = max(1, int(round(fps / extract_fps)))

    # --- sample K keyframes, verify alignment, compute ratios --------------
    import cv2  # anchor-only dependency

    k = min(int(args.keyframes), len(entries))
    sample_idx = ma.evenly_spaced(len(entries), k)
    downsample = int(args.raster_downsample)
    mm_per_unit = float(intrinsics.get("depth_scale_mm_per_unit", 1.0))

    frame_reports = []
    ratios = []
    grid_desc = None
    prev_delta = 0
    for si in sample_idx:
        entry = entries[int(si)]
        row = int(entry["traj_row"])
        if not (0 <= row < source_frame_id.shape[0]):
            frame_reports.append({"blob_key": entry.get("blob_key"),
                                  "skipped": "traj_row out of range"})
            continue
        fid = int(round(float(source_frame_id[row])))
        guess = fid * skip + prev_delta
        if not (0 <= guess < len(ts) + args.match_window):
            frame_reports.append({"blob_key": entry.get("blob_key"), "fid": fid,
                                  "skipped": f"guess {guess} outside capture "
                                             f"({len(ts)} frames) — check "
                                             "--extract-fps"})
            continue
        guess = min(max(guess, 0), len(ts) - 1)

        matched = None
        if not args.no_verify_frames:
            status, jpg = api_bytes(
                f"{broker}/api/scenes/{scan_id}/keyframes/{entry['blob_key']}",
                token)
            if status == 200:
                kf_img = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                      cv2.IMREAD_COLOR)
                if kf_img is not None:
                    matched = match_keyframe_offset(
                        kf_img, ts, Path(frame_map["capture_dir"]) / "color",
                        guess, int(args.match_window))
            if matched is None:
                frame_reports.append({"blob_key": entry.get("blob_key"),
                                      "fid": fid, "guess": guess,
                                      "skipped": "keyframe↔capture match "
                                                 f"< {MATCH_MIN_CORR} NCC"})
                continue
            raw_idx, corr = matched
            prev_delta = raw_idx - fid * skip
        else:
            raw_idx, corr = guess, None

        stem = ts[raw_idx]
        depth_path = depth_dir / f"{stem}.png"
        if not depth_path.is_file():
            frame_reports.append({"blob_key": entry.get("blob_key"),
                                  "ts": stem, "skipped": "no depth PNG"})
            continue
        depth_png = _read_depth_png(depth_path)
        if grid_desc is None:
            grid_desc = choose_gemini_grid(intrinsics, depth_png.shape)
            print(f"  depth grid: {grid_desc['grid']} camera "
                  f"{grid_desc['width']}x{grid_desc['height']} "
                  f"(downsample {downsample})")

        diag = pair_ratio(
            points_world, entry["c2w"], grid_desc, depth_png, ma,
            downsample=downsample, max_depth_m=float(args.max_depth_m),
            mm_per_unit=mm_per_unit)
        rep = {"blob_key": entry.get("blob_key"), "fid": fid,
               "raw_index": int(raw_idx), "ts": stem,
               "match_corr": None if corr is None else round(float(corr), 3),
               "match_delta": int(raw_idx - fid * skip)}
        if diag is None:
            rep["skipped"] = "too few shared valid pixels"
        else:
            rep.update({k2: diag[k2] for k2 in
                        ("ratio", "rel_iqr", "n", "slam_med", "metric_med_m")})
            ratios.append(float(diag["ratio"]))
        frame_reports.append(rep)
        print(f"  kf {entry.get('blob_key')}: fid={fid} → raw {raw_idx} "
              f"(Δ{rep['match_delta']:+d}"
              + (f", ncc {corr:.2f}" if corr is not None else "")
              + (f") ratio={diag['ratio']:.5f} n={diag['n']}" if diag else
                 ") — skipped"))

    # --- gate --------------------------------------------------------------
    gate = ratio_gate(ratios, ma.cov, cov_max)
    if not gate["ok"]:
        eprint("")
        eprint(manual_anchor_instructions(scan_id, broker, gate["reason"]))
        return 3
    ratio = float(gate["ratio"])
    print(f"\nDepth ratio: {ratio:.6f} m per SLAM unit "
          f"(CoV {gate['cov']:.3f} ≤ {cov_max:g}, {gate['n']} frames)")

    if args.dry_run:
        print("--dry-run: not applying the anchor")
        return 0

    # --- apply via the existing anchor route -------------------------------
    a, b, sep = pick_anchor_points(points_world)
    distance_m = sep * ratio
    body = {"point_a": [float(v) for v in a], "point_b": [float(v) for v in b],
            "distance_m": float(distance_m)}
    status, result = api_json("POST", f"{broker}/api/scenes/{scan_id}/anchor",
                              token, body=body, timeout=600.0)
    if status != 200:
        die(f"anchor POST failed: HTTP {status} {result}")
    applied = float(result.get("scale_factor", 0.0))
    if not math.isclose(applied, ratio, rel_tol=1e-6):
        die(f"anchor applied scale {applied} but depth ratio is {ratio} — "
            "refusing to write provenance for a mismatched anchor")
    print(f"Anchored: scale_factor={applied:.6f} "
          f"(cloud {result.get('cloud_extent_before'):.2f} SLAM-units → "
          f"{result.get('cloud_extent_after_m'):.2f} m extent)")

    # --- provenance doc (PUT demo/doc → derived/demo/metric/provenance.json)
    meta = frame_map.get("scene")
    capture_meta = {}
    cm_path = Path(frame_map["capture_dir"]) / CAPTURE_META_NAME
    if cm_path.is_file():
        capture_meta = json.loads(cm_path.read_text())
    provenance = {
        "version": 1,
        "scan_id": scan_id,
        "parent_artifact": "cloud.npz",
        "created_at": iso_now(),
        "run_id": None,
        "method": "gemini2_depth_ratio",
        "sensor": (capture_meta.get("device")
                   or {"name": "Orbbec Gemini 2"}),
        "scene": meta,
        "frames_sampled": int(k),
        "frames_used": int(gate["n"]),
        "frame_reports": frame_reports,
        "ratio_m_per_slam_unit": ratio,
        "cov": gate["cov"],
        "cov_max": cov_max,
        "capture_fps": fps,
        "extract_fps_assumed": extract_fps,
        "frame_skip": skip,
        "projection": {
            "grid": grid_desc["grid"] if grid_desc else None,
            "raster_downsample": downsample,
            "d2c_translation_ignored": True,
            "distortion_model_applied": False,
        },
        "anchor": {
            "point_a": body["point_a"],
            "point_b": body["point_b"],
            "distance_m": body["distance_m"],
            "scale_factor": applied,
            "calibrated_cloud_key": result.get("calibrated_cloud_key"),
        },
        "units": "m",
        "units_basis": f"anchor:{result.get('calibrated_cloud_key')}",
        "display_hint": "depth-scaled estimate",
        "note": PROVENANCE_NOTE,
        "claim_copy": CLAIM_COPY,
    }
    status, doc_resp = api_json(
        "PUT", f"{broker}/api/scenes/{scan_id}/demo/doc", token,
        body={"key_suffix": "metric/provenance.json", "json": provenance})
    if status not in (200, 201):
        die(f"provenance PUT failed: HTTP {status} {doc_resp} — the anchor IS "
            "applied; re-run anchor with --force after fixing, or write the "
            "doc manually")
    print(f"Provenance: derived/demo/metric/provenance.json written "
          f"({gate['n']} frames, CoV {gate['cov']:.3f})")
    local_copy = bundle_dir / "metric_provenance.json"
    local_copy.write_text(json.dumps(provenance, indent=2))
    print(f"Local copy: {local_copy}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="demo_ingest_gemini.py",
        description=("Gemini-2 capture → demo scene glue "
                     "(assemble → upload → [stretch] anchor)"))
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("assemble", help="color PNGs → capture.mp4 + frame_map")
    a.add_argument("--capture-dir", required=True,
                   help="captures/<scene>/gemini directory")
    a.add_argument("--fps", type=float, default=None,
                   help="override the assembly fps")
    a.add_argument("--out", default=None,
                   help="bundle dir (default: <capture-dir>/ingest)")
    a.add_argument("--allow-single", action="store_true",
                   help="assemble a single-frame capture anyway "
                        "(plumbing dry-runs only)")
    a.set_defaults(fn=cmd_assemble)

    u = sub.add_parser("upload", help="POST capture.mp4 to the deployed broker")
    u.add_argument("--bundle-dir", required=True,
                   help="the assemble output dir (…/gemini/ingest)")
    u.add_argument("--broker", default=None,
                   help=f"broker base URL (default ${BROKER_ENV} or "
                        f"{DEFAULT_BROKER})")
    u.add_argument("--token", default=None,
                   help=f"auth token (default ${TOKEN_ENV})")
    u.add_argument("--label", default=None, help="X-Scene-Label override")
    u.add_argument("--timeout-mins", type=float, default=45.0,
                   help="max minutes to poll the recon job (default 45)")
    u.add_argument("--re-upload", action="store_true",
                   help="upload again even though a receipt exists")
    u.set_defaults(fn=cmd_upload)

    n = sub.add_parser("anchor",
                       help="STRETCH: depth-ratio auto-anchor the scene")
    n.add_argument("--bundle-dir", required=True)
    n.add_argument("--scan-id", default=None,
                   help="default: read upload_receipt.json")
    n.add_argument("--broker", default=None)
    n.add_argument("--token", default=None)
    n.add_argument("--keyframes", type=int, default=DEFAULT_KEYFRAMES)
    n.add_argument("--extract-fps", type=float, default=DEFAULT_EXTRACT_FPS,
                   help="the recon job's frame-decimation fps (default 2.0 = "
                        "demo_recon_job default)")
    n.add_argument("--raster-downsample", type=int,
                   default=DEFAULT_RASTER_DOWNSAMPLE)
    n.add_argument("--max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M)
    n.add_argument("--cov-max", type=float, default=None,
                   help="CoV refusal gate (default: core DEFAULT_WITHIN_COV)")
    n.add_argument("--match-window", type=int, default=DEFAULT_MATCH_WINDOW)
    n.add_argument("--no-verify-frames", action="store_true",
                   help="skip keyframe↔capture image verification "
                        "(trust fid*skip; NOT recommended)")
    n.add_argument("--dry-run", action="store_true",
                   help="compute + print the ratio; do not anchor")
    n.add_argument("--force", action="store_true",
                   help="anchor even if source mismatches or an anchor exists")
    n.set_defaults(fn=cmd_anchor)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
