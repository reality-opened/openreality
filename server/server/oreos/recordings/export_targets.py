"""Real gated training-data exports from an Oreos session dir.

Adapter from what an Oreos session dir persists (TUM trajectory, extracted frames,
preview cloud, manifests) to the real export machinery in ``server/export``:

  - ``openreality/<session>/`` — the OpenReality export tree (docs/dataset-export.md §4),
    written by ``server.export.writer.export_from_record`` (the GPU-free Stage-4 path).
    Trajectory state/action math is the canonical ``server/export/trajectory.py`` code
    (via ``server.export.record.build_rows_from_trajectory``) — never re-derived here.
  - ``lerobot/`` — GR00T-LeRobot v2 (``server.export.to_lerobot``). The default target.
  - ``isaac/`` — optional Isaac Sim/Lab USD scene (``server/export/isaac``). The metric
    scale comes from ``consistency.json``'s ``umeyama_scale`` — the Sim(3) alignment of
    our trajectory onto the robot's own odometry.  Per the SCALE DIRECTION note in
    ``platform/experiments/slam_bench/metrics.py``, that ``s`` maps EST -> reference
    metres, exactly what the Isaac ``scale`` argument wants.  Mirroring that module's
    ``metric_valid`` philosophy, the scale is trusted ONLY when the QC verdict is
    ``high_confidence`` AND an odometry reference was available AND the scale is
    finite/positive with no alignment warning; anything else refuses the Isaac target
    with reason "no trustworthy scale".  Missing ``usd-core`` skips gracefully.

HONESTY MODEL (mirrors ``server/oreos/recordings/README.md``):
  - The SLAM frame stays up-to-scale / not gravity-aligned in the OpenReality + LeRobot
    outputs (``meta/info.json`` carries the caveats verbatim). Only the Isaac target is
    metric, only under the trusted-scale gate above, and its manifest notes that the
    reference is the robot's own drifting odometry — NOT ground truth.
  - What a session dir fundamentally lacks vs a live scan is *recorded*, never faked
    (``caveats`` in the returned dict + ``meta/oreos_provenance.json`` in the tree):
    per-frame camera intrinsics (``observation.intrinsics`` is all zeros), the semantic
    grounding sidecar (no scene report / detections were ever produced), and the
    full-resolution cloud (``map_preview.ply`` is the recon's voxel-downsampled preview).
  - ``videos/ego_view.mp4`` frame ``i`` must equal ``trajectory.parquet`` row ``i``
    (spec §7.4): every trajectory timestamp must resolve to a frame file, and the
    written mp4 is re-counted; any mismatch is an error, not a silent desync.

``build_exports`` never raises for data problems: every requested target reports
``{"status": "ok" | "refused" | "skipped" | "error", ...}`` so the pipeline's QC-gate
semantics (stage ok, per-target reasons in the bundle manifest) stay intact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from server.export.schema import DEFAULT_FPS

TARGET_LEROBOT = "lerobot"
TARGET_ISAAC = "isaac"
KNOWN_TARGETS = (TARGET_LEROBOT, TARGET_ISAAC)

# Nearest-frame association tolerance (s) — same default as consistency.associate.
_FRAME_MATCH_TOL_S = 0.05


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def build_exports(
    session_dir: Path,
    targets: Optional[list] = None,
    *,
    out_dir: Optional[Path] = None,
    fps: float = DEFAULT_FPS,
) -> dict:
    """Build real training-data exports for one Oreos session dir.

    ``targets`` ⊆ {"lerobot", "isaac"} (default ["lerobot"]). Outputs land under
    ``out_dir`` (default ``<session_dir>/bundle``)::

        <out_dir>/openreality/<session>/   OpenReality export tree (always built first)
        <out_dir>/lerobot/                 GR00T-LeRobot v2 dataset
        <out_dir>/isaac/                   scene.usd + trajectory.usd + manifest.json

    Returns ``{"session", "export_dir", "targets": {name: {"status", ...}}, "caveats"}``.
    Data problems are reported per-target (status error/refused/skipped with a written
    reason) — this function does not raise for them.
    """
    session_dir = Path(session_dir)
    out_root = Path(out_dir) if out_dir is not None else session_dir / "bundle"
    scan_id = session_dir.name or "session"

    result: dict = {"session": scan_id, "export_dir": None, "targets": {}, "caveats": []}

    requested = []
    for t in list(targets) if targets else [TARGET_LEROBOT]:
        if t not in KNOWN_TARGETS:
            result["targets"][t] = {
                "status": "error",
                "reason": f"unknown target {t!r} (known: {', '.join(KNOWN_TARGETS)})",
            }
        else:
            requested.append(t)
    if not requested:
        return result

    # -- base OpenReality tree (everything else transcodes from it) ----------
    try:
        sess = _load_session(session_dir, result["caveats"])
        export_dir = _write_openreality_tree(
            sess, out_root / "openreality", scan_id, fps, result["caveats"]
        )
    except Exception as exc:  # noqa: BLE001 — data problem: report, don't raise
        reason = f"openreality tree build failed — {type(exc).__name__}: {exc}"
        for t in requested:
            result["targets"][t] = {"status": "error", "reason": reason}
        return result
    result["export_dir"] = str(export_dir)

    for t in requested:
        if t == TARGET_LEROBOT:
            result["targets"][t] = _target_lerobot(export_dir, out_root, sess)
        elif t == TARGET_ISAAC:
            result["targets"][t] = _target_isaac(sess, out_root / "isaac", scan_id, fps)
    return result


# ---------------------------------------------------------------------------
# session dir -> arrays (the adapter proper)
# ---------------------------------------------------------------------------

def _load_session(session_dir: Path, caveats: list) -> dict:
    est = session_dir / "results" / "est_tum.txt"
    if not est.exists():
        raise FileNotFoundError("results/est_tum.txt missing — no trajectory to export")
    tokens, ts, pos, quat = _parse_tum(est)
    n = len(tokens)
    if n < 2:
        raise ValueError(
            f"est_tum.txt has {n} pose(s); need >= 2 (the ego-motion action needs a next pose)"
        )

    poses = _pose_matrices(pos, quat)
    images, frame_files = _load_frames(session_dir / "frames", tokens, ts)

    # Honest gap: a session dir has no per-frame camera intrinsics (the recon worker does
    # not persist K). Zeros, declared — never fabricated.
    intrinsics = np.zeros((n, 4), dtype=np.float32)
    caveats.append(
        "observation.intrinsics is all zeros: an Oreos session dir carries no per-frame "
        "camera intrinsics (the recon worker does not persist K)."
    )

    # source_frame_id is a float32 column (spec §4.1); absolute unix timestamps
    # (~1.8e9) don't survive float32, so store seconds since t0 and record t0.
    t0 = float(ts[0])
    rel_ts = (ts - t0).astype(np.float32)
    caveats.append(
        f"source_frame_id is seconds since t0_unix={t0:.6f} (absolute unix timestamps "
        "do not survive the schema's float32 column)."
    )

    cloud = None
    ply = session_dir / "results" / "map_preview.ply"
    if ply.exists():
        try:
            cloud = _load_cloud(ply)
            caveats.append(
                "clouds/cloud.npz is the recon's voxel-downsampled preview cloud "
                "(map_preview.ply), not the full-resolution point cloud."
            )
        except Exception as exc:  # noqa: BLE001 — cloud is optional; record why
            caveats.append(f"map_preview.ply unreadable ({exc}); clouds/cloud.npz omitted.")
    else:
        caveats.append("no results/map_preview.ply; clouds/cloud.npz omitted.")

    caveats.append(
        "no semantic layer in a session dir (no scene report / detections): the "
        "grounding/ sidecar and annotations are absent and the LeRobot task string is "
        "the generic navigation task."
    )

    return {
        "n": n,
        "t0": t0,
        "rel_ts": rel_ts,
        "poses": poses,
        "intrinsics": intrinsics,
        "images": images,
        "frame_files": frame_files,
        "cloud": cloud,
        "consistency": _read_json(session_dir / "consistency.json"),
        "confidence": _read_json(session_dir / "confidence.json"),
        "ingest": _read_json(session_dir / "ingest_manifest.json"),
    }


def _parse_tum(path: Path) -> tuple:
    """TUM file -> (raw timestamp tokens, ts (N,), pos (N,3), quat (N,4) xyzw).

    Same skip rules as ``server.oreos.recordings.consistency.load_tum``; additionally keeps the raw
    first token of each line because frame files are named by that exact string.
    """
    tokens: list = []
    rows: list = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        tokens.append(parts[0])
        rows.append([float(x) for x in parts[:8]])
    if not rows:
        return [], np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4))
    a = np.asarray(rows, dtype=np.float64)
    return tokens, a[:, 0], a[:, 1:4], a[:, 4:8]


def _pose_matrices(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """TUM (t, quat xyzw) rows -> cam->world 4x4 poses (N,4,4).

    Same convention as ``GraphMap.write_poses_to_file`` emits and
    ``experiments/slam_bench/metrics.py::_state_to_T`` re-reads (scipy quat is xyzw).
    """
    from scipy.spatial.transform import Rotation

    n = len(pos)
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    T[:, :3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _load_frames(frames_dir: Path, tokens: list, ts: np.ndarray) -> tuple:
    """Resolve every trajectory timestamp to a frame jpg -> (RGB images, file names).

    Exact stem match first (frames are named by the TUM timestamp token), then nearest
    within ``_FRAME_MATCH_TOL_S``. Any unresolved timestamp raises — mp4 frame i must
    equal trajectory row i (§7.4), so a partial video is never written.
    """
    import cv2

    if not frames_dir.is_dir():
        raise FileNotFoundError("frames/ missing — cannot build the ego_view video")
    by_stem = {p.stem: p for p in frames_dir.glob("*.jpg")}
    ordered = []
    for stem, p in by_stem.items():
        try:
            ordered.append((float(stem), p))
        except ValueError:
            continue
    ordered.sort(key=lambda t: t[0])
    stamps = np.asarray([s for s, _p in ordered], dtype=np.float64)

    matched: list = []
    missing: list = []
    for tok, t in zip(tokens, ts):
        p = by_stem.get(tok)
        if p is None and len(stamps):
            j = int(np.searchsorted(stamps, t))
            best, bestd = None, _FRAME_MATCH_TOL_S
            for k in (j - 1, j):
                if 0 <= k < len(stamps) and abs(stamps[k] - t) <= bestd:
                    best, bestd = k, abs(stamps[k] - t)
            if best is not None:
                p = ordered[best][1]
        if p is None:
            missing.append(tok)
        else:
            matched.append(p)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(tokens)} trajectory timestamps have no frame in "
            f"frames/ (first missing: {missing[:3]}); refusing to desync video and "
            "trajectory rows (§7.4)"
        )

    images: list = []
    for p in matched:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable frame {p.name}")
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images, [p.name for p in matched]


def _load_cloud(ply_path: Path) -> tuple:
    """map_preview.ply -> (positions (N,3) f32, colors (N,3) u8).

    open3d when available (the recon writes binary_little_endian); pure-python ASCII
    fallback keeps this loadable (and testable) in open3d-free envs.
    """
    try:
        import open3d as o3d
    except ImportError:
        return _parse_ascii_ply(ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    positions = np.asarray(pcd.points, dtype=np.float32).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("empty point cloud")
    cols = np.asarray(pcd.colors, dtype=np.float64).reshape(-1, 3)
    if cols.shape[0] == positions.shape[0]:
        colors = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full_like(positions, 128, dtype=np.uint8)
    return positions, colors


def _parse_ascii_ply(ply_path: Path) -> tuple:
    """Minimal ASCII-PLY vertex reader (x y z [+ red green blue uchar])."""
    with open(ply_path, "rb") as fh:
        header = fh.readline().decode("ascii", "replace").strip()
        if header != "ply":
            raise ValueError("not a PLY file")
        fmt = None
        n_vertex = 0
        props: list = []
        in_vertex = False
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line:
                raise ValueError("truncated PLY header")
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif line.startswith("property") and in_vertex:
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        if fmt != "ascii":
            raise ValueError(
                f"PLY format {fmt!r} needs open3d (pure-python fallback reads ascii only)"
            )
        idx = {name: i for i, name in enumerate(props)}
        for axis in ("x", "y", "z"):
            if axis not in idx:
                raise ValueError(f"PLY vertex has no {axis!r} property")
        pos_rows: list = []
        col_rows: list = []
        has_color = all(c in idx for c in ("red", "green", "blue"))
        for _ in range(n_vertex):
            vals = fh.readline().split()
            pos_rows.append([float(vals[idx[a]]) for a in ("x", "y", "z")])
            if has_color:
                col_rows.append([float(vals[idx[c]]) for c in ("red", "green", "blue")])
    positions = np.asarray(pos_rows, dtype=np.float32).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("empty point cloud")
    if has_color:
        colors = np.clip(np.asarray(col_rows), 0, 255).astype(np.uint8)
    else:
        colors = np.full_like(positions, 128, dtype=np.uint8)
    return positions, colors


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# base tree (OpenReality export format)
# ---------------------------------------------------------------------------

def _write_openreality_tree(
    sess: dict, out_root: Path, scan_id: str, fps: float, caveats: list
) -> Path:
    """Write ``<out_root>/<scan_id>/`` via the real Stage-4 record path + provenance."""
    from server.export.writer import export_from_record

    record = {
        "scan_id": scan_id,
        "fps": fps,
        "trajectory": {
            "poses": sess["poses"],
            "intrinsics": sess["intrinsics"],
            "source_frame_id": sess["rel_ts"],
        },
        "keyframes": [
            {"global_index": i, "image": img} for i, img in enumerate(sess["images"])
        ],
    }
    if sess["cloud"] is not None:
        record["cloud"] = sess["cloud"]
    export_dir = Path(export_from_record(record, str(out_root)))

    # §7.4 hard check: the Oreos frame set is full-coverage (unlike a sparse persisted
    # record), so re-count the written mp4 and fail loudly on any desync.
    n_video = _count_mp4_frames(export_dir / "videos" / "ego_view.mp4")
    if n_video != sess["n"]:
        raise RuntimeError(
            f"video/trajectory desync: {n_video} mp4 frames for {sess['n']} rows (§7.4)"
        )

    _write_provenance(export_dir, sess, caveats)
    return export_dir


def _count_mp4_frames(path: Path) -> int:
    import cv2

    if not path.exists():
        return 0
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


def _write_provenance(export_dir: Path, sess: dict, caveats: list) -> None:
    """``meta/oreos_provenance.json`` — where this dataset came from and what it lacks."""
    consistency = sess["consistency"] or {}
    confidence = sess["confidence"] or {}
    ingest = sess["ingest"] or {}
    prov = {
        "generator": "server.oreos.recordings.export_targets",
        "t0_unix": sess["t0"],
        "n_keyframes": sess["n"],
        "source": {
            k: ingest.get(k)
            for k in ("source", "format", "camera_stream", "odom_stream", "n_frames", "span_s")
        },
        "consistency": {
            k: consistency.get(k)
            for k in (
                "reference_available",
                "ate_rmse_m",
                "ate_pct_extent",
                "ate_pct_path",
                "umeyama_scale",
                "windowed_scale_cv",
            )
        },
        "confidence": {k: confidence.get(k) for k in ("level", "export_allowed", "reasons")},
        "caveats": list(caveats),
        "notes": (
            "Consistency numbers compare against the robot's own drifting odometry "
            "(Sim(3)-aligned), NOT ground truth. high_confidence means no known failure "
            "signature detected, not verified accurate (server/oreos/recordings/README.md)."
        ),
    }
    (export_dir / "meta" / "oreos_provenance.json").write_text(json.dumps(prov, indent=2))


# ---------------------------------------------------------------------------
# target: GR00T-LeRobot v2
# ---------------------------------------------------------------------------

def _target_lerobot(export_dir: Path, out_root: Path, sess: dict) -> dict:
    from server.export.to_lerobot import convert_to_groot_lerobot

    dest = out_root / "lerobot"
    try:
        convert_to_groot_lerobot(str(export_dir), str(dest))
    except Exception as exc:  # noqa: BLE001 — report, don't raise
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok",
        "path": str(dest),
        "format": "groot-lerobot-v2",
        "n_frames": sess["n"],
    }


# ---------------------------------------------------------------------------
# target: Isaac Sim/Lab USD (optional; scale-honesty gated)
# ---------------------------------------------------------------------------

def _trusted_scale(sess: dict) -> tuple:
    """Return ``(scale, None)`` when the odometry-derived scale is trustworthy, else
    ``(None, reason)``.

    Mirrors ``metric_valid`` in ``experiments/slam_bench/metrics.py``: a scale from a
    failed/absent alignment is meaningless, so the (scale-dependent) Isaac export must
    be skipped rather than authored with a made-up ``metersPerUnit``.
    """
    confidence = sess["confidence"] or {}
    consistency = sess["consistency"] or {}
    problems: list = []
    level = confidence.get("level")
    if level != "high_confidence":
        problems.append(f"confidence level is {level!r}, not high_confidence")
    if not confidence.get("export_allowed", False):
        problems.append("QC gate did not allow export")
    if not consistency.get("reference_available"):
        problems.append("no odometry reference (consistency.json reference_available is false)")
    if consistency.get("warning"):
        problems.append(f"consistency alignment warning: {consistency['warning']}")
    scale = consistency.get("umeyama_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        problems.append("consistency.json has no finite positive umeyama_scale")
    if problems:
        return None, "no trustworthy scale: " + "; ".join(problems)
    return float(scale), None


def _target_isaac(sess: dict, isaac_dir: Path, scan_id: str, fps: float) -> dict:
    scale, reason = _trusted_scale(sess)
    if scale is None:
        return {"status": "refused", "reason": reason}
    if sess["cloud"] is None:
        return {
            "status": "refused",
            "reason": "no usable world cloud (results/map_preview.ply missing or "
            "unreadable) — cannot author an Isaac scene",
        }
    try:
        import pxr  # noqa: F401
    except ImportError:
        return {
            "status": "skipped",
            "reason": "usd-core (pxr) not installed — `pip install usd-core` to enable "
            "the Isaac USD target",
            "trusted_scale": scale,
        }
    try:
        geometry = _write_isaac_usd(sess, isaac_dir, scan_id, fps, scale)
    except Exception as exc:  # noqa: BLE001 — report, don't raise
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok",
        "path": str(isaac_dir),
        "scale": scale,
        "scale_source": "oreos:consistency.umeyama_scale(odometry-reference)",
        "geometry": geometry,
    }


def _write_isaac_usd(
    sess: dict, isaac_dir: Path, scan_id: str, fps: float, scale: float
) -> dict:
    """Array-based mirror of ``server.export.isaac.writer.export_isaac`` (which needs a
    live ``Solver``): align -> mesh (best-effort) -> scene.usd + trajectory.usd + manifest.
    """
    from server.export.isaac import align as align_mod
    from server.export.isaac import usd_writer
    from server.export.isaac.schema import AlignmentInfo, IsaacManifest

    isaac_dir.mkdir(parents=True, exist_ok=True)
    positions, colors = sess["cloud"]
    poses = sess["poses"]
    cam_centers = poses[:, :3, 3]

    align_res = align_mod.compute_alignment(positions, cam_centers, scale=scale)
    positions_isaac = align_mod.apply_transform(align_res.transform, positions)

    mesh_tuple = None
    reconstruction = "none(points)"
    try:
        from server.export.isaac import mesh as mesh_mod

        mesh_tuple = mesh_mod.reconstruct_mesh(positions_isaac, colors, method="poisson")
        reconstruction = "poisson"
        if mesh_tuple is None or len(mesh_tuple[0]) == 0:
            mesh_tuple, reconstruction = None, "none(points)"
    except Exception as exc:  # noqa: BLE001 — points-only fallback, like export_isaac
        print(f"[oreos:isaac] WARN mesh reconstruction failed ({exc}); points-only scene.")
        mesh_tuple = None

    want_points = mesh_tuple is None
    geom_info = usd_writer.write_scene_usd(
        str(isaac_dir / "scene.usd"),
        mesh=mesh_tuple,
        points=positions_isaac if want_points else None,
        point_colors=colors if want_points else None,
        with_collision=True,
        reconstruction=reconstruction,
    )

    poses_isaac = np.stack(
        [align_mod.transform_camera_pose(align_res, P) for P in poses]
    )
    image_hw = tuple(sess["images"][0].shape[:2]) if sess["images"] else None
    # Session intrinsics are all zeros (honest gap) — pass None, not fake focal lengths.
    usd_writer.write_trajectory_usd(
        str(isaac_dir / "trajectory.usd"),
        poses_isaac,
        intrinsics=None,
        fps=fps,
        image_hw=image_hw,
    )

    manifest = IsaacManifest(
        scan_id=scan_id,
        fps=float(fps),
        num_keyframes=int(sess["n"]),
        alignment=AlignmentInfo(
            transform=align_res.transform.tolist(),
            scale=align_res.scale,
            scale_source="oreos:consistency.umeyama_scale(odometry-reference)",
            metric=align_res.metric,
            gravity_aligned=align_res.gravity_aligned,
            up_vector_slam=align_res.up_vector_slam.tolist(),
            floor_inlier_fraction=align_res.floor_inlier_fraction,
            gravity_source=align_res.gravity_source,
            recentered_xy=align_res.recentered_xy,
        ),
        geometry=geom_info,
        notes=(
            "Metric scale = consistency.json umeyama_scale: Sim(3) alignment of the "
            "camera-only trajectory onto the robot's OWN odometry — NOT ground truth. "
            "Trusted only because the QC verdict was high_confidence with a usable "
            "reference (server/oreos/recordings/export_targets.py gate)."
        ),
    )
    (isaac_dir / "manifest.json").write_text(json.dumps(manifest.model_dump(), indent=2))
    return geom_info.model_dump()
