"""Bridge: an Oreos recording session dir → a persisted OS scene.

The recordings pipeline (server.oreos.recordings.pipeline) historically ended at a local
session dir (report.html + bundle/). Merged into the OS, a finished session persists as a
first-class ``PersistedScene`` (``source="robot_recording"``) so it appears in the scene
picker with every panel honestly degraded, the same doctrine as imported splats
(``splat_import.build_scene_payload`` is the direct analog):

- report/facts are DEGRADED + geometry-only (no live solver, no detections);
- the cloud is the recon's voxel-downsampled ``map_preview.ply`` (declared, not hidden);
- poses come from ``results/est_tum.txt`` (TUM → cam->world 4x4, the export convention);
- intrinsics are ZEROS, declared — a session dir never carries K;
- keyframes are a stride-subset of the extracted frames (real capture photography);
- QC verdict + consistency + report.html land as ``derived/demo/recordings/*`` artifacts
  (the derived-key namespace keeps the historical ``demo`` token — persisted contract).

No flask, no modal at module scope — importable inside any worker container.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from server.oreos.recordings.export_targets import _load_cloud, _parse_tum, _pose_matrices

#: Keyframe cap for the persisted record — matches the OS report ceiling
#: (SCENE_REPORT_MAX_KEYFRAMES=48-class); recordings are long, so stride-subset.
MAX_KEYFRAMES = int(48)


def _read_json(path: Path) -> Optional[dict]:
    try:
        doc = json.loads(path.read_text())
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


def _keyframes_b64(frames_dir: Path, tokens: list, cap: int = MAX_KEYFRAMES) -> list[dict[str, Any]]:
    """Stride-subset of the extracted frames, encode_frames-shaped
    (``[{submap_id, frame_idx, image_b64}]``; recordings have no submaps → 0)."""
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        return []
    stride = max(1, len(files) // cap)
    picked = files[::stride][:cap]
    out: list[dict[str, Any]] = []
    for i, f in enumerate(picked):
        try:
            out.append(
                {
                    "submap_id": 0,
                    "frame_idx": int(i * stride),
                    "image_b64": base64.b64encode(f.read_bytes()).decode("ascii"),
                }
            )
        except Exception:
            continue
    return out


def build_recording_scene_payload(session_dir: str | Path, recording_name: str) -> dict[str, Any]:
    """Everything ``save_scene`` needs from a finished session dir.

    Raises ``FileNotFoundError``/``ValueError`` when the session has no usable
    trajectory — a scene without poses is not persistable (nothing to ground)."""
    from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport

    session = Path(session_dir)
    est = session / "results" / "est_tum.txt"
    if not est.exists():
        raise FileNotFoundError("results/est_tum.txt missing — recon produced no trajectory")
    tokens, ts, pos, quat = _parse_tum(est)
    if len(tokens) < 2:
        raise ValueError(f"est_tum.txt has {len(tokens)} pose(s); need >= 2")
    poses = _pose_matrices(pos, quat).astype(np.float32)
    t0 = float(ts[0])
    trajectory = {
        "poses": poses,
        "intrinsics": np.zeros((len(tokens), 4), dtype=np.float32),
        "source_frame_id": (ts - t0).astype(np.float32),
    }

    points = None
    ply = session / "results" / "map_preview.ply"
    if ply.exists():
        try:
            points = _load_cloud(ply)
        except Exception:
            points = None

    confidence = _read_json(session / "confidence.json") or {}
    consistency = _read_json(session / "consistency.json") or {}
    ingest = _read_json(session / "ingest_manifest.json") or {}

    if points is not None:
        positions = np.asarray(points[0])
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        extents = (bbox_max - bbox_min).astype(np.float64)
        metrics = SceneMetrics(
            dimensions=sorted((float(v) for v in extents), reverse=True),
            bbox_min=[float(v) for v in bbox_min],
            bbox_max=[float(v) for v in bbox_max],
            point_count=int(positions.shape[0]),
            vertical_axis_known=False,
        )
        scene_center = [float(v) for v in positions.mean(axis=0)]
    else:
        metrics = SceneMetrics(point_count=0, vertical_axis_known=False)
        scene_center = [0.0, 0.0, 0.0]

    level = str(confidence.get("level") or "unscored")
    reasons = confidence.get("reasons") or []
    ate = consistency.get("ate_rmse_m")
    ate_pct = consistency.get("ate_pct_extent")
    ref_available = bool(consistency.get("reference_available"))
    qc_line = f"QC: {level}" + (f" ({'; '.join(str(r) for r in reasons[:3])})" if reasons else "")
    cons_line = (
        (
            f" Consistency vs robot odometry: ATE {float(ate):.3f} rel-units RMSE"
            + (f" ({float(ate_pct):.1f}% of extent)" if isinstance(ate_pct, (int, float)) else "")
            + " (Sim(3)-aligned)."
        )
        if ref_available and isinstance(ate, (int, float))
        else " No usable odometry reference — consistency unscored."
    )

    facts = SceneFacts(
        metrics=metrics,
        scene_center=scene_center,
        units_note=(
            "Robot recording, camera-only reconstruction: coordinates are up-to-scale "
            "relative units; not metric until anchored."
        ),
    )
    report = SceneReport(
        summary=(
            f"Robot recording '{recording_name}' reconstructed camera-only "
            f"({len(tokens)} poses, {ingest.get('n_frames', '?')} frames, "
            f"{ingest.get('span_s', '?')} s). {qc_line}.{cons_line} The cloud shown is the "
            "recon's voxel-downsampled preview; no detections exist yet — annotation runs "
            "on the geometry and real capture keyframes."
        ),
        room_type="unknown",
        coverage_note="Robot-driven capture — coverage follows the recording's path.",
        facts=facts,
        degraded=True,
    )

    return {
        "report": report,
        "facts": facts,
        "points": points,
        "trajectory": trajectory,
        "keyframes_b64": _keyframes_b64(session / "frames", tokens),
        "qc_level": level,
    }


def persist_recording_scene(
    persistence: Any,
    user_id: str,
    scan_id: str,
    session_dir: str | Path,
    recording_name: str,
    label: Optional[str] = None,
) -> dict[str, Any]:
    """Persist the scene + the QC artifacts. Returns ``{scan_id, qc_level, derived}``."""
    payload = build_recording_scene_payload(session_dir, recording_name)
    persistence.save_scene(
        user_id,
        scan_id,
        payload["report"],
        payload["facts"],
        keyframes_b64=payload["keyframes_b64"],
        points=payload["points"],
        trajectory=payload["trajectory"],
        label=label or recording_name,
        source="robot_recording",
    )

    session = Path(session_dir)
    stamp = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    derived: dict[str, str] = {}
    for name in ("report.html", "confidence.json", "consistency.json", "ingest_manifest.json"):
        src = session / name
        if not src.exists():
            continue
        try:
            key = persistence.save_derived_artifact(
                user_id, scan_id, f"demo/recordings/{stamp}/{name}", src.read_bytes()
            )
            derived[name] = key
        except Exception:
            continue
    return {"scan_id": scan_id, "qc_level": payload["qc_level"], "derived": derived}
