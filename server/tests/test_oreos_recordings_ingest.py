"""Robot-recording ingest lane (merged Oreos pipeline): route + persistence bridge.

Route tests follow test_oreos_ingest_routes' blueprint-registration pattern; the
bridge test builds a synthetic session dir (TUM trajectory + frames + QC docs) and
asserts the persisted payload shape — no Modal, no GPU, no network.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
import pytest

from server.oreos.recordings.persist_scene import (
    build_recording_scene_payload,
    persist_recording_scene,
)
from server.scene_report.store import ModalScenePersistence


def _write_session(tmp_path: Path, n_poses: int = 6, with_ref: bool = True) -> Path:
    session = tmp_path / "session"
    (session / "results").mkdir(parents=True)
    (session / "frames").mkdir()
    lines = []
    for i in range(n_poses):
        t = 1000.0 + i * 0.5
        lines.append(f"{t:.6f} {i * 0.1:.4f} 0.0 0.0 0 0 0 1")
        # frame named by the timestamp token, tiny valid-enough JPEG bytes
        (session / "frames" / f"{t:.6f}.jpg").write_bytes(b"\xff\xd8\xff\xdbfake")
    (session / "results" / "est_tum.txt").write_text("\n".join(lines) + "\n")
    # ASCII PLY the export_targets fallback parser reads
    ply = session / "results" / "map_preview.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 2 1 0 0 255\n"
    )
    (session / "confidence.json").write_text(
        json.dumps({"level": "supported", "reasons": [], "export_allowed": True})
    )
    (session / "consistency.json").write_text(
        json.dumps({"reference_available": with_ref, "ate_rmse_m": 0.042, "ate_pct_extent": 2.1})
    )
    (session / "ingest_manifest.json").write_text(
        json.dumps({"n_frames": n_poses, "span_s": 3.0, "reference_usable": with_ref})
    )
    (session / "report.html").write_text("<html>qc</html>")
    return session


def test_bridge_builds_full_save_scene_payload(tmp_path):
    session = _write_session(tmp_path)
    payload = build_recording_scene_payload(session, "go2_short.db")

    assert payload["report"].degraded is True
    assert "robot" in payload["report"].summary.lower()
    assert "0.042" in payload["report"].summary  # consistency ATE quoted honestly
    assert payload["facts"].metrics.point_count == 3
    traj = payload["trajectory"]
    assert traj["poses"].shape == (6, 4, 4)
    assert traj["intrinsics"].shape == (6, 4) and float(traj["intrinsics"].sum()) == 0.0
    assert traj["source_frame_id"].dtype == np.float32
    kfs = payload["keyframes_b64"]
    assert 0 < len(kfs) <= 48
    assert base64.b64decode(kfs[0]["image_b64"]).startswith(b"\xff\xd8")


def test_bridge_without_reference_says_unscored(tmp_path):
    session = _write_session(tmp_path, with_ref=False)
    (session / "consistency.json").write_text(json.dumps({"reference_available": False}))
    payload = build_recording_scene_payload(session, "clip.db")
    assert "consistency unscored" in payload["report"].summary.lower()


def test_bridge_refuses_sessions_without_trajectory(tmp_path):
    session = tmp_path / "empty"
    (session / "results").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_recording_scene_payload(session, "x.db")


def test_persist_writes_scene_and_qc_derived_artifacts(tmp_path):
    session = _write_session(tmp_path)
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))

    result = persist_recording_scene(store, "user-a", "scan-r1", session, "go2_short.db")

    record = store.get_scene("user-a", "scan-r1")
    assert record is not None
    assert record.get("source") == "robot_recording"
    assert record.get("trajectory_key")
    assert result["qc_level"] == "supported"
    # QC bundle landed under the demo derived namespace (persisted-key contract)
    assert any(k.startswith("derived/demo/recordings/") for k in result["derived"].values())
    for key in result["derived"].values():
        path = store.get_derived_artifact_path("user-a", "scan-r1", key)
        assert path and os.path.isfile(path)


def test_recording_route_rejects_wrong_suffix_and_spawns_on_db(monkeypatch, tmp_path):
    flask = pytest.importorskip("flask")
    import server.oreos as oreos_pkg
    from server.oreos import routes_recordings as rr

    app = flask.Flask(__name__)
    try:
        app.register_blueprint(oreos_pkg.oreos_bp)
    except Exception:
        pass  # already registered by another test module in this process

    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    monkeypatch.setattr(oreos_pkg, "auth_user_id", lambda: "user-a", raising=False)
    monkeypatch.setattr(rr, "auth_user_id", lambda: "user-a")
    monkeypatch.setattr(rr, "scene_persistence", lambda: store)
    spawned: list[dict] = []
    rr.configure_recording_spawner(lambda **kw: spawned.append(kw))
    try:
        client = app.test_client()

        bad = client.post(
            "/api/workspace/ingest/recording",
            data=b"x",
            headers={"X-Upload-Filename": "clip.mp4"},
        )
        assert bad.status_code == 415

        ok = client.post(
            "/api/workspace/ingest/recording",
            data=b"sqlite-bytes",
            headers={"X-Upload-Filename": "go2_short.db"},
        )
        assert ok.status_code == 202
        body = ok.get_json()
        assert body["job_id"] and body["scan_id"]
        assert len(spawned) == 1
        assert spawned[0]["upload_rel_path"].split("/")[1] == "_uploads"
        assert spawned[0]["recording_name"] == "go2_short.db"
    finally:
        rr.configure_recording_spawner(None)
