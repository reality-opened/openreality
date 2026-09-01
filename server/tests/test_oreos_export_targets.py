"""Hermetic tests for server.oreos.recordings.export_targets (real gated training-data output).

No network, no GPU, no modal, no vggt_slam, no open3d, no usd-core: a synthetic session
dir (tiny TUM trajectory + jpg frames + ASCII ply + manifests) goes through the real
``build_exports`` adapter -> ``export_from_record`` -> ``convert_to_groot_lerobot``.

Assertion bar mirrors docs/dataset-export.md §4.1/§5 and what
tests/test_export_to_lerobot.py asserts for the GR00T-LeRobot v2 layout. The Isaac
target's scale-honesty refusal ("no trustworthy scale") is checked BEFORE the usd-core
import, so it is testable in any env; the usd-core skip path is exercised when pxr is
absent (this env / CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from scipy.spatial.transform import Rotation

from server.oreos.recordings.export_targets import build_exports

FPS = 10.0


# -- fixture synthesis --------------------------------------------------------


def _quat_xyzw(rz: float) -> np.ndarray:
    return Rotation.from_euler("z", rz).as_quat()


def _tum_rows(n: int, t0: float = 1000.0):
    """(token, ts, pos(3), quat xyzw) per keyframe; a gentle curve with rotation."""
    rows = []
    for i in range(n):
        t = t0 + 0.1 * i
        pos = np.array([0.3 * i, 0.02 * i * i, 0.1 * np.sin(i)])
        quat = _quat_xyzw(0.1 * i)
        rows.append((f"{t:.6f}", t, pos, quat))
    return rows


def _ply_arrays(n: int = 16) -> tuple:
    """Deterministic fixture cloud (seeded) so tests can regenerate the expected arrays."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(-2.0, 2.0, size=(n, 3))
    cols = rng.integers(0, 255, size=(n, 3))
    return pts.astype(np.float32), cols.astype(np.uint8)


def _write_ascii_ply(path: Path, n: int = 16) -> tuple:
    pts, cols = _ply_arrays(n)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p, c in zip(pts, cols):
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")
    return pts, cols


def make_session(
    tmp_path: Path,
    n: int = 8,
    level: str = "high_confidence",
    export_allowed: bool = True,
    reference_available: bool = True,
    umeyama_scale=2.0,
    with_ply: bool = True,
    drop_frame: int | None = None,
) -> Path:
    sess = tmp_path / "synth_sess"
    (sess / "results").mkdir(parents=True)
    (sess / "frames").mkdir()

    rows = _tum_rows(n)
    tum_lines = []
    for i, (token, _t, pos, quat) in enumerate(rows):
        tum_lines.append(
            f"{token} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
            f"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}"
        )
        if i != drop_frame:
            img = np.full((24, 32, 3), (i * 29) % 255, np.uint8)
            img[:, : 4 + i] = (10, 200, 60)  # distinguishable content
            assert cv2.imwrite(str(sess / "frames" / f"{token}.jpg"), img)
    (sess / "results" / "est_tum.txt").write_text("\n".join(tum_lines) + "\n")

    if with_ply:
        _write_ascii_ply(sess / "results" / "map_preview.ply")

    (sess / "ingest_manifest.json").write_text(
        json.dumps({"source": "synthetic.db", "format": "dimos-memory2-sqlite", "n_frames": n})
    )
    (sess / "consistency.json").write_text(
        json.dumps(
            {
                "reference_available": reference_available,
                "umeyama_scale": umeyama_scale,
                "ate_rmse_m": 0.01,
                "ate_pct_extent": 0.5,
            }
        )
    )
    (sess / "confidence.json").write_text(
        json.dumps({"level": level, "export_allowed": export_allowed, "reasons": []})
    )
    return sess


def _build_or_skip(sess: Path, targets):
    """build_exports, skipping the test if this env has no mp4 VideoWriter."""
    res = build_exports(sess, targets=targets, fps=FPS)
    for t in res["targets"].values():
        if t.get("status") == "error" and "VideoWriter" in (t.get("reason") or ""):
            pytest.skip(f"no mp4 VideoWriter in this env: {t['reason']}")
    return res


def _count_mp4_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


# -- 1. OpenReality tree + trajectory correctness -----------------------------


def test_openreality_tree_and_trajectory(tmp_path: Path) -> None:
    n = 8
    sess = make_session(tmp_path, n=n)
    res = _build_or_skip(sess, ["lerobot"])

    assert res["session"] == "synth_sess"
    assert res["targets"]["lerobot"]["status"] == "ok"
    export_dir = Path(res["export_dir"])
    assert export_dir == sess / "bundle" / "openreality" / "synth_sess"

    # meta/info.json carries the up-to-scale caveats (docs/dataset-export.md §4.2)
    info = json.loads((export_dir / "meta" / "info.json").read_text())
    assert info["up_to_scale"] is True
    assert info["gravity_aligned"] is False
    assert info["metric"] is False
    for feat in ("observation.images.ego_view", "observation.state", "action",
                 "observation.intrinsics"):
        assert feat in info["features"]

    # provenance sidecar records the honest gaps
    prov = json.loads((export_dir / "meta" / "oreos_provenance.json").read_text())
    assert prov["confidence"]["level"] == "high_confidence"
    assert prov["t0_unix"] == pytest.approx(1000.0)
    assert any("intrinsics" in c for c in prov["caveats"])

    # trajectory.parquet: §4.1 columns + state/action math against the TUM input
    df = pd.read_parquet(export_dir / "data" / "trajectory.parquet")
    assert list(df.columns) == [
        "index", "frame_index", "episode_index", "timestamp", "source_frame_id",
        "observation.state", "action", "observation.intrinsics",
    ]
    assert len(df) == n

    rows = _tum_rows(n)
    poses = []
    for i, (_token, t, pos, quat) in enumerate(rows):
        state = np.asarray(df["observation.state"].iloc[i], dtype=np.float64)
        assert len(state) == 7
        np.testing.assert_allclose(state[:3], pos, atol=1e-5)
        # quaternion double cover: q and -q are the same rotation
        assert (np.allclose(state[3:], quat, atol=1e-5)
                or np.allclose(state[3:], -quat, atol=1e-5))
        assert df["timestamp"].iloc[i] == pytest.approx(i / FPS)
        assert df["source_frame_id"].iloc[i] == pytest.approx(t - rows[0][1], abs=1e-4)
        intr = np.asarray(df["observation.intrinsics"].iloc[i])
        assert len(intr) == 4 and np.all(intr == 0.0)  # honest gap: no intrinsics
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        poses.append(T)

    # action[i] == ego-motion to the next keyframe (§7.3); last row zeros
    for i in range(n - 1):
        T_rel = np.linalg.inv(poses[i]) @ poses[i + 1]
        expected = np.concatenate(
            [T_rel[:3, 3], Rotation.from_matrix(T_rel[:3, :3]).as_rotvec()]
        )
        np.testing.assert_allclose(
            np.asarray(df["action"].iloc[i], dtype=np.float64), expected, atol=1e-5
        )
    assert np.all(np.asarray(df["action"].iloc[n - 1]) == 0.0)

    # video: frame i == row i (§7.4) — count must match exactly
    assert _count_mp4_frames(export_dir / "videos" / "ego_view.mp4") == n

    # cloud round-trips the ply
    ply_pts, ply_cols = _ply_arrays()
    npz = np.load(export_dir / "clouds" / "cloud.npz")
    np.testing.assert_allclose(npz["positions"], ply_pts, atol=1e-4)
    np.testing.assert_array_equal(npz["colors"], ply_cols)


# -- 2. GR00T-LeRobot v2 layout (same bar as test_export_to_lerobot) ----------


def test_lerobot_v2_layout_and_indices(tmp_path: Path) -> None:
    n = 6
    sess = make_session(tmp_path, n=n)
    res = _build_or_skip(sess, ["lerobot"])
    ler = res["targets"]["lerobot"]
    assert ler["status"] == "ok"
    out = Path(ler["path"])
    assert out == sess / "bundle" / "lerobot"

    for rel in (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
    ):
        assert (out / rel).is_file(), f"missing {rel}"

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == n

    modality = json.loads((out / "meta" / "modality.json").read_text())
    assert modality["state"]["camera_pose"]["end"] - modality["state"]["camera_pose"]["start"] == 7
    assert modality["action"]["ego_motion"]["end"] - modality["action"]["ego_motion"]["start"] == 6
    assert modality["video"]["ego_view"]["original_key"] == "observation.images.ego_view"

    df = pd.read_parquet(out / "data" / "chunk-000" / "episode_000000.parquet")
    assert len(df) == n
    for col in ("observation.state", "action", "task_index",
                "annotation.human.action.task_description", "annotation.scene.caption"):
        assert col in df.columns
    assert len(df["observation.state"].iloc[0]) == 7
    assert len(df["action"].iloc[0]) == 6

    tasks = [json.loads(l) for l in (out / "meta" / "tasks.jsonl").read_text().splitlines()]
    valid_ids = {t["task_index"] for t in tasks}
    for col in ("task_index", "annotation.human.action.task_description",
                "annotation.scene.caption"):
        assert set(int(v) for v in df[col]).issubset(valid_ids)

    episodes = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes[0]["episode_index"] == 0 and episodes[0]["length"] == n

    # the mp4 transcoded 1:1
    assert _count_mp4_frames(
        out / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    ) == n


# -- 3. Isaac: scale-honesty refusal + usd-core skip --------------------------


def test_isaac_refuses_without_trustworthy_scale(tmp_path: Path) -> None:
    # (a) high_confidence but no reference: scale exists in no meaningful frame
    sess = make_session(tmp_path / "a", reference_available=False)
    res = _build_or_skip(sess, ["isaac"])
    isaac = res["targets"]["isaac"]
    assert isaac["status"] == "refused"
    assert "no trustworthy scale" in isaac["reason"]
    assert not (sess / "bundle" / "isaac").exists()

    # (b) reference available but verdict below high_confidence
    sess_b = make_session(tmp_path / "b", level="needs_review")
    res_b = _build_or_skip(sess_b, ["isaac"])
    assert res_b["targets"]["isaac"]["status"] == "refused"
    assert "no trustworthy scale" in res_b["targets"]["isaac"]["reason"]
    assert "high_confidence" in res_b["targets"]["isaac"]["reason"]

    # (c) missing umeyama_scale entirely
    sess_c = make_session(tmp_path / "c", umeyama_scale=None)
    res_c = _build_or_skip(sess_c, ["isaac"])
    assert res_c["targets"]["isaac"]["status"] == "refused"
    assert "no trustworthy scale" in res_c["targets"]["isaac"]["reason"]


def test_isaac_trusted_scale_skips_or_builds(tmp_path: Path) -> None:
    sess = make_session(tmp_path, umeyama_scale=2.5)
    res = _build_or_skip(sess, ["lerobot", "isaac"])
    assert res["targets"]["lerobot"]["status"] == "ok"
    isaac = res["targets"]["isaac"]
    try:
        import pxr  # noqa: F401

        have_usd = True
    except ImportError:
        have_usd = False
    if not have_usd:
        assert isaac["status"] == "skipped"
        assert "usd-core" in isaac["reason"]
        assert isaac["trusted_scale"] == pytest.approx(2.5)  # gate passed BEFORE the skip
    else:
        assert isaac["status"] == "ok"
        assert isaac["scale"] == pytest.approx(2.5)
        assert (Path(isaac["path"]) / "scene.usd").is_file()
        assert (Path(isaac["path"]) / "trajectory.usd").is_file()
        manifest = json.loads((Path(isaac["path"]) / "manifest.json").read_text())
        assert manifest["alignment"]["metric"] is True
        assert "odometry" in manifest["alignment"]["scale_source"]


# -- 4. failure honesty: no silent desync, no partial output ------------------


def test_missing_frame_is_an_error_not_a_desync(tmp_path: Path) -> None:
    sess = make_session(tmp_path, n=6, drop_frame=3)
    res = build_exports(sess, targets=["lerobot"], fps=FPS)
    ler = res["targets"]["lerobot"]
    assert ler["status"] == "error"
    assert "no frame" in ler["reason"]
    assert res["export_dir"] is None
    # nothing partial was written (frames are matched before any tree write)
    assert not (sess / "bundle" / "openreality").exists()


def test_unknown_target_reported(tmp_path: Path) -> None:
    sess = make_session(tmp_path, n=3)
    res = build_exports(sess, targets=["bogus"], fps=FPS)
    assert res["targets"]["bogus"]["status"] == "error"
    assert "unknown target" in res["targets"]["bogus"]["reason"]
    assert res["export_dir"] is None


def test_missing_ply_keeps_lerobot_but_refuses_isaac(tmp_path: Path) -> None:
    sess = make_session(tmp_path, with_ply=False)
    res = _build_or_skip(sess, ["lerobot", "isaac"])
    assert res["targets"]["lerobot"]["status"] == "ok"
    export_dir = Path(res["export_dir"])
    assert not (export_dir / "clouds" / "cloud.npz").exists()
    assert any("cloud.npz omitted" in c for c in res["caveats"])
    isaac = res["targets"]["isaac"]
    assert isaac["status"] == "refused"
    assert "cloud" in isaac["reason"]


# -- 5. pipeline wiring: stage_export produces the real bundle ----------------


def test_stage_export_wires_real_targets(tmp_path: Path) -> None:
    from server.oreos.recordings.pipeline import OreosPipeline

    sess = make_session(tmp_path, n=5)
    p = OreosPipeline(tmp_path / "unused.db", sess, session_name="synth")

    res = p.stage_export()

    assert res.status == "ok"
    ler_status = res.detail.get("targets", {}).get("lerobot")
    if ler_status == "error":  # env without an mp4 writer
        manifest = json.loads((sess / "bundle" / "bundle_manifest.json").read_text())
        if "VideoWriter" in manifest["targets"]["lerobot"].get("reason", ""):
            pytest.skip("no mp4 VideoWriter in this env")
    assert ler_status == "ok"

    bundle = sess / "bundle"
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["session"] == "synth"
    assert manifest["kind"] == "oreos-scene-bundle-v0"
    assert manifest["confidence"] == "high_confidence"
    assert manifest["targets"]["lerobot"]["status"] == "ok"
    assert isinstance(manifest["caveats"], list) and manifest["caveats"]

    # v0 provenance copies still present next to the real targets
    for name in ("est_tum.txt", "map_preview.ply", "consistency.json",
                 "confidence.json", "ingest_manifest.json"):
        assert (bundle / name).exists(), f"missing provenance copy {name}"
    # real outputs
    assert (bundle / "openreality" / "synth_sess" / "data" / "trajectory.parquet").is_file()
    assert (bundle / "lerobot" / "meta" / "modality.json").is_file()
    # success wrote the marker (refusal path is covered by test_oreos_pipeline)
    assert (sess / ".export_done").exists()
