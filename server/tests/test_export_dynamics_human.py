"""Unit tests for the ``dynamics/human/`` world-motion sidecar (EXP-2 port).

Ground-truth checks with SYNTHETIC fixtures (EXP-2's validate-against-synthetic approach): a
STATIC world skeleton seen by a moving synthetic camera, round-tripped through the documented
SAM 3D Body output format (root-relative meters + a deliberately-corrupted monocular ``tz``),
then anchored back. A static person MUST anchor to a (near-)stationary world trajectory — an
end-to-end check of the convention + scale math with no real weights / SLAM run / GPU.

GPU-free / numpy-only: no torch, no vggt_slam — runs green in CI.

Reference numbers (from the committed EXP-2 metadata JSONs, for orientation — NOT asserted
here since they come from real data with the real MHR rest pose):
``platform:experiments/exp2_sam3dbody_motion/results/{tum_96f_metrics.json,dryrun_metrics.json}``
report λ_cv ≈ 0.045–0.056, bone-length CV mean ≈ 0.03, λ_person/λ_gt ≈ 0.88 (TUM mocap).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from server.export import dynamics_human as dh


# ---------------------------------------------------------------------------
# anchoring / scale math (ground truth)
# ---------------------------------------------------------------------------

def test_selftest_passes():
    """The built-in ground-truth self-test (λ within 10%, joint err < 10% of height)."""
    assert dh._selftest(seed=0) is True


@pytest.mark.parametrize("lam_true", [0.35, 1.0, 1.7])
def test_lambda_recovers_across_scales(lam_true):
    """λ is recovered from the person's body size across a range of true SLAM scales,
    including TUM's ≈0.35 u/m. This is the monocular-scale question, answered against GT."""
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(seed=1, lam_true=lam_true)
    res = dh.anchor_sequence(poses, Ks, depth, conf, per_frame)
    assert abs(res["lam"] - lam_true) / lam_true < 0.1
    assert res["lam_stats"]["lam_cv"] < 0.1          # per-frame stability


def test_static_person_anchors_stationary():
    """A STATIC world skeleton must anchor to a near-stationary world trajectory (the failure
    mode this guards against is joints dragging along with the camera = broken anchoring)."""
    poses, Ks, depth, conf, per_frame, gt = dh.synthetic_fixture(seed=2, lam_true=1.7)
    res = dh.anchor_sequence(poses, Ks, depth, conf, per_frame)
    world = res["world_raw"]
    err = np.linalg.norm(world - gt, axis=-1)
    assert np.nanmean(err) < 0.1 * 1.7 * 1.6         # < 10% of person height
    assert np.nanstd(world, axis=0).mean() < 0.05    # near-stationary in world


def test_no_axis_flip_reprojection():
    """Anchored WORLD joints reproject (through each frame's SLAM pose) back onto 3DB's own 2D
    keypoints — the whole OpenCV-both-sides convention chain, in pixels. With zero joint noise
    and λ pinned to ground truth the round-trip is sub-pixel (the residual is the pelvis
    2D-midpoint-vs-projected-3D-midpoint nonlinearity, ~0.2px); an axis flip would blow this
    to hundreds of px, so a sub-pixel tolerance is the real convention proof."""
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(
        seed=3, lam_true=1.7, joint_noise_m=0.0)
    res = dh.anchor_sequence(poses, Ks, depth, conf, per_frame, lam=1.7)
    world = res["world_raw"]
    px_err = []
    for i in range(world.shape[0]):
        if not res["valid"][i]:
            continue
        w2c = np.linalg.inv(poses[i])
        cam = world[i] @ w2c[:3, :3].T + w2c[:3, 3]
        uv = dh.project(cam, Ks[i])
        px_err.append(np.linalg.norm(uv - per_frame[i]["pred_keypoints_2d"], axis=-1).mean())
    assert np.mean(px_err) < 1.0                      # sub-pixel round-trip (no flip, correct K)


def test_smoothing_reduces_jerk():
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(seed=4, lam_true=1.7)
    world = dh.anchor_sequence(poses, Ks, depth, conf, per_frame)["world_raw"]
    sm = dh.smooth_oneeuro(world, fps=10.0)
    assert dh.mean_jerk(sm[:, :17], 10.0) < dh.mean_jerk(world[:, :17], 10.0)


def test_gravity_down_is_camera_plus_y():
    """Upright cameras => world 'down' ≈ +Y (world = frame-0 camera frame, +Y image-down)."""
    poses, *_ = dh.synthetic_fixture(seed=5)
    g = dh.estimate_gravity_down(poses)
    assert abs(np.linalg.norm(g) - 1.0) < 1e-6
    assert g[1] > 0.95                                # dominated by +Y


# ---------------------------------------------------------------------------
# serialization (schema)
# ---------------------------------------------------------------------------

def _write(tmp_path, seed=0, lam_true=1.7, knock=()):
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(seed=seed, lam_true=lam_true)
    for i in knock:                                  # simulate no-detection frames
        per_frame[i]["pred_keypoints_3d"] = np.full((70, 3), np.nan, np.float32)
        per_frame[i]["pred_keypoints_2d"] = np.full((70, 2), np.nan, np.float32)
        per_frame[i]["bbox"] = np.full(4, np.nan, np.float32)
    summary = dh.write_human_sidecar(str(tmp_path), poses, Ks, depth, conf, per_frame,
                                     fps=3.0, person_id=0)
    return summary


def test_sidecar_files_and_summary(tmp_path):
    summary = _write(tmp_path)
    assert summary["human_tracks_path"].endswith("dynamics/human/human_tracks.jsonl")
    assert summary["mhr_params_path"].endswith("dynamics/human/mhr_params.npz")
    assert summary["schema_version"] == dh.DYNAMICS_HUMAN_SCHEMA_VERSION
    # λ and gravity are explicit in the summary
    assert summary["lambda_slam_per_m"] is not None
    assert summary["gravity_world"] is not None and len(summary["gravity_world"]) == 3


def test_single_person_one_jsonl_line(tmp_path):
    summary = _write(tmp_path)
    lines = [l for l in open(summary["human_tracks_path"]) if l.strip()]
    assert len(lines) == 1                            # single-person v1


def test_jsonl_schema_fields_and_shapes(tmp_path):
    summary = _write(tmp_path)
    line = json.loads(open(summary["human_tracks_path"]).readline())
    for field in ("person_id", "body_model", "skeleton", "frames", "joints_world",
                  "root_world", "frame_visibility", "frame_confidence", "bbox_2d",
                  "lambda_slam_per_m", "gravity_world", "smoothing", "first_frame",
                  "last_frame", "mhr_ref", "up_to_scale", "gravity_aligned"):
        assert field in line, f"missing {field}"
    assert line["body_model"] == "mhr" and line["skeleton"] == "mhr70"
    assert line["up_to_scale"] is True and line["gravity_aligned"] is False
    assert line["mhr_ref"] == "p0"
    assert line["smoothing"]["method"] == "one_euro"
    assert {"min_cutoff", "beta", "fps"} <= set(line["smoothing"])
    # an anchored joints row is (70,3)
    first_ok = next(j for j in line["joints_world"] if j is not None)
    assert len(first_ok) == 70 and len(first_ok[0]) == 3


def test_gap_entries_are_null_not_fabricated(tmp_path):
    knock = (5, 6, 20)
    summary = _write(tmp_path, knock=knock)
    line = json.loads(open(summary["human_tracks_path"]).readline())
    for i in knock:
        assert line["joints_world"][i] is None        # honest gap, no fabrication
        assert line["root_world"][i] is None
        assert line["frame_visibility"][i] is False
        assert line["bbox_2d"][i] is None
    # smoothed variant IS emitted (gap-filled to feed the filter) but every filled row is
    # exactly a frame_visibility=False row, so consumers can drop them.
    assert "joints_world_smoothed" in line
    assert line["joints_world_smoothed"][5] is not None


def test_jsonl_is_strict_json_no_nan(tmp_path):
    """The written line must be valid JSON (no NaN/Infinity tokens) — consumers choke on them."""
    summary = _write(tmp_path, knock=(5, 6))
    raw = open(summary["human_tracks_path"]).read()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=_reject)           # raises if NaN/Infinity present


def _reject(tok):                                     # pragma: no cover - only fires on bad JSON
    raise AssertionError(f"non-finite JSON constant leaked: {tok}")


def test_first_last_frame_track_observed_span(tmp_path):
    # knock out the first two and last two frames -> observed span shrinks inward
    summary = _write(tmp_path, knock=(0, 1, 38, 39))
    line = json.loads(open(summary["human_tracks_path"]).readline())
    assert line["first_frame"] >= 2
    assert line["last_frame"] <= 37


def test_mhr_params_npz_keys_and_shapes(tmp_path):
    summary = _write(tmp_path)
    z = np.load(summary["mhr_params_path"])
    keys = set(z.keys())
    assert "p0/pred_keypoints_3d_cam" in keys
    assert z["p0/pred_keypoints_3d_cam"].shape == (40, 70, 3)
    assert "p0/pred_cam_t" in keys and z["p0/pred_cam_t"].shape == (40, 3)
    # LICENSE GUARD: v1 ships MHR params + world joints ONLY. No SMPL-X arrays — the retarget
    # is a documented consumer-side conversion (SMPL-X model license is research-only).
    assert not any("smpl" in k.lower() for k in keys)


def test_mhr_params_gap_frames_are_nan(tmp_path):
    summary = _write(tmp_path, knock=(5, 6, 20))
    z = np.load(summary["mhr_params_path"])
    kp = z["p0/pred_keypoints_3d_cam"]
    for i in (5, 6, 20):
        assert np.isnan(kp[i]).all()                  # gap frame -> NaN, not zeros


def test_frames_length_mismatch_raises(tmp_path):
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(seed=0)
    with pytest.raises(ValueError):
        dh.write_human_sidecar(str(tmp_path), poses, Ks, depth, conf, per_frame,
                               frames=list(range(3)))   # wrong length


def test_body_results_adapter_roundtrip():
    """The S-major-dict -> per-frame-list adapter preserves the shared clock and passes MHR
    params through, so the Modal npz plugs straight into anchor_sequence."""
    poses, Ks, depth, conf, per_frame, _gt = dh.synthetic_fixture(seed=0)
    S = len(per_frame)
    stacked = {
        "pred_keypoints_3d": np.stack([p["pred_keypoints_3d"] for p in per_frame]),
        "pred_keypoints_2d": np.stack([p["pred_keypoints_2d"] for p in per_frame]),
        "pred_cam_t": np.stack([p["pred_cam_t"] for p in per_frame]),
        "det_score": np.array([p["det_score"] for p in per_frame]),
    }
    rebuilt = dh.body_results_to_per_frame(stacked)
    assert len(rebuilt) == S
    np.testing.assert_allclose(rebuilt[0]["pred_keypoints_3d"], per_frame[0]["pred_keypoints_3d"])
    assert math.isclose(float(rebuilt[3]["det_score"]), float(per_frame[3]["det_score"]))
