"""Coverage for the detection-recall knobs (env-configurable CLIP/SAM thresholds +
frames-per-window). See ``server.streaming_slam._detection_env_defaults`` and the
window-quality study's detection-recall finding (finc: 5/12 discovered objects grounded;
chorus-1: 4) that motivated making these tunable instead of hardcoded.

``_detection_env_defaults`` is a pure env-reading function, split out of
``StreamingSLAM.__init__`` precisely so it's testable without constructing a full
``StreamingSLAM`` (GPU model load). The module import itself still needs
``conftest.load_streaming_slam_module`` (stubs torch/vggt_slam/vggt) since
``server/streaming_slam.py`` imports those at module top level.
"""

from __future__ import annotations

from conftest import load_streaming_slam_module


def test_detection_env_defaults_match_historical_hardcoded_values(monkeypatch):
    """Unset env -> exactly the pre-fix hardcoded defaults (0.15 / 0.80 / 1): this commit adds
    knobs, not new defaults."""
    for var in ("DETECTION_CLIP_THRESHOLD", "DETECTION_SAM_THRESHOLD", "DETECTION_FRAMES_PER_WINDOW"):
        monkeypatch.delenv(var, raising=False)
    mod = load_streaming_slam_module(monkeypatch)
    clip, sam, frames = mod._detection_env_defaults()
    assert clip == 0.15
    assert sam == 0.80
    assert frames == 1


def test_detection_env_defaults_reads_overrides(monkeypatch):
    monkeypatch.setenv("DETECTION_CLIP_THRESHOLD", "0.10")
    monkeypatch.setenv("DETECTION_SAM_THRESHOLD", "0.6")
    monkeypatch.setenv("DETECTION_FRAMES_PER_WINDOW", "3")
    mod = load_streaming_slam_module(monkeypatch)
    clip, sam, frames = mod._detection_env_defaults()
    assert clip == 0.10
    assert sam == 0.6
    assert frames == 3


def test_streaming_slam_new_instance_picks_up_env_defaults(monkeypatch):
    """A freshly-``__new__``'d instance re-running __init__'s (extracted) detection-config
    step reflects the env, matching what a real __init__ call would set."""
    monkeypatch.setenv("DETECTION_SAM_THRESHOLD", "0.55")
    monkeypatch.delenv("DETECTION_CLIP_THRESHOLD", raising=False)
    monkeypatch.delenv("DETECTION_FRAMES_PER_WINDOW", raising=False)
    mod = load_streaming_slam_module(monkeypatch)

    slam = mod.StreamingSLAM.__new__(mod.StreamingSLAM)
    clip, sam, frames = mod._detection_env_defaults()
    slam.detection_clip_thresholds = {"default": clip}
    slam.detection_sam_thresholds = {"default": sam}
    slam.SAM_TOP_K_FRAMES = frames

    assert slam.detection_clip_thresholds == {"default": 0.15}  # untouched default
    assert slam.detection_sam_thresholds == {"default": 0.55}  # overridden
    assert slam.SAM_TOP_K_FRAMES == 1  # class default, unset env
    # class-level default is unaffected -- only this instance was configured from env
    assert mod.StreamingSLAM.SAM_TOP_K_FRAMES == 1
