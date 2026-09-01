"""CPU-only tests for the offline wrong-place hygiene gate (vggt_slam.hygiene).

Covers the pure-numpy RobustVGGT-F scoring, the mid-window anchor default, and the
input-shape coercion (tapped patch features vs raw aggregator tokens). The VGGT
feature-extraction path is lazy (torch + the vendored model) and NOT exercised here,
so this file runs on a CPU box with neither torch nor VGGT installed.

Run:  pytest core/tests/test_hygiene.py -v
"""

import numpy as np
import pytest

# conftest.py puts the repo root on sys.path.
from vggt_slam import hygiene as hg


def test_mid_anchor_index():
    assert hg.mid_anchor_index(16) == 8          # EXP-1 window default
    assert hg.mid_anchor_index(15) == 7
    assert hg.mid_anchor_index(1) == 0
    with pytest.raises(ValueError):
        hg.mid_anchor_index(0)


def test_scores_anchor_is_self_similar_and_bounded():
    rng = np.random.RandomState(0)
    S, P, C = 6, 40, 32
    feats = rng.randn(S, P, C).astype(np.float32)
    scores = hg.robustvggt_f_scores(feats, anchor_index=3)
    assert scores.shape == (S,)
    # cosine-based => within [-1, 1]
    assert float(scores.min()) >= -1.0 - 1e-5 and float(scores.max()) <= 1.0 + 1e-5
    # the anchor frame is the most self-similar: no other frame outscores it
    assert np.argmax(scores) == 3


def test_wrongscene_frame_flagged_sameplace_not():
    # Model the real regime EXP-1 saw: within a frame the dense patches share a common
    # "scene appearance" direction, so the anchor's mean descriptor is meaningful.
    # Same-place frames' patches align with the scene direction (high cosine); a
    # wrong-scene frame's patches align with an ORTHOGONAL direction (cosine ~0), the
    # signature RobustVGGT collapse EXP-1 measured (distractors -> ~0.0).
    rng = np.random.RandomState(42)
    S, P, C = 16, 50, 64
    scene = rng.randn(C).astype(np.float32); scene /= np.linalg.norm(scene)
    other = rng.randn(C).astype(np.float32)
    other -= (other @ scene) * scene            # make it orthogonal to the scene dir
    other /= np.linalg.norm(other)

    def frame(dir_vec):
        # low within-frame noise -> same-place cosine lands in EXP-1's 0.69-0.96 band
        return dir_vec[None, :] + 0.05 * rng.randn(P, C).astype(np.float32)

    feats = np.stack([frame(scene) for _ in range(S)])
    feats[11] = frame(other)                    # wrong-scene distractor at idx 11

    out = hg.score_window(feats)  # mid anchor default (idx 8), tau=0.65
    assert out["anchor_index"] == 8
    assert out["num_frames"] == 16
    # the different-place frame collapses below tau; same-place frames stay well above
    assert out["flagged_frames"] == [11]
    same_scores = [out["scores"][i] for i in range(S) if i != 11]
    assert out["scores"][11] < 0.65 < min(same_scores)


def test_score_window_accepts_raw_aggregator_tokens_4d():
    # raw tokens (B, S, P_total, C) with the leading special tokens included:
    # score_window must squeeze the batch and drop patch_start_idx specials.
    rng = np.random.RandomState(1)
    S, P, C = 8, 30, 48
    specials = hg.DEFAULT_PATCH_START_IDX
    base = rng.randn(P, C).astype(np.float32)
    patches = np.stack([base + 0.02 * rng.randn(P, C).astype(np.float32) for _ in range(S)])
    # special tokens are wildly different from patches; dropping them must not change scores
    special_tok = 100.0 * rng.randn(S, specials, C).astype(np.float32)
    raw = np.concatenate([special_tok, patches], axis=1)[None]  # (1, S, specials+P, C)
    assert raw.shape == (1, S, specials + P, C)

    from_raw = hg.score_window(raw)
    from_patches = hg.score_window(patches)
    assert from_raw["num_frames"] == S
    assert np.allclose(from_raw["scores"], from_patches["scores"], atol=1e-5)


def test_anchor_override_and_threshold():
    rng = np.random.RandomState(7)
    feats = rng.randn(10, 20, 16).astype(np.float32)
    out = hg.score_window(feats, anchor_index=0, threshold=1.01)  # threshold above 1 flags all
    assert out["anchor_index"] == 0
    assert out["flagged_frames"] == list(range(10))
    # a threshold below the minimum flags nothing
    out2 = hg.score_window(feats, anchor_index=0, threshold=-1.01)
    assert out2["flagged_frames"] == []


def test_torch_tensor_input_supported_if_torch_present():
    torch = pytest.importorskip("torch")
    try:  # some envs ship a torch whose numpy bridge is ABI-broken; skip cleanly there
        _ = torch.zeros(1).numpy()
    except Exception:
        pytest.skip("torch<->numpy bridge unavailable in this environment")
    rng = np.random.RandomState(3)
    feats = rng.randn(5, 25, 32).astype(np.float32)
    out_np = hg.score_window(feats, anchor_index=2)
    out_t = hg.score_window(torch.as_tensor(feats), anchor_index=2)
    assert np.allclose(out_np["scores"], out_t["scores"], atol=1e-5)
