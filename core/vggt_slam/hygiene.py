"""Wrong-place hygiene gate for SLAM scan windows -- OFFLINE, not the live loop.

Reimplements the **RobustVGGT-F** score of *RobustVGGT* (arXiv 2512.04012) -- a
training-free per-frame outlier signal read straight off stock VGGT internals (no
fine-tune, no extra head). For a window of frames it measures, per frame, how much
that frame looks like the **same place** as an anchor frame, from the final
alternating-attention layer's dense patch features.

Why it lives here and what it is *for*
--------------------------------------
EXP-1 (2026-07-05, ``experiments/research/2026-07-05-exp1-robustvggt-gate.md``,
platform repo) tested this on our own data and found a sharp, load-bearing result:

* It separates a **different place** (a wrong-scene distractor / an elevator-ride
  discontinuity / a stray frame that landed in the wrong scan of a multi-clip
  aggregation) essentially perfectly -- score collapses to ~0 -- at a fixed
  threshold ``tau ~= 0.65`` with a **mid-window anchor**.
* It does **NOT** catch same-scene blur / defocus / exposure (those move the score
  ~20-60x less than a distractor, *inside* the healthy viewpoint-drift band). So this
  is a **wrong-place detector, not a frame-quality gate** -- use blur-native signals
  (Laplacian variance / the existing disparity + depth-confidence gates) for that.
* The **mid-window anchor is necessary**: a frame-0 anchor *inverts* on bad tour
  starts (the bad first frames become the window maximum). Default anchor is
  ``num_frames // 2`` for that reason.

Intended use: an **offline** hygiene pass over captured windows to flag elevator
discontinuities / wrong-unit contamination before they poison a scan or a Chorus-style
multi-scan aggregation. It is deliberately **not** wired into the live SLAM loop
(``main.py`` / ``Solver``); import it explicitly to use it.

Scoring is our own ~30-line reimplementation (RobustVGGT's repo carries no license;
no code was copied). Module import is numpy-only / CPU-safe -- ``torch`` and the VGGT
model are imported lazily, only inside :func:`extract_patch_features`.
"""

import numpy as np

# EXP-1-tuned threshold: flag a frame as a DIFFERENT place if its same-place score is
# below this. tau ~= 0.65 gave perfect precision/recall separating wrong-scene
# distractors from clean frames, with a comfortable margin (distractors sit at ~0.0,
# clean frames >= ~0.69 even across a full viewpoint pan). See EXP-1 Result 2.
DEFAULT_WRONGSCENE_THRESHOLD = 0.65

# VGGT prepends special (camera + register) tokens ahead of the dense patch tokens;
# EXP-1 confirmed patch_start_idx=5 for VGGT-1B (5 specials -> 1369 = 37x37 patches).
DEFAULT_PATCH_START_IDX = 5

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# core scoring (pure numpy, deterministic, CPU-safe)
# --------------------------------------------------------------------------- #
def mid_anchor_index(num_frames):
    """The mid-window anchor frame index (``num_frames // 2``).

    EXP-1 Result 5: a frame-0 anchor inverts on bad tour starts (bad first frames
    become the window *maximum*); the mid-window anchor is load-bearing.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return num_frames // 2


def robustvggt_f_scores(patch_features, anchor_index):
    """RobustVGGT-F per-frame same-place score against an anchor descriptor.

    Args:
        patch_features: ``(S, P, C)`` dense patch features from VGGT's **final**
            alternating-attention layer, with the special tokens already dropped
            (``S`` frames, ``P`` patches, ``C`` channels).
        anchor_index: index of the reference frame.

    Returns:
        ``(S,)`` float32 in ``[-1, 1]``: each frame's patches are L2-normalized, the
        anchor descriptor is the unit-mean of the anchor frame's unit patches, and the
        score is the pixel-averaged cosine of a frame's patches to that descriptor.
        Low => the frame looks like a *different place* than the anchor.
    """
    f = np.asarray(patch_features, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"expected (S, P, C) patch features, got shape {f.shape}")
    S = f.shape[0]
    if not (0 <= anchor_index < S):
        raise ValueError(f"anchor_index {anchor_index} out of range for {S} frames")
    fhat = f / np.clip(np.linalg.norm(f, axis=-1, keepdims=True), _EPS, None)  # unit patches
    d = fhat[anchor_index].mean(axis=0)                     # (C,) anchor mean descriptor
    d = d / max(float(np.linalg.norm(d)), _EPS)             # unit
    return (fhat @ d).mean(axis=1).astype(np.float32)       # (S,) pixel-averaged cosine


def score_window(features_or_frames, anchor_index=None,
                 threshold=DEFAULT_WRONGSCENE_THRESHOLD,
                 patch_start_idx=DEFAULT_PATCH_START_IDX, model=None, device=None):
    """Score one window for wrong-place (different-scene) contamination.

    OFFLINE hygiene helper -- not part of the live SLAM loop.

    Args:
        features_or_frames: one of
            * ``(S, P, C)`` precomputed final-layer **patch** features (numpy or a
              torch tensor, 3-D) -- scored directly (the fast, dependency-free path);
            * ``(B, S, P_total, C)`` raw aggregator tokens (4-D) *including* the
              special tokens -- batch squeezed and the leading ``patch_start_idx``
              tokens dropped, then scored;
            * a window of **frames** -- an ``(S, 3, H, W)`` / ``(S, H, W, 3)`` image
              array, a list of image paths, or an image-folder path -- run through
              VGGT's aggregator via :func:`extract_patch_features` (lazy torch + VGGT).
        anchor_index: reference frame; defaults to the **mid-window** anchor.
        threshold: flag a frame if its score ``< threshold`` (default EXP-1 tau=0.65).
        patch_start_idx: number of leading special tokens to drop when raw aggregator
            tokens / frames are given (default 5, VGGT-1B).
        model, device: optional, forwarded to :func:`extract_patch_features` for the
            frames path (reuse a loaded model / pick a device).

    Returns:
        ``dict`` with ``scores`` (per-frame float list), ``flags`` (per-frame bool),
        ``flagged_frames`` (indices scored below threshold => a *different place*),
        ``anchor_index``, ``threshold`` and ``num_frames``.
    """
    feats = _to_patch_features(features_or_frames, patch_start_idx=patch_start_idx,
                               model=model, device=device)
    S = int(feats.shape[0])
    anchor_index = mid_anchor_index(S) if anchor_index is None else int(anchor_index)
    scores = robustvggt_f_scores(feats, anchor_index)
    flags = scores < float(threshold)
    return {
        "scores": [float(x) for x in scores],
        "flags": [bool(x) for x in flags],
        "flagged_frames": [int(i) for i in np.nonzero(flags)[0]],
        "anchor_index": anchor_index,
        "threshold": float(threshold),
        "num_frames": S,
    }


def _to_patch_features(x, patch_start_idx=DEFAULT_PATCH_START_IDX, model=None, device=None):
    """Coerce the many accepted inputs into ``(S, P, C)`` patch features (numpy)."""
    if hasattr(x, "detach"):          # torch tensor -> numpy
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 3:               # (S, P, C) patch features -- already tapped
            return x
        if x.ndim == 4 and x.shape[1] not in (1, 3) and x.shape[-1] not in (1, 3):
            # (B, S, P_total, C) raw aggregator tokens -> squeeze batch, drop specials
            return x[0][:, patch_start_idx:, :]
        # else: an image array (S,3,H,W)/(S,H,W,3) -> extract below
    return extract_patch_features(x, patch_start_idx=patch_start_idx, model=model, device=device)


# --------------------------------------------------------------------------- #
# optional feature extraction (lazy torch + VGGT; never imported at module load)
# --------------------------------------------------------------------------- #
def extract_patch_features(images, patch_start_idx=DEFAULT_PATCH_START_IDX,
                           model=None, device=None):
    """Run VGGT's aggregator on a window of frames -> ``(S, P, C)`` patch features.

    Lazy/guarded so importing this module stays numpy-only and CPU-safe. Matches
    EXP-1's tap exactly: stock VGGT-1B, **eval** track, 518 px square crop, the
    aggregator's **final** alternating-attention layer, special tokens dropped.

    Args:
        images: an image-folder path, a list of image paths, or a preprocessed
            ``(S, 3, H, W)`` / ``(S, H, W, 3)`` tensor / array.
        patch_start_idx: leading special tokens to drop (overridden by the value the
            aggregator itself reports, when available).
        model: an already-loaded VGGT model with ``.aggregator(...)``; loaded from
            ``facebook/VGGT-1B`` if ``None``.
        device: torch device string (defaults to cuda if available, else cpu).

    Returns:
        ``(S, P, C)`` float32 numpy patch features.
    """
    import os
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve images -> a preprocessed (S, 3, H, W) tensor.
    imgs = images
    if isinstance(images, str) or (isinstance(images, (list, tuple))
                                   and images and isinstance(images[0], str)):
        from vggt.utils.load_fn import load_and_preprocess_images
        if isinstance(images, str) and os.path.isdir(images):
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            paths = sorted(os.path.join(images, f) for f in os.listdir(images)
                           if f.lower().endswith(exts))
        else:
            paths = list(images)
        imgs = load_and_preprocess_images(paths, mode="crop")
    else:
        imgs = torch.as_tensor(np.asarray(imgs))
        if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):  # (S,H,W,3) -> (S,3,H,W)
            imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.float()
        if float(imgs.max()) > 1.0:
            imgs = imgs / 255.0
    imgs = imgs.to(device)

    if model is None:
        from vggt.models.vggt import VGGT
        try:
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        except Exception:  # older fork: url-loaded state dict
            model = VGGT()
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model.load_state_dict(torch.hub.load_state_dict_from_url(url))
        model.eval().to(device)

    with torch.no_grad():
        agg_out = model.aggregator(imgs.unsqueeze(0) if imgs.dim() == 4 else imgs)
    # aggregator returns (tokens_list, patch_start_idx, ...)
    tokens_list, reported_start = agg_out[0], int(agg_out[1])
    final = tokens_list[-1]                       # (B, S, P_total, C)
    if final.dim() == 4:
        final = final[0]
    start = reported_start if reported_start else patch_start_idx
    return final[:, start:, :].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# OPTIONAL offline CLI: python -m vggt_slam.hygiene ...  (NOT the live loop)
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="OFFLINE wrong-place hygiene gate (RobustVGGT-F). Flags frames that "
                    "look like a DIFFERENT place than the window's mid anchor -- elevator "
                    "discontinuities / wrong-unit contamination. Not a blur/quality gate. "
                    "NOT part of the SLAM loop.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_folder", type=str,
                     help="Folder of a window's frames; run VGGT to extract features (GPU).")
    src.add_argument("--features_npz", type=str,
                     help="npz with a (S,P,C) 'features' array (skips the VGGT forward pass).")
    ap.add_argument("--anchor", type=int, default=None,
                    help="Anchor frame index (default: mid-window, num_frames//2).")
    ap.add_argument("--threshold", type=float, default=DEFAULT_WRONGSCENE_THRESHOLD,
                    help=f"Flag frame if score < threshold (default {DEFAULT_WRONGSCENE_THRESHOLD}).")
    ap.add_argument("--json", action="store_true", help="Print the result dict as JSON.")
    args = ap.parse_args(argv)

    if args.features_npz:
        feats = np.load(args.features_npz)["features"]
        result = score_window(feats, anchor_index=args.anchor, threshold=args.threshold)
    else:
        result = score_window(args.image_folder, anchor_index=args.anchor,
                              threshold=args.threshold)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        a = result["anchor_index"]
        print(f"wrong-place hygiene | frames={result['num_frames']} anchor(mid)={a} "
              f"tau={result['threshold']}")
        for i, (s, fl) in enumerate(zip(result["scores"], result["flags"])):
            mark = "  <-- WRONG PLACE" if fl else ""
            anch = " (anchor)" if i == a else ""
            print(f"  f{i:02d}  score={s:.3f}{anch}{mark}")
        if result["flagged_frames"]:
            print(f"flagged (different place): {result['flagged_frames']}")
        else:
            print("no wrong-place frames flagged.")
    return result


if __name__ == "__main__":
    _main()
