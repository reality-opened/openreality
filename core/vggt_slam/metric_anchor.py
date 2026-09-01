"""Per-submap metric scale anchor (EXP-12) — DEFAULT OFF, opt-in via ``--metric_scale_anchor``.

Attacks the SL(4)->Sim3 near-metric gap: each submap is internally metric-consistent, but the
composed global frame accumulates monotonic scale drift at submap boundaries because scale is
fixed only relative to the predecessor (a chain with one gauge prior on submap 0). This module
adds an **absolute, non-chaining 1-DOF per-submap scale reference** from a commercially-licensed
monocular metric-depth model and feeds it into the *same* scale DOF ``add_edge`` already owns
(``estimate_scale_pairwise``), as a gated drop-in correction.

RELATIVE cross-submap drift removal ONLY. The license-clean model
(Depth-Anything-V2-Metric-Hypersim-Large, Apache-2.0 end-to-end) is highly *consistent*
(across-frame CoV 3.6%, within-frame shape 4.0% on TUM) but carries a measured **+16.5% absolute
metric bias** on real indoor TUM vs mocap GT (EXP-12 §4). We therefore use the RATIO form
(``s_metric = r_curr / r_prev``), which cancels any constant multiplicative bias — absolute
metres are **NOT claimed** from the anchor alone. For true metres, add a one-known-dimension
calibration or upgrade to Metric3D-v2 (pending license). See
``experiments/research/2026-07-06-exp12-metric-anchor.md`` (§2 method, §6 integration + gate).

Design (EXP-12 §6, form 1 — minimal drop-in at ``add_edge``):
  * per submap i:  r_i = median over K keyframes of median_valid( metric_metres / slam_depth )
                       = METRES per LOCAL (submap-i-gauge) unit.
  * cross-submap metric scale (prior-gauge per current-gauge, matching what
    ``estimate_scale_pairwise`` returns):  s_metric = r_curr / r_prev.
  * a 4-tier sanity gate decides whether to trust the anchor over the geometric overlap scale.

Live-pipeline note: ``vggt.utils.geometry.unproject_depth_map_to_point_map`` returns
**camera-frame** points (the SPARK quirk, see ``vggt-omega-slam/vggt/utils/geometry.py``), so a
submap's ``pointclouds[fi][..., 2]`` IS the per-pixel SLAM camera z-depth directly — no z-buffer
of a merged cloud is needed (EXP-12's offline harness z-buffered only because its npz stored a
merged cloud). The RGB fed to the depth model is ``submap.colors[fi]`` (raw uint8, exactly the
``imgs`` EXP-12 fed to the model).

Everything above the ``MetricScaleAnchor`` class is pure numpy (unit-tested, import-clean with
only numpy). torch / cv2 / huggingface_hub / the Depth-Anything-V2 code are imported lazily
inside the model path so this module stays importable in CI / CPU test envs.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (EXP-12 §6 "Config flag"). Surfaced on Solver / CLI; all opt-in.
# ---------------------------------------------------------------------------
MODEL_MAX_DEPTH = 20.0              # DA-V2 indoor Hypersim head; pixels near the cap are dropped
CAP_FRACTION = 0.98                 # tier-1: drop metric pixels >= CAP_FRACTION * max_depth
DEFAULT_KEYFRAMES = 3               # K representative keyframes per submap (evenly spaced)
DEFAULT_GATE = 0.5                  # tier-3: max |ln(s_metric / scale_geom)| to auto-agree
DEFAULT_MIN_VALID = 200             # tier-1: min shared valid pixels per keyframe for a ratio
DEFAULT_WITHIN_COV = 0.15           # tier-2: skip anchor if K-frame ratio CoV >= this
DEFAULT_THIN_OVERLAP = 400          # tier-3: below this many overlap pixels the geometry, not
                                    #         the anchor, is the suspect on disagreement
DEFAULT_MODEL = "da2-metric-hypersim-large"
ABS_BIAS_NOTE = "+16.5% absolute (TUM vs mocap, EXP-12 §4) — relative/ratio use only"

# Registry of supported metric-depth models. DA-V2-Metric-Hypersim-Large is the license winner
# (Apache-2.0 end-to-end, indoor-trained). Metric3D-v2 is the named accuracy upgrade pending a
# written weights-license confirmation (EXP-12 §1); add it here when cleared.
_MODEL_REGISTRY = {
    "da2-metric-hypersim-large": {
        "hf_repo": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        "ckpt": "depth_anything_v2_metric_hypersim_vitl.pth",
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "max_depth": 20.0,
        "input_size": 518,
    },
}


# ===========================================================================
# Pure-numpy math (ports EXP-12 analyze.py exactly; unit-tested)
# ===========================================================================
def cov(x):
    """Coefficient of variation std/mean. NaN for empty / zero-mean input."""
    x = np.asarray(x, float)
    if len(x) == 0 or np.mean(x) == 0:
        return float("nan")
    return float(np.std(x) / np.mean(x))


def frame_ratio(slam_depth, slam_valid, metric_depth,
                max_depth=MODEL_MAX_DEPTH, min_valid=DEFAULT_MIN_VALID):
    """Robust per-frame metric/SLAM depth ratio on shared valid pixels (EXP-12 §2 step 3).

    Tier-1 hygiene is enforced here: drop metric pixels at/near the depth cap and non-finite /
    non-positive depths, then require >= ``min_valid`` shared pixels. Returns a diagnostics
    dict ``{ratio, rel_iqr, n, slam_med, metric_med_m}`` (``ratio`` = METRES per SLAM-depth
    unit) or ``None`` if too few valid pixels.
    """
    slam = np.asarray(slam_depth, float)
    valid = np.asarray(slam_valid, bool)
    met = np.asarray(metric_depth, float)
    good = (valid & np.isfinite(met) & (met > 1e-4) & (met < CAP_FRACTION * max_depth)
            & np.isfinite(slam) & (slam > 1e-6))
    if int(good.sum()) < int(min_valid):
        return None
    r = met[good] / slam[good]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    med = float(np.median(r))
    q1, q3 = np.percentile(r, [25, 75])
    return {"ratio": med, "rel_iqr": float((q3 - q1) / med) if med else float("nan"),
            "n": int(good.sum()), "slam_med": float(np.median(slam[good])),
            "metric_med_m": float(np.median(met[good]))}


def submap_ratio(frame_ratios):
    """Aggregate per-frame ratios into the submap's 1-DOF anchor (EXP-12 §2 steps 4).

    ``r_i`` = median over the K keyframes of the per-frame ratio (METRES per LOCAL unit);
    ``within_cov`` = CoV across the K frames (tier-2 reliability signal). Fewer than 2 valid
    frames -> ``within_cov`` is +inf (fail-safe: tier-2 will skip the anchor, never trust a
    single-frame ratio). Returns ``{r_i, within_cov, rel_iqr_med, n_frames, ok, per_frame}``.
    """
    per = [fr for fr in frame_ratios if fr is not None]
    n = len(per)
    if n == 0:
        return {"r_i": None, "within_cov": float("inf"), "rel_iqr_med": float("nan"),
                "n_frames": 0, "ok": False, "per_frame": []}
    ratios = [p["ratio"] for p in per]
    r_i = float(np.median(ratios))
    within = cov(ratios) if n >= 2 else float("inf")
    return {"r_i": r_i, "within_cov": within,
            "rel_iqr_med": float(np.median([p["rel_iqr"] for p in per])),
            "n_frames": n, "ok": np.isfinite(r_i) and r_i > 0, "per_frame": per}


def combine_scale(scale_geom, curr_ratio, prev_ratio, n_overlap,
                  gate=DEFAULT_GATE, within_cov_thresh=DEFAULT_WITHIN_COV,
                  thin_overlap=DEFAULT_THIN_OVERLAP):
    """4-tier sanity gate deciding the cross-submap scale (EXP-12 §6 "Failure modes + gate").

    ``curr_ratio`` / ``prev_ratio`` are ``submap_ratio(...)`` dicts for the current and prior
    submaps (or ``None``). ``scale_geom`` is ``estimate_scale_pairwise``'s geometric overlap
    scale; ``n_overlap`` is its supporting good-pixel count. Returns
    ``(s_final, decision)`` where ``decision`` records which tier fired. The anchor NEVER
    overrides good geometry blindly: on disagreement it defers to whichever of {geometry, anchor}
    the overlap richness implicates.

    Tiers:
      1. cap/conf hygiene + min_valid  -> already applied upstream (``frame_ratio``); a submap
         with no valid frame arrives here as ``ok=False`` -> anchor unavailable.
      2. within-submap consistency     -> either submap's K-frame CoV >= ``within_cov_thresh``
         (or <2 frames) -> unreliable anchor -> keep geometry.
      3. geometry-vs-anchor agreement  -> ``|ln(s_metric/scale_geom)| < gate`` -> apply. On
         disagreement, thin overlap -> geometry suspect, prefer anchor (+ degenerate-window
         warning); rich overlap -> anchor suspect, keep geometry (+ ``untrusted`` flag).
      4. absolute metres NOT claimed   -> we only ever return a *relative* cross-submap scale;
         the ratio form cancels the model's absolute bias. (No metres surfaced here.)
    """
    dec = {"applied": False, "reason": "", "s_metric": None, "scale_geom": float(scale_geom),
           "log_ratio": None, "untrusted": False, "degenerate": False, "n_overlap": int(n_overlap)}

    # Tier 1: anchor must exist for both submaps.
    if (curr_ratio is None or prev_ratio is None
            or not curr_ratio.get("ok") or not prev_ratio.get("ok")
            or not prev_ratio.get("r_i")):
        dec["reason"] = "anchor_unavailable"
        return float(scale_geom), dec

    s_metric = float(curr_ratio["r_i"]) / float(prev_ratio["r_i"])
    dec["s_metric"] = s_metric
    if not (np.isfinite(s_metric) and s_metric > 0):
        dec["reason"] = "anchor_nonfinite"
        return float(scale_geom), dec

    # Tier 2: within-submap reliability (both submaps).
    if (curr_ratio["within_cov"] >= within_cov_thresh
            or prev_ratio["within_cov"] >= within_cov_thresh):
        dec["reason"] = "within_submap_cov_high"
        return float(scale_geom), dec

    # Tier 3: geometry-vs-anchor agreement.
    if scale_geom <= 0 or not np.isfinite(scale_geom):
        # Degenerate geometry: prefer the anchor outright.
        dec.update(applied=True, reason="geom_degenerate", degenerate=True)
        return s_metric, dec
    log_ratio = float(abs(np.log(s_metric / scale_geom)))
    dec["log_ratio"] = log_ratio
    if log_ratio < gate:
        dec.update(applied=True, reason="agree")
        return s_metric, dec
    # Disagreement: arbitrate by overlap richness.
    if n_overlap < thin_overlap:
        dec.update(applied=True, reason="disagree_thin_overlap", degenerate=True)
        return s_metric, dec
    dec.update(applied=False, reason="disagree_rich_overlap", untrusted=True)
    return float(scale_geom), dec


def evenly_spaced(n, k):
    """K evenly-spaced indices in [0, n) (EXP-12 extract_keyframes.evenly_spaced)."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    step = (n - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


# ===========================================================================
# Live model path (lazy — torch / cv2 / huggingface_hub / Depth-Anything-V2)
# ===========================================================================
class MetricScaleAnchor:
    """Runs the metric-depth model on K keyframes per submap and produces its 1-DOF ratio r_i.

    The model + weights load lazily on first use (``metric_scale_anchor`` is opt-in, so a
    flags-OFF run never imports torch/cv2/HF here). For tests and Modal verification, pass
    ``infer_fn`` — a callable ``rgb_uint8 (H,W,3) -> depth_metres (H,W)`` — to bypass the real
    weights (mirrors how the D4RT phase-0 harness injects its Umeyama solver).
    """

    def __init__(self, model_name=DEFAULT_MODEL, keyframes=DEFAULT_KEYFRAMES,
                 gate=DEFAULT_GATE, min_valid=DEFAULT_MIN_VALID,
                 within_cov_thresh=DEFAULT_WITHIN_COV, thin_overlap=DEFAULT_THIN_OVERLAP,
                 device=None, infer_fn=None):
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(f"unknown metric_anchor_model {model_name!r}; "
                             f"choices: {list(_MODEL_REGISTRY)}")
        self.model_name = model_name
        self.spec = _MODEL_REGISTRY[model_name]
        self.max_depth = self.spec["max_depth"]
        self.input_size = self.spec["input_size"]
        self.keyframes = int(keyframes)
        self.gate = float(gate)
        self.min_valid = int(min_valid)
        self.within_cov_thresh = float(within_cov_thresh)
        self.thin_overlap = int(thin_overlap)
        self.device = device
        self._infer_fn = infer_fn      # injectable; None -> lazy-load the real model
        self._model = None

    # -- model loading / inference (lazy, GPU) ------------------------------
    def _load_model(self):
        if self._model is not None or self._infer_fn is not None:
            return
        import torch  # noqa: PLC0415 — lazy on purpose
        from huggingface_hub import hf_hub_download
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:  # pragma: no cover - environment wiring
            raise ImportError(
                "metric_scale_anchor needs the Depth-Anything-V2 metric_depth package on "
                "sys.path (git clone https://github.com/DepthAnything/Depth-Anything-V2 and add "
                "its metric_depth/ dir). See modal_app.py's metric-anchor image."
            ) from e
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = hf_hub_download(repo_id=self.spec["hf_repo"], filename=self.spec["ckpt"])
        cfg = {k: self.spec[k] for k in ("encoder", "features", "out_channels", "max_depth")}
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._model = model.to(self.device).eval()

    def _infer(self, rgb_uint8):
        """Metric depth (H,W metres) for one raw-RGB uint8 keyframe."""
        if self._infer_fn is not None:
            return np.asarray(self._infer_fn(rgb_uint8), dtype=np.float64)
        self._load_model()
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb_uint8), cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            depth_m = self._model.infer_image(bgr, self.input_size)  # infer_image does BGR2RGB
        return np.asarray(depth_m, dtype=np.float64)

    # -- per-submap ratio ---------------------------------------------------
    def compute_submap_ratio(self, submap):
        """Compute the submap's 1-DOF metric ratio r_i (+ diagnostics) — the ``submap_ratio``
        dict. LC submaps and empty submaps return ``ok=False`` (anchor unavailable). Uses the
        submap's own stored data: raw RGB ``colors[fi]``, camera-z depth ``pointclouds[fi][...,2]``,
        conf mask ``conf_masks[fi] > conf_threshold``.
        """
        if getattr(submap, "is_lc_submap", False):
            out = submap_ratio([])
            out["reason"] = "lc_submap"
            return out
        n_real = submap.get_last_non_loop_frame_index()
        n_real = (n_real + 1) if n_real is not None else len(submap.pointclouds)
        n_real = min(n_real, len(submap.pointclouds))
        idxs = evenly_spaced(n_real, self.keyframes)
        thr = submap.get_conf_threshold()
        frame_ratios = []
        for fi in idxs:
            rgb = self._submap_rgb(submap, fi)
            slam_depth = np.asarray(submap.pointclouds[fi], float)[..., 2]  # camera z-depth
            conf = submap.get_conf_masks_frame(fi)
            slam_valid = (conf > thr) & np.isfinite(slam_depth) & (slam_depth > 1e-6)
            metric_depth = self._infer(rgb)
            frame_ratios.append(frame_ratio(slam_depth, slam_valid, metric_depth,
                                            max_depth=self.max_depth, min_valid=self.min_valid))
        out = submap_ratio(frame_ratios)
        out["keyframes"] = list(idxs)
        return out

    @staticmethod
    def _submap_rgb(submap, fi):
        """Raw uint8 RGB (H,W,3) for keyframe fi — prefer the submap's stored ``colors`` (already
        ``(frames*255).astype(uint8)``); fall back to converting the frame tensor."""
        colors = getattr(submap, "colors", None)
        if colors is not None:
            return np.ascontiguousarray(colors[fi]).astype(np.uint8)
        frame = submap.get_frame_at_index(fi)  # torch (3,H,W) in [0,1]
        arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
        return (np.transpose(arr, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
