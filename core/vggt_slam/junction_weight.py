"""Junction-confidence weighting for the SL(4) pose graph — DEFAULT OFF, opt-in via
``--weighted_junctions``.

Each cross-submap junction (the between-factor `add_edge` adds with ``intra_submap_noise``) is
currently trusted equally regardless of how well-supported that junction actually is. This module
scales each junction factor's **covariance** by measured junction quality, so a weak junction gets
a *looser* factor and the graph — and any metric-anchor scale correction (Commit 1) — can override
it, while a strong junction stays tight.

Signals (all monotone; whichever are available are combined by geometric mean):
  * **scale dispersion** — spread (robust rel-IQR) of the per-point ‖Y‖/‖X‖ ratios inside
    ``estimate_scale_pairwise``. High spread = the overlap scale is ill-determined.
  * **overlap coverage** — number of conf-masked shared pixels backing the junction
    (``np.sum(good_mask)`` in ``add_edge``). Thin overlap = weak junction.
  * **VGGT image_match_ratio** — the loop-closure verification ratio (``run_predictions``),
    available at LC junctions only; optional.

quality q ∈ [0,1] (1 = most confident) → covariance multiplier ``MAX_COV_MULT ** (1 - q)`` ∈
``[1, MAX_COV_MULT]`` (1 = unchanged/tight, MAX = loosest). gtsam ``noiseModel.Diagonal.Sigmas``
takes standard deviations, so the *sigma* multiplier is ``sqrt(covariance multiplier)`` — see
``sigma_multiplier`` and ``PoseGraph.scaled_noise``.

This idea is the license-free, classical transmutation of the TTT3R cross-attention
confidence-gate (wave-2 R-1): weight the factor by measured agreement rather than trusting every
junction equally. Pure numpy, import-clean (unit-tested). All tunables live in ONE place below.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tunables — the single place to adjust the signal -> covariance mapping.
# ---------------------------------------------------------------------------
DISPERSION_SCALE = 0.30   # scale-ratio rel-IQR at which q_dispersion decays to 1/e (~0.37)
COVERAGE_REF = 2000       # shared conf-masked pixels for full coverage credit (q_coverage -> 1)
MATCH_MIN = 0.80          # image_match_ratio floor (q_match -> 0 at/below this)
MATCH_HI = 0.95           # image_match_ratio at which q_match saturates to 1
MAX_COV_MULT = 100.0      # loosest covariance multiplier (=> sigma up to sqrt(100)=10x looser)


def _q_dispersion(d):
    """Quality from the scale-ratio dispersion d (>=0): 1 at d=0, ->0 as d grows. Monotone dec."""
    d = float(max(d, 0.0))
    return float(np.exp(-d / DISPERSION_SCALE))


def _q_coverage(n):
    """Quality from the overlap pixel count n (>=0): linear-saturating to 1 at COVERAGE_REF."""
    return float(np.clip(float(n) / COVERAGE_REF, 0.0, 1.0))


def _q_match(r):
    """Quality from the VGGT image_match_ratio r: ramps 0->1 across [MATCH_MIN, MATCH_HI]."""
    return float(np.clip((float(r) - MATCH_MIN) / (MATCH_HI - MATCH_MIN), 0.0, 1.0))


def junction_quality(dispersion=None, n_overlap=None, image_match_ratio=None):
    """Combine the available junction signals into a single quality q ∈ [0,1] (1 = confident).

    Geometric mean of whichever signals are provided (monotone increasing in each). With no
    signal at all, returns 1.0 — absence of evidence never loosens a junction.
    """
    qs = []
    if dispersion is not None and np.isfinite(dispersion):
        qs.append(_q_dispersion(dispersion))
    if n_overlap is not None:
        qs.append(_q_coverage(n_overlap))
    if image_match_ratio is not None and np.isfinite(image_match_ratio):
        qs.append(_q_match(image_match_ratio))
    if not qs:
        return 1.0
    # geometric mean (weakest-link-aware but smooth); clamp away from 0 to avoid log(0).
    qs = np.clip(np.asarray(qs, float), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(qs))))


def junction_covariance_multiplier(dispersion=None, n_overlap=None, image_match_ratio=None):
    """Covariance multiplier for a junction factor: 1 (tight) .. MAX_COV_MULT (loosest).

    Monotone: more dispersion / thinner overlap / lower match ratio -> lower quality -> larger
    multiplier -> looser factor. Returns exactly 1.0 for a maximally-confident junction, so this
    is a no-op on strong junctions.
    """
    q = junction_quality(dispersion=dispersion, n_overlap=n_overlap,
                          image_match_ratio=image_match_ratio)
    return float(np.clip(MAX_COV_MULT ** (1.0 - q), 1.0, MAX_COV_MULT))


def sigma_multiplier(dispersion=None, n_overlap=None, image_match_ratio=None):
    """sqrt of the covariance multiplier — the factor to apply to noiseModel std-devs (Sigmas)."""
    return float(np.sqrt(junction_covariance_multiplier(
        dispersion=dispersion, n_overlap=n_overlap, image_match_ratio=image_match_ratio)))
