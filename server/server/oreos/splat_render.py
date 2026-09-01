"""Server-side gaussian rasterizer — renders of a persisted splat, without a browser.

Why this module exists
----------------------
``synthetic_views.py`` registers renders of a splat as first-class evidence, and until
now the only producer was a human: open the scene in the viewer, click "Capture views".
That blocks the two things the founder actually hit:

* an imported Splatica scene cannot be processed **headlessly** — there is no way to give
  it annotation evidence without a person at a browser;
* SAM 3D completion refuses any object whose evidence keyframe was never persisted
  (``not_completable: this object's evidence keyframe is not persisted for this scene``).
  The refusal names the fix — a rendered-view path — which did not exist.

So this is the same picture the viewer makes, made on the server. It is NOT a more
authoritative picture: a server render and a browser render are both renders, they carry
the same ``synthetic view`` provenance, and neither is a photograph.

Why pure PyTorch and not gsplat
-------------------------------
``gsplat`` needs a CUDA compile. Adding that to the streaming image days before a
recording risks the image build itself, and the failure mode (a build that breaks the
deploy) is worse than the thing it buys. ``torch==2.3.1`` is already in the image, and a
non-differentiable forward rasterizer is a couple of hundred lines: project each gaussian
to a 2D mean + 2x2 covariance, bin into tiles, sort by depth, alpha-composite front to
back. The bar is "an image a vision model can read", not "matches the reference
renderer" — the deviations are named in :data:`RENDER_CAVEATS` rather than hidden.

The compositing is EXACT rather than depth-capped: tiles are processed in blocks of
:data:`LAYER_BLOCK` gaussians with the transmittance carried between blocks, so a tile
with 4000 overlapping gaussians composites all 4000 (it just stops early once the tile is
opaque). A fixed per-tile layer cap was the obvious alternative and it silently loses the
back half of a dense scene, which reads as haze rather than as an error.

Frames — the whole risk (WORLD-TRANSFORM-CONTRACT.md)
-----------------------------------------------------
:func:`render` takes an **OpenCV camera-to-world** (x right, y DOWN, z forward), the
convention ``trajectory.npz`` uses and the one ``synthetic_views.gl_pose_to_cv_c2w``
converts the three.js wire pose into. Its pixel grid is the one
``synthetic_views.project_world_point`` predicts: a world point at pixel ``(u, v)`` lands
in column ``floor(u)``, row ``floor(v)``, and an isolated gaussian's alpha-weighted
centroid comes back at ``(u, v)`` to well under a pixel. That is asserted directly in
``tests/test_demo_splat_render.py`` and again in-container against the founder's real
scene (``modal run modal_oreos_render.py::roundtrip``) — a render at the wrong pose throws
nothing and looks entirely plausible, so it has to be measured, not reasoned about.

Pose planning is a port of the client's ``planRingCapture``
(``web/apps/webserver/src/demo/syntheticViews.ts``) — same defaults, same geometry, so a
headless ring and an operator-captured ring are the same set of viewpoints.

Numpy for I/O and selection, torch only inside :func:`render` (imported lazily, so the
module stays importable on the broker and in a torch-free test env).
"""

from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from server.oreos import lod as lod_mod

# ---------------------------------------------------------------------------
# Knobs. Every default here was measured on the founder's 8.49M-gaussian import
# (`modal run modal_oreos_render.py::ring`); the headline numbers are recorded in
# the CLAIM-LEDGER row for server-side synthetic views.
# ---------------------------------------------------------------------------

#: Gaussians the rasterizer will hold at once. 2M is the LOD level the viewer loads by
#: default, so a server render and the operator's screen are the same density.
DEFAULT_BUDGET = int(os.environ.get("DEMO_RENDER_BUDGET", "2000000"))

#: Cheaper level for the per-object renders on the SAM 3D path, where a request is
#: waiting on the result and the framing matters more than the last 3% of detail.
FAST_BUDGET = int(os.environ.get("DEMO_RENDER_FAST_BUDGET", "600000"))

#: Screen-space tile edge. Work is ~``(2r + TILE)^2`` per gaussian, so a smaller tile
#: culls harder; below 8 the (tile, gaussian) pair array grows faster than the saving.
TILE = 8

#: Gaussians composited per block. Bounds the (tiles, layers, pixels) temporary; the
#: transmittance carries across blocks so this is a memory knob, not a quality one.
LAYER_BLOCK = 64

#: Tiles per chunk. Tiles are processed in descending-occupancy order so a chunk is
#: roughly homogeneous and the padding to the chunk's deepest tile stays cheap. Bigger
#: chunks trade memory for fewer kernel launches, which is the dominant cost on a GPU.
TILES_PER_CHUNK = int(os.environ.get("DEMO_RENDER_TILES_PER_CHUNK", "256"))

#: Screen-space covariance dilation, in px^2. The standard 3DGS low-pass filter: without
#: it a sub-pixel gaussian aliases into nothing.
BLUR_2D = 0.3

#: A gaussian is dropped once it would touch more tiles than this. These are the handful
#: of enormous background blobs a decimated splat carries; each one costs more pair
#: entries than the rest of the frame combined. The count is reported, never swallowed.
MAX_TILES_PER_GAUSSIAN = 4096

#: Hard ceiling on (tile, gaussian) pairs, so a pathological view cannot OOM the box.
MAX_PAIRS = int(os.environ.get("DEMO_RENDER_MAX_PAIRS", str(48_000_000)))

#: Transmittance below which a tile is opaque enough to stop compositing.
MIN_TRANSMITTANCE = 1.0 / 255.0

#: Near plane, in scene units. Anything closer is culled rather than projected to a
#: near-infinite conic.
NEAR = 0.01

#: What a render of ours is and is not. Rides on the view metadata so a reader holding
#: only the record knows which renderer made the picture and how it differs.
RENDER_GENERATOR = "openreality-splat-render/1"
RENDER_CAVEATS = (
    "Rendered on the server by our own rasterizer, not by the browser viewer — "
    "sub-pixel filtering and the antialiasing convention differ slightly.",
    "Gaussians are composited front-to-back with a 0.3 px^2 screen-space dilation; "
    "very large background gaussians are dropped rather than tiled.",
)

# Spherical-harmonic basis constants (the standard 3DGS `computeColorFromSH`).
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

#: rest-coefficient count -> SH degree. Anything else is refused rather than guessed at:
#: a mis-read coefficient layout produces colour that is wrong but plausible.
_REST_TO_DEGREE = {0: 0, 3: 1, 8: 2, 15: 3}


class SplatRenderError(RuntimeError):
    """A render that cannot happen, with the reason spelled out.

    Everything downstream degrades on this rather than substituting a picture: a blank
    or invented frame registered as evidence is worse than no frame, because the
    annotator cannot tell the difference."""


# ---------------------------------------------------------------------------
# The cloud
# ---------------------------------------------------------------------------


@dataclass
class GaussianCloud:
    """One splat, in the exporter's own units (log scales, logit opacity, wxyz rot).

    ``sh`` is ``(N, (deg+1)^2, 3)`` with ``sh[:, 0]`` the DC term, i.e. the PLY's
    channel-major ``f_rest_*`` already transposed into coefficient-major order. Held as
    float32 numpy so selection and I/O never need torch."""

    means: np.ndarray
    scales_log: np.ndarray
    quats_wxyz: np.ndarray
    opacity_logit: np.ndarray
    sh: np.ndarray
    sh_degree: int
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.means.min(axis=0), self.means.max(axis=0)


def _degree_from_rest(n_rest_per_channel: int) -> int:
    if n_rest_per_channel not in _REST_TO_DEGREE:
        raise SplatRenderError(
            f"{n_rest_per_channel} SH rest coefficients per channel is not a whole "
            "degree (expected 0, 3, 8 or 15) — refusing to guess the layout"
        )
    return _REST_TO_DEGREE[n_rest_per_channel]


def _assemble_sh(f_dc: np.ndarray, f_rest: Optional[np.ndarray]) -> tuple[np.ndarray, int]:
    """``(N,3)`` DC + optional channel-major ``(N, 3*M)`` rest -> ``(N, K, 3)`` + degree.

    The PLY stores ``f_rest_{c*M + k}`` (all of red's coefficients, then green's, then
    blue's); the shading loop wants coefficient-major. Doing the transpose once, here,
    is why :func:`_sh_colour` can stay a straight transcription of the reference
    implementation instead of carrying an index-arithmetic hazard."""
    n = f_dc.shape[0]
    if f_rest is None or f_rest.size == 0:
        sh = np.empty((n, 1, 3), dtype=np.float32)
        sh[:, 0, :] = f_dc
        return sh, 0
    if f_rest.shape[1] % 3:
        raise SplatRenderError(
            f"{f_rest.shape[1]} f_rest properties is not divisible by 3 channels"
        )
    m = f_rest.shape[1] // 3
    degree = _degree_from_rest(m)
    sh = np.empty((n, m + 1, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc
    # (N, 3, M) channel-major -> (N, M, 3) coefficient-major
    sh[:, 1:, :] = f_rest.reshape(n, 3, m).transpose(0, 2, 1)
    return sh, degree


def load_ply_cloud(
    path: str, *, want_sh: bool = True, log: Optional[Callable[[str], None]] = None
) -> GaussianCloud:
    """Read a binary-LE 3DGS PLY into a :class:`GaussianCloud`.

    Column-at-a-time over ``lod.SplatSource``'s memmap, so a 14-property LOD artifact
    costs its own size and a 62-property source costs only the columns asked for."""
    emit = log or (lambda _m: None)
    src = lod_mod.SplatSource(path)
    try:
        n = src.count
        means = np.stack([src.column(a) for a in ("x", "y", "z")], axis=1)
        scales = np.stack([src.column(f"scale_{i}") for i in range(3)], axis=1)
        quats = np.stack([src.column(f"rot_{i}") for i in range(4)], axis=1)
        opacity = src.column("opacity")
        f_dc = np.stack([src.column(f"f_dc_{i}") for i in range(3)], axis=1)

        rest_names = sorted(
            (p for p in src.properties if p.startswith("f_rest_")),
            key=lambda p: int(p.rsplit("_", 1)[1]),
        )
        f_rest = None
        if want_sh and rest_names:
            f_rest = np.stack([src.column(p) for p in rest_names], axis=1)
        sh, degree = _assemble_sh(f_dc, f_rest)
        emit(
            f"    loaded {n:,} gaussians from {os.path.basename(path)} "
            f"(SH degree {degree}{'' if want_sh else ', SH not requested'})"
        )
        return GaussianCloud(
            means=means,
            scales_log=scales,
            quats_wxyz=quats,
            opacity_logit=opacity,
            sh=sh,
            sh_degree=degree,
            source={
                "path": path,
                "format": "ply",
                "gaussians": n,
                "sh_degree": degree,
                "sh_available": bool(rest_names),
            },
        )
    finally:
        src.close()


def load_spz_cloud(
    path: str, *, want_sh: bool = True, log: Optional[Callable[[str], None]] = None
) -> GaussianCloud:
    """Read a ``.spz`` via the decoder we already ship (``server/oreos/spz.py``).

    This is the only route to view-dependent colour on the founder's scene: the LOD PLY
    levels carry 14 properties (``lod.LOD_PROPERTIES``) and therefore no ``f_rest_*``,
    while ``derived/demo/lod/full.spz`` is a transcode of the ORIGINAL splat and keeps
    all 45 of them. Measured on that file: 161 MB on disk, 8,493,859 gaussians, SH
    degree 3, 8.9 s to decode."""
    emit = log or (lambda _m: None)
    from server.oreos import spz as spz_mod

    t0 = time.time()
    fields = spz_mod.decode_spz(path)
    n = int(fields["x"].shape[0])
    means = np.stack([fields[a] for a in ("x", "y", "z")], axis=1)
    scales = np.stack([fields[f"scale_{i}"] for i in range(3)], axis=1)
    quats = np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1)
    opacity = fields["opacity"]
    f_dc = np.stack([fields[f"f_dc_{i}"] for i in range(3)], axis=1)
    rest_names = sorted(
        (k for k in fields if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1])
    )
    f_rest = np.stack([fields[k] for k in rest_names], axis=1) if (want_sh and rest_names) else None
    sh, degree = _assemble_sh(f_dc, f_rest)
    del fields
    emit(f"    decoded {n:,} gaussians from {os.path.basename(path)} "
         f"(SH degree {degree}) in {time.time() - t0:.1f}s")
    return GaussianCloud(
        means=means,
        scales_log=scales,
        quats_wxyz=quats,
        opacity_logit=opacity,
        sh=sh,
        sh_degree=degree,
        source={
            "path": path,
            "format": "spz",
            "gaussians": n,
            "sh_degree": degree,
            "sh_available": bool(rest_names),
            "decode_seconds": round(time.time() - t0, 2),
        },
    )


def load_cloud(
    path: str, *, want_sh: bool = True, log: Optional[Callable[[str], None]] = None
) -> GaussianCloud:
    """``.ply`` / ``.spz`` -> :class:`GaussianCloud`, dispatched on the suffix."""
    lowered = path.lower()
    if lowered.endswith(".spz"):
        return load_spz_cloud(path, want_sh=want_sh, log=log)
    if lowered.endswith(".ply"):
        return load_ply_cloud(path, want_sh=want_sh, log=log)
    raise SplatRenderError(f"unsupported splat artifact: {os.path.basename(path)}")


def plan_source(
    lod_index: Optional[dict[str, Any]], *, budget: int = DEFAULT_BUDGET, want_sh: bool = True
) -> list[dict[str, Any]]:
    """Which artifacts to try, best first, given a scene's ``demo/lod/index.json``.

    The renderer never opens the source ``splat.ply``: on the founder's scene that is
    2.0 GB and 62 properties, and there is no version of "render a ring" that should pay
    that. Everything here is an LOD artifact.

    The order encodes one real trade. LOD *PLY* levels carry ``lod.LOD_PROPERTIES`` — 14
    properties, no ``f_rest_*`` — so they are SH degree 0 whatever the source was, and a
    render from one is view-INDEPENDENT (it is also exactly what the viewer shows at its
    default level, which is why it is a perfectly honest fallback and not a failure). The
    compressed full-detail ``.spz``, by contrast, is a transcode of the ORIGINAL splat and
    keeps all 45 rest coefficients. So when view-dependent colour is wanted, the full spz
    is tried first and decimated here; the caller confirms the degree from its 16-byte
    header before paying for the decode.

    Returns dicts of ``{key, kind, sh, needs_decimation, reason}``; an empty list means
    the scene has no LOD artifacts at all, which is a refusal, not a fallback.
    """
    if not isinstance(lod_index, dict):
        return []
    out: list[dict[str, Any]] = []

    full = lod_index.get("full_detail")
    if want_sh and isinstance(full, dict) and isinstance(full.get("key"), str):
        if str(full["key"]).lower().endswith(".spz"):
            out.append(
                {
                    "key": f"derived/{full['key']}",
                    "kind": "spz",
                    "sh": "maybe",
                    "needs_decimation": int(full.get("gaussians") or 0) > int(budget),
                    "reason": "full-detail spz keeps the source's SH coefficients",
                }
            )

    levels = [e for e in (lod_index.get("levels") or []) if isinstance(e, dict) and e.get("key")]
    levels.sort(key=lambda e: int(e.get("gaussians") or e.get("budget") or 0))
    # Smallest level that still meets the budget, else the largest available.
    chosen = next((e for e in levels if int(e.get("gaussians") or 0) >= int(budget)), None)
    ordered = ([chosen] if chosen else []) + [e for e in reversed(levels) if e is not chosen]
    for entry in ordered:
        out.append(
            {
                "key": f"derived/{entry['key']}",
                "kind": "ply",
                "sh": "degree 0 (LOD levels drop f_rest)",
                "needs_decimation": int(entry.get("gaussians") or 0) > int(budget),
                "reason": f"LOD level {entry.get('name')} ({entry.get('gaussians')} gaussians)",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Decimation — the LOD selection, run over in-memory columns
# ---------------------------------------------------------------------------


class _ArraySource:
    """``lod.SplatSource``'s read interface over arrays already in memory.

    ``lod.select_indices`` is the measured, A/B'd selection (voxel-stratified, one
    gaussian per occupied cell, highest screen contribution wins — its docstring records
    why global top-K is wrong on these scenes). It only ever calls ``.count`` and
    ``.column``, so an eight-line adapter reuses it verbatim rather than re-deriving a
    second decimation rule that would drift from the one the client's LOD used."""

    def __init__(self, cloud: GaussianCloud) -> None:
        self.count = cloud.count
        self._cols = {
            "x": cloud.means[:, 0],
            "y": cloud.means[:, 1],
            "z": cloud.means[:, 2],
            "opacity": cloud.opacity_logit,
            "scale_0": cloud.scales_log[:, 0],
            "scale_1": cloud.scales_log[:, 1],
            "scale_2": cloud.scales_log[:, 2],
        }
        self.properties = list(self._cols)

    def column(self, name: str) -> np.ndarray:
        return self._cols[name]


def decimate(
    cloud: GaussianCloud,
    target: int,
    *,
    method: str = "auto",
    rng_seed: int = 0,
    scale_floor_frac: float = lod_mod.SCALE_FLOOR_FRAC,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[GaussianCloud, dict[str, Any]]:
    """Thin ``cloud`` to ``target`` gaussians, keeping coverage (and SH) intact.

    Applies the same voxel-tied scale floor ``lod.write_lod_ply`` applies, for the same
    reason: thinning a surface makes the survivors too small to cover it and you see
    through the walls. That is a rendering approximation and it is recorded as one."""
    if target >= cloud.count:
        return cloud, {"method": "identity", "selected": cloud.count}
    idx, info = lod_mod.select_indices(
        _ArraySource(cloud), int(target), method=method, rng_seed=rng_seed, log=log
    )
    scales = cloud.scales_log[idx].copy()
    raised = 0
    voxel = info.get("voxel_size")
    if voxel and scale_floor_frac > 0:
        floor_log = float(np.log(float(voxel) * float(scale_floor_frac)))
        raised = int(np.count_nonzero(scales < floor_log))
        np.maximum(scales, floor_log, out=scales)
    info = {
        **info,
        "scale_floor_applied": raised > 0,
        "scale_axes_raised": raised,
        "source_gaussians": cloud.count,
    }
    return (
        GaussianCloud(
            means=cloud.means[idx],
            scales_log=scales,
            quats_wxyz=cloud.quats_wxyz[idx],
            opacity_logit=cloud.opacity_logit[idx],
            sh=cloud.sh[idx],
            sh_degree=cloud.sh_degree,
            source={**cloud.source, "decimated_to": int(idx.size)},
        ),
        info,
    )


def subset_in_obb(
    means: Any,
    center: Sequence[float],
    extents: Sequence[float],
    rotation: Optional[Sequence[Sequence[float]]] = None,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Indices of the points inside an OBB (rotation COLUMNS are the axes, ``extents``
    are FULL lengths — the WORLD-TRANSFORM-CONTRACT convention, and reading either the
    other way selects a plausible-looking wrong subset).

    Takes bare means rather than a cloud so the caller can pass a device tensor's
    positions without assembling a whole :class:`GaussianCloud` around them."""
    pts = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(extents, dtype=np.float64).reshape(3)
    half = (e / 2.0) * (1.0 + float(margin))
    rel = pts - c
    if rotation is not None:
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        rel = rel @ r  # columns are the axes, so this projects onto them
    return np.nonzero(np.all(np.abs(rel) <= np.maximum(half, 1e-9), axis=1))[0]


def take(cloud: GaussianCloud, idx: np.ndarray) -> GaussianCloud:
    """A cloud holding only ``idx`` (used to render one object's own gaussians)."""
    return GaussianCloud(
        means=cloud.means[idx],
        scales_log=cloud.scales_log[idx],
        quats_wxyz=cloud.quats_wxyz[idx],
        opacity_logit=cloud.opacity_logit[idx],
        sh=cloud.sh[idx],
        sh_degree=cloud.sh_degree,
        source={**cloud.source, "subset": int(np.asarray(idx).size)},
    )


# ---------------------------------------------------------------------------
# Pose planning — the port of planRingCapture (syntheticViews.ts)
# ---------------------------------------------------------------------------


@dataclass
class ViewSpec:
    """One planned camera, in the WORLD frame with a three.js-convention quaternion —
    exactly the shape the synthetic-views wire format takes."""

    position: list[float]
    quaternion: list[float]
    fov_y_deg: float
    width: int
    height: int
    label: Optional[str] = None
    #: What this camera is aimed at, in world coordinates. Carried so a camera can be
    #: MOVED (see :func:`clear_cameras`) and re-aimed at the same thing, rather than
    #: keeping an orientation that was only correct at its original position.
    aim: Optional[list[float]] = None
    #: Set by :func:`clear_cameras` when the camera had to be pulled out of geometry.
    placement: Optional[dict[str, Any]] = None

    def as_pose(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "quaternion": list(self.quaternion),
            "fov_y_deg": float(self.fov_y_deg),
            "width": int(self.width),
            "height": int(self.height),
            "label": self.label,
        }


#: planRingCapture's DEFAULT_PLAN, verbatim. Divergence here would mean an operator's
#: ring and a headless ring were different sets of viewpoints, which is exactly the kind
#: of difference nobody notices until the two disagree about the room.
RING_DEFAULTS = {
    "ring_count": 8,
    "elevated_count": 2,
    "eye_height_fraction": 0.62,
    "radius_fraction": 0.55,
    "fov_y_deg": 70.0,
    "width": 1024,
    "height": 768,
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def matrix_to_quaternion(m: np.ndarray) -> list[float]:
    """``(3,3)`` rotation -> ``[x, y, z, w]`` (three.js order). Shepperd's method, the
    branch-per-largest-diagonal form, because the trace branch alone loses precision on
    the 180-degree-ish rotations a downward-looking elevated camera produces."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2.0
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    n = float(np.linalg.norm(q))
    return [float(v / n) for v in q]


def look_at_quaternion(
    position: Sequence[float], target: Sequence[float], up: Sequence[float]
) -> list[float]:
    """World-frame look-at -> a three.js camera quaternion (looks down −Z, +Y up).

    Port of ``lookAtPose``, including its degenerate branch: a camera looking straight
    along ``up`` has no preferred roll, so one is chosen rather than an all-zero basis
    being emitted."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    tgt = np.asarray(target, dtype=np.float64).reshape(3)
    upv = _normalize(np.asarray(up, dtype=np.float64).reshape(3))

    forward = tgt - pos
    if float(forward @ forward) < 1e-12:
        forward = -upv.copy()
    forward = _normalize(forward)
    right = np.cross(forward, upv)
    if float(right @ right) < 1e-9:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if float(right @ right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
    right = _normalize(right)
    true_up = _normalize(np.cross(right, forward))
    basis = np.stack([right, true_up, -forward], axis=1)  # three.js camera basis
    return matrix_to_quaternion(basis)


def robust_bounds(means: Any, *, percentile: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-clipped extent of a point set.

    A raw SLAM/import AABB is outlier-stretched — ``lod.choose_voxel_size`` says the same
    thing and clips the same way, because a handful of stray gaussians can double the
    box. On the founder's scene the stored ``bbox`` spans 15.6 units vertically while the
    fitted floor-to-ceiling is 3.4, so a ring planned on the raw box puts every camera
    about 9 units above the room looking at empty space. Measured, not theorised: the
    first headless ring did exactly that."""
    pts = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    lo = np.percentile(pts, percentile, axis=0)
    hi = np.percentile(pts, 100.0 - percentile, axis=0)
    return lo, hi


def plan_ring(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    *,
    up: Optional[Sequence[float]] = None,
    floor_height: Optional[float] = None,
    ceiling_height: Optional[float] = None,
    elliptical: bool = False,
    **overrides: Any,
) -> list[ViewSpec]:
    """A capture plan that covers the room: a ring at eye height looking inward, plus
    elevated three-quarter views looking down at the centroid.

    All of it is done in the world frame around ``up`` (the fitted ground frame's
    vertical when the scene has one). An imported splat arrives in whatever frame its
    exporter chose, and a ring planned around the wrong axis produces eight views of the
    ceiling — which is why the axis is an argument and not an assumption.

    ``floor_height`` / ``ceiling_height`` (both in the ground frame's sense: a height is
    ``point . up``) replace the vertical span when the scene has a fitted ground frame,
    and ``elliptical`` scales the ring per in-plane axis. Without either the geometry is
    byte-for-byte the client's ``planRingCapture``; with them the cameras stand in the
    room the plane fit actually found rather than in the bounding box's idea of one.

    NOTE the fitted ``floor_extent`` on ``facts.metrics`` is deliberately NOT an input:
    it is measured in the floor plane's own ``(u, v)`` basis, which the record does not
    carry, so pairing its two numbers with this function's axes would be a coin flip
    between the right ellipse and one rotated 90 degrees. The in-plane extent used here
    is measured from the bounds the caller passes in."""
    o = {**RING_DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    upv = _normalize(np.asarray(up if up is not None else [0.0, 1.0, 0.0], dtype=np.float64).reshape(3))
    if not np.isfinite(upv).all() or float(upv @ upv) < 0.5:
        upv = np.array([0.0, 1.0, 0.0])

    lo = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    size = hi - lo
    centre = 0.5 * (lo + hi)
    if not np.isfinite(size).all() or float(np.linalg.norm(size)) < 1e-6:
        return []

    seed = np.array([0.0, 1.0, 0.0]) if abs(upv[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = _normalize(seed - upv * float(seed @ upv))
    v_axis = _normalize(np.cross(upv, u_axis))

    # Extent of the BOX along up, and the box's lowest point along up, both measured over
    # the corners rather than over the componentwise min. `(lo - centre) . up` — the
    # client's form, and this port's first one — is only the floor when `up` is +Y-ish:
    # the founder's fitted up is very nearly −Y, and there `lo` projects onto the TOP of
    # the box, so the eye height came out at −17.5 in a scene whose geometry stops at
    # −7.8. Every camera was below the floor and the whole first ring was unusable.
    # (The client's planRingCapture has the same latent bug; it has only ever been
    # exercised on +Y scenes.)
    vertical_extent = float(np.abs(size) @ np.abs(upv))
    floor_level = -0.5 * vertical_extent
    half_diagonal = 0.5 * math.hypot(abs(float(size @ u_axis)), abs(float(size @ v_axis)))
    if floor_height is not None and ceiling_height is not None:
        span = float(ceiling_height) - float(floor_height)
        if span > 1e-6:
            # `floor_level` is the floor's offset from the centroid ALONG up, which is
            # exactly what the fitted floor height is once the centroid's own height is
            # subtracted — so the two representations compose without a second frame.
            floor_level = float(floor_height) - float(centre @ upv)
            vertical_extent = span
    radius = max(half_diagonal * float(o["radius_fraction"]), 1e-3)
    eye_offset = floor_level + vertical_extent * float(o["eye_height_fraction"])
    # Aim below eye level: the interesting half of a room is the floor and what stands
    # on it, not the ceiling.
    target = centre + upv * (floor_level + vertical_extent * 0.4)

    # An ELLIPSE rather than a circle, when the caller asks for one. The client's circle
    # takes 0.55 of the half-DIAGONAL, which is fine against a bounding box that
    # overstates the room and wrong against a tight one: on the founder's ~8.6 x 11.9
    # floor a 4.05-unit circle puts the short-axis cameras hard against the wall, and
    # four of the first eight rendered the inside of it. Scaling each in-plane axis
    # independently keeps every camera the same fraction of the way to ITS OWN wall.
    #
    # Off by default so ``plan_ring(bounds, up=...)`` stays byte-for-byte the client's
    # planRingCapture; the server path opts in, and says so in the ring report.
    if elliptical:
        radius_u = max(0.5 * abs(float(size @ u_axis)) * float(o["radius_fraction"]), 1e-3)
        radius_v = max(0.5 * abs(float(size @ v_axis)) * float(o["radius_fraction"]), 1e-3)
    else:
        radius_u = radius_v = radius

    specs: list[ViewSpec] = []

    def push(position: np.ndarray, label: str) -> None:
        specs.append(
            ViewSpec(
                position=[float(v) for v in position],
                quaternion=look_at_quaternion(position, target, upv),
                fov_y_deg=float(o["fov_y_deg"]),
                width=int(o["width"]),
                height=int(o["height"]),
                label=label,
                aim=[float(v) for v in target],
            )
        )

    ring_count = int(o["ring_count"])
    for i in range(ring_count):
        angle = 2.0 * math.pi * i / max(ring_count, 1)
        position = (
            centre
            + u_axis * (radius_u * math.cos(angle))
            + v_axis * (radius_v * math.sin(angle))
            + upv * eye_offset
        )
        push(position, f"ring {i + 1}/{ring_count}")

    # An elevated camera at 1.05x the vertical extent is ABOVE a room that has a real
    # ceiling, so it looks down through the ceiling slab and the frame fills with
    # out-of-focus soffit — measured on the founder's scene, both elevated views were
    # unusable. When a ceiling is known the camera stops just under it.
    elevated_fraction = 1.05 if ceiling_height is None else 0.9
    elevated_count = int(o["elevated_count"])
    for i in range(elevated_count):
        # Offset from the ring angles so an elevated view is never a taller duplicate.
        angle = 2.0 * math.pi * (i + 0.5) / max(elevated_count, 1)
        position = (
            centre
            + u_axis * (radius_u * 0.75 * math.cos(angle))
            + v_axis * (radius_v * 0.75 * math.sin(angle))
            + upv * (floor_level + vertical_extent * elevated_fraction)
        )
        push(position, f"elevated {i + 1}/{elevated_count}")
    return specs


def clear_cameras(
    specs: list[ViewSpec],
    means: Any,
    *,
    clearance: float,
    allowed: int = 0,
    steps: int = 8,
    log: Optional[Callable[[str], None]] = None,
) -> list[ViewSpec]:
    """Pull any camera that is standing inside geometry back toward what it is aiming at.

    A ring is planned from a footprint, and a footprint does not know about the pillar,
    the display case or the partition wall that happens to be where camera 5 wants to
    stand. The render from inside one is not an error — it is a correct picture of the
    inside of a wall — so nothing downstream can tell it apart from a real view of a
    cluttered corner. Detecting it is cheap: count the gaussians within ``clearance`` of
    the eye point.

    Cameras are slid along the line to their own aim point (never past 85% of it, or the
    camera ends up inside the furniture it was framing) and re-aimed at the same target.
    A camera that cannot be freed keeps its best position and says so in ``placement``,
    because a view of a wall that is LABELLED a view of a wall is still information."""
    emit = log or (lambda _m: None)
    pts = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    out: list[ViewSpec] = []
    for spec in specs:
        origin = np.asarray(spec.position, dtype=np.float64)
        target = np.asarray(spec.aim if spec.aim is not None else origin, dtype=np.float64)
        best: Optional[tuple[int, float, np.ndarray]] = None
        for i in range(steps + 1):
            t = 0.85 * i / max(steps, 1)
            probe = origin + (target - origin) * t
            near = int(np.count_nonzero(np.linalg.norm(pts - probe.astype(np.float32), axis=1) <= clearance))
            if best is None or near < best[0]:
                best = (near, t, probe)
            if near <= allowed:
                break
        assert best is not None
        near, t, position = best
        placement = {"buried_gaussians": near, "moved_fraction": round(float(t), 3),
                     "clearance": round(float(clearance), 4), "clear": near <= allowed}
        if t > 0:
            emit(
                f"    {spec.label}: moved {t:.0%} toward the aim point "
                f"({near} gaussians still within {clearance:.2f})"
            )
        out.append(
            ViewSpec(
                position=[float(v) for v in position],
                quaternion=look_at_quaternion(position, target, [0.0, 1.0, 0.0])
                if spec.aim is None
                else spec.quaternion if t == 0 else look_at_quaternion(
                    position, target, _up_from_quaternion(spec.quaternion)
                ),
                fov_y_deg=spec.fov_y_deg,
                width=spec.width,
                height=spec.height,
                label=spec.label,
                aim=spec.aim,
                placement=placement,
            )
        )
    return out


def _up_from_quaternion(quat_xyzw: Sequence[float]) -> np.ndarray:
    """The +Y column of a three.js camera's rotation — its own notion of up.

    Re-aiming a moved camera needs the SAME up reference the plan used, and the plan's
    up is recoverable from the pose it produced. Re-deriving it from the scene would
    silently roll a camera whose ring was planned around a different axis."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], dtype=np.float64
    )


def plan_object_views(
    center: Sequence[float],
    extents: Sequence[float],
    *,
    room_center: Sequence[float],
    up: Optional[Sequence[float]] = None,
    candidates: int = 4,
    fill_fraction: float = 0.45,
    fov_y_deg: float = 55.0,
    width: int = 1024,
    height: int = 1024,
    room_bounds: Optional[tuple[Sequence[float], Sequence[float]]] = None,
) -> list[ViewSpec]:
    """Camera candidates framing one object's OBB, best-first.

    The first candidate looks at the object from the room's interior — the direction a
    person in the room would see it from, which is also the direction least likely to be
    inside a wall. The rest sweep the remaining azimuths, so a caller that finds the
    object occluded has somewhere to go instead of giving up. Distance puts the OBB's
    bounding sphere across ``fill_fraction`` of the frame: too tight and SAM 3D gets no
    context, too loose and the object is a handful of pixels."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(extents, dtype=np.float64).reshape(3)
    upv = _normalize(np.asarray(up if up is not None else [0.0, 1.0, 0.0], dtype=np.float64).reshape(3))
    room = np.asarray(room_center, dtype=np.float64).reshape(3)

    radius = max(0.5 * float(np.linalg.norm(e)), 1e-4)
    distance = radius / max(math.tan(math.radians(fov_y_deg) * 0.5) * float(fill_fraction), 1e-6)

    inward = room - c
    inward = inward - upv * float(inward @ upv)  # horizontal component only
    if float(inward @ inward) < 1e-9:
        seed = np.array([0.0, 1.0, 0.0]) if abs(upv[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        inward = seed - upv * float(seed @ upv)
    inward = _normalize(inward)
    side = _normalize(np.cross(upv, inward))

    specs: list[ViewSpec] = []
    for i in range(max(int(candidates), 1)):
        angle = 2.0 * math.pi * i / max(int(candidates), 1)
        direction = _normalize(inward * math.cos(angle) + side * math.sin(angle))
        # Slightly above the object, looking down: a level camera on a floor-standing
        # object sees its silhouette against a wall and little else.
        position = c + direction * distance + upv * (distance * 0.35)
        if room_bounds is not None:
            lo = np.asarray(room_bounds[0], dtype=np.float64).reshape(3)
            hi = np.asarray(room_bounds[1], dtype=np.float64).reshape(3)
            pad = 0.02 * (hi - lo)
            position = np.minimum(np.maximum(position, lo + pad), hi - pad)
        specs.append(
            ViewSpec(
                position=[float(v) for v in position],
                quaternion=look_at_quaternion(position, c, upv),
                fov_y_deg=float(fov_y_deg),
                width=int(width),
                height=int(height),
                label=f"object view {i + 1}/{candidates}",
            )
        )
    return specs


# ---------------------------------------------------------------------------
# The rasterizer
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    """One frame. ``rgb`` is ``(H, W, 3)`` uint8; ``alpha`` and ``depth`` are ``(H, W)``
    float32, with ``depth`` the alpha-weighted expected depth (0 where nothing was hit)
    — that is what makes a geometric occlusion test possible without a second pass."""

    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    stats: dict[str, Any]


def _torch():
    try:
        import torch  # noqa: PLC0415  (lazy: keeps this module importable without torch)
    except ImportError as exc:  # pragma: no cover - exercised only on a torch-free box
        raise SplatRenderError(
            "the server-side rasterizer needs torch, which is not importable here"
        ) from exc
    return torch


def _sh_colour(torch, sh, dirs, degree: int):
    """``(N,K,3)`` coefficients + ``(N,3)`` unit view directions -> ``(N,3)`` linear RGB.

    Transcription of the reference 3DGS ``computeColorFromSH``. The DC term alone is the
    base albedo; the higher bands are the view-dependent part, which on the founder's
    SH-3 import is the difference between a flat matte room and one with the specular
    response its capture actually recorded."""
    result = _SH_C0 * sh[:, 0, :]
    if degree >= 1:
        x = dirs[:, 0:1]
        y = dirs[:, 1:2]
        z = dirs[:, 2:3]
        result = result - _SH_C1 * y * sh[:, 1, :] + _SH_C1 * z * sh[:, 2, :] - _SH_C1 * x * sh[:, 3, :]
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _SH_C2[0] * xy * sh[:, 4, :]
                + _SH_C2[1] * yz * sh[:, 5, :]
                + _SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6, :]
                + _SH_C2[3] * xz * sh[:, 7, :]
                + _SH_C2[4] * (xx - yy) * sh[:, 8, :]
            )
            if degree >= 3:
                result = (
                    result
                    + _SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9, :]
                    + _SH_C3[1] * xy * z * sh[:, 10, :]
                    + _SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11, :]
                    + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12, :]
                    + _SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13, :]
                    + _SH_C3[5] * z * (xx - yy) * sh[:, 14, :]
                    + _SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15, :]
                )
    return torch.clamp(result + 0.5, min=0.0)


class DeviceCloud:
    """A :class:`GaussianCloud` uploaded once and rendered many times.

    A ring is ten views of one scene; moving 2M gaussians to the device per view would
    dominate the cost and hide the actual render time behind transfer time."""

    def __init__(self, cloud: GaussianCloud, device: str = "cpu", dtype: Any = None) -> None:
        torch = _torch()
        self.torch = torch
        self.device = torch.device(device)
        dt = dtype or torch.float32
        self.means = torch.as_tensor(np.ascontiguousarray(cloud.means), dtype=dt, device=self.device)
        # exp() once here rather than per view: these never change.
        self.scales = torch.exp(
            torch.as_tensor(np.ascontiguousarray(cloud.scales_log), dtype=dt, device=self.device)
        )
        quats = torch.as_tensor(np.ascontiguousarray(cloud.quats_wxyz), dtype=dt, device=self.device)
        self.rot = _quats_to_matrices(torch, quats)
        self.opacity = torch.sigmoid(
            torch.as_tensor(np.ascontiguousarray(cloud.opacity_logit), dtype=dt, device=self.device)
        )
        self.sh = torch.as_tensor(np.ascontiguousarray(cloud.sh), dtype=dt, device=self.device)
        self.sh_degree = int(cloud.sh_degree)
        self.count = cloud.count
        self.source = dict(cloud.source)

    def subset(self, idx: Any) -> "DeviceCloud":
        """A view of ``idx`` of these gaussians, still on the device.

        Rendering one object's own gaussians alone is how its mask is made, and how the
        projection round-trip is proved on a real scene. Doing that by round-tripping to
        numpy and re-uploading would cost more than the render."""
        torch = self.torch
        sel = idx if torch.is_tensor(idx) else torch.as_tensor(
            np.asarray(idx, dtype=np.int64), device=self.device
        )
        sel = sel.to(self.device)
        clone = DeviceCloud.__new__(DeviceCloud)
        clone.torch = torch
        clone.device = self.device
        clone.means = self.means[sel]
        clone.scales = self.scales[sel]
        clone.rot = self.rot[sel]
        clone.opacity = self.opacity[sel]
        clone.sh = self.sh[sel]
        clone.sh_degree = self.sh_degree
        clone.count = int(sel.shape[0])
        clone.source = {**self.source, "subset": clone.count}
        return clone

    def means_numpy(self) -> np.ndarray:
        return self.means.detach().float().cpu().numpy()


def _quats_to_matrices(torch, quats_wxyz):
    """``(N,4)`` wxyz (the PLY ``rot_0..rot_3`` order) -> ``(N,3,3)``, normalised."""
    q = quats_wxyz / torch.clamp(quats_wxyz.norm(dim=1, keepdim=True), min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def render(
    cloud: Any,
    c2w: Any,
    intrinsics: Any,
    width: int,
    height: int,
    *,
    device: str = "cpu",
    background: Sequence[float] = (0.0, 0.0, 0.0),
    near: float = NEAR,
    tile: int = TILE,
) -> RenderResult:
    """Rasterize ``cloud`` from an OpenCV camera-to-world into a ``height x width`` frame.

    ``c2w`` and ``intrinsics`` are exactly what ``synthetic_views.parse_view`` stores
    (``[fx, fy, cx, cy]``), so the pixel this produces and the pixel
    ``project_world_point`` predicts are the same pixel — which the tests assert rather
    than assume, because a pose error here is invisible in the output."""
    torch = _torch()
    dc = cloud if isinstance(cloud, DeviceCloud) else DeviceCloud(cloud, device=device)
    dev = dc.device
    started = time.time()

    m = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    fx, fy, cx, cy = (float(v) for v in np.asarray(intrinsics, dtype=np.float64).reshape(4))
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise SplatRenderError(f"degenerate frame size {width}x{height}")

    rot_c2w = torch.as_tensor(m[:3, :3], dtype=dc.means.dtype, device=dev)
    cam_pos = torch.as_tensor(m[:3, 3], dtype=dc.means.dtype, device=dev)

    # world -> camera. c2w's rotation columns are the camera axes in world, so its
    # transpose is the world->camera rotation; batched, that is a right-multiply by the
    # matrix itself.
    rel = dc.means - cam_pos
    cam = rel @ rot_c2w
    z = cam[:, 2]

    front = z > float(near)
    if not bool(front.any()):
        return _empty_result(width, height, background, started, dc, "nothing in front of the camera")

    idx_front = torch.nonzero(front, as_tuple=False).reshape(-1)
    cam_f = cam[idx_front]
    zf = cam_f[:, 2]
    u = fx * cam_f[:, 0] / zf + cx
    v = fy * cam_f[:, 1] / zf + cy

    # 3D covariance carried into camera space in one product: with W the world->camera
    # rotation and M = R_g diag(s), Sigma_cam = (W M)(W M)^T.
    wm = torch.matmul(rot_c2w.transpose(0, 1).unsqueeze(0), dc.rot[idx_front])
    wm = wm * dc.scales[idx_front].unsqueeze(1)
    sigma = torch.matmul(wm, wm.transpose(1, 2))

    inv_z = 1.0 / zf
    inv_z2 = inv_z * inv_z
    j00 = fx * inv_z
    j02 = -fx * cam_f[:, 0] * inv_z2
    j11 = fy * inv_z
    j12 = -fy * cam_f[:, 1] * inv_z2
    # Sigma_2D = J Sigma J^T, written out because J is sparse (two of six entries zero)
    # and the dense 2x3 matmul would be pure waste at 2M gaussians.
    s00, s01, s02 = sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2]
    s11, s12, s22 = sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2]
    a = j00 * j00 * s00 + 2.0 * j00 * j02 * s02 + j02 * j02 * s22 + BLUR_2D
    b = j00 * j11 * s01 + j00 * j12 * s02 + j02 * j11 * s12 + j02 * j12 * s22
    c = j11 * j11 * s11 + 2.0 * j11 * j12 * s12 + j12 * j12 * s22 + BLUR_2D

    det = a * c - b * b
    ok = det > 1e-12
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(lam, min=0.0)))

    on_screen = (
        (u + radius > 0) & (u - radius < width) & (v + radius > 0) & (v - radius < height)
    )
    keep = ok & on_screen & (radius >= 1.0)
    if not bool(keep.any()):
        return _empty_result(width, height, background, started, dc, "no gaussian projects into the frame")

    sel = torch.nonzero(keep, as_tuple=False).reshape(-1)
    # Compose the two culls into ONE index into the full cloud. Indexing twice would
    # materialise a full-length SH gather first (2M x 16 x 3 floats = 384 MB) purely to
    # throw most of it away.
    idx_vis = idx_front[sel]
    u, v, radius = u[sel], v[sel], radius[sel]
    zf = zf[sel]
    det_s = det[sel]
    conic_a = c[sel] / det_s
    conic_b = -b[sel] / det_s
    conic_c = a[sel] / det_s
    opacity = dc.opacity[idx_vis]

    dirs = dc.means[idx_vis] - cam_pos
    dirs = dirs / torch.clamp(dirs.norm(dim=1, keepdim=True), min=1e-12)
    colour = _sh_colour(torch, dc.sh[idx_vis], dirs, dc.sh_degree)

    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    tx0 = torch.clamp(torch.div(u - radius, tile, rounding_mode="floor").long(), 0, tiles_x)
    tx1 = torch.clamp(torch.div(u + radius, tile, rounding_mode="floor").long() + 1, 0, tiles_x)
    ty0 = torch.clamp(torch.div(v - radius, tile, rounding_mode="floor").long(), 0, tiles_y)
    ty1 = torch.clamp(torch.div(v + radius, tile, rounding_mode="floor").long() + 1, 0, tiles_y)
    tw = torch.clamp(tx1 - tx0, min=0)
    th = torch.clamp(ty1 - ty0, min=0)
    n_tiles_each = tw * th

    oversized = int((n_tiles_each > MAX_TILES_PER_GAUSSIAN).sum().item())
    usable = (n_tiles_each > 0) & (n_tiles_each <= MAX_TILES_PER_GAUSSIAN)
    if not bool(usable.any()):
        return _empty_result(width, height, background, started, dc, "every gaussian was culled")
    keep2 = torch.nonzero(usable, as_tuple=False).reshape(-1)
    u, v, zf = u[keep2], v[keep2], zf[keep2]
    conic_a, conic_b, conic_c = conic_a[keep2], conic_b[keep2], conic_c[keep2]
    opacity, colour = opacity[keep2], colour[keep2]
    tx0, ty0, tw = tx0[keep2], ty0[keep2], tw[keep2]
    n_tiles_each = n_tiles_each[keep2]

    total_pairs = int(n_tiles_each.sum().item())
    if total_pairs > MAX_PAIRS:
        raise SplatRenderError(
            f"this view needs {total_pairs:,} (tile, gaussian) pairs, over the "
            f"{MAX_PAIRS:,} ceiling — render a smaller LOD level or a smaller frame"
        )

    n_vis = int(u.shape[0])
    g_idx = torch.repeat_interleave(torch.arange(n_vis, device=dev), n_tiles_each)
    offsets = torch.cumsum(n_tiles_each, 0) - n_tiles_each
    within = torch.arange(total_pairs, device=dev) - offsets[g_idx]
    w_rep = tw[g_idx]
    tile_id = (ty0[g_idx] + torch.div(within, w_rep, rounding_mode="floor")) * tiles_x + (
        tx0[g_idx] + within % w_rep
    )

    # Front-to-back within each tile: pack (tile, depth rank) into one int64 key so a
    # single sort does both. The rank (not the float depth) keeps the key exact.
    depth_order = torch.argsort(zf)
    rank = torch.empty(n_vis, dtype=torch.int64, device=dev)
    rank[depth_order] = torch.arange(n_vis, dtype=torch.int64, device=dev)
    key = (tile_id.to(torch.int64) << 32) | rank[g_idx]
    sorted_g = g_idx[torch.argsort(key)]

    n_tiles = tiles_x * tiles_y
    counts = torch.bincount(tile_id, minlength=n_tiles)
    starts = torch.cumsum(counts, 0) - counts

    # Deepest tiles first: chunks then hold tiles of similar occupancy, so padding to the
    # chunk's deepest tile costs little. Sorting the other way would pad every cheap tile
    # up to the most expensive one in its chunk.
    nonempty = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
    tile_order = nonempty[torch.argsort(counts[nonempty], descending=True)]

    px = tile * tiles_x  # padded frame width; the pad column is discarded at the end
    image = torch.zeros((tiles_y * tile * px + 1, 3), dtype=dc.means.dtype, device=dev)
    acc_alpha = torch.zeros(tiles_y * tile * px + 1, dtype=dc.means.dtype, device=dev)
    acc_depth = torch.zeros(tiles_y * tile * px + 1, dtype=dc.means.dtype, device=dev)

    pixel_count = tile * tile
    local = torch.arange(pixel_count, device=dev)
    local_x = (local % tile).to(dc.means.dtype)
    local_y = torch.div(local, tile, rounding_mode="floor").to(dc.means.dtype)

    blocks_run = 0
    for chunk_start in range(0, int(tile_order.shape[0]), TILES_PER_CHUNK):
        t_ids = tile_order[chunk_start : chunk_start + TILES_PER_CHUNK]
        n_t = int(t_ids.shape[0])
        cnt = counts[t_ids]
        st = starts[t_ids]
        deepest = int(cnt.max().item())

        t_x = (t_ids % tiles_x).to(dc.means.dtype).unsqueeze(1)
        t_y = torch.div(t_ids, tiles_x, rounding_mode="floor").to(dc.means.dtype).unsqueeze(1)
        # Pixel CENTRES: a world point at u lands in column floor(u), whose centre is
        # floor(u) + 0.5. Sampling at the corner instead shifts every render half a
        # pixel, which is exactly the size of error a projection round-trip must catch.
        pu = t_x * tile + local_x.unsqueeze(0) + 0.5
        pv = t_y * tile + local_y.unsqueeze(0) + 0.5
        inside = ((pu < width) & (pv < height)).to(dc.means.dtype)

        flat = (
            (t_y.to(torch.int64) * tile + local_y.to(torch.int64).unsqueeze(0)) * px
            + t_x.to(torch.int64) * tile
            + local_x.to(torch.int64).unsqueeze(0)
        )
        flat = torch.where(inside > 0, flat, torch.full_like(flat, image.shape[0] - 1))

        transmit = torch.ones((n_t, pixel_count), dtype=dc.means.dtype, device=dev)
        chunk_rgb = torch.zeros((n_t, pixel_count, 3), dtype=dc.means.dtype, device=dev)
        chunk_alpha = torch.zeros((n_t, pixel_count), dtype=dc.means.dtype, device=dev)
        chunk_depth = torch.zeros((n_t, pixel_count), dtype=dc.means.dtype, device=dev)

        for block in range(0, deepest, LAYER_BLOCK):
            width_b = min(LAYER_BLOCK, deepest - block)
            layer = torch.arange(block, block + width_b, device=dev)
            gather = torch.clamp(st.unsqueeze(1) + layer.unsqueeze(0), max=total_pairs - 1)
            gsel = sorted_g[gather]                      # (T, B)
            live = (layer.unsqueeze(0) < cnt.unsqueeze(1)).to(dc.means.dtype)

            du = pu.unsqueeze(1) - u[gsel].unsqueeze(2)   # (T, B, P)
            dv = pv.unsqueeze(1) - v[gsel].unsqueeze(2)
            power = -0.5 * (
                conic_a[gsel].unsqueeze(2) * du * du + conic_c[gsel].unsqueeze(2) * dv * dv
            ) - conic_b[gsel].unsqueeze(2) * du * dv
            alpha = opacity[gsel].unsqueeze(2) * torch.exp(torch.clamp(power, max=0.0))
            alpha = torch.clamp(alpha, max=0.99) * live.unsqueeze(2) * inside.unsqueeze(1)

            # Front-to-back "over": T_k = prod_{j<k}(1 - a_j), carried across blocks so
            # the composite is the full ordered list, not the first LAYER_BLOCK of it.
            cumulative = torch.cumprod(1.0 - alpha, dim=1)
            prev = torch.cat(
                [torch.ones_like(cumulative[:, :1, :]), cumulative[:, :-1, :]], dim=1
            ) * transmit.unsqueeze(1)
            weight = alpha * prev
            chunk_rgb = chunk_rgb + torch.einsum("tbp,tbc->tpc", weight, colour[gsel])
            chunk_alpha = chunk_alpha + weight.sum(dim=1)
            chunk_depth = chunk_depth + (weight * zf[gsel].unsqueeze(2)).sum(dim=1)
            transmit = transmit * cumulative[:, -1, :]
            blocks_run += 1
            if float(transmit.max().item()) < MIN_TRANSMITTANCE:
                break

        flat_flat = flat.reshape(-1)
        image[flat_flat] = chunk_rgb.reshape(-1, 3)
        acc_alpha[flat_flat] = chunk_alpha.reshape(-1)
        acc_depth[flat_flat] = chunk_depth.reshape(-1)

    bg = torch.as_tensor(list(background), dtype=dc.means.dtype, device=dev).reshape(1, 1, 3)
    padded_rgb = image[:-1].reshape(tiles_y * tile, px, 3)[:height, :width, :]
    padded_alpha = acc_alpha[:-1].reshape(tiles_y * tile, px)[:height, :width]
    padded_depth = acc_depth[:-1].reshape(tiles_y * tile, px)[:height, :width]
    composed = padded_rgb + bg * (1.0 - padded_alpha).unsqueeze(2)

    rgb8 = (torch.clamp(composed, 0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    alpha = padded_alpha.float().cpu().numpy()
    depth = (padded_depth / torch.clamp(padded_alpha, min=1e-8)).float().cpu().numpy()
    depth[alpha < 1e-4] = 0.0

    # Median depth over the pixels that actually hit something. This is the number that
    # tells a camera standing 40 cm from a poster apart from a camera looking across a
    # room: both render a full frame at coverage 1.0, and only one of them is a view.
    hit = depth[alpha >= 0.5]
    depth_p50 = float(np.median(hit)) if hit.size else 0.0

    return RenderResult(
        rgb=rgb8,
        alpha=alpha,
        depth=depth,
        stats={
            "gaussians_total": dc.count,
            "gaussians_drawn": n_vis,
            "gaussians_oversized_dropped": oversized,
            "tile_pairs": total_pairs,
            "tiles_touched": int(tile_order.shape[0]),
            "layer_blocks": blocks_run,
            "sh_degree": dc.sh_degree,
            "device": str(dev),
            "seconds": round(time.time() - started, 3),
            "coverage": float(np.mean(alpha > 0.5)),
            "depth_p50": depth_p50,
        },
    )


def _empty_result(width, height, background, started, dc, reason: str) -> RenderResult:
    """A frame with nothing in it, and the reason recorded — a blank render is a real
    answer (the camera is inside a wall, say) and must not be confused with a crash."""
    bg = np.asarray(list(background), dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(np.repeat(np.repeat(bg, height, axis=0), width, axis=1) * 255.0, 0, 255).astype(
        np.uint8
    )
    return RenderResult(
        rgb=rgb,
        alpha=np.zeros((height, width), dtype=np.float32),
        depth=np.zeros((height, width), dtype=np.float32),
        stats={
            "gaussians_total": dc.count,
            "gaussians_drawn": 0,
            "empty_reason": reason,
            "sh_degree": dc.sh_degree,
            "device": str(dc.device),
            "seconds": round(time.time() - started, 3),
            "coverage": 0.0,
            "depth_p50": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Encoding + measurement helpers
# ---------------------------------------------------------------------------


def encode_png(rgb: np.ndarray) -> bytes:
    """``(H, W, 3)`` uint8 -> PNG bytes. PNG, losslessly, because the synthetic-view
    route stores exactly what was rendered — a re-encoded render is no longer faithful
    evidence of what the renderer produced."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Bool/float mask -> an 8-bit L PNG, the shape ``routes_sam3d`` already feeds SAM 3D."""
    from PIL import Image

    buf = io.BytesIO()
    arr = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def alpha_centroid(alpha: np.ndarray, threshold: float = 0.05) -> Optional[tuple[float, float]]:
    """Alpha-weighted centroid of a render, in the SAME pixel coordinates
    ``project_world_point`` returns (column + 0.5 for a pixel's centre).

    This is the executable half of the frame-convention proof: render a handful of
    gaussians around a known world point, and this must come back at the pixel the
    projection predicted. ``None`` when nothing was drawn."""
    a = np.asarray(alpha, dtype=np.float64)
    weights = np.where(a >= threshold, a, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    ys, xs = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    return (
        float((weights * (xs + 0.5)).sum() / total),
        float((weights * (ys + 0.5)).sum() / total),
    )


def visibility(
    object_alpha: np.ndarray,
    object_depth: np.ndarray,
    scene_depth: np.ndarray,
    *,
    alpha_threshold: float = 0.35,
    depth_tolerance: float = 0.05,
) -> dict[str, Any]:
    """How much of an object is actually SEEN in a render, and its unoccluded mask.

    The object's own gaussians are rendered alone (so they are never hidden), then
    compared against the full scene's expected depth: a pixel counts only where the
    scene's front surface is at or behind the object's. Without that test a "view of the
    object" can be a view of the wall in front of it, and SAM 3D would be handed a mask
    over furniture it cannot see — the sort of confident nonsense that is worse than a
    refusal."""
    obj = np.asarray(object_alpha, dtype=np.float32)
    hit = obj >= float(alpha_threshold)
    tol = float(depth_tolerance) * np.maximum(np.asarray(object_depth, dtype=np.float32), 1e-6)
    unoccluded = hit & (np.asarray(scene_depth, dtype=np.float32) >= (object_depth - tol))
    hit_px = int(hit.sum())
    seen_px = int(unoccluded.sum())
    return {
        "mask": unoccluded,
        "object_pixels": hit_px,
        "visible_pixels": seen_px,
        "visible_fraction": (seen_px / hit_px) if hit_px else 0.0,
        "frame_fraction": seen_px / float(obj.size),
    }
