"""Level-of-detail (LOD) generation for persisted gaussian splats.

WHY THIS EXISTS
---------------
Persisted demo scenes are far too big to render in a browser as exported:

    demo-ops/canonical-office-loop   18,846,947 gaussians   1.19 GiB splat.ply
    <founder scene>                  63,308,835 gaussians   4.01 GiB splat.ply

Both are unusable over the wire, and 63M gaussians will not render interactively
in ANY browser: Spark packs to 16 B/splat (≈1.0 GB of packed data at 63M) before
counting the per-frame radix-sort buffers, so the ceiling is memory, not just
bandwidth. Decimation is mandatory; transport compression alone cannot fix it.

WHAT THE REAL FILES LOOK LIKE (measured 2026-07-31, both scenes)
----------------------------------------------------------------
* 17 float32 properties, 68 B/gaussian, binary little-endian.
* ``nx, ny, nz`` are **identically zero** in both files (absmax 0.0). They are
  pure padding — 12 of every 68 bytes. Dropping them is a free 17.6% cut with
  zero fidelity loss (3DGS renderers, Spark included, never read normals).
* The founder scene's ``opacity`` is a **constant** (alpha == 0.9 for all 63.3M
  gaussians) and its scales are already clamped at the p99 tail, so the usual
  "opacity x area" importance signal is nearly flat: the top 1M gaussians by
  importance carry only 11.1% of total importance (17.7% on canonical).

That last measurement is decisive and it is why this module does NOT use
importance top-K selection. Top-K on a flat importance distribution degenerates
into "keep the biggest blobs", which leaves gaping holes in the surface. The
selection here is **spatially stratified**: a voxel grid over the scene, keeping
the single highest-importance gaussian per occupied voxel. That preserves
coverage (the thing you actually see) and stays importance-aware within each
cell, so it still does the right thing on a scene whose opacity is informative.

HOLE COMPENSATION (honest, and labelled in the UI)
--------------------------------------------------
Thinning a surface by 40x makes the survivors too small to cover it — you see
through the walls. We therefore apply a **scale floor** tied to the voxel size:
``scale_linear = max(scale_linear, voxel * SCALE_FLOOR_FRAC)``. This grows only
the gaussians that became too small for their new spacing and leaves already-large
ones untouched. It is a rendering approximation, not a measurement, so every LOD
records ``scale_floor_applied`` in its metadata and the client shows an LOD chip.
No geometry is invented: positions, colors, rotations and opacity are the
original values of surviving gaussians.

MEMORY DISCIPLINE
-----------------
Never loads a multi-GB splat into RAM. The vertex block is opened with
``np.memmap`` over the exact header offset, columns are pulled one at a time and
freed, and the output is streamed to disk. Runs in a Modal side-function
(``modal_oreos_lod.py``), never on the 1 CPU / 4 GB broker.

Dependency-light on purpose (numpy only, like ``scene_report/splat_io.py``) so it
stays importable anywhere in the server without dragging in torch/open3d.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Properties an LOD file carries, in write order. This is the 17-field exporter
#: schema minus the all-zero normals (see module docstring).
LOD_PROPERTIES = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)

#: Properties that must exist in the source for it to be a gaussian splat.
REQUIRED_SOURCE_PROPERTIES = LOD_PROPERTIES

#: Normal fields dropped when (and only when) they are identically zero.
NORMAL_PROPERTIES = ("nx", "ny", "nz")

#: Fraction of the voxel edge used as the post-decimation lower bound on
#: per-axis gaussian scale. 0.5 => a survivor is at least half a cell wide, which
#: closes the surface without visibly blurring detail (A/B'd on both real scenes).
SCALE_FLOOR_FRAC = 0.5

#: Bits per axis in the packed voxel key. 21 bits => 2,097,152 cells per axis,
#: far beyond any voxel resolution we target (~10^3 cells/axis).
_VOXEL_BITS = 21
_VOXEL_MAX = (1 << _VOXEL_BITS) - 1

#: Gaussian-count budgets, smallest first. Only levels strictly smaller than the
#: source count are built. Chosen against a MEASURED framerate knee on an M1 Pro
#: (ANGLE/Metal, 1280x800, delivered fps via sync readPixels):
#:
#:     <=2M -> ~119 fps   4M -> 73 fps   8M -> 38 fps   18.85M -> 15.6 fps
#:
#: The knee is sharp and sits between 2M and 4M, so 2M is the largest budget that
#: is still comfortably pinned at refresh rate — that is the default.
#:
#: 8M is carried as well, for RECORDING rather than for interaction. Measured
#: 2026-07-31 on the canonical scene, scoring each level's render against the
#: full-density render at identical camera poses
#: (docs/demo-2026-07/evidence/splat-density/scores.json):
#:
#:     level   SSIM    PSNR     high-frequency detail retained
#:     2M      0.932   29.4 dB  0.834
#:     8M      0.967   33.7 dB  0.975
#:
#: That last column is the one that matters for "it looks low resolution": at 2M
#: about 17% of the scene's high-frequency detail is gone, at 8M about 2.5% is.
#: 38 fps is unpleasant to drive but entirely fine to record, so the level exists
#: and the viewer offers it — it is never the default.
DEFAULT_LEVELS = (600_000, 2_000_000, 4_000_000, 8_000_000)

#: The level the client loads by default when it has no better signal.
DEFAULT_LEVEL = 2_000_000

#: Above this source size we do not offer a "full detail" artifact at all:
#: 63.3M gaussians is ~1.0 GB of packed splat data in the browser before sort
#: buffers, which does not render interactively on any GPU we target. Offering it
#: would be a trap, so the client is told plainly that the top LOD is the ceiling.
FULL_DETAIL_MAX_GAUSSIANS = 25_000_000

#: Derived-key namespace (BUILD-PLAN collision contract).
LOD_PREFIX = "demo/lod"
LOD_INDEX_KEY = "demo/lod/index.json"

#: Transport format. Spark reads .spz natively (SplatFileType.SPZ); it is ~9-15x
#: smaller than the equivalent PLY because it quantizes positions to 24-bit fixed
#: point, scales/rotations/colour to bytes, then gzips. Measured on the canonical
#: scene: 84.0 MB PLY (1.5M) -> 8.81 MB spz. We ship BOTH: .spz is what the client
#: loads, .ply stays as a debuggable, tool-compatible fallback.
SPZ_SUFFIX = ".spz"
PLY_SUFFIX = ".ply"

#: Schema version of index.json, so the client can refuse an old shape.
LOD_INDEX_VERSION = 1


def level_name(budget: int) -> str:
    """``2_000_000 -> "2000k"`` — the token used in the derived key."""
    return f"{int(budget) // 1000}k"


def level_key(budget: int, suffix: str = PLY_SUFFIX) -> str:
    """Derived key (relative to ``derived/``) for a level's splat."""
    return f"{LOD_PREFIX}/splat_{level_name(budget)}{suffix}"


def full_key(suffix: str = SPZ_SUFFIX) -> str:
    """Derived key for the compressed full-detail artifact (no decimation)."""
    return f"{LOD_PREFIX}/full{suffix}"


# ---------------------------------------------------------------------------
# PLY reading (memmap, never a full load)
# ---------------------------------------------------------------------------


class SplatSource:
    """A memmapped binary-LE gaussian PLY. Columns are read on demand."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.properties, self.count, self.data_offset = read_ply_header(path)
        self.itemsize = 4 * len(self.properties)
        self.file_bytes = os.path.getsize(path)
        expected = self.data_offset + self.itemsize * self.count
        if self.file_bytes < expected:
            raise ValueError(
                f"truncated PLY: header declares {self.count} vertices "
                f"({expected} bytes) but file is {self.file_bytes} bytes"
            )
        missing = [p for p in REQUIRED_SOURCE_PROPERTIES if p not in self.properties]
        if missing:
            raise ValueError(f"not a gaussian splat — missing propert(ies): {missing}")
        dtype = np.dtype([(name, "<f4") for name in self.properties])
        self._mm = np.memmap(path, dtype=dtype, mode="r", offset=self.data_offset, shape=(self.count,))

    def column(self, name: str) -> np.ndarray:
        """A contiguous float32 copy of one property (``count`` elements)."""
        return np.ascontiguousarray(self._mm[name], dtype=np.float32)

    def take(self, name: str, idx: np.ndarray) -> np.ndarray:
        """``column(name)[idx]`` without materialising the whole column."""
        return np.ascontiguousarray(self._mm[name][idx], dtype=np.float32)

    def close(self) -> None:
        self._mm = None  # type: ignore[assignment]

    def __enter__(self) -> "SplatSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_ply_header(path: str) -> tuple[list[str], int, int]:
    """Parse a binary-little-endian PLY header.

    Returns ``(property_names_in_order, vertex_count, data_offset_bytes)``.
    Only ``float`` (4-byte) properties are supported — the exporter schema this
    server produces and consumes is all-float32 (see ``splat_io``).
    """
    names: list[str] = []
    count: Optional[int] = None
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = f.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"unsupported PLY format: {fmt!r}")
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            stripped = line.strip()
            if stripped.startswith(b"element vertex"):
                count = int(stripped.split()[-1])
            elif stripped.startswith(b"element "):
                # A non-vertex element after the vertex block would invalidate our
                # flat-array assumption. Our exporter never writes one.
                raise ValueError(f"unsupported extra PLY element: {stripped!r}")
            elif stripped.startswith(b"property"):
                parts = stripped.split()
                if parts[1] != b"float":
                    raise ValueError(f"unsupported PLY property type: {parts[1]!r}")
                names.append(parts[-1].decode("ascii"))
            elif stripped == b"end_header":
                if count is None:
                    raise ValueError("PLY header missing 'element vertex <N>'")
                return names, count, f.tell()


def normals_are_zero(src: SplatSource) -> bool:
    """True when every ``nx/ny/nz`` is exactly 0 (so dropping them is lossless)."""
    for name in NORMAL_PROPERTIES:
        if name not in src.properties:
            return True  # already absent — nothing to drop
        col = src.column(name)
        nonzero = bool(np.any(col != 0.0))
        del col
        if nonzero:
            return False
    return True


# ---------------------------------------------------------------------------
# Importance + voxel selection
# ---------------------------------------------------------------------------


def compute_importance(src: SplatSource) -> np.ndarray:
    """Per-gaussian screen contribution proxy: ``alpha * projected_area``.

    ``alpha = sigmoid(opacity)`` (the exporter stores the logit) and area is
    approximated by the squared geometric mean of the three linear axis scales
    (the exporter stores ``log`` scales). Returns float32 ``(N,)``.

    On both real scenes this comes out nearly flat — which is exactly why it is
    used only to rank *within* a voxel, never to select globally.
    """
    opacity = src.column("opacity")
    alpha = np.empty_like(opacity)
    # sigmoid, computed stably in-place
    np.negative(opacity, out=alpha)
    np.exp(alpha, out=alpha)
    np.add(alpha, 1.0, out=alpha)
    np.reciprocal(alpha, out=alpha)
    del opacity

    log_mean = np.zeros(src.count, dtype=np.float32)
    for axis in ("scale_0", "scale_1", "scale_2"):
        col = src.column(axis)
        log_mean += col
        del col
    log_mean /= 3.0
    # (exp(mean_log))^2 == exp(2 * mean_log)
    log_mean *= 2.0
    np.exp(log_mean, out=log_mean)

    log_mean *= alpha
    del alpha
    return log_mean


def _voxel_keys(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, origin: np.ndarray, voxel: float
) -> np.ndarray:
    """Pack per-axis voxel coordinates into one int64 key per gaussian."""
    inv = 1.0 / float(voxel)
    key = np.empty(xs.shape[0], dtype=np.int64)
    for shift, col, lo in (
        (2 * _VOXEL_BITS, xs, origin[0]),
        (1 * _VOXEL_BITS, ys, origin[1]),
        (0, zs, origin[2]),
    ):
        grid = (col - lo) * inv
        np.floor(grid, out=grid)
        np.clip(grid, 0, _VOXEL_MAX, out=grid)
        part = grid.astype(np.int64)
        del grid
        if shift:
            part <<= shift
        if shift == 2 * _VOXEL_BITS:
            key[:] = part
        else:
            key |= part
        del part
    return key


def _occupied_count(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, origin: np.ndarray, voxel: float
) -> int:
    keys = _voxel_keys(xs, ys, zs, origin, voxel)
    n = int(np.unique(keys).size)
    del keys
    return n


def choose_voxel_size(
    src: SplatSource,
    target: int,
    *,
    probe_max: int = 8_000_000,
    max_iters: int = 10,
    tolerance: float = 0.03,
    rng_seed: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Newton-search (on the ``O ~ A/v^2`` surface law) the voxel edge whose
    occupied-cell count lands on ``target``.

    The search runs on a random probe subsample (default 8M) rather than the full
    array, because each iteration costs a sort. **The probe targets the SAME cell
    count as the full cloud, deliberately un-scaled by the sampling fraction** —
    occupied-cell count is dominated by geometry, not by sample count: at our
    densities a cell holds tens of gaussians, so keeping a fifth of them leaves
    essentially every cell still occupied. Scaling the target by the sampling
    fraction (the intuitive-but-wrong move) made the search undershoot the budget
    by 2.25x on the canonical scene.

    The estimate is still only an estimate; :func:`select_indices` verifies it
    against the full array and corrects. Returns ``(voxel, origin, info)``.
    """
    emit = log or (lambda _m: None)
    xs = src.column("x")
    ys = src.column("y")
    zs = src.column("z")

    # Robust extent: raw SLAM bounding boxes are outlier-stretched (a handful of
    # stray gaussians can double the box), which would make the first guess wild.
    lo = np.array(
        [np.percentile(xs, 0.5), np.percentile(ys, 0.5), np.percentile(zs, 0.5)], dtype=np.float64
    )
    hi = np.array(
        [np.percentile(xs, 99.5), np.percentile(ys, 99.5), np.percentile(zs, 99.5)],
        dtype=np.float64,
    )
    extent = np.maximum(hi - lo, 1e-6)
    # origin must cover the true min so no gaussian clips to cell 0
    origin = np.array([xs.min(), ys.min(), zs.min()], dtype=np.float64) - 1e-6

    if src.count > probe_max:
        rng = np.random.default_rng(rng_seed)
        probe_idx = rng.choice(src.count, size=probe_max, replace=False)
        probe_idx.sort()
        px, py, pz = xs[probe_idx], ys[probe_idx], zs[probe_idx]
        probe_n = probe_max
        del probe_idx
    else:
        px, py, pz = xs, ys, zs
        probe_n = src.count
    del xs, ys, zs

    probe_target = max(1, int(target))

    # First guess: assume the scene is a surface of area ~ the two largest extents.
    order = np.sort(extent)[::-1]
    approx_area = float(order[0] * order[1])
    voxel = float(np.sqrt(max(approx_area, 1e-12) / max(probe_target, 1)))
    history: list[dict[str, Any]] = []

    for _ in range(max_iters):
        occ = _occupied_count(px, py, pz, origin, voxel)
        history.append({"voxel": voxel, "occupied": occ})
        emit(f"    voxel={voxel:.6g} -> occupied(probe)={occ:,} (target {probe_target:,})")
        if occ == 0:
            voxel *= 0.5
            continue
        err = abs(occ - probe_target) / probe_target
        if err <= tolerance:
            break
        # O ~ v^-2  =>  v_next = v * sqrt(O / target)
        step = float(np.sqrt(occ / probe_target))
        step = float(np.clip(step, 0.25, 4.0))
        voxel *= step

    del px, py, pz
    info = {
        "voxel_size": voxel,
        "probe_points": int(probe_n),
        "probe_target": int(probe_target),
        "iterations": len(history),
        "history": history,
        "extent": extent.tolist(),
    }
    return voxel, origin, info


def select_indices(
    src: SplatSource,
    target: int,
    *,
    method: str = "voxel_importance",
    rng_seed: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose which gaussians survive at ``target`` count.

    Methods:
      ``voxel_importance`` (default) — voxel grid; keep the highest-importance
          gaussian in each occupied cell. Spatially uniform AND importance-aware.
      ``voxel_first`` — same grid, keep an arbitrary (deterministic) member.
          Cheaper by one sort; used to measure what the importance ranking buys.
      ``uniform`` — random subsample. The naive baseline.
      ``importance`` — global top-K by importance. Measured to be the WORST of
          the four on these scenes; kept so the comparison is reproducible.

    Returns ``(sorted_indices, info)``.
    """
    emit = log or (lambda _m: None)
    n = src.count
    if target >= n:
        return np.arange(n, dtype=np.int64), {"method": "identity", "selected": n}

    if method == "uniform":
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n, size=target, replace=False)
        idx.sort()
        return idx, {"method": method, "selected": int(idx.size)}

    if method == "importance":
        imp = compute_importance(src)
        idx = np.argpartition(imp, n - target)[n - target:]
        del imp
        idx = np.sort(idx.astype(np.int64))
        return idx, {"method": method, "selected": int(idx.size)}

    if method == "auto":
        # Importance ranking within a cell only earns its extra sort when opacity
        # actually varies. The founder's 63.3M scene stores a CONSTANT alpha
        # (0.9 everywhere), so ranking is provably a no-op there; detect that in
        # one cheap pass and take the 5x-faster path.
        opacity = src.column("opacity")
        sample = opacity if opacity.size <= 4_000_000 else opacity[:: max(1, opacity.size // 4_000_000)]
        varies = bool(np.ptp(sample) > 1e-6)
        del opacity, sample
        method = "voxel_importance" if varies else "voxel_first"
        emit(f"    auto -> {method} (opacity {'varies' if varies else 'is constant'})")

    if method not in ("voxel_importance", "voxel_first"):
        raise ValueError(f"unknown LOD selection method: {method!r}")

    voxel, origin, vinfo = choose_voxel_size(src, target, rng_seed=rng_seed, log=log)

    xs = src.column("x")
    ys = src.column("y")
    zs = src.column("z")
    order: Optional[np.ndarray] = None
    if method == "voxel_importance":
        imp = compute_importance(src)
        # Stable descending sort by importance; first-per-cell then wins the cell.
        order = np.argsort(-imp, kind="stable")
        del imp

    idx = np.empty(0, dtype=np.int64)
    achieved = 0
    attempts: list[dict[str, Any]] = []
    # The probe estimate is good but not exact. Verify against the full array and
    # Newton-correct; in practice this converges on the first or second pass.
    for attempt in range(3):
        keys = _voxel_keys(xs, ys, zs, origin, voxel)
        if order is not None:
            _, first = np.unique(keys[order], return_index=True)
            del keys
            idx = order[first]
        else:
            _, first = np.unique(keys, return_index=True)
            del keys
            idx = first.astype(np.int64)
        del first
        achieved = int(idx.size)
        attempts.append({"voxel": float(voxel), "cells": achieved})
        emit(f"    full pass {attempt + 1}: voxel={voxel:.6g} -> {achieved:,} cells (budget {target:,})")
        # Overshoot is cheap to fix (trim below); undershoot is not, so only
        # re-run when we are short or wildly over.
        if 0.90 * target <= achieved <= 1.15 * target:
            break
        if attempt == 2:
            break
        voxel *= float(np.clip(np.sqrt(achieved / target), 0.25, 4.0))

    del xs, ys, zs
    if order is not None:
        del order

    idx = np.sort(idx.astype(np.int64))
    trimmed = 0
    if idx.size > target:
        # Overshoot: thin uniformly (never importance-biased, which would
        # reintroduce exactly the spatial clustering the voxel grid removed).
        rng = np.random.default_rng(rng_seed)
        keep = rng.choice(idx.size, size=target, replace=False)
        keep.sort()
        trimmed = int(idx.size) - target
        idx = idx[keep]
        del keep

    info = {
        "method": method,
        "voxel_size": float(voxel),
        "voxel_search": vinfo,
        "full_passes": attempts,
        "occupied_cells": achieved,
        "trimmed_to_budget": trimmed,
        "selected": int(idx.size),
    }
    emit(f"    selected {idx.size:,} of {n:,} (cells {achieved:,}, trimmed {trimmed:,})")
    return idx, info


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _ply_header(count: int, properties: Iterable[str]) -> bytes:
    head = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(int(count))
    head += "".join(f"property float {name}\n" for name in properties)
    head += "end_header\n"
    return head.encode("ascii")


def write_lod_ply(
    src: SplatSource,
    idx: np.ndarray,
    out_path: str,
    *,
    voxel_size: Optional[float] = None,
    scale_floor_frac: float = SCALE_FLOOR_FRAC,
    drop_normals: bool = True,
) -> dict[str, Any]:
    """Write the selected gaussians to ``out_path`` as a 14-property splat PLY.

    Applies the voxel-tied scale floor (see module docstring) when ``voxel_size``
    is known and ``scale_floor_frac > 0``. Returns metadata describing exactly
    what was done, for the LOD index and the client's provenance chip.
    """
    props = list(LOD_PROPERTIES) if drop_normals else [
        p for p in src.properties if p in set(LOD_PROPERTIES) | set(NORMAL_PROPERTIES)
    ]
    n = int(idx.size)

    columns: dict[str, np.ndarray] = {}
    for name in props:
        columns[name] = src.take(name, idx)

    scale_floor_applied = False
    floor_log = None
    raised = 0
    if voxel_size and scale_floor_frac > 0:
        floor_linear = float(voxel_size) * float(scale_floor_frac)
        if floor_linear > 0:
            floor_log = float(np.log(floor_linear))
            for axis in ("scale_0", "scale_1", "scale_2"):
                col = columns[axis]
                raised += int(np.count_nonzero(col < floor_log))
                np.maximum(col, floor_log, out=col)
            scale_floor_applied = raised > 0

    dtype = np.dtype([(name, "<f4") for name in props])
    verts = np.empty(n, dtype=dtype)
    for name in props:
        verts[name] = columns[name]
        del columns[name]

    tmp_path = out_path + ".partial"
    with open(tmp_path, "wb") as f:
        f.write(_ply_header(n, props))
        verts.tofile(f)
    os.replace(tmp_path, out_path)
    del verts

    return {
        "gaussians": n,
        "bytes": os.path.getsize(out_path),
        "properties": props,
        "bytes_per_gaussian": 4 * len(props),
        "normals_dropped": bool(drop_normals),
        "scale_floor_applied": scale_floor_applied,
        "scale_floor_linear": (float(np.exp(floor_log)) if floor_log is not None else None),
        "scale_axes_raised": raised,
    }


class SpzEncodeError(RuntimeError):
    """Raised when the Spark-backed SPZ encoder is unavailable or fails."""


def encode_spz(ply_path: str, spz_path: str, *, timeout_s: int = 900) -> dict[str, Any]:
    """Encode ``ply_path`` to ``spz_path`` via Spark's own ``transcodeSpz``.

    Shells out to ``scripts/ply_to_spz.mjs`` (see that file for why the encoder is
    the renderer's own code rather than a Python reimplementation). Returns the
    script's JSON report. Raises :class:`SpzEncodeError` on any failure — callers
    are expected to degrade to serving the PLY, never to serve a corrupt spz.

    NOTE: never call this on the broker. It is invoked from the Modal LOD job,
    where node and the webserver's ``node_modules`` are present in the image.
    """
    import subprocess

    script = os.path.join(_repo_root(), "scripts", "ply_to_spz.mjs")
    if not os.path.isfile(script):
        raise SpzEncodeError(f"encoder script missing: {script}")
    cmd = ["node", script, ply_path, spz_path, "--verify"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError as exc:  # node not on PATH
        raise SpzEncodeError(f"node not available: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SpzEncodeError(f"spz encode timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        raise SpzEncodeError(
            f"spz encode failed (rc={proc.returncode}): {(proc.stderr or '').strip()[:500]}"
        )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        raise SpzEncodeError("spz encoder produced no output")
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError as exc:
        raise SpzEncodeError(f"unparseable encoder output: {line[-1][:200]}") from exc


def _repo_root() -> str:
    """``/root/project`` in the Modal image; the server checkout root locally."""
    env = os.environ.get("OPENREALITY_REPO_ROOT")
    if env:
        return env
    # server/oreos/lod.py -> server/oreos -> server -> <root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_level(
    src: SplatSource,
    budget: int,
    out_path: str,
    *,
    method: str = "auto",
    scale_floor_frac: float = SCALE_FLOOR_FRAC,
    drop_normals: bool = True,
    rng_seed: int = 0,
    spz_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Select + write one LOD level (PLY, and SPZ when ``spz_path`` is given).

    Returns the index entry. An SPZ failure is recorded on the entry and the level
    still ships as PLY — a slower load is a far better outcome than no scene.
    """
    emit = log or (lambda _m: None)
    emit(f"  level {level_name(budget)}: selecting (method={method})")
    idx, sel_info = select_indices(src, budget, method=method, rng_seed=rng_seed, log=log)
    write_info = write_lod_ply(
        src,
        idx,
        out_path,
        voxel_size=sel_info.get("voxel_size"),
        scale_floor_frac=scale_floor_frac,
        drop_normals=drop_normals,
    )
    entry = {
        "name": level_name(budget),
        "budget": int(budget),
        "key": level_key(budget, PLY_SUFFIX),
        "selection": sel_info,
        **write_info,
    }
    entry["reduction"] = round(src.count / max(entry["gaussians"], 1), 2)
    entry["size_reduction"] = round(src.file_bytes / max(entry["bytes"], 1), 2)

    if spz_path:
        try:
            rep = encode_spz(out_path, spz_path)
            entry["spz_key"] = level_key(budget, SPZ_SUFFIX)
            entry["spz_bytes"] = int(rep["output_bytes"])
            entry["spz_ratio_vs_ply"] = rep["ratio"]
            entry["spz_size_reduction"] = round(src.file_bytes / max(int(rep["output_bytes"]), 1), 2)
            entry["spz_verify"] = rep.get("verify")
            emit(
                f"  level {level_name(budget)}: spz {rep['output_bytes'] / 1e6:.2f} MB "
                f"({entry['spz_size_reduction']}x smaller than source)"
            )
        except SpzEncodeError as exc:
            entry["spz_error"] = str(exc)[:300]
            emit(f"  level {level_name(budget)}: SPZ FAILED ({exc}) — shipping PLY only")

    emit(
        f"  level {level_name(budget)}: {entry['gaussians']:,} gaussians, "
        f"PLY {entry['bytes'] / 1e6:.1f} MB ({entry['size_reduction']}x smaller)"
    )
    return entry


def plan_levels(source_count: int, levels: Iterable[int] = DEFAULT_LEVELS) -> list[int]:
    """Which budgets are worth building for a source of ``source_count``.

    A level is skipped when it would not meaningfully shrink the source (within
    1.3x), because serving a near-identical copy just wastes volume and time.
    """
    out = []
    for budget in sorted(int(b) for b in levels):
        if source_count > budget * 1.3:
            out.append(budget)
    return out


def build_index(
    *,
    scan_id: str,
    source_count: int,
    source_bytes: int,
    entries: list[dict[str, Any]],
    default_level: int = DEFAULT_LEVEL,
    generator: str = "voxel_importance",
    source_key: str = "splat.ply",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The ``derived/demo/lod/index.json`` document the client reads.

    ``source_key`` records WHICH splat these levels were decimated from — either
    ``"splat.ply"`` or a ``derived/...`` key when the scene carries an anchored or
    clamped splat. The client compares it against the scene's current splat key and
    treats a mismatch as stale, because an LOD built from the un-anchored splat
    would silently render at a different metric gauge than the overlays.
    """
    available = [e["budget"] for e in entries]
    chosen = None
    for budget in sorted(available):
        if budget >= default_level:
            chosen = budget
            break
    if chosen is None and available:
        chosen = max(available)

    doc: dict[str, Any] = {
        "version": LOD_INDEX_VERSION,
        "scan_id": scan_id,
        "source": {"key": source_key, "gaussians": int(source_count), "bytes": int(source_bytes)},
        "default_level": chosen,
        "levels": entries,
        # Honesty envelope — mirrors the object-layer pattern used elsewhere.
        "provenance": (
            "Decimated view of the reconstruction: spatially stratified selection "
            "(one gaussian per voxel cell, highest screen contribution wins). "
            "Positions, colours, rotations and opacity are original values; "
            "per-gaussian size is floored to the cell spacing so the surface stays "
            "closed. Full detail remains available."
        ),
        "generated": True,
        "generator": generator,
        "caveats": [
            "Fewer gaussians than the source reconstruction — fine detail is reduced.",
            "Gaussian sizes are floored to the decimation spacing (rendering approximation).",
        ],
    }
    if extra:
        doc.update(extra)
    return doc


def index_json_bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, indent=2, sort_keys=False).encode("utf-8")
