"""3DGS ``splat.ply`` (de)serialization + scale-clamp helpers for the product-workflow
API (``server/app.py``'s ``/api/scenes/<id>/anchor`` and ``/api/scenes/<id>/clamp``).

Deliberately dependency-light (numpy only — no plyfile/Open3D) so the always-on
CPU broker can read/rewrite a persisted splat with no GPU, mirroring
``server/scene_report/cloud_io.py``'s ``cloud.ply`` helpers.

SELF-CONTAINED BY DESIGN: this module reimplements the tiny bit of math from
``core``'s (``reality-opened/core``) unmerged splat-export work — it does not, and
must not, import ``vggt_slam`` at runtime (the deployed ``server`` pins a released
``core`` tag that predates this code; the anchor/sim3-anchor-graph branches are
unmerged). The PLY schema and ``clamp_scales_percentile`` below are a byte-for-byte
port of ``vggt_slam.splat_export`` (read as a spec, not imported) so a splat
exported by that pipeline round-trips through this module untouched:

    x, y, z, nx, ny, nz,
    f_dc_0, f_dc_1, f_dc_2,        # SH degree-0 DC term, from RGB
    opacity,                        # inverse-sigmoid (logit) of alpha
    scale_0, scale_1, scale_2,      # LOG of the per-axis gaussian scale
    rot_0, rot_1, rot_2, rot_3      # quaternion (w, x, y, z)

Everything here is generic over the property list actually present in a file's
header, so it also round-trips a splat with extra/missing SH-degree properties —
it just reads/writes whatever the header says, in order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Per-axis gaussian scale properties, in the order the exporter writes them.
SCALE_FIELDS = ("scale_0", "scale_1", "scale_2")

# Avoid log(0) when a clamped/degenerate scale collapses to zero.
SCALE_FLOOR = 1e-9

# Default export-time high-percentile clamp on per-gaussian scales (matches
# ``vggt_slam.splat_export.DEFAULT_SCALE_CLAMP_PERCENTILE``): the render "spike
# catastrophe" is entirely the top-1% per-gaussian scale tail; pulling every axis
# above the p99 scale value down to that value removes the spikes with no
# re-export, near-no-op on an already-tame splat.
DEFAULT_SCALE_CLAMP_PERCENTILE = 99.0


def read_splat_ply(path: str) -> dict[str, np.ndarray]:
    """Read a binary little-endian 3DGS ``.ply``. Returns a dict of ``(N,)`` float32
    arrays keyed by PLY property name, in header order — generic over whatever
    properties the file actually declares (no assumption about the 17-field schema
    above beyond "some named float properties")."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = f.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"unsupported PLY format: {fmt!r}")
        count = None
        names: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            line = line.strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            elif line.startswith(b"property"):
                names.append(line.split()[-1].decode("ascii"))
            elif line == b"end_header":
                break
        if count is None:
            raise ValueError("PLY header missing 'element vertex <N>'")
        dtype = np.dtype([(name, "<f4") for name in names])
        data = np.fromfile(f, dtype=dtype, count=count)
    return {name: data[name].copy() for name in names}


def serialize_splat_ply(fields: dict[str, np.ndarray]) -> bytes:
    """Serialize a raw field dict (as returned by :func:`read_splat_ply`, optionally
    modified) to binary little-endian 3DGS ``.ply`` bytes, preserving property order
    (dict insertion order) and vertex count. Every value must be a ``(N,)``-length
    array with the SAME ``N`` across fields. Pure/in-memory -- no disk I/O, so a route
    handler can hand the result straight to ``send_file(io.BytesIO(...))`` or a
    persistence ``save_derived_artifact`` call without a temp-file round trip."""
    names = list(fields.keys())
    n = int(np.asarray(fields[names[0]]).reshape(-1).shape[0]) if names else 0
    dtype = np.dtype([(name, "<f4") for name in names])
    verts = np.empty(n, dtype=dtype)
    for name in names:
        arr = np.asarray(fields[name], dtype="<f4").reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"field {name!r} has length {arr.shape[0]}, expected {n}")
        verts[name] = arr

    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(n)
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"
    return header.encode("ascii") + verts.tobytes(order="C")


def write_splat_ply(path: str, fields: dict[str, np.ndarray]) -> int:
    """:func:`serialize_splat_ply`, written to ``path``. Returns the vertex count."""
    data = serialize_splat_ply(fields)
    with open(path, "wb") as f:
        f.write(data)
    n = int(np.asarray(fields[next(iter(fields))]).reshape(-1).shape[0]) if fields else 0
    return n


def clamp_scales_percentile(
    scales: Any, percentile: float = DEFAULT_SCALE_CLAMP_PERCENTILE
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp per-axis gaussian scales down to a high global percentile of the tail.

    Exact port of ``vggt_slam.splat_export.clamp_scales_percentile`` (core
    ``feat/sim3-anchor-graph``, read as a spec — see platform ``CLAUDE.md``). Pulling
    every axis above the ``percentile``-th value of the (flattened, finite) scale
    distribution down to that value removes render spikes with no re-export; a
    near-no-op on an already-tame splat.

    Args:
        scales: ``(N, 3)`` or ``(N,)`` LINEAR (not log) per-axis scales.
        percentile: clamp threshold in ``(0, 100)``. ``None`` or ``>= 100`` (or a
            degenerate/non-finite distribution) leaves the scales untouched.

    Returns:
        ``(clamped scales (same shape) float32, info dict)`` where ``info`` carries
        ``percentile``, ``clamp_value``, and ``num_clamped`` (gaussians with any axis
        pulled down). Empty input passes through unchanged. Never raises.
    """
    scales = np.asarray(scales, dtype=np.float32)
    info: dict[str, Any] = {
        "percentile": None if percentile is None else float(percentile),
        "clamp_value": None,
        "num_clamped": 0,
    }
    if scales.size == 0 or percentile is None or percentile >= 100.0 or percentile <= 0.0:
        return scales, info
    finite = scales[np.isfinite(scales)]
    if finite.size == 0:
        return scales, info
    clamp_value = float(np.percentile(finite, percentile))
    if not np.isfinite(clamp_value) or clamp_value <= 0.0:
        return scales, info
    over = scales > clamp_value
    info["num_clamped"] = int(np.count_nonzero(over.any(axis=1) if scales.ndim == 2 else over))
    info["clamp_value"] = clamp_value
    return np.minimum(scales, clamp_value).astype(np.float32), info


def clamp_splat_fields(
    fields: dict[str, np.ndarray], percentile: float = DEFAULT_SCALE_CLAMP_PERCENTILE
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply :func:`clamp_scales_percentile` to a splat field dict's ``scale_0..2``
    (stored LOG per the exporter's convention — see module docstring), returning a
    NEW field dict (input is never mutated) plus an info dict carrying
    ``num_clamped``, ``clamp_percentile``, ``max_scale_before``, ``max_scale_after``
    (both in LINEAR units — the physically meaningful gaussian size, for a
    human-readable before/after signal).
    """
    missing = [name for name in SCALE_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"splat is missing expected scale field(s): {missing}")

    scale_log = np.stack([np.asarray(fields[name], dtype=np.float32) for name in SCALE_FIELDS], axis=1)
    n = int(scale_log.shape[0])
    scale_linear = np.exp(scale_log)
    max_before = float(np.max(scale_linear)) if n else 0.0

    clamped_linear, clamp_info = clamp_scales_percentile(scale_linear, percentile)
    clamped_log = np.log(np.maximum(clamped_linear, SCALE_FLOOR)).astype(np.float32)
    max_after = float(np.max(clamped_linear)) if n else 0.0

    out = dict(fields)  # preserve every other property + insertion order untouched
    for i, name in enumerate(SCALE_FIELDS):
        out[name] = clamped_log[:, i]

    info = {
        "gaussian_count": n,
        "clamped_gaussian_count": clamp_info["num_clamped"],
        "scale_clamp_percentile": clamp_info["percentile"],
        "clamp_value_linear": clamp_info["clamp_value"],
        "max_scale_before": max_before,
        "max_scale_after": max_after,
    }
    return out, info


def scale_splat_fields(fields: dict[str, np.ndarray], scale_factor: float) -> dict[str, np.ndarray]:
    """Apply a uniform metric-anchor gauge ``scale_factor`` to a splat field dict:
    multiplies gaussian CENTERS (``x, y, z``) by ``scale_factor`` and gaussian SIZES
    (LOG ``scale_0..2``) by the same factor (``log(s * k) == log(s) + log(k)``) so
    every gaussian's physical footprint stays consistent with its rescaled position.
    Returns a NEW field dict; input is never mutated."""
    if not (np.isfinite(scale_factor) and scale_factor > 0):
        raise ValueError(f"scale_factor must be a positive finite number, got {scale_factor!r}")
    out = dict(fields)
    log_k = float(np.log(scale_factor))
    for axis in ("x", "y", "z"):
        if axis in out:
            out[axis] = (np.asarray(out[axis], dtype=np.float32) * float(scale_factor)).astype(np.float32)
    for name in SCALE_FIELDS:
        if name in out:
            out[name] = (np.asarray(out[name], dtype=np.float32) + log_k).astype(np.float32)
    return out
