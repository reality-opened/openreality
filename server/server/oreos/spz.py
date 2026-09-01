"""SPZ -> 3DGS PLY decoder (Niantic's compressed gaussian-splat container).

``.spz`` was deferred on Day 1 (design/shell.md §4b: "self-contained decoder
(~150 lines, public SPZ spec) ... Day-1 fallback scope = .ply-only") and the
route answered an honest 400. It is cheap now for one specific reason: the
renderer we already ship, ``@sparkjsdev/spark`` 0.1.10, contains a complete
**readable** ``SpzReader`` *and* ``SpzWriter`` in its dist bundle. This module is
a numpy port of that reader, so the format is taken from a working
implementation rather than from prose — and, more usefully, ``SpzWriter`` gives
us ground truth: ``tests/test_demo_spz.py`` has Spark encode a known splat and
asserts this decoder recovers it within quantization error.

Format (v1/v2/v3), all little-endian, whole file gzip-compressed::

    magic u32 = 0x4E475350 ("NGSP") | version u32 | numSplats u32
    shDegree u8 | fractionalBits u8 | flags u8 | reserved u8      (16-byte header)
    centers   v1: 3x float16     v2/v3: 3x int24 / (1 << fractionalBits)
    alpha     u8 -> alpha/255                       (PLY stores the LOGIT of this)
    rgb       u8 -> f_dc = (b/255 - 0.5) / 0.15
    scales    u8 -> LOG scale = b/16 - 10           (PLY also stores log scale)
    rotation  v1/v2: 3x u8 xyz, w reconstructed
              v3: 32-bit "smallest three" packing
    sh        u8 -> (b - 128)/128, degree-major per splat

Deliberately dependency-light (gzip + numpy) so the broker can decode without a
GPU, mirroring ``scene_report/splat_io.py``. Fully vectorized: no Python loop
runs per gaussian.
"""

from __future__ import annotations

import gzip
import struct
from typing import Any, Optional

import numpy as np

# 1347635022 == 0x5053474E; stored as a little-endian u32 the bytes read "NGSP".
SPZ_MAGIC = 0x5053474E
SPZ_HEADER_BYTES = 16
SPZ_MAX_VERSION = 3

# Spark's SH_DEGREE_TO_VECS: coefficient vectors carried per SH degree.
_SH_DEGREE_TO_VECS = {0: 0, 1: 3, 2: 8, 3: 15}

# rgb byte -> f_dc. Spark decodes to colour space as
# ``(b/255 - 0.5) * (SH_C0/0.15) + 0.5`` and 3DGS colour is ``0.5 + SH_C0*f_dc``,
# so the SH_C0 cancels: f_dc = (b/255 - 0.5) / 0.15.
_RGB_DENORM = 0.15

# Opacity in a 3DGS PLY is the inverse-sigmoid (logit) of alpha; SPZ stores alpha
# directly. Clamp before the logit so alpha 0 / 255 don't produce +-inf.
_ALPHA_EPS = 1.0 / 512.0

# The 17-property degree-0 3DGS schema, in the canonical export order that
# ``splat_io`` documents. Higher SH degrees append f_rest_* after f_dc_*.
_BASE_FIELDS = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
)
_TAIL_FIELDS = (
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


class SpzError(ValueError):
    """Malformed / unsupported SPZ. The ingest layer turns this into an honest
    422 with the message attached."""


class SpzHeader:
    __slots__ = ("version", "num_splats", "sh_degree", "fractional_bits", "flags",
                 "antialiased")

    def __init__(self, version: int, num_splats: int, sh_degree: int,
                 fractional_bits: int, flags: int) -> None:
        self.version = version
        self.num_splats = num_splats
        self.sh_degree = sh_degree
        self.fractional_bits = fractional_bits
        self.flags = flags
        self.antialiased = bool(flags & 1)


def _parse_header(raw: bytes) -> SpzHeader:
    if len(raw) < SPZ_HEADER_BYTES:
        raise SpzError("file is shorter than the 16-byte SPZ header")
    magic, version, num_splats = struct.unpack_from("<III", raw, 0)
    sh_degree, fractional_bits, flags, _reserved = struct.unpack_from("<BBBB", raw, 12)
    if magic != SPZ_MAGIC:
        raise SpzError("not an SPZ file (bad magic — is it a raw .ply renamed to .spz?)")
    if version < 1 or version > SPZ_MAX_VERSION:
        raise SpzError(
            f"unsupported SPZ version {version} (this build reads versions 1-{SPZ_MAX_VERSION})"
        )
    if sh_degree not in _SH_DEGREE_TO_VECS:
        raise SpzError(f"unsupported SH degree {sh_degree} (0-3 supported)")
    if num_splats <= 0:
        raise SpzError("the SPZ header declares zero gaussians — nothing to import")
    return SpzHeader(version, num_splats, sh_degree, fractional_bits, flags)


def read_spz_header(path: str) -> SpzHeader:
    """Header only — decompresses just the first block, so a huge .spz can be
    gaussian-capped before anything is decoded."""
    try:
        with gzip.open(path, "rb") as fh:
            raw = fh.read(SPZ_HEADER_BYTES)
    except OSError as exc:  # not gzip, truncated, etc.
        raise SpzError(f"could not read the SPZ container: {exc}") from exc
    return _parse_header(raw)


def _int24(buf: np.ndarray, count: int) -> np.ndarray:
    """3-byte little-endian SIGNED integers -> int32. Spark does this with
    ``(b2<<24 | b1<<16 | b0<<8) >> 8``; vectorized here as an unsigned assemble
    plus a sign fold."""
    triples = buf[: count * 3].reshape(count, 3).astype(np.int32)
    value = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    return np.where(value >= (1 << 23), value - (1 << 24), value)


def _decode_quaternions(buf: np.ndarray, header: SpzHeader) -> np.ndarray:
    """-> (N, 4) float32 in **xyzw** order (Spark's callback order)."""
    n = header.num_splats
    if header.version < 3:
        xyz = buf[: n * 3].reshape(n, 3).astype(np.float32) / 127.5 - 1.0
        w = np.sqrt(np.maximum(0.0, 1.0 - (xyz ** 2).sum(axis=1)))
        return np.concatenate([xyz, w[:, None]], axis=1).astype(np.float32)

    # v3 "smallest three": 32 bits = [largestIndex:2][c:10][b:10][a:10], each
    # 10-bit group = [sign:1][magnitude:9], filled for i = 3,2,1,0 skipping the
    # largest component.
    packed = buf[: n * 4].reshape(n, 4).astype(np.uint32)
    combined = (packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
                | (packed[:, 3] << 24))
    value_mask = (1 << 9) - 1
    max_value = float(1.0 / np.sqrt(2.0))
    quat = np.zeros((n, 4), dtype=np.float32)
    largest = (combined >> 30).astype(np.int64)
    remaining = combined.copy()
    sum_squares = np.zeros(n, dtype=np.float64)
    rows = np.arange(n)
    for i in range(3, -1, -1):
        active = largest != i
        value = (remaining & value_mask).astype(np.float64)
        sign = ((remaining >> 9) & 1).astype(bool)
        component = max_value * (value / value_mask)
        component = np.where(sign, -component, component)
        quat[rows[active], i] = component[active].astype(np.float32)
        sum_squares += np.where(active, component ** 2, 0.0)
        # Only the three non-largest components consume 10 bits each.
        remaining = np.where(active, remaining >> 10, remaining)
    quat[rows, largest] = np.sqrt(np.maximum(1.0 - sum_squares, 0.0)).astype(np.float32)
    return quat


def decode_spz(path: str, max_gaussians: Optional[int] = None) -> dict[str, np.ndarray]:
    """Decode one ``.spz`` into the field dict ``splat_io.serialize_splat_ply``
    consumes (``{property_name: (N,) float32}``, header order preserved).

    ``max_gaussians`` (checked against the header, before any decode) lets the
    caller enforce the import cap without paying for the decompression."""
    header = read_spz_header(path)
    if max_gaussians is not None and header.num_splats > int(max_gaussians):
        raise SpzError(
            f"{header.num_splats:,} gaussians is over the {int(max_gaussians):,} "
            "import limit"
        )
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < SPZ_HEADER_BYTES:
        raise SpzError("SPZ payload is truncated")
    body = np.frombuffer(blob, dtype=np.uint8, offset=SPZ_HEADER_BYTES)

    n = header.num_splats
    sh_vecs = _SH_DEGREE_TO_VECS[header.sh_degree]
    center_bytes = n * 3 * (2 if header.version == 1 else 3)
    quat_bytes = n * (4 if header.version == 3 else 3)
    expected = center_bytes + n + (n * 3) + (n * 3) + quat_bytes + (n * sh_vecs * 3)
    if body.size < expected:
        raise SpzError(
            f"SPZ payload is truncated: {n:,} gaussians need {expected:,} bytes of "
            f"body but only {body.size:,} arrived"
        )

    offset = 0
    if header.version == 1:
        centers = (np.frombuffer(body[:center_bytes].tobytes(), dtype="<f2")
                   .astype(np.float32).reshape(n, 3))
    else:
        scale = float(1 << header.fractional_bits)
        centers = (_int24(body[:center_bytes], n * 3).astype(np.float32) / scale
                   ).reshape(n, 3)
    offset += center_bytes

    alpha = body[offset:offset + n].astype(np.float32) / 255.0
    offset += n
    rgb = body[offset:offset + n * 3].reshape(n, 3).astype(np.float32)
    offset += n * 3
    scales = body[offset:offset + n * 3].reshape(n, 3).astype(np.float32)
    offset += n * 3
    quats = _decode_quaternions(body[offset:offset + quat_bytes], header)
    offset += quat_bytes

    fields: dict[str, np.ndarray] = {}
    fields["x"] = np.ascontiguousarray(centers[:, 0])
    fields["y"] = np.ascontiguousarray(centers[:, 1])
    fields["z"] = np.ascontiguousarray(centers[:, 2])
    # SPZ carries no normals; the 3DGS schema declares them and every renderer
    # ignores them, so emit explicit zeros rather than dropping the properties.
    for name in ("nx", "ny", "nz"):
        fields[name] = np.zeros(n, dtype=np.float32)
    for i in range(3):
        fields[f"f_dc_{i}"] = np.ascontiguousarray(
            (rgb[:, i] / 255.0 - 0.5) / _RGB_DENORM
        )

    if header.sh_degree >= 1:
        sh_flat = (body[offset:offset + n * sh_vecs * 3].reshape(n, sh_vecs * 3)
                   .astype(np.float32) - 128.0) / 128.0
        # PLY orders f_rest_* channel-major (all R coefficients, then G, then B)
        # while SPZ stores them coefficient-major (rgb per coefficient).
        per_channel = sh_flat.reshape(n, sh_vecs, 3)
        for channel in range(3):
            for coeff in range(sh_vecs):
                fields[f"f_rest_{channel * sh_vecs + coeff}"] = np.ascontiguousarray(
                    per_channel[:, coeff, channel]
                )

    alpha_clamped = np.clip(alpha, _ALPHA_EPS, 1.0 - _ALPHA_EPS)
    fields["opacity"] = np.log(alpha_clamped / (1.0 - alpha_clamped)).astype(np.float32)
    for i in range(3):
        fields[f"scale_{i}"] = np.ascontiguousarray(scales[:, i] / 16.0 - 10.0)
    # PLY rot order is (w, x, y, z); SPZ/Spark hand back (x, y, z, w).
    fields["rot_0"] = np.ascontiguousarray(quats[:, 3])
    fields["rot_1"] = np.ascontiguousarray(quats[:, 0])
    fields["rot_2"] = np.ascontiguousarray(quats[:, 1])
    fields["rot_3"] = np.ascontiguousarray(quats[:, 2])

    # Re-key into canonical PLY property order.
    ordered: dict[str, np.ndarray] = {}
    for name in _BASE_FIELDS:
        ordered[name] = fields[name]
    for key in sorted((k for k in fields if k.startswith("f_rest_")),
                      key=lambda k: int(k.split("_")[-1])):
        ordered[key] = fields[key]
    for name in _TAIL_FIELDS:
        ordered[name] = fields[name]
    return ordered


def spz_to_ply_file(spz_path: str, ply_path: str,
                    max_gaussians: Optional[int] = None) -> dict[str, Any]:
    """Decode ``spz_path`` and write a binary-LE 3DGS ``.ply`` at ``ply_path``.
    Returns ``{gaussian_count, sh_degree, version}``."""
    from server.scene_report.splat_io import serialize_splat_ply

    header = read_spz_header(spz_path)
    fields = decode_spz(spz_path, max_gaussians=max_gaussians)
    with open(ply_path, "wb") as fh:
        fh.write(serialize_splat_ply(fields))
    return {
        "gaussian_count": int(header.num_splats),
        "sh_degree": int(header.sh_degree),
        "version": int(header.version),
    }
