"""Point-cloud (de)serialization helpers for the durable scan artifact.

Deliberately dependency-light (numpy only — no Open3D, cv2, or torch) so the
always-on CPU broker can downsample the stored cloud for display and stream a PLY
download without booting a GPU image. Shared by ``server/app.py``'s scene routes.
"""
from __future__ import annotations

import numpy as np


def downsample_cloud(positions, colors, max_points: int):
    """Uniformly subsample ``(positions, colors)`` to at most ``max_points`` for a
    snappy display payload (deterministic, seed 0). ``max_points <= 0`` keeps
    everything. Read-only — never mutates the stored full-fidelity artifact."""
    positions = np.asarray(positions)
    colors = np.asarray(colors)
    n = int(positions.shape[0])
    if max_points and n > int(max_points):
        sel = np.random.default_rng(0).choice(n, size=int(max_points), replace=False)
        sel.sort()
        return positions[sel], colors[sel]
    return positions, colors


def build_ply_bytes(positions, colors) -> bytes:
    """Serialize ``(positions (N,3) float32, colors (N,3) uint8)`` to a binary
    little-endian PLY. numpy packs the structured dtype to 15 bytes/vertex with no
    padding, so the body is exactly ``N * 15`` bytes following the ASCII header.
    Dependency-free (no Open3D) so it runs on the CPU broker."""
    positions = np.ascontiguousarray(positions, dtype="<f4").reshape(-1, 3)
    colors = np.ascontiguousarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = int(min(positions.shape[0], colors.shape[0]))
    positions, colors = positions[:n], colors[:n]
    vertex = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Open Reality scan export (world frame, up-to-scale)\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    return header + vertex.tobytes()


# Property name -> numpy dtype for the small binary-PLY schema this module reads back
# (exactly what ``build_ply_bytes`` emits, plus the common float-RGB variant). Kept tiny
# on purpose — this is a reader for OUR OWN derived-cloud artifacts, not a general PLY parser.
_PLY_DTYPE = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1"}


def read_ply_cloud(data: bytes):
    """Parse a binary-little-endian PLY (the exact shape ``build_ply_bytes`` writes) back into
    ``(positions (N,3) float32, colors (N,3) uint8)``. The inverse of ``build_ply_bytes`` — used
    to load a NON-DESTRUCTIVE ``derived/...`` calibrated cloud off the store for export/display.

    Only the vertex ``x/y/z`` + ``red/green/blue`` properties are required; any extra vertex
    properties are tolerated (parsed into the dtype, then ignored). Colors are read as ``uint8``
    (0-255); a float-typed RGB (0-1) variant is scaled up. Raises ``ValueError`` on a header this
    minimal reader doesn't understand — dependency-light (numpy only), like the writer."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("read_ply_cloud expects raw PLY bytes")
    marker = b"end_header\n"
    idx = data.find(marker)
    if idx < 0:
        raise ValueError("PLY has no end_header")
    header = data[:idx].decode("ascii", errors="replace").splitlines()
    body = data[idx + len(marker):]

    if not header or header[0].strip() != "ply":
        raise ValueError("not a PLY file")
    fmt = next((ln for ln in header if ln.startswith("format ")), "")
    if "binary_little_endian" not in fmt:
        raise ValueError(f"unsupported PLY format {fmt!r} (only binary_little_endian)")

    count = 0
    fields: list[tuple[str, str]] = []
    props_open = False
    for ln in header:
        parts = ln.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
            props_open = True
        elif props_open and parts and parts[0] == "element":
            props_open = False  # a later element block (e.g. faces) — stop collecting vertex props
        elif props_open and len(parts) == 3 and parts[0] == "property":
            ply_type, name = parts[1], parts[2]
            np_type = _PLY_DTYPE.get(ply_type)
            if np_type is None:
                raise ValueError(f"unsupported PLY property type {ply_type!r}")
            fields.append((name, np_type))

    names = {name for name, _ in fields}
    required = ("x", "y", "z", "red", "green", "blue")
    if not all(r in names for r in required):
        raise ValueError(f"PLY missing required vertex properties {required}")

    dtype = np.dtype([(name, np_type) for name, np_type in fields])
    if count == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    verts = np.frombuffer(body[: count * dtype.itemsize], dtype=dtype)
    positions = np.stack(
        [verts["x"], verts["y"], verts["z"]], axis=1
    ).astype(np.float32)
    color_fields = [verts["red"], verts["green"], verts["blue"]]
    if any(dtype[name].kind == "f" for name in ("red", "green", "blue")):
        colors = (np.stack(color_fields, axis=1) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        colors = np.stack(color_fields, axis=1).astype(np.uint8)
    return positions, colors
