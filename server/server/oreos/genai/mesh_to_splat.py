"""GLB (textured mesh) -> 3DGS splat ``.ply`` — the TRELLIS landing pad.

fal's ``fal-ai/trellis`` returns a **textured GLB mesh** (``model_mesh``,
``file_name: "model.glb"``, glTF 2.0 binary written by trimesh), *not* the
gaussian PLY our blind implementation assumed. Verified live 2026-07-31.

Everything downstream of the variants route — ``segment_geometry.read_ply_positions``
(all-float binary-LE PLY), ``segment_geometry.fit_asset_to_obb`` / ``quality_gate``,
``scene_report.splat_io.read_splat_ply``, and the web viewer's
``DemoSceneManager.addSplat`` (a Spark ``SplatMesh``) — speaks gaussian splat PLY
and nothing else. Rather than widen every one of those (and the viewer contract
in a sibling repo), we convert here: surface-sample the mesh, carry the texture
colour to each sample, and emit the exact INRIA layout
``server/scene_report/splat_io.py`` documents::

    x, y, z, nx, ny, nz,
    f_dc_0..2,                  # SH degree-0 DC term, from RGB
    opacity,                    # inverse-sigmoid (logit) of alpha
    scale_0..2,                 # LOG of the per-axis gaussian scale
    rot_0..3                    # quaternion (w, x, y, z)

The raw GLB is still stored alongside as ``asset.glb`` — the untouched generator
output is the provenance record, and the swap feature's ``GLTFLoader`` path can
consume it directly.

``trimesh`` is already a dependency (``requirements.txt`` and the
``modal_streaming.py`` broker image), so this adds no new install.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np

# SH degree-0 normalisation constant (INRIA 3DGS): f_dc = (rgb - 0.5) / C0.
SH_C0 = 0.28209479177387814

# Alpha the generated gaussians are written with, stored pre-sigmoid.
DEFAULT_ALPHA = 0.99

# Gaussian radius as a fraction of nearest-neighbour spacing. ~0.55 leaves the
# sampled surface just-overlapping — no moire holes, no bloat.
SPACING_TO_SCALE = 0.55

# Default sample budget. TRELLIS meshes are ~1.5k verts / 2.4k tris, so the
# vertices alone are far too sparse to read as a surface; we sample the faces.
DEFAULT_TARGET_POINTS = 200_000

SCALE_FLOOR = 1e-6

SPLAT_PROPERTIES = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


class MeshConversionError(RuntimeError):
    """The generator's artifact could not be turned into a splat."""


def _load_mesh(glb_bytes: bytes) -> Any:
    """Concatenated ``trimesh.Trimesh`` from GLB bytes (scenes are flattened,
    which also bakes each node's transform into the vertices)."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise MeshConversionError(f"trimesh is required to read the GLB artifact: {exc}") from exc

    try:
        loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", process=False)
    except Exception as exc:
        raise MeshConversionError(f"could not parse the GLB artifact: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise MeshConversionError("GLB contains no geometry")
        try:
            mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else loaded.dump(concatenate=True)
        except Exception:
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    if not hasattr(mesh, "faces") or len(getattr(mesh, "faces", ())) == 0:
        raise MeshConversionError("GLB has no triangle faces to sample")
    return mesh


# --- backdrop-plane stripping -------------------------------------------------
# TRELLIS reconstructs the conditioning photo's BACKGROUND as geometry: a
# full-footprint flat quad — the floor the object stands on, and/or the wall
# behind it. It is not an occasional artifact; it appeared on every in-situ
# evidence crop we measured (2026-07-31, det:5 'office chair'), unchanged by
# crop tightness, background compositing, sampling steps or seed.
#
# Left in place it dominates the asset's bounding box, and ``fit_asset_to_obb``
# (which matches asset axes to OBB axes by DESCENDING EXTENT) then aligns the
# BACKDROP instead of the object — the chair lands in the scene lying inside a
# vertical black slab.
#
# The quad has an unmistakable signature, identical across runs:
#
#     centre 0.003 of the bbox along its normal   (hard against one face)
#     thickness 0.0056                            (a zero-thickness sheet)
#     footprint span 0.998 x 0.997                (the entire footprint)
#     area 2.000 = 83% of the whole mesh          (majority of the surface)
#     +normal area 1.000, -normal area 1.000      (double-sided, coincident)
#
# Requiring ALL of those is what makes this safe on the flat objects these scans
# are full of: a whiteboard or poster is a genuine slab with thickness, is not
# double-sided-coincident, and leaves nothing behind if removed — so it never
# qualifies. The surviving-geometry guard is the final backstop, and every
# removal is recorded in the returned meta and surfaced as a caveat.
PLANE_NORMAL_DOT = 0.95  # |face normal . axis| to count as facing that axis
PLANE_MAX_THICKNESS = 0.02  # of the bbox extent along that axis
PLANE_MIN_AREA_FRAC = 0.40  # of total mesh area
PLANE_MIN_DOUBLE_SIDED = 0.30  # min(+n, -n) / band area
PLANE_EDGE_FRAC = 0.10  # centre must sit within this of a bbox face
BACKDROP_SPAN_FRAC = 0.90  # of the bbox, in the two non-normal axes
MIN_SURVIVING_AREA_FRAC = 0.05
MIN_SURVIVING_FACES = 64


def _strip_backdrop_planes(mesh: Any) -> tuple[Any, dict[str, Any]]:
    """Drop the generator's reconstructed floor/backdrop quad.

    Returns ``(mesh, info)``. The mesh is returned untouched when nothing matches
    the signature above or when removal would not leave a real object behind."""
    info: dict[str, Any] = {"removed_planes": 0, "removed_area_frac": 0.0}
    try:
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        centroids = np.asarray(mesh.triangles, dtype=np.float64).mean(axis=1)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
    except Exception:
        return mesh, info
    total = float(areas.sum())
    if total <= 0 or len(areas) < MIN_SURVIVING_FACES:
        return mesh, info
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1e-9)

    drop = np.zeros(len(areas), dtype=bool)
    found = []
    for axis in range(3):
        facing = np.abs(normals[:, axis]) > PLANE_NORMAL_DOT
        if not facing.any():
            continue
        idx = np.flatnonzero(facing)
        pos = (centroids[idx, axis] - lo[axis]) / span[axis]
        hist, edges = np.histogram(pos, bins=24, weights=areas[idx])
        peak = int(np.argmax(hist))
        in_peak = (pos >= edges[peak]) & (pos <= edges[peak + 1])
        if not in_peak.any():
            continue
        centre = float(np.average(pos[in_peak], weights=areas[idx][in_peak]))
        band = np.abs(pos - centre) <= PLANE_MAX_THICKNESS
        band_idx = idx[band]
        band_area = float(areas[band_idx].sum())
        if band_area / total < PLANE_MIN_AREA_FRAC:
            continue
        if not (centre <= PLANE_EDGE_FRAC or centre >= 1.0 - PLANE_EDGE_FRAC):
            continue  # slices through the middle -> part of the object
        up = float(areas[band_idx][normals[band_idx, axis] > 0].sum())
        down = float(areas[band_idx][normals[band_idx, axis] < 0].sum())
        if min(up, down) / band_area < PLANE_MIN_DOUBLE_SIDED:
            continue  # a real surface, not a coincident double-sided sheet
        others = [a for a in range(3) if a != axis]
        pts = centroids[band_idx][:, others]
        if not np.all((pts.max(axis=0) - pts.min(axis=0)) / span[others] >= BACKDROP_SPAN_FRAC):
            continue
        drop[band_idx] = True
        found.append({"axis": axis, "centre": round(centre, 4),
                      "area_frac": round(band_area / total, 4)})

    if not found or not (~drop).any():
        return mesh, info
    surviving_area = float(areas[~drop].sum())
    if (
        surviving_area / total < MIN_SURVIVING_AREA_FRAC
        or int((~drop).sum()) < MIN_SURVIVING_FACES
    ):
        info["kept_planar_as_object"] = True
        return mesh, info

    try:
        stripped = mesh.submesh([np.flatnonzero(~drop)], append=True, repair=False)
    except Exception:
        return mesh, info
    if not hasattr(stripped, "faces") or len(stripped.faces) < MIN_SURVIVING_FACES:
        return mesh, info
    info["removed_planes"] = len(found)
    info["removed_area_frac"] = round(1.0 - surviving_area / total, 4)
    info["planes"] = found
    return stripped, info


def _sample_colors(mesh: Any, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(points (N,3) float64, colors (N,3) uint8) sampled over the mesh surface.

    Colour comes from the GLB texture when trimesh can resolve it; a mesh with
    no usable visual falls back to neutral grey and the caller records a caveat."""
    import trimesh

    rng = np.random.default_rng(seed)
    try:
        trimesh.util.np.random.seed(seed)  # older trimesh paths use the global RNG
    except Exception:
        pass

    points: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None

    # Preferred: trimesh resolves UV -> texture per sample for us.
    try:
        sampled = trimesh.sample.sample_surface(mesh, count, sample_color=True)
        if len(sampled) >= 3:
            points = np.asarray(sampled[0], dtype=np.float64)
            colors = np.asarray(sampled[2])
    except Exception:
        points = None
        colors = None

    if points is None:
        sampled = trimesh.sample.sample_surface(mesh, count)
        points = np.asarray(sampled[0], dtype=np.float64)
        face_index = np.asarray(sampled[1], dtype=np.int64)
        colors = None
        # Fall back to per-vertex colour averaged over the sampled face.
        try:
            visual = mesh.visual.to_color()
            vertex_colors = np.asarray(visual.vertex_colors)
            if vertex_colors.size:
                tri = np.asarray(mesh.faces)[face_index]
                colors = vertex_colors[tri][:, :, :3].mean(axis=1)
        except Exception:
            colors = None

    if colors is None:
        colors = np.full((points.shape[0], 3), 170, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = np.repeat(colors[None, :], points.shape[0], axis=0)
        colors = colors[:, :3]
        if colors.dtype.kind == "f":
            # trimesh may hand back 0..1 floats; only rescale when it really is.
            colors = colors * 255.0 if float(np.nanmax(colors, initial=0.0)) <= 1.0 else colors
        colors = np.clip(np.nan_to_num(colors, nan=170.0), 0, 255).astype(np.uint8)

    del rng
    return points, colors


def glb_to_splat_ply(
    glb_bytes: bytes,
    *,
    target_points: int = DEFAULT_TARGET_POINTS,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[bytes, dict[str, Any]]:
    """Convert a GLB mesh to gaussian-splat PLY bytes.

    Returns ``(ply_bytes, meta)``. ``meta`` records what the conversion actually
    did — vertex/face counts, sample count, surface area, the derived gaussian
    scale, and any ``caveats`` — so the variants route can put honest numbers in
    the stored envelope instead of implying the splat was captured."""
    if not glb_bytes:
        raise MeshConversionError("empty GLB artifact")
    if glb_bytes[:4] != b"glTF":
        raise MeshConversionError(
            f"artifact is not a binary GLB (magic {glb_bytes[:4]!r}) — "
            "fal's TRELLIS returns glTF-binary"
        )

    mesh = _load_mesh(glb_bytes)
    caveats: list[str] = []

    raw_faces = int(len(mesh.faces))
    mesh, backdrop = _strip_backdrop_planes(mesh)
    if backdrop.get("removed_planes"):
        caveats.append(
            f"removed {backdrop['removed_planes']} full-footprint planar sheet(s) "
            f"({backdrop['removed_area_frac'] * 100:.0f}% of mesh area) — the generator's "
            "reconstruction of the conditioning photo's floor/backdrop, not the object"
        )

    n_faces = int(len(mesh.faces))
    n_vertices = int(len(mesh.vertices))
    count = max(1, int(target_points))

    points, colors = _sample_colors(mesh, count, seed)
    if points.size == 0:
        raise MeshConversionError("surface sampling produced no points")
    if int(np.unique(colors, axis=0).shape[0]) <= 1:
        caveats.append(
            "GLB texture could not be resolved per-sample — variation rendered in flat colour"
        )

    try:
        area = float(mesh.area)
    except Exception:
        area = 0.0
    if not np.isfinite(area) or area <= 0.0:
        # Degenerate/inverted mesh: fall back to bbox diagonal for the spacing.
        diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        area = max(diag * diag, 1e-9)
        caveats.append("mesh reported no surface area — gaussian scale estimated from bbox")

    spacing = float(np.sqrt(area / float(points.shape[0])))
    scale = max(spacing * SPACING_TO_SCALE, SCALE_FLOOR)

    n = int(points.shape[0])
    fields: dict[str, np.ndarray] = {
        "x": points[:, 0].astype(np.float32),
        "y": points[:, 1].astype(np.float32),
        "z": points[:, 2].astype(np.float32),
        "nx": np.zeros(n, dtype=np.float32),
        "ny": np.zeros(n, dtype=np.float32),
        "nz": np.zeros(n, dtype=np.float32),
        "opacity": np.full(n, _logit(alpha), dtype=np.float32),
        "scale_0": np.full(n, np.log(scale), dtype=np.float32),
        "scale_1": np.full(n, np.log(scale), dtype=np.float32),
        "scale_2": np.full(n, np.log(scale), dtype=np.float32),
        "rot_0": np.ones(n, dtype=np.float32),
        "rot_1": np.zeros(n, dtype=np.float32),
        "rot_2": np.zeros(n, dtype=np.float32),
        "rot_3": np.zeros(n, dtype=np.float32),
    }
    rgb = colors.astype(np.float32) / 255.0
    for i in range(3):
        fields[f"f_dc_{i}"] = ((rgb[:, i] - 0.5) / SH_C0).astype(np.float32)

    ply = serialize_splat_ply(fields)
    meta = {
        "source_format": "glb",
        "mesh_vertices": n_vertices,
        "mesh_faces": n_faces,
        "mesh_faces_raw": raw_faces,
        "backdrop": backdrop,
        "n_gaussians": n,
        "surface_area": round(area, 6),
        "sample_spacing": round(spacing, 8),
        "gaussian_scale": round(scale, 8),
        "alpha": alpha,
        "bbox_min": [round(float(v), 6) for v in points.min(axis=0)],
        "bbox_max": [round(float(v), 6) for v in points.max(axis=0)],
        "caveats": caveats,
    }
    return ply, meta


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def serialize_splat_ply(fields: dict[str, np.ndarray]) -> bytes:
    """Binary little-endian 3DGS PLY bytes from a field dict, in SPLAT_PROPERTIES
    order. Mirrors ``scene_report.splat_io.serialize_splat_ply`` but standalone so
    the demo genai package has no dependency on the product-workflow module."""
    names = [p for p in SPLAT_PROPERTIES if p in fields]
    extra = [k for k in fields if k not in SPLAT_PROPERTIES]
    names.extend(sorted(extra))
    if not names:
        raise MeshConversionError("no PLY properties to write")
    count = int(len(fields[names[0]]))
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    ).encode("ascii")
    table = np.empty((count, len(names)), dtype="<f4")
    for i, name in enumerate(names):
        column = np.asarray(fields[name], dtype="<f4").reshape(-1)
        if column.shape[0] != count:
            raise MeshConversionError(f"property {name!r} has {column.shape[0]} rows, expected {count}")
        table[:, i] = column
    return header + table.tobytes()
