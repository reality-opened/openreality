"""3D Gaussian Splatting (3DGS) export for VGGT-SLAM.

This module turns the SLAM result (a world-frame colored point cloud) into a
standard 3DGS ``splat.ply`` that opens in any 3DGS viewer.

Two levels are provided:

* **Baseline** (default, deterministic, CPU-only, no training): one gaussian per
  world point — :func:`point_cloud_to_splat_ply` / :func:`export_splat`. This is
  the robust path the real-estate pilot relies on; it never needs a GPU or any
  optional dependency.
* **Optional refinement** (guarded, GPU): photometric optimization with the
  `gsplat <https://github.com/nerfstudio-project/gsplat>`_ library, seeded by the
  baseline init and supervised by the keyframe images + their world poses —
  :func:`refine_with_gsplat`. Everything GPU/gsplat related is imported lazily and
  wrapped in ``try/except`` so importing this module on a CPU / no-gsplat box (and
  the whole baseline path) works regardless.

PLY layout (binary little-endian, all properties ``float32``), the de-facto 3DGS
vertex schema::

    x, y, z, nx, ny, nz,
    f_dc_0, f_dc_1, f_dc_2,        # SH degree-0 DC term, from RGB
    opacity,                        # inverse-sigmoid (logit) of alpha
    scale_0, scale_1, scale_2,      # log of the per-axis gaussian scale
    rot_0, rot_1, rot_2, rot_3      # quaternion (w, x, y, z)

Only numpy + the stdlib are imported at module load. ``scipy`` (nearest-neighbour
spacing), ``torch`` and ``gsplat`` are imported lazily where used.

See ``docs/splat-export.md``.
"""

import numpy as np

# 0-th spherical-harmonic basis coefficient. 3DGS stores view-independent color as
# the SH DC term f_dc = (rgb - 0.5) / C0 ; viewers invert it as rgb = f_dc * C0 + 0.5.
SH_C0 = 0.28209479177387814

# Default per-gaussian opacity (alpha) for the baseline init. ~0.9 keeps points
# visibly opaque without being fully saturated.
DEFAULT_OPACITY = 0.9

# Ordered 3DGS vertex property names. The serializer and reader both key off this.
SPLAT_PLY_FIELDS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

_SCALE_FLOOR = 1e-9  # avoid log(0) when a scale collapses to zero

# Colors above this are treated as 0-255 and divided by 255. A margin above 1.0 keeps a
# normalized float cloud with a stray slightly-over-1 value from being crushed to near-black.
_COLOR_255_THRESHOLD = 1.5

# Export-time high-percentile clamp on per-gaussian scales. EXP-7 (2026-07-05,
# experiments/research/2026-07-05-exp7-b3seg-probe.md) proved the refined/composed
# splat's render "spike catastrophe" is *entirely* the top-1% per-gaussian scale tail:
# pulling every axis above the p99 scale value down to that value removes the spikes with
# no re-export, while being a near-no-op on the already-tame isotropic baseline scales
# (whose tail sits right at the k-NN-capped max). Default ON at p99; disable per call with
# ``clamp_scales=False`` and tune with ``scale_clamp_percentile``.
DEFAULT_SCALE_CLAMP_PERCENTILE = 99.0


# --------------------------------------------------------------------------- #
# field math (pure numpy, deterministic)
# --------------------------------------------------------------------------- #
def rgb_to_sh_dc(rgb):
    """RGB in ``[0, 1]`` -> SH degree-0 DC coefficient ``(rgb - 0.5) / C0``."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(f_dc):
    """Inverse of :func:`rgb_to_sh_dc` (handy for round-trip checks)."""
    f_dc = np.asarray(f_dc, dtype=np.float32)
    return f_dc * SH_C0 + 0.5


def inverse_sigmoid(x):
    """Logit ``log(x / (1 - x))`` with clamping so ``x in {0, 1}`` is safe."""
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _global_spacing(positions):
    """A sane fallback point spacing: bbox diagonal / cube-root(N) (uniform-grid
    heuristic). Always returns a strictly positive finite float."""
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    if n == 0:
        return 1e-3
    finite = positions[np.isfinite(positions).all(axis=1)]
    if finite.shape[0] == 0:
        return 1e-3
    diag = float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0.0:
        return 1e-3
    return diag / max(n ** (1.0 / 3.0), 1.0)


def estimate_point_scales(positions, k=3, scale_factor=1.0, floor=_SCALE_FLOOR):
    """Per-point isotropic gaussian scale (linear, **not** log) from local spacing.

    Uses the mean distance to the ``k`` nearest neighbours via ``scipy.cKDTree``;
    falls back to a bbox-derived global spacing if scipy is unavailable or a point
    is degenerate. Returns ``(N,)`` float32. Never raises on a CPU box.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    global_spacing = _global_spacing(positions)

    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.full((1,), max(global_spacing * scale_factor, floor), dtype=np.float32)

    kk = int(min(k, n - 1))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=kk + 1)  # column 0 is the point itself
        dists = np.atleast_2d(dists)
        if dists.shape[0] == 1 and n != 1:
            dists = dists.T
        per_point = dists[:, 1:].mean(axis=1)
    except Exception:
        per_point = np.full((n,), global_spacing)

    per_point = np.where(np.isfinite(per_point) & (per_point > 0.0), per_point, global_spacing)
    scales = np.maximum(per_point * float(scale_factor), floor)
    return scales.astype(np.float32)


def clamp_scales_percentile(scales, percentile=DEFAULT_SCALE_CLAMP_PERCENTILE):
    """Clamp per-axis gaussian scales down to a high global percentile of the tail.

    The refined/composed splat's render spikes are entirely the top ``(100 -
    percentile)%`` per-gaussian scale tail (EXP-7); clamping every axis above the
    ``percentile``-th value of the (flattened, finite) scale distribution down to that
    value removes the spikes render-side with no re-export. A near-no-op on the tame
    isotropic baseline, where the tail already sits at the k-NN-capped max.

    Args:
        scales: ``(N, 3)`` or ``(N,)`` linear (**not** log) per-axis scales.
        percentile: clamp threshold in ``(0, 100)``. ``None`` or ``>= 100`` (or a
            degenerate / non-finite distribution) leaves the scales untouched.

    Returns:
        ``(clamped scales (same shape) float32, info dict)`` where ``info`` carries
        ``percentile``, ``clamp_value`` and ``num_clamped`` (gaussians with any axis
        pulled down) for logging / before-after spike counts. Empty input passes
        through unchanged. Never raises.
    """
    scales = np.asarray(scales, dtype=np.float32)
    info = {"percentile": None if percentile is None else float(percentile),
            "clamp_value": None, "num_clamped": 0}
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


# --------------------------------------------------------------------------- #
# PLY serialization
# --------------------------------------------------------------------------- #
def _splat_dtype():
    return np.dtype([(name, "<f4") for name in SPLAT_PLY_FIELDS])


def write_splat_ply(
    path,
    positions,
    colors,
    scales=None,
    opacities=DEFAULT_OPACITY,
    quats=None,
    normals=None,
    scale_factor=1.0,
    clamp_scales=True,
    scale_clamp_percentile=DEFAULT_SCALE_CLAMP_PERCENTILE,
):
    """Serialize gaussians to a binary little-endian 3DGS ``.ply``.

    Args:
        path: output ``.ply`` path.
        positions: ``(N, 3)`` gaussian centers.
        colors: ``(N, 3)`` RGB. Accepts ``[0, 1]`` floats or ``[0, 255]`` (auto
            normalized). Stored as the SH DC term.
        scales: per-axis gaussian scale (**linear**), broadcast from ``(N,)`` or
            ``(N, 3)``. If ``None``, estimated from local point spacing. Stored as
            ``log(scale)``.
        opacities: alpha in ``[0, 1]`` — scalar or ``(N,)``. Stored as logit.
        quats: ``(N, 4)`` rotation quaternions ``(w, x, y, z)``; normalized before
            writing. Defaults to identity ``(1, 0, 0, 0)``.
        normals: ``(N, 3)`` normals; default zeros (3DGS ignores them).
        scale_factor: multiplier applied only to *estimated* scales.
        clamp_scales: if ``True`` (default), clamp the per-axis scale tail down to
            ``scale_clamp_percentile`` before storing (:func:`clamp_scales_percentile`)
            — removes the refined-splat render spikes; near-no-op on baseline scales.
            Pass ``False`` to store scales exactly as given.
        scale_clamp_percentile: percentile for that clamp (default p99).

    Returns:
        The number of gaussians written (after dropping non-finite rows; may be 0).

    Non-finite rows are dropped: SLAM can emit NaN/Inf positions, colors, scales, or
    quaternions, and writing those into the PLY yields a splat that crashes or
    destabilizes viewers. Any row whose position/color (or a provided per-row
    scale/quat/opacity/normal) is not finite is filtered out before serialization.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = positions.shape[0]

    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if colors.shape[0] != n:
        raise ValueError(f"positions/colors length mismatch: {n} vs {colors.shape[0]}")

    # Coerce any PER-ROW optional inputs up front so they can be filtered by the same
    # finite mask as positions/colors (scalar/None forms are handled after filtering).
    scales_rows = None
    if scales is not None:
        s = np.asarray(scales, dtype=np.float32)
        if s.ndim == 1 and s.shape[0] == n:
            scales_rows = np.repeat(s.reshape(-1, 1), 3, axis=1)
        elif s.ndim == 2 and s.shape == (n, 3):
            scales_rows = s
    quats_rows = None
    if quats is not None:
        q = np.asarray(quats, dtype=np.float32).reshape(-1, 4)
        if q.shape[0] == n:
            quats_rows = q
    normals_rows = None
    if normals is not None:
        nm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if nm.shape[0] == n:
            normals_rows = nm
    opac_rows = None
    opac_arr = np.asarray(opacities, dtype=np.float64)
    if opac_arr.ndim >= 1 and opac_arr.reshape(-1).shape[0] == n:
        opac_rows = opac_arr.reshape(-1)

    # Drop non-finite rows (NaN/Inf) across every per-row array we have.
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(colors).all(axis=1)
    if scales_rows is not None:
        finite &= np.isfinite(scales_rows).all(axis=1)
    if quats_rows is not None:
        finite &= np.isfinite(quats_rows).all(axis=1)
    if normals_rows is not None:
        finite &= np.isfinite(normals_rows).all(axis=1)
    if opac_rows is not None:
        finite &= np.isfinite(opac_rows)
    if not bool(finite.all()):
        positions = positions[finite]
        colors = colors[finite]
        if scales_rows is not None:
            scales_rows = scales_rows[finite]
        if quats_rows is not None:
            quats_rows = quats_rows[finite]
        if normals_rows is not None:
            normals_rows = normals_rows[finite]
        if opac_rows is not None:
            opac_rows = opac_rows[finite]
        n = positions.shape[0]

    # Normalize colors AFTER filtering (so an Inf can't skew the 0-255 detection, and the
    # clip can't rescue a non-finite row into the output). Margin above 1.0 keeps a stray
    # ~1.001 from being crushed to near-black; clip clamps out-of-gamut finite values.
    if colors.size and float(colors.max()) > _COLOR_255_THRESHOLD:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    # Scales: per-row (filtered) if given, else broadcast scalar, else estimated from the
    # (now finite) point spacing.
    if scales_rows is not None:
        scales = scales_rows
    elif scales is not None and np.asarray(scales).ndim == 0:
        scales = np.full((n, 3), float(scales), dtype=np.float32)
    else:
        scales = estimate_point_scales(positions, scale_factor=scale_factor)
        scales = np.repeat(np.asarray(scales, dtype=np.float32).reshape(-1, 1), 3, axis=1)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (n, 3):
        raise ValueError(f"scales must broadcast to ({n}, 3), got {scales.shape}")

    # Clamp the per-gaussian scale spike tail (EXP-7) before storing. Additive/guarded:
    # ON by default, a near-no-op on the tame baseline; the refined path is where the
    # top-1% tail produces render spikes.
    if clamp_scales and n > 0:
        scales, _clamp_info = clamp_scales_percentile(scales, scale_clamp_percentile)
        if _clamp_info["num_clamped"]:
            print(
                f"[splat-export] scale clamp p{_clamp_info['percentile']:g} -> "
                f"{_clamp_info['clamp_value']:.5f}: pulled down {_clamp_info['num_clamped']} "
                f"of {n} gaussians (spike tail)"
            )

    if normals_rows is not None:
        normals = normals_rows
    else:
        normals = np.zeros((n, 3), dtype=np.float32)

    if quats_rows is not None:
        norm = np.linalg.norm(quats_rows, axis=1, keepdims=True)
        norm = np.where(norm > 0.0, norm, 1.0)
        quats = quats_rows / norm
    else:
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # identity (w, x, y, z)

    if opac_rows is not None:
        opacity_logit = inverse_sigmoid(opac_rows).astype(np.float32)
    else:
        opacity_logit = np.full((n,), float(inverse_sigmoid(opac_arr)), dtype=np.float32)

    f_dc = rgb_to_sh_dc(colors)
    scale_log = np.log(np.maximum(scales, _SCALE_FLOOR)).astype(np.float32)

    verts = np.empty((n,), dtype=_splat_dtype())
    verts["x"], verts["y"], verts["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    verts["opacity"] = opacity_logit
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = scale_log[:, 0], scale_log[:, 1], scale_log[:, 2]
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    )

    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(n)
    header += "".join("property float {}\n".format(name) for name in SPLAT_PLY_FIELDS)
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes(order="C"))

    return n


def read_splat_ply(path):
    """Read a binary little-endian 3DGS ``.ply`` written by :func:`write_splat_ply`.

    Returns a dict of ``(N,)`` float32 arrays keyed by the property names. Intended
    for round-trip tests and debugging (assumes the field layout above).
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt = f.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"unsupported PLY format: {fmt!r}")
        count = None
        names = []
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
        dtype = np.dtype([(name, "<f4") for name in names])
        data = np.fromfile(f, dtype=dtype, count=count)
    return {name: data[name].copy() for name in names}


def point_cloud_to_splat_ply(positions, colors, out_path, opacity=DEFAULT_OPACITY, scale_factor=1.0,
                             clamp_scales=True, scale_clamp_percentile=DEFAULT_SCALE_CLAMP_PERCENTILE):
    """Baseline conversion: world-frame colored point cloud -> 3DGS ``.ply``.

    Deterministic, CPU-only, one gaussian per point (estimated isotropic scale,
    identity rotation, default opacity). Returns the gaussian count. ``clamp_scales`` /
    ``scale_clamp_percentile`` forwarded to :func:`write_splat_ply` (p99 scale-tail
    clamp, default ON).
    """
    return write_splat_ply(
        out_path,
        positions,
        colors,
        scales=None,
        opacities=opacity,
        quats=None,
        normals=None,
        scale_factor=scale_factor,
        clamp_scales=clamp_scales,
        scale_clamp_percentile=scale_clamp_percentile,
    )


# --------------------------------------------------------------------------- #
# pulling the world cloud / posed keyframes out of the SLAM map
# --------------------------------------------------------------------------- #
def _overlap_point_count(submap, graph, overlap_frames):
    """Count confidence-filtered world points contributed by ``submap``'s first
    ``overlap_frames`` frame(s).

    Those points are the ones duplicated from the immediately preceding non-LC
    submap's carried-over overlap frame(s) -- see :func:`gather_world_cloud`. Uses
    the per-frame accessor (``Submap.get_points_list_in_world_frame``) so the count
    stays in lock-step with ``get_points_in_world_frame``'s own per-frame confidence
    filter and frame-major stacking order: frame 0's (then frame 1's, ...) points
    are always the leading entries of the flat arrays ``get_points_in_world_frame``/
    ``get_points_colors`` return, so this count is exactly how many leading rows to
    drop. Mirrors ``server.export.clouds._overlap_point_count`` (the server-side
    counterpart of this fix) exactly.
    """
    try:
        _points, _frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
    except Exception:
        return 0
    total = 0
    for mask in conf_masks[:overlap_frames]:
        if mask is None:
            continue
        try:
            total += int(np.count_nonzero(mask))
        except Exception:
            continue
    return total


def gather_world_cloud(graph_map, graph, overlap_frames=1):
    """Aggregate the world-frame colored point cloud from the SLAM map.

    Mirrors ``GraphMap.write_points_to_file``'s per-submap walk (confidence-filtered
    world points + their colors, concatenated), but EXCLUDES two sources of
    duplicated geometry -- the same fix already shipped server-side
    (``server.export.clouds.gather_full_cloud``; window-quality study,
    ``experiments/research/2026-07-03-omega-full-quality.md`` §2/§6, platform repo):

    - **Loop-closure (LC) re-observation submaps** (``submap.get_lc_status()``
      True). A LC submap exists only to anchor a loop-closure edge in the pose
      graph -- its frames are a copy of a query frame + the retrieved frame,
      already covered by their own "regular" submaps. Counting them into the fused
      cloud doubles geometry exactly at revisited places.
    - **The shared overlap frame(s)** at the start of every regular (non-LC)
      submap after the first. The streaming design carries the last
      ``overlap_frames`` frame(s) of a submap forward as the first frame(s) of the
      next one (to compute the SL(4) junction homography --
      ``--overlapping_window_size`` in ``main.py`` / ``overlapping_window_size`` in
      ``modal_app.py``), so those frames' points would otherwise be unprojected and
      counted twice: once as the tail of the earlier submap, once as the head of
      the later one.

    ``overlap_frames`` should match the caller's overlap window size. Default 1
    matches ``--overlapping_window_size``'s default (the only supported value
    today, per its help text). Pass ``overlap_frames=0`` to disable the
    overlap-frame skip and reproduce the old (pre-fix) totals; LC exclusion still
    applies unconditionally.

    Returns ``(positions (N,3) float32, colors (N,3) float32 in [0,1])``.
    """
    pcd_all, colors_all = [], []
    seen_regular = False
    for submap in graph_map.ordered_submaps_by_key():
        try:
            is_lc = submap.get_lc_status()
        except Exception:
            is_lc = False
        if is_lc:
            continue

        pcd = submap.get_points_in_world_frame(graph)
        if pcd is None:
            continue
        pcd = np.asarray(pcd).reshape(-1, 3)
        cols = np.asarray(submap.get_points_colors()).reshape(-1, 3)
        if pcd.shape[0] == 0:
            continue
        n = min(pcd.shape[0], cols.shape[0])
        pcd, cols = pcd[:n], cols[:n]

        # Drop the leading overlap-frame points on every regular submap after the
        # first -- see the docstring above. ``ordered_submaps_by_key()`` already
        # walks submaps in ascending id order, so "first regular submap seen" ==
        # "lowest-id regular submap".
        skip = 0
        if overlap_frames > 0 and seen_regular:
            skip = min(_overlap_point_count(submap, graph, overlap_frames), n)
        seen_regular = True
        if skip:
            pcd, cols = pcd[skip:], cols[skip:]
        if pcd.shape[0] == 0:
            continue

        pcd_all.append(pcd)
        colors_all.append(cols)

    if not pcd_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    positions = np.concatenate(pcd_all, axis=0).astype(np.float32)
    colors = np.concatenate(colors_all, axis=0).astype(np.float32)
    if colors.size and float(colors.max()) > 1.0:
        colors = colors / 255.0
    return positions, colors


def gather_posed_keyframes(graph_map, graph):
    """Posed keyframes for photometric refinement.

    Returns a list of ``(image (H,W,3) float32 [0,1], K (3,3) float32,
    world_to_cam (4,4) float32)`` over the non-loop-closure submaps. Reachable from
    the solver/map (``Submap.get_all_frames``/``get_all_poses_world``/``proj_mats``),
    so the optional gsplat refinement is supervisable. Returns ``[]`` if frames are
    unavailable.
    """
    out = []
    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        frames = submap.get_all_frames()
        if frames is None:
            continue
        if hasattr(frames, "detach"):  # torch tensor (S,3,H,W)
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        poses_c2w = np.asarray(submap.get_all_poses_world(graph))  # cam-to-world (S,4,4)
        proj_mats = np.asarray(submap.proj_mats)  # (S,4,4), K in [:3,:3]
        for i in range(frames.shape[0]):
            img = frames[i]
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            img = np.ascontiguousarray(img, dtype=np.float32)
            if img.size and float(img.max()) > 1.0:
                img = img / 255.0
            K = np.asarray(proj_mats[i], dtype=np.float32)[:3, :3]
            w2c = np.linalg.inv(poses_c2w[i]).astype(np.float32)
            out.append((img, K, w2c))
    return out


# --------------------------------------------------------------------------- #
# top-level entry point (wired into main.py)
# --------------------------------------------------------------------------- #
def export_splat(
    solver,
    out_path,
    refine=False,
    opacity=DEFAULT_OPACITY,
    scale_factor=1.0,
    overlap_frames=1,
    clamp_scales=True,
    scale_clamp_percentile=DEFAULT_SCALE_CLAMP_PERCENTILE,
    **refine_kwargs,
):
    """Export the SLAM result as a 3DGS ``splat.ply``.

    Always writes the deterministic baseline first. If ``refine`` is set, attempts
    the guarded gsplat refinement *in place* (overwriting the baseline file); any
    failure (no GPU, gsplat missing, runtime error) is caught and the baseline file
    is kept.

    Args:
        solver: a ``vggt_slam`` ``Solver`` (uses ``solver.map`` + ``solver.graph``).
        out_path: output ``.ply`` path.
        refine: enable the optional GPU/gsplat photometric refinement.
        overlap_frames: forwarded to :func:`gather_world_cloud` -- should match the
            SLAM run's ``--overlapping_window_size``.
        clamp_scales: p99 per-axis scale-tail clamp (EXP-7), default ON; removes the
            refined-splat render spikes. ``False`` stores scales as-is.
        scale_clamp_percentile: percentile for that clamp (default p99). Both are
            forwarded to the baseline write and the optional refinement.

    Returns:
        ``out_path`` on success, ``None`` if there was nothing to export.
    """
    import os

    graph_map = solver.map
    graph = solver.graph

    positions, colors = gather_world_cloud(graph_map, graph, overlap_frames=overlap_frames)
    if positions.shape[0] == 0:
        print("[splat-export] no world points to export; skipping.")
        return None

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = point_cloud_to_splat_ply(
        positions, colors, out_path, opacity=opacity, scale_factor=scale_factor,
        clamp_scales=clamp_scales, scale_clamp_percentile=scale_clamp_percentile,
    )
    if n == 0:
        # Every point was non-finite (all NaN/Inf) — don't leave an empty splat behind.
        print("[splat-export] all points were non-finite; no splat written.")
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None
    print(f"[splat-export] wrote baseline 3DGS init with {n} gaussians -> {out_path}")

    if refine:
        try:
            refine_with_gsplat(
                out_path, graph_map, graph, positions, colors,
                overlap_frames=overlap_frames,
                clamp_scales=clamp_scales, scale_clamp_percentile=scale_clamp_percentile,
                **refine_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - never let refinement break the baseline deliverable
            print(f"[splat-export] gsplat refinement unavailable/failed ({exc}); keeping baseline.")

    return out_path


# --------------------------------------------------------------------------- #
# optional GPU refinement (lazy gsplat import; never imported on the baseline path)
# --------------------------------------------------------------------------- #
def refine_with_gsplat(
    out_path,
    graph_map,
    graph,
    positions=None,
    colors=None,
    iters=1000,
    lr=1e-2,
    init_opacity=DEFAULT_OPACITY,
    scale_factor=1.0,
    overlap_frames=1,
    clamp_scales=True,
    scale_clamp_percentile=DEFAULT_SCALE_CLAMP_PERCENTILE,
):
    """Photometric 3DGS optimization with gsplat, seeded by the baseline init.

    GPU + the `gsplat` package required; both are imported lazily. Optimizes
    gaussian means / colors / scales / rotations / opacities against the posed
    keyframe images, then overwrites ``out_path`` with the refined splats.

    Raises on any missing dependency / no-CUDA / empty-keyframe condition so the
    caller (:func:`export_splat`) can fall back to the baseline. **This path has
    not been GPU-validated in CI** — the baseline above is the supported export.
    """
    import torch  # lazy: only needed for the GPU path
    from gsplat import rasterization  # lazy optional dep

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if positions is None or colors is None:
        positions, colors = gather_world_cloud(graph_map, graph, overlap_frames=overlap_frames)
    if positions.shape[0] == 0:
        raise RuntimeError("no world points to refine")

    keyframes = gather_posed_keyframes(graph_map, graph)
    if not keyframes:
        raise RuntimeError("no posed keyframes reachable for photometric refinement")

    device = "cuda"
    n = positions.shape[0]

    init_scales = estimate_point_scales(positions, scale_factor=scale_factor)

    means = torch.nn.Parameter(torch.tensor(positions, dtype=torch.float32, device=device))
    rgb = np.clip(colors, 1e-4, 1.0 - 1e-4)
    colors_logit = torch.nn.Parameter(
        torch.logit(torch.tensor(rgb, dtype=torch.float32, device=device))
    )
    scales_log = torch.nn.Parameter(
        torch.log(torch.tensor(init_scales, dtype=torch.float32, device=device)).unsqueeze(-1).repeat(1, 3)
    )
    quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    quats = torch.nn.Parameter(quats)
    opacity_logit = torch.nn.Parameter(
        torch.full((n,), float(np.log(init_opacity / (1.0 - init_opacity))), dtype=torch.float32, device=device)
    )

    params = [means, colors_logit, scales_log, quats, opacity_logit]
    optimizer = torch.optim.Adam(params, lr=lr)

    gt = [
        (
            torch.tensor(img, dtype=torch.float32, device=device),
            torch.tensor(K, dtype=torch.float32, device=device),
            torch.tensor(w2c, dtype=torch.float32, device=device),
        )
        for img, K, w2c in keyframes
    ]

    for it in range(int(iters)):
        idx = it % len(gt)
        img, K, w2c = gt[idx]
        h, w = img.shape[:2]

        render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(scales_log),
            opacities=torch.sigmoid(opacity_logit),
            colors=torch.sigmoid(colors_logit),
            viewmats=w2c[None],
            Ks=K[None],
            width=w,
            height=h,
        )
        loss = torch.nn.functional.mse_loss(render[0], img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"[splat-export] gsplat refine iter {it}/{iters} loss={loss.item():.6f}")

    with torch.no_grad():
        refined_positions = means.detach().cpu().numpy()
        refined_colors = torch.sigmoid(colors_logit).detach().cpu().numpy()
        refined_scales = torch.exp(scales_log).detach().cpu().numpy()
        refined_opacities = torch.sigmoid(opacity_logit).detach().cpu().numpy()
        q = quats.detach()
        q = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu().numpy()

    write_splat_ply(
        out_path,
        refined_positions,
        refined_colors,
        scales=refined_scales,
        opacities=refined_opacities,
        quats=q,
        clamp_scales=clamp_scales,
        scale_clamp_percentile=scale_clamp_percentile,
    )
    print(f"[splat-export] wrote refined 3DGS splat with {refined_positions.shape[0]} gaussians -> {out_path}")
    return out_path
