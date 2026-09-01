"""Ground frame — gravity, floor and ceiling for a scene that never told us which way is up.

Why this module exists
----------------------
An imported splat arrives in whatever frame its exporter felt like, so
``splat_import.py`` can only record raw AABB extents with ``vertical_axis_known=False``.
That is honest but useless: "3.9 x 2.7 x 2.4" with no named vertical axis is not a room,
and the Measure panel and the agent both (correctly) refuse to call any of those numbers
a ceiling height. This module derives the missing frame from the geometry itself, so an
imported scene can state a real floor area and a real ceiling height — or say plainly
that it could not.

Method, and why it reuses rather than re-fits
---------------------------------------------
Nothing here re-implements a plane fit. ``planes.estimate_up`` supplies the up-vector
ladder (poses -> heuristic) and ``planes.fit_floor`` reaches W4's exp25 RANSAC port,
which the world-transform contract names as THE floor plane for this system — a second
independent fit is exactly the drift the contract exists to prevent. What this module
adds on top is the frame the consumers actually need: heights along the up axis, a
footprint, and an honest confidence gate.

The ceiling is found by mirroring the floor's own logic (a density shelf, not an
extreme): the topmost run of adjacent height bins that is dense enough to be a surface.
Taking a max, or a high percentile, would happily return a light fitting, a stray
floater, or in a roofless outdoor capture, nothing at all dressed up as a ceiling.

Honesty gate
------------
A weak fit does not get to publish numbers. Below :data:`MIN_FLOOR_INLIER_RATIO` (or
below an absolute inlier floor) the result keeps ``vertical_axis_known=False`` and
carries ``note`` saying which test failed, and :func:`metrics_patch` then writes NO
heights at all — a consumer cannot read a bogus floor height that was never stored.
That is deliberate: gating on a flag that sits beside a plausible-looking number is a
bug waiting for the one consumer that forgets to check it.

All coordinates are WORLD frame, all lengths SLAM units (``units: "relative"``,
``units_basis: "slam_world_units"``) — metres exist only through the metric anchor,
which is a display concern. Flask-free (``planes.py`` posture); routes live in
``routes_imported.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from server.oreos import planes as planes_mod

_EPS = 1e-9

# Floor inliers as a fraction of the working cloud. A real room floor is a large share
# of the visible surface; a few hundred points that happen to be coplanar are not a
# floor, they are a table top or a RANSAC accident. Tuned to admit sparse imports (the
# founder's scene is a walk-through, so the floor is under-sampled versus the walls)
# while refusing fits with no support.
MIN_FLOOR_INLIER_RATIO = float(0.015)

# Absolute inlier floor — protects the ratio test on small clouds, where 1.5% of 2,000
# points is 30 points and any 3 of them define a plane.
MIN_FLOOR_INLIERS = 400

# Ceiling shelf detection. A shelf is a height band whose point density stands well
# clear of the AMBIENT density (walls + noise, which are spread over the whole height),
# not one measured against the tallest peak: the tallest peak is normally the floor, and
# normalizing to it makes the test pass or fail on how thin the FLOOR happens to be —
# a razor-thin synthetic floor then hides a perfectly real ceiling. Ratio-to-ambient is
# stable for both a 5 mm plane and a 15 cm-thick reconstructed slab.
CEILING_BINS = 60
CEILING_SHELF_RATIO = 4.0
CEILING_MIN_SHELF_FRAC = 0.01

# The ceiling must clear the floor by this fraction of the cloud's total height span,
# else what we found is furniture, not a ceiling.
MIN_ROOM_HEIGHT_FRAC = 0.35

# Footprint occupancy grid. ~48 cells across the longer axis keeps a real room's floor
# area stable while still cutting the empty quadrant out of an L-shaped plan.
FOOTPRINT_CELLS = 48

DERIVATION = "plane-fit"


class GroundFrameError(ValueError):
    """The ground frame could not be derived at all (no geometry / no floor).

    Distinct from a WEAK fit, which is a successful computation with an honest
    ``vertical_axis_known=False`` — this is "there was nothing to fit"."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _shelf_center_from_top(heights: np.ndarray) -> Optional[float]:
    """Height of the topmost dense shelf, or ``None`` when the top is all scatter.

    Walks bins downward from the top for the first one that clears the shelf bar (see
    :data:`CEILING_SHELF_RATIO`), then widens to the contiguous run around it at half
    that bar and returns the run's count-weighted centre — so a thin plane and a thick
    reconstructed slab both report their middle rather than one arbitrary bin edge.

    The 1-99 percentile clip is what keeps a single far floater from stretching the bin
    width until the real ceiling and the real floor share a bin."""
    if heights.shape[0] < 2 * CEILING_BINS:
        return None
    lo, hi = (float(v) for v in np.percentile(heights, [1.0, 99.0]))
    if hi - lo < _EPS:
        return None
    hist, edges = np.histogram(heights, bins=CEILING_BINS, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ambient = max(float(np.median(hist)), 1.0)
    bar = max(CEILING_SHELF_RATIO * ambient, CEILING_MIN_SHELF_FRAC * float(heights.shape[0]))
    peak = next((b for b in range(len(hist) - 1, -1, -1) if hist[b] >= bar), None)
    if peak is None:
        return None
    lo_b = hi_b = peak
    while lo_b > 0 and hist[lo_b - 1] >= 0.5 * bar:
        lo_b -= 1
    while hi_b < len(hist) - 1 and hist[hi_b + 1] >= 0.5 * bar:
        hi_b += 1
    run = hist[lo_b : hi_b + 1].astype(np.float64)
    return float(np.average(centers[lo_b : hi_b + 1], weights=run))


def _footprint(
    points: np.ndarray, floor_plane: Any, height_band: float
) -> tuple[list[float], float, int]:
    """Floor-inlier footprint -> ``([width, depth], occupied_area, n_inliers)``.

    ``width``/``depth`` are the percentile-clipped extent in the floor's own (u, v)
    basis (``planes._uv_bounds`` posture — RANSAC always admits a few stragglers at a
    relative threshold, and an unclipped bbox hands them the whole answer).

    The area is NOT ``width * depth``: it is the occupied fraction of a uv grid over
    that rectangle, times the rectangle. A real floor plan is rarely a filled rectangle,
    and quoting the bounding box as floor area systematically overstates every L-shaped
    room in the demo."""
    rel = points - np.asarray(floor_plane.point, dtype=np.float64)
    normal = np.asarray(floor_plane.normal, dtype=np.float64)
    inlier = np.abs(rel @ normal) < height_band
    n_inliers = int(inlier.sum())
    if n_inliers < 8:
        return [0.0, 0.0], 0.0, n_inliers
    u = rel[inlier] @ np.asarray(floor_plane.u_axis, dtype=np.float64)
    v = rel[inlier] @ np.asarray(floor_plane.v_axis, dtype=np.float64)
    u0, u1 = (float(x) for x in np.percentile(u, [1.0, 99.0]))
    v0, v1 = (float(x) for x in np.percentile(v, [1.0, 99.0]))
    width = max(u1 - u0, 0.0)
    depth = max(v1 - v0, 0.0)
    if width < _EPS or depth < _EPS:
        return [width, depth], 0.0, n_inliers

    cells_u = max(int(round(FOOTPRINT_CELLS * width / max(width, depth))), 4)
    cells_v = max(int(round(FOOTPRINT_CELLS * depth / max(width, depth))), 4)
    hist, _, _ = np.histogram2d(u, v, bins=(cells_u, cells_v), range=[[u0, u1], [v0, v1]])
    occupied = float((hist > 0).mean())
    return [width, depth], occupied * width * depth, n_inliers


def compute_ground_frame(
    points: np.ndarray,
    poses: Optional[np.ndarray] = None,
    up_override: Optional[Any] = None,
    *,
    max_points: int = 400_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Derive the scene's ground frame from its cloud.

    Returns a dict carrying the contract fields (``up_axis``, ``floor_height``,
    ``ceiling_height``, ``floor_extent``, ``vertical_axis_known``, ``derivation``) plus
    the diagnostics that justify them (inlier ratio/count, room height, floor area, the
    up-vector source, and ``note`` when the fit is too weak to publish).

    Heights are signed distances along ``up_axis`` in the WORLD frame — so
    ``ceiling_height - floor_height`` is the room height, and both numbers stay
    meaningful next to a world-frame object centre. Decimation is mandatory, not
    optional: the founder's scene is 8.5M gaussians and the RANSAC port is O(N) per
    iteration.

    Raises :class:`GroundFrameError` when there is nothing to fit;
    :class:`planes.PathplanUnavailable` propagates (the caller answers 503, matching
    ``routes_planes``)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 32:
        raise GroundFrameError(
            "no_geometry", f"only {pts.shape[0]} finite points — nothing to fit a floor to"
        )
    work, _ = planes_mod.downsample_points(pts, None, max_points=max_points, seed=seed)

    up, up_source = planes_mod.estimate_up(work, poses=poses, up_override=up_override)
    thresh = planes_mod.default_thresh(work)
    try:
        floor_plane = planes_mod.fit_floor(work, up, thresh=thresh, seed=seed)
    except planes_mod.PathplanUnavailable:
        raise  # the caller answers 503, exactly as routes_planes does
    except Exception as exc:
        # The port raises ValueError on degenerate geometry and its own NavError
        # ("no_floor", detail) on a failed fit; both mean the same thing to a caller.
        raise GroundFrameError(str(getattr(exc, "code", "no_floor")), str(exc)) from exc

    # The refit floor normal is a better up estimate than the seed heuristic (same
    # reasoning as routes_planes' wall constraint), so the frame is expressed in it.
    up_axis = np.asarray(floor_plane.normal, dtype=np.float64)
    up_axis = up_axis / max(float(np.linalg.norm(up_axis)), _EPS)

    heights = work @ up_axis
    floor_height = float(np.asarray(floor_plane.point, dtype=np.float64) @ up_axis)
    span = float(np.percentile(heights, 99.0) - np.percentile(heights, 1.0))

    floor_extent, floor_area, n_inliers = _footprint(work, floor_plane, thresh)
    inlier_ratio = n_inliers / float(work.shape[0])

    ceiling_height: Optional[float] = None
    ceiling_note: Optional[str] = None
    shelf = _shelf_center_from_top(heights)
    if shelf is None:
        ceiling_note = "no dense surface near the top of the cloud — the capture has no ceiling"
    elif shelf - floor_height < MIN_ROOM_HEIGHT_FRAC * max(span, _EPS):
        ceiling_note = (
            "the densest upper surface sits too close to the floor to be a ceiling "
            "(it is furniture, or the capture is open above)"
        )
    else:
        ceiling_height = float(shelf)

    weak: Optional[str] = None
    if n_inliers < MIN_FLOOR_INLIERS:
        weak = (
            f"only {n_inliers} points support the floor plane (need {MIN_FLOOR_INLIERS}) — "
            "not enough evidence to name a vertical axis"
        )
    elif inlier_ratio < MIN_FLOOR_INLIER_RATIO:
        weak = (
            f"the floor plane holds {inlier_ratio:.1%} of the cloud "
            f"(need {MIN_FLOOR_INLIER_RATIO:.1%}) — too weak to name a vertical axis"
        )

    return {
        "up_axis": [float(v) for v in up_axis],
        "floor_height": floor_height,
        "ceiling_height": ceiling_height,
        "room_height": None if ceiling_height is None else float(ceiling_height - floor_height),
        "floor_extent": [float(v) for v in floor_extent],
        "floor_area": float(floor_area),
        "vertical_axis_known": weak is None,
        "derivation": DERIVATION,
        "note": weak or ceiling_note,
        # Diagnostics — what the gate above actually measured.
        "floor_inliers": n_inliers,
        "floor_inlier_ratio": float(inlier_ratio),
        "floor_point": [float(v) for v in np.asarray(floor_plane.point, dtype=np.float64)],
        "up_source": up_source,
        "plane_thresh": float(thresh),
        "sampled_points": int(work.shape[0]),
        "height_span": span,
        "units": "relative",
        "units_basis": "slam_world_units",
    }


def metrics_patch(frame: dict[str, Any]) -> dict[str, Any]:
    """The subset of a ground frame that belongs on ``facts.metrics``.

    A weak fit contributes ONLY the flag, the derivation and the note — no heights, no
    axis, no extent. There is then no stored number a forgetful consumer could read as
    a floor height, which is a stronger guarantee than "gate on the boolean"."""
    patch: dict[str, Any] = {
        "vertical_axis_known": bool(frame.get("vertical_axis_known")),
        "derivation": str(frame.get("derivation") or DERIVATION),
    }
    note = frame.get("note")
    if note:
        patch["ground_frame_note"] = str(note)
    if not patch["vertical_axis_known"]:
        return patch
    patch["up_axis"] = [float(v) for v in frame["up_axis"]]
    patch["floor_height"] = float(frame["floor_height"])
    patch["floor_extent"] = [float(v) for v in frame["floor_extent"]]
    patch["floor_area"] = float(frame["floor_area"])
    # Ceiling can legitimately be absent on an open capture even when the floor is
    # solid; omit rather than write a null a consumer would have to special-case.
    if frame.get("ceiling_height") is not None:
        patch["ceiling_height"] = float(frame["ceiling_height"])
        patch["room_height"] = float(frame["room_height"])
    return patch


def compute_and_store(
    persistence: Any, user_id: str, scan_id: str, points: Any
) -> Optional[dict[str, Any]]:
    """Fit the ground frame and merge it into a persisted scan's ``facts.metrics``.

    The entry point for import time (``splat_import``) and for the backfill script;
    the route in ``routes_imported.py`` calls the two halves separately so it can offer
    ``dry_run``. Best-effort by design — a splat we cannot gravity-align still imported
    fine, so a failure logs and returns ``None`` rather than sinking the upload the
    founder just waited on."""
    try:
        frame = compute_ground_frame(points)
    except Exception as exc:
        print(f"[demo.ground_frame] fit skipped for {scan_id}: {exc}")
        return None
    try:
        persistence.update_scene_metrics(user_id, scan_id, metrics_patch(frame))
    except Exception as exc:
        print(f"[demo.ground_frame] metrics write failed for {scan_id}: {exc}")
        return None
    return frame
