"""Server-side rasterizer tests — GPU-free, and mostly about ONE thing.

The load-bearing tests here are ``test_a_lone_gaussian_lands_on_its_projected_pixel``
and its siblings. A render made at the wrong pose raises nothing and looks entirely
plausible; it just makes every mask, back-projection and dimension derived from it land
somewhere else in the room. So the renderer's pixel grid is checked against
``synthetic_views.project_world_point`` — the same function the client's
``projectWorldPointToPixel`` mirrors, pinned in ``tests/unit/syntheticViews.test.ts`` and
in ``tests/test_demo_imported.py`` against the same hand-computed numbers. Three suites,
one convention.

The camera cases below are deliberately the SAME ones ``test_demo_imported.py`` pins
(camera at ``(0, 0, 5)`` with an identity three.js quaternion; the +90-degrees-about-Y
camera on the +X axis), so if the convention ever drifts these fail together rather than
one of them silently disagreeing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the rasterizer's only heavy dependency
try:
    # A torch built against numpy 1.x sitting next to numpy 2.x imports fine and then
    # fails on every `.numpy()`. That is an environment mismatch, not a defect in the
    # code under test, so it skips with the reason rather than failing 13 assertions and
    # burying the real signal. The deployed image (torch==2.3.1) has no such gap, and
    # `modal run modal_oreos_render.py::unit` runs this same file inside it.
    torch.zeros(1).numpy()
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"torch cannot exchange arrays with this numpy build ({exc})", allow_module_level=True)

from server.oreos import splat_render as sr
from server.oreos import synthetic_views as sv


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _cloud(
    means,
    *,
    scale=0.05,
    opacity=6.0,
    colour=(1.0, 1.0, 1.0),
    sh_degree=0,
):
    """A minimal cloud of round gaussians. ``colour`` is the DC term expressed as linear
    RGB, inverted through the 3DGS decode ``rgb = 0.5 + C0 * f_dc`` so a test can ask for
    "red" and get red."""
    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    n = means.shape[0]
    f_dc = (np.asarray(colour, dtype=np.float32).reshape(1, 3) - 0.5) / sr._SH_C0
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc
    return sr.GaussianCloud(
        means=means,
        scales_log=np.full((n, 3), math.log(scale), dtype=np.float32),
        quats_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        opacity_logit=np.full(n, opacity, dtype=np.float32),
        sh=sh,
        sh_degree=sh_degree,
    )


def _view(position=(0.0, 0.0, 5.0), quat=(0.0, 0.0, 0.0, 1.0), width=256, height=192, fov=60.0):
    """``(c2w, intrinsics, width, height)`` through the SAME conversion the wire uses."""
    return (
        sv.gl_pose_to_cv_c2w(position, quat),
        sv.intrinsics_from_fov(fov, width, height),
        width,
        height,
    )


# ---------------------------------------------------------------------------
# the frame convention — the whole risk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "world_point",
    [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.45, 0.0), (-0.5, -0.3, 0.4)],
)
def test_a_lone_gaussian_lands_on_its_projected_pixel(world_point):
    """THE round trip. Render one gaussian at a known world point and confirm the light
    it puts on the sensor is centred exactly where ``project_world_point`` says it is.

    The centroid is alpha-weighted over the whole frame, so this catches a half-pixel
    sampling-origin slip as readily as a transposed rotation — and a transposed rotation
    is the failure that produces a perfectly reasonable-looking picture of the wrong
    part of the room."""
    c2w, intr, w, h = _view()
    result = sr.render(_cloud([world_point], scale=0.02), c2w, intr, w, h)

    expected = sv.project_world_point(c2w, intr, world_point)
    assert expected is not None
    got = sr.alpha_centroid(result.alpha)
    assert got is not None, "the gaussian rendered nothing at all"
    assert got == pytest.approx(expected, abs=0.25)


def test_a_yawed_camera_still_lands_on_the_same_world_point():
    """The case an accidentally-transposed rotation passes at identity and fails here:
    camera on the +X axis, yawed 90 degrees about +Y so it faces the origin again."""
    s = math.sqrt(0.5)
    c2w, intr, w, h = _view(position=(5.0, 0.0, 0.0), quat=(0.0, s, 0.0, s))
    result = sr.render(_cloud([(0.0, 0.3, 0.0)], scale=0.02), c2w, intr, w, h)

    expected = sv.project_world_point(c2w, intr, (0.0, 0.3, 0.0))
    assert sr.alpha_centroid(result.alpha) == pytest.approx(expected, abs=0.25)


def test_image_y_runs_down_like_the_projection_says_it_does():
    """A point ABOVE the optical axis must render in the UPPER half of the image. This is
    the one sign the OpenCV/GL flip exists to get right, and flipping it produces a
    vertically mirrored render that no other assertion here would notice."""
    c2w, intr, w, h = _view()
    result = sr.render(_cloud([(0.0, 0.8, 0.0)], scale=0.03), c2w, intr, w, h)
    _cx, cy = sr.alpha_centroid(result.alpha)
    assert cy < h / 2.0


def test_a_gaussian_behind_the_camera_renders_nothing_and_says_why():
    c2w, intr, w, h = _view()
    result = sr.render(_cloud([(0.0, 0.0, 9.0)]), c2w, intr, w, h)
    assert result.alpha.max() == 0.0
    assert result.stats["empty_reason"]


def test_the_frame_is_exactly_the_requested_size():
    """The tile grid pads to a multiple of TILE internally; the padding must never reach
    the caller, or every downstream pixel coordinate is off by the pad width."""
    c2w, intr, _w, _h = _view()
    result = sr.render(_cloud([(0.0, 0.0, 0.0)]), c2w, sv.intrinsics_from_fov(60.0, 101, 67), 101, 67)
    assert result.rgb.shape == (67, 101, 3)
    assert result.alpha.shape == (67, 101)


# ---------------------------------------------------------------------------
# compositing
# ---------------------------------------------------------------------------


def test_the_nearer_gaussian_wins():
    """Front-to-back "over": an opaque red gaussian in front of a blue one is red. Sort
    the other way and you get blue, which looks like a perfectly good render."""
    c2w, intr, w, h = _view()
    near = _cloud([(0.0, 0.0, 1.0)], scale=0.1, opacity=8.0, colour=(1.0, 0.0, 0.0))
    far = _cloud([(0.0, 0.0, -1.0)], scale=0.1, opacity=8.0, colour=(0.0, 0.0, 1.0))
    both = sr.GaussianCloud(
        means=np.concatenate([far.means, near.means]),  # far FIRST: order must not matter
        scales_log=np.concatenate([far.scales_log, near.scales_log]),
        quats_wxyz=np.concatenate([far.quats_wxyz, near.quats_wxyz]),
        opacity_logit=np.concatenate([far.opacity_logit, near.opacity_logit]),
        sh=np.concatenate([far.sh, near.sh]),
        sh_degree=0,
    )
    rgb = sr.render(both, c2w, intr, w, h).rgb
    centre = rgb[h // 2, w // 2]
    assert int(centre[0]) > 200 and int(centre[2]) < 60


def test_transparent_gaussians_let_the_background_through():
    c2w, intr, w, h = _view()
    result = sr.render(
        _cloud([(0.0, 0.0, 0.0)], scale=0.05, opacity=-6.0), c2w, intr, w, h,
        background=(0.0, 1.0, 0.0),
    )
    assert result.alpha.max() < 0.05
    assert int(result.rgb[h // 2, w // 2][1]) > 200


def test_expected_depth_is_the_distance_to_the_surface():
    """The depth buffer is what makes the occlusion test in :func:`visibility` possible,
    so it has to be a real distance and not a normalised or inverted one."""
    c2w, intr, w, h = _view()
    result = sr.render(_cloud([(0.0, 0.0, 1.0)], scale=0.05, opacity=8.0), c2w, intr, w, h)
    assert result.depth[h // 2, w // 2] == pytest.approx(4.0, abs=0.1)  # camera z=5, point z=1


# ---------------------------------------------------------------------------
# spherical harmonics
# ---------------------------------------------------------------------------


def test_degree_zero_colour_is_the_plain_dc_decode():
    """``rgb = 0.5 + C0 * f_dc`` — our own exports are SH-0, so this is the common path."""
    sh = np.zeros((1, 1, 3), dtype=np.float32)
    sh[0, 0] = [1.0, 2.0, 3.0]
    got = sr._sh_colour(torch, torch.as_tensor(sh), torch.zeros(1, 3), 0).numpy()[0]
    assert got == pytest.approx(0.5 + sr._SH_C0 * np.array([1.0, 2.0, 3.0]), rel=1e-6)


def test_degree_three_colour_actually_depends_on_the_view_direction():
    """The founder's import is SH degree 3, so this is not a hypothetical path: if the
    higher bands were dropped every render would come back view-independent, which looks
    like a flat, matte version of the same room rather than like an error."""
    rng = np.random.default_rng(0)
    sh = rng.normal(0.0, 0.4, size=(1, 16, 3)).astype(np.float32)
    a = sr._sh_colour(torch, torch.as_tensor(sh), torch.as_tensor([[0.0, 0.0, 1.0]]), 3).numpy()
    b = sr._sh_colour(torch, torch.as_tensor(sh), torch.as_tensor([[0.0, 0.0, -1.0]]), 3).numpy()
    assert not np.allclose(a, b, atol=1e-3)
    # and the DC term alone must be what degree 0 would have produced
    dc_only = sr._sh_colour(torch, torch.as_tensor(sh), torch.as_tensor([[0.0, 0.0, 1.0]]), 0).numpy()
    assert dc_only == pytest.approx(0.5 + sr._SH_C0 * sh[:, 0, :], rel=1e-5)


def test_channel_major_f_rest_is_transposed_into_coefficient_major():
    """The PLY stores all of red's rest coefficients, then green's, then blue's. Reading
    that as coefficient-major gives colour that is wrong and entirely plausible, which is
    the worst kind of wrong — so the layout is asserted rather than assumed."""
    n, m = 2, 15
    f_dc = np.zeros((n, 3), dtype=np.float32)
    f_rest = np.arange(n * 3 * m, dtype=np.float32).reshape(n, 3 * m)
    sh, degree = sr._assemble_sh(f_dc, f_rest)
    assert degree == 3 and sh.shape == (n, 16, 3)
    for channel in range(3):
        for coeff in range(m):
            assert sh[0, 1 + coeff, channel] == f_rest[0, channel * m + coeff]


def test_a_partial_sh_band_is_refused_not_guessed():
    with pytest.raises(sr.SplatRenderError):
        sr._assemble_sh(np.zeros((2, 3), np.float32), np.zeros((2, 3 * 7), np.float32))


# ---------------------------------------------------------------------------
# pose planning — parity with the client's planRingCapture
# ---------------------------------------------------------------------------


def test_the_ring_plan_matches_the_client_defaults():
    specs = sr.plan_ring([0.0, 0.0, 0.0], [6.0, 2.5, 4.0], up=[0.0, 1.0, 0.0])
    assert len(specs) == sr.RING_DEFAULTS["ring_count"] + sr.RING_DEFAULTS["elevated_count"]
    assert [s.label for s in specs][:2] == ["ring 1/8", "ring 2/8"]
    assert specs[-1].label == "elevated 2/2"
    assert all(s.width == 1024 and s.height == 768 and s.fov_y_deg == 70.0 for s in specs)


def test_every_planned_camera_actually_looks_at_the_room():
    """A ring planned around the wrong axis produces eight views of the ceiling and
    throws nothing. Each camera's aim point must project near the centre of its frame."""
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([6.0, 2.5, 4.0])
    specs = sr.plan_ring(lo, hi, up=[0.0, 1.0, 0.0])
    centre = 0.5 * (lo + hi)
    for spec in specs:
        c2w = sv.gl_pose_to_cv_c2w(spec.position, spec.quaternion)
        intr = sv.intrinsics_from_fov(spec.fov_y_deg, spec.width, spec.height)
        px = sv.project_world_point(c2w, intr, centre)
        assert px is not None, f"{spec.label} points away from the room"
        assert 0 <= px[0] <= spec.width and 0 <= px[1] <= spec.height


def test_the_ring_is_planned_around_the_given_up_axis():
    """The founder's import has up ~ −Y (its ground frame says so). Planning around +Y
    would put every camera under the floor."""
    up = np.array([0.0, -1.0, 0.0])
    specs = sr.plan_ring([0.0, -3.0, 0.0], [6.0, 0.0, 4.0], up=up)
    # eye height is a positive fraction of the extent ALONG up, so every camera sits on
    # the up side of the floor plane
    heights = [float(np.asarray(s.position) @ up) for s in specs]
    assert min(heights) > float(np.array([3.0, 0.0, 2.0]) @ up)


def test_a_degenerate_bbox_plans_nothing_rather_than_a_nan_camera():
    assert sr.plan_ring([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == []


#: The founder's scene, to the numbers its record actually carries: an up axis that is
#: very nearly −Y, a bbox 15.6 units tall, and a fitted room 3.4 units tall inside it.
FOUNDER_UP = [-0.0143, -0.99959, 0.0248]
FOUNDER_LO = [-12.07, -7.80, -11.85]
FOUNDER_HI = [8.09, 7.80, 7.61]
FOUNDER_FLOOR = -0.8795
FOUNDER_CEILING = 2.5154


def _heights(specs, up=FOUNDER_UP):
    u = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    return {s.label: float(np.asarray(s.position) @ u) for s in specs}


def test_the_floor_is_the_bottom_of_the_box_even_when_up_points_at_minus_y():
    """``(lo − centre) · up`` is only the floor when ``up`` is +Y-ish; against this
    scene's fitted −Y it projects onto the TOP of the box, and the eye height comes out
    BELOW the geometry. Measured: the first headless ring put every camera at height
    −17.5 in a scene whose gaussians stop at −7.7, so ten frames rendered nothing."""
    heights = _heights(sr.plan_ring(FOUNDER_LO, FOUNDER_HI, up=FOUNDER_UP))
    span = sorted(
        [float(np.asarray(FOUNDER_LO) @ np.asarray(FOUNDER_UP) / np.linalg.norm(FOUNDER_UP)),
         float(np.asarray(FOUNDER_HI) @ np.asarray(FOUNDER_UP) / np.linalg.norm(FOUNDER_UP))]
    )
    ring = [h for label, h in heights.items() if label.startswith("ring")]
    assert span[0] <= min(ring) and max(ring) <= span[1]


def test_a_fitted_room_height_replaces_the_bounding_boxs():
    """3.4 units of real room inside a 15.6-unit box. Eye height must come from the fit,
    exactly — 62% of the way from the fitted floor to the fitted ceiling."""
    fitted = sr.plan_ring(
        FOUNDER_LO, FOUNDER_HI, up=FOUNDER_UP,
        floor_height=FOUNDER_FLOOR, ceiling_height=FOUNDER_CEILING,
    )
    eye = _heights(fitted)["ring 1/8"]
    assert eye == pytest.approx(FOUNDER_FLOOR + 0.62 * (FOUNDER_CEILING - FOUNDER_FLOOR), abs=1e-6)
    naive_eye = _heights(sr.plan_ring(FOUNDER_LO, FOUNDER_HI, up=FOUNDER_UP))["ring 1/8"]
    assert abs(naive_eye - eye) > 0.5  # the box's idea of eye level is a different room


def test_an_elevated_camera_stays_under_a_known_ceiling():
    """1.05x the vertical extent is ABOVE a room that has a real ceiling, so the camera
    looks down through the slab and the frame fills with out-of-focus soffit. Both
    elevated views did exactly that on the founder's scene."""
    naive = _heights(sr.plan_ring(FOUNDER_LO, FOUNDER_HI, up=FOUNDER_UP))
    fitted = _heights(
        sr.plan_ring(
            FOUNDER_LO, FOUNDER_HI, up=FOUNDER_UP,
            floor_height=FOUNDER_FLOOR, ceiling_height=FOUNDER_CEILING,
        )
    )
    assert naive["elevated 1/2"] > FOUNDER_CEILING       # through the ceiling
    assert FOUNDER_FLOOR < fitted["elevated 1/2"] < FOUNDER_CEILING
    assert fitted["elevated 1/2"] > fitted["ring 1/8"]   # still elevated


def test_the_elliptical_ring_keeps_short_axis_cameras_off_the_wall():
    """0.55 of the half-DIAGONAL overshoots the short axis of a narrow room: on an
    8.6 x 11.9 floor that is 4.05 units against a 4.3-unit half-width."""
    lo, hi = [0.0, 0.0, 0.0], [8.6, 2.5, 11.9]
    circle = sr.plan_ring(lo, hi, up=[0.0, 1.0, 0.0], ring_count=4)
    ellipse = sr.plan_ring(lo, hi, up=[0.0, 1.0, 0.0], ring_count=4, elliptical=True)
    # ring 1/4 sits on the +u axis, which here is world +x (the short axis)
    assert circle[0].position[0] > 8.0   # essentially at the wall
    assert ellipse[0].position[0] < 7.0  # 55% of the way to it


def test_a_camera_standing_inside_geometry_is_pulled_out():
    """A render from inside a wall is a correct picture of the inside of a wall, and
    nothing downstream can tell it from a view of a cluttered corner — so the eye point
    is checked against the geometry rather than trusted."""
    rng = np.random.default_rng(3)
    # a slab of geometry exactly where the camera wants to stand
    blob = rng.normal([4.0, 1.0, 0.0], 0.15, size=(2000, 3))
    spec = sr.ViewSpec(
        position=[4.0, 1.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0],
        fov_y_deg=70.0, width=64, height=48, label="probe", aim=[0.0, 1.0, 0.0],
    )
    moved = sr.clear_cameras([spec], blob, clearance=0.5, allowed=0)[0]
    assert moved.placement["moved_fraction"] > 0
    assert moved.placement["clear"] is True
    assert moved.position[0] < 4.0  # slid toward the aim point


def test_a_camera_already_in_open_space_is_left_exactly_where_it_was():
    spec = sr.ViewSpec(
        position=[4.0, 1.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0],
        fov_y_deg=70.0, width=64, height=48, label="probe", aim=[0.0, 1.0, 0.0],
    )
    kept = sr.clear_cameras([spec], np.zeros((10, 3)), clearance=0.5, allowed=0)[0]
    assert kept.position == spec.position
    assert kept.quaternion == spec.quaternion
    assert kept.placement["moved_fraction"] == 0.0


def test_object_views_frame_the_object_and_start_from_the_room_side():
    centre = np.array([3.0, 0.5, 1.0])
    specs = sr.plan_object_views(
        centre, [0.8, 0.9, 0.6], room_center=[3.0, 1.25, 2.0], up=[0.0, 1.0, 0.0], candidates=4
    )
    assert len(specs) == 4
    for spec in specs:
        c2w = sv.gl_pose_to_cv_c2w(spec.position, spec.quaternion)
        intr = sv.intrinsics_from_fov(spec.fov_y_deg, spec.width, spec.height)
        px = sv.project_world_point(c2w, intr, centre)
        assert px == pytest.approx((spec.width / 2.0, spec.height / 2.0), abs=1.0)
    # the first candidate looks from the room's interior, i.e. the +z side here
    assert specs[0].position[2] > centre[2]


# ---------------------------------------------------------------------------
# selection + subsetting
# ---------------------------------------------------------------------------


def test_decimation_keeps_the_sh_band_and_hits_the_budget():
    rng = np.random.default_rng(1)
    cloud = _cloud(rng.uniform(-2.0, 2.0, size=(4000, 3)), sh_degree=3)
    cloud.sh[:] = rng.normal(0.0, 0.3, cloud.sh.shape).astype(np.float32)
    thinned, info = cloud, {}
    thinned, info = sr.decimate(cloud, 500, method="voxel_first")
    assert thinned.count <= 500
    assert thinned.sh.shape[1:] == (16, 3) and thinned.sh_degree == 3
    assert info["selected"] == thinned.count


def test_decimation_below_the_source_count_is_a_no_op():
    cloud = _cloud(np.zeros((10, 3)))
    thinned, info = sr.decimate(cloud, 100)
    assert thinned is cloud and info["method"] == "identity"


def test_obb_subset_uses_full_extents_and_column_axes():
    """WORLD-TRANSFORM-CONTRACT: ``extents`` are FULL lengths and the rotation's COLUMNS
    are the axes. Reading either the other way silently selects the wrong points."""
    means = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32)
    inside = sr.subset_in_obb(means, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert sorted(inside.tolist()) == [0, 1]  # 0.6 > half-extent 0.5

    # rotate the box 90 degrees about z: its former x half-extent now runs along y
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    narrow = sr.subset_in_obb(means, [0.0, 0.0, 0.0], [0.2, 1.4, 1.0], rot)
    assert sorted(narrow.tolist()) == [0, 1, 2]


def test_visibility_refuses_to_call_an_occluded_object_visible():
    """An object rendered alone is never hidden; the scene's own depth is what decides.
    Without this the SAM 3D path would happily hand the model a mask over a wall."""
    obj_alpha = np.ones((4, 4), dtype=np.float32)
    obj_depth = np.full((4, 4), 5.0, dtype=np.float32)
    wall_in_front = np.full((4, 4), 2.0, dtype=np.float32)
    clear = np.full((4, 4), 5.0, dtype=np.float32)
    assert sr.visibility(obj_alpha, obj_depth, wall_in_front)["visible_fraction"] == 0.0
    assert sr.visibility(obj_alpha, obj_depth, clear)["visible_fraction"] == 1.0


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def test_renders_encode_as_real_pngs():
    """``synthetic_views._decode_png`` asserts the magic bytes, so anything else here
    would be rejected at the moment of registration rather than at the moment of use."""
    pytest.importorskip("PIL")
    c2w, intr, w, h = _view()
    result = sr.render(_cloud([(0.0, 0.0, 0.0)]), c2w, intr, w, h)
    png = sr.encode_png(result.rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    mask = sr.encode_mask_png(result.alpha > 0.5)
    assert mask.startswith(b"\x89PNG\r\n\x1a\n")
