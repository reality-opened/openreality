"""F4 pathplan + nav-route tests (W4, docs/demo-2026-07) — GPU-free.

Two layers, mirroring ``test_demo_routes.py``:

1. Pure-engine tests against ``server.oreos.pathplan`` on a synthetic room fixture
   (floor + walls + a table + wall-hugging noise), built in a LOCAL metric frame and
   rotated into a "VGGT-like" world (y-down-ish, 14° first-camera tilt) — exactly the
   gauge the EXP-25 lessons were learned on. Covers: floor fit within tolerance, the
   occupancy footprint, largest-component restriction + unreachable pockets, the
   units ladder, pose-up vs heuristic-up vs override, determinism, snapping honesty,
   and the OpenCV pose contract.

2. Route-contract tests through a real Flask client + ``ModalScenePersistence`` on
   tmp_path + a stubbed ``server.app`` (the demo_app harness pattern): auth 401,
   bad goal 400, no_geometry / unknown_object 404, happy path + provenance line,
   grid-cache reuse (fit monkeypatched to explode → still 200), param-change grid
   rebuild, path-doc + manifest indexing, splat-means geometry fallback, and the
   ``?debug=1`` PNG artifact.
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("scipy")

from server.scene_report.schemas import SceneFacts, SceneReport  # noqa: E402
from server.scene_report.store import ModalScenePersistence  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic-room fixture (shared by both layers)
# ---------------------------------------------------------------------------

TABLE_BOX = (2.4, 3.6, 1.6, 2.4)  # x0, x1, y0, y1 (local metres); top at 0.72
POCKET_WALL_X = 5.0  # sealed closet occupies local x > POCKET_WALL_X when enabled


def _rotation_and_offset(tilt_deg: float = 14.0):
    """local (z-up, metres) → world: z→-y (OpenCV-ish y-down world) then tilt about x."""
    tilt = np.radians(tilt_deg)
    swap = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # z -> -y, y -> z
    ct, st = np.cos(tilt), np.sin(tilt)
    tilt_m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], float)
    rot = tilt_m @ swap
    return rot, np.array([2.0, -1.0, 0.5])


def make_room(
    seed: int = 7,
    tilt_deg: float = 14.0,
    z_up: bool = False,
    with_pocket: bool = False,
    n_floor: int = 30000,
):
    """→ dict(points, poses, rot, t, up_world, to_world(fn), to_local(fn)).

    6×4 m room, floor z=0 local, walls to 2.6 m, a 1.2×0.8 table (top 0.72 + legs),
    sparse wall-hugging floaters. ``with_pocket`` adds an interior wall at x=5 with NO
    door — free floor behind it is a genuinely unreachable island (the largest-component
    lesson's setup). ``z_up=True`` skips the rotation (heuristic-friendly world).
    """
    rng = np.random.default_rng(seed)
    pts = []
    pts.append(np.stack([rng.uniform(0, 6, n_floor), rng.uniform(0, 4, n_floor),
                         rng.normal(0, 0.01, n_floor)], 1))
    for wall in range(4):
        n = 7000
        h = rng.uniform(0, 2.6, n)
        if wall == 0:
            p = np.stack([rng.uniform(0, 6, n), rng.normal(0, 0.01, n), h], 1)
        elif wall == 1:
            p = np.stack([rng.uniform(0, 6, n), 4.0 + rng.normal(0, 0.01, n), h], 1)
        elif wall == 2:
            p = np.stack([rng.normal(0, 0.01, n), rng.uniform(0, 4, n), h], 1)
        else:
            p = np.stack([6.0 + rng.normal(0, 0.01, n), rng.uniform(0, 4, n), h], 1)
        pts.append(p)
    x0, x1, y0, y1 = TABLE_BOX
    n = 5000
    pts.append(np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n),
                         0.72 + rng.normal(0, 0.005, n)], 1))
    for lx, ly in [(x0 + 0.05, y0 + 0.05), (x1 - 0.05, y0 + 0.05),
                   (x0 + 0.05, y1 - 0.05), (x1 - 0.05, y1 - 0.05)]:
        n = 700
        pts.append(np.stack([lx + rng.normal(0, 0.01, n), ly + rng.normal(0, 0.01, n),
                             rng.uniform(0, 0.72, n)], 1))
    if with_pocket:
        n = 9000
        pts.append(np.stack([POCKET_WALL_X + rng.normal(0, 0.01, n),
                             rng.uniform(0, 4, n), rng.uniform(0, 2.6, n)], 1))
    n = 120  # sparse floaters hugging the long walls, like real feed-forward mush
    pts.append(np.stack([rng.uniform(-0.2, 6.2, n),
                         rng.choice([0.1, 3.9], n) + rng.normal(0, 0.15, n),
                         rng.uniform(-0.3, 3.0, n)], 1))
    local = np.concatenate(pts, 0)

    if z_up:
        rot, t = np.eye(3), np.zeros(3)
    else:
        rot, t = _rotation_and_offset(tilt_deg)
    world = local @ rot.T + t
    up_world = rot @ np.array([0.0, 0.0, 1.0])

    from server.oreos.pathplan import camera_rotation

    poses = []
    for k in range(48):
        ang = 2 * np.pi * k / 48
        pos_local = np.array([3.0 + 1.2 * np.cos(ang), 2.0 + 0.9 * np.sin(ang), 1.4])
        fwd_local = np.array([-np.sin(ang), np.cos(ang), 0.0])
        c2w = np.eye(4)
        c2w[:3, :3] = camera_rotation(rot @ fwd_local, up_world)
        c2w[:3, 3] = rot @ pos_local + t
        poses.append(c2w)
    poses = np.stack(poses)

    def to_world(p_local):
        return np.asarray(p_local, float) @ rot.T + t

    def to_local(p_world):
        return (np.asarray(p_world, float) - t) @ rot

    return {
        "points": world, "poses": poses, "rot": rot, "t": t, "up": up_world,
        "to_world": to_world, "to_local": to_local,
    }


@pytest.fixture(scope="module")
def room():
    return make_room()


@pytest.fixture(scope="module")
def pp():
    import server.oreos.pathplan as pathplan

    return pathplan


# ---------------------------------------------------------------------------
# 1. Engine tests
# ---------------------------------------------------------------------------


def test_floor_fit_within_tolerance(pp, room):
    ctx, units, _ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    assert ctx.up_source == "poses_gravity"
    assert float(ctx.up_vec @ room["up"]) > 0.999
    assert float(ctx.plane.normal @ room["up"]) > 0.999
    # plane passes through local z≈0
    assert abs(float(room["to_local"](ctx.plane.point)[2])) < 0.03
    # capture cameras ride at local 1.4 m
    assert ctx.capture_height_slam == pytest.approx(1.4, abs=0.05)


def test_occupancy_marks_table_footprint(pp, room):
    ctx, *_ = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    x0, x1, y0, y1 = TABLE_BOX
    grid = ctx.grid
    tu, tv, _ = pp.project_to_plane(
        room["to_world"]([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])[None, :], ctx.plane
    )
    assert grid.occ[grid.cell_of(float(tu[0]), float(tv[0]))]  # table centre = obstacle
    ou, ov, _ = pp.project_to_plane(room["to_world"]([1.0, 1.0, 0.0])[None, :], ctx.plane)
    cell = grid.cell_of(float(ou[0]), float(ov[0]))
    assert not grid.occ[cell] and grid.free[cell]  # open floor = free


def test_largest_component_excludes_sealed_pocket(pp):
    fx = make_room(with_pocket=True)
    ctx, *_ = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})
    assert ctx.n_components > 1
    # a point deep in the pocket is free-but-not-navigable
    u, v, _ = pp.project_to_plane(fx["to_world"]([5.6, 2.0, 0.0])[None, :], ctx.plane)
    _ij, _uv, tree_ij = ctx.free_lookup()[0], None, None
    free_ij, _free_uv, free_tree = ctx.free_lookup()
    _d, idx = free_tree.query([float(u[0]), float(v[0])])
    iu, iv = int(free_ij[idx][0]), int(free_ij[idx][1])
    assert ctx.grid.free[iu, iv] and not ctx.navigable[iu, iv]


def test_goal_in_pocket_unreachable_with_suggestion(pp):
    fx = make_room(with_pocket=True)
    ctx, _units, ps, _pu = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})
    with pytest.raises(pp.NavError) as exc:
        pp.plan_route(ctx, fx["to_world"]([5.6, 2.0, 0.0]),
                      fx["to_world"]([0.5, 0.5, 0.0]), ps)
    assert exc.value.code == "unreachable_goal"
    sugg = exc.value.extra["nearest_reachable"]["point_world"]
    # the suggested point must itself be plannable
    res = pp.plan_route(ctx, np.asarray(sugg), fx["to_world"]([0.5, 0.5, 0.0]), ps)
    assert res.c2ws.shape[0] >= 2


def test_goal_inside_obstacle_plans_an_approach(pp, room):
    """A goal inside a thing is an APPROACH, not a failure: the robot parks beside it."""
    ctx, _units, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    x0, x1, y0, y1 = TABLE_BOX
    res = pp.plan_route(ctx, room["to_world"]([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]),
                        room["to_world"]([0.5, 0.5, 0.0]), ps)
    assert res.goal_snap_slam > 1.5 * ctx.grid.cell_size
    assert res.goal_mode == "approach"
    assert any("APPROACH" in n for n in res.notes)
    gl = room["to_local"](res.goal_world)
    assert not (x0 < gl[0] < x1 and y0 < gl[1] < y1)  # parked OUTSIDE the footprint
    # the requested point is preserved so the client can draw target-vs-standoff
    assert res.goal_requested_world is not None
    req_local = room["to_local"](res.goal_requested_world)
    assert x0 < req_local[0] < x1 and y0 < req_local[1] < y1


def test_classify_goal_cell_three_classes(pp):
    """The classifier that replaced the 'nearest FREE cell navigable?' test."""
    fx = make_room(with_pocket=True)
    ctx, *_ = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})

    def _cls(local_xy):
        u, v, _ = pp.project_to_plane(fx["to_world"]([*local_xy, 0.0])[None, :], ctx.plane)
        return pp.classify_goal_cell(ctx, float(u[0]), float(v[0]))

    assert _cls([1.0, 2.0]) == pp.GOAL_FREE_NAVIGABLE      # open floor in the main room
    assert _cls([5.6, 2.0]) == pp.GOAL_FREE_POCKET         # open floor, sealed closet
    x0, x1, y0, y1 = TABLE_BOX
    assert _cls([(x0 + x1) / 2, (y0 + y1) / 2]) == pp.GOAL_OBSTRUCTED  # inside the table
    assert _cls([500.0, 500.0]) == pp.GOAL_OBSTRUCTED  # far outside the grid


def test_object_goal_beside_a_severed_pocket_still_plans(pp):
    """THE regression (founder, 2026-07-31): an object whose nearest FREE cell falls in a
    severed pocket used to 422, even though navigable floor sat a few cells the other way.

    Goal sits INSIDE the sealed pocket wall: nearest free cell is the pocket side (0.25 m),
    nearest NAVIGABLE cell is the room side (0.35 m). The old gate looked only at the former
    and refused; a robot would simply walk to the room side."""
    fx = make_room(with_pocket=True)
    ctx, _units, ps, _pu = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})
    goal = fx["to_world"]([POCKET_WALL_X + 0.05, 2.0, 0.4])
    u, v, _ = pp.project_to_plane(goal[None, :], ctx.plane)

    # precondition: the goal is obstructed AND its nearest free cell is NOT navigable —
    # i.e. exactly the configuration the old gate turned into a hard 422.
    assert pp.classify_goal_cell(ctx, float(u[0]), float(v[0])) == pp.GOAL_OBSTRUCTED
    free_ij, _fuv, free_tree = ctx.free_lookup()
    _d, idx = free_tree.query([float(u[0]), float(v[0])])
    assert not ctx.navigable[int(free_ij[idx][0]), int(free_ij[idx][1])]

    res = pp.plan_route(ctx, goal, fx["to_world"]([0.5, 0.5, 0.0]), ps)
    assert res.goal_mode == "approach"
    assert res.c2ws.shape[0] >= 2
    # parked on the ROOM side of the wall, not inside the sealed closet
    goal_local = fx["to_local"](res.goal_world)
    assert goal_local[0] < POCKET_WALL_X


def test_obstructed_goal_beyond_approach_radius_is_honestly_unreachable(pp):
    """The approach budget is not a licence to snap anywhere: past it we still refuse,
    with the nearest-reachable hint the client offers as a one-click retry."""
    fx = make_room(with_pocket=True)
    ctx, _units, ps, _pu = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})
    tight = dict(ps, approach_radius=0.02)  # ~2 cm of standoff allowed
    x0, x1, y0, y1 = TABLE_BOX
    with pytest.raises(pp.NavError) as exc:
        pp.plan_route(ctx, fx["to_world"]([(x0 + x1) / 2, (y0 + y1) / 2, 0.4]),
                      fx["to_world"]([0.5, 0.5, 0.0]), tight)
    assert exc.value.code == "unreachable_goal"
    assert exc.value.extra["goal_class"] == pp.GOAL_OBSTRUCTED
    sugg = exc.value.extra["nearest_reachable"]["point_world"]
    # the suggestion must itself be plannable (the one-click-retry contract)
    assert pp.plan_route(ctx, np.asarray(sugg), fx["to_world"]([0.5, 0.5, 0.0]),
                         ps).c2ws.shape[0] >= 2


def test_goal_footprint_widens_the_approach_budget(pp):
    """A big object earns more standoff than a small one at the same approach radius."""
    fx = make_room(with_pocket=True)
    ctx, _units, ps, _pu = pp.build_nav_context(fx["points"], fx["poses"], {"seed": 0})
    x0, x1, y0, y1 = TABLE_BOX
    goal = fx["to_world"]([(x0 + x1) / 2, (y0 + y1) / 2, 0.4])
    start = fx["to_world"]([0.5, 0.5, 0.0])
    tight = dict(ps, approach_radius=0.02)
    with pytest.raises(pp.NavError):
        pp.plan_route(ctx, goal, start, tight)
    # same radius, but the object declares its own half-size → the approach is allowed
    res = pp.plan_route(ctx, goal, start, tight, goal_footprint_slam=2.0)
    assert res.goal_mode == "approach"
    assert res.approach_radius_slam > 1.9


def test_path_avoids_obstacles_and_stays_on_floor(pp, room):
    ctx, _units, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    res = pp.plan_route(ctx, room["to_world"]([5.5, 3.5, 0.0]),
                        room["to_world"]([0.5, 0.5, 0.0]), ps)
    u, v, h = pp.project_to_plane(res.waypoints_world, ctx.plane)
    assert float(np.abs(h).max()) < 1e-9  # waypoints ON the plane
    assert all(ctx.grid.free[ctx.grid.cell_of(uu, vv)] for uu, vv in zip(u, v))
    wl = np.stack([room["to_local"](w) for w in res.waypoints_world])
    x0, x1, y0, y1 = TABLE_BOX
    inside = (wl[:, 0] > x0) & (wl[:, 0] < x1) & (wl[:, 1] > y0) & (wl[:, 1] < y1)
    assert int(inside.sum()) == 0


def test_units_ladder_three_rungs(pp, room):
    # rung 1: anchored → metres
    _ctx, units, ps, _pu = pp.build_nav_context(
        room["points"], room["poses"], {"seed": 0},
        anchor_scale_factor=0.5, anchor_key="derived/anchor/s/cloud.npz",
    )
    assert units.units == "m"
    assert units.units_basis == "anchor:derived/anchor/s/cloud.npz"
    # 0.3 m clearance at 0.5 m/slam-unit → 0.6 slam units
    assert ps["clearance"] == pytest.approx(0.6, rel=1e-6)

    # rung 2: unanchored + poses → capture-height fractions
    _ctx, units, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    assert (units.units, units.units_basis) == ("relative", "capture_height_fraction")
    # capture height ≈1.4 local-metres == nominal → defaults land ≈ metric
    assert ps["clearance"] == pytest.approx(0.3, rel=0.05)

    # rung 3: unanchored, no poses (override supplies the axis) → extent fractions
    _ctx, units, _ps, _pu = pp.build_nav_context(
        room["points"], None, {"seed": 0, "up_override": "-y"}
    )
    assert (units.units, units.units_basis) == ("relative", "extent_fraction")


def test_up_ladder_heuristic_and_override(pp, room):
    # no poses, no override → unsigned extent heuristic + the loud note
    ctx, *_ = pp.build_nav_context(room["points"], None, {"seed": 0})
    assert ctx.up_source == "extent_heuristic_unsigned"
    assert any("SIGN defaults to +1" in n for n in ctx.notes)
    # override -y recovers the true (tilted!) floor via the shrinking-threshold refit
    ctx2, *_ = pp.build_nav_context(room["points"], None, {"seed": 0, "up_override": "-y"})
    assert ctx2.up_source == "override:-y"
    assert float(ctx2.plane.normal @ room["up"]) > 0.999
    # vector override accepted too
    ctx3, *_ = pp.build_nav_context(
        room["points"], None, {"seed": 0, "up_override": [float(x) for x in room["up"]]}
    )
    assert ctx3.up_source.startswith("override:vec[")
    with pytest.raises(pp.NavError) as exc:
        pp.build_nav_context(room["points"], None, {"seed": 0, "up_override": "sideways"})
    assert exc.value.code == "bad_request"


def test_camera_below_floor_sanity_trips_no_floor(pp, room):
    # An upside-down rig: flip each pose's down column so gravity estimates point
    # AWAY from the true floor → the fit lands somewhere the cameras are below.
    bad = room["poses"].copy()
    bad[:, :3, 1] *= -1.0
    with pytest.raises(pp.NavError) as exc:
        pp.build_nav_context(room["points"], bad, {"seed": 0})
    assert exc.value.code == "no_floor"


def test_determinism_same_seed_identical_waypoints(pp, room):
    goal = room["to_world"]([5.5, 3.5, 0.0])
    start = room["to_world"]([0.5, 0.5, 0.0])
    outs = []
    for _ in range(2):
        ctx, _u, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 3})
        res = pp.plan_route(ctx, goal, start, ps)
        outs.append(res)
    assert np.array_equal(outs[0].waypoints_world, outs[1].waypoints_world)
    assert np.array_equal(outs[0].c2ws, outs[1].c2ws)


def test_poses_follow_opencv_contract(pp, room):
    ctx, _u, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    res = pp.plan_route(ctx, room["to_world"]([5.5, 3.5, 0.0]),
                        room["to_world"]([0.5, 0.5, 0.0]), ps)
    rots = res.c2ws[:, :3, :3]
    # orthonormal, right-handed
    assert np.allclose(np.einsum("nij,nik->njk", rots, rots), np.eye(3)[None], atol=1e-8)
    assert np.allclose(np.linalg.det(rots), 1.0, atol=1e-8)
    # down column anti-parallel to up; forward column horizontal
    assert float((rots[:, :, 1] @ ctx.plane.normal).max()) < -0.999
    assert float(np.abs(rots[:, :, 2] @ ctx.plane.normal).max()) < 1e-6
    # camera rides at eye height above the floor
    cam_h = (res.c2ws[:, :3, 3] - ctx.plane.point) @ ctx.plane.normal
    assert np.allclose(cam_h, ps["eye_height"], atol=1e-9)
    # timestamps at 1/fps
    assert np.allclose(np.diff(res.times), 1.0 / ps["fps"], atol=1e-12)
    # heading rate limit honored
    fw = rots[:, :, 2]
    dots = np.clip(np.einsum("ni,ni->n", fw[:-1], fw[1:]), -1.0, 1.0)
    max_turn = np.degrees(np.arccos(dots)).max()
    assert max_turn <= ps["max_ang_vel_deg"] / ps["fps"] + 1e-6


def test_param_validation_rejects_garbage(pp, room):
    cases = [
        ({"seed": 0, "clearance": -0.1}, "clearance"),
        ({"seed": 0, "band_lo": 1.0, "band_hi": 0.5}, "band_hi"),
        ({"seed": 0, "fps": 0}, "fps"),
        ({"seed": 0, "speed": 0}, "speed"),
        ({"seed": 0, "typo_key": 1}, "typo_key"),
        ({"seed": -1}, "seed"),
        ({"seed": 0, "eye_height": float("nan")}, "eye_height"),
    ]
    for params, needle in cases:
        with pytest.raises(pp.NavError) as exc:
            pp.build_nav_context(room["points"], room["poses"], params)
        assert exc.value.code == "bad_request"
        assert needle in exc.value.detail


def test_zero_free_auto_retry_labeled(pp):
    # Bare floor+walls room (no table/floaters): the largest obstacle-free disk has
    # radius ≈2.0 at the room centre, so clearance 3.0 wipes free space and the F4
    # ladder's single labeled retry at half clearance (1.5) recovers it. Driven at the
    # attach_grid level with SLAM-unit params so no units ladder blurs the margins.
    rng = np.random.default_rng(1)
    pts = [np.stack([rng.uniform(0, 6, 20000), rng.uniform(0, 4, 20000),
                     rng.normal(0, 0.01, 20000)], 1)]
    for wall in range(4):
        n, h = 5000, rng.uniform(0, 2.6, 5000)
        if wall == 0:
            pts.append(np.stack([rng.uniform(0, 6, n), rng.normal(0, 0.01, n), h], 1))
        elif wall == 1:
            pts.append(np.stack([rng.uniform(0, 6, n), 4.0 + rng.normal(0, 0.01, n), h], 1))
        elif wall == 2:
            pts.append(np.stack([rng.normal(0, 0.01, n), rng.uniform(0, 4, n), h], 1))
        else:
            pts.append(np.stack([6.0 + rng.normal(0, 0.01, n), rng.uniform(0, 4, n), h], 1))
    bare = np.concatenate(pts, 0)
    geom = pp.fit_context_geometry(bare, None, seed=0, up_override="+z")
    params_slam = {"cell_size": 0.05, "band_lo": 0.2, "band_hi": 1.5, "clearance": 3.0,
                   "floor_fill_radius": 0.15, "known_height_max": 3.0}
    ctx = pp.attach_grid(geom, params_slam)
    assert int(ctx.grid.free.sum()) > 0
    assert any("auto-retried once at half clearance" in n for n in ctx.notes)
    assert ctx.grid_params_slam["clearance"] == pytest.approx(1.5, rel=1e-6)
    # and a clearance no retry can save still fails honestly
    with pytest.raises(pp.NavError) as exc:
        pp.attach_grid(geom, {**params_slam, "clearance": 8.0})
    assert exc.value.code == "no_floor"


def test_grid_cache_roundtrip_and_rebuild(pp, room):
    ctx, _u, ps, _pu = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    blob = pp.serialize_nav_context(ctx, "scanX")
    back = pp.deserialize_nav_context(blob)
    assert back is not None
    assert np.array_equal(back.grid.free, ctx.grid.free)
    assert np.array_equal(back.navigable, ctx.navigable)
    assert back.up_source == ctx.up_source
    assert back.capture_height_slam == pytest.approx(ctx.capture_height_slam)
    assert back.last_cam_world == ctx.last_cam_world
    # same plan through the deserialized context
    goal = room["to_world"]([5.5, 3.5, 0.0]); start = room["to_world"]([0.5, 0.5, 0.0])
    a = pp.plan_route(ctx, goal, start, ps)
    b = pp.plan_route(back, goal, start, ps)
    assert np.array_equal(a.waypoints_world, b.waypoints_world)
    # corrupt blob degrades to None (recompute), never raises
    assert pp.deserialize_nav_context(b"not an npz") is None
    # grid-param change rebuilds only the grid, keeping the plane
    ps2 = dict(ps); ps2["clearance"] = ps["clearance"] * 0.5
    rebuilt = pp.rebuild_grid(back, ps2)
    assert np.array_equal(rebuilt.plane.normal, back.plane.normal)
    assert int(rebuilt.grid.free.sum()) > int(back.grid.free.sum())


def test_context_reusable_ladder(pp, room):
    ctx, *_ = pp.build_nav_context(room["points"], room["poses"], {"seed": 0})
    assert pp.context_reusable(ctx, 0, None, has_poses=True)
    assert not pp.context_reusable(ctx, 1, None, has_poses=True)  # new seed → refit
    assert not pp.context_reusable(ctx, 0, "-y", has_poses=True)  # override requested
    hctx, *_ = pp.build_nav_context(room["points"], None, {"seed": 0, "up_override": "-y"})
    assert pp.context_reusable(hctx, 0, "-y", has_poses=False)
    assert not pp.context_reusable(hctx, 0, None, has_poses=False)  # override dropped
    # heuristic cache upgrades when a trajectory appears
    zx = make_room(z_up=True)
    zctx, *_ = pp.build_nav_context(zx["points"], None, {"seed": 0})
    assert zctx.up_source == "extent_heuristic_unsigned"
    assert pp.context_reusable(zctx, 0, None, has_poses=False)
    assert not pp.context_reusable(zctx, 0, None, has_poses=True)


def test_voxel_downsample_deterministic_and_capped(pp):
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(50000, 3))
    a, va = pp.voxel_downsample(pts, 8000)
    b, vb = pp.voxel_downsample(pts, 8000)
    assert a.shape[0] <= 8000 and a.shape == b.shape and np.array_equal(a, b)
    assert va == vb and va > 0
    small, vs = pp.voxel_downsample(pts[:100], 8000)
    assert small.shape[0] == 100 and vs == 0.0


def test_rotation_to_quat_roundtrip(pp):
    rng = np.random.default_rng(4)
    for _ in range(25):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        ang = rng.uniform(-np.pi, np.pi)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        w, x, y, z = pp.rotation_to_quat(R)
        # rebuild and compare
        R2 = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        assert np.allclose(R, R2, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Route-contract tests (real Flask + stubbed server.app)
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def nav_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)
    nav_mod = sys.modules["server.oreos.routes_nav"]
    nav_mod.reset_nav_cache()
    yield app.test_client(), store, stub, nav_mod
    nav_mod.reset_nav_cache()


def _save_nav_scene(store, fx, user="user-a", scan="scan1", with_cloud=True,
                    with_trajectory=True, objects=None, **kwargs):
    n = fx["points"].shape[0]
    facts = SceneFacts()
    if objects:
        facts = SceneFacts(objects=objects)
    store.save_scene(
        user,
        scan,
        SceneReport(summary="s", room_type="office"),
        facts,
        points=(fx["points"].astype(np.float32), np.zeros((n, 3), np.uint8)) if with_cloud else None,
        trajectory=(
            {
                "poses": fx["poses"].astype(np.float32),
                "intrinsics": np.tile([500.0, 500.0, 320.0, 240.0], (fx["poses"].shape[0], 1)),
                "source_frame_id": np.arange(fx["poses"].shape[0], dtype=np.float32),
            }
            if with_trajectory
            else None
        ),
        **kwargs,
    )


@pytest.fixture(scope="module")
def route_room():
    return make_room(n_floor=15000)  # lighter cloud keeps the route suite quick


def test_route_auth_and_shape_contract(nav_app, route_room):
    client, store, stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}

    stub._auth_user_id = lambda: None
    assert client.post("/api/scenes/scan1/nav/plan", json=goal).status_code == 401
    stub._auth_user_id = lambda: "user-a"

    assert client.post("/api/scenes/nope/nav/plan", json=goal).status_code == 404
    r = client.post("/api/scenes/scan1/nav/plan", data="[]", content_type="application/json")
    assert r.status_code == 400
    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"point_world": [1, 2]}})
    assert r.status_code == 400 and r.get_json()["error"] == "bad_request"
    r = client.post("/api/scenes/scan1/nav/plan", json={})
    assert r.status_code == 400


def test_route_happy_path_response_contract(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    goal_w = [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]
    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"point_world": goal_w}})
    assert r.status_code == 200
    body = r.get_json()
    assert body["provenance"] == (
        "Planned in scanned free space of the point cloud — planner visualization, "
        "not certified navigation"
    )
    assert body["units"] == "relative" and body["units_basis"] == "capture_height_fraction"
    assert body["parent_artifact"] == "cloud.npz"
    assert set(body["grid"]) >= {"n_free", "n_components", "cell_size"}
    assert body["grid"]["n_free"] > 0
    assert len(body["waypoints_world"]) == body["stats"]["n_frames"] == len(body["poses"])
    assert body["poses"][0]["t"] == 0.0 and len(body["poses"][0]["c2w"]) == 4
    assert body["floor"]["up_source"] == "poses_gravity"
    assert body["start_source"] == "camera_below"
    assert body["cache"]["source"] == "computed"
    assert body["path_id"].startswith("p") and body["doc_key"]


def test_route_object_goal_and_unknown_object(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    x0, x1, y0, y1 = TABLE_BOX
    table_center = [float(v) for v in route_room["to_world"]([(x0 + x1) / 2, (y0 + y1) / 2, 0.4])]
    objects = [{"query": "table", "center": table_center, "confidence": 0.9}]
    _save_nav_scene(store, route_room, objects=objects)

    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"object_uid": "det:0"}})
    assert r.status_code == 200
    body = r.get_json()
    assert body["goal_requested"]["object_uid"] == "det:0"
    assert body["goal_requested"]["label"] == "table"
    # the object's centre is inside the table → the plan is an APPROACH to reachable floor
    assert any("APPROACH" in n for n in body["notes"])
    approach = body["goal_approach"]
    assert approach["mode"] == "approach"
    assert approach["standoff"] > 0
    assert approach["requested_point_world"] == pytest.approx(table_center, abs=1e-6)
    # the parked point is NOT the requested point (the robot stops beside the table)
    assert body["goal_world"] != approach["requested_point_world"]

    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"object_uid": "det:99"}})
    assert r.status_code == 404 and r.get_json()["error"] == "unknown_object"
    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"object_uid": "sel:abc"}})
    assert r.status_code == 400  # sel:/layer: arrive with the F3 store


def test_route_object_goal_on_furniture_no_longer_422s(nav_app):
    """Founder-reported 2026-07-31: "HTTP 422 when trying to navigate a path to an object".

    Measured against the real scenes before the fix: 11/95 object goals on
    canonical-office-loop and 133/184 on the founder's own 63.3M-point scene returned
    HTTP 422 ``unreachable_goal``. Every one of them sat inside its own object's occupied
    volume, where the old gate consulted the nearest FREE cell — routinely a severed pocket
    carved by that very object. This is the route-level guard for that whole class."""
    client, store, _stub, _nav = nav_app
    fx = make_room(with_pocket=True, n_floor=15000)
    # an object mounted ON the sealed pocket wall — nearest free cell is inside the closet
    obj_center = [float(v) for v in fx["to_world"]([POCKET_WALL_X + 0.05, 2.0, 0.9])]
    _save_nav_scene(store, fx, objects=[{"query": "monitor", "center": obj_center,
                                        "extent": [0.5, 0.1, 0.3], "confidence": 0.9}])

    r = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"object_uid": "det:0"}})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["goal_approach"]["mode"] == "approach"
    assert body["stats"]["n_frames"] >= 2
    # the object's own size widened the budget it was granted
    assert body["goal_requested"]["footprint_radius_slam"] > 0


def test_route_unreachable_goal_body_carries_an_actionable_retry(nav_app):
    """When we DO refuse, the body must stay actionable: an honest reason plus a
    nearest_reachable point the client offers as one-click retry."""
    client, store, _stub, _nav = nav_app
    fx = make_room(with_pocket=True, n_floor=15000)
    _save_nav_scene(store, fx)
    deep_in_pocket = [float(v) for v in fx["to_world"]([5.6, 2.0, 0.0])]

    r = client.post("/api/scenes/scan1/nav/plan",
                    json={"goal": {"point_world": deep_in_pocket}})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "unreachable_goal"
    assert body["goal_class"] == "free_pocket"  # the genuine EXP-25 severed-island case
    assert isinstance(body["detail"], str) and len(body["detail"]) > 20
    retry = body["nearest_reachable"]["point_world"]
    assert len(retry) == 3
    # the suggested point plans successfully — the retry button is not a dead end
    r2 = client.post("/api/scenes/scan1/nav/plan", json={"goal": {"point_world": retry}})
    assert r2.status_code == 200, r2.get_json()


def test_route_approach_radius_param_is_accepted_and_reported(nav_app, route_room):
    """The new knob round-trips through the units ladder into params + the path doc."""
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    r = client.post("/api/scenes/scan1/nav/plan",
                    json={"goal": {"point_world": [float(v) for v in
                                                   route_room["to_world"]([5.0, 3.0, 0.0])]},
                          "params": {"approach_radius": 1.25}})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["params"]["approach_radius"] == pytest.approx(1.25, rel=1e-6)
    r = client.post("/api/scenes/scan1/nav/plan",
                    json={"goal": {"point_world": [0.0, 0.0, 0.0]},
                          "params": {"approach_radius": -1.0}})
    assert r.status_code == 400


def test_route_no_geometry_404(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room, with_cloud=False, with_trajectory=False)
    r = client.post(
        "/api/scenes/scan1/nav/plan", json={"goal": {"point_world": [0.0, 0.0, 0.0]}}
    )
    assert r.status_code == 404
    assert r.get_json()["error"] == "no_geometry"


def test_route_cache_reuse_and_grid_rebuild(nav_app, route_room, monkeypatch):
    client, store, _stub, nav_mod = nav_app
    _save_nav_scene(store, route_room)
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}

    r1 = client.post("/api/scenes/scan1/nav/plan", json=goal)
    assert r1.status_code == 200 and r1.get_json()["cache"]["source"] == "computed"
    # the durable cache artifact exists
    assert store.get_derived_artifact_path(
        "user-a", "scan1", "derived/demo/nav/grid_cache.npz"
    )

    # second call must NOT refit: make the fit explode if touched
    import server.oreos.pathplan as pp_live

    def _boom(*a, **k):
        raise AssertionError("floor fit re-ran on a cache hit")

    monkeypatch.setattr(pp_live, "fit_floor_plane", _boom)
    r2 = client.post("/api/scenes/scan1/nav/plan", json=goal)
    assert r2.status_code == 200 and r2.get_json()["cache"]["source"] == "memory"
    assert np.allclose(r2.get_json()["waypoints_world"], r1.get_json()["waypoints_world"])

    # volume rung: drop the in-process cache, keep the fit booby-trapped
    nav_mod.reset_nav_cache()
    r3 = client.post("/api/scenes/scan1/nav/plan", json=goal)
    assert r3.status_code == 200 and r3.get_json()["cache"]["source"] == "volume"

    # param change → grid-only rebuild (still no refit)
    r4 = client.post(
        "/api/scenes/scan1/nav/plan",
        json={**goal, "params": {"clearance": 0.15}},
    )
    assert r4.status_code == 200
    b4 = r4.get_json()
    assert b4["cache"]["grid_rebuilt"] is True
    assert b4["params"]["clearance"] == pytest.approx(0.15)
    assert b4["grid"]["n_free"] > r1.get_json()["grid"]["n_free"]  # thinner inflation


def test_route_seed_change_refits(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}
    assert client.post("/api/scenes/scan1/nav/plan", json=goal).status_code == 200
    r = client.post("/api/scenes/scan1/nav/plan", json={**goal, "params": {"seed": 7}})
    assert r.status_code == 200 and r.get_json()["cache"]["source"] == "computed"


def test_route_path_doc_persisted_and_indexed(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}
    body = client.post("/api/scenes/scan1/nav/plan", json=goal).get_json()
    doc_key = body["doc_key"]
    assert doc_key == f"derived/demo/nav/paths/{body['path_id']}.json"

    raw = store.get_derived_artifact("user-a", "scan1", doc_key)
    doc = json.loads(raw.decode("utf-8"))
    assert doc["version"] == 1
    assert doc["planner"]["engine"] == "exp25-port"
    assert doc["planner"]["seed"] == 0
    assert doc["parent_artifact"] == "cloud.npz"
    assert doc["units"] == body["units"] and doc["units_basis"] == body["units_basis"]
    assert len(doc["frames"]) == body["stats"]["n_frames"]
    frame = doc["frames"][0]
    assert set(frame) == {"t", "pos", "quat"} and len(frame["quat"]) == 4
    assert doc["provenance"] == body["provenance"]

    manifest = client.get("/api/scenes/scan1/demo/manifest").get_json()
    assert doc_key in manifest["paths"] and doc_key in manifest["docs"]


def test_route_debug_png_artifact(nav_app, route_room):
    # Must probe the SUBMODULE: export_fakes parks a non-package "matplotlib"
    # stub in sys.modules when the real one is absent, so the bare name imports.
    pytest.importorskip("matplotlib.pyplot")
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}
    body = client.post("/api/scenes/scan1/nav/plan?debug=1", json=goal).get_json()
    assert body["debug_key"] == f"derived/demo/nav/paths/{body['path_id']}_debug.png"
    png = store.get_derived_artifact("user-a", "scan1", body["debug_key"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _gaussian_splat_ply_bytes(positions: np.ndarray) -> bytes:
    """Minimal binary-LE 3DGS ply (x,y,z,f_dc_0,opacity) for the import fallback test."""
    n = positions.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in ("x", "y", "z", "f_dc_0", "opacity"))
        + "end_header\n"
    ).encode("ascii")
    body = np.zeros((n, 5), dtype="<f4")
    body[:, :3] = positions.astype(np.float32)
    return header + body.tobytes()


def test_route_imported_splat_means_fallback(nav_app):
    client, store, _stub, _nav = nav_app
    fx = make_room(z_up=True, n_floor=15000)  # imports have no poses; z-up keeps heuristic sane
    _save_nav_scene(
        store, fx, with_cloud=False, with_trajectory=False,
        splat_bytes=_gaussian_splat_ply_bytes(fx["points"]),
        source="imported_splat",
    )
    goal = {"goal": {"point_world": [float(x) for x in fx["to_world"]([5.5, 3.5, 0.0])]}}
    r = client.post("/api/scenes/scan1/nav/plan", json=goal)
    assert r.status_code == 200
    body = r.get_json()
    assert body["parent_artifact"] == "splat.ply"
    assert body["units_basis"] == "extent_fraction"
    assert body["floor"]["up_source"] == "extent_heuristic_unsigned"
    assert body["start_source"] == "free_area_center"
    # F3-shared means cache written
    assert store.get_derived_artifact_path(
        "user-a", "scan1", "derived/demo/nav/cloud_from_splat.npz"
    )


def test_route_anchored_scene_reports_metres(nav_app, route_room):
    client, store, _stub, _nav = nav_app
    _save_nav_scene(store, route_room)
    store.set_derived_pointer(
        "user-a", "scan1",
        {"kind": "anchor", "source_key": "derived/anchor/s1/cloud.npz", "scale_factor": 0.5},
    )
    goal = {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}}
    body = client.post("/api/scenes/scan1/nav/plan", json=goal).get_json()
    assert body["units"] == "m"
    assert body["units_basis"] == "anchor:derived/anchor/s1/cloud.npz"
    # reported length = slam length × scale_factor
    assert body["stats"]["path_length"] == pytest.approx(
        body["stats"]["path_length_slam"] * 0.5, rel=1e-9
    )
    # anchoring AFTER a cached plan must flip units without a refit
    store.set_derived_pointer("user-a", "scan1", None)
    body2 = client.post("/api/scenes/scan1/nav/plan", json=goal).get_json()
    assert body2["units"] == "relative"


def test_tool_seam_plan_path_for_scene(nav_app, route_room):
    """W2's plan_path tool calls this exact function — no flask objects involved."""
    _client, store, _stub, nav_mod = nav_app
    _save_nav_scene(store, route_room)
    payload, status = nav_mod.plan_path_for_scene(
        "user-a", "scan1",
        {"goal": {"point_world": [float(x) for x in route_room["to_world"]([5.5, 3.5, 0.0])]}},
        store=store,
    )
    assert status == 200
    assert payload["provenance"].startswith("Planned in scanned free space")
    payload2, status2 = nav_mod.plan_path_for_scene(
        "user-a", "missing", {"goal": {"point_world": [0, 0, 0]}}, store=store
    )
    assert status2 == 404 and payload2["error"] == "not_found"
