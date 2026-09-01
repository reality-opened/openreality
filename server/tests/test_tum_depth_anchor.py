"""Depth→metric-scale on the TUM RGB-D benchmark — GPU-free unit tests.

Covers the two modules the TUM run added:

  * ``server/oreos/depth_ratio.py`` — the sensor-agnostic ratio core lifted out of
    ``scripts/demo_ingest_gemini.py``. Its geometry/gate functions keep their existing
    coverage through ``tests/test_oreos_gemini_ingest.py`` (which exercises them through the
    Gemini CLI's re-exports, so a regression there fails BOTH suites). What is tested here
    is what the move ADDED: ``pair_ratio`` end-to-end on a synthetic scene with a known
    ratio, exhaustive NCC matching, and the admitted→fed index recovery.
  * ``server/oreos/tum_rgbd.py`` — the dataset adapter: index/ground-truth parsing,
    nearest-timestamp association, trajectory length, Umeyama Sim(3), and the
    reconstruction-grid intrinsics rescale.

Core-dependent tests skip cleanly when neither an installed ``openreality-core`` nor a
platform-tree ``core/`` checkout is reachable — same posture as the export tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load ``server/oreos/<name>.py`` BY FILE PATH, under a private module name.

    Not ``import server.oreos.<name>``: that runs ``server/oreos/__init__.py``, which registers
    the Flask blueprint and imports every route module in the repo. And NOT the Modal
    scripts' ``sys.modules`` stub trick either — registering stub ``server``/``server.oreos``
    packages at test-collection time is a process-global side effect that changes how the
    REST of the suite imports (it measurably moved other modules' skip decisions). Both target
    modules are self-contained numpy, so a file load is exact and inert."""
    path = _REPO / "server" / "oreos" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tumtest_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dr = _load("depth_ratio")
tum = _load("tum_rgbd")


@pytest.fixture(scope="module")
def ma():
    try:
        return dr.load_metric_anchor_module()
    except ImportError as exc:  # pragma: no cover - CI without core
        pytest.skip(f"core metric_anchor unavailable: {exc}")


# ---------------------------------------------------------------------------
# depth_ratio: pair_ratio recovers a known scale
# ---------------------------------------------------------------------------

_CAM = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0,
        "width": 320, "height": 240, "rotation": None}


def _synthetic_surface(camera=_CAM):
    """A dense, SMOOTH surface filling the camera's view, in SLAM units, camera frame.

    A surface and not a random volume on purpose: the two sides of the ratio coarsen
    differently (the SLAM side is a nearest-z rasterization over each block, the depth side
    a block median), so on a scattered volume they measure different things and the ratio is
    biased by the within-block depth spread. Real scenes are surfaces; a fixture that is not
    would be testing an artefact of the fixture."""
    u = np.arange(camera["width"], dtype=np.float64)
    v = np.arange(camera["height"], dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    z = 2.0 + 0.2 * np.sin(uu / 120.0) + 0.15 * np.cos(vv / 90.0)
    x = (uu + 0.5 - camera["cx"]) * z / camera["fx"]
    y = (vv + 0.5 - camera["cy"]) * z / camera["fy"]
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _depth_png_from(points_cam, camera, true_ratio):
    """The hardware depth PNG a sensor WOULD have produced for this camera-frame cloud at
    ``true_ratio`` metres per SLAM unit — full resolution, TUM's 5000 units per metre."""
    depth, _valid = dr.zbuffer_project(
        points_cam, camera["fx"], camera["fy"], camera["cx"], camera["cy"],
        camera["width"], camera["height"], 1)
    metres = depth * true_ratio
    png = np.where(np.isfinite(metres), np.nan_to_num(metres) * tum.DEPTH_UNITS_PER_METRE, 0.0)
    return png.astype(np.uint16)


def _place(points_cam, c2w):
    """Camera-frame points → world, given the camera's ``c2w``."""
    return points_cam @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3]


def test_pair_ratio_recovers_a_known_metre_per_slam_unit(ma):
    true_ratio = 0.37
    cam_points = _synthetic_surface()
    png = _depth_png_from(cam_points, _CAM, true_ratio)
    diag = dr.pair_ratio(cam_points, np.eye(4), _CAM, png, ma,
                         downsample=4, max_depth_m=10.0, mm_per_unit=tum.MM_PER_UNIT)
    assert diag is not None
    assert diag["ratio"] == pytest.approx(true_ratio, rel=0.005)
    assert diag["n"] > 200
    assert diag["metric_med_m"] == pytest.approx(diag["slam_med"] * true_ratio, rel=0.01)


def test_pair_ratio_honours_the_pose(ma):
    """A rotated + translated camera must recover the SAME ratio — the c2w inversion is real,
    and a pose that was ignored would put the cloud behind the camera and yield nothing."""
    true_ratio = 0.37
    cam_points = _synthetic_surface()
    angle = 0.4
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(angle), 0.0, np.cos(angle)]])
    c2w[:3, 3] = [1.5, -0.7, 2.0]
    world = _place(cam_points, c2w)
    png = _depth_png_from(cam_points, _CAM, true_ratio)
    diag = dr.pair_ratio(world, c2w, _CAM, png, ma,
                         downsample=4, max_depth_m=10.0, mm_per_unit=tum.MM_PER_UNIT)
    assert diag is not None
    assert diag["ratio"] == pytest.approx(true_ratio, rel=0.005)


def test_pair_ratio_returns_none_without_overlap(ma):
    cam_points = _synthetic_surface()
    png = np.zeros((240, 320), dtype=np.uint16)  # every pixel invalid
    assert dr.pair_ratio(cam_points, np.eye(4), _CAM, png, ma, downsample=4,
                         max_depth_m=10.0, mm_per_unit=tum.MM_PER_UNIT) is None


# ---------------------------------------------------------------------------
# depth_ratio: exhaustive matching + admitted→fed recovery
# ---------------------------------------------------------------------------


def test_best_ncc_match_finds_the_exact_frame():
    rng = np.random.default_rng(3)
    cands = rng.normal(size=(50, 8, 8))
    hit = dr.best_ncc_match(cands[17], cands)
    assert hit is not None and hit[0] == 17 and hit[1] == pytest.approx(1.0)


def test_best_ncc_match_maps_through_supplied_indices():
    rng = np.random.default_rng(4)
    cands = rng.normal(size=(5, 8, 8))
    hit = dr.best_ncc_match(cands[2], cands, indices=[100, 200, 300, 400, 500])
    assert hit is not None and hit[0] == 300


def test_best_ncc_match_refuses_below_threshold():
    rng = np.random.default_rng(5)
    assert dr.best_ncc_match(rng.normal(size=(8, 8)), rng.normal(size=(30, 8, 8)),
                             min_corr=0.99) is None


def test_fed_index_recovery_is_exact_between_equal_anchors():
    """Between two verified pairs with the same rejection count, monotonicity PROVES the
    rows in between — nothing is interpolated."""
    fed, exact = dr.fed_index_for_admitted([0, 1, 2, 3, 4, 5], [(0, 0), (3, 3), (5, 7)])
    assert fed.tolist() == [0, 1, 2, 3, 4, 7]
    assert exact.tolist() == [True, True, True, True, False, True]


def test_fed_index_recovery_rejects_a_decreasing_delta():
    with pytest.raises(ValueError, match="non-decreasing"):
        dr.fed_index_for_admitted([0, 5], [(0, 4), (5, 5)])


def test_fed_index_recovery_needs_an_anchor():
    with pytest.raises(ValueError, match="anchor pair"):
        dr.fed_index_for_admitted([0, 1], [])


# ---------------------------------------------------------------------------
# tum_rgbd: parsing + association
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_read_index_skips_comments_and_keeps_order(tmp_path):
    p = _write(tmp_path, "rgb.txt",
               "# color images\n# file: x.bag\n# timestamp filename\n"
               "1305031910.765238 rgb/1305031910.765238.png\n"
               "1305031910.797230 rgb/1305031910.797230.png\n\n")
    ts, files = tum.read_index(p)
    assert ts.tolist() == [1305031910.765238, 1305031910.797230]
    assert files == ["rgb/1305031910.765238.png", "rgb/1305031910.797230.png"]


def test_read_groundtruth_shapes(tmp_path):
    p = _write(tmp_path, "gt.txt",
               "# ground truth\n1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n")
    t, pos, quat = tum.read_groundtruth(p)
    assert t.tolist() == [1.0, 2.0]
    assert pos.shape == (2, 3) and quat.shape == (2, 4)
    assert pos[1].tolist() == [1.0, 0.0, 0.0]


def test_read_groundtruth_rejects_an_empty_file(tmp_path):
    with pytest.raises(ValueError, match="no ground-truth rows"):
        tum.read_groundtruth(_write(tmp_path, "gt.txt", "# only a comment\n"))


def test_associate_nearest_picks_the_closer_side_and_drops_far_ones():
    idx, gap = tum.associate_nearest([1.00, 2.00, 9.00], [0.99, 2.03, 5.00])
    assert idx.tolist() == [0, -1, -1]  # 2.00 vs 2.03 is 30 ms > the 20 ms default
    assert gap[0] == pytest.approx(0.01)


def test_associate_nearest_respects_a_wider_tolerance():
    idx, _gap = tum.associate_nearest([2.00], [0.99, 2.03, 5.00], max_diff=0.05)
    assert idx.tolist() == [1]


def test_associate_nearest_handles_an_empty_target():
    idx, gap = tum.associate_nearest([1.0], [])
    assert idx.tolist() == [-1] and not np.isfinite(gap[0])


# ---------------------------------------------------------------------------
# tum_rgbd: trajectory metrics
# ---------------------------------------------------------------------------


def test_polyline_length_is_the_segment_sum():
    assert tum.polyline_length([[0, 0, 0], [3, 4, 0], [3, 4, 12]]) == pytest.approx(17.0)
    assert tum.polyline_length([[1, 1, 1]]) == 0.0


def test_umeyama_recovers_an_exact_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    dst = 2.5 * (src @ rot.T) + np.array([1.0, 2.0, 3.0])
    fit = tum.umeyama_sim3(src, dst)
    assert fit["scale"] == pytest.approx(2.5)
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fit["R"], rot)


def test_umeyama_never_returns_a_reflection():
    """A mirrored target must be fitted with a proper rotation (det +1) and a big residual,
    not silently 'solved' by flipping handedness."""
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    dst = src * np.array([1.0, 1.0, -1.0])
    fit = tum.umeyama_sim3(src, dst)
    assert np.linalg.det(fit["R"]) == pytest.approx(1.0)
    assert fit["rmse"] > 0.1


def test_umeyama_needs_three_pairs():
    with pytest.raises(ValueError, match=">=3"):
        tum.umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))


def test_interpolate_positions_is_linear_and_covers_checks_the_span():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    out = tum.interpolate_positions(t, pos, [0.5, 1.5])
    assert np.allclose(out, [[0.5, 0, 0], [1, 0.5, 0]])
    assert tum.covers(t, [0.0, 2.0])
    assert not tum.covers(t, [-0.1, 1.0])


# ---------------------------------------------------------------------------
# tum_rgbd: camera model
# ---------------------------------------------------------------------------


def test_balanced_grid_matches_the_persisted_reconstruction_grid():
    """640×480 through VGGT-Omega ``balanced@512`` is 592×448 — the same grid the persisted
    trajectory intrinsics imply (their principal points sit at 296, 224)."""
    assert tum.balanced_grid(640, 480) == (592, 448)


def test_rescale_intrinsics_is_a_pure_anisotropic_resize():
    cam = tum.rescale_intrinsics(100.0, 200.0, 50.0, 60.0, (100, 200), (300, 400))
    assert cam["fx"] == pytest.approx(300.0) and cam["cx"] == pytest.approx(150.0)
    assert cam["fy"] == pytest.approx(400.0) and cam["cy"] == pytest.approx(120.0)
    assert cam["width"] == 300 and cam["height"] == 400
    assert cam["rotation"] is None  # TUM depth is pre-registered to the colour frame


def test_camera_from_intrinsics_row_lands_near_the_published_calibration():
    """VGGT's own prediction for a real fr1 keyframe, rescaled to the depth grid, must be in
    the neighbourhood of TUM's published camera — a guard against getting the grid or the
    resize direction backwards, which would be off by ~8%, not ~1%."""
    cam = tum.camera_from_intrinsics_row(
        [472.96990966796875, 471.43646240234375, 295.1781311035156, 222.2547607421875],
        tum.balanced_grid(640, 480))
    assert cam["fx"] == pytest.approx(tum.TUM_ROS_DEFAULT["fx"], rel=0.05)
    assert cam["cx"] == pytest.approx(tum.TUM_ROS_DEFAULT["cx"], rel=0.02)


def test_published_camera_sets_and_refusal():
    assert tum.published_camera("ros_default")["fx"] == 525.0
    assert tum.published_camera("fr1")["cy"] == 255.3
    with pytest.raises(ValueError, match="unknown intrinsics set"):
        tum.published_camera("freiburg9")


# ---------------------------------------------------------------------------
# the extraction itself: these modules must stay runnable in a Flask-less container
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [
    "server.scene_report.anchor",   # the anchor route's implementation
    "server.oreos.depth_ratio",      # the ratio core
    "server.oreos.tum_rgbd",         # the dataset adapter
    "server.oreos.measure",          # what decides whether an "m" glyph may render
])
def test_module_imports_without_flask(module):
    """A regression guard on WHY these modules were split out.

    ``modal_tum_depth_anchor.py`` runs them in a numpy-only image with no flask, no
    opencv and no torch, reaching the scene store directly because a Clerk token lives 60
    seconds. A stray ``from flask import ...`` added to any of them would not fail a test
    here (the suite has flask) — it would fail in the container, days later. So this asserts
    it in a fresh interpreter, where the failure is immediate and legible."""
    import subprocess
    import textwrap

    code = textwrap.dedent(f"""
        import importlib, sys, types
        sys.path.insert(0, {str(_REPO)!r})
        for pkg, path in (("server", "server"), ("server.oreos", "server/oreos"),
                          ("server.scene_report", "server/scene_report")):
            m = types.ModuleType(pkg)
            m.__path__ = [{str(_REPO)!r} + "/" + path]
            sys.modules[pkg] = m
        importlib.import_module({module!r})
        for banned in ("flask", "cv2", "torch", "openai"):
            assert banned not in sys.modules, banned + " was imported"
        print("clean")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_gemini_cli_still_imports_without_numpy():
    """`demo_ingest_gemini.py`'s stated posture: ``assemble``/``upload`` are stdlib-pure so
    they run in the exp39 capture venv, and only ``anchor`` needs numpy.

    Moving the ratio core out put that at risk — the shared module imports numpy at module
    level, so every name re-exported from it is conditional, and one of them used to be a
    def-time default in a function signature (an import-time NameError in exactly the venv
    the CLI was written for). Asserted in a subprocess with numpy blocked, because the test
    env always has numpy and would never notice."""
    import subprocess
    import textwrap

    script = _REPO / "scripts" / "demo_ingest_gemini.py"
    code = textwrap.dedent(f"""
        import sys, importlib.util
        class Blocker:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in ("numpy", "cv2") else None
            def load_module(self, name):
                raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Blocker())
        spec = importlib.util.spec_from_file_location("gem_nonumpy", {str(script)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.np is None
        assert mod.build_parser() is not None      # the CLI is still constructible
        print("clean")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_depth_scale_constant_matches_the_dataset():
    assert tum.DEPTH_UNITS_PER_METRE == 5000.0
    assert tum.MM_PER_UNIT == pytest.approx(0.2)
    # One raw unit is 0.2 mm; block_median_depth converts to metres.
    png = np.full((4, 4), 5000, dtype=np.uint16)
    assert dr.block_median_depth(png, 4, tum.MM_PER_UNIT)[0, 0] == pytest.approx(1.0)
