"""LOD tests: the decimation core (`server/oreos/lod.py`) + the routes.

GPU-free and Modal-free. Same harness as tests/test_demo_ingest_routes.py: real
Flask + ``test_client`` on a freshly imported ``server.oreos`` blueprint, with
``server.app`` stubbed via ``sys.modules``, the jobs store swapped for a plain
dict, and the LOD job spawner swapped for a recording fake.

The decimation tests build small synthetic splats on disk and assert the
properties the demo actually depends on:
  * normals are dropped only when they are genuinely zero,
  * voxel selection beats uniform on SPATIAL COVERAGE (the measured failure mode
    on the real scenes is holes, not blur),
  * the budget is respected,
  * the scale floor only ever RAISES a scale, never lowers one.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.oreos import lod as lod_mod
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_splat(path, n, *, seed=0, zero_normals=True, vary_opacity=False, extent=10.0):
    """Write a synthetic 17-property gaussian splat PLY (the exporter schema)."""
    rng = np.random.default_rng(seed)
    # Points on a rough "surface" (a slab) so voxel selection has real structure.
    xyz = rng.uniform(0, extent, size=(n, 3)).astype(np.float32)
    xyz[:, 1] *= 0.05  # flatten in Y -> a slab, like a room's walls/floor

    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    cols = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "nx": np.zeros(n, np.float32) if zero_normals else rng.normal(size=n).astype(np.float32),
        "ny": np.zeros(n, np.float32),
        "nz": np.zeros(n, np.float32),
        "f_dc_0": rng.uniform(-1, 1, n).astype(np.float32),
        "f_dc_1": rng.uniform(-1, 1, n).astype(np.float32),
        "f_dc_2": rng.uniform(-1, 1, n).astype(np.float32),
        "opacity": (rng.uniform(-2, 4, n).astype(np.float32) if vary_opacity
                    else np.full(n, 2.1972246, np.float32)),  # logit(0.9)
        "scale_0": np.full(n, -6.0, np.float32),
        "scale_1": np.full(n, -6.0, np.float32),
        "scale_2": np.full(n, -6.0, np.float32),
        "rot_0": np.ones(n, np.float32),
        "rot_1": np.zeros(n, np.float32),
        "rot_2": np.zeros(n, np.float32),
        "rot_3": np.zeros(n, np.float32),
    }
    dtype = np.dtype([(nm, "<f4") for nm in names])
    verts = np.empty(n, dtype=dtype)
    for nm in names:
        verts[nm] = cols[nm]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n".format(n)
        + "".join(f"property float {nm}\n" for nm in names)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        verts.tofile(f)
    return str(path)


def _cell_coverage(src, idx, voxel):
    """Fraction of the source's occupied cells still hit by the selection."""
    origin = np.array([0.0, 0.0, 0.0]) - 1e-6
    full = np.unique(
        lod_mod._voxel_keys(src.column("x"), src.column("y"), src.column("z"), origin, voxel)
    )
    kept = np.unique(
        lod_mod._voxel_keys(src.take("x", idx), src.take("y", idx), src.take("z", idx), origin, voxel)
    )
    return np.intersect1d(kept, full, assume_unique=True).size / max(full.size, 1)


# ---------------------------------------------------------------------------
# lod.py — reading
# ---------------------------------------------------------------------------


def test_reads_header_and_rejects_non_gaussian(tmp_path):
    p = _write_splat(tmp_path / "a.ply", 500)
    src = lod_mod.SplatSource(p)
    assert src.count == 500
    assert src.properties[:3] == ["x", "y", "z"]
    assert src.itemsize == 68  # 17 float32 properties
    src.close()

    # a plain xyz point cloud is not a gaussian splat
    plain = tmp_path / "plain.ply"
    with open(plain, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                b"property float x\nproperty float y\nproperty float z\nend_header\n")
        np.zeros(6, "<f4").tofile(f)
    with pytest.raises(ValueError, match="not a gaussian splat"):
        lod_mod.SplatSource(str(plain))


def test_truncated_file_is_rejected(tmp_path):
    """A file whose HEADER promises more vertices than the body holds must be
    refused up front — memmapping past the end would otherwise read garbage."""
    p = _write_splat(tmp_path / "t.ply", 1000)
    _, count, offset = lod_mod.read_ply_header(p)
    assert count == 1000
    with open(p, "r+b") as f:
        f.truncate(offset + 68 * 10)  # header intact, only 10 of 1000 vertices present
    with pytest.raises(ValueError, match="truncated"):
        lod_mod.SplatSource(p)


def test_truncated_header_is_rejected(tmp_path):
    p = _write_splat(tmp_path / "th.ply", 1000)
    # Cut on a clean line boundary right after "element vertex 1000\n" so the
    # header simply ends rather than leaving a half-written property line.
    with open(p, "r+b") as f:
        f.truncate(len(b"ply\nformat binary_little_endian 1.0\nelement vertex 1000\n"))
    with pytest.raises(ValueError, match="EOF in PLY header"):
        lod_mod.SplatSource(p)


def test_normals_zero_detection(tmp_path):
    zero = lod_mod.SplatSource(_write_splat(tmp_path / "z.ply", 300, zero_normals=True))
    nonzero = lod_mod.SplatSource(_write_splat(tmp_path / "n.ply", 300, zero_normals=False, seed=3))
    assert lod_mod.normals_are_zero(zero) is True
    assert lod_mod.normals_are_zero(nonzero) is False
    zero.close()
    nonzero.close()


# ---------------------------------------------------------------------------
# lod.py — selection
# ---------------------------------------------------------------------------


def test_plan_levels_skips_pointless_levels():
    # The canonical scene is 2.36x the 8M level, so every budget earns its place.
    assert lod_mod.plan_levels(18_846_947) == [600_000, 2_000_000, 4_000_000, 8_000_000]
    assert lod_mod.plan_levels(500_000) == []          # already smaller than every budget
    assert lod_mod.plan_levels(700_000) == []          # within 1.3x of 600k -> not worth it
    assert lod_mod.plan_levels(2_600_000) == [600_000]  # exactly 1.3x of 2M -> 2M excluded
    assert lod_mod.plan_levels(3_000_000) == [600_000, 2_000_000]
    # 8M is a RECORDING level, not an interactive one: it is only built when the
    # source is big enough for it to be a real reduction (>1.3x, i.e. >10.4M).
    assert lod_mod.plan_levels(10_000_000) == [600_000, 2_000_000, 4_000_000]
    assert lod_mod.plan_levels(63_308_835) == [600_000, 2_000_000, 4_000_000, 8_000_000]


def test_selection_respects_budget_and_is_deterministic(tmp_path):
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 60_000, seed=1))
    idx_a, info = lod_mod.select_indices(src, 5_000, method="voxel_first")
    idx_b, _ = lod_mod.select_indices(src, 5_000, method="voxel_first")
    assert idx_a.size <= 5_000
    assert idx_a.size >= 4_000  # lands near the budget, never wildly under
    np.testing.assert_array_equal(idx_a, idx_b)  # deterministic
    assert np.all(np.diff(idx_a) > 0)            # sorted, unique
    assert info["voxel_size"] > 0
    src.close()


def test_target_above_source_returns_everything(tmp_path):
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 1_000))
    idx, info = lod_mod.select_indices(src, 5_000)
    assert idx.size == 1_000
    assert info["method"] == "identity"
    src.close()


def test_voxel_selection_beats_uniform_on_coverage(tmp_path):
    """The measured failure mode on the real scenes is holes, so the selection
    that maximises spatial spread is the right one — assert that property."""
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 120_000, seed=7))
    budget = 8_000
    voxel_idx, _ = lod_mod.select_indices(src, budget, method="voxel_first")
    uniform_idx, _ = lod_mod.select_indices(src, budget, method="uniform")
    importance_idx, _ = lod_mod.select_indices(src, budget, method="importance")

    eval_voxel = 0.25
    cov_voxel = _cell_coverage(src, voxel_idx, eval_voxel)
    cov_uniform = _cell_coverage(src, uniform_idx, eval_voxel)
    cov_importance = _cell_coverage(src, importance_idx, eval_voxel)

    assert cov_voxel > cov_uniform, (cov_voxel, cov_uniform)
    # Global importance top-K degenerates on our degenerate-gaussian exports.
    assert cov_voxel > cov_importance, (cov_voxel, cov_importance)
    src.close()


def test_auto_picks_cheap_path_when_opacity_is_constant(tmp_path):
    """Our exporter writes a CONSTANT opacity, so ranking within a cell is a
    provable no-op there; `auto` must notice and take the 5x-faster path."""
    flat = lod_mod.SplatSource(_write_splat(tmp_path / "flat.ply", 20_000, vary_opacity=False))
    varied = lod_mod.SplatSource(_write_splat(tmp_path / "var.ply", 20_000, vary_opacity=True, seed=5))
    _, info_flat = lod_mod.select_indices(flat, 2_000, method="auto")
    _, info_var = lod_mod.select_indices(varied, 2_000, method="auto")
    assert info_flat["method"] == "voxel_first"
    assert info_var["method"] == "voxel_importance"
    flat.close()
    varied.close()


def test_unknown_method_rejected(tmp_path):
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 1_000))
    with pytest.raises(ValueError, match="unknown LOD selection method"):
        lod_mod.select_indices(src, 100, method="nope")
    src.close()


# ---------------------------------------------------------------------------
# lod.py — writing
# ---------------------------------------------------------------------------


def test_write_drops_normals_and_floors_scales(tmp_path):
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 30_000, seed=2))
    idx, info = lod_mod.select_indices(src, 3_000, method="voxel_first")
    out = str(tmp_path / "lod.ply")
    meta = lod_mod.write_lod_ply(src, idx, out, voxel_size=info["voxel_size"])

    assert meta["properties"] == list(lod_mod.LOD_PROPERTIES)
    assert "nx" not in meta["properties"]
    assert meta["bytes_per_gaussian"] == 56  # 14 float32, down from 68
    assert meta["normals_dropped"] is True

    written = lod_mod.SplatSource(out)
    assert written.count == idx.size
    # The floor may only RAISE a scale — never shrink a gaussian.
    orig = src.take("scale_0", idx)
    new = written.column("scale_0")
    assert np.all(new >= orig - 1e-6)
    if meta["scale_floor_applied"]:
        assert meta["scale_axes_raised"] > 0
        assert meta["scale_floor_linear"] > 0
    # Positions must be untouched — decimation selects, it never moves geometry.
    np.testing.assert_allclose(written.column("x"), src.take("x", idx), rtol=0, atol=0)
    written.close()
    src.close()


def test_scale_floor_can_be_disabled(tmp_path):
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 20_000, seed=4))
    idx, info = lod_mod.select_indices(src, 2_000, method="voxel_first")
    out = str(tmp_path / "nofloor.ply")
    meta = lod_mod.write_lod_ply(src, idx, out, voxel_size=info["voxel_size"], scale_floor_frac=0.0)
    assert meta["scale_floor_applied"] is False
    assert meta["scale_axes_raised"] == 0
    written = lod_mod.SplatSource(out)
    np.testing.assert_allclose(written.column("scale_0"), src.take("scale_0", idx))
    written.close()
    src.close()


# ---------------------------------------------------------------------------
# lod.py — index
# ---------------------------------------------------------------------------


def test_index_shape_and_default_level():
    entries = [
        {"name": "600k", "budget": 600_000, "key": lod_mod.level_key(600_000), "gaussians": 600_000},
        {"name": "2000k", "budget": 2_000_000, "key": lod_mod.level_key(2_000_000), "gaussians": 1_983_597},
    ]
    doc = lod_mod.build_index(
        scan_id="scan-1", source_count=18_846_947, source_bytes=1_281_592_814, entries=entries
    )
    assert doc["version"] == lod_mod.LOD_INDEX_VERSION
    assert doc["default_level"] == 2_000_000
    assert doc["source"]["key"] == "splat.ply"
    assert doc["generated"] is True
    assert doc["caveats"] and all(isinstance(c, str) for c in doc["caveats"])
    assert json.loads(lod_mod.index_json_bytes(doc).decode()) == doc


def test_index_records_the_source_key_for_staleness():
    """An anchored scene's LOD is built from the DERIVED splat; the client compares
    this key to detect an LOD that no longer matches the scene's gauge."""
    doc = lod_mod.build_index(
        scan_id="s", source_count=10_000_000, source_bytes=1, entries=[],
        source_key="derived/anchor/2026/splat.ply",
    )
    assert doc["source"]["key"] == "derived/anchor/2026/splat.ply"


def test_level_keys():
    assert lod_mod.level_key(2_000_000) == "demo/lod/splat_2000k.ply"
    assert lod_mod.level_key(2_000_000, ".spz") == "demo/lod/splat_2000k.spz"
    assert lod_mod.full_key() == "demo/lod/full.spz"
    assert lod_mod.LOD_INDEX_KEY == "demo/lod/index.json"


def test_spz_encode_failure_is_reported_not_raised(tmp_path, monkeypatch):
    """An spz failure must degrade to shipping the PLY, never lose the level."""
    src = lod_mod.SplatSource(_write_splat(tmp_path / "s.ply", 20_000, seed=6))

    def boom(*a, **k):
        raise lod_mod.SpzEncodeError("node missing")

    monkeypatch.setattr(lod_mod, "encode_spz", boom)
    entry = lod_mod.build_level(
        src, 2_000, str(tmp_path / "l.ply"), method="voxel_first", spz_path=str(tmp_path / "l.spz")
    )
    assert "spz_error" in entry
    assert "spz_key" not in entry
    assert entry["gaussians"] > 0 and entry["bytes"] > 0  # the PLY still shipped
    src.close()


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def lod_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    jobs = sys.modules["server.oreos.jobs"]
    jobs.configure_jobs_store({})
    routes = sys.modules["server.oreos.routes_lod"]
    spawned: list[dict] = []
    routes.configure_lod_spawner(lambda **kw: spawned.append(kw))
    # a minimal persisted record so the scene resolves
    store._store["user-a:scan-1"] = {"scan_id": "scan-1", "user_id": "user-a", "point_count": 18_846_947}

    yield types.SimpleNamespace(
        client=app.test_client(), store=store, routes=routes, spawned=spawned, jobs=jobs
    )
    routes.configure_lod_spawner(None)
    jobs.configure_jobs_store(None)


def test_get_lod_unknown_scene_is_404(lod_app):
    r = lod_app.client.get("/api/scenes/nope/lod")
    assert r.status_code == 404
    assert r.get_json()["error"] == "not_found"


def test_get_lod_reports_none_before_generation(lod_app):
    r = lod_app.client.get("/api/scenes/scan-1/lod")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "none"
    assert body["index"] is None
    assert body["source_gaussians"] == 18_846_947
    assert body["default_level_target"] == lod_mod.DEFAULT_LEVEL


def test_post_lod_spawns_then_409s_while_running(lod_app):
    r = lod_app.client.post("/api/scenes/scan-1/lod", json={})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    assert len(lod_app.spawned) == 1
    assert lod_app.spawned[0]["scan_id"] == "scan-1"
    assert lod_app.spawned[0]["user_id"] == "user-a"

    # a second POST while the job is queued must not fan out another container
    r2 = lod_app.client.post("/api/scenes/scan-1/lod", json={})
    assert r2.status_code == 409
    body = r2.get_json()
    assert body["error"] == "lod_job_active"
    assert body["job_id"] == job_id
    assert len(lod_app.spawned) == 1


def test_post_lod_validates_body(lod_app):
    assert lod_app.client.post("/api/scenes/scan-1/lod", json={"levels": ["x"]}).status_code == 400
    assert lod_app.client.post("/api/scenes/scan-1/lod", json={"method": "nope"}).status_code == 400
    assert lod_app.spawned == []


def test_get_lod_ready_once_index_exists(lod_app):
    doc = lod_mod.build_index(
        scan_id="scan-1", source_count=18_846_947, source_bytes=1,
        entries=[{"name": "2000k", "budget": 2_000_000, "key": lod_mod.level_key(2_000_000),
                  "gaussians": 1_983_597, "spz_key": lod_mod.level_key(2_000_000, ".spz"),
                  "spz_bytes": 11_345_910}],
    )
    lod_app.store.save_derived_artifact(
        "user-a", "scan-1", lod_mod.LOD_INDEX_KEY, lod_mod.index_json_bytes(doc)
    )
    r = lod_app.client.get("/api/scenes/scan-1/lod")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ready"
    assert body["index"]["default_level"] == 2_000_000
    assert body["index"]["levels"][0]["spz_key"] == "demo/lod/splat_2000k.spz"

    # POST is a no-op once artifacts exist, unless forced
    r2 = lod_app.client.post("/api/scenes/scan-1/lod", json={})
    assert r2.status_code == 200 and r2.get_json()["status"] == "ready"
    assert lod_app.spawned == []
    r3 = lod_app.client.post("/api/scenes/scan-1/lod", json={"force": True})
    assert r3.status_code == 202
    assert len(lod_app.spawned) == 1 and lod_app.spawned[0]["force"] is True


def test_unauthenticated_get_is_401(lod_app, monkeypatch):
    stub = sys.modules["server.app"]
    monkeypatch.setattr(stub, "_auth_user_id", lambda: None)
    assert lod_app.client.get("/api/scenes/scan-1/lod").status_code == 401
    assert lod_app.client.post("/api/scenes/scan-1/lod", json={}).status_code == 401


def test_lod_routes_registered_once(lod_app):
    rules = [r for r in lod_app.client.application.url_map.iter_rules()
             if str(r.rule) == "/api/scenes/<scan_id>/lod"]
    methods = set()
    for r in rules:
        methods |= {m for m in r.methods if m in ("GET", "POST")}
    assert methods == {"GET", "POST"}


def test_lod_get_endpoint_name_is_share_allowlisted(lod_app):
    """The share embed reads GET /lod with a share token. app.py's
    ``SHARE_TOKEN_ALLOWED_ENDPOINTS`` is keyed by the endpoint name Flask reports, which
    for a Blueprint route is ``<bp.name>.<fn>`` — NOT the bare function name. The
    2026-08-24 deploy shipped the bare name and every share embed 403'd on /lod while
    the fake-request unit test stayed green. This pins the real name against a real
    Flask app; the matching literal lives in server/app.py (no import — importing
    server.app here would drag in torch)."""
    get_rules = [r for r in lod_app.client.application.url_map.iter_rules()
                 if str(r.rule) == "/api/scenes/<scan_id>/lod" and "GET" in r.methods]
    assert len(get_rules) == 1
    assert get_rules[0].endpoint == "demo.get_scene_lod_route"
    import ast, pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "server" / "app.py"
    tree = ast.parse(src.read_text())
    literals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SHARE_TOKEN_ALLOWED_ENDPOINTS" for t in node.targets
        ):
            literals = {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant)}
    assert "demo.get_scene_lod_route" in literals
    assert "demo.post_scene_lod_route" not in literals  # the build stays owner-only


# ---------------------------------------------------------------------------
# spz encoder script — environment guards
# ---------------------------------------------------------------------------


def test_encoder_script_shims_navigator_for_node20():
    """REGRESSION GUARD. Spark is a browser library whose module has an import-time
    side effect reading ``navigator.xr`` (VRButton). Node 21+ ships a global
    ``navigator`` so this is invisible on a dev machine — but the Modal image runs
    **Node 20**, where importing Spark dies with ``ReferenceError: navigator is not
    defined`` and every LOD level silently ships PLY-only, losing the ~9-15x
    transport win. This actually happened on the first deploy. Do not remove.
    """
    import os

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "ply_to_spz.mjs"
    )
    src = open(script, encoding="utf-8").read()
    assert "globalThis.navigator" in src
    # the shim must run BEFORE spark is imported, or it is useless
    assert src.index("globalThis.navigator") < src.index("await import(sparkUrl)")


@pytest.mark.skipif(
    __import__("shutil").which("node") is None, reason="node not available"
)
def test_encoder_round_trips_through_spark(tmp_path):
    """End-to-end: our PLY -> Spark's transcodeSpz -> Spark's SpzReader verify.

    Skipped when node is absent. Note this passes on any Node with a global
    ``navigator``; the Node-20 case is covered by the guard test above and by the
    deployed run.
    """
    import json as _json
    import os
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "ply_to_spz.mjs")
    spark = os.environ.get("SPARK_MODULE_PATH") or os.path.join(
        root, "web", "apps", "webserver", "node_modules",
        "@sparkjsdev", "spark", "dist", "spark.module.js",
    )
    if not os.path.isfile(spark):
        pytest.skip("spark not installed in this checkout")

    ply = _write_splat(tmp_path / "in.ply", 5_000, seed=11)
    src = lod_mod.SplatSource(ply)
    idx, info = lod_mod.select_indices(src, 2_000, method="voxel_first")
    lod_ply = str(tmp_path / "lod.ply")
    lod_mod.write_lod_ply(src, idx, lod_ply, voxel_size=info["voxel_size"])
    src.close()

    out = str(tmp_path / "lod.spz")
    env = dict(os.environ, SPARK_MODULE_PATH=spark)
    proc = subprocess.run(
        ["node", script, lod_ply, out, "--verify"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert rep["verify"]["version"] == 3
    assert rep["verify"]["num_splats"] == idx.size
    assert rep["output_bytes"] < rep["input_bytes"]  # compression actually happened
