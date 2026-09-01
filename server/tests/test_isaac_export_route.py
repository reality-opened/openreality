"""Isaac USD export WIRING tests (``GET /api/scenes/<id>/export?format=isaac_usd``).

Covers the NEW broker route wiring on top of the existing offline ``server/export/isaac/`` writer:

  * the metric-scale GATE (409 ``metric_scale_required`` when neither ``?scale`` nor a Metric
    anchor is available) — decidable WITHOUT the heavy USD deps;
  * scale resolution that can never double-scale (explicit factor -> original geometry; a Metric
    anchor -> its already-metric geometry at scale 1.0);
  * format routing (``isaac_usd`` dispatches to the isaac handler, not ``invalid_format``);
  * the lazy-dependency 501 (``isaac_unavailable``) when ``usd-core``/``open3d`` are absent — both
    the pre-check path (simulated missing) and the defensive ImportError-during-write path.

Like ``tests/test_product_workflow_routes.py`` these run WITHOUT ``openreality-core`` /
``vggt_slam`` (hand-built store + record). The only test that needs the real USD toolchain
(``pxr``/``open3d``) ``importorskip``s, mirroring the existing ``tests/test_isaac_*`` — real USD
generation + a human Isaac Sim load stay open verification rows (``docs/isaac-export.md`` §6).
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from conftest import load_app_module

# Reuse the self-contained fixtures/harness (hand-built store + fake request, no vggt_slam).
from test_product_workflow_routes import _fake_request, _store_with_scene


# ---------------------------------------------------------------------------
# dep-status + scale-arg parsing helpers (dep-free unit tests)
# ---------------------------------------------------------------------------


def test_dependency_status_reports_missing_usd_and_open3d(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    import importlib.util as _ilu

    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *a, **k):
        if name in ("pxr", "open3d"):
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(_ilu, "find_spec", fake_find_spec)
    assert app_mod._isaac_dependency_status() == ["usd-core", "open3d"]


def test_is_anchor_derived_key(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._is_anchor_derived_key("derived/anchor/1_abc/cloud.ply") is True
    assert app_mod._is_anchor_derived_key("derived/clamp/1_abc/splat.ply") is False
    assert app_mod._is_anchor_derived_key("anchor/1_abc/cloud.ply") is False
    assert app_mod._is_anchor_derived_key(None) is False
    assert app_mod._is_anchor_derived_key("") is False


def test_parse_isaac_scale_arg(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    assert app_mod._parse_isaac_scale_arg(None) is None
    assert app_mod._parse_isaac_scale_arg("") is None
    assert app_mod._parse_isaac_scale_arg("  ") is None
    assert app_mod._parse_isaac_scale_arg("0.25") == pytest.approx(0.25)
    for bad in ("0", "-1", "abc", "nan", "inf"):
        with pytest.raises((ValueError, TypeError)):
            app_mod._parse_isaac_scale_arg(bad)


# ---------------------------------------------------------------------------
# _resolve_isaac_scale — the anti-double-scale core (dep-free)
# ---------------------------------------------------------------------------


def test_resolve_isaac_scale_explicit_factor_uses_original_geometry(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    out = app_mod._resolve_isaac_scale({}, None, 0.25)
    assert out == (app_mod.EXPORT_SOURCE_ORIGINAL, 0.25, "user:factor", None)


def test_resolve_isaac_scale_explicit_anchor_source_is_prescaled(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    key = "derived/anchor/1_a/cloud.ply"
    out = app_mod._resolve_isaac_scale({}, key, None)
    # already-metric geometry -> scale 1.0, never re-applied
    assert out == (key, 1.0, "anchor:prescaled", None)


def test_resolve_isaac_scale_from_derived_latest_pointer(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    record = {
        "derived_latest": {
            "kind": "anchor",
            "source_key": "derived/anchor/9_z/cloud.ply",
            "scale_factor": 0.5,
        }
    }
    out = app_mod._resolve_isaac_scale(record, None, None)
    assert out == ("derived/anchor/9_z/cloud.ply", 1.0, "anchor:prescaled", 0.5)


def test_resolve_isaac_scale_gate_fails_without_scale_or_anchor(monkeypatch):
    app_mod = load_app_module(monkeypatch)
    # no scale, no pointer
    assert app_mod._resolve_isaac_scale({}, None, None) is None
    # a CLAMP pointer is not a metric anchor -> still a gate fail
    clamp_rec = {"derived_latest": {"kind": "clamp", "source_key": "derived/clamp/1/splat.ply"}}
    assert app_mod._resolve_isaac_scale(clamp_rec, None, None) is None
    # ?source=original with no scale/anchor -> gate fail
    assert app_mod._resolve_isaac_scale({}, app_mod.EXPORT_SOURCE_ORIGINAL, None) is None


# ---------------------------------------------------------------------------
# route: metric-scale gate (409) — no heavy deps needed
# ---------------------------------------------------------------------------


def test_route_isaac_without_scale_or_anchor_returns_409(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(monkeypatch, app_mod, args={"format": "isaac_usd"}, claims={"sub": "user-a"})

    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 409
    assert out[0]["args"][0]["error"] == "metric_scale_required"
    assert "Metric anchor" in out[0]["args"][0]["message"]


def test_route_isaac_bad_scale_value_returns_400(monkeypatch, tmp_path):
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "-3"}, claims={"sub": "user-a"}
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 400
    assert out[0]["args"][0]["error"] == "invalid_request"


# ---------------------------------------------------------------------------
# route: format routing + lazy-dependency 501 (simulated missing deps)
# ---------------------------------------------------------------------------


def test_route_isaac_with_scale_but_missing_deps_returns_501(monkeypatch, tmp_path):
    """Explicit ?scale satisfies the metric gate; the deps precheck then 501s honestly. This
    also proves format routing: isaac_usd reaches the isaac handler, not `invalid_format`."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    # simulate a bare runtime image with neither usd-core nor open3d installed
    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: ["usd-core", "open3d"])
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "0.25"}, claims={"sub": "user-a"}
    )

    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 501
    body = out[0]["args"][0]
    assert body["error"] == "isaac_unavailable"
    assert "usd-core" in body["message"] and "open3d" in body["message"]


def test_route_isaac_refuses_a_scene_too_large_to_mesh(monkeypatch, tmp_path):
    """A cloud the broker cannot hold is refused with the number, not attempted.

    REGRESSION (founder, 2026-08-03): exporting the 63,308,835-point scene answered a bare
    HTTP 500 with no message and no `[isaac]` log line. The geometry load ran OUTSIDE the
    handler's try/except and exhausted the broker's 4 GB about 5 s in; Poisson meshing that
    cloud ran past 900 s even on 8 GB. Same doctrine as the viewer's DIRECT_LOAD_MAX_BYTES:
    a request that cannot possibly succeed gets an honest refusal, never an attempt.

    The gate is dep-free ON PURPOSE — it sits before the usd-core/open3d precheck, so it is
    decidable (and testable) on a runtime with no USD toolchain at all, exactly like the
    metric-scale gate above.
    """
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    record = store.get_scene("user-a", "scan1")
    record["point_count"] = app_mod.ISAAC_MAX_POINTS + 1
    # Deps deliberately absent: a 501 here would prove the gate ran too late.
    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: ["usd-core", "open3d"])
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "0.25"}, claims={"sub": "user-a"}
    )

    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 413
    body = out[0]["args"][0]
    assert body["error"] == "scene_too_large"
    assert body["point_count"] == app_mod.ISAAC_MAX_POINTS + 1
    assert body["limit"] == app_mod.ISAAC_MAX_POINTS
    # The operator needs the actual numbers to decide what to do next.
    assert f"{app_mod.ISAAC_MAX_POINTS + 1:,}" in body["message"]


def test_route_isaac_allows_a_scene_at_the_limit(monkeypatch, tmp_path):
    """The ceiling is inclusive — both real demo scenes (18.8M, 17.2M) must stay exportable,
    so an off-by-one here would silently cut a working feature."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    record = store.get_scene("user-a", "scan1")
    record["point_count"] = app_mod.ISAAC_MAX_POINTS
    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: ["usd-core", "open3d"])
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "0.25"}, claims={"sub": "user-a"}
    )

    out = app_mod.export_scene_route("scan1")
    # Passes the size gate and falls through to the honest dependency 501.
    assert isinstance(out, tuple) and out[1] == 501
    assert out[0]["args"][0]["error"] == "isaac_unavailable"


def test_route_isaac_missing_only_pxr_returns_501(monkeypatch, tmp_path):
    """Simulate a runtime with open3d but no pxr (usd-core) — still an honest 501."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: ["usd-core"])
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "0.25"}, claims={"sub": "user-a"}
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 501
    assert out[0]["args"][0]["error"] == "isaac_unavailable"


def test_route_isaac_anchor_pointer_satisfies_gate_then_501(monkeypatch, tmp_path):
    """A persisted Metric-anchor pointer (kind=anchor) satisfies the gate WITHOUT ?scale; deps
    missing -> 501. Proves the derived_latest path is honored."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    store.set_derived_pointer(
        "user-a",
        "scan1",
        {"kind": "anchor", "source_key": "derived/anchor/x/cloud.ply", "scale_factor": 0.5},
    )
    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: ["usd-core", "open3d"])
    _fake_request(monkeypatch, app_mod, args={"format": "isaac_usd"}, claims={"sub": "user-a"})
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 501
    assert out[0]["args"][0]["error"] == "isaac_unavailable"


def test_route_invalid_format_still_400(monkeypatch, tmp_path):
    """A genuinely unknown format is still 400 invalid_format (isaac_usd is now recognized)."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=True, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    _fake_request(
        monkeypatch, app_mod, args={"format": "totally_made_up"}, claims={"sub": "user-a"}
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 400
    assert out[0]["args"][0]["error"] == "invalid_format"


# ---------------------------------------------------------------------------
# route: defensive ImportError-during-write -> 501 (deps "present" but pxr really absent).
# Exercises the WHOLE record->align->writer path (real anchor geometry, points-only fallback)
# up to the pxr authoring boundary — runs WITHOUT any heavy dep installed.
# ---------------------------------------------------------------------------


def test_route_isaac_build_importerror_maps_to_501(monkeypatch, tmp_path):
    """A lazy USD dep that vanishes between the precheck and the write raises ImportError from the
    writer; the route maps that to the same honest 501 (never a 500/crash). Deterministic: deps are
    faked present and the builder is stubbed to raise, so it runs with or without pxr installed."""
    app_mod = load_app_module(monkeypatch)
    store = _store_with_scene(tmp_path, with_splat=False, with_trajectory=True)
    app_mod.configure_scene_persistence(store)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [2, 0, 0], 1.0)
    anchor_key = result["calibrated_cloud_key"]

    monkeypatch.setattr(app_mod, "_isaac_dependency_status", lambda: [])  # precheck says "present"

    def _raise_import(*a, **k):
        raise ImportError("usd-core (pxr) is required to author Isaac USD scenes")

    monkeypatch.setattr(app_mod, "_build_isaac_zip", _raise_import)
    _fake_request(
        monkeypatch,
        app_mod,
        args={"format": "isaac_usd", "source": anchor_key},
        claims={"sub": "user-a"},
    )
    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, tuple) and out[1] == 501
    assert out[0]["args"][0]["error"] == "isaac_unavailable"


# ---------------------------------------------------------------------------
# REAL USD generation from a persisted record (needs the actual USD toolchain).
# Skipped where pxr isn't installed — mirrors tests/test_isaac_writer.py. open3d is not
# required here (reconstruct=False -> a points-only scene), so this can run on a pxr-only host.
# ---------------------------------------------------------------------------


def test_export_isaac_from_record_points_only(tmp_path):
    pytest.importorskip("pxr", reason="usd-core (pxr) not installed")
    from pxr import Usd, UsdGeom

    from server.export.isaac.writer import export_isaac_from_record

    n = 400
    rng = np.random.default_rng(0)
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(n, 3)).astype(np.float32)
    floor[:, 2] = 0.0
    above = rng.uniform([-1, -1, 0.3], [1, 1, 1.5], size=(120, 3)).astype(np.float32)
    positions = np.vstack([floor, above]) * np.float32(4.0)  # up-to-scale
    colors = rng.integers(0, 255, size=(len(positions), 3), dtype=np.uint8)
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(3)])
    for i in range(3):
        poses[i, :3, 3] = np.array([0.0, 0.0, 1.5], np.float32) * 4.0
    record = {
        "scan_id": "rec-fixture",
        "cloud": (positions, colors),
        "trajectory": {
            "poses": poses,
            "intrinsics": np.tile(np.array([500, 500, 320, 240], np.float32), (3, 1)),
            "source_frame_id": np.arange(3, dtype=np.float32),
        },
    }

    isaac_dir = export_isaac_from_record(
        record, str(tmp_path), scan_id="rec-fixture", scale=1.0 / 4.0, reconstruct=False
    )
    assert isaac_dir is not None
    base = tmp_path / "rec-fixture" / "isaac"
    assert (base / "scene.usd").exists() and (base / "trajectory.usd").exists()

    manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["alignment"]["metric"] is True  # gate guarantees a metric scene
    assert manifest["alignment"]["scale_source"] == "user:factor"
    assert manifest["num_keyframes"] == 3
    assert manifest["geometry"]["representation"] == "points"

    scene = Usd.Stage.Open(str(base / "scene.usd"))
    assert UsdGeom.GetStageUpAxis(scene) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(scene) == 1.0


def test_export_isaac_from_record_anchor_prescaled_provenance(tmp_path):
    """When the record already carries anchor-calibrated (metric) geometry, the writer applies
    scale 1.0 and stamps the manifest with anchor provenance (never re-scales)."""
    pytest.importorskip("pxr", reason="usd-core (pxr) not installed")

    from server.export.isaac.writer import export_isaac_from_record

    rng = np.random.default_rng(1)
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(400, 3)).astype(np.float32)
    floor[:, 2] = 0.0
    positions = np.vstack([floor, floor + [0, 0, 1.0]]).astype(np.float32)  # already metric
    colors = rng.integers(0, 255, size=(len(positions), 3), dtype=np.uint8)
    record = {"scan_id": "s", "cloud": (positions, colors), "trajectory": {}}

    isaac_dir = export_isaac_from_record(
        record,
        str(tmp_path),
        scan_id="s",
        scale=1.0,
        scale_source="anchor:prescaled",
        anchor_scale=0.5,
        reconstruct=False,
    )
    manifest = json.loads((tmp_path / "s" / "isaac" / "manifest.json").read_text())
    assert manifest["alignment"]["metric"] is True
    assert manifest["alignment"]["scale_source"] == "anchor:prescaled"
    assert manifest["alignment"]["scale"] == pytest.approx(1.0)
    assert "Metric-anchor" in manifest["notes"] and "0.5" in manifest["notes"]


def test_export_isaac_from_record_require_metric_rejects_unscaled(tmp_path):
    """Defense-in-depth: with no scale and require_metric=True the writer refuses to author an
    up-to-scale scene (the route gate should prevent ever reaching this, but belt-and-braces)."""
    from server.export.isaac.writer import export_isaac_from_record

    positions = np.random.default_rng(2).uniform(-1, 1, size=(50, 3)).astype(np.float32)
    colors = np.zeros((50, 3), np.uint8)
    record = {"scan_id": "s", "cloud": (positions, colors), "trajectory": {}}
    with pytest.raises(ValueError):
        export_isaac_from_record(record, str(tmp_path), scan_id="s", scale=None, require_metric=True)


def _store_with_floor_scene(tmp_path):
    """A persisted scan whose cloud has a dominant floor plane (so gravity/mesh recovery is
    meaningful) + a trajectory of cameras above it. No vggt_slam / live Solver."""
    from server.scene_report.schemas import SceneFacts, SceneReport
    from server.scene_report.store import ModalScenePersistence

    rng = np.random.default_rng(7)
    floor = rng.uniform([-2, -2, 0], [2, 2, 0], size=(1500, 3)).astype(np.float32)
    floor[:, 2] = 0.0
    above = rng.uniform([-1, -1, 0.3], [1, 1, 1.8], size=(300, 3)).astype(np.float32)
    positions = np.vstack([floor, above]) * np.float32(4.0)  # up-to-scale SLAM cloud
    colors = rng.integers(0, 255, size=(len(positions), 3), dtype=np.uint8)
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(3)])
    for i, x in enumerate((-0.5, 0.0, 0.5)):
        poses[i, :3, 3] = np.array([x, 0.0, 1.5], np.float32) * 4.0
    trajectory = {
        "poses": poses,
        "intrinsics": np.tile(np.array([500, 500, 320, 240], np.float32), (3, 1)),
        "source_frame_id": np.arange(3, dtype=np.float32),
    }
    store = ModalScenePersistence({}, str(tmp_path))
    store.save_scene(
        "user-a", "scan1", SceneReport(summary="s", room_type="office"), SceneFacts(),
        points=(positions, colors), trajectory=trajectory, splat_bytes=None,
    )
    return store


def test_route_isaac_full_zip_end_to_end(monkeypatch, tmp_path):
    """Full happy path: real USD toolchain present -> the route streams a real zip with
    scene.usd + trajectory.usd + manifest.json, gravity-aligned and METRIC (via ?scale). Skipped
    on a bare host — real USD generation is an open verification row (docs/isaac-export.md §6)."""
    pytest.importorskip("pxr", reason="usd-core (pxr) not installed")
    pytest.importorskip("open3d", reason="open3d not installed")

    app_mod = load_app_module(monkeypatch)
    store = _store_with_floor_scene(tmp_path)
    app_mod.configure_scene_persistence(store)
    _fake_request(
        monkeypatch, app_mod, args={"format": "isaac_usd", "scale": "0.25"}, claims={"sub": "user-a"}
    )

    out = app_mod.export_scene_route("scan1")
    # FakeFlask.send_file returns {"file": args, "kwargs": kwargs}; a JSON error would be a tuple.
    assert isinstance(out, dict) and "file" in out, f"expected a streamed zip, got {out!r}"
    assert out["kwargs"]["download_name"] == "scan-scan1-isaac_usd.zip"
    assert out["kwargs"]["mimetype"] == "application/zip"

    zip_bytes = out["file"][0].getvalue()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert any(n.endswith("scan1/isaac/scene.usd") for n in names)
        assert any(n.endswith("scan1/isaac/trajectory.usd") for n in names)
        manifest_name = next(n for n in names if n.endswith("manifest.json"))
        manifest = json.loads(zf.read(manifest_name))

    assert manifest["alignment"]["metric"] is True
    assert manifest["alignment"]["scale_source"] == "user:factor"
    assert manifest["alignment"]["scale"] == pytest.approx(0.25)
    assert manifest["alignment"]["gravity_aligned"] is True
    assert manifest["num_keyframes"] == 3
    # A collider mesh (open3d present) OR an honest points-only fallback — both are valid.
    assert manifest["geometry"]["representation"] in ("mesh", "mesh+points", "points")
    if manifest["geometry"]["representation"].startswith("mesh"):
        assert manifest["geometry"]["has_collision"] is True


def test_route_isaac_anchor_source_full_zip_prescaled(monkeypatch, tmp_path):
    """Full path via a real Metric anchor: the route consumes the anchor-calibrated (already
    metric) geometry at scale 1.0 and stamps anchor provenance — never double-scaled."""
    pytest.importorskip("pxr", reason="usd-core (pxr) not installed")
    pytest.importorskip("open3d", reason="open3d not installed")

    app_mod = load_app_module(monkeypatch)
    store = _store_with_floor_scene(tmp_path)
    app_mod.configure_scene_persistence(store)
    result = app_mod._apply_metric_anchor(store, "user-a", "scan1", [0, 0, 0], [4, 0, 0], 1.0)
    # anchor pointer persisted by the real route; drive default (no ?source, no ?scale) export
    app_mod._persist_derived_pointer(
        store, "user-a", "scan1",
        app_mod._derived_pointer(
            "anchor",
            cloud_key=result["calibrated_cloud_key"],
            trajectory_key=result["calibrated_trajectory_key"],
            splat_key=result["calibrated_splat_key"],
            scale_factor=result["scale_factor"],
        ),
    )
    _fake_request(monkeypatch, app_mod, args={"format": "isaac_usd"}, claims={"sub": "user-a"})

    out = app_mod.export_scene_route("scan1")
    assert isinstance(out, dict) and "file" in out
    with zipfile.ZipFile(io.BytesIO(out["file"][0].getvalue())) as zf:
        manifest = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith("manifest.json"))))
    assert manifest["alignment"]["metric"] is True
    assert manifest["alignment"]["scale_source"] == "anchor:prescaled"
    assert manifest["alignment"]["scale"] == pytest.approx(1.0)  # geometry already metric
