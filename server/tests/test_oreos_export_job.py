"""Prepared-export tests: the artifact vocabulary + retention rule
(`server/oreos/export_artifacts.py`), the atomic path-based publish
(`ModalScenePersistence.save_derived_artifact_file`), and the two broker routes.

GPU-free and Modal-free. Same harness as tests/test_demo_lod.py: real Flask +
``test_client`` on a freshly imported ``server.oreos`` blueprint, ``server.app`` stubbed
via ``sys.modules``, the jobs store swapped for a plain dict, and the export job spawner
swapped for a recording fake.

What these lock down, in the order the demo depends on them:
  * a re-prepare OVERWRITES its slot — N clicks cost one GB-class file, not N;
  * a half-written archive is never servable (atomic publish);
  * the panel's numbers come from the file on disk, never from a remembered estimate;
  * a second Prepare cannot fan out a second container onto the same slot.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

flask = pytest.importorskip("flask")

from server.oreos import export_artifacts as ea
from server.scene_report.store import ModalScenePersistence


# ---------------------------------------------------------------------------
# export_artifacts.py — keys, index, staleness, pruning
# ---------------------------------------------------------------------------


def test_slot_keys_are_deterministic_per_format():
    """One slot per (scene, format) IS the retention rule — the key may not carry a
    timestamp or a job id, or every click would leak a new GB-class file."""
    assert ea.zip_relative_key("openreality") == "demo/export/openreality.zip"
    assert ea.zip_derived_key("openreality") == "derived/demo/export/openreality.zip"
    assert ea.zip_derived_key("groot_lerobot_v2") == "derived/demo/export/groot_lerobot_v2.zip"
    assert ea.INDEX_DERIVED_KEY == "derived/demo/export/index.json"
    assert ea.zip_derived_key("openreality") == ea.zip_derived_key("openreality")


def test_merge_slot_keeps_the_other_format():
    a = ea.slot_record(
        export_format="openreality", source_key="original", zip_bytes=1, tree_bytes=2,
        file_count=3, build_seconds=4.0,
    )
    b = ea.slot_record(
        export_format="groot_lerobot_v2", source_key="original", zip_bytes=5, tree_bytes=6,
        file_count=7, build_seconds=8.0,
    )
    index = ea.merge_slot(ea.merge_slot({}, a), b)
    assert set(index["slots"]) == {"openreality", "groot_lerobot_v2"}
    assert ea.live_slot(index, "openreality")["bytes"] == 1
    # re-preparing one format replaces only that slot
    a2 = ea.slot_record(
        export_format="openreality", source_key="original", zip_bytes=99, tree_bytes=2,
        file_count=3, build_seconds=4.0,
    )
    index = ea.merge_slot(index, a2)
    assert ea.live_slot(index, "openreality")["bytes"] == 99
    assert ea.live_slot(index, "groot_lerobot_v2")["bytes"] == 5


def test_staleness_reasons_are_specific():
    slot = ea.slot_record(
        export_format="openreality", source_key="original", zip_bytes=1, tree_bytes=1,
        file_count=1, build_seconds=1.0,
    )
    assert ea.staleness(slot, "original") == (False, None)

    stale, reason = ea.staleness(slot, "derived/anchor/2026/cloud.ply")
    assert stale and "derived/anchor/2026/cloud.ply" in reason

    old = {**slot, "built_at": slot["built_at"] - (ea.EXPORT_STALE_S + 60)}
    stale, reason = ea.staleness(old, "original")
    assert stale and "ago" in reason


def test_prune_removes_partials_and_dead_formats(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    keep_names = {"openreality.zip", "groot_lerobot_v2.zip", "index.json"}
    for name in [*keep_names, ".openreality.zip.partial", "legacy_format.zip"]:
        (d / name).write_bytes(b"x")
    removed = ea.prune_orphans(str(d), keep_names, log=lambda *_a: None)
    assert set(removed) == {".openreality.zip.partial", "legacy_format.zip"}
    assert set(os.listdir(d)) == keep_names


# ---------------------------------------------------------------------------
# store — path-based, atomic derived publish
# ---------------------------------------------------------------------------


def test_save_derived_artifact_file_publishes_and_overwrites(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    src = tmp_path / "a.zip"
    src.write_bytes(b"first")
    key = store.save_derived_artifact_file("u", "s", "demo/export/openreality.zip", str(src))
    assert key == "derived/demo/export/openreality.zip"
    path = store.get_derived_artifact_path("u", "s", key)
    assert open(path, "rb").read() == b"first"
    assert os.path.isfile(src)  # copy, not move, by default

    # a rebuild overwrites the SAME path — the retention rule, at the filesystem
    src2 = tmp_path / "b.zip"
    src2.write_bytes(b"second-and-longer")
    store.save_derived_artifact_file("u", "s", "demo/export/openreality.zip", str(src2), move=True)
    assert store.get_derived_artifact_path("u", "s", key) == path
    assert open(path, "rb").read() == b"second-and-longer"
    assert not os.path.exists(src2)  # move=True consumed the source
    assert len(os.listdir(os.path.dirname(path))) == 1  # no second copy left behind


def test_failed_publish_leaves_no_servable_partial(tmp_path, monkeypatch):
    """A container that dies mid-write must not leave something the download route
    would happily stream as a complete archive."""
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    src = tmp_path / "a.zip"
    src.write_bytes(b"good")
    store.save_derived_artifact_file("u", "s", "demo/export/openreality.zip", str(src))
    path = store.get_derived_artifact_path("u", "s", "derived/demo/export/openreality.zip")

    import shutil as _shutil

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(_shutil, "copyfile", boom)
    with pytest.raises(OSError):
        store.save_derived_artifact_file("u", "s", "demo/export/openreality.zip", str(src))

    # the previous good archive survives, and no .partial is left in the directory
    assert open(path, "rb").read() == b"good"
    assert [n for n in os.listdir(os.path.dirname(path)) if n.startswith(".")] == []


def test_delete_derived_artifact_only_touches_derived(tmp_path):
    store = ModalScenePersistence({}, str(tmp_path / "blobs"))
    src = tmp_path / "a.zip"
    src.write_bytes(b"x")
    key = store.save_derived_artifact_file("u", "s", "demo/export/openreality.zip", str(src))
    assert store.delete_derived_artifact("u", "s", key) is True
    assert store.get_derived_artifact_path("u", "s", key) is None
    assert store.delete_derived_artifact("u", "s", key) is False
    # a non-derived key resolves to nothing and is a no-op, never a path escape
    assert store.delete_derived_artifact("u", "s", "splat.ply") is False


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def _fresh_demo_package():
    for name in [m for m in list(sys.modules) if m == "server.oreos" or (m.startswith("server.oreos.") and not m.startswith("server.oreos.recordings"))]:
        sys.modules.pop(name, None)
    return importlib.import_module("server.oreos")


@pytest.fixture()
def export_app(monkeypatch, tmp_path):
    demo_pkg = _fresh_demo_package()
    store = ModalScenePersistence({}, str(tmp_path))
    stub = types.SimpleNamespace(_scene_persistence=store, _auth_user_id=lambda: "user-a")
    import server as server_pkg

    monkeypatch.setitem(sys.modules, "server.app", stub)
    monkeypatch.setattr(server_pkg, "app", stub, raising=False)

    app = flask.Flask(__name__)
    app.register_blueprint(demo_pkg.oreos_bp)

    jobs = sys.modules["server.oreos.jobs"]
    jobs_store: dict = {}
    jobs.configure_jobs_store(jobs_store)
    routes = sys.modules["server.oreos.routes_export_job"]
    spawned: list[dict] = []
    routes.configure_export_spawner(lambda **kw: spawned.append(kw))
    store._store["user-a:scan-1"] = {"scan_id": "scan-1", "user_id": "user-a"}

    def publish_slot(fmt="openreality", source_key="original", payload=b"ZIPBYTES"):
        """Simulate what demo_export_job writes when it finishes."""
        blob = tmp_path / f"{fmt}.src.zip"
        blob.write_bytes(payload)
        store.save_derived_artifact_file(
            "user-a", "scan-1", ea.zip_relative_key(fmt), str(blob), move=True
        )
        slot = ea.slot_record(
            export_format=fmt, source_key=source_key, zip_bytes=len(payload),
            tree_bytes=len(payload) * 2, file_count=9, build_seconds=45.1,
        )
        index = ea.merge_slot(ea.read_index(store, "user-a", "scan-1"), slot)
        store.save_derived_artifact(
            "user-a", "scan-1", ea.INDEX_RELATIVE_KEY, ea.index_json_bytes(index)
        )
        return slot

    yield types.SimpleNamespace(
        client=app.test_client(), store=store, routes=routes, spawned=spawned,
        jobs=jobs, jobs_store=jobs_store, publish_slot=publish_slot,
    )
    routes.configure_export_spawner(None)
    jobs.configure_jobs_store(None)


PREPARED = "/api/scenes/scan-1/demo/export/prepared"
PREPARE = "/api/scenes/scan-1/demo/export/prepare"


def test_prepared_reports_none_before_anything_is_built(export_app):
    r = export_app.client.get(PREPARED)
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "none"
    assert body["artifact"] is None
    assert body["format"] == "openreality"  # the documented default


def test_prepared_unknown_scene_is_404(export_app):
    assert export_app.client.get("/api/scenes/nope/demo/export/prepared").status_code == 404
    assert export_app.client.post("/api/scenes/nope/demo/export/prepare", json={}).status_code == 404


def test_format_and_source_are_validated(export_app):
    # isaac_usd is now a PREPARED format. It was excluded when the axis was thought to be
    # output size (5.5 MB), but the cost is the BUILD: Poisson meshing measured 378 s and
    # a 63.3M-point load needs tens of GB, so it can no more live in a 150 s request than
    # a 1.39 GB zip can. See export_artifacts.EXPORT_JOB_FORMATS.
    assert export_app.client.get(f"{PREPARED}?format=isaac_usd").status_code == 200
    assert export_app.client.post(PREPARE, json={"format": "isaac_usd"}).status_code == 202
    export_app.spawned.clear()
    # A format nobody exports is still a hard 400 rather than a silent default.
    assert export_app.client.get(f"{PREPARED}?format=usdz_pointcloud").status_code == 400
    assert export_app.client.post(PREPARE, json={"format": "usdz_pointcloud"}).status_code == 400
    # a bogus derived selector is a hard 404, exactly as on the zip route — never a
    # silent degrade to the original geometry
    r = export_app.client.post(PREPARE, json={"source": "derived/nope/cloud.ply"})
    assert r.status_code == 404 and r.get_json()["error"] == "unknown_derived_key"
    assert export_app.spawned == []


def test_prepare_spawns_then_409s_while_running(export_app):
    r = export_app.client.post(PREPARE, json={})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    assert len(export_app.spawned) == 1
    assert export_app.spawned[0]["scan_id"] == "scan-1"
    assert export_app.spawned[0]["export_format"] == "openreality"
    # the polling record is seeded by the ROUTE so the panel never sees a 404 while the
    # container boots
    assert export_app.jobs_store[job_id]["status"] == "queued"

    r2 = export_app.client.post(PREPARE, json={})
    assert r2.status_code == 409
    assert r2.get_json()["error"] == "export_job_active"
    assert r2.get_json()["job_id"] == job_id
    assert len(export_app.spawned) == 1

    # ...but the OTHER format is a different slot and may run concurrently
    r3 = export_app.client.post(PREPARE, json={"format": "groot_lerobot_v2"})
    assert r3.status_code == 202
    assert len(export_app.spawned) == 2


def test_prepared_reports_running_stage_from_the_jobs_store(export_app):
    job_id = export_app.client.post(PREPARE, json={}).get_json()["job_id"]
    export_app.jobs_store[job_id] = {
        "job_id": job_id, "user_id": "user-a", "status": "running",
        "stage": "compress", "tree_bytes": 1_493_172_224,
    }
    body = export_app.client.get(PREPARED).get_json()
    assert body["status"] == "running"
    assert body["job_id"] == job_id
    assert body["stage"] == "compress"
    assert body["tree_bytes"] == 1_493_172_224


def test_prepared_ready_carries_the_real_on_disk_size(export_app):
    export_app.publish_slot(payload=b"x" * 4242)
    body = export_app.client.get(PREPARED).get_json()
    assert body["status"] == "ready"
    art = body["artifact"]
    assert art["bytes"] == 4242  # re-stat'ed, not the index's remembered number
    assert art["file_count"] == 9
    assert art["download_path"] == "/api/scenes/scan-1/derived/demo/export/openreality.zip"
    assert art["download_filename"] == "scan-scan-1-openreality.zip"
    assert body["stale"] is False


def test_prepared_index_without_a_blob_is_not_ready(export_app):
    """Index and blob are separate writes. If the archive is gone, 'ready' would send the
    operator to a 404 — report the truth instead."""
    slot = ea.slot_record(
        export_format="openreality", source_key="original", zip_bytes=10, tree_bytes=20,
        file_count=1, build_seconds=1.0,
    )
    export_app.store.save_derived_artifact(
        "user-a", "scan-1", ea.INDEX_RELATIVE_KEY, ea.index_json_bytes(ea.merge_slot({}, slot))
    )
    assert export_app.client.get(PREPARED).get_json()["status"] == "none"


def test_fresh_artifact_short_circuits_prepare_unless_forced(export_app):
    export_app.publish_slot()
    r = export_app.client.post(PREPARE, json={})
    assert r.status_code == 200 and r.get_json()["status"] == "ready"
    assert export_app.spawned == []  # no container spent rebuilding a byte-identical zip

    r2 = export_app.client.post(PREPARE, json={"force": True})
    assert r2.status_code == 202
    assert len(export_app.spawned) == 1


def test_stale_geometry_rebuilds_without_force(export_app):
    """A slot built from the original must not be served as current for an anchored
    export — the bytes really are different geometry."""
    export_app.publish_slot(source_key="original")
    export_app.store.save_derived_artifact("user-a", "scan-1", "anchor/1/cloud.ply", b"c")
    src = "derived/anchor/1/cloud.ply"

    body = export_app.client.get(f"{PREPARED}?source={src}").get_json()
    assert body["status"] == "ready" and body["stale"] is True
    assert "original" in body["stale_reason"]

    r = export_app.client.post(PREPARE, json={"source": src})
    assert r.status_code == 202  # rebuilt without needing force
    assert export_app.spawned[0]["source"] == src


def test_a_running_rebuild_still_offers_the_previous_artifact(export_app):
    export_app.publish_slot(payload=b"y" * 100)
    job_id = export_app.client.post(PREPARE, json={"force": True}).get_json()["job_id"]
    export_app.jobs_store[job_id] = {
        "job_id": job_id, "user_id": "user-a", "status": "running", "stage": "build_tree",
    }
    body = export_app.client.get(PREPARED).get_json()
    assert body["status"] == "running"
    assert body["artifact"]["bytes"] == 100  # last good download stays available


def test_failed_job_is_reported_as_error(export_app):
    job_id = export_app.client.post(PREPARE, json={}).get_json()["job_id"]
    export_app.jobs_store[job_id] = {
        "job_id": job_id, "user_id": "user-a", "status": "failed", "error": "ValueError: nope",
    }
    body = export_app.client.get(PREPARED).get_json()
    assert body["status"] == "error"
    assert body["error"] == "ValueError: nope"


def test_unauthenticated_is_401(export_app, monkeypatch):
    stub = sys.modules["server.app"]
    monkeypatch.setattr(stub, "_auth_user_id", lambda: None)
    assert export_app.client.get(PREPARED).status_code == 401
    assert export_app.client.post(PREPARE, json={}).status_code == 401


def test_routes_registered_once(export_app):
    rules = [str(r.rule) for r in export_app.client.application.url_map.iter_rules()]
    assert rules.count("/api/scenes/<scan_id>/demo/export/prepared") == 1
    assert rules.count("/api/scenes/<scan_id>/demo/export/prepare") == 1


def test_isaac_spawns_the_heavy_container_not_the_shared_one(monkeypatch):
    """The FORMAT picks the container.

    Isaac authors a Poisson mesh — 378 s measured on 5M points, and loading the founder's
    63.3M-point scene needs tens of GB. Running it on the shared 8 GB export container
    would OOM; sizing the shared container at 64 GB would bill that for every LeRobot zip.
    So the spawn chooses, and this pins which name it chooses.
    """
    import server.oreos.routes_export_job as rej

    picked: list = []

    class _FakeFn:
        def __init__(self, name):
            self.name = name

        def spawn(self, **kw):
            picked.append(self.name)

    class _FakeModal:
        class Function:
            @staticmethod
            def from_name(app, fn):
                return _FakeFn(fn)

    monkeypatch.setitem(__import__("sys").modules, "modal", _FakeModal)
    rej.configure_export_spawner(None)  # use the real lookup path
    try:
        rej._spawn_export_job(job_id="j", user_id="u", scan_id="s", export_format="isaac_usd")
        rej._spawn_export_job(job_id="j", user_id="u", scan_id="s", export_format="openreality")
    finally:
        rej.configure_export_spawner(None)

    assert picked == ["demo_isaac_export_job", "demo_export_job"]


def test_slot_records_decimation_so_nobody_reads_5m_as_63m():
    """A decimated export must say so in the slot.

    An Isaac scene built from a 5M subsample of a 63,308,835-point capture is a legitimate
    collider, and a lie if consumed as the scene's density — a 12x error. The counts ride
    on the slot rather than in a log line nobody reads.
    """
    from server.oreos import export_artifacts as ea

    slot = ea.slot_record(
        export_format="isaac_usd", source_key="original", zip_bytes=5_503_722,
        tree_bytes=0, file_count=3, build_seconds=378.1, job_id="j1",
        original_points=63_308_835, exported_points=5_000_000,
    )
    assert slot["decimated"] is True
    assert slot["original_points"] == 63_308_835
    assert slot["exported_points"] == 5_000_000

    whole = ea.slot_record(
        export_format="openreality", source_key="original", zip_bytes=1, tree_bytes=1,
        file_count=1, build_seconds=1.0, original_points=1000, exported_points=1000,
    )
    assert whole["decimated"] is False
