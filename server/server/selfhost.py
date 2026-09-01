"""Self-host composition: the whole broker in ONE process on local disk.

    OPENREALITY_AUTH=local OPENREALITY_LOCAL_TOKEN=<secret> \
        python -m server.selfhost --data-dir ~/openreality-data --port 8000

Mirrors ``modal_streaming.py::web()``'s wiring with local backends (a pickle
file-per-key store + a plain blob directory instead of modal.Dict + Volume) and
runs the workspace jobs of ``modal_oreos_{ingest,lod,anchor,export}.py`` as
in-process worker threads instead of Modal functions. The job glue below is a
deliberate mirror of those wrappers — when one of them changes, change the twin
here in the same commit.

What a self-hosted broker serves (parity with the hosted MCP workflow):
  upload video → recon job (needs a CUDA GPU + the ``vggt`` package installed)
  → persisted scene → scene features (measure/planes/nav/ground-frame/cards)
  → anchor / LOD / splat-import / export jobs → artifact downloads → API keys.

Not in self-host v1 (each answers with an honest failed/unsupported status):
  - robot-recording ingest (DimOS pipeline)
  - the isaac_usd export lane (usd-core + Poisson meshing; use the dataset
    formats openreality / groot_lerobot_v2)
  - live phone streaming sessions (the hosted deploy scales those onto per-user
    GPU workers; a self-hosted live scan runs ``python -m server.app`` instead)

Auth: pair with ``OPENREALITY_AUTH=local`` (see server/app.py) — the static
bearer bootstraps identity and `openreality-mcp login --token <bearer>` mints a
durable ork_ API key against the file-backed registry, exactly like hosted.
"""
from __future__ import annotations

import argparse
import base64
import os
import pickle
import queue
import secrets as _secrets
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional

_UPLOADS_DIRNAME = "_uploads"  # mirrors modal_oreos_ingest._UPLOADS_DIRNAME


# ---------------------------------------------------------------------------
# Local stores
# ---------------------------------------------------------------------------


class FileBackedDict:
    """Durable dict: one pickle file per key under ``root``.

    Satisfies every store contract in this codebase (``get``/``__setitem__``/
    ``pop`` for ModalScenePersistence, ``get``/``__setitem__`` for
    ApiKeyRegistry and the jobs store). Pickle rather than JSON because the
    persisted records mirror what modal.Dict holds (numpy scalars included).
    Trusted-local only: never point it at files you did not write.
    """

    def __init__(self, root: str):
        self._root = str(root)
        os.makedirs(self._root, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: Any) -> str:
        name = base64.urlsafe_b64encode(str(key).encode("utf-8")).decode("ascii")
        return os.path.join(self._root, name + ".pkl")

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            with open(self._path(key), "rb") as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            return default
        except Exception as exc:  # a corrupt entry must not take the route down
            print(f"[selfhost.store] unreadable entry for {key!r}: {exc}")
            return default

    def __getitem__(self, key: Any) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        path = self._path(key)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    pickle.dump(value, fh)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    def __contains__(self, key: Any) -> bool:
        return os.path.exists(self._path(key))

    def __delitem__(self, key: Any) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            raise KeyError(key) from None

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            value = self.get(key, default)
            try:
                os.remove(self._path(key))
            except FileNotFoundError:
                pass
            return value

    def keys(self) -> list:
        out = []
        for name in os.listdir(self._root):
            if name.endswith(".pkl"):
                try:
                    out.append(
                        base64.urlsafe_b64decode(name[: -len(".pkl")].encode("ascii")).decode("utf-8")
                    )
                except Exception:
                    continue
        return out

    def __len__(self) -> int:
        return len(self.keys())


# ---------------------------------------------------------------------------
# Local job runner (the modal Function.spawn twin)
# ---------------------------------------------------------------------------


class LocalJobRunner:
    """One daemon worker thread + queue: jobs run serialized, mirroring the
    hosted ``max_containers=1`` spend guard, and a crash in one job never takes
    the broker down (status reaches the jobs store via each job's publisher)."""

    def __init__(self) -> None:
        self._q: "queue.Queue[tuple[Callable[..., Any], dict]]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="selfhost-jobs", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[..., Any], **kwargs: Any) -> None:
        self._q.put((fn, kwargs))

    def _loop(self) -> None:
        while True:
            fn, kwargs = self._q.get()
            try:
                fn(**kwargs)
            except BaseException:  # job boundary — status already published by the job
                traceback.print_exc()
            finally:
                self._q.task_done()


def _publish(jobs_store: Any, job_id: Any, user_id: Any, *, status: str, stage: str,
             scan_id: Any = None, **extra: Any) -> None:
    from server.oreos.jobs import job_status_record

    try:
        jobs_store[str(job_id)] = job_status_record(
            job_id, user_id, status=status, stage=stage, scan_id=scan_id, **extra
        )
    except Exception as exc:  # status is telemetry — never kill the job over it
        print(f"[selfhost.jobs] status publish failed ({stage}): {exc}")


def _contain_upload(blob_root: str, upload_rel_path: str) -> str:
    """Resolve + contain a staged upload path inside the ``_uploads`` area —
    mirrors the containment check in modal_oreos_ingest.demo_recon_job."""
    rel = os.path.normpath(str(upload_rel_path)).lstrip("/")
    parts = rel.split(os.sep)
    path = os.path.realpath(os.path.join(blob_root, rel))
    root = os.path.realpath(blob_root)
    if ".." in parts or _UPLOADS_DIRNAME not in parts or not path.startswith(root + os.sep):
        raise ValueError(f"upload path escapes the {_UPLOADS_DIRNAME} area: {upload_rel_path!r}")
    return path


def _cleanup_upload(path: str) -> None:
    """Delete a staged upload's whole ``<upload_id>`` dir (or just the file when
    the layout is unexpected) — mirrors the modal wrappers' cleanup."""
    upload_dir = os.path.dirname(path)
    if os.path.basename(os.path.dirname(upload_dir)) == _UPLOADS_DIRNAME:
        shutil.rmtree(upload_dir, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


class _SourceStampingPersistence:
    """save_scene interceptor — the local twin of modal_oreos_ingest's
    ``_DemoScenePersistence``: stamps ``source=`` on the persisted record and
    marks the persist stage; everything else delegates."""

    def __init__(self, inner: Any, source: str, on_persist: Callable[[], None]):
        self._inner = inner
        self._source = source
        self._on_persist = on_persist

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def save_scene(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("source", self._source)
        self._on_persist()
        return self._inner.save_scene(*args, **kwargs)


# ---------------------------------------------------------------------------
# Job bodies — each mirrors its modal_oreos_* wrapper; keep them in sync.
# ---------------------------------------------------------------------------


def run_recon_job(persistence: Any, jobs_store: Any, *, job_id: str, user_id: str,
                  upload_rel_path: str, scan_id: Optional[str] = None,
                  label: Optional[str] = None, source: str = "recon_video",
                  extract_fps: float = 2.0, submap_size: int = 32,
                  export_splat: bool = True, detect_objects: bool = True,
                  detection_env: Optional[dict] = None,
                  demo_extensions: bool = True) -> None:
    """Local twin of modal_oreos_ingest.demo_recon_job (GPU: needs ``vggt``)."""
    scan_id = scan_id or uuid.uuid4().hex

    def publish(status: str, stage: str, **extra: Any) -> None:
        _publish(jobs_store, job_id, user_id, status=status, stage=stage,
                 scan_id=scan_id, **extra)

    blob_root = persistence._blob_root
    try:
        video_path = _contain_upload(blob_root, upload_rel_path)
    except ValueError:
        publish("failed", "upload", error="invalid_upload_path")
        raise
    if not os.path.isfile(video_path):
        publish("failed", "upload", error="upload_not_found")
        raise FileNotFoundError(video_path)

    # Import the SLAM stack only for a job that will actually run — the failure
    # paths above stay importable on GPU-free boxes (and in CI).
    from server.reconstruct_pilot import reconstruct_pilot

    for key, value in (detection_env or {}).items():
        os.environ[key] = str(value)

    stamped = _SourceStampingPersistence(
        persistence, source, on_persist=lambda: publish("running", "persist")
    )

    publish("running", "recon")
    t0 = time.time()
    try:
        reconstruct_pilot(
            videos=[video_path],
            user_id=user_id,
            export_splat=export_splat,
            detect_objects=detect_objects,
            extract_fps=extract_fps,
            blob_root=blob_root,
            submap_size=submap_size,
            scan_id=scan_id,
            mode="single",
            persistence=stamped,
            labels=[label] if label else None,
            demo_index=demo_extensions,
            persist_trajectory=demo_extensions,
        )
    except BaseException as exc:  # SystemExit included — reconstruct_pilot raises it
        publish("failed", "recon", error=str(exc) or exc.__class__.__name__)
        raise

    record = persistence.get_scene(user_id, scan_id)
    if record is None:
        publish("failed", "persist", error="nothing_persisted")
        raise SystemExit(f"reconstruction persisted no scene for scan {scan_id}")

    _cleanup_upload(video_path)
    elapsed = round(time.time() - t0, 1)
    publish(
        "done", "done", elapsed_s=elapsed,
        point_count=record.get("point_count"),
        has_splat=bool(record.get("splat_key")),
        has_trajectory=bool(record.get("trajectory_key")),
        keyframe_count=len(record.get("keyframes") or []),
    )
    print(f"[selfhost.recon] job {job_id} done → scan {scan_id} in {elapsed:.0f}s")


def run_splat_job(persistence: Any, jobs_store: Any, *, job_id: str, user_id: str,
                  upload_rel_path: str, scan_id: Optional[str] = None,
                  label: Optional[str] = None, filename: Optional[str] = None) -> None:
    """Local twin of modal_oreos_ingest.demo_splat_import_job (CPU only)."""
    scan_id = scan_id or uuid.uuid4().hex
    display_name = filename or os.path.basename(str(upload_rel_path))

    def publish(status: str, stage: str, **extra: Any) -> None:
        _publish(jobs_store, job_id, user_id, status=status, stage=stage,
                 scan_id=scan_id, kind="splat_import", **extra)

    try:
        splat_path = _contain_upload(persistence._blob_root, upload_rel_path)
    except ValueError:
        publish("failed", "upload", error="invalid_upload_path")
        raise
    if not os.path.isfile(splat_path):
        publish("failed", "upload", error="upload_not_found")
        raise FileNotFoundError(splat_path)

    from server.oreos import splat_import

    t0 = time.time()
    converted = None
    try:
        if splat_path.lower().endswith(".spz"):
            from server.oreos import spz as spz_mod

            publish("running", "validate", note="decoding .spz")
            converted = splat_path + ".ply"
            spz_mod.spz_to_ply_file(
                splat_path, converted, max_gaussians=splat_import.MAX_GAUSSIANS
            )
            parse_path = converted
        else:
            parse_path = splat_path

        result = splat_import.import_splat(
            persistence, user_id, parse_path, display_name,
            label=label, scan_id=scan_id,
            on_stage=lambda name: publish("running", name),
        )
    except splat_import.SplatRejected as exc:
        publish("failed", "validate", error=exc.error, detail=exc.detail, **exc.extra)
        _cleanup_upload(splat_path)
        return
    except BaseException as exc:
        publish("failed", "parse", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if converted:
            try:
                os.remove(converted)
            except OSError:
                pass

    _cleanup_upload(splat_path)
    elapsed = round(time.time() - t0, 1)
    publish(
        "done", "done", elapsed_s=elapsed,
        gaussian_count=result["gaussian_count"],
        point_count=result["point_count"],
        has_splat=True, has_trajectory=False, keyframe_count=0,
        **({"render_advisory": result["render_advisory"]}
           if "render_advisory" in result else {}),
    )


def run_lod_job(persistence: Any, jobs_store: Any, *, job_id: str, user_id: str,
                scan_id: str, levels: Optional[list] = None, method: str = "auto",
                force: bool = False) -> None:
    """Local twin of modal_oreos_lod.demo_lod_job (CPU only)."""
    from server.oreos import lod as lod_mod

    started = time.time()

    def publish(status: str, stage: str, **extra: Any) -> None:
        _publish(jobs_store, job_id, user_id, status=status, stage=stage,
                 scan_id=scan_id, kind="lod", **extra)

    publish("running", "reading")
    try:
        splat_path = persistence.get_splat_path(user_id, scan_id)
        if not splat_path or not os.path.isfile(splat_path):
            raise FileNotFoundError(f"no splat for {user_id}:{scan_id}")

        if not force:
            existing = persistence.get_derived_artifact_path(
                user_id, scan_id, f"derived/{lod_mod.LOD_INDEX_KEY}"
            )
            if existing and os.path.isfile(existing):
                publish("done", "done", elapsed_s=round(time.time() - started, 1), skipped=True)
                return

        src = lod_mod.SplatSource(splat_path)
        drop_normals = lod_mod.normals_are_zero(src)
        budgets = [int(b) for b in levels] if levels else list(lod_mod.DEFAULT_LEVELS)
        plan = lod_mod.plan_levels(src.count, budgets)

        tmpdir = tempfile.mkdtemp(prefix="lod_")
        entries: list = []
        try:
            for i, budget in enumerate(plan):
                publish("running", f"level_{lod_mod.level_name(budget)}",
                        level_index=i, level_count=len(plan))
                ply_tmp = os.path.join(tmpdir, f"splat_{lod_mod.level_name(budget)}.ply")
                spz_tmp = os.path.join(tmpdir, f"splat_{lod_mod.level_name(budget)}.spz")
                entry = lod_mod.build_level(
                    src, budget, ply_tmp, method=method,
                    drop_normals=drop_normals, spz_path=spz_tmp, log=print,
                )
                if entry.get("spz_key"):
                    with open(spz_tmp, "rb") as fh:
                        persistence.save_derived_artifact(
                            user_id, scan_id, lod_mod.level_key(budget, lod_mod.SPZ_SUFFIX), fh.read()
                        )
                with open(ply_tmp, "rb") as fh:
                    persistence.save_derived_artifact(
                        user_id, scan_id, lod_mod.level_key(budget, lod_mod.PLY_SUFFIX), fh.read()
                    )
                entries.append(entry)

            full_info = None
            if src.count <= lod_mod.FULL_DETAIL_MAX_GAUSSIANS:
                publish("running", "full_spz")
                full_tmp = os.path.join(tmpdir, "full.spz")
                try:
                    rep = lod_mod.encode_spz(splat_path, full_tmp)
                    with open(full_tmp, "rb") as fh:
                        persistence.save_derived_artifact(
                            user_id, scan_id, lod_mod.full_key(lod_mod.SPZ_SUFFIX), fh.read()
                        )
                    full_info = {
                        "key": lod_mod.full_key(lod_mod.SPZ_SUFFIX),
                        "gaussians": src.count,
                        "bytes": int(rep["output_bytes"]),
                        "size_reduction": round(src.file_bytes / max(int(rep["output_bytes"]), 1), 2),
                    }
                except lod_mod.SpzEncodeError as exc:
                    full_info = {"error": str(exc)[:300]}
            else:
                full_info = {
                    "unavailable_reason": "too_many_gaussians",
                    "gaussians": src.count,
                    "ceiling": lod_mod.FULL_DETAIL_MAX_GAUSSIANS,
                }

            doc = lod_mod.build_index(
                scan_id=scan_id, source_count=src.count, source_bytes=src.file_bytes,
                entries=entries, generator=method, source_key="splat.ply",
                extra={
                    "full_detail": full_info,
                    "built_at": time.time(),
                    "build_seconds": round(time.time() - started, 1),
                    "normals_dropped": bool(drop_normals),
                },
            )
            persistence.save_derived_artifact(
                user_id, scan_id, lod_mod.LOD_INDEX_KEY, lod_mod.index_json_bytes(doc)
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            src.close()

        publish("done", "done", elapsed_s=round(time.time() - started, 1),
                levels=len(entries), default_level=doc.get("default_level"))
    except Exception as exc:
        traceback.print_exc()
        publish("failed", "error", error=f"{type(exc).__name__}: {exc}"[:400],
                elapsed_s=round(time.time() - started, 1))
        raise


def run_anchor_job(persistence: Any, jobs_store: Any, *, job_id: str, user_id: str,
                   scan_id: str, scale_factor: float, applied_at: str,
                   measured: float, distance_m: float) -> None:
    """Local twin of modal_oreos_anchor.demo_anchor_job (CPU only)."""
    from server.scene_report import anchor as anchor_impl

    started = time.time()

    def publish(status: str, stage: str, **extra: Any) -> None:
        _publish(jobs_store, job_id, user_id, status=status, stage=stage,
                 scan_id=scan_id, kind="anchor", **extra)

    publish("running", "materializing")
    try:
        artifacts = anchor_impl.materialize_anchor_artifacts(
            persistence, user_id, scan_id, float(scale_factor)
        )
        record = persistence.get_scene(user_id, scan_id) or {}
        current = record.get("derived_latest") or {}
        if not isinstance(current, dict) or current.get("applied_at") == applied_at:
            anchor_impl.persist_derived_pointer(
                persistence, user_id, scan_id,
                anchor_impl.derived_pointer(
                    "anchor",
                    cloud_key=artifacts.get("calibrated_cloud_key"),
                    trajectory_key=artifacts.get("calibrated_trajectory_key"),
                    splat_key=artifacts.get("calibrated_splat_key"),
                    applied_at=applied_at,
                    scale_factor=float(scale_factor),
                ),
            )
        publish(
            "done", "done", elapsed_s=round(time.time() - started, 1),
            scale_factor=float(scale_factor), measured_distance=float(measured),
            distance_m=float(distance_m),
            **{k: v for k, v in artifacts.items() if k.startswith("calibrated_")},
        )
    except KeyError as exc:
        msg = "scan has no stored point cloud to calibrate" if "no_geometry" in str(exc) else str(exc)
        publish("error", "materializing", error=msg, elapsed_s=round(time.time() - started, 1))
    except Exception as exc:
        traceback.print_exc()
        publish("error", "materializing", error=str(exc), elapsed_s=round(time.time() - started, 1))


def run_export_job(persistence: Any, jobs_store: Any, *, job_id: str, user_id: str,
                   scan_id: str, export_format: str = "openreality",
                   source: Optional[str] = None) -> None:
    """Local twin of modal_oreos_export.demo_export_job — dataset formats only
    (``isaac_usd`` needs usd-core + Poisson meshing and is not in self-host v1)."""
    from server.oreos import export_artifacts as ea
    from server.export.record import load_record_from_store, resolved_source_key
    from server.export.zip_builder import build_export_zip_file

    started = time.time()
    facts: dict = {}

    def publish(status: str, stage: str, **extra: Any) -> None:
        _publish(jobs_store, job_id, user_id, status=status, stage=stage,
                 scan_id=scan_id, kind="export", export_format=export_format,
                 elapsed_s=round(time.time() - started, 1), **extra)

    publish("running", "reading")
    try:
        if export_format == "isaac_usd":
            raise ValueError("isaac_usd export is not supported on a self-hosted broker (v1)")
        if export_format not in ea.EXPORT_JOB_FORMATS:
            raise ValueError(f"unknown export format {export_format!r}")

        record = persistence.get_scene(user_id, scan_id)
        if not record:
            raise FileNotFoundError(f"no scene {user_id}:{scan_id}")

        normalized = load_record_from_store(persistence, user_id, scan_id, source=source)
        if normalized is None:
            raise ValueError(
                "scan has no persisted per-keyframe trajectory (pre-Stage-4 scan); "
                "cannot export GPU-free"
            )
        source_key = resolved_source_key(record, source)

        export_dir = persistence.derived_dir(user_id, scan_id, ea.EXPORT_ROOT)
        os.makedirs(export_dir, exist_ok=True)
        keep = {os.path.basename(ea.zip_relative_key(f)) for f in ea.EXPORT_JOB_FORMATS}
        keep.add(os.path.basename(ea.INDEX_RELATIVE_KEY))
        ea.prune_orphans(export_dir, keep, log=print)

        def on_stage(stage: str, **numbers: Any) -> None:
            facts.update(numbers)
            publish("running", stage, **facts)

        zip_path = build_export_zip_file(
            normalized, export_format, slot_dir=export_dir, on_stage=on_stage
        )
        try:
            zip_bytes = os.path.getsize(zip_path)
            facts["zip_bytes"] = zip_bytes
            publish("running", "persist", **facts)
            key = persistence.save_derived_artifact_file(
                user_id, scan_id, ea.zip_relative_key(export_format), zip_path, move=True
            )
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

        slot = ea.slot_record(
            export_format=export_format, source_key=source_key, zip_bytes=zip_bytes,
            tree_bytes=int(facts.get("tree_bytes") or 0),
            file_count=int(facts.get("file_count") or 0),
            build_seconds=time.time() - started, job_id=job_id,
        )
        index = ea.merge_slot(ea.read_index(persistence, user_id, scan_id), slot)
        persistence.save_derived_artifact(
            user_id, scan_id, ea.INDEX_RELATIVE_KEY, ea.index_json_bytes(index)
        )
        publish("done", "done", key=key, **facts)
    except Exception as exc:
        traceback.print_exc()
        publish("failed", "error", error=f"{type(exc).__name__}: {exc}"[:400])
        raise


def run_recording_job(jobs_store: Any, *, job_id: str, user_id: str,
                      scan_id: Optional[str] = None, **_ignored: Any) -> None:
    """Robot-recording ingest is NOT in self-host v1 (the DimOS pipeline needs its
    own dependency stack) — answer with an honest failed status, never a hang."""
    _publish(jobs_store, job_id, user_id, status="failed", stage="upload",
             scan_id=scan_id, kind="recording",
             error="recording_ingest_not_supported_selfhost")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def build_selfhost_app(data_dir: str, runner: Optional[LocalJobRunner] = None) -> Any:
    """Wire the broker onto local backends and return the ASGI app.

    The mirror of ``modal_streaming.py::web()``: same configure_* seams, local
    implementations. Deliberately NOT wired: ``serve_frontend`` (headless — the
    MCP workflow needs no SPA) and ``configure_gpu_session_broker`` (live phone
    streaming stays a hosted concern; a self-hosted live scan runs
    ``python -m server.app`` directly).

    Import note: ``server.app`` reads OPENREALITY_AUTH at import — set the auth
    env before calling this (``main()`` does).
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    records_dir = os.path.join(data_dir, "records")
    keys_dir = os.path.join(data_dir, "keys")
    blobs_dir = os.path.join(data_dir, "blobs")
    for d in (records_dir, keys_dir, blobs_dir):
        os.makedirs(d, exist_ok=True)

    from server import app as server_app
    from server.api_keys import ApiKeyRegistry
    from server.oreos import jobs as demo_jobs
    from server.oreos import routes_export_job, routes_ingest, routes_lod, routes_recordings
    from server.scene_report.store import ModalScenePersistence

    persistence = ModalScenePersistence(FileBackedDict(records_dir), blobs_dir)
    jobs_store: dict = {}
    runner = runner or LocalJobRunner()

    server_app.configure_scene_persistence(persistence)
    server_app.configure_api_key_registry(ApiKeyRegistry(FileBackedDict(keys_dir)))
    demo_jobs.configure_jobs_store(jobs_store)

    routes_ingest.configure_recon_spawner(
        lambda **kw: runner.submit(run_recon_job, persistence=persistence, jobs_store=jobs_store, **kw)
    )
    routes_ingest.configure_splat_spawner(
        lambda **kw: runner.submit(run_splat_job, persistence=persistence, jobs_store=jobs_store, **kw)
    )
    routes_lod.configure_lod_spawner(
        lambda **kw: runner.submit(run_lod_job, persistence=persistence, jobs_store=jobs_store, **kw)
    )
    routes_export_job.configure_export_spawner(
        lambda **kw: runner.submit(run_export_job, persistence=persistence, jobs_store=jobs_store, **kw)
    )
    server_app.configure_anchor_job_spawner(
        lambda **kw: runner.submit(run_anchor_job, persistence=persistence, jobs_store=jobs_store, **kw)
    )
    routes_recordings.configure_recording_spawner(
        lambda **kw: run_recording_job(jobs_store, **kw)  # instant honest refusal
    )

    return server_app.asgi_application


def _ensure_local_token(data_dir: str) -> str:
    """Load (or mint + persist, 0600) the static local bearer at
    ``<data_dir>/local_token`` when OPENREALITY_LOCAL_TOKEN is not already set."""
    existing = os.environ.get("OPENREALITY_LOCAL_TOKEN", "").strip()
    if existing:
        return existing
    path = os.path.join(data_dir, "local_token")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            token = fh.read().strip()
        if len(token) >= 16:
            return token
    except FileNotFoundError:
        pass
    token = _secrets.token_urlsafe(32)
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    os.chmod(path, 0o600)
    return token


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Open Reality self-hosted broker (single process, local disk)")
    parser.add_argument("--data-dir", default=os.environ.get("OPENREALITY_DATA_DIR")
                        or os.path.expanduser("~/openreality-data"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    os.makedirs(data_dir, exist_ok=True)

    # Auth env must be settled BEFORE server.app imports.
    os.environ.setdefault("OPENREALITY_AUTH", "local")
    if os.environ["OPENREALITY_AUTH"].strip().lower() == "local":
        os.environ["OPENREALITY_LOCAL_TOKEN"] = _ensure_local_token(data_dir)

    app = build_selfhost_app(data_dir)

    token = os.environ.get("OPENREALITY_LOCAL_TOKEN", "")
    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print("Open Reality self-hosted broker")
    print(f"  data dir : {data_dir}")
    print(f"  url      : {url}")
    if token:
        print(f"  token    : {token}")
        print(f"             (stored at {os.path.join(data_dir, 'local_token')})")
        print("  connect the MCP client with:")
        print(f"    OPENREALITY_URL={url} OPENREALITY_TOKEN={token} npx -y openreality-mcp serve")
        print("  or mint a durable API key once:")
        print(f"    OPENREALITY_URL={url} npx -y openreality-mcp login --token {token}")
    print("=" * 70)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
