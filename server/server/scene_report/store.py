"""Persistence for scene facts/reports.

Two implementations share one duck-typed shape (no ABCs, per repo style):

- ``InMemoryScenePersistence`` — the original non-durable default; facts/reports
  keyed by an opaque ``scene_id``. Kept for tests and local runs.
- ``ModalScenePersistence`` — durable, per-user store backed by an injected
  key/value ``store`` (a ``modal.Dict`` in production) for report/facts JSON +
  manifests, and a filesystem ``blob_root`` (a ``modal.Volume`` mount) for
  keyframe JPEGs and the downsampled point cloud. Keyed by ``user_id`` + a
  per-scan ``scan_id`` so scans survive worker teardown and can be browsed later
  from the always-on CPU broker without a GPU.

``ModalScenePersistence`` is deliberately free of ``import modal``: the Dict
handle, blob root, and ``commit``/``reload`` callables are injected by
``modal_streaming.py``. That keeps it unit-testable with a plain ``dict`` + a
``tmp_path`` and lets it ship inside the existing ``add_local_dir`` for the
package (no new image entry).
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import shutil
import threading
import time
from typing import Any, Optional

import numpy as np

from server.scene_report.schemas import SceneFacts, SceneReport
from server.scene_report.object_layer import parse_object_layer_manifest


class InMemoryScenePersistence:
    def __init__(self):
        self._lock = threading.Lock()
        self._facts: dict[str, SceneFacts] = {}
        self._reports: dict[str, SceneReport] = {}

    def save_facts(self, scene_id: str, facts: SceneFacts) -> None:
        with self._lock:
            self._facts[str(scene_id)] = facts

    def save_report(self, scene_id: str, report: SceneReport) -> None:
        with self._lock:
            self._reports[str(scene_id)] = report

    def get_facts(self, scene_id: str) -> Optional[SceneFacts]:
        with self._lock:
            return self._facts.get(str(scene_id))

    def get_report(self, scene_id: str) -> Optional[SceneReport]:
        with self._lock:
            return self._reports.get(str(scene_id))


# Path components are derived from a Clerk subject (``user_id``) and a uuid hex
# (``scan_id``); sanitize anyway so a hostile value can never escape ``blob_root``.
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]")
_KEYFRAME_BLOB = re.compile(r"^\d+_-?\d+\.jpg$")
# Object-layer GLB ids: manifest ids (e.g. "office_chair", "obj_104"). Strict allow-list so a
# hostile id can never traverse out of the scan's objects/ dir (same posture as _KEYFRAME_BLOB).
_OBJECT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_OBJECT_GLB = re.compile(r"^[A-Za-z0-9_-]{1,128}\.glb$")

# Derived-artifact path segments (product-workflow API) are caller-chosen filenames that
# NEED a literal ``.`` for the extension (``splat.ply``, ``trajectory.npz``) -- unlike
# ``_safe_component`` above, ``.`` is allowed here. The two traversal-special segments
# (``"."``/``".."``) are rejected explicitly below rather than by stripping the character,
# so an extension-bearing filename still round-trips untouched.
_SAFE_DERIVED_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_component(value: Any) -> str:
    return _SAFE_COMPONENT.sub("_", str(value))[:128] or "_"


def _safe_derived_segment(value: Any) -> str:
    cleaned = _SAFE_DERIVED_CHARS.sub("_", str(value))[:128] or "_"
    return "_" if cleaned in (".", "..") else cleaned


def _manifest_to_wire(manifest: Any) -> Optional[dict[str, Any]]:
    """Normalize an object-layer manifest (pydantic model or dict) to a wire-clean dict, or
    ``None`` if it is absent/malformed. ``exclude_none`` is REQUIRED: the TS item guard
    (``web/packages/protocol/objectLayer.ts``) rejects ``asset_url: null`` (only ``undefined``
    or a string), so a box-only object must omit the key entirely or the viewer drops the
    whole layer."""
    if manifest is None:
        return None
    parsed = parse_object_layer_manifest(manifest)
    if parsed is None:
        return None
    return parsed.model_dump(exclude_none=True)


def _dump(obj: Any) -> dict[str, Any]:
    """Coerce a pydantic model (or already-plain dict) to a JSON-able dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _dict_pop(store: Any, key: str) -> None:
    """``modal.Dict.pop`` does not accept a default; tolerate both shapes."""
    try:
        store.pop(key, None)
    except TypeError:
        try:
            store.pop(key)
        except KeyError:
            pass
    except KeyError:
        pass


class ModalScenePersistence:
    """Durable per-user scene store (metadata in a KV ``store``, blobs on disk).

    The injected ``store`` only needs dict-like ``get``/``__setitem__``/``pop``
    (both ``modal.Dict`` and a plain ``dict`` satisfy this). ``commit_fn`` /
    ``reload_fn`` are the Volume's ``commit`` / ``reload`` — the writer (GPU
    worker) commits after each save; the reader (broker) reloads before serving a
    blob. Metadata lives in the KV store, so listing/detail never needs a reload.
    """

    def __init__(
        self,
        store: Any,
        blob_root: str,
        commit_fn: Optional[Any] = None,
        reload_fn: Optional[Any] = None,
        max_scans_per_user: int = 0,
    ):
        self._store = store
        self._blob_root = str(blob_root)
        self._commit = commit_fn
        self._reload = reload_fn
        # 0 (the default) means "unlimited" — full-fidelity artifacts are kept for
        # every scan and never auto-pruned (per the evidence/artifact use-case).
        self._max_scans = max(0, int(max_scans_per_user))
        self._lock = threading.Lock()

    # -- keys / paths -------------------------------------------------

    @staticmethod
    def _index_key(user_id: str) -> str:
        return f"{user_id}:index"

    @staticmethod
    def _scene_key(user_id: str, scan_id: str) -> str:
        return f"{user_id}:{scan_id}"

    def _scene_dir(self, user_id: str, scan_id: str) -> str:
        return os.path.join(
            self._blob_root, _safe_component(user_id), _safe_component(scan_id)
        )

    # -- writes (GPU worker) ------------------------------------------

    def save_scene(
        self,
        user_id: str,
        scan_id: str,
        report: Any,
        facts: Any,
        keyframes_b64: Optional[list[dict[str, Any]]] = None,
        points: Optional[tuple[Any, Any]] = None,
        trajectory: Optional[dict[str, Any]] = None,
        splat_bytes: Optional[bytes] = None,
        splat_path: Optional[str] = None,
        client: Optional[str] = None,
        project: Optional[str] = None,
        label: Optional[str] = None,
        object_layer: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> str:
        """Persist one finished scan. ``keyframes_b64`` is the encode_frames shape
        (``[{submap_id, frame_idx, image_b64}]``); ``points`` is an optional
        ``(positions (N,3) float32, colors (N,3) uint8)`` tuple holding the
        **complete world-frame** cloud (the full-fidelity artifact — no stride/cap).
        ``trajectory`` is an optional ``{poses (N,4,4), intrinsics (N,4),
        source_frame_id (N,)}`` block (cheap, ~N×20 floats) that lets the GPU-free broker
        re-emit ``trajectory.parquet`` later without a ``Solver`` — see
        ``server/export/`` and docs/dataset-export.md §8 Stage 4.
        ``splat_bytes`` is an optional 3DGS ``splat.ply`` artifact (produced by core's
        ``--export_splat``); ``splat_path`` is the additive disk-backed alternative —
        the same artifact supplied as a path and copied through a small buffer, so a
        multi-GB imported splat is never held in RAM (pass one or the other, not both;
        ``splat_bytes`` wins if both are given).
        ``client``/``project`` are optional grouping tags (e.g. a
        pilot's ``client="efficura"``, ``project="chorus"``). All three are additive and
        backward-compatible — older records simply lack them.
        ``source`` (additive, demo build) records how the scene came to exist —
        ``"recon_video" | "imported_splat" | "recon_gemini2" | "recon_depthcam" |
        "robot_recording" | "recon_lidar"``; ``None`` for pre-demo scans. Mirror of
        ``web/packages/protocol/types.ts::PersistedSceneSource`` (the source of truth).
        Best-effort on blobs — a failed JPEG/cloud/trajectory/splat write does not abort
        the metadata save."""
        user_id = str(user_id)
        scan_id = str(scan_id)
        scene_dir = self._scene_dir(user_id, scan_id)
        kf_dir = os.path.join(scene_dir, "keyframes")
        os.makedirs(kf_dir, exist_ok=True)

        # A re-persisted scan_id (e.g. a re-run pilot capture) must not leave the
        # PREVIOUS run's keyframe JPEGs sitting alongside the new ones in kf_dir — only
        # the manifest built below points at the current set, so stale files would be
        # orphaned-but-present blobs (observed: re-running scan "finc" left the old run's
        # JPEGs next to the new ones). Best-effort, like the writes below, and scoped
        # strictly to files matching the keyframe blob naming — never any other blob.
        try:
            stale = [f for f in os.listdir(kf_dir) if _KEYFRAME_BLOB.match(f)]
        except Exception as exc:
            print(f"[scene_store] stale keyframe listdir failed: {exc}")
            stale = []
        for stale_name in stale:
            try:
                os.remove(os.path.join(kf_dir, stale_name))
            except Exception as exc:
                print(f"[scene_store] stale keyframe cleanup failed {stale_name}: {exc}")

        kf_manifest: list[dict[str, Any]] = []
        for kf in keyframes_b64 or []:
            b64 = kf.get("image_b64")
            if not b64:
                continue
            sm = int(kf.get("submap_id", -1))
            fr = int(kf.get("frame_idx", -1))
            blob_key = f"{sm}_{fr}.jpg"
            try:
                with open(os.path.join(kf_dir, blob_key), "wb") as fh:
                    fh.write(base64.b64decode(b64))
            except Exception as exc:
                print(f"[scene_store] keyframe write failed {blob_key}: {exc}")
                continue
            kf_manifest.append({"submap_id": sm, "frame_idx": fr, "blob_key": blob_key})

        # Full-fidelity point-cloud artifact: the COMPLETE world-frame cloud (no
        # stride/voxel/cap) saved once as a compressed ``cloud.npz`` — the durable
        # source of truth (e.g. for evidence/export). The mean is recorded as
        # ``scene_center`` so readers can recenter for display and map world-frame
        # object centers for Q&A focus, but the stored positions stay in the **world
        # frame**, un-recentered, so the artifact is geometrically faithful. The broker
        # derives a downsampled display view from this on read, and the PLY download is
        # generated from it. ``point_count`` records the full N.
        points_key: Optional[str] = None
        point_count = 0
        scene_center = [0.0, 0.0, 0.0]
        if points is not None:
            positions, colors = points
            positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            if positions.shape[0] > 0:
                colors_arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
                n = int(min(positions.shape[0], colors_arr.shape[0]))
                positions = positions[:n]
                colors_arr = colors_arr[:n]
                scene_center = [float(v) for v in positions.mean(axis=0)]
                try:
                    np.savez_compressed(
                        os.path.join(scene_dir, "cloud.npz"),
                        positions=positions,
                        colors=colors_arr,
                    )
                    points_key = "cloud.npz"
                    point_count = n
                except Exception as exc:
                    print(f"[scene_store] point cloud write failed: {exc}")
                    points_key = None

        # Per-keyframe poses + intrinsics for GPU-free dataset export (Stage 4). Small and
        # optional; persisted as a compressed ``trajectory.npz`` alongside the cloud.
        trajectory_key: Optional[str] = None
        trajectory_count = 0
        if trajectory is not None:
            try:
                poses = np.asarray(trajectory.get("poses"), dtype=np.float32).reshape(-1, 4, 4)
                if poses.shape[0] > 0:
                    intrinsics = np.asarray(
                        trajectory.get("intrinsics"), dtype=np.float32
                    ).reshape(-1, 4)
                    source_frame_id = np.asarray(
                        trajectory.get("source_frame_id"), dtype=np.float32
                    ).reshape(-1)
                    np.savez_compressed(
                        os.path.join(scene_dir, "trajectory.npz"),
                        poses=poses,
                        intrinsics=intrinsics,
                        source_frame_id=source_frame_id,
                    )
                    trajectory_key = "trajectory.npz"
                    trajectory_count = int(poses.shape[0])
            except Exception as exc:
                print(f"[scene_store] trajectory write failed: {exc}")
                trajectory_key = None

        # Optional 3DGS splat.ply artifact (core --export_splat). Stored as a plain
        # binary blob alongside cloud.npz; streamed back verbatim by the broker.
        # Two ways to supply it: ``splat_bytes`` (in RAM — the original path, right
        # for the ~100 MB artifacts the recon pipeline produces) or ``splat_path``
        # (additive, demo build) which **copies from disk through a small buffer**.
        # The path form exists for imported splats: a multi-GB upload must never be
        # read whole into a container's RAM just to be written back out again.
        splat_key: Optional[str] = None
        if splat_bytes or splat_path:
            try:
                dest = os.path.join(scene_dir, "splat.ply")
                if splat_bytes:
                    with open(dest, "wb") as fh:
                        fh.write(splat_bytes)
                else:
                    shutil.copyfile(str(splat_path), dest)
                splat_key = "splat.ply"
            except Exception as exc:
                print(f"[scene_store] splat write failed: {exc}")
                splat_key = None

        self._commit_blobs()

        record = {
            "scan_id": scan_id,
            "user_id": user_id,
            "created_at": time.time(),
            "report": _dump(report),
            "facts": _dump(facts),
            "keyframes": kf_manifest,
            "points_key": points_key,
            "point_count": point_count,
            "scene_center": scene_center,
            "trajectory_key": trajectory_key,
            "trajectory_count": trajectory_count,
            "splat_key": splat_key,
            "client": str(client) if client else None,
            "project": str(project) if project else None,
            # Optional human label for this scene (e.g. "Amenity Floor"). Usually set after the
            # fact by the operator; the building manifest prefers it over the room-type guess.
            "label": str(label) if label else None,
            # Optional provenance of the scene itself (demo build, additive): see docstring.
            "source": str(source) if source else None,
            # Optional AI-completed object layer (ObjectLayerManifest wire dict). Small,
            # metadata-sized JSON (like report/facts), so it rides in the KV record — the
            # world-baked GLBs it references are separate on-disk blobs (objects/*.glb). Usually
            # attached post-hoc by attach_object_layer(); additive + backward-compatible (older
            # records simply lack it). See platform/contracts + web/packages/protocol.
            "object_layer": _manifest_to_wire(object_layer),
        }
        with self._lock:
            self._store[self._scene_key(user_id, scan_id)] = record
            self._update_index(user_id, record)
        return scan_id

    def _update_index(self, user_id: str, record: dict[str, Any]) -> None:
        """Read-modify-write the per-user index (newest first). Caller holds the
        lock. Safe under one-writer-per-user; the broker only reads."""
        idx = self._store.get(self._index_key(user_id)) or {}
        scans = [
            s
            for s in (idx.get("scans") or [])
            if isinstance(s, dict) and s.get("scan_id") != record["scan_id"]
        ]
        scans.insert(0, self._summary_from_record(record))
        # _max_scans == 0 → unlimited (default): never prune, so every scan keeps its
        # full-fidelity artifact. A positive cap re-enables blob+metadata eviction.
        if self._max_scans and len(scans) > self._max_scans:
            for stale in scans[self._max_scans:]:
                self._delete_blobs(user_id, str(stale.get("scan_id")))
                _dict_pop(self._store, self._scene_key(user_id, str(stale.get("scan_id"))))
            scans = scans[: self._max_scans]
        self._store[self._index_key(user_id)] = {"scans": scans}

    @staticmethod
    def _summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
        report = record.get("report") or {}
        facts = record.get("facts") or {}
        kfs = record.get("keyframes") or []
        objects = report.get("objects") or facts.get("objects") or []
        return {
            "scan_id": record.get("scan_id"),
            "created_at": record.get("created_at", 0.0),
            "room_type": report.get("room_type", "unknown"),
            "summary": report.get("summary", ""),
            "object_count": len(objects),
            "point_count": record.get("point_count", 0),
            "thumbnail": kfs[0] if kfs else None,
            "has_splat": bool(record.get("splat_key")),
            "client": record.get("client"),
            "project": record.get("project"),
            "label": record.get("label"),
        }

    # -- reads (broker, no GPU) ---------------------------------------

    def list_scenes(self, user_id: str) -> list[dict[str, Any]]:
        idx = self._store.get(self._index_key(str(user_id))) or {}
        return [s for s in (idx.get("scans") or []) if isinstance(s, dict)]

    def get_scene(self, user_id: str, scan_id: str) -> Optional[dict[str, Any]]:
        record = self._store.get(self._scene_key(str(user_id), str(scan_id)))
        return record if isinstance(record, dict) else None

    def get_keyframe(self, user_id: str, scan_id: str, blob_key: str) -> Optional[bytes]:
        if not _KEYFRAME_BLOB.match(str(blob_key)):
            return None
        path = os.path.join(self._scene_dir(str(user_id), str(scan_id)), "keyframes", str(blob_key))
        return self._read_blob(path)

    def get_cloud(self, user_id: str, scan_id: str):
        """Load a scan's point cloud as ``(positions (N,3) float32, colors (N,3)
        uint8)`` in the **world frame**. Handles both the current ``cloud.npz`` artifact
        (stored world-frame, full) and the legacy ``points.bin`` (recentered + capped),
        for which the world frame is reconstructed by adding back ``scene_center``.
        Returns ``None`` if the scan has no stored cloud."""
        record = self.get_scene(user_id, scan_id)
        if not record or not record.get("points_key"):
            return None
        key = str(record["points_key"])
        path = os.path.join(self._scene_dir(str(user_id), str(scan_id)), key)
        self._reload_blobs()
        if not os.path.isfile(path):
            return None
        try:
            if key.endswith(".npz"):
                with np.load(path) as data:
                    positions = np.asarray(data["positions"], dtype=np.float32).reshape(-1, 3)
                    colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
                return positions, colors
            # Legacy flat buffer: N*3 float32 (recentered) then N*3 uint8.
            n = int(record.get("point_count", 0) or 0)
            if n <= 0:
                return None
            with open(path, "rb") as fh:
                blob = fh.read()
            positions = np.frombuffer(blob[: n * 12], dtype="<f4").reshape(-1, 3).astype(np.float32)
            colors = np.frombuffer(blob[n * 12 : n * 12 + n * 3], dtype=np.uint8).reshape(-1, 3)
            center = np.asarray(record.get("scene_center") or [0.0, 0.0, 0.0], dtype=np.float32)
            return positions + center, colors  # recentered → world frame
        except Exception as exc:
            print(f"[scene_store] cloud read failed {path}: {exc}")
            return None

    def get_splat(self, user_id: str, scan_id: str) -> Optional[bytes]:
        """Load a scan's 3DGS ``splat.ply`` artifact bytes, or ``None`` if the scan has
        no stored splat (older scans, or reconstruction-only runs without --export_splat)."""
        record = self.get_scene(user_id, scan_id)
        if not record or not record.get("splat_key"):
            return None
        # The writer only ever sets splat_key == "splat.ply"; enforce that on read so a
        # poisoned/forged metadata value can never traverse out of the scan dir.
        if str(record["splat_key"]) != "splat.ply":
            return None
        path = os.path.join(self._scene_dir(str(user_id), str(scan_id)), "splat.ply")
        return self._read_blob(path)

    def get_splat_path(self, user_id: str, scan_id: str) -> Optional[str]:
        """On-disk path to a scan's ``splat.ply`` (after a volume reload), or ``None`` if the
        scan has no stored splat. Lets a caller STREAM the file from disk instead of reading it
        all into RAM via ``get_splat`` — splat artifacts can be hundreds of MB, so a 1 GB broker
        would OOM serving even one through an in-memory buffer."""
        record = self.get_scene(user_id, scan_id)
        if not record or not record.get("splat_key"):
            return None
        if str(record["splat_key"]) != "splat.ply":
            return None
        path = os.path.join(self._scene_dir(str(user_id), str(scan_id)), "splat.ply")
        # Reload-on-miss: the file is usually already in the mounted Volume (written before this
        # broker started), so check first and skip the reload — a blanket reload() on every read
        # syncs the whole multi-GB volume and added ~60s first-byte latency. Only reload when the
        # file is absent (a scan a worker wrote AFTER this broker's view was last synced).
        if os.path.isfile(path):
            return path
        self._reload_blobs()
        return path if os.path.isfile(path) else None

    # -- derived (non-destructive) artifacts ---------------------------
    #
    # The product-workflow routes (metric anchor / auto-clamp — server/app.py) never
    # mutate a scene's ORIGINAL persisted assets (``cloud.npz`` / ``splat.ply`` / the
    # live pilot embeds ship from those). Corrected copies are written here, under a
    # brand-new key namespaced ``derived/...``, so the original is always retrievable
    # unchanged via ``get_cloud`` / ``get_splat_path`` above.

    _DERIVED_ROOT = "derived"

    def save_derived_artifact(self, user_id: str, scan_id: str, relative_key: str, data: bytes) -> str:
        """Write a NEW derived blob under ``<scene_dir>/derived/<relative_key>`` (e.g. a
        metric-anchor-calibrated cloud, or a scale-clamped splat). ``relative_key`` may
        contain ``/`` (each segment is sanitized independently, so a hostile/forged value
        can never escape the scan's derived dir). Returns the derived key
        (``"derived/<relative_key>"``) to hand back to ``get_derived_artifact_path`` /
        ``get_derived_artifact`` later, and to embed in the route's JSON response."""
        user_id, scan_id = str(user_id), str(scan_id)
        segments = [_safe_derived_segment(part) for part in str(relative_key).split("/") if part]
        if not segments:
            raise ValueError("relative_key must not be empty")
        full_path = os.path.join(self._scene_dir(user_id, scan_id), self._DERIVED_ROOT, *segments)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as fh:
            fh.write(data)
        self._commit_blobs()
        return "/".join([self._DERIVED_ROOT, *segments])

    def derived_dir(self, user_id: str, scan_id: str, relative_key: str = "") -> str:
        """On-disk directory a derived key lives in, whether or not it exists yet.

        ``get_derived_artifact_path`` deliberately answers ``None`` for a key with no
        file behind it, so a writer that needs to build INTO the derived tree (the export
        job stages its zip there, so publishing is a rename rather than a GB-class copy
        across filesystems) has no way to ask. Sanitizes each segment exactly as the
        save/read pair does — a forged key can never escape the scan's derived dir."""
        segments = [_safe_derived_segment(part) for part in str(relative_key).split("/") if part]
        return os.path.join(
            self._scene_dir(str(user_id), str(scan_id)), self._DERIVED_ROOT, *segments
        )

    def save_derived_artifact_file(
        self, user_id: str, scan_id: str, relative_key: str, src_path: str, *, move: bool = False
    ) -> str:
        """Path-based twin of ``save_derived_artifact``: publish an EXISTING file as a
        derived artifact without ever reading it into RAM.

        Needed because the prepared export zip is GB-class (607 MB for the canonical
        scene): the bytes API would materialize the whole archive in the caller's heap
        just to hand it to ``write()``.

        Publishes ATOMICALLY. The payload lands on a sibling ``.<name>.partial`` and is
        ``os.replace``d into position, so a concurrent reader can never open a
        half-written archive and a job that dies mid-write leaves a partial that is not
        servable under the real key (the next run overwrites it). ``move=True`` renames
        the source instead of copying — only correct when ``src_path`` is already on this
        volume's filesystem; otherwise the rename raises and we fall back to a copy."""
        user_id, scan_id = str(user_id), str(scan_id)
        segments = [_safe_derived_segment(part) for part in str(relative_key).split("/") if part]
        if not segments:
            raise ValueError("relative_key must not be empty")
        full_path = os.path.join(self._scene_dir(user_id, scan_id), self._DERIVED_ROOT, *segments)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        staging = os.path.join(
            os.path.dirname(full_path), f".{os.path.basename(full_path)}.partial"
        )
        try:
            if move:
                try:
                    os.replace(src_path, staging)
                except OSError:  # cross-filesystem — copy instead
                    shutil.copyfile(src_path, staging)
            else:
                shutil.copyfile(src_path, staging)
            os.replace(staging, full_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(staging)
            raise
        self._commit_blobs()
        return "/".join([self._DERIVED_ROOT, *segments])

    def delete_derived_artifact(self, user_id: str, scan_id: str, derived_key: str) -> bool:
        """Remove one derived blob. Returns ``True`` if a file was actually unlinked.

        Derived artifacts are normally append-only, but the prepared-export slot is a
        CACHE of a GB-class archive, so its owner needs a way to reclaim it. Never
        touches an original (``cloud.npz`` / ``splat.ply`` / ``trajectory.npz``) — the
        key must resolve inside this scan's ``derived/`` dir or nothing happens."""
        path = self.get_derived_artifact_path(user_id, scan_id, derived_key)
        if not path:
            return False
        try:
            os.remove(path)
        except OSError as exc:
            print(f"[scene_store] derived artifact delete failed {path}: {exc}")
            return False
        self._commit_blobs()
        return True

    def get_derived_artifact_path(self, user_id: str, scan_id: str, derived_key: str) -> Optional[str]:
        """On-disk path for a key returned by ``save_derived_artifact``, or ``None`` if
        absent / the key isn't a ``derived/...`` key."""
        user_id, scan_id = str(user_id), str(scan_id)
        key = str(derived_key)
        prefix = self._DERIVED_ROOT + "/"
        if not key.startswith(prefix):
            return None
        segments = [_safe_derived_segment(part) for part in key[len(prefix):].split("/") if part]
        if not segments:
            return None
        path = os.path.join(self._scene_dir(user_id, scan_id), self._DERIVED_ROOT, *segments)
        if os.path.isfile(path):
            return path
        self._reload_blobs()
        return path if os.path.isfile(path) else None

    def get_derived_artifact(self, user_id: str, scan_id: str, derived_key: str) -> Optional[bytes]:
        path = self.get_derived_artifact_path(user_id, scan_id, derived_key)
        if path is None:
            return None
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except Exception as exc:
            print(f"[scene_store] derived artifact read failed {path}: {exc}")
            return None

    def set_derived_pointer(
        self, user_id: str, scan_id: str, pointer: Optional[dict[str, Any]]
    ) -> bool:
        """Record the scan's "latest calibrated" derived artifact group on its METADATA record
        (a new ``derived_latest`` field) so a later default export / the web can find it without
        an explicit selector. METADATA-ONLY and NON-DESTRUCTIVE: never touches a blob, never
        rewrites ``points_key``/``splat_key``/``trajectory_key`` or any original artifact — the
        live pilot embeds keep shipping from the originals. ``pointer=None`` clears it. Returns
        ``True`` if the record was updated, ``False`` if the scan is unknown (best-effort — a KV
        hiccup is swallowed, like the blob writes in ``save_scene``)."""
        try:
            with self._lock:
                record = self._store.get(self._scene_key(str(user_id), str(scan_id)))
                if not isinstance(record, dict):
                    return False
                record = dict(record)  # copy-on-write; don't mutate the object other readers hold
                if pointer is None:
                    record.pop("derived_latest", None)
                else:
                    record["derived_latest"] = dict(pointer)
                self._store[self._scene_key(str(user_id), str(scan_id))] = record
            return True
        except Exception as exc:
            print(f"[scene_store] set_derived_pointer failed: {exc}")
            return False

    def set_synthetic_views(
        self, user_id: str, scan_id: str, views: Optional[list[dict[str, Any]]]
    ) -> bool:
        """Record a scan's SYNTHETIC-VIEW manifest (metadata only — the PNGs are derived
        blobs written via ``save_derived_artifact``).

        Synthetic views are renders of the scene's own splat, registered so the
        keyframe-shaped pipelines have something to look at on an imported scan that has
        no capture video. They get their own record field and are NEVER merged into
        ``keyframes``: a render and a photograph must stay distinguishable in the
        persisted record forever, not just in the UI that happens to be reading it today.

        ``views=None`` clears the field. Same copy-on-write posture as
        ``set_derived_pointer`` — ``modal.Dict.get`` hands back a copy, so the record has
        to be re-assigned for the mutation to persist. Returns ``False`` for an unknown
        scan."""
        try:
            with self._lock:
                key = self._scene_key(str(user_id), str(scan_id))
                record = self._store.get(key)
                if not isinstance(record, dict):
                    return False
                record = dict(record)
                if views is None:
                    record.pop("synthetic_views", None)
                else:
                    record["synthetic_views"] = [dict(v) for v in views]
                self._store[key] = record
            return True
        except Exception as exc:
            print(f"[scene_store] set_synthetic_views failed: {exc}")
            return False

    def get_synthetic_views(self, user_id: str, scan_id: str) -> list[dict[str, Any]]:
        record = self.get_scene(user_id, scan_id)
        views = (record or {}).get("synthetic_views")
        return [v for v in views if isinstance(v, dict)] if isinstance(views, list) else []

    def update_scene_metrics(
        self, user_id: str, scan_id: str, patch: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge ``patch`` into a scan's ``facts.metrics``, non-destructively.

        Used by the ground-frame recompute so an already-imported scan can gain
        ``up_axis``/``floor_height``/``vertical_axis_known`` WITHOUT a re-upload — the
        founder's 2 GB import must never have to be sent again to learn which way is up.
        Keys are merged, not replaced, so an unrelated metric (``point_count``, the AABB)
        can never be dropped by a partial patch; ``report.facts.metrics`` is updated in
        lockstep because the report embeds its own copy of the facts.

        Raises ``KeyError('not_found')`` for an unknown scan. Returns the merged metrics."""
        user_id, scan_id = str(user_id), str(scan_id)
        with self._lock:
            key = self._scene_key(user_id, scan_id)
            record = self._store.get(key)
            if not isinstance(record, dict):
                raise KeyError("not_found")
            record = dict(record)
            merged: dict[str, Any] = {}
            for holder in ("facts", "report"):
                block = record.get(holder)
                if not isinstance(block, dict):
                    continue
                facts = block if holder == "facts" else block.get("facts")
                if not isinstance(facts, dict):
                    continue
                metrics = dict(facts.get("metrics") or {})
                metrics.update(patch)
                facts["metrics"] = metrics
                if holder == "report":
                    block["facts"] = facts
                record[holder] = block if holder == "report" else facts
                merged = metrics
            self._store[key] = record
            return merged

    def get_trajectory(self, user_id: str, scan_id: str):
        """Load a scan's per-keyframe ``{poses (N,4,4), intrinsics (N,4),
        source_frame_id (N,)}`` block (Stage-4 dataset export), or ``None`` if absent."""
        record = self.get_scene(user_id, scan_id)
        if not record or not record.get("trajectory_key"):
            return None
        path = os.path.join(
            self._scene_dir(str(user_id), str(scan_id)), str(record["trajectory_key"])
        )
        self._reload_blobs()
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path) as data:
                return {
                    "poses": np.asarray(data["poses"], dtype=np.float32).reshape(-1, 4, 4),
                    "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32).reshape(-1, 4),
                    "source_frame_id": np.asarray(
                        data["source_frame_id"], dtype=np.float32
                    ).reshape(-1),
                }
        except Exception as exc:
            print(f"[scene_store] trajectory read failed {path}: {exc}")
            return None

    def delete_scene(self, user_id: str, scan_id: str) -> None:
        user_id = str(user_id)
        scan_id = str(scan_id)
        with self._lock:
            self._delete_blobs(user_id, scan_id)
            _dict_pop(self._store, self._scene_key(user_id, scan_id))
            idx = self._store.get(self._index_key(user_id)) or {}
            scans = [
                s
                for s in (idx.get("scans") or [])
                if isinstance(s, dict) and s.get("scan_id") != scan_id
            ]
            self._store[self._index_key(user_id)] = {"scans": scans}

    def update_object_edit(
        self,
        user_id: str,
        scan_id: str,
        object_id: int,
        *,
        label: Optional[str] = None,
        dismissed: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Persist a human review edit onto ONE object of a scan's inventory,
        NON-DESTRUCTIVELY (product-workflow Detect stage).

        ``object_id`` is the 0-based index into the persisted ``facts.objects`` list. That
        list is written exactly once at scan finalize and never reordered or filtered (this
        method mutates only per-object *edit metadata*, never the list's membership/order),
        so the index is a stable per-scan handle for an object.

        A relabel sets ``human_label`` ALONGSIDE the model's original ``query`` (which is
        never overwritten); ``dismissed`` flags the instance as hidden from the reviewed
        inventory without deleting the detection. Only the arguments explicitly passed
        (non-``None``) are changed, so a dismiss can't clobber a prior relabel and vice versa.

        Raises ``KeyError('not_found')`` when the scan doesn't exist for this user, and
        ``IndexError`` when ``object_id`` is out of range. Returns the updated object dict.

        The scene metadata record lives in the KV ``store`` (like ``save_scene``'s record
        write), so re-assigning it here persists the edit the same way — no blob commit
        needed. The per-user index summary is intentionally left untouched: dismiss is a
        review annotation, not a deletion, so the listing's ``object_count`` stays the raw
        detected count.
        """
        user_id = str(user_id)
        scan_id = str(scan_id)
        with self._lock:
            record = self._store.get(self._scene_key(user_id, scan_id))
            if not isinstance(record, dict):
                raise KeyError("not_found")
            facts = record.get("facts") or {}
            objects = facts.get("objects") or []
            if not isinstance(object_id, int) or isinstance(object_id, bool):
                raise IndexError(object_id)
            if object_id < 0 or object_id >= len(objects):
                raise IndexError(object_id)
            obj = objects[object_id]
            if label is not None:
                obj["human_label"] = str(label)
            if dismissed is not None:
                obj["dismissed"] = bool(dismissed)
            # Re-assign the record so KV stores that hand back a copy on read (e.g.
            # ``modal.Dict``) actually persist the mutation; a plain dict is a harmless no-op.
            record["facts"] = facts
            self._store[self._scene_key(user_id, scan_id)] = record
            return obj

    def replace_geometry_objects(
        self, user_id: str, scan_id: str, objects: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Replace a scan's GEOMETRY-DERIVED inventory (imported-splat objects).

        Companion to ``update_object_edit``: that one annotates an existing detection,
        this one writes the whole ``facts.objects`` list for a scan that never had any —
        an imported splat has no capture, so nothing ever detected anything for it (see
        ``server/oreos/imported_objects.py``).

        REFUSES to touch a scan holding real detections. Every object written here carries
        an ``imported_detection`` block; anything without one came from a capture, and
        overwriting those would destroy the evidence-backed inventory, the ``det:<idx>``
        uids pointing at it and every ``human_label`` a reviewer recorded. So the guard is
        the presence of that block on every existing entry, not the scan's ``source`` tag —
        a tag can be wrong, the block cannot be there by accident. Raises ``ValueError``
        when the guard trips.

        Re-running is idempotent by replacement: the previous geometry run's boxes go, the
        new ones take their positions. Positional identity (``det:<idx>``) is therefore
        stable only WITHIN a run — which is why the client refreshes the scene after one.

        Raises ``KeyError('not_found')`` for an unknown scan. Returns
        ``{"replaced": <count removed>, "count": <count written>}``.
        """
        user_id = str(user_id)
        scan_id = str(scan_id)
        with self._lock:
            record = self._store.get(self._scene_key(user_id, scan_id))
            if not isinstance(record, dict):
                raise KeyError("not_found")
            facts = record.get("facts") or {}
            existing = facts.get("objects") or []
            captured = [o for o in existing if not (isinstance(o, dict) and o.get("imported_detection"))]
            if captured:
                raise ValueError(
                    f"{len(captured)} of {len(existing)} objects on this scan came from a "
                    "capture — refusing to overwrite detected objects with geometric clusters"
                )
            facts["objects"] = [dict(o) for o in objects]
            record["facts"] = facts
            self._store[self._scene_key(user_id, scan_id)] = record
            # The picker's object_count is read off this summary, so a scan that just
            # gained an inventory must stop advertising zero objects.
            self._update_index(user_id, record)
            return {"replaced": len(existing), "count": len(facts["objects"])}

    # -- internals ----------------------------------------------------

    def _read_blob(self, path: str) -> Optional[bytes]:
        # Reload-on-miss (see get_splat_path): avoid a full volume reload when the blob is
        # already present in the mounted view; only reload to pick up a just-written file.
        if not os.path.isfile(path):
            self._reload_blobs()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except Exception as exc:
            print(f"[scene_store] blob read failed {path}: {exc}")
            return None

    def _delete_blobs(self, user_id: str, scan_id: str) -> None:
        shutil.rmtree(self._scene_dir(user_id, scan_id), ignore_errors=True)
        self._commit_blobs()

    def _commit_blobs(self) -> None:
        if self._commit is None:
            return
        try:
            self._commit()
        except Exception as exc:
            print(f"[scene_store] volume commit failed: {exc}")

    def _reload_blobs(self) -> None:
        if self._reload is None:
            return
        try:
            self._reload()
        except Exception as exc:
            print(f"[scene_store] volume reload failed: {exc}")

    # -- object layer (AI-completed 3D assets overlaid on the scan) ----
    # NOTE: kept together at the class tail (rather than beside get_trajectory as in the origin
    # feat/object-layer-embed branch) so this block does not overlap the derived-pointer /
    # object-edit method regions that the sibling product-workflow branches add — all three
    # merge cleanly. Purely additive + non-destructive: reads/attaches an object layer, never
    # rewrites points_key/splat_key/trajectory_key or any original scan artifact.

    def attach_object_layer(
        self,
        user_id: str,
        scan_id: str,
        manifest: Any,
        glb_sources: Optional[dict[str, str]] = None,
    ) -> str:
        """Attach (or replace) an object layer on an EXISTING scan: copy the world-baked GLBs to
        ``objects/<id>.glb`` and set the record's ``object_layer`` manifest (wire-clean dict).
        Used by the offline embed-regen tool (``scripts/attach_object_layer.py``). Raises
        ``KeyError`` if the scan doesn't exist (the regen attaches to a freshly reconstructed
        scan), ``ValueError`` if the manifest is malformed. ``glb_sources`` maps object id ->
        source ``.glb`` path; ids without a source stay box-only. Re-attaching replaces the prior
        asset set (stale GLBs are pruned), mirroring the keyframe overwrite behaviour."""
        user_id = str(user_id)
        scan_id = str(scan_id)
        wire = _manifest_to_wire(manifest)
        if wire is None:
            raise ValueError("object layer manifest is missing or malformed")
        sources = dict(glb_sources or {})
        with self._lock:
            record = self._store.get(self._scene_key(user_id, scan_id))
            if not isinstance(record, dict):
                raise KeyError(
                    f"no scan {scan_id!r} for user {user_id!r} to attach an object layer to"
                )
            obj_dir = os.path.join(self._scene_dir(user_id, scan_id), "objects")
            os.makedirs(obj_dir, exist_ok=True)
            wanted: set[str] = set()
            for oid, src in sources.items():
                if not _OBJECT_ID.match(str(oid)):
                    print(f"[scene_store] object layer: skipping unsafe id {oid!r}")
                    continue
                dst_name = f"{oid}.glb"
                try:
                    shutil.copyfile(src, os.path.join(obj_dir, dst_name))
                    wanted.add(dst_name)
                except Exception as exc:
                    print(f"[scene_store] object glb write failed {oid}: {exc}")
            # Prune GLBs from a previous attach that the new manifest no longer references —
            # scoped strictly to *.glb blobs, never any other file (cf. keyframe cleanup).
            try:
                existing = [f for f in os.listdir(obj_dir) if _OBJECT_GLB.match(f)]
            except Exception as exc:
                print(f"[scene_store] object glb listdir failed: {exc}")
                existing = []
            for name in existing:
                if name not in wanted:
                    try:
                        os.remove(os.path.join(obj_dir, name))
                    except Exception as exc:
                        print(f"[scene_store] object glb cleanup failed {name}: {exc}")
            self._commit_blobs()
            record["object_layer"] = wire
            self._store[self._scene_key(user_id, scan_id)] = record
        return scan_id

    def get_object_layer(self, user_id: str, scan_id: str) -> Optional[dict[str, Any]]:
        """The scan's ``ObjectLayerManifest`` wire dict, or ``None`` if it has no layer. Reads
        the KV record only (no disk probe), so ``get_scene`` on a layer-less scan is unaffected
        and never triggers a Volume reload."""
        record = self.get_scene(user_id, scan_id)
        if not isinstance(record, dict):
            return None
        layer = record.get("object_layer")
        return layer if isinstance(layer, dict) else None

    def get_object_asset_path(
        self, user_id: str, scan_id: str, object_id: str
    ) -> Optional[str]:
        """On-disk path to one object's world-baked ``.glb`` (after a volume reload-on-miss), or
        ``None``. Gated on the id (a) being a safe component and (b) actually appearing in this
        scan's manifest — so a share token can only fetch assets its own scan declares, and a
        layer-less scan short-circuits before any disk probe (no reload footgun)."""
        if not _OBJECT_ID.match(str(object_id)):
            return None
        record = self.get_scene(user_id, scan_id)
        if not isinstance(record, dict):
            return None
        layer = record.get("object_layer") or {}
        known = {
            str(o.get("id"))
            for o in (layer.get("objects") or [])
            if isinstance(o, dict)
        }
        if str(object_id) not in known:
            return None
        path = os.path.join(
            self._scene_dir(str(user_id), str(scan_id)), "objects", f"{object_id}.glb"
        )
        if os.path.isfile(path):
            return path
        self._reload_blobs()
        return path if os.path.isfile(path) else None

    def get_object_asset(
        self, user_id: str, scan_id: str, object_id: str
    ) -> Optional[bytes]:
        """One object's GLB bytes, or ``None``. In-memory fallback for stores without on-disk
        streaming (tests); the broker prefers ``get_object_asset_path`` to stream from disk."""
        path = self.get_object_asset_path(user_id, scan_id, object_id)
        return self._read_blob(path) if path else None
