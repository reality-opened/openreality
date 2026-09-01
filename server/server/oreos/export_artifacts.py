"""Prepared-export artifacts: keys, index shape and the retention rule.

Shared vocabulary between the writer (``modal_oreos_export.py``, its own container) and
the reader (``server/oreos/routes_export_job.py``, on the broker), so the two sides cannot
drift on where a prepared zip lives. Same role ``server/oreos/lod.py`` plays for LOD.

WHY A PREPARED ARTIFACT AT ALL
------------------------------
Measured on the broker's own shape (``scripts/debug_export_timing.py``, scan 243991c3…,
cpu=1.0/memory=4096): load record 0.6 s, build the 1.39 GB tree 9.9 s, deflate to 607 MB
34.6 s — **45.1 s before a single byte reaches the browser**, and only then does a 607 MB
transfer start. Modal caps a web request at 150 s, so that transfer had to sustain
≥5.8 MB/s or the request died and the operator got nothing after minutes of silence.
Building into the scenes volume first collapses both failure modes: the download is a
plain ``send_file`` of a finished file, so it starts at once and the 45 s of work happens
somewhere the operator can watch it.

RETENTION RULE
--------------
**One slot per (scan_id, export_format)**, at a deterministic key:

    derived/demo/export/<format>.zip      the archive
    derived/demo/export/index.json        what is in each slot (bytes, source, built_at)

Every rebuild OVERWRITES its slot in place, so N clicks cost ONE file, not N. That caps a
scene at 2 zips (``openreality`` + ``groot_lerobot_v2``) no matter how the panel is used —
the failure mode this design exists to prevent is a GB-class file per click on a shared
volume. A job also PRUNES on start: anything in ``demo/export/`` that is not a live slot
or the index is unlinked (renamed formats, ``.partial`` corpses from a killed container).

Nothing expires on a timer. A slot older than ``EXPORT_STALE_S`` or built from a different
geometry source is reported ``stale`` and the panel offers a rebuild — but it is still
served, because deleting the operator's only copy on demo morning is worse than serving a
labelled old one. Slots die with their scene: ``ModalScenePersistence._delete_blobs``
rmtree's the whole scene dir, ``derived/`` included.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

#: Prepared-export artifacts live under this path inside a scan's ``derived/`` dir.
EXPORT_ROOT = "demo/export"

#: Formats that have a prepared-zip slot.
#:
#: ``isaac_usd`` was originally absent because its OUTPUT is small (5.5 MB) — but size of
#: output was the wrong axis. Its BUILD is the expensive part: Poisson meshing measured at
#: 378 s, and loading the founder's 63.3M-point scene needed 64 GB, so the synchronous
#: route could only ever answer 413 above ISAAC_MAX_POINTS. It belongs here for the same
#: reason the dataset formats do: the work cannot happen inside a 150 s request.
EXPORT_JOB_FORMATS = ("openreality", "groot_lerobot_v2", "isaac_usd")

#: Formats whose build is heavy enough to need its own oversized container. Isaac authors
#: a Poisson mesh: MEASURED at 378 s on 5M points with 8 CPU / 64 GB, and loading the
#: founder's 63.3M-point scene alone needs far more RAM than the dataset formats ever do.
HEAVY_JOB_FORMATS = ("isaac_usd",)

#: Age past which a prepared slot is reported ``stale`` (still served — see the module
#: docstring). A day: long enough that a demo rehearsal's build is reused the next
#: morning, short enough that "ready" never silently means "from last week".
EXPORT_STALE_S = float(os.environ.get("DEMO_EXPORT_STALE_S", str(24 * 3600)))


def zip_relative_key(export_format: str) -> str:
    """``demo/export/<format>.zip`` — the relative key for ``save_derived_artifact*``."""
    return f"{EXPORT_ROOT}/{export_format}.zip"


def zip_derived_key(export_format: str) -> str:
    """``derived/demo/export/<format>.zip`` — the full key the derived GET route serves."""
    return f"derived/{zip_relative_key(export_format)}"


INDEX_RELATIVE_KEY = f"{EXPORT_ROOT}/index.json"
INDEX_DERIVED_KEY = f"derived/{INDEX_RELATIVE_KEY}"


def index_json_bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")


def slot_record(
    *,
    export_format: str,
    source_key: str,
    zip_bytes: int,
    tree_bytes: int,
    file_count: int,
    build_seconds: float,
    job_id: Optional[str] = None,
    original_points: Optional[int] = None,
    exported_points: Optional[int] = None,
) -> dict[str, Any]:
    """One slot's entry in ``index.json``. Every number here is MEASURED — ``zip_bytes``
    is the archive's size on disk, ``tree_bytes``/``file_count`` come from the tree that
    was actually walked. The panel renders these directly, so an estimate must never be
    smuggled in."""
    return {
        "format": str(export_format),
        "key": zip_derived_key(export_format),
        "source_key": str(source_key),
        "bytes": int(zip_bytes),
        "tree_bytes": int(tree_bytes),
        "file_count": int(file_count),
        "build_seconds": round(float(build_seconds), 1),
        "built_at": time.time(),
        "job_id": str(job_id) if job_id else None,
        # Decimation, when it happened, is part of the slot — not a footnote in a log.
        # An Isaac export of a 63M-point scene is built from a subsample, and a consumer
        # that reads `exported_points` as the scene's density would be wrong by 12x.
        "original_points": int(original_points) if original_points else None,
        "exported_points": int(exported_points) if exported_points else None,
        "decimated": bool(
            original_points and exported_points and exported_points < original_points
        ),
    }


def read_index(persistence: Any, user_id: str, scan_id: str) -> dict[str, Any]:
    """``index.json`` off the volume, or ``{}`` when there is none / it is unreadable.

    Absence is the normal case (nothing prepared yet), so it degrades to empty rather
    than raising — the caller reports ``status: "none"``."""
    try:
        path = persistence.get_derived_artifact_path(user_id, scan_id, INDEX_DERIVED_KEY)
    except Exception:
        return {}
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8"))
    except Exception as exc:
        print(f"[demo-export] index unreadable {path}: {exc}")
        return {}
    return doc if isinstance(doc, dict) else {}


def merge_slot(index: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    """Index with ``slot`` written over its format's entry (the other format is kept —
    preparing GR00T must not forget an already-built OpenReality zip)."""
    slots = dict(index.get("slots") or {})
    slots[str(slot["format"])] = slot
    return {"version": 1, "slots": slots, "updated_at": time.time()}


def live_slot(index: dict[str, Any], export_format: str) -> Optional[dict[str, Any]]:
    slots = index.get("slots")
    if not isinstance(slots, dict):
        return None
    slot = slots.get(str(export_format))
    return slot if isinstance(slot, dict) else None


def staleness(slot: dict[str, Any], wanted_source_key: str, now: Optional[float] = None):
    """``(stale, reason_or_None)`` for a slot against the geometry the caller wants.

    Two honest reasons only: the slot was built from different geometry, or it is older
    than ``EXPORT_STALE_S``. Anything else reports fresh — we never invent a reason to
    make the operator rebuild a GB-class artifact."""
    now = time.time() if now is None else now
    built_at = float(slot.get("built_at") or 0.0)
    if str(slot.get("source_key") or "") != str(wanted_source_key):
        return True, (
            f"prepared from {slot.get('source_key')!r}, this export would use "
            f"{wanted_source_key!r}"
        )
    age = now - built_at
    if built_at and age > EXPORT_STALE_S:
        return True, f"prepared {age / 3600:.1f} h ago"
    return False, None


def prune_orphans(export_dir: str, keep: set[str], log=print) -> list[str]:
    """Delete everything in ``export_dir`` that is not a live slot or the index.

    Sweeps ``.partial`` corpses from a container that died mid-write and zips for
    formats that no longer exist. Returns the basenames removed."""
    if not os.path.isdir(export_dir):
        return []
    removed: list[str] = []
    for name in sorted(os.listdir(export_dir)):
        if name in keep:
            continue
        full = os.path.join(export_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            os.remove(full)
            removed.append(name)
        except OSError as exc:
            log(f"[demo-export] could not prune {full}: {exc}")
    return removed
