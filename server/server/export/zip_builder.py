"""Build a scan's export tree and deflate it to a zip — ONE implementation.

Lifted verbatim out of ``server/app.py`` (it was ``_build_export_zip_file``) so the
background export job can call it WITHOUT importing ``server.app``: that module pulls in
torch, cv2 and the whole Flask/SocketIO surface at import time, none of which a
zip-building container needs. ``server.app`` now delegates here, so the synchronous
route and the job compress through the same code path and can never drift.

Compression stays DEFLATE. Measured on the broker's own shape (cpu=1.0/memory=4096,
``scripts/debug_export_timing.py``, scan 243991c3…): a 1.39 GB tree deflates to 607 MB in
34.6 s. ``ZIP_STORED`` cuts the build to 4.1 s but ships the full 1.39 GB — the payload
compresses to 43.6%, so deflate is earning its keep on every byte the operator waits for.

``on_stage`` exists for the job's progress bar. It reports STAGE BOUNDARIES with the
numbers that are actually known at that point (tree bytes once the tree exists, zip bytes
once the archive exists) — deliberately not a synthetic percentage, because nothing here
can honestly produce one.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from typing import Any, Callable, Optional

from server.export.to_lerobot import convert_to_groot_lerobot
from server.export.writer import export_from_record

#: ``on_stage(stage_name, **facts)`` — see the module docstring.
StageCallback = Callable[..., None]

EXPORT_ZIP_FORMATS = ("openreality", "groot_lerobot_v2")


def _tree_size(root: str) -> tuple[int, int]:
    """(total bytes, file count) of an on-disk export tree."""
    total = 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dirpath, fn))
                count += 1
    return total, count


def build_export_zip_file(
    record: dict[str, Any],
    export_format: str,
    *,
    slot_dir: Optional[str] = None,
    on_stage: Optional[StageCallback] = None,
) -> str:
    """Build the export tree from a normalized Stage-4 record
    (``server.export.record.load_record_from_store``), zip it, and return the path of the
    zip — parked OUTSIDE the build's TemporaryDirectory (a
    ``NamedTemporaryFile(delete=False)`` slot; same-tmpfs ``shutil.move`` is a rename).
    THE CALLER OWNS DELETION of the returned path.

    File-path (not bytes) by design: an anchored-source canonical export is ~1.4 GB and
    the broker has 4 GB RAM total — callers stream the archive from disk and never hold
    it in memory (demo BUILD-PLAN risk R8). GPU-free — no live ``Solver``.
    ``export_format`` is ``"openreality"`` (the native ``export/<scan_id>/{meta,data,
    videos,clouds,grounding}/`` tree) or ``"groot_lerobot_v2"`` (that tree transcoded to
    LeRobot v2 + ``modality.json``).

    ``slot_dir`` places the finished zip in a specific directory instead of the system
    temp dir — the background job points it at the scene's volume-backed staging dir so
    publishing the artifact is a rename rather than a 607 MB copy across filesystems.
    """
    if on_stage:
        on_stage("build_tree")
    with tempfile.TemporaryDirectory(prefix="openreality_export_") as tmp:
        base = export_from_record(record, os.path.join(tmp, "openreality"))
        if export_format == "openreality":
            archive_base = base
        else:  # "groot_lerobot_v2"
            groot_dir = os.path.join(tmp, f"{os.path.basename(base)}_groot_lerobot_v2")
            convert_to_groot_lerobot(base, groot_dir)
            archive_base = groot_dir

        if on_stage:
            tree_bytes, file_count = _tree_size(archive_base)
            on_stage("compress", tree_bytes=tree_bytes, file_count=file_count)

        zip_path = shutil.make_archive(
            os.path.join(tmp, "archive"),
            "zip",
            root_dir=os.path.dirname(archive_base),
            base_dir=os.path.basename(archive_base),
        )
        if slot_dir:
            os.makedirs(slot_dir, exist_ok=True)
        slot = tempfile.NamedTemporaryFile(
            prefix="openreality_export_", suffix=".zip", delete=False, dir=slot_dir
        )
        slot.close()
        try:
            if on_stage:
                on_stage("stage", zip_bytes=os.path.getsize(zip_path))
            shutil.move(zip_path, slot.name)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(slot.name)
            raise
        return slot.name
