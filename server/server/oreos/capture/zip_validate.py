"""Pure zip validation for LiDAR capture-session uploads.

A capture-session zip (produced by the iOS CaptureLogKit harness — see core's
``vggt_slam/capture_session.py``) is store-or-deflate ZIP whose entries are
relative to the session root: ``odometry.csv`` and ``rgb.mp4`` at the top level,
``depth/XXXXXX.png`` + ``confidence/XXXXXX.png`` subdirectories, plus optional
``imu_raw_gyro.csv``, ``imu_raw_accel.csv``, ``imu_fused.csv``, ``events.csv``,
``intrinsics.csv``, ``camera_matrix.csv``, ``meta.json``.

Two things are checked BEFORE a Modal job is ever spawned (the splat lane's "reject
a bad file for FREE" doctrine — see the finalize header-gate in
``server/oreos/routes_ingest.py``):

  1. zip-slip: no entry may be an absolute path or contain a ``..`` traversal
     segment — regardless of platform, since the zip is opened on the broker and
     later extracted inside a Modal container.
  2. required members: ``odometry.csv`` and ``rgb.mp4`` at the session root, plus
     at least one entry each under ``depth/`` and ``confidence/``.

No flask, no modal, no numpy even — stdlib ``zipfile`` only, so this is importable
and unit-testable anywhere.
"""

from __future__ import annotations

import os
import posixpath
import zipfile
from typing import Any, NamedTuple

#: Entries that must exist at the session root (zip top level).
REQUIRED_TOP_LEVEL = ("odometry.csv", "rgb.mp4")

#: Prefixes that must each have at least one entry under them.
REQUIRED_PREFIXES = ("depth/", "confidence/")


class CaptureZipRejected(Exception):
    """A capture-session zip we refuse, with the reason spelled out for the client.

    ``error`` is the stable machine code, ``detail`` a sentence a human can act on,
    ``status`` the HTTP status (400 for every validation failure in this lane, per
    the wire contract), and ``extra`` any numbers worth showing (missing entries,
    unsafe entry names)."""

    def __init__(self, error: str, detail: str, status: int = 400, **extra: Any) -> None:
        super().__init__(f"{error}: {detail}")
        self.error = error
        self.status = status
        self.detail = detail
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "detail": self.detail, **self.extra}


class CaptureZipInfo(NamedTuple):
    """Cheap summary of a validated zip (central-directory read only)."""

    names: tuple
    total_uncompressed: int
    n_entries: int


def is_unsafe_member(name: str) -> bool:
    """True if ``name`` cannot be safely extracted under a fixed root.

    Catches absolute POSIX paths, a leading ``\\`` (Windows-absolute smuggled into
    a POSIX-style zip entry), a drive letter (``C:\\...``), and any ``..``
    path-traversal segment — checked on the POSIX-normalized form so a
    clean-looking suffix (``a/../../etc/passwd``) can't hide the traversal.
    """
    if not name:
        return True
    if name.startswith("/") or name.startswith("\\"):
        return True
    if os.path.isabs(name):
        return True
    if len(name) >= 2 and name[1] == ":":  # C:\... / C:/...
        return True
    norm = posixpath.normpath(name.replace("\\", "/"))
    if norm == ".." or norm.startswith("../"):
        return True
    return False


def validate_capture_zip(path: str) -> CaptureZipInfo:
    """Open ``path`` as a zip and validate it before anything is extracted or any
    Modal job spawned.

    Raises :class:`CaptureZipRejected` for: a file that is not a valid zip, any
    zip-slip entry, or missing required members. Reads only the central directory
    (``ZipFile.infolist()``) — never decompresses an entry body, so this is cheap
    even on a multi-GB capture. Returns a small summary on success.
    """
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise CaptureZipRejected(
            "not_a_zip", f"could not open as a zip archive: {exc}"
        ) from exc

    try:
        infos = zf.infolist()
        unsafe = [i.filename for i in infos if is_unsafe_member(i.filename)]
        if unsafe:
            raise CaptureZipRejected(
                "zip_slip",
                f"{len(unsafe)} entr{'y is' if len(unsafe) == 1 else 'ies are'} unsafe "
                f"(absolute path or '..' traversal), e.g. {unsafe[:5]!r}",
                unsafe_entries=unsafe[:20],
            )

        names = [i.filename for i in infos if not i.filename.endswith("/")]
        top_level = {n for n in names if "/" not in n}
        missing_top = [n for n in REQUIRED_TOP_LEVEL if n not in top_level]
        missing_prefix = [
            p for p in REQUIRED_PREFIXES if not any(n.startswith(p) for n in names)
        ]
        missing = missing_top + missing_prefix
        if missing:
            have = sorted(top_level)
            raise CaptureZipRejected(
                "missing_required_entries",
                "capture-session zip is missing required entries: "
                + ", ".join(missing)
                + f" (top level has: {have[:10]}{'...' if len(have) > 10 else ''})",
                missing=missing,
            )

        total_uncompressed = sum(i.file_size for i in infos)
        return CaptureZipInfo(
            names=tuple(i.filename for i in infos),
            total_uncompressed=int(total_uncompressed),
            n_entries=len(infos),
        )
    finally:
        zf.close()
