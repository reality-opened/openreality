"""Safe extraction of a validated capture-session zip into a session directory.

Call :func:`server.oreos.capture.zip_validate.validate_capture_zip` FIRST — this
module does not repeat the required-entries check, but it NEVER trusts that a prior
validation pass ran: every entry is re-checked for zip-slip at the point of
extraction (belt + braces — this is exactly the code path that turns a hostile zip
into an arbitrary file write if it is ever skipped).

No flask, no modal — stdlib ``zipfile`` only, so this is importable and
unit-testable anywhere (the Modal job calls it inside the container, on the
already-staged zip file).
"""

from __future__ import annotations

import os
import shutil
import zipfile

from server.oreos.capture.zip_validate import CaptureZipRejected, is_unsafe_member


def extract_capture_session(zip_path: str, dest_dir: str) -> str:
    """Extract every entry of ``zip_path`` into ``dest_dir`` (created if missing),
    preserving the session-root-relative layout the
    ``vggt_slam.capture_session.CaptureSession`` loader expects
    (``odometry.csv``, ``rgb.mp4``, ``depth/*.png``, ``confidence/*.png``, ...).

    Every entry — file OR directory — is validated and extracted ONE AT A TIME by
    this function itself, never via ``ZipFile.extractall`` (which, on this Python
    version, applies no path-traversal protection of its own): a directory entry
    is just as capable of naming ``../../etc`` as a file entry, so it gets the
    identical zip-slip check, not a free pass because ``extractall`` "would have
    recreated it anyway".

    Returns ``dest_dir``. Raises :class:`CaptureZipRejected` if any entry is
    unsafe, or ``FileNotFoundError``/``zipfile.BadZipFile`` for an unreadable
    archive (should already have been caught by ``validate_capture_zip`` at
    finalize time, but this function must not assume that happened)."""
    os.makedirs(dest_dir, exist_ok=True)
    root = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if is_unsafe_member(info.filename):
                raise CaptureZipRejected(
                    "zip_slip",
                    f"unsafe entry at extraction time: {info.filename!r}",
                )
            target = os.path.realpath(os.path.join(dest_dir, info.filename))
            if target != root and not target.startswith(root + os.sep):
                raise CaptureZipRejected(
                    "zip_slip",
                    f"entry resolves outside the session root: {info.filename!r}",
                )
            if info.filename.endswith("/") or info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Streamed copy (not src.read() then write): rgb.mp4 can be the bulk
            # of a multi-GB capture, and this must never hold a whole entry in RAM.
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    return dest_dir
