"""The frame-filename contract shared by every ingest adapter.

A frame is written as ``<ts:.6f>.jpg`` and that name *is* the timestamp — three
independent consumers parse it back out:

  * ``vggt_slam.slam_utils.sort_images_by_number`` (core) — regex
    ``\\d+(?:\\.\\d+)?(?=\\.[^.]+$)``, i.e. a decimal immediately before the extension;
  * ``server/oreos/recordings/export_targets.py::_load_frames`` — ``float(path.stem)``;
  * ``server/oreos/recordings/live_probe/driver.py::parse_frame_ts`` — core's regex again.

So the stem must stay a plain, parseable, monotonically sortable decimal. That
rules out the obvious collision fix of an underscore counter: ``1.234567_1.jpg``
makes ``float(stem)`` raise and makes the regex silently read the trailing ``1``.

Collisions are real. Recording timestamps are epoch-scale float64 while the name
keeps only microseconds, so two observations less than 1 µs apart render to the
same string and the second silently overwrote the first — while the writer still
counted both. The manifest's ``n_frames`` then exceeded the files on disk, and the
damage surfaced far downstream as export_targets' frame/trajectory desync check
("N trajectory timestamps have no frame in frames/"), which points at the wrong
stage entirely.

The disambiguation is therefore a nudge of the timestamp itself to the next free
microsecond tick: deterministic, still a valid float, still correctly ordered,
and wrong by at most a few microseconds — five orders of magnitude below the
50 ms association tolerance in ``server/oreos/recordings/consistency.py``.
"""

from __future__ import annotations

FRAME_SUFFIX = ".jpg"

_US = 1_000_000


def frame_name(ts: float) -> str:
    """Canonical frame filename for a timestamp (non-negative, epoch-scale)."""
    return f"{ts:.6f}{FRAME_SUFFIX}"


def unique_frame_name(ts: float, taken: set[str]) -> str:
    """`frame_name(ts)`, advanced in 1 µs ticks until it is not already in `taken`.

    Adds the returned name to `taken`, so a writer loop needs no other bookkeeping.
    `taken` must hold only names written by the CURRENT run: seeding it from disk
    would make re-ingesting a recording duplicate every frame instead of
    overwriting it.

    Ticking is done on integer microseconds parsed out of the formatted name, not
    by adding 1e-6 to the float — at epoch scale one float64 ULP is ~5e-7 s, so
    float addition is not guaranteed to change the rendered name (an infinite
    loop), and re-deriving the name from a re-multiplied float could round
    differently from the value written for non-colliding frames.
    """
    name = frame_name(ts)
    if name not in taken:
        taken.add(name)
        return name
    whole, _, frac = name[: -len(FRAME_SUFFIX)].partition(".")
    micros = int(whole) * _US + int(frac)
    while name in taken:
        micros += 1
        name = f"{micros // _US}.{micros % _US:06d}{FRAME_SUFFIX}"
    taken.add(name)
    return name
