"""Gaussian-splat import core — Flask-free, so the SAME code validates on the
broker and does the heavy work inside a Modal job.

Why this module exists (the 2026-07-30 founder incident)
--------------------------------------------------------
``POST /api/workspace/ingest/splat`` was **synchronous on the broker**: stream the
whole body in, parse it, persist, answer 201. A ~2 GB Splatica export never
produced a scene and the founder saw only a spinner then a generic network error.
Root cause: **Modal enforces a hard 150-second HTTP request timeout on every web
function** (the ``timeout=86400`` in the ``@app.function`` decorator is the
*container* lifetime, not the request deadline). 2 GB inside 150 s needs ~115
Mbps sustained upstream with *zero* budget left for parsing — so the request was
killed mid-body, ``_stream_request_to`` never returned, and nothing was logged,
staged or persisted. Secondary landmines behind it: the old read path peaked at
~2x the file size in RAM against a 4 GB broker, and an 8.6M-gaussian file would
then have hit the old 8M cap anyway.

The fix, and what this module owns
----------------------------------
Big splats now arrive **chunked** (each request far inside 150 s) and are parsed
in a **Modal CPU job** with real RAM, exactly like the video path's job+Dict
pattern. This module is the single source of truth both callers share:

* :func:`inspect_splat` — header-only validation (a few KB read). Cheap enough to
  run on the broker at finalize time so a bad file is rejected *before* a job is
  spawned, and re-run inside the job as a guard.
* :func:`read_columns` — memory-lean column extraction via ``np.memmap``. The old
  path called ``splat_io.read_splat_ply``, which materializes **every** property
  twice (a structured array plus a per-property ``.copy()`` while the structured
  array is still alive). Import only needs 6 of ~62 properties, so this reads
  6 x 4 bytes per gaussian instead of the whole file twice: 32M gaussians =
  768 MB instead of ~16 GB.
* :func:`import_splat` — validate, synthesize the center cloud, persist a real
  ``PersistedScene`` with the honest ``degraded`` report.

Every rejection is a :class:`SplatRejected` carrying a machine-readable ``error``
code, an HTTP status and a **specific human sentence**. There is no code path
that fails silently: that was the actual bug the founder hit.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Limits. Every one of these is env-overridable and every one has a message.
# ---------------------------------------------------------------------------

# Hard ceiling on gaussians we will import at all. Raised 8M -> 32M now that the
# parse runs in a 32 GB Modal job instead of on the 4 GB broker: at the lean
# read above, 32M gaussians costs ~0.8 GB of columns + ~0.5 GB of cloud arrays.
# NOT a render budget — see RENDER_ADVISORY_GAUSSIANS.
MAX_GAUSSIANS = int(os.environ.get("DEMO_SPLAT_MAX_GAUSSIANS", str(32_000_000)))

# Total bytes accepted through the chunked route. Bounded by staging + persisted
# copy on the scenes volume (2x the file, transiently) and by the job's RAM
# headroom, not by any request timeout — chunks are individually tiny.
CHUNKED_MAX_BYTES = int(os.environ.get("DEMO_SPLAT_CHUNKED_MAX_BYTES", str(6 * 1024**3)))

# Bytes accepted in ONE synchronous request. This is the 150 s Modal web timeout
# expressed in bytes with a pessimistic-connection safety factor: 64 MiB is ~103 s
# at 5 Mbps, ~26 s at 20 Mbps, leaving room for the parse. Anything larger is
# routed to the chunked path INSTEAD OF being allowed to fail as a timeout.
INLINE_MAX_BYTES = int(os.environ.get("DEMO_SPLAT_INLINE_MAX_BYTES", str(64 * 1024**2)))

# One chunk of a chunked upload. 16 MiB is ~26 s at 5 Mbps and ~67 s at 2 Mbps —
# comfortably inside the 150 s ceiling even on a bad hotel connection.
CHUNK_BYTES = int(os.environ.get("DEMO_SPLAT_CHUNK_BYTES", str(16 * 1024**2)))

# Advisory only, never a rejection: above this the import still succeeds but the
# response carries an honest "this will be heavy to render" note. Grounded in the
# dense-vs-sparse measurements recorded in docs/demo-2026-07/BUILD-LOG.md.
RENDER_ADVISORY_GAUSSIANS = int(
    os.environ.get("DEMO_SPLAT_RENDER_ADVISORY_GAUSSIANS", str(4_000_000))
)

# Gaussian-splat PLY properties that MUST be present for the import to mean
# anything (the 3DGS schema splat_io documents). Without these it is a plain
# point cloud, not a splat.
REQUIRED_FIELDS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0",
)

# The only columns the import itself reads (means + DC colors). Everything else
# in the file is carried through byte-verbatim as the splat artifact.
_IMPORT_COLUMNS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")

# SH degree-0 basis constant: rgb = 0.5 + C0 * f_dc (the standard 3DGS DC decode).
SH_C0 = 0.28209479177387814

# The imported scene's own account of itself.
#
# The previous wording ("no keyframes, no detected objects, and no AI walkthrough report
# — geometry only") stopped being true when synthetic views landed: annotation, objects
# and dimensions all work on an imported splat now. But the honest half of that sentence
# — there is no capture video, so nothing here is grounded in a photograph — is exactly
# what must survive, because it is the fact the viewer cannot see for themselves. So the
# summary states the absence of photography and points at what replaces it, rather than
# declaring features unavailable that are, in fact, available.
# Split in two so the backfill script (scripts/backfill_imported_scene.py) can replace the
# evidence sentence on an ALREADY-imported scene without touching the identification
# sentence in front of it — that half carries the real filename and gaussian count, which
# a maintenance pass has no business re-deriving.
IMPORTED_EVIDENCE_NOTE = (
    "There is no capture video for this scene, so nothing here is grounded in a "
    "photograph: annotation and object detection run on the geometry itself and on views "
    "rendered from the splat, and every claim they produce is labelled as such."
)
IMPORTED_SUMMARY = "Imported Gaussian splat '{name}' ({count} gaussians). " + IMPORTED_EVIDENCE_NOTE


class SplatRejected(Exception):
    """A splat we refuse, with the reason spelled out for the client.

    ``error`` is the stable machine code, ``status`` the HTTP status, ``detail``
    a sentence a human can act on, and ``extra`` any numbers worth showing
    (counts, limits, missing fields)."""

    def __init__(self, error: str, status: int, detail: str, **extra: Any) -> None:
        super().__init__(f"{error}: {detail}")
        self.error = error
        self.status = status
        self.detail = detail
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "detail": self.detail, **self.extra}


class PlyHeader:
    """Parsed binary-LE PLY header: what the body claims to contain, and where
    the body starts."""

    __slots__ = ("count", "names", "data_offset", "itemsize", "file_size")

    def __init__(self, count: int, names: list[str], data_offset: int, file_size: int) -> None:
        self.count = count
        self.names = names
        self.data_offset = data_offset
        self.itemsize = 4 * len(names)  # every 3DGS property is float32
        self.file_size = file_size

    @property
    def body_bytes(self) -> int:
        return self.count * self.itemsize

    @property
    def dtype(self) -> np.dtype:
        return np.dtype([(name, "<f4") for name in self.names])


def read_ply_header(path: str) -> PlyHeader:
    """Vertex count + property names + body offset, reading only the header.

    Deliberately does NOT touch the body: the gaussian cap must be able to reject
    a 6 GB file after reading ~1 KB. Raises ``ValueError`` exactly where
    ``splat_io.read_splat_ply`` would, so the two agree on what a splat is."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file (missing the 'ply' magic line)")
        if f.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError(
                "unsupported PLY format — only 'binary_little_endian 1.0' is read "
                "(ASCII PLY exports must be converted first)"
            )
        count: Optional[int] = None
        names: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            stripped = line.strip()
            if stripped.startswith(b"element vertex"):
                count = int(stripped.split()[-1])
            elif stripped.startswith(b"property"):
                names.append(stripped.split()[-1].decode("ascii"))
            elif stripped == b"end_header":
                break
        data_offset = f.tell()
    if count is None:
        raise ValueError("PLY header missing 'element vertex <N>'")
    return PlyHeader(count, names, data_offset, os.path.getsize(path))


def inspect_splat(path: str, max_gaussians: Optional[int] = None) -> PlyHeader:
    """Header-only gate: is this a gaussian splat we can import, and is it within
    the caps? Raises :class:`SplatRejected` with a specific reason.

    Cheap by construction (~1 KB read) so the broker can run it at finalize time
    and refuse a bad 2 GB upload *without* spending a Modal job on it."""
    limit = MAX_GAUSSIANS if max_gaussians is None else int(max_gaussians)
    try:
        header = read_ply_header(path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SplatRejected("not_a_gaussian_splat", 422, str(exc)) from exc

    missing = [f for f in REQUIRED_FIELDS if f not in header.names]
    if missing:
        raise SplatRejected(
            "not_a_gaussian_splat", 422,
            "this PLY has no 3DGS gaussian properties — it looks like a plain point "
            "cloud, not a gaussian splat. Export with gaussian attributes "
            f"(missing: {', '.join(missing)}).",
            missing=missing,
        )
    if header.count <= 0:
        raise SplatRejected(
            "not_a_gaussian_splat", 422,
            "the PLY header declares zero gaussians — nothing to import.",
            gaussian_count=int(header.count),
        )
    if header.count > limit:
        raise SplatRejected(
            "too_many_gaussians", 422,
            f"{header.count:,} gaussians is over the {limit:,} import limit. "
            "Decimate the splat before importing (most exporters can cap the "
            "gaussian count directly).",
            gaussian_count=int(header.count), limit=limit,
        )

    # Truncation check — this is exactly what a half-finished upload looks like,
    # and saying so is the difference between an honest error and a mystery.
    expected = header.data_offset + header.body_bytes
    if header.file_size < expected:
        raise SplatRejected(
            "truncated_splat", 422,
            f"the file stops early: the header declares {header.count:,} gaussians "
            f"({expected:,} bytes expected) but only {header.file_size:,} bytes "
            "arrived. The upload was cut short — please re-upload.",
            expected_bytes=int(expected), actual_bytes=int(header.file_size),
        )
    return header


def read_columns(path: str, header: PlyHeader, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract just ``names`` as (N,) float32 arrays, via a read-only memmap.

    Peak RAM is ``len(names) * 4 * N`` — nothing else is ever materialized. (The
    generic ``splat_io.read_splat_ply`` is the wrong tool here: it costs ~2x the
    whole file, which is what put the old synchronous path within a rounding
    error of the 4 GB broker ceiling.)"""
    mm = np.memmap(path, dtype=header.dtype, mode="r", offset=header.data_offset,
                   shape=(header.count,))
    try:
        return {name: np.ascontiguousarray(mm[name]) for name in names}
    finally:
        del mm


def build_scene_payload(path: str, header: PlyHeader, filename: str) -> dict[str, Any]:
    """Validate the body and build everything ``save_scene`` needs: the
    synthesized center cloud (means + DC-SH colors) plus the honest degraded
    report and geometry-only facts. Splat bytes are NOT loaded — the caller
    hands ``save_scene`` the path so a multi-GB artifact never enters RAM."""
    from server.scene_report.schemas import SceneFacts, SceneMetrics, SceneReport

    columns = read_columns(path, header, _IMPORT_COLUMNS)
    positions = np.stack([columns[a] for a in ("x", "y", "z")], axis=1)
    finite = np.isfinite(positions).all(axis=1)
    if not bool(finite.any()):
        raise SplatRejected(
            "not_a_gaussian_splat", 422,
            "every gaussian centre in this file is NaN or infinite — the export is "
            "corrupt and there is no geometry to place.",
        )
    positions = positions[finite]
    # rgb = clamp(0.5 + C0 * f_dc); NaN colors clamp to grey via nan_to_num(0.5).
    dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)[finite]
    rgb01 = np.clip(np.nan_to_num(0.5 + SH_C0 * dc, nan=0.5), 0.0, 1.0)
    colors = np.round(rgb01 * 255.0).astype(np.uint8)
    del columns, dc, rgb01

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    extents = (bbox_max - bbox_min).astype(np.float64)
    display_name = os.path.basename(str(filename))

    facts = SceneFacts(
        metrics=SceneMetrics(
            dimensions=sorted((float(v) for v in extents), reverse=True),
            bbox_min=[float(v) for v in bbox_min],
            bbox_max=[float(v) for v in bbox_max],
            point_count=int(positions.shape[0]),
            vertical_axis_known=False,
        ),
        scene_center=[float(v) for v in positions.mean(axis=0)],
        units_note=(
            "Imported splat: coordinates are in the file's own arbitrary units; "
            "not metric until anchored."
        ),
    )
    report = SceneReport(
        summary=IMPORTED_SUMMARY.format(name=display_name, count=header.count),
        room_type="unknown",
        coverage_note="Imported artifact — capture coverage unknown.",
        facts=facts,
        degraded=True,
    )
    return {
        "report": report,
        "facts": facts,
        "points": (positions, colors),
        "display_name": display_name,
        "gaussian_count": int(header.count),
        "point_count": int(positions.shape[0]),
    }


def render_advisory(gaussian_count: int) -> Optional[dict[str, Any]]:
    """Honest note (never a rejection) when an import is big enough that the
    viewer will feel it. Returned alongside the success payload so the founder
    learns the tradeoff at import time rather than mid-demo."""
    if gaussian_count <= RENDER_ADVISORY_GAUSSIANS:
        return None
    return {
        "gaussian_count": int(gaussian_count),
        "comfortable_budget": RENDER_ADVISORY_GAUSSIANS,
        "detail": (
            f"{gaussian_count:,} gaussians imported successfully, but that is above the "
            f"~{RENDER_ADVISORY_GAUSSIANS:,} the viewer renders smoothly. Expect slower "
            "loads and reduced frame rate while orbiting; decimate for a recording."
        ),
    }


def import_splat(
    persistence: Any,
    user_id: str,
    ply_path: str,
    filename: str,
    *,
    label: Optional[str] = None,
    scan_id: Optional[str] = None,
    max_gaussians: Optional[int] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Validate + persist one gaussian ``.ply`` already on local disk.

    ``on_stage`` is called at each coarse boundary (``validate``/``parse``/
    ``persist``) so a job can publish a stage timeline; the synchronous route
    passes nothing. Raises :class:`SplatRejected` for every refusal."""
    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage("validate")
    header = inspect_splat(ply_path, max_gaussians=max_gaussians)

    stage("parse")
    payload = build_scene_payload(ply_path, header, filename)

    stage("persist")
    scan_id = scan_id or uuid.uuid4().hex
    persistence.save_scene(
        user_id,
        scan_id,
        payload["report"],
        payload["facts"],
        keyframes_b64=None,
        points=payload["points"],
        # Path, not bytes: a 6 GB artifact is copied through a small buffer
        # instead of being read whole into the container's RAM.
        splat_path=ply_path,
        label=label or payload["display_name"],
        source="imported_splat",
    )
    print(
        f"[demo.splat_import] imported {payload['display_name']!r} -> scan {scan_id} "
        f"({payload['gaussian_count']} gaussians, {payload['point_count']} cloud points) "
        f"for {user_id}"
    )
    result: dict[str, Any] = {
        "scan_id": scan_id,
        "gaussian_count": payload["gaussian_count"],
        "point_count": payload["point_count"],
    }

    # Derive which way is up while the cloud is still in hand. An imported splat arrives
    # in an arbitrary frame, and without this the scene can only ever quote raw AABB
    # extents. Best-effort inside the helper: a scene we cannot gravity-align still
    # imported fine, and re-runnable later via POST /api/scenes/<id>/ground_frame.
    stage("ground_frame")
    from server.oreos import ground_frame as ground_frame_mod

    frame = ground_frame_mod.compute_and_store(
        persistence, user_id, scan_id, payload["points"][0]
    )
    if frame is not None:
        result["ground_frame"] = {
            "vertical_axis_known": frame["vertical_axis_known"],
            "floor_inlier_ratio": round(frame["floor_inlier_ratio"], 4),
            "note": frame["note"],
        }
    advisory = render_advisory(payload["gaussian_count"])
    if advisory:
        result["render_advisory"] = advisory
    return result
