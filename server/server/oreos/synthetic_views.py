"""Synthetic views — renders of an imported splat, registered as first-class evidence.

Why this module exists
----------------------
An imported splat has no capture video, so ``splat_import.py`` persists it with
``keyframes_b64=None``. Every keyframe-shaped pipeline we own (VLM annotation, SAM
prompts, mask back-projection) then has nothing to look at, and the scene degrades to
a point cloud with a caption. We cannot invent photographs — but we *can* render the
splat, which is exactly what the viewer already does 60 times a second. A render is
honest evidence about that splat's geometry and appearance.

The honesty rule is structural, not editorial (IMPORTED-SPLAT-CONTRACT.md):

* synthetic views live in their own ``synthetic_views`` record field and their own blob
  namespace. They are NEVER written into ``keyframes`` — provenance must stay
  distinguishable forever, including for anyone reading the record years from now;
* every claim grounded on one carries the ``synthetic view`` provenance chip, never
  ``photo`` / ``keyframe``. :data:`SYNTHETIC_VIEW_PROVENANCE` is that fixed copy.

Frames + conventions (WORLD-TRANSFORM-CONTRACT.md)
--------------------------------------------------
The wire pose is ``{position, quaternion}`` in the **world / import frame**, with the
quaternion in the **three.js camera convention** (+x right, +y up, −z forward) because
the producer is a three.js renderer and the client must not be asked to do a handedness
flip on top of the display-frame inversion it already owes us. The single conversion to
the OpenCV convention the rest of the server speaks (``trajectory.npz`` c2w, x right,
y down, z forward) happens HERE, once, in :func:`gl_pose_to_cv_c2w` — so a stored view
is directly consumable by ``segment_geometry.lift_mask_to_points`` and by
``routes_sam3d``'s existing rendered-view path without any per-caller adapter.

:func:`project_world_point` is the executable statement of that convention: it is the
same pinhole projection three.js applies, and the client's round-trip test and this
module's tests assert the same world point lands on the same pixel. If that pair ever
disagrees, every back-projection downstream is silently landing in the wrong place —
which is why the check exists on both sides of the wire rather than in one of them.

Flask-free by design (the ``planes.py`` posture): routes live in ``routes_imported.py``.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from typing import Any, Optional

import numpy as np

# Blob namespace. A derived key (not a bare scene-dir sibling) so the artifacts inherit
# the derived route's owner-auth + Range serving for free, and so the path itself reads
# as provenance: nothing under demo/synthetic_views/ can be mistaken for a capture.
SYNTHETIC_VIEW_DIR = "demo/synthetic_views"

# Fixed provenance copy. Mirrored by the client chip (provenanceChip.ts) — change both
# together or not at all.
SYNTHETIC_VIEW_PROVENANCE = "synthetic view"
SYNTHETIC_VIEW_DETAIL = "rendered from the imported splat"

# Caps. A view set is evidence for one VLM pass, not a photo library: the annotator
# batches ~6 images per call, so a couple of dozen is already generous, and the record
# these ride on is a KV value that has to stay small.
MAX_VIEWS = int(os.environ.get("DEMO_SYNTHETIC_MAX_VIEWS", "24"))
MAX_VIEW_BYTES = int(os.environ.get("DEMO_SYNTHETIC_MAX_VIEW_BYTES", str(6 * 1024**2)))
MAX_DIMENSION = 4096
MIN_DIMENSION = 64

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_EPS = 1e-9


class SyntheticViewError(ValueError):
    """A rejected view payload, with the reason spelled out for the client.

    Same posture as ``splat_import.SplatRejected``: a machine code, an HTTP status and a
    sentence a human can act on. There is no path that drops a view silently — a view
    that vanished without explanation is indistinguishable from a transform bug."""

    def __init__(self, code: str, status: int, detail: str, **extra: Any) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.status = status
        self.detail = detail
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        return {"error": self.code, "detail": self.detail, **self.extra}


# ---------------------------------------------------------------------------
# camera math (the convention, executable)
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quat_xyzw: Any) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` -> its ``(3, 3)`` rotation matrix.

    ``xyzw`` order because that is what three.js ``Quaternion`` serializes to, and the
    wire format is written for that producer. (``splat_io`` rot fields are ``wxyz`` —
    the two orders coexist in this repo, so the order is named at every boundary.)"""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4 or not np.isfinite(q).all():
        raise SyntheticViewError(
            "invalid_view", 400, "quaternion must be 4 finite numbers [x, y, z, w]"
        )
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise SyntheticViewError("invalid_view", 400, "quaternion has zero length")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# GL camera (x right, y up, z back) -> OpenCV camera (x right, y down, z forward).
# Its own inverse, but named rather than inlined: this single sign convention is the
# difference between a mask landing on the object and landing behind the camera.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def gl_pose_to_cv_c2w(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    """World-frame three.js camera pose -> ``(4, 4)`` OpenCV camera-to-world.

    The result is the ``trajectory.npz`` convention verbatim, so a synthetic view drops
    straight into ``segment_geometry.lift_mask_to_points`` beside a real keyframe."""
    t = np.asarray(position, dtype=np.float64).reshape(-1)
    if t.shape[0] != 3 or not np.isfinite(t).all():
        raise SyntheticViewError(
            "invalid_view", 400, "position must be 3 finite numbers [x, y, z]"
        )
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quaternion_to_matrix(quaternion_xyzw) @ _GL_TO_CV
    c2w[:3, 3] = t
    return c2w


def intrinsics_from_fov(fov_y_deg: float, width: int, height: int) -> list[float]:
    """``[fx, fy, cx, cy]`` for a three.js perspective camera at this vertical FOV.

    Square pixels: three.js derives the horizontal FOV from ``aspect = width / height``,
    which makes ``fx == fy == (height / 2) / tan(fov_y / 2)`` exactly — the aspect ratio
    is already carried by ``width``, so folding it into ``fx`` again would double-count
    it (a classic way to get a plausible-looking but horizontally-sheared projection)."""
    fov = float(fov_y_deg)
    if not np.isfinite(fov) or not (1.0 <= fov <= 179.0):
        raise SyntheticViewError(
            "invalid_view", 400, f"fov_y_deg must be within [1, 179], got {fov_y_deg!r}"
        )
    f = (float(height) / 2.0) / float(np.tan(np.radians(fov) / 2.0))
    return [f, f, float(width) / 2.0, float(height) / 2.0]


def project_world_point(
    c2w: Any, intrinsics: Any, point_world: Any
) -> Optional[tuple[float, float]]:
    """World point -> ``(px, py)`` in this view's pixel grid, or ``None`` when the point
    is at/behind the camera plane.

    Pixels are NOT clamped to the image: a point that projects off-frame returns its
    off-frame coordinates so callers can tell "behind the camera" from "outside the
    crop" — two different failures that a clamp would merge into one."""
    m = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    fx, fy, cx, cy = (float(v) for v in np.asarray(intrinsics, dtype=np.float64).reshape(4))
    p = np.asarray(point_world, dtype=np.float64).reshape(3)
    cam = m[:3, :3].T @ (p - m[:3, 3])  # world -> camera (OpenCV: +z forward)
    if cam[2] <= _EPS:
        return None
    return (cx + fx * cam[0] / cam[2], cy + fy * cam[1] / cam[2])


# ---------------------------------------------------------------------------
# wire parsing
# ---------------------------------------------------------------------------


def _decode_png(raw: Any, index: int) -> bytes:
    """Base64 PNG bytes -> raw bytes, with the format asserted rather than assumed.

    The magic check is the point: a client that posts a JPEG data-URL, or an empty
    string from a failed readback, must fail loudly here instead of persisting a blob
    the annotator will silently refuse to look at later."""
    if not isinstance(raw, str) or not raw.strip():
        raise SyntheticViewError(
            "invalid_view", 400, f"views[{index}].image_b64 must be a non-empty string"
        )
    text = raw.strip()
    if text.startswith("data:"):
        raise SyntheticViewError(
            "invalid_view",
            400,
            f"views[{index}].image_b64 carries a 'data:' prefix — send raw base64 only",
        )
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SyntheticViewError(
            "invalid_view", 400, f"views[{index}].image_b64 is not valid base64 ({exc})"
        ) from exc
    if len(data) > MAX_VIEW_BYTES:
        raise SyntheticViewError(
            "view_too_large",
            413,
            f"views[{index}] is {len(data):,} bytes, over the {MAX_VIEW_BYTES:,} limit",
        )
    if not data.startswith(_PNG_MAGIC):
        raise SyntheticViewError(
            "invalid_view",
            400,
            f"views[{index}] is not a PNG (the route stores PNG only, losslessly — a "
            "re-encoded render is no longer faithful evidence of what was rendered)",
        )
    return data


def _dimension(raw: Any, name: str, index: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SyntheticViewError(
            "invalid_view", 400, f"views[{index}].{name} must be an integer"
        ) from None
    if not (MIN_DIMENSION <= value <= MAX_DIMENSION):
        raise SyntheticViewError(
            "invalid_view",
            400,
            f"views[{index}].{name} must be within [{MIN_DIMENSION}, {MAX_DIMENSION}], got {value}",
        )
    return value


def parse_view(raw: Any, index: int) -> dict[str, Any]:
    """One wire view -> ``{png, meta}``, fully validated.

    ``meta`` is what lands on the record (never the bytes): the pose as posted, plus the
    OpenCV ``c2w`` + ``intrinsics`` derived here so every downstream consumer reads one
    already-converted number instead of re-deriving the convention."""
    if not isinstance(raw, dict):
        raise SyntheticViewError("invalid_view", 400, f"views[{index}] must be an object")
    width = _dimension(raw.get("width"), "width", index)
    height = _dimension(raw.get("height"), "height", index)
    png = _decode_png(raw.get("image_b64"), index)
    c2w = gl_pose_to_cv_c2w(raw.get("position"), raw.get("quaternion"))
    fov_y_deg = raw.get("fov_y_deg", 60.0)
    intrinsics = intrinsics_from_fov(fov_y_deg, width, height)

    view_id = uuid.uuid4().hex[:16]
    meta = {
        "view_id": view_id,
        "index": index,
        "blob_key": f"derived/{SYNTHETIC_VIEW_DIR}/{view_id}.png",
        "position": [float(v) for v in np.asarray(raw.get("position"), dtype=np.float64).reshape(3)],
        "quaternion": [
            float(v) for v in np.asarray(raw.get("quaternion"), dtype=np.float64).reshape(4)
        ],
        "fov_y_deg": float(fov_y_deg),
        "width": width,
        "height": height,
        # OpenCV c2w + [fx, fy, cx, cy] — the trajectory.npz convention (module docstring).
        "c2w": [[float(v) for v in row] for row in c2w],
        "intrinsics": intrinsics,
        "bytes": len(png),
        "created_at": time.time(),
        # Carried on the metadata itself so a consumer reading only the record still
        # knows what it is holding.
        "provenance": SYNTHETIC_VIEW_PROVENANCE,
        "provenance_detail": SYNTHETIC_VIEW_DETAIL,
        "label": str(raw.get("label") or "").strip()[:80] or None,
    }
    # WHICH renderer drew it. Absent for a browser capture, present for a server-side
    # render (``splat_render.py``). This is bookkeeping, NOT a ranking: both are renders
    # of the same splat and both carry the same ``synthetic view`` provenance. It exists
    # so that a reader can reproduce a view — which artifact, which SH degree, which
    # decimation — without guessing from the pixels.
    renderer = raw.get("renderer")
    if isinstance(renderer, dict):
        meta["renderer"] = {str(k): v for k, v in renderer.items()}
    return {"png": png, "meta": meta}


def parse_views(raw_views: Any) -> list[dict[str, Any]]:
    """The POST body's ``views`` array -> parsed views, index-stamped in order."""
    if not isinstance(raw_views, list) or not raw_views:
        raise SyntheticViewError(
            "invalid_view", 400, "expected a non-empty 'views' array"
        )
    if len(raw_views) > MAX_VIEWS:
        raise SyntheticViewError(
            "too_many_views",
            400,
            f"{len(raw_views)} views exceeds the {MAX_VIEWS} per-scene limit",
            limit=MAX_VIEWS,
        )
    return [parse_view(raw, i) for i, raw in enumerate(raw_views)]


def view_blob_relative_key(view_id: str) -> str:
    """``save_derived_artifact`` relative key for one view's PNG."""
    return f"{SYNTHETIC_VIEW_DIR}/{view_id}.png"


def persist_views(
    store: Any,
    user_id: str,
    scan_id: str,
    parsed: list[dict[str, Any]],
    *,
    existing: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Write the PNGs, then extend the record's manifest. Returns the stored metadata.

    Blobs first, record second: a manifest entry pointing at bytes that never landed is
    the one failure mode a reader cannot detect (the blob route just 404s and the view
    looks like a permissions problem). The reverse — an orphan blob — is inert.

    Shared by the upload route and by the server-side renderer so a headless ring lands
    in *exactly* the same field, the same blob namespace and the same order as an
    operator's capture. A second, parallel write path is how the two would drift.
    """
    prior = list(existing or [])
    stored: list[dict[str, Any]] = []
    for offset, item in enumerate(parsed):
        meta = dict(item["meta"])
        meta["index"] = len(prior) + offset
        try:
            meta["blob_key"] = store.save_derived_artifact(
                user_id, scan_id, view_blob_relative_key(meta["view_id"]), item["png"]
            )
        except Exception as exc:
            raise SyntheticViewError(
                "view_write_failed", 500, f"could not store view {meta['index']}: {exc}"
            ) from exc
        stored.append(meta)

    if not store.set_synthetic_views(user_id, scan_id, prior + stored):
        raise SyntheticViewError("not_found", 404, f"no scene {scan_id} for this owner")
    return stored


def manifest(record: Any) -> list[dict[str, Any]]:
    """The scene record's synthetic-view metadata list (never the bytes), or ``[]``."""
    if not isinstance(record, dict):
        return []
    views = record.get("synthetic_views")
    if not isinstance(views, list):
        return []
    return [v for v in views if isinstance(v, dict) and v.get("view_id")]


def public_view(meta: dict[str, Any], url: Optional[str] = None) -> dict[str, Any]:
    """One manifest entry as the GET route serves it (contract field order)."""
    out = {
        "view_id": meta.get("view_id"),
        "index": meta.get("index"),
        "position": meta.get("position"),
        "quaternion": meta.get("quaternion"),
        "fov_y_deg": meta.get("fov_y_deg"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "c2w": meta.get("c2w"),
        "intrinsics": meta.get("intrinsics"),
        "created_at": meta.get("created_at"),
        "provenance": meta.get("provenance", SYNTHETIC_VIEW_PROVENANCE),
        "provenance_detail": meta.get("provenance_detail", SYNTHETIC_VIEW_DETAIL),
        "label": meta.get("label"),
    }
    if url is not None:
        out["url"] = url
    return out
