"""Thin fal.ai queue-API wrapper (design/features.md §0.4) — W3.

**Contracts below are VERIFIED LIVE against fal on 2026-07-31** (keyed calls from
a Modal function with the ``fal-secret`` mounted). Every shape the blind
implementation guessed is annotated with what it actually turned out to be.

Queue transport::

    POST https://queue.fal.run/<app_id>   -> {request_id, status_url, response_url, ...}
    GET  <status_url>                     -> {status: IN_QUEUE|IN_PROGRESS|COMPLETED|...}
    GET  <response_url>                   -> the model result JSON

⚠️ **Always follow the ``status_url``/``response_url`` fal returns — never rebuild
them from the app id.** For a *subpath* app such as ``fal-ai/sam-3/image`` fal
answers with polling URLs rooted at the OWNER/APP pair only
(``…/fal-ai/sam-3/requests/<id>``); the constructed
``…/fal-ai/sam-3/image/requests/<id>/status`` returns **HTTP 405**. That single
bug would have failed every SAM 3 call. :func:`_queue_root` is only a fallback,
for sessions (tests) that do not echo the URLs back.

Auth is ``Authorization: Key <FAL_API_KEY>``, mounted from the ``fal-secret``
Modal secret, attached to both runtimes unconditionally (see ``_DEMO_SECRETS`` in
``modal_streaming.py``). Absent key -> :class:`FalKeyMissing` -> the honest
``503 {"error": "fal_key_missing"}``; never a silent downgrade, never a mock.

App ids stay env-overridable so a future fal rename needs no code change:

    DEMO_FAL_SAM3_APP     (default ``fal-ai/sam-3/image``)  — SAM 3 point/box segmentation
    DEMO_FAL_TRELLIS_APP  (default ``fal-ai/trellis``)      — TRELLIS image -> 3D **GLB mesh**

Transport is plain ``requests`` (already a server dependency); tests inject a
fake session via :func:`configure_session`.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Callable, Optional

from server.billing import prices

FAL_QUEUE_BASE = "https://queue.fal.run"

SAM3_APP_ENV = "DEMO_FAL_SAM3_APP"
TRELLIS_APP_ENV = "DEMO_FAL_TRELLIS_APP"
# Verified live 2026-07-31:
#   "fal-ai/sam3"        -> 404 Application not found
#   "fal-ai/sam-3"       -> accepts a submit, but it is the model GROUP, not an
#                           endpoint; the real single-image endpoint is below
#                           (there is also .../video, billed per 16 frames)
#   "fal-ai/sam-3/image" -> the endpoint we want
# TRELLIS confirmed at "fal-ai/trellis" (A100, image-to-3D, textured GLB out).
DEFAULT_SAM3_APP = "fal-ai/sam-3/image"
DEFAULT_TRELLIS_APP = "fal-ai/trellis"

# The wire error code every keyless generative route answers with (503 body).
ERROR_FAL_KEY_MISSING = "fal_key_missing"

# SAM 3's `prompt` field DEFAULTS TO THE STRING "wheel" server-side. Sending an
# explicit empty prompt is what makes a pure point/box click-segmentation work —
# omit it and SAM 3 also hunts for wheels and can return that mask instead.
SAM3_NO_TEXT_PROMPT = ""


class FalKeyMissing(RuntimeError):
    """Raised when FAL_API_KEY is absent — callers answer 503 fal_key_missing."""


class FalError(RuntimeError):
    """A fal request failed (submit / poll / fetch / model error).

    ``status`` is the HTTP code when the failure was an HTTP response and
    ``detail`` is fal's own error text (fal answers ``{"detail": ...}``), so
    routes can surface the real reason instead of a swallowed generic message."""

    def __init__(self, message: str, *, status: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


class FalAccountError(FalError):
    """fal rejected the key itself — exhausted balance (403 ``User is locked.
    Reason: Exhausted balance.``), revoked key, or an unauthorized app. Kept
    distinct from a model failure because the fix is a founder billing/key
    action, not a retry. Observed live on 2026-07-31."""


def _fal_detail(resp: Any) -> Any:
    """fal's error body is ``{"detail": ...}``; fall back to raw text."""
    try:
        body = resp.json()
    except Exception:
        return (getattr(resp, "text", "") or "")[:300]
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


def _raise_for_status(resp: Any, what: str) -> None:
    status = int(getattr(resp, "status_code", 0) or 0)
    if status < 400:
        return
    detail = _fal_detail(resp)
    rendered = detail if isinstance(detail, str) else json.dumps(detail, default=str)[:300]
    msg = f"{what} -> HTTP {status}: {rendered}"
    if status in (401, 402, 403):
        raise FalAccountError(msg, status=status, detail=detail)
    raise FalError(msg, status=status, detail=detail)


_session: Any = None  # injected by tests; else a lazily created requests.Session


def configure_session(session: Any) -> None:
    """Inject the HTTP session (tests pass a fake with ``.post``/``.get``;
    ``None`` resets to the lazy real ``requests.Session``)."""
    global _session
    _session = session


def _get_session() -> Any:
    global _session
    if _session is None:
        import requests

        _session = requests.Session()
    return _session


def fal_api_key() -> Optional[str]:
    key = os.environ.get("FAL_API_KEY", "").strip()
    return key or None


def require_key() -> str:
    key = fal_api_key()
    if not key:
        raise FalKeyMissing(
            "FAL_API_KEY is not configured — create the fal-secret Modal secret and "
            "redeploy; SAM 3 / TRELLIS calls are disabled without it"
        )
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Key {require_key()}", "Content-Type": "application/json"}


def _queue_root(app_id: str) -> str:
    """The owner/app pair fal roots polling URLs at (drops endpoint subpaths).

    ``fal-ai/sam-3/image`` -> ``fal-ai/sam-3``; ``fal-ai/trellis`` unchanged.
    FALLBACK ONLY — the URLs fal returns are always preferred."""
    parts = [p for p in str(app_id).split("/") if p]
    return "/".join(parts[:2]) if len(parts) >= 2 else str(app_id)


def submit(app_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Queue one request. Returns fal's submit body — ``request_id`` plus the
    ``status_url``/``response_url`` that :func:`wait` must follow."""
    resp = _get_session().post(
        f"{FAL_QUEUE_BASE}/{app_id}", json=payload, headers=_headers(), timeout=30
    )
    _raise_for_status(resp, f"fal submit {app_id}")
    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        raise FalError(f"fal submit {app_id} returned no request_id: {data}")
    root = _queue_root(app_id)
    return {
        "request_id": str(request_id),
        "status_url": data.get("status_url")
        or f"{FAL_QUEUE_BASE}/{root}/requests/{request_id}/status",
        "response_url": data.get("response_url")
        or f"{FAL_QUEUE_BASE}/{root}/requests/{request_id}",
    }


def wait(
    submitted: dict[str, Any],
    *,
    timeout_s: float = 180.0,
    poll_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll ``submitted['status_url']`` until COMPLETED, then GET the result."""
    status_url = str(submitted["status_url"])
    response_url = str(submitted["response_url"])
    request_id = submitted.get("request_id")
    deadline = time.monotonic() + timeout_s
    while True:
        resp = _get_session().get(status_url, headers=_headers(), timeout=30)
        _raise_for_status(resp, "fal status")
        body = resp.json()
        status = str(body.get("status", "")).upper()
        if status == "COMPLETED":
            break
        if status in ("FAILED", "CANCELLED", "ERROR"):
            raise FalError(
                f"fal request {request_id} ended {status}: "
                f"{json.dumps(body.get('logs') or body.get('error') or {}, default=str)[:300]}",
                detail=body,
            )
        if time.monotonic() > deadline:
            raise FalError(f"fal request {request_id} timed out after {timeout_s}s")
        sleep(poll_s)
    resp = _get_session().get(response_url, headers=_headers(), timeout=60)
    _raise_for_status(resp, "fal result")
    return resp.json()


def run(app_id: str, payload: dict[str, Any], *, timeout_s: float = 180.0) -> dict[str, Any]:
    """submit + wait in one call (raises FalKeyMissing before any network I/O)."""
    require_key()
    return wait(submit(app_id, payload), timeout_s=timeout_s)


def download(url: str, *, timeout_s: float = 120.0) -> bytes:
    """Fetch a result artifact. fal returns an ``https://…fal.media/…`` URL by
    default and an inline ``data:<mime>;base64,…`` URI under ``sync_mode`` —
    both arrive in the same ``url`` field, so both are handled here."""
    text = str(url)
    if text.startswith("data:"):
        try:
            head, payload = text.split(",", 1)
        except ValueError as exc:
            raise FalError(f"malformed data URI artifact: {text[:60]}") from exc
        if ";base64" in head:
            return base64.b64decode(payload)
        from urllib.parse import unquote_to_bytes

        return unquote_to_bytes(payload)
    resp = _get_session().get(text, timeout=timeout_s)
    _raise_for_status(resp, "fal artifact fetch")
    return resp.content


# ---------------------------------------------------------------------------
# model-specific entry points — payload shapes VERIFIED LIVE 2026-07-31
# ---------------------------------------------------------------------------


def segment_sam3(
    image_b64_data_uri: str,
    *,
    point_px: Optional[tuple[float, float]] = None,
    box_px: Optional[tuple[float, float, float, float]] = None,
    text_prompt: str = SAM3_NO_TEXT_PROMPT,
    sync_mode: bool = False,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """SAM 3 point/box segmentation on one image (F3 prompt paths b/c).

    ``image_url`` accepts a ``data:image/png;base64,…`` URI directly — verified
    live, so our render-sized PNGs need no upload to fal's storage API.

    The real prompt schema is two SEPARATE typed arrays of INTEGER PIXEL
    coordinates, not the single tagged ``prompts`` list this module used to
    guess::

        point_prompts: [{"x": int, "y": int, "label": 1|0, "object_id"?, "frame_index"?}]
        box_prompts:   [{"x_min": int, "y_min": int, "x_max": int, "y_max": int, ...}]

    ``label`` is 1 for foreground / 0 for background. Coordinates are in the
    input image's own pixel grid, origin top-left — the same convention our
    renders and ``lift_mask_to_points`` already use, so no flip is required.

    Both prompt kinds were exercised live (2026-07-31): a point at the target
    and an out-of-order box both returned the target's exact bbox at score 0.99
    and excluded a distractor. Masks come back mode-L, values {0, 255}, at full
    input resolution — never cropped to the prompt.

    Returns the raw fal result; ``routes_sam3d._extract_fal_mask`` turns
    ``masks[]`` into a bool array."""
    payload: dict[str, Any] = {
        "image_url": image_b64_data_uri,
        # Suppress the server-side default text prompt ("wheel").
        "prompt": text_prompt,
        # False => `image` is the raw mask, not the RGB with the mask burned in.
        "apply_mask": False,
        "include_scores": True,
        "include_boxes": True,
    }
    if sync_mode:
        payload["sync_mode"] = True
    if point_px is not None:
        payload["point_prompts"] = [
            {"x": int(round(float(point_px[0]))), "y": int(round(float(point_px[1]))), "label": 1}
        ]
    elif box_px is not None:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box_px]
        payload["box_prompts"] = [
            {"x_min": min(x0, x1), "y_min": min(y0, y1), "x_max": max(x0, x1), "y_max": max(y0, y1)}
        ]
    else:
        raise ValueError("segment_sam3 needs point_px or box_px")
    started = time.time()
    result = run(os.environ.get(SAM3_APP_ENV, DEFAULT_SAM3_APP), payload, timeout_s=timeout_s)
    # Flat per-request rate, from fal's own billing metadata on a live run. This is
    # the only third-party call whose price we have measured rather than estimated.
    print(
        f"[cost] surface=fal_sam3 calls=1 usd={prices.FAL_SAM3_USD_PER_REQUEST} "
        f"basis=fal_flat elapsed_s={round(time.time() - started, 2)}"
    )
    return result


def trellis_generate(
    *,
    image_b64_data_uri: str,
    seed: Optional[int] = None,
    ss_sampling_steps: Optional[int] = None,
    slat_sampling_steps: Optional[int] = None,
    mesh_simplify: Optional[float] = None,
    texture_size: Optional[int] = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """TRELLIS image-conditioned 3D generation.

    **It is image-to-3D ONLY.** ``fal-ai/trellis``'s input schema has exactly one
    required field, ``image_url``, and NO ``prompt`` field — there is no
    text-conditioned mode on this endpoint. (fal accepts unknown extra keys
    silently, so a stray ``prompt`` is not an error; it is simply ignored, which
    is worse than a rejection. We therefore never send one.)

    **The output is a textured GLB mesh, not a gaussian PLY.** Verified result::

        {"model_mesh": {"url", "content_type": "application/octet-stream",
                        "file_name": "model.glb", "file_size": <int>},
         "timings": {"prepare", "generation", "export"}}

    There is no ``model_gaussian`` key — that was our blind guess. The caller
    converts the GLB to a splat PLY (``genai.mesh_to_splat``) so the viewer's
    existing gaussian path renders it unchanged."""
    if not image_b64_data_uri:
        raise ValueError("trellis_generate needs an image (fal's TRELLIS is image-to-3D only)")
    payload: dict[str, Any] = {"image_url": image_b64_data_uri}
    if seed is not None:
        payload["seed"] = int(seed)
    if ss_sampling_steps is not None:
        payload["ss_sampling_steps"] = int(ss_sampling_steps)
    if slat_sampling_steps is not None:
        payload["slat_sampling_steps"] = int(slat_sampling_steps)
    if mesh_simplify is not None:
        payload["mesh_simplify"] = float(mesh_simplify)
    if texture_size is not None:
        payload["texture_size"] = int(texture_size)
    started = time.time()
    result = run(os.environ.get(TRELLIS_APP_ENV, DEFAULT_TRELLIS_APP), payload, timeout_s=timeout_s)
    # TRELLIS bills per A100 GPU-second with no published flat rate, so this is an
    # estimate, not a bill. Wall time is logged alongside it so the estimate can be
    # replaced with a real per-second rate once the fal dashboard is checked.
    print(
        f"[cost] surface=fal_trellis calls=1 usd={prices.FAL_TRELLIS_USD_ESTIMATE} "
        f"basis=estimate elapsed_s={round(time.time() - started, 2)}"
    )
    return result
