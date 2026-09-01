"""Fine-grained object enrichment.

Turns a coarse detection ("laptop", "shoes") into a rich, evidence-grounded
``ObjectDescription`` ("Apple MacBook Pro 14\", Space Black, likely M3") by running a
vision-language captioning pass over a crop of the object's best keyframe — optionally
grounded by a reverse image search (e.g. Google Lens via SerpApi) so brand/model claims
cite real listings instead of being hallucinated.

Deliberately GPU-free and duck-typed (numpy + pydantic + a lazily-imported HTTP client),
so it imports without torch/cv2/open3d, unit-tests with stubs, and could run on the CPU
broker. The caller supplies the crop already encoded as base64 JPEG (the GPU worker has
the frame tensors and ``ObjectDetector.image_to_base64``); this module never touches the
SLAM stack.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from server.scene_report.schemas import ObjectDescription, ProductMatch


# ---------------------------------------------------------------------------
# Crop helper (pure numpy — no cv2/torch)
# ---------------------------------------------------------------------------

def crop_from_box(frame_np: np.ndarray, box_2d, pad: float = 0.15) -> np.ndarray:
    """Crop ``frame_np`` (H, W, 3 RGB) to ``box_2d`` ([x0, y0, x1, y1] px) with padding.

    Padding gives the VLM context (a logo often sits at the object's edge); too-tight a
    crop loses the very detail we want. Coordinates are clamped to the frame. Falls back
    to the full frame if the box is degenerate.
    """
    h, w = frame_np.shape[:2]
    try:
        x0, y0, x1, y1 = (float(v) for v in list(box_2d)[:4])
    except Exception:
        return frame_np
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    bw, bh = (x1 - x0), (y1 - y0)
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    x0 = int(max(0, np.floor(x0)))
    y0 = int(max(0, np.floor(y0)))
    x1 = int(min(w, np.ceil(x1)))
    y1 = int(min(h, np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return frame_np
    return frame_np[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Reverse image search (optional grounding source)
# ---------------------------------------------------------------------------

class ReverseImageSearchProvider(Protocol):
    """Given a publicly fetchable image URL, return candidate product matches."""

    def search(self, image_url: str) -> list[ProductMatch]:  # pragma: no cover - protocol
        ...


class SerpApiLensProvider:
    """Google Lens reverse image search via SerpApi.

    Google has no official Lens API; SerpApi runs Lens on their infra and exposes a REST
    endpoint that takes a **public image URL** (``url=``) — direct uploads are not
    supported, so the caller must host the crop somewhere fetchable first. Best-effort:
    any failure returns ``[]`` so enrichment degrades to VLM-only.
    """

    ENDPOINT = "https://serpapi.com/search.json"

    def __init__(
        self,
        api_key: str,
        timeout: float = 15.0,
        max_results: int = 8,
        search_type: str = "products",
    ):
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_results = int(max_results)
        self.search_type = search_type

    def search(self, image_url: str) -> list[ProductMatch]:
        if not self.api_key or not image_url:
            return []
        params = {
            "engine": "google_lens",
            "type": self.search_type,
            "url": image_url,
            "api_key": self.api_key,
        }
        try:
            data = self._get_json(self.ENDPOINT, params)
        except Exception as e:  # network / quota / parse — degrade to VLM-only
            print(f"[enrich] reverse image search failed: {e}")
            return []
        items = data.get("visual_matches") or data.get("products") or []
        matches: list[ProductMatch] = []
        for item in items[: self.max_results]:
            if not isinstance(item, dict):
                continue
            matches.append(
                ProductMatch(
                    title=str(item.get("title") or ""),
                    source=str(item.get("source") or ""),
                    link=str(item.get("link") or ""),
                    price=_extract_price(item.get("price")),
                )
            )
        return matches

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            import requests  # lazy: not needed unless reverse search is enabled

            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            import json as _json
            from urllib.parse import urlencode
            from urllib.request import urlopen

            full = f"{url}?{urlencode(params)}"
            with urlopen(full, timeout=self.timeout) as r:  # noqa: S310 - fixed host
                return _json.loads(r.read().decode("utf-8"))


def _extract_price(price: Any) -> str:
    if isinstance(price, dict):
        return str(price.get("value") or price.get("extracted_value") or "")
    return str(price) if price else ""


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You identify a single object from a cropped photo for a 3D scene catalog. "
    "Work strictly from visible evidence. First read any visible text, logos, or labels; "
    "then identify the brand and the specific model/variant and the color/colorway. "
    "Ground every brand and model claim in something you can actually see (a logo, a "
    "distinctive shape, a printed label) — if it is not legible, leave that field empty "
    "and lower your confidence. Never invent a model number, year, or name you cannot "
    "justify from the image. Always provide a safe generic `category` even when the "
    "fine-grained identity is unknown. `confidence` is calibrated 0..1 for the brand/model "
    "identification specifically: use below 0.4 when you are guessing. Put any hedging in "
    "`uncertainty_note`. Respond with ONLY a JSON object matching the requested schema."
)


class ObjectEnricher:
    """Crop -> (optional reverse image search) -> VLM -> ``ObjectDescription``.

    ``llm_client`` is an ``OpenRouterClient`` (its ``primary_model`` is the enrichment
    model); ``reverse_search`` is an optional provider used only when the caller passes a
    public ``crop_url``. Every path degrades gracefully — a failure yields a minimal
    description rather than raising into the click/report flow.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "",
        reverse_search: Optional[ReverseImageSearchProvider] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ):
        self.llm_client = llm_client
        self.model = model
        self.reverse_search = reverse_search
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def describe(
        self,
        crop_b64: str,
        query: str,
        crop_url: Optional[str] = None,
    ) -> ObjectDescription:
        query = (query or "").strip()

        matches: list[ProductMatch] = []
        if self.reverse_search is not None and crop_url:
            try:
                matches = self.reverse_search.search(crop_url) or []
            except Exception as e:  # provider should already swallow, belt-and-suspenders
                print(f"[enrich] reverse search error: {e}")
                matches = []
        grounded = bool(matches)

        user_prompt = self._build_user_prompt(query, matches)
        try:
            desc, response = self.llm_client.chat_json_model(
                ObjectDescription,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images_b64=[crop_b64] if crop_b64 else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            print(f"[enrich] VLM describe failed: {e}")
            return ObjectDescription(
                category=query,
                title=query,
                matches=matches,
                grounded_by_search=grounded,
            )

        # Server-authoritative fields: never trust the model to echo the real matches or
        # the grounding flag, and always guarantee a generic category.
        desc.matches = matches
        desc.grounded_by_search = grounded
        desc.model_name = getattr(response, "model", "") or self.model
        if not desc.category:
            desc.category = query
        if not desc.title:
            desc.title = " ".join(p for p in (desc.brand, desc.model) if p) or desc.category
        return desc

    @staticmethod
    def _build_user_prompt(query: str, matches: list[ProductMatch]) -> str:
        lines = [f'The object was detected by an open-vocabulary search for: "{query}".']
        if matches:
            lines.append(
                "\nReverse image search returned these candidate product listings. Use them "
                "to pin down the exact brand/model/variant, but only adopt a match that is "
                "consistent with what you actually see in the image:"
            )
            for i, m in enumerate(matches[:8], 1):
                parts = [m.title]
                if m.price:
                    parts.append(f"price {m.price}")
                if m.source:
                    parts.append(f"source {m.source}")
                lines.append(f"{i}. " + " — ".join(p for p in parts if p))
        else:
            lines.append("\nNo external search results are available; rely on the image alone.")
        lines.append(
            "\nReturn the JSON object with fields: title, category, brand, model, color, "
            "materials[], visible_text[], distinguishing_features[], confidence, "
            "uncertainty_note. Do NOT include a `matches` field."
        )
        return "\n".join(lines)
