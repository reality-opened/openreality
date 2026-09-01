"""Contract tests for ObjectEnricher + crop helper + SerpApi provider (no GPU / network).

Uses a stub OpenRouter client (canned JSON) and a stub reverse-search provider so the
whole enrichment flow — crop, grounding, VLM call, graceful degrade, and server-
authoritative fields — is exercised without torch/cv2 or any HTTP.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from server.scene_report.object_enricher import (
    ObjectEnricher,
    SerpApiLensProvider,
    crop_from_box,
)
from server.scene_report.schemas import ObjectDescription, ProductMatch


class StubLLM:
    """Stands in for OpenRouterClient.chat_json_model."""

    def __init__(self, payload=None, raise_exc=False, model="stub/model"):
        self.payload = payload or {}
        self.raise_exc = raise_exc
        self._model = model
        self.calls = []

    def chat_json_model(self, model_cls, system_prompt, user_prompt,
                        images_b64=None, temperature=0.2, max_tokens=600):
        self.calls.append({"user": user_prompt, "images": images_b64})
        if self.raise_exc:
            raise RuntimeError("boom")
        obj = model_cls.model_validate(self.payload)
        resp = SimpleNamespace(model=self._model, content="", degraded=False)
        return obj, resp


class StubReverse:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def search(self, image_url):
        self.calls.append(image_url)
        return self.matches


# --- crop helper -------------------------------------------------------------------

def test_crop_from_box_exact_no_pad():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = crop_from_box(frame, [10, 20, 30, 40], pad=0.0)  # x0,y0,x1,y1
    assert crop.shape == (20, 20, 3)
    assert np.array_equal(crop, frame[20:40, 10:30])


def test_crop_from_box_clamps_to_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = crop_from_box(frame, [40, 40, 80, 80], pad=0.5)  # exceeds bounds + padding
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 50 and crop.shape[1] <= 50


def test_crop_from_box_degenerate_falls_back_to_full_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert crop_from_box(frame, [5, 5, 5, 5]).shape == (10, 10, 3)  # zero-area
    assert crop_from_box(frame, None).shape == (10, 10, 3)  # bad box


# --- enricher ----------------------------------------------------------------------

def test_describe_vlm_only_sets_category_and_model():
    llm = StubLLM(payload={"brand": "Apple", "model": "MacBook Pro", "confidence": 0.7})
    enricher = ObjectEnricher(llm, model="cfg/model")
    desc = enricher.describe("Y3JvcA==", "laptop")
    assert isinstance(desc, ObjectDescription)
    assert desc.category == "laptop"  # always-safe fallback
    assert desc.title == "Apple MacBook Pro"  # derived from brand+model
    assert desc.model_name == "stub/model"  # from the response, not cfg
    assert desc.grounded_by_search is False
    assert desc.matches == []
    assert llm.calls[0]["images"] == ["Y3JvcA=="]  # crop forwarded as the image


def test_describe_grounded_uses_server_authoritative_matches():
    # The model hallucinates a match; the real reverse-search results must win.
    llm = StubLLM(payload={"title": "MacBook", "matches": [{"title": "HALLUCINATED"}]})
    real = [ProductMatch(title="Apple MacBook Pro 14", price="$1999", source="apple.com")]
    enricher = ObjectEnricher(llm, reverse_search=StubReverse(real))
    desc = enricher.describe("Y3JvcA==", "laptop", crop_url="https://w/crop.jpg")
    assert desc.grounded_by_search is True
    assert [m.title for m in desc.matches] == ["Apple MacBook Pro 14"]  # not HALLUCINATED


def test_describe_no_crop_url_skips_reverse_search():
    rev = StubReverse([ProductMatch(title="x")])
    enricher = ObjectEnricher(StubLLM(payload={"category": "shoe"}), reverse_search=rev)
    desc = enricher.describe("Y3JvcA==", "shoe")  # no crop_url
    assert rev.calls == []  # provider never queried
    assert desc.grounded_by_search is False


def test_describe_degrades_on_llm_failure():
    enricher = ObjectEnricher(StubLLM(raise_exc=True))
    desc = enricher.describe("Y3JvcA==", "chair")
    assert desc.category == "chair"
    assert desc.title == "chair"
    assert desc.confidence == 0.0  # honest, not a fabricated guess


# --- SerpApi provider --------------------------------------------------------------

def test_serpapi_parses_visual_matches(monkeypatch):
    provider = SerpApiLensProvider("KEY")
    monkeypatch.setattr(
        provider, "_get_json",
        lambda url, params: {
            "visual_matches": [
                {"title": "Air Jordan 1", "source": "nike.com", "link": "http://x",
                 "price": {"value": "$180"}},
                {"title": "knockoff", "source": "ebay", "link": "http://y", "price": "$40"},
            ]
        },
    )
    matches = provider.search("https://w/crop.jpg")
    assert [m.title for m in matches] == ["Air Jordan 1", "knockoff"]
    assert matches[0].price == "$180"  # dict-form price
    assert matches[1].price == "$40"   # scalar-form price


def test_serpapi_no_key_or_url_returns_empty():
    assert SerpApiLensProvider("").search("https://w/crop.jpg") == []
    assert SerpApiLensProvider("KEY").search("") == []


def test_serpapi_swallows_errors(monkeypatch):
    provider = SerpApiLensProvider("KEY")

    def boom(url, params):
        raise RuntimeError("network down")

    monkeypatch.setattr(provider, "_get_json", boom)
    assert provider.search("https://w/crop.jpg") == []  # degrades, never raises
