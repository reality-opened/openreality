"""OpenRouter client wrapper with retries, fallbacks, and JSON parsing helpers."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    model: str
    degraded: bool


class LLMFormatError(RuntimeError):
    """The model answered, but not as the JSON object the caller asked for.

    Subclasses ``RuntimeError`` so every existing handler keeps catching it, and it
    carries the raw completion so callers can decide what to do with the prose rather
    than conflating "the model said something unparseable" with "there was no model".
    The distinction matters: the second is an outage, the first is a formatting slip
    on an answer we already paid for.
    """

    def __init__(self, message: str, response: "LLMResponse"):
        super().__init__(message)
        self.response = response
        self.content = response.content


def _is_permanent_model_error(exc: Exception) -> bool:
    """True when retrying this model can never help — an unknown/retired slug (404)
    or a request the provider rejects outright (400, e.g. a text-only model handed
    images).

    Model rosters are env-configured and providers retire slugs without notice, so a
    bad primary is a routine condition, not an exception. Retrying it just burns wall
    clock before the fallback chain engages — on a live demo take that is the
    difference between a smooth degrade and a visible stall."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (400, 404):
        return True
    return type(exc).__name__ in ("NotFoundError", "BadRequestError")


class OpenRouterClient:
    """Small OpenRouter abstraction for robust text/json chat calls."""

    def __init__(
        self,
        api_key: str,
        primary_model: str,
        fallback_models: Optional[list[str]] = None,
        timeout: float = 20.0,
        app_name: str = "Open-Reality",
        referer: Optional[str] = None,
        max_retries: int = 2,
        usage_sink: Optional[Any] = None,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        self.primary_model = primary_model
        self.fallback_models = [m for m in (fallback_models or []) if m and m != primary_model]
        self.max_retries = max(0, int(max_retries))
        self.timeout = timeout
        self.app_name = app_name
        self.referer = referer or os.environ.get("OPENROUTER_HTTP_REFERER", "https://real-eyes.local")

        # Cost accounting. Attach a ``server.billing.UsageTally`` (anything with a
        # ``record(model, usage)`` method) and every call asks OpenRouter to return
        # its real USD charge. Left unset the client behaves exactly as before — the
        # extra_body is only sent when someone is listening, so an unattached caller
        # sends a byte-identical request.
        self.usage_sink = usage_sink

        self.last_model = primary_model
        self.degraded_mode = False

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_json_blob(text: str) -> Optional[str]:
        if not text:
            return None

        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            return match.group(1)

        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return None

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant", "system"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})

        if images_b64:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for img_b64 in images_b64:
                if not img_b64:
                    continue
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    }
                )
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    def _request_once(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        extra: dict[str, Any] = {}
        if self.usage_sink is not None:
            extra["extra_body"] = {"usage": {"include": True}}
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": self.referer,
                "X-OpenRouter-Title": self.app_name,
            },
            **extra,
        )
        if self.usage_sink is not None:
            # Record before parsing: a malformed body still cost us money.
            self.usage_sink.record(model, getattr(response, "usage", None))
        msg = response.choices[0].message
        return self._normalize_content(msg.content)

    def _retry_with_backoff(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # pragma: no cover - provider/network variance
                err = exc
                if _is_permanent_model_error(exc):
                    # Unknown slug / rejected request: skip the backoff, let the
                    # caller move to the next model in the chain immediately.
                    break
                if attempt < self.max_retries:
                    sleep_s = (0.25 * (2 ** attempt)) + random.uniform(0.0, 0.2)
                    time.sleep(sleep_s)
        if err is None:
            raise RuntimeError("Unknown OpenRouter error")
        raise err

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.4,
        max_tokens: int = 512,
    ) -> LLMResponse:
        models = [self.primary_model] + self.fallback_models
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
        )

        errors: list[str] = []
        for idx, model in enumerate(models):
            try:
                content = self._retry_with_backoff(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                degraded = idx > 0
                self.last_model = model
                self.degraded_mode = degraded
                if degraded:
                    # Loud on purpose: a silent fallback means the roster the founder
                    # thinks is running is NOT the roster that produced the output.
                    print(
                        f"[llm.openrouter] DEGRADED — primary '{self.primary_model}' "
                        f"unusable, answered with fallback '{model}' "
                        f"(failures: {' | '.join(errors)})"
                    )
                return LLMResponse(content=content, model=model, degraded=degraded)
            except Exception as exc:  # pragma: no cover - provider/network variance
                errors.append(f"{model}: {exc}")
                if _is_permanent_model_error(exc):
                    print(
                        f"[llm.openrouter] model '{model}' rejected permanently "
                        f"({type(exc).__name__}) — check the slug against "
                        f"GET openrouter.ai/api/v1/models; trying next in chain"
                    )

        raise RuntimeError("OpenRouter chat failed across all models: " + " | ".join(errors))

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 768,
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        blob = self._extract_json_blob(response.content)
        if blob is None:
            raise LLMFormatError("No JSON object found in LLM response", response)
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise LLMFormatError(f"Malformed JSON in LLM response: {exc}", response) from exc
        if not isinstance(parsed, dict):
            raise LLMFormatError("Expected top-level JSON object from LLM response", response)
        return parsed, response

    def chat_json_model(
        self,
        model_cls,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 768,
    ):
        parsed, response = self.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return model_cls.model_validate(parsed), response

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Batch text embeddings via the OpenAI-compatible ``/embeddings`` endpoint
        (OpenRouter supports it, so this reuses the same key/base URL as chat). One
        request for all inputs; returns vectors row-aligned to ``texts``. Raises on
        failure — callers that want graceful degradation should catch it."""
        if not texts:
            return []
        extra: dict[str, Any] = {}
        if self.usage_sink is not None:
            extra["extra_body"] = {"usage": {"include": True}}
        resp = self.client.embeddings.create(model=model, input=list(texts), **extra)
        if self.usage_sink is not None:
            # Embeddings were the one scene-report cost with no figure attached
            # anywhere (SCENE_QA_EMBED_MODEL); count them like any other call.
            self.usage_sink.record(model, getattr(resp, "usage", None))
        return [list(item.embedding) for item in resp.data]
