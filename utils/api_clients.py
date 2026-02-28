"""
Async API client wrappers for Perplexity Sonar and OpenRouter.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import os
from typing import Any

import httpx

try:
    import config
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES: int = 3
_BASE_DELAY: float = 1.0
_MAX_DELAY: float = 30.0


async def _retry_request(fn, *args, max_retries=_MAX_RETRIES, base_delay=_BASE_DELAY, max_delay=_MAX_DELAY, **kwargs) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response: httpx.Response = await fn(*args, **kwargs)
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            last_exc = httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
            logger.warning("Retryable HTTP %s on attempt %d/%d", response.status_code, attempt + 1, max_retries + 1)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            last_exc = exc
            logger.warning("Network error on attempt %d/%d: %s", attempt + 1, max_retries + 1, exc)

        if attempt < max_retries:
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


class SonarClient:
    """Async client for the Perplexity Sonar API."""

    _ENDPOINT: str = "https://api.perplexity.ai/chat/completions"
    _TIMEOUT_SECONDS: float = 60.0

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key: str = api_key or config.SONAR_API_KEY
        if not self._api_key:
            raise ValueError("Sonar API key is required. Set SONAR_API_KEY in your environment.")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SonarClient":
        self._client = httpx.AsyncClient(timeout=self._TIMEOUT_SECONDS)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self._TIMEOUT_SECONDS)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        client = self._get_client()
        own_client = self._client is None
        try:
            response = await _retry_request(client.post, self._ENDPOINT, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise APIClientError(f"Sonar API error {exc.response.status_code}: {exc.response.text}", status_code=exc.response.status_code, service="sonar") from exc
            return response.json()
        finally:
            if own_client:
                await client.aclose()

    async def search(self, query: str, model: str | None = None, system_prompt: str | None = None) -> dict[str, Any]:
        resolved_model = model or config.MODEL_ROUTING["search_deep"]
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        payload: dict[str, Any] = {"model": resolved_model, "messages": messages}
        return await self._post(payload)

    async def quick_search(self, query: str) -> dict[str, Any]:
        return await self.search(query, model=config.MODEL_ROUTING["search_quick"])


class OpenRouterClient:
    """Async client for the OpenRouter API (OpenAI-compatible)."""

    _ENDPOINT: str = "https://openrouter.ai/api/v1/chat/completions"
    _CHAT_TIMEOUT: float = 60.0
    _IMAGE_TIMEOUT: float = 120.0
    _HTTP_REFERER: str = "https://github.com/personal-ai-agent"
    _APP_TITLE: str = "Personal AI Agent"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key: str = api_key or config.OPENROUTER_API_KEY
        if not self._api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY in your environment.")
        self._chat_client: httpx.AsyncClient | None = None
        self._image_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenRouterClient":
        self._chat_client = httpx.AsyncClient(timeout=self._CHAT_TIMEOUT)
        self._image_client = httpx.AsyncClient(timeout=self._IMAGE_TIMEOUT)
        return self

    async def __aexit__(self, *_: Any) -> None:
        for client in (self._chat_client, self._image_client):
            if client:
                await client.aclose()
        self._chat_client = None
        self._image_client = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._HTTP_REFERER,
            "X-Title": self._APP_TITLE,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _post(self, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        if timeout == self._IMAGE_TIMEOUT and self._image_client is not None:
            client = self._image_client
            own_client = False
        elif self._chat_client is not None:
            client = self._chat_client
            own_client = False
        else:
            client = httpx.AsyncClient(timeout=timeout)
            own_client = True

        try:
            response = await _retry_request(client.post, self._ENDPOINT, headers=self._headers(), json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise APIClientError(f"OpenRouter API error {exc.response.status_code}: {exc.response.text}", status_code=exc.response.status_code, service="openrouter") from exc
            return response.json()
        finally:
            if own_client:
                await client.aclose()

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIClientError(f"Unexpected OpenRouter response shape: {exc}", status_code=None, service="openrouter") from exc

    async def chat(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.7, max_tokens: int = 4096) -> dict[str, Any]:
        resolved_model = model or config.MODEL_ROUTING["orchestration"]
        payload: dict[str, Any] = {"model": resolved_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        return await self._post(payload, timeout=self._CHAT_TIMEOUT)

    async def generate_image(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        resolved_model = model or config.MODEL_ROUTING["image"]
        payload: dict[str, Any] = {"model": resolved_model, "messages": [{"role": "user", "content": prompt}], "modalities": ["image"]}
        raw = await self._post(payload, timeout=self._IMAGE_TIMEOUT)
        images: list[str] = []
        try:
            msg = raw["choices"][0]["message"]
            raw_images = msg.get("images") or []
            images.extend(raw_images)
        except (KeyError, IndexError, TypeError):
            pass
        return {"images": images, "_raw": raw}

    async def code(self, messages: list[dict[str, str]]) -> str:
        response = await self.chat(messages=messages, model=config.MODEL_ROUTING["coding"])
        return self._extract_content(response)

    async def reason(self, messages: list[dict[str, str]]) -> str:
        response = await self.chat(messages=messages, model=config.MODEL_ROUTING["reasoning"])
        return self._extract_content(response)

    async def fast(self, messages: list[dict[str, str]]) -> str:
        response = await self.chat(messages=messages, model=config.MODEL_ROUTING["fast"])
        return self._extract_content(response)


class APIClientError(Exception):
    """Raised when an API call fails after all retries."""

    def __init__(self, message: str, status_code: int | None = None, service: str = "unknown") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.service = service

    def __repr__(self) -> str:
        return f"APIClientError(service={self.service!r}, status_code={self.status_code}, message={str(self)!r})"
