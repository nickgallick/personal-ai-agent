"""Anonymous Parallel Search MCP over the SDK's Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from utils.api_clients import APIClientError

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "personal-ai-agent/1.0 (+https://github.com/nickgallick/personal-ai-agent)"


class ParallelSearchClient:
    """Search without credentials; return the executor's content/citations contract."""

    _TIMEOUT_SECONDS = 60.0

    async def search(
        self, query: str, context: str = "", session_id: str | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("A non-empty search query is required.")
        if session_id is not None and not 1 <= len(session_id) <= 100:
            raise ValueError("Search session_id must contain 1 to 100 characters.")
        objective = query
        if context.strip():
            objective += "\n\nRelevant context:\n" + context[:4000]
        arguments: dict[str, Any] = {
            "objective": objective,
            "search_queries": [query[:200]],
        }
        if session_id is not None:
            arguments["session_id"] = session_id

        try:
            # Bound the entire operation, including discovery, and release transports
            # on errors or cancellation. No Authorization header or ambient API key.
            async with asyncio.timeout(self._TIMEOUT_SECONDS):
                async with httpx.AsyncClient(
                    headers={"User-Agent": USER_AGENT},
                    timeout=self._TIMEOUT_SECONDS,
                ) as http_client:
                    async with streamable_http_client(ENDPOINT, http_client=http_client) as (read, write, _):
                        async with ClientSession(
                            read, write,
                            read_timeout_seconds=timedelta(seconds=self._TIMEOUT_SECONDS),
                            client_info=types.Implementation(name="personal-ai-agent", version="1.0"),
                        ) as session:
                            await session.initialize()
                            result = await session.call_tool("web_search", arguments)
            return _normalize_result(result)
        except TimeoutError as exc:
            raise APIClientError("Parallel search timed out.", service="parallel") from exc
        except APIClientError:
            raise
        except Exception as exc:
            # AnyIO may wrap transport/protocol errors in task-group exceptions.
            # Keep their actual cause visible in the executor's error response.
            cause = exc
            while isinstance(cause, ExceptionGroup):
                cause = cause.exceptions[0]
            message = (
                "Parallel search timed out."
                if isinstance(cause, (TimeoutError, httpx.TimeoutException))
                else f"Parallel Search MCP failed: {cause}"
            )
            raise APIClientError(message, service="parallel") from exc


def _normalize_result(result: types.CallToolResult) -> dict[str, Any]:
    text = "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))
    if result.isError:
        raise APIClientError("Parallel search failed: " + (text or "MCP tool error"), service="parallel")
    data = result.structuredContent
    if data is None:
        # Some MCP servers supply the same JSON only as text content.
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise APIClientError("Invalid Parallel search response.", service="parallel") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise APIClientError("Invalid Parallel search response.", service="parallel")

    sections: list[str] = []
    citations: list[str] = []
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                continue
        except ValueError:
            continue
        excerpts = item.get("excerpts")
        if not isinstance(excerpts, list) or any(not isinstance(e, str) for e in excerpts):
            continue
        title = item.get("title") or url
        sections.append(f"{title}\n{url}\n" + "\n".join(excerpts))
        if url not in citations:
            citations.append(url)
    return {
        "content": "\n\n".join(sections) or "No web search results found.",
        "citations": citations,
        "success": True,
    }
