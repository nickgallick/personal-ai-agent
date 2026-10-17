"""Search executor returning content and citations from the selected provider."""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

try:
    import config
    from utils.api_clients import SonarClient
    from utils.error_handling import format_error
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import SonarClient
    from utils.error_handling import format_error

logger = logging.getLogger(__name__)


async def execute_search(
    description: str, context: str = "", session_id: str | None = None,
) -> dict[str, Any]:
    logger.debug("execute_search: query=%r context_len=%d", description[:80], len(context))

    try:
        provider = config.get_search_provider()
        if provider == "none":
            return {"content": "Web search is disabled (SEARCH_PROVIDER=none).", "citations": [], "success": False}
        if provider == "parallel":
            from utils.parallel_search import ParallelSearchClient

            return await ParallelSearchClient().search(description, context, session_id)

        client = SonarClient()
        system_prompt: str | None = context.strip() if context and context.strip() else None
        response = await client.search(query=description, system_prompt=system_prompt)

        content: str = ""
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            content = str(response)

        citations: list[str] = []
        raw_citations = response.get("citations", [])
        if isinstance(raw_citations, list):
            citations = [str(c) for c in raw_citations if c]

        return {"content": content, "citations": citations, "success": True}

    except Exception as exc:
        error_msg = format_error(exc)
        logger.error("execute_search failed: %s", error_msg)
        return {"content": error_msg, "citations": [], "success": False}
