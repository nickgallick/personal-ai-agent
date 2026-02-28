"""
General reasoning executor agent.

Sends a task description (and optional prior context) to a reasoning-grade
LLM via OpenRouter and returns the response.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error

logger = logging.getLogger(__name__)

_REASONING_SYSTEM_PROMPT: str = (
    "You are a highly capable reasoning assistant. Provide thorough, "
    "well-structured analysis. Use markdown formatting for clarity."
)


async def execute_reasoning(
    description: str,
    context: str = "",
    use_deep: bool = False,
) -> dict[str, Any]:
    model_key = "long_context" if use_deep else "reasoning"
    model = config.get_model(model_key)
    logger.debug("execute_reasoning: model=%s use_deep=%s description=%r", model, use_deep, description[:80])

    user_content_parts: list[str] = []
    if context and context.strip():
        user_content_parts.append(f"Context:\n{context.strip()}\n")
    user_content_parts.append(description)
    user_content = "\n".join(user_content_parts)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        client = OpenRouterClient()
        response = await client.chat(messages=messages, model=model, temperature=0.7, max_tokens=4096)
        content: str = client._extract_content(response)
        logger.debug("execute_reasoning succeeded: content_len=%d", len(content))
        return {"content": content, "success": True}

    except Exception as exc:
        error_msg = format_error(exc)
        logger.error("execute_reasoning failed: %s", error_msg)
        return {"content": error_msg, "success": False}
