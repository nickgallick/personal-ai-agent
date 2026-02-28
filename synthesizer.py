"""
Response synthesis module for Personal AI Agent.

Combines results from multiple subtasks (search, code execution, file
analysis, etc.) into a single coherent, well-formatted response using
the OpenRouter reasoning model.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Config / utils import — works from project root or top-level import
# ---------------------------------------------------------------------------
try:
    from utils.api_clients import OpenRouterClient
    import config
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.api_clients import OpenRouterClient
    import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are a response synthesizer. Combine the following task results into "
    "a clear, well-formatted response for the user. Use markdown formatting. "
    "If search results provided citations/URLs, include them as clickable "
    "markdown links [text](url) in your response. Ensure the response is "
    "coherent and directly addresses the user's original request."
)


# ---------------------------------------------------------------------------
# Helper: collect files_created
# ---------------------------------------------------------------------------


def collect_files_created(task_results: list[dict[str, Any]]) -> list[str]:
    """
    Collect all file paths created across multiple subtask results.

    Iterates through each subtask result dict, looks for a
    ``"files_created"`` key inside the nested ``"result"`` dict, and
    flattens all found paths into a single list.  Duplicate paths are
    preserved (callers may de-duplicate if needed).

    Parameters
    ----------
    task_results:
        List of subtask result dicts.  Each dict is expected to have at
        least a ``"result"`` key whose value is itself a dict that may
        contain a ``"files_created"`` list.

    Returns
    -------
    list[str]
        Flat list of file paths (strings) from all subtask results.
    """
    all_files: list[str] = []
    for task in task_results:
        result = task.get("result") or {}
        files = result.get("files_created") or []
        if isinstance(files, list):
            all_files.extend(str(f) for f in files)
        else:
            logger.warning(
                "Unexpected type for 'files_created' in task '%s': %s",
                task.get("type", "unknown"),
                type(files).__name__,
            )
    return all_files


# ---------------------------------------------------------------------------
# Helper: format_citations
# ---------------------------------------------------------------------------


def format_citations(citations: list[str]) -> str:
    """
    Format a list of URLs as a numbered markdown citations block.

    Each URL is rendered as a markdown link using the hostname as the
    display name so the list is human-readable.

    Parameters
    ----------
    citations:
        List of URL strings.  Empty or blank entries are skipped.

    Returns
    -------
    str
        A markdown-formatted citations string, for example::

            \\n\\n---\\n**Sources:**\\n1. [example.com](https://example.com)\\n...

        Returns an empty string if *citations* is empty.

    Examples
    --------
    >>> format_citations(["https://example.com", "https://other.com"])
    '\\n\\n---\\n**Sources:**\\n1. [example.com](https://example.com)\\n2. [other.com](https://other.com)'
    """
    clean: list[str] = [url.strip() for url in citations if url and url.strip()]
    if not clean:
        return ""

    lines: list[str] = ["\n\n---", "**Sources:**"]
    for idx, url in enumerate(clean, start=1):
        try:
            parsed = urlparse(url)
            display = parsed.netloc or parsed.path or url
        except Exception:  # noqa: BLE001
            display = url
        lines.append(f"{idx}. [{display}]({url})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_citations(task_results: list[dict[str, Any]]) -> list[str]:
    """
    Gather all unique citation URLs from all subtask results.

    Looks in ``task["result"]["citations"]`` for each task.  Deduplicates
    while preserving insertion order.

    Parameters
    ----------
    task_results:
        List of subtask result dicts.

    Returns
    -------
    list[str]
        Ordered, deduplicated list of citation URL strings.
    """
    seen: set[str] = set()
    citations: list[str] = []
    for task in task_results:
        result = task.get("result") or {}
        raw_citations = result.get("citations") or []
        if not isinstance(raw_citations, list):
            continue
        for url in raw_citations:
            if isinstance(url, str) and url.strip() and url not in seen:
                seen.add(url)
                citations.append(url)
    return citations


def _build_synthesis_prompt(
    user_message: str,
    task_results: list[dict[str, Any]],
    context: str,
) -> str:
    """
    Build the full user-turn prompt for the synthesis model call.

    The prompt contains:
    - The user's original message.
    - A numbered breakdown of each subtask (type, description, result).
    - Any additional context.

    Parameters
    ----------
    user_message:
        The original message the user sent to the agent.
    task_results:
        List of completed subtask dicts.
    context:
        Optional extra context string (e.g. conversation history summary).

    Returns
    -------
    str
        Formatted prompt string.
    """
    sections: list[str] = []

    sections.append(f"## User's Original Request\n{user_message}")

    if context:
        sections.append(f"## Additional Context\n{context}")

    sections.append(f"## Subtask Results ({len(task_results)} task(s))")

    for idx, task in enumerate(task_results, start=1):
        task_type = task.get("type", "unknown")
        task_desc = task.get("description", "(no description)")
        task_status = task.get("status", "unknown")
        result: dict[str, Any] = task.get("result") or {}

        # Determine the most useful content field to surface
        content_candidates = ["content", "response", "output", "text", "data"]
        result_text: str = ""
        for candidate in content_candidates:
            value = result.get(candidate)
            if value and isinstance(value, str):
                result_text = value
                break

        if not result_text:
            # Fall back to a readable repr of the whole result dict,
            # excluding large/binary fields
            filtered = {
                k: v
                for k, v in result.items()
                if k not in ("files_created", "citations") and not isinstance(v, (bytes, bytearray))
            }
            result_text = str(filtered) if filtered else "(empty result)"

        # Truncate individual result text to keep prompt manageable
        max_result_chars = 3000
        if len(result_text) > max_result_chars:
            result_text = (
                result_text[:max_result_chars]
                + f"\n[... truncated at {max_result_chars} characters ...]"
            )

        task_block = (
            f"### Task {idx}: {task_type}\n"
            f"**Description:** {task_desc}\n"
            f"**Status:** {task_status}\n"
            f"**Result:**\n{result_text}"
        )
        sections.append(task_block)

    sections.append(
        "## Instructions\nSynthesize the above task results into a single, "
        "coherent response that directly answers the user's original request. "
        "Use clear markdown formatting and include relevant citations as "
        "clickable links where appropriate."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def synthesize_response(
    task_results: list[dict[str, Any]],
    user_message: str,
    context: str = "",
) -> dict[str, Any]:
    """
    Combine results from multiple completed subtasks into a single response.

    The function:

    1. Collects all citation URLs from every subtask result.
    2. Collects all files created across subtasks.
    3. Builds a structured prompt containing the user's original message,
       optional context, and a numbered breakdown of every subtask result.
    4. Sends the prompt to the OpenRouter reasoning model with a synthesis
       system prompt.
    5. Returns the synthesized response along with deduplicated citations
       and a flat list of all files created.

    Parameters
    ----------
    task_results:
        List of completed subtask dicts.  Each dict should contain:

        - ``"type"`` (str): category label, e.g. ``"search"``, ``"code"``.
        - ``"description"`` (str): human-readable description of the subtask.
        - ``"result"`` (dict): output of the subtask execution.  May include
          ``"content"``, ``"citations"`` (list[str]), and/or
          ``"files_created"`` (list[str]).
        - ``"status"`` (str): completion status, e.g. ``"completed"``.

    user_message:
        The original message the user sent to the agent.
    context:
        Optional extra context string (e.g. a summary of prior conversation
        turns or additional instructions).

    Returns
    -------
    dict
        On success::

            {
                "response": "<synthesized markdown text>",
                "citations": ["https://...", ...],
                "files_created": ["<path1>", ...],
            }

        On error, ``"response"`` contains the error message and the other
        fields are empty lists.
    """
    logger.info(
        "synthesize_response called | tasks=%d | message=%r",
        len(task_results),
        user_message[:100],
    )

    # ------------------------------------------------------------------
    # Step 1 — Collect metadata from all subtasks
    # ------------------------------------------------------------------
    all_citations: list[str] = _collect_citations(task_results)
    all_files_created: list[str] = collect_files_created(task_results)

    logger.debug(
        "Collected %d citation(s) and %d created file(s) from subtasks",
        len(all_citations),
        len(all_files_created),
    )

    # ------------------------------------------------------------------
    # Step 2 — Build the synthesis prompt
    # ------------------------------------------------------------------
    synthesis_prompt = _build_synthesis_prompt(
        user_message=user_message,
        task_results=task_results,
        context=context,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": synthesis_prompt},
    ]

    # ------------------------------------------------------------------
    # Step 3 — Call the reasoning model
    # ------------------------------------------------------------------
    try:
        client = OpenRouterClient()
        async with client:
            logger.debug("Sending synthesis request to reasoning model")
            synthesized_text: str = await client.reason(messages)

        logger.info(
            "Synthesis complete | response_length=%d | citations=%d | files_created=%d",
            len(synthesized_text),
            len(all_citations),
            len(all_files_created),
        )

        return {
            "response": synthesized_text,
            "citations": all_citations,
            "files_created": all_files_created,
        }

    except Exception as exc:  # noqa: BLE001
        error_msg = f"Response synthesis failed: {type(exc).__name__}: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            "response": error_msg,
            "citations": [],
            "files_created": [],
        }
