"""
Task planning agent that decomposes user requests into subtasks.

The planner sends the user's message to the orchestration LLM and parses
its JSON response into a structured list of subtask dicts, each ready for
downstream executor agents to consume.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import os
from typing import Any

try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import PlanningError, format_error
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import PlanningError, format_error

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES: frozenset[str] = frozenset(
    {"search", "code", "reason", "browse", "image", "github", "file_analysis"}
)

_PLANNER_SYSTEM_PROMPT: str = (
    "You are a task planner for an AI agent. Analyze the user's request and "
    "decompose it into a list of subtasks. Each subtask must be a JSON object "
    "with: type (one of: search, code, reason, browse, image, github, "
    "file_analysis), description (clear instruction for the executor), and "
    "dependencies (list of integer indices of subtasks that must complete "
    "first, 0-indexed). For simple queries that need only one step, return a "
    "single subtask. If the user uploaded files, include file_analysis "
    "subtasks as appropriate. Return ONLY a valid JSON array, no other text."
)


def _strip_code_fences(text: str) -> str:
    pattern = r"```(?:json|JSON)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_array(text: str) -> str:
    stripped = _strip_code_fences(text)
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _validate_subtask(task: Any, index: int) -> dict:
    if not isinstance(task, dict):
        raise ValueError(f"Subtask {index} is not a dict: {task!r}")
    task_type = task.get("type", "")
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(f"Subtask {index} has invalid type {task_type!r}.")
    description = task.get("description", "")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Subtask {index} is missing a non-empty 'description'.")
    deps = task.get("dependencies", [])
    if not isinstance(deps, list):
        deps = []
    return {
        "type": task_type,
        "description": description.strip(),
        "dependencies": [int(d) for d in deps if isinstance(d, (int, float, str)) and str(d).isdigit()],
        "status": "pending",
        "result": None,
    }


def _fallback_plan(description: str) -> list[dict]:
    return [{"type": "reason", "description": description, "dependencies": [], "status": "pending", "result": None}]


async def plan_tasks(
    user_message: str,
    context: str = "",
    uploaded_files: list[dict] | None = None,
) -> list[dict]:
    logger.debug("plan_tasks called: message=%r, files=%s", user_message[:80], uploaded_files)

    prompt_parts: list[str] = [user_message]
    if context:
        prompt_parts.append(f"\n\nAdditional context:\n{context}")
    if uploaded_files:
        file_descriptions: list[str] = []
        for f in uploaded_files:
            name = f.get("name", "unknown")
            mime = f.get("type", "unknown type")
            size = f.get("size")
            size_str = f" ({size} bytes)" if size is not None else ""
            file_descriptions.append(f"  - {name} [{mime}]{size_str}")
        prompt_parts.append("\n\nThe user has uploaded the following file(s):\n" + "\n".join(file_descriptions))

    full_prompt = "".join(prompt_parts)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]

    try:
        client = OpenRouterClient()
        response = await client.chat(
            messages=messages,
            model=config.get_model("orchestration"),
            temperature=0.2,
            max_tokens=2048,
        )
        raw_content: str = client._extract_content(response)
    except Exception as exc:
        logger.error("Planner API call failed: %s", format_error(exc))
        return _fallback_plan(user_message)

    json_str = _extract_json_array(raw_content)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Planner JSON parse failed (%s). Attempting partial recovery.", exc)
        try:
            objects = re.findall(r"\{[^{}]*\}", json_str, re.DOTALL)
            if objects:
                parsed = json.loads("[" + ",".join(objects) + "]")
            else:
                raise ValueError("No JSON objects found in response")
        except Exception as inner_exc:
            logger.error("Planner partial recovery also failed: %s.", inner_exc)
            return _fallback_plan(user_message)

    if not isinstance(parsed, list):
        return _fallback_plan(user_message)

    validated: list[dict] = []
    for idx, raw_task in enumerate(parsed):
        try:
            validated.append(_validate_subtask(raw_task, idx))
        except ValueError as exc:
            logger.warning("Skipping invalid subtask %d: %s", idx, exc)

    if not validated:
        return _fallback_plan(user_message)

    logger.info("plan_tasks produced %d subtask(s).", len(validated))
    return validated
