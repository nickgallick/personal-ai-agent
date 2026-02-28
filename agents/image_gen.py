"""
Image generation executor agent for Personal AI Agent.

Calls the OpenRouter image generation endpoint (FLUX Schnell by default),
decodes the returned base64 image data, and saves the resulting PNG to
the session workspace.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error
    from workspace import WorkspaceManager
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error
    from workspace import WorkspaceManager

logger = logging.getLogger(__name__)

_FILENAME_DESC_MAX: int = 40


def _sanitize_filename_part(text: str, max_len: int = _FILENAME_DESC_MAX) -> str:
    clean = re.sub(r"[^\w\s-]", "", text.lower())
    clean = re.sub(r"[\s-]+", "_", clean).strip("_")
    return clean[:max_len] or "image"


def _decode_base64_image(data_url: str) -> tuple[bytes, str]:
    extension = "png"
    if data_url.startswith("data:"):
        header, _, b64_data = data_url.partition(",")
        mime_match = re.search(r"data:image/([a-zA-Z0-9+]+)", header)
        if mime_match:
            mime_subtype = mime_match.group(1).lower()
            ext_map = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}
            extension = ext_map.get(mime_subtype, mime_subtype)
    else:
        b64_data = data_url

    b64_clean = b64_data.replace("\n", "").replace("\r", "").strip()
    missing_padding = len(b64_clean) % 4
    if missing_padding:
        b64_clean += "=" * (4 - missing_padding)

    raw_bytes = base64.b64decode(b64_clean)
    return raw_bytes, extension


async def execute_image_generation(
    description: str,
    session_id: str = "default",
    model: str | None = None,
) -> dict[str, Any]:
    if not description:
        return {"content": "No description provided for image generation.", "files_created": [], "success": False}

    if not config.OPENROUTER_API_KEY:
        return {"content": "OpenRouter API key is not set.", "files_created": [], "success": False}

    logger.info("Generating image for prompt: %r", description[:80])

    try:
        client = OpenRouterClient()
        response = await client.generate_image(prompt=description, model=model)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Image generation API call failed: {format_error(exc)}"
        return {"content": error_msg, "files_created": [], "success": False}

    images: list[str] = response.get("images", [])
    if not images:
        raw = response.get("_raw", {})
        try:
            content_field = raw["choices"][0]["message"].get("content", "")
            if content_field and "base64" in content_field:
                images = [content_field]
        except (KeyError, IndexError, TypeError):
            pass

    if not images:
        return {"content": "Image generation succeeded but no image data was returned.", "files_created": [], "success": False}

    workspace = WorkspaceManager()
    files_created: list[str] = []
    errors: list[str] = []

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_filename_part(description)

    for idx, image_data in enumerate(images):
        try:
            raw_bytes, extension = _decode_base64_image(image_data)
            suffix = f"_{idx}" if len(images) > 1 else ""
            filename = f"{timestamp}_{slug}{suffix}.{extension}"
            saved_path = workspace.save_file(session_id, filename, raw_bytes)
            files_created.append(str(saved_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to save image {idx}: {format_error(exc)}")

    if not files_created:
        return {"content": f"Image generation failed: {'; '.join(errors) if errors else 'Unknown error.'}", "files_created": [], "success": False}

    paths_str = ", ".join(files_created)
    summary_parts = [f"Image generated successfully. Saved to: {paths_str}"]
    if errors:
        summary_parts.append(f"Warnings: {'; '.join(errors)}")
    return {"content": "\n".join(summary_parts), "files_created": files_created, "success": True}
