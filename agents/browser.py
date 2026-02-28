"""
Browser executor agent for Personal AI Agent.

Provides an async function that launches a headless Chromium browser via
Playwright, navigates to one or more URLs, and extracts textual content.
If no URLs are supplied, the fast OpenRouter model is consulted to derive
relevant URLs from the task description.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-level imports — handle running from project root or agents/ dir
# ---------------------------------------------------------------------------
try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import BrowsingError, format_error
    from workspace import WorkspaceManager
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import BrowsingError, format_error
    from workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Maximum characters of page text to return per URL
_MAX_CONTENT_CHARS: int = 8_000
# Navigation timeout in milliseconds
_NAV_TIMEOUT_MS: int = 30_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    """Return a filesystem-safe version of *text* truncated to *max_len* chars."""
    clean = re.sub(r"[^\w\s-]", "", text.lower())
    clean = re.sub(r"[\s-]+", "_", clean).strip("_")
    return clean[:max_len] or "page"


async def _extract_urls_from_description(description: str) -> list[str]:
    """
    Use the fast OpenRouter model to extract or suggest URLs for the task.

    Parameters
    ----------
    description:
        Natural-language description of what to browse.

    Returns
    -------
    list[str]
        Ordered list of URLs to visit (may be empty if the model cannot
        determine any).
    """
    prompt = (
        "You are a URL extractor. Given a task description, return a JSON array "
        "of URLs to visit to complete the task. If explicit URLs are mentioned, "
        "extract them. If not, suggest the most relevant URLs (e.g. official docs, "
        "Wikipedia, authoritative sources). Return ONLY a valid JSON array of "
        "strings — no markdown, no explanation. Example: "
        '["https://example.com", "https://docs.example.com"]'
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": description},
    ]
    try:
        client = OpenRouterClient()
        reply = await client.fast(messages)
        # Parse the JSON array from the reply
        # Strip markdown code fences if present
        reply_clean = reply.strip()
        if reply_clean.startswith("```"):
            reply_clean = re.sub(r"^```[a-zA-Z]*\n?", "", reply_clean)
            reply_clean = re.sub(r"\n?```$", "", reply_clean).strip()
        urls = json.loads(reply_clean)
        if isinstance(urls, list):
            return [str(u) for u in urls if isinstance(u, str) and u.startswith("http")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("URL extraction failed: %s", exc)
    return []


def _extract_urls_from_text(text: str) -> list[str]:
    """Return all http(s) URLs found in *text* via regex."""
    pattern = r"https?://[^\s\"'<>]+"
    return re.findall(pattern, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_browse(
    description: str,
    urls: list[str] | None = None,
    context: str = "",
    session_id: str = "default",
    save_screenshots: bool = False,
) -> dict[str, Any]:
    """
    Browse one or more web pages and return their text content.

    Parameters
    ----------
    description:
        Human-readable description of the browsing task.  Also used to
        extract or derive URLs when none are explicitly provided.
    urls:
        Explicit list of URLs to visit.  When ``None`` or empty, URLs are
        inferred from *description* via regex and, if still empty, by
        querying the fast LLM.
    context:
        Optional extra context passed alongside *description* to the LLM
        when URLs need to be inferred.
    session_id:
        Session identifier used for screenshot storage.
    save_screenshots:
        When ``True``, save a PNG screenshot for each page visited.

    Returns
    -------
    dict
        ``{"content": str, "urls_visited": list[str],
           "screenshots": list[str], "success": bool}``
    """
    # ------------------------------------------------------------------
    # Resolve which URLs to visit
    # ------------------------------------------------------------------
    resolved_urls: list[str] = list(urls or [])

    # Also try to pull explicit URLs from the description itself
    if not resolved_urls:
        resolved_urls = _extract_urls_from_text(description)

    if not resolved_urls and context:
        resolved_urls = _extract_urls_from_text(context)

    # Fall back to LLM-suggested URLs
    if not resolved_urls:
        full_desc = description if not context else f"{description}\n\nContext: {context}"
        logger.debug("No URLs supplied — asking fast model to suggest URLs")
        resolved_urls = await _extract_urls_from_description(full_desc)

    if not resolved_urls:
        return {
            "content": (
                "No URLs could be determined from the task description. "
                "Please provide explicit URLs or a more specific description."
            ),
            "urls_visited": [],
            "screenshots": [],
            "success": False,
        }

    # ------------------------------------------------------------------
    # Import Playwright (graceful failure when not installed)
    # ------------------------------------------------------------------
    try:
        from playwright.async_api import (
            async_playwright,
            TimeoutError as PlaywrightTimeout,
            Error as PlaywrightError,
        )
    except ImportError:
        msg = (
            "Playwright is not installed. Run: "
            "pip install playwright && playwright install chromium"
        )
        logger.error(msg)
        return {
            "content": msg,
            "urls_visited": [],
            "screenshots": [],
            "success": False,
        }

    workspace = WorkspaceManager()
    screenshots: list[str] = []
    pages_content: list[str] = []
    urls_visited: list[str] = []

    logger.info("Launching headless Chromium for %d URL(s)", len(resolved_urls))

    playwright_ctx = async_playwright()
    pw = await playwright_ctx.__aenter__()
    browser = None
    try:
        browser = await pw.chromium.launch(headless=True)
        bcontext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=True,
        )

        for url in resolved_urls:
            page_text = ""
            page_title = url
            try:
                logger.debug("Navigating to %s", url)
                page = await bcontext.new_page()
                try:
                    response = await page.goto(
                        url,
                        timeout=_NAV_TIMEOUT_MS,
                        wait_until="domcontentloaded",
                    )
                    # Best-effort wait for network to settle (non-fatal)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5_000)
                    except Exception:  # noqa: BLE001
                        pass

                    page_title = await page.title() or url

                    # Extract main text from body
                    try:
                        page_text = await page.inner_text("body")
                    except Exception:  # noqa: BLE001
                        # Fallback: evaluate JS to get innerText
                        try:
                            page_text = await page.evaluate("() => document.body.innerText")
                        except Exception:  # noqa: BLE001
                            page_text = ""

                    # Normalise whitespace and truncate
                    page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
                    if len(page_text) > _MAX_CONTENT_CHARS:
                        page_text = page_text[:_MAX_CONTENT_CHARS] + "\n\n[... content truncated ...]"

                    status = response.status if response else "unknown"
                    pages_content.append(
                        f"=== {page_title} ({url}) [HTTP {status}] ===\n{page_text}"
                    )
                    urls_visited.append(url)
                    logger.debug("Fetched %s — %d chars", url, len(page_text))

                    # Screenshot (optional)
                    if save_screenshots:
                        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        slug = _sanitize_filename(page_title)
                        screenshot_name = f"screenshot_{ts}_{slug}.png"
                        screenshot_bytes = await page.screenshot(full_page=False)
                        saved_path = workspace.save_file(
                            session_id, screenshot_name, screenshot_bytes
                        )
                        screenshots.append(str(saved_path))
                        logger.debug("Screenshot saved: %s", saved_path)

                except PlaywrightTimeout as exc:
                    msg = f"Timeout navigating to {url}: {exc}"
                    logger.warning(msg)
                    pages_content.append(f"=== {url} [ERROR: timeout] ===\n{msg}")

                except PlaywrightError as exc:
                    err_str = str(exc)
                    # Classify common errors for clearer messages
                    if "net::ERR_NAME_NOT_RESOLVED" in err_str:
                        label = "DNS resolution failed"
                    elif "net::ERR_CONNECTION_REFUSED" in err_str:
                        label = "connection refused"
                    elif "SSL" in err_str or "certificate" in err_str.lower():
                        label = "SSL/certificate error"
                    else:
                        label = "navigation error"
                    msg = f"{label} for {url}: {exc}"
                    logger.warning(msg)
                    pages_content.append(f"=== {url} [ERROR: {label}] ===\n{msg}")

                finally:
                    await page.close()

            except Exception as exc:  # noqa: BLE001
                msg = f"Unexpected error processing {url}: {format_error(exc)}"
                logger.error(msg, exc_info=True)
                pages_content.append(f"=== {url} [ERROR] ===\n{msg}")

        await bcontext.close()

    except Exception as exc:  # noqa: BLE001
        error_msg = f"Browser launch or context error: {format_error(exc)}"
        logger.error(error_msg, exc_info=True)
        return {
            "content": error_msg,
            "urls_visited": urls_visited,
            "screenshots": screenshots,
            "success": False,
        }
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            await playwright_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    combined_content = "\n\n".join(pages_content)
    success = len(urls_visited) > 0

    return {
        "content": combined_content,
        "urls_visited": urls_visited,
        "screenshots": screenshots,
        "success": success,
    }
