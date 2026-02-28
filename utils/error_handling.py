"""
Error handling utilities for the Personal AI Agent.

Provides custom exception hierarchy, an async retry decorator, a safe
coroutine executor, and a user-friendly error formatter.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AgentError(Exception):
    """Base exception for all Personal AI Agent errors."""


class APIError(AgentError):
    """Raised when an external API call fails."""

    def __init__(self, message: str, status_code: int | None = None, service: str = "unknown") -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.service: str = service

    def __repr__(self) -> str:
        return f"APIError(service={self.service!r}, status_code={self.status_code}, message={str(self)!r})"


class ExecutionError(AgentError):
    """Raised when a code-execution step fails."""

    def __init__(self, message: str, stderr: str = "", exit_code: int | None = None) -> None:
        super().__init__(message)
        self.stderr: str = stderr
        self.exit_code: int | None = exit_code

    def __repr__(self) -> str:
        return f"ExecutionError(exit_code={self.exit_code}, stderr={self.stderr!r}, message={str(self)!r})"


class PlanningError(AgentError):
    """Raised when the agent fails to produce or parse a valid task plan."""


class BrowsingError(AgentError):
    """Raised when a web-browsing or page-extraction step fails."""


async def retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> T:
    last_exc: Exception | None = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "retry_async: attempt %d/%d failed (%s: %s). Retrying in %.1fs...",
                    attempt + 1, total_attempts, type(exc).__name__, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("retry_async: all %d attempts exhausted. Last error: %s: %s", total_attempts, type(exc).__name__, exc)

    raise last_exc  # type: ignore[misc]


async def safe_execute(
    coro: Coroutine[Any, Any, T],
    fallback: Any = None,
    error_msg: str = "Operation failed",
) -> tuple[T | Any, str | None]:
    try:
        result = await coro
        return result, None
    except Exception as exc:  # noqa: BLE001
        error_string = f"{error_msg}: {format_error(exc)}"
        logger.error("safe_execute caught: %s", error_string, exc_info=exc)
        return fallback, error_string


def format_error(error: Exception) -> str:
    base_msg = str(error) or repr(error)

    if isinstance(error, APIError):
        parts = [f"API error ({error.service})"]
        if error.status_code is not None:
            parts.append(f"[HTTP {error.status_code}]")
        parts.append(base_msg)
        return " ".join(parts)

    if isinstance(error, ExecutionError):
        lines = [f"Code execution failed: {base_msg}"]
        if error.exit_code is not None:
            lines.append(f"  Exit code: {error.exit_code}")
        if error.stderr:
            stderr_tail = "\n".join(error.stderr.splitlines()[-10:])
            lines.append(f"  Stderr:\n{stderr_tail}")
        return "\n".join(lines)

    if isinstance(error, PlanningError):
        return f"Planning failed: {base_msg}"

    if isinstance(error, BrowsingError):
        return f"Browsing failed: {base_msg}"

    if isinstance(error, AgentError):
        return f"Agent error: {base_msg}"

    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    condensed = "".join(tb_lines[-3:]).strip()
    return f"{type(error).__name__}: {base_msg}\n{condensed}"
