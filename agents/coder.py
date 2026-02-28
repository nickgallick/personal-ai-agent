"""
Code generation and execution agent with self-correction loop.

Generates Python code for a given task using the coding LLM, saves it to the
session workspace, executes it in a subprocess, and automatically retries with
LLM-assisted corrections if the code fails.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import ExecutionError, format_error
    from workspace import WorkspaceManager
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import ExecutionError, format_error
    from workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CODING_SYSTEM_PROMPT: str = (
    "You are an expert programmer. Write complete, runnable Python code to "
    "accomplish the task. Output ONLY the Python code, no explanations. "
    "Include all necessary imports."
)

_FIX_PROMPT_TEMPLATE: str = (
    "The code produced this error:\n{stderr}\n\n"
    "Fix the code and return the complete corrected version. "
    "Output ONLY the corrected Python code."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from *text* and return the raw code.

    Handles both ````python ... ``` `` and ```` ``` ... ``` `` forms.
    Falls back to returning the original text unchanged.
    """
    pattern = r"```(?:python|py|Python)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fences found; return as-is (model may have obeyed the instruction)
    return text.strip()


def _snapshot_files(workspace: Path) -> set[str]:
    """Return the set of filenames currently present in *workspace*."""
    if not workspace.exists():
        return set()
    return {entry.name for entry in workspace.iterdir() if entry.is_file()}


def _run_code(script_path: Path, timeout: int) -> subprocess.CompletedProcess:
    """
    Execute *script_path* with Python 3 in a subprocess.

    Parameters
    ----------
    script_path:
        Absolute path to the Python script to run.
    timeout:
        Maximum wall-clock seconds to allow.

    Returns
    -------
    subprocess.CompletedProcess
        Always returns a result; ``TimeoutExpired`` is caught and re-raised
        as a ``subprocess.CompletedProcess`` with a non-zero returncode so
        the caller does not need special-case handling.
    """
    try:
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            cwd=str(script_path.parent),  # run from the workspace dir
        )
    except subprocess.TimeoutExpired:
        logger.warning("Code execution timed out after %ds: %s", timeout, script_path)
        # Re-raise wrapped so the retry loop can surface it clearly
        raise ExecutionError(
            f"Code execution timed out after {timeout} seconds.",
            stderr="TimeoutExpired",
            exit_code=-1,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_code(
    description: str,
    context: str = "",
    session_id: str = "default",
) -> dict[str, Any]:
    """
    Generate Python code for *description*, execute it, and self-correct on failure.

    Parameters
    ----------
    description:
        Natural-language task description passed to the coding LLM.
    context:
        Optional prior context (e.g. search results, earlier subtask outputs)
        to include alongside the task description.
    session_id:
        Session identifier used to scope the workspace directory where the
        generated script and any output files are stored.

    Returns
    -------
    dict
        On success::

            {
                "content":       str,        # stdout from the executed code
                "files_created": list[str],  # basenames of new files created
                "code":          str,        # final version of the code
                "success":       True,
                "error":         None,
            }

        On failure (all retries exhausted)::

            {
                "content":       str,        # last stdout (may be empty)
                "files_created": list[str],  # any files created before failure
                "code":          str,        # last version of the code attempted
                "success":       False,
                "error":         str,        # description of the final error
            }
    """
    logger.debug(
        "execute_code: session=%r description=%r context_len=%d",
        session_id,
        description[:80],
        len(context),
    )

    # ------------------------------------------------------------------
    # Set up workspace
    # ------------------------------------------------------------------
    wm = WorkspaceManager()
    workspace: Path = wm.get_workspace(session_id)

    # Record baseline file listing before we do anything
    files_before: set[str] = _snapshot_files(workspace)

    # Unique script filename based on a simple timestamp to avoid collisions
    script_name = f"task_{int(time.time() * 1000)}.py"
    script_path: Path = workspace / script_name

    # ------------------------------------------------------------------
    # Step 1 — Generate initial code
    # ------------------------------------------------------------------
    client = OpenRouterClient()

    user_prompt_parts: list[str] = []
    if context and context.strip():
        user_prompt_parts.append(f"Context:\n{context.strip()}\n")
    user_prompt_parts.append(description)
    user_prompt = "\n".join(user_prompt_parts)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _CODING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_code_response = await client.code(messages)
    except Exception as exc:
        error_msg = format_error(exc)
        logger.error("execute_code: initial LLM call failed: %s", error_msg)
        return {
            "content": "",
            "files_created": [],
            "code": "",
            "success": False,
            "error": f"Code generation failed: {error_msg}",
        }

    code: str = _strip_code_fences(raw_code_response)
    logger.debug("execute_code: initial code generated (%d chars)", len(code))

    # Write the initial code to disk
    try:
        script_path.write_text(code, encoding="utf-8")
    except OSError as exc:
        error_msg = format_error(exc)
        logger.error("execute_code: failed to write script: %s", error_msg)
        return {
            "content": "",
            "files_created": [],
            "code": code,
            "success": False,
            "error": f"Failed to write generated code to workspace: {error_msg}",
        }

    # ------------------------------------------------------------------
    # Step 2 — Execute with self-correction loop
    # ------------------------------------------------------------------
    stdout: str = ""
    stderr: str = ""
    last_error: str | None = None
    succeeded = False

    for attempt in range(config.MAX_CODE_RETRIES + 1):
        is_retry = attempt > 0
        logger.debug(
            "execute_code: running attempt %d/%d for %s",
            attempt + 1,
            config.MAX_CODE_RETRIES + 1,
            script_path.name,
        )

        try:
            result = _run_code(script_path, timeout=config.CODE_EXECUTION_TIMEOUT)
        except ExecutionError as exc:
            # Timeout wrapped as ExecutionError
            stderr = exc.stderr
            last_error = str(exc)
            logger.warning("execute_code: attempt %d timed out.", attempt + 1)
        else:
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode == 0 and not _has_fatal_error(stderr):
                # Success
                succeeded = True
                logger.info(
                    "execute_code: succeeded on attempt %d. stdout_len=%d",
                    attempt + 1,
                    len(stdout),
                )
                break
            else:
                last_error = (
                    f"Exit code {result.returncode}. "
                    f"Stderr: {stderr[:500]}"
                )
                logger.warning(
                    "execute_code: attempt %d failed (exit=%d). stderr=%r",
                    attempt + 1,
                    result.returncode,
                    stderr[:200],
                )

        # If this was the last attempt, do not request another fix
        if attempt >= config.MAX_CODE_RETRIES:
            logger.error(
                "execute_code: all %d attempt(s) exhausted for session %r.",
                config.MAX_CODE_RETRIES + 1,
                session_id,
            )
            break

        # ------------------------------------------------------------------
        # Self-correction: ask the LLM to fix the error
        # ------------------------------------------------------------------
        fix_prompt = _FIX_PROMPT_TEMPLATE.format(
            stderr=stderr[-2000:] if stderr else last_error or "unknown error"
        )

        # Append the broken code and error to the conversation so the model
        # has full context for the fix.
        messages.append({"role": "assistant", "content": _code_block(code)})
        messages.append({"role": "user", "content": fix_prompt})

        try:
            raw_fix_response = await client.code(messages)
        except Exception as exc:
            error_msg = format_error(exc)
            logger.error("execute_code: fix LLM call failed on attempt %d: %s", attempt + 1, error_msg)
            last_error = f"Fix generation failed: {error_msg}"
            break

        code = _strip_code_fences(raw_fix_response)
        logger.debug(
            "execute_code: received corrected code (%d chars) for attempt %d",
            len(code),
            attempt + 2,
        )

        # Overwrite the script with the corrected version
        try:
            script_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            error_msg = format_error(exc)
            logger.error("execute_code: failed to overwrite script: %s", error_msg)
            last_error = f"Failed to save corrected code: {error_msg}"
            break

    # ------------------------------------------------------------------
    # Step 3 — Determine newly created files
    # ------------------------------------------------------------------
    files_after: set[str] = _snapshot_files(workspace)
    new_files: list[str] = sorted(
        files_after - files_before - {script_name}
    )
    logger.debug("execute_code: new files created: %s", new_files)

    # ------------------------------------------------------------------
    # Step 4 — Build and return result
    # ------------------------------------------------------------------
    if succeeded:
        content = stdout if stdout.strip() else "Code executed successfully (no stdout output)."
        return {
            "content": content,
            "files_created": new_files,
            "code": code,
            "success": True,
            "error": None,
        }
    else:
        return {
            "content": stdout,
            "files_created": new_files,
            "code": code,
            "success": False,
            "error": last_error or "Code execution failed after all retries.",
        }


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _has_fatal_error(stderr: str) -> bool:
    """
    Return ``True`` if *stderr* contains an unambiguous Python runtime error.

    Some libraries (e.g. matplotlib, TensorFlow) print warnings to stderr
    even on a successful run, so we only flag lines that look like proper
    Python tracebacks or error lines.
    """
    if not stderr:
        return False
    # Look for canonical Python exception indicators
    fatal_markers = (
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "SyntaxError",
        "IndentationError",
        "NameError",
        "TypeError",
        "ValueError",
        "ImportError",
        "ModuleNotFoundError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "RuntimeError",
        "OSError",
        "IOError",
        "FileNotFoundError",
        "PermissionError",
        "ZeroDivisionError",
        "RecursionError",
        "MemoryError",
        "SystemExit",
    )
    for marker in fatal_markers:
        if marker in stderr:
            return True
    return False


def _code_block(code: str) -> str:
    """Wrap *code* in a markdown Python code fence for LLM message context."""
    return f"```python\n{code}\n```"
