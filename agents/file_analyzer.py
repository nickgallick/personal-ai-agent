"""
File analysis agent for Personal AI Agent.

Handles extraction of content from various file types (PDF, CSV, XLSX,
images, code, text/JSON/YAML/Markdown) and sends the extracted content
to the OpenRouter reasoning model for analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config / utils import — works from project root or agents/ sub-directory
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
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTENT_CHARS: int = 15_000  # truncation limit per-file and combined
_SYSTEM_PROMPT: str = (
    "You are a file analysis expert. Analyze the provided file contents and "
    "answer the user's questions or provide a comprehensive summary. "
    "Use markdown formatting."
)

# ---------------------------------------------------------------------------
# File-type extension groups
# ---------------------------------------------------------------------------

_PDF_EXTS: frozenset[str] = frozenset({".pdf"})
_CSV_EXTS: frozenset[str] = frozenset({".csv"})
_EXCEL_EXTS: frozenset[str] = frozenset({".xlsx", ".xls"})
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif"})
_CODE_EXTS: frozenset[str] = frozenset({".py", ".js", ".html", ".css", ".ts", ".jsx", ".tsx"})
_TEXT_EXTS: frozenset[str] = frozenset({".txt", ".json", ".md", ".yaml", ".yml"})


# ---------------------------------------------------------------------------
# Per-type extraction helpers
# ---------------------------------------------------------------------------


def _extract_pdf(file_path: Path) -> str:
    try:
        import PyPDF2  # type: ignore[import]
    except ImportError:
        logger.warning("PyPDF2 is not installed; cannot extract PDF text.")
        return "[PyPDF2 not installed — cannot extract PDF text]"

    try:
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            pages: list[str] = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    pages.append(f"--- Page {page_num} ---\n{text}")
                except Exception as exc:  # noqa: BLE001
                    pages.append(f"--- Page {page_num} --- [extraction error: {exc}]")
            return "\n".join(pages)
    except PyPDF2.errors.PdfReadError as exc:
        return f"[Could not read PDF — file may be corrupted or encrypted: {exc}]"
    except Exception as exc:  # noqa: BLE001
        return f"[Unexpected error reading PDF: {exc}]"


def _extract_csv(file_path: Path) -> str:
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError:
        return "[pandas not installed — cannot parse CSV]"

    try:
        df = pd.read_csv(file_path)
        parts = [
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"\nColumns:\n{list(df.columns)}",
            f"\nData Types:\n{df.dtypes.to_string()}",
            f"\nFirst 5 rows:\n{df.head(5).to_string()}",
            f"\nDescriptive Statistics:\n{df.describe(include='all').to_string()}",
        ]
        return "\n".join(parts)
    except pd.errors.EmptyDataError:
        return "[CSV file is empty or contains no parseable data]"
    except Exception as exc:  # noqa: BLE001
        return f"[Unexpected error reading CSV: {exc}]"


def _extract_excel(file_path: Path) -> str:
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError:
        return "[pandas not installed — cannot parse Excel]"

    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
        sheet_summaries: list[str] = []
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                parts = [
                    f"=== Sheet: {sheet_name} ===",
                    f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
                    f"Columns: {list(df.columns)}",
                    f"Data Types:\n{df.dtypes.to_string()}",
                    f"First 5 rows:\n{df.head(5).to_string()}",
                    f"Descriptive Statistics:\n{df.describe(include='all').to_string()}",
                ]
                sheet_summaries.append("\n".join(parts))
            except Exception as exc:  # noqa: BLE001
                sheet_summaries.append(f"=== Sheet: {sheet_name} === [parse error: {exc}]")
        return "\n\n".join(sheet_summaries)
    except Exception as exc:  # noqa: BLE001
        return f"[Unexpected error reading Excel file: {exc}]"


def _describe_image(file_path: Path) -> str:
    try:
        size_bytes = file_path.stat().st_size
        size_kb = size_bytes / 1024
        return (
            f"[Image file]\n"
            f"Filename: {file_path.name}\n"
            f"Size: {size_kb:.1f} KB ({size_bytes} bytes)\n"
            f"Note: Full visual analysis requires a vision-capable model."
        )
    except OSError as exc:
        return f"[Image file: {file_path.name}] [Could not stat file: {exc}]"


def _read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding="latin-1")
        except Exception as exc:  # noqa: BLE001
            return f"[Could not decode file as text: {exc}]"
    except Exception as exc:  # noqa: BLE001
        return f"[Error reading file: {exc}]"


def _read_unknown_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        size_bytes = file_path.stat().st_size if file_path.exists() else 0
        return f"[Binary file: {file_path.name}] (size: {size_bytes} bytes — cannot display as text)"
    except Exception as exc:  # noqa: BLE001
        return f"[Could not read file {file_path.name}: {exc}]"


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------


def _extract_file_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in _PDF_EXTS:
        raw = _extract_pdf(file_path)
    elif suffix in _CSV_EXTS:
        raw = _extract_csv(file_path)
    elif suffix in _EXCEL_EXTS:
        raw = _extract_excel(file_path)
    elif suffix in _IMAGE_EXTS:
        raw = _describe_image(file_path)
    elif suffix in _CODE_EXTS:
        raw = _read_text_file(file_path)
    elif suffix in _TEXT_EXTS:
        raw = _read_text_file(file_path)
    else:
        raw = _read_unknown_file(file_path)

    if len(raw) > _MAX_CONTENT_CHARS:
        truncation_note = (
            f"\n\n[... content truncated at {_MAX_CONTENT_CHARS} characters "
            f"(original length: {len(raw)} characters) ...]"
        )
        raw = raw[:_MAX_CONTENT_CHARS] + truncation_note

    return raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_file_analysis(
    description: str,
    file_paths: list[str] | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    logger.info(
        "execute_file_analysis called | session=%s | files=%s | description=%r",
        session_id,
        file_paths,
        description[:100],
    )

    resolved_paths: list[str] = file_paths or []
    files_analyzed: list[str] = []
    file_sections: list[str] = []

    for raw_path in resolved_paths:
        fp = Path(raw_path)
        if not fp.exists():
            file_sections.append(f"### File: {fp.name}\n[File not found at path: {raw_path}]")
            continue

        try:
            content = _extract_file_content(fp)
            file_sections.append(f"### File: {fp.name}\n{content}")
            files_analyzed.append(str(fp))
        except Exception as exc:  # noqa: BLE001
            file_sections.append(f"### File: {fp.name}\n[Extraction failed: {exc}]")

    combined_files_block = "\n\n".join(file_sections) if file_sections else "(no files provided)"

    if len(combined_files_block) > _MAX_CONTENT_CHARS:
        combined_files_block = (
            combined_files_block[:_MAX_CONTENT_CHARS]
            + f"\n\n[... combined file content truncated at {_MAX_CONTENT_CHARS} characters ...]"
        )

    user_content = (
        f"## User Request\n{description}\n\n"
        f"## Extracted File Contents\n{combined_files_block}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        async with OpenRouterClient() as client:
            analysis_text: str = await client.reason(messages)

        return {
            "content": analysis_text,
            "files_analyzed": files_analyzed,
            "success": True,
        }

    except Exception as exc:  # noqa: BLE001
        error_msg = f"File analysis failed: {type(exc).__name__}: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            "content": error_msg,
            "files_analyzed": [],
            "success": False,
        }
