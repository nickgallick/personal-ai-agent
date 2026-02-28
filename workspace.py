"""
Workspace and file management module for Personal AI Agent.

Provides per-session workspace directories where files can be saved, read,
listed, and cleaned up. All path handling uses :mod:`pathlib`.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages per-session file workspaces for the personal AI agent.

    Each conversation session receives its own sub-directory under
    ``root``.  Files written during a session (uploaded documents,
    generated outputs, code artefacts) are stored there and can be
    retrieved or listed at any time.
    """

    def __init__(self, root: str | None = None) -> None:
        """
        Initialise the WorkspaceManager.

        Args:
            root: Root directory for all session workspaces. Defaults to
                  ``config.WORKSPACE_ROOT``.
        """
        self.root: Path = Path(root or config.WORKSPACE_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("WorkspaceManager initialised with root: %s", self.root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        """Return the Path for a session directory (does NOT create it)."""
        # Sanitise session_id to prevent path traversal
        safe_id = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in session_id
        )
        return self.root / safe_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_workspace(self, session_id: str) -> Path:
        """
        Return (and create if necessary) the workspace directory for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            :class:`~pathlib.Path` pointing to the session workspace directory.
        """
        workspace = self._session_dir(session_id)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def save_file(
        self,
        session_id: str,
        filename: str,
        content: bytes | str,
    ) -> Path:
        """
        Write *content* to a file in the session workspace.

        Args:
            session_id: Target session.
            filename: Name of the file (basename only; any directory
                      components are stripped for safety).
            content: File contents — either :class:`bytes` (binary) or
                     :class:`str` (UTF-8 encoded automatically).

        Returns:
            Absolute :class:`~pathlib.Path` of the saved file.

        Raises:
            PermissionError: If the workspace directory cannot be written.
            OSError: On other I/O failures.
        """
        workspace = self.get_workspace(session_id)
        # Strip any directory component from the filename
        safe_name = Path(filename).name
        file_path = workspace / safe_name

        try:
            if isinstance(content, str):
                file_path.write_text(content, encoding="utf-8")
            else:
                file_path.write_bytes(content)
            logger.debug("Saved file: %s", file_path)
            return file_path
        except PermissionError as exc:
            logger.error("Permission denied writing %s: %s", file_path, exc)
            raise
        except OSError as exc:
            logger.error("Failed to save file %s: %s", file_path, exc)
            raise

    def save_uploaded_file(
        self,
        session_id: str,
        uploaded_file: Any,
    ) -> Path:
        """
        Persist a Streamlit ``UploadedFile`` object to the session workspace.

        The *uploaded_file* argument is expected to have:
        - ``.name`` — original filename (``str``)
        - ``.read()`` — method returning the raw bytes
        - ``.type`` — MIME type string (informational only)

        Args:
            session_id: Target session.
            uploaded_file: A Streamlit ``UploadedFile`` (or any object with
                           the attributes described above).

        Returns:
            Absolute :class:`~pathlib.Path` of the saved file.

        Raises:
            AttributeError: If *uploaded_file* is missing expected attributes.
            PermissionError: If the workspace directory cannot be written.
            OSError: On other I/O failures.
        """
        try:
            filename: str = uploaded_file.name
            raw_bytes: bytes = uploaded_file.read()
            mime_type: str = getattr(uploaded_file, "type", "application/octet-stream")
            logger.debug(
                "Saving uploaded file '%s' (type=%s, size=%d bytes)",
                filename,
                mime_type,
                len(raw_bytes),
            )
            return self.save_file(session_id, filename, raw_bytes)
        except AttributeError as exc:
            logger.error("Invalid uploaded_file object: %s", exc)
            raise
        except (PermissionError, OSError):
            raise

    def list_files(self, session_id: str) -> list[dict]:
        """
        List all files in a session workspace.

        Args:
            session_id: Target session.

        Returns:
            List of dicts — one per file — each containing:
            ``name`` (str), ``path`` (str), ``size`` (int, bytes),
            and ``modified`` (ISO-8601 str).
            Returns an empty list if the workspace does not exist yet.
        """
        workspace = self._session_dir(session_id)
        if not workspace.exists():
            return []

        results: list[dict] = []
        try:
            for entry in sorted(workspace.iterdir()):
                if entry.is_file():
                    stat = entry.stat()
                    modified = datetime.fromtimestamp(stat.st_mtime).isoformat(
                        sep=" ", timespec="seconds"
                    )
                    results.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "size": stat.st_size,
                            "modified": modified,
                        }
                    )
        except PermissionError as exc:
            logger.error("Permission denied listing workspace %s: %s", workspace, exc)
        except OSError as exc:
            logger.error("Error listing workspace %s: %s", workspace, exc)

        return results

    def get_file_path(self, session_id: str, filename: str) -> Path | None:
        """
        Resolve the full path to a named file in a session workspace.

        Args:
            session_id: Target session.
            filename: Basename of the file to locate.

        Returns:
            Absolute :class:`~pathlib.Path` if the file exists,
            ``None`` otherwise.
        """
        workspace = self._session_dir(session_id)
        safe_name = Path(filename).name
        file_path = workspace / safe_name
        if file_path.is_file():
            return file_path
        logger.debug("File not found: %s", file_path)
        return None

    def read_file(self, session_id: str, filename: str) -> bytes:
        """
        Read and return the raw bytes of a file in the session workspace.

        Args:
            session_id: Target session.
            filename: Basename of the file to read.

        Returns:
            File contents as :class:`bytes`.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
            OSError: On other I/O failures.
        """
        file_path = self.get_file_path(session_id, filename)
        if file_path is None:
            raise FileNotFoundError(
                f"File '{filename}' not found in session workspace '{session_id}'"
            )
        try:
            data = file_path.read_bytes()
            logger.debug("Read %d bytes from %s", len(data), file_path)
            return data
        except PermissionError as exc:
            logger.error("Permission denied reading %s: %s", file_path, exc)
            raise
        except OSError as exc:
            logger.error("Failed to read file %s: %s", file_path, exc)
            raise

    def cleanup_old_workspaces(self, retention_days: int | None = None) -> None:
        """
        Remove session workspace directories that have not been modified
        within *retention_days*.

        Uses the directory's ``mtime`` (modification time) to assess age.

        Args:
            retention_days: Directories older than this many days are
                            deleted. Defaults to ``config.WORKSPACE_RETENTION_DAYS``.
        """
        days = retention_days if retention_days is not None else config.WORKSPACE_RETENTION_DAYS
        cutoff: datetime = datetime.now() - timedelta(days=days)
        logger.info(
            "Cleaning up workspaces older than %d day(s) (cutoff: %s)",
            days,
            cutoff.isoformat(sep=" ", timespec="seconds"),
        )

        if not self.root.exists():
            logger.debug("Workspace root does not exist; nothing to clean.")
            return

        removed: list[str] = []
        errors: list[str] = []

        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(entry)
                    removed.append(entry.name)
                    logger.debug("Removed old workspace: %s (mtime=%s)", entry.name, mtime)
            except PermissionError as exc:
                msg = f"{entry.name}: permission denied — {exc}"
                logger.warning("Could not remove workspace %s: %s", entry.name, exc)
                errors.append(msg)
            except OSError as exc:
                msg = f"{entry.name}: {exc}"
                logger.warning("Error removing workspace %s: %s", entry.name, exc)
                errors.append(msg)

        if removed:
            logger.info("Removed %d old workspace(s): %s", len(removed), ", ".join(removed))
        else:
            logger.info("No workspaces met the cleanup criteria.")

        if errors:
            logger.warning("Cleanup encountered %d error(s): %s", len(errors), "; ".join(errors))

    def get_workspace_size(self, session_id: str) -> int:
        """
        Calculate the total size (in bytes) of all files in a session workspace.

        Args:
            session_id: Target session.

        Returns:
            Total size in bytes. Returns ``0`` if the workspace does not
            exist or contains no files.
        """
        workspace = self._session_dir(session_id)
        if not workspace.exists():
            return 0

        total: int = 0
        try:
            for entry in workspace.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError as exc:
                        logger.warning("Could not stat %s: %s", entry, exc)
        except PermissionError as exc:
            logger.error("Permission denied sizing workspace %s: %s", workspace, exc)
        except OSError as exc:
            logger.error("Error sizing workspace %s: %s", workspace, exc)

        return total
