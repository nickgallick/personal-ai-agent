"""
Persistent memory system for Personal AI Agent.

Provides SQLite-backed storage for conversation history, user preferences,
and task execution history. All database access is synchronous (sqlite3)
since SQLite does not benefit from async I/O.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory set, auto-committing on success."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Manages persistent memory for the personal AI agent.

    Stores conversation history, user preferences, and task execution records
    in a local SQLite database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialise the MemoryManager.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                     ``config.DATABASE_PATH``.
        """
        self.db_path: str = db_path or config.DATABASE_PATH

        # Ensure the parent directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._initialise_db()
        logger.info("MemoryManager initialised with database at %s", self.db_path)

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _initialise_db(self) -> None:
        """Create tables and indexes if they do not already exist."""
        ddl_statements = [
            # Conversations table
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp TEXT    DEFAULT CURRENT_TIMESTAMP,
                citations TEXT
            )
            """,
            # User preferences table
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT UNIQUE NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Task history table
            """
            CREATE TABLE IF NOT EXISTS task_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id        TEXT UNIQUE NOT NULL,
                description    TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'running',
                result_summary TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at   TEXT
            )
            """,
            # Indexes
            "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences (key)",
        ]
        with _get_connection(self.db_path) as conn:
            for stmt in ddl_statements:
                conn.execute(stmt)
        logger.debug("Database schema verified/created.")

    # ------------------------------------------------------------------
    # Conversation methods
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ) -> None:
        """
        Persist a chat message for a given session.

        Args:
            session_id: Unique identifier for the conversation session.
            role: Message role — e.g. ``"user"`` or ``"assistant"``.
            content: The message text.
            citations: Optional list of citation URLs or references.
        """
        citations_json: str | None = json.dumps(citations) if citations is not None else None
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (session_id, role, content, citations)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, content, citations_json),
                )
            logger.debug("Saved message: session=%s role=%s", session_id, role)
        except sqlite3.Error as exc:
            logger.error("Failed to save message: %s", exc)
            raise

    def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        """
        Retrieve the most recent messages for a session.

        Args:
            session_id: The session to query.
            limit: Maximum number of messages to return (default 20).

        Returns:
            List of message dicts ordered oldest-first, each containing
            ``role``, ``content``, ``citations``, and ``timestamp``.
        """
        try:
            with _get_connection(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, citations, timestamp
                    FROM (
                        SELECT role, content, citations, timestamp, id
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
            messages = []
            for row in rows:
                citations_raw = row["citations"]
                citations: list[str] | None = (
                    json.loads(citations_raw) if citations_raw else None
                )
                messages.append(
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "citations": citations,
                        "timestamp": row["timestamp"],
                    }
                )
            return messages
        except sqlite3.Error as exc:
            logger.error("Failed to get recent messages: %s", exc)
            return []

    def get_all_sessions(self) -> list[dict]:
        """
        Return a summary of all distinct conversation sessions.

        Returns:
            List of dicts ordered by most-recent first, each containing
            ``session_id``, ``last_timestamp``, and ``preview`` (truncated
            last message content).
        """
        try:
            with _get_connection(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT
                        session_id,
                        MAX(timestamp) AS last_timestamp,
                        content        AS last_content
                    FROM conversations
                    GROUP BY session_id
                    ORDER BY last_timestamp DESC
                    """
                ).fetchall()
            sessions = []
            for row in rows:
                preview = (row["last_content"] or "")[:120]
                if len(row["last_content"] or "") > 120:
                    preview += "…"
                sessions.append(
                    {
                        "session_id": row["session_id"],
                        "last_timestamp": row["last_timestamp"],
                        "preview": preview,
                    }
                )
            return sessions
        except sqlite3.Error as exc:
            logger.error("Failed to get all sessions: %s", exc)
            return []

    def delete_session(self, session_id: str) -> None:
        """
        Delete all messages belonging to a session.

        Args:
            session_id: The session whose messages should be removed.
        """
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
            logger.info("Deleted session: %s", session_id)
        except sqlite3.Error as exc:
            logger.error("Failed to delete session %s: %s", session_id, exc)
            raise

    # ------------------------------------------------------------------
    # Preference methods
    # ------------------------------------------------------------------

    # Patterns that indicate the user is sharing a personal preference.
    _PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
        # "my name is Alice" -> key="name", value="Alice"
        (re.compile(r"my name is\s+(.+)", re.IGNORECASE), "name"),
        # "remember that I prefer dark mode" -> key="remember", value rest
        (re.compile(r"remember that\s+(.+)", re.IGNORECASE), "remember"),
        # "always use metric units" -> key="always", value rest
        (re.compile(r"always\s+(.+)", re.IGNORECASE), "always"),
        # "I prefer Python over JavaScript" -> key="prefer", value rest
        (re.compile(r"(?:i\s+)?prefer\s+(.+)", re.IGNORECASE), "prefer"),
        # "my timezone is UTC+2" -> key="timezone", value="UTC+2"
        (re.compile(r"my timezone is\s+(.+)", re.IGNORECASE), "timezone"),
        # "my location is London" -> key="location", value="London"
        (re.compile(r"my location is\s+(.+)", re.IGNORECASE), "location"),
        # "call me Bob" -> key="nickname", value="Bob"
        (re.compile(r"call me\s+(.+)", re.IGNORECASE), "nickname"),
        # "I am a software engineer" -> key="occupation", value rest
        (re.compile(r"i am (?:a |an )?(.+)", re.IGNORECASE), "occupation"),
    ]

    def detect_and_store_preference(self, content: str) -> None:
        """
        Scan a message for preference-related patterns and persist matches.

        Uses simple regex heuristics to identify statements such as
        ``"my name is …"``, ``"always …"``, ``"prefer …"``, etc.

        Args:
            content: Raw message text to analyse.
        """
        stored: list[tuple[str, str]] = []
        for pattern, base_key in self._PREFERENCE_PATTERNS:
            match = pattern.search(content)
            if match:
                raw_value = match.group(1).strip().rstrip(".")
                # Deduplicate: use a compound key when the base key is generic
                if base_key in {"remember", "always", "prefer", "occupation"}:
                    # Use first 5 words of value as a slug for the key
                    slug = "_".join(raw_value.lower().split()[:5])
                    slug = re.sub(r"[^a-z0-9_]", "", slug)
                    key = f"{base_key}_{slug}" if slug else base_key
                else:
                    key = base_key
                self.set_preference(key, raw_value)
                stored.append((key, raw_value))

        if stored:
            logger.debug("Detected preferences: %s", stored)

    def set_preference(self, key: str, value: str) -> None:
        """
        Directly insert or replace a user preference.

        Args:
            key: Preference identifier.
            value: Preference value.
        """
        now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO user_preferences (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                   updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )
            logger.debug("Set preference: %s = %s", key, value)
        except sqlite3.Error as exc:
            logger.error("Failed to set preference %s: %s", key, exc)
            raise

    def get_preference(self, key: str) -> str | None:
        """
        Retrieve a single preference value.

        Args:
            key: Preference identifier.

        Returns:
            The stored value, or ``None`` if the key does not exist.
        """
        try:
            with _get_connection(self.db_path) as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE key = ?", (key,)
                ).fetchone()
            return row["value"] if row else None
        except sqlite3.Error as exc:
            logger.error("Failed to get preference %s: %s", key, exc)
            return None

    def get_all_preferences(self) -> dict[str, str]:
        """
        Return all stored user preferences.

        Returns:
            Dict mapping preference keys to their values.
        """
        try:
            with _get_connection(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT key, value FROM user_preferences ORDER BY key"
                ).fetchall()
            return {row["key"]: row["value"] for row in rows}
        except sqlite3.Error as exc:
            logger.error("Failed to get all preferences: %s", exc)
            return {}

    def delete_preference(self, key: str) -> None:
        """
        Remove a preference entry by key.

        Args:
            key: Preference identifier to delete.
        """
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM user_preferences WHERE key = ?", (key,)
                )
            logger.debug("Deleted preference: %s", key)
        except sqlite3.Error as exc:
            logger.error("Failed to delete preference %s: %s", key, exc)
            raise

    # ------------------------------------------------------------------
    # Task history methods
    # ------------------------------------------------------------------

    def save_task(
        self,
        task_id: str,
        description: str,
        status: str = "running",
    ) -> None:
        """
        Create a new task history entry.

        Args:
            task_id: Unique identifier for the task.
            description: Human-readable description of the task.
            status: Initial status (default ``"running"``).
        """
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO task_history (task_id, description, status)
                    VALUES (?, ?, ?)
                    """,
                    (task_id, description, status),
                )
            logger.debug("Saved task: %s (%s)", task_id, status)
        except sqlite3.Error as exc:
            logger.error("Failed to save task %s: %s", task_id, exc)
            raise

    def update_task(
        self,
        task_id: str,
        status: str,
        result_summary: str | None = None,
    ) -> None:
        """
        Update the status (and optionally the result summary) of a task.

        Sets ``completed_at`` automatically when *status* is
        ``"completed"`` or ``"failed"``.

        Args:
            task_id: The task to update.
            status: New status value.
            result_summary: Optional summary of the task outcome.
        """
        completed_at: str | None = None
        if status in {"completed", "failed"}:
            completed_at = datetime.utcnow().isoformat(sep=" ", timespec="seconds")

        try:
            with _get_connection(self.db_path) as conn:
                if result_summary is not None:
                    conn.execute(
                        """
                        UPDATE task_history
                        SET status = ?, result_summary = ?, completed_at = ?
                        WHERE task_id = ?
                        """,
                        (status, result_summary, completed_at, task_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE task_history
                        SET status = ?, completed_at = ?
                        WHERE task_id = ?
                        """,
                        (status, completed_at, task_id),
                    )
            logger.debug("Updated task %s -> %s", task_id, status)
        except sqlite3.Error as exc:
            logger.error("Failed to update task %s: %s", task_id, exc)
            raise

    def get_recent_tasks(self, limit: int = 20) -> list[dict]:
        """
        Return the most recently created tasks.

        Args:
            limit: Maximum number of tasks to return (default 20).

        Returns:
            List of task dicts ordered newest-first, each containing
            ``task_id``, ``description``, ``status``, ``result_summary``,
            ``created_at``, and ``completed_at``.
        """
        try:
            with _get_connection(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT task_id, description, status, result_summary,
                           created_at, completed_at
                    FROM task_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error("Failed to get recent tasks: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_memory_summary(self) -> str:
        """
        Build a human-readable memory summary for the ``/memory`` command.

        Returns:
            Formatted string containing stored preferences and recent tasks.
        """
        lines: list[str] = ["=== Memory Summary ===", ""]

        # Preferences
        preferences = self.get_all_preferences()
        lines.append("--- User Preferences ---")
        if preferences:
            for key, value in preferences.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  (none stored)")
        lines.append("")

        # Recent tasks
        tasks = self.get_recent_tasks(limit=10)
        lines.append("--- Recent Tasks (last 10) ---")
        if tasks:
            for task in tasks:
                status_label = task["status"].upper()
                summary = task["result_summary"] or "(no summary)"
                lines.append(
                    f"  [{status_label}] {task['description']} — {summary}"
                )
        else:
            lines.append("  (no tasks recorded)")

        return "\n".join(lines)

    def build_context_prompt(self, session_id: str) -> str:
        """
        Compose a context string suitable for injecting into the orchestrator.

        Includes stored user preferences and the most recent conversation
        messages so the model maintains continuity across turns.

        Args:
            session_id: The active session identifier.

        Returns:
            A formatted string ready to prepend to or include in a system
            or user prompt.
        """
        parts: list[str] = []

        # User preferences
        preferences = self.get_all_preferences()
        if preferences:
            pref_lines = "\n".join(
                f"  - {k}: {v}" for k, v in preferences.items()
            )
            parts.append(f"User Preferences:\n{pref_lines}")

        # Recent conversation history
        messages = self.get_recent_messages(session_id, limit=20)
        if messages:
            history_lines: list[str] = []
            for msg in messages:
                role_label = msg["role"].capitalize()
                history_lines.append(f"  {role_label}: {msg['content']}")
                if msg.get("citations"):
                    history_lines.append(
                        "    Citations: " + ", ".join(msg["citations"])
                    )
            parts.append("Recent Conversation:\n" + "\n".join(history_lines))

        if not parts:
            return ""

        return "\n\n".join(parts)
