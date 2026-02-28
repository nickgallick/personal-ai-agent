"""
Task scheduling module for Personal AI Agent.

Provides APScheduler-backed scheduling with SQLite persistence, natural-language
schedule parsing, and integration with the orchestrator and MemoryManager.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.job import Job

import config
from memory import MemoryManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone job target — must be importable at module level by APScheduler
# ---------------------------------------------------------------------------

def execute_scheduled_task(task_description: str, task_id: str) -> None:
    """
    Execute a scheduled task by invoking the orchestrator's run_agent function.

    This function is called by APScheduler in a background thread.  It creates
    its own event loop so that any async orchestrator logic can run safely.
    The result (or error) is persisted to MemoryManager's task_history table.

    Args:
        task_description: The natural-language task to run.
        task_id: Unique identifier used when recording the result.
    """
    logger.info("Scheduled task triggered: task_id=%s description=%r", task_id, task_description)

    memory = MemoryManager()

    # Record that this execution has started
    run_id = f"{task_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    try:
        memory.save_task(run_id, task_description, status="running")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not record task start for %s: %s", run_id, exc)

    result_summary: str
    final_status: str

    try:
        # Import here to avoid circular imports at module load time
        from orchestrator import run_agent  # type: ignore[import]

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run_agent(task_description))
        finally:
            loop.close()

        # Normalise whatever the orchestrator returns to a string summary
        if isinstance(result, dict):
            result_summary = result.get("response") or result.get("result") or str(result)
        elif result is None:
            result_summary = "(no output)"
        else:
            result_summary = str(result)

        final_status = "completed"
        logger.info("Scheduled task completed: task_id=%s", run_id)

    except Exception as exc:
        result_summary = f"Error: {exc}"
        final_status = "failed"
        logger.error("Scheduled task failed: task_id=%s error=%s", run_id, exc, exc_info=True)

    try:
        memory.update_task(run_id, final_status, result_summary=result_summary[:2000])
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not record task result for %s: %s", run_id, exc)


# ---------------------------------------------------------------------------
# Schedule string parser
# ---------------------------------------------------------------------------

# Mapping of weekday names to APScheduler cron abbreviations
_WEEKDAY_MAP: dict[str, str] = {
    "monday": "mon",
    "tuesday": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}

# Pre-compiled patterns, ordered from most-specific to least-specific
_SCHEDULE_PATTERNS: list[tuple[re.Pattern, Any]] = [
    # "every Monday at 8am" / "every Friday at 3:30pm"
    (
        re.compile(
            r"every\s+(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?",
            re.IGNORECASE,
        ),
        "weekday_at",
    ),
    # "weekdays at 9am" / "weekdays at 8:30am"
    (
        re.compile(
            r"weekdays?\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?",
            re.IGNORECASE,
        ),
        "weekdays_at",
    ),
    # "weekends at 10am"
    (
        re.compile(
            r"weekends?\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?",
            re.IGNORECASE,
        ),
        "weekends_at",
    ),
    # "monthly on the 1st at noon" / "monthly on the 15th at 9am"
    (
        re.compile(
            r"monthly\s+on\s+(?:the\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)?"
            r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?|"
            r"monthly\s+on\s+(?:the\s+)?(?P<day2>\d{1,2})(?:st|nd|rd|th)?"
            r"\s+at\s+(?P<time2>noon|midnight)",
            re.IGNORECASE,
        ),
        "monthly_at",
    ),
    # "daily at 9am" / "every day at noon"
    (
        re.compile(
            r"(?:daily|every\s+day)\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?",
            re.IGNORECASE,
        ),
        "daily_at",
    ),
    # "daily at noon" / "every day at midnight"
    (
        re.compile(
            r"(?:daily|every\s+day)\s+at\s+(?P<time>noon|midnight)",
            re.IGNORECASE,
        ),
        "daily_named",
    ),
    # "every 2 hours" / "every hour"
    (
        re.compile(
            r"every\s+(?P<n>\d+)\s+hours?",
            re.IGNORECASE,
        ),
        "interval_hours",
    ),
    (
        re.compile(r"every\s+hour\b", re.IGNORECASE),
        "interval_1hour",
    ),
    # "every 30 minutes" / "every 5 minutes"
    (
        re.compile(
            r"every\s+(?P<n>\d+)\s+minutes?",
            re.IGNORECASE,
        ),
        "interval_minutes",
    ),
    # "every minute"
    (
        re.compile(r"every\s+minute\b", re.IGNORECASE),
        "interval_1minute",
    ),
]


def _resolve_hour_minute(
    hour_str: str,
    minute_str: str | None,
    ampm: str | None,
) -> tuple[int, int]:
    """Convert raw hour/minute/ampm strings to 24-hour integers."""
    hour = int(hour_str)
    minute = int(minute_str) if minute_str else 0
    if ampm:
        ampm_lower = ampm.lower()
        if ampm_lower == "am":
            if hour == 12:
                hour = 0
        elif ampm_lower == "pm":
            if hour != 12:
                hour += 12
    return hour, minute


def parse_schedule(schedule_str: str) -> dict:
    """
    Parse a natural-language schedule string into APScheduler trigger kwargs.

    Supported formats include:
        - "every Monday at 8am"
        - "daily at 9am"
        - "every 2 hours"
        - "every 30 minutes"
        - "weekdays at 8am"
        - "every hour"
        - "monthly on the 1st at noon"

    Args:
        schedule_str: Human-readable schedule description.

    Returns:
        Dict with ``"trigger"`` key (``"cron"`` or ``"interval"``) plus the
        remaining kwargs required to instantiate the corresponding APScheduler
        trigger class.

    Raises:
        ValueError: When the string does not match any known pattern.
    """
    s = schedule_str.strip()

    for pattern, kind in _SCHEDULE_PATTERNS:
        m = pattern.search(s)
        if m is None:
            continue

        groups = m.groupdict()

        # ---------------------------------------------------------------
        if kind == "weekday_at":
            day_abbr = _WEEKDAY_MAP[groups["day"].lower()]
            hour, minute = _resolve_hour_minute(
                groups["hour"], groups.get("minute"), groups.get("ampm")
            )
            return {"trigger": "cron", "day_of_week": day_abbr, "hour": hour, "minute": minute}

        # ---------------------------------------------------------------
        if kind == "weekdays_at":
            hour, minute = _resolve_hour_minute(
                groups["hour"], groups.get("minute"), groups.get("ampm")
            )
            return {"trigger": "cron", "day_of_week": "mon-fri", "hour": hour, "minute": minute}

        # ---------------------------------------------------------------
        if kind == "weekends_at":
            hour, minute = _resolve_hour_minute(
                groups["hour"], groups.get("minute"), groups.get("ampm")
            )
            return {"trigger": "cron", "day_of_week": "sat,sun", "hour": hour, "minute": minute}

        # ---------------------------------------------------------------
        if kind == "monthly_at":
            # Two alternative sub-patterns: numeric time or named time
            day_val = groups.get("day") or groups.get("day2")
            time_named = groups.get("time2")
            if time_named:
                hour = 12 if time_named.lower() == "noon" else 0
                minute = 0
            else:
                hour, minute = _resolve_hour_minute(
                    groups["hour"], groups.get("minute"), groups.get("ampm")
                )
            return {"trigger": "cron", "day": int(day_val), "hour": hour, "minute": minute}

        # ---------------------------------------------------------------
        if kind == "daily_at":
            hour, minute = _resolve_hour_minute(
                groups["hour"], groups.get("minute"), groups.get("ampm")
            )
            return {"trigger": "cron", "hour": hour, "minute": minute}

        # ---------------------------------------------------------------
        if kind == "daily_named":
            named = groups["time"].lower()
            hour = 12 if named == "noon" else 0
            return {"trigger": "cron", "hour": hour, "minute": 0}

        # ---------------------------------------------------------------
        if kind == "interval_hours":
            return {"trigger": "interval", "hours": int(groups["n"])}

        if kind == "interval_1hour":
            return {"trigger": "interval", "hours": 1}

        # ---------------------------------------------------------------
        if kind == "interval_minutes":
            return {"trigger": "interval", "minutes": int(groups["n"])}

        if kind == "interval_1minute":
            return {"trigger": "interval", "minutes": 1}

    raise ValueError(
        f"Could not parse schedule string: {schedule_str!r}\n"
        "Supported formats include:\n"
        "  • \"every Monday at 8am\"\n"
        "  • \"daily at 9am\" / \"every day at noon\"\n"
        "  • \"weekdays at 8:30am\"\n"
        "  • \"weekends at 10am\"\n"
        "  • \"every 2 hours\" / \"every hour\"\n"
        "  • \"every 30 minutes\" / \"every minute\"\n"
        "  • \"monthly on the 1st at noon\"\n"
    )


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def format_task_list(tasks: list[dict]) -> str:
    """
    Format a list of task dicts as a markdown table.

    Args:
        tasks: List of dicts as returned by :meth:`TaskScheduler.list_tasks`.

    Returns:
        Markdown-formatted string, or a plain message when there are no tasks.
    """
    if not tasks:
        return "_No scheduled tasks._"

    col_id = max(len("Task ID"), max(len(t.get("id", "")) for t in tasks))
    col_desc = max(len("Description"), max(len(t.get("description", "")) for t in tasks))
    col_next = max(len("Next Run"), max(len(str(t.get("next_run_time", "N/A"))) for t in tasks))
    col_trig = max(len("Trigger"), max(len(str(t.get("trigger", ""))) for t in tasks))

    sep = (
        f"| {'-' * col_id} | {'-' * col_desc} | {'-' * col_next} | {'-' * col_trig} |"
    )
    header = (
        f"| {'Task ID'.ljust(col_id)} "
        f"| {'Description'.ljust(col_desc)} "
        f"| {'Next Run'.ljust(col_next)} "
        f"| {'Trigger'.ljust(col_trig)} |"
    )

    rows = [header, sep]
    for task in tasks:
        task_id = (task.get("id") or "").ljust(col_id)
        desc = (task.get("description") or "").ljust(col_desc)
        next_run = str(task.get("next_run_time") or "N/A").ljust(col_next)
        trig = str(task.get("trigger") or "").ljust(col_trig)
        rows.append(f"| {task_id} | {desc} | {next_run} | {trig} |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """
    APScheduler-backed task scheduler with SQLite persistence.

    Wraps a :class:`~apscheduler.schedulers.background.BackgroundScheduler`
    and provides a clean interface for adding, removing, and listing scheduled
    tasks. Jobs survive application restarts via the SQLAlchemy job store.
    """

    def __init__(self) -> None:
        # Ensure the data directory exists
        data_dir = Path(config.SCHEDULER_DB_PATH).parent
        data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite job store via SQLAlchemy
        db_url = f"sqlite:///{config.SCHEDULER_DB_PATH}"
        jobstores = {
            "default": SQLAlchemyJobStore(url=db_url),
        }

        executors = {
            "default": ThreadPoolExecutor(max_workers=5),
        }

        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
        }

        self._scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone.utc,
        )

        self._memory = MemoryManager()
        logger.info("TaskScheduler initialised (not yet started).")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler."""
        if self._scheduler.running:
            logger.warning("TaskScheduler.start() called but scheduler is already running.")
            return

        self._scheduler.start()
        jobs = self._scheduler.get_jobs()
        logger.info(
            "TaskScheduler started. Active jobs (%d): %s",
            len(jobs),
            [j.id for j in jobs],
        )

    def stop(self) -> None:
        """Shut down the background scheduler gracefully."""
        if not self._scheduler.running:
            logger.warning("TaskScheduler.stop() called but scheduler is not running.")
            return

        self._scheduler.shutdown(wait=True)
        logger.info("TaskScheduler stopped.")

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_scheduled_task(
        self,
        description: str,
        schedule_str: str,
        task_id: str | None = None,
    ) -> dict:
        """
        Schedule a new task.

        Args:
            description: What the task should do (passed verbatim to the
                         orchestrator when the job fires).
            schedule_str: Natural-language schedule, e.g. ``"daily at 9am"``.
            task_id: Optional stable identifier; generated from UUID4 if omitted.

        Returns:
            Dict with ``task_id``, ``description``, ``next_run``, and
            ``success`` keys.

        Raises:
            ValueError: If *schedule_str* cannot be parsed.
        """
        if not task_id:
            task_id = str(uuid4())

        trigger_kwargs = parse_schedule(schedule_str)  # raises ValueError on bad input
        trigger_type = trigger_kwargs.pop("trigger")

        if trigger_type == "cron":
            trigger = CronTrigger(**trigger_kwargs, timezone=timezone.utc)
        elif trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_kwargs, timezone=timezone.utc)
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type!r}")

        job: Job = self._scheduler.add_job(
            func=execute_scheduled_task,
            trigger=trigger,
            id=task_id,
            name=description,
            kwargs={"task_description": description, "task_id": task_id},
            replace_existing=True,
        )

        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        logger.info(
            "Scheduled task added: id=%s description=%r next_run=%s",
            task_id, description, next_run,
        )

        return {
            "task_id": task_id,
            "description": description,
            "next_run": next_run,
            "success": True,
        }

    def remove_task(self, task_id: str) -> dict:
        """
        Remove a scheduled task by ID.

        Args:
            task_id: The identifier of the job to remove.

        Returns:
            Dict with ``success`` (bool) and ``message`` keys.
        """
        try:
            self._scheduler.remove_job(task_id)
            logger.info("Removed scheduled task: %s", task_id)
            return {"success": True, "message": f"Task '{task_id}' removed."}
        except Exception as exc:
            logger.warning("Could not remove task %s: %s", task_id, exc)
            return {"success": False, "message": f"Task '{task_id}' not found or could not be removed: {exc}"}

    def list_tasks(self) -> list[dict]:
        """
        Return a list of all currently scheduled jobs.

        Returns:
            List of dicts, each containing ``id``, ``description``,
            ``next_run_time``, and ``trigger``.
        """
        jobs = self._scheduler.get_jobs()
        result = []
        for job in jobs:
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            result.append(
                {
                    "id": job.id,
                    "description": job.name or "",
                    "next_run_time": next_run,
                    "trigger": str(job.trigger),
                }
            )
        return result

    def get_task(self, task_id: str) -> dict | None:
        """
        Return info about a single scheduled task.

        Args:
            task_id: The job identifier to look up.

        Returns:
            Dict with job details, or ``None`` if not found.
        """
        job: Job | None = self._scheduler.get_job(task_id)
        if job is None:
            return None

        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        return {
            "id": job.id,
            "description": job.name or "",
            "next_run_time": next_run,
            "trigger": str(job.trigger),
        }
