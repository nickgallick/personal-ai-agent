"""
Personal AI Agent — Streamlit Application Entry Point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import importlib
import mimetypes
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure the personal-agent package directory is on sys.path
# so local imports (config, memory, workspace, scheduler, agents) resolve.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Local imports (all optional — degrade gracefully if missing)
# ---------------------------------------------------------------------------
try:
    import config
    from config import validate_config, MODEL_ROUTING
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False
    config = None  # type: ignore[assignment]

try:
    from memory import MemoryManager
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False
    MemoryManager = None  # type: ignore[assignment]

try:
    from workspace import WorkspaceManager
    _HAS_WORKSPACE = True
except ImportError:
    _HAS_WORKSPACE = False
    WorkspaceManager = None  # type: ignore[assignment]

# Scheduler is optional
try:
    from scheduler import TaskScheduler, parse_schedule, format_task_list
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False
    TaskScheduler = None  # type: ignore[assignment]

# Orchestrator — the main async entry point
try:
    _orch_module = importlib.import_module("agents.orchestrator")
    run_agent = getattr(_orch_module, "run_agent")
    _HAS_ORCHESTRATOR = True
except Exception:
    _HAS_ORCHESTRATOR = False
    run_agent = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def _load_css() -> None:
    css_path = _HERE / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        # Minimal fallback if styles.css is missing
        st.markdown(
            "<style>body{background:#1a1a2e;color:#e8e8f0;}</style>",
            unsafe_allow_html=True,
        )


_load_css()


# ---------------------------------------------------------------------------
# Singleton helpers — cached across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_memory() -> "MemoryManager | None":
    if _HAS_MEMORY:
        try:
            return MemoryManager()
        except Exception:
            return None
    return None


@st.cache_resource
def _get_workspace() -> "WorkspaceManager | None":
    if _HAS_WORKSPACE:
        try:
            return WorkspaceManager()
        except Exception:
            return None
    return None


@st.cache_resource
def _get_scheduler() -> "TaskScheduler | None":
    if _HAS_SCHEDULER:
        try:
            scheduler = TaskScheduler()
            scheduler.start()
            return scheduler
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "status_text" not in st.session_state:
        st.session_state.status_text = "Thinking…"
    if "pending_uploads" not in st.session_state:
        st.session_state.pending_uploads = []
    if "schedule_suggestion" not in st.session_state:
        st.session_state.schedule_suggestion = None
    if "schedule_confirmed" not in st.session_state:
        st.session_state.schedule_confirmed = False


_init_session_state()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _file_icon(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    icons = {
        ".pdf": "📄", ".csv": "📊", ".xlsx": "📊", ".xls": "📊",
        ".png": "🖼️", ".jpg": "🖼️", ".jpeg": "🖼️", ".gif": "🖼️", ".webp": "🖼️",
        ".txt": "📝", ".md": "📝",
        ".py": "🐍", ".js": "🟨", ".ts": "🔷", ".jsx": "⚛️", ".tsx": "⚛️",
        ".html": "🌐", ".css": "🎨", ".json": "📋",
        ".yaml": "⚙️", ".yml": "⚙️",
        ".zip": "🗜️", ".tar": "🗜️", ".gz": "🗜️",
        ".mp3": "🎵", ".wav": "🎵", ".mp4": "🎬",
    }
    return icons.get(ext, "📎")


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _detect_schedule_phrase(text: str) -> bool:
    """Return True if the message looks like it contains a scheduling intent."""
    patterns = [
        r"\bevery\s+\w+day\b",
        r"\bevery\s+\d+\s+(?:hours?|minutes?|mins?|seconds?)\b",
        r"\bdaily\s+at\b",
        r"\bevery\s+(?:morning|evening|night|hour|day|week|month)\b",
        r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"\bschedule\s+this\b",
        r"\bremind\s+me\b",
        r"\brepeat\s+(?:every|daily|weekly)\b",
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def _extract_schedule_str(text: str) -> str:
    """Best-effort extraction of a cron-like schedule description from the text."""
    # Common patterns
    daily_at = re.search(r"daily\s+at\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)", text, re.IGNORECASE)
    if daily_at:
        return f"daily at {daily_at.group(1)}"
    every_n_hours = re.search(r"every\s+(\d+)\s+hours?", text, re.IGNORECASE)
    if every_n_hours:
        return f"every {every_n_hours.group(1)} hours"
    every_weekday = re.search(r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text, re.IGNORECASE)
    if every_weekday:
        return f"every {every_weekday.group(1)}"
    every_morning = re.search(r"every\s+morning", text, re.IGNORECASE)
    if every_morning:
        return "daily at 9am"
    every_evening = re.search(r"every\s+evening", text, re.IGNORECASE)
    if every_evening:
        return "daily at 6pm"
    return "daily"


def _format_timestamp(ts: str) -> str:
    """Format an ISO timestamp to a human-friendly string."""
    try:
        dt = datetime.fromisoformat(ts.replace(" ", "T"))
        now = datetime.utcnow()
        diff = now - dt
        if diff.days == 0:
            minutes = int(diff.total_seconds() / 60)
            if minutes < 1:
                return "just now"
            if minutes < 60:
                return f"{minutes}m ago"
            hours = minutes // 60
            return f"{hours}h ago"
        if diff.days == 1:
            return "yesterday"
        if diff.days < 7:
            return f"{diff.days}d ago"
        return dt.strftime("%b %d")
    except Exception:
        return ts[:16] if ts else ""


def _run_async(coro) -> dict:
    """Run an async coroutine safely from a synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing event loop (e.g. pytest / some environments)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------

def _render_citations(citations: list[str]) -> None:
    if not citations:
        return
    parts = ['<div class="citation-row"><span class="citation-label">Sources:</span>']
    for i, url in enumerate(citations, 1):
        label = url
        # Shorten display text
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            label = parsed.netloc.removeprefix("www.") or url[:40]
        except Exception:
            label = url[:40]
        parts.append(
            f'<a class="citation-badge" href="{url}" target="_blank" rel="noopener">'
            f'[{i}] {label}</a>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# File card rendering + download
# ---------------------------------------------------------------------------

def _render_file_cards(files: list[str], session_id: str) -> None:
    """Show file cards with download buttons for each generated/uploaded file."""
    ws = _get_workspace()
    for fpath in files:
        p = Path(fpath)
        filename = p.name
        icon = _file_icon(filename)

        # Try to read the file bytes for download
        file_bytes: bytes | None = None
        try:
            if p.exists():
                file_bytes = p.read_bytes()
            elif ws:
                file_bytes = ws.read_file(session_id, filename)
        except Exception:
            file_bytes = None

        col1, col2 = st.columns([6, 2])
        with col1:
            size_str = _format_bytes(len(file_bytes)) if file_bytes else "—"
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            st.markdown(
                f"""<div class="file-card">
                  <span class="file-icon">{icon}</span>
                  <div class="file-info">
                    <div class="file-name">{filename}</div>
                    <div class="file-meta">{size_str} · {mime}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            # Display images inline
            if _is_image_file(filename) and file_bytes:
                st.image(file_bytes, use_container_width=True)

        with col2:
            if file_bytes:
                st.download_button(
                    label="⬇ Download",
                    data=file_bytes,
                    file_name=filename,
                    mime=mime,
                    key=f"dl_{fpath}_{uuid.uuid4().hex[:6]}",
                )


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------

def _render_message(msg: dict, session_id: str) -> None:
    role = msg["role"]
    content = msg.get("content", "")
    citations = msg.get("citations") or []
    files = msg.get("files") or []

    with st.chat_message(role):
        st.markdown(content)
        if citations:
            _render_citations(citations)
        if files:
            _render_file_cards(files, session_id)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_memory_command() -> dict:
    mem = _get_memory()
    if mem is None:
        return {
            "response": "Memory system is not available.",
            "citations": [],
            "files_created": [],
        }
    summary = mem.get_memory_summary()
    return {
        "response": f"```\n{summary}\n```",
        "citations": [],
        "files_created": [],
    }


def _handle_schedule_list_command() -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "response": "Scheduler is not available.",
            "citations": [],
            "files_created": [],
        }
    try:
        tasks = scheduler.list_tasks()
        if not tasks:
            return {
                "response": "No scheduled tasks currently configured.",
                "citations": [],
                "files_created": [],
            }
        if _HAS_SCHEDULER:
            try:
                formatted = format_task_list(tasks)
                return {"response": formatted, "citations": [], "files_created": []}
            except Exception:
                pass
        lines = ["**Scheduled Tasks**\n"]
        for t in tasks:
            lines.append(
                f"- **{t.get('task_id', '?')}** — {t.get('description', '')}  \n"
                f"  Schedule: `{t.get('schedule', '')}` | "
                f"Next: `{t.get('next_run', 'unknown')}`"
            )
        return {"response": "\n".join(lines), "citations": [], "files_created": []}
    except Exception as exc:
        return {
            "response": f"Error listing tasks: {exc}",
            "citations": [],
            "files_created": [],
        }


def _handle_schedule_remove_command(task_id: str) -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"response": "Scheduler is not available.", "citations": [], "files_created": []}
    try:
        scheduler.remove_task(task_id.strip())
        return {
            "response": f"Task `{task_id}` removed successfully.",
            "citations": [],
            "files_created": [],
        }
    except Exception as exc:
        return {
            "response": f"Failed to remove task `{task_id}`: {exc}",
            "citations": [],
            "files_created": [],
        }


def _offer_schedule(user_message: str) -> None:
    """Detect scheduling phrases and store a suggestion in session state."""
    if _detect_schedule_phrase(user_message) and not st.session_state.schedule_suggestion:
        st.session_state.schedule_suggestion = {
            "description": user_message,
            "schedule_str": _extract_schedule_str(user_message),
        }


# ---------------------------------------------------------------------------
# Status callback builder
# ---------------------------------------------------------------------------

def _make_status_callback(status_placeholder) -> Callable[[str], None]:
    """Return a callable that updates the status placeholder text."""
    def callback(status: str) -> None:
        try:
            status_placeholder.markdown(
                f'<p style="color:var(--text-muted);font-size:0.82rem;">'
                f'⚙️ {status}</p>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    return callback


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    mem = _get_memory()
    scheduler = _get_scheduler()

    with st.sidebar:
        # ---- Brand header ------------------------------------------------
        st.markdown(
            """
            <div style="text-align:center;padding:0.5rem 0 1rem;">
              <div style="font-size:2rem;line-height:1;">🤖</div>
              <div style="font-size:1.05rem;font-weight:700;
                          color:var(--text-primary);margin-top:0.3rem;">
                Personal AI Agent
              </div>
              <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem;">
                Your autonomous assistant
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- New Chat button ----------------------------------------------
        if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.schedule_suggestion = None
            st.session_state.pending_uploads = []
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- Conversation history -----------------------------------------
        st.markdown(
            '<div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">'
            "Recent Conversations</div>",
            unsafe_allow_html=True,
        )

        if mem:
            try:
                sessions = mem.get_all_sessions()
            except Exception:
                sessions = []

            if sessions:
                for sess in sessions[:15]:
                    sid = sess.get("session_id", "")
                    preview = sess.get("preview", "")[:55] or "(empty)"
                    ts = _format_timestamp(sess.get("last_timestamp", ""))
                    is_active = sid == st.session_state.session_id
                    active_class = " active" if is_active else ""

                    # Render the item as HTML for styling, but also provide
                    # a real button for navigation
                    st.markdown(
                        f'<div class="session-item{active_class}">'
                        f'<div class="session-preview">{preview}</div>'
                        f'<div class="session-time">{ts}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if not is_active:
                        if st.button(
                            "Load",
                            key=f"load_sess_{sid}",
                            help=f"Session: {sid[:8]}…",
                        ):
                            # Load messages from memory
                            msgs = mem.get_recent_messages(sid, limit=100)
                            st.session_state.messages = [
                                {
                                    "role": m["role"],
                                    "content": m["content"],
                                    "citations": m.get("citations") or [],
                                    "files": [],
                                }
                                for m in msgs
                            ]
                            st.session_state.session_id = sid
                            st.rerun()
            else:
                st.markdown(
                    '<div style="color:var(--text-muted);font-size:0.8rem;'
                    'padding:0.3rem 0;">No conversations yet.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:var(--text-muted);font-size:0.8rem;">'
                "Memory unavailable.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- Scheduled Tasks ---------------------------------------------
        with st.expander("🕐 Scheduled Tasks", expanded=False):
            if scheduler:
                try:
                    tasks = scheduler.list_tasks()
                except Exception:
                    tasks = []

                task_count = len(tasks)
                st.markdown(
                    f'<div style="font-size:0.8rem;color:var(--text-secondary);'
                    f'margin-bottom:0.5rem;">'
                    f"<b>{task_count}</b> active task{'s' if task_count != 1 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if tasks:
                    for t in tasks[:5]:
                        tid = t.get("task_id", "?")
                        desc = (t.get("description") or "")[:50]
                        schedule = t.get("schedule", "")
                        next_run = t.get("next_run", "unknown")
                        st.markdown(
                            f'<div class="task-item">'
                            f'<div class="task-desc">{desc}</div>'
                            f'<div class="task-schedule">⏱ {schedule} · next: {next_run}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    if task_count > 5:
                        st.caption(f"… and {task_count - 5} more")

                    if st.button("View all tasks", key="view_all_tasks"):
                        result = _handle_schedule_list_command()
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["response"],
                                "citations": [],
                                "files": [],
                            }
                        )
                        st.rerun()
                else:
                    st.caption("No tasks scheduled. Ask the agent to schedule something!")
            else:
                st.caption("Scheduler unavailable.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- Settings ----------------------------------------------------
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);'
                'margin-bottom:0.5rem;">API Keys</div>',
                unsafe_allow_html=True,
            )

            if _HAS_CONFIG:
                status = validate_config()
                key_labels = {
                    "sonar_api_key": ("Sonar (Search)", "SONAR_API_KEY"),
                    "openrouter_api_key": ("OpenRouter (LLM)", "OPENROUTER_API_KEY"),
                    "github_token": ("GitHub", "GITHUB_TOKEN"),
                }
                for key, (label, _env) in key_labels.items():
                    connected = status.get(key, False)
                    dot_class = "connected" if connected else "missing"
                    status_text = "Connected" if connected else "Missing"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;'
                        f'font-size:0.82rem;margin-bottom:0.3rem;">'
                        f'<span class="status-dot {dot_class}"></span>'
                        f'<span style="color:var(--text-secondary)">{label}</span>'
                        f'<span style="margin-left:auto;color:var(--text-muted);'
                        f'font-size:0.75rem;">{status_text}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:var(--text-muted);font-size:0.8rem;">'
                    "Config unavailable.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);'
                'margin:0.7rem 0 0.4rem;">Model Routing</div>',
                unsafe_allow_html=True,
            )

            if _HAS_CONFIG:
                routing_display = {
                    "orchestration": "Orchestrator",
                    "coding": "Code",
                    "reasoning": "Reasoning",
                    "fast": "Fast responses",
                    "search_deep": "Deep search",
                    "search_quick": "Quick search",
                    "image": "Image gen",
                }
                for route_key, route_label in routing_display.items():
                    model_id = MODEL_ROUTING.get(route_key, "—")
                    short_model = model_id.split("/")[-1] if "/" in model_id else model_id
                    st.markdown(
                        f'<div style="display:flex;align-items:center;'
                        f'font-size:0.78rem;margin-bottom:0.25rem;">'
                        f'<span style="color:var(--text-secondary);flex:1">{route_label}</span>'
                        f'<span class="model-badge">{short_model}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ---- App info footer ---------------------------------------------
        st.markdown(
            '<div style="position:absolute;bottom:1rem;left:0;right:0;'
            'text-align:center;font-size:0.7rem;color:var(--text-muted);">'
            "Personal AI Agent · Local</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Setup warning banner
# ---------------------------------------------------------------------------

def _render_setup_warning() -> None:
    if not _HAS_CONFIG:
        st.markdown(
            """
            <div class="setup-banner">
              <h3>⚠️ Configuration Missing</h3>
              <p>Could not import <code>config.py</code>.
                 Make sure the application files are in the correct directory.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = validate_config()
    missing = [k for k, v in status.items() if not v]

    if not missing:
        return  # All keys present — no warning needed

    key_docs = {
        "sonar_api_key": ("SONAR_API_KEY", "https://www.perplexity.ai/settings/api", "Perplexity Sonar (web search)"),
        "openrouter_api_key": ("OPENROUTER_API_KEY", "https://openrouter.ai/keys", "OpenRouter (LLM routing)"),
        "github_token": ("GITHUB_TOKEN", "https://github.com/settings/tokens", "GitHub (optional)"),
    }

    items_html = ""
    for k in missing:
        if k in key_docs:
            env, url, desc = key_docs[k]
            items_html += (
                f'<li><code>{env}</code> — {desc}. '
                f'<a href="{url}" target="_blank">Get key →</a></li>'
            )

    st.markdown(
        f"""
        <div class="setup-banner">
          <h3>⚙️ Setup Required</h3>
          <p>Some API keys are missing. Add them to your <code>.env</code> file:</p>
          <ul>{items_html}</ul>
          <p style="margin-bottom:0">The agent will work in limited mode until keys are configured.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

def _render_welcome() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem 2rem;">
          <div style="font-size:3.5rem;margin-bottom:0.75rem;">🤖</div>
          <h2 style="font-size:1.6rem;font-weight:700;margin:0 0 0.5rem;">
            How can I help you today?
          </h2>
          <p style="color:var(--text-muted);font-size:0.95rem;max-width:500px;margin:0 auto;">
            I can search the web, write and run code, analyse files, manage tasks,
            and remember your preferences across conversations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Quick action suggestions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Search the web", key="qa_search", use_container_width=True):
            st.session_state._quick_input = "Search the web for the latest AI news"
    with col2:
        if st.button("💻 Write some code", key="qa_code", use_container_width=True):
            st.session_state._quick_input = "Write a Python script that "
    with col3:
        if st.button("📁 Analyse a file", key="qa_file", use_container_width=True):
            st.session_state._quick_input = "Please upload a file and I'll analyse it."


# ---------------------------------------------------------------------------
# Schedule suggestion UI
# ---------------------------------------------------------------------------

def _render_schedule_suggestion() -> None:
    suggestion = st.session_state.get("schedule_suggestion")
    if not suggestion or not _HAS_SCHEDULER:
        return

    sched_str = suggestion["schedule_str"]
    description = suggestion["description"][:80]

    st.markdown(
        f"""
        <div class="schedule-banner">
          🕐 This looks like something you might want to schedule.
          <b>Run "{description}…" {sched_str}?</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([2, 6])
    with col1:
        if st.button("Yes, schedule it", key="confirm_schedule"):
            scheduler = _get_scheduler()
            if scheduler:
                try:
                    scheduler.add_scheduled_task(
                        description=suggestion["description"],
                        schedule_str=sched_str,
                    )
                    st.success(f"Task scheduled: {sched_str}")
                except Exception as exc:
                    st.error(f"Failed to schedule: {exc}")
            st.session_state.schedule_suggestion = None
            st.rerun()
    with col2:
        if st.button("No thanks", key="dismiss_schedule"):
            st.session_state.schedule_suggestion = None
            st.rerun()


# ---------------------------------------------------------------------------
# File upload handling
# ---------------------------------------------------------------------------

def _process_uploaded_files(uploaded_files: list) -> list[dict]:
    """Save uploaded files to the workspace, return list of file dicts."""
    ws = _get_workspace()
    saved: list[dict] = []
    for uf in uploaded_files:
        try:
            if ws:
                path = ws.save_uploaded_file(st.session_state.session_id, uf)
                saved.append({
                    "name": uf.name,
                    "path": str(path),
                    "type": getattr(uf, "type", "application/octet-stream"),
                    "size": path.stat().st_size if path.exists() else 0,
                })
            else:
                # Fallback: store bytes in memory
                data = uf.read()
                saved.append({
                    "name": uf.name,
                    "path": "",
                    "type": getattr(uf, "type", "application/octet-stream"),
                    "size": len(data),
                    "data": data,
                })
        except Exception as exc:
            st.warning(f"Could not save file '{uf.name}': {exc}")
    return saved


# ---------------------------------------------------------------------------
# Main message processing
# ---------------------------------------------------------------------------

def _process_message(user_message: str, uploaded_files: list[dict]) -> dict:
    """
    Route the user message to the appropriate handler.
    Returns a response dict with keys: response, citations, files_created.
    """
    stripped = user_message.strip()
    lower = stripped.lower()

    # --- Special commands -------------------------------------------------
    if lower == "/memory":
        return _handle_memory_command()

    if lower in ("/schedule list", "/schedule"):
        return _handle_schedule_list_command()

    if lower.startswith("/schedule remove "):
        task_id = stripped[len("/schedule remove "):].strip()
        return _handle_schedule_remove_command(task_id)

    # --- Normal agent call ------------------------------------------------
    if not _HAS_ORCHESTRATOR or run_agent is None:
        return {
            "response": (
                "The agent orchestrator is not available. "
                "Please check that `agents/orchestrator.py` exists and all "
                "dependencies are installed."
            ),
            "citations": [],
            "files_created": [],
        }

    # Build a status callback placeholder
    status_ph = st.empty()
    callback = _make_status_callback(status_ph)

    try:
        result = _run_async(
            run_agent(
                user_message=user_message,
                session_id=st.session_state.session_id,
                uploaded_files=uploaded_files if uploaded_files else None,
                status_callback=callback,
            )
        )
    except Exception as exc:
        tb = traceback.format_exc()
        result = {
            "response": (
                f"An error occurred while processing your request:\n\n"
                f"```\n{exc}\n```\n\n"
                "<details><summary>Full traceback</summary>\n\n"
                f"```\n{tb}\n```\n\n</details>"
            ),
            "citations": [],
            "files_created": [],
        }
    finally:
        status_ph.empty()

    return result


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    # Render sidebar
    _render_sidebar()

    # ---- Main content area -----------------------------------------------
    # Setup warning (if keys missing)
    _render_setup_warning()

    # Welcome screen if no messages
    if not st.session_state.messages:
        _render_welcome()

    # Render all existing messages
    for msg in st.session_state.messages:
        _render_message(msg, st.session_state.session_id)

    # Schedule suggestion banner (shown between messages and input)
    _render_schedule_suggestion()

    # ---- File uploader (above chat input) --------------------------------
    uploaded_files = st.file_uploader(
        "Attach files",
        type=[
            "pdf", "csv", "xlsx", "xls",
            "png", "jpg", "jpeg", "gif", "webp",
            "txt", "json", "py", "js", "ts", "jsx", "tsx",
            "html", "md", "css", "yaml", "yml",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
        help="Upload files for the agent to analyse or work with",
    )

    # ---- Chat input ------------------------------------------------------
    # Handle quick-action button pre-fills
    prefill = st.session_state.pop("_quick_input", None)

    prompt = st.chat_input(
        "Ask anything…",
        disabled=st.session_state.processing,
    )

    if prefill and not prompt:
        # Show pre-fill as a user message immediately
        st.session_state.messages.append(
            {"role": "user", "content": prefill, "citations": [], "files": []}
        )
        st.rerun()

    if prompt:
        st.session_state.processing = True

        # Process any uploaded files
        saved_files: list[dict] = []
        if uploaded_files:
            with st.spinner("Saving uploaded files…"):
                saved_files = _process_uploaded_files(uploaded_files)

        # Add user message to history
        user_msg = {
            "role": "user",
            "content": prompt,
            "citations": [],
            "files": [f["path"] for f in saved_files if f.get("path")],
        }
        st.session_state.messages.append(user_msg)

        # Persist user message to memory
        mem = _get_memory()
        if mem:
            try:
                mem.save_message(
                    session_id=st.session_state.session_id,
                    role="user",
                    content=prompt,
                )
                mem.detect_and_store_preference(prompt)
            except Exception:
                pass

        # Render user message immediately
        _render_message(user_msg, st.session_state.session_id)

        # Detect scheduling intent (offered after response)
        _offer_schedule(prompt)

        # Process the message with a spinner
        with st.spinner(""):
            status_container = st.empty()
            status_container.markdown(
                '<p style="color:var(--text-muted);font-size:0.82rem;">'
                '⚙️ Thinking…</p>',
                unsafe_allow_html=True,
            )

            result = _process_message(prompt, saved_files)
            status_container.empty()

        response_text = result.get("response", "(no response)")
        citations = result.get("citations") or []
        files_created = result.get("files_created") or []

        # Add assistant message to history
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "citations": citations,
            "files": files_created,
        }
        st.session_state.messages.append(assistant_msg)

        # Persist assistant message to memory
        if mem:
            try:
                mem.save_message(
                    session_id=st.session_state.session_id,
                    role="assistant",
                    content=response_text,
                    citations=citations or None,
                )
            except Exception:
                pass

        st.session_state.processing = False

        # Rerun to re-render the full message list (ensures assistant bubble appears)
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__" or True:
    main()
