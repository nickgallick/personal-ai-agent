"""
Personal AI Agent — Streamlit Application Entry Point.
Perplexity.ai-inspired light theme UI.

Run with:
    streamlit run app_new.py
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
# Path setup — ensure the project root is on sys.path
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
    MODEL_ROUTING = {}  # type: ignore[assignment]

    def validate_config():  # type: ignore[misc]
        return {}

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

try:
    from scheduler import TaskScheduler, parse_schedule, format_task_list
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False
    TaskScheduler = None  # type: ignore[assignment]
    parse_schedule = None  # type: ignore[assignment]
    format_task_list = None  # type: ignore[assignment]

# Orchestrator — CRITICAL: import from root orchestrator.py, NOT agents.orchestrator
try:
    _orch_module = importlib.import_module("orchestrator")
    run_agent = getattr(_orch_module, "run_agent")
    _HAS_ORCHESTRATOR = True
except Exception:
    _HAS_ORCHESTRATOR = False
    run_agent = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Perplexity",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _load_css() -> None:
    css_path = _HERE / "styles_new.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        # Minimal fallback
        css_path_orig = _HERE / "styles.css"
        if css_path_orig.exists():
            st.markdown(f"<style>{css_path_orig.read_text()}</style>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<style>body{background:#F8F7F4;color:#1A1A1A;}</style>",
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
    if "selected_model" not in st.session_state:
        if _HAS_CONFIG and MODEL_ROUTING:
            st.session_state.selected_model = list(MODEL_ROUTING.keys())[0]
        else:
            st.session_state.selected_model = "default"
    if "_quick_input" not in st.session_state:
        st.session_state._quick_input = None
    if "active_view" not in st.session_state:
        st.session_state.active_view = "search"  # "search" | "computer"


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
        ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".gif": "🖼", ".webp": "🖼",
        ".txt": "📝", ".md": "📝",
        ".py": "🐍", ".js": "📜", ".ts": "📜", ".jsx": "📜", ".tsx": "📜",
        ".html": "🌐", ".css": "🎨", ".json": "📋",
        ".yaml": "⚙", ".yml": "⚙",
        ".zip": "🗜", ".tar": "🗜", ".gz": "🗜",
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
    """Best-effort extraction of a cron-like schedule description."""
    daily_at = re.search(r"daily\s+at\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)", text, re.IGNORECASE)
    if daily_at:
        return f"daily at {daily_at.group(1)}"
    every_n_hours = re.search(r"every\s+(\d+)\s+hours?", text, re.IGNORECASE)
    if every_n_hours:
        return f"every {every_n_hours.group(1)} hours"
    every_weekday = re.search(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        text, re.IGNORECASE,
    )
    if every_weekday:
        return f"every {every_weekday.group(1)}"
    if re.search(r"every\s+morning", text, re.IGNORECASE):
        return "daily at 9am"
    if re.search(r"every\s+evening", text, re.IGNORECASE):
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
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _get_favicon_url(url: str) -> str:
    """Return a Google favicon URL for a given source URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
    except Exception:
        return ""


def _shorten_url(url: str) -> str:
    """Return a short display label for a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        netloc = (parsed.netloc or "").replace("www.", "")
        return netloc or url[:40]
    except Exception:
        return url[:40]


def _generate_followups(last_user_msg: str, last_assistant_msg: str) -> list[str]:
    """Generate follow-up suggestions based on the last exchange."""
    # Heuristic follow-ups — in production these could come from the LLM
    if "code" in last_user_msg.lower() or "script" in last_user_msg.lower():
        return [
            "Explain this code in detail",
            "Add error handling",
            "Write tests for this",
            "Optimize for performance",
        ]
    if any(w in last_user_msg.lower() for w in ["news", "latest", "today", "current"]):
        return [
            "Tell me more about this",
            "What are the implications?",
            "Show related topics",
            "Summarize the key points",
        ]
    if any(w in last_user_msg.lower() for w in ["analyze", "analyse", "review", "check"]):
        return [
            "Provide recommendations",
            "Compare with alternatives",
            "Show detailed breakdown",
            "What should I do next?",
        ]
    return [
        "Tell me more",
        "Summarize the key points",
        "What are the implications?",
        "Give me examples",
    ]


# ---------------------------------------------------------------------------
# Citation / source card rendering
# ---------------------------------------------------------------------------

def _render_source_cards(citations: list[str]) -> None:
    """Render horizontal scrollable source cards."""
    if not citations:
        return

    cards_html = ['<div class="sources-section"><div class="sources-label">Sources</div><div class="sources-scroll">']
    for i, url in enumerate(citations, 1):
        favicon = _get_favicon_url(url)
        label = _shorten_url(url)
        cards_html.append(
            f'<a class="source-card" href="{url}" target="_blank" rel="noopener">'
            f'<img class="source-favicon" src="{favicon}" onerror="this.style.display=\'none\'" alt="" />'
            f'<span class="source-label">{label}</span>'
            f'<span class="source-num">{i}</span>'
            f'</a>'
        )
    cards_html.append('</div></div>')
    st.markdown("".join(cards_html), unsafe_allow_html=True)


def _render_inline_citations(citations: list[str], content: str) -> str:
    """Return content with inline citation badges appended."""
    if not citations:
        return content
    badges = " ".join(
        f'<a class="citation-badge" href="{url}" target="_blank" rel="noopener">[{i}]</a>'
        for i, url in enumerate(citations, 1)
    )
    return content + f'\n\n<div class="citations-inline">{badges}</div>'


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
            if _is_image_file(filename) and file_bytes:
                st.image(file_bytes, use_container_width=True)
        with col2:
            if file_bytes:
                st.download_button(
                    label="Download",
                    data=file_bytes,
                    file_name=filename,
                    mime=mime,
                    key=f"dl_{fpath}_{uuid.uuid4().hex[:6]}",
                )


# ---------------------------------------------------------------------------
# Message rendering (Perplexity-style)
# ---------------------------------------------------------------------------

def _render_user_message(msg: dict, idx: int) -> None:
    """Render a user message as large flat text (no bubble)."""
    content = msg.get("content", "")
    st.markdown(
        f'<div class="user-query">{content}</div>',
        unsafe_allow_html=True,
    )


def _render_assistant_message(msg: dict, idx: int, session_id: str, is_latest: bool = False) -> None:
    """Render an assistant answer as a clean card with sources, content, actions, follow-ups."""
    content = msg.get("content", "")
    citations = msg.get("citations") or []
    files = msg.get("files") or []

    # Sources row
    _render_source_cards(citations)

    # Answer content
    st.markdown(f'<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(content)
    if files:
        _render_file_cards(files, session_id)
    st.markdown('</div>', unsafe_allow_html=True)

    # Action bar
    col_copy, col_share, col_spacer = st.columns([1, 1, 8])
    with col_copy:
        if st.button("Copy", key=f"copy_{idx}_{uuid.uuid4().hex[:4]}", help="Copy answer"):
            # JavaScript copy not directly possible in Streamlit, show toast
            st.toast("Copied to clipboard!", icon="✓")
    with col_share:
        if st.button("Share", key=f"share_{idx}_{uuid.uuid4().hex[:4]}", help="Share"):
            st.toast("Share link copied!", icon="✓")

    # Follow-up suggestions (only for the latest answer)
    if is_latest and st.session_state.messages:
        # Find the user message that preceded this one
        prev_user = ""
        for m in reversed(st.session_state.messages):
            if m["role"] == "user":
                prev_user = m.get("content", "")
                break
        followups = _generate_followups(prev_user, content)
        st.markdown('<div class="followups-section">', unsafe_allow_html=True)
        for i, suggestion in enumerate(followups):
            st.markdown(
                f'<div class="follow-up-item" id="followup_{idx}_{i}">'
                f'<span class="follow-up-icon">↗</span>{suggestion}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Actual clickable buttons for follow-ups (invisible, overlapping)
        followup_cols = st.columns(len(followups))
        for i, (col, suggestion) in enumerate(zip(followup_cols, followups)):
            with col:
                if st.button(suggestion, key=f"fu_{idx}_{i}_{uuid.uuid4().hex[:4]}", help=suggestion):
                    st.session_state._quick_input = suggestion


def _render_message(msg: dict, idx: int, session_id: str, is_latest: bool = False) -> None:
    role = msg["role"]
    if role == "user":
        _render_user_message(msg, idx)
    elif role == "assistant":
        _render_assistant_message(msg, idx, session_id, is_latest)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_memory_command() -> dict:
    mem = _get_memory()
    if mem is None:
        return {"response": "Memory system is not available.", "citations": [], "files_created": []}
    summary = mem.get_memory_summary()
    return {"response": f"```\n{summary}\n```", "citations": [], "files_created": []}


def _handle_schedule_list_command() -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"response": "Scheduler is not available.", "citations": [], "files_created": []}
    try:
        tasks = scheduler.list_tasks()
        if not tasks:
            return {"response": "No scheduled tasks currently configured.", "citations": [], "files_created": []}
        if _HAS_SCHEDULER and format_task_list:
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
        return {"response": f"Error listing tasks: {exc}", "citations": [], "files_created": []}


def _handle_schedule_remove_command(task_id: str) -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"response": "Scheduler is not available.", "citations": [], "files_created": []}
    try:
        scheduler.remove_task(task_id.strip())
        return {"response": f"Task `{task_id}` removed successfully.", "citations": [], "files_created": []}
    except Exception as exc:
        return {"response": f"Failed to remove task `{task_id}`: {exc}", "citations": [], "files_created": []}


def _offer_schedule(user_message: str) -> None:
    if _detect_schedule_phrase(user_message) and not st.session_state.schedule_suggestion:
        st.session_state.schedule_suggestion = {
            "description": user_message,
            "schedule_str": _extract_schedule_str(user_message),
        }


# ---------------------------------------------------------------------------
# Status callback builder
# ---------------------------------------------------------------------------

def _make_status_callback(status_placeholder) -> Callable[[str], None]:
    def callback(status: str) -> None:
        try:
            status_placeholder.markdown(
                f'<div class="thinking-container">'
                f'<div class="thinking-dot"></div>'
                f'<div class="thinking-dot"></div>'
                f'<div class="thinking-dot"></div>'
                f'<span class="thinking-status">{status}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    return callback


# ---------------------------------------------------------------------------
# Thinking indicator
# ---------------------------------------------------------------------------

def _render_thinking(status: str = "Searching...") -> None:
    st.markdown(
        f'<div class="thinking-container">'
        f'<div class="thinking-dot"></div>'
        f'<div class="thinking-dot"></div>'
        f'<div class="thinking-dot"></div>'
        f'<span class="thinking-status">{status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
            <div class="sidebar-brand">
              <span class="sidebar-brand-icon">✦</span>
              <span class="sidebar-brand-text">perplexity</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Mode toggles ------------------------------------------------
        col_search, col_computer = st.columns(2)
        with col_search:
            search_active = st.session_state.active_view == "search"
            if st.button(
                "Search",
                key="mode_search",
                use_container_width=True,
                type="primary" if search_active else "secondary",
            ):
                st.session_state.active_view = "search"
                st.rerun()
        with col_computer:
            computer_active = st.session_state.active_view == "computer"
            if st.button(
                "Computer",
                key="mode_computer",
                use_container_width=True,
                type="primary" if computer_active else "secondary",
            ):
                st.session_state.active_view = "computer"
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ---- New Thread --------------------------------------------------
        if st.button("+ New Thread", key="new_thread_btn", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.schedule_suggestion = None
            st.session_state.pending_uploads = []
            st.session_state._quick_input = None
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ---- Navigation items -------------------------------------------
        nav_items = [
            ("🕐", "History"),
            ("🧭", "Discover"),
            ("⬡", "Spaces"),
            ("📈", "Finance"),
            ("•••", "More"),
        ]
        for icon, label in nav_items:
            st.markdown(
                f'<div class="sidebar-nav-item">'
                f'<span class="sidebar-nav-icon">{icon}</span>'
                f'<span class="sidebar-nav-label">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ---- Model selector ---------------------------------------------
        st.markdown(
            '<div class="sidebar-section-label">Model</div>',
            unsafe_allow_html=True,
        )
        if _HAS_CONFIG and MODEL_ROUTING:
            model_keys = list(MODEL_ROUTING.keys())
            model_labels = {k: MODEL_ROUTING[k].split("/")[-1] if "/" in MODEL_ROUTING[k] else MODEL_ROUTING[k]
                            for k in model_keys}
            current_model = st.session_state.get("selected_model", model_keys[0])
            try:
                current_idx = model_keys.index(current_model)
            except ValueError:
                current_idx = 0
            selected = st.selectbox(
                "Model",
                options=model_keys,
                index=current_idx,
                format_func=lambda k: model_labels.get(k, k),
                key="model_selector",
                label_visibility="collapsed",
            )
            st.session_state.selected_model = selected
        else:
            st.markdown(
                '<div class="model-selector-container"><span style="color:var(--text-muted);font-size:0.82rem;">No models configured</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ---- Recent conversations ----------------------------------------
        st.markdown(
            '<div class="sidebar-section-label">Recent</div>',
            unsafe_allow_html=True,
        )

        if mem:
            try:
                sessions = mem.get_all_sessions()
            except Exception:
                sessions = []

            if sessions:
                for sess in sessions[:12]:
                    sid = sess.get("session_id", "")
                    preview = (sess.get("preview", "") or "(empty)")[:50]
                    ts = _format_timestamp(sess.get("last_timestamp", ""))
                    is_active = sid == st.session_state.session_id

                    if is_active:
                        st.markdown(
                            f'<div class="session-item active">'
                            f'<div class="session-preview">{preview}</div>'
                            f'<div class="session-time">{ts}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # Use a button for clickable sessions
                        if st.button(
                            f"{preview} · {ts}",
                            key=f"sess_{sid}",
                            help=f"Session {sid[:8]}…",
                            use_container_width=True,
                        ):
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
                    '<div style="color:var(--text-muted);font-size:0.8rem;padding:0.3rem 0;">No conversations yet.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:var(--text-muted);font-size:0.8rem;">Memory unavailable.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ---- Scheduled Tasks --------------------------------------------
        with st.expander("Scheduled Tasks", expanded=False):
            if scheduler:
                try:
                    tasks = scheduler.list_tasks()
                except Exception:
                    tasks = []

                task_count = len(tasks)
                st.markdown(
                    f'<div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.5rem;">'
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
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if task_count > 5:
                        st.caption(f"… and {task_count - 5} more")
                    if st.button("View all tasks", key="view_all_tasks"):
                        result = _handle_schedule_list_command()
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["response"], "citations": [], "files": []}
                        )
                        st.rerun()
                else:
                    st.caption("No tasks scheduled. Ask me to schedule something!")
            else:
                st.caption("Scheduler unavailable.")

        # ---- Settings ---------------------------------------------------
        with st.expander("Settings", expanded=False):
            st.markdown(
                '<div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);margin-bottom:0.5rem;">API Keys</div>',
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
                    dot_color = "#1B7F64" if connected else "#EF4444"
                    status_text = "Connected" if connected else "Missing"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;font-size:0.82rem;margin-bottom:0.35rem;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block;margin-right:0.5rem;flex-shrink:0;"></span>'
                        f'<span style="color:var(--text-secondary);flex:1">{label}</span>'
                        f'<span style="color:var(--text-muted);font-size:0.75rem;">{status_text}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:var(--text-muted);font-size:0.8rem;">Config unavailable.</div>',
                    unsafe_allow_html=True,
                )

            if _HAS_CONFIG and MODEL_ROUTING:
                st.markdown(
                    '<div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);margin:0.7rem 0 0.4rem;">Model Routing</div>',
                    unsafe_allow_html=True,
                )
                routing_display = {
                    "orchestration": "Orchestrator",
                    "coding": "Code",
                    "reasoning": "Reasoning",
                    "fast": "Fast",
                    "search_deep": "Deep Search",
                    "search_quick": "Quick Search",
                    "image": "Image",
                }
                for route_key, route_label in routing_display.items():
                    model_id = MODEL_ROUTING.get(route_key, "—")
                    short_model = model_id.split("/")[-1] if "/" in model_id else model_id
                    st.markdown(
                        f'<div style="display:flex;align-items:center;font-size:0.78rem;margin-bottom:0.25rem;">'
                        f'<span style="color:var(--text-secondary);flex:1">{route_label}</span>'
                        f'<code style="font-size:0.72rem;background:#F3F4F6;border-radius:4px;padding:1px 5px;color:#374151;">{short_model}</code>'
                        f'</div>',
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
              <strong>Configuration Missing</strong> — Could not import
              <code>config.py</code>. Make sure all project files are in the
              correct directory.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = validate_config()
    missing = [k for k, v in status.items() if not v]
    if not missing:
        return

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
                f'<span class="setup-item"><code>{env}</code> — {desc} '
                f'<a href="{url}" target="_blank" style="color:var(--accent);">Get key →</a></span>'
            )

    st.markdown(
        f"""
        <div class="setup-banner">
          <strong>Setup Required</strong> — Some API keys are missing:
          {items_html}
          The agent will work in limited mode until keys are configured.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Welcome / Home screen
# ---------------------------------------------------------------------------

def _render_welcome() -> None:
    """Render the Perplexity-style welcome screen with centered input."""
    st.markdown(
        """
        <div class="welcome-container">
          <div class="welcome-brand">perplexity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick action chips
    chips = ["Latest News", "Troubleshoot", "Make a plan", "Analyze", "Write code"]
    chip_cols = st.columns(len(chips))
    for col, chip in zip(chip_cols, chips):
        with col:
            if st.button(chip, key=f"chip_{chip.lower().replace(' ', '_')}", use_container_width=True):
                prompts = {
                    "Latest News": "What are the top news stories right now?",
                    "Troubleshoot": "Help me troubleshoot an issue: ",
                    "Make a plan": "Help me make a detailed plan for: ",
                    "Analyze": "Analyze the following and provide insights: ",
                    "Write code": "Write code to: ",
                }
                st.session_state._quick_input = prompts.get(chip, chip)


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
          This looks like something you might want to schedule.
          Run <em>"{description}…"</em> <strong>{sched_str}</strong>?
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

def _render_file_uploader() -> list[dict]:
    """Render file uploader and return list of uploaded file dicts."""
    uploaded = st.file_uploader(
        "Attach files",
        type=[
            "pdf", "csv", "xlsx", "xls",
            "png", "jpg", "jpeg", "gif", "webp",
            "txt", "md",
            "py", "js", "ts", "jsx", "tsx",
            "html", "css", "json",
            "yaml", "yml",
        ],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    ws = _get_workspace()
    results = []
    if uploaded:
        for uf in uploaded:
            file_dict = {
                "name": uf.name,
                "type": uf.type or mimetypes.guess_type(uf.name)[0] or "application/octet-stream",
                "size": uf.size,
                "content": uf.read(),
            }
            # Persist to workspace if available
            if ws:
                try:
                    saved_path = ws.save_uploaded_file(
                        session_id=st.session_state.session_id,
                        filename=uf.name,
                        content=file_dict["content"],
                    )
                    file_dict["path"] = str(saved_path)
                except Exception:
                    pass
            results.append(file_dict)
    return results


# ---------------------------------------------------------------------------
# Main message processing
# ---------------------------------------------------------------------------

def _process_message(user_message: str, uploaded_files: list[dict]) -> None:
    """Process a user message: handle slash commands or invoke the orchestrator."""
    session_id = st.session_state.session_id
    mem = _get_memory()

    # ---- Save user message to memory ------------------------------------
    if mem:
        try:
            mem.save_message(session_id, "user", user_message)
        except Exception:
            pass

    # ---- Slash command handling -----------------------------------------
    stripped = user_message.strip()
    result: dict | None = None

    if stripped.lower() == "/memory":
        result = _handle_memory_command()
    elif stripped.lower() == "/schedule list":
        result = _handle_schedule_list_command()
    elif stripped.lower().startswith("/schedule remove "):
        task_id = stripped[len("/schedule remove "):].strip()
        result = _handle_schedule_remove_command(task_id)
    else:
        # Schedule detection
        _offer_schedule(user_message)

    if result is None:
        # ---- Invoke orchestrator ----------------------------------------
        if not _HAS_ORCHESTRATOR or run_agent is None:
            result = {
                "response": (
                    "The orchestrator is not available. "
                    "Please ensure all project dependencies are installed."
                ),
                "citations": [],
                "files_created": [],
            }
        else:
            status_placeholder = st.empty()
            status_callback = _make_status_callback(status_placeholder)

            # Show thinking indicator
            status_callback("Searching...")

            try:
                result = _run_async(
                    run_agent(
                        user_message=user_message,
                        session_id=session_id,
                        uploaded_files=uploaded_files,
                        status_callback=status_callback,
                    )
                )
            except Exception as exc:
                tb = traceback.format_exc()
                result = {
                    "response": f"An error occurred while processing your request:\n\n```\n{tb}\n```",
                    "citations": [],
                    "files_created": [],
                }
            finally:
                status_placeholder.empty()

    # ---- Append assistant message to history ----------------------------
    assistant_msg = {
        "role": "assistant",
        "content": result.get("response", ""),
        "citations": result.get("citations") or [],
        "files": result.get("files_created") or [],
    }
    st.session_state.messages.append(assistant_msg)

    # ---- Save assistant message to memory --------------------------------
    if mem:
        try:
            mem.save_message(
                session_id,
                "assistant",
                assistant_msg["content"],
            )
        except Exception:
            pass

        # Detect and store any user preferences from the message
        try:
            mem.detect_and_store_preference(session_id, user_message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Thread rendering
# ---------------------------------------------------------------------------

def _render_thread() -> None:
    """Render the full conversation thread."""
    session_id = st.session_state.session_id
    messages = st.session_state.messages

    for i, msg in enumerate(messages):
        is_latest = (i == len(messages) - 1) and (msg["role"] == "assistant")
        _render_message(msg, i, session_id, is_latest=is_latest)

        # Thread separator between QA pairs
        if (
            msg["role"] == "assistant"
            and i < len(messages) - 1
        ):
            st.markdown('<div class="thread-separator"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()

    # ---- Main content column (centered, max-width 800px) ----------------
    # We use columns to fake centering
    _, col_main, _ = st.columns([1, 6, 1])

    with col_main:
        _render_setup_warning()

        has_messages = bool(st.session_state.messages)

        if not has_messages:
            # Welcome screen
            _render_welcome()
        else:
            # Render conversation thread
            _render_thread()

        # Schedule suggestion banner
        _render_schedule_suggestion()

        st.markdown('<div style="height:6rem;"></div>', unsafe_allow_html=True)

        # ---- Chat input area -------------------------------------------
        st.markdown('<div class="input-area-wrapper">', unsafe_allow_html=True)

        # File uploader toggle
        with st.expander("Attach files", expanded=False):
            uploaded_files = _render_file_uploader()
            st.session_state.pending_uploads = uploaded_files

        # Pre-fill from quick action
        placeholder_text = "Ask anything..."
        default_val = ""
        if st.session_state._quick_input:
            default_val = st.session_state._quick_input
            st.session_state._quick_input = None

        # Chat input
        user_input = st.chat_input(
            placeholder_text,
            key="chat_input",
        )

        # Handle default_val injection (for quick chips / follow-ups)
        # Since st.chat_input doesn't support default values, we use a text_input fallback
        # when a quick_input is set
        if default_val and not user_input:
            with st.form(key="quick_input_form", clear_on_submit=True):
                qi_col, qb_col = st.columns([8, 1])
                with qi_col:
                    quick_text = st.text_input(
                        "Quick input",
                        value=default_val,
                        label_visibility="collapsed",
                        key="quick_input_field",
                    )
                with qb_col:
                    submitted = st.form_submit_button("→")
                if submitted and quick_text.strip():
                    user_input = quick_text.strip()

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Process submitted input ------------------------------------
        if user_input and user_input.strip():
            message = user_input.strip()

            # Add user message to state immediately
            st.session_state.messages.append({
                "role": "user",
                "content": message,
                "citations": [],
                "files": [],
            })

            pending_files = st.session_state.get("pending_uploads") or []

            # Show thinking state
            with st.spinner(""):
                _process_message(message, pending_files)

            # Clear uploads after processing
            st.session_state.pending_uploads = []
            st.rerun()


if __name__ == "__main__":
    main()
