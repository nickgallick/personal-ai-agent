"""
LangGraph orchestrator for Personal AI Agent.

This module defines the full StateGraph that drives the agent loop:
  START -> planner -> router -> [executor] -> router -> ... -> synthesizer -> END

The graph handles task decomposition, parallel-dependency-aware routing,
per-executor nodes, error recovery, and final synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Any, Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

import config
import memory as memory_module
import synthesizer as synthesizer_module
from agents import planner, search, coder, reasoner, browser, image_gen, github_agent, file_analyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons (created once, reused across graph invocations)
# ---------------------------------------------------------------------------

_memory_manager: memory_module.MemoryManager | None = None


def _get_memory() -> memory_module.MemoryManager:
    """Return the shared MemoryManager singleton, creating it on first call."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = memory_module.MemoryManager()
    return _memory_manager


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class SubTask(TypedDict):
    type: str            # search | code | reason | browse | image | github | file_analysis
    description: str
    dependencies: list[int]
    status: str          # pending | running | completed | failed
    result: dict | None


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]       # conversation history
    task_plan: list[SubTask]
    current_task_index: int
    files_created: Annotated[list[str], operator.add]   # accumulated created files
    errors: Annotated[list[str], operator.add]          # accumulated errors
    retry_count: int
    final_response: str
    uploaded_files: list[dict]                          # [{path, name, type}]
    session_id: str
    status_message: str                                 # UI progress indicator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_user_message(state: AgentState) -> str:
    """Extract the latest user message text from state."""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _build_context_from_results(task_plan: list[SubTask], up_to_index: int) -> str:
    """
    Concatenate the result content of all completed subtasks before *up_to_index*
    into a single context string for the next executor.
    """
    parts: list[str] = []
    for i, task in enumerate(task_plan):
        if i >= up_to_index:
            break
        if task["status"] == "completed" and task.get("result"):
            result = task["result"]
            content = (
                result.get("content")
                or result.get("response")
                or result.get("output")
                or ""
            )
            if content:
                parts.append(
                    f"[Task {i} — {task['type']}]: {str(content)[:2000]}"
                )
    return "\n\n".join(parts)


def _mark_task(
    task_plan: list[SubTask],
    index: int,
    status: str,
    result: dict | None = None,
) -> list[SubTask]:
    """Return a copy of *task_plan* with the task at *index* updated."""
    updated = [dict(t) for t in task_plan]  # shallow copy of each SubTask
    updated[index]["status"] = status
    if result is not None:
        updated[index]["result"] = result
    return updated  # type: ignore[return-value]


def _all_deps_completed(task: SubTask, task_plan: list[SubTask]) -> bool:
    """Return True if all dependency indices of *task* are in 'completed' status."""
    for dep_idx in task.get("dependencies", []):
        if dep_idx < 0 or dep_idx >= len(task_plan):
            continue  # out-of-range dep — skip safely
        if task_plan[dep_idx]["status"] != "completed":
            return False
    return True


def _find_next_pending(task_plan: list[SubTask]) -> int:
    """
    Find the first pending task whose dependencies are all completed.

    Returns the 0-based index, or -1 if no such task exists (either all
    are done / failed, or remaining tasks have unmet dependencies).
    """
    for idx, task in enumerate(task_plan):
        if task["status"] == "pending" and _all_deps_completed(task, task_plan):
            return idx
    return -1


def _all_tasks_terminal(task_plan: list[SubTask]) -> bool:
    """Return True when every task is in a terminal state (completed or failed)."""
    return all(t["status"] in {"completed", "failed"} for t in task_plan)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def planner_node(state: AgentState) -> dict:
    """
    Decompose the latest user message into a list of SubTasks.

    Side-effect: calls detect_and_store_preference on the user message so
    any stated preferences (name, timezone, etc.) are persisted.
    """
    user_message = _get_user_message(state)
    session_id = state.get("session_id", "default")
    uploaded_files = state.get("uploaded_files", [])

    mem = _get_memory()

    # Persist preference hints from the raw user message
    try:
        mem.detect_and_store_preference(user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_and_store_preference failed: %s", exc)

    # Build context from memory (recent history + stored prefs)
    try:
        context = mem.build_context_prompt(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_context_prompt failed: %s", exc)
        context = ""

    # Call the planner
    try:
        plan = await planner.plan_tasks(
            user_message=user_message,
            context=context,
            uploaded_files=uploaded_files if uploaded_files else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("planner.plan_tasks raised: %s", exc, exc_info=True)
        # Fallback: single reason task
        plan = [
            {
                "type": "reason",
                "description": user_message or "Answer the user's request.",
                "dependencies": [],
                "status": "pending",
                "result": None,
            }
        ]

    task_count = len(plan)
    logger.info("planner_node: produced %d subtask(s)", task_count)

    return {
        "task_plan": plan,
        "status_message": f"Planning complete — {task_count} subtask(s) identified.",
    }


async def router_node(state: AgentState) -> dict:
    """
    Decide which subtask to execute next, or signal synthesis if done.

    Returns current_task_index = -1 to indicate the synthesizer should run.
    """
    task_plan = state.get("task_plan", [])

    # Edge case: empty plan
    if not task_plan:
        logger.warning("router_node: task_plan is empty — routing to synthesizer")
        return {"current_task_index": -1, "status_message": "No tasks to execute."}

    # If all tasks are in terminal states, we're done
    if _all_tasks_terminal(task_plan):
        logger.info("router_node: all tasks terminal — routing to synthesizer")
        return {"current_task_index": -1, "status_message": "All tasks complete."}

    next_idx = _find_next_pending(task_plan)

    if next_idx == -1:
        # No pending task has all dependencies met yet, but not all are terminal.
        # This can happen if a dependency failed. Check if we're stuck.
        # Find any remaining pending task (even with unmet deps) to avoid deadlock.
        stuck_idx = next(
            (i for i, t in enumerate(task_plan) if t["status"] == "pending"),
            -1,
        )
        if stuck_idx == -1:
            # Nothing pending — all must be terminal
            logger.info("router_node: no pending tasks remain — routing to synthesizer")
            return {"current_task_index": -1, "status_message": "All tasks complete."}

        # Force-run the stuck task even though deps may have failed
        logger.warning(
            "router_node: dependency deadlock detected — forcing execution of task %d",
            stuck_idx,
        )
        next_idx = stuck_idx

    task_type = task_plan[next_idx]["type"]
    logger.info("router_node: next task index=%d type=%s", next_idx, task_type)

    # Mark the task as running
    updated_plan = _mark_task(task_plan, next_idx, "running")

    return {
        "current_task_index": next_idx,
        "task_plan": updated_plan,
        "status_message": f"Executing: {task_type}...",
    }


async def search_executor_node(state: AgentState) -> dict:
    """Execute a web search subtask via the Sonar API."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]

    context = _build_context_from_results(task_plan, idx)

    try:
        result = await search.execute_search(
            description=task["description"],
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("search_executor_node error: %s", exc, exc_info=True)
        result = {"content": f"Search failed: {exc}", "citations": [], "success": False}

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (search) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "errors": new_errors,
        "status_message": "Search complete." if status == "completed" else "Search failed.",
    }


async def code_executor_node(state: AgentState) -> dict:
    """Generate and execute Python code for the current subtask."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]
    session_id = state.get("session_id", "default")

    context = _build_context_from_results(task_plan, idx)

    try:
        result = await coder.execute_code(
            description=task["description"],
            context=context,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("code_executor_node error: %s", exc, exc_info=True)
        result = {
            "content": f"Code execution failed: {exc}",
            "files_created": [],
            "code": "",
            "success": False,
            "error": str(exc),
        }

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    # Collect any files the code created
    new_files: list[str] = result.get("files_created") or []
    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(
            f"Task {idx} (code) failed: {result.get('error') or result.get('content', '')}"
        )

    return {
        "task_plan": updated_plan,
        "files_created": new_files,
        "errors": new_errors,
        "status_message": "Code execution complete." if status == "completed" else "Code execution failed.",
    }


async def reason_executor_node(state: AgentState) -> dict:
    """Execute a reasoning subtask via the OpenRouter reasoning model."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]

    context = _build_context_from_results(task_plan, idx)

    try:
        result = await reasoner.execute_reasoning(
            description=task["description"],
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("reason_executor_node error: %s", exc, exc_info=True)
        result = {"content": f"Reasoning failed: {exc}", "success": False}

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (reason) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "errors": new_errors,
        "status_message": "Reasoning complete." if status == "completed" else "Reasoning failed.",
    }


async def browse_executor_node(state: AgentState) -> dict:
    """Execute a browsing subtask using headless Playwright."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]
    session_id = state.get("session_id", "default")

    context = _build_context_from_results(task_plan, idx)

    try:
        result = await browser.execute_browse(
            description=task["description"],
            context=context,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("browse_executor_node error: %s", exc, exc_info=True)
        result = {
            "content": f"Browse failed: {exc}",
            "urls_visited": [],
            "screenshots": [],
            "success": False,
        }

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (browse) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "errors": new_errors,
        "status_message": "Browse complete." if status == "completed" else "Browse failed.",
    }


async def image_executor_node(state: AgentState) -> dict:
    """Generate an image and save it to the session workspace."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]
    session_id = state.get("session_id", "default")

    try:
        result = await image_gen.execute_image_generation(
            description=task["description"],
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("image_executor_node error: %s", exc, exc_info=True)
        result = {"content": f"Image generation failed: {exc}", "files_created": [], "success": False}

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_files: list[str] = result.get("files_created") or []
    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (image) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "files_created": new_files,
        "errors": new_errors,
        "status_message": "Image generation complete." if status == "completed" else "Image generation failed.",
    }


async def github_executor_node(state: AgentState) -> dict:
    """Execute a GitHub operation (create repo, push files, create issue)."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]
    session_id = state.get("session_id", "default")

    # Provide the accumulated files_created list to the GitHub agent
    files_so_far: list[str] = list(state.get("files_created") or [])

    try:
        result = await github_agent.execute_github(
            description=task["description"],
            files_to_push=files_so_far if files_so_far else None,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("github_executor_node error: %s", exc, exc_info=True)
        result = {"content": f"GitHub operation failed: {exc}", "urls": [], "success": False}

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (github) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "errors": new_errors,
        "status_message": "GitHub operation complete." if status == "completed" else "GitHub operation failed.",
    }


async def file_analysis_executor_node(state: AgentState) -> dict:
    """Analyse uploaded files and return an LLM-generated analysis."""
    task_plan = list(state["task_plan"])
    idx = state["current_task_index"]
    task = task_plan[idx]
    session_id = state.get("session_id", "default")

    # Extract absolute paths from the uploaded_files metadata
    uploaded_files = state.get("uploaded_files") or []
    file_paths: list[str] = [f["path"] for f in uploaded_files if f.get("path")]

    try:
        result = await file_analyzer.execute_file_analysis(
            description=task["description"],
            file_paths=file_paths if file_paths else None,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("file_analysis_executor_node error: %s", exc, exc_info=True)
        result = {"content": f"File analysis failed: {exc}", "files_analyzed": [], "success": False}

    status = "completed" if result.get("success") else "failed"
    updated_plan = _mark_task(task_plan, idx, status, result)

    new_errors: list[str] = []
    if not result.get("success"):
        new_errors.append(f"Task {idx} (file_analysis) failed: {result.get('content', '')}")

    return {
        "task_plan": updated_plan,
        "errors": new_errors,
        "status_message": "File analysis complete." if status == "completed" else "File analysis failed.",
    }


async def synthesizer_node(state: AgentState) -> dict:
    """
    Collect all subtask results and synthesize a final response via the
    reasoning model, then persist the assistant message to memory.
    """
    task_plan = state.get("task_plan", [])
    session_id = state.get("session_id", "default")
    user_message = _get_user_message(state)

    # Gather completed tasks (include failed ones too so the synthesizer can
    # acknowledge partial results rather than ignoring them)
    relevant_tasks = [t for t in task_plan if t["status"] in {"completed", "failed"}]

    # Fall back to a simple "no tasks ran" message when the plan was empty
    if not relevant_tasks and not task_plan:
        final_response = "I wasn't able to determine any tasks to run. Please try rephrasing your request."
        _persist_response(session_id, final_response, citations=[])
        return {
            "final_response": final_response,
            "status_message": "Complete.",
        }

    mem = _get_memory()
    try:
        context = mem.build_context_prompt(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesizer_node: build_context_prompt failed: %s", exc)
        context = ""

    try:
        synthesis = await synthesizer_module.synthesize_response(
            task_results=relevant_tasks,
            user_message=user_message,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("synthesizer_node: synthesize_response raised: %s", exc, exc_info=True)
        synthesis = {
            "response": f"I encountered an error while preparing my response: {exc}",
            "citations": [],
            "files_created": [],
        }

    final_response: str = synthesis.get("response", "")
    citations: list[str] = synthesis.get("citations") or []

    # Append citations block if present
    if citations:
        final_response += synthesizer_module.format_citations(citations)

    # Persist assistant message to memory
    _persist_response(session_id, final_response, citations)

    # Collect any files surfaced by the synthesizer (edge case)
    synth_files: list[str] = synthesis.get("files_created") or []

    logger.info(
        "synthesizer_node: response_len=%d citations=%d synth_files=%d",
        len(final_response),
        len(citations),
        len(synth_files),
    )

    return {
        "final_response": final_response,
        "files_created": synth_files,
        "status_message": "Complete.",
    }


async def error_handler_node(state: AgentState) -> dict:
    """
    Handle a failed subtask.

    If the retry budget allows, re-routes via the reason executor as a
    fallback.  Otherwise, generates a graceful error message and terminates
    the graph by routing to the synthesizer.
    """
    task_plan = list(state.get("task_plan", []))
    idx = state.get("current_task_index", -1)
    retry_count = state.get("retry_count", 0)
    max_retries = config.MAX_TASK_RETRIES

    if idx < 0 or idx >= len(task_plan):
        # Nothing to recover — go straight to synthesizer
        error_msg = "An internal error occurred. Please try again."
        return {
            "final_response": error_msg,
            "current_task_index": -1,
            "status_message": "Error — terminating.",
        }

    failed_task = task_plan[idx]
    error_detail = ""
    if failed_task.get("result"):
        error_detail = failed_task["result"].get("content", "")

    logger.warning(
        "error_handler_node: task %d (%s) failed. retry_count=%d/%d",
        idx,
        failed_task["type"],
        retry_count,
        max_retries,
    )

    if retry_count < max_retries:
        # Convert the failed task into a 'reason' fallback and reset to pending
        updated_plan = [dict(t) for t in task_plan]
        updated_plan[idx]["type"] = "reason"
        updated_plan[idx]["status"] = "pending"
        updated_plan[idx]["result"] = None
        # Augment the description with the failure context so the reasoner has info
        original_desc = updated_plan[idx]["description"]
        updated_plan[idx]["description"] = (
            f"{original_desc}\n\n[Note: previous attempt failed with: {error_detail[:300]}]"
        )

        return {
            "task_plan": updated_plan,  # type: ignore[return-value]
            "retry_count": retry_count + 1,
            "current_task_index": -2,  # sentinel: re-route via router
            "status_message": f"Retrying as reasoning task (attempt {retry_count + 1})...",
            "errors": [
                f"Task {idx} ({failed_task['type']}) failed — retrying as reason (attempt {retry_count + 1})"
            ],
        }
    else:
        # Exhausted retries — build a user-friendly message and hand off to synthesizer
        err_msg = (
            f"Task '{failed_task['description'][:100]}' could not be completed "
            f"after {max_retries} attempt(s)."
        )
        if error_detail:
            err_msg += f" Last error: {error_detail[:200]}"

        logger.error("error_handler_node: task %d exhausted retries. %s", idx, err_msg)

        return {
            "task_plan": task_plan,  # type: ignore[return-value]
            "current_task_index": -1,  # force synthesizer
            "final_response": "",
            "status_message": "Error — proceeding with partial results.",
            "errors": [err_msg],
        }


# ---------------------------------------------------------------------------
# Routing / conditional edge functions
# ---------------------------------------------------------------------------

def route_task(state: AgentState) -> str:
    """
    Route to the appropriate executor based on current_task_index.

    Special values:
      -1  -> synthesizer (all done)
      -2  -> router     (error_handler requested a re-route)
    """
    idx = state.get("current_task_index", -1)

    if idx == -1:
        return "synthesizer"

    if idx == -2:
        # error_handler wants us to go back through the router
        return "router"

    task_plan = state.get("task_plan", [])
    if not task_plan or idx >= len(task_plan):
        return "synthesizer"

    task = task_plan[idx]
    type_to_node: dict[str, str] = {
        "search": "search_executor",
        "code": "code_executor",
        "reason": "reason_executor",
        "browse": "browse_executor",
        "image": "image_executor",
        "github": "github_executor",
        "file_analysis": "file_analysis_executor",
    }
    return type_to_node.get(task["type"], "reason_executor")


def route_after_executor(state: AgentState) -> str:
    """
    After an executor completes, decide whether to handle an error or loop
    back to the router.
    """
    task_plan = state.get("task_plan", [])
    idx = state.get("current_task_index", -1)

    if idx < 0 or idx >= len(task_plan):
        return "router"

    if task_plan[idx]["status"] == "failed":
        return "error_handler"

    return "router"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _persist_response(session_id: str, response: str, citations: list[str]) -> None:
    """Save the assistant's final response to memory (best-effort)."""
    try:
        mem = _get_memory()
        mem.save_message(
            session_id=session_id,
            role="assistant",
            content=response,
            citations=citations if citations else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("_persist_response: failed to save to memory: %s", exc)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns the compiled graph object ready for invocation.
    """
    graph = StateGraph(AgentState)

    # -----------------------------------------------------------------------
    # Register nodes
    # -----------------------------------------------------------------------
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("search_executor", search_executor_node)
    graph.add_node("code_executor", code_executor_node)
    graph.add_node("reason_executor", reason_executor_node)
    graph.add_node("browse_executor", browse_executor_node)
    graph.add_node("image_executor", image_executor_node)
    graph.add_node("github_executor", github_executor_node)
    graph.add_node("file_analysis_executor", file_analysis_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("error_handler", error_handler_node)

    # -----------------------------------------------------------------------
    # Edges
    # -----------------------------------------------------------------------

    # Entry point
    graph.add_edge(START, "planner")

    # Planner always goes to router
    graph.add_edge("planner", "router")

    # Router uses conditional dispatch to the right executor or synthesizer
    graph.add_conditional_edges(
        "router",
        route_task,
        {
            "search_executor": "search_executor",
            "code_executor": "code_executor",
            "reason_executor": "reason_executor",
            "browse_executor": "browse_executor",
            "image_executor": "image_executor",
            "github_executor": "github_executor",
            "file_analysis_executor": "file_analysis_executor",
            "synthesizer": "synthesizer",
            "router": "router",  # used when error_handler asks to re-route (idx == -2)
        },
    )

    # Each executor conditionally goes to error_handler (on failure) or router (on success)
    for executor_name in [
        "search_executor",
        "code_executor",
        "reason_executor",
        "browse_executor",
        "image_executor",
        "github_executor",
        "file_analysis_executor",
    ]:
        graph.add_conditional_edges(
            executor_name,
            route_after_executor,
            {
                "router": "router",
                "error_handler": "error_handler",
            },
        )

    # error_handler routes back to router (retry) or synthesizer (give up)
    graph.add_conditional_edges(
        "error_handler",
        route_task,
        {
            "router": "router",
            "synthesizer": "synthesizer",
            # Handle the re-route sentinel used by error_handler
            "search_executor": "search_executor",
            "code_executor": "code_executor",
            "reason_executor": "reason_executor",
            "browse_executor": "browse_executor",
            "image_executor": "image_executor",
            "github_executor": "github_executor",
            "file_analysis_executor": "file_analysis_executor",
        },
    )

    # Synthesizer is the terminal node
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Compiled graph singleton
# ---------------------------------------------------------------------------

_compiled_graph: Any | None = None
_graph_lock = asyncio.Lock()


async def _get_compiled_graph() -> Any:
    """Return the shared compiled graph, building it on first call (thread-safe)."""
    global _compiled_graph
    async with _graph_lock:
        if _compiled_graph is None:
            _compiled_graph = build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    user_message: str,
    session_id: str,
    uploaded_files: list[dict] | None = None,
    status_callback: Any | None = None,
) -> dict:
    """
    Main entry point called by the Streamlit UI (or any other caller).

    Parameters
    ----------
    user_message:
        The raw text typed by the user.
    session_id:
        Unique session identifier for memory / workspace scoping.
    uploaded_files:
        Optional list of dicts, each with keys ``path``, ``name``, ``type``.
    status_callback:
        Optional callable(str) that receives status_message updates so the
        UI can display a live progress indicator.  Called after every node
        that updates the status_message field.

    Returns
    -------
    dict
        {
            "response":      str,        # final synthesized response
            "citations":     list[str],  # de-duplicated source URLs
            "files_created": list[str],  # absolute paths to any created files
        }
    """
    logger.info(
        "run_agent: session=%s message=%r files=%d",
        session_id,
        user_message[:100],
        len(uploaded_files or []),
    )

    # Persist the user's message to memory
    mem = _get_memory()
    try:
        mem.save_message(session_id=session_id, role="user", content=user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_agent: failed to save user message to memory: %s", exc)

    # Build the initial state
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_message}],
        "task_plan": [],
        "current_task_index": 0,
        "files_created": [],
        "errors": [],
        "retry_count": 0,
        "final_response": "",
        "uploaded_files": uploaded_files or [],
        "session_id": session_id,
        "status_message": "Starting...",
    }

    compiled = await _get_compiled_graph()

    # Invoke the graph — LangGraph runs the full loop synchronously within
    # ainvoke, but each async node is awaited correctly.
    try:
        final_state: AgentState = await compiled.ainvoke(
            initial_state,
            config={"recursion_limit": 100},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("run_agent: graph invocation failed: %s", exc, exc_info=True)
        error_response = (
            f"I'm sorry, I encountered an unexpected error while processing your request. "
            f"Please try again. (Error: {type(exc).__name__}: {exc})"
        )
        # Attempt to persist the error response
        _persist_response(session_id, error_response, citations=[])
        return {
            "response": error_response,
            "citations": [],
            "files_created": [],
        }

    # Fire the final status callback if provided
    if status_callback is not None:
        try:
            status_callback(final_state.get("status_message", "Complete."))
        except Exception as exc:  # noqa: BLE001
            logger.debug("status_callback raised (ignored): %s", exc)

    # Extract return values from the final state
    final_response: str = final_state.get("final_response", "")
    files_created: list[str] = list(dict.fromkeys(final_state.get("files_created") or []))

    # Pull citations from the task plan results (de-duplicated)
    citations: list[str] = _extract_citations_from_plan(
        final_state.get("task_plan", [])
    )

    if not final_response:
        final_response = "I completed the requested tasks but was unable to generate a summary response."
        logger.warning("run_agent: final_response was empty after graph completion")

    logger.info(
        "run_agent complete: response_len=%d citations=%d files_created=%d",
        len(final_response),
        len(citations),
        len(files_created),
    )

    return {
        "response": final_response,
        "citations": citations,
        "files_created": files_created,
    }


def _extract_citations_from_plan(task_plan: list[SubTask]) -> list[str]:
    """
    Collect and de-duplicate citation URLs from all completed task results.
    """
    seen: set[str] = set()
    citations: list[str] = []
    for task in task_plan:
        if task.get("status") != "completed":
            continue
        result = task.get("result") or {}
        raw = result.get("citations") or []
        if isinstance(raw, list):
            for url in raw:
                if isinstance(url, str) and url.strip() and url not in seen:
                    seen.add(url)
                    citations.append(url)
    return citations
