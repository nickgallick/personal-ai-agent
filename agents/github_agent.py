"""
GitHub operations executor agent for Personal AI Agent.

Uses PyGitHub (synchronous) wrapped in ``asyncio.to_thread()`` to avoid
blocking the event loop.  Supports three operations:

* **create_repo** — Create a new GitHub repository.
* **push_files**  — Create or update files in an existing (or newly created) repo.
* **create_issue** — Create an issue on a repository.

The operation to perform is determined by asking the fast OpenRouter model
to parse the task description.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error
    from workspace import WorkspaceManager
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.api_clients import OpenRouterClient
    from utils.error_handling import format_error
    from workspace import WorkspaceManager

logger = logging.getLogger(__name__)

_Result = dict[str, Any]

_PARSE_SYSTEM_PROMPT = """
You are a GitHub task parser. Given a natural-language task description,
identify the GitHub operation to perform and extract relevant fields.

Return a JSON object with the following structure:
{
  "operation": "<create_repo | push_files | create_issue>",
  "repo_name": "<name of the repository, snake_case or kebab-case>",
  "private": <true | false>,
  "description": "<optional repository or issue description>",
  "issue_title": "<title for the issue (create_issue only)>",
  "issue_body": "<body for the issue (create_issue only)>",
  "commit_message": "<commit message for push_files, or null>",
  "branch": "<branch name for push_files, defaults to 'main'>"
}

Rules:
- If the task is about creating a repository, set operation = "create_repo".
- If the task is about pushing, uploading, committing, or saving files, set operation = "push_files".
- If the task is about creating, filing, or opening an issue or bug report, set operation = "create_issue".
- repo_name must be lowercase with hyphens or underscores only (no spaces).
- If no repo name is mentioned, use the default placeholder "__default__".
- Return ONLY valid JSON — no markdown, no explanation.
""".strip()


async def _parse_operation(description: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    fallback: dict[str, Any] = {
        "operation": "push_files",
        "repo_name": "__default__",
        "private": True,
        "description": description,
        "commit_message": "Update via Personal AI Agent",
        "branch": "main",
        "issue_title": "",
        "issue_body": "",
    }
    try:
        client = OpenRouterClient()
        reply = await client.fast(messages)
        reply_clean = reply.strip()
        if reply_clean.startswith("```"):
            reply_clean = re.sub(r"^```[a-zA-Z]*\n?", "", reply_clean)
            reply_clean = re.sub(r"\n?```$", "", reply_clean).strip()
        parsed = json.loads(reply_clean)
        if isinstance(parsed, dict) and "operation" in parsed:
            for key, val in fallback.items():
                parsed.setdefault(key, val)
            return parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("Operation parsing failed: %s — using fallback", exc)
    return fallback


def _gh_create_repo(token, repo_name, description="", private=True):
    from github import Github, GithubException  # type: ignore
    g = Github(token)
    user = g.get_user()
    try:
        repo = user.create_repo(name=repo_name, description=description, private=private, auto_init=True)
        return {"repo_name": repo.full_name, "clone_url": repo.clone_url, "html_url": repo.html_url}
    except GithubException as exc:
        if exc.status == 422:
            repo = user.get_repo(repo_name)
            return {"repo_name": repo.full_name, "clone_url": repo.clone_url, "html_url": repo.html_url}
        raise


def _gh_get_or_create_repo(token, repo_name, description="", private=True):
    from github import Github, GithubException  # type: ignore
    g = Github(token)
    user = g.get_user()
    try:
        return user.get_repo(repo_name)
    except GithubException:
        pass
    try:
        return user.create_repo(name=repo_name, description=description, private=private, auto_init=True)
    except GithubException as exc:
        if exc.status == 422:
            return user.get_repo(repo_name)
        raise


def _gh_push_files(token, repo_name, file_paths, commit_message="Update via Personal AI Agent", branch="main", repo_description="", private=True):
    from github import GithubException  # type: ignore
    repo = _gh_get_or_create_repo(token, repo_name, repo_description, private)
    actual_branch = repo.default_branch or branch
    commit_url: str = ""
    committed: list[str] = []
    for local_path_str in file_paths:
        local_path = Path(local_path_str)
        if not local_path.is_file():
            continue
        file_content = local_path.read_bytes()
        remote_path = local_path.name
        try:
            existing = repo.get_contents(remote_path, ref=actual_branch)
            result = repo.update_file(path=remote_path, message=commit_message, content=file_content, sha=existing.sha, branch=actual_branch)
        except GithubException as exc:
            if exc.status == 404:
                result = repo.create_file(path=remote_path, message=commit_message, content=file_content, branch=actual_branch)
            else:
                raise
        commit_url = result["commit"].html_url
        committed.append(remote_path)
    return {"committed_files": committed, "commit_url": commit_url, "repo_url": repo.html_url}


def _gh_create_issue(token, repo_name, title, body="", repo_description="", private=True):
    repo = _gh_get_or_create_repo(token, repo_name, repo_description, private)
    issue = repo.create_issue(title=title, body=body)
    return {"issue_number": str(issue.number), "issue_url": issue.html_url}


def _collect_workspace_files(session_id: str) -> list[str]:
    wm = WorkspaceManager()
    file_records = wm.list_files(session_id)
    return [rec["path"] for rec in file_records]


async def execute_github(description: str, files_to_push: list[str] | None = None, session_id: str = "default") -> _Result:
    if not config.GITHUB_TOKEN:
        return {"content": "GitHub token is not configured.", "urls": [], "success": False}

    try:
        import github  # noqa: F401
    except ImportError:
        return {"content": "PyGitHub is not installed. Run: pip install PyGithub", "urls": [], "success": False}

    try:
        parsed = await _parse_operation(description)
    except Exception as exc:  # noqa: BLE001
        return {"content": f"Failed to parse GitHub operation: {format_error(exc)}", "urls": [], "success": False}

    operation: str = parsed.get("operation", "push_files")
    raw_repo_name: str = parsed.get("repo_name", "__default__")
    repo_name: str = config.GITHUB_DEFAULT_REPO if raw_repo_name in ("__default__", "", None) else raw_repo_name
    private: bool = bool(parsed.get("private", True))
    repo_description: str = parsed.get("description", "")
    commit_message: str = parsed.get("commit_message", None) or "Update via Personal AI Agent"
    branch: str = parsed.get("branch", None) or "main"
    issue_title: str = parsed.get("issue_title", "") or description[:100]
    issue_body: str = parsed.get("issue_body", "")

    try:
        if operation == "create_repo":
            result = await asyncio.to_thread(_gh_create_repo, config.GITHUB_TOKEN, repo_name, repo_description, private)
            content = f"Repository '{result['repo_name']}' created successfully.\nURL: {result['html_url']}\nClone: {result['clone_url']}"
            return {"content": content, "urls": [result["html_url"]], "success": True}

        elif operation == "push_files":
            local_files: list[str] = list(files_to_push or [])
            if not local_files:
                local_files = _collect_workspace_files(session_id)
            if not local_files:
                return {"content": "No files to push.", "urls": [], "success": False}
            result = await asyncio.to_thread(_gh_push_files, config.GITHUB_TOKEN, repo_name, local_files, commit_message, branch, repo_description, private)
            committed = result.get("committed_files", [])
            commit_url = result.get("commit_url", "")
            repo_url = result.get("repo_url", "")
            lines = [f"Pushed {len(committed)} file(s) to '{repo_name}'.", f"Files: {', '.join(committed) if committed else 'none'}"]
            if commit_url:
                lines.append(f"Commit: {commit_url}")
            if repo_url:
                lines.append(f"Repository: {repo_url}")
            return {"content": "\n".join(lines), "urls": [u for u in [commit_url, repo_url] if u], "success": True}

        elif operation == "create_issue":
            result = await asyncio.to_thread(_gh_create_issue, config.GITHUB_TOKEN, repo_name, issue_title, issue_body, repo_description, private)
            content = f"Issue #{result['issue_number']} created on '{repo_name}'.\nURL: {result['issue_url']}"
            return {"content": content, "urls": [result["issue_url"]], "success": True}

        else:
            return {"content": f"Unknown GitHub operation '{operation}'.", "urls": [], "success": False}

    except Exception as exc:  # noqa: BLE001
        error_msg = f"GitHub operation '{operation}' failed: {format_error(exc)}"
        logger.error(error_msg, exc_info=True)
        return {"content": error_msg, "urls": [], "success": False}
