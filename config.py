"""
Configuration module for Personal AI Agent.
Loads environment variables and defines model routing.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
SONAR_API_KEY: str = os.getenv("SONAR_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
SONAR_ENDPOINT: str = "https://api.perplexity.ai/chat/completions"
OPENROUTER_ENDPOINT: str = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Model Routing — change models here without touching any other code
# ---------------------------------------------------------------------------
MODEL_ROUTING: dict[str, str] = {
    "orchestration": os.getenv("MODEL_ORCHESTRATION", "anthropic/claude-sonnet-4"),
    "coding": os.getenv("MODEL_CODING", "anthropic/claude-sonnet-4"),
    "reasoning": os.getenv("MODEL_REASONING", "openai/gpt-4.1"),
    "fast": os.getenv("MODEL_FAST", "openai/gpt-4.1-mini"),
    "long_context": os.getenv("MODEL_LONG_CONTEXT", "openai/gpt-4.1"),
    "image": os.getenv("MODEL_IMAGE", "black-forest-labs/flux-schnell"),
    "search_deep": "sonar-pro",
    "search_quick": "sonar",
}

# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------
DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/agent.db")
WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", "workspaces")
WORKSPACE_RETENTION_DAYS: int = int(os.getenv("WORKSPACE_RETENTION_DAYS", "30"))

# Code execution
CODE_EXECUTION_TIMEOUT: int = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))
MAX_CODE_RETRIES: int = int(os.getenv("MAX_CODE_RETRIES", "3"))
MAX_TASK_RETRIES: int = int(os.getenv("MAX_TASK_RETRIES", "3"))

# Scheduler
SCHEDULER_DB_PATH: str = os.getenv("SCHEDULER_DB_PATH", "data/scheduler.db")

# GitHub
GITHUB_DEFAULT_REPO: str = os.getenv("GITHUB_DEFAULT_REPO", "personal-ai-agent")

# App
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8501"))


def validate_config() -> dict[str, bool]:
    """Return a dict indicating which API keys are present and non-empty."""
    return {
        "sonar_api_key": bool(SONAR_API_KEY),
        "openrouter_api_key": bool(OPENROUTER_API_KEY),
        "github_token": bool(GITHUB_TOKEN),
    }


def get_model(task_type: str) -> str:
    """Get the model identifier for a given task type."""
    return MODEL_ROUTING.get(task_type, MODEL_ROUTING["fast"])
