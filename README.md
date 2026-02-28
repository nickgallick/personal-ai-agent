# Personal AI Agent

A production-ready personal AI agent that replicates the core functionality of Perplexity Computer. It uses a LangGraph-powered orchestrator to decompose complex requests into subtasks, routing each to the best-suited model and tool.

## Features

- **Multi-model orchestration** — LangGraph StateGraph routes tasks to specialized executors
- **Web search with citations** — Perplexity Sonar Pro API returns sourced, cited answers
- **Code generation and execution** — Writes, runs, and self-corrects code (up to 3 retries)
- **Web browsing** — Playwright-powered headless Chromium for navigating real pages
- **Image generation** — OpenRouter image models (Flux, DALL·E, etc.)
- **File analysis** — Upload and analyze PDFs, CSVs, Excel files, images, code files
- **GitHub integration** — Create repos, push code, create issues
- **Scheduled tasks** — APScheduler with SQLite persistence for recurring jobs
- **Persistent memory** — SQLite-backed conversations, preferences, and task history
- **Dark-themed UI** — Streamlit chat interface styled like Perplexity

## Architecture

```
User Message
     │
     ▼
  Planner  ──→  Decomposes into subtasks
     │
     ▼
   Router  ──→  Finds next ready subtask
     │
     ├──→ Search Executor   (Sonar Pro API)
     ├──→ Code Executor     (Claude via OpenRouter + subprocess)
     ├──→ Reason Executor   (GPT via OpenRouter)
     ├──→ Browse Executor   (Playwright)
     ├──→ Image Executor    (Flux via OpenRouter)
     ├──→ GitHub Executor   (PyGitHub)
     └──→ File Analyzer     (PyPDF2, pandas, etc.)
     │
     ▼
  Synthesizer  ──→  Combines all results with citations
     │
     ▼
  Response to User
```

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/personal-ai-agent.git
cd personal-ai-agent
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (see [Getting API Keys](#getting-api-keys) below).

### 3. Run with Docker (recommended)

```bash
docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run without Docker (local Python)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```

---

## Getting API Keys

### Perplexity Sonar API (required)

1. Go to [https://www.perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
2. Sign in or create an account
3. Generate an API key
4. Add to `.env` as `SONAR_API_KEY=pplx-...`

### OpenRouter API (required)

1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign in or create an account
3. Create a new API key
4. Add credits to your account (pay-as-you-go)
5. Add to `.env` as `OPENROUTER_API_KEY=sk-or-...`

### GitHub Token (optional)

Only needed if you want the agent to create repos and push code.

1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Select scopes: `repo`, `workflow`
4. Add to `.env` as `GITHUB_TOKEN=ghp_...`

---

## Usage

### Chat basics

Type your question or request in the chat input. The agent handles everything from simple questions to complex multi-step projects.

**Simple question:**
> "What is the current market cap of Apple?"

**Complex project:**
> "Research the top 5 JavaScript frameworks, compare their performance benchmarks, create a summary table as code, and push the results to my GitHub."

### File upload

Click the file upload area above the chat input or drag and drop files. Supported formats:

- Documents: PDF, TXT, MD, JSON, YAML
- Data: CSV, XLSX, XLS
- Images: PNG, JPG, JPEG, GIF
- Code: PY, JS, TS, JSX, TSX, HTML, CSS

Then ask a question about the uploaded files:
> "Summarize this PDF"
> "Analyze this CSV and find trends"

### Viewing generated files

When the agent creates files (code, images, reports), they appear as downloadable cards below the response. Images are displayed inline.

### Scheduled tasks

Use natural language to schedule recurring tasks:

> "Every Monday at 8am, research the latest AI news and summarize it"
> "Daily at 9am, check the price of Bitcoin"

**Commands:**
- `/schedule list` — View all scheduled tasks
- `/schedule remove <task_id>` — Remove a scheduled task

### Memory

The agent automatically remembers your preferences when you say things like:
> "Remember that I prefer Python over JavaScript"
> "My name is Alex"
> "Always use dark mode examples"

**Commands:**
- `/memory` — View all remembered preferences and recent tasks

---

## Model Routing

Models are configured in `config.py` via the `MODEL_ROUTING` dictionary. Change them without touching any other code:

| Task Type | Default Model | Used For |
|-----------|---------------|----------|
| Orchestration | `anthropic/claude-sonnet-4` | Task planning and decomposition |
| Coding | `anthropic/claude-sonnet-4` | Code generation and fixing |
| Reasoning | `openai/gpt-4.1` | Analysis, synthesis, general reasoning |
| Fast | `openai/gpt-4.1-mini` | Quick lookups, parsing |
| Long Context | `openai/gpt-4.1` | Large document processing |
| Image | `black-forest-labs/flux-schnell` | Image generation |
| Deep Search | `sonar-pro` | In-depth web search |
| Quick Search | `sonar` | Fast web lookups |

You can override any model via environment variables:
```bash
MODEL_ORCHESTRATION=anthropic/claude-sonnet-4
MODEL_CODING=anthropic/claude-sonnet-4
MODEL_REASONING=openai/gpt-4.1
MODEL_FAST=openai/gpt-4.1-mini
MODEL_IMAGE=black-forest-labs/flux-schnell
```

---

## Deployment

### Docker Compose (any server)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove data
docker-compose down -v
```

Data persists in Docker volumes (`agent-data` for databases, `agent-workspaces` for files).

### Fly.io

1. Install the Fly CLI: [https://fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)

2. Sign in:
```bash
fly auth login
```

3. Launch the app:
```bash
fly launch --no-deploy
```

4. Create a persistent volume:
```bash
fly volumes create agent_data --size 1 --region ord
```

5. Set your secrets:
```bash
fly secrets set SONAR_API_KEY=pplx-your-key-here
fly secrets set OPENROUTER_API_KEY=sk-or-your-key-here
fly secrets set GITHUB_TOKEN=ghp_your-token-here  # optional
```

6. Deploy:
```bash
fly deploy
```

7. Open the app:
```bash
fly open
```

### Railway

1. Push your code to GitHub

2. Go to [https://railway.com](https://railway.com) and create a new project

3. Select **Deploy from GitHub repo** and connect your repo

4. Add environment variables in the Railway dashboard:
   - `SONAR_API_KEY`
   - `OPENROUTER_API_KEY`
   - `GITHUB_TOKEN` (optional)

5. Add a persistent volume:
   - Go to your service settings
   - Under **Volumes**, add a volume mounted at `/data`

6. Railway will auto-detect the Dockerfile and deploy

---

## Project Structure

```
personal-agent/
├── app.py                    # Streamlit UI entry point
├── config.py                 # Configuration, model routing, env loading
├── orchestrator.py           # LangGraph StateGraph definition
├── agents/
│   ├── __init__.py
│   ├── planner.py            # Task planning agent
│   ├── search.py             # Sonar API search executor
│   ├── coder.py              # Code generation + execution + self-correction
│   ├── reasoner.py           # General reasoning executor
│   ├── browser.py            # Playwright web browsing executor
│   ├── image_gen.py          # Image generation executor
│   ├── github_agent.py       # GitHub operations executor
│   └── file_analyzer.py      # File upload analysis executor
├── memory.py                 # SQLite conversation memory + preferences
├── scheduler.py              # APScheduler task scheduling
├── workspace.py              # File/workspace management
├── synthesizer.py            # Response synthesis and citation formatting
├── utils/
│   ├── __init__.py
│   ├── api_clients.py        # Sonar and OpenRouter API client wrappers
│   └── error_handling.py     # Error handling and fallback logic
├── styles.css                # Custom dark theme CSS for Streamlit
├── .env.example              # Template with all required env vars
├── requirements.txt          # Python dependencies with pinned versions
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Docker compose with persistence volumes
├── fly.toml                  # Fly.io deployment config
├── railway.json              # Railway deployment config
├── README.md                 # This file
└── .gitignore                # Ignore env, workspaces, db, etc.
```

---

## Troubleshooting

### Missing or invalid API keys

The app shows status indicators in the sidebar settings. Green = connected, red = missing.

- **Sonar API key**: Must start with `pplx-`. Get one at [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
- **OpenRouter API key**: Must start with `sk-or-`. Get one at [openrouter.ai/keys](https://openrouter.ai/keys)
- **GitHub token**: Must start with `ghp_` or `github_pat_`. Get one at [github.com/settings/tokens](https://github.com/settings/tokens)

### Docker issues

**Port already in use:**
```bash
# Change the port in .env or docker-compose.yml
APP_PORT=8502 docker-compose up
```

**Permission denied on volumes:**
```bash
# Fix ownership
sudo chown -R $USER:$USER ./data ./workspaces
```

**Build fails:**
```bash
# Clean rebuild
docker-compose build --no-cache
```

### Playwright issues

**Browser not installed:**
```bash
playwright install chromium
```

**Missing system dependencies (Linux):**
```bash
playwright install-deps chromium
```

**Browsing times out:**
- Some sites block headless browsers. The agent handles this gracefully and reports the error.

### Database issues

**Corrupted database:**
```bash
# Delete and let the app recreate it
rm data/agent.db data/scheduler.db
```

**Migration:** The app auto-creates tables on startup. No manual migration is needed.

---

## Cost Estimates

| Usage Level | Sonar API | OpenRouter | Total |
|-------------|-----------|------------|-------|
| Light (50 queries/day) | ~$5/mo | ~$5/mo | ~$10/mo |
| Medium (200 queries/day) | ~$15/mo | ~$15/mo | ~$30/mo |
| Heavy (500+ queries/day) | ~$30/mo | ~$30/mo | ~$60/mo |

Cloud hosting adds $0–10/month depending on provider and plan.

---

## License

This project is for personal use. Feel free to modify and extend it as you wish.
