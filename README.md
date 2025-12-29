# Sports Search Agent

A multi-step AI agent powered by **Flyte v2** that searches the web for sports information and synthesizes answers using LLMs.

## Features

- **Semantic Caching**: Normalizes queries so "who won nba 2023?" and "2023 NBA champion?" hit the same cache
- **Multi-step Orchestration**: Web search (Tavily) → LLM synthesis (Claude)
- **Observability**: Traced LLM calls, HTML reports in Union UI
- **Resilient**: Retry strategies with automatic retries

## Architecture

```
User Query
    ↓
[normalize_query] → Canonical form (Claude)
    ↓
[search_sports_web] → Web search (Tavily) [CACHED]
    ↓
[synthesize_response] → Answer generation (Claude) [CACHED]
    ↓
HTML Report + Response
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Union.ai account ([sign up](https://www.union.ai/))

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd sports-search-agent

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install flyte streamlit
```

### 2. Configure Union.ai

```bash
# Login to Union (opens browser)
flyte login

# This creates .flyte/config.yaml
```

### 3. Set up secrets

You'll need API keys for:
- **Tavily** (web search): https://tavily.com/
- **Anthropic** (Claude LLM): https://console.anthropic.com/

```bash
# Create secrets in Flyte
flyte create secret EWH_TAVILY_API_KEY --value 'your-tavily-key'
flyte create secret NEW_ANTHROPIC_API_KEY --value 'your-anthropic-key'
```

### 4. Run the Streamlit app

```bash
streamlit run src/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Usage

### Via Streamlit UI

1. Open http://localhost:8501
2. Enter a sports question (e.g., "Who won the 2024 NBA Finals?")
3. Click Search
4. View the answer and sources

### Via Python

```python
import flyte
from src.sports_search_agent import sports_search_agent

flyte.init_from_config()

run = flyte.run(sports_search_agent, query="Who won the 2024 Super Bowl?")
print(f"Run URL: {run.url}")

# Get result
import asyncio
result = asyncio.run(run.result())
print(result.answer)
```

### Via CLI

```bash
# Run workflow
python src/sports_search_agent.py

# Monitor logs
flyte get logs <run-name>

# Check actions
flyte get action <run-name>
```

## Project Structure

```
.
├── src/
│   ├── sports_search_agent.py  # Main Flyte workflow
│   └── streamlit_app.py        # Web UI
├── .flyte/
│   └── config.yaml             # Union.ai config (not in git)
├── CLAUDE.md                   # Claude Code instructions
└── README.md
```

## Viewing Results in Union UI

1. Go to the Run URL printed when workflow starts
2. **Actions tab**: See each step (normalize, search, synthesize)
3. **Report tab**: View the HTML report with formatted results
4. **Logs tab**: See detailed execution logs

## Cache Demo

To demonstrate semantic caching:

1. Ask: "who was 2023 nba champion?"
2. Then ask: "who won the nba championship in 2023?"

Both normalize to the same canonical query, so the second request hits the cache for search and synthesis steps.

## Customization

### Change LLM Provider

Edit `src/sports_search_agent.py` and update the model in `llm_normalize` and `llm_synthesize`:

```python
# OpenAI
model="gpt-4o-mini"

# Gemini
model="gemini/gemini-2.0-flash"

# Claude (current)
model="claude-3-haiku-20240307"
```

### Adjust Cache Behavior

```python
@search_env.task(
    cache=flyte.Cache(behavior="auto", version_override="v2"),
)
```

Change `version_override` to invalidate cache when logic changes.

## License

MIT
