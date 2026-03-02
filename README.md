# Deep Research Agent — Built in Raw Python

> A production-ready Deep Research Agent built without any agent framework — giving you full visibility and control over every layer of the system.

---

## Quickstart

**You need:** Python 3.11+, a [Tavily API key](https://app.tavily.com) (free), and an Azure AI Foundry project with 3 models deployed.

```bash
# 1. Clone and enter the project
cd research-agent

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]" 2>/dev/null || pip install openai pydantic-settings httpx trafilatura streamlit pytest

# 3. Set up your credentials
cp .env.example .env
# → open .env and fill in: FOUNDRY_ENDPOINT, FOUNDRY_API_KEY, SMART_MODEL, CHEAP_MODEL, EMBEDDING_MODEL, TAVILY_API_KEY

# 4. Verify everything works (no API keys needed)
python -m pytest tests/unit/ -q
# Expected: 340 passed

# 5a. Run via Python
python - <<'EOF'
from agent.loop import run_research
state = run_research(
    "What are the latest breakthroughs in solid-state battery technology?",
    on_progress=print,
)
print(state.final_report)
EOF

# 5b. Or run the Streamlit web UI
streamlit run app.py
# → opens http://localhost:8501
```

**Where to get your keys:**
- **Azure AI Foundry** — [portal.azure.com](https://portal.azure.com) → AI Foundry → your project → Settings → API keys. Deploy `gpt-4o`, `gpt-4o-mini`, and `text-embedding-3-small`.
- **Tavily** — free at [app.tavily.com](https://app.tavily.com) → API Keys → Create key. 1,000 free credits/month (~80 research runs).

---

## What This Project Is

A system that takes a plain English question, plans a research strategy, searches the web, reads pages, compresses context, reflects on gaps, searches again, and synthesizes a cited structured report.

Built entirely in raw Python. No LangChain. No LangGraph. No GPT-Researcher. Every component — the research loop, the reflect-search iteration, the context compression, the synthesis pipeline — is written from scratch. No framework hiding what's actually happening.

---

## What This Covers

| Concept | How it works here |
|---|---|
| Planning | Planner decomposes the question into targeted subqueries |
| Tools | search, fetch, extract, summarize — each a clean isolated module |
| Retry / backoff | Exponential backoff on all network calls (Tavily, Jina, trafilatura) |
| Context management | Top-K most relevant pages via embedding-based cosine similarity ranking |
| Stopping condition | Reflector decides when coverage is sufficient |
| Output | Detailed multi-section Markdown report (1000-2000 words) with inline `[N]` citations |
| Cost control | Hard cap — cost checked before every subquery and round |
| Observability | Structured logging + span-based tracing for every pipeline step |
| Guardrails | Input validation, SSRF-safe URL checks, citation bounds, query dedup |
| Evaluations | Keyword recall, citation accuracy, citation density, composite score |
| Web UI | Streamlit app — Ask tab, Dashboard tab, Traces tab |

---

## Architecture

```
User Question
        │
        ▼
┌──────────────────────────────────────┐
│  Guardrails: validate_query()        │
│  Reject empty / too short / too long │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Planner                             │
│  LLM decomposes into 3-5 subqueries  │
│  deduplicate_queries() removes dupes │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Research Loop  (up to max_research_rounds)              │
│                                                          │
│  For each subquery:                                      │
│    Researcher:                                           │
│      tavily.search(query) → URLs + page content          │
│      retry_with_backoff() on timeout/transient errors    │
│      is_safe_url() check before every fetch              │
│      cheap_llm.summarize(page) → detailed bullets        │
│      → PageSummary added to ResearchState                │
│                                                          │
│  Reflector: "What gaps remain? Follow-up query?"         │
│    → gap found → next round with follow-up query         │
│    → no gap OR max rounds OR source/cost cap → stop      │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Synthesizer (two-shot)                                  │
│  Rank summaries by cosine similarity to original query   │
│  Shot 1: outline (5-8 section headings)                  │
│  Shot 2: full detailed report, [N] citations             │
│  check_citation_bounds() validates                       │
│  References section auto-appended                        │
└──────────────────────────────────────────────────────────┘
        │
        ▼
    ResearchState returned
    (final_report, sources, cost_usd, rounds, status, spans)
```

---

## Production Hardening

The following production features are implemented and tested:

### Retry with exponential backoff
All external network calls retry on transient failures (timeouts). Non-retryable errors (auth failures, 404s) fail fast — no wasted retries.

```
Attempt 1 → fails (timeout) → wait 1s
Attempt 2 → fails (timeout) → wait 2s
Attempt 3 → fails → raise (search() catches → returns [])
```

### Never raises — always returns a state
`run_research()` catches all exceptions, records them in `state.errors`, and returns a `ResearchState` with `status=FAILED`. Your calling code never needs a try/except.

### Hard stopping conditions
The loop stops immediately on: cost cap hit, source cap hit, max rounds reached, or unexpected exception. All three conditions break both the inner subquery loop and the outer rounds loop.

### Embedding-based summary selection
The synthesizer ranks all collected page summaries by cosine similarity to the original query and selects the most relevant ones — not just the first ones collected. Gap-fill summaries from later rounds get a fair shot at entering the report.

### Structured logging
Every module uses `get_logger(__name__)`. All production `print()` calls are replaced. Output is filterable by level and module name:

```
[2026-03-02 14:32:01] research_agent.agent.loop INFO Planning complete — 4 search angles
[2026-03-02 14:32:04] research_agent.tools.retry WARNING Retry 1/2 for _jina_get after TimeoutException (backoff 1.0s)
[2026-03-02 14:32:06] research_agent.agent.researcher WARNING Fetch failed for https://..., skipping
```

### No silent failures
Every `except` block logs a warning before returning a safe fallback. Cost tracking failures, LLM errors, and embedding failures are all visible in logs.

---

## Project Structure

```
research-agent/
│
├── agent/
│   ├── state.py          # ResearchState dataclass + ResearchStatus enum
│   ├── planner.py        # query decomposition — one question → N subqueries
│   ├── researcher.py     # search + fetch + summarize for one subquery
│   ├── reflector.py      # gap detection — should we search again?
│   ├── synthesizer.py    # two-shot outline → report; embedding-based ranking
│   ├── guardrails.py     # validate_query, is_safe_url, check_citation_bounds, dedup
│   └── loop.py           # orchestrates the full pipeline; never raises
│
├── prompts/
│   ├── planner.py        # DECOMPOSE_PROMPT
│   ├── reflector.py      # REFLECT_PROMPT
│   └── synthesizer.py    # OUTLINE_PROMPT, REPORT_PROMPT (1000-2000 word target)
│
├── tools/
│   ├── search.py         # Tavily API wrapper → list[SearchResult]; retry on timeout
│   ├── fetch.py          # Jina Reader + trafilatura → FetchResult; retry on timeout
│   ├── extract.py        # HTML → clean text, truncation helpers
│   └── retry.py          # retry_with_backoff() decorator — exponential backoff
│
├── llm/
│   ├── client.py         # Azure AI Foundry wrapper: generate, generate_cheap, embed
│   │                     # tracks token usage + cost per call; logs tracking failures
│   └── utils.py          # extract_response_text() — shared response text extractor
│
├── observability/
│   ├── tracer.py         # Span + Trace dataclasses; context-manager instrumentation
│   ├── dashboard.py      # load_traces, summary_stats, latency_stats, cost_stats
│   └── logging.py        # get_logger() — structured logging for all modules
│
├── evals/
│   ├── dataset.py        # 5 eval questions with ground-truth keywords
│   ├── metrics.py        # citation_accuracy, citation_density, keyword_coverage,
│   │                     # source_quality, run_score (composite)
│   └── runner.py         # CLI runner — coloured output, summary table, JSON export
│
├── tests/
│   ├── unit/             # 340 tests, no API calls required
│   │   ├── test_state.py
│   │   ├── test_planner.py
│   │   ├── test_reflector.py
│   │   ├── test_researcher.py
│   │   ├── test_loop.py
│   │   ├── test_guardrails.py
│   │   ├── test_synthesizer.py
│   │   ├── test_tracer.py
│   │   ├── test_dashboard.py
│   │   ├── test_fetch.py
│   │   ├── test_search.py
│   │   ├── test_extract.py
│   │   ├── test_retry.py
│   │   └── test_llm_utils.py
│   └── integration/      # end-to-end tests (needs real API keys)
│
├── app.py                # Streamlit UI — Ask, Dashboard, Traces tabs
├── config.py             # pydantic-settings — all env vars typed + validated
├── pyproject.toml
└── .env.example
```

---

## Setup

> See the **Quickstart** at the top for the fastest path. This section covers alternatives and details.

### Prerequisites
- Python 3.11+
- Microsoft Azure AI Foundry project with 3 models deployed (smart, cheap, embeddings)
- Tavily API key — free at [app.tavily.com](https://app.tavily.com)

### Models needed in Azure Foundry
| Role | Example model | Used for |
|---|---|---|
| Smart | `gpt-4o` | Planning, reflection, synthesis |
| Cheap | `gpt-4o-mini` | Per-page summarization (runs 50-100x per query) |
| Embeddings | `text-embedding-3-small` | Semantic summary ranking before synthesis |

### Install — with uv (recommended)

```bash
pip install uv          # install uv if you don't have it
uv sync                 # installs all dependencies from uv.lock
```

### Install — with plain pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai pydantic-settings httpx trafilatura streamlit pytest
```

### Configure

```bash
cp .env.example .env
# Open .env and fill in your values — comments in the file explain each key
```

### Run unit tests (no API keys needed)

```bash
pytest tests/unit/ -q
# Expected: 340 passed
```

### Run from Python

```python
from agent.loop import run_research

state = run_research(
    "What are the major breakthroughs in solid-state battery technology in 2024?",
    on_progress=print,   # stream live progress to stdout
)
print(state.final_report)
print(f"Sources: {len(state.sources)}  Cost: ${state.estimated_cost_usd:.4f}")
```

### Run the Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
# Ask tab: run a research query and watch live progress
# Dashboard tab: stats across all your past runs
# Traces tab: inspect every span from any run
```

### Run evaluations (needs API keys)

```bash
# Run all 5 eval questions and see composite scores
python -m evals.runner

# Run one question by category
python -m evals.runner --category science

# Save results to JSON
python -m evals.runner --output results.json
```

---

## Configuration

All settings are driven by environment variables (`.env` file). Key ones:

| Variable | Default | Description |
|---|---|---|
| `MAX_RESEARCH_ROUNDS` | `3` | Max rounds of search-reflect iteration |
| `MAX_SOURCES_PER_RUN` | `30` | Hard cap on total pages collected |
| `MAX_COST_USD` | `0.50` | Budget cap — stops the loop if exceeded |
| `TOP_K_SUMMARIES` | `30` | How many summaries enter synthesis (ranked by relevance) |
| `MAX_SUMMARY_TOKENS` | `500` | Token budget for per-page summaries |
| `MAX_FETCH_RETRIES` | `2` | Retry attempts for network calls |
| `SEARCH_DEPTH` | `basic` | Tavily depth — `basic` (1 credit) or `advanced` (2 credits) |

---

## Build Status

| Phase | Status | What it builds |
|---|---|---|
| 1 — Foundation | ✅ Complete | Search, fetch, extract tools + integration tests |
| 2 — Research Loop | ✅ Complete | Planner, researcher, reflector, ResearchState, prompts |
| 3 — Synthesizer | ✅ Complete | Two-shot outline → report, mandatory `[N]` citations, References |
| 4 — Observability | ✅ Complete | Span-based tracer, dashboard metrics, cost tracking |
| 5 — Streamlit UI | ✅ Complete | Ask + Dashboard + Traces tabs, live `on_progress` callback |
| 6 — Production Ready | ✅ Complete | Retry/backoff, structured logging, embedding ranking, silent failure audit, 340 tests |

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | GPT-4o + GPT-4o-mini via Microsoft Azure AI Foundry |
| Search | Tavily API |
| URL fetching | Jina Reader + trafilatura (with retry) |
| Agent framework | Raw Python — no LangChain, no LangGraph |
| Config | `pydantic-settings` v2 |
| Web UI | Streamlit |
| Package manager | `uv` |
| Tests | `pytest` (340 unit tests) |

---

*Last updated: March 2026*
