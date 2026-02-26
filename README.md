# Deep Research Agent — Built in Raw Python

> A complete, working Deep Research Agent built without any agent framework — giving you full visibility and control over every layer of the system.

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
| Context management | 50-100 pages compressed via cheap model + relevance filter |
| Stopping condition | Reflector decides when coverage is sufficient |
| Output | Multi-section Markdown report with inline `[N]` citations + References section |
| Cost control | Hard cap — cost checked before every round |
| Observability | Span-based tracing for every pipeline step; saved to `logs/traces/` |
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
│  Research Loop  (up to max_research_rounds)               │
│                                                           │
│  For each subquery:                                       │
│    Researcher:                                            │
│      tavily.search(query) → URLs + page content          │
│      is_safe_url() check before every fetch              │
│      cheap_llm.summarize(page) → 200-word bullets        │
│      → PageSummary added to ResearchState                │
│                                                           │
│  Reflector: "What gaps remain? Follow-up query?"         │
│    → gap found → next round with follow-up query         │
│    → no gap OR max rounds → stop                         │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Synthesizer (two-shot)              │
│  Shot 1: outline (section headings)  │
│  Shot 2: full report, [N] citations  │
│  check_citation_bounds() validates   │
│  References section auto-appended    │
└──────────────────────────────────────┘
        │
        ▼
    ResearchState returned
    (final_report, sources, cost_usd, rounds, status, spans)
```

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
│   ├── synthesizer.py    # two-shot outline → report with mandatory citations
│   ├── guardrails.py     # validate_query, is_safe_url, check_citation_bounds, dedup
│   └── loop.py           # orchestrates the full pipeline; never raises
│
├── prompts/
│   ├── planner.py        # DECOMPOSE_PROMPT
│   ├── reflector.py      # REFLECT_PROMPT
│   └── synthesizer.py    # OUTLINE_PROMPT, REPORT_PROMPT
│
├── tools/
│   ├── search.py         # Tavily API wrapper → list[SearchResult]
│   ├── fetch.py          # Jina Reader + trafilatura → FetchResult (SSRF-safe)
│   └── extract.py        # HTML → clean text, truncation helpers
│
├── llm/
│   └── client.py         # Azure AI Foundry wrapper: generate, generate_cheap, embed
│                         # tracks token usage + cost per call
│
├── observability/
│   ├── tracer.py         # Span + Trace dataclasses; context-manager instrumentation
│   └── dashboard.py      # load_traces, summary_stats, latency_stats, cost_stats
│
├── evals/
│   ├── dataset.py        # 5 eval questions with ground-truth keywords
│   ├── metrics.py        # citation_accuracy, citation_density, keyword_coverage,
│   │                     # source_quality, run_score (composite)
│   └── runner.py         # CLI runner — coloured output, summary table, JSON export
│
├── tests/
│   ├── unit/             # 314 tests, no API calls required
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
│   │   └── test_extract.py
│   └── integration/      # end-to-end tests (needs real API keys)
│
├── app.py                # Streamlit UI — Ask, Dashboard, Traces tabs
├── config.py             # pydantic-settings — all env vars typed + validated
├── pyproject.toml
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Microsoft Azure AI Foundry project with models deployed
- Tavily API key (free at [app.tavily.com](https://app.tavily.com))

### Models needed in Azure Foundry
| Role | Used for |
|---|---|
| Smart (e.g. `gpt-4o`) | Planning, reflection, synthesis |
| Cheap (e.g. `gpt-4o-mini`) | Per-page summarization (50-100x per run) |
| Embeddings (e.g. `text-embedding-3-small`) | Context relevance filtering |

### Install

```bash
cd research-agent
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your Azure AI Foundry and Tavily credentials
```

### Run unit tests (no API keys needed)

```bash
uv run pytest tests/unit/ -v
# 314 passed
```

### Run the Streamlit UI

```bash
uv run streamlit run app.py
# Opens at http://localhost:8501
```

### Run from Python

```python
from agent.loop import run_research

state = run_research(
    "What are the major breakthroughs in solid-state battery technology in 2024?",
    on_progress=print,   # stream progress to stdout
)
print(state.final_report)
print(f"Sources: {len(state.sources)}  Cost: ${state.estimated_cost_usd:.4f}")
```

### Run evaluations (needs API keys)

```bash
# Run all 5 eval questions
uv run python -m evals.runner

# Run one question by category
uv run python -m evals.runner --category science

# Save results to JSON
uv run python -m evals.runner --output results.json
```

---

## Build Status

| Phase | Status | What it builds |
|---|---|---|
| 1 — Foundation | ✅ Complete | Search, fetch, extract tools + integration tests |
| 2 — Research Loop | ✅ Complete | Planner, researcher, reflector, ResearchState, prompts folder |
| 3 — Synthesizer | ✅ Complete | Two-shot outline → report, mandatory `[N]` citations, References section |
| 4 — Observability | ✅ Complete | Span-based tracer, dashboard metrics, cost tracking in LLMClient |
| 5 — Streamlit UI | ✅ Complete | Ask + Dashboard + Traces tabs, live progress via `on_progress` callback |
| Pre-6 Audit | ✅ Complete | Guardrails at every layer, 314 unit tests, eval framework |
| 6 — Production Ready | 🔲 Planned | Parallel fetching, checkpoint + resume, streaming, job queue |

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | GPT-4o + GPT-4o-mini via Microsoft Azure AI Foundry |
| Search | Tavily API |
| URL fetching | Jina Reader + trafilatura |
| Agent framework | Raw Python — no LangChain, no LangGraph |
| Config | `pydantic-settings` v2 |
| Web UI | Streamlit |
| Package manager | `uv` |
| Tests | `pytest` (314 unit tests) |

---

*Last updated: February 2026*
