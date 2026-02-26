# Deep Research Agent — Built in Raw Python

> A complete, working Deep Research Agent built without any agent framework — giving you full visibility and control over every layer of the system.

---

## What This Project Is

A system that takes a plain English question, plans a research strategy, searches the web, reads pages, compresses context, reflects on gaps, searches again, and synthesizes a cited structured report.

Built entirely in raw Python. No LangChain. No LangGraph. No GPT-Researcher. Every component — the research loop, the reflect-search iteration, the context compression, the synthesis pipeline — is written from scratch. No framework hiding what's actually happening.

---

## What This Covers

**Core capabilities:**

| Concept | How it works here |
|---|---|
| Planning | Planner decomposes question into targeted subqueries |
| Tools | search, fetch, extract, summarize — each a clean isolated module |
| Input type | Unstructured HTML and web pages |
| Context management | 50-100 pages compressed via cheap model + relevance filter |
| Stopping condition | Reflector decides when coverage is sufficient |
| Output | Multi-section report with inline citations |
| Cost control | Hard cap — research runs are unbounded without it |

**Production patterns (Phase 6):**
- Job queue: decouple submission from execution
- Checkpoint + resume: never lose work to a crash
- Hard cost cap: ceiling on unbounded LLM spend
- Timeout + graceful degradation: one slow URL never blocks everything
- Streaming progress events: tell the user what's happening in real time

---

## Architecture

```
User Question
        │
        ▼
┌──────────────────────────────────┐
│  Planner                         │
│  LLM decomposes into 3-5 queries │
└──────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Research Loop                                          │
│                                                         │
│  Round 1: asyncio.gather(                              │
│    Researcher(subquery_1),  ← parallel                 │
│    Researcher(subquery_2),  ← parallel                 │
│    Researcher(subquery_3),  ← parallel                 │
│  )                                                      │
│                                                         │
│  Each Researcher:                                       │
│    tavily.search(query) → URLs + page content          │
│    cheap_llm.summarize(page) → 200-word bullets        │
│                                                         │
│  Reflector: "What gaps remain? Follow-up query?"       │
│    → gap found → Round 2 with new targeted query       │
│    → no gap OR max rounds → stop                       │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Context Filter                  │
│  cosine_similarity(question,     │
│    each summary) → top 20        │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Synthesizer                     │
│  Outline → section by section    │
│  Inline citations [1][2]         │
└──────────────────────────────────┘
        │
        ▼
    ResearchState returned
    (report, sources, cost_usd, rounds, status)
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
│   ├── synthesizer.py    # outline + section-by-section report generation
│   └── loop.py           # orchestrates the full research pipeline
│
├── tools/
│   ├── search.py         # Tavily API wrapper → list[SearchResult]
│   ├── fetch.py          # Jina Reader + trafilatura → FetchResult
│   └── extract.py        # HTML → clean text, truncation helpers
│
├── llm/
│   └── client.py         # Azure Foundry wrapper: generate, generate_cheap, embed
│
├── observability/
│   └── tracer.py         # Span + Trace structured logging (from SQL Agent)
│
├── evals/
│   ├── dataset/
│   │   └── golden.jsonl  # 20 research questions with expected topics
│   ├── runner.py         # runs evals against real LLM
│   └── metrics.py        # coverage metric, source quality metric
│
├── tests/
│   ├── unit/             # pure logic tests, no API calls
│   └── integration/      # end-to-end tests with real APIs
│
├── config.py             # pydantic-settings — all env vars typed + validated
├── app.py                # Streamlit UI (Phase 5)
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
| Role | Model | Used for |
|---|---|---|
| Smart | `gpt-5.2-chat` | Planning, reflection, synthesis |
| Cheap | `gpt-4o-mini` | Per-page summarization (50-100x per run) |
| Embeddings | `text-embedding-3-small` | Context relevance filtering |

### Install

```bash
cd research-agent
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Run unit tests (no API keys needed)

```bash
uv run pytest tests/unit/ -v
```

### Run integration test (needs real API keys)

```bash
uv run python tests/integration/test_phase1_tools.py
```

---

## Build Status

| Phase | Status | What it builds |
|---|---|---|
| 1 — Foundation | ✅ Complete | Search, fetch, extract tools |
| 2 — Research Loop | 🔲 | Planner, researcher, reflector, ResearchState |
| 3 — Synthesis + Report | 🔲 | Section-by-section report, citations |
| 4 — Evals + Observability | 🔲 | Coverage metrics, cost tracking, traces |
| 5 — Parallel Fetching + UI | 🔲 | asyncio fan-out, Streamlit |
| 6 — Production Ready | 🔲 | Job queue, checkpoints, cost cap, streaming |

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | GPT-5.2 + GPT-4o-mini via Microsoft Azure AI Foundry |
| Search | Tavily API |
| URL fetching | Jina Reader + trafilatura |
| Agent framework | Raw Python — no LangChain, no LangGraph |
| Config | `pydantic-settings` v2 |
| Web UI | Streamlit (Phase 5) |
| Package manager | `uv` |
| Tests | `pytest` |

---

*Last updated: February 2026*
