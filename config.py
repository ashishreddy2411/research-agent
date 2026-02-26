"""
config.py — Single source of truth for all research agent settings.

Same pattern as SQL Agent: pydantic-settings reads .env at import time,
raises a clear error if required fields are missing, typed everywhere.

NEW CONCEPTS vs SQL Agent config:

  1. Two LLM models with different roles:
       smart_model  — planning, reflection, synthesis (best quality, ~10 calls/run)
       cheap_model  — per-page summarization (50-100 calls/run, the cost multiplier)

     This is model tiering. Using gpt-5.2-chat for page summarization when
     gpt-4o-mini works just as well costs ~10-30x more per run. With 100 pages
     per run, that multiplier matters enormously. Routing by task complexity
     is the single biggest cost lever in production LLM systems.

  2. Embedding model for context filtering:
       After gathering 50-100 page summaries, we don't feed all of them to
       the LLM — context windows overflow and synthesis gets unfocused.
       We embed the question and each summary, compute cosine similarity,
       keep only the top-k most relevant. text-embedding-3-small handles this.

  3. Hard cost cap:
       max_cost_usd is a hard stop. When a run exceeds it, the agent stops
       the research loop, synthesizes from whatever it has, and returns
       status=PARTIAL. SQL Agent had no cost cap because 5 LLM calls is
       predictable. 100 LLM calls across 100 URLs is not.

  4. Search and fetch tuning:
       max_search_results — URLs per search query
       fetch_timeout_seconds — one slow URL never blocks everything else
       max_research_rounds — how many reflect-search iterations before stop

USAGE:
  from config import settings
  print(settings.smart_model)     # "gpt-5.2-chat"
  print(settings.cheap_model)     # "gpt-4o-mini"
  print(settings.max_cost_usd)    # 2.0
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure AI Foundry ───────────────────────────────────────────────────────
    foundry_endpoint: str = Field(
        description="Full Foundry project endpoint URL"
    )
    foundry_api_key: str = Field(
        default="",
        description="API key — leave blank to use DefaultAzureCredential",
    )
    api_version: str = Field(
        default="2025-04-01-preview",
        description="Azure OpenAI API version for cognitiveservices endpoints",
    )

    # ── LLM Models ────────────────────────────────────────────────────────────
    # WHY TWO MODELS:
    #   smart_model is called ~10 times per run: once to plan, once to reflect
    #   each round, once to build the outline, once per report section.
    #
    #   cheap_model is called once per URL — potentially 100 times per run.
    #   It only needs to answer: "what facts in this page are relevant to the query?"
    #   gpt-4o-mini does this as well as gpt-5.2-chat at 1/30th the cost.
    smart_model: str = Field(
        default="gpt-5.2-chat",
        description="High-quality model for planning, reflection, and final synthesis",
    )
    cheap_model: str = Field(
        default="gpt-4o-mini",
        description="Fast cheap model for per-page summarization (called 50-100x per run)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for relevance-based context filtering",
    )

    # ── Search (Tavily) ───────────────────────────────────────────────────────
    tavily_api_key: str = Field(
        description="Tavily API key — free tier at app.tavily.com (1000 credits/month)"
    )
    max_search_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="URLs to retrieve per search query (1 Tavily credit per call)",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="'basic' = 1 credit, full content. 'advanced' = 2 credits, deeper extraction.",
    )

    # ── URL Fetching ──────────────────────────────────────────────────────────
    # WHY A TIMEOUT:
    #   Some URLs are slow — paywalled sites, overloaded servers, redirects.
    #   Without a timeout, one slow URL blocks all 10 parallel fetches in a batch.
    #   With a timeout, we skip the slow URL and process the 9 that responded.
    #   Partial data from 9 good sources beats waiting 60s for 1 slow one.
    fetch_timeout_seconds: float = Field(
        default=10.0,
        description="Max seconds to wait for one URL fetch — exceeded = skip that URL",
    )
    max_fetch_retries: int = Field(
        default=2,
        description="Retry count for failed fetches before giving up on that URL",
    )

    # ── Research Loop ─────────────────────────────────────────────────────────
    # WHY THESE LIMITS:
    #   Without a round cap, the reflect loop can run indefinitely.
    #   The reflector LLM sometimes finds gaps even when coverage is sufficient.
    #   3 rounds = initial research + 2 gap-filling passes. Covers most questions.
    #
    #   max_sources_per_run is a hard stop independent of rounds. If round 1
    #   returns 40 sources and max is 50, round 2 gets 10 slots, not 40.
    max_research_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max reflect-search iterations before forcing synthesis",
    )
    max_sources_per_run: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Hard cap on total URLs processed across all rounds",
    )
    max_summary_tokens: int = Field(
        default=300,
        description="Token cap on each page summary (cheap model output limit)",
    )
    top_k_summaries: int = Field(
        default=20,
        description="How many summaries pass the relevance filter into synthesis",
    )

    # ── Cost Control (Phase 6 — Production Ready) ─────────────────────────────
    # WHY A HARD CAP:
    #   A normal research run costs $0.20-$2.00.
    #   A runaway query with max rounds + max sources can cost $10-$20.
    #   max_cost_usd stops the loop when the ceiling is reached, synthesizes
    #   from whatever was collected, and returns status=PARTIAL.
    #   This is the most important single production guard for LLM systems.
    max_cost_usd: float = Field(
        default=2.0,
        description="Hard cost ceiling per run (USD). Exceeded → stop + return PARTIAL.",
    )
    # Update these to match your deployed model pricing.
    # Find current pricing in Azure Foundry → your deployment → Pricing tab.
    smart_input_cost_per_1k: float = Field(
        default=0.005,
        description="Cost per 1K input tokens for smart_model (USD)",
    )
    smart_output_cost_per_1k: float = Field(
        default=0.015,
        description="Cost per 1K output tokens for smart_model (USD)",
    )
    cheap_input_cost_per_1k: float = Field(
        default=0.00015,
        description="Cost per 1K input tokens for cheap_model (USD)",
    )
    cheap_output_cost_per_1k: float = Field(
        default=0.0006,
        description="Cost per 1K output tokens for cheap_model (USD)",
    )

    # ── Observability ─────────────────────────────────────────────────────────
    log_dir: str = Field(
        default="logs/",
        description="Directory for structured JSON run logs",
    )
    slow_run_threshold_seconds: float = Field(
        default=300.0,
        description="Flag any research run exceeding this duration (5 min default)",
    )


# Module-level singleton — import this everywhere, never instantiate Settings again.
settings = Settings()
