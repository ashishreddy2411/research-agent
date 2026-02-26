"""
agent/state.py — ResearchState dataclass + ResearchStatus enum.

Design principles:
  - A dataclass, not a dict — typos become AttributeError, not silent new keys
  - One object passed through the entire pipeline — every component reads from
    and writes to the same state object
  - Explicit status machine — no ambiguity about where in the pipeline we are

Key design decisions:

  PageSummary dataclass:
    The agent produces 50-100 intermediate summaries on the way to the final
    report. Each summary carries its provenance — which URL it came from,
    which search query found that URL, and which round of research it belongs
    to. round_number matters because summaries from later rounds often answer
    gaps that earlier rounds didn't cover — you want that metadata at synthesis.

  knowledge_gaps list:
    The reflect loop generates a gap after every round. Storing all gaps
    (not just the latest) lets you see the agent's reasoning over time:
    "Round 1 gap: no coverage of solid-state electrolytes. Round 2 gap:
    missing commercial timeline." This is the agent's self-correction history.

  status=PARTIAL:
    RUNNING / SUCCESS / PARTIAL / FAILED.
    PARTIAL means: the agent hit a cost cap or timeout mid-run, synthesized
    from whatever it collected, and returned an incomplete but real report.
    PARTIAL is better than FAILED — it has a report.

  estimated_cost_usd:
    Cost tracked in dollars, not just tokens, because cost is the primary
    production constraint. Tokens vary by model; dollars are universal.

USAGE:
  from agent.state import ResearchState, ResearchStatus, PageSummary

  state = ResearchState(query="What are solid state battery breakthroughs?")
  state.add_summary(PageSummary(...))
  state.record_success(report="# Report...", sources=[...])
  print(state.status)           # ResearchStatus.SUCCESS
  print(state.estimated_cost_usd)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# ── Status enum ────────────────────────────────────────────────────────────────

class ResearchStatus(str, Enum):
    """
    The lifecycle of a research run.

    RUNNING  → research is in progress
    SUCCESS  → full report produced, all rounds completed or reflector satisfied
    PARTIAL  → report produced but run was cut short (cost cap, timeout, max rounds)
               partial is better than failed — a real report exists
    FAILED   → no report produced — unexpected error stopped the run entirely
    """
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


# ── PageSummary ────────────────────────────────────────────────────────────────

@dataclass
class PageSummary:
    """
    A summarized page — the core unit of research output.

    One PageSummary is produced for each URL the agent reads.
    The cheap model produces a 200-word bullet-point summary of the page
    focused on the research query. The summary, not the raw page, is
    what gets fed into synthesis.

    WHY STORE round_number:
      Summaries from later rounds are often more targeted (they answer
      specific gaps identified by the reflector). Knowing which round
      a summary came from helps debug research quality.

    WHY STORE subquery:
      The final report has citations. To say "According to [1]" we need
      to know which URL produced which fact. The subquery tells us what
      research thread found this page.
    """
    url: str
    title: str
    summary: str          # 200-word bullet points from cheap_model
    subquery: str         # which search query found this URL
    round_number: int     # which research round (1, 2, 3...)
    word_count: int       # len(summary.split()) — for quick stats
    source: str           # "tavily", "jina", or "trafilatura"


# ── ResearchState ──────────────────────────────────────────────────────────────

@dataclass
class ResearchState:
    """
    The complete state of one research run.

    Passed through every component in the pipeline:
      Planner reads query → writes subqueries
      Researcher reads subqueries → writes page_summaries, visited_urls
      Reflector reads page_summaries → writes knowledge_gaps
      Synthesizer reads page_summaries, query → writes final_report
      Loop reads everything → updates status, cost, timing

    Never instantiate a second ResearchState mid-run.
    One state object, one run. That's the contract.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    query: str
    """The original user question. Never modified after construction."""

    # ── Planner output ─────────────────────────────────────────────────────────
    subqueries: list[str] = field(default_factory=list)
    """Decomposed search queries. Set once by the Planner, never modified."""

    # ── Researcher output (grows each round) ───────────────────────────────────
    page_summaries: list[PageSummary] = field(default_factory=list)
    """All page summaries across all rounds. The input to synthesis."""

    visited_urls: set[str] = field(default_factory=set)
    """All URLs seen — used for deduplication. Never fetch the same URL twice."""

    # ── Reflector output ───────────────────────────────────────────────────────
    knowledge_gaps: list[str] = field(default_factory=list)
    """
    Gap identified after each round. If empty, reflector found no gaps.
    The last entry is the most recent gap → next round's follow-up query.
    """

    rounds_completed: int = 0
    """How many reflect-search rounds have finished."""

    # ── Synthesizer output ─────────────────────────────────────────────────────
    outline: list[str] = field(default_factory=list)
    """Section headers for the report. Set by Synthesizer before writing."""

    final_report: str = ""
    """The completed Markdown report. Set when synthesis finishes."""

    sources: list[str] = field(default_factory=list)
    """Ordered list of source URLs for the reference section [1], [2], ..."""

    # ── Status and metadata ────────────────────────────────────────────────────
    status: ResearchStatus = ResearchStatus.RUNNING

    stop_reason: str = ""
    """
    Human-readable explanation of why the run stopped.
    Empty for SUCCESS. Explains the constraint hit for PARTIAL.
    Examples:
      "Cost cap $2.00 reached after round 2"
      "Max rounds (3) reached"
      "Reflector found no remaining gaps"
    """

    errors: list[str] = field(default_factory=list)
    """Non-fatal errors encountered during the run (failed fetches, etc.)."""

    # ── Cost and token tracking ────────────────────────────────────────────────
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # ── Timing ─────────────────────────────────────────────────────────────────
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""

    # ── Convenience methods ────────────────────────────────────────────────────

    def add_summary(self, summary: PageSummary) -> None:
        """Add a page summary and mark its URL as visited."""
        self.page_summaries.append(summary)
        self.visited_urls.add(summary.url)

    def add_gap(self, gap: str) -> None:
        """Record a knowledge gap from the reflector."""
        self.knowledge_gaps.append(gap)

    def add_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "smart",
    ) -> None:
        """
        Add token usage and update estimated cost.

        model: "smart" or "cheap" — determines cost rate.
        Rates come from config. Call this after every LLM response.
        """
        from config import settings

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if model == "cheap":
            cost = (
                input_tokens / 1000 * settings.cheap_input_cost_per_1k
                + output_tokens / 1000 * settings.cheap_output_cost_per_1k
            )
        else:
            cost = (
                input_tokens / 1000 * settings.smart_input_cost_per_1k
                + output_tokens / 1000 * settings.smart_output_cost_per_1k
            )

        self.estimated_cost_usd += cost

    def record_success(self, report: str, sources: list[str]) -> None:
        """Mark run as SUCCESS with the final report."""
        self.final_report = report
        self.sources = sources
        self.status = ResearchStatus.SUCCESS
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def record_partial(self, report: str, sources: list[str], reason: str) -> None:
        """
        Mark run as PARTIAL — report produced but run was cut short.

        PARTIAL is not a failure. It means: "I hit a constraint but I have
        real research to return." A cost-capped run with 2 rounds of research
        is more valuable than an error message.
        """
        self.final_report = report
        self.sources = sources
        self.status = ResearchStatus.PARTIAL
        self.stop_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def record_failure(self, reason: str) -> None:
        """Mark run as FAILED with no report."""
        self.status = ResearchStatus.FAILED
        self.stop_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def total_sources(self) -> int:
        return len(self.page_summaries)

    @property
    def is_running(self) -> bool:
        return self.status == ResearchStatus.RUNNING

    @property
    def has_report(self) -> bool:
        return bool(self.final_report)

    @property
    def latest_gap(self) -> str | None:
        """The most recent knowledge gap — the follow-up query for the next round."""
        return self.knowledge_gaps[-1] if self.knowledge_gaps else None
