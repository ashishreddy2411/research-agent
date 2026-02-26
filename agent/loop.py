"""
agent/loop.py — Full research pipeline: Planner → Researcher → Reflector → Synthesizer.

THE PIPELINE:

  1. Planner decomposes the query into subqueries
  2. For each subquery: Researcher searches and summarizes
  3. Reflector evaluates coverage and either:
       a. Finds a gap → generate follow-up query → go to step 2 (new round)
       b. No gap → proceed to synthesis
  4. Synthesizer turns page summaries into a final Markdown report with citations
  5. Stopping conditions (hard stops that override the reflector):
       - max_research_rounds reached
       - max_sources_per_run reached
       - cost cap exceeded
       - any unexpected exception

WHAT run_research() RETURNS:
  Always returns a ResearchState, never raises.
  status=SUCCESS  → final_report, sources, outline all populated
  status=PARTIAL  → hit a hard stop; research done but synthesis skipped
  status=FAILED   → unexpected crash, no report

PROGRESS CALLBACK:
  Pass on_progress=callable to get live updates during the run.
  Useful for streaming progress to a UI without threading.
  The callback receives a plain string message at each meaningful step.

USAGE:
  from agent.loop import run_research
  state = run_research("What are the latest advances in solid-state batteries?")
  print(state.status)
  print(state.final_report)
  print(f"Sources: {state.total_sources}, Cost: ${state.estimated_cost_usd:.4f}")
"""

from typing import Callable

from agent.state import ResearchState, ResearchStatus
from agent.planner import Planner
from agent.researcher import Researcher
from agent.reflector import Reflector
from agent.synthesizer import Synthesizer
from agent.guardrails import validate_query
from observability.tracer import Tracer
from llm.client import LLMClient
from config import settings


def run_research(
    query: str,
    on_progress: Callable[[str], None] | None = None,
) -> ResearchState:
    """
    Execute the full research pipeline for a query.

    Args:
        query:       The research question.
        on_progress: Optional callback called with a status string at each
                     meaningful step. Use this to stream progress to a UI.

    Never raises. All exceptions are caught and recorded in state.errors.
    """
    try:
        query = validate_query(query)
    except Exception as e:
        state = ResearchState(query=str(query) if query else "")
        state.errors.append(str(e))
        state.record_failure(f"Invalid query: {e}")
        return state

    state = ResearchState(query=query)
    client = LLMClient()
    tracer = Tracer(query=query)

    def _progress(msg: str) -> None:
        _log(msg)
        if on_progress:
            on_progress(msg)

    try:
        planner = Planner(client=client)
        researcher = Researcher(client=client)
        reflector = Reflector(client=client)
        synthesizer = Synthesizer(client=client)

        _run_loop(state, planner, researcher, reflector, tracer, client, _progress)

        # Only synthesize if research didn't hit a hard stop
        if state.status == ResearchStatus.RUNNING:
            _progress(f"Synthesizing report from {state.total_sources} sources...")
            with tracer.span("synthesizer") as span:
                synthesizer.synthesize(state)
                client.update_state_cost(state)
                span.metadata["n_sources"] = len(state.sources)
                span.metadata["report_chars"] = len(state.final_report)
                span.metadata["n_sections"] = len(state.outline)
            _progress(f"Report complete — {len(state.final_report)} chars")

    except Exception as e:
        state.errors.append(f"Unexpected error in research loop: {type(e).__name__}: {e}")
        state.record_failure(f"Unexpected error: {type(e).__name__}")

    finally:
        client.update_state_cost(state)
        tracer.finish(state)
        path = tracer.save()
        _log(f"Trace saved → {path}")

    return state


def _run_loop(
    state: ResearchState,
    planner: Planner,
    researcher: Researcher,
    reflector: Reflector,
    tracer: Tracer,
    client: LLMClient,
    progress: Callable[[str], None],
) -> None:
    """The actual loop logic. Modifies state in place."""

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    progress("Planning: breaking your question into search angles...")
    with tracer.span("planner") as span:
        planner.plan(state)
        client.update_state_cost(state)
        span.metadata["subqueries"] = state.subqueries
        span.metadata["n_subqueries"] = len(state.subqueries)
    progress(f"Planning complete — {len(state.subqueries)} search angles")

    # ── Step 2+: Research rounds ──────────────────────────────────────────────
    current_queries = state.subqueries.copy()

    for round_num in range(1, settings.max_research_rounds + 1):

        progress(f"Round {round_num} of {settings.max_research_rounds}")

        for subquery in current_queries:
            # Cost cap check — always use live cost from client
            client.update_state_cost(state)
            if state.estimated_cost_usd >= settings.max_cost_usd:
                state.errors.append(
                    f"Cost cap ${settings.max_cost_usd} reached in round {round_num}"
                )
                state.record_partial(
                    report="",
                    sources=[s.url for s in state.page_summaries],
                    reason=f"Cost cap ${settings.max_cost_usd:.2f} reached after round {round_num - 1}",
                )
                return

            # Source cap check
            if state.total_sources >= settings.max_sources_per_run:
                progress(f"Source cap {settings.max_sources_per_run} reached")
                break

            progress(f"Searching: {subquery[:70]}...")
            with tracer.span("researcher") as span:
                span.metadata["subquery"] = subquery
                span.metadata["round"] = round_num
                new_count = researcher.research_into_state(subquery, state, round_num)
                client.update_state_cost(state)
                span.metadata["n_new_sources"] = new_count
                span.metadata["total_sources"] = state.total_sources
            progress(f"Found {new_count} new sources — {state.total_sources} total")

        state.rounds_completed = round_num

        # ── Step 3: Reflect ────────────────────────────────────────────────────
        if round_num >= settings.max_research_rounds:
            progress("Max rounds reached — moving to synthesis")
            break

        progress(f"Evaluating coverage across {state.total_sources} sources...")
        with tracer.span("reflector") as span:
            span.metadata["round"] = round_num
            reflection = reflector.reflect_on_state(state)
            client.update_state_cost(state)
            span.metadata["has_gap"] = reflection.has_gap
            span.metadata["gap_description"] = reflection.gap_description
            span.metadata["follow_up_query"] = reflection.follow_up_query or ""

        if not reflection.has_gap:
            progress(f"Coverage complete — {reflection.gap_description}")
            break

        follow_up = reflection.follow_up_query or ""
        progress(f"Gap found: {follow_up[:70]}...")
        current_queries = [follow_up] if follow_up else state.subqueries[:1]

    _log(
        f"Research complete. "
        f"Rounds: {state.rounds_completed}, "
        f"Sources: {state.total_sources}, "
        f"Cost: ${state.estimated_cost_usd:.4f}"
    )


def _log(message: str) -> None:
    print(f"[research-loop] {message}")
