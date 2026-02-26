"""
agent/loop.py — The research pipeline: orchestrates Planner → Researcher → Reflector.

THE CORE CONCEPT:
  This file wires the components together into a running pipeline.
  It owns the loop logic, the stopping conditions, and the cost cap enforcement.
  Individual components (Planner, Researcher, Reflector) don't know about
  each other. The loop connects them through ResearchState.

THE LOOP:

  1. Planner decomposes the query into subqueries
  2. For each subquery: Researcher searches and summarizes
  3. Reflector evaluates coverage and either:
       a. Finds a gap → generate follow-up query → go to step 2 (new round)
       b. No gap → stop and synthesize (Phase 3)
  4. Stopping conditions (hard stops that override the reflector):
       - max_research_rounds reached
       - max_sources_per_run reached
       - cost cap exceeded
       - any unexpected exception

  Phase 2 produces a ResearchState with page_summaries filled in.
  Phase 3 (next phase) adds the Synthesizer to turn those summaries into a report.
  For now: run_research() returns the state with summaries — the raw material.

WHAT run_research() RETURNS:
  Always returns a ResearchState, never raises.
  status=RUNNING if synthesis hasn't happened yet (Phase 2 ends here)
  status=PARTIAL if a hard stop was hit during research
  status=FAILED only if an unexpected crash occurred

  This matches the production contract: every code path returns something.

USAGE:
  from agent.loop import run_research
  state = run_research("What are the latest advances in solid-state batteries?")
  print(f"Rounds: {state.rounds_completed}")
  print(f"Sources: {state.total_sources}")
  print(f"Cost: ${state.estimated_cost_usd:.4f}")
  for s in state.page_summaries[:3]:
      print(s.url)
      print(s.summary[:200])
"""

from agent.state import ResearchState, ResearchStatus
from agent.planner import Planner
from agent.researcher import Researcher
from agent.reflector import Reflector
from llm.client import LLMClient
from config import settings


def run_research(query: str) -> ResearchState:
    """
    Execute the full research loop for a query.

    Runs Planner → Researcher → Reflector loop until a stopping condition
    is met. Returns ResearchState with page_summaries populated.

    Phase 3 will extend this to call the Synthesizer and produce a report.
    For now, the state is returned with status=RUNNING (research done,
    synthesis not yet implemented).

    Never raises. All exceptions are caught and recorded in state.errors.
    """
    state = ResearchState(query=query)
    client = LLMClient()
    planner = Planner(client=client)
    researcher = Researcher(client=client)
    reflector = Reflector(client=client)

    try:
        _run_loop(state, planner, researcher, reflector)
    except Exception as e:
        state.errors.append(f"Unexpected error in research loop: {type(e).__name__}: {e}")
        state.record_failure(f"Unexpected error: {type(e).__name__}")

    return state


def _run_loop(
    state: ResearchState,
    planner: Planner,
    researcher: Researcher,
    reflector: Reflector,
) -> None:
    """
    The actual loop logic. Modifies state in place.

    Separated from run_research() so the outer function can catch any
    exception that leaks out and record it properly.
    """
    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    planner.plan(state)
    _log(f"Planned {len(state.subqueries)} subqueries: {state.subqueries}")

    # ── Step 2+: Research rounds ──────────────────────────────────────────────
    # Round 1: research all subqueries from the planner
    # Round 2+: research the follow-up query from the reflector
    current_queries = state.subqueries.copy()

    for round_num in range(1, settings.max_research_rounds + 1):

        _log(f"Round {round_num}: researching {len(current_queries)} queries")

        # Research each query in this round
        for subquery in current_queries:
            # Cost cap check — before each search call
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
                _log(f"Source cap {settings.max_sources_per_run} reached")
                break

            new_count = researcher.research_into_state(subquery, state, round_num)
            _log(f"  '{subquery[:60]}' → {new_count} new summaries")

        state.rounds_completed = round_num
        _log(f"Round {round_num} complete. Total sources: {state.total_sources}")

        # ── Step 3: Reflect ────────────────────────────────────────────────────
        if round_num >= settings.max_research_rounds:
            _log("Max rounds reached — stopping research loop")
            break

        reflection = reflector.reflect_on_state(state)

        if not reflection.has_gap:
            _log(f"Reflector: no gaps found. Reason: {reflection.gap_description}")
            break

        _log(f"Reflector: gap found — '{reflection.follow_up_query}'")
        current_queries = [reflection.follow_up_query]

    # Research loop complete. Synthesis happens in Phase 3.
    _log(
        f"Research complete. "
        f"Rounds: {state.rounds_completed}, "
        f"Sources: {state.total_sources}, "
        f"Cost: ${state.estimated_cost_usd:.4f}"
    )


def _log(message: str) -> None:
    """Simple structured log. Phase 4 replaces this with Trace/Span."""
    print(f"[research-loop] {message}")
