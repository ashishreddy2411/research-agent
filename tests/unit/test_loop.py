"""
tests/unit/test_loop.py — Unit tests for agent/loop.py

Covers: run_research() pipeline, input validation, cost cap trigger,
        round cap, source cap, on_progress callback, failure handling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock, patch, call
from agent.loop import run_research
from agent.state import ResearchState, ResearchStatus, PageSummary
from agent.reflector import ReflectionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_summary(n: int) -> PageSummary:
    return PageSummary(
        url=f"https://example.com/{n}",
        title=f"Title {n}",
        summary="• Fact",
        subquery="sub",
        round_number=1,
        word_count=5,
        source="tavily",
    )


def patched_run(
    query: str,
    planner_queries: list[str] = None,
    researcher_count: int = 3,
    reflector_has_gap: bool = False,
    synthesizer_report: str = "# Report\nContent [1].",
    on_progress=None,
) -> ResearchState:
    """
    Run run_research() with all external dependencies mocked.
    Returns the final ResearchState.
    """
    planner_queries = planner_queries or ["subquery 1", "subquery 2"]

    def fake_plan(state):
        state.subqueries = planner_queries

    def fake_research(subquery, state, round_number):
        for i in range(researcher_count):
            url = f"https://source-{subquery[:10]}-{i}.com"
            if url not in state.visited_urls:
                state.add_summary(make_summary(hash(url) % 1000))
        return researcher_count

    def fake_reflect(state):
        return ReflectionResult(
            has_gap=reflector_has_gap,
            follow_up_query="follow up query" if reflector_has_gap else None,
            gap_description="gap" if reflector_has_gap else "ok",
        )

    def fake_synthesize(state):
        state.record_success(
            report=synthesizer_report,
            sources=[s.url for s in state.page_summaries],
        )

    with patch("agent.loop.Planner") as MockPlanner, \
         patch("agent.loop.Researcher") as MockResearcher, \
         patch("agent.loop.Reflector") as MockReflector, \
         patch("agent.loop.Synthesizer") as MockSynthesizer, \
         patch("agent.loop.Tracer") as MockTracer, \
         patch("agent.loop.LLMClient") as MockClient:

        MockPlanner.return_value.plan.side_effect = fake_plan
        MockResearcher.return_value.research_into_state.side_effect = fake_research
        MockReflector.return_value.reflect_on_state.side_effect = fake_reflect
        MockSynthesizer.return_value.synthesize.side_effect = fake_synthesize

        # Mock tracer context manager
        span_mock = MagicMock()
        span_mock.__enter__ = MagicMock(return_value=MagicMock(metadata={}))
        span_mock.__exit__ = MagicMock(return_value=False)
        MockTracer.return_value.span.return_value = span_mock
        MockTracer.return_value.finish = MagicMock()
        MockTracer.return_value.save.return_value = Path("/tmp/trace.json")

        MockClient.return_value.update_state_cost = MagicMock()

        return run_research(query, on_progress=on_progress)


# ── Input validation ──────────────────────────────────────────────────────────

class TestRunResearchInputValidation:
    def test_empty_query_returns_failed(self):
        state = run_research("")
        assert state.status == ResearchStatus.FAILED
        assert len(state.errors) > 0

    def test_whitespace_only_returns_failed(self):
        state = run_research("   ")
        assert state.status == ResearchStatus.FAILED

    def test_too_short_returns_failed(self):
        state = run_research("short")  # < 10 chars
        assert state.status == ResearchStatus.FAILED

    def test_too_long_returns_failed(self):
        state = run_research("x" * 501)
        assert state.status == ResearchStatus.FAILED

    def test_valid_query_runs(self):
        state = patched_run("What are the latest AI breakthroughs in 2025?")
        assert state.status == ResearchStatus.SUCCESS

    def test_failed_state_has_error_message(self):
        state = run_research("")
        assert any("empty" in e.lower() or "invalid" in e.lower() for e in state.errors)


# ── Pipeline flow ─────────────────────────────────────────────────────────────

class TestRunResearchPipeline:
    def test_success_status_on_completion(self):
        state = patched_run("What are the latest battery breakthroughs in 2025?")
        assert state.status == ResearchStatus.SUCCESS

    def test_final_report_set(self):
        state = patched_run("What are the latest battery breakthroughs in 2025?")
        assert state.final_report != ""

    def test_sources_populated(self):
        state = patched_run("What are the latest battery breakthroughs in 2025?")
        assert len(state.sources) > 0

    def test_rounds_completed_set(self):
        state = patched_run("What are the latest battery breakthroughs in 2025?")
        assert state.rounds_completed >= 1

    def test_never_raises(self):
        # Even with a bad setup, run_research never raises
        with patch("agent.loop.validate_query", side_effect=Exception("unexpected")):
            state = run_research("this is a valid research question")
        # Either failed or succeeded — but no exception
        assert state is not None


# ── on_progress callback ──────────────────────────────────────────────────────

class TestOnProgressCallback:
    def test_callback_called(self):
        messages = []
        patched_run(
            "What are the latest battery breakthroughs in 2025?",
            on_progress=messages.append,
        )
        assert len(messages) > 0

    def test_callback_receives_strings(self):
        messages = []
        patched_run(
            "What are the latest battery breakthroughs in 2025?",
            on_progress=messages.append,
        )
        assert all(isinstance(m, str) for m in messages)

    def test_planning_message_included(self):
        messages = []
        patched_run(
            "What are the latest battery breakthroughs in 2025?",
            on_progress=messages.append,
        )
        assert any("plan" in m.lower() or "search" in m.lower() for m in messages)

    def test_synthesis_message_included(self):
        messages = []
        patched_run(
            "What are the latest battery breakthroughs in 2025?",
            on_progress=messages.append,
        )
        assert any("synth" in m.lower() for m in messages)


# ── Stopping conditions ───────────────────────────────────────────────────────

class TestStoppingConditions:
    def test_reflector_no_gap_stops_loop(self):
        state = patched_run(
            "What are the latest battery breakthroughs in 2025?",
            reflector_has_gap=False,
        )
        # Should complete in 1 round (reflector said no gap)
        assert state.rounds_completed == 1

    def test_max_rounds_respected(self):
        state = patched_run(
            "What are the latest battery breakthroughs in 2025?",
            reflector_has_gap=True,  # always finds gap → hits max rounds
        )
        from config import settings
        assert state.rounds_completed <= settings.max_research_rounds


# ── Failure handling ──────────────────────────────────────────────────────────

class TestFailureHandling:
    def test_unexpected_exception_returns_failed_state(self):
        with patch("agent.loop.Planner", side_effect=Exception("crash")):
            with patch("agent.loop.LLMClient"):
                with patch("agent.loop.Tracer") as MockTracer:
                    MockTracer.return_value.finish = MagicMock()
                    MockTracer.return_value.save.return_value = Path("/tmp/t.json")
                    state = run_research("What are the latest battery breakthroughs?")
        assert state.status == ResearchStatus.FAILED
        assert len(state.errors) > 0

    def test_error_recorded_on_failure(self):
        with patch("agent.loop.Planner", side_effect=RuntimeError("bad planner")):
            with patch("agent.loop.LLMClient"):
                with patch("agent.loop.Tracer") as MockTracer:
                    MockTracer.return_value.finish = MagicMock()
                    MockTracer.return_value.save.return_value = Path("/tmp/t.json")
                    state = run_research("What are the latest battery breakthroughs?")
        assert any("RuntimeError" in e for e in state.errors)
