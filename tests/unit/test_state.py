"""
tests/unit/test_state.py — Unit tests for agent/state.py

Covers: ResearchState methods, PageSummary, ResearchStatus enum,
        add_cost(), record_success/partial/failure, all properties.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from agent.state import ResearchState, ResearchStatus, PageSummary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_summary(n: int = 1) -> PageSummary:
    return PageSummary(
        url=f"https://example.com/{n}",
        title=f"Title {n}",
        summary=f"• Fact {n}",
        subquery="test query",
        round_number=1,
        word_count=5,
        source="tavily",
    )

def make_state() -> ResearchState:
    return ResearchState(query="What is solid-state battery technology?")


# ── ResearchStatus ────────────────────────────────────────────────────────────

class TestResearchStatus:
    def test_values(self):
        assert ResearchStatus.RUNNING.value == "running"
        assert ResearchStatus.SUCCESS.value == "success"
        assert ResearchStatus.PARTIAL.value == "partial"
        assert ResearchStatus.FAILED.value  == "failed"

    def test_is_string_enum(self):
        assert isinstance(ResearchStatus.SUCCESS, str)
        assert ResearchStatus.SUCCESS == "success"


# ── ResearchState construction ────────────────────────────────────────────────

class TestResearchStateConstruction:
    def test_initial_status_running(self):
        state = make_state()
        assert state.status == ResearchStatus.RUNNING

    def test_query_stored(self):
        state = ResearchState(query="test question here")
        assert state.query == "test question here"

    def test_default_empty_collections(self):
        state = make_state()
        assert state.subqueries == []
        assert state.page_summaries == []
        assert state.visited_urls == set()
        assert state.knowledge_gaps == []
        assert state.errors == []
        assert state.sources == []

    def test_default_zero_numerics(self):
        state = make_state()
        assert state.rounds_completed == 0
        assert state.total_input_tokens == 0
        assert state.total_output_tokens == 0
        assert state.estimated_cost_usd == 0.0

    def test_started_at_populated(self):
        state = make_state()
        assert state.started_at != ""


# ── add_summary ───────────────────────────────────────────────────────────────

class TestAddSummary:
    def test_appends_to_page_summaries(self):
        state = make_state()
        state.add_summary(make_summary(1))
        assert len(state.page_summaries) == 1

    def test_adds_url_to_visited(self):
        state = make_state()
        s = make_summary(1)
        state.add_summary(s)
        assert s.url in state.visited_urls

    def test_multiple_summaries(self):
        state = make_state()
        for i in range(5):
            state.add_summary(make_summary(i))
        assert len(state.page_summaries) == 5
        assert len(state.visited_urls) == 5

    def test_total_sources_property(self):
        state = make_state()
        assert state.total_sources == 0
        state.add_summary(make_summary(1))
        state.add_summary(make_summary(2))
        assert state.total_sources == 2


# ── add_gap ───────────────────────────────────────────────────────────────────

class TestAddGap:
    def test_appends_gap(self):
        state = make_state()
        state.add_gap("Missing commercial timeline")
        assert "Missing commercial timeline" in state.knowledge_gaps

    def test_multiple_gaps_ordered(self):
        state = make_state()
        state.add_gap("Gap 1")
        state.add_gap("Gap 2")
        assert state.knowledge_gaps == ["Gap 1", "Gap 2"]

    def test_latest_gap_property(self):
        state = make_state()
        assert state.latest_gap is None
        state.add_gap("Gap A")
        state.add_gap("Gap B")
        assert state.latest_gap == "Gap B"

    def test_latest_gap_none_when_empty(self):
        state = make_state()
        assert state.latest_gap is None


# ── add_cost ──────────────────────────────────────────────────────────────────

class TestAddCost:
    def test_accumulates_tokens(self):
        state = make_state()
        state.add_cost(100, 50, model="smart")
        state.add_cost(200, 100, model="smart")
        assert state.total_input_tokens == 300
        assert state.total_output_tokens == 150

    def test_smart_model_cost_calculation(self):
        from config import settings
        state = make_state()
        state.add_cost(1000, 1000, model="smart")
        expected = (
            1000 / 1000 * settings.smart_input_cost_per_1k
            + 1000 / 1000 * settings.smart_output_cost_per_1k
        )
        assert abs(state.estimated_cost_usd - expected) < 1e-9

    def test_cheap_model_cost_calculation(self):
        from config import settings
        state = make_state()
        state.add_cost(1000, 1000, model="cheap")
        expected = (
            1000 / 1000 * settings.cheap_input_cost_per_1k
            + 1000 / 1000 * settings.cheap_output_cost_per_1k
        )
        assert abs(state.estimated_cost_usd - expected) < 1e-9

    def test_cost_accumulates_across_calls(self):
        state = make_state()
        state.add_cost(100, 50, model="smart")
        cost1 = state.estimated_cost_usd
        state.add_cost(100, 50, model="smart")
        assert state.estimated_cost_usd == pytest.approx(cost1 * 2)


# ── record_success ────────────────────────────────────────────────────────────

class TestRecordSuccess:
    def test_sets_status_success(self):
        state = make_state()
        state.record_success(report="# Report", sources=["https://a.com"])
        assert state.status == ResearchStatus.SUCCESS

    def test_stores_report(self):
        state = make_state()
        state.record_success(report="# My Report", sources=[])
        assert state.final_report == "# My Report"

    def test_stores_sources(self):
        state = make_state()
        state.record_success(report="", sources=["https://a.com", "https://b.com"])
        assert state.sources == ["https://a.com", "https://b.com"]

    def test_sets_completed_at(self):
        state = make_state()
        state.record_success(report="", sources=[])
        assert state.completed_at != ""


# ── record_partial ────────────────────────────────────────────────────────────

class TestRecordPartial:
    def test_sets_status_partial(self):
        state = make_state()
        state.record_partial(report="partial", sources=[], reason="cost cap")
        assert state.status == ResearchStatus.PARTIAL

    def test_stores_stop_reason(self):
        state = make_state()
        state.record_partial(report="", sources=[], reason="Cost cap $2.00 reached")
        assert state.stop_reason == "Cost cap $2.00 reached"

    def test_partial_has_report(self):
        state = make_state()
        state.record_partial(report="partial report", sources=[], reason="cap")
        assert state.has_report is True


# ── record_failure ────────────────────────────────────────────────────────────

class TestRecordFailure:
    def test_sets_status_failed(self):
        state = make_state()
        state.record_failure("Unexpected crash")
        assert state.status == ResearchStatus.FAILED

    def test_stores_stop_reason(self):
        state = make_state()
        state.record_failure("ValueError in planner")
        assert state.stop_reason == "ValueError in planner"

    def test_no_report_on_failure(self):
        state = make_state()
        state.record_failure("crash")
        assert state.has_report is False


# ── Properties ────────────────────────────────────────────────────────────────

class TestProperties:
    def test_is_running_true_initially(self):
        assert make_state().is_running is True

    def test_is_running_false_after_success(self):
        state = make_state()
        state.record_success(report="", sources=[])
        assert state.is_running is False

    def test_has_report_false_initially(self):
        assert make_state().has_report is False

    def test_has_report_true_after_success(self):
        state = make_state()
        state.record_success(report="# Report", sources=[])
        assert state.has_report is True

    def test_total_sources_matches_page_summaries(self):
        state = make_state()
        for i in range(7):
            state.add_summary(make_summary(i))
        assert state.total_sources == 7
