"""
tests/unit/test_tracer.py — Unit tests for observability/tracer.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock
from observability.tracer import Tracer, Span, Trace
from agent.state import ResearchState, ResearchStatus, PageSummary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_state(status: ResearchStatus = ResearchStatus.SUCCESS) -> ResearchState:
    state = ResearchState(query="test query")
    state.rounds_completed = 2
    state.estimated_cost_usd = 0.042
    if status == ResearchStatus.SUCCESS:
        state.record_success(report="# Report\nContent.", sources=["https://a.com"])
    return state


# ── Span ──────────────────────────────────────────────────────────────────────

class TestSpan:
    def test_finish_sets_duration(self):
        s = Span(name="test", step=1, started_at=time.monotonic())
        time.sleep(0.01)
        s.finish()
        assert s.duration_ms > 5

    def test_finish_default_status_success(self):
        s = Span(name="test", step=1, started_at=time.monotonic())
        s.finish()
        assert s.status == "success"

    def test_finish_error_status(self):
        s = Span(name="test", step=1, started_at=time.monotonic())
        s.finish(status="error", error="ValueError: bad input")
        assert s.status == "error"
        assert "ValueError" in s.error

    def test_finish_sets_ended_at(self):
        s = Span(name="test", step=1, started_at=time.monotonic())
        s.finish()
        assert s.ended_at > s.started_at


# ── Tracer.span() context manager ─────────────────────────────────────────────

class TestTracerSpan:
    def test_span_added_to_trace(self):
        tracer = Tracer(query="test")
        with tracer.span("planner"):
            pass
        assert len(tracer._trace.spans) == 1
        assert tracer._trace.spans[0].name == "planner"

    def test_span_status_success_on_clean_exit(self):
        tracer = Tracer(query="test")
        with tracer.span("planner"):
            pass
        assert tracer._trace.spans[0].status == "success"

    def test_span_status_error_on_exception(self):
        tracer = Tracer(query="test")
        with pytest.raises(ValueError):
            with tracer.span("researcher"):
                raise ValueError("search failed")
        assert tracer._trace.spans[0].status == "error"
        assert "ValueError" in tracer._trace.spans[0].error

    def test_exception_is_reraised(self):
        tracer = Tracer(query="test")
        with pytest.raises(RuntimeError, match="oops"):
            with tracer.span("reflector"):
                raise RuntimeError("oops")

    def test_metadata_can_be_set(self):
        tracer = Tracer(query="test")
        with tracer.span("researcher") as span:
            span.metadata["subquery"] = "solid state batteries"
            span.metadata["n_new_sources"] = 5
        s = tracer._trace.spans[0]
        assert s.metadata["subquery"] == "solid state batteries"
        assert s.metadata["n_new_sources"] == 5

    def test_step_counter_increments(self):
        tracer = Tracer(query="test")
        with tracer.span("planner"):
            pass
        with tracer.span("researcher"):
            pass
        assert tracer._trace.spans[0].step == 1
        assert tracer._trace.spans[1].step == 2

    def test_multiple_spans_all_recorded(self):
        tracer = Tracer(query="test")
        for name in ["planner", "researcher", "reflector", "synthesizer"]:
            with tracer.span(name):
                pass
        assert len(tracer._trace.spans) == 4
        names = [s.name for s in tracer._trace.spans]
        assert names == ["planner", "researcher", "reflector", "synthesizer"]

    def test_span_duration_measured(self):
        tracer = Tracer(query="test")
        with tracer.span("slow_step"):
            time.sleep(0.02)
        assert tracer._trace.spans[0].duration_ms >= 15


# ── Tracer.finish() ───────────────────────────────────────────────────────────

class TestTracerFinish:
    def test_finish_sets_status(self):
        tracer = Tracer(query="test")
        state = make_state(ResearchStatus.SUCCESS)
        tracer.finish(state)
        assert tracer._trace.status == "success"

    def test_finish_sets_n_sources(self):
        tracer = Tracer(query="test")
        state = make_state()
        tracer.finish(state)
        assert tracer._trace.n_sources == state.total_sources

    def test_finish_sets_cost(self):
        tracer = Tracer(query="test")
        state = make_state()
        state.estimated_cost_usd = 0.123
        tracer.finish(state)
        assert tracer._trace.estimated_cost_usd == 0.123

    def test_finish_sets_total_duration(self):
        tracer = Tracer(query="test")
        time.sleep(0.01)
        state = make_state()
        tracer.finish(state)
        assert tracer._trace.total_duration_ms > 5

    def test_finish_sets_completed_at(self):
        tracer = Tracer(query="test")
        state = make_state()
        tracer.finish(state)
        assert tracer._trace.completed_at != ""

    def test_finish_sets_report_chars(self):
        tracer = Tracer(query="test")
        state = make_state()
        tracer.finish(state)
        assert tracer._trace.final_report_chars == len(state.final_report)


# ── Tracer.save() ─────────────────────────────────────────────────────────────

class TestTracerSave:
    def test_save_creates_file(self, tmp_path):
        tracer = Tracer(query="test query", run_id="test123")
        state = make_state()
        tracer.finish(state)
        path = tracer.save(log_dir=tmp_path)
        assert path.exists()
        assert path.name == "test123.json"

    def test_save_valid_json(self, tmp_path):
        tracer = Tracer(query="test query", run_id="json_test")
        state = make_state()
        tracer.finish(state)
        path = tracer.save(log_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == "json_test"
        assert data["query"] == "test query"

    def test_save_includes_spans(self, tmp_path):
        tracer = Tracer(query="test", run_id="spans_test")
        with tracer.span("planner") as span:
            span.metadata["n_subqueries"] = 4
        state = make_state()
        tracer.finish(state)
        path = tracer.save(log_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["spans"]) == 1
        assert data["spans"][0]["name"] == "planner"
        assert data["spans"][0]["metadata"]["n_subqueries"] == 4

    def test_save_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        tracer = Tracer(query="test", run_id="dir_test")
        state = make_state()
        tracer.finish(state)
        path = tracer.save(log_dir=new_dir)
        assert path.exists()

    def test_run_id_generated_if_not_provided(self):
        t1 = Tracer(query="q1")
        t2 = Tracer(query="q2")
        assert t1.run_id != t2.run_id
        assert len(t1.run_id) == 12
