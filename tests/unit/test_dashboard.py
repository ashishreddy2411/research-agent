"""
tests/unit/test_dashboard.py — Unit tests for observability/dashboard.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from observability.dashboard import (
    load_traces,
    summary_stats,
    latency_stats,
    cost_stats,
    span_failure_rates,
    slow_runs,
    recent_runs,
    _percentiles,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_trace(
    run_id: str = "abc",
    status: str = "success",
    duration_ms: float = 5000,
    n_sources: int = 10,
    n_rounds: int = 2,
    cost: float = 0.05,
    spans: list = None,
) -> dict:
    return {
        "run_id": run_id,
        "query": f"query for {run_id}",
        "started_at": "2026-02-26T00:00:00+00:00",
        "completed_at": "2026-02-26T00:00:05+00:00",
        "status": status,
        "total_duration_ms": duration_ms,
        "n_sources": n_sources,
        "n_rounds": n_rounds,
        "estimated_cost_usd": cost,
        "final_report_chars": 1200,
        "spans": spans or [],
    }


def make_span(name: str, duration_ms: float = 500, status: str = "success") -> dict:
    return {
        "name": name,
        "step": 1,
        "started_at": 0.0,
        "ended_at": 0.5,
        "duration_ms": duration_ms,
        "status": status,
        "metadata": {},
        "error": "" if status == "success" else "SomeError: details",
    }


# ── load_traces ───────────────────────────────────────────────────────────────

class TestLoadTraces:
    def test_returns_empty_list_if_dir_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert load_traces(log_dir=missing) == []

    def test_loads_json_files(self, tmp_path):
        for i in range(3):
            path = tmp_path / f"trace{i}.json"
            path.write_text(json.dumps(make_trace(run_id=str(i))))
        traces = load_traces(log_dir=tmp_path)
        assert len(traces) == 3

    def test_respects_n_limit(self, tmp_path):
        for i in range(10):
            path = tmp_path / f"trace{i:02d}.json"
            path.write_text(json.dumps(make_trace(run_id=str(i))))
        traces = load_traces(n=5, log_dir=tmp_path)
        assert len(traces) == 5

    def test_skips_corrupt_files(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps(make_trace()))
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{{")
        traces = load_traces(log_dir=tmp_path)
        assert len(traces) == 1


# ── summary_stats ─────────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_empty_returns_empty_dict(self):
        assert summary_stats([]) == {}

    def test_counts_statuses(self):
        traces = [
            make_trace(status="success"),
            make_trace(status="success"),
            make_trace(status="partial"),
            make_trace(status="failed"),
        ]
        stats = summary_stats(traces)
        assert stats["total"] == 4
        assert stats["success"] == 2
        assert stats["partial"] == 1
        assert stats["failed"] == 1

    def test_success_rate(self):
        traces = [make_trace(status="success")] * 3 + [make_trace(status="failed")]
        stats = summary_stats(traces)
        assert stats["success_rate"] == 0.75

    def test_avg_rounds(self):
        traces = [make_trace(n_rounds=2), make_trace(n_rounds=4)]
        assert summary_stats(traces)["avg_rounds"] == 3.0

    def test_avg_sources(self):
        traces = [make_trace(n_sources=10), make_trace(n_sources=20)]
        assert summary_stats(traces)["avg_sources"] == 15.0


# ── latency_stats ─────────────────────────────────────────────────────────────

class TestLatencyStats:
    def test_empty_returns_empty(self):
        assert latency_stats([]) == {}

    def test_run_key_present(self):
        traces = [make_trace(duration_ms=1000), make_trace(duration_ms=2000)]
        result = latency_stats(traces)
        assert "run" in result

    def test_per_span_keys(self):
        spans = [make_span("planner", 200), make_span("researcher", 800)]
        traces = [make_trace(spans=spans)]
        result = latency_stats(traces)
        assert "planner" in result
        assert "researcher" in result

    def test_percentiles_populated(self):
        spans = [make_span("researcher", 500)]
        traces = [make_trace(spans=spans)] * 5
        result = latency_stats(traces)
        assert "p50" in result["researcher"]
        assert "p90" in result["researcher"]
        assert "p95" in result["researcher"]

    def test_single_value_percentiles(self):
        result = _percentiles([100.0])
        assert result["p50"] == 100.0
        assert result["p90"] == 100.0
        assert result["p95"] == 100.0

    def test_multiple_values(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = _percentiles(values)
        assert result["p50"] == 55.0
        assert result["p90"] == 91.0
        assert result["p95"] == 95.5


# ── cost_stats ────────────────────────────────────────────────────────────────

class TestCostStats:
    def test_empty_returns_empty(self):
        assert cost_stats([]) == {}

    def test_avg_cost(self):
        traces = [make_trace(cost=0.10), make_trace(cost=0.20)]
        result = cost_stats(traces)
        assert result["avg_usd"] == pytest.approx(0.15, abs=1e-5)

    def test_min_max(self):
        traces = [make_trace(cost=0.05), make_trace(cost=0.30), make_trace(cost=0.15)]
        result = cost_stats(traces)
        assert result["min_usd"] == pytest.approx(0.05, abs=1e-5)
        assert result["max_usd"] == pytest.approx(0.30, abs=1e-5)

    def test_total(self):
        traces = [make_trace(cost=0.10)] * 5
        result = cost_stats(traces)
        assert result["total_usd"] == pytest.approx(0.50, abs=1e-5)


# ── span_failure_rates ────────────────────────────────────────────────────────

class TestSpanFailureRates:
    def test_empty_returns_empty(self):
        assert span_failure_rates([]) == {}

    def test_zero_failures(self):
        spans = [make_span("researcher", status="success")] * 3
        traces = [make_trace(spans=spans)]
        result = span_failure_rates(traces)
        assert result["researcher"]["error_rate"] == 0.0

    def test_partial_failures(self):
        spans = [
            make_span("researcher", status="success"),
            make_span("researcher", status="error"),
            make_span("researcher", status="success"),
            make_span("researcher", status="error"),
        ]
        traces = [make_trace(spans=spans)]
        result = span_failure_rates(traces)
        assert result["researcher"]["error_rate"] == 0.5
        assert result["researcher"]["errors"] == 2
        assert result["researcher"]["total"] == 4

    def test_sorted_by_error_rate_desc(self):
        spans = [
            make_span("researcher", status="error"),   # 100% fail
            make_span("planner", status="success"),     # 0% fail
        ]
        traces = [make_trace(spans=spans)]
        result = span_failure_rates(traces)
        keys = list(result.keys())
        assert keys[0] == "researcher"


# ── slow_runs ─────────────────────────────────────────────────────────────────

class TestSlowRuns:
    def test_filters_by_threshold(self):
        traces = [
            make_trace(run_id="fast", duration_ms=5_000),
            make_trace(run_id="slow", duration_ms=120_000),
        ]
        result = slow_runs(traces, threshold_ms=60_000)
        assert len(result) == 1
        assert result[0]["run_id"] == "slow"

    def test_empty_if_none_slow(self):
        traces = [make_trace(duration_ms=1000)]
        assert slow_runs(traces, threshold_ms=60_000) == []

    def test_sorted_slowest_first(self):
        traces = [
            make_trace(run_id="a", duration_ms=70_000),
            make_trace(run_id="b", duration_ms=200_000),
            make_trace(run_id="c", duration_ms=90_000),
        ]
        result = slow_runs(traces, threshold_ms=60_000)
        assert result[0]["run_id"] == "b"
        assert result[1]["run_id"] == "c"

    def test_query_truncated_at_80(self):
        t = make_trace(run_id="long_q", duration_ms=100_000)
        t["query"] = "x" * 200
        result = slow_runs([t], threshold_ms=60_000)
        assert len(result[0]["query"]) <= 80


# ── recent_runs ───────────────────────────────────────────────────────────────

class TestRecentRuns:
    def test_returns_most_recent_first(self):
        traces = [make_trace(run_id=str(i)) for i in range(5)]
        result = recent_runs(traces, n=3)
        assert result[0]["run_id"] == "4"
        assert result[1]["run_id"] == "3"
        assert result[2]["run_id"] == "2"

    def test_respects_n(self):
        traces = [make_trace(run_id=str(i)) for i in range(10)]
        result = recent_runs(traces, n=4)
        assert len(result) == 4

    def test_fields_present(self):
        traces = [make_trace()]
        result = recent_runs(traces)
        r = result[0]
        assert "run_id" in r
        assert "query" in r
        assert "status" in r
        assert "duration_ms" in r
        assert "n_sources" in r
        assert "cost_usd" in r
