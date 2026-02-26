"""
observability/dashboard.py — Metrics over saved trace files.

THE CORE CONCEPT:
  Every run saves a trace JSON to logs/traces/. The dashboard loads those
  files and computes aggregate metrics: success rate, latency distribution,
  cost per run, which steps fail most often.

  This answers the questions you can't answer from a single run:
    - "Is the reflector always finding gaps even when it shouldn't?"
    - "What's the p95 latency for researcher calls?"
    - "How much does an average run cost?"
    - "Which round do most runs stop at?"

FUNCTIONS:
  load_traces(n)         → last N trace dicts from disk
  summary_stats(traces)  → success/partial/failed counts + rates
  latency_stats(traces)  → per-step p50/p90/p95 duration_ms
  cost_stats(traces)     → avg/min/max/total cost
  span_failure_rates(traces) → which step names fail most
  slow_runs(traces, ms)  → runs that exceeded a duration threshold

USAGE:
  from observability.dashboard import load_traces, summary_stats, latency_stats

  traces = load_traces(n=20)
  print(summary_stats(traces))
  print(latency_stats(traces))
"""

import json
import statistics
from pathlib import Path


# ── Loader ────────────────────────────────────────────────────────────────────

def load_traces(n: int = 20, log_dir: Path | None = None) -> list[dict]:
    """
    Load the last N trace JSON files from logs/traces/.

    Returns a list of raw dicts (as saved by Tracer.save()).
    Files are sorted by modification time — most recent last.
    Returns [] if the directory doesn't exist or is empty.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs" / "traces"

    if not log_dir.exists():
        return []

    files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    recent = files[-n:] if len(files) > n else files

    traces = []
    for path in recent:
        try:
            with open(path, encoding="utf-8") as f:
                traces.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # skip corrupt files

    return traces


# ── Metrics ───────────────────────────────────────────────────────────────────

def summary_stats(traces: list[dict]) -> dict:
    """
    High-level counts and rates across all traces.

    Returns:
        total, success, partial, failed counts
        success_rate, partial_rate, failed_rate (0.0–1.0)
        avg_rounds, avg_sources
    """
    if not traces:
        return {}

    total = len(traces)
    statuses = [t.get("status", "unknown") for t in traces]
    success = statuses.count("success")
    partial = statuses.count("partial")
    failed  = statuses.count("failed")

    rounds  = [t.get("n_rounds", 0) for t in traces]
    sources = [t.get("n_sources", 0) for t in traces]

    return {
        "total": total,
        "success": success,
        "partial": partial,
        "failed": failed,
        "success_rate": round(success / total, 3),
        "partial_rate": round(partial / total, 3),
        "failed_rate": round(failed / total, 3),
        "avg_rounds": round(statistics.mean(rounds), 2) if rounds else 0,
        "avg_sources": round(statistics.mean(sources), 2) if sources else 0,
    }


def latency_stats(traces: list[dict]) -> dict:
    """
    Per-step latency percentiles (p50, p90, p95) in milliseconds.

    Collects all spans of each name across all traces, then computes
    percentiles. Also includes overall run duration percentiles.

    Returns:
        {
          "run":        {"p50": ..., "p90": ..., "p95": ...},
          "planner":    {"p50": ..., "p90": ..., "p95": ...},
          "researcher": {"p50": ..., "p90": ..., "p95": ...},
          "reflector":  {"p50": ..., "p90": ..., "p95": ...},
          "synthesizer":{"p50": ..., "p90": ..., "p95": ...},
        }
    """
    if not traces:
        return {}

    # Overall run durations
    run_durations = [t.get("total_duration_ms", 0) for t in traces if t.get("total_duration_ms")]

    # Per-span-name durations
    span_durations: dict[str, list[float]] = {}
    for trace in traces:
        for span in trace.get("spans", []):
            name = span.get("name", "unknown")
            ms = span.get("duration_ms", 0)
            span_durations.setdefault(name, []).append(ms)

    result = {}
    if run_durations:
        result["run"] = _percentiles(run_durations)
    for name, durations in span_durations.items():
        result[name] = _percentiles(durations)

    return result


def cost_stats(traces: list[dict]) -> dict:
    """
    Cost distribution across runs in USD.

    Returns avg, min, max, total estimated cost.
    """
    if not traces:
        return {}

    costs = [t.get("estimated_cost_usd", 0.0) for t in traces]
    costs = [c for c in costs if c is not None]

    if not costs:
        return {}

    return {
        "avg_usd": round(statistics.mean(costs), 5),
        "min_usd": round(min(costs), 5),
        "max_usd": round(max(costs), 5),
        "total_usd": round(sum(costs), 5),
        "n_runs": len(costs),
    }


def span_failure_rates(traces: list[dict]) -> dict:
    """
    Which step names have the highest error rate.

    Returns {span_name: {"total": N, "errors": M, "error_rate": 0.0–1.0}}
    sorted by error_rate descending.
    """
    if not traces:
        return {}

    counts: dict[str, dict] = {}
    for trace in traces:
        for span in trace.get("spans", []):
            name = span.get("name", "unknown")
            if name not in counts:
                counts[name] = {"total": 0, "errors": 0}
            counts[name]["total"] += 1
            if span.get("status") == "error":
                counts[name]["errors"] += 1

    result = {}
    for name, c in counts.items():
        result[name] = {
            "total": c["total"],
            "errors": c["errors"],
            "error_rate": round(c["errors"] / c["total"], 3) if c["total"] else 0.0,
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["error_rate"], reverse=True))


def slow_runs(traces: list[dict], threshold_ms: float = 60_000) -> list[dict]:
    """
    Return traces where total_duration_ms exceeded the threshold.

    Useful for identifying outlier runs. Default threshold: 60 seconds.

    Each returned item has: run_id, query (truncated), duration_ms, status.
    """
    slow = []
    for t in traces:
        duration = t.get("total_duration_ms", 0)
        if duration >= threshold_ms:
            slow.append({
                "run_id": t.get("run_id", ""),
                "query": t.get("query", "")[:80],
                "duration_ms": duration,
                "status": t.get("status", "unknown"),
                "n_sources": t.get("n_sources", 0),
            })
    return sorted(slow, key=lambda x: x["duration_ms"], reverse=True)


def recent_runs(traces: list[dict], n: int = 10) -> list[dict]:
    """
    Summary of the last N runs for display in a UI or CLI.

    Each item: run_id, query (truncated), status, duration_ms, n_sources,
    n_rounds, estimated_cost_usd.
    """
    recent = traces[-n:] if len(traces) > n else traces
    result = []
    for t in reversed(recent):  # most recent first
        result.append({
            "run_id": t.get("run_id", ""),
            "query": t.get("query", "")[:60],
            "status": t.get("status", "unknown"),
            "duration_ms": t.get("total_duration_ms", 0),
            "n_sources": t.get("n_sources", 0),
            "n_rounds": t.get("n_rounds", 0),
            "cost_usd": t.get("estimated_cost_usd", 0.0),
            "started_at": t.get("started_at", ""),
        })
    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _percentiles(values: list[float]) -> dict:
    """Compute p50, p90, p95 from a list of numeric values."""
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0}
    s = sorted(values)
    return {
        "p50": round(_pct(s, 50), 2),
        "p90": round(_pct(s, 90), 2),
        "p95": round(_pct(s, 95), 2),
    }


def _pct(sorted_values: list[float], p: float) -> float:
    """Linear interpolation percentile."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
