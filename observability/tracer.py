"""
observability/tracer.py — Span-based tracing for one research run.

THE CORE CONCEPT:
  Every meaningful step in the pipeline is a Span: a named unit of work
  with a start time, end time, status, and metadata dict.

  A Trace collects all spans for one run and saves them to disk as JSON.
  This gives you a permanent record of exactly what the agent did:
    - Which subqueries the planner generated
    - Which URLs each researcher call found
    - Whether the reflector found a gap and what it was
    - How long synthesis took
    - Where failures happened and why

WHY SPANS INSTEAD OF JUST LOGS:
  Logs are linear — one message per line. Spans are structured: each has
  a name, duration, and typed metadata. This lets you ask questions like
  "what's the p95 latency of reflector calls?" or "which subqueries always
  return 0 sources?" You can't answer those from raw log lines.

WHAT GETS TRACED:
  - planner      → subqueries generated, duration
  - researcher   → subquery, n_sources found, duration
  - reflector    → has_gap, follow_up_query, duration
  - synthesizer  → n_sources used, report_chars, duration
  - run          → overall: status, rounds, total_sources, cost, duration

USAGE:
  tracer = Tracer(query="...", run_id="abc123")

  with tracer.span("planner") as span:
      planner.plan(state)
      span.metadata["subqueries"] = state.subqueries

  tracer.finish(state)
  path = tracer.save()
"""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


# ── Span ──────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """
    One named step in the pipeline.

    status is "success" or "error".
    metadata holds step-specific data (subquery, n_sources, has_gap, etc.).
    """
    name: str
    step: int
    started_at: float       # time.monotonic() — for duration math
    ended_at: float = 0.0
    duration_ms: float = 0.0
    status: str = "success"
    metadata: dict = field(default_factory=dict)
    error: str = ""

    def finish(self, status: str = "success", error: str = "") -> None:
        self.ended_at = time.monotonic()
        self.duration_ms = round((self.ended_at - self.started_at) * 1000, 2)
        self.status = status
        self.error = error


# ── Trace ─────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """
    Complete record of one research run: all spans + summary stats.

    Saved to logs/traces/{run_id}.json after the run completes.
    """
    run_id: str
    query: str
    started_at: str         # ISO timestamp
    completed_at: str = ""
    spans: list[Span] = field(default_factory=list)

    # Summary stats (filled by finish())
    status: str = "running"
    n_rounds: int = 0
    n_sources: int = 0
    estimated_cost_usd: float = 0.0
    final_report_chars: int = 0
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict handles nested dataclasses automatically
        return d


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """
    Collects spans for one run and saves the trace to disk.

    Context manager interface:
        with tracer.span("planner") as span:
            span.metadata["subqueries"] = [...]
        # span is automatically finished when the with-block exits

    On error inside the with-block: span status is set to "error"
    and the exception is re-raised — the tracer never swallows errors.
    """

    def __init__(self, query: str, run_id: str | None = None) -> None:
        self._query = query
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._started = time.monotonic()
        self._trace = Trace(
            run_id=self._run_id,
            query=query,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._step_counter = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    @contextmanager
    def span(self, name: str):
        """
        Context manager that creates, times, and closes a span.

        Usage:
            with tracer.span("researcher") as span:
                span.metadata["subquery"] = subquery
                result = researcher.research(...)
                span.metadata["n_sources"] = len(result)

        On exception: span is marked "error", exception is re-raised.
        """
        self._step_counter += 1
        s = Span(name=name, step=self._step_counter, started_at=time.monotonic())
        self._trace.spans.append(s)
        try:
            yield s
            s.finish(status="success")
        except Exception as exc:
            s.finish(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    def finish(self, state) -> None:
        """
        Populate summary stats from the final ResearchState.
        Call this after all spans are done.
        """
        elapsed = time.monotonic() - self._started
        self._trace.completed_at = datetime.now(timezone.utc).isoformat()
        self._trace.total_duration_ms = round(elapsed * 1000, 2)
        self._trace.status = state.status.value
        self._trace.n_rounds = state.rounds_completed
        self._trace.n_sources = state.total_sources
        self._trace.estimated_cost_usd = state.estimated_cost_usd
        self._trace.final_report_chars = len(state.final_report)

    def save(self, log_dir: Path | None = None) -> Path:
        """
        Write the trace to logs/traces/{run_id}.json.
        Returns the path written. Creates the directory if needed.
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs" / "traces"
        log_dir.mkdir(parents=True, exist_ok=True)

        path = log_dir / f"{self._run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._trace.to_dict(), f, indent=2, default=str)

        return path
