"""
app.py — Streamlit web UI for the Research Agent.

Three tabs:
  Ask       — run a research query, see the report with citations + trace
  Dashboard — aggregate metrics across all saved runs
  Traces    — browse individual run trace files span by span

Run with:
  uv run streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from agent.loop import run_research
from agent.state import ResearchStatus
from observability.dashboard import (
    load_traces,
    summary_stats,
    latency_stats,
    cost_stats,
    span_failure_rates,
    recent_runs,
    slow_runs,
)
from config import settings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    layout="wide",
    page_icon="🔬",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Agent")
    st.divider()

    st.subheader("Configuration")
    st.caption(f"**Smart model:** {settings.smart_model}")
    st.caption(f"**Cheap model:** {settings.cheap_model}")
    st.caption(f"**Max rounds:** {settings.max_research_rounds}")
    st.caption(f"**Max sources:** {settings.max_sources_per_run}")
    st.caption(f"**Cost cap:** ${settings.max_cost_usd:.2f}")

    st.divider()

    # Show stats from last run if available
    if "last_result" in st.session_state:
        state = st.session_state["last_result"]
        st.subheader("Last Run")
        st.caption(f"**Status:** {state.status.value.upper()}")
        st.caption(f"**Rounds:** {state.rounds_completed}")
        st.caption(f"**Sources:** {state.total_sources}")
        st.caption(f"**Cost:** ${state.estimated_cost_usd:.4f}")
        if state.errors:
            st.caption(f"**Errors:** {len(state.errors)}")

    st.divider()
    traces_count = len(load_traces(n=1000))
    st.caption(f"**Saved traces:** {traces_count}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_latest_trace() -> dict | None:
    """Load the most recently saved trace file."""
    traces = load_traces(n=1)
    return traces[0] if traces else None


def _render_spans_table(spans: list[dict]) -> None:
    """Render a spans list as a Streamlit dataframe."""
    if not spans:
        st.caption("No spans recorded.")
        return

    import pandas as pd

    rows = []
    for s in spans:
        meta = s.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v != "" and v is not None)
        rows.append({
            "Step": s.get("step", ""),
            "Name": s.get("name", ""),
            "Status": s.get("status", ""),
            "Duration ms": s.get("duration_ms", 0),
            "Metadata": meta_str[:120],
            "Error": s.get("error", ""),
        })

    df = pd.DataFrame(rows)

    def color_status(val):
        if val == "success":
            return "color: green"
        if val == "error":
            return "color: red"
        return ""

    st.dataframe(
        df.style.map(color_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_ask, tab_dashboard, tab_traces = st.tabs(["Ask", "Dashboard", "Traces"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK
# ══════════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.header("Ask a Research Question")

    question = st.text_area(
        label="Research question",
        placeholder=(
            "Examples:\n"
            "• What are the latest breakthroughs in solid-state batteries?\n"
            "• How does CRISPR gene editing work and what are its limitations?\n"
            "• What caused the 2008 financial crisis?"
        ),
        height=120,
        key="question_input",
    )

    run_clicked = st.button("Run Research", type="primary", use_container_width=False)

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a research question.")
        else:
            # Clear previous result
            st.session_state.pop("last_result", None)

            with st.status("Researching...", expanded=True) as status:
                log_el = st.empty()
                lines: list[str] = []

                def on_progress(msg: str) -> None:
                    lines.append(msg)
                    log_el.markdown(
                        "\n".join(f"- {m}" for m in lines[-10:])
                    )

                state = run_research(question.strip(), on_progress=on_progress)
                st.session_state["last_result"] = state

                label = (
                    f"Done — {state.total_sources} sources, "
                    f"{state.rounds_completed} round(s), "
                    f"${state.estimated_cost_usd:.4f}"
                )
                status.update(
                    label=label,
                    state="complete" if state.status == ResearchStatus.SUCCESS else "error",
                    expanded=False,
                )

    # ── Display result ────────────────────────────────────────────────────────

    if "last_result" in st.session_state:
        state = st.session_state["last_result"]

        # Status banner
        if state.status == ResearchStatus.SUCCESS:
            st.success(f"Research complete — {state.total_sources} sources, {state.rounds_completed} round(s)")
        elif state.status == ResearchStatus.PARTIAL:
            st.warning(f"Partial result — {state.stop_reason}")
        elif state.status == ResearchStatus.FAILED:
            st.error(f"Research failed — {state.stop_reason}")

        if state.final_report:
            # Report
            with st.expander("Report", expanded=True):
                st.markdown(state.final_report)

            # Sources
            with st.expander(f"Sources ({len(state.sources)})", expanded=False):
                for i, url in enumerate(state.sources, 1):
                    st.markdown(f"**[{i}]** {url}")

            # Outline
            if state.outline:
                with st.expander("Report outline", expanded=False):
                    for i, section in enumerate(state.outline, 1):
                        st.markdown(f"{i}. {section}")

        # Trace spans (load from disk)
        trace_data = _load_latest_trace()
        if trace_data:
            with st.expander("Trace — step timing", expanded=False):
                _render_spans_table(trace_data.get("spans", []))
                st.caption(
                    f"Run ID: `{trace_data.get('run_id', '')}` | "
                    f"Total: {trace_data.get('total_duration_ms', 0):.0f} ms | "
                    f"Cost: ${trace_data.get('estimated_cost_usd', 0):.4f}"
                )

        # Errors
        if state.errors:
            with st.expander(f"Errors ({len(state.errors)})", expanded=False):
                for err in state.errors:
                    st.code(err, language=None)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.header("Dashboard")

    n_traces = st.number_input("Load last N runs", min_value=1, max_value=500, value=20)

    @st.cache_data(ttl=30)
    def get_traces(n: int):
        return load_traces(n=n)

    traces = get_traces(n_traces)

    if not traces:
        st.info("No trace files found. Run a research query first.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        stats = summary_stats(traces)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total runs", stats.get("total", 0))
        col2.metric("Success rate", f"{stats.get('success_rate', 0) * 100:.0f}%")
        col3.metric("Avg sources", f"{stats.get('avg_sources', 0):.1f}")
        col4.metric("Avg rounds", f"{stats.get('avg_rounds', 0):.1f}")

        st.divider()

        # ── Cost ──────────────────────────────────────────────────────────────
        costs = cost_stats(traces)
        if costs:
            st.subheader("Cost")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg / run", f"${costs.get('avg_usd', 0):.4f}")
            c2.metric("Min", f"${costs.get('min_usd', 0):.4f}")
            c3.metric("Max", f"${costs.get('max_usd', 0):.4f}")
            c4.metric("Total", f"${costs.get('total_usd', 0):.4f}")

        st.divider()

        # ── Latency percentiles ───────────────────────────────────────────────
        st.subheader("Latency (ms)")
        lat = latency_stats(traces)
        if lat:
            import pandas as pd
            lat_rows = []
            for step, pcts in lat.items():
                lat_rows.append({
                    "Step": step,
                    "p50 ms": pcts.get("p50", 0),
                    "p90 ms": pcts.get("p90", 0),
                    "p95 ms": pcts.get("p95", 0),
                })
            st.dataframe(
                pd.DataFrame(lat_rows).set_index("Step"),
                use_container_width=True,
            )

        st.divider()

        # ── Failure rates ─────────────────────────────────────────────────────
        failures = span_failure_rates(traces)
        if failures:
            st.subheader("Step failure rates")
            import pandas as pd
            rows = [
                {"Step": name, "Total calls": v["total"], "Errors": v["errors"], "Error rate": f"{v['error_rate']*100:.1f}%"}
                for name, v in failures.items()
            ]
            st.dataframe(pd.DataFrame(rows).set_index("Step"), use_container_width=True)

        st.divider()

        # ── Slow runs ─────────────────────────────────────────────────────────
        slow = slow_runs(traces, threshold_ms=60_000)
        if slow:
            st.subheader(f"Slow runs (>60s) — {len(slow)} found")
            for r in slow:
                st.warning(
                    f"`{r['run_id']}` — {r['duration_ms'] / 1000:.1f}s — "
                    f"{r['status']} — {r['n_sources']} sources — {r['query']}"
                )

        st.divider()

        # ── Recent runs table ─────────────────────────────────────────────────
        st.subheader("Recent runs")
        import pandas as pd
        rows = recent_runs(traces, n=10)
        if rows:
            df = pd.DataFrame(rows)
            df["duration_s"] = (df["duration_ms"] / 1000).round(1)
            df["cost_usd"] = df["cost_usd"].round(4)
            df = df[["run_id", "query", "status", "duration_s", "n_sources", "n_rounds", "cost_usd", "started_at"]]
            st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRACES
# ══════════════════════════════════════════════════════════════════════════════

with tab_traces:
    st.header("Traces")

    all_traces = load_traces(n=100)

    if not all_traces:
        st.info("No trace files found. Run a research query first.")
    else:
        # Build dropdown options: "run_id — truncated query"
        options = [
            f"{t.get('run_id', '?')} — {t.get('query', '')[:60]}"
            for t in reversed(all_traces)
        ]
        selected = st.selectbox("Select a run", options)

        if selected:
            run_id = selected.split(" — ")[0]
            trace = next((t for t in all_traces if t.get("run_id") == run_id), None)

            if trace:
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Status", trace.get("status", "?").upper())
                col_b.metric("Duration", f"{trace.get('total_duration_ms', 0) / 1000:.1f}s")
                col_c.metric("Sources", trace.get("n_sources", 0))
                col_d.metric("Cost", f"${trace.get('estimated_cost_usd', 0):.4f}")

                st.caption(f"Query: *{trace.get('query', '')}*")
                st.caption(f"Run ID: `{run_id}` | Started: {trace.get('started_at', '')}")

                st.divider()
                st.subheader("Spans")
                _render_spans_table(trace.get("spans", []))


