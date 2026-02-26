"""
tests/unit/test_synthesizer.py — Unit tests for agent/synthesizer.py.

Tests cover:
  - Outline parsing from valid/invalid JSON
  - Reference section formatting (title + URL)
  - Fallback report generation
  - synthesize() with mocked LLM calls
  - Edge cases: no summaries, LLM failure, empty sections
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from agent.synthesizer import (
    Synthesizer,
    _parse_outline,
    _build_references,
    _fallback_report,
    _format_summaries_for_prompt,
    _format_sources_for_prompt,
)
from agent.state import ResearchState, ResearchStatus, PageSummary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_summary(n: int, title: str = None, url: str = None) -> PageSummary:
    return PageSummary(
        url=url or f"https://example.com/page{n}",
        title=title or f"Page {n} Title",
        summary=f"• Fact A from page {n}\n• Fact B from page {n}",
        subquery=f"subquery {n}",
        round_number=1,
        word_count=10,
        source="tavily",
    )


def make_state(n_summaries: int = 3) -> ResearchState:
    state = ResearchState(query="What are the latest solid-state battery breakthroughs?")
    for i in range(1, n_summaries + 1):
        state.add_summary(make_summary(i))
    return state


def mock_client_with(outline_json: str, report_text: str) -> MagicMock:
    """Return a mock LLMClient whose generate() alternates outline then report."""
    client = MagicMock()
    response1 = MagicMock()
    response1.output_text = outline_json
    response2 = MagicMock()
    response2.output_text = report_text
    client.generate.side_effect = [response1, response2]
    return client


# ── _parse_outline ─────────────────────────────────────────────────────────────

class TestParseOutline:
    def test_valid_json(self):
        text = '{"sections": ["Section A", "Section B", "Section C"]}'
        result = _parse_outline(text)
        assert result == ["Section A", "Section B", "Section C"]

    def test_strips_markdown_fences(self):
        text = '```json\n{"sections": ["A", "B"]}\n```'
        result = _parse_outline(text)
        assert result == ["A", "B"]

    def test_empty_sections_filtered(self):
        text = '{"sections": ["A", "", "  ", "B"]}'
        result = _parse_outline(text)
        assert result == ["A", "B"]

    def test_invalid_json_returns_empty(self):
        assert _parse_outline("not json at all") == []

    def test_missing_sections_key_returns_empty(self):
        assert _parse_outline('{"other": ["A", "B"]}') == []

    def test_empty_string_returns_empty(self):
        assert _parse_outline("") == []

    def test_sections_stripped(self):
        text = '{"sections": ["  Leading spaces  ", "  Trailing  "]}'
        result = _parse_outline(text)
        assert result == ["Leading spaces", "Trailing"]


# ── _build_references ──────────────────────────────────────────────────────────

class TestBuildReferences:
    def test_contains_references_header(self):
        summaries = [make_summary(1)]
        result = _build_references(summaries)
        assert "## References" in result

    def test_contains_title_and_url(self):
        s = make_summary(1, title="Battery Study 2025", url="https://battery.org/study")
        result = _build_references([s])
        assert "Battery Study 2025" in result
        assert "https://battery.org/study" in result

    def test_numbered_correctly(self):
        summaries = [make_summary(i) for i in range(1, 4)]
        result = _build_references(summaries)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_no_title_uses_untitled(self):
        s = make_summary(1, title=None)
        s.title = None
        result = _build_references([s])
        assert "Untitled" in result

    def test_multiple_sources_all_present(self):
        summaries = [make_summary(i, url=f"https://source{i}.com") for i in range(1, 6)]
        result = _build_references(summaries)
        for i in range(1, 6):
            assert f"https://source{i}.com" in result


# ── _fallback_report ──────────────────────────────────────────────────────────

class TestFallbackReport:
    def test_contains_query(self):
        state = make_state(2)
        result = _fallback_report(state.query, state.page_summaries)
        assert state.query in result

    def test_contains_all_summaries(self):
        state = make_state(3)
        result = _fallback_report(state.query, state.page_summaries)
        for s in state.page_summaries:
            assert s.summary in result

    def test_is_markdown(self):
        state = make_state(2)
        result = _fallback_report(state.query, state.page_summaries)
        assert result.startswith("# ")

    def test_note_about_unavailability(self):
        state = make_state(1)
        result = _fallback_report(state.query, state.page_summaries)
        assert "synthesis unavailable" in result.lower() or "raw findings" in result.lower()


# ── _format helpers ───────────────────────────────────────────────────────────

class TestFormatHelpers:
    def test_format_summaries_includes_title(self):
        s = make_summary(1, title="My Title")
        result = _format_summaries_for_prompt([s])
        assert "My Title" in result

    def test_format_summaries_truncates_at_300(self):
        s = make_summary(1)
        s.summary = "x" * 500
        result = _format_summaries_for_prompt([s])
        # 300 chars of summary shown, not 500
        assert "x" * 301 not in result

    def test_format_sources_includes_url(self):
        s = make_summary(1, url="https://mysite.com/article")
        result = _format_sources_for_prompt([s])
        assert "https://mysite.com/article" in result

    def test_format_sources_numbered(self):
        summaries = [make_summary(i) for i in range(1, 4)]
        result = _format_sources_for_prompt(summaries)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result


# ── Synthesizer.synthesize() ──────────────────────────────────────────────────

class TestSynthesizerSynthesize:
    def test_success_sets_status(self):
        state = make_state(3)
        outline_json = '{"sections": ["Background", "Key Findings", "Implications"]}'
        report_text = "## Background\nSome facts [1].\n\n## Key Findings\nMore facts [2].\n\n## Implications\nFinal thoughts [3]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert state.status == ResearchStatus.SUCCESS

    def test_final_report_contains_report_body(self):
        state = make_state(3)
        outline_json = '{"sections": ["A", "B"]}'
        # Must be >100 chars to pass the minimum-length check in _generate_report
        report_text = "## A\nSolid-state batteries showed 40% energy density improvement in 2025 [1]. Several companies including QuantumScape reached production milestones.\n\n## B\nCommercial availability is expected by 2027 according to industry analysts [2][3]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert "## A" in state.final_report
        assert "## B" in state.final_report

    def test_final_report_contains_references(self):
        state = make_state(2)
        outline_json = '{"sections": ["A"]}'
        report_text = "## A\nFact [1]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert "## References" in state.final_report
        assert state.page_summaries[0].url in state.final_report

    def test_outline_stored_in_state(self):
        state = make_state(3)
        outline_json = '{"sections": ["Sec1", "Sec2", "Sec3"]}'
        report_text = "## Sec1\nText [1].\n\n## Sec2\nText [2].\n\n## Sec3\nText [3]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert state.outline == ["Sec1", "Sec2", "Sec3"]

    def test_sources_stored_in_state(self):
        state = make_state(3)
        outline_json = '{"sections": ["A"]}'
        report_text = "## A\nFact [1][2][3]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert len(state.sources) == 3
        for s in state.page_summaries:
            assert s.url in state.sources

    def test_no_summaries_sets_partial(self):
        state = ResearchState(query="test query")
        client = MagicMock()

        Synthesizer(client=client).synthesize(state)

        assert state.status == ResearchStatus.PARTIAL
        client.generate.assert_not_called()

    def test_outline_llm_failure_uses_fallback_heading(self):
        state = make_state(2)
        # First call (outline) fails, second call (report) returns text
        client = MagicMock()
        client.generate.side_effect = [Exception("LLM down"), MagicMock(output_text="## Research Findings: What are the latest solid-state battery breakthroughs?\nFact [1].")]

        Synthesizer(client=client).synthesize(state)

        # Should still get a report (fallback heading used)
        assert state.final_report

    def test_report_llm_failure_uses_fallback_report(self):
        state = make_state(2)
        outline_response = MagicMock()
        outline_response.output_text = '{"sections": ["A", "B"]}'
        client = MagicMock()
        client.generate.side_effect = [outline_response, Exception("LLM down")]

        Synthesizer(client=client).synthesize(state)

        # Fallback report still sets SUCCESS (synthesize() calls record_success via fallback path)
        assert state.final_report
        assert "## References" in state.final_report

    def test_references_include_title_and_url(self):
        state = make_state(1)
        state.page_summaries[0].title = "QuantumScape Battery Report"
        state.page_summaries[0].url = "https://quantumscape.com/report"

        outline_json = '{"sections": ["A"]}'
        report_text = "## A\nFact [1]."
        client = mock_client_with(outline_json, report_text)

        Synthesizer(client=client).synthesize(state)

        assert "QuantumScape Battery Report" in state.final_report
        assert "https://quantumscape.com/report" in state.final_report

    def test_caps_summaries_at_top_k(self):
        """Synthesizer should not send all 100 summaries — capped at top_k_summaries."""
        state = make_state(0)
        for i in range(50):
            state.add_summary(make_summary(i))

        outline_json = '{"sections": ["A"]}'
        report_text = "## A\nFact [1]."
        client = mock_client_with(outline_json, report_text)

        with patch("agent.synthesizer.settings") as mock_settings:
            mock_settings.top_k_summaries = 10
            Synthesizer(client=client).synthesize(state)

        # sources should be capped at 10
        assert len(state.sources) == 10
