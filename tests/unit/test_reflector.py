"""
tests/unit/test_reflector.py — Unit tests for agent/reflector.py

Covers: ReflectionResult, reflect(), reflect_on_state(),
        _parse_reflection(), _format_summaries(), failure fallbacks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock
from agent.reflector import (
    Reflector,
    ReflectionResult,
    _parse_reflection,
    _format_summaries,
    _extract_text,
)
from agent.state import ResearchState, PageSummary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_summary(n: int, round_num: int = 1) -> PageSummary:
    return PageSummary(
        url=f"https://example.com/{n}",
        title=f"Source {n}",
        summary=f"• Fact A from source {n}\n• Fact B",
        subquery="test query",
        round_number=round_num,
        word_count=10,
        source="tavily",
    )


def mock_client(output_text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.output_text = output_text
    client.generate.return_value = response
    return client


# ── ReflectionResult ──────────────────────────────────────────────────────────

class TestReflectionResult:
    def test_gap_found(self):
        r = ReflectionResult(has_gap=True, follow_up_query="follow up", gap_description="missing X")
        assert r.has_gap is True
        assert r.follow_up_query == "follow up"

    def test_no_gap(self):
        r = ReflectionResult(has_gap=False, follow_up_query=None, gap_description="coverage ok")
        assert r.has_gap is False
        assert r.follow_up_query is None


# ── _parse_reflection ─────────────────────────────────────────────────────────

class TestParseReflection:
    def test_gap_found(self):
        text = '{"knowledge_gap": "missing cost data", "follow_up_query": "solid state battery cost 2025"}'
        result = _parse_reflection(text)
        assert result.has_gap is True
        assert result.follow_up_query == "solid state battery cost 2025"
        assert result.gap_description == "missing cost data"

    def test_no_gap_null_follow_up(self):
        text = '{"knowledge_gap": null, "follow_up_query": null}'
        result = _parse_reflection(text)
        assert result.has_gap is False
        assert result.follow_up_query is None

    def test_no_gap_string_null(self):
        text = '{"knowledge_gap": "none", "follow_up_query": "null"}'
        result = _parse_reflection(text)
        assert result.has_gap is False

    def test_no_gap_empty_follow_up(self):
        text = '{"knowledge_gap": "", "follow_up_query": ""}'
        result = _parse_reflection(text)
        assert result.has_gap is False

    def test_strips_markdown_fences(self):
        text = '```json\n{"knowledge_gap": "gap", "follow_up_query": "search query"}\n```'
        result = _parse_reflection(text)
        assert result.has_gap is True

    def test_invalid_json_returns_no_gap(self):
        result = _parse_reflection("not json at all")
        assert result.has_gap is False
        assert result.follow_up_query is None

    def test_empty_string_returns_no_gap(self):
        result = _parse_reflection("")
        assert result.has_gap is False

    def test_gap_description_stored(self):
        text = '{"knowledge_gap": "no timeline data", "follow_up_query": "battery timeline 2025"}'
        result = _parse_reflection(text)
        assert result.gap_description == "no timeline data"


# ── _format_summaries ─────────────────────────────────────────────────────────

class TestFormatSummaries:
    def test_includes_title(self):
        s = make_summary(1)
        result = _format_summaries([s])
        assert "Source 1" in result

    def test_includes_round_number(self):
        s = make_summary(1, round_num=2)
        result = _format_summaries([s])
        assert "Round 2" in result

    def test_numbered(self):
        summaries = [make_summary(i) for i in range(3)]
        result = _format_summaries(summaries)
        assert "[1]" in result
        assert "[3]" in result

    def test_caps_at_30(self):
        summaries = [make_summary(i) for i in range(50)]
        result = _format_summaries(summaries)
        assert "[30]" in result
        assert "[31]" not in result

    def test_truncates_summary_at_500(self):
        s = make_summary(1)
        s.summary = "x" * 600
        result = _format_summaries([s])
        assert "x" * 501 not in result

    def test_empty_list(self):
        result = _format_summaries([])
        assert result == ""

    def test_uses_url_when_no_title(self):
        s = make_summary(1)
        s.title = None
        result = _format_summaries([s])
        assert s.url in result


# ── Reflector.reflect() ───────────────────────────────────────────────────────

class TestReflectorReflect:
    def test_no_summaries_returns_gap_with_original_query(self):
        reflector = Reflector(client=MagicMock())
        result = reflector.reflect(
            query="original question", summaries=[], rounds_completed=0
        )
        assert result.has_gap is True
        assert result.follow_up_query == "original question"

    def test_gap_found(self):
        json_text = '{"knowledge_gap": "missing cost data", "follow_up_query": "battery cost comparison"}'
        client = mock_client(json_text)
        reflector = Reflector(client=client)
        result = reflector.reflect(
            query="test", summaries=[make_summary(1)], rounds_completed=1
        )
        assert result.has_gap is True
        assert result.follow_up_query == "battery cost comparison"

    def test_no_gap(self):
        json_text = '{"knowledge_gap": null, "follow_up_query": null}'
        client = mock_client(json_text)
        reflector = Reflector(client=client)
        result = reflector.reflect(
            query="test", summaries=[make_summary(1)], rounds_completed=1
        )
        assert result.has_gap is False

    def test_llm_failure_returns_no_gap(self):
        client = MagicMock()
        client.generate.side_effect = Exception("LLM down")
        reflector = Reflector(client=client)
        result = reflector.reflect(
            query="test", summaries=[make_summary(1)], rounds_completed=1
        )
        assert result.has_gap is False

    def test_prompt_includes_query(self):
        client = mock_client('{"knowledge_gap": null, "follow_up_query": null}')
        reflector = Reflector(client=client)
        reflector.reflect(query="unique_query_xyz", summaries=[make_summary(1)], rounds_completed=1)
        call_args = client.generate.call_args
        prompt = call_args[1]["input"][0]["content"]
        assert "unique_query_xyz" in prompt


# ── Reflector.reflect_on_state() ──────────────────────────────────────────────

class TestReflectorReflectOnState:
    def test_gap_added_to_state(self):
        json_text = '{"knowledge_gap": "missing X", "follow_up_query": "follow up search"}'
        client = mock_client(json_text)
        reflector = Reflector(client=client)
        state = ResearchState(query="test")
        state.add_summary(make_summary(1))
        result = reflector.reflect_on_state(state)
        assert result.has_gap is True
        assert "follow up search" in state.knowledge_gaps

    def test_no_gap_not_added_to_state(self):
        json_text = '{"knowledge_gap": null, "follow_up_query": null}'
        client = mock_client(json_text)
        reflector = Reflector(client=client)
        state = ResearchState(query="test")
        state.add_summary(make_summary(1))
        reflector.reflect_on_state(state)
        assert state.knowledge_gaps == []

    def test_returns_reflection_result(self):
        json_text = '{"knowledge_gap": null, "follow_up_query": null}'
        client = mock_client(json_text)
        reflector = Reflector(client=client)
        state = ResearchState(query="test")
        state.add_summary(make_summary(1))
        result = reflector.reflect_on_state(state)
        assert isinstance(result, ReflectionResult)
