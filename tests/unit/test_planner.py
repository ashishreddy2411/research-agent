"""
tests/unit/test_planner.py — Unit tests for agent/planner.py

Covers: decompose(), plan(), _parse_queries(), _extract_text(), fallback behavior,
        deduplication via guardrails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from agent.planner import Planner, _parse_queries, _extract_text
from agent.state import ResearchState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def mock_client(output_text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.output_text = output_text
    client.generate.return_value = response
    return client


# ── _parse_queries ────────────────────────────────────────────────────────────

class TestParseQueries:
    def test_valid_json(self):
        text = '{"queries": ["query A", "query B", "query C"]}'
        assert _parse_queries(text) == ["query A", "query B", "query C"]

    def test_strips_markdown_fences(self):
        text = '```json\n{"queries": ["A", "B"]}\n```'
        assert _parse_queries(text) == ["A", "B"]

    def test_strips_code_fence_no_language(self):
        text = '```\n{"queries": ["X"]}\n```'
        assert _parse_queries(text) == ["X"]

    def test_empty_strings_filtered(self):
        text = '{"queries": ["A", "", "  ", "B"]}'
        assert _parse_queries(text) == ["A", "B"]

    def test_invalid_json_returns_empty(self):
        assert _parse_queries("not json") == []

    def test_missing_queries_key_returns_empty(self):
        assert _parse_queries('{"other": ["A"]}') == []

    def test_empty_string_returns_empty(self):
        assert _parse_queries("") == []

    def test_queries_stripped(self):
        text = '{"queries": ["  leading  ", "  trailing  "]}'
        result = _parse_queries(text)
        assert result == ["leading", "trailing"]

    def test_non_string_items_converted(self):
        text = '{"queries": [123, "real query"]}'
        result = _parse_queries(text)
        assert "123" in result
        assert "real query" in result


# ── _extract_text ─────────────────────────────────────────────────────────────

class TestExtractText:
    def test_output_text_attribute(self):
        response = MagicMock()
        response.output_text = "  Hello World  "
        assert _extract_text(response) == "Hello World"

    def test_falls_back_to_output_items(self):
        response = MagicMock()
        response.output_text = None
        block = MagicMock()
        block.text = "block text"
        item = MagicMock()
        item.type = "message"
        item.content = [block]
        response.output = [item]
        assert _extract_text(response) == "block text"

    def test_returns_empty_on_exception(self):
        response = MagicMock()
        response.output_text = None
        response.output = None  # will cause AttributeError
        assert _extract_text(response) == ""


# ── Planner.decompose() ───────────────────────────────────────────────────────

class TestPlannerDecompose:
    def test_returns_parsed_queries(self):
        client = mock_client('{"queries": ["angle A", "angle B", "angle C", "angle D"]}')
        planner = Planner(client=client)
        result = planner.decompose("What are the latest AI breakthroughs?")
        assert result == ["angle A", "angle B", "angle C", "angle D"]

    def test_fallback_on_llm_failure(self):
        client = MagicMock()
        client.generate.side_effect = Exception("LLM down")
        planner = Planner(client=client)
        result = planner.decompose("test query")
        assert result == ["test query"]

    def test_fallback_on_empty_response(self):
        client = mock_client("")
        planner = Planner(client=client)
        result = planner.decompose("original query")
        assert result == ["original query"]

    def test_fallback_on_invalid_json(self):
        client = mock_client("not valid json at all")
        planner = Planner(client=client)
        result = planner.decompose("original query")
        assert result == ["original query"]

    def test_never_returns_empty_list(self):
        client = mock_client('{"queries": []}')
        planner = Planner(client=client)
        result = planner.decompose("my query")
        assert len(result) >= 1
        assert result == ["my query"]

    def test_custom_n_passed_to_prompt(self):
        client = mock_client('{"queries": ["A", "B"]}')
        planner = Planner(client=client)
        planner.decompose("query", n=2)
        call_args = client.generate.call_args
        prompt_content = call_args[1]["input"][0]["content"]
        assert "2" in prompt_content

    def test_default_n_is_4(self):
        client = mock_client('{"queries": ["A", "B", "C", "D"]}')
        planner = Planner(client=client)
        planner.decompose("query")
        call_args = client.generate.call_args
        prompt_content = call_args[1]["input"][0]["content"]
        assert "4" in prompt_content

    def test_deduplicates_identical_subqueries(self):
        client = mock_client('{"queries": ["same query", "same query", "different query"]}')
        planner = Planner(client=client)
        result = planner.decompose("test")
        assert len(result) == 2
        assert result.count("same query") == 1


# ── Planner.plan() ────────────────────────────────────────────────────────────

class TestPlannerPlan:
    def test_writes_subqueries_to_state(self):
        client = mock_client('{"queries": ["A", "B", "C"]}')
        planner = Planner(client=client)
        state = ResearchState(query="What are battery breakthroughs?")
        planner.plan(state)
        assert state.subqueries == ["A", "B", "C"]

    def test_fallback_subquery_is_original_query(self):
        client = MagicMock()
        client.generate.side_effect = Exception("fail")
        planner = Planner(client=client)
        state = ResearchState(query="original question")
        planner.plan(state)
        assert state.subqueries == ["original question"]

    def test_deduplication_in_plan(self):
        client = mock_client('{"queries": ["dup", "dup", "unique"]}')
        planner = Planner(client=client)
        state = ResearchState(query="test")
        planner.plan(state)
        assert len(state.subqueries) == 2
