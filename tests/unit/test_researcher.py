"""
tests/unit/test_researcher.py — Unit tests for agent/researcher.py

Covers: research(), research_into_state(), _summarize_result(),
        deduplication, content threshold, LLM failure fallback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from agent.researcher import Researcher
from agent.state import ResearchState, PageSummary
from tools.search import SearchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_search_result(
    url: str = "https://example.com/article",
    title: str = "Article Title",
    content: str = "short",
    raw_content: str = "",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        url=url,
        title=title,
        content=content,
        raw_content=raw_content,
        score=score,
        query="test query",
    )


def make_rich_result(url: str = "https://example.com/article") -> SearchResult:
    """A search result with enough content to pass the word threshold."""
    return SearchResult(
        url=url,
        title="Rich Article",
        content="short snippet",
        raw_content=" ".join(["word"] * 150),  # 150 words — passes 100-word threshold
        score=0.9,
        query="test query",
    )


def mock_client(summary_text: str = "• Fact A\n• Fact B\n• Fact C") -> MagicMock:
    client = MagicMock()
    client.generate_cheap.return_value = summary_text
    return client


# ── Researcher.research() ─────────────────────────────────────────────────────

class TestResearcherResearch:
    def test_empty_results_returns_empty(self):
        client = mock_client()
        researcher = Researcher(client=client)
        with patch("agent.researcher.search", return_value=[]):
            result = researcher.research("query", visited_urls=set(), round_number=1)
        assert result == []

    def test_already_visited_urls_skipped(self):
        client = mock_client()
        researcher = Researcher(client=client)
        results = [make_rich_result("https://already-visited.com")]
        with patch("agent.researcher.search", return_value=results):
            summaries = researcher.research(
                "query",
                visited_urls={"https://already-visited.com"},
                round_number=1,
            )
        assert summaries == []

    def test_new_urls_processed(self):
        client = mock_client("• Fact A\n• Fact B\n• Fact C\n• Fact D\n• Fact E")
        researcher = Researcher(client=client)
        results = [make_rich_result("https://new-url.com")]
        with patch("agent.researcher.search", return_value=results):
            summaries = researcher.research("query", visited_urls=set(), round_number=1)
        assert len(summaries) == 1
        assert summaries[0].url == "https://new-url.com"

    def test_round_number_stored_in_summary(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        results = [make_rich_result()]
        with patch("agent.researcher.search", return_value=results):
            summaries = researcher.research("query", visited_urls=set(), round_number=3)
        assert summaries[0].round_number == 3

    def test_multiple_results_deduplicated(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        url = "https://same.com"
        results = [make_rich_result(url), make_rich_result(url)]
        with patch("agent.researcher.search", return_value=results):
            # Only the first should be processed — second is duplicate
            summaries = researcher.research("query", visited_urls=set(), round_number=1)
        # search returns both, but researcher deduplicates against visited_urls
        # The first result is processed; second has same URL so is skipped
        assert len(summaries) == 1


# ── Researcher._summarize_result() ───────────────────────────────────────────

class TestSummarizeResult:
    def test_returns_page_summary(self):
        client = mock_client("• Fact A\n• Fact B\n• Fact C\n• Fact D\n• Fact E")
        researcher = Researcher(client=client)
        result = make_rich_result()
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary is not None
        assert isinstance(summary, PageSummary)

    def test_url_stored_in_summary(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        result = make_rich_result("https://mysite.com/article")
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary.url == "https://mysite.com/article"

    def test_thin_content_triggers_fetch(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        # result with thin content (< 100 words)
        result = make_search_result(raw_content="only a few words here short")
        fetch_result = MagicMock()
        fetch_result.success = True
        fetch_result.content = " ".join(["word"] * 200)
        fetch_result.source = "jina"
        with patch("agent.researcher.fetch_page", return_value=fetch_result):
            summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary is not None

    def test_very_thin_content_returns_none(self):
        client = mock_client()
        researcher = Researcher(client=client)
        result = make_search_result(content="few words", raw_content="")
        fetch_result = MagicMock()
        fetch_result.success = False
        fetch_result.content = ""
        with patch("agent.researcher.fetch_page", return_value=fetch_result):
            summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary is None

    def test_llm_failure_returns_none(self):
        client = MagicMock()
        client.generate_cheap.side_effect = Exception("LLM error")
        researcher = Researcher(client=client)
        result = make_rich_result()
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary is None

    def test_empty_llm_response_returns_none(self):
        client = mock_client("   ")  # whitespace only
        researcher = Researcher(client=client)
        result = make_rich_result()
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary is None

    def test_source_field_set(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        result = make_rich_result()  # has rich raw_content → tavily source
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary.source == "tavily"

    def test_word_count_calculated(self):
        summary_text = "• Word one\n• Word two three"
        client = mock_client(summary_text)
        researcher = Researcher(client=client)
        result = make_rich_result()
        summary = researcher._summarize_result(result, "query", round_number=1)
        assert summary.word_count == len(summary_text.split())


# ── Researcher.research_into_state() ─────────────────────────────────────────

class TestResearchIntoState:
    def test_adds_summaries_to_state(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        state = ResearchState(query="test")
        results = [make_rich_result("https://source1.com"), make_rich_result("https://source2.com")]
        with patch("agent.researcher.search", return_value=results):
            count = researcher.research_into_state("query", state, round_number=1)
        assert count == 2
        assert state.total_sources == 2

    def test_updates_visited_urls(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        state = ResearchState(query="test")
        results = [make_rich_result("https://tracked.com")]
        with patch("agent.researcher.search", return_value=results):
            researcher.research_into_state("query", state, round_number=1)
        assert "https://tracked.com" in state.visited_urls

    def test_returns_new_count(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        state = ResearchState(query="test")
        with patch("agent.researcher.search", return_value=[]):
            count = researcher.research_into_state("query", state, round_number=1)
        assert count == 0

    def test_skips_already_visited(self):
        client = mock_client("• F1\n• F2\n• F3\n• F4\n• F5")
        researcher = Researcher(client=client)
        state = ResearchState(query="test")
        state.visited_urls.add("https://already-seen.com")
        results = [make_rich_result("https://already-seen.com")]
        with patch("agent.researcher.search", return_value=results):
            count = researcher.research_into_state("query", state, round_number=1)
        assert count == 0
        assert state.total_sources == 0
