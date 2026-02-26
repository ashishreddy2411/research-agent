"""
Unit tests for tools/search.py

What we test (no API calls):
  - SearchResult.best_content — picks raw_content when rich, falls back to content
  - SearchResult.word_count — correct word counting
  - search() error handling — returns [] on HTTP errors, never raises
  - Result sorting — results sorted by score descending

What we do NOT test here:
  - Whether Tavily actually returns results (that's the integration test)
  - API authentication (that's an env/infra concern)
"""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from tools.search import SearchResult, search


# ── SearchResult.best_content ──────────────────────────────────────────────────

class TestSearchResultBestContent:
    def test_prefers_raw_content_when_long(self):
        """raw_content > 200 chars → best_content returns raw_content."""
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="Short snippet.",
            raw_content="A" * 300,  # 300 chars — long enough
            score=0.9,
            query="test",
        )
        assert result.best_content == "A" * 300

    def test_falls_back_to_content_when_raw_short(self):
        """raw_content < 200 chars → best_content falls back to content."""
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="The real content is here.",
            raw_content="Too short",  # 9 chars — below threshold
            score=0.9,
            query="test",
        )
        assert result.best_content == "The real content is here."

    def test_falls_back_to_content_when_raw_empty(self):
        """raw_content empty → best_content falls back to content."""
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="Snippet content.",
            raw_content="",
            score=0.9,
            query="test",
        )
        assert result.best_content == "Snippet content."

    def test_returns_raw_at_exactly_201_chars(self):
        """Boundary: exactly 201 chars in raw_content → returns raw_content."""
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="short",
            raw_content="B" * 201,
            score=0.9,
            query="test",
        )
        assert result.best_content == "B" * 201

    def test_falls_back_at_exactly_200_chars(self):
        """Boundary: exactly 200 chars in raw_content → falls back (not > 200)."""
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="fallback content",
            raw_content="C" * 200,
            score=0.9,
            query="test",
        )
        assert result.best_content == "fallback content"


# ── SearchResult.word_count ────────────────────────────────────────────────────

class TestSearchResultWordCount:
    def test_counts_words_in_best_content(self):
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="one two three",
            raw_content="",
            score=0.9,
            query="test",
        )
        assert result.word_count == 3

    def test_counts_words_in_raw_when_long(self):
        long_raw = " ".join(["word"] * 50) + " extra"  # 51 words, > 200 chars
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="short",
            raw_content=long_raw,
            score=0.9,
            query="test",
        )
        assert result.word_count == 51

    def test_zero_words_for_empty_content(self):
        result = SearchResult(
            url="https://example.com",
            title="Test",
            content="",
            raw_content="",
            score=0.9,
            query="test",
        )
        assert result.word_count == 0


# ── search() error handling ────────────────────────────────────────────────────

class TestSearchErrorHandling:
    """
    These tests mock httpx to simulate API failures.
    We verify search() returns [] and never raises — matching the
    never-raise executor contract from SQL Agent.
    """

    @patch("tools.search.httpx.post")
    def test_returns_empty_list_on_timeout(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        results = search("test query")
        assert results == []

    @patch("tools.search.httpx.post")
    def test_returns_empty_list_on_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        mock_post.return_value = mock_response
        results = search("test query")
        assert results == []

    @patch("tools.search.httpx.post")
    def test_returns_empty_list_on_unexpected_error(self, mock_post):
        mock_post.side_effect = RuntimeError("unexpected")
        results = search("test query")
        assert results == []

    @patch("tools.search.httpx.post")
    def test_returns_empty_list_when_no_results_key(self, mock_post):
        """Tavily response missing 'results' key → empty list, no crash."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"query": "test"}  # no 'results' key
        mock_post.return_value = mock_response
        results = search("test query")
        assert results == []


# ── search() result parsing ────────────────────────────────────────────────────

class TestSearchResultParsing:
    @patch("tools.search.httpx.post")
    def test_parses_results_correctly(self, mock_post):
        """Mock a valid Tavily response — verify SearchResult fields are populated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "query": "battery tech",
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Battery Breakthrough",
                    "content": "Short snippet about batteries.",
                    "raw_content": "Full article text about solid-state batteries " * 10,
                    "score": 0.92,
                }
            ],
        }
        mock_post.return_value = mock_response

        results = search("battery tech")

        assert len(results) == 1
        assert results[0].url == "https://example.com/article"
        assert results[0].title == "Battery Breakthrough"
        assert results[0].score == 0.92
        assert results[0].query == "battery tech"

    @patch("tools.search.httpx.post")
    def test_results_sorted_by_score_descending(self, mock_post):
        """Results must come back sorted highest score first."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"url": "a.com", "title": "A", "content": "a", "raw_content": "", "score": 0.5},
                {"url": "b.com", "title": "B", "content": "b", "raw_content": "", "score": 0.9},
                {"url": "c.com", "title": "C", "content": "c", "raw_content": "", "score": 0.7},
            ]
        }
        mock_post.return_value = mock_response

        results = search("test")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("tools.search.httpx.post")
    def test_missing_fields_default_gracefully(self, mock_post):
        """Tavily result missing optional fields → defaults, no KeyError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"url": "https://x.com"}  # minimal — no title, content, score
            ]
        }
        mock_post.return_value = mock_response

        results = search("test")
        assert len(results) == 1
        assert results[0].title == ""
        assert results[0].content == ""
        assert results[0].score == 0.0
