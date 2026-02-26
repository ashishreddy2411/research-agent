"""
Unit tests for tools/fetch.py

What we test (no real HTTP calls):
  - _extract_title_from_markdown() — pulls # heading from Jina output
  - _failed() helper — returns correct FetchResult shape
  - FetchResult.word_count — correct word counting
  - Jina tier: handles 429, non-200, short content, timeout
  - trafilatura tier: handles empty extraction, timeout
  - fetch_page() waterfall — falls back from Jina to trafilatura on failure
"""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from tools.fetch import (
    FetchResult,
    fetch_page,
    _fetch_via_jina,
    _fetch_via_trafilatura,
    _failed,
    _extract_title_from_markdown,
)


# ── FetchResult ────────────────────────────────────────────────────────────────

class TestFetchResult:
    def test_word_count_counts_content_words(self):
        result = FetchResult(
            url="https://example.com",
            content="one two three four five",
            title="Test",
            success=True,
            source="jina",
            error=None,
        )
        assert result.word_count == 5

    def test_word_count_zero_on_empty_content(self):
        result = FetchResult(
            url="https://example.com",
            content="",
            title="",
            success=False,
            source="failed",
            error="timeout",
        )
        assert result.word_count == 0


# ── _failed() helper ───────────────────────────────────────────────────────────

class TestFailedHelper:
    def test_returns_failed_result_shape(self):
        result = _failed("https://example.com", "some error")
        assert result.success is False
        assert result.source == "failed"
        assert result.content == ""
        assert result.error == "some error"
        assert result.url == "https://example.com"

    def test_failed_title_is_empty(self):
        result = _failed("https://x.com", "err")
        assert result.title == ""


# ── _extract_title_from_markdown() ────────────────────────────────────────────

class TestExtractTitleFromMarkdown:
    def test_extracts_h1_from_first_line(self):
        text = "# Battery Technology Overview\n\nSome content here."
        assert _extract_title_from_markdown(text) == "Battery Technology Overview"

    def test_returns_empty_when_no_heading(self):
        text = "Just some plain text without any heading."
        assert _extract_title_from_markdown(text) == ""

    def test_skips_empty_lines_before_heading(self):
        text = "\n\n# The Title\n\nContent."
        assert _extract_title_from_markdown(text) == "The Title"

    def test_does_not_pick_up_h2_or_deeper(self):
        text = "## Section Title\nContent."
        assert _extract_title_from_markdown(text) == ""

    def test_handles_empty_string(self):
        assert _extract_title_from_markdown("") == ""


# ── Jina Reader tier ───────────────────────────────────────────────────────────

class TestFetchViaJina:
    @patch("tools.fetch.httpx.get")
    def test_success_returns_content(self, mock_get):
        """Happy path: Jina returns good markdown content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Article Title\n\n" + "Content about batteries. " * 20
        mock_get.return_value = mock_response

        result = _fetch_via_jina("https://example.com/article")

        assert result.success is True
        assert result.source == "jina"
        assert result.title == "Article Title"
        assert result.error is None

    @patch("tools.fetch.httpx.get")
    def test_returns_failed_on_429(self, mock_get):
        """Rate limit → failed, not exception."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = _fetch_via_jina("https://example.com")
        assert result.success is False
        assert "429" in result.error

    @patch("tools.fetch.httpx.get")
    def test_returns_failed_on_non_200(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = _fetch_via_jina("https://example.com")
        assert result.success is False
        assert "503" in result.error

    @patch("tools.fetch.httpx.get")
    def test_returns_failed_when_content_too_short(self, mock_get):
        """Content under 200 chars → failed (likely paywall or error page)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Too short."
        mock_get.return_value = mock_response

        result = _fetch_via_jina("https://example.com")
        assert result.success is False

    @patch("tools.fetch.httpx.get")
    def test_returns_failed_on_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")

        result = _fetch_via_jina("https://example.com")
        assert result.success is False
        assert "timeout" in result.error.lower()

    @patch("tools.fetch.httpx.get")
    def test_returns_failed_on_unexpected_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("something broke")

        result = _fetch_via_jina("https://example.com")
        assert result.success is False
        assert result.error is not None


# ── fetch_page() waterfall ─────────────────────────────────────────────────────

class TestFetchPageWaterfall:
    @patch("tools.fetch._fetch_via_trafilatura")
    @patch("tools.fetch._fetch_via_jina")
    def test_returns_jina_result_when_jina_succeeds(self, mock_jina, mock_trafilatura):
        """If Jina succeeds, trafilatura is never called."""
        mock_jina.return_value = FetchResult(
            url="https://x.com",
            content="Good content " * 30,
            title="Title",
            success=True,
            source="jina",
            error=None,
        )

        result = fetch_page("https://x.com")

        assert result.success is True
        assert result.source == "jina"
        mock_trafilatura.assert_not_called()

    @patch("tools.fetch._fetch_via_trafilatura")
    @patch("tools.fetch._fetch_via_jina")
    def test_falls_back_to_trafilatura_when_jina_fails(self, mock_jina, mock_trafilatura):
        """If Jina fails, trafilatura is called."""
        mock_jina.return_value = FetchResult(
            url="https://x.com", content="", title="", success=False, source="failed",
            error="429"
        )
        mock_trafilatura.return_value = FetchResult(
            url="https://x.com",
            content="Extracted content " * 20,
            title="Title",
            success=True,
            source="trafilatura",
            error=None,
        )

        result = fetch_page("https://x.com")

        assert result.success is True
        assert result.source == "trafilatura"
        mock_trafilatura.assert_called_once()

    @patch("tools.fetch._fetch_via_trafilatura")
    @patch("tools.fetch._fetch_via_jina")
    def test_returns_failed_when_both_tiers_fail(self, mock_jina, mock_trafilatura):
        """If both tiers fail, final result is failed — no exception raised."""
        mock_jina.return_value = _failed("https://x.com", "jina error")
        mock_trafilatura.return_value = _failed("https://x.com", "trafilatura error")

        result = fetch_page("https://x.com")

        assert result.success is False
        assert result.source == "failed"
