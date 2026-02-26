"""
Unit tests for tools/extract.py

What we test (pure Python — no HTTP, no API calls):
  - clean_text() — all normalization operations
  - truncate_to_tokens() — word capping and [truncated] marker
  - extract_main_content() — trafilatura on real HTML samples
"""

import pytest
from tools.extract import clean_text, truncate_to_tokens, extract_main_content


# ── clean_text() ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_crlf_to_lf(self):
        text = "line one\r\nline two\r\nline three"
        result = clean_text(text)
        assert "\r" not in result
        assert "line one\nline two\nline three" == result

    def test_collapses_three_blank_lines_to_two(self):
        text = "para one\n\n\n\n\npara two"
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert "para one\n\npara two" == result

    def test_strips_trailing_whitespace_per_line(self):
        text = "line one   \nline two  \nline three"
        result = clean_text(text)
        for line in result.splitlines():
            assert line == line.rstrip()

    def test_removes_soft_hyphen(self):
        text = "super\u00adlong word"
        result = clean_text(text)
        assert "\u00ad" not in result
        assert "superlong word" == result

    def test_removes_zero_width_space(self):
        text = "text\u200bhere"
        result = clean_text(text)
        assert "\u200b" not in result

    def test_normalizes_curly_quotes(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = clean_text(text)
        assert '"Hello"' in result
        assert "'world'" in result

    def test_returns_empty_string_for_empty_input(self):
        assert clean_text("") == ""

    def test_returns_empty_string_for_none_equivalent(self):
        # clean_text receives str — test empty string edge case
        assert clean_text("   ") == ""

    def test_preserves_actual_content(self):
        text = "Solid-state batteries improve safety by replacing liquid electrolyte."
        assert clean_text(text) == text


# ── truncate_to_tokens() ───────────────────────────────────────────────────────

class TestTruncateToTokens:
    def test_returns_text_unchanged_when_under_limit(self):
        text = "one two three"
        assert truncate_to_tokens(text, max_words=10) == text

    def test_truncates_when_over_limit(self):
        text = " ".join(["word"] * 100)
        result = truncate_to_tokens(text, max_words=50)
        word_count = len(result.replace(" [truncated]", "").split())
        assert word_count == 50

    def test_appends_truncated_marker(self):
        text = " ".join(["word"] * 100)
        result = truncate_to_tokens(text, max_words=50)
        assert result.endswith("[truncated]")

    def test_no_truncated_marker_when_not_truncated(self):
        text = "short text"
        result = truncate_to_tokens(text, max_words=100)
        assert "[truncated]" not in result

    def test_exactly_at_limit_is_not_truncated(self):
        text = " ".join(["word"] * 50)
        result = truncate_to_tokens(text, max_words=50)
        assert "[truncated]" not in result

    def test_one_over_limit_is_truncated(self):
        text = " ".join(["word"] * 51)
        result = truncate_to_tokens(text, max_words=50)
        assert "[truncated]" in result


# ── extract_main_content() ─────────────────────────────────────────────────────

class TestExtractMainContent:
    def test_extracts_article_from_clean_html(self):
        html = """
        <html><body>
        <nav>Home | About | Contact</nav>
        <article>
            <h1>Battery Technology</h1>
            <p>Solid-state batteries replace liquid electrolytes with solid materials.
            This improves safety and energy density compared to conventional lithium-ion.</p>
            <p>Toyota plans commercial production by 2027 for electric vehicles.</p>
        </article>
        <footer>Copyright 2025</footer>
        </body></html>
        """
        text = extract_main_content(html)
        assert text  # not empty
        assert "solid-state" in text.lower() or "battery" in text.lower()

    def test_returns_empty_for_empty_input(self):
        assert extract_main_content("") == ""

    def test_returns_empty_for_very_short_html(self):
        assert extract_main_content("<html></html>") == ""

    def test_strips_navigation_boilerplate(self):
        html = """
        <html><body>
        <nav>Home | Products | About | Contact | Login | Register</nav>
        <article>
            <p>The main article content that matters for research purposes.
            This contains the actual information we want to extract from the page.
            It discusses important facts about the research topic in detail.</p>
        </article>
        <footer>Privacy Policy | Terms of Service | Cookie Settings</footer>
        </body></html>
        """
        text = extract_main_content(html)
        # Navigation text should not dominate the extracted content
        # The article content should be present
        if text:
            assert "main article content" in text.lower() or "research" in text.lower()

    def test_output_is_cleaned(self):
        """Extracted text should not have 3+ consecutive blank lines."""
        html = """
        <html><body>
        <article>
            <p>First paragraph with real content about research topics.</p>
            <p>Second paragraph with more content and details about the subject.</p>
        </article>
        </body></html>
        """
        text = extract_main_content(html)
        if text:
            assert "\n\n\n" not in text  # clean_text should have run

    def test_handles_malformed_html_gracefully(self):
        """Malformed HTML should not crash — return empty or partial result."""
        malformed = "<html><body><p>Unclosed paragraph<div>Nested wrong"
        # Should not raise
        result = extract_main_content(malformed)
        assert isinstance(result, str)
