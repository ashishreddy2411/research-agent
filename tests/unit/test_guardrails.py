"""
tests/unit/test_guardrails.py — Unit tests for agent/guardrails.py

Covers: validate_query(), is_safe_url(), check_citation_bounds(),
        deduplicate_queries().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from agent.guardrails import (
    validate_query,
    is_safe_url,
    check_citation_bounds,
    deduplicate_queries,
    MIN_QUERY_LENGTH,
    MAX_QUERY_LENGTH,
)


# ── validate_query ────────────────────────────────────────────────────────────

class TestValidateQuery:
    def test_valid_query_returned(self):
        q = "What are the effects of climate change on agriculture?"
        assert validate_query(q) == q

    def test_strips_leading_trailing_whitespace(self):
        result = validate_query("  What are the effects of climate change?  ")
        assert result == "What are the effects of climate change?"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_query("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_query("   ")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            validate_query("short")  # 5 chars < MIN_QUERY_LENGTH

    def test_exactly_min_length_passes(self):
        query = "x" * MIN_QUERY_LENGTH
        assert validate_query(query) == query

    def test_one_below_min_raises(self):
        with pytest.raises(ValueError, match="too short"):
            validate_query("x" * (MIN_QUERY_LENGTH - 1))

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            validate_query("x" * (MAX_QUERY_LENGTH + 1))

    def test_exactly_max_length_passes(self):
        query = "x" * MAX_QUERY_LENGTH
        assert validate_query(query) == query

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            validate_query(None)

    def test_integer_raises(self):
        with pytest.raises(ValueError):
            validate_query(42)

    def test_error_message_includes_length(self):
        short = "x" * 5
        try:
            validate_query(short)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "5" in str(e)

    def test_valid_question_with_punctuation(self):
        q = "What are the effects of CRISPR-Cas9 on gene therapy (2023-2024)?"
        result = validate_query(q)
        assert result == q


# ── is_safe_url ───────────────────────────────────────────────────────────────

class TestIsSafeUrl:
    # Valid URLs
    def test_https_url_safe(self):
        assert is_safe_url("https://www.example.com/article") is True

    def test_http_url_safe(self):
        assert is_safe_url("http://news.example.org/story") is True

    def test_url_with_path_safe(self):
        assert is_safe_url("https://en.wikipedia.org/wiki/CRISPR") is True

    def test_url_with_port_safe(self):
        assert is_safe_url("https://example.com:8080/api") is True

    def test_url_with_query_string_safe(self):
        assert is_safe_url("https://example.com/search?q=crispr&lang=en") is True

    # Blocked schemes
    def test_file_scheme_blocked(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert is_safe_url("ftp://example.com/file.txt") is False

    def test_data_uri_blocked(self):
        assert is_safe_url("data:text/html,<script>alert(1)</script>") is False

    def test_javascript_blocked(self):
        assert is_safe_url("javascript:alert(1)") is False

    # Blocked hosts
    def test_localhost_blocked(self):
        assert is_safe_url("http://localhost/admin") is False

    def test_127_0_0_1_blocked(self):
        assert is_safe_url("http://127.0.0.1/") is False

    def test_127_x_x_x_blocked(self):
        assert is_safe_url("http://127.0.0.2/secret") is False

    def test_0_0_0_0_blocked(self):
        assert is_safe_url("http://0.0.0.0/") is False

    def test_private_10_range_blocked(self):
        assert is_safe_url("http://10.0.0.1/internal") is False

    def test_private_172_16_range_blocked(self):
        assert is_safe_url("http://172.16.0.1/") is False

    def test_private_172_31_range_blocked(self):
        assert is_safe_url("http://172.31.255.255/") is False

    def test_private_192_168_range_blocked(self):
        assert is_safe_url("http://192.168.1.1/router") is False

    def test_ipv6_loopback_blocked(self):
        assert is_safe_url("http://::1/") is False

    # Edge cases
    def test_empty_string_blocked(self):
        assert is_safe_url("") is False

    def test_none_blocked(self):
        assert is_safe_url(None) is False

    def test_non_string_blocked(self):
        assert is_safe_url(12345) is False

    def test_just_scheme_blocked(self):
        assert is_safe_url("https://") is False

    def test_172_15_not_blocked(self):
        # 172.15.x.x is NOT in the private range (only 172.16-172.31)
        assert is_safe_url("http://172.15.0.1/public") is True

    def test_172_32_not_blocked(self):
        # 172.32.x.x is also public
        assert is_safe_url("http://172.32.0.1/public") is True


# ── check_citation_bounds ────────────────────────────────────────────────────

class TestCheckCitationBounds:
    def test_all_valid_citations_returns_empty(self):
        report = "Fact one [1]. Fact two [2]. Fact three [3]."
        assert check_citation_bounds(report, n_sources=5) == []

    def test_out_of_bounds_citation_detected(self):
        report = "Claim one [1]. Claim two [25]."
        result = check_citation_bounds(report, n_sources=10)
        assert result == [25]

    def test_multiple_out_of_bounds_returned(self):
        report = "A [1] B [50] C [99]."
        result = check_citation_bounds(report, n_sources=10)
        assert 50 in result
        assert 99 in result

    def test_zero_citation_is_out_of_bounds(self):
        report = "Claim [0]."
        result = check_citation_bounds(report, n_sources=5)
        assert 0 in result

    def test_exactly_n_sources_valid(self):
        report = "Fact [5]."
        assert check_citation_bounds(report, n_sources=5) == []

    def test_n_sources_plus_one_is_out_of_bounds(self):
        report = "Fact [6]."
        result = check_citation_bounds(report, n_sources=5)
        assert result == [6]

    def test_empty_report_returns_empty(self):
        assert check_citation_bounds("", n_sources=10) == []

    def test_no_citations_returns_empty(self):
        assert check_citation_bounds("No citations here.", n_sources=5) == []

    def test_n_sources_zero_returns_empty(self):
        # n_sources=0 means no valid citations; guard triggers
        assert check_citation_bounds("[1][2]", n_sources=0) == []

    def test_n_sources_negative_returns_empty(self):
        assert check_citation_bounds("[1]", n_sources=-1) == []

    def test_duplicates_deduplicated_in_result(self):
        report = "A [99] B [99] C [99]."
        result = check_citation_bounds(report, n_sources=5)
        assert result.count(99) == 1

    def test_result_sorted(self):
        report = "[20] [5] [15]."
        result = check_citation_bounds(report, n_sources=3)
        assert result == sorted(result)


# ── deduplicate_queries ───────────────────────────────────────────────────────

class TestDeduplicateQueries:
    def test_empty_list_returns_empty(self):
        assert deduplicate_queries([]) == []

    def test_single_item_unchanged(self):
        assert deduplicate_queries(["only query"]) == ["only query"]

    def test_no_duplicates_unchanged(self):
        queries = ["query one", "query two", "query three"]
        assert deduplicate_queries(queries) == queries

    def test_exact_duplicates_removed(self):
        queries = ["solid state battery", "electrolyte research", "solid state battery"]
        result = deduplicate_queries(queries)
        assert result == ["solid state battery", "electrolyte research"]

    def test_case_insensitive_dedup(self):
        queries = ["Solid State Battery", "solid state battery"]
        result = deduplicate_queries(queries)
        assert len(result) == 1
        assert result[0] == "Solid State Battery"  # first occurrence kept

    def test_preserves_order(self):
        queries = ["c", "a", "b", "a"]
        result = deduplicate_queries(queries)
        assert result == ["c", "a", "b"]

    def test_whitespace_difference_considered_duplicate(self):
        # "  query  " normalized to "query" — same as "query"
        queries = ["  solid state battery  ", "solid state battery"]
        result = deduplicate_queries(queries)
        assert len(result) == 1

    def test_different_queries_all_kept(self):
        queries = ["CRISPR gene editing", "mRNA vaccines", "solid state batteries"]
        assert len(deduplicate_queries(queries)) == 3

    def test_all_duplicates_collapsed_to_one(self):
        queries = ["same query"] * 10
        result = deduplicate_queries(queries)
        assert result == ["same query"]
