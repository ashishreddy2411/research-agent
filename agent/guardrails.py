"""
agent/guardrails.py — Input validation and output safety checks.

WHAT GUARDRAILS DO:
  They catch bad inputs before they waste LLM calls or produce garbage output,
  and verify outputs before they reach the user.

  Without guardrails:
    - run_research("") → planner sends an empty string to the LLM
    - run_research("x" * 5000) → oversized prompt, unexpected cost
    - Report with [99] citation when there are 20 sources → broken references

  With guardrails:
    - Bad input is rejected immediately with a clear error, no API calls made
    - Citation bounds are verified before the report is returned

LAYERS COVERED:
  1. Input (query validation)     — before any LLM call
  2. URL safety                   — before any fetch call
  3. Citation bounds              — after synthesis, before returning

USAGE:
  from agent.guardrails import validate_query, is_safe_url, check_citation_bounds

  # Raises ValueError with a clear message on bad input
  clean = validate_query(query)

  # Returns False for unsafe URLs (localhost, file://, empty)
  if not is_safe_url(url):
      skip()

  # Returns list of out-of-bounds citation numbers found in report
  bad = check_citation_bounds(report_text, n_sources=20)
  if bad:
      log_warning(f"Citation(s) out of range: {bad}")
"""

import re


# ── Query validation ──────────────────────────────────────────────────────────

MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 500


def validate_query(query: str) -> str:
    """
    Validate and clean a research query before running the pipeline.

    Returns the stripped query if valid.
    Raises ValueError with a human-readable message if invalid.

    Checks:
      - Not None or non-string
      - Not empty or whitespace-only
      - At least MIN_QUERY_LENGTH characters (catches single-word nonsense)
      - At most MAX_QUERY_LENGTH characters (prevents prompt injection bloat)
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    if len(query) < MIN_QUERY_LENGTH:
        raise ValueError(
            f"Query too short ({len(query)} chars). "
            f"Minimum is {MIN_QUERY_LENGTH} characters. "
            f"Try a more specific question."
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query too long ({len(query)} chars). "
            f"Maximum is {MAX_QUERY_LENGTH} characters. "
            f"Please shorten your question."
        )

    return query


# ── URL safety ────────────────────────────────────────────────────────────────

# Patterns that indicate an internal/unsafe URL target
_BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0"
    r"|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+"
    r"|192\.168\.\d+\.\d+|::1)$",
    re.IGNORECASE,
)


def is_safe_url(url: str) -> bool:
    """
    Return True if the URL is safe to fetch.

    Blocks:
      - Empty or non-string URLs
      - Non-http/https schemes (file://, ftp://, data://, etc.)
      - Localhost and private IP ranges (SSRF prevention)

    This is a fast structural check, not a full security audit.
    It prevents the most common classes of URL-based attacks in agent systems.
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()

    # Must start with http:// or https://
    if not url.startswith(("http://", "https://")):
        return False

    # Extract host from URL
    try:
        # Simple host extraction — handles http://host/path and http://host:port/path
        without_scheme = url.split("://", 1)[1]
        host = without_scheme.split("/")[0].split(":")[0].lower()
    except (IndexError, AttributeError):
        return False

    if not host:
        return False

    # Block internal/private hosts
    if _BLOCKED_HOSTS.match(host):
        return False

    return True


# ── Citation bounds check ─────────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def check_citation_bounds(report: str, n_sources: int) -> list[int]:
    """
    Find citation numbers in the report that exceed the number of sources.

    Returns a list of out-of-bounds citation numbers found (empty = all OK).

    Example:
        report has [1][2][25], n_sources=20 → returns [25]

    The synthesizer is instructed to use [1]..[N] citations.
    If the LLM hallucinates a [99] when there are 20 sources, the reference
    section won't have an entry for it — broken link in the final report.
    """
    if not report or n_sources <= 0:
        return []

    found = set()
    for match in _CITATION_PATTERN.finditer(report):
        n = int(match.group(1))
        if n < 1 or n > n_sources:
            found.add(n)

    return sorted(found)


# ── Subquery deduplication ────────────────────────────────────────────────────

def deduplicate_queries(queries: list[str]) -> list[str]:
    """
    Remove exact duplicate subqueries from a list, preserving order.

    The planner LLM occasionally generates two nearly-identical queries.
    Exact deduplication catches the obvious case (same string twice).
    Near-duplicate detection (embedding similarity) is a Phase 6 concern.
    """
    seen: set[str] = set()
    result = []
    for q in queries:
        normalized = q.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(q)
    return result
