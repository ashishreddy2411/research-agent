"""
tools/search.py — Tavily web search API wrapper.

THE CORE CONCEPT: Web search as a tool
  SQL Agent had execute_sql() — queries a database, returns rows.
  Research Agent has search() — queries the web, returns pages.

  Same pattern: structured input → structured output → fed back to LLM.
  The LLM doesn't call Google directly. It calls search(), gets back a
  structured list of SearchResult objects, and reasons over those.

WHY TAVILY AND NOT GOOGLE/BING/DUCKDUCKGO:
  Most search APIs return snippets — 150-300 character extracts from the page.
  Tavily returns the full extracted page content alongside the URL.

  For an LLM agent, this matters enormously:
    - DuckDuckGo: gives you "Apple announced new battery tech..." (150 chars)
    - Tavily: gives you the full article text, already cleaned

  One Tavily call replaces two steps: search + fetch + extract.
  For most URLs in a research run, Tavily's content is sufficient.
  We only need the separate fetch_page() tool for URLs Tavily didn't
  fully extract or for follow-up URLs discovered during synthesis.

TAVILY API:
  POST https://api.tavily.com/search
  Body: {
    "api_key": "...",
    "query": "battery technology breakthroughs 2025",
    "max_results": 10,
    "include_raw_content": true,   ← full page text, not just snippet
    "search_depth": "basic"        ← "advanced" costs 2 credits (deeper)
  }
  Response: {
    "query": "...",
    "results": [
      {
        "url": "https://...",
        "title": "...",
        "content": "...",          ← cleaned extract (~500 chars)
        "raw_content": "...",      ← full page text (if include_raw_content=True)
        "score": 0.94              ← Tavily relevance score 0.0-1.0
      }
    ]
  }

CREDITS:
  - "basic" search = 1 credit per call
  - "advanced" search = 2 credits per call
  - Free tier: 1,000 credits/month
  - A 3-round research run with 4 subqueries uses ~12-15 credits

USAGE:
  from tools.search import search
  results = search("solid state battery commercialization timeline")
  for r in results:
      print(r.url, r.score)
      print(r.content[:300])
"""

import httpx
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import settings
from observability.logging import get_logger

logger = get_logger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    One result from a Tavily search.

    content vs raw_content:
      content    — Tavily's short cleaned extract (~500 chars). Always present.
      raw_content — Full page text extracted by Tavily. Present when
                    include_raw_content=True. Can be 5,000-50,000 chars.

    We prefer raw_content for summarization (more signal) and fall back
    to content when raw_content is empty (Tavily couldn't extract full page).

    score: Tavily's relevance score 0.0-1.0. Higher = more relevant to query.
    We don't filter by score here — the context filter in Phase 2 does that
    using our own cosine similarity against the original research question.
    """
    url: str
    title: str
    content: str           # Tavily's short extract — always present
    raw_content: str       # Full page text — present if Tavily extracted it
    score: float           # Tavily relevance score 0.0-1.0
    query: str             # which search query produced this result
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def best_content(self) -> str:
        """
        Return the richest available content.

        raw_content is preferred — more text = more signal for summarization.
        Fall back to content (short extract) when raw_content is empty or tiny.

        A raw_content shorter than 200 chars usually means Tavily hit a
        JavaScript-only page and couldn't extract the body — fall back to snippet.
        """
        if self.raw_content and len(self.raw_content) > 200:
            return self.raw_content
        return self.content

    @property
    def word_count(self) -> int:
        return len(self.best_content.split())


# ── Search function ────────────────────────────────────────────────────────────

def search(
    query: str,
    *,
    max_results: int | None = None,
    search_depth: str | None = None,
) -> list[SearchResult]:
    """
    Search the web via Tavily and return structured results with page content.

    Args:
        query:        The search query. Should be a specific, targeted question.
                      "What is the current state of solid-state battery production?"
                      works better than "batteries".
        max_results:  Override settings.max_search_results for this call.
        search_depth: Override settings.search_depth — "basic" or "advanced".

    Returns:
        List of SearchResult objects sorted by Tavily relevance score (highest first).
        Empty list if the API call fails (never raises — caller handles empty list).

    Note on errors:
        This function logs the error and returns [] rather than raising.
        The research loop treats an empty result list as a failed search and
        can retry or skip — it doesn't crash. This matches the SQL Agent pattern
        of never-raise executors.
    """
    n = max_results or settings.max_search_results
    depth = search_depth or settings.search_depth

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": n,
        "search_depth": depth,
        "include_raw_content": True,   # the whole point of using Tavily
        "include_answer": False,        # we'll synthesize our own answer
    }

    try:
        data = _tavily_request(payload, query)
    except Exception:
        # _tavily_request logs all failures; return empty on final failure
        return []

    results = []
    for item in data.get("results", []):
        results.append(SearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            content=item.get("content", ""),
            raw_content=item.get("raw_content") or "",
            score=float(item.get("score", 0.0)),
            query=query,
        ))

    # Tavily returns results sorted by score already, but be explicit
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ── Retry-wrapped Tavily call ────────────────────────────────────────────────

from tools.retry import retry_with_backoff


@retry_with_backoff(
    max_retries=settings.max_fetch_retries,
    base_delay=1.0,
    retryable_exceptions=(httpx.TimeoutException,),
)
def _tavily_post(payload: dict) -> httpx.Response:
    """Retry-wrapped POST to Tavily. Only retries on timeout, not on 4xx/5xx."""
    return httpx.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=30.0,
    )


def _tavily_request(payload: dict, query: str) -> dict:
    """
    Make the Tavily API call with retry on transient failures.

    Retries on timeouts. Raises on final failure so search() can catch and return [].
    """
    try:
        response = _tavily_post(payload)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        logger.warning("Tavily timeout (all retries exhausted) for query: %r", query)
        raise
    except httpx.HTTPStatusError as e:
        logger.warning("Tavily HTTP %d for query: %r", e.response.status_code, query)
        raise
    except Exception as e:
        logger.warning("Tavily unexpected error (%s) for query: %r", type(e).__name__, query)
        raise
