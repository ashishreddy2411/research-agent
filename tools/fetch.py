"""
tools/fetch.py — Fetch a URL and return clean text content.

THE CORE CONCEPT: Following leads
  search() returns URLs + content for 10 results per query.
  But research isn't just reading the first 10 results.

  Real research follows leads:
    - The LLM reads a summary and spots a reference to a specific paper
    - A Wikipedia page links to the primary source
    - One article mentions a company whose site has the actual data
    - A cited URL wasn't in the search results at all

  fetch_page() is how the agent follows those leads. It fetches any URL
  and returns clean text — the same format as Tavily's raw_content.

TWO-TIER FETCHING STRATEGY:
  The challenge: the web is messy. Some pages are clean HTML. Others are
  JavaScript single-page apps that return empty HTML without a browser.
  We handle this with a two-tier waterfall:

  Tier 1 — Jina Reader (r.jina.ai/{url})
    Jina Reader is a free service that takes any URL, renders it in a
    headless browser, extracts the main content, and returns clean Markdown.

    Why it's fast: Jina caches popular pages. Most news/blog URLs return
    immediately from cache.

    How it works — just one HTTP GET:
      GET https://r.jina.ai/https://example.com/article
      → returns: # Article Title\n\nParagraph text...\n\n[links]

    Handles JavaScript-heavy sites because it runs a real browser.
    No API key needed for basic rate-limited usage.

  Tier 2 — trafilatura (local extraction)
    If Jina fails (rate limit, timeout, or very new URL not cached),
    we fall back to fetching the raw HTML ourselves with httpx and
    extracting the main content with trafilatura.

    trafilatura is a Python library that removes navigation, ads, headers,
    footers, and sidebars — keeping only the main article text.
    Fastest benchmark performance among Python extraction libraries.
    Works entirely locally, no external dependency.
    Does NOT handle JavaScript — returns empty for SPAs.

  Why NOT Playwright (full browser)?
    Playwright runs a 100-200MB Chromium instance per context.
    50-100 URLs in a research run would require 5-20GB RAM.
    Jina Reader handles JavaScript without us managing browsers.
    Playwright is the right tool when you control exactly which sites
    you need. For general web research, Jina + trafilatura is sufficient.

USAGE:
  from tools.fetch import fetch_page

  result = fetch_page("https://example.com/article")
  if result.success:
      print(result.content[:500])
      print(f"Source: {result.source}")  # "jina" or "trafilatura"
  else:
      print(f"Failed: {result.error}")
"""

import httpx
import trafilatura
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import settings


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """
    The outcome of fetching a URL.

    success=False means both tiers failed.
    In that case, content is empty and error explains why.
    The research loop skips failed fetches — it never raises on fetch failure.

    source tells you which tier succeeded:
      "jina"        — Jina Reader handled it
      "trafilatura" — local extraction handled it
      "tavily"      — content came from Tavily's search result (no fetch needed)
      "failed"      — both tiers failed
    """
    url: str
    content: str           # clean text or markdown — empty if failed
    title: str
    success: bool
    source: str            # "jina", "trafilatura", or "failed"
    error: str | None
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def word_count(self) -> int:
        return len(self.content.split())


# ── Main function ──────────────────────────────────────────────────────────────

def fetch_page(url: str) -> FetchResult:
    """
    Fetch a URL and return clean text content.

    Tries Jina Reader first, falls back to trafilatura.
    Never raises — returns FetchResult(success=False) on all failures.

    Args:
        url: Any HTTP/HTTPS URL.

    Returns:
        FetchResult with clean text content, or success=False if both tiers fail.
    """
    # Tier 1: Jina Reader
    result = _fetch_via_jina(url)
    if result.success:
        return result

    # Tier 2: trafilatura
    return _fetch_via_trafilatura(url)


# ── Tier 1: Jina Reader ────────────────────────────────────────────────────────

def _fetch_via_jina(url: str) -> FetchResult:
    """
    Fetch via Jina Reader (r.jina.ai/{url}).

    Jina prepends r.jina.ai/ to the URL, renders it in a headless browser,
    strips boilerplate, and returns clean Markdown. One HTTP GET.

    Rate limit: ~20 requests/minute on the free tier.
    If rate-limited (429), we fall through to trafilatura.
    """
    jina_url = f"https://r.jina.ai/{url}"

    try:
        response = httpx.get(
            jina_url,
            timeout=settings.fetch_timeout_seconds,
            headers={
                # Tell Jina we want plain text output (not HTML)
                "Accept": "text/plain",
                # User-Agent so Jina knows this is a legitimate research tool
                "X-No-Cache": "false",  # allow Jina to use its cache (faster)
            },
            follow_redirects=True,
        )

        if response.status_code == 429:
            return _failed(url, "Jina rate limit (429) — falling back to trafilatura")

        if response.status_code != 200:
            return _failed(url, f"Jina returned HTTP {response.status_code}")

        content = response.text.strip()

        # If Jina returned very little content, it likely hit a paywall or error page
        if len(content) < 200:
            return _failed(url, f"Jina returned too little content ({len(content)} chars)")

        # Extract title from first Markdown heading if present
        title = _extract_title_from_markdown(content)

        return FetchResult(
            url=url,
            content=content,
            title=title,
            success=True,
            source="jina",
            error=None,
        )

    except httpx.TimeoutException:
        return _failed(url, f"Jina timeout after {settings.fetch_timeout_seconds}s")
    except Exception as e:
        return _failed(url, f"Jina error: {type(e).__name__}: {e}")


# ── Tier 2: trafilatura ────────────────────────────────────────────────────────

def _fetch_via_trafilatura(url: str) -> FetchResult:
    """
    Fetch via httpx + trafilatura extraction.

    Fetches raw HTML with httpx, then uses trafilatura to strip navigation,
    ads, headers, footers — keeping only the main article content.

    Does NOT handle JavaScript-rendered content. If the page requires JS
    to render its main content, trafilatura will return empty or near-empty.
    In that case, we return success=False and the URL is skipped.
    """
    try:
        # Step 1: fetch raw HTML
        response = httpx.get(
            url,
            timeout=settings.fetch_timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ResearchAgent/1.0; "
                    "+https://github.com/research-agent)"
                )
            },
            follow_redirects=True,
        )

        if response.status_code != 200:
            return _failed(url, f"HTTP {response.status_code}")

        html = response.text

    except httpx.TimeoutException:
        return _failed(url, f"Fetch timeout after {settings.fetch_timeout_seconds}s")
    except Exception as e:
        return _failed(url, f"Fetch error: {type(e).__name__}: {e}")

    # Step 2: extract main content with trafilatura
    try:
        content = trafilatura.extract(
            html,
            include_links=False,       # links add noise for LLM summarization
            include_tables=True,       # tables often contain the key data
            include_images=False,      # image alt text is rarely useful
            output_format="txt",       # plain text, not XML
            with_metadata=False,       # we get metadata separately
        )

        if not content or len(content) < 200:
            return _failed(url, "trafilatura returned empty/insufficient content (likely JS-rendered)")

        # Extract title from the HTML directly (trafilatura may not include it)
        title = trafilatura.extract_metadata(html)
        title_str = title.title if title and title.title else ""

        return FetchResult(
            url=url,
            content=content,
            title=title_str,
            success=True,
            source="trafilatura",
            error=None,
        )

    except Exception as e:
        return _failed(url, f"trafilatura error: {type(e).__name__}: {e}")


# ── Private helpers ────────────────────────────────────────────────────────────

def _failed(url: str, error: str) -> FetchResult:
    """Return a failed FetchResult. Never raises."""
    return FetchResult(
        url=url,
        content="",
        title="",
        success=False,
        source="failed",
        error=error,
    )


def _extract_title_from_markdown(text: str) -> str:
    """
    Pull the title from the first Markdown heading in Jina Reader output.
    Jina typically formats the page title as '# Title' on the first line.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""
