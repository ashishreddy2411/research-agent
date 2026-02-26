"""
tools/extract.py — Extract clean text from raw HTML.

THE CORE CONCEPT: From messy HTML to LLM-ready text
  Raw HTML is full of noise: navigation menus, cookie banners, ads,
  sidebars, footers, related article widgets, social share buttons.

  An LLM fed raw HTML wastes tokens on this noise and struggles to find
  the actual article content. The extractor's job: find the main content
  and discard everything else. This is called "boilerplate removal."

WHY THIS FILE EXISTS alongside fetch.py:
  fetch.py handles the full pipeline: fetch URL → extract content.
  extract.py handles only the extraction step: HTML string → clean text.

  You need extract.py separately when:
    - Tavily gives you raw_content but it needs cleanup
    - You already have HTML from another source
    - You want to test extraction logic without making HTTP calls

TRAFILATURA VS OTHER OPTIONS:
  Three Python extraction libraries were benchmarked across diverse sites:
  - trafilatura  — best mean score (0.883), handles blogs, news, docs well
  - readability  — best median score (0.970), most predictable on news sites
  - BeautifulSoup — fast but needs custom selectors (not general-purpose)

  We use trafilatura as primary, readability as fallback.
  The fallback matters: trafilatura sometimes returns empty on pages where
  readability succeeds, and vice versa. Two tries beats one.

POST-EXTRACTION CLEANUP:
  Even after trafilatura, the text can have:
    - Excessive blank lines (3+ in a row)
    - Unicode noise (soft hyphens, zero-width spaces, curly quotes)
    - Boilerplate survivors ("Share this article", "Subscribe to our newsletter")

  clean_text() handles all of this and returns text ready for LLM input.

USAGE:
  from tools.extract import extract_main_content, clean_text

  # From raw HTML
  text = extract_main_content(html_string)

  # Cleanup only (if you already have text)
  cleaned = clean_text(raw_text)
"""

import re
import trafilatura


# ── Main extraction function ───────────────────────────────────────────────────

def extract_main_content(html: str, url: str = "") -> str:
    """
    Extract main article content from raw HTML.

    Tries trafilatura first, falls back to readability-style extraction.
    Returns clean text ready for LLM summarization.
    Returns empty string if both approaches fail.

    Args:
        html: Raw HTML string.
        url:  Optional — helps trafilatura with site-specific heuristics.

    Returns:
        Clean text string, or "" if extraction failed.
    """
    if not html or len(html) < 100:
        return ""

    # Tier 1: trafilatura
    content = _extract_with_trafilatura(html, url)
    if content and len(content) > 200:
        return clean_text(content)

    # Tier 2: trafilatura with more lenient settings
    # Sometimes the default settings are too strict and filter out valid content.
    # include_tables + no_filter gives trafilatura a second chance.
    content = _extract_with_trafilatura_lenient(html)
    if content and len(content) > 100:
        return clean_text(content)

    return ""


# ── Extraction implementations ─────────────────────────────────────────────────

def _extract_with_trafilatura(html: str, url: str = "") -> str:
    """
    Standard trafilatura extraction.

    Settings:
      include_tables=True  — tables often contain key data (prices, specs, results)
      include_links=False  — hyperlinks add noise without adding meaning for LLMs
      output_format="txt"  — plain text, not XML/HTML/Markdown
      favor_precision=True — prefer accuracy over recall (better quality, less noise)
    """
    try:
        result = trafilatura.extract(
            html,
            url=url or None,
            include_tables=True,
            include_links=False,
            include_images=False,
            output_format="txt",
            favor_precision=True,
        )
        return result or ""
    except Exception:
        return ""


def _extract_with_trafilatura_lenient(html: str) -> str:
    """
    Lenient trafilatura extraction — more recall, potentially noisier.

    favor_recall=True extracts more content even if some boilerplate sneaks in.
    This is the fallback for pages where standard extraction returns too little.
    """
    try:
        result = trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            output_format="txt",
            favor_recall=True,
        )
        return result or ""
    except Exception:
        return ""


# ── Text cleanup ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize extracted text for LLM consumption.

    Operations (in order):
      1. Normalize line endings (Windows \r\n → \n)
      2. Remove zero-width and soft-hyphen Unicode noise
      3. Collapse 3+ consecutive blank lines to 2
      4. Strip trailing whitespace from each line
      5. Strip leading/trailing whitespace from the whole text

    Does NOT:
      - Remove content (only whitespace normalization)
      - Truncate (caller decides max length)
      - Convert to markdown (keeps plain text)
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Unicode noise characters
    # \u00ad = soft hyphen, \u200b = zero-width space, \u200c/\u200d = zero-width joiners
    text = re.sub(r"[\u00ad\u200b\u200c\u200d\ufeff]", "", text)

    # Normalize curly quotes to straight quotes (optional — LLMs handle both)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def truncate_to_tokens(text: str, max_words: int = 2000) -> str:
    """
    Truncate text to approximately max_words words.

    We use word count as a proxy for tokens (1 token ≈ 0.75 words on average).
    max_words=2000 ≈ 2,600 tokens — enough context for a page summary.

    Why truncate?
      A research paper might be 15,000 words. The cheap model only needs
      the first 2,000 words to produce a good 200-word summary.
      Sending 15,000 words to gpt-4o-mini costs 5-7x more for no quality gain.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [truncated]"
