"""
tools/test_phase1.py — Single-shot end-to-end test of the Phase 1 tool chain.

This is not a pytest test — it's a manual proof-of-life script.
Run it once after setup to confirm every tool in the chain works.

What it tests:
  1. search()     — Tavily returns results with content
  2. fetch_page() — Jina/trafilatura fetches a URL and returns text
  3. extract_main_content() — HTML → clean text works correctly
  4. LLMClient.generate_cheap() — cheap model responds
  5. LLMClient.embed() — embedding model returns a vector

Run:
  uv run python tools/test_phase1.py

Expected output:
  [1/5] Search...      ✓  Found 5 results
  [2/5] Fetch page...  ✓  1,243 words via jina
  [3/5] Extract HTML.. ✓  387 words extracted
  [4/5] Cheap LLM...   ✓  Summary: "Solid-state batteries use..."
  [5/5] Embeddings...  ✓  Vector length: 1536
  Phase 1 complete — all tools working.
"""

import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.search import search
from tools.fetch import fetch_page
from tools.extract import extract_main_content, truncate_to_tokens
from llm.client import LLMClient


QUERY = "solid state battery commercialization 2025"
TEST_URL = "https://en.wikipedia.org/wiki/Solid-state_battery"
TEST_HTML = """
<html><body>
<nav>Home | About | Contact</nav>
<article>
<h1>Solid-State Batteries</h1>
<p>Solid-state batteries replace the liquid electrolyte in conventional
lithium-ion batteries with a solid material. This improves safety by
eliminating flammable liquids and potentially increases energy density.</p>
<p>Companies including Toyota, Samsung, and QuantumScape are pursuing
commercial production. Toyota has announced plans for vehicle integration
by 2027-2028.</p>
</article>
<footer>Copyright 2025</footer>
</body></html>
"""


def run_phase1_test():
    passed = 0
    failed = 0

    # ── Test 1: Search ─────────────────────────────────────────────────────────
    print("[1/5] Search (Tavily)...", end="  ")
    try:
        results = search(QUERY, max_results=5)
        assert len(results) > 0, "No results returned"
        assert results[0].url, "First result has no URL"
        assert results[0].best_content, "First result has no content"
        print(f"✓  Found {len(results)} results | Top URL: {results[0].url[:60]}...")
        passed += 1
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed += 1

    # ── Test 2: Fetch page ─────────────────────────────────────────────────────
    print("[2/5] Fetch page (Jina/trafilatura)...", end="  ")
    try:
        result = fetch_page(TEST_URL)
        assert result.success, f"Fetch failed: {result.error}"
        assert result.word_count > 100, f"Too little content: {result.word_count} words"
        print(f"✓  {result.word_count} words via {result.source}")
        passed += 1
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed += 1

    # ── Test 3: Extract from HTML ──────────────────────────────────────────────
    print("[3/5] Extract from HTML (trafilatura)...", end="  ")
    try:
        text = extract_main_content(TEST_HTML)
        assert text, "Extraction returned empty string"
        assert "solid-state" in text.lower() or "solid" in text.lower(), \
            "Expected content not found in extraction"
        word_count = len(text.split())
        print(f"✓  {word_count} words extracted")
        passed += 1
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed += 1

    # ── Test 4: Cheap LLM (summarization) ─────────────────────────────────────
    print("[4/5] Cheap LLM (gpt-4o-mini)...", end="  ")
    try:
        client = LLMClient()
        content = results[0].best_content if passed >= 1 else TEST_HTML
        prompt = (
            f"Summarize in 2 sentences what this text says about '{QUERY}':\n\n"
            f"{truncate_to_tokens(content, max_words=500)}"
        )
        summary = client.generate_cheap(prompt)
        assert summary and len(summary) > 20, "Summary too short or empty"
        print(f"✓  Summary: \"{summary[:80]}...\"")
        passed += 1
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed += 1

    # ── Test 5: Embeddings ─────────────────────────────────────────────────────
    print("[5/5] Embeddings (text-embedding-3-small)...", end="  ")
    try:
        client = LLMClient()
        vector = client.embed(QUERY)
        assert isinstance(vector, list), "embed() should return a list"
        assert len(vector) == 1536, f"Expected 1536 dims, got {len(vector)}"
        assert all(isinstance(v, float) for v in vector[:10]), "Vector should contain floats"
        print(f"✓  Vector length: {len(vector)}")
        passed += 1
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if failed == 0:
        print(f"Phase 1 complete — all {passed}/5 tools working.")
    else:
        print(f"{passed}/5 passed, {failed}/5 FAILED.")
        print("Fix the failures above before moving to Phase 2.")
        sys.exit(1)


if __name__ == "__main__":
    run_phase1_test()
