"""
agent/planner.py — Query decomposition: one question → N targeted search queries.

THE CORE CONCEPT:
  "What are the latest breakthroughs in solid-state batteries?" is one question
  but needs multiple search angles to answer comprehensively:
    - "solid state battery breakthroughs 2025"
    - "solid state battery commercial production timeline"
    - "solid state battery companies QuantumScape Toyota Samsung"
    - "solid state vs lithium ion battery energy density comparison"

  Each angle finds different sources. A single search query finds one perspective.
  Four targeted queries find four distinct bodies of evidence.

WHY NOT SEARCH THE ORIGINAL QUESTION DIRECTLY:
  Search engines are keyword matchers. A conversational question as a search
  query returns articles about the question's phrasing, not focused content.
  Decomposing into specific, concrete queries returns documents that directly
  answer parts of the question — higher signal, less noise.

THE PROMPT:
  Returns JSON: {"queries": [...]}. If the LLM wraps it in markdown fences,
  we strip those before parsing. If parsing fails, we fall back to
  [original_query] — the research loop always gets something to work with.

USAGE:
  from agent.planner import Planner
  from llm.client import LLMClient

  planner = Planner(client=LLMClient())
  subqueries = planner.decompose("What caused the 2008 financial crisis?")
  # ["2008 financial crisis causes", "subprime mortgage crisis 2007 2008",
  #  "Lehman Brothers collapse causes", "Federal Reserve response 2008 crisis"]
"""

import json
import re

from llm.client import LLMClient
from agent.state import ResearchState
from prompts.planner import DECOMPOSE_PROMPT


# ── Prompt lives in prompts/planner.py ───────────────────────────────────────


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    """
    Decomposes a research question into targeted search queries.

    One LLM call (smart model). Returns list of query strings.
    Falls back to [original_query] if the LLM call fails or returns
    unparseable output — the research loop always gets something to work with.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def decompose(self, query: str, n: int | None = None) -> list[str]:
        """
        Break one research question into N targeted search queries.

        Args:
            query: The original user question.
            n:     How many subqueries to generate. Defaults to 4.
                   More = broader coverage but more cost and time.
                   Less = faster but may miss important angles.

        Returns:
            List of search query strings.
            Never empty — falls back to [query] on any failure.
        """
        n = n or 4
        prompt = DECOMPOSE_PROMPT.format(question=query, n=n)

        try:
            response = self._client.generate(
                input=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(response)
            queries = _parse_queries(text)

            if queries:
                return queries

        except Exception:
            pass  # fall through to fallback

        return [query]

    def plan(self, state: ResearchState) -> None:
        """
        Decompose state.query and write subqueries into state.
        Called once at the start of the research loop.
        """
        state.subqueries = self.decompose(state.query)


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """Pull text from a Responses API response object."""
    try:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if hasattr(block, "text"):
                        return block.text.strip()
    except Exception:
        pass
    return ""


def _parse_queries(text: str) -> list[str]:
    """
    Parse {"queries": [...]} from LLM output.

    LLMs sometimes wrap JSON in markdown code fences (```json ... ```).
    Strip those before parsing. Return empty list if parsing fails.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    try:
        data = json.loads(text)
        queries = data.get("queries", [])
        return [str(q).strip() for q in queries if q and str(q).strip()]
    except (json.JSONDecodeError, AttributeError):
        return []
