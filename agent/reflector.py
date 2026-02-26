"""
agent/reflector.py — Gap detection: should the agent search again?

THE CORE CONCEPT:
  After each research round, the agent asks itself: "Do I have enough to
  write a good report, or is something important missing?"

  This is what makes a research agent different from a single-pass search:
  the agent reflects on its own knowledge state and decides whether to
  continue. If it finds a gap, it generates a targeted follow-up query.
  If coverage is sufficient, it stops and moves to synthesis.

  Without reflection, the agent does N fixed rounds regardless of quality.
  With reflection, it stops early when coverage is sufficient (cheaper)
  and searches deeper when gaps remain (higher quality).

WHAT "GAP" MEANS:
  A gap is a specific aspect of the original question that the collected
  summaries don't adequately address. Not just "more information" — a
  concrete missing angle.

  Good gaps (actionable):
    "No coverage of the cost comparison between solid-state and lithium-ion"
    "Missing commercial availability timeline from major manufacturers"

  Bad gaps (too vague to search for):
    "More detail needed"
    "Could be more comprehensive"

  The prompt is designed to distinguish these. If the reflector can't
  name a specific gap with a specific follow-up query, it returns None.

THE STOPPING CONDITIONS (two of them):
  1. Reflector returns None → "coverage is sufficient, stop"
  2. max_research_rounds reached → "hard stop, synthesize anyway"

  The hard stop exists because the reflector LLM sometimes keeps finding
  gaps even when coverage is genuinely sufficient. A round cap prevents
  runaway research loops.

USAGE:
  from agent.reflector import Reflector
  from llm.client import LLMClient

  reflector = Reflector(client=LLMClient())
  gap = reflector.reflect(query="...", summaries=[...])
  if gap:
      print(f"Follow-up query: {gap}")
  else:
      print("Coverage sufficient — proceed to synthesis")
"""

import json
import re
from dataclasses import dataclass

from agent.state import PageSummary, ResearchState
from llm.client import LLMClient
from prompts.reflector import REFLECT_PROMPT


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ReflectionResult:
    """
    The outcome of one reflection pass.

    has_gap=True  → follow_up_query contains the next search query
    has_gap=False → coverage is sufficient, proceed to synthesis
    """
    has_gap: bool
    follow_up_query: str | None
    gap_description: str


# ── Prompt lives in prompts/reflector.py ─────────────────────────────────────


# ── Reflector ─────────────────────────────────────────────────────────────────

class Reflector:
    """
    Evaluates research coverage and decides whether to continue searching.

    One LLM call (smart model) per round. Returns a ReflectionResult
    indicating whether a follow-up search is needed and what to search for.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def reflect(
        self,
        query: str,
        summaries: list[PageSummary],
        rounds_completed: int,
    ) -> ReflectionResult:
        """
        Evaluate research coverage and identify any gaps.

        Args:
            query:            The original research question.
            summaries:        All PageSummary objects collected so far.
            rounds_completed: How many rounds have been completed.

        Returns:
            ReflectionResult — has_gap=True means search again.
            Falls back to has_gap=False on any LLM failure (stop, don't loop).
        """
        if not summaries:
            # Nothing collected yet — can't reflect, assume gap
            return ReflectionResult(
                has_gap=True,
                follow_up_query=query,
                gap_description="No summaries collected yet",
            )

        summaries_text = _format_summaries(summaries)

        prompt = REFLECT_PROMPT.format(
            question=query,
            n_summaries=len(summaries),
            n_rounds=rounds_completed,
            summaries_text=summaries_text,
        )

        try:
            response = self._client.generate(
                input=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(response)
            return _parse_reflection(text)

        except Exception:
            # On failure: stop the loop, don't keep searching
            return ReflectionResult(
                has_gap=False,
                follow_up_query=None,
                gap_description="Reflection failed — stopping to synthesize",
            )

    def reflect_on_state(self, state: ResearchState) -> ReflectionResult:
        """
        Reflect on current state and optionally add a gap to state.

        If a gap is found, appends it to state.knowledge_gaps.
        Returns the ReflectionResult for the loop to act on.
        """
        result = self.reflect(
            query=state.query,
            summaries=state.page_summaries,
            rounds_completed=state.rounds_completed,
        )
        if result.has_gap and result.follow_up_query:
            state.add_gap(result.follow_up_query)
        return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _format_summaries(summaries: list[PageSummary]) -> str:
    """
    Format summaries for the reflection prompt.

    Each summary shown as: [Round N] Title (URL)\n{summary text}
    Capped at 30 summaries to prevent prompt overflow.
    """
    shown = summaries[:30]
    lines = []
    for i, s in enumerate(shown, 1):
        title = s.title or s.url
        lines.append(f"[{i}] Round {s.round_number} — {title}")
        lines.append(s.summary[:500])  # cap each summary at 500 chars
        lines.append("")
    return "\n".join(lines)


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


def _parse_reflection(text: str) -> ReflectionResult:
    """
    Parse {"knowledge_gap": "...", "follow_up_query": "..." or null}.

    Returns has_gap=False on any parse failure — safe default is to stop.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    try:
        data = json.loads(text)
        follow_up = data.get("follow_up_query")
        gap_desc = data.get("knowledge_gap") or ""

        if follow_up and str(follow_up).strip().lower() not in ("null", "none", ""):
            return ReflectionResult(
                has_gap=True,
                follow_up_query=str(follow_up).strip(),
                gap_description=str(gap_desc),
            )
        else:
            return ReflectionResult(
                has_gap=False,
                follow_up_query=None,
                gap_description=str(gap_desc) or "Coverage sufficient",
            )

    except (json.JSONDecodeError, AttributeError):
        return ReflectionResult(
            has_gap=False,
            follow_up_query=None,
            gap_description="Could not parse reflection response",
        )
