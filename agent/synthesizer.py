"""
agent/synthesizer.py — Turns page summaries into a final Markdown report.

THE CORE CONCEPT:
  After research rounds complete, we have 10-50 PageSummary objects — bullet
  points from individual pages, each anchored to a subquery. Raw material.
  The Synthesizer's job is to turn that raw material into a coherent report
  a human can actually read.

TWO-SHOT SYNTHESIS:
  Shot 1 — Outline: ask the smart model "given these summaries, what sections
  should the report have?" Returns 4-7 section headings.

  Shot 2 — Report: ask the smart model to write the full report using those
  headings, citing sources inline as [1], [2], etc.

  Why two shots instead of one?
  - One shot often produces rambling prose without clear structure
  - Two shots forces the model to plan before writing — better organization
  - The outline is also stored in state for debugging and UI display

CITATIONS:
  Each PageSummary has a URL. We number them 1..N in the order they appear
  in state.page_summaries. The report prompt instructs the model to use [N]
  inline citations. After the report body, we append a ## References section
  with the actual URLs.

SUMMARY SELECTION:
  We cap at top_k_summaries (default 20) to stay within the context window.
  Selection is simple: first N summaries by order collected (round 1 first).
  Phase 5 replaces this with embedding-based relevance ranking.

USAGE:
  from agent.synthesizer import Synthesizer
  from llm.client import LLMClient

  synthesizer = Synthesizer(client=LLMClient())
  synthesizer.synthesize(state)
  print(state.final_report)
  print(state.status)   # ResearchStatus.SUCCESS
"""

import json
import re

from agent.state import ResearchState, ResearchStatus
from agent.guardrails import check_citation_bounds
from llm.client import LLMClient
from config import settings
from prompts.synthesizer import OUTLINE_PROMPT, REPORT_PROMPT


class Synthesizer:
    """
    Converts collected page summaries into a final Markdown report.

    Two smart-model calls: outline generation then report writing.
    Writes final_report, outline, sources into state and sets status=SUCCESS.
    Falls back gracefully — a partial report is better than nothing.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def synthesize(self, state: ResearchState) -> None:
        """
        Generate the final report from state.page_summaries.

        Modifies state in place:
          - state.outline    → section headings
          - state.final_report → full Markdown report with citations
          - state.sources    → ordered list of source URLs
          - state.status     → SUCCESS (or PARTIAL if fallback was used)

        Never raises.
        """
        if not state.page_summaries:
            state.record_partial(
                report="No sources were collected. Cannot generate a report.",
                sources=[],
                reason="No page summaries available for synthesis",
            )
            return

        summaries = state.page_summaries[: settings.top_k_summaries]
        sources = [s.url for s in summaries]

        # Shot 1: outline
        sections = self._generate_outline(state.query, summaries)
        state.outline = sections

        # Shot 2: report body
        report_body = self._generate_report(state.query, sections, summaries)

        # Append references — title + URL so every claim is traceable
        references = _build_references(summaries)
        final_report = report_body.strip() + "\n\n" + references

        # Citation bounds check — warn if model hallucinated out-of-range refs
        bad_citations = check_citation_bounds(final_report, len(sources))
        if bad_citations:
            state.errors.append(
                f"Out-of-bounds citations in report: {bad_citations} "
                f"(only {len(sources)} sources available)"
            )

        state.record_success(report=final_report, sources=sources)

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_outline(
        self,
        query: str,
        summaries: list,
    ) -> list[str]:
        """
        Ask the smart model for section headings.

        Returns a list of heading strings.
        Falls back to a single generic section on failure.
        """
        summaries_text = _format_summaries_for_prompt(summaries)

        prompt = OUTLINE_PROMPT.format(
            question=query,
            n_summaries=len(summaries),
            summaries_text=summaries_text,
        )

        try:
            response = self._client.generate(
                input=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(response)
            sections = _parse_outline(text)
            if sections:
                return sections
        except Exception:
            pass

        return [f"Research Findings: {query}"]

    def _generate_report(
        self,
        query: str,
        sections: list[str],
        summaries: list,
    ) -> str:
        """
        Write the full report using the outline and sources.

        Returns Markdown string. Falls back to a plain summary list on failure.
        """
        sections_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sections))
        sources_text = _format_sources_for_prompt(summaries)

        prompt = REPORT_PROMPT.format(
            question=query,
            sections_text=sections_text,
            sources_text=sources_text,
        )

        try:
            response = self._client.generate(
                input=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(response)
            if text and len(text.strip()) > 100:
                return text
        except Exception:
            pass

        # Fallback: bullet list of summaries
        return _fallback_report(query, summaries)


# ── Private helpers ───────────────────────────────────────────────────────────

def _format_summaries_for_prompt(summaries: list) -> str:
    """
    Format summaries for the outline prompt.
    Shows title + first 300 chars of summary per source.
    """
    lines = []
    for i, s in enumerate(summaries, 1):
        title = s.title or s.url
        lines.append(f"[{i}] {title}")
        lines.append(s.summary[:300])
        lines.append("")
    return "\n".join(lines)


def _format_sources_for_prompt(summaries: list) -> str:
    """
    Format sources for the report prompt — numbered with full summaries.
    The model uses these numbers for inline citations.
    """
    lines = []
    for i, s in enumerate(summaries, 1):
        title = s.title or s.url
        lines.append(f"[{i}] {title} ({s.url})")
        lines.append(s.summary[:500])
        lines.append("")
    return "\n".join(lines)


def _build_references(summaries: list) -> str:
    """
    Build a ## References section with numbered entries.

    Each entry: [N] Title — URL
    Title makes it human-readable; URL makes it verifiable.
    Matches the [N] inline citations the model was told to use.
    """
    lines = ["## References", ""]
    for i, s in enumerate(summaries, 1):
        title = s.title or "Untitled"
        lines.append(f"[{i}] {title}  ")
        lines.append(f"    {s.url}")
        lines.append("")
    return "\n".join(lines)


def _fallback_report(query: str, summaries: list) -> str:
    """
    Last-resort report: bullet list of raw summaries.
    Used when the LLM call fails entirely.
    """
    lines = [f"# Research: {query}", ""]
    lines.append("*Note: Full synthesis unavailable. Raw findings below.*")
    lines.append("")
    for i, s in enumerate(summaries, 1):
        title = s.title or s.url
        lines.append(f"### [{i}] {title}")
        lines.append(s.summary)
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


def _parse_outline(text: str) -> list[str]:
    """
    Parse {"sections": [...]} from LLM output.
    Returns empty list on failure — caller falls back to generic heading.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    try:
        data = json.loads(text)
        sections = data.get("sections", [])
        return [str(s).strip() for s in sections if s and str(s).strip()]
    except (json.JSONDecodeError, AttributeError):
        return []
