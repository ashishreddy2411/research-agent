"""
evals/metrics.py — Evaluation metrics for research agent output.

METRICS:

  citation_accuracy(report, n_sources)
    Structural check: are all [N] references within bounds?
    Returns: {"total_citations": N, "out_of_bounds": [...], "accuracy": 0.0-1.0}

  citation_density(report)
    What fraction of sentences contain at least one citation?
    Higher = better grounded. Uncited sentences are potential hallucinations.
    Returns: {"cited_sentences": N, "total_sentences": N, "density": 0.0-1.0}

  keyword_coverage(report, expected_keywords)
    What fraction of expected keywords appear in the report?
    This is recall: did the agent find the key facts?
    Returns: {"found": [...], "missing": [...], "recall": 0.0-1.0}

  source_quality(page_summaries)
    Average Tavily relevance score across collected sources.
    Higher = more relevant sources were found.
    Returns: {"avg_score": float, "min_score": float, "n_sources": int}

  run_score(state, expected_keywords)
    Composite score combining all metrics for one run.
    Returns a dict with all sub-scores and an overall 0.0-1.0 score.

USAGE:
  from evals.metrics import run_score
  from agent.loop import run_research

  state = run_research("What are the effects of CRISPR on gene therapy?")
  score = run_score(state, expected_keywords=["CRISPR", "gene", "therapy", "off-target"])
  print(score)
"""

import re
from agent.state import ResearchState, PageSummary


# ── Citation accuracy ─────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_accuracy(report: str, n_sources: int) -> dict:
    """
    Check that all [N] citations in the report are within [1..n_sources].

    Returns:
        total_citations: how many [N] patterns found
        out_of_bounds:   list of citation numbers outside valid range
        accuracy:        fraction of citations that are valid (1.0 = all OK)
    """
    if not report or n_sources <= 0:
        return {"total_citations": 0, "out_of_bounds": [], "accuracy": 1.0}

    all_citations = [int(m.group(1)) for m in _CITATION_RE.finditer(report)]
    out_of_bounds = [n for n in all_citations if n < 1 or n > n_sources]

    total = len(all_citations)
    accuracy = 1.0 - (len(out_of_bounds) / total) if total > 0 else 1.0

    return {
        "total_citations": total,
        "out_of_bounds": sorted(set(out_of_bounds)),
        "accuracy": round(accuracy, 3),
    }


# ── Citation density ──────────────────────────────────────────────────────────

def citation_density(report: str) -> dict:
    """
    Fraction of sentences in the report that contain at least one citation.

    A citation-dense report is better grounded — fewer uncited claims.
    We only consider non-empty, non-heading, non-reference lines as sentences.

    Returns:
        cited_sentences:  count of sentences with ≥1 citation
        total_sentences:  count of all content sentences
        density:          cited / total (1.0 = every sentence cited)
    """
    if not report:
        return {"cited_sentences": 0, "total_sentences": 0, "density": 0.0}

    lines = report.splitlines()
    content_lines = []
    in_references = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## References"):
            in_references = True
            continue
        if in_references:
            continue
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue  # headings
        content_lines.append(stripped)

    if not content_lines:
        return {"cited_sentences": 0, "total_sentences": 0, "density": 0.0}

    cited = sum(1 for line in content_lines if _CITATION_RE.search(line))
    total = len(content_lines)
    density = cited / total if total > 0 else 0.0

    return {
        "cited_sentences": cited,
        "total_sentences": total,
        "density": round(density, 3),
    }


# ── Keyword coverage (recall) ─────────────────────────────────────────────────

def keyword_coverage(report: str, expected_keywords: list[str]) -> dict:
    """
    What fraction of expected keywords appear in the report?

    This is recall: did the agent surface the key facts?
    Matching is case-insensitive substring search.

    Returns:
        found:   keywords present in the report
        missing: keywords absent from the report
        recall:  found / total (1.0 = all expected facts covered)
    """
    if not report or not expected_keywords:
        return {"found": [], "missing": list(expected_keywords), "recall": 0.0}

    report_lower = report.lower()
    found = [kw for kw in expected_keywords if kw.lower() in report_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in report_lower]
    recall = len(found) / len(expected_keywords)

    return {
        "found": found,
        "missing": missing,
        "recall": round(recall, 3),
    }


# ── Source quality ────────────────────────────────────────────────────────────

def source_quality(page_summaries: list[PageSummary]) -> dict:
    """
    Average Tavily relevance score across the collected sources.

    PageSummary doesn't store the Tavily score directly (it's on SearchResult),
    so this metric uses proxy signals: word_count and source type.

    Returns:
        avg_word_count:  mean words per summary
        n_sources:       total sources collected
        tavily_fraction: fraction of sources from Tavily (vs fetch fallback)
    """
    if not page_summaries:
        return {"avg_word_count": 0, "n_sources": 0, "tavily_fraction": 0.0}

    word_counts = [s.word_count for s in page_summaries]
    tavily_count = sum(1 for s in page_summaries if s.source == "tavily")

    return {
        "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
        "n_sources": len(page_summaries),
        "tavily_fraction": round(tavily_count / len(page_summaries), 3),
    }


# ── Composite run score ───────────────────────────────────────────────────────

def run_score(state: ResearchState, expected_keywords: list[str]) -> dict:
    """
    Composite evaluation score for one completed research run.

    Combines citation accuracy, citation density, keyword recall, and
    source quality into sub-scores and an overall score.

    The overall score weights:
      - keyword recall:       50% (did we answer the question?)
      - citation accuracy:    30% (are references valid?)
      - citation density:     20% (are claims grounded?)

    Returns a dict with all sub-scores and "overall" (0.0–1.0).
    """
    report = state.final_report
    n_sources = len(state.sources)

    cit_acc = citation_accuracy(report, n_sources)
    cit_den = citation_density(report)
    kw_cov  = keyword_coverage(report, expected_keywords)
    src_q   = source_quality(state.page_summaries)

    overall = (
        0.50 * kw_cov["recall"]
        + 0.30 * cit_acc["accuracy"]
        + 0.20 * cit_den["density"]
    )

    return {
        "status": state.status.value,
        "n_sources": n_sources,
        "n_rounds": state.rounds_completed,
        "cost_usd": state.estimated_cost_usd,
        "citation_accuracy": cit_acc,
        "citation_density": cit_den,
        "keyword_coverage": kw_cov,
        "source_quality": src_q,
        "overall": round(overall, 3),
    }
