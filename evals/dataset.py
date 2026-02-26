"""
evals/dataset.py — Ground-truth eval questions with expected keywords.

WHAT THIS IS:
  A small set of research questions where we know what a good answer
  should contain. Each question has:

  - question:          The research query to run
  - expected_keywords: Facts/terms that MUST appear in a good report
  - category:          Topic area (for grouping results)

  This is the simplest form of eval: keyword presence checking.
  It gives you "recall" — did the agent find and report the key facts?
  It does NOT measure precision (did it hallucinate anything extra).

WHY KEYWORD-BASED AND NOT LLM-JUDGE:
  LLM-as-judge eval is more flexible but requires another API call per
  eval question. For a CI check or quick regression test, keyword matching
  is free, fast, and deterministic. LLM judge is a Phase 6 enhancement.

COVERAGE SCORE:
  For each question: score = keywords_found / total_keywords
  A score of 1.0 = all expected facts present in the report.
  A score below 0.5 = significant gaps — the agent missed key information.
"""

from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    """
    One evaluation question with ground-truth keywords.

    expected_keywords: case-insensitive substrings that should appear
    in a good report on this topic.
    """
    question: str
    expected_keywords: list[str]
    category: str
    description: str = ""


# ── Eval dataset ──────────────────────────────────────────────────────────────

EVAL_DATASET: list[EvalQuestion] = [
    EvalQuestion(
        question="What are the major breakthroughs in solid-state battery technology in 2023 and 2024?",
        expected_keywords=[
            "solid-state",
            "electrolyte",
            "energy density",
            "lithium",
            "QuantumScape",
        ],
        category="technology",
        description="Battery technology — tests coverage of technical facts and key companies",
    ),
    EvalQuestion(
        question="What caused the 2008 global financial crisis?",
        expected_keywords=[
            "subprime",
            "mortgage",
            "Lehman Brothers",
            "housing",
            "credit",
        ],
        category="economics",
        description="Well-documented historical event — high-recall baseline test",
    ),
    EvalQuestion(
        question="How does CRISPR-Cas9 gene editing work and what are its main limitations?",
        expected_keywords=[
            "CRISPR",
            "Cas9",
            "DNA",
            "guide RNA",
            "off-target",
        ],
        category="science",
        description="Science topic — tests both mechanism (how it works) and limitations",
    ),
    EvalQuestion(
        question="What are the key differences between supervised and unsupervised machine learning?",
        expected_keywords=[
            "supervised",
            "unsupervised",
            "label",
            "clustering",
            "classification",
        ],
        category="AI",
        description="Fundamental ML concept — broad coverage expected across multiple sources",
    ),
    EvalQuestion(
        question="What are the main effects of climate change on global food security?",
        expected_keywords=[
            "crop",
            "drought",
            "food security",
            "temperature",
            "yield",
        ],
        category="environment",
        description="Policy/science topic — tests synthesis across diverse sources",
    ),
]
