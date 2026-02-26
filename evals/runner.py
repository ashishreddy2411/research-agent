"""
evals/runner.py — Run the eval dataset and produce a score report.

WHAT THIS DOES:
  Runs each question in EVAL_DATASET through run_research(), scores the
  resulting state with run_score(), and prints a summary table.

  This is NOT a regression test (it makes real API calls and takes minutes).
  Use it manually to benchmark agent quality before major changes.

HOW TO USE:

  Run all 5 questions:
    uv run python -m evals.runner

  Run a single category:
    uv run python -m evals.runner --category science

  Run one question by index (0-based):
    uv run python -m evals.runner --index 2

  Save results to JSON:
    uv run python -m evals.runner --output results.json

OUTPUT:
  Per-question scores printed to stdout:
    [1/5] What caused the 2008 global financial crisis?
          Status:   SUCCESS   Rounds: 2   Sources: 14   Cost: $0.023
          Recall:   1.000  (5/5 keywords)
          Cit Acc:  1.000  (all 18 citations valid)
          Cit Den:  0.818  (18/22 sentences cited)
          Overall:  0.927

  Summary table at the end:
    ┌──────────────────────────────┬────────┬─────────┬──────────┬─────────┐
    │ Question                     │ Status │  Recall │  Cit Acc │ Overall │
    ├──────────────────────────────┼────────┼─────────┼──────────┼─────────┤
    │ 2008 financial crisis?       │ SUCCESS│  1.000  │  1.000   │  0.927  │
    │ CRISPR-Cas9 gene editing?    │ SUCCESS│  1.000  │  0.944   │  0.889  │
    │ Solid-state batteries?       │ SUCCESS│  0.800  │  1.000   │  0.850  │
    │ Supervised vs unsupervised?  │ SUCCESS│  1.000  │  1.000   │  0.940  │
    │ Climate & food security?     │ SUCCESS│  1.000  │  0.966   │  0.913  │
    └──────────────────────────────┴────────┴─────────┴──────────┴─────────┘
    Average overall score: 0.904
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import run_research
from evals.dataset import EVAL_DATASET, EvalQuestion
from evals.metrics import run_score


# ── ANSI colours (stripped when not a TTY) ────────────────────────────────────

def _colour(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

GREEN  = lambda t: _colour(t, "32")
YELLOW = lambda t: _colour(t, "33")
RED    = lambda t: _colour(t, "31")
BOLD   = lambda t: _colour(t, "1")
DIM    = lambda t: _colour(t, "2")


def _status_colour(status: str) -> str:
    if status == "success":
        return GREEN(status.upper())
    if status == "partial":
        return YELLOW(status.upper())
    return RED(status.upper())


def _score_colour(score: float) -> str:
    text = f"{score:.3f}"
    if score >= 0.8:
        return GREEN(text)
    if score >= 0.5:
        return YELLOW(text)
    return RED(text)


# ── Per-question runner ───────────────────────────────────────────────────────

def run_one(
    question: EvalQuestion,
    index: int,
    total: int,
    verbose: bool = False,
) -> dict:
    """
    Run a single eval question and return its score dict.

    The score dict is the output of run_score() enriched with:
      - question (str)
      - category (str)
      - elapsed_sec (float)
    """
    label = f"[{index}/{total}]"
    q_short = question.question[:70] + ("..." if len(question.question) > 70 else "")
    print(f"\n{BOLD(label)} {q_short}")
    print(DIM(f"       Category: {question.category}  |  Keywords: {question.expected_keywords}"))

    start = time.monotonic()

    progress_lines: list[str] = []

    def on_progress(msg: str) -> None:
        progress_lines.append(msg)
        if verbose:
            print(DIM(f"         ↳ {msg}"))

    state = run_research(question.question, on_progress=on_progress)
    elapsed = time.monotonic() - start

    scores = run_score(state, question.expected_keywords)
    scores["question"] = question.question
    scores["category"] = question.category
    scores["elapsed_sec"] = round(elapsed, 1)

    # Print summary
    status_str = _status_colour(scores["status"])
    print(
        f"       Status:  {status_str}   "
        f"Rounds: {scores['n_rounds']}   "
        f"Sources: {scores['n_sources']}   "
        f"Cost: ${scores['cost_usd']:.4f}   "
        f"Time: {elapsed:.1f}s"
    )

    kw = scores["keyword_coverage"]
    ca = scores["citation_accuracy"]
    cd = scores["citation_density"]
    overall = scores["overall"]

    found_count = len(kw["found"])
    total_kw   = found_count + len(kw["missing"])
    print(
        f"       Recall:  {_score_colour(kw['recall'])}  "
        f"({found_count}/{total_kw} keywords)"
    )
    if kw["missing"]:
        print(DIM(f"         Missing: {kw['missing']}"))
    print(
        f"       Cit Acc: {_score_colour(ca['accuracy'])}  "
        f"({ca['total_citations']} citations, {len(ca['out_of_bounds'])} out-of-bounds)"
    )
    print(
        f"       Cit Den: {_score_colour(cd['density'])}  "
        f"({cd['cited_sentences']}/{cd['total_sentences']} sentences cited)"
    )
    print(f"       {BOLD('Overall:')} {_score_colour(overall)}")

    if state.errors:
        for err in state.errors:
            print(RED(f"       ERROR: {err}"))

    return scores


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    """Print a compact summary table of all results."""
    if not results:
        return

    print("\n" + BOLD("─" * 72))
    print(BOLD("SUMMARY"))
    print(BOLD("─" * 72))

    header = f"  {'Question':<42}  {'Status':<8}  {'Recall':>6}  {'CitAcc':>6}  {'Overall':>7}"
    print(header)
    print("  " + "─" * 68)

    for r in results:
        q_short = r["question"][:40] + ".." if len(r["question"]) > 40 else r["question"]
        status = r["status"].upper()[:7]
        recall  = r["keyword_coverage"]["recall"]
        cit_acc = r["citation_accuracy"]["accuracy"]
        overall = r["overall"]

        print(
            f"  {q_short:<42}  {status:<8}  "
            f"{_score_colour(recall):>6}  "
            f"{_score_colour(cit_acc):>6}  "
            f"{_score_colour(overall):>7}"
        )

    avg_overall = sum(r["overall"] for r in results) / len(results)
    avg_recall  = sum(r["keyword_coverage"]["recall"] for r in results) / len(results)
    avg_cit_acc = sum(r["citation_accuracy"]["accuracy"] for r in results) / len(results)

    print("  " + "─" * 68)
    print(
        f"  {'AVERAGE':<42}  {'':8}  "
        f"{_score_colour(avg_recall):>6}  "
        f"{_score_colour(avg_cit_acc):>6}  "
        f"{_score_colour(avg_overall):>7}"
    )
    print(BOLD("─" * 72))
    print(f"\n  Average overall score: {BOLD(_score_colour(avg_overall))}")
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"  Total cost:            ${total_cost:.4f}")
    total_time = sum(r.get("elapsed_sec", 0) for r in results)
    print(f"  Total time:            {total_time:.1f}s\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run research agent evaluation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        help="Only run questions matching this category (e.g. science, AI, economics)",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Only run the question at this 0-based index in EVAL_DATASET",
    )
    parser.add_argument(
        "--output",
        help="Save full results to this JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print live progress messages from the agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Select questions
    questions = EVAL_DATASET

    if args.index is not None:
        if args.index < 0 or args.index >= len(questions):
            print(f"Error: --index {args.index} out of range (0..{len(questions)-1})")
            sys.exit(1)
        questions = [questions[args.index]]

    elif args.category:
        questions = [q for q in questions if q.category.lower() == args.category.lower()]
        if not questions:
            cats = sorted({q.category for q in EVAL_DATASET})
            print(f"Error: no questions for category '{args.category}'. Available: {cats}")
            sys.exit(1)

    total = len(questions)
    print(BOLD(f"\nRunning {total} eval question(s)..."))

    results: list[dict] = []
    for i, question in enumerate(questions, start=1):
        result = run_one(question, index=i, total=total, verbose=args.verbose)
        results.append(result)

    _print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
