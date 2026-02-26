"""
agent/researcher.py — Search + fetch + summarize for one subquery.

THE CORE CONCEPT:
  The Researcher is the workhorse. Given one search query, it:
    1. Searches Tavily → gets URLs + page content
    2. Deduplicates against already-visited URLs
    3. Summarizes each page with the cheap model (200-word bullets)
    4. Returns PageSummary objects to be stored in ResearchState

  The research loop calls Researcher once per subquery, per round.
  In Phase 5 this becomes parallel (asyncio.gather across subqueries).
  In Phase 2, it runs serially — understand the logic before parallelizing.

THE SUMMARIZATION PROMPT:
  Each page is summarized with the cheap model in isolation.
  The prompt anchors the summary to the research query: "Extract facts
  relevant to: {query}". Without this anchor, the cheap model summarizes
  whatever the page is about — not what we're researching.

  Output is 200-word bullet points. Why bullets not prose?
  - Bullets force the model to extract discrete facts, not paraphrase
  - Easier to deduplicate at synthesis (same fact from 3 sources)
  - Faster to scan when debugging research quality

WHY CHEAP MODEL FOR SUMMARIZATION:
  This function is called once per URL — potentially 100 times per run.
  The task is simple: extract relevant facts from this page.
  The cheap model (gpt-4o-mini) does this as well as the smart model
  at a fraction of the cost. This is the highest-leverage cost optimization.

USAGE:
  from agent.researcher import Researcher
  from llm.client import LLMClient

  researcher = Researcher(client=LLMClient())
  summaries = researcher.research("solid state battery energy density 2025",
                                   visited_urls=set(), round_number=1)
  for s in summaries:
      print(s.url, s.word_count)
      print(s.summary[:200])
"""

from tools.search import search, SearchResult
from tools.fetch import fetch_page
from tools.extract import truncate_to_tokens
from agent.state import PageSummary, ResearchState
from llm.client import LLMClient
from config import settings
from prompts.researcher import SUMMARIZE_PROMPT


# ── Prompt lives in prompts/researcher.py ────────────────────────────────────


# ── Researcher ────────────────────────────────────────────────────────────────

class Researcher:
    """
    Executes one research subquery: search → deduplicate → summarize.

    Returns a list of PageSummary objects ready to be added to ResearchState.
    Never raises — returns whatever summaries it produced, even if partial.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def research(
        self,
        subquery: str,
        visited_urls: set[str],
        round_number: int,
    ) -> list[PageSummary]:
        """
        Search for subquery, fetch and summarize new URLs.

        Args:
            subquery:     The search query to execute.
            visited_urls: URLs already processed — skip these (deduplication).
            round_number: Which research round this is (1, 2, 3...).

        Returns:
            List of PageSummary objects. May be empty if all URLs were
            already visited or all fetches failed.
        """
        # Step 1: Search
        results = search(subquery, max_results=settings.max_search_results)
        if not results:
            return []

        # Step 2: Filter out already-visited URLs + deduplicate within this batch
        seen_in_batch: set[str] = set()
        new_results = []
        for r in results:
            if r.url not in visited_urls and r.url not in seen_in_batch:
                seen_in_batch.add(r.url)
                new_results.append(r)
        if not new_results:
            return []

        # Step 3: Summarize each result
        summaries = []
        for result in new_results:
            summary = self._summarize_result(result, subquery, round_number)
            if summary:
                summaries.append(summary)

        return summaries

    def research_into_state(
        self,
        subquery: str,
        state: ResearchState,
        round_number: int,
    ) -> int:
        """
        Research subquery and add results directly into state.

        Returns the number of new summaries added.
        """
        summaries = self.research(subquery, state.visited_urls, round_number)
        for summary in summaries:
            state.add_summary(summary)
        return len(summaries)

    # ── Private ───────────────────────────────────────────────────────────────

    def _summarize_result(
        self,
        result: SearchResult,
        subquery: str,
        round_number: int,
    ) -> PageSummary | None:
        """
        Summarize one search result into a PageSummary.

        Uses Tavily's extracted content first (already available from search).
        Only fetches the URL separately if Tavily's content is too short.
        Returns None if we can't get usable content.
        """
        # Get the best available content
        content = result.best_content

        # If Tavily content is thin, try fetching the page directly
        if len(content.split()) < 100:
            fetch_result = fetch_page(result.url)
            if fetch_result.success:
                content = fetch_result.content
                source = fetch_result.source
            else:
                # Even thin Tavily content is better than nothing
                source = "tavily"
        else:
            source = "tavily"

        if not content or len(content.split()) < 30:
            return None

        # Truncate to budget before sending to cheap model
        content = truncate_to_tokens(content, max_words=2000)

        prompt = SUMMARIZE_PROMPT.format(
            query=subquery,
            title=result.title or result.url,
            url=result.url,
            content=content,
            max_words=settings.max_summary_tokens,
        )

        try:
            summary_text = self._client.generate_cheap(prompt)
        except Exception:
            return None

        if not summary_text or len(summary_text.strip()) < 20:
            return None

        return PageSummary(
            url=result.url,
            title=result.title,
            summary=summary_text.strip(),
            subquery=subquery,
            round_number=round_number,
            word_count=len(summary_text.split()),
            source=source,
        )
