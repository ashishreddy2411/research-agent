"""
prompts/researcher.py — Prompt for per-page summarization.
"""

SUMMARIZE_PROMPT = """\
Research query: {query}

Page title: {title}
Page URL: {url}

Page content:
{content}

Extract ALL facts from this page that are relevant to the research query above.
Write bullet points only. Each bullet = one discrete fact.
Be specific: include numbers, dates, names, percentages where present.
Ignore content unrelated to the query.
Maximum {max_words} words total."""
