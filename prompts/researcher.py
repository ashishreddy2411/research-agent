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

Rules:
- Write bullet points only. Each bullet = one discrete fact.
- Be THOROUGH — capture every relevant detail, not just the headline finding.
- Include specific data: numbers, dates, names, percentages, comparisons.
- Capture context, methodology, and caveats — not just conclusions.
- Include relationships between facts (causes, effects, comparisons).
- Note any limitations, criticisms, or counterpoints mentioned.
- Ignore content unrelated to the query.

Maximum {max_words} words total."""
