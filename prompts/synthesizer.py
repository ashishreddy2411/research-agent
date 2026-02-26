"""
prompts/synthesizer.py — Prompts for two-shot report synthesis.

Shot 1: OUTLINE_PROMPT  → generate section headers from summaries
Shot 2: REPORT_PROMPT   → write the full report using those headers
"""

OUTLINE_PROMPT = """\
You are planning the structure of a research report.

RESEARCH QUESTION: {question}

SOURCES COLLECTED ({n_summaries} pages):
{summaries_text}

Generate a report outline: a list of section headings that together fully answer the research question.

Rules:
- 4 to 7 sections maximum
- Each heading is a clear, specific statement (not vague like "Introduction" or "Overview")
- Together they must cover the research question completely
- Order them logically (context → findings → implications)

Respond with ONLY valid JSON. No other text.
Format: {{"sections": ["Heading 1", "Heading 2", "Heading 3"]}}"""


REPORT_PROMPT = """\
You are writing a research report based on collected sources.

RESEARCH QUESTION: {question}

REPORT SECTIONS TO WRITE:
{sections_text}

SOURCES (numbered for citation):
{sources_text}

Write the full report in Markdown. Rules:
- Use the section headings exactly as given (## level)
- EVERY factual claim must have an inline citation [N] — no exceptions
- Multiple citations are fine: "batteries improved 40% [1][3]"
- Be specific: include numbers, dates, names where the sources provide them
- Do NOT invent facts not present in the sources
- Do NOT include a References section — that is appended automatically
- Aim for 400-800 words total

Write the report now:"""
