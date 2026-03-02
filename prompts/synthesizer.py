"""
prompts/synthesizer.py — Prompts for two-shot report synthesis.

Shot 1: OUTLINE_PROMPT  → generate section headers from summaries
Shot 2: REPORT_PROMPT   → write the full report using those headers
"""

OUTLINE_PROMPT = """\
You are planning the structure of a comprehensive research report.

RESEARCH QUESTION: {question}

SOURCES COLLECTED ({n_summaries} pages):
{summaries_text}

Generate a report outline: a list of section headings that together fully answer the research question.

Rules:
- 5 to 8 sections maximum
- Each heading is a clear, specific statement (not vague like "Introduction" or "Overview")
- Each section should address a distinct, substantive aspect of the question
- Together they must cover the research question completely — breadth AND depth
- Order them logically (context → findings → analysis → implications)
- Include sections for key data, comparisons, and practical takeaways where relevant

Respond with ONLY valid JSON. No other text.
Format: {{"sections": ["Heading 1", "Heading 2", "Heading 3"]}}"""


REPORT_PROMPT = """\
You are writing a comprehensive, detailed research report based on collected sources.

RESEARCH QUESTION: {question}

REPORT SECTIONS TO WRITE:
{sections_text}

SOURCES (numbered for citation):
{sources_text}

Write the full report in Markdown. Rules:
- Use the section headings exactly as given (## level)
- EVERY factual claim must have an inline citation [N] — no exceptions
- Multiple citations are fine: "batteries improved 40% [1][3]"
- Be specific: include numbers, dates, names, percentages where the sources provide them
- Do NOT invent facts not present in the sources
- Do NOT include a References section — that is appended automatically

CRITICAL — write a DETAILED and COMPREHENSIVE report:
- Aim for 1000-2000 words total — this is a deep research report, not a summary
- For each section, provide thorough analysis with context, not just one-line bullet points
- Explain the significance of findings — why they matter, not just what they are
- Include comparisons, trends, and relationships between different data points
- Cover methodology, limitations, and caveats where sources mention them
- Synthesize information across multiple sources — don't just list facts from each source in isolation
- Connect the dots: how do different findings relate to or inform each other?

Write the report now:"""
