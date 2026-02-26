"""
prompts/reflector.py — Prompt for gap detection / coverage evaluation.
"""

REFLECT_PROMPT = """\
You are evaluating research coverage for the following question:

RESEARCH QUESTION: {question}

SUMMARIES COLLECTED ({n_summaries} sources across {n_rounds} round(s)):
{summaries_text}

Evaluate whether the collected summaries adequately answer the research question.

Ask yourself:
- Are there important aspects of the question that are NOT covered?
- Is a specific subtopic missing entirely?
- Would a targeted follow-up search meaningfully improve the answer?

If YES — there is a meaningful gap:
  Return JSON with a specific follow-up query that addresses the gap.

If NO — coverage is sufficient:
  Return JSON with follow_up_query as null.

Respond with ONLY valid JSON. No other text.
Format: {{"knowledge_gap": "<what is missing or null>", "follow_up_query": "<specific search query or null>"}}"""
