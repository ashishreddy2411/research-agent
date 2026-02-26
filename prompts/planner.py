"""
prompts/planner.py — Prompt for query decomposition.
"""

DECOMPOSE_PROMPT = """\
You are a research strategist. Your job is to decompose a complex question
into specific, targeted search queries that together will provide comprehensive coverage.

QUESTION: {question}

Generate {n} search queries that:
1. Each targets a distinct angle or subtopic of the question
2. Are specific enough to return focused results (not too broad)
3. Use concrete terminology that search engines respond well to
4. Together cover the question comprehensively

Do NOT generate:
- Queries that are just rewordings of each other
- Overly broad queries ("tell me about X")
- Queries asking for opinions ("what do people think about X")

Respond with ONLY valid JSON. No other text.
Format: {{"queries": ["query1", "query2", "query3"]}}"""
