"""
llm/utils.py — Shared utilities for LLM response handling.

This file exists to eliminate code duplication. The _extract_text() function
was previously copy-pasted in planner.py, reflector.py, and synthesizer.py.
Now it lives here, once.
"""

from observability.logging import get_logger

logger = get_logger(__name__)


def extract_response_text(response) -> str:
    """
    Pull plain text from an OpenAI Responses API response object.

    Tries response.output_text first (the common path), then walks
    response.output looking for message blocks with text content.

    Returns empty string if extraction fails — caller decides what to do.
    Never raises.
    """
    try:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if hasattr(block, "text"):
                        return block.text.strip()
    except Exception as e:
        logger.warning("Failed to extract text from LLM response: %s", e)
    return ""
