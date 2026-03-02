"""
tests/unit/test_llm_utils.py — Unit tests for llm/utils.py

Covers: extract_response_text() — the shared helper that replaces
three copies of _extract_text() in planner, reflector, synthesizer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock
from llm.utils import extract_response_text


# ── Success: output_text ─────────────────────────────────────────────────────

class TestExtractFromOutputText:
    def test_returns_output_text_when_present(self):
        response = MagicMock()
        response.output_text = "The answer is 42."
        assert extract_response_text(response) == "The answer is 42."

    def test_strips_whitespace(self):
        response = MagicMock()
        response.output_text = "  padded text  \n"
        assert extract_response_text(response) == "padded text"


# ── Success: message blocks ──────────────────────────────────────────────────

class TestExtractFromMessageBlocks:
    def test_falls_back_to_message_block_text(self):
        response = MagicMock()
        response.output_text = ""  # empty — triggers fallback

        text_block = MagicMock()
        text_block.text = "From message block."
        message = MagicMock()
        message.type = "message"
        message.content = [text_block]
        response.output = [message]

        assert extract_response_text(response) == "From message block."

    def test_skips_non_message_items(self):
        response = MagicMock()
        response.output_text = ""

        tool_call = MagicMock()
        tool_call.type = "tool_call"

        text_block = MagicMock()
        text_block.text = "Found it."
        message = MagicMock()
        message.type = "message"
        message.content = [text_block]

        response.output = [tool_call, message]

        assert extract_response_text(response) == "Found it."


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestExtractEdgeCases:
    def test_returns_empty_on_none_output_text(self):
        response = MagicMock()
        response.output_text = None
        response.output = []
        assert extract_response_text(response) == ""

    def test_returns_empty_on_no_content(self):
        response = MagicMock()
        response.output_text = ""
        response.output = []
        assert extract_response_text(response) == ""

    def test_returns_empty_on_exception(self):
        response = MagicMock()
        response.output_text = ""
        # Make .output raise an exception
        type(response).output = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))
        assert extract_response_text(response) == ""

    def test_returns_empty_on_none_response(self):
        # Completely invalid response object
        assert extract_response_text(None) == ""
