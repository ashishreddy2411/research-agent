"""
observability/logging.py — Structured logging setup for the research agent.

Replaces all print() calls with proper Python logging.

Every module gets a named logger via get_logger(__name__).
Format includes timestamp, logger name, and level — searchable and filterable.

USAGE:
    from observability.logging import get_logger
    logger = get_logger(__name__)

    logger.info("Planning complete — %d subqueries", len(subqueries))
    logger.warning("Jina rate limited, falling back to trafilatura")
    logger.error("Synthesis failed: %s", error)
"""

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    """
    One-time configuration of the root research-agent logger.

    All loggers created via get_logger() inherit this config.
    Output goes to stderr so it doesn't mix with Streamlit or eval stdout.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("research_agent")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger under the research_agent namespace.

    Example:
        get_logger("tools.search") → logger named "research_agent.tools.search"
    """
    _configure_root()
    return logging.getLogger(f"research_agent.{name}")
