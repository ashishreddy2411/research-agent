"""
tools/retry.py — Retry with exponential backoff for network calls.

Every external call in the research agent (Tavily search, Jina fetch,
trafilatura fetch) can fail transiently: timeouts, 5xx errors, rate limits.

This module provides a simple retry wrapper that:
  - Retries only on specified exception types (transient failures)
  - Uses exponential backoff (1s, 2s, 4s by default)
  - Logs each retry attempt
  - Gives up after max_retries and re-raises the last exception

USAGE:
    from tools.retry import retry_with_backoff
    import httpx

    @retry_with_backoff(
        max_retries=2,
        base_delay=1.0,
        retryable_exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
    )
    def fetch_something(url: str):
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        return response
"""

import time
import functools
from typing import Type

from observability.logging import get_logger

logger = get_logger("tools.retry")


def retry_with_backoff(
    max_retries: int = 2,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[Type[BaseException], ...] = (Exception,),
):
    """
    Decorator that retries a function on transient failures with exponential backoff.

    Args:
        max_retries:          Number of retry attempts (0 = no retries, just run once).
        base_delay:           Initial delay in seconds. Doubles each retry (1s, 2s, 4s...).
        retryable_exceptions: Tuple of exception types that trigger a retry.
                              Non-matching exceptions propagate immediately.

    The decorated function runs at most (1 + max_retries) times.
    On final failure, the last exception is re-raised.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %s: %s (backoff %.1fs)",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            type(e).__name__,
                            str(e)[:100],
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "All %d retries exhausted for %s: %s: %s",
                            max_retries,
                            func.__name__,
                            type(e).__name__,
                            str(e)[:100],
                        )
            raise last_exception
        return wrapper
    return decorator
