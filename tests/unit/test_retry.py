"""
tests/unit/test_retry.py — Unit tests for tools/retry.py

Covers: retry_with_backoff decorator behavior:
  - Succeeds on first try (no retries)
  - Retries on retryable exception, succeeds on 2nd try
  - Gives up after max_retries and re-raises
  - Non-retryable exceptions propagate immediately
  - Backoff delays are correct
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from tools.retry import retry_with_backoff


# ── Custom exceptions for testing ─────────────────────────────────────────────

class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


# ── Success on first call ────────────────────────────────────────────────────

class TestRetrySuccess:
    def test_returns_result_on_first_call(self):
        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def always_works():
            return "ok"

        assert always_works() == "ok"

    def test_no_sleep_on_success(self):
        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def always_works():
            return "ok"

        with patch("tools.retry.time.sleep") as mock_sleep:
            always_works()
            mock_sleep.assert_not_called()


# ── Retry then succeed ───────────────────────────────────────────────────────

class TestRetryThenSucceed:
    def test_succeeds_on_second_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("transient")
            return "recovered"

        with patch("tools.retry.time.sleep"):
            result = fails_once()

        assert result == "recovered"
        assert call_count == 2

    def test_succeeds_on_third_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("transient")
            return "recovered"

        with patch("tools.retry.time.sleep"):
            result = fails_twice()

        assert result == "recovered"
        assert call_count == 3


# ── Gives up after max retries ───────────────────────────────────────────────

class TestRetryExhausted:
    def test_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01, retryable_exceptions=(TransientError,))
        def always_fails():
            raise TransientError("still failing")

        with patch("tools.retry.time.sleep"):
            with pytest.raises(TransientError, match="still failing"):
                always_fails()

    def test_total_calls_is_1_plus_max_retries(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TransientError("fail")

        with patch("tools.retry.time.sleep"):
            with pytest.raises(TransientError):
                always_fails()

        assert call_count == 4  # 1 initial + 3 retries


# ── Non-retryable exceptions propagate immediately ───────────────────────────

class TestNonRetryable:
    def test_permanent_error_not_retried(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, retryable_exceptions=(TransientError,))
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise PermanentError("permanent")

        with pytest.raises(PermanentError, match="permanent"):
            permanent_failure()

        assert call_count == 1  # no retries


# ── Backoff delays ───────────────────────────────────────────────────────────

class TestBackoffDelays:
    def test_exponential_backoff_values(self):
        @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(TransientError,))
        def always_fails():
            raise TransientError("fail")

        with patch("tools.retry.time.sleep") as mock_sleep:
            with pytest.raises(TransientError):
                always_fails()

        # Expect: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0, 1.0 * 2^2 = 4.0
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    def test_zero_retries_means_no_retry(self):
        call_count = 0

        @retry_with_backoff(max_retries=0, base_delay=0.01, retryable_exceptions=(TransientError,))
        def fails_once():
            nonlocal call_count
            call_count += 1
            raise TransientError("fail")

        with pytest.raises(TransientError):
            fails_once()

        assert call_count == 1


# ── Arguments passed through ─────────────────────────────────────────────────

class TestArgsPassed:
    def test_args_and_kwargs_forwarded(self):
        @retry_with_backoff(max_retries=1, base_delay=0.01, retryable_exceptions=(TransientError,))
        def add(a, b, extra=0):
            return a + b + extra

        assert add(1, 2, extra=3) == 6
