"""
Tests for utils/error_handler.py - CircuitBreaker, retry_with_backoff, categorize_exception.

Covers: circuit breaker state transitions, recovery via timeout, retry decorator
success/exhaustion, exception categorization, and registry singleton behavior.
"""
import time
from unittest.mock import MagicMock

import pytest

from utils.error_handler import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    ErrorCategory,
    RateLimitException,
    AuthException,
    BrokerException,
    TradingException,
    categorize_exception,
    retry_with_backoff,
)


class TestCircuitBreaker:
    """Test CircuitBreaker state machine."""

    def test_circuit_breaker_opens_after_failures(self):
        """5 consecutive failures -> state=OPEN, allow_request=False."""
        cb = CircuitBreaker(name="test", failure_threshold=5, recovery_timeout=60)

        # Starts CLOSED
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.allow_request() is True

        # Record 4 failures -- still CLOSED
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.allow_request() is True

        # 5th failure triggers OPEN
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.allow_request() is False

    def test_circuit_breaker_recovers(self):
        """OPEN -> wait past recovery_timeout -> HALF_OPEN -> 3 successes -> CLOSED."""
        cb = CircuitBreaker(
            name="recovery-test",
            failure_threshold=3,
            recovery_timeout=0.1,  # 100ms for fast test
            half_open_max_calls=3,
        )

        # Push to OPEN
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.allow_request() is False

        # Wait past recovery timeout
        time.sleep(0.2)

        # State check should transition OPEN -> HALF_OPEN
        assert cb.state == CircuitBreaker.State.HALF_OPEN
        assert cb.allow_request() is True

        # Record 3 successes to close the breaker
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitBreaker.State.HALF_OPEN  # Not yet closed

        cb.record_success()
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.allow_request() is True

    def test_circuit_breaker_half_open_failure_reopens(self):
        """A failure in HALF_OPEN state should reopen the breaker."""
        cb = CircuitBreaker(
            name="reopen-test",
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_max_calls=3,
        )

        # Push to OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN

        # Wait for recovery
        time.sleep(0.1)
        assert cb.state == CircuitBreaker.State.HALF_OPEN

        # A failure in HALF_OPEN reopens
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.allow_request() is False


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_retry_with_backoff_succeeds(self):
        """Function fails twice then succeeds -> returns result, no exception."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3  # 2 failures + 1 success

    def test_retry_exhausted_raises(self):
        """Function always fails -> exception raised after max_retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            always_fails()

        # Initial attempt + 2 retries = 3 total calls
        assert call_count == 3

    def test_retry_skips_on_exception(self):
        """skip_on exceptions are raised immediately without retry."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            skip_on=(AuthException,),
        )
        def auth_fail():
            nonlocal call_count
            call_count += 1
            raise AuthException("invalid token")

        with pytest.raises(AuthException):
            auth_fail()

        # Should be called only once (no retries for skip_on)
        assert call_count == 1


class TestCategorizeException:
    """Test categorize_exception for different error types."""

    def test_categorize_rate_limit(self):
        """Exception with '429' in message -> RATE_LIMIT."""
        exc = Exception("HTTP 429 Too Many Requests")
        assert categorize_exception(exc) == ErrorCategory.RATE_LIMIT

    def test_categorize_rate_limit_text(self):
        """Exception with 'rate limit' text -> RATE_LIMIT."""
        exc = Exception("rate limit exceeded, please slow down")
        assert categorize_exception(exc) == ErrorCategory.RATE_LIMIT

    def test_categorize_auth(self):
        """Exception with '401 unauthorized' -> AUTH."""
        exc = Exception("401 unauthorized access denied")
        assert categorize_exception(exc) == ErrorCategory.AUTH

    def test_categorize_auth_token(self):
        """Exception with 'token expired' -> AUTH."""
        exc = Exception("token expired, re-authenticate")
        assert categorize_exception(exc) == ErrorCategory.AUTH

    def test_categorize_timeout(self):
        """Exception with 'timeout' -> TRANSIENT."""
        exc = Exception("Connection timeout after 30s")
        assert categorize_exception(exc) == ErrorCategory.TRANSIENT

    def test_categorize_custom_exception(self):
        """Custom TradingException subclasses return their own category."""
        rate_exc = RateLimitException("slow down", retry_after=30)
        assert categorize_exception(rate_exc) == ErrorCategory.RATE_LIMIT
        assert rate_exc.retry_after == 30

        auth_exc = AuthException("bad key")
        assert categorize_exception(auth_exc) == ErrorCategory.AUTH

        broker_exc = BrokerException("order failed", ErrorCategory.TRANSIENT)
        assert categorize_exception(broker_exc) == ErrorCategory.TRANSIENT

    def test_categorize_unknown_fatal(self):
        """Unknown exception message defaults to FATAL."""
        exc = Exception("some completely unexpected error")
        assert categorize_exception(exc) == ErrorCategory.FATAL


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry singleton behavior."""

    def test_circuit_breaker_registry(self):
        """Get same endpoint twice -> same breaker instance."""
        registry = CircuitBreakerRegistry(failure_threshold=5, recovery_timeout=60)

        breaker1 = registry.get("quotes")
        breaker2 = registry.get("quotes")
        breaker_other = registry.get("orders")

        # Same endpoint returns the exact same object
        assert breaker1 is breaker2

        # Different endpoint returns a different object
        assert breaker1 is not breaker_other

        # Breaker names match endpoints
        assert breaker1.name == "quotes"
        assert breaker_other.name == "orders"

    def test_registry_allow_request_delegates(self):
        """Registry.allow_request delegates to the endpoint's circuit breaker."""
        registry = CircuitBreakerRegistry(failure_threshold=2, recovery_timeout=60)

        # Initially open
        assert registry.allow_request("test-ep") is True

        # Trip the breaker via the registry's breaker
        breaker = registry.get("test-ep")
        breaker.record_failure()
        breaker.record_failure()

        # Now blocked
        assert registry.allow_request("test-ep") is False

        # Other endpoints unaffected
        assert registry.allow_request("other-ep") is True
