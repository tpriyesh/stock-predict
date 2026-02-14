"""
Error Handler - Centralized error handling and recovery.

Provides:
- Categorized exceptions for different failure types
- Retry decorators with exponential backoff
- Thread-safe circuit breaker pattern (per-endpoint)
- Error logging and alerting

Fixes:
- Circuit breaker is now thread-safe (uses Lock)
- Per-endpoint circuit breakers via CircuitBreakerRegistry
- Retry delay calculation fixed (uses actual_delay, not unbounded delay)
- Categorize exception more carefully (avoid false positives on 'retry')
"""
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import Callable, Any, Optional, Type, Tuple
from enum import Enum
from loguru import logger

from utils.platform import now_ist


class ErrorCategory(Enum):
    """Categories of errors for handling decisions."""
    TRANSIENT = "TRANSIENT"      # Retry-able (network issues, timeouts)
    RATE_LIMIT = "RATE_LIMIT"    # Back off significantly
    FATAL = "FATAL"              # Don't retry, needs manual intervention
    DATA = "DATA"                # Bad data, skip this item
    AUTH = "AUTH"                # Authentication failure


class TradingException(Exception):
    """Base exception for trading system."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.TRANSIENT):
        self.message = message
        self.category = category
        try:
            from utils.platform import now_ist
            self.timestamp = now_ist().replace(tzinfo=None)
        except ImportError:
            self.timestamp = datetime.now()
        super().__init__(message)


class BrokerException(TradingException):
    """Broker-related errors."""
    pass


class OrderException(TradingException):
    """Order placement/execution errors."""
    pass


class DataException(TradingException):
    """Data fetching/processing errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.DATA)


class RateLimitException(TradingException):
    """Rate limit hit."""

    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message, ErrorCategory.RATE_LIMIT)
        self.retry_after = retry_after


class AuthException(TradingException):
    """Authentication errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.AUTH)


def categorize_exception(e: Exception) -> ErrorCategory:
    """
    Categorize an exception for handling decisions.
    """
    # Check our custom exceptions first
    if isinstance(e, TradingException):
        return e.category

    error_str = str(e).lower()

    # Rate limiting
    if any(x in error_str for x in ['429', 'rate limit', 'too many requests', 'throttle', 'quota exceeded']):
        return ErrorCategory.RATE_LIMIT

    # Authentication
    if any(x in error_str for x in ['401', '403', 'unauthorized', 'token expired', 'invalid key']):
        return ErrorCategory.AUTH

    # Transient network issues
    if any(x in error_str for x in [
        'timeout', 'connectionerror', 'networkerror', 'socket', 'dns',
        '500', '502', '503', '504', 'service unavailable'
    ]):
        return ErrorCategory.TRANSIENT

    # Data issues
    if any(x in error_str for x in ['no data', 'not found', 'invalid symbol', 'empty response']):
        return ErrorCategory.DATA

    # Default to fatal for unknown errors
    return ErrorCategory.FATAL


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    skip_on: Tuple[Type[Exception], ...] = ()
):
    """
    Decorator for retrying functions with exponential backoff.

    Fixed: delay calculation now properly bounded by max_delay.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except skip_on as e:
                    # Don't retry these
                    logger.error(f"{func.__name__} failed (no retry): {e}")
                    raise

                except retry_on as e:
                    last_exception = e
                    category = categorize_exception(e)

                    if attempt < max_retries:
                        # Calculate backoff (bounded by max_delay)
                        if category == ErrorCategory.RATE_LIMIT:
                            actual_delay = min(delay * 3, max_delay)
                        else:
                            actual_delay = min(delay, max_delay)

                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {actual_delay:.1f}s..."
                        )

                        time.sleep(actual_delay)
                        # Use actual_delay for next iteration (fixed from using unbounded delay)
                        delay = min(actual_delay * 2, max_delay)

                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Thread-safe circuit breaker pattern.

    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    class State(Enum):
        CLOSED = "CLOSED"
        OPEN = "OPEN"
        HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._lock = Lock()
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

    @property
    def state(self) -> State:
        """Get current state, potentially transitioning from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == self.State.OPEN:
                if self._last_failure_time:
                    elapsed = (now_ist().replace(tzinfo=None) - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = self.State.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info(f"Circuit breaker [{self.name}]: OPEN -> HALF_OPEN")
            return self._state

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = self.State.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker [{self.name}]: HALF_OPEN -> CLOSED")
            elif self._state == self.State.CLOSED:
                # Decay failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = now_ist().replace(tzinfo=None)

            if self._state == self.State.HALF_OPEN:
                self._state = self.State.OPEN
                logger.warning(f"Circuit breaker [{self.name}]: HALF_OPEN -> OPEN")

            elif self._state == self.State.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.State.OPEN
                    logger.warning(
                        f"Circuit breaker [{self.name}]: CLOSED -> OPEN "
                        f"({self._failure_count} failures)"
                    )

    def allow_request(self) -> bool:
        """Check if requests should be allowed."""
        state = self.state  # This may transition OPEN -> HALF_OPEN
        if state == self.State.OPEN:
            return False
        return True

    def reset(self):
        """Reset circuit breaker to CLOSED state. Call at start of each trading session."""
        with self._lock:
            old_state = self._state
            self._state = self.State.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            if old_state != self.State.CLOSED:
                logger.info(f"Circuit breaker [{self.name}]: {old_state.value} -> CLOSED (session reset)")

    def protect(self):
        """Context manager for protecting calls."""
        return CircuitBreakerContext(self)


class CircuitBreakerRegistry:
    """
    Registry of per-endpoint circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get("quotes")
        with breaker.protect():
            fetch_quotes()
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def get(self, endpoint: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an endpoint."""
        with self._lock:
            if endpoint not in self._breakers:
                self._breakers[endpoint] = CircuitBreaker(
                    name=endpoint,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout=self._recovery_timeout
                )
            return self._breakers[endpoint]

    def allow_request(self, endpoint: str) -> bool:
        """Check if an endpoint allows requests."""
        return self.get(endpoint).allow_request()

    def reset_all(self):
        """Reset all circuit breakers. Call at start of each trading session."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info(f"All circuit breakers reset ({len(self._breakers)} endpoints)")


class CircuitBreakerContext:
    """Context manager for circuit breaker."""

    def __init__(self, breaker: CircuitBreaker):
        self.breaker = breaker

    def __enter__(self):
        if not self.breaker.allow_request():
            raise BrokerException(
                f"Circuit breaker [{self.breaker.name}] is OPEN - service unavailable",
                ErrorCategory.TRANSIENT
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.breaker.record_success()
        else:
            self.breaker.record_failure()
        return False  # Don't suppress exception


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """Execute a function safely, catching exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"{func.__name__} failed: {e}")
        return default


def log_exception(e: Exception, context: str = ""):
    """Log an exception with full context."""
    category = categorize_exception(e)
    tb = traceback.format_exc()

    if context:
        logger.error(f"[{category.value}] {context}: {e}")
    else:
        logger.error(f"[{category.value}] {e}")

    logger.debug(f"Traceback:\n{tb}")
