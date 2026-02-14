"""
Rate Limiter - Prevents API rate limiting and bans.

Features:
- Configurable requests per minute/second
- Exponential backoff on failures
- Automatic retry with jitter
- Per-endpoint tracking

Fixes:
- Sleep OUTSIDE lock (was blocking all threads)
- Better jitter spread (50-150% instead of 10-50%)
- Per-endpoint cooldown tracking
"""
import time
import random
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
from typing import Callable, Any, Optional
from functools import wraps
from loguru import logger

from utils.platform import now_ist


class RateLimiter:
    """
    Thread-safe rate limiter with exponential backoff.

    Usage:
        limiter = RateLimiter(requests_per_minute=30)
        limiter.wait()  # Blocks if rate limit would be exceeded
        make_api_call()
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_second: int = 5,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0
    ):
        self.rpm = requests_per_minute
        self.rps = requests_per_second
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

        # Track request times
        self._request_times: list[datetime] = []
        self._lock = Lock()

        # Track failures per endpoint for adaptive backoff
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._last_failure_time: dict[str, datetime] = {}

        # Per-endpoint cooldown state
        self._cooldown_until: dict[str, datetime] = {}

    def wait(self, endpoint: str = "default") -> None:
        """
        Wait if necessary to respect rate limits.

        Args:
            endpoint: API endpoint identifier for tracking
        """
        sleep_time = 0.0

        with self._lock:
            now = now_ist().replace(tzinfo=None)

            # Check if this endpoint is in cooldown
            cooldown = self._cooldown_until.get(endpoint)
            if cooldown and now < cooldown:
                sleep_time = (cooldown - now).total_seconds()

        # Sleep OUTSIDE the lock to avoid blocking other threads
        if sleep_time > 0:
            logger.warning(f"Rate limit cooldown for {endpoint}: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

        sleep_time = 0.0
        with self._lock:
            now = now_ist().replace(tzinfo=None)

            # Clean old request times (keep last minute)
            cutoff = now - timedelta(minutes=1)
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check per-minute limit
            if len(self._request_times) >= self.rpm:
                oldest = self._request_times[0]
                wait_until = oldest + timedelta(minutes=1)
                sleep_time = (wait_until - now).total_seconds()

        if sleep_time > 0:
            logger.debug(f"Rate limit (RPM): waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

        sleep_time = 0.0
        with self._lock:
            now = now_ist().replace(tzinfo=None)

            # Check per-second burst limit
            one_second_ago = now - timedelta(seconds=1)
            recent = [t for t in self._request_times if t > one_second_ago]
            if len(recent) >= self.rps:
                sleep_time = 1.0 - (now - recent[0]).total_seconds()

        if sleep_time > 0:
            logger.debug(f"Rate limit (RPS): waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Record this request
        with self._lock:
            self._request_times.append(now_ist().replace(tzinfo=None))

    def record_success(self, endpoint: str = "default") -> None:
        """Record successful API call - resets failure count."""
        with self._lock:
            self._failure_counts[endpoint] = 0

    def record_failure(self, endpoint: str = "default", is_rate_limit: bool = False) -> float:
        """
        Record failed API call and return backoff time.

        Args:
            endpoint: API endpoint identifier
            is_rate_limit: True if failure was due to rate limiting (429)

        Returns:
            Recommended backoff time in seconds
        """
        with self._lock:
            self._failure_counts[endpoint] += 1
            self._last_failure_time[endpoint] = now_ist().replace(tzinfo=None)

            failures = self._failure_counts[endpoint]

            # Calculate exponential backoff with jitter
            backoff = min(
                self.base_backoff * (2 ** (failures - 1)),
                self.max_backoff
            )
            # Better jitter: 50-150% spread for true collision avoidance
            jitter = backoff * random.uniform(-0.5, 0.5)
            total_backoff = max(0.5, backoff + jitter)

            # If rate limited, enter per-endpoint cooldown mode
            if is_rate_limit:
                cooldown_time = total_backoff * 2
                self._cooldown_until[endpoint] = now_ist().replace(tzinfo=None) + timedelta(seconds=cooldown_time)
                logger.warning(f"Rate limit hit on {endpoint}! Cooldown for {cooldown_time:.1f}s")

            logger.warning(
                f"API failure #{failures} for {endpoint}. "
                f"Backing off {total_backoff:.1f}s"
            )

            return total_backoff

    def should_retry(self, endpoint: str = "default") -> bool:
        """Check if we should retry after a failure."""
        return self._failure_counts.get(endpoint, 0) < self.max_retries


def rate_limited(
    limiter: RateLimiter,
    endpoint: str = "default",
    retry_on: tuple = (Exception,)
):
    """
    Decorator for rate-limited API calls with automatic retry.

    Usage:
        limiter = RateLimiter()

        @rate_limited(limiter, "quotes")
        def get_quote(symbol):
            return api.get_quote(symbol)

    Args:
        limiter: RateLimiter instance
        endpoint: Endpoint identifier for tracking
        retry_on: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            # Initial wait
            limiter.wait(endpoint)

            while limiter.should_retry(endpoint):
                try:
                    result = func(*args, **kwargs)
                    limiter.record_success(endpoint)
                    return result

                except retry_on as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check if it's a rate limit error
                    is_rate_limit = any(x in error_str for x in [
                        '429', 'rate limit', 'too many requests',
                        'throttle', 'quota exceeded'
                    ])

                    backoff = limiter.record_failure(endpoint, is_rate_limit)

                    if limiter.should_retry(endpoint):
                        time.sleep(backoff)
                    else:
                        break

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Rate limiter exhausted for {endpoint}")

        return wrapper
    return decorator


class BrokerRateLimiter(RateLimiter):
    """
    Rate limiter with broker-specific defaults.

    Upstox limits: ~20 requests/second, 250 requests/minute
    Zerodha limits: ~3 requests/second, 120 requests/minute

    We use conservative limits to avoid any issues.
    """

    # Broker-specific conservative limits
    BROKER_LIMITS = {
        'upstox': {'rpm': 200, 'rps': 15},
        'zerodha': {'rpm': 100, 'rps': 2},
        'paper': {'rpm': 1000, 'rps': 100},  # No real limits
    }

    def __init__(self, broker: str = 'upstox'):
        limits = self.BROKER_LIMITS.get(broker, self.BROKER_LIMITS['upstox'])
        super().__init__(
            requests_per_minute=limits['rpm'],
            requests_per_second=limits['rps'],
            max_retries=3,
            base_backoff=2.0,
            max_backoff=120.0
        )
        self.broker = broker
        logger.info(
            f"Initialized {broker} rate limiter: "
            f"{limits['rpm']} RPM, {limits['rps']} RPS"
        )


# Global limiters for different services
_limiters: dict[str, RateLimiter] = {}


def get_limiter(service: str = "default") -> RateLimiter:
    """Get or create a rate limiter for a service."""
    if service not in _limiters:
        if service in BrokerRateLimiter.BROKER_LIMITS:
            _limiters[service] = BrokerRateLimiter(service)
        else:
            _limiters[service] = RateLimiter()
    return _limiters[service]
