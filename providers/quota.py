"""
Centralized API Quota Manager.

Single source of truth for ALL third-party API usage:
- Daily quota tracking (NewsAPI: 100/day, GNews: 100/day, etc.)
- Rate limiting via RateLimiter (RPM/RPS)
- Cooldown periods after rate limits (exponential backoff)
- Health tracking (consecutive failures → mark unhealthy)
- Cost tracking for paid APIs (OpenAI daily cap)
- Alerts at configurable threshold (default 80%)

Thread-safe. Resets daily at midnight IST.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from threading import Lock
from typing import Dict, Optional, Tuple

from loguru import logger

from utils.platform import now_ist, today_ist
from utils.rate_limiter import RateLimiter


@dataclass
class ProviderQuota:
    """Mutable state for one provider's quota/health."""
    name: str
    daily_limit: Optional[int] = None        # None = unlimited
    daily_cost_cap: Optional[float] = None   # None = no cost tracking
    rpm: int = 60
    rps: int = 5

    # Mutable state
    daily_used: int = 0
    daily_cost: float = 0.0
    reset_date: Optional[date] = None
    limiter: Optional[RateLimiter] = None
    cooldown_until: Optional[datetime] = None
    cooldown_seconds: float = 60.0           # Current cooldown (doubles on repeat)
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    is_healthy: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_rate_limit: Optional[datetime] = None


class APIQuotaManager:
    """Centralized quota, rate limit, and health tracking for all providers.

    Usage:
        quota = APIQuotaManager()
        quota.register("newsapi", daily_limit=100, rpm=30)
        quota.register("openai", rpm=180, daily_cost_cap=5.0)

        allowed, reason = quota.can_request("newsapi")
        if allowed:
            quota.record_request("newsapi")
            try:
                result = call_api()
                quota.record_success("newsapi")
            except RateLimitError:
                quota.record_rate_limit("newsapi")
            except Exception:
                quota.record_failure("newsapi")
    """

    # Configurable thresholds
    UNHEALTHY_THRESHOLD = 5       # Consecutive failures before marking unhealthy
    UNHEALTHY_COOLDOWN = 300      # Seconds to skip unhealthy provider
    QUOTA_ALERT_PCT = 0.80        # Alert when usage exceeds this fraction
    BASE_COOLDOWN = 60.0          # Initial cooldown after rate limit (seconds)
    MAX_COOLDOWN = 900.0          # Maximum cooldown (15 minutes)

    def __init__(self):
        self._providers: Dict[str, ProviderQuota] = {}
        self._lock = Lock()
        self._alert_sent: Dict[str, bool] = {}  # Track if 80% alert already sent

    def register(self, name: str, daily_limit: Optional[int] = None,
                 rpm: int = 60, rps: int = 5,
                 daily_cost_cap: Optional[float] = None):
        """Register a provider with its limits."""
        with self._lock:
            limiter = RateLimiter(
                requests_per_minute=rpm,
                requests_per_second=rps,
                max_retries=3,
                base_backoff=1.0,
                max_backoff=30.0,
            )
            self._providers[name] = ProviderQuota(
                name=name,
                daily_limit=daily_limit,
                daily_cost_cap=daily_cost_cap,
                rpm=rpm,
                rps=rps,
                limiter=limiter,
                reset_date=today_ist(),
            )
            logger.debug(
                f"Registered provider '{name}': "
                f"daily={daily_limit or 'unlimited'}, rpm={rpm}, rps={rps}"
            )

    def can_request(self, name: str) -> Tuple[bool, str]:
        """Check if a request is allowed right now.

        Returns (allowed, reason). Does NOT consume quota — call
        record_request() after a successful API call.
        """
        with self._lock:
            pq = self._providers.get(name)
            if not pq:
                return True, "unregistered"  # Unregistered = no limits

            self._maybe_reset_daily(pq)

            # 1. Health check
            if not pq.is_healthy:
                elapsed = (now_ist().replace(tzinfo=None) -
                           (pq.last_failure or now_ist().replace(tzinfo=None))
                           ).total_seconds()
                if elapsed < self.UNHEALTHY_COOLDOWN:
                    return False, f"unhealthy (cooldown {self.UNHEALTHY_COOLDOWN - elapsed:.0f}s remaining)"
                # Recovery: try again
                pq.is_healthy = True
                pq.consecutive_failures = 0
                logger.info(f"Provider '{name}' recovered from unhealthy state")

            # 2. Cooldown check (after rate limit)
            if pq.cooldown_until:
                now = now_ist().replace(tzinfo=None)
                if now < pq.cooldown_until:
                    remaining = (pq.cooldown_until - now).total_seconds()
                    return False, f"rate limit cooldown ({remaining:.0f}s remaining)"
                # Cooldown expired
                pq.cooldown_until = None

            # 3. Daily quota check
            if pq.daily_limit is not None and pq.daily_used >= pq.daily_limit:
                return False, f"daily quota exhausted ({pq.daily_used}/{pq.daily_limit})"

            # 4. Daily cost cap check
            if pq.daily_cost_cap is not None and pq.daily_cost >= pq.daily_cost_cap:
                return False, f"daily cost cap reached (${pq.daily_cost:.2f}/${pq.daily_cost_cap:.2f})"

            return True, "ok"

    def wait_and_record(self, name: str, cost: float = 0.0) -> Tuple[bool, str]:
        """Check quota, wait for rate limiter, and record the request.

        Convenience method combining can_request + rate limiter wait + record.
        Returns (allowed, reason). If allowed=True, the rate limiter wait
        has already completed and the request is recorded.
        """
        allowed, reason = self.can_request(name)
        if not allowed:
            return False, reason

        pq = self._providers.get(name)
        if pq and pq.limiter:
            pq.limiter.wait(name)

        self.record_request(name, cost=cost)
        return True, "ok"

    def record_request(self, name: str, cost: float = 0.0):
        """Record that a request was made (consumes quota)."""
        with self._lock:
            pq = self._providers.get(name)
            if not pq:
                return
            self._maybe_reset_daily(pq)
            pq.daily_used += 1
            pq.total_requests += 1
            pq.daily_cost += cost

            # Alert at threshold
            if (pq.daily_limit and
                    pq.daily_used >= int(pq.daily_limit * self.QUOTA_ALERT_PCT) and
                    not self._alert_sent.get(name)):
                self._alert_sent[name] = True
                pct = pq.daily_used / pq.daily_limit * 100
                logger.warning(
                    f"API QUOTA WARNING: {name} at {pct:.0f}% "
                    f"({pq.daily_used}/{pq.daily_limit})"
                )
                try:
                    from utils.alerts import alert_error
                    alert_error(
                        f"API quota warning: {name}",
                        f"Used {pq.daily_used}/{pq.daily_limit} ({pct:.0f}%). "
                        f"May exhaust before market close."
                    )
                except Exception:
                    pass

    def record_success(self, name: str):
        """Record successful API call — resets failure counter."""
        with self._lock:
            pq = self._providers.get(name)
            if not pq:
                return
            pq.consecutive_failures = 0
            pq.last_success = now_ist().replace(tzinfo=None)
            if pq.limiter:
                pq.limiter.record_success(name)

    def record_failure(self, name: str, is_rate_limit: bool = False):
        """Record failed API call. Tracks consecutive failures for health."""
        with self._lock:
            pq = self._providers.get(name)
            if not pq:
                return

            pq.consecutive_failures += 1
            pq.total_failures += 1
            pq.last_failure = now_ist().replace(tzinfo=None)

            if pq.limiter:
                pq.limiter.record_failure(name, is_rate_limit=is_rate_limit)

            if is_rate_limit:
                self._enter_cooldown(pq)

            # Health degradation
            if pq.consecutive_failures >= self.UNHEALTHY_THRESHOLD:
                pq.is_healthy = False
                logger.warning(
                    f"Provider '{name}' marked UNHEALTHY after "
                    f"{pq.consecutive_failures} consecutive failures"
                )

    def record_rate_limit(self, name: str, retry_after: Optional[int] = None):
        """Record a 429 / rate limit response. Enters cooldown."""
        with self._lock:
            pq = self._providers.get(name)
            if not pq:
                return
            pq.last_rate_limit = now_ist().replace(tzinfo=None)
            if retry_after:
                pq.cooldown_seconds = float(retry_after)
            self._enter_cooldown(pq)

        logger.warning(
            f"Rate limit hit for '{name}'. "
            f"Cooldown: {pq.cooldown_seconds:.0f}s"
        )

    def _enter_cooldown(self, pq: ProviderQuota):
        """Enter rate limit cooldown with exponential backoff."""
        now = now_ist().replace(tzinfo=None)
        from datetime import timedelta
        pq.cooldown_until = now + timedelta(seconds=pq.cooldown_seconds)
        # Double cooldown for next time (capped)
        pq.cooldown_seconds = min(pq.cooldown_seconds * 2, self.MAX_COOLDOWN)

    def _maybe_reset_daily(self, pq: ProviderQuota):
        """Reset daily counters at midnight IST."""
        today = today_ist()
        if pq.reset_date != today:
            pq.daily_used = 0
            pq.daily_cost = 0.0
            pq.reset_date = today
            pq.cooldown_seconds = self.BASE_COOLDOWN
            pq.cooldown_until = None
            # Reset alert tracking
            self._alert_sent.pop(pq.name, None)

    def get_usage(self) -> Dict[str, Dict]:
        """Get daily usage stats for all providers. For daily report."""
        with self._lock:
            result = {}
            for name, pq in self._providers.items():
                self._maybe_reset_daily(pq)
                entry = {
                    "used": pq.daily_used,
                    "limit": pq.daily_limit,
                    "pct": (pq.daily_used / pq.daily_limit * 100
                            if pq.daily_limit else None),
                    "cost": pq.daily_cost,
                    "cost_cap": pq.daily_cost_cap,
                    "healthy": pq.is_healthy,
                    "consecutive_failures": pq.consecutive_failures,
                    "total_requests": pq.total_requests,
                    "total_failures": pq.total_failures,
                    "in_cooldown": (pq.cooldown_until is not None and
                                    now_ist().replace(tzinfo=None) < pq.cooldown_until
                                    if pq.cooldown_until else False),
                }
                result[name] = entry
            return result

    def format_health_report(self) -> str:
        """Format API health for daily report file."""
        usage = self.get_usage()
        if not usage:
            return ""

        lines = [
            "-" * 60,
            "  API HEALTH",
            "-" * 60,
        ]
        for name, stats in sorted(usage.items()):
            status = "OK" if stats["healthy"] else "UNHEALTHY"
            if stats["in_cooldown"]:
                status = "COOLDOWN"

            if stats["limit"]:
                lines.append(
                    f"  {name:12s}  {stats['used']:>4d}/{stats['limit']:<4d} "
                    f"({stats['pct']:.0f}%)  {status}"
                )
            elif stats["cost_cap"]:
                lines.append(
                    f"  {name:12s}  ${stats['cost']:.2f}/${stats['cost_cap']:.2f}  "
                    f"{stats['used']} calls  {status}"
                )
            else:
                lines.append(
                    f"  {name:12s}  {stats['used']:>4d} calls  {status}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_quota_manager: Optional[APIQuotaManager] = None


def get_quota_manager() -> APIQuotaManager:
    """Get the global APIQuotaManager singleton."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = APIQuotaManager()
    return _quota_manager
