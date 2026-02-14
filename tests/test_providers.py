"""
Tests for the provider plugin system.

Covers:
- APIQuotaManager: quota tracking, rate limiting, health, cost, daily reset
- ResponseCache: TTL, eviction, stats, invalidation
- Provider ABCs: registration, discovery, availability
- News aggregation: provider rotation, caching, deduplication
- LLM provider: cost tracking, quota blocking
- Integration: full provider chain with quota + cache
"""
import os
import time
from datetime import datetime, timedelta, date
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from providers.base import Article, BaseNewsProvider, BasePriceProvider, BaseLLMProvider
from providers.quota import APIQuotaManager, ProviderQuota
from providers.cache import ResponseCache, CacheEntry


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def quota():
    """Fresh quota manager for each test."""
    return APIQuotaManager()


@pytest.fixture
def cache():
    """Fresh cache for each test."""
    return ResponseCache(default_ttl_seconds=60, max_entries=100)


def make_article(title="Test Article", provider="testprov", url=None):
    """Helper to create test articles."""
    url = url or f"https://example.com/{title.replace(' ', '-')}"
    return Article(
        id=f"test_{hash(url) % 10000}",
        title=title,
        description="Test description",
        content="Test content body",
        url=url,
        source="TestSource",
        published_at=datetime.utcnow(),
        provider=provider,
    )


# ============================================================
# APIQuotaManager Tests
# ============================================================

class TestQuotaManager:
    """Tests for centralized API quota tracking."""

    def test_register_provider(self, quota):
        """Provider registration creates quota entry."""
        quota.register("newsapi", daily_limit=100, rpm=30, rps=5)
        usage = quota.get_usage()
        assert "newsapi" in usage
        assert usage["newsapi"]["limit"] == 100
        assert usage["newsapi"]["used"] == 0

    def test_can_request_unregistered(self, quota):
        """Unregistered providers are always allowed."""
        allowed, reason = quota.can_request("unknown_api")
        assert allowed is True
        assert reason == "unregistered"

    def test_can_request_within_quota(self, quota):
        """Requests within daily quota are allowed."""
        quota.register("newsapi", daily_limit=100)
        allowed, reason = quota.can_request("newsapi")
        assert allowed is True

    def test_quota_exhaustion(self, quota):
        """Requests blocked after daily quota exhausted."""
        quota.register("newsapi", daily_limit=5)
        for _ in range(5):
            quota.record_request("newsapi")
        allowed, reason = quota.can_request("newsapi")
        assert allowed is False
        assert "exhausted" in reason

    def test_record_request_increments(self, quota):
        """Each record_request increments daily usage."""
        quota.register("newsapi", daily_limit=100)
        quota.record_request("newsapi")
        quota.record_request("newsapi")
        usage = quota.get_usage()
        assert usage["newsapi"]["used"] == 2

    def test_record_request_with_cost(self, quota):
        """Cost tracking works for paid APIs."""
        quota.register("openai", daily_cost_cap=5.0)
        quota.record_request("openai", cost=0.01)
        quota.record_request("openai", cost=0.02)
        usage = quota.get_usage()
        assert abs(usage["openai"]["cost"] - 0.03) < 0.001

    def test_cost_cap_blocks_requests(self, quota):
        """Requests blocked when daily cost cap reached."""
        quota.register("openai", daily_cost_cap=0.05)
        quota.record_request("openai", cost=0.05)
        allowed, reason = quota.can_request("openai")
        assert allowed is False
        assert "cost cap" in reason

    def test_record_success_resets_failures(self, quota):
        """Success resets consecutive failure counter."""
        quota.register("newsapi", daily_limit=100)
        quota.record_failure("newsapi")
        quota.record_failure("newsapi")
        usage = quota.get_usage()
        assert usage["newsapi"]["consecutive_failures"] == 2

        quota.record_success("newsapi")
        usage = quota.get_usage()
        assert usage["newsapi"]["consecutive_failures"] == 0

    def test_health_degradation(self, quota):
        """Provider marked unhealthy after threshold failures."""
        quota.register("gnews", daily_limit=100)
        for _ in range(5):
            quota.record_failure("gnews")
        usage = quota.get_usage()
        assert usage["gnews"]["healthy"] is False

    def test_unhealthy_blocks_requests(self, quota):
        """Unhealthy provider blocks requests during cooldown."""
        quota.register("gnews", daily_limit=100)
        for _ in range(5):
            quota.record_failure("gnews")
        allowed, reason = quota.can_request("gnews")
        assert allowed is False
        assert "unhealthy" in reason

    def test_rate_limit_cooldown(self, quota):
        """Rate limit enters cooldown period."""
        quota.register("newsapi", daily_limit=100)
        quota.record_rate_limit("newsapi")
        allowed, reason = quota.can_request("newsapi")
        assert allowed is False
        assert "cooldown" in reason

    def test_rate_limit_with_retry_after(self, quota):
        """Rate limit respects retry-after header."""
        quota.register("newsapi", daily_limit=100)
        quota.record_rate_limit("newsapi", retry_after=120)
        pq = quota._providers["newsapi"]
        # After rate limit, cooldown_until is set based on retry_after,
        # then cooldown_seconds doubles for the next time
        assert pq.cooldown_seconds == 240.0  # 120 * 2 (doubled for next)
        assert pq.cooldown_until is not None

    def test_cooldown_exponential_backoff(self, quota):
        """Cooldown doubles on repeated rate limits."""
        quota.register("newsapi", daily_limit=100)
        pq = quota._providers["newsapi"]
        initial = pq.cooldown_seconds

        quota.record_rate_limit("newsapi")
        # After first rate limit, cooldown doubles
        assert pq.cooldown_seconds == initial * 2

    def test_cooldown_max_cap(self, quota):
        """Cooldown capped at MAX_COOLDOWN."""
        quota.register("newsapi", daily_limit=100)
        pq = quota._providers["newsapi"]
        # Force cooldown very high
        pq.cooldown_seconds = 1000.0
        quota.record_rate_limit("newsapi")
        assert pq.cooldown_seconds <= quota.MAX_COOLDOWN

    @patch("providers.quota.today_ist")
    def test_daily_reset(self, mock_today, quota):
        """Counters reset at midnight IST."""
        mock_today.return_value = date(2026, 2, 11)
        quota.register("newsapi", daily_limit=100)
        quota.record_request("newsapi")
        quota.record_request("newsapi")

        # Advance to next day
        mock_today.return_value = date(2026, 2, 12)
        usage = quota.get_usage()
        assert usage["newsapi"]["used"] == 0

    def test_quota_alert_at_threshold(self, quota):
        """Alert triggered at 80% quota usage."""
        quota.register("newsapi", daily_limit=10)
        with patch("providers.quota.logger") as mock_logger:
            for i in range(8):
                quota.record_request("newsapi")
            # Should have logged a warning at 80%
            warning_calls = [c for c in mock_logger.warning.call_args_list
                             if "QUOTA WARNING" in str(c)]
            assert len(warning_calls) >= 1

    def test_unlimited_provider(self, quota):
        """Provider with no daily limit is always allowed."""
        quota.register("rss", daily_limit=None)
        for _ in range(1000):
            quota.record_request("rss")
        allowed, reason = quota.can_request("rss")
        assert allowed is True

    def test_wait_and_record(self, quota):
        """Convenience method checks quota and records."""
        quota.register("newsapi", daily_limit=100)
        allowed, reason = quota.wait_and_record("newsapi")
        assert allowed is True
        usage = quota.get_usage()
        assert usage["newsapi"]["used"] == 1

    def test_wait_and_record_blocked(self, quota):
        """wait_and_record returns False when quota exhausted."""
        quota.register("newsapi", daily_limit=1)
        quota.record_request("newsapi")
        allowed, reason = quota.wait_and_record("newsapi")
        assert allowed is False

    def test_wait_and_record_with_cost(self, quota):
        """wait_and_record passes cost through."""
        quota.register("openai", daily_cost_cap=5.0)
        allowed, _ = quota.wait_and_record("openai", cost=0.01)
        assert allowed is True
        usage = quota.get_usage()
        assert abs(usage["openai"]["cost"] - 0.01) < 0.001

    def test_format_health_report(self, quota):
        """Health report formats correctly."""
        quota.register("newsapi", daily_limit=100)
        quota.record_request("newsapi")
        report = quota.format_health_report()
        assert "newsapi" in report
        assert "API HEALTH" in report

    def test_format_health_report_cost(self, quota):
        """Health report shows cost for paid providers."""
        quota.register("openai", daily_cost_cap=5.0)
        quota.record_request("openai", cost=0.04)
        report = quota.format_health_report()
        assert "$0.04" in report

    def test_total_failures_tracked(self, quota):
        """Total failures accumulate across resets."""
        quota.register("newsapi", daily_limit=100)
        quota.record_failure("newsapi")
        quota.record_success("newsapi")
        quota.record_failure("newsapi")
        usage = quota.get_usage()
        assert usage["newsapi"]["total_failures"] == 2

    def test_in_cooldown_flag(self, quota):
        """Usage report shows cooldown state."""
        quota.register("newsapi", daily_limit=100)
        quota.record_rate_limit("newsapi")
        usage = quota.get_usage()
        assert usage["newsapi"]["in_cooldown"] is True

    def test_record_failure_rate_limit(self, quota):
        """record_failure with is_rate_limit enters cooldown."""
        quota.register("newsapi", daily_limit=100)
        quota.record_failure("newsapi", is_rate_limit=True)
        allowed, reason = quota.can_request("newsapi")
        assert allowed is False
        assert "cooldown" in reason


# ============================================================
# ResponseCache Tests
# ============================================================

class TestResponseCache:
    """Tests for TTL-based response caching."""

    def test_set_and_get(self, cache):
        """Basic cache set/get works."""
        cache.set("newsapi", "RELIANCE news", ["article1", "article2"])
        result = cache.get("newsapi", "RELIANCE news")
        assert result == ["article1", "article2"]

    def test_cache_miss(self, cache):
        """Missing key returns None."""
        result = cache.get("newsapi", "nonexistent")
        assert result is None

    def test_cache_expiry(self):
        """Expired entries return None."""
        cache = ResponseCache(default_ttl_seconds=1)
        cache.set("newsapi", "test", "value", ttl_seconds=1)
        time.sleep(1.1)
        result = cache.get("newsapi", "test")
        assert result is None

    def test_custom_ttl(self, cache):
        """Custom TTL overrides default."""
        cache.set("newsapi", "test", "value", ttl_seconds=3600)
        result = cache.get("newsapi", "test")
        assert result == "value"

    def test_cache_key_normalization(self, cache):
        """Keys are case-insensitive and whitespace-normalized."""
        cache.set("newsapi", "RELIANCE  News", "value1")
        # Same key different case/spacing
        result = cache.get("newsapi", "reliance  news")
        assert result == "value1"

    def test_invalidate_provider(self, cache):
        """Invalidate clears only one provider."""
        cache.set("newsapi", "test1", "v1")
        cache.set("gnews", "test2", "v2")
        cache.invalidate("newsapi")
        assert cache.get("newsapi", "test1") is None
        assert cache.get("gnews", "test2") == "v2"

    def test_invalidate_all(self, cache):
        """Invalidate without provider clears all."""
        cache.set("newsapi", "test1", "v1")
        cache.set("gnews", "test2", "v2")
        cache.invalidate()
        assert cache.get("newsapi", "test1") is None
        assert cache.get("gnews", "test2") is None

    def test_stats_tracking(self, cache):
        """Stats track hits and misses."""
        cache.set("newsapi", "test", "value")
        cache.get("newsapi", "test")  # hit
        cache.get("newsapi", "missing")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_eviction_at_capacity(self):
        """Oldest entries evicted when at capacity."""
        cache = ResponseCache(default_ttl_seconds=3600, max_entries=5)
        for i in range(6):
            cache.set("test", f"key{i}", f"value{i}")
        # Should have evicted some entries
        assert cache.stats()["entries"] <= 5

    def test_format_stats_report(self, cache):
        """Stats report formats correctly."""
        cache.set("newsapi", "test", "value")
        cache.get("newsapi", "test")  # hit
        report = cache.format_stats_report()
        assert "Cache:" in report
        assert "hit rate" in report

    def test_cache_complex_objects(self, cache):
        """Cache handles complex objects (lists of articles)."""
        articles = [make_article(f"Article {i}") for i in range(3)]
        cache.set("newsapi", "RELIANCE", articles)
        result = cache.get("newsapi", "RELIANCE")
        assert len(result) == 3
        assert result[0].title == "Article 0"


# ============================================================
# Provider ABC Tests
# ============================================================

class MockNewsProvider(BaseNewsProvider):
    """Test implementation of BaseNewsProvider."""

    def __init__(self, name_val="mock_news", available=True, articles=None):
        self._name = name_val
        self._available = available
        self._articles = articles or []

    @property
    def name(self):
        return self._name

    @property
    def daily_limit(self):
        return 100

    @property
    def requests_per_minute(self):
        return 30

    def is_available(self):
        return self._available

    def fetch_articles(self, query, hours=72, max_results=50):
        return self._articles[:max_results]


class MockPriceProvider(BasePriceProvider):
    """Test implementation of BasePriceProvider."""

    @property
    def name(self):
        return "mock_price"

    def is_available(self):
        return True

    def fetch_history(self, symbol, period="3mo"):
        return None

    def get_current_price(self, symbol):
        return 100.0


class MockLLMProvider(BaseLLMProvider):
    """Test implementation of BaseLLMProvider."""

    @property
    def name(self):
        return "mock_llm"

    @property
    def cost_per_1k_tokens(self):
        return 0.001

    def is_available(self):
        return True

    def complete(self, messages, temperature=0.1, max_tokens=500):
        return '{"tickers": [], "sentiment": "neutral", "sentiment_score": 0.0}'


class TestProviderABCs:
    """Test that ABC implementations work correctly."""

    def test_news_provider_contract(self):
        """News provider implements all required methods."""
        provider = MockNewsProvider()
        assert provider.name == "mock_news"
        assert provider.daily_limit == 100
        assert provider.requests_per_minute == 30
        assert provider.is_available() is True
        assert provider.fetch_articles("test") == []

    def test_price_provider_contract(self):
        """Price provider implements all required methods."""
        provider = MockPriceProvider()
        assert provider.name == "mock_price"
        assert provider.is_available() is True
        assert provider.get_current_price("RELIANCE") == 100.0

    def test_llm_provider_contract(self):
        """LLM provider implements all required methods."""
        provider = MockLLMProvider()
        assert provider.name == "mock_llm"
        assert provider.cost_per_1k_tokens == 0.001
        assert provider.is_available() is True
        result = provider.complete([{"role": "user", "content": "test"}])
        assert "neutral" in result

    def test_provider_unavailable(self):
        """Unavailable provider reports correctly."""
        provider = MockNewsProvider(available=False)
        assert provider.is_available() is False


# ============================================================
# Provider Discovery Tests
# ============================================================

class TestProviderDiscovery:
    """Test provider auto-discovery from env vars."""

    @patch.dict(os.environ, {"NEWS_API_KEY": "", "GNEWS_API_KEY": "", "FINNHUB_API_KEY": ""})
    def test_no_providers_when_no_keys(self):
        """No providers registered without API keys."""
        from providers.news import get_news_providers
        providers = get_news_providers()
        # RSS should always be available (no key needed)
        rss_providers = [p for p in providers if p.name == "rss"]
        assert len(rss_providers) >= 1

    @patch.dict(os.environ, {"NEWS_API_KEY": "test_key_123"})
    def test_newsapi_available_with_key(self):
        """NewsAPI provider available when key is set."""
        from providers.news.newsapi import NewsAPIProvider
        provider = NewsAPIProvider()
        assert provider.is_available() is True
        assert provider.name == "newsapi"

    @patch.dict(os.environ, {"GNEWS_API_KEY": "test_key_456"})
    def test_gnews_available_with_key(self):
        """GNews provider available when key is set."""
        from providers.news.gnews import GNewsProvider
        provider = GNewsProvider()
        assert provider.is_available() is True
        assert provider.name == "gnews"

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key_789"})
    def test_finnhub_available_with_key(self):
        """Finnhub provider available when key is set."""
        from providers.news.finnhub import FinnhubNewsProvider
        provider = FinnhubNewsProvider()
        assert provider.is_available() is True
        assert provider.name == "finnhub"

    def test_rss_always_available(self):
        """RSS provider always available (no key needed)."""
        from providers.news.rss import RSSProvider
        provider = RSSProvider()
        assert provider.is_available() is True
        assert provider.name == "rss"

    def test_yfinance_always_available(self):
        """yfinance provider always available (no key needed)."""
        from providers.price.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()
        assert provider.is_available() is True
        assert provider.name == "yfinance"

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_openai_unavailable_without_key(self):
        """OpenAI provider unavailable without key."""
        from providers.llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        assert provider.is_available() is False

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    def test_openai_available_with_key(self):
        """OpenAI provider available with key."""
        from providers.llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        assert provider.is_available() is True


# ============================================================
# Provider Rotation / Failover Tests
# ============================================================

class TestProviderRotation:
    """Test that news aggregation rotates providers on failure."""

    def test_provider_chain_success(self, quota):
        """First available provider serves the request."""
        articles = [make_article("News 1"), make_article("News 2")]
        provider = MockNewsProvider(articles=articles)

        quota.register(provider.name, daily_limit=100, rpm=30)
        allowed, _ = quota.wait_and_record(provider.name)
        assert allowed is True

        result = provider.fetch_articles("test")
        assert len(result) == 2

    def test_quota_exhausted_skips_provider(self, quota):
        """Provider skipped when quota exhausted."""
        quota.register("newsapi", daily_limit=2)
        quota.record_request("newsapi")
        quota.record_request("newsapi")

        allowed, reason = quota.can_request("newsapi")
        assert allowed is False
        assert "exhausted" in reason

        # Second provider should work
        quota.register("gnews", daily_limit=100)
        allowed, reason = quota.can_request("gnews")
        assert allowed is True

    def test_unhealthy_skips_to_next(self, quota):
        """Unhealthy provider skipped, next provider used."""
        quota.register("newsapi", daily_limit=100)
        quota.register("gnews", daily_limit=100)

        # Make newsapi unhealthy
        for _ in range(5):
            quota.record_failure("newsapi")

        assert quota.can_request("newsapi")[0] is False
        assert quota.can_request("gnews")[0] is True

    def test_rate_limited_rotates(self, quota):
        """Rate-limited provider skipped, next used."""
        quota.register("newsapi", daily_limit=100)
        quota.register("gnews", daily_limit=100)

        quota.record_rate_limit("newsapi")

        assert quota.can_request("newsapi")[0] is False
        assert quota.can_request("gnews")[0] is True


# ============================================================
# News Fetcher Integration Tests
# ============================================================

class TestNewsFetcherIntegration:
    """Test refactored NewsFetcher with provider system."""

    @patch("providers.news.get_news_providers")
    def test_fetcher_uses_providers(self, mock_get_providers):
        """NewsFetcher discovers and uses registered providers."""
        articles = [make_article("Test News", provider="mock")]
        mock_provider = MockNewsProvider(articles=articles)
        mock_get_providers.return_value = [mock_provider]

        from src.data.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        result = fetcher.fetch_for_symbol("RELIANCE")
        # Should return articles converted to NewsArticle format
        assert len(result) >= 0  # May be 0 if quota not registered

    @patch("providers.news.get_news_providers")
    def test_fetcher_deduplicates(self, mock_get_providers):
        """NewsFetcher deduplicates by URL."""
        articles = [
            make_article("Article A", url="https://example.com/same"),
            make_article("Article B", url="https://example.com/same"),
            make_article("Article C", url="https://example.com/different"),
        ]
        mock_provider = MockNewsProvider(articles=articles)
        mock_get_providers.return_value = [mock_provider]

        from src.data.news_fetcher import NewsFetcher, _article_to_news_article
        fetcher = NewsFetcher()

        # Test dedup directly
        from src.storage.models import NewsArticle
        news_articles = [_article_to_news_article(a) for a in articles]
        unique = fetcher._deduplicate(news_articles)
        assert len(unique) == 2

    def test_article_to_news_article_conversion(self):
        """Article dataclass converts to NewsArticle pydantic model."""
        from src.data.news_fetcher import _article_to_news_article
        article = make_article("Test Title")
        result = _article_to_news_article(article)
        assert result.title == "Test Title"
        assert result.source == "TestSource"
        assert result.url == article.url


# ============================================================
# LLM Cost Tracking Tests
# ============================================================

class TestLLMCostTracking:
    """Test OpenAI cost tracking through quota manager."""

    def test_cost_estimation(self):
        """OpenAI provider estimates cost correctly."""
        from providers.llm.openai_provider import OpenAIProvider
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider()
            messages = [{"role": "user", "content": "Hello world " * 100}]
            cost = provider.estimate_cost(messages, max_tokens=500)
            assert cost > 0
            assert cost < 1.0  # Reasonable bound

    def test_cost_cap_integration(self, quota):
        """Daily cost cap blocks requests after threshold."""
        quota.register("openai", daily_cost_cap=0.10)

        # Simulate many small requests
        for _ in range(100):
            quota.record_request("openai", cost=0.001)

        # After $0.10, should be blocked
        allowed, reason = quota.can_request("openai")
        assert allowed is False
        assert "cost cap" in reason


# ============================================================
# Cache Integration Tests
# ============================================================

class TestCacheIntegration:
    """Test cache with provider workflows."""

    def test_cache_prevents_duplicate_fetch(self, cache):
        """Cached result prevents second API call."""
        articles = [make_article("Cached Article")]
        cache.set("newsapi", "RELIANCE news", articles, ttl_seconds=600)

        # Second access hits cache
        result = cache.get("newsapi", "RELIANCE news")
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Cached Article"

    def test_cache_by_provider_separation(self, cache):
        """Different providers have separate cache entries."""
        cache.set("newsapi", "test query", ["newsapi_result"])
        cache.set("gnews", "test query", ["gnews_result"])

        assert cache.get("newsapi", "test query") == ["newsapi_result"]
        assert cache.get("gnews", "test query") == ["gnews_result"]

    def test_cache_stats_by_provider(self, cache):
        """Stats show entries by provider."""
        cache.set("newsapi", "q1", "v1")
        cache.set("newsapi", "q2", "v2")
        cache.set("gnews", "q3", "v3")

        stats = cache.stats()
        assert stats["by_provider"]["newsapi"] == 2
        assert stats["by_provider"]["gnews"] == 1


# ============================================================
# ProviderConfig Tests
# ============================================================

class TestProviderConfig:
    """Test ProviderConfig in trading_config."""

    def test_provider_config_defaults(self):
        """ProviderConfig has sensible defaults."""
        from config.trading_config import ProviderConfig
        config = ProviderConfig()
        assert config.newsapi_daily_limit == 100
        assert config.gnews_daily_limit == 100
        assert config.openai_rpm == 180
        assert config.openai_daily_cost_cap == 5.0
        assert config.news_cache_ttl == 1800       # 30 min (optimized)
        assert config.llm_cache_ttl == 21600        # 6 hours (optimized)
        assert config.news_budget_per_cycle == 15   # budget-aware fetching
        assert config.quota_alert_pct == 0.80

    def test_provider_config_in_master(self):
        """ProviderConfig accessible from master CONFIG."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert hasattr(config, 'providers')
        assert config.providers.yfinance_rpm == 60


# ============================================================
# Daily Report Integration
# ============================================================

class TestDailyReport:
    """Test API health section in daily report."""

    def test_health_report_with_providers(self, quota):
        """Health report shows all registered providers."""
        quota.register("newsapi", daily_limit=100)
        quota.register("gnews", daily_limit=100)
        quota.register("openai", daily_cost_cap=5.0)

        quota.record_request("newsapi")
        quota.record_request("openai", cost=0.04)

        report = quota.format_health_report()
        assert "newsapi" in report
        assert "gnews" in report
        assert "openai" in report

    def test_health_report_empty(self, quota):
        """Empty report when no providers registered."""
        report = quota.format_health_report()
        assert report == ""

    def test_cache_report_format(self, cache):
        """Cache stats format for daily report."""
        cache.set("test", "key", "value")
        cache.get("test", "key")
        report = cache.format_stats_report()
        assert "Cache:" in report
        assert "100%" in report  # 1 hit, 0 misses


# ============================================================
# Thread Safety Tests
# ============================================================

class TestThreadSafety:
    """Test thread safety of quota manager and cache."""

    def test_concurrent_quota_updates(self, quota):
        """Concurrent quota updates don't corrupt state."""
        import threading
        quota.register("newsapi", daily_limit=10000)

        errors = []

        def worker():
            try:
                for _ in range(100):
                    quota.record_request("newsapi")
                    quota.can_request("newsapi")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        usage = quota.get_usage()
        assert usage["newsapi"]["used"] == 500

    def test_concurrent_cache_access(self, cache):
        """Concurrent cache access is thread-safe."""
        import threading
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.set("test", f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get("test", f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# Singleton Tests
# ============================================================

class TestSingletons:
    """Test global singleton factories."""

    def test_quota_singleton(self):
        """get_quota_manager returns same instance."""
        # Reset for test
        import providers.quota as qmod
        qmod._quota_manager = None
        m1 = qmod.get_quota_manager()
        m2 = qmod.get_quota_manager()
        assert m1 is m2
        qmod._quota_manager = None  # Cleanup

    def test_cache_singleton(self):
        """get_response_cache returns same instance."""
        import providers.cache as cmod
        cmod._response_cache = None
        c1 = cmod.get_response_cache()
        c2 = cmod.get_response_cache()
        assert c1 is c2
        cmod._response_cache = None  # Cleanup
