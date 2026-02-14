"""
News fetcher — aggregates news from all registered providers.

Uses the provider plugin system for rate limiting, quota tracking,
caching, and graceful degradation. Iterates providers by priority;
if one fails or is exhausted, falls through to the next.

Public API unchanged from the original:
    fetcher = NewsFetcher()
    articles = fetcher.fetch_for_symbol("RELIANCE", hours=72)
    market = fetcher.fetch_market_news(hours=24)
"""
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from loguru import logger

from providers.base import Article
from providers.cache import get_response_cache
from providers.quota import get_quota_manager
from src.storage.models import NewsArticle
from src.utils.logger import audit_log


def _article_to_news_article(article: Article) -> NewsArticle:
    """Convert provider Article to storage NewsArticle."""
    # Normalize published_at to UTC-aware to avoid naive vs aware comparison errors
    pub = article.published_at
    if pub is not None and pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return NewsArticle(
        id=article.id,
        source=article.source,
        url=article.url,
        title=article.title,
        description=article.description or None,
        content=article.content or None,
        published_at=pub,
    )


class NewsFetcher:
    """
    Fetches financial news from registered provider plugins.

    On init, discovers available providers and registers them with the
    quota manager. Each fetch call checks cache first, then iterates
    providers in priority order.
    """

    def __init__(self):
        from providers.news import get_news_providers

        self._providers = get_news_providers()
        self._quota = get_quota_manager()
        self._cache = get_response_cache()

        # Cache TTL from env (default 10 minutes)
        self._news_cache_ttl = int(os.getenv("NEWS_CACHE_TTL", "600"))

        # Register each provider with the quota manager
        for p in self._providers:
            self._quota.register(
                name=p.name,
                daily_limit=p.daily_limit,
                rpm=p.requests_per_minute,
                rps=p.requests_per_second,
            )

    def _fetch_with_providers(self, query: str, hours: int = 72,
                              max_results: int = 50) -> List[Article]:
        """Fetch articles using provider chain with quota + cache.

        1. Check cache
        2. Iterate providers in priority order
        3. For each: check quota → fetch → record success/failure → cache
        4. Stop when we have enough articles
        """
        cache_key = f"news:{query}:{hours}"

        # 1. Check cache
        cached = self._cache.get("news", cache_key)
        if cached is not None:
            logger.debug(f"News cache hit for '{query}' ({len(cached)} articles)")
            return cached

        # 2. Iterate providers
        all_articles: List[Article] = []
        providers_tried = 0

        for provider in self._providers:
            allowed, reason = self._quota.can_request(provider.name)
            if not allowed:
                logger.debug(
                    f"Skipping {provider.name}: {reason}"
                )
                continue

            providers_tried += 1
            try:
                # Wait for rate limiter and record request
                self._quota.wait_and_record(provider.name)

                articles = provider.fetch_articles(
                    query=query, hours=hours, max_results=max_results
                )
                self._quota.record_success(provider.name)

                if articles:
                    all_articles.extend(articles)
                    audit_log(
                        f"{provider.name.upper()}_FETCH",
                        query=query, count=len(articles)
                    )
                    logger.info(
                        f"{provider.name}: fetched {len(articles)} articles "
                        f"for '{query}'"
                    )

                    # Stop if we have enough
                    if len(all_articles) >= max_results:
                        break

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in [
                    "429", "rate limit", "too many requests",
                    "throttle", "quota exceeded",
                ])

                if is_rate_limit:
                    self._quota.record_rate_limit(provider.name)
                    logger.warning(f"{provider.name}: rate limited — rotating")
                else:
                    self._quota.record_failure(provider.name)
                    logger.warning(f"{provider.name}: fetch failed: {e}")

                # Continue to next provider
                continue

        if not all_articles and providers_tried == 0:
            logger.warning("No news providers available (all exhausted/unhealthy)")

        # 3. Cache results if we got articles.
        #    Don't cache empty results — `[]` is falsy in Python and causes
        #    downstream confusion between "no articles found" and "not cached".
        if all_articles:
            self._cache.set("news", cache_key, all_articles,
                            ttl_seconds=self._news_cache_ttl)

        return all_articles

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Deduplicate articles by URL."""
        seen_urls = set()
        unique = []
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique.append(article)
        return unique

    def fetch_for_symbol(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        hours: int = 72
    ) -> list[NewsArticle]:
        """Fetch news for a specific stock symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            company_name: Full company name for better search
            hours: Lookback period in hours

        Returns:
            List of NewsArticle objects
        """
        queries = [symbol]
        if company_name:
            queries.append(company_name)

        search_query = " OR ".join(queries)

        raw_articles = self._fetch_with_providers(
            query=search_query, hours=hours, max_results=10
        )

        # Convert to NewsArticle
        news_articles = [_article_to_news_article(a) for a in raw_articles]

        # Deduplicate
        unique = self._deduplicate(news_articles)

        logger.info(f"Found {len(unique)} unique articles for {symbol}")
        return unique

    def fetch_market_news(self, hours: int = 24) -> list[NewsArticle]:
        """Fetch general Indian stock market news.

        Args:
            hours: Lookback period

        Returns:
            List of NewsArticle objects
        """
        queries = [
            "NSE India stock market",
            "NIFTY Sensex",
            "Indian stock market",
        ]

        all_articles: List[NewsArticle] = []

        for query in queries:
            raw = self._fetch_with_providers(
                query=query, hours=hours, max_results=20
            )
            all_articles.extend([_article_to_news_article(a) for a in raw])

        # Deduplicate and sort by recency
        unique = self._deduplicate(all_articles)
        unique.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(f"Found {len(unique)} unique market news articles")
        return unique

    def search_news(
        self,
        keywords: list[str],
        hours: int = 72,
        limit: int = 50
    ) -> list[NewsArticle]:
        """Search news with multiple keywords.

        Args:
            keywords: List of search keywords
            hours: Lookback period
            limit: Max results

        Returns:
            List of NewsArticle objects
        """
        query = " OR ".join(keywords)
        raw = self._fetch_with_providers(
            query=query, hours=hours, max_results=limit
        )
        return [_article_to_news_article(a) for a in raw]
