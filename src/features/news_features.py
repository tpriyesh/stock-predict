"""
News feature extraction using LLM provider.
Extracts structured information from news articles for trading signals.

Uses the provider plugin system for rate limiting, quota tracking,
and cost monitoring. Falls back gracefully if LLM is unavailable.
"""
import hashlib
import json
import os
from typing import Optional
from datetime import datetime, timezone
from loguru import logger

from providers.quota import get_quota_manager
from providers.cache import get_response_cache
from config.settings import get_settings
from src.storage.models import NewsArticle, EventType, Sentiment
from src.utils.logger import audit_log


class NewsFeatureExtractor:
    """
    Extract structured features from news articles using LLM.
    """

    EXTRACTION_PROMPT = """Analyze this financial news article and extract structured information.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_at}
Content: {content}

TASK: Extract the following in JSON format:
1. tickers: List of stock symbols mentioned (use NSE symbols like RELIANCE, TCS, INFY)
2. event_type: One of [earnings, guidance, order_win, regulatory, macro, mna, dividend, split, other]
3. sentiment: One of [positive, neutral, negative]
4. sentiment_score: Float from -1.0 (very negative) to 1.0 (very positive)
5. key_claims: List of 2-3 key factual claims (not opinions)
6. relevance_score: How relevant is this for trading decisions (0.0 to 1.0)
7. risk_flags: List of any red flags like [rumor, unconfirmed, opinion, outdated]

Respond ONLY with valid JSON, no explanation.

Example output:
{{
    "tickers": ["RELIANCE", "ONGC"],
    "event_type": "earnings",
    "sentiment": "positive",
    "sentiment_score": 0.7,
    "key_claims": ["Q3 profit up 15% YoY", "Revenue beat estimates by 8%"],
    "relevance_score": 0.9,
    "risk_flags": []
}}"""

    def __init__(self):
        self._llm_provider = None
        self._quota = get_quota_manager()
        self._cache = get_response_cache()
        self._llm_cache_ttl = int(os.getenv("LLM_CACHE_TTL", "1800"))

        # Initialize LLM provider
        self._init_provider()

    def _init_provider(self):
        """Lazy-load the LLM provider from the provider registry."""
        try:
            from providers.llm import get_llm_providers
            providers = get_llm_providers()
            if providers:
                self._llm_provider = providers[0]
                # Register with quota manager
                self._quota.register(
                    name=self._llm_provider.name,
                    rpm=self._llm_provider.requests_per_minute,
                    daily_cost_cap=getattr(self._llm_provider, 'daily_cost_cap', None),
                )
                logger.info(f"LLM provider: {self._llm_provider.name}")
            else:
                logger.warning("No LLM providers available - news extraction disabled")
        except Exception as e:
            logger.warning(f"Failed to init LLM provider: {e}")

    def extract_features(self, article: NewsArticle) -> NewsArticle:
        """
        Extract features from a single article using LLM.

        Args:
            article: NewsArticle with raw content

        Returns:
            NewsArticle with extracted features populated
        """
        if not self._llm_provider:
            return article

        # Use URL-based cache key — stable across providers and re-fetches.
        # Article IDs may vary when the same article is fetched from different
        # providers, but the URL stays the same.
        url_or_id = article.url or str(article.id)
        stable_id = hashlib.md5(url_or_id.encode()).hexdigest()[:16]
        cache_key = f"extract:{stable_id}"
        cached = self._cache.get("llm", cache_key)
        if cached is not None:
            return cached

        content = article.content or article.description or article.title

        prompt = self.EXTRACTION_PROMPT.format(
            title=article.title,
            source=article.source,
            published_at=article.published_at.isoformat(),
            content=content[:2000]  # Limit content length
        )

        messages = [
            {"role": "system", "content": "You are a financial news analyst. Extract structured data from articles."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Check quota, wait for rate limiter, and record request
            cost = 0.0
            if hasattr(self._llm_provider, 'estimate_cost'):
                cost = self._llm_provider.estimate_cost(messages)

            allowed, reason = self._quota.wait_and_record(
                self._llm_provider.name, cost=cost
            )
            if not allowed:
                logger.debug(f"LLM quota blocked: {reason}")
                return article

            result_text = self._llm_provider.complete(
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )

            if not result_text:
                self._quota.record_failure(self._llm_provider.name)
                return article

            self._quota.record_success(self._llm_provider.name)

            # Parse JSON response
            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            result = json.loads(result_text)

            # Update article with extracted features
            updated_article = NewsArticle(
                id=article.id,
                source=article.source,
                url=article.url,
                title=article.title,
                description=article.description,
                content=article.content,
                published_at=article.published_at,
                fetched_at=article.fetched_at,
                tickers=result.get('tickers', []),
                event_type=EventType(result.get('event_type', 'other')),
                sentiment=Sentiment(result.get('sentiment', 'neutral')),
                sentiment_score=float(result.get('sentiment_score', 0.0)),
                key_claims=result.get('key_claims', []),
                relevance_score=float(result.get('relevance_score', 0.5))
            )

            # Cache the extracted result
            self._cache.set("llm", cache_key, updated_article,
                            ttl_seconds=self._llm_cache_ttl)

            audit_log(
                "NEWS_EXTRACTION",
                article_id=article.id,
                tickers=result.get('tickers'),
                sentiment=result.get('sentiment'),
                event_type=result.get('event_type')
            )

            return updated_article

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction result: {e}")
            self._quota.record_failure(self._llm_provider.name)
            return article
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in [
                "429", "rate limit", "too many requests",
            ])
            if is_rate_limit:
                self._quota.record_rate_limit(self._llm_provider.name)
            else:
                self._quota.record_failure(self._llm_provider.name)
            logger.error(f"News extraction failed: {e}")
            return article

    def extract_batch(
        self,
        articles: list[NewsArticle],
        max_concurrent: int = 5
    ) -> list[NewsArticle]:
        """
        Extract features from multiple articles.

        Args:
            articles: List of articles to process
            max_concurrent: Max parallel extractions (for rate limiting)

        Returns:
            List of articles with extracted features
        """
        if not self._llm_provider:
            return articles

        results = []
        for article in articles:
            try:
                extracted = self.extract_features(article)
                results.append(extracted)
            except Exception as e:
                logger.error(f"Failed to extract features from {article.id}: {e}")
                results.append(article)

        return results

    def get_symbol_news_summary(
        self,
        articles: list[NewsArticle],
        symbol: str
    ) -> dict:
        """
        Get aggregated news summary for a symbol.

        Args:
            articles: List of extracted articles
            symbol: Stock symbol to filter for

        Returns:
            Dict with aggregated news metrics
        """
        # Filter articles mentioning this symbol
        relevant = [a for a in articles if symbol in a.tickers]

        if not relevant:
            return {
                'symbol': symbol,
                'article_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_trend': 'NEUTRAL',
                'key_events': [],
                'recent_claims': []
            }

        # Time-decayed weighted average sentiment
        from utils.platform import now_ist
        import pytz
        now = now_ist()
        total_weight = 0.0
        weighted_sentiment = 0.0
        for a in relevant:
            pub_time = a.published_at
            if pub_time.tzinfo is None:
                pub_time = pytz.utc.localize(pub_time)
            hours_old = max(0, (now - pub_time).total_seconds() / 3600)
            weight = max(0.1, 1.0 - (hours_old / 72.0))
            weighted_sentiment += a.sentiment_score * weight
            total_weight += weight
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0

        # Determine sentiment trend
        if avg_sentiment > 0.3:
            sentiment_trend = 'POSITIVE'
        elif avg_sentiment < -0.3:
            sentiment_trend = 'NEGATIVE'
        else:
            sentiment_trend = 'NEUTRAL'

        # Collect key events
        event_types = {}
        for a in relevant:
            event_types[a.event_type.value] = event_types.get(a.event_type.value, 0) + 1

        # Get recent claims
        recent_claims = []
        for a in sorted(relevant, key=lambda x: x.published_at, reverse=True)[:5]:
            recent_claims.extend(a.key_claims[:2])

        return {
            'symbol': symbol,
            'article_count': len(relevant),
            'avg_sentiment': round(avg_sentiment, 3),
            'sentiment_trend': sentiment_trend,
            'event_types': event_types,
            'recent_claims': recent_claims[:5],
            'latest_article': relevant[0].title if relevant else None,
            'latest_source': relevant[0].source if relevant else None,
            'latest_url': relevant[0].url if relevant else None
        }

    def generate_news_reasoning(
        self,
        articles: list[NewsArticle],
        symbol: str
    ) -> str:
        """
        Generate human-readable reasoning from news.

        Args:
            articles: List of extracted articles
            symbol: Stock symbol

        Returns:
            Reasoning string for the signal
        """
        relevant = [a for a in articles if symbol in a.tickers]

        if not relevant:
            return "No recent news coverage found."

        # Sort by relevance and recency
        relevant.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)

        reasoning_parts = []

        for article in relevant[:3]:  # Top 3 most relevant
            sentiment_emoji = {
                Sentiment.POSITIVE: "+",
                Sentiment.NEGATIVE: "-",
                Sentiment.NEUTRAL: "~"
            }.get(article.sentiment, "~")

            pub_time = article.published_at
            if pub_time.tzinfo is None:
                pub_time = pub_time.replace(tzinfo=timezone.utc)
            time_ago = (datetime.now(timezone.utc) - pub_time).total_seconds() / 3600
            time_str = f"{int(time_ago)}h ago" if time_ago < 24 else f"{int(time_ago/24)}d ago"

            claims_str = "; ".join(article.key_claims[:2]) if article.key_claims else "No specific claims"

            reasoning_parts.append(
                f"[{sentiment_emoji}] {article.event_type.value.upper()}: {claims_str} "
                f"(source: {article.source}, {time_str})"
            )

        return "\n".join(reasoning_parts)
