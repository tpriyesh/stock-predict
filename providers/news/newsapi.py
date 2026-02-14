"""
NewsAPI.org provider plugin.

Free tier: 100 requests/day, 250 requests/day on dev plan.
Endpoint: https://newsapi.org/v2/everything
"""
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests
from loguru import logger

from providers.base import Article, BaseNewsProvider


class NewsAPIProvider(BaseNewsProvider):
    """NewsAPI.org — global news aggregator with financial coverage."""

    ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self):
        self._api_key = os.getenv("NEWS_API_KEY", "")

    @property
    def name(self) -> str:
        return "newsapi"

    @property
    def daily_limit(self) -> Optional[int]:
        return int(os.getenv("NEWSAPI_DAILY_LIMIT", "100"))

    @property
    def requests_per_minute(self) -> int:
        return int(os.getenv("NEWSAPI_RPM", "30"))

    @property
    def requests_per_second(self) -> int:
        return 5

    def is_available(self) -> bool:
        return bool(self._api_key)

    def fetch_articles(self, query: str, hours: int = 72,
                       max_results: int = 50) -> List[Article]:
        """Fetch articles from NewsAPI."""
        now_utc = datetime.now(timezone.utc)
        from_date = now_utc - timedelta(hours=hours)

        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": now_utc.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),
            "apiKey": self._api_key,
        }

        response = requests.get(self.ENDPOINT, params=params, timeout=30)

        # Let caller handle rate limit detection
        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                "429 Too Many Requests", response=response
            )
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "ok":
            raise ValueError(f"NewsAPI error: {data.get('message', 'unknown')}")

        articles = []
        for item in data.get("articles", []):
            try:
                url = item.get("url", "")
                articles.append(Article(
                    id=hashlib.md5(url.encode()).hexdigest()[:16],
                    title=item.get("title") or "",
                    description=item.get("description") or "",
                    content=item.get("content") or "",
                    url=url,
                    source=item.get("source", {}).get("name", "Unknown"),
                    published_at=datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    ),
                    provider=self.name,
                ))
            except Exception as e:
                logger.warning(f"NewsAPI: error parsing article: {e}")

        return articles
