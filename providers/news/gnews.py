"""
GNews API provider plugin.

Free tier: 100 requests/day.
Endpoint: https://gnews.io/api/v4/search
"""
import hashlib
import os
from datetime import datetime
from typing import List, Optional

import requests
from loguru import logger

from providers.base import Article, BaseNewsProvider


class GNewsProvider(BaseNewsProvider):
    """GNews.io — global news aggregator with Indian news coverage."""

    ENDPOINT = "https://gnews.io/api/v4/search"

    def __init__(self):
        self._api_key = os.getenv("GNEWS_API_KEY", "")

    @property
    def name(self) -> str:
        return "gnews"

    @property
    def daily_limit(self) -> Optional[int]:
        return int(os.getenv("GNEWS_DAILY_LIMIT", "100"))

    @property
    def requests_per_minute(self) -> int:
        return int(os.getenv("GNEWS_RPM", "30"))

    @property
    def requests_per_second(self) -> int:
        return 5

    def is_available(self) -> bool:
        return bool(self._api_key)

    def fetch_articles(self, query: str, hours: int = 72,
                       max_results: int = 50) -> List[Article]:
        """Fetch articles from GNews."""
        import re
        # Sanitize query: GNews API rejects hyphens and ampersands
        clean_query = re.sub(r'[&\-]', ' ', query).strip()
        if not clean_query:
            return []
        params = {
            "q": clean_query,
            "lang": "en",
            "country": "in",
            "max": min(max_results, 100),
            "apikey": self._api_key,
        }

        response = requests.get(self.ENDPOINT, params=params, timeout=30)

        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                "429 Too Many Requests", response=response
            )
        response.raise_for_status()

        data = response.json()
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
                logger.warning(f"GNews: error parsing article: {e}")

        return articles
