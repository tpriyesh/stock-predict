"""
Finnhub news provider plugin.

Free tier: 60 calls/minute, no daily limit.
Endpoint: https://finnhub.io/api/v1/company-news
Financial-specific news with built-in sentiment.
"""
import hashlib
import os
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from loguru import logger

from providers.base import Article, BaseNewsProvider


class FinnhubNewsProvider(BaseNewsProvider):
    """Finnhub.io — financial news with sentiment (free 60 RPM)."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self):
        self._api_key = os.getenv("FINNHUB_API_KEY", "")

    @property
    def name(self) -> str:
        return "finnhub"

    @property
    def daily_limit(self) -> Optional[int]:
        return None  # Unlimited

    @property
    def requests_per_minute(self) -> int:
        return int(os.getenv("FINNHUB_RPM", "60"))

    @property
    def requests_per_second(self) -> int:
        return 5

    def is_available(self) -> bool:
        return bool(self._api_key)

    def fetch_articles(self, query: str, hours: int = 72,
                       max_results: int = 50) -> List[Article]:
        """Fetch company news from Finnhub.

        Finnhub uses symbol-based queries, not free text.
        We extract the first term from the query as the symbol.
        Falls back to general market news if no clear symbol.
        """
        # Extract symbol from query (e.g., "RELIANCE OR Reliance Industries")
        symbol = query.split()[0].strip().upper() if query else ""

        # Finnhub expects international symbols; for NSE use .NS suffix
        # But also try general market news endpoint
        from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

        articles = []

        # Try company-specific news
        if symbol:
            articles.extend(
                self._fetch_company_news(f"{symbol}.NS", from_date, to_date)
            )
            # Also try without suffix for broader results
            if not articles:
                articles.extend(
                    self._fetch_company_news(symbol, from_date, to_date)
                )

        # Fall back to general market news
        if not articles:
            articles.extend(self._fetch_general_news())

        return articles[:max_results]

    def _fetch_company_news(self, symbol: str, from_date: str,
                            to_date: str) -> List[Article]:
        """Fetch news for a specific company."""
        url = f"{self.BASE_URL}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self._api_key,
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                "429 Too Many Requests", response=response
            )
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            return []

        articles = []
        for item in data:
            try:
                url_str = item.get("url", "")
                # Finnhub timestamps are Unix epoch
                pub_ts = item.get("datetime", 0)
                pub_date = datetime.utcfromtimestamp(pub_ts) if pub_ts else datetime.utcnow()

                articles.append(Article(
                    id=hashlib.md5(url_str.encode()).hexdigest()[:16],
                    title=item.get("headline", ""),
                    description=item.get("summary", ""),
                    content=item.get("summary", ""),
                    url=url_str,
                    source=item.get("source", "finnhub"),
                    published_at=pub_date,
                    provider=self.name,
                ))
            except Exception as e:
                logger.warning(f"Finnhub: error parsing article: {e}")

        return articles

    def _fetch_general_news(self) -> List[Article]:
        """Fetch general market news."""
        url = f"{self.BASE_URL}/news"
        params = {
            "category": "general",
            "token": self._api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                raise requests.exceptions.HTTPError(
                    "429 Too Many Requests", response=response
                )
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, list):
                return []

            articles = []
            for item in data:
                try:
                    url_str = item.get("url", "")
                    pub_ts = item.get("datetime", 0)
                    pub_date = datetime.utcfromtimestamp(pub_ts) if pub_ts else datetime.utcnow()

                    articles.append(Article(
                        id=hashlib.md5(url_str.encode()).hexdigest()[:16],
                        title=item.get("headline", ""),
                        description=item.get("summary", ""),
                        content=item.get("summary", ""),
                        url=url_str,
                        source=item.get("source", "finnhub"),
                        published_at=pub_date,
                        provider=self.name,
                    ))
                except Exception as e:
                    logger.warning(f"Finnhub: error parsing general news: {e}")

            return articles

        except Exception as e:
            logger.error(f"Finnhub general news fetch failed: {e}")
            return []
