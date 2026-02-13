"""
RSS feed provider plugin.

No API key needed. Free and unlimited.
Fetches from Indian financial news RSS feeds.
"""
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feedparser
from loguru import logger

from providers.base import Article, BaseNewsProvider


class RSSProvider(BaseNewsProvider):
    """Indian financial news via RSS feeds — free, no key needed."""

    DEFAULT_FEEDS: Dict[str, str] = {
        "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
        "economictimes_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "livemint": "https://www.livemint.com/rss/markets",
        "businessstandard": "https://www.business-standard.com/rss/markets-104.rss",
    }

    def __init__(self):
        # Allow overriding feeds via env (comma-separated keys)
        feed_keys = os.getenv("RSS_FEEDS", "")
        if feed_keys:
            keys = [k.strip() for k in feed_keys.split(",") if k.strip()]
            self._feeds = {k: self.DEFAULT_FEEDS[k]
                           for k in keys if k in self.DEFAULT_FEEDS}
        else:
            self._feeds = dict(self.DEFAULT_FEEDS)

    @property
    def name(self) -> str:
        return "rss"

    @property
    def daily_limit(self) -> Optional[int]:
        return None  # Unlimited

    @property
    def requests_per_minute(self) -> int:
        return 30

    @property
    def requests_per_second(self) -> int:
        return 5

    def is_available(self) -> bool:
        return bool(self._feeds)

    def fetch_articles(self, query: str, hours: int = 72,
                       max_results: int = 50) -> List[Article]:
        """Fetch articles from all configured RSS feeds.

        Note: RSS feeds return all recent articles — we don't filter by query
        here. The aggregator/caller handles relevance filtering.
        """
        articles = []
        query_lower = query.lower()

        for feed_name, url in self._feeds.items():
            try:
                feed = feedparser.parse(url)

                for entry in feed.entries:
                    try:
                        # Parse publication date
                        pub_date = None
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                        else:
                            pub_date = datetime.now(timezone.utc)

                        title = getattr(entry, "title", "")
                        summary = getattr(entry, "summary", "")
                        link = getattr(entry, "link", "")

                        # Basic relevance filter: check if query terms appear
                        # For short symbols (<=3 chars), require word boundary match
                        # to avoid false positives (e.g. "LT" matching "SALT", "ALTAR")
                        text = f"{title} {summary}".lower()
                        query_terms = [t.strip().lower() for t in query.split("OR")]
                        import re
                        matched = False
                        for term in query_terms:
                            term = term.strip()
                            if len(term) <= 3:
                                if re.search(r'\b' + re.escape(term) + r'\b', text):
                                    matched = True
                                    break
                            elif term in text:
                                matched = True
                                break
                        if not matched:
                            continue

                        content_list = entry.get("content", [{}])
                        content = (content_list[0].get("value", "")
                                   if content_list else "")

                        articles.append(Article(
                            id=hashlib.md5(link.encode()).hexdigest()[:16],
                            title=title,
                            description=summary,
                            content=content,
                            url=link,
                            source=feed_name,
                            published_at=pub_date,
                            provider=self.name,
                        ))
                    except Exception as e:
                        logger.warning(f"RSS: error parsing entry from {feed_name}: {e}")

            except Exception as e:
                logger.error(f"RSS: failed to fetch {feed_name}: {e}")

        return articles[:max_results]
