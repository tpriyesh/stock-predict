"""
Firecrawl integration for scraping Indian financial news sites.
Scrapes MoneyControl, Economic Times, Livemint, etc.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Optional
import requests
from loguru import logger

from config.settings import get_settings
from src.storage.models import NewsArticle, EventType, Sentiment


class FirecrawlFetcher:
    """
    Fetches financial news and data using Firecrawl.

    Firecrawl provides:
    - /scrape: Extract content from single page
    - /crawl: Crawl multiple pages
    - /extract: LLM-powered structured extraction (uses v1 API)
    """

    BASE_URL = "https://api.firecrawl.dev/v1"  # v1 for extract support
    BASE_URL_V2 = "https://api.firecrawl.dev/v2"  # v2 for basic scraping

    # Indian financial news sources
    SOURCES = {
        'moneycontrol': {
            'base_url': 'https://www.moneycontrol.com',
            'news_url': 'https://www.moneycontrol.com/news/business/stocks/',
            'stock_url': 'https://www.moneycontrol.com/india/stockpricequote/{sector}/{symbol}',
        },
        'economictimes': {
            'base_url': 'https://economictimes.indiatimes.com',
            'markets_url': 'https://economictimes.indiatimes.com/markets/stocks/news',
            'stock_url': 'https://economictimes.indiatimes.com/{symbol}/stocks/companyid-{id}.cms',
        },
        'livemint': {
            'base_url': 'https://www.livemint.com',
            'markets_url': 'https://www.livemint.com/market/stock-market-news',
        },
        'businessstandard': {
            'base_url': 'https://www.business-standard.com',
            'markets_url': 'https://www.business-standard.com/markets/news',
        },
        'ndtvprofit': {
            'base_url': 'https://www.ndtvprofit.com',
            'markets_url': 'https://www.ndtvprofit.com/markets',
        }
    }

    # Schema for extracting stock news
    NEWS_EXTRACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "url": {"type": "string"},
                        "published_time": {"type": "string"},
                        "stocks_mentioned": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"]
                        },
                        "event_type": {
                            "type": "string",
                            "enum": ["earnings", "order_win", "regulatory", "guidance", "dividend", "other"]
                        },
                        "key_numbers": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["title", "summary"]
                }
            }
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Firecrawl fetcher.

        Args:
            api_key: Firecrawl API key (or set FIRECRAWL_API_KEY env var)
        """
        if api_key:
            self.api_key = api_key
        else:
            # Try settings first, then env var
            try:
                from config.settings import get_settings
                settings = get_settings()
                self.api_key = settings.firecrawl_api_key
            except:
                self.api_key = os.getenv('FIRECRAWL_API_KEY', '')

        if not self.api_key:
            logger.warning("Firecrawl API key not set. Set FIRECRAWL_API_KEY env var.")

    def _make_request(self, endpoint: str, payload: dict, use_v2: bool = False) -> dict:
        """Make request to Firecrawl API.

        Args:
            endpoint: API endpoint (e.g., 'scrape')
            payload: Request payload
            use_v2: Use v2 API (for basic scraping). Default v1 (for extract).
        """
        if not self.api_key:
            return {'error': 'No API key configured'}

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        base_url = self.BASE_URL_V2 if use_v2 else self.BASE_URL

        try:
            response = requests.post(
                f"{base_url}/{endpoint}",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Firecrawl request failed: {e}")
            return {'error': str(e)}

    def scrape_url(self, url: str, formats: list = None) -> dict:
        """
        Scrape a single URL.

        Args:
            url: URL to scrape
            formats: Output formats ['markdown', 'html', 'links', 'screenshot']

        Returns:
            Scraped content
        """
        payload = {
            'url': url,
            'formats': formats or ['markdown']
        }

        return self._make_request('scrape', payload, use_v2=True)

    def extract_structured(self, url: str, schema: dict, prompt: str = None) -> dict:
        """
        Extract structured data from URL using LLM.

        Args:
            url: URL to extract from
            schema: JSON schema for extraction
            prompt: Optional extraction prompt

        Returns:
            Extracted structured data
        """
        payload = {
            'url': url,
            'formats': ['extract'],
            'extract': {
                'schema': schema
            }
        }

        if prompt:
            payload['extract']['prompt'] = prompt

        return self._make_request('scrape', payload)

    def scrape_market_news(self, source: str = 'moneycontrol', limit: int = 20) -> list[NewsArticle]:
        """
        Scrape latest market news from a source.

        Args:
            source: News source name
            limit: Max articles to return

        Returns:
            List of NewsArticle objects
        """
        if source not in self.SOURCES:
            logger.error(f"Unknown source: {source}")
            return []

        source_config = self.SOURCES[source]
        url = source_config.get('news_url') or source_config.get('markets_url')

        if not url:
            return []

        logger.info(f"Scraping market news from {source}...")

        # Extract structured news
        result = self.extract_structured(
            url=url,
            schema=self.NEWS_EXTRACTION_SCHEMA,
            prompt=f"Extract the latest {limit} stock market news articles. Include title, summary, mentioned stocks (NSE symbols), sentiment, and any key numbers mentioned."
        )

        if 'error' in result:
            logger.error(f"Failed to scrape {source}: {result['error']}")
            return []

        articles = []
        extracted = result.get('data', {}).get('extract', {}).get('articles', [])

        for item in extracted[:limit]:
            try:
                article = NewsArticle(
                    id=None,
                    source=source,
                    url=item.get('url', url),
                    title=item.get('title', 'No title'),
                    description=item.get('summary', ''),
                    content=item.get('summary', ''),
                    published_at=self._parse_time(item.get('published_time')),
                    tickers=item.get('stocks_mentioned', []),
                    sentiment=Sentiment(item.get('sentiment', 'neutral')),
                    event_type=EventType(item.get('event_type', 'other')),
                    key_claims=item.get('key_numbers', []),
                    relevance_score=0.7
                )
                articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")

        logger.info(f"Scraped {len(articles)} articles from {source}")
        return articles

    def scrape_stock_news(self, symbol: str, sources: list = None) -> list[NewsArticle]:
        """
        Scrape news for a specific stock from multiple sources.

        Args:
            symbol: Stock symbol (NSE)
            sources: List of sources to scrape (default: all)

        Returns:
            List of NewsArticle objects
        """
        sources = sources or ['moneycontrol', 'economictimes']
        all_articles = []

        for source in sources:
            try:
                # Search for stock-specific news
                search_url = f"https://www.google.com/search?q={symbol}+stock+news+site:{self.SOURCES.get(source, {}).get('base_url', '')}"

                # Or use direct stock page if available
                articles = self._scrape_stock_page(symbol, source)
                all_articles.extend(articles)

            except Exception as e:
                logger.warning(f"Failed to scrape {source} for {symbol}: {e}")

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article.title not in seen_titles:
                seen_titles.add(article.title)
                unique_articles.append(article)

        return unique_articles

    def _scrape_stock_page(self, symbol: str, source: str) -> list[NewsArticle]:
        """Scrape stock-specific page from a source."""

        # Build stock-specific URL based on source
        if source == 'moneycontrol':
            # MoneyControl uses lowercase symbol in URL
            url = f"https://www.moneycontrol.com/company-article/{symbol.lower()}/news/{symbol.lower()}"
        elif source == 'economictimes':
            url = f"https://economictimes.indiatimes.com/topic/{symbol}"
        else:
            return []

        result = self.extract_structured(
            url=url,
            schema=self.NEWS_EXTRACTION_SCHEMA,
            prompt=f"Extract all news articles about {symbol} stock. Include title, summary, sentiment, and key numbers."
        )

        if 'error' in result:
            return []

        articles = []
        extracted = result.get('data', {}).get('extract', {}).get('articles', [])

        for item in extracted:
            try:
                article = NewsArticle(
                    id=None,
                    source=source,
                    url=item.get('url', url),
                    title=item.get('title', ''),
                    description=item.get('summary', ''),
                    content=item.get('summary', ''),
                    published_at=self._parse_time(item.get('published_time')),
                    tickers=[symbol],
                    sentiment=Sentiment(item.get('sentiment', 'neutral')),
                    event_type=EventType(item.get('event_type', 'other')),
                    key_claims=item.get('key_numbers', []),
                    relevance_score=0.8
                )
                articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")

        return articles

    def scrape_earnings_calendar(self) -> list[dict]:
        """
        Scrape upcoming earnings calendar.

        Returns:
            List of upcoming earnings with dates
        """
        url = "https://www.moneycontrol.com/markets/earnings/"

        schema = {
            "type": "object",
            "properties": {
                "earnings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company": {"type": "string"},
                            "symbol": {"type": "string"},
                            "date": {"type": "string"},
                            "quarter": {"type": "string"}
                        }
                    }
                }
            }
        }

        result = self.extract_structured(
            url=url,
            schema=schema,
            prompt="Extract the upcoming earnings announcements. Include company name, NSE symbol, announcement date, and quarter."
        )

        if 'error' in result:
            return []

        return result.get('data', {}).get('extract', {}).get('earnings', [])

    def scrape_fii_dii_data(self) -> dict:
        """
        Scrape FII/DII activity data.

        Returns:
            Dict with FII/DII buy/sell data
        """
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"

        schema = {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "fii": {
                    "type": "object",
                    "properties": {
                        "buy_value": {"type": "number"},
                        "sell_value": {"type": "number"},
                        "net_value": {"type": "number"}
                    }
                },
                "dii": {
                    "type": "object",
                    "properties": {
                        "buy_value": {"type": "number"},
                        "sell_value": {"type": "number"},
                        "net_value": {"type": "number"}
                    }
                }
            }
        }

        result = self.extract_structured(
            url=url,
            schema=schema,
            prompt="Extract today's FII and DII activity. Get buy value, sell value, and net value in crores for both FII and DII."
        )

        if 'error' in result:
            return {}

        return result.get('data', {}).get('extract', {})

    def scrape_top_gainers_losers(self) -> dict:
        """
        Scrape top gainers and losers.

        Returns:
            Dict with gainers and losers lists
        """
        url = "https://www.moneycontrol.com/stocks/marketstats/nsegainer/index.php"

        schema = {
            "type": "object",
            "properties": {
                "gainers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "company": {"type": "string"},
                            "price": {"type": "number"},
                            "change_pct": {"type": "number"}
                        }
                    }
                },
                "losers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "company": {"type": "string"},
                            "price": {"type": "number"},
                            "change_pct": {"type": "number"}
                        }
                    }
                }
            }
        }

        result = self.extract_structured(
            url=url,
            schema=schema,
            prompt="Extract top 10 gainers and top 10 losers. Include NSE symbol, company name, current price, and percentage change."
        )

        if 'error' in result:
            return {'gainers': [], 'losers': []}

        return result.get('data', {}).get('extract', {})

    def _parse_time(self, time_str: Optional[str]) -> datetime:
        """Parse time string to datetime."""
        if not time_str:
            return datetime.utcnow()

        # Try common formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%d %b %Y, %H:%M',
            '%d %B %Y',
            '%Y-%m-%d',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        # If "ago" format (e.g., "2 hours ago")
        if 'ago' in time_str.lower():
            return datetime.utcnow()

        return datetime.utcnow()

    def get_all_market_news(self, limit_per_source: int = 10) -> list[NewsArticle]:
        """
        Get news from all configured sources.

        Args:
            limit_per_source: Max articles per source

        Returns:
            Combined list of articles from all sources
        """
        all_articles = []

        for source in self.SOURCES.keys():
            try:
                articles = self.scrape_market_news(source, limit=limit_per_source)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to scrape {source}: {e}")

        # Sort by recency
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        # Deduplicate
        seen_titles = set()
        unique = []
        for article in all_articles:
            if article.title not in seen_titles:
                seen_titles.add(article.title)
                unique.append(article)

        return unique
