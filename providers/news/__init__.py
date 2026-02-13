"""
News provider plugins.

Discovers and registers available news providers based on API keys.
Priority order is configurable via NEWS_PROVIDERS env var.
"""
import os
from typing import List

from loguru import logger

from providers.base import BaseNewsProvider


def get_news_providers() -> List[BaseNewsProvider]:
    """Discover and return available news providers in priority order.

    Reads NEWS_PROVIDERS env var for ordering. Default: newsapi,gnews,finnhub,rss
    Only returns providers that have valid credentials (is_available() == True).
    """
    from providers.news.newsapi import NewsAPIProvider
    from providers.news.gnews import GNewsProvider
    from providers.news.rss import RSSProvider
    from providers.news.finnhub import FinnhubNewsProvider

    # Registry: name -> class
    registry = {
        "newsapi": NewsAPIProvider,
        "gnews": GNewsProvider,
        "finnhub": FinnhubNewsProvider,
        "rss": RSSProvider,
    }

    # Read priority order from env
    order_str = os.getenv("NEWS_PROVIDERS", "newsapi,gnews,finnhub,rss")
    ordered_names = [s.strip().lower() for s in order_str.split(",") if s.strip()]

    providers = []
    for name in ordered_names:
        cls = registry.get(name)
        if cls is None:
            logger.warning(f"Unknown news provider '{name}', skipping")
            continue
        try:
            provider = cls()
            if provider.is_available():
                providers.append(provider)
                logger.info(f"News provider '{name}' registered (available)")
            else:
                logger.info(f"News provider '{name}' skipped (no credentials)")
        except Exception as e:
            logger.error(f"Failed to init news provider '{name}': {e}")

    if not providers:
        logger.warning("No news providers available! Check API keys.")

    return providers
