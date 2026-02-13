"""
Provider package — plugin-based API integrations.

Exposes factory functions for getting singletons:
    from providers import get_quota_manager, get_response_cache
    from providers.news import get_news_providers
"""
from providers.quota import get_quota_manager, APIQuotaManager
from providers.cache import get_response_cache, ResponseCache

__all__ = [
    "get_quota_manager",
    "get_response_cache",
    "APIQuotaManager",
    "ResponseCache",
]
