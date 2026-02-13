"""
Abstract Base Classes for all data providers.

Pattern mirrors broker/base.py — each provider type has a minimal contract.
Implement one ABC, register it, and the system handles rate limiting,
caching, quota tracking, and failover automatically.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Article:
    """Normalized news article from any provider."""
    id: str
    title: str
    description: str
    content: str
    url: str
    source: str
    published_at: datetime
    provider: str = ""  # Which provider returned this


class BaseNewsProvider(ABC):
    """Plugin interface for news data sources.

    Implement this to add a new news API. The system will:
    - Rate-limit calls via APIQuotaManager
    - Cache responses via ResponseCache
    - Rotate to next provider on failure/quota exhaustion
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g., 'newsapi', 'gnews')."""
        ...

    @abstractmethod
    def fetch_articles(self, query: str, hours: int = 72,
                       max_results: int = 50) -> List[Article]:
        """Fetch articles matching query from last `hours` hours.

        Should NOT handle rate limiting or caching — the aggregator does that.
        Should raise on HTTP errors (aggregator catches and records failure).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if provider has valid credentials and is configured."""
        ...

    @property
    @abstractmethod
    def daily_limit(self) -> Optional[int]:
        """Max requests per day. None = unlimited."""
        ...

    @property
    @abstractmethod
    def requests_per_minute(self) -> int:
        """Max requests per minute."""
        ...

    @property
    def requests_per_second(self) -> int:
        """Max requests per second. Default: no burst limit."""
        return self.requests_per_minute  # No sub-second throttle by default


class BasePriceProvider(ABC):
    """Plugin interface for price/market data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fetch_history(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Fetch OHLCV history. Returns DataFrame with columns:
        open, high, low, close, volume (lowercase). Index = datetime.
        Returns None on failure.
        """
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price. Returns None on failure."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    def requests_per_minute(self) -> int:
        return 60

    @property
    def requests_per_second(self) -> int:
        return 2


class BaseLLMProvider(ABC):
    """Plugin interface for LLM services (OpenAI, Ollama, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = 0.1,
                 max_tokens: int = 500) -> Optional[str]:
        """Chat completion. Returns response text or None on failure."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> float:
        """Approximate cost per 1K tokens (input+output blended). 0 = free."""
        ...

    @property
    def requests_per_minute(self) -> int:
        return 60
