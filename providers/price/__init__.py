"""
Price provider plugins.
"""
import os
from typing import List

from loguru import logger

from providers.base import BasePriceProvider


def get_price_providers() -> List[BasePriceProvider]:
    """Return available price data providers in priority order."""
    from providers.price.yfinance_provider import YFinanceProvider

    registry = {
        "yfinance": YFinanceProvider,
    }

    order_str = os.getenv("PRICE_PROVIDERS", "yfinance")
    ordered_names = [s.strip().lower() for s in order_str.split(",") if s.strip()]

    providers = []
    for name in ordered_names:
        cls = registry.get(name)
        if cls is None:
            logger.warning(f"Unknown price provider '{name}', skipping")
            continue
        try:
            provider = cls()
            if provider.is_available():
                providers.append(provider)
                logger.info(f"Price provider '{name}' registered")
        except Exception as e:
            logger.error(f"Failed to init price provider '{name}': {e}")

    return providers
