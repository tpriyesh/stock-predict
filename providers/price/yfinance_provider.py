"""
yfinance price provider plugin.

Free, no API key needed. Rate limit via self-imposed throttle.
Uses existing yfinance_with_timeout() for hang protection.
"""
import os
from datetime import timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from providers.base import BasePriceProvider
from utils.platform import yfinance_with_timeout


class YFinanceProvider(BasePriceProvider):
    """Yahoo Finance via yfinance — free, no key, self-throttled."""

    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"

    def __init__(self, exchange: str = "NSE"):
        self._exchange = exchange
        self._suffix = self.NSE_SUFFIX if exchange == "NSE" else self.BSE_SUFFIX

    @property
    def name(self) -> str:
        return "yfinance"

    @property
    def requests_per_minute(self) -> int:
        return int(os.getenv("YFINANCE_RPM", "60"))

    @property
    def requests_per_second(self) -> int:
        return int(os.getenv("YFINANCE_RPS", "2"))

    def is_available(self) -> bool:
        return True  # No API key required

    def _to_yahoo(self, symbol: str) -> str:
        symbol = symbol.replace("&", "%26")
        if not symbol.endswith(self._suffix):
            return f"{symbol}{self._suffix}"
        return symbol

    def fetch_history(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Fetch OHLCV history via yfinance with timeout protection."""
        yahoo_symbol = self._to_yahoo(symbol)

        _period = period

        def _fetch():
            ticker = yf.Ticker(yahoo_symbol)
            return ticker.history(period=_period)

        try:
            df = yfinance_with_timeout(_fetch, timeout_seconds=30)

            if df is None or df.empty:
                return None

            # Normalize column names to lowercase
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })

            # Keep only OHLCV
            cols = ["open", "high", "low", "close", "volume"]
            available = [c for c in cols if c in df.columns]
            return df[available]

        except Exception as e:
            logger.error(f"yfinance history failed for {yahoo_symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price via yfinance with timeout protection."""
        yahoo_symbol = self._to_yahoo(symbol)

        def _fetch():
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.fast_info
            for attr in ["last_price", "regular_market_price", "previous_close"]:
                if hasattr(info, attr):
                    price = getattr(info, attr)
                    if price and price > 0:
                        return round(float(price), 2)
            # Fallback: last close from history
            df = ticker.history(period="1d")
            if not df.empty:
                return round(float(df["Close"].iloc[-1]), 2)
            return None

        try:
            return yfinance_with_timeout(_fetch, timeout_seconds=15)
        except Exception as e:
            logger.error(f"yfinance current price failed for {yahoo_symbol}: {e}")
            return None
