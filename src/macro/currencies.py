"""
Currency Rate Fetcher

Fetches and analyzes currency rates that impact Indian stocks:
- USD/INR - Critical for IT, Pharma, all exporters
- DXY (Dollar Index) - Global dollar strength
- EUR/USD - European trade indicator
- GBP/USD - UK trade indicator

All data from Yahoo Finance (free).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class CurrencyData:
    """Single currency pair data."""
    symbol: str
    name: str
    rate: float
    previous_close: float
    change_1d: float           # 1-day change %
    change_5d: float           # 5-day change %
    change_20d: float          # 20-day change %
    change_ytd: float          # Year-to-date change %
    high_52w: float
    low_52w: float
    volatility_20d: float      # 20-day volatility
    trend: str                 # 'strengthening', 'weakening', 'stable'
    momentum: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CurrencySignal:
    """Currency-based trading signal for Indian stocks."""
    currency_pair: str
    signal: float              # -1 to +1
    confidence: float          # 0 to 1
    inr_direction: str         # 'strengthening', 'weakening', 'stable'
    exporter_impact: str       # 'positive', 'negative', 'neutral'
    importer_impact: str       # 'positive', 'negative', 'neutral'
    affected_sectors: List[str]
    reasoning: str


class CurrencyFetcher:
    """
    Fetches currency rates from Yahoo Finance.

    Provides signals for exporter/importer impact analysis.
    """

    CURRENCIES = {
        'usdinr': {'symbol': 'USDINR=X', 'name': 'USD/INR'},
        'dxy': {'symbol': 'DX-Y.NYB', 'name': 'US Dollar Index'},
        'eurusd': {'symbol': 'EURUSD=X', 'name': 'EUR/USD'},
        'gbpusd': {'symbol': 'GBPUSD=X', 'name': 'GBP/USD'},
        'usdjpy': {'symbol': 'USDJPY=X', 'name': 'USD/JPY'},
    }

    # Sector impact - focus on USD/INR
    EXPORTER_SECTORS = ['IT', 'Pharma', 'Chemicals', 'Metals']  # Benefit from weak INR
    IMPORTER_SECTORS = ['Oil_Gas', 'Auto', 'Power', 'Telecom']  # Hurt by weak INR

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize currency fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[datetime, CurrencyData]] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cache_time, _ = self._cache[key]
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl

    def fetch_currency(self, currency_key: str, period: str = '3mo') -> Optional[CurrencyData]:
        """
        Fetch single currency pair data.

        Args:
            currency_key: Key from CURRENCIES dict (e.g., 'usdinr')
            period: Data period

        Returns:
            CurrencyData or None if fetch fails
        """
        # Check cache
        if self._is_cache_valid(currency_key):
            return self._cache[currency_key][1]

        if currency_key not in self.CURRENCIES:
            logger.warning(f"Unknown currency: {currency_key}")
            return None

        info = self.CURRENCIES[currency_key]
        symbol = info['symbol']
        name = info['name']

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 5:
                logger.warning(f"Insufficient data for {currency_key}")
                return None

            # Current and previous
            rate = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]

            # Calculate changes
            change_1d = (rate / previous_close - 1) * 100

            if len(hist) >= 5:
                change_5d = (rate / hist['Close'].iloc[-5] - 1) * 100
            else:
                change_5d = change_1d

            if len(hist) >= 20:
                change_20d = (rate / hist['Close'].iloc[-20] - 1) * 100
                # 20-day volatility
                returns = hist['Close'].pct_change().dropna()[-20:]
                volatility_20d = returns.std() * np.sqrt(252) * 100  # Annualized
            else:
                change_20d = change_5d
                volatility_20d = 0

            # YTD change (approximate)
            change_ytd = (rate / hist['Close'].iloc[0] - 1) * 100

            # 52-week range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()

            # Determine trend (for USDINR, up = INR weakening)
            if currency_key == 'usdinr':
                if change_20d > 1:
                    trend = 'weakening'  # INR weakening (USD/INR going up)
                elif change_20d < -1:
                    trend = 'strengthening'  # INR strengthening
                else:
                    trend = 'stable'
            else:
                if change_20d > 1:
                    trend = 'strengthening'
                elif change_20d < -1:
                    trend = 'weakening'
                else:
                    trend = 'stable'

            # Momentum
            if change_5d > 2 and change_1d > 0.3:
                momentum = 'strong_up'
            elif change_5d > 0.5:
                momentum = 'up'
            elif change_5d < -2 and change_1d < -0.3:
                momentum = 'strong_down'
            elif change_5d < -0.5:
                momentum = 'down'
            else:
                momentum = 'neutral'

            data = CurrencyData(
                symbol=symbol,
                name=name,
                rate=float(rate),
                previous_close=float(previous_close),
                change_1d=float(change_1d),
                change_5d=float(change_5d),
                change_20d=float(change_20d),
                change_ytd=float(change_ytd),
                high_52w=float(high_52w),
                low_52w=float(low_52w),
                volatility_20d=float(volatility_20d),
                trend=trend,
                momentum=momentum
            )

            # Cache result
            self._cache[currency_key] = (datetime.now(), data)

            logger.info(f"Fetched {currency_key}: {rate:.4f} ({change_1d:+.2f}%)")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch {currency_key}: {e}")
            return None

    def fetch_all(self) -> Dict[str, CurrencyData]:
        """
        Fetch all currency pairs.

        Returns:
            Dictionary of currency_key -> CurrencyData
        """
        results = {}
        for key in self.CURRENCIES:
            data = self.fetch_currency(key)
            if data:
                results[key] = data
        return results

    def get_usdinr_signal(self) -> Optional[CurrencySignal]:
        """
        Get trading signal based on USD/INR movement.

        This is the most important currency for Indian stocks.

        Returns:
            CurrencySignal with exporter/importer impacts
        """
        data = self.fetch_currency('usdinr')
        if not data:
            return None

        # Signal: positive = INR weakening (good for exporters)
        # Negative = INR strengthening (good for importers)
        momentum_score = {
            'strong_up': 0.8,    # INR weakening fast
            'up': 0.4,           # INR weakening
            'neutral': 0.0,
            'down': -0.4,        # INR strengthening
            'strong_down': -0.8  # INR strengthening fast
        }.get(data.momentum, 0.0)

        signal = np.clip(momentum_score, -1, 1)
        confidence = abs(momentum_score) * 0.5 + 0.4

        # INR direction
        if signal > 0.2:
            inr_direction = 'weakening'
            exporter_impact = 'positive'
            importer_impact = 'negative'
        elif signal < -0.2:
            inr_direction = 'strengthening'
            exporter_impact = 'negative'
            importer_impact = 'positive'
        else:
            inr_direction = 'stable'
            exporter_impact = 'neutral'
            importer_impact = 'neutral'

        # Affected sectors
        if inr_direction == 'weakening':
            affected = self.EXPORTER_SECTORS
        elif inr_direction == 'strengthening':
            affected = self.IMPORTER_SECTORS
        else:
            affected = []

        # Reasoning
        reasoning = f"USD/INR at {data.rate:.2f} ({data.change_1d:+.2f}% today, {data.change_5d:+.2f}% 5-day). "
        reasoning += f"INR is {inr_direction}. "
        if inr_direction == 'weakening':
            reasoning += "Benefits: IT, Pharma (higher INR revenue). Hurts: Oil imports, Auto (higher input costs)."
        elif inr_direction == 'strengthening':
            reasoning += "Benefits: Importers (cheaper imports). Hurts: IT, Pharma (lower INR realization)."

        return CurrencySignal(
            currency_pair='usdinr',
            signal=signal,
            confidence=confidence,
            inr_direction=inr_direction,
            exporter_impact=exporter_impact,
            importer_impact=importer_impact,
            affected_sectors=affected,
            reasoning=reasoning
        )

    def get_dxy_signal(self) -> Optional[CurrencySignal]:
        """
        Get signal from Dollar Index (DXY).

        Strong DXY generally negative for emerging markets.

        Returns:
            CurrencySignal
        """
        data = self.fetch_currency('dxy')
        if not data:
            return None

        # Strong dollar = negative for emerging markets
        momentum_score = {
            'strong_up': -0.6,    # Strong dollar = EM outflows
            'up': -0.3,
            'neutral': 0.0,
            'down': 0.3,          # Weak dollar = EM inflows
            'strong_down': 0.6
        }.get(data.momentum, 0.0)

        signal = np.clip(momentum_score, -1, 1)
        confidence = abs(momentum_score) * 0.5 + 0.3

        # Reasoning
        reasoning = f"DXY at {data.rate:.2f} ({data.change_1d:+.2f}% today). "
        if momentum_score < 0:
            reasoning += "Strong dollar = potential FII outflows from Indian markets."
        elif momentum_score > 0:
            reasoning += "Weak dollar = favorable for EM flows into India."

        return CurrencySignal(
            currency_pair='dxy',
            signal=signal,
            confidence=confidence,
            inr_direction='correlated',
            exporter_impact='mixed',
            importer_impact='mixed',
            affected_sectors=['Banking', 'Finance', 'IT'],
            reasoning=reasoning
        )

    def get_sector_currency_impact(self, sector: str) -> float:
        """
        Get currency impact score for a specific sector.

        Args:
            sector: Sector name

        Returns:
            Impact score (-1 to +1)
        """
        usdinr = self.get_usdinr_signal()
        if not usdinr:
            return 0.0

        # Exporters benefit from weak INR
        if sector in self.EXPORTER_SECTORS:
            return usdinr.signal  # Positive when INR weakening

        # Importers benefit from strong INR
        if sector in self.IMPORTER_SECTORS:
            return -usdinr.signal  # Positive when INR strengthening

        # Neutral sectors
        return 0.0

    def get_currency_summary(self) -> Dict[str, any]:
        """
        Get summary of currency markets.

        Returns:
            Dictionary with key currency data and signals
        """
        usdinr = self.fetch_currency('usdinr')
        dxy = self.fetch_currency('dxy')

        if not usdinr:
            return {'available': False}

        usdinr_signal = self.get_usdinr_signal()

        return {
            'available': True,
            'usdinr_rate': usdinr.rate,
            'usdinr_change_1d': usdinr.change_1d,
            'usdinr_change_5d': usdinr.change_5d,
            'usdinr_trend': usdinr.trend,
            'inr_direction': usdinr_signal.inr_direction if usdinr_signal else 'unknown',
            'dxy': dxy.rate if dxy else None,
            'dxy_trend': dxy.trend if dxy else None,
            'exporter_signal': usdinr_signal.exporter_impact if usdinr_signal else 'unknown',
            'importer_signal': usdinr_signal.importer_impact if usdinr_signal else 'unknown',
        }


# Singleton instance
_fetcher_instance: Optional[CurrencyFetcher] = None


def get_currency_fetcher() -> CurrencyFetcher:
    """Get singleton currency fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = CurrencyFetcher()
    return _fetcher_instance
