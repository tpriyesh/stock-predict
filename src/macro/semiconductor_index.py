"""
Semiconductor Index Fetcher

Fetches and analyzes semiconductor index (SOX) that impacts IT sector:
- PHLX Semiconductor Index (SOX) - Global chip demand indicator
- Affects IT hardware, software (through capex), auto (chips)

All data from Yahoo Finance (free).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger


@dataclass
class SemiconductorData:
    """Semiconductor index data."""
    symbol: str
    name: str
    price: float
    previous_close: float
    change_1d: float           # 1-day change %
    change_5d: float           # 5-day change %
    change_20d: float          # 20-day change %
    high_52w: float
    low_52w: float
    position_in_range: float   # 0-1
    trend: str                 # 'up', 'down', 'sideways'
    momentum: str
    relative_strength: float   # vs S&P 500
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SemiconductorSignal:
    """Semiconductor-based trading signal."""
    signal: float              # -1 to +1
    confidence: float          # 0 to 1
    direction: str             # 'bullish', 'bearish', 'neutral'
    tech_capex_outlook: str    # 'expanding', 'stable', 'contracting'
    affected_sectors: list
    reasoning: str


class SemiconductorFetcher:
    """
    Fetches semiconductor index from Yahoo Finance.

    SOX is a leading indicator for:
    - IT sector health
    - Tech capex cycle
    - Auto chip supply
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize semiconductor fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache: Optional[Tuple[datetime, SemiconductorData]] = None
        self._sp500_cache: Optional[Tuple[datetime, float]] = None

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache is None:
            return False
        cache_time, _ = self._cache
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl

    def _fetch_sp500_change(self, period: str = '1mo') -> float:
        """Fetch S&P 500 change for relative comparison."""
        try:
            ticker = yf.Ticker('^GSPC')
            hist = ticker.history(period=period)
            if len(hist) >= 2:
                return (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        except:
            pass
        return 0.0

    def fetch_semiconductor_index(self, period: str = '3mo') -> Optional[SemiconductorData]:
        """
        Fetch semiconductor index data.

        Args:
            period: Data period

        Returns:
            SemiconductorData or None if fetch fails
        """
        # Check cache
        if self._is_cache_valid():
            return self._cache[1]

        try:
            ticker = yf.Ticker('^SOX')
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 5:
                logger.warning("Insufficient SOX data")
                return None

            # Current and previous
            price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]

            # Calculate changes
            change_1d = (price / previous_close - 1) * 100

            if len(hist) >= 5:
                change_5d = (price / hist['Close'].iloc[-5] - 1) * 100
            else:
                change_5d = change_1d

            if len(hist) >= 20:
                change_20d = (price / hist['Close'].iloc[-20] - 1) * 100
            else:
                change_20d = change_5d

            # 52-week range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()

            if high_52w > low_52w:
                position_in_range = (price - low_52w) / (high_52w - low_52w)
            else:
                position_in_range = 0.5

            # Trend
            if change_20d > 5:
                trend = 'up'
            elif change_20d < -5:
                trend = 'down'
            else:
                trend = 'sideways'

            # Momentum
            if change_5d > 5 and change_1d > 1:
                momentum = 'strong_up'
            elif change_5d > 2:
                momentum = 'up'
            elif change_5d < -5 and change_1d < -1:
                momentum = 'strong_down'
            elif change_5d < -2:
                momentum = 'down'
            else:
                momentum = 'neutral'

            # Relative strength vs S&P 500
            sp500_change = self._fetch_sp500_change('1mo')
            sox_monthly = change_20d if len(hist) >= 20 else change_5d
            relative_strength = sox_monthly - sp500_change

            data = SemiconductorData(
                symbol='^SOX',
                name='PHLX Semiconductor Index',
                price=float(price),
                previous_close=float(previous_close),
                change_1d=float(change_1d),
                change_5d=float(change_5d),
                change_20d=float(change_20d),
                high_52w=float(high_52w),
                low_52w=float(low_52w),
                position_in_range=float(position_in_range),
                trend=trend,
                momentum=momentum,
                relative_strength=float(relative_strength)
            )

            # Cache result
            self._cache = (datetime.now(), data)

            logger.info(f"Fetched SOX: {price:.2f} ({change_1d:+.2f}%)")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch SOX: {e}")
            return None

    def get_semiconductor_signal(self) -> Optional[SemiconductorSignal]:
        """
        Get trading signal based on semiconductor index.

        Returns:
            SemiconductorSignal
        """
        data = self.fetch_semiconductor_index()
        if not data:
            return None

        # Calculate signal
        momentum_score = {
            'strong_up': 0.8,
            'up': 0.4,
            'neutral': 0.0,
            'down': -0.4,
            'strong_down': -0.8
        }.get(data.momentum, 0.0)

        # Adjust for relative strength
        if data.relative_strength > 5:
            momentum_score += 0.2
        elif data.relative_strength < -5:
            momentum_score -= 0.2

        signal = np.clip(momentum_score, -1, 1)
        confidence = abs(momentum_score) * 0.5 + 0.4

        # Direction
        if signal > 0.2:
            direction = 'bullish'
            tech_capex_outlook = 'expanding'
        elif signal < -0.2:
            direction = 'bearish'
            tech_capex_outlook = 'contracting'
        else:
            direction = 'neutral'
            tech_capex_outlook = 'stable'

        # Affected sectors
        affected = ['IT']
        if abs(signal) > 0.4:
            affected.append('Auto')  # Chip supply affects auto

        # Reasoning
        reasoning = f"SOX at {data.price:.0f} ({data.change_1d:+.2f}% today, {data.change_5d:+.2f}% 5-day). "
        reasoning += f"Trend: {data.trend}, Momentum: {data.momentum}. "
        reasoning += f"Relative strength vs S&P: {data.relative_strength:+.1f}%. "

        if direction == 'bullish':
            reasoning += "Strong chip demand = healthy tech capex cycle. Positive for IT sector."
        elif direction == 'bearish':
            reasoning += "Weak chip demand = slowing tech spending. Cautious on IT sector."

        return SemiconductorSignal(
            signal=signal,
            confidence=confidence,
            direction=direction,
            tech_capex_outlook=tech_capex_outlook,
            affected_sectors=affected,
            reasoning=reasoning
        )

    def get_it_sector_impact(self) -> float:
        """
        Get semiconductor impact on IT sector.

        Returns:
            Impact score (-1 to +1)
        """
        signal = self.get_semiconductor_signal()
        if not signal:
            return 0.0
        return signal.signal * 0.5  # 50% weight in IT impact


# Singleton instance
_fetcher_instance: Optional[SemiconductorFetcher] = None


def get_semiconductor_fetcher() -> SemiconductorFetcher:
    """Get singleton semiconductor fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = SemiconductorFetcher()
    return _fetcher_instance
