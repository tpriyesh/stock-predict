"""
Bond Yield Fetcher

Fetches and analyzes bond yields that impact Indian stocks:
- US 10Y Treasury - Global risk-free rate, affects all equities
- US 2Y Treasury - Fed policy indicator
- US 30Y Treasury - Long-term growth expectations
- Yield curve analysis (2Y-10Y spread)

Higher yields generally negative for growth stocks.
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
class BondData:
    """Single bond yield data."""
    symbol: str
    name: str
    yield_pct: float           # Current yield in %
    previous_close: float
    change_1d_bps: float       # 1-day change in basis points
    change_5d_bps: float       # 5-day change in bps
    change_20d_bps: float      # 20-day change in bps
    high_52w: float
    low_52w: float
    trend: str                 # 'rising', 'falling', 'stable'
    level: str                 # 'high', 'normal', 'low'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BondSignal:
    """Bond-based trading signal."""
    bond: str
    signal: float              # -1 to +1 (negative = bearish for equities)
    confidence: float          # 0 to 1
    yield_direction: str       # 'rising', 'falling', 'stable'
    equity_impact: str         # 'negative', 'positive', 'neutral'
    growth_stock_impact: str   # More sensitive to yields
    affected_sectors: List[str]
    reasoning: str


@dataclass
class YieldCurve:
    """Yield curve analysis."""
    spread_2y10y: float        # 10Y - 2Y spread in bps
    is_inverted: bool          # Spread < 0
    curve_signal: str          # 'normal', 'flat', 'inverted'
    recession_risk: str        # 'low', 'moderate', 'high'
    reasoning: str


class BondFetcher:
    """
    Fetches bond yields from Yahoo Finance.

    Provides signals for equity market impact analysis.
    """

    BONDS = {
        'us_10y': {'symbol': '^TNX', 'name': 'US 10Y Treasury Yield'},
        'us_2y': {'symbol': '^IRX', 'name': 'US 2Y Treasury Yield'},
        'us_30y': {'symbol': '^TYX', 'name': 'US 30Y Treasury Yield'},
    }

    # Rate-sensitive sectors
    RATE_SENSITIVE = {
        'Banking': 'mixed',     # NIMs can benefit, but loan growth slows
        'Finance': 'negative',  # Higher funding costs
        'IT': 'negative',       # Growth stocks derated
        'Infra': 'negative',    # Higher project financing costs
        'Auto': 'negative',     # Higher loan EMIs reduce demand
        'Consumer': 'negative', # Discretionary spending falls
    }

    def __init__(self, cache_ttl: int = 600):
        """
        Initialize bond fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 10 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[datetime, BondData]] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cache_time, _ = self._cache[key]
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl

    def fetch_bond(self, bond_key: str, period: str = '3mo') -> Optional[BondData]:
        """
        Fetch single bond yield data.

        Args:
            bond_key: Key from BONDS dict (e.g., 'us_10y')
            period: Data period

        Returns:
            BondData or None if fetch fails
        """
        # Check cache
        if self._is_cache_valid(bond_key):
            return self._cache[bond_key][1]

        if bond_key not in self.BONDS:
            logger.warning(f"Unknown bond: {bond_key}")
            return None

        info = self.BONDS[bond_key]
        symbol = info['symbol']
        name = info['name']

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 5:
                logger.warning(f"Insufficient data for {bond_key}")
                return None

            # Current and previous
            yield_pct = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]

            # Calculate changes in basis points (1 bp = 0.01%)
            change_1d_bps = (yield_pct - previous_close) * 100

            if len(hist) >= 5:
                change_5d_bps = (yield_pct - hist['Close'].iloc[-5]) * 100
            else:
                change_5d_bps = change_1d_bps

            if len(hist) >= 20:
                change_20d_bps = (yield_pct - hist['Close'].iloc[-20]) * 100
            else:
                change_20d_bps = change_5d_bps

            # 52-week range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()

            # Determine trend
            if change_20d_bps > 25:
                trend = 'rising'
            elif change_20d_bps < -25:
                trend = 'falling'
            else:
                trend = 'stable'

            # Determine level (for US 10Y)
            if bond_key == 'us_10y':
                if yield_pct > 4.5:
                    level = 'high'
                elif yield_pct < 3.5:
                    level = 'low'
                else:
                    level = 'normal'
            else:
                level = 'normal'

            data = BondData(
                symbol=symbol,
                name=name,
                yield_pct=float(yield_pct),
                previous_close=float(previous_close),
                change_1d_bps=float(change_1d_bps),
                change_5d_bps=float(change_5d_bps),
                change_20d_bps=float(change_20d_bps),
                high_52w=float(high_52w),
                low_52w=float(low_52w),
                trend=trend,
                level=level
            )

            # Cache result
            self._cache[bond_key] = (datetime.now(), data)

            logger.info(f"Fetched {bond_key}: {yield_pct:.2f}% ({change_1d_bps:+.1f} bps)")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch {bond_key}: {e}")
            return None

    def fetch_all(self) -> Dict[str, BondData]:
        """
        Fetch all bond yields.

        Returns:
            Dictionary of bond_key -> BondData
        """
        results = {}
        for key in self.BONDS:
            data = self.fetch_bond(key)
            if data:
                results[key] = data
        return results

    def get_yield_curve(self) -> Optional[YieldCurve]:
        """
        Analyze yield curve (2Y-10Y spread).

        Inverted yield curve is recession indicator.

        Returns:
            YieldCurve analysis
        """
        us_10y = self.fetch_bond('us_10y')
        us_2y = self.fetch_bond('us_2y')

        if not us_10y or not us_2y:
            return None

        # Spread in basis points
        spread = (us_10y.yield_pct - us_2y.yield_pct) * 100

        is_inverted = spread < 0

        if spread < -50:
            curve_signal = 'inverted'
            recession_risk = 'high'
        elif spread < 0:
            curve_signal = 'inverted'
            recession_risk = 'moderate'
        elif spread < 50:
            curve_signal = 'flat'
            recession_risk = 'moderate'
        else:
            curve_signal = 'normal'
            recession_risk = 'low'

        reasoning = f"2Y-10Y spread: {spread:.0f} bps. "
        if is_inverted:
            reasoning += "Inverted yield curve historically precedes recession by 12-18 months. "
            reasoning += "Suggests Fed may need to cut rates. Risk-off for equities."
        elif spread < 50:
            reasoning += "Flat curve indicates slowing growth expectations."
        else:
            reasoning += "Normal curve indicates healthy growth expectations."

        return YieldCurve(
            spread_2y10y=spread,
            is_inverted=is_inverted,
            curve_signal=curve_signal,
            recession_risk=recession_risk,
            reasoning=reasoning
        )

    def get_10y_signal(self) -> Optional[BondSignal]:
        """
        Get trading signal based on US 10Y Treasury yield.

        This is the most important yield for equity markets.

        Returns:
            BondSignal
        """
        data = self.fetch_bond('us_10y')
        if not data:
            return None

        # Rising yields = negative for equities (especially growth)
        if data.trend == 'rising' and data.change_5d_bps > 10:
            signal = -0.6
            yield_direction = 'rising'
            equity_impact = 'negative'
            growth_stock_impact = 'very_negative'
        elif data.trend == 'rising':
            signal = -0.3
            yield_direction = 'rising'
            equity_impact = 'negative'
            growth_stock_impact = 'negative'
        elif data.trend == 'falling' and data.change_5d_bps < -10:
            signal = 0.6
            yield_direction = 'falling'
            equity_impact = 'positive'
            growth_stock_impact = 'positive'
        elif data.trend == 'falling':
            signal = 0.3
            yield_direction = 'falling'
            equity_impact = 'positive'
            growth_stock_impact = 'positive'
        else:
            signal = 0.0
            yield_direction = 'stable'
            equity_impact = 'neutral'
            growth_stock_impact = 'neutral'

        confidence = min(abs(data.change_5d_bps) / 20, 0.8) + 0.2

        # Affected sectors (growth stocks most affected)
        affected = ['IT', 'Consumer', 'Finance']
        if signal < -0.3:
            affected.extend(['Infra', 'Auto'])

        # Reasoning
        reasoning = f"US 10Y at {data.yield_pct:.2f}% ({data.change_1d_bps:+.1f} bps today, "
        reasoning += f"{data.change_5d_bps:+.1f} bps 5-day). Level: {data.level}. "
        if data.trend == 'rising':
            reasoning += "Rising yields = higher discount rates = lower equity valuations. "
            reasoning += "Growth stocks (IT, Consumer) most affected."
        elif data.trend == 'falling':
            reasoning += "Falling yields = supportive for equity valuations. "
            reasoning += "Growth stocks benefit most."

        return BondSignal(
            bond='us_10y',
            signal=signal,
            confidence=confidence,
            yield_direction=yield_direction,
            equity_impact=equity_impact,
            growth_stock_impact=growth_stock_impact,
            affected_sectors=affected,
            reasoning=reasoning
        )

    def get_sector_bond_impact(self, sector: str) -> float:
        """
        Get bond yield impact score for a specific sector.

        Args:
            sector: Sector name

        Returns:
            Impact score (-1 to +1)
        """
        signal = self.get_10y_signal()
        if not signal:
            return 0.0

        sensitivity = self.RATE_SENSITIVE.get(sector, 'neutral')

        if sensitivity == 'negative':
            return signal.signal  # Already negative when yields rise
        elif sensitivity == 'mixed':
            return signal.signal * 0.5  # Partial impact
        else:
            return 0.0

    def get_bond_summary(self) -> Dict[str, any]:
        """
        Get summary of bond markets.

        Returns:
            Dictionary with key bond data and signals
        """
        us_10y = self.fetch_bond('us_10y')
        curve = self.get_yield_curve()
        signal = self.get_10y_signal()

        if not us_10y:
            return {'available': False}

        return {
            'available': True,
            'us_10y_yield': us_10y.yield_pct,
            'us_10y_change_1d_bps': us_10y.change_1d_bps,
            'us_10y_change_5d_bps': us_10y.change_5d_bps,
            'us_10y_trend': us_10y.trend,
            'us_10y_level': us_10y.level,
            'yield_curve_spread': curve.spread_2y10y if curve else None,
            'yield_curve_signal': curve.curve_signal if curve else None,
            'recession_risk': curve.recession_risk if curve else None,
            'equity_impact': signal.equity_impact if signal else 'unknown',
            'growth_stock_impact': signal.growth_stock_impact if signal else 'unknown',
        }


# Singleton instance
_fetcher_instance: Optional[BondFetcher] = None


def get_bond_fetcher() -> BondFetcher:
    """Get singleton bond fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = BondFetcher()
    return _fetcher_instance
