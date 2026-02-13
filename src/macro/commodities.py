"""
Commodity Price Fetcher

Fetches and analyzes commodity prices that impact Indian stocks:
- Crude Oil (Brent, WTI) - Affects Oil_Gas, Chemicals, FMCG, Auto
- Gold - Affects Banking, Metals, Consumer
- Silver - Affects Metals
- Copper - Affects Metals, Auto, Infra
- Natural Gas - Affects Power, Chemicals
- Steel (POSCO proxy) - Affects Metals, Auto, Infra

All data from Yahoo Finance (free).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from loguru import logger
import json
from pathlib import Path


@dataclass
class CommodityData:
    """Single commodity data point."""
    symbol: str
    name: str
    current_price: float
    previous_close: float
    change_1d: float           # 1-day change %
    change_5d: float           # 5-day change %
    change_20d: float          # 20-day change %
    high_52w: float
    low_52w: float
    position_in_range: float   # 0-1, where in 52w range
    trend: str                 # 'up', 'down', 'sideways'
    momentum: str              # 'strong_up', 'up', 'neutral', 'down', 'strong_down'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommoditySignal:
    """Commodity-based trading signal."""
    commodity: str
    signal: float              # -1 to +1 (negative = bearish for commodity-linked stocks)
    confidence: float          # 0 to 1
    direction: str             # 'bullish', 'bearish', 'neutral'
    affected_sectors: List[str]
    reasoning: str


class CommodityFetcher:
    """
    Fetches commodity prices from Yahoo Finance.

    Provides momentum signals for sector impact analysis.
    """

    COMMODITIES = {
        'crude_brent': {'symbol': 'BZ=F', 'name': 'Brent Crude Oil'},
        'crude_wti': {'symbol': 'CL=F', 'name': 'WTI Crude Oil'},
        'gold': {'symbol': 'GC=F', 'name': 'Gold Futures'},
        'silver': {'symbol': 'SI=F', 'name': 'Silver Futures'},
        'copper': {'symbol': 'HG=F', 'name': 'Copper Futures'},
        'natural_gas': {'symbol': 'NG=F', 'name': 'Natural Gas'},
        'steel_proxy': {'symbol': 'PKX', 'name': 'POSCO (Steel Proxy)'},
    }

    # Sector impact mapping
    SECTOR_IMPACT = {
        'crude_brent': {
            'Oil_Gas': 'positive',      # Higher oil = higher profits for upstream
            'Chemicals': 'negative',    # Higher feedstock costs
            'FMCG': 'negative',         # Higher packaging/logistics costs
            'Auto': 'negative',         # Reduces car demand
            'Power': 'negative',        # Higher fuel costs
        },
        'crude_wti': {
            'Oil_Gas': 'positive',
            'Chemicals': 'negative',
        },
        'gold': {
            'Metals': 'positive',       # Gold mining companies benefit
            'Banking': 'negative',      # Safe haven competes with deposits
            'Consumer': 'negative',     # Competes for discretionary spending
        },
        'silver': {
            'Metals': 'positive',
        },
        'copper': {
            'Metals': 'positive',       # Copper producers benefit
            'Auto': 'negative',         # Input cost for EVs
            'Infra': 'negative',        # Electrical infrastructure costs
        },
        'natural_gas': {
            'Power': 'negative',        # Gas-based power plants
            'Chemicals': 'negative',    # Fertilizer feedstock
        },
        'steel_proxy': {
            'Metals': 'positive',
            'Auto': 'negative',         # Major input cost
            'Infra': 'negative',        # Construction costs
        }
    }

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize commodity fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[datetime, CommodityData]] = {}
        self._load_sector_config()

    def _load_sector_config(self):
        """Load sector dependencies config if available."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'sector_dependencies.json'
        try:
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
                logger.info("Loaded sector dependencies config")
            else:
                self.config = {}
        except Exception as e:
            logger.warning(f"Could not load sector config: {e}")
            self.config = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cache_time, _ = self._cache[key]
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl

    def fetch_commodity(self, commodity_key: str, period: str = '3mo') -> Optional[CommodityData]:
        """
        Fetch single commodity data.

        Args:
            commodity_key: Key from COMMODITIES dict (e.g., 'crude_brent')
            period: Data period (default 3 months for momentum calculation)

        Returns:
            CommodityData or None if fetch fails
        """
        # Check cache
        if self._is_cache_valid(commodity_key):
            return self._cache[commodity_key][1]

        if commodity_key not in self.COMMODITIES:
            logger.warning(f"Unknown commodity: {commodity_key}")
            return None

        info = self.COMMODITIES[commodity_key]
        symbol = info['symbol']
        name = info['name']

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 5:
                logger.warning(f"Insufficient data for {commodity_key}")
                return None

            # Current and previous
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]

            # Calculate changes
            change_1d = (current_price / previous_close - 1) * 100

            if len(hist) >= 5:
                change_5d = (current_price / hist['Close'].iloc[-5] - 1) * 100
            else:
                change_5d = change_1d

            if len(hist) >= 20:
                change_20d = (current_price / hist['Close'].iloc[-20] - 1) * 100
            else:
                change_20d = change_5d

            # 52-week range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()

            if high_52w > low_52w:
                position_in_range = (current_price - low_52w) / (high_52w - low_52w)
            else:
                position_in_range = 0.5

            # Determine trend
            if change_20d > 5:
                trend = 'up'
            elif change_20d < -5:
                trend = 'down'
            else:
                trend = 'sideways'

            # Determine momentum
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

            data = CommodityData(
                symbol=symbol,
                name=name,
                current_price=float(current_price),
                previous_close=float(previous_close),
                change_1d=float(change_1d),
                change_5d=float(change_5d),
                change_20d=float(change_20d),
                high_52w=float(high_52w),
                low_52w=float(low_52w),
                position_in_range=float(position_in_range),
                trend=trend,
                momentum=momentum
            )

            # Cache result
            self._cache[commodity_key] = (datetime.now(), data)

            logger.info(f"Fetched {commodity_key}: ${current_price:.2f} ({change_1d:+.2f}%)")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch {commodity_key}: {e}")
            return None

    def fetch_all(self) -> Dict[str, CommodityData]:
        """
        Fetch all commodities.

        Returns:
            Dictionary of commodity_key -> CommodityData
        """
        results = {}
        for key in self.COMMODITIES:
            data = self.fetch_commodity(key)
            if data:
                results[key] = data
        return results

    def get_commodity_signal(self, commodity_key: str) -> Optional[CommoditySignal]:
        """
        Get trading signal based on commodity movement.

        Args:
            commodity_key: Key from COMMODITIES dict

        Returns:
            CommoditySignal with direction and affected sectors
        """
        data = self.fetch_commodity(commodity_key)
        if not data:
            return None

        # Calculate signal strength (-1 to +1)
        # Positive signal = commodity going up
        momentum_score = {
            'strong_up': 0.8,
            'up': 0.4,
            'neutral': 0.0,
            'down': -0.4,
            'strong_down': -0.8
        }.get(data.momentum, 0.0)

        # Adjust based on position in range
        range_adjustment = (data.position_in_range - 0.5) * 0.4

        signal = np.clip(momentum_score + range_adjustment, -1, 1)

        # Confidence based on momentum strength
        confidence = abs(momentum_score) * 0.6 + 0.3

        # Direction
        if signal > 0.2:
            direction = 'bullish'
        elif signal < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Affected sectors
        affected = list(self.SECTOR_IMPACT.get(commodity_key, {}).keys())

        # Reasoning
        reasoning = f"{data.name}: {data.change_1d:+.2f}% today, {data.change_5d:+.2f}% 5-day. "
        reasoning += f"Trend: {data.trend}, Momentum: {data.momentum}. "
        reasoning += f"At {data.position_in_range:.0%} of 52-week range."

        return CommoditySignal(
            commodity=commodity_key,
            signal=signal,
            confidence=confidence,
            direction=direction,
            affected_sectors=affected,
            reasoning=reasoning
        )

    def get_sector_commodity_impact(self, sector: str) -> Dict[str, float]:
        """
        Get commodity impact scores for a specific sector.

        Args:
            sector: Sector name (e.g., 'Oil_Gas', 'Auto')

        Returns:
            Dictionary of commodity -> impact score
        """
        impacts = {}

        for commodity_key, sector_map in self.SECTOR_IMPACT.items():
            if sector not in sector_map:
                continue

            direction = sector_map[sector]
            signal = self.get_commodity_signal(commodity_key)

            if signal is None:
                continue

            # Apply direction
            if direction == 'positive':
                impact = signal.signal  # Commodity up = stock up
            elif direction == 'negative':
                impact = -signal.signal  # Commodity up = stock down
            else:
                impact = 0

            impacts[commodity_key] = impact

        return impacts

    def get_oil_summary(self) -> Dict[str, any]:
        """
        Get summary of oil market for quick reference.

        Returns:
            Dictionary with oil prices and signals
        """
        brent = self.fetch_commodity('crude_brent')
        wti = self.fetch_commodity('crude_wti')

        if not brent and not wti:
            return {'available': False}

        primary = brent or wti

        return {
            'available': True,
            'brent_price': brent.current_price if brent else None,
            'wti_price': wti.current_price if wti else None,
            'change_1d': primary.change_1d,
            'change_5d': primary.change_5d,
            'trend': primary.trend,
            'momentum': primary.momentum,
            'signal': 'bullish' if primary.momentum in ['strong_up', 'up'] else
                     'bearish' if primary.momentum in ['strong_down', 'down'] else 'neutral'
        }

    def get_metals_summary(self) -> Dict[str, any]:
        """
        Get summary of metals market.

        Returns:
            Dictionary with metal prices and signals
        """
        gold = self.fetch_commodity('gold')
        copper = self.fetch_commodity('copper')
        steel = self.fetch_commodity('steel_proxy')

        metals = [m for m in [gold, copper, steel] if m]

        if not metals:
            return {'available': False}

        avg_change = np.mean([m.change_1d for m in metals])

        return {
            'available': True,
            'gold_price': gold.current_price if gold else None,
            'copper_price': copper.current_price if copper else None,
            'steel_price': steel.current_price if steel else None,
            'avg_change_1d': avg_change,
            'gold_trend': gold.trend if gold else None,
            'copper_trend': copper.trend if copper else None,
            'overall_signal': 'bullish' if avg_change > 1 else 'bearish' if avg_change < -1 else 'neutral'
        }


# Singleton instance
_fetcher_instance: Optional[CommodityFetcher] = None


def get_commodity_fetcher() -> CommodityFetcher:
    """Get singleton commodity fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = CommodityFetcher()
    return _fetcher_instance
