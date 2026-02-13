"""
RegimeAwarePositionSizer - Dynamic Position Sizing Based on Market Conditions

CRITICAL: Fixed position sizing ignores market context.
During high volatility or bear markets, position sizes should shrink.

This module provides:
1. Market regime detection (bull/bear/sideways/crisis)
2. Volatility-adjusted position sizing
3. Correlation-aware portfolio allocation
4. Drawdown-triggered risk reduction
5. Event-aware sizing (earnings, F&O expiry, etc.)

Position size = base_size * regime_multiplier * vol_adjustment * correlation_penalty * event_factor
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"     # Strong uptrend, low vol
    BULL = "bull"                   # Uptrend
    NEUTRAL = "neutral"             # Range-bound
    BEAR = "bear"                   # Downtrend
    STRONG_BEAR = "strong_bear"     # Strong downtrend, high vol
    CRISIS = "crisis"               # Extreme volatility, correlation spike


@dataclass
class RegimeMetrics:
    """Metrics used to determine market regime."""
    trend_score: float          # -1 (bearish) to +1 (bullish)
    volatility_percentile: float  # Current vol vs historical (0-100)
    correlation_regime: float   # Average correlation (higher = risk-off)
    breadth_score: float        # Market breadth (-1 to +1)
    momentum_score: float       # Multi-timeframe momentum
    vix_equivalent: float       # India VIX or proxy


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    base_size_pct: float        # Starting position size
    regime_multiplier: float    # Regime adjustment
    volatility_adjustment: float
    correlation_penalty: float
    event_factor: float
    drawdown_factor: float
    final_size_pct: float       # Final recommended size
    max_position_pct: float     # Absolute cap
    regime: MarketRegime
    regime_metrics: RegimeMetrics
    reasons: List[str]


class MarketRegimeDetector:
    """
    Detects current market regime from index data.

    Uses multiple signals:
    1. Price vs moving averages
    2. Volatility level
    3. Market breadth
    4. Cross-asset correlations
    """

    def __init__(self):
        # Regime multipliers for position sizing
        self.REGIME_MULTIPLIERS = {
            MarketRegime.STRONG_BULL: 1.2,
            MarketRegime.BULL: 1.0,
            MarketRegime.NEUTRAL: 0.8,
            MarketRegime.BEAR: 0.6,
            MarketRegime.STRONG_BEAR: 0.4,
            MarketRegime.CRISIS: 0.2
        }

    def detect_regime(self,
                       index_data: pd.DataFrame,
                       vix_data: Optional[pd.DataFrame] = None) -> Tuple[MarketRegime, RegimeMetrics]:
        """
        Detect current market regime.

        Args:
            index_data: DataFrame with OHLCV for market index (e.g., NIFTY50)
            vix_data: Optional VIX data

        Returns:
            Tuple of (MarketRegime, RegimeMetrics)
        """
        if index_data.empty or len(index_data) < 200:
            return MarketRegime.NEUTRAL, self._default_metrics()

        close = index_data['close']

        # 1. Trend Score
        trend_score = self._calculate_trend_score(close)

        # 2. Volatility Percentile
        vol_percentile = self._calculate_volatility_percentile(close)

        # 3. Momentum
        momentum_score = self._calculate_momentum(close)

        # 4. Market Breadth (if available) - using proxy
        breadth_score = self._estimate_breadth(index_data)

        # 5. Correlation regime
        correlation = self._estimate_correlation_regime(index_data)

        # 6. VIX or equivalent
        if vix_data is not None and not vix_data.empty:
            vix_value = float(vix_data['close'].iloc[-1])
        else:
            # Estimate from realized volatility
            returns = close.pct_change().dropna()
            realized_vol = returns.tail(20).std() * np.sqrt(252) * 100
            vix_value = realized_vol

        metrics = RegimeMetrics(
            trend_score=trend_score,
            volatility_percentile=vol_percentile,
            correlation_regime=correlation,
            breadth_score=breadth_score,
            momentum_score=momentum_score,
            vix_equivalent=vix_value
        )

        # Determine regime
        regime = self._classify_regime(metrics)

        return regime, metrics

    def _calculate_trend_score(self, close: pd.Series) -> float:
        """Calculate trend score from -1 to +1."""
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        score = 0.0

        # Price vs MAs
        if current > sma_20:
            score += 0.2
        else:
            score -= 0.2

        if current > sma_50:
            score += 0.3
        else:
            score -= 0.3

        if current > sma_200:
            score += 0.3
        else:
            score -= 0.3

        # MA alignment
        if sma_20 > sma_50 > sma_200:
            score += 0.2
        elif sma_20 < sma_50 < sma_200:
            score -= 0.2

        return max(-1, min(1, score))

    def _calculate_volatility_percentile(self, close: pd.Series) -> float:
        """Calculate current volatility percentile vs history."""
        returns = close.pct_change().dropna()

        # Current 20-day volatility
        current_vol = returns.tail(20).std()

        # Historical volatility distribution (1-year rolling)
        historical_vols = returns.rolling(20).std().dropna()

        if len(historical_vols) > 0:
            percentile = (historical_vols < current_vol).mean() * 100
        else:
            percentile = 50

        return percentile

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate multi-timeframe momentum."""
        # 5-day momentum
        mom_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        # 20-day momentum
        mom_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        # 60-day momentum
        mom_60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0

        # Weighted average
        score = (mom_5 * 0.4 + mom_20 * 0.35 + mom_60 * 0.25) / 10
        return max(-1, min(1, score))

    def _estimate_breadth(self, index_data: pd.DataFrame) -> float:
        """Estimate market breadth from index behavior."""
        # Without actual breadth data, use price action as proxy
        close = index_data['close']
        volume = index_data.get('volume', pd.Series([1] * len(close)))

        # Volume on up days vs down days
        returns = close.pct_change()
        up_volume = volume[returns > 0].tail(20).sum()
        down_volume = volume[returns <= 0].tail(20).sum()

        if up_volume + down_volume > 0:
            breadth = (up_volume - down_volume) / (up_volume + down_volume)
        else:
            breadth = 0

        return breadth

    def _estimate_correlation_regime(self, index_data: pd.DataFrame) -> float:
        """Estimate correlation regime (risk-on vs risk-off)."""
        # High correlation = risk-off environment
        # During crisis, correlations spike to 1

        close = index_data['close']
        returns = close.pct_change().dropna().tail(20)

        if len(returns) < 10:
            return 0.5

        # Use absolute returns as proxy for correlation
        # High absolute returns with same direction = high correlation
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()

        # Clustering indicates high correlation
        if max(positive_days, negative_days) > 15:
            return 0.8  # High correlation
        elif max(positive_days, negative_days) > 12:
            return 0.6
        else:
            return 0.4  # Normal correlation

    def _classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify market regime from metrics."""

        # Crisis check first
        if metrics.vix_equivalent > 35 or metrics.volatility_percentile > 90:
            if metrics.trend_score < -0.3:
                return MarketRegime.CRISIS

        # Strong bear
        if metrics.trend_score < -0.5 and metrics.momentum_score < -0.5:
            return MarketRegime.STRONG_BEAR

        # Bear
        if metrics.trend_score < -0.2 or metrics.momentum_score < -0.3:
            return MarketRegime.BEAR

        # Strong bull
        if metrics.trend_score > 0.5 and metrics.momentum_score > 0.3:
            if metrics.volatility_percentile < 50:
                return MarketRegime.STRONG_BULL

        # Bull
        if metrics.trend_score > 0.2 or metrics.momentum_score > 0.2:
            return MarketRegime.BULL

        return MarketRegime.NEUTRAL

    def _default_metrics(self) -> RegimeMetrics:
        return RegimeMetrics(
            trend_score=0,
            volatility_percentile=50,
            correlation_regime=0.5,
            breadth_score=0,
            momentum_score=0,
            vix_equivalent=15
        )


class RegimeAwarePositionSizer:
    """
    Position sizing that adapts to market conditions.
    """

    def __init__(self,
                 base_position_pct: float = 5.0,      # 5% base position
                 max_position_pct: float = 15.0,      # 15% max single position
                 max_portfolio_pct: float = 90.0,     # 90% max total exposure
                 max_sector_pct: float = 30.0):       # 30% max per sector
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.max_portfolio_pct = max_portfolio_pct
        self.max_sector_pct = max_sector_pct
        self.regime_detector = MarketRegimeDetector()

        # Event calendar (simplified - would fetch from API in production)
        self.known_events = []

    def calculate_position_size(self,
                                 symbol: str,
                                 index_data: pd.DataFrame,
                                 stock_data: pd.DataFrame,
                                 current_positions: Dict[str, float],
                                 current_drawdown_pct: float = 0,
                                 confidence: float = 0.7,
                                 trade_date: date = None) -> PositionSizeResult:
        """
        Calculate optimal position size considering all factors.

        Args:
            symbol: Stock symbol
            index_data: Market index OHLCV
            stock_data: Stock OHLCV
            current_positions: Dict of symbol -> position size %
            current_drawdown_pct: Current portfolio drawdown
            confidence: Prediction confidence (0-1)
            trade_date: Trade date for event checking

        Returns:
            PositionSizeResult with full breakdown
        """
        reasons = []
        trade_date = trade_date or date.today()

        # 1. Detect regime
        regime, metrics = self.regime_detector.detect_regime(index_data)
        regime_multiplier = self.regime_detector.REGIME_MULTIPLIERS[regime]
        reasons.append(f"Regime: {regime.value} (mult: {regime_multiplier:.2f})")

        # 2. Volatility adjustment for individual stock
        vol_adjustment = self._calculate_volatility_adjustment(stock_data)
        reasons.append(f"Stock volatility adjustment: {vol_adjustment:.2f}")

        # 3. Correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(
            symbol, stock_data, current_positions
        )
        reasons.append(f"Correlation penalty: {correlation_penalty:.2f}")

        # 4. Event factor
        event_factor = self._calculate_event_factor(symbol, trade_date)
        if event_factor < 1.0:
            reasons.append(f"Event proximity: {event_factor:.2f}")

        # 5. Drawdown factor
        drawdown_factor = self._calculate_drawdown_factor(current_drawdown_pct)
        if drawdown_factor < 1.0:
            reasons.append(f"Drawdown protection: {drawdown_factor:.2f}")

        # 6. Confidence scaling
        confidence_factor = 0.6 + (confidence * 0.4)  # Range: 0.6 to 1.0
        reasons.append(f"Confidence factor: {confidence_factor:.2f}")

        # Calculate final size
        final_size = (
            self.base_position_pct *
            regime_multiplier *
            vol_adjustment *
            correlation_penalty *
            event_factor *
            drawdown_factor *
            confidence_factor
        )

        # Apply caps
        total_current = sum(current_positions.values())
        available_capacity = self.max_portfolio_pct - total_current

        final_size = min(final_size, self.max_position_pct)
        final_size = min(final_size, available_capacity)
        final_size = max(0.5, final_size)  # Minimum 0.5%

        return PositionSizeResult(
            base_size_pct=self.base_position_pct,
            regime_multiplier=regime_multiplier,
            volatility_adjustment=vol_adjustment,
            correlation_penalty=correlation_penalty,
            event_factor=event_factor,
            drawdown_factor=drawdown_factor,
            final_size_pct=round(final_size, 2),
            max_position_pct=self.max_position_pct,
            regime=regime,
            regime_metrics=metrics,
            reasons=reasons
        )

    def _calculate_volatility_adjustment(self, stock_data: pd.DataFrame) -> float:
        """Adjust for stock-specific volatility."""
        if stock_data.empty or len(stock_data) < 20:
            return 0.8  # Conservative for unknown

        returns = stock_data['close'].pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized %

        # Higher vol = smaller position
        if volatility > 50:      # Very high vol
            return 0.5
        elif volatility > 35:    # High vol
            return 0.7
        elif volatility > 25:    # Normal
            return 0.85
        elif volatility > 15:    # Low vol
            return 1.0
        else:                    # Very low vol
            return 1.1

    def _calculate_correlation_penalty(self,
                                         symbol: str,
                                         stock_data: pd.DataFrame,
                                         current_positions: Dict[str, float]) -> float:
        """Penalize positions correlated with existing holdings."""
        if not current_positions:
            return 1.0

        # Simplified sector-based correlation
        # Would use actual return correlations in production
        sector_mapping = {
            'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
            'HDFCBANK': 'BANK', 'ICICIBANK': 'BANK', 'KOTAKBANK': 'BANK', 'SBIN': 'BANK', 'AXISBANK': 'BANK',
            'RELIANCE': 'ENERGY', 'ONGC': 'ENERGY', 'BPCL': 'ENERGY',
            'BHARTIARTL': 'TELECOM',
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
        }

        symbol_sector = sector_mapping.get(symbol, 'OTHER')

        # Count same-sector positions
        same_sector_weight = 0
        for pos_symbol, weight in current_positions.items():
            if sector_mapping.get(pos_symbol, 'OTHER') == symbol_sector:
                same_sector_weight += weight

        # Penalize concentration
        if same_sector_weight > 20:
            return 0.5
        elif same_sector_weight > 10:
            return 0.7
        elif same_sector_weight > 5:
            return 0.9
        else:
            return 1.0

    def _calculate_event_factor(self, symbol: str, trade_date: date) -> float:
        """Reduce size around known events."""
        # F&O expiry check (last Thursday of month)
        year = trade_date.year
        month = trade_date.month

        # Find last Thursday
        import calendar
        cal = calendar.monthcalendar(year, month)
        last_thursday = max(week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY])
        expiry_date = date(year, month, last_thursday)

        days_to_expiry = (expiry_date - trade_date).days

        if 0 <= days_to_expiry <= 2:
            return 0.6  # Reduce near expiry
        elif days_to_expiry < 0 and days_to_expiry >= -1:
            return 0.7  # Day after expiry

        # Would also check earnings calendar, dividend dates, etc.
        return 1.0

    def _calculate_drawdown_factor(self, current_drawdown_pct: float) -> float:
        """Reduce exposure during drawdowns."""
        if current_drawdown_pct >= 15:
            return 0.3  # Severe drawdown - minimal new positions
        elif current_drawdown_pct >= 10:
            return 0.5
        elif current_drawdown_pct >= 5:
            return 0.75
        else:
            return 1.0


def demo():
    """Demonstrate regime-aware position sizing."""
    print("=" * 60)
    print("RegimeAwarePositionSizer Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='B')

    # Bull market scenario
    bull_returns = np.random.normal(0.001, 0.01, 300)  # Slight positive drift
    bull_prices = 100 * np.cumprod(1 + bull_returns)

    bull_data = pd.DataFrame({
        'open': bull_prices * 0.995,
        'high': bull_prices * 1.01,
        'low': bull_prices * 0.99,
        'close': bull_prices,
        'volume': np.random.uniform(1e6, 5e6, 300)
    }, index=dates)

    sizer = RegimeAwarePositionSizer()

    # Test in bull market
    result = sizer.calculate_position_size(
        symbol='TCS',
        index_data=bull_data,
        stock_data=bull_data,
        current_positions={'INFY': 5.0, 'WIPRO': 3.0},
        current_drawdown_pct=2.0,
        confidence=0.75
    )

    print("\n--- Bull Market Scenario ---")
    print(f"Regime: {result.regime.value}")
    print(f"Base size: {result.base_size_pct}%")
    print(f"Final size: {result.final_size_pct}%")
    print("Adjustments:")
    for reason in result.reasons:
        print(f"  - {reason}")

    # Bear market scenario
    bear_returns = np.random.normal(-0.002, 0.02, 300)  # Negative drift, higher vol
    bear_prices = 100 * np.cumprod(1 + bear_returns)

    bear_data = pd.DataFrame({
        'open': bear_prices * 0.995,
        'high': bear_prices * 1.015,
        'low': bear_prices * 0.985,
        'close': bear_prices,
        'volume': np.random.uniform(2e6, 8e6, 300)
    }, index=dates)

    result = sizer.calculate_position_size(
        symbol='RELIANCE',
        index_data=bear_data,
        stock_data=bear_data,
        current_positions={},
        current_drawdown_pct=8.0,
        confidence=0.65
    )

    print("\n--- Bear Market Scenario ---")
    print(f"Regime: {result.regime.value}")
    print(f"Base size: {result.base_size_pct}%")
    print(f"Final size: {result.final_size_pct}%")
    print("Adjustments:")
    for reason in result.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    demo()
