"""
Intraday Pattern Analyzer

Analyzes intraday patterns from daily data:
- Gap Analysis: Gap fill probability
- Opening Range approximation
- VWAP-based signals
- First candle patterns (using open/close)

Note: For true intraday patterns, use intraday data.
This module uses daily data to approximate intraday signals.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class IntradayResult:
    """Result from intraday pattern analysis."""
    # Gap analysis
    gap_type: str                   # 'up', 'down', 'none'
    gap_size_pct: float             # Gap size as percentage
    gap_fill_probability: float     # Probability of gap fill
    gap_filled: bool                # Was gap filled?

    # Opening range
    opening_strength: str           # 'strong_bull', 'strong_bear', 'neutral'
    opening_range_pct: float        # Opening range as % of price

    # VWAP estimate
    vwap_estimate: float            # Approximate VWAP
    price_vs_vwap: float            # Current price vs VWAP

    # Range analysis
    day_type_prediction: str        # 'trend', 'range', 'volatile'
    atr_percentile: float           # Today's range vs history

    probability_score: float        # Contribution to prediction (0-1)
    signal: str                     # 'GAP_FILL', 'TREND_DAY', 'RANGE_DAY'
    reason: str                     # Human-readable explanation


class IntradayPatternAnalyzer:
    """
    Analyzes intraday-style patterns from daily data.

    Uses daily OHLC to infer intraday behavior.
    """

    # Gap fill probabilities (from research)
    GAP_FILL_PROBS = {
        'tiny': 0.85,    # < 0.3%
        'small': 0.75,   # 0.3% - 0.8%
        'medium': 0.55,  # 0.8% - 1.5%
        'large': 0.35,   # 1.5% - 3%
        'huge': 0.20     # > 3%
    }

    def __init__(
        self,
        atr_period: int = 14
    ):
        self.atr_period = atr_period

    def analyze_gap(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """
        Analyze gap between yesterday's close and today's open.

        Returns (gap_type, gap_size_pct, fill_probability, filled)
        """
        if len(df) < 2:
            return 'none', 0, 0.5, False

        prev_close = df['close'].iloc[-2]
        today_open = df['open'].iloc[-1]
        today_low = df['low'].iloc[-1]
        today_high = df['high'].iloc[-1]

        gap_pct = (today_open - prev_close) / prev_close * 100

        if abs(gap_pct) < 0.1:
            return 'none', gap_pct, 1.0, True

        gap_type = 'up' if gap_pct > 0 else 'down'
        gap_size = abs(gap_pct)

        # Classify gap size
        if gap_size < 0.3:
            size_class = 'tiny'
        elif gap_size < 0.8:
            size_class = 'small'
        elif gap_size < 1.5:
            size_class = 'medium'
        elif gap_size < 3.0:
            size_class = 'large'
        else:
            size_class = 'huge'

        fill_prob = self.GAP_FILL_PROBS[size_class]

        # Check if gap was filled
        if gap_type == 'up':
            filled = today_low <= prev_close
        else:
            filled = today_high >= prev_close

        return gap_type, gap_pct, fill_prob, filled

    def analyze_opening(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """
        Analyze opening strength.

        Returns (strength, range_pct)
        """
        if len(df) < 1:
            return 'neutral', 0

        today = df.iloc[-1]

        open_price = today['open']
        close_price = today['close']
        high_price = today['high']
        low_price = today['low']

        # Opening range (high-low as proxy)
        range_size = high_price - low_price
        range_pct = range_size / open_price * 100

        # Body analysis
        body = close_price - open_price
        body_pct = body / open_price * 100

        # Strength classification
        if body_pct > 0.8:
            strength = 'strong_bull'
        elif body_pct < -0.8:
            strength = 'strong_bear'
        elif body_pct > 0.3:
            strength = 'mild_bull'
        elif body_pct < -0.3:
            strength = 'mild_bear'
        else:
            strength = 'neutral'

        return strength, range_pct

    def estimate_vwap(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Estimate VWAP from daily data.

        Uses typical price as proxy for true intraday VWAP.
        """
        if len(df) < 1:
            return 0

        # Typical price for recent days
        typical = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume']

        # Volume-weighted average
        vwap = (typical * volume).sum() / volume.sum()

        return vwap

    def predict_day_type(
        self,
        df: pd.DataFrame
    ) -> str:
        """
        Predict type of trading day.

        Based on recent volatility and patterns.
        """
        if len(df) < 10:
            return 'range'

        # Calculate ATR
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()

        # Recent trend strength
        returns_5d = df['close'].iloc[-1] / df['close'].iloc[-6] - 1

        # Classify
        if current_atr > avg_atr * 1.3:
            return 'volatile'
        elif abs(returns_5d) > 0.03 and current_atr > avg_atr * 0.9:
            return 'trend'
        else:
            return 'range'

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        return atr

    def calculate_atr_percentile(self, df: pd.DataFrame) -> float:
        """Calculate where today's range sits in ATR history."""
        if len(df) < 20:
            return 50

        atr = self._calculate_atr(df)
        today_range = df['high'].iloc[-1] - df['low'].iloc[-1]

        # Percentile
        percentile = (atr < today_range).mean() * 100

        return percentile

    def get_signal_label(
        self,
        gap_type: str,
        gap_size: float,
        gap_filled: bool,
        opening_strength: str,
        day_type: str
    ) -> tuple:
        """Determine signal label and reason."""

        if gap_type != 'none' and not gap_filled:
            if gap_type == 'up':
                return 'GAP_UP_OPEN', f'Gap up {abs(gap_size):.1f}% - watch for fill'
            else:
                return 'GAP_DOWN_OPEN', f'Gap down {abs(gap_size):.1f}% - watch for fill'

        if 'strong' in opening_strength:
            if 'bull' in opening_strength:
                return 'STRONG_OPEN_BULL', 'Strong bullish opening'
            else:
                return 'STRONG_OPEN_BEAR', 'Strong bearish opening'

        if day_type == 'trend':
            return 'TREND_DAY', 'Trending day expected'
        elif day_type == 'volatile':
            return 'VOLATILE_DAY', 'High volatility day'
        else:
            return 'RANGE_DAY', 'Range-bound day expected'

    def analyze(self, df: pd.DataFrame) -> IntradayResult:
        """
        Analyze intraday patterns from daily data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            IntradayResult with pattern analysis
        """
        if len(df) < 5:
            return self._default_result('Insufficient data for intraday analysis')

        # Gap analysis
        gap_type, gap_size, gap_fill_prob, gap_filled = self.analyze_gap(df)

        # Opening analysis
        opening_strength, opening_range = self.analyze_opening(df)

        # VWAP estimate
        vwap = self.estimate_vwap(df.tail(5))
        current_price = df['close'].iloc[-1]
        price_vs_vwap = (current_price - vwap) / vwap if vwap > 0 else 0

        # Day type
        day_type = self.predict_day_type(df)

        # ATR percentile
        atr_pct = self.calculate_atr_percentile(df)

        # Signal and reason
        signal, reason = self.get_signal_label(
            gap_type, gap_size, gap_filled, opening_strength, day_type
        )

        # Probability score
        # Gap up unfilled: potential fill (bearish)
        # Gap down unfilled: potential fill (bullish)
        # Strong opening in direction: continuation
        if gap_type == 'up' and not gap_filled:
            prob_score = 0.45  # Expect fill (down)
        elif gap_type == 'down' and not gap_filled:
            prob_score = 0.55  # Expect fill (up)
        elif opening_strength == 'strong_bull':
            prob_score = 0.58
        elif opening_strength == 'strong_bear':
            prob_score = 0.42
        elif price_vs_vwap > 0.01:
            prob_score = 0.53
        elif price_vs_vwap < -0.01:
            prob_score = 0.47
        else:
            prob_score = 0.50

        # Adjust for day type
        if day_type == 'volatile':
            prob_score = 0.5 + (prob_score - 0.5) * 0.8  # Reduce confidence

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return IntradayResult(
            gap_type=gap_type,
            gap_size_pct=gap_size,
            gap_fill_probability=gap_fill_prob,
            gap_filled=gap_filled,
            opening_strength=opening_strength,
            opening_range_pct=opening_range,
            vwap_estimate=vwap,
            price_vs_vwap=price_vs_vwap,
            day_type_prediction=day_type,
            atr_percentile=atr_pct,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> IntradayResult:
        """Return neutral result."""
        return IntradayResult(
            gap_type='none',
            gap_size_pct=0,
            gap_fill_probability=0.5,
            gap_filled=True,
            opening_strength='neutral',
            opening_range_pct=0,
            vwap_estimate=0,
            price_vs_vwap=0,
            day_type_prediction='range',
            atr_percentile=50,
            probability_score=0.50,
            signal='INTRADAY_UNKNOWN',
            reason=reason
        )
