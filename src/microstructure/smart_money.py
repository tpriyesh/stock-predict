"""
Smart Money Detector

Detects institutional activity patterns:
- Unusual volume at key levels
- Accumulation vs Distribution
- Block trade detection
- Institutional vs Retail signatures

Trading Signals:
- Accumulation pattern: Bullish
- Distribution pattern: Bearish
- Block buying at support: Strong bullish
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class SmartMoneyResult:
    """Result from smart money detection."""
    accumulation_distribution: str   # 'accumulation', 'distribution', 'neutral'
    ad_score: float                  # -1 to 1 (positive = accumulation)
    unusual_volume_detected: bool    # Volume anomaly present
    unusual_volume_type: str         # 'spike', 'drought', 'climax'
    institutional_activity: str      # 'high', 'moderate', 'low'
    key_levels_with_volume: List[float]  # Price levels with unusual volume
    obv_divergence: str              # 'bullish', 'bearish', 'none'
    money_flow_index: float          # 0-100 (MFI)
    probability_score: float         # Contribution to prediction (0-1)
    signal: str                      # 'ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL'
    reason: str                      # Human-readable explanation


class SmartMoneyDetector:
    """
    Detects institutional/smart money activity.

    Uses volume analysis, OBV divergence, and price-volume patterns.
    """

    def __init__(
        self,
        volume_threshold: float = 2.0,
        obv_lookback: int = 20,
        mfi_period: int = 14
    ):
        self.volume_threshold = volume_threshold
        self.obv_lookback = obv_lookback
        self.mfi_period = mfi_period

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(df['close'].diff())
        obv = (direction * df['volume']).cumsum()
        return obv

    def detect_obv_divergence(self, df: pd.DataFrame) -> str:
        """
        Detect OBV divergence from price.

        Bullish divergence: Price down, OBV up
        Bearish divergence: Price up, OBV down
        """
        if len(df) < self.obv_lookback:
            return 'none'

        obv = self.calculate_obv(df)

        # Compare trends over lookback period
        price_trend = df['close'].iloc[-1] / df['close'].iloc[-self.obv_lookback] - 1
        obv_trend = obv.iloc[-1] / obv.iloc[-self.obv_lookback] - 1 if obv.iloc[-self.obv_lookback] != 0 else 0

        # Divergence detection
        if price_trend < -0.02 and obv_trend > 0.05:
            return 'bullish'
        elif price_trend > 0.02 and obv_trend < -0.05:
            return 'bearish'
        else:
            return 'none'

    def calculate_accumulation_distribution(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate Accumulation/Distribution score.

        AD = CLV * Volume
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        """
        if len(df) < 5:
            return 0

        data = df.tail(20)

        # Close Location Value
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / \
              (data['high'] - data['low'] + 1e-10)

        # AD line
        ad = (clv * data['volume']).cumsum()

        # Normalize to -1 to 1
        ad_change = ad.iloc[-1] - ad.iloc[0]
        max_possible = data['volume'].sum()

        ad_score = ad_change / max_possible if max_possible > 0 else 0
        ad_score = np.clip(ad_score, -1, 1)

        return ad_score

    def detect_unusual_volume(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """
        Detect unusual volume patterns.

        Returns (detected, type)
        """
        if len(df) < 20:
            return False, 'none'

        volume = df['volume']
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()

        # Z-score of recent volume
        z_score = (volume.iloc[-1] - vol_mean.iloc[-1]) / (vol_std.iloc[-1] + 1e-10)

        # Volume spike
        if z_score > self.volume_threshold:
            # Check if reversal candle (climax)
            if self._is_reversal_candle(df.iloc[-1]):
                return True, 'climax'
            return True, 'spike'

        # Volume drought
        if z_score < -1.5 and (volume.tail(3) < vol_mean.tail(3) * 0.5).all():
            return True, 'drought'

        return False, 'none'

    def _is_reversal_candle(self, row: pd.Series) -> bool:
        """Check if candle shows reversal pattern."""
        body = abs(row['close'] - row['open'])
        range_size = row['high'] - row['low']

        if range_size == 0:
            return False

        body_ratio = body / range_size

        # Long wick on one side
        if row['close'] > row['open']:
            lower_wick = (min(row['open'], row['close']) - row['low']) / range_size
            return lower_wick > 0.5 and body_ratio < 0.3
        else:
            upper_wick = (row['high'] - max(row['open'], row['close'])) / range_size
            return upper_wick > 0.5 and body_ratio < 0.3

    def calculate_money_flow_index(self, df: pd.DataFrame) -> float:
        """
        Calculate Money Flow Index (volume-weighted RSI).
        """
        if len(df) < self.mfi_period + 1:
            return 50

        # Typical price
        typical = (df['high'] + df['low'] + df['close']) / 3

        # Money flow
        money_flow = typical * df['volume']

        # Positive and negative flow
        flow_direction = typical.diff()
        positive_flow = money_flow.where(flow_direction > 0, 0)
        negative_flow = money_flow.where(flow_direction < 0, 0)

        # Sum over period
        positive_sum = positive_flow.rolling(self.mfi_period).sum()
        negative_sum = negative_flow.rolling(self.mfi_period).sum()

        # MFI
        money_ratio = positive_sum / (negative_sum + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi.iloc[-1]

    def classify_institutional_activity(
        self,
        df: pd.DataFrame
    ) -> str:
        """
        Classify level of institutional activity.

        Based on volume patterns and concentration.
        """
        if len(df) < 20:
            return 'moderate'

        volume = df['volume']

        # Average volume ratio (recent vs long-term)
        recent_avg = volume.tail(5).mean()
        long_avg = volume.mean()
        vol_ratio = recent_avg / long_avg

        # Volume concentration in certain hours would indicate institutional
        # (With daily data, we use volume spikes as proxy)
        n_spikes = (volume.tail(20) > long_avg * 1.5).sum()

        if vol_ratio > 1.3 and n_spikes >= 5:
            return 'high'
        elif vol_ratio > 1.1 or n_spikes >= 3:
            return 'moderate'
        else:
            return 'low'

    def find_key_levels_with_volume(
        self,
        df: pd.DataFrame
    ) -> List[float]:
        """Find price levels with unusually high volume."""
        if len(df) < 10:
            return []

        levels = []
        vol_mean = df['volume'].mean()

        for idx, row in df.iterrows():
            if row['volume'] > vol_mean * self.volume_threshold:
                # Use typical price as key level
                level = (row['high'] + row['low'] + row['close']) / 3
                levels.append(round(level, 2))

        return list(set(levels))[:5]

    def get_signal_label(
        self,
        ad_score: float,
        obv_div: str,
        unusual_vol_type: str
    ) -> tuple:
        """Determine signal label and reason."""

        if ad_score > 0.3 and obv_div == 'bullish':
            return 'STRONG_ACCUMULATION', 'Clear accumulation with bullish OBV divergence'
        elif ad_score < -0.3 and obv_div == 'bearish':
            return 'STRONG_DISTRIBUTION', 'Clear distribution with bearish OBV divergence'
        elif ad_score > 0.2:
            return 'ACCUMULATION', f'Accumulation pattern (AD score: {ad_score:.2f})'
        elif ad_score < -0.2:
            return 'DISTRIBUTION', f'Distribution pattern (AD score: {ad_score:.2f})'
        elif unusual_vol_type == 'climax':
            return 'CLIMAX_VOLUME', 'Climax volume - potential reversal'
        elif unusual_vol_type == 'drought':
            return 'VOLUME_DROUGHT', 'Volume drought - breakout may be pending'
        else:
            return 'NEUTRAL', 'No clear smart money signal'

    def analyze(self, df: pd.DataFrame) -> SmartMoneyResult:
        """
        Analyze for smart money patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            SmartMoneyResult with analysis
        """
        if len(df) < 20:
            return self._default_result('Insufficient data for smart money analysis')

        # Accumulation/Distribution
        ad_score = self.calculate_accumulation_distribution(df)

        if ad_score > 0.1:
            ad_type = 'accumulation'
        elif ad_score < -0.1:
            ad_type = 'distribution'
        else:
            ad_type = 'neutral'

        # OBV divergence
        obv_div = self.detect_obv_divergence(df)

        # Unusual volume
        unusual_detected, unusual_type = self.detect_unusual_volume(df)

        # Institutional activity
        inst_activity = self.classify_institutional_activity(df)

        # Key levels
        key_levels = self.find_key_levels_with_volume(df)

        # MFI
        mfi = self.calculate_money_flow_index(df)

        # Signal and reason
        signal, reason = self.get_signal_label(ad_score, obv_div, unusual_type)

        # Probability score
        if 'ACCUMULATION' in signal:
            prob_score = 0.55 + abs(ad_score) * 0.15
        elif 'DISTRIBUTION' in signal:
            prob_score = 0.45 - abs(ad_score) * 0.10
        elif unusual_type == 'climax':
            # Potential reversal
            recent_return = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
            prob_score = 0.45 if recent_return > 0 else 0.55
        else:
            prob_score = 0.50

        # MFI adjustment
        if mfi > 80:
            prob_score -= 0.03  # Overbought
        elif mfi < 20:
            prob_score += 0.03  # Oversold

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return SmartMoneyResult(
            accumulation_distribution=ad_type,
            ad_score=ad_score,
            unusual_volume_detected=unusual_detected,
            unusual_volume_type=unusual_type,
            institutional_activity=inst_activity,
            key_levels_with_volume=key_levels,
            obv_divergence=obv_div,
            money_flow_index=mfi,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> SmartMoneyResult:
        """Return neutral result."""
        return SmartMoneyResult(
            accumulation_distribution='neutral',
            ad_score=0,
            unusual_volume_detected=False,
            unusual_volume_type='none',
            institutional_activity='moderate',
            key_levels_with_volume=[],
            obv_divergence='none',
            money_flow_index=50,
            probability_score=0.50,
            signal='SMARTMONEY_UNKNOWN',
            reason=reason
        )
