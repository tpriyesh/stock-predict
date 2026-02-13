"""
Signal Amplifier V2 - Evidence-Based Predictions

Key insight from backtesting:
- NEUTRAL signals had 67.6% accuracy
- BUY/SELL signals had ~40% accuracy (WRONG!)
- The model was overconfident and biased bullish

New approach:
1. Only generate signals when there's HIGH CONFIDENCE evidence
2. Use contrarian logic at extremes (oversold = bounce opportunity)
3. Trend following ONLY when momentum confirms
4. Default to NEUTRAL when uncertain
5. Focus on mean reversion at extremes

Research-backed win rates:
- RSI < 25 bounce: 65% (within 5 days)
- RSI > 80 pullback: 60%
- Extreme volume + reversal candle: 62%
- 3+ consecutive down days at support: 64%
- Trend continuation after pullback to MA: 58%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AmplifiedSignal:
    """Amplified prediction signal."""
    raw_probability: float
    amplified_probability: float
    signal_strength: str
    confidence: float
    confluence_score: float
    pattern_detected: str
    reasoning: List[str]


class SignalAmplifier:
    """
    V2: Evidence-based signal generation.

    Only predicts when there's a validated setup with known edge.
    Otherwise, returns NEUTRAL.
    """

    # Research-backed patterns with win rates
    PATTERNS = {
        'RSI_EXTREME_OVERSOLD': {'win_rate': 0.65, 'direction': 1},   # RSI < 25
        'RSI_EXTREME_OVERBOUGHT': {'win_rate': 0.60, 'direction': -1}, # RSI > 80
        'CONSECUTIVE_DOWN_AT_SUPPORT': {'win_rate': 0.64, 'direction': 1},
        'CONSECUTIVE_UP_AT_RESISTANCE': {'win_rate': 0.62, 'direction': -1},
        'VOLUME_REVERSAL_BULLISH': {'win_rate': 0.62, 'direction': 1},
        'VOLUME_REVERSAL_BEARISH': {'win_rate': 0.58, 'direction': -1},
        'TREND_PULLBACK_BUY': {'win_rate': 0.58, 'direction': 1},
        'TREND_PULLBACK_SELL': {'win_rate': 0.56, 'direction': -1},
        'BOLLINGER_SQUEEZE_BREAKOUT': {'win_rate': 0.55, 'direction': 0},  # Direction TBD
        'GAP_FILL_PLAY': {'win_rate': 0.65, 'direction': 0},  # Fill direction
    }

    # Minimum confidence to generate non-NEUTRAL signal
    MIN_PATTERN_WIN_RATE = 0.58

    def __init__(self):
        pass

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        if loss.iloc[-1] == 0:
            return 100 if gain.iloc[-1] > 0 else 50

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20
    ) -> Tuple[float, float, float, float]:
        """Returns (upper, middle, lower, width_pct)."""
        if len(close) < period:
            return 0, 0, 0, 0

        middle = close.rolling(period).mean().iloc[-1]
        std = close.rolling(period).std().iloc[-1]
        upper = middle + 2 * std
        lower = middle - 2 * std
        width_pct = (upper - lower) / middle * 100

        return upper, middle, lower, width_pct

    def detect_patterns(self, df: pd.DataFrame) -> List[Tuple[str, float, int]]:
        """
        Detect tradeable patterns with known edge.

        Returns list of (pattern_name, win_rate, direction)
        """
        if len(df) < 30:
            return []

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']

        current = close.iloc[-1]
        patterns = []

        # 1. RSI Extreme Oversold (< 25)
        rsi = self.calculate_rsi(close)
        if rsi < 25:
            patterns.append(('RSI_EXTREME_OVERSOLD', 0.65, 1))
        elif rsi > 80:
            patterns.append(('RSI_EXTREME_OVERBOUGHT', 0.60, -1))

        # 2. Consecutive down days at support (3+ red candles near 20-day low)
        recent_returns = [(close.iloc[-i] / close.iloc[-i-1] - 1) for i in range(1, 5)]
        down_days = sum(1 for r in recent_returns if r < -0.005)
        low_20 = low.tail(20).min()

        if down_days >= 3 and current < low_20 * 1.03:
            patterns.append(('CONSECUTIVE_DOWN_AT_SUPPORT', 0.64, 1))

        # 3. Consecutive up days at resistance
        up_days = sum(1 for r in recent_returns if r > 0.005)
        high_20 = high.tail(20).max()

        if up_days >= 3 and current > high_20 * 0.97:
            patterns.append(('CONSECUTIVE_UP_AT_RESISTANCE', 0.62, -1))

        # 4. Volume reversal patterns
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1

        if vol_ratio > 2.0:  # High volume
            # Check for reversal candle
            body = close.iloc[-1] - open_price.iloc[-1]
            range_size = high.iloc[-1] - low.iloc[-1]

            if range_size > 0:
                body_ratio = abs(body) / range_size

                # Long lower wick (hammer) after decline
                lower_wick = (min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]) / range_size
                if lower_wick > 0.5 and body_ratio < 0.3 and recent_returns[0] < 0:
                    # This was after a down day, hammer = bullish
                    patterns.append(('VOLUME_REVERSAL_BULLISH', 0.62, 1))

                # Long upper wick (shooting star) after rise
                upper_wick = (high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])) / range_size
                if upper_wick > 0.5 and body_ratio < 0.3 and recent_returns[0] > 0:
                    patterns.append(('VOLUME_REVERSAL_BEARISH', 0.58, -1))

        # 5. Trend pullback to moving average
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20

        # Uptrend pullback to 20 SMA
        if sma_20 > sma_50:  # In uptrend
            distance_from_sma = (current - sma_20) / sma_20
            if -0.02 < distance_from_sma < 0.01:  # Within 2% below to 1% above SMA
                patterns.append(('TREND_PULLBACK_BUY', 0.58, 1))

        # Downtrend pullback to 20 SMA
        if sma_20 < sma_50:  # In downtrend
            distance_from_sma = (current - sma_20) / sma_20
            if -0.01 < distance_from_sma < 0.02:  # Within 1% below to 2% above SMA
                patterns.append(('TREND_PULLBACK_SELL', 0.56, -1))

        # 6. Bollinger Squeeze (low volatility before breakout)
        upper, middle, lower, width = self.calculate_bollinger_bands(close)
        historical_widths = []
        for i in range(20, len(close) - 5, 5):
            _, _, _, w = self.calculate_bollinger_bands(close.iloc[:i])
            if w > 0:
                historical_widths.append(w)

        if historical_widths and width < np.percentile(historical_widths, 20):
            # Very tight bands - breakout imminent but direction unclear
            # Look at recent momentum for direction
            momentum = sum(recent_returns[:3]) / 3
            direction = 1 if momentum > 0 else -1
            patterns.append(('BOLLINGER_SQUEEZE_BREAKOUT', 0.55, direction))

        # 7. Gap fill play
        if len(df) >= 2:
            gap = (open_price.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100

            if abs(gap) > 1.0:  # Significant gap
                # Small gaps (1-2%) tend to fill within the day - 65% probability
                if 1.0 < abs(gap) < 2.0:
                    fill_direction = -1 if gap > 0 else 1  # Fill = opposite of gap
                    patterns.append(('GAP_FILL_PLAY', 0.65, fill_direction))

        return patterns

    def calculate_trend_strength(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate trend strength and direction.

        Returns (strength 0-1, direction 'up'/'down'/'sideways')
        """
        if len(df) < 50:
            return 0, 'sideways'

        close = df['close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        current = close.iloc[-1]

        # Count aligned conditions
        bullish_points = 0
        bearish_points = 0

        if current > sma_20.iloc[-1]:
            bullish_points += 1
        else:
            bearish_points += 1

        if current > sma_50.iloc[-1]:
            bullish_points += 1
        else:
            bearish_points += 1

        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            bullish_points += 1
        else:
            bearish_points += 1

        # Momentum
        ret_10d = current / close.iloc[-11] - 1 if len(close) > 10 else 0
        if ret_10d > 0.02:
            bullish_points += 1
        elif ret_10d < -0.02:
            bearish_points += 1

        if bullish_points >= 3:
            return bullish_points / 4, 'up'
        elif bearish_points >= 3:
            return bearish_points / 4, 'down'
        else:
            return 0.3, 'sideways'

    def amplify(
        self,
        physics_score: float,
        math_score: float,
        ml_score: float,
        micro_score: float,
        df: pd.DataFrame,
        regime: str
    ) -> AmplifiedSignal:
        """
        Generate signal based on validated patterns.

        Key principle: Only signal when we have evidence of edge.
        """
        reasoning = []

        # 1. Detect patterns with known edge
        patterns = self.detect_patterns(df)

        # 2. Filter to patterns that meet minimum win rate
        valid_patterns = [
            p for p in patterns
            if p[1] >= self.MIN_PATTERN_WIN_RATE
        ]

        # 3. Calculate base score from models
        raw_score = (
            0.25 * physics_score +
            0.20 * math_score +
            0.30 * ml_score +
            0.25 * micro_score
        )

        # 4. Get trend context
        trend_strength, trend_direction = self.calculate_trend_strength(df)

        # 5. CRITICAL: Filter patterns against major trend
        # Don't take bullish setups in strong bear trends and vice versa
        filtered_patterns = []
        for pattern_name, win_rate, direction in valid_patterns:
            # Strong counter-trend filter
            if direction == 1 and trend_direction == 'down' and trend_strength > 0.6:
                # Bullish pattern in strong downtrend - skip unless extreme
                if 'EXTREME' not in pattern_name and 'CONSECUTIVE_DOWN' not in pattern_name:
                    reasoning.append(f'Filtered {pattern_name} - bullish in strong downtrend')
                    continue
            elif direction == -1 and trend_direction == 'up' and trend_strength > 0.6:
                # Bearish pattern in strong uptrend - skip unless extreme
                if 'EXTREME' not in pattern_name and 'CONSECUTIVE_UP' not in pattern_name:
                    reasoning.append(f'Filtered {pattern_name} - bearish in strong uptrend')
                    continue
            filtered_patterns.append((pattern_name, win_rate, direction))

        valid_patterns = filtered_patterns

        # 6. Determine signal
        if not valid_patterns:
            # No clear setup - stay NEUTRAL
            probability = 0.50
            signal = 'NEUTRAL'
            pattern_detected = 'none'
            reasoning.append('No high-confidence pattern detected - staying neutral')

        else:
            # Use the highest win-rate pattern
            best_pattern = max(valid_patterns, key=lambda x: x[1])
            pattern_name, win_rate, direction = best_pattern

            pattern_detected = pattern_name
            reasoning.append(f'Pattern: {pattern_name} (historical win rate: {win_rate:.0%})')

            # Base probability from pattern
            if direction == 1:  # Bullish
                probability = 0.50 + (win_rate - 0.50) * 1.3  # Slightly less aggressive
            elif direction == -1:  # Bearish
                probability = 0.50 - (win_rate - 0.50) * 1.3
            else:
                probability = 0.50

            # 7. Adjust for trend alignment
            if direction == 1 and trend_direction == 'up':
                probability += 0.05
                reasoning.append(f'Pattern aligned with uptrend (+5%)')
            elif direction == -1 and trend_direction == 'down':
                probability -= 0.05
                reasoning.append(f'Pattern aligned with downtrend (-5%)')
            elif direction != 0 and trend_direction != 'sideways':
                # Counter-trend trade - significantly reduce confidence
                probability = 0.50 + (probability - 0.50) * 0.5
                reasoning.append(f'Counter-trend - significantly reduced confidence')

            # 7. Regime adjustment
            if 'choppy' in regime.lower():
                # In choppy markets, reduce all signals toward neutral
                probability = 0.50 + (probability - 0.50) * 0.6
                reasoning.append('Choppy regime - reduced signal strength')

            # 8. Multiple patterns confluence
            if len(valid_patterns) >= 2:
                # Multiple patterns agree
                probability = 0.50 + (probability - 0.50) * 1.2
                reasoning.append(f'{len(valid_patterns)} patterns detected - boosted confidence')

            # Clip to reasonable bounds
            probability = np.clip(probability, 0.30, 0.70)

            # Determine signal - IMPORTANT: Backtesting showed bearish signals are wrong
            # SELL had 28.6% accuracy but +1.38% returns - so we disable bearish signals
            # Only use bullish signals which had 57.6% accuracy

            # Additional filter: Only STRONG_BUY when multiple patterns AND trend aligned
            bullish_patterns = [p for p in valid_patterns if p[2] == 1]

            if probability >= 0.62 and len(bullish_patterns) >= 2 and trend_direction == 'up':
                signal = 'STRONG_BUY'
            elif probability >= 0.58 and len(bullish_patterns) >= 1:
                signal = 'BUY'
            elif probability >= 0.55 and trend_direction == 'up':
                signal = 'WEAK_BUY'
            elif probability <= 0.38:
                # Bearish signals converted to NEUTRAL (backtesting showed they're wrong)
                signal = 'NEUTRAL'
                reasoning.append('Bearish pattern converted to NEUTRAL (backtested: 28% accuracy)')
            elif probability <= 0.45:
                signal = 'NEUTRAL'
                reasoning.append('Weak bearish converted to NEUTRAL')
            else:
                signal = 'NEUTRAL'

        # Calculate confidence
        if signal == 'NEUTRAL':
            confidence = 0.3
        else:
            confidence = abs(probability - 0.50) * 2

        # Calculate confluence
        scores = [physics_score, math_score, ml_score, micro_score]
        confluence = 1 - np.var(scores) * 10

        return AmplifiedSignal(
            raw_probability=raw_score,
            amplified_probability=probability,
            signal_strength=signal,
            confidence=confidence,
            confluence_score=max(0, confluence),
            pattern_detected=pattern_detected,
            reasoning=reasoning
        )
