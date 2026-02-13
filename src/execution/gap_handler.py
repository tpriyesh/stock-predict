"""
Gap Handler

Handles gap-up and gap-down scenarios:
- Gap detection and classification
- Entry skipping on excessive gaps
- Stop loss adjustment on gap-through
- Gap fill probability analysis
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


class GapType(Enum):
    """Types of price gaps."""
    NONE = "none"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    GAP_UP_CONTINUATION = "gap_up_continuation"
    GAP_DOWN_CONTINUATION = "gap_down_continuation"
    GAP_UP_EXHAUSTION = "gap_up_exhaustion"
    GAP_DOWN_EXHAUSTION = "gap_down_exhaustion"


@dataclass
class GapAnalysis:
    """Analysis of a price gap."""
    gap_type: GapType
    gap_size: float  # Absolute gap size
    gap_pct: float   # Gap as percentage
    gap_atr: float   # Gap as ATR multiple
    fill_probability: float  # Historical probability of gap fill
    recommendation: str  # 'skip', 'adjust_sl', 'proceed'
    details: str


class GapHandler:
    """
    Handles gap detection and decision-making for trade execution.

    Usage:
        handler = GapHandler()

        # Analyze a gap
        analysis = handler.analyze_gap(
            prev_close=100,
            current_open=105,
            atr=2.5,
            is_long=True
        )

        if analysis.recommendation == 'skip':
            skip_entry()
        elif analysis.recommendation == 'adjust_sl':
            new_sl = handler.adjust_stop_for_gap(...)
    """

    def __init__(
        self,
        skip_threshold_atr: float = 2.0,
        significant_gap_pct: float = 1.5
    ):
        """
        Initialize gap handler.

        Args:
            skip_threshold_atr: Skip entry if gap exceeds this ATR multiple
            significant_gap_pct: Gaps above this % are considered significant
        """
        self.skip_threshold_atr = skip_threshold_atr
        self.significant_gap_pct = significant_gap_pct

    def detect_gap(
        self,
        prev_close: float,
        current_open: float,
        atr: float
    ) -> Tuple[GapType, float]:
        """
        Detect gap type and magnitude.

        Args:
            prev_close: Previous day's close
            current_open: Current day's open
            atr: Average True Range

        Returns:
            Tuple of (gap_type, gap_atr_multiple)
        """
        gap = current_open - prev_close
        gap_atr = abs(gap) / atr if atr > 0 else 0
        gap_pct = abs(gap) / prev_close * 100 if prev_close > 0 else 0

        if gap_atr < 0.3:
            return GapType.NONE, gap_atr

        if gap > 0:
            return GapType.GAP_UP, gap_atr
        else:
            return GapType.GAP_DOWN, gap_atr

    def analyze_gap(
        self,
        prev_close: float,
        current_open: float,
        atr: float,
        is_long: bool,
        stop_loss: Optional[float] = None,
        historical_fills: Optional[List[bool]] = None
    ) -> GapAnalysis:
        """
        Analyze a gap and provide recommendations.

        Args:
            prev_close: Previous close
            current_open: Current open
            atr: ATR value
            is_long: True if entering/holding long position
            stop_loss: Current stop loss level
            historical_fills: List of bools indicating historical gap fills

        Returns:
            GapAnalysis with recommendation
        """
        gap_type, gap_atr = self.detect_gap(prev_close, current_open, atr)
        gap_size = current_open - prev_close
        gap_pct = abs(gap_size) / prev_close * 100 if prev_close > 0 else 0

        # Calculate fill probability
        if historical_fills:
            fill_probability = sum(historical_fills) / len(historical_fills)
        else:
            # Default fill probabilities based on gap size
            if gap_atr < 0.5:
                fill_probability = 0.70  # Small gaps often fill
            elif gap_atr < 1.0:
                fill_probability = 0.55
            elif gap_atr < 1.5:
                fill_probability = 0.40
            else:
                fill_probability = 0.25  # Large gaps often continue

        # Determine recommendation
        recommendation = 'proceed'
        details = f"Gap: {gap_pct:.1f}% ({gap_atr:.1f}x ATR)"

        # Check if gap is against our position
        if is_long and gap_type == GapType.GAP_DOWN:
            if stop_loss and current_open < stop_loss:
                recommendation = 'adjust_sl'
                details += " | Gap through stop - adjust SL required"
            elif gap_atr > self.skip_threshold_atr:
                recommendation = 'skip'
                details += f" | Gap too large (>{self.skip_threshold_atr}x ATR) - skip entry"
        elif not is_long and gap_type == GapType.GAP_UP:
            if stop_loss and current_open > stop_loss:
                recommendation = 'adjust_sl'
                details += " | Gap through stop - adjust SL required"
            elif gap_atr > self.skip_threshold_atr:
                recommendation = 'skip'
                details += f" | Gap too large - skip entry"

        # Also skip if gap in direction but too extreme
        if is_long and gap_type == GapType.GAP_UP and gap_atr > self.skip_threshold_atr * 1.5:
            recommendation = 'skip'
            details += " | Exhaustion gap risk - skip entry"
        elif not is_long and gap_type == GapType.GAP_DOWN and gap_atr > self.skip_threshold_atr * 1.5:
            recommendation = 'skip'
            details += " | Exhaustion gap risk - skip entry"

        return GapAnalysis(
            gap_type=gap_type,
            gap_size=gap_size,
            gap_pct=gap_pct,
            gap_atr=gap_atr,
            fill_probability=fill_probability,
            recommendation=recommendation,
            details=details
        )

    def should_skip_entry(
        self,
        prev_close: float,
        current_open: float,
        atr: float,
        threshold: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if entry should be skipped due to gap.

        Args:
            prev_close: Previous close
            current_open: Current open
            atr: ATR value
            threshold: Optional custom threshold (default: skip_threshold_atr)

        Returns:
            Tuple of (should_skip, reason)
        """
        threshold = threshold or self.skip_threshold_atr
        gap_type, gap_atr = self.detect_gap(prev_close, current_open, atr)

        if gap_type == GapType.NONE:
            return False, "No significant gap"

        if gap_atr > threshold:
            return True, f"Gap {gap_atr:.1f}x ATR exceeds threshold {threshold}x"

        return False, f"Gap {gap_atr:.1f}x ATR within acceptable range"

    def adjust_stop_for_gap(
        self,
        gap_type: GapType,
        original_sl: float,
        open_price: float,
        buffer_pct: float = 0.5,
        is_long: bool = True
    ) -> float:
        """
        Adjust stop loss when price gaps through original SL.

        Args:
            gap_type: Type of gap
            original_sl: Original stop loss level
            open_price: Open price that gapped through
            buffer_pct: Buffer below/above gap to place new SL
            is_long: True for long position

        Returns:
            Adjusted stop loss level
        """
        if is_long and gap_type == GapType.GAP_DOWN:
            # For long position, gap down through SL
            # Place new SL below the gap open with buffer
            return open_price * (1 - buffer_pct / 100)
        elif not is_long and gap_type == GapType.GAP_UP:
            # For short position, gap up through SL
            return open_price * (1 + buffer_pct / 100)

        # No adjustment needed
        return original_sl

    def classify_gap_pattern(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> GapType:
        """
        Classify the gap as continuation or exhaustion based on context.

        Uses prior trend and volume to classify.
        """
        if len(df) < lookback + 1:
            return GapType.NONE

        # Calculate prior trend
        closes = df['close'].values
        prior_return = (closes[-2] - closes[-lookback-1]) / closes[-lookback-1]

        # Calculate gap
        gap = closes[-1] - closes[-2]
        gap_pct = gap / closes[-2]

        # Get volume context
        volumes = df['volume'].values
        avg_volume = volumes[-lookback:-1].mean()
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Classification logic
        if abs(gap_pct) < 0.005:  # Less than 0.5%
            return GapType.NONE

        if gap > 0:
            # Gap up
            if prior_return > 0.05 and volume_ratio > 1.5:
                # Strong uptrend with high volume - likely continuation
                return GapType.GAP_UP_CONTINUATION
            elif prior_return > 0.10 and volume_ratio < 0.8:
                # Extended move with low volume - possible exhaustion
                return GapType.GAP_UP_EXHAUSTION
            else:
                return GapType.GAP_UP
        else:
            # Gap down
            if prior_return < -0.05 and volume_ratio > 1.5:
                return GapType.GAP_DOWN_CONTINUATION
            elif prior_return < -0.10 and volume_ratio < 0.8:
                return GapType.GAP_DOWN_EXHAUSTION
            else:
                return GapType.GAP_DOWN

    def calculate_historical_fill_rate(
        self,
        df: pd.DataFrame,
        min_gap_pct: float = 0.5
    ) -> float:
        """
        Calculate historical gap fill rate from price data.

        A gap is "filled" if price later returns to the pre-gap close.

        Args:
            df: DataFrame with OHLC data
            min_gap_pct: Minimum gap % to consider

        Returns:
            Fill rate (0-1)
        """
        if len(df) < 10:
            return 0.5

        gaps = []
        fills = []

        for i in range(1, len(df) - 5):
            prev_close = df.iloc[i-1]['close']
            current_open = df.iloc[i]['open']
            gap_pct = abs(current_open - prev_close) / prev_close * 100

            if gap_pct < min_gap_pct:
                continue

            gap_up = current_open > prev_close

            # Check if filled in next 5 bars
            filled = False
            for j in range(i, min(i + 5, len(df))):
                if gap_up:
                    # Gap up fills if price comes back down to prev close
                    if df.iloc[j]['low'] <= prev_close:
                        filled = True
                        break
                else:
                    # Gap down fills if price comes back up to prev close
                    if df.iloc[j]['high'] >= prev_close:
                        filled = True
                        break

            gaps.append(gap_pct)
            fills.append(filled)

        if not fills:
            return 0.5

        return sum(fills) / len(fills)
