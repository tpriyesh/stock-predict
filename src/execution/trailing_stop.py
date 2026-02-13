"""
Trailing Stop Calculator

Provides various trailing stop strategies:
- ATR-based trailing
- Percentage-based trailing
- Chandelier exit
- Parabolic SAR style
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np


class TrailingMethod(Enum):
    """Trailing stop calculation methods."""
    ATR_MULTIPLE = "atr_multiple"
    PERCENTAGE = "percentage"
    CHANDELIER = "chandelier"
    STEP = "step"  # Move stop in fixed increments


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop calculation."""
    method: TrailingMethod = TrailingMethod.ATR_MULTIPLE
    atr_multiple: float = 1.5
    percentage: float = 2.0  # For percentage method
    step_r: float = 0.5  # For step method - move stop every 0.5R
    activation_r: float = 1.0  # Activate trailing after 1R profit
    min_profit_lock_r: float = 0.5  # Minimum profit to lock in once trailing


class TrailingStopCalculator:
    """
    Calculates trailing stop levels using various methods.

    Usage:
        calc = TrailingStopCalculator(TrailingStopConfig())
        new_stop = calc.calculate(
            entry_price=100,
            current_stop=95,
            highest_price=110,
            current_price=108,
            atr=2.5,
            is_long=True
        )
    """

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or TrailingStopConfig()

    def calculate(
        self,
        entry_price: float,
        original_stop: float,
        current_stop: float,
        highest_price: float,
        current_price: float,
        atr: float,
        is_long: bool = True
    ) -> float:
        """
        Calculate new trailing stop level.

        Args:
            entry_price: Original entry price
            original_stop: Original stop loss level
            current_stop: Current stop level (may have been trailed already)
            highest_price: Highest price seen since entry (or lowest for short)
            current_price: Current price
            atr: Current ATR value
            is_long: True for long, False for short

        Returns:
            New stop level (same as current if no change)
        """
        # Calculate current R
        risk = abs(entry_price - original_stop)
        if risk == 0:
            return current_stop

        if is_long:
            current_r = (highest_price - entry_price) / risk
        else:
            current_r = (entry_price - highest_price) / risk

        # Check activation threshold
        if current_r < self.config.activation_r:
            return current_stop

        # Calculate new stop based on method
        if self.config.method == TrailingMethod.ATR_MULTIPLE:
            new_stop = self._calculate_atr_trailing(
                highest_price, atr, is_long
            )
        elif self.config.method == TrailingMethod.PERCENTAGE:
            new_stop = self._calculate_percentage_trailing(
                highest_price, is_long
            )
        elif self.config.method == TrailingMethod.CHANDELIER:
            new_stop = self._calculate_chandelier(
                highest_price, atr, is_long
            )
        elif self.config.method == TrailingMethod.STEP:
            new_stop = self._calculate_step_trailing(
                entry_price, original_stop, current_r, is_long
            )
        else:
            new_stop = current_stop

        # Ensure stop only moves in favorable direction
        if is_long:
            return max(current_stop, new_stop)
        else:
            return min(current_stop, new_stop)

    def _calculate_atr_trailing(
        self,
        highest_price: float,
        atr: float,
        is_long: bool
    ) -> float:
        """ATR-based trailing stop."""
        trail_distance = atr * self.config.atr_multiple

        if is_long:
            return highest_price - trail_distance
        else:
            return highest_price + trail_distance

    def _calculate_percentage_trailing(
        self,
        highest_price: float,
        is_long: bool
    ) -> float:
        """Percentage-based trailing stop."""
        trail_distance = highest_price * (self.config.percentage / 100)

        if is_long:
            return highest_price - trail_distance
        else:
            return highest_price + trail_distance

    def _calculate_chandelier(
        self,
        highest_price: float,
        atr: float,
        is_long: bool
    ) -> float:
        """
        Chandelier Exit - classic volatility trailing method.

        Uses 3x ATR from highest high (for longs).
        """
        chandelier_multiple = 3.0
        trail_distance = atr * chandelier_multiple

        if is_long:
            return highest_price - trail_distance
        else:
            return highest_price + trail_distance

    def _calculate_step_trailing(
        self,
        entry_price: float,
        original_stop: float,
        current_r: float,
        is_long: bool
    ) -> float:
        """
        Step trailing - move stop in fixed R increments.

        Example: With step_r=0.5 and min_profit_lock=0.5:
        - At 1R profit: stop moves to 0.5R profit (lock in 0.5R)
        - At 1.5R profit: stop moves to 1R profit
        - At 2R profit: stop moves to 1.5R profit
        """
        risk = abs(entry_price - original_stop)

        # Calculate how many steps we've moved
        steps_above_activation = int(
            (current_r - self.config.activation_r) / self.config.step_r
        )

        if steps_above_activation < 0:
            return original_stop

        # New stop locks in profit at: activation_r + steps * step_r - some buffer
        locked_r = self.config.activation_r - self.config.min_profit_lock_r + (
            steps_above_activation * self.config.step_r
        )

        if is_long:
            return entry_price + (risk * locked_r)
        else:
            return entry_price - (risk * locked_r)

    def get_stop_at_r(
        self,
        entry_price: float,
        original_stop: float,
        target_r: float,
        is_long: bool
    ) -> float:
        """
        Get stop level at a specific R target.

        Useful for calculating where stop should be
        if price reaches a certain R multiple.
        """
        risk = abs(entry_price - original_stop)

        if is_long:
            return entry_price + (risk * target_r)
        else:
            return entry_price - (risk * target_r)


def calculate_dynamic_atr_multiple(
    volatility_percentile: float,
    base_multiple: float = 1.5
) -> float:
    """
    Calculate ATR multiple based on market volatility.

    In high volatility, use wider stops.
    In low volatility, use tighter stops.

    Args:
        volatility_percentile: Current volatility as percentile (0-100)
        base_multiple: Base ATR multiple

    Returns:
        Adjusted ATR multiple
    """
    # Scale from 0.8x to 2.0x based on volatility
    if volatility_percentile < 25:
        # Low volatility - tighter stops
        return base_multiple * 0.8
    elif volatility_percentile < 50:
        # Normal volatility
        return base_multiple
    elif volatility_percentile < 75:
        # Elevated volatility
        return base_multiple * 1.3
    else:
        # High volatility - wider stops
        return base_multiple * 1.8
