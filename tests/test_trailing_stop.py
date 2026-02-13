"""
Tests for src/execution/trailing_stop.py - TrailingStopCalculator.

All tests are deterministic with no network calls.
Verifies each trailing method, activation gating, directional ratchet,
and edge cases (zero risk, short positions).
"""
import pytest

from src.execution.trailing_stop import (
    TrailingMethod,
    TrailingStopConfig,
    TrailingStopCalculator,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def pct_calc():
    """Calculator with PERCENTAGE method, 2% trail."""
    config = TrailingStopConfig(
        method=TrailingMethod.PERCENTAGE,
        percentage=2.0,
        activation_r=1.0,
        min_profit_lock_r=0.5,
    )
    return TrailingStopCalculator(config)


@pytest.fixture
def atr_calc():
    """Calculator with ATR_MULTIPLE method, 1.5x multiplier."""
    config = TrailingStopConfig(
        method=TrailingMethod.ATR_MULTIPLE,
        atr_multiple=1.5,
        activation_r=1.0,
    )
    return TrailingStopCalculator(config)


@pytest.fixture
def chandelier_calc():
    """Calculator with CHANDELIER method (3x ATR from highest)."""
    config = TrailingStopConfig(
        method=TrailingMethod.CHANDELIER,
        activation_r=1.0,
    )
    return TrailingStopCalculator(config)


@pytest.fixture
def step_calc():
    """Calculator with STEP method, step_r=0.5, activation_r=1.0, min_profit_lock=0.5."""
    config = TrailingStopConfig(
        method=TrailingMethod.STEP,
        step_r=0.5,
        activation_r=1.0,
        min_profit_lock_r=0.5,
    )
    return TrailingStopCalculator(config)


# ============================================
# TESTS
# ============================================

class TestTrailingStopCalculator:
    """Tests for trailing stop calculation logic."""

    def test_no_trail_before_activation(self, pct_calc):
        """
        When price has moved less than activation_r (1R), the stop
        should remain unchanged at current_stop.

        Setup: entry=100, stop=95 => risk=5, 1R target=105.
        Price at 103 => only 0.6R => no trailing.
        """
        result = pct_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=95.0,
            highest_price=103.0,  # (103-100)/5 = 0.6R < 1.0R
            current_price=102.0,
            atr=2.0,
            is_long=True,
        )
        assert result == 95.0

    def test_percentage_trailing(self, pct_calc):
        """
        PERCENTAGE method: 2% from highest high.

        entry=100, stop=95, highest=110 => 2R (activated).
        new_stop = 110 * (1 - 0.02) = 107.8
        Since 107.8 > current_stop(95), should trail up.
        """
        result = pct_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=95.0,
            highest_price=110.0,
            current_price=108.0,
            atr=2.0,
            is_long=True,
        )
        assert result == pytest.approx(107.8)

    def test_atr_trailing(self, atr_calc):
        """
        ATR_MULTIPLE method: highest - (ATR * multiple).

        entry=100, stop=95, highest=110 => 2R (activated).
        atr=2, mult=1.5 => trail_distance=3.
        new_stop = 110 - 3 = 107.
        """
        result = atr_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=95.0,
            highest_price=110.0,
            current_price=108.0,
            atr=2.0,
            is_long=True,
        )
        assert result == pytest.approx(107.0)

    def test_stop_never_moves_down(self, atr_calc):
        """
        If the new calculated stop is BELOW the current stop,
        the calculator should keep the current (higher) stop.

        current_stop already at 108, new ATR-based stop would be
        highest(110) - 1.5*4 = 104 => keep 108.
        """
        result = atr_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=108.0,
            highest_price=110.0,  # still 2R, activated
            current_price=109.0,
            atr=4.0,  # wider ATR => 110 - 6 = 104
            is_long=True,
        )
        assert result == 108.0

    def test_step_trailing_increments(self, step_calc):
        """
        STEP method with step_r=0.5, activation_r=1.0, min_profit_lock=0.5.

        entry=100, stop=95 => risk=5.
        At highest=110 => current_r = (110-100)/5 = 2.0R.
        steps_above_activation = int((2.0 - 1.0) / 0.5) = 2
        locked_r = 1.0 - 0.5 + (2 * 0.5) = 1.5
        new_stop = 100 + (5 * 1.5) = 107.5
        """
        result = step_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=95.0,
            highest_price=110.0,
            current_price=108.0,
            atr=2.0,
            is_long=True,
        )
        assert result == pytest.approx(107.5)

    def test_zero_risk_returns_current(self, atr_calc):
        """
        When entry_price == original_stop (risk=0), the calculator
        should return current_stop without crashing (no division by zero).
        """
        result = atr_calc.calculate(
            entry_price=100.0,
            original_stop=100.0,  # risk = 0
            current_stop=98.0,
            highest_price=110.0,
            current_price=108.0,
            atr=2.0,
            is_long=True,
        )
        assert result == 98.0

    def test_chandelier_exit(self, chandelier_calc):
        """
        CHANDELIER method: 3x ATR from highest high.

        entry=100, stop=95, highest=115 => 3R (activated).
        atr=2.0 => trail_distance = 3 * 2 = 6.
        new_stop = 115 - 6 = 109.
        """
        result = chandelier_calc.calculate(
            entry_price=100.0,
            original_stop=95.0,
            current_stop=95.0,
            highest_price=115.0,
            current_price=112.0,
            atr=2.0,
            is_long=True,
        )
        assert result == pytest.approx(109.0)

    def test_short_position_trailing(self):
        """
        For short positions (is_long=False), the stop should move DOWN
        (favorable direction for shorts), and the PERCENTAGE method
        should add the trail distance above the lowest price.

        entry=100, stop=105 => risk=5 (short: entry - stop is negative,
        abs gives 5). highest_price for shorts means lowest price = 90.
        current_r = (100 - 90) / 5 = 2.0R (activated).

        Percentage 2%: new_stop = 90 * 1.02 = 91.8
        Since 91.8 < current_stop(105), the min() picks 91.8.
        """
        config = TrailingStopConfig(
            method=TrailingMethod.PERCENTAGE,
            percentage=2.0,
            activation_r=1.0,
        )
        calc = TrailingStopCalculator(config)

        result = calc.calculate(
            entry_price=100.0,
            original_stop=105.0,
            current_stop=105.0,
            highest_price=90.0,   # For shorts, this tracks the lowest price
            current_price=92.0,
            atr=2.0,
            is_long=False,
        )
        assert result == pytest.approx(91.8)
        # Stop moved DOWN from 105 to 91.8 (favorable for short)
        assert result < 105.0
