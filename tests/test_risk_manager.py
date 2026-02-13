"""
Tests for risk/manager.py - RiskManager class.

Covers: kill switch, daily loss limit, trade validation, position sizing,
force shutdown, and daily state reset. All tests use MockBroker from conftest.py
with no network calls.
"""
from datetime import date
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. Kill switch triggers when broker balance < hard_stop
# ---------------------------------------------------------------------------

def test_kill_switch_triggers_below_hard_stop(mock_broker, risk_manager):
    """Portfolio balance (cash + margin) below 80K should trigger the kill switch."""
    # Drain cash so total_balance drops below hard_stop
    mock_broker.available_cash = 75000
    mock_broker.used_margin = 0

    allowed, reason = risk_manager.can_trade()

    assert allowed is False
    assert risk_manager.is_killed is True
    assert "75000" in reason or "hard stop" in reason.lower()


# ---------------------------------------------------------------------------
# 2. Kill switch includes unrealized P&L in portfolio value
# ---------------------------------------------------------------------------

def test_kill_switch_includes_unrealized_pnl(mock_broker, risk_manager):
    """Cash=85K but a position with -10K unrealized loss -> effective portfolio=75K < 80K."""
    # Start with 100K, add a losing position: 15K margin, ltp shows -10K unrealized
    mock_broker.add_position("SBIN", qty=100, avg_price=150.0, ltp=50.0)
    # After add_position: cash = 100000 - 15000 = 85000, margin = 15000
    # total_balance = 85000 + 15000 = 100000
    # unrealized = (50 - 150) * 100 = -10000
    # portfolio = 100000 + (-10000) = 90000 -- still above 80K

    # Need a bigger loss to actually breach 80K.
    # Let's set cash directly for a cleaner scenario.
    mock_broker.available_cash = 70000
    mock_broker.used_margin = 15000
    # total_balance = 85000, unrealized = -10000, portfolio = 75000 < 80000

    allowed, reason = risk_manager.can_trade()

    assert allowed is False
    assert risk_manager.is_killed is True


# ---------------------------------------------------------------------------
# 3. Daily loss limit blocks trading after enough realized losses
# ---------------------------------------------------------------------------

def test_daily_loss_limit_blocks_trading(mock_broker, risk_manager):
    """Accumulate realized losses exceeding 5K -> can_trade returns False."""
    # max_daily_loss = 100000 * 0.05 = 5000
    risk_manager.on_trade_complete(-2000)
    risk_manager.on_trade_complete(-2000)
    risk_manager.on_trade_complete(-1500)
    # daily_pnl = -5500, unrealized = 0, effective = -5500 > -5000 (in absolute)

    allowed, reason = risk_manager.can_trade()

    assert allowed is False
    assert "daily loss" in reason.lower() or "Daily loss" in reason


# ---------------------------------------------------------------------------
# 4. Daily loss uses internal tracking, not broker.get_pnl().realized
# ---------------------------------------------------------------------------

def test_daily_loss_uses_internal_tracking(mock_broker, risk_manager):
    """Even if broker._realized_pnl is 0, internal on_trade_complete accumulates losses."""
    # Broker realized P&L stays at 0 (simulating Upstox bug)
    assert mock_broker._realized_pnl == 0.0

    risk_manager.on_trade_complete(-3000)
    risk_manager.on_trade_complete(-2500)
    # Internal daily_pnl = -5500

    allowed, _ = risk_manager.can_trade()
    assert allowed is False
    assert risk_manager.daily_pnl == -5500.0


# ---------------------------------------------------------------------------
# 5. Daily loss includes unrealized P&L from open positions
# ---------------------------------------------------------------------------

def test_daily_loss_includes_unrealized(mock_broker, risk_manager):
    """Internal realized=-3K + unrealized=-2.5K = -5.5K > limit of -5K."""
    risk_manager.on_trade_complete(-3000)
    # Add a position with unrealized loss of -2500
    mock_broker.add_position("INFY", qty=100, avg_price=1500.0, ltp=1475.0)
    # unrealized = (1475 - 1500) * 100 = -2500
    # effective_daily_pnl = -3000 + (-2500) = -5500

    allowed, reason = risk_manager.can_trade()

    assert allowed is False
    assert "daily loss" in reason.lower() or "Daily loss" in reason


# ---------------------------------------------------------------------------
# 6. Daily state resets on a new day
# ---------------------------------------------------------------------------

def test_daily_reset_on_new_day(mock_broker, risk_manager):
    """When date changes, daily_pnl and trades_today reset to 0."""
    risk_manager.on_trade_complete(-4000)
    risk_manager.trades_today = 5
    assert risk_manager.daily_pnl == -4000

    # Simulate the next day by changing trading_date to yesterday
    fake_yesterday = date(2025, 1, 1)
    risk_manager.trading_date = fake_yesterday

    # can_trade triggers _reset_daily_state
    allowed, reason = risk_manager.can_trade()

    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.trades_today == 0
    assert risk_manager.trading_date == date.today()
    assert allowed is True


# ---------------------------------------------------------------------------
# 7. validate_trade rejects a stop loss that is too wide (> 3%)
# ---------------------------------------------------------------------------

def test_validate_trade_rejects_wide_stop(mock_broker, risk_manager):
    """Stop loss distance > 3% of entry should be rejected."""
    entry = 1000.0
    # 5% distance: stop at 950
    stop = 950.0

    result = risk_manager.validate_trade("RELIANCE", entry, stop, "BUY")

    assert result.allowed is False
    assert "wide" in result.reason.lower() or "too wide" in result.reason.lower()


# ---------------------------------------------------------------------------
# 8. validate_trade rejects a stop loss that is too tight (< 0.5%)
# ---------------------------------------------------------------------------

def test_validate_trade_rejects_tight_stop(mock_broker, risk_manager):
    """Stop loss distance < 0.5% of entry should be rejected."""
    entry = 1000.0
    # 0.2% distance: stop at 998
    stop = 998.0

    result = risk_manager.validate_trade("RELIANCE", entry, stop, "BUY")

    assert result.allowed is False
    assert "tight" in result.reason.lower() or "too tight" in result.reason.lower()


# ---------------------------------------------------------------------------
# 9. validate_trade rejects when max positions (3) already held
# ---------------------------------------------------------------------------

def test_validate_trade_max_positions(mock_broker, risk_manager):
    """Having max open positions should block the next trade."""
    mock_broker.add_position("RELIANCE", qty=10, avg_price=2500.0)
    mock_broker.add_position("TCS", qty=5, avg_price=3500.0)
    mock_broker.add_position("INFY", qty=10, avg_price=1500.0)
    mock_broker.add_position("HDFCBANK", qty=8, avg_price=1600.0)
    mock_broker.add_position("ICICIBANK", qty=10, avg_price=1200.0)

    result = risk_manager.validate_trade("SBIN", 500.0, 490.0, "BUY")

    assert result.allowed is False
    assert "max positions" in result.reason.lower() or "Max positions" in result.reason


# ---------------------------------------------------------------------------
# 10. validate_trade rejects duplicate symbol already in portfolio
# ---------------------------------------------------------------------------

def test_validate_trade_duplicate_symbol(mock_broker, risk_manager):
    """Trying to buy a stock already in the portfolio should be rejected."""
    mock_broker.add_position("RELIANCE", qty=10, avg_price=2500.0)

    result = risk_manager.validate_trade("RELIANCE", 2510.0, 2475.0, "BUY")

    assert result.allowed is False
    assert "already" in result.reason.lower()


# ---------------------------------------------------------------------------
# 11. Position sizing respects the 2% per-trade risk cap
# ---------------------------------------------------------------------------

def test_position_sizing_risk_cap(mock_broker, risk_manager):
    """Qty should be limited so max loss <= 2% of capital (Rs.2000)."""
    entry = 500.0
    stop = 490.0  # risk per share = Rs.10, 2% distance

    qty, value = risk_manager.calculate_position_size(entry, stop)

    # max_per_trade = 100000 * 0.02 = 2000
    # qty_by_risk = 2000 / 10 = 200
    # qty_by_capital = (100000 * 0.25) / 500 = 50
    # min(200, 50, 10000) = 50 -> capped by concentration
    # But let's verify the risk-based constraint holds:
    max_loss = qty * abs(entry - stop)
    assert max_loss <= 2000 + 1  # +1 for rounding tolerance
    assert qty > 0


# ---------------------------------------------------------------------------
# 12. Position sizing respects the 25% capital concentration cap
# ---------------------------------------------------------------------------

def test_position_sizing_concentration_cap(mock_broker, risk_manager):
    """Position value should not exceed 30% of available cash."""
    entry = 100.0
    stop = 99.0  # risk per share = Rs.1

    qty, value = risk_manager.calculate_position_size(entry, stop)

    # qty_by_risk = 2000 / 1 = 2000
    # qty_by_capital = (100000 * 0.30) / 100 = 300
    # min(2000, 300, 10000) = 300
    assert qty == 300
    assert value == 300 * 100.0
    assert value <= 100000 * 0.30 + 1


# ---------------------------------------------------------------------------
# 13. Position sizing caps at 10000 shares for penny stocks
# ---------------------------------------------------------------------------

def test_position_sizing_penny_stock_cap(mock_broker, risk_manager):
    """Very cheap stocks should be capped at 10000 shares max."""
    entry = 1.0
    stop = 0.99  # risk per share = Rs.0.01

    qty, value = risk_manager.calculate_position_size(entry, stop)

    # qty_by_risk = 2000 / 0.01 = 200000
    # qty_by_capital = (100000 * 0.25) / 1.0 = 25000
    # min(200000, 25000, 10000) = 10000
    assert qty == 10000
    assert value == 10000 * 1.0


# ---------------------------------------------------------------------------
# 14. on_trade_complete accumulates daily P&L across multiple trades
# ---------------------------------------------------------------------------

def test_on_trade_complete_tracks_pnl(mock_broker, risk_manager):
    """Multiple trade completions should accumulate in daily_pnl."""
    risk_manager.on_trade_complete(500)
    assert risk_manager.daily_pnl == 500.0
    assert risk_manager.trades_today == 1

    risk_manager.on_trade_complete(-200)
    assert risk_manager.daily_pnl == 300.0
    assert risk_manager.trades_today == 2

    risk_manager.on_trade_complete(100)
    assert risk_manager.daily_pnl == 400.0
    assert risk_manager.trades_today == 3


# ---------------------------------------------------------------------------
# 15. force_shutdown sets is_killed and blocks further trading
# ---------------------------------------------------------------------------

def test_force_shutdown_kills(mock_broker, risk_manager):
    """force_shutdown should set is_killed=True and block can_trade."""
    assert risk_manager.is_killed is False

    risk_manager.force_shutdown("manual emergency stop")

    assert risk_manager.is_killed is True
    assert risk_manager.kill_reason == "manual emergency stop"

    allowed, reason = risk_manager.can_trade()
    assert allowed is False
    assert "KILLED" in reason
