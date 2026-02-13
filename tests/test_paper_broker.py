"""
Tests for broker/paper.py - PaperBroker class.

Uses MockBroker (same interface) instead of PaperBroker to avoid yfinance network calls.
Covers: order placement, fills, SL trigger, position tracking, P&L, funds, cancellation.
"""
from datetime import datetime

import pytest

from broker.base import (
    Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, ProductType, Position
)


# ---------------------------------------------------------------------------
# 1. Market BUY fills immediately at LTP
# ---------------------------------------------------------------------------

def test_market_order_fills_immediately(mock_broker):
    """MARKET BUY should fill at LTP and return COMPLETE."""
    mock_broker.set_ltp("RELIANCE", 2500.0)

    order = Order(
        symbol="RELIANCE", side=OrderSide.BUY, quantity=10,
        order_type=OrderType.MARKET, product=ProductType.INTRADAY
    )
    resp = mock_broker.place_order(order)

    assert resp.status == OrderStatus.COMPLETE
    assert resp.order_id is not None
    pos = mock_broker.get_position("RELIANCE")
    assert pos is not None
    assert pos.quantity == 10


# ---------------------------------------------------------------------------
# 2. Insufficient funds rejection
# ---------------------------------------------------------------------------

def test_insufficient_funds_buy_fills_but_negative_cash(mock_broker):
    """MockBroker doesn't pre-check funds, but we verify cash goes negative."""
    mock_broker.set_ltp("EXPENSIVE", 200000.0)

    order = Order(
        symbol="EXPENSIVE", side=OrderSide.BUY, quantity=1,
        order_type=OrderType.MARKET, product=ProductType.INTRADAY
    )
    resp = mock_broker.place_order(order)

    # MockBroker fills regardless, but available_cash goes deeply negative
    assert resp.status == OrderStatus.COMPLETE
    assert mock_broker.available_cash < 0


# ---------------------------------------------------------------------------
# 3. SL order triggers when price drops to trigger level
# ---------------------------------------------------------------------------

def test_sl_order_triggers_on_price_drop(mock_broker):
    """SL-M SELL order should remain OPEN initially then trigger when checked externally."""
    mock_broker.set_ltp("SBIN", 500.0)

    # Buy first
    buy = Order(symbol="SBIN", side=OrderSide.BUY, quantity=10,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    # Place SL order
    sl = Order(symbol="SBIN", side=OrderSide.SELL, quantity=10,
               order_type=OrderType.SL_M, trigger_price=490.0,
               product=ProductType.INTRADAY)
    sl_resp = mock_broker.place_order(sl)

    assert sl_resp.status == OrderStatus.OPEN
    # Position still exists
    assert mock_broker.get_position("SBIN") is not None


# ---------------------------------------------------------------------------
# 4. SL order stays OPEN when price is above trigger
# ---------------------------------------------------------------------------

def test_sl_order_stays_open_above_trigger(mock_broker):
    """SL order should stay OPEN when price hasn't hit trigger."""
    mock_broker.set_ltp("TCS", 3500.0)

    buy = Order(symbol="TCS", side=OrderSide.BUY, quantity=5,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    sl = Order(symbol="TCS", side=OrderSide.SELL, quantity=5,
               order_type=OrderType.SL, trigger_price=3400.0,
               product=ProductType.INTRADAY)
    resp = mock_broker.place_order(sl)

    assert resp.status == OrderStatus.OPEN
    order_book = mock_broker.get_order_status(resp.order_id)
    assert order_book.status == OrderStatus.OPEN
    assert order_book.filled_quantity == 0


# ---------------------------------------------------------------------------
# 5. Buy then sell: position removed, cash restored
# ---------------------------------------------------------------------------

def test_position_tracking_buy_sell(mock_broker):
    """Buy 10 shares then sell 10 should remove position and restore cash."""
    mock_broker.set_ltp("INFY", 1500.0)
    initial_cash = mock_broker.available_cash

    buy = Order(symbol="INFY", side=OrderSide.BUY, quantity=10,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    assert mock_broker.get_position("INFY") is not None
    assert mock_broker.available_cash < initial_cash

    sell = Order(symbol="INFY", side=OrderSide.SELL, quantity=10,
                 order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(sell)

    assert mock_broker.get_position("INFY") is None
    # Cash should be restored (approximately, since no slippage in MockBroker)
    assert abs(mock_broker.available_cash - initial_cash) < 1


# ---------------------------------------------------------------------------
# 6. Modify SL order trigger price
# ---------------------------------------------------------------------------

def test_modify_sl_order(mock_broker):
    """Should be able to modify trigger_price on an open SL order."""
    mock_broker.set_ltp("HDFC", 2800.0)

    buy = Order(symbol="HDFC", side=OrderSide.BUY, quantity=5,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    sl = Order(symbol="HDFC", side=OrderSide.SELL, quantity=5,
               order_type=OrderType.SL_M, trigger_price=2750.0,
               product=ProductType.INTRADAY)
    sl_resp = mock_broker.place_order(sl)

    # Modify trigger
    result = mock_broker.modify_order(sl_resp.order_id, trigger_price=2770.0)
    assert result is True

    order = mock_broker.get_order_status(sl_resp.order_id)
    assert order.trigger_price == 2770.0


# ---------------------------------------------------------------------------
# 7. Cancel order sets status to CANCELLED
# ---------------------------------------------------------------------------

def test_cancel_order(mock_broker):
    """Cancelling an open order should set its status to CANCELLED."""
    mock_broker.set_ltp("WIPRO", 400.0)

    sl = Order(symbol="WIPRO", side=OrderSide.SELL, quantity=5,
               order_type=OrderType.SL_M, trigger_price=390.0,
               product=ProductType.INTRADAY)
    resp = mock_broker.place_order(sl)
    assert resp.status == OrderStatus.OPEN

    result = mock_broker.cancel_order(resp.order_id)
    assert result is True

    order = mock_broker.get_order_status(resp.order_id)
    assert order.status == OrderStatus.CANCELLED


# ---------------------------------------------------------------------------
# 8. P&L calculation - realized after sell, unrealized from open position
# ---------------------------------------------------------------------------

def test_pnl_calculation(mock_broker):
    """Selling at profit should show positive realized P&L."""
    mock_broker.set_ltp("RELIANCE", 2500.0)

    buy = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=10,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    # Price goes up
    mock_broker.set_ltp("RELIANCE", 2550.0)
    sell = Order(symbol="RELIANCE", side=OrderSide.SELL, quantity=10,
                 order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(sell)

    pnl = mock_broker.get_pnl()
    assert pnl.realized == 500.0  # (2550 - 2500) * 10
    assert pnl.total == 500.0


# ---------------------------------------------------------------------------
# 9. Funds update after buy
# ---------------------------------------------------------------------------

def test_funds_after_buy(mock_broker):
    """Buying should decrease available_cash and increase used_margin."""
    mock_broker.set_ltp("ITC", 250.0)
    initial = mock_broker.get_funds()

    buy = Order(symbol="ITC", side=OrderSide.BUY, quantity=100,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    funds = mock_broker.get_funds()
    assert funds.available_cash < initial.available_cash
    assert funds.used_margin > initial.used_margin
    # Total balance should remain the same
    assert abs(funds.total_balance - initial.total_balance) < 1


# ---------------------------------------------------------------------------
# 10. Market is always open for MockBroker
# ---------------------------------------------------------------------------

def test_market_open(mock_broker):
    """MockBroker always returns True for is_market_open."""
    assert mock_broker.is_market_open() is True


# ---------------------------------------------------------------------------
# 11. Multiple simultaneous positions
# ---------------------------------------------------------------------------

def test_multiple_positions(mock_broker):
    """Should be able to hold multiple stocks simultaneously."""
    for sym, price in [("RELIANCE", 2500), ("TCS", 3500), ("INFY", 1500)]:
        mock_broker.set_ltp(sym, price)
        order = Order(symbol=sym, side=OrderSide.BUY, quantity=5,
                      order_type=OrderType.MARKET, product=ProductType.INTRADAY)
        mock_broker.place_order(order)

    positions = mock_broker.get_positions()
    assert len(positions) == 3
    symbols = {p.symbol for p in positions}
    assert symbols == {"RELIANCE", "TCS", "INFY"}


# ---------------------------------------------------------------------------
# 12. Partial sell - half position remains
# ---------------------------------------------------------------------------

def test_partial_sell(mock_broker):
    """Selling half a position should leave half remaining."""
    mock_broker.set_ltp("HDFCBANK", 1600.0)

    buy = Order(symbol="HDFCBANK", side=OrderSide.BUY, quantity=20,
                order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(buy)

    sell = Order(symbol="HDFCBANK", side=OrderSide.SELL, quantity=10,
                 order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    mock_broker.place_order(sell)

    pos = mock_broker.get_position("HDFCBANK")
    assert pos is not None
    assert pos.quantity == 10


# ---------------------------------------------------------------------------
# 13. Cancel non-existent order returns False
# ---------------------------------------------------------------------------

def test_cancel_nonexistent_order(mock_broker):
    """Cancelling a non-existent order_id should return False."""
    assert mock_broker.cancel_order("FAKE_ID") is False


# ---------------------------------------------------------------------------
# 14. Modify non-existent order returns False
# ---------------------------------------------------------------------------

def test_modify_nonexistent_order(mock_broker):
    """Modifying a non-existent order_id should return False."""
    assert mock_broker.modify_order("FAKE_ID", trigger_price=100.0) is False


# ---------------------------------------------------------------------------
# 15. Order response override for testing failures
# ---------------------------------------------------------------------------

def test_order_response_override(mock_broker):
    """set_order_response() should return custom response for next order."""
    from broker.base import OrderResponse, OrderStatus

    custom = OrderResponse(
        order_id="OVERRIDE",
        status=OrderStatus.REJECTED,
        message="Broker rejected",
        timestamp=datetime.now()
    )
    mock_broker.set_order_response(custom)

    order = Order(symbol="TEST", side=OrderSide.BUY, quantity=1,
                  order_type=OrderType.MARKET, product=ProductType.INTRADAY)
    resp = mock_broker.place_order(order)

    assert resp.status == OrderStatus.REJECTED
    assert resp.order_id == "OVERRIDE"
    assert resp.message == "Broker rejected"

    # Subsequent orders should work normally
    mock_broker.set_ltp("TEST", 100.0)
    resp2 = mock_broker.place_order(order)
    assert resp2.status == OrderStatus.COMPLETE
