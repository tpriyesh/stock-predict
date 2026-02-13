"""
Shared test fixtures for stock-predict tests.

Provides MockBroker (no network), PaperBroker, RiskManager, TradeDB, sample data.
"""
import sys
import os
import tempfile
from datetime import datetime, date, time as dt_time
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch env BEFORE importing project modules so CONFIG doesn't hit missing .env
os.environ.setdefault("INITIAL_CAPITAL", "100000")
os.environ.setdefault("HARD_STOP_LOSS", "80000")
os.environ.setdefault("TRADING_MODE", "paper")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "")
os.environ.setdefault("TELEGRAM_CHAT_ID", "")
# Force test-specific config overrides (override .env values)
os.environ["MAX_POSITIONS"] = "5"
os.environ["MAX_POSITION_PCT"] = "0.30"
os.environ["BASE_POSITION_PCT"] = "0.25"
os.environ["ADX_STRONG_TREND"] = "25"
os.environ["MIN_CONFIDENCE"] = "0.65"
os.environ["MIN_RISK_REWARD"] = "1.8"
os.environ["TRAILING_DISTANCE_PCT"] = "0.008"
os.environ["ESTIMATED_FEE_PCT"] = "0.0005"
os.environ["SIGNAL_MAX_AGE_SECONDS"] = "1800"
os.environ["GAP_DOWN_REJECT_PCT"] = "0.05"
os.environ["CIRCUIT_BREAKER_WARN_PCT"] = "0.08"
os.environ["MIN_VOLUME_CR"] = "10.0"
os.environ["EXIT_MAX_RETRIES"] = "5"
os.environ["EXIT_RETRY_BASE_DELAY"] = "0.01"  # Fast for tests
os.environ["ENTRY_ORDER_WAIT_SECONDS"] = "0"  # No wait in tests
os.environ["MAX_COMPLETED_TRADES_IN_MEMORY"] = "200"

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)


# ============================================
# MOCK BROKER (no network, fully controllable)
# ============================================

class MockBroker(BaseBroker):
    """
    Controllable mock broker for unit tests.
    No yfinance calls, no network, instant responses.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.used_margin = 0.0
        self._connected = True
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderBook] = {}
        self._ltp_prices: Dict[str, float] = {}
        self._next_order_id = 1
        self._order_response_override: Optional[OrderResponse] = None
        self._realized_pnl = 0.0

    def set_ltp(self, symbol: str, price: float):
        self._ltp_prices[symbol] = price

    def add_position(self, symbol: str, qty: int, avg_price: float, ltp: float = None):
        ltp = ltp or avg_price
        pnl = (ltp - avg_price) * qty
        self._positions[symbol] = Position(
            symbol=symbol, quantity=qty, average_price=avg_price,
            last_price=ltp, pnl=pnl, pnl_pct=(pnl / (avg_price * qty)) * 100 if avg_price * qty > 0 else 0,
            product=ProductType.INTRADAY, value=qty * ltp
        )
        self.available_cash -= avg_price * qty
        self.used_margin += avg_price * qty

    def set_order_response(self, response: OrderResponse):
        """Override next place_order response (for testing failures)."""
        self._order_response_override = response

    def set_partial_fill(self, fill_qty: int):
        """Next market order will partially fill with this quantity."""
        self._partial_fill_qty = fill_qty

    def set_quote_ohlc(self, symbol: str, open_price: float, close_price: float):
        """Set open/close for gap-down and circuit breaker tests."""
        if not hasattr(self, '_quote_overrides'):
            self._quote_overrides = {}
        self._quote_overrides[symbol] = {'open': open_price, 'close': close_price}

    def set_funds_error(self, should_error: bool = True):
        """Make get_funds() raise exception (for kill switch API failure test)."""
        self._funds_error = should_error

    # --- BaseBroker interface ---

    def connect(self) -> bool:
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        self._connected = False

    def get_ltp(self, symbol: str) -> float:
        return self._ltp_prices.get(symbol, 100.0)

    def get_quote(self, symbol: str) -> Quote:
        ltp = self.get_ltp(symbol)
        overrides = getattr(self, '_quote_overrides', {}).get(symbol, {})
        open_price = overrides.get('open', ltp)
        close_price = overrides.get('close', ltp)
        return Quote(
            symbol=symbol, ltp=ltp, open=open_price, high=ltp * 1.02,
            low=ltp * 0.98, close=close_price, volume=100000,
            bid=ltp * 0.999, ask=ltp * 1.001, timestamp=datetime.now()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    def place_order(self, order: Order) -> OrderResponse:
        if self._order_response_override:
            resp = self._order_response_override
            self._order_response_override = None
            return resp

        oid = f"ORD{self._next_order_id:04d}"
        self._next_order_id += 1
        ltp = self.get_ltp(order.symbol)

        if order.order_type in (OrderType.SL, OrderType.SL_M):
            # SL orders stay OPEN
            self._orders[oid] = OrderBook(
                order_id=oid, symbol=order.symbol, side=order.side,
                quantity=order.quantity, filled_quantity=0,
                pending_quantity=order.quantity, order_type=order.order_type,
                price=order.price, trigger_price=order.trigger_price,
                average_price=None, status=OrderStatus.OPEN,
                placed_at=datetime.now(), updated_at=datetime.now()
            )
            return OrderResponse(oid, OrderStatus.OPEN, "SL order placed", datetime.now())

        # MARKET orders fill immediately
        fill_price = ltp
        if order.side == OrderSide.BUY:
            self.available_cash -= fill_price * order.quantity
            self.used_margin += fill_price * order.quantity
            self._positions[order.symbol] = Position(
                symbol=order.symbol, quantity=order.quantity,
                average_price=fill_price, last_price=fill_price,
                pnl=0, pnl_pct=0, product=ProductType.INTRADAY,
                value=fill_price * order.quantity
            )
        elif order.side == OrderSide.SELL:
            pos = self._positions.get(order.symbol)
            if pos:
                pnl = (fill_price - pos.average_price) * order.quantity
                self._realized_pnl += pnl
                self.available_cash += fill_price * order.quantity
                self.used_margin -= pos.average_price * order.quantity
                remaining = pos.quantity - order.quantity
                if remaining <= 0:
                    del self._positions[order.symbol]
                else:
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol, quantity=remaining,
                        average_price=pos.average_price, last_price=fill_price,
                        pnl=0, pnl_pct=0, product=ProductType.INTRADAY,
                        value=remaining * fill_price
                    )

        self._orders[oid] = OrderBook(
            order_id=oid, symbol=order.symbol, side=order.side,
            quantity=order.quantity, filled_quantity=order.quantity,
            pending_quantity=0, order_type=order.order_type,
            price=order.price, trigger_price=order.trigger_price,
            average_price=fill_price, status=OrderStatus.COMPLETE,
            placed_at=datetime.now(), updated_at=datetime.now()
        )
        return OrderResponse(oid, OrderStatus.COMPLETE, "Filled", datetime.now())

    def modify_order(self, order_id, price=None, trigger_price=None, quantity=None) -> bool:
        if order_id not in self._orders:
            return False
        ob = self._orders[order_id]
        if trigger_price is not None:
            ob.trigger_price = trigger_price
        if price is not None:
            ob.price = price
        return True

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> Optional[OrderBook]:
        return self._orders.get(order_id)

    def get_order_book(self) -> List[OrderBook]:
        return list(self._orders.values())

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_funds(self) -> Funds:
        if getattr(self, '_funds_error', False):
            raise Exception("Broker API unavailable")
        return Funds(
            available_cash=round(self.available_cash, 2),
            used_margin=round(self.used_margin, 2),
            total_balance=round(self.available_cash + self.used_margin, 2)
        )

    def get_pnl(self) -> PnL:
        unrealized = sum(p.pnl for p in self._positions.values())
        return PnL(
            realized=round(self._realized_pnl, 2),
            unrealized=round(unrealized, 2),
            total=round(self._realized_pnl + unrealized, 2)
        )

    def is_market_open(self) -> bool:
        return True

    def get_holidays(self) -> List[date]:
        return []


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def mock_broker():
    """MockBroker with ₹1L capital, no network calls."""
    return MockBroker(initial_capital=100000)


@pytest.fixture
def risk_manager(mock_broker):
    """RiskManager wired to MockBroker."""
    from risk.manager import RiskManager
    return RiskManager(broker=mock_broker, initial_capital=100000, hard_stop=80000)


@pytest.fixture
def trade_db(tmp_path):
    """Fresh TradeDB using temp file (auto-cleaned)."""
    from utils.trade_db import TradeDB
    db_path = tmp_path / "test_trades.db"
    return TradeDB(db_path=db_path)


@pytest.fixture
def sample_ohlcv_df():
    """100-row DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3
    volume = np.random.randint(100000, 1000000, n)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(float),
    }, index=dates)
    return df


@pytest.fixture
def flat_market_df():
    """20-row DataFrame where high == low == open == close (zero volatility)."""
    n = 20
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    price = np.full(n, 100.0)
    return pd.DataFrame({
        'open': price,
        'high': price,
        'low': price,
        'close': price,
        'volume': np.full(n, 50000.0),
    }, index=dates)


@pytest.fixture
def sample_trade_signal():
    """A known TradeSignal for testing."""
    from agent.signal_adapter import TradeSignal, TradeDecision
    return TradeSignal(
        symbol='RELIANCE',
        decision=TradeDecision.BUY,
        confidence=0.72,
        current_price=2500.0,
        entry_price=2500.0,
        stop_loss=2450.0,
        target_price=2600.0,
        risk_reward_ratio=2.0,
        atr_pct=1.8,
        position_size_pct=0.15,
        reasons=['[TA] RSI oversold bounce', '[FILTER] Supertrend confirmed uptrend'],
    )
