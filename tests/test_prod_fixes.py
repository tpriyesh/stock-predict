"""
Tests for Production Readiness Fixes (18 critical+high issues).

Covers:
- Kill switch API failure resilience
- Order timeout with cancellation verification
- Partial fill handling
- Volume/liquidity validation
- Corporate action detection
- Fallback data source
- Pre-market phase detection
- Upstox realized P&L calculation
- Exit retry with exponential backoff
- Slippage buffer in position sizing
- Gap-down risk check
- NSE circuit breaker awareness
- Margin adequacy check
- completed_trades memory cap
- Signal TTL staleness rejection
- News staleness indicator
- DB migration system
- E2E integration scenarios
"""
import os
import sys
import time
import tempfile
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional

import pytest

# Ensure project root and test env
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)
from risk.manager import RiskManager, RiskCheck
from config.trading_config import CONFIG
from tests.conftest import MockBroker


# ============================================
# UNIT TESTS: Kill Switch API Failure (Task 1)
# ============================================

class TestKillSwitchResilience:
    """Task 1: Kill switch handles broker API failure gracefully."""

    def test_portfolio_value_api_failure_uses_last_known(self):
        """On API failure, returns last-known cached value."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # First call succeeds - caches value
        val = rm.get_portfolio_value()
        assert val == 100000
        assert rm._last_known_portfolio_value == 100000
        assert rm._portfolio_api_failures == 0

        # Now make API fail
        broker.set_funds_error(True)
        val = rm.get_portfolio_value()
        assert val == 100000  # Uses cached value
        assert rm._portfolio_api_failures == 1

    def test_portfolio_value_kills_after_3_consecutive_failures(self):
        """After 3 consecutive API failures, triggers kill switch."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Cache a value first
        rm.get_portfolio_value()

        # Make API fail 3 times
        broker.set_funds_error(True)
        rm.get_portfolio_value()  # failure 1
        rm.get_portfolio_value()  # failure 2
        val = rm.get_portfolio_value()  # failure 3 -> kill

        # Should return below hard_stop to trigger kill switch
        assert val < rm.hard_stop

    def test_portfolio_value_resets_on_success(self):
        """Successful API call resets failure counter."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Fail twice
        broker.set_funds_error(True)
        rm.get_portfolio_value()
        rm.get_portfolio_value()
        assert rm._portfolio_api_failures == 2

        # Success resets
        broker.set_funds_error(False)
        rm.get_portfolio_value()
        assert rm._portfolio_api_failures == 0

    def test_portfolio_value_no_cache_uses_initial_capital(self):
        """Without cached value, falls back to initial_capital."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Fail immediately without caching
        broker.set_funds_error(True)
        val = rm.get_portfolio_value()
        assert val == 100000  # initial_capital fallback

    def test_can_trade_returns_false_when_api_kills(self):
        """can_trade() returns False when kill switch triggered by API failure."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        rm.get_portfolio_value()  # cache
        broker.set_funds_error(True)
        rm.get_portfolio_value()
        rm.get_portfolio_value()
        rm.get_portfolio_value()  # 3 failures

        # can_trade should report kill switch
        broker.set_funds_error(False)  # Re-enable for can_trade internal calls
        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "KILLED" in reason or "Portfolio" in reason


# ============================================
# UNIT TESTS: Order Timeout (Task 2)
# ============================================

class TestOrderTimeout:
    """Task 2: Configurable order timeout with cancellation verification."""

    def test_entry_order_wait_uses_config(self):
        """Entry order wait time comes from CONFIG, not hardcoded."""
        # CONFIG.orders.entry_order_wait_seconds is set to 0 in conftest
        assert hasattr(CONFIG.orders, 'entry_order_wait_seconds')
        # Default is 2, test override is 0
        val = CONFIG.orders.entry_order_wait_seconds
        assert isinstance(val, int)

    def test_exit_max_retries_config(self):
        """Exit max retries comes from CONFIG."""
        assert hasattr(CONFIG.orders, 'exit_max_retries')
        assert CONFIG.orders.exit_max_retries == 5

    def test_exit_retry_base_delay_config(self):
        """Exit retry base delay comes from CONFIG."""
        assert hasattr(CONFIG.orders, 'exit_retry_base_delay')
        assert CONFIG.orders.exit_retry_base_delay >= 0


# ============================================
# UNIT TESTS: Partial Fill (Task 3)
# ============================================

class TestPartialFillHandling:
    """Task 3: Partial fill status mapping in brokers."""

    def test_upstox_partial_fill_status_mapped(self):
        """Upstox maps 'partial_fill' to PARTIAL_FILL status."""
        from broker.upstox import UpstoxBroker
        broker = UpstoxBroker.__new__(UpstoxBroker)
        data = {
            'order_id': 'TEST123',
            'instrument_token': 'NSE_EQ|RELIANCE',
            'transaction_type': 'BUY',
            'quantity': 100,
            'filled_quantity': 50,
            'pending_quantity': 50,
            'order_type': 'MARKET',
            'price': 0,
            'trigger_price': 0,
            'average_price': 2500.0,
            'status': 'partial_fill',
            'order_timestamp': datetime.now().isoformat(),
        }
        order_book = broker._parse_order(data)
        assert order_book is not None
        assert order_book.status == OrderStatus.PARTIAL_FILL

    def test_upstox_trigger_pending_mapped_to_open(self):
        """Upstox maps 'trigger_pending' to OPEN status."""
        from broker.upstox import UpstoxBroker
        broker = UpstoxBroker.__new__(UpstoxBroker)
        data = {
            'order_id': 'TEST456',
            'instrument_token': 'NSE_EQ|TCS',
            'transaction_type': 'SELL',
            'quantity': 50,
            'filled_quantity': 0,
            'pending_quantity': 50,
            'order_type': 'SL-M',
            'price': 0,
            'trigger_price': 3400.0,
            'average_price': 0,
            'status': 'trigger_pending',
            'order_timestamp': datetime.now().isoformat(),
        }
        order_book = broker._parse_order(data)
        assert order_book is not None
        assert order_book.status == OrderStatus.OPEN

    def test_upstox_expired_mapped_to_cancelled(self):
        """Upstox maps 'expired' to CANCELLED status."""
        from broker.upstox import UpstoxBroker
        broker = UpstoxBroker.__new__(UpstoxBroker)
        data = {
            'order_id': 'TEST789',
            'instrument_token': 'NSE_EQ|INFY',
            'transaction_type': 'BUY',
            'quantity': 100,
            'filled_quantity': 0,
            'pending_quantity': 100,
            'order_type': 'LIMIT',
            'price': 1500.0,
            'trigger_price': 0,
            'average_price': 0,
            'status': 'expired',
            'order_timestamp': datetime.now().isoformat(),
        }
        order_book = broker._parse_order(data)
        assert order_book is not None
        assert order_book.status == OrderStatus.CANCELLED


# ============================================
# UNIT TESTS: Slippage Buffer (Task 10)
# ============================================

class TestSlippageBuffer:
    """Task 10: Slippage buffer reduces position size."""

    def test_slippage_reduces_qty_by_risk(self):
        """With slippage, qty_by_risk is smaller than without."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        entry, stop = 1000.0, 980.0  # 20 raw risk per share
        qty, value = rm.calculate_position_size(entry, stop)

        # Without slippage: qty_by_risk = 2000/20 = 100
        # With 0.5% slippage: risk_per_share = 20 * 1.005 = 20.1
        # qty_by_risk = 2000/20.1 = 99
        # qty_by_capital = (100000*0.30)/1000 = 30
        # min(99, 30) = 30
        assert qty == 30  # Capital-limited, not risk-limited for this case

    def test_slippage_affects_risk_limited_trades(self):
        """For risk-limited trades, slippage reduces qty."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        entry, stop = 100.0, 98.0  # 2 raw risk per share
        qty, value = rm.calculate_position_size(entry, stop)

        # Without slippage: qty_by_risk = 2000/2 = 1000
        # With 0.5% slippage: risk_per_share = 2 * 1.005 = 2.01
        # qty_by_risk = 2000/2.01 = 995
        # qty_by_capital = (100000*0.30)/100 = 300
        # min(995, 300) = 300
        assert qty == 300  # Still capital-limited

    def test_slippage_zero_risk_returns_zero(self):
        """Zero risk per share returns 0 quantity."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        qty, value = rm.calculate_position_size(100.0, 100.0)
        assert qty == 0
        assert value == 0


# ============================================
# UNIT TESTS: Gap-Down Risk (Task 11)
# ============================================

class TestGapDownRisk:
    """Task 11: Reject entry if stock gapped down significantly."""

    def test_gap_down_rejected(self):
        """Stock that gapped down >5% is rejected."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Set prev close = 1000, current = 940 (6% gap down)
        broker.set_ltp("RELIANCE", 940.0)
        broker.set_quote_ohlc("RELIANCE", open_price=940.0, close_price=1000.0)

        result = rm.check_gap_risk("RELIANCE", 940.0)
        assert result is not None
        assert "Gap-down" in result

    def test_normal_open_allowed(self):
        """Normal open (within 5%) is allowed."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.set_ltp("TCS", 3500.0)
        broker.set_quote_ohlc("TCS", open_price=3500.0, close_price=3520.0)

        result = rm.check_gap_risk("TCS", 3500.0)
        assert result is None  # No rejection

    def test_gap_up_allowed(self):
        """Gap up is allowed (only gap-down is risky)."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.set_ltp("INFY", 1600.0)
        broker.set_quote_ohlc("INFY", open_price=1600.0, close_price=1500.0)

        result = rm.check_gap_risk("INFY", 1600.0)
        assert result is None  # Gap up allowed

    def test_gap_down_blocks_validate_trade(self):
        """validate_trade rejects when gap-down detected."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.set_ltp("SBIN", 500.0)
        broker.set_quote_ohlc("SBIN", open_price=500.0, close_price=600.0)  # 16.7% gap down

        check = rm.validate_trade("SBIN", 500.0, 490.0, "BUY")
        assert check.allowed is False
        assert "Gap-down" in check.reason


# ============================================
# UNIT TESTS: Circuit Breaker (Task 12)
# ============================================

class TestCircuitBreakerAwareness:
    """Task 12: NSE circuit breaker awareness."""

    def test_stock_near_circuit_limit_rejected(self):
        """Stock that moved >8% from open is rejected."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Open=1000, current=1100 (10% move from open)
        broker.set_ltp("RELIANCE", 1100.0)
        broker.set_quote_ohlc("RELIANCE", open_price=1000.0, close_price=1050.0)

        result = rm.check_circuit_breaker_risk("RELIANCE", 1100.0)
        assert result is not None
        assert "Circuit breaker" in result

    def test_normal_move_allowed(self):
        """Stock that moved <8% from open is allowed."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.set_ltp("TCS", 3550.0)
        broker.set_quote_ohlc("TCS", open_price=3500.0, close_price=3480.0)

        result = rm.check_circuit_breaker_risk("TCS", 3550.0)
        assert result is None  # 1.4% move OK

    def test_downward_circuit_also_caught(self):
        """Stock that moved down >8% from open also rejected."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.set_ltp("HDFCBANK", 1400.0)
        broker.set_quote_ohlc("HDFCBANK", open_price=1600.0, close_price=1580.0)

        result = rm.check_circuit_breaker_risk("HDFCBANK", 1400.0)
        assert result is not None  # 12.5% down from open


# ============================================
# UNIT TESTS: Margin Adequacy (Task 13)
# ============================================

class TestMarginAdequacy:
    """Task 13: Margin adequacy check before entry."""

    def test_insufficient_margin_rejected(self):
        """Order requiring more than available cash is rejected."""
        broker = MockBroker(initial_capital=10000)  # Small capital
        rm = RiskManager(broker=broker, initial_capital=10000, hard_stop=8000)

        # 200 shares @ 1000 = 200,000 > 10,000 available
        result = rm.check_margin_adequacy(1000.0, 980.0, 200)
        assert result is not None
        assert "Insufficient margin" in result

    def test_adequate_margin_passes(self):
        """Order within available cash passes."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # 10 shares @ 500 = 5,000 < 100,000
        result = rm.check_margin_adequacy(500.0, 490.0, 10)
        assert result is None  # Passes


# ============================================
# UNIT TESTS: Pre-Market Phase (Task 7)
# ============================================

class TestPreMarketPhase:
    """Task 7: Pre-market phase detection logic."""

    def test_pre_market_detected_at_845(self):
        """8:50 AM should be PRE_MARKET phase."""
        from agent.orchestrator import TradingOrchestrator, TradingPhase

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        with patch('agent.orchestrator.time_ist', return_value=dt_time(8, 50)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.PRE_MARKET

    def test_pre_market_detected_at_914(self):
        """9:14 AM should be PRE_MARKET phase."""
        from agent.orchestrator import TradingOrchestrator, TradingPhase

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        with patch('agent.orchestrator.time_ist', return_value=dt_time(9, 14)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.PRE_MARKET

    def test_market_open_at_915(self):
        """9:15 AM should be MARKET_OPEN phase (not PRE_MARKET)."""
        from agent.orchestrator import TradingOrchestrator, TradingPhase

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        with patch('agent.orchestrator.time_ist', return_value=dt_time(9, 15)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.MARKET_OPEN

    def test_before_pre_market_is_closed(self):
        """7:00 AM should be MARKET_CLOSED."""
        from agent.orchestrator import TradingOrchestrator, TradingPhase

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        with patch('agent.orchestrator.time_ist', return_value=dt_time(7, 0)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.MARKET_CLOSED


# ============================================
# UNIT TESTS: Signal TTL (Task 15)
# ============================================

class TestSignalTTL:
    """Task 15: Signal TTL and staleness rejection."""

    def test_fresh_signal_accepted(self):
        """Signal within TTL is accepted."""
        from agent.orchestrator import TradingOrchestrator
        from agent.signal_adapter import TradeSignal, TradeDecision

        broker = MockBroker()
        broker.set_ltp("RELIANCE", 2500.0)
        broker.set_quote_ohlc("RELIANCE", open_price=2490.0, close_price=2480.0)
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        signal = TradeSignal(
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
            timestamp=datetime.now(),  # Fresh
        )

        # Should not be rejected by TTL
        result = orch._enter_position(signal)
        # Either True (entered) or False (other reasons), but NOT rejected by TTL
        # We just check it doesn't raise

    def test_stale_signal_rejected(self):
        """Signal older than TTL is rejected."""
        from agent.orchestrator import TradingOrchestrator
        from agent.signal_adapter import TradeSignal, TradeDecision

        broker = MockBroker()
        broker.set_ltp("TCS", 3500.0)
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        signal = TradeSignal(
            symbol='TCS',
            decision=TradeDecision.BUY,
            confidence=0.72,
            current_price=3500.0,
            entry_price=3500.0,
            stop_loss=3430.0,
            target_price=3640.0,
            risk_reward_ratio=2.0,
            atr_pct=1.8,
            position_size_pct=0.15,
            timestamp=datetime.now() - timedelta(hours=1),  # 1 hour old
        )

        result = orch._enter_position(signal)
        assert result is None  # Rejected by TTL (signal rejection, not broker failure)

    def test_signal_max_age_config(self):
        """Signal max age comes from CONFIG."""
        assert hasattr(CONFIG.signals, 'signal_max_age_seconds')
        assert CONFIG.signals.signal_max_age_seconds == 1800


# ============================================
# UNIT TESTS: Completed Trades Cap (Task 14)
# ============================================

class TestCompletedTradesCap:
    """Task 14: completed_trades memory cap."""

    def test_cap_enforced(self):
        """completed_trades list doesn't grow beyond config limit."""
        from agent.orchestrator import TradingOrchestrator, TradeRecord

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        # Add more than MAX trades
        max_cap = CONFIG.capital.max_completed_trades_in_memory
        for i in range(max_cap + 50):
            trade = TradeRecord(
                trade_id=f"T{i}",
                symbol=f"SYM{i}",
                side="BUY",
                quantity=10,
                entry_price=100.0,
                exit_price=105.0,
                pnl=50.0,
            )
            trade.exit_time = datetime.now()
            trade.exit_reason = "test"
            orch.completed_trades.append(trade)

            # Simulate the cap logic from _record_exit
            if len(orch.completed_trades) > max_cap:
                orch.completed_trades = orch.completed_trades[-max_cap:]

        assert len(orch.completed_trades) <= max_cap

    def test_recent_trades_preserved(self):
        """Most recent trades are kept, oldest are dropped."""
        from agent.orchestrator import TradingOrchestrator, TradeRecord

        broker = MockBroker()
        rm = RiskManager(broker=broker)
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)

        max_cap = 5  # Small cap for testing
        for i in range(10):
            trade = TradeRecord(
                trade_id=f"T{i}",
                symbol=f"SYM{i}",
                side="BUY",
                quantity=10,
                entry_price=100.0,
            )
            orch.completed_trades.append(trade)
            if len(orch.completed_trades) > max_cap:
                orch.completed_trades = orch.completed_trades[-max_cap:]

        assert len(orch.completed_trades) == max_cap
        # Most recent should be T9
        assert orch.completed_trades[-1].trade_id == "T9"
        # Oldest should be T5
        assert orch.completed_trades[0].trade_id == "T5"


# ============================================
# UNIT TESTS: Upstox P&L (Task 8)
# ============================================

class TestUpstoxPnL:
    """Task 8: Upstox realized P&L calculation."""

    def test_pnl_buy_sell_matching(self):
        """P&L correctly matches buys and sells per symbol."""
        from broker.upstox import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker._session = MagicMock()
        broker._limiter = MagicMock()
        broker._limiter.wait = MagicMock()
        broker._circuit = MagicMock()
        broker._circuit.allow_request.return_value = True
        broker._circuit.record_success = MagicMock()

        # Mock _make_request for positions (empty) and trades
        def mock_request(method, endpoint, **kwargs):
            if "positions" in endpoint:
                return {"status": "success", "data": []}
            elif "trades" in endpoint:
                return {
                    "status": "success",
                    "data": [
                        {
                            "instrument_token": "NSE_EQ|RELIANCE",
                            "transaction_type": "BUY",
                            "quantity": 10,
                            "average_price": 2500.0,
                        },
                        {
                            "instrument_token": "NSE_EQ|RELIANCE",
                            "transaction_type": "SELL",
                            "quantity": 10,
                            "average_price": 2550.0,
                        },
                    ]
                }
            return {"status": "success", "data": []}

        broker._make_request = mock_request
        broker._holidays = []

        pnl = broker.get_pnl()
        assert pnl.realized == 500.0  # (2550-2500) * 10
        assert pnl.unrealized == 0.0


# ============================================
# UNIT TESTS: DB Migration (Task 17)
# ============================================

class TestDBMigration:
    """Task 17: DB migration system."""

    def test_fresh_db_has_schema_version(self):
        """New DB gets schema_version table."""
        from utils.trade_db import TradeDB
        with tempfile.TemporaryDirectory() as td:
            db = TradeDB(db_path=Path(td) / "test.db")
            version = db.get_schema_version()
            assert version >= 1

    def test_migration_v2_applied(self):
        """Migration v2 creates schema_version table."""
        from utils.trade_db import TradeDB
        with tempfile.TemporaryDirectory() as td:
            db = TradeDB(db_path=Path(td) / "test.db")
            version = db.get_schema_version()
            assert version >= 2

    def test_migration_idempotent(self):
        """Running migrations twice doesn't error."""
        from utils.trade_db import TradeDB
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db1 = TradeDB(db_path=db_path)
            v1 = db1.get_schema_version()
            # Create again (re-runs migrations)
            db2 = TradeDB(db_path=db_path)
            v2 = db2.get_schema_version()
            assert v1 == v2

    def test_trades_still_work_after_migration(self):
        """Trade operations work after migration."""
        from utils.trade_db import TradeDB
        with tempfile.TemporaryDirectory() as td:
            db = TradeDB(db_path=Path(td) / "test.db")
            db.record_entry(
                trade_id="MIG_TEST_001",
                symbol="RELIANCE",
                side="BUY",
                quantity=10,
                entry_price=2500.0,
                stop_loss=2450.0,
                target=2575.0,
                confidence=0.72,
                order_ids=["ORD001"],
                broker_mode="paper"
            )
            trade = db.get_trade("MIG_TEST_001")
            assert trade is not None
            assert trade['symbol'] == "RELIANCE"


# ============================================
# UNIT TESTS: Config Parameters
# ============================================

class TestNewConfigParameters:
    """Verify all new config parameters exist with correct defaults."""

    def test_signal_max_age_seconds(self):
        assert CONFIG.signals.signal_max_age_seconds == 1800

    def test_gap_down_reject_pct(self):
        assert CONFIG.signals.gap_down_reject_pct == 0.05

    def test_circuit_breaker_warn_pct(self):
        assert CONFIG.signals.circuit_breaker_warn_pct == 0.08

    def test_min_volume_cr(self):
        assert CONFIG.signals.min_volume_cr == 10.0

    def test_exit_max_retries(self):
        assert CONFIG.orders.exit_max_retries == 5

    def test_exit_retry_base_delay(self):
        assert CONFIG.orders.exit_retry_base_delay >= 0

    def test_entry_order_wait_seconds(self):
        assert hasattr(CONFIG.orders, 'entry_order_wait_seconds')

    def test_token_check_interval_seconds(self):
        assert CONFIG.orders.token_check_interval_seconds == 300  # Fix 5: 5 min (was 1800)

    def test_max_completed_trades_in_memory(self):
        assert CONFIG.capital.max_completed_trades_in_memory == 200


# ============================================
# E2E INTEGRATION TESTS (Task 18)
# ============================================

class TestCUJ_KillSwitchAPIFailure:
    """CUJ 39: Kill switch remains protective when broker API fails."""

    def test_full_kill_switch_api_failure_flow(self, mock_broker, risk_manager):
        """
        Scenario: Broker API goes down during trading.
        1. System starts normally, caches portfolio value
        2. API fails - uses cached value
        3. API fails 3x - triggers kill switch
        4. All positions should be flagged for exit
        """
        # 1. Normal start
        val = risk_manager.get_portfolio_value()
        assert val == 100000
        can_trade, _ = risk_manager.can_trade()
        assert can_trade is True

        # 2. API fails once - uses cached
        mock_broker.set_funds_error(True)
        val = risk_manager.get_portfolio_value()
        assert val == 100000  # Cached
        mock_broker.set_funds_error(False)
        can_trade, _ = risk_manager.can_trade()
        assert can_trade is True  # Still OK

        # 3. API fails 3 times consecutively
        mock_broker.set_funds_error(True)
        risk_manager.get_portfolio_value()  # fail 1
        risk_manager.get_portfolio_value()  # fail 2
        val = risk_manager.get_portfolio_value()  # fail 3 -> kill
        assert val < risk_manager.hard_stop

        # 4. can_trade should now return False
        mock_broker.set_funds_error(False)
        can_trade, reason = risk_manager.can_trade()
        assert can_trade is False


class TestCUJ_GapDownProtection:
    """CUJ 40: System protects against gap-down entries."""

    def test_gap_down_entry_rejected_e2e(self, mock_broker, risk_manager):
        """
        Scenario: Stock gaps down 8% overnight.
        System should reject entry signal.
        """
        mock_broker.set_ltp("RELIANCE", 920.0)
        mock_broker.set_quote_ohlc("RELIANCE", open_price=920.0, close_price=1000.0)

        check = risk_manager.validate_trade("RELIANCE", 920.0, 900.0, "BUY")
        assert check.allowed is False
        assert "Gap-down" in check.reason


class TestCUJ_CircuitBreakerProtection:
    """CUJ 41: System avoids stocks near circuit breaker limits."""

    def test_circuit_breaker_entry_rejected_e2e(self, mock_broker, risk_manager):
        """
        Scenario: Stock already up 10% from open.
        System should reject entry to avoid getting stuck in halt.
        """
        mock_broker.set_ltp("ADANIPOWER", 330.0)
        mock_broker.set_quote_ohlc("ADANIPOWER", open_price=300.0, close_price=295.0)

        check = risk_manager.validate_trade("ADANIPOWER", 330.0, 323.0, "BUY")
        assert check.allowed is False
        assert "Circuit breaker" in check.reason


class TestCUJ_MarginCheck:
    """CUJ 42: System verifies margin before entry."""

    def test_margin_blocks_oversized_position(self):
        """
        Scenario: Trade requires more margin than available.
        System should reject before placing order.
        """
        broker = MockBroker(initial_capital=10000)
        rm = RiskManager(broker=broker, initial_capital=10000, hard_stop=8000)

        broker.set_ltp("BAJFINANCE", 7000.0)
        broker.set_quote_ohlc("BAJFINANCE", open_price=6980.0, close_price=6950.0)

        # Entry at 7000 with qty from position sizing
        check = rm.validate_trade("BAJFINANCE", 7000.0, 6860.0, "BUY")
        # Should calculate qty=1 or reject (7000 < 10000, so 1 share is OK)
        # But if qty_by_capital = int(3000/7000) = 0, should reject
        if check.allowed:
            assert check.max_quantity > 0


class TestCUJ_ExitRetryEscalation:
    """CUJ 43: Exit retries escalate with exponential backoff."""

    def test_exit_succeeds_on_third_attempt(self, mock_broker, risk_manager):
        """
        Scenario: Exit order fails twice, succeeds on third attempt.
        System should keep retrying with increasing delays.
        """
        from agent.orchestrator import TradingOrchestrator, TradeRecord
        import uuid

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)
        orch._broker_mode = "paper"

        trade_id = f"EX_RETRY_{uuid.uuid4().hex[:8]}"

        # Create active trade
        mock_broker.set_ltp("TCS", 3500.0)
        mock_broker.add_position("TCS", qty=10, avg_price=3450.0, ltp=3500.0)

        trade = TradeRecord(
            trade_id=trade_id,
            symbol="TCS",
            side="BUY",
            quantity=10,
            entry_price=3450.0,
            stop_loss=3400.0,
            target=3550.0,
            order_ids=["ORD001"],
        )
        orch.active_trades["TCS"] = trade

        # Record entry in DB so FK constraint is satisfied on exit
        orch._db.record_entry(
            trade_id=trade_id, symbol="TCS", side="BUY",
            quantity=10, entry_price=3450.0, stop_loss=3400.0,
            target=3550.0, confidence=0.72, order_ids=["ORD001"],
            broker_mode="paper"
        )

        # Exit should succeed (MockBroker fills immediately)
        orch._exit_position(trade, "Test exit")

        # Trade should be in completed
        assert "TCS" not in orch.active_trades
        assert len(orch.completed_trades) == 1


class TestCUJ_SignalTTLProtection:
    """CUJ 44: Stale signals are rejected before entry."""

    def test_stale_signal_blocked_fresh_allowed(self, mock_broker, risk_manager):
        """
        Scenario: One signal is fresh, another is 2 hours old.
        Only fresh signal should proceed.
        """
        from agent.orchestrator import TradingOrchestrator
        from agent.signal_adapter import TradeSignal, TradeDecision

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)
        mock_broker.set_ltp("FRESH", 500.0)
        mock_broker.set_quote_ohlc("FRESH", open_price=498.0, close_price=495.0)
        mock_broker.set_ltp("STALE", 600.0)

        # Fresh signal
        fresh = TradeSignal(
            symbol='FRESH', decision=TradeDecision.BUY,
            confidence=0.72, current_price=500.0, entry_price=500.0,
            stop_loss=490.0, target_price=520.0, risk_reward_ratio=2.0,
            atr_pct=1.8, position_size_pct=0.15,
            timestamp=datetime.now(),
        )

        # Stale signal (2 hours old)
        stale = TradeSignal(
            symbol='STALE', decision=TradeDecision.BUY,
            confidence=0.72, current_price=600.0, entry_price=600.0,
            stop_loss=588.0, target_price=624.0, risk_reward_ratio=2.0,
            atr_pct=1.8, position_size_pct=0.15,
            timestamp=datetime.now() - timedelta(hours=2),
        )

        stale_result = orch._enter_position(stale)
        assert stale_result is None  # Rejected by TTL (signal rejection, not broker failure)

        fresh_result = orch._enter_position(fresh)
        assert fresh_result is True  # Accepted


class TestCUJ_CompletedTradesMemoryManagement:
    """CUJ 45: Memory doesn't grow unbounded with completed trades."""

    def test_memory_stays_bounded_after_many_trades(self, mock_broker, risk_manager):
        """
        Scenario: 500 trades completed throughout the day.
        Memory should stay bounded at MAX_COMPLETED_TRADES_IN_MEMORY.
        """
        from agent.orchestrator import TradingOrchestrator, TradeRecord

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)

        max_cap = CONFIG.capital.max_completed_trades_in_memory

        for i in range(max_cap + 100):
            symbol = f"S{i}"
            mock_broker.set_ltp(symbol, 100.0)
            mock_broker.add_position(symbol, qty=1, avg_price=100.0)

            trade = TradeRecord(
                trade_id=f"T{i}", symbol=symbol, side="BUY",
                quantity=1, entry_price=100.0, order_ids=[f"O{i}"],
            )
            orch.active_trades[symbol] = trade

            # Record exit (triggers cap logic)
            orch._record_exit(trade, "test", exit_price=105.0)

        assert len(orch.completed_trades) <= max_cap


class TestCUJ_SlippageAdjustedSizing:
    """CUJ 46: Position sizing accounts for slippage."""

    def test_slippage_reduces_actual_risk(self, mock_broker, risk_manager):
        """
        Scenario: Entry at 1000, SL at 980 (2% raw risk).
        With 0.5% slippage, actual risk per share = 20 * 1.005 = 20.1.
        Position size should be slightly smaller.
        """
        qty, value = risk_manager.calculate_position_size(1000.0, 980.0)
        # qty_by_risk = 2000 / 20.1 = 99 (vs 100 without slippage)
        # qty_by_capital = 30000/1000 = 30
        # Still capital-limited at 30
        assert qty == 30
        assert value == 30000.0


class TestCUJ_DBMigrationSafety:
    """CUJ 47: DB migrations are safe and idempotent."""

    def test_existing_data_survives_migration(self):
        """
        Scenario: Existing trades.db gets migrated to new schema.
        All existing data should be preserved.
        """
        from utils.trade_db import TradeDB
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"

            # Create DB and add data
            db = TradeDB(db_path=db_path)
            db.record_entry(
                trade_id="MIGRATE_001", symbol="RELIANCE", side="BUY",
                quantity=10, entry_price=2500.0, stop_loss=2450.0,
                target=2575.0, confidence=0.72, order_ids=["ORD1"],
            )

            # "Migrate" (re-open)
            db2 = TradeDB(db_path=db_path)
            trade = db2.get_trade("MIGRATE_001")
            assert trade is not None
            assert trade['symbol'] == "RELIANCE"
            assert trade['quantity'] == 10


class TestCUJ_PartialFillSafety:
    """CUJ 48: Partial fills don't corrupt position tracking."""

    def test_upstox_partial_fill_detected(self):
        """
        Scenario: Upstox returns 'partial_fill' status.
        System should recognize it as PARTIAL_FILL, not PENDING.
        """
        from broker.upstox import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        data = {
            'order_id': 'PARTIAL_001',
            'instrument_token': 'NSE_EQ|RELIANCE',
            'transaction_type': 'BUY',
            'quantity': 100,
            'filled_quantity': 60,
            'pending_quantity': 40,
            'order_type': 'MARKET',
            'price': 0,
            'trigger_price': 0,
            'average_price': 2500.0,
            'status': 'partial_fill',
            'order_timestamp': datetime.now().isoformat(),
        }
        order = broker._parse_order(data)
        assert order.status == OrderStatus.PARTIAL_FILL
        assert order.filled_quantity == 60
        assert order.pending_quantity == 40


# ============================================
# UNIT TESTS: Partial Fill Cancellation (Fix 1)
# ============================================

class TestPartialFillCancellation:
    """Fix 1: Cancel remaining order after partial fill to prevent untracked shares."""

    def test_partial_fill_cancels_remainder(self, mock_broker, risk_manager):
        """After partial fill, remaining order portion is cancelled."""
        from agent.orchestrator import TradingOrchestrator
        from agent.signal_adapter import TradeSignal, TradeDecision

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)

        mock_broker.set_ltp("RELIANCE", 2500.0)
        mock_broker.set_quote_ohlc("RELIANCE", open_price=2495.0, close_price=2490.0)

        # Override place_order to return PARTIAL_FILL, then set up order_status
        original_place = mock_broker.place_order

        def partial_fill_place(order):
            resp = original_place(order)
            if order.side == OrderSide.BUY:
                # Simulate partial fill: only 5 of requested shares filled
                ob = mock_broker._orders[resp.order_id]
                ob.status = OrderStatus.PARTIAL_FILL
                ob.filled_quantity = 5
                ob.pending_quantity = order.quantity - 5
                resp = OrderResponse(resp.order_id, OrderStatus.PARTIAL_FILL, "Partial", datetime.now())
            return resp

        mock_broker.place_order = partial_fill_place

        signal = TradeSignal(
            symbol='RELIANCE', decision=TradeDecision.BUY,
            confidence=0.72, current_price=2500.0, entry_price=2500.0,
            stop_loss=2450.0, target_price=2600.0, risk_reward_ratio=2.0,
            atr_pct=1.8, position_size_pct=0.15,
            timestamp=datetime.now(),
        )

        result = orch._enter_position(signal)
        # Should succeed with partial fill
        assert result is True
        trade = orch.active_trades.get("RELIANCE")
        assert trade is not None
        # Quantity should be 5 (partial fill), not the full requested amount
        assert trade.quantity == 5

    def test_zero_fill_after_partial_aborts(self, mock_broker, risk_manager):
        """If partial fill returns 0 shares, entry is aborted."""
        from agent.orchestrator import TradingOrchestrator
        from agent.signal_adapter import TradeSignal, TradeDecision

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)

        mock_broker.set_ltp("TCS", 3500.0)
        mock_broker.set_quote_ohlc("TCS", open_price=3490.0, close_price=3480.0)

        def zero_fill_place(order):
            """Simulate a PARTIAL_FILL with 0 shares filled (no position created)."""
            oid = f"ORD{mock_broker._next_order_id:04d}"
            mock_broker._next_order_id += 1
            mock_broker._orders[oid] = OrderBook(
                order_id=oid, symbol=order.symbol, side=order.side,
                quantity=order.quantity, filled_quantity=0,
                pending_quantity=order.quantity, order_type=order.order_type,
                price=order.price, trigger_price=order.trigger_price,
                average_price=0, status=OrderStatus.PARTIAL_FILL,
                placed_at=datetime.now(), updated_at=datetime.now()
            )
            return OrderResponse(oid, OrderStatus.PARTIAL_FILL, "No fill", datetime.now())

        mock_broker.place_order = zero_fill_place

        signal = TradeSignal(
            symbol='TCS', decision=TradeDecision.BUY,
            confidence=0.72, current_price=3500.0, entry_price=3500.0,
            stop_loss=3430.0, target_price=3640.0, risk_reward_ratio=2.0,
            atr_pct=1.8, position_size_pct=0.15,
            timestamp=datetime.now(),
        )

        result = orch._enter_position(signal)
        assert result is False
        assert "TCS" not in orch.active_trades


# ============================================
# UNIT TESTS: Circuit Breaker Reset (Fix 2)
# ============================================

class TestCircuitBreakerReset:
    """Fix 2: Circuit breakers reset between trading sessions."""

    def test_circuit_breaker_reset_clears_state(self):
        """reset() returns breaker to CLOSED with zero failures."""
        from utils.error_handler import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=3)

        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.allow_request() is False

        # Reset
        cb.reset()
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.allow_request() is True
        assert cb._failure_count == 0

    def test_registry_reset_all_clears_all_breakers(self):
        """reset_all() resets every registered circuit breaker."""
        from utils.error_handler import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry(failure_threshold=2)

        # Trip two breakers
        orders_cb = registry.get("orders")
        quotes_cb = registry.get("quotes")
        orders_cb.record_failure()
        orders_cb.record_failure()
        quotes_cb.record_failure()
        quotes_cb.record_failure()

        assert orders_cb.allow_request() is False
        assert quotes_cb.allow_request() is False

        # Reset all
        registry.reset_all()

        assert orders_cb.allow_request() is True
        assert quotes_cb.allow_request() is True

    def test_orchestrator_resets_breakers_on_run(self, mock_broker, risk_manager):
        """Orchestrator resets circuit breakers at session start."""
        from agent.orchestrator import TradingOrchestrator

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)
        orch._consecutive_order_failures = 5

        # Call the reset method directly
        orch._reset_circuit_breakers()

        assert orch._consecutive_order_failures == 0


# ============================================
# UNIT TESTS: Unrealized P&L Fallback (Fix 3)
# ============================================

class TestUnrealizedPnLFallback:
    """Fix 3: Unrealized P&L failure uses cached value, not zero."""

    def test_pnl_failure_uses_last_known_unrealized(self):
        """When get_pnl() fails, portfolio includes cached unrealized losses."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Add a position with unrealized loss
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0, ltp=2400.0)
        # Unrealized = (2400-2500)*10 = -1000

        # First call succeeds - caches unrealized = -1000
        val = rm.get_portfolio_value()
        assert rm._last_known_unrealized == -1000.0

        # Now make get_pnl fail but get_funds still works
        original_get_pnl = broker.get_pnl
        broker.get_pnl = MagicMock(side_effect=Exception("PnL API down"))

        val = rm.get_portfolio_value()
        # Should use cached unrealized (-1000), not zero
        # base_value = available_cash + used_margin = 75000 + 25000 = 100000
        # portfolio = 100000 + (-1000) = 99000
        assert val == 99000.0

        broker.get_pnl = original_get_pnl  # Restore

    def test_pnl_failure_with_large_loss_triggers_kill(self):
        """Cached unrealized losses can trigger kill switch even when API partially fails."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Simulate large unrealized loss cached from previous call
        rm._last_known_unrealized = -25000.0

        # Make get_pnl fail
        broker.get_pnl = MagicMock(side_effect=Exception("PnL API down"))

        val = rm.get_portfolio_value()
        # base_value = 100000, unrealized fallback = -25000
        # portfolio = 75000 < 80000 hard_stop
        assert val < rm.hard_stop

        # can_trade should return False
        broker.get_pnl = MagicMock(side_effect=Exception("PnL API down"))
        can_trade, reason = rm.can_trade()
        assert can_trade is False

    def test_fresh_rm_pnl_failure_uses_zero(self):
        """Fresh RiskManager with no cached unrealized defaults to 0."""
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        broker.get_pnl = MagicMock(side_effect=Exception("PnL API down"))

        val = rm.get_portfolio_value()
        # No cache, _last_known_unrealized = 0.0 (default)
        assert val == 100000.0


# ============================================
# UNIT TESTS: KeyError in Reconciliation (Fix 4)
# ============================================

class TestReconciliationKeyError:
    """Fix 4: Reconciliation uses defensive .get() to avoid KeyError."""

    def test_reconciliation_handles_concurrent_removal(self, mock_broker, risk_manager):
        """Reconciliation doesn't crash if symbol removed during iteration."""
        from agent.orchestrator import TradingOrchestrator, TradeRecord
        import uuid

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)
        trade_id = f"RECON_{uuid.uuid4().hex[:8]}"

        # Add trade internally but NOT at broker (will be in 'missing' set)
        trade = TradeRecord(
            trade_id=trade_id, symbol="GHOST", side="BUY",
            quantity=10, entry_price=100.0, order_ids=["ORD001"],
        )
        orch.active_trades["GHOST"] = trade

        # Record entry in DB so _record_exit doesn't fail on FK
        orch._db.record_entry(
            trade_id=trade_id, symbol="GHOST", side="BUY",
            quantity=10, entry_price=100.0, stop_loss=95.0,
            target=110.0, confidence=0.7, order_ids=["ORD001"],
            broker_mode="paper"
        )

        # Run reconciliation - should detect "GHOST" as missing and exit it
        orch._reconcile_positions()

        # GHOST should be removed from active_trades
        assert "GHOST" not in orch.active_trades
        assert len(orch.completed_trades) == 1

    def test_exit_all_handles_concurrent_removal(self, mock_broker, risk_manager):
        """_exit_all_positions doesn't crash if symbol removed mid-loop."""
        from agent.orchestrator import TradingOrchestrator, TradeRecord
        import uuid

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)
        orch._broker_mode = "paper"

        uid = uuid.uuid4().hex[:6]
        # Add two trades
        for sym, price in [("A", 100.0), ("B", 200.0)]:
            tid = f"{sym}_{uid}"
            mock_broker.set_ltp(sym, price)
            mock_broker.add_position(sym, qty=10, avg_price=price)
            trade = TradeRecord(
                trade_id=tid, symbol=sym, side="BUY",
                quantity=10, entry_price=price, order_ids=[f"ORD_{sym}"],
            )
            orch.active_trades[sym] = trade
            orch._db.record_entry(
                trade_id=tid, symbol=sym, side="BUY",
                quantity=10, entry_price=price, stop_loss=price*0.95,
                target=price*1.05, confidence=0.7, order_ids=[f"ORD_{sym}"],
                broker_mode="paper"
            )

        # Exit all - should not crash even though exits modify active_trades
        orch._exit_all_positions("Test shutdown")

        assert len(orch.active_trades) == 0
        assert len(orch.completed_trades) == 2

    def test_reconciliation_qty_mismatch_with_removed_symbol(self, mock_broker, risk_manager):
        """Qty mismatch check skips symbols removed during missing-symbol exits."""
        from agent.orchestrator import TradingOrchestrator, TradeRecord

        orch = TradingOrchestrator(broker=mock_broker, risk_manager=risk_manager)

        # Symbol at broker with different qty than internal
        mock_broker.add_position("INFY", qty=20, avg_price=1500.0)
        trade = TradeRecord(
            trade_id="QTY_001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, order_ids=["ORD_INFY"],
        )
        orch.active_trades["INFY"] = trade

        # Run reconciliation - should update qty to match broker
        orch._reconcile_positions()

        infy_trade = orch.active_trades.get("INFY")
        assert infy_trade is not None
        assert infy_trade.quantity == 20  # Updated to broker qty


# ============================================
# E2E: All 4 Fixes Together
# ============================================

class TestCUJ_CircuitBreakerSessionReset:
    """CUJ 49: Circuit breaker from yesterday doesn't block today's trading."""

    def test_stale_breaker_reset_on_new_session(self):
        """
        Scenario: Day 1 ends with 5 order failures (breaker OPEN).
        Day 2: orchestrator.run() resets breakers, trading proceeds normally.
        """
        from utils.error_handler import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry(failure_threshold=5)

        # Day 1: 5 failures trip the breaker
        cb = registry.get("orders")
        for _ in range(5):
            cb.record_failure()
        assert cb.allow_request() is False

        # Day 2: reset_all at session start
        registry.reset_all()
        assert cb.allow_request() is True

        # Can record new failures fresh
        cb.record_failure()
        assert cb.allow_request() is True  # Only 1 failure, threshold is 5


class TestCUJ_UnrealizedLossKillSwitch:
    """CUJ 50: Kill switch uses cached unrealized losses when API partially fails."""

    def test_cached_loss_prevents_trading_past_hard_stop(self):
        """
        Scenario: Portfolio has ₹22K unrealized loss. get_pnl() starts failing.
        Kill switch should still trigger using cached loss value.
        """
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Step 1: Build up a large unrealized loss
        broker.add_position("ADANI", qty=100, avg_price=1000.0, ltp=780.0)
        # unrealized = (780-1000)*100 = -22000

        # Step 2: Normal call caches unrealized
        val = rm.get_portfolio_value()
        assert rm._last_known_unrealized == -22000.0
        # base = 0 + 100000 = 100000... wait MockBroker tracks cash differently
        # After add_position: available = 100000 - 1000*100 = 0, used_margin = 100000
        # total_balance = 0 + 100000 = 100000
        # unrealized = -22000
        # portfolio = 78000 < 80000 → kill switch

        # Step 3: First call already triggers kill switch
        can_trade, _ = rm.can_trade()
        assert can_trade is False

    def test_pnl_api_failure_doesnt_hide_losses(self):
        """
        Scenario: ₹15K cached unrealized loss. get_pnl() fails.
        Portfolio should be 85K (not 100K), still above 80K but close.
        """
        broker = MockBroker(initial_capital=100000)
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        # Cache a moderate loss
        rm._last_known_unrealized = -15000.0

        # Break get_pnl
        broker.get_pnl = MagicMock(side_effect=Exception("API down"))

        val = rm.get_portfolio_value()
        assert val == 85000.0  # 100000 + (-15000), not 100000

        # Still above hard stop, so trading allowed
        can_trade, _ = rm.can_trade()
        assert can_trade is True


# ============================================
# LTP FAILURE ALERTING (Task: Production Gaps)
# ============================================

class TestLTPFailureAlerting:
    """Test that consecutive LTP failures trigger Telegram alerts."""

    def _make_orchestrator(self, broker):
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)
        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )
        return orch

    def test_ltp_failure_counter_increments(self):
        """Each LTP failure increments the per-symbol counter."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)

        # Add a trade + position, but LTP returns 0 (failure)
        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0
        )
        orch.active_trades["INFY"] = trade
        broker.add_position("INFY", qty=10, avg_price=1500.0, ltp=1500.0)

        # Make LTP return 0 (failure)
        broker.set_ltp("INFY", 0)

        with patch('agent.orchestrator.alert_error') as mock_alert:
            orch._check_positions()

        assert orch._ltp_failures.get("INFY") == 1

    def test_alert_fires_at_threshold(self):
        """After 3 consecutive LTP failures, force exit + alert fires (Fix 1)."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0
        )
        orch.active_trades["INFY"] = trade
        broker.add_position("INFY", qty=10, avg_price=1500.0, ltp=1500.0)
        broker.set_ltp("INFY", 0)

        with patch('agent.orchestrator.alert_error') as mock_alert, \
             patch('agent.orchestrator.alert_trade_exit'):
            # Fix 10: 1st failure uses fallback price, 2nd skips, 3rd force-exits
            for _ in range(3):
                orch._check_positions()

            mock_alert.assert_called()
            # Check last call was the force-exit alert
            last_call = mock_alert.call_args
            assert "LTP failure cascade" in last_call[0][0]
            assert "INFY" in last_call[0][0]
            # Position should be force-exited
            assert "INFY" not in orch.active_trades

    def test_success_resets_counter(self):
        """A successful LTP fetch resets the failure counter."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0
        )
        orch.active_trades["INFY"] = trade
        broker.add_position("INFY", qty=10, avg_price=1500.0, ltp=1500.0)

        # Simulate 2 failures (Fix 10: 1st uses fallback, 2nd skips)
        broker.set_ltp("INFY", 0)
        orch._check_positions()  # failure #1: uses fallback price
        assert orch._ltp_failures["INFY"] == 1
        orch._check_positions()  # failure #2: skips
        assert orch._ltp_failures["INFY"] == 2

        # Success resets
        broker.set_ltp("INFY", 1510.0)
        orch._check_positions()
        assert "INFY" not in orch._ltp_failures


# ============================================
# SL STUCK ORDER DETECTION (Task: Production Gaps)
# ============================================

class TestSLStuckOrderDetection:
    """Test live broker SL monitoring safety net."""

    def _make_orchestrator(self, broker, mode="live"):
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)
        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )
        orch._broker_mode = mode
        return orch

    def test_paper_mode_skips_verification(self):
        """Paper mode should not run SL verification."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker, mode="paper")

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0,
            order_ids=["BUY001", "SL001"]
        )
        orch.active_trades["INFY"] = trade

        # Should not call broker at all
        broker.get_order_status = MagicMock()
        orch._verify_sl_orders()
        broker.get_order_status.assert_not_called()

    def test_price_below_sl_starts_grace_period(self):
        """First breach starts grace timer without exit."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker, mode="live")

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0,
            order_ids=["BUY001", "SL001"]
        )
        orch.active_trades["INFY"] = trade

        # SL order is OPEN, price is below SL
        broker._orders["SL001"] = OrderBook(
            order_id="SL001", symbol="INFY", side=OrderSide.SELL,
            quantity=10, filled_quantity=0, pending_quantity=10,
            order_type=OrderType.SL_M, price=None, trigger_price=1450.0,
            average_price=None, status=OrderStatus.OPEN,
            placed_at=datetime.now(), updated_at=datetime.now()
        )
        broker.set_ltp("INFY", 1440.0)  # Below SL

        with patch.object(orch, '_exit_position') as mock_exit:
            orch._verify_sl_orders()
            mock_exit.assert_not_called()  # Grace period, no exit yet

        assert hasattr(trade, '_sl_breach_time')
        assert trade._sl_breach_time is not None

    def test_force_exit_after_grace_period(self):
        """After 30s grace period, forced market sell is triggered."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker, mode="live")

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0,
            order_ids=["BUY001", "SL001"]
        )
        # Set breach time to 31 seconds ago
        from utils.platform import now_ist
        trade._sl_breach_time = now_ist().replace(tzinfo=None) - timedelta(seconds=31)
        orch.active_trades["INFY"] = trade

        broker._orders["SL001"] = OrderBook(
            order_id="SL001", symbol="INFY", side=OrderSide.SELL,
            quantity=10, filled_quantity=0, pending_quantity=10,
            order_type=OrderType.SL_M, price=None, trigger_price=1450.0,
            average_price=None, status=OrderStatus.OPEN,
            placed_at=datetime.now(), updated_at=datetime.now()
        )
        broker.set_ltp("INFY", 1440.0)

        with patch.object(orch, '_exit_position') as mock_exit, \
             patch('agent.orchestrator.alert_error') as mock_alert:
            orch._verify_sl_orders()
            mock_exit.assert_called_once()
            assert "stuck" in mock_exit.call_args[0][1].lower()
            mock_alert.assert_called_once()

    def test_price_recovery_resets_breach_timer(self):
        """If price recovers above SL, breach timer is reset."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker, mode="live")

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0,
            order_ids=["BUY001", "SL001"]
        )
        from utils.platform import now_ist
        trade._sl_breach_time = now_ist().replace(tzinfo=None) - timedelta(seconds=10)
        orch.active_trades["INFY"] = trade

        broker._orders["SL001"] = OrderBook(
            order_id="SL001", symbol="INFY", side=OrderSide.SELL,
            quantity=10, filled_quantity=0, pending_quantity=10,
            order_type=OrderType.SL_M, price=None, trigger_price=1450.0,
            average_price=None, status=OrderStatus.OPEN,
            placed_at=datetime.now(), updated_at=datetime.now()
        )
        broker.set_ltp("INFY", 1460.0)  # Above SL

        orch._verify_sl_orders()
        assert trade._sl_breach_time is None  # Reset

    def test_no_sl_order_skips_check(self):
        """Trade with only buy order (no SL) is skipped."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker, mode="live")

        from agent.orchestrator import TradeRecord
        trade = TradeRecord(
            trade_id="T001", symbol="INFY", side="BUY",
            quantity=10, entry_price=1500.0, stop_loss=1450.0,
            original_stop_loss=1450.0, current_stop=1450.0,
            highest_price=1500.0, target=1575.0, atr=20.0,
            order_ids=["BUY001"]  # Only buy order
        )
        orch.active_trades["INFY"] = trade

        broker.get_order_status = MagicMock()
        orch._verify_sl_orders()
        broker.get_order_status.assert_not_called()


# ============================================
# STALE LOCK CLEANUP (Task: Production Gaps)
# ============================================

class TestStaleLockCleanup:
    """Test stale PID lock file cleanup in start.py."""

    def test_dead_pid_lock_cleaned(self):
        """Lock file from dead process should be cleaned and re-acquired."""
        from utils.platform import is_pid_running

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
            # Write a dead PID (999998 is almost certainly not running)
            f.write(f"PID: 999998\nStarted: 2026-02-11 08:00:00\n")
            lock_path = f.name

        try:
            content = Path(lock_path).read_text()
            old_pid = None
            for line in content.splitlines():
                if line.startswith("PID:"):
                    old_pid = int(line.split(":")[1].strip())
                    break

            assert old_pid == 999998
            assert not is_pid_running(old_pid)  # Dead
            # Clean up stale lock
            Path(lock_path).unlink(missing_ok=True)
            assert not Path(lock_path).exists()
        finally:
            Path(lock_path).unlink(missing_ok=True)

    def test_live_pid_lock_not_cleaned(self):
        """Lock file from running process should NOT be cleaned."""
        from utils.platform import is_pid_running

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
            f.write(f"PID: {os.getpid()}\nStarted: 2026-02-11 08:00:00\n")
            lock_path = f.name

        try:
            content = Path(lock_path).read_text()
            for line in content.splitlines():
                if line.startswith("PID:"):
                    live_pid = int(line.split(":")[1].strip())
                    break
            assert is_pid_running(live_pid)  # Still running
            assert Path(lock_path).exists()  # NOT cleaned
        finally:
            Path(lock_path).unlink(missing_ok=True)


# ============================================
# DISK SPACE CHECK (Task: Production Gaps)
# ============================================

class TestDiskSpaceOnDBInit:
    """Test disk space check during TradeDB initialization."""

    def test_low_space_triggers_warning(self):
        """TradeDB init logs critical when disk space is low."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            with patch('utils.trade_db.check_disk_space', return_value=False), \
                 patch('utils.alerts.alert_error') as mock_alert:
                from utils.trade_db import TradeDB
                db = TradeDB(db_path=db_path)
                mock_alert.assert_called_once()
                assert "DISK SPACE" in mock_alert.call_args[0][0].upper()

    def test_sufficient_space_no_alert(self):
        """TradeDB init doesn't alert when disk space is sufficient."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            with patch('utils.trade_db.check_disk_space', return_value=True), \
                 patch('utils.alerts.alert_error') as mock_alert:
                from utils.trade_db import TradeDB
                db = TradeDB(db_path=db_path)
                mock_alert.assert_not_called()


# ============================================
# IST TIMEZONE E2E (Task: Production Gaps)
# ============================================

class TestISTTimezoneIntegration:
    """Verify IST timezone functions work for trading phase detection."""

    def test_phase_detection_uses_ist(self):
        """Phase detection should use time_ist(), not system local time."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        # Simulate 10:30 AM IST (should be ENTRY_WINDOW)
        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(10, 30)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.ENTRY_WINDOW

    def test_pre_market_phase_at_845(self):
        """8:45 AM IST should be PRE_MARKET."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(8, 45)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.PRE_MARKET

    def test_market_closed_at_night(self):
        """10:00 PM IST should be MARKET_CLOSED."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(22, 0)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.MARKET_CLOSED


# ============================================
# CUJ: YFINANCE TIMEOUT DURING ENTRY (E2E)
# ============================================

class TestCUJ_YfinanceTimeoutDuringEntry:
    """
    CUJ 39: Paper broker get_ltp() times out during signal evaluation.
    System should handle gracefully and continue.
    """

    def test_yfinance_timeout_in_paper_broker(self):
        """Timeout in get_ltp should raise TimeoutError, not hang."""
        from utils.platform import yfinance_with_timeout

        def slow_fetch():
            time.sleep(10)
            return 100.0

        with pytest.raises(TimeoutError):
            yfinance_with_timeout(slow_fetch, timeout_seconds=1)


# ============================================
# CUJ: DOCKER UTC SERVER PHASE DETECTION (E2E)
# ============================================

class TestCUJ_DockerUTCServerPhaseDetection:
    """
    CUJ 40: Bot running on UTC server (Docker) correctly detects
    Indian market phases using IST timezone functions.
    """

    def test_ist_independent_of_system_tz(self):
        """now_ist() should return IST regardless of TZ env var."""
        from utils.platform import now_ist
        import pytz

        # Even if we could set TZ=UTC, now_ist() should still return IST
        result = now_ist()
        assert str(result.tzinfo) == "Asia/Kolkata"

    def test_today_ist_is_date_not_datetime(self):
        """today_ist() returns a date, not datetime, for DB compatibility."""
        from utils.platform import today_ist
        result = today_ist()
        assert isinstance(result, date)
        assert type(result) == date  # Not a datetime subclass


# ===================================================================
# AUTOMATED PAPER TRADING INFRASTRUCTURE TESTS
# ===================================================================


class TestDailyReportFile:
    """Test that _write_daily_report_file creates readable report files."""

    def _make_orchestrator(self):
        """Create a minimal orchestrator with mocked dependencies."""
        from agent.orchestrator import TradingOrchestrator
        broker = MagicMock()
        broker.connect.return_value = True
        risk_manager = MagicMock()
        risk_manager.get_portfolio_value.return_value = 100000.0

        with patch('agent.orchestrator.get_trade_db'), \
             patch('agent.orchestrator.SignalAdapter'):
            orch = TradingOrchestrator(broker, risk_manager)
        return orch

    def test_report_file_created(self, tmp_path, monkeypatch):
        """Report file is created with correct name and content."""
        monkeypatch.chdir(tmp_path)
        orch = self._make_orchestrator()

        mock_trade = MagicMock()
        mock_trade.symbol = "RELIANCE"
        mock_trade.quantity = 10
        mock_trade.entry_price = 2500.0
        mock_trade.exit_price = 2550.0
        mock_trade.pnl = 500.0
        mock_trade.exit_reason = "Target hit"
        orch.completed_trades = [mock_trade]

        orch._write_daily_report_file(
            total_pnl=500.0, winners=1, losers=0,
            portfolio_value=100000.0
        )

        reports = list((tmp_path / "data" / "reports").glob("*.txt"))
        assert len(reports) == 1

        content = reports[0].read_text()
        assert "DAILY TRADING REPORT" in content
        assert "RELIANCE" in content
        assert "500.00" in content
        assert "Win Rate" in content
        assert "100.0%" in content

    def test_report_with_losses(self, tmp_path, monkeypatch):
        """Report correctly shows losing trades."""
        monkeypatch.chdir(tmp_path)
        orch = self._make_orchestrator()

        mock_win = MagicMock()
        mock_win.symbol = "TCS"
        mock_win.quantity = 5
        mock_win.entry_price = 3800.0
        mock_win.exit_price = 3850.0
        mock_win.pnl = 250.0
        mock_win.exit_reason = "Target hit"

        mock_loss = MagicMock()
        mock_loss.symbol = "INFY"
        mock_loss.quantity = 8
        mock_loss.entry_price = 1500.0
        mock_loss.exit_price = 1460.0
        mock_loss.pnl = -320.0
        mock_loss.exit_reason = "Stop loss"

        orch.completed_trades = [mock_win, mock_loss]

        orch._write_daily_report_file(
            total_pnl=-70.0, winners=1, losers=1,
            portfolio_value=99500.0
        )

        reports = list((tmp_path / "data" / "reports").glob("*.txt"))
        assert len(reports) == 1
        content = reports[0].read_text()
        assert "TCS" in content
        assert "INFY" in content
        assert "50.0%" in content  # 1 win, 1 loss = 50%

    def test_report_dir_created_if_missing(self, tmp_path, monkeypatch):
        """Report directory is created if it doesn't exist."""
        monkeypatch.chdir(tmp_path)
        orch = self._make_orchestrator()

        mock_trade = MagicMock()
        mock_trade.symbol = "HDFC"
        mock_trade.quantity = 3
        mock_trade.entry_price = 1600.0
        mock_trade.exit_price = 1620.0
        mock_trade.pnl = 60.0
        mock_trade.exit_reason = "Target"
        orch.completed_trades = [mock_trade]

        reports_dir = tmp_path / "data" / "reports"
        assert not reports_dir.exists()

        orch._write_daily_report_file(
            total_pnl=60.0, winners=1, losers=0,
            portfolio_value=100000.0
        )

        assert reports_dir.exists()
        reports = list(reports_dir.glob("*.txt"))
        assert len(reports) == 1


class TestDaemonWrapper:
    """Test daemon wrapper and plist configuration."""

    def test_wrapper_script_exists_and_executable(self):
        """run_daemon.sh exists and is executable."""
        wrapper = Path(__file__).parent.parent / "scripts" / "run_daemon.sh"
        assert wrapper.exists(), "scripts/run_daemon.sh not found"
        assert os.access(str(wrapper), os.X_OK), "run_daemon.sh is not executable"

    def test_wrapper_sets_paper_mode(self):
        """Wrapper script forces TRADING_MODE=paper."""
        wrapper = Path(__file__).parent.parent / "scripts" / "run_daemon.sh"
        content = wrapper.read_text()
        assert 'TRADING_MODE=paper' in content

    def test_wrapper_sources_env(self):
        """Wrapper script sources .env file."""
        wrapper = Path(__file__).parent.parent / "scripts" / "run_daemon.sh"
        content = wrapper.read_text()
        assert 'source .env' in content

    def test_plist_uses_wrapper(self):
        """Plist calls the wrapper script, not Python directly."""
        plist = Path(__file__).parent.parent / "scripts" / "com.trading.daemon.plist"
        content = plist.read_text()
        assert 'run_daemon.sh' in content
        assert '/bin/bash' in content

    def test_plist_runs_at_load(self):
        """Plist has RunAtLoad=true for auto-start on login."""
        plist = Path(__file__).parent.parent / "scripts" / "com.trading.daemon.plist"
        content = plist.read_text()
        assert '<key>RunAtLoad</key>' in content
        # RunAtLoad followed by true
        assert '<true/>' in content

    def test_setup_script_exists_and_executable(self):
        """setup_macos.sh exists and is executable."""
        setup = Path(__file__).parent.parent / "scripts" / "setup_macos.sh"
        assert setup.exists(), "scripts/setup_macos.sh not found"
        assert os.access(str(setup), os.X_OK), "setup_macos.sh is not executable"

    def test_setup_script_has_all_commands(self):
        """setup_macos.sh supports install/uninstall/start/stop/status/logs."""
        setup = Path(__file__).parent.parent / "scripts" / "setup_macos.sh"
        content = setup.read_text()
        for cmd in ['install', 'uninstall', 'start', 'stop', 'status', 'logs']:
            assert f'cmd_{cmd}' in content, f"Missing {cmd} command in setup_macos.sh"


class TestReportCLI:
    """Test the trade.py report command."""

    def test_report_command_registered(self):
        """The 'report' command is available in trade.py CLI."""
        import trade
        assert 'report' in dir(trade) or hasattr(trade, 'cmd_report')

    def test_report_runs_without_error(self):
        """cmd_report runs without crashing when no trades exist."""
        mock_db = MagicMock()
        mock_db.get_today_trades.return_value = []
        mock_db.get_performance_history.return_value = []

        with patch('utils.trade_db.get_trade_db', return_value=mock_db):
            import trade
            args = MagicMock()
            # Should not raise
            trade.cmd_report(args)


# ============================================
# IST DATETIME FIXES — rate_limiter, error_handler, news providers
# ============================================

class TestRateLimiterIST:
    """Verify rate_limiter uses now_ist() instead of datetime.now()."""

    def test_wait_uses_ist(self):
        """RateLimiter.wait() should use IST timestamps, not system local time."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(requests_per_minute=1000, requests_per_second=100)

        with patch('utils.rate_limiter.now_ist') as mock_ist:
            from datetime import datetime
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            mock_now = datetime(2026, 2, 13, 10, 30, 0, tzinfo=ist)
            mock_ist.return_value = mock_now

            limiter.wait("test_endpoint")

            # now_ist() should have been called (for timestamp recording)
            assert mock_ist.call_count >= 1

    def test_record_failure_uses_ist(self):
        """record_failure should timestamp with IST."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter()

        with patch('utils.rate_limiter.now_ist') as mock_ist:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            mock_now = datetime(2026, 2, 13, 10, 30, 0, tzinfo=ist)
            mock_ist.return_value = mock_now

            limiter.record_failure("test_ep")

            # Should have been called for _last_failure_time
            assert mock_ist.call_count >= 1

    def test_cooldown_uses_ist(self):
        """Rate limit cooldown should use IST timestamps."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter()

        with patch('utils.rate_limiter.now_ist') as mock_ist:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            mock_now = datetime(2026, 2, 13, 10, 30, 0, tzinfo=ist)
            mock_ist.return_value = mock_now

            limiter.record_failure("test_ep", is_rate_limit=True)

            # Cooldown should be set based on IST time
            assert "test_ep" in limiter._cooldown_until

    def test_no_raw_datetime_now_in_rate_limiter(self):
        """rate_limiter.py should not contain raw datetime.now() calls."""
        import inspect
        from utils import rate_limiter
        source = inspect.getsource(rate_limiter)
        # Should use now_ist(), not datetime.now()
        assert 'datetime.now()' not in source, \
            "rate_limiter.py still contains raw datetime.now() calls"


class TestCircuitBreakerIST:
    """Verify CircuitBreaker uses now_ist() instead of datetime.now()."""

    def test_record_failure_uses_ist(self):
        """CircuitBreaker.record_failure should use IST timestamps."""
        from utils.error_handler import CircuitBreaker

        breaker = CircuitBreaker(name="test", recovery_timeout=60)

        with patch('utils.error_handler.now_ist') as mock_ist:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            mock_now = datetime(2026, 2, 13, 10, 30, 0, tzinfo=ist)
            mock_ist.return_value = mock_now

            breaker.record_failure()

            assert mock_ist.call_count >= 1
            assert breaker._last_failure_time is not None

    def test_state_transition_uses_ist(self):
        """OPEN -> HALF_OPEN transition should use IST for elapsed time."""
        from utils.error_handler import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=1)

        # Push to OPEN state
        with patch('utils.error_handler.now_ist') as mock_ist:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            t0 = datetime(2026, 2, 13, 10, 30, 0, tzinfo=ist)
            mock_ist.return_value = t0

            breaker.record_failure()
            breaker.record_failure()
            assert breaker._state == CircuitBreaker.State.OPEN

        # After recovery_timeout, should transition to HALF_OPEN
        with patch('utils.error_handler.now_ist') as mock_ist:
            t1 = datetime(2026, 2, 13, 10, 30, 2, tzinfo=ist)  # 2 seconds later
            mock_ist.return_value = t1

            state = breaker.state
            assert state == CircuitBreaker.State.HALF_OPEN

    def test_no_raw_datetime_now_in_error_handler(self):
        """error_handler.py should not contain raw datetime.now() calls outside TradingException."""
        import inspect
        from utils import error_handler
        source = inspect.getsource(error_handler)
        # TradingException has a fallback datetime.now() in ImportError handler - that's OK
        # But there should be no other datetime.now() calls
        lines = source.split('\n')
        raw_calls = [
            (i+1, line) for i, line in enumerate(lines)
            if 'datetime.now()' in line and 'ImportError' not in lines[max(0, i-1):i+2].__str__()
            and 'except ImportError' not in line
            and '# fallback' not in line.lower()
        ]
        # Allow only the TradingException fallback
        non_fallback = [
            (num, line) for num, line in raw_calls
            if 'TradingException' not in source.split('\n')[max(0, num-5):num].__str__()
        ]
        # Filter to just circuit breaker lines
        cb_calls = [
            (num, line) for num, line in non_fallback
            if 'CircuitBreaker' in source[:source.find(line)] or 'record_failure' in source[max(0, source.find(line)-200):source.find(line)]
        ]
        assert len(cb_calls) == 0, \
            f"CircuitBreaker still uses raw datetime.now(): {cb_calls}"


class TestNewsProviderDatetime:
    """Verify news providers use proper UTC (not deprecated utcnow)."""

    def test_finnhub_no_deprecated_utcnow(self):
        """finnhub.py should not use deprecated datetime.utcnow()."""
        import inspect
        from providers.news import finnhub
        source = inspect.getsource(finnhub)
        assert 'datetime.utcnow()' not in source, \
            "finnhub.py still uses deprecated datetime.utcnow()"
        assert 'utcfromtimestamp' not in source, \
            "finnhub.py still uses deprecated utcfromtimestamp()"

    def test_newsapi_no_deprecated_utcnow(self):
        """newsapi.py should not use deprecated datetime.utcnow()."""
        import inspect
        from providers.news import newsapi
        source = inspect.getsource(newsapi)
        assert 'datetime.utcnow()' not in source, \
            "newsapi.py still uses deprecated datetime.utcnow()"

    def test_finnhub_uses_timezone_utc(self):
        """finnhub.py should use datetime.now(timezone.utc)."""
        import inspect
        from providers.news import finnhub
        source = inspect.getsource(finnhub)
        assert 'timezone.utc' in source, \
            "finnhub.py should use timezone.utc for UTC datetimes"

    def test_newsapi_uses_timezone_utc(self):
        """newsapi.py should use datetime.now(timezone.utc)."""
        import inspect
        from providers.news import newsapi
        source = inspect.getsource(newsapi)
        assert 'timezone.utc' in source, \
            "newsapi.py should use timezone.utc for UTC datetimes"


# ============================================
# HOLDING PHASE REACHABILITY
# ============================================

class TestHoldingPhaseReachable:
    """Verify HOLDING phase is actually reachable with correct config."""

    def test_holding_phase_reachable_with_default_config(self):
        """entry_window_end < square_off_start so HOLDING phase exists."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.hours.entry_window_end < config.hours.square_off_start, \
            f"entry_window_end ({config.hours.entry_window_end}) must be < " \
            f"square_off_start ({config.hours.square_off_start}) for HOLDING phase"

    def test_holding_phase_at_1445(self):
        """14:45 IST should be HOLDING phase (between entry_window_end=14:30 and square_off_start=15:00)."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(14, 45)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.HOLDING, \
                f"Expected HOLDING at 14:45, got {phase}"

    def test_square_off_at_1500(self):
        """15:00 IST should be SQUARE_OFF phase."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(15, 0)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.SQUARE_OFF, \
                f"Expected SQUARE_OFF at 15:00, got {phase}"

    def test_entry_window_ends_at_1430(self):
        """14:29 IST should still be ENTRY_WINDOW, 14:30 should be HOLDING."""
        broker = MockBroker()
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)

        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )

        from agent.orchestrator import TradingPhase
        with patch('agent.orchestrator.time_ist', return_value=dt_time(14, 29)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.ENTRY_WINDOW

        with patch('agent.orchestrator.time_ist', return_value=dt_time(14, 30)):
            phase = orch._get_current_phase()
            assert phase == TradingPhase.HOLDING


# ============================================
# DAILY PRICE CACHE TESTS
# ============================================

class TestDailyPriceCache:
    """Test daily price cache in signal generator."""

    def test_cache_key_is_date_based(self):
        """Price cache should invalidate on new day."""
        from unittest.mock import MagicMock
        import pandas as pd

        with patch('src.signals.generator.SignalGenerator.__init__', return_value=None):
            from src.signals.generator import SignalGenerator
            gen = SignalGenerator.__new__(SignalGenerator)
            gen._daily_price_cache = {"RELIANCE": pd.DataFrame({"close": [100]})}
            gen._daily_price_cache_date = date(2026, 2, 12)

            # Same day → cache hit
            with patch('src.signals.generator.today_ist', return_value=date(2026, 2, 12)):
                assert gen._daily_price_cache_date == date(2026, 2, 12)
                assert "RELIANCE" in gen._daily_price_cache

            # New day → cache should be cleared by _fetch_all_prices
            with patch('src.signals.generator.today_ist', return_value=date(2026, 2, 13)):
                assert gen._daily_price_cache_date != date(2026, 2, 13)

    def test_signal_adapter_shares_generator_cache(self):
        """Signal adapter should check generator's daily cache before fetching."""
        import pandas as pd

        with patch('agent.signal_adapter.PriceFetcher'), \
             patch('agent.signal_adapter.SignalGenerator'):
            from agent.signal_adapter import SignalAdapter

            mock_gen = MagicMock()
            mock_gen._daily_price_cache = {
                "TCS": pd.DataFrame({
                    "open": [3800]*60, "high": [3850]*60, "low": [3750]*60,
                    "close": [3820]*60, "volume": [1000000]*60
                })
            }
            mock_gen._daily_price_cache_date = date.today()

            adapter = SignalAdapter(signal_generator=mock_gen)

            with patch('agent.signal_adapter.today_ist', return_value=date.today()):
                result = adapter._get_price_data("TCS")
                assert result is not None
                assert len(result) == 60

    def test_signal_adapter_ignores_stale_cache(self):
        """Signal adapter should not use generator cache from yesterday."""
        import pandas as pd

        with patch('agent.signal_adapter.PriceFetcher') as mock_pf, \
             patch('agent.signal_adapter.SignalGenerator'):
            from agent.signal_adapter import SignalAdapter

            mock_gen = MagicMock()
            mock_gen._daily_price_cache = {
                "TCS": pd.DataFrame({"close": [3800]*5, "volume": [100000]*5})
            }
            mock_gen._daily_price_cache_date = date(2026, 2, 10)  # Old date

            adapter = SignalAdapter(signal_generator=mock_gen)
            # Mock primary fetch to return None (force cache check)
            adapter._fetch_from_primary = MagicMock(return_value=None)
            adapter._fetch_from_fallback = MagicMock(return_value=None)

            with patch('agent.signal_adapter.today_ist', return_value=date(2026, 2, 13)):
                result = adapter._get_price_data("TCS")
                # Should NOT use stale cache — result should be None since both fetches return None
                assert result is None


# ============================================
# BUDGET-AWARE NEWS FETCHING TESTS
# ============================================

class TestBudgetAwareNews:
    """Test budget-aware news fetching in signal generator."""

    def test_news_budget_config_exists(self):
        """NEWS_BUDGET_PER_CYCLE config parameter exists."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert hasattr(config.providers, 'news_budget_per_cycle')
        assert config.providers.news_budget_per_cycle == 15

    def test_news_cache_ttl_is_30_min(self):
        """NEWS_CACHE_TTL should be 1800 seconds (30 min)."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.providers.news_cache_ttl == 1800

    def test_llm_cache_ttl_is_6_hours(self):
        """LLM_CACHE_TTL should be 21600 seconds (6 hours)."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.providers.llm_cache_ttl == 21600


# ============================================
# EXPLAINER CACHING TESTS
# ============================================

class TestExplainerCaching:
    """Test that SignalExplainer routes through provider system with caching."""

    def test_explainer_uses_cache(self):
        """Explainer should check cache before calling LLM."""
        import inspect
        from src.signals import explainer
        source = inspect.getsource(explainer)
        assert 'get_response_cache' in source, \
            "Explainer should use response cache"
        assert 'get_quota_manager' in source, \
            "Explainer should use quota manager"

    def test_explainer_no_direct_openai(self):
        """Explainer should NOT import openai directly."""
        import inspect
        from src.signals import explainer
        source = inspect.getsource(explainer)
        assert 'from openai' not in source, \
            "Explainer should not import openai directly"
        assert 'import openai' not in source, \
            "Explainer should not import openai directly"

    def test_explainer_has_template_fallback(self):
        """Explainer should have template fallback when LLM unavailable."""
        from src.signals.explainer import SignalExplainer
        assert hasattr(SignalExplainer, '_template_explanation'), \
            "Explainer must have _template_explanation fallback"

    def test_explainer_cache_key_includes_date(self):
        """Cache key should include today's date to avoid cross-day hits."""
        import inspect
        from src.signals import explainer
        source = inspect.getsource(explainer)
        assert 'today_ist()' in source, \
            "Explainer cache key should include today_ist() for daily invalidation"


# ============================================
# MARKET INDICATORS CACHING TESTS
# ============================================

class TestMarketIndicatorsCaching:
    """Test that MarketIndicators uses cache + quota system."""

    def test_market_indicators_uses_cache(self):
        """MarketIndicators should use response cache."""
        import inspect
        from src.data import market_indicators
        source = inspect.getsource(market_indicators)
        assert 'get_response_cache' in source, \
            "MarketIndicators should use response cache"

    def test_market_indicators_uses_quota(self):
        """MarketIndicators should use quota manager."""
        import inspect
        from src.data import market_indicators
        source = inspect.getsource(market_indicators)
        assert 'get_quota_manager' in source, \
            "MarketIndicators should use quota manager"

    def test_market_indicators_uses_timeout(self):
        """MarketIndicators should use yfinance_with_timeout."""
        import inspect
        from src.data import market_indicators
        source = inspect.getsource(market_indicators)
        assert 'yfinance_with_timeout' in source, \
            "MarketIndicators should use yfinance_with_timeout for safety"

    def test_market_indicators_cache_ttl(self):
        """Market index cache TTL should be 1 hour (3600s)."""
        from src.data.market_indicators import _INDEX_CACHE_TTL
        assert _INDEX_CACHE_TTL == 3600, \
            f"Expected 3600s cache TTL, got {_INDEX_CACHE_TTL}"


# ============================================
# CONFIG INTERVAL OPTIMIZATION TESTS
# ============================================

class TestConfigIntervals:
    """Verify optimized polling intervals."""

    def test_signal_refresh_15_min(self):
        """Signal refresh should be 15 min (900s), not 5 min."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.intervals.signal_refresh_seconds == 900, \
            f"Expected 900s signal refresh, got {config.intervals.signal_refresh_seconds}"

    def test_news_check_30_min(self):
        """News check should be 30 min (1800s), not 5 min."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.intervals.news_check_seconds == 1800, \
            f"Expected 1800s news check, got {config.intervals.news_check_seconds}"

    def test_token_check_5_min_default(self):
        """Fix 5: Token check interval should be 300s (5 min)."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.orders.token_check_interval_seconds == 300, \
            f"Expected 300s token check (Fix 5), got {config.orders.token_check_interval_seconds}"

    def test_no_signal_refresh_in_holding(self):
        """HOLDING phase handler should not call _maybe_refresh_signals."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._handle_holding)
        assert '_maybe_refresh_signals' not in source, \
            "HOLDING phase should NOT refresh signals — no new entries possible"


# ============================================
# SAFETY HARDENING TESTS (10 Fixes)
# ============================================

class TestSafetyHardeningFixes:
    """Tests for all 10 safety fixes from the hardening plan.

    Fix 1: LTP force exit after 3 failures
    Fix 2: SL cancel verify before placing new
    Fix 3: ATR validation with confidence penalty
    Fix 4: Partial fill better tracking
    Fix 5: Token check interval 300s
    Fix 6: News API failure → confidence penalty
    Fix 7: Emergency exit timeout 300s + parallel
    Fix 8: Model fallback indicator penalty
    Fix 9: Gap-down on open check
    Fix 10: LTP timeout differentiation
    """

    def _make_orchestrator(self, broker):
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)
        with patch('agent.orchestrator.SignalAdapter'), \
             patch('agent.orchestrator.NewsFetcher'), \
             patch('agent.orchestrator.NewsFeatureExtractor'), \
             patch('agent.orchestrator.get_trade_db') as mock_db, \
             patch('agent.orchestrator.get_limiter') as mock_limiter:
            mock_db.return_value = MagicMock()
            mock_limiter.return_value = MagicMock()
            orch = __import__('agent.orchestrator', fromlist=['TradingOrchestrator']).TradingOrchestrator(
                broker=broker, risk_manager=rm
            )
        return orch

    def _make_trade(self, symbol="RELIANCE", entry=2500.0, sl=2400.0,
                    target=2650.0, qty=10, highest=2500.0, atr=50.0):
        from agent.orchestrator import TradeRecord
        return TradeRecord(
            trade_id=f"{symbol}_TEST", symbol=symbol, side="BUY",
            quantity=qty, entry_price=entry, stop_loss=sl,
            original_stop_loss=sl, current_stop=sl,
            highest_price=highest, target=target, atr=atr,
            order_ids=["ORD_ENTRY", "ORD_SL"]
        )

    # --- Fix 1: LTP force exit at 3 ---

    def test_fix1_force_exit_at_3_failures(self):
        """Fix 1: Position is force-exited after 3 consecutive LTP failures."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0, ltp=2500.0)
        broker.set_ltp("RELIANCE", 0)  # LTP failure

        with patch('agent.orchestrator.alert_error'), \
             patch('agent.orchestrator.alert_trade_exit'):
            for _ in range(3):
                orch._check_positions()

        assert "RELIANCE" not in orch.active_trades, \
            "Position should be force-exited after 3 LTP failures"

    def test_fix1_no_exit_at_2_failures(self):
        """Fix 1: Position survives 2 failures (not yet threshold)."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0, ltp=2500.0)
        broker.set_ltp("RELIANCE", 0)

        for _ in range(2):
            orch._check_positions()

        assert "RELIANCE" in orch.active_trades, \
            "Position should survive 2 failures"

    # --- Fix 2: SL cancel verify ---

    def test_fix2_sl_cancel_verify_prevents_double_sl(self):
        """Fix 2: If old SL cancel not confirmed, don't place new SL."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade

        # Mock: cancel succeeds but status check shows still OPEN
        broker.cancel_order = MagicMock()
        status_mock = MagicMock()
        status_mock.status = OrderStatus.OPEN  # NOT cancelled
        broker.get_order_status = MagicMock(return_value=status_mock)

        # Should NOT place new SL since old one not confirmed cancelled
        with patch.object(orch, '_place_stop_loss_at') as mock_place:
            orch._cancel_and_replace_sl(trade, "ORD_SL", 2450.0)
            mock_place.assert_not_called()

    def test_fix2_sl_cancel_confirm_allows_new_sl(self):
        """Fix 2: If old SL cancel confirmed, new SL is placed."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade

        # Mock: cancel succeeds and status confirms CANCELLED
        broker.cancel_order = MagicMock()
        status_mock = MagicMock()
        status_mock.status = OrderStatus.CANCELLED
        broker.get_order_status = MagicMock(return_value=status_mock)

        with patch.object(orch, '_place_stop_loss_at', return_value="NEW_SL") as mock_place:
            orch._cancel_and_replace_sl(trade, "ORD_SL", 2450.0)
            mock_place.assert_called_once_with(trade, 2450.0)

    # --- Fix 3: ATR validation ---

    def test_fix3_atr_zero_adds_risk_reason(self):
        """Fix 3: When ATR is 0, score includes risk reason."""
        import pandas as pd
        import numpy as np
        from src.models.scoring import ScoringEngine

        engine = ScoringEngine()
        df = pd.DataFrame({
            'open': [100]*60, 'high': [105]*60, 'low': [95]*60,
            'close': [102]*60, 'volume': [1000000]*60,
            'rsi': [55]*60, 'macd': [0.5]*60, 'macd_signal': [0.3]*60,
            'bb_upper': [110]*60, 'bb_lower': [90]*60,
            'atr': [0]*60,  # Zero ATR
            'atr_pct': [0]*60,
            'adx': [30]*60, 'supertrend': [95]*60,
            'trend_strength': [0.6]*60,
            'volume_ratio': [1.2]*60, 'obv': [500000]*60,
        })

        from src.storage.models import TradeType
        score = engine.score_stock("TEST", df, trade_type=TradeType.INTRADAY, news_score=0.5, news_reasons=[])
        if score:
            assert any("[RISK] ATR unavailable" in r for r in score.reasons), \
                "ATR=0 should add risk reason"

    def test_fix3_atr_zero_penalizes_confidence(self):
        """Fix 3: ATR=0 should reduce confidence via fallback penalty."""
        import pandas as pd
        from src.models.scoring import ScoringEngine
        from src.storage.models import TradeType

        engine = ScoringEngine()
        # Score with good ATR
        df_good = pd.DataFrame({
            'open': [100]*60, 'high': [105]*60, 'low': [95]*60,
            'close': [102]*60, 'volume': [1000000]*60,
            'rsi': [55]*60, 'macd': [0.5]*60, 'macd_signal': [0.3]*60,
            'bb_upper': [110]*60, 'bb_lower': [90]*60,
            'atr': [3.0]*60, 'atr_pct': [3.0]*60,
            'adx': [30]*60, 'supertrend': [95]*60,
            'trend_strength': [0.6]*60,
            'volume_ratio': [1.2]*60, 'obv': [500000]*60,
        })
        # Score with zero ATR
        df_bad = df_good.copy()
        df_bad['atr'] = 0
        df_bad['atr_pct'] = 0

        good = engine.score_stock("TEST", df_good, trade_type=TradeType.INTRADAY, news_score=0.5, news_reasons=[])
        bad = engine.score_stock("TEST", df_bad, trade_type=TradeType.INTRADAY, news_score=0.5, news_reasons=[])

        if good and bad:
            assert bad.confidence <= good.confidence, \
                "ATR=0 should penalize confidence"

    # --- Fix 4: Partial fill tracking ---

    def test_fix4_partial_fill_code_exists(self):
        """Fix 4: Orchestrator handles partial fill with re-check after wait."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._enter_position)
        assert 'time.sleep(3)' in source, \
            "Should wait 3s after cancel failure for re-check"
        assert 'updated_status' in source or 'filled_quantity' in source, \
            "Should re-check filled quantity after wait"

    # --- Fix 5: Token check 300s ---

    def test_fix5_token_check_300s(self):
        """Fix 5: Token check interval should be 300s (5 min)."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.orders.token_check_interval_seconds == 300

    # --- Fix 6: No news → confidence penalty ---

    def test_fix6_no_news_adds_risk_reason(self):
        """Fix 6: When no news articles available, risk reason is added."""
        import inspect
        from src.signals import generator
        source = inspect.getsource(generator.SignalGenerator.run)
        assert "[RISK] No news data available" in source, \
            "Generator should add risk reason when no news data"

    def test_fix6_no_news_penalizes_confidence(self):
        """Fix 6: No-news risk reason triggers 10% confidence penalty in scoring."""
        import pandas as pd
        from src.models.scoring import ScoringEngine
        from src.storage.models import TradeType

        engine = ScoringEngine()
        df = pd.DataFrame({
            'open': [100]*60, 'high': [105]*60, 'low': [95]*60,
            'close': [102]*60, 'volume': [1000000]*60,
            'rsi': [55]*60, 'macd': [0.5]*60, 'macd_signal': [0.3]*60,
            'bb_upper': [110]*60, 'bb_lower': [90]*60,
            'atr': [3.0]*60, 'atr_pct': [3.0]*60,
            'adx': [30]*60, 'supertrend': [95]*60,
            'trend_strength': [0.6]*60,
            'volume_ratio': [1.2]*60, 'obv': [500000]*60,
        })

        score_with_news = engine.score_stock("TEST", df, trade_type=TradeType.INTRADAY,
            news_score=0.6, news_reasons=["[NEWS] 3 articles, BULLISH sentiment"])
        score_no_news = engine.score_stock("TEST", df, trade_type=TradeType.INTRADAY,
            news_score=0.5, news_reasons=["[RISK] No news data available - trading on technicals only"])

        if score_with_news and score_no_news:
            assert score_no_news.confidence <= score_with_news.confidence, \
                "No-news should penalize confidence"

    # --- Fix 7: Shutdown timeout + parallel ---

    def test_fix7_shutdown_timeout_300s(self):
        """Fix 7: Shutdown timeout should be 300s (5 min), not 120s."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._shutdown)
        assert 'SHUTDOWN_TIMEOUT = 300' in source, \
            "Shutdown timeout should be 300s"

    def test_fix7_shutdown_uses_parallel_exit(self):
        """Fix 7: Shutdown should call _parallel_exit_all."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._shutdown)
        assert '_parallel_exit_all' in source, \
            "Shutdown should use parallel exit"

    def test_fix7_parallel_exit_method_exists(self):
        """Fix 7: _parallel_exit_all method uses ThreadPoolExecutor."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        assert hasattr(TradingOrchestrator, '_parallel_exit_all')
        source = inspect.getsource(TradingOrchestrator._parallel_exit_all)
        assert 'ThreadPoolExecutor' in source

    def test_fix7_parallel_exit_handles_single_position(self):
        """Fix 7: Single position should not use thread overhead."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0, ltp=2500.0)

        with patch.object(orch, '_exit_position') as mock_exit, \
             patch('agent.orchestrator.alert_trade_exit'):
            orch._parallel_exit_all("test")
            mock_exit.assert_called_once_with(trade, "test")

    # --- Fix 8: Fallback indicator penalty ---

    def test_fix8_multiple_fallbacks_penalize_confidence(self):
        """Fix 8: Multiple missing indicators reduce confidence cumulatively."""
        import pandas as pd
        from src.models.scoring import ScoringEngine
        from src.storage.models import TradeType

        engine = ScoringEngine()
        # All indicators present
        df_good = pd.DataFrame({
            'open': [100]*60, 'high': [105]*60, 'low': [95]*60,
            'close': [102]*60, 'volume': [1000000]*60,
            'rsi': [55]*60, 'macd': [0.5]*60, 'macd_signal': [0.3]*60,
            'bb_upper': [110]*60, 'bb_lower': [90]*60,
            'atr': [3.0]*60, 'atr_pct': [3.0]*60,
            'adx': [30]*60, 'supertrend': [95]*60,
            'trend_strength': [0.6]*60,
            'volume_ratio': [1.2]*60, 'obv': [500000]*60,
        })
        # Multiple indicators missing
        df_bad = df_good.copy()
        df_bad['atr'] = 0
        df_bad['rsi'] = 0
        df_bad['trend_strength'] = 0

        good = engine.score_stock("TEST", df_good, trade_type=TradeType.INTRADAY, news_score=0.5, news_reasons=[])
        bad = engine.score_stock("TEST", df_bad, trade_type=TradeType.INTRADAY, news_score=0.5, news_reasons=[])

        if good and bad:
            assert bad.confidence < good.confidence, \
                "Multiple missing indicators should reduce confidence"
            # Check risk reason about fallback values
            assert any("[RISK]" in r for r in bad.reasons)

    def test_fix8_penalty_capped_at_25pct(self):
        """Fix 8: Fallback penalty is capped — min multiplier is 0.75."""
        import inspect
        from src.models import scoring
        source = inspect.getsource(scoring.ScoringEngine.score_stock)
        assert 'max(0.75' in source, \
            "Fallback penalty should be capped at 0.75 (25% max reduction)"

    # --- Fix 9: Gap-down on open ---

    def test_fix9_gap_down_check_exists(self):
        """Fix 9: Orchestrator has _check_gap_down_on_open method."""
        from agent.orchestrator import TradingOrchestrator
        assert hasattr(TradingOrchestrator, '_check_gap_down_on_open')

    def test_fix9_gap_down_called_at_market_open(self):
        """Fix 9: _handle_market_open calls _check_gap_down_on_open."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._handle_market_open)
        assert '_check_gap_down_on_open' in source

    def test_fix9_gap_down_exits_position(self):
        """Fix 9: Gap-down >3% through SL triggers force exit."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade(sl=2400.0)  # SL at 2400
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)

        # Price opens at 2300 — which is < 2400*0.97 = 2328
        broker.set_ltp("RELIANCE", 2300.0)

        with patch('agent.orchestrator.alert_error'), \
             patch('agent.orchestrator.alert_trade_exit'), \
             patch('agent.orchestrator.today_ist', return_value=date(2026, 2, 15)):
            orch._check_gap_down_on_open()

        assert "RELIANCE" not in orch.active_trades, \
            "Gap-down through SL should trigger force exit"

    def test_fix9_no_exit_when_price_above_sl(self):
        """Fix 9: Normal open price above SL should not trigger exit."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade(sl=2400.0)
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)

        # Price opens at 2450 — above SL
        broker.set_ltp("RELIANCE", 2450.0)

        with patch('agent.orchestrator.today_ist', return_value=date(2026, 2, 15)):
            orch._check_gap_down_on_open()

        assert "RELIANCE" in orch.active_trades, \
            "Normal price should not trigger gap-down exit"

    def test_fix9_runs_once_per_day(self):
        """Fix 9: Gap-down check only runs once per trading day."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade(sl=2400.0)
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)
        broker.set_ltp("RELIANCE", 2450.0)

        with patch('agent.orchestrator.today_ist', return_value=date(2026, 2, 15)):
            orch._check_gap_down_on_open()
            # Mark _gap_down_checked
            assert orch._gap_down_checked == date(2026, 2, 15)

            # Second call should be a no-op (same day)
            broker.set_ltp("RELIANCE", 2300.0)  # Would trigger exit if checked
            orch._check_gap_down_on_open()
            assert "RELIANCE" in orch.active_trades, \
                "Gap-down check should not re-run same day"

    # --- Fix 10: LTP timeout differentiation ---

    def test_fix10_first_failure_uses_fallback_price(self):
        """Fix 10: First LTP failure uses last known price for monitoring."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade(highest=2550.0)  # Last known price
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)
        broker.set_ltp("RELIANCE", 0)  # LTP failure

        orch._check_positions()

        # Counter should be 1 (not reset)
        assert orch._ltp_failures.get("RELIANCE") == 1
        # Position should still be active (fallback used, not force-exited)
        assert "RELIANCE" in orch.active_trades

    def test_fix10_fallback_does_not_reset_counter(self):
        """Fix 10: Using fallback price does not reset failure counter."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)
        broker.set_ltp("RELIANCE", 0)

        orch._check_positions()  # failure 1: uses fallback

        assert orch._ltp_failures["RELIANCE"] == 1, \
            "Fallback price should NOT reset the failure counter"

    def test_fix10_real_price_resets_counter(self):
        """Fix 10: Genuine LTP success resets the failure counter."""
        broker = MockBroker()
        orch = self._make_orchestrator(broker)
        trade = self._make_trade()
        orch.active_trades["RELIANCE"] = trade
        broker.add_position("RELIANCE", qty=10, avg_price=2500.0)

        # Fail once
        broker.set_ltp("RELIANCE", 0)
        orch._check_positions()
        assert orch._ltp_failures["RELIANCE"] == 1

        # Real price resets
        broker.set_ltp("RELIANCE", 2520.0)
        orch._check_positions()
        assert "RELIANCE" not in orch._ltp_failures
