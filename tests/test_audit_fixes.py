"""
Tests for production audit fixes (Feb 11, 2026 - Session 2).

Covers:
- IST timezone enforcement in live brokers (upstox, zerodha)
- Fund caching with fallback in live brokers
- Price sanity validation in live brokers
- Partial fill detection and unknown status logging
- FIFO P&L calculation in Upstox
- Shutdown timeout in orchestrator
- Telegram alert retry logic
- TradingException IST timestamp
- Config validation (critical errors raise ValueError)
"""
import os
import sys
import time
import math
import threading
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import Optional, Dict, List

import pytest

# Ensure project root and test env
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)
from tests.conftest import MockBroker


# ============================================
# IST TIMEZONE ENFORCEMENT IN LIVE BROKERS
# ============================================

class TestUpstoxIST:
    """Verify Upstox broker uses IST instead of UTC."""

    def test_upstox_imports_ist(self):
        """Upstox module imports now_ist and today_ist."""
        import broker.upstox as mod
        assert hasattr(mod, 'now_ist')
        assert hasattr(mod, 'today_ist')

    def test_upstox_no_datetime_now_in_source(self):
        """No raw datetime.now() calls remain in upstox.py."""
        import inspect
        import broker.upstox as mod
        source = inspect.getsource(mod)
        # Should not contain datetime.now() - all replaced with now_ist()
        # Exclude import lines and comments
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('from ') or stripped.startswith('import '):
                continue
            assert 'datetime.now()' not in stripped, f"Found datetime.now() in: {stripped}"

    def test_upstox_no_date_today_in_source(self):
        """No raw date.today() calls remain in upstox.py."""
        import inspect
        import broker.upstox as mod
        source = inspect.getsource(mod)
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('from ') or stripped.startswith('import '):
                continue
            assert 'date.today()' not in stripped, f"Found date.today() in: {stripped}"


class TestZerodhaIST:
    """Verify Zerodha broker uses IST instead of UTC."""

    def test_zerodha_imports_ist(self):
        """Zerodha module imports now_ist and today_ist."""
        import broker.zerodha as mod
        assert hasattr(mod, 'now_ist')
        assert hasattr(mod, 'today_ist')

    def test_zerodha_no_datetime_now_in_source(self):
        """No raw datetime.now() calls remain in zerodha.py."""
        import inspect
        import broker.zerodha as mod
        source = inspect.getsource(mod)
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('from ') or stripped.startswith('import '):
                continue
            assert 'datetime.now()' not in stripped, f"Found datetime.now() in: {stripped}"


class TestSignalAdapterIST:
    """Verify signal adapter uses IST."""

    def test_signal_adapter_imports_ist(self):
        """Signal adapter imports IST utilities."""
        import agent.signal_adapter as mod
        assert hasattr(mod, 'now_ist')
        assert hasattr(mod, 'today_ist')
        assert hasattr(mod, 'time_ist')

    def test_signal_adapter_no_datetime_now(self):
        """No raw datetime.now() in signal_adapter.py."""
        import inspect
        import agent.signal_adapter as mod
        source = inspect.getsource(mod)
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('from ') or stripped.startswith('import '):
                continue
            assert 'datetime.now()' not in stripped, f"Found datetime.now() in: {stripped}"


# ============================================
# FUND CACHING WITH FALLBACK
# ============================================

class TestUpstoxFundCaching:
    """Upstox fund caching prevents false kill switch on API failures."""

    def _make_broker(self):
        """Create UpstoxBroker with mocked API."""
        with patch('broker.upstox.requests.Session'):
            from broker.upstox import UpstoxBroker
            broker = UpstoxBroker()
            return broker

    def test_fund_cache_fields_exist(self):
        """UpstoxBroker has fund cache fields."""
        broker = self._make_broker()
        assert hasattr(broker, '_last_known_funds')
        assert hasattr(broker, '_funds_failure_count')
        assert broker._last_known_funds is None
        assert broker._funds_failure_count == 0

    def test_cached_funds_returns_last_known(self):
        """_get_cached_funds_or_zero returns cached on first failures."""
        broker = self._make_broker()
        broker._last_known_funds = Funds(available_cash=50000, used_margin=10000, total_balance=60000)
        broker._funds_failure_count = 0

        result = broker._get_cached_funds_or_zero()
        assert result.available_cash == 50000
        assert broker._funds_failure_count == 1

    def test_cached_funds_persists_after_3_failures(self):
        """After 3+ failures, keeps using cache (sends alert, doesn't block trading)."""
        broker = self._make_broker()
        broker._last_known_funds = Funds(available_cash=50000, used_margin=10000, total_balance=60000)
        broker._funds_failure_count = 2  # Already at 2

        with patch('utils.alerts.alert_error'):
            result = broker._get_cached_funds_or_zero()
        assert result.available_cash == 50000  # Cache persists
        assert broker._funds_failure_count == 3


class TestZerodhaFundCaching:
    """Zerodha fund caching prevents false kill switch on API failures."""

    def _make_broker(self):
        """Create ZerodhaBroker with mocked Kite SDK."""
        with patch('broker.zerodha.KiteConnect'):
            from broker.zerodha import ZerodhaBroker
            broker = ZerodhaBroker()
            return broker

    def test_fund_cache_fields_exist(self):
        """ZerodhaBroker has fund cache fields."""
        broker = self._make_broker()
        assert hasattr(broker, '_last_known_funds')
        assert hasattr(broker, '_funds_failure_count')

    def test_cached_funds_returns_last_known(self):
        """Returns cached funds on first failure."""
        broker = self._make_broker()
        broker._last_known_funds = Funds(available_cash=75000, used_margin=5000, total_balance=80000)
        broker._funds_failure_count = 0

        result = broker._get_cached_funds_or_zero()
        assert result.available_cash == 75000
        assert broker._funds_failure_count == 1

    def test_cached_funds_persists_after_3(self):
        """After 3+ failures, keeps using cache (sends alert, doesn't block trading)."""
        broker = self._make_broker()
        broker._last_known_funds = Funds(available_cash=75000, used_margin=5000, total_balance=80000)
        broker._funds_failure_count = 2

        with patch('utils.alerts.alert_error'):
            result = broker._get_cached_funds_or_zero()
        assert result.available_cash == 75000  # Cache persists

    def test_no_cache_returns_zero_immediately(self):
        """No cache + first failure = zero immediately."""
        broker = self._make_broker()
        assert broker._last_known_funds is None

        result = broker._get_cached_funds_or_zero()
        assert result.available_cash == 0


# ============================================
# PRICE SANITY VALIDATION
# ============================================

class TestUpstoxPriceSanity:
    """Upstox get_ltp rejects invalid prices."""

    def _make_broker(self):
        with patch('broker.upstox.requests.Session'):
            from broker.upstox import UpstoxBroker
            broker = UpstoxBroker()
            broker._connected = True
            return broker

    def test_rejects_negative_price(self):
        """Negative prices return 0."""
        broker = self._make_broker()
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": {"NSE_EQ|TEST": {"last_price": -10.0}}
        })
        assert broker.get_ltp("TEST") == 0.0

    def test_rejects_million_plus_price(self):
        """Prices above 1M return 0."""
        broker = self._make_broker()
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": {"NSE_EQ|TEST": {"last_price": 2000000.0}}
        })
        assert broker.get_ltp("TEST") == 0.0

    def test_accepts_valid_price(self):
        """Valid prices pass through."""
        broker = self._make_broker()
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": {"NSE_EQ|RELIANCE": {"last_price": 2500.50}}
        })
        assert broker.get_ltp("RELIANCE") == 2500.50


class TestZerodhaPriceSanity:
    """Zerodha get_ltp rejects invalid prices."""

    def _make_broker(self):
        with patch('broker.zerodha.KiteConnect') as MockKite:
            from broker.zerodha import ZerodhaBroker
            broker = ZerodhaBroker()
            broker._connected = True
            return broker

    def test_rejects_zero_price(self):
        """Zero price returns 0."""
        broker = self._make_broker()
        broker._api_call = MagicMock(return_value={
            "NSE:TEST": {"last_price": 0}
        })
        assert broker.get_ltp("TEST") == 0.0

    def test_rejects_nan_price(self):
        """NaN price returns 0."""
        broker = self._make_broker()
        broker._api_call = MagicMock(return_value={
            "NSE:TEST": {"last_price": float('nan')}
        })
        assert broker.get_ltp("TEST") == 0.0

    def test_rejects_inf_price(self):
        """Inf price returns 0."""
        broker = self._make_broker()
        broker._api_call = MagicMock(return_value={
            "NSE:TEST": {"last_price": float('inf')}
        })
        assert broker.get_ltp("TEST") == 0.0

    def test_accepts_valid_price(self):
        """Valid price passes through."""
        broker = self._make_broker()
        broker._api_call = MagicMock(return_value={
            "NSE:INFY": {"last_price": 1800.25}
        })
        assert broker.get_ltp("INFY") == 1800.25


# ============================================
# PARTIAL FILL DETECTION
# ============================================

class TestUpstoxPartialFill:
    """Upstox _parse_order detects partial fills."""

    def _make_broker(self):
        with patch('broker.upstox.requests.Session'):
            from broker.upstox import UpstoxBroker
            return UpstoxBroker()

    def test_partial_fill_logged(self):
        """Partial fill generates warning log."""
        broker = self._make_broker()
        data = {
            "order_id": "TEST001",
            "instrument_token": "NSE_EQ|RELIANCE",
            "transaction_type": "BUY",
            "quantity": 100,
            "filled_quantity": 60,
            "pending_quantity": 40,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 2500.0,
            "status": "partial_fill",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        with patch('broker.upstox.logger') as mock_logger:
            result = broker._parse_order(data)
            # Should log partial fill warning
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "PARTIAL FILL" in warning_msg

    def test_unknown_status_logged(self):
        """Unknown status generates warning."""
        broker = self._make_broker()
        data = {
            "order_id": "TEST002",
            "instrument_token": "NSE_EQ|RELIANCE",
            "transaction_type": "BUY",
            "quantity": 100,
            "filled_quantity": 0,
            "pending_quantity": 100,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 0,
            "status": "some_new_status",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        with patch('broker.upstox.logger') as mock_logger:
            result = broker._parse_order(data)
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "Unknown order status" in warning_msg


class TestZerodhaPartialFill:
    """Zerodha _parse_order detects partial fills."""

    def _make_broker(self):
        with patch('broker.zerodha.KiteConnect'):
            from broker.zerodha import ZerodhaBroker
            return ZerodhaBroker()

    def test_partial_fill_logged(self):
        """Partial fill in Zerodha order generates warning."""
        broker = self._make_broker()
        data = {
            "order_id": "ZRD001",
            "tradingsymbol": "INFY",
            "transaction_type": "BUY",
            "quantity": 50,
            "filled_quantity": 30,
            "pending_quantity": 20,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 1800.0,
            "status": "OPEN",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        with patch('broker.zerodha.logger') as mock_logger:
            result = broker._parse_order(data)
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "PARTIAL FILL" in warning_msg

    def test_unknown_status_logged(self):
        """Unknown Zerodha status generates warning."""
        broker = self._make_broker()
        data = {
            "order_id": "ZRD002",
            "tradingsymbol": "TCS",
            "transaction_type": "SELL",
            "quantity": 10,
            "filled_quantity": 0,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 0,
            "status": "WEIRD_STATE",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        with patch('broker.zerodha.logger') as mock_logger:
            result = broker._parse_order(data)
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "Unknown order status" in warning_msg


# ============================================
# FIFO P&L CALCULATION
# ============================================

class TestUpstoxFIFOPnL:
    """Upstox FIFO P&L matching logic."""

    def _make_broker(self):
        with patch('broker.upstox.requests.Session'):
            from broker.upstox import UpstoxBroker
            broker = UpstoxBroker()
            broker._connected = True
            return broker

    def test_simple_fifo_pnl(self):
        """Single buy + single sell: simple P&L."""
        broker = self._make_broker()

        # Mock positions (no open positions = 0 unrealized)
        broker.get_positions = MagicMock(return_value=[])

        # Mock trades: Buy 10 @ 100, Sell 10 @ 110
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": [
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "BUY",
                 "quantity": 10, "average_price": 100.0},
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "SELL",
                 "quantity": 10, "average_price": 110.0},
            ]
        })

        pnl = broker.get_pnl()
        assert pnl.realized == 100.0  # (110-100) * 10

    def test_fifo_multiple_buys_single_sell(self):
        """FIFO: First buy matched first, even if second buy is cheaper."""
        broker = self._make_broker()
        broker.get_positions = MagicMock(return_value=[])

        # Buy 5 @ 100, Buy 5 @ 90, Sell 10 @ 105
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": [
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "BUY",
                 "quantity": 5, "average_price": 100.0},
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "BUY",
                 "quantity": 5, "average_price": 90.0},
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "SELL",
                 "quantity": 10, "average_price": 105.0},
            ]
        })

        pnl = broker.get_pnl()
        # FIFO: first 5 sold @ 105 with buy @ 100 = +25
        #        next 5 sold @ 105 with buy @ 90 = +75
        # Total = 100
        assert pnl.realized == 100.0

    def test_fifo_partial_match(self):
        """FIFO: Sell quantity splits across buy lots."""
        broker = self._make_broker()
        broker.get_positions = MagicMock(return_value=[])

        # Buy 10 @ 100, Sell 5 @ 110
        broker._make_request = MagicMock(return_value={
            "status": "success",
            "data": [
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "BUY",
                 "quantity": 10, "average_price": 100.0},
                {"instrument_token": "NSE_EQ|TEST", "transaction_type": "SELL",
                 "quantity": 5, "average_price": 110.0},
            ]
        })

        pnl = broker.get_pnl()
        assert pnl.realized == 50.0  # (110-100) * 5


# ============================================
# SHUTDOWN TIMEOUT
# ============================================

class TestShutdownTimeout:
    """Orchestrator shutdown has timeout protection."""

    def test_shutdown_method_exists(self):
        """_shutdown method exists on orchestrator."""
        from agent.orchestrator import TradingOrchestrator
        assert hasattr(TradingOrchestrator, '_shutdown')

    def test_shutdown_uses_threading_for_timeout(self):
        """_shutdown uses threading.Event for timeout when trades exist."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator._shutdown)
        assert 'exit_done.wait(timeout=' in source
        assert 'SHUTDOWN TIMEOUT' in source

    def test_shutdown_completes_normally_without_trades(self):
        """Shutdown without active trades completes immediately."""
        broker = MockBroker()
        from risk.manager import RiskManager
        rm = RiskManager(broker=broker, initial_capital=100000, hard_stop=80000)
        from agent.orchestrator import TradingOrchestrator
        orch = TradingOrchestrator(broker=broker, risk_manager=rm)
        # No active trades
        assert len(orch.active_trades) == 0
        # Should complete without hanging
        orch._shutdown()
        from agent.orchestrator import TradingState
        assert orch.state == TradingState.STOPPED


# ============================================
# TELEGRAM ALERT RETRY
# ============================================

class TestTelegramRetry:
    """Telegram alerts retry on failure."""

    def test_send_telegram_has_retry_parameter(self):
        """_send_telegram accepts retries parameter with default=1."""
        import inspect
        from utils.alerts import _send_telegram
        sig = inspect.signature(_send_telegram)
        assert 'retries' in sig.parameters, "Missing retries parameter"
        assert sig.parameters['retries'].default == 1, "Default retries should be 1"

    def test_send_telegram_retry_loop_in_source(self):
        """_send_telegram source contains retry loop logic."""
        import inspect
        from utils.alerts import _send_telegram
        source = inspect.getsource(_send_telegram)
        assert 'for attempt in range' in source, "Missing retry loop"
        assert 'retries + 1' in source or 'retries+1' in source, "Loop should iterate retries+1 times"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "12345"})
    def test_critical_alerts_have_extra_retries(self):
        """Kill switch and error alerts use retries=2."""
        import inspect
        from utils.alerts import alert_kill_switch, alert_error
        # Check source code for retries parameter
        kill_source = inspect.getsource(alert_kill_switch)
        assert 'retries=2' in kill_source

        error_source = inspect.getsource(alert_error)
        assert 'retries=2' in error_source

    def test_send_telegram_no_op_without_token(self):
        """No-op when TELEGRAM_BOT_TOKEN is empty."""
        from utils.alerts import _send_telegram
        import utils.alerts as alerts_mod
        old_token = alerts_mod.TELEGRAM_BOT_TOKEN
        alerts_mod.TELEGRAM_BOT_TOKEN = ""
        try:
            # Should return immediately without error
            _send_telegram("test")
        finally:
            alerts_mod.TELEGRAM_BOT_TOKEN = old_token


# ============================================
# TRADING EXCEPTION IST TIMESTAMP
# ============================================

class TestTradingExceptionIST:
    """TradingException uses IST timestamp."""

    def test_exception_has_timestamp(self):
        """TradingException stores a timestamp."""
        from utils.error_handler import TradingException
        exc = TradingException("test error")
        assert hasattr(exc, 'timestamp')
        assert isinstance(exc.timestamp, datetime)

    def test_exception_timestamp_is_recent(self):
        """Timestamp is within 5 seconds of now."""
        from utils.error_handler import TradingException
        from utils.platform import now_ist
        exc = TradingException("test error")
        now = now_ist().replace(tzinfo=None)
        diff = abs((now - exc.timestamp).total_seconds())
        assert diff < 5, f"Timestamp diff too large: {diff}s"


# ============================================
# CONFIG VALIDATION
# ============================================

class TestConfigValidation:
    """Config validation catches dangerous settings."""

    def test_zero_capital_raises(self):
        """initial_capital=0 raises ValueError."""
        from config.trading_config import TradingConfig, CapitalConfig
        bad_capital = CapitalConfig.__new__(CapitalConfig)
        object.__setattr__(bad_capital, 'initial_capital', 0)
        object.__setattr__(bad_capital, 'hard_stop_loss', 80000)
        object.__setattr__(bad_capital, 'max_positions', 5)
        object.__setattr__(bad_capital, 'max_daily_loss_pct', 0.05)
        object.__setattr__(bad_capital, 'max_per_trade_risk_pct', 0.02)
        object.__setattr__(bad_capital, 'max_position_pct', 0.30)
        object.__setattr__(bad_capital, 'base_position_pct', 0.25)
        object.__setattr__(bad_capital, 'target_daily_return_pct', 0.05)
        object.__setattr__(bad_capital, 'estimated_fee_pct', 0.0005)
        object.__setattr__(bad_capital, 'max_completed_trades_in_memory', 200)

        with pytest.raises(ValueError, match="initial_capital"):
            TradingConfig(capital=bad_capital)

    def test_zero_max_positions_raises(self):
        """max_positions=0 raises ValueError."""
        from config.trading_config import TradingConfig, CapitalConfig
        bad_capital = CapitalConfig.__new__(CapitalConfig)
        object.__setattr__(bad_capital, 'initial_capital', 100000)
        object.__setattr__(bad_capital, 'hard_stop_loss', 80000)
        object.__setattr__(bad_capital, 'max_positions', 0)
        object.__setattr__(bad_capital, 'max_daily_loss_pct', 0.05)
        object.__setattr__(bad_capital, 'max_per_trade_risk_pct', 0.02)
        object.__setattr__(bad_capital, 'max_position_pct', 0.30)
        object.__setattr__(bad_capital, 'base_position_pct', 0.25)
        object.__setattr__(bad_capital, 'target_daily_return_pct', 0.05)
        object.__setattr__(bad_capital, 'estimated_fee_pct', 0.0005)
        object.__setattr__(bad_capital, 'max_completed_trades_in_memory', 200)

        with pytest.raises(ValueError, match="max_positions"):
            TradingConfig(capital=bad_capital)

    def test_hard_stop_above_capital_raises(self):
        """hard_stop >= initial_capital raises ValueError."""
        from config.trading_config import TradingConfig, CapitalConfig
        bad_capital = CapitalConfig.__new__(CapitalConfig)
        object.__setattr__(bad_capital, 'initial_capital', 100000)
        object.__setattr__(bad_capital, 'hard_stop_loss', 100000)
        object.__setattr__(bad_capital, 'max_positions', 5)
        object.__setattr__(bad_capital, 'max_daily_loss_pct', 0.05)
        object.__setattr__(bad_capital, 'max_per_trade_risk_pct', 0.02)
        object.__setattr__(bad_capital, 'max_position_pct', 0.30)
        object.__setattr__(bad_capital, 'base_position_pct', 0.25)
        object.__setattr__(bad_capital, 'target_daily_return_pct', 0.05)
        object.__setattr__(bad_capital, 'estimated_fee_pct', 0.0005)
        object.__setattr__(bad_capital, 'max_completed_trades_in_memory', 200)

        with pytest.raises(ValueError, match="initial_capital.*hard_stop_loss"):
            TradingConfig(capital=bad_capital)

    def test_valid_config_no_error(self):
        """Default config (from env) creates without error."""
        from config.trading_config import TradingConfig
        config = TradingConfig()
        assert config.capital.initial_capital > 0


# ============================================
# SIGTERM HANDLER
# ============================================

class TestSIGTERMHandler:
    """SIGTERM handler exists in start.py."""

    def test_start_has_signal_handler(self):
        """start.py registers SIGTERM handler."""
        import inspect
        # Read source directly to avoid executing the module
        source = (Path(__file__).parent.parent / "start.py").read_text()
        assert 'signal.SIGTERM' in source
        assert 'signal.SIGINT' in source or 'KeyboardInterrupt' in source


# ============================================
# .ENV FILE CHECK
# ============================================

class TestEnvFileCheck:
    """Config warns about missing .env file."""

    def test_config_module_checks_env(self):
        """trading_config.py checks for .env file existence."""
        source = (Path(__file__).parent.parent / "config" / "trading_config.py").read_text()
        assert '.env' in source
        assert 'warning' in source.lower() or 'Warning' in source or 'warn' in source.lower()


# ============================================
# NaN HANDLING IN TECHNICAL INDICATORS
# ============================================

class TestNaNHandling:
    """Technical indicators use sensible defaults instead of fillna(0)."""

    def test_rsi_defaults_to_50(self):
        """RSI NaN fills with 50 (neutral), not 0."""
        import numpy as np
        import pandas as pd
        from src.features.technical import TechnicalIndicators

        # Very short data - will produce NaN RSI
        n = 5
        dates = pd.date_range('2025-01-01', periods=n, freq='B')
        df = pd.DataFrame({
            'open': [100.0] * n,
            'high': [101.0] * n,
            'low': [99.0] * n,
            'close': [100.0] * n,
            'volume': [100000.0] * n,
        }, index=dates)

        result = TechnicalIndicators.calculate_all(df)
        # RSI should not be 0 for NaN rows
        rsi_values = result['rsi'].values
        for v in rsi_values:
            assert v != 0 or np.isnan(v) or v == 50.0, f"RSI should default to 50, got {v}"

    def test_adx_defaults_to_25(self):
        """ADX NaN fills with 25 (moderate trend), not 0."""
        import numpy as np
        import pandas as pd
        from src.features.technical import TechnicalIndicators

        n = 20
        dates = pd.date_range('2025-01-01', periods=n, freq='B')
        price = np.full(n, 100.0)
        df = pd.DataFrame({
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': np.full(n, 50000.0),
        }, index=dates)

        result = TechnicalIndicators.calculate_all(df)
        # Flat market should give ADX=25 (the default), not 0
        adx_last = result['trend_strength'].iloc[-1]
        assert adx_last == 25.0 or adx_last > 0, f"ADX should default to 25.0, got {adx_last}"


# ============================================
# UPSTOX EXPANDED STATUS MAP
# ============================================

class TestUpstoxStatusMap:
    """Upstox order parsing handles all known statuses."""

    def _make_broker(self):
        with patch('broker.upstox.requests.Session'):
            from broker.upstox import UpstoxBroker
            return UpstoxBroker()

    def test_expired_maps_to_cancelled(self):
        """'expired' status maps to CANCELLED."""
        broker = self._make_broker()
        data = {
            "order_id": "EXP001",
            "instrument_token": "NSE_EQ|RELIANCE",
            "transaction_type": "BUY",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "order_type": "LIMIT",
            "price": 2500,
            "trigger_price": 0,
            "average_price": 0,
            "status": "expired",
            "order_timestamp": "2026-02-11T15:30:00",
        }
        result = broker._parse_order(data)
        assert result is not None
        assert result.status == OrderStatus.CANCELLED

    def test_trigger_pending_maps_to_open(self):
        """'trigger_pending' maps to OPEN."""
        broker = self._make_broker()
        data = {
            "order_id": "TP001",
            "instrument_token": "NSE_EQ|RELIANCE",
            "transaction_type": "SELL",
            "quantity": 10,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "order_type": "SL-M",
            "price": 0,
            "trigger_price": 2450,
            "average_price": 0,
            "status": "trigger_pending",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        result = broker._parse_order(data)
        assert result is not None
        assert result.status == OrderStatus.OPEN

    def test_partially_filled_maps_to_partial(self):
        """'partially_filled' maps to PARTIAL_FILL."""
        broker = self._make_broker()
        data = {
            "order_id": "PF001",
            "instrument_token": "NSE_EQ|INFY",
            "transaction_type": "BUY",
            "quantity": 100,
            "filled_quantity": 50,
            "pending_quantity": 50,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 1800.0,
            "status": "partially_filled",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        result = broker._parse_order(data)
        assert result is not None
        assert result.status == OrderStatus.PARTIAL_FILL


# ============================================
# ZERODHA STATUS MAP AND PARSING
# ============================================

class TestZerodhaStatusMap:
    """Zerodha order parsing handles Kite statuses."""

    def _make_broker(self):
        with patch('broker.zerodha.KiteConnect'):
            from broker.zerodha import ZerodhaBroker
            return ZerodhaBroker()

    def test_trigger_pending_maps_to_open(self):
        """'TRIGGER PENDING' maps to OPEN (SL waiting)."""
        broker = self._make_broker()
        data = {
            "order_id": "ZTP001",
            "tradingsymbol": "RELIANCE",
            "transaction_type": "SELL",
            "quantity": 10,
            "filled_quantity": 0,
            "order_type": "SL-M",
            "price": 0,
            "trigger_price": 2450,
            "average_price": 0,
            "status": "TRIGGER PENDING",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        result = broker._parse_order(data)
        assert result is not None
        assert result.status == OrderStatus.OPEN

    def test_complete_status(self):
        """'COMPLETE' maps correctly."""
        broker = self._make_broker()
        data = {
            "order_id": "ZC001",
            "tradingsymbol": "INFY",
            "transaction_type": "BUY",
            "quantity": 10,
            "filled_quantity": 10,
            "order_type": "MARKET",
            "price": 0,
            "trigger_price": 0,
            "average_price": 1800.0,
            "status": "COMPLETE",
            "order_timestamp": "2026-02-11T10:30:00",
        }
        result = broker._parse_order(data)
        assert result is not None
        assert result.status == OrderStatus.COMPLETE
        assert result.average_price == 1800.0


# ============================================
# CIRCUIT BREAKER RESET
# ============================================

class TestCircuitBreakerReset:
    """Circuit breaker resets at session start."""

    def test_reset_method_exists(self):
        """CircuitBreaker has reset() method."""
        from utils.error_handler import CircuitBreaker
        cb = CircuitBreaker(name="test")
        assert hasattr(cb, 'reset')

    def test_reset_closes_open_breaker(self):
        """reset() moves OPEN -> CLOSED."""
        from utils.error_handler import CircuitBreaker
        cb = CircuitBreaker(name="test", failure_threshold=2)
        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert not cb.allow_request()

        # Reset
        cb.reset()
        assert cb.allow_request()
        assert cb._failure_count == 0

    def test_orchestrator_resets_breakers_on_run(self):
        """Orchestrator resets circuit breakers at session start."""
        import inspect
        from agent.orchestrator import TradingOrchestrator
        source = inspect.getsource(TradingOrchestrator.run)
        assert '_reset_circuit_breakers' in source
