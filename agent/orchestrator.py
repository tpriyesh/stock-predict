"""
Trading Orchestrator - Main trading loop with error handling.

All timing, limits, and thresholds read from CONFIG (env-driven).

Fixes applied:
- SQLite trade persistence (crash recovery + audit trail)
- Position reconciliation (startup + periodic)
- SL modify verification with cancel+replace fallback
- Trailing stop uses real ATR (not entry-to-stop proxy)
- Stop loss check uses < (not <=) to avoid random tick exits
- Shutdown always exits positions if active_trades exist
- News check sets timestamp BEFORE calling (prevents spam on failure)
- Order placement is NOT retried (prevents double orders)
- Health monitoring via Telegram alerts
"""
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, time as dt_time, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from config.trading_config import CONFIG
from utils.platform import now_ist, today_ist, time_ist
from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    ProductType, Position, OrderResponse
)
from risk.manager import RiskManager
from agent.signal_adapter import SignalAdapter, TradeSignal, TradeDecision
from utils.rate_limiter import get_limiter
from utils.trade_db import get_trade_db
from utils.alerts import (
    alert_trade_entry, alert_trade_exit, alert_kill_switch,
    alert_error, alert_daily_report, alert_sl_modify_failed,
    alert_position_reconciliation_mismatch, alert_startup
)
from src.execution.trailing_stop import TrailingStopCalculator, TrailingStopConfig, TrailingMethod
from src.data.news_fetcher import NewsFetcher
from src.features.news_features import NewsFeatureExtractor


class TradingPhase(Enum):
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    ENTRY_WINDOW = "ENTRY_WINDOW"
    HOLDING = "HOLDING"
    SQUARE_OFF = "SQUARE_OFF"
    MARKET_CLOSED = "MARKET_CLOSED"


class TradingState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class TradeRecord:
    """Record of a trade for audit trail."""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float = 0.0
    original_stop_loss: float = 0.0
    current_stop: float = 0.0
    highest_price: float = 0.0
    target: float = 0.0
    pnl: float = 0.0
    entry_time: datetime = field(default_factory=lambda: now_ist().replace(tzinfo=None))
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    signal_confidence: float = 0.0
    order_ids: List[str] = field(default_factory=list)
    atr: float = 0.0  # Real ATR for trailing stop


class TradingOrchestrator:
    """
    Main trading loop orchestrator.

    All parameters from CONFIG (env-driven).
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: RiskManager,
        signal_adapter: Optional[SignalAdapter] = None
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_adapter = signal_adapter or SignalAdapter()

        # State
        self.state = TradingState.IDLE
        self.phase = TradingPhase.MARKET_CLOSED
        self._shutdown_requested = False
        self._broker_mode = "paper"  # Set in run()

        # Trade tracking
        self.active_trades: Dict[str, TradeRecord] = {}
        self.completed_trades: List[TradeRecord] = []
        self.pending_signals: List[TradeSignal] = []

        # Timing
        self._last_signal_refresh: Optional[datetime] = None
        self._last_position_check: Optional[datetime] = None
        self._last_news_check: Optional[datetime] = None
        self._last_reconciliation: Optional[datetime] = None
        self._consecutive_order_failures: int = 0

        # Token health check (for live broker)
        self._last_token_check: Optional[datetime] = None

        # LTP failure tracking (per symbol) for alerting
        self._ltp_failures: Dict[str, int] = {}
        LTP_FAILURE_ALERT_THRESHOLD = 5

        # Trailing stop calculator
        risk_pct = CONFIG.capital.max_per_trade_risk_pct or 0.02  # Guard against zero
        self._trailing_calc = TrailingStopCalculator(TrailingStopConfig(
            method=TrailingMethod.PERCENTAGE,
            percentage=CONFIG.strategy.trailing_distance_pct * 100,
            activation_r=CONFIG.strategy.trailing_start_pct / risk_pct,
        ))

        # Fee estimation (captured at init so tests can control via CONFIG mock)
        self._fee_pct = CONFIG.capital.estimated_fee_pct

        # News monitoring for held positions
        self._news_fetcher = NewsFetcher()
        self._news_extractor = NewsFeatureExtractor()

        # Persistence
        self._db = get_trade_db()

        # Rate limiter
        broker_name = broker.__class__.__name__.lower().replace('broker', '')
        self._limiter = get_limiter(broker_name)

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info("TradingOrchestrator initialized")
        CONFIG.print_summary()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _reset_circuit_breakers(self):
        """Reset all circuit breakers at session start to prevent stale OPEN states."""
        try:
            from utils.error_handler import CircuitBreakerRegistry
            # Reset the broker's circuit breaker registry if it has one
            if hasattr(self.broker, '_circuit') and hasattr(self.broker._circuit, 'reset_all'):
                self.broker._circuit.reset_all()
            elif hasattr(self.broker, '_circuit') and hasattr(self.broker._circuit, 'reset'):
                self.broker._circuit.reset()
            # Reset order failure counter
            self._consecutive_order_failures = 0
            logger.info("Circuit breakers reset for new session")
        except Exception as e:
            logger.warning(f"Could not reset circuit breakers: {e}")

    def run(self, paper_mode: bool = True):
        """Main trading loop."""
        self._broker_mode = "paper" if paper_mode else "live"
        logger.info(f"Starting trading loop (paper_mode={paper_mode})")
        self.state = TradingState.RUNNING

        # Reset circuit breakers from previous session
        self._reset_circuit_breakers()

        # Startup: recover from crash and reconcile
        self._recover_from_crash()
        self._reconcile_positions()

        # Alert startup
        try:
            pv = self.risk_manager.get_portfolio_value()
            alert_startup(self._broker_mode, pv)
        except Exception:
            pass

        try:
            while not self._shutdown_requested:
                try:
                    self.phase = self._get_current_phase()

                    can_trade, reason = self.risk_manager.can_trade()
                    if not can_trade:
                        logger.warning(f"Cannot trade: {reason}")
                        if self.active_trades:
                            self._emergency_exit_all("Risk manager blocked")
                        self.state = TradingState.PAUSED
                        time.sleep(60)
                        continue

                    if self.phase == TradingPhase.PRE_MARKET:
                        self._handle_pre_market()
                    elif self.phase == TradingPhase.MARKET_OPEN:
                        self._handle_market_open()
                    elif self.phase == TradingPhase.ENTRY_WINDOW:
                        self._handle_entry_window()
                    elif self.phase == TradingPhase.HOLDING:
                        self._handle_holding()
                    elif self.phase == TradingPhase.SQUARE_OFF:
                        self._handle_square_off()
                    elif self.phase == TradingPhase.MARKET_CLOSED:
                        self._handle_market_closed()

                    time.sleep(1)

                except Exception as e:
                    logger.exception(f"Error in trading loop: {e}")
                    alert_error("Trading loop", str(e))
                    self.state = TradingState.ERROR
                    time.sleep(10)

        finally:
            self._shutdown()

    def _get_current_phase(self) -> TradingPhase:
        now = time_ist()
        h = CONFIG.hours

        # Check time ranges FIRST (before is_market_open) to handle pre-market
        if now < h.pre_market_start:
            return TradingPhase.MARKET_CLOSED
        elif now < h.market_open:
            # Pre-market: load signals before market opens (don't rely on is_market_open)
            return TradingPhase.PRE_MARKET
        elif now >= h.market_close:
            return TradingPhase.MARKET_CLOSED

        # During market hours, verify market is actually open (weekday/holiday check)
        if not self.broker.is_market_open():
            return TradingPhase.MARKET_CLOSED

        if now < h.entry_window_start:
            return TradingPhase.MARKET_OPEN
        elif now < h.entry_window_end:
            return TradingPhase.ENTRY_WINDOW
        elif now < h.square_off_start:
            return TradingPhase.HOLDING
        else:
            return TradingPhase.SQUARE_OFF

    # === PHASE HANDLERS ===

    def _handle_pre_market(self):
        if self._last_signal_refresh is None:
            self._refresh_signals()
        time.sleep(30)

    def _handle_market_open(self):
        self._maybe_refresh_signals()
        self._check_positions()
        time.sleep(10)

    def _handle_entry_window(self):
        self._maybe_check_token()
        self._maybe_refresh_signals()

        if len(self.active_trades) < CONFIG.capital.max_positions:
            self._try_enter_positions()

        self._check_positions()
        self._verify_sl_orders()
        self._maybe_reconcile()
        time.sleep(CONFIG.intervals.quote_refresh_seconds)

    def _handle_holding(self):
        self._maybe_check_token()
        self._check_positions()
        self._verify_sl_orders()
        self._maybe_check_news()
        self._maybe_reconcile()
        time.sleep(CONFIG.intervals.position_check_seconds)

    def _handle_square_off(self):
        if self.active_trades:
            logger.info(f"Square off phase - exiting {len(self.active_trades)} positions")
            self._exit_all_positions("End of day square off")
        time.sleep(30)

    def _handle_market_closed(self):
        if self.state == TradingState.RUNNING:
            self._generate_daily_report()
            logger.info("Market closed. Stopping.")
            self.state = TradingState.IDLE
            self._shutdown_requested = True

    # === TOKEN HEALTH CHECK ===

    def _maybe_check_token(self):
        """Periodically verify broker token is still valid (every 30 min)."""
        if self._broker_mode == "paper":
            return  # Paper mode doesn't need token checks

        if self._last_token_check is not None:
            elapsed = (now_ist().replace(tzinfo=None) - self._last_token_check).total_seconds()
            if elapsed < CONFIG.orders.token_check_interval_seconds:
                return

        self._last_token_check = now_ist().replace(tzinfo=None)

        try:
            if hasattr(self.broker, 'check_token_valid') and not self.broker.check_token_valid():
                logger.critical("BROKER TOKEN EXPIRED - initiating emergency shutdown")
                alert_error("CRITICAL: Broker token expired",
                           "Token is invalid/expired. Exiting all positions and shutting down. "
                           "Re-authenticate with: python scripts/upstox_auth.py")
                if self.active_trades:
                    self._emergency_exit_all("Broker token expired")
                self._shutdown_requested = True
        except Exception as e:
            logger.warning(f"Token health check failed: {e}")

    # === SIGNAL MANAGEMENT ===

    def _refresh_signals(self):
        logger.info("Refreshing signals...")
        try:
            SIGNAL_TIMEOUT = 600  # 10 minutes max for signal generation

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.signal_adapter.get_trade_signals, force_refresh=True
                )
                try:
                    signals = future.result(timeout=SIGNAL_TIMEOUT)
                except FutureTimeoutError:
                    logger.error(
                        f"Signal generation timed out ({SIGNAL_TIMEOUT}s). "
                        f"News/data APIs may be slow. Skipping this cycle."
                    )
                    self._last_signal_refresh = now_ist().replace(tzinfo=None)
                    return

            self.pending_signals = [
                s for s in signals
                if s.decision == TradeDecision.BUY
                and s.symbol not in self.active_trades
            ]
            self._last_signal_refresh = now_ist().replace(tzinfo=None)
            logger.info(f"Got {len(self.pending_signals)} actionable signals")

            for sig in self.pending_signals[:5]:
                logger.info(
                    f"  {sig.symbol}: {sig.decision.value} "
                    f"(conf={sig.confidence:.2f}, R:R={sig.risk_reward_ratio:.1f})"
                )
        except Exception as e:
            logger.error(f"Failed to refresh signals: {e}")

    def _maybe_refresh_signals(self):
        if self._last_signal_refresh is None:
            self._refresh_signals()
            return
        elapsed = (now_ist().replace(tzinfo=None) - self._last_signal_refresh).total_seconds()
        if elapsed > CONFIG.intervals.signal_refresh_seconds:
            self._refresh_signals()

    # === ENTRY ===

    def _try_enter_positions(self):
        available_slots = CONFIG.capital.max_positions - len(self.active_trades)
        if available_slots <= 0 or not self.pending_signals:
            return

        # Circuit breaker: pause entries after 5 consecutive ORDER failures
        # (signal rejections don't count - only actual broker/order failures)
        if self._consecutive_order_failures >= 5:
            logger.warning("Order circuit breaker: 5 consecutive failures, pausing entries")
            alert_error("Order circuit breaker", "5 consecutive order failures, pausing new entries")
            return

        for signal in self.pending_signals[:available_slots]:
            try:
                result = self._enter_position(signal)
                if result is True:
                    self.pending_signals.remove(signal)
                    self._consecutive_order_failures = 0
                elif result is None:
                    # Signal rejected (not a broker failure) - remove stale signal
                    self.pending_signals.remove(signal)
                else:
                    # result is False - actual broker/order failure
                    self._consecutive_order_failures += 1
            except Exception as e:
                logger.error(f"Failed to enter {signal.symbol}: {e}")
                self._consecutive_order_failures += 1

    def _enter_position(self, signal: TradeSignal) -> bool:
        symbol = signal.symbol
        logger.info(f"Attempting to enter {symbol} at {signal.entry_price:.2f}")

        # Signal TTL check - reject stale signals
        # Returns None (not False) because this is a signal rejection, not a broker failure
        signal_age = (now_ist().replace(tzinfo=None) - signal.timestamp).total_seconds()
        max_age = CONFIG.signals.signal_max_age_seconds
        if signal_age > max_age:
            logger.warning(
                f"{symbol}: Signal too old ({signal_age:.0f}s > {max_age}s), skipping"
            )
            return None  # Signal rejection — don't count toward circuit breaker

        # Validate trade through risk manager BEFORE placing any order
        risk_check = self.risk_manager.validate_trade(
            symbol=symbol,
            entry_price=signal.current_price,
            stop_loss=signal.stop_loss,
            direction='BUY'
        )
        if not risk_check.allowed:
            logger.warning(f"Risk manager rejected {symbol}: {risk_check.reason}")
            return None  # Signal rejection — don't count toward circuit breaker

        # Use the SMALLER of: risk manager's max quantity vs signal's suggested size
        funds = self.broker.get_funds()
        if signal.current_price <= 0:
            logger.warning(f"Invalid price {signal.current_price} for {symbol}, skipping entry")
            return None  # Signal rejection — don't count toward circuit breaker
        signal_qty = int(funds.available_cash * signal.position_size_pct / signal.current_price)
        quantity = min(risk_check.max_quantity, signal_qty) if signal_qty > 0 else risk_check.max_quantity

        if quantity <= 0:
            logger.warning(f"Position size too small for {symbol}")
            return None  # Signal rejection — don't count toward circuit breaker

        self._limiter.wait("orders")

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product=ProductType.INTRADAY,
            tag=f"SIG_{signal.confidence:.0%}"
        )

        # NO RETRY for order placement - prevents double orders
        try:
            response = self.broker.place_order(order)
        except Exception as e:
            logger.error(f"Order placement exception for {symbol}: {e}")
            return False

        if response.status not in (OrderStatus.COMPLETE, OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIAL_FILL):
            logger.warning(f"Order not filled: {response.message}")
            return False

        # For MARKET orders, wait briefly if not immediately complete
        if response.status not in (OrderStatus.COMPLETE, OrderStatus.PARTIAL_FILL):
            wait_seconds = CONFIG.orders.entry_order_wait_seconds
            time.sleep(wait_seconds)
            order_book = self.broker.get_order_status(response.order_id)
            if not order_book or order_book.status not in (OrderStatus.COMPLETE, OrderStatus.PARTIAL_FILL):
                logger.warning(f"Order {response.order_id} not filled after {wait_seconds}s, cancelling")
                try:
                    self.broker.cancel_order(response.order_id)
                    # Verify cancellation succeeded
                    cancel_check = self.broker.get_order_status(response.order_id)
                    if cancel_check and cancel_check.status in (OrderStatus.COMPLETE, OrderStatus.PARTIAL_FILL):
                        logger.warning(f"Order {response.order_id} filled during cancel attempt, proceeding")
                        # Order filled - continue with the fill
                    else:
                        return False
                except Exception:
                    # Cancel failed - check if order filled anyway
                    recheck = self.broker.get_order_status(response.order_id)
                    if not recheck or recheck.status not in (OrderStatus.COMPLETE, OrderStatus.PARTIAL_FILL):
                        return False

        # Get fill price and actual filled quantity
        order_book = self.broker.get_order_status(response.order_id)
        fill_price = signal.current_price
        actual_quantity = quantity
        if order_book:
            if order_book.average_price and order_book.average_price > 0:
                fill_price = order_book.average_price
            # Use filled_quantity from broker (handles partial fills correctly)
            if order_book.filled_quantity is not None and order_book.filled_quantity >= 0:
                actual_quantity = order_book.filled_quantity

        # If partial fill, cancel the remaining unfilled portion to prevent
        # unexpected late fills that create untracked/unprotected positions
        if actual_quantity < quantity:
            logger.warning(
                f"{symbol}: Partial fill - got {actual_quantity}/{quantity} shares. "
                f"Cancelling remaining order."
            )
            try:
                self.broker.cancel_order(response.order_id)
            except Exception as e:
                logger.warning(f"{symbol}: Could not cancel partial order remainder: {e}")

            if actual_quantity <= 0:
                logger.warning(f"{symbol}: Zero shares filled after partial, aborting entry")
                return False

        # Get real ATR for trailing stop
        atr = self._get_atr_for_symbol(symbol, fill_price, signal.stop_loss)

        trade = TradeRecord(
            trade_id=f"{symbol}_{now_ist().replace(tzinfo=None).strftime('%H%M%S')}",
            symbol=symbol,
            side="BUY",
            quantity=actual_quantity,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            original_stop_loss=signal.stop_loss,
            current_stop=signal.stop_loss,
            highest_price=fill_price,
            target=signal.target_price,
            signal_confidence=signal.confidence,
            order_ids=[response.order_id],
            atr=atr
        )

        self.active_trades[symbol] = trade

        # Persist to DB immediately — if DB fails, exit position (fail-safe)
        try:
            self._db.record_entry(
                trade_id=trade.trade_id,
                symbol=symbol,
                side="BUY",
                quantity=actual_quantity,
                entry_price=fill_price,
                stop_loss=signal.stop_loss,
                target=signal.target_price,
                confidence=signal.confidence,
                order_ids=[response.order_id],
                broker_mode=self._broker_mode
            )

            # Record entry order
            self._db.record_order(
                order_id=response.order_id,
                trade_id=trade.trade_id,
                symbol=symbol,
                side="BUY",
                order_type="MARKET",
                quantity=actual_quantity,
                price=None,
                trigger_price=None,
                status="COMPLETE",
                tag=order.tag
            )
        except Exception as e:
            logger.error(f"DB write failed for {symbol}: {e}. Exiting position (fail-safe).")
            alert_error("DB write failed", f"{symbol}: Trade not persisted, exiting position. Error: {e}")
            self._exit_position(trade, "DB write failed - fail-safe exit")
            return False

        logger.info(
            f"Entered {symbol}: {actual_quantity} @ {fill_price:.2f} "
            f"(SL={signal.stop_loss:.2f}, T={signal.target_price:.2f})"
        )

        # Alert
        alert_trade_entry(symbol, actual_quantity, fill_price, signal.stop_loss, signal.target_price)

        # Place stop loss
        sl_order_id = self._place_stop_loss(trade)
        if not sl_order_id:
            logger.error(f"{symbol}: SL order FAILED - position unprotected, exiting immediately")
            alert_error("SL placement failed", f"{symbol}: Exiting unprotected position")
            self._exit_position(trade, "SL order placement failed")
            return False
        return True

    def _get_atr_for_symbol(self, symbol: str, fill_price: float, stop_loss: float) -> float:
        """Get real ATR for trailing stop. Falls back to risk distance."""
        try:
            df = self.signal_adapter._get_price_data(symbol)
            if df is not None and 'atr' in df.columns and len(df) > 0:
                atr_val = float(df['atr'].iloc[-1])
                if atr_val > 0:
                    return atr_val
            # If ATR column missing, try to calculate
            if df is not None and len(df) > 14:
                from src.features.technical import TechnicalIndicators
                df = TechnicalIndicators.calculate_all(df)
                if 'atr' in df.columns:
                    atr_val = float(df['atr'].iloc[-1])
                    if atr_val > 0:
                        return atr_val
        except Exception as e:
            logger.debug(f"Could not get ATR for {symbol}: {e}")

        # Fallback: use risk distance as ATR proxy
        return abs(fill_price - stop_loss)

    def _place_stop_loss(self, trade: TradeRecord) -> Optional[str]:
        self._limiter.wait("orders")
        sl_order = Order(
            symbol=trade.symbol,
            side=OrderSide.SELL,
            quantity=trade.quantity,
            order_type=OrderType.SL_M,
            trigger_price=trade.stop_loss,
            product=ProductType.INTRADAY,
            tag="SL"
        )

        # Single attempt for SL placement (no retry - it's an idempotency risk)
        try:
            response = self.broker.place_order(sl_order)
        except Exception as e:
            logger.error(f"SL placement exception for {trade.symbol}: {e}")
            return None

        if response.status in (OrderStatus.OPEN, OrderStatus.PENDING):
            trade.order_ids.append(response.order_id)
            self._db.add_order_id(trade.trade_id, response.order_id)
            self._db.record_order(
                order_id=response.order_id,
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                side="SELL",
                order_type="SL-M",
                quantity=trade.quantity,
                price=None,
                trigger_price=trade.stop_loss,
                status="OPEN",
                tag="SL"
            )
            logger.info(f"SL placed for {trade.symbol} at {trade.stop_loss:.2f}")
            return response.order_id
        else:
            logger.warning(f"Failed to place SL for {trade.symbol}: {response.message}")
            return None

    # === POSITION MONITORING ===

    def _check_positions(self):
        if not self.active_trades:
            return

        self._limiter.wait("positions")
        try:
            broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.warning(f"Failed to get broker positions, skipping position check: {e}")
            return

        for symbol, trade in list(self.active_trades.items()):
            position = broker_positions.get(symbol)

            if not position or position.quantity == 0:
                self._record_exit(trade, "Position closed externally")
                continue

            self._limiter.wait("quotes")
            current_price = self.broker.get_ltp(symbol)

            # Validate price data - skip check if price is invalid
            if not current_price or current_price <= 0:
                self._ltp_failures[symbol] = self._ltp_failures.get(symbol, 0) + 1
                count = self._ltp_failures[symbol]
                logger.warning(f"{symbol}: Invalid price {current_price} (failure #{count})")
                if count == 5:  # Alert after 5 consecutive failures
                    alert_error(
                        f"LTP failures: {symbol}",
                        f"{count} consecutive LTP fetch failures for held position {symbol}. "
                        f"Position may be unprotected! Qty={trade.quantity}, Entry={trade.entry_price:.2f}"
                    )
                continue

            # Reset LTP failure counter on success
            self._ltp_failures.pop(symbol, None)

            # Sanity check: reject absurd price moves (>20% from entry = likely data error)
            price_change_pct = abs(current_price - trade.entry_price) / trade.entry_price
            if price_change_pct > 0.20:
                logger.warning(
                    f"{symbol}: Price {current_price:.2f} is {price_change_pct:.0%} from entry "
                    f"{trade.entry_price:.2f} - possible data error, skipping"
                )
                continue

            # 1. Check stop loss hit (use < not <= to avoid random tick exits)
            if current_price < trade.current_stop:
                logger.warning(f"{symbol} STOP LOSS hit at {current_price:.2f} (stop={trade.current_stop:.2f})")
                self._exit_position(trade, f"Stop loss hit ({trade.current_stop:.2f})")
                continue

            # 2. Check target hit
            if current_price >= trade.target:
                logger.info(f"{symbol} TARGET hit at {current_price:.2f}")
                self._exit_position(trade, "Target hit")
                continue

            # 3. Update highest price tracking
            if current_price > trade.highest_price:
                trade.highest_price = current_price

            # 4. Calculate trailing stop using real ATR
            new_stop = self._trailing_calc.calculate(
                entry_price=trade.entry_price,
                original_stop=trade.original_stop_loss,
                current_stop=trade.current_stop,
                highest_price=trade.highest_price,
                current_price=current_price,
                atr=trade.atr,
                is_long=True
            )

            # 5. Update stop if trailing moved it up
            if new_stop > trade.current_stop:
                old_stop = trade.current_stop
                trade.current_stop = new_stop
                trade.stop_loss = new_stop
                logger.info(
                    f"{symbol} trailing stop: {old_stop:.2f} -> {new_stop:.2f} "
                    f"(price={current_price:.2f}, high={trade.highest_price:.2f})"
                )
                # Persist stop update
                self._db.update_stop(trade.trade_id, new_stop, trade.highest_price)
                # Try to modify the broker SL order with verification
                self._update_broker_stop(trade, new_stop)

    def _verify_sl_orders(self):
        """
        Safety net for live brokers: if price is below SL and SL order
        is still OPEN after 30s grace period, force a market sell.

        Catches edge cases where broker SL orders don't trigger
        (exchange issues, gap-down through SL, stuck orders).
        """
        if self._broker_mode == "paper":
            return  # Paper broker handles SL internally

        for symbol, trade in list(self.active_trades.items()):
            if len(trade.order_ids) < 2:
                continue  # No SL order placed

            sl_order_id = trade.order_ids[-1]
            try:
                self._limiter.wait("orders")
                sl_status = self.broker.get_order_status(sl_order_id)

                if not sl_status or sl_status.status != OrderStatus.OPEN:
                    continue  # SL already filled or cancelled

                # SL is still OPEN — check if price is below trigger
                current_price = self.broker.get_ltp(symbol)
                if not current_price or current_price <= 0:
                    continue

                if current_price < trade.current_stop:
                    # Price is below SL but SL hasn't triggered
                    if not hasattr(trade, '_sl_breach_time') or trade._sl_breach_time is None:
                        trade._sl_breach_time = now_ist().replace(tzinfo=None)
                        logger.warning(
                            f"{symbol}: Price {current_price:.2f} < SL {trade.current_stop:.2f} "
                            f"but SL order still OPEN. Grace period started."
                        )
                        continue

                    elapsed = (now_ist().replace(tzinfo=None) - trade._sl_breach_time).total_seconds()
                    if elapsed >= 30:
                        logger.critical(
                            f"{symbol}: SL STUCK! Price {current_price:.2f} < SL {trade.current_stop:.2f} "
                            f"for {elapsed:.0f}s. Forcing market sell."
                        )
                        alert_error(
                            f"SL ORDER STUCK: {symbol}",
                            f"Price={current_price:.2f}, SL={trade.current_stop:.2f}. "
                            f"SL order OPEN for {elapsed:.0f}s. Forcing market sell!"
                        )
                        self._exit_position(trade, "SL order stuck - forced market sell")
                else:
                    # Price recovered above SL — reset breach timer
                    if hasattr(trade, '_sl_breach_time'):
                        trade._sl_breach_time = None

            except Exception as e:
                logger.warning(f"SL verification failed for {symbol}: {e}")

    def _maybe_check_news(self):
        """Check news for held positions periodically."""
        if not self.active_trades:
            return
        if self._last_news_check is not None:
            elapsed = (now_ist().replace(tzinfo=None) - self._last_news_check).total_seconds()
            if elapsed < CONFIG.intervals.news_check_seconds:
                return
        # Set timestamp BEFORE calling to prevent spam on failure
        self._last_news_check = now_ist().replace(tzinfo=None)
        self._check_news_for_positions()

    def _check_news_for_positions(self):
        """Fetch fresh news for held positions and tighten stops on bad news."""
        logger.info("Checking news for held positions...")
        news_available = False

        for symbol, trade in list(self.active_trades.items()):
            try:
                articles = self._news_fetcher.fetch_for_symbol(symbol, hours=4)
                if not articles:
                    logger.debug(f"{symbol}: No news articles found (news unavailable/stale)")
                    continue
                news_available = True

                articles = self._news_extractor.extract_batch(articles)
                summary = self._news_extractor.get_symbol_news_summary(articles, symbol)
                avg_sentiment = summary.get('avg_sentiment', 0)

                # Negative sentiment: tighten stop to breakeven or closer
                if avg_sentiment < -0.3 and len(articles) >= 2:
                    breakeven_stop = trade.entry_price
                    if breakeven_stop > trade.current_stop:
                        old_stop = trade.current_stop
                        trade.current_stop = breakeven_stop
                        trade.stop_loss = breakeven_stop
                        logger.warning(
                            f"{symbol} NEGATIVE NEWS detected (sentiment={avg_sentiment:.2f}, "
                            f"{len(articles)} articles) -> stop tightened to breakeven "
                            f"({old_stop:.2f} -> {breakeven_stop:.2f})"
                        )
                        self._db.update_stop(trade.trade_id, breakeven_stop, trade.highest_price)
                        self._update_broker_stop(trade, breakeven_stop)
                elif avg_sentiment < -0.5:
                    # Very negative sentiment with even 1 article - exit immediately
                    logger.warning(
                        f"{symbol} STRONGLY NEGATIVE NEWS (sentiment={avg_sentiment:.2f}) -> exiting"
                    )
                    self._exit_position(trade, f"Negative news (sentiment={avg_sentiment:.2f})")

            except Exception as e:
                logger.warning(f"News check failed for {symbol}: {e}")

        if self.active_trades and not news_available:
            logger.warning("NEWS STALENESS: No news data available for any held positions")

    def _update_broker_stop(self, trade: TradeRecord, new_stop: float):
        """Update the broker-side SL order with verification and cancel+replace fallback."""
        if len(trade.order_ids) < 2:
            return
        sl_order_id = trade.order_ids[-1]
        try:
            self._limiter.wait("orders")
            result = self.broker.modify_order(
                order_id=sl_order_id,
                trigger_price=new_stop
            )

            # Verify the modification succeeded
            if result:
                self._limiter.wait("orders")
                order_book = self.broker.get_order_status(sl_order_id)
                if order_book and order_book.trigger_price and abs(order_book.trigger_price - new_stop) < 0.01:
                    logger.debug(f"Broker SL order verified for {trade.symbol}: {new_stop:.2f}")
                    self._db.update_order_status(sl_order_id, "OPEN")
                    return
                else:
                    logger.warning(f"{trade.symbol}: SL modify returned OK but verification failed, trying cancel+replace")
            else:
                logger.warning(f"{trade.symbol}: SL modify returned False, trying cancel+replace")

            # Fallback: cancel old SL and place new one
            self._cancel_and_replace_sl(trade, sl_order_id, new_stop)

        except Exception as e:
            logger.warning(f"Failed to modify broker SL for {trade.symbol}: {e}")
            alert_sl_modify_failed(trade.symbol, trade.current_stop, new_stop)
            # Try cancel+replace as last resort
            try:
                self._cancel_and_replace_sl(trade, sl_order_id, new_stop)
            except Exception as e2:
                logger.error(f"Cancel+replace also failed for {trade.symbol}: {e2}")
                alert_error("SL update completely failed", f"{trade.symbol}: Position may be unprotected")

    def _cancel_and_replace_sl(self, trade: TradeRecord, old_order_id: str, new_stop: float):
        """Cancel existing SL and place a new one."""
        try:
            self._limiter.wait("orders")
            self.broker.cancel_order(old_order_id)
            self._db.update_order_status(old_order_id, "CANCELLED")
        except Exception:
            pass  # May already be cancelled/filled

        new_sl_id = self._place_stop_loss_at(trade, new_stop)
        if new_sl_id:
            trade.order_ids.append(new_sl_id)
            self._db.add_order_id(trade.trade_id, new_sl_id)
            logger.info(f"{trade.symbol}: SL replaced successfully at {new_stop:.2f}")
        else:
            logger.error(f"{trade.symbol}: Failed to replace SL - position may be unprotected!")
            alert_error("SL replace failed", f"{trade.symbol}: No stop loss protection!")

    def _place_stop_loss_at(self, trade: TradeRecord, stop_price: float) -> Optional[str]:
        """Place a new SL order at a specific price."""
        self._limiter.wait("orders")
        sl_order = Order(
            symbol=trade.symbol,
            side=OrderSide.SELL,
            quantity=trade.quantity,
            order_type=OrderType.SL_M,
            trigger_price=stop_price,
            product=ProductType.INTRADAY,
            tag="SL_REPLACE"
        )
        try:
            response = self.broker.place_order(sl_order)
            if response.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                self._db.record_order(
                    order_id=response.order_id,
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    side="SELL",
                    order_type="SL-M",
                    quantity=trade.quantity,
                    price=None,
                    trigger_price=stop_price,
                    status="OPEN",
                    tag="SL_REPLACE"
                )
                return response.order_id
        except Exception as e:
            logger.error(f"SL replacement order failed for {trade.symbol}: {e}")
        return None

    # === EXIT ===

    def _exit_position(self, trade: TradeRecord, reason: str):
        symbol = trade.symbol
        logger.info(f"Exiting {symbol}: {reason}")

        # Check if broker SL order already filled BEFORE cancelling or placing new sell
        # This prevents the double-sell race condition on live brokers
        sl_already_filled = False
        sl_fill_price = None
        for order_id in trade.order_ids[1:]:  # SL orders are after the entry order
            try:
                self._limiter.wait("orders")
                order_status = self.broker.get_order_status(order_id)
                if order_status and order_status.status == OrderStatus.COMPLETE:
                    sl_already_filled = True
                    sl_fill_price = order_status.average_price
                    logger.info(f"{symbol}: SL order {order_id} already filled @ {sl_fill_price}")
                    self._db.update_order_status(order_id, "COMPLETE")
                    break
            except Exception as e:
                logger.warning(f"Failed to check SL order {order_id}: {e}")

        if sl_already_filled:
            # SL already filled at broker - just record the exit, don't place another sell
            exit_price = sl_fill_price if sl_fill_price and sl_fill_price > 0 else None
            self._record_exit(trade, reason, exit_price)
            return

        # Cancel remaining pending SL orders (not yet filled)
        for order_id in trade.order_ids[1:]:
            try:
                self._limiter.wait("orders")
                self.broker.cancel_order(order_id)
                self._db.update_order_status(order_id, "CANCELLED")
            except Exception as e:
                logger.warning(f"Failed to cancel order {order_id}: {e}")

        # Verify position still exists at broker before selling
        # (SL could have filled between our check and cancel)
        try:
            broker_pos = self.broker.get_position(symbol)
            if not broker_pos or broker_pos.quantity <= 0:
                logger.warning(f"{symbol}: Position no longer at broker, recording exit without sell")
                # Try to find the actual exit price from the last SL fill
                for order_id in reversed(trade.order_ids[1:]):
                    try:
                        order_status = self.broker.get_order_status(order_id)
                        if order_status and order_status.status == OrderStatus.COMPLETE:
                            sl_fill_price = order_status.average_price
                            break
                    except Exception:
                        pass
                self._record_exit(trade, reason, sl_fill_price)
                return
        except Exception as e:
            logger.warning(f"Could not verify position for {symbol}: {e}, proceeding with exit")

        # Try exit with exponential backoff retry
        max_retries = CONFIG.orders.exit_max_retries
        base_delay = CONFIG.orders.exit_retry_base_delay
        for attempt in range(max_retries):
            self._limiter.wait("orders")
            exit_order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=trade.quantity,
                order_type=OrderType.MARKET,
                product=ProductType.INTRADAY,
                tag="EXIT" if attempt == 0 else f"EXIT_RETRY_{attempt}"
            )

            try:
                response = self.broker.place_order(exit_order)
            except Exception as e:
                logger.error(f"Exit order exception for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
                alert_error(f"CRITICAL: Exit order failed {max_retries} times",
                           f"{symbol}: {trade.quantity} shares stuck open! Manual intervention needed. Error: {e}")
                return

            if response.status == OrderStatus.COMPLETE:
                order_book = self.broker.get_order_status(response.order_id)
                exit_price = None
                if order_book and order_book.average_price and order_book.average_price > 0:
                    exit_price = order_book.average_price
                if not exit_price:
                    exit_price = self.broker.get_ltp(symbol) or trade.entry_price
                self._record_exit(trade, reason, exit_price)
                self._db.record_order(
                    order_id=response.order_id,
                    trade_id=trade.trade_id,
                    symbol=symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=trade.quantity,
                    price=None,
                    trigger_price=None,
                    status="COMPLETE",
                    tag=exit_order.tag
                )
                return  # Success - exit the retry loop

            # Order placed but not filled
            if response.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                # Wait briefly for fill
                time.sleep(3)
                order_book = self.broker.get_order_status(response.order_id)
                if order_book and order_book.status == OrderStatus.COMPLETE:
                    exit_price = order_book.average_price or self.broker.get_ltp(symbol) or trade.entry_price
                    self._record_exit(trade, reason, exit_price)
                    self._db.record_order(
                        order_id=response.order_id,
                        trade_id=trade.trade_id,
                        symbol=symbol,
                        side="SELL",
                        order_type="MARKET",
                        quantity=trade.quantity,
                        price=None,
                        trigger_price=None,
                        status="COMPLETE",
                        tag=exit_order.tag
                    )
                    return  # Filled after wait

            # Attempt failed - retry with exponential backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Exit attempt {attempt+1}/{max_retries} failed for {symbol}: "
                    f"{response.message}. Retrying in {delay:.0f}s..."
                )
                try:
                    self.broker.cancel_order(response.order_id)
                except Exception:
                    pass
                time.sleep(delay)
            else:
                # All attempts failed - CRITICAL alert
                logger.critical(
                    f"CRITICAL: Failed to exit {symbol} after {max_retries} attempts: {response.message}"
                )
                alert_error("CRITICAL: Exit failed - position stuck open",
                           f"{symbol}: {trade.quantity} shares @ entry {trade.entry_price:.2f}. "
                           f"Tried {max_retries} times. Manual intervention needed! "
                           f"Last error: {response.message}")

    def _record_exit(self, trade: TradeRecord, reason: str, exit_price: Optional[float] = None):
        symbol = trade.symbol

        if not exit_price or exit_price <= 0:
            try:
                exit_price = self.broker.get_ltp(symbol)
            except Exception:
                pass
        if not exit_price or exit_price <= 0:
            logger.warning(f"{symbol}: Could not get valid exit price, using entry price")
            exit_price = trade.entry_price

        trade.exit_price = exit_price
        trade.exit_time = now_ist().replace(tzinfo=None)
        trade.exit_reason = reason

        # Gross P&L (before fees)
        gross_pnl = (exit_price - trade.entry_price) * trade.quantity

        # Deduct estimated brokerage/STT/fees (applied on both buy + sell turnover)
        fee_pct = self._fee_pct
        turnover = (trade.entry_price * trade.quantity) + (exit_price * trade.quantity)
        estimated_fees = turnover * fee_pct
        trade.pnl = gross_pnl - estimated_fees

        self.completed_trades.append(trade)
        # Cap in-memory list to prevent unbounded growth (rest stays in DB)
        max_in_memory = CONFIG.capital.max_completed_trades_in_memory
        if len(self.completed_trades) > max_in_memory:
            self.completed_trades = self.completed_trades[-max_in_memory:]
        self.active_trades.pop(symbol, None)

        # Persist exit to DB
        self._db.record_exit(
            trade_id=trade.trade_id,
            exit_price=exit_price,
            pnl=trade.pnl,
            exit_reason=reason
        )

        # Update risk manager daily PnL tracking
        self.risk_manager.on_trade_complete(trade.pnl)

        pnl_str = f"+{trade.pnl:.2f}" if trade.pnl >= 0 else f"{trade.pnl:.2f}"
        logger.info(f"Closed {symbol}: P&L = {pnl_str} ({reason})")

        # Alert
        alert_trade_exit(symbol, trade.quantity, trade.entry_price, exit_price, trade.pnl, reason)

    def _exit_all_positions(self, reason: str):
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades.get(symbol)
            if not trade:
                continue  # Already removed by a previous exit in this loop
            try:
                self._exit_position(trade, reason)
            except Exception as e:
                logger.error(f"Failed to exit {symbol}: {e}")

    def _emergency_exit_all(self, reason: str):
        logger.warning(f"EMERGENCY EXIT: {reason}")
        alert_error("Emergency exit", reason)
        self._exit_all_positions(f"EMERGENCY: {reason}")

    # === RECONCILIATION ===

    def _maybe_reconcile(self):
        """Periodic position reconciliation (every 2 minutes)."""
        if self._last_reconciliation is not None:
            elapsed = (now_ist().replace(tzinfo=None) - self._last_reconciliation).total_seconds()
            if elapsed < 120:
                return
        self._last_reconciliation = now_ist().replace(tzinfo=None)
        self._reconcile_positions()

    def _reconcile_positions(self):
        """Compare broker positions to internal active_trades and alert on mismatch."""
        try:
            self._limiter.wait("positions")
            broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.warning(f"Reconciliation failed - could not get broker positions: {e}")
            return

        broker_symbols = set(s for s, p in broker_positions.items() if p.quantity > 0)
        internal_symbols = set(self.active_trades.keys())

        # Positions at broker but not tracked internally
        orphaned = broker_symbols - internal_symbols
        if orphaned:
            logger.warning(f"RECONCILIATION: Broker has positions not tracked internally: {orphaned}")
            alert_position_reconciliation_mismatch(
                list(orphaned), list(internal_symbols)
            )
            # Save snapshot for audit
            self._db.save_position_snapshot([
                {
                    'symbol': s,
                    'quantity': broker_positions[s].quantity,
                    'average_price': broker_positions[s].average_price,
                    'last_price': broker_positions[s].last_price,
                    'pnl': broker_positions[s].pnl,
                    'source': 'broker_orphaned'
                }
                for s in orphaned
            ])

        # Positions tracked internally but not at broker
        missing = internal_symbols - broker_symbols
        if missing:
            logger.warning(f"RECONCILIATION: Internal trades not found at broker: {missing}")
            for symbol in missing:
                trade = self.active_trades.get(symbol)
                if not trade:
                    continue  # Already removed by a concurrent exit
                self._record_exit(trade, "Position closed externally (reconciliation)")

        # Quantity mismatches
        for symbol in broker_symbols & internal_symbols:
            trade = self.active_trades.get(symbol)
            if not trade:
                continue  # Removed during missing-symbol exits above
            broker_qty = broker_positions[symbol].quantity
            if broker_qty != trade.quantity:
                logger.warning(
                    f"RECONCILIATION: {symbol} quantity mismatch - "
                    f"broker={broker_qty}, internal={trade.quantity}"
                )
                # Update internal to match broker and persist
                trade.quantity = broker_qty
                try:
                    self._db.save_position_snapshot([{
                        'symbol': symbol,
                        'quantity': broker_qty,
                        'average_price': broker_positions[symbol].average_price,
                        'last_price': broker_positions[symbol].last_price,
                        'pnl': broker_positions[symbol].pnl,
                        'source': 'qty_mismatch_fix'
                    }])
                except Exception:
                    pass

    def _recover_from_crash(self):
        """On startup, check DB for trades that were OPEN but never closed.

        Two-phase recovery:
        1. Mark previous-day OPEN trades as ORPHANED (definitely stale)
        2. Check same-day OPEN trades against broker positions:
           - If broker has the position, reload into active_trades
           - If broker doesn't have it, mark as ORPHANED (crash after exit)
        """
        # Phase 1: Mark previous-day orphans
        orphaned_count = self._db.mark_orphaned_trades()
        if orphaned_count > 0:
            logger.warning(f"Found {orphaned_count} orphaned trades from previous crash")
            alert_error("Crash recovery", f"Found {orphaned_count} orphaned trades from previous sessions")

        # Phase 2: Check same-day OPEN trades
        open_trades = self._db.get_open_trades()
        if not open_trades:
            return

        try:
            broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.warning(f"Could not get broker positions during crash recovery: {e}")
            return

        for db_trade in open_trades:
            symbol = db_trade['symbol']
            if symbol in broker_positions and broker_positions[symbol].quantity > 0:
                # Position still exists at broker — reload into active_trades
                import json as _json
                order_ids = _json.loads(db_trade.get('order_ids', '[]'))
                trade = TradeRecord(
                    trade_id=db_trade['trade_id'],
                    symbol=symbol,
                    side=db_trade['side'],
                    quantity=broker_positions[symbol].quantity,
                    entry_price=db_trade['entry_price'],
                    stop_loss=db_trade.get('stop_loss', 0),
                    original_stop_loss=db_trade.get('original_stop_loss', 0),
                    current_stop=db_trade.get('current_stop', 0),
                    highest_price=db_trade.get('highest_price', 0),
                    target=db_trade.get('target', 0),
                    signal_confidence=db_trade.get('signal_confidence', 0),
                    order_ids=order_ids,
                )
                self.active_trades[symbol] = trade
                logger.info(f"Crash recovery: Reloaded {symbol} ({trade.quantity} shares @ {trade.entry_price})")
            else:
                # Position not at broker — try to find actual exit price
                exit_price = db_trade['entry_price']  # Default: assume flat
                pnl = 0.0
                try:
                    # Try broker LTP as best approximation of exit price
                    ltp = self.broker.get_ltp(symbol)
                    if ltp and ltp > 0:
                        exit_price = ltp
                        entry_price = db_trade['entry_price']
                        qty = db_trade.get('quantity', 0)
                        pnl = (exit_price - entry_price) * qty
                        logger.info(f"Crash recovery: {symbol} exit price from LTP: {exit_price:.2f}")
                except Exception as e:
                    logger.warning(f"Crash recovery: Could not get LTP for {symbol}: {e}")

                self._db.record_exit(
                    trade_id=db_trade['trade_id'],
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason="Crash recovery - position not found at broker"
                )
                logger.warning(f"Crash recovery: {symbol} not at broker, marked closed (exit={exit_price:.2f}, P&L={pnl:.2f})")

    # === REPORTING ===

    def _generate_daily_report(self):
        if not self.completed_trades:
            logger.info("No trades today")
            return

        total_pnl = sum(t.pnl for t in self.completed_trades)
        winners = [t for t in self.completed_trades if t.pnl > 0]
        losers = [t for t in self.completed_trades if t.pnl < 0]

        logger.info("=" * 60)
        logger.info("DAILY TRADING REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {len(self.completed_trades)}")
        logger.info(f"Winners: {len(winners)}")
        logger.info(f"Losers: {len(losers)}")
        logger.info(f"Win Rate: {len(winners)/len(self.completed_trades)*100:.1f}%")
        logger.info(f"Total P&L: {total_pnl:+.2f}")
        logger.info("=" * 60)

        for trade in self.completed_trades:
            pnl_str = f"+{trade.pnl:.2f}" if trade.pnl >= 0 else f"{trade.pnl:.2f}"
            logger.info(
                f"  {trade.symbol}: {trade.quantity} @ {trade.entry_price:.2f} -> "
                f"{trade.exit_price:.2f} = {pnl_str} ({trade.exit_reason})"
            )

        # Save daily summary to DB
        try:
            portfolio_value = self.risk_manager.get_portfolio_value()
            self._db.save_daily_summary(
                total_trades=len(self.completed_trades),
                winners=len(winners),
                losers=len(losers),
                total_pnl=total_pnl,
                portfolio_value=portfolio_value,
                broker_mode=self._broker_mode
            )
        except Exception as e:
            logger.warning(f"Failed to save daily summary: {e}")

        # Send Telegram alert
        try:
            alert_daily_report(
                len(self.completed_trades), len(winners),
                total_pnl, self.risk_manager.get_portfolio_value()
            )
        except Exception:
            pass

        # Write daily report to file
        try:
            self._write_daily_report_file(
                total_pnl=total_pnl,
                winners=len(winners),
                losers=len(losers),
                portfolio_value=self.risk_manager.get_portfolio_value()
            )
        except Exception as e:
            logger.warning(f"Failed to write daily report file: {e}")

    def _write_daily_report_file(self, total_pnl: float, winners: int,
                                  losers: int, portfolio_value: float):
        """Write a human-readable daily report to data/reports/YYYY-MM-DD.txt."""
        from pathlib import Path
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        today_str = today_ist().isoformat()
        report_path = reports_dir / f"{today_str}.txt"

        total = len(self.completed_trades)
        win_rate = winners / total * 100 if total > 0 else 0

        lines = [
            "=" * 60,
            f"  DAILY TRADING REPORT - {today_str}",
            f"  Mode: {self._broker_mode.upper()}",
            "=" * 60,
            "",
            f"  Total Trades:  {total}",
            f"  Winners:       {winners}",
            f"  Losers:        {losers}",
            f"  Win Rate:      {win_rate:.1f}%",
            f"  Total P&L:     {'+'if total_pnl>=0 else ''}Rs.{total_pnl:.2f}",
            f"  Portfolio:     Rs.{portfolio_value:,.2f}",
            "",
            "-" * 60,
            "  TRADE DETAILS",
            "-" * 60,
        ]

        for trade in self.completed_trades:
            pnl_str = f"+Rs.{trade.pnl:.2f}" if trade.pnl >= 0 else f"Rs.{trade.pnl:.2f}"
            lines.append(
                f"  {trade.symbol}: {trade.quantity} @ Rs.{trade.entry_price:.2f} "
                f"-> Rs.{trade.exit_price:.2f} = {pnl_str} ({trade.exit_reason})"
            )

        # Append issues summary from today's issues log
        lines.append("")
        issues_file = Path("logs") / f"issues_{today_ist().strftime('%Y-%m-%d')}.log"
        if issues_file.exists():
            issue_lines = issues_file.read_text().strip().splitlines()
            if issue_lines:
                # Categorize issues
                categories = {
                    'API Rate Limit': [],
                    'Data Gaps': [],
                    'Broker/Order': [],
                    'System': [],
                    'Other': [],
                }
                for line in issue_lines:
                    lower = line.lower()
                    if '403' in lower or 'rate limit' in lower or 'forbidden' in lower:
                        categories['API Rate Limit'].append(line)
                    elif 'no data' in lower or 'fetch' in lower or 'timeout' in lower or 'yfinance' in lower:
                        categories['Data Gaps'].append(line)
                    elif 'broker' in lower or 'order' in lower or 'token' in lower:
                        categories['Broker/Order'].append(line)
                    elif 'disk' in lower or 'memory' in lower or 'pid' in lower or 'lock' in lower:
                        categories['System'].append(line)
                    else:
                        categories['Other'].append(line)

                lines.append("-" * 60)
                lines.append(f"  ISSUES TODAY ({len(issue_lines)} total)")
                lines.append("-" * 60)
                for cat, items in categories.items():
                    if items:
                        lines.append(f"  [{cat}]: {len(items)} occurrences")
                        # Show first 3 unique messages per category
                        seen = set()
                        for item in items[:10]:
                            msg = item.split('|')[-1].strip()[:80]
                            if msg not in seen:
                                seen.add(msg)
                                lines.append(f"    - {msg}")
                            if len(seen) >= 3:
                                break

        # API health section
        try:
            from providers.quota import get_quota_manager
            from providers.cache import get_response_cache
            health_report = get_quota_manager().format_health_report()
            if health_report:
                lines.append("")
                lines.append(health_report)
            cache_report = get_response_cache().format_stats_report()
            if cache_report:
                lines.append(cache_report)
        except Exception:
            pass

        lines.append("")
        lines.append(f"  Generated at {now_ist().strftime('%H:%M:%S')} IST")
        lines.append("=" * 60)

        report_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Daily report written to {report_path}")

    # === SHUTDOWN ===

    def _shutdown(self):
        logger.info("Shutting down orchestrator...")
        # Always try to exit if positions exist (even if market appears closed)
        if self.active_trades:
            logger.warning("Positions still open during shutdown!")
            # Use timeout to prevent hanging on unresponsive broker
            import threading
            exit_done = threading.Event()

            def _exit_with_timeout():
                try:
                    self._exit_all_positions("Agent shutdown")
                finally:
                    exit_done.set()

            t = threading.Thread(target=_exit_with_timeout, daemon=True)
            t.start()
            if not exit_done.wait(timeout=120):
                logger.critical(
                    "SHUTDOWN TIMEOUT: Could not exit all positions within 120s. "
                    f"Stuck positions: {list(self.active_trades.keys())}"
                )
                alert_error(
                    "SHUTDOWN TIMEOUT",
                    f"Could not exit positions within 120s. "
                    f"Manual intervention needed: {list(self.active_trades.keys())}"
                )
        self._generate_daily_report()
        self.state = TradingState.STOPPED
        logger.info("Orchestrator stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'phase': self.phase.value,
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'pending_signals': len(self.pending_signals),
            'daily_pnl': sum(t.pnl for t in self.completed_trades),
            'positions': list(self.active_trades.keys())
        }
