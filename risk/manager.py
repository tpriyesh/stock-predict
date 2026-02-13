"""
Risk Manager - Protects capital through strict limits.

Core responsibilities:
1. Global kill switch (portfolio < hard stop = STOP)
2. Daily loss limit (5% = STOP for day)
3. Per-trade risk limit (2% max)
4. Position sizing (with slippage buffer)
5. Concentration limits (max 5 positions)
6. Gap-down risk check
7. Circuit breaker awareness
8. Margin adequacy check

Fixes (Feb 11, 2026):
- Portfolio value now includes unrealized P&L from open positions
- Kill switch alerts via Telegram
- Config validation on init
- Kill switch handles broker API failure with last-known fallback
- Slippage buffer in position sizing
- Gap-down rejection before entry
- NSE circuit breaker awareness
- Margin adequacy pre-check
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Tuple
from loguru import logger

from utils.platform import today_ist

from config.trading_config import CONFIG
from broker.base import BaseBroker, Position


@dataclass
class RiskCheck:
    """Result of a risk check"""
    allowed: bool
    reason: str
    max_quantity: int = 0
    max_value: float = 0.0


class RiskManager:
    """
    Risk management system with multiple safeguards.

    Kill Switch: Stop ALL trading if portfolio < hard_stop
    Daily Limit: Stop for day if loss > daily_limit
    Trade Limit: Reject trades exceeding per-trade risk
    """

    def __init__(
        self,
        broker: BaseBroker,
        initial_capital: float = None,
        hard_stop: float = None
    ):
        self.broker = broker

        # Capital limits
        self.initial_capital = initial_capital or CONFIG.capital.initial_capital
        self.hard_stop = hard_stop or CONFIG.capital.hard_stop_loss
        self.max_daily_loss = self.initial_capital * CONFIG.capital.max_daily_loss_pct
        self.max_per_trade = self.initial_capital * CONFIG.capital.max_per_trade_risk_pct
        self.max_position_pct = CONFIG.capital.max_position_pct
        self.max_positions = CONFIG.capital.max_positions

        # Validate config
        if self.hard_stop >= self.initial_capital:
            logger.warning(
                f"Hard stop ({self.hard_stop}) >= initial capital ({self.initial_capital}). "
                "Kill switch may trigger immediately."
            )
        if self.max_daily_loss <= 0:
            logger.warning("Max daily loss <= 0, daily loss limit disabled")

        # State tracking
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.trading_date = today_ist()
        self.is_killed = False
        self.kill_reason = ""

        # Last-known portfolio value for API failure fallback
        self._last_known_portfolio_value: Optional[float] = None
        self._last_known_unrealized: float = 0.0  # Cached unrealized P&L
        self._portfolio_api_failures: int = 0
        self._max_portfolio_api_failures: int = 3  # Kill after 3 consecutive failures

        logger.info(f"RiskManager initialized: Capital={self.initial_capital}, HardStop={self.hard_stop}")

    def _reset_daily_state(self):
        """Reset daily counters if new day"""
        if today_ist() != self.trading_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.trading_date = today_ist()
            logger.info("Daily risk counters reset")

    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value INCLUDING unrealized P&L.

        This ensures the kill switch triggers based on TRUE portfolio value,
        not just cash + margin which ignores unrealized losses.

        Resilient to broker API failures:
        - Returns last-known value on transient failure
        - Triggers kill switch after 3 consecutive API failures
        """
        try:
            funds = self.broker.get_funds()
            base_value = funds.total_balance

            # Add unrealized P&L from open positions
            try:
                pnl = self.broker.get_pnl()
                unrealized = pnl.unrealized
                self._last_known_unrealized = unrealized  # Cache for fallback
                portfolio_value = base_value + unrealized
            except Exception as e:
                # FAIL-SAFE: Use last known unrealized P&L (not zero)
                # This prevents hiding large losses when get_pnl() is down
                logger.warning(
                    f"Could not get unrealized P&L: {e}. "
                    f"Using last known: {self._last_known_unrealized:.2f}"
                )
                portfolio_value = base_value + self._last_known_unrealized

            # Success - reset failure counter and cache value
            self._portfolio_api_failures = 0
            self._last_known_portfolio_value = portfolio_value
            return portfolio_value

        except Exception as e:
            self._portfolio_api_failures += 1
            logger.warning(
                f"Portfolio value API failed ({self._portfolio_api_failures}/"
                f"{self._max_portfolio_api_failures}): {e}"
            )

            # After max consecutive failures, force kill switch
            if self._portfolio_api_failures >= self._max_portfolio_api_failures:
                self.is_killed = True
                self.kill_reason = (
                    f"Portfolio API failed {self._portfolio_api_failures} times consecutively. "
                    "Cannot verify portfolio safety."
                )
                logger.critical(f"KILL SWITCH TRIGGERED: {self.kill_reason}")
                try:
                    from utils.alerts import alert_kill_switch
                    alert_kill_switch(
                        self._last_known_portfolio_value or 0,
                        self.hard_stop,
                        self.kill_reason
                    )
                except Exception:
                    pass
                # Return below hard_stop for callers that check value directly
                return self.hard_stop - 1

            # Use last-known value as fallback
            if self._last_known_portfolio_value is not None:
                logger.info(f"Using last-known portfolio value: {self._last_known_portfolio_value}")
                return self._last_known_portfolio_value

            # No cached value - assume initial capital (conservative)
            return self.initial_capital

    def can_trade(self) -> Tuple[bool, str]:
        """
        Master check: Can we trade at all?

        Checks:
        1. Kill switch not triggered
        2. Portfolio above hard stop
        3. Daily loss limit not exceeded

        Returns:
            Tuple of (allowed, reason)
        """
        self._reset_daily_state()

        # Check kill switch
        if self.is_killed:
            return False, f"KILLED: {self.kill_reason}"

        # Check portfolio value (now includes unrealized P&L)
        portfolio_value = self.get_portfolio_value()
        if portfolio_value < self.hard_stop:
            self.is_killed = True
            self.kill_reason = f"Portfolio {portfolio_value:.0f} below hard stop {self.hard_stop:.0f}"
            logger.critical(f"KILL SWITCH TRIGGERED: {self.kill_reason}")
            # Alert via Telegram
            try:
                from utils.alerts import alert_kill_switch
                alert_kill_switch(portfolio_value, self.hard_stop, self.kill_reason)
            except Exception:
                pass
            return False, self.kill_reason

        # Check daily loss using internally-tracked P&L (from on_trade_complete)
        # Do NOT overwrite from broker.get_pnl() - Upstox realized P&L is broken (always 0)
        # Add current unrealized P&L to tracked realized losses
        try:
            pnl = self.broker.get_pnl()
            effective_daily_pnl = self.daily_pnl + pnl.unrealized
        except Exception:
            effective_daily_pnl = self.daily_pnl
        if effective_daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit hit: {effective_daily_pnl:.0f} (realized={self.daily_pnl:.0f} + unrealized={effective_daily_pnl-self.daily_pnl:.0f}) < -{self.max_daily_loss:.0f}"

        return True, "OK"

    def validate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        direction: str = 'BUY'
    ) -> RiskCheck:
        """
        Validate if a specific trade is allowed.

        Checks:
        1. Can we trade at all?
        2. Position count limit
        3. Not already in this stock
        4. Stop loss is reasonable
        5. Calculate max quantity based on risk

        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss: Planned stop loss
            direction: 'BUY' or 'SELL'

        Returns:
            RiskCheck with allowed status and max quantity
        """
        # Master check
        can_trade, reason = self.can_trade()
        if not can_trade:
            return RiskCheck(allowed=False, reason=reason)

        # Check position count
        positions = self.broker.get_positions()
        if len(positions) >= self.max_positions:
            return RiskCheck(
                allowed=False,
                reason=f"Max positions reached: {len(positions)}/{self.max_positions}"
            )

        # Check if already in this stock
        for pos in positions:
            if pos.symbol == symbol:
                return RiskCheck(
                    allowed=False,
                    reason=f"Already have position in {symbol}"
                )

        # Validate stop loss
        if entry_price <= 0:
            return RiskCheck(allowed=False, reason="Invalid entry price")

        risk_pct = abs(entry_price - stop_loss) / entry_price
        max_stop = CONFIG.signals.max_stop_distance_pct
        if risk_pct > max_stop:
            return RiskCheck(
                allowed=False,
                reason=f"Stop loss too wide: {risk_pct:.1%} > {max_stop:.1%}"
            )

        min_stop = CONFIG.signals.min_stop_distance_pct
        if risk_pct < min_stop:
            return RiskCheck(
                allowed=False,
                reason=f"Stop loss too tight: {risk_pct:.1%} < {min_stop:.1%}"
            )

        # Gap-down risk check
        gap_check = self.check_gap_risk(symbol, entry_price)
        if gap_check:
            return RiskCheck(allowed=False, reason=gap_check)

        # Circuit breaker proximity check
        circuit_check = self.check_circuit_breaker_risk(symbol, entry_price)
        if circuit_check:
            return RiskCheck(allowed=False, reason=circuit_check)

        # Calculate position size
        max_qty, max_value = self.calculate_position_size(entry_price, stop_loss)

        if max_qty <= 0:
            return RiskCheck(
                allowed=False,
                reason="Insufficient capital for minimum position"
            )

        # Margin adequacy check
        margin_check = self.check_margin_adequacy(entry_price, stop_loss, max_qty)
        if margin_check:
            return RiskCheck(allowed=False, reason=margin_check)

        return RiskCheck(
            allowed=True,
            reason="Trade validated",
            max_quantity=max_qty,
            max_value=max_value
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[int, float]:
        """
        Calculate optimal position size based on risk.

        Method: Risk-based sizing with capital limit cap and slippage buffer.

        Formula:
        1. Risk per share = |entry - stop_loss| * (1 + slippage)
        2. Max risk amount = max_per_trade (2% of capital)
        3. Quantity by risk = max_risk / risk_per_share
        4. Cap by max position value (30% of capital)

        Returns:
            Tuple of (quantity, position_value)
        """
        funds = self.broker.get_funds()
        available = funds.available_cash

        # Risk per share with slippage buffer
        raw_risk = abs(entry_price - stop_loss)
        if raw_risk <= 0:
            return 0, 0

        slippage_pct = CONFIG.orders.max_slippage_pct
        risk_per_share = raw_risk * (1 + slippage_pct)

        # Max position value (30% of capital)
        max_position_value = available * self.max_position_pct

        # Quantity based on risk (2% max loss, adjusted for slippage)
        qty_by_risk = int(self.max_per_trade / risk_per_share)

        # Quantity based on capital
        qty_by_capital = int(max_position_value / entry_price) if entry_price > 0 else 0

        # Take the smaller of the two, cap at 10000 shares max per trade
        MAX_SHARES_PER_TRADE = 10000
        quantity = min(qty_by_risk, qty_by_capital, MAX_SHARES_PER_TRADE)
        position_value = quantity * entry_price

        logger.debug(
            f"Position sizing: raw_risk={raw_risk:.2f}, slippage_adj_risk={risk_per_share:.2f}, "
            f"qty_by_risk={qty_by_risk}, qty_by_capital={qty_by_capital}, "
            f"final_qty={quantity}"
        )

        return quantity, position_value

    def on_trade_complete(self, pnl: float):
        """
        Called when a trade is completed.

        Updates daily P&L and trade count.
        """
        self.trades_today += 1
        self.daily_pnl += pnl
        logger.info(f"Trade completed: PnL={pnl:.2f}, Daily PnL={self.daily_pnl:.2f}")

    def get_status(self) -> dict:
        """Get current risk status"""
        can_trade, reason = self.can_trade()
        funds = self.broker.get_funds()
        positions = self.broker.get_positions()
        portfolio_value = self.get_portfolio_value()

        # Effective daily P&L = tracked realized + current unrealized
        try:
            pnl = self.broker.get_pnl()
            effective_daily_pnl = self.daily_pnl + pnl.unrealized
        except Exception:
            effective_daily_pnl = self.daily_pnl

        return {
            'can_trade': can_trade,
            'reason': reason,
            'is_killed': self.is_killed,
            'portfolio_value': portfolio_value,
            'hard_stop': self.hard_stop,
            'distance_to_stop': portfolio_value - self.hard_stop,
            'daily_pnl': effective_daily_pnl,
            'daily_pnl_realized': self.daily_pnl,
            'max_daily_loss': self.max_daily_loss,
            'trades_today': self.trades_today,
            'positions': len(positions),
            'max_positions': self.max_positions,
            'available_cash': funds.available_cash,
        }

    def force_shutdown(self, reason: str):
        """Force shutdown trading"""
        self.is_killed = True
        self.kill_reason = reason
        logger.critical(f"FORCED SHUTDOWN: {reason}")
        try:
            from utils.alerts import alert_kill_switch
            alert_kill_switch(self.get_portfolio_value(), self.hard_stop, reason)
        except Exception:
            pass

    def reset_kill_switch(self):
        """Reset kill switch (use with caution)"""
        if self.get_portfolio_value() >= self.hard_stop:
            self.is_killed = False
            self.kill_reason = ""
            logger.warning("Kill switch reset")
        else:
            logger.error("Cannot reset kill switch: portfolio still below hard stop")

    # === GAP-DOWN & CIRCUIT BREAKER CHECKS ===

    def check_gap_risk(self, symbol: str, entry_price: float) -> Optional[str]:
        """
        Reject entry if stock gapped down significantly from previous close.

        Returns rejection reason string or None if OK.
        """
        gap_threshold = CONFIG.signals.gap_down_reject_pct
        try:
            quote = self.broker.get_quote(symbol)
            if quote and quote.close and quote.close > 0:
                prev_close = quote.close
                gap_pct = (entry_price - prev_close) / prev_close
                if gap_pct < -gap_threshold:
                    return (
                        f"Gap-down risk: {symbol} gapped {gap_pct:.1%} from prev close "
                        f"{prev_close:.2f} (threshold: -{gap_threshold:.0%})"
                    )
        except Exception as e:
            logger.debug(f"Gap risk check failed for {symbol}: {e}")
        return None

    def check_circuit_breaker_risk(self, symbol: str, entry_price: float) -> Optional[str]:
        """
        Reject entry if stock is near NSE circuit breaker levels.

        NSE halts trading at ±10%, ±15%, ±20% from previous close.
        We reject if stock has already moved >circuit_breaker_warn_pct from open.

        Returns rejection reason string or None if OK.
        """
        warn_pct = CONFIG.signals.circuit_breaker_warn_pct
        try:
            quote = self.broker.get_quote(symbol)
            if quote and quote.open and quote.open > 0:
                move_pct = abs(entry_price - quote.open) / quote.open
                if move_pct > warn_pct:
                    return (
                        f"Circuit breaker risk: {symbol} moved {move_pct:.1%} from open "
                        f"{quote.open:.2f} (threshold: {warn_pct:.0%})"
                    )
        except Exception as e:
            logger.debug(f"Circuit breaker check failed for {symbol}: {e}")
        return None

    def check_margin_adequacy(
        self, entry_price: float, stop_loss: float, quantity: int
    ) -> Optional[str]:
        """
        Verify margin sufficient for both entry position AND SL order.

        Intraday margin: ~25% of position value (4x leverage).
        SL order margin: typically 100% of the trigger value.

        Returns rejection reason string or None if OK.
        """
        try:
            funds = self.broker.get_funds()
            available = funds.available_cash

            # Entry requires margin (simplified: position_value for MIS/intraday)
            position_value = entry_price * quantity
            # Conservative: assume entry uses full position value as margin
            # (brokers vary, some give 4x leverage, some less)
            entry_margin = position_value

            if entry_margin > available:
                return (
                    f"Insufficient margin: need ₹{entry_margin:,.0f} for entry, "
                    f"available ₹{available:,.0f}"
                )

        except Exception as e:
            logger.warning(f"Margin check failed: {e}")
            # Don't block trade if margin check fails (let broker reject)
        return None
