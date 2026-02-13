"""
Advanced Trade Executor

Handles complex execution logic including:
- Trailing stops
- Partial profit booking
- Gap handling
- Dynamic stop adjustments
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
from loguru import logger


class ExecutionAction(Enum):
    """Types of execution actions."""
    HOLD = "hold"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    UPDATE_STOP = "update_stop"
    SKIP_ENTRY = "skip_entry"
    GAP_ADJUST = "gap_adjust"


@dataclass
class ExecutionConfig:
    """Configuration for trade execution rules."""
    # Trailing stop config
    use_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    trailing_activation_r: float = 1.0  # Activate trailing after 1R profit

    # Partial profit booking
    use_partial_exits: bool = True
    partial_exits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.25),   # At 1R, book 25%
        (2.0, 0.50),   # At target (2R), book 50%
        # Remaining 25% trails
    ])

    # Gap handling
    skip_if_gap_exceeds_atr_multiple: float = 2.0
    adjust_sl_on_gap_through: bool = True
    gap_sl_buffer_pct: float = 0.5  # 0.5% buffer below gap

    # Re-entry rules
    allow_reentry_after_sl: bool = True
    reentry_cooldown_bars: int = 3
    reentry_requires_new_signal: bool = True
    reentry_position_scale: float = 0.5  # 50% size on reentry

    # Time-based rules
    max_holding_days_intraday: int = 1
    max_holding_days_swing: int = 5
    max_holding_days_positional: int = 20


@dataclass
class PartialPosition:
    """Tracks partial position state."""
    trade_id: str
    symbol: str
    original_quantity: float
    current_quantity: float
    average_entry: float
    original_stop: float
    original_target: float
    current_stop: float  # May be trailing
    atr_at_entry: float

    # Partial exits completed
    partial_exits: List[Tuple[date, float, float]] = field(default_factory=list)
    # (date, price, quantity)

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price_seen: float = 0.0
    bars_held: int = 0
    trailing_activated: bool = False

    @property
    def is_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.current_quantity <= 0

    @property
    def remaining_fraction(self) -> float:
        """Fraction of original position remaining."""
        if self.original_quantity == 0:
            return 0.0
        return self.current_quantity / self.original_quantity

    def get_risk_r(self, current_price: float) -> float:
        """Get current R (risk units) from entry."""
        risk = abs(self.average_entry - self.original_stop)
        if risk == 0:
            return 0.0
        return (current_price - self.average_entry) / risk


@dataclass
class ExecutionEvent:
    """Records an execution event."""
    timestamp: datetime
    action: ExecutionAction
    price: float
    quantity: float = 0.0
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class AdvancedTradeExecutor:
    """
    Handles complex execution logic including trailing, partials, gaps.

    Usage:
        executor = AdvancedTradeExecutor(config)
        position = PartialPosition(...)

        for bar in price_bars:
            position, events = executor.execute_day(
                position, bar, atr
            )
            for event in events:
                process_event(event)

            if position.is_closed:
                break
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def execute_day(
        self,
        position: PartialPosition,
        bar: Dict[str, float],  # OHLC: open, high, low, close
        atr: float,
        bar_date: date,
        is_long: bool = True
    ) -> Tuple[PartialPosition, List[ExecutionEvent]]:
        """
        Process a single day's price action for a position.

        Args:
            position: Current position state
            bar: OHLC dict with 'open', 'high', 'low', 'close'
            atr: Current ATR value
            bar_date: Date of this bar
            is_long: True for long position, False for short

        Returns:
            Updated position and list of execution events
        """
        events = []
        position.bars_held += 1

        open_price = bar['open']
        high_price = bar['high']
        low_price = bar['low']
        close_price = bar['close']

        # Update highest price seen (for trailing)
        if is_long:
            position.highest_price_seen = max(position.highest_price_seen, high_price)
        else:
            position.highest_price_seen = min(position.highest_price_seen, low_price)

        # 1. Check gap handling (on open)
        gap_event = self._handle_gap(position, open_price, atr, is_long)
        if gap_event:
            events.append(gap_event)
            if gap_event.action == ExecutionAction.SKIP_ENTRY:
                return position, events

        # 2. Check stop loss (check low/high before anything else)
        stop_event = self._check_stop_loss(
            position, bar, bar_date, is_long
        )
        if stop_event:
            events.append(stop_event)
            return position, events

        # 3. Check partial profit targets
        partial_events = self._check_partial_exits(
            position, bar, bar_date, is_long
        )
        events.extend(partial_events)

        # 4. Update trailing stop if active
        if self.config.use_trailing_stop:
            trail_event = self._update_trailing_stop(
                position, bar, atr, is_long
            )
            if trail_event:
                events.append(trail_event)

        # 5. Update unrealized PnL
        if is_long:
            position.unrealized_pnl = (close_price - position.average_entry) * position.current_quantity
        else:
            position.unrealized_pnl = (position.average_entry - close_price) * position.current_quantity

        return position, events

    def _handle_gap(
        self,
        position: PartialPosition,
        open_price: float,
        atr: float,
        is_long: bool
    ) -> Optional[ExecutionEvent]:
        """Handle gap scenarios."""
        if position.bars_held > 1:
            # Gap handling only relevant for entry day
            return None

        prev_close = position.average_entry  # Approximate
        gap = abs(open_price - prev_close)
        gap_atr = gap / atr if atr > 0 else 0

        # Check if gap is too large
        if gap_atr > self.config.skip_if_gap_exceeds_atr_multiple:
            return ExecutionEvent(
                timestamp=datetime.now(),
                action=ExecutionAction.SKIP_ENTRY,
                price=open_price,
                reason=f"Gap too large: {gap_atr:.1f}x ATR (max {self.config.skip_if_gap_exceeds_atr_multiple}x)"
            )

        # Check if gap through stop
        if self.config.adjust_sl_on_gap_through:
            if is_long and open_price < position.current_stop:
                # Gap down through stop
                new_stop = open_price * (1 - self.config.gap_sl_buffer_pct / 100)
                old_stop = position.current_stop
                position.current_stop = new_stop
                return ExecutionEvent(
                    timestamp=datetime.now(),
                    action=ExecutionAction.GAP_ADJUST,
                    price=open_price,
                    reason=f"Gap through stop, adjusted {old_stop:.2f} -> {new_stop:.2f}",
                    details={'old_stop': old_stop, 'new_stop': new_stop}
                )
            elif not is_long and open_price > position.current_stop:
                # Gap up through stop (short position)
                new_stop = open_price * (1 + self.config.gap_sl_buffer_pct / 100)
                old_stop = position.current_stop
                position.current_stop = new_stop
                return ExecutionEvent(
                    timestamp=datetime.now(),
                    action=ExecutionAction.GAP_ADJUST,
                    price=open_price,
                    reason=f"Gap through stop, adjusted {old_stop:.2f} -> {new_stop:.2f}",
                    details={'old_stop': old_stop, 'new_stop': new_stop}
                )

        return None

    def _check_stop_loss(
        self,
        position: PartialPosition,
        bar: Dict[str, float],
        bar_date: date,
        is_long: bool
    ) -> Optional[ExecutionEvent]:
        """Check if stop loss was hit."""
        low_price = bar['low']
        high_price = bar['high']

        # Check stop based on position direction
        stop_hit = False
        exit_price = position.current_stop

        if is_long and low_price <= position.current_stop:
            stop_hit = True
            # Exit at stop (or worse if gapped through)
            exit_price = min(position.current_stop, bar['open'])
        elif not is_long and high_price >= position.current_stop:
            stop_hit = True
            exit_price = max(position.current_stop, bar['open'])

        if stop_hit:
            # Calculate P&L
            if is_long:
                pnl = (exit_price - position.average_entry) * position.current_quantity
            else:
                pnl = (position.average_entry - exit_price) * position.current_quantity

            position.realized_pnl += pnl
            quantity_exited = position.current_quantity
            position.current_quantity = 0
            position.unrealized_pnl = 0

            return ExecutionEvent(
                timestamp=datetime.now(),
                action=ExecutionAction.FULL_EXIT,
                price=exit_price,
                quantity=quantity_exited,
                reason=f"Stop loss hit at {exit_price:.2f}",
                details={
                    'exit_type': 'stop_loss',
                    'pnl': pnl,
                    'bars_held': position.bars_held
                }
            )

        return None

    def _check_partial_exits(
        self,
        position: PartialPosition,
        bar: Dict[str, float],
        bar_date: date,
        is_long: bool
    ) -> List[ExecutionEvent]:
        """Check and execute partial profit targets."""
        events = []

        if not self.config.use_partial_exits:
            return events

        high_price = bar['high']
        low_price = bar['low']

        current_r = position.get_risk_r(high_price if is_long else low_price)

        for r_target, exit_fraction in self.config.partial_exits:
            # Check if this target was already hit
            already_hit = any(
                e[1] >= self._calculate_r_price(position, r_target, is_long)
                for e in position.partial_exits
            )
            if already_hit:
                continue

            target_price = self._calculate_r_price(position, r_target, is_long)

            # Check if target hit
            target_hit = False
            if is_long and high_price >= target_price:
                target_hit = True
            elif not is_long and low_price <= target_price:
                target_hit = True

            if target_hit:
                # Calculate quantity to exit
                quantity_to_exit = position.original_quantity * exit_fraction

                # Don't exit more than we have
                quantity_to_exit = min(quantity_to_exit, position.current_quantity)

                if quantity_to_exit <= 0:
                    continue

                # Calculate P&L for this partial
                if is_long:
                    pnl = (target_price - position.average_entry) * quantity_to_exit
                else:
                    pnl = (position.average_entry - target_price) * quantity_to_exit

                # Update position
                position.current_quantity -= quantity_to_exit
                position.realized_pnl += pnl
                position.partial_exits.append((bar_date, target_price, quantity_to_exit))

                events.append(ExecutionEvent(
                    timestamp=datetime.now(),
                    action=ExecutionAction.PARTIAL_EXIT,
                    price=target_price,
                    quantity=quantity_to_exit,
                    reason=f"Partial exit at {r_target}R target ({exit_fraction:.0%})",
                    details={
                        'r_target': r_target,
                        'exit_fraction': exit_fraction,
                        'pnl': pnl,
                        'remaining_quantity': position.current_quantity
                    }
                ))

        return events

    def _update_trailing_stop(
        self,
        position: PartialPosition,
        bar: Dict[str, float],
        atr: float,
        is_long: bool
    ) -> Optional[ExecutionEvent]:
        """Update trailing stop if conditions are met."""
        # Check if trailing should activate
        current_r = position.get_risk_r(bar['high'] if is_long else bar['low'])

        if not position.trailing_activated:
            if current_r >= self.config.trailing_activation_r:
                position.trailing_activated = True
                logger.debug(f"Trailing stop activated at {current_r:.1f}R")

        if not position.trailing_activated:
            return None

        # Calculate new trailing stop
        trail_distance = atr * self.config.trailing_stop_atr_multiplier

        if is_long:
            new_stop = position.highest_price_seen - trail_distance
            # Only move stop up, never down
            if new_stop > position.current_stop:
                old_stop = position.current_stop
                position.current_stop = new_stop
                return ExecutionEvent(
                    timestamp=datetime.now(),
                    action=ExecutionAction.UPDATE_STOP,
                    price=new_stop,
                    reason=f"Trailing stop raised: {old_stop:.2f} -> {new_stop:.2f}",
                    details={
                        'old_stop': old_stop,
                        'new_stop': new_stop,
                        'highest_seen': position.highest_price_seen
                    }
                )
        else:
            new_stop = position.highest_price_seen + trail_distance
            # Only move stop down for shorts
            if new_stop < position.current_stop:
                old_stop = position.current_stop
                position.current_stop = new_stop
                return ExecutionEvent(
                    timestamp=datetime.now(),
                    action=ExecutionAction.UPDATE_STOP,
                    price=new_stop,
                    reason=f"Trailing stop lowered: {old_stop:.2f} -> {new_stop:.2f}",
                    details={
                        'old_stop': old_stop,
                        'new_stop': new_stop,
                        'lowest_seen': position.highest_price_seen
                    }
                )

        return None

    def _calculate_r_price(
        self,
        position: PartialPosition,
        r_multiple: float,
        is_long: bool
    ) -> float:
        """Calculate price at a given R multiple."""
        risk = abs(position.average_entry - position.original_stop)
        if is_long:
            return position.average_entry + (risk * r_multiple)
        else:
            return position.average_entry - (risk * r_multiple)

    def check_time_exit(
        self,
        position: PartialPosition,
        trade_type: str,
        bar: Dict[str, float],
        bar_date: date,
        is_long: bool
    ) -> Optional[ExecutionEvent]:
        """Check for time-based exit."""
        max_days = {
            'INTRADAY': self.config.max_holding_days_intraday,
            'SWING': self.config.max_holding_days_swing,
            'POSITIONAL': self.config.max_holding_days_positional,
        }.get(trade_type.upper(), 5)

        if position.bars_held >= max_days:
            exit_price = bar['close']

            if is_long:
                pnl = (exit_price - position.average_entry) * position.current_quantity
            else:
                pnl = (position.average_entry - exit_price) * position.current_quantity

            position.realized_pnl += pnl
            quantity_exited = position.current_quantity
            position.current_quantity = 0

            return ExecutionEvent(
                timestamp=datetime.now(),
                action=ExecutionAction.FULL_EXIT,
                price=exit_price,
                quantity=quantity_exited,
                reason=f"Time exit after {position.bars_held} bars (max {max_days})",
                details={
                    'exit_type': 'time_based',
                    'pnl': pnl,
                    'bars_held': position.bars_held
                }
            )

        return None
