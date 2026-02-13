"""
Re-entry Manager

Manages re-entry logic after stop losses:
- Cooldown periods
- Signal confirmation requirements
- Position scaling on re-entry
- Maximum re-entry limits
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger


@dataclass
class ReentryRule:
    """Rules for re-entering after stop loss."""
    cooldown_bars: int = 3  # Bars to wait after stop
    requires_new_signal: bool = True  # Require fresh signal
    max_reentries_per_month: int = 2  # Max re-entries per symbol per month
    scale_down_on_reentry: float = 0.5  # 50% position size on first reentry
    scale_down_increment: float = 0.25  # Additional scaling per reentry
    min_scale: float = 0.25  # Minimum position scale


@dataclass
class StopLossEvent:
    """Record of a stop loss event."""
    symbol: str
    stop_date: date
    entry_price: float
    stop_price: float
    signal_type: str  # 'BUY' or 'SELL'
    reason: str = ""


class ReentryManager:
    """
    Manages re-entry logic after stop losses.

    Tracks stop loss history and determines when re-entry is allowed.

    Usage:
        manager = ReentryManager(ReentryRule())

        # After a stop loss
        manager.register_stop_loss(symbol, date, price, ...)

        # Before entering a new trade
        can_enter, scale = manager.can_reenter(
            symbol, current_date, has_new_signal
        )
        if can_enter:
            position_size *= scale
    """

    def __init__(self, rule: Optional[ReentryRule] = None):
        self.rule = rule or ReentryRule()
        self.stop_history: Dict[str, List[StopLossEvent]] = {}
        self._last_cleanup: date = date.today()

    def register_stop_loss(
        self,
        symbol: str,
        stop_date: date,
        entry_price: float,
        stop_price: float,
        signal_type: str = 'BUY',
        reason: str = ""
    ) -> None:
        """
        Record a stop loss event.

        Args:
            symbol: Stock symbol
            stop_date: Date of stop loss
            entry_price: Original entry price
            stop_price: Price at which stop was hit
            signal_type: 'BUY' or 'SELL'
            reason: Optional reason for stop
        """
        if symbol not in self.stop_history:
            self.stop_history[symbol] = []

        event = StopLossEvent(
            symbol=symbol,
            stop_date=stop_date,
            entry_price=entry_price,
            stop_price=stop_price,
            signal_type=signal_type,
            reason=reason
        )

        self.stop_history[symbol].append(event)
        logger.debug(f"Registered stop loss for {symbol} on {stop_date}")

        # Cleanup old entries periodically
        self._cleanup_old_entries()

    def can_reenter(
        self,
        symbol: str,
        current_date: date,
        has_new_signal: bool,
        current_bar_index: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Check if re-entry is allowed for a symbol.

        Args:
            symbol: Stock symbol
            current_date: Current date
            has_new_signal: Whether there's a fresh prediction signal
            current_bar_index: Optional bar index since last stop

        Returns:
            Tuple of (can_reenter: bool, position_scale: float)
        """
        # No history = can enter at full size
        if symbol not in self.stop_history:
            return True, 1.0

        recent_stops = self.stop_history[symbol]
        if not recent_stops:
            return True, 1.0

        last_stop = recent_stops[-1]

        # Check cooldown period
        days_since_stop = (current_date - last_stop.stop_date).days
        if days_since_stop < self.rule.cooldown_bars:
            return False, 0.0

        # Check new signal requirement
        if self.rule.requires_new_signal and not has_new_signal:
            return False, 0.0

        # Check max re-entries per month
        month_start = date(current_date.year, current_date.month, 1)
        monthly_stops = sum(
            1 for s in recent_stops
            if s.stop_date >= month_start
        )
        if monthly_stops >= self.rule.max_reentries_per_month:
            logger.debug(
                f"{symbol}: Max monthly re-entries ({self.rule.max_reentries_per_month}) reached"
            )
            return False, 0.0

        # Calculate position scale
        # Scale down based on number of recent stops
        recent_count = len([
            s for s in recent_stops
            if (current_date - s.stop_date).days < 30
        ])

        if recent_count == 0:
            scale = 1.0
        elif recent_count == 1:
            scale = self.rule.scale_down_on_reentry
        else:
            scale = max(
                self.rule.min_scale,
                self.rule.scale_down_on_reentry - (
                    self.rule.scale_down_increment * (recent_count - 1)
                )
            )

        return True, scale

    def get_recent_stops(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> List[StopLossEvent]:
        """Get recent stop losses for a symbol."""
        if symbol not in self.stop_history:
            return []

        cutoff = date.today() - timedelta(days=lookback_days)
        return [
            s for s in self.stop_history[symbol]
            if s.stop_date >= cutoff
        ]

    def get_stop_statistics(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 90
    ) -> Dict:
        """
        Get statistics about stop losses.

        Args:
            symbol: Optional symbol to filter (None = all)
            lookback_days: Days to look back

        Returns:
            Dict with stop statistics
        """
        cutoff = date.today() - timedelta(days=lookback_days)

        if symbol:
            stops = [
                s for s in self.stop_history.get(symbol, [])
                if s.stop_date >= cutoff
            ]
        else:
            stops = [
                s
                for symbol_stops in self.stop_history.values()
                for s in symbol_stops
                if s.stop_date >= cutoff
            ]

        if not stops:
            return {
                'total_stops': 0,
                'avg_loss_pct': 0,
                'symbols_affected': 0,
                'by_symbol': {}
            }

        # Calculate average loss
        losses = []
        for s in stops:
            if s.signal_type == 'BUY':
                loss_pct = (s.stop_price - s.entry_price) / s.entry_price * 100
            else:
                loss_pct = (s.entry_price - s.stop_price) / s.entry_price * 100
            losses.append(loss_pct)

        # By symbol breakdown
        by_symbol = {}
        for s in stops:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = 0
            by_symbol[s.symbol] += 1

        return {
            'total_stops': len(stops),
            'avg_loss_pct': sum(losses) / len(losses) if losses else 0,
            'symbols_affected': len(set(s.symbol for s in stops)),
            'by_symbol': by_symbol
        }

    def _cleanup_old_entries(self, max_age_days: int = 90) -> None:
        """Remove stop loss records older than max_age_days."""
        today = date.today()

        # Only cleanup once per day
        if self._last_cleanup == today:
            return

        cutoff = today - timedelta(days=max_age_days)

        for symbol in list(self.stop_history.keys()):
            self.stop_history[symbol] = [
                s for s in self.stop_history[symbol]
                if s.stop_date >= cutoff
            ]

            # Remove empty entries
            if not self.stop_history[symbol]:
                del self.stop_history[symbol]

        self._last_cleanup = today

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear stop loss history.

        Args:
            symbol: Symbol to clear (None = clear all)
        """
        if symbol:
            if symbol in self.stop_history:
                del self.stop_history[symbol]
        else:
            self.stop_history.clear()

    def should_avoid_symbol(
        self,
        symbol: str,
        current_date: date,
        max_monthly_stops: int = 3
    ) -> Tuple[bool, str]:
        """
        Check if a symbol should be avoided due to frequent stops.

        Args:
            symbol: Stock symbol
            current_date: Current date
            max_monthly_stops: Maximum stops before avoiding

        Returns:
            Tuple of (should_avoid, reason)
        """
        month_start = date(current_date.year, current_date.month, 1)

        if symbol not in self.stop_history:
            return False, ""

        monthly_stops = sum(
            1 for s in self.stop_history[symbol]
            if s.stop_date >= month_start
        )

        if monthly_stops >= max_monthly_stops:
            return True, f"Too many stops this month ({monthly_stops})"

        # Also check for rapid consecutive stops
        recent = sorted(
            [s for s in self.stop_history[symbol]
             if (current_date - s.stop_date).days < 14],
            key=lambda x: x.stop_date,
            reverse=True
        )

        if len(recent) >= 2:
            # Two stops within 2 weeks
            return True, f"Multiple recent stops ({len(recent)} in 14 days)"

        return False, ""
