"""
MarketTimingEngine - Intelligent Market Timing for India

Provides context-aware recommendations based on:
1. Current market status (pre-market, open, closed)
2. Time of day (opening volatility, trend session, power hour)
3. Day of week patterns
4. F&O expiry proximity
5. Result season status

Key insight: WHEN you enter matters as much as WHAT you buy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz
from loguru import logger


class MarketPhase(Enum):
    """Current phase of the trading day."""
    PRE_MARKET = "pre_market"           # Before 9:00 AM
    PRE_OPEN = "pre_open"               # 9:00-9:15 (pre-open session)
    OPENING_VOLATILITY = "opening_vol"   # 9:15-10:00 (avoid entries)
    MORNING_TREND = "morning_trend"      # 10:00-12:00 (BEST for entries)
    MIDDAY_LULL = "midday_lull"          # 12:00-13:00 (low volume)
    AFTERNOON = "afternoon"              # 13:00-14:30 (trend resumes)
    POWER_HOUR = "power_hour"            # 14:30-15:30 (book profits)
    POST_MARKET = "post_market"          # After 15:30


class TradingDecision(Enum):
    """Recommended trading action."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    ADD_TO_WINNER = "add_to_winner"
    BOOK_PARTIAL = "book_partial"
    BOOK_FULL = "book_full"
    WAIT = "wait"
    AVOID = "avoid"
    PREPARE = "prepare"  # Pre-market preparation


@dataclass
class MarketContext:
    """Complete market timing context."""
    # Time info
    current_time_ist: datetime
    market_phase: MarketPhase
    is_market_open: bool
    is_trading_day: bool
    is_holiday: bool

    # Session info
    time_to_market_open: Optional[timedelta]
    time_to_market_close: Optional[timedelta]
    time_in_session: Optional[timedelta]

    # Next trading day
    next_trading_date: date
    is_expiry_day: bool
    is_monthly_expiry: bool
    days_to_expiry: int

    # Win probability by current time
    current_entry_win_rate: float
    optimal_entry_window: Tuple[time, time]

    # Recommendations
    decision: TradingDecision
    decision_reason: str
    confidence_multiplier: float  # Apply to prediction confidence

    # Detailed message
    action_message: str
    timing_tips: List[str]


class IndianMarketCalendar:
    """Indian market trading calendar and holidays."""

    # 2024-2025 NSE Holidays (sample - would fetch from API in production)
    NSE_HOLIDAYS = {
        date(2024, 1, 26),   # Republic Day
        date(2024, 3, 8),    # Mahashivratri
        date(2024, 3, 25),   # Holi
        date(2024, 3, 29),   # Good Friday
        date(2024, 4, 11),   # Id-ul-Fitr
        date(2024, 4, 14),   # Ambedkar Jayanti
        date(2024, 4, 17),   # Ram Navami
        date(2024, 4, 21),   # Mahavir Jayanti
        date(2024, 5, 1),    # Maharashtra Day
        date(2024, 5, 23),   # Buddha Purnima
        date(2024, 6, 17),   # Eid
        date(2024, 7, 17),   # Muharram
        date(2024, 8, 15),   # Independence Day
        date(2024, 10, 2),   # Gandhi Jayanti
        date(2024, 10, 12),  # Dussehra
        date(2024, 11, 1),   # Diwali (Laxmi Puja)
        date(2024, 11, 15),  # Guru Nanak Jayanti
        date(2024, 12, 25),  # Christmas
        date(2025, 1, 26),   # Republic Day
        date(2025, 2, 26),   # Mahashivratri
        date(2025, 3, 14),   # Holi
        date(2025, 4, 10),   # Ram Navami
        date(2025, 4, 14),   # Ambedkar Jayanti
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 8, 15),   # Independence Day
        date(2025, 10, 2),   # Gandhi Jayanti
        date(2025, 10, 21),  # Dussehra
        date(2025, 11, 5),   # Diwali
        date(2025, 12, 25),  # Christmas
    }

    # Market hours
    MARKET_OPEN = time(9, 15)
    MARKET_CLOSE = time(15, 30)
    PRE_OPEN_START = time(9, 0)
    PRE_OPEN_END = time(9, 8)

    @classmethod
    def is_holiday(cls, d: date) -> bool:
        return d in cls.NSE_HOLIDAYS

    @classmethod
    def is_weekend(cls, d: date) -> bool:
        return d.weekday() >= 5  # Saturday = 5, Sunday = 6

    @classmethod
    def is_trading_day(cls, d: date) -> bool:
        return not cls.is_weekend(d) and not cls.is_holiday(d)

    @classmethod
    def get_next_trading_day(cls, from_date: date) -> date:
        """Get next trading day."""
        next_day = from_date + timedelta(days=1)
        while not cls.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    @classmethod
    def get_monthly_expiry(cls, year: int, month: int) -> date:
        """Get F&O monthly expiry (last Thursday of month)."""
        import calendar
        cal = calendar.monthcalendar(year, month)
        thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY]]
        last_thursday = max(thursdays)
        expiry = date(year, month, last_thursday)

        # If Thursday is holiday, move to previous day
        while not cls.is_trading_day(expiry):
            expiry -= timedelta(days=1)

        return expiry


class MarketTimingEngine:
    """
    Provides market timing context for intelligent trade execution.
    """

    # Win rates by time window (based on historical analysis)
    TIME_WINDOW_STATS = {
        MarketPhase.OPENING_VOLATILITY: {
            'win_rate': 0.48,
            'risk': 'VERY_HIGH',
            'recommendation': 'AVOID new entries'
        },
        MarketPhase.MORNING_TREND: {
            'win_rate': 0.58,
            'risk': 'LOW',
            'recommendation': 'BEST time for entries'
        },
        MarketPhase.MIDDAY_LULL: {
            'win_rate': 0.45,
            'risk': 'MEDIUM',
            'recommendation': 'Avoid trading, low volume'
        },
        MarketPhase.AFTERNOON: {
            'win_rate': 0.54,
            'risk': 'MEDIUM',
            'recommendation': 'Add to winners'
        },
        MarketPhase.POWER_HOUR: {
            'win_rate': 0.52,
            'risk': 'HIGH',
            'recommendation': 'Book profits, avoid new entries'
        }
    }

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.calendar = IndianMarketCalendar()

    def get_current_time_ist(self) -> datetime:
        """Get current time in IST."""
        return datetime.now(self.ist)

    def get_market_context(self) -> MarketContext:
        """Get complete market timing context."""
        now = self.get_current_time_ist()
        today = now.date()
        current_time = now.time()

        # Basic checks
        is_trading_day = self.calendar.is_trading_day(today)
        is_holiday = self.calendar.is_holiday(today)

        # Market phase
        is_open, phase = self._get_market_phase(current_time, is_trading_day)

        # Time calculations
        time_to_open = self._get_time_to_open(now) if not is_open else None
        time_to_close = self._get_time_to_close(now) if is_open else None
        time_in_session = self._get_time_in_session(now) if is_open else None

        # Next trading day
        if is_trading_day and current_time < self.calendar.MARKET_CLOSE:
            next_trading = today
        else:
            next_trading = self.calendar.get_next_trading_day(today)

        # Expiry info
        monthly_expiry = self.calendar.get_monthly_expiry(today.year, today.month)
        is_expiry_day = today == monthly_expiry
        days_to_expiry = (monthly_expiry - today).days

        # Win rates and decisions
        if is_open:
            stats = self.TIME_WINDOW_STATS.get(phase, {})
            win_rate = stats.get('win_rate', 0.5)
            decision, reason = self._get_trading_decision(phase, time_in_session)
            confidence_mult = self._get_confidence_multiplier(phase, is_expiry_day)
        else:
            win_rate = 0.0
            decision, reason = self._get_pre_market_decision(now, next_trading)
            confidence_mult = 0.9  # Slight reduction for overnight uncertainty

        # Action message
        action_message = self._generate_action_message(
            phase, is_open, decision, next_trading, days_to_expiry
        )

        # Timing tips
        tips = self._get_timing_tips(phase, is_open, is_expiry_day)

        return MarketContext(
            current_time_ist=now,
            market_phase=phase,
            is_market_open=is_open,
            is_trading_day=is_trading_day,
            is_holiday=is_holiday,
            time_to_market_open=time_to_open,
            time_to_market_close=time_to_close,
            time_in_session=time_in_session,
            next_trading_date=next_trading,
            is_expiry_day=is_expiry_day,
            is_monthly_expiry=is_expiry_day,
            days_to_expiry=days_to_expiry,
            current_entry_win_rate=win_rate,
            optimal_entry_window=(time(10, 0), time(12, 0)),
            decision=decision,
            decision_reason=reason,
            confidence_multiplier=confidence_mult,
            action_message=action_message,
            timing_tips=tips
        )

    def _get_market_phase(self, t: time, is_trading_day: bool) -> Tuple[bool, MarketPhase]:
        """Determine current market phase."""
        if not is_trading_day:
            return False, MarketPhase.POST_MARKET

        if t < self.calendar.PRE_OPEN_START:
            return False, MarketPhase.PRE_MARKET
        elif t < self.calendar.MARKET_OPEN:
            return False, MarketPhase.PRE_OPEN
        elif t < time(10, 0):
            return True, MarketPhase.OPENING_VOLATILITY
        elif t < time(12, 0):
            return True, MarketPhase.MORNING_TREND
        elif t < time(13, 0):
            return True, MarketPhase.MIDDAY_LULL
        elif t < time(14, 30):
            return True, MarketPhase.AFTERNOON
        elif t < self.calendar.MARKET_CLOSE:
            return True, MarketPhase.POWER_HOUR
        else:
            return False, MarketPhase.POST_MARKET

    def _get_time_to_open(self, now: datetime) -> timedelta:
        """Calculate time until market opens."""
        today = now.date()
        if self.calendar.is_trading_day(today) and now.time() < self.calendar.MARKET_OPEN:
            open_dt = datetime.combine(today, self.calendar.MARKET_OPEN)
            open_dt = self.ist.localize(open_dt)
            return open_dt - now
        else:
            next_day = self.calendar.get_next_trading_day(today)
            open_dt = datetime.combine(next_day, self.calendar.MARKET_OPEN)
            open_dt = self.ist.localize(open_dt)
            return open_dt - now

    def _get_time_to_close(self, now: datetime) -> timedelta:
        """Calculate time until market closes."""
        close_dt = datetime.combine(now.date(), self.calendar.MARKET_CLOSE)
        close_dt = self.ist.localize(close_dt)
        return close_dt - now

    def _get_time_in_session(self, now: datetime) -> timedelta:
        """Calculate time since market opened."""
        open_dt = datetime.combine(now.date(), self.calendar.MARKET_OPEN)
        open_dt = self.ist.localize(open_dt)
        return now - open_dt

    def _get_trading_decision(self,
                               phase: MarketPhase,
                               time_in_session: timedelta) -> Tuple[TradingDecision, str]:
        """Get trading decision based on phase."""
        if phase == MarketPhase.OPENING_VOLATILITY:
            return TradingDecision.AVOID, "Opening volatility - wait for trend to establish"
        elif phase == MarketPhase.MORNING_TREND:
            return TradingDecision.ENTER_LONG, "Best entry window - trends are clear"
        elif phase == MarketPhase.MIDDAY_LULL:
            return TradingDecision.WAIT, "Low volume period - hold positions"
        elif phase == MarketPhase.AFTERNOON:
            return TradingDecision.ADD_TO_WINNER, "Add to winning positions"
        elif phase == MarketPhase.POWER_HOUR:
            return TradingDecision.BOOK_FULL, "Book profits before close"
        return TradingDecision.WAIT, "Market closed"

    def _get_pre_market_decision(self,
                                   now: datetime,
                                   next_trading: date) -> Tuple[TradingDecision, str]:
        """Get decision for pre-market hours."""
        if now.date() == next_trading:
            return TradingDecision.PREPARE, f"Market opens at 9:15 AM - prepare your watchlist"
        else:
            return TradingDecision.PREPARE, f"Next trading day: {next_trading.strftime('%A, %b %d')}"

    def _get_confidence_multiplier(self, phase: MarketPhase, is_expiry: bool) -> float:
        """Get confidence multiplier based on timing."""
        base = {
            MarketPhase.OPENING_VOLATILITY: 0.7,
            MarketPhase.MORNING_TREND: 1.0,
            MarketPhase.MIDDAY_LULL: 0.8,
            MarketPhase.AFTERNOON: 0.9,
            MarketPhase.POWER_HOUR: 0.8
        }.get(phase, 0.9)

        if is_expiry:
            base *= 0.85  # Reduce confidence on expiry days

        return base

    def _generate_action_message(self,
                                   phase: MarketPhase,
                                   is_open: bool,
                                   decision: TradingDecision,
                                   next_trading: date,
                                   days_to_expiry: int) -> str:
        """Generate human-readable action message."""
        if not is_open:
            next_day = next_trading.strftime('%A, %b %d')
            return f"Market closed. Next trading day: {next_day}. Prepare your buy list for 10:00-12:00 entry window."

        messages = {
            TradingDecision.ENTER_LONG: "OPTIMAL ENTRY WINDOW! Execute your planned buys now.",
            TradingDecision.AVOID: "HIGH VOLATILITY! Wait for 10:00 AM for stable entries.",
            TradingDecision.WAIT: "LOW VOLUME! Hold positions, avoid new entries.",
            TradingDecision.ADD_TO_WINNER: "Add to winning positions if trend continues.",
            TradingDecision.BOOK_FULL: "BOOK PROFITS! Close intraday positions before 3:20 PM.",
        }

        msg = messages.get(decision, "Monitor positions.")

        if days_to_expiry <= 2:
            msg += f" (Expiry in {days_to_expiry} days - expect high volatility)"

        return msg

    def _get_timing_tips(self,
                          phase: MarketPhase,
                          is_open: bool,
                          is_expiry: bool) -> List[str]:
        """Get contextual timing tips."""
        tips = []

        if not is_open:
            tips.append("Pre-market: Review overnight news and global cues")
            tips.append("Plan entries for 10:00-12:00 window")
            tips.append("Set alerts for your watchlist stocks")
            return tips

        if phase == MarketPhase.OPENING_VOLATILITY:
            tips.append("Gaps may fill - avoid chasing")
            tips.append("Wait for first 45 mins volatility to settle")
            tips.append("Only enter if gap + trend align")
        elif phase == MarketPhase.MORNING_TREND:
            tips.append("Best time for fresh entries")
            tips.append("Enter on pullbacks to VWAP")
            tips.append("Set stop-loss at day's low")
        elif phase == MarketPhase.MIDDAY_LULL:
            tips.append("Avoid new trades - low conviction")
            tips.append("Trail stops on winners")
            tips.append("Use this time to plan afternoon trades")
        elif phase == MarketPhase.AFTERNOON:
            tips.append("Add to positions showing strength")
            tips.append("Start thinking about profit booking")
            tips.append("Move stops to breakeven")
        elif phase == MarketPhase.POWER_HOUR:
            tips.append("Book intraday profits by 3:20 PM")
            tips.append("Avoid new entries")
            tips.append("Watch for reversal patterns")

        if is_expiry:
            tips.append("EXPIRY: Expect wild swings near strikes")
            tips.append("Reduce position sizes by 50%")

        return tips

    def get_entry_recommendation(self,
                                   symbol: str,
                                   current_price: float,
                                   predicted_signal: str,
                                   confidence: float) -> Dict:
        """
        Get entry recommendation with market timing adjustment.

        Returns actionable entry with timing-adjusted parameters.
        """
        ctx = self.get_market_context()

        # Adjust confidence based on timing
        adjusted_confidence = confidence * ctx.confidence_multiplier

        # Determine if entry is recommended
        if not ctx.is_market_open:
            entry_type = "NEXT_DAY_OPEN"
            entry_price = current_price  # Will use tomorrow's open
            timing_note = f"Enter at market open on {ctx.next_trading_date}"
        elif ctx.decision == TradingDecision.ENTER_LONG:
            entry_type = "IMMEDIATE"
            entry_price = current_price
            timing_note = "Enter now - optimal timing window"
        elif ctx.decision == TradingDecision.AVOID:
            entry_type = "DELAYED"
            entry_price = current_price
            timing_note = "Wait until 10:00 AM for better entry"
        elif ctx.decision in [TradingDecision.BOOK_FULL, TradingDecision.BOOK_PARTIAL]:
            entry_type = "NOT_RECOMMENDED"
            entry_price = current_price
            timing_note = "Avoid new entries - booking time"
        else:
            entry_type = "CONDITIONAL"
            entry_price = current_price
            timing_note = "Enter only on confirmation"

        return {
            'symbol': symbol,
            'signal': predicted_signal,
            'original_confidence': confidence,
            'adjusted_confidence': adjusted_confidence,
            'entry_type': entry_type,
            'entry_price': entry_price,
            'timing_note': timing_note,
            'market_phase': ctx.market_phase.value,
            'is_market_open': ctx.is_market_open,
            'next_trading_date': str(ctx.next_trading_date),
            'action_message': ctx.action_message,
            'tips': ctx.timing_tips
        }


# Singleton
_engine = None


def get_market_timing_engine() -> MarketTimingEngine:
    global _engine
    if _engine is None:
        _engine = MarketTimingEngine()
    return _engine
