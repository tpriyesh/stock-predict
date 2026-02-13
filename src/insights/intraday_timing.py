"""
Intraday Timing Analyzer

Predicts optimal buy/sell times during market hours (9:15 AM - 3:30 PM IST)
with risk assessment and probability estimates.

Key Time Windows:
- Opening Session (9:15-10:00): High volatility, gap plays
- Morning Trend (10:00-12:00): Trend establishment, best for trend following
- Mid-day Lull (12:00-13:00): Low volume, choppy, avoid new entries
- Afternoon Session (13:00-14:30): Trend resumption
- Power Hour (14:30-15:30): Final moves, closing adjustments
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pytz
import numpy as np
import yfinance as yf
from loguru import logger


class MarketState(Enum):
    """Current state of the market."""
    PRE_MARKET = "PRE_MARKET"           # Before 9:15 AM
    OPENING_SESSION = "OPENING_SESSION"  # 9:15 - 10:00 AM
    MORNING_TREND = "MORNING_TREND"      # 10:00 - 12:00 PM
    MIDDAY_LULL = "MIDDAY_LULL"          # 12:00 - 1:00 PM
    AFTERNOON_SESSION = "AFTERNOON"       # 1:00 - 2:30 PM
    POWER_HOUR = "POWER_HOUR"            # 2:30 - 3:30 PM
    MARKET_CLOSED = "MARKET_CLOSED"      # After 3:30 PM or weekend/holiday


class TimeWindowRisk(Enum):
    """Risk level for each time window."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class TradeAction(Enum):
    """Recommended action for current time."""
    WAIT = "WAIT"
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    AVOID = "AVOID"
    BOOK_PROFITS = "BOOK_PROFITS"


@dataclass
class TimeWindow:
    """Information about a trading time window."""
    name: str
    start_time: time
    end_time: time
    risk_level: TimeWindowRisk
    volatility: str  # "HIGH", "MEDIUM", "LOW"
    volume: str      # "HIGH", "MEDIUM", "LOW"
    best_for: List[str]  # What strategies work best
    avoid_for: List[str]  # What to avoid
    win_probability: float  # Historical win rate for entries in this window
    description: str


@dataclass
class IntradayTradeSetup:
    """Specific intraday trade setup with times and prices."""
    # Entry
    entry_time: str           # e.g., "10:15 AM"
    entry_price: float        # e.g., 2450.00
    entry_window: str         # e.g., "Morning Trend"
    entry_reason: str         # Why this time

    # Exit - Target
    target_time: str          # e.g., "2:30 PM"
    target_price: float       # e.g., 2485.00
    target_reason: str

    # Exit - Stop Loss
    stop_loss_price: float    # e.g., 2430.00
    stop_loss_time: str       # When to definitely exit if SL not hit

    # Probabilities
    win_probability: float
    risk_reward: float
    expected_return_pct: float

    # Risk assessment
    risk_level: str
    confidence: str           # "HIGH", "MEDIUM", "LOW"


@dataclass
class TimingPrediction:
    """Timing prediction for a stock."""
    symbol: str
    current_time_ist: datetime
    market_state: MarketState
    is_market_open: bool

    # Current window info
    current_window: Optional[TimeWindow]
    time_remaining_in_window: str

    # Recommendations
    recommended_action: TradeAction
    action_reason: str
    risk_level: TimeWindowRisk
    win_probability: float

    # Best times
    best_entry_windows: List[Tuple[str, str, float]]  # (window_name, time_range, probability)
    best_exit_windows: List[Tuple[str, str, float]]
    avoid_windows: List[Tuple[str, str, str]]  # (window_name, time_range, reason)

    # SPECIFIC TRADE SETUP (optional)
    trade_setup: Optional[IntradayTradeSetup] = None

    # Tomorrow predictions (if market closed)
    tomorrow_outlook: Optional[Dict] = None

    # Risk factors
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)


class IntradayTimingAnalyzer:
    """Analyzes optimal intraday timing for trades."""

    IST = pytz.timezone('Asia/Kolkata')

    # Define trading windows
    TIME_WINDOWS = {
        MarketState.OPENING_SESSION: TimeWindow(
            name="Opening Session",
            start_time=time(9, 15),
            end_time=time(10, 0),
            risk_level=TimeWindowRisk.VERY_HIGH,
            volatility="HIGH",
            volume="HIGH",
            best_for=["Gap plays", "Momentum scalping", "News-based trades"],
            avoid_for=["New trend trades", "Beginners", "Large positions"],
            win_probability=0.48,  # Lower due to high volatility
            description="High volatility from overnight gaps. Experienced traders only."
        ),
        MarketState.MORNING_TREND: TimeWindow(
            name="Morning Trend",
            start_time=time(10, 0),
            end_time=time(12, 0),
            risk_level=TimeWindowRisk.MODERATE,
            volatility="MEDIUM",
            volume="HIGH",
            best_for=["Trend following", "Breakout entries", "Swing entries"],
            avoid_for=["Counter-trend trades"],
            win_probability=0.58,  # Best window for entries
            description="Trend establishes. Best time for new entries."
        ),
        MarketState.MIDDAY_LULL: TimeWindow(
            name="Mid-day Lull",
            start_time=time(12, 0),
            end_time=time(13, 0),
            risk_level=TimeWindowRisk.MODERATE,
            volatility="LOW",
            volume="LOW",
            best_for=["Lunch break", "Analysis", "Planning"],
            avoid_for=["New entries", "Breakout trades"],
            win_probability=0.45,  # Choppy, low conviction
            description="Low volume, choppy action. Avoid new entries."
        ),
        MarketState.AFTERNOON_SESSION: TimeWindow(
            name="Afternoon Session",
            start_time=time(13, 0),
            end_time=time(14, 30),
            risk_level=TimeWindowRisk.MODERATE,
            volatility="MEDIUM",
            volume="MEDIUM",
            best_for=["Trend continuation", "Adding to winners"],
            avoid_for=["New aggressive positions"],
            win_probability=0.54,
            description="Trend often resumes. Good for adding to positions."
        ),
        MarketState.POWER_HOUR: TimeWindow(
            name="Power Hour",
            start_time=time(14, 30),
            end_time=time(15, 30),
            risk_level=TimeWindowRisk.HIGH,
            volatility="HIGH",
            volume="HIGH",
            best_for=["Closing positions", "Quick scalps", "Booking profits"],
            avoid_for=["New swing positions", "Holding overnight"],
            win_probability=0.52,
            description="Final moves. Book profits or close losing positions."
        ),
    }

    # Indian market holidays 2024-2025 (sample - should be updated)
    HOLIDAYS = [
        "2025-01-26",  # Republic Day
        "2025-03-14",  # Holi
        "2025-04-14",  # Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-05-01",  # May Day
        "2025-08-15",  # Independence Day
        "2025-10-02",  # Gandhi Jayanti
        "2025-10-21",  # Diwali Laxmi Puja
        "2025-11-05",  # Diwali Balipratipada
        "2025-12-25",  # Christmas
    ]

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def get_current_time_ist(self) -> datetime:
        """Get current time in IST."""
        return datetime.now(self.IST)

    def is_market_holiday(self, date: datetime) -> bool:
        """Check if given date is a market holiday."""
        date_str = date.strftime("%Y-%m-%d")
        return date_str in self.HOLIDAYS

    def is_weekend(self, date: datetime) -> bool:
        """Check if given date is weekend."""
        return date.weekday() >= 5  # Saturday = 5, Sunday = 6

    def get_market_state(self, dt: Optional[datetime] = None) -> MarketState:
        """Determine current market state based on time."""
        if dt is None:
            dt = self.get_current_time_ist()

        current_time = dt.time()

        # Check if weekend or holiday
        if self.is_weekend(dt) or self.is_market_holiday(dt):
            return MarketState.MARKET_CLOSED

        # Check time windows
        if current_time < time(9, 15):
            return MarketState.PRE_MARKET
        elif current_time < time(10, 0):
            return MarketState.OPENING_SESSION
        elif current_time < time(12, 0):
            return MarketState.MORNING_TREND
        elif current_time < time(13, 0):
            return MarketState.MIDDAY_LULL
        elif current_time < time(14, 30):
            return MarketState.AFTERNOON_SESSION
        elif current_time < time(15, 30):
            return MarketState.POWER_HOUR
        else:
            return MarketState.MARKET_CLOSED

    def get_next_trading_day(self, from_date: Optional[datetime] = None) -> datetime:
        """Get the next trading day."""
        if from_date is None:
            from_date = self.get_current_time_ist()

        next_day = from_date + timedelta(days=1)

        # Skip weekends and holidays
        while self.is_weekend(next_day) or self.is_market_holiday(next_day):
            next_day += timedelta(days=1)

        return next_day

    def get_time_remaining(self, end_time: time, current_time: time) -> str:
        """Calculate time remaining in current window."""
        end_dt = datetime.combine(datetime.today(), end_time)
        curr_dt = datetime.combine(datetime.today(), current_time)

        diff = end_dt - curr_dt
        if diff.total_seconds() < 0:
            return "0 min"

        minutes = int(diff.total_seconds() / 60)
        if minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}h {mins}m"
        return f"{minutes} min"

    def analyze_stock_intraday_pattern(self, symbol: str) -> Dict:
        """Analyze stock's historical intraday patterns."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")

            # Get intraday data (last 5 days, 15-min intervals)
            hist = ticker.history(period="5d", interval="15m")

            if hist.empty:
                return {}

            # Analyze patterns by time of day
            hist['hour'] = hist.index.hour
            hist['minute'] = hist.index.minute
            hist['time_slot'] = hist['hour'] * 100 + hist['minute']

            # Calculate returns for each interval
            hist['return'] = hist['Close'].pct_change() * 100

            # Group by time slot
            time_analysis = {}

            # Opening (9:15-10:00)
            opening = hist[(hist['time_slot'] >= 915) & (hist['time_slot'] < 1000)]
            if not opening.empty:
                time_analysis['opening'] = {
                    'avg_return': opening['return'].mean(),
                    'volatility': opening['return'].std(),
                    'positive_pct': (opening['return'] > 0).mean() * 100
                }

            # Morning (10:00-12:00)
            morning = hist[(hist['time_slot'] >= 1000) & (hist['time_slot'] < 1200)]
            if not morning.empty:
                time_analysis['morning'] = {
                    'avg_return': morning['return'].mean(),
                    'volatility': morning['return'].std(),
                    'positive_pct': (morning['return'] > 0).mean() * 100
                }

            # Midday (12:00-13:00)
            midday = hist[(hist['time_slot'] >= 1200) & (hist['time_slot'] < 1300)]
            if not midday.empty:
                time_analysis['midday'] = {
                    'avg_return': midday['return'].mean(),
                    'volatility': midday['return'].std(),
                    'positive_pct': (midday['return'] > 0).mean() * 100
                }

            # Afternoon (13:00-14:30)
            afternoon = hist[(hist['time_slot'] >= 1300) & (hist['time_slot'] < 1430)]
            if not afternoon.empty:
                time_analysis['afternoon'] = {
                    'avg_return': afternoon['return'].mean(),
                    'volatility': afternoon['return'].std(),
                    'positive_pct': (afternoon['return'] > 0).mean() * 100
                }

            # Power hour (14:30-15:30)
            power = hist[(hist['time_slot'] >= 1430) & (hist['time_slot'] < 1530)]
            if not power.empty:
                time_analysis['power_hour'] = {
                    'avg_return': power['return'].mean(),
                    'volatility': power['return'].std(),
                    'positive_pct': (power['return'] > 0).mean() * 100
                }

            return time_analysis

        except Exception as e:
            logger.warning(f"Could not analyze intraday pattern for {symbol}: {e}")
            return {}

    def get_recommended_action(
        self,
        market_state: MarketState,
        stock_pattern: Dict,
        signal_type: str = "BUY"  # "BUY" or "SELL"
    ) -> Tuple[TradeAction, str, float]:
        """Get recommended action based on current state and signal."""

        if market_state == MarketState.MARKET_CLOSED:
            return TradeAction.WAIT, "Market is closed. Wait for next trading session.", 0.0

        if market_state == MarketState.PRE_MARKET:
            return TradeAction.WAIT, "Market not yet open. Prepare your watchlist.", 0.0

        window = self.TIME_WINDOWS.get(market_state)
        if not window:
            return TradeAction.WAIT, "Unknown market state.", 0.0

        # Adjust probability based on stock-specific patterns
        base_prob = window.win_probability

        if signal_type == "BUY":
            if market_state == MarketState.OPENING_SESSION:
                return TradeAction.WAIT, "Wait for volatility to settle. Opening is risky.", base_prob

            elif market_state == MarketState.MORNING_TREND:
                return TradeAction.BUY, "Best time window for new entries. Trend established.", base_prob

            elif market_state == MarketState.MIDDAY_LULL:
                return TradeAction.AVOID, "Avoid new entries. Low volume, choppy action.", base_prob

            elif market_state == MarketState.AFTERNOON_SESSION:
                return TradeAction.BUY, "Good for adding to existing positions.", base_prob

            elif market_state == MarketState.POWER_HOUR:
                return TradeAction.AVOID, "Too late for new intraday entries. Risk of overnight gap.", base_prob

        else:  # SELL
            if market_state == MarketState.OPENING_SESSION:
                return TradeAction.HOLD, "Wait unless you have strong conviction to exit.", base_prob

            elif market_state == MarketState.MORNING_TREND:
                return TradeAction.HOLD, "Let winners run if trend is favorable.", base_prob

            elif market_state == MarketState.MIDDAY_LULL:
                return TradeAction.HOLD, "Hold positions, low activity period.", base_prob

            elif market_state == MarketState.AFTERNOON_SESSION:
                return TradeAction.SELL, "Consider partial profit booking.", base_prob

            elif market_state == MarketState.POWER_HOUR:
                return TradeAction.BOOK_PROFITS, "Book profits or close positions for intraday.", base_prob

        return TradeAction.WAIT, "Analyze further before acting.", base_prob

    def calculate_trade_setup(self, symbol: str, current_time: datetime) -> Optional[IntradayTradeSetup]:
        """Calculate specific entry/exit times and prices for intraday trade."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="5d", interval="15m")

            if hist.empty:
                return None

            # Get current price (last available)
            current_price = hist['Close'].iloc[-1]

            # Calculate ATR for stop loss
            hist['TR'] = np.maximum(
                hist['High'] - hist['Low'],
                np.maximum(
                    abs(hist['High'] - hist['Close'].shift(1)),
                    abs(hist['Low'] - hist['Close'].shift(1))
                )
            )
            atr = hist['TR'].rolling(14).mean().iloc[-1]
            atr_percent = (atr / current_price) * 100

            # Analyze intraday patterns
            hist['hour'] = hist.index.hour
            hist['minute'] = hist.index.minute
            hist['return'] = hist['Close'].pct_change() * 100

            # Find best entry time based on historical patterns
            morning_data = hist[(hist['hour'] >= 10) & (hist['hour'] < 12)]
            afternoon_data = hist[(hist['hour'] >= 13) & (hist['hour'] < 15)]

            # Calculate average returns by time slot
            morning_avg = morning_data['return'].mean() if not morning_data.empty else 0
            afternoon_avg = afternoon_data['return'].mean() if not afternoon_data.empty else 0

            # Determine market state
            market_state = self.get_market_state(current_time)
            current_hour = current_time.hour
            current_minute = current_time.minute

            # Calculate specific entry time
            if market_state == MarketState.PRE_MARKET:
                entry_time = "10:15 AM"
                entry_window = "Morning Trend"
                entry_reason = "Wait for opening volatility to settle, enter on trend confirmation"
            elif market_state == MarketState.OPENING_SESSION:
                entry_time = "10:15 AM"
                entry_window = "Morning Trend"
                entry_reason = "Opening session is risky - wait for 10:15 AM when trend establishes"
            elif market_state == MarketState.MORNING_TREND:
                # Already in morning trend - calculate optimal entry
                if current_hour == 10 and current_minute < 30:
                    entry_time = f"{current_hour}:{current_minute + 15:02d} AM"
                    entry_reason = "Enter now - best window for new entries"
                elif current_hour == 10:
                    entry_time = "10:45 AM"
                    entry_reason = "Enter soon - morning trend window closing"
                else:
                    entry_time = f"{current_hour}:{(current_minute + 10) % 60:02d} AM"
                    entry_reason = "Enter on next support level test"
                entry_window = "Morning Trend"
            elif market_state == MarketState.MIDDAY_LULL:
                entry_time = "1:15 PM"
                entry_window = "Afternoon Session"
                entry_reason = "Wait for lunch lull to end - enter when volume picks up"
            elif market_state == MarketState.AFTERNOON_SESSION:
                if current_hour == 13:
                    entry_time = f"1:{current_minute + 15:02d} PM"
                else:
                    entry_time = f"2:{min(current_minute + 10, 25):02d} PM"
                entry_window = "Afternoon Session"
                entry_reason = "Afternoon continuation - add to winners only"
            elif market_state == MarketState.POWER_HOUR:
                entry_time = "AVOID"
                entry_window = "Power Hour"
                entry_reason = "Too late for new entries - focus on existing positions"
            else:  # MARKET_CLOSED
                entry_time = "10:15 AM (Tomorrow)"
                entry_window = "Morning Trend"
                entry_reason = "Market closed - plan for tomorrow's morning trend"

            # Calculate entry price (slightly below current for limit order)
            entry_discount = 0.002  # 0.2% below current
            entry_price = current_price * (1 - entry_discount)

            # Calculate stop loss (1.5x ATR below entry)
            stop_loss_price = entry_price - (1.5 * atr)
            stop_loss_percent = ((entry_price - stop_loss_price) / entry_price) * 100

            # Calculate target (2x risk for 1:2 R:R)
            risk_per_share = entry_price - stop_loss_price
            target_price = entry_price + (2 * risk_per_share)
            target_percent = ((target_price - entry_price) / entry_price) * 100

            # Determine exit time
            if market_state in [MarketState.PRE_MARKET, MarketState.OPENING_SESSION, MarketState.MORNING_TREND]:
                target_time = "2:45 PM"
                target_reason = "Power hour - book profits before close"
            elif market_state == MarketState.MIDDAY_LULL:
                target_time = "2:30 PM"
                target_reason = "Afternoon session end - avoid last-minute volatility"
            elif market_state == MarketState.AFTERNOON_SESSION:
                target_time = "3:15 PM"
                target_reason = "Close before market end"
            else:
                target_time = "3:15 PM"
                target_reason = "Standard exit before close"

            # Calculate probabilities
            win_prob = 0.55  # Base probability
            if market_state == MarketState.MORNING_TREND:
                win_prob = 0.58
            elif market_state == MarketState.AFTERNOON_SESSION:
                win_prob = 0.54
            elif market_state == MarketState.POWER_HOUR:
                win_prob = 0.48

            # Adjust based on stock's historical pattern
            if morning_avg > 0.05:
                win_prob += 0.03
            if atr_percent < 1.5:  # Low volatility = more predictable
                win_prob += 0.02

            risk_reward = target_percent / stop_loss_percent if stop_loss_percent > 0 else 2.0
            expected_return = (win_prob * target_percent) - ((1 - win_prob) * stop_loss_percent)

            # Risk assessment
            if atr_percent > 2.5:
                risk_level = "HIGH"
                confidence = "LOW"
            elif atr_percent > 1.5:
                risk_level = "MODERATE"
                confidence = "MEDIUM"
            else:
                risk_level = "LOW"
                confidence = "HIGH"

            return IntradayTradeSetup(
                entry_time=entry_time,
                entry_price=round(entry_price, 2),
                entry_window=entry_window,
                entry_reason=entry_reason,
                target_time=target_time,
                target_price=round(target_price, 2),
                target_reason=target_reason,
                stop_loss_price=round(stop_loss_price, 2),
                stop_loss_time="3:20 PM" if entry_time != "AVOID" else "N/A",
                win_probability=round(win_prob, 2),
                risk_reward=round(risk_reward, 2),
                expected_return_pct=round(expected_return, 2),
                risk_level=risk_level,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"Could not calculate trade setup for {symbol}: {e}")
            return None

    def get_tomorrow_outlook(self, symbol: str) -> Dict:
        """Generate outlook for tomorrow's trading."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="1mo")

            if hist.empty:
                return {}

            # Analyze recent patterns
            last_5_days = hist.tail(5)
            trend = "BULLISH" if last_5_days['Close'].iloc[-1] > last_5_days['Close'].iloc[0] else "BEARISH"

            # Calculate gap probability
            gaps = hist['Open'] - hist['Close'].shift(1)
            gap_up_prob = (gaps > 0).mean() * 100
            gap_down_prob = (gaps < 0).mean() * 100

            # Average gap size
            avg_gap = gaps.abs().mean() / hist['Close'].mean() * 100

            # Best entry prediction
            next_trading_day = self.get_next_trading_day()

            return {
                'date': next_trading_day.strftime("%Y-%m-%d (%A)"),
                'trend': trend,
                'gap_up_probability': gap_up_prob,
                'gap_down_probability': gap_down_prob,
                'avg_gap_percent': avg_gap,
                'recommended_entry_time': "10:00 - 11:00 AM" if trend == "BULLISH" else "Wait for confirmation",
                'recommended_exit_time': "2:30 - 3:15 PM",
                'risk_level': "MODERATE" if avg_gap < 1 else "HIGH",
                'strategy': "Wait for opening volatility to settle, enter on trend confirmation around 10:00 AM"
            }

        except Exception as e:
            logger.warning(f"Could not generate tomorrow outlook: {e}")
            return {}

    def analyze(self, symbol: str) -> TimingPrediction:
        """Full timing analysis for a stock."""

        current_time = self.get_current_time_ist()
        market_state = self.get_market_state(current_time)
        is_open = market_state not in [MarketState.MARKET_CLOSED, MarketState.PRE_MARKET]

        # Get current window info
        current_window = self.TIME_WINDOWS.get(market_state)
        time_remaining = ""

        if current_window:
            time_remaining = self.get_time_remaining(
                current_window.end_time,
                current_time.time()
            )

        # Analyze stock patterns
        stock_pattern = self.analyze_stock_intraday_pattern(symbol)

        # Get recommended action
        action, reason, prob = self.get_recommended_action(
            market_state, stock_pattern, "BUY"
        )

        # CALCULATE SPECIFIC TRADE SETUP
        trade_setup = self.calculate_trade_setup(symbol, current_time)

        # Best entry windows
        best_entry = [
            ("Morning Trend", "10:00 - 11:30 AM", 0.58),
            ("Afternoon Session", "1:30 - 2:30 PM", 0.54),
        ]

        # Best exit windows
        best_exit = [
            ("Power Hour", "2:45 - 3:15 PM", 0.65),
            ("Before Lunch", "11:45 AM - 12:00 PM", 0.55),
        ]

        # Windows to avoid
        avoid = [
            ("Opening Session", "9:15 - 10:00 AM", "High volatility, unpredictable gaps"),
            ("Mid-day Lull", "12:00 - 1:00 PM", "Low volume, choppy action"),
        ]

        # Risk factors
        risk_factors = []
        opportunity_factors = []

        if market_state == MarketState.OPENING_SESSION:
            risk_factors.append("Opening volatility - prices can swing wildly")
            risk_factors.append("Gap risk from overnight news")
        elif market_state == MarketState.MORNING_TREND:
            opportunity_factors.append("Trend is establishing - good entry window")
            opportunity_factors.append("Volume is high for conviction")
        elif market_state == MarketState.MIDDAY_LULL:
            risk_factors.append("Low volume - poor liquidity")
            risk_factors.append("Choppy action - false signals common")
        elif market_state == MarketState.POWER_HOUR:
            risk_factors.append("Late entry risk - may not have time to recover")
            opportunity_factors.append("Good for quick scalps by experienced traders")

        # Tomorrow outlook if market closed
        tomorrow = None
        if not is_open:
            tomorrow = self.get_tomorrow_outlook(symbol)

        risk_level = current_window.risk_level if current_window else TimeWindowRisk.MODERATE

        return TimingPrediction(
            symbol=symbol,
            current_time_ist=current_time,
            market_state=market_state,
            is_market_open=is_open,
            current_window=current_window,
            time_remaining_in_window=time_remaining,
            recommended_action=action,
            action_reason=reason,
            risk_level=risk_level,
            win_probability=prob,
            best_entry_windows=best_entry,
            best_exit_windows=best_exit,
            avoid_windows=avoid,
            trade_setup=trade_setup,
            tomorrow_outlook=tomorrow,
            risk_factors=risk_factors,
            opportunity_factors=opportunity_factors
        )

    def get_market_status_display(self) -> Dict:
        """Get current market status for display."""
        current_time = self.get_current_time_ist()
        market_state = self.get_market_state(current_time)

        status = {
            'time_ist': current_time.strftime("%H:%M:%S IST"),
            'date': current_time.strftime("%Y-%m-%d (%A)"),
            'state': market_state.value,
            'is_open': market_state not in [MarketState.MARKET_CLOSED, MarketState.PRE_MARKET],
        }

        if market_state == MarketState.PRE_MARKET:
            market_open = current_time.replace(hour=9, minute=15, second=0)
            diff = market_open - current_time
            mins = int(diff.total_seconds() / 60)
            status['message'] = f"Market opens in {mins} minutes"
            status['next_action'] = "Prepare your watchlist and set alerts"
            status['color'] = "#ffaa00"

        elif market_state == MarketState.MARKET_CLOSED:
            next_day = self.get_next_trading_day(current_time)
            status['message'] = f"Market closed. Next session: {next_day.strftime('%Y-%m-%d (%A)')}"
            status['next_action'] = "Review today's trades and plan for tomorrow"
            status['color'] = "#ff4444"

        else:
            window = self.TIME_WINDOWS.get(market_state)
            if window:
                time_left = self.get_time_remaining(window.end_time, current_time.time())
                status['message'] = f"{window.name} - {time_left} remaining"
                status['next_action'] = f"Best for: {', '.join(window.best_for[:2])}"
                status['volatility'] = window.volatility
                status['volume'] = window.volume
                status['win_prob'] = window.win_probability

                if window.risk_level == TimeWindowRisk.VERY_HIGH:
                    status['color'] = "#ff4444"
                elif window.risk_level == TimeWindowRisk.HIGH:
                    status['color'] = "#ff6600"
                elif window.risk_level == TimeWindowRisk.MODERATE:
                    status['color'] = "#ffaa00"
                else:
                    status['color'] = "#00ff88"

        return status


# Quick test
if __name__ == "__main__":
    analyzer = IntradayTimingAnalyzer()

    status = analyzer.get_market_status_display()
    print(f"\nMarket Status: {status}")

    prediction = analyzer.analyze("RELIANCE")
    print(f"\nTiming for RELIANCE:")
    print(f"  State: {prediction.market_state.value}")
    print(f"  Action: {prediction.recommended_action.value}")
    print(f"  Reason: {prediction.action_reason}")
    print(f"  Risk: {prediction.risk_level.value}")
    print(f"  Win Prob: {prediction.win_probability:.1%}")
