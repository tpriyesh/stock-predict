"""
EnhancedDataHealthChecker - Extended Data Quality Validation

Adds critical checks missing from base implementation:
1. Earnings proximity check (high volatility risk)
2. Ex-dividend date detection (price drop expected)
3. F&O expiry impact (monthly expiry volatility)
4. Result season awareness (sector-wide uncertainty)
5. Corporate action adjustment validation
6. Index rebalancing periods
7. Global market event awareness

These event-based checks significantly impact prediction reliability.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import calendar
from loguru import logger

from src.core.data_health import (
    DataHealthChecker, DataHealth, DataIssue, DataStatus, IssueType
)


class EventType(Enum):
    """Types of market events that affect predictions."""
    EARNINGS = "earnings"
    EX_DIVIDEND = "ex_dividend"
    BONUS = "bonus"
    SPLIT = "split"
    RIGHTS = "rights"
    FO_EXPIRY = "fo_expiry"
    INDEX_REBALANCE = "index_rebalance"
    RESULT_SEASON = "result_season"
    GLOBAL_EVENT = "global_event"
    BUDGET = "budget"
    RBI_POLICY = "rbi_policy"


@dataclass
class UpcomingEvent:
    """An upcoming market event."""
    event_type: EventType
    date: date
    symbol: Optional[str]       # None for market-wide events
    description: str
    impact_level: str           # 'high', 'medium', 'low'
    volatility_multiplier: float


@dataclass
class ExtendedDataIssue(DataIssue):
    """Extended issue with event information."""
    event: Optional[UpcomingEvent] = None


@dataclass
class ExtendedDataHealth(DataHealth):
    """Extended health report with event awareness."""
    upcoming_events: List[UpcomingEvent] = field(default_factory=list)
    event_risk_score: float = 0.0
    trading_recommendation: str = "proceed"  # 'proceed', 'caution', 'avoid'
    optimal_entry_window: Optional[Tuple[date, date]] = None


class EventCalendar:
    """
    Manages known market events and their impact.

    In production, would fetch from:
    - NSE/BSE corporate actions API
    - Company earnings calendars
    - RBI policy dates
    - Government calendar
    """

    # Indian result seasons (approximate)
    RESULT_SEASONS = [
        # Q4 results (Apr-May)
        {'start_month': 4, 'start_day': 1, 'end_month': 5, 'end_day': 31},
        # Q1 results (Jul-Aug)
        {'start_month': 7, 'start_day': 1, 'end_month': 8, 'end_day': 15},
        # Q2 results (Oct-Nov)
        {'start_month': 10, 'start_day': 1, 'end_month': 11, 'end_day': 15},
        # Q3 results (Jan-Feb)
        {'start_month': 1, 'start_day': 1, 'end_month': 2, 'end_day': 15},
    ]

    # Known RBI policy dates (sample - would fetch dynamically)
    RBI_POLICY_DATES_2024 = [
        date(2024, 2, 8), date(2024, 4, 5), date(2024, 6, 7),
        date(2024, 8, 8), date(2024, 10, 9), date(2024, 12, 6)
    ]

    # NIFTY 50 rebalancing dates (semi-annual)
    INDEX_REBALANCE_MONTHS = [3, 9]  # March and September

    def __init__(self):
        # Cache for corporate actions (would be populated from API)
        self.earnings_calendar: Dict[str, List[date]] = {}
        self.dividend_calendar: Dict[str, List[Tuple[date, float]]] = {}
        self.corporate_actions: Dict[str, List[Dict]] = {}

    def get_fo_expiry(self, year: int, month: int) -> date:
        """Get F&O expiry date (last Thursday of month)."""
        cal = calendar.monthcalendar(year, month)
        # Find last Thursday
        thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY]]
        last_thursday = max(thursdays)
        return date(year, month, last_thursday)

    def get_upcoming_events(self,
                             symbol: str,
                             from_date: date,
                             days_ahead: int = 7) -> List[UpcomingEvent]:
        """Get all upcoming events for a symbol."""
        events = []
        to_date = from_date + timedelta(days=days_ahead)

        # 1. F&O expiry
        fo_expiry = self.get_fo_expiry(from_date.year, from_date.month)
        if from_date <= fo_expiry <= to_date:
            events.append(UpcomingEvent(
                event_type=EventType.FO_EXPIRY,
                date=fo_expiry,
                symbol=None,
                description=f"F&O expiry on {fo_expiry}",
                impact_level='high',
                volatility_multiplier=1.5
            ))

        # 2. Check if in result season
        if self._is_result_season(from_date):
            events.append(UpcomingEvent(
                event_type=EventType.RESULT_SEASON,
                date=from_date,
                symbol=None,
                description="Currently in result season",
                impact_level='medium',
                volatility_multiplier=1.3
            ))

        # 3. Index rebalancing
        if from_date.month in self.INDEX_REBALANCE_MONTHS:
            if 20 <= from_date.day <= 31:
                events.append(UpcomingEvent(
                    event_type=EventType.INDEX_REBALANCE,
                    date=date(from_date.year, from_date.month, 25),
                    symbol=None,
                    description="NIFTY rebalancing period",
                    impact_level='medium',
                    volatility_multiplier=1.2
                ))

        # 4. RBI Policy
        for policy_date in self.RBI_POLICY_DATES_2024:
            if from_date <= policy_date <= to_date:
                events.append(UpcomingEvent(
                    event_type=EventType.RBI_POLICY,
                    date=policy_date,
                    symbol=None,
                    description=f"RBI Policy announcement on {policy_date}",
                    impact_level='high',
                    volatility_multiplier=1.4
                ))

        # 5. Stock-specific events from cache
        if symbol in self.earnings_calendar:
            for earnings_date in self.earnings_calendar[symbol]:
                if from_date <= earnings_date <= to_date:
                    events.append(UpcomingEvent(
                        event_type=EventType.EARNINGS,
                        date=earnings_date,
                        symbol=symbol,
                        description=f"{symbol} earnings on {earnings_date}",
                        impact_level='high',
                        volatility_multiplier=2.0
                    ))

        if symbol in self.dividend_calendar:
            for div_date, div_amount in self.dividend_calendar[symbol]:
                if from_date <= div_date <= to_date:
                    events.append(UpcomingEvent(
                        event_type=EventType.EX_DIVIDEND,
                        date=div_date,
                        symbol=symbol,
                        description=f"{symbol} ex-dividend Rs.{div_amount} on {div_date}",
                        impact_level='medium',
                        volatility_multiplier=1.1
                    ))

        return sorted(events, key=lambda x: x.date)

    def _is_result_season(self, check_date: date) -> bool:
        """Check if date falls in result season."""
        for season in self.RESULT_SEASONS:
            start = date(check_date.year, season['start_month'], season['start_day'])
            end = date(check_date.year, season['end_month'], season['end_day'])
            if start <= check_date <= end:
                return True
        return False

    def register_earnings(self, symbol: str, earnings_dates: List[date]):
        """Register known earnings dates for a stock."""
        self.earnings_calendar[symbol] = earnings_dates

    def register_dividend(self, symbol: str, ex_date: date, amount: float):
        """Register ex-dividend date for a stock."""
        if symbol not in self.dividend_calendar:
            self.dividend_calendar[symbol] = []
        self.dividend_calendar[symbol].append((ex_date, amount))


class EnhancedDataHealthChecker(DataHealthChecker):
    """
    Extended data health checker with event awareness.
    """

    def __init__(self,
                 min_rows: int = 252,
                 max_missing_pct: float = 0.05,
                 min_volume: int = 100_000,
                 strict_mode: bool = False):
        super().__init__(min_rows, max_missing_pct, min_volume, strict_mode)
        self.event_calendar = EventCalendar()

    def check_extended(self,
                        df: pd.DataFrame,
                        symbol: str = "Unknown",
                        trade_date: date = None) -> ExtendedDataHealth:
        """
        Perform comprehensive data health check including events.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            trade_date: Date for event checking

        Returns:
            ExtendedDataHealth with complete assessment
        """
        # Get base health check
        base_health = super().check(df, symbol)

        trade_date = trade_date or date.today()

        # Get upcoming events
        events = self.event_calendar.get_upcoming_events(symbol, trade_date, days_ahead=7)

        # Additional checks
        additional_issues = []

        # 1. Corporate action detection from price data
        corp_action_issues = self._detect_corporate_actions_from_data(df, symbol)
        additional_issues.extend(corp_action_issues)

        # 2. Pre-announcement volatility pattern
        volatility_issues = self._check_pre_event_volatility(df, symbol, events)
        additional_issues.extend(volatility_issues)

        # 3. Dividend adjustment check
        div_issues = self._check_dividend_adjustment(df, symbol)
        additional_issues.extend(div_issues)

        # 4. Split detection
        split_issues = self._check_for_splits(df, symbol)
        additional_issues.extend(split_issues)

        # Calculate event risk score
        event_risk_score = self._calculate_event_risk(events)

        # Determine trading recommendation
        recommendation = self._determine_recommendation(
            base_health, events, event_risk_score
        )

        # Find optimal entry window
        optimal_window = self._find_optimal_window(trade_date, events)

        # Combine issues
        all_issues = base_health.issues + additional_issues

        # Recalculate confidence penalty
        penalty = base_health.confidence_penalty
        penalty += event_risk_score * 0.2  # Event risk adds to penalty
        penalty = min(0.6, penalty)

        return ExtendedDataHealth(
            status=self._recalculate_status(all_issues, events),
            issues=all_issues,
            metrics=base_health.metrics,
            recommendations=base_health.recommendations + self._get_event_recommendations(events),
            can_predict=base_health.can_predict and recommendation != 'avoid',
            confidence_penalty=penalty,
            upcoming_events=events,
            event_risk_score=event_risk_score,
            trading_recommendation=recommendation,
            optimal_entry_window=optimal_window
        )

    def _detect_corporate_actions_from_data(self,
                                             df: pd.DataFrame,
                                             symbol: str) -> List[DataIssue]:
        """Detect corporate actions from price/volume patterns."""
        issues = []

        if len(df) < 10:
            return issues

        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else None

        # Look for sudden price gaps (potential splits/bonus)
        returns = close.pct_change().dropna()

        # Find gaps > 40% (potential split)
        large_gaps = returns[returns.abs() > 0.40]
        for idx in large_gaps.index:
            gap_return = returns[idx]

            # Check if it's a split (price drops but market cap should stay same)
            if gap_return < -0.40 and volume is not None:
                # Volume should spike on split adjustment
                vol_ratio = volume[idx] / volume.shift(1)[idx] if pd.notna(volume.shift(1)[idx]) else 1

                issues.append(DataIssue(
                    issue_type=IssueType.CORPORATE_ACTION,
                    severity=DataStatus.WARNING,
                    message=f"Potential stock split detected on {idx.date() if hasattr(idx, 'date') else idx}: {gap_return:.1%}",
                    value=gap_return,
                    threshold=0.40,
                    affected_dates=[str(idx)]
                ))

        return issues

    def _check_pre_event_volatility(self,
                                     df: pd.DataFrame,
                                     symbol: str,
                                     events: List[UpcomingEvent]) -> List[DataIssue]:
        """Check for unusual pre-event volatility."""
        issues = []

        if len(df) < 20:
            return issues

        returns = df['close'].pct_change().dropna()
        recent_vol = returns.tail(5).std()
        normal_vol = returns.tail(20).std()

        if normal_vol > 0:
            vol_ratio = recent_vol / normal_vol

            if vol_ratio > 2.0:
                # Check if this correlates with upcoming event
                high_impact_events = [e for e in events if e.impact_level == 'high']

                if high_impact_events:
                    issues.append(DataIssue(
                        issue_type=IssueType.HIGH_VOLATILITY,
                        severity=DataStatus.WARNING,
                        message=f"Pre-event volatility surge: {vol_ratio:.1f}x normal",
                        value=vol_ratio,
                        threshold=2.0
                    ))

        return issues

    def _check_dividend_adjustment(self,
                                    df: pd.DataFrame,
                                    symbol: str) -> List[DataIssue]:
        """Check if prices appear dividend-adjusted."""
        issues = []

        if len(df) < 50:
            return issues

        # Look for small gaps consistent with dividend payments
        returns = df['close'].pct_change().dropna()

        # Dividends typically cause 1-3% drops
        dividend_like_drops = returns[(returns < -0.005) & (returns > -0.05)]

        # If many such drops, data might not be adjusted
        drops_per_year = len(dividend_like_drops) * 252 / len(df)

        if drops_per_year > 10:
            issues.append(DataIssue(
                issue_type=IssueType.CORPORATE_ACTION,
                severity=DataStatus.WARNING,
                message="Data may contain unadjusted dividend drops",
                value=drops_per_year,
                threshold=10
            ))

        return issues

    def _check_for_splits(self,
                          df: pd.DataFrame,
                          symbol: str) -> List[DataIssue]:
        """Detect potential stock splits."""
        issues = []

        if len(df) < 20:
            return issues

        close = df['close']

        # Common split ratios and their return signatures
        split_signatures = {
            '2:1': (-0.45, -0.55),   # Price halves
            '5:1': (-0.75, -0.85),   # Price drops 80%
            '10:1': (-0.88, -0.92),  # Price drops 90%
            '1:2': (0.90, 1.10),     # Reverse split - price doubles
        }

        returns = close.pct_change().dropna()

        for split_type, (low, high) in split_signatures.items():
            matches = returns[(returns > low) & (returns < high)]

            for idx in matches.index:
                issues.append(DataIssue(
                    issue_type=IssueType.CORPORATE_ACTION,
                    severity=DataStatus.WARNING,
                    message=f"Potential {split_type} split detected on {idx.date() if hasattr(idx, 'date') else idx}",
                    value=float(returns[idx]),
                    threshold=0.5,
                    affected_dates=[str(idx)]
                ))

        return issues

    def _calculate_event_risk(self, events: List[UpcomingEvent]) -> float:
        """Calculate combined event risk score."""
        if not events:
            return 0.0

        risk = 0.0
        for event in events:
            if event.impact_level == 'high':
                risk += 0.3
            elif event.impact_level == 'medium':
                risk += 0.15
            else:
                risk += 0.05

        return min(1.0, risk)

    def _determine_recommendation(self,
                                   base_health: DataHealth,
                                   events: List[UpcomingEvent],
                                   event_risk: float) -> str:
        """Determine trading recommendation."""
        # Avoid if data is bad
        if not base_health.can_predict:
            return 'avoid'

        # Check for high-impact imminent events
        high_impact_imminent = any(
            e.impact_level == 'high' and (e.date - date.today()).days <= 2
            for e in events
        )

        if high_impact_imminent:
            return 'avoid'

        # Caution during high event risk
        if event_risk > 0.4:
            return 'caution'

        # Caution if moderate confidence penalty
        if base_health.confidence_penalty > 0.2:
            return 'caution'

        return 'proceed'

    def _find_optimal_window(self,
                              trade_date: date,
                              events: List[UpcomingEvent]) -> Optional[Tuple[date, date]]:
        """Find optimal trading window avoiding events."""
        if not events:
            return (trade_date, trade_date + timedelta(days=7))

        # Find gaps between events
        event_dates = sorted([e.date for e in events if e.impact_level in ['high', 'medium']])

        if not event_dates:
            return (trade_date, trade_date + timedelta(days=7))

        # If first event is far, window is now until event
        first_event = event_dates[0]
        days_to_event = (first_event - trade_date).days

        if days_to_event > 3:
            return (trade_date, first_event - timedelta(days=2))

        # If events are close, find next gap
        for i in range(len(event_dates) - 1):
            gap = (event_dates[i + 1] - event_dates[i]).days
            if gap > 5:
                return (event_dates[i] + timedelta(days=2), event_dates[i + 1] - timedelta(days=2))

        # After last event
        return (event_dates[-1] + timedelta(days=2), event_dates[-1] + timedelta(days=9))

    def _recalculate_status(self,
                             issues: List[DataIssue],
                             events: List[UpcomingEvent]) -> DataStatus:
        """Recalculate status considering events."""
        base_status = super()._determine_status(issues)

        # Elevate status if high-impact event imminent
        high_impact = any(e.impact_level == 'high' for e in events)

        if high_impact and base_status == DataStatus.OK:
            return DataStatus.WARNING

        return base_status

    def _get_event_recommendations(self, events: List[UpcomingEvent]) -> List[str]:
        """Generate event-specific recommendations."""
        recs = []

        for event in events:
            if event.event_type == EventType.EARNINGS:
                recs.append(f"Consider waiting until after earnings on {event.date}")
                recs.append("Reduce position size due to earnings volatility risk")

            elif event.event_type == EventType.EX_DIVIDEND:
                recs.append(f"Price will drop by dividend amount on {event.date}")

            elif event.event_type == EventType.FO_EXPIRY:
                recs.append("Expect higher volatility around F&O expiry")
                recs.append("Consider wider stop-losses near expiry")

            elif event.event_type == EventType.RBI_POLICY:
                recs.append("Bank and NBFC stocks will be volatile around RBI policy")

            elif event.event_type == EventType.RESULT_SEASON:
                recs.append("Many stocks will report earnings - expect sector-wide volatility")

        return recs


def demo():
    """Demonstrate enhanced data health checking."""
    print("=" * 60)
    print("EnhancedDataHealthChecker Demo")
    print("=" * 60)

    # Create sample data with corporate action
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='B')

    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 300))

    # Simulate a 2:1 split at day 150
    prices[150:] = prices[150:] / 2

    df = pd.DataFrame({
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, 300)
    }, index=dates)

    checker = EnhancedDataHealthChecker()

    # Register some events
    checker.event_calendar.register_earnings('TCS', [date.today() + timedelta(days=3)])

    print("\n--- Checking data with split and upcoming earnings ---")
    health = checker.check_extended(df, symbol='TCS', trade_date=date.today())

    print(f"Status: {health.status.value}")
    print(f"Can Predict: {health.can_predict}")
    print(f"Event Risk Score: {health.event_risk_score:.2f}")
    print(f"Trading Recommendation: {health.trading_recommendation}")
    print(f"Confidence Penalty: {health.confidence_penalty:.1%}")

    print("\nUpcoming Events:")
    for event in health.upcoming_events:
        print(f"  - [{event.impact_level}] {event.description}")

    print("\nIssues Detected:")
    for issue in health.issues:
        print(f"  - [{issue.severity.value}] {issue.message}")

    print("\nRecommendations:")
    for rec in health.recommendations:
        print(f"  * {rec}")

    if health.optimal_entry_window:
        print(f"\nOptimal Entry Window: {health.optimal_entry_window[0]} to {health.optimal_entry_window[1]}")


if __name__ == "__main__":
    demo()
