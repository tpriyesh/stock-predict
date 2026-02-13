"""
Data validation utilities to ensure data quality.
CRITICAL: Money is involved - no false or incomplete data allowed.
"""
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.storage.models import StockPrice


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataValidator:
    """
    Validates market data for quality and completeness.

    All validation errors are logged and raised - we never silently
    accept bad data when money is involved.
    """

    # Indian market holidays 2024-2025 (sample - should be updated annually)
    MARKET_HOLIDAYS = {
        date(2024, 1, 26),   # Republic Day
        date(2024, 3, 8),    # Maha Shivaratri
        date(2024, 3, 25),   # Holi
        date(2024, 3, 29),   # Good Friday
        date(2024, 4, 11),   # Id-Ul-Fitr
        date(2024, 4, 14),   # Dr. Ambedkar Jayanti
        date(2024, 4, 17),   # Ram Navami
        date(2024, 4, 21),   # Mahavir Jayanti
        date(2024, 5, 23),   # Buddha Purnima
        date(2024, 6, 17),   # Id-Ul-Adha
        date(2024, 7, 17),   # Muharram
        date(2024, 8, 15),   # Independence Day
        date(2024, 10, 2),   # Mahatma Gandhi Jayanti
        date(2024, 11, 1),   # Diwali
        date(2024, 11, 15),  # Guru Nanak Jayanti
        date(2024, 12, 25),  # Christmas
        date(2025, 1, 26),   # Republic Day
        date(2025, 2, 26),   # Maha Shivaratri
        date(2025, 3, 14),   # Holi
        date(2025, 3, 31),   # Id-Ul-Fitr
        date(2025, 4, 6),    # Ram Navami
        date(2025, 4, 10),   # Mahavir Jayanti
        date(2025, 4, 14),   # Dr. Ambedkar Jayanti
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 12),   # Buddha Purnima
        date(2025, 6, 7),    # Id-Ul-Adha
        date(2025, 7, 6),    # Muharram
        date(2025, 8, 15),   # Independence Day
        date(2025, 8, 16),   # Janmashtami
        date(2025, 10, 2),   # Mahatma Gandhi Jayanti/Dussehra
        date(2025, 10, 21),  # Diwali
        date(2025, 11, 5),   # Guru Nanak Jayanti
        date(2025, 12, 25),  # Christmas
    }

    @classmethod
    def is_trading_day(cls, d: date) -> bool:
        """Check if a date is a trading day (not weekend or holiday)."""
        if d.weekday() >= 5:  # Saturday or Sunday
            return False
        if d in cls.MARKET_HOLIDAYS:
            return False
        return True

    @classmethod
    def get_trading_days(cls, start: date, end: date) -> list[date]:
        """Get list of trading days between two dates."""
        days = []
        current = start
        while current <= end:
            if cls.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    @classmethod
    def validate_ohlc(cls, price: StockPrice) -> tuple[bool, list[str]]:
        """
        Validate OHLC data for a single record.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check positive values
        for field in ['open', 'high', 'low', 'close']:
            if getattr(price, field) <= 0:
                errors.append(f"{field} must be positive, got {getattr(price, field)}")

        # Check OHLC relationships
        if price.high < price.low:
            errors.append(f"high ({price.high}) < low ({price.low})")

        if price.high < price.open:
            errors.append(f"high ({price.high}) < open ({price.open})")

        if price.high < price.close:
            errors.append(f"high ({price.high}) < close ({price.close})")

        if price.low > price.open:
            errors.append(f"low ({price.low}) > open ({price.open})")

        if price.low > price.close:
            errors.append(f"low ({price.low}) > close ({price.close})")

        # Check volume (0 is suspicious but not always invalid)
        if price.volume < 0:
            errors.append(f"volume must be non-negative, got {price.volume}")

        # Check for suspicious price movements (>20% in a day is unusual)
        day_range_pct = (price.high - price.low) / price.low * 100 if price.low > 0 else 0
        if day_range_pct > 20:
            logger.warning(
                f"Large price range for {price.symbol} on {price.date}: "
                f"{day_range_pct:.1f}% (may be valid for volatile stocks)"
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_price_series(
        cls,
        prices: pd.DataFrame,
        symbol: str,
        max_gap_days: int = 5
    ) -> tuple[bool, list[str]]:
        """
        Validate a time series of price data.

        Args:
            prices: DataFrame with OHLCV columns and date index
            symbol: Stock symbol for logging
            max_gap_days: Maximum allowed gap between trading days

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if prices.empty:
            errors.append(f"No price data for {symbol}")
            return False, errors

        # Ensure sorted by date
        prices = prices.sort_index()

        # Check for duplicates
        if prices.index.duplicated().any():
            dup_dates = prices.index[prices.index.duplicated()].tolist()
            errors.append(f"Duplicate dates found: {dup_dates[:5]}...")

        # Check for gaps (excluding holidays/weekends)
        dates = pd.to_datetime(prices.index).date if hasattr(prices.index, 'date') else prices.index.tolist()
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            expected_days = cls.get_trading_days(prev_date + timedelta(days=1), curr_date)
            if len(expected_days) > max_gap_days:
                errors.append(
                    f"Gap of {len(expected_days)} trading days between "
                    f"{prev_date} and {curr_date}"
                )

        # Check for zero/negative values
        for col in ['open', 'high', 'low', 'close']:
            if col in prices.columns:
                if (prices[col] <= 0).any():
                    bad_dates = prices[prices[col] <= 0].index.tolist()
                    errors.append(f"Non-positive {col} values on: {bad_dates[:3]}...")

        # Check for NaN values
        if prices[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
            nan_counts = prices[['open', 'high', 'low', 'close', 'volume']].isna().sum()
            errors.append(f"NaN values found: {nan_counts.to_dict()}")

        # Check OHLC relationships row by row
        invalid_ohlc = prices[
            (prices['high'] < prices['low']) |
            (prices['high'] < prices['open']) |
            (prices['high'] < prices['close']) |
            (prices['low'] > prices['open']) |
            (prices['low'] > prices['close'])
        ]
        if not invalid_ohlc.empty:
            errors.append(f"Invalid OHLC relationships on {len(invalid_ohlc)} rows")

        # Check for suspicious jumps (>50% overnight)
        if len(prices) > 1:
            prices['prev_close'] = prices['close'].shift(1)
            prices['overnight_pct'] = abs(
                (prices['open'] - prices['prev_close']) / prices['prev_close'] * 100
            )
            jumps = prices[prices['overnight_pct'] > 50].dropna()
            if not jumps.empty:
                logger.warning(
                    f"{symbol}: {len(jumps)} overnight jumps >50% detected. "
                    f"May indicate splits/bonuses. Verify data."
                )

        return len(errors) == 0, errors

    @classmethod
    def validate_data_freshness(
        cls,
        latest_date: date,
        symbol: str,
        max_staleness_days: int = 1
    ) -> tuple[bool, str]:
        """
        Check if data is fresh enough for trading decisions.

        Args:
            latest_date: Most recent data date
            symbol: Stock symbol
            max_staleness_days: Maximum allowed staleness

        Returns:
            (is_fresh, message)
        """
        today = date.today()

        # Find the most recent expected trading day
        check_date = today
        while not cls.is_trading_day(check_date) and check_date > latest_date:
            check_date -= timedelta(days=1)

        staleness = (check_date - latest_date).days

        if staleness > max_staleness_days:
            return False, (
                f"{symbol} data is stale: latest={latest_date}, "
                f"expected={check_date}, gap={staleness} days"
            )

        return True, f"{symbol} data is fresh (latest={latest_date})"

    @classmethod
    def validate_volume_liquidity(
        cls,
        prices: pd.DataFrame,
        symbol: str,
        min_avg_volume_cr: float = 10.0,
        lookback_days: int = 20
    ) -> tuple[bool, float, str]:
        """
        Check if stock has sufficient liquidity.

        Args:
            prices: DataFrame with close and volume columns
            symbol: Stock symbol
            min_avg_volume_cr: Minimum average daily value in Crores
            lookback_days: Days to average

        Returns:
            (meets_threshold, avg_value_cr, message)
        """
        if len(prices) < lookback_days:
            return False, 0.0, f"Insufficient data: {len(prices)} < {lookback_days} days"

        recent = prices.tail(lookback_days)
        avg_value = (recent['close'] * recent['volume']).mean()
        avg_value_cr = avg_value / 1e7  # Convert to Crores

        if avg_value_cr < min_avg_volume_cr:
            return False, avg_value_cr, (
                f"{symbol} liquidity too low: ₹{avg_value_cr:.1f}Cr < ₹{min_avg_volume_cr}Cr"
            )

        return True, avg_value_cr, f"{symbol} liquidity OK: ₹{avg_value_cr:.1f}Cr/day"
