"""
DataHealthChecker - Validates Data Quality Before Prediction

CRITICAL: Garbage In = Garbage Out

This module performs comprehensive data quality checks:
1. Minimum history requirements
2. Missing data detection
3. Liquidity validation (volume)
4. Corporate action anomaly detection
5. Stale data detection
6. Price sanity checks
7. Data freshness validation

If data fails checks, predictions should be REJECTED - not made with bad data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class DataStatus(Enum):
    """Data health status levels."""
    OK = "OK"
    WARNING = "WARNING"
    INSUFFICIENT = "INSUFFICIENT"
    CRITICAL = "CRITICAL"


class IssueType(Enum):
    """Types of data issues."""
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    HIGH_MISSING_DATA = "HIGH_MISSING_DATA"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    STALE_DATA = "STALE_DATA"
    PRICE_ANOMALY = "PRICE_ANOMALY"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    DATA_GAP = "DATA_GAP"
    ZERO_VOLUME = "ZERO_VOLUME"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    EXTREME_MOVE = "EXTREME_MOVE"


@dataclass
class DataIssue:
    """A single data quality issue."""
    issue_type: IssueType
    severity: DataStatus
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    affected_dates: Optional[List[str]] = None


@dataclass
class DataHealth:
    """Comprehensive data health report."""
    status: DataStatus
    issues: List[DataIssue]
    metrics: Dict[str, float]
    recommendations: List[str]
    can_predict: bool
    confidence_penalty: float  # 0-1, reduce confidence by this amount

    def __post_init__(self):
        """Calculate confidence penalty based on issues."""
        if not hasattr(self, 'confidence_penalty') or self.confidence_penalty == 0:
            penalty = 0.0
            for issue in self.issues:
                if issue.severity == DataStatus.WARNING:
                    penalty += 0.05
                elif issue.severity == DataStatus.INSUFFICIENT:
                    penalty += 0.15
                elif issue.severity == DataStatus.CRITICAL:
                    penalty += 0.30
            self.confidence_penalty = min(0.5, penalty)  # Cap at 50% penalty

    def to_dict(self) -> dict:
        """Serialize for API/storage."""
        return {
            'status': self.status.value,
            'issues': [
                {
                    'type': i.issue_type.value,
                    'severity': i.severity.value,
                    'message': i.message,
                    'value': i.value,
                    'threshold': i.threshold
                }
                for i in self.issues
            ],
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'can_predict': self.can_predict,
            'confidence_penalty': self.confidence_penalty
        }


class DataHealthChecker:
    """
    Comprehensive data quality validator for stock prediction.

    Usage:
        checker = DataHealthChecker()
        health = checker.check(df)

        if not health.can_predict:
            raise ValueError(f"Cannot predict: {health.issues}")

        if health.confidence_penalty > 0:
            prediction.confidence *= (1 - health.confidence_penalty)
    """

    # Configurable thresholds
    MIN_ROWS = 252                    # 1 year minimum for reliable predictions
    MIN_ROWS_HARD = 100               # Absolute minimum (will not predict below this)
    MAX_MISSING_PCT = 0.05            # 5% max missing data
    MAX_MISSING_PCT_HARD = 0.15       # 15% = insufficient
    MIN_VOLUME_AVG = 100_000          # Minimum average daily volume
    MIN_VOLUME_AVG_HARD = 10_000      # Below this = illiquid
    MAX_STALE_DAYS = 3                # Max days since last data
    MAX_SINGLE_DAY_MOVE = 0.20        # 20% single day move = potential corp action
    MAX_GAP_DAYS = 5                  # Max consecutive missing days
    MIN_TRADING_DAYS_PER_MONTH = 15   # Minimum trading days expected

    def __init__(self,
                 min_rows: int = 252,
                 max_missing_pct: float = 0.05,
                 min_volume: int = 100_000,
                 strict_mode: bool = False):
        """
        Initialize health checker.

        Args:
            min_rows: Minimum required rows (default 1 year)
            max_missing_pct: Maximum allowed missing data percentage
            min_volume: Minimum average daily volume
            strict_mode: If True, any warning becomes INSUFFICIENT
        """
        self.min_rows = min_rows
        self.max_missing_pct = max_missing_pct
        self.min_volume = min_volume
        self.strict_mode = strict_mode

    def check(self, df: pd.DataFrame, symbol: str = "Unknown") -> DataHealth:
        """
        Perform comprehensive data health check.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            symbol: Stock symbol for logging

        Returns:
            DataHealth object with complete assessment
        """
        issues: List[DataIssue] = []
        metrics: Dict[str, float] = {}
        recommendations: List[str] = []

        logger.debug(f"Checking data health for {symbol}: {len(df)} rows")

        # 1. Row count check
        issues.extend(self._check_row_count(df, metrics))

        # 2. Missing data check
        issues.extend(self._check_missing_data(df, metrics))

        # 3. Volume/liquidity check
        issues.extend(self._check_liquidity(df, metrics))

        # 4. Data freshness check
        issues.extend(self._check_freshness(df, metrics))

        # 5. Price sanity check
        issues.extend(self._check_price_sanity(df, metrics))

        # 6. Corporate action detection
        issues.extend(self._check_corporate_actions(df, metrics))

        # 7. Data gap detection
        issues.extend(self._check_data_gaps(df, metrics))

        # 8. Volatility check
        issues.extend(self._check_volatility(df, metrics))

        # Determine overall status
        status = self._determine_status(issues)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        # Can we make a prediction?
        can_predict = status not in [DataStatus.INSUFFICIENT, DataStatus.CRITICAL]

        # In strict mode, warnings also block prediction
        if self.strict_mode and status == DataStatus.WARNING:
            can_predict = False

        # Calculate confidence penalty
        penalty = sum(
            0.05 if i.severity == DataStatus.WARNING else
            0.15 if i.severity == DataStatus.INSUFFICIENT else
            0.30 if i.severity == DataStatus.CRITICAL else 0
            for i in issues
        )
        penalty = min(0.5, penalty)

        return DataHealth(
            status=status,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            can_predict=can_predict,
            confidence_penalty=penalty
        )

    def _check_row_count(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check if we have enough historical data."""
        issues = []
        n_rows = len(df)
        metrics['row_count'] = n_rows
        metrics['years_of_data'] = n_rows / 252

        if n_rows < self.MIN_ROWS_HARD:
            issues.append(DataIssue(
                issue_type=IssueType.INSUFFICIENT_HISTORY,
                severity=DataStatus.CRITICAL,
                message=f"Only {n_rows} rows (need at least {self.MIN_ROWS_HARD})",
                value=n_rows,
                threshold=self.MIN_ROWS_HARD
            ))
        elif n_rows < self.min_rows:
            issues.append(DataIssue(
                issue_type=IssueType.INSUFFICIENT_HISTORY,
                severity=DataStatus.WARNING,
                message=f"Only {n_rows} rows (recommended: {self.min_rows})",
                value=n_rows,
                threshold=self.min_rows
            ))

        return issues

    def _check_missing_data(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check for missing/null values."""
        issues = []

        # Overall missing percentage
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_pct = missing_cells / total_cells if total_cells > 0 else 0

        metrics['missing_pct'] = missing_pct
        metrics['missing_cells'] = missing_cells

        # Per-column analysis
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            if col in df.columns:
                col_missing = df[col].isnull().sum()
                metrics[f'{col.lower()}_missing'] = col_missing

        if missing_pct > self.MAX_MISSING_PCT_HARD:
            issues.append(DataIssue(
                issue_type=IssueType.HIGH_MISSING_DATA,
                severity=DataStatus.CRITICAL,
                message=f"Missing data: {missing_pct:.1%} (threshold: {self.MAX_MISSING_PCT_HARD:.1%})",
                value=missing_pct,
                threshold=self.MAX_MISSING_PCT_HARD
            ))
        elif missing_pct > self.max_missing_pct:
            issues.append(DataIssue(
                issue_type=IssueType.HIGH_MISSING_DATA,
                severity=DataStatus.WARNING,
                message=f"Missing data: {missing_pct:.1%} (recommended: {self.max_missing_pct:.1%})",
                value=missing_pct,
                threshold=self.max_missing_pct
            ))

        return issues

    def _check_liquidity(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check trading volume for liquidity."""
        issues = []

        if 'Volume' not in df.columns:
            issues.append(DataIssue(
                issue_type=IssueType.LOW_LIQUIDITY,
                severity=DataStatus.WARNING,
                message="No volume data available"
            ))
            return issues

        avg_volume = df['Volume'].mean()
        metrics['avg_volume'] = avg_volume

        # Check for zero volume days
        zero_vol_days = (df['Volume'] == 0).sum()
        zero_vol_pct = zero_vol_days / len(df)
        metrics['zero_volume_days'] = zero_vol_days
        metrics['zero_volume_pct'] = zero_vol_pct

        if zero_vol_pct > 0.10:  # More than 10% zero volume days
            issues.append(DataIssue(
                issue_type=IssueType.ZERO_VOLUME,
                severity=DataStatus.WARNING,
                message=f"{zero_vol_days} days ({zero_vol_pct:.1%}) with zero volume",
                value=zero_vol_pct,
                threshold=0.10
            ))

        if avg_volume < self.MIN_VOLUME_AVG_HARD:
            issues.append(DataIssue(
                issue_type=IssueType.LOW_LIQUIDITY,
                severity=DataStatus.CRITICAL,
                message=f"Average volume {avg_volume:,.0f} (need {self.MIN_VOLUME_AVG_HARD:,})",
                value=avg_volume,
                threshold=self.MIN_VOLUME_AVG_HARD
            ))
        elif avg_volume < self.min_volume:
            issues.append(DataIssue(
                issue_type=IssueType.LOW_LIQUIDITY,
                severity=DataStatus.WARNING,
                message=f"Average volume {avg_volume:,.0f} (recommended {self.min_volume:,})",
                value=avg_volume,
                threshold=self.min_volume
            ))

        return issues

    def _check_freshness(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check if data is stale."""
        issues = []

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex - skipping freshness check")
            return issues

        latest_date = df.index.max()
        today = datetime.now()

        # Account for weekends
        days_since = (today - latest_date).days
        business_days_since = np.busday_count(
            latest_date.date(),
            today.date()
        )

        metrics['latest_date'] = str(latest_date.date())
        metrics['days_since_update'] = days_since
        metrics['business_days_since'] = business_days_since

        if business_days_since > self.MAX_STALE_DAYS:
            issues.append(DataIssue(
                issue_type=IssueType.STALE_DATA,
                severity=DataStatus.WARNING,
                message=f"Data is {business_days_since} business days old (latest: {latest_date.date()})",
                value=business_days_since,
                threshold=self.MAX_STALE_DAYS
            ))

        return issues

    def _check_price_sanity(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check for price anomalies."""
        issues = []

        if 'Close' not in df.columns:
            return issues

        close = df['Close']

        # Negative prices
        neg_prices = (close <= 0).sum()
        if neg_prices > 0:
            issues.append(DataIssue(
                issue_type=IssueType.NEGATIVE_PRICE,
                severity=DataStatus.CRITICAL,
                message=f"{neg_prices} rows with zero or negative price",
                value=neg_prices,
                threshold=0
            ))

        # Check OHLC relationship
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            ).sum()

            metrics['invalid_ohlc_rows'] = invalid_ohlc
            if invalid_ohlc > 0:
                issues.append(DataIssue(
                    issue_type=IssueType.PRICE_ANOMALY,
                    severity=DataStatus.WARNING,
                    message=f"{invalid_ohlc} rows with invalid OHLC relationship",
                    value=invalid_ohlc,
                    threshold=0
                ))

        return issues

    def _check_corporate_actions(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Detect potential corporate actions (splits, bonus, etc.)."""
        issues = []

        if 'Close' not in df.columns or len(df) < 2:
            return issues

        returns = df['Close'].pct_change().dropna()

        # Count extreme moves
        extreme_moves = (returns.abs() > self.MAX_SINGLE_DAY_MOVE).sum()
        metrics['extreme_move_count'] = extreme_moves

        if extreme_moves > 5:
            # Find dates of extreme moves
            extreme_dates = returns[returns.abs() > self.MAX_SINGLE_DAY_MOVE].index
            date_strings = [str(d.date()) for d in extreme_dates[:5]]  # First 5

            issues.append(DataIssue(
                issue_type=IssueType.CORPORATE_ACTION,
                severity=DataStatus.WARNING,
                message=f"{extreme_moves} days with >20% move (potential corporate actions)",
                value=extreme_moves,
                threshold=5,
                affected_dates=date_strings
            ))

        return issues

    def _check_data_gaps(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check for gaps in trading data."""
        issues = []

        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return issues

        # Calculate gaps
        date_diffs = pd.Series(df.index).diff().dt.days
        max_gap = date_diffs.max()
        metrics['max_gap_days'] = max_gap

        # Count gaps > threshold
        significant_gaps = (date_diffs > self.MAX_GAP_DAYS).sum()
        metrics['significant_gaps'] = significant_gaps

        if max_gap > 10:  # More than 2 weeks
            issues.append(DataIssue(
                issue_type=IssueType.DATA_GAP,
                severity=DataStatus.WARNING,
                message=f"Maximum gap of {max_gap} days in data",
                value=max_gap,
                threshold=self.MAX_GAP_DAYS
            ))

        return issues

    def _check_volatility(self, df: pd.DataFrame, metrics: Dict) -> List[DataIssue]:
        """Check for extreme volatility that might indicate data issues."""
        issues = []

        if 'Close' not in df.columns or len(df) < 20:
            return issues

        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        metrics['annualized_volatility'] = volatility

        # Check if volatility is unreasonably high
        if volatility > 1.5:  # 150% annualized volatility
            issues.append(DataIssue(
                issue_type=IssueType.HIGH_VOLATILITY,
                severity=DataStatus.WARNING,
                message=f"Extremely high volatility: {volatility:.0%} annualized",
                value=volatility,
                threshold=1.5
            ))

        return issues

    def _determine_status(self, issues: List[DataIssue]) -> DataStatus:
        """Determine overall status from issues."""
        if not issues:
            return DataStatus.OK

        severities = [i.severity for i in issues]

        if DataStatus.CRITICAL in severities:
            return DataStatus.CRITICAL
        elif DataStatus.INSUFFICIENT in severities:
            return DataStatus.INSUFFICIENT
        elif DataStatus.WARNING in severities:
            return DataStatus.WARNING

        return DataStatus.OK

    def _generate_recommendations(self, issues: List[DataIssue]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        issue_types = set(i.issue_type for i in issues)

        if IssueType.INSUFFICIENT_HISTORY in issue_types:
            recommendations.append("Wait for more historical data before trading this stock")

        if IssueType.HIGH_MISSING_DATA in issue_types:
            recommendations.append("Use data from a different source or forward-fill missing values")

        if IssueType.LOW_LIQUIDITY in issue_types:
            recommendations.append("Avoid large position sizes due to liquidity risk")
            recommendations.append("Consider wider stop-losses to account for slippage")

        if IssueType.STALE_DATA in issue_types:
            recommendations.append("Refresh data before making trading decisions")

        if IssueType.CORPORATE_ACTION in issue_types:
            recommendations.append("Verify if stock has been adjusted for splits/bonuses")
            recommendations.append("Consider using adjusted close prices")

        if IssueType.HIGH_VOLATILITY in issue_types:
            recommendations.append("Reduce position size due to high volatility")
            recommendations.append("Use wider stop-losses or shorter holding periods")

        return recommendations


def demo():
    """Demonstrate data health checking."""
    print("=" * 60)
    print("DataHealthChecker Demo")
    print("=" * 60)

    # Create sample good data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=300, freq='B')
    good_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 300),
        'High': np.random.uniform(105, 115, 300),
        'Low': np.random.uniform(95, 105, 300),
        'Close': np.random.uniform(100, 110, 300),
        'Volume': np.random.uniform(100000, 500000, 300)
    }, index=dates)

    # Ensure OHLC relationships are valid
    good_data['High'] = good_data[['Open', 'Close']].max(axis=1) + np.random.uniform(1, 5, 300)
    good_data['Low'] = good_data[['Open', 'Close']].min(axis=1) - np.random.uniform(1, 5, 300)

    checker = DataHealthChecker()

    print("\n--- Good Data ---")
    health = checker.check(good_data, "GOOD_STOCK")
    print(f"Status: {health.status.value}")
    print(f"Can Predict: {health.can_predict}")
    print(f"Confidence Penalty: {health.confidence_penalty:.1%}")
    print(f"Issues: {len(health.issues)}")

    # Create problematic data
    bad_data = good_data.copy()
    bad_data = bad_data.iloc[:50]  # Too few rows
    bad_data.loc[bad_data.index[10:15], 'Close'] = np.nan  # Missing data
    bad_data['Volume'] = bad_data['Volume'] / 100  # Low volume

    print("\n--- Bad Data ---")
    health = checker.check(bad_data, "BAD_STOCK")
    print(f"Status: {health.status.value}")
    print(f"Can Predict: {health.can_predict}")
    print(f"Confidence Penalty: {health.confidence_penalty:.1%}")
    print("Issues:")
    for issue in health.issues:
        print(f"  - [{issue.severity.value}] {issue.message}")
    print("Recommendations:")
    for rec in health.recommendations:
        print(f"  * {rec}")


if __name__ == "__main__":
    demo()
