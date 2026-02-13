"""
LogReturnsCalculator - Mathematically Correct Return Calculations

CRITICAL: Simple percentage returns are WRONG for statistical analysis.

Problem with pct_change():
    returns = df['Close'].pct_change()
    # 1. Asymmetric: +10% then -10% ≠ 0
    # 2. Non-additive: Multi-period return ≠ sum of daily returns
    # 3. Non-stationary: Harder for ML models

Solution: Log Returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    # 1. Symmetric: +10% and -10% have equal magnitude in log space
    # 2. Additive: Weekly return = sum of daily log returns
    # 3. More stationary: Closer to normal distribution

Mathematical Background:
    Simple return: R = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1
    Log return: r = ln(P_t/P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Relationship: r = ln(1 + R)
    For small R: r ≈ R (log return ≈ simple return)

Properties:
    1. Additivity: r_{t-n:t} = r_{t-n} + r_{t-n+1} + ... + r_t
    2. Symmetry: +10% = +0.0953, -10% = -0.1054 (similar magnitude)
    3. Normality: Log returns are closer to normal distribution
    4. Stationarity: Better for time series models
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class ReturnStats:
    """Statistics calculated from log returns."""
    mean_daily: float
    std_daily: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    win_rate: float
    n_observations: int


class LogReturnsCalculator:
    """
    Calculator for log returns and related statistics.

    All returns in this class are LOG RETURNS unless specified otherwise.

    Usage:
        calc = LogReturnsCalculator()

        # Basic log returns
        log_rets = calc.calculate(df['Close'])

        # Multi-period returns
        weekly_rets = calc.calculate_period(df['Close'], periods=5)

        # Full statistics
        stats = calc.get_statistics(df['Close'])
    """

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.05  # 5% annual (India 10Y bond)

    def __init__(self,
                 annualization_factor: int = 252,
                 risk_free_rate: float = 0.05):
        """
        Initialize calculator.

        Args:
            annualization_factor: Trading days per year (default 252)
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate

    def calculate(self,
                  prices: pd.Series,
                  fill_first: bool = True) -> pd.Series:
        """
        Calculate log returns from price series.

        Args:
            prices: Price series
            fill_first: If True, fill first NaN with 0

        Returns:
            Log returns series
        """
        # Handle edge cases
        if prices is None or len(prices) < 2:
            return pd.Series(dtype=float)

        # Ensure positive prices
        prices = prices.replace(0, np.nan)

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        if fill_first:
            log_returns = log_returns.fillna(0)

        return log_returns

    def calculate_period(self,
                         prices: pd.Series,
                         periods: int = 5) -> pd.Series:
        """
        Calculate log returns over multiple periods.

        For log returns, multi-period return = sum of daily returns.

        Args:
            prices: Price series
            periods: Number of periods

        Returns:
            Multi-period log returns
        """
        if len(prices) < periods + 1:
            return pd.Series(dtype=float)

        return np.log(prices / prices.shift(periods))

    def simple_to_log(self, simple_return: float) -> float:
        """Convert simple return to log return."""
        if simple_return <= -1:
            return -np.inf
        return np.log(1 + simple_return)

    def log_to_simple(self, log_return: float) -> float:
        """Convert log return to simple return."""
        return np.exp(log_return) - 1

    def cumulative_return(self, log_returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative return series.

        For log returns, cumulative = exp(sum of log returns).
        """
        return np.exp(log_returns.cumsum()) - 1

    def get_statistics(self,
                       prices: pd.Series,
                       rf_rate: Optional[float] = None) -> ReturnStats:
        """
        Calculate comprehensive statistics from price series.

        Args:
            prices: Price series
            rf_rate: Risk-free rate (default: instance value)

        Returns:
            ReturnStats object
        """
        rf_rate = rf_rate if rf_rate is not None else self.risk_free_rate

        log_rets = self.calculate(prices)

        if len(log_rets) < 2:
            return ReturnStats(
                mean_daily=0, std_daily=0,
                annualized_return=0, annualized_volatility=0,
                sharpe_ratio=0, skewness=0, kurtosis=0,
                max_drawdown=0, win_rate=0.5, n_observations=0
            )

        # Basic statistics
        mean_daily = log_rets.mean()
        std_daily = log_rets.std()

        # Annualized
        ann_return = mean_daily * self.annualization_factor
        ann_vol = std_daily * np.sqrt(self.annualization_factor)

        # Sharpe ratio
        if ann_vol > 0:
            sharpe = (ann_return - rf_rate) / ann_vol
        else:
            sharpe = 0

        # Higher moments
        skewness = float(log_rets.skew())
        kurtosis = float(log_rets.kurtosis())

        # Drawdown
        cum_returns = self.cumulative_return(log_rets)
        rolling_max = (1 + cum_returns).cummax()
        drawdown = (1 + cum_returns) / rolling_max - 1
        max_dd = abs(drawdown.min())

        # Win rate
        win_rate = (log_rets > 0).mean()

        return ReturnStats(
            mean_daily=float(mean_daily),
            std_daily=float(std_daily),
            annualized_return=float(ann_return),
            annualized_volatility=float(ann_vol),
            sharpe_ratio=float(sharpe),
            skewness=skewness,
            kurtosis=kurtosis,
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            n_observations=len(log_rets)
        )

    def rolling_volatility(self,
                           prices: pd.Series,
                           window: int = 20,
                           annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            prices: Price series
            window: Rolling window size
            annualize: If True, annualize the volatility

        Returns:
            Rolling volatility series
        """
        log_rets = self.calculate(prices)
        vol = log_rets.rolling(window).std()

        if annualize:
            vol = vol * np.sqrt(self.annualization_factor)

        return vol

    def realized_volatility(self,
                            prices: pd.Series,
                            period: int = 21) -> float:
        """
        Calculate realized volatility over a period.

        This is a backward-looking measure of actual volatility.
        """
        log_rets = self.calculate(prices).tail(period)

        if len(log_rets) < 2:
            return 0.0

        return float(log_rets.std() * np.sqrt(self.annualization_factor))

    def expected_shortfall(self,
                           prices: pd.Series,
                           confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (CVaR) - average loss in worst cases.

        More robust than VaR as it considers the magnitude of tail losses.

        Args:
            prices: Price series
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Expected shortfall (positive number = loss)
        """
        log_rets = self.calculate(prices)

        if len(log_rets) < 30:
            return 0.0

        var_threshold = log_rets.quantile(1 - confidence)
        tail_losses = log_rets[log_rets <= var_threshold]

        if len(tail_losses) == 0:
            return 0.0

        return -float(tail_losses.mean())

    def calculate_features(self,
                           prices: pd.Series,
                           include_advanced: bool = True) -> pd.DataFrame:
        """
        Calculate log-return based features for ML models.

        All features are stationary by construction (based on log returns).

        Args:
            prices: Price series
            include_advanced: Include higher-order features

        Returns:
            DataFrame with features
        """
        df = pd.DataFrame(index=prices.index)

        # Basic log returns at different horizons
        log_rets = self.calculate(prices)
        df['log_return_1d'] = log_rets
        df['log_return_5d'] = self.calculate_period(prices, 5)
        df['log_return_10d'] = self.calculate_period(prices, 10)
        df['log_return_20d'] = self.calculate_period(prices, 20)

        # Volatility features
        df['volatility_5d'] = log_rets.rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = log_rets.rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # Momentum features
        df['momentum_5d'] = log_rets.rolling(5).sum()
        df['momentum_20d'] = log_rets.rolling(20).sum()

        # Mean reversion features
        df['distance_from_mean_20d'] = log_rets - log_rets.rolling(20).mean()

        if include_advanced:
            # Higher moments
            df['skewness_20d'] = log_rets.rolling(20).skew()
            df['kurtosis_20d'] = log_rets.rolling(20).kurt()

            # Downside volatility
            negative_rets = log_rets.copy()
            negative_rets[negative_rets > 0] = 0
            df['downside_vol_20d'] = negative_rets.rolling(20).std() * np.sqrt(252)

            # Positive vs negative return asymmetry
            positive_days = (log_rets > 0).rolling(20).sum()
            df['positive_days_ratio'] = positive_days / 20

            # Consecutive direction
            df['direction'] = np.sign(log_rets)
            df['consecutive_direction'] = (
                df['direction']
                .groupby((df['direction'] != df['direction'].shift()).cumsum())
                .cumcount() + 1
            ) * df['direction']

        return df.dropna()


class VolatilityEstimator:
    """
    Advanced volatility estimation using log returns.

    Includes:
    - Simple historical volatility
    - Exponentially weighted volatility (EWMA)
    - Parkinson range-based volatility
    - Garman-Klass volatility
    - Yang-Zhang volatility
    """

    def __init__(self, annualization_factor: int = 252):
        self.ann_factor = annualization_factor

    def simple_volatility(self,
                          prices: pd.Series,
                          window: int = 20) -> pd.Series:
        """Simple historical volatility."""
        log_rets = np.log(prices / prices.shift(1))
        return log_rets.rolling(window).std() * np.sqrt(self.ann_factor)

    def ewma_volatility(self,
                        prices: pd.Series,
                        span: int = 20) -> pd.Series:
        """Exponentially weighted moving average volatility."""
        log_rets = np.log(prices / prices.shift(1))
        return log_rets.ewm(span=span).std() * np.sqrt(self.ann_factor)

    def parkinson_volatility(self,
                             high: pd.Series,
                             low: pd.Series,
                             window: int = 20) -> pd.Series:
        """
        Parkinson volatility using high-low range.

        More efficient than close-close volatility because
        it uses intraday information.
        """
        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))
        return np.sqrt(factor * log_hl.rolling(window).mean() * self.ann_factor)

    def garman_klass_volatility(self,
                                 open_: pd.Series,
                                 high: pd.Series,
                                 low: pd.Series,
                                 close: pd.Series,
                                 window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility using OHLC data.

        Most efficient estimator using open, high, low, close.
        """
        log_hl = (np.log(high / low)) ** 2
        log_co = (np.log(close / open_)) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window).mean() * self.ann_factor)

    def yang_zhang_volatility(self,
                               open_: pd.Series,
                               high: pd.Series,
                               low: pd.Series,
                               close: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Yang-Zhang volatility - handles overnight jumps.

        Combines overnight, open-close, and Rogers-Satchell volatility.
        """
        log_oc = np.log(open_ / close.shift(1))  # Overnight
        log_co = np.log(close / open_)           # Open-to-close
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)

        # Rogers-Satchell
        rs = log_ho * log_hc + log_lo * log_lc

        # Variance components
        var_overnight = log_oc.rolling(window).var()
        var_open_close = log_co.rolling(window).var()
        var_rs = rs.rolling(window).mean()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        yz_var = var_overnight + k * var_open_close + (1 - k) * var_rs

        return np.sqrt(yz_var * self.ann_factor)


def demo():
    """Demonstrate log returns calculations."""
    print("=" * 60)
    print("LogReturnsCalculator Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 252  # 1 year

    # Simulate price path with drift
    returns = np.random.normal(0.0003, 0.02, n)  # Small positive drift
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range(start='2024-01-01', periods=n, freq='B')
    prices = pd.Series(prices, index=dates)

    calc = LogReturnsCalculator()

    # Basic calculations
    log_rets = calc.calculate(prices)

    print("\n--- Basic Log Returns ---")
    print(f"Last 5 log returns: {log_rets.tail().values}")
    print(f"Mean daily log return: {log_rets.mean():.6f}")
    print(f"Daily volatility: {log_rets.std():.4f}")

    # Compare to simple returns
    simple_rets = prices.pct_change()
    print("\n--- Log vs Simple Returns ---")
    print(f"Log returns mean: {log_rets.mean():.6f}")
    print(f"Simple returns mean: {simple_rets.mean():.6f}")
    print(f"Difference: {(log_rets.mean() - simple_rets.mean()):.6f}")

    # Demonstrate additivity
    print("\n--- Additivity Property ---")
    weekly_log = calc.calculate_period(prices, 5).iloc[-1]
    sum_daily_log = log_rets.tail(5).sum()
    print(f"5-day log return: {weekly_log:.6f}")
    print(f"Sum of 5 daily log returns: {sum_daily_log:.6f}")
    print(f"Difference (should be ~0): {abs(weekly_log - sum_daily_log):.10f}")

    # Statistics
    print("\n--- Comprehensive Statistics ---")
    stats = calc.get_statistics(prices)
    print(f"Annualized Return: {stats.annualized_return:.1%}")
    print(f"Annualized Volatility: {stats.annualized_volatility:.1%}")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {stats.max_drawdown:.1%}")
    print(f"Win Rate: {stats.win_rate:.1%}")
    print(f"Skewness: {stats.skewness:.2f}")
    print(f"Kurtosis: {stats.kurtosis:.2f}")

    # Expected Shortfall
    es = calc.expected_shortfall(prices, confidence=0.95)
    print(f"\n95% Expected Shortfall: {es:.4f}")
    print(f"(Average loss in worst 5% of days)")


if __name__ == "__main__":
    demo()
