"""
Fractal Dimension Model (Hurst Exponent)

Uses R/S analysis to calculate the Hurst exponent, which measures:
- Market memory and predictability
- Whether to use momentum or mean-reversion strategies

Key Concepts:
- H > 0.5: Trending/persistent (momentum works)
- H = 0.5: Random walk (unpredictable)
- H < 0.5: Mean-reverting/anti-persistent (contrarian works)

Trading Signals:
- H > 0.55: Use momentum strategies
- H < 0.45: Use mean-reversion strategies
- H ~ 0.5: Reduce position sizing
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class FractalResult:
    """Result from fractal dimension analysis."""
    hurst_exponent: float           # H: 0-1 (0.5 = random walk)
    fractal_dimension: float        # D = 2 - H
    market_character: str           # 'trending', 'random', 'mean_reverting'
    predictability: float           # 0-1 (how far from 0.5)
    optimal_strategy: str           # 'momentum', 'neutral', 'reversion'
    confidence_in_estimate: float   # R^2 of regression (0-1)
    lookback_used: int              # Days used for calculation
    probability_score: float        # Contribution to final prediction (0-1)
    signal: str                     # 'MOMENTUM_REGIME', 'RANDOM_REGIME', 'REVERSION_REGIME'
    reason: str                     # Human-readable explanation


class FractalDimensionModel:
    """
    Calculates Hurst exponent using R/S (Rescaled Range) analysis.

    The R/S method:
    1. Calculate cumulative deviations from mean
    2. R = Range of cumulative deviations
    3. S = Standard deviation
    4. For different time scales n: (R/S)_n ~ n^H
    5. H = slope of log(R/S) vs log(n)
    """

    def __init__(
        self,
        min_window: int = 10,
        max_window: int = 100,
        n_windows: int = 10
    ):
        self.min_window = min_window
        self.max_window = max_window
        self.n_windows = n_windows

    def calculate_rs(self, returns: np.ndarray) -> float:
        """
        Calculate R/S (Rescaled Range) for a return series.

        R = max(cumulative deviation) - min(cumulative deviation)
        S = standard deviation
        """
        n = len(returns)
        if n < 2:
            return 0

        mean_return = np.mean(returns)

        # Cumulative deviation from mean
        cumulative = np.cumsum(returns - mean_return)

        # Range
        R = np.max(cumulative) - np.min(cumulative)

        # Standard deviation
        S = np.std(returns, ddof=1) if n > 1 else 1e-10

        if S < 1e-10:
            return 0

        return R / S

    def calculate_hurst(self, prices: pd.Series) -> Tuple[float, float]:
        """
        Calculate Hurst exponent using R/S analysis.

        Returns (hurst_exponent, r_squared)
        """
        returns = prices.pct_change().dropna().values
        n = len(returns)

        if n < self.max_window:
            # Not enough data, use what we have
            max_window = n // 2
            if max_window < self.min_window:
                return 0.5, 0.0  # Default to random walk
        else:
            max_window = self.max_window

        # Calculate R/S for different window sizes
        window_sizes = np.linspace(
            self.min_window,
            max_window,
            self.n_windows,
            dtype=int
        )
        window_sizes = np.unique(window_sizes)

        log_sizes = []
        log_rs = []

        for window in window_sizes:
            if window > n:
                continue

            # Number of non-overlapping windows
            n_windows = n // window

            if n_windows < 1:
                continue

            rs_values = []
            for i in range(n_windows):
                start = i * window
                end = start + window
                segment = returns[start:end]

                rs = self.calculate_rs(segment)
                if rs > 0:
                    rs_values.append(rs)

            if rs_values:
                avg_rs = np.mean(rs_values)
                if avg_rs > 0:
                    log_sizes.append(np.log(window))
                    log_rs.append(np.log(avg_rs))

        if len(log_sizes) < 3:
            return 0.5, 0.0

        # Linear regression: log(R/S) = H * log(n) + c
        slope, intercept = np.polyfit(log_sizes, log_rs, 1)

        # R-squared
        y_pred = slope * np.array(log_sizes) + intercept
        ss_res = np.sum((np.array(log_rs) - y_pred) ** 2)
        ss_tot = np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Hurst exponent is the slope, bounded to [0, 1]
        hurst = np.clip(slope, 0, 1)

        return hurst, max(0, r_squared)

    def classify_market(self, hurst: float) -> str:
        """Classify market based on Hurst exponent."""
        if hurst > 0.55:
            return 'trending'
        elif hurst < 0.45:
            return 'mean_reverting'
        else:
            return 'random'

    def get_optimal_strategy(self, market_type: str) -> str:
        """Recommend strategy based on market type."""
        strategies = {
            'trending': 'momentum',
            'random': 'neutral',
            'mean_reverting': 'reversion'
        }
        return strategies.get(market_type, 'neutral')

    def get_signal_label(
        self,
        hurst: float,
        market_type: str,
        confidence: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if confidence < 0.5:
            return 'HURST_LOW_CONFIDENCE', f'Hurst={hurst:.2f} but low confidence (R²={confidence:.2f})'

        if market_type == 'trending':
            return 'MOMENTUM_REGIME', f'Hurst={hurst:.2f} indicates trending/persistent market'
        elif market_type == 'mean_reverting':
            return 'REVERSION_REGIME', f'Hurst={hurst:.2f} indicates mean-reverting market'
        else:
            return 'RANDOM_REGIME', f'Hurst={hurst:.2f} near random walk - low predictability'

    def score(self, df: pd.DataFrame) -> FractalResult:
        """
        Calculate comprehensive Hurst exponent analysis.

        Args:
            df: DataFrame with OHLCV data (minimum 100 days recommended)

        Returns:
            FractalResult with Hurst analysis
        """
        if len(df) < 50:
            return self._default_result('Insufficient data for Hurst analysis')

        prices = df['close']

        # Calculate Hurst exponent
        hurst, r_squared = self.calculate_hurst(prices)

        # Derived metrics
        fractal_dim = 2 - hurst
        market_type = self.classify_market(hurst)
        strategy = self.get_optimal_strategy(market_type)

        # Predictability = how far from 0.5 (normalized to 0-1)
        predictability = abs(hurst - 0.5) * 2

        # Signal and reason
        signal, reason = self.get_signal_label(hurst, market_type, r_squared)

        # Probability score
        # In trending markets, momentum works (higher prob for bullish if recent trend up)
        # In mean-reverting markets, reversion works (higher prob if oversold)
        # We can't determine direction from Hurst alone, but we can assess confidence

        if r_squared > 0.6:
            # High confidence in Hurst estimate
            if market_type == 'trending':
                # Check recent trend
                recent_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
                if recent_return > 0:
                    prob_score = 0.55 + predictability * 0.15
                else:
                    prob_score = 0.45 - predictability * 0.10
            elif market_type == 'mean_reverting':
                # Check if oversold (for reversion buy)
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                current = df['close'].iloc[-1]
                deviation = (current - sma_20) / sma_20

                if deviation < -0.03:  # Oversold
                    prob_score = 0.55 + predictability * 0.15
                elif deviation > 0.03:  # Overbought
                    prob_score = 0.45 - predictability * 0.10
                else:
                    prob_score = 0.50
            else:
                # Random walk
                prob_score = 0.50
        else:
            # Low confidence, stay neutral
            prob_score = 0.50

        prob_score = np.clip(prob_score, 0.35, 0.70)

        return FractalResult(
            hurst_exponent=hurst,
            fractal_dimension=fractal_dim,
            market_character=market_type,
            predictability=predictability,
            optimal_strategy=strategy,
            confidence_in_estimate=r_squared,
            lookback_used=len(df),
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> FractalResult:
        """Return neutral result when analysis isn't possible."""
        return FractalResult(
            hurst_exponent=0.5,
            fractal_dimension=1.5,
            market_character='random',
            predictability=0,
            optimal_strategy='neutral',
            confidence_in_estimate=0,
            lookback_used=0,
            probability_score=0.50,
            signal='HURST_UNKNOWN',
            reason=reason
        )
