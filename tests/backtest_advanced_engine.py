"""
Comprehensive Backtesting Framework for Advanced Prediction Engine

Tests predictions against actual market outcomes using walk-forward validation.
Measures accuracy, identifies biases, and validates the 70%+ target.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.engines import AdvancedPredictionEngine


@dataclass
class PredictionResult:
    """Single prediction and its outcome."""
    symbol: str
    date: str
    predicted_probability: float
    predicted_signal: str
    predicted_regime: str
    actual_return_1d: float
    actual_return_3d: float
    actual_return_5d: float
    actual_direction_1d: int  # 1 = up, 0 = down
    actual_direction_3d: int
    actual_direction_5d: int
    correct_1d: bool
    correct_3d: bool
    correct_5d: bool


@dataclass
class BacktestStats:
    """Aggregated backtest statistics."""
    total_predictions: int
    accuracy_1d: float
    accuracy_3d: float
    accuracy_5d: float

    # By signal strength
    strong_buy_accuracy: float
    buy_accuracy: float
    neutral_accuracy: float
    sell_accuracy: float

    # By regime
    accuracy_by_regime: Dict[str, float]

    # Bias analysis
    bullish_bias: float  # % of bullish predictions
    avg_predicted_prob: float
    avg_actual_win_rate: float
    calibration_error: float

    # Returns analysis
    avg_return_when_buy: float
    avg_return_when_sell: float
    avg_return_when_neutral: float


class AdvancedBacktester:
    """
    Walk-forward backtester for the advanced prediction engine.

    Uses historical data to validate predictions.
    """

    # Test stocks - mix of sectors and volatility
    TEST_STOCKS = [
        # Large Cap - Stable
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        # Large Cap - Volatile
        'TATAMOTORS.NS', 'ADANIENT.NS', 'BAJFINANCE.NS', 'SBIN.NS', 'AXISBANK.NS',
        # Mid Cap
        'PIIND.NS', 'POLYCAB.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'MPHASIS.NS',
    ]

    def __init__(self, lookback_days: int = 100, prediction_horizon: int = 5):
        """
        Initialize backtester.

        Args:
            lookback_days: Days of history for each prediction
            prediction_horizon: Days to measure actual outcome
        """
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.engine = AdvancedPredictionEngine()
        self.results: List[PredictionResult] = []

    def fetch_data(self, symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """Fetch historical data for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if len(df) < self.lookback_days + self.prediction_horizon + 10:
                print(f"  Insufficient data for {symbol}")
                return None

            # Lowercase columns for engine compatibility
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return None

    def make_prediction(
        self,
        symbol: str,
        df: pd.DataFrame,
        end_idx: int
    ) -> Optional[Tuple[float, str, str]]:
        """
        Make a prediction using data up to end_idx.

        Returns (probability, signal, regime) or None if error.
        """
        try:
            # Get lookback data
            start_idx = max(0, end_idx - self.lookback_days)
            historical = df.iloc[start_idx:end_idx].copy()

            if len(historical) < 20:
                return None

            # Run prediction
            prediction = self.engine.predict(
                symbol=symbol.replace('.NS', ''),
                df=historical,
                timeframe='swing'
            )

            return (
                prediction.final_probability,
                prediction.signal.value,
                prediction.detected_regime
            )
        except Exception as e:
            return None

    def calculate_actual_outcome(
        self,
        df: pd.DataFrame,
        pred_idx: int
    ) -> Tuple[float, float, float, int, int, int]:
        """
        Calculate actual returns after prediction date.

        Returns (return_1d, return_3d, return_5d, dir_1d, dir_3d, dir_5d)
        """
        close = df['close']
        pred_price = close.iloc[pred_idx]

        # 1-day return
        if pred_idx + 1 < len(close):
            ret_1d = (close.iloc[pred_idx + 1] / pred_price - 1) * 100
        else:
            ret_1d = 0

        # 3-day return
        if pred_idx + 3 < len(close):
            ret_3d = (close.iloc[pred_idx + 3] / pred_price - 1) * 100
        else:
            ret_3d = ret_1d

        # 5-day return
        if pred_idx + 5 < len(close):
            ret_5d = (close.iloc[pred_idx + 5] / pred_price - 1) * 100
        else:
            ret_5d = ret_3d

        return (
            ret_1d, ret_3d, ret_5d,
            1 if ret_1d > 0 else 0,
            1 if ret_3d > 0 else 0,
            1 if ret_5d > 0 else 0
        )

    def is_prediction_correct(
        self,
        probability: float,
        actual_direction: int
    ) -> bool:
        """
        Check if prediction was correct.

        probability > 0.5 means bullish prediction
        actual_direction = 1 means price went up
        """
        predicted_up = probability > 0.5
        actual_up = actual_direction == 1
        return predicted_up == actual_up

    def run_backtest(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        step_days: int = 5  # Make prediction every N days
    ) -> List[PredictionResult]:
        """
        Run walk-forward backtest.

        Args:
            symbols: List of stock symbols (default: TEST_STOCKS)
            start_date: Start date for backtest
            end_date: End date for backtest
            step_days: Days between predictions

        Returns:
            List of prediction results
        """
        if symbols is None:
            symbols = self.TEST_STOCKS

        self.results = []

        print("=" * 70)
        print("ADVANCED PREDICTION ENGINE - BACKTEST")
        print("=" * 70)
        print(f"Stocks: {len(symbols)}")
        print(f"Lookback: {self.lookback_days} days")
        print(f"Prediction step: every {step_days} days")
        print("=" * 70)

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")

            df = self.fetch_data(symbol)
            if df is None:
                continue

            # Walk through the data
            n_predictions = 0
            for idx in range(self.lookback_days, len(df) - self.prediction_horizon, step_days):
                # Make prediction
                pred = self.make_prediction(symbol, df, idx)
                if pred is None:
                    continue

                prob, signal, regime = pred

                # Get actual outcome
                ret_1d, ret_3d, ret_5d, dir_1d, dir_3d, dir_5d = \
                    self.calculate_actual_outcome(df, idx)

                # Check correctness
                correct_1d = self.is_prediction_correct(prob, dir_1d)
                correct_3d = self.is_prediction_correct(prob, dir_3d)
                correct_5d = self.is_prediction_correct(prob, dir_5d)

                result = PredictionResult(
                    symbol=symbol,
                    date=str(df.index[idx].date()),
                    predicted_probability=prob,
                    predicted_signal=signal,
                    predicted_regime=regime,
                    actual_return_1d=ret_1d,
                    actual_return_3d=ret_3d,
                    actual_return_5d=ret_5d,
                    actual_direction_1d=dir_1d,
                    actual_direction_3d=dir_3d,
                    actual_direction_5d=dir_5d,
                    correct_1d=correct_1d,
                    correct_3d=correct_3d,
                    correct_5d=correct_5d
                )

                self.results.append(result)
                n_predictions += 1

            print(f"  Made {n_predictions} predictions")

        print(f"\nTotal predictions: {len(self.results)}")
        return self.results

    def calculate_statistics(self) -> BacktestStats:
        """Calculate comprehensive statistics from results."""
        if not self.results:
            raise ValueError("No results to analyze. Run backtest first.")

        n = len(self.results)

        # Overall accuracy
        acc_1d = sum(r.correct_1d for r in self.results) / n
        acc_3d = sum(r.correct_3d for r in self.results) / n
        acc_5d = sum(r.correct_5d for r in self.results) / n

        # By signal strength
        def accuracy_for_signal(signal_prefix: str, horizon: str = '3d') -> float:
            filtered = [r for r in self.results if r.predicted_signal.startswith(signal_prefix)]
            if not filtered:
                return 0
            attr = f'correct_{horizon}'
            return sum(getattr(r, attr) for r in filtered) / len(filtered)

        strong_buy_acc = accuracy_for_signal('STRONG_BUY')
        buy_acc = accuracy_for_signal('BUY')
        neutral_acc = accuracy_for_signal('NEUTRAL')
        sell_acc = accuracy_for_signal('SELL')

        # By regime
        regime_accuracy = defaultdict(list)
        for r in self.results:
            regime_accuracy[r.predicted_regime].append(r.correct_3d)

        acc_by_regime = {
            regime: sum(vals) / len(vals)
            for regime, vals in regime_accuracy.items() if vals
        }

        # Bias analysis
        bullish_preds = sum(1 for r in self.results if r.predicted_probability > 0.5)
        bullish_bias = bullish_preds / n

        avg_pred_prob = sum(r.predicted_probability for r in self.results) / n
        actual_ups = sum(r.actual_direction_3d for r in self.results)
        avg_actual_win = actual_ups / n

        calibration_error = abs(avg_pred_prob - avg_actual_win)

        # Returns by signal
        def avg_return_for_signal(signal_prefix: str) -> float:
            filtered = [r for r in self.results if r.predicted_signal.startswith(signal_prefix)]
            if not filtered:
                return 0
            return sum(r.actual_return_3d for r in filtered) / len(filtered)

        return BacktestStats(
            total_predictions=n,
            accuracy_1d=acc_1d,
            accuracy_3d=acc_3d,
            accuracy_5d=acc_5d,
            strong_buy_accuracy=strong_buy_acc,
            buy_accuracy=buy_acc,
            neutral_accuracy=neutral_acc,
            sell_accuracy=sell_acc,
            accuracy_by_regime=acc_by_regime,
            bullish_bias=bullish_bias,
            avg_predicted_prob=avg_pred_prob,
            avg_actual_win_rate=avg_actual_win,
            calibration_error=calibration_error,
            avg_return_when_buy=avg_return_for_signal('BUY'),
            avg_return_when_sell=avg_return_for_signal('SELL'),
            avg_return_when_neutral=avg_return_for_signal('NEUTRAL')
        )

    def print_detailed_report(self):
        """Print comprehensive backtest report."""
        stats = self.calculate_statistics()

        print("\n" + "=" * 70)
        print("BACKTEST RESULTS REPORT")
        print("=" * 70)

        print(f"\n## OVERALL ACCURACY")
        print(f"Total Predictions: {stats.total_predictions}")
        print(f"1-Day Accuracy:    {stats.accuracy_1d:.1%}")
        print(f"3-Day Accuracy:    {stats.accuracy_3d:.1%}")
        print(f"5-Day Accuracy:    {stats.accuracy_5d:.1%}")

        print(f"\n## ACCURACY BY SIGNAL")
        print(f"STRONG_BUY: {stats.strong_buy_accuracy:.1%}")
        print(f"BUY:        {stats.buy_accuracy:.1%}")
        print(f"NEUTRAL:    {stats.neutral_accuracy:.1%}")
        print(f"SELL:       {stats.sell_accuracy:.1%}")

        print(f"\n## ACCURACY BY REGIME")
        for regime, acc in sorted(stats.accuracy_by_regime.items()):
            print(f"{regime:15s}: {acc:.1%}")

        print(f"\n## BIAS ANALYSIS")
        print(f"Bullish Prediction Bias: {stats.bullish_bias:.1%}")
        print(f"Avg Predicted Probability: {stats.avg_predicted_prob:.1%}")
        print(f"Actual Win Rate: {stats.avg_actual_win_rate:.1%}")
        print(f"Calibration Error: {stats.calibration_error:.1%}")

        print(f"\n## RETURNS BY SIGNAL (3-day)")
        print(f"When BUY signal:     {stats.avg_return_when_buy:+.2f}%")
        print(f"When SELL signal:    {stats.avg_return_when_sell:+.2f}%")
        print(f"When NEUTRAL signal: {stats.avg_return_when_neutral:+.2f}%")

        # Assessment
        print(f"\n## ASSESSMENT")
        target = 0.70
        if stats.accuracy_3d >= target:
            print(f"[PASS] 3-day accuracy {stats.accuracy_3d:.1%} >= {target:.0%} target")
        else:
            gap = target - stats.accuracy_3d
            print(f"[FAIL] 3-day accuracy {stats.accuracy_3d:.1%} < {target:.0%} target (gap: {gap:.1%})")

        if stats.calibration_error < 0.05:
            print(f"[PASS] Calibration error {stats.calibration_error:.1%} < 5%")
        else:
            print(f"[WARN] Calibration error {stats.calibration_error:.1%} >= 5%")

        if 0.45 <= stats.bullish_bias <= 0.55:
            print(f"[PASS] No significant directional bias")
        else:
            direction = "bullish" if stats.bullish_bias > 0.55 else "bearish"
            print(f"[WARN] {direction.capitalize()} bias detected: {stats.bullish_bias:.1%}")

        print("=" * 70)

        return stats

    def analyze_failures(self, n_examples: int = 10):
        """Analyze incorrect predictions to identify patterns."""
        print("\n" + "=" * 70)
        print("FAILURE ANALYSIS")
        print("=" * 70)

        # Get high-confidence failures
        high_conf_failures = [
            r for r in self.results
            if not r.correct_3d and abs(r.predicted_probability - 0.5) > 0.1
        ]

        print(f"\nHigh-confidence failures: {len(high_conf_failures)}")

        if high_conf_failures:
            print(f"\nExamples of high-confidence wrong predictions:")
            for r in high_conf_failures[:n_examples]:
                direction = "UP" if r.predicted_probability > 0.5 else "DOWN"
                actual = "UP" if r.actual_direction_3d == 1 else "DOWN"
                print(f"  {r.symbol} {r.date}: Predicted {direction} ({r.predicted_probability:.1%}), "
                      f"Actual {actual} ({r.actual_return_3d:+.1f}%), Regime: {r.predicted_regime}")

        # Analyze by conditions
        print(f"\n## Failure patterns:")

        # After big moves
        big_move_failures = [
            r for r in self.results
            if not r.correct_3d and abs(r.actual_return_1d) > 2
        ]
        print(f"Failures after big 1-day moves (>2%): {len(big_move_failures)}")

        # In choppy regime
        choppy_failures = [
            r for r in self.results
            if not r.correct_3d and 'choppy' in r.predicted_regime.lower()
        ]
        print(f"Failures in choppy regime: {len(choppy_failures)}")


def run_quick_test():
    """Run a quick test with fewer stocks."""
    print("Running QUICK backtest (3 stocks, recent data)...")

    backtester = AdvancedBacktester(lookback_days=60, prediction_horizon=5)

    # Just 3 stocks for quick test
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'TATAMOTORS.NS']

    results = backtester.run_backtest(
        symbols=test_stocks,
        step_days=3  # More frequent predictions
    )

    stats = backtester.print_detailed_report()
    backtester.analyze_failures()

    return backtester, stats


def run_full_test():
    """Run comprehensive backtest."""
    print("Running FULL backtest (15 stocks, 2 years)...")

    backtester = AdvancedBacktester(lookback_days=100, prediction_horizon=5)

    results = backtester.run_backtest(step_days=5)

    stats = backtester.print_detailed_report()
    backtester.analyze_failures()

    return backtester, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full test')
    args = parser.parse_args()

    if args.full:
        run_full_test()
    else:
        run_quick_test()
