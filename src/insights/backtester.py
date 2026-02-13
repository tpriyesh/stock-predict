"""
Backtesting Engine - Test if our stock picks actually work.

This module:
1. Tests momentum strategy on historical data
2. Calculates actual win rate
3. Calibrates probability predictions
4. Tracks performance over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import yfinance as yf
import json
from pathlib import Path


@dataclass
class BacktestResult:
    """Result of a single stock backtest."""
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    return_pct: float
    holding_days: int
    won: bool
    signal_strength: float  # Our predicted score


@dataclass
class StrategyPerformance:
    """Overall strategy performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    avg_winner: float
    avg_loser: float
    max_winner: float
    max_loser: float
    profit_factor: float
    sharpe_ratio: float
    total_return: float


class BacktestEngine:
    """
    Backtest our stock picking strategy on historical data.

    This answers: "If we had used this strategy in the past,
    what would our win rate and returns have been?"
    """

    def __init__(self):
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def backtest_momentum_strategy(
        self,
        symbols: List[str],
        lookback_days: int = 252,  # 1 year of history
        holding_period: int = 5,   # Hold for 5 days
        momentum_threshold: float = 0.1,  # Min momentum score to buy
    ) -> StrategyPerformance:
        """
        Backtest the momentum strategy.

        Strategy: Buy stocks with momentum_score > threshold,
        hold for N days, then sell.
        """
        logger.info(f"Backtesting {len(symbols)} stocks over {lookback_days} days...")

        all_results = []

        for symbol in symbols:
            try:
                results = self._backtest_single_stock(
                    symbol, lookback_days, holding_period, momentum_threshold
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to backtest {symbol}: {e}")
                continue

        if not all_results:
            logger.warning("No backtest results generated")
            return None

        # Calculate performance metrics
        performance = self._calculate_performance(all_results)

        # Save results
        self._save_results(all_results, performance)

        return performance

    def _backtest_single_stock(
        self,
        symbol: str,
        lookback_days: int,
        holding_period: int,
        momentum_threshold: float
    ) -> List[BacktestResult]:
        """Backtest strategy on a single stock."""

        # Fetch historical data
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period='2y')  # Get 2 years for lookback

        if len(hist) < lookback_days + 50:
            return []

        results = []

        # Slide through history, simulating daily decisions
        for i in range(50, len(hist) - holding_period, holding_period):
            # Calculate momentum score at this point in time
            # (using only data available at that time - no lookahead!)

            slice_data = hist.iloc[:i+1]

            # Calculate returns at this point
            current_price = slice_data['Close'].iloc[-1]

            def safe_return(days):
                if len(slice_data) >= days:
                    past_price = slice_data['Close'].iloc[-days]
                    if past_price > 0:
                        return ((current_price / past_price) - 1) * 100
                return 0

            change_1w = safe_return(5)
            change_1m = safe_return(21)
            change_3m = safe_return(63)

            # Calculate momentum score (same formula as screener)
            momentum_score = (
                change_1w * 0.2 +
                change_1m * 0.3 +
                change_3m * 0.5
            ) / 100

            # Would we have bought this stock?
            if momentum_score >= momentum_threshold:
                entry_date = hist.index[i]
                entry_price = hist['Close'].iloc[i]

                exit_idx = min(i + holding_period, len(hist) - 1)
                exit_date = hist.index[exit_idx]
                exit_price = hist['Close'].iloc[exit_idx]

                return_pct = ((exit_price / entry_price) - 1) * 100

                results.append(BacktestResult(
                    symbol=symbol,
                    entry_date=str(entry_date.date()),
                    entry_price=round(entry_price, 2),
                    exit_date=str(exit_date.date()),
                    exit_price=round(exit_price, 2),
                    return_pct=round(return_pct, 2),
                    holding_days=holding_period,
                    won=return_pct > 0,
                    signal_strength=round(momentum_score, 3)
                ))

        return results

    def _calculate_performance(self, results: List[BacktestResult]) -> StrategyPerformance:
        """Calculate overall strategy performance."""

        returns = [r.return_pct for r in results]
        winners = [r.return_pct for r in results if r.won]
        losers = [r.return_pct for r in results if not r.won]

        total_trades = len(results)
        winning_trades = len(winners)
        losing_trades = len(losers)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        max_winner = max(winners) if winners else 0
        max_loser = min(losers) if losers else 0

        # Profit factor = gross profits / gross losses
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252/5) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Total return (compounded)
        total_return = np.prod([1 + r/100 for r in returns]) - 1
        total_return *= 100

        return StrategyPerformance(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 3),
            avg_return=round(avg_return, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            max_winner=round(max_winner, 2),
            max_loser=round(max_loser, 2),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            total_return=round(total_return, 2)
        )

    def _save_results(self, results: List[BacktestResult], performance: StrategyPerformance):
        """Save backtest results to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_data = [
            {
                'symbol': r.symbol,
                'entry_date': r.entry_date,
                'entry_price': float(r.entry_price),
                'exit_date': r.exit_date,
                'exit_price': float(r.exit_price),
                'return_pct': float(r.return_pct),
                'won': bool(r.won),
                'signal_strength': float(r.signal_strength)
            }
            for r in results
        ]

        results_file = self.results_dir / f"trades_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save performance summary
        perf_data = {
            'timestamp': timestamp,
            'total_trades': performance.total_trades,
            'win_rate': performance.win_rate,
            'avg_return': performance.avg_return,
            'profit_factor': performance.profit_factor,
            'sharpe_ratio': performance.sharpe_ratio,
            'total_return': performance.total_return
        }

        perf_file = self.results_dir / f"performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f, indent=2)

        logger.info(f"Saved backtest results to {results_file}")

    def analyze_by_signal_strength(self, results: List[BacktestResult]) -> Dict:
        """
        Analyze win rate by signal strength buckets.

        This tells us: "When our momentum score is 0.2, what's the actual win rate?"
        This is crucial for calibrating our predictions.
        """

        buckets = {
            '0.0-0.1': [],
            '0.1-0.2': [],
            '0.2-0.3': [],
            '0.3-0.5': [],
            '0.5+': []
        }

        for r in results:
            sig = r.signal_strength
            if sig < 0.1:
                buckets['0.0-0.1'].append(r)
            elif sig < 0.2:
                buckets['0.1-0.2'].append(r)
            elif sig < 0.3:
                buckets['0.2-0.3'].append(r)
            elif sig < 0.5:
                buckets['0.3-0.5'].append(r)
            else:
                buckets['0.5+'].append(r)

        analysis = {}
        for bucket_name, bucket_results in buckets.items():
            if bucket_results:
                wins = sum(1 for r in bucket_results if r.won)
                total = len(bucket_results)
                avg_ret = np.mean([r.return_pct for r in bucket_results])
                analysis[bucket_name] = {
                    'count': total,
                    'win_rate': round(wins / total, 3),
                    'avg_return': round(avg_ret, 2)
                }

        return analysis


def run_backtest_demo():
    """Run a demo backtest on popular stocks."""

    print("=" * 70)
    print("MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)

    # Test on a few stocks first
    test_symbols = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
        'SBIN', 'ITC', 'BHARTIARTL', 'WIPRO', 'TATASTEEL'
    ]

    engine = BacktestEngine()

    performance = engine.backtest_momentum_strategy(
        symbols=test_symbols,
        lookback_days=252,
        holding_period=5,
        momentum_threshold=0.1
    )

    if performance:
        print(f"\n### STRATEGY PERFORMANCE ###")
        print(f"Total Trades: {performance.total_trades}")
        print(f"Win Rate: {performance.win_rate:.1%}")
        print(f"Avg Return per Trade: {performance.avg_return:.2f}%")
        print(f"Avg Winner: +{performance.avg_winner:.2f}%")
        print(f"Avg Loser: {performance.avg_loser:.2f}%")
        print(f"Profit Factor: {performance.profit_factor:.2f}")
        print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
        print(f"Total Return: {performance.total_return:.2f}%")

        if performance.win_rate > 0.55:
            print("\n✅ Strategy has EDGE (win rate > 55%)")
        else:
            print("\n❌ Strategy needs improvement (win rate < 55%)")


if __name__ == "__main__":
    run_backtest_demo()
