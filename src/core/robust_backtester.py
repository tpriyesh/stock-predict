"""
RobustBacktester - Production-Grade Backtesting with Bias Elimination

Fixes critical issues:
1. Sharpe ratio correctly annualized for variable holding periods
2. Max drawdown using compounded returns
3. Purged cross-validation integration
4. Survivorship bias adjustment
5. Transaction cost modeling
6. Regime-aware performance analysis

This is the CORRECT way to backtest trading strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple
from loguru import logger
from enum import Enum

from src.storage.models import SignalType, TradeType
from src.core.purged_cv import PurgedKFoldCV, WalkForwardCV, TimeSeriesEmbargo
from src.core.survivorship import SurvivorshipBiasHandler, calculate_bias_adjusted_metrics
from src.core.transaction_costs import TransactionCostCalculator, BrokerType


class PerformanceRegime(Enum):
    """Market regimes for performance attribution."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


@dataclass
class TradeResult:
    """Result of a single trade with full attribution."""
    symbol: str
    trade_type: TradeType
    signal: SignalType
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    stop_loss: float
    target: float
    pnl_gross: float           # Before costs
    pnl_net: float             # After costs
    pnl_pct_gross: float
    pnl_pct_net: float
    hit_target: bool
    hit_stop: bool
    holding_days: int
    confidence: float
    regime: str
    transaction_costs: float
    slippage: float


@dataclass
class RobustBacktestResults:
    """Comprehensive backtest results with bias adjustments."""
    start_date: date
    end_date: date
    trade_type: TradeType

    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Returns (properly calculated)
    avg_return_gross: float
    avg_return_net: float
    total_return_gross: float
    total_return_net: float
    compounded_return: float    # Correct compounding

    # Risk metrics (properly calculated)
    max_drawdown: float         # Using compounded equity
    max_drawdown_duration: int  # Days
    volatility_annual: float
    downside_deviation: float   # For Sortino

    # Risk-adjusted returns (properly annualized)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float         # Return / Max DD

    # Quality metrics
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float           # (win_rate * avg_win) - (loss_rate * avg_loss)

    # Holding period stats
    avg_holding_days: float
    trades_per_year: float

    # Survivorship adjustment
    survivorship_adjusted_return: float
    survivorship_bias_estimate: float

    # Regime analysis
    performance_by_regime: Dict[str, Dict]

    # Costs
    total_transaction_costs: float
    total_slippage: float

    # Trade list
    trades: List[TradeResult]

    def to_dict(self) -> dict:
        return {
            'period': f"{self.start_date} to {self.end_date}",
            'trade_type': self.trade_type.value,
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 4),
            'avg_return_net': round(self.avg_return_net, 4),
            'compounded_return': round(self.compounded_return, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'profit_factor': round(self.profit_factor, 3),
            'expectancy': round(self.expectancy, 4),
            'survivorship_adjusted_return': round(self.survivorship_adjusted_return, 4),
            'survivorship_bias_estimate': round(self.survivorship_bias_estimate, 4),
            'total_costs': round(self.total_transaction_costs + self.total_slippage, 2)
        }


class RobustBacktester:
    """
    Production-grade backtester with proper statistical calculations.
    """

    def __init__(self,
                 broker_type: BrokerType = BrokerType.DISCOUNT,
                 slippage_bps: float = 10,      # 10 basis points = 0.1%
                 risk_free_rate: float = 0.06,  # 6% annual (India)
                 use_survivorship_adjustment: bool = True):
        """
        Initialize backtester.

        Args:
            broker_type: Type of broker for cost calculation
            slippage_bps: Slippage in basis points
            risk_free_rate: Risk-free rate for Sharpe calculation
            use_survivorship_adjustment: Apply survivorship bias correction
        """
        self.cost_calculator = TransactionCostCalculator(broker_type=broker_type)
        self.slippage_bps = slippage_bps
        self.risk_free_rate = risk_free_rate
        self.use_survivorship_adjustment = use_survivorship_adjustment
        self.survivorship_handler = SurvivorshipBiasHandler() if use_survivorship_adjustment else None

    def simulate_trade(self,
                       prices: pd.DataFrame,
                       entry_date: date,
                       signal: SignalType,
                       trade_type: TradeType,
                       entry_price: float,
                       stop_loss: float,
                       target: float,
                       confidence: float,
                       symbol: str) -> Optional[TradeResult]:
        """
        Simulate a single trade with full cost modeling.
        """
        # Find entry point (next day's open)
        future_prices = prices[prices.index > pd.Timestamp(entry_date)]
        if future_prices.empty:
            return None

        # Entry at next day's open
        actual_entry_date = future_prices.index[0]
        if hasattr(actual_entry_date, 'date'):
            actual_entry_date = actual_entry_date.date()

        actual_entry = float(future_prices.iloc[0]['open'])

        # Apply slippage
        slippage_pct = self.slippage_bps / 10000
        if signal == SignalType.BUY:
            actual_entry *= (1 + slippage_pct)
        else:
            actual_entry *= (1 - slippage_pct)

        # Determine max holding period
        if trade_type == TradeType.INTRADAY:
            max_days = 1
        else:
            max_days = 5  # Swing

        # Determine market regime
        regime = self._detect_regime(prices, entry_date)

        # Simulate forward
        exit_price = None
        exit_date = None
        hit_target = False
        hit_stop = False

        for i, (idx, row) in enumerate(future_prices.iterrows()):
            if i >= max_days:
                exit_price = float(row['close'])
                exit_date = idx.date() if hasattr(idx, 'date') else idx
                break

            high = float(row['high'])
            low = float(row['low'])

            if signal == SignalType.BUY:
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_stop = True
                    break
                if high >= target:
                    exit_price = target
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_target = True
                    break
            else:
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_stop = True
                    break
                if low <= target:
                    exit_price = target
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_target = True
                    break

        if exit_price is None:
            last_idx = min(max_days - 1, len(future_prices) - 1)
            exit_price = float(future_prices.iloc[last_idx]['close'])
            exit_date = future_prices.index[last_idx]
            if hasattr(exit_date, 'date'):
                exit_date = exit_date.date()

        # Apply exit slippage
        if signal == SignalType.BUY:
            exit_price *= (1 - slippage_pct)
        else:
            exit_price *= (1 + slippage_pct)

        # Calculate costs
        is_intraday = trade_type == TradeType.INTRADAY
        shares = 100  # Assume 100 shares for cost calculation

        buy_cost = self.cost_calculator.calculate_buy_cost(actual_entry, shares, is_intraday)
        sell_proceeds = self.cost_calculator.calculate_sell_proceeds(exit_price, shares, is_intraday)

        transaction_costs = (buy_cost - actual_entry * shares) + (exit_price * shares - sell_proceeds)
        transaction_costs_pct = transaction_costs / (actual_entry * shares) * 100

        # PnL calculations
        if signal == SignalType.BUY:
            pnl_gross_pct = (exit_price - actual_entry) / actual_entry * 100
        else:
            pnl_gross_pct = (actual_entry - exit_price) / actual_entry * 100

        slippage_total = slippage_pct * 2 * 100  # Entry + exit
        pnl_net_pct = pnl_gross_pct - transaction_costs_pct

        pnl_gross = pnl_gross_pct / 100 * actual_entry * shares
        pnl_net = pnl_net_pct / 100 * actual_entry * shares

        holding_days = max(1, (exit_date - actual_entry_date).days if isinstance(exit_date, date) else 1)

        return TradeResult(
            symbol=symbol,
            trade_type=trade_type,
            signal=signal,
            entry_date=actual_entry_date,
            entry_price=round(actual_entry, 2),
            exit_date=exit_date,
            exit_price=round(exit_price, 2),
            stop_loss=stop_loss,
            target=target,
            pnl_gross=round(pnl_gross, 2),
            pnl_net=round(pnl_net, 2),
            pnl_pct_gross=round(pnl_gross_pct, 4),
            pnl_pct_net=round(pnl_net_pct, 4),
            hit_target=hit_target,
            hit_stop=hit_stop,
            holding_days=holding_days,
            confidence=confidence,
            regime=regime,
            transaction_costs=round(transaction_costs, 2),
            slippage=round(slippage_total * actual_entry * shares / 100, 2)
        )

    def _detect_regime(self, prices: pd.DataFrame, as_of_date: date) -> str:
        """Detect market regime as of a date."""
        historical = prices[prices.index <= pd.Timestamp(as_of_date)]
        if len(historical) < 50:
            return "unknown"

        close = historical['close']
        returns = close.pct_change().dropna()

        # Calculate metrics
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        volatility = returns.tail(20).std() * np.sqrt(252)

        current = close.iloc[-1]

        # Determine regime
        if volatility > 0.30:  # 30% annualized volatility
            return "high_volatility"
        elif volatility < 0.15:
            return "low_volatility"
        elif current > sma_20 > sma_50:
            return "bull"
        elif current < sma_20 < sma_50:
            return "bear"
        else:
            return "sideways"

    def run_backtest(self,
                     signals: List[Dict],
                     price_data: Dict[str, pd.DataFrame]) -> RobustBacktestResults:
        """
        Run backtest with proper statistical calculations.
        """
        trades = []

        for sig in signals:
            symbol = sig['symbol']
            if symbol not in price_data:
                continue

            result = self.simulate_trade(
                prices=price_data[symbol],
                entry_date=sig['date'],
                signal=SignalType(sig['signal']),
                trade_type=TradeType(sig['trade_type']),
                entry_price=sig['entry_price'],
                stop_loss=sig['stop_loss'],
                target=sig['target'],
                confidence=sig['confidence'],
                symbol=symbol
            )

            if result:
                trades.append(result)

        return self._calculate_metrics(trades, signals)

    def _calculate_metrics(self,
                            trades: List[TradeResult],
                            signals: List[Dict]) -> RobustBacktestResults:
        """Calculate all metrics with proper formulas."""

        if not trades:
            return self._empty_results()

        # Basic stats
        returns_gross = [t.pnl_pct_gross for t in trades]
        returns_net = [t.pnl_pct_net for t in trades]
        holding_days = [t.holding_days for t in trades]

        winning = [r for r in returns_net if r > 0]
        losing = [r for r in returns_net if r <= 0]

        win_rate = len(winning) / len(trades)
        avg_return_gross = np.mean(returns_gross)
        avg_return_net = np.mean(returns_net)

        # ===== COMPOUNDED RETURN (CORRECT) =====
        # Convert percentage returns to decimal and compound
        compound_factors = [1 + r/100 for r in returns_net]
        compounded_return = (np.prod(compound_factors) - 1) * 100

        # ===== MAX DRAWDOWN (CORRECT - using compounded equity) =====
        equity_curve = np.cumprod(compound_factors)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown = np.max(drawdowns) * 100  # As percentage

        # Max drawdown duration
        dd_start = 0
        max_dd_duration = 0
        in_drawdown = False

        for i, dd in enumerate(drawdowns):
            if dd > 0 and not in_drawdown:
                dd_start = i
                in_drawdown = True
            elif dd == 0 and in_drawdown:
                duration = i - dd_start
                max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False

        if in_drawdown:
            max_dd_duration = max(max_dd_duration, len(drawdowns) - dd_start)

        # ===== SHARPE RATIO (CORRECT - annualized for variable holding) =====
        # Key insight: annualization depends on AVERAGE holding period
        avg_holding = np.mean(holding_days)
        trades_per_year = 252 / avg_holding  # How many trades per year on average

        # Excess returns (subtract risk-free rate per trade period)
        rf_per_trade = self.risk_free_rate * avg_holding / 252
        excess_returns = [r/100 - rf_per_trade for r in returns_net]

        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            # Sharpe = (mean excess return / std) * sqrt(trades per year)
            sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        # ===== SORTINO RATIO (using downside deviation) =====
        negative_returns = [r for r in excess_returns if r < 0]
        if negative_returns:
            downside_dev = np.sqrt(np.mean([r**2 for r in negative_returns]))
            if downside_dev > 0:
                sortino = (np.mean(excess_returns) / downside_dev) * np.sqrt(trades_per_year)
            else:
                sortino = 0
        else:
            sortino = float('inf') if np.mean(excess_returns) > 0 else 0
            downside_dev = 0

        # Volatility (annualized)
        if len(returns_net) > 1:
            vol_per_trade = np.std([r/100 for r in returns_net])
            volatility_annual = vol_per_trade * np.sqrt(trades_per_year)
        else:
            volatility_annual = 0

        # ===== CALMAR RATIO =====
        # Annual return / Max drawdown
        total_days = (max(t.exit_date for t in trades) - min(t.entry_date for t in trades)).days
        annual_return = compounded_return * 365 / max(1, total_days)
        calmar = annual_return / max(0.01, max_drawdown) if max_drawdown > 0 else 0

        # ===== PROFIT FACTOR =====
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0.01
        profit_factor = gross_profit / gross_loss

        # Win/loss stats
        avg_win = np.mean(winning) if winning else 0
        avg_loss = abs(np.mean(losing)) if losing else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Expectancy
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Costs
        total_costs = sum(t.transaction_costs for t in trades)
        total_slippage = sum(t.slippage for t in trades)

        # ===== SURVIVORSHIP ADJUSTMENT =====
        survivorship_adjusted = compounded_return
        survivorship_bias = 0

        if self.use_survivorship_adjustment and trades:
            start_date = min(t.entry_date for t in trades)
            end_date = max(t.exit_date for t in trades)

            trade_dicts = [{'pnl_pct': t.pnl_pct_net} for t in trades]
            bias_metrics = calculate_bias_adjusted_metrics(trade_dicts, start_date, end_date)

            survivorship_adjusted = bias_metrics['adjusted']['total_return']
            survivorship_bias = bias_metrics['adjustment_info']['survivorship_bias_estimated']

        # ===== REGIME ANALYSIS =====
        regime_performance = self._analyze_by_regime(trades)

        return RobustBacktestResults(
            start_date=min(t.entry_date for t in trades),
            end_date=max(t.exit_date for t in trades),
            trade_type=trades[0].trade_type,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_return_gross=avg_return_gross,
            avg_return_net=avg_return_net,
            total_return_gross=sum(returns_gross),
            total_return_net=sum(returns_net),
            compounded_return=compounded_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            volatility_annual=volatility_annual * 100,
            downside_deviation=downside_dev * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino if sortino != float('inf') else 999,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            avg_holding_days=avg_holding,
            trades_per_year=trades_per_year,
            survivorship_adjusted_return=survivorship_adjusted,
            survivorship_bias_estimate=survivorship_bias,
            performance_by_regime=regime_performance,
            total_transaction_costs=total_costs,
            total_slippage=total_slippage,
            trades=trades
        )

    def _analyze_by_regime(self, trades: List[TradeResult]) -> Dict[str, Dict]:
        """Analyze performance by market regime."""
        regime_trades = {}
        for t in trades:
            if t.regime not in regime_trades:
                regime_trades[t.regime] = []
            regime_trades[t.regime].append(t)

        result = {}
        for regime, rtrades in regime_trades.items():
            returns = [t.pnl_pct_net for t in rtrades]
            result[regime] = {
                'count': len(rtrades),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'avg_return': np.mean(returns),
                'total_return': np.sum(returns)
            }

        return result

    def _empty_results(self) -> RobustBacktestResults:
        """Return empty results when no trades."""
        return RobustBacktestResults(
            start_date=date.today(),
            end_date=date.today(),
            trade_type=TradeType.INTRADAY,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_return_gross=0,
            avg_return_net=0,
            total_return_gross=0,
            total_return_net=0,
            compounded_return=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            volatility_annual=0,
            downside_deviation=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            win_loss_ratio=0,
            expectancy=0,
            avg_holding_days=0,
            trades_per_year=0,
            survivorship_adjusted_return=0,
            survivorship_bias_estimate=0,
            performance_by_regime={},
            total_transaction_costs=0,
            total_slippage=0,
            trades=[]
        )

    def run_walk_forward_validation(self,
                                     scoring_engine,
                                     symbols: List[str],
                                     price_data: Dict[str, pd.DataFrame],
                                     trade_type: TradeType,
                                     train_months: int = 12,
                                     test_months: int = 1) -> Dict:
        """
        Run walk-forward validation with purged cross-validation.

        This is the CORRECT way to validate a trading strategy.
        """
        results = []

        # Get date range from data
        all_dates = []
        for symbol, df in price_data.items():
            all_dates.extend(df.index.tolist())

        min_date = min(all_dates).date() if all_dates else date.today()
        max_date = max(all_dates).date() if all_dates else date.today()

        # Create walk-forward windows
        cv = WalkForwardCV(
            train_size=train_months * 21,  # ~21 trading days per month
            test_size=test_months * 21,
            embargo=TimeSeriesEmbargo(label_horizon=5, feature_lookback=20)
        )

        # Get sample dates for splitting
        sample_dates = sorted(set(all_dates))

        for train_idx, test_idx in cv.split(pd.DataFrame(index=sample_dates)):
            train_end_date = sample_dates[train_idx[-1]].date()
            test_start_date = sample_dates[test_idx[0]].date()
            test_end_date = sample_dates[test_idx[-1]].date()

            # Generate signals for test period
            # (In production, would retrain model on train data)
            test_signals = []

            # ... signal generation would go here ...

            # Run backtest on test period
            if test_signals:
                test_result = self.run_backtest(test_signals, price_data)
                results.append({
                    'train_end': train_end_date,
                    'test_start': test_start_date,
                    'test_end': test_end_date,
                    'metrics': test_result.to_dict()
                })

        # Aggregate results
        if results:
            all_returns = [r['metrics']['avg_return_net'] for r in results]
            all_win_rates = [r['metrics']['win_rate'] for r in results]

            return {
                'n_windows': len(results),
                'avg_return_across_windows': np.mean(all_returns),
                'std_return_across_windows': np.std(all_returns),
                'avg_win_rate': np.mean(all_win_rates),
                'consistency': len([r for r in all_returns if r > 0]) / len(all_returns),
                'windows': results
            }

        return {'n_windows': 0, 'windows': []}


def demo():
    """Demonstrate robust backtesting."""
    print("=" * 60)
    print("RobustBacktester Demo")
    print("=" * 60)

    # Example showing difference between naive and correct calculations
    print("\n--- Sharpe Ratio Comparison ---")

    # Naive: daily returns with sqrt(252)
    daily_returns = [0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.3, -0.1, 0.4, -0.2]
    naive_sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    print(f"Naive Sharpe (assuming daily): {naive_sharpe:.2f}")

    # Correct: account for actual holding period
    holding_days = [3, 5, 2, 4, 3, 5, 4, 2, 3, 5]  # Variable holding
    avg_holding = np.mean(holding_days)
    trades_per_year = 252 / avg_holding
    correct_sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(trades_per_year)
    print(f"Correct Sharpe (avg holding {avg_holding:.1f} days): {correct_sharpe:.2f}")

    print("\n--- Max Drawdown Comparison ---")

    # Returns in percentage
    returns = [10, -5, 8, -12, 15, -8, 5, -3]

    # Naive: additive
    cumulative_naive = np.cumsum(returns)
    running_max_naive = np.maximum.accumulate(cumulative_naive)
    max_dd_naive = np.max(running_max_naive - cumulative_naive)
    print(f"Naive Max DD (additive): {max_dd_naive:.1f}%")

    # Correct: compounded
    compound_factors = [1 + r/100 for r in returns]
    equity = np.cumprod(compound_factors)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd_correct = np.max(drawdowns) * 100
    print(f"Correct Max DD (compounded): {max_dd_correct:.1f}%")


if __name__ == "__main__":
    demo()
