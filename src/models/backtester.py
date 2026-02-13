"""
Walk-forward backtesting to validate model performance.
CRITICAL: No lookahead bias - only use data available at prediction time.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from src.storage.models import SignalType, TradeType
from src.execution import (
    ExecutionConfig,
    AdvancedTradeExecutor,
    PartialPosition,
    TrailingStopCalculator,
)


@dataclass
class TradeResult:
    """Result of a single trade."""
    symbol: str
    trade_type: TradeType
    signal: SignalType
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    stop_loss: float
    target: float
    pnl_pct: float
    hit_target: bool
    hit_stop: bool
    holding_days: int
    confidence: float
    # Enhanced execution fields
    partial_exits: List[Tuple[date, float, float]] = None  # (date, price, pct_of_position)
    trailing_stop_activated: bool = False
    final_trailing_stop: float = 0.0
    regime: str = ""


@dataclass
class BacktestResults:
    """Aggregated backtest results."""
    start_date: date
    end_date: date
    trade_type: TradeType
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_holding_days: float
    trades: list[TradeResult]

    def to_dict(self) -> dict:
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'trade_type': self.trade_type.value,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'avg_return': round(self.avg_return, 4),
            'total_return': round(self.total_return, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'profit_factor': round(self.profit_factor, 4),
            'avg_holding_days': round(self.avg_holding_days, 2)
        }


class Backtester:
    """
    Walk-forward backtester for stock prediction signals.

    Simulates trading based on signals without lookahead bias.
    Supports advanced execution features: trailing stops, partial exits.
    """

    def __init__(
        self,
        slippage_pct: float = 0.1,
        commission_pct: float = 0.05,
        execution_config: Optional[ExecutionConfig] = None,
        use_advanced_execution: bool = False
    ):
        """
        Initialize backtester.

        Args:
            slippage_pct: Assumed slippage per trade (%)
            commission_pct: Trading commission (%)
            execution_config: Optional execution configuration for advanced features
            use_advanced_execution: Enable trailing stops and partial exits
        """
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.use_advanced_execution = use_advanced_execution
        self.execution_config = execution_config or ExecutionConfig()

        # Initialize advanced executor if enabled
        if self.use_advanced_execution:
            self.executor = AdvancedTradeExecutor(self.execution_config)
            self.trailing_calculator = TrailingStopCalculator()
            logger.info("Advanced execution enabled: trailing stops, partial exits")

    def simulate_trade(
        self,
        prices: pd.DataFrame,
        entry_date: date,
        signal: SignalType,
        trade_type: TradeType,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        symbol: str,
        regime: str = ""
    ) -> Optional[TradeResult]:
        """
        Simulate a single trade forward from entry date.

        Args:
            prices: DataFrame with OHLCV data
            entry_date: Signal generation date
            signal: BUY or SELL
            trade_type: INTRADAY or SWING
            entry_price: Suggested entry price
            stop_loss: Stop-loss level
            target: Target price
            confidence: Signal confidence
            symbol: Stock symbol
            regime: Market regime at entry

        Returns:
            TradeResult or None if cannot simulate
        """
        # Use advanced execution if enabled
        if self.use_advanced_execution:
            return self._simulate_trade_advanced(
                prices, entry_date, signal, trade_type,
                entry_price, stop_loss, target, confidence, symbol, regime
            )
        # Find entry point (next day's open)
        future_prices = prices[prices.index > pd.Timestamp(entry_date)]
        if future_prices.empty:
            return None

        # Entry at next day's open + slippage
        actual_entry_date = future_prices.index[0]
        if hasattr(actual_entry_date, 'date'):
            actual_entry_date = actual_entry_date.date()

        actual_entry = float(future_prices.iloc[0]['open'])
        if signal == SignalType.BUY:
            actual_entry *= (1 + self.slippage_pct / 100)
        else:
            actual_entry *= (1 - self.slippage_pct / 100)

        # Determine max holding period
        if trade_type == TradeType.INTRADAY:
            max_days = 1
        else:
            max_days = 5  # Swing trades

        # Simulate forward
        exit_price = None
        exit_date = None
        hit_target = False
        hit_stop = False

        for i, (idx, row) in enumerate(future_prices.iterrows()):
            if i >= max_days:
                # Time-based exit
                exit_price = float(row['close'])
                exit_date = idx.date() if hasattr(idx, 'date') else idx
                break

            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])

            if signal == SignalType.BUY:
                # Check stop-loss first (conservative)
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_stop = True
                    break
                # Check target
                if high >= target:
                    exit_price = target
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_target = True
                    break
            else:  # SELL signal
                # Check stop-loss (price going up is bad for shorts)
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_stop = True
                    break
                # Check target (price going down is good for shorts)
                if low <= target:
                    exit_price = target
                    exit_date = idx.date() if hasattr(idx, 'date') else idx
                    hit_target = True
                    break

        # If no exit yet, use last available close
        if exit_price is None:
            last_row = future_prices.iloc[min(max_days - 1, len(future_prices) - 1)]
            exit_price = float(last_row['close'])
            exit_date = future_prices.index[min(max_days - 1, len(future_prices) - 1)]
            if hasattr(exit_date, 'date'):
                exit_date = exit_date.date()

        # Apply exit slippage and commission
        if signal == SignalType.BUY:
            exit_price *= (1 - self.slippage_pct / 100)
            pnl_pct = ((exit_price - actual_entry) / actual_entry * 100) - (2 * self.commission_pct)
        else:
            exit_price *= (1 + self.slippage_pct / 100)
            pnl_pct = ((actual_entry - exit_price) / actual_entry * 100) - (2 * self.commission_pct)

        holding_days = (exit_date - actual_entry_date).days if isinstance(exit_date, date) and isinstance(actual_entry_date, date) else 1
        holding_days = max(1, holding_days)

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
            pnl_pct=round(pnl_pct, 4),
            hit_target=hit_target,
            hit_stop=hit_stop,
            holding_days=holding_days,
            confidence=confidence,
            partial_exits=None,
            trailing_stop_activated=False,
            final_trailing_stop=0.0,
            regime=regime
        )

    def _simulate_trade_advanced(
        self,
        prices: pd.DataFrame,
        entry_date: date,
        signal: SignalType,
        trade_type: TradeType,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        symbol: str,
        regime: str
    ) -> Optional[TradeResult]:
        """
        Advanced trade simulation with trailing stops and partial exits.

        Features:
        - ATR-based trailing stop (activates after 1R gain)
        - Partial profit booking (25% at 1R, 50% at target, 25% trails)
        """
        future_prices = prices[prices.index > pd.Timestamp(entry_date)]
        if future_prices.empty:
            return None

        # Entry at next day's open + slippage
        actual_entry_date = future_prices.index[0]
        if hasattr(actual_entry_date, 'date'):
            actual_entry_date = actual_entry_date.date()

        actual_entry = float(future_prices.iloc[0]['open'])
        if signal == SignalType.BUY:
            actual_entry *= (1 + self.slippage_pct / 100)
        else:
            actual_entry *= (1 - self.slippage_pct / 100)

        # Calculate 1R (risk per share)
        risk_per_share = abs(actual_entry - stop_loss)
        one_r_target = actual_entry + risk_per_share if signal == SignalType.BUY else actual_entry - risk_per_share

        # Determine max holding period
        if trade_type == TradeType.INTRADAY:
            max_days = 1
        else:
            max_days = 10  # Extended for swing with trailing

        # Tracking variables
        exit_price = None
        exit_date = None
        hit_target = False
        hit_stop = False
        partial_exits: List[Tuple[date, float, float]] = []
        remaining_position = 1.0  # Start with full position
        total_pnl = 0.0

        # Trailing stop state
        trailing_stop_activated = False
        current_trailing_stop = stop_loss
        highest_price = actual_entry if signal == SignalType.BUY else float('inf')
        lowest_price = actual_entry if signal == SignalType.SELL else 0

        # Get ATR for trailing stop calculation
        atr = float(prices['high'].tail(14).max() - prices['low'].tail(14).min()) / 14
        if 'atr' in prices.columns:
            atr = float(prices['atr'].iloc[-1])

        for i, (idx, row) in enumerate(future_prices.iterrows()):
            if i >= max_days or remaining_position <= 0:
                break

            bar_date = idx.date() if hasattr(idx, 'date') else idx
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])

            if signal == SignalType.BUY:
                # Update highest price for trailing
                highest_price = max(highest_price, high)

                # Check if trailing stop should activate (after 1R gain)
                if not trailing_stop_activated and high >= one_r_target:
                    trailing_stop_activated = True
                    # Book 25% at 1R if partial exits enabled
                    if self.execution_config.partial_exits:
                        partial_pnl = ((one_r_target - actual_entry) / actual_entry * 100) * 0.25
                        total_pnl += partial_pnl
                        partial_exits.append((bar_date, one_r_target, 0.25))
                        remaining_position -= 0.25

                # Update trailing stop if activated
                if trailing_stop_activated and self.execution_config.use_trailing_stop:
                    new_trailing = highest_price - (atr * self.execution_config.trailing_stop_atr_multiplier)
                    current_trailing_stop = max(current_trailing_stop, new_trailing)

                # Check stop (use trailing if activated)
                active_stop = current_trailing_stop if trailing_stop_activated else stop_loss
                if low <= active_stop:
                    exit_price = active_stop
                    exit_date = bar_date
                    hit_stop = not trailing_stop_activated  # Only true stop if trailing not active
                    partial_pnl = ((exit_price - actual_entry) / actual_entry * 100) * remaining_position
                    total_pnl += partial_pnl
                    remaining_position = 0
                    break

                # Check target
                if high >= target and remaining_position > 0:
                    exit_price = target
                    exit_date = bar_date
                    hit_target = True
                    # Book remaining position at target
                    if remaining_position > 0.25:
                        # Book 50% at target, let 25% trail
                        partial_pnl = ((target - actual_entry) / actual_entry * 100) * (remaining_position - 0.25)
                        total_pnl += partial_pnl
                        partial_exits.append((bar_date, target, remaining_position - 0.25))
                        remaining_position = 0.25
                    else:
                        partial_pnl = ((target - actual_entry) / actual_entry * 100) * remaining_position
                        total_pnl += partial_pnl
                        remaining_position = 0
                    if remaining_position == 0:
                        break

            else:  # SELL signal (short)
                lowest_price = min(lowest_price, low)

                if not trailing_stop_activated and low <= one_r_target:
                    trailing_stop_activated = True
                    if self.execution_config.partial_exits:
                        partial_pnl = ((actual_entry - one_r_target) / actual_entry * 100) * 0.25
                        total_pnl += partial_pnl
                        partial_exits.append((bar_date, one_r_target, 0.25))
                        remaining_position -= 0.25

                if trailing_stop_activated and self.execution_config.use_trailing_stop:
                    new_trailing = lowest_price + (atr * self.execution_config.trailing_stop_atr_multiplier)
                    current_trailing_stop = min(current_trailing_stop, new_trailing) if current_trailing_stop != stop_loss else new_trailing

                active_stop = current_trailing_stop if trailing_stop_activated else stop_loss
                if high >= active_stop:
                    exit_price = active_stop
                    exit_date = bar_date
                    hit_stop = not trailing_stop_activated
                    partial_pnl = ((actual_entry - exit_price) / actual_entry * 100) * remaining_position
                    total_pnl += partial_pnl
                    remaining_position = 0
                    break

                if low <= target and remaining_position > 0:
                    exit_price = target
                    exit_date = bar_date
                    hit_target = True
                    if remaining_position > 0.25:
                        partial_pnl = ((actual_entry - target) / actual_entry * 100) * (remaining_position - 0.25)
                        total_pnl += partial_pnl
                        partial_exits.append((bar_date, target, remaining_position - 0.25))
                        remaining_position = 0.25
                    else:
                        partial_pnl = ((actual_entry - target) / actual_entry * 100) * remaining_position
                        total_pnl += partial_pnl
                        remaining_position = 0
                    if remaining_position == 0:
                        break

        # Close any remaining position at last price
        if remaining_position > 0:
            last_idx = min(max_days - 1, len(future_prices) - 1)
            last_row = future_prices.iloc[last_idx]
            if exit_price is None:
                exit_price = float(last_row['close'])
                exit_date = future_prices.index[last_idx]
                if hasattr(exit_date, 'date'):
                    exit_date = exit_date.date()

            final_close = float(last_row['close'])
            if signal == SignalType.BUY:
                partial_pnl = ((final_close - actual_entry) / actual_entry * 100) * remaining_position
            else:
                partial_pnl = ((actual_entry - final_close) / actual_entry * 100) * remaining_position
            total_pnl += partial_pnl
            remaining_position = 0

        # Apply costs
        total_pnl -= (2 * self.commission_pct)  # Entry + exit commission

        holding_days = (exit_date - actual_entry_date).days if isinstance(exit_date, date) and isinstance(actual_entry_date, date) else 1
        holding_days = max(1, holding_days)

        return TradeResult(
            symbol=symbol,
            trade_type=trade_type,
            signal=signal,
            entry_date=actual_entry_date,
            entry_price=round(actual_entry, 2),
            exit_date=exit_date,
            exit_price=round(exit_price, 2) if exit_price else 0,
            stop_loss=stop_loss,
            target=target,
            pnl_pct=round(total_pnl, 4),
            hit_target=hit_target,
            hit_stop=hit_stop,
            holding_days=holding_days,
            confidence=confidence,
            partial_exits=partial_exits if partial_exits else None,
            trailing_stop_activated=trailing_stop_activated,
            final_trailing_stop=round(current_trailing_stop, 2),
            regime=regime
        )

    def run_backtest(
        self,
        signals: list[dict],
        price_data: dict[str, pd.DataFrame]
    ) -> BacktestResults:
        """
        Run backtest on a list of historical signals.

        Args:
            signals: List of signal dicts with keys:
                - symbol, date, signal, trade_type, entry_price, stop_loss, target, confidence
            price_data: Dict mapping symbol to DataFrame with prices

        Returns:
            BacktestResults with performance metrics
        """
        trades = []

        for sig in signals:
            symbol = sig['symbol']
            if symbol not in price_data:
                continue

            prices = price_data[symbol]

            result = self.simulate_trade(
                prices=prices,
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

        if not trades:
            return BacktestResults(
                start_date=date.today(),
                end_date=date.today(),
                trade_type=TradeType.INTRADAY,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_return=0,
                total_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_holding_days=0,
                trades=[]
            )

        # Calculate metrics
        returns = [t.pnl_pct for t in trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]

        win_rate = len(winning) / len(trades) if trades else 0
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns)

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown (simplified)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Average holding days
        avg_holding = np.mean([t.holding_days for t in trades])

        return BacktestResults(
            start_date=min(t.entry_date for t in trades),
            end_date=max(t.exit_date for t in trades),
            trade_type=trades[0].trade_type if trades else TradeType.INTRADAY,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_return=avg_return,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            trades=trades
        )

    def analyze_by_confidence(
        self,
        trades: list[TradeResult]
    ) -> dict:
        """
        Analyze performance by confidence buckets.

        Returns:
            Dict with performance by confidence level
        """
        buckets = {
            '0.60-0.70': [],
            '0.70-0.80': [],
            '0.80-0.90': [],
            '0.90-1.00': []
        }

        for t in trades:
            if t.confidence < 0.70:
                buckets['0.60-0.70'].append(t.pnl_pct)
            elif t.confidence < 0.80:
                buckets['0.70-0.80'].append(t.pnl_pct)
            elif t.confidence < 0.90:
                buckets['0.80-0.90'].append(t.pnl_pct)
            else:
                buckets['0.90-1.00'].append(t.pnl_pct)

        result = {}
        for bucket, returns in buckets.items():
            if returns:
                result[bucket] = {
                    'count': len(returns),
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'avg_return': np.mean(returns)
                }
            else:
                result[bucket] = {'count': 0, 'win_rate': 0, 'avg_return': 0}

        return result
