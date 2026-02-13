"""
Trade Tracker - Record predictions and actual outcomes for calibration.

This is CRITICAL for improving accuracy over time.
By tracking what we predicted vs what happened, we can:
1. Calibrate probability estimates
2. Identify which factors actually work
3. Improve the model over time
4. Train the DQN position sizing agent with real trade data
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import yfinance as yf

# RL Integration - Optional (gracefully handle if not available)
try:
    from src.rl import (
        ExperienceCollector,
        DQNTrainingScheduler,
        RewardConfig,
        TradingState,
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("RL module not available, DQN training disabled")

# Regime Calibration - Optional
try:
    from src.core.regime_calibration import (
        get_regime_calibrator,
        CalibrationRegime,
    )
    REGIME_CALIBRATION_AVAILABLE = True
except ImportError:
    REGIME_CALIBRATION_AVAILABLE = False
    logger.warning("Regime calibration not available")


class TradeStatus(Enum):
    PENDING = "PENDING"       # Trade recorded, waiting for outcome
    WON = "WON"               # Hit target
    LOST = "LOST"             # Hit stop loss
    EXPIRED = "EXPIRED"       # Time expired without hitting either
    CANCELLED = "CANCELLED"   # Cancelled before entry


@dataclass
class TrackedTrade:
    """A trade that we're tracking for calibration."""
    # Identification
    id: str
    symbol: str
    timestamp: str

    # Prediction
    predicted_probability: float  # Our win probability estimate
    predicted_return: float       # Expected return
    signal_strength: str          # STRONG_BUY, BUY, etc.
    timeframe: str                # INTRADAY, SWING, etc.

    # Levels
    entry_price: float
    stop_loss: float
    target_price: float

    # Factors used
    rsi_at_entry: float
    volume_ratio_at_entry: float
    trend_score: float
    factors_bullish: List[str]
    factors_bearish: List[str]

    # Outcome (filled later)
    status: str = "PENDING"
    actual_exit_price: float = 0.0
    actual_return: float = 0.0
    exit_timestamp: str = ""
    exit_reason: str = ""         # HIT_TARGET, HIT_STOP, TIME_EXPIRED

    # Calibration
    was_correct: bool = False

    # RL Integration fields
    regime: str = ""              # Market regime at entry
    position_size: float = 1.0    # Position size (0-1, for DQN)
    holding_days: int = 0         # Days held


class TradeTracker:
    """
    Track trades and calibrate probabilities.

    This is how we turn predictions into calibrated probabilities:
    1. Record every prediction with probability
    2. Track actual outcome
    3. Compare predicted vs actual
    4. Adjust future predictions based on calibration
    5. Feed completed trades to DQN for position sizing learning
    6. Update regime-specific calibration curves

    Example:
    - If we predict 60% win rate but actual is 50%
    - Calibration factor = 50/60 = 0.83
    - Future 60% predictions become 60% * 0.83 = 50%

    RL Integration:
    - Automatically records trade experiences for DQN training
    - Triggers DQN training based on configured schedule
    """

    def __init__(
        self,
        enable_rl: bool = True,
        enable_regime_calibration: bool = True,
        dqn_agent=None  # DQNPositionAgent instance (optional)
    ):
        self.data_dir = Path("data/trades")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.trades_file = self.data_dir / "tracked_trades.json"
        self.calibration_file = self.data_dir / "calibration.json"

        self.trades = self._load_trades()
        self.calibration = self._load_calibration()

        # RL Integration
        self.enable_rl = enable_rl and RL_AVAILABLE
        self.experience_collector: Optional[ExperienceCollector] = None
        self.dqn_trainer: Optional[DQNTrainingScheduler] = None

        if self.enable_rl:
            try:
                self.experience_collector = ExperienceCollector()
                if dqn_agent is not None:
                    self.dqn_trainer = DQNTrainingScheduler(
                        agent=dqn_agent,
                        experience_collector=self.experience_collector,
                        training_frequency="daily",
                        min_experiences=30
                    )
                logger.info("RL feedback loop enabled for trade tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize RL components: {e}")
                self.enable_rl = False

        # Regime Calibration
        self.enable_regime_calibration = enable_regime_calibration and REGIME_CALIBRATION_AVAILABLE
        self.regime_calibrator = None

        if self.enable_regime_calibration:
            try:
                self.regime_calibrator = get_regime_calibrator()
                logger.info("Regime-specific calibration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize regime calibrator: {e}")
                self.enable_regime_calibration = False

    def _load_trades(self) -> List[Dict]:
        """Load tracked trades."""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
        return []

    def _save_trades(self):
        """Save tracked trades."""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    def _load_calibration(self) -> Dict:
        """Load calibration data."""
        try:
            if self.calibration_file.exists():
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {
            'overall_factor': 1.0,
            'by_probability_bucket': {},
            'by_signal_strength': {},
            'by_timeframe': {},
            'last_updated': None
        }

    def _save_calibration(self):
        """Save calibration data."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def record_trade(
        self,
        symbol: str,
        predicted_probability: float,
        predicted_return: float,
        signal_strength: str,
        timeframe: str,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        rsi: float,
        volume_ratio: float,
        trend_score: float,
        factors_bullish: List[str],
        factors_bearish: List[str],
        regime: str = "",
        position_size: float = 1.0,
        df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Record a new trade prediction.

        Args:
            symbol: Stock symbol
            predicted_probability: Win probability (0-1)
            predicted_return: Expected return percentage
            signal_strength: Signal strength (STRONG_BUY, BUY, etc.)
            timeframe: Trade timeframe (INTRADAY, SWING)
            entry_price: Entry price
            stop_loss: Stop loss level
            target_price: Target price
            rsi: RSI at entry
            volume_ratio: Volume ratio at entry
            trend_score: Trend score (0-1)
            factors_bullish: List of bullish factors
            factors_bearish: List of bearish factors
            regime: Market regime (optional, for RL/calibration)
            position_size: Position size 0-1 (optional, for RL)
            df: Price DataFrame at entry (optional, for RL state extraction)

        Returns:
            Trade ID
        """
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade = {
            'id': trade_id,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'predicted_probability': predicted_probability,
            'predicted_return': predicted_return,
            'signal_strength': signal_strength,
            'timeframe': timeframe,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'rsi_at_entry': rsi,
            'volume_ratio_at_entry': volume_ratio,
            'trend_score': trend_score,
            'factors_bullish': factors_bullish,
            'factors_bearish': factors_bearish,
            'status': 'PENDING',
            'actual_exit_price': 0,
            'actual_return': 0,
            'exit_timestamp': '',
            'exit_reason': '',
            'was_correct': False,
            'regime': regime,
            'position_size': position_size,
            'holding_days': 0
        }

        self.trades.append(trade)
        self._save_trades()

        # Record to RL ExperienceCollector if enabled
        if self.enable_rl and self.experience_collector and df is not None:
            try:
                # Map signal_strength to DQN action (simplified mapping)
                action_map = {
                    'STRONG_BUY': 3,  # BUY_100
                    'BUY': 2,         # BUY_50
                    'HOLD': 0,        # HOLD
                }
                action = action_map.get(signal_strength, 2)

                self.experience_collector.record_trade_entry(
                    trade_id=trade_id,
                    symbol=symbol,
                    df=df,
                    action=action,
                    position_size=position_size,
                    entry_date=datetime.now().date(),
                    signal_strength=signal_strength,
                    confidence=predicted_probability
                )
                logger.debug(f"Recorded RL entry for trade {trade_id}")
            except Exception as e:
                logger.warning(f"Failed to record RL entry: {e}")

        logger.info(f"Recorded trade {trade_id}: {symbol} @ {entry_price}, prob={predicted_probability:.1%}")

        return trade_id

    def update_trade_outcome(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        df: Optional[pd.DataFrame] = None,
        holding_days: Optional[int] = None
    ):
        """
        Update a trade with its actual outcome.

        Args:
            trade_id: Trade ID to update
            exit_price: Actual exit price
            exit_reason: HIT_TARGET, HIT_STOP, or TIME_EXPIRED
            df: Price DataFrame at exit (optional, for RL state extraction)
            holding_days: Days held (optional, calculated if not provided)
        """
        for trade in self.trades:
            if trade['id'] == trade_id:
                trade['actual_exit_price'] = exit_price
                trade['exit_timestamp'] = datetime.now().isoformat()
                trade['exit_reason'] = exit_reason

                # Calculate actual return
                entry = trade['entry_price']
                actual_return = ((exit_price - entry) / entry) * 100
                trade['actual_return'] = round(actual_return, 2)

                # Calculate holding days if not provided
                if holding_days is None:
                    try:
                        entry_time = datetime.fromisoformat(trade['timestamp'])
                        holding_days = (datetime.now() - entry_time).days
                        holding_days = max(1, holding_days)
                    except:
                        holding_days = 1
                trade['holding_days'] = holding_days

                # Determine if won
                hit_target = exit_reason == 'HIT_TARGET'
                hit_stop = exit_reason == 'HIT_STOP'

                if hit_target:
                    trade['status'] = 'WON'
                    trade['was_correct'] = True
                elif hit_stop:
                    trade['status'] = 'LOST'
                    trade['was_correct'] = False
                else:
                    trade['status'] = 'EXPIRED'
                    trade['was_correct'] = actual_return > 0

                self._save_trades()
                logger.info(f"Updated trade {trade_id}: {trade['status']} with {actual_return:.2f}%")

                # Record to RL ExperienceCollector if enabled
                if self.enable_rl and self.experience_collector:
                    try:
                        regime = trade.get('regime', 'unknown')
                        if df is not None:
                            self.experience_collector.record_trade_exit(
                                trade_id=trade_id,
                                df=df,
                                pnl_pct=actual_return,
                                holding_days=holding_days,
                                hit_target=hit_target,
                                hit_stop=hit_stop,
                                exit_date=datetime.now().date(),
                                regime=regime
                            )
                            logger.debug(f"Recorded RL exit for trade {trade_id}")

                        # Trigger DQN training if scheduler available
                        if self.dqn_trainer and self.dqn_trainer.should_train(datetime.now().date()):
                            result = self.dqn_trainer.train_batch()
                            logger.info(f"DQN training triggered: loss={result.avg_loss:.4f}, reward={result.avg_reward:.3f}")
                    except Exception as e:
                        logger.warning(f"Failed to record RL exit or train: {e}")

                # Update regime-specific calibration if enabled
                if self.enable_regime_calibration and self.regime_calibrator:
                    try:
                        regime = trade.get('regime', 'unknown')
                        raw_score = trade['predicted_probability']
                        outcome = 1 if trade['was_correct'] else 0
                        self.regime_calibrator.record_outcome(
                            raw_score=raw_score,
                            actual_outcome=outcome,
                            regime=regime
                        )
                        logger.debug(f"Updated regime calibration for {regime}")
                    except Exception as e:
                        logger.warning(f"Failed to update regime calibration: {e}")

                # Recalibrate
                self.recalibrate()
                return

        logger.warning(f"Trade {trade_id} not found")

    def check_pending_trades(self):
        """
        Check all pending trades and update their status.
        Should be run periodically.
        """
        pending = [t for t in self.trades if t['status'] == 'PENDING']

        for trade in pending:
            try:
                symbol = trade['symbol']
                ticker = yf.Ticker(f"{symbol}.NS")

                # Get intraday data for same-day trades
                if trade['timeframe'] == 'INTRADAY':
                    hist = ticker.history(period='1d', interval='5m')
                else:
                    hist = ticker.history(period='5d')

                if hist.empty:
                    continue

                entry = trade['entry_price']
                stop = trade['stop_loss']
                target = trade['target_price']

                # Check if stop or target was hit
                high = hist['High'].max()
                low = hist['Low'].min()
                current = hist['Close'].iloc[-1]

                if high >= target:
                    self.update_trade_outcome(trade['id'], target, 'HIT_TARGET')
                elif low <= stop:
                    self.update_trade_outcome(trade['id'], stop, 'HIT_STOP')
                else:
                    # Check for time expiry
                    trade_time = datetime.fromisoformat(trade['timestamp'])
                    if trade['timeframe'] == 'INTRADAY':
                        # Expire same day
                        if datetime.now().date() > trade_time.date():
                            self.update_trade_outcome(trade['id'], current, 'TIME_EXPIRED')
                    elif trade['timeframe'] == 'SWING':
                        # Expire after 10 days
                        if datetime.now() > trade_time + timedelta(days=10):
                            self.update_trade_outcome(trade['id'], current, 'TIME_EXPIRED')

            except Exception as e:
                logger.warning(f"Failed to check trade {trade['id']}: {e}")

    def recalibrate(self):
        """
        Recalibrate probabilities based on actual outcomes.

        This is the key insight:
        - If predicted 60% win rate but actual is 50%
        - Calibration factor = 0.83
        - Apply to future predictions
        """
        completed = [t for t in self.trades if t['status'] in ['WON', 'LOST', 'EXPIRED']]

        if len(completed) < 10:
            logger.info("Not enough completed trades for calibration")
            return

        # Overall calibration
        total_predicted = sum(t['predicted_probability'] for t in completed)
        total_actual_wins = sum(1 for t in completed if t['was_correct'])

        avg_predicted = total_predicted / len(completed)
        actual_win_rate = total_actual_wins / len(completed)

        if avg_predicted > 0:
            overall_factor = actual_win_rate / avg_predicted
            overall_factor = max(0.7, min(1.3, overall_factor))  # Limit adjustment
        else:
            overall_factor = 1.0

        self.calibration['overall_factor'] = round(overall_factor, 3)

        # Calibration by probability bucket
        buckets = {
            '0.40-0.50': [],
            '0.50-0.60': [],
            '0.60-0.70': [],
            '0.70+': []
        }

        for t in completed:
            prob = t['predicted_probability']
            if prob < 0.50:
                buckets['0.40-0.50'].append(t)
            elif prob < 0.60:
                buckets['0.50-0.60'].append(t)
            elif prob < 0.70:
                buckets['0.60-0.70'].append(t)
            else:
                buckets['0.70+'].append(t)

        bucket_factors = {}
        for bucket, trades in buckets.items():
            if len(trades) >= 5:
                predicted = sum(t['predicted_probability'] for t in trades) / len(trades)
                actual = sum(1 for t in trades if t['was_correct']) / len(trades)
                if predicted > 0:
                    bucket_factors[bucket] = round(actual / predicted, 3)

        self.calibration['by_probability_bucket'] = bucket_factors

        # Calibration by signal strength
        by_signal = {}
        signals = set(t['signal_strength'] for t in completed)
        for signal in signals:
            signal_trades = [t for t in completed if t['signal_strength'] == signal]
            if len(signal_trades) >= 5:
                wins = sum(1 for t in signal_trades if t['was_correct'])
                by_signal[signal] = round(wins / len(signal_trades), 3)

        self.calibration['by_signal_strength'] = by_signal

        # Calibration by timeframe
        by_timeframe = {}
        timeframes = set(t['timeframe'] for t in completed)
        for tf in timeframes:
            tf_trades = [t for t in completed if t['timeframe'] == tf]
            if len(tf_trades) >= 5:
                wins = sum(1 for t in tf_trades if t['was_correct'])
                by_timeframe[tf] = round(wins / len(tf_trades), 3)

        self.calibration['by_timeframe'] = by_timeframe

        self.calibration['last_updated'] = datetime.now().isoformat()
        self._save_calibration()

        logger.info(f"Calibration updated: overall factor = {overall_factor:.3f}")

    def get_calibrated_probability(
        self,
        raw_probability: float,
        signal_strength: str,
        timeframe: str
    ) -> float:
        """
        Apply calibration to a raw probability estimate.
        """
        # Start with raw probability
        calibrated = raw_probability

        # Apply overall factor
        calibrated *= self.calibration.get('overall_factor', 1.0)

        # Apply bucket-specific factor
        if raw_probability < 0.50:
            bucket = '0.40-0.50'
        elif raw_probability < 0.60:
            bucket = '0.50-0.60'
        elif raw_probability < 0.70:
            bucket = '0.60-0.70'
        else:
            bucket = '0.70+'

        bucket_factors = self.calibration.get('by_probability_bucket', {})
        if bucket in bucket_factors:
            calibrated = raw_probability * bucket_factors[bucket]

        # Bound to reasonable range
        calibrated = max(0.35, min(0.75, calibrated))

        return round(calibrated, 3)

    def get_statistics(self) -> Dict:
        """Get comprehensive tracking statistics."""
        completed = [t for t in self.trades if t['status'] in ['WON', 'LOST', 'EXPIRED']]
        pending = [t for t in self.trades if t['status'] == 'PENDING']

        if not completed:
            return {
                'total_trades': len(self.trades),
                'completed': 0,
                'pending': len(pending),
                'win_rate': 0,
                'avg_predicted_prob': 0,
                'avg_return': 0,
                'calibration_factor': self.calibration.get('overall_factor', 1.0)
            }

        wins = [t for t in completed if t['was_correct']]
        losses = [t for t in completed if not t['was_correct']]

        returns = [t['actual_return'] for t in completed]
        winning_returns = [t['actual_return'] for t in wins]
        losing_returns = [t['actual_return'] for t in losses]

        return {
            'total_trades': len(self.trades),
            'completed': len(completed),
            'pending': len(pending),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(completed),
            'avg_predicted_prob': sum(t['predicted_probability'] for t in completed) / len(completed),
            'avg_return': np.mean(returns),
            'avg_winner': np.mean(winning_returns) if winning_returns else 0,
            'avg_loser': np.mean(losing_returns) if losing_returns else 0,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'profit_factor': abs(sum(winning_returns)) / abs(sum(losing_returns)) if losing_returns else 0,
            'calibration_factor': self.calibration.get('overall_factor', 1.0),
            'by_signal': self.calibration.get('by_signal_strength', {}),
            'by_timeframe': self.calibration.get('by_timeframe', {})
        }

    def print_report(self):
        """Print a readable performance report."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("TRADE TRACKING REPORT")
        print("=" * 60)

        print(f"\nTotal Trades: {stats['total_trades']}")
        print(f"Completed: {stats['completed']} | Pending: {stats['pending']}")

        if stats['completed'] > 0:
            print(f"\nWins: {stats['wins']} | Losses: {stats['losses']}")
            print(f"Win Rate: {stats['win_rate']:.1%}")
            print(f"Avg Predicted Prob: {stats['avg_predicted_prob']:.1%}")

            print(f"\nAvg Return: {stats['avg_return']:.2f}%")
            print(f"Avg Winner: +{stats['avg_winner']:.2f}%")
            print(f"Avg Loser: {stats['avg_loser']:.2f}%")
            print(f"Best Trade: +{stats['best_trade']:.2f}%")
            print(f"Worst Trade: {stats['worst_trade']:.2f}%")

            print(f"\nProfit Factor: {stats['profit_factor']:.2f}")
            print(f"Calibration Factor: {stats['calibration_factor']:.3f}")

            if stats['by_signal']:
                print("\nWin Rate by Signal:")
                for signal, rate in stats['by_signal'].items():
                    print(f"  {signal}: {rate:.1%}")

            if stats['by_timeframe']:
                print("\nWin Rate by Timeframe:")
                for tf, rate in stats['by_timeframe'].items():
                    print(f"  {tf}: {rate:.1%}")


def demo():
    """Demo the trade tracker."""
    print("Trade Tracker Demo")
    print("=" * 50)

    tracker = TradeTracker()

    # Record a sample trade
    trade_id = tracker.record_trade(
        symbol="RELIANCE",
        predicted_probability=0.62,
        predicted_return=1.5,
        signal_strength="BUY",
        timeframe="SWING",
        entry_price=2850.0,
        stop_loss=2800.0,
        target_price=2950.0,
        rsi=45.0,
        volume_ratio=1.3,
        trend_score=0.7,
        factors_bullish=["RSI neutral", "Above 50 MA"],
        factors_bearish=["High volatility"]
    )

    print(f"Recorded trade: {trade_id}")

    # Simulate outcome
    tracker.update_trade_outcome(trade_id, 2920.0, "HIT_TARGET")

    # Get calibrated probability
    raw = 0.60
    calibrated = tracker.get_calibrated_probability(raw, "BUY", "SWING")
    print(f"\nRaw probability: {raw:.1%}")
    print(f"Calibrated: {calibrated:.1%}")

    # Print report
    tracker.print_report()


if __name__ == "__main__":
    demo()
