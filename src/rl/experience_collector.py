"""
Experience Collector for RL Training

Collects and stores experiences from real trades for DQN training.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TradingState:
    """
    State representation for DQN.

    Matches the state space used in dqn_position_sizing.py.
    """
    # Price changes (normalized)
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    price_change_20d: float = 0.0

    # Technical indicators
    volatility: float = 0.0
    rsi: float = 0.5  # Normalized 0-1
    macd_signal: float = 0.0
    volume_ratio: float = 1.0

    # Regime (0-4)
    regime: int = 0

    # Position info
    current_position: float = 0.0  # -1 to +1
    unrealized_pnl: float = 0.0  # Normalized

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for DQN input."""
        return np.array([
            self.price_change_1d,
            self.price_change_5d,
            self.price_change_20d,
            self.volatility,
            self.rsi,
            self.macd_signal,
            self.volume_ratio,
            self.regime / 4.0,  # Normalize to 0-1
            self.current_position,
            self.unrealized_pnl,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradingState":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'price_change_1d': self.price_change_1d,
            'price_change_5d': self.price_change_5d,
            'price_change_20d': self.price_change_20d,
            'volatility': self.volatility,
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'volume_ratio': self.volume_ratio,
            'regime': self.regime,
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
        }


@dataclass
class RealTradeExperience:
    """Experience from actual trade execution."""
    trade_id: str
    symbol: str

    # State at entry
    entry_state: TradingState

    # Action taken
    action: int  # DQN action (0-6: HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100)
    position_size: float  # Actual position size (0-1)

    # Outcome
    pnl_pct: float
    holding_days: int
    hit_target: bool
    hit_stop: bool

    # Next state (market state at exit)
    exit_state: TradingState

    # Metadata
    entry_date: date
    exit_date: date
    regime: str
    signal_strength: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_state': self.entry_state.to_dict(),
            'action': self.action,
            'position_size': self.position_size,
            'pnl_pct': self.pnl_pct,
            'holding_days': self.holding_days,
            'hit_target': self.hit_target,
            'hit_stop': self.hit_stop,
            'exit_state': self.exit_state.to_dict(),
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'regime': self.regime,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RealTradeExperience":
        """Create from dictionary."""
        return cls(
            trade_id=d['trade_id'],
            symbol=d['symbol'],
            entry_state=TradingState.from_dict(d['entry_state']),
            action=d['action'],
            position_size=d['position_size'],
            pnl_pct=d['pnl_pct'],
            holding_days=d['holding_days'],
            hit_target=d['hit_target'],
            hit_stop=d['hit_stop'],
            exit_state=TradingState.from_dict(d['exit_state']),
            entry_date=date.fromisoformat(d['entry_date']),
            exit_date=date.fromisoformat(d['exit_date']),
            regime=d['regime'],
            signal_strength=d.get('signal_strength', ''),
            confidence=d.get('confidence', 0.0),
        )


class ExperienceCollector:
    """
    Collects and stores experiences from real trades for DQN training.

    Usage:
        collector = ExperienceCollector()

        # At trade entry
        collector.record_trade_entry(
            trade_id="T123",
            symbol="RELIANCE.NS",
            df=price_df,
            action=3,  # BUY_100
            position_size=1.0,
            entry_date=today
        )

        # At trade exit
        experience = collector.record_trade_exit(
            trade_id="T123",
            df=price_df,
            pnl_pct=2.5,
            holding_days=3,
            hit_target=True,
            hit_stop=False,
            exit_date=exit_day,
            regime="trending_bull"
        )

        # Get experiences for training
        experiences = collector.get_recent_experiences(n=100)
    """

    def __init__(self, storage_path: str = "data/rl/experiences.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.experiences: List[RealTradeExperience] = []
        self._pending_entries: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        """Load experiences from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self.experiences = [
                    RealTradeExperience.from_dict(d)
                    for d in data.get('experiences', [])
                ]
                self._pending_entries = data.get('pending', {})
                logger.info(f"Loaded {len(self.experiences)} experiences from storage")
            except Exception as e:
                logger.warning(f"Failed to load experiences: {e}")
                self.experiences = []
                self._pending_entries = {}

    def _save(self) -> None:
        """Save experiences to storage."""
        try:
            data = {
                'experiences': [e.to_dict() for e in self.experiences],
                'pending': self._pending_entries,
                'last_updated': datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")

    def extract_state(
        self,
        df: pd.DataFrame,
        position: float = 0.0,
        unrealized_pnl: float = 0.0
    ) -> TradingState:
        """
        Extract trading state from price DataFrame.

        Args:
            df: DataFrame with OHLCV and indicators
            position: Current position (-1 to +1)
            unrealized_pnl: Unrealized P&L (normalized)

        Returns:
            TradingState for DQN
        """
        if len(df) < 20:
            return TradingState()

        close = df['close'].values
        volume = df['volume'].values

        # Price changes
        price_change_1d = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0
        price_change_5d = (close[-1] - close[-6]) / close[-6] if len(close) > 5 and close[-6] > 0 else 0
        price_change_20d = (close[-1] - close[-21]) / close[-21] if len(close) > 20 and close[-21] > 0 else 0

        # Volatility (20-day)
        returns = np.diff(np.log(close[-21:]))
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2

        # RSI (if available, else calculate)
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1] / 100  # Normalize to 0-1
        else:
            rsi = 0.5

        # MACD signal
        if 'macd_signal' in df.columns:
            macd_signal = np.tanh(df['macd_signal'].iloc[-1] / close[-1] * 100)
        else:
            macd_signal = 0.0

        # Volume ratio
        avg_volume = volume[-21:-1].mean() if len(volume) > 20 else volume.mean()
        volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1.0

        # Regime (from HMM if available)
        regime = 0
        if 'regime' in df.columns:
            regime_map = {
                'trending_bull': 0,
                'trending_bear': 1,
                'ranging': 2,
                'choppy': 3,
                'transition': 4,
            }
            regime = regime_map.get(str(df['regime'].iloc[-1]).lower(), 0)

        return TradingState(
            price_change_1d=np.clip(price_change_1d, -0.2, 0.2),
            price_change_5d=np.clip(price_change_5d, -0.5, 0.5),
            price_change_20d=np.clip(price_change_20d, -1.0, 1.0),
            volatility=np.clip(volatility, 0, 1),
            rsi=np.clip(rsi, 0, 1),
            macd_signal=np.clip(macd_signal, -1, 1),
            volume_ratio=np.clip(volume_ratio, 0, 5),
            regime=regime,
            current_position=np.clip(position, -1, 1),
            unrealized_pnl=np.clip(unrealized_pnl, -1, 1),
        )

    def record_trade_entry(
        self,
        trade_id: str,
        symbol: str,
        df: pd.DataFrame,
        action: int,
        position_size: float,
        entry_date: date,
        signal_strength: str = "",
        confidence: float = 0.0
    ) -> None:
        """
        Record state and action at trade entry.

        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            df: Price DataFrame at entry time
            action: DQN action (0-6)
            position_size: Actual position size taken
            entry_date: Entry date
            signal_strength: Signal strength label
            confidence: Prediction confidence
        """
        entry_state = self.extract_state(df, position=0, unrealized_pnl=0)

        self._pending_entries[trade_id] = {
            'symbol': symbol,
            'entry_state': entry_state.to_dict(),
            'action': action,
            'position_size': position_size,
            'entry_date': entry_date.isoformat(),
            'signal_strength': signal_strength,
            'confidence': confidence,
        }

        self._save()
        logger.debug(f"Recorded trade entry: {trade_id}")

    def record_trade_exit(
        self,
        trade_id: str,
        df: pd.DataFrame,
        pnl_pct: float,
        holding_days: int,
        hit_target: bool,
        hit_stop: bool,
        exit_date: date,
        regime: str
    ) -> Optional[RealTradeExperience]:
        """
        Complete experience when trade exits.

        Args:
            trade_id: Trade identifier (must have recorded entry)
            df: Price DataFrame at exit time
            pnl_pct: Realized P&L percentage
            holding_days: Days held
            hit_target: Whether target was hit
            hit_stop: Whether stop was hit
            exit_date: Exit date
            regime: Market regime at exit

        Returns:
            Complete RealTradeExperience, or None if entry not found
        """
        if trade_id not in self._pending_entries:
            logger.warning(f"No entry found for trade {trade_id}")
            return None

        entry = self._pending_entries.pop(trade_id)

        # Calculate unrealized PnL for exit state (normalized)
        unrealized_pnl = np.clip(pnl_pct / 10, -1, 1)

        exit_state = self.extract_state(
            df,
            position=0,  # Position closed
            unrealized_pnl=unrealized_pnl
        )

        experience = RealTradeExperience(
            trade_id=trade_id,
            symbol=entry['symbol'],
            entry_state=TradingState.from_dict(entry['entry_state']),
            action=entry['action'],
            position_size=entry['position_size'],
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            hit_target=hit_target,
            hit_stop=hit_stop,
            exit_state=exit_state,
            entry_date=date.fromisoformat(entry['entry_date']),
            exit_date=exit_date,
            regime=regime,
            signal_strength=entry.get('signal_strength', ''),
            confidence=entry.get('confidence', 0.0),
        )

        self.experiences.append(experience)

        # Keep last 1000 experiences
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]

        self._save()
        logger.debug(f"Recorded trade exit: {trade_id}, PnL: {pnl_pct:.2f}%")

        return experience

    def get_recent_experiences(
        self,
        n: int = 100,
        regime: Optional[str] = None
    ) -> List[RealTradeExperience]:
        """
        Get recent experiences for training.

        Args:
            n: Maximum number of experiences
            regime: Optional filter by regime

        Returns:
            List of experiences
        """
        if regime:
            filtered = [e for e in self.experiences if e.regime == regime]
        else:
            filtered = self.experiences

        return filtered[-n:]

    def get_experiences_as_arrays(
        self,
        n: int = 100
    ) -> tuple:
        """
        Get experiences as numpy arrays for batch training.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        from src.rl.reward_shaping import RewardCalculator

        experiences = self.get_recent_experiences(n)

        if not experiences:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            )

        reward_calc = RewardCalculator()

        states = np.array([e.entry_state.to_array() for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([reward_calc.calculate(e) for e in experiences])
        next_states = np.array([e.exit_state.to_array() for e in experiences])
        dones = np.ones(len(experiences), dtype=np.float32)  # All are terminal

        return states, actions, rewards, next_states, dones

    def get_statistics(self) -> Dict:
        """Get statistics about collected experiences."""
        if not self.experiences:
            return {
                'total_experiences': 0,
                'pending_entries': len(self._pending_entries),
            }

        pnls = [e.pnl_pct for e in self.experiences]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

        return {
            'total_experiences': len(self.experiences),
            'pending_entries': len(self._pending_entries),
            'win_rate': win_rate,
            'avg_pnl': np.mean(pnls),
            'max_pnl': max(pnls),
            'min_pnl': min(pnls),
            'avg_holding_days': np.mean([e.holding_days for e in self.experiences]),
            'target_hit_rate': sum(1 for e in self.experiences if e.hit_target) / len(self.experiences),
            'stop_hit_rate': sum(1 for e in self.experiences if e.hit_stop) / len(self.experiences),
            'by_regime': {
                regime: len([e for e in self.experiences if e.regime == regime])
                for regime in set(e.regime for e in self.experiences)
            },
        }
