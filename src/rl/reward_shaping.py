"""
Reward Shaping for DQN Training

Provides configurable reward functions for RL training.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Base reward scaling
    pnl_scale: float = 10.0  # Scale PnL percentage

    # Target/Stop bonuses
    target_bonus: float = 0.5
    stop_penalty: float = -0.3

    # Time efficiency
    time_penalty_per_day: float = -0.02
    max_time_penalty: float = -0.20

    # Risk adjustment
    use_risk_adjustment: bool = True
    risk_scale: float = 1.5  # Larger positions get higher rewards/penalties

    # Position size awareness
    small_position_bonus: float = 0.1  # Bonus for conservative sizing
    large_position_penalty: float = -0.1

    # Win/loss asymmetry (optional)
    loss_multiplier: float = 1.5  # Penalize losses more than rewarding wins

    # Regime-specific adjustments
    bull_regime_bonus: float = 0.1
    bear_regime_penalty: float = -0.05
    choppy_regime_penalty: float = -0.1


class RewardCalculator:
    """
    Calculates reward signal for DQN training.

    Reward shaping:
    - Base: PnL percentage (scaled)
    - Bonus: +0.5 for hitting target
    - Penalty: -0.3 for hitting stop
    - Time penalty: -0.02 per day held (encourages efficiency)
    - Risk adjustment: scale by inverse of position size

    Usage:
        calc = RewardCalculator(RewardConfig())
        reward = calc.calculate(experience)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def calculate(self, experience) -> float:
        """
        Calculate reward for a trade experience.

        Args:
            experience: RealTradeExperience object

        Returns:
            Shaped reward value
        """
        # Base reward from P&L
        base_reward = experience.pnl_pct * self.config.pnl_scale / 100

        # Apply loss multiplier for asymmetry
        if base_reward < 0:
            base_reward *= self.config.loss_multiplier

        # Target/stop bonuses
        if experience.hit_target:
            base_reward += self.config.target_bonus
        elif experience.hit_stop:
            base_reward += self.config.stop_penalty

        # Time penalty
        time_penalty = self.config.time_penalty_per_day * experience.holding_days
        time_penalty = max(time_penalty, self.config.max_time_penalty)
        base_reward += time_penalty

        # Position size adjustment
        if self.config.use_risk_adjustment:
            if experience.position_size < 0.5:
                # Conservative sizing bonus
                base_reward += self.config.small_position_bonus
            elif experience.position_size > 0.8:
                # Aggressive sizing penalty
                base_reward += self.config.large_position_penalty

            # Scale reward by position size (risk-aware)
            risk_factor = 1.0 + (experience.position_size - 0.5) * self.config.risk_scale
            base_reward *= risk_factor

        # Regime adjustments
        regime = experience.regime.lower() if experience.regime else ''
        if 'bull' in regime:
            base_reward += self.config.bull_regime_bonus
        elif 'bear' in regime:
            base_reward += self.config.bear_regime_penalty
        elif 'choppy' in regime:
            base_reward += self.config.choppy_regime_penalty

        return float(np.clip(base_reward, -5, 5))

    def calculate_batch(self, experiences) -> np.ndarray:
        """Calculate rewards for a batch of experiences."""
        return np.array([self.calculate(e) for e in experiences])

    def explain_reward(self, experience) -> dict:
        """
        Explain reward breakdown for debugging.

        Returns dict with component breakdown.
        """
        components = {}

        # Base P&L
        base_pnl = experience.pnl_pct * self.config.pnl_scale / 100
        components['base_pnl'] = base_pnl

        if base_pnl < 0:
            components['loss_multiplier'] = base_pnl * (self.config.loss_multiplier - 1)

        # Target/stop
        if experience.hit_target:
            components['target_bonus'] = self.config.target_bonus
        elif experience.hit_stop:
            components['stop_penalty'] = self.config.stop_penalty

        # Time
        time_penalty = self.config.time_penalty_per_day * experience.holding_days
        time_penalty = max(time_penalty, self.config.max_time_penalty)
        components['time_penalty'] = time_penalty

        # Position size
        if experience.position_size < 0.5:
            components['size_adjustment'] = self.config.small_position_bonus
        elif experience.position_size > 0.8:
            components['size_adjustment'] = self.config.large_position_penalty

        # Regime
        regime = experience.regime.lower() if experience.regime else ''
        if 'bull' in regime:
            components['regime_adjustment'] = self.config.bull_regime_bonus
        elif 'bear' in regime:
            components['regime_adjustment'] = self.config.bear_regime_penalty
        elif 'choppy' in regime:
            components['regime_adjustment'] = self.config.choppy_regime_penalty

        components['total'] = self.calculate(experience)

        return components


def create_custom_reward_function(
    pnl_weight: float = 1.0,
    sharpe_weight: float = 0.0,
    drawdown_weight: float = 0.0
):
    """
    Factory for creating custom reward functions.

    Args:
        pnl_weight: Weight for P&L component
        sharpe_weight: Weight for Sharpe-like risk-adjusted return
        drawdown_weight: Weight for drawdown penalty

    Returns:
        Reward calculation function
    """
    def custom_reward(experience, historical_returns=None):
        reward = experience.pnl_pct * pnl_weight

        if sharpe_weight > 0 and historical_returns is not None:
            # Add Sharpe-like component
            if len(historical_returns) > 1:
                sharpe = np.mean(historical_returns) / (np.std(historical_returns) + 1e-6)
                reward += sharpe * sharpe_weight

        if drawdown_weight > 0:
            # Penalize if this trade contributed to drawdown
            if experience.pnl_pct < 0:
                reward += experience.pnl_pct * drawdown_weight

        return reward

    return custom_reward
