"""
Momentum Conservation Model

Based on Newton's First Law: An object in motion tends to stay in motion
unless acted upon by an external force.

Financial Interpretation:
- Mass (m) = Dollar Volume = Institutional weight in the stock
- Velocity (v) = Rate of price change
- Momentum (p) = m * v = Tendency to continue moving

Key Signals:
- High momentum + high persistence = Trend continuation
- High momentum + high friction = Potential reversal
- Low momentum = Wait for breakout
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MomentumResult:
    """Result from momentum conservation analysis."""
    momentum_magnitude: float      # Absolute momentum value
    momentum_direction: int        # +1 (bullish) or -1 (bearish)
    persistence_score: float       # How well momentum predicts future (0-1)
    friction_level: float          # Forces opposing momentum (0-2+)
    expected_continuation_days: int  # Expected days of continuation
    velocity: float                # Rate of price change
    mass: float                    # Dollar volume (institutional weight)
    probability_score: float       # Contribution to final prediction (0-1)
    signal: str                    # 'STRONG_MOMENTUM', 'MODERATE', 'WEAK', 'REVERSAL_WARNING'
    reason: str                    # Human-readable explanation


class MomentumConservationModel:
    """
    Implements momentum conservation from physics for stock prediction.

    The model calculates:
    1. Financial momentum (mass * velocity)
    2. Persistence (correlation of momentum with future returns)
    3. Friction (forces opposing momentum)
    4. Expected continuation
    """

    def __init__(
        self,
        velocity_period: int = 5,
        mass_period: int = 20,
        persistence_forward: int = 5,
        friction_period: int = 5
    ):
        self.velocity_period = velocity_period
        self.mass_period = mass_period
        self.persistence_forward = persistence_forward
        self.friction_period = friction_period

    def calculate_mass(self, df: pd.DataFrame) -> float:
        """
        Calculate financial mass = average dollar volume.
        Higher dollar volume = more institutional interest = more 'mass'.
        """
        dollar_volume = df['close'] * df['volume']
        mass = dollar_volume.rolling(self.mass_period).mean().iloc[-1]
        return mass

    def calculate_velocity(self, df: pd.DataFrame) -> float:
        """
        Calculate velocity = rate of price change.
        velocity = (price_t - price_{t-n}) / n
        """
        price_change = df['close'].iloc[-1] - df['close'].iloc[-self.velocity_period]
        velocity = price_change / self.velocity_period
        return velocity

    def calculate_momentum(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate momentum = mass * velocity.
        Returns (momentum, mass, velocity).
        """
        mass = self.calculate_mass(df)
        velocity = self.calculate_velocity(df)
        momentum = mass * velocity
        return momentum, mass, velocity

    def calculate_persistence(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum persistence score.

        Measures correlation between lagged momentum and forward returns.
        High persistence = momentum is a good predictor.
        """
        if len(df) < 60:
            return 0.5  # Not enough data

        returns = df['close'].pct_change()
        dollar_volume = df['close'] * df['volume']

        # Calculate rolling momentum
        mass_rolling = dollar_volume.rolling(self.mass_period).mean()
        velocity_rolling = df['close'].diff(self.velocity_period) / self.velocity_period
        momentum_rolling = mass_rolling * velocity_rolling

        # Forward returns
        forward_returns = returns.shift(-self.persistence_forward)

        # Correlation (only on valid data)
        valid_idx = ~(momentum_rolling.isna() | forward_returns.isna())
        if valid_idx.sum() < 30:
            return 0.5

        correlation = momentum_rolling[valid_idx].corr(forward_returns[valid_idx])

        # Handle NaN correlation
        if pd.isna(correlation):
            return 0.5

        # Convert to 0-1 score (correlation is -1 to 1)
        persistence = (correlation + 1) / 2
        return np.clip(persistence, 0, 1)

    def calculate_friction(self, df: pd.DataFrame) -> float:
        """
        Calculate friction coefficient.

        Friction = actual volatility / expected volatility based on momentum
        High friction = momentum is being opposed by market forces.
        """
        returns = df['close'].pct_change().dropna()

        # Actual volatility in recent period
        actual_vol = returns.iloc[-self.friction_period:].std()

        # Expected based on momentum (directional move)
        expected_move = abs(df['close'].iloc[-1] - df['close'].iloc[-self.friction_period])
        expected_vol = expected_move / (df['close'].iloc[-self.friction_period] * self.friction_period)

        if expected_vol < 1e-10:
            return 1.0  # No expected movement, neutral friction

        friction = actual_vol / expected_vol
        return np.clip(friction, 0, 3.0)  # Cap at 3.0

    def get_signal_label(
        self,
        momentum_normalized: float,
        persistence: float,
        friction: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if momentum_normalized > 0.7 and persistence > 0.6 and friction < 1.0:
            return 'STRONG_MOMENTUM', 'Strong bullish momentum with high persistence and low friction'
        elif momentum_normalized < -0.7 and persistence > 0.6 and friction < 1.0:
            return 'STRONG_MOMENTUM', 'Strong bearish momentum with high persistence and low friction'
        elif abs(momentum_normalized) > 0.5 and friction > 1.5:
            return 'REVERSAL_WARNING', 'High momentum but excessive friction - potential reversal'
        elif abs(momentum_normalized) > 0.3:
            return 'MODERATE', 'Moderate momentum - trend may continue'
        else:
            return 'WEAK', 'Weak momentum - no clear directional bias'

    def score(self, df: pd.DataFrame) -> MomentumResult:
        """
        Calculate comprehensive momentum conservation score.

        Args:
            df: DataFrame with OHLCV data (minimum 60 rows recommended)

        Returns:
            MomentumResult with all momentum metrics and probability score
        """
        if len(df) < 20:
            return self._default_result('Insufficient data for momentum analysis')

        # Calculate components
        momentum, mass, velocity = self.calculate_momentum(df)
        persistence = self.calculate_persistence(df)
        friction = self.calculate_friction(df)

        # Direction
        direction = 1 if momentum > 0 else -1

        # Normalize momentum for comparison
        avg_price = df['close'].mean()
        avg_volume = df['volume'].mean()
        momentum_normalized = momentum / (avg_price * avg_volume) if avg_volume > 0 else 0
        momentum_normalized = np.clip(momentum_normalized, -1, 1)

        # Expected continuation based on persistence and friction
        base_days = 10
        if persistence > 0.5 and friction < 1.0:
            expected_days = int(base_days * (persistence / 0.5) * (1 / (friction + 0.1)))
        else:
            expected_days = int(base_days * 0.5)
        expected_days = np.clip(expected_days, 1, 30)

        # Signal and reason
        signal, reason = self.get_signal_label(momentum_normalized, persistence, friction)

        # Probability score calculation
        # Components: momentum strength, persistence, inverse friction
        momentum_factor = min(abs(momentum_normalized), 1) * 0.3
        persistence_factor = persistence * 0.4
        friction_factor = max(0, 1 - friction / 2) * 0.3

        prob_score = 0.5 + (momentum_factor + persistence_factor + friction_factor - 0.5)

        # Adjust for direction alignment
        if direction == 1:  # Bullish
            prob_score = np.clip(prob_score, 0.3, 0.8)
        else:  # Bearish - invert for buy probability
            prob_score = 1 - np.clip(prob_score, 0.3, 0.8)

        return MomentumResult(
            momentum_magnitude=abs(momentum),
            momentum_direction=direction,
            persistence_score=persistence,
            friction_level=friction,
            expected_continuation_days=expected_days,
            velocity=velocity,
            mass=mass,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> MomentumResult:
        """Return neutral result when calculation isn't possible."""
        return MomentumResult(
            momentum_magnitude=0,
            momentum_direction=0,
            persistence_score=0.5,
            friction_level=1.0,
            expected_continuation_days=0,
            velocity=0,
            mass=0,
            probability_score=0.5,
            signal='WEAK',
            reason=reason
        )
