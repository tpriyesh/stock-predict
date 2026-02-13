"""
Hidden Markov Model Regime Detector

Uses HMM to detect market regimes (hidden states) from observable features.

Regimes:
- TRENDING_BULL: Sustained upward movement, moderate volatility
- TRENDING_BEAR: Sustained downward movement, high volatility
- RANGING: Mean-reverting, low volatility
- CHOPPY: High volatility, no clear direction
- TRANSITION: Changing regimes (unstable)

Observable Features:
- Daily returns (normalized)
- Realized volatility
- Volume ratio
- RSI deviation from 50
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    CHOPPY = "choppy"
    TRANSITION = "transition"


@dataclass
class RegimeResult:
    """Result from HMM regime detection."""
    current_regime: MarketRegime
    regime_probability: float       # Probability of being in this regime
    regime_duration: int            # Days in current regime
    all_probabilities: Dict[str, float]  # Probability of each regime
    transition_probs: Dict[str, float]   # P(next regime)
    regime_stability: float         # How stable is current regime (0-1)
    expected_duration: float        # Expected remaining days in regime
    regime_history: List[str]       # Recent regime sequence
    observation_score: float        # How well observations fit model
    probability_score: float        # Contribution to prediction (0-1)
    signal: str                     # 'STABLE_BULL', 'REGIME_CHANGE', etc.
    reason: str                     # Human-readable explanation


class HMMRegimeDetector:
    """
    Simplified HMM-based regime detector.

    Uses Gaussian emissions for continuous observations.
    Implements Viterbi decoding for most likely regime sequence.

    Note: For production, consider using hmmlearn library.
    This implementation provides the core logic without external dependencies.
    """

    # Regime parameters (mean, std for each observation)
    # Observations: [return, volatility, volume_ratio, rsi_deviation]
    REGIME_PARAMS = {
        MarketRegime.TRENDING_BULL: {
            'return_mean': 0.08, 'return_std': 0.8,
            'vol_mean': 0.15, 'vol_std': 0.05,
            'volume_mean': 1.2, 'volume_std': 0.3,
            'rsi_mean': 0.2, 'rsi_std': 0.15
        },
        MarketRegime.TRENDING_BEAR: {
            'return_mean': -0.10, 'return_std': 1.2,
            'vol_mean': 0.25, 'vol_std': 0.08,
            'volume_mean': 1.4, 'volume_std': 0.4,
            'rsi_mean': -0.2, 'rsi_std': 0.15
        },
        MarketRegime.RANGING: {
            'return_mean': 0.0, 'return_std': 0.5,
            'vol_mean': 0.12, 'vol_std': 0.03,
            'volume_mean': 0.9, 'volume_std': 0.2,
            'rsi_mean': 0.0, 'rsi_std': 0.10
        },
        MarketRegime.CHOPPY: {
            'return_mean': 0.0, 'return_std': 1.5,
            'vol_mean': 0.30, 'vol_std': 0.10,
            'volume_mean': 1.3, 'volume_std': 0.5,
            'rsi_mean': 0.0, 'rsi_std': 0.25
        },
        MarketRegime.TRANSITION: {
            'return_mean': 0.0, 'return_std': 1.0,
            'vol_mean': 0.20, 'vol_std': 0.08,
            'volume_mean': 1.1, 'volume_std': 0.4,
            'rsi_mean': 0.0, 'rsi_std': 0.20
        }
    }

    # Transition matrix (calibrated from historical data)
    TRANSITION_MATRIX = {
        MarketRegime.TRENDING_BULL: {
            MarketRegime.TRENDING_BULL: 0.85,
            MarketRegime.TRENDING_BEAR: 0.02,
            MarketRegime.RANGING: 0.08,
            MarketRegime.CHOPPY: 0.03,
            MarketRegime.TRANSITION: 0.02
        },
        MarketRegime.TRENDING_BEAR: {
            MarketRegime.TRENDING_BULL: 0.03,
            MarketRegime.TRENDING_BEAR: 0.82,
            MarketRegime.RANGING: 0.05,
            MarketRegime.CHOPPY: 0.08,
            MarketRegime.TRANSITION: 0.02
        },
        MarketRegime.RANGING: {
            MarketRegime.TRENDING_BULL: 0.12,
            MarketRegime.TRENDING_BEAR: 0.08,
            MarketRegime.RANGING: 0.72,
            MarketRegime.CHOPPY: 0.05,
            MarketRegime.TRANSITION: 0.03
        },
        MarketRegime.CHOPPY: {
            MarketRegime.TRENDING_BULL: 0.08,
            MarketRegime.TRENDING_BEAR: 0.10,
            MarketRegime.RANGING: 0.10,
            MarketRegime.CHOPPY: 0.67,
            MarketRegime.TRANSITION: 0.05
        },
        MarketRegime.TRANSITION: {
            MarketRegime.TRENDING_BULL: 0.20,
            MarketRegime.TRENDING_BEAR: 0.20,
            MarketRegime.RANGING: 0.25,
            MarketRegime.CHOPPY: 0.20,
            MarketRegime.TRANSITION: 0.15
        }
    }

    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        rsi_period: int = 14
    ):
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.rsi_period = rsi_period
        self.regimes = list(MarketRegime)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract observable features for HMM.

        Features:
        1. Normalized return (z-score)
        2. Realized volatility (annualized)
        3. Volume ratio
        4. RSI deviation from 50 (normalized)
        """
        features = []

        # Returns
        returns = df['close'].pct_change()
        returns_mean = returns.mean()
        returns_std = returns.std()

        # Volatility
        volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)

        # Volume ratio
        volume = df['volume']
        volume_sma = volume.rolling(self.volume_window).mean()
        volume_ratio = volume / volume_sma

        # RSI
        rsi = self._calculate_rsi(df['close'], self.rsi_period)
        rsi_deviation = (rsi - 50) / 50  # Normalized to [-1, 1]

        # Combine features for each day
        for i in range(len(df)):
            if i < max(self.volatility_window, self.rsi_period):
                continue

            ret = returns.iloc[i]
            vol = volatility.iloc[i]
            vol_r = volume_ratio.iloc[i]
            rsi_d = rsi_deviation.iloc[i]

            # Normalize return
            ret_norm = (ret - returns_mean) / (returns_std + 1e-6)

            features.append([
                ret_norm,
                vol if not np.isnan(vol) else 0.15,
                vol_r if not np.isnan(vol_r) else 1.0,
                rsi_d if not np.isnan(rsi_d) else 0.0
            ])

        return np.array(features)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def emission_probability(
        self,
        observation: np.ndarray,
        regime: MarketRegime
    ) -> float:
        """
        Calculate P(observation | regime) using Gaussian emission.
        """
        params = self.REGIME_PARAMS[regime]

        # Feature order: [return, volatility, volume_ratio, rsi_deviation]
        log_prob = 0

        # Return
        log_prob += self._gaussian_log_prob(
            observation[0], params['return_mean'], params['return_std']
        )

        # Volatility
        log_prob += self._gaussian_log_prob(
            observation[1], params['vol_mean'], params['vol_std']
        )

        # Volume ratio
        log_prob += self._gaussian_log_prob(
            observation[2], params['volume_mean'], params['volume_std']
        )

        # RSI deviation
        log_prob += self._gaussian_log_prob(
            observation[3], params['rsi_mean'], params['rsi_std']
        )

        return np.exp(log_prob)

    def _gaussian_log_prob(self, x: float, mean: float, std: float) -> float:
        """Calculate log probability under Gaussian."""
        return -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((x - mean) / std)**2

    def viterbi_decode(
        self,
        observations: np.ndarray
    ) -> Tuple[List[MarketRegime], np.ndarray]:
        """
        Viterbi algorithm to find most likely regime sequence.

        Returns (best_path, state_probabilities)
        """
        n_obs = len(observations)
        n_states = len(self.regimes)

        # Initialize
        viterbi = np.zeros((n_obs, n_states))
        backpointer = np.zeros((n_obs, n_states), dtype=int)

        # Initial probabilities (uniform)
        initial_prob = 1.0 / n_states

        # First observation
        for j, regime in enumerate(self.regimes):
            emission = self.emission_probability(observations[0], regime)
            viterbi[0, j] = np.log(initial_prob + 1e-10) + np.log(emission + 1e-10)

        # Forward pass
        for t in range(1, n_obs):
            for j, regime_j in enumerate(self.regimes):
                emission = self.emission_probability(observations[t], regime_j)

                max_prob = -np.inf
                max_state = 0

                for i, regime_i in enumerate(self.regimes):
                    trans_prob = self.TRANSITION_MATRIX[regime_i][regime_j]
                    prob = viterbi[t-1, i] + np.log(trans_prob + 1e-10)

                    if prob > max_prob:
                        max_prob = prob
                        max_state = i

                viterbi[t, j] = max_prob + np.log(emission + 1e-10)
                backpointer[t, j] = max_state

        # Backtrack
        best_path = [0] * n_obs
        best_path[-1] = np.argmax(viterbi[-1])

        for t in range(n_obs - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # Convert to regimes
        regime_path = [self.regimes[i] for i in best_path]

        # Calculate state probabilities for last observation
        state_probs = np.exp(viterbi[-1] - np.max(viterbi[-1]))
        state_probs = state_probs / state_probs.sum()

        return regime_path, state_probs

    def calculate_regime_duration(
        self,
        regime_history: List[MarketRegime],
        current_regime: MarketRegime
    ) -> int:
        """Calculate how long we've been in current regime."""
        duration = 0
        for regime in reversed(regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break
        return duration

    def calculate_stability(
        self,
        regime_history: List[MarketRegime],
        lookback: int = 20
    ) -> float:
        """
        Calculate regime stability.

        Stability = fraction of recent days in most common regime
        """
        recent = regime_history[-lookback:] if len(regime_history) >= lookback else regime_history

        if not recent:
            return 0.5

        # Count occurrences
        counts = {}
        for r in recent:
            counts[r] = counts.get(r, 0) + 1

        # Stability = max count / total
        max_count = max(counts.values())
        stability = max_count / len(recent)

        return stability

    def get_signal_label(
        self,
        regime: MarketRegime,
        stability: float,
        duration: int
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if stability < 0.5:
            return 'REGIME_UNSTABLE', f'Regime unstable (stability: {stability:.0%})'

        if regime == MarketRegime.TRENDING_BULL:
            if duration > 10:
                return 'STABLE_BULL', f'Sustained bull trend ({duration} days)'
            else:
                return 'EARLY_BULL', f'New bull trend ({duration} days)'
        elif regime == MarketRegime.TRENDING_BEAR:
            if duration > 10:
                return 'STABLE_BEAR', f'Sustained bear trend ({duration} days)'
            else:
                return 'EARLY_BEAR', f'New bear trend ({duration} days)'
        elif regime == MarketRegime.RANGING:
            return 'RANGING', f'Range-bound market ({duration} days)'
        elif regime == MarketRegime.CHOPPY:
            return 'CHOPPY', f'Choppy/volatile conditions ({duration} days)'
        else:
            return 'TRANSITION', 'Market in transition - unclear direction'

    def detect_regime(self, df: pd.DataFrame) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            df: DataFrame with OHLCV data (minimum 60 days)

        Returns:
            RegimeResult with regime detection
        """
        if len(df) < 40:
            return self._default_result('Insufficient data for regime detection')

        # Extract features
        features = self.extract_features(df)

        if len(features) < 10:
            return self._default_result('Could not extract enough features')

        # Viterbi decode
        regime_path, state_probs = self.viterbi_decode(features)

        # Current regime
        current_regime = regime_path[-1]

        # All regime probabilities
        all_probs = {r.value: state_probs[i] for i, r in enumerate(self.regimes)}

        # Regime probability
        regime_prob = max(state_probs)

        # Duration in current regime
        duration = self.calculate_regime_duration(regime_path, current_regime)

        # Stability
        stability = self.calculate_stability(regime_path)

        # Transition probabilities
        trans_probs = {
            r.value: self.TRANSITION_MATRIX[current_regime][r]
            for r in self.regimes
        }

        # Expected duration (1 / (1 - self-transition))
        self_trans = self.TRANSITION_MATRIX[current_regime][current_regime]
        expected_duration = 1 / (1 - self_trans + 0.01)

        # Recent history
        regime_history = [r.value for r in regime_path[-10:]]

        # Observation score (how well last observation fits detected regime)
        if len(features) > 0:
            obs_prob = self.emission_probability(features[-1], current_regime)
            obs_score = min(obs_prob * 10, 1.0)  # Scale to 0-1
        else:
            obs_score = 0.5

        # Signal and reason
        signal, reason = self.get_signal_label(current_regime, stability, duration)

        # Probability score for BUY
        if current_regime == MarketRegime.TRENDING_BULL:
            prob_score = 0.60 + stability * 0.10
        elif current_regime == MarketRegime.TRENDING_BEAR:
            prob_score = 0.40 - stability * 0.10
        elif current_regime == MarketRegime.RANGING:
            # Neutral, slight mean reversion bias
            prob_score = 0.52
        elif current_regime == MarketRegime.CHOPPY:
            prob_score = 0.48
        else:  # Transition
            prob_score = 0.50

        prob_score = np.clip(prob_score, 0.30, 0.75)

        return RegimeResult(
            current_regime=current_regime,
            regime_probability=regime_prob,
            regime_duration=duration,
            all_probabilities=all_probs,
            transition_probs=trans_probs,
            regime_stability=stability,
            expected_duration=expected_duration,
            regime_history=regime_history,
            observation_score=obs_score,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> RegimeResult:
        """Return neutral result when detection isn't possible."""
        return RegimeResult(
            current_regime=MarketRegime.TRANSITION,
            regime_probability=0.5,
            regime_duration=0,
            all_probabilities={r.value: 0.2 for r in MarketRegime},
            transition_probs={r.value: 0.2 for r in MarketRegime},
            regime_stability=0.5,
            expected_duration=5,
            regime_history=[],
            observation_score=0.5,
            probability_score=0.50,
            signal='REGIME_UNKNOWN',
            reason=reason
        )
