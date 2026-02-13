"""
MLE-Calibrated Markov Regime Model

Mathematical Foundation:
========================
Hidden Markov Model with Gaussian Emissions:

Emission Distributions (per regime):
   P(rₜ | Sₜ = k) = N(μₖ, σₖ²)

Transition Matrix Learning (Baum-Welch/EM Algorithm):
═════════════════════════════════════════════════════
E-Step:
   Forward: αₜ(i) = P(r₁...rₜ, Sₜ=i)
   Backward: βₜ(i) = P(rₜ₊₁...rₜ | Sₜ=i)

   γₜ(i) = P(Sₜ=i | r₁...rₜ) = αₜ(i)βₜ(i) / P(r₁...rₜ)
   ξₜ(i,j) = P(Sₜ=i, Sₜ₊₁=j | r₁...rₜ)

M-Step:
   aᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)     (transition probability)
   μₖ = Σₜ γₜ(k)rₜ / Σₜ γₜ(k)      (regime mean)
   σₖ² = Σₜ γₜ(k)(rₜ-μₖ)² / Σₜ γₜ(k)  (regime variance)

Regime Duration Distribution:
   P(duration = d | state = k) = (1-aₖₖ)^(d-1) × aₖₖ
   Expected Duration = 1/(1-aₖₖ)

References:
- Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series
- Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """Market regime states."""
    BULL_HIGH_VOL = "BULL_HIGH_VOL"      # Strong rally, high volatility
    BULL_LOW_VOL = "BULL_LOW_VOL"        # Steady uptrend, low volatility
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"      # Crash/panic
    BEAR_LOW_VOL = "BEAR_LOW_VOL"        # Grinding down
    SIDEWAYS = "SIDEWAYS"                 # Range-bound
    TRANSITION = "TRANSITION"             # Regime change in progress


@dataclass
class RegimeParameters:
    """Parameters for each regime."""
    mean_return: float  # Expected daily return
    volatility: float  # Daily volatility (std)
    skewness: float  # Return skewness
    kurtosis: float  # Excess kurtosis


@dataclass
class RegimeStatistics:
    """Statistics about regime transitions."""
    # Transition probabilities
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray

    # Regime parameters
    regime_params: Dict[int, RegimeParameters]

    # Duration statistics
    expected_durations: Dict[int, float]

    # Current state
    current_regime: MarketRegime
    current_probability: float
    regime_probabilities: Dict[MarketRegime, float]

    # Forecasts
    next_regime_probabilities: Dict[MarketRegime, float]
    regime_stability: float  # How stable is current regime

    # Model quality
    log_likelihood: float
    aic: float
    bic: float


@dataclass
class MLEResult:
    """Results from MLE estimation."""
    # Learned parameters
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    # State sequence
    state_sequence: np.ndarray  # Viterbi decoded
    state_probabilities: np.ndarray  # Smoothed probabilities

    # Convergence
    converged: bool
    n_iterations: int
    log_likelihood: float

    # Information criteria
    aic: float
    bic: float


class MLEMarkovRegime:
    """
    Maximum Likelihood Estimation for Markov Regime Model.

    Learns transition probabilities and emission parameters from data
    using the Baum-Welch (EM) algorithm.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        min_variance: float = 1e-6
    ):
        self.n_regimes = n_regimes
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_variance = min_variance

        # Model parameters (learned)
        self.transition_matrix = None
        self.means = None
        self.variances = None
        self.initial_probs = None

        # Regime mapping
        self.regime_names = [
            MarketRegime.BULL_LOW_VOL,
            MarketRegime.BULL_HIGH_VOL,
            MarketRegime.SIDEWAYS,
            MarketRegime.BEAR_LOW_VOL,
            MarketRegime.BEAR_HIGH_VOL
        ][:n_regimes]

    def fit(self, returns: np.ndarray) -> MLEResult:
        """
        Fit Markov regime model using Baum-Welch (EM) algorithm.

        Args:
            returns: Array of log returns

        Returns:
            MLEResult with learned parameters
        """
        n = len(returns)
        k = self.n_regimes

        if n < 50:
            logger.warning("Insufficient data for MLE. Using default parameters.")
            return self._default_result(returns)

        # Initialize parameters
        self._initialize_parameters(returns)

        # EM iterations
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iterations):
            # E-step: Forward-Backward algorithm
            alpha, beta, scaling = self._forward_backward(returns)

            # Compute gamma and xi
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(returns, alpha, beta, scaling)

            # M-step: Update parameters
            self._update_parameters(returns, gamma, xi)

            # Compute log-likelihood
            ll = np.sum(np.log(scaling + 1e-300))

            # Check convergence
            if abs(ll - prev_ll) < self.convergence_threshold:
                converged = True
                logger.info(f"Baum-Welch converged after {iteration + 1} iterations")
                break

            prev_ll = ll

        # Viterbi decoding for most likely state sequence
        state_sequence = self._viterbi(returns)

        # Information criteria
        n_params = k * (k - 1) + 2 * k  # Transition + means + variances
        aic = 2 * n_params - 2 * ll
        bic = np.log(n) * n_params - 2 * ll

        return MLEResult(
            transition_matrix=self.transition_matrix.copy(),
            means=self.means.copy(),
            variances=self.variances.copy(),
            state_sequence=state_sequence,
            state_probabilities=gamma,
            converged=converged,
            n_iterations=iteration + 1 if converged else self.max_iterations,
            log_likelihood=ll,
            aic=aic,
            bic=bic
        )

    def _initialize_parameters(self, returns: np.ndarray):
        """Initialize model parameters using k-means like approach."""
        k = self.n_regimes
        n = len(returns)

        # Sort returns to initialize means
        sorted_returns = np.sort(returns)
        quantiles = np.linspace(0, 1, k + 1)

        # Initialize means at quantile centers
        self.means = np.array([
            np.mean(sorted_returns[int(quantiles[i] * n):int(quantiles[i + 1] * n)])
            for i in range(k)
        ])

        # Sort means to ensure ordering (bull > sideways > bear)
        self.means = np.sort(self.means)[::-1]

        # Initialize variances based on data spread
        overall_var = np.var(returns)
        self.variances = np.array([
            overall_var * (0.5 + i * 0.3) for i in range(k)
        ])
        self.variances = np.maximum(self.variances, self.min_variance)

        # Initialize transition matrix (high persistence)
        self.transition_matrix = np.zeros((k, k))
        for i in range(k):
            self.transition_matrix[i, i] = 0.9  # 90% stay in same state
            for j in range(k):
                if i != j:
                    self.transition_matrix[i, j] = 0.1 / (k - 1)

        # Initial state probabilities
        self.initial_probs = np.ones(k) / k

    def _forward_backward(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward-Backward algorithm for HMM.

        Returns:
            (alpha, beta, scaling_factors)
        """
        n = len(returns)
        k = self.n_regimes

        alpha = np.zeros((n, k))
        beta = np.zeros((n, k))
        scaling = np.zeros(n)

        # Forward pass
        emission = self._emission_prob(returns[0])
        alpha[0] = self.initial_probs * emission
        scaling[0] = np.sum(alpha[0])
        alpha[0] /= scaling[0] + 1e-300

        for t in range(1, n):
            emission = self._emission_prob(returns[t])
            alpha[t] = emission * (alpha[t - 1] @ self.transition_matrix)
            scaling[t] = np.sum(alpha[t])
            alpha[t] /= scaling[t] + 1e-300

        # Backward pass
        beta[n - 1] = 1.0 / (scaling[n - 1] + 1e-300)

        for t in range(n - 2, -1, -1):
            emission = self._emission_prob(returns[t + 1])
            beta[t] = (self.transition_matrix @ (emission * beta[t + 1])) / (scaling[t] + 1e-300)

        return alpha, beta, scaling

    def _emission_prob(self, observation: float) -> np.ndarray:
        """Compute emission probability for each state (Gaussian)."""
        probs = np.zeros(self.n_regimes)
        for i in range(self.n_regimes):
            var = self.variances[i]
            mean = self.means[i]
            probs[i] = np.exp(-0.5 * (observation - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
        return probs + 1e-300

    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute state occupation probabilities γₜ(i)."""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300
        return gamma

    def _compute_xi(
        self,
        returns: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        scaling: np.ndarray
    ) -> np.ndarray:
        """Compute transition probabilities ξₜ(i,j)."""
        n = len(returns)
        k = self.n_regimes

        xi = np.zeros((n - 1, k, k))

        for t in range(n - 1):
            emission = self._emission_prob(returns[t + 1])
            for i in range(k):
                for j in range(k):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.transition_matrix[i, j] *
                        emission[j] *
                        beta[t + 1, j]
                    )

            xi[t] /= np.sum(xi[t]) + 1e-300

        return xi

    def _update_parameters(
        self,
        returns: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """M-step: Update model parameters."""
        n = len(returns)
        k = self.n_regimes

        # Update transition matrix
        for i in range(k):
            for j in range(k):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator

        # Normalize rows
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix /= row_sums + 1e-300

        # Update means
        for i in range(k):
            weight = gamma[:, i]
            weight_sum = np.sum(weight)
            if weight_sum > 0:
                self.means[i] = np.sum(weight * returns) / weight_sum

        # Update variances
        for i in range(k):
            weight = gamma[:, i]
            weight_sum = np.sum(weight)
            if weight_sum > 0:
                diff = returns - self.means[i]
                self.variances[i] = np.sum(weight * diff ** 2) / weight_sum
                self.variances[i] = max(self.variances[i], self.min_variance)

        # Update initial probabilities
        self.initial_probs = gamma[0]

    def _viterbi(self, returns: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for most likely state sequence.
        """
        n = len(returns)
        k = self.n_regimes

        # Log probabilities for numerical stability
        log_trans = np.log(self.transition_matrix + 1e-300)
        log_init = np.log(self.initial_probs + 1e-300)

        # Viterbi variables
        V = np.zeros((n, k))
        backpointer = np.zeros((n, k), dtype=int)

        # Initialize
        emission = self._emission_prob(returns[0])
        V[0] = log_init + np.log(emission + 1e-300)

        # Forward pass
        for t in range(1, n):
            emission = self._emission_prob(returns[t])
            for j in range(k):
                probs = V[t - 1] + log_trans[:, j]
                backpointer[t, j] = np.argmax(probs)
                V[t, j] = probs[backpointer[t, j]] + np.log(emission[j] + 1e-300)

        # Backtrack
        states = np.zeros(n, dtype=int)
        states[n - 1] = np.argmax(V[n - 1])

        for t in range(n - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def _default_result(self, returns: np.ndarray) -> MLEResult:
        """Return default result when data is insufficient."""
        k = self.n_regimes
        n = len(returns)

        # Default transition matrix (high persistence)
        trans = np.eye(k) * 0.9
        for i in range(k):
            for j in range(k):
                if i != j:
                    trans[i, j] = 0.1 / (k - 1)

        # Default parameters from data
        mean = np.mean(returns)
        std = np.std(returns)

        means = np.linspace(mean + 2 * std, mean - 2 * std, k)
        variances = np.full(k, std ** 2)

        return MLEResult(
            transition_matrix=trans,
            means=means,
            variances=variances,
            state_sequence=np.zeros(n, dtype=int),
            state_probabilities=np.ones((n, k)) / k,
            converged=False,
            n_iterations=0,
            log_likelihood=0.0,
            aic=0.0,
            bic=0.0
        )

    def get_regime_statistics(self, returns: np.ndarray) -> RegimeStatistics:
        """
        Get comprehensive regime statistics.
        """
        # Fit model if not already fitted
        result = self.fit(returns)

        # Compute stationary distribution
        stationary = self._compute_stationary_distribution()

        # Regime parameters
        regime_params = {}
        for i in range(self.n_regimes):
            regime_params[i] = RegimeParameters(
                mean_return=self.means[i],
                volatility=np.sqrt(self.variances[i]),
                skewness=0.0,  # Would need to estimate from data
                kurtosis=0.0
            )

        # Expected durations
        expected_durations = {}
        for i in range(self.n_regimes):
            persistence = self.transition_matrix[i, i]
            expected_durations[i] = 1 / (1 - persistence + 1e-10)

        # Current regime
        current_state = result.state_sequence[-1]
        current_regime = self.regime_names[current_state]
        current_prob = result.state_probabilities[-1, current_state]

        # Regime probabilities
        regime_probs = {
            self.regime_names[i]: result.state_probabilities[-1, i]
            for i in range(self.n_regimes)
        }

        # Next step regime probabilities
        next_probs_array = result.state_probabilities[-1] @ self.transition_matrix
        next_regime_probs = {
            self.regime_names[i]: next_probs_array[i]
            for i in range(self.n_regimes)
        }

        # Regime stability
        regime_stability = self.transition_matrix[current_state, current_state]

        return RegimeStatistics(
            transition_matrix=self.transition_matrix,
            stationary_distribution=stationary,
            regime_params=regime_params,
            expected_durations=expected_durations,
            current_regime=current_regime,
            current_probability=current_prob,
            regime_probabilities=regime_probs,
            next_regime_probabilities=next_regime_probs,
            regime_stability=regime_stability,
            log_likelihood=result.log_likelihood,
            aic=result.aic,
            bic=result.bic
        )

    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of Markov chain.

        Solves: π = π × P, Σπᵢ = 1
        """
        k = self.n_regimes

        # Method: Eigenvalue decomposition
        # Stationary distribution is the left eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize
        stationary = np.abs(stationary)
        stationary /= np.sum(stationary)

        return stationary

    def predict_regime_probability(
        self,
        current_probs: np.ndarray,
        horizon: int = 5
    ) -> np.ndarray:
        """
        Predict regime probabilities h steps ahead.

        Uses: P(Sₜ₊ₕ | Sₜ) = (transition_matrix)^h
        """
        trans_h = np.linalg.matrix_power(self.transition_matrix, horizon)
        return current_probs @ trans_h

    def get_regime_trading_signal(
        self,
        returns: np.ndarray
    ) -> Tuple[str, float, float]:
        """
        Generate trading signal from regime analysis.

        Returns:
            (signal, probability, confidence)
        """
        stats = self.get_regime_statistics(returns)

        current = stats.current_regime
        stability = stats.regime_stability
        prob = stats.current_probability

        # Bullish regimes
        if current in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]:
            base_prob = 0.55 + 0.15 * stability
            signal = 'BUY' if base_prob >= 0.55 else 'HOLD'

        # Bearish regimes
        elif current in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
            base_prob = 0.45 - 0.15 * stability
            signal = 'SELL' if base_prob <= 0.45 else 'HOLD'

        # Sideways
        elif current == MarketRegime.SIDEWAYS:
            base_prob = 0.50
            signal = 'HOLD'

        # Transition
        else:
            base_prob = 0.50
            signal = 'HOLD'

        # Confidence based on state probability and stability
        confidence = prob * stability

        return signal, base_prob, confidence


def compute_regime_adjusted_probability(
    base_probability: float,
    returns: np.ndarray,
    n_regimes: int = 5
) -> Tuple[float, MarketRegime, float]:
    """
    Adjust prediction probability based on detected regime.

    Returns:
        (adjusted_probability, current_regime, confidence)
    """
    if len(returns) < 30:
        return base_probability, MarketRegime.SIDEWAYS, 0.5

    model = MLEMarkovRegime(n_regimes=n_regimes)
    stats = model.get_regime_statistics(returns)

    current = stats.current_regime
    stability = stats.regime_stability

    # Regime adjustments
    adjustments = {
        MarketRegime.BULL_LOW_VOL: 1.15,    # +15% for bullish
        MarketRegime.BULL_HIGH_VOL: 1.10,   # +10% for volatile bullish
        MarketRegime.SIDEWAYS: 1.0,          # No adjustment
        MarketRegime.BEAR_LOW_VOL: 0.90,    # -10% for bearish
        MarketRegime.BEAR_HIGH_VOL: 0.85,   # -15% for volatile bearish
        MarketRegime.TRANSITION: 1.0,        # No adjustment during transition
    }

    factor = adjustments.get(current, 1.0)

    # Apply adjustment scaled by stability
    # If regime is unstable, reduce adjustment magnitude
    effective_factor = 1.0 + (factor - 1.0) * stability
    adjusted_prob = base_probability * effective_factor

    # Clamp to valid probability range
    adjusted_prob = max(0.2, min(0.8, adjusted_prob))

    return adjusted_prob, current, stability
