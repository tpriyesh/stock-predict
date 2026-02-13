"""
Kalman Filter for Optimal State Estimation

Mathematical Foundation:
========================
State-Space Model:
   xₜ = Axₜ₋₁ + Buₜ + wₜ    (State Transition)
   zₜ = Hxₜ + vₜ            (Observation)

Where:
- xₜ = [level, slope, acceleration]ᵀ (hidden state)
- zₜ = observed price/returns
- A = State transition matrix
- H = Observation matrix
- wₜ ~ N(0, Q) (process noise)
- vₜ ~ N(0, R) (observation noise)

Kalman Gain:
   Kₜ = PₜHᵀ(HPₜHᵀ + R)⁻¹

Update:
   x̂ₜ = x̂ₜ|ₜ₋₁ + Kₜ(zₜ - Hx̂ₜ|ₜ₋₁)
   Pₜ = (I - KₜH)Pₜ|ₜ₋₁

Advantages over Moving Average:
- No fixed lag (adapts to signal/noise ratio)
- Uncertainty quantification (Pₜ gives confidence interval)
- Optimal MSE estimator
- Handles missing data naturally

References:
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems
- Harvey, A.C. (1990). Forecasting, Structural Time Series Models and the Kalman Filter
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class StateModel(Enum):
    """Types of state-space models for financial data."""
    LOCAL_LEVEL = "local_level"  # Random walk + noise
    LOCAL_TREND = "local_trend"  # Level + slope
    LOCAL_ACCELERATION = "local_acceleration"  # Level + slope + acceleration
    CYCLE_PLUS_TREND = "cycle_plus_trend"  # Trend + cyclical component


@dataclass
class KalmanState:
    """Current state estimate from Kalman filter."""
    # State estimates
    level: float  # Current price level estimate
    slope: float  # Current trend (change per period)
    acceleration: float  # Rate of trend change

    # Uncertainty
    level_std: float  # Uncertainty in level
    slope_std: float  # Uncertainty in slope
    acceleration_std: float  # Uncertainty in acceleration

    # Confidence intervals (95%)
    level_ci: Tuple[float, float]
    slope_ci: Tuple[float, float]

    # Kalman gain (adaptiveness)
    kalman_gain: float

    # Innovation (prediction error)
    innovation: float
    normalized_innovation: float  # Should be ~N(0,1) if filter is correct


@dataclass
class KalmanResult:
    """Complete results from Kalman filtering."""
    # Filtered estimates (using data up to time t)
    filtered_level: np.ndarray
    filtered_slope: np.ndarray

    # Smoothed estimates (using all data)
    smoothed_level: np.ndarray
    smoothed_slope: np.ndarray

    # Uncertainty bands
    level_upper: np.ndarray  # 95% upper bound
    level_lower: np.ndarray  # 95% lower bound

    # Predictions
    forecast: np.ndarray
    forecast_upper: np.ndarray
    forecast_lower: np.ndarray

    # Diagnostics
    innovations: np.ndarray
    normalized_innovations: np.ndarray
    log_likelihood: float

    # Trading signals
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0-1
    trend_confidence: float  # Based on slope uncertainty
    signal_quality: float  # Based on innovation statistics


class KalmanTrendFilter:
    """
    Kalman Filter for trend estimation in stock prices.

    Advantages over Moving Averages:
    1. No lag: Adapts to signal/noise ratio automatically
    2. Uncertainty: Provides confidence intervals on trend
    3. Optimal: Minimizes mean squared error
    4. Flexible: Handles missing data and irregular observations
    """

    def __init__(
        self,
        model: StateModel = StateModel.LOCAL_TREND,
        process_noise_ratio: float = 0.1,  # Q/R ratio
        observation_noise_var: Optional[float] = None  # R (estimated if None)
    ):
        self.model = model
        self.process_noise_ratio = process_noise_ratio
        self.observation_noise_var = observation_noise_var

        # State dimensions
        if model == StateModel.LOCAL_LEVEL:
            self.state_dim = 1
        elif model == StateModel.LOCAL_TREND:
            self.state_dim = 2
        elif model == StateModel.LOCAL_ACCELERATION:
            self.state_dim = 3
        else:
            self.state_dim = 2

        # Matrices (set in initialize())
        self.A = None  # State transition
        self.H = None  # Observation
        self.Q = None  # Process noise covariance
        self.R = None  # Observation noise variance

        # Current state
        self.x = None  # State estimate
        self.P = None  # State covariance

    def _initialize_matrices(self, observation_variance: float):
        """Initialize state-space matrices."""

        if self.model == StateModel.LOCAL_LEVEL:
            # State: [level]
            self.A = np.array([[1.0]])
            self.H = np.array([[1.0]])
            self.Q = np.array([[observation_variance * self.process_noise_ratio]])

        elif self.model == StateModel.LOCAL_TREND:
            # State: [level, slope]
            self.A = np.array([
                [1.0, 1.0],
                [0.0, 1.0]
            ])
            self.H = np.array([[1.0, 0.0]])
            self.Q = np.array([
                [observation_variance * self.process_noise_ratio * 0.1, 0],
                [0, observation_variance * self.process_noise_ratio * 0.01]
            ])

        elif self.model == StateModel.LOCAL_ACCELERATION:
            # State: [level, slope, acceleration]
            self.A = np.array([
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0]
            ])
            self.H = np.array([[1.0, 0.0, 0.0]])
            self.Q = np.array([
                [observation_variance * self.process_noise_ratio * 0.1, 0, 0],
                [0, observation_variance * self.process_noise_ratio * 0.01, 0],
                [0, 0, observation_variance * self.process_noise_ratio * 0.001]
            ])

        self.R = np.array([[observation_variance]])

    def _initialize_state(self, initial_observations: np.ndarray):
        """Initialize state from first few observations."""
        n = len(initial_observations)

        if self.model == StateModel.LOCAL_LEVEL:
            self.x = np.array([np.mean(initial_observations)])
            self.P = np.array([[np.var(initial_observations) if n > 1 else self.R[0, 0]]])

        elif self.model == StateModel.LOCAL_TREND:
            level = np.mean(initial_observations)
            if n > 1:
                slope = (initial_observations[-1] - initial_observations[0]) / (n - 1)
            else:
                slope = 0.0

            self.x = np.array([level, slope])
            self.P = np.array([
                [np.var(initial_observations) if n > 1 else self.R[0, 0], 0],
                [0, 0.01 * self.R[0, 0]]
            ])

        elif self.model == StateModel.LOCAL_ACCELERATION:
            level = np.mean(initial_observations)
            if n > 2:
                # Fit quadratic to get slope and acceleration
                t = np.arange(n)
                coeffs = np.polyfit(t, initial_observations, 2)
                acceleration = 2 * coeffs[0]
                slope = coeffs[1]
            elif n > 1:
                slope = (initial_observations[-1] - initial_observations[0]) / (n - 1)
                acceleration = 0.0
            else:
                slope = 0.0
                acceleration = 0.0

            self.x = np.array([level, slope, acceleration])
            self.P = np.diag([
                np.var(initial_observations) if n > 1 else self.R[0, 0],
                0.01 * self.R[0, 0],
                0.001 * self.R[0, 0]
            ])

    def filter(self, observations: np.ndarray) -> KalmanResult:
        """
        Apply Kalman filter to observation sequence.

        Args:
            observations: Array of observed values (e.g., prices)

        Returns:
            KalmanResult with filtered and smoothed estimates
        """
        n = len(observations)
        if n < 5:
            return self._empty_result(n)

        # Estimate observation noise if not provided
        if self.observation_noise_var is None:
            # Use returns variance as observation noise estimate
            returns = np.diff(observations)
            obs_var = np.var(returns) if len(returns) > 1 else 1.0
        else:
            obs_var = self.observation_noise_var

        # Initialize matrices
        self._initialize_matrices(obs_var)

        # Initialize state from first few observations
        self._initialize_state(observations[:min(10, n)])

        # Storage for results
        filtered_states = np.zeros((n, self.state_dim))
        filtered_covs = np.zeros((n, self.state_dim, self.state_dim))
        innovations = np.zeros(n)
        innovation_vars = np.zeros(n)
        log_likelihood = 0.0

        # Forward pass (filtering)
        for t in range(n):
            z = observations[t]

            # Predict
            x_pred = self.A @ self.x
            P_pred = self.A @ self.P @ self.A.T + self.Q

            # Innovation
            y = z - (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]

            # Kalman gain
            K = (P_pred @ self.H.T) / S

            # Update
            self.x = x_pred + K.flatten() * y
            self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

            # Store
            filtered_states[t] = self.x
            filtered_covs[t] = self.P
            innovations[t] = y
            innovation_vars[t] = S

            # Log likelihood
            if S > 0:
                log_likelihood -= 0.5 * (np.log(2 * np.pi * S) + y ** 2 / S)

        # Backward pass (smoothing)
        smoothed_states = filtered_states.copy()
        smoothed_covs = filtered_covs.copy()

        for t in range(n - 2, -1, -1):
            # Predict from t to t+1
            P_pred = self.A @ filtered_covs[t] @ self.A.T + self.Q

            # Smoother gain
            J = filtered_covs[t] @ self.A.T @ np.linalg.inv(P_pred + 1e-10 * np.eye(self.state_dim))

            # Smooth
            smoothed_states[t] = filtered_states[t] + J @ (smoothed_states[t + 1] - self.A @ filtered_states[t])
            smoothed_covs[t] = filtered_covs[t] + J @ (smoothed_covs[t + 1] - P_pred) @ J.T

        # Extract level and slope
        filtered_level = filtered_states[:, 0]
        smoothed_level = smoothed_states[:, 0]

        if self.state_dim >= 2:
            filtered_slope = filtered_states[:, 1]
            smoothed_slope = smoothed_states[:, 1]
        else:
            # Estimate slope from level differences
            filtered_slope = np.concatenate([[0], np.diff(filtered_level)])
            smoothed_slope = np.concatenate([[0], np.diff(smoothed_level)])

        # Confidence bands (95%)
        level_std = np.sqrt(smoothed_covs[:, 0, 0])
        level_upper = smoothed_level + 1.96 * level_std
        level_lower = smoothed_level - 1.96 * level_std

        # Forecast
        forecast_horizon = min(5, n // 10 + 1)
        forecast = np.zeros(forecast_horizon)
        forecast_var = np.zeros(forecast_horizon)

        x_forecast = self.x.copy()
        P_forecast = self.P.copy()

        for h in range(forecast_horizon):
            x_forecast = self.A @ x_forecast
            P_forecast = self.A @ P_forecast @ self.A.T + self.Q
            forecast[h] = x_forecast[0]
            forecast_var[h] = P_forecast[0, 0] + self.R[0, 0]

        forecast_std = np.sqrt(forecast_var)
        forecast_upper = forecast + 1.96 * forecast_std
        forecast_lower = forecast - 1.96 * forecast_std

        # Normalized innovations (should be ~N(0,1))
        normalized_innovations = innovations / np.sqrt(innovation_vars + 1e-10)

        # Trading signals
        trend_direction, trend_strength, trend_confidence = self._compute_trend_signal(
            smoothed_slope, smoothed_covs
        )

        # Signal quality from innovation statistics
        signal_quality = self._compute_signal_quality(normalized_innovations)

        return KalmanResult(
            filtered_level=filtered_level,
            filtered_slope=filtered_slope,
            smoothed_level=smoothed_level,
            smoothed_slope=smoothed_slope,
            level_upper=level_upper,
            level_lower=level_lower,
            forecast=forecast,
            forecast_upper=forecast_upper,
            forecast_lower=forecast_lower,
            innovations=innovations,
            normalized_innovations=normalized_innovations,
            log_likelihood=log_likelihood,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_confidence=trend_confidence,
            signal_quality=signal_quality
        )

    def _compute_trend_signal(
        self,
        slope: np.ndarray,
        covs: np.ndarray
    ) -> Tuple[str, float, float]:
        """Compute trading signal from slope estimate."""
        if len(slope) == 0:
            return 'neutral', 0.0, 0.0

        current_slope = slope[-1]

        # Slope uncertainty
        if self.state_dim >= 2:
            slope_std = np.sqrt(covs[-1, 1, 1])
        else:
            slope_std = np.std(slope) if len(slope) > 1 else 1.0

        # Direction
        if current_slope > 0:
            direction = 'bullish'
        elif current_slope < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Strength (normalized by uncertainty)
        if slope_std > 0:
            z_score = abs(current_slope) / slope_std
            strength = min(1.0, z_score / 2.0)  # z=2 -> strength=1
        else:
            strength = 0.5

        # Confidence (how certain are we about the slope)
        if slope_std > 0 and abs(current_slope) > 0:
            confidence = min(1.0, abs(current_slope) / (slope_std * 2))
        else:
            confidence = 0.5

        return direction, strength, confidence

    def _compute_signal_quality(self, normalized_innovations: np.ndarray) -> float:
        """
        Compute signal quality from innovation statistics.

        If filter is correct, innovations should be:
        - Mean zero
        - Unit variance
        - Uncorrelated (white noise)
        """
        if len(normalized_innovations) < 10:
            return 0.5

        # Mean should be ~0
        mean_penalty = 1 - min(1, abs(np.mean(normalized_innovations)))

        # Variance should be ~1
        var = np.var(normalized_innovations)
        var_penalty = 1 - min(1, abs(var - 1))

        # Autocorrelation should be ~0
        if len(normalized_innovations) > 1:
            autocorr = np.corrcoef(normalized_innovations[:-1], normalized_innovations[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
            autocorr_penalty = 1 - abs(autocorr)
        else:
            autocorr_penalty = 0.5

        # Weighted average
        signal_quality = 0.4 * mean_penalty + 0.4 * var_penalty + 0.2 * autocorr_penalty

        return signal_quality

    def _empty_result(self, n: int) -> KalmanResult:
        """Return empty result for insufficient data."""
        empty = np.zeros(n)
        return KalmanResult(
            filtered_level=empty,
            filtered_slope=empty,
            smoothed_level=empty,
            smoothed_slope=empty,
            level_upper=empty,
            level_lower=empty,
            forecast=np.zeros(1),
            forecast_upper=np.zeros(1),
            forecast_lower=np.zeros(1),
            innovations=empty,
            normalized_innovations=empty,
            log_likelihood=0.0,
            trend_direction='neutral',
            trend_strength=0.0,
            trend_confidence=0.0,
            signal_quality=0.0
        )

    def get_current_state(self) -> KalmanState:
        """Get current state estimate."""
        if self.x is None or self.P is None:
            return KalmanState(
                level=0, slope=0, acceleration=0,
                level_std=0, slope_std=0, acceleration_std=0,
                level_ci=(0, 0), slope_ci=(0, 0),
                kalman_gain=0, innovation=0, normalized_innovation=0
            )

        level = self.x[0]
        level_std = np.sqrt(self.P[0, 0])

        if self.state_dim >= 2:
            slope = self.x[1]
            slope_std = np.sqrt(self.P[1, 1])
        else:
            slope = 0.0
            slope_std = 0.0

        if self.state_dim >= 3:
            acceleration = self.x[2]
            acceleration_std = np.sqrt(self.P[2, 2])
        else:
            acceleration = 0.0
            acceleration_std = 0.0

        return KalmanState(
            level=level,
            slope=slope,
            acceleration=acceleration,
            level_std=level_std,
            slope_std=slope_std,
            acceleration_std=acceleration_std,
            level_ci=(level - 1.96 * level_std, level + 1.96 * level_std),
            slope_ci=(slope - 1.96 * slope_std, slope + 1.96 * slope_std),
            kalman_gain=0.0,  # Would need to store from last update
            innovation=0.0,
            normalized_innovation=0.0
        )


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter that automatically adjusts noise parameters.

    Uses innovation-based adaptation to tune Q and R online,
    making the filter more robust to changing market conditions.
    """

    def __init__(
        self,
        model: StateModel = StateModel.LOCAL_TREND,
        adaptation_rate: float = 0.1,
        window_size: int = 20
    ):
        self.model = model
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size

        self.base_filter = KalmanTrendFilter(model=model)

        # Innovation history for adaptation
        self.innovation_history = []

    def filter_adaptive(self, observations: np.ndarray) -> KalmanResult:
        """
        Apply adaptive Kalman filter with online noise estimation.

        Adapts Q and R based on observed innovation statistics.
        """
        n = len(observations)
        if n < self.window_size:
            return self.base_filter.filter(observations)

        # Initial filter pass
        result = self.base_filter.filter(observations)

        # Check if adaptation needed
        recent_innovations = result.innovations[-self.window_size:]
        innovation_var = np.var(recent_innovations)

        # Expected innovation variance = H P H' + R
        expected_var = 1.0  # Normalized

        # Adaptation factor
        adaptation_factor = innovation_var / (expected_var + 1e-10)

        # If innovations are too large, increase observation noise
        # If innovations are too small, decrease observation noise
        if adaptation_factor > 1.5:
            # Innovations too large -> model is underfit
            # Decrease Q (trust model more) or increase R (trust observations less)
            new_obs_var = self.base_filter.observation_noise_var * (1 + self.adaptation_rate)
            self.base_filter.observation_noise_var = new_obs_var
            logger.debug(f"Kalman adaptation: increased R to {new_obs_var:.4f}")
            # Re-run filter
            result = self.base_filter.filter(observations)

        elif adaptation_factor < 0.5:
            # Innovations too small -> model is overfit or noise overestimated
            # Increase Q or decrease R
            if self.base_filter.observation_noise_var is not None:
                new_obs_var = self.base_filter.observation_noise_var * (1 - self.adaptation_rate)
                self.base_filter.observation_noise_var = max(0.0001, new_obs_var)
                logger.debug(f"Kalman adaptation: decreased R to {new_obs_var:.4f}")
                result = self.base_filter.filter(observations)

        return result

    def get_optimal_parameters(
        self,
        observations: np.ndarray,
        q_range: Tuple[float, float] = (0.01, 1.0),
        r_range: Tuple[float, float] = (0.01, 1.0),
        n_trials: int = 20
    ) -> Dict[str, float]:
        """
        Find optimal Q/R parameters using grid search on log-likelihood.
        """
        best_ll = -np.inf
        best_params = {'q_ratio': 0.1, 'r': None}

        returns_var = np.var(np.diff(observations)) if len(observations) > 1 else 1.0

        for _ in range(n_trials):
            # Random search
            q_ratio = np.exp(np.random.uniform(np.log(q_range[0]), np.log(q_range[1])))
            r_factor = np.exp(np.random.uniform(np.log(r_range[0]), np.log(r_range[1])))
            r = returns_var * r_factor

            test_filter = KalmanTrendFilter(
                model=self.model,
                process_noise_ratio=q_ratio,
                observation_noise_var=r
            )

            result = test_filter.filter(observations)

            if result.log_likelihood > best_ll:
                best_ll = result.log_likelihood
                best_params = {'q_ratio': q_ratio, 'r': r, 'log_likelihood': best_ll}

        return best_params


def compute_kalman_probability(
    prices: np.ndarray,
    horizon: int = 5
) -> Tuple[float, float, str]:
    """
    Compute bullish probability using Kalman filter.

    Returns:
        (probability, confidence, signal)
    """
    if len(prices) < 20:
        return 0.5, 0.0, 'HOLD'

    # Use local trend model
    kf = KalmanTrendFilter(model=StateModel.LOCAL_TREND)
    result = kf.filter(prices)

    # Get current slope and uncertainty
    current_slope = result.smoothed_slope[-1]
    slope_history = result.smoothed_slope[-20:]
    slope_std = np.std(slope_history) if len(slope_history) > 1 else 1.0

    # Convert slope to probability using logistic function
    # slope / std gives z-score
    if slope_std > 0:
        z_score = current_slope / slope_std
    else:
        z_score = 0

    # Logistic sigmoid: P(up) = 1 / (1 + exp(-z))
    probability = 1 / (1 + np.exp(-z_score))

    # Confidence from signal quality and trend confidence
    confidence = (result.signal_quality + result.trend_confidence) / 2

    # Signal
    if probability >= 0.6 and confidence >= 0.5:
        signal = 'BUY'
    elif probability <= 0.4 and confidence >= 0.5:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return probability, confidence, signal
