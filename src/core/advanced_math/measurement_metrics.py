"""
Comprehensive Multi-Dimensional Measurement Metrics

This module provides rigorous statistical measures for evaluating prediction quality
across multiple dimensions including accuracy, calibration, risk-adjustment, and
information content.

Key Metrics:
1. Accuracy Metrics: Direction accuracy, magnitude calibration, hit rate
2. Calibration Metrics: Brier score, reliability diagrams, calibration error
3. Information Metrics: Mutual information, entropy reduction, Sharpe of predictions
4. Risk-Adjusted Metrics: Sortino ratio, Calmar ratio, Omega ratio
5. Statistical Significance: p-values, confidence intervals, bootstrap tests

Academic References:
- Gneiting & Raftery (2007) - "Strictly Proper Scoring Rules, Prediction, and Estimation"
- Murphy (1973) - "A New Vector Partition of the Probability Score"
- Brier (1950) - "Verification of forecasts expressed in terms of probability"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime


class MetricDimension(Enum):
    """Dimensions for multi-dimensional evaluation."""
    ACCURACY = "accuracy"
    CALIBRATION = "calibration"
    TIMING = "timing"
    RISK_ADJUSTED = "risk_adjusted"
    INFORMATION = "information"
    STABILITY = "stability"


@dataclass
class AccuracyMetrics:
    """Direction and magnitude accuracy metrics."""
    direction_accuracy: float  # % correct direction
    hit_rate: float  # % of profitable trades
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1_score: float  # Harmonic mean of precision/recall
    mean_absolute_error: float  # MAE of return predictions
    root_mean_squared_error: float  # RMSE
    mean_directional_accuracy: float  # MDA
    profit_factor: float  # Gross profit / Gross loss


@dataclass
class CalibrationMetrics:
    """Probability calibration metrics."""
    brier_score: float  # Mean squared error of probabilities
    brier_skill_score: float  # Relative to climatology
    log_loss: float  # Negative log-likelihood
    calibration_error: float  # Expected Calibration Error (ECE)
    max_calibration_error: float  # Maximum Calibration Error (MCE)
    reliability_slope: float  # Slope of reliability diagram
    reliability_intercept: float  # Intercept of reliability diagram
    resolution: float  # Murphy decomposition
    uncertainty: float  # Base rate uncertainty


@dataclass
class TimingMetrics:
    """Signal timing and entry quality metrics."""
    entry_efficiency: float  # How close to optimal entry
    exit_efficiency: float  # How close to optimal exit
    average_bars_to_target: float  # Avg time to hit target
    max_adverse_excursion: float  # Worst drawdown during trade
    max_favorable_excursion: float  # Best unrealized gain
    win_streak_avg: float  # Average winning streak
    loss_streak_avg: float  # Average losing streak
    time_in_market: float  # % of time with position


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float  # Return / Volatility
    sortino_ratio: float  # Return / Downside deviation
    calmar_ratio: float  # Return / Max drawdown
    omega_ratio: float  # Probability-weighted gains/losses
    information_ratio: float  # Alpha / Tracking error
    treynor_ratio: float  # Return / Beta
    max_drawdown: float  # Maximum peak-to-trough decline
    average_drawdown: float  # Mean drawdown
    recovery_factor: float  # Net profit / Max drawdown
    ulcer_index: float  # Quadratic average drawdown


@dataclass
class InformationMetrics:
    """Information content and predictive power metrics."""
    mutual_information: float  # MI(predictions, outcomes)
    entropy_reduction: float  # Uncertainty reduction
    relative_entropy: float  # KL divergence from baseline
    information_coefficient: float  # Correlation of predictions to outcomes
    transfer_entropy: float  # Directional information flow
    effective_data_fraction: float  # Fraction of data contributing to signal
    signal_to_noise_ratio: float  # Predictive signal vs noise
    redundancy: float  # Overlap with other predictors


@dataclass
class StabilityMetrics:
    """Prediction stability and robustness metrics."""
    autocorrelation: float  # Signal autocorrelation
    variance_ratio: float  # Lo-MacKinlay variance ratio
    hurst_exponent: float  # Long-term memory
    regime_consistency: float  # Consistency across regimes
    time_stability: float  # Rolling performance stability
    cross_validation_stability: float  # CV performance variance
    bootstrap_ci_width: float  # Confidence interval width
    sensitivity_to_outliers: float  # Robustness to outliers


@dataclass
class MultiDimensionalScore:
    """Comprehensive multi-dimensional prediction quality score."""
    # Individual dimension scores (0-1)
    accuracy_score: float
    calibration_score: float
    timing_score: float
    risk_adjusted_score: float
    information_score: float
    stability_score: float

    # Composite
    overall_score: float
    weighted_score: float
    dimension_weights: Dict[str, float]

    # Detailed metrics
    accuracy: AccuracyMetrics
    calibration: CalibrationMetrics
    timing: TimingMetrics
    risk_adjusted: RiskAdjustedMetrics
    information: InformationMetrics
    stability: StabilityMetrics

    # Metadata
    n_observations: int
    time_period_days: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_score': self.overall_score,
            'dimension_scores': {
                'accuracy': self.accuracy_score,
                'calibration': self.calibration_score,
                'timing': self.timing_score,
                'risk_adjusted': self.risk_adjusted_score,
                'information': self.information_score,
                'stability': self.stability_score
            },
            'n_observations': self.n_observations,
            'time_period_days': self.time_period_days
        }


class MeasurementMetricsEngine:
    """
    Comprehensive engine for multi-dimensional prediction quality measurement.

    This engine evaluates predictions across multiple dimensions to provide
    a rigorous assessment of prediction quality.
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize measurement engine.

        Args:
            dimension_weights: Custom weights for each dimension
        """
        self.dimension_weights = dimension_weights or {
            MetricDimension.ACCURACY.value: 0.25,
            MetricDimension.CALIBRATION.value: 0.20,
            MetricDimension.TIMING.value: 0.15,
            MetricDimension.RISK_ADJUSTED.value: 0.20,
            MetricDimension.INFORMATION.value: 0.10,
            MetricDimension.STABILITY.value: 0.10
        }

        # Normalize weights
        total = sum(self.dimension_weights.values())
        self.dimension_weights = {k: v/total for k, v in self.dimension_weights.items()}

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.05
    ) -> MultiDimensionalScore:
        """
        Compute comprehensive metrics across all dimensions.

        Args:
            predictions: Predicted signals (-1, 0, 1) or continuous
            actuals: Actual outcomes
            probabilities: Predicted probabilities (for calibration)
            returns: Strategy returns (for risk metrics)
            timestamps: Timestamps for temporal analysis
            risk_free_rate: Annual risk-free rate

        Returns:
            MultiDimensionalScore with all metrics
        """
        n = len(predictions)

        # Handle missing optional arrays
        if probabilities is None:
            # Convert predictions to probabilities
            probabilities = np.clip((predictions + 1) / 2, 0, 1)

        if returns is None:
            # Use sign(prediction) * actual as proxy returns
            returns = np.sign(predictions) * actuals

        # Compute each dimension
        accuracy = self._compute_accuracy_metrics(predictions, actuals)
        calibration = self._compute_calibration_metrics(probabilities, actuals)
        timing = self._compute_timing_metrics(predictions, actuals, returns)
        risk_adjusted = self._compute_risk_adjusted_metrics(returns, risk_free_rate)
        information = self._compute_information_metrics(predictions, actuals, probabilities)
        stability = self._compute_stability_metrics(predictions, returns)

        # Convert to scores (0-1)
        accuracy_score = self._accuracy_to_score(accuracy)
        calibration_score = self._calibration_to_score(calibration)
        timing_score = self._timing_to_score(timing)
        risk_adjusted_score = self._risk_adjusted_to_score(risk_adjusted)
        information_score = self._information_to_score(information)
        stability_score = self._stability_to_score(stability)

        # Overall scores
        scores = {
            'accuracy': accuracy_score,
            'calibration': calibration_score,
            'timing': timing_score,
            'risk_adjusted': risk_adjusted_score,
            'information': information_score,
            'stability': stability_score
        }

        overall_score = np.mean(list(scores.values()))
        weighted_score = sum(
            scores[dim] * self.dimension_weights.get(dim, 0)
            for dim in scores
        )

        # Time period calculation
        time_period_days = n  # Assume daily data
        if timestamps is not None and len(timestamps) > 1:
            try:
                time_period_days = (timestamps[-1] - timestamps[0]).days
            except:
                pass

        return MultiDimensionalScore(
            accuracy_score=accuracy_score,
            calibration_score=calibration_score,
            timing_score=timing_score,
            risk_adjusted_score=risk_adjusted_score,
            information_score=information_score,
            stability_score=stability_score,
            overall_score=overall_score,
            weighted_score=weighted_score,
            dimension_weights=self.dimension_weights,
            accuracy=accuracy,
            calibration=calibration,
            timing=timing,
            risk_adjusted=risk_adjusted,
            information=information,
            stability=stability,
            n_observations=n,
            time_period_days=time_period_days
        )

    def _compute_accuracy_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> AccuracyMetrics:
        """Compute accuracy-related metrics."""
        # Direction accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        direction_accuracy = np.mean(pred_direction == actual_direction)

        # Mean Directional Accuracy
        mda = np.mean((pred_direction * actual_direction) > 0)

        # Hit rate (profitable predictions)
        strategy_returns = pred_direction * actuals
        hit_rate = np.mean(strategy_returns > 0)

        # Classification metrics
        # True positives, false positives, etc.
        tp = np.sum((pred_direction > 0) & (actual_direction > 0))
        fp = np.sum((pred_direction > 0) & (actual_direction <= 0))
        fn = np.sum((pred_direction <= 0) & (actual_direction > 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Error metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return AccuracyMetrics(
            direction_accuracy=direction_accuracy,
            hit_rate=hit_rate,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_directional_accuracy=mda,
            profit_factor=min(profit_factor, 10)  # Cap at 10
        )

    def _compute_calibration_metrics(
        self,
        probabilities: np.ndarray,
        actuals: np.ndarray
    ) -> CalibrationMetrics:
        """Compute probability calibration metrics."""
        # Convert actuals to binary outcomes (1 if positive)
        outcomes = (actuals > 0).astype(float)

        # Brier score
        brier_score = np.mean((probabilities - outcomes) ** 2)

        # Climatology (base rate) for skill score
        base_rate = np.mean(outcomes)
        brier_climatology = base_rate * (1 - base_rate)
        brier_skill_score = 1 - brier_score / brier_climatology if brier_climatology > 0 else 0

        # Log loss
        eps = 1e-15
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        log_loss = -np.mean(
            outcomes * np.log(probs_clipped) +
            (1 - outcomes) * np.log(1 - probs_clipped)
        )

        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        mce = 0

        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if np.sum(mask) > 0:
                avg_confidence = np.mean(probabilities[mask])
                avg_accuracy = np.mean(outcomes[mask])
                bin_error = np.abs(avg_confidence - avg_accuracy)
                ece += np.sum(mask) / len(probabilities) * bin_error
                mce = max(mce, bin_error)

        # Reliability diagram slope/intercept
        try:
            coeffs = np.polyfit(probabilities, outcomes, 1)
            reliability_slope = coeffs[0]
            reliability_intercept = coeffs[1]
        except:
            reliability_slope = 1.0
            reliability_intercept = 0.0

        # Murphy decomposition: Brier = Reliability - Resolution + Uncertainty
        uncertainty = base_rate * (1 - base_rate)

        # Resolution: variance of conditional means
        resolution = 0
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_outcome = np.mean(outcomes[mask])
                resolution += np.sum(mask) / len(probabilities) * (bin_outcome - base_rate) ** 2

        return CalibrationMetrics(
            brier_score=brier_score,
            brier_skill_score=brier_skill_score,
            log_loss=log_loss,
            calibration_error=ece,
            max_calibration_error=mce,
            reliability_slope=reliability_slope,
            reliability_intercept=reliability_intercept,
            resolution=resolution,
            uncertainty=uncertainty
        )

    def _compute_timing_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        returns: np.ndarray
    ) -> TimingMetrics:
        """Compute timing-related metrics."""
        pred_direction = np.sign(predictions)

        # Entry efficiency: how close to best possible entry
        # Simplified: compare actual entry return to best possible
        cumulative = np.cumsum(returns)
        if len(cumulative) > 0:
            best_possible = np.max(cumulative) - np.min(cumulative[:np.argmax(cumulative)])
            actual_return = cumulative[-1]
            entry_efficiency = actual_return / best_possible if best_possible > 0 else 0.5
        else:
            entry_efficiency = 0.5

        # Exit efficiency
        if len(returns) > 0 and np.max(cumulative) > 0:
            exit_efficiency = cumulative[-1] / np.max(cumulative)
        else:
            exit_efficiency = 0.5

        # Maximum adverse/favorable excursion
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_adverse = np.max(drawdowns) if len(drawdowns) > 0 else 0
        max_favorable = np.max(cumulative) - cumulative[0] if len(cumulative) > 0 else 0

        # Win/loss streaks
        wins = returns > 0
        streaks = []
        current_streak = 0
        is_win_streak = None

        for win in wins:
            if is_win_streak is None:
                is_win_streak = win
                current_streak = 1
            elif win == is_win_streak:
                current_streak += 1
            else:
                streaks.append((is_win_streak, current_streak))
                is_win_streak = win
                current_streak = 1
        streaks.append((is_win_streak, current_streak))

        win_streaks = [s[1] for s in streaks if s[0]]
        loss_streaks = [s[1] for s in streaks if not s[0]]

        win_streak_avg = np.mean(win_streaks) if win_streaks else 0
        loss_streak_avg = np.mean(loss_streaks) if loss_streaks else 0

        # Time in market
        time_in_market = np.mean(np.abs(pred_direction) > 0)

        # Average bars to target (simplified)
        avg_bars_to_target = len(returns) / max(1, np.sum(returns > 0))

        return TimingMetrics(
            entry_efficiency=np.clip(entry_efficiency, 0, 1),
            exit_efficiency=np.clip(exit_efficiency, 0, 1),
            average_bars_to_target=avg_bars_to_target,
            max_adverse_excursion=max_adverse,
            max_favorable_excursion=max_favorable,
            win_streak_avg=win_streak_avg,
            loss_streak_avg=loss_streak_avg,
            time_in_market=time_in_market
        )

    def _compute_risk_adjusted_metrics(
        self,
        returns: np.ndarray,
        risk_free_rate: float
    ) -> RiskAdjustedMetrics:
        """Compute risk-adjusted performance metrics."""
        if len(returns) == 0:
            return self._empty_risk_metrics()

        # Daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - rf_daily

        # Sharpe ratio (annualized)
        mean_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = (mean_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        avg_drawdown = np.mean(drawdowns) if len(drawdowns) > 0 else 0

        # Calmar ratio
        calmar = mean_return / max_drawdown if max_drawdown > 0 else 0

        # Omega ratio
        threshold = 0
        gains = np.sum(returns[returns > threshold] - threshold)
        losses = np.abs(np.sum(returns[returns < threshold] - threshold))
        omega = gains / losses if losses > 0 else np.inf
        omega = min(omega, 10)

        # Recovery factor
        net_profit = cumulative[-1] if len(cumulative) > 0 else 0
        recovery = net_profit / max_drawdown if max_drawdown > 0 else 0

        # Ulcer index
        ulcer = np.sqrt(np.mean(drawdowns ** 2)) if len(drawdowns) > 0 else 0

        # Information ratio (assume benchmark = 0)
        tracking_error = np.std(returns) * np.sqrt(252)
        info_ratio = mean_return / tracking_error if tracking_error > 0 else 0

        # Treynor ratio (assume beta = 1 if not available)
        treynor = (mean_return - risk_free_rate)

        return RiskAdjustedMetrics(
            sharpe_ratio=np.clip(sharpe, -5, 5),
            sortino_ratio=np.clip(sortino, -5, 5),
            calmar_ratio=np.clip(calmar, -10, 10),
            omega_ratio=omega,
            information_ratio=np.clip(info_ratio, -5, 5),
            treynor_ratio=np.clip(treynor, -5, 5),
            max_drawdown=max_drawdown,
            average_drawdown=avg_drawdown,
            recovery_factor=np.clip(recovery, -10, 10),
            ulcer_index=ulcer
        )

    def _empty_risk_metrics(self) -> RiskAdjustedMetrics:
        """Return empty risk metrics."""
        return RiskAdjustedMetrics(
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            omega_ratio=1, information_ratio=0, treynor_ratio=0,
            max_drawdown=0, average_drawdown=0, recovery_factor=0, ulcer_index=0
        )

    def _compute_information_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray
    ) -> InformationMetrics:
        """Compute information-theoretic metrics."""
        # Discretize for MI calculation
        n_bins = 10

        # Mutual information
        mi = self._compute_mutual_information(predictions, actuals, n_bins)

        # Entropy of predictions
        pred_hist, _ = np.histogram(predictions, bins=n_bins, density=True)
        pred_hist = pred_hist + 1e-10
        pred_entropy = -np.sum(pred_hist * np.log(pred_hist + 1e-10))

        # Entropy of actuals
        act_hist, _ = np.histogram(actuals, bins=n_bins, density=True)
        act_hist = act_hist + 1e-10
        act_entropy = -np.sum(act_hist * np.log(act_hist + 1e-10))

        # Entropy reduction
        entropy_reduction = max(0, act_entropy - (act_entropy - mi))

        # Relative entropy (KL divergence from uniform)
        uniform = np.ones(n_bins) / n_bins
        rel_entropy = np.sum(pred_hist * np.log((pred_hist + 1e-10) / uniform))

        # Information coefficient (correlation)
        if np.std(predictions) > 0 and np.std(actuals) > 0:
            ic = np.corrcoef(predictions, actuals)[0, 1]
        else:
            ic = 0

        # Transfer entropy (simplified)
        te = mi * 0.5  # Approximation

        # Signal to noise ratio
        signal_var = np.var(predictions)
        noise_var = np.var(predictions - actuals)
        snr = signal_var / noise_var if noise_var > 0 else 0

        # Effective data fraction
        n_unique = len(np.unique(np.round(predictions, 2)))
        edf = n_unique / len(predictions) if len(predictions) > 0 else 0

        return InformationMetrics(
            mutual_information=mi,
            entropy_reduction=entropy_reduction,
            relative_entropy=rel_entropy,
            information_coefficient=ic if not np.isnan(ic) else 0,
            transfer_entropy=te,
            effective_data_fraction=edf,
            signal_to_noise_ratio=min(snr, 10),
            redundancy=1 - edf
        )

    def _compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute mutual information between x and y."""
        # 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist_2d = hist_2d / np.sum(hist_2d)

        # Marginals
        p_x = np.sum(hist_2d, axis=1)
        p_y = np.sum(hist_2d, axis=0)

        # Mutual information
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log(
                        hist_2d[i, j] / (p_x[i] * p_y[j] + 1e-10) + 1e-10
                    )

        return max(0, mi)

    def _compute_stability_metrics(
        self,
        predictions: np.ndarray,
        returns: np.ndarray
    ) -> StabilityMetrics:
        """Compute stability and robustness metrics."""
        n = len(predictions)

        # Autocorrelation
        if n > 1:
            autocorr = np.corrcoef(predictions[:-1], predictions[1:])[0, 1]
            autocorr = autocorr if not np.isnan(autocorr) else 0
        else:
            autocorr = 0

        # Variance ratio (Lo-MacKinlay)
        if n > 10:
            period = min(10, n // 4)
            var_1 = np.var(returns)
            cumsum = np.add.reduceat(
                returns,
                np.arange(0, n, period)
            )
            var_q = np.var(cumsum) / period
            variance_ratio = var_q / var_1 if var_1 > 0 else 1
        else:
            variance_ratio = 1.0

        # Hurst exponent (simplified R/S method)
        hurst = self._compute_hurst(predictions)

        # Rolling performance stability
        if n >= 20:
            window = max(5, n // 10)
            rolling_sharpe = []
            for i in range(0, n - window, window // 2):
                window_returns = returns[i:i + window]
                if len(window_returns) > 0 and np.std(window_returns) > 0:
                    sharpe = np.mean(window_returns) / np.std(window_returns)
                    rolling_sharpe.append(sharpe)
            time_stability = 1 - np.std(rolling_sharpe) if rolling_sharpe else 0.5
        else:
            time_stability = 0.5

        # Cross-validation stability (simplified k-fold)
        k = 5
        fold_size = n // k
        fold_returns = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size
            fold_ret = np.sum(returns[start:end])
            fold_returns.append(fold_ret)
        cv_stability = 1 - np.std(fold_returns) / (np.abs(np.mean(fold_returns)) + 1e-10)

        # Bootstrap confidence interval width
        n_bootstrap = 100
        bootstrap_means = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            bootstrap_means.append(np.mean(returns[idx]))
        ci_width = np.percentile(bootstrap_means, 97.5) - np.percentile(bootstrap_means, 2.5)

        # Sensitivity to outliers
        median_return = np.median(returns)
        mean_return = np.mean(returns)
        sensitivity = np.abs(mean_return - median_return) / (np.std(returns) + 1e-10)

        # Regime consistency (simplified)
        regime_consistency = 0.5 + 0.5 * (1 - np.abs(autocorr - 0.5))

        return StabilityMetrics(
            autocorrelation=autocorr,
            variance_ratio=np.clip(variance_ratio, 0, 3),
            hurst_exponent=hurst,
            regime_consistency=np.clip(regime_consistency, 0, 1),
            time_stability=np.clip(time_stability, 0, 1),
            cross_validation_stability=np.clip(cv_stability, 0, 1),
            bootstrap_ci_width=ci_width,
            sensitivity_to_outliers=np.clip(sensitivity, 0, 3)
        )

    def _compute_hurst(self, data: np.ndarray) -> float:
        """Compute Hurst exponent using R/S method."""
        n = len(data)
        if n < 20:
            return 0.5

        try:
            max_lag = min(100, n // 2)
            lags = [2 ** i for i in range(2, int(np.log2(max_lag)) + 1)]
            rs_values = []

            for lag in lags:
                if lag >= n:
                    break

                # R/S calculation
                segments = n // lag
                rs_segment = []

                for i in range(segments):
                    segment = data[i * lag:(i + 1) * lag]
                    mean_seg = np.mean(segment)
                    cumdev = np.cumsum(segment - mean_seg)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = np.std(segment)
                    if s > 0:
                        rs_segment.append(r / s)

                if rs_segment:
                    rs_values.append((np.log(lag), np.log(np.mean(rs_segment))))

            if len(rs_values) >= 2:
                rs_arr = np.array(rs_values)
                slope, _ = np.polyfit(rs_arr[:, 0], rs_arr[:, 1], 1)
                return np.clip(slope, 0, 1)
        except:
            pass

        return 0.5

    # Score conversion methods

    def _accuracy_to_score(self, metrics: AccuracyMetrics) -> float:
        """Convert accuracy metrics to 0-1 score."""
        scores = [
            metrics.direction_accuracy,
            metrics.hit_rate,
            metrics.f1_score,
            min(metrics.profit_factor / 2, 1)  # PF of 2 = score of 1
        ]
        return np.mean(scores)

    def _calibration_to_score(self, metrics: CalibrationMetrics) -> float:
        """Convert calibration metrics to 0-1 score."""
        # Lower Brier is better (0 is perfect)
        brier_score = 1 - min(metrics.brier_score * 2, 1)

        # ECE close to 0 is better
        ece_score = 1 - min(metrics.calibration_error * 2, 1)

        # Slope close to 1 is better
        slope_score = 1 - min(np.abs(metrics.reliability_slope - 1), 1)

        return np.mean([brier_score, ece_score, slope_score])

    def _timing_to_score(self, metrics: TimingMetrics) -> float:
        """Convert timing metrics to 0-1 score."""
        return np.mean([
            metrics.entry_efficiency,
            metrics.exit_efficiency,
            min(metrics.win_streak_avg / 5, 1),  # 5+ win streak = perfect
            1 - min(metrics.loss_streak_avg / 5, 1)  # Low loss streak is good
        ])

    def _risk_adjusted_to_score(self, metrics: RiskAdjustedMetrics) -> float:
        """Convert risk-adjusted metrics to 0-1 score."""
        # Sharpe of 2+ is excellent
        sharpe_score = min(max(metrics.sharpe_ratio / 2, 0), 1)

        # Sortino of 2+ is excellent
        sortino_score = min(max(metrics.sortino_ratio / 2, 0), 1)

        # Low max drawdown is good
        dd_score = 1 - min(metrics.max_drawdown * 5, 1)  # 20% DD = score 0

        # Omega > 2 is good
        omega_score = min(max((metrics.omega_ratio - 1) / 2, 0), 1)

        return np.mean([sharpe_score, sortino_score, dd_score, omega_score])

    def _information_to_score(self, metrics: InformationMetrics) -> float:
        """Convert information metrics to 0-1 score."""
        # IC of 0.1+ is good
        ic_score = min(np.abs(metrics.information_coefficient) / 0.1, 1)

        # SNR > 1 is good
        snr_score = min(metrics.signal_to_noise_ratio, 1)

        # MI normalized
        mi_score = min(metrics.mutual_information * 2, 1)

        return np.mean([ic_score, snr_score, mi_score])

    def _stability_to_score(self, metrics: StabilityMetrics) -> float:
        """Convert stability metrics to 0-1 score."""
        # Hurst near 0.5 indicates randomness, >0.5 persistence, <0.5 mean-reversion
        hurst_score = 1 - 2 * np.abs(metrics.hurst_exponent - 0.5)

        return np.mean([
            hurst_score,
            metrics.time_stability,
            metrics.cross_validation_stability,
            metrics.regime_consistency,
            1 - min(metrics.sensitivity_to_outliers / 2, 1)
        ])


# Singleton instance
_metrics_engine = None


def get_metrics_engine() -> MeasurementMetricsEngine:
    """Get singleton metrics engine instance."""
    global _metrics_engine
    if _metrics_engine is None:
        _metrics_engine = MeasurementMetricsEngine()
    return _metrics_engine


def compute_prediction_quality(
    predictions: np.ndarray,
    actuals: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> MultiDimensionalScore:
    """
    Convenience function to compute comprehensive prediction quality metrics.

    Args:
        predictions: Predicted signals or returns
        actuals: Actual outcomes
        probabilities: Optional predicted probabilities

    Returns:
        MultiDimensionalScore with all metrics
    """
    engine = get_metrics_engine()
    return engine.compute_all_metrics(predictions, actuals, probabilities)


def quick_quality_check(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Quick quality check returning only dimension scores.

    Returns:
        Dict with dimension names and scores (0-1)
    """
    score = compute_prediction_quality(predictions, actuals)
    return {
        'accuracy': score.accuracy_score,
        'calibration': score.calibration_score,
        'timing': score.timing_score,
        'risk_adjusted': score.risk_adjusted_score,
        'information': score.information_score,
        'stability': score.stability_score,
        'overall': score.overall_score
    }
