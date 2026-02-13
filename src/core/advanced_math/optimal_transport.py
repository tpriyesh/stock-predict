"""
Optimal Transport Module for Distribution Shift Detection

Implements Wasserstein distances and optimal transport methods:
- Wasserstein-1 (Earth Mover's Distance)
- Wasserstein-2 (Quadratic)
- Sinkhorn Distance (Regularized OT)
- Sliced Wasserstein Distance

Applications:
- Regime change detection
- Distribution drift monitoring
- Feature importance via transport cost

Mathematical Foundation:
- W_p(μ,ν) = (inf_γ∈Π(μ,ν) ∫|x-y|^p dγ(x,y))^(1/p)
- Sinkhorn: K_ij = exp(-C_ij/ε)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimalTransportResult:
    """Result of optimal transport analysis"""
    wasserstein_1: float  # W1 distance
    wasserstein_2: float  # W2 distance
    sinkhorn_distance: float  # Regularized distance
    transport_plan: np.ndarray  # Optimal coupling
    marginal_cost: np.ndarray  # Cost per source bin
    regime_shift_score: float  # 0-1 shift indicator
    drift_direction: str  # left, right, stable
    transport_interpretation: Dict[str, Any]


class WassersteinDistance:
    """
    Wasserstein (Earth Mover's) Distance computation.

    W_p(P,Q) = (inf_γ E[|X-Y|^p])^(1/p)

    For 1D distributions, uses sorting-based exact computation.
    """

    def __init__(
        self,
        p: int = 1,
        n_bins: int = 100
    ):
        self.p = p
        self.n_bins = n_bins

    def compute_1d(
        self,
        samples_p: np.ndarray,
        samples_q: np.ndarray
    ) -> float:
        """
        Compute Wasserstein-p distance for 1D samples.

        For 1D: W_p = (∫|F_P^(-1)(t) - F_Q^(-1)(t)|^p dt)^(1/p)
        """
        # Sort samples
        p_sorted = np.sort(samples_p)
        q_sorted = np.sort(samples_q)

        # Interpolate to same length
        n = max(len(p_sorted), len(q_sorted))
        p_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(p_sorted)),
            p_sorted
        )
        q_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(q_sorted)),
            q_sorted
        )

        # Compute Wasserstein distance
        if self.p == 1:
            distance = np.mean(np.abs(p_interp - q_interp))
        else:
            distance = np.mean(np.abs(p_interp - q_interp) ** self.p) ** (1/self.p)

        return float(distance)

    def compute_discrete(
        self,
        weights_p: np.ndarray,
        weights_q: np.ndarray,
        locations_p: Optional[np.ndarray] = None,
        locations_q: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute Wasserstein distance between discrete distributions.

        Uses linear programming formulation:
        min Σᵢⱼ cᵢⱼ γᵢⱼ s.t. γ1 = p, γᵀ1 = q, γ ≥ 0
        """
        n = len(weights_p)
        m = len(weights_q)

        if locations_p is None:
            locations_p = np.arange(n, dtype=float)
        if locations_q is None:
            locations_q = np.arange(m, dtype=float)

        # Normalize weights
        weights_p = weights_p / (weights_p.sum() + 1e-10)
        weights_q = weights_q / (weights_q.sum() + 1e-10)

        # Cost matrix
        cost = np.abs(locations_p[:, None] - locations_q[None, :]) ** self.p

        # North-West corner method (approximate)
        transport = np.zeros((n, m))
        supply = weights_p.copy()
        demand = weights_q.copy()

        i, j = 0, 0
        while i < n and j < m:
            flow = min(supply[i], demand[j])
            transport[i, j] = flow
            supply[i] -= flow
            demand[j] -= flow

            if supply[i] < 1e-10:
                i += 1
            if demand[j] < 1e-10:
                j += 1

        # Total cost
        total_cost = np.sum(transport * cost)

        if self.p > 1:
            total_cost = total_cost ** (1/self.p)

        return float(total_cost), transport


class SinkhornDistance:
    """
    Sinkhorn (Entropic Regularized) Optimal Transport.

    Solves: min <γ,C> + εH(γ)
    Where H(γ) = -Σᵢⱼ γᵢⱼ(log γᵢⱼ - 1)

    Uses Sinkhorn-Knopp algorithm:
    K = exp(-C/ε)
    a ← p / (Kb)
    b ← q / (Kᵀa)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        threshold: float = 1e-6
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold

    def compute(
        self,
        weights_p: np.ndarray,
        weights_q: np.ndarray,
        cost_matrix: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute Sinkhorn distance.
        """
        n, m = cost_matrix.shape

        # Normalize
        p = weights_p / (weights_p.sum() + 1e-10)
        q = weights_q / (weights_q.sum() + 1e-10)

        # Gibbs kernel
        K = np.exp(-cost_matrix / self.epsilon)

        # Initialize scaling vectors
        a = np.ones(n)
        b = np.ones(m)

        for iteration in range(self.max_iter):
            a_prev = a.copy()

            # Update a
            Kb = K @ b
            a = p / (Kb + 1e-10)

            # Update b
            Ka = K.T @ a
            b = q / (Ka + 1e-10)

            # Check convergence
            if np.max(np.abs(a - a_prev)) < self.threshold:
                break

        # Transport plan
        transport = np.diag(a) @ K @ np.diag(b)

        # Sinkhorn distance (corrected for entropic regularization)
        sinkhorn_cost = np.sum(transport * cost_matrix)

        return float(sinkhorn_cost), transport


class SlicedWassersteinDistance:
    """
    Sliced Wasserstein Distance for high-dimensional distributions.

    SW(P,Q) = ∫ W₁(Proj_θP, Proj_θQ) dθ

    Projects onto random directions and averages 1D Wasserstein.
    """

    def __init__(
        self,
        n_projections: int = 50
    ):
        self.n_projections = n_projections
        self.wasserstein = WassersteinDistance(p=1)

    def compute(
        self,
        samples_p: np.ndarray,  # Shape: (n, d)
        samples_q: np.ndarray   # Shape: (m, d)
    ) -> float:
        """
        Compute sliced Wasserstein distance.
        """
        if samples_p.ndim == 1:
            samples_p = samples_p.reshape(-1, 1)
        if samples_q.ndim == 1:
            samples_q = samples_q.reshape(-1, 1)

        d = samples_p.shape[1]

        # Generate random projections
        projections = np.random.randn(self.n_projections, d)
        projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

        distances = []
        for proj in projections:
            # Project samples
            p_proj = samples_p @ proj
            q_proj = samples_q @ proj

            # 1D Wasserstein
            dist = self.wasserstein.compute_1d(p_proj, q_proj)
            distances.append(dist)

        return float(np.mean(distances))


class DistributionDriftDetector:
    """
    Detect distribution drift using optimal transport metrics.
    """

    def __init__(
        self,
        baseline_window: int = 60,
        current_window: int = 20,
        n_bins: int = 50
    ):
        self.baseline_window = baseline_window
        self.current_window = current_window
        self.n_bins = n_bins
        self.wasserstein = WassersteinDistance(p=1, n_bins=n_bins)
        self.sinkhorn = SinkhornDistance(epsilon=0.1)

    def detect(
        self,
        returns: np.ndarray
    ) -> OptimalTransportResult:
        """
        Detect distribution drift in return series.
        """
        total_needed = self.baseline_window + self.current_window

        if len(returns) < total_needed:
            return self._default_result()

        # Split into baseline and current
        baseline = returns[-total_needed:-self.current_window]
        current = returns[-self.current_window:]

        # Wasserstein distances
        w1 = self.wasserstein.compute_1d(baseline, current)

        w2_calc = WassersteinDistance(p=2, n_bins=self.n_bins)
        w2 = w2_calc.compute_1d(baseline, current)

        # Histogram-based Sinkhorn
        all_data = np.concatenate([baseline, current])
        bins = np.linspace(np.min(all_data) - 0.01, np.max(all_data) + 0.01, self.n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2

        hist_baseline, _ = np.histogram(baseline, bins=bins)
        hist_current, _ = np.histogram(current, bins=bins)

        # Cost matrix
        cost = np.abs(centers[:, None] - centers[None, :])

        sinkhorn_dist, transport = self.sinkhorn.compute(
            hist_baseline.astype(float) + 1e-10,
            hist_current.astype(float) + 1e-10,
            cost
        )

        # Marginal transport cost
        marginal_cost = np.sum(transport * cost, axis=1)

        # Regime shift score
        baseline_std = np.std(baseline)
        normalized_w1 = w1 / (baseline_std + 1e-10)
        shift_score = 1 - np.exp(-normalized_w1)

        # Drift direction
        mean_diff = np.mean(current) - np.mean(baseline)
        if mean_diff > 0.5 * baseline_std:
            drift_direction = "right"
        elif mean_diff < -0.5 * baseline_std:
            drift_direction = "left"
        else:
            drift_direction = "stable"

        # Interpretation
        interpretation = self._interpret_transport(
            baseline, current, transport, centers
        )

        return OptimalTransportResult(
            wasserstein_1=w1,
            wasserstein_2=w2,
            sinkhorn_distance=sinkhorn_dist,
            transport_plan=transport,
            marginal_cost=marginal_cost,
            regime_shift_score=float(shift_score),
            drift_direction=drift_direction,
            transport_interpretation=interpretation
        )

    def _interpret_transport(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        transport: np.ndarray,
        centers: np.ndarray
    ) -> Dict[str, Any]:
        """Interpret the transport plan."""

        # Mean and variance shifts
        mean_shift = np.mean(current) - np.mean(baseline)
        var_ratio = np.var(current) / (np.var(baseline) + 1e-10)

        # Tail changes
        baseline_left_tail = np.percentile(baseline, 5)
        current_left_tail = np.percentile(current, 5)
        baseline_right_tail = np.percentile(baseline, 95)
        current_right_tail = np.percentile(current, 95)

        left_tail_shift = current_left_tail - baseline_left_tail
        right_tail_shift = current_right_tail - baseline_right_tail

        # Concentration change
        baseline_iqr = np.percentile(baseline, 75) - np.percentile(baseline, 25)
        current_iqr = np.percentile(current, 75) - np.percentile(current, 25)
        concentration_change = (baseline_iqr - current_iqr) / (baseline_iqr + 1e-10)

        return {
            "mean_shift": float(mean_shift),
            "variance_ratio": float(var_ratio),
            "left_tail_shift": float(left_tail_shift),
            "right_tail_shift": float(right_tail_shift),
            "concentration_change": float(concentration_change),
            "is_variance_expanding": var_ratio > 1.2,
            "is_variance_contracting": var_ratio < 0.8,
            "tail_asymmetry": float(right_tail_shift - left_tail_shift)
        }

    def _default_result(self) -> OptimalTransportResult:
        return OptimalTransportResult(
            wasserstein_1=0.0,
            wasserstein_2=0.0,
            sinkhorn_distance=0.0,
            transport_plan=np.array([]),
            marginal_cost=np.array([]),
            regime_shift_score=0.0,
            drift_direction="stable",
            transport_interpretation={}
        )


class OptimalTransportEngine:
    """
    Unified Optimal Transport Engine.
    """

    def __init__(
        self,
        baseline_window: int = 60,
        current_window: int = 20
    ):
        self.drift_detector = DistributionDriftDetector(
            baseline_window=baseline_window,
            current_window=current_window
        )
        self.sliced_wasserstein = SlicedWassersteinDistance()

    def analyze(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive optimal transport analysis.
        """
        if len(returns) < 80:
            return self._default_result()

        # Basic drift detection
        drift_result = self.drift_detector.detect(returns)

        # Multi-scale analysis
        multi_scale_shifts = {}
        for window in [10, 20, 40]:
            if len(returns) >= 60 + window:
                detector = DistributionDriftDetector(
                    baseline_window=60,
                    current_window=window
                )
                result = detector.detect(returns)
                multi_scale_shifts[f"window_{window}"] = {
                    "wasserstein": result.wasserstein_1,
                    "shift_score": result.regime_shift_score
                }

        # Feature-space transport (if features available)
        if features is not None and len(features) == len(returns):
            feature_drift = self._analyze_feature_drift(features)
        else:
            feature_drift = {}

        return {
            "wasserstein_1": drift_result.wasserstein_1,
            "wasserstein_2": drift_result.wasserstein_2,
            "sinkhorn_distance": drift_result.sinkhorn_distance,
            "regime_shift_score": drift_result.regime_shift_score,
            "drift_direction": drift_result.drift_direction,
            "interpretation": drift_result.transport_interpretation,
            "multi_scale_shifts": multi_scale_shifts,
            "feature_drift": feature_drift
        }

    def _analyze_feature_drift(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Analyze drift in feature space."""
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n = features.shape[0]
        split = n // 2

        baseline = features[:split]
        current = features[split:]

        # Sliced Wasserstein in feature space
        sw_distance = self.sliced_wasserstein.compute(baseline, current)

        return {
            "sliced_wasserstein": sw_distance
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "wasserstein_1": 0.0,
            "wasserstein_2": 0.0,
            "sinkhorn_distance": 0.0,
            "regime_shift_score": 0.0,
            "drift_direction": "stable",
            "interpretation": {}
        }
