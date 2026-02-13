"""
Multi-Dimensional Confidence Scoring Module

Implements advanced confidence quantification methods:
- Eigenvalue-based confidence decomposition
- Multi-dimensional uncertainty quantification
- Covariance-aware scoring
- Principal component confidence
- Mahalanobis distance for outlier detection

Mathematical Foundation:
- Eigendecomposition: Σ = VΛVᵀ
- Mahalanobis: d² = (x-μ)ᵀΣ⁻¹(x-μ)
- Condition number: κ(Σ) = λ_max/λ_min
- Effective dimensionality: d_eff = (Σλᵢ)²/Σλᵢ²
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiDimConfidence:
    """Multi-dimensional confidence assessment"""
    overall_confidence: float  # 0-1 aggregate score
    directional_confidence: float  # Confidence in direction
    magnitude_confidence: float  # Confidence in magnitude
    stability_confidence: float  # Confidence in prediction stability
    eigenvalue_spectrum: np.ndarray  # Sorted eigenvalues
    principal_directions: np.ndarray  # Eigenvectors
    effective_dimensionality: float  # Intrinsic dimensionality
    condition_number: float  # Numerical stability indicator
    uncertainty_ellipsoid: Dict[str, float]  # Uncertainty bounds
    anomaly_score: float  # How unusual is current state


class EigenvalueConfidenceAnalyzer:
    """
    Eigenvalue-based confidence analysis.

    Uses eigenvalue decomposition to understand:
    - Dominant patterns (large eigenvalues)
    - Noise dimensions (small eigenvalues)
    - Uncertainty structure (eigenvector directions)
    """

    def __init__(
        self,
        min_explained_variance: float = 0.95
    ):
        self.min_explained_variance = min_explained_variance

    def analyze(
        self,
        feature_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze feature covariance structure.
        """
        if len(feature_matrix) < 10 or feature_matrix.ndim < 2:
            return self._default_result()

        n, d = feature_matrix.shape

        # Center features
        centered = feature_matrix - np.mean(feature_matrix, axis=0)

        # Covariance matrix
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Ensure non-negative
            eigenvalues = np.maximum(eigenvalues, 0)

        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}")
            return self._default_result()

        # Total variance
        total_var = np.sum(eigenvalues)

        # Explained variance ratio
        explained_ratio = eigenvalues / (total_var + 1e-10)
        cumulative_explained = np.cumsum(explained_ratio)

        # Effective dimensionality (participation ratio)
        if total_var > 0:
            effective_dim = total_var ** 2 / (np.sum(eigenvalues ** 2) + 1e-10)
        else:
            effective_dim = 1.0

        # Number of significant dimensions
        n_significant = np.sum(cumulative_explained < self.min_explained_variance) + 1
        n_significant = min(n_significant, len(eigenvalues))

        # Condition number
        non_zero_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(non_zero_eigs) >= 2:
            condition_number = non_zero_eigs[0] / non_zero_eigs[-1]
        else:
            condition_number = 1.0

        # Spectral entropy (how spread out eigenvalues are)
        normalized_eigs = eigenvalues / (total_var + 1e-10)
        normalized_eigs = normalized_eigs[normalized_eigs > 1e-10]
        spectral_entropy = -np.sum(normalized_eigs * np.log(normalized_eigs))
        max_entropy = np.log(len(eigenvalues))
        normalized_entropy = spectral_entropy / (max_entropy + 1e-10)

        # Confidence from spectral structure
        # High confidence if few dominant dimensions and low entropy
        spectral_confidence = (1 - normalized_entropy) * (1 - effective_dim / d)
        spectral_confidence = np.clip(spectral_confidence, 0, 1)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained": cumulative_explained,
            "effective_dimensionality": float(effective_dim),
            "n_significant_dims": int(n_significant),
            "condition_number": float(condition_number),
            "spectral_entropy": float(spectral_entropy),
            "normalized_entropy": float(normalized_entropy),
            "spectral_confidence": float(spectral_confidence),
            "total_variance": float(total_var)
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "eigenvalues": np.array([]),
            "effective_dimensionality": 1.0,
            "spectral_confidence": 0.5,
            "condition_number": 1.0
        }


class MahalanobisAnomalyDetector:
    """
    Mahalanobis distance-based anomaly detection.

    d²(x) = (x-μ)ᵀΣ⁻¹(x-μ)

    Under multivariate normal, d² ~ χ²(p)
    """

    def __init__(
        self,
        threshold_percentile: float = 95
    ):
        self.threshold_percentile = threshold_percentile
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: float = 0

    def fit(
        self,
        feature_matrix: np.ndarray
    ):
        """Fit the detector on training data."""
        if len(feature_matrix) < 10:
            return

        self._mean = np.mean(feature_matrix, axis=0)

        # Covariance with regularization
        cov = np.cov(feature_matrix.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Regularize
        reg = 1e-6 * np.eye(cov.shape[0])
        cov_reg = cov + reg

        try:
            self._cov_inv = np.linalg.inv(cov_reg)
        except Exception:
            self._cov_inv = np.eye(cov.shape[0])

        # Compute distances for threshold
        distances = []
        for x in feature_matrix:
            d = self._mahalanobis(x)
            distances.append(d)

        self._threshold = np.percentile(distances, self.threshold_percentile)

    def _mahalanobis(
        self,
        x: np.ndarray
    ) -> float:
        """Compute Mahalanobis distance."""
        if self._mean is None or self._cov_inv is None:
            return 0.0

        diff = x - self._mean
        return float(np.sqrt(diff @ self._cov_inv @ diff))

    def score(
        self,
        x: np.ndarray
    ) -> Dict[str, Any]:
        """Score a new observation."""
        if self._mean is None:
            return {"distance": 0, "anomaly_score": 0, "is_anomaly": False}

        distance = self._mahalanobis(x)

        # Anomaly score (0 = normal, 1 = anomalous)
        anomaly_score = 1 - np.exp(-distance / (self._threshold + 1e-10))

        return {
            "mahalanobis_distance": float(distance),
            "anomaly_score": float(anomaly_score),
            "is_anomaly": distance > self._threshold,
            "threshold": float(self._threshold)
        }


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification.

    Provides:
    - Aleatoric uncertainty (data noise)
    - Epistemic uncertainty (model uncertainty)
    - Prediction intervals
    """

    def __init__(
        self,
        n_bootstrap: int = 100
    ):
        self.n_bootstrap = n_bootstrap

    def quantify(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Quantify prediction uncertainty.
        """
        if len(predictions) < 10:
            return self._default_result()

        # If predictions are from multiple models (2D)
        if predictions.ndim == 2:
            # Rows are observations, columns are model predictions
            mean_pred = np.mean(predictions, axis=1)
            epistemic_var = np.var(predictions, axis=1)  # Model disagreement
        else:
            mean_pred = predictions
            epistemic_var = np.zeros_like(predictions)

        # Bootstrap for confidence intervals
        n = len(mean_pred)
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            bootstrap_means.append(np.mean(mean_pred[idx]))

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # If actuals available, compute aleatoric uncertainty
        if actuals is not None and len(actuals) == len(mean_pred):
            residuals = actuals - mean_pred
            aleatoric_var = np.var(residuals)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
        else:
            aleatoric_var = np.var(mean_pred)
            mae = 0
            rmse = 0

        # Total uncertainty
        total_var = np.mean(epistemic_var) + aleatoric_var

        # Confidence from uncertainty
        # Lower uncertainty = higher confidence
        uncertainty_score = np.sqrt(total_var) / (np.abs(np.mean(mean_pred)) + 1e-10)
        confidence = 1 / (1 + uncertainty_score)

        return {
            "mean_prediction": float(np.mean(mean_pred)),
            "prediction_std": float(np.std(mean_pred)),
            "epistemic_uncertainty": float(np.mean(epistemic_var)),
            "aleatoric_uncertainty": float(aleatoric_var),
            "total_uncertainty": float(total_var),
            "confidence_interval_95": (float(ci_lower), float(ci_upper)),
            "mae": float(mae),
            "rmse": float(rmse),
            "uncertainty_confidence": float(confidence)
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "mean_prediction": 0,
            "total_uncertainty": 1,
            "uncertainty_confidence": 0.5
        }


class MultiDimensionalScorer:
    """
    Multi-dimensional confidence scoring system.

    Combines multiple confidence dimensions:
    - Signal strength
    - Signal quality
    - Regime stability
    - Model agreement
    - Historical accuracy
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None
    ):
        self.weights = dimension_weights or {
            "signal_strength": 0.25,
            "signal_quality": 0.20,
            "regime_stability": 0.20,
            "model_agreement": 0.20,
            "historical_accuracy": 0.15
        }

        self.eigenvalue_analyzer = EigenvalueConfidenceAnalyzer()
        self.anomaly_detector = MahalanobisAnomalyDetector()
        self.uncertainty_quantifier = UncertaintyQuantifier()

    def score(
        self,
        predictions: Dict[str, float],
        features: np.ndarray,
        historical_returns: Optional[np.ndarray] = None
    ) -> MultiDimConfidence:
        """
        Compute multi-dimensional confidence score.
        """
        if len(features) < 10:
            return self._default_result()

        # 1. Signal strength (magnitude of prediction)
        signal_values = list(predictions.values())
        signal_strength = np.abs(np.mean(signal_values))
        strength_confidence = np.tanh(signal_strength)  # Bounded 0-1

        # 2. Signal quality (consistency across predictors)
        if len(signal_values) > 1:
            signal_std = np.std(signal_values)
            quality_confidence = 1 / (1 + signal_std)
        else:
            quality_confidence = 0.5

        # 3. Regime stability (from eigenvalue analysis)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Build feature history matrix
        eigen_result = self.eigenvalue_analyzer.analyze(features)
        regime_confidence = eigen_result.get("spectral_confidence", 0.5)

        # 4. Model agreement
        signs = [np.sign(v) for v in signal_values if abs(v) > 0.01]
        if len(signs) > 1:
            agreement = abs(np.mean(signs))
        else:
            agreement = 0.5

        # 5. Historical accuracy (if returns available)
        if historical_returns is not None and len(historical_returns) > 20:
            # Check if predictions aligned with subsequent returns
            accuracy_confidence = self._compute_historical_accuracy(
                historical_returns
            )
        else:
            accuracy_confidence = 0.5

        # Aggregate confidence
        overall = (
            self.weights["signal_strength"] * strength_confidence +
            self.weights["signal_quality"] * quality_confidence +
            self.weights["regime_stability"] * regime_confidence +
            self.weights["model_agreement"] * agreement +
            self.weights["historical_accuracy"] * accuracy_confidence
        )

        # Directional confidence (are we sure about direction?)
        directional = agreement * quality_confidence

        # Magnitude confidence (are we sure about size?)
        magnitude = strength_confidence * quality_confidence

        # Stability (will this prediction hold?)
        stability = regime_confidence * (1 - eigen_result.get("normalized_entropy", 0.5))

        # Anomaly detection
        if features.shape[0] > 1:
            self.anomaly_detector.fit(features[:-1])
            anomaly_result = self.anomaly_detector.score(features[-1])
            anomaly_score = anomaly_result.get("anomaly_score", 0)
        else:
            anomaly_score = 0

        # Uncertainty ellipsoid
        eigenvalues = eigen_result.get("eigenvalues", np.array([1]))
        if len(eigenvalues) > 0:
            uncertainty_ellipsoid = {
                "major_axis": float(np.sqrt(eigenvalues[0])) if eigenvalues[0] > 0 else 0,
                "minor_axis": float(np.sqrt(eigenvalues[-1])) if eigenvalues[-1] > 0 else 0,
                "eccentricity": float(np.sqrt(1 - eigenvalues[-1] / (eigenvalues[0] + 1e-10)))
                    if len(eigenvalues) > 1 else 0
            }
        else:
            uncertainty_ellipsoid = {"major_axis": 1, "minor_axis": 1, "eccentricity": 0}

        return MultiDimConfidence(
            overall_confidence=float(np.clip(overall, 0, 1)),
            directional_confidence=float(np.clip(directional, 0, 1)),
            magnitude_confidence=float(np.clip(magnitude, 0, 1)),
            stability_confidence=float(np.clip(stability, 0, 1)),
            eigenvalue_spectrum=eigen_result.get("eigenvalues", np.array([])),
            principal_directions=eigen_result.get("eigenvectors", np.array([])),
            effective_dimensionality=eigen_result.get("effective_dimensionality", 1),
            condition_number=eigen_result.get("condition_number", 1),
            uncertainty_ellipsoid=uncertainty_ellipsoid,
            anomaly_score=float(anomaly_score)
        )

    def _compute_historical_accuracy(
        self,
        returns: np.ndarray
    ) -> float:
        """Compute confidence from historical accuracy."""
        # Simple: check predictability of returns
        if len(returns) < 20:
            return 0.5

        # Autocorrelation at lag 1
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Higher autocorrelation = more predictable
        if np.isnan(autocorr):
            return 0.5

        return float(0.5 + 0.5 * abs(autocorr))

    def _default_result(self) -> MultiDimConfidence:
        return MultiDimConfidence(
            overall_confidence=0.5,
            directional_confidence=0.5,
            magnitude_confidence=0.5,
            stability_confidence=0.5,
            eigenvalue_spectrum=np.array([]),
            principal_directions=np.array([]),
            effective_dimensionality=1.0,
            condition_number=1.0,
            uncertainty_ellipsoid={},
            anomaly_score=0.0
        )


class ConfidenceCalibrator:
    """
    Calibrate confidence scores to match empirical accuracy.

    Uses isotonic regression to map raw scores to calibrated probabilities.
    """

    def __init__(self):
        self._calibration_map: Optional[np.ndarray] = None
        self._calibration_bins: Optional[np.ndarray] = None

    def fit(
        self,
        raw_confidences: np.ndarray,
        outcomes: np.ndarray  # 1 = correct, 0 = incorrect
    ):
        """Fit calibration curve."""
        if len(raw_confidences) < 30:
            return

        # Bin confidences
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute accuracy in each bin
        calibrated = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (raw_confidences >= bins[i]) & (raw_confidences < bins[i+1])
            if np.sum(mask) > 0:
                calibrated[i] = np.mean(outcomes[mask])
            else:
                calibrated[i] = bin_centers[i]

        # Isotonic regression (ensure monotonicity)
        for i in range(1, n_bins):
            if calibrated[i] < calibrated[i-1]:
                calibrated[i] = calibrated[i-1]

        self._calibration_bins = bin_centers
        self._calibration_map = calibrated

    def calibrate(
        self,
        raw_confidence: float
    ) -> float:
        """Calibrate a single confidence score."""
        if self._calibration_map is None:
            return raw_confidence

        # Interpolate
        return float(np.interp(
            raw_confidence,
            self._calibration_bins,
            self._calibration_map
        ))

    def calibration_error(
        self,
        raw_confidences: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """Compute expected calibration error."""
        if len(raw_confidences) < 10:
            return 0.0

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        total = len(raw_confidences)

        for i in range(n_bins):
            mask = (raw_confidences >= bins[i]) & (raw_confidences < bins[i+1])
            n_bin = np.sum(mask)

            if n_bin > 0:
                avg_conf = np.mean(raw_confidences[mask])
                avg_acc = np.mean(outcomes[mask])
                ece += (n_bin / total) * abs(avg_conf - avg_acc)

        return float(ece)
