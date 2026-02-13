"""
Regime-Specific Calibration

Maintains separate calibration curves per market regime.
This improves prediction accuracy by recognizing that
bull and bear market calibrations differ.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from loguru import logger

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, falling back to simple calibration")


class CalibrationRegime(Enum):
    """Regimes for calibration."""
    BULL = "bull"
    BEAR = "bear"
    RANGING = "ranging"
    CHOPPY = "choppy"
    HIGH_VOL = "high_volatility"
    UNKNOWN = "unknown"

    @classmethod
    def from_hmm_regime(cls, hmm_regime: str) -> "CalibrationRegime":
        """Map HMM regime to calibration regime."""
        mapping = {
            'trending_bull': cls.BULL,
            'trending_bear': cls.BEAR,
            'ranging': cls.RANGING,
            'choppy': cls.CHOPPY,
            'transition': cls.UNKNOWN,
        }
        return mapping.get(hmm_regime.lower(), cls.UNKNOWN)


@dataclass
class RegimeCalibrationData:
    """Calibration data for a single regime."""
    regime: CalibrationRegime
    history: List[Tuple[float, int]] = field(default_factory=list)  # (raw_score, outcome)
    n_samples: int = 0
    accuracy: float = 0.5
    last_fit: Optional[datetime] = None
    _calibrator: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'regime': self.regime.value,
            'history': self.history[-500:],  # Keep last 500
            'n_samples': self.n_samples,
            'accuracy': self.accuracy,
            'last_fit': self.last_fit.isoformat() if self.last_fit else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RegimeCalibrationData":
        """Create from dictionary."""
        return cls(
            regime=CalibrationRegime(d['regime']),
            history=d.get('history', []),
            n_samples=d.get('n_samples', 0),
            accuracy=d.get('accuracy', 0.5),
            last_fit=datetime.fromisoformat(d['last_fit']) if d.get('last_fit') else None,
        )


class RegimeSpecificCalibrator:
    """
    Maintains separate calibration curves per market regime.

    Each regime has its own IsotonicRegression model fitted
    on historical predictions made during that regime.

    Usage:
        calibrator = RegimeSpecificCalibrator()

        # Record outcome
        calibrator.record_outcome(raw_score=0.65, actual_outcome=1, regime="bull")

        # Calibrate a score
        calibrated = calibrator.calibrate(raw_score=0.70, regime="bull")

        # Get stats
        stats = calibrator.get_regime_stats()
    """

    def __init__(
        self,
        cache_dir: str = "data/calibration",
        min_regime_samples: int = 30,
        refit_interval: int = 50
    ):
        """
        Initialize regime-specific calibrator.

        Args:
            cache_dir: Directory for storing calibration data
            min_regime_samples: Minimum samples before using regime-specific calibration
            refit_interval: Refit calibrator every N samples
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.min_regime_samples = min_regime_samples
        self.refit_interval = refit_interval

        # Separate calibrator per regime
        self.calibrators: Dict[CalibrationRegime, RegimeCalibrationData] = {}
        self._initialize_calibrators()

        # Global calibrator (fallback)
        self.global_history: List[Tuple[float, int]] = []
        self._global_calibrator: Optional[object] = None
        self._global_accuracy: float = 0.5

    def _initialize_calibrators(self) -> None:
        """Initialize or load calibrators for each regime."""
        cache_file = self.cache_dir / "regime_calibration.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                for regime_data in data.get('regimes', []):
                    calib_data = RegimeCalibrationData.from_dict(regime_data)
                    self.calibrators[calib_data.regime] = calib_data
                    # Refit if we have enough data
                    if calib_data.n_samples >= self.min_regime_samples:
                        self._fit_regime_calibrator(calib_data.regime)

                self.global_history = data.get('global_history', [])[-500:]

                logger.info(f"Loaded calibration data for {len(self.calibrators)} regimes")
            except Exception as e:
                logger.warning(f"Failed to load calibration data: {e}")

        # Initialize missing regimes
        for regime in CalibrationRegime:
            if regime not in self.calibrators:
                self.calibrators[regime] = RegimeCalibrationData(regime=regime)

    def _save(self) -> None:
        """Save calibration data to cache."""
        cache_file = self.cache_dir / "regime_calibration.json"
        try:
            data = {
                'regimes': [d.to_dict() for d in self.calibrators.values()],
                'global_history': self.global_history[-500:],
                'last_updated': datetime.now().isoformat(),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")

    def record_outcome(
        self,
        raw_score: float,
        actual_outcome: int,
        regime: str
    ) -> None:
        """
        Record prediction outcome for a specific regime.

        Args:
            raw_score: Raw ensemble score (0-1)
            actual_outcome: 1 if prediction was correct, 0 otherwise
            regime: Market regime (string from HMM or CalibrationRegime value)
        """
        # Map to CalibrationRegime
        if isinstance(regime, str):
            calib_regime = CalibrationRegime.from_hmm_regime(regime)
        else:
            calib_regime = regime

        # Add to regime-specific history
        if calib_regime in self.calibrators:
            data = self.calibrators[calib_regime]
            data.history.append((raw_score, actual_outcome))
            data.n_samples += 1

            # Keep last 500 samples
            if len(data.history) > 500:
                data.history = data.history[-500:]

            # Refit if enough new samples
            if data.n_samples % self.refit_interval == 0:
                self._fit_regime_calibrator(calib_regime)

        # Also record globally
        self.global_history.append((raw_score, actual_outcome))
        if len(self.global_history) > 500:
            self.global_history = self.global_history[-500:]

        # Refit global periodically
        if len(self.global_history) % self.refit_interval == 0:
            self._fit_global_calibrator()

        self._save()

    def calibrate(
        self,
        raw_score: float,
        regime: str
    ) -> float:
        """
        Calibrate score using regime-specific model.

        Falls back to global if insufficient regime data.

        Args:
            raw_score: Raw ensemble score (0-1)
            regime: Market regime

        Returns:
            Calibrated probability (0-1)
        """
        # Map to CalibrationRegime
        if isinstance(regime, str):
            calib_regime = CalibrationRegime.from_hmm_regime(regime)
        else:
            calib_regime = regime

        # Try regime-specific first
        if calib_regime in self.calibrators:
            data = self.calibrators[calib_regime]

            if (data.n_samples >= self.min_regime_samples and
                data._calibrator is not None and
                SKLEARN_AVAILABLE):
                try:
                    return float(data._calibrator.predict([[raw_score]])[0])
                except Exception:
                    pass

        # Fallback to global
        return self._calibrate_global(raw_score)

    def _calibrate_global(self, raw_score: float) -> float:
        """Calibrate using global model."""
        if self._global_calibrator is not None and SKLEARN_AVAILABLE:
            try:
                return float(self._global_calibrator.predict([[raw_score]])[0])
            except Exception:
                pass

        # Simple shrinkage fallback
        return 0.5 + (raw_score - 0.5) * 0.7

    def _fit_regime_calibrator(self, regime: CalibrationRegime) -> None:
        """Fit isotonic regression for a specific regime."""
        if not SKLEARN_AVAILABLE:
            return

        data = self.calibrators.get(regime)
        if not data or len(data.history) < self.min_regime_samples:
            return

        try:
            scores = np.array([x[0] for x in data.history])
            outcomes = np.array([x[1] for x in data.history])

            calibrator = IsotonicRegression(
                y_min=0.05,
                y_max=0.95,
                out_of_bounds='clip'
            )
            calibrator.fit(scores.reshape(-1, 1), outcomes)

            # Calculate accuracy (calibration error)
            calibrated = calibrator.predict(scores.reshape(-1, 1))
            bins = np.linspace(0, 1, 11)
            accuracies = []

            for i in range(10):
                mask = (calibrated >= bins[i]) & (calibrated < bins[i+1])
                if mask.sum() > 3:
                    expected = calibrated[mask].mean()
                    actual = outcomes[mask].mean()
                    accuracies.append(1 - abs(expected - actual))

            data._calibrator = calibrator
            data.accuracy = np.mean(accuracies) if accuracies else 0.5
            data.last_fit = datetime.now()

            logger.debug(
                f"Fitted {regime.value} calibrator: "
                f"{data.n_samples} samples, accuracy {data.accuracy:.2f}"
            )

        except Exception as e:
            logger.warning(f"Failed to fit {regime.value} calibrator: {e}")

    def _fit_global_calibrator(self) -> None:
        """Fit global calibrator."""
        if not SKLEARN_AVAILABLE or len(self.global_history) < self.min_regime_samples:
            return

        try:
            scores = np.array([x[0] for x in self.global_history])
            outcomes = np.array([x[1] for x in self.global_history])

            self._global_calibrator = IsotonicRegression(
                y_min=0.05,
                y_max=0.95,
                out_of_bounds='clip'
            )
            self._global_calibrator.fit(scores.reshape(-1, 1), outcomes)

            # Calculate accuracy
            calibrated = self._global_calibrator.predict(scores.reshape(-1, 1))
            bins = np.linspace(0, 1, 11)
            accuracies = []

            for i in range(10):
                mask = (calibrated >= bins[i]) & (calibrated < bins[i+1])
                if mask.sum() > 3:
                    expected = calibrated[mask].mean()
                    actual = outcomes[mask].mean()
                    accuracies.append(1 - abs(expected - actual))

            self._global_accuracy = np.mean(accuracies) if accuracies else 0.5

        except Exception as e:
            logger.warning(f"Failed to fit global calibrator: {e}")

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Get calibration statistics per regime."""
        return {
            regime.value: {
                'n_samples': data.n_samples,
                'is_fitted': data._calibrator is not None,
                'accuracy': data.accuracy,
                'last_fit': data.last_fit.isoformat() if data.last_fit else None,
                'win_rate': (
                    sum(x[1] for x in data.history) / len(data.history)
                    if data.history else 0.5
                ),
            }
            for regime, data in self.calibrators.items()
        }

    def get_calibration_curve(
        self,
        regime: str,
        n_points: int = 20
    ) -> Tuple[List[float], List[float]]:
        """
        Get calibration curve for visualization.

        Args:
            regime: Market regime
            n_points: Number of points in curve

        Returns:
            Tuple of (raw_scores, calibrated_scores)
        """
        raw_scores = np.linspace(0.3, 0.8, n_points).tolist()
        calibrated_scores = [self.calibrate(s, regime) for s in raw_scores]
        return raw_scores, calibrated_scores

    def compare_regimes(
        self,
        raw_score: float
    ) -> Dict[str, float]:
        """
        Compare calibrated score across all regimes.

        Useful for understanding how regime affects confidence.
        """
        return {
            regime.value: self.calibrate(raw_score, regime.value)
            for regime in CalibrationRegime
            if regime != CalibrationRegime.UNKNOWN
        }


# Singleton instance
_regime_calibrator: Optional[RegimeSpecificCalibrator] = None


def get_regime_calibrator() -> RegimeSpecificCalibrator:
    """Get or create singleton regime calibrator."""
    global _regime_calibrator
    if _regime_calibrator is None:
        _regime_calibrator = RegimeSpecificCalibrator()
    return _regime_calibrator
