"""
Probability calibration for confidence scores.
Ensures that "70% confidence" means ~70% historical win rate.
"""
from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from loguru import logger


class ProbabilityCalibrator:
    """
    Calibrates raw scores to true probabilities using historical data.

    Uses isotonic regression for non-parametric calibration.
    """

    def __init__(self):
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted = False

    def fit(
        self,
        predicted_scores: list[float],
        actual_outcomes: list[int]
    ) -> None:
        """
        Fit calibrator on historical predictions and outcomes.

        Args:
            predicted_scores: Raw confidence scores (0-1)
            actual_outcomes: 1 if trade was profitable, 0 otherwise
        """
        if len(predicted_scores) < 20:
            logger.warning("Insufficient data for calibration (need 20+ samples)")
            return

        X = np.array(predicted_scores)
        y = np.array(actual_outcomes)

        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self.calibrator.fit(X, y)
        self.is_fitted = True

        logger.info(f"Calibrator fitted on {len(X)} samples")

    def calibrate(self, score: float) -> float:
        """
        Calibrate a raw score to probability.

        Args:
            score: Raw confidence score (0-1)

        Returns:
            Calibrated probability (0-1)
        """
        if not self.is_fitted or self.calibrator is None:
            # Return raw score if not calibrated
            return score

        calibrated = self.calibrator.predict([score])[0]
        return float(calibrated)

    def calibrate_batch(self, scores: list[float]) -> list[float]:
        """Calibrate multiple scores."""
        return [self.calibrate(s) for s in scores]

    def get_reliability_diagram(
        self,
        predicted_scores: list[float],
        actual_outcomes: list[int],
        n_bins: int = 10
    ) -> dict:
        """
        Generate reliability diagram data for visualization.

        Args:
            predicted_scores: Predicted probabilities
            actual_outcomes: Actual binary outcomes

        Returns:
            Dict with bin data for plotting
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (np.array(predicted_scores) >= bins[i]) & \
                   (np.array(predicted_scores) < bins[i + 1])
            count = np.sum(mask)

            if count > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_accuracies.append(np.mean(np.array(actual_outcomes)[mask]))
                bin_counts.append(int(count))

        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'perfect_line': [0, 1]
        }

    def calculate_brier_score(
        self,
        predicted_scores: list[float],
        actual_outcomes: list[int]
    ) -> float:
        """
        Calculate Brier score (lower is better).

        Brier score = mean((predicted - actual)^2)
        Perfect = 0, Worst = 1
        """
        predictions = np.array(predicted_scores)
        actuals = np.array(actual_outcomes)

        return float(np.mean((predictions - actuals) ** 2))
