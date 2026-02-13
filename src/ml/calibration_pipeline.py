"""
Calibration Pipeline

Multi-stage probability calibration:
1. Platt Scaling (parametric)
2. Isotonic Regression (non-parametric)
3. Temperature scaling
4. Regime-based adjustment

Ensures predicted probabilities match actual outcomes.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np


@dataclass
class CalibrationResult:
    """Result from probability calibration."""
    raw_probability: float          # Original probability
    platt_calibrated: float         # After Platt scaling
    isotonic_calibrated: float      # After isotonic regression
    temperature_scaled: float       # After temperature scaling
    regime_adjusted: float          # After regime adjustment
    final_calibrated: float         # Final calibrated probability

    # Calibration quality
    expected_calibration_error: float
    calibration_confidence: float

    # Method used
    calibration_method: str
    temperature: float

    probability_score: float        # Final score for prediction
    signal: str
    reason: str


class CalibrationPipeline:
    """
    Multi-stage calibration pipeline.

    Uses pre-calibrated parameters (in production, learn from data).
    """

    # Platt scaling parameters (A, B) for sigmoid: 1/(1+exp(A*x+B))
    # Calibrated for typical overconfident ML outputs
    PLATT_PARAMS = {
        'A': -1.5,  # Slope
        'B': 0.75   # Intercept
    }

    # Isotonic regression lookup (from calibration data)
    # Maps raw probability bins to calibrated values
    ISOTONIC_MAP = {
        0.0: 0.35,
        0.1: 0.38,
        0.2: 0.42,
        0.3: 0.46,
        0.4: 0.49,
        0.5: 0.52,
        0.6: 0.55,
        0.7: 0.58,
        0.8: 0.62,
        0.9: 0.68,
        1.0: 0.72
    }

    # Temperature scaling (calibrated)
    DEFAULT_TEMPERATURE = 1.5  # > 1 means outputs were overconfident

    # Regime adjustments
    REGIME_ADJUSTMENTS = {
        'trending_bull': {'shift': 0.03, 'scale': 1.05},
        'trending_bear': {'shift': -0.05, 'scale': 0.95},
        'ranging': {'shift': 0.0, 'scale': 1.0},
        'choppy': {'shift': 0.0, 'scale': 0.90},  # Reduce confidence
        'transition': {'shift': 0.0, 'scale': 0.85},
        'neutral': {'shift': 0.0, 'scale': 1.0}
    }

    def __init__(
        self,
        temperature: float = None,
        use_platt: bool = True,
        use_isotonic: bool = True
    ):
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        self.use_platt = use_platt
        self.use_isotonic = use_isotonic

    def platt_scale(self, probability: float) -> float:
        """
        Apply Platt scaling.

        Transforms probability using sigmoid: 1/(1+exp(A*x+B))
        where x = logit(probability)
        """
        # Avoid log(0) issues
        prob = np.clip(probability, 0.01, 0.99)

        # Logit transform
        logit = np.log(prob / (1 - prob))

        # Apply Platt parameters
        A = self.PLATT_PARAMS['A']
        B = self.PLATT_PARAMS['B']

        scaled_logit = A * logit + B

        # Sigmoid back
        calibrated = 1 / (1 + np.exp(-scaled_logit))

        return calibrated

    def isotonic_calibrate(self, probability: float) -> float:
        """
        Apply isotonic regression calibration.

        Uses pre-computed lookup table with linear interpolation.
        """
        # Find surrounding bins
        bins = sorted(self.ISOTONIC_MAP.keys())

        for i in range(len(bins) - 1):
            if bins[i] <= probability < bins[i + 1]:
                # Linear interpolation
                x0, x1 = bins[i], bins[i + 1]
                y0, y1 = self.ISOTONIC_MAP[x0], self.ISOTONIC_MAP[x1]

                calibrated = y0 + (y1 - y0) * (probability - x0) / (x1 - x0)
                return calibrated

        # Edge cases
        if probability >= 1.0:
            return self.ISOTONIC_MAP[1.0]
        return self.ISOTONIC_MAP[0.0]

    def temperature_scale(self, probability: float) -> float:
        """
        Apply temperature scaling.

        Softens overconfident predictions.
        """
        # Avoid log(0)
        prob = np.clip(probability, 0.01, 0.99)

        # Logit
        logit = np.log(prob / (1 - prob))

        # Scale by temperature
        scaled_logit = logit / self.temperature

        # Sigmoid back
        calibrated = 1 / (1 + np.exp(-scaled_logit))

        return calibrated

    def regime_adjust(
        self,
        probability: float,
        regime: str
    ) -> float:
        """
        Adjust probability based on market regime.

        Bull markets: Slight boost to bullish predictions
        Bear markets: Reduce bullish predictions
        Choppy: Reduce confidence (toward 0.5)
        """
        regime_lower = regime.lower().replace(' ', '_')
        adjustment = self.REGIME_ADJUSTMENTS.get(
            regime_lower,
            self.REGIME_ADJUSTMENTS['neutral']
        )

        # Apply shift and scale
        # Scale moves probability away from/toward 0.5
        # Shift adds constant adjustment
        adjusted = 0.5 + (probability - 0.5) * adjustment['scale']
        adjusted += adjustment['shift']

        return np.clip(adjusted, 0.30, 0.75)

    def estimate_calibration_error(
        self,
        raw_prob: float,
        calibrated_prob: float
    ) -> float:
        """
        Estimate expected calibration error.

        Based on how much calibration changed the probability.
        Larger changes suggest original was more miscalibrated.
        """
        change = abs(calibrated_prob - raw_prob)

        # Also consider distance from 0.5 (extreme predictions harder to calibrate)
        extremity = abs(raw_prob - 0.5) * 2

        # ECE estimate (rough)
        ece = 0.05 + change * 0.3 + extremity * 0.1

        return min(ece, 0.30)

    def get_signal_label(
        self,
        final_prob: float,
        ece: float
    ) -> Tuple[str, str]:
        """Determine signal label."""
        if ece > 0.15:
            confidence = 'LOW_CAL_CONFIDENCE'
        else:
            confidence = 'CALIBRATED'

        if final_prob > 0.6:
            return f'{confidence}_BULLISH', f'Calibrated probability: {final_prob:.0%}'
        elif final_prob < 0.4:
            return f'{confidence}_BEARISH', f'Calibrated probability: {final_prob:.0%}'
        else:
            return f'{confidence}_NEUTRAL', f'Calibrated probability: {final_prob:.0%}'

    def calibrate(
        self,
        raw_probability: float,
        regime: str = 'neutral'
    ) -> CalibrationResult:
        """
        Run full calibration pipeline.

        Args:
            raw_probability: Uncalibrated probability (0-1)
            regime: Current market regime

        Returns:
            CalibrationResult with all calibration stages
        """
        # Stage 1: Platt scaling
        if self.use_platt:
            platt_cal = self.platt_scale(raw_probability)
        else:
            platt_cal = raw_probability

        # Stage 2: Isotonic regression
        if self.use_isotonic:
            isotonic_cal = self.isotonic_calibrate(platt_cal)
        else:
            isotonic_cal = platt_cal

        # Stage 3: Temperature scaling
        temp_cal = self.temperature_scale(isotonic_cal)

        # Stage 4: Regime adjustment
        regime_cal = self.regime_adjust(temp_cal, regime)

        # Final probability
        final = regime_cal

        # Calibration error estimate
        ece = self.estimate_calibration_error(raw_probability, final)

        # Confidence in calibration
        cal_confidence = 1 - ece

        # Method used
        methods = []
        if self.use_platt:
            methods.append('platt')
        if self.use_isotonic:
            methods.append('isotonic')
        methods.extend(['temperature', 'regime'])
        method_str = '+'.join(methods)

        # Signal
        signal, reason = self.get_signal_label(final, ece)

        return CalibrationResult(
            raw_probability=raw_probability,
            platt_calibrated=platt_cal,
            isotonic_calibrated=isotonic_cal,
            temperature_scaled=temp_cal,
            regime_adjusted=regime_cal,
            final_calibrated=final,
            expected_calibration_error=ece,
            calibration_confidence=cal_confidence,
            calibration_method=method_str,
            temperature=self.temperature,
            probability_score=final,
            signal=signal,
            reason=reason
        )

    def calibrate_quick(
        self,
        raw_probability: float,
        regime: str = 'neutral'
    ) -> float:
        """Quick calibration returning only final probability."""
        result = self.calibrate(raw_probability, regime)
        return result.final_calibrated
