"""
Regime model adapter.

Wraps HMMRegimeDetector (Hidden Markov Model regime detection).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.ml.hmm_regime_detector import HMMRegimeDetector, RegimeResult, MarketRegime


class RegimeModelAdapter(BaseModelAdapter):
    """
    Adapter for the HMM regime detection model.

    Converts RegimeResult output to StandardizedModelOutput.

    RegimeResult fields used:
    - probability_score (0-1): Contribution to BUY prediction
    - regime_probability (0-1): Confidence in detected regime
    - regime_stability (0-1): How stable current regime is
    - current_regime: MarketRegime enum
    - reason: Human-readable explanation
    """

    def __init__(self, detector: Optional[HMMRegimeDetector] = None):
        super().__init__()
        self.detector = detector or HMMRegimeDetector()

    @property
    def model_name(self) -> str:
        return "regime"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from HMM regime detector.
        """
        # Detect regime
        result: RegimeResult = self.detector.detect(df)

        # probability_score is already 0-1 (regime-adjusted BUY probability)
        p_buy = result.probability_score

        # Uncertainty based on regime stability and probability
        # More stable regime = lower uncertainty
        uncertainty_value = 1.0 - (result.regime_stability * result.regime_probability)
        uncertainty = ModelUncertainty.from_confidence(1.0 - uncertainty_value)

        # Coverage based on how well observations fit the HMM
        coverage = result.observation_score

        # Collect reasoning
        reasoning = []

        # Current regime
        regime = result.current_regime
        if regime == MarketRegime.TRENDING_BULL:
            reasoning.append(f"HMM: Bull market regime ({result.regime_probability:.0%} confidence)")
            reasoning.append(f"Regime duration: {result.regime_duration} days")
        elif regime == MarketRegime.TRENDING_BEAR:
            reasoning.append(f"HMM: Bear market regime ({result.regime_probability:.0%} confidence)")
        elif regime == MarketRegime.RANGING:
            reasoning.append(f"HMM: Ranging/sideways market ({result.regime_probability:.0%})")
        elif regime == MarketRegime.CHOPPY:
            reasoning.append(f"HMM: Choppy/volatile regime ({result.regime_probability:.0%})")
        elif regime == MarketRegime.TRANSITION:
            reasoning.append(f"HMM: Regime transition in progress")

        # Stability insight
        if result.regime_stability > 0.7:
            reasoning.append(f"Regime stable ({result.regime_stability:.0%})")
        elif result.regime_stability < 0.4:
            reasoning.append(f"Regime unstable, possible change")

        # Expected duration
        if result.expected_duration > 5:
            reasoning.append(f"Expected to continue ~{result.expected_duration:.0f} days")

        # Add signal
        reasoning.append(f"Signal: {result.signal}")

        # Warnings
        warnings = []
        if regime == MarketRegime.TRANSITION:
            warnings.append("Regime changing - increased uncertainty")
        if result.regime_stability < 0.3:
            warnings.append("Low regime stability - signal may flip")

        # Transition probabilities
        if result.transition_probs:
            highest_trans = max(result.transition_probs.items(), key=lambda x: x[1])
            if highest_trans[0] != regime.value and highest_trans[1] > 0.15:
                warnings.append(f"Possible transition to {highest_trans[0]} ({highest_trans[1]:.0%})")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'current_regime': regime.value,
                'regime_probability': result.regime_probability,
                'regime_stability': result.regime_stability,
                'regime_duration': result.regime_duration,
                'expected_duration': result.expected_duration,
                'signal': result.signal,
                'all_probabilities': result.all_probabilities,
                'transition_probs': result.transition_probs,
            }
        )
