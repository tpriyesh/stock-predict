"""
Alternative data model adapter.

Wraps AlternativeDataEngine (earnings, options, institutional, confluence).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.alternative_data.alternative_engine import AlternativeDataEngine, AlternativeDataScore


class AlternativeDataAdapter(BaseModelAdapter):
    """
    Adapter for the alternative data engine.

    Converts AlternativeDataScore output to StandardizedModelOutput.

    AlternativeDataScore fields used:
    - composite_probability (0-1): Weighted combination of all sources
    - confidence (0-1): Signal confidence
    - direction: 1=bullish, -1=bearish, 0=neutral
    - Component probabilities: earnings, options, institutional, confluence
    - reasoning: List of reasons
    """

    def __init__(self, engine: Optional[AlternativeDataEngine] = None):
        super().__init__()
        self.engine = engine or AlternativeDataEngine()

    @property
    def model_name(self) -> str:
        return "alternative"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from alternative data engine.

        kwargs:
            include_options: Whether to include options analysis (default True)
        """
        include_options = kwargs.get('include_options', True)

        # Get alternative data score
        score: AlternativeDataScore = self.engine.analyze(
            symbol=symbol,
            df=df,
            include_options=include_options
        )

        # composite_probability is already 0-1
        p_buy = score.composite_probability

        # Create uncertainty from confidence
        uncertainty = ModelUncertainty.from_confidence(score.confidence)

        # Coverage based on how many sources contributed
        source_coverage = []
        if 0.3 < score.earnings_prob < 0.7:
            source_coverage.append(0.25)  # Neutral = no signal
        else:
            source_coverage.append(1.0)

        if include_options:
            if 0.3 < score.options_prob < 0.7:
                source_coverage.append(0.25)
            else:
                source_coverage.append(1.0)

        if 0.3 < score.institutional_prob < 0.7:
            source_coverage.append(0.25)
        else:
            source_coverage.append(1.0)

        if score.confluence_signal and 0.3 < score.confluence_prob < 0.7:
            source_coverage.append(0.25)
        elif score.confluence_signal:
            source_coverage.append(1.0)
        else:
            source_coverage.append(0.0)

        coverage = sum(source_coverage) / len(source_coverage) if source_coverage else 0.5

        # Collect reasoning
        reasoning = []

        # Signal strength
        strength_map = {
            'strong': 'STRONG',
            'moderate': 'MODERATE',
            'weak': 'WEAK',
            'neutral': 'NEUTRAL'
        }
        strength = strength_map.get(score.signal_strength, score.signal_strength)
        if score.direction == 1:
            reasoning.append(f"Alt Data: {strength} bullish signal")
        elif score.direction == -1:
            reasoning.append(f"Alt Data: {strength} bearish signal")
        else:
            reasoning.append(f"Alt Data: Neutral/no clear signal")

        # Source contributions
        if score.confluence_prob > 0.6:
            reasoning.append(f"Multi-timeframe confluence bullish ({score.confluence_prob:.0%})")
        elif score.confluence_prob < 0.4:
            reasoning.append(f"Multi-timeframe confluence bearish ({score.confluence_prob:.0%})")

        if score.earnings_prob > 0.6:
            reasoning.append(f"Positive earnings catalyst ({score.earnings_prob:.0%})")
        elif score.earnings_prob < 0.4:
            reasoning.append(f"Negative earnings risk ({score.earnings_prob:.0%})")

        if score.institutional_prob > 0.6:
            reasoning.append(f"Institutional accumulation ({score.institutional_prob:.0%})")
        elif score.institutional_prob < 0.4:
            reasoning.append(f"Institutional distribution ({score.institutional_prob:.0%})")

        if include_options and score.options_prob > 0.6:
            reasoning.append(f"Options flow bullish ({score.options_prob:.0%})")
        elif include_options and score.options_prob < 0.4:
            reasoning.append(f"Options flow bearish ({score.options_prob:.0%})")

        # Add original reasoning
        if score.reasoning:
            reasoning.extend(score.reasoning[:2])

        # Warnings
        warnings = []
        if score.confidence < 0.5:
            warnings.append("Low confidence in alternative data signal")

        # Check for conflicting sources
        probs = [score.earnings_prob, score.institutional_prob, score.confluence_prob]
        if include_options:
            probs.append(score.options_prob)
        bullish = sum(1 for p in probs if p > 0.6)
        bearish = sum(1 for p in probs if p < 0.4)
        if bullish > 0 and bearish > 0:
            warnings.append("Conflicting signals from alternative data sources")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'composite_probability': score.composite_probability,
                'confidence': score.confidence,
                'direction': score.direction,
                'signal_strength': score.signal_strength,
                'earnings_prob': score.earnings_prob,
                'options_prob': score.options_prob,
                'institutional_prob': score.institutional_prob,
                'confluence_prob': score.confluence_prob,
            }
        )
