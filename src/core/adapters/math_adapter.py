"""
Math model adapter.

Wraps MathEngine (Fourier, fractal, entropy, statistical mechanics).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
    UncertaintyType,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.math_models.math_engine import MathEngine, MathScore


class MathModelAdapter(BaseModelAdapter):
    """
    Adapter for the mathematical analysis engine.

    Converts MathScore output to StandardizedModelOutput.

    MathScore fields used:
    - composite_score (0-1): Weighted combination of 4 math models
    - predictability (0-1): Overall market predictability
    - market_character: 'trending', 'random', 'mean_reverting'
    - insights, warnings: Reasoning
    """

    def __init__(self, engine: Optional[MathEngine] = None):
        super().__init__()
        self.engine = engine or MathEngine()

    @property
    def model_name(self) -> str:
        return "math"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from math engine.

        kwargs:
            regime: Market regime for weight adjustment (default 'neutral')
        """
        regime = kwargs.get('regime', 'neutral')

        # Get score from engine
        score: MathScore = self.engine.score(
            symbol=symbol,
            df=df,
            regime=regime
        )

        # composite_score is already 0-1
        p_buy = score.composite_score

        # Use predictability for uncertainty
        # High predictability = low uncertainty
        uncertainty = ModelUncertainty(
            type=UncertaintyType.STANDARD_DEVIATION,
            value=1.0 - score.predictability,
            lower_bound=max(0, p_buy - (1 - score.predictability) / 2),
            upper_bound=min(1, p_buy + (1 - score.predictability) / 2)
        )

        # Coverage based on predictability
        # Low predictability means model is less confident in its analysis
        coverage = score.predictability

        # Collect reasoning
        reasoning = []

        # Market character insight
        if score.market_character == 'trending':
            reasoning.append(f"Math: Trending market detected (follow momentum)")
        elif score.market_character == 'mean_reverting':
            reasoning.append(f"Math: Mean-reverting dynamics (fade extremes)")
        elif score.market_character == 'random':
            reasoning.append(f"Math: Random walk behavior (low predictability)")

        # Strategy recommendation
        if score.recommended_strategy == 'momentum':
            reasoning.append("Fourier/Hurst: Trend continuation expected")
        elif score.recommended_strategy == 'reversion':
            reasoning.append("Entropy: Reversion setup identified")

        # Add insights
        if score.insights:
            reasoning.extend(score.insights[:2])

        # Add model-specific scores
        if score.fourier_score > 0.65:
            reasoning.append(f"Cycle analysis bullish ({score.fourier_score:.0%})")
        if score.fractal_score > 0.65:
            reasoning.append(f"Hurst exponent suggests trend ({score.fractal_score:.0%})")
        if score.entropy_score > 0.65:
            reasoning.append(f"Low entropy (predictable) ({score.entropy_score:.0%})")

        # Predictability warning
        warnings = list(score.warnings) if score.warnings else []
        if score.predictability < 0.3:
            warnings.append(f"Low predictability: {score.predictability:.0%}")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'fourier_score': score.fourier_score,
                'fractal_score': score.fractal_score,
                'entropy_score': score.entropy_score,
                'statmech_score': score.statmech_score,
                'market_character': score.market_character,
                'predictability': score.predictability,
                'recommended_strategy': score.recommended_strategy,
            }
        )
