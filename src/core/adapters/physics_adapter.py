"""
Physics model adapter.

Wraps PhysicsEngine (momentum, spring, energy, network models).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.physics.physics_engine import PhysicsEngine, PhysicsScore


class PhysicsModelAdapter(BaseModelAdapter):
    """
    Adapter for the physics-inspired prediction engine.

    Converts PhysicsScore output to StandardizedModelOutput.

    PhysicsScore fields used:
    - composite_score (0-1): Weighted combination of 4 physics models
    - confidence (0-1): Model confidence
    - bullish_reasons, bearish_reasons: Reasoning
    - warnings: Model warnings
    """

    def __init__(self, engine: Optional[PhysicsEngine] = None):
        super().__init__()
        self.engine = engine or PhysicsEngine()

    @property
    def model_name(self) -> str:
        return "physics"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from physics engine.

        kwargs:
            regime: Market regime for weight adjustment (default 'neutral')
            sector_data: Dict of peer stock data for network analysis
        """
        regime = kwargs.get('regime', 'neutral')
        sector_data = kwargs.get('sector_data')

        # Get score from engine
        score: PhysicsScore = self.engine.score(
            symbol=symbol,
            df=df,
            regime=regime,
            sector_data=sector_data
        )

        # composite_score is already 0-1
        p_buy = score.composite_score

        # Create uncertainty from confidence
        uncertainty = ModelUncertainty.from_confidence(score.confidence)

        # Calculate coverage from individual model availability
        model_scores = [
            score.momentum_score,
            score.spring_score,
            score.energy_score,
            score.network_score
        ]
        # Coverage is 1.0 if all models contributed
        valid_models = sum(1 for s in model_scores if 0.1 < s < 0.9)
        coverage = valid_models / 4.0

        # Collect reasoning
        reasoning = []

        # Add strategy recommendation
        if score.recommended_strategy == 'momentum':
            reasoning.append("Physics: Momentum strategy favored")
        elif score.recommended_strategy == 'reversion':
            reasoning.append("Physics: Mean reversion setup")
        elif score.recommended_strategy == 'breakout':
            reasoning.append("Physics: Breakout potential detected")
        elif score.recommended_strategy == 'avoid':
            reasoning.append("Physics: High uncertainty, avoid")

        # Add bullish/bearish reasons
        if score.bullish_reasons:
            reasoning.extend([f"+ {r}" for r in score.bullish_reasons[:2]])
        if score.bearish_reasons:
            reasoning.extend([f"- {r}" for r in score.bearish_reasons[:2]])

        # Add model-specific insights
        if score.momentum_score > 0.65:
            reasoning.append(f"Momentum conservation strong ({score.momentum_score:.0%})")
        if score.spring_score > 0.65:
            reasoning.append(f"Spring reversion signal ({score.spring_score:.0%})")
        if score.energy_score > 0.65:
            reasoning.append(f"Energy clustering bullish ({score.energy_score:.0%})")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=score.warnings if score.warnings else [],
            raw_output={
                'momentum_score': score.momentum_score,
                'spring_score': score.spring_score,
                'energy_score': score.energy_score,
                'network_score': score.network_score,
                'recommended_strategy': score.recommended_strategy,
            }
        )
