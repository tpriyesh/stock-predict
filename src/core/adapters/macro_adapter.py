"""
Macro model adapter.

Wraps MacroEngine (commodities, currencies, bonds, semiconductors).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.macro.macro_engine import MacroEngine, MacroSignal


class MacroModelAdapter(BaseModelAdapter):
    """
    Adapter for the macroeconomic analysis engine.

    Converts MacroSignal output to StandardizedModelOutput.

    MacroSignal fields used:
    - signal (-1 to +1): Macro impact on sector
    - confidence (0-1): Signal confidence
    - direction: 'bullish', 'bearish', 'neutral'
    - reasoning: List of reasons
    - component impacts: commodity, currency, bond, semiconductor
    """

    def __init__(self, engine: Optional[MacroEngine] = None):
        super().__init__()
        self.engine = engine or MacroEngine()

    @property
    def model_name(self) -> str:
        return "macro"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from macro engine.

        kwargs:
            sector: Stock's sector (required for macro analysis)
        """
        sector = kwargs.get('sector', 'Unknown')

        if sector == 'Unknown':
            return self._fallback_output(symbol, "Sector not provided for macro analysis")

        # Get sector macro signal
        signal: MacroSignal = self.engine.get_sector_macro_signal(sector)

        # Convert signal from [-1, +1] to [0, 1]
        p_buy = self.signal_to_probability(signal.signal)

        # Create uncertainty from confidence
        uncertainty = ModelUncertainty.from_confidence(signal.confidence)

        # Coverage is based on confidence
        coverage = signal.confidence

        # Collect reasoning
        reasoning = []

        # Direction
        if signal.direction == 'bullish':
            reasoning.append(f"Macro: Bullish for {sector} sector")
        elif signal.direction == 'bearish':
            reasoning.append(f"Macro: Bearish for {sector} sector")
        else:
            reasoning.append(f"Macro: Neutral impact on {sector}")

        # Add primary drivers
        if signal.primary_drivers:
            reasoning.append(f"Key drivers: {', '.join(signal.primary_drivers[:2])}")

        # Component breakdown
        components = []
        if abs(signal.commodity_impact) > 0.1:
            direction = "+" if signal.commodity_impact > 0 else "-"
            components.append(f"Commodities {direction}")
        if abs(signal.currency_impact) > 0.1:
            direction = "+" if signal.currency_impact > 0 else "-"
            components.append(f"Currency {direction}")
        if abs(signal.bond_impact) > 0.1:
            direction = "+" if signal.bond_impact > 0 else "-"
            components.append(f"Bonds {direction}")
        if abs(signal.semiconductor_impact) > 0.1:
            direction = "+" if signal.semiconductor_impact > 0 else "-"
            components.append(f"Semiconductors {direction}")

        if components:
            reasoning.append(f"Impacts: {', '.join(components)}")

        # Add original reasoning
        if signal.reasoning:
            reasoning.extend(signal.reasoning[:2])

        # Warnings
        warnings = []
        if signal.confidence < 0.5:
            warnings.append("Low confidence in macro signal")

        # Conflicting signals
        impacts = [
            signal.commodity_impact,
            signal.currency_impact,
            signal.bond_impact,
            signal.semiconductor_impact
        ]
        positive = sum(1 for i in impacts if i > 0.1)
        negative = sum(1 for i in impacts if i < -0.1)
        if positive > 0 and negative > 0:
            warnings.append("Mixed macro signals")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'signal': signal.signal,
                'confidence': signal.confidence,
                'direction': signal.direction,
                'sector': sector,
                'commodity_impact': signal.commodity_impact,
                'currency_impact': signal.currency_impact,
                'bond_impact': signal.bond_impact,
                'semiconductor_impact': signal.semiconductor_impact,
            }
        )
