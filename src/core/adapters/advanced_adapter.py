"""
Advanced math model adapter.

Wraps AdvancedPredictionEngine (PCA, Wavelet, Kalman, Markov, DQN, Factor).
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

# Import with fallback
try:
    from src.core.advanced_engine import (
        AdvancedPredictionEngine,
        AdvancedPredictionResult,
        PredictionSignal,
    )
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError:
    ADVANCED_ENGINE_AVAILABLE = False
    logger.warning("AdvancedPredictionEngine not available")


class AdvancedMathAdapter(BaseModelAdapter):
    """
    Adapter for the advanced mathematical prediction engine.

    Converts AdvancedPredictionResult output to StandardizedModelOutput.

    AdvancedPredictionResult fields used:
    - probability (0-1): Ensemble prediction
    - confidence (0-1): Prediction confidence
    - signal: PredictionSignal enum
    - validation_score: Multi-dimensional validation
    - model_contributions: Individual model outputs
    """

    def __init__(self, engine: Optional["AdvancedPredictionEngine"] = None):
        super().__init__()
        if ADVANCED_ENGINE_AVAILABLE:
            self.engine = engine or AdvancedPredictionEngine()
        else:
            self.engine = None

    @property
    def model_name(self) -> str:
        return "advanced"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from advanced math engine.

        kwargs:
            trade_type: TradeType for position sizing
        """
        if not ADVANCED_ENGINE_AVAILABLE or self.engine is None:
            return self._fallback_output(symbol, "Advanced engine not available")

        trade_type = kwargs.get('trade_type')

        # Get advanced prediction
        result: AdvancedPredictionResult = self.engine.predict(
            symbol=symbol,
            df=df,
            trade_type=trade_type
        )

        # probability is already 0-1
        p_buy = result.probability

        # Create detailed uncertainty from validation and model contributions
        if result.forecast_confidence_interval:
            lower, upper = result.forecast_confidence_interval
            uncertainty = ModelUncertainty(
                type=UncertaintyType.CONFIDENCE_INTERVAL,
                value=upper - lower,
                lower_bound=max(0, p_buy - (upper - lower) / 2),
                upper_bound=min(1, p_buy + (upper - lower) / 2)
            )
        else:
            uncertainty = ModelUncertainty.from_confidence(result.confidence)

        # Coverage based on validation score
        coverage = result.validation_score if result.validation_score else 0.5

        # Collect reasoning
        reasoning = []

        # Main signal
        signal_map = {
            PredictionSignal.STRONG_BUY: "STRONG BUY",
            PredictionSignal.BUY: "BUY",
            PredictionSignal.HOLD: "HOLD",
            PredictionSignal.SELL: "SELL",
            PredictionSignal.STRONG_SELL: "STRONG SELL",
            PredictionSignal.AVOID: "AVOID"
        }
        reasoning.append(f"Advanced: {signal_map.get(result.signal, result.signal.value)}")

        # Regime info
        if result.current_regime:
            reasoning.append(f"Regime: {result.current_regime.value} ({result.regime_probability:.0%})")

        # Kalman trend
        if result.kalman_trend:
            reasoning.append(f"Kalman: {result.kalman_trend} trend (conf: {result.kalman_confidence:.0%})")

        # Timeframe alignment
        if result.trend_alignment and result.trend_alignment > 0.7:
            reasoning.append(f"Strong timeframe alignment ({result.trend_alignment:.0%})")
        elif result.trend_alignment and result.trend_alignment < 0.3:
            reasoning.append(f"Conflicting timeframes ({result.trend_alignment:.0%})")

        # Optimal timeframe
        if result.optimal_timeframe:
            reasoning.append(f"Optimal: {result.optimal_timeframe.value} trades")

        # Factor analysis
        if result.systematic_risk_pct > 0.7:
            reasoning.append(f"High systematic risk ({result.systematic_risk_pct:.0%})")
        elif result.idiosyncratic_opportunity > 0.3:
            reasoning.append(f"Idiosyncratic opportunity ({result.idiosyncratic_opportunity:.0%})")

        # Wavelet signal
        if result.wavelet_signal:
            reasoning.append(f"Wavelet: {result.wavelet_signal}")

        # Position recommendation from DQN
        if result.position_action:
            reasoning.append(f"DQN position: {result.position_action}")

        # Model contributions
        if result.model_contributions:
            for contrib in result.model_contributions[:2]:
                if contrib.probability > 0.6:
                    reasoning.append(f"{contrib.model_name}: bullish ({contrib.probability:.0%})")
                elif contrib.probability < 0.4:
                    reasoning.append(f"{contrib.model_name}: bearish ({contrib.probability:.0%})")

        # Warnings
        warnings = []

        # Validation warnings
        if result.validation_warnings:
            warnings.extend(result.validation_warnings[:2])

        # Trade worthiness
        if not result.trade_worthy:
            warnings.append("Signal not trade-worthy per validation")

        # Risk metrics
        if result.var_95 and result.var_95 > 0.05:
            warnings.append(f"High VaR (95%): {result.var_95:.1%}")
        if result.max_drawdown_estimate and result.max_drawdown_estimate > 0.1:
            warnings.append(f"Max drawdown risk: {result.max_drawdown_estimate:.1%}")

        # Regime instability
        if result.regime_stability and result.regime_stability < 0.3:
            warnings.append("Unstable regime - increased uncertainty")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'probability': result.probability,
                'confidence': result.confidence,
                'signal': result.signal.value,
                'validation_score': result.validation_score,
                'regime': result.current_regime.value if result.current_regime else None,
                'regime_stability': result.regime_stability,
                'trend_alignment': result.trend_alignment,
                'kalman_trend': result.kalman_trend,
                'optimal_timeframe': result.optimal_timeframe.value if result.optimal_timeframe else None,
                'position_action': result.position_action,
                'trade_worthy': result.trade_worthy,
            }
        )
