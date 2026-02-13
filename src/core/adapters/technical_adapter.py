"""
Technical model adapter.

Wraps ScoringEngine (base technical/momentum/volume/news scoring).
"""

from typing import Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
)
from src.core.adapters.base_adapter import BaseModelAdapter
from src.models.scoring import ScoringEngine, StockScore
from src.storage.models import TradeType


class TechnicalModelAdapter(BaseModelAdapter):
    """
    Adapter for the base technical scoring engine.

    Converts StockScore output to StandardizedModelOutput.

    StockScore fields used:
    - raw_score (0-1): Base probability
    - confidence (0-1): Calibrated confidence
    - reasons: List of reasoning strings
    """

    def __init__(self, engine: Optional[ScoringEngine] = None):
        super().__init__()
        self.engine = engine or ScoringEngine()

    @property
    def model_name(self) -> str:
        return "technical"

    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate prediction from technical scoring engine.

        kwargs:
            trade_type: TradeType (default SWING)
            news_score: Pre-calculated news score (default 0.5)
            market_context: Market context dict
        """
        trade_type = kwargs.get('trade_type', TradeType.SWING)
        news_score = kwargs.get('news_score', 0.5)
        news_reasons = kwargs.get('news_reasons', [])
        market_context = kwargs.get('market_context')

        # Get score from engine
        score: Optional[StockScore] = self.engine.score_stock(
            symbol=symbol,
            df=df,
            trade_type=trade_type,
            news_score=news_score,
            news_reasons=news_reasons,
            market_context=market_context
        )

        if score is None:
            return self._fallback_output(symbol, "Scoring engine returned None")

        # Convert to standardized output
        # raw_score is already 0-1
        p_buy = score.raw_score

        # Create uncertainty from confidence
        # Higher confidence = lower uncertainty
        uncertainty = ModelUncertainty.from_confidence(score.confidence)

        # Coverage is 1.0 if we got all data, reduce if data issues
        coverage = 1.0
        if score.liquidity_cr < 5:  # Low liquidity reduces coverage
            coverage *= 0.8
        if score.atr_pct > 5:  # High volatility reduces coverage
            coverage *= 0.9

        # Collect reasoning
        reasoning = []
        if score.technical_score > 0.6:
            reasoning.append(f"Technical indicators bullish ({score.technical_score:.0%})")
        elif score.technical_score < 0.4:
            reasoning.append(f"Technical indicators bearish ({score.technical_score:.0%})")

        if score.momentum_score > 0.6:
            reasoning.append(f"Strong momentum ({score.momentum_score:.0%})")
        elif score.momentum_score < 0.4:
            reasoning.append(f"Weak momentum ({score.momentum_score:.0%})")

        if score.volume_score > 0.6:
            reasoning.append("Volume confirms move")

        # Add original reasons if available
        if score.reasons:
            reasoning.extend(score.reasons[:3])

        # Warnings
        warnings = []
        if score.atr_pct > 4:
            warnings.append(f"High volatility: {score.atr_pct:.1f}% ATR")
        if score.liquidity_cr < 10:
            warnings.append(f"Lower liquidity: Rs {score.liquidity_cr:.1f} Cr daily")

        return StandardizedModelOutput(
            p_buy=p_buy,
            uncertainty=uncertainty,
            coverage=coverage,
            reasoning=reasoning,
            model_name=self.model_name,
            warnings=warnings,
            raw_output={
                'technical_score': score.technical_score,
                'momentum_score': score.momentum_score,
                'news_score': score.news_score,
                'sector_score': score.sector_score,
                'volume_score': score.volume_score,
                'signal': score.signal.value,
                'entry_price': score.entry_price,
                'stop_loss': score.stop_loss,
                'target_price': score.target_price,
            }
        )
