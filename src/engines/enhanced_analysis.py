"""
Enhanced Analysis Helper

Provides enhanced multi-model scoring that can be integrated
into existing analysis workflows. Adds 4-model ensemble scoring
to any stock analysis.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
from loguru import logger

from src.models.enhanced_scoring import EnhancedScoringEngine, EnhancedStockScore
from src.storage.models import TradeType


@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result to augment existing analysis."""
    symbol: str
    model_agreement: int  # 0-4, how many models agree
    model_votes: Dict[str, str]  # Each model's vote
    ensemble_score: float  # Combined score 0-1
    signal_strength: str  # 'strong', 'moderate', 'weak', 'none'
    regime: str  # Current market regime
    regime_stability: float  # How stable the regime is
    predictability: float  # Market predictability
    recommended_strategy: str  # 'momentum', 'reversion', etc.
    warnings: List[str]  # Risk warnings
    confidence_boost: float  # Adjustment to apply to base confidence


class EnhancedAnalysisHelper:
    """
    Helper to add enhanced multi-model scoring to any analysis.

    Usage:
        helper = EnhancedAnalysisHelper()
        enhanced = helper.get_enhanced_analysis(symbol, df, TradeType.INTRADAY)
        if enhanced.model_agreement >= 3:
            # High conviction signal
    """

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = EnhancedScoringEngine()
        return self._engine

    def get_enhanced_analysis(
        self,
        symbol: str,
        df: pd.DataFrame,
        trade_type: TradeType,
        news_score: float = 0.5
    ) -> Optional[EnhancedAnalysisResult]:
        """
        Get enhanced multi-model analysis for a stock.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data (needs 60+ days)
            trade_type: INTRADAY or SWING
            news_score: Pre-computed news sentiment (0-1)

        Returns:
            EnhancedAnalysisResult or None
        """
        try:
            score = self.engine.score_stock(
                symbol=symbol,
                df=df,
                trade_type=trade_type,
                news_score=news_score
            )

            if not score:
                return None

            # Calculate confidence boost based on model agreement
            # If 3+ models agree, boost confidence; if only 1, reduce
            agreement_boost = {
                4: 0.10,   # Strong boost for full agreement
                3: 0.05,   # Moderate boost
                2: 0.00,   # Neutral
                1: -0.05,  # Slight penalty
                0: -0.10   # Strong penalty
            }

            return EnhancedAnalysisResult(
                symbol=symbol,
                model_agreement=score.model_agreement,
                model_votes=score.model_votes,
                ensemble_score=score.ensemble_score,
                signal_strength=score.signal_strength,
                regime=score.regime,
                regime_stability=score.regime_stability,
                predictability=score.market_predictability,
                recommended_strategy=score.recommended_strategy,
                warnings=score.warnings,
                confidence_boost=agreement_boost.get(score.model_agreement, 0)
            )

        except Exception as e:
            logger.warning(f"Enhanced analysis failed for {symbol}: {e}")
            return None

    def get_model_agreement_display(self, result: EnhancedAnalysisResult) -> str:
        """Get a visual display of model agreement."""
        icons = []
        for model, vote in result.model_votes.items():
            if vote == 'BUY':
                icons.append(f"✅ {model.title()}")
            elif vote == 'SELL':
                icons.append(f"❌ {model.title()}")
            elif vote == 'AVOID':
                icons.append(f"⚠️ {model.title()}")
            else:
                icons.append(f"⚪ {model.title()}")
        return " | ".join(icons)

    def get_conviction_level(self, result: EnhancedAnalysisResult) -> Tuple[str, str]:
        """
        Get conviction level based on model agreement.

        Returns:
            (level, color) tuple for display
        """
        if result.model_agreement >= 4:
            return "VERY HIGH", "#00ff00"  # Bright green
        elif result.model_agreement >= 3:
            return "HIGH", "#00cc00"  # Green
        elif result.model_agreement >= 2:
            return "MODERATE", "#ffaa00"  # Yellow
        else:
            return "LOW", "#ff6600"  # Orange

    def should_show_warning(self, result: EnhancedAnalysisResult) -> bool:
        """Check if warnings should be prominently displayed."""
        return (
            result.regime == 'choppy' or
            result.regime_stability < 0.5 or
            result.predictability < 0.2 or
            len(result.warnings) > 2
        )


# Global helper instance
_helper = None


def get_enhanced_helper() -> EnhancedAnalysisHelper:
    """Get singleton enhanced analysis helper."""
    global _helper
    if _helper is None:
        _helper = EnhancedAnalysisHelper()
    return _helper


def enhance_stock_result(
    result: dict,
    df: pd.DataFrame,
    trade_type: TradeType
) -> dict:
    """
    Enhance an existing stock result dict with multi-model scoring.

    This can be called from UI to add enhanced data to existing analysis.

    Args:
        result: Existing result dict with 'symbol' key
        df: Price DataFrame for the stock
        trade_type: Trading type

    Returns:
        result dict with added 'enhanced' key containing EnhancedAnalysisResult data
    """
    helper = get_enhanced_helper()
    symbol = result.get('symbol', '')

    if not symbol or df is None or df.empty:
        return result

    enhanced = helper.get_enhanced_analysis(symbol, df, trade_type)

    if enhanced:
        result['enhanced'] = {
            'model_agreement': enhanced.model_agreement,
            'model_votes': enhanced.model_votes,
            'ensemble_score': enhanced.ensemble_score,
            'signal_strength': enhanced.signal_strength,
            'regime': enhanced.regime,
            'regime_stability': enhanced.regime_stability,
            'predictability': enhanced.predictability,
            'recommended_strategy': enhanced.recommended_strategy,
            'warnings': enhanced.warnings,
            'confidence_boost': enhanced.confidence_boost,
            'conviction_display': helper.get_model_agreement_display(enhanced),
            'conviction_level': helper.get_conviction_level(enhanced)[0],
            'show_warning': helper.should_show_warning(enhanced)
        }

    return result
