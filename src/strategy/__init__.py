"""
Strategy Module - Use-Case Specific Investment Strategies

Provides:
- StrategyManager: Unified strategy selection for all timeframes
- LongTermPredictor: 1-5 year fundamental-based predictions
- Multi-bagger detection: Find stocks with 5-10x potential
- Sector analysis: Best companies in each sector

Strategy Types:
- INTRADAY: Technical-heavy, same-day trades
- SWING: Multi-factor, 2-10 days
- POSITIONAL: Balanced, 2-8 weeks
- LONG_TERM: Fundamental-heavy, 1-5 years
- MULTI_BAGGER: Growth + Value, small-cap focused
"""

from .strategy_manager import (
    StrategyManager,
    StrategyType,
    RiskProfile,
    StrategyRecommendation,
    SectorAnalysis,
)

from .long_term_predictor import (
    LongTermPredictor,
    LongTermPrediction,
    InvestmentHorizon,
    CompetitiveMoat,
)

__all__ = [
    # Strategy Manager
    'StrategyManager',
    'StrategyType',
    'RiskProfile',
    'StrategyRecommendation',
    'SectorAnalysis',
    # Long-term Predictor
    'LongTermPredictor',
    'LongTermPrediction',
    'InvestmentHorizon',
    'CompetitiveMoat',
]
