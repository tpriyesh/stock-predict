"""
Core Production-Grade Components for Stock Prediction

This module contains bias-free, mathematically rigorous components:

ORIGINAL COMPONENTS:
- SafeFeatureScaler: Prevents data leakage in feature scaling
- DataHealthChecker: Validates data quality before prediction
- TransactionCosts: Realistic Indian market cost modeling
- PurgedKFoldCV: Time-series cross-validation with embargo
- AdaptiveWeightCalculator: Dynamic ensemble weight learning
- MetaLearnerStacker: ML-based optimal weight combination
- LogReturnsCalculator: Mathematically correct return calculations

NEW ROBUST COMPONENTS (90% Completeness):
- CalibratedScoringEngine: Empirical probability calibration
- EnhancedDataHealthChecker: Event-aware data validation
- RegimeAwarePositionSizer: Dynamic position sizing by market regime
- DeterministicSentimentAnalyzer: Reproducible news sentiment
- SurvivorshipBiasHandler: Bias-adjusted backtesting
- RobustBacktester: Correct Sharpe/drawdown calculations
- UnifiedRobustPredictor: Complete production-grade prediction system
"""

# Original components
from .safe_scaler import SafeFeatureScaler
from .data_health import DataHealthChecker, DataHealth
from .transaction_costs import TransactionCostCalculator, BrokerType
from .purged_cv import PurgedKFoldCV, WalkForwardCV, TimeSeriesEmbargo
from .adaptive_weights import AdaptiveWeightCalculator
from .meta_learner import MetaLearnerStacker
from .log_returns import LogReturnsCalculator

# New robust components
from .calibrated_scoring import (
    CalibratedScoringEngine,
    CalibratedScore,
    AdaptiveWeightManager,
    EmpiricalCalibrator,
    MonteCarloSimulator,
    ConfidenceInterval
)
from .enhanced_data_health import (
    EnhancedDataHealthChecker,
    ExtendedDataHealth,
    EventCalendar,
    EventType,
    UpcomingEvent
)
from .regime_position_sizing import (
    RegimeAwarePositionSizer,
    MarketRegimeDetector,
    MarketRegime,
    PositionSizeResult
)
from .deterministic_sentiment import (
    DeterministicSentimentAnalyzer,
    RuleBasedSentimentEngine,
    SentimentResult,
    get_sentiment_analyzer
)
from .survivorship import (
    SurvivorshipBiasHandler,
    DelistedStock,
    SurvivorshipAdjustment,
    calculate_bias_adjusted_metrics
)
from .robust_backtester import (
    RobustBacktester,
    RobustBacktestResults,
    TradeResult
)
from .unified_robust_predictor import (
    UnifiedRobustPredictor,
    RobustPrediction,
    create_predictor
)
from .market_timing import (
    MarketTimingEngine,
    MarketPhase,
    MarketContext,
    TradingDecision,
    get_market_timing_engine
)

__all__ = [
    # Original
    'SafeFeatureScaler',
    'DataHealthChecker',
    'DataHealth',
    'TransactionCostCalculator',
    'BrokerType',
    'PurgedKFoldCV',
    'WalkForwardCV',
    'TimeSeriesEmbargo',
    'AdaptiveWeightCalculator',
    'MetaLearnerStacker',
    'LogReturnsCalculator',

    # Calibrated Scoring
    'CalibratedScoringEngine',
    'CalibratedScore',
    'AdaptiveWeightManager',
    'EmpiricalCalibrator',
    'MonteCarloSimulator',
    'ConfidenceInterval',

    # Enhanced Data Health
    'EnhancedDataHealthChecker',
    'ExtendedDataHealth',
    'EventCalendar',
    'EventType',
    'UpcomingEvent',

    # Position Sizing
    'RegimeAwarePositionSizer',
    'MarketRegimeDetector',
    'MarketRegime',
    'PositionSizeResult',

    # Sentiment
    'DeterministicSentimentAnalyzer',
    'RuleBasedSentimentEngine',
    'SentimentResult',
    'get_sentiment_analyzer',

    # Survivorship
    'SurvivorshipBiasHandler',
    'DelistedStock',
    'SurvivorshipAdjustment',
    'calculate_bias_adjusted_metrics',

    # Backtesting
    'RobustBacktester',
    'RobustBacktestResults',
    'TradeResult',

    # Unified Predictor
    'UnifiedRobustPredictor',
    'RobustPrediction',
    'create_predictor',

    # Market Timing
    'MarketTimingEngine',
    'MarketPhase',
    'MarketContext',
    'TradingDecision',
    'get_market_timing_engine',
]
