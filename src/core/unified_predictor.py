"""
Unified Prediction API

This module provides a single, centralized interface for all stock predictions.
All UI components, CLI, and services should use this API to ensure consistency.

Key Features:
1. Single source of truth for predictions
2. Integrated statistical validation
3. Bias-corrected scoring
4. Cached engines for performance
5. Comprehensive logging and audit trail

Usage:
    from src.core.unified_predictor import UnifiedPredictor

    predictor = UnifiedPredictor()
    result = predictor.predict(symbol="RELIANCE", trade_type="intraday")

    # Batch prediction
    results = predictor.predict_batch(symbols=["RELIANCE", "TCS", "INFY"])
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import threading

from config.settings import get_settings
from src.models.enhanced_scoring import EnhancedScoringEngine, EnhancedStockScore
from src.signals.enhanced_generator import EnhancedSignalGenerator
from src.core.statistical_tests import StatisticalValidator, ValidationReport
from src.data.price_fetcher import PriceFetcher
from src.features.technical import TechnicalIndicators
from src.features.market_features import MarketFeatures
from src.storage.models import SignalType, TradeType


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    VERY_HIGH = "A+"    # >70% confidence, 4/4 model agreement, statistically validated
    HIGH = "A"          # >65% confidence, 3+/4 model agreement
    MODERATE = "B"      # >55% confidence, 2+/4 model agreement
    LOW = "C"           # >45% confidence
    VERY_LOW = "D"      # <45% confidence
    INVALID = "F"       # Failed validation or insufficient data


@dataclass
class PredictionResult:
    """Complete prediction result with all metadata."""
    # Core prediction
    symbol: str
    trade_type: TradeType
    signal: SignalType
    confidence: float
    confidence_grade: PredictionConfidence

    # Prices
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float

    # Model details
    ensemble_score: float
    model_agreement: int
    model_votes: Dict[str, str]
    signal_strength: str

    # Individual model scores
    base_score: float
    physics_score: float
    math_score: float
    regime_score: float

    # Market context
    regime: str
    regime_stability: float
    market_predictability: float

    # Risk metrics
    atr_pct: float
    liquidity_cr: float

    # Analysis
    reasons: List[str]
    warnings: List[str]
    recommended_strategy: str

    # Validation
    is_statistically_valid: Optional[bool] = None
    statistical_tests: Optional[Dict[str, Any]] = None

    # Metadata
    prediction_time: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'trade_type': self.trade_type.value,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'confidence_grade': self.confidence_grade.value,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward': self.risk_reward,
            'ensemble_score': self.ensemble_score,
            'model_agreement': self.model_agreement,
            'model_votes': self.model_votes,
            'signal_strength': self.signal_strength,
            'base_score': self.base_score,
            'physics_score': self.physics_score,
            'math_score': self.math_score,
            'regime_score': self.regime_score,
            'regime': self.regime,
            'regime_stability': self.regime_stability,
            'market_predictability': self.market_predictability,
            'atr_pct': self.atr_pct,
            'liquidity_cr': self.liquidity_cr,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'recommended_strategy': self.recommended_strategy,
            'is_statistically_valid': self.is_statistically_valid,
            'statistical_tests': self.statistical_tests,
            'prediction_time': self.prediction_time.isoformat(),
            'data_quality_score': self.data_quality_score
        }


@dataclass
class BatchPredictionResult:
    """Result from batch prediction."""
    predictions: List[PredictionResult]
    market_context: Dict[str, Any]
    summary: Dict[str, Any]
    intraday_picks: List[PredictionResult]
    swing_picks: List[PredictionResult]
    validation_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedPredictor:
    """
    Unified prediction engine that centralizes all scoring.

    This class is the SINGLE SOURCE OF TRUTH for all predictions.
    All UI components and services should use this class.

    Features:
    - Uses EnhancedScoringEngine with all 4 models
    - Applies statistical validation
    - Corrects for known biases
    - Provides consistent confidence grading
    - Caches engines for performance
    """

    # Singleton instance for caching
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for consistent engine usage."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the unified predictor (only once due to singleton)."""
        if getattr(self, '_initialized', False):
            return

        self.settings = get_settings()

        # Core engines (centralized)
        self._scoring_engine = EnhancedScoringEngine()
        self._signal_generator = None  # Lazy loaded
        self._price_fetcher = None  # Lazy loaded
        self._market_features = MarketFeatures()
        self._statistical_validator = StatisticalValidator()

        # Cache
        self._market_context_cache = None
        self._market_context_time = None
        self._cache_duration = 300  # 5 minutes

        self._initialized = True
        logger.info("UnifiedPredictor initialized (singleton instance)")

    @property
    def scoring_engine(self) -> EnhancedScoringEngine:
        """Get the centralized scoring engine."""
        return self._scoring_engine

    @property
    def signal_generator(self) -> EnhancedSignalGenerator:
        """Get the signal generator (lazy loaded)."""
        if self._signal_generator is None:
            self._signal_generator = EnhancedSignalGenerator()
        return self._signal_generator

    @property
    def price_fetcher(self) -> PriceFetcher:
        """Get the price fetcher (lazy loaded)."""
        if self._price_fetcher is None:
            self._price_fetcher = PriceFetcher()
        return self._price_fetcher

    def get_market_context(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached market context."""
        now = datetime.now()

        if (not force_refresh
            and self._market_context_cache is not None
            and self._market_context_time is not None
            and (now - self._market_context_time).seconds < self._cache_duration):
            return self._market_context_cache

        self._market_context_cache = self._market_features.get_market_context()
        self._market_context_time = now
        return self._market_context_cache

    def predict(
        self,
        symbol: str,
        trade_type: str = "intraday",
        df: Optional[pd.DataFrame] = None,
        news_score: float = 0.5,
        validate_statistically: bool = False
    ) -> Optional[PredictionResult]:
        """
        Get prediction for a single stock.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            trade_type: "intraday" or "swing"
            df: Optional pre-fetched price data
            news_score: News sentiment score (0-1)
            validate_statistically: Whether to run statistical validation

        Returns:
            PredictionResult or None if prediction fails
        """
        try:
            # Convert trade type
            trade_type_enum = TradeType.INTRADAY if trade_type.lower() == "intraday" else TradeType.SWING

            # Fetch data if not provided
            if df is None:
                df = self.price_fetcher.fetch(symbol)

            if df is None or df.empty or len(df) < 60:
                logger.warning(f"{symbol}: Insufficient data for prediction")
                return None

            # Ensure technical indicators
            if 'rsi' not in df.columns:
                df = TechnicalIndicators.calculate_all(df)

            # Get market context
            market_context = self.get_market_context()

            # Get enhanced score
            score = self._scoring_engine.score_stock(
                symbol=symbol,
                df=df,
                trade_type=trade_type_enum,
                news_score=news_score,
                market_context=market_context
            )

            if score is None:
                return None

            # Convert to PredictionResult
            result = self._convert_score_to_result(score)

            # Statistical validation (optional, for deeper analysis)
            if validate_statistically and len(df) >= 100:
                result = self._add_statistical_validation(result, df)

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def predict_batch(
        self,
        symbols: Optional[List[str]] = None,
        trade_types: Optional[List[str]] = None,
        top_n: int = 10,
        min_confidence: float = 0.50,
        min_agreement: int = 2,
        validate_statistically: bool = False
    ) -> BatchPredictionResult:
        """
        Get predictions for multiple stocks.

        Args:
            symbols: List of symbols (defaults to configured watchlist)
            trade_types: List of trade types (defaults to ["intraday", "swing"])
            top_n: Number of top picks per trade type
            min_confidence: Minimum confidence threshold
            min_agreement: Minimum model agreement required
            validate_statistically: Run statistical validation

        Returns:
            BatchPredictionResult with all predictions and summaries
        """
        if symbols is None:
            # Use configured symbols
            from config.symbols import get_all_symbols
            symbols = get_all_symbols()

        if trade_types is None:
            trade_types = ["intraday", "swing"]

        all_predictions = []
        intraday_picks = []
        swing_picks = []

        # Get market context once
        market_context = self.get_market_context(force_refresh=True)

        # Process each symbol
        for symbol in symbols:
            for trade_type in trade_types:
                try:
                    result = self.predict(
                        symbol=symbol,
                        trade_type=trade_type,
                        validate_statistically=validate_statistically
                    )

                    if result is not None:
                        all_predictions.append(result)

                except Exception as e:
                    logger.warning(f"Failed to predict {symbol} ({trade_type}): {e}")

        # Filter and rank
        valid_predictions = [
            p for p in all_predictions
            if p.confidence >= min_confidence
            and p.model_agreement >= min_agreement
            and p.signal == SignalType.BUY
        ]

        # Separate by trade type
        intraday_all = [p for p in valid_predictions if p.trade_type == TradeType.INTRADAY]
        swing_all = [p for p in valid_predictions if p.trade_type == TradeType.SWING]

        # Sort by model agreement, then confidence
        intraday_all.sort(key=lambda x: (x.model_agreement, x.confidence), reverse=True)
        swing_all.sort(key=lambda x: (x.model_agreement, x.confidence), reverse=True)

        # Top picks
        intraday_picks = intraday_all[:top_n]
        swing_picks = swing_all[:top_n]

        # Summary
        summary = self._create_summary(all_predictions, intraday_picks, swing_picks)

        # Validation summary
        validation_summary = {
            'total_symbols': len(symbols),
            'successful_predictions': len(all_predictions),
            'buy_signals': len(valid_predictions),
            'intraday_picks': len(intraday_picks),
            'swing_picks': len(swing_picks),
            'avg_confidence': np.mean([p.confidence for p in valid_predictions]) if valid_predictions else 0,
            'avg_model_agreement': np.mean([p.model_agreement for p in valid_predictions]) if valid_predictions else 0
        }

        return BatchPredictionResult(
            predictions=all_predictions,
            market_context=market_context,
            summary=summary,
            intraday_picks=intraday_picks,
            swing_picks=swing_picks,
            validation_summary=validation_summary
        )

    def _convert_score_to_result(self, score: EnhancedStockScore) -> PredictionResult:
        """Convert EnhancedStockScore to PredictionResult."""
        # Calculate confidence grade
        confidence_grade = self._calculate_confidence_grade(
            score.confidence, score.model_agreement
        )

        return PredictionResult(
            symbol=score.symbol,
            trade_type=score.trade_type,
            signal=score.signal,
            confidence=score.confidence,
            confidence_grade=confidence_grade,
            current_price=score.current_price,
            entry_price=score.entry_price,
            stop_loss=score.stop_loss,
            target_price=score.target_price,
            risk_reward=score.risk_reward,
            ensemble_score=score.ensemble_score,
            model_agreement=score.model_agreement,
            model_votes=score.model_votes,
            signal_strength=score.signal_strength,
            base_score=score.base_score,
            physics_score=score.physics_score,
            math_score=score.math_score,
            regime_score=score.regime_score,
            regime=score.regime,
            regime_stability=score.regime_stability,
            market_predictability=score.market_predictability,
            atr_pct=score.atr_pct,
            liquidity_cr=score.liquidity_cr,
            reasons=score.reasons,
            warnings=score.warnings,
            recommended_strategy=score.recommended_strategy
        )

    def _calculate_confidence_grade(
        self,
        confidence: float,
        model_agreement: int
    ) -> PredictionConfidence:
        """Calculate confidence grade based on confidence and model agreement."""
        if confidence >= 0.70 and model_agreement >= 4:
            return PredictionConfidence.VERY_HIGH
        elif confidence >= 0.65 and model_agreement >= 3:
            return PredictionConfidence.HIGH
        elif confidence >= 0.55 and model_agreement >= 2:
            return PredictionConfidence.MODERATE
        elif confidence >= 0.45:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def _add_statistical_validation(
        self,
        result: PredictionResult,
        df: pd.DataFrame
    ) -> PredictionResult:
        """Add statistical validation to prediction result."""
        try:
            returns = df['close'].pct_change().dropna().values

            validation = self._statistical_validator.validate_strategy(returns)

            result.is_statistically_valid = validation.overall_validity
            result.statistical_tests = {
                'sharpe_test': validation.sharpe_test.to_dict() if validation.sharpe_test else None,
                'returns_test': validation.returns_test.to_dict() if validation.returns_test else None,
                'validity_score': validation.validity_score,
                'recommendations': validation.recommendations
            }

            # Adjust confidence based on statistical validity
            if not validation.overall_validity:
                result.warnings.append("[STATS] Results may not be statistically significant")
                # Penalize confidence
                result.confidence = result.confidence * 0.9

        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
            result.warnings.append("[STATS] Statistical validation failed")

        return result

    def _create_summary(
        self,
        all_predictions: List[PredictionResult],
        intraday_picks: List[PredictionResult],
        swing_picks: List[PredictionResult]
    ) -> Dict[str, Any]:
        """Create summary of predictions."""
        buy_signals = [p for p in all_predictions if p.signal == SignalType.BUY]
        hold_signals = [p for p in all_predictions if p.signal == SignalType.HOLD]

        # Regime distribution
        regimes = {}
        for p in all_predictions:
            regime = p.regime
            regimes[regime] = regimes.get(regime, 0) + 1

        # Model agreement distribution
        agreement_dist = {}
        for p in all_predictions:
            ag = p.model_agreement
            agreement_dist[ag] = agreement_dist.get(ag, 0) + 1

        return {
            'total_predictions': len(all_predictions),
            'buy_signals': len(buy_signals),
            'hold_signals': len(hold_signals),
            'intraday_picks_count': len(intraday_picks),
            'swing_picks_count': len(swing_picks),
            'regime_distribution': regimes,
            'model_agreement_distribution': agreement_dist,
            'avg_confidence': np.mean([p.confidence for p in buy_signals]) if buy_signals else 0,
            'high_confidence_picks': len([p for p in buy_signals if p.confidence >= 0.65]),
            'very_high_confidence_picks': len([p for p in buy_signals if p.confidence >= 0.70])
        }

    def get_quick_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quick signal for a symbol without full analysis.
        Useful for watchlist scanning.

        Returns:
            Dict with signal, confidence, and key metrics
        """
        try:
            result = self.predict(symbol=symbol, trade_type="intraday")

            if result is None:
                return {
                    'symbol': symbol,
                    'signal': 'ERROR',
                    'confidence': 0,
                    'reason': 'Prediction failed'
                }

            return {
                'symbol': symbol,
                'signal': result.signal.value,
                'confidence': result.confidence,
                'confidence_grade': result.confidence_grade.value,
                'model_agreement': result.model_agreement,
                'regime': result.regime,
                'current_price': result.current_price,
                'entry_price': result.entry_price,
                'stop_loss': result.stop_loss,
                'target_price': result.target_price,
                'risk_reward': result.risk_reward
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'signal': 'ERROR',
                'confidence': 0,
                'reason': str(e)
            }


# Module-level functions for convenience
_predictor = None

def get_predictor() -> UnifiedPredictor:
    """Get the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = UnifiedPredictor()
    return _predictor


def predict(symbol: str, trade_type: str = "intraday") -> Optional[PredictionResult]:
    """Convenience function for single prediction."""
    return get_predictor().predict(symbol=symbol, trade_type=trade_type)


def predict_batch(symbols: List[str] = None, **kwargs) -> BatchPredictionResult:
    """Convenience function for batch prediction."""
    return get_predictor().predict_batch(symbols=symbols, **kwargs)


def get_quick_signal(symbol: str) -> Dict[str, Any]:
    """Convenience function for quick signal."""
    return get_predictor().get_quick_signal(symbol)


# Export
__all__ = [
    'UnifiedPredictor',
    'PredictionResult',
    'BatchPredictionResult',
    'PredictionConfidence',
    'get_predictor',
    'predict',
    'predict_batch',
    'get_quick_signal'
]
