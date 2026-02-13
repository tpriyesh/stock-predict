"""
UI Prediction API

This module provides a centralized prediction interface for all UI components.
It ensures that all prediction requests go through the same engine, providing
consistent results across different tabs and features.

IMPORTANT: This is the ONLY module that UI components should use for predictions.
Do NOT instantiate scoring engines or signal generators directly in the UI.

The API now supports the Advanced Prediction Engine which integrates:
- PCA & Eigenvalue Decomposition
- Wavelet Multi-Resolution Analysis
- Kalman Filter State Estimation
- MLE-Calibrated Markov Regime
- Deep Q-Network Position Sizing
- Covariance Factor Model
- Advanced Statistical Validation

Usage:
    from src.core.ui_prediction_api import UIPredictionAPI

    api = UIPredictionAPI.get_instance()
    result = api.get_stock_prediction("RELIANCE", "intraday")
    batch = api.scan_stocks(symbols, min_confidence=0.55)

    # For advanced predictions:
    advanced = api.get_advanced_prediction("RELIANCE")
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from src.storage.models import SignalType, TradeType


class PredictionSource(Enum):
    """Source of prediction for transparency."""
    UNIFIED_PREDICTOR = "unified"      # Main 4-model ensemble
    ENHANCED_SCORING = "enhanced"       # Enhanced scoring engine
    QUICK_SCAN = "quick"                # Fast scan mode
    ADVANCED_ENGINE = "advanced"        # Advanced 6-model with PCA/Wavelet/Kalman/Markov/DQN/Factor
    UNIFIED_ADVANCED = "unified_advanced"  # Full 17-model with all advanced math


@dataclass
class UIStockPrediction:
    """
    Standardized prediction result for UI display.

    All UI components should use this format for consistency.
    """
    # Identity
    symbol: str
    trade_type: str  # "intraday" or "swing"

    # Signal
    signal: str  # "BUY", "HOLD", "SELL"
    signal_strength: str  # "strong", "moderate", "weak"
    confidence: float  # 0.0 to 1.0
    confidence_grade: str  # "A+", "A", "B", "C", "D", "F"

    # Prices
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float

    # Model details
    ensemble_score: float
    model_agreement: int  # 0-4 models agree
    model_votes: Dict[str, str]

    # Individual scores
    base_score: float
    physics_score: float
    math_score: float
    regime_score: float

    # Market context
    regime: str
    regime_stability: float
    market_predictability: float

    # Risk
    atr_pct: float
    liquidity_cr: float

    # Analysis
    reasons: List[str]
    warnings: List[str]
    recommended_strategy: str

    # Validation
    is_valid: bool = True
    robustness_level: str = "ACCEPTABLE"

    # Metadata
    source: PredictionSource = PredictionSource.UNIFIED_PREDICTOR
    timestamp: datetime = field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dict format suitable for UI display."""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
            'confidence_pct': f"{self.confidence * 100:.1f}%",
            'confidence_grade': self.confidence_grade,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward': self.risk_reward,
            'model_agreement': self.model_agreement,
            'model_votes': self.model_votes,
            'regime': self.regime,
            'atr_pct': self.atr_pct,
            'liquidity_cr': self.liquidity_cr,
            'reasons': self.reasons[:5],  # Limit for display
            'warnings': self.warnings[:3],
            'trade_type': self.trade_type,
            'is_valid': self.is_valid,
            'robustness': self.robustness_level
        }


@dataclass
class UIScanResult:
    """Result from batch stock scanning."""
    intraday_picks: List[UIStockPrediction]
    swing_picks: List[UIStockPrediction]
    all_predictions: List[UIStockPrediction]
    market_context: Dict[str, Any]
    summary: Dict[str, Any]
    scan_time: datetime = field(default_factory=datetime.now)

    @property
    def total_buy_signals(self) -> int:
        return len([p for p in self.all_predictions if p.signal == "BUY"])

    @property
    def high_confidence_picks(self) -> List[UIStockPrediction]:
        return [p for p in self.all_predictions if p.confidence >= 0.65 and p.signal == "BUY"]


@dataclass
class UIAdvancedPrediction:
    """
    Advanced prediction result with full multi-model analysis.

    This includes all the advanced mathematical models:
    - PCA decomposition
    - Wavelet multi-resolution
    - Kalman filter trend
    - MLE Markov regime
    - Factor model analysis
    - DQN position recommendation
    - Statistical validation
    """
    # Core
    symbol: str
    signal: str
    probability: float
    confidence: float

    # Model contributions (6 models)
    model_contributions: Dict[str, Dict[str, Any]]

    # Regime analysis
    current_regime: str
    regime_probability: float
    regime_stability: float

    # Multi-resolution
    trend_alignment: float
    optimal_timeframe: str
    wavelet_signal: str

    # Kalman
    kalman_trend: str
    kalman_slope: float
    price_forecast: List[float]

    # Factor
    systematic_risk_pct: float
    idiosyncratic_opportunity: float
    beta: float

    # Position
    position_action: str
    target_position_pct: float

    # Validation
    validation_score: float
    dimension_scores: Dict[str, float]
    validation_warnings: List[str]

    # Risk
    expected_volatility: float
    var_95: float

    # Trade
    trade_worthy: bool
    entry_condition: str
    stop_loss_pct: float
    target_pct: float
    risk_reward_ratio: float

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: PredictionSource = PredictionSource.ADVANCED_ENGINE

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dict for UI display."""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'probability': f"{self.probability:.1%}",
            'confidence': f"{self.confidence:.1%}",
            'regime': self.current_regime,
            'regime_stability': f"{self.regime_stability:.1%}",
            'trend_alignment': f"{self.trend_alignment:.1%}",
            'kalman_trend': self.kalman_trend,
            'validation_score': f"{self.validation_score:.1%}",
            'trade_worthy': self.trade_worthy,
            'position_action': self.position_action,
            'risk_reward': f"{self.risk_reward_ratio:.2f}",
            'warnings': self.validation_warnings[:3]
        }


@dataclass
class UIUnifiedAdvancedPrediction:
    """
    Unified Advanced prediction with ALL 17 mathematical models.

    This combines:
    - Original 6: PCA, Wavelet, Kalman, Markov, Factor Model, DQN
    - New 11: Quantum-Inspired, Bellman, SDE, Information Geometry,
              Tensor Decomposition, Spectral Graph, Optimal Transport,
              Taylor Series, Heavy-Tailed, Copula, Multi-Dim Confidence
    """
    # Core
    symbol: str
    final_signal: str
    final_probability: float
    final_confidence: float
    total_models_used: int

    # Base engine (6 models)
    base_signal: str
    base_probability: float
    base_confidence: float
    model_contributions: Dict[str, Dict[str, Any]]

    # Regime analysis
    current_regime: str
    regime_probability: float
    regime_stability: float

    # Multi-resolution (Wavelet)
    trend_alignment: float
    wavelet_signal: str

    # State estimation (Kalman)
    kalman_trend: str
    kalman_slope: float
    price_forecast: List[float]

    # Factor analysis
    systematic_risk_pct: float
    idiosyncratic_opportunity: float
    beta: float

    # Position (DQN)
    position_action: str
    target_position_pct: float

    # Validation
    validation_score: float
    dimension_scores: Dict[str, float]

    # Risk metrics
    expected_volatility: float
    var_95: float
    max_drawdown_estimate: float

    # Trade info
    trade_worthy: bool
    entry_condition: str
    stop_loss_pct: float
    target_pct: float
    risk_reward_ratio: float

    # Unified Advanced (11 new modules)
    unified_available: bool
    unified_signal: float
    unified_confidence: float
    unified_regime: str
    unified_action: str

    # Quantum analysis
    quantum_signal: float = 0.0
    quantum_confidence: float = 0.0

    # Bellman analysis
    optimal_action: str = "hold"
    bellman_value: float = 0.0

    # SDE analysis
    sde_drift: float = 0.0
    sde_volatility: float = 0.0
    jump_intensity: float = 0.0

    # Information geometry
    fisher_information: float = 0.0
    kl_divergence: float = 0.0

    # Heavy-tailed analysis
    tail_index: float = 2.0
    var_adjusted: float = 0.0

    # Copula
    tail_dependence: float = 0.0

    # Multi-dimensional confidence
    multidim_confidence: float = 0.5
    anomaly_score: float = 0.0

    # Human-readable insights
    insights: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: PredictionSource = PredictionSource.UNIFIED_ADVANCED

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dict for UI display."""
        return {
            'symbol': self.symbol,
            'final_signal': self.final_signal,
            'final_probability': f"{self.final_probability:.1%}",
            'final_confidence': f"{self.final_confidence:.1%}",
            'total_models': self.total_models_used,
            'base_signal': self.base_signal,
            'regime': self.current_regime,
            'regime_stability': f"{self.regime_stability:.1%}",
            'trend_alignment': f"{self.trend_alignment:.1%}",
            'kalman_trend': self.kalman_trend,
            'validation_score': f"{self.validation_score:.1%}",
            'trade_worthy': self.trade_worthy,
            'risk_reward': f"{self.risk_reward_ratio:.2f}",
            'unified_available': self.unified_available,
            'unified_signal': f"{self.unified_signal:.2f}",
            'quantum_signal': f"{self.quantum_signal:.2f}",
            'tail_index': f"{self.tail_index:.2f}",
            'anomaly_score': f"{self.anomaly_score:.2f}",
            'insights': self.insights[:5],
            'modules_breakdown': {
                'original_6': ['PCA', 'Wavelet', 'Kalman', 'Markov', 'Factor', 'DQN'],
                'advanced_11': [
                    'Quantum-Inspired', 'Bellman', 'SDE', 'InfoGeometry',
                    'Tensor', 'Spectral', 'OptimalTransport', 'Taylor',
                    'HeavyTailed', 'Copula', 'MultiDimConfidence'
                ] if self.unified_available else []
            }
        }

    def get_model_summary(self) -> str:
        """Get a text summary of the prediction."""
        return (
            f"{self.symbol}: {self.final_signal} ({self.final_probability:.1%} prob, "
            f"{self.final_confidence:.1%} conf) using {self.total_models_used} models. "
            f"{'TRADE WORTHY' if self.trade_worthy else 'NOT trade worthy'}. "
            f"Regime: {self.current_regime}, Trend alignment: {self.trend_alignment:.1%}"
        )


class UIPredictionAPI:
    """
    Centralized Prediction API for UI Components.

    This class is the SINGLE POINT OF ENTRY for all stock predictions in the UI.
    It ensures consistency by routing all requests through the same underlying engines.

    Thread-safe singleton pattern ensures consistent state across Streamlit sessions.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        # Lazy-loaded engines
        self._unified_predictor = None
        self._enhanced_scoring_engine = None
        self._advanced_engine = None
        self._price_fetcher = None
        self._market_features = None
        self._robustness_checker = None

        # Cache
        self._market_context_cache = None
        self._market_context_time = None
        self._cache_duration_seconds = 300  # 5 minutes

        self._initialized = True
        logger.info("UIPredictionAPI initialized (singleton)")

    @classmethod
    def get_instance(cls) -> 'UIPredictionAPI':
        """Get the singleton instance."""
        return cls()

    # ===== LAZY-LOADED ENGINES =====

    @property
    def unified_predictor(self):
        """Get the unified predictor (lazy-loaded)."""
        if self._unified_predictor is None:
            try:
                from src.core.unified_predictor import UnifiedPredictor
                self._unified_predictor = UnifiedPredictor()
            except ImportError as e:
                logger.warning(f"UnifiedPredictor not available: {e}")
                self._unified_predictor = None
        return self._unified_predictor

    @property
    def enhanced_scoring_engine(self):
        """Get enhanced scoring engine as fallback."""
        if self._enhanced_scoring_engine is None:
            try:
                from src.models.enhanced_scoring import EnhancedScoringEngine
                self._enhanced_scoring_engine = EnhancedScoringEngine()
            except ImportError as e:
                logger.warning(f"EnhancedScoringEngine not available: {e}")
        return self._enhanced_scoring_engine

    @property
    def advanced_engine(self):
        """Get advanced prediction engine with PCA/Wavelet/Kalman/Markov/DQN/Factor models."""
        if self._advanced_engine is None:
            try:
                from src.core.advanced_engine import get_advanced_engine
                self._advanced_engine = get_advanced_engine()
                logger.info("Advanced Prediction Engine loaded")
            except ImportError as e:
                logger.warning(f"AdvancedPredictionEngine not available: {e}")
        return self._advanced_engine

    @property
    def price_fetcher(self):
        """Get price fetcher."""
        if self._price_fetcher is None:
            try:
                from src.data.price_fetcher import PriceFetcher
                self._price_fetcher = PriceFetcher()
            except ImportError:
                pass
        return self._price_fetcher

    @property
    def market_features(self):
        """Get market features."""
        if self._market_features is None:
            try:
                from src.features.market_features import MarketFeatures
                self._market_features = MarketFeatures()
            except ImportError:
                pass
        return self._market_features

    @property
    def robustness_checker(self):
        """Get robustness checker."""
        if self._robustness_checker is None:
            try:
                from src.core.robustness_checks import RobustnessChecker
                self._robustness_checker = RobustnessChecker()
            except ImportError:
                pass
        return self._robustness_checker

    # ===== MAIN PREDICTION METHODS =====

    def get_stock_prediction(
        self,
        symbol: str,
        trade_type: str = "intraday",
        df: Optional[pd.DataFrame] = None,
        include_robustness: bool = False
    ) -> Optional[UIStockPrediction]:
        """
        Get prediction for a single stock.

        This is the PRIMARY method for getting stock predictions.
        All UI components should use this instead of direct engine calls.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            trade_type: "intraday" or "swing"
            df: Optional pre-fetched price data
            include_robustness: Whether to run robustness checks

        Returns:
            UIStockPrediction or None if prediction fails
        """
        try:
            # Try unified predictor first
            if self.unified_predictor:
                result = self.unified_predictor.predict(
                    symbol=symbol,
                    trade_type=trade_type,
                    df=df
                )

                if result:
                    prediction = self._convert_unified_result(result)

                    # Optional robustness check
                    if include_robustness and self.robustness_checker and df is not None:
                        robustness = self.robustness_checker.check_data_quality(df)
                        prediction.robustness_level = "ROBUST" if robustness.quality_score > 0.8 else "ACCEPTABLE"
                        prediction.is_valid = robustness.is_valid

                    return prediction

            # Fallback to enhanced scoring engine
            if self.enhanced_scoring_engine and df is not None:
                return self._get_prediction_from_enhanced(symbol, trade_type, df)

            return None

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def scan_stocks(
        self,
        symbols: Optional[List[str]] = None,
        min_confidence: float = 0.50,
        min_agreement: int = 2,
        top_n: int = 10,
        include_robustness: bool = False
    ) -> UIScanResult:
        """
        Scan multiple stocks and return top picks.

        This is the PRIMARY method for batch scanning.

        Args:
            symbols: List of symbols to scan (defaults to watchlist)
            min_confidence: Minimum confidence threshold
            min_agreement: Minimum model agreement
            top_n: Number of top picks per trade type
            include_robustness: Run robustness checks

        Returns:
            UIScanResult with all predictions and summaries
        """
        if symbols is None:
            try:
                from config.symbols import get_all_symbols
                symbols = get_all_symbols()
            except ImportError:
                symbols = []

        all_predictions = []
        intraday_picks = []
        swing_picks = []

        # Get market context once
        market_context = self.get_market_context()

        for symbol in symbols:
            # Intraday prediction
            intraday = self.get_stock_prediction(
                symbol=symbol,
                trade_type="intraday",
                include_robustness=include_robustness
            )
            if intraday:
                all_predictions.append(intraday)

            # Swing prediction
            swing = self.get_stock_prediction(
                symbol=symbol,
                trade_type="swing",
                include_robustness=include_robustness
            )
            if swing:
                all_predictions.append(swing)

        # Filter valid predictions
        valid_predictions = [
            p for p in all_predictions
            if p.signal == "BUY"
            and p.confidence >= min_confidence
            and p.model_agreement >= min_agreement
            and p.is_valid
        ]

        # Separate by trade type and sort
        intraday_all = [p for p in valid_predictions if p.trade_type == "intraday"]
        swing_all = [p for p in valid_predictions if p.trade_type == "swing"]

        intraday_all.sort(key=lambda x: (x.model_agreement, x.confidence), reverse=True)
        swing_all.sort(key=lambda x: (x.model_agreement, x.confidence), reverse=True)

        intraday_picks = intraday_all[:top_n]
        swing_picks = swing_all[:top_n]

        # Summary
        summary = {
            'total_scanned': len(symbols),
            'total_predictions': len(all_predictions),
            'buy_signals': len(valid_predictions),
            'intraday_picks': len(intraday_picks),
            'swing_picks': len(swing_picks),
            'avg_confidence': np.mean([p.confidence for p in valid_predictions]) if valid_predictions else 0,
            'high_confidence': len([p for p in valid_predictions if p.confidence >= 0.65])
        }

        return UIScanResult(
            intraday_picks=intraday_picks,
            swing_picks=swing_picks,
            all_predictions=all_predictions,
            market_context=market_context,
            summary=summary
        )

    def get_quick_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get quick signal for watchlist display.

        Returns minimal data for fast rendering.
        """
        try:
            if self.unified_predictor:
                return self.unified_predictor.get_quick_signal(symbol)

            # Fallback
            prediction = self.get_stock_prediction(symbol, "intraday")
            if prediction:
                return {
                    'symbol': symbol,
                    'signal': prediction.signal,
                    'confidence': prediction.confidence,
                    'model_agreement': prediction.model_agreement
                }

            return {'symbol': symbol, 'signal': 'ERROR', 'confidence': 0}

        except Exception as e:
            return {'symbol': symbol, 'signal': 'ERROR', 'confidence': 0, 'error': str(e)}

    def get_market_context(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached market context."""
        now = datetime.now()

        if (not force_refresh
            and self._market_context_cache is not None
            and self._market_context_time is not None
            and (now - self._market_context_time).seconds < self._cache_duration_seconds):
            return self._market_context_cache

        try:
            if self.market_features:
                self._market_context_cache = self.market_features.get_market_context()
                self._market_context_time = now
                return self._market_context_cache
        except Exception as e:
            logger.warning(f"Failed to get market context: {e}")

        return {}

    # ===== ADVANCED PREDICTION METHODS =====

    def get_advanced_prediction(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Optional[UIAdvancedPrediction]:
        """
        Get advanced multi-model prediction for a stock.

        This uses the advanced engine with:
        - PCA & Eigenvalue Decomposition
        - Wavelet Multi-Resolution Analysis
        - Kalman Filter State Estimation
        - MLE-Calibrated Markov Regime
        - Factor Model Analysis
        - Deep Q-Network Position Sizing
        - Advanced Statistical Validation

        Args:
            symbol: Stock symbol
            df: Optional pre-fetched price data with OHLCV
            benchmark_returns: Optional benchmark returns for factor model

        Returns:
            UIAdvancedPrediction with full analysis or None
        """
        try:
            # Fetch data if not provided
            if df is None:
                if self.price_fetcher:
                    df = self.price_fetcher.get_historical_data(symbol, period="1y")
                else:
                    logger.warning(f"No data available for {symbol}")
                    return None

            if df is None or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}: need 60+ rows")
                return None

            # Get advanced engine prediction
            if self.advanced_engine is None:
                logger.warning("Advanced engine not available")
                return None

            result = self.advanced_engine.predict(
                df=df,
                symbol=symbol,
                benchmark_returns=benchmark_returns
            )

            if result is None:
                return None

            # Convert to UI format
            return self._convert_advanced_result(symbol, result)

        except Exception as e:
            logger.error(f"Advanced prediction failed for {symbol}: {e}")
            return None

    def get_advanced_batch(
        self,
        symbols: List[str],
        min_probability: float = 0.55,
        top_n: int = 10
    ) -> List[UIAdvancedPrediction]:
        """
        Get advanced predictions for multiple stocks.

        Args:
            symbols: List of stock symbols
            min_probability: Minimum buy probability threshold
            top_n: Number of top picks to return

        Returns:
            List of UIAdvancedPrediction sorted by probability
        """
        predictions = []

        for symbol in symbols:
            try:
                pred = self.get_advanced_prediction(symbol)
                if pred and pred.probability >= min_probability:
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Advanced prediction failed for {symbol}: {e}")
                continue

        # Sort by probability and trade worthiness
        predictions.sort(
            key=lambda p: (p.trade_worthy, p.probability, p.validation_score),
            reverse=True
        )

        return predictions[:top_n]

    # ===== UNIFIED ADVANCED PREDICTION METHODS (17 models) =====

    def get_unified_advanced_prediction(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[UIUnifiedAdvancedPrediction]:
        """
        Get unified advanced prediction using ALL 17 mathematical models.

        This combines:
        - Original 6: PCA, Wavelet, Kalman, Markov, Factor Model, DQN
        - New 11: Quantum-Inspired, Bellman, SDE, Information Geometry,
                  Tensor Decomposition, Spectral Graph, Optimal Transport,
                  Taylor Series, Heavy-Tailed, Copula, Multi-Dim Confidence

        Args:
            symbol: Stock symbol
            df: Optional pre-fetched price data with OHLCV

        Returns:
            UIUnifiedAdvancedPrediction with full 17-model analysis or None
        """
        try:
            # Fetch data if not provided
            if df is None:
                if self.price_fetcher:
                    df = self.price_fetcher.get_historical_data(symbol, period="1y")
                else:
                    logger.warning(f"No data available for {symbol}")
                    return None

            if df is None or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}: need 60+ rows")
                return None

            # Get advanced engine
            if self.advanced_engine is None:
                logger.warning("Advanced engine not available")
                return None

            # Use the new unified prediction method
            result = self.advanced_engine.predict_with_unified_math(
                prices=df,
                symbol=symbol
            )

            if result is None:
                return None

            # Get insights from unified analysis summary
            summary = self.advanced_engine.get_unified_analysis_summary(df, symbol)
            insights = summary.get('insights', [])

            # Extract unified advanced data
            unified = result.get('unified_advanced', {})

            return UIUnifiedAdvancedPrediction(
                symbol=symbol,
                final_signal=result.get('final_signal', 'HOLD'),
                final_probability=result.get('final_probability', 0.5),
                final_confidence=result.get('final_confidence', 0.0),
                total_models_used=result.get('total_models_used', 6),
                base_signal=result.get('base_signal', 'HOLD'),
                base_probability=result.get('base_probability', 0.5),
                base_confidence=result.get('base_confidence', 0.0),
                model_contributions=result.get('model_contributions', {}),
                current_regime=result.get('regime', 'UNKNOWN'),
                regime_probability=result.get('regime_probability', 0.5),
                regime_stability=result.get('regime_stability', 0.5),
                trend_alignment=result.get('trend_alignment', 0.5),
                wavelet_signal=result.get('wavelet_signal', 'HOLD'),
                kalman_trend=result.get('kalman_trend', 'neutral'),
                kalman_slope=result.get('kalman_slope', 0.0),
                price_forecast=result.get('price_forecast', []),
                systematic_risk_pct=result.get('systematic_risk_pct', 0.5),
                idiosyncratic_opportunity=result.get('idiosyncratic_opportunity', 0.5),
                beta=result.get('beta', 1.0),
                position_action=result.get('position_action', 'HOLD'),
                target_position_pct=result.get('target_position_pct', 0.0),
                validation_score=result.get('validation_score', 0.5),
                dimension_scores=result.get('dimension_scores', {}),
                expected_volatility=result.get('expected_volatility', 0.02),
                var_95=result.get('var_95', 0.03),
                max_drawdown_estimate=result.get('max_drawdown_estimate', 0.1),
                trade_worthy=result.get('trade_worthy', False),
                entry_condition=result.get('entry_condition', 'Wait'),
                stop_loss_pct=result.get('stop_loss_pct', 0.0),
                target_pct=result.get('target_pct', 0.0),
                risk_reward_ratio=result.get('risk_reward_ratio', 0.0),
                unified_available=unified.get('available', False),
                unified_signal=unified.get('signal', 0.0),
                unified_confidence=unified.get('confidence', 0.0),
                unified_regime=unified.get('regime', 'unknown'),
                unified_action=unified.get('action', 'hold'),
                quantum_signal=unified.get('quantum_signal', 0.0),
                quantum_confidence=unified.get('quantum_confidence', 0.0),
                optimal_action=unified.get('optimal_action', 'hold'),
                bellman_value=unified.get('bellman_value', 0.0),
                sde_drift=unified.get('sde_drift', 0.0),
                sde_volatility=unified.get('sde_volatility', 0.0),
                jump_intensity=unified.get('jump_intensity', 0.0),
                fisher_information=unified.get('fisher_information', 0.0),
                kl_divergence=unified.get('kl_divergence', 0.0),
                tail_index=unified.get('tail_index', 2.0),
                var_adjusted=unified.get('var_adjusted', 0.0),
                tail_dependence=unified.get('tail_dependence', 0.0),
                multidim_confidence=unified.get('multidim_confidence', 0.5),
                anomaly_score=unified.get('anomaly_score', 0.0),
                insights=insights,
                source=PredictionSource.UNIFIED_ADVANCED
            )

        except Exception as e:
            logger.error(f"Unified advanced prediction failed for {symbol}: {e}")
            return None

    def get_unified_advanced_batch(
        self,
        symbols: List[str],
        min_probability: float = 0.55,
        top_n: int = 10
    ) -> List[UIUnifiedAdvancedPrediction]:
        """
        Get unified advanced predictions for multiple stocks.

        Uses all 17 mathematical models for comprehensive analysis.

        Args:
            symbols: List of stock symbols
            min_probability: Minimum final probability threshold
            top_n: Number of top picks to return

        Returns:
            List of UIUnifiedAdvancedPrediction sorted by probability
        """
        predictions = []

        for symbol in symbols:
            try:
                pred = self.get_unified_advanced_prediction(symbol)
                if pred and pred.final_probability >= min_probability:
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Unified advanced prediction failed for {symbol}: {e}")
                continue

        # Sort by final probability, trade worthiness, and total models
        predictions.sort(
            key=lambda p: (p.trade_worthy, p.final_probability, p.total_models_used),
            reverse=True
        )

        return predictions[:top_n]

    def get_advanced_math_insights(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get human-readable insights from the 11 advanced mathematical modules.

        Returns a summary of findings from:
        - Quantum-inspired optimization
        - Bellman optimal stopping
        - Stochastic differential equations
        - Information geometry
        - Tensor decomposition
        - Spectral graph theory
        - Optimal transport
        - Taylor series expansions
        - Heavy-tailed distributions
        - Copula models
        - Multi-dimensional confidence

        Args:
            symbol: Stock symbol
            df: Optional pre-fetched price data

        Returns:
            Dictionary with insights and module summaries
        """
        try:
            if df is None:
                if self.price_fetcher:
                    df = self.price_fetcher.get_historical_data(symbol, period="1y")
                else:
                    return {'available': False, 'message': 'No data available'}

            if df is None or len(df) < 60:
                return {'available': False, 'message': 'Insufficient data'}

            if self.advanced_engine is None:
                return {'available': False, 'message': 'Advanced engine not available'}

            # Get unified analysis summary
            summary = self.advanced_engine.get_unified_analysis_summary(df, symbol)

            return summary

        except Exception as e:
            logger.error(f"Failed to get advanced math insights for {symbol}: {e}")
            return {'available': False, 'message': str(e)}

    def _convert_advanced_result(self, symbol: str, result) -> UIAdvancedPrediction:
        """Convert AdvancedPredictionResult to UIAdvancedPrediction."""
        # Determine signal from probability
        if result.final_probability > 0.55:
            signal = "BUY"
        elif result.final_probability < 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Build model contributions dict
        model_contributions = {}
        for model_name, weight in result.model_weights.items():
            model_contributions[model_name] = {
                'weight': weight,
                'contribution': weight * result.final_probability
            }

        # Calculate validation score from dimensions
        dimension_scores = {}
        validation_warnings = []

        if result.validation_report:
            dimension_scores = result.validation_report.dimension_scores
            # Generate warnings for low scores
            for dim, score in dimension_scores.items():
                if score < 0.5:
                    validation_warnings.append(f"Low {dim} score: {score:.1%}")

        validation_score = np.mean(list(dimension_scores.values())) if dimension_scores else 0.5

        # Extract regime info
        regime_probs = result.regime_probabilities
        if regime_probs:
            current_regime = max(regime_probs, key=regime_probs.get)
            regime_probability = regime_probs[current_regime]
            regime_stability = result.regime_statistics.regime_stability if result.regime_statistics else 0.5
        else:
            current_regime = "UNKNOWN"
            regime_probability = 0.5
            regime_stability = 0.5

        # Extract wavelet info
        trend_alignment = result.wavelet_result.trend_alignment if result.wavelet_result else 0.5
        if result.wavelet_result:
            if trend_alignment > 0.7:
                optimal_timeframe = "ALIGNED"
            elif trend_alignment > 0.5:
                optimal_timeframe = "MODERATE"
            else:
                optimal_timeframe = "CONFLICTING"
            wavelet_signal = result.wavelet_result.primary_signal if hasattr(result.wavelet_result, 'primary_signal') else signal
        else:
            optimal_timeframe = "UNKNOWN"
            wavelet_signal = signal

        # Extract Kalman info
        if result.kalman_result:
            kalman_slope = result.kalman_result.slope
            kalman_trend = "UP" if kalman_slope > 0 else "DOWN"
            price_forecast = result.kalman_result.forecast_prices[:5] if hasattr(result.kalman_result, 'forecast_prices') else []
        else:
            kalman_trend = "NEUTRAL"
            kalman_slope = 0.0
            price_forecast = []

        # Extract factor model info
        if result.factor_result:
            systematic_risk_pct = result.factor_result.systematic_pct
            idiosyncratic_opportunity = 1.0 - systematic_risk_pct
            beta = result.factor_result.beta
        else:
            systematic_risk_pct = 0.5
            idiosyncratic_opportunity = 0.5
            beta = 1.0

        # Extract DQN info
        if result.position_recommendation:
            pos_rec = result.position_recommendation
            position_action = pos_rec.action
            target_position_pct = pos_rec.position_size
        else:
            position_action = "HOLD"
            target_position_pct = 0.0

        # Determine trade worthiness
        trade_worthy = (
            result.final_probability > 0.58 and
            validation_score > 0.5 and
            trend_alignment > 0.5
        )

        # Risk metrics
        expected_volatility = result.expected_volatility if hasattr(result, 'expected_volatility') else 0.02
        var_95 = expected_volatility * 1.645  # Approx VaR at 95%

        # Trade parameters
        if trade_worthy and signal == "BUY":
            entry_condition = "Enter on pullback to support or breakout confirmation"
            stop_loss_pct = expected_volatility * 2  # 2x daily vol
            target_pct = stop_loss_pct * 2.5  # 2.5:1 reward
            risk_reward_ratio = 2.5
        elif trade_worthy and signal == "SELL":
            entry_condition = "Exit on bounce to resistance"
            stop_loss_pct = expected_volatility * 1.5
            target_pct = stop_loss_pct * 2.0
            risk_reward_ratio = 2.0
        else:
            entry_condition = "Wait for clearer signal"
            stop_loss_pct = 0.0
            target_pct = 0.0
            risk_reward_ratio = 0.0

        return UIAdvancedPrediction(
            symbol=symbol,
            signal=signal,
            probability=result.final_probability,
            confidence=result.confidence,
            model_contributions=model_contributions,
            current_regime=current_regime,
            regime_probability=regime_probability,
            regime_stability=regime_stability,
            trend_alignment=trend_alignment,
            optimal_timeframe=optimal_timeframe,
            wavelet_signal=wavelet_signal,
            kalman_trend=kalman_trend,
            kalman_slope=kalman_slope,
            price_forecast=price_forecast,
            systematic_risk_pct=systematic_risk_pct,
            idiosyncratic_opportunity=idiosyncratic_opportunity,
            beta=beta,
            position_action=position_action,
            target_position_pct=target_position_pct,
            validation_score=validation_score,
            dimension_scores=dimension_scores,
            validation_warnings=validation_warnings,
            expected_volatility=expected_volatility,
            var_95=var_95,
            trade_worthy=trade_worthy,
            entry_condition=entry_condition,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            risk_reward_ratio=risk_reward_ratio,
            source=PredictionSource.ADVANCED_ENGINE
        )

    # ===== HELPER METHODS =====

    def _convert_unified_result(self, result) -> UIStockPrediction:
        """Convert UnifiedPredictor result to UIStockPrediction."""
        return UIStockPrediction(
            symbol=result.symbol,
            trade_type=result.trade_type.value if hasattr(result.trade_type, 'value') else str(result.trade_type),
            signal=result.signal.value if hasattr(result.signal, 'value') else str(result.signal),
            signal_strength=result.signal_strength,
            confidence=result.confidence,
            confidence_grade=result.confidence_grade.value if hasattr(result.confidence_grade, 'value') else str(result.confidence_grade),
            current_price=result.current_price,
            entry_price=result.entry_price,
            stop_loss=result.stop_loss,
            target_price=result.target_price,
            risk_reward=result.risk_reward,
            ensemble_score=result.ensemble_score,
            model_agreement=result.model_agreement,
            model_votes=result.model_votes,
            base_score=result.base_score,
            physics_score=result.physics_score,
            math_score=result.math_score,
            regime_score=result.regime_score,
            regime=result.regime,
            regime_stability=result.regime_stability,
            market_predictability=result.market_predictability,
            atr_pct=result.atr_pct,
            liquidity_cr=result.liquidity_cr,
            reasons=result.reasons,
            warnings=result.warnings,
            recommended_strategy=result.recommended_strategy,
            source=PredictionSource.UNIFIED_PREDICTOR
        )

    def _get_prediction_from_enhanced(
        self,
        symbol: str,
        trade_type: str,
        df: pd.DataFrame
    ) -> Optional[UIStockPrediction]:
        """Fallback: get prediction from enhanced scoring engine."""
        try:
            from src.features.technical import TechnicalIndicators

            if 'rsi' not in df.columns:
                df = TechnicalIndicators.calculate_all(df)

            trade_type_enum = TradeType.INTRADAY if trade_type == "intraday" else TradeType.SWING

            score = self.enhanced_scoring_engine.score_stock(
                symbol=symbol,
                df=df,
                trade_type=trade_type_enum
            )

            if score is None:
                return None

            # Determine confidence grade
            if score.confidence >= 0.70 and score.model_agreement >= 4:
                grade = "A+"
            elif score.confidence >= 0.65 and score.model_agreement >= 3:
                grade = "A"
            elif score.confidence >= 0.55:
                grade = "B"
            elif score.confidence >= 0.45:
                grade = "C"
            else:
                grade = "D"

            return UIStockPrediction(
                symbol=symbol,
                trade_type=trade_type,
                signal=score.signal.value,
                signal_strength=score.signal_strength,
                confidence=score.confidence,
                confidence_grade=grade,
                current_price=score.current_price,
                entry_price=score.entry_price,
                stop_loss=score.stop_loss,
                target_price=score.target_price,
                risk_reward=score.risk_reward,
                ensemble_score=score.ensemble_score,
                model_agreement=score.model_agreement,
                model_votes=score.model_votes,
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
                recommended_strategy=score.recommended_strategy,
                source=PredictionSource.ENHANCED_SCORING
            )

        except Exception as e:
            logger.error(f"Enhanced scoring failed for {symbol}: {e}")
            return None


# ===== CONVENIENCE FUNCTIONS FOR UI =====

_api_instance = None

def get_ui_prediction_api() -> UIPredictionAPI:
    """Get the singleton UI prediction API."""
    global _api_instance
    if _api_instance is None:
        _api_instance = UIPredictionAPI()
    return _api_instance


def get_stock_prediction(
    symbol: str,
    trade_type: str = "intraday"
) -> Optional[UIStockPrediction]:
    """Convenience function to get single stock prediction."""
    return get_ui_prediction_api().get_stock_prediction(symbol, trade_type)


def scan_stocks(
    symbols: List[str] = None,
    **kwargs
) -> UIScanResult:
    """Convenience function to scan multiple stocks."""
    return get_ui_prediction_api().scan_stocks(symbols, **kwargs)


def get_quick_signal(symbol: str) -> Dict[str, Any]:
    """Convenience function to get quick signal."""
    return get_ui_prediction_api().get_quick_signal(symbol)


def get_advanced_prediction(
    symbol: str,
    df: Optional[pd.DataFrame] = None
) -> Optional[UIAdvancedPrediction]:
    """
    Convenience function to get advanced multi-model prediction.

    Uses the 6-model advanced engine with PCA, Wavelet, Kalman,
    Markov, Factor Model, and DQN for comprehensive analysis.
    """
    return get_ui_prediction_api().get_advanced_prediction(symbol, df)


def get_advanced_batch(
    symbols: List[str],
    min_probability: float = 0.55,
    top_n: int = 10
) -> List[UIAdvancedPrediction]:
    """
    Convenience function to get advanced predictions for multiple stocks.

    Returns top picks sorted by probability and trade worthiness.
    """
    return get_ui_prediction_api().get_advanced_batch(symbols, min_probability, top_n)


def get_unified_advanced_prediction(
    symbol: str,
    df: Optional[pd.DataFrame] = None
) -> Optional[UIUnifiedAdvancedPrediction]:
    """
    Convenience function to get unified advanced prediction with ALL 17 models.

    Combines:
    - Original 6: PCA, Wavelet, Kalman, Markov, Factor Model, DQN
    - New 11: Quantum-Inspired, Bellman, SDE, Information Geometry,
              Tensor Decomposition, Spectral Graph, Optimal Transport,
              Taylor Series, Heavy-Tailed, Copula, Multi-Dim Confidence
    """
    return get_ui_prediction_api().get_unified_advanced_prediction(symbol, df)


def get_unified_advanced_batch(
    symbols: List[str],
    min_probability: float = 0.55,
    top_n: int = 10
) -> List[UIUnifiedAdvancedPrediction]:
    """
    Convenience function to get unified advanced predictions for multiple stocks.

    Uses all 17 mathematical models for comprehensive analysis.
    Returns top picks sorted by probability and trade worthiness.
    """
    return get_ui_prediction_api().get_unified_advanced_batch(symbols, min_probability, top_n)


def get_advanced_math_insights(
    symbol: str,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Convenience function to get human-readable insights from 11 advanced math modules.

    Returns insights from: Quantum-Inspired, Bellman, SDE, Information Geometry,
    Tensor, Spectral Graph, Optimal Transport, Taylor, Heavy-Tailed, Copula,
    Multi-Dim Confidence.
    """
    return get_ui_prediction_api().get_advanced_math_insights(symbol, df)


# Export
__all__ = [
    # Classes
    'UIPredictionAPI',
    'UIStockPrediction',
    'UIScanResult',
    'UIAdvancedPrediction',
    'UIUnifiedAdvancedPrediction',
    'PredictionSource',
    # Convenience functions
    'get_ui_prediction_api',
    'get_stock_prediction',
    'scan_stocks',
    'get_quick_signal',
    'get_advanced_prediction',
    'get_advanced_batch',
    # NEW: Unified Advanced (17 models)
    'get_unified_advanced_prediction',
    'get_unified_advanced_batch',
    'get_advanced_math_insights'
]
