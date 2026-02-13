"""
Unified Advanced Prediction Engine

This is the CENTRALIZED engine that integrates all advanced mathematical models:
- PCA & Eigenvalue Decomposition
- Wavelet Multi-Resolution Analysis
- Kalman Filter State Estimation
- MLE-Calibrated Markov Regime
- Deep Q-Network Position Sizing
- Covariance Factor Model
- Advanced Statistical Validation

All predictions flow through this engine to ensure consistency.

Architecture:
=============
Raw Data → Feature Extraction → Dimensionality Reduction (PCA)
                                        ↓
         ┌──────────────────────────────┼──────────────────────────────┐
         ↓                              ↓                              ↓
    Wavelet Analysis              Kalman Filter               Markov Regime
    (Multi-scale patterns)     (Trend estimation)          (State detection)
         ↓                              ↓                              ↓
         └──────────────────────────────┼──────────────────────────────┘
                                        ↓
                              Factor Model Analysis
                           (Systematic vs Idiosyncratic)
                                        ↓
                              Ensemble Prediction
                                        ↓
                          Statistical Validation
                                        ↓
                          DQN Position Recommendation
                                        ↓
                          Final Prediction Result
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger

# Import advanced modules (Original 6)
from src.core.advanced_math.pca_decomposition import (
    PCADecomposer, EigenvalueAnalyzer, create_feature_matrix
)
from src.core.advanced_math.wavelet_analysis import (
    WaveletAnalyzer, MultiResolutionDecomposer, WaveletType
)
from src.core.advanced_math.kalman_filter import (
    KalmanTrendFilter, AdaptiveKalmanFilter, StateModel, compute_kalman_probability
)
from src.core.advanced_math.markov_regime_mle import (
    MLEMarkovRegime, MarketRegime, compute_regime_adjusted_probability
)
from src.core.advanced_math.dqn_position_sizing import (
    DQNPositionAgent, get_dqn_position_recommendation, create_trading_features
)
from src.core.advanced_math.factor_model import (
    FactorModel, CovarianceDecomposer, compute_factor_adjusted_probability
)
from src.core.advanced_math.statistical_validation import (
    AdvancedStatisticalValidator, compute_multi_dimensional_score
)

# Import NEW Advanced Mathematical Modules (11 new modules)
try:
    from src.core.advanced_math.unified_advanced_engine import (
        UnifiedAdvancedEngine,
        UnifiedPrediction,
        predict_with_advanced_math
    )
    from src.core.advanced_math.quantum_inspired import (
        QuantumInspiredOptimizer,
        QuantumPortfolioOptimizer
    )
    from src.core.advanced_math.bellman_optimal import (
        BellmanOptimalStopping,
        HamiltonJacobiBellman
    )
    from src.core.advanced_math.stochastic_de import (
        StochasticDEEngine,
        JumpDetector,
        MeanReversionAnalyzer
    )
    from src.core.advanced_math.information_geometry import (
        InformationGeometryEngine,
        FisherInformationAnalyzer
    )
    from src.core.advanced_math.tensor_decomposition import (
        CPDecomposition,
        TuckerDecomposition,
        MultiScalePatternExtractor
    )
    from src.core.advanced_math.spectral_graph import (
        SpectralGraphEngine,
        RandomMatrixTheory
    )
    from src.core.advanced_math.optimal_transport import (
        OptimalTransportEngine,
        WassersteinDistance
    )
    from src.core.advanced_math.taylor_series import (
        ExpansionEngine,
        EdgeworthExpansion,
        CornishFisherExpansion
    )
    from src.core.advanced_math.heavy_tailed import (
        HeavyTailedEngine,
        StableDistribution,
        GeneralizedParetoDistribution
    )
    from src.core.advanced_math.copula_models import (
        CopulaEngine,
        GaussianCopula,
        StudentTCopula
    )
    from src.core.advanced_math.multidim_confidence import (
        MultiDimensionalScorer,
        EigenvalueConfidenceAnalyzer,
        MahalanobisAnomalyDetector
    )
    UNIFIED_ADVANCED_AVAILABLE = True
    logger.info("Advanced mathematical modules loaded successfully")
except ImportError as e:
    logger.warning(f"Some advanced math modules not available: {e}")
    UNIFIED_ADVANCED_AVAILABLE = False
    UnifiedAdvancedEngine = None


class PredictionSignal(Enum):
    """Prediction signals."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    AVOID = "AVOID"


class TradeTimeframe(Enum):
    """Trading timeframes."""
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


@dataclass
class ModelContribution:
    """Contribution of each model to final prediction."""
    model_name: str
    probability: float
    confidence: float
    weight: float
    signal: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedPredictionResult:
    """Comprehensive prediction result from advanced engine."""
    # Core prediction
    symbol: str
    timestamp: datetime
    signal: PredictionSignal
    probability: float  # 0-1 probability of positive return
    confidence: float  # 0-1 confidence in prediction

    # Model contributions
    model_contributions: List[ModelContribution]
    ensemble_method: str  # How models were combined

    # Regime analysis
    current_regime: MarketRegime
    regime_probability: float
    regime_stability: float
    expected_regime_duration: float

    # Multi-resolution analysis
    trend_alignment: float  # 0-1, how aligned are different timeframes
    optimal_timeframe: TradeTimeframe
    wavelet_signal: str
    wavelet_confidence: float

    # Kalman analysis
    kalman_trend: str  # 'bullish', 'bearish', 'neutral'
    kalman_slope: float
    kalman_confidence: float
    price_forecast: List[float]
    forecast_confidence_interval: Tuple[float, float]

    # Factor analysis
    systematic_risk_pct: float
    idiosyncratic_opportunity: float
    beta: float

    # Position recommendation
    position_action: str
    target_position_pct: float
    position_confidence: float

    # Statistical validation
    validation_score: float
    dimension_scores: Dict[str, float]
    validation_warnings: List[str]

    # Risk metrics
    expected_volatility: float
    var_95: float  # Value at Risk
    max_drawdown_estimate: float

    # Final recommendation
    trade_worthy: bool
    entry_condition: str
    stop_loss_pct: float
    target_pct: float
    risk_reward_ratio: float


@dataclass
class EngineConfig:
    """Configuration for advanced engine."""
    # Model weights
    pca_weight: float = 0.10
    wavelet_weight: float = 0.15
    kalman_weight: float = 0.20
    markov_weight: float = 0.25
    factor_weight: float = 0.10
    dqn_weight: float = 0.20

    # PCA settings
    pca_variance_threshold: float = 0.90
    pca_max_components: int = 8

    # Markov settings
    n_regimes: int = 5
    markov_max_iterations: int = 100

    # Kalman settings
    kalman_model: StateModel = StateModel.LOCAL_TREND

    # Wavelet settings
    wavelet_type: WaveletType = WaveletType.DAUBECHIES_4
    wavelet_decomposition_level: int = 5

    # Validation settings
    min_validation_score: float = 0.5
    require_statistical_significance: bool = True

    # Trading settings
    min_probability_for_trade: float = 0.55
    min_confidence_for_trade: float = 0.50


class AdvancedPredictionEngine:
    """
    Unified Advanced Prediction Engine.

    This is the SINGLE SOURCE OF TRUTH for all advanced predictions.
    All UI components and services should use this engine.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

        # Initialize all models (Original 6)
        self.pca_decomposer = PCADecomposer(
            variance_threshold=self.config.pca_variance_threshold,
            max_components=self.config.pca_max_components
        )
        self.eigenvalue_analyzer = EigenvalueAnalyzer()

        self.wavelet_analyzer = WaveletAnalyzer(
            wavelet_type=self.config.wavelet_type,
            decomposition_level=self.config.wavelet_decomposition_level
        )
        self.multi_resolution = MultiResolutionDecomposer()

        self.kalman_filter = KalmanTrendFilter(model=self.config.kalman_model)
        self.adaptive_kalman = AdaptiveKalmanFilter(model=self.config.kalman_model)

        self.markov_model = MLEMarkovRegime(
            n_regimes=self.config.n_regimes,
            max_iterations=self.config.markov_max_iterations
        )

        self.factor_model = FactorModel(n_statistical_factors=5)
        self.covariance_decomposer = CovarianceDecomposer(n_factors=5)

        self.dqn_agent = DQNPositionAgent()

        self.validator = AdvancedStatisticalValidator()

        # Initialize NEW Unified Advanced Engine (11 advanced modules)
        self.unified_advanced_engine = None
        if UNIFIED_ADVANCED_AVAILABLE:
            try:
                self.unified_advanced_engine = UnifiedAdvancedEngine()
                logger.info("Unified Advanced Engine initialized with 11 modules: "
                          "Quantum, Bellman, SDE, InfoGeometry, Tensor, Spectral, "
                          "OptimalTransport, Taylor, HeavyTailed, Copula, MultiDimConfidence")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedAdvancedEngine: {e}")
                self.unified_advanced_engine = None

        # Cache for efficiency
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info("Advanced Prediction Engine initialized with all modules")

    def predict(
        self,
        prices: pd.DataFrame,
        symbol: str,
        market_prices: Optional[pd.DataFrame] = None,
        current_position: float = 0.0
    ) -> AdvancedPredictionResult:
        """
        Generate comprehensive prediction using all advanced models.

        Args:
            prices: OHLCV DataFrame for the stock
            symbol: Stock symbol
            market_prices: Optional market benchmark prices
            current_position: Current position (-1 to 1)

        Returns:
            AdvancedPredictionResult with all analysis
        """
        logger.info(f"Generating advanced prediction for {symbol}")

        # Extract close prices
        close_col = 'close' if 'close' in prices.columns else 'Close'
        close_prices = prices[close_col].values

        if len(close_prices) < 60:
            return self._insufficient_data_result(symbol)

        # Calculate returns
        returns = np.diff(np.log(close_prices))

        # 1. PCA Analysis
        pca_result = self._run_pca_analysis(prices)

        # 2. Wavelet Analysis
        wavelet_result = self._run_wavelet_analysis(close_prices)

        # 3. Kalman Filter Analysis
        kalman_result = self._run_kalman_analysis(close_prices)

        # 4. Markov Regime Analysis
        markov_result = self._run_markov_analysis(returns)

        # 5. Factor Model Analysis
        factor_result = self._run_factor_analysis(returns, market_prices)

        # 6. DQN Position Analysis
        dqn_result = self._run_dqn_analysis(prices, current_position)

        # 7. Combine model predictions
        model_contributions = [
            ModelContribution("PCA", pca_result['probability'], pca_result['confidence'],
                            self.config.pca_weight, pca_result['signal'], pca_result),
            ModelContribution("Wavelet", wavelet_result['probability'], wavelet_result['confidence'],
                            self.config.wavelet_weight, wavelet_result['signal'], wavelet_result),
            ModelContribution("Kalman", kalman_result['probability'], kalman_result['confidence'],
                            self.config.kalman_weight, kalman_result['signal'], kalman_result),
            ModelContribution("Markov", markov_result['probability'], markov_result['confidence'],
                            self.config.markov_weight, markov_result['signal'], markov_result),
            ModelContribution("Factor", factor_result['probability'], factor_result['confidence'],
                            self.config.factor_weight, factor_result['signal'], factor_result),
            ModelContribution("DQN", dqn_result['probability'], dqn_result['confidence'],
                            self.config.dqn_weight, dqn_result['signal'], dqn_result),
        ]

        # 8. Ensemble prediction
        ensemble_prob, ensemble_confidence = self._ensemble_predictions(model_contributions)

        # 9. Statistical validation
        validation_result = self._run_validation(returns)

        # 10. Generate final signal
        signal = self._generate_signal(ensemble_prob, ensemble_confidence, validation_result)

        # 11. Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns, close_prices)

        # 12. Determine if trade-worthy
        trade_worthy = self._is_trade_worthy(
            ensemble_prob, ensemble_confidence,
            validation_result, markov_result, risk_metrics
        )

        # 13. Calculate trade levels
        entry_condition, stop_loss, target, rr_ratio = self._calculate_trade_levels(
            close_prices, signal, risk_metrics
        )

        return AdvancedPredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=signal,
            probability=ensemble_prob,
            confidence=ensemble_confidence,
            model_contributions=model_contributions,
            ensemble_method="weighted_average",
            current_regime=markov_result['regime'],
            regime_probability=markov_result['regime_probability'],
            regime_stability=markov_result['stability'],
            expected_regime_duration=markov_result['expected_duration'],
            trend_alignment=wavelet_result['trend_alignment'],
            optimal_timeframe=wavelet_result['optimal_timeframe'],
            wavelet_signal=wavelet_result['signal'],
            wavelet_confidence=wavelet_result['confidence'],
            kalman_trend=kalman_result['trend_direction'],
            kalman_slope=kalman_result['slope'],
            kalman_confidence=kalman_result['confidence'],
            price_forecast=kalman_result['forecast'],
            forecast_confidence_interval=kalman_result['forecast_ci'],
            systematic_risk_pct=factor_result['systematic_risk_pct'],
            idiosyncratic_opportunity=factor_result['idiosyncratic_opportunity'],
            beta=factor_result['beta'],
            position_action=dqn_result['action'],
            target_position_pct=dqn_result['target_position'],
            position_confidence=dqn_result['confidence'],
            validation_score=validation_result['overall_score'],
            dimension_scores=validation_result['dimension_scores'],
            validation_warnings=validation_result['warnings'],
            expected_volatility=risk_metrics['volatility'],
            var_95=risk_metrics['var_95'],
            max_drawdown_estimate=risk_metrics['max_dd_estimate'],
            trade_worthy=trade_worthy,
            entry_condition=entry_condition,
            stop_loss_pct=stop_loss,
            target_pct=target,
            risk_reward_ratio=rr_ratio
        )

    def _run_pca_analysis(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """Run PCA analysis on price features."""
        try:
            features, feature_names = create_feature_matrix(prices)

            if len(features) < 30:
                return {'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD'}

            result = self.pca_decomposer.fit_transform(features, feature_names)

            # Use first PC direction for signal
            recent_pc1 = result.transformed_data[-5:, 0]
            pc1_trend = np.mean(np.diff(recent_pc1))

            if pc1_trend > 0:
                probability = 0.5 + min(0.2, abs(pc1_trend) * 10)
                signal = 'BUY'
            elif pc1_trend < 0:
                probability = 0.5 - min(0.2, abs(pc1_trend) * 10)
                signal = 'SELL'
            else:
                probability = 0.5
                signal = 'HOLD'

            confidence = min(1.0, result.total_variance_explained)

            return {
                'probability': probability,
                'confidence': confidence,
                'signal': signal,
                'n_components': result.n_components_selected,
                'variance_explained': result.total_variance_explained,
                'condition_number': result.condition_number
            }
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            return {'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD'}

    def _run_wavelet_analysis(self, prices: np.ndarray) -> Dict[str, Any]:
        """Run wavelet multi-resolution analysis."""
        try:
            result = self.multi_resolution.analyze(prices)

            return {
                'probability': result.bullish_probability,
                'confidence': result.confidence_score,
                'signal': result.signal,
                'trend_alignment': result.trend_alignment,
                'optimal_timeframe': TradeTimeframe.SWING if result.optimal_timeframe == 'swing'
                                    else TradeTimeframe.INTRADAY,
                'ultra_short_trend': result.ultra_short_trend,
                'weekly_trend': result.weekly_trend,
                'swing_trend': result.swing_trend,
                'monthly_trend': result.monthly_trend
            }
        except Exception as e:
            logger.warning(f"Wavelet analysis failed: {e}")
            return {
                'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD',
                'trend_alignment': 0.0, 'optimal_timeframe': TradeTimeframe.SWING
            }

    def _run_kalman_analysis(self, prices: np.ndarray) -> Dict[str, Any]:
        """Run Kalman filter analysis."""
        try:
            result = self.adaptive_kalman.filter_adaptive(prices)

            # Convert slope to probability
            probability = 1 / (1 + np.exp(-result.smoothed_slope[-1] * 100))

            return {
                'probability': probability,
                'confidence': result.trend_confidence,
                'signal': 'BUY' if probability > 0.55 else ('SELL' if probability < 0.45 else 'HOLD'),
                'trend_direction': result.trend_direction,
                'slope': result.smoothed_slope[-1],
                'forecast': result.forecast.tolist(),
                'forecast_ci': (float(result.forecast_lower[0]), float(result.forecast_upper[0])),
                'signal_quality': result.signal_quality
            }
        except Exception as e:
            logger.warning(f"Kalman analysis failed: {e}")
            return {
                'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD',
                'trend_direction': 'neutral', 'slope': 0.0,
                'forecast': [float(prices[-1])], 'forecast_ci': (0, 0)
            }

    def _run_markov_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """Run MLE Markov regime analysis."""
        try:
            stats = self.markov_model.get_regime_statistics(returns)

            # Convert regime to probability
            regime = stats.current_regime
            if regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]:
                probability = 0.55 + 0.15 * stats.regime_stability
            elif regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
                probability = 0.45 - 0.15 * stats.regime_stability
            else:
                probability = 0.50

            return {
                'probability': probability,
                'confidence': stats.current_probability,
                'signal': 'BUY' if probability > 0.55 else ('SELL' if probability < 0.45 else 'HOLD'),
                'regime': regime,
                'regime_probability': stats.current_probability,
                'stability': stats.regime_stability,
                'expected_duration': stats.expected_durations.get(0, 10),
                'next_regime_probs': stats.next_regime_probabilities
            }
        except Exception as e:
            logger.warning(f"Markov analysis failed: {e}")
            return {
                'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD',
                'regime': MarketRegime.SIDEWAYS, 'regime_probability': 0.5,
                'stability': 0.5, 'expected_duration': 10
            }

    def _run_factor_analysis(
        self,
        returns: np.ndarray,
        market_prices: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Run factor model analysis."""
        try:
            if market_prices is not None:
                market_col = 'close' if 'close' in market_prices.columns else 'Close'
                market_returns = market_prices[market_col].pct_change().dropna().values

                if len(market_returns) >= len(returns):
                    market_returns = market_returns[-len(returns):]
                else:
                    market_returns = np.zeros(len(returns))

                prob, beta, idio_ratio = compute_factor_adjusted_probability(
                    pd.Series(returns),
                    pd.Series(market_returns),
                    0.5  # Base probability
                )
            else:
                prob = 0.5
                beta = 1.0
                idio_ratio = 0.5

            return {
                'probability': prob,
                'confidence': idio_ratio,  # Higher idiosyncratic = more confidence in signal
                'signal': 'BUY' if prob > 0.55 else ('SELL' if prob < 0.45 else 'HOLD'),
                'systematic_risk_pct': 1 - idio_ratio,
                'idiosyncratic_opportunity': idio_ratio,
                'beta': beta
            }
        except Exception as e:
            logger.warning(f"Factor analysis failed: {e}")
            return {
                'probability': 0.5, 'confidence': 0.5, 'signal': 'HOLD',
                'systematic_risk_pct': 0.5, 'idiosyncratic_opportunity': 0.5, 'beta': 1.0
            }

    def _run_dqn_analysis(
        self,
        prices: pd.DataFrame,
        current_position: float
    ) -> Dict[str, Any]:
        """Run DQN position sizing analysis."""
        try:
            result = get_dqn_position_recommendation(prices, current_position, self.dqn_agent)

            # Convert action to probability
            if 'BUY' in result['action']:
                probability = 0.55 + 0.1 * result['confidence']
            elif 'SELL' in result['action']:
                probability = 0.45 - 0.1 * result['confidence']
            else:
                probability = 0.5

            return {
                'probability': probability,
                'confidence': result['confidence'],
                'signal': result['action'].split('_')[0] if '_' in result['action'] else result['action'],
                'action': result['action'],
                'target_position': result['target_position'],
                'position_change': result.get('position_change', 0)
            }
        except Exception as e:
            logger.warning(f"DQN analysis failed: {e}")
            return {
                'probability': 0.5, 'confidence': 0.0, 'signal': 'HOLD',
                'action': 'HOLD', 'target_position': current_position
            }

    def _run_validation(self, returns: np.ndarray) -> Dict[str, Any]:
        """Run statistical validation."""
        try:
            scores = compute_multi_dimensional_score(returns)

            report = self.validator.validate_signal(returns)
            warnings = report.warnings

            return {
                'overall_score': scores['overall'],
                'dimension_scores': {k: v for k, v in scores.items() if k not in ['overall', 'confidence', 'n_tests_passed', 'n_tests_total']},
                'warnings': warnings,
                'n_tests_passed': scores['n_tests_passed'],
                'n_tests_total': scores['n_tests_total']
            }
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {
                'overall_score': 0.5,
                'dimension_scores': {},
                'warnings': [str(e)],
                'n_tests_passed': 0,
                'n_tests_total': 0
            }

    def _ensemble_predictions(
        self,
        contributions: List[ModelContribution]
    ) -> Tuple[float, float]:
        """Combine model predictions using weighted average."""
        total_weight = sum(c.weight for c in contributions)

        if total_weight == 0:
            return 0.5, 0.0

        # Weighted probability
        weighted_prob = sum(c.probability * c.weight for c in contributions) / total_weight

        # Weighted confidence (also consider model agreement)
        weighted_conf = sum(c.confidence * c.weight for c in contributions) / total_weight

        # Agreement bonus: if models agree, increase confidence
        signals = [c.signal for c in contributions]
        buy_count = sum(1 for s in signals if 'BUY' in s)
        sell_count = sum(1 for s in signals if 'SELL' in s)
        hold_count = sum(1 for s in signals if 'HOLD' in s)

        max_agreement = max(buy_count, sell_count, hold_count)
        agreement_ratio = max_agreement / len(signals)

        # Boost confidence if models agree
        final_confidence = weighted_conf * (0.7 + 0.3 * agreement_ratio)

        return weighted_prob, min(1.0, final_confidence)

    def _generate_signal(
        self,
        probability: float,
        confidence: float,
        validation: Dict[str, Any]
    ) -> PredictionSignal:
        """Generate final signal from ensemble."""
        validation_score = validation.get('overall_score', 0.5)

        # Adjust thresholds based on validation
        if validation_score < 0.4:
            # Low validation score = stricter thresholds
            buy_threshold = 0.65
            strong_buy_threshold = 0.75
            sell_threshold = 0.35
            strong_sell_threshold = 0.25
        else:
            buy_threshold = 0.55
            strong_buy_threshold = 0.70
            sell_threshold = 0.45
            strong_sell_threshold = 0.30

        if probability >= strong_buy_threshold and confidence >= 0.6:
            return PredictionSignal.STRONG_BUY
        elif probability >= buy_threshold and confidence >= 0.4:
            return PredictionSignal.BUY
        elif probability <= strong_sell_threshold and confidence >= 0.6:
            return PredictionSignal.STRONG_SELL
        elif probability <= sell_threshold and confidence >= 0.4:
            return PredictionSignal.SELL
        else:
            return PredictionSignal.HOLD

    def _calculate_risk_metrics(
        self,
        returns: np.ndarray,
        prices: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk metrics."""
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) * np.sqrt(5)  # 5-day VaR

        # Max drawdown estimate
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)

        return {
            'volatility': volatility,
            'var_95': abs(var_95),
            'max_dd_estimate': abs(max_dd)
        }

    def _is_trade_worthy(
        self,
        probability: float,
        confidence: float,
        validation: Dict[str, Any],
        markov: Dict[str, Any],
        risk: Dict[str, float]
    ) -> bool:
        """Determine if trade is worthy based on all factors."""
        # Basic probability threshold
        if abs(probability - 0.5) < 0.05:
            return False

        # Confidence threshold
        if confidence < self.config.min_confidence_for_trade:
            return False

        # Validation score
        if validation.get('overall_score', 0) < self.config.min_validation_score:
            return False

        # Avoid high volatility regimes
        regime = markov.get('regime', MarketRegime.SIDEWAYS)
        if regime in [MarketRegime.BEAR_HIGH_VOL]:
            return False

        # Risk check
        if risk.get('volatility', 0) > 0.5:  # 50% annualized vol is extreme
            return False

        return True

    def _calculate_trade_levels(
        self,
        prices: np.ndarray,
        signal: PredictionSignal,
        risk: Dict[str, float]
    ) -> Tuple[str, float, float, float]:
        """Calculate entry, stop-loss, and target levels."""
        current_price = prices[-1]
        atr_estimate = risk.get('volatility', 0.02) / np.sqrt(252) * current_price

        if signal in [PredictionSignal.STRONG_BUY, PredictionSignal.BUY]:
            entry_condition = f"Buy at market or on pullback to {current_price * 0.99:.2f}"
            stop_loss = 2.0  # 2% stop
            target = 4.0  # 4% target
            rr_ratio = target / stop_loss
        elif signal in [PredictionSignal.STRONG_SELL, PredictionSignal.SELL]:
            entry_condition = f"Sell at market or on bounce to {current_price * 1.01:.2f}"
            stop_loss = 2.0
            target = 4.0
            rr_ratio = target / stop_loss
        else:
            entry_condition = "Wait for clearer signal"
            stop_loss = 0.0
            target = 0.0
            rr_ratio = 0.0

        return entry_condition, stop_loss, target, rr_ratio

    def _run_unified_advanced_analysis(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Run the Unified Advanced Engine with 11 advanced mathematical modules.

        Modules: Quantum-Inspired, Bellman Optimal Stopping, SDEs, Information Geometry,
        Tensor Decomposition, Spectral Graph, Optimal Transport, Taylor Series,
        Heavy-Tailed Distributions, Copulas, Multi-Dimensional Confidence.

        Args:
            prices: Close prices array
            returns: Log returns array
            symbol: Stock symbol

        Returns:
            Dictionary with advanced analysis results
        """
        if self.unified_advanced_engine is None:
            return {
                'available': False,
                'signal': 0.0,
                'confidence': 0.0,
                'regime': 'unknown',
                'action': 'hold'
            }

        try:
            # Run the unified advanced prediction
            result = self.unified_advanced_engine.predict(prices, returns)

            return {
                'available': True,
                'signal': result.signal,
                'confidence': result.confidence,
                'regime': result.regime,
                'action': result.action,
                # Quantum analysis
                'quantum_signal': result.quantum_signal if hasattr(result, 'quantum_signal') else 0.0,
                'quantum_confidence': result.quantum_confidence if hasattr(result, 'quantum_confidence') else 0.0,
                # Bellman analysis
                'optimal_action': result.optimal_action if hasattr(result, 'optimal_action') else 'hold',
                'bellman_value': result.bellman_value if hasattr(result, 'bellman_value') else 0.0,
                # SDE analysis
                'sde_drift': result.sde_drift if hasattr(result, 'sde_drift') else 0.0,
                'sde_volatility': result.sde_volatility if hasattr(result, 'sde_volatility') else 0.0,
                'jump_intensity': result.jump_intensity if hasattr(result, 'jump_intensity') else 0.0,
                # Information geometry
                'fisher_information': result.fisher_information if hasattr(result, 'fisher_information') else 0.0,
                'kl_divergence': result.kl_divergence if hasattr(result, 'kl_divergence') else 0.0,
                # Heavy-tailed analysis
                'tail_index': result.tail_index if hasattr(result, 'tail_index') else 2.0,
                'var_adjusted': result.var_adjusted if hasattr(result, 'var_adjusted') else 0.0,
                # Copula analysis
                'tail_dependence': result.tail_dependence if hasattr(result, 'tail_dependence') else 0.0,
                # Multi-dimensional confidence
                'multidim_confidence': result.multidim_confidence if hasattr(result, 'multidim_confidence') else 0.5,
                'anomaly_score': result.anomaly_score if hasattr(result, 'anomaly_score') else 0.0,
                # Component results (raw for detailed UI display)
                'component_results': result.component_results if hasattr(result, 'component_results') else {}
            }

        except Exception as e:
            logger.warning(f"Unified advanced analysis failed for {symbol}: {e}")
            return {
                'available': False,
                'error': str(e),
                'signal': 0.0,
                'confidence': 0.0,
                'regime': 'unknown',
                'action': 'hold'
            }

    def predict_with_unified_math(
        self,
        prices: pd.DataFrame,
        symbol: str,
        market_prices: Optional[pd.DataFrame] = None,
        current_position: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate prediction using ALL mathematical models (Original 6 + New 11).

        This method combines:
        - Original 6: PCA, Wavelet, Kalman, Markov, DQN, Factor Model
        - New 11: Quantum, Bellman, SDE, InfoGeometry, Tensor, Spectral,
                 OptimalTransport, Taylor, HeavyTailed, Copula, MultiDimConfidence

        Args:
            prices: OHLCV DataFrame
            symbol: Stock symbol
            market_prices: Optional market benchmark
            current_position: Current position size

        Returns:
            Combined prediction with all models
        """
        # Get base prediction from original 6 models
        base_result = self.predict(prices, symbol, market_prices, current_position)

        # Extract close prices for unified engine
        close_col = 'close' if 'close' in prices.columns else 'Close'
        close_prices = prices[close_col].values
        returns = np.diff(np.log(close_prices))

        # Run unified advanced analysis (11 new modules)
        unified_result = self._run_unified_advanced_analysis(close_prices, returns, symbol)

        # Combine results
        combined = {
            # Base prediction
            'symbol': base_result.symbol,
            'timestamp': base_result.timestamp,
            'base_signal': base_result.signal.value,
            'base_probability': base_result.probability,
            'base_confidence': base_result.confidence,

            # Original 6 model contributions
            'model_contributions': {
                mc.model_name: {
                    'probability': mc.probability,
                    'confidence': mc.confidence,
                    'weight': mc.weight,
                    'signal': mc.signal
                } for mc in base_result.model_contributions
            },

            # Regime info
            'regime': base_result.current_regime.value if hasattr(base_result.current_regime, 'value') else str(base_result.current_regime),
            'regime_probability': base_result.regime_probability,
            'regime_stability': base_result.regime_stability,

            # Wavelet
            'trend_alignment': base_result.trend_alignment,
            'wavelet_signal': base_result.wavelet_signal,

            # Kalman
            'kalman_trend': base_result.kalman_trend,
            'kalman_slope': base_result.kalman_slope,
            'price_forecast': base_result.price_forecast,

            # Factor
            'systematic_risk_pct': base_result.systematic_risk_pct,
            'idiosyncratic_opportunity': base_result.idiosyncratic_opportunity,
            'beta': base_result.beta,

            # DQN
            'position_action': base_result.position_action,
            'target_position_pct': base_result.target_position_pct,

            # Validation
            'validation_score': base_result.validation_score,
            'dimension_scores': base_result.dimension_scores,

            # Risk
            'expected_volatility': base_result.expected_volatility,
            'var_95': base_result.var_95,
            'max_drawdown_estimate': base_result.max_drawdown_estimate,

            # Trade info
            'trade_worthy': base_result.trade_worthy,
            'entry_condition': base_result.entry_condition,
            'stop_loss_pct': base_result.stop_loss_pct,
            'target_pct': base_result.target_pct,
            'risk_reward_ratio': base_result.risk_reward_ratio,

            # NEW: Unified Advanced Analysis (11 modules)
            'unified_advanced': unified_result,
            'unified_available': unified_result.get('available', False),

            # Combined signal (blend of base and unified)
            'final_signal': self._compute_final_signal(base_result, unified_result),
            'final_probability': self._compute_final_probability(base_result, unified_result),
            'final_confidence': self._compute_final_confidence(base_result, unified_result),

            # Total model count
            'total_models_used': 6 + (11 if unified_result.get('available', False) else 0)
        }

        return combined

    def _compute_final_signal(
        self,
        base_result: AdvancedPredictionResult,
        unified_result: Dict[str, Any]
    ) -> str:
        """Compute final signal combining both engines."""
        if not unified_result.get('available', False):
            return base_result.signal.value

        # Weight: 60% base (6 models), 40% unified (11 models)
        base_signal_value = {
            'STRONG_BUY': 1.0, 'BUY': 0.7, 'HOLD': 0.5,
            'SELL': 0.3, 'STRONG_SELL': 0.0, 'AVOID': 0.5
        }.get(base_result.signal.value, 0.5)

        unified_signal = unified_result.get('signal', 0.0)
        # Normalize unified signal from [-1, 1] to [0, 1]
        unified_signal_normalized = (unified_signal + 1) / 2

        combined = 0.6 * base_signal_value + 0.4 * unified_signal_normalized

        if combined >= 0.7:
            return 'STRONG_BUY'
        elif combined >= 0.55:
            return 'BUY'
        elif combined <= 0.3:
            return 'STRONG_SELL'
        elif combined <= 0.45:
            return 'SELL'
        else:
            return 'HOLD'

    def _compute_final_probability(
        self,
        base_result: AdvancedPredictionResult,
        unified_result: Dict[str, Any]
    ) -> float:
        """Compute final probability combining both engines."""
        if not unified_result.get('available', False):
            return base_result.probability

        unified_signal = unified_result.get('signal', 0.0)
        # Convert signal to probability
        unified_prob = (unified_signal + 1) / 2

        # Weighted average: 60% base, 40% unified
        return 0.6 * base_result.probability + 0.4 * unified_prob

    def _compute_final_confidence(
        self,
        base_result: AdvancedPredictionResult,
        unified_result: Dict[str, Any]
    ) -> float:
        """Compute final confidence combining both engines."""
        if not unified_result.get('available', False):
            return base_result.confidence

        unified_conf = unified_result.get('confidence', 0.0)
        multidim_conf = unified_result.get('multidim_confidence', 0.5)

        # Average of all confidence measures
        avg_unified = (unified_conf + multidim_conf) / 2

        # Weighted average: 60% base, 40% unified
        combined = 0.6 * base_result.confidence + 0.4 * avg_unified

        # Boost if both engines agree on direction
        base_bullish = base_result.probability > 0.55
        unified_bullish = unified_result.get('signal', 0.0) > 0.1

        if base_bullish == unified_bullish:
            combined = min(1.0, combined * 1.1)  # 10% boost for agreement

        return combined

    def get_unified_analysis_summary(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get a summary of the unified advanced analysis for UI display.

        Returns human-readable insights from the 11 advanced modules.
        """
        close_col = 'close' if 'close' in prices.columns else 'Close'
        close_prices = prices[close_col].values
        returns = np.diff(np.log(close_prices))

        result = self._run_unified_advanced_analysis(close_prices, returns, symbol)

        if not result.get('available', False):
            return {
                'available': False,
                'message': 'Advanced mathematical analysis not available',
                'modules': []
            }

        # Generate human-readable insights
        insights = []

        # Quantum analysis insight
        quantum_signal = result.get('quantum_signal', 0)
        if abs(quantum_signal) > 0.3:
            direction = 'bullish' if quantum_signal > 0 else 'bearish'
            insights.append(f"Quantum-inspired optimization indicates {direction} tendency")

        # Bellman optimal action
        optimal_action = result.get('optimal_action', 'hold')
        if optimal_action != 'hold':
            insights.append(f"Bellman dynamic programming suggests optimal action: {optimal_action}")

        # SDE analysis
        drift = result.get('sde_drift', 0)
        jump_intensity = result.get('jump_intensity', 0)
        if abs(drift) > 0.0001:
            trend = 'upward' if drift > 0 else 'downward'
            insights.append(f"Stochastic DE model detects {trend} drift")
        if jump_intensity > 0.1:
            insights.append(f"Jump-diffusion model detects elevated jump risk")

        # Tail analysis
        tail_index = result.get('tail_index', 2.0)
        if tail_index < 2.0:
            insights.append(f"Heavy-tailed distribution detected (α={tail_index:.2f}) - elevated tail risk")

        # Anomaly detection
        anomaly_score = result.get('anomaly_score', 0)
        if anomaly_score > 0.7:
            insights.append(f"Mahalanobis anomaly detector flags unusual pattern")

        return {
            'available': True,
            'signal': result.get('signal', 0),
            'confidence': result.get('confidence', 0),
            'regime': result.get('regime', 'unknown'),
            'action': result.get('action', 'hold'),
            'insights': insights,
            'modules_used': [
                'Quantum-Inspired', 'Bellman Optimal', 'Stochastic DE',
                'Information Geometry', 'Tensor Decomposition', 'Spectral Graph',
                'Optimal Transport', 'Taylor Series', 'Heavy-Tailed',
                'Copula', 'Multi-Dim Confidence'
            ],
            'detailed': result
        }

    def _insufficient_data_result(self, symbol: str) -> AdvancedPredictionResult:
        """Return result when insufficient data."""
        return AdvancedPredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=PredictionSignal.AVOID,
            probability=0.5,
            confidence=0.0,
            model_contributions=[],
            ensemble_method="none",
            current_regime=MarketRegime.SIDEWAYS,
            regime_probability=0.5,
            regime_stability=0.5,
            expected_regime_duration=0,
            trend_alignment=0.0,
            optimal_timeframe=TradeTimeframe.SWING,
            wavelet_signal="HOLD",
            wavelet_confidence=0.0,
            kalman_trend="neutral",
            kalman_slope=0.0,
            kalman_confidence=0.0,
            price_forecast=[],
            forecast_confidence_interval=(0, 0),
            systematic_risk_pct=0.5,
            idiosyncratic_opportunity=0.5,
            beta=1.0,
            position_action="HOLD",
            target_position_pct=0.0,
            position_confidence=0.0,
            validation_score=0.0,
            dimension_scores={},
            validation_warnings=["Insufficient data"],
            expected_volatility=0.0,
            var_95=0.0,
            max_drawdown_estimate=0.0,
            trade_worthy=False,
            entry_condition="Wait for more data",
            stop_loss_pct=0.0,
            target_pct=0.0,
            risk_reward_ratio=0.0
        )


    def evaluate_prediction_quality(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate quality of historical predictions across multiple dimensions.

        This uses the comprehensive measurement metrics engine to assess:
        - Accuracy (direction, magnitude, hit rate)
        - Calibration (Brier score, reliability)
        - Timing (entry/exit efficiency)
        - Risk-adjusted (Sharpe, Sortino, max drawdown)
        - Information (mutual information, IC)
        - Stability (robustness, consistency)

        Args:
            predictions: Historical predicted signals (-1, 0, 1)
            actuals: Historical actual returns
            probabilities: Optional predicted probabilities

        Returns:
            Dictionary with all quality metrics
        """
        try:
            from src.core.advanced_math.measurement_metrics import (
                compute_prediction_quality,
                MultiDimensionalScore
            )

            score = compute_prediction_quality(predictions, actuals, probabilities)

            return {
                'overall_score': score.overall_score,
                'weighted_score': score.weighted_score,
                'dimension_scores': {
                    'accuracy': score.accuracy_score,
                    'calibration': score.calibration_score,
                    'timing': score.timing_score,
                    'risk_adjusted': score.risk_adjusted_score,
                    'information': score.information_score,
                    'stability': score.stability_score
                },
                'detailed_metrics': {
                    'accuracy': {
                        'direction_accuracy': score.accuracy.direction_accuracy,
                        'hit_rate': score.accuracy.hit_rate,
                        'f1_score': score.accuracy.f1_score,
                        'profit_factor': score.accuracy.profit_factor
                    },
                    'calibration': {
                        'brier_score': score.calibration.brier_score,
                        'calibration_error': score.calibration.calibration_error,
                        'reliability_slope': score.calibration.reliability_slope
                    },
                    'risk_adjusted': {
                        'sharpe_ratio': score.risk_adjusted.sharpe_ratio,
                        'sortino_ratio': score.risk_adjusted.sortino_ratio,
                        'max_drawdown': score.risk_adjusted.max_drawdown,
                        'omega_ratio': score.risk_adjusted.omega_ratio
                    },
                    'information': {
                        'information_coefficient': score.information.information_coefficient,
                        'signal_to_noise_ratio': score.information.signal_to_noise_ratio,
                        'mutual_information': score.information.mutual_information
                    },
                    'stability': {
                        'hurst_exponent': score.stability.hurst_exponent,
                        'time_stability': score.stability.time_stability,
                        'cv_stability': score.stability.cross_validation_stability
                    }
                },
                'n_observations': score.n_observations,
                'is_reliable': score.overall_score > 0.5
            }
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e),
                'is_reliable': False
            }

    def backtest_quality(
        self,
        prices: pd.DataFrame,
        symbol: str,
        lookback: int = 60,
        step: int = 5
    ) -> Dict[str, Any]:
        """
        Backtest prediction quality using rolling window.

        Args:
            prices: OHLCV DataFrame
            symbol: Stock symbol
            lookback: Window size for predictions
            step: Step size for rolling

        Returns:
            Quality metrics from backtesting
        """
        close_col = 'close' if 'close' in prices.columns else 'Close'
        close_prices = prices[close_col].values
        n = len(close_prices)

        if n < lookback + step:
            return {'error': 'Insufficient data for backtesting'}

        predictions = []
        actuals = []
        probabilities = []

        for i in range(lookback, n - step, step):
            # Create window dataframe
            window_df = prices.iloc[i - lookback:i].copy()

            try:
                # Get prediction
                result = self.predict(window_df, symbol)

                # Record prediction
                if result.signal in [PredictionSignal.STRONG_BUY, PredictionSignal.BUY]:
                    pred_signal = 1
                elif result.signal in [PredictionSignal.STRONG_SELL, PredictionSignal.SELL]:
                    pred_signal = -1
                else:
                    pred_signal = 0

                predictions.append(pred_signal)
                probabilities.append(result.probability)

                # Actual return over step period
                actual_return = (close_prices[i + step] - close_prices[i]) / close_prices[i]
                actuals.append(actual_return)

            except Exception as e:
                logger.warning(f"Backtest step {i} failed: {e}")
                continue

        if len(predictions) < 10:
            return {'error': 'Not enough valid predictions for evaluation'}

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        probabilities = np.array(probabilities)

        return self.evaluate_prediction_quality(predictions, actuals, probabilities)


# Singleton instance for centralized access
_engine_instance: Optional[AdvancedPredictionEngine] = None


def get_advanced_engine(config: Optional[EngineConfig] = None) -> AdvancedPredictionEngine:
    """Get or create the singleton advanced engine instance."""
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = AdvancedPredictionEngine(config)

    return _engine_instance


def get_advanced_prediction(
    prices: pd.DataFrame,
    symbol: str,
    market_prices: Optional[pd.DataFrame] = None,
    current_position: float = 0.0
) -> AdvancedPredictionResult:
    """
    Convenience function to get advanced prediction.

    This is the main entry point for getting predictions from the advanced engine.
    """
    engine = get_advanced_engine()
    return engine.predict(prices, symbol, market_prices, current_position)
