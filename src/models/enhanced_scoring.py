"""
Enhanced Scoring Engine

Integrates all available prediction models (7-model ensemble):
- Base Technical/Momentum/Volume/News scoring
- Physics Engine (momentum conservation, spring reversion, energy clustering, network)
- Math Engine (Fourier cycles, Hurst exponent, entropy, statistical mechanics)
- HMM Regime Detection
- Macro Engine (commodities, currencies, bonds, semiconductors) [NEW]
- Alternative Data Engine (earnings, options, institutional flow, multi-timeframe) [NEW]
- Advanced Math Engine (Kalman, Wavelet, PCA, Markov, etc.) [NEW]

Uses ensemble voting for more robust predictions.
Addresses identified gaps:
1. SELL signal inversion bug (fixed by disabling unreliable SELL signals)
2. Unused advanced engines (now integrated)
3. Hardcoded thresholds (calibrated from backtest data)
4. Macro and alternative data now integrated for 65-68% target accuracy
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import get_settings
from config.thresholds import (
    SIGNAL_THRESHOLDS, CONFIDENCE_CALIBRATION, ENSEMBLE_WEIGHTS,
    RISK_THRESHOLDS, get_confidence_for_score, apply_risk_penalties
)
from src.features.technical import TechnicalIndicators
from src.features.market_features import MarketFeatures
from src.storage.models import SignalType, TradeType
from src.models.scoring import ScoringEngine, StockScore
from src.physics.physics_engine import PhysicsEngine, PhysicsScore
from src.math_models.math_engine import MathEngine, MathScore
from src.ml.hmm_regime_detector import HMMRegimeDetector, RegimeResult, MarketRegime

# Standardized model output interfaces
from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    EnsembleInput,
    ModelUncertainty,
)

# Model adapters for standardized outputs
from src.core.adapters import (
    TechnicalModelAdapter,
    PhysicsModelAdapter,
    MathModelAdapter,
    RegimeModelAdapter,
    MacroModelAdapter,
    AlternativeDataAdapter,
    AdvancedMathAdapter,
)

# New engine imports for 7-model ensemble
try:
    from src.macro.macro_engine import MacroEngine, get_macro_engine
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False
    logger.warning("Macro engine not available")

try:
    from src.alternative_data.alternative_engine import AlternativeDataEngine
    ALTERNATIVE_AVAILABLE = True
except ImportError:
    ALTERNATIVE_AVAILABLE = False
    logger.warning("Alternative data engine not available")

try:
    from src.core.advanced_engine import AdvancedPredictionEngine, get_advanced_engine
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    logger.warning("Advanced math engine not available")


@dataclass
class EnhancedStockScore:
    """Complete enhanced score breakdown for a stock."""
    symbol: str
    date: date
    trade_type: TradeType

    # Base scores (0-1) - Original 4 models
    base_score: float
    physics_score: float
    math_score: float
    regime_score: float

    # New model scores (0-1) - 7-model ensemble additions
    macro_score: float = 0.5       # Macro indicators (commodities, currencies, bonds)
    alternative_score: float = 0.5  # Alternative data (earnings, options, institutional)
    advanced_score: float = 0.5     # Advanced math models (Kalman, Wavelet, PCA)

    # Ensemble result
    ensemble_score: float = 0.5
    model_agreement: int = 0  # Number of models agreeing (0-7 for 7-model ensemble)
    total_models: int = 4     # Total models in ensemble (4 or 7)

    # Final calibrated confidence
    confidence: float = 0.5
    signal: SignalType = SignalType.HOLD
    signal_strength: str = 'none'  # 'strong', 'moderate', 'weak', 'none'

    # Trading levels
    current_price: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_reward: float = 0.0

    # Risk metrics
    atr_pct: float = 0.0
    liquidity_cr: float = 0.0
    regime: str = 'unknown'
    regime_stability: float = 0.5

    # Insights
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model_votes: Dict[str, str] = field(default_factory=dict)  # Each model's vote
    recommended_strategy: str = 'neutral'
    market_predictability: float = 0.5

    # Sector info (for macro analysis)
    sector: str = 'Unknown'

    # Optional detailed signals from new engines
    macro_signal: Optional[Any] = None
    alternative_signal: Optional[Any] = None
    advanced_signal: Optional[Any] = None


class EnhancedScoringEngine:
    """
    Enhanced scoring engine combining all prediction models (7-model ensemble).

    Improvements over base ScoringEngine:
    1. Integrates Physics, Math, and HMM engines (original 4 models)
    2. Integrates Macro, Alternative Data, and Advanced Math engines (3 new models)
    3. Uses ensemble voting for signal generation
    4. Disables unreliable SELL signals (28.6% accuracy = inverted)
    5. Calibrates confidence based on backtest results
    6. Requires multi-model agreement for strong signals
    7. Graceful fallback to 4-model if new engines unavailable
    """

    # Calibrated thresholds from backtest - loaded from centralized config
    # See config/thresholds.py for values and documentation
    THRESHOLDS = {
        'strong_buy': SIGNAL_THRESHOLDS.strong_buy,
        'buy': SIGNAL_THRESHOLDS.buy,
        'hold_upper': SIGNAL_THRESHOLDS.hold_upper,
        'hold_lower': SIGNAL_THRESHOLDS.hold_lower,
        # NOTE: SELL signals disabled - see SIGNAL_THRESHOLDS.sell_signals_enabled
    }

    # Sector name mapping for macro signals
    SECTOR_MAPPING = {
        'IT': 'IT',
        'Banking': 'Banking',
        'Finance': 'Finance',
        'Auto': 'Auto',
        'Pharma': 'Pharma',
        'Oil_Gas': 'Oil_Gas',
        'Metals': 'Metals',
        'FMCG': 'FMCG',
        'Infra': 'Infra',
        'Power': 'Power',
        'Telecom': 'Telecom',
        'Consumer': 'Consumer',
        'Chemicals': 'Chemicals',
        'Others': 'Consumer',
        'Unknown': 'Consumer'
    }

    def __init__(self, use_7_model_ensemble: bool = True):
        """
        Initialize the enhanced scoring engine.

        Args:
            use_7_model_ensemble: If True, attempt to use all 7 models.
                                  Falls back to 4-model if new engines fail.
        """
        self.settings = get_settings()
        self.base_engine = ScoringEngine()
        self.physics_engine = PhysicsEngine()
        self.math_engine = MathEngine()
        self.hmm_detector = HMMRegimeDetector()
        self.market_features = MarketFeatures()

        # Initialize new engines for 7-model ensemble
        self.use_7_model = use_7_model_ensemble
        self.macro_engine = None
        self.alternative_engine = None
        self.advanced_engine = None

        if self.use_7_model:
            self._initialize_7_model_engines()

        # Initialize standardized model adapters
        self._initialize_adapters()

    def _initialize_7_model_engines(self):
        """Initialize the three new engines for 7-model ensemble."""
        engines_initialized = 0

        # Macro Engine
        if MACRO_AVAILABLE:
            try:
                self.macro_engine = get_macro_engine()
                engines_initialized += 1
                logger.debug("Macro engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize macro engine: {e}")
                self.macro_engine = None

        # Alternative Data Engine
        if ALTERNATIVE_AVAILABLE:
            try:
                self.alternative_engine = AlternativeDataEngine()
                engines_initialized += 1
                logger.debug("Alternative data engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize alternative data engine: {e}")
                self.alternative_engine = None

        # Advanced Math Engine
        if ADVANCED_AVAILABLE:
            try:
                self.advanced_engine = get_advanced_engine()
                engines_initialized += 1
                logger.debug("Advanced math engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced math engine: {e}")
                self.advanced_engine = None

        # Determine if we can use 7-model ensemble
        if engines_initialized == 3:
            logger.info("7-model ensemble initialized successfully (all engines available)")
        elif engines_initialized > 0:
            logger.info(f"Partial 7-model ensemble: {engines_initialized}/3 new engines available")
        else:
            logger.warning("Falling back to 4-model ensemble (no new engines available)")
            self.use_7_model = False

    def _initialize_adapters(self):
        """Initialize standardized model adapters."""
        # Core adapters (always available)
        self.technical_adapter = TechnicalModelAdapter(self.base_engine)
        self.physics_adapter = PhysicsModelAdapter(self.physics_engine)
        self.math_adapter = MathModelAdapter(self.math_engine)
        self.regime_adapter = RegimeModelAdapter(self.hmm_detector)

        # Optional adapters (may not have engines)
        self.macro_adapter = MacroModelAdapter(self.macro_engine) if self.macro_engine else None
        self.alternative_adapter = AlternativeDataAdapter(self.alternative_engine) if self.alternative_engine else None
        self.advanced_adapter = AdvancedMathAdapter(self.advanced_engine) if self.advanced_engine else None

        logger.debug("Model adapters initialized")

    def get_standardized_predictions(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> EnsembleInput:
        """
        Get predictions from all models in standardized format.

        This method provides the unified interface for all models,
        making it easier to aggregate, calibrate, and compare predictions.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            **kwargs: Additional parameters passed to adapters
                - trade_type: TradeType for technical adapter
                - sector: Sector name for macro adapter
                - regime: Market regime for physics/math adapters

        Returns:
            EnsembleInput containing StandardizedModelOutput from all models
        """
        outputs = []

        # Get sector and regime for context
        sector = kwargs.get('sector') or self._get_symbol_sector(symbol)
        kwargs['sector'] = sector

        # Detect regime first (used by other models)
        regime_output = self.regime_adapter.predict_standardized(symbol, df, **kwargs)
        outputs.append(regime_output)

        # Extract regime for other models
        regime = 'neutral'
        if regime_output.raw_output and 'current_regime' in regime_output.raw_output:
            regime = regime_output.raw_output['current_regime']
        kwargs['regime'] = regime

        # Core models (always available)
        technical_output = self.technical_adapter.predict_standardized(symbol, df, **kwargs)
        outputs.append(technical_output)

        physics_output = self.physics_adapter.predict_standardized(symbol, df, **kwargs)
        outputs.append(physics_output)

        math_output = self.math_adapter.predict_standardized(symbol, df, **kwargs)
        outputs.append(math_output)

        # Optional models (if available)
        if self.macro_adapter:
            macro_output = self.macro_adapter.predict_standardized(symbol, df, **kwargs)
            outputs.append(macro_output)

        if self.alternative_adapter:
            alt_output = self.alternative_adapter.predict_standardized(symbol, df, **kwargs)
            outputs.append(alt_output)

        if self.advanced_adapter:
            adv_output = self.advanced_adapter.predict_standardized(symbol, df, **kwargs)
            outputs.append(adv_output)

        return EnsembleInput(
            outputs=outputs,
            symbol=symbol,
            timestamp=datetime.now()
        )

    def aggregate_standardized_predictions(
        self,
        ensemble_input: EnsembleInput
    ) -> Tuple[float, float, int, List[str], List[str]]:
        """
        Aggregate standardized predictions into final ensemble score.

        Args:
            ensemble_input: EnsembleInput from get_standardized_predictions()

        Returns:
            Tuple of (ensemble_score, confidence, model_agreement, reasoning, warnings)
        """
        weights = self._get_weights()
        outputs = ensemble_input.outputs

        # Map model names to weights
        model_weights = {
            'technical': weights.get('base', 0.25),
            'physics': weights.get('physics', 0.18),
            'math': weights.get('math', 0.14),
            'regime': weights.get('regime', 0.13),
            'macro': weights.get('macro', 0.10),
            'alternative': weights.get('alternative', 0.10),
            'advanced': weights.get('advanced', 0.10),
        }

        # Calculate weighted ensemble score
        total_weight = 0.0
        weighted_sum = 0.0
        coverage_sum = 0.0

        for output in outputs:
            weight = model_weights.get(output.model_name, 0.10)
            # Weight by coverage (models with better data get more weight)
            effective_weight = weight * output.coverage
            weighted_sum += output.p_buy * effective_weight
            total_weight += effective_weight
            coverage_sum += output.coverage

        if total_weight > 0:
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = ensemble_input.mean_p_buy

        # Count model agreement
        buy_threshold = 0.55
        buy_votes = sum(1 for o in outputs if o.p_buy > buy_threshold)
        sell_votes = sum(1 for o in outputs if o.p_buy < 0.45)
        model_agreement = max(buy_votes, sell_votes)

        # Aggregate uncertainty for confidence
        avg_uncertainty = sum(o.uncertainty.value for o in outputs) / len(outputs)
        base_confidence = 1.0 - avg_uncertainty

        # Apply agreement bonus
        agreement_bonus = 0.0
        if model_agreement >= 5:
            agreement_bonus = 0.08
        elif model_agreement >= 4:
            agreement_bonus = 0.05
        elif model_agreement >= 3:
            agreement_bonus = 0.02

        confidence = min(0.80, max(0.35, base_confidence + agreement_bonus))

        # Collect all reasoning and warnings
        reasoning = ensemble_input.get_all_reasoning()
        warnings = ensemble_input.get_all_warnings()

        return ensemble_score, confidence, model_agreement, reasoning, warnings

    def _get_weights(self) -> Dict[str, float]:
        """Get weights based on ensemble mode."""
        if self.use_7_model and any([self.macro_engine, self.alternative_engine, self.advanced_engine]):
            return {
                'base': ENSEMBLE_WEIGHTS.base_technical_7m,
                'physics': ENSEMBLE_WEIGHTS.physics_7m,
                'math': ENSEMBLE_WEIGHTS.math_7m,
                'regime': ENSEMBLE_WEIGHTS.regime_7m,
                'macro': ENSEMBLE_WEIGHTS.macro_7m,
                'alternative': ENSEMBLE_WEIGHTS.alternative_7m,
                'advanced': ENSEMBLE_WEIGHTS.advanced_7m
            }
        else:
            return {
                'base': ENSEMBLE_WEIGHTS.base_technical,
                'physics': ENSEMBLE_WEIGHTS.physics,
                'math': ENSEMBLE_WEIGHTS.math,
                'regime': ENSEMBLE_WEIGHTS.regime
            }

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol, mapping to MacroEngine sector names."""
        raw_sector = self.market_features.get_symbol_sector(symbol)
        return self.SECTOR_MAPPING.get(raw_sector, 'Consumer')

    def score_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        trade_type: TradeType,
        news_score: float = 0.5,
        news_reasons: List[str] = None,
        market_context: Optional[dict] = None
    ) -> Optional[EnhancedStockScore]:
        """
        Calculate enhanced score using all models.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV + technical indicators
            trade_type: INTRADAY or SWING
            news_score: Pre-calculated news sentiment (0-1)
            news_reasons: News-based reasons
            market_context: Pre-fetched market context

        Returns:
            EnhancedStockScore or None if insufficient data
        """
        if df.empty or len(df) < 60:  # Need more data for advanced models
            logger.warning(f"{symbol}: Insufficient data for enhanced scoring (need 60+ days)")
            return None

        # Ensure technical indicators are calculated
        if 'rsi' not in df.columns:
            df = TechnicalIndicators.calculate_all(df)

        latest = df.iloc[-1]
        current_price = float(latest['close'])

        # Get latest date
        latest_date = df.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        reasons = list(news_reasons) if news_reasons else []
        warnings = []
        model_votes = {}

        # ===== 1. BASE SCORE (Technical + Momentum + Volume + News) =====
        try:
            base_result = self.base_engine.score_stock(
                symbol=symbol,
                df=df,
                trade_type=trade_type,
                news_score=news_score,
                news_reasons=[],  # We'll collect our own
                market_context=market_context
            )
            base_score = base_result.raw_score if base_result else 0.5
            if base_result:
                reasons.extend(base_result.reasons)
                model_votes['base'] = 'BUY' if base_score > 0.55 else ('SELL' if base_score < 0.45 else 'HOLD')
        except Exception as e:
            logger.warning(f"{symbol}: Base scoring failed: {e}")
            base_score = 0.5
            model_votes['base'] = 'HOLD'

        # ===== 2. PHYSICS SCORE =====
        try:
            physics_result = self.physics_engine.score(symbol, df)
            physics_score = physics_result.composite_score

            # Add physics insights
            reasons.extend(physics_result.bullish_reasons)
            warnings.extend(physics_result.warnings)

            # Physics vote
            if physics_score > 0.55 and physics_result.recommended_strategy in ['momentum', 'breakout']:
                model_votes['physics'] = 'BUY'
            elif physics_score < 0.45 or physics_result.recommended_strategy == 'avoid':
                model_votes['physics'] = 'SELL'
            else:
                model_votes['physics'] = 'HOLD'

            if physics_result.bearish_reasons:
                for r in physics_result.bearish_reasons[:2]:
                    warnings.append(r)

        except Exception as e:
            logger.warning(f"{symbol}: Physics scoring failed: {e}")
            physics_score = 0.5
            model_votes['physics'] = 'HOLD'
            physics_result = None

        # ===== 3. MATH SCORE =====
        try:
            math_result = self.math_engine.score(symbol, df)
            math_score = math_result.composite_score

            # Add math insights
            for insight in math_result.insights[:3]:
                reasons.append(insight)
            warnings.extend(math_result.warnings)

            # Math vote
            if math_score > 0.55 and math_result.predictability > 0.3:
                model_votes['math'] = 'BUY'
            elif math_score < 0.45 or math_result.predictability < 0.15:
                model_votes['math'] = 'SELL'
            else:
                model_votes['math'] = 'HOLD'

            market_predictability = math_result.predictability
            recommended_strategy = math_result.recommended_strategy

        except Exception as e:
            logger.warning(f"{symbol}: Math scoring failed: {e}")
            math_score = 0.5
            model_votes['math'] = 'HOLD'
            market_predictability = 0.5
            recommended_strategy = 'neutral'
            math_result = None

        # ===== 4. REGIME SCORE =====
        try:
            regime_result = self.hmm_detector.detect_regime(df)
            regime_score = regime_result.probability_score
            regime = regime_result.current_regime.value
            regime_stability = regime_result.regime_stability

            # Add regime insight
            reasons.append(f"[REGIME] {regime_result.reason}")

            # Regime vote
            if regime_result.current_regime == MarketRegime.TRENDING_BULL:
                model_votes['regime'] = 'BUY'
            elif regime_result.current_regime == MarketRegime.TRENDING_BEAR:
                model_votes['regime'] = 'SELL'
            elif regime_result.current_regime == MarketRegime.CHOPPY:
                model_votes['regime'] = 'AVOID'
                warnings.append("[REGIME] Choppy market - reduce position size")
            else:
                model_votes['regime'] = 'HOLD'

            if regime_stability < 0.5:
                warnings.append(f"[REGIME] Unstable regime (stability: {regime_stability:.0%})")

        except Exception as e:
            logger.warning(f"{symbol}: Regime detection failed: {e}")
            regime_score = 0.5
            regime = 'unknown'
            regime_stability = 0.5
            model_votes['regime'] = 'HOLD'

        # ===== 5. MACRO SCORE (NEW - 7-model ensemble) =====
        macro_score = 0.5
        macro_signal = None
        sector = self._get_symbol_sector(symbol)

        if self.use_7_model and self.macro_engine:
            try:
                macro_signal = self.macro_engine.get_sector_macro_signal(sector)

                # Convert macro signal (-1 to +1) to score (0 to 1)
                macro_score = (macro_signal.signal + 1) / 2

                # Add macro insights to reasons
                if macro_signal.direction == 'bullish':
                    reasons.append(f"[MACRO] {sector} sector: Bullish ({macro_signal.confidence:.0%} confidence)")
                    model_votes['macro'] = 'BUY'
                elif macro_signal.direction == 'bearish':
                    warnings.append(f"[MACRO] {sector} sector: Bearish headwinds")
                    model_votes['macro'] = 'SELL'
                else:
                    model_votes['macro'] = 'HOLD'

                # Add primary macro drivers
                if hasattr(macro_signal, 'primary_drivers') and macro_signal.primary_drivers:
                    for driver in macro_signal.primary_drivers[:2]:
                        reasons.append(f"[MACRO] {driver}")

            except Exception as e:
                logger.warning(f"{symbol}: Macro scoring failed: {e}")
                macro_score = 0.5
                model_votes['macro'] = 'HOLD'

        # ===== 6. ALTERNATIVE DATA SCORE (NEW - 7-model ensemble) =====
        alternative_score = 0.5
        alternative_signal = None

        if self.use_7_model and self.alternative_engine:
            try:
                alternative_signal = self.alternative_engine.analyze(symbol, df)
                alternative_score = alternative_signal.composite_probability

                # Add alternative data insights
                if alternative_signal.direction == 1:
                    model_votes['alternative'] = 'BUY'
                elif alternative_signal.direction == -1:
                    model_votes['alternative'] = 'SELL'
                else:
                    model_votes['alternative'] = 'HOLD'

                # Add key alternative data reasons
                if hasattr(alternative_signal, 'reasoning') and alternative_signal.reasoning:
                    for reason in alternative_signal.reasoning[:2]:
                        reasons.append(reason)

                if hasattr(alternative_signal, 'signal_strength'):
                    if alternative_signal.signal_strength in ['STRONG_BUY', 'STRONG_SELL']:
                        reasons.append(f"[ALT] {alternative_signal.signal_strength} from alternative data")

            except Exception as e:
                logger.warning(f"{symbol}: Alternative data scoring failed: {e}")
                alternative_score = 0.5
                model_votes['alternative'] = 'HOLD'

        # ===== 7. ADVANCED MATH SCORE (NEW - 7-model ensemble) =====
        advanced_score = 0.5
        advanced_signal = None

        if self.use_7_model and self.advanced_engine:
            try:
                advanced_result = self.advanced_engine.predict(df, symbol)
                advanced_signal = advanced_result

                # Get probability (already 0-1)
                advanced_score = advanced_result.probability

                # Add advanced insights
                if hasattr(advanced_result, 'signal'):
                    signal_value = advanced_result.signal.value if hasattr(advanced_result.signal, 'value') else str(advanced_result.signal)
                    if signal_value in ['STRONG_BUY', 'BUY']:
                        model_votes['advanced'] = 'BUY'
                        reasons.append(f"[ADV] {signal_value} from 6-model math ensemble")
                    elif signal_value in ['STRONG_SELL', 'SELL']:
                        model_votes['advanced'] = 'SELL'
                        warnings.append(f"[ADV] {signal_value} from math models")
                    else:
                        model_votes['advanced'] = 'HOLD'

                # Add regime info from advanced engine
                if hasattr(advanced_result, 'current_regime') and advanced_result.current_regime:
                    regime_value = advanced_result.current_regime.value if hasattr(advanced_result.current_regime, 'value') else str(advanced_result.current_regime)
                    reasons.append(f"[ADV] Math regime: {regime_value}")

            except Exception as e:
                logger.warning(f"{symbol}: Advanced math scoring failed: {e}")
                advanced_score = 0.5
                model_votes['advanced'] = 'HOLD'

        # ===== ENSEMBLE CALCULATION =====
        # Using centralized weights from config/thresholds.py
        weights = self._get_weights()
        total_models = len(model_votes)

        if self.use_7_model and total_models >= 5:
            # 7-model ensemble (or partial)
            ensemble_score = (
                weights.get('base', 0.25) * base_score +
                weights.get('physics', 0.18) * physics_score +
                weights.get('math', 0.14) * math_score +
                weights.get('regime', 0.13) * regime_score +
                weights.get('macro', 0.10) * macro_score +
                weights.get('alternative', 0.10) * alternative_score +
                weights.get('advanced', 0.10) * advanced_score
            )
        else:
            # 4-model fallback
            ensemble_score = (
                weights.get('base', 0.35) * base_score +
                weights.get('physics', 0.25) * physics_score +
                weights.get('math', 0.20) * math_score +
                weights.get('regime', 0.20) * regime_score
            )
            total_models = 4

        # Count model agreement
        buy_votes = sum(1 for v in model_votes.values() if v == 'BUY')
        sell_votes = sum(1 for v in model_votes.values() if v == 'SELL')
        model_agreement = max(buy_votes, sell_votes)

        # ===== SIGNAL DETERMINATION =====
        # Adjust agreement requirements based on model count
        if total_models >= 7:
            strong_agreement = 5  # 5 out of 7
            moderate_agreement = 3  # 3 out of 7
        else:
            strong_agreement = 3  # 3 out of 4
            moderate_agreement = 2  # 2 out of 4

        # Using centralized thresholds from config/thresholds.py
        # SELL signals disabled - see SIGNAL_THRESHOLDS.sell_signals_enabled
        if ensemble_score >= self.THRESHOLDS['strong_buy'] and buy_votes >= strong_agreement:
            signal = SignalType.BUY
            signal_strength = 'strong'
            reasons.insert(0, f"[ENSEMBLE] Strong BUY: {buy_votes}/{total_models} models agree")
        elif ensemble_score >= self.THRESHOLDS['buy'] and buy_votes >= moderate_agreement:
            signal = SignalType.BUY
            signal_strength = 'moderate'
            reasons.insert(0, f"[ENSEMBLE] BUY signal: {buy_votes}/{total_models} models agree")
        elif ensemble_score >= self.THRESHOLDS['hold_upper']:
            signal = SignalType.BUY
            signal_strength = 'weak'
            reasons.insert(0, f"[ENSEMBLE] Weak BUY signal - consider smaller position")
        else:
            # IMPORTANT: We return HOLD instead of SELL
            # Backtest showed SELL signals have 28.6% accuracy = actually bullish!
            signal = SignalType.HOLD
            signal_strength = 'none'
            if sell_votes >= moderate_agreement:
                warnings.append(f"[CAUTION] SELL signals disabled - historically inverted (71% wrong)")

        # ===== CONFIDENCE CALIBRATION =====
        # Using centralized calibration from config/thresholds.py
        atr_pct = float(latest.get('atr_pct', 2.0))
        avg_value = float((df['close'] * df['volume']).tail(20).mean())
        liquidity_cr = avg_value / 1e7

        # Get base confidence from centralized function (with total_models for proper scaling)
        confidence = get_confidence_for_score(ensemble_score, model_agreement, total_models)

        # Apply risk penalties using centralized function from config/thresholds.py
        confidence, risk_warnings = apply_risk_penalties(
            confidence,
            atr_pct=atr_pct,
            liquidity_cr=liquidity_cr,
            regime_stability=regime_stability,
            predictability=market_predictability
        )
        warnings.extend(risk_warnings)

        # Apply confidence bounds from centralized config
        confidence = round(
            min(CONFIDENCE_CALIBRATION.max_confidence,
                max(CONFIDENCE_CALIBRATION.min_confidence, confidence)),
            3
        )

        # ===== TRADING LEVELS =====
        atr = float(latest.get('atr', current_price * 0.02))
        entry_price, stop_loss, target_price = self._calculate_levels(
            current_price, atr, signal, trade_type
        )

        risk_reward = abs(target_price - entry_price) / abs(entry_price - stop_loss) if entry_price != stop_loss else 0

        return EnhancedStockScore(
            symbol=symbol,
            date=latest_date,
            trade_type=trade_type,
            # Original 4 model scores
            base_score=base_score,
            physics_score=physics_score,
            math_score=math_score,
            regime_score=regime_score,
            # New 7-model ensemble scores
            macro_score=macro_score,
            alternative_score=alternative_score,
            advanced_score=advanced_score,
            # Ensemble results
            ensemble_score=ensemble_score,
            model_agreement=model_agreement,
            total_models=total_models,
            # Confidence and signal
            confidence=confidence,
            signal=signal,
            signal_strength=signal_strength,
            # Trading levels
            current_price=current_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward=risk_reward,
            # Risk metrics
            atr_pct=atr_pct,
            liquidity_cr=liquidity_cr,
            regime=regime,
            regime_stability=regime_stability,
            # Insights
            reasons=reasons[:15],  # Limit reasons
            warnings=warnings[:5],  # Limit warnings
            model_votes=model_votes,
            recommended_strategy=recommended_strategy,
            market_predictability=market_predictability,
            # Sector and detailed signals
            sector=sector,
            macro_signal=macro_signal,
            alternative_signal=alternative_signal,
            advanced_signal=advanced_signal
        )

    def _calibrate_confidence(self, ensemble_score: float, model_agreement: int) -> float:
        """
        Calibrate raw ensemble score to realistic confidence.

        Based on backtest results:
        - 57.6% accuracy on BUY signals
        - Best case (all models agree) might reach 62-65%
        """
        # Base calibration from score
        for (low, high), conf in self.CONFIDENCE_CALIBRATION.items():
            if low <= ensemble_score < high:
                base_conf = conf
                break
        else:
            base_conf = 0.50

        # Agreement bonus
        agreement_bonus = {
            4: 0.08,  # All 4 models agree
            3: 0.05,  # 3 models agree
            2: 0.02,  # 2 models agree
            1: 0.00,  # Only 1 model
            0: -0.05  # No agreement (shouldn't happen)
        }

        confidence = base_conf + agreement_bonus.get(model_agreement, 0)

        return np.clip(confidence, 0.35, 0.80)

    def _calculate_levels(
        self,
        current_price: float,
        atr: float,
        signal: SignalType,
        trade_type: TradeType
    ) -> Tuple[float, float, float]:
        """Calculate entry, stop-loss, and target levels."""

        if trade_type == TradeType.INTRADAY:
            stop_mult = 1.0
            target_mult = 1.5
        else:  # SWING
            stop_mult = 1.5
            target_mult = 2.5

        if signal == SignalType.BUY:
            entry = current_price
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)
        else:  # HOLD or (disabled) SELL
            entry = current_price
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)

        return round(entry, 2), round(stop, 2), round(target, 2)

    def rank_stocks(
        self,
        scores: List[EnhancedStockScore],
        top_n: int = 10,
        min_confidence: float = 0.50,
        min_agreement: int = 2,
        signal_filter: Optional[SignalType] = SignalType.BUY
    ) -> List[EnhancedStockScore]:
        """
        Rank and filter enhanced stock scores.

        Args:
            scores: List of EnhancedStockScore objects
            top_n: Number of top picks
            min_confidence: Minimum confidence threshold
            min_agreement: Minimum number of models that must agree
            signal_filter: Only include this signal type

        Returns:
            Sorted list of top picks
        """
        filtered = []
        for s in scores:
            # Apply filters
            if s.confidence < min_confidence:
                continue
            if signal_filter and s.signal != signal_filter:
                continue
            if s.model_agreement < min_agreement:
                continue
            if s.atr_pct > self.settings.max_atr_percent:
                continue
            if s.liquidity_cr < self.settings.min_liquidity_cr:
                continue
            # Skip choppy regime
            if s.regime == 'choppy':
                continue

            filtered.append(s)

        # Sort by confidence * model_agreement (both matter)
        filtered.sort(key=lambda x: (x.model_agreement, x.confidence), reverse=True)

        # Apply sector diversification
        diversified = self._apply_sector_diversification(filtered, top_n)

        return diversified[:top_n]

    def _apply_sector_diversification(
        self,
        scores: List[EnhancedStockScore],
        top_n: int
    ) -> List[EnhancedStockScore]:
        """Limit stocks per sector for diversification."""
        max_per_sector = self.settings.max_sector_concentration
        sector_counts = {}
        result = []

        for score in scores:
            sector = self.market_features.get_symbol_sector(score.symbol)
            current_count = sector_counts.get(sector, 0)

            if current_count < max_per_sector:
                result.append(score)
                sector_counts[sector] = current_count + 1

            if len(result) >= top_n * 2:
                break

        return result


def convert_enhanced_to_base(enhanced: EnhancedStockScore) -> StockScore:
    """Convert EnhancedStockScore to base StockScore for compatibility."""
    return StockScore(
        symbol=enhanced.symbol,
        date=enhanced.date,
        trade_type=enhanced.trade_type,
        technical_score=enhanced.base_score,  # Use base as technical
        momentum_score=enhanced.physics_score,  # Use physics as momentum proxy
        news_score=0.5,  # News is in base
        sector_score=enhanced.math_score,  # Use math as sector proxy
        volume_score=enhanced.regime_score,  # Use regime as volume proxy
        raw_score=enhanced.ensemble_score,
        confidence=enhanced.confidence,
        signal=enhanced.signal,
        current_price=enhanced.current_price,
        entry_price=enhanced.entry_price,
        stop_loss=enhanced.stop_loss,
        target_price=enhanced.target_price,
        risk_reward=enhanced.risk_reward,
        atr_pct=enhanced.atr_pct,
        liquidity_cr=enhanced.liquidity_cr,
        reasons=enhanced.reasons
    )
