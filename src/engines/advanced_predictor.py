"""
Advanced Prediction Engine

Master engine combining all prediction layers:
- Physics-Math Engine (momentum, spring, energy, fourier, fractal, entropy)
- ML/Statistical Engine (HMM regime, Bayesian, ensemble, calibration)
- Microstructure Engine (volume profile, smart money, intraday, sentiment)

Uses regime-conditioned weighting for dynamic adaptation.
Target: 70%+ accuracy through multi-factor confirmation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from ..physics import PhysicsEngine, PhysicsScore
from ..math_models import MathEngine, MathScore
from ..ml import MLEngine, MLScore
from ..microstructure import MicrostructureEngine, MicrostructureScore
from ..alternative_data import AlternativeDataEngine, AlternativeDataScore
from .signal_amplifier import SignalAmplifier, AmplifiedSignal


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradeType(Enum):
    """Trading timeframe."""
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"


@dataclass
class PredictionConfidence:
    """Confidence metrics for prediction."""
    probability: float           # 0-1 probability of upward move
    confidence_level: str        # 'high', 'medium', 'low'
    uncertainty: float           # Standard deviation of estimates
    lower_bound: float           # 95% CI lower
    upper_bound: float           # 95% CI upper
    model_agreement: float       # How much models agree (0-1)
    regime_confidence: float     # Confidence in regime detection


@dataclass
class TradeRecommendation:
    """Actionable trade recommendation."""
    action: SignalStrength
    trade_type: TradeType
    entry_timing: str            # 'immediate', 'wait_pullback', 'breakout_confirm'
    target_pct: float            # Expected move percentage
    stop_loss_pct: float         # Suggested stop loss
    position_size: str           # 'full', 'half', 'quarter'
    time_horizon: str            # '1d', '3-5d', '1-2w'
    risk_reward: float           # Risk/reward ratio


@dataclass
class LayerScores:
    """Scores from each prediction layer."""
    physics_score: float
    math_score: float
    ml_score: float
    microstructure_score: float
    alternative_score: float = 0.50  # Alternative data score

    # Detailed results
    physics_result: Optional[PhysicsScore] = None
    math_result: Optional[MathScore] = None
    ml_result: Optional[MLScore] = None
    microstructure_result: Optional[MicrostructureScore] = None
    alternative_result: Optional[AlternativeDataScore] = None


@dataclass
class AdvancedPrediction:
    """Complete prediction output."""
    symbol: str
    timestamp: datetime

    # Core prediction
    final_probability: float     # Calibrated probability (0-1)
    signal: SignalStrength
    confidence: PredictionConfidence
    recommendation: TradeRecommendation

    # Market context
    detected_regime: str
    regime_description: str

    # Layer breakdown
    layer_scores: LayerScores
    layer_weights: Dict[str, float]

    # Signals and reasoning
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]
    primary_reason: str

    # Strategy guidance
    optimal_strategy: str        # 'momentum', 'mean_reversion', 'breakout', 'wait'
    strategy_confidence: float


class DomainPredictionEngine:
    """
    Domain-based prediction engine combining all layers.

    NOTE: This was renamed from AdvancedPredictionEngine to avoid confusion
    with src/core/advanced_engine.py which uses mathematical models.

    This engine uses DOMAIN-SPECIFIC models:
    - Physics Engine (momentum, spring, energy, network)
    - Math Engine (Fourier, Hurst, entropy, statistical mechanics)
    - ML Engine (HMM regime, Bayesian, ensemble)
    - Microstructure Engine (volume profile, smart money, intraday)
    - Alternative Data Engine (options flow, institutional, earnings)

    For MATHEMATICAL models (PCA, Wavelet, Kalman, DQN, Factor), use:
    src/core/advanced_engine.AdvancedPredictionEngine

    Architecture:
    1. Detect market regime (HMM)
    2. Score each layer independently
    3. Apply regime-conditioned weights
    4. Calibrate final probability
    5. Generate actionable recommendation

    Target: 70%+ accuracy through:
    - Multi-factor confirmation
    - Regime filtering
    - Proper calibration
    """

    # Regime-dependent layer weights (now including alternative data)
    # Alternative data gets significant weight as it has highest accuracy (70%+ when aligned)
    REGIME_WEIGHTS = {
        'TRENDING_BULL': {
            'physics': 0.15,
            'math': 0.10,
            'ml': 0.15,
            'microstructure': 0.20,
            'alternative': 0.40  # High weight - MTF confluence strongest in trends
        },
        'TRENDING_BEAR': {
            'physics': 0.15,
            'math': 0.10,
            'ml': 0.20,
            'microstructure': 0.15,
            'alternative': 0.40  # High weight - MTF confluence strongest in trends
        },
        'RANGING': {
            'physics': 0.25,  # Spring reversion stronger
            'math': 0.15,
            'ml': 0.15,
            'microstructure': 0.15,
            'alternative': 0.30  # Options/Institutional flow good in ranges
        },
        'CHOPPY': {
            'physics': 0.10,
            'math': 0.15,
            'ml': 0.20,
            'microstructure': 0.20,
            'alternative': 0.35  # Event-driven setups work in choppy markets
        },
        'TRANSITION': {
            'physics': 0.15,
            'math': 0.15,
            'ml': 0.20,
            'microstructure': 0.15,
            'alternative': 0.35  # Wait for MTF alignment in transitions
        }
    }

    # Default weights when regime unknown
    DEFAULT_WEIGHTS = {
        'physics': 0.15,
        'math': 0.10,
        'ml': 0.20,
        'microstructure': 0.20,
        'alternative': 0.35
    }

    # Timeframe-specific adjustments
    TIMEFRAME_ADJUSTMENTS = {
        'intraday': {
            'physics': -0.05,      # Less physics
            'math': 0.0,
            'ml': 0.0,
            'microstructure': +0.05,  # More microstructure
            'alternative': 0.0     # Keep same
        },
        'swing': {
            'physics': 0.0,
            'math': 0.0,
            'ml': 0.0,
            'microstructure': 0.0,
            'alternative': 0.0
        },
        'positional': {
            'physics': +0.05,
            'math': +0.05,
            'ml': 0.0,
            'microstructure': -0.10,
            'alternative': +0.05  # MTF confluence more valuable for positional
        }
    }

    def __init__(self, use_ml_ensemble: bool = False, use_alternative_data: bool = True):
        """
        Initialize all prediction engines.

        Args:
            use_ml_ensemble: Kept for backward compatibility (not used)
            use_alternative_data: Whether to use alternative data sources (default True)
        """
        self.physics_engine = PhysicsEngine()
        self.math_engine = MathEngine()
        self.ml_engine = MLEngine()
        self.microstructure_engine = MicrostructureEngine()
        self.signal_amplifier = SignalAmplifier()

        # Alternative data engine for 70%+ accuracy
        self.use_alternative_data = use_alternative_data
        if use_alternative_data:
            self.alternative_engine = AlternativeDataEngine()
        else:
            self.alternative_engine = None

        # Calibration history for adaptive learning
        self._prediction_history: List[Tuple[float, bool]] = []
        self._calibration_adjustment = 0.0

    def get_weights(
        self,
        regime: str,
        timeframe: str = 'swing'
    ) -> Dict[str, float]:
        """Get layer weights based on regime and timeframe."""
        # Start with regime weights
        base_weights = self.REGIME_WEIGHTS.get(regime, self.DEFAULT_WEIGHTS).copy()

        # Apply timeframe adjustments
        adjustments = self.TIMEFRAME_ADJUSTMENTS.get(timeframe.lower(), {})
        for layer, adj in adjustments.items():
            if layer in base_weights:
                base_weights[layer] += adj

        # Normalize to sum to 1
        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()}

    def calculate_model_agreement(
        self,
        scores: List[float]
    ) -> float:
        """
        Calculate how much models agree.

        High agreement = all scores similar
        Low agreement = scores diverge
        """
        if len(scores) < 2:
            return 1.0

        # Use coefficient of variation (lower = more agreement)
        mean = np.mean(scores)
        std = np.std(scores)

        if mean == 0:
            return 0.5

        cv = std / abs(mean)

        # Convert to 0-1 scale (lower CV = higher agreement)
        agreement = max(0, 1 - cv)

        return agreement

    def calculate_uncertainty(
        self,
        scores: List[float],
        regime_confidence: float
    ) -> Tuple[float, float, float]:
        """
        Calculate prediction uncertainty.

        Returns (uncertainty, lower_bound, upper_bound)
        """
        mean = np.mean(scores)
        std = np.std(scores)

        # Adjust uncertainty by regime confidence
        adjusted_std = std * (2 - regime_confidence)

        # 95% confidence interval
        z = 1.96
        lower = max(0, mean - z * adjusted_std)
        upper = min(1, mean + z * adjusted_std)

        return adjusted_std, lower, upper

    def determine_signal_strength(
        self,
        probability: float,
        confidence: float,
        agreement: float
    ) -> SignalStrength:
        """Determine signal strength from probability and confidence."""
        # Adjust thresholds based on confidence
        confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 to 1.0

        # Strong signals require high probability AND confidence
        if probability >= 0.70 and agreement >= 0.7:
            return SignalStrength.STRONG_BUY
        elif probability >= 0.62:
            return SignalStrength.BUY
        elif probability >= 0.55:
            return SignalStrength.WEAK_BUY
        elif probability <= 0.30 and agreement >= 0.7:
            return SignalStrength.STRONG_SELL
        elif probability <= 0.38:
            return SignalStrength.SELL
        elif probability <= 0.45:
            return SignalStrength.WEAK_SELL
        else:
            return SignalStrength.NEUTRAL

    def generate_recommendation(
        self,
        signal: SignalStrength,
        probability: float,
        regime: str,
        timeframe: str,
        layer_scores: LayerScores
    ) -> TradeRecommendation:
        """Generate actionable trade recommendation."""
        # Determine trade type
        if timeframe.lower() == 'intraday':
            trade_type = TradeType.INTRADAY
            time_horizon = '1d'
        elif timeframe.lower() == 'positional':
            trade_type = TradeType.POSITIONAL
            time_horizon = '1-2w'
        else:
            trade_type = TradeType.SWING
            time_horizon = '3-5d'

        # Entry timing based on regime and signal
        if regime == 'TRENDING_BULL' and signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
            entry_timing = 'immediate'
        elif regime == 'RANGING':
            entry_timing = 'wait_pullback'
        elif regime == 'CHOPPY':
            entry_timing = 'breakout_confirm'
        else:
            entry_timing = 'immediate' if probability > 0.6 else 'wait_pullback'

        # Target and stop based on signal strength
        if signal == SignalStrength.STRONG_BUY:
            target_pct = 5.0 if trade_type == TradeType.INTRADAY else 8.0
            stop_loss_pct = 2.0
            position_size = 'full'
        elif signal == SignalStrength.BUY:
            target_pct = 3.0 if trade_type == TradeType.INTRADAY else 5.0
            stop_loss_pct = 1.5
            position_size = 'half'
        elif signal == SignalStrength.WEAK_BUY:
            target_pct = 2.0 if trade_type == TradeType.INTRADAY else 3.0
            stop_loss_pct = 1.0
            position_size = 'quarter'
        elif signal == SignalStrength.STRONG_SELL:
            target_pct = -5.0 if trade_type == TradeType.INTRADAY else -8.0
            stop_loss_pct = 2.0
            position_size = 'full'
        elif signal == SignalStrength.SELL:
            target_pct = -3.0 if trade_type == TradeType.INTRADAY else -5.0
            stop_loss_pct = 1.5
            position_size = 'half'
        elif signal == SignalStrength.WEAK_SELL:
            target_pct = -2.0 if trade_type == TradeType.INTRADAY else -3.0
            stop_loss_pct = 1.0
            position_size = 'quarter'
        else:
            target_pct = 0.0
            stop_loss_pct = 0.0
            position_size = 'none'

        # Risk/reward ratio
        risk_reward = abs(target_pct / stop_loss_pct) if stop_loss_pct > 0 else 0

        return TradeRecommendation(
            action=signal,
            trade_type=trade_type,
            entry_timing=entry_timing,
            target_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
            position_size=position_size,
            time_horizon=time_horizon,
            risk_reward=risk_reward
        )

    def determine_optimal_strategy(
        self,
        regime: str,
        physics_result: Optional[PhysicsScore],
        math_result: Optional[MathScore]
    ) -> Tuple[str, float]:
        """Determine optimal trading strategy based on analysis."""
        # Default strategy
        strategy = 'momentum'
        confidence = 0.5

        # Check Hurst exponent from math models
        if math_result and math_result.fractal_result:
            hurst = math_result.fractal_result.hurst_exponent
            if hurst > 0.55:
                strategy = 'momentum'
                confidence = min(0.9, 0.5 + (hurst - 0.5) * 2)
            elif hurst < 0.45:
                strategy = 'mean_reversion'
                confidence = min(0.9, 0.5 + (0.5 - hurst) * 2)

        # Regime-based adjustment
        if regime == 'TRENDING_BULL' or regime == 'TRENDING_BEAR':
            strategy = 'momentum'
            confidence = max(confidence, 0.6)
        elif regime == 'RANGING':
            strategy = 'mean_reversion'
            confidence = max(confidence, 0.6)
        elif regime == 'CHOPPY':
            strategy = 'wait'
            confidence = 0.4

        # Check physics for spring tension
        if physics_result and physics_result.spring_result:
            displacement = getattr(physics_result.spring_result, 'displacement', 0)
            if abs(displacement) > 0.05:  # 5% displacement from equilibrium
                strategy = 'mean_reversion'
                confidence = max(confidence, 0.7)

        return strategy, confidence

    def collect_factors(
        self,
        physics_result: Optional[PhysicsScore],
        math_result: Optional[MathScore],
        ml_result: Optional[MLScore],
        micro_result: Optional[MicrostructureScore],
        alt_result: Optional[AlternativeDataScore] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Collect bullish, bearish factors and warnings from all layers."""
        bullish = []
        bearish = []
        warnings = []

        # Physics factors (uses bullish_reasons/bearish_reasons)
        if physics_result:
            bullish.extend(getattr(physics_result, 'bullish_reasons', []))
            bearish.extend(getattr(physics_result, 'bearish_reasons', []))
            warnings.extend(getattr(physics_result, 'warnings', []))

        # Math factors (uses insights - classify based on content)
        if math_result:
            insights = getattr(math_result, 'insights', [])
            for insight in insights:
                # Simple heuristic: if contains positive keywords, it's bullish
                lower = insight.lower()
                if any(w in lower for w in ['bullish', 'trending', 'momentum', 'persistent']):
                    bullish.append(f'[MATH] {insight}')
                elif any(w in lower for w in ['bearish', 'reverting', 'random', 'unpredictable']):
                    bearish.append(f'[MATH] {insight}')
            warnings.extend(getattr(math_result, 'warnings', []))

        # ML factors
        if ml_result:
            bullish.extend(getattr(ml_result, 'bullish_factors', []))
            bearish.extend(getattr(ml_result, 'bearish_factors', []))
            warnings.extend(getattr(ml_result, 'warnings', []))

        # Microstructure factors
        if micro_result:
            bullish.extend(getattr(micro_result, 'bullish_factors', []))
            bearish.extend(getattr(micro_result, 'bearish_factors', []))
            warnings.extend(getattr(micro_result, 'warnings', []))

        # Alternative data factors
        if alt_result:
            for reason in alt_result.reasoning:
                if alt_result.direction == 1:
                    bullish.append(reason)
                elif alt_result.direction == -1:
                    bearish.append(reason)

            # Add specific signals
            if alt_result.confluence_signal and alt_result.confluence_signal.confluence_score >= 0.8:
                bullish.insert(0, f"[MTF] Strong multi-timeframe alignment ({alt_result.confluence_signal.confluence_score:.0%})")

            if alt_result.earnings_signals:
                for sig in alt_result.earnings_signals[:2]:
                    if sig.direction == 1:
                        bullish.append(f"[EVENT] {sig.reasoning}")
                    elif sig.direction == -1:
                        bearish.append(f"[EVENT] {sig.reasoning}")

        return bullish, bearish, warnings

    def generate_primary_reason(
        self,
        signal: SignalStrength,
        bullish: List[str],
        bearish: List[str],
        regime: str
    ) -> str:
        """Generate primary reason for the recommendation."""
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            if bullish:
                return f"{regime} regime: {bullish[0]}"
            return f"{regime} regime with bullish bias"
        elif signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            if bearish:
                return f"{regime} regime: {bearish[0]}"
            return f"{regime} regime with bearish bias"
        else:
            return f"{regime} regime - mixed signals, caution advised"

    def get_regime_description(self, regime: str) -> str:
        """Get human-readable regime description."""
        descriptions = {
            'TRENDING_BULL': 'Strong uptrend - momentum strategies favored',
            'TRENDING_BEAR': 'Strong downtrend - defensive positioning',
            'RANGING': 'Range-bound market - mean reversion opportunities',
            'CHOPPY': 'Choppy/volatile - reduced position sizing advised',
            'TRANSITION': 'Market in transition - wait for confirmation'
        }
        return descriptions.get(regime, 'Unknown market condition')

    def calibrate_probability(
        self,
        raw_probability: float,
        regime: str,
        agreement: float
    ) -> float:
        """
        Apply final calibration to probability.

        Adjusts for:
        - Historical calibration error
        - Regime-specific biases
        - Model disagreement
        """
        prob = raw_probability

        # Apply historical adjustment
        prob += self._calibration_adjustment

        # Reduce confidence when models disagree
        if agreement < 0.5:
            prob = 0.5 + (prob - 0.5) * agreement

        # Regime-specific adjustments
        if regime == 'CHOPPY':
            # Pull toward 0.5 in choppy markets
            prob = 0.5 + (prob - 0.5) * 0.7

        # Ensure bounds
        return np.clip(prob, 0.05, 0.95)

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = 'swing',
        sector_returns: Optional[Dict[str, float]] = None
    ) -> AdvancedPrediction:
        """
        Generate comprehensive prediction.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            timeframe: 'intraday', 'swing', or 'positional'
            sector_returns: Optional sector return data for network analysis

        Returns:
            AdvancedPrediction with complete analysis
        """
        timestamp = datetime.now()

        # 1. Detect regime using ML engine
        ml_result = self.ml_engine.score(symbol, df)
        detected_regime = ml_result.current_regime
        regime_confidence = ml_result.regime_stability

        # 2. Score each layer
        physics_result = self.physics_engine.score(
            symbol, df,
            regime=detected_regime,
            sector_data=None  # sector_returns not used in this simplified version
        )

        math_result = self.math_engine.score(
            symbol, df,
            regime=detected_regime
        )

        micro_result = self.microstructure_engine.score(
            symbol, df,
            timeframe=timeframe
        )

        # 2b. Alternative data analysis (if enabled)
        if self.use_alternative_data and self.alternative_engine:
            try:
                alt_result = self.alternative_engine.analyze(symbol, df, include_options=True)
                alt_composite = alt_result.composite_probability
            except Exception as e:
                alt_result = None
                alt_composite = 0.50
        else:
            alt_result = None
            alt_composite = 0.50

        # 3. Get regime-adjusted weights
        weights = self.get_weights(detected_regime, timeframe)

        # 4. Calculate weighted composite score
        # Get composite scores (handle different field names)
        physics_composite = physics_result.composite_score
        math_composite = math_result.composite_score
        ml_composite = ml_result.calibrated_composite
        micro_composite = micro_result.composite_score

        scores = [
            physics_composite,
            math_composite,
            ml_composite,
            micro_composite,
            alt_composite
        ]

        # 5. AMPLIFY SIGNALS - This is critical for actionable predictions
        amplified = self.signal_amplifier.amplify(
            physics_score=physics_composite,
            math_score=math_composite,
            ml_score=ml_composite,
            micro_score=micro_composite,
            df=df,
            regime=detected_regime
        )

        # 5b. Calculate weighted score including alternative data
        # Alternative data gets high weight when it has strong signals
        base_weighted = amplified.amplified_probability

        if self.use_alternative_data and alt_result:
            # Blend amplified signal with alternative data
            alt_weight = weights.get('alternative', 0.35)
            other_weight = 1 - alt_weight

            # Boost if alternative data agrees with amplified signal
            alt_direction = 1 if alt_composite > 0.55 else (-1 if alt_composite < 0.45 else 0)
            amp_direction = 1 if base_weighted > 0.55 else (-1 if base_weighted < 0.45 else 0)

            if alt_direction == amp_direction and alt_direction != 0:
                # Agreement - boost confidence
                weighted_score = (base_weighted * other_weight + alt_composite * alt_weight) * 1.1
                # Cap boost
                if weighted_score > 0.5:
                    weighted_score = min(weighted_score, 0.80)
                else:
                    weighted_score = max(weighted_score, 0.20)
            elif alt_direction != 0 and amp_direction != 0 and alt_direction != amp_direction:
                # Disagreement - pull toward neutral
                weighted_score = 0.50 + (base_weighted - 0.50) * 0.5
            else:
                # Neutral - blend normally
                weighted_score = base_weighted * other_weight + alt_composite * alt_weight
        else:
            weighted_score = base_weighted

        # 6. Calculate agreement and uncertainty
        agreement = self.calculate_model_agreement(scores)
        uncertainty, lower, upper = self.calculate_uncertainty(scores, regime_confidence)

        # 7. Calibrate final probability (light touch - amplifier already calibrated)
        final_probability = self.calibrate_probability(
            weighted_score,
            detected_regime,
            agreement
        )

        # 7. Determine signal strength
        confidence_level = 'high' if agreement > 0.7 else ('medium' if agreement > 0.5 else 'low')
        signal = self.determine_signal_strength(final_probability, regime_confidence, agreement)

        # 8. Build confidence object
        confidence = PredictionConfidence(
            probability=final_probability,
            confidence_level=confidence_level,
            uncertainty=uncertainty,
            lower_bound=lower,
            upper_bound=upper,
            model_agreement=agreement,
            regime_confidence=regime_confidence
        )

        # 9. Build layer scores
        layer_scores = LayerScores(
            physics_score=physics_composite,
            math_score=math_composite,
            ml_score=ml_composite,
            microstructure_score=micro_composite,
            alternative_score=alt_composite,
            physics_result=physics_result,
            math_result=math_result,
            ml_result=ml_result,
            microstructure_result=micro_result,
            alternative_result=alt_result
        )

        # 10. Generate recommendation
        recommendation = self.generate_recommendation(
            signal, final_probability, detected_regime, timeframe, layer_scores
        )

        # 11. Determine optimal strategy
        optimal_strategy, strategy_confidence = self.determine_optimal_strategy(
            detected_regime, physics_result, math_result
        )

        # 12. Collect factors
        bullish, bearish, warnings = self.collect_factors(
            physics_result, math_result, ml_result, micro_result, alt_result
        )

        # Add amplifier reasoning
        for reason in amplified.reasoning:
            if '+' in reason or 'boost' in reason.lower():
                bullish.insert(0, f'[AMPLIFIER] {reason}')
            elif '-' in reason:
                bearish.insert(0, f'[AMPLIFIER] {reason}')

        # Add pattern detection
        if amplified.pattern_detected != 'none':
            if 'BULL' in amplified.pattern_detected or 'UP' in amplified.pattern_detected or 'BOUNCE' in amplified.pattern_detected or 'OVERSOLD' in amplified.pattern_detected:
                bullish.insert(0, f'[PATTERN] {amplified.pattern_detected} detected')
            elif 'BEAR' in amplified.pattern_detected or 'DOWN' in amplified.pattern_detected or 'OVERBOUGHT' in amplified.pattern_detected:
                bearish.insert(0, f'[PATTERN] {amplified.pattern_detected} detected')

        # 13. Generate primary reason
        primary_reason = self.generate_primary_reason(signal, bullish, bearish, detected_regime)

        return AdvancedPrediction(
            symbol=symbol,
            timestamp=timestamp,
            final_probability=final_probability,
            signal=signal,
            confidence=confidence,
            recommendation=recommendation,
            detected_regime=detected_regime,
            regime_description=self.get_regime_description(detected_regime),
            layer_scores=layer_scores,
            layer_weights=weights,
            bullish_factors=bullish[:5],  # Top 5
            bearish_factors=bearish[:5],
            warnings=warnings[:5],
            primary_reason=primary_reason,
            optimal_strategy=optimal_strategy,
            strategy_confidence=strategy_confidence
        )

    def predict_quick(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = 'swing'
    ) -> Tuple[float, SignalStrength, str]:
        """
        Quick prediction returning just essentials.

        Returns (probability, signal, regime)
        """
        prediction = self.predict(symbol, df, timeframe)
        return (
            prediction.final_probability,
            prediction.signal,
            prediction.detected_regime
        )

    def batch_predict(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        timeframe: str = 'swing'
    ) -> Dict[str, AdvancedPrediction]:
        """
        Predict for multiple symbols.

        Args:
            symbols: List of stock symbols
            data: Dict mapping symbol to DataFrame
            timeframe: Trading timeframe

        Returns:
            Dict mapping symbol to prediction
        """
        predictions = {}

        for symbol in symbols:
            if symbol in data and len(data[symbol]) >= 20:
                try:
                    predictions[symbol] = self.predict(symbol, data[symbol], timeframe)
                except Exception as e:
                    # Log error but continue
                    print(f"Error predicting {symbol}: {e}")

        return predictions

    def rank_predictions(
        self,
        predictions: Dict[str, AdvancedPrediction],
        min_confidence: str = 'medium'
    ) -> List[Tuple[str, AdvancedPrediction]]:
        """
        Rank predictions by strength and confidence.

        Returns list of (symbol, prediction) sorted by attractiveness.
        """
        # Filter by confidence
        confidence_order = ['high', 'medium', 'low']
        min_idx = confidence_order.index(min_confidence)

        filtered = [
            (sym, pred) for sym, pred in predictions.items()
            if confidence_order.index(pred.confidence.confidence_level) <= min_idx
        ]

        # Sort by probability distance from 0.5 (stronger signals first)
        filtered.sort(key=lambda x: abs(x[1].final_probability - 0.5), reverse=True)

        return filtered

    def update_calibration(
        self,
        predicted_probability: float,
        actual_outcome: bool
    ):
        """
        Update calibration based on actual outcome.

        Args:
            predicted_probability: What we predicted
            actual_outcome: True if price went up
        """
        self._prediction_history.append((predicted_probability, actual_outcome))

        # Keep last 100 predictions
        if len(self._prediction_history) > 100:
            self._prediction_history = self._prediction_history[-100:]

        # Calculate calibration adjustment
        if len(self._prediction_history) >= 20:
            predicted = np.array([p for p, _ in self._prediction_history])
            actual = np.array([float(o) for _, o in self._prediction_history])

            # Average prediction error
            error = np.mean(actual) - np.mean(predicted)

            # Smooth adjustment
            self._calibration_adjustment = error * 0.5


# Backward-compatible alias - DEPRECATED, use DomainPredictionEngine instead
# This alias exists to prevent breaking existing code that imports AdvancedPredictionEngine
# from this module. New code should use DomainPredictionEngine explicitly.
AdvancedPredictionEngine = DomainPredictionEngine
