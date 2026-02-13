"""
Unified Advanced Mathematical Prediction Engine

Integrates all advanced mathematical modules into a cohesive prediction system:
- Quantum-Inspired Optimization
- Bellman Optimal Stopping
- Stochastic Differential Equations
- Information Geometry
- Tensor Decomposition
- Spectral Graph Theory
- Optimal Transport
- Taylor Series Expansions
- Heavy-Tailed Distributions
- Copula Models
- Multi-Dimensional Confidence

This engine provides:
1. Multi-dimensional signal generation
2. Robust confidence quantification
3. Regime-aware predictions
4. Distribution-corrected risk metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

# Import all modules
from .quantum_inspired import QuantumInspiredOptimizer, QuantumPrediction
from .bellman_optimal import BellmanOptimalStopping, OptimalPolicy, TradingAction
from .stochastic_de import StochasticDEEngine, SDEPrediction, MeanReversionAnalyzer
from .information_geometry import InformationGeometryEngine, InformationGeometryResult
from .tensor_decomposition import MultiScalePatternExtractor, TensorDecompositionResult
from .spectral_graph import SpectralGraphEngine, SpectralAnalysisResult
from .optimal_transport import OptimalTransportEngine, OptimalTransportResult
from .taylor_series import ExpansionEngine, ExpansionResult
from .heavy_tailed import HeavyTailedEngine, HeavyTailedFit
from .copula_models import CopulaEngine, CopulaResult
from .multidim_confidence import MultiDimensionalScorer, MultiDimConfidence

logger = logging.getLogger(__name__)


@dataclass
class UnifiedPrediction:
    """Comprehensive prediction from unified engine"""
    # Core prediction
    signal: float  # -1 (strong sell) to +1 (strong buy)
    confidence: float  # 0 to 1
    action: TradingAction

    # Multi-dimensional components
    quantum_signal: float
    sde_signal: float
    information_signal: float
    tensor_signal: float
    spectral_signal: float
    transport_signal: float

    # Confidence breakdown
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    # Regime analysis
    regime: str = "unknown"
    regime_stability: float = 0.5
    regime_change_probability: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    expected_return: float = 0.0
    expected_volatility: float = 0.0

    # Optimal execution
    optimal_horizon: int = 1
    optimal_position: float = 0.0

    # Distribution characteristics
    tail_index: float = 2.0
    is_heavy_tailed: bool = False
    skewness: float = 0.0

    # Detailed results (optional)
    detailed_results: Dict[str, Any] = field(default_factory=dict)


class UnifiedAdvancedEngine:
    """
    Master engine integrating all advanced mathematical prediction modules.

    Architecture:
    1. Each module generates independent signals
    2. Signals are combined with adaptive weights
    3. Multi-dimensional confidence is computed
    4. Distribution corrections are applied
    5. Optimal execution parameters are determined
    """

    def __init__(
        self,
        module_weights: Optional[Dict[str, float]] = None
    ):
        # Initialize all modules
        self.quantum = QuantumInspiredOptimizer()
        self.bellman = BellmanOptimalStopping()
        self.sde = StochasticDEEngine()
        self.info_geometry = InformationGeometryEngine()
        self.tensor = MultiScalePatternExtractor()
        self.spectral = SpectralGraphEngine()
        self.transport = OptimalTransportEngine()
        self.taylor = ExpansionEngine()
        self.heavy_tail = HeavyTailedEngine()
        self.copula = CopulaEngine()
        self.confidence_scorer = MultiDimensionalScorer()
        self.mean_reversion = MeanReversionAnalyzer()

        # Default module weights
        self.weights = module_weights or {
            "quantum": 0.15,
            "sde": 0.20,
            "info_geometry": 0.10,
            "tensor": 0.10,
            "spectral": 0.10,
            "transport": 0.15,
            "mean_reversion": 0.20
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

    def predict(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        entry_price: Optional[float] = None,
        market_returns: Optional[np.ndarray] = None
    ) -> UnifiedPrediction:
        """
        Generate comprehensive prediction using all modules.

        Args:
            prices: Historical price series
            returns: Pre-computed returns (optional)
            features: Additional features (optional)
            entry_price: Entry price for position (for optimal stopping)
            market_returns: Market index returns (for copula analysis)

        Returns:
            UnifiedPrediction with complete analysis
        """
        if len(prices) < 60:
            return self._default_prediction()

        # Compute returns if not provided
        if returns is None:
            returns = np.diff(np.log(prices))

        current_price = prices[-1]
        if entry_price is None:
            entry_price = current_price

        # 1. Quantum-Inspired Analysis
        quantum_result = self.quantum.analyze(returns)
        quantum_signal = quantum_result.signal_strength

        # 2. SDE Analysis
        sde_models = self.sde.estimate_all_models(prices, returns)
        best_model_name, best_model = self.sde.select_best_model(sde_models)
        sde_pred = self.sde.predict(prices, best_model, horizon=5)
        sde_signal = sde_pred.probability_up * 2 - 1  # Map to [-1, 1]

        # 3. Mean Reversion Analysis
        mr_result = self.mean_reversion.analyze(prices, returns)
        mr_signal = 0.0
        if mr_result["is_mean_reverting"]:
            # Signal based on z-score
            z_score = mr_result["z_score"]
            mr_signal = -np.tanh(z_score / 2)  # Contrarian signal

        # 4. Information Geometry
        info_result = self.info_geometry.analyze(returns, features)
        # Lower KL divergence = more stable regime = higher confidence
        regime_stability = 1 - np.tanh(info_result.kl_divergence)
        info_signal = info_result.predictability_score * np.sign(np.mean(returns[-10:]))

        # 5. Tensor Decomposition
        tensor_result = self.tensor.extract(returns, features)
        # Extract signal from dominant component
        if tensor_result.components:
            dominant = tensor_result.components[0]
            tensor_signal = 0.5 if dominant["temporal_trend"] == "rising" else -0.5
        else:
            tensor_signal = 0.0

        # 6. Optimal Transport (Regime Shift Detection)
        transport_result = self.transport.analyze(returns, features)
        regime_shift = transport_result.get("regime_shift_score", 0)
        # High regime shift = reduce confidence
        transport_confidence = 1 - regime_shift

        # Transport signal based on drift direction
        drift_dir = transport_result.get("drift_direction", "stable")
        if drift_dir == "right":
            transport_signal = 0.3
        elif drift_dir == "left":
            transport_signal = -0.3
        else:
            transport_signal = 0.0

        # 7. Taylor Series (Momentum Analysis)
        taylor_result = self.taylor.analyze(prices, returns)
        velocity = taylor_result.moments.get("mean", 0)
        taylor_signal = np.tanh(velocity * 100)  # Scale and bound

        # 8. Heavy-Tailed Distribution Analysis
        tail_result = self.heavy_tail.analyze(returns)
        tail_index = tail_result.tail_index
        is_heavy_tailed = tail_result.is_infinite_variance or tail_index < 2

        # 9. Bellman Optimal Stopping (if position exists)
        bellman_result = self.bellman.compute_optimal_policy(
            returns, entry_price, current_price
        )

        # 10. Spectral Analysis (if multiple assets available)
        if market_returns is not None and len(market_returns) == len(returns):
            combined = np.column_stack([returns, market_returns])
            spectral_result = self.spectral.analyze(combined)
            market_mode = spectral_result.market_mode_strength
            spectral_signal = (market_mode - 0.5) * np.sign(np.mean(market_returns[-10:]))
        else:
            spectral_result = None
            spectral_signal = 0.0
            market_mode = 0.5

        # 11. Copula Analysis (if market returns available)
        if market_returns is not None:
            copula_result = self.copula.analyze(returns[-100:], market_returns[-100:])
            tail_dependence = max(
                copula_result.lower_tail_dependence,
                copula_result.upper_tail_dependence
            )
        else:
            copula_result = None
            tail_dependence = 0.0

        # === COMBINE SIGNALS ===

        # Weighted combination
        combined_signal = (
            self.weights["quantum"] * quantum_signal +
            self.weights["sde"] * sde_signal +
            self.weights["info_geometry"] * info_signal +
            self.weights["tensor"] * tensor_signal +
            self.weights["spectral"] * spectral_signal +
            self.weights["transport"] * transport_signal +
            self.weights["mean_reversion"] * mr_signal
        )

        # Add Taylor momentum influence
        combined_signal = 0.9 * combined_signal + 0.1 * taylor_signal

        # Bound signal
        combined_signal = float(np.clip(combined_signal, -1, 1))

        # === COMPUTE CONFIDENCE ===

        # Build predictions dict for multi-dim scorer
        predictions = {
            "quantum": quantum_signal,
            "sde": sde_signal,
            "info_geometry": info_signal,
            "tensor": tensor_signal,
            "spectral": spectral_signal,
            "transport": transport_signal,
            "mean_reversion": mr_signal
        }

        # Create feature matrix from recent returns
        if features is None:
            feature_matrix = returns[-20:].reshape(-1, 1)
        else:
            feature_matrix = features[-20:] if len(features) >= 20 else features

        multidim_conf = self.confidence_scorer.score(
            predictions, feature_matrix, returns
        )

        # Adjust confidence based on regime stability and tail risk
        base_confidence = multidim_conf.overall_confidence
        adjusted_confidence = (
            base_confidence *
            regime_stability *
            transport_confidence *
            (1 - 0.3 * is_heavy_tailed)  # Penalize heavy tails
        )

        confidence = float(np.clip(adjusted_confidence, 0.1, 0.95))

        # === DETERMINE ACTION ===

        if bellman_result.action != TradingAction.HOLD:
            action = bellman_result.action
        else:
            if combined_signal > 0.6 and confidence > 0.6:
                action = TradingAction.BUY_FULL
            elif combined_signal > 0.3 and confidence > 0.5:
                action = TradingAction.BUY_SMALL
            elif combined_signal < -0.6 and confidence > 0.6:
                action = TradingAction.SELL_ALL
            elif combined_signal < -0.3 and confidence > 0.5:
                action = TradingAction.SELL_PARTIAL
            else:
                action = TradingAction.HOLD

        # === RISK METRICS ===

        # From Cornish-Fisher expansion
        skewness = taylor_result.moments.get("skewness", 0)
        excess_kurt = taylor_result.moments.get("excess_kurtosis", 0)

        var_95 = tail_result.risk_metrics.get("var_95", np.percentile(returns, 5))
        cvar_95 = tail_result.risk_metrics.get("cvar_95", var_95)

        expected_return = sde_pred.expected_return
        expected_vol = sde_pred.expected_volatility

        # === OPTIMAL EXECUTION ===

        optimal_horizon = bellman_result.optimal_horizon
        optimal_position = quantum_result.optimal_position

        # === COMPILE RESULT ===

        confidence_breakdown = {
            "signal_agreement": float(multidim_conf.directional_confidence),
            "regime_stability": float(regime_stability),
            "transport_stability": float(transport_confidence),
            "eigenvalue_confidence": float(multidim_conf.stability_confidence),
            "anomaly_score": float(multidim_conf.anomaly_score)
        }

        # Determine regime
        if sde_pred.regime_indicator:
            regime = sde_pred.regime_indicator
        elif mr_result.get("is_mean_reverting", False):
            regime = "mean_reverting"
        else:
            regime = "trending" if abs(combined_signal) > 0.3 else "sideways"

        return UnifiedPrediction(
            signal=combined_signal,
            confidence=confidence,
            action=action,
            quantum_signal=float(quantum_signal),
            sde_signal=float(sde_signal),
            information_signal=float(info_signal),
            tensor_signal=float(tensor_signal),
            spectral_signal=float(spectral_signal),
            transport_signal=float(transport_signal),
            confidence_breakdown=confidence_breakdown,
            regime=regime,
            regime_stability=float(regime_stability),
            regime_change_probability=float(quantum_result.tunneling_probability),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            expected_return=float(expected_return),
            expected_volatility=float(expected_vol),
            optimal_horizon=optimal_horizon,
            optimal_position=float(optimal_position),
            tail_index=float(tail_index),
            is_heavy_tailed=is_heavy_tailed,
            skewness=float(skewness),
            detailed_results={
                "quantum": {
                    "coherence_time": quantum_result.coherence_time,
                    "superposition": quantum_result.superposition_amplitudes,
                    "energy_landscape": quantum_result.energy_landscape
                },
                "sde": {
                    "model": best_model_name,
                    "drift": best_model.drift,
                    "volatility": best_model.volatility,
                    "jump_intensity": best_model.jump_intensity
                },
                "mean_reversion": {
                    "half_life": mr_result.get("half_life_days", float('inf')),
                    "z_score": mr_result.get("z_score", 0),
                    "equilibrium": mr_result.get("equilibrium_level", current_price)
                },
                "information_geometry": {
                    "kl_divergence": info_result.kl_divergence,
                    "hellinger": info_result.hellinger_distance,
                    "mutual_info": info_result.mutual_information
                },
                "transport": {
                    "wasserstein_1": transport_result.get("wasserstein_1", 0),
                    "regime_shift_score": regime_shift
                },
                "distribution": {
                    "type": tail_result.distribution,
                    "tail_index": tail_index,
                    "parameters": tail_result.parameters
                },
                "bellman": {
                    "continuation_value": bellman_result.continuation_value,
                    "stopping_value": bellman_result.stopping_value,
                    "optimal_horizon": optimal_horizon
                }
            }
        )

    def _default_prediction(self) -> UnifiedPrediction:
        """Return default prediction for insufficient data."""
        return UnifiedPrediction(
            signal=0.0,
            confidence=0.0,
            action=TradingAction.HOLD,
            quantum_signal=0.0,
            sde_signal=0.0,
            information_signal=0.0,
            tensor_signal=0.0,
            spectral_signal=0.0,
            transport_signal=0.0,
            confidence_breakdown={},
            regime="unknown",
            regime_stability=0.0,
            regime_change_probability=0.5,
            var_95=0.0,
            cvar_95=0.0,
            expected_return=0.0,
            expected_volatility=0.0,
            optimal_horizon=1,
            optimal_position=0.0,
            tail_index=2.0,
            is_heavy_tailed=False,
            skewness=0.0,
            detailed_results={}
        )

    def get_full_analysis(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analysis without prediction.

        Useful for exploration and understanding current market state.
        """
        if len(prices) < 60:
            return {"error": "Insufficient data"}

        if returns is None:
            returns = np.diff(np.log(prices))

        analysis = {}

        # SDE analysis
        sde_models = self.sde.estimate_all_models(prices, returns)
        analysis["sde_models"] = {
            name: {
                "drift": params.drift,
                "volatility": params.volatility,
                "aic": params.aic,
                "bic": params.bic
            }
            for name, params in sde_models.items()
        }

        # Mean reversion
        analysis["mean_reversion"] = self.mean_reversion.analyze(prices, returns)

        # Information geometry
        info_result = self.info_geometry.analyze(returns)
        analysis["information_geometry"] = {
            "kl_divergence": info_result.kl_divergence,
            "hellinger_distance": info_result.hellinger_distance,
            "entropy": info_result.entropy,
            "predictability": info_result.predictability_score
        }

        # Quantum
        quantum_result = self.quantum.analyze(returns)
        analysis["quantum"] = {
            "signal_strength": quantum_result.signal_strength,
            "tunneling_probability": quantum_result.tunneling_probability,
            "coherence_time": quantum_result.coherence_time,
            "superposition": quantum_result.superposition_amplitudes
        }

        # Transport
        transport_result = self.transport.analyze(returns)
        analysis["optimal_transport"] = transport_result

        # Taylor/Moments
        taylor_result = self.taylor.analyze(prices, returns)
        analysis["moments"] = taylor_result.moments
        analysis["risk_metrics"] = taylor_result.risk_metrics

        # Heavy tails
        tail_result = self.heavy_tail.analyze(returns)
        analysis["distribution"] = {
            "type": tail_result.distribution,
            "parameters": tail_result.parameters,
            "tail_index": tail_result.tail_index,
            "is_heavy_tailed": tail_result.is_infinite_variance
        }

        return analysis


# Convenience function for quick prediction
def predict_with_advanced_math(
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> UnifiedPrediction:
    """
    Quick prediction using unified advanced mathematical engine.

    Args:
        prices: Historical price series (minimum 60 points)
        returns: Pre-computed returns (optional)

    Returns:
        UnifiedPrediction with signal, confidence, and detailed analysis
    """
    engine = UnifiedAdvancedEngine()
    return engine.predict(prices, returns)
