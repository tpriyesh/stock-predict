"""
Unified ML Engine

Combines all ML components:
- HMM Regime Detection
- Bayesian Predictor
- Ensemble Engine
- Calibration Pipeline

Provides a single interface for ML-based predictions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .hmm_regime_detector import HMMRegimeDetector, RegimeResult, MarketRegime
from .bayesian_predictor import BayesianPredictor, BayesianResult
from .ensemble_engine import EnsembleEngine, EnsembleResult
from .calibration_pipeline import CalibrationPipeline, CalibrationResult


@dataclass
class MLScore:
    """Complete output from ML engine."""
    symbol: str
    timestamp: datetime

    # Individual model scores (0-1)
    regime_score: float
    bayesian_score: float
    ensemble_score: float

    # Composite scores
    raw_composite: float            # Before calibration
    calibrated_composite: float     # After calibration

    # Regime information
    current_regime: str
    regime_stability: float

    # Uncertainty quantification
    prediction_uncertainty: float   # Standard deviation
    credible_interval: tuple        # 80% credible interval

    # Individual results
    regime_result: RegimeResult
    bayesian_result: BayesianResult
    ensemble_result: EnsembleResult
    calibration_result: CalibrationResult

    # Aggregated signals
    signals: List[str]
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]


class MLEngine:
    """
    Unified ML engine combining all components.

    Pipeline:
    1. Detect market regime (HMM)
    2. Run Bayesian predictor
    3. Run ensemble models
    4. Combine and calibrate
    """

    # Component weights
    COMPONENT_WEIGHTS = {
        'regime': 0.25,
        'bayesian': 0.35,
        'ensemble': 0.40
    }

    # Regime-specific weight adjustments
    REGIME_WEIGHT_ADJUSTMENTS = {
        'trending_bull': {'regime': 0.30, 'bayesian': 0.30, 'ensemble': 0.40},
        'trending_bear': {'regime': 0.30, 'bayesian': 0.30, 'ensemble': 0.40},
        'ranging': {'regime': 0.20, 'bayesian': 0.40, 'ensemble': 0.40},
        'choppy': {'regime': 0.35, 'bayesian': 0.30, 'ensemble': 0.35},
        'transition': {'regime': 0.40, 'bayesian': 0.30, 'ensemble': 0.30}
    }

    def __init__(self):
        self.regime_detector = HMMRegimeDetector()
        self.bayesian_predictor = BayesianPredictor()
        self.ensemble_engine = EnsembleEngine()
        self.calibration_pipeline = CalibrationPipeline()

    def get_weights(self, regime: str) -> Dict[str, float]:
        """Get component weights based on regime."""
        regime_lower = regime.lower().replace(' ', '_')
        return self.REGIME_WEIGHT_ADJUSTMENTS.get(
            regime_lower,
            self.COMPONENT_WEIGHTS
        )

    def combine_predictions(
        self,
        regime_score: float,
        bayesian_score: float,
        ensemble_score: float,
        weights: Dict[str, float]
    ) -> float:
        """Combine component predictions."""
        combined = (
            weights['regime'] * regime_score +
            weights['bayesian'] * bayesian_score +
            weights['ensemble'] * ensemble_score
        )
        return combined

    def calculate_uncertainty(
        self,
        bayesian: BayesianResult,
        ensemble: EnsembleResult
    ) -> tuple:
        """Calculate prediction uncertainty."""
        # Use Bayesian credible interval
        ci = bayesian.credible_interval_80

        # Combine with ensemble std
        combined_std = np.sqrt(
            bayesian.posterior_std**2 +
            ensemble.prediction_std**2
        ) / 2

        return combined_std, ci

    def collect_factors(
        self,
        regime: RegimeResult,
        bayesian: BayesianResult,
        ensemble: EnsembleResult
    ) -> tuple:
        """Collect bullish/bearish factors and warnings."""
        signals = []
        bullish = []
        bearish = []
        warnings = []

        # Regime signals
        if regime.current_regime == MarketRegime.TRENDING_BULL:
            signals.append('REGIME:BULL')
            bullish.append(f'[REGIME] {regime.reason}')
        elif regime.current_regime == MarketRegime.TRENDING_BEAR:
            signals.append('REGIME:BEAR')
            bearish.append(f'[REGIME] {regime.reason}')
        elif regime.current_regime == MarketRegime.CHOPPY:
            warnings.append(f'[REGIME] {regime.reason}')

        if regime.regime_stability < 0.5:
            warnings.append('[REGIME] Low regime stability')

        # Bayesian signals
        if 'BULLISH' in bayesian.signal:
            signals.append('BAYESIAN:BULLISH')
            bullish.append(f'[BAYESIAN] {bayesian.reason}')
        elif 'BEARISH' in bayesian.signal:
            signals.append('BAYESIAN:BEARISH')
            bearish.append(f'[BAYESIAN] {bayesian.reason}')

        if bayesian.most_influential_signal:
            bullish.append(f'[SIGNAL] {bayesian.most_influential_signal}') if 'bullish' in bayesian.most_influential_signal.lower() or 'oversold' in bayesian.most_influential_signal.lower() else None

        # Ensemble signals
        if 'BULLISH' in ensemble.signal:
            signals.append('ENSEMBLE:BULLISH')
            bullish.append(f'[ENSEMBLE] {ensemble.reason}')
        elif 'BEARISH' in ensemble.signal:
            signals.append('ENSEMBLE:BEARISH')
            bearish.append(f'[ENSEMBLE] {ensemble.reason}')

        if ensemble.model_agreement < 0.5:
            warnings.append('[ENSEMBLE] Models disagree')

        # Clean up None values
        bullish = [b for b in bullish if b]
        bearish = [b for b in bearish if b]

        return signals, bullish, bearish, warnings

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        sector: str = '',
        stock_history: Optional[Dict] = None
    ) -> MLScore:
        """
        Calculate comprehensive ML-based prediction.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            sector: Stock sector (for Bayesian priors)
            stock_history: Optional historical win/loss data

        Returns:
            MLScore with all ML predictions
        """
        timestamp = datetime.now()

        # Step 1: Detect regime
        regime_result = self.regime_detector.detect_regime(df)
        regime_str = regime_result.current_regime.value

        # Step 2: Bayesian prediction
        bayesian_result = self.bayesian_predictor.predict(
            df, symbol, sector, stock_history
        )

        # Step 3: Ensemble prediction
        ensemble_result = self.ensemble_engine.predict(df)

        # Get individual scores
        scores = {
            'regime': regime_result.probability_score,
            'bayesian': bayesian_result.probability_score,
            'ensemble': ensemble_result.probability_score
        }

        # Get regime-adjusted weights
        weights = self.get_weights(regime_str)

        # Combine predictions
        raw_composite = self.combine_predictions(
            scores['regime'],
            scores['bayesian'],
            scores['ensemble'],
            weights
        )

        # Step 4: Calibrate
        calibration_result = self.calibration_pipeline.calibrate(
            raw_composite,
            regime_str
        )
        calibrated_composite = calibration_result.final_calibrated

        # Calculate uncertainty
        uncertainty, ci = self.calculate_uncertainty(
            bayesian_result, ensemble_result
        )

        # Collect factors
        signals, bullish, bearish, warnings = self.collect_factors(
            regime_result, bayesian_result, ensemble_result
        )

        return MLScore(
            symbol=symbol,
            timestamp=timestamp,
            regime_score=scores['regime'],
            bayesian_score=scores['bayesian'],
            ensemble_score=scores['ensemble'],
            raw_composite=raw_composite,
            calibrated_composite=calibrated_composite,
            current_regime=regime_str,
            regime_stability=regime_result.regime_stability,
            prediction_uncertainty=uncertainty,
            credible_interval=ci,
            regime_result=regime_result,
            bayesian_result=bayesian_result,
            ensemble_result=ensemble_result,
            calibration_result=calibration_result,
            signals=signals,
            bullish_factors=bullish,
            bearish_factors=bearish,
            warnings=warnings
        )

    def score_quick(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> float:
        """Quick scoring returning only calibrated composite."""
        full_score = self.score(symbol, df)
        return full_score.calibrated_composite
