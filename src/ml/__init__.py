"""
Machine Learning Engine for Stock Prediction

This module implements ML and statistical inference:
- HMM Regime Detection: Hidden Markov Model for market states
- Bayesian Predictor: Hierarchical Bayesian inference
- Ensemble Engine: Gradient boosting with feature importance
- Calibration Pipeline: Multi-stage probability calibration
"""

from .hmm_regime_detector import HMMRegimeDetector, RegimeResult
from .bayesian_predictor import BayesianPredictor, BayesianResult
from .ensemble_engine import EnsembleEngine, EnsembleResult
from .calibration_pipeline import CalibrationPipeline, CalibrationResult
from .ml_engine import MLEngine, MLScore

__all__ = [
    'HMMRegimeDetector',
    'RegimeResult',
    'BayesianPredictor',
    'BayesianResult',
    'EnsembleEngine',
    'EnsembleResult',
    'CalibrationPipeline',
    'CalibrationResult',
    'MLEngine',
    'MLScore',
]
