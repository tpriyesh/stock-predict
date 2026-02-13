"""
Advanced Mathematical Models for Stock Prediction

This package contains sophisticated mathematical algorithms:

CORE MODULES (Original):
- PCA & Eigenvalue Decomposition for dimensionality reduction
- Wavelet Multi-Resolution Analysis for pattern detection
- Kalman Filter for optimal state estimation
- MLE-Calibrated Markov Regime Models
- Deep Q-Network for position sizing
- Covariance Factor Models for risk decomposition
- Advanced Statistical Validation
- Comprehensive Multi-Dimensional Measurement Metrics

ADVANCED MODULES (New):
- Quantum-Inspired Optimization (quantum annealing, QAOA)
- Bellman Optimal Stopping (dynamic programming for entry/exit)
- Stochastic Differential Equations (GBM, OU, Jump-Diffusion, Heston)
- Information Geometry (Fisher information, KL divergence, mutual information)
- Tensor Decomposition (CP, Tucker for multi-scale patterns)
- Spectral Graph Theory (Laplacian eigenmaps, random matrix theory)
- Optimal Transport (Wasserstein distance, regime shift detection)
- Taylor Series Expansions (Edgeworth, Cornish-Fisher for risk)
- Heavy-Tailed Distributions (Stable, Generalized Hyperbolic, GPD)
- Copula Models (Gaussian, Student-t, Clayton, Gumbel for tail dependence)
- Multi-Dimensional Confidence Scoring

All modules integrate with the centralized prediction engine.
"""

# Original modules
from .pca_decomposition import PCADecomposer, EigenvalueAnalyzer
from .wavelet_analysis import WaveletAnalyzer, MultiResolutionDecomposer
from .kalman_filter import KalmanTrendFilter, AdaptiveKalmanFilter
from .markov_regime_mle import MLEMarkovRegime, RegimeStatistics
from .dqn_position_sizing import DQNPositionAgent, ReplayBuffer
from .factor_model import FactorModel, CovarianceDecomposer
from .statistical_validation import AdvancedStatisticalValidator
from .measurement_metrics import (
    MeasurementMetricsEngine,
    MultiDimensionalScore,
    AccuracyMetrics,
    CalibrationMetrics,
    RiskAdjustedMetrics,
    InformationMetrics,
    StabilityMetrics,
    compute_prediction_quality,
    quick_quality_check
)

# New Advanced Modules
from .quantum_inspired import (
    QuantumInspiredOptimizer,
    QuantumPortfolioOptimizer,
    QuantumPrediction
)
from .bellman_optimal import (
    BellmanOptimalStopping,
    HamiltonJacobiBellman,
    ReinforcementLearningTrader,
    OptimalPolicy,
    TradingAction
)
from .stochastic_de import (
    StochasticDEEngine,
    JumpDetector,
    MeanReversionAnalyzer,
    SDEParameters,
    SDEPrediction
)
from .information_geometry import (
    InformationGeometryEngine,
    FisherInformationAnalyzer,
    KLDivergenceAnalyzer,
    MutualInformationAnalyzer,
    RenyiEntropyAnalyzer,
    InformationGeometryResult
)
from .tensor_decomposition import (
    CPDecomposition,
    TuckerDecomposition,
    MultiScalePatternExtractor,
    CrossAssetTensorAnalysis,
    TensorDecompositionResult
)
from .spectral_graph import (
    SpectralGraphEngine,
    LaplacianEigenmaps,
    RandomMatrixTheory,
    SpectralClustering,
    NetworkCentrality,
    SpectralAnalysisResult
)
from .optimal_transport import (
    OptimalTransportEngine,
    WassersteinDistance,
    SinkhornDistance,
    SlicedWassersteinDistance,
    DistributionDriftDetector,
    OptimalTransportResult
)
from .taylor_series import (
    ExpansionEngine,
    TaylorSeriesAnalyzer,
    EdgeworthExpansion,
    CornishFisherExpansion,
    MomentGeneratingFunction,
    ExpansionResult
)
from .heavy_tailed import (
    HeavyTailedEngine,
    StableDistribution,
    GeneralizedHyperbolic,
    GeneralizedParetoDistribution,
    HeavyTailedFit
)
from .copula_models import (
    CopulaEngine,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    CopulaResult
)
from .multidim_confidence import (
    MultiDimensionalScorer,
    EigenvalueConfidenceAnalyzer,
    MahalanobisAnomalyDetector,
    UncertaintyQuantifier,
    ConfidenceCalibrator,
    MultiDimConfidence
)
from .unified_advanced_engine import (
    UnifiedAdvancedEngine,
    UnifiedPrediction,
    predict_with_advanced_math
)

__all__ = [
    # === ORIGINAL MODULES ===
    # PCA
    'PCADecomposer',
    'EigenvalueAnalyzer',
    # Wavelet
    'WaveletAnalyzer',
    'MultiResolutionDecomposer',
    # Kalman
    'KalmanTrendFilter',
    'AdaptiveKalmanFilter',
    # Markov
    'MLEMarkovRegime',
    'RegimeStatistics',
    # DQN
    'DQNPositionAgent',
    'ReplayBuffer',
    # Factor Model
    'FactorModel',
    'CovarianceDecomposer',
    # Validation
    'AdvancedStatisticalValidator',
    # Measurement Metrics
    'MeasurementMetricsEngine',
    'MultiDimensionalScore',
    'AccuracyMetrics',
    'CalibrationMetrics',
    'RiskAdjustedMetrics',
    'InformationMetrics',
    'StabilityMetrics',
    'compute_prediction_quality',
    'quick_quality_check',

    # === NEW ADVANCED MODULES ===
    # Quantum-Inspired
    'QuantumInspiredOptimizer',
    'QuantumPortfolioOptimizer',
    'QuantumPrediction',
    # Bellman Optimal Stopping
    'BellmanOptimalStopping',
    'HamiltonJacobiBellman',
    'ReinforcementLearningTrader',
    'OptimalPolicy',
    'TradingAction',
    # Stochastic DEs
    'StochasticDEEngine',
    'JumpDetector',
    'MeanReversionAnalyzer',
    'SDEParameters',
    'SDEPrediction',
    # Information Geometry
    'InformationGeometryEngine',
    'FisherInformationAnalyzer',
    'KLDivergenceAnalyzer',
    'MutualInformationAnalyzer',
    'RenyiEntropyAnalyzer',
    'InformationGeometryResult',
    # Tensor Decomposition
    'CPDecomposition',
    'TuckerDecomposition',
    'MultiScalePatternExtractor',
    'CrossAssetTensorAnalysis',
    'TensorDecompositionResult',
    # Spectral Graph Theory
    'SpectralGraphEngine',
    'LaplacianEigenmaps',
    'RandomMatrixTheory',
    'SpectralClustering',
    'NetworkCentrality',
    'SpectralAnalysisResult',
    # Optimal Transport
    'OptimalTransportEngine',
    'WassersteinDistance',
    'SinkhornDistance',
    'SlicedWassersteinDistance',
    'DistributionDriftDetector',
    'OptimalTransportResult',
    # Taylor Series & Expansions
    'ExpansionEngine',
    'TaylorSeriesAnalyzer',
    'EdgeworthExpansion',
    'CornishFisherExpansion',
    'MomentGeneratingFunction',
    'ExpansionResult',
    # Heavy-Tailed Distributions
    'HeavyTailedEngine',
    'StableDistribution',
    'GeneralizedHyperbolic',
    'GeneralizedParetoDistribution',
    'HeavyTailedFit',
    # Copula Models
    'CopulaEngine',
    'GaussianCopula',
    'StudentTCopula',
    'ClaytonCopula',
    'GumbelCopula',
    'CopulaResult',
    # Multi-Dimensional Confidence
    'MultiDimensionalScorer',
    'EigenvalueConfidenceAnalyzer',
    'MahalanobisAnomalyDetector',
    'UncertaintyQuantifier',
    'ConfidenceCalibrator',
    'MultiDimConfidence',
    # Unified Engine
    'UnifiedAdvancedEngine',
    'UnifiedPrediction',
    'predict_with_advanced_math',
]
