"""
Advanced Mathematics Models for Stock Prediction

This module implements mathematical concepts for pattern detection:
- Fourier Analysis: Detect hidden price cycles
- Fractal Dimension: Measure market complexity (Hurst exponent)
- Information Entropy: Quantify predictability
- Statistical Mechanics: Market temperature and phase transitions
"""

from .fourier_cycles import FourierCycleModel, FourierResult
from .fractal_dimension import FractalDimensionModel, FractalResult
from .information_entropy import InformationEntropyModel, EntropyResult
from .statistical_mechanics import StatisticalMechanicsModel, StatMechResult
from .math_engine import MathEngine, MathScore

__all__ = [
    'FourierCycleModel',
    'FourierResult',
    'FractalDimensionModel',
    'FractalResult',
    'InformationEntropyModel',
    'EntropyResult',
    'StatisticalMechanicsModel',
    'StatMechResult',
    'MathEngine',
    'MathScore',
]
