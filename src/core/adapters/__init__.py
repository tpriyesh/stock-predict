"""
Model adapters for standardized output.

Each adapter wraps a prediction model and produces StandardizedModelOutput.
"""

from .base_adapter import BaseModelAdapter
from .technical_adapter import TechnicalModelAdapter
from .physics_adapter import PhysicsModelAdapter
from .math_adapter import MathModelAdapter
from .regime_adapter import RegimeModelAdapter
from .macro_adapter import MacroModelAdapter
from .alternative_adapter import AlternativeDataAdapter
from .advanced_adapter import AdvancedMathAdapter

__all__ = [
    "BaseModelAdapter",
    "TechnicalModelAdapter",
    "PhysicsModelAdapter",
    "MathModelAdapter",
    "RegimeModelAdapter",
    "MacroModelAdapter",
    "AlternativeDataAdapter",
    "AdvancedMathAdapter",
]
