"""
API Module - Unified Stock Prediction Interface

Provides:
- StockPredictor: Single entry point for all prediction use cases
- UseCase: INTRADAY, SWING, LONG_TERM, MULTI_BAGGER, etc.
- UnifiedPrediction: Standardized output format
"""

from .unified_api import (
    StockPredictor,
    UseCase,
    UnifiedPrediction,
)

__all__ = [
    'StockPredictor',
    'UseCase',
    'UnifiedPrediction',
]
