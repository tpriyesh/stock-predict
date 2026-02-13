"""
Base adapter class for model standardization.

All model adapters inherit from this class to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.core.interfaces.model_output import (
    StandardizedModelOutput,
    ModelUncertainty,
    UncertaintyType,
)


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    All adapters must implement:
    - predict_standardized(): Generate StandardizedModelOutput
    - model_name property: Return unique identifier

    Provides common utilities:
    - Error handling with fallback output
    - Logging
    - Data validation
    """

    def __init__(self):
        self._last_error: Optional[str] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return unique model identifier."""
        pass

    @abstractmethod
    def _predict_impl(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Internal prediction implementation.

        Subclasses implement this method with model-specific logic.
        """
        pass

    def predict_standardized(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate standardized prediction output with error handling.

        This method wraps _predict_impl() with:
        - Input validation
        - Error handling
        - Fallback output generation

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            **kwargs: Model-specific parameters

        Returns:
            StandardizedModelOutput (never raises, returns fallback on error)
        """
        try:
            # Validate inputs
            if df is None or df.empty:
                return self._fallback_output(
                    symbol,
                    reason="Empty dataframe provided"
                )

            if len(df) < 20:
                return self._fallback_output(
                    symbol,
                    reason=f"Insufficient data: {len(df)} rows (need 20+)"
                )

            # Call implementation
            output = self._predict_impl(symbol, df, **kwargs)
            self._last_error = None
            return output

        except Exception as e:
            self._last_error = str(e)
            logger.warning(f"[{self.model_name}] Prediction failed for {symbol}: {e}")
            return self._fallback_output(symbol, reason=str(e))

    def _fallback_output(self, symbol: str, reason: str) -> StandardizedModelOutput:
        """
        Generate neutral fallback output when prediction fails.

        Returns p_buy=0.5 (neutral) with low coverage to indicate
        the model couldn't make a confident prediction.
        """
        return StandardizedModelOutput(
            p_buy=0.5,
            uncertainty=ModelUncertainty(
                type=UncertaintyType.NONE,
                value=0.5,  # High uncertainty
            ),
            coverage=0.0,  # Zero coverage indicates failure
            reasoning=[f"[{self.model_name}] {reason}"],
            model_name=self.model_name,
            warnings=[f"Model fallback: {reason}"],
        )

    def validate_output(self, output: StandardizedModelOutput) -> bool:
        """Validate that output meets interface requirements."""
        try:
            if not 0 <= output.p_buy <= 1:
                logger.warning(f"[{self.model_name}] p_buy out of range: {output.p_buy}")
                return False
            if not 0 <= output.coverage <= 1:
                logger.warning(f"[{self.model_name}] coverage out of range: {output.coverage}")
                return False
            return True
        except Exception as e:
            logger.error(f"[{self.model_name}] Output validation failed: {e}")
            return False

    @staticmethod
    def normalize_to_probability(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to [0, 1] probability range.

        Args:
            value: Input value
            min_val: Expected minimum (maps to 0)
            max_val: Expected maximum (maps to 1)

        Returns:
            Normalized probability in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def signal_to_probability(signal: float, center: float = 0.0) -> float:
        """
        Convert a signal (-1 to +1 scale) to probability (0 to 1 scale).

        Args:
            signal: Signal value in [-1, +1]
            center: Center point (default 0)

        Returns:
            Probability in [0, 1]
        """
        return (signal - center + 1) / 2
