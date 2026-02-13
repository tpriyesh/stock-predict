"""
Standardized model output interface for all prediction models.

All 7 models in the ensemble must produce output conforming to this interface
to ensure consistent aggregation and calibration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd


class UncertaintyType(Enum):
    """Type of uncertainty estimate provided by a model."""
    CONFIDENCE_INTERVAL = "confidence_interval"  # Lower/upper bounds
    STANDARD_DEVIATION = "standard_deviation"    # Single std value
    QUANTILES = "quantiles"                      # Multiple percentiles
    NONE = "none"                                # No uncertainty estimate


@dataclass
class ModelUncertainty:
    """
    Standardized uncertainty representation.

    All models should provide some form of uncertainty quantification.
    This helps the ensemble weight models appropriately and provides
    better calibration.
    """
    type: UncertaintyType
    value: float  # Primary uncertainty value (e.g., std dev or interval width)
    lower_bound: Optional[float] = None  # For confidence intervals
    upper_bound: Optional[float] = None  # For confidence intervals
    quantiles: Optional[Dict[int, float]] = None  # For quantile estimates {5: 0.45, 50: 0.65, 95: 0.80}

    @classmethod
    def from_confidence(cls, confidence: float) -> "ModelUncertainty":
        """
        Create uncertainty from a simple confidence score.

        Converts confidence (0-1) to uncertainty where:
        - High confidence (0.9) -> low uncertainty (0.1)
        - Low confidence (0.5) -> high uncertainty (0.5)
        """
        uncertainty_value = 1.0 - confidence
        return cls(
            type=UncertaintyType.STANDARD_DEVIATION,
            value=uncertainty_value,
            lower_bound=max(0, confidence - uncertainty_value / 2),
            upper_bound=min(1, confidence + uncertainty_value / 2)
        )

    @classmethod
    def from_interval(cls, lower: float, upper: float) -> "ModelUncertainty":
        """Create uncertainty from confidence interval bounds."""
        return cls(
            type=UncertaintyType.CONFIDENCE_INTERVAL,
            value=upper - lower,
            lower_bound=lower,
            upper_bound=upper
        )

    @classmethod
    def none(cls) -> "ModelUncertainty":
        """Create a placeholder for models without uncertainty estimates."""
        return cls(
            type=UncertaintyType.NONE,
            value=0.3,  # Default moderate uncertainty
            lower_bound=None,
            upper_bound=None
        )


@dataclass
class StandardizedModelOutput:
    """
    Unified output interface for all prediction models.

    Every model adapter must produce this output format to ensure
    consistent ensemble aggregation.

    Key fields:
    - p_buy: Probability of positive return (0-1 scale, REQUIRED)
    - uncertainty: Quantification of prediction uncertainty
    - coverage: How much of the required data was available (0-1)
    - reasoning: Human-readable explanations

    Example:
        output = StandardizedModelOutput(
            p_buy=0.72,
            uncertainty=ModelUncertainty.from_confidence(0.8),
            coverage=1.0,
            reasoning=["Strong momentum", "Above 200 SMA"],
            model_name="physics"
        )
    """
    # Core prediction (REQUIRED - all on 0-1 scale)
    p_buy: float              # Probability of positive return (0 = bearish, 1 = bullish)
    uncertainty: ModelUncertainty  # Uncertainty quantification
    coverage: float           # Data/signal coverage (0-1, 1 = full data available)

    # Reasoning (REQUIRED)
    reasoning: List[str]      # Human-readable reasons for the prediction

    # Metadata
    model_name: str = ""      # Identifier for this model
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional: Raw output for debugging
    raw_output: Optional[Dict[str, Any]] = None

    # Optional: Additional signals
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate output values are in expected ranges."""
        if not 0 <= self.p_buy <= 1:
            raise ValueError(f"p_buy must be in [0, 1], got {self.p_buy}")
        if not 0 <= self.coverage <= 1:
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")

    @property
    def signal(self) -> str:
        """Derive signal from probability."""
        if self.p_buy >= 0.65:
            return "BUY"
        elif self.p_buy <= 0.35:
            return "SELL"
        return "HOLD"

    @property
    def confidence(self) -> float:
        """
        Derive confidence from uncertainty.

        Returns a value in [0, 1] where higher is more confident.
        """
        if self.uncertainty.type == UncertaintyType.NONE:
            return 0.5
        return max(0, min(1, 1 - self.uncertainty.value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "p_buy": self.p_buy,
            "uncertainty_type": self.uncertainty.type.value,
            "uncertainty_value": self.uncertainty.value,
            "coverage": self.coverage,
            "reasoning": self.reasoning,
            "model_name": self.model_name,
            "signal": self.signal,
            "confidence": self.confidence,
            "warnings": self.warnings
        }


@runtime_checkable
class PredictionModel(Protocol):
    """
    Protocol that all prediction model adapters must implement.

    This ensures type safety and consistent interface across all models.

    Usage:
        class MyModelAdapter:
            def predict_standardized(self, symbol: str, df: pd.DataFrame, **kwargs) -> StandardizedModelOutput:
                # Implementation
                ...

            @property
            def model_name(self) -> str:
                return "my_model"
    """

    def predict_standardized(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs
    ) -> StandardizedModelOutput:
        """
        Generate standardized prediction output.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            df: DataFrame with OHLCV data (must have Open, High, Low, Close, Volume)
            **kwargs: Model-specific parameters (e.g., sector, trade_type)

        Returns:
            StandardizedModelOutput with p_buy, uncertainty, coverage, reasoning
        """
        ...

    @property
    def model_name(self) -> str:
        """Return unique model identifier."""
        ...


@dataclass
class EnsembleInput:
    """
    Collection of standardized outputs from all models for ensemble aggregation.

    This is passed to the ensemble aggregator which combines all model outputs
    into a final prediction.
    """
    outputs: List[StandardizedModelOutput]
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def model_names(self) -> List[str]:
        """Get list of model names in this ensemble."""
        return [o.model_name for o in self.outputs]

    @property
    def mean_p_buy(self) -> float:
        """Simple average of all p_buy values."""
        if not self.outputs:
            return 0.5
        return sum(o.p_buy for o in self.outputs) / len(self.outputs)

    @property
    def weighted_p_buy(self) -> float:
        """Coverage-weighted average of p_buy values."""
        if not self.outputs:
            return 0.5
        total_weight = sum(o.coverage for o in self.outputs)
        if total_weight == 0:
            return 0.5
        return sum(o.p_buy * o.coverage for o in self.outputs) / total_weight

    def get_agreement_count(self, threshold: float = 0.55) -> int:
        """Count models agreeing on BUY signal (p_buy > threshold)."""
        return sum(1 for o in self.outputs if o.p_buy > threshold)

    def get_all_reasoning(self) -> List[str]:
        """Collect all reasoning from all models with model tags."""
        all_reasons = []
        for output in self.outputs:
            tag = f"[{output.model_name.upper()}]"
            for reason in output.reasoning:
                all_reasons.append(f"{tag} {reason}")
        return all_reasons

    def get_all_warnings(self) -> List[str]:
        """Collect all warnings from all models."""
        all_warnings = []
        for output in self.outputs:
            tag = f"[{output.model_name.upper()}]"
            for warning in output.warnings:
                all_warnings.append(f"{tag} {warning}")
        return all_warnings
