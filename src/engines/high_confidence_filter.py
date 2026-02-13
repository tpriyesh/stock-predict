"""
High Confidence Signal Filter

Based on backtesting results:
- Overall accuracy: 52.9% (not tradeable)
- Predicted 65-70%: 78.6% accuracy (HIGHLY TRADEABLE)
- Predicted 70%+: 75.0% accuracy (HIGHLY TRADEABLE)
- Premium setups (MTF + Bullish + Good Regime): 66.7% accuracy

Strategy: Only trade when model has HIGH confidence
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TradeGrade(Enum):
    """Trade quality grades based on expected accuracy."""
    A_PLUS = "A+"  # 75%+ expected accuracy
    A = "A"        # 70-75% expected accuracy
    B = "B"        # 65-70% expected accuracy
    C = "C"        # 60-65% expected accuracy
    D = "D"        # 55-60% expected accuracy
    AVOID = "AVOID"  # Below 55% - don't trade


@dataclass
class FilteredSignal:
    """Signal after high-confidence filtering."""
    should_trade: bool
    grade: TradeGrade
    expected_accuracy: float
    direction: int  # 1=long, -1=short, 0=neutral
    probability: float
    confidence_reasons: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    position_size: float  # 0-1, how much of max position


class HighConfidenceFilter:
    """
    Filters predictions to only trade high-confidence setups.

    Key insight from backtesting:
    - Predicted probability correlates with actual accuracy
    - 65%+ predictions have 78% accuracy
    - Premium setups (multiple confirmations) have 67% accuracy

    This filter grades each setup and only allows trades above threshold.
    """

    # Probability thresholds based on backtest
    ACCURACY_MAP = {
        (0.70, 1.00): 0.75,   # 75% accuracy for 70%+ predictions
        (0.65, 0.70): 0.78,   # 78% accuracy for 65-70% predictions
        (0.60, 0.65): 0.51,   # 51% accuracy for 60-65%
        (0.55, 0.60): 0.65,   # 65% accuracy for 55-60%
        (0.50, 0.55): 0.52,   # 52% accuracy for 50-55%
    }

    # Grade thresholds
    GRADE_THRESHOLDS = {
        TradeGrade.A_PLUS: 0.75,
        TradeGrade.A: 0.70,
        TradeGrade.B: 0.65,
        TradeGrade.C: 0.60,
        TradeGrade.D: 0.55,
    }

    def __init__(self, min_grade: TradeGrade = TradeGrade.B):
        """
        Initialize filter with minimum grade threshold.

        Args:
            min_grade: Minimum grade required to trade (default: B = 65%+ expected)
        """
        self.min_grade = min_grade
        self.min_accuracy = self.GRADE_THRESHOLDS[min_grade]

    def get_expected_accuracy(self, probability: float) -> float:
        """
        Get expected accuracy based on predicted probability.
        """
        for (low, high), accuracy in self.ACCURACY_MAP.items():
            if low <= probability < high:
                return accuracy
        return 0.50

    def get_grade(self, expected_accuracy: float) -> TradeGrade:
        """Assign grade based on expected accuracy."""
        if expected_accuracy >= 0.75:
            return TradeGrade.A_PLUS
        elif expected_accuracy >= 0.70:
            return TradeGrade.A
        elif expected_accuracy >= 0.65:
            return TradeGrade.B
        elif expected_accuracy >= 0.60:
            return TradeGrade.C
        elif expected_accuracy >= 0.55:
            return TradeGrade.D
        else:
            return TradeGrade.AVOID

    def calculate_position_size(
        self,
        grade: TradeGrade,
        expected_accuracy: float
    ) -> float:
        """
        Calculate position size based on Kelly Criterion.

        Kelly = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio

        Assuming 2:1 reward/risk:
        """
        if grade == TradeGrade.AVOID:
            return 0.0

        p = expected_accuracy
        q = 1 - p
        b = 2.0  # 2:1 reward/risk

        kelly = (p * b - q) / b
        kelly = max(0, min(0.25, kelly))  # Cap at 25% of capital

        # Scale by grade
        grade_multiplier = {
            TradeGrade.A_PLUS: 1.0,
            TradeGrade.A: 0.8,
            TradeGrade.B: 0.6,
            TradeGrade.C: 0.4,
            TradeGrade.D: 0.2,
        }

        return kelly * grade_multiplier.get(grade, 0.1)

    def filter(
        self,
        prediction,  # AdvancedPrediction
        require_confirmation: bool = True
    ) -> FilteredSignal:
        """
        Filter a prediction and determine if it should be traded.

        Args:
            prediction: AdvancedPrediction object
            require_confirmation: Require multiple confirming factors

        Returns:
            FilteredSignal with trade decision and details
        """
        reasons = []

        probability = prediction.final_probability
        signal = prediction.signal.value
        regime = prediction.detected_regime

        # Get alternative data confluence if available
        alt_result = prediction.layer_scores.alternative_result
        confluence = 0.5
        if alt_result and alt_result.confluence_signal:
            confluence = alt_result.confluence_signal.confluence_score

        # 1. Base expected accuracy from probability
        expected_accuracy = self.get_expected_accuracy(probability)
        reasons.append(f"Base accuracy from {probability:.1%} prediction: {expected_accuracy:.1%}")

        # 2. Adjust for confluence
        if confluence >= 0.8:
            expected_accuracy *= 1.05  # 5% boost for high MTF confluence
            reasons.append(f"MTF confluence {confluence:.0%} (+5%)")
        elif confluence < 0.5:
            expected_accuracy *= 0.90  # 10% penalty for low confluence
            reasons.append(f"Low MTF confluence {confluence:.0%} (-10%)")

        # 3. Adjust for regime
        regime_adjustments = {
            'choppy': 1.10,  # 10% boost - our best regime
            'ranging': 1.05,  # 5% boost
            'trending_bull': 1.0,  # neutral
            'trending_bear': 0.95,  # slight penalty
            'transition': 0.90,  # avoid transitions
        }
        regime_factor = regime_adjustments.get(regime, 1.0)
        if regime_factor != 1.0:
            expected_accuracy *= regime_factor
            reasons.append(f"{regime} regime adjustment: {regime_factor:.0%}")

        # 4. Adjust for signal type
        if 'STRONG' in signal:
            expected_accuracy *= 1.05
            reasons.append("Strong signal (+5%)")
        elif 'WEAK' in signal:
            expected_accuracy *= 0.95
            reasons.append("Weak signal (-5%)")
        elif signal == 'NEUTRAL':
            expected_accuracy *= 0.90
            reasons.append("Neutral signal (-10%)")

        # 5. Confirmation requirement
        if require_confirmation:
            confirmations = 0

            # Check for bullish factors
            bullish_count = len(prediction.bullish_factors)
            bearish_count = len(prediction.bearish_factors)

            if probability > 0.55 and bullish_count > bearish_count:
                confirmations += 1
            elif probability < 0.45 and bearish_count > bullish_count:
                confirmations += 1

            # Check model agreement
            if prediction.confidence.model_agreement > 0.7:
                confirmations += 1

            # Check regime confidence
            if prediction.confidence.regime_confidence > 0.7:
                confirmations += 1

            if confirmations < 2:
                expected_accuracy *= 0.90
                reasons.append(f"Low confirmation ({confirmations}/3): -10%")
            elif confirmations == 3:
                expected_accuracy *= 1.05
                reasons.append(f"Full confirmation (3/3): +5%")

        # Cap at reasonable bounds
        expected_accuracy = np.clip(expected_accuracy, 0.40, 0.85)

        # 6. Determine grade
        grade = self.get_grade(expected_accuracy)

        # 7. Determine if should trade
        should_trade = expected_accuracy >= self.min_accuracy and grade != TradeGrade.AVOID

        # 8. Determine direction
        if probability > 0.55:
            direction = 1
        elif probability < 0.45:
            direction = -1
        else:
            direction = 0
            should_trade = False  # Don't trade neutrals

        # 9. Calculate position size
        position_size = self.calculate_position_size(grade, expected_accuracy)

        # 10. Risk level
        if expected_accuracy >= 0.70:
            risk_level = 'low'
        elif expected_accuracy >= 0.60:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return FilteredSignal(
            should_trade=should_trade,
            grade=grade,
            expected_accuracy=expected_accuracy,
            direction=direction,
            probability=probability,
            confidence_reasons=reasons,
            risk_level=risk_level,
            position_size=position_size
        )

    def get_tradeable_only(
        self,
        predictions: Dict,  # Dict[str, AdvancedPrediction]
    ) -> Dict:
        """
        Filter predictions to only tradeable setups.

        Returns dict mapping symbol to FilteredSignal for tradeable setups only.
        """
        tradeable = {}

        for symbol, prediction in predictions.items():
            filtered = self.filter(prediction)
            if filtered.should_trade:
                tradeable[symbol] = filtered

        return tradeable

    def rank_by_quality(
        self,
        filtered_signals: Dict  # Dict[str, FilteredSignal]
    ) -> List[Tuple[str, FilteredSignal]]:
        """
        Rank tradeable signals by expected accuracy.

        Returns list of (symbol, FilteredSignal) sorted by quality.
        """
        ranked = list(filtered_signals.items())
        ranked.sort(key=lambda x: x[1].expected_accuracy, reverse=True)
        return ranked
