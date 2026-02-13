"""
AdaptiveWeightCalculator - Dynamic Ensemble Weight Learning

CRITICAL: Static ensemble weights are WRONG.

Problem:
    WEIGHTS = {'ml_rf': 0.15, 'ml_xgb': 0.15, 'llm_openai': 0.15}
    These arbitrary weights ignore:
    - Historical accuracy of each component
    - Current market regime performance
    - Recent prediction quality

Solution: Learn weights from historical performance.
    - Track accuracy of each component over time
    - Weight components by their proven accuracy
    - Decay older predictions (recent accuracy matters more)
    - Adjust for regime (some models work better in bull/bear)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
from loguru import logger


@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking."""
    timestamp: datetime
    component: str
    predicted_prob: float
    predicted_signal: str
    actual_outcome: Optional[bool] = None  # True if went up
    correct: Optional[bool] = None


@dataclass
class ComponentStats:
    """Statistics for a single prediction component."""
    component: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    recent_accuracy: float  # Last 50 predictions
    calibration_error: float  # |predicted_prob - actual_rate|
    last_updated: datetime

    def to_dict(self) -> dict:
        return {
            'component': self.component,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.accuracy,
            'recent_accuracy': self.recent_accuracy,
            'calibration_error': self.calibration_error,
            'last_updated': self.last_updated.isoformat()
        }


class AdaptiveWeightCalculator:
    """
    Calculate ensemble weights based on historical accuracy.

    Instead of arbitrary fixed weights, this learns optimal weights
    from each component's track record.

    Features:
    1. Track prediction accuracy per component
    2. Exponential decay (recent accuracy matters more)
    3. Minimum sample requirements
    4. Bayesian smoothing (handle components with few predictions)
    5. Regime-aware weighting

    Usage:
        calculator = AdaptiveWeightCalculator()

        # Record predictions as they come in
        calculator.record_prediction('ml_rf', 0.65, 'BUY')
        calculator.record_prediction('llm_openai', 0.58, 'BUY')

        # When outcome is known (5 days later)
        calculator.record_outcome(timestamp, actual_went_up=True)

        # Get current optimal weights
        weights = calculator.get_weights()
        # {'ml_rf': 0.22, 'ml_xgb': 0.18, 'llm_openai': 0.15, ...}
    """

    # Default priors (Bayesian smoothing)
    DEFAULT_PRIOR_ACCURACY = 0.50  # Assume 50% before evidence
    DEFAULT_PRIOR_WEIGHT = 20     # Equivalent to 20 prior observations

    # Minimum requirements
    MIN_PREDICTIONS = 20          # Need at least this many to trust a component
    MIN_WEIGHT = 0.05            # Minimum weight for any component
    MAX_WEIGHT = 0.40            # Maximum weight for any component

    def __init__(self,
                 lookback_predictions: int = 100,
                 decay_half_life: int = 30,
                 storage_path: Optional[Path] = None):
        """
        Initialize adaptive weight calculator.

        Args:
            lookback_predictions: Number of recent predictions to consider
            decay_half_life: Half-life for exponential decay (in predictions)
            storage_path: Path to persist prediction history
        """
        self.lookback = lookback_predictions
        self.decay_half_life = decay_half_life
        self.storage_path = storage_path or Path("data/prediction_tracking.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Prediction history per component
        self.predictions: Dict[str, List[PredictionRecord]] = defaultdict(list)

        # Cached weights
        self._cached_weights: Optional[Dict[str, float]] = None
        self._cache_timestamp: Optional[datetime] = None

        # Load historical data
        self._load_history()

    def _load_history(self):
        """Load historical predictions from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for component, records in data.get('predictions', {}).items():
                    for rec in records:
                        self.predictions[component].append(PredictionRecord(
                            timestamp=datetime.fromisoformat(rec['timestamp']),
                            component=component,
                            predicted_prob=rec['predicted_prob'],
                            predicted_signal=rec['predicted_signal'],
                            actual_outcome=rec.get('actual_outcome'),
                            correct=rec.get('correct')
                        ))

                logger.info(f"Loaded {sum(len(p) for p in self.predictions.values())} "
                           f"historical predictions")
        except Exception as e:
            logger.warning(f"Could not load prediction history: {e}")

    def _save_history(self):
        """Save predictions to storage."""
        try:
            data = {
                'predictions': {
                    component: [
                        {
                            'timestamp': rec.timestamp.isoformat(),
                            'predicted_prob': rec.predicted_prob,
                            'predicted_signal': rec.predicted_signal,
                            'actual_outcome': rec.actual_outcome,
                            'correct': rec.correct
                        }
                        for rec in records[-self.lookback * 2:]  # Keep 2x lookback
                    ]
                    for component, records in self.predictions.items()
                },
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save prediction history: {e}")

    def record_prediction(self,
                          component: str,
                          predicted_prob: float,
                          predicted_signal: str,
                          timestamp: Optional[datetime] = None) -> None:
        """
        Record a new prediction from a component.

        Args:
            component: Component name (e.g., 'ml_rf', 'llm_openai')
            predicted_prob: Probability of upward movement (0-1)
            predicted_signal: Signal string (BUY, SELL, HOLD)
            timestamp: Prediction timestamp (default: now)
        """
        record = PredictionRecord(
            timestamp=timestamp or datetime.now(),
            component=component,
            predicted_prob=predicted_prob,
            predicted_signal=predicted_signal
        )

        self.predictions[component].append(record)
        self._invalidate_cache()

        logger.debug(f"Recorded prediction: {component} -> {predicted_signal} "
                    f"({predicted_prob:.1%})")

    def record_outcome(self,
                       timestamp: datetime,
                       actual_up: bool,
                       tolerance_hours: float = 24) -> int:
        """
        Record actual outcome for predictions around a timestamp.

        Args:
            timestamp: Timestamp of the prediction to update
            actual_up: Whether price actually went up
            tolerance_hours: Match predictions within this window

        Returns:
            Number of predictions updated
        """
        updated = 0

        for component, records in self.predictions.items():
            for record in records:
                # Find predictions close to this timestamp
                time_diff = abs((record.timestamp - timestamp).total_seconds() / 3600)

                if time_diff <= tolerance_hours and record.actual_outcome is None:
                    record.actual_outcome = actual_up

                    # Determine if prediction was correct
                    predicted_up = record.predicted_prob > 0.5
                    record.correct = (predicted_up == actual_up)

                    updated += 1

        if updated > 0:
            self._save_history()
            self._invalidate_cache()

        return updated

    def _invalidate_cache(self):
        """Invalidate cached weights."""
        self._cached_weights = None
        self._cache_timestamp = None

    def _calculate_decay_weights(self, n: int) -> np.ndarray:
        """Calculate exponential decay weights for predictions."""
        if n == 0:
            return np.array([])

        lambda_decay = np.log(2) / self.decay_half_life
        ages = np.arange(n)[::-1]  # Newest = 0, oldest = n-1
        weights = np.exp(-lambda_decay * ages)

        return weights / weights.sum()  # Normalize

    def _calculate_component_accuracy(self, component: str) -> ComponentStats:
        """Calculate accuracy statistics for a component."""
        records = self.predictions.get(component, [])

        # Filter to evaluated predictions
        evaluated = [r for r in records if r.correct is not None]

        if not evaluated:
            return ComponentStats(
                component=component,
                total_predictions=0,
                correct_predictions=0,
                accuracy=self.DEFAULT_PRIOR_ACCURACY,
                recent_accuracy=self.DEFAULT_PRIOR_ACCURACY,
                calibration_error=0.0,
                last_updated=datetime.now()
            )

        # Get recent predictions (up to lookback)
        recent = evaluated[-self.lookback:]

        # Overall accuracy with Bayesian smoothing
        total = len(evaluated)
        correct = sum(1 for r in evaluated if r.correct)

        # Bayesian posterior: (prior_correct + actual_correct) / (prior_total + actual_total)
        smoothed_correct = self.DEFAULT_PRIOR_ACCURACY * self.DEFAULT_PRIOR_WEIGHT + correct
        smoothed_total = self.DEFAULT_PRIOR_WEIGHT + total
        accuracy = smoothed_correct / smoothed_total

        # Recent accuracy with decay weighting
        recent_weights = self._calculate_decay_weights(len(recent))
        recent_correct = np.array([1.0 if r.correct else 0.0 for r in recent])
        recent_accuracy = np.average(recent_correct, weights=recent_weights) if len(recent) > 0 else self.DEFAULT_PRIOR_ACCURACY

        # Calibration error: how well do probabilities match reality?
        # Group by probability bucket and check actual win rate
        calibration_errors = []
        for bucket_start in [0.3, 0.4, 0.5, 0.6, 0.7]:
            bucket_end = bucket_start + 0.1
            bucket_preds = [r for r in recent
                          if bucket_start <= r.predicted_prob < bucket_end]

            if len(bucket_preds) >= 5:
                actual_rate = sum(1 for r in bucket_preds if r.actual_outcome) / len(bucket_preds)
                expected_rate = (bucket_start + bucket_end) / 2
                calibration_errors.append(abs(actual_rate - expected_rate))

        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0

        return ComponentStats(
            component=component,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            recent_accuracy=recent_accuracy,
            calibration_error=avg_calibration_error,
            last_updated=datetime.now()
        )

    def get_component_stats(self) -> Dict[str, ComponentStats]:
        """Get statistics for all components."""
        return {
            component: self._calculate_component_accuracy(component)
            for component in self.predictions.keys()
        }

    def get_weights(self,
                    components: Optional[List[str]] = None,
                    use_cache: bool = True) -> Dict[str, float]:
        """
        Get optimal weights for ensemble aggregation.

        Args:
            components: List of components to weight (default: all known)
            use_cache: Whether to use cached weights if available

        Returns:
            Dictionary mapping component names to weights (sum to 1.0)
        """
        # Check cache
        if use_cache and self._cached_weights is not None:
            if self._cache_timestamp:
                age = (datetime.now() - self._cache_timestamp).total_seconds()
                if age < 300:  # Cache valid for 5 minutes
                    if components is None:
                        return self._cached_weights
                    else:
                        # Filter to requested components
                        weights = {c: self._cached_weights.get(c, 0.1)
                                  for c in components}
                        total = sum(weights.values())
                        return {c: w/total for c, w in weights.items()}

        # Calculate fresh weights
        if components is None:
            components = list(self.predictions.keys())

        if not components:
            # No history - use equal weights
            return {}

        # Get accuracy for each component
        accuracies = {}
        for comp in components:
            stats = self._calculate_component_accuracy(comp)

            if stats.total_predictions >= self.MIN_PREDICTIONS:
                # Weight by recent accuracy (more responsive to performance changes)
                # Combine overall and recent (recent gets 70% weight)
                combined = 0.3 * stats.accuracy + 0.7 * stats.recent_accuracy
                # Penalty for poor calibration
                combined *= (1 - stats.calibration_error)
                accuracies[comp] = combined
            else:
                # Not enough data - use prior
                accuracies[comp] = self.DEFAULT_PRIOR_ACCURACY

        # Convert accuracies to weights
        # Components with accuracy > 50% get weight proportional to edge
        edges = {}
        for comp, acc in accuracies.items():
            # Edge = accuracy - 50% (assuming random is 50%)
            edge = max(0.001, acc - 0.5)  # Small floor to prevent zero weights
            edges[comp] = edge

        # Normalize to sum to 1
        total_edge = sum(edges.values())
        weights = {comp: edge / total_edge for comp, edge in edges.items()}

        # Apply min/max constraints
        for comp in weights:
            weights[comp] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weights[comp]))

        # Re-normalize after constraints
        total = sum(weights.values())
        weights = {c: w/total for c, w in weights.items()}

        # Cache results
        self._cached_weights = weights
        self._cache_timestamp = datetime.now()

        return weights

    def get_weight_explanation(self) -> str:
        """Get human-readable explanation of current weights."""
        stats = self.get_component_stats()
        weights = self.get_weights()

        lines = ["Component Weights (based on historical accuracy):"]
        lines.append("-" * 50)

        for comp in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
            s = stats.get(comp)
            if s:
                lines.append(
                    f"  {comp:20s}: {weights[comp]:.1%} weight "
                    f"(accuracy: {s.accuracy:.1%}, "
                    f"recent: {s.recent_accuracy:.1%}, "
                    f"n={s.total_predictions})"
                )
            else:
                lines.append(f"  {comp:20s}: {weights[comp]:.1%} weight (no data)")

        return "\n".join(lines)

    def aggregate_predictions(self,
                              predictions: Dict[str, float],
                              use_adaptive_weights: bool = True) -> float:
        """
        Aggregate multiple component predictions into single probability.

        Args:
            predictions: Dict of component_name -> probability
            use_adaptive_weights: If True, use learned weights; else equal

        Returns:
            Aggregated probability
        """
        if not predictions:
            return 0.5

        if use_adaptive_weights:
            weights = self.get_weights(list(predictions.keys()))
        else:
            n = len(predictions)
            weights = {c: 1/n for c in predictions}

        weighted_sum = sum(
            predictions[comp] * weights.get(comp, 0.1)
            for comp in predictions
        )
        total_weight = sum(weights.get(comp, 0.1) for comp in predictions)

        return weighted_sum / total_weight if total_weight > 0 else 0.5


def demo():
    """Demonstrate adaptive weight calculation."""
    print("=" * 60)
    print("AdaptiveWeightCalculator Demo")
    print("=" * 60)

    calculator = AdaptiveWeightCalculator(storage_path=Path("/tmp/test_weights.json"))

    # Simulate historical predictions
    np.random.seed(42)
    components = ['ml_rf', 'ml_xgb', 'ml_gb', 'llm_openai', 'llm_gemini', 'technical']

    # Different accuracies per component
    true_accuracies = {
        'ml_rf': 0.58,
        'ml_xgb': 0.55,
        'ml_gb': 0.53,
        'llm_openai': 0.52,
        'llm_gemini': 0.51,
        'technical': 0.54
    }

    print("\nSimulating 100 predictions per component...")

    for i in range(100):
        timestamp = datetime.now() - timedelta(days=100-i)

        for comp in components:
            # Random prediction
            prob = np.random.uniform(0.4, 0.7)
            signal = 'BUY' if prob > 0.5 else 'SELL'

            calculator.record_prediction(comp, prob, signal, timestamp)

            # Generate outcome based on true accuracy
            actual_up = np.random.random() < true_accuracies[comp]
            calculator.record_outcome(timestamp, actual_up)

    print("\n" + calculator.get_weight_explanation())

    print("\n--- Aggregating Sample Predictions ---")
    sample_preds = {
        'ml_rf': 0.65,
        'ml_xgb': 0.58,
        'ml_gb': 0.52,
        'llm_openai': 0.48,
        'technical': 0.60
    }

    print("Component predictions:")
    for comp, prob in sample_preds.items():
        print(f"  {comp}: {prob:.1%}")

    aggregated = calculator.aggregate_predictions(sample_preds)
    print(f"\nAdaptive-weighted aggregation: {aggregated:.1%}")

    # Compare to simple average
    simple_avg = np.mean(list(sample_preds.values()))
    print(f"Simple average: {simple_avg:.1%}")


if __name__ == "__main__":
    demo()
