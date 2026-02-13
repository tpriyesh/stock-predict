"""
PredictionMonitor - Track and Monitor Live Prediction Performance

CRITICAL: Without monitoring, you won't know when your model breaks.

Problems this solves:
1. Model drift - market changes, model accuracy degrades
2. Calibration drift - 60% predictions no longer win 60%
3. Data quality issues - bad data leading to bad predictions
4. Silent failures - model keeps running but is wrong

This module tracks every prediction and its outcome to:
- Calculate rolling accuracy
- Detect calibration drift
- Generate alerts when performance drops
- Trigger kill switch when critical thresholds are breached
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from enum import Enum
import json
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of monitoring alerts."""
    ACCURACY_DROP = "ACCURACY_DROP"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    DATA_QUALITY = "DATA_QUALITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class MonitoringAlert:
    """A monitoring alert."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float
    should_disable: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'should_disable': self.should_disable,
            'resolved': self.resolved
        }


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    id: str
    symbol: str
    timestamp: datetime
    predicted_prob: float
    predicted_signal: str
    confidence: float

    # Component scores
    ml_score: Optional[float] = None
    llm_score: Optional[float] = None
    technical_score: Optional[float] = None

    # Trade setup
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None

    # Outcomes (filled later)
    actual_outcome: Optional[bool] = None
    actual_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'predicted_prob': self.predicted_prob,
            'predicted_signal': self.predicted_signal,
            'confidence': self.confidence,
            'ml_score': self.ml_score,
            'llm_score': self.llm_score,
            'technical_score': self.technical_score,
            'entry_price': self.entry_price,
            'actual_outcome': self.actual_outcome,
            'actual_price': self.actual_price,
            'outcome_timestamp': self.outcome_timestamp.isoformat() if self.outcome_timestamp else None
        }


class PredictionMonitor:
    """
    Monitor prediction performance in real-time.

    Features:
    1. Track all predictions and outcomes
    2. Calculate rolling accuracy metrics
    3. Detect calibration drift
    4. Generate alerts for performance issues
    5. Provide kill switch signal when needed

    Usage:
        monitor = PredictionMonitor()

        # Record each prediction
        record_id = monitor.record_prediction(
            symbol="RELIANCE",
            predicted_prob=0.65,
            predicted_signal="BUY",
            confidence=0.75
        )

        # Later, when outcome is known
        monitor.record_outcome(record_id, actual_went_up=True, actual_price=2650)

        # Check health
        status = monitor.check_health()
        if not status['trading_enabled']:
            print(f"Trading disabled: {status['reason']}")
    """

    # Thresholds
    ACCURACY_CRITICAL = 0.45      # Below this = kill switch
    ACCURACY_WARNING = 0.50       # Below this = warning
    CALIBRATION_MAX_ERROR = 0.15  # Max difference between predicted and actual
    CONSECUTIVE_LOSS_LIMIT = 7    # Max consecutive losing predictions
    MIN_SAMPLES_FOR_CHECK = 20    # Need at least this many evaluated predictions

    def __init__(self,
                 window_size: int = 50,
                 storage_path: Optional[Path] = None):
        """
        Initialize monitor.

        Args:
            window_size: Number of recent predictions to consider
            storage_path: Path for persistence
        """
        self.window_size = window_size
        self.storage_path = storage_path or Path("data/prediction_monitor.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Prediction records
        self.predictions: Dict[str, PredictionRecord] = {}
        self.prediction_order: List[str] = []  # Maintain order

        # Alerts
        self.alerts: List[MonitoringAlert] = []

        # State
        self.trading_enabled = True
        self.last_health_check: Optional[datetime] = None

        # Load historical
        self._load_state()

    def _load_state(self):
        """Load historical state."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for rec_data in data.get('predictions', []):
                    rec = PredictionRecord(
                        id=rec_data['id'],
                        symbol=rec_data['symbol'],
                        timestamp=datetime.fromisoformat(rec_data['timestamp']),
                        predicted_prob=rec_data['predicted_prob'],
                        predicted_signal=rec_data['predicted_signal'],
                        confidence=rec_data['confidence'],
                        ml_score=rec_data.get('ml_score'),
                        llm_score=rec_data.get('llm_score'),
                        technical_score=rec_data.get('technical_score'),
                        entry_price=rec_data.get('entry_price'),
                        actual_outcome=rec_data.get('actual_outcome'),
                        actual_price=rec_data.get('actual_price'),
                        outcome_timestamp=datetime.fromisoformat(rec_data['outcome_timestamp'])
                            if rec_data.get('outcome_timestamp') else None
                    )
                    self.predictions[rec.id] = rec
                    self.prediction_order.append(rec.id)

                self.trading_enabled = data.get('trading_enabled', True)

                logger.info(f"Loaded {len(self.predictions)} historical predictions")
        except Exception as e:
            logger.warning(f"Could not load monitor state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            # Keep only recent predictions
            recent_ids = self.prediction_order[-self.window_size * 2:]

            data = {
                'predictions': [
                    self.predictions[pid].to_dict()
                    for pid in recent_ids
                    if pid in self.predictions
                ],
                'trading_enabled': self.trading_enabled,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save monitor state: {e}")

    def generate_id(self) -> str:
        """Generate unique prediction ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def record_prediction(self,
                          symbol: str,
                          predicted_prob: float,
                          predicted_signal: str,
                          confidence: float,
                          ml_score: Optional[float] = None,
                          llm_score: Optional[float] = None,
                          technical_score: Optional[float] = None,
                          entry_price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          target_price: Optional[float] = None) -> str:
        """
        Record a new prediction.

        Returns:
            Prediction ID for later outcome recording
        """
        record_id = self.generate_id()

        record = PredictionRecord(
            id=record_id,
            symbol=symbol,
            timestamp=datetime.now(),
            predicted_prob=predicted_prob,
            predicted_signal=predicted_signal,
            confidence=confidence,
            ml_score=ml_score,
            llm_score=llm_score,
            technical_score=technical_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price
        )

        self.predictions[record_id] = record
        self.prediction_order.append(record_id)

        logger.debug(f"Recorded prediction {record_id}: {symbol} {predicted_signal} "
                    f"({predicted_prob:.1%})")

        return record_id

    def record_outcome(self,
                       prediction_id: str,
                       actual_went_up: bool,
                       actual_price: Optional[float] = None) -> bool:
        """
        Record the actual outcome for a prediction.

        Args:
            prediction_id: ID from record_prediction
            actual_went_up: Whether price actually went up
            actual_price: Actual price at evaluation time

        Returns:
            True if outcome recorded successfully
        """
        if prediction_id not in self.predictions:
            logger.warning(f"Prediction {prediction_id} not found")
            return False

        record = self.predictions[prediction_id]
        record.actual_outcome = actual_went_up
        record.actual_price = actual_price
        record.outcome_timestamp = datetime.now()

        self._save_state()

        # Check if we should run health check
        self._maybe_check_health()

        return True

    def _get_evaluated_predictions(self) -> List[PredictionRecord]:
        """Get predictions that have outcomes recorded."""
        return [
            self.predictions[pid]
            for pid in self.prediction_order[-self.window_size:]
            if pid in self.predictions and
            self.predictions[pid].actual_outcome is not None
        ]

    def calculate_accuracy(self) -> Tuple[float, int]:
        """Calculate rolling accuracy."""
        evaluated = self._get_evaluated_predictions()

        if len(evaluated) < 10:
            return 0.5, len(evaluated)

        correct = sum(
            1 for p in evaluated
            if (p.predicted_prob > 0.5) == p.actual_outcome
        )

        return correct / len(evaluated), len(evaluated)

    def calculate_calibration_error(self) -> Dict[str, float]:
        """
        Calculate calibration error per probability bucket.

        Good calibration means:
        - When we predict 60%, outcomes are ~60% positive
        - When we predict 40%, outcomes are ~40% positive
        """
        evaluated = self._get_evaluated_predictions()

        if len(evaluated) < 30:
            return {}

        # Bucket predictions
        buckets = defaultdict(list)
        for p in evaluated:
            bucket = round(p.predicted_prob, 1)  # 0.1 buckets
            buckets[bucket].append(p.actual_outcome)

        calibration = {}
        for bucket, outcomes in buckets.items():
            if len(outcomes) >= 5:
                actual_rate = sum(outcomes) / len(outcomes)
                error = abs(actual_rate - bucket)
                calibration[f"bucket_{bucket}"] = {
                    'predicted': bucket,
                    'actual': actual_rate,
                    'error': error,
                    'n_samples': len(outcomes)
                }

        # Average calibration error
        if calibration:
            avg_error = np.mean([c['error'] for c in calibration.values()])
            calibration['average_error'] = avg_error

        return calibration

    def check_consecutive_losses(self) -> int:
        """Check for consecutive losing predictions."""
        evaluated = self._get_evaluated_predictions()

        if not evaluated:
            return 0

        # Count consecutive losses from most recent
        consecutive = 0
        for p in reversed(evaluated):
            was_correct = (p.predicted_prob > 0.5) == p.actual_outcome
            if not was_correct:
                consecutive += 1
            else:
                break

        return consecutive

    def check_health(self) -> Dict:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status and metrics
        """
        self.last_health_check = datetime.now()

        alerts_generated = []

        # Check accuracy
        accuracy, n_samples = self.calculate_accuracy()

        if n_samples >= self.MIN_SAMPLES_FOR_CHECK:
            if accuracy < self.ACCURACY_CRITICAL:
                alert = MonitoringAlert(
                    alert_type=AlertType.ACCURACY_DROP,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Accuracy dropped to {accuracy:.1%} (critical threshold: {self.ACCURACY_CRITICAL:.1%})",
                    timestamp=datetime.now(),
                    metric_value=accuracy,
                    threshold=self.ACCURACY_CRITICAL,
                    should_disable=True
                )
                alerts_generated.append(alert)
                self.alerts.append(alert)

            elif accuracy < self.ACCURACY_WARNING:
                alert = MonitoringAlert(
                    alert_type=AlertType.ACCURACY_DROP,
                    severity=AlertSeverity.WARNING,
                    message=f"Accuracy at {accuracy:.1%} (warning threshold: {self.ACCURACY_WARNING:.1%})",
                    timestamp=datetime.now(),
                    metric_value=accuracy,
                    threshold=self.ACCURACY_WARNING,
                    should_disable=False
                )
                alerts_generated.append(alert)
                self.alerts.append(alert)

        # Check calibration
        calibration = self.calculate_calibration_error()
        avg_cal_error = calibration.get('average_error', 0)

        if avg_cal_error > self.CALIBRATION_MAX_ERROR:
            alert = MonitoringAlert(
                alert_type=AlertType.CALIBRATION_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"Calibration error at {avg_cal_error:.1%} (threshold: {self.CALIBRATION_MAX_ERROR:.1%})",
                timestamp=datetime.now(),
                metric_value=avg_cal_error,
                threshold=self.CALIBRATION_MAX_ERROR,
                should_disable=False
            )
            alerts_generated.append(alert)
            self.alerts.append(alert)

        # Check consecutive losses
        consecutive_losses = self.check_consecutive_losses()

        if consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
            alert = MonitoringAlert(
                alert_type=AlertType.CONSECUTIVE_LOSSES,
                severity=AlertSeverity.CRITICAL,
                message=f"{consecutive_losses} consecutive losing predictions",
                timestamp=datetime.now(),
                metric_value=consecutive_losses,
                threshold=self.CONSECUTIVE_LOSS_LIMIT,
                should_disable=True
            )
            alerts_generated.append(alert)
            self.alerts.append(alert)

        # Determine if trading should be enabled
        should_disable = any(a.should_disable for a in alerts_generated)

        if should_disable:
            self.trading_enabled = False
            disable_reason = next(
                (a.message for a in alerts_generated if a.should_disable),
                "Performance threshold breached"
            )
        else:
            disable_reason = None

        self._save_state()

        return {
            'trading_enabled': self.trading_enabled,
            'disable_reason': disable_reason,
            'accuracy': accuracy,
            'n_samples': n_samples,
            'calibration_error': avg_cal_error,
            'consecutive_losses': consecutive_losses,
            'alerts': [a.to_dict() for a in alerts_generated],
            'checked_at': self.last_health_check.isoformat()
        }

    def _maybe_check_health(self):
        """Check health if enough time has passed or enough new data."""
        evaluated = self._get_evaluated_predictions()

        # Check every 10 new predictions
        if len(evaluated) % 10 == 0 and len(evaluated) > 0:
            self.check_health()

    def should_trade(self) -> Tuple[bool, str]:
        """
        Quick check if trading should proceed.

        Returns:
            (can_trade, reason)
        """
        if not self.trading_enabled:
            return False, "Trading disabled due to performance issues"

        # Check if health was checked recently
        if self.last_health_check:
            hours_since = (datetime.now() - self.last_health_check).total_seconds() / 3600
            if hours_since > 24:
                status = self.check_health()
                if not status['trading_enabled']:
                    return False, status['disable_reason']

        return True, "OK"

    def reset_trading(self, reason: str = "Manual reset"):
        """Re-enable trading (manual override)."""
        self.trading_enabled = True
        logger.warning(f"Trading re-enabled: {reason}")
        self._save_state()

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        evaluated = self._get_evaluated_predictions()

        if not evaluated:
            return {'status': 'No evaluated predictions'}

        # Accuracy by signal type
        by_signal = defaultdict(list)
        for p in evaluated:
            by_signal[p.predicted_signal].append(
                (p.predicted_prob > 0.5) == p.actual_outcome
            )

        signal_accuracy = {
            signal: sum(results) / len(results)
            for signal, results in by_signal.items()
        }

        # Accuracy by confidence bucket
        by_confidence = {'high': [], 'medium': [], 'low': []}
        for p in evaluated:
            if p.confidence >= 0.7:
                bucket = 'high'
            elif p.confidence >= 0.5:
                bucket = 'medium'
            else:
                bucket = 'low'
            by_confidence[bucket].append(
                (p.predicted_prob > 0.5) == p.actual_outcome
            )

        confidence_accuracy = {
            bucket: sum(results) / len(results) if results else 0
            for bucket, results in by_confidence.items()
        }

        accuracy, n = self.calculate_accuracy()
        calibration = self.calculate_calibration_error()

        return {
            'overall_accuracy': accuracy,
            'n_predictions': n,
            'accuracy_by_signal': signal_accuracy,
            'accuracy_by_confidence': confidence_accuracy,
            'calibration': calibration,
            'consecutive_losses': self.check_consecutive_losses(),
            'trading_enabled': self.trading_enabled,
            'active_alerts': len([a for a in self.alerts if not a.resolved])
        }


def demo():
    """Demonstrate prediction monitoring."""
    print("=" * 60)
    print("PredictionMonitor Demo")
    print("=" * 60)

    monitor = PredictionMonitor(
        window_size=50,
        storage_path=Path("/tmp/test_monitor.json")
    )

    # Simulate predictions
    np.random.seed(42)

    print("\nSimulating 60 predictions...")

    for i in range(60):
        prob = np.random.uniform(0.4, 0.7)
        signal = 'BUY' if prob > 0.5 else 'SELL'

        record_id = monitor.record_prediction(
            symbol="TEST",
            predicted_prob=prob,
            predicted_signal=signal,
            confidence=np.random.uniform(0.5, 0.9)
        )

        # Simulate outcome (55% accuracy)
        if i < 55:
            predicted_up = prob > 0.5
            if np.random.random() < 0.55:
                actual_up = predicted_up
            else:
                actual_up = not predicted_up

            monitor.record_outcome(record_id, actual_up)

    # Check health
    print("\n--- Health Check ---")
    status = monitor.check_health()
    print(f"Trading Enabled: {status['trading_enabled']}")
    print(f"Accuracy: {status['accuracy']:.1%}")
    print(f"Calibration Error: {status['calibration_error']:.1%}")
    print(f"Consecutive Losses: {status['consecutive_losses']}")

    if status['alerts']:
        print("\nAlerts:")
        for alert in status['alerts']:
            print(f"  [{alert['severity']}] {alert['message']}")

    # Performance summary
    print("\n--- Performance Summary ---")
    summary = monitor.get_performance_summary()
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Predictions Evaluated: {summary['n_predictions']}")

    print("\nAccuracy by Signal:")
    for signal, acc in summary['accuracy_by_signal'].items():
        print(f"  {signal}: {acc:.1%}")

    print("\nAccuracy by Confidence:")
    for bucket, acc in summary['accuracy_by_confidence'].items():
        print(f"  {bucket}: {acc:.1%}")


if __name__ == "__main__":
    demo()
