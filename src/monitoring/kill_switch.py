"""
KillSwitch - Automated Trading Disable System

CRITICAL: When models fail, they should STOP making predictions.

This module provides:
1. Automatic detection of system failures
2. Immediate trading halt when thresholds breached
3. Gradual re-enablement after recovery
4. Alert notifications

The kill switch activates when:
- Accuracy drops below critical threshold
- Consecutive losses exceed limit
- System errors occur
- Data quality issues detected
- Manual override triggered
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from loguru import logger


class TradingStatus(Enum):
    """Current trading status."""
    ENABLED = "ENABLED"              # Normal operation
    DEGRADED = "DEGRADED"            # Warnings present, proceed with caution
    SUSPENDED = "SUSPENDED"          # Temporarily halted, will auto-recover
    DISABLED = "DISABLED"            # Halted, requires manual intervention
    MAINTENANCE = "MAINTENANCE"      # Planned maintenance window


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""
    status: TradingStatus
    reason: str
    triggered_at: Optional[datetime]
    auto_recover_at: Optional[datetime]
    manual_override: bool
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            'status': self.status.value,
            'reason': self.reason,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'auto_recover_at': self.auto_recover_at.isoformat() if self.auto_recover_at else None,
            'manual_override': self.manual_override,
            'metrics': self.metrics
        }


class KillSwitch:
    """
    Automated kill switch for the prediction system.

    Monitors key metrics and automatically disables predictions
    when performance degrades below acceptable thresholds.

    Usage:
        kill_switch = KillSwitch()

        # Before making a prediction
        can_predict, reason = kill_switch.check()
        if not can_predict:
            return None  # Don't make prediction

        # Update metrics periodically
        kill_switch.update_metrics(
            accuracy=0.48,
            consecutive_losses=5,
            data_quality=0.95
        )

        # Manual controls
        kill_switch.trigger("Manual halt for investigation")
        kill_switch.reset("Investigation complete, resuming")
    """

    # Thresholds
    ACCURACY_CRITICAL = 0.45
    ACCURACY_WARNING = 0.50
    CONSECUTIVE_LOSS_CRITICAL = 7
    CONSECUTIVE_LOSS_WARNING = 5
    DATA_QUALITY_CRITICAL = 0.80
    CALIBRATION_ERROR_CRITICAL = 0.20

    # Recovery
    AUTO_RECOVERY_HOURS = 4  # Auto-recover after this time if metrics improve
    RECOVERY_CHECK_INTERVAL = 60  # Seconds between recovery checks

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize kill switch."""
        self.storage_path = storage_path or Path("data/kill_switch_state.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # State
        self.state = KillSwitchState(
            status=TradingStatus.ENABLED,
            reason="System operational",
            triggered_at=None,
            auto_recover_at=None,
            manual_override=False,
            metrics={}
        )

        # History
        self.trigger_history: List[Dict] = []

        # Callbacks
        self.on_trigger_callbacks: List[callable] = []
        self.on_recover_callbacks: List[callable] = []

        self._load_state()

    def _load_state(self):
        """Load state from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                self.state = KillSwitchState(
                    status=TradingStatus(data.get('status', 'ENABLED')),
                    reason=data.get('reason', ''),
                    triggered_at=datetime.fromisoformat(data['triggered_at'])
                        if data.get('triggered_at') else None,
                    auto_recover_at=datetime.fromisoformat(data['auto_recover_at'])
                        if data.get('auto_recover_at') else None,
                    manual_override=data.get('manual_override', False),
                    metrics=data.get('metrics', {})
                )
                logger.info(f"Loaded kill switch state: {self.state.status.value}")
        except Exception as e:
            logger.warning(f"Could not load kill switch state: {e}")

    def _save_state(self):
        """Save state to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save kill switch state: {e}")

    def check(self) -> Tuple[bool, str]:
        """
        Check if predictions/trading can proceed.

        Returns:
            (can_proceed, reason)
        """
        # Check for auto-recovery
        self._check_auto_recovery()

        if self.state.status == TradingStatus.ENABLED:
            return True, "OK"

        elif self.state.status == TradingStatus.DEGRADED:
            return True, f"Warning: {self.state.reason}"

        elif self.state.status == TradingStatus.SUSPENDED:
            return False, f"Suspended: {self.state.reason}"

        elif self.state.status == TradingStatus.DISABLED:
            return False, f"Disabled: {self.state.reason}"

        elif self.state.status == TradingStatus.MAINTENANCE:
            return False, "System under maintenance"

        return False, "Unknown state"

    def update_metrics(self,
                       accuracy: Optional[float] = None,
                       consecutive_losses: Optional[int] = None,
                       calibration_error: Optional[float] = None,
                       data_quality: Optional[float] = None,
                       **kwargs) -> bool:
        """
        Update performance metrics and check thresholds.

        Returns:
            True if status changed
        """
        # Update metrics
        if accuracy is not None:
            self.state.metrics['accuracy'] = accuracy
        if consecutive_losses is not None:
            self.state.metrics['consecutive_losses'] = consecutive_losses
        if calibration_error is not None:
            self.state.metrics['calibration_error'] = calibration_error
        if data_quality is not None:
            self.state.metrics['data_quality'] = data_quality

        for k, v in kwargs.items():
            self.state.metrics[k] = v

        # Check thresholds
        status_changed = self._evaluate_thresholds()

        self._save_state()

        return status_changed

    def _evaluate_thresholds(self) -> bool:
        """Evaluate metrics against thresholds."""
        metrics = self.state.metrics
        original_status = self.state.status

        critical_reasons = []
        warning_reasons = []

        # Accuracy check
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            if acc < self.ACCURACY_CRITICAL:
                critical_reasons.append(f"Accuracy {acc:.1%} < {self.ACCURACY_CRITICAL:.1%}")
            elif acc < self.ACCURACY_WARNING:
                warning_reasons.append(f"Accuracy {acc:.1%} < {self.ACCURACY_WARNING:.1%}")

        # Consecutive losses
        if 'consecutive_losses' in metrics:
            losses = metrics['consecutive_losses']
            if losses >= self.CONSECUTIVE_LOSS_CRITICAL:
                critical_reasons.append(f"{losses} consecutive losses")
            elif losses >= self.CONSECUTIVE_LOSS_WARNING:
                warning_reasons.append(f"{losses} consecutive losses")

        # Data quality
        if 'data_quality' in metrics:
            quality = metrics['data_quality']
            if quality < self.DATA_QUALITY_CRITICAL:
                critical_reasons.append(f"Data quality {quality:.1%}")

        # Calibration
        if 'calibration_error' in metrics:
            cal_err = metrics['calibration_error']
            if cal_err > self.CALIBRATION_ERROR_CRITICAL:
                warning_reasons.append(f"Calibration error {cal_err:.1%}")

        # Determine new status
        if critical_reasons:
            self._trigger_internal(
                TradingStatus.SUSPENDED,
                "; ".join(critical_reasons)
            )
        elif warning_reasons:
            self.state.status = TradingStatus.DEGRADED
            self.state.reason = "; ".join(warning_reasons)
        elif original_status in [TradingStatus.DEGRADED, TradingStatus.SUSPENDED]:
            # Metrics improved, consider recovery
            self._consider_recovery()

        return self.state.status != original_status

    def _trigger_internal(self, status: TradingStatus, reason: str):
        """Internal trigger (from metrics)."""
        if self.state.manual_override:
            return  # Don't override manual decisions

        self.state.status = status
        self.state.reason = reason
        self.state.triggered_at = datetime.now()
        self.state.auto_recover_at = datetime.now() + timedelta(hours=self.AUTO_RECOVERY_HOURS)

        self.trigger_history.append({
            'status': status.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.state.metrics.copy()
        })

        logger.warning(f"Kill switch triggered: {status.value} - {reason}")

        # Call callbacks
        for callback in self.on_trigger_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Trigger callback failed: {e}")

    def trigger(self, reason: str, status: TradingStatus = TradingStatus.DISABLED):
        """
        Manually trigger the kill switch.

        Args:
            reason: Reason for triggering
            status: New status (default: DISABLED)
        """
        self.state.status = status
        self.state.reason = reason
        self.state.triggered_at = datetime.now()
        self.state.manual_override = True
        self.state.auto_recover_at = None  # No auto-recovery for manual

        self.trigger_history.append({
            'status': status.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'manual': True
        })

        self._save_state()

        logger.warning(f"Kill switch manually triggered: {reason}")

        for callback in self.on_trigger_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Trigger callback failed: {e}")

    def reset(self, reason: str = "Manual reset"):
        """
        Reset kill switch to enabled state.

        Args:
            reason: Reason for reset
        """
        self.state.status = TradingStatus.ENABLED
        self.state.reason = reason
        self.state.triggered_at = None
        self.state.auto_recover_at = None
        self.state.manual_override = False

        self._save_state()

        logger.info(f"Kill switch reset: {reason}")

        for callback in self.on_recover_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

    def _check_auto_recovery(self):
        """Check if auto-recovery should happen."""
        if self.state.manual_override:
            return  # Don't auto-recover manual triggers

        if self.state.status not in [TradingStatus.SUSPENDED]:
            return

        if self.state.auto_recover_at and datetime.now() >= self.state.auto_recover_at:
            # Check if metrics have improved
            metrics = self.state.metrics

            can_recover = True
            if 'accuracy' in metrics and metrics['accuracy'] < self.ACCURACY_WARNING:
                can_recover = False
            if 'consecutive_losses' in metrics and metrics['consecutive_losses'] >= self.CONSECUTIVE_LOSS_WARNING:
                can_recover = False

            if can_recover:
                self.reset("Auto-recovery: metrics improved")
            else:
                # Extend suspension
                self.state.auto_recover_at = datetime.now() + timedelta(hours=2)
                logger.info("Auto-recovery postponed: metrics not improved")

    def _consider_recovery(self):
        """Consider recovering from degraded/suspended state."""
        if self.state.status == TradingStatus.DEGRADED:
            self.state.status = TradingStatus.ENABLED
            self.state.reason = "Metrics improved"

    def enter_maintenance(self, duration_hours: float = 1.0, reason: str = "Scheduled maintenance"):
        """Enter maintenance mode."""
        self.state.status = TradingStatus.MAINTENANCE
        self.state.reason = reason
        self.state.triggered_at = datetime.now()
        self.state.auto_recover_at = datetime.now() + timedelta(hours=duration_hours)

        self._save_state()
        logger.info(f"Entered maintenance mode: {reason}")

    def get_status_report(self) -> Dict:
        """Get comprehensive status report."""
        return {
            'status': self.state.status.value,
            'reason': self.state.reason,
            'triggered_at': self.state.triggered_at.isoformat() if self.state.triggered_at else None,
            'auto_recover_at': self.state.auto_recover_at.isoformat() if self.state.auto_recover_at else None,
            'manual_override': self.state.manual_override,
            'current_metrics': self.state.metrics,
            'thresholds': {
                'accuracy_critical': self.ACCURACY_CRITICAL,
                'accuracy_warning': self.ACCURACY_WARNING,
                'consecutive_loss_critical': self.CONSECUTIVE_LOSS_CRITICAL,
                'data_quality_critical': self.DATA_QUALITY_CRITICAL
            },
            'recent_triggers': self.trigger_history[-5:] if self.trigger_history else []
        }

    def add_trigger_callback(self, callback: callable):
        """Add callback for when kill switch triggers."""
        self.on_trigger_callbacks.append(callback)

    def add_recover_callback(self, callback: callable):
        """Add callback for when kill switch recovers."""
        self.on_recover_callbacks.append(callback)


def demo():
    """Demonstrate kill switch functionality."""
    print("=" * 60)
    print("KillSwitch Demo")
    print("=" * 60)

    kill_switch = KillSwitch(storage_path=Path("/tmp/test_kill_switch.json"))

    # Initial check
    can_trade, reason = kill_switch.check()
    print(f"\nInitial: can_trade={can_trade}, reason={reason}")

    # Good metrics
    print("\n--- Updating with good metrics ---")
    kill_switch.update_metrics(
        accuracy=0.58,
        consecutive_losses=2,
        data_quality=0.95
    )
    can_trade, reason = kill_switch.check()
    print(f"Status: {kill_switch.state.status.value}")
    print(f"Can trade: {can_trade}")

    # Warning metrics
    print("\n--- Updating with warning metrics ---")
    kill_switch.update_metrics(
        accuracy=0.48,
        consecutive_losses=4
    )
    can_trade, reason = kill_switch.check()
    print(f"Status: {kill_switch.state.status.value}")
    print(f"Reason: {reason}")

    # Critical metrics
    print("\n--- Updating with critical metrics ---")
    kill_switch.update_metrics(
        accuracy=0.42,
        consecutive_losses=8
    )
    can_trade, reason = kill_switch.check()
    print(f"Status: {kill_switch.state.status.value}")
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")

    # Manual reset
    print("\n--- Manual reset ---")
    kill_switch.reset("Investigation complete")
    can_trade, reason = kill_switch.check()
    print(f"Status: {kill_switch.state.status.value}")
    print(f"Can trade: {can_trade}")

    # Status report
    print("\n--- Status Report ---")
    report = kill_switch.get_status_report()
    print(f"Current Status: {report['status']}")
    print(f"Current Metrics: {report['current_metrics']}")


if __name__ == "__main__":
    demo()
