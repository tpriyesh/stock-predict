"""
Monitoring Module - Production-Grade Prediction Monitoring

Provides:
- PredictionMonitor: Track live prediction accuracy and calibration
- AlertManager: Generate alerts for performance degradation
- KillSwitch: Automatically disable trading when performance drops
"""

from .monitor import PredictionMonitor, MonitoringAlert, PredictionRecord
from .kill_switch import KillSwitch, TradingStatus

__all__ = [
    'PredictionMonitor',
    'MonitoringAlert',
    'PredictionRecord',
    'KillSwitch',
    'TradingStatus',
]
