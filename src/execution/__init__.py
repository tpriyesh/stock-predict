"""
Execution module for advanced trade management.

Provides:
- Trailing stops (ATR-based)
- Partial profit booking
- Gap handling
- Re-entry rules
"""

from .trade_executor import (
    ExecutionConfig,
    PartialPosition,
    ExecutionEvent,
    AdvancedTradeExecutor,
)
from .trailing_stop import TrailingStopCalculator
from .gap_handler import GapHandler, GapType
from .reentry_manager import ReentryManager, ReentryRule

__all__ = [
    "ExecutionConfig",
    "PartialPosition",
    "ExecutionEvent",
    "AdvancedTradeExecutor",
    "TrailingStopCalculator",
    "GapHandler",
    "GapType",
    "ReentryManager",
    "ReentryRule",
]
