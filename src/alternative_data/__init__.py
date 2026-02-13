"""
Alternative Data Sources for 70%+ Accuracy

These modules provide alpha through non-traditional data sources:
- Earnings calendar and event detection
- Options flow analysis
- Institutional flow tracking (FII/DII)
- Multi-timeframe confluence
- Insider trading signals
"""

from .earnings_events import EarningsEventAnalyzer, EventSignal
from .options_flow import OptionsFlowAnalyzer, OptionsFlowSignal
from .institutional_flow import InstitutionalFlowAnalyzer, InstitutionalSignal
from .multi_timeframe import MultiTimeframeConfluence, ConfluenceSignal
from .alternative_engine import AlternativeDataEngine, AlternativeDataScore

__all__ = [
    'EarningsEventAnalyzer',
    'EventSignal',
    'OptionsFlowAnalyzer',
    'OptionsFlowSignal',
    'InstitutionalFlowAnalyzer',
    'InstitutionalSignal',
    'MultiTimeframeConfluence',
    'ConfluenceSignal',
    'AlternativeDataEngine',
    'AlternativeDataScore',
]
