"""
Risk Management Module - Portfolio-Level Risk Controls

Provides:
- PortfolioRiskManager: Correlation-aware position sizing
- ExposureManager: Sector and total exposure limits
- DrawdownProtection: Maximum drawdown controls
"""

from .portfolio_manager import PortfolioRiskManager, Position, PositionSizeResult

__all__ = [
    'PortfolioRiskManager',
    'Position',
    'PositionSizeResult',
]
