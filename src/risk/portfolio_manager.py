"""
PortfolioRiskManager - Correlation-Aware Portfolio Risk Management

CRITICAL: Individual position sizing ignores portfolio-level risk.

Problems:
1. Adding correlated positions doesn't diversify risk
2. Sector concentration can blow up in sector crashes
3. Individual Kelly sizing ignores existing exposure
4. No limit on total portfolio risk

This module provides:
- Correlation-based position size adjustment
- Sector concentration limits
- Total portfolio exposure caps
- Drawdown-based risk reduction
- Value at Risk (VaR) monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class Position:
    """Current position in a stock."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    sector: str
    entry_date: datetime
    beta: float = 1.0

    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_return(self) -> float:
        """Unrealized return percentage."""
        return (self.current_price / self.entry_price - 1) if self.entry_price > 0 else 0


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_allocation: float  # As percentage of capital
    recommended_quantity: int
    recommended_value: float

    # Adjustments applied
    base_allocation: float
    correlation_penalty: float
    sector_penalty: float
    exposure_penalty: float
    drawdown_penalty: float

    # Reasons
    adjustments_applied: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        return {
            'recommended_allocation': self.recommended_allocation,
            'recommended_quantity': self.recommended_quantity,
            'recommended_value': self.recommended_value,
            'base_allocation': self.base_allocation,
            'adjustments': {
                'correlation': self.correlation_penalty,
                'sector': self.sector_penalty,
                'exposure': self.exposure_penalty,
                'drawdown': self.drawdown_penalty
            },
            'adjustments_applied': self.adjustments_applied,
            'warnings': self.warnings
        }


class PortfolioRiskManager:
    """
    Portfolio-level risk management for position sizing.

    Adjusts individual position sizes based on:
    1. Correlation with existing positions
    2. Sector concentration
    3. Total portfolio exposure
    4. Current drawdown level

    Usage:
        manager = PortfolioRiskManager(capital=1_000_000)

        # Add existing positions
        manager.add_position(Position(
            symbol="RELIANCE", quantity=100, entry_price=2500,
            current_price=2650, sector="Energy", entry_date=datetime.now()
        ))

        # Get adjusted position size for new trade
        result = manager.calculate_position_size(
            symbol="TATAMOTORS",
            base_allocation=0.10,
            price=500,
            sector="Auto",
            returns_data=returns_matrix
        )

        print(f"Recommended: {result.recommended_allocation:.1%}")
    """

    # Configuration
    MAX_POSITION_SIZE = 0.15          # Max 15% in single stock
    MAX_SECTOR_EXPOSURE = 0.30        # Max 30% in single sector
    MAX_PORTFOLIO_EXPOSURE = 0.90     # Max 90% invested
    MIN_POSITION_SIZE = 0.02          # Min 2% position size

    # Correlation penalty thresholds
    HIGH_CORRELATION = 0.7
    MEDIUM_CORRELATION = 0.4

    # Drawdown adjustments
    DRAWDOWN_WARNING = 0.05           # 5% drawdown
    DRAWDOWN_CRITICAL = 0.10          # 10% drawdown
    DRAWDOWN_EMERGENCY = 0.15         # 15% drawdown

    def __init__(self,
                 capital: float = 1_000_000,
                 max_position: float = 0.15,
                 max_sector: float = 0.30,
                 max_exposure: float = 0.90):
        """
        Initialize portfolio risk manager.

        Args:
            capital: Total portfolio capital
            max_position: Maximum single position size (0-1)
            max_sector: Maximum sector exposure (0-1)
            max_exposure: Maximum total exposure (0-1)
        """
        self.capital = capital
        self.max_position = max_position
        self.max_sector = max_sector
        self.max_exposure = max_exposure

        # Current positions
        self.positions: Dict[str, Position] = {}

        # Portfolio state
        self.peak_value = capital
        self.current_drawdown = 0.0

        # Sector mapping (for common Indian stocks)
        self.sector_map = self._initialize_sector_map()

    def _initialize_sector_map(self) -> Dict[str, str]:
        """Initialize sector mapping for common stocks."""
        return {
            # IT
            'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
            'LTIM': 'IT', 'MPHASIS': 'IT', 'PERSISTENT': 'IT',

            # Banking
            'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'KOTAKBANK': 'Banking',
            'AXISBANK': 'Banking', 'SBIN': 'Banking', 'INDUSINDBK': 'Banking',
            'BANKBARODA': 'Banking', 'PNB': 'Banking', 'IDFCFIRSTB': 'Banking',

            # Financial Services
            'BAJFINANCE': 'Finance', 'BAJAJFINSV': 'Finance', 'HDFC': 'Finance',
            'SBILIFE': 'Finance', 'HDFCLIFE': 'Finance', 'ICICIPRULI': 'Finance',

            # Auto
            'TATAMOTORS': 'Auto', 'MARUTI': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto',
            'HEROMOTOCO': 'Auto', 'EICHERMOT': 'Auto', 'ASHOKLEY': 'Auto',

            # Energy
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 'BPCL': 'Energy',
            'NTPC': 'Energy', 'POWERGRID': 'Energy', 'ADANIGREEN': 'Energy',
            'TATAPOWER': 'Energy', 'ADANIENT': 'Energy',

            # Metals
            'TATASTEEL': 'Metals', 'HINDALCO': 'Metals', 'JSWSTEEL': 'Metals',
            'VEDL': 'Metals', 'COAL': 'Metals', 'NMDC': 'Metals',

            # Pharma
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma',
            'DIVISLAB': 'Pharma', 'APOLLOHOSP': 'Pharma', 'BIOCON': 'Pharma',

            # FMCG
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
            'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
            'COLPAL': 'FMCG', 'GODREJCP': 'FMCG', 'TATACONSUM': 'FMCG',

            # Cement
            'ULTRACEMCO': 'Cement', 'SHREECEM': 'Cement', 'AMBUJACEM': 'Cement',
            'ACC': 'Cement', 'DALMIACW': 'Cement',

            # Telecom
            'BHARTIARTL': 'Telecom', 'IDEA': 'Telecom',
        }

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_map.get(symbol.upper(), 'Other')

    def add_position(self, position: Position):
        """Add or update a position."""
        self.positions[position.symbol] = position
        self._update_portfolio_state()

    def remove_position(self, symbol: str):
        """Remove a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_portfolio_state()

    def _update_portfolio_state(self):
        """Update portfolio state after position changes."""
        portfolio_value = self.get_portfolio_value()

        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value

    def get_portfolio_value(self) -> float:
        """Get current portfolio value (cash + positions)."""
        invested = sum(p.market_value for p in self.positions.values())
        cash = self.capital - sum(p.cost_basis for p in self.positions.values())
        return invested + max(0, cash)

    def get_current_exposure(self) -> float:
        """Get current exposure as percentage of capital."""
        invested = sum(p.market_value for p in self.positions.values())
        return invested / self.capital if self.capital > 0 else 0

    def get_sector_exposure(self) -> Dict[str, float]:
        """Get exposure by sector."""
        sector_values = defaultdict(float)

        for position in self.positions.values():
            sector = position.sector or self.get_sector(position.symbol)
            sector_values[sector] += position.market_value

        return {
            sector: value / self.capital
            for sector, value in sector_values.items()
        }

    def calculate_correlation_penalty(self,
                                        symbol: str,
                                        returns_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate correlation penalty for new position.

        High correlation with existing positions = higher penalty.

        Args:
            symbol: New symbol to add
            returns_data: DataFrame with returns for all symbols

        Returns:
            Penalty multiplier (0.5 to 1.0)
        """
        if not self.positions or returns_data is None:
            return 1.0  # No penalty if no positions

        if symbol not in returns_data.columns:
            return 1.0

        correlations = []
        for existing_symbol in self.positions:
            if existing_symbol in returns_data.columns:
                corr = returns_data[symbol].corr(returns_data[existing_symbol])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return 1.0

        avg_corr = np.mean(correlations)
        max_corr = max(correlations)

        # Penalty based on average and max correlation
        if max_corr > self.HIGH_CORRELATION:
            # High correlation with at least one position
            penalty = 0.5 + (1 - max_corr) * 0.5
        elif avg_corr > self.MEDIUM_CORRELATION:
            # Medium average correlation
            penalty = 0.7 + (1 - avg_corr) * 0.3
        else:
            # Low correlation - good diversification
            penalty = 1.0

        return penalty

    def calculate_sector_penalty(self, sector: str) -> float:
        """
        Calculate penalty based on sector concentration.

        Args:
            sector: Sector of new position

        Returns:
            Penalty multiplier (0.3 to 1.0)
        """
        sector_exposure = self.get_sector_exposure()
        current_sector_exposure = sector_exposure.get(sector, 0)

        if current_sector_exposure >= self.max_sector:
            # Already at max - severe penalty
            return 0.3
        elif current_sector_exposure >= self.max_sector * 0.75:
            # Approaching max
            remaining = (self.max_sector - current_sector_exposure) / self.max_sector
            return 0.5 + remaining * 0.5
        else:
            # Plenty of room
            return 1.0

    def calculate_exposure_penalty(self) -> float:
        """
        Calculate penalty based on total portfolio exposure.

        Returns:
            Penalty multiplier (0.3 to 1.0)
        """
        current_exposure = self.get_current_exposure()

        if current_exposure >= self.max_exposure:
            # Already at max
            return 0.3
        elif current_exposure >= self.max_exposure * 0.8:
            # Approaching max
            remaining = (self.max_exposure - current_exposure) / self.max_exposure
            return 0.5 + remaining * 0.5
        else:
            return 1.0

    def calculate_drawdown_penalty(self) -> float:
        """
        Calculate penalty based on current drawdown.

        When in drawdown, reduce position sizes to protect capital.

        Returns:
            Penalty multiplier (0.3 to 1.0)
        """
        if self.current_drawdown >= self.DRAWDOWN_EMERGENCY:
            # Emergency: minimal new positions
            return 0.3
        elif self.current_drawdown >= self.DRAWDOWN_CRITICAL:
            # Critical: significantly reduced
            return 0.5
        elif self.current_drawdown >= self.DRAWDOWN_WARNING:
            # Warning: somewhat reduced
            return 0.75
        else:
            return 1.0

    def calculate_position_size(self,
                                 symbol: str,
                                 base_allocation: float,
                                 price: float,
                                 sector: Optional[str] = None,
                                 returns_data: Optional[pd.DataFrame] = None) -> PositionSizeResult:
        """
        Calculate adjusted position size considering all risk factors.

        Args:
            symbol: Stock symbol
            base_allocation: Base allocation from Kelly or other method (0-1)
            price: Current price per share
            sector: Sector (if known)
            returns_data: Historical returns for correlation calculation

        Returns:
            PositionSizeResult with adjusted allocation
        """
        sector = sector or self.get_sector(symbol)
        adjustments = []
        warnings = []

        # Start with base allocation
        allocation = base_allocation

        # 1. Cap at maximum position size
        if allocation > self.max_position:
            allocation = self.max_position
            adjustments.append(f"Capped at max position size ({self.max_position:.0%})")

        # 2. Correlation penalty
        corr_penalty = self.calculate_correlation_penalty(symbol, returns_data)
        if corr_penalty < 1.0:
            allocation *= corr_penalty
            adjustments.append(f"Correlation penalty: {corr_penalty:.2f}")

        # 3. Sector concentration penalty
        sector_penalty = self.calculate_sector_penalty(sector)
        if sector_penalty < 1.0:
            allocation *= sector_penalty
            adjustments.append(f"Sector penalty: {sector_penalty:.2f}")
            if sector_penalty < 0.5:
                warnings.append(f"High {sector} sector concentration")

        # 4. Total exposure penalty
        exposure_penalty = self.calculate_exposure_penalty()
        if exposure_penalty < 1.0:
            allocation *= exposure_penalty
            adjustments.append(f"Exposure penalty: {exposure_penalty:.2f}")
            if exposure_penalty < 0.5:
                warnings.append("Portfolio near maximum exposure")

        # 5. Drawdown penalty
        dd_penalty = self.calculate_drawdown_penalty()
        if dd_penalty < 1.0:
            allocation *= dd_penalty
            adjustments.append(f"Drawdown penalty: {dd_penalty:.2f}")
            warnings.append(f"Portfolio in {self.current_drawdown:.1%} drawdown")

        # 6. Floor at minimum position size
        if allocation < self.MIN_POSITION_SIZE:
            if allocation > 0:
                warnings.append("Position too small after adjustments")
            allocation = 0  # Don't take tiny positions

        # Calculate quantity
        position_value = self.capital * allocation
        quantity = int(position_value / price) if price > 0 else 0

        return PositionSizeResult(
            recommended_allocation=allocation,
            recommended_quantity=quantity,
            recommended_value=quantity * price,
            base_allocation=base_allocation,
            correlation_penalty=corr_penalty,
            sector_penalty=sector_penalty,
            exposure_penalty=exposure_penalty,
            drawdown_penalty=dd_penalty,
            adjustments_applied=adjustments,
            warnings=warnings
        )

    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        return {
            'capital': self.capital,
            'portfolio_value': self.get_portfolio_value(),
            'current_exposure': self.get_current_exposure(),
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'n_positions': len(self.positions),
            'sector_exposure': self.get_sector_exposure(),
            'positions': {
                symbol: {
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'unrealized_pnl': p.unrealized_pnl,
                    'sector': p.sector
                }
                for symbol, p in self.positions.items()
            }
        }


def demo():
    """Demonstrate portfolio risk management."""
    print("=" * 60)
    print("PortfolioRiskManager Demo")
    print("=" * 60)

    manager = PortfolioRiskManager(capital=1_000_000)

    # Add some existing positions
    print("\n--- Adding Existing Positions ---")

    manager.add_position(Position(
        symbol="RELIANCE",
        quantity=100,
        entry_price=2500,
        current_price=2650,
        sector="Energy",
        entry_date=datetime.now()
    ))

    manager.add_position(Position(
        symbol="TCS",
        quantity=50,
        entry_price=3500,
        current_price=3600,
        sector="IT",
        entry_date=datetime.now()
    ))

    manager.add_position(Position(
        symbol="HDFCBANK",
        quantity=80,
        entry_price=1600,
        current_price=1650,
        sector="Banking",
        entry_date=datetime.now()
    ))

    print("Positions added:")
    for symbol, pos in manager.positions.items():
        print(f"  {symbol}: {pos.quantity} @ {pos.current_price}, "
              f"sector={pos.sector}, value={pos.market_value:,.0f}")

    # Portfolio summary
    summary = manager.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Capital: Rs {summary['capital']:,.0f}")
    print(f"  Current Value: Rs {summary['portfolio_value']:,.0f}")
    print(f"  Exposure: {summary['current_exposure']:.1%}")
    print(f"  Drawdown: {summary['current_drawdown']:.1%}")

    print(f"\nSector Exposure:")
    for sector, exp in summary['sector_exposure'].items():
        print(f"  {sector}: {exp:.1%}")

    # Calculate position size for new stock
    print("\n--- Calculating Position Size for INFY ---")

    result = manager.calculate_position_size(
        symbol="INFY",
        base_allocation=0.12,  # Kelly suggested 12%
        price=1500,
        sector="IT"
    )

    print(f"Base Allocation: {result.base_allocation:.1%}")
    print(f"Recommended Allocation: {result.recommended_allocation:.1%}")
    print(f"Recommended Quantity: {result.recommended_quantity}")
    print(f"Recommended Value: Rs {result.recommended_value:,.0f}")

    print("\nAdjustments Applied:")
    for adj in result.adjustments_applied:
        print(f"  - {adj}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ! {warning}")

    # Test with high correlation scenario
    print("\n--- Position Size for Another Energy Stock (ONGC) ---")

    result2 = manager.calculate_position_size(
        symbol="ONGC",
        base_allocation=0.10,
        price=180,
        sector="Energy"  # Same sector as RELIANCE
    )

    print(f"Base: {result2.base_allocation:.1%} -> Recommended: {result2.recommended_allocation:.1%}")
    print(f"Sector Penalty: {result2.sector_penalty:.2f}")
    print(f"Adjustments: {', '.join(result2.adjustments_applied)}")


if __name__ == "__main__":
    demo()
