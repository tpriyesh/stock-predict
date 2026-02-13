"""
Global Macro Indicators Module

Tracks global macroeconomic factors that influence Indian stock markets:
- Commodities: Oil (Brent/WTI), Gold, Silver, Copper, Natural Gas, Steel
- Currencies: USD/INR, DXY, EUR/USD, GBP/USD
- Bonds: US Treasury yields (2Y, 10Y, 30Y)
- Indices: Semiconductor (SOX), VIX, S&P 500, NASDAQ

All data sourced from Yahoo Finance for free, real-time access.
"""

from .commodities import CommodityFetcher, CommodityData, CommoditySignal
from .currencies import CurrencyFetcher, CurrencyData, CurrencySignal
from .bonds import BondFetcher, BondData, BondSignal
from .semiconductor_index import SemiconductorFetcher, SemiconductorData
from .macro_engine import (
    MacroEngine,
    MacroAnalysis,
    MacroSignal,
    get_macro_engine
)

__all__ = [
    # Commodities
    'CommodityFetcher',
    'CommodityData',
    'CommoditySignal',
    # Currencies
    'CurrencyFetcher',
    'CurrencyData',
    'CurrencySignal',
    # Bonds
    'BondFetcher',
    'BondData',
    'BondSignal',
    # Semiconductor
    'SemiconductorFetcher',
    'SemiconductorData',
    # Engine
    'MacroEngine',
    'MacroAnalysis',
    'MacroSignal',
    'get_macro_engine',
]
