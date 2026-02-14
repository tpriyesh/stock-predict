"""
Market-wide features for context and regime detection.
"""
import os
from datetime import datetime
from typing import Optional
import json
from pathlib import Path
from loguru import logger

from src.data.market_indicators import MarketIndicators
from utils.platform import now_ist


# How long to cache the full market context (default 1 hour)
_CONTEXT_CACHE_TTL = int(os.getenv("MARKET_CONTEXT_REFRESH_SECONDS", "3600"))


class MarketFeatures:
    """
    Compute market-wide features for trading context.
    """

    def __init__(self):
        self.indicators = MarketIndicators()
        self._cached_context: Optional[dict] = None
        self._context_cache_time: Optional[datetime] = None
        self._load_sector_mapping()

    def _load_sector_mapping(self):
        """Load sector mapping from symbols.json."""
        try:
            symbols_path = Path(__file__).parent.parent.parent / "config" / "symbols.json"
            with open(symbols_path) as f:
                data = json.load(f)
            self.sector_mapping = {}
            for sector, symbols in data.get('sectors', {}).items():
                for symbol in symbols:
                    self.sector_mapping[symbol] = sector
        except Exception as e:
            logger.warning(f"Could not load sector mapping: {e}")
            self.sector_mapping = {}

    def get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_mapping.get(symbol, 'Unknown')

    def get_market_context(self) -> dict:
        """
        Get comprehensive market context for trading decisions.

        Cached for 1 hour — market regime, VIX, and sector strengths
        change slowly and don't need per-cycle refresh.

        Returns:
            Dict with market regime, VIX, sector strengths, etc.
        """
        # Return cached context if fresh
        now = now_ist().replace(tzinfo=None)
        if (self._cached_context is not None
                and self._context_cache_time is not None
                and (now - self._context_cache_time).total_seconds() < _CONTEXT_CACHE_TTL):
            logger.debug("Using cached market context")
            return self._cached_context

        context = {
            'timestamp': None,
            'regime': None,
            'vix': None,
            'nifty_levels': None,
            'sector_strengths': None,
        }

        try:
            # Get market regime
            regime = self.indicators.get_market_regime()
            context['regime'] = regime
            context['timestamp'] = regime.get('timestamp')

            # Get NIFTY levels
            nifty_levels = self.indicators.get_nifty_levels()
            context['nifty_levels'] = nifty_levels

            # Get all sector strengths
            sector_strengths = self.indicators.get_all_sector_strengths()
            context['sector_strengths'] = sector_strengths

        except Exception as e:
            logger.error(f"Failed to get market context: {e}")

        self._cached_context = context
        self._context_cache_time = now
        return context

    def get_sector_score(self, symbol: str, market_context: Optional[dict] = None) -> float:
        """
        Get sector strength score for a symbol.

        Args:
            symbol: Stock symbol
            market_context: Optional pre-fetched market context

        Returns:
            Score from 0.0 to 1.0
        """
        sector = self.get_symbol_sector(symbol)
        if sector == 'Unknown':
            return 0.5

        if market_context and 'sector_strengths' in market_context:
            for s in market_context['sector_strengths']:
                if s['sector'] == sector:
                    return s.get('score', 0.5)

        # Fetch directly if not in context
        strength = self.indicators.get_sector_strength(sector)
        return strength.get('score', 0.5)

    def get_sector_strength(self, sector: str) -> dict:
        """
        Get detailed sector strength metrics.

        Args:
            sector: Sector name (e.g., 'IT', 'Banking')

        Returns:
            Dict with strength metrics
        """
        return self.indicators.get_sector_strength(sector)

    def get_regime_adjustment(self, market_context: Optional[dict] = None) -> float:
        """
        Get regime-based confidence adjustment.

        In RISK_OFF regime, lower confidence in long signals.
        In RISK_ON regime, boost confidence slightly.

        Returns:
            Multiplier (0.7 to 1.2)
        """
        if not market_context:
            market_context = self.get_market_context()

        regime = market_context.get('regime', {}).get('overall', 'NEUTRAL')

        adjustments = {
            'RISK_ON': 1.1,
            'NEUTRAL': 1.0,
            'RISK_OFF': 0.8
        }

        return adjustments.get(regime, 1.0)

    def get_vix_signal(self, market_context: Optional[dict] = None) -> dict:
        """
        Get VIX-based signal interpretation.

        Returns:
            Dict with VIX level and trading implications
        """
        if not market_context:
            market_context = self.get_market_context()

        vix = market_context.get('regime', {}).get('vix_level')
        vix_regime = market_context.get('regime', {}).get('vix_regime', 'NORMAL')

        signal = {
            'vix_level': vix,
            'regime': vix_regime,
            'implication': 'NEUTRAL',
            'description': ''
        }

        if vix_regime == 'LOW_FEAR':
            signal['implication'] = 'CAUTION'
            signal['description'] = 'Low VIX suggests complacency - potential for volatility spike'
        elif vix_regime == 'HIGH_FEAR':
            signal['implication'] = 'OPPORTUNITY'
            signal['description'] = 'High VIX often marks panic bottoms - contrarian buy zone'
        else:
            signal['implication'] = 'NORMAL'
            signal['description'] = 'VIX in normal range - follow technical signals'

        return signal

    def format_market_summary(self, market_context: Optional[dict] = None) -> str:
        """
        Format market context as readable summary.

        Returns:
            Multi-line summary string
        """
        if not market_context:
            market_context = self.get_market_context()

        lines = []

        # Regime
        regime = market_context.get('regime', {})
        lines.append(f"Market Regime: {regime.get('overall', 'UNKNOWN')}")
        lines.append(f"NIFTY Trend: {regime.get('nifty_trend', 'UNKNOWN')}")

        # VIX
        vix = regime.get('vix_level')
        if vix:
            lines.append(f"India VIX: {vix:.2f} ({regime.get('vix_regime', 'NORMAL')})")

        # NIFTY levels
        nifty = market_context.get('nifty_levels', {})
        if nifty:
            lines.append(f"NIFTY: {nifty.get('current', 'N/A')} "
                        f"(Range: {nifty.get('day_low', 'N/A')}-{nifty.get('day_high', 'N/A')})")

        # Top sectors
        if regime.get('sector_leaders'):
            lines.append(f"Leading Sectors: {', '.join(regime['sector_leaders'])}")
        if regime.get('sector_laggards'):
            lines.append(f"Lagging Sectors: {', '.join(regime['sector_laggards'])}")

        return "\n".join(lines)
