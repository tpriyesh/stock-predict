"""
Macro Engine - Master Aggregator

Combines all macro indicators to provide sector-specific impact scores:
- Commodities (Oil, Gold, Copper, Steel, Natural Gas)
- Currencies (USD/INR, DXY)
- Bonds (US 10Y, yield curve)
- Semiconductors (SOX)

Provides unified macro signal for each sector.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import numpy as np

from .commodities import CommodityFetcher, get_commodity_fetcher
from .currencies import CurrencyFetcher, get_currency_fetcher
from .bonds import BondFetcher, get_bond_fetcher
from .semiconductor_index import SemiconductorFetcher, get_semiconductor_fetcher


@dataclass
class MacroSignal:
    """Macro signal for a specific sector."""
    sector: str
    signal: float              # -1 to +1 (positive = bullish for sector)
    confidence: float          # 0 to 1
    direction: str             # 'bullish', 'bearish', 'neutral'

    # Component breakdown
    commodity_impact: float
    currency_impact: float
    bond_impact: float
    semiconductor_impact: float

    # Key drivers
    primary_drivers: List[str]
    reasoning: List[str]

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MacroAnalysis:
    """Complete macro analysis."""
    # Overall market signal
    market_signal: float       # -1 to +1
    market_direction: str      # 'risk_on', 'risk_off', 'neutral'
    market_confidence: float

    # Sector signals
    sector_signals: Dict[str, MacroSignal]

    # Summaries
    oil_summary: Dict[str, Any]
    currency_summary: Dict[str, Any]
    bond_summary: Dict[str, Any]
    metals_summary: Dict[str, Any]

    # Key insights
    key_insights: List[str]
    risk_factors: List[str]

    timestamp: datetime = field(default_factory=datetime.now)


class MacroEngine:
    """
    Master macro engine that aggregates all indicators.

    Provides sector-specific impact scores based on:
    - Commodity prices and their sector dependencies
    - Currency movements (exporter/importer impact)
    - Bond yields (rate sensitivity)
    - Semiconductor index (tech cycle)
    """

    # Default sector weights for macro factors
    DEFAULT_WEIGHTS = {
        'commodities': 0.35,
        'currencies': 0.30,
        'bonds': 0.25,
        'semiconductors': 0.10
    }

    SECTORS = [
        'IT', 'Banking', 'Finance', 'Oil_Gas', 'Pharma',
        'Metals', 'Auto', 'FMCG', 'Infra', 'Power',
        'Telecom', 'Consumer', 'Chemicals'
    ]

    def __init__(self):
        """Initialize macro engine with all sub-fetchers."""
        self.commodities = get_commodity_fetcher()
        self.currencies = get_currency_fetcher()
        self.bonds = get_bond_fetcher()
        self.semiconductors = get_semiconductor_fetcher()

        # Load sector dependency config
        self.sector_config = self._load_sector_config()

        logger.info("MacroEngine initialized with all sub-engines")

    def _load_sector_config(self) -> Dict:
        """Load sector dependencies from config file."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'sector_dependencies.json'
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                logger.info("Loaded sector dependencies config")
                return config.get('sector_dependencies', {})
        except Exception as e:
            logger.warning(f"Could not load sector config: {e}")
        return {}

    def get_sector_macro_signal(self, sector: str) -> MacroSignal:
        """
        Get macro signal for a specific sector.

        Args:
            sector: Sector name (e.g., 'IT', 'Oil_Gas')

        Returns:
            MacroSignal with impact scores and reasoning
        """
        # Get sector config
        sector_cfg = self.sector_config.get(sector, {})
        macro_cfg = sector_cfg.get('macro', {})

        # Calculate commodity impact
        commodity_impact = self._calculate_commodity_impact(sector, macro_cfg)

        # Calculate currency impact
        currency_impact = self._calculate_currency_impact(sector, macro_cfg)

        # Calculate bond impact
        bond_impact = self._calculate_bond_impact(sector, macro_cfg)

        # Calculate semiconductor impact (mainly for IT)
        semi_impact = self._calculate_semiconductor_impact(sector)

        # Combine with sector-specific weights
        # Use social_weight from config as indicator of sensitivity
        social_weight = sector_cfg.get('social_weight', 0.10)

        # Adjust weights based on sector characteristics
        if sector in ['Oil_Gas', 'Metals', 'Chemicals']:
            weights = {'commodities': 0.50, 'currencies': 0.20, 'bonds': 0.20, 'semiconductors': 0.10}
        elif sector in ['IT', 'Pharma']:
            weights = {'commodities': 0.15, 'currencies': 0.40, 'bonds': 0.25, 'semiconductors': 0.20}
        elif sector in ['Banking', 'Finance']:
            weights = {'commodities': 0.10, 'currencies': 0.25, 'bonds': 0.55, 'semiconductors': 0.10}
        else:
            weights = self.DEFAULT_WEIGHTS

        # Calculate weighted signal
        signal = (
            weights['commodities'] * commodity_impact +
            weights['currencies'] * currency_impact +
            weights['bonds'] * bond_impact +
            weights['semiconductors'] * semi_impact
        )

        signal = np.clip(signal, -1, 1)

        # Confidence based on data availability and signal strength
        confidence = min(abs(signal) + 0.4, 0.9)

        # Direction
        if signal > 0.15:
            direction = 'bullish'
        elif signal < -0.15:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Identify primary drivers
        drivers = []
        impacts = [
            ('Commodities', commodity_impact),
            ('Currency', currency_impact),
            ('Bonds', bond_impact),
            ('Semiconductors', semi_impact)
        ]
        for name, impact in sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:2]:
            if abs(impact) > 0.1:
                direction_word = 'positive' if impact > 0 else 'negative'
                drivers.append(f"{name}: {direction_word} ({impact:+.2f})")

        # Generate reasoning
        reasoning = self._generate_reasoning(
            sector, commodity_impact, currency_impact, bond_impact, semi_impact
        )

        return MacroSignal(
            sector=sector,
            signal=signal,
            confidence=confidence,
            direction=direction,
            commodity_impact=commodity_impact,
            currency_impact=currency_impact,
            bond_impact=bond_impact,
            semiconductor_impact=semi_impact,
            primary_drivers=drivers,
            reasoning=reasoning
        )

    def _calculate_commodity_impact(self, sector: str, macro_cfg: Dict) -> float:
        """Calculate commodity impact for sector."""
        impact = 0.0
        total_weight = 0.0

        for commodity_key, cfg in macro_cfg.items():
            if commodity_key not in ['crude_brent', 'crude_wti', 'gold', 'silver',
                                     'copper', 'natural_gas', 'steel_proxy']:
                continue

            signal = self.commodities.get_commodity_signal(commodity_key)
            if not signal:
                continue

            direction = cfg.get('direction', 'positive')
            weight = cfg.get('weight', 0.20)

            if direction == 'positive':
                contribution = signal.signal * weight
            elif direction == 'negative':
                contribution = -signal.signal * weight
            else:
                contribution = 0

            impact += contribution
            total_weight += weight

        return impact if total_weight > 0 else 0.0

    def _calculate_currency_impact(self, sector: str, macro_cfg: Dict) -> float:
        """Calculate currency impact for sector."""
        impact = 0.0

        # USDINR is the primary currency
        if 'usdinr' in macro_cfg:
            cfg = macro_cfg['usdinr']
            usdinr_signal = self.currencies.get_usdinr_signal()

            if usdinr_signal:
                direction = cfg.get('direction', 'positive')
                weight = cfg.get('weight', 0.20)

                if direction == 'positive':
                    # Exporter: benefit from weak INR (positive signal)
                    impact = usdinr_signal.signal * weight
                elif direction == 'negative':
                    # Importer: hurt by weak INR
                    impact = -usdinr_signal.signal * weight

        # DXY impact
        if 'dxy' in macro_cfg:
            cfg = macro_cfg['dxy']
            dxy_signal = self.currencies.get_dxy_signal()

            if dxy_signal:
                weight = cfg.get('weight', 0.10)
                impact += dxy_signal.signal * weight

        return np.clip(impact, -1, 1)

    def _calculate_bond_impact(self, sector: str, macro_cfg: Dict) -> float:
        """Calculate bond yield impact for sector."""
        impact = 0.0

        if 'us_10y' in macro_cfg:
            cfg = macro_cfg['us_10y']
            bond_signal = self.bonds.get_10y_signal()

            if bond_signal:
                direction = cfg.get('direction', 'negative')
                weight = cfg.get('weight', 0.20)

                if direction == 'negative':
                    # Most sectors hurt by rising rates
                    impact = bond_signal.signal * weight
                elif direction == 'positive':
                    # Some benefit from rates (rare)
                    impact = -bond_signal.signal * weight

        return np.clip(impact, -1, 1)

    def _calculate_semiconductor_impact(self, sector: str) -> float:
        """Calculate semiconductor impact for sector."""
        # Mainly affects IT
        if sector not in ['IT', 'Auto']:
            return 0.0

        semi_signal = self.semiconductors.get_semiconductor_signal()
        if not semi_signal:
            return 0.0

        if sector == 'IT':
            return semi_signal.signal * 0.6
        elif sector == 'Auto':
            return semi_signal.signal * 0.3

        return 0.0

    def _generate_reasoning(
        self,
        sector: str,
        commodity: float,
        currency: float,
        bond: float,
        semi: float
    ) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []

        # Commodity reasoning
        if abs(commodity) > 0.1:
            if sector == 'Oil_Gas' and commodity > 0:
                reasoning.append("Rising oil prices support upstream profits")
            elif sector in ['Chemicals', 'FMCG'] and commodity < 0:
                reasoning.append("Higher commodity costs pressure margins")
            elif sector == 'Metals' and commodity > 0:
                reasoning.append("Strong metal prices benefit producers")

        # Currency reasoning
        if abs(currency) > 0.1:
            if sector in ['IT', 'Pharma'] and currency > 0:
                reasoning.append("Weak INR boosts export realizations")
            elif sector in ['IT', 'Pharma'] and currency < 0:
                reasoning.append("Strong INR reduces export competitiveness")
            elif sector in ['Oil_Gas', 'Auto'] and currency < 0:
                reasoning.append("Weak INR increases import costs")

        # Bond reasoning
        if abs(bond) > 0.1:
            if bond > 0:
                reasoning.append("Falling yields supportive for equity valuations")
            else:
                reasoning.append("Rising yields pressure growth stock valuations")

        # Semiconductor reasoning
        if sector == 'IT' and abs(semi) > 0.1:
            if semi > 0:
                reasoning.append("Strong semiconductor demand indicates healthy tech capex")
            else:
                reasoning.append("Weak chip demand suggests slowing tech spending")

        if not reasoning:
            reasoning.append("Macro factors neutral for this sector")

        return reasoning

    def get_full_analysis(self) -> MacroAnalysis:
        """
        Get complete macro analysis for all sectors.

        Returns:
            MacroAnalysis with all sector signals and summaries
        """
        # Get sector signals
        sector_signals = {}
        for sector in self.SECTORS:
            sector_signals[sector] = self.get_sector_macro_signal(sector)

        # Calculate overall market signal
        signals = [s.signal for s in sector_signals.values()]
        market_signal = np.mean(signals) if signals else 0.0

        if market_signal > 0.1:
            market_direction = 'risk_on'
        elif market_signal < -0.1:
            market_direction = 'risk_off'
        else:
            market_direction = 'neutral'

        market_confidence = np.mean([s.confidence for s in sector_signals.values()])

        # Get summaries
        oil_summary = self.commodities.get_oil_summary()
        currency_summary = self.currencies.get_currency_summary()
        bond_summary = self.bonds.get_bond_summary()
        metals_summary = self.commodities.get_metals_summary()

        # Generate key insights
        key_insights = self._generate_key_insights(
            sector_signals, oil_summary, currency_summary, bond_summary
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            oil_summary, currency_summary, bond_summary
        )

        return MacroAnalysis(
            market_signal=market_signal,
            market_direction=market_direction,
            market_confidence=market_confidence,
            sector_signals=sector_signals,
            oil_summary=oil_summary,
            currency_summary=currency_summary,
            bond_summary=bond_summary,
            metals_summary=metals_summary,
            key_insights=key_insights,
            risk_factors=risk_factors
        )

    def _generate_key_insights(
        self,
        sector_signals: Dict[str, MacroSignal],
        oil: Dict,
        currency: Dict,
        bond: Dict
    ) -> List[str]:
        """Generate key macro insights."""
        insights = []

        # Oil insight
        if oil.get('available'):
            if oil.get('momentum') in ['strong_up', 'up']:
                insights.append(f"Oil rising ({oil.get('change_5d', 0):+.1f}% 5D): Bullish Oil_Gas, Cautious Chemicals/Auto")
            elif oil.get('momentum') in ['strong_down', 'down']:
                insights.append(f"Oil falling ({oil.get('change_5d', 0):+.1f}% 5D): Bearish Oil_Gas, Positive for input costs")

        # Currency insight
        if currency.get('available'):
            if currency.get('inr_direction') == 'weakening':
                insights.append(f"INR weakening ({currency.get('usdinr_change_5d', 0):+.2f}% 5D): Benefits IT, Pharma exporters")
            elif currency.get('inr_direction') == 'strengthening':
                insights.append(f"INR strengthening: Benefits importers, challenges exporters")

        # Bond insight
        if bond.get('available'):
            if bond.get('us_10y_trend') == 'rising':
                insights.append(f"US 10Y rising ({bond.get('us_10y_change_5d_bps', 0):+.0f} bps): Pressure on growth stocks")
            elif bond.get('us_10y_trend') == 'falling':
                insights.append(f"US 10Y falling: Supportive for equity valuations")

        # Top sectors
        bullish = [s for s, sig in sector_signals.items() if sig.direction == 'bullish']
        bearish = [s for s, sig in sector_signals.items() if sig.direction == 'bearish']

        if bullish:
            insights.append(f"Macro bullish: {', '.join(bullish[:3])}")
        if bearish:
            insights.append(f"Macro cautious: {', '.join(bearish[:3])}")

        return insights

    def _identify_risk_factors(
        self,
        oil: Dict,
        currency: Dict,
        bond: Dict
    ) -> List[str]:
        """Identify macro risk factors."""
        risks = []

        # Oil volatility
        if oil.get('available') and abs(oil.get('change_5d', 0)) > 5:
            risks.append(f"High oil volatility ({oil.get('change_5d'):+.1f}% 5D)")

        # Currency volatility
        if currency.get('available') and abs(currency.get('usdinr_change_5d', 0)) > 1:
            risks.append(f"INR volatility ({currency.get('usdinr_change_5d'):+.2f}% 5D)")

        # Yield curve
        if bond.get('yield_curve_signal') == 'inverted':
            risks.append(f"Inverted yield curve: Recession signal")
        elif bond.get('recession_risk') == 'high':
            risks.append(f"Elevated recession risk from yield curve")

        # High yields
        if bond.get('us_10y_level') == 'high':
            risks.append(f"US 10Y at elevated levels ({bond.get('us_10y_yield'):.2f}%)")

        return risks

    def get_sector_impact(self, sector: str) -> Dict[str, Any]:
        """
        Get simple impact dict for integration with other systems.

        Args:
            sector: Sector name

        Returns:
            Dictionary with signal, confidence, and key data
        """
        signal = self.get_sector_macro_signal(sector)

        return {
            'sector': sector,
            'signal': signal.signal,
            'confidence': signal.confidence,
            'direction': signal.direction,
            'commodity_impact': signal.commodity_impact,
            'currency_impact': signal.currency_impact,
            'bond_impact': signal.bond_impact,
            'primary_drivers': signal.primary_drivers,
            'reasoning': signal.reasoning,
        }


# Singleton instance
_engine_instance: Optional[MacroEngine] = None


def get_macro_engine() -> MacroEngine:
    """Get singleton macro engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MacroEngine()
    return _engine_instance
