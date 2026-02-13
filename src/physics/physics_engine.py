"""
Unified Physics Engine

Combines all physics-inspired models into a single scoring engine:
- Momentum Conservation
- Spring Reversion
- Energy Clustering
- Network Propagation

Each model contributes a probability score, weighted and combined.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from .momentum_conservation import MomentumConservationModel, MomentumResult
from .spring_reversion import SpringReversionModel, SpringResult
from .energy_clustering import EnergyClusteringModel, EnergyResult
from .network_propagation import NetworkPropagationModel, NetworkResult


@dataclass
class PhysicsScore:
    """Complete output from physics engine."""
    symbol: str
    timestamp: datetime

    # Individual model scores (0-1)
    momentum_score: float
    spring_score: float
    energy_score: float
    network_score: float

    # Composite score (0-1)
    composite_score: float

    # Strategy recommendation
    recommended_strategy: str  # 'momentum', 'reversion', 'breakout', 'avoid'
    confidence: float

    # Individual model results
    momentum_result: MomentumResult
    spring_result: SpringResult
    energy_result: EnergyResult
    network_result: NetworkResult

    # Aggregated signals and reasons
    signals: List[str]
    bullish_reasons: List[str]
    bearish_reasons: List[str]
    warnings: List[str]


class PhysicsEngine:
    """
    Unified engine combining all physics models.

    Weights can be adjusted based on market regime or backtesting results.
    """

    # Default weights for each model
    DEFAULT_WEIGHTS = {
        'momentum': 0.30,
        'spring': 0.25,
        'energy': 0.25,
        'network': 0.20
    }

    # Regime-specific weights
    REGIME_WEIGHTS = {
        'trending_bull': {
            'momentum': 0.40,
            'spring': 0.15,
            'energy': 0.25,
            'network': 0.20
        },
        'trending_bear': {
            'momentum': 0.35,
            'spring': 0.20,
            'energy': 0.25,
            'network': 0.20
        },
        'ranging': {
            'momentum': 0.15,
            'spring': 0.40,
            'energy': 0.25,
            'network': 0.20
        },
        'choppy': {
            'momentum': 0.20,
            'spring': 0.25,
            'energy': 0.35,
            'network': 0.20
        },
        'neutral': {
            'momentum': 0.30,
            'spring': 0.25,
            'energy': 0.25,
            'network': 0.20
        }
    }

    def __init__(self):
        self.momentum_model = MomentumConservationModel()
        self.spring_model = SpringReversionModel()
        self.energy_model = EnergyClusteringModel()
        self.network_model = NetworkPropagationModel()

    def get_weights(self, regime: str = 'neutral') -> Dict[str, float]:
        """Get model weights based on market regime."""
        regime_lower = regime.lower().replace(' ', '_')
        return self.REGIME_WEIGHTS.get(regime_lower, self.DEFAULT_WEIGHTS)

    def determine_strategy(
        self,
        momentum: MomentumResult,
        spring: SpringResult,
        energy: EnergyResult,
        network: NetworkResult
    ) -> tuple:
        """
        Determine optimal trading strategy based on model consensus.

        Returns (strategy, confidence)
        """
        votes = {
            'momentum': 0,
            'reversion': 0,
            'breakout': 0,
            'avoid': 0
        }

        # Momentum model vote
        if momentum.persistence_score > 0.55 and momentum.friction_level < 1.2:
            votes['momentum'] += 1.0
        elif momentum.signal == 'REVERSAL_WARNING':
            votes['reversion'] += 0.5

        # Spring model vote
        if abs(spring.displacement) > 0.04 and spring.spring_constant > 0.5:
            votes['reversion'] += 1.0
        elif abs(spring.displacement) < 0.02:
            votes['momentum'] += 0.3  # Near equilibrium, follow trend

        # Energy model vote
        if energy.phase == 'solid':
            votes['breakout'] += 1.0
        elif energy.phase == 'gas' and energy.energy_regime == 'extreme':
            votes['avoid'] += 1.0
        elif energy.phase == 'liquid':
            # Normal conditions, could go either way
            votes['momentum'] += 0.3
            votes['reversion'] += 0.3

        # Network model vote
        if network.interference_type == 'constructive':
            if network.network_momentum > 0:
                votes['momentum'] += 0.5
            else:
                votes['reversion'] += 0.5
        elif network.interference_type == 'destructive':
            votes['avoid'] += 0.5

        # Find winner
        best_strategy = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[best_strategy] / total_votes if total_votes > 0 else 0.5

        return best_strategy, confidence

    def collect_reasons(
        self,
        momentum: MomentumResult,
        spring: SpringResult,
        energy: EnergyResult,
        network: NetworkResult
    ) -> tuple:
        """Collect bullish reasons, bearish reasons, and warnings."""
        bullish = []
        bearish = []
        warnings = []

        # Momentum
        if momentum.momentum_direction == 1 and momentum.persistence_score > 0.55:
            bullish.append(f'[MOMENTUM] {momentum.reason}')
        elif momentum.momentum_direction == -1 and momentum.persistence_score > 0.55:
            bearish.append(f'[MOMENTUM] {momentum.reason}')
        if momentum.signal == 'REVERSAL_WARNING':
            warnings.append(f'[MOMENTUM] {momentum.reason}')

        # Spring
        if spring.direction == 1 and spring.reversion_probability > 0.6:
            bullish.append(f'[SPRING] {spring.reason}')
        elif spring.direction == -1 and spring.reversion_probability > 0.6:
            bearish.append(f'[SPRING] {spring.reason}')

        # Energy
        if energy.phase == 'solid':
            warnings.append(f'[ENERGY] {energy.reason}')
        elif energy.phase == 'gas':
            warnings.append(f'[ENERGY] {energy.reason}')
        elif not energy.tradeable:
            warnings.append(f'[ENERGY] Conditions not ideal for trading')

        # Network
        if 'BULLISH' in network.signal:
            bullish.append(f'[NETWORK] {network.reason}')
        elif 'BEARISH' in network.signal:
            bearish.append(f'[NETWORK] {network.reason}')
        if network.interference_type == 'destructive':
            warnings.append(f'[NETWORK] Sector signals conflicting')

        return bullish, bearish, warnings

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str = 'neutral',
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        fundamental_value: Optional[float] = None
    ) -> PhysicsScore:
        """
        Calculate comprehensive physics-based prediction score.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data (minimum 60 days recommended)
            regime: Market regime for weight adjustment
            sector_data: Optional dict of peer stock DataFrames for network analysis
            fundamental_value: Optional fair value for spring model

        Returns:
            PhysicsScore with all models' outputs and composite score
        """
        timestamp = datetime.now()

        # Run all models
        momentum_result = self.momentum_model.score(df)
        spring_result = self.spring_model.score(df, regime, fundamental_value)
        energy_result = self.energy_model.score(df)
        network_result = self.network_model.score(symbol, df, sector_data)

        # Get individual probability scores
        scores = {
            'momentum': momentum_result.probability_score,
            'spring': spring_result.probability_score,
            'energy': energy_result.probability_score,
            'network': network_result.probability_score
        }

        # Get regime-based weights
        weights = self.get_weights(regime)

        # Calculate weighted composite score
        composite = sum(weights[k] * scores[k] for k in weights)

        # Determine strategy
        strategy, strategy_confidence = self.determine_strategy(
            momentum_result, spring_result, energy_result, network_result
        )

        # Collect reasons
        bullish, bearish, warnings = self.collect_reasons(
            momentum_result, spring_result, energy_result, network_result
        )

        # Collect all active signals
        signals = []
        if momentum_result.signal not in ['WEAK', 'NEUTRAL']:
            signals.append(f'MOMENTUM:{momentum_result.signal}')
        if spring_result.signal not in ['NEUTRAL', 'NEAR_EQUILIBRIUM']:
            signals.append(f'SPRING:{spring_result.signal}')
        if energy_result.signal not in ['NEUTRAL', 'NORMAL']:
            signals.append(f'ENERGY:{energy_result.signal}')
        if network_result.signal not in ['NEUTRAL_NETWORK', 'NO_NETWORK', 'LIMITED_DATA']:
            signals.append(f'NETWORK:{network_result.signal}')

        return PhysicsScore(
            symbol=symbol,
            timestamp=timestamp,
            momentum_score=scores['momentum'],
            spring_score=scores['spring'],
            energy_score=scores['energy'],
            network_score=scores['network'],
            composite_score=composite,
            recommended_strategy=strategy,
            confidence=strategy_confidence,
            momentum_result=momentum_result,
            spring_result=spring_result,
            energy_result=energy_result,
            network_result=network_result,
            signals=signals,
            bullish_reasons=bullish,
            bearish_reasons=bearish,
            warnings=warnings
        )

    def score_quick(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str = 'neutral'
    ) -> float:
        """
        Quick scoring for bulk operations.

        Returns only composite score without full result objects.
        """
        full_score = self.score(symbol, df, regime)
        return full_score.composite_score
