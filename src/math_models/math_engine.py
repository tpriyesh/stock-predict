"""
Unified Math Models Engine

Combines all advanced mathematics models:
- Fourier Cycles
- Fractal Dimension (Hurst)
- Information Entropy
- Statistical Mechanics

Each model contributes a probability score, weighted and combined.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

from .fourier_cycles import FourierCycleModel, FourierResult
from .fractal_dimension import FractalDimensionModel, FractalResult
from .information_entropy import InformationEntropyModel, EntropyResult
from .statistical_mechanics import StatisticalMechanicsModel, StatMechResult


@dataclass
class MathScore:
    """Complete output from math models engine."""
    symbol: str
    timestamp: datetime

    # Individual model scores (0-1)
    fourier_score: float
    fractal_score: float
    entropy_score: float
    statmech_score: float

    # Composite score (0-1)
    composite_score: float

    # Derived insights
    market_character: str          # trending, random, mean_reverting
    predictability: float          # Overall predictability (0-1)
    recommended_strategy: str      # momentum, reversion, neutral, avoid

    # Individual model results
    fourier_result: FourierResult
    fractal_result: FractalResult
    entropy_result: EntropyResult
    statmech_result: StatMechResult

    # Aggregated signals and reasons
    signals: List[str]
    insights: List[str]
    warnings: List[str]


class MathEngine:
    """
    Unified engine combining all math models.

    Provides:
    1. Individual model scores
    2. Weighted composite score
    3. Strategy recommendations based on mathematical analysis
    """

    # Default weights for each model
    DEFAULT_WEIGHTS = {
        'fourier': 0.20,
        'fractal': 0.30,
        'entropy': 0.25,
        'statmech': 0.25
    }

    # Regime-specific weights
    REGIME_WEIGHTS = {
        'trending_bull': {
            'fourier': 0.15,
            'fractal': 0.35,
            'entropy': 0.25,
            'statmech': 0.25
        },
        'ranging': {
            'fourier': 0.30,
            'fractal': 0.25,
            'entropy': 0.20,
            'statmech': 0.25
        },
        'choppy': {
            'fourier': 0.15,
            'fractal': 0.25,
            'entropy': 0.30,
            'statmech': 0.30
        },
        'neutral': {
            'fourier': 0.20,
            'fractal': 0.30,
            'entropy': 0.25,
            'statmech': 0.25
        }
    }

    def __init__(self):
        self.fourier_model = FourierCycleModel()
        self.fractal_model = FractalDimensionModel()
        self.entropy_model = InformationEntropyModel()
        self.statmech_model = StatisticalMechanicsModel()

    def get_weights(self, regime: str = 'neutral') -> Dict[str, float]:
        """Get model weights based on market regime."""
        regime_lower = regime.lower().replace(' ', '_')
        return self.REGIME_WEIGHTS.get(regime_lower, self.DEFAULT_WEIGHTS)

    def determine_market_character(
        self,
        fractal: FractalResult,
        entropy: EntropyResult
    ) -> str:
        """
        Determine overall market character from multiple models.
        """
        # Primary from fractal (Hurst)
        character = fractal.market_character

        # Validate with entropy
        if character == 'trending' and entropy.predictability_index < 0.15:
            # Low predictability contradicts trending
            character = 'random'
        elif character == 'random' and entropy.predictability_index > 0.3:
            # High predictability contradicts random
            character = 'trending'

        return character

    def calculate_overall_predictability(
        self,
        fractal: FractalResult,
        entropy: EntropyResult,
        fourier: FourierResult
    ) -> float:
        """
        Calculate overall market predictability.
        """
        # Combine different predictability measures
        hurst_pred = fractal.predictability
        entropy_pred = entropy.predictability_index
        fourier_pred = 1 - fourier.spectral_entropy

        # Weighted average
        predictability = (
            0.40 * hurst_pred +
            0.35 * entropy_pred +
            0.25 * fourier_pred
        )

        return np.clip(predictability, 0, 1)

    def determine_strategy(
        self,
        fractal: FractalResult,
        statmech: StatMechResult,
        predictability: float
    ) -> str:
        """
        Determine optimal strategy based on math analysis.
        """
        # If low predictability, avoid
        if predictability < 0.15:
            return 'neutral'

        # Phase-based strategy
        if statmech.phase == 'solid':
            return 'breakout'
        elif statmech.phase == 'gas' and statmech.phase_transition_risk > 0.6:
            return 'avoid'

        # Hurst-based strategy
        if fractal.market_character == 'trending':
            return 'momentum'
        elif fractal.market_character == 'mean_reverting':
            return 'reversion'

        return 'neutral'

    def collect_insights(
        self,
        fourier: FourierResult,
        fractal: FractalResult,
        entropy: EntropyResult,
        statmech: StatMechResult
    ) -> tuple:
        """Collect insights and warnings from all models."""
        signals = []
        insights = []
        warnings = []

        # Fourier
        if 'BULLISH' in fourier.signal:
            signals.append(f'FOURIER:{fourier.signal}')
            insights.append(f'[CYCLES] {fourier.reason}')
        elif 'BEARISH' in fourier.signal:
            signals.append(f'FOURIER:{fourier.signal}')
            insights.append(f'[CYCLES] {fourier.reason}')
        if fourier.spectral_entropy > 0.8:
            warnings.append('[CYCLES] High spectral entropy - cycles unreliable')

        # Fractal
        signals.append(f'HURST:{fractal.signal}')
        insights.append(f'[HURST] {fractal.reason}')
        if fractal.confidence_in_estimate < 0.5:
            warnings.append('[HURST] Low confidence in Hurst estimate')

        # Entropy
        if 'HIGH' in entropy.signal:
            insights.append(f'[ENTROPY] {entropy.reason}')
        elif 'LOW' in entropy.signal:
            warnings.append(f'[ENTROPY] {entropy.reason}')

        # StatMech
        signals.append(f'PHASE:{statmech.phase.upper()}')
        insights.append(f'[STATMECH] {statmech.reason}')
        if statmech.phase_transition_risk > 0.6:
            warnings.append(f'[STATMECH] High phase transition risk ({statmech.phase_transition_risk:.0%})')

        return signals, insights, warnings

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str = 'neutral'
    ) -> MathScore:
        """
        Calculate comprehensive math-based prediction score.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            regime: Market regime for weight adjustment

        Returns:
            MathScore with all models' outputs and composite score
        """
        timestamp = datetime.now()

        # Run all models
        fourier_result = self.fourier_model.score(df)
        fractal_result = self.fractal_model.score(df)
        entropy_result = self.entropy_model.score(df)
        statmech_result = self.statmech_model.score(df)

        # Get individual probability scores
        scores = {
            'fourier': fourier_result.probability_score,
            'fractal': fractal_result.probability_score,
            'entropy': entropy_result.probability_score,
            'statmech': statmech_result.probability_score
        }

        # Get regime-based weights
        weights = self.get_weights(regime)

        # Calculate weighted composite score
        composite = sum(weights[k] * scores[k] for k in weights)

        # Derived metrics
        market_character = self.determine_market_character(fractal_result, entropy_result)
        predictability = self.calculate_overall_predictability(
            fractal_result, entropy_result, fourier_result
        )
        strategy = self.determine_strategy(fractal_result, statmech_result, predictability)

        # Collect insights
        signals, insights, warnings = self.collect_insights(
            fourier_result, fractal_result, entropy_result, statmech_result
        )

        return MathScore(
            symbol=symbol,
            timestamp=timestamp,
            fourier_score=scores['fourier'],
            fractal_score=scores['fractal'],
            entropy_score=scores['entropy'],
            statmech_score=scores['statmech'],
            composite_score=composite,
            market_character=market_character,
            predictability=predictability,
            recommended_strategy=strategy,
            fourier_result=fourier_result,
            fractal_result=fractal_result,
            entropy_result=entropy_result,
            statmech_result=statmech_result,
            signals=signals,
            insights=insights,
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

        Returns only composite score.
        """
        full_score = self.score(symbol, df, regime)
        return full_score.composite_score
