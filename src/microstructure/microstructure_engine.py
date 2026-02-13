"""
Unified Microstructure Engine

Combines all microstructure components:
- Volume Profile
- Smart Money Detection
- Intraday Patterns
- Enhanced Sentiment

Provides single interface for microstructure-based predictions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .volume_profile import VolumeProfileAnalyzer, VolumeProfileResult
from .smart_money import SmartMoneyDetector, SmartMoneyResult
from .intraday_patterns import IntradayPatternAnalyzer, IntradayResult
from .enhanced_sentiment import EnhancedSentimentAnalyzer, SentimentResult


@dataclass
class MicrostructureScore:
    """Complete output from microstructure engine."""
    symbol: str
    timestamp: datetime

    # Individual scores (0-1)
    volume_profile_score: float
    smart_money_score: float
    intraday_score: float
    sentiment_score: float

    # Composite score
    composite_score: float

    # Key metrics
    poc_price: float
    accumulation_distribution: str
    gap_status: str
    sentiment_label: str

    # Individual results
    volume_profile_result: VolumeProfileResult
    smart_money_result: SmartMoneyResult
    intraday_result: IntradayResult
    sentiment_result: SentimentResult

    # Aggregated signals
    signals: List[str]
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]


class MicrostructureEngine:
    """
    Unified engine combining all microstructure models.

    Weights can be adjusted based on timeframe:
    - Intraday: Higher weight on volume profile and intraday patterns
    - Swing: Higher weight on smart money and sentiment
    """

    # Default weights
    DEFAULT_WEIGHTS = {
        'volume_profile': 0.25,
        'smart_money': 0.30,
        'intraday': 0.20,
        'sentiment': 0.25
    }

    # Timeframe-specific weights
    TIMEFRAME_WEIGHTS = {
        'intraday': {
            'volume_profile': 0.30,
            'smart_money': 0.20,
            'intraday': 0.30,
            'sentiment': 0.20
        },
        'swing': {
            'volume_profile': 0.20,
            'smart_money': 0.35,
            'intraday': 0.15,
            'sentiment': 0.30
        },
        'positional': {
            'volume_profile': 0.20,
            'smart_money': 0.40,
            'intraday': 0.10,
            'sentiment': 0.30
        }
    }

    def __init__(self):
        self.volume_profile_analyzer = VolumeProfileAnalyzer()
        self.smart_money_detector = SmartMoneyDetector()
        self.intraday_analyzer = IntradayPatternAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()

    def get_weights(self, timeframe: str = 'swing') -> Dict[str, float]:
        """Get component weights based on timeframe."""
        return self.TIMEFRAME_WEIGHTS.get(timeframe.lower(), self.DEFAULT_WEIGHTS)

    def collect_factors(
        self,
        volume_profile: VolumeProfileResult,
        smart_money: SmartMoneyResult,
        intraday: IntradayResult,
        sentiment: SentimentResult
    ) -> tuple:
        """Collect bullish/bearish factors and warnings."""
        signals = []
        bullish = []
        bearish = []
        warnings = []

        # Volume Profile
        if 'ABOVE' in volume_profile.signal:
            signals.append('VP:ABOVE_POC')
            bullish.append(f'[VOLUME_PROFILE] {volume_profile.reason}')
        elif 'BELOW' in volume_profile.signal:
            signals.append('VP:BELOW_POC')
            bearish.append(f'[VOLUME_PROFILE] {volume_profile.reason}')
        if 'EXTENDED' in volume_profile.signal:
            warnings.append(f'[VOLUME_PROFILE] Price extended from value area')

        # Smart Money
        if 'ACCUMULATION' in smart_money.signal:
            signals.append('SM:ACCUMULATION')
            bullish.append(f'[SMART_MONEY] {smart_money.reason}')
        elif 'DISTRIBUTION' in smart_money.signal:
            signals.append('SM:DISTRIBUTION')
            bearish.append(f'[SMART_MONEY] {smart_money.reason}')
        if smart_money.obv_divergence == 'bullish':
            bullish.append('[SMART_MONEY] Bullish OBV divergence')
        elif smart_money.obv_divergence == 'bearish':
            bearish.append('[SMART_MONEY] Bearish OBV divergence')
        if smart_money.unusual_volume_type == 'climax':
            warnings.append('[SMART_MONEY] Climax volume - potential reversal')

        # Intraday
        if intraday.gap_type == 'up' and not intraday.gap_filled:
            signals.append('ID:GAP_UP')
            warnings.append(f'[INTRADAY] Gap up unfilled - {intraday.gap_fill_probability:.0%} fill probability')
        elif intraday.gap_type == 'down' and not intraday.gap_filled:
            signals.append('ID:GAP_DOWN')
            bullish.append(f'[INTRADAY] Gap down unfilled - potential fill rally')
        if 'strong_bull' in intraday.opening_strength:
            bullish.append('[INTRADAY] Strong bullish opening')
        elif 'strong_bear' in intraday.opening_strength:
            bearish.append('[INTRADAY] Strong bearish opening')

        # Sentiment
        if sentiment.contrarian_signal == 'buy_fear':
            signals.append('SENT:CONTRARIAN_BUY')
            bullish.append(f'[SENTIMENT] {sentiment.reason}')
        elif sentiment.contrarian_signal == 'sell_euphoria':
            signals.append('SENT:CONTRARIAN_SELL')
            bearish.append(f'[SENTIMENT] {sentiment.reason}')
        elif 'BULLISH' in sentiment.signal:
            bullish.append(f'[SENTIMENT] {sentiment.reason}')
        elif 'BEARISH' in sentiment.signal:
            bearish.append(f'[SENTIMENT] {sentiment.reason}')

        return signals, bullish, bearish, warnings

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = 'swing'
    ) -> MicrostructureScore:
        """
        Calculate comprehensive microstructure score.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            timeframe: 'intraday', 'swing', or 'positional'

        Returns:
            MicrostructureScore with all analysis
        """
        timestamp = datetime.now()

        # Run all analyzers
        volume_profile_result = self.volume_profile_analyzer.analyze(df)
        smart_money_result = self.smart_money_detector.analyze(df)
        intraday_result = self.intraday_analyzer.analyze(df)
        sentiment_result = self.sentiment_analyzer.analyze(df)

        # Get individual scores
        scores = {
            'volume_profile': volume_profile_result.probability_score,
            'smart_money': smart_money_result.probability_score,
            'intraday': intraday_result.probability_score,
            'sentiment': sentiment_result.probability_score
        }

        # Get weights
        weights = self.get_weights(timeframe)

        # Calculate composite
        composite = sum(weights[k] * scores[k] for k in weights)

        # Key metrics
        poc_price = volume_profile_result.poc_price
        ad_status = smart_money_result.accumulation_distribution
        gap_status = f'{intraday_result.gap_type}_{intraday_result.gap_size_pct:.1f}%' \
                     if intraday_result.gap_type != 'none' else 'no_gap'
        sent_label = sentiment_result.sentiment_label

        # Collect factors
        signals, bullish, bearish, warnings = self.collect_factors(
            volume_profile_result,
            smart_money_result,
            intraday_result,
            sentiment_result
        )

        return MicrostructureScore(
            symbol=symbol,
            timestamp=timestamp,
            volume_profile_score=scores['volume_profile'],
            smart_money_score=scores['smart_money'],
            intraday_score=scores['intraday'],
            sentiment_score=scores['sentiment'],
            composite_score=composite,
            poc_price=poc_price,
            accumulation_distribution=ad_status,
            gap_status=gap_status,
            sentiment_label=sent_label,
            volume_profile_result=volume_profile_result,
            smart_money_result=smart_money_result,
            intraday_result=intraday_result,
            sentiment_result=sentiment_result,
            signals=signals,
            bullish_factors=bullish,
            bearish_factors=bearish,
            warnings=warnings
        )

    def score_quick(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = 'swing'
    ) -> float:
        """Quick scoring returning only composite."""
        full_score = self.score(symbol, df, timeframe)
        return full_score.composite_score
