"""
Enhanced Sentiment Analyzer

Multi-dimensional sentiment analysis:
- Price-based sentiment proxy
- Volume-based sentiment
- Momentum sentiment
- Contrarian signals

Note: For true sentiment, integrate with news APIs.
This module uses price/volume as sentiment proxies.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    # Price-based sentiment
    price_sentiment: float          # -1 to 1
    price_momentum: float           # Rate of change of sentiment

    # Volume sentiment
    volume_sentiment: float         # -1 to 1 (high vol on up = bullish)

    # Overall
    composite_sentiment: float      # -1 to 1
    sentiment_label: str            # 'bullish', 'bearish', 'neutral'

    # Contrarian signals
    extreme_sentiment: bool         # At sentiment extreme?
    contrarian_signal: str          # 'buy_fear', 'sell_euphoria', 'none'

    # Sentiment momentum
    sentiment_accelerating: bool    # Is sentiment getting stronger?
    sentiment_direction: str        # 'improving', 'deteriorating', 'stable'

    probability_score: float        # Contribution to prediction (0-1)
    signal: str                     # 'SENTIMENT_BULLISH', 'CONTRARIAN_BUY', etc.
    reason: str                     # Human-readable explanation


class EnhancedSentimentAnalyzer:
    """
    Analyzes market sentiment using price and volume proxies.

    For production, integrate with:
    - News sentiment APIs
    - Social media sentiment
    - Options sentiment (put/call ratios)
    """

    def __init__(
        self,
        momentum_period: int = 10,
        extreme_threshold: float = 0.7
    ):
        self.momentum_period = momentum_period
        self.extreme_threshold = extreme_threshold

    def calculate_price_sentiment(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate sentiment from price action.

        Uses returns and position relative to recent range.
        """
        if len(df) < 20:
            return 0

        close = df['close']

        # Recent returns
        return_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(df) > 5 else 0
        return_10d = close.iloc[-1] / close.iloc[-11] - 1 if len(df) > 10 else 0
        return_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(df) > 20 else 0

        # Position in range
        high_20 = close.tail(20).max()
        low_20 = close.tail(20).min()
        range_position = (close.iloc[-1] - low_20) / (high_20 - low_20 + 1e-10)

        # Combine
        sentiment = (
            return_5d * 3 +      # Recent returns weighted more
            return_10d * 2 +
            return_20d * 1 +
            (range_position - 0.5) * 2  # Position in range
        ) / 8

        return np.clip(sentiment, -1, 1)

    def calculate_volume_sentiment(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate sentiment from volume patterns.

        High volume on up days = bullish
        High volume on down days = bearish
        """
        if len(df) < 10:
            return 0

        data = df.tail(10)

        # Direction of each day
        direction = np.sign(data['close'].diff())

        # Volume relative to average
        vol_ratio = data['volume'] / data['volume'].rolling(10).mean()

        # Volume-weighted sentiment
        weighted_direction = (direction * vol_ratio).dropna()

        if len(weighted_direction) == 0:
            return 0

        sentiment = weighted_direction.sum() / len(weighted_direction)

        return np.clip(sentiment, -1, 1)

    def calculate_momentum_sentiment(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """
        Calculate sentiment momentum (rate of change).

        Returns (momentum, direction)
        """
        if len(df) < 20:
            return 0, 'stable'

        # Calculate rolling sentiment
        sentiments = []
        for i in range(10, len(df)):
            subset = df.iloc[:i]
            sent = self.calculate_price_sentiment(subset)
            sentiments.append(sent)

        if len(sentiments) < 5:
            return 0, 'stable'

        # Momentum = change in sentiment
        recent = sentiments[-5:]
        earlier = sentiments[-10:-5]

        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)

        momentum = recent_avg - earlier_avg

        if momentum > 0.1:
            direction = 'improving'
        elif momentum < -0.1:
            direction = 'deteriorating'
        else:
            direction = 'stable'

        return momentum, direction

    def detect_extreme_sentiment(
        self,
        sentiment: float
    ) -> tuple:
        """
        Detect extreme sentiment for contrarian signals.

        Returns (is_extreme, contrarian_signal)
        """
        if abs(sentiment) < self.extreme_threshold:
            return False, 'none'

        if sentiment > self.extreme_threshold:
            # Extreme bullishness - potential top
            return True, 'sell_euphoria'
        else:
            # Extreme bearishness - potential bottom
            return True, 'buy_fear'

    def combine_sentiment(
        self,
        price_sentiment: float,
        volume_sentiment: float
    ) -> float:
        """Combine different sentiment measures."""
        # Weight price sentiment more
        composite = (
            price_sentiment * 0.6 +
            volume_sentiment * 0.4
        )
        return np.clip(composite, -1, 1)

    def classify_sentiment(self, sentiment: float) -> str:
        """Classify sentiment level."""
        if sentiment > 0.2:
            return 'bullish'
        elif sentiment < -0.2:
            return 'bearish'
        else:
            return 'neutral'

    def get_signal_label(
        self,
        sentiment_label: str,
        contrarian_signal: str,
        direction: str
    ) -> tuple:
        """Determine signal label and reason."""

        if contrarian_signal == 'buy_fear':
            return 'CONTRARIAN_BUY', 'Extreme bearish sentiment - contrarian buy opportunity'
        elif contrarian_signal == 'sell_euphoria':
            return 'CONTRARIAN_SELL', 'Extreme bullish sentiment - contrarian sell signal'

        if sentiment_label == 'bullish' and direction == 'improving':
            return 'SENTIMENT_STRONG_BULL', 'Bullish sentiment and improving'
        elif sentiment_label == 'bearish' and direction == 'deteriorating':
            return 'SENTIMENT_STRONG_BEAR', 'Bearish sentiment and deteriorating'
        elif sentiment_label == 'bullish':
            return 'SENTIMENT_BULLISH', 'Bullish sentiment'
        elif sentiment_label == 'bearish':
            return 'SENTIMENT_BEARISH', 'Bearish sentiment'
        else:
            return 'SENTIMENT_NEUTRAL', 'Neutral sentiment'

    def analyze(self, df: pd.DataFrame) -> SentimentResult:
        """
        Analyze market sentiment.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            SentimentResult with sentiment analysis
        """
        if len(df) < 20:
            return self._default_result('Insufficient data for sentiment analysis')

        # Calculate components
        price_sent = self.calculate_price_sentiment(df)
        volume_sent = self.calculate_volume_sentiment(df)
        momentum, direction = self.calculate_momentum_sentiment(df)

        # Combine
        composite = self.combine_sentiment(price_sent, volume_sent)
        label = self.classify_sentiment(composite)

        # Contrarian
        is_extreme, contrarian = self.detect_extreme_sentiment(composite)

        # Acceleration
        accelerating = abs(momentum) > 0.15

        # Signal and reason
        signal, reason = self.get_signal_label(label, contrarian, direction)

        # Probability score
        if contrarian == 'buy_fear':
            prob_score = 0.62  # Contrarian buy
        elif contrarian == 'sell_euphoria':
            prob_score = 0.38  # Contrarian sell
        elif label == 'bullish':
            prob_score = 0.55 + composite * 0.10
        elif label == 'bearish':
            prob_score = 0.45 + composite * 0.10
        else:
            prob_score = 0.50

        # Adjust for momentum
        if direction == 'improving':
            prob_score += 0.02
        elif direction == 'deteriorating':
            prob_score -= 0.02

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return SentimentResult(
            price_sentiment=price_sent,
            price_momentum=momentum,
            volume_sentiment=volume_sent,
            composite_sentiment=composite,
            sentiment_label=label,
            extreme_sentiment=is_extreme,
            contrarian_signal=contrarian,
            sentiment_accelerating=accelerating,
            sentiment_direction=direction,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> SentimentResult:
        """Return neutral result."""
        return SentimentResult(
            price_sentiment=0,
            price_momentum=0,
            volume_sentiment=0,
            composite_sentiment=0,
            sentiment_label='neutral',
            extreme_sentiment=False,
            contrarian_signal='none',
            sentiment_accelerating=False,
            sentiment_direction='stable',
            probability_score=0.50,
            signal='SENTIMENT_UNKNOWN',
            reason=reason
        )
