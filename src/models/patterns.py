"""
Historical pattern matching for Indian stocks.
Finds similar setups from the past and their outcomes.
"""
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class PatternMatcher:
    """
    Finds historical patterns similar to current setup.
    Uses technical indicator similarity to find past analogues.
    """

    def __init__(self):
        self.min_similarity = 0.7  # Minimum similarity threshold

    def find_similar_patterns(
        self,
        df: pd.DataFrame,
        lookback_days: int = 252,
        forward_days: int = 5,
        top_n: int = 10
    ) -> list[dict]:
        """
        Find historical patterns similar to current setup.

        Args:
            df: DataFrame with OHLCV + technical indicators
            lookback_days: How far back to search for patterns
            forward_days: How many days forward to check outcome
            top_n: Number of similar patterns to return

        Returns:
            List of similar patterns with their outcomes
        """
        if len(df) < lookback_days + forward_days:
            logger.warning("Insufficient data for pattern matching")
            return []

        # Current pattern (latest row)
        current = df.iloc[-1]

        # Features to compare
        features = [
            'rsi', 'macd_histogram', 'bb_position',
            'price_vs_sma20', 'price_vs_sma50',
            'volume_ratio', 'atr_pct', 'momentum_5', 'momentum_20'
        ]

        # Filter features that exist
        features = [f for f in features if f in df.columns and pd.notna(current.get(f))]

        if len(features) < 5:
            logger.warning("Insufficient features for pattern matching")
            return []

        # Get current feature vector
        current_vector = np.array([current[f] for f in features])

        # Search through history
        similar_patterns = []

        for i in range(lookback_days, len(df) - forward_days - 1):
            historical = df.iloc[i]

            # Skip if missing features
            if any(pd.isna(historical.get(f)) for f in features):
                continue

            # Calculate similarity
            hist_vector = np.array([historical[f] for f in features])
            similarity = self._calculate_similarity(current_vector, hist_vector)

            if similarity >= self.min_similarity:
                # Calculate outcome (forward return)
                entry_price = df.iloc[i]['close']
                exit_price = df.iloc[i + forward_days]['close']
                forward_return = (exit_price - entry_price) / entry_price * 100

                # Max drawdown in forward period
                forward_prices = df.iloc[i:i + forward_days + 1]['low']
                max_drawdown = (entry_price - forward_prices.min()) / entry_price * 100

                # Max gain
                forward_highs = df.iloc[i:i + forward_days + 1]['high']
                max_gain = (forward_highs.max() - entry_price) / entry_price * 100

                similar_patterns.append({
                    'date': df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                    'similarity': round(similarity, 3),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'forward_return': round(forward_return, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'max_gain': round(max_gain, 2),
                    'profitable': forward_return > 0,
                    'features': {f: round(float(historical[f]), 3) for f in features}
                })

        # Sort by similarity
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)

        return similar_patterns[:top_n]

    def _calculate_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors.
        Uses cosine similarity normalized to 0-1 range.
        """
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        cosine_sim = np.dot(v1, v2) / (norm1 * norm2)

        # Convert from [-1, 1] to [0, 1]
        return (cosine_sim + 1) / 2

    def get_pattern_statistics(self, patterns: list[dict]) -> dict:
        """
        Calculate statistics from similar patterns.

        Returns:
            Dict with win rate, avg return, etc.
        """
        if not patterns:
            return {
                'count': 0,
                'win_rate': 0,
                'avg_return': 0,
                'median_return': 0,
                'best_return': 0,
                'worst_return': 0,
                'avg_max_gain': 0,
                'avg_max_drawdown': 0,
                'confidence': 0
            }

        returns = [p['forward_return'] for p in patterns]
        profitable = [p for p in patterns if p['profitable']]

        stats = {
            'count': len(patterns),
            'win_rate': len(profitable) / len(patterns),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_max_gain': np.mean([p['max_gain'] for p in patterns]),
            'avg_max_drawdown': np.mean([p['max_drawdown'] for p in patterns]),
            'std_return': np.std(returns),
        }

        # Confidence based on sample size and consistency
        if len(patterns) >= 10 and stats['win_rate'] > 0.6:
            stats['confidence'] = min(0.9, stats['win_rate'] * (len(patterns) / 20))
        elif len(patterns) >= 5:
            stats['confidence'] = min(0.7, stats['win_rate'] * 0.8)
        else:
            stats['confidence'] = min(0.5, stats['win_rate'] * 0.5)

        return stats

    def format_pattern_report(
        self,
        patterns: list[dict],
        stats: dict,
        symbol: str
    ) -> str:
        """Format pattern matching results as readable report."""

        if not patterns:
            return f"No similar historical patterns found for {symbol}"

        report = f"""
## Historical Pattern Analysis: {symbol}

Found **{stats['count']}** similar setups in the past year.

### Statistics (next 5 trading days):
| Metric | Value |
|--------|-------|
| Win Rate | {stats['win_rate']:.0%} |
| Avg Return | {stats['avg_return']:+.2f}% |
| Median Return | {stats['median_return']:+.2f}% |
| Best Case | {stats['best_return']:+.2f}% |
| Worst Case | {stats['worst_return']:+.2f}% |
| Avg Max Gain | {stats['avg_max_gain']:+.2f}% |
| Avg Max Drawdown | -{stats['avg_max_drawdown']:.2f}% |
| Pattern Confidence | {stats['confidence']:.0%} |

### Most Similar Past Setups:
"""
        for i, p in enumerate(patterns[:5], 1):
            emoji = "✅" if p['profitable'] else "❌"
            report += f"""
**{i}. {p['date']}** (Similarity: {p['similarity']:.0%})
   Entry: ₹{p['entry_price']} → Exit: ₹{p['exit_price']} = {emoji} {p['forward_return']:+.2f}%
"""

        return report


class IndianMarketPatterns:
    """
    India-specific pattern detection.
    Includes patterns that work well in Indian markets.
    """

    @staticmethod
    def detect_fii_driven_momentum(df: pd.DataFrame) -> dict:
        """
        Detect FII-driven momentum pattern.
        High volume + price rise often indicates FII buying.
        """
        if len(df) < 20:
            return {'detected': False}

        latest = df.iloc[-1]
        recent = df.tail(5)

        # Strong volume for 3+ days
        high_volume_days = (recent['volume_ratio'] > 1.3).sum()

        # Consistent upward movement
        up_days = (recent['close'] > recent['open']).sum()

        # Price above key MAs
        above_sma20 = latest['close'] > latest.get('sma_20', 0)
        above_sma50 = latest['close'] > latest.get('sma_50', 0)

        detected = high_volume_days >= 3 and up_days >= 3 and above_sma20 and above_sma50

        return {
            'detected': detected,
            'pattern': 'FII_DRIVEN_MOMENTUM',
            'high_volume_days': int(high_volume_days),
            'up_days': int(up_days),
            'strength': min(1.0, (high_volume_days + up_days) / 10),
            'description': 'Consistent buying with high volume - possible institutional accumulation'
        }

    @staticmethod
    def detect_nifty_correlation_breakout(
        stock_df: pd.DataFrame,
        nifty_df: pd.DataFrame
    ) -> dict:
        """
        Detect when stock is outperforming NIFTY.
        Relative strength breakout.
        """
        if len(stock_df) < 20 or len(nifty_df) < 20:
            return {'detected': False}

        # Calculate relative performance
        stock_return = (stock_df['close'].iloc[-1] / stock_df['close'].iloc[-20] - 1) * 100
        nifty_return = (nifty_df['close'].iloc[-1] / nifty_df['close'].iloc[-20] - 1) * 100

        outperformance = stock_return - nifty_return

        detected = outperformance > 5  # Outperforming by >5%

        return {
            'detected': detected,
            'pattern': 'RELATIVE_STRENGTH_BREAKOUT',
            'stock_return': round(stock_return, 2),
            'nifty_return': round(nifty_return, 2),
            'outperformance': round(outperformance, 2),
            'description': f'Outperforming NIFTY by {outperformance:.1f}% over 20 days'
        }

    @staticmethod
    def detect_earnings_momentum(df: pd.DataFrame, days_since_earnings: int = 10) -> dict:
        """
        Detect post-earnings momentum.
        Stocks often trend for 2-3 weeks after results.
        """
        if len(df) < days_since_earnings:
            return {'detected': False}

        recent = df.tail(days_since_earnings)

        # Calculate momentum since "earnings"
        price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100

        # Volume surge
        avg_volume_before = df.iloc[-days_since_earnings - 20:-days_since_earnings]['volume'].mean()
        avg_volume_after = recent['volume'].mean()
        volume_surge = avg_volume_after / avg_volume_before if avg_volume_before > 0 else 1

        detected = abs(price_change) > 5 and volume_surge > 1.5

        return {
            'detected': detected,
            'pattern': 'EARNINGS_MOMENTUM',
            'direction': 'BULLISH' if price_change > 0 else 'BEARISH',
            'price_change': round(price_change, 2),
            'volume_surge': round(volume_surge, 2),
            'description': f'Post-event momentum: {price_change:+.1f}% with {volume_surge:.1f}x volume'
        }

    @staticmethod
    def detect_all_patterns(df: pd.DataFrame) -> list[dict]:
        """Detect all Indian market patterns."""
        patterns = []

        fii = IndianMarketPatterns.detect_fii_driven_momentum(df)
        if fii['detected']:
            patterns.append(fii)

        earnings = IndianMarketPatterns.detect_earnings_momentum(df)
        if earnings['detected']:
            patterns.append(earnings)

        return patterns
