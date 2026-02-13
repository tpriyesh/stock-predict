"""
Technical indicators calculation.
All indicators are computed with explicit formulas for transparency.
"""
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class TechnicalIndicators:
    """
    Calculate technical indicators for stock analysis.

    All calculations are deterministic and use standard formulas.
    """

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            Various period parameters

        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(df) < 30:
            logger.warning(f"Insufficient data for reliable indicators: {len(df)} rows (need 30+)")

        # Price change
        df['change'] = df['close'].diff()
        df['change_pct'] = df['close'].pct_change() * 100

        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=rsi_period)

        # MACD
        macd_result = TechnicalIndicators.macd(
            df['close'],
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal
        )
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['signal']
        df['macd_histogram'] = macd_result['histogram']

        # Bollinger Bands
        bb = TechnicalIndicators.bollinger_bands(df['close'], period=bb_period, std=bb_std)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        bb_middle_safe = bb['middle'].replace(0, np.nan)
        bb_range = (bb['upper'] - bb['lower']).replace(0, np.nan)
        df['bb_width'] = (bb['upper'] - bb['lower']) / bb_middle_safe * 100
        df['bb_position'] = (df['close'] - bb['lower']) / bb_range

        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=atr_period)
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # Price vs MAs (guard against NaN/zero in early rows)
        sma20_safe = df['sma_20'].replace(0, np.nan)
        sma50_safe = df['sma_50'].replace(0, np.nan)
        sma200_safe = df['sma_200'].replace(0, np.nan)
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / sma20_safe * 100
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / sma50_safe * 100
        df['price_vs_sma200'] = (df['close'] - df['sma_200']) / sma200_safe * 100

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        vol_sma_safe = df['volume_sma_20'].replace(0, np.nan)
        df['volume_ratio'] = df['volume'] / vol_sma_safe

        # VWAP (daily-reset for intraday relevance)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_numerator = (df['volume'] * typical_price)
        if hasattr(df.index, 'date'):
            # Group by date so VWAP resets each day
            date_groups = df.index.date
            df['vwap'] = vwap_numerator.groupby(date_groups).cumsum() / df['volume'].groupby(date_groups).cumsum()
        else:
            # Fallback: single-day data or no datetime index
            df['vwap'] = vwap_numerator.cumsum() / df['volume'].cumsum()

        # Momentum
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100

        # Support/Resistance levels
        sr_levels = TechnicalIndicators.support_resistance(df, lookback=20)
        df['support'] = sr_levels['support']
        df['resistance'] = sr_levels['resistance']

        # Trend strength (ADX-like)
        df['trend_strength'] = TechnicalIndicators.trend_strength(df, period=14)

        # Candlestick patterns
        patterns = TechnicalIndicators.candlestick_patterns(df)
        for col in patterns.columns:
            df[col] = patterns[col]

        # Fill NaN with indicator-specific neutral values instead of 0
        # RSI=0 would mean "maximally oversold" instead of "no data"
        # ADX=0 would mean "no trend" when actually it's unknown
        _indicator_defaults = {
            'rsi': 50.0,              # Neutral RSI (not oversold/overbought)
            'macd': 0.0,              # No momentum
            'macd_signal': 0.0,       # No signal
            'macd_histogram': 0.0,    # No divergence
            'trend_strength': 25.0,   # Weak trend (below strong threshold)
            'atr': 0.0,              # No volatility data
            'atr_pct': 0.0,          # No volatility data
            'bb_width': 0.0,         # Will be overridden by BB band fill below
            'bb_position': 0.5,      # Mid-band (neutral)
            'volume_ratio': 1.0,     # Normal volume (not high/low)
            'momentum_5': 0.0,       # No momentum
            'momentum_10': 0.0,      # No momentum
            'momentum_20': 0.0,      # No momentum
            'change': 0.0,           # No change
            'change_pct': 0.0,       # No change
        }
        for col, default in _indicator_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        # For Bollinger Bands, use close price as neutral when NaN
        # (bands collapse to price = no signal, rather than 0 = crash)
        if 'close' in df.columns:
            for bb_col in ['bb_upper', 'bb_middle', 'bb_lower']:
                if bb_col in df.columns:
                    df[bb_col] = df[bb_col].fillna(df['close'])

        # For moving averages, use close price as neutral when NaN
        # (price = MA means no crossover signal, rather than 0 = massive divergence)
        if 'close' in df.columns:
            for ma_col in ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'vwap']:
                if ma_col in df.columns:
                    df[ma_col] = df[ma_col].fillna(df['close'])
            # Price vs MA should be 0 (no divergence) when MA is unknown
            for pv_col in ['price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200']:
                if pv_col in df.columns:
                    df[pv_col] = df[pv_col].fillna(0.0)

        # Support/resistance: use low/high as fallback
        if 'support' in df.columns:
            df['support'] = df['support'].fillna(df['low'])
        if 'resistance' in df.columns:
            df['resistance'] = df['resistance'].fillna(df['high'])

        # Fill any remaining numeric NaN with 0 (catches OHLCV, patterns, etc.)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use exponential moving average for smoother results
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        avg_loss_safe = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss_safe
        rsi = 100 - (100 / (1 + rs))
        # When avg_loss is 0 (all gains), RSI should be 100
        rsi = rsi.fillna(100)

        return rsi

    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD Line = 12 EMA - 26 EMA
        Signal Line = 9 EMA of MACD Line
        Histogram = MACD Line - Signal Line
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std: float = 2.0
    ) -> dict:
        """
        Calculate Bollinger Bands.

        Middle = SMA(period)
        Upper = Middle + (std * StdDev)
        Lower = Middle - (std * StdDev)
        """
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.

        TR = max(H-L, |H-Pc|, |L-Pc|)
        ATR = EMA(TR, period)
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Calculate dynamic support and resistance levels.

        Uses recent swing highs/lows.
        """
        support = df['low'].rolling(window=lookback).min()
        resistance = df['high'].rolling(window=lookback).max()

        return {
            'support': support,
            'resistance': resistance
        }

    @staticmethod
    def trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate trend strength (simplified ADX-like indicator).

        Returns value 0-100 where:
        - < 25: Weak trend / ranging
        - 25-50: Moderate trend
        - > 50: Strong trend
        """
        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR
        atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period)

        # Directional indicators (guard against zero ATR)
        atr_safe = atr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr_safe)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr_safe)

        # DX and ADX (guard against zero denominator)
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / di_sum
        adx = dx.ewm(span=period).mean()

        return adx.fillna(25.0)  # Neutral: weak trend, not "no trend"

    @staticmethod
    def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candlestick patterns.

        Returns DataFrame with pattern signals (-1, 0, 1).
        """
        patterns = pd.DataFrame(index=df.index)

        body = df['close'] - df['open']
        body_abs = abs(body)
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']

        # Doji (small body relative to range) - guard total_range == 0
        total_range_safe = total_range.replace(0, np.nan)
        patterns['doji'] = (body_abs / total_range_safe < 0.1).fillna(False).astype(int)

        # Hammer (long lower shadow, small upper shadow, small body at top)
        hammer_cond = (
            (lower_shadow > 2 * body_abs) &
            (upper_shadow < body_abs * 0.5) &
            (body > 0)
        )
        patterns['hammer'] = hammer_cond.astype(int)

        # Shooting Star (long upper shadow, small lower shadow, small body at bottom)
        shooting_star_cond = (
            (upper_shadow > 2 * body_abs) &
            (lower_shadow < body_abs * 0.5) &
            (body < 0)
        )
        patterns['shooting_star'] = (-shooting_star_cond).astype(int)

        # Engulfing patterns
        prev_body = body.shift(1)
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)

        bullish_engulfing = (
            (prev_body < 0) &  # Previous was bearish
            (body > 0) &  # Current is bullish
            (df['open'] < prev_close) &  # Opens below prev close
            (df['close'] > prev_open)  # Closes above prev open
        )
        patterns['bullish_engulfing'] = bullish_engulfing.astype(int)

        bearish_engulfing = (
            (prev_body > 0) &  # Previous was bullish
            (body < 0) &  # Current is bearish
            (df['open'] > prev_close) &  # Opens above prev close
            (df['close'] < prev_open)  # Closes below prev open
        )
        patterns['bearish_engulfing'] = (-bearish_engulfing).astype(int)

        # Morning/Evening star (3-candle pattern) - simplified
        prev2_body = body.shift(2)
        morning_star = (
            (prev2_body < 0) &  # Day 1: bearish
            (abs(prev_body) < body_abs * 0.3) &  # Day 2: small body (doji-like)
            (body > 0) &  # Day 3: bullish
            (df['close'] > df['open'].shift(2))  # Closes above day 1 open
        )
        patterns['morning_star'] = morning_star.astype(int)

        return patterns

    @staticmethod
    def get_signal_summary(df: pd.DataFrame) -> dict:
        """
        Get a summary of technical signals from the latest row.

        Returns dict with signal interpretations.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        summary = {
            'date': df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1]),
            'close': round(latest['close'], 2),
            'change_pct': round(latest['change_pct'], 2) if pd.notna(latest.get('change_pct')) else 0,
        }

        # RSI interpretation
        rsi = latest.get('rsi')
        if pd.notna(rsi):
            summary['rsi'] = round(rsi, 2)
            if rsi < 30:
                summary['rsi_signal'] = 'OVERSOLD'
            elif rsi > 70:
                summary['rsi_signal'] = 'OVERBOUGHT'
            else:
                summary['rsi_signal'] = 'NEUTRAL'

        # MACD interpretation
        macd_hist = latest.get('macd_histogram')
        prev_hist = df['macd_histogram'].iloc[-2] if len(df) > 1 else None
        if pd.notna(macd_hist):
            summary['macd_histogram'] = round(macd_hist, 4)
            if macd_hist > 0 and (prev_hist is None or macd_hist > prev_hist):
                summary['macd_signal'] = 'BULLISH_MOMENTUM'
            elif macd_hist < 0 and (prev_hist is None or macd_hist < prev_hist):
                summary['macd_signal'] = 'BEARISH_MOMENTUM'
            else:
                summary['macd_signal'] = 'NEUTRAL'

        # Bollinger Band position
        bb_pos = latest.get('bb_position')
        if pd.notna(bb_pos):
            summary['bb_position'] = round(bb_pos, 2)
            if bb_pos < 0.2:
                summary['bb_signal'] = 'NEAR_LOWER_BAND'
            elif bb_pos > 0.8:
                summary['bb_signal'] = 'NEAR_UPPER_BAND'
            else:
                summary['bb_signal'] = 'MIDDLE'

        # Trend (price vs MAs)
        if pd.notna(latest.get('sma_20')) and pd.notna(latest.get('sma_50')):
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                summary['trend'] = 'UPTREND'
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                summary['trend'] = 'DOWNTREND'
            else:
                summary['trend'] = 'SIDEWAYS'

        # Volume
        vol_ratio = latest.get('volume_ratio')
        if pd.notna(vol_ratio):
            summary['volume_ratio'] = round(vol_ratio, 2)
            if vol_ratio > 1.5:
                summary['volume_signal'] = 'HIGH_VOLUME'
            elif vol_ratio < 0.5:
                summary['volume_signal'] = 'LOW_VOLUME'
            else:
                summary['volume_signal'] = 'NORMAL'

        # ATR for volatility
        atr_pct = latest.get('atr_pct')
        if pd.notna(atr_pct):
            summary['atr_pct'] = round(atr_pct, 2)

        # Candlestick patterns
        patterns_detected = []
        for pattern in ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'morning_star']:
            if pattern in latest and latest[pattern] != 0:
                patterns_detected.append(pattern.upper())
        summary['patterns'] = patterns_detected

        return summary
