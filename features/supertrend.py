"""
Supertrend Indicator - Most reliable intraday trend indicator.

Based on ATR (Average True Range) for volatility-adjusted trend detection.
Used by: alice_blue_futures, vwap_supertrend_bot

Parameters:
- period: ATR lookback (default 7, range 7-10 optimal)
- multiplier: Band distance (default 2.5, range 2-3 optimal)
"""
import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    True Range = max(H-L, |H-Pc|, |L-Pc|)
    ATR = EMA(TR, period) using Wilder's smoothing
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (alpha = 1/period)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 2.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator.

    Algorithm:
    1. Calculate ATR
    2. Calculate basic upper/lower bands from HL2 ± (multiplier * ATR)
    3. Final bands "stick" to price during trends:
       - Upper band only moves DOWN
       - Lower band only moves UP
    4. Trend direction flips when price crosses bands

    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 7)
        multiplier: Band multiplier (default 2.5)

    Returns:
        Tuple of (supertrend_line, direction)
        - supertrend_line: The ST value (lower band in uptrend, upper in downtrend)
        - direction: 1 for uptrend, -1 for downtrend
    """
    if len(df) < period + 1:
        logger.warning(f"Insufficient data for Supertrend: {len(df)} rows")
        return pd.Series([np.nan] * len(df)), pd.Series([0] * len(df))

    # Calculate ATR
    atr = calculate_atr(df, period)

    # HL2 = (High + Low) / 2
    hl2 = (df['high'] + df['low']) / 2

    # Basic bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Final bands (with stickiness)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    close = df['close']

    for i in range(1, len(df)):
        # Upper band: only move DOWN (resistance gets tighter)
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        # Lower band: only move UP (support gets tighter)
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    # Determine Supertrend value and direction
    supertrend = pd.Series([np.nan] * len(df), index=df.index)
    direction = pd.Series([1] * len(df), index=df.index)  # 1=up, -1=down

    for i in range(1, len(df)):
        # Previous trend was UP (using lower band)
        if supertrend.iloc[i-1] == final_lower.iloc[i-1]:
            if close.iloc[i] < final_lower.iloc[i]:
                # Price broke support - trend flips DOWN
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                # Still in uptrend
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1

        # Previous trend was DOWN (using upper band)
        elif supertrend.iloc[i-1] == final_upper.iloc[i-1]:
            if close.iloc[i] > final_upper.iloc[i]:
                # Price broke resistance - trend flips UP
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                # Still in downtrend
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

        # Initial value (first valid row)
        else:
            if close.iloc[i] > basic_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return supertrend, direction


def get_supertrend_signal(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 2.5
) -> dict:
    """
    Get Supertrend trading signal.

    Returns:
        dict with:
        - signal: 'BUY', 'SELL', or 'HOLD'
        - supertrend: current ST value
        - direction: 1 (up) or -1 (down)
        - trend_changed: True if trend just flipped
        - close_vs_st: close price relative to ST (% above/below)
    """
    supertrend, direction = calculate_supertrend(df, period, multiplier)

    if len(df) < 2:
        return {
            'signal': 'HOLD',
            'supertrend': None,
            'direction': 0,
            'trend_changed': False,
            'close_vs_st': 0
        }

    current_direction = direction.iloc[-1]
    prev_direction = direction.iloc[-2]
    current_st = supertrend.iloc[-1]
    current_close = df['close'].iloc[-1]

    trend_changed = current_direction != prev_direction

    # Calculate close vs supertrend
    if not np.isnan(current_st) and current_st > 0:
        close_vs_st = (current_close - current_st) / current_st * 100
    else:
        close_vs_st = 0

    # Generate signal
    if trend_changed:
        if current_direction == 1:
            signal = 'BUY'
        else:
            signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'signal': signal,
        'supertrend': round(current_st, 2) if not np.isnan(current_st) else None,
        'direction': current_direction,
        'trend_changed': trend_changed,
        'close_vs_st': round(close_vs_st, 2)
    }


def add_supertrend_to_df(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 2.5
) -> pd.DataFrame:
    """
    Add Supertrend columns to DataFrame.

    Adds:
    - 'supertrend': The ST line value
    - 'st_direction': 1 for uptrend, -1 for downtrend
    - 'st_signal': 'BUY' on flip up, 'SELL' on flip down, else 'HOLD'
    """
    df = df.copy()

    supertrend, direction = calculate_supertrend(df, period, multiplier)

    df['supertrend'] = supertrend
    df['st_direction'] = direction

    # Generate signals on direction change
    df['st_signal'] = 'HOLD'
    direction_change = direction.diff()
    df.loc[direction_change == 2, 'st_signal'] = 'BUY'   # -1 to 1
    df.loc[direction_change == -2, 'st_signal'] = 'SELL' # 1 to -1

    return df


# Demo function
def demo_supertrend():
    """Demo the Supertrend calculation"""
    import yfinance as yf

    print("=" * 60)
    print("SUPERTREND INDICATOR DEMO")
    print("=" * 60)

    # Fetch sample data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period='3mo')
    df.columns = df.columns.str.lower()

    # Calculate Supertrend
    signal_info = get_supertrend_signal(df, period=7, multiplier=2.5)

    print(f"\nSymbol: RELIANCE")
    print(f"Close: ₹{df['close'].iloc[-1]:.2f}")
    print(f"Supertrend: ₹{signal_info['supertrend']}")
    print(f"Direction: {'UPTREND' if signal_info['direction'] == 1 else 'DOWNTREND'}")
    print(f"Signal: {signal_info['signal']}")
    print(f"Trend Changed: {signal_info['trend_changed']}")
    print(f"Close vs ST: {signal_info['close_vs_st']:+.2f}%")


if __name__ == "__main__":
    demo_supertrend()
