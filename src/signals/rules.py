"""
Deterministic trading rules for entry, exit, and position sizing.
These rules are explicit and reproducible - no black boxes.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from loguru import logger


@dataclass
class TradeSetup:
    """A complete trade setup with entry/exit rules."""
    symbol: str
    setup_type: str  # breakout, pullback, reversal, etc.
    direction: str   # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    position_size_pct: float = 5.0  # % of capital
    entry_condition: str = ""
    exit_condition: str = ""
    invalidation: str = ""


class TradingRules:
    """
    Deterministic trading rules for various setups.

    All rules are explicit and can be verified.
    """

    @staticmethod
    def breakout_long(
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 20
    ) -> Optional[TradeSetup]:
        """
        Breakout above recent high with volume confirmation.

        Entry: Close > highest high of last N days
        Stop: Below breakout level (lowest low of last 5 days)
        Target: 2x risk from entry
        """
        if len(df) < lookback:
            return None

        latest = df.iloc[-1]
        recent = df.tail(lookback)

        highest_high = recent['high'].max()
        lowest_low = recent.tail(5)['low'].min()
        avg_volume = recent['volume'].mean()

        close = float(latest['close'])
        volume = float(latest['volume'])

        # Breakout conditions
        is_breakout = close > highest_high
        has_volume = volume > avg_volume * 1.3

        if not (is_breakout and has_volume):
            return None

        entry = close
        stop = min(lowest_low, close * 0.97)  # Max 3% stop
        risk = entry - stop
        target = entry + (risk * 2)

        return TradeSetup(
            symbol=symbol,
            setup_type="BREAKOUT",
            direction="LONG",
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_1=round(target, 2),
            target_2=round(entry + risk * 3, 2),
            entry_condition=f"Close ({close:.2f}) > {lookback}D High ({highest_high:.2f}) with volume > 1.3x avg",
            exit_condition="Hit target or stop; trail stop after 50% target reached",
            invalidation=f"Close below {stop:.2f}"
        )

    @staticmethod
    def pullback_to_ma(
        df: pd.DataFrame,
        symbol: str,
        ma_period: int = 20
    ) -> Optional[TradeSetup]:
        """
        Buy pullback to rising moving average.

        Entry: Price pulls back to SMA and shows reversal candle
        Stop: Below recent swing low
        Target: Previous high
        """
        if len(df) < ma_period + 5:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(latest['close'])
        sma = float(latest.get('sma_20', df['close'].tail(20).mean()))
        sma_prev = float(df['close'].tail(21).head(20).mean())

        # MA must be rising
        ma_rising = sma > sma_prev

        # Price near MA (within 1%)
        near_ma = abs(close - sma) / sma < 0.01

        # Bullish candle (close > open)
        bullish_candle = close > float(latest['open'])

        # Previous candle was bearish (pullback)
        prev_bearish = float(prev['close']) < float(prev['open'])

        if not (ma_rising and near_ma and bullish_candle and prev_bearish):
            return None

        recent_low = df.tail(5)['low'].min()
        recent_high = df.tail(20)['high'].max()

        entry = close
        stop = float(recent_low) * 0.99
        target = float(recent_high)

        return TradeSetup(
            symbol=symbol,
            setup_type="PULLBACK_TO_MA",
            direction="LONG",
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_1=round(target, 2),
            entry_condition=f"Pullback to rising {ma_period}MA ({sma:.2f}) with bullish reversal",
            exit_condition="Hit target or stop",
            invalidation=f"Close below {stop:.2f} or MA turns down"
        )

    @staticmethod
    def oversold_bounce(
        df: pd.DataFrame,
        symbol: str
    ) -> Optional[TradeSetup]:
        """
        Buy RSI oversold bounce.

        Entry: RSI < 30 and turning up, price above recent low
        Stop: Below recent swing low
        Target: Middle Bollinger Band
        """
        if 'rsi' not in df.columns:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = float(latest.get('rsi', 50))
        rsi_prev = float(prev.get('rsi', 50))

        # RSI oversold and turning up
        oversold = rsi < 35
        turning_up = rsi > rsi_prev

        if not (oversold and turning_up):
            return None

        close = float(latest['close'])
        bb_middle = float(latest.get('bb_middle', close * 1.02))
        recent_low = df.tail(5)['low'].min()

        entry = close
        stop = float(recent_low) * 0.99
        target = bb_middle

        return TradeSetup(
            symbol=symbol,
            setup_type="OVERSOLD_BOUNCE",
            direction="LONG",
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_1=round(target, 2),
            entry_condition=f"RSI oversold ({rsi:.1f}) and bouncing from {rsi_prev:.1f}",
            exit_condition=f"Target at middle BB ({bb_middle:.2f}) or stop hit",
            invalidation=f"RSI continues lower or price breaks {recent_low:.2f}"
        )

    @staticmethod
    def mean_reversion_short(
        df: pd.DataFrame,
        symbol: str
    ) -> Optional[TradeSetup]:
        """
        Short overbought stock for mean reversion.

        Entry: RSI > 70 and turning down, price extended above BB upper
        Stop: Above recent high
        Target: Middle Bollinger Band
        """
        if 'rsi' not in df.columns:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = float(latest.get('rsi', 50))
        rsi_prev = float(prev.get('rsi', 50))

        # RSI overbought and turning down
        overbought = rsi > 70
        turning_down = rsi < rsi_prev

        if not (overbought and turning_down):
            return None

        close = float(latest['close'])
        bb_upper = float(latest.get('bb_upper', close))
        bb_middle = float(latest.get('bb_middle', close * 0.98))

        # Price should be near or above upper BB
        extended = close >= bb_upper * 0.98

        if not extended:
            return None

        recent_high = df.tail(5)['high'].max()

        entry = close
        stop = float(recent_high) * 1.01
        target = bb_middle

        return TradeSetup(
            symbol=symbol,
            setup_type="MEAN_REVERSION",
            direction="SHORT",
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_1=round(target, 2),
            entry_condition=f"RSI overbought ({rsi:.1f}) and reversing, price at upper BB",
            exit_condition=f"Target at middle BB ({bb_middle:.2f}) or stop hit",
            invalidation=f"Price breaks above {recent_high:.2f}"
        )

    @staticmethod
    def find_setups(df: pd.DataFrame, symbol: str) -> list[TradeSetup]:
        """
        Find all valid trade setups for a stock.

        Returns list of TradeSetup objects.
        """
        setups = []

        # Try each setup type
        for setup_func in [
            TradingRules.breakout_long,
            TradingRules.pullback_to_ma,
            TradingRules.oversold_bounce,
            TradingRules.mean_reversion_short
        ]:
            try:
                setup = setup_func(df, symbol)
                if setup:
                    setups.append(setup)
            except Exception as e:
                logger.warning(f"Error finding {setup_func.__name__} for {symbol}: {e}")

        return setups

    @staticmethod
    def calculate_position_size(
        capital: float,
        entry: float,
        stop: float,
        risk_per_trade_pct: float = 1.0
    ) -> tuple[int, float]:
        """
        Calculate position size based on risk management.

        Args:
            capital: Total capital
            entry: Entry price
            stop: Stop-loss price
            risk_per_trade_pct: Max risk per trade as % of capital

        Returns:
            (shares, position_value)
        """
        risk_amount = capital * (risk_per_trade_pct / 100)
        risk_per_share = abs(entry - stop)

        if risk_per_share <= 0:
            return 0, 0

        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry

        # Cap at 10% of capital per position
        max_position = capital * 0.10
        if position_value > max_position:
            shares = int(max_position / entry)
            position_value = shares * entry

        return shares, round(position_value, 2)
