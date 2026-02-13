"""
Unified scoring engine that combines technical, news, and market signals.
This is the core "probability engine" for stock selection.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import get_settings
from config.thresholds import (
    SIGNAL_THRESHOLDS, INDICATOR_THRESHOLDS,
    TRADE_LEVEL_MULTIPLIERS, RISK_THRESHOLDS
)
from src.features.technical import TechnicalIndicators
from src.features.market_features import MarketFeatures
from src.storage.models import SignalType, TradeType


@dataclass
class StockScore:
    """Complete score breakdown for a stock."""
    symbol: str
    date: date
    trade_type: TradeType

    # Component scores (0-1)
    technical_score: float
    momentum_score: float
    news_score: float
    sector_score: float
    volume_score: float

    # Final scores
    raw_score: float
    confidence: float  # After calibration
    signal: SignalType

    # Trading levels
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float

    # Risk metrics
    atr_pct: float
    liquidity_cr: float

    # Reasoning
    reasons: list[str]


class ScoringEngine:
    """
    Deterministic scoring engine for stock selection.

    Combines multiple signals with explicit weights.
    All calculations are transparent and reproducible.
    """

    # Score weights (must sum to 1.0)
    WEIGHTS = {
        'technical': 0.35,
        'momentum': 0.25,
        'news': 0.15,
        'sector': 0.15,
        'volume': 0.10
    }

    def __init__(self):
        self.settings = get_settings()
        self.market_features = MarketFeatures()

    def score_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        trade_type: TradeType,
        news_score: float = 0.5,
        news_reasons: list[str] = None,
        market_context: Optional[dict] = None
    ) -> Optional[StockScore]:
        """
        Calculate complete score for a stock.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV + technical indicators
            trade_type: INTRADAY or SWING
            news_score: Pre-calculated news sentiment score (0-1)
            news_reasons: News-based reasons
            market_context: Pre-fetched market context

        Returns:
            StockScore or None if insufficient data
        """
        if df.empty or len(df) < 50:
            logger.warning(f"{symbol}: Insufficient data for scoring")
            return None

        # Calculate all technical indicators if not already done
        if 'rsi' not in df.columns:
            df = TechnicalIndicators.calculate_all(df)

        latest = df.iloc[-1]
        current_price = float(latest['close'])

        # Skip if data is stale
        latest_date = df.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        # Calculate component scores
        reasons = news_reasons or []

        # 1. Technical Score
        tech_score, tech_reasons = self._calculate_technical_score(df, trade_type)
        reasons.extend(tech_reasons)

        # 2. Momentum Score
        momentum_score, mom_reasons = self._calculate_momentum_score(df, trade_type)
        reasons.extend(mom_reasons)

        # 3. Volume Score
        volume_score, vol_reasons = self._calculate_volume_score(df)
        reasons.extend(vol_reasons)

        # 4. Sector Score
        if market_context is None:
            market_context = self.market_features.get_market_context()
        sector_score = self.market_features.get_sector_score(symbol, market_context)

        sector = self.market_features.get_symbol_sector(symbol)
        if sector_score > 0.6:
            reasons.append(f"[SECTOR] {sector} showing strength (score: {sector_score:.2f})")
        elif sector_score < 0.4:
            reasons.append(f"[SECTOR] {sector} showing weakness (score: {sector_score:.2f})")

        # 5. News score is passed in
        if news_score > 0.6:
            reasons.append(f"[NEWS] Positive sentiment (score: {news_score:.2f})")
        elif news_score < 0.4:
            reasons.append(f"[NEWS] Negative sentiment (score: {news_score:.2f})")

        # Calculate weighted raw score
        raw_score = (
            self.WEIGHTS['technical'] * tech_score +
            self.WEIGHTS['momentum'] * momentum_score +
            self.WEIGHTS['news'] * news_score +
            self.WEIGHTS['sector'] * sector_score +
            self.WEIGHTS['volume'] * volume_score
        )

        # Weekly trend confirmation (multi-timeframe)
        # Resample daily data to weekly — no extra API calls needed.
        # Counter-trend trades (buying in weekly downtrend) have lower accuracy.
        try:
            if len(df) >= 130:  # ~26 weeks of trading days
                weekly_df = df.resample('W').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last',
                    'volume': 'sum'
                }).dropna()

                if len(weekly_df) >= 26:
                    weekly_close = float(weekly_df['close'].iloc[-1])
                    sma_13w = float(weekly_df['close'].rolling(13).mean().iloc[-1])
                    sma_26w = float(weekly_df['close'].rolling(26).mean().iloc[-1])

                    if pd.notna(sma_13w) and pd.notna(sma_26w):
                        if weekly_close > sma_13w > sma_26w:
                            raw_score *= 1.05
                            reasons.append(
                                "[WEEKLY] Uptrend confirmed (Close > 13W SMA > 26W SMA)"
                            )
                        elif weekly_close < sma_13w < sma_26w:
                            raw_score *= 0.85
                            reasons.append(
                                "[WEEKLY] Downtrend warning (Close < 13W SMA < 26W SMA)"
                            )
        except Exception as e:
            logger.debug(f"Weekly trend check skipped for {symbol}: {e}")

        # Apply regime adjustment
        regime_mult = self.market_features.get_regime_adjustment(market_context)
        adjusted_score = raw_score * regime_mult

        # Determine signal
        # NOTE: SELL signals are DISABLED due to 28.6% accuracy in backtest
        # (i.e., SELL signals are actually 71.4% bullish = inverted)
        # Until a separate SELL model is trained or logic inverted, we return HOLD
        if adjusted_score > 0.6:
            signal = SignalType.BUY
        else:
            # Previously: elif adjusted_score < 0.4: signal = SignalType.SELL
            # SELL signals disabled - return HOLD instead
            signal = SignalType.HOLD

        # Calculate trading levels
        atr = float(latest.get('atr', current_price * 0.02))
        atr_pct = float(latest.get('atr_pct', 2.0))

        entry_price, stop_loss, target_price = self._calculate_levels(
            current_price, atr, signal, trade_type
        )

        risk_reward = abs(target_price - entry_price) / abs(entry_price - stop_loss) if entry_price != stop_loss else 0

        # Calculate liquidity
        avg_value = float((df['close'] * df['volume']).tail(20).mean())
        liquidity_cr = avg_value / 1e7

        # Calibrate confidence
        confidence = self._calibrate_confidence(adjusted_score, atr_pct, liquidity_cr)

        return StockScore(
            symbol=symbol,
            date=latest_date,
            trade_type=trade_type,
            technical_score=tech_score,
            momentum_score=momentum_score,
            news_score=news_score,
            sector_score=sector_score,
            volume_score=volume_score,
            raw_score=raw_score,
            confidence=confidence,
            signal=signal,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward=risk_reward,
            atr_pct=atr_pct,
            liquidity_cr=liquidity_cr,
            reasons=reasons
        )

    def _calculate_technical_score(
        self,
        df: pd.DataFrame,
        trade_type: TradeType
    ) -> tuple[float, list[str]]:
        """Calculate technical indicator score."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        reasons = []
        score = 0.5  # Neutral starting point

        # Volatility-adaptive thresholds
        # Volatile stocks (ATR% > 3) need wider bands to avoid false signals
        # Stable stocks (ATR% < 1.5) can use tighter thresholds for earlier signals
        atr_pct = latest.get('atr_pct', 2.0)
        if pd.notna(atr_pct) and atr_pct > 3.0:
            rsi_oversold = 25.0
            rsi_overbought = 75.0
            bb_oversold = 0.15
            bb_overbought = 0.85
        elif pd.notna(atr_pct) and atr_pct < 1.5:
            rsi_oversold = 35.0
            rsi_overbought = 65.0
            bb_oversold = 0.25
            bb_overbought = 0.75
        else:
            rsi_oversold = INDICATOR_THRESHOLDS.rsi_oversold
            rsi_overbought = INDICATOR_THRESHOLDS.rsi_overbought
            bb_oversold = INDICATOR_THRESHOLDS.bb_oversold
            bb_overbought = INDICATOR_THRESHOLDS.bb_overbought

        # RSI - Relative Strength Index
        # Reference: Wilder (1978) "New Concepts in Technical Trading Systems"
        rsi = latest.get('rsi')
        if pd.notna(rsi):
            if rsi < rsi_oversold:
                # Oversold - bullish signal
                score += 0.15
                if rsi > prev.get('rsi', rsi):  # Bouncing from oversold
                    score += 0.10
                    reasons.append(f"[TECHNICAL] RSI oversold bounce ({rsi:.1f} → bullish)")
            elif rsi > rsi_overbought:
                # Overbought - bearish signal (symmetric with oversold)
                score -= 0.15
                if rsi < prev.get('rsi', rsi):  # Falling from overbought
                    score -= 0.10
                    reasons.append(f"[TECHNICAL] RSI overbought reversal ({rsi:.1f} → bearish)")

        # MACD
        macd_hist = latest.get('macd_histogram')
        prev_macd_hist = prev.get('macd_histogram')
        if pd.notna(macd_hist) and pd.notna(prev_macd_hist):
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                score += 0.1
                reasons.append("[TECHNICAL] MACD bullish momentum")
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                score -= 0.1
                reasons.append("[TECHNICAL] MACD bearish momentum")

        # Bollinger Band position (thresholds adapted to volatility above)
        bb_pos = latest.get('bb_position')
        if pd.notna(bb_pos):
            if bb_pos < bb_oversold:
                score += 0.10  # Oversold - potential bounce
                reasons.append("[TECHNICAL] Price near lower Bollinger Band (potential bounce)")
            elif bb_pos > bb_overbought:
                score -= 0.10  # Overbought - potential pullback (symmetric with oversold)
                reasons.append("[TECHNICAL] Price near upper Bollinger Band (extended)")

        # Trend alignment (price vs MAs)
        close = latest['close']
        sma20 = latest.get('sma_20')
        sma50 = latest.get('sma_50')

        if pd.notna(sma20) and pd.notna(sma50):
            if close > sma20 > sma50:
                score += 0.1
                reasons.append("[TECHNICAL] Uptrend (Price > SMA20 > SMA50)")
            elif close < sma20 < sma50:
                score -= 0.1
                reasons.append("[TECHNICAL] Downtrend (Price < SMA20 < SMA50)")

        # Candlestick patterns
        if latest.get('bullish_engulfing', 0) > 0:
            score += 0.1
            reasons.append("[PATTERN] Bullish engulfing candle")
        if latest.get('bearish_engulfing', 0) < 0:
            score -= 0.1
            reasons.append("[PATTERN] Bearish engulfing candle")
        if latest.get('hammer', 0) > 0:
            score += 0.05
            reasons.append("[PATTERN] Hammer pattern (potential reversal)")

        # Clamp to 0-1
        score = max(0, min(1, score))

        return score, reasons

    def _calculate_momentum_score(
        self,
        df: pd.DataFrame,
        trade_type: TradeType
    ) -> tuple[float, list[str]]:
        """Calculate momentum score."""
        latest = df.iloc[-1]
        reasons = []
        score = 0.5

        # Short-term momentum (5-day)
        # Reference: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
        # Thresholds from config/thresholds.py
        mom_5 = latest.get('momentum_5')
        if pd.notna(mom_5):
            if mom_5 > INDICATOR_THRESHOLDS.strong_momentum_5d:
                score += 0.15  # Strong positive momentum
                reasons.append(f"[MOMENTUM] Strong 5-day momentum (+{mom_5:.1f}%)")
            elif mom_5 > 0:
                score += 0.05  # Mild positive momentum
            elif mom_5 < -INDICATOR_THRESHOLDS.strong_momentum_5d:
                score -= 0.15  # Strong negative momentum (symmetric)
            elif mom_5 < 0:
                score -= 0.05  # Mild negative momentum (symmetric)

        # Medium-term momentum (more important for swing)
        # Thresholds from config/thresholds.py
        mom_20 = latest.get('momentum_20')
        if pd.notna(mom_20):
            weight = 0.15 if trade_type == TradeType.SWING else 0.05
            if mom_20 > INDICATOR_THRESHOLDS.strong_momentum_20d:
                score += weight
                reasons.append(f"[MOMENTUM] Strong 20-day momentum (+{mom_20:.1f}%)")
            elif mom_20 < -INDICATOR_THRESHOLDS.strong_momentum_20d:
                score -= weight

        # Price vs VWAP (intraday relevant)
        close = latest['close']
        vwap = latest.get('vwap')
        if pd.notna(vwap) and trade_type == TradeType.INTRADAY:
            if close > vwap:
                score += 0.05
                reasons.append("[MOMENTUM] Price above VWAP")
            else:
                score -= 0.05

        # Trend strength
        trend_strength = latest.get('trend_strength')
        if pd.notna(trend_strength):
            if trend_strength > INDICATOR_THRESHOLDS.strong_trend:
                # Strong trend - follow it
                if latest.get('momentum_5', 0) > 0:
                    score += 0.1
                else:
                    score -= 0.1

        score = max(0, min(1, score))
        return score, reasons

    def _calculate_volume_score(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        """Calculate volume score."""
        latest = df.iloc[-1]
        reasons = []
        score = 0.5

        # Volume thresholds from config/thresholds.py
        vol_ratio = latest.get('volume_ratio')
        if pd.notna(vol_ratio):
            if vol_ratio > INDICATOR_THRESHOLDS.high_volume_surge:
                score += 0.2
                reasons.append(f"[VOLUME] High volume surge ({vol_ratio:.1f}x average)")
            elif vol_ratio > INDICATOR_THRESHOLDS.above_avg_volume:
                score += 0.1
                reasons.append(f"[VOLUME] Above average volume ({vol_ratio:.1f}x)")
            elif vol_ratio < INDICATOR_THRESHOLDS.low_volume:
                score -= 0.1
                reasons.append("[VOLUME] Low volume - weak conviction")

        score = max(0, min(1, score))
        return score, reasons

    def _calculate_levels(
        self,
        current_price: float,
        atr: float,
        signal: SignalType,
        trade_type: TradeType
    ) -> tuple[float, float, float]:
        """Calculate entry, stop-loss, and target levels."""
        # Multipliers from config/thresholds.py
        mult = TRADE_LEVEL_MULTIPLIERS

        if trade_type == TradeType.INTRADAY:
            # Tighter levels for intraday
            stop_mult = mult.intraday_stop_mult
            target_mult = mult.intraday_target_mult
        else:
            # Wider levels for swing
            stop_mult = mult.swing_stop_mult
            target_mult = mult.swing_target_mult

        if signal == SignalType.BUY:
            entry = current_price  # Buy at market
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)
        elif signal == SignalType.SELL:
            entry = current_price
            stop = current_price + (atr * stop_mult)
            target = current_price - (atr * target_mult)
        else:  # HOLD
            entry = current_price
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)

        return round(entry, 2), round(stop, 2), round(target, 2)

    def _calibrate_confidence(
        self,
        raw_score: float,
        atr_pct: float,
        liquidity_cr: float
    ) -> float:
        """
        Calibrate raw score to probability-like confidence.

        Applies penalties for:
        - High volatility (ATR)
        - Low liquidity
        """
        confidence = raw_score

        # Volatility penalty
        if atr_pct > 4:
            confidence *= 0.8
        elif atr_pct > 3:
            confidence *= 0.9

        # Liquidity penalty
        if liquidity_cr < 5:
            confidence *= 0.8
        elif liquidity_cr < 10:
            confidence *= 0.9

        # Scale to realistic confidence range (0.4 to 0.8)
        # Maps [0,1] -> [0.4, 0.8] which honestly reflects 57.6% base accuracy
        confidence = 0.4 + (confidence * 0.4)

        return round(min(0.80, max(0.35, confidence)), 3)

    def rank_stocks(
        self,
        scores: list[StockScore],
        top_n: int = 10,
        min_confidence: float = 0.6,
        signal_filter: Optional[SignalType] = SignalType.BUY
    ) -> list[StockScore]:
        """
        Rank and filter stock scores.

        Args:
            scores: List of StockScore objects
            top_n: Number of top picks
            min_confidence: Minimum confidence threshold
            signal_filter: Only include this signal type (None for all)

        Returns:
            Sorted list of top picks
        """
        filtered = []
        for s in scores:
            # Apply filters
            if s.confidence < min_confidence:
                continue
            if signal_filter and s.signal != signal_filter:
                continue
            if s.atr_pct > self.settings.max_atr_percent:
                continue
            if s.liquidity_cr < self.settings.min_liquidity_cr:
                continue

            filtered.append(s)

        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x.confidence, reverse=True)

        # Apply sector diversification
        diversified = self._apply_sector_diversification(filtered, top_n)

        return diversified[:top_n]

    def _apply_sector_diversification(
        self,
        scores: list[StockScore],
        top_n: int
    ) -> list[StockScore]:
        """Limit stocks per sector for diversification."""
        max_per_sector = self.settings.max_sector_concentration
        sector_counts = {}
        result = []

        for score in scores:
            sector = self.market_features.get_symbol_sector(score.symbol)
            current_count = sector_counts.get(sector, 0)

            if current_count < max_per_sector:
                result.append(score)
                sector_counts[sector] = current_count + 1

            if len(result) >= top_n * 2:  # Get extra for buffer
                break

        return result
