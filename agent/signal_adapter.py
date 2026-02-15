"""
Signal Adapter - Bridges existing stock-predict signals with trading decisions.

This module connects the existing SignalGenerator from src/signals/generator.py
with the trading agent, translating predictions into actionable trade signals.

Filters applied:
- Min confidence (CONFIG.signals.min_confidence)
- Min risk:reward (CONFIG.signals.min_risk_reward)
- Supertrend confirmation (direction must be uptrend for BUY)
- ADX filter (trend strength must exceed CONFIG.strategy.adx_strong_trend)
"""
from dataclasses import dataclass, field
from datetime import datetime, time, date
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import pandas as pd
from loguru import logger

from config.trading_config import CONFIG
from utils.platform import now_ist, today_ist, time_ist
from src.signals.generator import SignalGenerator
from src.data.price_fetcher import PriceFetcher
from src.features.technical import TechnicalIndicators
from features.supertrend import get_supertrend_signal
from src.storage.models import TradeType, SignalType


class TradeDecision(Enum):
    """Final trading decision."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"


@dataclass
class TradeSignal:
    """
    Actionable trade signal for the execution engine.
    """
    symbol: str
    decision: TradeDecision
    confidence: float

    # Prices
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float

    # Risk metrics
    risk_reward_ratio: float
    atr_pct: float
    position_size_pct: float

    # Liquidity
    avg_daily_volume: int = 0

    # Context
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: now_ist().replace(tzinfo=None))
    source_prediction: Optional[Dict[str, Any]] = None

    @property
    def risk_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100

    @property
    def reward_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        return abs(self.target_price - self.entry_price) / self.entry_price * 100


class SignalAdapter:
    """
    Adapts stock-predict signals to trading decisions.

    All thresholds read from CONFIG (env-driven).
    """

    def __init__(self, signal_generator: Optional[SignalGenerator] = None):
        self._generator = signal_generator
        self._price_fetcher = PriceFetcher()

        # Cache to avoid duplicate API calls
        self._last_run: Optional[datetime] = None
        self._cached_signals: Dict[str, TradeSignal] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._price_cache_time: Optional[datetime] = None

        logger.info("SignalAdapter initialized")

    @property
    def generator(self) -> SignalGenerator:
        if self._generator is None:
            self._generator = SignalGenerator()
        return self._generator

    def get_trade_signals(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> List[TradeSignal]:
        """Get actionable trade signals."""
        if not force_refresh and self._is_cache_valid():
            logger.debug("Using cached signals")
            if symbols:
                return [s for s in self._cached_signals.values() if s.symbol in symbols]
            return list(self._cached_signals.values())

        logger.info("Running signal generator...")
        try:
            results = self.generator.run(
                symbols=symbols,
                include_intraday=True,
                include_swing=False
            )
        except Exception as e:
            logger.error(f"Signal generator failed: {e}")
            return []

        signals = []
        predictions = results.get('intraday', {}).get('predictions', [])

        for pred in predictions:
            signal = self._convert_prediction(pred)
            if signal and self._passes_filters(signal):
                signals.append(signal)
                self._cached_signals[signal.symbol] = signal

        self._last_run = now_ist().replace(tzinfo=None)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"Generated {len(signals)} trade signals from {len(predictions)} predictions")
        return signals

    def get_single_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Get trade signal for a single symbol."""
        try:
            pred = self.generator.get_single_prediction(
                symbol=symbol,
                trade_type=TradeType.INTRADAY
            )
            if not pred:
                return None

            signal = self._convert_prediction(pred)
            if signal and self._passes_filters(signal):
                return signal
            return None

        except Exception as e:
            logger.error(f"Failed to get signal for {symbol}: {e}")
            return None

    def _convert_prediction(self, pred: Dict[str, Any]) -> Optional[TradeSignal]:
        """Convert a prediction dict to TradeSignal."""
        try:
            signal_type = pred.get('signal', '').upper()

            if signal_type == 'BUY' and CONFIG.signals.enable_buy_signals:
                decision = TradeDecision.BUY
            elif signal_type == 'SELL' and CONFIG.signals.enable_sell_signals:
                decision = TradeDecision.SELL
            elif signal_type == 'HOLD':
                decision = TradeDecision.HOLD
            else:
                decision = TradeDecision.WAIT

            entry = pred.get('entry_price', pred.get('current_price', 0))
            stop = pred.get('stop_loss', 0)
            target = pred.get('target_price', 0)

            if entry and stop and target and entry != stop:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                # Deduct estimated round-trip fees from reward
                estimated_fees = entry * CONFIG.capital.estimated_fee_pct * 2
                reward = max(0, reward - estimated_fees)
                rr_ratio = reward / risk if risk > 0 else 0
            else:
                rr_ratio = 0

            confidence = pred.get('confidence', 0)
            position_pct = self._calculate_position_size(confidence, rr_ratio)

            return TradeSignal(
                symbol=pred.get('symbol', ''),
                decision=decision,
                confidence=confidence,
                current_price=pred.get('current_price', 0),
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                risk_reward_ratio=rr_ratio,
                atr_pct=0,
                position_size_pct=position_pct,
                reasons=pred.get('reasons', []),
                source_prediction=pred
            )

        except Exception as e:
            logger.warning(f"Failed to convert prediction: {e}")
            return None

    def _passes_filters(self, signal: TradeSignal) -> bool:
        """Apply filters from CONFIG, Supertrend confirmation, and ADX filter."""
        if signal.decision not in (TradeDecision.BUY, TradeDecision.SELL):
            return False

        if signal.confidence < CONFIG.signals.min_confidence:
            logger.debug(f"{signal.symbol}: Low confidence {signal.confidence:.2f}")
            return False

        if signal.risk_reward_ratio < CONFIG.signals.min_risk_reward:
            logger.debug(f"{signal.symbol}: Low R:R {signal.risk_reward_ratio:.2f}")
            return False

        now = time_ist()
        if not (CONFIG.hours.entry_window_start <= now <= CONFIG.hours.entry_window_end):
            logger.debug(f"{signal.symbol}: Outside entry window")
            return False

        if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.target_price <= 0:
            return False

        if signal.decision == TradeDecision.BUY:
            if signal.stop_loss >= signal.entry_price:
                return False
            if signal.target_price <= signal.entry_price:
                return False

        # Volume/liquidity check (also populates signal.avg_daily_volume)
        if not self._passes_volume_check(signal.symbol, signal=signal):
            logger.info(f"{signal.symbol}: REJECTED by volume filter (illiquid)")
            return False

        # Corporate action detection (skip stocks with suspected splits/bonuses)
        if self._detect_corporate_action(signal.symbol):
            logger.info(f"{signal.symbol}: REJECTED - suspected corporate action (split/bonus)")
            return False

        # Supertrend confirmation + ADX filter
        st_direction, adx_ok = self._check_trend_filters(signal.symbol)

        # Supertrend: BUY requires uptrend (1), SELL requires downtrend (-1)
        # Exception: mean-reversion BUY allowed when RSI is oversold (< rsi_oversold)
        if signal.decision == TradeDecision.BUY and st_direction != 1:
            rsi = self._get_rsi(signal.symbol)
            if rsi is not None and rsi < CONFIG.strategy.rsi_oversold:
                logger.info(
                    f"{signal.symbol}: Supertrend downtrend but RSI={rsi:.1f} (oversold) - "
                    f"allowing mean-reversion BUY with reduced confidence"
                )
                signal.confidence *= 0.85
                signal.reasons.append(
                    f"[FILTER] Mean-reversion: RSI={rsi:.1f} oversold, confidence reduced 0.85x"
                )
            else:
                logger.info(f"{signal.symbol}: REJECTED by Supertrend (downtrend, need uptrend for BUY)")
                signal.reasons.append("[FILTER] Rejected: Supertrend in downtrend")
                return False
        elif signal.decision == TradeDecision.SELL and st_direction != -1:
            logger.info(f"{signal.symbol}: REJECTED by Supertrend (uptrend, need downtrend for SELL)")
            signal.reasons.append("[FILTER] Rejected: Supertrend in uptrend")
            return False

        if not adx_ok:
            logger.info(f"{signal.symbol}: REJECTED by ADX (weak trend)")
            signal.reasons.append("[FILTER] Rejected: ADX below threshold (weak trend)")
            return False

        trend_str = "uptrend" if st_direction == 1 else "downtrend"
        signal.reasons.append(f"[FILTER] Supertrend confirmed {trend_str}")
        return True

    def _check_trend_filters(self, symbol: str) -> Tuple[int, bool]:
        """
        Check Supertrend direction and ADX strength for a symbol.

        Returns:
            Tuple of (supertrend_direction, adx_ok)
            supertrend_direction: 1=uptrend, -1=downtrend, 0=unknown
        """
        try:
            df = self._get_price_data(symbol)
            if df is None or len(df) < 50:
                # Insufficient data - reject (conservative approach)
                logger.debug(f"{symbol}: Insufficient price data for trend filters")
                return 0, False

            # Supertrend check
            st_info = get_supertrend_signal(
                df,
                period=CONFIG.strategy.supertrend_period,
                multiplier=CONFIG.strategy.supertrend_multiplier
            )
            st_direction = st_info['direction']  # 1=up, -1=down

            # ADX check
            if 'trend_strength' not in df.columns:
                df = TechnicalIndicators.calculate_all(df)
            adx_value = float(df['trend_strength'].iloc[-1]) if pd.notna(df['trend_strength'].iloc[-1]) else 0
            adx_ok = adx_value >= CONFIG.strategy.adx_strong_trend

            logger.debug(
                f"{symbol}: Supertrend={'UP' if st_direction == 1 else 'DOWN'}, "
                f"ADX={adx_value:.1f} (threshold={CONFIG.strategy.adx_strong_trend})"
            )

            return st_direction, adx_ok

        except Exception as e:
            logger.warning(f"Trend filter check failed for {symbol}: {e}")
            return 0, False  # Reject on error (conservative)

    def _get_rsi(self, symbol: str) -> Optional[float]:
        """Get current RSI for a symbol (for mean-reversion check)."""
        try:
            df = self._get_price_data(symbol)
            if df is None or len(df) < 20:
                return None
            if 'rsi' not in df.columns:
                df = TechnicalIndicators.calculate_all(df)
            rsi_val = df['rsi'].iloc[-1]
            return float(rsi_val) if pd.notna(rsi_val) else None
        except Exception as e:
            logger.debug(f"RSI check failed for {symbol}: {e}")
            return None

    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data for a symbol, using cache (expires after signal_cache_minutes).

        Tries primary source (PriceFetcher/yfinance) first, falls back to
        direct yfinance call if primary fails.
        """
        # Expire price cache when signal cache expires
        if self._price_cache_time:
            age_min = (now_ist().replace(tzinfo=None) - self._price_cache_time).total_seconds() / 60
            if age_min > CONFIG.signals.signal_cache_minutes:
                self._price_cache.clear()
                self._price_cache_time = None

        if symbol in self._price_cache:
            return self._price_cache[symbol]

        # Try generator's daily price cache first (avoids duplicate yfinance calls)
        if (self._generator is not None
                and hasattr(self._generator, '_daily_price_cache')
                and getattr(self._generator, '_daily_price_cache_date', None) == today_ist()
                and symbol in self._generator._daily_price_cache):
            df = self._generator._daily_price_cache[symbol]
            self._price_cache[symbol] = df
            if not self._price_cache_time:
                self._price_cache_time = now_ist().replace(tzinfo=None)
            return df

        # Primary source: PriceFetcher
        df = self._fetch_from_primary(symbol)

        # Fallback: direct yfinance if primary fails
        if df is None:
            df = self._fetch_from_fallback(symbol)

        if df is not None:
            self._price_cache[symbol] = df
            if not self._price_cache_time:
                self._price_cache_time = now_ist().replace(tzinfo=None)

        return df

    def _fetch_from_primary(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from primary data source (PriceFetcher)."""
        try:
            prices = self._price_fetcher.fetch_prices(symbol, period="3mo")
            if not prices:
                return None
            df = pd.DataFrame([p.model_dump() for p in prices])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except Exception as e:
            logger.warning(f"Primary data source failed for {symbol}: {e}")
            return None

    def _fetch_from_fallback(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fallback: direct yfinance call when primary source fails (with timeout).

        Rate-limited via quota manager to prevent yfinance abuse.
        """
        try:
            import yfinance as yf
            from utils.platform import yfinance_with_timeout
            from providers.quota import get_quota_manager

            quota = get_quota_manager()
            allowed, reason = quota.can_request("yfinance")
            if not allowed:
                logger.warning(f"Fallback yfinance blocked for {symbol}: {reason}")
                return None

            quota.wait_and_record("yfinance")

            def _fetch():
                ticker = yf.Ticker(f"{symbol}.NS")
                return ticker.history(period="3mo")

            df = yfinance_with_timeout(_fetch, timeout_seconds=30)

            if df.empty:
                logger.warning(f"Fallback data source also empty for {symbol}")
                quota.record_success("yfinance")
                return None

            quota.record_success("yfinance")
            df.columns = [c.lower() for c in df.columns]
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                return None
            logger.info(f"Using fallback data source for {symbol} ({len(df)} rows)")
            return df
        except Exception as e:
            try:
                from providers.quota import get_quota_manager
                get_quota_manager().record_failure("yfinance")
            except Exception:
                pass
            logger.error(f"Fallback data source also failed for {symbol}: {e}")
            return None

    def _calculate_position_size(self, confidence: float, rr_ratio: float) -> float:
        """Calculate position size from CONFIG values."""
        base = CONFIG.capital.base_position_pct
        confidence_mult = 0.5 + confidence
        rr_mult = min(1 + (rr_ratio - 1) * 0.25, 1.5)
        position = base * confidence_mult * rr_mult
        return min(position, CONFIG.capital.max_position_pct)

    def _passes_volume_check(self, symbol: str, signal: 'TradeSignal' = None) -> bool:
        """Check if stock has adequate daily volume for safe entry/exit."""
        min_volume_cr = CONFIG.signals.min_volume_cr
        try:
            df = self._get_price_data(symbol)
            if df is None or len(df) < 5:
                return False
            # Average volume over last 5 days
            avg_volume = df['volume'].tail(5).mean()
            avg_price = df['close'].tail(5).mean()
            if avg_volume <= 0 or avg_price <= 0:
                return False
            # Daily turnover in crores
            daily_turnover_cr = (avg_volume * avg_price) / 1e7
            if daily_turnover_cr < min_volume_cr:
                logger.debug(
                    f"{symbol}: Daily turnover ₹{daily_turnover_cr:.1f}Cr < "
                    f"min ₹{min_volume_cr:.0f}Cr"
                )
                return False
            # Populate avg volume on signal for downstream liquidity cap
            if signal is not None:
                signal.avg_daily_volume = int(avg_volume)
            return True
        except Exception as e:
            logger.debug(f"Volume check failed for {symbol}: {e}")
            return False

    def _detect_corporate_action(self, symbol: str) -> bool:
        """Detect suspected corporate actions (splits/bonuses) via overnight price jumps."""
        try:
            df = self._get_price_data(symbol)
            if df is None or len(df) < 3:
                return False
            # Check last 2 days for >40% overnight jump (typical of stock split)
            for i in range(-1, -min(3, len(df)), -1):
                if i - 1 < -len(df):
                    break
                prev_close = df['close'].iloc[i - 1]
                curr_open = df['open'].iloc[i]
                if prev_close > 0:
                    overnight_change = abs(curr_open - prev_close) / prev_close
                    if overnight_change > 0.40:
                        logger.warning(
                            f"{symbol}: Suspected corporate action - "
                            f"{overnight_change:.0%} overnight price change"
                        )
                        return True
            return False
        except Exception as e:
            logger.debug(f"Corporate action check failed for {symbol}: {e}")
            return False

    def _is_cache_valid(self) -> bool:
        if not self._last_run:
            return False
        age = (now_ist().replace(tzinfo=None) - self._last_run).total_seconds() / 60
        return age < CONFIG.signals.signal_cache_minutes

    def clear_cache(self):
        self._last_run = None
        self._cached_signals.clear()
        self._price_cache.clear()
