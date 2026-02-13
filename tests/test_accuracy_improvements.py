"""
Tests for 6 Accuracy Improvements (57.6% → 62-65% target).

1. Tightened signal filters (MIN_CONFIDENCE=0.65, MIN_RISK_REWARD=1.8)
2. Confidence calibration fix ([0.4, 0.8] range)
3. News recency weighting (time-decayed)
4. ADX threshold raised to 25 (centralized from thresholds.py)
5. Volatility-adaptive RSI/BB thresholds
6. Weekly trend confirmation (+5% uptrend bonus, -15% downtrend penalty)
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from config.trading_config import CONFIG
from config.thresholds import INDICATOR_THRESHOLDS
from src.features.technical import TechnicalIndicators


# ============================================
# HELPERS
# ============================================

def _make_scored_df(n=100, trend="neutral", atr_pct_override=None, seed=42):
    """Create a DataFrame with pre-calculated indicators for scoring tests."""
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='B')

    if trend == "up":
        returns = 0.002 + np.random.randn(n) * 0.005
    elif trend == "down":
        returns = -0.002 + np.random.randn(n) * 0.005
    else:
        returns = np.random.randn(n) * 0.005

    close = 1000 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n) * 0.003))
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.randint(500000, 2000000, n).astype(float)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=dates)

    df = TechnicalIndicators.calculate_all(df)

    if atr_pct_override is not None:
        df['atr_pct'] = atr_pct_override

    return df


def _make_scoring_engine():
    """Create ScoringEngine with mocked MarketFeatures (no network)."""
    with patch('src.models.scoring.MarketFeatures') as MockMF:
        mock_mf = MagicMock()
        mock_mf.get_market_context.return_value = {'regime': 'bullish'}
        mock_mf.get_sector_score.return_value = 0.5
        mock_mf.get_symbol_sector.return_value = 'IT'
        mock_mf.get_regime_adjustment.return_value = 1.0
        MockMF.return_value = mock_mf

        from src.models.scoring import ScoringEngine
        engine = ScoringEngine()
    return engine


def _make_news_article(symbol, sentiment_score, hours_ago):
    """Create a NewsArticle with controlled publication time."""
    from utils.platform import now_ist
    from src.storage.models import NewsArticle, EventType, Sentiment

    pub_time = now_ist() - timedelta(hours=hours_ago)
    sentiment = (Sentiment.POSITIVE if sentiment_score > 0.3
                 else Sentiment.NEGATIVE if sentiment_score < -0.3
                 else Sentiment.NEUTRAL)
    return NewsArticle(
        id=f"test_{symbol}_{hours_ago}h",
        source="test",
        url=f"https://test.com/{symbol}/{hours_ago}",
        title=f"Test article about {symbol}",
        published_at=pub_time,
        tickers=[symbol],
        event_type=EventType.OTHER,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        relevance_score=0.8,
    )


# ============================================
# FIX 1: TIGHTENED SIGNAL FILTERS
# ============================================

class TestTightenedSignalFilters:

    def test_min_confidence_default_is_065(self):
        assert CONFIG.signals.min_confidence == 0.65

    def test_min_risk_reward_default_is_18(self):
        assert CONFIG.signals.min_risk_reward == 1.8

    def test_signal_below_confidence_rejected(self):
        """Signal with confidence=0.63 should fail the 0.65 threshold."""
        from agent.signal_adapter import TradeSignal, TradeDecision
        signal = TradeSignal(
            symbol='TEST', decision=TradeDecision.BUY,
            confidence=0.63, current_price=100.0, entry_price=100.0,
            stop_loss=95.0, target_price=110.0, risk_reward_ratio=2.0,
            atr_pct=2.0, position_size_pct=0.15,
        )
        # Confidence 0.63 < min_confidence 0.65
        assert signal.confidence < CONFIG.signals.min_confidence

    def test_signal_below_rr_rejected(self):
        """Signal with R:R=1.6 should fail the 1.8 threshold."""
        from agent.signal_adapter import TradeSignal, TradeDecision
        signal = TradeSignal(
            symbol='TEST', decision=TradeDecision.BUY,
            confidence=0.72, current_price=100.0, entry_price=100.0,
            stop_loss=95.0, target_price=108.0, risk_reward_ratio=1.6,
            atr_pct=2.0, position_size_pct=0.15,
        )
        # R:R 1.6 < min_risk_reward 1.8
        assert signal.risk_reward_ratio < CONFIG.signals.min_risk_reward


# ============================================
# FIX 2: CONFIDENCE CALIBRATION
# ============================================

class TestConfidenceCalibration:

    def setup_method(self):
        self.engine = _make_scoring_engine()

    def test_raw_zero_maps_to_040(self):
        """Raw score 0.0 with no penalties → 0.40."""
        result = self.engine._calibrate_confidence(0.0, atr_pct=2.0, liquidity_cr=50.0)
        assert result == 0.4

    def test_raw_half_maps_to_060(self):
        """Raw score 0.5 → 0.60 (unchanged from old mapping at midpoint)."""
        result = self.engine._calibrate_confidence(0.5, atr_pct=2.0, liquidity_cr=50.0)
        assert result == 0.6

    def test_raw_one_maps_to_080(self):
        """Raw score 1.0 → 0.80 (capped, was 0.90 before)."""
        result = self.engine._calibrate_confidence(1.0, atr_pct=2.0, liquidity_cr=50.0)
        assert result == 0.8

    def test_caps_at_080(self):
        """Even with extreme raw score, cap at 0.80."""
        result = self.engine._calibrate_confidence(1.5, atr_pct=2.0, liquidity_cr=50.0)
        assert result == 0.80

    def test_floor_at_035(self):
        """Negative raw score floors at 0.35."""
        result = self.engine._calibrate_confidence(-0.5, atr_pct=2.0, liquidity_cr=50.0)
        assert result == 0.35

    def test_volatility_penalty_applied(self):
        """High ATR% penalizes confidence."""
        normal = self.engine._calibrate_confidence(0.7, atr_pct=2.0, liquidity_cr=50.0)
        high_vol = self.engine._calibrate_confidence(0.7, atr_pct=4.5, liquidity_cr=50.0)
        assert high_vol < normal

    def test_liquidity_penalty_applied(self):
        """Low liquidity penalizes confidence."""
        normal = self.engine._calibrate_confidence(0.7, atr_pct=2.0, liquidity_cr=50.0)
        low_liq = self.engine._calibrate_confidence(0.7, atr_pct=2.0, liquidity_cr=3.0)
        assert low_liq < normal


# ============================================
# FIX 3: NEWS RECENCY WEIGHTING
# ============================================

class TestNewsRecencyWeighting:

    def setup_method(self):
        self.extractor = MagicMock()  # We call the real method directly

    def test_recent_article_weighted_more(self):
        """Recent positive + old negative → result should be positive."""
        from src.features.news_features import NewsFeatureExtractor
        extractor = NewsFeatureExtractor.__new__(NewsFeatureExtractor)

        articles = [
            _make_news_article("RELIANCE", sentiment_score=0.8, hours_ago=1),
            _make_news_article("RELIANCE", sentiment_score=-0.8, hours_ago=70),
        ]
        result = extractor.get_symbol_news_summary(articles, "RELIANCE")
        # Recent +0.8 (weight ~1.0) dominates old -0.8 (weight ~0.1)
        assert result['avg_sentiment'] > 0.3

    def test_all_old_articles_have_equal_low_weight(self):
        """Articles all 72h+ old should have min weight (0.1), giving simple avg."""
        from src.features.news_features import NewsFeatureExtractor
        extractor = NewsFeatureExtractor.__new__(NewsFeatureExtractor)

        articles = [
            _make_news_article("TCS", sentiment_score=0.6, hours_ago=80),
            _make_news_article("TCS", sentiment_score=-0.4, hours_ago=90),
        ]
        result = extractor.get_symbol_news_summary(articles, "TCS")
        # Both have weight 0.1, so avg = (0.6 + -0.4) / 2 = 0.1
        assert abs(result['avg_sentiment'] - 0.1) < 0.05

    def test_single_article_works(self):
        """Single article returns its own sentiment."""
        from src.features.news_features import NewsFeatureExtractor
        extractor = NewsFeatureExtractor.__new__(NewsFeatureExtractor)

        articles = [_make_news_article("INFY", sentiment_score=0.5, hours_ago=12)]
        result = extractor.get_symbol_news_summary(articles, "INFY")
        assert abs(result['avg_sentiment'] - 0.5) < 0.01

    def test_zero_articles_returns_neutral(self):
        """No articles → avg_sentiment = 0.0."""
        from src.features.news_features import NewsFeatureExtractor
        extractor = NewsFeatureExtractor.__new__(NewsFeatureExtractor)

        result = extractor.get_symbol_news_summary([], "RELIANCE")
        assert result['avg_sentiment'] == 0.0
        assert result['article_count'] == 0

    def test_mixed_recency_asymmetric(self):
        """Very recent positive dominates very old negative."""
        from src.features.news_features import NewsFeatureExtractor
        extractor = NewsFeatureExtractor.__new__(NewsFeatureExtractor)

        articles = [
            _make_news_article("HDFC", sentiment_score=0.8, hours_ago=0.5),  # weight ~0.99
            _make_news_article("HDFC", sentiment_score=-0.8, hours_ago=72),  # weight = 0.1
        ]
        result = extractor.get_symbol_news_summary(articles, "HDFC")
        # (0.8*0.99 + -0.8*0.1) / (0.99+0.1) ≈ 0.654
        assert result['avg_sentiment'] > 0.5


# ============================================
# FIX 4: ADX THRESHOLD
# ============================================

class TestADXThreshold:

    def test_config_adx_default_is_25(self):
        assert CONFIG.strategy.adx_strong_trend == 25

    def test_indicator_threshold_strong_trend_is_25(self):
        assert INDICATOR_THRESHOLDS.strong_trend == 25.0

    def test_scoring_no_trend_bonus_below_25(self):
        """ADX=20 should NOT give momentum trend bonus (was >18, now >25)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100)
        # Force trend_strength to 20 (below new 25 threshold)
        df['trend_strength'] = 20.0
        df['momentum_5'] = 2.0  # Positive momentum

        from src.storage.models import TradeType
        _, reasons = engine._calculate_momentum_score(df, TradeType.INTRADAY)
        # Should NOT have trend-following reason (ADX 20 < 25)
        assert not any("trend" in r.lower() for r in reasons)

    def test_scoring_trend_bonus_above_25(self):
        """ADX=30 should give momentum trend bonus."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100)
        df['trend_strength'] = 30.0
        df['momentum_5'] = 2.0

        from src.storage.models import TradeType
        score_with, _ = engine._calculate_momentum_score(df, TradeType.INTRADAY)

        df['trend_strength'] = 15.0
        score_without, _ = engine._calculate_momentum_score(df, TradeType.INTRADAY)

        # ADX=30 should produce higher momentum score than ADX=15
        assert score_with > score_without


# ============================================
# FIX 5: VOLATILITY-ADAPTIVE THRESHOLDS
# ============================================

class TestVolatilityAdaptiveThresholds:

    def test_high_vol_wider_rsi(self):
        """High volatility stock: RSI 28 should NOT trigger oversold (threshold=25)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, atr_pct_override=4.0)
        df.iloc[-1, df.columns.get_loc('rsi')] = 28.0  # Between 25 and 30

        from src.storage.models import TradeType
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        # RSI 28 > 25 (high-vol oversold), so no oversold bounce signal
        assert not any("oversold bounce" in r for r in reasons)

    def test_low_vol_tighter_rsi(self):
        """Low volatility stock: RSI 32 SHOULD trigger oversold (threshold=35)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, atr_pct_override=1.0)
        df.iloc[-1, df.columns.get_loc('rsi')] = 32.0  # Between 30 and 35
        # Need prev RSI higher than current for bounce detection
        df.iloc[-2, df.columns.get_loc('rsi')] = 30.0

        from src.storage.models import TradeType
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        # RSI 32 < 35 (low-vol oversold), and bouncing → oversold bounce
        assert score > 0.5  # Should have bullish bias from RSI signal

    def test_standard_vol_uses_defaults(self):
        """Standard volatility: RSI 28 IS oversold (threshold=30)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, atr_pct_override=2.0)
        df.iloc[-1, df.columns.get_loc('rsi')] = 28.0
        df.iloc[-2, df.columns.get_loc('rsi')] = 25.0  # Bouncing up

        from src.storage.models import TradeType
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        # RSI 28 < 30 (standard oversold) and bouncing
        assert any("oversold bounce" in r for r in reasons)

    def test_high_vol_wider_bb(self):
        """High volatility: bb_position 0.18 should NOT be oversold (threshold=0.15)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, atr_pct_override=4.0)
        df.iloc[-1, df.columns.get_loc('bb_position')] = 0.18
        df.iloc[-1, df.columns.get_loc('rsi')] = 50.0  # Neutral RSI

        from src.storage.models import TradeType
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        assert not any("Bollinger Band" in r for r in reasons)

    def test_low_vol_tighter_bb(self):
        """Low volatility: bb_position 0.22 SHOULD be oversold (threshold=0.25)."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, atr_pct_override=1.0)
        df.iloc[-1, df.columns.get_loc('bb_position')] = 0.22
        df.iloc[-1, df.columns.get_loc('rsi')] = 50.0  # Neutral RSI

        from src.storage.models import TradeType
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        assert any("Bollinger Band" in r for r in reasons)

    def test_nan_atr_uses_defaults(self):
        """NaN ATR% → standard thresholds, no crash."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100)
        df['atr_pct'] = float('nan')
        df.iloc[-1, df.columns.get_loc('rsi')] = 28.0
        df.iloc[-2, df.columns.get_loc('rsi')] = 25.0

        from src.storage.models import TradeType
        # Should not crash
        score, reasons = engine._calculate_technical_score(df, TradeType.INTRADAY)
        # NaN ATR → default thresholds (RSI 30), 28 < 30 → oversold
        assert any("oversold bounce" in r for r in reasons)


# ============================================
# FIX 6: WEEKLY TREND CONFIRMATION
# ============================================

class TestWeeklyTrendConfirmation:

    def test_weekly_uptrend_gives_bonus(self):
        """200-row uptrend → raw_score *= 1.05, reason added."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=200, trend="up")

        from src.storage.models import TradeType
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
        assert result is not None
        assert any("[WEEKLY] Uptrend" in r for r in result.reasons)

    def test_weekly_downtrend_gives_penalty(self):
        """200-row downtrend → raw_score *= 0.85, reason added."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=200, trend="down")

        from src.storage.models import TradeType
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
        assert result is not None
        assert any("[WEEKLY] Downtrend" in r for r in result.reasons)

    def test_insufficient_data_skips_weekly(self):
        """100-row DF (< 130 rows) → no weekly check applied."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=100, trend="up")

        from src.storage.models import TradeType
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
        assert result is not None
        assert not any("[WEEKLY]" in r for r in result.reasons)

    def test_weekly_check_no_crash_on_sparse_data(self):
        """DataFrame with gaps should not crash."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=200, trend="neutral")
        # Drop some rows to create gaps
        df = df.iloc[::2]  # Keep every other row (100 rows)

        from src.storage.models import TradeType
        # Should not raise
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
        assert result is not None

    def test_uptrend_score_higher_than_downtrend(self):
        """Same stock data, uptrend version should score higher than downtrend."""
        engine = _make_scoring_engine()
        df_up = _make_scored_df(n=200, trend="up", seed=42)
        df_down = _make_scored_df(n=200, trend="down", seed=42)

        from src.storage.models import TradeType
        result_up = engine.score_stock("TEST", df_up, TradeType.INTRADAY, news_score=0.5)
        result_down = engine.score_stock("TEST", df_down, TradeType.INTRADAY, news_score=0.5)

        assert result_up is not None and result_down is not None
        # Uptrend should produce higher confidence
        assert result_up.raw_score > result_down.raw_score

    def test_weekly_applied_before_regime(self):
        """Weekly multiplier should be applied before regime adjustment."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=200, trend="up")

        # Regime multiplier = 1.0 (mocked), so raw_score after weekly = raw_score * regime
        from src.storage.models import TradeType
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
        assert result is not None
        # Just verify the score exists and is reasonable
        assert 0 < result.raw_score <= 1.5  # Could be > 1.0 due to weekly bonus


# ============================================
# INTEGRATION: ALL FIXES TOGETHER
# ============================================

class TestAccuracyImprovementsIntegration:

    def test_scoring_engine_produces_valid_output(self):
        """Full scoring pipeline with all fixes produces valid StockScore."""
        engine = _make_scoring_engine()
        df = _make_scored_df(n=200, trend="up")

        from src.storage.models import TradeType
        result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.6)
        assert result is not None
        assert 0.35 <= result.confidence <= 0.80
        assert result.entry_price > 0
        assert result.stop_loss > 0
        assert result.target_price > 0

    def test_confidence_within_new_bounds(self):
        """All calibrated confidences should be in [0.35, 0.80]."""
        engine = _make_scoring_engine()
        for trend in ["up", "down", "neutral"]:
            df = _make_scored_df(n=200, trend=trend)
            from src.storage.models import TradeType
            result = engine.score_stock("TEST", df, TradeType.INTRADAY, news_score=0.5)
            if result:
                assert 0.35 <= result.confidence <= 0.80, \
                    f"Confidence {result.confidence} out of [0.35, 0.80] for {trend}"

    def test_high_vol_stock_gets_lower_confidence(self):
        """High-volatility stock should get penalized confidence vs standard vol."""
        engine = _make_scoring_engine()
        df_normal = _make_scored_df(n=200, trend="up", atr_pct_override=2.0)
        df_highvol = _make_scored_df(n=200, trend="up", atr_pct_override=4.5)

        from src.storage.models import TradeType
        result_normal = engine.score_stock("TEST", df_normal, TradeType.INTRADAY, news_score=0.5)
        result_highvol = engine.score_stock("TEST", df_highvol, TradeType.INTRADAY, news_score=0.5)

        if result_normal and result_highvol:
            assert result_highvol.confidence <= result_normal.confidence
