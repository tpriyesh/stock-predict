"""
Tests for src/features/technical.py - TechnicalIndicators class.

Covers: calculate_all, rsi, macd, bollinger_bands, atr, candlestick_patterns.
Focuses on edge cases from production hardening: division-by-zero guards,
flat market data, zero volume, empty dataframes, and NaN handling.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalIndicators


class TestCalculateAllHappyPath:
    """Test calculate_all with normal OHLCV data."""

    def test_calculate_all_happy_path(self, sample_ohlcv_df):
        """sample_ohlcv_df (100 rows) -> key indicator columns exist and have values."""
        result = TechnicalIndicators.calculate_all(sample_ohlcv_df)

        # Core indicator columns must be present
        expected_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'atr_pct',
            'sma_20', 'sma_50', 'sma_200', 'ema_20',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'volume_sma_20', 'volume_ratio', 'vwap',
            'momentum_5', 'momentum_10', 'momentum_20',
            'support', 'resistance', 'trend_strength',
            'change', 'change_pct',
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Candlestick pattern columns
        pattern_columns = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'morning_star']
        for col in pattern_columns:
            assert col in result.columns, f"Missing pattern column: {col}"

        # Output has same number of rows as input
        assert len(result) == len(sample_ohlcv_df)

        # RSI is bounded [0, 100] (after fillna, 0 is also valid for early rows)
        assert result['rsi'].between(0, 100).all(), "RSI out of [0, 100] range"

        # ATR should be non-negative
        assert (result['atr'] >= 0).all(), "ATR has negative values"

        # Original columns preserved
        assert 'open' in result.columns
        assert 'close' in result.columns


class TestEdgeCases:
    """Test edge cases that previously caused crashes."""

    def test_empty_dataframe(self):
        """Empty DataFrame with correct columns should not crash."""
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        df = df.astype(float)
        result = TechnicalIndicators.calculate_all(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_missing_columns(self):
        """DataFrame without required 'volume' column should raise ValueError."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            # 'volume' intentionally missing
        })
        with pytest.raises(ValueError, match="Missing required column.*volume"):
            TechnicalIndicators.calculate_all(df)

    def test_nan_handling(self, sample_ohlcv_df):
        """After calculate_all, the last row should have no NaN (fillna(0) applied)."""
        result = TechnicalIndicators.calculate_all(sample_ohlcv_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not pd.isna(result[col].iloc[-1]), f"NaN found in last row, column: {col}"

    def test_rsi_all_gains(self):
        """Monotonically increasing prices -> RSI near 100, no crash."""
        prices = pd.Series(range(1, 51), dtype=float)
        rsi = TechnicalIndicators.rsi(prices, period=14)

        # Should not crash (avg_loss=0 guarded by .replace(0, np.nan))
        assert len(rsi) == 50

        # After enough data, RSI should be near 100 (all gains, zero losses)
        # The last values should be very high
        assert rsi.iloc[-1] >= 90, f"Expected RSI >= 90 for all-gain series, got {rsi.iloc[-1]}"

        # No NaN in final computed values (fillna(100) for zero-loss case)
        assert not pd.isna(rsi.iloc[-1]), "RSI is NaN for all-gains series"

    def test_zero_volume(self):
        """All volumes = 0 should not crash; vwap column should exist."""
        n = 30
        dates = pd.date_range('2025-01-01', periods=n, freq='B')
        df = pd.DataFrame({
            'open': np.linspace(100, 110, n),
            'high': np.linspace(101, 111, n),
            'low': np.linspace(99, 109, n),
            'close': np.linspace(100, 110, n),
            'volume': np.zeros(n),
        }, index=dates)

        # Should not crash (volume_ratio uses safe denominator)
        result = TechnicalIndicators.calculate_all(df)
        assert 'vwap' in result.columns
        assert 'volume_ratio' in result.columns
        # After fillna(0), no NaN in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].iloc[-1].isna().sum() == 0

    def test_flat_market(self, flat_market_df):
        """Flat market (high==low==open==close) should not crash.

        Division-by-zero guards handle:
        - total_range = 0 in candlestick doji detection
        - bb_width when middle band is constant (std_dev=0)
        - atr = 0 in trend_strength
        """
        result = TechnicalIndicators.calculate_all(flat_market_df)

        # Should complete without exception
        assert len(result) == len(flat_market_df)

        # ATR should be 0 (no range)
        assert (result['atr'] == 0).all(), "ATR should be 0 for flat market"

        # RSI: no changes -> should be handled gracefully (fillna applied)
        assert not result['rsi'].isna().any(), "RSI has NaN for flat market"

        # trend_strength should be 25.0 (neutral/weak trend) for flat market
        # NaN from zero DI sum is filled with 25.0 instead of 0 to avoid false signals
        assert (result['trend_strength'] == 25.0).all()

    def test_single_row(self):
        """Single-row DataFrame should not crash (indicators will mostly be 0/fillna)."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [500000.0],
        }, index=pd.date_range('2025-01-01', periods=1, freq='B'))

        # Should not raise any exception
        result = TechnicalIndicators.calculate_all(df)
        assert len(result) == 1

        # After fillna(0), numeric columns should have no NaN
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].iloc[0].isna().sum() == 0, (
            f"NaN found in columns: {result[numeric_cols].columns[result[numeric_cols].iloc[0].isna()].tolist()}"
        )
