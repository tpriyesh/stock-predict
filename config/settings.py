"""
Configuration management for Stock Prediction Engine.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Get base directory at module level
_BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    firecrawl_api_key: str = Field(default="", alias="FIRECRAWL_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    gnews_api_key: str = Field(default="", alias="GNEWS_API_KEY")
    alpha_vantage_key: str = Field(default="", alias="ALPHA_VANTAGE_KEY")

    # Paths - use defaults directly
    base_dir: Path = _BASE_DIR
    data_dir: Path = _BASE_DIR / "data"
    db_path: Path = _BASE_DIR / "data" / "stock_predict.duckdb"

    # Trading parameters
    min_liquidity_cr: float = 10.0  # Minimum ₹10 Cr daily volume
    max_atr_percent: float = 5.0    # Max 5% ATR (volatility cap)
    min_confidence: float = 0.60    # Only show >60% confidence
    max_sector_concentration: int = 3  # Max stocks from same sector
    top_n_picks: int = 10           # Number of recommendations

    # Technical indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ma_short: int = 20
    ma_medium: int = 50
    ma_long: int = 200

    # Data settings
    lookback_days: int = 365  # 1 year of historical data
    news_lookback_hours: int = 168  # 7 days of news (extended)

    # Model settings
    intraday_target_pct: float = 0.5  # Target 0.5% gain for intraday
    swing_target_pct: float = 1.0     # Target 1% outperformance for swing
    swing_horizon_days: int = 5       # 5-day holding period for swing

    # OpenAI settings
    openai_model: str = "gpt-4o-mini"  # Cost-effective for extraction

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
