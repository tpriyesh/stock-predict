"""
Pydantic models for data validation and serialization.
"""
from datetime import datetime, date
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeType(str, Enum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ORDER_WIN = "order_win"
    REGULATORY = "regulatory"
    MACRO = "macro"
    MNA = "mna"  # Mergers & Acquisitions
    DIVIDEND = "dividend"
    SPLIT = "split"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class StockPrice(BaseModel):
    """OHLCV price data for a single day."""
    symbol: str
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    adj_close: Optional[float] = None

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v, info):
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v, info):
        for field in ["open", "close"]:
            if field in info.data and v < info.data[field]:
                raise ValueError(f"high must be >= {field}")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_close(cls, v, info):
        for field in ["open", "close"]:
            if field in info.data and v > info.data[field]:
                raise ValueError(f"low must be <= {field}")
        return v

    class Config:
        frozen = True


class NewsArticle(BaseModel):
    """Structured news article data."""
    id: Optional[str] = None
    source: str
    url: str
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    published_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    # Extracted features (populated by OpenAI)
    tickers: list[str] = Field(default_factory=list)
    event_type: EventType = EventType.OTHER
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    key_claims: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)

    class Config:
        frozen = True


class TechnicalSignals(BaseModel):
    """Technical indicator values for a stock."""
    symbol: str
    date: date

    # Price metrics
    close: float
    change_pct: float

    # Trend indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volatility
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None

    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma20: Optional[float] = None
    price_vs_sma50: Optional[float] = None
    price_vs_sma200: Optional[float] = None

    # Volume
    volume: int = 0
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Support/Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class Signal(BaseModel):
    """Trading signal with full reasoning."""
    symbol: str
    date: date
    signal_type: SignalType
    trade_type: TradeType
    confidence: float = Field(ge=0.0, le=1.0)

    # Entry/Exit
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float

    # Scores breakdown
    technical_score: float = Field(ge=0.0, le=1.0)
    momentum_score: float = Field(ge=0.0, le=1.0)
    news_score: float = Field(ge=0.0, le=1.0)
    sector_score: float = Field(ge=0.0, le=1.0)
    flow_score: float = Field(ge=0.0, le=1.0)

    # Reasoning
    reasons: list[str] = Field(default_factory=list)
    news_sources: list[str] = Field(default_factory=list)

    # Risk metrics
    atr_pct: float
    liquidity_score: float = Field(ge=0.0, le=1.0)

    class Config:
        frozen = True


class Prediction(BaseModel):
    """Final prediction with full lineage."""
    id: Optional[str] = None
    symbol: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    trade_type: TradeType

    # Signal
    signal: SignalType
    confidence: float = Field(ge=0.0, le=1.0)

    # Trading plan
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float

    # Explanation (human-readable)
    summary: str
    reasons: list[str]

    # Data lineage
    technical_data: TechnicalSignals
    news_articles: list[NewsArticle] = Field(default_factory=list)

    # Verification links
    chart_url: Optional[str] = None
    news_urls: list[str] = Field(default_factory=list)

    # Outcome tracking (filled after trade closes)
    outcome_pct: Optional[float] = None
    outcome_hit_target: Optional[bool] = None
    outcome_hit_stop: Optional[bool] = None
