"""
Trading Configuration - Single source of truth.

ALL trading parameters are configured here, read from .env with defaults.
Every other module imports from here - no hardcoded values anywhere else.

Usage:
    from config.trading_config import CONFIG
    print(CONFIG.capital.initial_capital)
    print(CONFIG.hours.market_open)
"""
import os
from dataclasses import dataclass, field
from datetime import time, date
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent.parent / ".env")

_env_path = Path(__file__).parent.parent / ".env"
if not _env_path.exists():
    import warnings
    warnings.warn(
        f".env file not found at {_env_path}. Using default values. "
        "Copy .env.example to .env and configure your settings.",
        UserWarning,
        stacklevel=2
    )


def _env(key: str, default: str) -> str:
    """Get env var with default."""
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    """Get float env var."""
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    """Get int env var."""
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    """Get bool env var (case-insensitive)."""
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("true", "1", "yes", "on")


def _env_time(key: str, default: str) -> time:
    """Get time env var (format: HH:MM)."""
    val = os.getenv(key, default)
    parts = val.split(":")
    return time(int(parts[0]), int(parts[1]))


def _env_list(key: str, default: str) -> List[str]:
    """Get comma-separated list env var."""
    val = os.getenv(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]


# ============================================
# CAPITAL & RISK
# ============================================

@dataclass(frozen=True)
class CapitalConfig:
    """How much money, how much risk."""
    initial_capital: float = field(
        default_factory=lambda: _env_float("INITIAL_CAPITAL", 100000)
    )
    hard_stop_loss: float = field(
        default_factory=lambda: _env_float("HARD_STOP_LOSS", 80000)
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 0.05)
    )
    max_per_trade_risk_pct: float = field(
        default_factory=lambda: _env_float("MAX_PER_TRADE_RISK_PCT", 0.02)
    )
    max_position_pct: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_PCT", 0.30)
    )
    base_position_pct: float = field(
        default_factory=lambda: _env_float("BASE_POSITION_PCT", 0.25)
    )
    max_positions: int = field(
        default_factory=lambda: _env_int("MAX_POSITIONS", 5)
    )
    target_daily_return_pct: float = field(
        default_factory=lambda: _env_float("TARGET_DAILY_RETURN_PCT", 0.05)
    )
    estimated_fee_pct: float = field(
        default_factory=lambda: _env_float("ESTIMATED_FEE_PCT", 0.0005)
    )  # ~0.05% round-trip (brokerage + STT + exchange + GST + stamp)
    max_completed_trades_in_memory: int = field(
        default_factory=lambda: _env_int("MAX_COMPLETED_TRADES_IN_MEMORY", 200)
    )  # Cap in-memory list, rest in DB


# ============================================
# MARKET HOURS
# ============================================

@dataclass(frozen=True)
class TradingHours:
    """When to trade (all times in IST)."""
    timezone: str = field(
        default_factory=lambda: _env("TIMEZONE", "Asia/Kolkata")
    )
    pre_market_start: time = field(
        default_factory=lambda: _env_time("PRE_MARKET_START", "8:45")
    )
    market_open: time = field(
        default_factory=lambda: _env_time("MARKET_OPEN", "9:15")
    )
    entry_window_start: time = field(
        default_factory=lambda: _env_time("ENTRY_WINDOW_START", "9:30")
    )
    entry_window_end: time = field(
        default_factory=lambda: _env_time("ENTRY_WINDOW_END", "14:30")
    )
    square_off_start: time = field(
        default_factory=lambda: _env_time("SQUARE_OFF_START", "14:30")
    )
    square_off_time: time = field(
        default_factory=lambda: _env_time("SQUARE_OFF_TIME", "15:00")
    )
    market_close: time = field(
        default_factory=lambda: _env_time("MARKET_CLOSE", "15:30")
    )
    post_market_end: time = field(
        default_factory=lambda: _env_time("POST_MARKET_END", "16:00")
    )


# ============================================
# SIGNAL THRESHOLDS
# ============================================

@dataclass(frozen=True)
class SignalConfig:
    """When to buy/sell."""
    min_confidence: float = field(
        default_factory=lambda: _env_float("MIN_CONFIDENCE", 0.65)
    )
    min_risk_reward: float = field(
        default_factory=lambda: _env_float("MIN_RISK_REWARD", 1.8)
    )
    enable_buy_signals: bool = field(
        default_factory=lambda: _env_bool("ENABLE_BUY_SIGNALS", True)
    )
    enable_sell_signals: bool = field(
        default_factory=lambda: _env_bool("ENABLE_SELL_SIGNALS", False)
    )
    signal_cache_minutes: int = field(
        default_factory=lambda: _env_int("SIGNAL_CACHE_MINUTES", 5)
    )
    min_stop_distance_pct: float = field(
        default_factory=lambda: _env_float("MIN_STOP_DISTANCE_PCT", 0.005)
    )
    max_stop_distance_pct: float = field(
        default_factory=lambda: _env_float("MAX_STOP_DISTANCE_PCT", 0.03)
    )
    signal_max_age_seconds: int = field(
        default_factory=lambda: _env_int("SIGNAL_MAX_AGE_SECONDS", 1800)
    )  # Reject signals older than 30 min
    gap_down_reject_pct: float = field(
        default_factory=lambda: _env_float("GAP_DOWN_REJECT_PCT", 0.05)
    )  # Reject entry if stock gapped down >5% from prev close
    circuit_breaker_warn_pct: float = field(
        default_factory=lambda: _env_float("CIRCUIT_BREAKER_WARN_PCT", 0.08)
    )  # Warn/skip if stock moved >8% from open (near NSE circuit limit)
    min_volume_cr: float = field(
        default_factory=lambda: _env_float("MIN_VOLUME_CR", 10.0)
    )  # Minimum daily volume in crores for liquidity


# ============================================
# STRATEGY PARAMETERS
# ============================================

@dataclass(frozen=True)
class StrategyConfig:
    """Indicator settings."""
    # Supertrend
    supertrend_period: int = field(
        default_factory=lambda: _env_int("SUPERTREND_PERIOD", 7)
    )
    supertrend_multiplier: float = field(
        default_factory=lambda: _env_float("SUPERTREND_MULTIPLIER", 2.5)
    )
    # RSI
    rsi_period: int = field(
        default_factory=lambda: _env_int("RSI_PERIOD", 14)
    )
    rsi_oversold: float = field(
        default_factory=lambda: _env_float("RSI_OVERSOLD", 30)
    )
    rsi_overbought: float = field(
        default_factory=lambda: _env_float("RSI_OVERBOUGHT", 70)
    )
    # ADX
    adx_period: int = field(
        default_factory=lambda: _env_int("ADX_PERIOD", 14)
    )
    adx_strong_trend: float = field(
        default_factory=lambda: _env_float("ADX_STRONG_TREND", 25)
    )
    # Volume
    volume_threshold: float = field(
        default_factory=lambda: _env_float("VOLUME_THRESHOLD", 1.5)
    )
    # Targets
    target_multiplier: float = field(
        default_factory=lambda: _env_float("TARGET_MULTIPLIER", 2.0)
    )
    trailing_start_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_START_PCT", 0.01)
    )
    trailing_distance_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_DISTANCE_PCT", 0.008)
    )


# ============================================
# ORDER EXECUTION
# ============================================

@dataclass(frozen=True)
class OrderConfig:
    """How to execute orders."""
    max_slippage_pct: float = field(
        default_factory=lambda: _env_float("MAX_SLIPPAGE_PCT", 0.005)
    )
    order_timeout_seconds: int = field(
        default_factory=lambda: _env_int("ORDER_TIMEOUT_SECONDS", 30)
    )
    order_retry_count: int = field(
        default_factory=lambda: _env_int("ORDER_RETRY_COUNT", 3)
    )
    order_retry_delay: float = field(
        default_factory=lambda: _env_float("ORDER_RETRY_DELAY", 2.0)
    )
    use_limit_orders: bool = field(
        default_factory=lambda: _env_bool("USE_LIMIT_ORDERS", True)
    )
    entry_order_wait_seconds: int = field(
        default_factory=lambda: _env_int("ENTRY_ORDER_WAIT_SECONDS", 2)
    )
    exit_max_retries: int = field(
        default_factory=lambda: _env_int("EXIT_MAX_RETRIES", 5)
    )
    exit_retry_base_delay: float = field(
        default_factory=lambda: _env_float("EXIT_RETRY_BASE_DELAY", 3.0)
    )
    token_check_interval_seconds: int = field(
        default_factory=lambda: _env_int("TOKEN_CHECK_INTERVAL_SECONDS", 1800)
    )


# ============================================
# INTERVALS (polling/refresh)
# ============================================

@dataclass(frozen=True)
class IntervalConfig:
    """How often to check things."""
    signal_refresh_seconds: int = field(
        default_factory=lambda: _env_int("SIGNAL_REFRESH_SECONDS", 300)
    )
    position_check_seconds: int = field(
        default_factory=lambda: _env_int("POSITION_CHECK_SECONDS", 60)
    )
    quote_refresh_seconds: int = field(
        default_factory=lambda: _env_int("QUOTE_REFRESH_SECONDS", 30)
    )
    news_check_seconds: int = field(
        default_factory=lambda: _env_int("NEWS_CHECK_SECONDS", 300)
    )


# ============================================
# BROKER
# ============================================

@dataclass(frozen=True)
class BrokerConfig:
    """Broker settings."""
    broker: str = field(
        default_factory=lambda: _env("BROKER", "upstox")
    )
    mode: str = field(
        default_factory=lambda: _env("TRADING_MODE", "paper")
    )
    exchange: str = field(
        default_factory=lambda: _env("EXCHANGE", "NSE")
    )


# ============================================
# WATCHLIST
# ============================================

DEFAULT_WATCHLIST_STR = (
    "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,"
    "HINDUNILVR,SBIN,BHARTIARTL,ITC,KOTAKBANK,"
    "LT,HCLTECH,AXISBANK,ASIANPAINT,MARUTI,"
    "SUNPHARMA,TITAN,BAJFINANCE,WIPRO,ULTRACEMCO"
)


# ============================================
# NSE HOLIDAYS 2026
# ============================================

NSE_HOLIDAYS_2026 = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Maha Shivaratri
    date(2026, 3, 17),   # Holi
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 6),    # Mahavir Jayanti
    date(2026, 4, 10),   # Good Friday
    date(2026, 4, 14),   # Ambedkar Jayanti
    date(2026, 5, 1),    # May Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Muharram
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 21),  # Dussehra
    date(2026, 10, 28),  # Milad-un-Nabi
    date(2026, 11, 4),   # Diwali (Laxmi Puja)
    date(2026, 11, 5),   # Diwali (Balipratipada)
    date(2026, 11, 16),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]


# ============================================
# API PROVIDER SETTINGS
# ============================================

@dataclass(frozen=True)
class ProviderConfig:
    """API provider limits, costs, and cache settings."""
    # News providers (comma-separated, order = priority)
    news_providers: str = field(
        default_factory=lambda: _env("NEWS_PROVIDERS", "newsapi,gnews,finnhub,rss")
    )
    # Per-provider daily limits
    newsapi_daily_limit: int = field(
        default_factory=lambda: _env_int("NEWSAPI_DAILY_LIMIT", 100)
    )
    gnews_daily_limit: int = field(
        default_factory=lambda: _env_int("GNEWS_DAILY_LIMIT", 100)
    )
    finnhub_rpm: int = field(
        default_factory=lambda: _env_int("FINNHUB_RPM", 60)
    )
    yfinance_rpm: int = field(
        default_factory=lambda: _env_int("YFINANCE_RPM", 60)
    )
    yfinance_rps: int = field(
        default_factory=lambda: _env_int("YFINANCE_RPS", 2)
    )
    openai_rpm: int = field(
        default_factory=lambda: _env_int("OPENAI_RPM", 180)
    )
    openai_daily_cost_cap: float = field(
        default_factory=lambda: _env_float("OPENAI_DAILY_COST_CAP", 5.0)
    )
    # Cache TTLs (seconds)
    news_cache_ttl: int = field(
        default_factory=lambda: _env_int("NEWS_CACHE_TTL", 600)
    )
    price_cache_ttl: int = field(
        default_factory=lambda: _env_int("PRICE_CACHE_TTL", 300)
    )
    llm_cache_ttl: int = field(
        default_factory=lambda: _env_int("LLM_CACHE_TTL", 1800)
    )
    # Health thresholds
    unhealthy_threshold: int = field(
        default_factory=lambda: _env_int("UNHEALTHY_THRESHOLD", 5)
    )
    quota_alert_pct: float = field(
        default_factory=lambda: _env_float("QUOTA_ALERT_PCT", 0.80)
    )


# ============================================
# MASTER CONFIG (import this)
# ============================================

@dataclass
class TradingConfig:
    """Master config - single import for everything."""
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    hours: TradingHours = field(default_factory=TradingHours)
    signals: SignalConfig = field(default_factory=SignalConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    orders: OrderConfig = field(default_factory=OrderConfig)
    intervals: IntervalConfig = field(default_factory=IntervalConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)
    watchlist: List[str] = field(
        default_factory=lambda: _env_list("WATCHLIST", DEFAULT_WATCHLIST_STR)
    )
    holidays: List[date] = field(default_factory=lambda: NSE_HOLIDAYS_2026)

    def __post_init__(self):
        """Validate config values on creation."""
        # Critical validations - raise ValueError immediately
        critical_errors = []

        if self.capital.initial_capital <= 0:
            critical_errors.append(
                f"initial_capital ({self.capital.initial_capital}) must be > 0"
            )
        if self.capital.hard_stop_loss <= 0:
            critical_errors.append(
                f"hard_stop_loss ({self.capital.hard_stop_loss}) must be > 0"
            )
        if self.capital.initial_capital > 0 and self.capital.hard_stop_loss > 0:
            if self.capital.initial_capital <= self.capital.hard_stop_loss:
                critical_errors.append(
                    f"initial_capital ({self.capital.initial_capital}) must be > "
                    f"hard_stop_loss ({self.capital.hard_stop_loss})"
                )
        if self.capital.max_positions <= 0:
            critical_errors.append(
                f"max_positions ({self.capital.max_positions}) must be > 0"
            )
        if not (0 < self.capital.max_daily_loss_pct <= 1):
            critical_errors.append(
                f"max_daily_loss_pct ({self.capital.max_daily_loss_pct}) must be between 0 (exclusive) and 1 (inclusive)"
            )
        if not (0 < self.capital.max_per_trade_risk_pct <= 1):
            critical_errors.append(
                f"max_per_trade_risk_pct ({self.capital.max_per_trade_risk_pct}) must be between 0 (exclusive) and 1 (inclusive)"
            )

        if critical_errors:
            raise ValueError(
                "Invalid trading configuration:\n  - " + "\n  - ".join(critical_errors)
            )

        # Non-critical warnings
        errors = []

        # Time window validation
        if self.hours.entry_window_start >= self.hours.entry_window_end:
            errors.append(
                f"entry_window_start ({self.hours.entry_window_start}) must be < "
                f"entry_window_end ({self.hours.entry_window_end})"
            )
        if self.hours.square_off_start > self.hours.market_close:
            errors.append(
                f"square_off_start ({self.hours.square_off_start}) must be <= "
                f"market_close ({self.hours.market_close})"
            )

        # Signal validation
        if self.signals.min_confidence < 0 or self.signals.min_confidence > 1:
            errors.append(f"min_confidence ({self.signals.min_confidence}) must be 0-1")
        if self.signals.min_risk_reward <= 0:
            errors.append(f"min_risk_reward ({self.signals.min_risk_reward}) must be > 0")

        if errors:
            from loguru import logger
            for err in errors:
                logger.warning(f"CONFIG WARNING: {err}")

    def print_summary(self):
        """Print config summary."""
        print(f"  Capital:     Rs.{self.capital.initial_capital:,.0f}")
        print(f"  Hard Stop:   Rs.{self.capital.hard_stop_loss:,.0f}")
        print(f"  Daily Target: {self.capital.target_daily_return_pct:.0%}")
        print(f"  Max Loss:    {self.capital.max_daily_loss_pct:.0%}")
        print(f"  Positions:   {self.capital.max_positions} max")
        print(f"  Entry:       {self.hours.entry_window_start.strftime('%H:%M')} - {self.hours.entry_window_end.strftime('%H:%M')}")
        print(f"  Square Off:  {self.hours.square_off_time.strftime('%H:%M')}")
        print(f"  Confidence:  {self.signals.min_confidence:.0%} min")
        print(f"  R:R Ratio:   {self.signals.min_risk_reward}:1 min")
        print(f"  Mode:        {self.broker.mode}")
        print(f"  Watchlist:   {len(self.watchlist)} stocks")


# Singleton - use this everywhere
CONFIG = TradingConfig()


# Backward-compatible aliases
CAPITAL = CONFIG.capital
HOURS = CONFIG.hours
ORDERS = CONFIG.orders
STRATEGY = CONFIG.strategy
BROKER = CONFIG.broker
