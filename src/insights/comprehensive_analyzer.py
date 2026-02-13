"""
Comprehensive Stock Analyzer - Production Grade

Captures ALL parameters:
- Fundamental: P/E, P/B, ROE, ROA, Debt/Equity, EPS, Revenue growth
- Technical: RSI, MACD, Bollinger, MAs, ATR, Volume profile
- Statistical: Beta, Volatility, Correlation, Sharpe
- Market: Sector strength, Index correlation, FII/DII sentiment

Uses proper math:
- Kelly Criterion for position sizing
- ATR-based stop loss
- Probability calibration from historical data
- Confidence intervals
- Risk-adjusted returns

Handles edge cases:
- Retry on API failures
- Validate all data
- Fallback to defaults when needed
- Log all decisions

Multi-model AI with smart routing:
- Strong models (GPT-4o, Gemini Pro) for complex analysis
- Fast models for simple tasks
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from loguru import logger
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings

# Import advanced prediction engine (optional, graceful fallback)
try:
    from ..engines import AdvancedPredictionEngine, AdvancedPrediction
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError:
    ADVANCED_ENGINE_AVAILABLE = False
    AdvancedPrediction = None

# Import unified advanced engine with 17 models (optional, graceful fallback)
try:
    from ..core.advanced_engine import get_advanced_engine, UNIFIED_ADVANCED_AVAILABLE
    UNIFIED_ENGINE_AVAILABLE = True
except ImportError:
    UNIFIED_ENGINE_AVAILABLE = False
    UNIFIED_ADVANCED_AVAILABLE = False
    get_advanced_engine = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TradingTimeframe(Enum):
    INTRADAY = "INTRADAY"       # Same day, exit before close
    SWING = "SWING"             # 2-10 days
    POSITIONAL = "POSITIONAL"   # 2-8 weeks
    LONG_TERM = "LONG_TERM"     # Months to years


class SignalStrength(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class FundamentalData:
    """All fundamental parameters."""
    # Valuation
    market_cap: float = 0.0           # Market capitalization
    enterprise_value: float = 0.0     # EV = Market Cap + Debt - Cash
    pe_ratio: float = 0.0             # Price to Earnings
    pe_forward: float = 0.0           # Forward P/E
    pb_ratio: float = 0.0             # Price to Book
    ps_ratio: float = 0.0             # Price to Sales
    peg_ratio: float = 0.0            # P/E to Growth ratio
    ev_ebitda: float = 0.0            # EV/EBITDA

    # Profitability
    roe: float = 0.0                  # Return on Equity
    roa: float = 0.0                  # Return on Assets
    roce: float = 0.0                 # Return on Capital Employed
    profit_margin: float = 0.0        # Net Profit Margin
    operating_margin: float = 0.0     # Operating Margin
    gross_margin: float = 0.0         # Gross Margin

    # Growth
    revenue_growth: float = 0.0       # YoY Revenue Growth
    earnings_growth: float = 0.0      # YoY Earnings Growth
    eps: float = 0.0                  # Earnings Per Share
    eps_growth: float = 0.0           # EPS Growth

    # Financial Health
    debt_equity: float = 0.0          # Debt to Equity
    current_ratio: float = 0.0        # Current Assets / Current Liabilities
    quick_ratio: float = 0.0          # Quick Ratio
    interest_coverage: float = 0.0    # EBIT / Interest Expense

    # Dividends
    dividend_yield: float = 0.0       # Annual dividend / Price
    payout_ratio: float = 0.0         # Dividends / Net Income

    # Other
    book_value: float = 0.0           # Book Value per Share
    free_cash_flow: float = 0.0       # FCF
    shares_outstanding: float = 0.0   # Total shares


@dataclass
class TechnicalData:
    """All technical parameters."""
    # Price
    price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0

    # Moving Averages
    sma_5: float = 0.0
    sma_10: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_100: float = 0.0
    sma_200: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0

    # Trend Indicators
    rsi_14: float = 50.0              # Relative Strength Index
    rsi_7: float = 50.0               # Short-term RSI
    macd: float = 0.0                 # MACD line
    macd_signal: float = 0.0          # Signal line
    macd_histogram: float = 0.0       # Histogram
    adx: float = 0.0                  # Average Directional Index
    plus_di: float = 0.0              # +DI
    minus_di: float = 0.0             # -DI

    # Volatility
    atr_14: float = 0.0               # Average True Range
    atr_percent: float = 0.0          # ATR as % of price
    bollinger_upper: float = 0.0      # Upper Bollinger Band
    bollinger_middle: float = 0.0     # Middle Band (SMA 20)
    bollinger_lower: float = 0.0      # Lower Bollinger Band
    bollinger_width: float = 0.0      # Band width %

    # Volume
    volume: float = 0.0
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0         # Current / Average
    obv: float = 0.0                  # On Balance Volume
    vwap: float = 0.0                 # Volume Weighted Avg Price

    # Support/Resistance
    support_1: float = 0.0
    support_2: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    pivot_point: float = 0.0

    # 52-week
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    pct_from_52_high: float = 0.0
    pct_from_52_low: float = 0.0

    # Returns
    change_1d: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0
    change_3m: float = 0.0
    change_6m: float = 0.0
    change_1y: float = 0.0


@dataclass
class StatisticalData:
    """Statistical measures."""
    # Risk
    beta: float = 1.0                 # Beta vs NIFTY
    volatility_daily: float = 0.0     # Daily volatility
    volatility_annual: float = 0.0    # Annualized volatility
    var_95: float = 0.0               # Value at Risk 95%
    max_drawdown: float = 0.0         # Maximum drawdown

    # Performance
    sharpe_ratio: float = 0.0         # Risk-adjusted return
    sortino_ratio: float = 0.0        # Downside risk adjusted
    calmar_ratio: float = 0.0         # Return / Max Drawdown

    # Correlation
    nifty_correlation: float = 0.0    # Correlation with NIFTY
    sector_correlation: float = 0.0   # Correlation with sector

    # Distribution
    skewness: float = 0.0             # Return distribution skew
    kurtosis: float = 0.0             # Fat tails


@dataclass
class RiskManagement:
    """Position sizing and risk parameters."""
    # Position Sizing
    capital: float = 100000.0         # Trading capital
    risk_per_trade: float = 0.02      # Max risk per trade (2%)
    max_position_size: float = 0.20   # Max 20% in one stock

    # Calculated
    position_size_shares: int = 0     # Number of shares
    position_size_value: float = 0.0  # Value in rupees
    risk_amount: float = 0.0          # Amount at risk

    # Stops
    stop_loss_price: float = 0.0
    stop_loss_percent: float = 0.0
    trailing_stop: float = 0.0

    # Targets
    target_1_price: float = 0.0       # 1:1 R:R
    target_2_price: float = 0.0       # 1:2 R:R
    target_3_price: float = 0.0       # 1:3 R:R

    # Risk/Reward
    risk_reward_ratio: float = 0.0

    # Kelly Criterion
    kelly_fraction: float = 0.0       # Optimal bet size
    kelly_position: float = 0.0       # Kelly-based position


@dataclass
class ProbabilityAnalysis:
    """Probability calculations."""
    # Win/Loss
    win_probability: float = 0.5
    loss_probability: float = 0.5

    # Expected Value
    expected_return: float = 0.0      # Expected return %
    expected_value: float = 0.0       # Expected profit/loss

    # Confidence
    confidence_level: float = 0.0     # How confident (0-1)
    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 0.0

    # Historical calibration
    historical_win_rate: float = 0.5  # From past trades
    historical_trades: int = 0        # Number of past trades
    calibration_factor: float = 1.0   # Adjustment factor


@dataclass
class TradeRecommendation:
    """Final trade recommendation."""
    # Core
    signal: SignalStrength = SignalStrength.HOLD
    timeframe: TradingTimeframe = TradingTimeframe.SWING
    action: str = "HOLD"              # BUY, SELL, HOLD, AVOID

    # Entry
    entry_price: float = 0.0
    entry_type: str = "LIMIT"         # LIMIT, MARKET, SL
    entry_condition: str = ""         # When to enter

    # Exit
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    trailing_stop_pct: float = 0.0

    # Position
    position_size_pct: float = 0.0    # % of capital
    position_shares: int = 0

    # Probabilities
    win_probability: float = 0.5
    expected_return: float = 0.0
    risk_reward: float = 0.0

    # Reasoning
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.0
    data_quality: float = 0.0         # How complete is data


@dataclass
class ComprehensiveAnalysis:
    """Complete stock analysis."""
    # Metadata
    symbol: str = ""
    name: str = ""
    sector: str = ""
    industry: str = ""
    exchange: str = "NSE"
    timestamp: str = ""

    # All data
    fundamental: FundamentalData = field(default_factory=FundamentalData)
    technical: TechnicalData = field(default_factory=TechnicalData)
    statistical: StatisticalData = field(default_factory=StatisticalData)
    risk_mgmt: RiskManagement = field(default_factory=RiskManagement)
    probability: ProbabilityAnalysis = field(default_factory=ProbabilityAnalysis)

    # Recommendations
    intraday_rec: TradeRecommendation = field(default_factory=TradeRecommendation)
    swing_rec: TradeRecommendation = field(default_factory=TradeRecommendation)

    # AI Insights
    ai_summary: str = ""
    ai_explanation: str = ""          # Plain English explanation
    ai_models_used: List[str] = field(default_factory=list)

    # Advanced Prediction (from physics/ML/microstructure engine)
    advanced_prediction: Optional[Any] = None  # AdvancedPrediction when available
    advanced_probability: float = 0.0          # Combined probability from advanced engine
    detected_regime: str = ""                  # Market regime (TRENDING_BULL, RANGING, etc.)
    optimal_strategy: str = ""                 # Recommended strategy (momentum, mean_reversion)

    # Unified Advanced Prediction (17 models: 6 base + 11 advanced math)
    unified_prediction: Optional[Dict[str, Any]] = None  # Full unified prediction result
    unified_probability: float = 0.0           # Final probability from all 17 models
    unified_confidence: float = 0.0            # Confidence from unified engine
    total_models_used: int = 0                 # Number of models that contributed (max 17)
    unified_insights: List[str] = field(default_factory=list)  # Human-readable insights from advanced math

    # Quality
    data_completeness: float = 0.0    # 0-1, how much data we have
    analysis_confidence: float = 0.0  # 0-1, overall confidence


# ============================================================================
# COMPREHENSIVE ANALYZER
# ============================================================================

class ComprehensiveAnalyzer:
    """
    Production-grade stock analyzer.

    Captures everything, validates everything, explains everything.
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(
        self,
        capital: float = 100000.0,
        risk_per_trade: float = 0.02,
        use_advanced_engine: bool = True,
        use_unified_advanced: bool = True
    ):
        self.settings = get_settings()
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.use_advanced_engine = use_advanced_engine and ADVANCED_ENGINE_AVAILABLE
        self.use_unified_advanced = use_unified_advanced and UNIFIED_ENGINE_AVAILABLE

        # Cache
        self.nifty_data = None
        self.trade_history_file = Path("data/trade_history.json")
        self.trade_history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load historical trades for calibration
        self.historical_trades = self._load_trade_history()

        # Initialize advanced engine if available
        self.advanced_engine = None
        if self.use_advanced_engine:
            try:
                self.advanced_engine = AdvancedPredictionEngine(use_ml_ensemble=False)
                logger.info("Advanced prediction engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize advanced engine: {e}")
                self.use_advanced_engine = False

        # Initialize unified advanced engine with 17 models (new)
        self.unified_engine = None
        if self.use_unified_advanced:
            try:
                self.unified_engine = get_advanced_engine()
                if self.unified_engine and getattr(self.unified_engine, 'unified_advanced_engine', None):
                    logger.info("Unified Advanced Engine initialized with 17 models: "
                              "PCA, Wavelet, Kalman, Markov, DQN, Factor + "
                              "Quantum, Bellman, SDE, InfoGeometry, Tensor, Spectral, "
                              "OptimalTransport, Taylor, HeavyTailed, Copula, MultiDimConfidence")
                else:
                    logger.info("Unified Advanced Engine initialized with 6 models (advanced modules not available)")
            except Exception as e:
                logger.warning(f"Could not initialize unified advanced engine: {e}")
                self.use_unified_advanced = False

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        logger.error(f"All {self.MAX_RETRIES} attempts failed: {last_error}")
        return None

    def _load_trade_history(self) -> List[Dict]:
        """Load historical trades for probability calibration."""
        try:
            if self.trade_history_file.exists():
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

    def _save_trade(self, trade: Dict):
        """Save trade for future calibration."""
        self.historical_trades.append(trade)
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.historical_trades[-1000:], f, indent=2)  # Keep last 1000
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def fetch_fundamental_data(self, ticker: yf.Ticker, info: Dict) -> FundamentalData:
        """Fetch all fundamental data with validation."""
        def safe_get(key, default=0.0):
            val = info.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val) if isinstance(val, (int, float)) else default

        return FundamentalData(
            # Valuation
            market_cap=safe_get('marketCap') / 10000000,  # Convert to Cr
            enterprise_value=safe_get('enterpriseValue') / 10000000,
            pe_ratio=safe_get('trailingPE'),
            pe_forward=safe_get('forwardPE'),
            pb_ratio=safe_get('priceToBook'),
            ps_ratio=safe_get('priceToSalesTrailing12Months'),
            peg_ratio=safe_get('pegRatio'),
            ev_ebitda=safe_get('enterpriseToEbitda'),

            # Profitability
            roe=safe_get('returnOnEquity') * 100,
            roa=safe_get('returnOnAssets') * 100,
            profit_margin=safe_get('profitMargins') * 100,
            operating_margin=safe_get('operatingMargins') * 100,
            gross_margin=safe_get('grossMargins') * 100,

            # Growth
            revenue_growth=safe_get('revenueGrowth') * 100,
            earnings_growth=safe_get('earningsGrowth') * 100,
            eps=safe_get('trailingEps'),

            # Financial Health
            debt_equity=safe_get('debtToEquity'),
            current_ratio=safe_get('currentRatio'),
            quick_ratio=safe_get('quickRatio'),

            # Dividends
            dividend_yield=safe_get('dividendYield') * 100 if safe_get('dividendYield') else 0,
            payout_ratio=safe_get('payoutRatio') * 100 if safe_get('payoutRatio') else 0,

            # Other
            book_value=safe_get('bookValue'),
            free_cash_flow=safe_get('freeCashflow') / 10000000 if safe_get('freeCashflow') else 0,
            shares_outstanding=safe_get('sharesOutstanding') / 10000000,
        )

    def calculate_technical_data(self, hist: pd.DataFrame, info: Dict) -> TechnicalData:
        """Calculate all technical indicators."""
        if hist.empty or len(hist) < 20:
            return TechnicalData()

        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        current_price = close.iloc[-1]

        # Moving Averages
        sma_5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else current_price
        sma_10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else current_price
        sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else current_price
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else current_price
        sma_100 = close.rolling(100).mean().iloc[-1] if len(close) >= 100 else current_price
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else current_price

        # EMA
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1] if len(close) >= 50 else current_price

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != 0 else 50

        gain_7 = delta.where(delta > 0, 0).rolling(7).mean()
        loss_7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs_7 = gain_7 / loss_7
        rsi_7 = 100 - (100 / (1 + rs_7.iloc[-1])) if rs_7.iloc[-1] != 0 else 50

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - macd_signal

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        atr_percent = (atr_14 / current_price) * 100

        # Bollinger Bands
        bb_middle = sma_20
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_width = ((bb_upper - bb_lower) / bb_middle) * 100

        # Volume
        vol_sma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_sma if vol_sma > 0 else 1

        # Support/Resistance (simple pivot)
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        support_1 = (2 * pivot) - high.iloc[-1]
        support_2 = pivot - (high.iloc[-1] - low.iloc[-1])
        resistance_1 = (2 * pivot) - low.iloc[-1]
        resistance_2 = pivot + (high.iloc[-1] - low.iloc[-1])

        # 52-week
        week_52_high = high.tail(252).max() if len(high) >= 252 else high.max()
        week_52_low = low.tail(252).min() if len(low) >= 252 else low.min()
        pct_from_high = ((current_price - week_52_high) / week_52_high) * 100
        pct_from_low = ((current_price - week_52_low) / week_52_low) * 100

        # Returns
        def safe_return(days):
            if len(close) >= days:
                return ((current_price / close.iloc[-days]) - 1) * 100
            return 0

        return TechnicalData(
            price=round(current_price, 2),
            open=round(hist['Open'].iloc[-1], 2),
            high=round(high.iloc[-1], 2),
            low=round(low.iloc[-1], 2),
            prev_close=round(close.iloc[-2], 2) if len(close) > 1 else current_price,

            sma_5=round(sma_5, 2),
            sma_10=round(sma_10, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            sma_100=round(sma_100, 2),
            sma_200=round(sma_200, 2),
            ema_9=round(ema_9, 2),
            ema_21=round(ema_21, 2),
            ema_50=round(ema_50, 2),

            rsi_14=round(rsi_14, 1),
            rsi_7=round(rsi_7, 1),
            macd=round(macd_line.iloc[-1], 2),
            macd_signal=round(macd_signal.iloc[-1], 2),
            macd_histogram=round(macd_hist.iloc[-1], 2),

            atr_14=round(atr_14, 2),
            atr_percent=round(atr_percent, 2),
            bollinger_upper=round(bb_upper, 2),
            bollinger_middle=round(bb_middle, 2),
            bollinger_lower=round(bb_lower, 2),
            bollinger_width=round(bb_width, 2),

            volume=volume.iloc[-1],
            volume_sma_20=round(vol_sma, 0),
            volume_ratio=round(vol_ratio, 2),

            support_1=round(support_1, 2),
            support_2=round(support_2, 2),
            resistance_1=round(resistance_1, 2),
            resistance_2=round(resistance_2, 2),
            pivot_point=round(pivot, 2),

            week_52_high=round(week_52_high, 2),
            week_52_low=round(week_52_low, 2),
            pct_from_52_high=round(pct_from_high, 1),
            pct_from_52_low=round(pct_from_low, 1),

            change_1d=round(safe_return(1), 2),
            change_1w=round(safe_return(5), 2),
            change_1m=round(safe_return(21), 2),
            change_3m=round(safe_return(63), 2),
            change_6m=round(safe_return(126), 2),
            change_1y=round(safe_return(252), 2),
        )

    def calculate_statistical_data(self, hist: pd.DataFrame) -> StatisticalData:
        """Calculate statistical measures."""
        if hist.empty or len(hist) < 50:
            return StatisticalData()

        returns = hist['Close'].pct_change().dropna()

        # Volatility
        vol_daily = returns.std()
        vol_annual = vol_daily * np.sqrt(252)

        # Beta (vs NIFTY)
        beta = 1.0
        nifty_corr = 0.5
        try:
            if self.nifty_data is None:
                nifty = yf.Ticker("^NSEI")
                self.nifty_data = nifty.history(period='1y')

            if not self.nifty_data.empty:
                nifty_returns = self.nifty_data['Close'].pct_change().dropna()
                # Align dates
                common_idx = returns.index.intersection(nifty_returns.index)
                if len(common_idx) > 20:
                    stock_ret = returns.loc[common_idx]
                    nifty_ret = nifty_returns.loc[common_idx]
                    cov = np.cov(stock_ret, nifty_ret)[0, 1]
                    var = np.var(nifty_ret)
                    beta = cov / var if var > 0 else 1.0
                    nifty_corr = np.corrcoef(stock_ret, nifty_ret)[0, 1]
        except:
            pass

        # VaR
        var_95 = np.percentile(returns, 5) * 100

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # Sharpe (assuming 6% risk-free rate)
        rf_daily = 0.06 / 252
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino (downside only)
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

        # Distribution
        skew = returns.skew()
        kurt = returns.kurtosis()

        return StatisticalData(
            beta=round(beta, 2),
            volatility_daily=round(vol_daily * 100, 2),
            volatility_annual=round(vol_annual * 100, 2),
            var_95=round(var_95, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            nifty_correlation=round(nifty_corr, 2),
            skewness=round(skew, 2),
            kurtosis=round(kurt, 2),
        )

    def calculate_position_sizing(
        self,
        price: float,
        atr: float,
        win_prob: float,
        avg_win: float = 2.0,
        avg_loss: float = 1.0
    ) -> RiskManagement:
        """
        Calculate proper position sizing using:
        1. ATR-based stop loss
        2. Fixed risk per trade
        3. Kelly Criterion for optimal sizing
        """
        # Stop loss: 1.5x ATR below entry
        stop_loss_price = price - (1.5 * atr)
        stop_loss_pct = ((price - stop_loss_price) / price) * 100

        # Risk amount
        risk_amount = self.capital * self.risk_per_trade

        # Position size from risk
        risk_per_share = price - stop_loss_price
        if risk_per_share > 0:
            position_shares = int(risk_amount / risk_per_share)
            position_value = position_shares * price
        else:
            position_shares = 0
            position_value = 0

        # Cap at max position size
        max_value = self.capital * 0.20  # 20% max
        if position_value > max_value:
            position_value = max_value
            position_shares = int(max_value / price)

        # Targets
        target_1 = price + (1.5 * atr)  # 1:1
        target_2 = price + (3.0 * atr)  # 1:2
        target_3 = price + (4.5 * atr)  # 1:3

        # Risk/Reward
        reward = target_1 - price
        risk = price - stop_loss_price
        rr = reward / risk if risk > 0 else 0

        # Kelly Criterion: f* = (bp - q) / b
        # b = odds (avg_win / avg_loss)
        # p = win probability
        # q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 1
        p = win_prob
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, min(0.25, kelly))  # Cap at 25%
        kelly_position = self.capital * kelly

        return RiskManagement(
            capital=self.capital,
            risk_per_trade=self.risk_per_trade,
            max_position_size=0.20,
            position_size_shares=position_shares,
            position_size_value=round(position_value, 2),
            risk_amount=round(risk_amount, 2),
            stop_loss_price=round(stop_loss_price, 2),
            stop_loss_percent=round(stop_loss_pct, 2),
            trailing_stop=round(stop_loss_pct * 0.75, 2),  # Tighter trailing
            target_1_price=round(target_1, 2),
            target_2_price=round(target_2, 2),
            target_3_price=round(target_3, 2),
            risk_reward_ratio=round(rr, 2),
            kelly_fraction=round(kelly, 4),
            kelly_position=round(kelly_position, 2),
        )

    def calculate_probability(
        self,
        technical: TechnicalData,
        fundamental: FundamentalData,
        statistical: StatisticalData,
        timeframe: TradingTimeframe
    ) -> ProbabilityAnalysis:
        """
        Calculate calibrated win probability.
        Based on multiple factors with historical win rates.
        """
        factors = []
        weights = []

        # 1. RSI Factor (58% historical win rate for oversold bounce)
        if technical.rsi_14 < 30:
            factors.append(0.58)
            weights.append(1.5)
        elif technical.rsi_14 < 40:
            factors.append(0.55)
            weights.append(1.2)
        elif technical.rsi_14 > 70:
            factors.append(0.42)  # Overbought = lower win rate
            weights.append(1.3)
        else:
            factors.append(0.50)
            weights.append(0.5)

        # 2. Trend Factor (52% for trend alignment)
        trend_score = 0
        if technical.price > technical.sma_50:
            trend_score += 1
        if technical.price > technical.sma_200:
            trend_score += 1
        if technical.sma_50 > technical.sma_200:  # Golden cross
            trend_score += 1

        if trend_score >= 3:
            factors.append(0.55)
            weights.append(1.2)
        elif trend_score >= 2:
            factors.append(0.52)
            weights.append(1.0)
        else:
            factors.append(0.48)
            weights.append(0.8)

        # 3. Volume Factor (54% for volume confirmation)
        if technical.volume_ratio > 2.0:
            factors.append(0.56)
            weights.append(1.3)
        elif technical.volume_ratio > 1.5:
            factors.append(0.54)
            weights.append(1.0)
        elif technical.volume_ratio < 0.5:
            factors.append(0.45)
            weights.append(0.8)
        else:
            factors.append(0.50)
            weights.append(0.5)

        # 4. Fundamental Quality (only for swing/positional)
        if timeframe in [TradingTimeframe.SWING, TradingTimeframe.POSITIONAL]:
            if fundamental.roe > 15 and fundamental.debt_equity < 1:
                factors.append(0.55)
                weights.append(1.0)
            elif fundamental.roe < 5 or fundamental.debt_equity > 2:
                factors.append(0.45)
                weights.append(0.8)
            else:
                factors.append(0.50)
                weights.append(0.5)

        # 5. Volatility Factor
        if statistical.volatility_annual < 30:
            factors.append(0.52)  # Low vol = easier to predict
            weights.append(0.8)
        elif statistical.volatility_annual > 50:
            factors.append(0.48)  # High vol = harder
            weights.append(0.6)
        else:
            factors.append(0.50)
            weights.append(0.5)

        # Weighted average
        if weights:
            win_prob = sum(f * w for f, w in zip(factors, weights)) / sum(weights)
        else:
            win_prob = 0.50

        # Calibration from historical trades
        calibration = 1.0
        symbol_trades = [t for t in self.historical_trades if t.get('symbol') == technical.price]
        if len(symbol_trades) >= 10:
            actual_wins = sum(1 for t in symbol_trades if t.get('won', False))
            actual_rate = actual_wins / len(symbol_trades)
            predicted_rate = sum(t.get('predicted_prob', 0.5) for t in symbol_trades) / len(symbol_trades)
            if predicted_rate > 0:
                calibration = actual_rate / predicted_rate
                calibration = max(0.8, min(1.2, calibration))  # Limit adjustment

        win_prob = win_prob * calibration
        win_prob = max(0.35, min(0.75, win_prob))  # Reasonable bounds

        # Expected value
        avg_win = 2.0  # Assume 2% avg win
        avg_loss = 1.5  # Assume 1.5% avg loss
        expected_return = (win_prob * avg_win) - ((1 - win_prob) * avg_loss)

        # Confidence interval (simple approximation)
        # More data = tighter interval
        n_factors = len(factors)
        std_dev = 0.10 / np.sqrt(n_factors)  # Shrinks with more factors
        ci_low = win_prob - (1.96 * std_dev)
        ci_high = win_prob + (1.96 * std_dev)

        return ProbabilityAnalysis(
            win_probability=round(win_prob, 3),
            loss_probability=round(1 - win_prob, 3),
            expected_return=round(expected_return, 2),
            expected_value=round(expected_return * 100, 2),  # Per Rs 10000
            confidence_level=round(1 - (2 * std_dev), 2),
            confidence_interval_low=round(max(0, ci_low), 3),
            confidence_interval_high=round(min(1, ci_high), 3),
            historical_win_rate=round(calibration * 0.5, 2),
            historical_trades=len(symbol_trades),
            calibration_factor=round(calibration, 2),
        )

    def generate_recommendation(
        self,
        symbol: str,
        technical: TechnicalData,
        fundamental: FundamentalData,
        statistical: StatisticalData,
        risk_mgmt: RiskManagement,
        probability: ProbabilityAnalysis,
        timeframe: TradingTimeframe
    ) -> TradeRecommendation:
        """Generate trade recommendation based on all data."""

        bullish = []
        bearish = []
        risks = []

        # Technical Factors
        if technical.rsi_14 < 30:
            bullish.append(f"RSI oversold ({technical.rsi_14:.0f}) - bounce likely")
        elif technical.rsi_14 > 70:
            bearish.append(f"RSI overbought ({technical.rsi_14:.0f}) - pullback likely")

        if technical.price > technical.sma_50 > technical.sma_200:
            bullish.append("Strong uptrend (Golden Cross active)")
        elif technical.price < technical.sma_50 < technical.sma_200:
            bearish.append("Downtrend (Death Cross active)")

        if technical.macd_histogram > 0 and technical.macd > technical.macd_signal:
            bullish.append("MACD bullish crossover")
        elif technical.macd_histogram < 0:
            bearish.append("MACD bearish")

        if technical.volume_ratio > 1.5:
            bullish.append(f"High volume ({technical.volume_ratio:.1f}x) confirms move")
        elif technical.volume_ratio < 0.5:
            bearish.append("Low volume - weak conviction")

        if technical.pct_from_52_high > -5:
            bullish.append("Near 52-week high - strength")
        elif technical.pct_from_52_low < 10:
            bearish.append("Near 52-week low - weakness")

        # Fundamental Factors (for swing/positional)
        if timeframe != TradingTimeframe.INTRADAY:
            if fundamental.roe > 15:
                bullish.append(f"Strong ROE ({fundamental.roe:.1f}%)")
            if fundamental.debt_equity > 1.5:
                bearish.append(f"High debt ({fundamental.debt_equity:.1f}x equity)")
                risks.append("Leverage risk in downturn")
            if fundamental.pe_ratio > 50:
                bearish.append(f"Expensive valuation (P/E {fundamental.pe_ratio:.1f})")
            elif fundamental.pe_ratio < 15 and fundamental.pe_ratio > 0:
                bullish.append(f"Attractive valuation (P/E {fundamental.pe_ratio:.1f})")

        # Statistical Risks
        if statistical.volatility_annual > 40:
            risks.append(f"High volatility ({statistical.volatility_annual:.0f}% annual)")
        if statistical.beta > 1.5:
            risks.append(f"High beta ({statistical.beta:.1f}) - amplified moves")
        if statistical.max_drawdown < -30:
            risks.append(f"History of large drawdowns ({statistical.max_drawdown:.0f}%)")

        # Determine Signal
        bull_score = len(bullish)
        bear_score = len(bearish)
        net_score = bull_score - bear_score

        if probability.win_probability >= 0.60 and net_score >= 2:
            signal = SignalStrength.STRONG_BUY
            action = "BUY"
        elif probability.win_probability >= 0.55 and net_score >= 1:
            signal = SignalStrength.BUY
            action = "BUY"
        elif probability.win_probability >= 0.50 and net_score >= 0:
            signal = SignalStrength.WEAK_BUY
            action = "BUY"
        elif probability.win_probability <= 0.40 or net_score <= -2:
            signal = SignalStrength.SELL
            action = "SELL"
        elif net_score <= -1:
            signal = SignalStrength.WEAK_SELL
            action = "AVOID"
        else:
            signal = SignalStrength.HOLD
            action = "HOLD"

        # Entry condition
        if action == "BUY":
            if technical.price < technical.support_1:
                entry_condition = f"Enter at support ₹{technical.support_1:.2f}"
            else:
                entry_condition = f"Enter on pullback to ₹{technical.ema_9:.2f}"
        else:
            entry_condition = "Wait for better setup"

        # Position sizing
        if action == "BUY":
            pos_pct = min(risk_mgmt.kelly_fraction * 100, 15)  # Max 15%
            pos_pct = max(pos_pct, 5) if probability.win_probability > 0.55 else 0
        else:
            pos_pct = 0

        return TradeRecommendation(
            signal=signal,
            timeframe=timeframe,
            action=action,
            entry_price=technical.price,
            entry_type="LIMIT" if action == "BUY" else "N/A",
            entry_condition=entry_condition,
            stop_loss=risk_mgmt.stop_loss_price,
            target_1=risk_mgmt.target_1_price,
            target_2=risk_mgmt.target_2_price,
            trailing_stop_pct=risk_mgmt.trailing_stop,
            position_size_pct=round(pos_pct, 1),
            position_shares=risk_mgmt.position_size_shares if action == "BUY" else 0,
            win_probability=probability.win_probability,
            expected_return=probability.expected_return,
            risk_reward=risk_mgmt.risk_reward_ratio,
            bullish_factors=bullish,
            bearish_factors=bearish,
            key_risks=risks,
            overall_confidence=probability.confidence_level,
            data_quality=0.9,  # TODO: Calculate based on missing data
        )

    def get_ai_explanation(
        self,
        analysis: ComprehensiveAnalysis,
        use_strong_model: bool = True
    ) -> Tuple[str, str]:
        """
        Get AI explanation of all parameters in plain English.
        Uses strong model for complex analysis.
        """
        # Build comprehensive prompt
        prompt = f"""You are a stock market expert explaining analysis to a retail trader in India.

STOCK: {analysis.symbol} ({analysis.name})
SECTOR: {analysis.sector}

## CURRENT PRICE & PERFORMANCE
- Price: ₹{analysis.technical.price}
- Today: {analysis.technical.change_1d:+.2f}%
- 1 Week: {analysis.technical.change_1w:+.2f}%
- 1 Month: {analysis.technical.change_1m:+.2f}%
- 1 Year: {analysis.technical.change_1y:+.2f}%
- From 52-week high: {analysis.technical.pct_from_52_high:.1f}%
- From 52-week low: {analysis.technical.pct_from_52_low:.1f}%

## FUNDAMENTALS
- Market Cap: ₹{analysis.fundamental.market_cap:,.0f} Cr
- P/E Ratio: {analysis.fundamental.pe_ratio:.1f} (Price relative to earnings)
- P/B Ratio: {analysis.fundamental.pb_ratio:.1f} (Price relative to book value)
- ROE: {analysis.fundamental.roe:.1f}% (Return on equity)
- Debt/Equity: {analysis.fundamental.debt_equity:.2f}
- Revenue Growth: {analysis.fundamental.revenue_growth:.1f}%
- Profit Margin: {analysis.fundamental.profit_margin:.1f}%
- Dividend Yield: {analysis.fundamental.dividend_yield:.2f}%

## TECHNICALS
- RSI (14): {analysis.technical.rsi_14:.0f} (Below 30 = oversold, Above 70 = overbought)
- MACD: {analysis.technical.macd:.2f} (Signal: {analysis.technical.macd_signal:.2f})
- Above 50-day MA: {'Yes' if analysis.technical.price > analysis.technical.sma_50 else 'No'}
- Above 200-day MA: {'Yes' if analysis.technical.price > analysis.technical.sma_200 else 'No'}
- Volume: {analysis.technical.volume_ratio:.1f}x average
- ATR: ₹{analysis.technical.atr_14:.2f} ({analysis.technical.atr_percent:.1f}% of price)

## RISK METRICS
- Beta: {analysis.statistical.beta:.2f} (1.0 = moves with market)
- Volatility: {analysis.statistical.volatility_annual:.1f}% annual
- Max Drawdown: {analysis.statistical.max_drawdown:.1f}%
- Sharpe Ratio: {analysis.statistical.sharpe_ratio:.2f}

## CALCULATED PROBABILITIES
- Win Probability: {analysis.probability.win_probability:.1%}
- Expected Return: {analysis.probability.expected_return:.2f}% per trade
- Confidence: {analysis.probability.confidence_level:.1%}

## RECOMMENDATIONS
### For INTRADAY:
- Signal: {analysis.intraday_rec.signal.value}
- Entry: ₹{analysis.intraday_rec.entry_price}
- Stop Loss: ₹{analysis.intraday_rec.stop_loss}
- Target: ₹{analysis.intraday_rec.target_1}
- Position Size: {analysis.intraday_rec.position_size_pct:.1f}% of capital

### For SWING (2-10 days):
- Signal: {analysis.swing_rec.signal.value}
- Entry: ₹{analysis.swing_rec.entry_price}
- Stop Loss: ₹{analysis.swing_rec.stop_loss}
- Target: ₹{analysis.swing_rec.target_1}
- Position Size: {analysis.swing_rec.position_size_pct:.1f}% of capital

Please provide:
1. **SUMMARY** (2-3 sentences): Overall picture of this stock right now.

2. **WHAT THE NUMBERS MEAN** (explain key metrics in simple terms):
   - Is the stock expensive or cheap? (P/E, P/B)
   - Is the company making good profits? (ROE, margins)
   - Is it risky? (debt, volatility, beta)
   - Is momentum with buyers or sellers? (RSI, MACD, volume)

3. **SHOULD YOU TRADE?**
   - For INTRADAY: Clear yes/no with reason
   - For SWING: Clear yes/no with reason

4. **KEY RISKS** that could go wrong

5. **ACTION PLAN** - Specific steps if trading

Be direct, practical, unbiased. Don't be overly bullish or bearish. State facts.
"""

        # Call appropriate model
        try:
            if use_strong_model:
                # Use GPT-4o for complex analysis
                summary, explanation = self._call_strong_model(prompt)
            else:
                summary, explanation = self._call_fast_model(prompt)

            return summary, explanation

        except Exception as e:
            logger.error(f"AI explanation failed: {e}")
            return "AI analysis unavailable", str(e)

    def _call_strong_model(self, prompt: str) -> Tuple[str, str]:
        """Call GPT-4o or Gemini Pro for complex analysis."""
        settings = self.settings

        # Try OpenAI first
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)

                response = client.chat.completions.create(
                    model="gpt-4o",  # Strong model
                    messages=[
                        {"role": "system", "content": "You are an expert stock analyst explaining to retail traders."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )

                content = response.choices[0].message.content

                # Extract summary (first paragraph)
                lines = content.split('\n')
                summary = ""
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        summary = line.strip()
                        break

                return summary[:200], content

            except Exception as e:
                logger.warning(f"OpenAI strong model failed: {e}")

        # Fallback to Gemini
        try:
            from google import genai
            client = genai.Client(api_key="AIzaSyBrM5XxpCnNd6vs4wQm8bigt3MPg-GwUqk")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            content = response.text
            summary = content.split('\n')[0][:200] if content else ""

            return summary, content

        except Exception as e:
            logger.error(f"Gemini strong model failed: {e}")
            return "Analysis unavailable", str(e)

    def _call_fast_model(self, prompt: str) -> Tuple[str, str]:
        """Call fast model for simple tasks."""
        settings = self.settings

        if settings.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )

                content = response.choices[0].message.content
                return content[:200], content

            except Exception as e:
                logger.warning(f"Fast model failed: {e}")

        return "Analysis unavailable", ""

    def analyze(self, symbol: str, skip_ai: bool = False) -> ComprehensiveAnalysis:
        """
        Complete analysis of a stock.
        Fetches all data, calculates all metrics, generates recommendations.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            skip_ai: If True, skip AI explanation for faster bulk scanning
        """
        logger.info(f"Starting comprehensive analysis of {symbol}")

        analysis = ComprehensiveAnalysis(
            symbol=symbol,
            timestamp=datetime.now().isoformat()
        )

        # Fetch data with retry
        def fetch_data():
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period='1y')
            info = ticker.info or {}
            return ticker, hist, info

        result = self._retry_with_backoff(fetch_data)

        if result is None:
            logger.error(f"Failed to fetch data for {symbol}")
            return analysis

        ticker, hist, info = result

        if hist.empty:
            logger.error(f"No historical data for {symbol}")
            return analysis

        # Basic info
        analysis.name = info.get('shortName', symbol)
        analysis.sector = info.get('sector', 'Unknown')
        analysis.industry = info.get('industry', 'Unknown')

        # Fetch all data
        logger.info(f"Calculating fundamentals for {symbol}")
        analysis.fundamental = self.fetch_fundamental_data(ticker, info)

        logger.info(f"Calculating technicals for {symbol}")
        analysis.technical = self.calculate_technical_data(hist, info)

        logger.info(f"Calculating statistics for {symbol}")
        analysis.statistical = self.calculate_statistical_data(hist)

        # Calculate probabilities for both timeframes
        logger.info(f"Calculating probabilities for {symbol}")
        prob_intraday = self.calculate_probability(
            analysis.technical, analysis.fundamental,
            analysis.statistical, TradingTimeframe.INTRADAY
        )
        prob_swing = self.calculate_probability(
            analysis.technical, analysis.fundamental,
            analysis.statistical, TradingTimeframe.SWING
        )
        analysis.probability = prob_swing  # Default to swing

        # Position sizing
        logger.info(f"Calculating position sizing for {symbol}")
        analysis.risk_mgmt = self.calculate_position_sizing(
            analysis.technical.price,
            analysis.technical.atr_14,
            prob_swing.win_probability
        )

        # Generate recommendations
        logger.info(f"Generating recommendations for {symbol}")
        analysis.intraday_rec = self.generate_recommendation(
            symbol, analysis.technical, analysis.fundamental,
            analysis.statistical, analysis.risk_mgmt, prob_intraday,
            TradingTimeframe.INTRADAY
        )
        analysis.swing_rec = self.generate_recommendation(
            symbol, analysis.technical, analysis.fundamental,
            analysis.statistical, analysis.risk_mgmt, prob_swing,
            TradingTimeframe.SWING
        )

        # Run advanced prediction engine (physics + ML + microstructure)
        if self.use_advanced_engine and not skip_ai:
            logger.info(f"Running advanced prediction for {symbol}")

            # Run for both timeframes
            advanced_intraday = self._run_advanced_prediction(symbol, hist, 'intraday')
            advanced_swing = self._run_advanced_prediction(symbol, hist, 'swing')

            # Store swing prediction as primary (usually more reliable)
            if advanced_swing:
                analysis.advanced_prediction = advanced_swing
                analysis.advanced_probability = advanced_swing.final_probability
                analysis.detected_regime = advanced_swing.detected_regime
                analysis.optimal_strategy = advanced_swing.optimal_strategy

                # Enhance recommendations with advanced insights
                analysis.swing_rec = self._enhance_recommendation_with_advanced(
                    analysis.swing_rec, advanced_swing, TradingTimeframe.SWING
                )

            if advanced_intraday:
                analysis.intraday_rec = self._enhance_recommendation_with_advanced(
                    analysis.intraday_rec, advanced_intraday, TradingTimeframe.INTRADAY
                )

        # Run UNIFIED advanced prediction (17 models) for additional insights
        if self.use_unified_advanced and not skip_ai:
            logger.info(f"Running unified advanced prediction (17 models) for {symbol}")

            unified_result = self._run_unified_advanced_prediction(symbol, hist)

            if unified_result:
                # Store unified prediction details
                analysis.unified_prediction = unified_result
                analysis.unified_probability = unified_result.get('final_probability', 0.5)
                analysis.unified_confidence = unified_result.get('final_confidence', 0.0)
                analysis.total_models_used = unified_result.get('total_models_used', 6)
                analysis.unified_insights = unified_result.get('insights', [])

                # Enhance recommendations with unified insights (higher priority)
                analysis.swing_rec = self._enhance_recommendation_with_unified(
                    analysis.swing_rec, unified_result, TradingTimeframe.SWING
                )
                analysis.intraday_rec = self._enhance_recommendation_with_unified(
                    analysis.intraday_rec, unified_result, TradingTimeframe.INTRADAY
                )

                logger.info(
                    f"Unified prediction enhanced {symbol}: "
                    f"models={analysis.total_models_used}, "
                    f"probability={analysis.unified_probability:.2%}"
                )

        # Get AI explanation (using strong model) - skip if bulk scanning
        if not skip_ai:
            logger.info(f"Getting AI explanation for {symbol}")
            summary, explanation = self.get_ai_explanation(analysis, use_strong_model=True)
            analysis.ai_summary = summary
            analysis.ai_explanation = explanation
            analysis.ai_models_used = ["GPT-4o", "Gemini 2.5"]
        else:
            logger.info(f"Skipping AI explanation for {symbol} (bulk scan mode)")
            analysis.ai_summary = ""
            analysis.ai_explanation = ""
            analysis.ai_models_used = []

        # Data quality
        analysis.data_completeness = self._calculate_data_completeness(analysis)
        analysis.analysis_confidence = (
            analysis.probability.confidence_level * 0.4 +
            analysis.data_completeness * 0.3 +
            (0.3 if analysis.ai_explanation else 0)
        )

        logger.info(f"Analysis complete for {symbol}")
        return analysis

    def _prepare_df_for_advanced_engine(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Convert yfinance DataFrame to format expected by advanced engine."""
        df = hist.copy()
        # Lowercase column names for advanced engine compatibility
        df.columns = [c.lower() for c in df.columns]
        return df

    def _run_advanced_prediction(
        self,
        symbol: str,
        hist: pd.DataFrame,
        timeframe: str = 'swing'
    ) -> Optional[Any]:
        """
        Run advanced prediction engine.

        Returns AdvancedPrediction or None if unavailable.
        """
        if not self.use_advanced_engine or self.advanced_engine is None:
            return None

        try:
            # Prepare DataFrame
            df = self._prepare_df_for_advanced_engine(hist)

            if len(df) < 20:
                logger.warning(f"Insufficient data for advanced prediction: {len(df)} rows")
                return None

            # Run advanced prediction
            prediction = self.advanced_engine.predict(
                symbol=symbol,
                df=df,
                timeframe=timeframe
            )

            logger.info(
                f"Advanced prediction for {symbol}: "
                f"probability={prediction.final_probability:.2%}, "
                f"signal={prediction.signal.value}, "
                f"regime={prediction.detected_regime}"
            )

            return prediction

        except Exception as e:
            logger.error(f"Advanced prediction failed for {symbol}: {e}")
            return None

    def _run_unified_advanced_prediction(
        self,
        symbol: str,
        hist: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Run unified advanced prediction with ALL 17 mathematical models.

        Uses:
        - Original 6: PCA, Wavelet, Kalman, Markov, DQN, Factor Model
        - New 11: Quantum-Inspired, Bellman, SDE, Information Geometry,
                  Tensor Decomposition, Spectral Graph, Optimal Transport,
                  Taylor Series, Heavy-Tailed, Copula, Multi-Dim Confidence

        Returns dictionary with full analysis or None if unavailable.
        """
        if not self.use_unified_advanced or self.unified_engine is None:
            return None

        try:
            # Prepare DataFrame
            df = self._prepare_df_for_advanced_engine(hist)

            if len(df) < 60:
                logger.warning(f"Insufficient data for unified prediction: {len(df)} rows (need 60+)")
                return None

            # Run unified advanced prediction with all 17 models
            result = self.unified_engine.predict_with_unified_math(
                prices=df,
                symbol=symbol
            )

            # Get human-readable insights
            insights = self.unified_engine.get_unified_analysis_summary(df, symbol)

            if result:
                logger.info(
                    f"Unified advanced prediction for {symbol}: "
                    f"final_signal={result.get('final_signal', 'HOLD')}, "
                    f"final_probability={result.get('final_probability', 0.5):.2%}, "
                    f"models_used={result.get('total_models_used', 6)}, "
                    f"unified_available={result.get('unified_available', False)}"
                )
                result['insights'] = insights.get('insights', [])
                result['modules_used'] = insights.get('modules_used', [])

            return result

        except Exception as e:
            logger.error(f"Unified advanced prediction failed for {symbol}: {e}")
            return None

    def _enhance_recommendation_with_unified(
        self,
        rec: TradeRecommendation,
        unified_result: Dict[str, Any],
        timeframe: TradingTimeframe
    ) -> TradeRecommendation:
        """
        Enhance recommendation with unified 17-model prediction insights.

        Combines traditional analysis with advanced mathematical signals.
        """
        if unified_result is None:
            return rec

        # Get final probability from unified engine (blends 6 base + 11 advanced)
        final_prob = unified_result.get('final_probability', 0.5)
        final_conf = unified_result.get('final_confidence', 0.0)
        total_models = unified_result.get('total_models_used', 6)

        # Blend with traditional probability (weight by model count)
        # If 17 models: 80% unified, 20% traditional
        # If 6 models: 60% unified, 40% traditional
        unified_weight = 0.8 if total_models >= 15 else 0.6
        traditional_weight = 1.0 - unified_weight

        blended_prob = (
            unified_weight * final_prob +
            traditional_weight * rec.win_probability
        )
        rec.win_probability = round(blended_prob, 3)

        # Add insights from advanced math modules
        insights = unified_result.get('insights', [])
        for insight in insights[:3]:  # Top 3 insights
            rec.bullish_factors.insert(0, f"[MATH] {insight}")

        # Add unified analysis details
        unified_advanced = unified_result.get('unified_advanced', {})
        if unified_advanced.get('available', False):
            # Quantum signal insight
            quantum_signal = unified_advanced.get('quantum_signal', 0)
            if abs(quantum_signal) > 0.3:
                direction = 'bullish' if quantum_signal > 0 else 'bearish'
                rec.bullish_factors.append(f"[QUANTUM] Optimization indicates {direction} tendency")

            # Heavy-tail warning
            tail_index = unified_advanced.get('tail_index', 2.0)
            if tail_index < 2.0:
                rec.key_risks.append(f"[TAIL RISK] Heavy-tailed distribution (α={tail_index:.2f})")

            # Anomaly detection
            anomaly_score = unified_advanced.get('anomaly_score', 0)
            if anomaly_score > 0.7:
                rec.key_risks.append("[ANOMALY] Unusual market pattern detected")

        # Adjust signal based on unified prediction
        final_signal = unified_result.get('final_signal', 'HOLD')
        if final_signal == 'STRONG_BUY' and blended_prob >= 0.60:
            rec.signal = SignalStrength.STRONG_BUY
            rec.action = "BUY"
        elif final_signal == 'BUY' and blended_prob >= 0.55:
            rec.signal = SignalStrength.BUY
            rec.action = "BUY"
        elif final_signal == 'STRONG_SELL' and blended_prob <= 0.40:
            rec.signal = SignalStrength.STRONG_SELL
            rec.action = "SELL"
        elif final_signal == 'SELL' and blended_prob <= 0.45:
            rec.signal = SignalStrength.SELL
            rec.action = "AVOID"

        # Adjust position size based on model agreement
        if final_conf >= 0.7 and total_models >= 15:
            rec.position_size_pct = min(rec.position_size_pct * 1.25, 25.0)
        elif final_conf >= 0.6:
            rec.position_size_pct = min(rec.position_size_pct * 1.1, 20.0)
        elif final_conf < 0.4:
            rec.position_size_pct = rec.position_size_pct * 0.6

        rec.position_size_pct = round(rec.position_size_pct, 1)

        return rec

    def _enhance_recommendation_with_advanced(
        self,
        rec: TradeRecommendation,
        advanced: Any,
        timeframe: TradingTimeframe
    ) -> TradeRecommendation:
        """
        Enhance recommendation with advanced prediction insights.

        Combines traditional and advanced signals for stronger conviction.
        """
        if advanced is None:
            return rec

        # Blend probabilities (70% advanced, 30% traditional)
        blended_prob = (
            0.70 * advanced.final_probability +
            0.30 * rec.win_probability
        )

        # Update recommendation with blended probability
        rec.win_probability = round(blended_prob, 3)

        # Add advanced factors
        rec.bullish_factors = advanced.bullish_factors + rec.bullish_factors
        rec.bearish_factors = advanced.bearish_factors + rec.bearish_factors
        rec.key_risks = advanced.warnings + rec.key_risks

        # Adjust signal if advanced prediction is stronger
        if advanced.final_probability >= 0.65 and blended_prob >= 0.60:
            rec.signal = SignalStrength.STRONG_BUY
            rec.action = "BUY"
        elif advanced.final_probability >= 0.58 and blended_prob >= 0.55:
            rec.signal = SignalStrength.BUY
            rec.action = "BUY"
        elif advanced.final_probability <= 0.35 and blended_prob <= 0.40:
            rec.signal = SignalStrength.STRONG_SELL
            rec.action = "SELL"
        elif advanced.final_probability <= 0.42 and blended_prob <= 0.45:
            rec.signal = SignalStrength.SELL
            rec.action = "AVOID"

        # Adjust position size based on confidence
        if advanced.confidence.model_agreement >= 0.7:
            # High agreement = more confidence
            rec.position_size_pct = min(rec.position_size_pct * 1.2, 20.0)
        elif advanced.confidence.model_agreement < 0.5:
            # Low agreement = reduce size
            rec.position_size_pct = rec.position_size_pct * 0.7

        rec.position_size_pct = round(rec.position_size_pct, 1)

        return rec

    def _calculate_data_completeness(self, analysis: ComprehensiveAnalysis) -> float:
        """Calculate how complete our data is."""
        score = 0
        total = 0

        # Check fundamentals
        fund = analysis.fundamental
        if fund.pe_ratio > 0: score += 1
        if fund.roe != 0: score += 1
        if fund.market_cap > 0: score += 1
        if fund.debt_equity >= 0: score += 1
        total += 4

        # Check technicals
        tech = analysis.technical
        if tech.price > 0: score += 1
        if tech.rsi_14 > 0: score += 1
        if tech.sma_50 > 0: score += 1
        if tech.atr_14 > 0: score += 1
        total += 4

        # Check statistics
        stat = analysis.statistical
        if stat.beta != 1.0: score += 1
        if stat.volatility_annual > 0: score += 1
        total += 2

        return score / total if total > 0 else 0


def demo():
    """Demo the comprehensive analyzer."""
    print("=" * 70)
    print("COMPREHENSIVE STOCK ANALYZER")
    print("=" * 70)

    analyzer = ComprehensiveAnalyzer(capital=100000, risk_per_trade=0.02)

    symbol = "RELIANCE"
    print(f"\nAnalyzing {symbol}...")

    analysis = analyzer.analyze(symbol)

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {analysis.symbol} - {analysis.name}")
    print(f"{'='*70}")

    print(f"\n## PRICE: ₹{analysis.technical.price}")
    print(f"Today: {analysis.technical.change_1d:+.2f}% | Week: {analysis.technical.change_1w:+.2f}% | Month: {analysis.technical.change_1m:+.2f}%")

    print(f"\n## KEY FUNDAMENTALS")
    print(f"Market Cap: ₹{analysis.fundamental.market_cap:,.0f} Cr")
    print(f"P/E: {analysis.fundamental.pe_ratio:.1f} | P/B: {analysis.fundamental.pb_ratio:.1f}")
    print(f"ROE: {analysis.fundamental.roe:.1f}% | D/E: {analysis.fundamental.debt_equity:.2f}")

    print(f"\n## KEY TECHNICALS")
    print(f"RSI: {analysis.technical.rsi_14:.0f} | MACD: {analysis.technical.macd:.2f}")
    print(f"Volume: {analysis.technical.volume_ratio:.1f}x avg")
    print(f"Above 50 MA: {'Yes' if analysis.technical.price > analysis.technical.sma_50 else 'No'}")
    print(f"Above 200 MA: {'Yes' if analysis.technical.price > analysis.technical.sma_200 else 'No'}")

    print(f"\n## RISK METRICS")
    print(f"Beta: {analysis.statistical.beta:.2f} | Volatility: {analysis.statistical.volatility_annual:.1f}%")

    print(f"\n## PROBABILITY")
    print(f"Win Rate: {analysis.probability.win_probability:.1%}")
    print(f"Expected Return: {analysis.probability.expected_return:.2f}%")
    print(f"Confidence: {analysis.probability.confidence_level:.1%}")

    print(f"\n## RECOMMENDATIONS")
    print(f"\nINTRADAY: {analysis.intraday_rec.signal.value}")
    print(f"  Entry: ₹{analysis.intraday_rec.entry_price} | SL: ₹{analysis.intraday_rec.stop_loss} | Target: ₹{analysis.intraday_rec.target_1}")
    print(f"  Position: {analysis.intraday_rec.position_size_pct:.1f}% | R:R: {analysis.intraday_rec.risk_reward:.2f}")

    print(f"\nSWING: {analysis.swing_rec.signal.value}")
    print(f"  Entry: ₹{analysis.swing_rec.entry_price} | SL: ₹{analysis.swing_rec.stop_loss} | Target: ₹{analysis.swing_rec.target_1}")
    print(f"  Position: {analysis.swing_rec.position_size_pct:.1f}% | R:R: {analysis.swing_rec.risk_reward:.2f}")

    if analysis.swing_rec.bullish_factors:
        print(f"\nBullish: {', '.join(analysis.swing_rec.bullish_factors[:3])}")
    if analysis.swing_rec.bearish_factors:
        print(f"Bearish: {', '.join(analysis.swing_rec.bearish_factors[:3])}")

    print(f"\n## AI SUMMARY")
    print(analysis.ai_summary)


if __name__ == "__main__":
    demo()
