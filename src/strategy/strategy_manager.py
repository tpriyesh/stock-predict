"""
StrategyManager - Unified Strategy Selection for All Use Cases

Handles different investment strategies with appropriate logic:
1. INTRADAY - Technical signals, high volume, quick trades
2. SWING - Multi-factor model, 5-day horizon
3. LONG_TERM - Fundamentals, quality investing, 1-5 year horizon
4. MULTI_BAGGER - Penny/small-cap with high growth potential
5. SECTOR_BEST - Find best companies in each sector

Each strategy uses DIFFERENT logic because:
- Intraday: Technical > Fundamentals (price action matters)
- Long-term: Fundamentals > Technical (business quality matters)
- Multi-bagger: Growth + Value + Small Size (asymmetric upside)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import yfinance as yf

from config.settings import get_settings


class StrategyType(Enum):
    """Investment strategy types."""
    INTRADAY = "intraday"      # Same-day trades
    SWING = "swing"            # 2-10 days
    POSITIONAL = "positional"  # 2-8 weeks
    LONG_TERM = "long_term"    # 1-5 years
    MULTI_BAGGER = "multi_bagger"  # High growth potential


class RiskProfile(Enum):
    """Investor risk profile."""
    CONSERVATIVE = "conservative"  # Low risk, stable returns
    MODERATE = "moderate"          # Balanced risk/return
    AGGRESSIVE = "aggressive"      # High risk, high return potential


@dataclass
class StrategyRecommendation:
    """Recommendation from strategy analysis."""
    symbol: str
    name: str
    sector: str
    price: float

    # Strategy-specific
    strategy: StrategyType
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1

    # Scores (different weights per strategy)
    technical_score: float
    fundamental_score: float
    growth_score: float
    value_score: float
    quality_score: float
    momentum_score: float

    # Risk metrics
    volatility: float
    max_drawdown: float
    risk_level: str  # LOW, MEDIUM, HIGH

    # Trade setup
    entry_price: float
    stop_loss: float
    targets: List[float]
    holding_period: str

    # Expected outcomes
    expected_return_pct: float
    probability_of_profit: float
    risk_reward_ratio: float

    # Multi-bagger specific
    multi_bagger_score: Optional[float] = None
    growth_runway: Optional[str] = None

    # Reasoning
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    recommendation_reason: str = ""


@dataclass
class SectorAnalysis:
    """Analysis of a sector."""
    sector: str
    overall_outlook: str  # BULLISH, BEARISH, NEUTRAL
    top_picks: List[StrategyRecommendation]
    sector_momentum: float
    sector_pe: float
    global_factors: List[str]
    risks: List[str]


class StrategyManager:
    """
    Unified strategy manager for all investment approaches.

    Key Principle: Different strategies need DIFFERENT logic.

    INTRADAY:
        - Weight: 80% Technical, 20% Momentum
        - Focus: Volume spikes, RSI, intraday patterns
        - Ignore: P/E ratios, long-term growth

    LONG_TERM:
        - Weight: 70% Fundamental, 30% Technical
        - Focus: ROE, earnings growth, competitive moat
        - Ignore: Daily price fluctuations

    MULTI_BAGGER:
        - Weight: 40% Growth, 30% Value, 30% Quality
        - Focus: Small market cap, high growth, reasonable price
        - Look for: 10x potential over 3-5 years
    """

    # Sector mapping for Indian stocks
    SECTOR_MAP = {
        # IT
        'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
        'LTIM': 'IT', 'MPHASIS': 'IT', 'COFORGE': 'IT', 'PERSISTENT': 'IT',

        # Banking
        'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking',
        'KOTAKBANK': 'Banking', 'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking',
        'BANKBARODA': 'Banking', 'PNB': 'Banking', 'CANBK': 'Banking',

        # NBFC
        'BAJFINANCE': 'NBFC', 'BAJAJFINSV': 'NBFC', 'SHRIRAMFIN': 'NBFC',
        'CHOLAFIN': 'NBFC', 'MUTHOOTFIN': 'NBFC', 'M&MFIN': 'NBFC',

        # Auto
        'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto',
        'BAJAJ-AUTO': 'Auto', 'HEROMOTOCO': 'Auto', 'EICHERMOT': 'Auto',
        'TVSMOTOR': 'Auto', 'ASHOKLEY': 'Auto',

        # Pharma
        'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma',
        'DIVISLAB': 'Pharma', 'LUPIN': 'Pharma', 'AUROPHARMA': 'Pharma',
        'BIOCON': 'Pharma', 'TORNTPHARM': 'Pharma',

        # FMCG
        'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
        'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
        'GODREJCP': 'FMCG', 'TATACONSUM': 'FMCG', 'COLPAL': 'FMCG',

        # Energy
        'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy',
        'IOC': 'Energy', 'GAIL': 'Energy', 'PETRONET': 'Energy',

        # Metals
        'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals',
        'VEDL': 'Metals', 'NMDC': 'Metals', 'COALINDIA': 'Metals',
        'JINDALSTEL': 'Metals', 'SAIL': 'Metals',

        # Infrastructure
        'LT': 'Infrastructure', 'ADANIPORTS': 'Infrastructure',
        'ADANIENT': 'Infrastructure', 'ULTRACEMCO': 'Infrastructure',
        'GRASIM': 'Infrastructure', 'ACC': 'Infrastructure',
        'AMBUJACEM': 'Infrastructure', 'DLF': 'Infrastructure',

        # Power
        'NTPC': 'Power', 'POWERGRID': 'Power', 'TATAPOWER': 'Power',
        'ADANIGREEN': 'Power', 'NHPC': 'Power', 'SJVN': 'Power',

        # Telecom
        'BHARTIARTL': 'Telecom', 'IDEA': 'Telecom',

        # Consumer
        'TITAN': 'Consumer', 'ASIANPAINT': 'Consumer', 'PIDILITIND': 'Consumer',
        'HAVELLS': 'Consumer', 'VOLTAS': 'Consumer', 'CROMPTON': 'Consumer',

        # Defence/PSU
        'HAL': 'Defence', 'BEL': 'Defence', 'BHEL': 'Defence',

        # Others
        'IRCTC': 'Travel', 'INDIGO': 'Travel',
    }

    # Multi-bagger criteria thresholds
    MULTI_BAGGER_CRITERIA = {
        'max_market_cap_cr': 10000,      # Below 10,000 Cr market cap
        'min_roe': 0.15,                  # At least 15% ROE
        'min_revenue_growth': 0.15,       # At least 15% revenue growth
        'max_pe': 40,                     # Reasonable valuation
        'min_pe': 5,                      # Not too cheap (value trap)
        'max_debt_equity': 1.0,           # Low debt
        'min_promoter_holding': 0.40,     # Strong promoter stake
    }

    def __init__(self):
        self.settings = get_settings()
        self._cache = {}

    def analyze(self,
                symbol: str,
                strategy: StrategyType = StrategyType.SWING,
                risk_profile: RiskProfile = RiskProfile.MODERATE) -> Optional[StrategyRecommendation]:
        """
        Analyze a stock using the specified strategy.

        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            strategy: Investment strategy type
            risk_profile: Investor risk tolerance

        Returns:
            StrategyRecommendation with all analysis
        """
        try:
            # Fetch data
            ticker = yf.Ticker(f"{symbol}.NS")

            # Get historical data - different periods for different strategies
            if strategy == StrategyType.INTRADAY:
                hist = ticker.history(period='3mo', interval='1d')
            elif strategy == StrategyType.LONG_TERM:
                hist = ticker.history(period='5y')
            else:
                hist = ticker.history(period='1y')

            if hist.empty or len(hist) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            info = ticker.info or {}

            # Calculate all scores
            technical = self._calculate_technical_score(hist)
            fundamental = self._calculate_fundamental_score(info)
            growth = self._calculate_growth_score(info)
            value = self._calculate_value_score(info)
            quality = self._calculate_quality_score(info)
            momentum = self._calculate_momentum_score(hist)

            # Weight scores based on strategy
            weights = self._get_strategy_weights(strategy)

            composite = (
                technical * weights['technical'] +
                fundamental * weights['fundamental'] +
                growth * weights['growth'] +
                value * weights['value'] +
                quality * weights['quality'] +
                momentum * weights['momentum']
            )

            # Calculate risk metrics
            volatility = self._calculate_volatility(hist)
            max_dd = self._calculate_max_drawdown(hist)
            risk_level = self._assess_risk_level(volatility, max_dd, strategy)

            # Generate signal
            signal, confidence = self._generate_signal(
                composite, technical, fundamental, strategy, risk_profile
            )

            # Calculate trade levels
            current_price = hist['Close'].iloc[-1]
            atr = self._calculate_atr(hist)

            trade_setup = self._calculate_trade_setup(
                current_price, atr, signal, strategy, risk_profile
            )

            # Multi-bagger analysis (if applicable)
            mb_score = None
            growth_runway = None
            if strategy == StrategyType.MULTI_BAGGER:
                mb_score = self._calculate_multibagger_score(info, growth, quality)
                growth_runway = self._assess_growth_runway(info)

            # Generate factors
            bullish, bearish = self._generate_factors(
                info, technical, fundamental, growth, momentum, strategy
            )

            # Generate recommendation reason
            reason = self._generate_recommendation_reason(
                symbol, signal, strategy, composite, bullish, bearish
            )

            return StrategyRecommendation(
                symbol=symbol,
                name=info.get('shortName', symbol),
                sector=self.SECTOR_MAP.get(symbol, info.get('sector', 'Unknown')),
                price=round(current_price, 2),
                strategy=strategy,
                signal=signal,
                confidence=round(confidence, 3),
                technical_score=round(technical, 3),
                fundamental_score=round(fundamental, 3),
                growth_score=round(growth, 3),
                value_score=round(value, 3),
                quality_score=round(quality, 3),
                momentum_score=round(momentum, 3),
                volatility=round(volatility, 4),
                max_drawdown=round(max_dd, 4),
                risk_level=risk_level,
                entry_price=trade_setup['entry'],
                stop_loss=trade_setup['stop_loss'],
                targets=trade_setup['targets'],
                holding_period=trade_setup['holding_period'],
                expected_return_pct=trade_setup['expected_return'],
                probability_of_profit=trade_setup['win_prob'],
                risk_reward_ratio=trade_setup['risk_reward'],
                multi_bagger_score=mb_score,
                growth_runway=growth_runway,
                bullish_factors=bullish,
                bearish_factors=bearish,
                recommendation_reason=reason
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _get_strategy_weights(self, strategy: StrategyType) -> Dict[str, float]:
        """Get scoring weights for each strategy."""
        if strategy == StrategyType.INTRADAY:
            return {
                'technical': 0.50,
                'fundamental': 0.00,  # Ignore for intraday
                'growth': 0.00,
                'value': 0.00,
                'quality': 0.00,
                'momentum': 0.50
            }
        elif strategy == StrategyType.SWING:
            return {
                'technical': 0.35,
                'fundamental': 0.15,
                'growth': 0.10,
                'value': 0.10,
                'quality': 0.10,
                'momentum': 0.20
            }
        elif strategy == StrategyType.POSITIONAL:
            return {
                'technical': 0.25,
                'fundamental': 0.25,
                'growth': 0.15,
                'value': 0.15,
                'quality': 0.10,
                'momentum': 0.10
            }
        elif strategy == StrategyType.LONG_TERM:
            return {
                'technical': 0.10,
                'fundamental': 0.30,
                'growth': 0.20,
                'value': 0.15,
                'quality': 0.20,
                'momentum': 0.05
            }
        elif strategy == StrategyType.MULTI_BAGGER:
            return {
                'technical': 0.10,
                'fundamental': 0.15,
                'growth': 0.35,  # Growth is key
                'value': 0.20,  # Value matters
                'quality': 0.15,
                'momentum': 0.05
            }
        else:
            return {k: 1/6 for k in ['technical', 'fundamental', 'growth',
                                      'value', 'quality', 'momentum']}

    def _calculate_technical_score(self, hist: pd.DataFrame) -> float:
        """Calculate technical analysis score."""
        try:
            close = hist['Close']
            current = close.iloc[-1]

            score = 0.5  # Start neutral

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != 0 else 50

            if 30 < rsi < 70:
                score += 0.1  # Healthy range
            elif rsi < 30:
                score += 0.15  # Oversold (potential bounce)
            else:
                score -= 0.1  # Overbought

            # Moving averages
            if len(hist) >= 50:
                ma50 = close.rolling(50).mean().iloc[-1]
                if current > ma50:
                    score += 0.1
                else:
                    score -= 0.05

            if len(hist) >= 200:
                ma200 = close.rolling(200).mean().iloc[-1]
                if current > ma200:
                    score += 0.1
                else:
                    score -= 0.05

            # Volume trend
            vol = hist['Volume']
            avg_vol = vol.rolling(20).mean().iloc[-1]
            recent_vol = vol.iloc[-5:].mean()
            if recent_vol > avg_vol * 1.2:
                score += 0.05  # Increasing interest

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()

            if macd.iloc[-1] > signal_line.iloc[-1]:
                score += 0.05
            else:
                score -= 0.05

            return max(0, min(1, score))

        except Exception as e:
            logger.warning(f"Technical score error: {e}")
            return 0.5

    def _calculate_fundamental_score(self, info: dict) -> float:
        """Calculate fundamental score."""
        score = 0.5

        try:
            # PE ratio
            pe = info.get('trailingPE')
            if pe:
                if 10 < pe < 25:
                    score += 0.1
                elif pe < 10:
                    score += 0.05  # Could be value trap
                elif pe > 40:
                    score -= 0.1

            # Profit margin
            margin = info.get('profitMargins')
            if margin:
                if margin > 0.15:
                    score += 0.1
                elif margin > 0.10:
                    score += 0.05
                elif margin < 0:
                    score -= 0.15

            # Debt to equity
            de = info.get('debtToEquity')
            if de is not None:
                if de < 50:
                    score += 0.1
                elif de > 150:
                    score -= 0.1

            # Current ratio
            cr = info.get('currentRatio')
            if cr:
                if cr > 1.5:
                    score += 0.05
                elif cr < 1:
                    score -= 0.1

            return max(0, min(1, score))

        except:
            return 0.5

    def _calculate_growth_score(self, info: dict) -> float:
        """Calculate growth score."""
        score = 0.5

        try:
            # Revenue growth
            rev_growth = info.get('revenueGrowth')
            if rev_growth:
                if rev_growth > 0.25:
                    score += 0.2
                elif rev_growth > 0.15:
                    score += 0.1
                elif rev_growth < 0:
                    score -= 0.15

            # Earnings growth
            earn_growth = info.get('earningsGrowth')
            if earn_growth:
                if earn_growth > 0.25:
                    score += 0.15
                elif earn_growth > 0.10:
                    score += 0.05
                elif earn_growth < 0:
                    score -= 0.1

            # Forward PE vs Trailing (growth expectation)
            fwd_pe = info.get('forwardPE')
            trail_pe = info.get('trailingPE')
            if fwd_pe and trail_pe and trail_pe > 0:
                if fwd_pe < trail_pe * 0.8:  # Expected earnings growth
                    score += 0.1

            return max(0, min(1, score))

        except:
            return 0.5

    def _calculate_value_score(self, info: dict) -> float:
        """Calculate value score (is stock cheap relative to fundamentals)."""
        score = 0.5

        try:
            # Price to book
            pb = info.get('priceToBook')
            if pb:
                if pb < 2:
                    score += 0.15
                elif pb < 3:
                    score += 0.05
                elif pb > 5:
                    score -= 0.1

            # PEG ratio (PE relative to growth)
            peg = info.get('pegRatio')
            if peg:
                if peg < 1:
                    score += 0.2  # Great value for growth
                elif peg < 1.5:
                    score += 0.1
                elif peg > 2:
                    score -= 0.1

            # Distance from 52-week high
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            high52 = info.get('fiftyTwoWeekHigh')
            if current and high52:
                pct_from_high = (current / high52 - 1) * 100
                if pct_from_high < -30:
                    score += 0.1  # Significant discount
                elif pct_from_high > -5:
                    score -= 0.05  # Near highs

            return max(0, min(1, score))

        except:
            return 0.5

    def _calculate_quality_score(self, info: dict) -> float:
        """Calculate business quality score."""
        score = 0.5

        try:
            # Return on Equity
            roe = info.get('returnOnEquity')
            if roe:
                if roe > 0.20:
                    score += 0.2
                elif roe > 0.15:
                    score += 0.1
                elif roe < 0.05:
                    score -= 0.15

            # Return on Assets
            roa = info.get('returnOnAssets')
            if roa:
                if roa > 0.10:
                    score += 0.1
                elif roa < 0.03:
                    score -= 0.1

            # Free cash flow
            fcf = info.get('freeCashflow')
            if fcf:
                if fcf > 0:
                    score += 0.1
                else:
                    score -= 0.1

            # Institutional holding (smart money)
            inst = info.get('heldPercentInstitutions')
            if inst:
                if inst > 0.50:
                    score += 0.05

            return max(0, min(1, score))

        except:
            return 0.5

    def _calculate_momentum_score(self, hist: pd.DataFrame) -> float:
        """Calculate price momentum score."""
        try:
            close = hist['Close']
            current = close.iloc[-1]

            score = 0.5

            # 1-week momentum
            if len(hist) >= 5:
                mom_1w = (current / close.iloc[-5] - 1) * 100
                if mom_1w > 5:
                    score += 0.15
                elif mom_1w > 2:
                    score += 0.05
                elif mom_1w < -5:
                    score -= 0.1

            # 1-month momentum
            if len(hist) >= 21:
                mom_1m = (current / close.iloc[-21] - 1) * 100
                if mom_1m > 10:
                    score += 0.15
                elif mom_1m > 5:
                    score += 0.05
                elif mom_1m < -10:
                    score -= 0.1

            # 3-month momentum
            if len(hist) >= 63:
                mom_3m = (current / close.iloc[-63] - 1) * 100
                if mom_3m > 15:
                    score += 0.1
                elif mom_3m < -15:
                    score -= 0.1

            return max(0, min(1, score))

        except:
            return 0.5

    def _calculate_volatility(self, hist: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        try:
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            return returns.std() * np.sqrt(252)
        except:
            return 0.25

    def _calculate_max_drawdown(self, hist: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        try:
            close = hist['Close']
            rolling_max = close.cummax()
            drawdown = (close - rolling_max) / rolling_max
            return abs(drawdown.min())
        except:
            return 0.20

    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = hist['High']
            low = hist['Low']
            close = hist['Close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]
        except:
            return hist['Close'].iloc[-1] * 0.02  # Default 2%

    def _assess_risk_level(self, volatility: float, max_dd: float,
                           strategy: StrategyType) -> str:
        """Assess overall risk level."""
        # Adjust thresholds by strategy
        if strategy == StrategyType.INTRADAY:
            vol_threshold = (0.30, 0.50)
            dd_threshold = (0.10, 0.20)
        elif strategy == StrategyType.LONG_TERM:
            vol_threshold = (0.25, 0.40)
            dd_threshold = (0.25, 0.40)
        else:
            vol_threshold = (0.25, 0.40)
            dd_threshold = (0.15, 0.30)

        vol_risk = 0 if volatility < vol_threshold[0] else (
            1 if volatility < vol_threshold[1] else 2
        )
        dd_risk = 0 if max_dd < dd_threshold[0] else (
            1 if max_dd < dd_threshold[1] else 2
        )

        combined = vol_risk + dd_risk
        if combined <= 1:
            return "LOW"
        elif combined <= 2:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_signal(self, composite: float, technical: float,
                         fundamental: float, strategy: StrategyType,
                         risk_profile: RiskProfile) -> Tuple[str, float]:
        """Generate trading signal and confidence."""
        # Adjust thresholds by risk profile
        if risk_profile == RiskProfile.CONSERVATIVE:
            buy_threshold = 0.65
            sell_threshold = 0.35
        elif risk_profile == RiskProfile.AGGRESSIVE:
            buy_threshold = 0.55
            sell_threshold = 0.45
        else:
            buy_threshold = 0.60
            sell_threshold = 0.40

        # For intraday, rely more on technical
        if strategy == StrategyType.INTRADAY:
            score = technical * 0.7 + composite * 0.3
        elif strategy == StrategyType.LONG_TERM:
            score = fundamental * 0.5 + composite * 0.5
        else:
            score = composite

        if score >= buy_threshold:
            signal = "BUY"
            confidence = min(1.0, 0.5 + (score - buy_threshold) * 2)
        elif score <= sell_threshold:
            signal = "SELL"
            confidence = min(1.0, 0.5 + (sell_threshold - score) * 2)
        else:
            signal = "HOLD"
            confidence = 0.5

        return signal, confidence

    def _calculate_trade_setup(self, price: float, atr: float, signal: str,
                               strategy: StrategyType,
                               risk_profile: RiskProfile) -> Dict:
        """Calculate trade entry, stop loss, and targets."""
        # ATR multipliers by risk profile
        if risk_profile == RiskProfile.CONSERVATIVE:
            sl_mult = 1.0
            target_mults = [1.5, 2.5]
        elif risk_profile == RiskProfile.AGGRESSIVE:
            sl_mult = 2.0
            target_mults = [2.0, 4.0]
        else:
            sl_mult = 1.5
            target_mults = [2.0, 3.0]

        if signal == "BUY":
            stop_loss = price - (sl_mult * atr)
            targets = [price + (m * atr) for m in target_mults]
        elif signal == "SELL":
            stop_loss = price + (sl_mult * atr)
            targets = [price - (m * atr) for m in target_mults]
        else:
            stop_loss = price - (sl_mult * atr)
            targets = [price + atr, price + 2*atr]

        # Holding periods by strategy
        holding_periods = {
            StrategyType.INTRADAY: "Same day",
            StrategyType.SWING: "3-10 days",
            StrategyType.POSITIONAL: "2-8 weeks",
            StrategyType.LONG_TERM: "1-5 years",
            StrategyType.MULTI_BAGGER: "3-5 years"
        }

        risk = abs(price - stop_loss)
        reward = abs(targets[0] - price)
        rr = reward / risk if risk > 0 else 0

        # Win probability (adjusted by risk/reward)
        base_prob = 0.50
        if rr > 2:
            win_prob = base_prob + 0.05
        elif rr > 1.5:
            win_prob = base_prob + 0.02
        else:
            win_prob = base_prob - 0.05

        expected_return = (win_prob * reward - (1-win_prob) * risk) / price * 100

        return {
            'entry': round(price, 2),
            'stop_loss': round(stop_loss, 2),
            'targets': [round(t, 2) for t in targets],
            'holding_period': holding_periods.get(strategy, "Variable"),
            'risk_reward': round(rr, 2),
            'win_prob': round(win_prob, 3),
            'expected_return': round(expected_return, 2)
        }

    def _calculate_multibagger_score(self, info: dict, growth_score: float,
                                     quality_score: float) -> float:
        """
        Calculate multi-bagger potential score.

        Multi-baggers typically have:
        1. Small market cap (room to grow)
        2. High revenue growth (>20% p.a.)
        3. Strong ROE (>15%)
        4. Reasonable valuation (PEG < 1.5)
        5. Low debt
        """
        score = 0.0
        weights_sum = 0.0

        # Market cap (smaller = more potential, but not too small)
        mc = info.get('marketCap', 0) / 1e7  # In Crores
        if 500 < mc < 5000:
            score += 0.25  # Sweet spot
            weights_sum += 0.25
        elif mc < 500:
            score += 0.15  # Micro cap - higher risk
            weights_sum += 0.25
        elif mc < 10000:
            score += 0.10  # Small cap
            weights_sum += 0.25
        else:
            weights_sum += 0.25  # Large cap - less multi-bagger potential

        # Growth trajectory
        score += growth_score * 0.30
        weights_sum += 0.30

        # Business quality
        score += quality_score * 0.25
        weights_sum += 0.25

        # Valuation relative to growth
        peg = info.get('pegRatio')
        if peg and peg < 1:
            score += 0.15
        elif peg and peg < 1.5:
            score += 0.10
        weights_sum += 0.15

        # Promoter holding
        insider = info.get('heldPercentInsiders')
        if insider and insider > 0.50:
            score += 0.05
        weights_sum += 0.05

        return score / weights_sum if weights_sum > 0 else 0.5

    def _assess_growth_runway(self, info: dict) -> str:
        """Assess how much growth runway the company has."""
        mc = info.get('marketCap', 0) / 1e7  # Crores
        rev_growth = info.get('revenueGrowth', 0)

        if mc < 1000 and rev_growth > 0.25:
            return "Very High - 5-10x potential over 5 years"
        elif mc < 5000 and rev_growth > 0.15:
            return "High - 3-5x potential over 5 years"
        elif mc < 10000 and rev_growth > 0.10:
            return "Moderate - 2-3x potential over 5 years"
        else:
            return "Limited - Steady growth expected"

    def _generate_factors(self, info: dict, technical: float, fundamental: float,
                          growth: float, momentum: float,
                          strategy: StrategyType) -> Tuple[List[str], List[str]]:
        """Generate bullish and bearish factors."""
        bullish = []
        bearish = []

        # Technical factors
        if technical > 0.6:
            bullish.append("Strong technical setup")
        elif technical < 0.4:
            bearish.append("Weak technical indicators")

        # Fundamental factors
        if fundamental > 0.6:
            bullish.append("Solid fundamentals")
        elif fundamental < 0.4:
            bearish.append("Weak fundamentals")

        # Growth
        if growth > 0.6:
            bullish.append("High growth trajectory")
        elif growth < 0.4:
            bearish.append("Declining growth")

        # Momentum
        if momentum > 0.6:
            bullish.append("Positive price momentum")
        elif momentum < 0.4:
            bearish.append("Negative momentum")

        # Specific metrics
        pe = info.get('trailingPE')
        if pe:
            if pe < 15:
                bullish.append(f"Low P/E ({pe:.1f})")
            elif pe > 40:
                bearish.append(f"High P/E ({pe:.1f})")

        roe = info.get('returnOnEquity')
        if roe:
            if roe > 0.20:
                bullish.append(f"High ROE ({roe*100:.0f}%)")
            elif roe < 0.05:
                bearish.append(f"Low ROE ({roe*100:.0f}%)")

        de = info.get('debtToEquity')
        if de is not None:
            if de < 30:
                bullish.append("Low debt")
            elif de > 150:
                bearish.append("High debt levels")

        return bullish[:5], bearish[:5]

    def _generate_recommendation_reason(self, symbol: str, signal: str,
                                         strategy: StrategyType, composite: float,
                                         bullish: List[str],
                                         bearish: List[str]) -> str:
        """Generate human-readable recommendation reason."""
        strategy_name = strategy.value.replace('_', ' ').title()

        if signal == "BUY":
            reason = f"{symbol} is a {signal} for {strategy_name} strategy "
            reason += f"(composite score: {composite:.0%}). "
            if bullish:
                reason += f"Key strengths: {', '.join(bullish[:3])}. "
            if bearish:
                reason += f"Watch: {', '.join(bearish[:2])}."
        elif signal == "SELL":
            reason = f"{symbol} shows weakness for {strategy_name} strategy "
            reason += f"(score: {composite:.0%}). "
            if bearish:
                reason += f"Concerns: {', '.join(bearish[:3])}."
        else:
            reason = f"{symbol} is neutral for {strategy_name} "
            reason += f"(score: {composite:.0%}). Wait for clearer signals."

        return reason

    def find_multibaggers(self, symbols: List[str],
                          min_score: float = 0.6) -> List[StrategyRecommendation]:
        """
        Scan for potential multi-bagger stocks.

        Multi-baggers are stocks that can multiply 5-10x over 3-5 years.
        Criteria:
        - Small to mid cap (< 10,000 Cr)
        - High growth (>15% revenue growth)
        - Quality business (ROE > 15%)
        - Reasonable valuation (PEG < 1.5)
        """
        logger.info(f"Scanning {len(symbols)} stocks for multi-bagger potential...")

        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.analyze, sym, StrategyType.MULTI_BAGGER
                ): sym for sym in symbols
            }

            for future in as_completed(futures):
                try:
                    rec = future.result()
                    if rec and rec.multi_bagger_score and rec.multi_bagger_score >= min_score:
                        results.append(rec)
                except Exception as e:
                    pass

        # Sort by multi-bagger score
        results.sort(key=lambda x: x.multi_bagger_score or 0, reverse=True)

        return results

    def analyze_sector(self, sector: str,
                       symbols: Optional[List[str]] = None) -> SectorAnalysis:
        """
        Analyze a sector and find best companies within it.

        Args:
            sector: Sector name (e.g., 'IT', 'Banking')
            symbols: Optional list of symbols to analyze

        Returns:
            SectorAnalysis with top picks and outlook
        """
        # Get sector stocks
        if symbols is None:
            symbols = [sym for sym, sec in self.SECTOR_MAP.items() if sec == sector]

        if not symbols:
            logger.warning(f"No stocks found for sector: {sector}")
            return SectorAnalysis(
                sector=sector,
                overall_outlook="UNKNOWN",
                top_picks=[],
                sector_momentum=0.5,
                sector_pe=0,
                global_factors=[],
                risks=[]
            )

        logger.info(f"Analyzing {len(symbols)} stocks in {sector} sector...")

        # Analyze all stocks
        results = []
        sector_returns = []
        sector_pes = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.analyze, sym, StrategyType.LONG_TERM
                ): sym for sym in symbols
            }

            for future in as_completed(futures):
                try:
                    rec = future.result()
                    if rec:
                        results.append(rec)
                        sector_returns.append(rec.momentum_score)
                        # Get PE from info
                except:
                    pass

        # Sort by composite score
        results.sort(key=lambda x: (
            x.fundamental_score * 0.4 +
            x.quality_score * 0.3 +
            x.growth_score * 0.3
        ), reverse=True)

        # Determine sector outlook
        avg_momentum = np.mean(sector_returns) if sector_returns else 0.5
        if avg_momentum > 0.6:
            outlook = "BULLISH"
        elif avg_momentum < 0.4:
            outlook = "BEARISH"
        else:
            outlook = "NEUTRAL"

        # Global factors affecting sector
        global_factors = self._get_sector_global_factors(sector)
        risks = self._get_sector_risks(sector)

        return SectorAnalysis(
            sector=sector,
            overall_outlook=outlook,
            top_picks=results[:5],  # Top 5 picks
            sector_momentum=round(avg_momentum, 3),
            sector_pe=0,  # Would need aggregation
            global_factors=global_factors,
            risks=risks
        )

    def _get_sector_global_factors(self, sector: str) -> List[str]:
        """Get global factors affecting a sector."""
        factors_map = {
            'IT': ['US tech spending', 'USD/INR exchange rate', 'AI/Cloud demand'],
            'Banking': ['RBI interest rates', 'NPA trends', 'Credit growth'],
            'Pharma': ['US FDA approvals', 'Generic drug prices', 'API demand'],
            'Auto': ['Semiconductor supply', 'EV adoption', 'Fuel prices'],
            'FMCG': ['Rural demand', 'Inflation', 'Consumer sentiment'],
            'Metals': ['China demand', 'Global commodity prices', 'Infrastructure spend'],
            'Energy': ['Crude oil prices', 'OPEC decisions', 'Refining margins'],
            'Infrastructure': ['Govt capex', 'Cement prices', 'Project orders'],
            'Power': ['Power demand growth', 'Renewable push', 'Coal availability'],
            'Telecom': ['ARPU trends', '5G rollout', 'Regulatory changes'],
        }
        return factors_map.get(sector, ['Economic growth', 'Market sentiment'])

    def _get_sector_risks(self, sector: str) -> List[str]:
        """Get key risks for a sector."""
        risks_map = {
            'IT': ['US recession risk', 'Visa restrictions', 'Margin pressure'],
            'Banking': ['Asset quality deterioration', 'Rising NPAs', 'Rate cuts'],
            'Pharma': ['FDA warning letters', 'Price erosion', 'Patent cliffs'],
            'Auto': ['EV transition disruption', 'Supply chain issues', 'Demand slowdown'],
            'FMCG': ['Input cost inflation', 'Competition', 'Rural slowdown'],
            'Metals': ['China slowdown', 'Price volatility', 'Environmental regulations'],
            'Energy': ['Oil price crash', 'Green transition', 'Margin compression'],
        }
        return risks_map.get(sector, ['Regulatory changes', 'Economic slowdown'])

    def get_best_across_sectors(self,
                                strategy: StrategyType = StrategyType.LONG_TERM,
                                n_per_sector: int = 2) -> Dict[str, List[StrategyRecommendation]]:
        """
        Get best stocks from each sector.

        Returns:
            Dict mapping sector to top N recommendations
        """
        all_symbols = list(self.SECTOR_MAP.keys())
        sectors = set(self.SECTOR_MAP.values())

        results = {}

        for sector in sectors:
            sector_symbols = [s for s, sec in self.SECTOR_MAP.items() if sec == sector]

            sector_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self.analyze, sym, strategy): sym
                    for sym in sector_symbols
                }

                for future in as_completed(futures):
                    try:
                        rec = future.result()
                        if rec and rec.signal == "BUY":
                            sector_results.append(rec)
                    except:
                        pass

            # Sort and take top N
            sector_results.sort(key=lambda x: x.confidence, reverse=True)
            results[sector] = sector_results[:n_per_sector]

        return results


def demo():
    """Demonstrate the strategy manager."""
    print("=" * 70)
    print("STRATEGY MANAGER DEMO")
    print("=" * 70)

    manager = StrategyManager()

    # Test different strategies on same stock
    symbol = "RELIANCE"

    print(f"\n--- Analyzing {symbol} with different strategies ---\n")

    for strategy in [StrategyType.INTRADAY, StrategyType.SWING,
                     StrategyType.LONG_TERM, StrategyType.MULTI_BAGGER]:
        rec = manager.analyze(symbol, strategy)
        if rec:
            print(f"{strategy.value.upper()}:")
            print(f"  Signal: {rec.signal} (Confidence: {rec.confidence:.0%})")
            print(f"  Technical: {rec.technical_score:.2f} | "
                  f"Fundamental: {rec.fundamental_score:.2f}")
            print(f"  Entry: ₹{rec.entry_price} | SL: ₹{rec.stop_loss} | "
                  f"Target: ₹{rec.targets[0]}")
            print(f"  Holding: {rec.holding_period}")
            print()

    # Find multi-baggers
    print("\n--- Searching for Multi-Baggers ---\n")
    penny_stocks = ['IRCTC', 'TATAPOWER', 'NMDC', 'VEDL', 'PNB',
                    'COALINDIA', 'SAIL', 'NHPC', 'SJVN']

    multibaggers = manager.find_multibaggers(penny_stocks, min_score=0.4)

    if multibaggers:
        print(f"Found {len(multibaggers)} potential multi-baggers:\n")
        for mb in multibaggers[:3]:
            print(f"  {mb.symbol}: Score {mb.multi_bagger_score:.0%}")
            print(f"    Growth Runway: {mb.growth_runway}")
            print(f"    Price: ₹{mb.price} | Confidence: {mb.confidence:.0%}")
            print()

    # Sector analysis
    print("\n--- Sector Analysis: IT ---\n")
    it_analysis = manager.analyze_sector('IT')
    print(f"Outlook: {it_analysis.overall_outlook}")
    print(f"Momentum: {it_analysis.sector_momentum:.0%}")
    print(f"Global Factors: {', '.join(it_analysis.global_factors)}")
    print(f"Top Picks:")
    for pick in it_analysis.top_picks[:3]:
        print(f"  - {pick.symbol}: {pick.signal} ({pick.confidence:.0%})")


if __name__ == "__main__":
    demo()
