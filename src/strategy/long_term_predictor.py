"""
LongTermPredictor - 1-5 Year Investment Analysis

For long-term investing, fundamentals matter MORE than technicals.

Key Principles:
1. Business Quality > Price Action
2. Sustainable Growth > Short-term Momentum
3. Competitive Moat > Current Valuation
4. Management Quality > Technical Patterns

This predictor focuses on:
- Earnings growth sustainability
- Competitive advantages (moat)
- Balance sheet strength
- Management track record
- Industry tailwinds
- Valuation relative to growth
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


class InvestmentHorizon(Enum):
    """Investment time horizons."""
    ONE_YEAR = 1
    THREE_YEARS = 3
    FIVE_YEARS = 5


class CompetitiveMoat(Enum):
    """Types of competitive advantage."""
    WIDE = "wide"      # Durable advantage, hard to replicate
    NARROW = "narrow"  # Some advantage, but can be eroded
    NONE = "none"      # No significant competitive advantage


@dataclass
class LongTermPrediction:
    """Long-term investment prediction."""
    symbol: str
    name: str
    sector: str
    current_price: float

    # Investment rating
    rating: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: float

    # Fundamental scores (0-100)
    overall_score: float
    quality_score: float
    growth_score: float
    value_score: float
    safety_score: float
    momentum_score: float

    # Competitive position
    moat_type: CompetitiveMoat
    moat_sources: List[str]

    # Price targets (for different horizons)
    target_1y: float
    target_3y: float
    target_5y: float

    # Expected returns
    expected_cagr_1y: float  # Compound Annual Growth Rate
    expected_cagr_3y: float
    expected_cagr_5y: float

    # Risk assessment
    downside_risk: float  # Potential loss in bear case
    upside_potential: float  # Potential gain in bull case
    risk_reward_ratio: float

    # Key metrics
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    roe: Optional[float]
    debt_equity: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    dividend_yield: Optional[float]
    free_cash_flow: Optional[float]

    # Multi-bagger potential
    multi_bagger_score: float  # 0-100, chance of 5x+ in 5 years
    multi_bagger_factors: List[str]

    # Analysis
    investment_thesis: str
    key_strengths: List[str]
    key_risks: List[str]
    catalysts: List[str]


class LongTermPredictor:
    """
    Long-term (1-5 year) investment predictor.

    Uses fundamental analysis to identify:
    1. Quality businesses with sustainable advantages
    2. Growth stocks with reasonable valuations
    3. Value opportunities with turnaround potential
    4. Multi-bagger candidates (5-10x potential)

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    DATA COLLECTION                       │
    │   Financials   Ratios   Growth   Industry   Sentiment   │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────┴─────────────────────────────┐
    │                    SCORING ENGINE                        │
    │                                                          │
    │   Quality  │  Growth  │  Value  │  Safety  │  Momentum  │
    │   (30%)    │  (25%)   │  (20%)  │  (15%)   │  (10%)     │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────┴─────────────────────────────┐
    │                    MOAT ANALYSIS                         │
    │                                                          │
    │   Brand Power   Network Effects   Cost Advantages       │
    │   Switching Costs   Intangible Assets   Scale          │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────┴─────────────────────────────┐
    │                    VALUATION                             │
    │                                                          │
    │   DCF   Relative   Sum-of-Parts   Historical Range     │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────┴─────────────────────────────┐
    │                    PREDICTION                            │
    │                                                          │
    │   1-Year Target   3-Year Target   5-Year Target         │
    │   Multi-bagger Score   Risk-Reward   Investment Thesis  │
    └─────────────────────────────────────────────────────────┘
    """

    # Scoring weights for long-term investing
    SCORING_WEIGHTS = {
        'quality': 0.30,   # Business quality is most important
        'growth': 0.25,    # Growth potential
        'value': 0.20,     # Valuation attractiveness
        'safety': 0.15,    # Financial safety
        'momentum': 0.10   # Price momentum (least important for long-term)
    }

    # Moat indicators
    MOAT_INDICATORS = {
        'high_roe_sustained': 'ROE > 20% for 5+ years',
        'pricing_power': 'Consistent margin expansion',
        'market_leader': 'Top 3 market share in industry',
        'network_effects': 'Value increases with users',
        'high_switching_costs': 'Customers locked in',
        'intangible_assets': 'Strong brand, patents, licenses',
        'cost_advantage': 'Lowest cost producer',
        'efficient_scale': 'Natural monopoly characteristics'
    }

    def __init__(self):
        self.settings = get_settings()

    def predict(self, symbol: str,
                horizon: InvestmentHorizon = InvestmentHorizon.THREE_YEARS
                ) -> Optional[LongTermPrediction]:
        """
        Generate long-term prediction for a stock.

        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            horizon: Investment horizon (1, 3, or 5 years)

        Returns:
            LongTermPrediction with comprehensive analysis
        """
        try:
            # Fetch all data
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info or {}
            hist = ticker.history(period='5y')

            if hist.empty or len(hist) < 252:  # Need at least 1 year
                logger.warning(f"Insufficient data for {symbol}")
                return None

            current_price = hist['Close'].iloc[-1]

            # Calculate all scores
            quality_score = self._calculate_quality_score(info, hist)
            growth_score = self._calculate_growth_score(info)
            value_score = self._calculate_value_score(info, hist)
            safety_score = self._calculate_safety_score(info)
            momentum_score = self._calculate_momentum_score(hist)

            # Overall score
            overall_score = (
                quality_score * self.SCORING_WEIGHTS['quality'] +
                growth_score * self.SCORING_WEIGHTS['growth'] +
                value_score * self.SCORING_WEIGHTS['value'] +
                safety_score * self.SCORING_WEIGHTS['safety'] +
                momentum_score * self.SCORING_WEIGHTS['momentum']
            )

            # Moat analysis
            moat_type, moat_sources = self._analyze_moat(info, hist)

            # Calculate price targets
            targets = self._calculate_price_targets(
                info, hist, current_price, growth_score, quality_score
            )

            # Risk assessment
            risk_metrics = self._assess_risk(info, hist, current_price, targets)

            # Multi-bagger analysis
            mb_score, mb_factors = self._analyze_multibagger_potential(
                info, growth_score, quality_score, value_score
            )

            # Generate rating
            rating, confidence = self._generate_rating(
                overall_score, risk_metrics['risk_reward']
            )

            # Generate thesis
            thesis = self._generate_investment_thesis(
                symbol, info, rating, overall_score, moat_type
            )

            # Collect factors
            strengths = self._identify_strengths(info, quality_score, growth_score)
            risks = self._identify_risks(info, safety_score)
            catalysts = self._identify_catalysts(info)

            return LongTermPrediction(
                symbol=symbol,
                name=info.get('shortName', symbol),
                sector=info.get('sector', 'Unknown'),
                current_price=round(current_price, 2),
                rating=rating,
                confidence=round(confidence, 3),
                overall_score=round(overall_score, 1),
                quality_score=round(quality_score, 1),
                growth_score=round(growth_score, 1),
                value_score=round(value_score, 1),
                safety_score=round(safety_score, 1),
                momentum_score=round(momentum_score, 1),
                moat_type=moat_type,
                moat_sources=moat_sources,
                target_1y=targets['1y'],
                target_3y=targets['3y'],
                target_5y=targets['5y'],
                expected_cagr_1y=targets['cagr_1y'],
                expected_cagr_3y=targets['cagr_3y'],
                expected_cagr_5y=targets['cagr_5y'],
                downside_risk=risk_metrics['downside'],
                upside_potential=risk_metrics['upside'],
                risk_reward_ratio=risk_metrics['risk_reward'],
                pe_ratio=info.get('trailingPE'),
                peg_ratio=info.get('pegRatio'),
                roe=info.get('returnOnEquity'),
                debt_equity=info.get('debtToEquity'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                dividend_yield=info.get('dividendYield'),
                free_cash_flow=info.get('freeCashflow'),
                multi_bagger_score=round(mb_score, 1),
                multi_bagger_factors=mb_factors,
                investment_thesis=thesis,
                key_strengths=strengths,
                key_risks=risks,
                catalysts=catalysts
            )

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def _calculate_quality_score(self, info: dict, hist: pd.DataFrame) -> float:
        """Calculate business quality score (0-100)."""
        score = 50  # Start neutral

        # ROE (Return on Equity) - Most important quality metric
        roe = info.get('returnOnEquity')
        if roe:
            if roe > 0.25:
                score += 20  # Excellent
            elif roe > 0.20:
                score += 15  # Very Good
            elif roe > 0.15:
                score += 10  # Good
            elif roe > 0.10:
                score += 5   # Average
            elif roe < 0.05:
                score -= 10  # Poor

        # ROA (Return on Assets)
        roa = info.get('returnOnAssets')
        if roa:
            if roa > 0.15:
                score += 10
            elif roa > 0.10:
                score += 5
            elif roa < 0.03:
                score -= 5

        # Profit Margin
        margin = info.get('profitMargins')
        if margin:
            if margin > 0.20:
                score += 10
            elif margin > 0.15:
                score += 5
            elif margin < 0.05:
                score -= 10

        # Free Cash Flow (positive is good)
        fcf = info.get('freeCashflow')
        if fcf:
            if fcf > 0:
                score += 5
            else:
                score -= 10

        # Operating margin consistency (from historical data)
        # Higher is better for quality

        return max(0, min(100, score))

    def _calculate_growth_score(self, info: dict) -> float:
        """Calculate growth potential score (0-100)."""
        score = 50

        # Revenue Growth
        rev_growth = info.get('revenueGrowth')
        if rev_growth:
            if rev_growth > 0.30:
                score += 20  # Hypergrowth
            elif rev_growth > 0.20:
                score += 15  # High growth
            elif rev_growth > 0.10:
                score += 10  # Moderate growth
            elif rev_growth > 0:
                score += 5   # Some growth
            else:
                score -= 15  # Declining

        # Earnings Growth
        earn_growth = info.get('earningsGrowth')
        if earn_growth:
            if earn_growth > 0.25:
                score += 15
            elif earn_growth > 0.15:
                score += 10
            elif earn_growth > 0:
                score += 5
            else:
                score -= 10

        # Forward vs Trailing PE (growth expectation)
        fwd_pe = info.get('forwardPE')
        trail_pe = info.get('trailingPE')
        if fwd_pe and trail_pe and trail_pe > 0:
            pe_change = (fwd_pe / trail_pe - 1)
            if pe_change < -0.20:  # Forward PE much lower
                score += 10  # Strong earnings growth expected
            elif pe_change > 0.20:
                score -= 5   # Earnings decline expected

        return max(0, min(100, score))

    def _calculate_value_score(self, info: dict, hist: pd.DataFrame) -> float:
        """Calculate value attractiveness score (0-100)."""
        score = 50

        # PE Ratio
        pe = info.get('trailingPE')
        if pe:
            if pe < 10:
                score += 15  # Very cheap
            elif pe < 15:
                score += 10  # Cheap
            elif pe < 20:
                score += 5   # Fair
            elif pe > 40:
                score -= 15  # Expensive
            elif pe > 30:
                score -= 10  # Rich

        # PEG Ratio (PE relative to growth)
        peg = info.get('pegRatio')
        if peg:
            if peg < 0.5:
                score += 15  # Very attractive for growth
            elif peg < 1.0:
                score += 10  # Attractive
            elif peg < 1.5:
                score += 5   # Fair
            elif peg > 2.5:
                score -= 10  # Expensive for growth

        # Price to Book
        pb = info.get('priceToBook')
        if pb:
            if pb < 1:
                score += 10  # Below book value
            elif pb < 2:
                score += 5
            elif pb > 5:
                score -= 5

        # Distance from 52-week high
        current = info.get('currentPrice') or info.get('regularMarketPrice')
        high52 = info.get('fiftyTwoWeekHigh')
        if current and high52:
            pct_from_high = (current / high52 - 1) * 100
            if pct_from_high < -40:
                score += 10  # Major discount
            elif pct_from_high < -20:
                score += 5   # Good discount
            elif pct_from_high > -5:
                score -= 5   # Near highs

        return max(0, min(100, score))

    def _calculate_safety_score(self, info: dict) -> float:
        """Calculate financial safety score (0-100)."""
        score = 50

        # Debt to Equity
        de = info.get('debtToEquity')
        if de is not None:
            if de < 30:
                score += 15  # Very low debt
            elif de < 50:
                score += 10  # Low debt
            elif de < 100:
                score += 5   # Moderate debt
            elif de > 200:
                score -= 20  # High debt
            elif de > 150:
                score -= 10

        # Current Ratio
        cr = info.get('currentRatio')
        if cr:
            if cr > 2:
                score += 10  # Very liquid
            elif cr > 1.5:
                score += 5   # Liquid
            elif cr < 1:
                score -= 15  # Liquidity risk

        # Interest Coverage (implied from margins)
        margin = info.get('operatingMargins')
        if margin:
            if margin > 0.20:
                score += 5
            elif margin < 0.05:
                score -= 5

        # Cash position
        cash = info.get('totalCash')
        debt = info.get('totalDebt')
        if cash and debt:
            if cash > debt:
                score += 10  # Net cash
            elif cash > debt * 0.5:
                score += 5   # Good cash buffer

        return max(0, min(100, score))

    def _calculate_momentum_score(self, hist: pd.DataFrame) -> float:
        """Calculate price momentum score (0-100)."""
        score = 50

        try:
            close = hist['Close']
            current = close.iloc[-1]

            # 3-month momentum
            if len(hist) >= 63:
                mom_3m = (current / close.iloc[-63] - 1) * 100
                if mom_3m > 20:
                    score += 15
                elif mom_3m > 10:
                    score += 10
                elif mom_3m > 0:
                    score += 5
                elif mom_3m < -20:
                    score -= 15
                elif mom_3m < -10:
                    score -= 10

            # 1-year momentum
            if len(hist) >= 252:
                mom_1y = (current / close.iloc[-252] - 1) * 100
                if mom_1y > 30:
                    score += 10
                elif mom_1y > 15:
                    score += 5
                elif mom_1y < -30:
                    score -= 10
                elif mom_1y < -15:
                    score -= 5

            # Above 200 DMA
            if len(hist) >= 200:
                ma200 = close.rolling(200).mean().iloc[-1]
                if current > ma200:
                    score += 5
                else:
                    score -= 5

        except:
            pass

        return max(0, min(100, score))

    def _analyze_moat(self, info: dict,
                      hist: pd.DataFrame) -> Tuple[CompetitiveMoat, List[str]]:
        """Analyze competitive moat."""
        moat_signals = []
        moat_strength = 0

        # High sustained ROE indicates moat
        roe = info.get('returnOnEquity')
        if roe and roe > 0.20:
            moat_signals.append("High ROE (>20%) indicates pricing power")
            moat_strength += 2
        elif roe and roe > 0.15:
            moat_signals.append("Good ROE (>15%)")
            moat_strength += 1

        # High margins indicate pricing power
        margin = info.get('profitMargins')
        if margin and margin > 0.20:
            moat_signals.append("High profit margins (>20%)")
            moat_strength += 2
        elif margin and margin > 0.15:
            moat_signals.append("Good profit margins")
            moat_strength += 1

        # Low debt with high returns = financial moat
        de = info.get('debtToEquity')
        if de is not None and de < 50 and roe and roe > 0.15:
            moat_signals.append("Strong balance sheet with high returns")
            moat_strength += 1

        # Market cap indicates scale
        mc = info.get('marketCap', 0) / 1e7  # Crores
        if mc > 100000:  # > 1 Lakh Cr
            moat_signals.append("Large scale provides cost advantages")
            moat_strength += 1

        # Determine moat type
        if moat_strength >= 4:
            moat_type = CompetitiveMoat.WIDE
        elif moat_strength >= 2:
            moat_type = CompetitiveMoat.NARROW
        else:
            moat_type = CompetitiveMoat.NONE

        return moat_type, moat_signals

    def _calculate_price_targets(self, info: dict, hist: pd.DataFrame,
                                 current_price: float, growth_score: float,
                                 quality_score: float) -> Dict:
        """Calculate price targets for different horizons."""
        # Base growth rate estimation
        rev_growth = info.get('revenueGrowth', 0.10) or 0.10
        earn_growth = info.get('earningsGrowth', 0.10) or 0.10

        # Adjust growth expectations based on scores
        quality_mult = 0.8 + (quality_score / 100) * 0.4  # 0.8 to 1.2
        growth_mult = 0.8 + (growth_score / 100) * 0.4

        # Expected annual return
        base_return = (rev_growth + earn_growth) / 2
        adjusted_return = base_return * quality_mult * growth_mult

        # Cap at reasonable levels
        adjusted_return = max(-0.10, min(0.35, adjusted_return))

        # Calculate targets
        target_1y = current_price * (1 + adjusted_return)
        target_3y = current_price * ((1 + adjusted_return) ** 3)
        target_5y = current_price * ((1 + adjusted_return) ** 5)

        # CAGR calculations
        cagr_1y = adjusted_return * 100
        cagr_3y = ((target_3y / current_price) ** (1/3) - 1) * 100
        cagr_5y = ((target_5y / current_price) ** (1/5) - 1) * 100

        return {
            '1y': round(target_1y, 2),
            '3y': round(target_3y, 2),
            '5y': round(target_5y, 2),
            'cagr_1y': round(cagr_1y, 1),
            'cagr_3y': round(cagr_3y, 1),
            'cagr_5y': round(cagr_5y, 1)
        }

    def _assess_risk(self, info: dict, hist: pd.DataFrame,
                     current_price: float, targets: Dict) -> Dict:
        """Assess investment risk."""
        # Calculate volatility
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)

        # Downside risk (based on historical drawdown and volatility)
        max_dd = (hist['Close'] / hist['Close'].cummax() - 1).min()
        downside = abs(max_dd) * 100

        # Upside potential (based on target)
        upside = (targets['3y'] / current_price - 1) * 100

        # Risk-reward ratio
        risk_reward = upside / downside if downside > 0 else 0

        return {
            'downside': round(downside, 1),
            'upside': round(upside, 1),
            'risk_reward': round(risk_reward, 2),
            'volatility': round(volatility * 100, 1)
        }

    def _analyze_multibagger_potential(self, info: dict, growth_score: float,
                                       quality_score: float,
                                       value_score: float) -> Tuple[float, List[str]]:
        """Analyze potential to become a multi-bagger (5x+ in 5 years)."""
        score = 0
        factors = []

        # Market cap (smaller = more potential)
        mc = info.get('marketCap', 0) / 1e7  # Crores
        if mc < 2000:
            score += 25
            factors.append(f"Small cap (₹{mc:.0f} Cr) with room to grow")
        elif mc < 10000:
            score += 15
            factors.append("Mid cap with growth potential")
        elif mc < 50000:
            score += 5

        # Growth trajectory
        if growth_score > 70:
            score += 25
            factors.append("High growth trajectory")
        elif growth_score > 50:
            score += 15

        # Quality (sustainable growth needs quality)
        if quality_score > 70:
            score += 20
            factors.append("High quality business")
        elif quality_score > 50:
            score += 10

        # Value (growth at reasonable price)
        if value_score > 60:
            score += 15
            factors.append("Reasonable valuation for growth")
        elif value_score > 40:
            score += 5

        # Low debt (safety for growth)
        de = info.get('debtToEquity')
        if de is not None and de < 50:
            score += 10
            factors.append("Low debt provides flexibility")

        # High ROE (compounding machine)
        roe = info.get('returnOnEquity')
        if roe and roe > 0.20:
            score += 15
            factors.append(f"High ROE ({roe*100:.0f}%) for compounding")

        # Revenue growth
        rev_growth = info.get('revenueGrowth')
        if rev_growth and rev_growth > 0.25:
            score += 10
            factors.append(f"Strong revenue growth ({rev_growth*100:.0f}%)")

        return min(100, score), factors

    def _generate_rating(self, overall_score: float,
                         risk_reward: float) -> Tuple[str, float]:
        """Generate investment rating."""
        # Adjust score for risk-reward
        adjusted_score = overall_score * 0.8 + (min(risk_reward, 3) / 3 * 20)

        if adjusted_score >= 75:
            rating = "STRONG_BUY"
            confidence = 0.7 + (adjusted_score - 75) / 100
        elif adjusted_score >= 60:
            rating = "BUY"
            confidence = 0.6 + (adjusted_score - 60) / 100
        elif adjusted_score >= 45:
            rating = "HOLD"
            confidence = 0.5
        elif adjusted_score >= 30:
            rating = "SELL"
            confidence = 0.5 + (45 - adjusted_score) / 100
        else:
            rating = "STRONG_SELL"
            confidence = 0.6

        return rating, min(0.95, confidence)

    def _generate_investment_thesis(self, symbol: str, info: dict,
                                     rating: str, score: float,
                                     moat: CompetitiveMoat) -> str:
        """Generate investment thesis."""
        name = info.get('shortName', symbol)
        sector = info.get('sector', 'Unknown')

        if rating in ['STRONG_BUY', 'BUY']:
            thesis = f"{name} is a quality {sector} company "
            if moat == CompetitiveMoat.WIDE:
                thesis += "with a wide competitive moat. "
            elif moat == CompetitiveMoat.NARROW:
                thesis += "with a narrow competitive moat. "
            else:
                thesis += "in a growing market. "

            roe = info.get('returnOnEquity')
            if roe and roe > 0.15:
                thesis += f"Strong ROE of {roe*100:.0f}% indicates capital efficiency. "

            growth = info.get('revenueGrowth')
            if growth and growth > 0.10:
                thesis += f"Revenue growing at {growth*100:.0f}% provides visibility. "

            thesis += f"Overall score of {score:.0f}/100 supports a {rating} rating."

        elif rating == 'HOLD':
            thesis = f"{name} is fairly valued. Wait for better entry or "
            thesis += "more clarity on growth prospects."

        else:
            thesis = f"{name} faces challenges. "
            de = info.get('debtToEquity')
            if de and de > 100:
                thesis += f"High debt ({de:.0f}%) is concerning. "
            growth = info.get('earningsGrowth')
            if growth and growth < 0:
                thesis += "Declining earnings indicate fundamental issues. "
            thesis += f"Overall score of {score:.0f}/100 suggests caution."

        return thesis

    def _identify_strengths(self, info: dict, quality_score: float,
                            growth_score: float) -> List[str]:
        """Identify key strengths."""
        strengths = []

        if quality_score > 70:
            strengths.append("High quality business with strong fundamentals")

        roe = info.get('returnOnEquity')
        if roe and roe > 0.20:
            strengths.append(f"Excellent ROE of {roe*100:.0f}%")

        growth = info.get('revenueGrowth')
        if growth and growth > 0.15:
            strengths.append(f"Strong revenue growth ({growth*100:.0f}%)")

        margin = info.get('profitMargins')
        if margin and margin > 0.15:
            strengths.append(f"Healthy profit margins ({margin*100:.0f}%)")

        de = info.get('debtToEquity')
        if de is not None and de < 50:
            strengths.append("Strong balance sheet with low debt")

        fcf = info.get('freeCashflow')
        if fcf and fcf > 0:
            strengths.append("Positive free cash flow")

        return strengths[:5]

    def _identify_risks(self, info: dict, safety_score: float) -> List[str]:
        """Identify key risks."""
        risks = []

        if safety_score < 40:
            risks.append("Financial safety concerns")

        de = info.get('debtToEquity')
        if de and de > 100:
            risks.append(f"High debt-to-equity ratio ({de:.0f}%)")

        growth = info.get('earningsGrowth')
        if growth and growth < 0:
            risks.append("Declining earnings")

        pe = info.get('trailingPE')
        if pe and pe > 40:
            risks.append(f"High valuation (P/E: {pe:.0f})")

        margin = info.get('profitMargins')
        if margin and margin < 0.05:
            risks.append("Low profit margins")

        cr = info.get('currentRatio')
        if cr and cr < 1:
            risks.append("Liquidity risk (current ratio < 1)")

        if not risks:
            risks.append("Market risk and economic slowdown")

        return risks[:5]

    def _identify_catalysts(self, info: dict) -> List[str]:
        """Identify potential catalysts."""
        catalysts = []

        # Analyst targets
        target = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('regularMarketPrice')
        if target and current and target > current * 1.2:
            upside = (target / current - 1) * 100
            catalysts.append(f"Analyst target implies {upside:.0f}% upside")

        rec = info.get('recommendationKey')
        if rec and rec in ['buy', 'strong_buy']:
            catalysts.append("Positive analyst recommendations")

        # Growth catalyst
        rev_growth = info.get('revenueGrowth')
        if rev_growth and rev_growth > 0.20:
            catalysts.append("Accelerating revenue growth")

        # General catalysts
        sector = info.get('sector', '')
        if 'technology' in sector.lower():
            catalysts.append("Digital transformation tailwinds")
        elif 'financial' in sector.lower():
            catalysts.append("Credit growth recovery")
        elif 'consumer' in sector.lower():
            catalysts.append("Consumption recovery")
        elif 'energy' in sector.lower():
            catalysts.append("Commodity price movements")

        if not catalysts:
            catalysts.append("Sector rotation opportunity")
            catalysts.append("Valuation re-rating potential")

        return catalysts[:5]

    def find_best_long_term_picks(self, symbols: List[str],
                                  min_score: float = 60,
                                  top_n: int = 10) -> List[LongTermPrediction]:
        """
        Find best long-term investment picks from a list of symbols.

        Args:
            symbols: List of stock symbols to analyze
            min_score: Minimum overall score (0-100)
            top_n: Number of top picks to return

        Returns:
            List of top long-term predictions
        """
        logger.info(f"Analyzing {len(symbols)} stocks for long-term investment...")

        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.predict, sym): sym for sym in symbols
            }

            for future in as_completed(futures):
                try:
                    pred = future.result()
                    if pred and pred.overall_score >= min_score:
                        results.append(pred)
                except Exception as e:
                    pass

        # Sort by overall score
        results.sort(key=lambda x: x.overall_score, reverse=True)

        return results[:top_n]

    def find_multibaggers(self, symbols: List[str],
                          min_mb_score: float = 50) -> List[LongTermPrediction]:
        """
        Find potential multi-bagger stocks.

        Multi-baggers = stocks that can multiply 5x+ in 5 years.

        Args:
            symbols: List of symbols to analyze
            min_mb_score: Minimum multi-bagger score (0-100)

        Returns:
            List of potential multi-baggers
        """
        logger.info(f"Scanning {len(symbols)} stocks for multi-bagger potential...")

        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.predict, sym): sym for sym in symbols
            }

            for future in as_completed(futures):
                try:
                    pred = future.result()
                    if pred and pred.multi_bagger_score >= min_mb_score:
                        results.append(pred)
                except:
                    pass

        # Sort by multi-bagger score
        results.sort(key=lambda x: x.multi_bagger_score, reverse=True)

        return results


def demo():
    """Demonstrate long-term predictor."""
    print("=" * 70)
    print("LONG-TERM PREDICTOR DEMO (1-5 Year Horizon)")
    print("=" * 70)

    predictor = LongTermPredictor()

    # Analyze a quality stock
    print("\n--- Analyzing RELIANCE for long-term ---\n")

    pred = predictor.predict("RELIANCE")

    if pred:
        print(f"Stock: {pred.name} ({pred.symbol})")
        print(f"Sector: {pred.sector}")
        print(f"Price: ₹{pred.current_price}")
        print(f"\nRating: {pred.rating} (Confidence: {pred.confidence:.0%})")

        print(f"\n--- Scores (0-100) ---")
        print(f"Overall:  {pred.overall_score:.0f}")
        print(f"Quality:  {pred.quality_score:.0f}")
        print(f"Growth:   {pred.growth_score:.0f}")
        print(f"Value:    {pred.value_score:.0f}")
        print(f"Safety:   {pred.safety_score:.0f}")
        print(f"Momentum: {pred.momentum_score:.0f}")

        print(f"\n--- Competitive Moat ---")
        print(f"Type: {pred.moat_type.value.upper()}")
        for source in pred.moat_sources:
            print(f"  - {source}")

        print(f"\n--- Price Targets ---")
        print(f"1-Year: ₹{pred.target_1y} (CAGR: {pred.expected_cagr_1y:+.1f}%)")
        print(f"3-Year: ₹{pred.target_3y} (CAGR: {pred.expected_cagr_3y:+.1f}%)")
        print(f"5-Year: ₹{pred.target_5y} (CAGR: {pred.expected_cagr_5y:+.1f}%)")

        print(f"\n--- Risk Assessment ---")
        print(f"Downside Risk: {pred.downside_risk:.1f}%")
        print(f"Upside Potential: {pred.upside_potential:.1f}%")
        print(f"Risk-Reward: {pred.risk_reward_ratio:.2f}x")

        print(f"\n--- Multi-Bagger Score: {pred.multi_bagger_score:.0f}/100 ---")
        for factor in pred.multi_bagger_factors:
            print(f"  + {factor}")

        print(f"\n--- Investment Thesis ---")
        print(pred.investment_thesis)

        print(f"\n--- Key Strengths ---")
        for s in pred.key_strengths:
            print(f"  + {s}")

        print(f"\n--- Key Risks ---")
        for r in pred.key_risks:
            print(f"  - {r}")

    # Find multi-baggers
    print("\n\n--- Scanning for Multi-Baggers ---\n")

    penny_midcap = ['IRCTC', 'TATAPOWER', 'NMDC', 'VEDL', 'PNB',
                    'COALINDIA', 'NHPC', 'SJVN', 'BEL', 'HAL']

    multibaggers = predictor.find_multibaggers(penny_midcap, min_mb_score=40)

    if multibaggers:
        print(f"Found {len(multibaggers)} potential multi-baggers:\n")
        for mb in multibaggers[:5]:
            print(f"  {mb.symbol}: Score {mb.multi_bagger_score:.0f}/100")
            print(f"    Rating: {mb.rating} | Price: ₹{mb.current_price}")
            print(f"    5-Year Target: ₹{mb.target_5y} ({mb.expected_cagr_5y:+.0f}% CAGR)")
            print(f"    Growth Runway: {mb.multi_bagger_factors[0] if mb.multi_bagger_factors else 'N/A'}")
            print()
    else:
        print("No multi-baggers found with score >= 40")


if __name__ == "__main__":
    demo()
