"""
InsightsEngine - Human-Readable Stock Analysis with LLM Explanations

This engine provides:
1. Plain English explanations of all numbers
2. WHY to buy/sell based on multiple factors
3. Sector and global trend analysis
4. Calendar patterns (day/week/month/year trends)
5. News and event impact assessment
6. AI/Solar/Energy/Tech sector tracking

Philosophy: Every number should have a human-readable explanation.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import yfinance as yf

from config.settings import get_settings


# =============================================================================
# SECTOR DEFINITIONS AND TRENDS
# =============================================================================

class MegaTrend(Enum):
    """Global mega-trends affecting stock performance."""
    AI_COMPUTING = "AI & Computing"
    RENEWABLE_ENERGY = "Renewable Energy"
    ELECTRIC_VEHICLES = "Electric Vehicles"
    DIGITAL_PAYMENTS = "Digital Payments"
    HEALTHCARE_BIOTECH = "Healthcare & Biotech"
    INFRASTRUCTURE = "Infrastructure"
    CONSUMER_DIGITAL = "Consumer Digital"


# Sector to mega-trend mapping
SECTOR_MEGATRENDS = {
    'Information Technology': [MegaTrend.AI_COMPUTING, MegaTrend.DIGITAL_PAYMENTS],
    'Technology': [MegaTrend.AI_COMPUTING],
    'Consumer Electronics': [MegaTrend.AI_COMPUTING, MegaTrend.CONSUMER_DIGITAL],
    'Renewable Energy': [MegaTrend.RENEWABLE_ENERGY],
    'Oil & Gas': [MegaTrend.RENEWABLE_ENERGY],  # Inverse relationship
    'Automobiles': [MegaTrend.ELECTRIC_VEHICLES],
    'Auto Components': [MegaTrend.ELECTRIC_VEHICLES],
    'Banks': [MegaTrend.DIGITAL_PAYMENTS],
    'Financial Services': [MegaTrend.DIGITAL_PAYMENTS],
    'Pharmaceuticals': [MegaTrend.HEALTHCARE_BIOTECH],
    'Healthcare': [MegaTrend.HEALTHCARE_BIOTECH],
    'Construction': [MegaTrend.INFRASTRUCTURE],
    'Infrastructure': [MegaTrend.INFRASTRUCTURE],
    'Power': [MegaTrend.RENEWABLE_ENERGY, MegaTrend.INFRASTRUCTURE],
    'Metals & Mining': [MegaTrend.ELECTRIC_VEHICLES, MegaTrend.INFRASTRUCTURE],
}

# Global trend keywords
TREND_KEYWORDS = {
    MegaTrend.AI_COMPUTING: ['ai', 'artificial intelligence', 'gpu', 'machine learning',
                             'data center', 'cloud', 'nvidia', 'semiconductor', 'chip'],
    MegaTrend.RENEWABLE_ENERGY: ['solar', 'wind', 'green energy', 'renewable', 'clean energy',
                                  'battery', 'hydrogen', 'carbon neutral', 'climate'],
    MegaTrend.ELECTRIC_VEHICLES: ['ev', 'electric vehicle', 'tesla', 'lithium', 'battery',
                                   'charging', 'byd', 'tata motors ev'],
    MegaTrend.DIGITAL_PAYMENTS: ['upi', 'digital payment', 'fintech', 'paytm', 'phonepe',
                                  'mobile banking', 'digital banking'],
}


@dataclass
class CalendarPattern:
    """Historical calendar patterns for a stock."""
    # Day of week patterns
    best_day: str
    worst_day: str
    day_win_rates: Dict[str, float]

    # Monthly patterns
    best_month: str
    worst_month: str
    month_win_rates: Dict[str, float]

    # Quarterly patterns
    quarter_performance: Dict[str, float]

    # Time-of-day patterns (intraday)
    best_entry_hour: int
    best_exit_hour: int


@dataclass
class TrendAlignment:
    """How well a stock aligns with global trends."""
    megatrends: List[MegaTrend]
    trend_score: float  # 0-1, how aligned with positive trends
    trend_momentum: str  # ACCELERATING, STEADY, DECELERATING
    key_catalysts: List[str]
    key_risks: List[str]


@dataclass
class InsightExplanation:
    """Human-readable explanation for any metric."""
    metric_name: str
    value: Any
    interpretation: str  # What does this mean?
    implication: str     # So what? What should I do?
    confidence: str      # HIGH, MEDIUM, LOW
    source: str          # What data backs this?


@dataclass
class TradeInsight:
    """Complete insight for a trade recommendation."""
    symbol: str
    timestamp: datetime

    # Core recommendation
    action: str  # BUY, SELL, HOLD, AVOID
    conviction: str  # HIGH, MEDIUM, LOW
    timeframe: str  # INTRADAY, SWING, POSITIONAL, LONG_TERM

    # Numbers with explanations
    entry_price: float
    stop_loss: float
    target_price: float
    position_size_pct: float

    # Human-readable WHY
    summary: str  # 2-3 sentence summary
    detailed_reasoning: str  # Full explanation

    # Factors
    technical_factors: List[InsightExplanation]
    fundamental_factors: List[InsightExplanation]
    sentiment_factors: List[InsightExplanation]

    # Trend alignment
    trend_alignment: TrendAlignment

    # Calendar patterns
    calendar_insight: str

    # Key questions answered
    why_buy_now: str
    what_could_go_wrong: str
    when_to_exit: str

    # AI-generated narrative
    ai_narrative: str


class InsightsEngine:
    """
    Engine that generates human-readable insights for stock analysis.

    Every number is explained in plain English.
    Every recommendation has a WHY.
    """

    def __init__(self):
        self.settings = get_settings()
        self.openai_key = self.settings.openai_api_key

    def explain_metric(self, metric_name: str, value: Any, context: dict = None) -> InsightExplanation:
        """
        Generate human-readable explanation for any metric.
        """
        context = context or {}

        explanations = {
            'rsi': self._explain_rsi,
            'pe_ratio': self._explain_pe,
            'macd': self._explain_macd,
            'volume_ratio': self._explain_volume,
            'beta': self._explain_beta,
            'debt_equity': self._explain_debt_equity,
            'roe': self._explain_roe,
            'price_change': self._explain_price_change,
            'win_probability': self._explain_win_prob,
        }

        explainer = explanations.get(metric_name.lower())
        if explainer:
            return explainer(value, context)

        # Default explanation
        return InsightExplanation(
            metric_name=metric_name,
            value=value,
            interpretation=f"{metric_name} is {value}",
            implication="No specific trading implication",
            confidence="LOW",
            source="Calculated from market data"
        )

    def _explain_rsi(self, value: float, context: dict) -> InsightExplanation:
        """Explain RSI in plain English."""
        if value > 70:
            interpretation = f"RSI of {value:.0f} indicates the stock is OVERBOUGHT. This means buyers have pushed the price up aggressively and it may be due for a pullback."
            implication = "Consider WAITING for a pullback before buying. If you own it, consider taking partial profits. The stock has risen too fast."
            confidence = "HIGH"
        elif value < 30:
            interpretation = f"RSI of {value:.0f} indicates the stock is OVERSOLD. This means sellers have pushed the price down aggressively and it may be due for a bounce."
            implication = "This could be a buying opportunity. The stock has fallen too fast and may bounce. Look for confirmation before entering."
            confidence = "HIGH"
        elif 40 <= value <= 60:
            interpretation = f"RSI of {value:.0f} is in the neutral zone. The stock is neither overbought nor oversold."
            implication = "No clear signal from RSI. Look at other indicators for direction."
            confidence = "MEDIUM"
        else:
            interpretation = f"RSI of {value:.0f} shows moderate momentum."
            implication = "The trend is present but not extreme. Follow the trend carefully."
            confidence = "MEDIUM"

        return InsightExplanation(
            metric_name="RSI (14-day)",
            value=f"{value:.0f}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Calculated from last 14 trading days"
        )

    def _explain_pe(self, value: float, context: dict) -> InsightExplanation:
        """Explain P/E ratio in plain English."""
        sector_avg = context.get('sector_pe', 20)

        if value <= 0:
            interpretation = "Negative or zero P/E means the company is making losses."
            implication = "Avoid unless you understand why it's losing money and expect a turnaround."
            confidence = "HIGH"
        elif value < 10:
            interpretation = f"P/E of {value:.1f} is very low. Either the company is undervalued OR the market expects earnings to decline."
            implication = "Could be a value opportunity, but verify WHY it's cheap. Check if earnings are expected to fall."
            confidence = "MEDIUM"
        elif value > 50:
            interpretation = f"P/E of {value:.1f} is very high. The market expects very high growth, or the stock is expensive."
            implication = "Only buy if you believe in exceptional future growth. High risk if growth disappoints."
            confidence = "HIGH"
        else:
            comparison = "above" if value > sector_avg else "below"
            interpretation = f"P/E of {value:.1f} is {comparison} the sector average of {sector_avg:.1f}."
            implication = f"Fairly valued relative to peers. {'Premium' if value > sector_avg else 'Discount'} may be justified by growth or quality."
            confidence = "MEDIUM"

        return InsightExplanation(
            metric_name="P/E Ratio",
            value=f"{value:.1f}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Current price / Trailing 12-month earnings"
        )

    def _explain_macd(self, value: float, context: dict) -> InsightExplanation:
        """Explain MACD in plain English."""
        signal = context.get('macd_signal', 0)
        histogram = context.get('macd_histogram', value - signal)

        if histogram > 0 and value > signal:
            interpretation = f"MACD histogram is positive ({histogram:.2f}). Bullish momentum is building."
            implication = "Momentum favors buyers. Good for trend-following entries."
            confidence = "MEDIUM"
        elif histogram < 0 and value < signal:
            interpretation = f"MACD histogram is negative ({histogram:.2f}). Bearish momentum is building."
            implication = "Momentum favors sellers. Be cautious with long positions."
            confidence = "MEDIUM"
        else:
            interpretation = f"MACD is at a crossover point. Trend may be changing."
            implication = "Wait for confirmation before trading."
            confidence = "LOW"

        return InsightExplanation(
            metric_name="MACD",
            value=f"{value:.2f}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="12-day EMA - 26-day EMA"
        )

    def _explain_volume(self, value: float, context: dict) -> InsightExplanation:
        """Explain volume ratio in plain English."""
        if value > 2:
            interpretation = f"Volume is {value:.1f}x the average - VERY HIGH interest in this stock today."
            implication = "Big players are active. If price is up, this confirms the move. If price is down, selling pressure is strong."
            confidence = "HIGH"
        elif value > 1.5:
            interpretation = f"Volume is {value:.1f}x the average - Above normal activity."
            implication = "Increased conviction in the current move. Follow the trend."
            confidence = "MEDIUM"
        elif value < 0.5:
            interpretation = f"Volume is {value:.1f}x the average - Very low activity."
            implication = "Low conviction. Any price move may not be sustainable."
            confidence = "HIGH"
        else:
            interpretation = f"Volume is {value:.1f}x the average - Normal trading activity."
            implication = "No unusual activity. Trade based on other signals."
            confidence = "LOW"

        return InsightExplanation(
            metric_name="Volume Ratio",
            value=f"{value:.1f}x",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Today's volume / 20-day average"
        )

    def _explain_beta(self, value: float, context: dict) -> InsightExplanation:
        """Explain beta in plain English."""
        if value > 1.5:
            interpretation = f"Beta of {value:.2f} means this stock moves {value:.1f}x as much as the market."
            implication = "HIGH RISK, HIGH REWARD. If market rises 1%, this stock may rise {:.1f}%. But losses are also amplified.".format(value)
            confidence = "HIGH"
        elif value < 0.5:
            interpretation = f"Beta of {value:.2f} means this stock is relatively stable compared to the market."
            implication = "DEFENSIVE stock. Good for uncertain times, but won't capture full market upside."
            confidence = "HIGH"
        else:
            interpretation = f"Beta of {value:.2f} means this stock moves roughly in line with the market."
            implication = "Normal market correlation. Stock will generally follow NIFTY."
            confidence = "MEDIUM"

        return InsightExplanation(
            metric_name="Beta",
            value=f"{value:.2f}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Correlation with NIFTY 50 over 1 year"
        )

    def _explain_debt_equity(self, value: float, context: dict) -> InsightExplanation:
        """Explain debt/equity in plain English."""
        if value > 2:
            interpretation = f"Debt/Equity of {value:.2f} - Company has HIGH debt relative to its equity."
            implication = "RISK: High interest payments. Vulnerable if interest rates rise or earnings fall."
            confidence = "HIGH"
        elif value < 0.3:
            interpretation = f"Debt/Equity of {value:.2f} - Company has very LOW debt."
            implication = "Financially conservative. Lower risk but may be missing growth opportunities."
            confidence = "HIGH"
        else:
            interpretation = f"Debt/Equity of {value:.2f} - Moderate debt levels."
            implication = "Balanced capital structure. Normal for most industries."
            confidence = "MEDIUM"

        return InsightExplanation(
            metric_name="Debt/Equity",
            value=f"{value:.2f}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Total Debt / Shareholders' Equity"
        )

    def _explain_roe(self, value: float, context: dict) -> InsightExplanation:
        """Explain ROE in plain English."""
        if value > 20:
            interpretation = f"ROE of {value:.1f}% - Company generates HIGH returns on shareholder money."
            implication = "QUALITY company. Management is efficient at using capital to generate profits."
            confidence = "HIGH"
        elif value < 5:
            interpretation = f"ROE of {value:.1f}% - Company generates LOW returns on shareholder money."
            implication = "Inefficient capital usage. Why is this company not generating better returns?"
            confidence = "HIGH"
        else:
            interpretation = f"ROE of {value:.1f}% - Moderate profitability."
            implication = "Average capital efficiency. Look for improvement trend."
            confidence = "MEDIUM"

        return InsightExplanation(
            metric_name="Return on Equity",
            value=f"{value:.1f}%",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Net Income / Shareholders' Equity"
        )

    def _explain_price_change(self, value: float, context: dict) -> InsightExplanation:
        """Explain price change in plain English."""
        period = context.get('period', 'today')

        if abs(value) > 5:
            direction = "UP" if value > 0 else "DOWN"
            interpretation = f"Price is {direction} {abs(value):.1f}% {period} - SIGNIFICANT move."
            implication = f"Big move. Check what caused this. {'Bullish momentum' if value > 0 else 'Bearish momentum'} is strong."
            confidence = "HIGH"
        elif abs(value) > 2:
            direction = "up" if value > 0 else "down"
            interpretation = f"Price is {direction} {abs(value):.1f}% {period} - moderate move."
            implication = f"{'Positive' if value > 0 else 'Negative'} trend developing. Monitor for continuation."
            confidence = "MEDIUM"
        else:
            interpretation = f"Price is {value:+.1f}% {period} - minimal movement."
            implication = "No clear direction. Wait for a bigger move."
            confidence = "LOW"

        return InsightExplanation(
            metric_name=f"Price Change ({period})",
            value=f"{value:+.1f}%",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Price comparison"
        )

    def _explain_win_prob(self, value: float, context: dict) -> InsightExplanation:
        """Explain win probability in plain English."""
        if value >= 0.65:
            interpretation = f"Win probability of {value:.0%} - This setup has historically worked well."
            implication = "FAVORABLE setup. The odds are in your favor. Consider taking this trade."
            confidence = "HIGH"
        elif value >= 0.55:
            interpretation = f"Win probability of {value:.0%} - Slightly better than coin flip."
            implication = "MODERATE edge. Trade with smaller size. Expected value is positive but not by much."
            confidence = "MEDIUM"
        else:
            interpretation = f"Win probability of {value:.0%} - Odds are not clearly in your favor."
            implication = "AVOID or take very small position. Better opportunities may exist."
            confidence = "HIGH"

        return InsightExplanation(
            metric_name="Win Probability",
            value=f"{value:.0%}",
            interpretation=interpretation,
            implication=implication,
            confidence=confidence,
            source="Calculated from historical patterns and current conditions"
        )

    def analyze_calendar_patterns(self, symbol: str, hist: pd.DataFrame = None) -> CalendarPattern:
        """
        Analyze historical calendar patterns for a stock.
        """
        if hist is None:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period='2y')
            except:
                return self._default_calendar_pattern()

        if hist.empty or len(hist) < 100:
            return self._default_calendar_pattern()

        hist['returns'] = hist['Close'].pct_change()
        hist['day_of_week'] = hist.index.dayofweek
        hist['month'] = hist.index.month
        hist['quarter'] = hist.index.quarter

        # Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_win_rates = {}
        for day in range(5):
            day_returns = hist[hist['day_of_week'] == day]['returns']
            win_rate = (day_returns > 0).mean()
            day_win_rates[day_names[day]] = win_rate

        best_day = max(day_win_rates, key=day_win_rates.get)
        worst_day = min(day_win_rates, key=day_win_rates.get)

        # Month analysis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_win_rates = {}
        for month in range(1, 13):
            month_returns = hist[hist['month'] == month]['returns']
            if len(month_returns) > 5:
                win_rate = (month_returns > 0).mean()
                month_win_rates[month_names[month-1]] = win_rate

        best_month = max(month_win_rates, key=month_win_rates.get) if month_win_rates else 'N/A'
        worst_month = min(month_win_rates, key=month_win_rates.get) if month_win_rates else 'N/A'

        # Quarter analysis
        quarter_perf = {}
        for q in range(1, 5):
            q_returns = hist[hist['quarter'] == q]['returns']
            if len(q_returns) > 10:
                quarter_perf[f'Q{q}'] = q_returns.mean() * 100

        return CalendarPattern(
            best_day=best_day,
            worst_day=worst_day,
            day_win_rates=day_win_rates,
            best_month=best_month,
            worst_month=worst_month,
            month_win_rates=month_win_rates,
            quarter_performance=quarter_perf,
            best_entry_hour=10,  # 10 AM IST typically
            best_exit_hour=15,   # 3 PM IST
        )

    def _default_calendar_pattern(self) -> CalendarPattern:
        """Default calendar pattern when data is insufficient."""
        return CalendarPattern(
            best_day='Wednesday',
            worst_day='Monday',
            day_win_rates={'Monday': 0.48, 'Tuesday': 0.51, 'Wednesday': 0.53,
                          'Thursday': 0.52, 'Friday': 0.50},
            best_month='Nov',
            worst_month='Sep',
            month_win_rates={'Jan': 0.52, 'Feb': 0.50, 'Mar': 0.51, 'Apr': 0.49,
                            'May': 0.48, 'Jun': 0.49, 'Jul': 0.51, 'Aug': 0.50,
                            'Sep': 0.47, 'Oct': 0.52, 'Nov': 0.54, 'Dec': 0.53},
            quarter_performance={'Q1': 0.5, 'Q2': 0.3, 'Q3': 0.2, 'Q4': 0.8},
            best_entry_hour=10,
            best_exit_hour=15,
        )

    def analyze_trend_alignment(self, symbol: str, sector: str) -> TrendAlignment:
        """
        Analyze how well a stock aligns with global mega-trends.
        """
        # Get mega-trends for this sector
        megatrends = SECTOR_MEGATRENDS.get(sector, [])

        # Calculate trend score
        # Higher score = better alignment with positive global trends
        trend_score = 0.5  # Default neutral

        if MegaTrend.AI_COMPUTING in megatrends:
            trend_score += 0.15  # AI is a strong positive trend
        if MegaTrend.RENEWABLE_ENERGY in megatrends:
            trend_score += 0.10  # Green energy is growing
        if MegaTrend.ELECTRIC_VEHICLES in megatrends:
            trend_score += 0.12  # EV adoption accelerating
        if MegaTrend.DIGITAL_PAYMENTS in megatrends:
            trend_score += 0.08  # Digitization continuing

        trend_score = min(1.0, trend_score)

        # Determine momentum
        if trend_score > 0.7:
            momentum = "ACCELERATING"
        elif trend_score > 0.5:
            momentum = "STEADY"
        else:
            momentum = "DECELERATING"

        # Key catalysts and risks based on trends
        catalysts = []
        risks = []

        if MegaTrend.AI_COMPUTING in megatrends:
            catalysts.append("AI/Data center demand growth")
            catalysts.append("Enterprise AI adoption")
        if MegaTrend.RENEWABLE_ENERGY in megatrends:
            catalysts.append("Government green energy push")
            catalysts.append("Falling solar/wind costs")
            risks.append("Policy changes in renewables")
        if MegaTrend.ELECTRIC_VEHICLES in megatrends:
            catalysts.append("EV adoption acceleration")
            risks.append("Raw material (lithium) price volatility")
        if MegaTrend.DIGITAL_PAYMENTS in megatrends:
            catalysts.append("UPI transaction growth")
            catalysts.append("Financial inclusion drive")

        # General risks
        risks.append("Global economic slowdown")
        risks.append("Currency fluctuation (INR)")

        return TrendAlignment(
            megatrends=megatrends,
            trend_score=trend_score,
            trend_momentum=momentum,
            key_catalysts=catalysts[:4],
            key_risks=risks[:3]
        )

    def generate_trade_insight(self, symbol: str, analysis_data: dict) -> TradeInsight:
        """
        Generate comprehensive trade insight with human-readable explanations.
        """
        # Extract data
        tech = analysis_data.get('technical', {})
        fund = analysis_data.get('fundamental', {})
        prob = analysis_data.get('probability', {})
        rec = analysis_data.get('recommendation', {})

        # Get sector and trend alignment
        sector = analysis_data.get('sector', 'Unknown')
        trend = self.analyze_trend_alignment(symbol, sector)

        # Get calendar patterns
        calendar = self.analyze_calendar_patterns(symbol)

        # Generate explanations for key metrics
        tech_factors = []
        if 'rsi_14' in tech:
            tech_factors.append(self.explain_metric('rsi', tech['rsi_14']))
        if 'volume_ratio' in tech:
            tech_factors.append(self.explain_metric('volume_ratio', tech['volume_ratio']))
        if 'macd' in tech:
            tech_factors.append(self.explain_metric('macd', tech['macd'],
                                {'macd_signal': tech.get('macd_signal', 0)}))

        fund_factors = []
        if 'pe_ratio' in fund:
            fund_factors.append(self.explain_metric('pe_ratio', fund['pe_ratio']))
        if 'roe' in fund:
            fund_factors.append(self.explain_metric('roe', fund['roe']))
        if 'debt_equity' in fund:
            fund_factors.append(self.explain_metric('debt_equity', fund['debt_equity']))

        # Generate calendar insight
        today = datetime.now()
        day_name = today.strftime('%A')
        month_name = today.strftime('%b')

        day_rate = calendar.day_win_rates.get(day_name, 0.5)
        month_rate = calendar.month_win_rates.get(month_name, 0.5)

        if day_rate > 0.52 and month_rate > 0.52:
            calendar_insight = f"Good timing! {day_name}s historically show {day_rate:.0%} win rate, and {month_name} is typically a good month ({month_rate:.0%})."
        elif day_rate < 0.48 or month_rate < 0.48:
            calendar_insight = f"Caution: {day_name}s have lower win rates ({day_rate:.0%}). Consider waiting for {calendar.best_day}."
        else:
            calendar_insight = f"Neutral timing. Historical patterns don't show strong bias today."

        # Generate summary and detailed reasoning
        action = rec.get('action', 'HOLD')
        win_prob = prob.get('win_probability', 0.5)

        summary = self._generate_summary(symbol, action, win_prob, trend, tech, fund)
        detailed = self._generate_detailed_reasoning(symbol, action, tech_factors, fund_factors, trend, calendar)

        # Generate key answers
        why_buy = self._generate_why_buy(action, tech_factors, fund_factors, trend)
        what_wrong = self._generate_what_could_go_wrong(trend, tech)
        when_exit = self._generate_when_to_exit(rec, tech)

        # Generate AI narrative if API available
        ai_narrative = self._generate_ai_narrative(symbol, analysis_data, trend, calendar) if self.openai_key else ""

        return TradeInsight(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            conviction=rec.get('conviction', 'MEDIUM'),
            timeframe=rec.get('timeframe', 'SWING'),
            entry_price=rec.get('entry_price', tech.get('price', 0)),
            stop_loss=rec.get('stop_loss', 0),
            target_price=rec.get('target_1', 0),
            position_size_pct=rec.get('position_size_pct', 5),
            summary=summary,
            detailed_reasoning=detailed,
            technical_factors=tech_factors,
            fundamental_factors=fund_factors,
            sentiment_factors=[],
            trend_alignment=trend,
            calendar_insight=calendar_insight,
            why_buy_now=why_buy,
            what_could_go_wrong=what_wrong,
            when_to_exit=when_exit,
            ai_narrative=ai_narrative
        )

    def _generate_summary(self, symbol: str, action: str, win_prob: float,
                          trend: TrendAlignment, tech: dict, fund: dict) -> str:
        """Generate 2-3 sentence summary."""
        price = tech.get('price', 0)
        rsi = tech.get('rsi_14', 50)
        pe = fund.get('pe_ratio', 0)

        if action == 'BUY':
            summary = f"{symbol} shows a BUY setup with {win_prob:.0%} win probability. "
            if rsi < 40:
                summary += f"RSI at {rsi:.0f} suggests oversold conditions with bounce potential. "
            if trend.trend_score > 0.6:
                summary += f"Stock aligns well with positive mega-trends ({trend.megatrends[0].value if trend.megatrends else 'growth'})."
        elif action == 'SELL':
            summary = f"{symbol} shows weakness. "
            if rsi > 70:
                summary += f"Overbought RSI ({rsi:.0f}) suggests near-term pullback risk. "
            summary += "Consider reducing exposure or waiting for better levels."
        else:
            summary = f"{symbol} is in HOLD territory. No clear edge for new positions. "
            summary += "Current price levels don't offer attractive risk/reward."

        return summary

    def _generate_detailed_reasoning(self, symbol: str, action: str,
                                     tech_factors: List[InsightExplanation],
                                     fund_factors: List[InsightExplanation],
                                     trend: TrendAlignment,
                                     calendar: CalendarPattern) -> str:
        """Generate detailed reasoning paragraph."""
        parts = []

        parts.append(f"**Technical Analysis:** ")
        for tf in tech_factors[:2]:
            parts.append(f"{tf.interpretation} {tf.implication}")

        if fund_factors:
            parts.append(f"\n\n**Fundamental View:** ")
            for ff in fund_factors[:2]:
                parts.append(f"{ff.interpretation}")

        parts.append(f"\n\n**Trend Alignment:** ")
        if trend.megatrends:
            trends_str = ", ".join([t.value for t in trend.megatrends[:2]])
            parts.append(f"This stock is exposed to: {trends_str}. ")
            parts.append(f"Trend momentum is {trend.trend_momentum}. ")

        if trend.key_catalysts:
            parts.append(f"Key catalysts: {', '.join(trend.key_catalysts[:2])}.")

        return " ".join(parts)

    def _generate_why_buy(self, action: str, tech_factors: List,
                          fund_factors: List, trend: TrendAlignment) -> str:
        """Generate answer to 'Why buy now?'"""
        if action != 'BUY':
            return "This is not a buy recommendation. Wait for better setup."

        reasons = []
        for tf in tech_factors:
            if tf.confidence == 'HIGH' and 'oversold' in tf.interpretation.lower():
                reasons.append("Technical indicators show oversold bounce setup")

        for ff in fund_factors:
            if ff.confidence == 'HIGH' and ('quality' in ff.interpretation.lower() or 'high' in ff.interpretation.lower()):
                reasons.append("Strong fundamentals support the valuation")

        if trend.trend_score > 0.6:
            reasons.append(f"Well-aligned with growth mega-trends ({trend.trend_momentum})")

        if not reasons:
            reasons.append("Multiple factors align to create positive expected value")

        return "; ".join(reasons)

    def _generate_what_could_go_wrong(self, trend: TrendAlignment, tech: dict) -> str:
        """Generate answer to 'What could go wrong?'"""
        risks = []

        # Add trend risks
        if trend.key_risks:
            risks.extend(trend.key_risks[:2])

        # Technical risks
        rsi = tech.get('rsi_14', 50)
        if rsi > 65:
            risks.append("Stock may be overbought, pullback possible")

        vol = tech.get('volatility_annual', 0)
        if vol > 40:
            risks.append("High volatility increases risk of sharp moves against position")

        if not risks:
            risks = ["Overall market correction", "Sector-specific headwinds", "Earnings disappointment"]

        return "; ".join(risks[:3])

    def _generate_when_to_exit(self, rec: dict, tech: dict) -> str:
        """Generate answer to 'When to exit?'"""
        stop = rec.get('stop_loss', 0)
        target = rec.get('target_1', 0)

        exit_rules = []

        if stop > 0:
            exit_rules.append(f"Exit if price falls below ₹{stop:,.2f} (stop loss)")
        if target > 0:
            exit_rules.append(f"Book profits at ₹{target:,.2f} (target)")

        exit_rules.append("For intraday: Exit by 3:15 PM regardless of profit/loss")
        exit_rules.append("For swing: Review position if RSI crosses 70")

        return "; ".join(exit_rules)

    def _generate_ai_narrative(self, symbol: str, data: dict,
                               trend: TrendAlignment, calendar: CalendarPattern) -> str:
        """Generate AI-powered narrative using LLM."""
        if not self.openai_key:
            return ""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            prompt = f"""You are a friendly stock market expert explaining a trade idea to a retail investor.

Stock: {symbol}
Sector: {data.get('sector', 'Unknown')}
Current Price: ₹{data.get('technical', {}).get('price', 0):,.2f}
Signal: {data.get('recommendation', {}).get('action', 'HOLD')}
Win Probability: {data.get('probability', {}).get('win_probability', 0.5):.0%}

Key Metrics:
- RSI: {data.get('technical', {}).get('rsi_14', 50):.0f}
- P/E Ratio: {data.get('fundamental', {}).get('pe_ratio', 0):.1f}
- ROE: {data.get('fundamental', {}).get('roe', 0):.1f}%
- Debt/Equity: {data.get('fundamental', {}).get('debt_equity', 0):.2f}

Mega-trends: {[t.value for t in trend.megatrends] if trend.megatrends else ['None identified']}
Best trading day: {calendar.best_day}
Today is: {datetime.now().strftime('%A')}

Write a 3-4 sentence narrative explaining:
1. What makes this stock interesting (or not) right now
2. One key reason to consider it
3. One key risk to watch
4. A simple action suggestion

Write in simple, conversational Indian English. Avoid jargon."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI narrative generation failed: {e}")
            return ""


def demo():
    """Demo the InsightsEngine."""
    engine = InsightsEngine()

    # Test metric explanations
    print("=" * 60)
    print("INSIGHTS ENGINE DEMO")
    print("=" * 60)

    print("\n1. RSI Explanation:")
    rsi_insight = engine.explain_metric('rsi', 28)
    print(f"   Value: {rsi_insight.value}")
    print(f"   Interpretation: {rsi_insight.interpretation}")
    print(f"   Implication: {rsi_insight.implication}")

    print("\n2. P/E Explanation:")
    pe_insight = engine.explain_metric('pe_ratio', 45, {'sector_pe': 25})
    print(f"   Value: {pe_insight.value}")
    print(f"   Interpretation: {pe_insight.interpretation}")
    print(f"   Implication: {pe_insight.implication}")

    print("\n3. Calendar Pattern Analysis (TCS):")
    calendar = engine.analyze_calendar_patterns('TCS')
    print(f"   Best Day: {calendar.best_day}")
    print(f"   Best Month: {calendar.best_month}")
    print(f"   Day Win Rates: {calendar.day_win_rates}")

    print("\n4. Trend Alignment (IT Sector):")
    trend = engine.analyze_trend_alignment('TCS', 'Information Technology')
    print(f"   Mega-trends: {[t.value for t in trend.megatrends]}")
    print(f"   Trend Score: {trend.trend_score:.2f}")
    print(f"   Momentum: {trend.trend_momentum}")
    print(f"   Catalysts: {trend.key_catalysts}")


if __name__ == "__main__":
    demo()
