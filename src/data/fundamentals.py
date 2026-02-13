"""
Fundamental data fetcher for Indian stocks.
Includes PE ratio, earnings, financials, and valuation metrics.
"""
from datetime import datetime, date
from typing import Optional
import yfinance as yf
from loguru import logger


class FundamentalsFetcher:
    """
    Fetches fundamental data for stocks.
    Uses Yahoo Finance which provides decent coverage for NSE stocks.
    """

    def __init__(self):
        self.cache = {}

    def get_fundamentals(self, symbol: str) -> dict:
        """
        Get comprehensive fundamental data for a stock.

        Args:
            symbol: NSE stock symbol (e.g., RELIANCE)

        Returns:
            Dict with fundamental metrics
        """
        yahoo_symbol = f"{symbol}.NS"

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            fundamentals = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),

                # Valuation
                'market_cap': info.get('marketCap', 0),
                'market_cap_cr': round(info.get('marketCap', 0) / 1e7, 2),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),

                # Profitability
                'eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),

                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),

                # Dividends
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),

                # Financial Health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'free_cash_flow': info.get('freeCashflow'),

                # Price Metrics
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_day_average': info.get('fiftyDayAverage'),
                'two_hundred_day_average': info.get('twoHundredDayAverage'),

                # Analyst
                'target_mean_price': info.get('targetMeanPrice'),
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'recommendation': info.get('recommendationKey'),
                'num_analysts': info.get('numberOfAnalystOpinions'),

                # Book Value
                'book_value': info.get('bookValue'),

                # Shares
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'held_by_institutions': info.get('heldPercentInstitutions'),
                'held_by_insiders': info.get('heldPercentInsiders'),
            }

            # Calculate additional metrics
            if fundamentals['current_price'] and fundamentals['fifty_two_week_high']:
                fundamentals['pct_from_52w_high'] = round(
                    (fundamentals['current_price'] / fundamentals['fifty_two_week_high'] - 1) * 100, 2
                )
            if fundamentals['current_price'] and fundamentals['fifty_two_week_low']:
                fundamentals['pct_from_52w_low'] = round(
                    (fundamentals['current_price'] / fundamentals['fifty_two_week_low'] - 1) * 100, 2
                )

            # Valuation score (simple)
            fundamentals['valuation_score'] = self._calculate_valuation_score(fundamentals)

            # Quality score
            fundamentals['quality_score'] = self._calculate_quality_score(fundamentals)

            return fundamentals

        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def _calculate_valuation_score(self, f: dict) -> float:
        """
        Calculate valuation score (0-1, higher = more undervalued).
        """
        score = 0.5  # Neutral

        # PE ratio (lower is better, but not negative)
        pe = f.get('pe_ratio')
        if pe and pe > 0:
            if pe < 15:
                score += 0.15
            elif pe < 25:
                score += 0.05
            elif pe > 40:
                score -= 0.15
            elif pe > 30:
                score -= 0.05

        # PB ratio
        pb = f.get('pb_ratio')
        if pb and pb > 0:
            if pb < 2:
                score += 0.1
            elif pb > 5:
                score -= 0.1

        # Distance from 52-week high (buying at discount)
        pct_from_high = f.get('pct_from_52w_high')
        if pct_from_high:
            if pct_from_high < -20:
                score += 0.1  # Good discount
            elif pct_from_high > -5:
                score -= 0.05  # Near highs

        # Analyst target upside
        current = f.get('current_price')
        target = f.get('target_mean_price')
        if current and target:
            upside = (target / current - 1) * 100
            if upside > 20:
                score += 0.1
            elif upside < 0:
                score -= 0.1

        return max(0, min(1, score))

    def _calculate_quality_score(self, f: dict) -> float:
        """
        Calculate quality score (0-1, higher = better quality business).
        """
        score = 0.5

        # ROE (higher is better)
        roe = f.get('roe')
        if roe:
            if roe > 0.20:
                score += 0.15
            elif roe > 0.15:
                score += 0.1
            elif roe < 0.05:
                score -= 0.1

        # Profit margin
        margin = f.get('profit_margin')
        if margin:
            if margin > 0.15:
                score += 0.1
            elif margin < 0.05:
                score -= 0.1

        # Debt to equity (lower is better)
        de = f.get('debt_to_equity')
        if de:
            if de < 50:
                score += 0.1
            elif de > 150:
                score -= 0.15

        # Earnings growth
        growth = f.get('earnings_growth')
        if growth:
            if growth > 0.20:
                score += 0.1
            elif growth < 0:
                score -= 0.1

        return max(0, min(1, score))

    def get_earnings_dates(self, symbol: str) -> dict:
        """Get upcoming and past earnings dates."""
        yahoo_symbol = f"{symbol}.NS"

        try:
            ticker = yf.Ticker(yahoo_symbol)
            calendar = ticker.calendar

            if calendar is not None and not calendar.empty:
                return {
                    'next_earnings': calendar.get('Earnings Date', [None])[0],
                    'earnings_high': calendar.get('Earnings High', None),
                    'earnings_low': calendar.get('Earnings Low', None),
                    'revenue_high': calendar.get('Revenue High', None),
                    'revenue_low': calendar.get('Revenue Low', None),
                }
            return {}

        except Exception as e:
            logger.warning(f"Could not get earnings dates for {symbol}: {e}")
            return {}

    def get_quick_summary(self, symbol: str) -> str:
        """Get a quick text summary of fundamentals."""
        f = self.get_fundamentals(symbol)

        if 'error' in f:
            return f"Could not fetch fundamentals: {f['error']}"

        summary = f"""
**{f['name']}** ({f['sector']})

📊 Valuation:
  PE: {f.get('pe_ratio', 'N/A'):.1f if f.get('pe_ratio') else 'N/A'} | PB: {f.get('pb_ratio', 'N/A'):.1f if f.get('pb_ratio') else 'N/A'}
  Market Cap: ₹{f.get('market_cap_cr', 0):,.0f} Cr

📈 Performance:
  From 52W High: {f.get('pct_from_52w_high', 0):+.1f}%
  From 52W Low: {f.get('pct_from_52w_low', 0):+.1f}%

💰 Profitability:
  ROE: {f.get('roe', 0)*100:.1f}% | Margin: {f.get('profit_margin', 0)*100:.1f}%

🎯 Analyst Target: ₹{f.get('target_mean_price', 'N/A')}
  Recommendation: {f.get('recommendation', 'N/A').upper() if f.get('recommendation') else 'N/A'}

Scores: Valuation {f.get('valuation_score', 0.5):.0%} | Quality {f.get('quality_score', 0.5):.0%}
"""
        return summary
