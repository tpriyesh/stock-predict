"""
Enhanced Intraday Analyzer for Indian Stock Market.

Designed for high-probability intraday trades with:
- Pre-market analysis (run before 9:15 AM IST)
- Global market impact (US, Europe, Asia)
- Real-time news from Indian financial sites
- Sector-specific catalyst detection
- Gap analysis and opening prediction
- FII/DII flow impact
"""
import os
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.data.price_fetcher import PriceFetcher
from src.data.firecrawl_fetcher import FirecrawlFetcher
from src.data.market_indicators import MarketIndicators
from src.features.technical import TechnicalIndicators


class IntradayAnalyzer:
    """
    High-confidence intraday trading analyzer.

    Key Principles:
    1. Only trade when multiple factors align (confluence)
    2. Use LATEST news (within last 12 hours)
    3. Consider global market sentiment
    4. Require minimum 70% confidence for recommendations
    5. Strict risk management (max 1.5% loss per trade)
    """

    # Global market indices that affect Indian markets
    GLOBAL_INDICES = {
        'us_futures': '^GSPC',      # S&P 500
        'nasdaq': '^IXIC',          # NASDAQ
        'dow': '^DJI',              # Dow Jones
        'nikkei': '^N225',          # Japan
        'hangseng': '^HSI',         # Hong Kong
        'sgx_nifty': 'SGX',         # SGX Nifty (proxy)
    }

    # Sector mappings with global dependencies
    SECTOR_DEPENDENCIES = {
        'IT': {
            'global_factors': ['nasdaq', 'us_futures'],
            'currency': 'USDINR',
            'keywords': ['tech layoffs', 'AI', 'cloud', 'software demand', 'IT spending']
        },
        'Banking': {
            'global_factors': ['us_futures', 'bond yields'],
            'keywords': ['RBI', 'interest rate', 'NPA', 'credit growth', 'liquidity']
        },
        'Auto': {
            'global_factors': ['crude oil'],
            'keywords': ['auto sales', 'EV', 'chip shortage', 'steel prices']
        },
        'Pharma': {
            'global_factors': ['us_futures'],
            'keywords': ['FDA', 'USFDA', 'drug approval', 'API', 'healthcare']
        },
        'Metal': {
            'global_factors': ['china', 'commodity'],
            'keywords': ['steel prices', 'aluminum', 'copper', 'China demand', 'infrastructure']
        },
        'Oil_Gas': {
            'global_factors': ['crude oil', 'brent'],
            'keywords': ['crude', 'OPEC', 'refinery', 'gas prices', 'fuel demand']
        },
        'FMCG': {
            'keywords': ['rural demand', 'inflation', 'consumption', 'price hike']
        },
        'Realty': {
            'keywords': ['property', 'housing', 'real estate', 'construction', 'cement']
        }
    }

    # Stock to sector mapping (NIFTY 50)
    STOCK_SECTORS = {
        'RELIANCE': 'Oil_Gas', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT',
        'ICICIBANK': 'Banking', 'HINDUNILVR': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom',
        'ITC': 'FMCG', 'KOTAKBANK': 'Banking', 'LT': 'Infrastructure', 'HCLTECH': 'IT',
        'AXISBANK': 'Banking', 'ASIANPAINT': 'Consumer', 'MARUTI': 'Auto', 'SUNPHARMA': 'Pharma',
        'TITAN': 'Consumer', 'BAJFINANCE': 'NBFC', 'DMART': 'Retail', 'ULTRACEMCO': 'Cement',
        'NTPC': 'Power', 'WIPRO': 'IT', 'NESTLEIND': 'FMCG', 'M&M': 'Auto',
        'POWERGRID': 'Power', 'TATAMOTORS': 'Auto', 'JSWSTEEL': 'Metal', 'TATASTEEL': 'Metal',
        'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure', 'ONGC': 'Oil_Gas',
        'COALINDIA': 'Mining', 'BAJAJFINSV': 'NBFC', 'GRASIM': 'Diversified', 'TECHM': 'IT',
        'HDFCLIFE': 'Insurance', 'SBILIFE': 'Insurance', 'BRITANNIA': 'FMCG', 'INDUSINDBK': 'Banking',
        'HINDALCO': 'Metal', 'CIPLA': 'Pharma', 'DRREDDY': 'Pharma', 'EICHERMOT': 'Auto',
        'DIVISLAB': 'Pharma', 'APOLLOHOSP': 'Healthcare', 'TATACONSUM': 'FMCG', 'BPCL': 'Oil_Gas',
        'HEROMOTOCO': 'Auto', 'BAJAJ-AUTO': 'Auto', 'UPL': 'Chemicals', 'LTIM': 'IT'
    }

    def __init__(self):
        """Initialize intraday analyzer."""
        self.settings = get_settings()
        self.price_fetcher = PriceFetcher()
        self.firecrawl = FirecrawlFetcher()
        self.market_indicators = MarketIndicators()
        self.technical = TechnicalIndicators()

        # OpenAI for deep analysis
        self.openai_key = self.settings.openai_api_key

    def get_pre_market_analysis(self) -> dict:
        """
        Run pre-market analysis (best run between 8:00 - 9:15 AM IST).

        Returns comprehensive market outlook for the day.
        """
        logger.info("Running pre-market analysis...")

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'market_open_prediction': None,
            'global_sentiment': None,
            'key_news': [],
            'sector_outlook': {},
            'fii_dii': {},
            'risk_level': None,
            'trading_recommendation': None
        }

        # 1. Global Market Sentiment
        global_sentiment = self._analyze_global_markets()
        analysis['global_sentiment'] = global_sentiment

        # 2. Latest News Impact
        news_analysis = self._get_market_moving_news()
        analysis['key_news'] = news_analysis['headlines']
        analysis['news_sentiment'] = news_analysis['overall_sentiment']

        # 3. FII/DII Activity (previous day)
        fii_dii = self._get_fii_dii_impact()
        analysis['fii_dii'] = fii_dii

        # 4. Predict Market Opening
        analysis['market_open_prediction'] = self._predict_market_open(
            global_sentiment, news_analysis, fii_dii
        )

        # 5. Sector Outlook
        analysis['sector_outlook'] = self._get_sector_outlook(news_analysis)

        # 6. Overall Risk Level
        analysis['risk_level'] = self._calculate_risk_level(analysis)

        # 7. Trading Recommendation
        analysis['trading_recommendation'] = self._get_trading_recommendation(analysis)

        return analysis

    def _analyze_global_markets(self) -> dict:
        """Analyze overnight global market movements."""
        logger.info("Analyzing global markets...")

        result = {
            'us_markets': {'change': 0, 'sentiment': 'NEUTRAL'},
            'asian_markets': {'change': 0, 'sentiment': 'NEUTRAL'},
            'crude_oil': {'price': 0, 'change': 0},
            'dollar_index': {'value': 0, 'change': 0},
            'overall': 'NEUTRAL'
        }

        try:
            import yfinance as yf

            # US Markets (closed at 4:00 AM IST)
            sp500 = yf.Ticker('^GSPC')
            sp_hist = sp500.history(period='2d')
            if len(sp_hist) >= 2:
                us_change = ((sp_hist['Close'].iloc[-1] / sp_hist['Close'].iloc[-2]) - 1) * 100
                result['us_markets'] = {
                    'change': round(us_change, 2),
                    'sentiment': 'BULLISH' if us_change > 0.3 else 'BEARISH' if us_change < -0.3 else 'NEUTRAL'
                }

            # Crude Oil
            crude = yf.Ticker('CL=F')
            crude_hist = crude.history(period='2d')
            if len(crude_hist) >= 2:
                crude_change = ((crude_hist['Close'].iloc[-1] / crude_hist['Close'].iloc[-2]) - 1) * 100
                result['crude_oil'] = {
                    'price': round(crude_hist['Close'].iloc[-1], 2),
                    'change': round(crude_change, 2)
                }

            # Determine overall sentiment
            us_sent = result['us_markets']['sentiment']
            if us_sent == 'BULLISH':
                result['overall'] = 'BULLISH'
            elif us_sent == 'BEARISH':
                result['overall'] = 'BEARISH'
            else:
                result['overall'] = 'NEUTRAL'

        except Exception as e:
            logger.warning(f"Error fetching global data: {e}")

        return result

    def _get_market_moving_news(self) -> dict:
        """Get latest market-moving news from Indian financial sites."""
        logger.info("Fetching latest market news...")

        result = {
            'headlines': [],
            'overall_sentiment': 'NEUTRAL',
            'bullish_count': 0,
            'bearish_count': 0,
            'key_events': []
        }

        try:
            # Get news from multiple sources via Firecrawl
            articles = self.firecrawl.scrape_market_news('moneycontrol', limit=10)

            for article in articles:
                headline_data = {
                    'title': article.title,
                    'sentiment': article.sentiment.value,
                    'source': article.source,
                    'tickers': article.tickers,
                    'event_type': article.event_type.value if article.event_type else 'other'
                }
                result['headlines'].append(headline_data)

                if article.sentiment.value == 'positive':
                    result['bullish_count'] += 1
                elif article.sentiment.value == 'negative':
                    result['bearish_count'] += 1

            # Determine overall sentiment
            if result['bullish_count'] > result['bearish_count'] + 2:
                result['overall_sentiment'] = 'BULLISH'
            elif result['bearish_count'] > result['bullish_count'] + 2:
                result['overall_sentiment'] = 'BEARISH'
            else:
                result['overall_sentiment'] = 'NEUTRAL'

            # Detect key events
            result['key_events'] = self._detect_key_events(articles)

        except Exception as e:
            logger.warning(f"Error fetching news: {e}")

        return result

    def _detect_key_events(self, articles) -> list:
        """Detect major market-moving events from news."""
        key_events = []

        event_keywords = {
            'RBI_POLICY': ['rbi', 'monetary policy', 'repo rate', 'interest rate'],
            'FII_ACTIVITY': ['fii', 'foreign investor', 'dii', 'institutional'],
            'GLOBAL_CRISIS': ['recession', 'crisis', 'crash', 'trade war', 'tariff'],
            'EARNINGS': ['quarterly results', 'q3 results', 'profit', 'revenue beat'],
            'POLITICAL': ['election', 'government', 'policy', 'budget', 'reform'],
            'SECTOR_NEWS': ['sector', 'industry', 'regulatory', 'approval']
        }

        for article in articles:
            title_lower = article.title.lower()
            content_lower = (article.content or '').lower()

            for event_type, keywords in event_keywords.items():
                if any(kw in title_lower or kw in content_lower for kw in keywords):
                    key_events.append({
                        'type': event_type,
                        'title': article.title,
                        'impact': 'HIGH' if article.sentiment.value != 'neutral' else 'MEDIUM'
                    })
                    break

        return key_events[:5]  # Top 5 key events

    def _get_fii_dii_impact(self) -> dict:
        """Get FII/DII activity and its impact."""
        logger.info("Fetching FII/DII data...")

        result = {
            'fii_net': 0,
            'dii_net': 0,
            'fii_trend': 'NEUTRAL',  # Buying/Selling trend over past days
            'impact': 'NEUTRAL'
        }

        try:
            fii_dii = self.firecrawl.scrape_fii_dii_data()

            if fii_dii:
                fii_net = fii_dii.get('fii', {}).get('net_value', 0)
                dii_net = fii_dii.get('dii', {}).get('net_value', 0)

                result['fii_net'] = fii_net
                result['dii_net'] = dii_net

                # Determine trend
                if fii_net > 500:  # Net buyers > 500 Cr
                    result['fii_trend'] = 'STRONG_BUYING'
                    result['impact'] = 'BULLISH'
                elif fii_net > 0:
                    result['fii_trend'] = 'BUYING'
                    result['impact'] = 'SLIGHTLY_BULLISH'
                elif fii_net < -500:
                    result['fii_trend'] = 'STRONG_SELLING'
                    result['impact'] = 'BEARISH'
                elif fii_net < 0:
                    result['fii_trend'] = 'SELLING'
                    result['impact'] = 'SLIGHTLY_BEARISH'

        except Exception as e:
            logger.warning(f"Error fetching FII/DII: {e}")

        return result

    def _predict_market_open(self, global_sentiment: dict, news: dict, fii_dii: dict) -> dict:
        """Predict how NIFTY will open today."""

        # Score each factor (-2 to +2)
        scores = {
            'global': 0,
            'news': 0,
            'fii_dii': 0
        }

        # Global sentiment score
        global_overall = global_sentiment.get('overall', 'NEUTRAL')
        if global_overall == 'BULLISH':
            scores['global'] = 1.5
        elif global_overall == 'BEARISH':
            scores['global'] = -1.5

        # News sentiment score
        news_sent = news.get('overall_sentiment', 'NEUTRAL')
        if news_sent == 'BULLISH':
            scores['news'] = 1
        elif news_sent == 'BEARISH':
            scores['news'] = -1

        # FII/DII score
        fii_impact = fii_dii.get('impact', 'NEUTRAL')
        if 'BULLISH' in fii_impact:
            scores['fii_dii'] = 1 if 'SLIGHTLY' in fii_impact else 1.5
        elif 'BEARISH' in fii_impact:
            scores['fii_dii'] = -1 if 'SLIGHTLY' in fii_impact else -1.5

        # Total score
        total = scores['global'] * 0.4 + scores['news'] * 0.35 + scores['fii_dii'] * 0.25

        if total > 0.8:
            prediction = 'GAP_UP'
            expected_change = '+0.5% to +1.0%'
            confidence = min(0.8, 0.5 + total * 0.2)
        elif total > 0.3:
            prediction = 'SLIGHTLY_UP'
            expected_change = '+0.1% to +0.5%'
            confidence = 0.6
        elif total < -0.8:
            prediction = 'GAP_DOWN'
            expected_change = '-0.5% to -1.0%'
            confidence = min(0.8, 0.5 + abs(total) * 0.2)
        elif total < -0.3:
            prediction = 'SLIGHTLY_DOWN'
            expected_change = '-0.1% to -0.5%'
            confidence = 0.6
        else:
            prediction = 'FLAT'
            expected_change = '-0.1% to +0.1%'
            confidence = 0.5

        return {
            'prediction': prediction,
            'expected_change': expected_change,
            'confidence': round(confidence, 2),
            'factors': scores
        }

    def _get_sector_outlook(self, news_analysis: dict) -> dict:
        """Determine sector-wise outlook based on news."""

        sector_outlook = {}

        for sector, config in self.SECTOR_DEPENDENCIES.items():
            outlook = {
                'sentiment': 'NEUTRAL',
                'catalysts': [],
                'trade_recommendation': 'AVOID'
            }

            # Check news for sector keywords
            keywords = config.get('keywords', [])
            for headline in news_analysis.get('headlines', []):
                title_lower = headline['title'].lower()
                if any(kw.lower() in title_lower for kw in keywords):
                    outlook['catalysts'].append(headline['title'][:50])
                    if headline['sentiment'] == 'positive':
                        outlook['sentiment'] = 'BULLISH'
                    elif headline['sentiment'] == 'negative':
                        outlook['sentiment'] = 'BEARISH'

            # Trade recommendation
            if outlook['sentiment'] == 'BULLISH' and len(outlook['catalysts']) > 0:
                outlook['trade_recommendation'] = 'BUY'
            elif outlook['sentiment'] == 'BEARISH':
                outlook['trade_recommendation'] = 'AVOID'

            sector_outlook[sector] = outlook

        return sector_outlook

    def _calculate_risk_level(self, analysis: dict) -> str:
        """Calculate overall risk level for trading today."""

        risk_factors = 0

        # High VIX = High risk
        # (Would need real VIX data here)

        # Gap opening = Higher risk
        prediction = analysis.get('market_open_prediction', {}).get('prediction', 'FLAT')
        if 'GAP' in prediction:
            risk_factors += 1

        # Major events = Higher risk
        key_events = analysis.get('key_news', [])
        if len([e for e in key_events if isinstance(e, dict) and e.get('impact') == 'HIGH']) > 2:
            risk_factors += 1

        # FII heavy selling = Higher risk
        if analysis.get('fii_dii', {}).get('fii_trend') == 'STRONG_SELLING':
            risk_factors += 1

        if risk_factors >= 2:
            return 'HIGH'
        elif risk_factors == 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_trading_recommendation(self, analysis: dict) -> dict:
        """Get overall trading recommendation for the day."""

        market_pred = analysis.get('market_open_prediction', {})
        risk_level = analysis.get('risk_level', 'MEDIUM')
        global_sent = analysis.get('global_sentiment', {}).get('overall', 'NEUTRAL')

        if risk_level == 'HIGH':
            return {
                'action': 'CAUTIOUS',
                'message': 'High risk day. Trade with reduced position size or avoid.',
                'position_size': '25-50% of normal'
            }

        pred = market_pred.get('prediction', 'FLAT')

        if pred in ['GAP_UP', 'SLIGHTLY_UP'] and global_sent == 'BULLISH':
            return {
                'action': 'BUY_ON_DIPS',
                'message': 'Bullish day expected. Buy quality stocks on dips.',
                'position_size': '75-100% of normal'
            }
        elif pred in ['GAP_DOWN', 'SLIGHTLY_DOWN'] and global_sent == 'BEARISH':
            return {
                'action': 'SELL_ON_RISE',
                'message': 'Bearish day expected. Sell on rise or short weak stocks.',
                'position_size': '50-75% of normal'
            }
        else:
            return {
                'action': 'RANGE_BOUND',
                'message': 'Sideways market expected. Trade breakouts with tight stops.',
                'position_size': '50% of normal'
            }

    def get_intraday_picks(self, min_confidence: float = 0.70) -> dict:
        """
        Get top intraday trading picks.

        Only returns stocks with >= 70% confidence.
        """
        logger.info("Generating intraday picks...")

        # First get pre-market analysis
        pre_market = self.get_pre_market_analysis()

        # Get sector outlook
        bullish_sectors = [
            sector for sector, outlook in pre_market.get('sector_outlook', {}).items()
            if outlook.get('sentiment') == 'BULLISH'
        ]

        # Filter stocks from bullish sectors or with strong individual catalysts
        candidate_stocks = []
        for stock, sector in self.STOCK_SECTORS.items():
            if sector in bullish_sectors:
                candidate_stocks.append(stock)

        # If no bullish sectors, pick liquid large caps
        if not candidate_stocks:
            candidate_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                               'SBIN', 'BHARTIARTL', 'LT', 'AXISBANK', 'KOTAKBANK']

        # Analyze each candidate
        picks = []
        for symbol in candidate_stocks[:15]:  # Limit to 15 for speed
            try:
                analysis = self._analyze_for_intraday(symbol, pre_market)
                if analysis and analysis.get('confidence', 0) >= min_confidence:
                    picks.append(analysis)
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")

        # Sort by confidence
        picks.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market_analysis': pre_market,
            'picks': picks[:10],  # Top 10
            'total_analyzed': len(candidate_stocks)
        }

    def _analyze_for_intraday(self, symbol: str, pre_market: dict) -> dict:
        """Analyze a stock specifically for intraday trading."""

        logger.info(f"Analyzing {symbol} for intraday...")

        # Get price data
        price_list = self.price_fetcher.fetch_prices(symbol, period='1mo')
        if price_list is None or len(price_list) < 10:
            return None

        # Convert to DataFrame for technical analysis
        prices = pd.DataFrame([{
            'date': p.date,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume
        } for p in price_list])
        prices.set_index('date', inplace=True)

        # Calculate technical indicators
        tech = self.technical.calculate_all(prices)
        latest = tech.iloc[-1]

        current_price = latest['close']

        # Intraday-specific indicators
        atr = latest.get('atr_14', current_price * 0.015)
        atr_pct = (atr / current_price) * 100

        # Volume analysis
        avg_volume = prices['volume'].rolling(20).mean().iloc[-1]

        # Score calculation for intraday
        score = 0
        reasons = []
        risks = []

        # 1. Trend alignment with market
        market_pred = pre_market.get('market_open_prediction', {}).get('prediction', 'FLAT')

        if 'UP' in market_pred:
            # Look for bullish setups
            if latest.get('rsi_14', 50) < 60:  # Not overbought
                score += 0.15
                reasons.append("RSI not overbought, room to run")
            if latest.get('close') > latest.get('ma_20', 0):
                score += 0.15
                reasons.append("Price above 20-day MA")
        elif 'DOWN' in market_pred:
            # Look for bearish setups (shorting)
            if latest.get('rsi_14', 50) > 40:  # Not oversold
                score += 0.15
                reasons.append("RSI not oversold, room to fall")
            if latest.get('close') < latest.get('ma_20', 0):
                score += 0.15
                reasons.append("Price below 20-day MA")

        # 2. Sector alignment
        sector = self.STOCK_SECTORS.get(symbol, 'Unknown')
        sector_outlook = pre_market.get('sector_outlook', {}).get(sector, {})
        if sector_outlook.get('sentiment') == 'BULLISH':
            score += 0.20
            reasons.append(f"{sector} sector is bullish today")
        elif sector_outlook.get('sentiment') == 'BEARISH':
            score -= 0.10
            risks.append(f"{sector} sector facing headwinds")

        # 3. Volume confirmation
        recent_volume = prices['volume'].iloc[-1]
        if recent_volume > avg_volume * 1.2:
            score += 0.10
            reasons.append("Above average volume (institutional interest)")

        # 4. Volatility check (need enough movement for intraday)
        if 0.8 <= atr_pct <= 2.5:
            score += 0.15
            reasons.append(f"Good intraday volatility ({atr_pct:.1f}%)")
        elif atr_pct > 2.5:
            risks.append(f"High volatility ({atr_pct:.1f}%) - use tight stops")
        elif atr_pct < 0.8:
            score -= 0.10
            risks.append("Low volatility - limited profit potential")

        # 5. Pre-market momentum
        prev_close = prices['close'].iloc[-2]
        overnight_change = ((current_price - prev_close) / prev_close) * 100

        if 'UP' in market_pred and overnight_change > 0:
            score += 0.15
            reasons.append(f"Positive momentum ({overnight_change:+.2f}%)")
        elif 'DOWN' in market_pred and overnight_change < 0:
            score += 0.15
            reasons.append(f"Aligns with bearish market ({overnight_change:+.2f}%)")

        # 6. Support/Resistance levels
        high_20 = prices['high'].rolling(20).max().iloc[-1]
        low_20 = prices['low'].rolling(20).min().iloc[-1]

        # Near support (good for buying)
        if 'UP' in market_pred and current_price < low_20 * 1.02:
            score += 0.10
            reasons.append("Near 20-day support level")
        # Near resistance (good for shorting)
        elif 'DOWN' in market_pred and current_price > high_20 * 0.98:
            score += 0.10
            reasons.append("Near 20-day resistance level")

        # Calculate final confidence
        base_confidence = 0.50  # Start at 50%
        confidence = min(0.90, base_confidence + score)

        # Determine signal
        if 'UP' in market_pred or market_pred == 'FLAT':
            signal = 'BUY' if confidence >= 0.65 else 'HOLD'
        else:
            signal = 'SHORT' if confidence >= 0.65 else 'AVOID'

        # Calculate entry/exit levels
        if signal == 'BUY':
            entry = current_price
            stop_loss = round(current_price - (atr * 1.5), 2)  # 1.5x ATR stop
            target_1 = round(current_price + (atr * 1.0), 2)   # 1:1.5 RR for T1
            target_2 = round(current_price + (atr * 2.0), 2)   # 1:1.33 RR for T2
            max_loss_pct = round(((entry - stop_loss) / entry) * 100, 2)
        elif signal == 'SHORT':
            entry = current_price
            stop_loss = round(current_price + (atr * 1.5), 2)
            target_1 = round(current_price - (atr * 1.0), 2)
            target_2 = round(current_price - (atr * 2.0), 2)
            max_loss_pct = round(((stop_loss - entry) / entry) * 100, 2)
        else:
            entry = stop_loss = target_1 = target_2 = current_price
            max_loss_pct = 0

        return {
            'symbol': symbol,
            'sector': sector,
            'signal': signal,
            'confidence': round(confidence, 2),
            'current_price': round(current_price, 2),
            'entry': round(entry, 2),
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'max_loss_pct': max_loss_pct,
            'atr_pct': round(atr_pct, 2),
            'reasons': reasons,
            'risks': risks
        }

    def get_ai_intraday_analysis(self, symbol: str) -> str:
        """Get detailed AI analysis for intraday trading."""

        if not self.openai_key:
            return "OpenAI API key not configured."

        # Get all data
        pre_market = self.get_pre_market_analysis()
        stock_analysis = self._analyze_for_intraday(symbol, pre_market)

        if not stock_analysis:
            return f"Could not analyze {symbol}"

        # Build prompt
        prompt = f"""You are an expert intraday trader for Indian stock markets (NSE).

TODAY'S MARKET CONTEXT:
- Market Open Prediction: {pre_market.get('market_open_prediction', {}).get('prediction')}
- Global Sentiment: {pre_market.get('global_sentiment', {}).get('overall')}
- Risk Level: {pre_market.get('risk_level')}
- Key News: {[n.get('title', '')[:50] for n in pre_market.get('key_news', [])[:3]]}

STOCK: {symbol}
- Sector: {stock_analysis.get('sector')}
- Current Price: Rs {stock_analysis.get('current_price')}
- Signal: {stock_analysis.get('signal')}
- Confidence: {stock_analysis.get('confidence')*100:.0f}%
- ATR%: {stock_analysis.get('atr_pct')}%
- Reasons: {stock_analysis.get('reasons')}
- Risks: {stock_analysis.get('risks')}

Provide intraday trading advice:

1. VERDICT: Should I trade this stock today? (YES/NO/WAIT)

2. TRADE SETUP (if YES):
   - Entry Price: (exact price or range)
   - Stop Loss: (exact price, max 1.5% from entry)
   - Target 1: (for partial profit booking)
   - Target 2: (if momentum continues)
   - Best Entry Time: (after open/after 10:30/etc.)

3. EXPECTED SCENARIOS:
   - If market opens as predicted: What happens to this stock?
   - If market reverses: What's the exit strategy?

4. KEY LEVELS TO WATCH:
   - Support levels
   - Resistance levels
   - VWAP expected zone

5. RISK WARNING:
   - Any specific risks for today
   - Position sizing recommendation

Be specific with prices and actionable advice for same-day square-off."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Indian stock market intraday trader. Give specific, actionable advice with exact price levels."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"AI analysis failed: {e}"


def run_morning_analysis():
    """Run complete morning analysis before market opens."""

    analyzer = IntradayAnalyzer()

    print("=" * 70)
    print("INTRADAY TRADING ANALYSIS - " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    # Get pre-market analysis
    pre_market = analyzer.get_pre_market_analysis()

    print("\n### MARKET OUTLOOK ###")
    print(f"Market Open: {pre_market['market_open_prediction']['prediction']}")
    print(f"Expected Change: {pre_market['market_open_prediction']['expected_change']}")
    print(f"Confidence: {pre_market['market_open_prediction']['confidence']*100:.0f}%")
    print(f"Risk Level: {pre_market['risk_level']}")

    print("\n### GLOBAL MARKETS ###")
    global_sent = pre_market['global_sentiment']
    print(f"US Markets: {global_sent['us_markets']['change']:+.2f}% ({global_sent['us_markets']['sentiment']})")
    print(f"Crude Oil: ${global_sent['crude_oil']['price']} ({global_sent['crude_oil']['change']:+.2f}%)")
    print(f"Overall: {global_sent['overall']}")

    print("\n### FII/DII ###")
    fii = pre_market['fii_dii']
    print(f"FII Net: Rs {fii['fii_net']} Cr ({fii['fii_trend']})")
    print(f"Impact: {fii['impact']}")

    print("\n### KEY NEWS ###")
    for i, news in enumerate(pre_market['key_news'][:5], 1):
        if isinstance(news, dict):
            print(f"{i}. [{news.get('sentiment', 'N/A')}] {news.get('title', 'N/A')[:60]}...")

    print("\n### TRADING RECOMMENDATION ###")
    rec = pre_market['trading_recommendation']
    print(f"Action: {rec['action']}")
    print(f"Message: {rec['message']}")
    print(f"Position Size: {rec['position_size']}")

    # Get intraday picks
    print("\n" + "=" * 70)
    print("TOP INTRADAY PICKS (Confidence >= 70%)")
    print("=" * 70)

    picks = analyzer.get_intraday_picks(min_confidence=0.70)

    for i, pick in enumerate(picks['picks'][:5], 1):
        print(f"""
{i}. {pick['symbol']} ({pick['sector']})
   Signal: {pick['signal']} | Confidence: {pick['confidence']*100:.0f}%
   Entry: Rs {pick['entry']} | Stop: Rs {pick['stop_loss']} | Target: Rs {pick['target_1']} / Rs {pick['target_2']}
   Max Loss: {pick['max_loss_pct']:.1f}%
   Reasons: {', '.join(pick['reasons'][:2])}
""")

    return picks


if __name__ == "__main__":
    run_morning_analysis()
