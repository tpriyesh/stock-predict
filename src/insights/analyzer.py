"""
Comprehensive Stock Analyzer - Combines all data sources for final prediction.
This is the main entry point for "Tomorrow's Picks" analysis.
"""
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.data.price_fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.fundamentals import FundamentalsFetcher
from src.data.firecrawl_fetcher import FirecrawlFetcher
from src.features.technical import TechnicalIndicators
from src.features.news_features import NewsFeatureExtractor
from src.features.market_features import MarketFeatures
from src.models.patterns import PatternMatcher, IndianMarketPatterns
from src.insights.ai_engine import AIInsightEngine, QuickInsight


class ComprehensiveAnalyzer:
    """
    Comprehensive stock analyzer that combines:
    - 1 Year Price Data + Technical Indicators
    - 7 Days News + Sentiment Analysis
    - Fundamental Data (PE, ROE, etc.)
    - Historical Pattern Matching
    - AI-Powered Insights

    Output: Actionable "Tomorrow's Trade" recommendations
    """

    def __init__(self, use_ai: bool = True, ai_provider: str = "openai", use_firecrawl: bool = True):
        """
        Initialize analyzer.

        Args:
            use_ai: Whether to use AI for insights (requires API key or Ollama)
            ai_provider: "openai" or "ollama"
            use_firecrawl: Whether to use Firecrawl for news scraping
        """
        self.price_fetcher = PriceFetcher()
        self.news_fetcher = NewsFetcher()
        self.fundamentals_fetcher = FundamentalsFetcher()
        self.news_extractor = NewsFeatureExtractor()
        self.market_features = MarketFeatures()
        self.pattern_matcher = PatternMatcher()
        self.settings = get_settings()

        # Firecrawl for scraping Indian news sites
        self.use_firecrawl = use_firecrawl
        if use_firecrawl:
            self.firecrawl = FirecrawlFetcher()
        else:
            self.firecrawl = None

        self.use_ai = use_ai
        if use_ai:
            self.ai_engine = AIInsightEngine(provider=ai_provider)
        else:
            self.ai_engine = None

    def analyze_stock(self, symbol: str, use_ai_insight: bool = None) -> dict:
        """
        Perform comprehensive analysis on a single stock.

        Args:
            symbol: NSE stock symbol
            use_ai_insight: Override default AI setting

        Returns:
            Complete analysis dict with all data and recommendations
        """
        logger.info(f"Starting comprehensive analysis for {symbol}")
        use_ai = use_ai_insight if use_ai_insight is not None else self.use_ai

        result = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'status': 'success'
        }

        try:
            # 1. Fetch 1 year price data
            logger.info(f"{symbol}: Fetching price data...")
            prices = self.price_fetcher.fetch_prices(symbol, period='1y')
            if not prices:
                return {'symbol': symbol, 'status': 'error', 'error': 'No price data'}

            df = pd.DataFrame([p.model_dump() for p in prices])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # 2. Calculate technical indicators
            logger.info(f"{symbol}: Calculating technical indicators...")
            df = TechnicalIndicators.calculate_all(df)

            # Extract key metrics
            latest = df.iloc[-1]
            week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
            month_ago = df.iloc[-22] if len(df) >= 22 else df.iloc[0]
            six_month_ago = df.iloc[-126] if len(df) >= 126 else df.iloc[0]
            year_ago = df.iloc[0]

            technical_data = {
                'close': float(latest['close']),
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'volume': int(latest['volume']),
                'week_change': round((latest['close'] / week_ago['close'] - 1) * 100, 2),
                'month_change': round((latest['close'] / month_ago['close'] - 1) * 100, 2),
                'six_month_change': round((latest['close'] / six_month_ago['close'] - 1) * 100, 2),
                'year_change': round((latest['close'] / year_ago['close'] - 1) * 100, 2),
                'rsi': round(float(latest.get('rsi', 50)), 1),
                'rsi_signal': 'OVERSOLD' if latest.get('rsi', 50) < 30 else 'OVERBOUGHT' if latest.get('rsi', 50) > 70 else 'NEUTRAL',
                'macd': round(float(latest.get('macd_histogram', 0)), 4),
                'macd_signal': 'BULLISH' if latest.get('macd_histogram', 0) > 0 else 'BEARISH',
                'bb_position': round(float(latest.get('bb_position', 0.5)), 2),
                'bb_signal': 'OVERSOLD' if latest.get('bb_position', 0.5) < 0.2 else 'OVERBOUGHT' if latest.get('bb_position', 0.5) > 0.8 else 'NEUTRAL',
                'volume_ratio': round(float(latest.get('volume_ratio', 1)), 2),
                'atr': round(float(latest.get('atr', 0)), 2),
                'atr_pct': round(float(latest.get('atr_pct', 2)), 2),
                'support': round(float(latest.get('support', 0)), 2),
                'resistance': round(float(latest.get('resistance', 0)), 2),
                'sma_20': round(float(latest.get('sma_20', 0)), 2),
                'sma_50': round(float(latest.get('sma_50', 0)), 2),
                'sma_200': round(float(latest.get('sma_200', 0)), 2),
            }

            # Determine trend
            if latest['close'] > latest.get('sma_20', 0) > latest.get('sma_50', 0):
                technical_data['trend'] = 'UPTREND'
            elif latest['close'] < latest.get('sma_20', float('inf')) < latest.get('sma_50', float('inf')):
                technical_data['trend'] = 'DOWNTREND'
            else:
                technical_data['trend'] = 'SIDEWAYS'

            result['technical'] = technical_data

            # 3. Fetch fundamentals
            logger.info(f"{symbol}: Fetching fundamentals...")
            fundamentals = self.fundamentals_fetcher.get_fundamentals(symbol)
            result['fundamentals'] = fundamentals

            # 4. Fetch news (7 days) - Try Firecrawl first, then fallback to NewsAPI
            logger.info(f"{symbol}: Fetching news (7 days)...")
            news_articles = []

            # Try Firecrawl for Indian financial sites
            if self.use_firecrawl and self.firecrawl:
                try:
                    firecrawl_articles = self.firecrawl.scrape_stock_news(symbol)
                    news_articles.extend(firecrawl_articles)
                    logger.info(f"{symbol}: Got {len(firecrawl_articles)} articles from Firecrawl")
                except Exception as e:
                    logger.warning(f"Firecrawl failed for {symbol}: {e}")

            # Fallback to NewsAPI/GNews
            if not news_articles:
                try:
                    api_articles = self.news_fetcher.fetch_for_symbol(symbol, hours=168)
                    if api_articles:
                        news_articles.extend(api_articles)
                except Exception as e:
                    logger.warning(f"NewsAPI failed for {symbol}: {e}")

            if news_articles:
                # Extract features using OpenAI (if not already extracted by Firecrawl)
                # Note: Firecrawl already extracts sentiment/tickers, so we mainly need this for NewsAPI articles
                articles_needing_extraction = [a for a in news_articles if not a.key_claims]
                if articles_needing_extraction and self.news_extractor.api_key:
                    try:
                        extracted = self.news_extractor.extract_batch(articles_needing_extraction)
                        # Replace articles with extracted versions (frozen models can't be modified)
                        extracted_map = {e.url: e for e in extracted}
                        news_articles = [extracted_map.get(a.url, a) for a in news_articles]
                    except Exception as e:
                        logger.warning(f"OpenAI extraction failed: {e}")

                news_summary = self.news_extractor.get_symbol_news_summary(news_articles, symbol)
                news_text = self.news_extractor.generate_news_reasoning(news_articles, symbol)
            else:
                news_summary = {'article_count': 0, 'avg_sentiment': 0}
                news_text = "No recent news found."

            result['news'] = {
                'article_count': len(news_articles),
                'sentiment': news_summary.get('sentiment_trend', 'NEUTRAL'),
                'avg_sentiment_score': news_summary.get('avg_sentiment', 0),
                'summary': news_text,
                'recent_headlines': [a.title for a in news_articles[:5]] if news_articles else [],
                'sources': list(set(a.source for a in news_articles)) if news_articles else []
            }

            # 5. Historical pattern matching
            logger.info(f"{symbol}: Finding historical patterns...")
            similar_patterns = self.pattern_matcher.find_similar_patterns(df)
            pattern_stats = self.pattern_matcher.get_pattern_statistics(similar_patterns)
            result['patterns'] = {
                'count': pattern_stats.get('count', 0),
                'win_rate': pattern_stats.get('win_rate', 0),
                'avg_return': pattern_stats.get('avg_return', 0),
                'best_return': pattern_stats.get('best_return', 0),
                'worst_return': pattern_stats.get('worst_return', 0),
                'confidence': pattern_stats.get('confidence', 0)
            }

            # 6. Indian market specific patterns
            india_patterns = IndianMarketPatterns.detect_all_patterns(df)
            result['india_patterns'] = india_patterns

            # 7. Market context
            logger.info(f"{symbol}: Getting market context...")
            market_context = self.market_features.get_market_context()
            sector = fundamentals.get('sector', 'Unknown')
            sector_strength = self.market_features.get_sector_strength(sector)

            market_context['sector_strength'] = sector_strength.get('strength', 'NEUTRAL')
            result['market_context'] = {
                'regime': market_context.get('regime', {}).get('overall', 'NEUTRAL'),
                'nifty_trend': market_context.get('regime', {}).get('nifty_trend', 'UNKNOWN'),
                'vix': market_context.get('regime', {}).get('vix_level'),
                'sector': sector,
                'sector_strength': sector_strength
            }

            # 8. Quick rule-based insight (always generated)
            quick_insight = QuickInsight.generate(
                symbol, technical_data, fundamentals, pattern_stats
            )
            result['quick_insight'] = quick_insight

            # 9. AI-powered deep insight (if enabled)
            if use_ai and self.ai_engine:
                logger.info(f"{symbol}: Generating AI insight...")
                ai_insight = self.ai_engine.generate_insight(
                    symbol=symbol,
                    technical_data=technical_data,
                    fundamental_data=fundamentals,
                    pattern_stats=pattern_stats,
                    news_summary=news_text,
                    market_context=market_context
                )
                result['ai_insight'] = ai_insight
            else:
                result['ai_insight'] = None

            # 10. Final recommendation (combining all signals)
            result['recommendation'] = self._generate_final_recommendation(result)

            logger.info(f"{symbol}: Analysis complete - {result['recommendation']['signal']}")

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def _generate_final_recommendation(self, analysis: dict) -> dict:
        """
        Generate final recommendation combining all signals.
        """
        quick = analysis.get('quick_insight', {})
        technical = analysis.get('technical', {})
        patterns = analysis.get('patterns', {})
        news = analysis.get('news', {})
        fundamentals = analysis.get('fundamentals', {})

        # Start with quick insight
        signal = quick.get('signal', 'HOLD')
        confidence = quick.get('confidence', 0.5)
        entry = quick.get('entry', 0)
        stop = quick.get('stop_loss', 0)
        target = quick.get('target', 0)

        # Adjust based on pattern history
        if patterns.get('count', 0) >= 5:
            if patterns.get('win_rate', 0) > 0.7:
                confidence = min(0.9, confidence + 0.1)
            elif patterns.get('win_rate', 0) < 0.4:
                confidence = max(0.3, confidence - 0.1)

        # Adjust for news sentiment
        news_sentiment = news.get('avg_sentiment_score', 0)
        if news_sentiment > 0.3 and 'BUY' in signal:
            confidence = min(0.9, confidence + 0.05)
        elif news_sentiment < -0.3 and 'SELL' in signal:
            confidence = min(0.9, confidence + 0.05)
        elif news_sentiment < -0.3 and 'BUY' in signal:
            confidence = max(0.4, confidence - 0.1)

        # Calculate expected scenarios
        current = technical.get('close', 0)
        atr_pct = technical.get('atr_pct', 2)

        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'entry_price': round(entry, 2),
            'stop_loss': round(stop, 2),
            'target_price': round(target, 2),
            'expected_return': round((target - entry) / entry * 100, 2) if entry > 0 else 0,
            'max_risk': round((entry - stop) / entry * 100, 2) if entry > 0 else 0,
            'risk_reward': round((target - entry) / (entry - stop), 2) if (entry - stop) != 0 else 0,
            'scenarios': {
                'best_case': round(patterns.get('best_return', atr_pct * 2), 2),
                'base_case': round(patterns.get('avg_return', atr_pct), 2),
                'worst_case': round(patterns.get('worst_return', -atr_pct * 1.5), 2)
            },
            'holding_period': '1-5 days',
            'reasons': quick.get('reasons', []),
            'risks': quick.get('risks', [])
        }

    def get_tomorrow_picks(
        self,
        symbols: list[str] = None,
        top_n: int = 10,
        min_confidence: float = 0.6
    ) -> dict:
        """
        Get tomorrow's top trading picks.

        Args:
            symbols: List of symbols to analyze (None = NIFTY 50)
            top_n: Number of top picks to return
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with buy picks, sell picks, and summary
        """
        if symbols is None:
            # Load NIFTY 50
            import json
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent / "config" / "symbols.json"
            with open(config_path) as f:
                symbols = json.load(f).get('nifty50', [])

        logger.info(f"Analyzing {len(symbols)} stocks for tomorrow's picks...")

        all_analyses = []
        for symbol in symbols:
            try:
                analysis = self.analyze_stock(symbol, use_ai_insight=False)
                if analysis.get('status') == 'success':
                    all_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")

        # Sort by confidence
        buy_picks = [
            a for a in all_analyses
            if 'BUY' in a.get('recommendation', {}).get('signal', '')
            and a.get('recommendation', {}).get('confidence', 0) >= min_confidence
        ]
        buy_picks.sort(key=lambda x: x['recommendation']['confidence'], reverse=True)

        sell_picks = [
            a for a in all_analyses
            if 'SELL' in a.get('recommendation', {}).get('signal', '')
            and a.get('recommendation', {}).get('confidence', 0) >= min_confidence
        ]
        sell_picks.sort(key=lambda x: x['recommendation']['confidence'], reverse=True)

        # Format output
        return {
            'generated_at': datetime.now().isoformat(),
            'analyzed_count': len(all_analyses),
            'buy_picks': self._format_picks(buy_picks[:top_n]),
            'sell_picks': self._format_picks(sell_picks[:top_n // 2]),
            'summary': self._generate_summary(buy_picks[:top_n], sell_picks[:top_n // 2])
        }

    def _format_picks(self, picks: list) -> list:
        """Format picks for output."""
        formatted = []
        for p in picks:
            rec = p.get('recommendation', {})
            tech = p.get('technical', {})
            formatted.append({
                'symbol': p['symbol'],
                'signal': rec.get('signal'),
                'confidence': rec.get('confidence'),
                'current_price': tech.get('close'),
                'entry': rec.get('entry_price'),
                'stop_loss': rec.get('stop_loss'),
                'target': rec.get('target_price'),
                'expected_return': rec.get('expected_return'),
                'max_risk': rec.get('max_risk'),
                'risk_reward': rec.get('risk_reward'),
                'reasons': rec.get('reasons', [])[:3]
            })
        return formatted

    def _generate_summary(self, buys: list, sells: list) -> str:
        """Generate text summary."""
        summary = f"""
# Tomorrow's Trading Picks - {datetime.now().strftime('%Y-%m-%d')}

## Top Buy Signals:
"""
        for i, p in enumerate(buys[:5], 1):
            rec = p.get('recommendation', {})
            summary += f"""
{i}. **{p['symbol']}** - {rec.get('signal')} ({rec.get('confidence', 0):.0%} confidence)
   Entry: ₹{rec.get('entry_price')} | Stop: ₹{rec.get('stop_loss')} | Target: ₹{rec.get('target_price')}
   Expected: +{rec.get('expected_return', 0):.1f}% | Risk: -{rec.get('max_risk', 0):.1f}%
   Why: {'; '.join(rec.get('reasons', [])[:2])}
"""

        if sells:
            summary += "\n## Top Sell Signals:\n"
            for i, p in enumerate(sells[:3], 1):
                rec = p.get('recommendation', {})
                summary += f"""
{i}. **{p['symbol']}** - {rec.get('signal')} ({rec.get('confidence', 0):.0%} confidence)
"""

        summary += "\n\n⚠️ **Disclaimer**: This is algorithmic analysis. Always do your own research."

        return summary
