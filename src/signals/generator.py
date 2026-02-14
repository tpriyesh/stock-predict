"""
Main signal generator - orchestrates the entire prediction pipeline.
"""
import json
from datetime import datetime, date
from utils.platform import now_ist, today_ist
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.data.price_fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.market_indicators import MarketIndicators
from src.features.technical import TechnicalIndicators
from src.features.news_features import NewsFeatureExtractor
from src.features.market_features import MarketFeatures
from src.models.scoring import ScoringEngine, StockScore
from src.models.intraday_model import IntradayModel
from src.models.swing_model import SwingModel
from src.signals.explainer import SignalExplainer
from src.storage.database import Database
from src.storage.models import TradeType, SignalType, Prediction


class SignalGenerator:
    """
    Main orchestrator for generating stock predictions.

    Pipeline:
    1. Fetch price data for universe
    2. Calculate technical indicators
    3. Fetch and analyze news
    4. Score each stock
    5. Rank and filter
    6. Generate explanations
    7. Output predictions
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize signal generator with all components."""
        settings = get_settings()

        self.settings = settings
        self.price_fetcher = PriceFetcher()
        self.news_fetcher = NewsFetcher()
        self.news_extractor = NewsFeatureExtractor()
        self.market_features = MarketFeatures()
        self.intraday_model = IntradayModel()
        self.swing_model = SwingModel()
        self.explainer = SignalExplainer()
        self.scoring_engine = ScoringEngine()

        # Database
        if db_path is None:
            db_path = settings.db_path
        self.db = Database(db_path)

        # Daily price cache — daily candles don't change intraday
        self._daily_price_cache: dict[str, pd.DataFrame] = {}
        self._daily_price_cache_date: Optional[date] = None

        # Load universe
        self._load_universe()

    def _load_universe(self):
        """Load stock universe from config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "symbols.json"
        try:
            with open(config_path) as f:
                data = json.load(f)
            # Use NIFTY 50 as primary universe
            self.universe = data.get('nifty50', [])
            self.sector_mapping = {}
            for sector, symbols in data.get('sectors', {}).items():
                for symbol in symbols:
                    self.sector_mapping[symbol] = sector
            logger.info(f"Loaded universe of {len(self.universe)} stocks")
        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
            self.universe = []
            self.sector_mapping = {}

    def run(
        self,
        symbols: list[str] = None,
        include_intraday: bool = True,
        include_swing: bool = True
    ) -> dict:
        """
        Run the full prediction pipeline.

        Args:
            symbols: List of symbols to analyze (None = full universe)
            include_intraday: Generate intraday signals
            include_swing: Generate swing signals

        Returns:
            Dict with predictions and metadata
        """
        start_time = now_ist().replace(tzinfo=None)
        symbols = symbols or self.universe

        logger.info(f"Starting prediction run for {len(symbols)} symbols")

        # Step 1: Get market context
        logger.info("Fetching market context...")
        market_context = self.market_features.get_market_context()

        # Step 2: Fetch all price data
        logger.info("Fetching price data...")
        price_data = self._fetch_all_prices(symbols)

        # Step 3: Calculate technical indicators
        logger.info("Calculating technical indicators...")
        for symbol, df in price_data.items():
            try:
                price_data[symbol] = TechnicalIndicators.calculate_all(df)
            except Exception as e:
                logger.warning(f"Failed to calculate indicators for {symbol}: {e}")

        # Step 4: Fetch and analyze news
        logger.info("Fetching and analyzing news...")
        news_data = self._fetch_and_analyze_news(symbols)

        # Step 5: Score all stocks
        logger.info("Scoring stocks...")
        intraday_scores = []
        swing_scores = []

        for symbol in symbols:
            if symbol not in price_data or price_data[symbol].empty:
                continue

            df = price_data[symbol]
            # Convert sentiment (-1 to +1) to score (0 to 1)
            raw_sentiment = news_data.get(symbol, {}).get('avg_sentiment', 0)
            # avg_sentiment is in range [-1, 1], convert to [0, 1]
            news_score = (raw_sentiment + 1) / 2
            news_score = max(0, min(1, news_score))
            news_reasons = []

            news_summary = news_data.get(symbol, {})
            if news_summary.get('article_count', 0) > 0:
                sentiment = news_summary.get('sentiment_trend', 'NEUTRAL')
                news_reasons.append(f"[NEWS] {news_summary.get('article_count')} articles, {sentiment} sentiment")
                if news_summary.get('recent_claims'):
                    news_reasons.append(f"[NEWS] {news_summary['recent_claims'][0][:100]}")
            else:
                news_reasons.append("[RISK] No news data available - trading on technicals only")

            try:
                if include_intraday:
                    score = self.intraday_model.score(
                        symbol=symbol,
                        df=df,
                        news_score=news_score,
                        news_reasons=news_reasons.copy(),
                        market_context=market_context
                    )
                    if score:
                        intraday_scores.append(score)

                if include_swing:
                    score = self.swing_model.score(
                        symbol=symbol,
                        df=df,
                        news_score=news_score,
                        news_reasons=news_reasons.copy(),
                        market_context=market_context
                    )
                    if score:
                        swing_scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to score {symbol}: {e}")

        # Step 6: Rank and filter
        logger.info("Ranking predictions...")
        top_intraday = self.scoring_engine.rank_stocks(
            intraday_scores,
            top_n=self.settings.top_n_picks,
            min_confidence=self.settings.min_confidence
        )

        top_swing = self.scoring_engine.rank_stocks(
            swing_scores,
            top_n=self.settings.top_n_picks,
            min_confidence=self.settings.min_confidence
        )

        # Step 7: Generate explanations
        logger.info("Generating explanations...")
        intraday_predictions = []
        for score in top_intraday:
            articles = [a for a in news_data.get(score.symbol, {}).get('articles', [])]
            explanation = self.explainer.explain(score, articles)
            pred = self._score_to_prediction(score, explanation, articles)
            intraday_predictions.append(pred)

        swing_predictions = []
        for score in top_swing:
            articles = [a for a in news_data.get(score.symbol, {}).get('articles', [])]
            explanation = self.explainer.explain(score, articles)
            pred = self._score_to_prediction(score, explanation, articles)
            swing_predictions.append(pred)

        # Step 8: Save to database
        logger.info("Saving predictions...")
        for pred in intraday_predictions + swing_predictions:
            try:
                self.db.save_prediction(pred)
            except Exception as e:
                logger.warning(f"Failed to save prediction: {e}")

        # Generate summary
        summary = self.explainer.generate_daily_summary(
            top_intraday, top_swing, market_context
        )

        elapsed = (now_ist().replace(tzinfo=None) - start_time).total_seconds()
        logger.info(f"Prediction run complete in {elapsed:.1f}s")

        return {
            'timestamp': now_ist().replace(tzinfo=None).isoformat(),
            'elapsed_seconds': elapsed,
            'market_context': market_context,
            'intraday': {
                'predictions': [self._prediction_to_dict(p) for p in intraday_predictions],
                'count': len(intraday_predictions)
            },
            'swing': {
                'predictions': [self._prediction_to_dict(p) for p in swing_predictions],
                'count': len(swing_predictions)
            },
            'summary': summary
        }

    def _fetch_all_prices(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch price data for all symbols with daily caching.

        Daily candles don't change intraday — once fetched, they're valid
        for the entire trading day. Only fetches symbols not already cached.
        """
        today = today_ist()

        # Invalidate cache on new trading day
        if self._daily_price_cache_date != today:
            self._daily_price_cache.clear()
            self._daily_price_cache_date = today

        # Identify symbols that need fetching
        uncached = [s for s in symbols if s not in self._daily_price_cache]
        cached_count = len(symbols) - len(uncached)

        if not uncached:
            logger.info(f"All {len(symbols)} symbols using daily price cache")
            return {s: self._daily_price_cache[s] for s in symbols
                    if s in self._daily_price_cache}

        if cached_count > 0:
            logger.info(
                f"Fetching price data for {len(uncached)} symbols "
                f"({cached_count} cached)"
            )

        stale_count = 0
        for symbol in uncached:
            try:
                prices = self.price_fetcher.fetch_prices(symbol, period="1y")
                if prices:
                    df = pd.DataFrame([p.model_dump() for p in prices])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')

                    # Data freshness check: reject data older than 3 trading days
                    latest_date = df.index.max().date() if not df.empty else None
                    if latest_date:
                        days_stale = (today - latest_date).days
                        # Allow weekends (2 days) + 1 buffer = 3 days
                        if days_stale > 5:
                            logger.warning(
                                f"{symbol}: Data is {days_stale} days stale "
                                f"(latest={latest_date}), SKIPPING"
                            )
                            stale_count += 1
                            continue

                    # NaN check: ensure latest row has valid OHLC
                    if not df.empty:
                        latest_row = df.iloc[-1]
                        if pd.isna(latest_row.get('close', None)):
                            logger.warning(f"{symbol}: Latest close is NaN, SKIPPING")
                            continue

                    self._daily_price_cache[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        if stale_count > 0:
            logger.warning(f"Skipped {stale_count} symbols due to stale data")
        logger.info(f"Fetched price data for {len(self._daily_price_cache)}/{len(symbols)} symbols")

        return {s: self._daily_price_cache[s] for s in symbols
                if s in self._daily_price_cache}

    def _fetch_and_analyze_news(self, symbols: list[str]) -> dict:
        """Fetch and analyze news with budget-aware fetching.

        Limits per-symbol news API calls to NEWS_BUDGET_PER_CYCLE to stay
        within free tier limits (NewsAPI: 100/day, GNews: 100/day).
        Symbols with cached news data are served from cache.
        """
        from providers.cache import get_response_cache
        cache = get_response_cache()

        news_data = {}

        # Fetch general market news (limit LLM extraction to top 15)
        market_news = self.news_fetcher.fetch_market_news(hours=48)
        if market_news:
            market_news = self.news_extractor.extract_batch(market_news[:15])

        # Budget: limit fresh news API calls per cycle
        try:
            from config.trading_config import CONFIG
            budget = CONFIG.providers.news_budget_per_cycle
        except Exception:
            budget = 15

        # Separate symbols into cached vs needs-fetch.
        # The news cache stores raw Article objects. We convert them to
        # NewsArticle and run LLM extraction (which has its own 6-hour cache).
        uncached_symbols = []
        for symbol in symbols:
            cache_key = f"news:{symbol}:{self.settings.news_lookback_hours}"
            cached = cache.get("news", cache_key)
            if cached is not None and len(cached) > 0:
                # Use cached articles — no news API call needed
                try:
                    from src.data.news_fetcher import _article_to_news_article
                    articles = [_article_to_news_article(a) for a in cached]
                    # LLM extraction has its own 6-hour cache (keyed by URL),
                    # so re-extracting cached articles is essentially free
                    articles = self.news_extractor.extract_batch(articles[:5])
                    summary = self.news_extractor.get_symbol_news_summary(articles, symbol)
                    summary['articles'] = articles
                    news_data[symbol] = summary
                except Exception:
                    uncached_symbols.append(symbol)
            else:
                uncached_symbols.append(symbol)

        # Only fetch for budget-limited subset
        symbols_to_fetch = uncached_symbols[:budget]
        skipped = len(uncached_symbols) - len(symbols_to_fetch)
        if skipped > 0:
            logger.info(
                f"News fetch: {len(symbols_to_fetch)}/{len(symbols)} symbols "
                f"(budget={budget}, {len(symbols) - len(uncached_symbols)} cached, "
                f"{skipped} deferred)"
            )

        for symbol in symbols_to_fetch:
            try:
                articles = self.news_fetcher.fetch_for_symbol(
                    symbol,
                    hours=self.settings.news_lookback_hours
                )

                if articles:
                    articles = self.news_extractor.extract_batch(articles[:5])
                    summary = self.news_extractor.get_symbol_news_summary(articles, symbol)
                    summary['articles'] = articles
                    news_data[symbol] = summary
                else:
                    # Check if mentioned in market news
                    if market_news:
                        relevant = [a for a in market_news if symbol in a.tickers]
                        if relevant:
                            summary = self.news_extractor.get_symbol_news_summary(relevant, symbol)
                            summary['articles'] = relevant
                            news_data[symbol] = summary

            except Exception as e:
                logger.warning(f"News fetch failed for {symbol}: {e}")

        # For skipped symbols, check market news as fallback
        if market_news:
            for symbol in uncached_symbols[budget:]:
                if symbol not in news_data:
                    relevant = [a for a in market_news if symbol in a.tickers]
                    if relevant:
                        summary = self.news_extractor.get_symbol_news_summary(relevant, symbol)
                        summary['articles'] = relevant
                        news_data[symbol] = summary

        logger.info(f"Analyzed news for {len(news_data)} symbols")
        return news_data

    def _score_to_prediction(
        self,
        score: StockScore,
        explanation: str,
        articles: list
    ) -> Prediction:
        """Convert StockScore to Prediction model."""
        from src.storage.models import TechnicalSignals

        tech_signals = TechnicalSignals(
            symbol=score.symbol,
            date=score.date,
            close=score.current_price,
            change_pct=0.0,  # Would need previous close
            atr_pct=score.atr_pct
        )

        return Prediction(
            symbol=score.symbol,
            generated_at=now_ist().replace(tzinfo=None),
            trade_type=score.trade_type,
            signal=score.signal,
            confidence=score.confidence,
            current_price=score.current_price,
            entry_price=score.entry_price,
            stop_loss=score.stop_loss,
            target_price=score.target_price,
            summary=explanation,
            reasons=score.reasons,
            technical_data=tech_signals,
            news_articles=articles[:5] if articles else [],
            news_urls=[a.url for a in articles[:5]] if articles else []
        )

    def _prediction_to_dict(self, pred: Prediction) -> dict:
        """Convert Prediction to serializable dict."""
        return {
            'symbol': pred.symbol,
            'generated_at': pred.generated_at.isoformat(),
            'trade_type': pred.trade_type.value,
            'signal': pred.signal.value,
            'confidence': pred.confidence,
            'current_price': pred.current_price,
            'entry_price': pred.entry_price,
            'stop_loss': pred.stop_loss,
            'target_price': pred.target_price,
            'summary': pred.summary,
            'reasons': pred.reasons,
            'news_urls': pred.news_urls
        }

    def get_single_prediction(
        self,
        symbol: str,
        trade_type: TradeType = TradeType.INTRADAY
    ) -> Optional[dict]:
        """
        Get prediction for a single symbol.

        Useful for quick lookups without running full pipeline.
        """
        try:
            prices = self.price_fetcher.fetch_prices(symbol, period="1y")
            if not prices:
                return None

            df = pd.DataFrame([p.model_dump() for p in prices])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = TechnicalIndicators.calculate_all(df)

            # Get news
            articles = self.news_fetcher.fetch_for_symbol(symbol, hours=72)
            articles = self.news_extractor.extract_batch(articles) if articles else []
            news_summary = self.news_extractor.get_symbol_news_summary(articles, symbol)

            news_score = (news_summary.get('avg_sentiment', 0) + 1) / 2
            news_reasons = []
            if news_summary.get('article_count', 0) > 0:
                news_reasons.append(f"[NEWS] {news_summary['article_count']} articles found")

            # Score
            model = self.intraday_model if trade_type == TradeType.INTRADAY else self.swing_model
            score = model.score(
                symbol=symbol,
                df=df,
                news_score=news_score,
                news_reasons=news_reasons
            )

            if not score:
                return None

            explanation = self.explainer.explain(score, articles)
            pred = self._score_to_prediction(score, explanation, articles)

            return self._prediction_to_dict(pred)

        except Exception as e:
            logger.error(f"Failed to get prediction for {symbol}: {e}")
            return None
