"""
Enhanced Signal Generator

Uses the EnhancedScoringEngine which integrates a 7-model ensemble:
- Base Technical/Momentum/Volume/News scoring
- Physics Engine (momentum, spring, energy, network)
- Math Engine (Fourier, Hurst, entropy, statistical mechanics)
- HMM Regime Detection
- Macro Engine (commodities, currencies, bonds, semiconductors) [NEW]
- Alternative Data Engine (earnings, options, institutional flow, multi-timeframe) [NEW]
- Advanced Math Engine (Kalman, Wavelet, PCA, Markov, Factor, DQN) [NEW]
- Ensemble voting for robust signals

Key improvements:
1. Multi-model agreement required for signals (3/7 for moderate, 5/7 for strong)
2. SELL signals disabled (historically inverted)
3. Regime-aware filtering
4. Calibrated confidence from backtest data
5. Macro indicators for sector-level impact analysis
6. Alternative data for earnings and institutional flow signals
7. Advanced mathematical models for pattern detection
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.data.price_fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.market_indicators import MarketIndicators
from src.features.technical import TechnicalIndicators
from src.features.news_features import NewsFeatureExtractor
from src.features.market_features import MarketFeatures
from src.models.enhanced_scoring import EnhancedScoringEngine, EnhancedStockScore, convert_enhanced_to_base
from src.models.intraday_model import IntradayModel
from src.models.swing_model import SwingModel
from src.signals.explainer import SignalExplainer
from src.storage.database import Database
from src.storage.models import TradeType, SignalType, Prediction


class EnhancedSignalGenerator:
    """
    Enhanced signal generator using multi-model ensemble approach.

    Pipeline:
    1. Fetch price data for universe
    2. Calculate technical indicators
    3. Fetch and analyze news
    4. Score each stock with enhanced scoring (4 model ensemble)
    5. Filter by regime and model agreement
    6. Rank and filter
    7. Generate explanations
    8. Output predictions
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize enhanced signal generator."""
        settings = get_settings()

        self.settings = settings
        self.price_fetcher = PriceFetcher()
        self.news_fetcher = NewsFetcher()
        self.news_extractor = NewsFeatureExtractor()
        self.market_features = MarketFeatures()
        self.intraday_model = IntradayModel()
        self.swing_model = SwingModel()
        self.explainer = SignalExplainer()

        # Use enhanced scoring engine
        self.scoring_engine = EnhancedScoringEngine()

        # Database
        if db_path is None:
            db_path = settings.db_path
        self.db = Database(db_path)

        # Load universe
        self._load_universe()

    def _load_universe(self):
        """Load stock universe from config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "symbols.json"
        try:
            with open(config_path) as f:
                data = json.load(f)
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
        symbols: List[str] = None,
        include_intraday: bool = True,
        include_swing: bool = True,
        min_model_agreement: int = 3  # Updated for 7-model ensemble (was 2 for 4-model)
    ) -> dict:
        """
        Run the enhanced prediction pipeline.

        Args:
            symbols: List of symbols to analyze (None = full universe)
            include_intraday: Generate intraday signals
            include_swing: Generate swing signals
            min_model_agreement: Minimum models that must agree for signal

        Returns:
            Dict with predictions and metadata
        """
        start_time = datetime.now()
        symbols = symbols or self.universe

        logger.info(f"Starting ENHANCED prediction run for {len(symbols)} symbols")

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

        # Step 5: Score all stocks with ENHANCED scoring
        logger.info("Scoring stocks with enhanced multi-model ensemble...")
        intraday_scores = []
        swing_scores = []

        for symbol in symbols:
            if symbol not in price_data or price_data[symbol].empty:
                continue

            df = price_data[symbol]
            news_score = news_data.get(symbol, {}).get('avg_sentiment', 0.5) + 0.5
            news_score = max(0, min(1, news_score))
            news_reasons = []

            news_summary = news_data.get(symbol, {})
            if news_summary.get('article_count', 0) > 0:
                sentiment = news_summary.get('sentiment_trend', 'NEUTRAL')
                news_reasons.append(f"[NEWS] {news_summary.get('article_count')} articles, {sentiment} sentiment")
                if news_summary.get('recent_claims'):
                    news_reasons.append(f"[NEWS] {news_summary['recent_claims'][0][:100]}")

            try:
                if include_intraday:
                    score = self.scoring_engine.score_stock(
                        symbol=symbol,
                        df=df,
                        trade_type=TradeType.INTRADAY,
                        news_score=news_score,
                        news_reasons=news_reasons.copy(),
                        market_context=market_context
                    )
                    if score:
                        intraday_scores.append(score)

                if include_swing:
                    score = self.scoring_engine.score_stock(
                        symbol=symbol,
                        df=df,
                        trade_type=TradeType.SWING,
                        news_score=news_score,
                        news_reasons=news_reasons.copy(),
                        market_context=market_context
                    )
                    if score:
                        swing_scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to score {symbol}: {e}")

        # Step 6: Rank and filter with enhanced criteria
        logger.info("Ranking predictions with multi-model agreement filter...")

        top_intraday = self.scoring_engine.rank_stocks(
            intraday_scores,
            top_n=self.settings.top_n_picks,
            min_confidence=self.settings.min_confidence,
            min_agreement=min_model_agreement
        )

        top_swing = self.scoring_engine.rank_stocks(
            swing_scores,
            top_n=self.settings.top_n_picks,
            min_confidence=self.settings.min_confidence,
            min_agreement=min_model_agreement
        )

        # Step 7: Generate explanations
        logger.info("Generating enhanced explanations...")
        intraday_predictions = []
        for score in top_intraday:
            articles = [a for a in news_data.get(score.symbol, {}).get('articles', [])]
            base_score = convert_enhanced_to_base(score)
            explanation = self._generate_enhanced_explanation(score)
            pred = self._score_to_prediction(score, explanation, articles)
            intraday_predictions.append(pred)

        swing_predictions = []
        for score in top_swing:
            articles = [a for a in news_data.get(score.symbol, {}).get('articles', [])]
            explanation = self._generate_enhanced_explanation(score)
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
        summary = self._generate_enhanced_summary(
            top_intraday, top_swing, market_context
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Enhanced prediction run complete in {elapsed:.1f}s")

        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'market_context': market_context,
            'enhancement_info': {
                'scoring_engine': 'EnhancedScoringEngine',
                'models_used': ['base', 'physics', 'math', 'regime', 'macro', 'alternative', 'advanced'],
                'model_count': 7 if self.scoring_engine.use_7_model else 4,
                'min_agreement_required': min_model_agreement,
                'sell_signals_disabled': True,
                'reason': 'SELL signals have 28.6% accuracy (inverted)',
                'new_features': [
                    'Macro indicators (commodities, currencies, bonds)',
                    'Alternative data (earnings, options, institutional flow)',
                    'Advanced math (Kalman, Wavelet, PCA, Markov)'
                ]
            },
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

    def _fetch_all_prices(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all symbols."""
        price_data = {}

        for symbol in symbols:
            try:
                prices = self.price_fetcher.fetch_prices(symbol, period="1y")
                if prices:
                    df = pd.DataFrame([p.model_dump() for p in prices])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    price_data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Fetched price data for {len(price_data)}/{len(symbols)} symbols")
        return price_data

    def _fetch_and_analyze_news(self, symbols: List[str]) -> Dict:
        """Fetch and analyze news for all symbols."""
        news_data = {}

        market_news = self.news_fetcher.fetch_market_news(hours=48)
        if market_news:
            market_news = self.news_extractor.extract_batch(market_news)

        for symbol in symbols:
            try:
                articles = self.news_fetcher.fetch_for_symbol(
                    symbol,
                    hours=self.settings.news_lookback_hours
                )

                if articles:
                    articles = self.news_extractor.extract_batch(articles)
                    summary = self.news_extractor.get_symbol_news_summary(articles, symbol)
                    summary['articles'] = articles
                    news_data[symbol] = summary
                else:
                    relevant = [a for a in market_news if symbol in a.tickers]
                    if relevant:
                        summary = self.news_extractor.get_symbol_news_summary(relevant, symbol)
                        summary['articles'] = relevant
                        news_data[symbol] = summary

            except Exception as e:
                logger.warning(f"News fetch failed for {symbol}: {e}")

        logger.info(f"Analyzed news for {len(news_data)} symbols")
        return news_data

    def _generate_enhanced_explanation(self, score: EnhancedStockScore) -> str:
        """Generate explanation highlighting multi-model agreement."""
        lines = []

        # Header with confidence and agreement
        total_models = getattr(score, 'total_models', 4)
        lines.append(f"**{score.symbol}** - {score.signal.value.upper()} ({score.signal_strength})")
        lines.append(f"Confidence: {score.confidence:.1%} | Models Agreeing: {score.model_agreement}/{total_models}")
        lines.append("")

        # Model votes
        lines.append("**Model Consensus:**")
        for model, vote in score.model_votes.items():
            emoji = "🟢" if vote == "BUY" else ("🔴" if vote == "SELL" else "⚪")
            lines.append(f"  {emoji} {model.title()}: {vote}")
        lines.append("")

        # Regime info
        lines.append(f"**Market Regime:** {score.regime.replace('_', ' ').title()}")
        lines.append(f"**Predictability:** {score.market_predictability:.0%}")
        lines.append(f"**Strategy:** {score.recommended_strategy.title()}")
        lines.append("")

        # Trading levels
        lines.append("**Trade Setup:**")
        lines.append(f"  Entry: {score.entry_price:.2f}")
        lines.append(f"  Stop Loss: {score.stop_loss:.2f}")
        lines.append(f"  Target: {score.target_price:.2f}")
        lines.append(f"  Risk/Reward: {score.risk_reward:.1f}:1")
        lines.append("")

        # Key reasons
        if score.reasons:
            lines.append("**Key Signals:**")
            for reason in score.reasons[:5]:
                lines.append(f"  - {reason}")
            lines.append("")

        # Warnings
        if score.warnings:
            lines.append("**Warnings:**")
            for warning in score.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)

    def _generate_enhanced_summary(
        self,
        intraday: List[EnhancedStockScore],
        swing: List[EnhancedStockScore],
        market_context: dict
    ) -> str:
        """Generate enhanced daily summary."""
        lines = []
        lines.append("# Enhanced Prediction Summary")
        lines.append("")
        lines.append("## Methodology")
        if self.scoring_engine.use_7_model:
            lines.append("Using 7-model ensemble:")
            lines.append("  - Base (Technical/Momentum/News) + Physics + Math + HMM Regime")
            lines.append("  - Macro (Commodities/Currencies/Bonds) + Alternative Data + Advanced Math")
        else:
            lines.append("Using 4-model ensemble: Technical/Momentum/News + Physics + Math + HMM Regime")
        lines.append("SELL signals disabled (28.6% accuracy = inverted)")
        lines.append("")

        # Market overview
        lines.append("## Market Conditions")
        if market_context:
            nifty_change = market_context.get('nifty_change', 0)
            vix = market_context.get('india_vix', 0)
            lines.append(f"- NIFTY 50: {'+' if nifty_change >= 0 else ''}{nifty_change:.1f}%")
            lines.append(f"- India VIX: {vix:.1f}")
        lines.append("")

        # Intraday picks
        if intraday:
            lines.append(f"## Top Intraday Picks ({len(intraday)} stocks)")
            for i, score in enumerate(intraday[:5], 1):
                total = getattr(score, 'total_models', 4)
                lines.append(f"{i}. **{score.symbol}** - Confidence: {score.confidence:.0%} ({score.model_agreement}/{total} models)")
        else:
            lines.append("## Intraday Picks: None (insufficient model agreement)")
        lines.append("")

        # Swing picks
        if swing:
            lines.append(f"## Top Swing Picks ({len(swing)} stocks)")
            for i, score in enumerate(swing[:5], 1):
                total = getattr(score, 'total_models', 4)
                lines.append(f"{i}. **{score.symbol}** - Confidence: {score.confidence:.0%} ({score.model_agreement}/{total} models)")
        else:
            lines.append("## Swing Picks: None (insufficient model agreement)")

        return "\n".join(lines)

    def _score_to_prediction(
        self,
        score: EnhancedStockScore,
        explanation: str,
        articles: list
    ) -> Prediction:
        """Convert EnhancedStockScore to Prediction model."""
        from src.storage.models import TechnicalSignals

        tech_signals = TechnicalSignals(
            symbol=score.symbol,
            date=score.date,
            close=score.current_price,
            change_pct=0.0,
            atr_pct=score.atr_pct
        )

        return Prediction(
            symbol=score.symbol,
            generated_at=datetime.now(),
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
    ) -> Optional[Dict]:
        """Get enhanced prediction for a single symbol."""
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

            # Enhanced score
            score = self.scoring_engine.score_stock(
                symbol=symbol,
                df=df,
                trade_type=trade_type,
                news_score=news_score,
                news_reasons=news_reasons
            )

            if not score:
                return None

            explanation = self._generate_enhanced_explanation(score)
            pred = self._score_to_prediction(score, explanation, articles)

            result = self._prediction_to_dict(pred)
            # Add enhanced metadata
            result['enhanced_data'] = {
                'model_agreement': score.model_agreement,
                'total_models': getattr(score, 'total_models', 4),
                'model_votes': score.model_votes,
                'signal_strength': score.signal_strength,
                'regime': score.regime,
                'regime_stability': score.regime_stability,
                'market_predictability': score.market_predictability,
                'recommended_strategy': score.recommended_strategy,
                'warnings': score.warnings,
                # New 7-model ensemble data
                'sector': getattr(score, 'sector', 'Unknown'),
                'macro_score': getattr(score, 'macro_score', 0.5),
                'alternative_score': getattr(score, 'alternative_score', 0.5),
                'advanced_score': getattr(score, 'advanced_score', 0.5),
            }

            return result

        except Exception as e:
            logger.error(f"Failed to get enhanced prediction for {symbol}: {e}")
            return None
