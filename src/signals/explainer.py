"""
Human-readable explanation generator for trading signals.
Uses LLM provider system for rate limiting, quota tracking, and caching.
Falls back to template-based explanations when LLM is unavailable.
"""
from typing import Optional
from datetime import datetime
from loguru import logger

from config.settings import get_settings
from providers.cache import get_response_cache
from providers.quota import get_quota_manager
from src.models.scoring import StockScore
from src.storage.models import NewsArticle
from utils.platform import now_ist, today_ist


class SignalExplainer:
    """
    Generates human-readable explanations for trading signals.

    Routes through the LLM provider system for quota/cache/rate-limit
    integration instead of calling OpenAI directly.
    """

    EXPLANATION_PROMPT = """Generate a concise, actionable trading signal explanation.

STOCK: {symbol}
SIGNAL: {signal} with {confidence:.0%} confidence
TRADE TYPE: {trade_type}

CURRENT PRICE: ₹{current_price}
ENTRY: ₹{entry_price}
STOP-LOSS: ₹{stop_loss}
TARGET: ₹{target_price}
RISK-REWARD: {risk_reward:.2f}

SCORE BREAKDOWN:
- Technical: {technical_score:.0%}
- Momentum: {momentum_score:.0%}
- News: {news_score:.0%}
- Sector: {sector_score:.0%}
- Volume: {volume_score:.0%}

KEY REASONS:
{reasons}

NEWS CONTEXT:
{news_context}

Generate a 3-4 sentence explanation that:
1. States the signal clearly (BUY/SELL with entry/stop/target)
2. Explains the PRIMARY reason (strongest factor)
3. Notes any risks or caveats
4. Is written for a trader to act on immediately

Keep it professional and fact-based. No fluff."""

    def __init__(self):
        self._llm_provider = None
        self._cache = get_response_cache()
        self._quota = get_quota_manager()
        self._init_provider()

    def _init_provider(self):
        """Lazy-load the LLM provider from the provider registry."""
        try:
            from providers.llm import get_llm_providers
            providers = get_llm_providers()
            if providers:
                self._llm_provider = providers[0]
                logger.info(f"Explainer using LLM provider: {self._llm_provider.name}")
            else:
                logger.warning("No LLM providers available - using template explanations")
        except Exception as e:
            logger.warning(f"Failed to init LLM provider for explainer: {e}")

    def explain(
        self,
        score: StockScore,
        news_articles: list[NewsArticle] = None
    ) -> str:
        """
        Generate explanation for a stock score.

        Uses cache to avoid redundant OpenAI calls for the same
        stock/signal/confidence combination within the same day.

        Args:
            score: StockScore object
            news_articles: Related news articles

        Returns:
            Human-readable explanation string
        """
        # Format reasons
        reasons_text = "\n".join(f"- {r}" for r in score.reasons[:10])

        # Format news context
        news_context = "No recent news."
        if news_articles:
            news_items = []
            for article in news_articles[:3]:
                news_items.append(
                    f"- [{article.source}] {article.title} ({article.published_at.strftime('%Y-%m-%d %H:%M')})"
                )
            if news_items:
                news_context = "\n".join(news_items)

        if not self._llm_provider:
            return self._template_explanation(score, news_context)

        # Cache key: same stock + signal + confidence = same explanation
        cache_key = (
            f"explain:{score.symbol}:{score.signal.value}:"
            f"{score.confidence:.2f}:{today_ist()}"
        )
        cached = self._cache.get("llm", cache_key)
        if cached is not None:
            return cached

        # Check quota before calling LLM
        allowed, reason = self._quota.can_request(self._llm_provider.name)
        if not allowed:
            logger.debug(f"Explainer LLM quota blocked: {reason}")
            return self._template_explanation(score, news_context)

        prompt = self.EXPLANATION_PROMPT.format(
            symbol=score.symbol,
            signal=score.signal.value,
            confidence=score.confidence,
            trade_type=score.trade_type.value,
            current_price=score.current_price,
            entry_price=score.entry_price,
            stop_loss=score.stop_loss,
            target_price=score.target_price,
            risk_reward=score.risk_reward,
            technical_score=score.technical_score,
            momentum_score=score.momentum_score,
            news_score=score.news_score,
            sector_score=score.sector_score,
            volume_score=score.volume_score,
            reasons=reasons_text,
            news_context=news_context
        )

        messages = [
            {"role": "system", "content": "You are a financial analyst providing actionable trading signals."},
            {"role": "user", "content": prompt}
        ]

        try:
            self._quota.wait_and_record(self._llm_provider.name)

            result_text = self._llm_provider.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )

            if not result_text:
                self._quota.record_failure(self._llm_provider.name)
                return self._template_explanation(score, news_context)

            self._quota.record_success(self._llm_provider.name)

            explanation = result_text.strip()

            # Cache for the rest of the trading day (6 hours)
            self._cache.set("llm", cache_key, explanation, ttl_seconds=21600)

            return explanation

        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            try:
                self._quota.record_failure(self._llm_provider.name)
            except Exception:
                pass
            return self._template_explanation(score, news_context)

    def _template_explanation(self, score: StockScore, news_context: str) -> str:
        """Fallback template-based explanation."""
        direction = "Buy" if score.signal.value == "BUY" else "Sell" if score.signal.value == "SELL" else "Hold"

        # Find strongest factor
        factors = [
            ("Technical", score.technical_score),
            ("Momentum", score.momentum_score),
            ("News", score.news_score),
            ("Sector", score.sector_score),
            ("Volume", score.volume_score)
        ]
        strongest = max(factors, key=lambda x: x[1])

        explanation = f"""**{score.symbol}**: {direction} at ₹{score.entry_price} ({score.confidence:.0%} confidence)

**Trade Plan**: Entry ₹{score.entry_price} | Stop ₹{score.stop_loss} | Target ₹{score.target_price} (RR: {score.risk_reward:.1f})

**Primary Driver**: {strongest[0]} indicators ({strongest[1]:.0%})

**Key Reasons**:
"""
        for reason in score.reasons[:5]:
            explanation += f"• {reason}\n"

        if score.atr_pct > 3:
            explanation += f"\n**Risk Note**: High volatility (ATR {score.atr_pct:.1f}%)"

        return explanation

    def generate_daily_summary(
        self,
        intraday_picks: list[StockScore],
        swing_picks: list[StockScore],
        market_context: dict
    ) -> str:
        """
        Generate a daily summary of all picks.

        Args:
            intraday_picks: Top intraday picks
            swing_picks: Top swing picks
            market_context: Market regime and context

        Returns:
            Formatted daily summary
        """
        summary = f"""
# Stock Predictions - {now_ist().replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')}

## Market Context
- **Regime**: {market_context.get('regime', {}).get('overall', 'N/A')}
- **NIFTY Trend**: {market_context.get('regime', {}).get('nifty_trend', 'N/A')}
- **VIX**: {market_context.get('regime', {}).get('vix_level', 'N/A')}
- **Leading Sectors**: {', '.join(market_context.get('regime', {}).get('sector_leaders', []))}

---

## Top Intraday Picks (Same-Day Trades)

"""
        for i, pick in enumerate(intraday_picks[:10], 1):
            emoji = "🟢" if pick.signal.value == "BUY" else "🔴" if pick.signal.value == "SELL" else "⚪"
            summary += f"""
### {i}. {emoji} {pick.symbol} - {pick.signal.value} ({pick.confidence:.0%})
- **Entry**: ₹{pick.entry_price} | **Stop**: ₹{pick.stop_loss} | **Target**: ₹{pick.target_price}
- **Risk-Reward**: {pick.risk_reward:.1f}
- **Reasons**: {'; '.join(pick.reasons[:3])}

"""

        summary += """
---

## Top Swing Picks (2-10 Day Holds)

"""
        for i, pick in enumerate(swing_picks[:10], 1):
            emoji = "🟢" if pick.signal.value == "BUY" else "🔴" if pick.signal.value == "SELL" else "⚪"
            summary += f"""
### {i}. {emoji} {pick.symbol} - {pick.signal.value} ({pick.confidence:.0%})
- **Entry**: ₹{pick.entry_price} | **Stop**: ₹{pick.stop_loss} | **Target**: ₹{pick.target_price}
- **Risk-Reward**: {pick.risk_reward:.1f}
- **Reasons**: {'; '.join(pick.reasons[:3])}

"""

        summary += """
---

**Disclaimer**: This is algorithmic analysis for educational purposes.
Always do your own research. Past performance doesn't guarantee future results.
"""

        return summary

    def format_verification_links(
        self,
        symbol: str,
        news_articles: list[NewsArticle] = None
    ) -> str:
        """
        Generate verification links for manual checking.

        Returns markdown with links to charts and news.
        """
        links = f"""
**Verify {symbol}**:
- [TradingView Chart](https://www.tradingview.com/chart/?symbol=NSE%3A{symbol})
- [MoneyControl](https://www.moneycontrol.com/india/stockpricequote/{symbol.lower()})
- [Screener.in](https://www.screener.in/company/{symbol}/)
"""

        if news_articles:
            links += "\n**Source News**:\n"
            for article in news_articles[:5]:
                links += f"- [{article.title[:50]}...]({article.url})\n"

        return links
