"""
AI Insight Engine - Deep analysis using OpenAI or local models (Ollama).
Provides actionable insights combining all data sources.
"""
import json
import requests
from typing import Optional
from datetime import datetime
from openai import OpenAI
from loguru import logger

from config.settings import get_settings


class AIInsightEngine:
    """
    AI-powered insight generation for stock analysis.

    Supports:
    - OpenAI (GPT-4o, GPT-4o-mini)
    - Ollama (local models like Llama, Mistral)
    - Groq (fast inference)
    """

    ANALYSIS_PROMPT = """You are an expert Indian stock market analyst. Analyze this stock and provide actionable trading insights.

## Stock: {symbol}

### Price Data (1 Year):
- Current Price: ₹{current_price}
- 1-Week Change: {week_change}%
- 1-Month Change: {month_change}%
- 6-Month Change: {six_month_change}%
- 1-Year Change: {year_change}%
- 52-Week High: ₹{high_52w} ({pct_from_high}% from high)
- 52-Week Low: ₹{low_52w} ({pct_from_low}% from low)

### Technical Indicators:
- RSI (14): {rsi} ({rsi_signal})
- MACD Histogram: {macd} ({macd_signal})
- Bollinger Position: {bb_position} ({bb_signal})
- Trend: {trend}
- Volume: {volume_ratio}x average
- ATR: {atr_pct}% (volatility)
- Support: ₹{support} | Resistance: ₹{resistance}

### Fundamentals:
- PE Ratio: {pe_ratio}
- PB Ratio: {pb_ratio}
- ROE: {roe}%
- Profit Margin: {profit_margin}%
- Debt/Equity: {debt_equity}
- Market Cap: ₹{market_cap} Cr
- Analyst Target: ₹{analyst_target}

### Historical Pattern Match:
{pattern_summary}

### Recent News (Last 7 Days):
{news_summary}

### Market Context:
- NIFTY Trend: {nifty_trend}
- India VIX: {vix}
- Sector: {sector} ({sector_strength})

---

## TASK: Provide a detailed analysis with:

1. **VERDICT**: Clear BUY / SELL / HOLD with confidence (e.g., "BUY with 75% confidence")

2. **TOMORROW'S TRADE PLAN**:
   - Entry Price: ₹___
   - Stop Loss: ₹___ (max loss: __%)
   - Target 1: ₹___ (potential gain: __%)
   - Target 2: ₹___ (if momentum continues)
   - Expected holding period: ___ days

3. **EXPECTED PROFIT/LOSS SCENARIOS**:
   - Best Case: +__% (if ___)
   - Base Case: +__% (most likely)
   - Worst Case: -__% (if ___)

4. **KEY REASONS** (top 3 with evidence):
   - Reason 1: [Technical/Fundamental/News] ...
   - Reason 2: ...
   - Reason 3: ...

5. **RISKS & RED FLAGS**:
   - Risk 1: ...
   - Risk 2: ...

6. **NEWS IMPACT ASSESSMENT**:
   - Is news already priced in?
   - Any upcoming events?

Be specific with numbers. No vague statements. This is for actual trading decisions.
"""

    def __init__(self, provider: str = "openai"):
        """
        Initialize AI engine.

        Args:
            provider: "openai", "ollama", or "groq"
        """
        self.provider = provider
        self.settings = get_settings()

        if provider == "openai":
            if self.settings.openai_api_key:
                self.client = OpenAI(api_key=self.settings.openai_api_key)
                self.model = "gpt-4o-mini"  # Cost-effective
            else:
                logger.warning("OpenAI API key not set, falling back to Ollama")
                self.provider = "ollama"
                self.client = None

        if provider == "ollama" or self.provider == "ollama":
            self.ollama_url = "http://localhost:11434"
            self.model = "llama3.2"  # or mistral, phi3, etc.

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama model."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not running. Start with: ollama serve")
            return "Error: Ollama not available. Please start Ollama or configure OpenAI API key."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {e}"

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Indian stock market analyst with 20 years of experience. Provide specific, actionable trading advice with exact numbers."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error: {e}"

    def generate_insight(
        self,
        symbol: str,
        technical_data: dict,
        fundamental_data: dict,
        pattern_stats: dict,
        news_summary: str,
        market_context: dict
    ) -> str:
        """
        Generate comprehensive AI insight for a stock.

        Args:
            symbol: Stock symbol
            technical_data: Technical indicator values
            fundamental_data: Fundamental metrics
            pattern_stats: Historical pattern statistics
            news_summary: Recent news summary
            market_context: Market regime and context

        Returns:
            Detailed analysis string
        """
        # Format pattern summary
        if pattern_stats.get('count', 0) > 0:
            pattern_summary = f"""
Found {pattern_stats['count']} similar setups in history:
- Win Rate: {pattern_stats['win_rate']:.0%}
- Avg Return (5 days): {pattern_stats['avg_return']:+.2f}%
- Best Case: {pattern_stats['best_return']:+.2f}%
- Worst Case: {pattern_stats['worst_return']:+.2f}%
"""
        else:
            pattern_summary = "No similar historical patterns found."

        prompt = self.ANALYSIS_PROMPT.format(
            symbol=symbol,
            current_price=technical_data.get('close', 0),
            week_change=technical_data.get('week_change', 0),
            month_change=technical_data.get('month_change', 0),
            six_month_change=technical_data.get('six_month_change', 0),
            year_change=technical_data.get('year_change', 0),
            high_52w=fundamental_data.get('fifty_two_week_high', 'N/A'),
            low_52w=fundamental_data.get('fifty_two_week_low', 'N/A'),
            pct_from_high=fundamental_data.get('pct_from_52w_high', 'N/A'),
            pct_from_low=fundamental_data.get('pct_from_52w_low', 'N/A'),
            rsi=technical_data.get('rsi', 'N/A'),
            rsi_signal=technical_data.get('rsi_signal', 'N/A'),
            macd=technical_data.get('macd', 'N/A'),
            macd_signal=technical_data.get('macd_signal', 'N/A'),
            bb_position=technical_data.get('bb_position', 'N/A'),
            bb_signal=technical_data.get('bb_signal', 'N/A'),
            trend=technical_data.get('trend', 'N/A'),
            volume_ratio=technical_data.get('volume_ratio', 'N/A'),
            atr_pct=technical_data.get('atr_pct', 'N/A'),
            support=technical_data.get('support', 'N/A'),
            resistance=technical_data.get('resistance', 'N/A'),
            pe_ratio=fundamental_data.get('pe_ratio', 'N/A'),
            pb_ratio=fundamental_data.get('pb_ratio', 'N/A'),
            roe=round(fundamental_data.get('roe', 0) * 100, 1) if fundamental_data.get('roe') else 'N/A',
            profit_margin=round(fundamental_data.get('profit_margin', 0) * 100, 1) if fundamental_data.get('profit_margin') else 'N/A',
            debt_equity=fundamental_data.get('debt_to_equity', 'N/A'),
            market_cap=fundamental_data.get('market_cap_cr', 'N/A'),
            analyst_target=fundamental_data.get('target_mean_price', 'N/A'),
            pattern_summary=pattern_summary,
            news_summary=news_summary or "No recent news available.",
            nifty_trend=market_context.get('regime', {}).get('nifty_trend', 'N/A'),
            vix=market_context.get('regime', {}).get('vix_level', 'N/A'),
            sector=fundamental_data.get('sector', 'Unknown'),
            sector_strength=market_context.get('sector_strength', 'N/A')
        )

        if self.provider == "openai" and self.client:
            return self._call_openai(prompt)
        else:
            return self._call_ollama(prompt)

    def get_tomorrow_picks(
        self,
        all_analyses: list[dict],
        top_n: int = 5
    ) -> str:
        """
        Generate "Tomorrow's Top Picks" summary.

        Args:
            all_analyses: List of analysis dicts for each stock
            top_n: Number of top picks

        Returns:
            Formatted picks summary
        """
        prompt = f"""Based on these {len(all_analyses)} stock analyses, identify the TOP {top_n} stocks to trade TOMORROW.

ANALYSES:
{json.dumps(all_analyses[:20], indent=2, default=str)}

For each pick, provide:
1. Stock symbol
2. Action: BUY or SELL
3. Entry price
4. Stop loss
5. Target price
6. Expected return %
7. Confidence level
8. One-line reason

Format as a clean table.
Also provide:
- Total expected portfolio return if all trades hit target
- Risk warning
"""

        if self.provider == "openai" and self.client:
            return self._call_openai(prompt)
        else:
            return self._call_ollama(prompt)


class QuickInsight:
    """
    Quick insight generator without full AI analysis.
    Uses rule-based logic for instant results.
    """

    @staticmethod
    def generate(
        symbol: str,
        technical: dict,
        fundamental: dict,
        pattern_stats: dict
    ) -> dict:
        """Generate quick rule-based insight."""

        score = 0
        reasons = []
        risks = []

        # Technical scoring
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            score += 2
            reasons.append(f"RSI oversold at {rsi:.0f} - potential bounce")
        elif rsi > 70:
            score -= 2
            risks.append(f"RSI overbought at {rsi:.0f} - may pullback")

        if technical.get('trend') == 'UPTREND':
            score += 1
            reasons.append("Price in uptrend above key moving averages")
        elif technical.get('trend') == 'DOWNTREND':
            score -= 1
            risks.append("Price in downtrend - fighting the trend")

        if technical.get('volume_ratio', 1) > 1.5:
            score += 1
            reasons.append(f"High volume ({technical.get('volume_ratio'):.1f}x) confirms move")

        # Fundamental scoring
        pe = fundamental.get('pe_ratio')
        if pe and pe < 15:
            score += 1
            reasons.append(f"Low PE ({pe:.1f}) - potentially undervalued")
        elif pe and pe > 40:
            score -= 1
            risks.append(f"High PE ({pe:.1f}) - expensive valuation")

        # Pattern scoring
        if pattern_stats.get('win_rate', 0) > 0.6:
            score += 1
            reasons.append(f"Historical patterns show {pattern_stats['win_rate']:.0%} win rate")

        # Determine signal
        if score >= 3:
            signal = "STRONG BUY"
            confidence = min(0.85, 0.6 + score * 0.05)
        elif score >= 1:
            signal = "BUY"
            confidence = min(0.75, 0.55 + score * 0.05)
        elif score <= -3:
            signal = "STRONG SELL"
            confidence = min(0.85, 0.6 + abs(score) * 0.05)
        elif score <= -1:
            signal = "SELL"
            confidence = min(0.75, 0.55 + abs(score) * 0.05)
        else:
            signal = "HOLD"
            confidence = 0.5

        # Calculate trade levels
        current = technical.get('close', 0)
        atr = technical.get('atr', current * 0.02)

        if "BUY" in signal:
            entry = current
            stop = current - (atr * 1.5)
            target = current + (atr * 2.5)
        else:
            entry = current
            stop = current + (atr * 1.5)
            target = current - (atr * 2.5)

        expected_return = abs(target - entry) / entry * 100
        max_loss = abs(entry - stop) / entry * 100

        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': round(confidence, 2),
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'target': round(target, 2),
            'expected_return': round(expected_return, 2),
            'max_loss': round(max_loss, 2),
            'risk_reward': round(expected_return / max_loss, 2) if max_loss > 0 else 0,
            'reasons': reasons[:3],
            'risks': risks[:3],
            'score': score
        }
