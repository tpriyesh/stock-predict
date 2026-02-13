"""
SentimentAnalyzer - LLM for Text Analysis ONLY

CRITICAL: LLMs should NOT analyze numbers or make price predictions.

WRONG (Current):
    "RSI: 45, MACD: -0.5, P/E: 25 - Predict BUY/SELL"
    LLMs are terrible at math and will hallucinate

CORRECT (This module):
    "Analyze this news: 'Company beats earnings expectations by 15%'"
    LLMs excel at understanding text, sentiment, and context

This module uses LLMs for what they're good at:
- News headline sentiment
- Earnings call analysis
- Management commentary interpretation
- Market sentiment from social media
- Sector trend understanding

ML models handle the numbers. LLMs handle the text.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from config.settings import get_settings


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    source: str  # 'openai', 'gemini', 'claude'
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    key_catalysts: List[str]
    key_risks: List[str]
    news_quality: str  # 'high', 'medium', 'low'
    reasoning: str

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'confidence': self.confidence,
            'key_catalysts': self.key_catalysts,
            'key_risks': self.key_risks,
            'news_quality': self.news_quality,
            'reasoning': self.reasoning
        }


@dataclass
class ConsensusSentiment:
    """Aggregated sentiment from multiple LLMs."""
    average_score: float  # -1.0 to 1.0
    consensus_label: str  # 'bullish', 'bearish', 'neutral'
    agreement_level: float  # How much LLMs agree (0-1)
    probability_adjustment: float  # How to adjust base prediction
    individual_results: List[SentimentResult]
    combined_catalysts: List[str]
    combined_risks: List[str]
    summary: str

    def to_dict(self) -> dict:
        return {
            'average_score': self.average_score,
            'consensus_label': self.consensus_label,
            'agreement_level': self.agreement_level,
            'probability_adjustment': self.probability_adjustment,
            'individual_results': [r.to_dict() for r in self.individual_results],
            'combined_catalysts': self.combined_catalysts,
            'combined_risks': self.combined_risks,
            'summary': self.summary
        }


class SentimentAnalyzer:
    """
    Multi-LLM sentiment analyzer for stock news and text.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     TEXT SOURCES                             │
    │   News Headlines   Earnings Calls   Market Commentary       │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  SENTIMENT ANALYZER   │
                    │                       │
                    │   OpenAI GPT-4       │
                    │   Gemini             │
                    │   Claude             │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  CONSENSUS ENGINE    │
                    │                       │
                    │  Aggregates results  │
                    │  Calculates agreement│
                    │  Probability adjust  │
                    └───────────────────────┘

    Usage:
        analyzer = SentimentAnalyzer()

        result = analyzer.analyze(
            symbol="RELIANCE",
            news_headlines=["Reliance beats Q3 expectations by 15%"],
            sector="Energy",
            market_context="Bullish market with rising oil prices"
        )

        # Use sentiment to adjust ML prediction
        ml_probability = 0.55
        adjusted = ml_probability + result.probability_adjustment
    """

    # Sentiment prompt template - NO NUMBERS, TEXT ONLY
    SENTIMENT_PROMPT = """You are a financial news sentiment analyst. Analyze the sentiment of the following news and market context.

IMPORTANT RULES:
1. DO NOT analyze any numbers, charts, or technical indicators
2. ONLY analyze the TEXT content (news headlines, commentary, context)
3. Focus on sentiment, tone, and implications of the news
4. Be objective - identify both positive and negative factors

STOCK: {symbol}
SECTOR: {sector}

RECENT NEWS:
{news_text}

MARKET CONTEXT:
{market_context}

Respond ONLY in this JSON format:
{{
    "sentiment_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "sentiment_label": "bullish" | "bearish" | "neutral",
    "confidence": <float from 0.0 to 1.0>,
    "key_catalysts": ["list of positive factors from the news"],
    "key_risks": ["list of negative factors from the news"],
    "news_quality": "high" | "medium" | "low",
    "reasoning": "2-3 sentence explanation of your sentiment assessment"
}}

Be balanced and objective. Consider both bullish and bearish factors."""

    def __init__(self):
        """Initialize sentiment analyzer."""
        self.settings = get_settings()

        # API Keys
        self.openai_key = self.settings.openai_api_key
        self.gemini_key = "AIzaSyBrM5XxpCnNd6vs4wQm8bigt3MPg-GwUqk"
        self.claude_key = os.getenv("ANTHROPIC_API_KEY")

    def _create_prompt(self,
                       symbol: str,
                       news_headlines: List[str],
                       sector: str,
                       market_context: str) -> str:
        """Create the sentiment analysis prompt."""
        news_text = "\n".join(f"- {headline}" for headline in news_headlines)

        return self.SENTIMENT_PROMPT.format(
            symbol=symbol,
            sector=sector,
            news_text=news_text if news_headlines else "No recent news available",
            market_context=market_context or "No specific market context provided"
        )

    def _parse_response(self, content: str, source: str) -> Optional[SentimentResult]:
        """Parse LLM response into SentimentResult."""
        try:
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            return SentimentResult(
                source=source,
                sentiment_score=float(data.get("sentiment_score", 0)),
                sentiment_label=data.get("sentiment_label", "neutral"),
                confidence=float(data.get("confidence", 0.5)),
                key_catalysts=data.get("key_catalysts", []),
                key_risks=data.get("key_risks", []),
                news_quality=data.get("news_quality", "medium"),
                reasoning=data.get("reasoning", "")
            )
        except Exception as e:
            logger.error(f"Failed to parse {source} response: {e}")
            return None

    def _call_openai(self, prompt: str) -> Optional[SentimentResult]:
        """Get sentiment from OpenAI."""
        if not self.openai_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst. Respond only in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return self._parse_response(content, "openai")

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[SentimentResult]:
        """Get sentiment from Gemini."""
        if not self.gemini_key:
            return None

        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)

            models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

            for model in models:
                try:
                    response = client.models.generate_content(model=model, contents=prompt)
                    return self._parse_response(response.text, "gemini")
                except Exception as e:
                    if "quota" in str(e).lower():
                        continue
                    raise

            return None

        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    def _call_claude(self, prompt: str) -> Optional[SentimentResult]:
        """Get sentiment from Claude."""
        if not self.claude_key:
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.claude_key)

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            return self._parse_response(content, "claude")

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None

    def analyze(self,
                symbol: str,
                news_headlines: List[str],
                sector: str = "Unknown",
                market_context: str = "") -> ConsensusSentiment:
        """
        Analyze sentiment using multiple LLMs.

        Args:
            symbol: Stock symbol
            news_headlines: List of recent news headlines
            sector: Stock sector
            market_context: Overall market context

        Returns:
            ConsensusSentiment with aggregated results
        """
        prompt = self._create_prompt(symbol, news_headlines, sector, market_context)

        results: List[SentimentResult] = []

        # Call all LLMs in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._call_openai, prompt): 'openai',
                executor.submit(self._call_gemini, prompt): 'gemini',
                executor.submit(self._call_claude, prompt): 'claude'
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Sentiment {futures[future]}: {result.sentiment_label} "
                                   f"({result.sentiment_score:+.2f})")
                except Exception as e:
                    logger.error(f"LLM {futures[future]} failed: {e}")

        # Aggregate results
        return self._aggregate_results(results, symbol)

    def _aggregate_results(self,
                           results: List[SentimentResult],
                           symbol: str) -> ConsensusSentiment:
        """Aggregate multiple LLM results into consensus."""
        if not results:
            return ConsensusSentiment(
                average_score=0,
                consensus_label="neutral",
                agreement_level=0,
                probability_adjustment=0,
                individual_results=[],
                combined_catalysts=[],
                combined_risks=[],
                summary="No LLM responses available"
            )

        # Calculate average sentiment
        scores = [r.sentiment_score for r in results]
        confidences = [r.confidence for r in results]

        # Confidence-weighted average
        weighted_scores = [s * c for s, c in zip(scores, confidences)]
        avg_score = sum(weighted_scores) / sum(confidences) if sum(confidences) > 0 else 0

        # Determine consensus label
        if avg_score >= 0.3:
            consensus_label = "bullish"
        elif avg_score <= -0.3:
            consensus_label = "bearish"
        else:
            consensus_label = "neutral"

        # Agreement level (1 - std of scores normalized)
        score_std = np.std(scores) if len(scores) > 1 else 0
        agreement = max(0, 1 - score_std)  # Higher agreement = lower std

        # Convert sentiment to probability adjustment
        # Sentiment ranges from -1 to 1
        # We want to adjust probability by at most ±10%
        prob_adjustment = avg_score * 0.10 * agreement

        # Combine catalysts and risks (deduplicated)
        all_catalysts = []
        all_risks = []
        for r in results:
            all_catalysts.extend(r.key_catalysts)
            all_risks.extend(r.key_risks)

        unique_catalysts = list(dict.fromkeys(all_catalysts))[:5]
        unique_risks = list(dict.fromkeys(all_risks))[:5]

        # Create summary
        labels = [r.sentiment_label for r in results]
        n_bullish = labels.count("bullish")
        n_bearish = labels.count("bearish")
        n_neutral = labels.count("neutral")

        summary = f"{symbol}: {len(results)} LLMs analyzed. "
        summary += f"Consensus: {consensus_label.upper()} "
        summary += f"({n_bullish} bullish, {n_neutral} neutral, {n_bearish} bearish). "
        summary += f"Agreement: {agreement:.0%}."

        return ConsensusSentiment(
            average_score=avg_score,
            consensus_label=consensus_label,
            agreement_level=agreement,
            probability_adjustment=prob_adjustment,
            individual_results=results,
            combined_catalysts=unique_catalysts,
            combined_risks=unique_risks,
            summary=summary
        )

    def analyze_earnings(self,
                         symbol: str,
                         earnings_summary: str,
                         guidance: str = "") -> SentimentResult:
        """
        Specialized analysis for earnings announcements.

        Args:
            symbol: Stock symbol
            earnings_summary: Summary of earnings (beat/miss, key numbers)
            guidance: Forward guidance from management

        Returns:
            SentimentResult
        """
        prompt = f"""Analyze the sentiment of this earnings announcement.

STOCK: {symbol}

EARNINGS SUMMARY:
{earnings_summary}

FORWARD GUIDANCE:
{guidance or "No guidance provided"}

Focus on:
1. Beat/miss expectations
2. Revenue and profit trends
3. Management tone
4. Forward outlook

Respond in JSON format with sentiment_score, sentiment_label, confidence, key_catalysts, key_risks, news_quality, and reasoning."""

        # Use OpenAI for earnings (most capable for nuanced analysis)
        result = self._call_openai(prompt)

        if result:
            return result
        else:
            return SentimentResult(
                source="none",
                sentiment_score=0,
                sentiment_label="neutral",
                confidence=0,
                key_catalysts=[],
                key_risks=[],
                news_quality="low",
                reasoning="Could not analyze earnings"
            )


def demo():
    """Demonstrate sentiment analysis."""
    print("=" * 60)
    print("SentimentAnalyzer Demo - Text-Only LLM Analysis")
    print("=" * 60)

    analyzer = SentimentAnalyzer()

    # Sample news headlines
    news = [
        "Reliance Industries Q3 profit jumps 18% YoY, beats street estimates",
        "Reliance Jio adds 8 million subscribers in December quarter",
        "Analysts upgrade Reliance to 'Buy' on strong retail performance",
        "Global oil prices surge 5% on supply concerns, positive for Reliance"
    ]

    print("\n--- Analyzing News Sentiment ---")
    print("News Headlines:")
    for headline in news:
        print(f"  - {headline}")

    result = analyzer.analyze(
        symbol="RELIANCE",
        news_headlines=news,
        sector="Energy/Conglomerate",
        market_context="Market is bullish with FII buying. Energy sector outperforming."
    )

    print(f"\n--- Consensus Result ---")
    print(f"Average Sentiment Score: {result.average_score:+.2f}")
    print(f"Consensus Label: {result.consensus_label.upper()}")
    print(f"Agreement Level: {result.agreement_level:.0%}")
    print(f"Probability Adjustment: {result.probability_adjustment:+.2%}")

    print(f"\n{result.summary}")

    print("\nKey Catalysts (from LLMs):")
    for catalyst in result.combined_catalysts[:3]:
        print(f"  + {catalyst}")

    print("\nKey Risks (from LLMs):")
    for risk in result.combined_risks[:3]:
        print(f"  - {risk}")

    print("\nIndividual LLM Results:")
    for r in result.individual_results:
        print(f"  {r.source}: {r.sentiment_label} ({r.sentiment_score:+.2f}), "
              f"confidence={r.confidence:.0%}")


if __name__ == "__main__":
    demo()
