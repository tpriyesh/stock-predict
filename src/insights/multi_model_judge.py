"""
Multi-Model AI Judge System

Uses multiple AI models (OpenAI, Gemini, Claude) as judges to:
1. Analyze stocks independently
2. Cross-verify recommendations
3. Return consensus-based verdicts

Consensus > Single Model Opinion
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings


class Verdict(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class ModelJudgment:
    """Single model's judgment."""
    model_name: str
    verdict: str
    confidence: float
    reasoning: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    time_horizon: str
    key_factors: List[str]


@dataclass
class ConsensusResult:
    """Consensus from all models."""
    symbol: str
    final_verdict: str
    consensus_score: float  # 0-1, how much models agree
    confidence: float

    # Individual judgments
    judgments: List[ModelJudgment]

    # Aggregated
    avg_entry: float
    avg_stop_loss: float
    avg_target: float

    # Summary
    bull_count: int
    bear_count: int
    neutral_count: int

    combined_reasoning: str


class MultiModelJudge:
    """
    Multi-Model AI Judge for Stock Analysis.

    Uses OpenAI GPT-4, Google Gemini, and optionally Claude
    to get consensus-based stock recommendations.
    """

    def __init__(self):
        settings = get_settings()

        # API Keys
        self.openai_key = settings.openai_api_key
        self.gemini_key = "AIzaSyBrM5XxpCnNd6vs4wQm8bigt3MPg-GwUqk"  # User provided
        self.claude_key = os.getenv("ANTHROPIC_API_KEY")

        # Available models
        self.available_models = []
        if self.openai_key:
            self.available_models.append("openai")
        if self.gemini_key:
            self.available_models.append("gemini")
        if self.claude_key:
            self.available_models.append("claude")

        logger.info(f"Multi-Model Judge initialized with: {self.available_models}")

    def _create_analysis_prompt(self, symbol: str, stock_data: Dict) -> str:
        """Create standardized prompt for all models."""
        return f"""You are an expert Indian stock market analyst. Analyze this stock and provide your judgment.

STOCK: {symbol}
CURRENT PRICE: ₹{stock_data.get('price', 'N/A')}
SECTOR: {stock_data.get('sector', 'N/A')}
MARKET CAP: ₹{stock_data.get('market_cap_cr', 'N/A')} Cr

PERFORMANCE:
- 1 Day: {stock_data.get('change_1d', 'N/A')}%
- 1 Week: {stock_data.get('change_1w', 'N/A')}%
- 1 Month: {stock_data.get('change_1m', 'N/A')}%
- 3 Months: {stock_data.get('change_3m', 'N/A')}%
- 6 Months: {stock_data.get('change_6m', 'N/A')}%
- 1 Year: {stock_data.get('change_1y', 'N/A')}%

TECHNICAL INDICATORS:
- RSI (14): {stock_data.get('rsi', 'N/A')}
- Above 50-day MA: {stock_data.get('above_ma50', 'N/A')}
- Above 200-day MA: {stock_data.get('above_ma200', 'N/A')}
- Volume Ratio: {stock_data.get('volume_ratio', 'N/A')}x average

You MUST respond in this EXACT JSON format:
{{
    "verdict": "STRONG_BUY" or "BUY" or "HOLD" or "SELL" or "STRONG_SELL",
    "confidence": 0.0 to 1.0,
    "reasoning": "2-3 sentence explanation",
    "entry_price": number or null,
    "stop_loss": number or null,
    "target_price": number or null,
    "time_horizon": "INTRADAY" or "SWING" or "POSITIONAL" or "LONG_TERM",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Be objective and data-driven. Consider both bullish and bearish factors."""

    def _call_openai(self, prompt: str) -> Optional[ModelJudgment]:
        """Get judgment from OpenAI GPT-4."""
        if not self.openai_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a stock market analyst. Always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content

            # Parse JSON from response
            try:
                # Try to extract JSON from the response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                return ModelJudgment(
                    model_name="OpenAI GPT-4o-mini",
                    verdict=data.get("verdict", "HOLD"),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    entry_price=data.get("entry_price"),
                    stop_loss=data.get("stop_loss"),
                    target_price=data.get("target_price"),
                    time_horizon=data.get("time_horizon", "SWING"),
                    key_factors=data.get("key_factors", [])
                )
            except json.JSONDecodeError:
                logger.warning(f"OpenAI returned invalid JSON: {content[:100]}")
                return None

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[ModelJudgment]:
        """Get judgment from Google Gemini."""
        if not self.gemini_key:
            return None

        try:
            from google import genai

            client = genai.Client(api_key=self.gemini_key)

            # Try multiple models in case of quota issues (using full model paths)
            models_to_try = [
                "gemini-2.5-flash",      # Latest stable
                "gemini-2.0-flash",      # Fallback
                "gemini-2.0-flash-lite", # Lightweight fallback
            ]
            response = None
            used_model_name = None

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    used_model_name = model_name
                    break  # Success, exit loop
                except Exception as model_error:
                    error_str = str(model_error).lower()
                    if "quota" in error_str or "429" in error_str or "exhausted" in error_str:
                        logger.debug(f"Quota exhausted for {model_name}, trying next...")
                        continue  # Try next model
                    raise  # Other error, re-raise

            if response is None:
                logger.warning("All Gemini models exhausted quota")
                return None

            content = response.text

            # Parse JSON from response
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                return ModelJudgment(
                    model_name=f"Google {used_model_name or 'Gemini'}",
                    verdict=data.get("verdict", "HOLD"),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    entry_price=data.get("entry_price"),
                    stop_loss=data.get("stop_loss"),
                    target_price=data.get("target_price"),
                    time_horizon=data.get("time_horizon", "SWING"),
                    key_factors=data.get("key_factors", [])
                )
            except json.JSONDecodeError:
                logger.warning(f"Gemini returned invalid JSON: {content[:100]}")
                return None

        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    def _call_claude(self, prompt: str) -> Optional[ModelJudgment]:
        """Get judgment from Anthropic Claude."""
        if not self.claude_key:
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.claude_key)

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text

            # Parse JSON from response
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                return ModelJudgment(
                    model_name="Anthropic Claude 3 Haiku",
                    verdict=data.get("verdict", "HOLD"),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    entry_price=data.get("entry_price"),
                    stop_loss=data.get("stop_loss"),
                    target_price=data.get("target_price"),
                    time_horizon=data.get("time_horizon", "SWING"),
                    key_factors=data.get("key_factors", [])
                )
            except json.JSONDecodeError:
                logger.warning(f"Claude returned invalid JSON: {content[:100]}")
                return None

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None

    def get_consensus(self, symbol: str, stock_data: Dict) -> Optional[ConsensusResult]:
        """
        Get consensus judgment from all available models.

        Each model analyzes independently, then we aggregate results.
        """
        prompt = self._create_analysis_prompt(symbol, stock_data)

        judgments = []

        # Call all models in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if "openai" in self.available_models:
                futures[executor.submit(self._call_openai, prompt)] = "openai"
            if "gemini" in self.available_models:
                futures[executor.submit(self._call_gemini, prompt)] = "gemini"
            if "claude" in self.available_models:
                futures[executor.submit(self._call_claude, prompt)] = "claude"

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        judgments.append(result)
                        logger.info(f"{model_name}: {result.verdict} ({result.confidence:.0%})")
                except Exception as e:
                    logger.error(f"{model_name} failed: {e}")

        if not judgments:
            logger.warning("No model returned valid judgment")
            return None

        # Calculate consensus
        return self._calculate_consensus(symbol, judgments)

    def _calculate_consensus(self, symbol: str, judgments: List[ModelJudgment]) -> ConsensusResult:
        """Calculate consensus from multiple judgments."""

        # Count verdicts
        verdict_scores = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -2
        }

        total_score = 0
        total_confidence = 0
        bull_count = 0
        bear_count = 0
        neutral_count = 0

        entries = []
        stop_losses = []
        targets = []
        all_factors = []
        all_reasoning = []

        for j in judgments:
            score = verdict_scores.get(j.verdict, 0)
            total_score += score * j.confidence
            total_confidence += j.confidence

            if score > 0:
                bull_count += 1
            elif score < 0:
                bear_count += 1
            else:
                neutral_count += 1

            if j.entry_price:
                entries.append(j.entry_price)
            if j.stop_loss:
                stop_losses.append(j.stop_loss)
            if j.target_price:
                targets.append(j.target_price)

            all_factors.extend(j.key_factors)
            all_reasoning.append(f"**{j.model_name}**: {j.reasoning}")

        # Calculate weighted average score
        n = len(judgments)
        avg_score = total_score / total_confidence if total_confidence > 0 else 0

        # Convert score back to verdict
        if avg_score >= 1.5:
            final_verdict = "STRONG_BUY"
        elif avg_score >= 0.5:
            final_verdict = "BUY"
        elif avg_score >= -0.5:
            final_verdict = "HOLD"
        elif avg_score >= -1.5:
            final_verdict = "SELL"
        else:
            final_verdict = "STRONG_SELL"

        # Calculate consensus score (how much models agree)
        # 1.0 = all models agree, 0.0 = complete disagreement
        verdicts = [j.verdict for j in judgments]
        unique_verdicts = len(set(verdicts))
        consensus_score = 1 - (unique_verdicts - 1) / max(n - 1, 1)

        # Average confidence
        avg_confidence = total_confidence / n

        # Average prices
        avg_entry = sum(entries) / len(entries) if entries else 0
        avg_sl = sum(stop_losses) / len(stop_losses) if stop_losses else 0
        avg_target = sum(targets) / len(targets) if targets else 0

        # Combined reasoning
        combined = "\n\n".join(all_reasoning)

        return ConsensusResult(
            symbol=symbol,
            final_verdict=final_verdict,
            consensus_score=round(consensus_score, 2),
            confidence=round(avg_confidence, 2),
            judgments=judgments,
            avg_entry=round(avg_entry, 2),
            avg_stop_loss=round(avg_sl, 2),
            avg_target=round(avg_target, 2),
            bull_count=bull_count,
            bear_count=bear_count,
            neutral_count=neutral_count,
            combined_reasoning=combined
        )

    def analyze_stock_with_consensus(self, symbol: str) -> Optional[ConsensusResult]:
        """
        Full analysis: fetch data + get multi-model consensus.
        """
        # Fetch stock data
        try:
            from src.insights.stock_screener import StockScreener
            screener = StockScreener()
            stock_data = screener.fetch_stock_data(symbol)

            if not stock_data:
                logger.error(f"Could not fetch data for {symbol}")
                return None

            # Convert to dict for prompt
            data_dict = {
                'price': stock_data.price,
                'sector': stock_data.sector,
                'market_cap_cr': stock_data.market_cap_cr,
                'change_1d': stock_data.change_1d,
                'change_1w': stock_data.change_1w,
                'change_1m': stock_data.change_1m,
                'change_3m': stock_data.change_3m,
                'change_6m': stock_data.change_6m,
                'change_1y': stock_data.change_1y,
                'rsi': stock_data.rsi,
                'above_ma50': stock_data.above_ma50,
                'above_ma200': stock_data.above_ma200,
                'volume_ratio': stock_data.volume_ratio,
            }

            return self.get_consensus(symbol, data_dict)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None


def demo_multi_model():
    """Demo the multi-model judge."""
    print("=" * 70)
    print("MULTI-MODEL AI JUDGE SYSTEM")
    print("=" * 70)

    judge = MultiModelJudge()
    print(f"\nAvailable models: {judge.available_models}")

    # Test with RELIANCE
    print("\nAnalyzing RELIANCE with multiple AI models...")

    result = judge.analyze_stock_with_consensus("RELIANCE")

    if result:
        print(f"\n{'=' * 50}")
        print(f"CONSENSUS RESULT for {result.symbol}")
        print(f"{'=' * 50}")
        print(f"Final Verdict: {result.final_verdict}")
        print(f"Consensus Score: {result.consensus_score:.0%} (how much models agree)")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"\nVotes: {result.bull_count} Bullish | {result.neutral_count} Neutral | {result.bear_count} Bearish")

        if result.avg_entry > 0:
            print(f"\nTrade Setup:")
            print(f"  Entry: ₹{result.avg_entry}")
            print(f"  Stop Loss: ₹{result.avg_stop_loss}")
            print(f"  Target: ₹{result.avg_target}")

        print(f"\n--- Individual Model Opinions ---")
        for j in result.judgments:
            print(f"\n{j.model_name}:")
            print(f"  Verdict: {j.verdict} ({j.confidence:.0%} confident)")
            print(f"  {j.reasoning[:100]}...")


if __name__ == "__main__":
    demo_multi_model()
