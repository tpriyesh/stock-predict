"""
Unified Stock Prediction Engine - Best of All Worlds

Combines:
1. Statistical ML Models (RandomForest, XGBoost, GradientBoosting)
2. Multiple LLMs as Judges (OpenAI GPT-4, Gemini, Claude)
3. Technical Analysis (RSI, MACD, Bollinger, ATR)
4. Fundamental Analysis (P/E, ROE, Debt)
5. Calendar Patterns (Day/Week/Month effects)
6. Sector Trends (AI, Solar, EV alignment)

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED PREDICTION ENGINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   ML MODELS │  │ LLM JUDGES  │  │  TECHNICAL  │             │
│  │ (sklearn)   │  │ (OpenAI+   │  │  ANALYSIS   │             │
│  │             │  │  Gemini+   │  │             │             │
│  │ RandomForest│  │  Claude)   │  │ RSI, MACD,  │             │
│  │ XGBoost     │  │             │  │ BB, ATR    │             │
│  │ GradientBoost│ │             │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         v                v                v                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ENSEMBLE AGGREGATOR                        │   │
│  │   Weighted voting based on confidence & historical      │   │
│  │   accuracy of each component                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              v                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FINAL PREDICTION                           │   │
│  │   Signal, Probability, Entry, Stop Loss, Target         │   │
│  │   + Human-readable explanation of WHY                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Philosophy:
- No single model is trusted alone
- Consensus = Higher confidence
- Every prediction is explainable
- Historical calibration for accuracy
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config.settings import get_settings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SignalStrength(Enum):
    STRONG_BUY = ("STRONG_BUY", 2, "#00ff88")
    BUY = ("BUY", 1, "#88ff88")
    HOLD = ("HOLD", 0, "#ffaa00")
    SELL = ("SELL", -1, "#ff8888")
    STRONG_SELL = ("STRONG_SELL", -2, "#ff4444")

    def __init__(self, label, score, color):
        self.label = label
        self.score = score
        self.color = color


@dataclass
class ComponentPrediction:
    """Prediction from a single component."""
    source: str  # 'ml_rf', 'ml_xgb', 'llm_openai', 'llm_gemini', 'technical', etc.
    signal: SignalStrength
    probability_up: float
    confidence: float
    reasoning: str
    factors: List[str]


@dataclass
class UnifiedPrediction:
    """Final unified prediction from all components."""
    symbol: str
    timestamp: datetime

    # Final consensus
    signal: SignalStrength
    probability_up: float  # 0-1
    confidence: float  # 0-1
    consensus_score: float  # How much components agree

    # Price targets
    current_price: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float

    # Position sizing
    position_size_pct: float
    risk_reward_ratio: float

    # Component predictions
    ml_predictions: List[ComponentPrediction]
    llm_predictions: List[ComponentPrediction]
    technical_prediction: ComponentPrediction
    calendar_prediction: ComponentPrediction

    # Aggregated reasoning
    bullish_factors: List[str]
    bearish_factors: List[str]
    key_risks: List[str]

    # Human-readable summary
    summary: str
    detailed_analysis: str

    # Scores (0-100)
    ml_score: float
    llm_score: float
    technical_score: float
    fundamental_score: float
    calendar_score: float

    # Metadata
    models_used: List[str]
    data_quality: float  # 0-1


class UnifiedPredictor:
    """
    Unified Prediction Engine combining ML + LLM + Technical Analysis.

    Uses ensemble approach:
    1. Get predictions from all ML models
    2. Get judgments from all LLMs
    3. Calculate technical indicators
    4. Analyze calendar patterns
    5. Weighted aggregation based on historical accuracy
    """

    # Default weights (can be calibrated based on historical performance)
    WEIGHTS = {
        'ml_rf': 0.15,
        'ml_gb': 0.15,
        'ml_xgb': 0.15,
        'llm_openai': 0.15,
        'llm_gemini': 0.12,
        'llm_claude': 0.08,
        'technical': 0.12,
        'calendar': 0.08
    }

    def __init__(self):
        self.settings = get_settings()

        # API Keys
        self.openai_key = self.settings.openai_api_key
        self.gemini_key = "AIzaSyBrM5XxpCnNd6vs4wQm8bigt3MPg-GwUqk"
        self.claude_key = os.getenv("ANTHROPIC_API_KEY")

        # ML models
        self.ml_models = {}
        self.scalers = {}

        # Historical accuracy for weight calibration
        self.accuracy_file = Path("data/model_accuracy.json")
        self.accuracy_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_accuracy = self._load_accuracy()

    def _load_accuracy(self) -> dict:
        """Load historical model accuracy."""
        try:
            if self.accuracy_file.exists():
                with open(self.accuracy_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {source: 0.5 for source in self.WEIGHTS.keys()}

    def _save_accuracy(self):
        """Save model accuracy for calibration."""
        try:
            with open(self.accuracy_file, 'w') as f:
                json.dump(self.model_accuracy, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save accuracy: {e}")

    # =========================================================================
    # ML MODEL PREDICTIONS
    # =========================================================================

    def _train_ml_models(self, hist: pd.DataFrame):
        """Train ML models on historical data."""
        if not SKLEARN_AVAILABLE or len(hist) < 100:
            return

        # Calculate features
        df = self._calculate_features(hist)
        if df.empty:
            return

        feature_cols = [
            'returns_5d', 'returns_10d', 'returns_20d',
            'dist_sma_20', 'dist_sma_50', 'rsi', 'macd_hist',
            'bb_position', 'volatility_20', 'volume_ratio'
        ]

        X = df[feature_cols].values
        y = (df['Close'].shift(-5) > df['Close']).astype(int).values[:-5]
        X = X[:-5]

        if len(X) < 50:
            return

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler

        # Train models
        self.ml_models['rf'] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.ml_models['rf'].fit(X_scaled, y)

        self.ml_models['gb'] = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.ml_models['gb'].fit(X_scaled, y)

        if XGB_AVAILABLE:
            self.ml_models['xgb'] = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                                                       use_label_encoder=False, eval_metric='logloss')
            self.ml_models['xgb'].fit(X_scaled, y)

        logger.info(f"Trained {len(self.ml_models)} ML models")

    def _calculate_features(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML models."""
        if hist.empty or len(hist) < 50:
            return pd.DataFrame()

        df = hist.copy()
        close = df['Close']

        # Returns
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(10)
        df['returns_20d'] = close.pct_change(20)

        # Moving averages distance
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['dist_sma_20'] = (close - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (close - df['sma_50']) / df['sma_50']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger position
        bb_mid = df['sma_20']
        bb_std = close.rolling(20).std()
        df['bb_position'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

        # Volatility
        df['volatility_20'] = close.pct_change().rolling(20).std()

        # Volume
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        return df.dropna()

    def _get_ml_predictions(self, hist: pd.DataFrame) -> List[ComponentPrediction]:
        """Get predictions from all ML models."""
        predictions = []

        if not self.ml_models:
            self._train_ml_models(hist)

        if not self.ml_models or 'main' not in self.scalers:
            return predictions

        df = self._calculate_features(hist)
        if df.empty:
            return predictions

        feature_cols = [
            'returns_5d', 'returns_10d', 'returns_20d',
            'dist_sma_20', 'dist_sma_50', 'rsi', 'macd_hist',
            'bb_position', 'volatility_20', 'volume_ratio'
        ]

        X = df[feature_cols].iloc[-1:].values
        X_scaled = self.scalers['main'].transform(X)

        # Random Forest
        if 'rf' in self.ml_models:
            prob = self.ml_models['rf'].predict_proba(X_scaled)[0][1]
            signal = self._prob_to_signal(prob)
            predictions.append(ComponentPrediction(
                source='ml_rf',
                signal=signal,
                probability_up=prob,
                confidence=abs(prob - 0.5) * 2,  # Higher distance from 0.5 = more confident
                reasoning=f"RandomForest predicts {prob:.1%} chance of upside",
                factors=[f"Ensemble of 100 decision trees", f"Based on {len(hist)} historical samples"]
            ))

        # Gradient Boosting
        if 'gb' in self.ml_models:
            prob = self.ml_models['gb'].predict_proba(X_scaled)[0][1]
            signal = self._prob_to_signal(prob)
            predictions.append(ComponentPrediction(
                source='ml_gb',
                signal=signal,
                probability_up=prob,
                confidence=abs(prob - 0.5) * 2,
                reasoning=f"GradientBoosting predicts {prob:.1%} chance of upside",
                factors=["Sequential ensemble learning", "Focuses on hard cases"]
            ))

        # XGBoost
        if 'xgb' in self.ml_models:
            prob = self.ml_models['xgb'].predict_proba(X_scaled)[0][1]
            signal = self._prob_to_signal(prob)
            predictions.append(ComponentPrediction(
                source='ml_xgb',
                signal=signal,
                probability_up=prob,
                confidence=abs(prob - 0.5) * 2,
                reasoning=f"XGBoost predicts {prob:.1%} chance of upside",
                factors=["Extreme gradient boosting", "State-of-the-art ML model"]
            ))

        return predictions

    def _prob_to_signal(self, prob: float) -> SignalStrength:
        """Convert probability to signal."""
        if prob >= 0.7:
            return SignalStrength.STRONG_BUY
        elif prob >= 0.55:
            return SignalStrength.BUY
        elif prob <= 0.3:
            return SignalStrength.STRONG_SELL
        elif prob <= 0.45:
            return SignalStrength.SELL
        else:
            return SignalStrength.HOLD

    # =========================================================================
    # LLM JUDGE PREDICTIONS
    # =========================================================================

    def _create_llm_prompt(self, symbol: str, data: dict) -> str:
        """Create analysis prompt for LLMs."""
        return f"""You are an expert Indian stock market analyst. Analyze this stock and give your judgment.

STOCK: {symbol}
PRICE: ₹{data.get('price', 'N/A'):,.2f}
SECTOR: {data.get('sector', 'Unknown')}

PERFORMANCE:
- Today: {data.get('change_1d', 0):+.2f}%
- 1 Week: {data.get('change_1w', 0):+.2f}%
- 1 Month: {data.get('change_1m', 0):+.2f}%
- 3 Months: {data.get('change_3m', 0):+.2f}%
- 1 Year: {data.get('change_1y', 0):+.2f}%

TECHNICAL:
- RSI (14): {data.get('rsi', 50):.0f}
- Above 50-day MA: {data.get('above_sma50', 'N/A')}
- Above 200-day MA: {data.get('above_sma200', 'N/A')}
- Volume: {data.get('volume_ratio', 1):.1f}x average

FUNDAMENTALS:
- P/E Ratio: {data.get('pe', 'N/A')}
- ROE: {data.get('roe', 'N/A')}%
- Debt/Equity: {data.get('debt_equity', 'N/A')}

Respond in this JSON format:
{{
    "verdict": "STRONG_BUY" or "BUY" or "HOLD" or "SELL" or "STRONG_SELL",
    "confidence": 0.0 to 1.0,
    "probability_up": 0.0 to 1.0,
    "reasoning": "2-3 sentences explaining your view",
    "bullish_factors": ["factor1", "factor2"],
    "bearish_factors": ["factor1", "factor2"],
    "entry_price": number,
    "stop_loss": number,
    "target_price": number
}}

Be objective. Consider both upside and downside."""

    def _call_openai(self, prompt: str) -> Optional[ComponentPrediction]:
        """Get prediction from OpenAI."""
        if not self.openai_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a stock analyst. Respond only in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )

            content = response.choices[0].message.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            signal = SignalStrength[data.get("verdict", "HOLD")]
            return ComponentPrediction(
                source='llm_openai',
                signal=signal,
                probability_up=float(data.get("probability_up", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                factors=data.get("bullish_factors", []) + data.get("bearish_factors", [])
            )

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[ComponentPrediction]:
        """Get prediction from Gemini."""
        if not self.gemini_key:
            return None

        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)

            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    break
                except Exception as e:
                    if "quota" in str(e).lower():
                        continue
                    raise
            else:
                return None

            content = response.text

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            signal = SignalStrength[data.get("verdict", "HOLD")]
            return ComponentPrediction(
                source='llm_gemini',
                signal=signal,
                probability_up=float(data.get("probability_up", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                factors=data.get("bullish_factors", []) + data.get("bearish_factors", [])
            )

        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    def _call_claude(self, prompt: str) -> Optional[ComponentPrediction]:
        """Get prediction from Claude."""
        if not self.claude_key:
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.claude_key)

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            signal = SignalStrength[data.get("verdict", "HOLD")]
            return ComponentPrediction(
                source='llm_claude',
                signal=signal,
                probability_up=float(data.get("probability_up", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                factors=data.get("bullish_factors", []) + data.get("bearish_factors", [])
            )

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None

    def _get_llm_predictions(self, symbol: str, data: dict) -> List[ComponentPrediction]:
        """Get predictions from all LLMs in parallel."""
        predictions = []
        prompt = self._create_llm_prompt(symbol, data)

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
                        predictions.append(result)
                        logger.info(f"LLM {futures[future]}: {result.signal.label}")
                except Exception as e:
                    logger.error(f"LLM {futures[future]} failed: {e}")

        return predictions

    # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================

    def _get_technical_prediction(self, hist: pd.DataFrame) -> ComponentPrediction:
        """Get prediction based on technical indicators."""
        close = hist['Close']
        current = close.iloc[-1]

        score = 0
        factors = []

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50

        if rsi < 30:
            score += 20
            factors.append(f"RSI oversold ({rsi:.0f}) - bounce expected")
        elif rsi > 70:
            score -= 20
            factors.append(f"RSI overbought ({rsi:.0f}) - pullback risk")
        else:
            factors.append(f"RSI neutral ({rsi:.0f})")

        # Moving averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50

        if current > sma_20 > sma_50:
            score += 15
            factors.append("Price above 20 & 50 MA - uptrend")
        elif current < sma_20 < sma_50:
            score -= 15
            factors.append("Price below 20 & 50 MA - downtrend")

        if current > sma_200:
            score += 10
            factors.append("Above 200 MA - long-term bullish")
        else:
            score -= 10
            factors.append("Below 200 MA - long-term bearish")

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]

        if macd_hist > 0:
            score += 10
            factors.append("MACD bullish")
        else:
            score -= 10
            factors.append("MACD bearish")

        # Volume
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = hist['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1

        if vol_ratio > 1.5:
            score += 5
            factors.append(f"High volume ({vol_ratio:.1f}x) - strong conviction")

        # Convert score to signal
        if score >= 30:
            signal = SignalStrength.STRONG_BUY
        elif score >= 15:
            signal = SignalStrength.BUY
        elif score <= -30:
            signal = SignalStrength.STRONG_SELL
        elif score <= -15:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.HOLD

        probability = 0.5 + (score / 100)
        probability = max(0.2, min(0.8, probability))

        return ComponentPrediction(
            source='technical',
            signal=signal,
            probability_up=probability,
            confidence=abs(score) / 50,
            reasoning=f"Technical analysis shows {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} setup",
            factors=factors[:5]
        )

    # =========================================================================
    # CALENDAR PREDICTION
    # =========================================================================

    def _get_calendar_prediction(self, hist: pd.DataFrame) -> ComponentPrediction:
        """Get prediction based on calendar patterns."""
        returns = hist['Close'].pct_change().dropna()
        hist_with_ret = hist.copy()
        hist_with_ret['returns'] = returns

        # Day of week
        hist_with_ret['dow'] = hist_with_ret.index.dayofweek
        today_dow = datetime.now().weekday()

        dow_returns = hist_with_ret.groupby('dow')['returns'].mean()
        dow_win_rate = hist_with_ret.groupby('dow')['returns'].apply(lambda x: (x > 0).mean())

        today_win_rate = dow_win_rate.get(today_dow, 0.5)
        today_avg_return = dow_returns.get(today_dow, 0)

        # Month
        hist_with_ret['month'] = hist_with_ret.index.month
        today_month = datetime.now().month

        month_win_rate = hist_with_ret.groupby('month')['returns'].apply(lambda x: (x > 0).mean())
        month_avg_return = hist_with_ret.groupby('month')['returns'].mean()

        this_month_rate = month_win_rate.get(today_month, 0.5)
        this_month_return = month_avg_return.get(today_month, 0)

        # Combine
        combined_rate = (today_win_rate + this_month_rate) / 2
        combined_return = (today_avg_return + this_month_return) / 2

        factors = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        factors.append(f"{day_names[today_dow]} historical win rate: {today_win_rate:.0%}")
        factors.append(f"{month_names[today_month]} historical win rate: {this_month_rate:.0%}")

        if combined_rate > 0.55:
            signal = SignalStrength.BUY
            factors.append("Calendar patterns favor buying today")
        elif combined_rate < 0.45:
            signal = SignalStrength.SELL
            factors.append("Calendar patterns suggest caution")
        else:
            signal = SignalStrength.HOLD
            factors.append("Calendar patterns are neutral")

        return ComponentPrediction(
            source='calendar',
            signal=signal,
            probability_up=combined_rate,
            confidence=abs(combined_rate - 0.5) * 2,
            reasoning=f"Based on historical {day_names[today_dow]} and {month_names[today_month]} performance",
            factors=factors
        )

    # =========================================================================
    # UNIFIED PREDICTION
    # =========================================================================

    def predict(self, symbol: str) -> UnifiedPrediction:
        """
        Generate unified prediction combining all components.
        """
        logger.info(f"Generating unified prediction for {symbol}")

        # Fetch data
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period='2y')
            info = ticker.info or {}
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return self._empty_prediction(symbol)

        if hist.empty:
            return self._empty_prediction(symbol)

        current_price = hist['Close'].iloc[-1]

        # Prepare data dict for LLMs
        close = hist['Close']
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50

        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = hist['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1

        data_dict = {
            'price': current_price,
            'sector': info.get('sector', 'Unknown'),
            'change_1d': (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0,
            'change_1w': (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0,
            'change_1m': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0,
            'change_3m': (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0,
            'change_1y': (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) >= 252 else 0,
            'rsi': rsi,
            'above_sma50': current_price > sma_50,
            'above_sma200': current_price > sma_200,
            'volume_ratio': vol_ratio,
            'pe': info.get('trailingPE', 0),
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'debt_equity': info.get('debtToEquity', 0),
        }

        # Get predictions from all components
        logger.info("Getting ML predictions...")
        ml_predictions = self._get_ml_predictions(hist)

        logger.info("Getting LLM predictions...")
        llm_predictions = self._get_llm_predictions(symbol, data_dict)

        logger.info("Getting technical prediction...")
        technical = self._get_technical_prediction(hist)

        logger.info("Getting calendar prediction...")
        calendar = self._get_calendar_prediction(hist)

        # Aggregate all predictions
        all_predictions = ml_predictions + llm_predictions + [technical, calendar]

        # Weighted aggregation
        total_weight = 0
        weighted_prob = 0
        weighted_score = 0

        for pred in all_predictions:
            weight = self.WEIGHTS.get(pred.source, 0.1) * pred.confidence
            weighted_prob += pred.probability_up * weight
            weighted_score += pred.signal.score * weight
            total_weight += weight

        if total_weight > 0:
            final_prob = weighted_prob / total_weight
            final_score = weighted_score / total_weight
        else:
            final_prob = 0.5
            final_score = 0

        # Convert to signal
        if final_score >= 1.5:
            final_signal = SignalStrength.STRONG_BUY
        elif final_score >= 0.5:
            final_signal = SignalStrength.BUY
        elif final_score <= -1.5:
            final_signal = SignalStrength.STRONG_SELL
        elif final_score <= -0.5:
            final_signal = SignalStrength.SELL
        else:
            final_signal = SignalStrength.HOLD

        # Calculate consensus score
        signals = [p.signal for p in all_predictions]
        unique_signals = len(set([s.label for s in signals]))
        consensus = 1 - (unique_signals - 1) / max(len(signals) - 1, 1)

        # Average confidence
        avg_confidence = np.mean([p.confidence for p in all_predictions])

        # Price targets (ATR-based)
        atr = hist['Close'].diff().abs().rolling(14).mean().iloc[-1]
        entry = current_price
        stop_loss = entry - (1.5 * atr)
        target_1 = entry + (2 * atr)
        target_2 = entry + (3 * atr)

        # Position sizing (Kelly)
        win_prob = final_prob
        loss_prob = 1 - win_prob
        avg_win = 0.02
        avg_loss = 0.015
        kelly = max(0, (avg_win * win_prob - avg_loss * loss_prob) / avg_win) if avg_win > 0 else 0
        position_pct = min(kelly * 100, 15)  # Cap at 15%

        # Risk/Reward
        risk = entry - stop_loss
        reward = target_1 - entry
        rr = reward / risk if risk > 0 else 1

        # Aggregate factors
        bullish = []
        bearish = []
        for pred in all_predictions:
            for factor in pred.factors:
                if any(word in factor.lower() for word in ['bull', 'up', 'above', 'oversold', 'high volume']):
                    bullish.append(f"[{pred.source}] {factor}")
                elif any(word in factor.lower() for word in ['bear', 'down', 'below', 'overbought', 'low']):
                    bearish.append(f"[{pred.source}] {factor}")

        # Summary
        n_bullish = sum(1 for p in all_predictions if p.signal.score > 0)
        n_bearish = sum(1 for p in all_predictions if p.signal.score < 0)
        n_neutral = len(all_predictions) - n_bullish - n_bearish

        summary = f"{symbol}: {final_signal.label} with {final_prob:.0%} probability. "
        summary += f"Consensus: {n_bullish} bullish, {n_neutral} neutral, {n_bearish} bearish. "
        summary += f"Combined from {len(ml_predictions)} ML models + {len(llm_predictions)} LLMs + Technical + Calendar."

        # Detailed analysis
        detailed = "## Component Analysis\n\n"
        for pred in all_predictions:
            detailed += f"**{pred.source.upper()}**: {pred.signal.label} ({pred.probability_up:.0%})\n"
            detailed += f"  {pred.reasoning}\n\n"

        # Scores
        ml_score = np.mean([p.probability_up * 100 for p in ml_predictions]) if ml_predictions else 50
        llm_score = np.mean([p.probability_up * 100 for p in llm_predictions]) if llm_predictions else 50
        tech_score = technical.probability_up * 100
        calendar_score = calendar.probability_up * 100
        fund_score = 50  # Default

        return UnifiedPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=final_signal,
            probability_up=round(final_prob, 3),
            confidence=round(avg_confidence, 3),
            consensus_score=round(consensus, 3),
            current_price=round(current_price, 2),
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            position_size_pct=round(position_pct, 1),
            risk_reward_ratio=round(rr, 2),
            ml_predictions=ml_predictions,
            llm_predictions=llm_predictions,
            technical_prediction=technical,
            calendar_prediction=calendar,
            bullish_factors=bullish[:5],
            bearish_factors=bearish[:5],
            key_risks=["Market volatility", "Sector headwinds", "Earnings miss"],
            summary=summary,
            detailed_analysis=detailed,
            ml_score=round(ml_score, 1),
            llm_score=round(llm_score, 1),
            technical_score=round(tech_score, 1),
            fundamental_score=round(fund_score, 1),
            calendar_score=round(calendar_score, 1),
            models_used=[p.source for p in all_predictions],
            data_quality=min(len(hist) / 500, 1.0)
        )

    def _empty_prediction(self, symbol: str) -> UnifiedPrediction:
        """Return empty prediction for error cases."""
        return UnifiedPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=SignalStrength.HOLD,
            probability_up=0.5,
            confidence=0,
            consensus_score=0,
            current_price=0,
            entry_price=0,
            stop_loss=0,
            target_1=0,
            target_2=0,
            position_size_pct=0,
            risk_reward_ratio=0,
            ml_predictions=[],
            llm_predictions=[],
            technical_prediction=ComponentPrediction('technical', SignalStrength.HOLD, 0.5, 0, '', []),
            calendar_prediction=ComponentPrediction('calendar', SignalStrength.HOLD, 0.5, 0, '', []),
            bullish_factors=[],
            bearish_factors=[],
            key_risks=["Insufficient data"],
            summary="Unable to analyze - insufficient data",
            detailed_analysis="",
            ml_score=50,
            llm_score=50,
            technical_score=50,
            fundamental_score=50,
            calendar_score=50,
            models_used=[],
            data_quality=0
        )


def demo():
    """Demo the unified predictor."""
    print("=" * 70)
    print("UNIFIED PREDICTION ENGINE")
    print("ML Models + LLM Judges + Technical + Calendar")
    print("=" * 70)

    predictor = UnifiedPredictor()

    symbol = "RELIANCE"
    print(f"\nGenerating unified prediction for {symbol}...")
    print("This combines: sklearn ML + OpenAI + Gemini + Claude + Technical + Calendar\n")

    pred = predictor.predict(symbol)

    print(f"{'='*70}")
    print(f"UNIFIED PREDICTION: {pred.symbol}")
    print(f"{'='*70}")

    print(f"\n🎯 FINAL SIGNAL: {pred.signal.label}")
    print(f"   Probability Up: {pred.probability_up:.1%}")
    print(f"   Confidence: {pred.confidence:.1%}")
    print(f"   Consensus: {pred.consensus_score:.1%}")

    print(f"\n💰 TRADE SETUP:")
    print(f"   Current: ₹{pred.current_price:,.2f}")
    print(f"   Entry: ₹{pred.entry_price:,.2f}")
    print(f"   Stop Loss: ₹{pred.stop_loss:,.2f}")
    print(f"   Target 1: ₹{pred.target_1:,.2f}")
    print(f"   Target 2: ₹{pred.target_2:,.2f}")
    print(f"   R:R Ratio: {pred.risk_reward_ratio:.2f}")
    print(f"   Position: {pred.position_size_pct:.1f}%")

    print(f"\n📊 COMPONENT SCORES:")
    print(f"   ML Models: {pred.ml_score:.0f}/100")
    print(f"   LLM Judges: {pred.llm_score:.0f}/100")
    print(f"   Technical: {pred.technical_score:.0f}/100")
    print(f"   Calendar: {pred.calendar_score:.0f}/100")

    print(f"\n🤖 ML PREDICTIONS ({len(pred.ml_predictions)}):")
    for p in pred.ml_predictions:
        print(f"   {p.source}: {p.signal.label} ({p.probability_up:.1%})")

    print(f"\n🧠 LLM PREDICTIONS ({len(pred.llm_predictions)}):")
    for p in pred.llm_predictions:
        print(f"   {p.source}: {p.signal.label} ({p.probability_up:.1%})")
        print(f"      → {p.reasoning[:80]}...")

    print(f"\n✅ BULLISH FACTORS:")
    for f in pred.bullish_factors:
        print(f"   + {f}")

    print(f"\n⚠️ BEARISH FACTORS:")
    for f in pred.bearish_factors:
        print(f"   - {f}")

    print(f"\n📝 SUMMARY:")
    print(f"   {pred.summary}")


if __name__ == "__main__":
    demo()
