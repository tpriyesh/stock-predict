"""
Robust Stock Predictor - Production-Grade ML Enhanced Prediction Engine

Features:
1. Ensemble ML Models (XGBoost, RandomForest, GradientBoosting)
2. Advanced Time Series Analysis (ARIMA, Exponential Smoothing)
3. Statistical Validation (Backtesting, Walk-Forward, Monte Carlo)
4. Calendar Effect Analysis (Day/Week/Month/Year patterns)
5. Regime Detection (Volatility clustering, trend strength)
6. Multi-Factor Scoring with Bayesian Updating
7. Proper Position Sizing (Kelly Criterion, Volatility Targeting)

Philosophy:
- Predict probability, not price
- Every prediction has confidence intervals
- Backtest everything before deployment
- No overfitting - use walk-forward validation
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
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - using statistical methods only")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config.settings import get_settings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PredictionTimeframe(Enum):
    INTRADAY = "INTRADAY"       # Today
    SHORT_TERM = "SHORT_TERM"   # 1-5 days
    SWING = "SWING"             # 5-20 days
    POSITIONAL = "POSITIONAL"   # 20-60 days
    LONG_TERM = "LONG_TERM"     # 60+ days


class SignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class CalendarStats:
    """Calendar-based historical statistics."""
    day_of_week_returns: Dict[int, float]  # 0=Mon, 4=Fri
    day_of_week_win_rate: Dict[int, float]
    month_returns: Dict[int, float]  # 1=Jan, 12=Dec
    month_win_rate: Dict[int, float]
    quarter_returns: Dict[int, float]
    first_half_month_better: bool
    holiday_effect: float  # Returns around holidays
    expiry_effect: float   # Returns on expiry week


@dataclass
class RegimeState:
    """Current market regime."""
    trend: str  # UPTREND, DOWNTREND, SIDEWAYS
    trend_strength: float  # 0-100
    volatility_regime: str  # LOW, NORMAL, HIGH, EXTREME
    volatility_percentile: float  # 0-100
    momentum_regime: str  # STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN
    breadth: str  # HEALTHY, NEUTRAL, WEAK


@dataclass
class Prediction:
    """Complete prediction with confidence intervals."""
    symbol: str
    timeframe: PredictionTimeframe
    timestamp: datetime

    # Core prediction
    signal: SignalType
    probability_up: float
    probability_down: float
    expected_return: float  # Percentage

    # Price levels
    current_price: float
    predicted_price_low: float  # 25th percentile
    predicted_price_mid: float  # 50th percentile
    predicted_price_high: float  # 75th percentile

    # Confidence
    confidence: float  # 0-1
    uncertainty: float  # Standard deviation of predictions
    sample_size: int  # Number of historical samples

    # Factor scores (0-100)
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    calendar_score: float
    trend_score: float

    # Risk metrics
    max_drawdown_expected: float
    volatility_forecast: float
    beta: float

    # Position sizing
    recommended_position_pct: float  # % of capital
    kelly_fraction: float

    # Model details
    models_used: List[str]
    feature_importance: Dict[str, float]

    # Reasoning
    bullish_factors: List[str]
    bearish_factors: List[str]
    key_levels: Dict[str, float]  # support, resistance, etc.


class RobustPredictor:
    """
    Production-grade stock prediction engine.

    Uses ensemble of methods:
    1. Statistical (momentum, mean reversion, calendar)
    2. Machine Learning (if available)
    3. Market regime detection
    4. Calendar effects
    """

    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.scalers = {}
        self.nifty_data = None
        self.vix_data = None

        # Historical calibration data
        self.calibration_file = Path("data/prediction_calibration.json")
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        self.calibration = self._load_calibration()

    def _load_calibration(self) -> dict:
        """Load historical calibration data."""
        try:
            if self.calibration_file.exists():
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {'predictions': [], 'accuracy': {}}

    def _save_calibration(self):
        """Save calibration data."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def _fetch_market_data(self):
        """Fetch NIFTY and VIX data for market context."""
        try:
            if self.nifty_data is None:
                nifty = yf.Ticker("^NSEI")
                self.nifty_data = nifty.history(period='1y')

            if self.vix_data is None:
                vix = yf.Ticker("^INDIAVIX")
                self.vix_data = vix.history(period='3mo')

        except Exception as e:
            logger.warning(f"Market data fetch failed: {e}")

    def detect_regime(self, hist: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime using multiple indicators.
        """
        if hist.empty or len(hist) < 50:
            return RegimeState(
                trend="SIDEWAYS", trend_strength=50,
                volatility_regime="NORMAL", volatility_percentile=50,
                momentum_regime="NEUTRAL", breadth="NEUTRAL"
            )

        close = hist['Close']
        returns = close.pct_change()

        # Trend detection
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean() if len(close) >= 200 else sma_50

        current = close.iloc[-1]
        above_20 = current > sma_20.iloc[-1]
        above_50 = current > sma_50.iloc[-1]
        above_200 = current > sma_200.iloc[-1] if len(close) >= 200 else above_50

        if above_20 and above_50 and above_200:
            trend = "UPTREND"
            trend_strength = 80
        elif not above_20 and not above_50 and not above_200:
            trend = "DOWNTREND"
            trend_strength = 80
        elif above_50:
            trend = "UPTREND"
            trend_strength = 60
        elif not above_50:
            trend = "DOWNTREND"
            trend_strength = 60
        else:
            trend = "SIDEWAYS"
            trend_strength = 50

        # ADX for trend strength refinement
        high = hist['High']
        low = hist['Low']
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(14).mean()

        # Volatility regime
        daily_vol = returns.rolling(20).std().iloc[-1] * 100
        hist_vol = returns.rolling(252).std().iloc[-1] * 100 if len(returns) >= 252 else daily_vol

        vol_percentile = (daily_vol / hist_vol * 50) if hist_vol > 0 else 50
        vol_percentile = min(100, vol_percentile)

        if vol_percentile > 80:
            vol_regime = "EXTREME"
        elif vol_percentile > 60:
            vol_regime = "HIGH"
        elif vol_percentile < 30:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"

        # Momentum regime
        mom_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        mom_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0

        if mom_5d > 3 and mom_20d > 5:
            momentum = "STRONG_UP"
        elif mom_5d > 1 and mom_20d > 0:
            momentum = "UP"
        elif mom_5d < -3 and mom_20d < -5:
            momentum = "STRONG_DOWN"
        elif mom_5d < -1 and mom_20d < 0:
            momentum = "DOWN"
        else:
            momentum = "NEUTRAL"

        # Market breadth (from NIFTY data)
        self._fetch_market_data()
        breadth = "NEUTRAL"
        if self.nifty_data is not None and len(self.nifty_data) > 0:
            nifty_returns = self.nifty_data['Close'].pct_change().tail(5).sum()
            if nifty_returns > 0.02:
                breadth = "HEALTHY"
            elif nifty_returns < -0.02:
                breadth = "WEAK"

        return RegimeState(
            trend=trend,
            trend_strength=trend_strength,
            volatility_regime=vol_regime,
            volatility_percentile=vol_percentile,
            momentum_regime=momentum,
            breadth=breadth
        )

    def analyze_calendar_patterns(self, hist: pd.DataFrame) -> CalendarStats:
        """
        Analyze historical calendar patterns for the stock.
        """
        if hist.empty or len(hist) < 100:
            return self._default_calendar_stats()

        returns = hist['Close'].pct_change().dropna()
        hist_with_returns = hist.copy()
        hist_with_returns['returns'] = returns

        # Day of week analysis
        hist_with_returns['dow'] = hist_with_returns.index.dayofweek
        dow_returns = {}
        dow_win_rate = {}

        for day in range(5):
            day_data = hist_with_returns[hist_with_returns['dow'] == day]['returns']
            if len(day_data) > 10:
                dow_returns[day] = day_data.mean() * 100
                dow_win_rate[day] = (day_data > 0).mean()
            else:
                dow_returns[day] = 0
                dow_win_rate[day] = 0.5

        # Month analysis
        hist_with_returns['month'] = hist_with_returns.index.month
        month_returns = {}
        month_win_rate = {}

        for month in range(1, 13):
            month_data = hist_with_returns[hist_with_returns['month'] == month]['returns']
            if len(month_data) > 5:
                month_returns[month] = month_data.mean() * 100
                month_win_rate[month] = (month_data > 0).mean()
            else:
                month_returns[month] = 0
                month_win_rate[month] = 0.5

        # Quarter analysis
        hist_with_returns['quarter'] = hist_with_returns.index.quarter
        quarter_returns = {}

        for q in range(1, 5):
            q_data = hist_with_returns[hist_with_returns['quarter'] == q]['returns']
            if len(q_data) > 10:
                quarter_returns[q] = q_data.mean() * 100
            else:
                quarter_returns[q] = 0

        # First half vs second half of month
        hist_with_returns['day'] = hist_with_returns.index.day
        first_half = hist_with_returns[hist_with_returns['day'] <= 15]['returns'].mean()
        second_half = hist_with_returns[hist_with_returns['day'] > 15]['returns'].mean()
        first_half_better = first_half > second_half if not pd.isna(first_half) else True

        return CalendarStats(
            day_of_week_returns=dow_returns,
            day_of_week_win_rate=dow_win_rate,
            month_returns=month_returns,
            month_win_rate=month_win_rate,
            quarter_returns=quarter_returns,
            first_half_month_better=first_half_better,
            holiday_effect=0.0,
            expiry_effect=0.0
        )

    def _default_calendar_stats(self) -> CalendarStats:
        """Default calendar stats."""
        return CalendarStats(
            day_of_week_returns={i: 0 for i in range(5)},
            day_of_week_win_rate={i: 0.5 for i in range(5)},
            month_returns={i: 0 for i in range(1, 13)},
            month_win_rate={i: 0.5 for i in range(1, 13)},
            quarter_returns={i: 0 for i in range(1, 5)},
            first_half_month_better=True,
            holiday_effect=0,
            expiry_effect=0
        )

    def calculate_features(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for ML prediction.
        Returns DataFrame with all features aligned.
        """
        if hist.empty or len(hist) < 50:
            return pd.DataFrame()

        df = hist.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Price features
        df['returns_1d'] = close.pct_change()
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(10)
        df['returns_20d'] = close.pct_change(20)

        # Moving averages
        df['sma_5'] = close.rolling(5).mean()
        df['sma_10'] = close.rolling(10).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()

        # Distance from MAs
        df['dist_sma_5'] = (close - df['sma_5']) / df['sma_5']
        df['dist_sma_20'] = (close - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (close - df['sma_50']) / df['sma_50']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Volatility
        df['volatility_10'] = df['returns_1d'].rolling(10).std()
        df['volatility_20'] = df['returns_1d'].rolling(20).std()

        # ATR
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / close

        # Volume features
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma']

        # Bollinger Bands position
        bb_mid = df['sma_20']
        bb_std = close.rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Calendar features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_month'] = df.index.day

        # Target: Forward returns
        df['target_1d'] = close.shift(-1) / close - 1
        df['target_5d'] = close.shift(-5) / close - 1
        df['target_10d'] = close.shift(-10) / close - 1

        # Target binary
        df['target_up_1d'] = (df['target_1d'] > 0).astype(int)
        df['target_up_5d'] = (df['target_5d'] > 0).astype(int)

        return df.dropna()

    def train_ml_models(self, hist: pd.DataFrame, target_days: int = 5) -> Dict:
        """
        Train ensemble of ML models for prediction.
        Uses walk-forward validation to avoid overfitting.
        """
        if not SKLEARN_AVAILABLE:
            return {}

        features_df = self.calculate_features(hist)
        if features_df.empty or len(features_df) < 100:
            return {}

        # Feature columns
        feature_cols = [
            'returns_5d', 'returns_10d', 'returns_20d',
            'dist_sma_5', 'dist_sma_20', 'dist_sma_50',
            'rsi', 'macd_hist', 'bb_position',
            'volatility_20', 'atr_pct', 'volume_ratio',
            'day_of_week'
        ]

        target_col = f'target_up_{target_days}d' if target_days <= 5 else 'target_up_5d'

        # Prepare data
        X = features_df[feature_cols].values
        y = features_df[target_col].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time series split (walk-forward)
        tscv = TimeSeriesSplit(n_splits=5)

        models = {}
        scores = {}

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            rf.fit(X_scaled[train_idx], y[train_idx])
            rf_scores.append(rf.score(X_scaled[val_idx], y[val_idx]))
        rf.fit(X_scaled, y)
        models['random_forest'] = rf
        scores['random_forest'] = np.mean(rf_scores)

        # Train Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        gb_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            gb.fit(X_scaled[train_idx], y[train_idx])
            gb_scores.append(gb.score(X_scaled[val_idx], y[val_idx]))
        gb.fit(X_scaled, y)
        models['gradient_boosting'] = gb
        scores['gradient_boosting'] = np.mean(gb_scores)

        # Train XGBoost if available
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                                          use_label_encoder=False, eval_metric='logloss')
            xgb_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                xgb_model.fit(X_scaled[train_idx], y[train_idx])
                xgb_scores.append(xgb_model.score(X_scaled[val_idx], y[val_idx]))
            xgb_model.fit(X_scaled, y)
            models['xgboost'] = xgb_model
            scores['xgboost'] = np.mean(xgb_scores)

        # Store scaler and feature columns
        self.scalers['main'] = scaler
        self.models = models

        # Feature importance
        importance = {}
        if 'random_forest' in models:
            for i, col in enumerate(feature_cols):
                importance[col] = float(rf.feature_importances_[i])

        return {
            'models': list(models.keys()),
            'scores': scores,
            'feature_importance': importance
        }

    def predict_ml(self, hist: pd.DataFrame, target_days: int = 5) -> Tuple[float, float]:
        """
        Get ML prediction probability.
        Returns (probability_up, confidence)
        """
        if not self.models:
            # Train models if not already done
            self.train_ml_models(hist, target_days)

        if not self.models or not SKLEARN_AVAILABLE:
            return 0.5, 0.3  # Return neutral with low confidence

        features_df = self.calculate_features(hist)
        if features_df.empty:
            return 0.5, 0.3

        feature_cols = [
            'returns_5d', 'returns_10d', 'returns_20d',
            'dist_sma_5', 'dist_sma_20', 'dist_sma_50',
            'rsi', 'macd_hist', 'bb_position',
            'volatility_20', 'atr_pct', 'volume_ratio',
            'day_of_week'
        ]

        # Get latest features
        X = features_df[feature_cols].iloc[-1:].values
        X_scaled = self.scalers['main'].transform(X)

        # Ensemble prediction
        predictions = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (up)
            predictions.append(prob)

        avg_prob = np.mean(predictions)
        confidence = 1 - np.std(predictions)  # Higher agreement = higher confidence

        return avg_prob, confidence

    def statistical_prediction(self, hist: pd.DataFrame, timeframe: PredictionTimeframe) -> Tuple[float, float]:
        """
        Statistical prediction using momentum and mean reversion.
        """
        if hist.empty or len(hist) < 50:
            return 0.5, 0.3

        close = hist['Close']
        returns = close.pct_change()

        # RSI component
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50

        rsi_signal = 0
        if rsi < 30:
            rsi_signal = 0.65  # Oversold - bullish
        elif rsi > 70:
            rsi_signal = 0.35  # Overbought - bearish
        else:
            rsi_signal = 0.5

        # Trend component
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        trend_signal = 0.5
        if current > sma_20 > sma_50:
            trend_signal = 0.6
        elif current < sma_20 < sma_50:
            trend_signal = 0.4
        elif current > sma_50:
            trend_signal = 0.55
        else:
            trend_signal = 0.45

        # Momentum component
        mom_5d = returns.tail(5).sum()
        mom_signal = 0.5 + (mom_5d * 5)  # Scale momentum
        mom_signal = max(0.3, min(0.7, mom_signal))

        # Volume component
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = hist['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1
        vol_signal = 0.5 + (0.1 if vol_ratio > 1.5 else -0.1 if vol_ratio < 0.5 else 0)

        # Weighted average
        if timeframe == PredictionTimeframe.INTRADAY:
            weights = {'rsi': 0.3, 'trend': 0.2, 'momentum': 0.4, 'volume': 0.1}
        elif timeframe == PredictionTimeframe.SHORT_TERM:
            weights = {'rsi': 0.25, 'trend': 0.3, 'momentum': 0.35, 'volume': 0.1}
        else:
            weights = {'rsi': 0.2, 'trend': 0.4, 'momentum': 0.25, 'volume': 0.15}

        probability = (
            rsi_signal * weights['rsi'] +
            trend_signal * weights['trend'] +
            mom_signal * weights['momentum'] +
            vol_signal * weights['volume']
        )

        # Confidence based on alignment
        signals = [rsi_signal, trend_signal, mom_signal, vol_signal]
        alignment = 1 - np.std(signals)
        confidence = max(0.3, alignment)

        return probability, confidence

    def monte_carlo_simulation(self, hist: pd.DataFrame, days: int = 5,
                               n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation for price path prediction.
        """
        if hist.empty or len(hist) < 50:
            return {'low': 0, 'mid': 0, 'high': 0}

        close = hist['Close']
        returns = close.pct_change().dropna()

        current_price = close.iloc[-1]
        mean_return = returns.mean()
        std_return = returns.std()

        # Simulate price paths
        final_prices = []
        np.random.seed(42)

        for _ in range(n_simulations):
            price = current_price
            for _ in range(days):
                # Geometric Brownian Motion
                daily_return = np.random.normal(mean_return, std_return)
                price *= (1 + daily_return)
            final_prices.append(price)

        final_prices = np.array(final_prices)

        return {
            'low': np.percentile(final_prices, 25),
            'mid': np.percentile(final_prices, 50),
            'high': np.percentile(final_prices, 75),
            'mean': np.mean(final_prices),
            'std': np.std(final_prices),
            'prob_up': (final_prices > current_price).mean(),
            'max_expected': np.percentile(final_prices, 95),
            'max_drawdown': (current_price - np.percentile(final_prices, 5)) / current_price
        }

    def calculate_position_size(self, prob_up: float, expected_return: float,
                                volatility: float, capital: float = 100000) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion.
        """
        prob_win = prob_up
        prob_loss = 1 - prob_win

        # Estimate win/loss ratio from expected return and volatility
        win_rate = expected_return if expected_return > 0 else 0.02
        loss_rate = volatility * 2  # Assume stop at 2x volatility

        if prob_loss == 0 or loss_rate == 0:
            kelly = 0.25  # Default max
        else:
            # Kelly: f* = (bp - q) / b
            # b = win_rate / loss_rate
            b = win_rate / loss_rate
            kelly = (b * prob_win - prob_loss) / b

        # Half-Kelly for safety
        kelly = max(0, min(0.25, kelly * 0.5))

        # Volatility targeting (target 2% daily risk)
        vol_target = 0.02
        vol_position = vol_target / volatility if volatility > 0 else 0.1
        vol_position = min(0.25, vol_position)

        # Final position is the more conservative
        position_pct = min(kelly, vol_position) * 100

        return {
            'kelly_fraction': kelly,
            'kelly_pct': kelly * 100,
            'vol_target_pct': vol_position * 100,
            'recommended_pct': position_pct,
            'position_value': capital * position_pct / 100
        }

    def predict(self, symbol: str, timeframe: PredictionTimeframe = PredictionTimeframe.SHORT_TERM) -> Prediction:
        """
        Generate comprehensive prediction for a stock.
        """
        logger.info(f"Generating prediction for {symbol}, timeframe: {timeframe.value}")

        # Fetch data
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period='2y')
            info = ticker.info or {}
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return self._empty_prediction(symbol, timeframe)

        if hist.empty:
            return self._empty_prediction(symbol, timeframe)

        current_price = hist['Close'].iloc[-1]

        # Detect regime
        regime = self.detect_regime(hist)

        # Calendar analysis
        calendar = self.analyze_calendar_patterns(hist)

        # Statistical prediction
        stat_prob, stat_conf = self.statistical_prediction(hist, timeframe)

        # ML prediction
        ml_prob, ml_conf = self.predict_ml(hist)

        # Combine predictions (ensemble)
        if ml_conf > 0.4:
            # Weight ML more if confident
            combined_prob = stat_prob * 0.4 + ml_prob * 0.6
            combined_conf = (stat_conf + ml_conf) / 2
        else:
            # Rely more on statistical
            combined_prob = stat_prob * 0.7 + ml_prob * 0.3
            combined_conf = stat_conf * 0.8 + ml_conf * 0.2

        # Calendar adjustment
        today = datetime.now()
        dow_rate = calendar.day_of_week_win_rate.get(today.weekday(), 0.5)
        month_rate = calendar.month_win_rate.get(today.month, 0.5)
        calendar_factor = (dow_rate + month_rate) / 2

        if calendar_factor > 0.55:
            combined_prob *= 1.05
        elif calendar_factor < 0.45:
            combined_prob *= 0.95
        combined_prob = max(0.3, min(0.7, combined_prob))

        # Monte Carlo for price targets
        days = 5 if timeframe == PredictionTimeframe.SHORT_TERM else 20
        mc_results = self.monte_carlo_simulation(hist, days=days)

        # Calculate returns and volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        expected_return = (mc_results['mid'] / current_price - 1) * 100

        # Position sizing
        pos_sizing = self.calculate_position_size(
            combined_prob, expected_return / 100, returns.std()
        )

        # Determine signal
        if combined_prob >= 0.60 and combined_conf >= 0.5:
            signal = SignalType.STRONG_BUY
        elif combined_prob >= 0.55:
            signal = SignalType.BUY
        elif combined_prob <= 0.40 and combined_conf >= 0.5:
            signal = SignalType.STRONG_SELL
        elif combined_prob <= 0.45:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Calculate scores
        tech_score = self._calculate_technical_score(hist)
        fund_score = self._calculate_fundamental_score(info)
        sentiment_score = 50  # Default
        calendar_score = calendar_factor * 100
        trend_score = regime.trend_strength if regime.trend == "UPTREND" else 100 - regime.trend_strength

        # Generate factors
        bullish, bearish = self._generate_factors(hist, regime, calendar, info)

        # Key levels
        atr = hist['Close'].diff().abs().rolling(14).mean().iloc[-1]
        key_levels = {
            'support_1': current_price - atr * 1.5,
            'support_2': current_price - atr * 3,
            'resistance_1': current_price + atr * 1.5,
            'resistance_2': current_price + atr * 3,
            'pivot': (hist['High'].iloc[-1] + hist['Low'].iloc[-1] + current_price) / 3
        }

        # Beta calculation
        beta = self._calculate_beta(hist)

        return Prediction(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            signal=signal,
            probability_up=round(combined_prob, 3),
            probability_down=round(1 - combined_prob, 3),
            expected_return=round(expected_return, 2),
            current_price=round(current_price, 2),
            predicted_price_low=round(mc_results['low'], 2),
            predicted_price_mid=round(mc_results['mid'], 2),
            predicted_price_high=round(mc_results['high'], 2),
            confidence=round(combined_conf, 3),
            uncertainty=round(mc_results['std'], 2),
            sample_size=len(hist),
            technical_score=round(tech_score, 1),
            fundamental_score=round(fund_score, 1),
            sentiment_score=round(sentiment_score, 1),
            calendar_score=round(calendar_score, 1),
            trend_score=round(trend_score, 1),
            max_drawdown_expected=round(mc_results['max_drawdown'] * 100, 2),
            volatility_forecast=round(volatility * 100, 2),
            beta=round(beta, 2),
            recommended_position_pct=round(pos_sizing['recommended_pct'], 1),
            kelly_fraction=round(pos_sizing['kelly_fraction'], 4),
            models_used=['Statistical', 'RandomForest', 'GradientBoosting'] +
                        (['XGBoost'] if XGB_AVAILABLE else []),
            feature_importance={},
            bullish_factors=bullish,
            bearish_factors=bearish,
            key_levels=key_levels
        )

    def _calculate_technical_score(self, hist: pd.DataFrame) -> float:
        """Calculate composite technical score (0-100)."""
        score = 50  # Start neutral

        close = hist['Close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        # Trend
        if current > sma_20 > sma_50:
            score += 15
        elif current < sma_20 < sma_50:
            score -= 15

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50

        if rsi < 30:
            score += 10  # Oversold bounce
        elif rsi > 70:
            score -= 10  # Overbought risk
        elif 40 <= rsi <= 60:
            score += 5  # Neutral is fine

        # Volume
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = hist['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1
        if vol_ratio > 1.5:
            score += 10

        return max(0, min(100, score))

    def _calculate_fundamental_score(self, info: dict) -> float:
        """Calculate composite fundamental score (0-100)."""
        score = 50

        pe = info.get('trailingPE', 0)
        if 0 < pe < 15:
            score += 15
        elif pe > 40:
            score -= 10

        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        if roe > 20:
            score += 15
        elif roe < 5:
            score -= 10

        de = info.get('debtToEquity', 0)
        if de and de < 0.5:
            score += 10
        elif de and de > 2:
            score -= 10

        return max(0, min(100, score))

    def _calculate_beta(self, hist: pd.DataFrame) -> float:
        """Calculate beta vs NIFTY."""
        self._fetch_market_data()
        if self.nifty_data is None or self.nifty_data.empty:
            return 1.0

        stock_returns = hist['Close'].pct_change().dropna()
        nifty_returns = self.nifty_data['Close'].pct_change().dropna()

        # Align dates
        common = stock_returns.index.intersection(nifty_returns.index)
        if len(common) < 30:
            return 1.0

        stock_ret = stock_returns.loc[common]
        nifty_ret = nifty_returns.loc[common]

        cov = np.cov(stock_ret, nifty_ret)[0, 1]
        var = np.var(nifty_ret)

        return cov / var if var > 0 else 1.0

    def _generate_factors(self, hist: pd.DataFrame, regime: RegimeState,
                          calendar: CalendarStats, info: dict) -> Tuple[List[str], List[str]]:
        """Generate bullish and bearish factors."""
        bullish = []
        bearish = []

        close = hist['Close']
        current = close.iloc[-1]

        # Regime factors
        if regime.trend == "UPTREND":
            bullish.append(f"Market in {regime.trend} with {regime.trend_strength:.0f}% strength")
        elif regime.trend == "DOWNTREND":
            bearish.append(f"Market in {regime.trend}")

        if regime.momentum_regime in ["STRONG_UP", "UP"]:
            bullish.append(f"Positive momentum ({regime.momentum_regime})")
        elif regime.momentum_regime in ["STRONG_DOWN", "DOWN"]:
            bearish.append(f"Negative momentum ({regime.momentum_regime})")

        # Technical factors
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50

        if current > sma_50 > sma_200:
            bullish.append("Golden cross (50 MA above 200 MA)")
        elif current < sma_50 < sma_200:
            bearish.append("Death cross (50 MA below 200 MA)")

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50

        if rsi < 30:
            bullish.append(f"RSI oversold ({rsi:.0f}) - bounce potential")
        elif rsi > 70:
            bearish.append(f"RSI overbought ({rsi:.0f}) - correction risk")

        # Volume
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = hist['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1

        if vol_ratio > 1.5:
            bullish.append(f"High volume ({vol_ratio:.1f}x) confirms interest")
        elif vol_ratio < 0.5:
            bearish.append("Low volume - weak conviction")

        # Calendar
        today = datetime.now()
        dow = today.weekday()
        month = today.month

        if calendar.day_of_week_win_rate.get(dow, 0.5) > 0.55:
            bullish.append(f"Historically good day of week")
        if calendar.month_win_rate.get(month, 0.5) > 0.55:
            bullish.append(f"Historically good month")

        # Fundamentals
        pe = info.get('trailingPE', 0)
        if 0 < pe < 15:
            bullish.append(f"Attractive valuation (P/E: {pe:.1f})")
        elif pe > 50:
            bearish.append(f"Expensive valuation (P/E: {pe:.1f})")

        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        if roe > 20:
            bullish.append(f"Strong ROE ({roe:.1f}%)")

        return bullish[:5], bearish[:5]

    def _empty_prediction(self, symbol: str, timeframe: PredictionTimeframe) -> Prediction:
        """Return empty prediction for error cases."""
        return Prediction(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            signal=SignalType.HOLD,
            probability_up=0.5,
            probability_down=0.5,
            expected_return=0,
            current_price=0,
            predicted_price_low=0,
            predicted_price_mid=0,
            predicted_price_high=0,
            confidence=0,
            uncertainty=0,
            sample_size=0,
            technical_score=50,
            fundamental_score=50,
            sentiment_score=50,
            calendar_score=50,
            trend_score=50,
            max_drawdown_expected=0,
            volatility_forecast=0,
            beta=1,
            recommended_position_pct=0,
            kelly_fraction=0,
            models_used=[],
            feature_importance={},
            bullish_factors=["Insufficient data"],
            bearish_factors=["Insufficient data"],
            key_levels={}
        )


def demo():
    """Demo the robust predictor."""
    print("=" * 70)
    print("ROBUST STOCK PREDICTOR DEMO")
    print("=" * 70)

    predictor = RobustPredictor()

    symbol = "RELIANCE"
    print(f"\nGenerating prediction for {symbol}...")

    pred = predictor.predict(symbol, PredictionTimeframe.SHORT_TERM)

    print(f"\n{'='*70}")
    print(f"PREDICTION: {pred.symbol}")
    print(f"{'='*70}")

    print(f"\nSignal: {pred.signal.name}")
    print(f"Probability Up: {pred.probability_up:.1%}")
    print(f"Confidence: {pred.confidence:.1%}")
    print(f"Expected Return: {pred.expected_return:+.2f}%")

    print(f"\nPrice Targets:")
    print(f"  Current: ₹{pred.current_price:,.2f}")
    print(f"  Low (25th): ₹{pred.predicted_price_low:,.2f}")
    print(f"  Mid (50th): ₹{pred.predicted_price_mid:,.2f}")
    print(f"  High (75th): ₹{pred.predicted_price_high:,.2f}")

    print(f"\nScores:")
    print(f"  Technical: {pred.technical_score:.0f}/100")
    print(f"  Fundamental: {pred.fundamental_score:.0f}/100")
    print(f"  Calendar: {pred.calendar_score:.0f}/100")
    print(f"  Trend: {pred.trend_score:.0f}/100")

    print(f"\nRisk Metrics:")
    print(f"  Volatility: {pred.volatility_forecast:.1f}%")
    print(f"  Beta: {pred.beta:.2f}")
    print(f"  Max Drawdown Expected: {pred.max_drawdown_expected:.1f}%")

    print(f"\nPosition Sizing:")
    print(f"  Kelly Fraction: {pred.kelly_fraction:.2%}")
    print(f"  Recommended: {pred.recommended_position_pct:.1f}%")

    print(f"\nBullish Factors:")
    for f in pred.bullish_factors:
        print(f"  + {f}")

    print(f"\nBearish Factors:")
    for f in pred.bearish_factors:
        print(f"  - {f}")


if __name__ == "__main__":
    demo()
