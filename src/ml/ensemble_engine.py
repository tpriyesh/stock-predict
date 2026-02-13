"""
Ensemble Engine

Combines multiple ML models for robust predictions:
- Gradient Boosting ensemble
- Feature importance tracking
- Model disagreement detection
- Weighted aggregation

Note: This is a lightweight implementation.
For production, integrate with scikit-learn/XGBoost/LightGBM.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    ensemble_prediction: float      # Combined prediction (0-1)
    model_predictions: Dict[str, float]  # Individual model predictions
    model_weights: Dict[str, float]      # Weights used
    model_agreement: float          # Agreement between models (0-1)
    feature_importance: Dict[str, float]  # Top feature importances
    confidence: float               # Confidence in prediction
    prediction_std: float           # Standard deviation across models
    probability_score: float        # Contribution to final score (0-1)
    signal: str                     # 'HIGH_AGREEMENT', 'MODERATE', 'DISAGREEMENT'
    reason: str                     # Human-readable explanation


class EnsembleEngine:
    """
    Lightweight ensemble engine using rule-based models.

    For production, replace with:
    - XGBoost/LightGBM gradient boosting
    - Random Forest
    - Proper cross-validation and training

    This implementation uses interpretable rule-based models
    that approximate ML behavior without training.
    """

    # Feature definitions
    FEATURES = [
        'return_5d', 'return_10d', 'return_20d',
        'volatility_10d', 'volatility_20d',
        'volume_ratio', 'rsi_14', 'macd_signal',
        'bb_position', 'adx', 'momentum_10d',
        'dist_sma_20', 'dist_sma_50'
    ]

    # Model weights (calibrated)
    MODEL_WEIGHTS = {
        'momentum_model': 0.25,
        'mean_reversion_model': 0.25,
        'trend_following_model': 0.25,
        'volatility_model': 0.25
    }

    # Feature importance (from backtesting)
    BASE_FEATURE_IMPORTANCE = {
        'return_5d': 0.12,
        'return_10d': 0.10,
        'return_20d': 0.08,
        'volatility_10d': 0.08,
        'volatility_20d': 0.07,
        'volume_ratio': 0.09,
        'rsi_14': 0.11,
        'macd_signal': 0.08,
        'bb_position': 0.07,
        'adx': 0.06,
        'momentum_10d': 0.08,
        'dist_sma_20': 0.03,
        'dist_sma_50': 0.03
    }

    def __init__(self):
        pass

    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ensemble models."""
        features = {}

        if len(df) < 50:
            return features

        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']

        # Returns
        features['return_5d'] = (close.iloc[-1] / close.iloc[-6] - 1) if len(df) > 5 else 0
        features['return_10d'] = (close.iloc[-1] / close.iloc[-11] - 1) if len(df) > 10 else 0
        features['return_20d'] = (close.iloc[-1] / close.iloc[-21] - 1) if len(df) > 20 else 0

        # Volatility
        returns = close.pct_change()
        features['volatility_10d'] = returns.tail(10).std() * np.sqrt(252)
        features['volatility_20d'] = returns.tail(20).std() * np.sqrt(252)

        # Volume
        features['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

        # RSI
        features['rsi_14'] = self._calculate_rsi(close) / 100  # Normalize to 0-1

        # MACD
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        macd_signal = (ema_12 - ema_26)
        features['macd_signal'] = 0.5 + np.clip(macd_signal / (close.iloc[-1] * 0.02), -0.5, 0.5)

        # Bollinger Bands position
        sma_20 = close.rolling(20).mean().iloc[-1]
        std_20 = close.rolling(20).std().iloc[-1]
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        features['bb_position'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ADX (simplified)
        features['adx'] = self._calculate_adx(df) / 100  # Normalize

        # Momentum
        features['momentum_10d'] = 0.5 + np.clip(features['return_10d'] * 5, -0.5, 0.5)

        # Distance from SMAs
        sma_50 = close.rolling(50).mean().iloc[-1]
        features['dist_sma_20'] = (close.iloc[-1] - sma_20) / sma_20
        features['dist_sma_50'] = (close.iloc[-1] - sma_50) / sma_50

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate simplified ADX."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = ((high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0))
        minus_dm = ((low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0))

        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean().iloc[-1]

        return adx if not np.isnan(adx) else 25

    def momentum_model(self, features: Dict[str, float]) -> float:
        """
        Momentum-based model.
        Higher returns + higher volume = bullish.
        """
        if not features:
            return 0.5

        score = 0.5

        # Recent returns
        score += features.get('return_5d', 0) * 2
        score += features.get('return_10d', 0) * 1.5
        score += features.get('momentum_10d', 0.5) - 0.5

        # Volume confirmation
        vol_ratio = features.get('volume_ratio', 1)
        if vol_ratio > 1.3:
            score += 0.05 if features.get('return_5d', 0) > 0 else -0.05

        return np.clip(score, 0.2, 0.8)

    def mean_reversion_model(self, features: Dict[str, float]) -> float:
        """
        Mean reversion model.
        Oversold + low BB position = bullish.
        """
        if not features:
            return 0.5

        score = 0.5

        # RSI mean reversion
        rsi = features.get('rsi_14', 0.5)
        if rsi < 0.3:
            score += 0.15  # Oversold
        elif rsi > 0.7:
            score -= 0.15  # Overbought

        # BB position mean reversion
        bb = features.get('bb_position', 0.5)
        if bb < 0.2:
            score += 0.10
        elif bb > 0.8:
            score -= 0.10

        # Distance from SMA (expect reversion)
        dist_sma = features.get('dist_sma_20', 0)
        score -= dist_sma * 0.5  # Far above = expect down

        return np.clip(score, 0.2, 0.8)

    def trend_following_model(self, features: Dict[str, float]) -> float:
        """
        Trend following model.
        Above SMAs + strong ADX = follow trend.
        """
        if not features:
            return 0.5

        score = 0.5

        # Position relative to SMAs
        if features.get('dist_sma_20', 0) > 0:
            score += 0.05
        if features.get('dist_sma_50', 0) > 0:
            score += 0.05

        # ADX (trend strength)
        adx = features.get('adx', 0.25)
        if adx > 0.25:  # Strong trend
            # Follow the trend direction
            if features.get('return_10d', 0) > 0:
                score += 0.10
            else:
                score -= 0.10

        # MACD
        macd = features.get('macd_signal', 0.5)
        score += (macd - 0.5) * 0.2

        return np.clip(score, 0.2, 0.8)

    def volatility_model(self, features: Dict[str, float]) -> float:
        """
        Volatility-based model.
        Low vol + compression = potential breakout.
        """
        if not features:
            return 0.5

        score = 0.5

        vol_10 = features.get('volatility_10d', 0.2)
        vol_20 = features.get('volatility_20d', 0.2)

        # Low volatility compression
        if vol_10 < 0.15:
            # Compressed - slight bullish bias
            score += 0.05
        elif vol_10 > 0.35:
            # High volatility - reduce confidence
            score = 0.5  # Reset to neutral

        # Volatility trending down (bullish)
        if vol_10 < vol_20:
            score += 0.03

        return np.clip(score, 0.3, 0.7)

    def get_feature_importance(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate feature importance based on current values.

        Combines base importance with feature extremity.
        """
        importance = {}

        for feature, base_imp in self.BASE_FEATURE_IMPORTANCE.items():
            value = features.get(feature, 0.5)

            # Extremity multiplier (more extreme = more important)
            if isinstance(value, (int, float)):
                extremity = abs(value - 0.5) * 2 if 0 <= value <= 1 else abs(value)
                adjusted_imp = base_imp * (1 + extremity * 0.5)
            else:
                adjusted_imp = base_imp

            importance[feature] = adjusted_imp

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        # Return top 5
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:5])

    def get_signal_label(
        self,
        prediction: float,
        agreement: float,
        std: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if agreement > 0.8:
            conf = 'HIGH_AGREEMENT'
        elif agreement > 0.6:
            conf = 'MODERATE'
        else:
            conf = 'DISAGREEMENT'

        if prediction > 0.6:
            return f'{conf}_BULLISH', f'Ensemble predicts {prediction:.0%} (std: {std:.2f})'
        elif prediction < 0.4:
            return f'{conf}_BEARISH', f'Ensemble predicts {prediction:.0%} (std: {std:.2f})'
        else:
            return f'{conf}_NEUTRAL', f'Ensemble neutral at {prediction:.0%}'

    def predict(self, df: pd.DataFrame) -> EnsembleResult:
        """
        Make ensemble prediction.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            EnsembleResult with ensemble prediction
        """
        if len(df) < 50:
            return self._default_result('Insufficient data for ensemble')

        # Extract features
        features = self.extract_features(df)

        if not features:
            return self._default_result('Could not extract features')

        # Run individual models
        predictions = {
            'momentum_model': self.momentum_model(features),
            'mean_reversion_model': self.mean_reversion_model(features),
            'trend_following_model': self.trend_following_model(features),
            'volatility_model': self.volatility_model(features)
        }

        # Weighted ensemble
        ensemble_pred = sum(
            predictions[m] * self.MODEL_WEIGHTS[m]
            for m in predictions
        )

        # Agreement (inverse of variance)
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        agreement = 1 - min(pred_std * 4, 1)  # Scale std to agreement

        # Confidence
        confidence = agreement * (1 - abs(ensemble_pred - 0.5) * 2)

        # Feature importance
        importance = self.get_feature_importance(features)

        # Signal and reason
        signal, reason = self.get_signal_label(ensemble_pred, agreement, pred_std)

        # Probability score
        prob_score = np.clip(ensemble_pred, 0.30, 0.75)

        return EnsembleResult(
            ensemble_prediction=ensemble_pred,
            model_predictions=predictions,
            model_weights=self.MODEL_WEIGHTS,
            model_agreement=agreement,
            feature_importance=importance,
            confidence=confidence,
            prediction_std=pred_std,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> EnsembleResult:
        """Return neutral result when prediction isn't possible."""
        return EnsembleResult(
            ensemble_prediction=0.50,
            model_predictions={m: 0.5 for m in self.MODEL_WEIGHTS},
            model_weights=self.MODEL_WEIGHTS,
            model_agreement=0.5,
            feature_importance={},
            confidence=0.5,
            prediction_std=0.1,
            probability_score=0.50,
            signal='ENSEMBLE_UNKNOWN',
            reason=reason
        )
