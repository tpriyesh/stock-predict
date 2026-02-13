"""
MetaLearnerStacker - ML-Based Ensemble Weight Learning

ADVANCED: Instead of fixed or accuracy-based weights, train a model
to learn optimal weights dynamically based on market conditions.

The meta-learner takes as input:
1. Predictions from all base models
2. Market regime features
3. Volatility state
4. Correlation between base predictions

And outputs the optimal probability/signal.

This is more sophisticated than simple weighted averaging because:
- Weights can vary based on market conditions
- Can detect when models disagree and adjust confidence
- Learns non-linear combinations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - MetaLearnerStacker limited")


@dataclass
class MetaFeatures:
    """Features for the meta-learner."""
    # Base model predictions
    ml_rf_prob: float = 0.5
    ml_gb_prob: float = 0.5
    ml_xgb_prob: float = 0.5
    llm_sentiment: float = 0.0
    technical_score: float = 50.0
    calendar_score: float = 50.0

    # Agreement/disagreement features
    prediction_std: float = 0.0  # Standard deviation of predictions
    prediction_range: float = 0.0  # Max - min prediction
    n_bullish: int = 0  # Count of bullish predictions
    n_bearish: int = 0  # Count of bearish predictions

    # Market regime features
    volatility_regime: int = 0  # 0=low, 1=normal, 2=high
    trend_strength: float = 0.0  # -1 to 1
    market_regime: int = 0  # 0=bear, 1=sideways, 2=bull

    # Additional context
    rsi_level: float = 50.0
    volume_ratio: float = 1.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.ml_rf_prob,
            self.ml_gb_prob,
            self.ml_xgb_prob,
            self.llm_sentiment,
            self.technical_score / 100,  # Normalize
            self.calendar_score / 100,
            self.prediction_std,
            self.prediction_range,
            self.n_bullish,
            self.n_bearish,
            self.volatility_regime,
            self.trend_strength,
            self.market_regime,
            self.rsi_level / 100,
            self.volume_ratio
        ]).reshape(1, -1)

    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names for interpretability."""
        return [
            'ml_rf_prob', 'ml_gb_prob', 'ml_xgb_prob',
            'llm_sentiment', 'technical_score', 'calendar_score',
            'prediction_std', 'prediction_range',
            'n_bullish', 'n_bearish',
            'volatility_regime', 'trend_strength', 'market_regime',
            'rsi_level', 'volume_ratio'
        ]


class MetaLearnerStacker:
    """
    Meta-learner that combines base model predictions optimally.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     BASE MODELS                              │
    │  RandomForest  XGBoost  GradBoost  LLM  Technical  Calendar │
    │      0.65       0.58     0.52     0.48    0.60      0.55    │
    └───────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │ META-FEATURES │
                    │ + predictions │
                    │ + agreement   │
                    │ + regime      │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │ META-LEARNER  │
                    │ (Logistic Reg)│
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │ FINAL PROB    │
                    │    0.612      │
                    └───────────────┘

    Benefits over simple averaging:
    1. Learns which models to trust in different regimes
    2. Detects and adjusts for model agreement/disagreement
    3. Can learn non-linear combination rules
    4. Calibrated output probability
    """

    def __init__(self,
                 model_type: str = 'logistic',
                 regularization: float = 0.1,
                 use_calibration: bool = True):
        """
        Initialize meta-learner.

        Args:
            model_type: 'logistic', 'rf', or 'gb'
            regularization: Regularization strength (higher = simpler model)
            use_calibration: Whether to calibrate output probabilities
        """
        self.model_type = model_type
        self.regularization = regularization
        self.use_calibration = use_calibration

        self.meta_model = None
        self.scaler = None
        self.is_trained = False

        # Training history
        self.training_history: List[Dict] = []

        # Performance tracking
        self.validation_accuracy = 0.0
        self.feature_importances: Dict[str, float] = {}

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the meta-model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - using fallback")
            return

        if self.model_type == 'logistic':
            # Logistic regression with L2 regularization
            # Good for interpretability and calibrated probabilities
            self.meta_model = LogisticRegression(
                C=1.0 / self.regularization,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'rf':
            # Random Forest - can capture non-linear relationships
            self.meta_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                min_samples_leaf=20,
                random_state=42
            )
        elif self.model_type == 'gb':
            # Gradient Boosting - powerful but may overfit
            self.meta_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=20,
                random_state=42
            )

        self.scaler = StandardScaler()

    def prepare_meta_features(self, component_predictions: Dict[str, float],
                              regime_info: Optional[Dict] = None) -> MetaFeatures:
        """
        Prepare meta-features from component predictions.

        Args:
            component_predictions: Dict of component -> probability
            regime_info: Optional dict with regime information

        Returns:
            MetaFeatures object
        """
        regime_info = regime_info or {}

        # Base predictions
        probs = list(component_predictions.values())

        # Calculate agreement features
        pred_std = np.std(probs) if probs else 0
        pred_range = max(probs) - min(probs) if probs else 0
        n_bullish = sum(1 for p in probs if p > 0.55)
        n_bearish = sum(1 for p in probs if p < 0.45)

        # LLM sentiment conversion (-1 to 1)
        llm_sentiment = 0.0
        for comp in ['llm_openai', 'llm_gemini', 'llm_claude', 'llm_sentiment']:
            if comp in component_predictions:
                # Convert 0-1 prob to -1 to 1 sentiment
                llm_sentiment = (component_predictions[comp] - 0.5) * 2
                break

        return MetaFeatures(
            ml_rf_prob=component_predictions.get('ml_rf', 0.5),
            ml_gb_prob=component_predictions.get('ml_gb', 0.5),
            ml_xgb_prob=component_predictions.get('ml_xgb', 0.5),
            llm_sentiment=llm_sentiment,
            technical_score=component_predictions.get('technical', 0.5) * 100,
            calendar_score=component_predictions.get('calendar', 0.5) * 100,
            prediction_std=pred_std,
            prediction_range=pred_range,
            n_bullish=n_bullish,
            n_bearish=n_bearish,
            volatility_regime=regime_info.get('volatility_regime', 1),
            trend_strength=regime_info.get('trend_strength', 0),
            market_regime=regime_info.get('market_regime', 1),
            rsi_level=regime_info.get('rsi', 50),
            volume_ratio=regime_info.get('volume_ratio', 1.0)
        )

    def train(self,
              historical_predictions: List[Dict[str, float]],
              outcomes: List[bool],
              regime_infos: Optional[List[Dict]] = None,
              validation_split: float = 0.2) -> Dict:
        """
        Train the meta-learner on historical data.

        Args:
            historical_predictions: List of component prediction dicts
            outcomes: List of actual outcomes (True = went up)
            regime_infos: Optional list of regime info dicts
            validation_split: Fraction for validation

        Returns:
            Training results dict
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train meta-learner without sklearn")
            return {'error': 'sklearn not available'}

        if len(historical_predictions) < 50:
            logger.warning("Not enough data to train meta-learner")
            return {'error': 'insufficient data'}

        # Prepare features
        regime_infos = regime_infos or [{}] * len(historical_predictions)

        X = []
        for preds, regime in zip(historical_predictions, regime_infos):
            features = self.prepare_meta_features(preds, regime)
            X.append(features.to_array().flatten())

        X = np.array(X)
        y = np.array(outcomes).astype(int)

        # Time-ordered split (NOT random!)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        self.meta_model.fit(X_train_scaled, y_train)

        # Optionally wrap with calibration
        if self.use_calibration and len(X_val) >= 30:
            calibrated = CalibratedClassifierCV(
                self.meta_model,
                method='sigmoid',
                cv='prefit'
            )
            calibrated.fit(X_val_scaled, y_val)
            self.meta_model = calibrated

        # Validation
        train_acc = self.meta_model.score(X_train_scaled, y_train)
        val_acc = self.meta_model.score(X_val_scaled, y_val)
        self.validation_accuracy = val_acc

        # Feature importance (for logistic regression)
        if self.model_type == 'logistic' and hasattr(self.meta_model, 'coef_'):
            coefs = self.meta_model.coef_.flatten()
            feature_names = MetaFeatures.feature_names()
            self.feature_importances = {
                name: float(abs(coef))
                for name, coef in zip(feature_names, coefs)
            }
        elif hasattr(self.meta_model, 'feature_importances_'):
            feature_names = MetaFeatures.feature_names()
            self.feature_importances = {
                name: float(imp)
                for name, imp in zip(feature_names, self.meta_model.feature_importances_)
            }

        self.is_trained = True

        result = {
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'feature_importances': self.feature_importances
        }

        logger.info(f"Meta-learner trained: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

        return result

    def predict_proba(self,
                      component_predictions: Dict[str, float],
                      regime_info: Optional[Dict] = None) -> float:
        """
        Get final probability from meta-learner.

        Args:
            component_predictions: Dict of component -> probability
            regime_info: Optional regime information

        Returns:
            Calibrated probability of upward movement
        """
        if not self.is_trained:
            # Fallback to simple average if not trained
            probs = list(component_predictions.values())
            return np.mean(probs) if probs else 0.5

        features = self.prepare_meta_features(component_predictions, regime_info)
        X = features.to_array()
        X_scaled = self.scaler.transform(X)

        # Get probability of class 1 (up)
        prob = self.meta_model.predict_proba(X_scaled)[0, 1]

        return float(prob)

    def get_prediction_explanation(self,
                                    component_predictions: Dict[str, float],
                                    regime_info: Optional[Dict] = None) -> Dict:
        """
        Get explanation for a prediction.

        Returns breakdown of which factors contributed most.
        """
        if not self.is_trained or not self.feature_importances:
            return {'explanation': 'Meta-learner not trained'}

        features = self.prepare_meta_features(component_predictions, regime_info)
        X = features.to_array().flatten()

        # Calculate feature contributions
        contributions = {}
        feature_names = MetaFeatures.feature_names()

        for i, name in enumerate(feature_names):
            importance = self.feature_importances.get(name, 0)
            value = X[i]

            # Contribution = importance * (value - 0.5 for probabilities)
            if 'prob' in name or 'score' in name:
                contrib = importance * (value - 0.5)
            else:
                contrib = importance * value

            contributions[name] = {
                'importance': importance,
                'value': float(value),
                'contribution': float(contrib)
            }

        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )

        final_prob = self.predict_proba(component_predictions, regime_info)

        return {
            'final_probability': final_prob,
            'top_contributors': sorted_contributions[:5],
            'model_type': self.model_type,
            'validation_accuracy': self.validation_accuracy
        }

    def save(self, path: Path):
        """Save trained model to disk."""
        import pickle

        if not self.is_trained:
            logger.warning("Model not trained - nothing to save")
            return

        data = {
            'model': self.meta_model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'validation_accuracy': self.validation_accuracy,
            'feature_importances': self.feature_importances
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Meta-learner saved to {path}")

    def load(self, path: Path) -> bool:
        """Load trained model from disk."""
        import pickle

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.meta_model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.validation_accuracy = data['validation_accuracy']
            self.feature_importances = data['feature_importances']
            self.is_trained = True

            logger.info(f"Meta-learner loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Could not load meta-learner: {e}")
            return False


def demo():
    """Demonstrate meta-learner stacking."""
    print("=" * 60)
    print("MetaLearnerStacker Demo")
    print("=" * 60)

    if not SKLEARN_AVAILABLE:
        print("sklearn not available - demo requires sklearn")
        return

    # Create sample training data
    np.random.seed(42)
    n_samples = 200

    # Simulate component predictions with different accuracies
    training_data = []
    outcomes = []

    for i in range(n_samples):
        # True probability (what actually happened)
        true_prob = np.random.random()
        outcome = true_prob > 0.5

        # Component predictions with noise
        preds = {
            'ml_rf': np.clip(true_prob + np.random.normal(0, 0.15), 0.2, 0.8),
            'ml_gb': np.clip(true_prob + np.random.normal(0, 0.18), 0.2, 0.8),
            'ml_xgb': np.clip(true_prob + np.random.normal(0, 0.16), 0.2, 0.8),
            'llm_sentiment': np.clip(true_prob + np.random.normal(0, 0.25), 0.2, 0.8),
            'technical': np.clip(true_prob + np.random.normal(0, 0.20), 0.2, 0.8),
            'calendar': np.clip(true_prob + np.random.normal(0, 0.30), 0.2, 0.8),
        }

        training_data.append(preds)
        outcomes.append(outcome)

    # Create and train meta-learner
    stacker = MetaLearnerStacker(model_type='logistic')

    print("\nTraining meta-learner...")
    results = stacker.train(training_data, outcomes)

    print(f"\nTraining Results:")
    print(f"  Train Accuracy: {results['train_accuracy']:.1%}")
    print(f"  Validation Accuracy: {results['validation_accuracy']:.1%}")
    print(f"  Training Samples: {results['n_train']}")
    print(f"  Validation Samples: {results['n_val']}")

    print("\nFeature Importances (top 5):")
    sorted_imp = sorted(results['feature_importances'].items(),
                       key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp[:5]:
        print(f"  {name}: {imp:.3f}")

    # Test prediction
    test_preds = {
        'ml_rf': 0.65,
        'ml_gb': 0.58,
        'ml_xgb': 0.62,
        'llm_sentiment': 0.55,
        'technical': 0.70,
        'calendar': 0.52
    }

    print("\n--- Test Prediction ---")
    print("Component predictions:")
    for comp, prob in test_preds.items():
        print(f"  {comp}: {prob:.1%}")

    final_prob = stacker.predict_proba(test_preds)
    print(f"\nMeta-learner output: {final_prob:.1%}")

    # Compare to simple average
    simple_avg = np.mean(list(test_preds.values()))
    print(f"Simple average: {simple_avg:.1%}")

    # Explanation
    explanation = stacker.get_prediction_explanation(test_preds)
    print("\nTop contributors to this prediction:")
    for name, data in explanation['top_contributors'][:3]:
        print(f"  {name}: contribution={data['contribution']:.3f}")


if __name__ == "__main__":
    demo()
