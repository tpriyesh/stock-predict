"""
CalibratedScoringEngine - Production-Grade Scoring with Empirical Calibration

This module addresses critical gaps:
1. Empirical probability calibration (not ad-hoc penalties)
2. Adaptive weights based on historical performance
3. Transaction cost-adjusted targets
4. Confidence intervals via Monte Carlo
5. Regime-aware adjustments

CRITICAL: "70% confidence" MUST mean 70% historical win rate.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple
from sklearn.isotonic import IsotonicRegression
from loguru import logger
import hashlib
import json
import os

from config.settings import get_settings
from src.features.technical import TechnicalIndicators
from src.features.market_features import MarketFeatures
from src.storage.models import SignalType, TradeType
from src.core.transaction_costs import TransactionCostCalculator, BrokerType

# Regime-Specific Calibration - Optional
try:
    from src.core.regime_calibration import (
        get_regime_calibrator,
        RegimeSpecificCalibrator,
        CalibrationRegime,
    )
    REGIME_CALIBRATION_AVAILABLE = True
except ImportError:
    REGIME_CALIBRATION_AVAILABLE = False
    logger.warning("Regime calibration module not available")


@dataclass
class ConfidenceInterval:
    """Prediction uncertainty bounds."""
    lower_5: float      # 5th percentile
    lower_25: float     # 25th percentile
    median: float       # 50th percentile
    upper_75: float     # 75th percentile
    upper_95: float     # 95th percentile

    def to_dict(self) -> dict:
        return {
            'p5': round(self.lower_5, 2),
            'p25': round(self.lower_25, 2),
            'p50': round(self.median, 2),
            'p75': round(self.upper_75, 2),
            'p95': round(self.upper_95, 2)
        }


@dataclass
class CalibratedScore:
    """Complete calibrated score with uncertainty quantification."""
    symbol: str
    date: date
    trade_type: TradeType

    # Component scores (0-1)
    technical_score: float
    momentum_score: float
    news_score: float
    sector_score: float
    volume_score: float

    # Weights used (may be adaptive)
    weights_used: Dict[str, float]

    # Raw and calibrated scores
    raw_score: float
    calibrated_confidence: float  # Empirically calibrated

    signal: SignalType

    # Trading levels (transaction cost adjusted)
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    target_net_of_costs: float  # After STT, brokerage, slippage
    risk_reward: float
    risk_reward_net: float  # Net of costs

    # Confidence intervals
    price_intervals: ConfidenceInterval

    # Risk metrics
    atr_pct: float
    liquidity_cr: float
    regime: str
    regime_adjustment: float

    # Calibration metadata
    calibration_samples: int
    calibration_accuracy: float

    # Reasoning
    reasons: List[str]

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'date': str(self.date),
            'trade_type': self.trade_type.value,
            'signal': self.signal.value,
            'calibrated_confidence': round(self.calibrated_confidence, 3),
            'raw_score': round(self.raw_score, 3),
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'target_net_of_costs': self.target_net_of_costs,
            'risk_reward': round(self.risk_reward, 2),
            'risk_reward_net': round(self.risk_reward_net, 2),
            'price_intervals': self.price_intervals.to_dict(),
            'regime': self.regime,
            'weights_used': {k: round(v, 3) for k, v in self.weights_used.items()},
            'calibration_samples': self.calibration_samples,
            'calibration_accuracy': round(self.calibration_accuracy, 3),
            'reasons': self.reasons
        }


class AdaptiveWeightManager:
    """
    Learns optimal weights from historical component performance.

    Instead of hardcoded weights, tracks which components predict well
    and adjusts weights accordingly.
    """

    DEFAULT_WEIGHTS = {
        'technical': 0.30,
        'momentum': 0.25,
        'news': 0.15,
        'sector': 0.15,
        'volume': 0.15
    }

    def __init__(self, history_file: str = None):
        self.history_file = history_file or '/tmp/weight_history.json'
        self.component_accuracy: Dict[str, List[float]] = {
            'technical': [],
            'momentum': [],
            'news': [],
            'sector': [],
            'volume': []
        }
        self.min_samples = 30
        self.decay_factor = 0.95  # Exponential decay for older samples
        self._load_history()

    def _load_history(self):
        """Load historical accuracy data."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.component_accuracy = data.get('accuracy', self.component_accuracy)
            except Exception as e:
                logger.warning(f"Could not load weight history: {e}")

    def _save_history(self):
        """Persist accuracy data."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({'accuracy': self.component_accuracy}, f)
        except Exception as e:
            logger.warning(f"Could not save weight history: {e}")

    def record_outcome(self,
                       component_scores: Dict[str, float],
                       actual_outcome: int):
        """
        Record prediction outcome for weight learning.

        Args:
            component_scores: Dict of component name to score (0-1)
            actual_outcome: 1 if trade was profitable, 0 otherwise
        """
        for component, score in component_scores.items():
            if component in self.component_accuracy:
                # Score > 0.5 predicted positive, check if correct
                predicted_positive = score > 0.5
                was_correct = (predicted_positive and actual_outcome == 1) or \
                             (not predicted_positive and actual_outcome == 0)
                self.component_accuracy[component].append(1.0 if was_correct else 0.0)

                # Keep last 500 samples
                if len(self.component_accuracy[component]) > 500:
                    self.component_accuracy[component] = self.component_accuracy[component][-500:]

        self._save_history()

    def get_adaptive_weights(self) -> Tuple[Dict[str, float], bool]:
        """
        Calculate weights based on component performance.

        Returns:
            Tuple of (weights dict, is_adaptive flag)
        """
        # Check if we have enough samples
        min_samples = min(len(v) for v in self.component_accuracy.values())

        if min_samples < self.min_samples:
            logger.debug(f"Insufficient samples ({min_samples}) for adaptive weights, using defaults")
            return self.DEFAULT_WEIGHTS.copy(), False

        # Calculate exponentially weighted accuracy for each component
        accuracies = {}
        for component, history in self.component_accuracy.items():
            if not history:
                accuracies[component] = 0.5
                continue

            # Apply exponential decay (recent samples matter more)
            weights = np.array([self.decay_factor ** i for i in range(len(history))])[::-1]
            weighted_acc = np.average(history, weights=weights)
            accuracies[component] = weighted_acc

        # Convert to weights (higher accuracy = higher weight)
        # Use softmax-like normalization
        acc_values = np.array(list(accuracies.values()))

        # Shift to avoid division issues
        shifted = acc_values - acc_values.min() + 0.1
        normalized = shifted / shifted.sum()

        weights = {}
        for i, component in enumerate(accuracies.keys()):
            weights[component] = float(normalized[i])

        logger.info(f"Adaptive weights: {weights} (from {min_samples} samples)")
        return weights, True

    def get_component_stats(self) -> Dict[str, Dict]:
        """Get detailed stats for each component."""
        stats = {}
        for component, history in self.component_accuracy.items():
            if history:
                stats[component] = {
                    'samples': len(history),
                    'accuracy': np.mean(history),
                    'recent_accuracy': np.mean(history[-50:]) if len(history) >= 50 else np.mean(history),
                    'trend': 'improving' if len(history) >= 20 and np.mean(history[-10:]) > np.mean(history[-20:-10]) else 'stable'
                }
            else:
                stats[component] = {'samples': 0, 'accuracy': 0.5, 'recent_accuracy': 0.5, 'trend': 'unknown'}
        return stats


class EmpiricalCalibrator:
    """
    Empirical probability calibration using isotonic regression.

    Ensures that stated confidence matches historical win rate.
    """

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or '/tmp/calibration_cache.json'
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted = False
        self.n_samples = 0
        self.calibration_accuracy = 0.0
        self.prediction_history: List[Tuple[float, int]] = []
        self._load_cache()

    def _load_cache(self):
        """Load calibration history."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.prediction_history = [tuple(x) for x in data.get('history', [])]
                    if len(self.prediction_history) >= 20:
                        self._fit_calibrator()
            except Exception as e:
                logger.warning(f"Could not load calibration cache: {e}")

    def _save_cache(self):
        """Persist calibration history."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'history': self.prediction_history[-1000:]}, f)  # Keep last 1000
        except Exception as e:
            logger.warning(f"Could not save calibration cache: {e}")

    def _fit_calibrator(self):
        """Fit isotonic regression on historical data."""
        if len(self.prediction_history) < 20:
            return

        scores = np.array([x[0] for x in self.prediction_history])
        outcomes = np.array([x[1] for x in self.prediction_history])

        self.calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
        self.calibrator.fit(scores, outcomes)
        self.is_fitted = True
        self.n_samples = len(self.prediction_history)

        # Calculate calibration accuracy (how well calibrated scores match actual rates)
        calibrated = self.calibrator.predict(scores)
        # Bin and check
        bins = np.linspace(0, 1, 11)
        accuracies = []
        for i in range(10):
            mask = (calibrated >= bins[i]) & (calibrated < bins[i+1])
            if mask.sum() > 0:
                expected = calibrated[mask].mean()
                actual = outcomes[mask].mean()
                accuracies.append(1 - abs(expected - actual))

        self.calibration_accuracy = np.mean(accuracies) if accuracies else 0.5
        logger.info(f"Calibrator fitted on {self.n_samples} samples, accuracy: {self.calibration_accuracy:.2%}")

    def record_outcome(self, raw_score: float, actual_outcome: int):
        """Record a prediction outcome for calibration learning."""
        self.prediction_history.append((raw_score, actual_outcome))

        # Refit periodically
        if len(self.prediction_history) % 50 == 0:
            self._fit_calibrator()

        self._save_cache()

    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate raw score to probability.

        Args:
            raw_score: Raw confidence score (0-1)

        Returns:
            Calibrated probability (0-1) that matches historical win rate
        """
        if not self.is_fitted or self.calibrator is None:
            # Apply conservative shrinkage towards 0.5 if uncalibrated
            # This prevents overconfident predictions
            shrinkage = 0.3
            return 0.5 + (raw_score - 0.5) * (1 - shrinkage)

        return float(self.calibrator.predict([raw_score])[0])

    def get_reliability_data(self) -> Dict:
        """Get data for reliability diagram."""
        if len(self.prediction_history) < 20:
            return {'bins': [], 'actual': [], 'predicted': [], 'counts': []}

        scores = np.array([x[0] for x in self.prediction_history])
        outcomes = np.array([x[1] for x in self.prediction_history])

        bins = np.linspace(0, 1, 11)
        bin_centers = []
        actual_rates = []
        predicted_rates = []
        counts = []

        for i in range(10):
            mask = (scores >= bins[i]) & (scores < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                actual_rates.append(outcomes[mask].mean())
                predicted_rates.append(scores[mask].mean())
                counts.append(int(mask.sum()))

        return {
            'bins': bin_centers,
            'actual': actual_rates,
            'predicted': predicted_rates,
            'counts': counts
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for price prediction intervals.

    Provides uncertainty quantification instead of single-point predictions.
    """

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def simulate_price_paths(self,
                             current_price: float,
                             daily_volatility: float,
                             drift: float,
                             holding_days: int) -> np.ndarray:
        """
        Simulate future price paths using Geometric Brownian Motion.

        Args:
            current_price: Current stock price
            daily_volatility: Daily return volatility (as decimal, e.g., 0.02 for 2%)
            drift: Expected daily return (as decimal)
            holding_days: Number of days to simulate

        Returns:
            Array of final prices from simulations
        """
        dt = 1  # Daily steps

        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, holding_days))

        # GBM formula: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        log_returns = (drift - 0.5 * daily_volatility**2) * dt + daily_volatility * np.sqrt(dt) * Z

        # Cumulative log returns
        cumulative_returns = np.cumsum(log_returns, axis=1)

        # Final prices
        final_prices = current_price * np.exp(cumulative_returns[:, -1])

        return final_prices

    def get_confidence_intervals(self,
                                  current_price: float,
                                  atr_pct: float,
                                  signal: SignalType,
                                  trade_type: TradeType) -> ConfidenceInterval:
        """
        Generate price confidence intervals.

        Args:
            current_price: Current price
            atr_pct: Average True Range as percentage
            signal: BUY or SELL
            trade_type: INTRADAY or SWING

        Returns:
            ConfidenceInterval with price percentiles
        """
        # Convert ATR% to daily volatility
        daily_volatility = atr_pct / 100 / np.sqrt(14)  # ATR is typically 14-day

        # Set drift based on signal
        if signal == SignalType.BUY:
            drift = 0.0005  # Slight positive bias for buys
        elif signal == SignalType.SELL:
            drift = -0.0005
        else:
            drift = 0

        # Holding period
        if trade_type == TradeType.INTRADAY:
            holding_days = 1
        else:
            holding_days = 5

        final_prices = self.simulate_price_paths(
            current_price, daily_volatility, drift, holding_days
        )

        return ConfidenceInterval(
            lower_5=float(np.percentile(final_prices, 5)),
            lower_25=float(np.percentile(final_prices, 25)),
            median=float(np.percentile(final_prices, 50)),
            upper_75=float(np.percentile(final_prices, 75)),
            upper_95=float(np.percentile(final_prices, 95))
        )


class CalibratedScoringEngine:
    """
    Production-grade scoring engine with:
    1. Empirical probability calibration (global or regime-specific)
    2. Adaptive weights based on component performance
    3. Transaction cost-adjusted targets
    4. Monte Carlo confidence intervals
    5. Regime-aware adjustments
    """

    def __init__(self,
                 use_adaptive_weights: bool = True,
                 broker_type: BrokerType = BrokerType.DISCOUNT,
                 use_regime_calibration: bool = True):
        self.settings = get_settings()
        self.market_features = MarketFeatures()
        self.weight_manager = AdaptiveWeightManager()
        self.calibrator = EmpiricalCalibrator()  # Global fallback
        self.mc_simulator = MonteCarloSimulator(n_simulations=1000)
        self.cost_calculator = TransactionCostCalculator(broker_type=broker_type)
        self.use_adaptive_weights = use_adaptive_weights

        # Regime-Specific Calibration
        self.use_regime_calibration = use_regime_calibration and REGIME_CALIBRATION_AVAILABLE
        self.regime_calibrator: Optional[RegimeSpecificCalibrator] = None

        if self.use_regime_calibration:
            try:
                self.regime_calibrator = get_regime_calibrator()
                logger.info("Regime-specific calibration enabled in scoring engine")
            except Exception as e:
                logger.warning(f"Failed to initialize regime calibrator: {e}")
                self.use_regime_calibration = False

    def score_stock(self,
                    symbol: str,
                    df: pd.DataFrame,
                    trade_type: TradeType,
                    news_score: float = 0.5,
                    news_reasons: List[str] = None,
                    market_context: Optional[dict] = None) -> Optional[CalibratedScore]:
        """
        Calculate complete calibrated score for a stock.
        """
        if df.empty or len(df) < 50:
            logger.warning(f"{symbol}: Insufficient data for scoring")
            return None

        # Calculate technical indicators
        if 'rsi' not in df.columns:
            df = TechnicalIndicators.calculate_all(df)

        latest = df.iloc[-1]
        current_price = float(latest['close'])

        # Get date
        latest_date = df.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        reasons = news_reasons or []

        # Calculate component scores
        tech_score, tech_reasons = self._calculate_technical_score(df, trade_type)
        reasons.extend(tech_reasons)

        momentum_score, mom_reasons = self._calculate_momentum_score(df, trade_type)
        reasons.extend(mom_reasons)

        volume_score, vol_reasons = self._calculate_volume_score(df)
        reasons.extend(vol_reasons)

        # Market context
        if market_context is None:
            market_context = self.market_features.get_market_context()

        sector_score = self.market_features.get_sector_score(symbol, market_context)
        regime = market_context.get('regime', 'NEUTRAL')
        regime_adjustment = self.market_features.get_regime_adjustment(market_context)

        sector = self.market_features.get_symbol_sector(symbol)
        if sector_score > 0.6:
            reasons.append(f"[SECTOR] {sector} showing strength ({sector_score:.2f})")
        elif sector_score < 0.4:
            reasons.append(f"[SECTOR] {sector} showing weakness ({sector_score:.2f})")

        # News
        if news_score > 0.6:
            reasons.append(f"[NEWS] Positive sentiment ({news_score:.2f})")
        elif news_score < 0.4:
            reasons.append(f"[NEWS] Negative sentiment ({news_score:.2f})")

        # Get weights (adaptive or default)
        component_scores = {
            'technical': tech_score,
            'momentum': momentum_score,
            'news': news_score,
            'sector': sector_score,
            'volume': volume_score
        }

        if self.use_adaptive_weights:
            weights, is_adaptive = self.weight_manager.get_adaptive_weights()
            if is_adaptive:
                reasons.append("[WEIGHTS] Using adaptive weights based on historical performance")
        else:
            weights = AdaptiveWeightManager.DEFAULT_WEIGHTS

        # Calculate raw score
        raw_score = sum(weights[k] * component_scores[k] for k in weights)

        # Apply regime adjustment
        adjusted_score = raw_score * regime_adjustment

        # CALIBRATE the score - prefer regime-specific if available
        if self.use_regime_calibration and self.regime_calibrator:
            calibrated_confidence = self.regime_calibrator.calibrate(adjusted_score, regime)
        else:
            calibrated_confidence = self.calibrator.calibrate(adjusted_score)

        # Determine signal
        if calibrated_confidence > 0.6:
            signal = SignalType.BUY
        elif calibrated_confidence < 0.4:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Calculate trading levels
        atr = float(latest.get('atr', current_price * 0.02))
        atr_pct = float(latest.get('atr_pct', 2.0))

        entry_price, stop_loss, target_price = self._calculate_levels(
            current_price, atr, signal, trade_type
        )

        # Apply transaction costs to get NET target
        if signal == SignalType.BUY:
            # Buy costs on entry
            entry_with_costs = self.cost_calculator.calculate_buy_cost(
                entry_price, shares=1, is_intraday=(trade_type == TradeType.INTRADAY)
            )
            # Sell costs on exit
            exit_net = self.cost_calculator.calculate_sell_proceeds(
                target_price, shares=1, is_intraday=(trade_type == TradeType.INTRADAY)
            )
            # Cost impact on stop
            stop_net = self.cost_calculator.calculate_sell_proceeds(
                stop_loss, shares=1, is_intraday=(trade_type == TradeType.INTRADAY)
            )

            target_net = exit_net - (entry_with_costs - entry_price)  # Adjust for entry costs
        else:
            target_net = target_price  # Simplified for sells

        # Risk-reward calculations
        if entry_price != stop_loss:
            risk_reward = abs(target_price - entry_price) / abs(entry_price - stop_loss)
            risk_reward_net = abs(target_net - entry_price) / abs(entry_price - stop_loss)
        else:
            risk_reward = 0
            risk_reward_net = 0

        # Monte Carlo confidence intervals
        price_intervals = self.mc_simulator.get_confidence_intervals(
            current_price, atr_pct, signal, trade_type
        )

        # Liquidity
        avg_value = float((df['close'] * df['volume']).tail(20).mean())
        liquidity_cr = avg_value / 1e7

        return CalibratedScore(
            symbol=symbol,
            date=latest_date,
            trade_type=trade_type,
            technical_score=tech_score,
            momentum_score=momentum_score,
            news_score=news_score,
            sector_score=sector_score,
            volume_score=volume_score,
            weights_used=weights,
            raw_score=raw_score,
            calibrated_confidence=calibrated_confidence,
            signal=signal,
            current_price=round(current_price, 2),
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            target_net_of_costs=round(target_net, 2),
            risk_reward=risk_reward,
            risk_reward_net=risk_reward_net,
            price_intervals=price_intervals,
            atr_pct=atr_pct,
            liquidity_cr=liquidity_cr,
            regime=regime,
            regime_adjustment=regime_adjustment,
            calibration_samples=self.calibrator.n_samples,
            calibration_accuracy=self.calibrator.calibration_accuracy,
            reasons=reasons
        )

    def record_trade_outcome(self,
                             component_scores: Dict[str, float],
                             raw_score: float,
                             actual_outcome: int,
                             regime: str = ""):
        """
        Record trade outcome for learning.

        Args:
            component_scores: Dict of component scores
            raw_score: Raw prediction score
            actual_outcome: 1 if profitable, 0 otherwise
            regime: Market regime at trade time (for regime-specific calibration)
        """
        self.weight_manager.record_outcome(component_scores, actual_outcome)
        self.calibrator.record_outcome(raw_score, actual_outcome)

        # Also record to regime-specific calibrator if available
        if self.use_regime_calibration and self.regime_calibrator and regime:
            self.regime_calibrator.record_outcome(raw_score, actual_outcome, regime)

    def _calculate_technical_score(self, df: pd.DataFrame, trade_type: TradeType) -> Tuple[float, List[str]]:
        """Calculate technical indicator score."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        reasons = []
        score = 0.5

        # RSI
        rsi = latest.get('rsi')
        if pd.notna(rsi):
            if rsi < 30:
                score += 0.15
                if rsi > prev.get('rsi', rsi):
                    score += 0.1
                    reasons.append(f"[TECHNICAL] RSI oversold bounce ({rsi:.1f})")
            elif rsi > 70:
                score -= 0.15
                if rsi < prev.get('rsi', rsi):
                    score -= 0.1
                    reasons.append(f"[TECHNICAL] RSI overbought reversal ({rsi:.1f})")
            elif 40 < rsi < 60:
                score += 0.05

        # MACD
        macd_hist = latest.get('macd_histogram')
        prev_macd_hist = prev.get('macd_histogram')
        if pd.notna(macd_hist) and pd.notna(prev_macd_hist):
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                score += 0.1
                reasons.append("[TECHNICAL] MACD bullish momentum")
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                score -= 0.1
                reasons.append("[TECHNICAL] MACD bearish momentum")

        # Bollinger Bands
        bb_pos = latest.get('bb_position')
        if pd.notna(bb_pos):
            if bb_pos < 0.2:
                score += 0.1
                reasons.append("[TECHNICAL] Near lower Bollinger Band")
            elif bb_pos > 0.8:
                score -= 0.05

        # Trend alignment
        close = latest['close']
        sma20 = latest.get('sma_20')
        sma50 = latest.get('sma_50')

        if pd.notna(sma20) and pd.notna(sma50):
            if close > sma20 > sma50:
                score += 0.1
                reasons.append("[TECHNICAL] Uptrend (Price > SMA20 > SMA50)")
            elif close < sma20 < sma50:
                score -= 0.1
                reasons.append("[TECHNICAL] Downtrend")

        # Patterns
        if latest.get('bullish_engulfing', 0) > 0:
            score += 0.1
            reasons.append("[PATTERN] Bullish engulfing")
        if latest.get('bearish_engulfing', 0) < 0:
            score -= 0.1
            reasons.append("[PATTERN] Bearish engulfing")

        return max(0, min(1, score)), reasons

    def _calculate_momentum_score(self, df: pd.DataFrame, trade_type: TradeType) -> Tuple[float, List[str]]:
        """Calculate momentum score."""
        latest = df.iloc[-1]
        reasons = []
        score = 0.5

        mom_5 = latest.get('momentum_5')
        if pd.notna(mom_5):
            if mom_5 > 3:
                score += 0.15
                reasons.append(f"[MOMENTUM] Strong 5-day momentum (+{mom_5:.1f}%)")
            elif mom_5 > 0:
                score += 0.05
            elif mom_5 < -3:
                score -= 0.1

        mom_20 = latest.get('momentum_20')
        if pd.notna(mom_20):
            weight = 0.15 if trade_type == TradeType.SWING else 0.05
            if mom_20 > 5:
                score += weight
                reasons.append(f"[MOMENTUM] Strong 20-day momentum (+{mom_20:.1f}%)")
            elif mom_20 < -5:
                score -= weight

        # VWAP for intraday
        close = latest['close']
        vwap = latest.get('vwap')
        if pd.notna(vwap) and trade_type == TradeType.INTRADAY:
            if close > vwap:
                score += 0.05
                reasons.append("[MOMENTUM] Price above VWAP")
            else:
                score -= 0.05

        # Trend strength
        trend_strength = latest.get('trend_strength')
        if pd.notna(trend_strength) and trend_strength > 25:
            if latest.get('momentum_5', 0) > 0:
                score += 0.1
            else:
                score -= 0.1

        return max(0, min(1, score)), reasons

    def _calculate_volume_score(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate volume score."""
        latest = df.iloc[-1]
        reasons = []
        score = 0.5

        vol_ratio = latest.get('volume_ratio')
        if pd.notna(vol_ratio):
            if vol_ratio > 2.0:
                score += 0.2
                reasons.append(f"[VOLUME] High volume ({vol_ratio:.1f}x average)")
            elif vol_ratio > 1.3:
                score += 0.1
                reasons.append(f"[VOLUME] Above average ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.5:
                score -= 0.1
                reasons.append("[VOLUME] Low volume")

        return max(0, min(1, score)), reasons

    def _calculate_levels(self,
                          current_price: float,
                          atr: float,
                          signal: SignalType,
                          trade_type: TradeType) -> Tuple[float, float, float]:
        """Calculate entry, stop-loss, and target levels."""
        if trade_type == TradeType.INTRADAY:
            stop_mult = 1.0
            target_mult = 1.5
        else:
            stop_mult = 1.5
            target_mult = 2.5

        if signal == SignalType.BUY:
            entry = current_price
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)
        elif signal == SignalType.SELL:
            entry = current_price
            stop = current_price + (atr * stop_mult)
            target = current_price - (atr * target_mult)
        else:
            entry = current_price
            stop = current_price - (atr * stop_mult)
            target = current_price + (atr * target_mult)

        return entry, stop, target

    def get_diagnostics(self) -> Dict:
        """Get system diagnostics and health."""
        diagnostics = {
            'calibration': {
                'samples': self.calibrator.n_samples,
                'accuracy': self.calibrator.calibration_accuracy,
                'is_fitted': self.calibrator.is_fitted,
                'reliability': self.calibrator.get_reliability_data()
            },
            'weights': {
                'is_adaptive': self.use_adaptive_weights,
                'current_weights': self.weight_manager.get_adaptive_weights()[0],
                'component_stats': self.weight_manager.get_component_stats()
            }
        }

        # Add regime-specific calibration stats if available
        if self.use_regime_calibration and self.regime_calibrator:
            diagnostics['regime_calibration'] = {
                'enabled': True,
                'stats': self.regime_calibrator.get_regime_stats()
            }
        else:
            diagnostics['regime_calibration'] = {'enabled': False}

        return diagnostics
