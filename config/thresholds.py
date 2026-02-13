"""
Centralized Thresholds Configuration

This module contains ALL signal thresholds, calibration values, and scoring
parameters used across the prediction system. Centralizing these ensures:

1. Consistency across all scoring engines
2. Easy tuning based on backtest results
3. Single source of truth for threshold values

IMPORTANT: When modifying thresholds, update the corresponding backtest
validation to ensure the new values are empirically validated.

Calibration source: BACKTEST_RESULTS.md
- BUY Signal Accuracy: 57.6% (330 trades)
- SELL Signal Accuracy: 28.6% (inverted - disabled)
- Calibration Error: 1.9%
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SignalThresholds:
    """
    Signal generation thresholds.

    These determine when BUY/HOLD/SELL signals are generated based on
    ensemble scores. Values calibrated from backtest results.
    """
    # BUY thresholds
    strong_buy: float = 0.70      # High confidence BUY - require 3+ model agreement
    buy: float = 0.60             # Standard BUY - require 2+ model agreement
    weak_buy: float = 0.55        # Weak BUY - use smaller position

    # HOLD zone
    hold_upper: float = 0.55      # Upper bound of HOLD zone
    hold_lower: float = 0.45      # Lower bound of HOLD zone

    # SELL thresholds (DISABLED - 28.6% accuracy means inverted)
    # NOTE: SELL signals are currently disabled in scoring engines
    # These values are kept for reference but not used
    sell: float = 0.40            # Standard SELL threshold
    strong_sell: float = 0.30     # Strong SELL threshold

    # Flag indicating SELL signals are disabled
    sell_signals_enabled: bool = False


@dataclass(frozen=True)
class ConfidenceCalibration:
    """
    Maps raw ensemble scores to realistic confidence levels.

    Based on backtest results showing 57.6% overall BUY accuracy.
    Higher scores correlate with higher actual win rates.
    """
    # Score range -> Expected win rate
    # Format: (score_low, score_high): expected_accuracy
    very_high_score_accuracy: float = 0.65    # Score 0.75-1.00
    high_score_accuracy: float = 0.60         # Score 0.65-0.75
    moderate_score_accuracy: float = 0.55     # Score 0.55-0.65
    neutral_score_accuracy: float = 0.50      # Score 0.45-0.55
    below_neutral_accuracy: float = 0.45      # Score 0.00-0.45

    # Model agreement bonuses
    # When multiple models agree, confidence increases
    four_model_agreement_bonus: float = 0.08  # All 4 models agree
    three_model_agreement_bonus: float = 0.05 # 3 models agree
    two_model_agreement_bonus: float = 0.02   # 2 models agree
    one_model_agreement_bonus: float = 0.00   # Only 1 model
    no_agreement_penalty: float = -0.05       # No agreement

    # Confidence bounds
    max_confidence: float = 0.80              # Never claim > 80% accuracy
    min_confidence: float = 0.35              # Minimum displayable confidence


@dataclass(frozen=True)
class EnsembleWeights:
    """
    Weights for combining model predictions into ensemble score.

    These weights determine how much each model type contributes
    to the final prediction. Calibrated from backtest performance.
    """
    # Base 4-model ensemble weights (preserved for backward compatibility)
    base_technical: float = 0.35   # Technical + Momentum + News + Volume
    physics: float = 0.25          # Physics-inspired models
    math: float = 0.20             # Mathematical models
    regime: float = 0.20           # HMM regime detection

    # 7-model ensemble weights (when all models are enabled)
    # Redistributed to accommodate Macro, Alternative, and Advanced engines
    base_technical_7m: float = 0.25   # Technical + Momentum + News + Volume
    physics_7m: float = 0.18          # Physics-inspired models
    math_7m: float = 0.14             # Mathematical models
    regime_7m: float = 0.13           # HMM regime detection
    macro_7m: float = 0.10            # Macro indicators (commodities, currencies, bonds)
    alternative_7m: float = 0.10      # Alternative data (earnings, options, institutional)
    advanced_7m: float = 0.10         # Advanced math models (Kalman, Wavelet, PCA, etc.)

    # Advanced engine weights (6-model ensemble)
    pca: float = 0.10              # PCA dimensionality reduction
    wavelet: float = 0.15          # Wavelet multi-resolution
    kalman: float = 0.20           # Kalman trend filter
    markov: float = 0.25           # Markov regime detection
    factor: float = 0.10           # Factor model
    dqn: float = 0.20              # DQN position sizing


@dataclass(frozen=True)
class RiskThresholds:
    """
    Risk management thresholds.

    These filter out high-risk stocks and apply penalties
    to confidence scores for risky situations.
    """
    # Volatility (ATR) thresholds
    max_atr_percent: float = 5.0          # Max allowed ATR%
    high_volatility_atr: float = 4.0      # Apply 0.85x confidence penalty
    moderate_volatility_atr: float = 3.0  # Apply 0.92x confidence penalty

    # Liquidity thresholds (in Crores INR)
    min_liquidity_cr: float = 10.0        # Minimum daily volume
    low_liquidity_cr: float = 5.0         # Apply 0.85x confidence penalty
    moderate_liquidity_cr: float = 10.0   # Apply 0.92x confidence penalty

    # Regime stability
    unstable_regime_threshold: float = 0.5  # Apply penalty if below

    # Predictability
    low_predictability: float = 0.2       # Apply 0.90x penalty if below


@dataclass(frozen=True)
class TradeLevelMultipliers:
    """
    Multipliers for calculating stop-loss and target levels.

    These are applied to ATR to determine trading levels.
    """
    # Intraday trades
    intraday_stop_mult: float = 1.0
    intraday_target_mult: float = 2.0

    # Swing trades
    swing_stop_mult: float = 1.5
    swing_target_mult: float = 2.5

    # Positional trades
    positional_stop_mult: float = 2.0
    positional_target_mult: float = 3.5


@dataclass(frozen=True)
class IndicatorThresholds:
    """
    Technical indicator thresholds.

    Standard values for RSI, Bollinger Bands, and other indicators.
    """
    # RSI thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Bollinger Band position
    bb_oversold: float = 0.2
    bb_overbought: float = 0.8

    # Momentum thresholds
    strong_momentum_5d: float = 3.0   # 5-day momentum %
    strong_momentum_20d: float = 5.0  # 20-day momentum %

    # Volume ratio
    high_volume_surge: float = 2.0
    above_avg_volume: float = 1.3
    low_volume: float = 0.5

    # Trend strength (ADX)
    strong_trend: float = 25.0


# Global singleton instances
SIGNAL_THRESHOLDS = SignalThresholds()
CONFIDENCE_CALIBRATION = ConfidenceCalibration()
ENSEMBLE_WEIGHTS = EnsembleWeights()
RISK_THRESHOLDS = RiskThresholds()
TRADE_LEVEL_MULTIPLIERS = TradeLevelMultipliers()
INDICATOR_THRESHOLDS = IndicatorThresholds()


def get_confidence_for_score(score: float, model_agreement: int = 0, total_models: int = 4) -> float:
    """
    Get calibrated confidence for a given ensemble score.

    Args:
        score: Raw ensemble score (0-1)
        model_agreement: Number of models agreeing (0-4 or 0-7)
        total_models: Total number of models in ensemble (4 or 7)

    Returns:
        Calibrated confidence value
    """
    cal = CONFIDENCE_CALIBRATION

    # Base confidence from score range
    if score >= 0.75:
        base = cal.very_high_score_accuracy
    elif score >= 0.65:
        base = cal.high_score_accuracy
    elif score >= 0.55:
        base = cal.moderate_score_accuracy
    elif score >= 0.45:
        base = cal.neutral_score_accuracy
    else:
        base = cal.below_neutral_accuracy

    # Agreement bonus - scaled for 7-model ensemble
    if total_models == 7:
        agreement_map = {
            7: cal.four_model_agreement_bonus + 0.03,   # All 7 agree = +11%
            6: cal.four_model_agreement_bonus,          # 6/7 = +8%
            5: cal.three_model_agreement_bonus + 0.01,  # 5/7 = +6%
            4: cal.three_model_agreement_bonus,         # 4/7 = +5%
            3: cal.two_model_agreement_bonus,           # 3/7 = +2%
            2: cal.one_model_agreement_bonus,           # 2/7 = 0%
            1: cal.no_agreement_penalty,                # 1/7 = -5%
            0: cal.no_agreement_penalty - 0.05          # 0/7 = -10%
        }
    else:
        agreement_map = {
            4: cal.four_model_agreement_bonus,
            3: cal.three_model_agreement_bonus,
            2: cal.two_model_agreement_bonus,
            1: cal.one_model_agreement_bonus,
            0: cal.no_agreement_penalty
        }
    bonus = agreement_map.get(model_agreement, 0)

    # Calculate final confidence
    confidence = base + bonus

    # Apply bounds
    return max(cal.min_confidence, min(cal.max_confidence, confidence))


def apply_risk_penalties(
    confidence: float,
    atr_pct: float = 0.0,
    liquidity_cr: float = 100.0,
    regime_stability: float = 1.0,
    predictability: float = 1.0
) -> Tuple[float, list]:
    """
    Apply risk-based penalties to confidence score.

    Args:
        confidence: Base confidence score
        atr_pct: ATR as percentage of price
        liquidity_cr: Daily trading volume in Crores
        regime_stability: HMM regime stability (0-1)
        predictability: Market predictability score (0-1)

    Returns:
        Tuple of (adjusted_confidence, list of warnings)
    """
    risk = RISK_THRESHOLDS
    warnings = []

    # Volatility penalty
    if atr_pct > risk.high_volatility_atr:
        confidence *= 0.85
        warnings.append(f"[RISK] High volatility ({atr_pct:.1f}% ATR)")
    elif atr_pct > risk.moderate_volatility_atr:
        confidence *= 0.92

    # Liquidity penalty
    if liquidity_cr < risk.low_liquidity_cr:
        confidence *= 0.85
        warnings.append(f"[RISK] Low liquidity ({liquidity_cr:.1f} Cr)")
    elif liquidity_cr < risk.moderate_liquidity_cr:
        confidence *= 0.92

    # Regime stability penalty
    if regime_stability < risk.unstable_regime_threshold:
        confidence *= 0.90
        warnings.append(f"[RISK] Unstable regime ({regime_stability:.0%})")

    # Predictability penalty
    if predictability < risk.low_predictability:
        confidence *= 0.90
        warnings.append(f"[RISK] Low predictability ({predictability:.0%})")

    return confidence, warnings
