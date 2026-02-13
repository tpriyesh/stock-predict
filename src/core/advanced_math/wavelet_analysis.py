"""
Wavelet Multi-Resolution Analysis for Stock Prediction

Mathematical Foundation:
========================
Discrete Wavelet Transform (DWT):
   W(a,b) = (1/√a) Σₜ x(t) ψ((t-b)/a)

Decomposition Levels (for daily data):
- Level 1 (D1): 1-2 day noise (filter out)
- Level 2 (D2): 2-4 day ultra-short patterns
- Level 3 (D3): 4-8 day weekly patterns
- Level 4 (D4): 8-16 day swing patterns
- Level 5 (D5): 16-32 day monthly cycles
- Approximation (A5): Underlying trend

Advantages over FFT:
- Time-frequency localization (FFT only frequency)
- Non-stationary signal handling
- Multi-scale pattern detection

References:
- Percival, D.B. & Walden, A.T. (2000). Wavelet Methods for Time Series Analysis
- Ramsey, J.B. (2002). Wavelets in Economics and Finance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class WaveletType(Enum):
    """Types of wavelet functions for financial analysis."""
    HAAR = "haar"  # Simplest, good for step changes
    DAUBECHIES_4 = "db4"  # Best for smooth signals
    DAUBECHIES_8 = "db8"  # Better frequency resolution
    SYMLET_4 = "sym4"  # Symmetric version of db4
    COIFLET_2 = "coif2"  # Good for continuous signals


class SignalComponent(Enum):
    """Components extracted from wavelet decomposition."""
    NOISE = "noise"  # D1 - 1-2 day (filter out)
    ULTRA_SHORT = "ultra_short"  # D2 - 2-4 day
    WEEKLY = "weekly"  # D3 - 4-8 day
    SWING = "swing"  # D4 - 8-16 day
    MONTHLY = "monthly"  # D5 - 16-32 day
    TREND = "trend"  # Approximation - underlying trend


@dataclass
class WaveletDecomposition:
    """Results from wavelet decomposition."""
    # Raw coefficients at each level
    approximation: np.ndarray  # Trend (A5)
    details: Dict[int, np.ndarray]  # D1-D5

    # Reconstructed signals
    denoised_signal: np.ndarray  # A5 + D5 + D4 (trend + monthly + swing)
    trend_signal: np.ndarray  # A5 only
    cycle_signal: np.ndarray  # D4 + D5 (swing + monthly)
    noise_signal: np.ndarray  # D1 + D2 + D3

    # Energy distribution
    energy_by_level: Dict[str, float]  # % of energy at each level
    signal_to_noise_ratio: float

    # Pattern detection
    breakout_detected: bool
    breakout_confidence: float
    pattern_type: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class MultiResolutionResult:
    """Results from multi-resolution analysis."""
    # Signals at each time scale
    ultra_short_trend: str  # 2-4 day trend
    weekly_trend: str  # 4-8 day trend
    swing_trend: str  # 8-16 day trend
    monthly_trend: str  # 16-32 day trend
    overall_trend: str  # Underlying trend

    # Alignment score (how aligned are different timeframes)
    trend_alignment: float  # 0-1, higher = all timeframes agree

    # Probability adjustments
    bullish_probability: float
    confidence_score: float

    # Trading signals
    signal: str  # BUY, SELL, HOLD
    optimal_timeframe: str  # Which timeframe is most reliable


class WaveletAnalyzer:
    """
    Wavelet-based multi-resolution analysis for stock prices.

    Uses Discrete Wavelet Transform (DWT) to decompose price series
    into different time scales, enabling:
    - Noise filtering
    - Multi-scale trend detection
    - Breakout identification
    - Pattern recognition at specific frequencies
    """

    def __init__(
        self,
        wavelet_type: WaveletType = WaveletType.DAUBECHIES_4,
        decomposition_level: int = 5,
        denoise_level: int = 3  # Filter out D1-D3 as noise
    ):
        self.wavelet_type = wavelet_type
        self.decomposition_level = decomposition_level
        self.denoise_level = denoise_level

    def decompose(self, signal: np.ndarray) -> WaveletDecomposition:
        """
        Decompose signal using Discrete Wavelet Transform.

        Uses custom implementation to avoid external dependencies.
        """
        # Ensure signal length is power of 2 (pad if necessary)
        n = len(signal)
        padded_length = 2 ** int(np.ceil(np.log2(n)))
        padded_signal = np.zeros(padded_length)
        padded_signal[:n] = signal

        # Get wavelet filter coefficients
        low_pass, high_pass = self._get_wavelet_filters()

        # Decompose
        approximation = padded_signal.copy()
        details = {}

        for level in range(1, self.decomposition_level + 1):
            # Single level decomposition
            approx_coeffs, detail_coeffs = self._single_level_dwt(
                approximation, low_pass, high_pass
            )
            details[level] = detail_coeffs
            approximation = approx_coeffs

        # Reconstruct signals
        # Denoised = Approximation + higher-level details (filter out D1-D3)
        denoised = self._reconstruct_partial(
            approximation, details,
            include_details=list(range(self.denoise_level + 1, self.decomposition_level + 1))
        )[:n]

        trend_only = self._reconstruct_partial(
            approximation, details,
            include_details=[]
        )[:n]

        cycle = self._reconstruct_partial(
            np.zeros_like(approximation), details,
            include_details=[4, 5]  # D4 + D5
        )[:n]

        noise = self._reconstruct_partial(
            np.zeros_like(approximation), details,
            include_details=[1, 2, 3]  # D1 + D2 + D3
        )[:n]

        # Calculate energy distribution
        total_energy = np.sum(signal ** 2)
        energy_by_level = {
            'approximation': np.sum(approximation ** 2) / total_energy if total_energy > 0 else 0,
        }
        for level, coeffs in details.items():
            energy_by_level[f'D{level}'] = np.sum(coeffs ** 2) / total_energy if total_energy > 0 else 0

        # Signal to noise ratio
        signal_energy = np.sum(denoised ** 2)
        noise_energy = np.sum(noise ** 2)
        snr = signal_energy / noise_energy if noise_energy > 0 else float('inf')

        # Detect breakout from detail coefficients
        breakout_detected, breakout_confidence, pattern_type = self._detect_breakout(details, signal)

        return WaveletDecomposition(
            approximation=approximation,
            details=details,
            denoised_signal=denoised,
            trend_signal=trend_only,
            cycle_signal=cycle,
            noise_signal=noise,
            energy_by_level=energy_by_level,
            signal_to_noise_ratio=snr,
            breakout_detected=breakout_detected,
            breakout_confidence=breakout_confidence,
            pattern_type=pattern_type
        )

    def _get_wavelet_filters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get low-pass and high-pass filter coefficients for wavelet.

        Returns:
            (low_pass, high_pass) filter coefficient arrays
        """
        if self.wavelet_type == WaveletType.HAAR:
            # Haar wavelet (simplest)
            low_pass = np.array([1, 1]) / np.sqrt(2)
            high_pass = np.array([1, -1]) / np.sqrt(2)

        elif self.wavelet_type == WaveletType.DAUBECHIES_4:
            # Daubechies-4 coefficients
            low_pass = np.array([
                0.4829629131445341,
                0.8365163037378079,
                0.2241438680420134,
                -0.1294095225512604
            ])
            high_pass = np.array([
                -0.1294095225512604,
                -0.2241438680420134,
                0.8365163037378079,
                -0.4829629131445341
            ])

        elif self.wavelet_type == WaveletType.DAUBECHIES_8:
            # Daubechies-8 coefficients
            low_pass = np.array([
                0.23037781330885523,
                0.7148465705525415,
                0.6308807679295904,
                -0.02798376941698385,
                -0.18703481171888114,
                0.030841381835986965,
                0.032883011666982945,
                -0.010597401784997278
            ])
            high_pass = np.array([
                -0.010597401784997278,
                -0.032883011666982945,
                0.030841381835986965,
                0.18703481171888114,
                -0.02798376941698385,
                -0.6308807679295904,
                0.7148465705525415,
                -0.23037781330885523
            ])

        else:
            # Default to Haar
            low_pass = np.array([1, 1]) / np.sqrt(2)
            high_pass = np.array([1, -1]) / np.sqrt(2)

        return low_pass, high_pass

    def _single_level_dwt(
        self, signal: np.ndarray,
        low_pass: np.ndarray,
        high_pass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-level discrete wavelet transform.

        Args:
            signal: Input signal
            low_pass: Low-pass filter coefficients
            high_pass: High-pass filter coefficients

        Returns:
            (approximation_coefficients, detail_coefficients)
        """
        n = len(signal)
        filter_len = len(low_pass)

        # Convolve and downsample by 2
        approx = np.convolve(signal, low_pass, mode='full')
        detail = np.convolve(signal, high_pass, mode='full')

        # Downsample by 2
        approx = approx[::2][:n//2]
        detail = detail[::2][:n//2]

        return approx, detail

    def _single_level_idwt(
        self, approx: np.ndarray,
        detail: np.ndarray,
        low_pass: np.ndarray,
        high_pass: np.ndarray
    ) -> np.ndarray:
        """
        Perform single-level inverse discrete wavelet transform.
        """
        # Upsample by 2 (insert zeros)
        n = len(approx) * 2
        approx_up = np.zeros(n)
        detail_up = np.zeros(n)
        approx_up[::2] = approx
        detail_up[::2] = detail

        # Reconstruction filters (reversed)
        low_recon = low_pass[::-1]
        high_recon = high_pass[::-1]

        # Convolve
        recon_approx = np.convolve(approx_up, low_recon, mode='same')
        recon_detail = np.convolve(detail_up, high_recon, mode='same')

        return recon_approx + recon_detail

    def _reconstruct_partial(
        self,
        approximation: np.ndarray,
        details: Dict[int, np.ndarray],
        include_details: List[int]
    ) -> np.ndarray:
        """
        Reconstruct signal from selected wavelet coefficients.
        """
        low_pass, high_pass = self._get_wavelet_filters()

        # Start with approximation
        result = approximation.copy()

        # Reconstruct level by level (from coarsest to finest)
        for level in range(self.decomposition_level, 0, -1):
            if level in include_details:
                detail = details[level]
            else:
                detail = np.zeros_like(details[level])

            result = self._single_level_idwt(result, detail, low_pass, high_pass)

        return result

    def _detect_breakout(
        self,
        details: Dict[int, np.ndarray],
        original_signal: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        Detect breakout patterns from wavelet coefficients.

        Breakouts are characterized by:
        - Sudden increase in D2 + D3 energy (short-term volatility spike)
        - Trend continuation in D4 + D5 (swing/monthly alignment)
        """
        # Get recent coefficients (last 10% of signal)
        n = len(original_signal)
        recent_idx = max(1, int(len(details[2]) * 0.1))

        # Energy in recent short-term details
        recent_d2_energy = np.sum(details[2][-recent_idx:] ** 2)
        recent_d3_energy = np.sum(details[3][-recent_idx:] ** 2)
        total_d2_energy = np.sum(details[2] ** 2)
        total_d3_energy = np.sum(details[3] ** 2)

        # Relative energy concentration in recent period
        if total_d2_energy > 0 and total_d3_energy > 0:
            energy_concentration = (
                (recent_d2_energy / total_d2_energy) +
                (recent_d3_energy / total_d3_energy)
            ) / 2 * (len(details[2]) / recent_idx)  # Normalize for time period
        else:
            energy_concentration = 0

        # Check direction from D4 + D5 (swing/monthly trend)
        d4_trend = np.mean(details[4][-recent_idx:])
        d5_trend = np.mean(details[5][-recent_idx:]) if 5 in details else 0

        # Determine pattern type
        if d4_trend + d5_trend > 0:
            pattern_type = "bullish"
        elif d4_trend + d5_trend < 0:
            pattern_type = "bearish"
        else:
            pattern_type = "neutral"

        # Breakout if energy concentration is significantly above average
        breakout_threshold = 2.0  # 2x normal concentration
        breakout_detected = energy_concentration > breakout_threshold
        breakout_confidence = min(1.0, energy_concentration / (breakout_threshold * 2))

        return breakout_detected, breakout_confidence, pattern_type


class MultiResolutionDecomposer:
    """
    Multi-resolution analysis for stock trading decisions.

    Analyzes price at multiple time scales to:
    - Identify trend alignment across timeframes
    - Calculate probability adjustments
    - Generate trading signals
    """

    def __init__(self):
        self.wavelet_analyzer = WaveletAnalyzer(
            wavelet_type=WaveletType.DAUBECHIES_4,
            decomposition_level=5
        )

    def analyze(self, prices: np.ndarray) -> MultiResolutionResult:
        """
        Perform multi-resolution analysis on price series.

        Args:
            prices: Array of closing prices

        Returns:
            MultiResolutionResult with trends at each scale
        """
        if len(prices) < 64:  # Need at least 64 points for 5-level decomposition
            return self._default_result()

        # Compute returns for analysis
        returns = np.diff(np.log(prices))

        # Decompose
        decomposition = self.wavelet_analyzer.decompose(returns)

        # Analyze trend at each level
        trends = self._analyze_trends(decomposition, returns)

        # Calculate trend alignment
        trend_values = {
            'ultra_short': 1 if trends['ultra_short'] == 'bullish' else (-1 if trends['ultra_short'] == 'bearish' else 0),
            'weekly': 1 if trends['weekly'] == 'bullish' else (-1 if trends['weekly'] == 'bearish' else 0),
            'swing': 1 if trends['swing'] == 'bullish' else (-1 if trends['swing'] == 'bearish' else 0),
            'monthly': 1 if trends['monthly'] == 'bullish' else (-1 if trends['monthly'] == 'bearish' else 0),
            'overall': 1 if trends['overall'] == 'bullish' else (-1 if trends['overall'] == 'bearish' else 0),
        }

        # Alignment: how much do timeframes agree
        total_direction = sum(trend_values.values())
        max_agreement = len(trend_values)
        trend_alignment = abs(total_direction) / max_agreement

        # Calculate probabilities
        base_prob = 0.5 + (total_direction / max_agreement) * 0.2

        # Adjust for SNR
        snr_factor = min(1.0, decomposition.signal_to_noise_ratio / 5.0)
        confidence = trend_alignment * snr_factor

        # Adjust for breakout
        if decomposition.breakout_detected:
            if decomposition.pattern_type == 'bullish':
                base_prob += 0.1 * decomposition.breakout_confidence
            elif decomposition.pattern_type == 'bearish':
                base_prob -= 0.1 * decomposition.breakout_confidence
            confidence = min(1.0, confidence + 0.1)

        # Determine signal
        if base_prob >= 0.6 and trend_alignment >= 0.6:
            signal = "BUY"
        elif base_prob <= 0.4 and trend_alignment >= 0.6:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Optimal timeframe (based on energy distribution)
        energy = decomposition.energy_by_level
        if energy.get('D4', 0) + energy.get('D5', 0) > 0.4:
            optimal_timeframe = "swing"
        elif energy.get('D3', 0) > 0.3:
            optimal_timeframe = "weekly"
        else:
            optimal_timeframe = "intraday"

        return MultiResolutionResult(
            ultra_short_trend=trends['ultra_short'],
            weekly_trend=trends['weekly'],
            swing_trend=trends['swing'],
            monthly_trend=trends['monthly'],
            overall_trend=trends['overall'],
            trend_alignment=trend_alignment,
            bullish_probability=base_prob,
            confidence_score=confidence,
            signal=signal,
            optimal_timeframe=optimal_timeframe
        )

    def _analyze_trends(
        self,
        decomposition: WaveletDecomposition,
        original_returns: np.ndarray
    ) -> Dict[str, str]:
        """
        Analyze trend direction at each time scale.
        """
        trends = {}

        # Ultra-short (D2): 2-4 day
        d2_recent = decomposition.details[2][-5:]
        trends['ultra_short'] = self._classify_trend(d2_recent)

        # Weekly (D3): 4-8 day
        d3_recent = decomposition.details[3][-5:]
        trends['weekly'] = self._classify_trend(d3_recent)

        # Swing (D4): 8-16 day
        d4_recent = decomposition.details[4][-3:]
        trends['swing'] = self._classify_trend(d4_recent)

        # Monthly (D5): 16-32 day
        if 5 in decomposition.details:
            d5_recent = decomposition.details[5][-2:]
            trends['monthly'] = self._classify_trend(d5_recent)
        else:
            trends['monthly'] = 'neutral'

        # Overall trend from approximation
        approx_recent = decomposition.approximation[-3:]
        if len(approx_recent) > 1:
            approx_trend = np.mean(np.diff(approx_recent))
            if approx_trend > 0:
                trends['overall'] = 'bullish'
            elif approx_trend < 0:
                trends['overall'] = 'bearish'
            else:
                trends['overall'] = 'neutral'
        else:
            trends['overall'] = 'neutral'

        return trends

    def _classify_trend(self, coefficients: np.ndarray) -> str:
        """Classify trend from wavelet coefficients."""
        if len(coefficients) == 0:
            return 'neutral'

        mean_coeff = np.mean(coefficients)
        std_coeff = np.std(coefficients) if len(coefficients) > 1 else 1.0

        if std_coeff == 0:
            std_coeff = 1.0

        # Threshold: mean must be significant relative to std
        threshold = 0.3 * std_coeff

        if mean_coeff > threshold:
            return 'bullish'
        elif mean_coeff < -threshold:
            return 'bearish'
        else:
            return 'neutral'

    def _default_result(self) -> MultiResolutionResult:
        """Return default result when insufficient data."""
        return MultiResolutionResult(
            ultra_short_trend='neutral',
            weekly_trend='neutral',
            swing_trend='neutral',
            monthly_trend='neutral',
            overall_trend='neutral',
            trend_alignment=0.0,
            bullish_probability=0.5,
            confidence_score=0.0,
            signal='HOLD',
            optimal_timeframe='unknown'
        )


def compute_wavelet_coherence(
    signal1: np.ndarray,
    signal2: np.ndarray,
    wavelet_type: WaveletType = WaveletType.DAUBECHIES_4
) -> float:
    """
    Compute wavelet coherence between two signals.

    Measures how correlated two signals are across different frequency bands.
    Used for sector correlation and lead-lag analysis.

    Returns:
        Coherence score 0-1 (1 = perfectly correlated at all scales)
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    analyzer = WaveletAnalyzer(wavelet_type=wavelet_type)

    decomp1 = analyzer.decompose(signal1)
    decomp2 = analyzer.decompose(signal2)

    # Compute correlation at each level
    correlations = []
    weights = []

    # Weight higher levels more (they contain more signal)
    level_weights = {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.3, 5: 0.4}

    for level in decomp1.details.keys():
        d1 = decomp1.details[level]
        d2 = decomp2.details[level]

        # Ensure same length
        min_len = min(len(d1), len(d2))
        d1 = d1[:min_len]
        d2 = d2[:min_len]

        if len(d1) > 0 and np.std(d1) > 0 and np.std(d2) > 0:
            corr = np.corrcoef(d1, d2)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
                weights.append(level_weights.get(level, 0.1))

    if not correlations:
        return 0.0

    # Weighted average correlation
    coherence = np.average(correlations, weights=weights)

    return float(coherence)
