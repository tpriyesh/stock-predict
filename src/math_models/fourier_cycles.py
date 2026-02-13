"""
Fourier Cycle Detection Model

Uses Discrete Fourier Transform to detect hidden cycles in price data.

Key Concepts:
- Price movements often have cyclical components (weekly, monthly, quarterly)
- FFT reveals dominant frequencies in price data
- Current phase within cycle helps predict direction
- Low spectral entropy = more predictable cycles

Trading Signals:
- At cycle trough + rising phase = BUY
- At cycle peak + falling phase = SELL
- Multiple cycles aligned = stronger signal
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class FourierResult:
    """Result from Fourier cycle analysis."""
    dominant_periods: List[int]           # Top cycle periods (trading days)
    cycle_strengths: List[float]          # Strength of each cycle (0-1)
    current_phases: Dict[int, float]      # Phase for each period (-pi to pi)
    cycle_predictions: Dict[int, float]   # Direction prediction per cycle (-1 to 1)
    combined_signal: float                # Weighted combination (-1 to 1)
    spectral_entropy: float               # Randomness of spectrum (0-1)
    cycle_alignment: float                # How aligned cycles are (0-1)
    probability_score: float              # Contribution to final prediction (0-1)
    signal: str                           # 'CYCLE_BULLISH', 'CYCLE_BEARISH', 'CYCLES_MIXED'
    reason: str                           # Human-readable explanation


class FourierCycleModel:
    """
    Implements Fourier analysis for cycle detection.

    The model:
    1. Detrends price data
    2. Applies FFT to find dominant frequencies
    3. Calculates current phase within each cycle
    4. Predicts direction based on phase
    """

    # Known market cycles (in trading days)
    EXPECTED_CYCLES = {
        5: 'weekly',
        10: 'bi-weekly',
        21: 'monthly',
        42: 'bi-monthly',
        63: 'quarterly',
        126: 'semi-annual',
        252: 'annual'
    }

    def __init__(
        self,
        n_cycles: int = 3,
        min_period: int = 5,
        max_period: int = 126
    ):
        self.n_cycles = n_cycles
        self.min_period = min_period
        self.max_period = max_period

    def detrend(self, prices: pd.Series) -> np.ndarray:
        """
        Remove linear trend for better cycle detection.

        Detrending prevents low-frequency trend from dominating spectrum.
        """
        n = len(prices)
        x = np.arange(n)

        # Fit linear trend
        slope, intercept = np.polyfit(x, prices.values, 1)
        trend = slope * x + intercept

        # Subtract trend
        detrended = prices.values - trend

        return detrended

    def apply_window(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Hanning window to reduce spectral leakage.
        """
        window = np.hanning(len(data))
        return data * window

    def compute_fft(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT and power spectral density.

        Returns (frequencies, power_spectrum)
        """
        # Detrend
        detrended = self.detrend(prices)

        # Window
        windowed = self.apply_window(detrended)

        # FFT (real input)
        fft_result = np.fft.rfft(windowed)

        # Power spectrum
        psd = np.abs(fft_result) ** 2

        # Frequencies
        n = len(prices)
        freqs = np.fft.rfftfreq(n)

        return freqs, psd

    def find_dominant_cycles(self, prices: pd.Series) -> Tuple[List[int], List[float]]:
        """
        Find dominant cycles (periods) in the data.

        Returns (periods, normalized_strengths)
        """
        freqs, psd = self.compute_fft(prices)
        n = len(prices)

        # Convert to periods and filter valid range
        period_power = []

        for i in range(1, len(freqs)):  # Skip DC component
            if freqs[i] > 0:
                period = int(round(1 / freqs[i]))

                # Filter by valid period range
                if self.min_period <= period <= min(self.max_period, n // 2):
                    period_power.append((period, psd[i]))

        if not period_power:
            return [21], [0.5]  # Default to monthly

        # Sort by power
        period_power.sort(key=lambda x: x[1], reverse=True)

        # Take top N unique periods
        seen_periods = set()
        top_periods = []
        top_powers = []

        for period, power in period_power:
            # Avoid very similar periods
            if not any(abs(period - p) < 3 for p in seen_periods):
                top_periods.append(period)
                top_powers.append(power)
                seen_periods.add(period)

                if len(top_periods) >= self.n_cycles:
                    break

        # Normalize powers to sum to 1
        total_power = sum(top_powers)
        if total_power > 0:
            normalized = [p / total_power for p in top_powers]
        else:
            normalized = [1.0 / len(top_powers)] * len(top_powers)

        return top_periods, normalized

    def calculate_phase(self, prices: pd.Series, period: int) -> float:
        """
        Calculate current phase within a cycle.

        Phase: -pi (trough) to 0 (rising) to pi (peak) to -pi (falling)
        """
        n = len(prices)

        # Where we are in the current cycle
        position_in_cycle = n % period
        phase = (position_in_cycle / period) * 2 * np.pi - np.pi

        return phase

    def predict_from_phase(self, phase: float) -> float:
        """
        Predict direction based on phase.

        Phase near -pi (trough): Expect up (+1)
        Phase near 0 (rising): Expect continue up (+0.5)
        Phase near pi (peak): Expect down (-1)

        Uses -sin(phase) since sin is 0 at peak/trough
        """
        return -np.sin(phase)

    def calculate_spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Calculate spectral entropy.

        Low entropy = power concentrated in few frequencies = predictable
        High entropy = power spread across frequencies = random
        """
        # Normalize to probability distribution
        psd_positive = psd[psd > 0]
        if len(psd_positive) == 0:
            return 1.0

        psd_norm = psd_positive / psd_positive.sum()

        # Shannon entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

        # Normalize (max entropy = log2(n))
        max_entropy = np.log2(len(psd_norm))
        normalized = entropy / max_entropy if max_entropy > 0 else 1.0

        return np.clip(normalized, 0, 1)

    def calculate_cycle_alignment(
        self,
        predictions: Dict[int, float],
        strengths: List[float]
    ) -> float:
        """
        Calculate how aligned the cycles are.

        All cycles predicting same direction = high alignment
        """
        if not predictions:
            return 0.5

        pred_values = list(predictions.values())
        signs = [1 if p > 0 else -1 for p in pred_values]

        # What fraction agree with weighted majority
        weighted_sum = sum(s * w for s, w in zip(signs, strengths))
        majority = 1 if weighted_sum > 0 else -1

        # Weighted agreement
        agreement = sum(w for s, w in zip(signs, strengths) if s == majority)

        return agreement

    def get_signal_label(
        self,
        combined_signal: float,
        alignment: float,
        entropy: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if entropy > 0.85:
            return 'CYCLES_NOISY', 'High spectral entropy - cycles not reliable'

        if alignment < 0.5:
            return 'CYCLES_MIXED', 'Cycles giving conflicting signals'

        if combined_signal > 0.3 and alignment > 0.7:
            return 'CYCLE_BULLISH', f'Cycles aligned bullish (signal: {combined_signal:.2f})'
        elif combined_signal < -0.3 and alignment > 0.7:
            return 'CYCLE_BEARISH', f'Cycles aligned bearish (signal: {combined_signal:.2f})'
        elif combined_signal > 0.1:
            return 'CYCLE_WEAK_BULLISH', 'Weak bullish cycle signal'
        elif combined_signal < -0.1:
            return 'CYCLE_WEAK_BEARISH', 'Weak bearish cycle signal'
        else:
            return 'CYCLES_NEUTRAL', 'Cycles at neutral phase'

    def score(self, df: pd.DataFrame) -> FourierResult:
        """
        Calculate comprehensive Fourier cycle score.

        Args:
            df: DataFrame with OHLCV data (minimum 60 days)

        Returns:
            FourierResult with cycle analysis
        """
        if len(df) < 60:
            return self._default_result('Insufficient data for cycle analysis')

        prices = df['close']

        # Find dominant cycles
        periods, strengths = self.find_dominant_cycles(prices)

        # Calculate phase and predictions for each cycle
        current_phases = {}
        predictions = {}

        for period in periods:
            phase = self.calculate_phase(prices, period)
            current_phases[period] = phase
            predictions[period] = self.predict_from_phase(phase)

        # Combined signal (weighted by strength)
        combined = sum(
            predictions[p] * strengths[i]
            for i, p in enumerate(periods)
        )

        # Spectral entropy
        _, psd = self.compute_fft(prices)
        entropy = self.calculate_spectral_entropy(psd)

        # Cycle alignment
        alignment = self.calculate_cycle_alignment(predictions, strengths)

        # Signal and reason
        signal, reason = self.get_signal_label(combined, alignment, entropy)

        # Probability score
        # Low entropy + high alignment + bullish signal = higher probability
        if entropy < 0.75 and alignment > 0.6:
            # Cycles are meaningful
            base_prob = 0.55
            signal_contribution = combined * 0.15  # -0.15 to +0.15
            alignment_bonus = (alignment - 0.5) * 0.1  # Up to +0.05
            prob_score = base_prob + signal_contribution + alignment_bonus
        else:
            # Cycles not reliable
            prob_score = 0.50

        prob_score = np.clip(prob_score, 0.35, 0.70)

        return FourierResult(
            dominant_periods=periods,
            cycle_strengths=strengths,
            current_phases=current_phases,
            cycle_predictions=predictions,
            combined_signal=combined,
            spectral_entropy=entropy,
            cycle_alignment=alignment,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> FourierResult:
        """Return neutral result when analysis isn't possible."""
        return FourierResult(
            dominant_periods=[21],
            cycle_strengths=[1.0],
            current_phases={21: 0},
            cycle_predictions={21: 0},
            combined_signal=0,
            spectral_entropy=1.0,
            cycle_alignment=0.5,
            probability_score=0.50,
            signal='CYCLES_UNKNOWN',
            reason=reason
        )
