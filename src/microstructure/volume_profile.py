"""
Volume Profile Analyzer

Analyzes volume distribution across price levels:
- Volume at Price (VAP): Volume distribution
- Point of Control (POC): Price with maximum volume
- Value Area: Price range containing 70% of volume

Trading Signals:
- Price above POC: Bullish bias
- Price below POC: Bearish bias
- Near Value Area boundary: Potential reversal
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class VolumeProfileResult:
    """Result from volume profile analysis."""
    poc_price: float                # Point of Control price
    value_area_high: float          # Upper 70% boundary
    value_area_low: float           # Lower 70% boundary
    current_price_position: str     # 'above_va', 'in_va', 'below_va'
    price_vs_poc: float             # % distance from POC
    volume_concentration: float     # How concentrated is volume (0-1)
    high_volume_levels: list        # Key price levels with high volume
    developing_poc: bool            # Is POC still shifting?
    probability_score: float        # Contribution to prediction (0-1)
    signal: str                     # 'ABOVE_POC', 'AT_POC', 'BELOW_POC'
    reason: str                     # Human-readable explanation


class VolumeProfileAnalyzer:
    """
    Analyzes volume profile from price data.

    Uses daily data to approximate volume profile.
    For intraday profiles, use intraday data.
    """

    def __init__(
        self,
        num_bins: int = 50,
        value_area_pct: float = 0.70,
        lookback: int = 20
    ):
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.lookback = lookback

    def calculate_volume_profile(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate volume at each price level.

        Uses typical price (H+L+C)/3 for each bar.
        Returns (price_bins, volume_by_bin)
        """
        # Use lookback period
        data = df.tail(self.lookback).copy()

        if len(data) < 5:
            return np.array([]), np.array([])

        # Calculate typical price
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3

        # Create price bins
        price_min = data['low'].min()
        price_max = data['high'].max()
        price_range = price_max - price_min

        if price_range < 0.01:
            return np.array([data['close'].mean()]), np.array([data['volume'].sum()])

        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Distribute volume across bins
        volume_by_bin = np.zeros(self.num_bins)

        for _, row in data.iterrows():
            # Each bar's volume is distributed across its range
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']

            # Find bins that this bar touches
            for i in range(self.num_bins):
                bin_low = bins[i]
                bin_high = bins[i + 1]

                # Check overlap
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)

                if overlap_high > overlap_low:
                    # Calculate fraction of bar in this bin
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        fraction = (overlap_high - overlap_low) / bar_range
                        volume_by_bin[i] += bar_volume * fraction

        return bin_centers, volume_by_bin

    def find_poc(
        self,
        price_bins: np.ndarray,
        volume_by_bin: np.ndarray
    ) -> float:
        """Find Point of Control (price with max volume)."""
        if len(volume_by_bin) == 0:
            return 0

        max_idx = np.argmax(volume_by_bin)
        return price_bins[max_idx]

    def calculate_value_area(
        self,
        price_bins: np.ndarray,
        volume_by_bin: np.ndarray,
        poc_idx: int
    ) -> Tuple[float, float]:
        """
        Calculate Value Area (70% of volume).

        Starts from POC and expands outward.
        """
        if len(volume_by_bin) == 0:
            return 0, 0

        total_volume = volume_by_bin.sum()
        target_volume = total_volume * self.value_area_pct

        # Start with POC bin
        included = np.zeros(len(volume_by_bin), dtype=bool)
        included[poc_idx] = True
        current_volume = volume_by_bin[poc_idx]

        # Expand outward
        low_idx = poc_idx
        high_idx = poc_idx

        while current_volume < target_volume:
            # Check which direction to expand
            can_go_low = low_idx > 0
            can_go_high = high_idx < len(volume_by_bin) - 1

            if not can_go_low and not can_go_high:
                break

            vol_low = volume_by_bin[low_idx - 1] if can_go_low else 0
            vol_high = volume_by_bin[high_idx + 1] if can_go_high else 0

            if vol_low >= vol_high and can_go_low:
                low_idx -= 1
                current_volume += volume_by_bin[low_idx]
            elif can_go_high:
                high_idx += 1
                current_volume += volume_by_bin[high_idx]
            else:
                low_idx -= 1
                current_volume += volume_by_bin[low_idx]

        # Return prices at boundaries
        val = price_bins[low_idx]
        vah = price_bins[high_idx]

        return val, vah

    def calculate_volume_concentration(
        self,
        volume_by_bin: np.ndarray
    ) -> float:
        """
        Calculate how concentrated volume is.

        High concentration = volume clustered at few levels
        Low concentration = volume spread evenly
        """
        if len(volume_by_bin) == 0 or volume_by_bin.sum() == 0:
            return 0.5

        # Normalize to probability distribution
        prob = volume_by_bin / volume_by_bin.sum()

        # Calculate entropy
        entropy = -np.sum(prob * np.log(prob + 1e-10))

        # Max entropy = log(n)
        max_entropy = np.log(len(volume_by_bin))

        # Concentration = 1 - normalized_entropy
        concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.5

        return concentration

    def get_high_volume_levels(
        self,
        price_bins: np.ndarray,
        volume_by_bin: np.ndarray,
        n_levels: int = 3
    ) -> list:
        """Get top N high volume price levels."""
        if len(volume_by_bin) == 0:
            return []

        top_idx = np.argsort(volume_by_bin)[-n_levels:][::-1]
        return [price_bins[i] for i in top_idx]

    def is_developing_poc(
        self,
        df: pd.DataFrame
    ) -> bool:
        """
        Check if POC is still developing (shifting recently).
        """
        if len(df) < 10:
            return True

        # Calculate POC for first half and second half
        mid = len(df) // 2

        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]

        bins1, vol1 = self.calculate_volume_profile(first_half)
        bins2, vol2 = self.calculate_volume_profile(second_half)

        if len(bins1) == 0 or len(bins2) == 0:
            return True

        poc1 = self.find_poc(bins1, vol1)
        poc2 = self.find_poc(bins2, vol2)

        # If POC shifted more than 2%, it's developing
        poc_shift = abs(poc2 - poc1) / poc1 if poc1 > 0 else 0

        return poc_shift > 0.02

    def get_signal_label(
        self,
        position: str,
        price_vs_poc: float,
        concentration: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if position == 'above_va':
            return 'EXTENDED_ABOVE', f'Price extended {abs(price_vs_poc):.1%} above POC - potential resistance'
        elif position == 'below_va':
            return 'EXTENDED_BELOW', f'Price extended {abs(price_vs_poc):.1%} below POC - potential support'
        elif price_vs_poc > 0.02:
            return 'ABOVE_POC', f'Price {price_vs_poc:.1%} above POC - bullish control'
        elif price_vs_poc < -0.02:
            return 'BELOW_POC', f'Price {abs(price_vs_poc):.1%} below POC - bearish control'
        else:
            return 'AT_POC', 'Price at Point of Control - balance zone'

    def analyze(self, df: pd.DataFrame) -> VolumeProfileResult:
        """
        Analyze volume profile.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            VolumeProfileResult with volume profile analysis
        """
        if len(df) < 10:
            return self._default_result('Insufficient data for volume profile')

        # Calculate profile
        price_bins, volume_by_bin = self.calculate_volume_profile(df)

        if len(price_bins) == 0:
            return self._default_result('Could not calculate volume profile')

        # Find POC
        poc_idx = np.argmax(volume_by_bin)
        poc_price = price_bins[poc_idx]

        # Calculate Value Area
        val, vah = self.calculate_value_area(price_bins, volume_by_bin, poc_idx)

        # Current price position
        current_price = df['close'].iloc[-1]

        if current_price > vah:
            position = 'above_va'
        elif current_price < val:
            position = 'below_va'
        else:
            position = 'in_va'

        # Distance from POC
        price_vs_poc = (current_price - poc_price) / poc_price

        # Volume concentration
        concentration = self.calculate_volume_concentration(volume_by_bin)

        # High volume levels
        high_vol_levels = self.get_high_volume_levels(price_bins, volume_by_bin)

        # Developing POC
        developing = self.is_developing_poc(df)

        # Signal and reason
        signal, reason = self.get_signal_label(position, price_vs_poc, concentration)

        # Probability score
        # Above POC = bullish
        # Below POC = bearish
        # In Value Area = neutral
        if position == 'above_va':
            # Extended above - potential pullback
            prob_score = 0.48
        elif position == 'below_va':
            # Extended below - potential bounce
            prob_score = 0.58
        elif price_vs_poc > 0.01:
            # Above POC
            prob_score = 0.55
        elif price_vs_poc < -0.01:
            # Below POC
            prob_score = 0.45
        else:
            prob_score = 0.50

        # Adjust for concentration
        if concentration > 0.6:
            # High concentration at key levels
            prob_score = 0.5 + (prob_score - 0.5) * 1.1

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return VolumeProfileResult(
            poc_price=poc_price,
            value_area_high=vah,
            value_area_low=val,
            current_price_position=position,
            price_vs_poc=price_vs_poc,
            volume_concentration=concentration,
            high_volume_levels=high_vol_levels,
            developing_poc=developing,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> VolumeProfileResult:
        """Return neutral result."""
        return VolumeProfileResult(
            poc_price=0,
            value_area_high=0,
            value_area_low=0,
            current_price_position='unknown',
            price_vs_poc=0,
            volume_concentration=0.5,
            high_volume_levels=[],
            developing_poc=True,
            probability_score=0.50,
            signal='PROFILE_UNKNOWN',
            reason=reason
        )
