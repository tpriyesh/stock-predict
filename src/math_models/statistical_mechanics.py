"""
Statistical Mechanics Model

Applies concepts from statistical physics to markets:
- Market Temperature: Volatility * Volume intensity
- Phase Transitions: Solid (consolidation) -> Liquid (normal) -> Gas (panic)
- Pressure: Trading intensity
- Boltzmann Distribution: Expected move sizes

Key Concepts:
- Low temperature (solid): Consolidation, breakout pending
- Normal temperature (liquid): Standard trading conditions
- High temperature (gas): High volatility, potential reversal
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class StatMechResult:
    """Result from statistical mechanics analysis."""
    temperature: float              # Market temperature
    temperature_percentile: float   # Where temp sits in history (0-100)
    phase: str                      # 'solid', 'liquid', 'gas'
    pressure: float                 # Trading pressure
    equilibrium_price: float        # Boltzmann-weighted fair price
    move_distribution: Dict[str, float]  # Probability of different move sizes
    phase_transition_risk: float    # Probability of phase change (0-1)
    heat_capacity: float            # Sensitivity to external shocks
    probability_score: float        # Contribution to final prediction (0-1)
    signal: str                     # 'COMPRESSION', 'NORMAL', 'OVERHEATED', 'PHASE_TRANSITION'
    reason: str                     # Human-readable explanation


class StatisticalMechanicsModel:
    """
    Applies statistical mechanics to market analysis.

    Thermodynamic Analogies:
    - Temperature = Volatility * sqrt(Volume_Ratio)
    - Pressure = Trading intensity = n * T / V (adapted from PV=nRT)
    - Energy = Variance of returns
    - Entropy = Disorder in price patterns
    """

    # Phase transition thresholds (percentile-based)
    PHASE_THRESHOLDS = {
        'solid': 25,    # Below 25th percentile
        'liquid': 75,   # 25th to 75th percentile
        'gas': 100      # Above 75th percentile
    }

    def __init__(
        self,
        temp_window: int = 20,
        history_window: int = 252,
        move_bins: list = None
    ):
        self.temp_window = temp_window
        self.history_window = history_window
        self.move_bins = move_bins or [0.5, 1.0, 2.0, 3.0, 5.0]

    def calculate_temperature(self, df: pd.DataFrame) -> float:
        """
        Calculate market temperature.

        Temperature = Volatility * sqrt(Volume_Ratio)

        High volatility + high volume = high temperature
        """
        returns = df['close'].pct_change().dropna()

        # Realized volatility (annualized)
        volatility = returns.tail(self.temp_window).std() * np.sqrt(252)

        # Volume ratio
        volume = df['volume']
        recent_vol = volume.tail(self.temp_window).mean()
        avg_vol = volume.mean()
        volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        # Temperature
        temperature = volatility * np.sqrt(volume_ratio)

        return temperature

    def calculate_temperature_percentile(self, df: pd.DataFrame) -> float:
        """Calculate where current temperature sits in history."""
        temperatures = []

        for i in range(self.temp_window, min(len(df), self.history_window)):
            subset = df.iloc[:i]
            temp = self.calculate_temperature(subset)
            if not np.isnan(temp):
                temperatures.append(temp)

        if len(temperatures) < 10:
            return 50.0

        current_temp = self.calculate_temperature(df)
        percentile = (np.array(temperatures) < current_temp).mean() * 100

        return percentile

    def classify_phase(self, temp_percentile: float) -> str:
        """Classify market phase based on temperature percentile."""
        if temp_percentile < self.PHASE_THRESHOLDS['solid']:
            return 'solid'
        elif temp_percentile < self.PHASE_THRESHOLDS['liquid']:
            return 'liquid'
        else:
            return 'gas'

    def calculate_pressure(self, df: pd.DataFrame) -> float:
        """
        Calculate trading pressure.

        Pressure = n * T / V (adapted from ideal gas law)
        n = number of periods (trades)
        T = temperature
        V = price range (high - low) as % of price
        """
        n = len(df.tail(self.temp_window))
        T = self.calculate_temperature(df)

        # Volume = price range as percentage
        high = df['high'].tail(self.temp_window).max()
        low = df['low'].tail(self.temp_window).min()
        mid = df['close'].tail(self.temp_window).mean()

        V = (high - low) / mid if mid > 0 else 0.01
        V = max(V, 0.005)  # Minimum range

        pressure = n * T / (V * 100)  # Scale appropriately

        return pressure

    def calculate_equilibrium_price(self, df: pd.DataFrame) -> float:
        """
        Calculate Boltzmann-weighted equilibrium price.

        States with lower "energy" (closer to mean) have higher probability.
        """
        temperature = self.calculate_temperature(df)
        temperature = max(temperature, 0.01)  # Avoid division by zero

        prices = df['close'].tail(self.temp_window).values
        mean_price = prices.mean()

        # Energy = deviation from mean (normalized)
        energies = np.abs(prices - mean_price) / mean_price

        # Boltzmann weights: exp(-E/kT), using T as our temperature
        weights = np.exp(-energies / temperature)
        weights = weights / weights.sum()

        # Weighted price
        equilibrium = np.sum(prices * weights)

        return equilibrium

    def calculate_move_distribution(
        self,
        temperature: float
    ) -> Dict[str, float]:
        """
        Calculate probability distribution of move sizes.

        Boltzmann distribution: P(move) ~ exp(-|move|/T)
        """
        probabilities = {}

        # Calculate unnormalized probabilities
        raw_probs = [np.exp(-m / (temperature + 0.1)) for m in self.move_bins]
        total = sum(raw_probs)

        for i, move in enumerate(self.move_bins):
            prob = raw_probs[i] / total if total > 0 else 0
            probabilities[f'{move}%'] = prob

        return probabilities

    def calculate_phase_transition_risk(self, df: pd.DataFrame) -> float:
        """
        Calculate probability of phase transition.

        Based on volatility of temperature itself.
        """
        if len(df) < 50:
            return 0.5

        # Calculate rolling temperature
        temperatures = []
        for i in range(self.temp_window, len(df)):
            subset = df.iloc[:i]
            temp = self.calculate_temperature(subset)
            if not np.isnan(temp):
                temperatures.append(temp)

        if len(temperatures) < 10:
            return 0.5

        # Coefficient of variation of temperature
        temp_mean = np.mean(temperatures)
        temp_std = np.std(temperatures)

        if temp_mean > 0:
            cv = temp_std / temp_mean
        else:
            cv = 0.5

        # Higher CV = higher transition risk
        risk = min(1.0, cv * 2)

        return risk

    def calculate_heat_capacity(self, df: pd.DataFrame) -> float:
        """
        Calculate heat capacity.

        How sensitive is the market to external shocks?
        High heat capacity = absorbs shocks easily
        Low heat capacity = reacts strongly to shocks
        """
        returns = df['close'].pct_change().dropna()

        # Measure response to large moves
        large_moves = returns[returns.abs() > returns.std() * 2]

        if len(large_moves) == 0:
            return 1.0

        # Following returns after large moves
        next_returns = returns.shift(-1)
        response = next_returns.loc[large_moves.index].dropna()

        if len(response) == 0:
            return 1.0

        # Heat capacity ~ 1 / response magnitude
        avg_response = response.abs().mean()
        heat_capacity = 1 / (avg_response * 100 + 0.1)

        return np.clip(heat_capacity, 0.1, 2.0)

    def get_signal_label(
        self,
        phase: str,
        transition_risk: float,
        pressure: float
    ) -> tuple:
        """Determine signal label and reason."""

        if transition_risk > 0.7:
            return 'PHASE_TRANSITION', f'High phase transition risk ({transition_risk:.0%})'

        if phase == 'solid':
            if pressure > 1.0:
                return 'COMPRESSION_HIGH_PRESSURE', 'Low volatility but high pressure - breakout likely'
            else:
                return 'COMPRESSION', 'Low volatility consolidation - wait for breakout'
        elif phase == 'gas':
            return 'OVERHEATED', 'High volatility (gas phase) - potential reversal zone'
        else:
            if pressure > 1.5:
                return 'NORMAL_HIGH_PRESSURE', 'Normal volatility with elevated pressure'
            else:
                return 'NORMAL', 'Normal trading conditions (liquid phase)'

    def score(self, df: pd.DataFrame) -> StatMechResult:
        """
        Calculate comprehensive statistical mechanics analysis.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            StatMechResult with all thermodynamic measures
        """
        if len(df) < 30:
            return self._default_result('Insufficient data for stat-mech analysis')

        # Calculate all metrics
        temperature = self.calculate_temperature(df)
        temp_percentile = self.calculate_temperature_percentile(df)
        phase = self.classify_phase(temp_percentile)
        pressure = self.calculate_pressure(df)
        equilibrium = self.calculate_equilibrium_price(df)
        move_dist = self.calculate_move_distribution(temperature)
        transition_risk = self.calculate_phase_transition_risk(df)
        heat_capacity = self.calculate_heat_capacity(df)

        # Signal and reason
        signal, reason = self.get_signal_label(phase, transition_risk, pressure)

        # Probability score
        # Solid phase (compression) = potential breakout
        # Liquid phase = normal trading
        # Gas phase = avoid or fade

        current_price = df['close'].iloc[-1]
        price_vs_equilibrium = (current_price - equilibrium) / equilibrium

        if phase == 'solid':
            # Compression - wait for breakout, slight bullish bias if below equilibrium
            if price_vs_equilibrium < -0.02:
                prob_score = 0.58
            elif price_vs_equilibrium > 0.02:
                prob_score = 0.48
            else:
                prob_score = 0.53
        elif phase == 'liquid':
            # Normal - use equilibrium as guide
            if price_vs_equilibrium < -0.02:
                prob_score = 0.55
            elif price_vs_equilibrium > 0.02:
                prob_score = 0.45
            else:
                prob_score = 0.52
        else:  # gas
            # Overheated - contrarian approach
            if price_vs_equilibrium < -0.03:
                prob_score = 0.58  # Oversold in panic
            elif price_vs_equilibrium > 0.03:
                prob_score = 0.42  # Overbought in panic
            else:
                prob_score = 0.48

        # Reduce confidence if high transition risk
        if transition_risk > 0.5:
            prob_score = 0.5 + (prob_score - 0.5) * (1 - transition_risk)

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return StatMechResult(
            temperature=temperature,
            temperature_percentile=temp_percentile,
            phase=phase,
            pressure=pressure,
            equilibrium_price=equilibrium,
            move_distribution=move_dist,
            phase_transition_risk=transition_risk,
            heat_capacity=heat_capacity,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> StatMechResult:
        """Return neutral result when analysis isn't possible."""
        return StatMechResult(
            temperature=0.1,
            temperature_percentile=50,
            phase='liquid',
            pressure=1.0,
            equilibrium_price=0,
            move_distribution={'0.5%': 0.4, '1.0%': 0.3, '2.0%': 0.2, '3.0%': 0.07, '5.0%': 0.03},
            phase_transition_risk=0.5,
            heat_capacity=1.0,
            probability_score=0.50,
            signal='STATMECH_UNKNOWN',
            reason=reason
        )
