"""
Spring Reversion Model (Hooke's Law)

Based on Hooke's Law: F = -k * x
The restoring force is proportional to displacement from equilibrium.

Financial Interpretation:
- Equilibrium = Fair value (weighted VWAP, SMAs, fundamental value)
- Displacement (x) = Current price deviation from equilibrium
- Spring constant (k) = Strength of mean reversion (varies by liquidity, volatility)
- Restoring Force = Expected move back toward equilibrium

Key Signals:
- High displacement + high k = Strong reversion expected
- Low k (during trends) = Reversion may not work
- Damping = How quickly price oscillates back
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class SpringResult:
    """Result from spring reversion analysis."""
    displacement: float            # % deviation from equilibrium
    equilibrium_price: float       # Calculated fair value
    spring_constant: float         # Strength of mean reversion (0-2)
    potential_energy: float        # 0.5 * k * x^2 (overextension measure)
    damping_factor: float          # How quickly price oscillates
    reversion_probability: float   # Probability of reverting to mean
    expected_reversion_pct: float  # Expected % move toward equilibrium
    time_to_equilibrium: int       # Expected periods to revert
    direction: int                 # +1 expect up, -1 expect down, 0 neutral
    probability_score: float       # Contribution to final prediction (0-1)
    signal: str                    # 'OVERSOLD_SPRING', 'OVERBOUGHT_SPRING', 'NEAR_EQUILIBRIUM'
    reason: str                    # Human-readable explanation


class SpringReversionModel:
    """
    Implements Hooke's Law mean reversion for stock prediction.

    The model calculates:
    1. Equilibrium price (multi-factor fair value)
    2. Displacement from equilibrium
    3. Spring constant (varies by market conditions)
    4. Expected reversion and timing
    """

    # Weights for equilibrium calculation
    EQUILIBRIUM_WEIGHTS = {
        'vwap': 0.35,
        'sma_20': 0.30,
        'sma_50': 0.25,
        'fundamental': 0.10
    }

    def __init__(
        self,
        sma_short: int = 20,
        sma_long: int = 50,
        vwap_period: int = 20,
        atr_period: int = 14
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.vwap_period = vwap_period
        self.atr_period = atr_period

    def calculate_equilibrium(
        self,
        df: pd.DataFrame,
        fundamental_value: Optional[float] = None
    ) -> float:
        """
        Calculate multi-factor equilibrium (fair) price.

        Components:
        - VWAP: Volume-weighted average price (institutional benchmark)
        - SMA20: Short-term trend center
        - SMA50: Medium-term trend center
        - Fundamental: P/E or DCF derived value (if available)
        """
        close = df['close']
        volume = df['volume']

        # VWAP (rolling for recent period)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * volume).tail(self.vwap_period).sum() / volume.tail(self.vwap_period).sum()

        # SMAs
        sma_20 = close.rolling(self.sma_short).mean().iloc[-1]
        sma_50 = close.rolling(self.sma_long).mean().iloc[-1] if len(df) >= self.sma_long else sma_20

        # Fundamental value (if not provided, use price mean as proxy)
        if fundamental_value is None:
            fundamental_value = close.mean()

        # Weighted equilibrium
        equilibrium = (
            self.EQUILIBRIUM_WEIGHTS['vwap'] * vwap +
            self.EQUILIBRIUM_WEIGHTS['sma_20'] * sma_20 +
            self.EQUILIBRIUM_WEIGHTS['sma_50'] * sma_50 +
            self.EQUILIBRIUM_WEIGHTS['fundamental'] * fundamental_value
        )

        return equilibrium

    def calculate_spring_constant(
        self,
        df: pd.DataFrame,
        regime: str = 'neutral'
    ) -> float:
        """
        Calculate spring constant k.

        Higher k = stronger mean reversion
        Lower k = weaker mean reversion (trends persist)

        Factors:
        - Liquidity: High liquidity -> higher k (faster reversion)
        - Volatility: High volatility -> lower k (wider oscillations)
        - Regime: Trending -> lower k (trends persist)
        """
        # Liquidity factor (normalized dollar volume)
        dollar_volume = df['close'] * df['volume']
        avg_dollar_vol = dollar_volume.mean()

        # Normalize to typical large cap volume (~100 Cr daily)
        liquidity_factor = min(avg_dollar_vol / 1e9, 2.0)
        liquidity_factor = max(liquidity_factor, 0.3)

        # Volatility factor (inverse relationship)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        volatility_factor = 1 / (1 + volatility)

        # Regime factor
        regime_factors = {
            'strong_bull': 0.3,   # Trends persist, weak reversion
            'bull': 0.5,
            'neutral': 1.0,       # Mean reversion works well
            'bear': 0.5,
            'strong_bear': 0.3,
            'ranging': 1.2,       # Range-bound = strong reversion
            'choppy': 0.7
        }
        regime_factor = regime_factors.get(regime.lower(), 1.0)

        k = liquidity_factor * volatility_factor * regime_factor
        return np.clip(k, 0.1, 2.0)

    def calculate_potential_energy(self, displacement: float, k: float) -> float:
        """
        Calculate potential energy = 0.5 * k * x^2

        Higher energy = more overextended = stronger expected reversion
        """
        return 0.5 * k * (displacement ** 2)

    def calculate_damping(self, df: pd.DataFrame) -> float:
        """
        Calculate damping coefficient.

        Damping determines how quickly oscillations decay.
        High damping = quick convergence to equilibrium
        Low damping = prolonged oscillation
        """
        atr = self._calculate_atr(df)
        current_price = df['close'].iloc[-1]

        # Damping ~ ATR/Price (volatility relative to price)
        damping = atr / current_price if current_price > 0 else 0.02
        return np.clip(damping, 0.005, 0.1)

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0

    def get_signal_label(
        self,
        displacement: float,
        k: float,
        reversion_prob: float
    ) -> tuple:
        """Determine signal label and reason."""

        if displacement < -0.05 and k > 0.5 and reversion_prob > 0.6:
            return 'OVERSOLD_SPRING', f'Price {abs(displacement)*100:.1f}% below equilibrium - spring loaded for bounce'
        elif displacement > 0.05 and k > 0.5 and reversion_prob > 0.6:
            return 'OVERBOUGHT_SPRING', f'Price {displacement*100:.1f}% above equilibrium - spring loaded for pullback'
        elif abs(displacement) < 0.02:
            return 'NEAR_EQUILIBRIUM', 'Price near fair value - no spring tension'
        elif k < 0.4:
            return 'WEAK_SPRING', 'Low spring constant - trend may persist over reversion'
        else:
            return 'MODERATE_TENSION', f'Moderate displacement ({displacement*100:.1f}%) - watch for reversion'

    def score(
        self,
        df: pd.DataFrame,
        regime: str = 'neutral',
        fundamental_value: Optional[float] = None
    ) -> SpringResult:
        """
        Calculate comprehensive spring reversion score.

        Args:
            df: DataFrame with OHLCV data
            regime: Market regime (affects spring constant)
            fundamental_value: Optional fair value from fundamentals

        Returns:
            SpringResult with all reversion metrics and probability score
        """
        if len(df) < 20:
            return self._default_result('Insufficient data for spring analysis')

        current_price = df['close'].iloc[-1]

        # Calculate components
        equilibrium = self.calculate_equilibrium(df, fundamental_value)
        displacement = (current_price - equilibrium) / equilibrium
        k = self.calculate_spring_constant(df, regime)
        energy = self.calculate_potential_energy(displacement, k)
        damping = self.calculate_damping(df)

        # Reversion probability (logistic function of displacement)
        # Higher displacement = higher probability of reversion
        reversion_prob = 1 / (1 + np.exp(-15 * abs(displacement)))
        reversion_prob = reversion_prob * k  # Adjust by spring constant
        reversion_prob = np.clip(reversion_prob, 0.3, 0.85)

        # Expected reversion (partial, not full - market doesn't fully correct)
        # Use spring force: F = -k * x, expect partial correction
        expected_reversion = -displacement * k * 0.4  # 40% of theoretical

        # Time to equilibrium (critically damped: t ~ 4/gamma)
        time_to_eq = int(4 / (damping + 0.01))
        time_to_eq = np.clip(time_to_eq, 2, 60)

        # Direction: expect move opposite to displacement
        if displacement > 0.03:
            direction = -1  # Expect down (overbought)
        elif displacement < -0.03:
            direction = 1   # Expect up (oversold)
        else:
            direction = 0   # Near equilibrium

        # Signal and reason
        signal, reason = self.get_signal_label(displacement, k, reversion_prob)

        # Probability score for BUY
        # Reference: Poterba & Summers (1988) "Mean Reversion in Stock Prices"
        # Oversold (negative displacement) = higher buy probability
        # Overbought (positive displacement) = lower buy probability
        # FIX: Made symmetric scoring for both directions
        if displacement < -0.03:
            # Oversold - good for buying
            # Use symmetric formula with same magnitude impact
            prob_score = 0.5 + (abs(displacement) * k * reversion_prob * 0.5)
        elif displacement > 0.03:
            # Overbought - bad for buying (symmetric impact)
            prob_score = 0.5 - (abs(displacement) * k * reversion_prob * 0.5)
        else:
            # Near equilibrium - neutral
            prob_score = 0.5

        # FIX: Symmetric clipping range centered at 0.5
        # Range: 0.25 to 0.75 gives equal room for bullish and bearish signals
        prob_score = np.clip(prob_score, 0.25, 0.75)

        return SpringResult(
            displacement=displacement,
            equilibrium_price=equilibrium,
            spring_constant=k,
            potential_energy=energy,
            damping_factor=damping,
            reversion_probability=reversion_prob,
            expected_reversion_pct=expected_reversion * 100,
            time_to_equilibrium=time_to_eq,
            direction=direction,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> SpringResult:
        """Return neutral result when calculation isn't possible."""
        return SpringResult(
            displacement=0,
            equilibrium_price=0,
            spring_constant=1.0,
            potential_energy=0,
            damping_factor=0.02,
            reversion_probability=0.5,
            expected_reversion_pct=0,
            time_to_equilibrium=10,
            direction=0,
            probability_score=0.5,
            signal='NEUTRAL',
            reason=reason
        )
