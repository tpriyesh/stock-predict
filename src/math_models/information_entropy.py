"""
Information Entropy Model

Uses Shannon entropy and related measures to quantify:
- Predictability of price movements
- Information content in price patterns
- How much past predicts future

Key Concepts:
- High entropy = more random = harder to predict
- Low entropy = more structured = more predictable
- Mutual information = how much past predicts future
- Conditional entropy = remaining uncertainty after observing signals
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class EntropyResult:
    """Result from information entropy analysis."""
    return_entropy: float           # Entropy of return distribution (bits)
    conditional_entropy: float      # Entropy given past returns (bits)
    mutual_information: float       # How much past predicts future (bits)
    predictability_index: float     # 1 - normalized_entropy (0-1)
    sample_entropy: float           # Complexity of time series
    regime_entropy: float           # Entropy of up/down patterns
    information_ratio: float        # Mutual info / Total entropy
    probability_score: float        # Contribution to final prediction (0-1)
    signal: str                     # 'HIGH_PREDICTABILITY', 'LOW_PREDICTABILITY'
    reason: str                     # Human-readable explanation


class InformationEntropyModel:
    """
    Calculates information-theoretic measures for predictability.

    Methods:
    1. Shannon entropy of discretized returns
    2. Conditional entropy (prediction improvement from past)
    3. Mutual information (shared information between past/future)
    4. Sample entropy (time series complexity)
    """

    def __init__(
        self,
        n_bins: int = 20,
        lag: int = 1,
        sample_entropy_m: int = 2,
        sample_entropy_r: float = 0.2
    ):
        self.n_bins = n_bins
        self.lag = lag
        self.sample_entropy_m = sample_entropy_m
        self.sample_entropy_r = sample_entropy_r

    def discretize_returns(self, returns: pd.Series) -> np.ndarray:
        """
        Discretize returns into bins using percentiles.

        Percentile-based bins are more robust than equal-width.
        """
        returns_clean = returns.dropna()

        if len(returns_clean) < 10:
            return np.array([])

        # Create percentile-based bins
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(returns_clean, percentiles)

        # Ensure unique bins
        bins = np.unique(bins)
        if len(bins) < 3:
            return np.digitize(returns_clean.values, [-0.01, 0, 0.01])

        digitized = np.digitize(returns_clean.values, bins[:-1])
        return digitized

    def calculate_entropy(self, sequence: np.ndarray) -> float:
        """
        Calculate Shannon entropy in bits.

        H(X) = -sum(p(x) * log2(p(x)))
        """
        if len(sequence) == 0:
            return 0

        _, counts = np.unique(sequence, return_counts=True)
        probs = counts / len(sequence)

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def calculate_conditional_entropy(self, returns: pd.Series) -> float:
        """
        Calculate H(X_t | X_{t-1}).

        Conditional entropy = uncertainty remaining after knowing past.
        """
        digitized = self.discretize_returns(returns)
        n = len(digitized)

        if n < 10:
            return 0.5

        # Build transition counts
        transitions = {}

        for i in range(n - self.lag):
            prev = digitized[i]
            curr = digitized[i + self.lag]

            if prev not in transitions:
                transitions[prev] = {}
            if curr not in transitions[prev]:
                transitions[prev][curr] = 0
            transitions[prev][curr] += 1

        # Calculate conditional entropy
        cond_entropy = 0.0
        total = n - self.lag

        for prev, next_counts in transitions.items():
            prev_count = sum(next_counts.values())

            for curr, count in next_counts.items():
                p_joint = count / total
                p_cond = count / prev_count

                if p_cond > 0:
                    cond_entropy -= p_joint * np.log2(p_cond)

        return cond_entropy

    def calculate_mutual_information(
        self,
        entropy: float,
        conditional_entropy: float
    ) -> float:
        """
        Calculate mutual information.

        I(X_t; X_{t-1}) = H(X_t) - H(X_t | X_{t-1})

        How much knowing the past reduces uncertainty about future.
        """
        mi = entropy - conditional_entropy
        return max(0, mi)  # Should be non-negative

    def calculate_sample_entropy(self, returns: pd.Series) -> float:
        """
        Calculate Sample Entropy (SampEn).

        Measures complexity/regularity of time series.
        Lower = more regular/predictable.
        """
        data = returns.dropna().values
        n = len(data)

        if n < 50:
            return 0.5

        m = self.sample_entropy_m
        r = self.sample_entropy_r * np.std(data)

        def count_matches(template_length: int) -> int:
            count = 0
            for i in range(n - template_length):
                for j in range(i + 1, n - template_length):
                    # Check if templates match within tolerance r
                    template_i = data[i:i + template_length]
                    template_j = data[j:j + template_length]

                    if np.max(np.abs(template_i - template_j)) < r:
                        count += 1
            return count

        # Count matches for length m and m+1
        b = count_matches(m)
        a = count_matches(m + 1)

        if b == 0:
            return 0

        if a == 0:
            return np.log(b)  # Approximate

        return -np.log(a / b)

    def calculate_regime_entropy(self, returns: pd.Series) -> float:
        """
        Calculate entropy of up/down regime sequence.

        Low entropy = persistent regimes (trending)
        High entropy = frequent switching (choppy)
        """
        returns_clean = returns.dropna()

        if len(returns_clean) < 10:
            return 0.5

        # Convert to up/down sequence
        signs = np.sign(returns_clean.values)
        signs[signs == 0] = 1  # Treat flat as up

        # Count transitions
        transitions = np.diff(signs)
        n_transitions = np.sum(transitions != 0)
        n_total = len(signs) - 1

        if n_total == 0:
            return 0.5

        # Transition probability
        p_switch = n_transitions / n_total
        p_stay = 1 - p_switch

        # Binary entropy
        if p_switch <= 0 or p_switch >= 1:
            regime_entropy = 0
        else:
            regime_entropy = -(p_switch * np.log2(p_switch) +
                              p_stay * np.log2(p_stay))

        return regime_entropy

    def get_signal_label(
        self,
        predictability: float,
        mutual_info: float,
        entropy: float
    ) -> tuple:
        """Determine signal label and reason."""

        if predictability > 0.3 and mutual_info > 0.1:
            return 'HIGH_PREDICTABILITY', f'Low entropy ({entropy:.2f} bits), high mutual info ({mutual_info:.2f} bits)'
        elif predictability > 0.2:
            return 'MODERATE_PREDICTABILITY', f'Moderate structure in returns'
        elif mutual_info > 0.05:
            return 'WEAK_PREDICTABILITY', f'Weak but present autocorrelation'
        else:
            return 'LOW_PREDICTABILITY', f'High entropy ({entropy:.2f} bits) - near random'

    def score(self, df: pd.DataFrame) -> EntropyResult:
        """
        Calculate comprehensive entropy analysis.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            EntropyResult with all entropy measures
        """
        if len(df) < 30:
            return self._default_result('Insufficient data for entropy analysis')

        returns = df['close'].pct_change()

        # Calculate all entropy measures
        digitized = self.discretize_returns(returns)

        if len(digitized) == 0:
            return self._default_result('Could not discretize returns')

        return_entropy = self.calculate_entropy(digitized)
        cond_entropy = self.calculate_conditional_entropy(returns)
        mutual_info = self.calculate_mutual_information(return_entropy, cond_entropy)

        # Normalized predictability
        max_entropy = np.log2(self.n_bins)
        normalized_entropy = return_entropy / max_entropy if max_entropy > 0 else 1
        predictability = 1 - normalized_entropy

        # Sample entropy (use subset for speed)
        sample_ent = self.calculate_sample_entropy(returns.tail(100))

        # Regime entropy
        regime_ent = self.calculate_regime_entropy(returns)

        # Information ratio
        info_ratio = mutual_info / return_entropy if return_entropy > 0 else 0

        # Signal and reason
        signal, reason = self.get_signal_label(predictability, mutual_info, return_entropy)

        # Probability score
        # Higher predictability = more confidence in predictions
        # Check recent trend to determine direction
        recent_return = returns.tail(5).sum()

        if predictability > 0.25 and mutual_info > 0.08:
            # High predictability - can trust signals more
            if recent_return > 0:
                prob_score = 0.55 + predictability * 0.15
            else:
                prob_score = 0.45 - predictability * 0.10
        elif predictability > 0.15:
            # Moderate predictability
            prob_score = 0.52 if recent_return > 0 else 0.48
        else:
            # Low predictability - stay neutral
            prob_score = 0.50

        prob_score = np.clip(prob_score, 0.35, 0.68)

        return EntropyResult(
            return_entropy=return_entropy,
            conditional_entropy=cond_entropy,
            mutual_information=mutual_info,
            predictability_index=predictability,
            sample_entropy=sample_ent,
            regime_entropy=regime_ent,
            information_ratio=info_ratio,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> EntropyResult:
        """Return neutral result when analysis isn't possible."""
        return EntropyResult(
            return_entropy=0,
            conditional_entropy=0,
            mutual_information=0,
            predictability_index=0,
            sample_entropy=0.5,
            regime_entropy=0.5,
            information_ratio=0,
            probability_score=0.50,
            signal='ENTROPY_UNKNOWN',
            reason=reason
        )
