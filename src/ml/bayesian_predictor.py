"""
Hierarchical Bayesian Predictor

Implements Bayesian inference for stock prediction:
- Hierarchical priors (market -> sector -> stock)
- Signal-based likelihood updates
- Posterior probability with uncertainty quantification

Key Concepts:
- Prior: What we believe before seeing today's signals
- Likelihood: How likely are current signals given the outcome
- Posterior: Updated belief after seeing signals
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BayesianResult:
    """Result from Bayesian prediction."""
    # Point estimates
    posterior_mean: float           # E[theta] - expected probability
    posterior_mode: float           # MAP estimate
    posterior_median: float         # Median estimate

    # Uncertainty quantification
    credible_interval_95: Tuple[float, float]
    credible_interval_80: Tuple[float, float]
    posterior_std: float

    # Prior information
    prior_alpha: float
    prior_beta: float
    prior_mean: float
    prior_source: str               # 'market', 'sector', 'stock'

    # Likelihood information
    likelihood_ratio: float         # P(signals|up) / P(signals|down)
    signal_contributions: Dict[str, float]
    most_influential_signal: str

    # Model quality
    effective_sample_size: float
    prior_weight: float             # How much prior influenced result

    probability_score: float        # Contribution to prediction (0-1)
    signal: str                     # 'HIGH_CONFIDENCE', 'MODERATE', 'LOW_CONFIDENCE'
    reason: str                     # Human-readable explanation


class BayesianPredictor:
    """
    Hierarchical Bayesian predictor using Beta-Binomial model.

    Hierarchy:
    1. Market prior: Based on overall market win rate
    2. Sector prior: Shrinkage toward market prior
    3. Stock prior: Shrinkage toward sector prior
    4. Posterior: After observing current signals

    Uses conjugate Beta prior for tractable inference.
    """

    # Default priors (calibrated from historical data)
    MARKET_PRIOR = {'alpha': 52, 'beta': 48}  # 52% base win rate

    SECTOR_PRIORS = {
        'IT': {'alpha': 54, 'beta': 46},
        'BANKING': {'alpha': 51, 'beta': 49},
        'AUTO': {'alpha': 52, 'beta': 48},
        'PHARMA': {'alpha': 53, 'beta': 47},
        'FMCG': {'alpha': 55, 'beta': 45},
        'METAL': {'alpha': 48, 'beta': 52},
        'ENERGY': {'alpha': 50, 'beta': 50},
        'DEFAULT': {'alpha': 52, 'beta': 48}
    }

    # Signal accuracies (from backtesting)
    SIGNAL_ACCURACIES = {
        'rsi_oversold': 0.58,
        'rsi_overbought': 0.42,
        'macd_bullish': 0.55,
        'macd_bearish': 0.45,
        'price_above_sma50': 0.56,
        'price_below_sma50': 0.44,
        'volume_surge': 0.54,
        'golden_cross': 0.60,
        'death_cross': 0.40,
        'bullish_engulfing': 0.57,
        'bearish_engulfing': 0.43,
        'near_support': 0.58,
        'near_resistance': 0.42,
        'momentum_positive': 0.55,
        'momentum_negative': 0.45
    }

    def __init__(
        self,
        shrinkage_factor: float = 0.3,
        min_observations: int = 20
    ):
        self.shrinkage_factor = shrinkage_factor
        self.min_observations = min_observations

    def get_market_prior(self) -> Tuple[float, float]:
        """Get market-level prior."""
        return self.MARKET_PRIOR['alpha'], self.MARKET_PRIOR['beta']

    def get_sector_prior(
        self,
        sector: str,
        n_sector_obs: int = 100
    ) -> Tuple[float, float, str]:
        """
        Get sector-level prior with shrinkage toward market.

        Shrinkage increases with fewer observations.
        """
        market_alpha, market_beta = self.get_market_prior()

        # Get sector parameters
        sector_upper = sector.upper() if sector else 'DEFAULT'
        sector_params = self.SECTOR_PRIORS.get(sector_upper, self.SECTOR_PRIORS['DEFAULT'])

        # Shrinkage weight (more observations = less shrinkage)
        w = min(1.0, self.min_observations / max(n_sector_obs, 1))

        # Shrunk parameters
        alpha = w * market_alpha + (1 - w) * sector_params['alpha']
        beta = w * market_beta + (1 - w) * sector_params['beta']

        source = f'sector_{sector_upper}' if w < 0.5 else 'market_shrunk'

        return alpha, beta, source

    def get_stock_prior(
        self,
        symbol: str,
        sector: str,
        stock_history: Optional[Dict] = None
    ) -> Tuple[float, float, str]:
        """
        Get stock-level prior with shrinkage toward sector.

        If stock history available, use empirical Bayes.
        """
        sector_alpha, sector_beta, _ = self.get_sector_prior(sector)

        if stock_history is None or 'wins' not in stock_history:
            return sector_alpha, sector_beta, f'sector_{sector}'

        wins = stock_history.get('wins', 0)
        losses = stock_history.get('losses', 0)
        total = wins + losses

        if total < self.min_observations:
            # Not enough data, use sector prior with slight adjustment
            w = self.min_observations / max(total, 1)
            alpha = w * sector_alpha + (1 - w) * (sector_alpha + wins)
            beta = w * sector_beta + (1 - w) * (sector_beta + losses)
            return alpha, beta, 'sector_shrunk'

        # Enough data, use stock-specific with light shrinkage
        w = 0.2  # 20% shrinkage toward sector
        alpha = w * sector_alpha + (1 - w) * (wins + 1)
        beta = w * sector_beta + (1 - w) * (losses + 1)

        return alpha, beta, f'stock_{symbol}'

    def extract_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Extract trading signals from price data.

        Returns dict of {signal_name: is_bullish}
        """
        signals = {}

        if len(df) < 50:
            return signals

        close = df['close'].iloc[-1]
        volume = df['volume'].iloc[-1]

        # Calculate indicators
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50

        # RSI
        rsi = self._calculate_rsi(df['close'])

        # MACD
        ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        macd_prev = df['close'].ewm(span=12).mean().iloc[-2] - df['close'].ewm(span=26).mean().iloc[-2]

        # Volume
        volume_sma = df['volume'].rolling(20).mean().iloc[-1]

        # Generate signals
        if rsi < 30:
            signals['rsi_oversold'] = True
        elif rsi > 70:
            signals['rsi_overbought'] = True

        if macd > 0 and macd_prev <= 0:
            signals['macd_bullish'] = True
        elif macd < 0 and macd_prev >= 0:
            signals['macd_bearish'] = True

        if close > sma_50:
            signals['price_above_sma50'] = True
        else:
            signals['price_below_sma50'] = True

        if volume > 1.5 * volume_sma:
            signals['volume_surge'] = True

        # Golden/Death cross
        if len(df) >= 200:
            sma_50_prev = df['close'].rolling(50).mean().iloc[-2]
            sma_200_prev = df['close'].rolling(200).mean().iloc[-2]

            if sma_50 > sma_200 and sma_50_prev <= sma_200_prev:
                signals['golden_cross'] = True
            elif sma_50 < sma_200 and sma_50_prev >= sma_200_prev:
                signals['death_cross'] = True

        # Momentum
        returns_5d = (close / df['close'].iloc[-6] - 1) if len(df) > 5 else 0
        if returns_5d > 0.02:
            signals['momentum_positive'] = True
        elif returns_5d < -0.02:
            signals['momentum_negative'] = True

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        return 100 - (100 / (1 + rs))

    def compute_likelihood_ratio(
        self,
        signals: Dict[str, bool]
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Compute likelihood ratio P(signals|up) / P(signals|down).

        Returns (ratio, contributions, most_influential)
        """
        log_lik_up = 0.0
        log_lik_down = 0.0
        contributions = {}

        for signal_name, is_present in signals.items():
            if not is_present:
                continue

            accuracy = self.SIGNAL_ACCURACIES.get(signal_name, 0.5)

            # Bullish signals (accuracy > 0.5 when up)
            if 'oversold' in signal_name or 'bullish' in signal_name or \
               'above' in signal_name or 'surge' in signal_name or \
               'golden' in signal_name or 'support' in signal_name or \
               'positive' in signal_name:
                log_lik_up += np.log(accuracy + 1e-10)
                log_lik_down += np.log(1 - accuracy + 1e-10)
                contributions[signal_name] = accuracy
            # Bearish signals
            else:
                log_lik_up += np.log(1 - accuracy + 1e-10)
                log_lik_down += np.log(accuracy + 1e-10)
                contributions[signal_name] = 1 - accuracy

        if not contributions:
            return 1.0, {}, ''

        # Most influential signal
        most_influential = max(contributions, key=lambda k: abs(contributions[k] - 0.5))

        # Likelihood ratio
        ratio = np.exp(log_lik_up - log_lik_down)

        return ratio, contributions, most_influential

    def compute_posterior(
        self,
        prior_alpha: float,
        prior_beta: float,
        likelihood_ratio: float
    ) -> Tuple[float, float]:
        """
        Compute posterior parameters using Bayes' theorem.

        For Beta-Binomial with single observation:
        posterior_alpha = prior_alpha + (lr / (1 + lr))
        posterior_beta = prior_beta + (1 / (1 + lr))

        This is an approximation using likelihood ratio as pseudo-observation.
        """
        # Convert likelihood ratio to pseudo-observation
        pseudo_success = likelihood_ratio / (1 + likelihood_ratio)
        pseudo_failure = 1 - pseudo_success

        # Scale by effective observations (based on signal count)
        n_effective = min(10, max(1, np.log(likelihood_ratio + 1) * 2))

        post_alpha = prior_alpha + pseudo_success * n_effective
        post_beta = prior_beta + pseudo_failure * n_effective

        return post_alpha, post_beta

    def get_signal_label(
        self,
        posterior_mean: float,
        posterior_std: float,
        prior_weight: float
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        # Confidence based on uncertainty
        if posterior_std < 0.05:
            confidence = 'HIGH_CONFIDENCE'
        elif posterior_std < 0.10:
            confidence = 'MODERATE'
        else:
            confidence = 'LOW_CONFIDENCE'

        if posterior_mean > 0.6:
            return f'{confidence}_BULLISH', f'Posterior mean {posterior_mean:.0%} with std {posterior_std:.2f}'
        elif posterior_mean < 0.4:
            return f'{confidence}_BEARISH', f'Posterior mean {posterior_mean:.0%} with std {posterior_std:.2f}'
        else:
            return f'{confidence}_NEUTRAL', f'Posterior mean {posterior_mean:.0%} - near neutral'

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = '',
        sector: str = '',
        stock_history: Optional[Dict] = None
    ) -> BayesianResult:
        """
        Make Bayesian prediction for stock.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            sector: Stock sector
            stock_history: Optional dict with 'wins' and 'losses'

        Returns:
            BayesianResult with posterior prediction
        """
        if len(df) < 30:
            return self._default_result('Insufficient data for Bayesian prediction')

        # Get prior
        prior_alpha, prior_beta, prior_source = self.get_stock_prior(
            symbol, sector, stock_history
        )
        prior_mean = prior_alpha / (prior_alpha + prior_beta)

        # Extract signals and compute likelihood
        signals = self.extract_signals(df)
        lr, contributions, most_influential = self.compute_likelihood_ratio(signals)

        # Compute posterior
        post_alpha, post_beta = self.compute_posterior(prior_alpha, prior_beta, lr)

        # Posterior statistics
        posterior_mean = post_alpha / (post_alpha + post_beta)
        posterior_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else posterior_mean
        posterior_median = stats.beta.ppf(0.5, post_alpha, post_beta)
        posterior_std = np.sqrt((post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)))

        # Credible intervals
        ci_95 = (
            stats.beta.ppf(0.025, post_alpha, post_beta),
            stats.beta.ppf(0.975, post_alpha, post_beta)
        )
        ci_80 = (
            stats.beta.ppf(0.10, post_alpha, post_beta),
            stats.beta.ppf(0.90, post_alpha, post_beta)
        )

        # Effective sample size
        ess = post_alpha + post_beta

        # Prior weight (how much prior influenced vs likelihood)
        total_info = (prior_alpha + prior_beta) + np.log(lr + 1) * 5
        prior_weight = (prior_alpha + prior_beta) / total_info if total_info > 0 else 0.5

        # Signal and reason
        signal, reason = self.get_signal_label(posterior_mean, posterior_std, prior_weight)

        # Probability score
        prob_score = np.clip(posterior_mean, 0.30, 0.75)

        return BayesianResult(
            posterior_mean=posterior_mean,
            posterior_mode=posterior_mode,
            posterior_median=posterior_median,
            credible_interval_95=ci_95,
            credible_interval_80=ci_80,
            posterior_std=posterior_std,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            prior_mean=prior_mean,
            prior_source=prior_source,
            likelihood_ratio=lr,
            signal_contributions=contributions,
            most_influential_signal=most_influential,
            effective_sample_size=ess,
            prior_weight=prior_weight,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _default_result(self, reason: str) -> BayesianResult:
        """Return neutral result when prediction isn't possible."""
        return BayesianResult(
            posterior_mean=0.50,
            posterior_mode=0.50,
            posterior_median=0.50,
            credible_interval_95=(0.35, 0.65),
            credible_interval_80=(0.40, 0.60),
            posterior_std=0.15,
            prior_alpha=52,
            prior_beta=48,
            prior_mean=0.52,
            prior_source='default',
            likelihood_ratio=1.0,
            signal_contributions={},
            most_influential_signal='',
            effective_sample_size=100,
            prior_weight=1.0,
            probability_score=0.50,
            signal='BAYESIAN_UNKNOWN',
            reason=reason
        )
