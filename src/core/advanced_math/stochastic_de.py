"""
Stochastic Differential Equations Engine for Stock Prediction

Implements advanced SDE models for price dynamics:
- Geometric Brownian Motion (GBM)
- Ornstein-Uhlenbeck (Mean Reversion)
- Jump-Diffusion (Merton Model)
- Heston Stochastic Volatility
- SABR Model
- Cox-Ingersoll-Ross (CIR) for volatility

Mathematical Foundation:
- Itô's Lemma: d(f(X)) = f'(X)dX + ½f''(X)(dX)²
- GBM: dS = μS dt + σS dW
- OU: dX = θ(μ - X)dt + σ dW
- Jump-Diffusion: dS = μS dt + σS dW + (J-1)S dN
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class SDEModel(Enum):
    """Available SDE models"""
    GBM = "geometric_brownian_motion"
    OU = "ornstein_uhlenbeck"
    JUMP_DIFFUSION = "merton_jump_diffusion"
    HESTON = "heston_stochastic_vol"
    SABR = "sabr"
    CIR = "cox_ingersoll_ross"


@dataclass
class SDEParameters:
    """Estimated SDE parameters"""
    model: SDEModel
    drift: float  # μ
    volatility: float  # σ
    mean_reversion_rate: float  # θ (for OU/CIR)
    long_term_mean: float  # μ̄ (for OU/CIR)
    jump_intensity: float  # λ (for jump-diffusion)
    jump_mean: float  # μ_J (for jump-diffusion)
    jump_std: float  # σ_J (for jump-diffusion)
    vol_of_vol: float  # ξ (for Heston/SABR)
    correlation: float  # ρ (for Heston)
    log_likelihood: float
    aic: float
    bic: float


@dataclass
class SDEPrediction:
    """Prediction from SDE model"""
    expected_price: float
    price_std: float
    confidence_interval_95: Tuple[float, float]
    probability_up: float
    expected_return: float
    expected_volatility: float
    jump_probability: float  # Probability of jump in horizon
    regime_indicator: str
    monte_carlo_paths: Optional[np.ndarray] = None
    prediction_horizon: int = 1


class StochasticDEEngine:
    """
    Stochastic Differential Equations Engine.

    Provides:
    1. Parameter estimation using MLE
    2. Monte Carlo simulation
    3. Probability distribution forecasting
    4. Model selection (AIC/BIC)
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)

    def estimate_all_models(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, SDEParameters]:
        """
        Estimate parameters for all SDE models and return best fit.
        """
        if returns is None:
            returns = np.diff(np.log(prices))

        results = {}

        # Geometric Brownian Motion
        results['gbm'] = self._estimate_gbm(returns)

        # Ornstein-Uhlenbeck
        results['ou'] = self._estimate_ou(prices)

        # Jump-Diffusion
        results['jump_diffusion'] = self._estimate_jump_diffusion(returns)

        # Heston
        results['heston'] = self._estimate_heston(returns)

        # CIR for volatility
        vol_series = self._compute_rolling_volatility(returns)
        if len(vol_series) > 20:
            results['cir'] = self._estimate_cir(vol_series)

        return results

    def select_best_model(
        self,
        model_params: Dict[str, SDEParameters]
    ) -> Tuple[str, SDEParameters]:
        """Select best model using BIC criterion."""
        best_model = None
        best_bic = float('inf')

        for name, params in model_params.items():
            if params.bic < best_bic:
                best_bic = params.bic
                best_model = name

        return best_model, model_params[best_model]

    def predict(
        self,
        prices: np.ndarray,
        model_params: SDEParameters,
        horizon: int = 1,
        return_paths: bool = False
    ) -> SDEPrediction:
        """
        Generate prediction using estimated SDE model.
        """
        current_price = prices[-1]
        returns = np.diff(np.log(prices))

        if model_params.model == SDEModel.GBM:
            paths = self._simulate_gbm(
                current_price, model_params, horizon
            )
        elif model_params.model == SDEModel.OU:
            # For OU on log-prices
            log_price = np.log(current_price)
            log_paths = self._simulate_ou(log_price, model_params, horizon)
            paths = np.exp(log_paths)
        elif model_params.model == SDEModel.JUMP_DIFFUSION:
            paths = self._simulate_jump_diffusion(
                current_price, model_params, horizon
            )
        elif model_params.model == SDEModel.HESTON:
            paths, _ = self._simulate_heston(
                current_price, model_params, horizon
            )
        else:
            # Default to GBM
            paths = self._simulate_gbm(
                current_price, model_params, horizon
            )

        # Terminal prices
        terminal_prices = paths[:, -1]

        # Statistics
        expected_price = float(np.mean(terminal_prices))
        price_std = float(np.std(terminal_prices))
        ci_lower = float(np.percentile(terminal_prices, 2.5))
        ci_upper = float(np.percentile(terminal_prices, 97.5))

        # Probability of up move
        prob_up = float(np.mean(terminal_prices > current_price))

        # Expected return and volatility
        simulated_returns = np.log(terminal_prices / current_price)
        expected_return = float(np.mean(simulated_returns))
        expected_vol = float(np.std(simulated_returns))

        # Jump probability (for jump-diffusion)
        if model_params.model == SDEModel.JUMP_DIFFUSION:
            jump_prob = 1 - np.exp(-model_params.jump_intensity * horizon)
        else:
            jump_prob = 0.0

        # Regime indicator
        regime = self._classify_regime(model_params, returns)

        return SDEPrediction(
            expected_price=expected_price,
            price_std=price_std,
            confidence_interval_95=(ci_lower, ci_upper),
            probability_up=prob_up,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            jump_probability=float(jump_prob),
            regime_indicator=regime,
            monte_carlo_paths=paths if return_paths else None,
            prediction_horizon=horizon
        )

    def _estimate_gbm(self, returns: np.ndarray) -> SDEParameters:
        """
        Estimate GBM parameters using MLE.

        dS = μS dt + σS dW
        log-returns are normal: r ~ N(μ - σ²/2, σ²)
        """
        n = len(returns)

        # MLE estimates
        sigma = np.std(returns, ddof=1)
        mu = np.mean(returns) + 0.5 * sigma ** 2

        # Log-likelihood
        ll = -n/2 * np.log(2 * np.pi * sigma**2) - \
             np.sum((returns - mu + 0.5*sigma**2)**2) / (2 * sigma**2)

        # AIC and BIC (2 parameters: μ, σ)
        k = 2
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return SDEParameters(
            model=SDEModel.GBM,
            drift=float(mu),
            volatility=float(sigma),
            mean_reversion_rate=0.0,
            long_term_mean=0.0,
            jump_intensity=0.0,
            jump_mean=0.0,
            jump_std=0.0,
            vol_of_vol=0.0,
            correlation=0.0,
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic)
        )

    def _estimate_ou(self, prices: np.ndarray) -> SDEParameters:
        """
        Estimate Ornstein-Uhlenbeck parameters.

        dX = θ(μ - X)dt + σ dW

        Using discrete approximation:
        X_{t+1} - X_t = θ(μ - X_t)Δt + σ√Δt ε
        """
        log_prices = np.log(prices)
        n = len(log_prices) - 1

        # OLS regression: X_{t+1} = a + b*X_t + ε
        X = log_prices[:-1]
        Y = log_prices[1:]

        # Normal equations
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        cov_XY = np.mean((X - X_mean) * (Y - Y_mean))
        var_X = np.var(X)

        if var_X > 0:
            b = cov_XY / var_X
            a = Y_mean - b * X_mean
        else:
            b = 0.99
            a = 0.01 * X_mean

        # Extract OU parameters
        # b = exp(-θΔt) ≈ 1 - θΔt for small θ
        # a = μ(1 - exp(-θΔt)) ≈ μθΔt
        dt = 1.0  # daily

        if b < 1 and b > 0:
            theta = -np.log(b) / dt
        else:
            theta = 0.1

        if theta > 0:
            mu = a / (1 - np.exp(-theta * dt))
        else:
            mu = np.mean(log_prices)

        # Residual volatility
        residuals = Y - (a + b * X)
        sigma = np.std(residuals) / np.sqrt(dt)

        # Log-likelihood
        ll = -n/2 * np.log(2 * np.pi * sigma**2 * dt) - \
             np.sum(residuals**2) / (2 * sigma**2 * dt)

        # AIC and BIC (3 parameters: θ, μ, σ)
        k = 3
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return SDEParameters(
            model=SDEModel.OU,
            drift=float(mu),  # Long-term mean
            volatility=float(sigma),
            mean_reversion_rate=float(theta),
            long_term_mean=float(mu),
            jump_intensity=0.0,
            jump_mean=0.0,
            jump_std=0.0,
            vol_of_vol=0.0,
            correlation=0.0,
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic)
        )

    def _estimate_jump_diffusion(self, returns: np.ndarray) -> SDEParameters:
        """
        Estimate Merton jump-diffusion parameters.

        dS/S = μ dt + σ dW + (J-1) dN

        Where:
        - N is Poisson process with intensity λ
        - J is jump size: log(J) ~ N(μ_J, σ_J²)
        """
        n = len(returns)

        # Identify jumps using threshold
        vol = np.std(returns)
        threshold = 3 * vol  # >3σ = jump

        jump_mask = np.abs(returns) > threshold
        n_jumps = np.sum(jump_mask)

        # Separate diffusion and jump components
        diffusion_returns = returns[~jump_mask]
        jump_returns = returns[jump_mask]

        # Estimate diffusion parameters
        if len(diffusion_returns) > 5:
            sigma_d = np.std(diffusion_returns)
            mu_d = np.mean(diffusion_returns) + 0.5 * sigma_d ** 2
        else:
            sigma_d = vol
            mu_d = np.mean(returns)

        # Estimate jump parameters
        lambda_jump = n_jumps / n  # Jump intensity (daily)

        if n_jumps > 0:
            mu_j = np.mean(jump_returns)
            sigma_j = np.std(jump_returns) if n_jumps > 1 else vol
        else:
            mu_j = 0.0
            sigma_j = 2 * vol

        # Log-likelihood (mixture of diffusion and jump components)
        # Simplified: sum of diffusion and jump log-likelihoods
        ll_diff = -len(diffusion_returns)/2 * np.log(2 * np.pi * sigma_d**2) - \
                  np.sum((diffusion_returns - mu_d + 0.5*sigma_d**2)**2) / (2 * sigma_d**2)

        if n_jumps > 0 and sigma_j > 0:
            ll_jump = -n_jumps/2 * np.log(2 * np.pi * sigma_j**2) - \
                      np.sum((jump_returns - mu_j)**2) / (2 * sigma_j**2)
        else:
            ll_jump = 0

        ll = ll_diff + ll_jump

        # AIC and BIC (5 parameters: μ, σ, λ, μ_J, σ_J)
        k = 5
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return SDEParameters(
            model=SDEModel.JUMP_DIFFUSION,
            drift=float(mu_d),
            volatility=float(sigma_d),
            mean_reversion_rate=0.0,
            long_term_mean=0.0,
            jump_intensity=float(lambda_jump),
            jump_mean=float(mu_j),
            jump_std=float(sigma_j),
            vol_of_vol=0.0,
            correlation=0.0,
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic)
        )

    def _estimate_heston(self, returns: np.ndarray) -> SDEParameters:
        """
        Estimate Heston stochastic volatility model parameters.

        dS = μS dt + √V S dW_1
        dV = κ(θ - V)dt + ξ√V dW_2
        corr(dW_1, dW_2) = ρ
        """
        n = len(returns)

        # Estimate base drift and volatility
        sigma = np.std(returns)
        mu = np.mean(returns) + 0.5 * sigma ** 2

        # Estimate vol-of-vol from realized volatility series
        rolling_vol = self._compute_rolling_volatility(returns, window=10)

        if len(rolling_vol) > 20:
            vol_returns = np.diff(np.log(rolling_vol + 1e-10))
            xi = np.std(vol_returns) if len(vol_returns) > 1 else 0.5

            # Mean reversion of volatility
            vol_mean = np.mean(rolling_vol)
            vol_auto = np.corrcoef(rolling_vol[:-1], rolling_vol[1:])[0, 1] \
                if len(rolling_vol) > 2 else 0.9
            kappa = -np.log(max(vol_auto, 0.01))

            # Correlation between returns and volatility changes
            if len(vol_returns) >= len(returns) - 11:
                min_len = min(len(returns[10:]), len(vol_returns))
                rho = np.corrcoef(returns[10:10+min_len], vol_returns[:min_len])[0, 1]
            else:
                rho = -0.5  # Typical negative leverage effect
        else:
            xi = 0.5
            kappa = 1.0
            vol_mean = sigma
            rho = -0.5

        # Simplified log-likelihood (not exact Heston MLE)
        ll = -n/2 * np.log(2 * np.pi * sigma**2) - \
             np.sum((returns - mu + 0.5*sigma**2)**2) / (2 * sigma**2)

        # Penalty for stochastic vol (rough approximation)
        ll -= 0.5 * len(rolling_vol) * np.log(xi**2 + 1)

        # AIC and BIC (5 parameters: μ, θ, κ, ξ, ρ)
        k = 5
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return SDEParameters(
            model=SDEModel.HESTON,
            drift=float(mu),
            volatility=float(sigma),  # Long-term vol
            mean_reversion_rate=float(kappa),
            long_term_mean=float(vol_mean ** 2),  # θ in variance space
            jump_intensity=0.0,
            jump_mean=0.0,
            jump_std=0.0,
            vol_of_vol=float(xi),
            correlation=float(np.clip(rho, -0.99, 0.99)) if not np.isnan(rho) else -0.5,
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic)
        )

    def _estimate_cir(self, volatility_series: np.ndarray) -> SDEParameters:
        """
        Estimate Cox-Ingersoll-Ross for volatility.

        dV = κ(θ - V)dt + ξ√V dW
        """
        n = len(volatility_series) - 1

        V = volatility_series[:-1]
        dV = np.diff(volatility_series)

        # OLS: dV = κ(θ - V)dt + ε
        # Rewrite: dV = κθ - κV + ε
        X = np.column_stack([np.ones(n), -V])
        y = dV

        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            kappa_theta = coeffs[0]
            kappa = coeffs[1]

            if kappa > 0:
                theta = kappa_theta / kappa
            else:
                kappa = 0.1
                theta = np.mean(volatility_series)

            residuals = y - X @ coeffs
            xi = np.std(residuals / (np.sqrt(V) + 1e-10))

        except Exception:
            kappa = 0.1
            theta = np.mean(volatility_series)
            xi = 0.5
            residuals = dV

        # Log-likelihood
        sigma_resid = np.std(residuals)
        ll = -n/2 * np.log(2 * np.pi * sigma_resid**2) - \
             np.sum(residuals**2) / (2 * sigma_resid**2)

        k = 3
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return SDEParameters(
            model=SDEModel.CIR,
            drift=0.0,
            volatility=float(theta),  # Long-term vol
            mean_reversion_rate=float(abs(kappa)),
            long_term_mean=float(theta),
            jump_intensity=0.0,
            jump_mean=0.0,
            jump_std=0.0,
            vol_of_vol=float(abs(xi)),
            correlation=0.0,
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic)
        )

    def _compute_rolling_volatility(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Compute rolling volatility."""
        n = len(returns)
        if n < window:
            return np.array([np.std(returns)])

        rolling_vol = np.array([
            np.std(returns[max(0, i-window+1):i+1])
            for i in range(window-1, n)
        ])
        return rolling_vol

    def _simulate_gbm(
        self,
        S0: float,
        params: SDEParameters,
        horizon: int
    ) -> np.ndarray:
        """
        Simulate GBM paths.

        S_t = S_0 exp((μ - σ²/2)t + σW_t)
        """
        dt = 1.0 / 252  # Daily
        n_steps = horizon

        # Standard normal random numbers
        Z = self.rng.standard_normal((self.n_simulations, n_steps))

        # Cumulative sum for Brownian motion
        W = np.cumsum(Z, axis=1) * np.sqrt(dt)

        # Time vector
        t = np.linspace(dt, horizon * dt, n_steps)

        # GBM formula
        drift_term = (params.drift - 0.5 * params.volatility ** 2) * t
        diffusion_term = params.volatility * W

        paths = S0 * np.exp(drift_term + diffusion_term)

        # Prepend initial price
        paths = np.column_stack([np.full(self.n_simulations, S0), paths])

        return paths

    def _simulate_ou(
        self,
        X0: float,
        params: SDEParameters,
        horizon: int
    ) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck paths.

        Exact solution:
        X_t = μ + (X_0 - μ)e^{-θt} + σ∫₀ᵗe^{-θ(t-s)}dW_s
        """
        dt = 1.0 / 252
        n_steps = horizon

        theta = params.mean_reversion_rate
        mu = params.long_term_mean
        sigma = params.volatility

        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = X0

        for t in range(n_steps):
            Z = self.rng.standard_normal(self.n_simulations)

            # Exact discretization
            mean = mu + (paths[:, t] - mu) * np.exp(-theta * dt)
            std = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta + 1e-10))

            paths[:, t + 1] = mean + std * Z

        return paths

    def _simulate_jump_diffusion(
        self,
        S0: float,
        params: SDEParameters,
        horizon: int
    ) -> np.ndarray:
        """
        Simulate Merton jump-diffusion paths.

        dS/S = μ dt + σ dW + (J-1) dN
        """
        dt = 1.0 / 252
        n_steps = horizon

        mu = params.drift
        sigma = params.volatility
        lambda_j = params.jump_intensity
        mu_j = params.jump_mean
        sigma_j = params.jump_std

        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = S0

        for t in range(n_steps):
            # Diffusion component
            Z_diff = self.rng.standard_normal(self.n_simulations)
            diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_diff

            # Jump component
            n_jumps = self.rng.poisson(lambda_j * dt, self.n_simulations)
            jump_sizes = np.zeros(self.n_simulations)

            for i in range(self.n_simulations):
                if n_jumps[i] > 0:
                    jumps = self.rng.normal(mu_j, sigma_j, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps)

            # Combined log return
            log_return = diffusion + jump_sizes

            paths[:, t + 1] = paths[:, t] * np.exp(log_return)

        return paths

    def _simulate_heston(
        self,
        S0: float,
        params: SDEParameters,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston stochastic volatility paths.

        dS = μS dt + √V S dW_1
        dV = κ(θ - V)dt + ξ√V dW_2
        """
        dt = 1.0 / 252
        n_steps = horizon

        mu = params.drift
        kappa = params.mean_reversion_rate
        theta = params.long_term_mean
        xi = params.vol_of_vol
        rho = params.correlation
        V0 = params.volatility ** 2

        price_paths = np.zeros((self.n_simulations, n_steps + 1))
        vol_paths = np.zeros((self.n_simulations, n_steps + 1))

        price_paths[:, 0] = S0
        vol_paths[:, 0] = V0

        for t in range(n_steps):
            # Correlated Brownian motions
            Z1 = self.rng.standard_normal(self.n_simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * self.rng.standard_normal(self.n_simulations)

            V = np.maximum(vol_paths[:, t], 1e-10)  # Ensure positive

            # Volatility update (full truncation scheme)
            dV = kappa * (theta - V) * dt + xi * np.sqrt(V * dt) * Z2
            vol_paths[:, t + 1] = np.maximum(V + dV, 1e-10)

            # Price update
            dS = mu * dt + np.sqrt(V * dt) * Z1
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(dS - 0.5 * V * dt)

        return price_paths, vol_paths

    def _classify_regime(
        self,
        params: SDEParameters,
        returns: np.ndarray
    ) -> str:
        """Classify current regime based on model parameters."""
        if params.model == SDEModel.JUMP_DIFFUSION:
            if params.jump_intensity > 0.05:  # >5% daily jump probability
                return "high_jump_risk"

        if params.mean_reversion_rate > 0.5:
            return "mean_reverting"

        sharpe = params.drift / (params.volatility + 1e-10) * np.sqrt(252)

        if sharpe > 1.0:
            return "strong_trend"
        elif sharpe > 0.3:
            return "mild_trend"
        elif sharpe > -0.3:
            return "sideways"
        elif sharpe > -1.0:
            return "mild_downtrend"
        else:
            return "strong_downtrend"


class JumpDetector:
    """
    Detect and analyze jumps in price series.

    Uses statistical tests to identify:
    - Jump occurrence
    - Jump size distribution
    - Jump clustering
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.0,
        min_jump_size: float = 0.02  # 2% minimum
    ):
        self.threshold_multiplier = threshold_multiplier
        self.min_jump_size = min_jump_size

    def detect_jumps(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Detect jumps using bipower variation test.
        """
        n = len(returns)
        if n < 20:
            return self._empty_result()

        # Realized variance
        rv = np.sum(returns ** 2)

        # Bipower variation (robust to jumps)
        bv = (np.pi / 2) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))

        # Jump test statistic
        quad_var = (np.pi**2 / 4 + np.pi - 5) * \
                   np.sum(np.abs(returns[:-2]) * np.abs(returns[1:-1]) * np.abs(returns[2:]) ** 2)

        if quad_var > 0 and bv > 0:
            z_stat = (rv - bv) / np.sqrt(quad_var / bv ** 2)
        else:
            z_stat = 0

        # p-value (one-sided test for positive jumps)
        p_value = 1 - stats.norm.cdf(z_stat)

        # Identify individual jumps
        vol = np.std(returns)
        threshold = self.threshold_multiplier * vol

        jump_mask = (np.abs(returns) > threshold) & (np.abs(returns) > self.min_jump_size)
        jump_indices = np.where(jump_mask)[0]
        jump_returns = returns[jump_mask]

        # Jump characteristics
        n_jumps = len(jump_indices)
        if n_jumps > 0:
            avg_jump_size = float(np.mean(np.abs(jump_returns)))
            jump_skewness = float(np.mean(jump_returns) / (np.std(jump_returns) + 1e-10)) if n_jumps > 1 else 0
            jump_intensity = n_jumps / n
        else:
            avg_jump_size = 0
            jump_skewness = 0
            jump_intensity = 0

        # Jump clustering (Hawkes process indication)
        if n_jumps > 1:
            inter_jump_times = np.diff(jump_indices)
            clustering_ratio = np.std(inter_jump_times) / (np.mean(inter_jump_times) + 1e-10)
            is_clustered = clustering_ratio > 1.5
        else:
            clustering_ratio = 0
            is_clustered = False

        return {
            "jump_detected": p_value < 0.05,
            "jump_test_statistic": float(z_stat),
            "p_value": float(p_value),
            "n_jumps": n_jumps,
            "jump_indices": jump_indices.tolist(),
            "jump_returns": jump_returns.tolist(),
            "avg_jump_size": avg_jump_size,
            "jump_skewness": jump_skewness,
            "jump_intensity": jump_intensity,
            "is_clustered": is_clustered,
            "clustering_ratio": float(clustering_ratio),
            "realized_variance": float(rv),
            "bipower_variation": float(bv),
            "jump_variation": float(max(rv - bv, 0))
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "jump_detected": False,
            "jump_test_statistic": 0,
            "p_value": 1.0,
            "n_jumps": 0,
            "jump_indices": [],
            "jump_returns": [],
            "avg_jump_size": 0,
            "jump_skewness": 0,
            "jump_intensity": 0,
            "is_clustered": False,
            "clustering_ratio": 0,
            "realized_variance": 0,
            "bipower_variation": 0,
            "jump_variation": 0
        }


class MeanReversionAnalyzer:
    """
    Analyze mean reversion properties using Ornstein-Uhlenbeck model.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze mean reversion properties.

        Returns:
        - Half-life of mean reversion
        - Equilibrium level
        - Current deviation
        - Trading signals
        """
        log_prices = np.log(prices)
        n = len(log_prices)

        if n < 20:
            return self._default_result()

        # Fit AR(1): X_{t+1} = a + b*X_t + ε
        X = log_prices[:-1]
        Y = log_prices[1:]

        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        cov_XY = np.mean((X - X_mean) * (Y - Y_mean))
        var_X = np.var(X)

        if var_X > 0:
            b = cov_XY / var_X
            a = Y_mean - b * X_mean
        else:
            return self._default_result()

        # Mean reversion parameters
        if b < 1 and b > 0:
            theta = -np.log(b)  # Mean reversion speed
            half_life = np.log(2) / theta  # Days to mean-revert halfway
            mu = a / (1 - b)  # Equilibrium level
        elif b >= 1:
            # Unit root or explosive - no mean reversion
            theta = 0
            half_life = float('inf')
            mu = np.mean(log_prices)
        else:
            # Negative autocorrelation
            theta = -np.log(abs(b))
            half_life = np.log(2) / theta
            mu = np.mean(log_prices)

        # Current deviation from equilibrium
        current_log_price = log_prices[-1]
        deviation = current_log_price - mu
        deviation_pct = (np.exp(deviation) - 1) * 100

        # Residual volatility
        residuals = Y - (a + b * X)
        sigma = np.std(residuals)

        # Z-score for current deviation
        if sigma > 0 and theta > 0:
            stationary_std = sigma / np.sqrt(2 * theta)
            z_score = deviation / stationary_std
        else:
            z_score = 0

        # ADF test for stationarity
        adf_stat, adf_pvalue = self._adf_test(log_prices)

        # Trading signal
        if z_score > 2:
            signal = "sell"
            signal_strength = min(1.0, (z_score - 2) / 2)
        elif z_score < -2:
            signal = "buy"
            signal_strength = min(1.0, (-z_score - 2) / 2)
        else:
            signal = "neutral"
            signal_strength = 0

        # Expected price movement (if mean reverting)
        if theta > 0 and half_life < 100:
            expected_move = -deviation * (1 - np.exp(-theta))
            expected_return = (np.exp(expected_move) - 1) * 100
        else:
            expected_return = 0

        return {
            "is_mean_reverting": half_life < 60 and adf_pvalue < 0.1,
            "half_life_days": float(half_life) if half_life < 1000 else float('inf'),
            "mean_reversion_speed": float(theta),
            "equilibrium_level": float(np.exp(mu)),
            "current_deviation_pct": float(deviation_pct),
            "z_score": float(z_score),
            "residual_volatility": float(sigma),
            "adf_statistic": float(adf_stat),
            "adf_pvalue": float(adf_pvalue),
            "trading_signal": signal,
            "signal_strength": float(signal_strength),
            "expected_return_1d": float(expected_return)
        }

    def _adf_test(self, series: np.ndarray) -> Tuple[float, float]:
        """
        Simplified ADF test for stationarity.
        Tests H0: series has unit root (non-stationary)
        """
        n = len(series)
        if n < 20:
            return 0, 1.0

        # First difference
        diff = np.diff(series)

        # Lagged level
        lag = series[:-1]

        # OLS: Δy_t = α + γy_{t-1} + ε
        X = np.column_stack([np.ones(n-1), lag])
        y = diff

        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            gamma = coeffs[1]

            # Standard error
            residuals = y - X @ coeffs
            s2 = np.sum(residuals**2) / (n - 3)
            se_gamma = np.sqrt(s2 / np.sum((lag - np.mean(lag))**2))

            # t-statistic
            t_stat = gamma / se_gamma

            # Critical values (approximate)
            # 1%: -3.43, 5%: -2.86, 10%: -2.57
            if t_stat < -3.43:
                p_value = 0.01
            elif t_stat < -2.86:
                p_value = 0.05
            elif t_stat < -2.57:
                p_value = 0.10
            else:
                p_value = 0.5 * (1 + np.tanh((t_stat + 2) / 0.5))

            return float(t_stat), float(p_value)

        except Exception:
            return 0, 1.0

    def _default_result(self) -> Dict[str, Any]:
        return {
            "is_mean_reverting": False,
            "half_life_days": float('inf'),
            "mean_reversion_speed": 0,
            "equilibrium_level": 0,
            "current_deviation_pct": 0,
            "z_score": 0,
            "residual_volatility": 0,
            "adf_statistic": 0,
            "adf_pvalue": 1.0,
            "trading_signal": "neutral",
            "signal_strength": 0,
            "expected_return_1d": 0
        }
