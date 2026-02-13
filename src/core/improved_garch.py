"""
Improved GARCH Implementation

This module provides a proper GARCH(1,1) implementation using
Maximum Likelihood Estimation instead of moment-based approximation.

Reference:
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"

GARCH(1,1) Model:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t, where z_t ~ N(0,1)
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Constraints:
    - omega > 0
    - alpha >= 0, beta >= 0
    - alpha + beta < 1 (stationarity)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Optional
from loguru import logger


@dataclass
class GARCHResult:
    """Result from GARCH estimation."""
    omega: float           # Long-run variance weight
    alpha: float           # ARCH coefficient (reaction to shocks)
    beta: float            # GARCH coefficient (persistence)
    mu: float              # Mean return
    unconditional_var: float  # omega / (1 - alpha - beta)
    persistence: float     # alpha + beta
    half_life: float       # ln(0.5) / ln(persistence)
    log_likelihood: float  # Log-likelihood of the fit
    aic: float             # Akaike Information Criterion
    bic: float             # Bayesian Information Criterion
    converged: bool        # Whether optimization converged
    conditional_volatility: np.ndarray  # Time series of volatility


class ImprovedGARCH:
    """
    Improved GARCH(1,1) estimator using Maximum Likelihood Estimation.

    This is a significant improvement over moment-based estimation because:
    1. MLE is asymptotically efficient (minimum variance)
    2. Provides proper uncertainty quantification
    3. Allows for hypothesis testing via likelihood ratio tests
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        constrain_stationarity: bool = True
    ):
        """
        Initialize GARCH estimator.

        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            constrain_stationarity: Enforce alpha + beta < 1
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.constrain_stationarity = constrain_stationarity

    def fit(self, returns: np.ndarray) -> GARCHResult:
        """
        Fit GARCH(1,1) model using Maximum Likelihood Estimation.

        Args:
            returns: Array of log returns

        Returns:
            GARCHResult with estimated parameters and diagnostics
        """
        returns = np.asarray(returns).flatten()
        n = len(returns)

        if n < 50:
            logger.warning(f"GARCH: Only {n} observations. Results may be unreliable.")
            if n < 20:
                return self._default_result(returns)

        # Remove mean from returns
        mu = np.mean(returns)
        residuals = returns - mu

        # Initialize parameters using moment-based estimates
        omega_init, alpha_init, beta_init = self._initialize_params(residuals)

        # Parameter bounds
        # omega > 0, alpha in [0, 0.5], beta in [0, 0.99]
        bounds = [
            (1e-10, np.var(residuals) * 2),  # omega
            (1e-10, 0.5),                      # alpha
            (0.5, 0.999)                       # beta
        ]

        # Constraints: alpha + beta < 0.9999 for stationarity
        if self.constrain_stationarity:
            constraints = {
                'type': 'ineq',
                'fun': lambda x: 0.9999 - x[1] - x[2]  # alpha + beta < 0.9999
            }
        else:
            constraints = None

        # Initial parameters
        x0 = np.array([omega_init, alpha_init, beta_init])

        # Optimize using L-BFGS-B with bounds
        try:
            result = minimize(
                fun=lambda x: -self._log_likelihood(x, residuals),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance
                }
            )
            converged = result.success
            omega, alpha, beta = result.x
            log_lik = -result.fun

        except Exception as e:
            logger.warning(f"GARCH optimization failed: {e}. Using moment estimates.")
            omega, alpha, beta = omega_init, alpha_init, beta_init
            log_lik = self._log_likelihood([omega, alpha, beta], residuals)
            converged = False

        # Enforce constraints
        alpha = np.clip(alpha, 0.01, 0.5)
        beta = np.clip(beta, 0.5, 0.98)

        if alpha + beta >= 0.9999:
            # Rescale to ensure stationarity
            total = alpha + beta
            alpha = alpha * 0.98 / total
            beta = beta * 0.98 / total

        # Calculate unconditional variance
        if alpha + beta < 1:
            unconditional_var = omega / (1 - alpha - beta)
        else:
            unconditional_var = np.var(residuals)

        # Calculate persistence and half-life
        persistence = alpha + beta
        if persistence > 0 and persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = float('inf')

        # Calculate conditional volatility series
        conditional_vol = self._calculate_conditional_volatility(
            residuals, omega, alpha, beta
        )

        # Information criteria
        k = 3  # Number of parameters
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        return GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            unconditional_var=unconditional_var,
            persistence=persistence,
            half_life=half_life,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            converged=converged,
            conditional_volatility=conditional_vol
        )

    def forecast(
        self,
        garch_result: GARCHResult,
        last_residual: float,
        last_variance: float,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Forecast future volatility.

        Args:
            garch_result: Fitted GARCH result
            last_residual: Last observed residual (r_t - mu)
            last_variance: Last conditional variance
            horizon: Number of periods to forecast

        Returns:
            Array of forecasted variances
        """
        omega = garch_result.omega
        alpha = garch_result.alpha
        beta = garch_result.beta
        unconditional = garch_result.unconditional_var

        forecasts = np.zeros(horizon)

        # First step forecast
        forecasts[0] = omega + alpha * last_residual**2 + beta * last_variance

        # Multi-step forecasts (converge to unconditional variance)
        persistence = alpha + beta
        for h in range(1, horizon):
            # E[sigma_{t+h}^2 | F_t] = unconditional + persistence^h * (sigma_{t+1}^2 - unconditional)
            forecasts[h] = unconditional + (persistence ** h) * (forecasts[0] - unconditional)

        return forecasts

    def _initialize_params(self, residuals: np.ndarray) -> Tuple[float, float, float]:
        """
        Initialize GARCH parameters using method of moments.

        This provides a good starting point for MLE optimization.
        """
        var = np.var(residuals)
        squared_resid = residuals ** 2

        # Estimate autocorrelation of squared residuals
        if len(squared_resid) > 1:
            autocorr = np.corrcoef(squared_resid[:-1], squared_resid[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.85
        else:
            autocorr = 0.85

        # Initialize beta from autocorrelation (persistence proxy)
        beta = np.clip(autocorr * 0.9, 0.6, 0.95)

        # Initialize alpha
        alpha = np.clip(0.08, 0.02, 0.2)

        # Initialize omega from unconditional variance
        omega = var * (1 - alpha - beta)
        omega = max(omega, 1e-8)

        return omega, alpha, beta

    def _log_likelihood(
        self,
        params: np.ndarray,
        residuals: np.ndarray
    ) -> float:
        """
        Calculate log-likelihood for GARCH(1,1).

        L = -0.5 * sum(log(sigma_t^2) + epsilon_t^2 / sigma_t^2)

        Args:
            params: [omega, alpha, beta]
            residuals: Demeaned returns

        Returns:
            Log-likelihood value
        """
        omega, alpha, beta = params
        n = len(residuals)

        # Enforce parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0:
            return -1e10

        if alpha + beta >= 1:
            return -1e10

        # Calculate conditional variances
        sigma2 = np.zeros(n)

        # Initialize with unconditional variance
        sigma2[0] = np.var(residuals)

        # Recursion
        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]

            # Prevent numerical issues
            if sigma2[t] < 1e-10:
                sigma2[t] = 1e-10

        # Log-likelihood (Gaussian)
        log_lik = -0.5 * np.sum(np.log(sigma2) + residuals**2 / sigma2)

        # Penalty for extreme values
        if np.isnan(log_lik) or np.isinf(log_lik):
            return -1e10

        return log_lik

    def _calculate_conditional_volatility(
        self,
        residuals: np.ndarray,
        omega: float,
        alpha: float,
        beta: float
    ) -> np.ndarray:
        """
        Calculate conditional volatility time series.

        Args:
            residuals: Demeaned returns
            omega, alpha, beta: GARCH parameters

        Returns:
            Array of conditional volatilities (sigma, not sigma^2)
        """
        n = len(residuals)
        sigma2 = np.zeros(n)

        # Initialize
        sigma2[0] = np.var(residuals)

        # Recursion
        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-10)

        return np.sqrt(sigma2)

    def _default_result(self, returns: np.ndarray) -> GARCHResult:
        """Return default result when fitting fails or insufficient data."""
        var = np.var(returns) if len(returns) > 0 else 0.0004
        vol = np.sqrt(var)

        return GARCHResult(
            omega=var * 0.05,
            alpha=0.10,
            beta=0.85,
            mu=np.mean(returns) if len(returns) > 0 else 0,
            unconditional_var=var,
            persistence=0.95,
            half_life=13.5,  # ln(0.5) / ln(0.95) ≈ 13.5 days
            log_likelihood=0,
            aic=0,
            bic=0,
            converged=False,
            conditional_volatility=np.full(len(returns), vol) if len(returns) > 0 else np.array([vol])
        )


# Convenience function
def fit_garch(returns: np.ndarray) -> GARCHResult:
    """
    Fit GARCH(1,1) model to returns.

    Args:
        returns: Array of log returns

    Returns:
        GARCHResult with estimated parameters
    """
    garch = ImprovedGARCH()
    return garch.fit(returns)


def forecast_volatility(
    returns: np.ndarray,
    horizon: int = 1
) -> np.ndarray:
    """
    Forecast volatility using GARCH(1,1).

    Args:
        returns: Historical returns
        horizon: Forecast horizon

    Returns:
        Array of forecasted volatilities (sigma)
    """
    garch = ImprovedGARCH()
    result = garch.fit(returns)

    # Get last values
    mu = result.mu
    last_residual = returns[-1] - mu
    last_variance = result.conditional_volatility[-1] ** 2

    # Forecast
    forecast_var = garch.forecast(result, last_residual, last_variance, horizon)

    return np.sqrt(forecast_var)


# Export
__all__ = ['ImprovedGARCH', 'GARCHResult', 'fit_garch', 'forecast_volatility']
