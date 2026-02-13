"""
Taylor Series and Asymptotic Expansion Module

Implements advanced approximation methods:
- Taylor Series Expansion for price dynamics
- Edgeworth Expansion for distribution approximation
- Cornish-Fisher Expansion for VaR
- Gram-Charlier Expansion
- Moment Generating Functions

Applications:
- Non-linear price dynamics approximation
- Higher-moment adjusted risk metrics
- Option pricing approximations
- Volatility surface modeling

Mathematical Foundation:
- Taylor: f(x) = Σₙ f⁽ⁿ⁾(a)/n! (x-a)ⁿ
- Edgeworth: f(x) ≈ φ(x)[1 + κ₃/6 He₃(x) + κ₄/24 He₄(x) + ...]
- Cornish-Fisher: z_α ≈ z + (z²-1)κ₃/6 + (z³-3z)κ₄/24 - (2z³-5z)κ₃²/36
"""

import numpy as np
from math import factorial
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result of series expansion analysis"""
    coefficients: np.ndarray  # Expansion coefficients
    approximation: np.ndarray  # Approximated values
    residual_error: float  # Approximation error
    convergence_radius: float  # Estimated convergence radius
    moments: Dict[str, float]  # Distribution moments
    cumulants: Dict[str, float]  # Distribution cumulants
    risk_metrics: Dict[str, float]  # Corrected risk metrics


class TaylorSeriesAnalyzer:
    """
    Taylor Series expansion for price dynamics modeling.

    Expands price function around a point:
    P(t+Δt) ≈ P(t) + P'(t)Δt + P''(t)Δt²/2 + ...
    """

    def __init__(
        self,
        order: int = 4
    ):
        self.order = order

    def expand_price_dynamics(
        self,
        prices: np.ndarray,
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Expand price dynamics using Taylor series.
        """
        if len(prices) < self.order + 5:
            return self._default_result()

        # Compute numerical derivatives
        derivatives = self._compute_derivatives(prices)

        # Current values
        current_price = prices[-1]
        dt = 1.0  # 1 day

        # Taylor approximation
        approximation = current_price
        coefficients = [current_price]

        for n in range(1, min(self.order + 1, len(derivatives))):
            term = derivatives[n-1][-1] * (dt ** n) / factorial(n)
            approximation += term
            coefficients.append(term)

        # Multi-step forecast
        forecasts = []
        for h in range(1, horizon + 1):
            forecast = current_price
            for n in range(1, min(self.order + 1, len(derivatives))):
                forecast += derivatives[n-1][-1] * ((h * dt) ** n) / factorial(n)
            forecasts.append(forecast)

        # Estimate convergence radius
        # Using ratio test on derivatives
        if len(derivatives) >= 2:
            ratios = []
            for i in range(len(derivatives) - 1):
                d1, d2 = derivatives[i][-1], derivatives[i+1][-1]
                if abs(d2) > 1e-10:
                    ratios.append(abs(d1 / d2))
            convergence_radius = np.mean(ratios) if ratios else float('inf')
        else:
            convergence_radius = float('inf')

        # Residual error estimation
        if len(prices) > 10 and derivatives:
            # Backtest on recent data
            errors = []
            # Get the minimum length across derivatives
            min_deriv_len = min(len(d) for d in derivatives) if derivatives else 0

            for t in range(10, min(len(prices), min_deriv_len + 1)):
                predicted = prices[t-1]
                for n in range(1, min(self.order + 1, len(derivatives))):
                    deriv_idx = t - 1 - n  # Account for derivative shortening
                    if 0 <= deriv_idx < len(derivatives[n-1]):
                        predicted += derivatives[n-1][deriv_idx] * dt ** n / factorial(n)
                errors.append(abs(prices[t] - predicted))
            residual_error = np.mean(errors) if errors else np.std(prices)
        else:
            residual_error = np.std(prices)

        return {
            "coefficients": np.array(coefficients),
            "forecasts": np.array(forecasts),
            "derivatives": [d[-1] for d in derivatives],
            "convergence_radius": float(convergence_radius),
            "residual_error": float(residual_error),
            "current_velocity": float(derivatives[0][-1]) if derivatives else 0,
            "current_acceleration": float(derivatives[1][-1]) if len(derivatives) > 1 else 0
        }

    def _compute_derivatives(
        self,
        prices: np.ndarray
    ) -> List[np.ndarray]:
        """Compute numerical derivatives up to order."""
        derivatives = []
        current = prices.copy()

        for _ in range(self.order):
            # Forward difference
            derivative = np.diff(current)
            derivatives.append(derivative)
            current = derivative

            if len(derivative) < 2:
                break

        return derivatives

    def _default_result(self) -> Dict[str, Any]:
        return {
            "coefficients": np.array([]),
            "forecasts": np.array([]),
            "convergence_radius": 0.0,
            "residual_error": float('inf')
        }


class EdgeworthExpansion:
    """
    Edgeworth Expansion for distribution approximation.

    Corrects Gaussian approximation using higher cumulants:
    f(x) ≈ φ(x)[1 + κ₃/6 He₃(x) + κ₄/24 He₄(x) + κ₃²/72 He₆(x) + ...]

    Where He_n are probabilists' Hermite polynomials.
    """

    def __init__(
        self,
        order: int = 4
    ):
        self.order = order

    def fit(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fit Edgeworth expansion to return distribution.
        """
        if len(returns) < 30:
            return self._default_result()

        # Standardize
        mu = np.mean(returns)
        sigma = np.std(returns)
        z = (returns - mu) / (sigma + 1e-10)

        # Compute cumulants
        cumulants = self._compute_cumulants(z)

        # Expansion coefficients
        k3 = cumulants.get("kappa_3", 0)  # Skewness
        k4 = cumulants.get("kappa_4", 0)  # Excess kurtosis

        # Edgeworth correction at each point
        correction = (
            1 +
            k3 / 6 * self._hermite(z, 3) +
            k4 / 24 * self._hermite(z, 4) +
            k3 ** 2 / 72 * self._hermite(z, 6)
        )

        # PDF approximation
        phi_z = stats.norm.pdf(z)
        pdf_edgeworth = phi_z * correction

        # Ensure non-negative
        pdf_edgeworth = np.maximum(pdf_edgeworth, 0)

        # Evaluate fit
        # Compare to empirical histogram
        hist, bin_edges = np.histogram(z, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        phi_bins = stats.norm.pdf(bin_centers)
        correction_bins = (
            1 +
            k3 / 6 * self._hermite(bin_centers, 3) +
            k4 / 24 * self._hermite(bin_centers, 4)
        )
        pdf_bins = np.maximum(phi_bins * correction_bins, 0)

        # MSE
        mse_gaussian = np.mean((hist - phi_bins) ** 2)
        mse_edgeworth = np.mean((hist - pdf_bins) ** 2)

        return {
            "cumulants": cumulants,
            "skewness": float(k3),
            "excess_kurtosis": float(k4),
            "mean": float(mu),
            "std": float(sigma),
            "mse_gaussian": float(mse_gaussian),
            "mse_edgeworth": float(mse_edgeworth),
            "improvement_ratio": float(mse_gaussian / (mse_edgeworth + 1e-10)),
            "is_heavy_tailed": k4 > 1,
            "is_skewed": abs(k3) > 0.5
        }

    def _compute_cumulants(
        self,
        z: np.ndarray
    ) -> Dict[str, float]:
        """Compute cumulants from standardized data."""
        n = len(z)

        # Raw moments
        m2 = np.mean(z ** 2)
        m3 = np.mean(z ** 3)
        m4 = np.mean(z ** 4)
        m5 = np.mean(z ** 5) if self.order >= 5 else 0
        m6 = np.mean(z ** 6) if self.order >= 6 else 0

        # Cumulants (standardized, so κ₁=0, κ₂=1)
        kappa_3 = m3  # Skewness
        kappa_4 = m4 - 3  # Excess kurtosis

        # Higher cumulants
        kappa_5 = m5 - 10 * m3 if self.order >= 5 else 0
        kappa_6 = m6 - 15 * m4 - 10 * m3 ** 2 + 30 if self.order >= 6 else 0

        return {
            "kappa_2": float(m2),
            "kappa_3": float(kappa_3),
            "kappa_4": float(kappa_4),
            "kappa_5": float(kappa_5),
            "kappa_6": float(kappa_6)
        }

    def _hermite(
        self,
        x: np.ndarray,
        n: int
    ) -> np.ndarray:
        """Probabilists' Hermite polynomial He_n(x)."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        elif n == 2:
            return x ** 2 - 1
        elif n == 3:
            return x ** 3 - 3 * x
        elif n == 4:
            return x ** 4 - 6 * x ** 2 + 3
        elif n == 5:
            return x ** 5 - 10 * x ** 3 + 15 * x
        elif n == 6:
            return x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
        else:
            # Recurrence: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
            H_nm1 = self._hermite(x, n - 1)
            H_nm2 = self._hermite(x, n - 2)
            return x * H_nm1 - (n - 1) * H_nm2

    def _default_result(self) -> Dict[str, Any]:
        return {
            "cumulants": {},
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "improvement_ratio": 1.0
        }


class CornishFisherExpansion:
    """
    Cornish-Fisher Expansion for VaR and risk quantile correction.

    Corrects Gaussian quantiles for skewness and kurtosis:
    z_α^CF ≈ z_α + (z_α²-1)γ₁/6 + (z_α³-3z_α)γ₂/24 - (2z_α³-5z_α)γ₁²/36

    Where:
    - γ₁ = skewness
    - γ₂ = excess kurtosis
    """

    def __init__(self):
        pass

    def compute_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute Cornish-Fisher adjusted VaR.
        """
        if len(returns) < 30:
            return self._default_result()

        mu = np.mean(returns)
        sigma = np.std(returns)

        # Skewness and kurtosis
        z = (returns - mu) / (sigma + 1e-10)
        gamma1 = np.mean(z ** 3)  # Skewness
        gamma2 = np.mean(z ** 4) - 3  # Excess kurtosis

        # Gaussian quantile
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(alpha)

        # Cornish-Fisher correction
        z_cf = (
            z_alpha +
            (z_alpha ** 2 - 1) * gamma1 / 6 +
            (z_alpha ** 3 - 3 * z_alpha) * gamma2 / 24 -
            (2 * z_alpha ** 3 - 5 * z_alpha) * gamma1 ** 2 / 36
        )

        # VaR values
        var_gaussian = mu + z_alpha * sigma
        var_cf = mu + z_cf * sigma
        var_historical = np.percentile(returns, alpha * 100)

        # CVaR (Expected Shortfall)
        # For Cornish-Fisher, approximate by averaging below VaR
        tail_returns = returns[returns < var_cf]
        cvar_cf = np.mean(tail_returns) if len(tail_returns) > 0 else var_cf

        tail_returns_hist = returns[returns < var_historical]
        cvar_historical = np.mean(tail_returns_hist) if len(tail_returns_hist) > 0 else var_historical

        return {
            "var_gaussian": float(var_gaussian),
            "var_cornish_fisher": float(var_cf),
            "var_historical": float(var_historical),
            "cvar_cornish_fisher": float(cvar_cf),
            "cvar_historical": float(cvar_historical),
            "z_gaussian": float(z_alpha),
            "z_cornish_fisher": float(z_cf),
            "skewness": float(gamma1),
            "excess_kurtosis": float(gamma2),
            "var_adjustment": float(var_cf - var_gaussian),
            "is_left_skewed": gamma1 < -0.3,
            "is_fat_tailed": gamma2 > 1
        }

    def compute_quantiles(
        self,
        returns: np.ndarray,
        probabilities: List[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple quantiles with Cornish-Fisher correction.
        """
        if len(returns) < 30:
            return {"quantiles": {}}

        mu = np.mean(returns)
        sigma = np.std(returns)
        z = (returns - mu) / (sigma + 1e-10)
        gamma1 = np.mean(z ** 3)
        gamma2 = np.mean(z ** 4) - 3

        quantiles = {}
        for p in probabilities:
            z_p = stats.norm.ppf(p)

            # Cornish-Fisher
            z_cf = (
                z_p +
                (z_p ** 2 - 1) * gamma1 / 6 +
                (z_p ** 3 - 3 * z_p) * gamma2 / 24 -
                (2 * z_p ** 3 - 5 * z_p) * gamma1 ** 2 / 36
            )

            quantiles[f"p_{int(p*100)}"] = {
                "gaussian": float(mu + z_p * sigma),
                "cornish_fisher": float(mu + z_cf * sigma),
                "historical": float(np.percentile(returns, p * 100))
            }

        return {"quantiles": quantiles}

    def _default_result(self) -> Dict[str, float]:
        return {
            "var_gaussian": 0.0,
            "var_cornish_fisher": 0.0,
            "var_historical": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0
        }


class MomentGeneratingFunction:
    """
    Moment Generating Function analysis.

    M_X(t) = E[e^{tX}]

    Useful for:
    - Computing moments: E[X^n] = M^(n)(0)
    - Distribution characterization
    - Tail behavior analysis
    """

    def __init__(
        self,
        t_range: Tuple[float, float] = (-2.0, 2.0),
        n_points: int = 100
    ):
        self.t_range = t_range
        self.n_points = n_points

    def compute(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute empirical MGF and derived properties.
        """
        if len(returns) < 30:
            return self._default_result()

        # Standardize
        mu = np.mean(returns)
        sigma = np.std(returns)
        z = (returns - mu) / (sigma + 1e-10)

        # Evaluate MGF at grid points
        t_values = np.linspace(self.t_range[0], self.t_range[1], self.n_points)
        mgf_values = np.zeros(self.n_points)

        for i, t in enumerate(t_values):
            # Empirical MGF: M(t) ≈ (1/n) Σ exp(t*x_i)
            mgf_values[i] = np.mean(np.exp(t * z))

        # Log-MGF (cumulant generating function)
        log_mgf = np.log(np.maximum(mgf_values, 1e-10))

        # Moments from numerical differentiation of log-MGF at t=0
        center_idx = self.n_points // 2
        dt = t_values[1] - t_values[0]

        # First derivative (mean)
        kappa_1 = (log_mgf[center_idx + 1] - log_mgf[center_idx - 1]) / (2 * dt)

        # Second derivative (variance)
        kappa_2 = (log_mgf[center_idx + 1] - 2 * log_mgf[center_idx] + log_mgf[center_idx - 1]) / (dt ** 2)

        # Check for heavy tails (MGF explosion)
        # Heavy tails: MGF undefined for some t
        max_stable_t = self._find_max_stable_t(z)

        # Compare to Gaussian MGF
        mgf_gaussian = np.exp(t_values ** 2 / 2)
        divergence = np.mean((mgf_values - mgf_gaussian) ** 2)

        return {
            "t_values": t_values,
            "mgf_values": mgf_values,
            "log_mgf_values": log_mgf,
            "mean_from_mgf": float(kappa_1),
            "variance_from_mgf": float(kappa_2),
            "actual_mean": float(np.mean(z)),
            "actual_variance": float(np.var(z)),
            "max_stable_t": float(max_stable_t),
            "gaussian_divergence": float(divergence),
            "has_light_tails": max_stable_t > 1.5,
            "has_heavy_tails": max_stable_t < 0.5
        }

    def _find_max_stable_t(
        self,
        z: np.ndarray
    ) -> float:
        """Find maximum t where MGF is stable."""
        for t in np.linspace(0.1, 3.0, 30):
            try:
                val = np.mean(np.exp(t * z))
                if np.isinf(val) or np.isnan(val) or val > 1e10:
                    return t - 0.1
            except Exception:
                return t - 0.1
        return 3.0

    def _default_result(self) -> Dict[str, Any]:
        return {
            "t_values": np.array([]),
            "mgf_values": np.array([]),
            "max_stable_t": 0.0
        }


class ExpansionEngine:
    """
    Unified engine for Taylor series and asymptotic expansions.
    """

    def __init__(self):
        self.taylor = TaylorSeriesAnalyzer(order=4)
        self.edgeworth = EdgeworthExpansion(order=4)
        self.cornish_fisher = CornishFisherExpansion()
        self.mgf = MomentGeneratingFunction()

    def analyze(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> ExpansionResult:
        """
        Comprehensive expansion analysis.
        """
        if len(prices) < 30:
            return self._default_result()

        if returns is None:
            returns = np.diff(np.log(prices))

        # Taylor series on prices
        taylor_result = self.taylor.expand_price_dynamics(prices)

        # Edgeworth on returns
        edgeworth_result = self.edgeworth.fit(returns)

        # Cornish-Fisher VaR
        cf_result = self.cornish_fisher.compute_var(returns)

        # MGF analysis
        mgf_result = self.mgf.compute(returns)

        # Combine results
        coefficients = taylor_result.get("coefficients", np.array([]))

        approximation = taylor_result.get("forecasts", np.array([]))

        residual_error = taylor_result.get("residual_error", 1.0)
        convergence_radius = taylor_result.get("convergence_radius", float('inf'))

        moments = {
            "mean": edgeworth_result.get("mean", 0),
            "std": edgeworth_result.get("std", 1),
            "skewness": edgeworth_result.get("skewness", 0),
            "excess_kurtosis": edgeworth_result.get("excess_kurtosis", 0)
        }

        cumulants = edgeworth_result.get("cumulants", {})

        risk_metrics = {
            "var_95_gaussian": cf_result.get("var_gaussian", 0),
            "var_95_cf": cf_result.get("var_cornish_fisher", 0),
            "var_95_historical": cf_result.get("var_historical", 0),
            "cvar_95_cf": cf_result.get("cvar_cornish_fisher", 0)
        }

        return ExpansionResult(
            coefficients=coefficients,
            approximation=approximation,
            residual_error=residual_error,
            convergence_radius=convergence_radius,
            moments=moments,
            cumulants=cumulants,
            risk_metrics=risk_metrics
        )

    def _default_result(self) -> ExpansionResult:
        return ExpansionResult(
            coefficients=np.array([]),
            approximation=np.array([]),
            residual_error=1.0,
            convergence_radius=0.0,
            moments={},
            cumulants={},
            risk_metrics={}
        )
