"""
Copula Models for Tail Dependence Analysis

Implements copula methods for modeling multivariate dependence:
- Gaussian Copula
- Student-t Copula
- Clayton Copula (lower tail dependence)
- Gumbel Copula (upper tail dependence)
- Frank Copula (symmetric, no tail dependence)

Applications:
- Portfolio risk modeling
- Tail risk estimation
- Correlation breakdown detection
- Stress testing

Mathematical Foundation:
- Sklar's Theorem: F(x,y) = C(F_X(x), F_Y(y))
- Tail dependence: λ_L = lim_{u→0} P(U₂<u|U₁<u)
- Kendall's τ for copula parameter estimation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class CopulaResult:
    """Result of copula analysis"""
    copula_type: str
    parameters: Dict[str, float]
    kendall_tau: float
    spearman_rho: float
    lower_tail_dependence: float
    upper_tail_dependence: float
    log_likelihood: float
    aic: float
    bic: float
    tail_risk_metrics: Dict[str, float]


class GaussianCopula:
    """
    Gaussian (Normal) Copula.

    C(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))

    Properties:
    - No tail dependence (λ_L = λ_U = 0 for |ρ| < 1)
    - Kendall's τ = (2/π) arcsin(ρ)
    """

    def __init__(self):
        pass

    def fit(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Dict[str, Any]:
        """Fit Gaussian copula to pseudo-observations."""
        if len(u) < 20:
            return self._default_result()

        # Empirical Kendall's tau
        tau = self._kendall_tau(u, v)

        # Invert: ρ = sin(π τ / 2)
        rho = np.sin(np.pi * tau / 2)
        rho = np.clip(rho, -0.99, 0.99)

        # Log-likelihood
        ll = self._log_likelihood(u, v, rho)

        n = len(u)
        aic = 2 * 1 - 2 * ll
        bic = 1 * np.log(n) - 2 * ll

        return {
            "rho": float(rho),
            "kendall_tau": float(tau),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "lower_tail_dependence": 0.0,  # Always 0 for Gaussian
            "upper_tail_dependence": 0.0
        }

    def _kendall_tau(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> float:
        """Compute Kendall's tau."""
        n = len(u)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                sign_u = np.sign(u[i] - u[j])
                sign_v = np.sign(v[i] - v[j])
                if sign_u * sign_v > 0:
                    concordant += 1
                elif sign_u * sign_v < 0:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.0
        return (concordant - discordant) / total

    def _log_likelihood(
        self,
        u: np.ndarray,
        v: np.ndarray,
        rho: float
    ) -> float:
        """Gaussian copula log-likelihood."""
        # Transform to normal
        x = stats.norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        y = stats.norm.ppf(np.clip(v, 1e-10, 1-1e-10))

        # Copula density
        # c(u,v) = (1/√(1-ρ²)) exp(-(ρ²(x²+y²) - 2ρxy)/(2(1-ρ²)))
        rho2 = rho ** 2

        if abs(1 - rho2) < 1e-10:
            return float('-inf')

        log_c = (
            -0.5 * np.log(1 - rho2) -
            (rho2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho2))
        )

        return float(np.sum(log_c))

    def _default_result(self) -> Dict[str, Any]:
        return {
            "rho": 0.0,
            "kendall_tau": 0.0,
            "lower_tail_dependence": 0.0,
            "upper_tail_dependence": 0.0
        }


class StudentTCopula:
    """
    Student-t Copula.

    Has tail dependence:
    λ = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

    Better for financial data than Gaussian.
    """

    def __init__(self):
        pass

    def fit(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Dict[str, Any]:
        """Fit Student-t copula."""
        if len(u) < 20:
            return self._default_result()

        # Estimate Kendall's tau
        tau = self._kendall_tau(u, v)
        rho = np.sin(np.pi * tau / 2)
        rho = np.clip(rho, -0.99, 0.99)

        # Estimate degrees of freedom via MLE
        def neg_ll(nu):
            if nu < 2.1:
                return float('inf')
            return -self._log_likelihood(u, v, rho, nu)

        result = minimize_scalar(neg_ll, bounds=(2.1, 100), method='bounded')
        nu = result.x

        ll = -result.fun
        n = len(u)
        aic = 2 * 2 - 2 * ll
        bic = 2 * np.log(n) - 2 * ll

        # Tail dependence
        lambda_tail = self._tail_dependence(rho, nu)

        return {
            "rho": float(rho),
            "nu": float(nu),
            "kendall_tau": float(tau),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "lower_tail_dependence": float(lambda_tail),
            "upper_tail_dependence": float(lambda_tail)  # Symmetric
        }

    def _kendall_tau(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> float:
        """Compute Kendall's tau."""
        n = len(u)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                sign_u = np.sign(u[i] - u[j])
                sign_v = np.sign(v[i] - v[j])
                if sign_u * sign_v > 0:
                    concordant += 1
                elif sign_u * sign_v < 0:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.0
        return (concordant - discordant) / total

    def _log_likelihood(
        self,
        u: np.ndarray,
        v: np.ndarray,
        rho: float,
        nu: float
    ) -> float:
        """Student-t copula log-likelihood."""
        # Transform to t-quantiles
        x = stats.t.ppf(np.clip(u, 1e-10, 1-1e-10), nu)
        y = stats.t.ppf(np.clip(v, 1e-10, 1-1e-10), nu)

        n = len(u)
        rho2 = rho ** 2

        if abs(1 - rho2) < 1e-10:
            return float('-inf')

        # Log copula density
        # Using the formula for bivariate t copula
        q = (x**2 + y**2 - 2*rho*x*y) / (1 - rho2)

        log_c = (
            gammaln((nu + 2) / 2) - gammaln(nu / 2) -
            np.log(np.pi * nu * np.sqrt(1 - rho2)) +
            (nu / 2 + 1) * (np.log(1 + x**2/nu) + np.log(1 + y**2/nu)) -
            (nu + 2) / 2 * np.log(1 + q / nu)
        )

        return float(np.sum(log_c))

    def _tail_dependence(
        self,
        rho: float,
        nu: float
    ) -> float:
        """Compute tail dependence coefficient."""
        # λ = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
        if abs(1 + rho) < 1e-10:
            return 0.0

        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_tail = 2 * stats.t.cdf(arg, nu + 1)

        return float(lambda_tail)

    def _default_result(self) -> Dict[str, Any]:
        return {
            "rho": 0.0,
            "nu": 10.0,
            "lower_tail_dependence": 0.0,
            "upper_tail_dependence": 0.0
        }


class ClaytonCopula:
    """
    Clayton Copula.

    C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)

    Properties:
    - Lower tail dependence: λ_L = 2^(-1/θ)
    - Upper tail dependence: λ_U = 0
    - Kendall's τ = θ/(θ+2)
    """

    def __init__(self):
        pass

    def fit(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Dict[str, Any]:
        """Fit Clayton copula."""
        if len(u) < 20:
            return self._default_result()

        # Estimate from Kendall's tau
        tau = self._kendall_tau(u, v)

        if tau <= 0:
            # Clayton only for positive dependence
            return self._default_result()

        # θ = 2τ/(1-τ)
        theta = 2 * tau / (1 - tau + 1e-10)
        theta = max(theta, 0.01)

        # Log-likelihood
        ll = self._log_likelihood(u, v, theta)

        n = len(u)
        aic = 2 * 1 - 2 * ll
        bic = 1 * np.log(n) - 2 * ll

        # Tail dependence
        lambda_L = 2 ** (-1/theta)

        return {
            "theta": float(theta),
            "kendall_tau": float(tau),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "lower_tail_dependence": float(lambda_L),
            "upper_tail_dependence": 0.0
        }

    def _kendall_tau(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> float:
        """Compute Kendall's tau."""
        n = len(u)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                sign_u = np.sign(u[i] - u[j])
                sign_v = np.sign(v[i] - v[j])
                if sign_u * sign_v > 0:
                    concordant += 1
                elif sign_u * sign_v < 0:
                    discordant += 1

        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0

    def _log_likelihood(
        self,
        u: np.ndarray,
        v: np.ndarray,
        theta: float
    ) -> float:
        """Clayton copula log-likelihood."""
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)

        # Copula density
        # c(u,v) = (1+θ)(uv)^(-1-θ) (u^(-θ) + v^(-θ) - 1)^(-2-1/θ)
        log_c = (
            np.log(1 + theta) -
            (1 + theta) * (np.log(u) + np.log(v)) -
            (2 + 1/theta) * np.log(u**(-theta) + v**(-theta) - 1)
        )

        return float(np.sum(log_c[np.isfinite(log_c)]))

    def _default_result(self) -> Dict[str, Any]:
        return {
            "theta": 1.0,
            "lower_tail_dependence": 0.5,
            "upper_tail_dependence": 0.0
        }


class GumbelCopula:
    """
    Gumbel Copula.

    C(u,v) = exp(-((-log u)^θ + (-log v)^θ)^(1/θ))

    Properties:
    - Upper tail dependence: λ_U = 2 - 2^(1/θ)
    - Lower tail dependence: λ_L = 0
    - Kendall's τ = 1 - 1/θ
    """

    def __init__(self):
        pass

    def fit(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Dict[str, Any]:
        """Fit Gumbel copula."""
        if len(u) < 20:
            return self._default_result()

        tau = self._kendall_tau(u, v)

        if tau <= 0:
            return self._default_result()

        # θ = 1/(1-τ)
        theta = 1 / (1 - tau + 1e-10)
        theta = max(theta, 1.01)

        ll = self._log_likelihood(u, v, theta)

        n = len(u)
        aic = 2 * 1 - 2 * ll
        bic = 1 * np.log(n) - 2 * ll

        lambda_U = 2 - 2 ** (1/theta)

        return {
            "theta": float(theta),
            "kendall_tau": float(tau),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "lower_tail_dependence": 0.0,
            "upper_tail_dependence": float(lambda_U)
        }

    def _kendall_tau(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> float:
        """Compute Kendall's tau."""
        n = len(u)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                sign_u = np.sign(u[i] - u[j])
                sign_v = np.sign(v[i] - v[j])
                if sign_u * sign_v > 0:
                    concordant += 1
                elif sign_u * sign_v < 0:
                    discordant += 1

        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0

    def _log_likelihood(
        self,
        u: np.ndarray,
        v: np.ndarray,
        theta: float
    ) -> float:
        """Gumbel copula log-likelihood."""
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)

        neg_log_u = -np.log(u)
        neg_log_v = -np.log(v)

        A = (neg_log_u ** theta + neg_log_v ** theta) ** (1/theta)

        # Copula density (simplified)
        log_c = (
            -A +
            (theta - 1) * (np.log(neg_log_u) + np.log(neg_log_v)) +
            (1/theta - 2) * np.log(neg_log_u ** theta + neg_log_v ** theta) +
            np.log(A + theta - 1)
        )

        return float(np.sum(log_c[np.isfinite(log_c)]))

    def _default_result(self) -> Dict[str, Any]:
        return {
            "theta": 2.0,
            "lower_tail_dependence": 0.0,
            "upper_tail_dependence": 0.29
        }


class CopulaEngine:
    """
    Unified Copula Analysis Engine.
    """

    def __init__(self):
        self.gaussian = GaussianCopula()
        self.student_t = StudentTCopula()
        self.clayton = ClaytonCopula()
        self.gumbel = GumbelCopula()

    def analyze(
        self,
        returns_x: np.ndarray,
        returns_y: np.ndarray
    ) -> CopulaResult:
        """
        Fit all copula models and select best.
        """
        if len(returns_x) < 30 or len(returns_x) != len(returns_y):
            return self._default_result()

        # Transform to pseudo-observations (uniform marginals)
        n = len(returns_x)
        u = (stats.rankdata(returns_x) - 0.5) / n
        v = (stats.rankdata(returns_y) - 0.5) / n

        # Fit all copulas
        gaussian_fit = self.gaussian.fit(u, v)
        t_fit = self.student_t.fit(u, v)
        clayton_fit = self.clayton.fit(u, v)
        gumbel_fit = self.gumbel.fit(u, v)

        # Compare by BIC
        models = {
            "gaussian": gaussian_fit,
            "student_t": t_fit,
            "clayton": clayton_fit,
            "gumbel": gumbel_fit
        }

        best_model = min(models, key=lambda k: models[k].get("bic", float('inf')))
        best_fit = models[best_model]

        # Tail risk metrics
        tail_risk = self._compute_tail_risk(returns_x, returns_y, best_fit)

        # Spearman's rho
        spearman_rho = stats.spearmanr(returns_x, returns_y)[0]

        return CopulaResult(
            copula_type=best_model,
            parameters={k: v for k, v in best_fit.items()
                       if k not in ["log_likelihood", "aic", "bic",
                                   "lower_tail_dependence", "upper_tail_dependence",
                                   "kendall_tau"]},
            kendall_tau=best_fit.get("kendall_tau", 0),
            spearman_rho=float(spearman_rho) if not np.isnan(spearman_rho) else 0,
            lower_tail_dependence=best_fit.get("lower_tail_dependence", 0),
            upper_tail_dependence=best_fit.get("upper_tail_dependence", 0),
            log_likelihood=best_fit.get("log_likelihood", 0),
            aic=best_fit.get("aic", 0),
            bic=best_fit.get("bic", 0),
            tail_risk_metrics=tail_risk
        )

    def _compute_tail_risk(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fit: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute tail risk metrics."""
        # Empirical joint extreme probability
        q_x = np.percentile(x, 5)
        q_y = np.percentile(y, 5)

        joint_extreme = np.mean((x < q_x) & (y < q_y))
        marginal_x = np.mean(x < q_x)
        marginal_y = np.mean(y < q_y)

        # Conditional probability
        if marginal_x > 0:
            cond_prob = joint_extreme / marginal_x
        else:
            cond_prob = 0

        # Expected shortfall correlation
        tail_mask = (x < q_x) | (y < q_y)
        if np.sum(tail_mask) > 5:
            tail_corr = np.corrcoef(x[tail_mask], y[tail_mask])[0, 1]
        else:
            tail_corr = np.corrcoef(x, y)[0, 1]

        return {
            "joint_extreme_prob": float(joint_extreme),
            "conditional_extreme_prob": float(cond_prob),
            "tail_correlation": float(tail_corr) if not np.isnan(tail_corr) else 0,
            "independence_prob": float(marginal_x * marginal_y),
            "dependence_ratio": float(joint_extreme / (marginal_x * marginal_y + 1e-10))
        }

    def _default_result(self) -> CopulaResult:
        return CopulaResult(
            copula_type="gaussian",
            parameters={"rho": 0.0},
            kendall_tau=0.0,
            spearman_rho=0.0,
            lower_tail_dependence=0.0,
            upper_tail_dependence=0.0,
            log_likelihood=0.0,
            aic=0.0,
            bic=0.0,
            tail_risk_metrics={}
        )
