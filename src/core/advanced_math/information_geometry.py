"""
Information Geometry Module for Stock Prediction

Implements information-theoretic and geometric methods:
- Fisher Information Matrix (parameter sensitivity)
- Kullback-Leibler Divergence (distribution shift detection)
- Riemannian Metrics for probability manifolds
- Natural Gradient for optimization
- α-divergences (Tsallis, Rényi)
- Mutual Information for feature relevance

Mathematical Foundation:
- Fisher Information: I(θ) = E[(∂log p/∂θ)²]
- KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
- Natural Gradient: θ̃ = F⁻¹ ∇θL
- Riemannian distance: d(P,Q) = √(2(1 - ∫√(pq)dx))
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import gamma as gamma_func, digamma
import logging

logger = logging.getLogger(__name__)


@dataclass
class InformationGeometryResult:
    """Result of information geometry analysis"""
    fisher_information: np.ndarray  # Fisher information matrix
    kl_divergence: float  # KL divergence from baseline
    hellinger_distance: float  # Hellinger distance
    mutual_information: float  # Feature-target MI
    entropy: float  # Shannon entropy
    regime_divergence: Dict[str, float]  # Divergence from different regimes
    natural_gradient: np.ndarray  # Natural gradient direction
    cramer_rao_bound: float  # Theoretical variance lower bound
    information_velocity: float  # Rate of information change
    predictability_score: float  # 0-1 score for predictability


class FisherInformationAnalyzer:
    """
    Fisher Information Matrix analysis for trading signals.

    The Fisher Information quantifies how much information data carries
    about unknown parameters. For trading:
    - High FI: Parameters are precisely estimable → stable predictions
    - Low FI: High uncertainty → less reliable predictions

    I(θ) = -E[∂²log p(X|θ)/∂θ²] = E[(∂log p(X|θ)/∂θ)²]
    """

    def __init__(
        self,
        distribution_family: str = "gaussian",
        n_bootstrap: int = 100
    ):
        self.distribution_family = distribution_family
        self.n_bootstrap = n_bootstrap

    def compute(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute Fisher Information Matrix for return distribution.
        """
        if len(returns) < 20:
            return self._default_result()

        if self.distribution_family == "gaussian":
            return self._compute_gaussian_fisher(returns)
        elif self.distribution_family == "student_t":
            return self._compute_student_t_fisher(returns)
        elif self.distribution_family == "mixture":
            return self._compute_mixture_fisher(returns)
        else:
            return self._compute_gaussian_fisher(returns)

    def _compute_gaussian_fisher(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fisher Information for Gaussian distribution.

        For N(μ, σ²):
        I = [[1/σ², 0], [0, 1/(2σ⁴)]]

        Cramér-Rao bounds:
        Var(μ̂) ≥ σ²/n
        Var(σ̂²) ≥ 2σ⁴/n
        """
        n = len(returns)
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Fisher Information Matrix
        fisher = np.array([
            [1 / (sigma ** 2 + 1e-10), 0],
            [0, 1 / (2 * sigma ** 4 + 1e-10)]
        ])

        # Cramér-Rao lower bound for variance
        cr_mu = sigma ** 2 / n
        cr_sigma = 2 * sigma ** 4 / n

        # Total information (trace)
        total_info = np.trace(fisher)

        # Effective sample size (accounting for autocorrelation)
        ess = self._effective_sample_size(returns)

        # Adjusted CR bound
        adjusted_cr = cr_mu * n / ess

        # Information ratio (actual variance / CR bound)
        actual_var = np.var(returns) / n
        info_ratio = actual_var / (cr_mu + 1e-10)

        # Eigenvalue analysis for confidence
        eigenvalues = np.linalg.eigvals(fisher)

        return {
            "fisher_matrix": fisher,
            "total_information": float(total_info),
            "cramer_rao_bound_mu": float(cr_mu),
            "cramer_rao_bound_sigma": float(cr_sigma),
            "adjusted_cr_bound": float(adjusted_cr),
            "effective_sample_size": float(ess),
            "information_ratio": float(info_ratio),
            "eigenvalues": np.real(eigenvalues).tolist(),
            "condition_number": float(np.max(np.real(eigenvalues)) / (np.min(np.real(eigenvalues)) + 1e-10)),
            "parameter_estimates": {"mu": float(mu), "sigma": float(sigma)},
            "estimation_confidence": float(np.clip(1 / (1 + adjusted_cr), 0, 1))
        }

    def _compute_student_t_fisher(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fisher Information for Student-t distribution.

        Better for fat-tailed financial returns.
        """
        n = len(returns)

        # Fit t-distribution
        nu, loc, scale = stats.t.fit(returns)
        nu = max(nu, 2.1)  # Ensure finite variance

        # Fisher Information for t-distribution
        # Approximation for location and scale
        fisher_mu = (nu + 1) / ((nu + 3) * scale ** 2)
        fisher_sigma = nu / (2 * scale ** 4)
        fisher_nu = self._fisher_nu_t(nu)

        fisher = np.diag([fisher_mu, fisher_sigma, fisher_nu])

        # Effective degrees of freedom
        excess_kurtosis = 6 / (nu - 4) if nu > 4 else 10
        eff_n = n / (1 + excess_kurtosis / 3)

        return {
            "fisher_matrix": fisher,
            "total_information": float(np.trace(fisher)),
            "degrees_of_freedom": float(nu),
            "location": float(loc),
            "scale": float(scale),
            "effective_sample_size": float(eff_n),
            "tail_heaviness": float(excess_kurtosis),
            "estimation_confidence": float(np.clip(eff_n / 50, 0, 1))
        }

    def _compute_mixture_fisher(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fisher Information for Gaussian Mixture.

        Models regime switching in markets.
        """
        n = len(returns)

        # Simple 2-component EM
        # Initialize with k-means style
        median = np.median(returns)
        group1 = returns[returns < median]
        group2 = returns[returns >= median]

        if len(group1) < 2 or len(group2) < 2:
            return self._compute_gaussian_fisher(returns)

        # EM iterations
        mu1, sigma1 = np.mean(group1), np.std(group1) + 1e-6
        mu2, sigma2 = np.mean(group2), np.std(group2) + 1e-6
        pi = len(group1) / n

        for _ in range(20):
            # E-step
            p1 = pi * stats.norm.pdf(returns, mu1, sigma1)
            p2 = (1 - pi) * stats.norm.pdf(returns, mu2, sigma2)
            total_p = p1 + p2 + 1e-10

            gamma1 = p1 / total_p
            gamma2 = p2 / total_p

            # M-step
            n1 = np.sum(gamma1)
            n2 = np.sum(gamma2)

            if n1 > 1 and n2 > 1:
                pi = n1 / n
                mu1 = np.sum(gamma1 * returns) / n1
                mu2 = np.sum(gamma2 * returns) / n2
                sigma1 = np.sqrt(np.sum(gamma1 * (returns - mu1) ** 2) / n1) + 1e-6
                sigma2 = np.sqrt(np.sum(gamma2 * (returns - mu2) ** 2) / n2) + 1e-6

        # Fisher for mixture (incomplete data)
        # Lower bound using observed information
        fisher_approx = np.zeros((5, 5))
        fisher_approx[0, 0] = n1 / (sigma1 ** 2)
        fisher_approx[1, 1] = n1 / (2 * sigma1 ** 4)
        fisher_approx[2, 2] = n2 / (sigma2 ** 2)
        fisher_approx[3, 3] = n2 / (2 * sigma2 ** 4)
        fisher_approx[4, 4] = n / (pi * (1 - pi) + 1e-10)

        # Regime classification
        current_prob_regime1 = pi * stats.norm.pdf(returns[-1], mu1, sigma1) / \
                               (pi * stats.norm.pdf(returns[-1], mu1, sigma1) +
                                (1-pi) * stats.norm.pdf(returns[-1], mu2, sigma2) + 1e-10)

        return {
            "fisher_matrix": fisher_approx,
            "total_information": float(np.trace(fisher_approx)),
            "regime_1": {"mu": float(mu1), "sigma": float(sigma1), "weight": float(pi)},
            "regime_2": {"mu": float(mu2), "sigma": float(sigma2), "weight": float(1-pi)},
            "current_regime_1_probability": float(current_prob_regime1),
            "regime_separation": float(abs(mu1 - mu2) / (sigma1 + sigma2)),
            "estimation_confidence": float(np.clip(min(n1, n2) / 30, 0, 1))
        }

    def _fisher_nu_t(self, nu: float) -> float:
        """Fisher information for degrees of freedom parameter."""
        if nu <= 2:
            return 0.01
        return 0.5 * (digamma((nu + 1) / 2) - digamma(nu / 2) - 1 / nu)

    def _effective_sample_size(self, returns: np.ndarray) -> float:
        """
        Compute effective sample size accounting for autocorrelation.
        ESS = n / (1 + 2Σᵏ ρₖ)
        """
        n = len(returns)
        if n < 10:
            return float(n)

        # Compute autocorrelations
        max_lag = min(20, n // 3)
        autocorrs = []

        for lag in range(1, max_lag + 1):
            corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            if np.isnan(corr):
                break
            autocorrs.append(corr)

        # ESS formula
        sum_rho = sum(autocorrs)
        ess = n / (1 + 2 * sum_rho)

        return max(1.0, float(ess))

    def _default_result(self) -> Dict[str, Any]:
        return {
            "fisher_matrix": np.eye(2),
            "total_information": 2.0,
            "cramer_rao_bound_mu": 1.0,
            "estimation_confidence": 0.0
        }


class KLDivergenceAnalyzer:
    """
    Kullback-Leibler Divergence for regime change detection.

    D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))

    Measures "surprise" when using Q to encode samples from P.
    For trading: detect when current distribution shifts from baseline.
    """

    def __init__(
        self,
        n_bins: int = 50,
        baseline_window: int = 60,
        current_window: int = 20
    ):
        self.n_bins = n_bins
        self.baseline_window = baseline_window
        self.current_window = current_window

    def compute(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute KL divergence between current and baseline distributions.
        """
        if len(returns) < self.baseline_window + self.current_window:
            return self._default_result()

        baseline = returns[-(self.baseline_window + self.current_window):-self.current_window]
        current = returns[-self.current_window:]

        # Histogram-based KL divergence
        all_returns = np.concatenate([baseline, current])
        bins = np.linspace(np.min(all_returns), np.max(all_returns), self.n_bins + 1)

        p_baseline, _ = np.histogram(baseline, bins=bins, density=True)
        p_current, _ = np.histogram(current, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_baseline = p_baseline + eps
        p_current = p_current + eps

        # Normalize
        p_baseline = p_baseline / p_baseline.sum()
        p_current = p_current / p_current.sum()

        # KL divergence
        kl_forward = np.sum(p_current * np.log(p_current / p_baseline))
        kl_backward = np.sum(p_baseline * np.log(p_baseline / p_current))

        # Symmetric KL (Jensen-Shannon)
        m = 0.5 * (p_baseline + p_current)
        js_divergence = 0.5 * (
            np.sum(p_baseline * np.log(p_baseline / m)) +
            np.sum(p_current * np.log(p_current / m))
        )

        # Hellinger distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(p_baseline) - np.sqrt(p_current)) ** 2))

        # Total variation distance
        tv_distance = 0.5 * np.sum(np.abs(p_baseline - p_current))

        # Parametric comparison
        mu_baseline, sigma_baseline = np.mean(baseline), np.std(baseline)
        mu_current, sigma_current = np.mean(current), np.std(current)

        # Gaussian KL (closed form)
        if sigma_baseline > 0 and sigma_current > 0:
            kl_gaussian = (
                np.log(sigma_current / sigma_baseline) +
                (sigma_baseline ** 2 + (mu_baseline - mu_current) ** 2) / (2 * sigma_current ** 2) -
                0.5
            )
        else:
            kl_gaussian = 0

        # Regime change detection
        is_regime_change = kl_forward > 0.1 or abs(mu_current - mu_baseline) > 2 * sigma_baseline

        return {
            "kl_divergence_forward": float(kl_forward),
            "kl_divergence_backward": float(kl_backward),
            "js_divergence": float(js_divergence),
            "hellinger_distance": float(hellinger),
            "tv_distance": float(tv_distance),
            "kl_gaussian": float(kl_gaussian),
            "baseline_mu": float(mu_baseline),
            "baseline_sigma": float(sigma_baseline),
            "current_mu": float(mu_current),
            "current_sigma": float(sigma_current),
            "is_regime_change": is_regime_change,
            "regime_change_confidence": float(np.clip(kl_forward / 0.2, 0, 1))
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "kl_divergence_forward": 0.0,
            "kl_divergence_backward": 0.0,
            "js_divergence": 0.0,
            "hellinger_distance": 0.0,
            "is_regime_change": False,
            "regime_change_confidence": 0.0
        }


class MutualInformationAnalyzer:
    """
    Mutual Information for feature-target relationship analysis.

    I(X;Y) = H(X) + H(Y) - H(X,Y)
           = Σ p(x,y) log(p(x,y) / (p(x)p(y)))

    Quantifies how much knowing X reduces uncertainty about Y.
    """

    def __init__(
        self,
        n_bins: int = 20
    ):
        self.n_bins = n_bins

    def compute(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute mutual information between features and target.
        """
        if len(features) != len(target) or len(features) < 20:
            return self._default_result()

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_features = features.shape[1]
        mi_scores = []
        normalized_mi = []

        for i in range(n_features):
            mi = self._compute_mi_1d(features[:, i], target)
            mi_scores.append(mi)

            # Normalize by entropy
            h_x = self._entropy_1d(features[:, i])
            h_y = self._entropy_1d(target)

            if h_x > 0 and h_y > 0:
                nmi = mi / np.sqrt(h_x * h_y)
            else:
                nmi = 0

            normalized_mi.append(nmi)

        # Overall target entropy
        target_entropy = self._entropy_1d(target)

        # Information explained ratio
        max_mi = max(mi_scores) if mi_scores else 0
        info_explained = max_mi / (target_entropy + 1e-10)

        return {
            "mutual_information": mi_scores,
            "normalized_mi": normalized_mi,
            "target_entropy": float(target_entropy),
            "max_mi": float(max_mi),
            "information_explained_ratio": float(info_explained),
            "most_informative_feature": int(np.argmax(mi_scores)) if mi_scores else 0,
            "predictability_score": float(np.clip(info_explained, 0, 1))
        }

    def _compute_mi_1d(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute MI between two 1D arrays."""
        # Discretize
        x_binned = self._discretize(x)
        y_binned = self._discretize(y)

        # Joint histogram
        joint_hist = np.zeros((self.n_bins, self.n_bins))
        for xi, yi in zip(x_binned, y_binned):
            joint_hist[xi, yi] += 1

        joint_hist = joint_hist / len(x)  # Normalize

        # Marginals
        p_x = np.sum(joint_hist, axis=1)
        p_y = np.sum(joint_hist, axis=0)

        # MI calculation
        mi = 0.0
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                if joint_hist[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (p_x[i] * p_y[j])
                    )

        return max(0.0, float(mi))

    def _entropy_1d(self, x: np.ndarray) -> float:
        """Compute entropy of 1D array."""
        binned = self._discretize(x)
        hist = np.bincount(binned, minlength=self.n_bins)
        p = hist / len(x)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        percentiles = np.percentile(x, np.linspace(0, 100, self.n_bins + 1))
        return np.clip(np.digitize(x, percentiles[1:-1]), 0, self.n_bins - 1)

    def _default_result(self) -> Dict[str, Any]:
        return {
            "mutual_information": [],
            "normalized_mi": [],
            "target_entropy": 0.0,
            "predictability_score": 0.0
        }


class RenyiEntropyAnalyzer:
    """
    Rényi entropy and α-divergences for robust information measures.

    H_α(X) = (1/(1-α)) log(Σ pᵢ^α)

    Special cases:
    - α → 1: Shannon entropy
    - α = 0: Hartley entropy (log of support size)
    - α = 2: Collision entropy
    - α → ∞: Min-entropy
    """

    def __init__(
        self,
        alphas: List[float] = [0.5, 1.0, 2.0, 5.0]
    ):
        self.alphas = alphas

    def compute(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Rényi entropy for various α values.
        """
        if len(returns) < 20:
            return self._default_result()

        # Estimate probability distribution
        hist, _ = np.histogram(returns, bins=50, density=True)
        p = hist / hist.sum()
        p = p[p > 0]  # Remove zeros

        results = {}
        for alpha in self.alphas:
            if abs(alpha - 1.0) < 1e-6:
                # Shannon entropy
                h = -np.sum(p * np.log(p))
            else:
                # Rényi entropy
                h = (1 / (1 - alpha)) * np.log(np.sum(p ** alpha))

            results[f"H_{alpha:.1f}"] = float(h)

        # Tsallis entropy (non-additive)
        q = 2.0
        tsallis = (1 - np.sum(p ** q)) / (q - 1)

        # Concentration measures
        concentration = np.sum(p ** 2)  # Gini-Simpson
        effective_states = 1 / concentration

        return {
            "renyi_entropies": results,
            "tsallis_entropy": float(tsallis),
            "concentration": float(concentration),
            "effective_states": float(effective_states),
            "uniformity_score": float(1 - concentration),  # Higher = more uniform
            "tail_weight": float(1 - results.get("H_2.0", 0) / results.get("H_0.5", 1))
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "renyi_entropies": {},
            "tsallis_entropy": 0.0,
            "concentration": 1.0,
            "effective_states": 1.0
        }


class InformationGeometryEngine:
    """
    Unified Information Geometry Engine combining all analyzers.
    """

    def __init__(self):
        self.fisher_analyzer = FisherInformationAnalyzer()
        self.kl_analyzer = KLDivergenceAnalyzer()
        self.mi_analyzer = MutualInformationAnalyzer()
        self.renyi_analyzer = RenyiEntropyAnalyzer()

    def analyze(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> InformationGeometryResult:
        """
        Perform comprehensive information geometry analysis.
        """
        if len(returns) < 20:
            return self._default_result()

        # Fisher Information
        fisher_result = self.fisher_analyzer.compute(returns)
        fisher_matrix = fisher_result.get("fisher_matrix", np.eye(2))

        # KL Divergence (regime detection)
        kl_result = self.kl_analyzer.compute(returns)
        kl_div = kl_result.get("kl_divergence_forward", 0.0)
        hellinger = kl_result.get("hellinger_distance", 0.0)

        # Mutual Information
        if features is not None:
            future_returns = np.roll(returns, -1)[:-1]
            features_aligned = features[:-1] if len(features) > len(future_returns) else features
            mi_result = self.mi_analyzer.compute(features_aligned[:len(future_returns)], future_returns)
            mi = mi_result.get("max_mi", 0.0)
        else:
            # Use lagged returns as features
            lagged = returns[:-1]
            target = returns[1:]
            mi_result = self.mi_analyzer.compute(lagged.reshape(-1, 1), target)
            mi = mi_result.get("max_mi", 0.0)

        # Rényi entropy
        renyi_result = self.renyi_analyzer.compute(returns)

        # Shannon entropy
        entropy = renyi_result.get("renyi_entropies", {}).get("H_1.0", 0.0)

        # Regime divergences
        regime_divergence = {
            "vs_baseline": kl_div,
            "vs_gaussian": self._kl_from_gaussian(returns),
            "vs_uniform": self._kl_from_uniform(returns)
        }

        # Natural gradient (for optimization)
        if isinstance(fisher_matrix, np.ndarray) and fisher_matrix.shape[0] == fisher_matrix.shape[1]:
            try:
                fisher_inv = np.linalg.inv(fisher_matrix + 1e-6 * np.eye(fisher_matrix.shape[0]))
                gradient = np.array([np.mean(returns), np.std(returns)])
                if len(gradient) <= fisher_inv.shape[0]:
                    gradient = np.pad(gradient, (0, fisher_inv.shape[0] - len(gradient)))
                natural_gradient = fisher_inv @ gradient[:fisher_inv.shape[0]]
            except Exception:
                natural_gradient = np.zeros(fisher_matrix.shape[0])
        else:
            natural_gradient = np.zeros(2)

        # Cramér-Rao bound
        cr_bound = fisher_result.get("cramer_rao_bound_mu", 1.0)

        # Information velocity (rate of change)
        if len(returns) >= 40:
            kl_20 = self._compute_kl_window(returns, 20, 10)
            kl_40 = self._compute_kl_window(returns, 40, 20)
            info_velocity = abs(kl_20 - kl_40)
        else:
            info_velocity = 0.0

        # Predictability score
        predictability = self._compute_predictability(
            mi, entropy, kl_div, fisher_result.get("estimation_confidence", 0.5)
        )

        return InformationGeometryResult(
            fisher_information=fisher_matrix,
            kl_divergence=kl_div,
            hellinger_distance=hellinger,
            mutual_information=mi,
            entropy=entropy,
            regime_divergence=regime_divergence,
            natural_gradient=natural_gradient,
            cramer_rao_bound=cr_bound,
            information_velocity=info_velocity,
            predictability_score=predictability
        )

    def _kl_from_gaussian(self, returns: np.ndarray) -> float:
        """KL divergence from fitted Gaussian."""
        mu = np.mean(returns)
        sigma = np.std(returns)

        # Empirical distribution
        hist, edges = np.histogram(returns, bins=50, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dx = edges[1] - edges[0]

        # Gaussian PDF
        gaussian_pdf = stats.norm.pdf(centers, mu, sigma)

        # KL divergence
        eps = 1e-10
        p = hist * dx + eps
        q = gaussian_pdf * dx + eps

        p = p / p.sum()
        q = q / q.sum()

        kl = np.sum(p * np.log(p / q))
        return float(max(0, kl))

    def _kl_from_uniform(self, returns: np.ndarray) -> float:
        """KL divergence from uniform distribution."""
        hist, _ = np.histogram(returns, bins=50, density=True)
        p = hist / hist.sum()

        uniform = np.ones_like(p) / len(p)

        eps = 1e-10
        p = p + eps
        uniform = uniform + eps

        p = p / p.sum()
        uniform = uniform / uniform.sum()

        kl = np.sum(p * np.log(p / uniform))
        return float(max(0, kl))

    def _compute_kl_window(
        self,
        returns: np.ndarray,
        window1: int,
        window2: int
    ) -> float:
        """Compute KL between two windows."""
        if len(returns) < window1:
            return 0.0

        r1 = returns[-window1:-window1+window2]
        r2 = returns[-window2:]

        if len(r1) < 5 or len(r2) < 5:
            return 0.0

        # Simple parametric KL
        mu1, s1 = np.mean(r1), np.std(r1)
        mu2, s2 = np.mean(r2), np.std(r2)

        if s1 > 0 and s2 > 0:
            kl = np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
            return float(max(0, kl))
        return 0.0

    def _compute_predictability(
        self,
        mi: float,
        entropy: float,
        kl: float,
        estimation_confidence: float
    ) -> float:
        """Compute overall predictability score."""
        # High MI and low entropy = more predictable
        if entropy > 0:
            mi_ratio = mi / entropy
        else:
            mi_ratio = 0

        # Regime stability (low KL = stable)
        stability = np.exp(-kl)

        # Combined score
        score = 0.4 * mi_ratio + 0.3 * stability + 0.3 * estimation_confidence

        return float(np.clip(score, 0, 1))

    def _default_result(self) -> InformationGeometryResult:
        return InformationGeometryResult(
            fisher_information=np.eye(2),
            kl_divergence=0.0,
            hellinger_distance=0.0,
            mutual_information=0.0,
            entropy=0.0,
            regime_divergence={},
            natural_gradient=np.zeros(2),
            cramer_rao_bound=1.0,
            information_velocity=0.0,
            predictability_score=0.0
        )
