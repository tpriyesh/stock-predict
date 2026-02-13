"""
Statistical Significance Testing Module

This module provides rigorous statistical tests to validate that trading signals
and backtest results are statistically significant and not due to random chance.

Key Tests:
1. Sharpe Ratio Significance - Is the Sharpe ratio statistically different from 0?
2. Permutation Tests - Are signals better than random shuffled signals?
3. Bootstrap Confidence Intervals - True uncertainty bounds for metrics
4. Multiple Testing Correction - Control false discovery rate
5. Autocorrelation Tests - Validate independence assumptions

References:
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing
- Politis, D. N., & Romano, J. P. (1994). Stationary Bootstrap
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import warnings


class SignificanceLevel(Enum):
    """Standard significance levels for hypothesis testing."""
    HIGHLY_SIGNIFICANT = 0.01  # 99% confidence
    SIGNIFICANT = 0.05         # 95% confidence
    MARGINALLY_SIGNIFICANT = 0.10  # 90% confidence
    NOT_SIGNIFICANT = 1.0


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    significance_level: SignificanceLevel
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'significance_level': self.significance_level.name,
            'confidence_interval': self.confidence_interval,
            'effect_size': self.effect_size,
            'interpretation': self.interpretation,
            'warnings': self.warnings
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for a trading strategy."""
    sharpe_test: Optional[StatisticalTestResult] = None
    returns_test: Optional[StatisticalTestResult] = None
    permutation_test: Optional[StatisticalTestResult] = None
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None
    autocorrelation_test: Optional[StatisticalTestResult] = None
    multiple_testing_adjustment: Optional[Dict[str, float]] = None
    overall_validity: bool = False
    validity_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sharpe_test': self.sharpe_test.to_dict() if self.sharpe_test else None,
            'returns_test': self.returns_test.to_dict() if self.returns_test else None,
            'permutation_test': self.permutation_test.to_dict() if self.permutation_test else None,
            'bootstrap_ci': self.bootstrap_ci,
            'autocorrelation_test': self.autocorrelation_test.to_dict() if self.autocorrelation_test else None,
            'multiple_testing_adjustment': self.multiple_testing_adjustment,
            'overall_validity': self.overall_validity,
            'validity_score': self.validity_score,
            'recommendations': self.recommendations
        }


class StatisticalValidator:
    """
    Comprehensive statistical validation for trading strategies.

    This class provides rigorous statistical tests to ensure that trading
    signals and backtest results are statistically significant.

    Usage:
        validator = StatisticalValidator()
        report = validator.validate_strategy(returns, signals, benchmark_returns)
    """

    # Constants with citations
    TRADING_DAYS_PER_YEAR = 252  # Standard for US/India equity markets
    RISK_FREE_RATE = 0.05  # 5% - approximate India 10Y bond yield 2024

    def __init__(
        self,
        significance_level: float = 0.05,
        n_permutations: int = 10000,
        n_bootstrap: int = 10000,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the statistical validator.

        Args:
            significance_level: Alpha level for hypothesis tests (default 0.05 = 95% confidence)
            n_permutations: Number of permutations for permutation tests
            n_bootstrap: Number of bootstrap samples for confidence intervals
            random_seed: Random seed for reproducibility
        """
        self.significance_level = significance_level
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)

    def validate_strategy(
        self,
        returns: np.ndarray,
        signals: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> ValidationReport:
        """
        Perform comprehensive statistical validation of a trading strategy.

        Args:
            returns: Array of strategy returns (daily)
            signals: Optional array of trading signals (+1 buy, -1 sell, 0 hold)
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            ValidationReport with all test results
        """
        report = ValidationReport()
        validity_checks = []

        # Clean data
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 30:
            report.recommendations.append(
                f"Insufficient data: {len(returns)} observations. Need at least 30 for reliable tests."
            )
            return report

        # 1. Test if mean return is significantly different from zero
        report.returns_test = self.test_mean_return(returns)
        validity_checks.append(report.returns_test.is_significant)

        # 2. Test if Sharpe ratio is significantly different from zero
        report.sharpe_test = self.test_sharpe_ratio(returns)
        validity_checks.append(report.sharpe_test.is_significant)

        # 3. Permutation test for signal validity (if signals provided)
        if signals is not None:
            signals = np.array(signals)
            if len(signals) == len(returns):
                report.permutation_test = self.permutation_test_signals(returns, signals)
                validity_checks.append(report.permutation_test.is_significant)

        # 4. Bootstrap confidence intervals
        report.bootstrap_ci = self.bootstrap_confidence_intervals(returns)

        # 5. Autocorrelation test (returns should be approximately independent)
        report.autocorrelation_test = self.test_autocorrelation(returns)
        # High autocorrelation is a WARNING, not a validity failure
        if report.autocorrelation_test.is_significant:
            report.recommendations.append(
                "Returns show significant autocorrelation. Sharpe ratio may be inflated."
            )

        # Calculate overall validity
        if validity_checks:
            report.validity_score = sum(validity_checks) / len(validity_checks)
            report.overall_validity = report.validity_score >= 0.5

        # Generate recommendations
        self._generate_recommendations(report)

        return report

    def test_mean_return(self, returns: np.ndarray) -> StatisticalTestResult:
        """
        Test if the mean return is significantly different from zero.

        Uses a one-sample t-test with the null hypothesis that mean = 0.

        Reference: Student's t-test (Gosset, 1908)
        """
        n = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Standard error of the mean
        se = std_return / np.sqrt(n)

        # t-statistic
        if se > 0:
            t_stat = mean_return / se
        else:
            t_stat = 0.0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

        # Confidence interval for mean
        t_critical = stats.t.ppf(1 - self.significance_level/2, df=n-1)
        ci_lower = mean_return - t_critical * se
        ci_upper = mean_return + t_critical * se

        # Effect size (Cohen's d)
        effect_size = mean_return / std_return if std_return > 0 else 0.0

        is_significant = p_value < self.significance_level
        sig_level = self._get_significance_level(p_value)

        # Annualized metrics for interpretation
        ann_return = mean_return * self.TRADING_DAYS_PER_YEAR

        interpretation = (
            f"Mean daily return: {mean_return:.4%} (annualized: {ann_return:.2%}). "
            f"{'Statistically significant' if is_significant else 'Not statistically significant'} "
            f"at {self.significance_level:.0%} level (p={p_value:.4f})."
        )

        return StatisticalTestResult(
            test_name="Mean Return T-Test",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=sig_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            interpretation=interpretation
        )

    def test_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Test if the Sharpe ratio is significantly different from zero.

        Uses the methodology from:
        - Lo, A. W. (2002). "The Statistics of Sharpe Ratios"
        - Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"

        The standard error of the Sharpe ratio is approximately:
        SE(SR) = sqrt((1 + 0.5*SR^2) / n) for normally distributed returns

        For non-normal returns, we use the more accurate formula:
        SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / n)
        """
        if risk_free_rate is None:
            risk_free_rate = self.RISK_FREE_RATE

        n = len(returns)

        # Daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1/self.TRADING_DAYS_PER_YEAR) - 1

        # Excess returns
        excess_returns = returns - rf_daily

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess <= 0:
            return StatisticalTestResult(
                test_name="Sharpe Ratio Test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                significance_level=SignificanceLevel.NOT_SIGNIFICANT,
                interpretation="Cannot compute Sharpe ratio: zero volatility"
            )

        # Daily Sharpe ratio
        sharpe_daily = mean_excess / std_excess

        # Annualized Sharpe ratio
        sharpe_annual = sharpe_daily * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Calculate moments for accurate standard error
        skewness = stats.skew(excess_returns)
        kurtosis = stats.kurtosis(excess_returns, fisher=True)  # Excess kurtosis

        # Standard error of Sharpe ratio (Lo, 2002; adjusted for non-normality)
        # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt)/4*SR^2) / n)
        se_sharpe = np.sqrt(
            (1 + 0.5 * sharpe_daily**2
             - skewness * sharpe_daily
             + (kurtosis / 4) * sharpe_daily**2) / n
        )

        # Annualized standard error
        se_sharpe_annual = se_sharpe * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Z-statistic (approximately normal for large n)
        if se_sharpe_annual > 0:
            z_stat = sharpe_annual / se_sharpe_annual
        else:
            z_stat = 0.0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.significance_level/2)
        ci_lower = sharpe_annual - z_critical * se_sharpe_annual
        ci_upper = sharpe_annual + z_critical * se_sharpe_annual

        is_significant = p_value < self.significance_level
        sig_level = self._get_significance_level(p_value)

        warnings_list = []
        if abs(skewness) > 1:
            warnings_list.append(f"High skewness ({skewness:.2f}) may affect test reliability")
        if kurtosis > 3:
            warnings_list.append(f"Fat tails (excess kurtosis={kurtosis:.2f}) detected")

        interpretation = (
            f"Annualized Sharpe Ratio: {sharpe_annual:.3f} "
            f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]). "
            f"{'Statistically significant' if is_significant else 'Not statistically significant'} "
            f"at {self.significance_level:.0%} level (p={p_value:.4f})."
        )

        return StatisticalTestResult(
            test_name="Sharpe Ratio Significance Test (Lo, 2002)",
            statistic=z_stat,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=sig_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=sharpe_annual,
            interpretation=interpretation,
            warnings=warnings_list
        )

    def permutation_test_signals(
        self,
        returns: np.ndarray,
        signals: np.ndarray
    ) -> StatisticalTestResult:
        """
        Permutation test to check if signals have predictive power.

        Null hypothesis: Signals have no relationship with returns
        (i.e., shuffling signals produces equivalent or better results)

        Reference: Good, P. (2005). Permutation, Parametric and Bootstrap Tests
        """
        n = len(returns)

        # Calculate actual strategy return (signal * return)
        # Signals: +1 = buy (expect positive return), -1 = sell (expect negative return)
        strategy_returns = signals * returns
        actual_total_return = np.sum(strategy_returns)

        # Permutation distribution
        permuted_returns = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            # Shuffle signals
            shuffled_signals = self.rng.permutation(signals)
            permuted_strategy = shuffled_signals * returns
            permuted_returns[i] = np.sum(permuted_strategy)

        # P-value: proportion of permutations >= actual
        # For two-tailed test, we look at both tails
        p_value = np.mean(np.abs(permuted_returns) >= np.abs(actual_total_return))

        # Effect size: how many standard deviations from permutation mean
        perm_mean = np.mean(permuted_returns)
        perm_std = np.std(permuted_returns)

        if perm_std > 0:
            effect_size = (actual_total_return - perm_mean) / perm_std
        else:
            effect_size = 0.0

        is_significant = p_value < self.significance_level
        sig_level = self._get_significance_level(p_value)

        # Confidence interval from permutation distribution
        ci_lower = np.percentile(permuted_returns, 2.5)
        ci_upper = np.percentile(permuted_returns, 97.5)

        interpretation = (
            f"Actual cumulative return: {actual_total_return:.4f}. "
            f"Permutation distribution: mean={perm_mean:.4f}, std={perm_std:.4f}. "
            f"{'Signals show predictive power' if is_significant else 'Signals are not better than random'} "
            f"(p={p_value:.4f}, {self.n_permutations:,} permutations)."
        )

        return StatisticalTestResult(
            test_name="Permutation Test for Signal Validity",
            statistic=effect_size,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=sig_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            interpretation=interpretation
        )

    def bootstrap_confidence_intervals(
        self,
        returns: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for key metrics.

        Uses block bootstrap to preserve time-series structure.

        Reference: Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap"

        Args:
            returns: Array of returns
            metrics: List of metrics to calculate CIs for
                    Default: ['mean', 'sharpe', 'max_drawdown', 'volatility']

        Returns:
            Dictionary of metric -> (lower_ci, upper_ci)
        """
        if metrics is None:
            metrics = ['mean', 'sharpe', 'max_drawdown', 'volatility']

        n = len(returns)

        # Block size for block bootstrap (sqrt(n) is common heuristic)
        block_size = max(5, int(np.sqrt(n)))

        # Bootstrap samples
        bootstrap_stats = {metric: [] for metric in metrics}

        for _ in range(self.n_bootstrap):
            # Block bootstrap: sample blocks with replacement
            n_blocks = int(np.ceil(n / block_size))
            sampled_indices = []

            for _ in range(n_blocks):
                # Random starting point for block
                start = self.rng.randint(0, n - block_size + 1)
                sampled_indices.extend(range(start, start + block_size))

            # Trim to original length
            sampled_indices = sampled_indices[:n]
            bootstrap_returns = returns[sampled_indices]

            # Calculate metrics
            if 'mean' in metrics:
                bootstrap_stats['mean'].append(np.mean(bootstrap_returns))

            if 'sharpe' in metrics:
                mean_ret = np.mean(bootstrap_returns)
                std_ret = np.std(bootstrap_returns, ddof=1)
                if std_ret > 0:
                    sharpe = (mean_ret * self.TRADING_DAYS_PER_YEAR) / (std_ret * np.sqrt(self.TRADING_DAYS_PER_YEAR))
                else:
                    sharpe = 0.0
                bootstrap_stats['sharpe'].append(sharpe)

            if 'max_drawdown' in metrics:
                cumulative = np.cumprod(1 + bootstrap_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                bootstrap_stats['max_drawdown'].append(np.min(drawdowns))

            if 'volatility' in metrics:
                vol = np.std(bootstrap_returns, ddof=1) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                bootstrap_stats['volatility'].append(vol)

        # Calculate confidence intervals (percentile method)
        confidence_intervals = {}
        alpha = self.significance_level

        for metric in metrics:
            values = bootstrap_stats[metric]
            ci_lower = np.percentile(values, 100 * alpha / 2)
            ci_upper = np.percentile(values, 100 * (1 - alpha / 2))
            confidence_intervals[metric] = (ci_lower, ci_upper)

        return confidence_intervals

    def test_autocorrelation(
        self,
        returns: np.ndarray,
        max_lag: int = 10
    ) -> StatisticalTestResult:
        """
        Test for significant autocorrelation in returns using Ljung-Box test.

        High autocorrelation suggests:
        1. Sharpe ratios may be inflated
        2. Returns violate i.i.d. assumption
        3. Strategy may have data snooping issues

        Reference: Ljung, G. M., & Box, G. E. P. (1978)
        """
        n = len(returns)

        if n < max_lag + 10:
            max_lag = max(1, n // 3)

        # Calculate autocorrelations
        mean_ret = np.mean(returns)
        demeaned = returns - mean_ret
        var_ret = np.var(returns)

        if var_ret <= 0:
            return StatisticalTestResult(
                test_name="Ljung-Box Autocorrelation Test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                significance_level=SignificanceLevel.NOT_SIGNIFICANT,
                interpretation="Cannot compute autocorrelation: zero variance"
            )

        autocorrs = []
        for lag in range(1, max_lag + 1):
            if lag < n:
                autocorr = np.sum(demeaned[:-lag] * demeaned[lag:]) / (n * var_ret)
                autocorrs.append(autocorr)

        # Ljung-Box Q statistic
        # Q = n(n+2) * sum(r_k^2 / (n-k)) for k=1 to max_lag
        q_stat = 0.0
        for k, r_k in enumerate(autocorrs, start=1):
            q_stat += (r_k ** 2) / (n - k)
        q_stat *= n * (n + 2)

        # P-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(q_stat, df=max_lag)

        is_significant = p_value < self.significance_level
        sig_level = self._get_significance_level(p_value)

        # Report first few significant autocorrelations
        significant_lags = []
        for lag, r_k in enumerate(autocorrs, start=1):
            # Approximate 95% CI for autocorrelation under null: ±1.96/sqrt(n)
            threshold = 1.96 / np.sqrt(n)
            if abs(r_k) > threshold:
                significant_lags.append((lag, r_k))

        warnings_list = []
        if is_significant:
            warnings_list.append(
                f"Significant autocorrelation detected. Sharpe ratio confidence intervals may be too narrow."
            )
        if significant_lags:
            lag_str = ", ".join([f"lag {l}: {r:.3f}" for l, r in significant_lags[:3]])
            warnings_list.append(f"Significant autocorrelations: {lag_str}")

        interpretation = (
            f"Ljung-Box Q({max_lag}) = {q_stat:.2f}, p-value = {p_value:.4f}. "
            f"{'Significant autocorrelation present' if is_significant else 'No significant autocorrelation'} "
            f"at {self.significance_level:.0%} level."
        )

        return StatisticalTestResult(
            test_name="Ljung-Box Autocorrelation Test",
            statistic=q_stat,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=sig_level,
            effect_size=autocorrs[0] if autocorrs else 0.0,  # First-order autocorr
            interpretation=interpretation,
            warnings=warnings_list
        )

    def multiple_testing_correction(
        self,
        p_values: Dict[str, float],
        method: str = 'fdr_bh'
    ) -> Dict[str, float]:
        """
        Apply multiple testing correction to control false discovery rate.

        Methods:
        - 'bonferroni': Conservative, controls family-wise error rate
        - 'fdr_bh': Benjamini-Hochberg, controls false discovery rate (recommended)

        Reference: Benjamini, Y., & Hochberg, Y. (1995)

        Args:
            p_values: Dictionary of test_name -> p_value
            method: Correction method ('bonferroni' or 'fdr_bh')

        Returns:
            Dictionary of test_name -> adjusted_p_value
        """
        names = list(p_values.keys())
        pvals = np.array([p_values[name] for name in names])
        n_tests = len(pvals)

        if method == 'bonferroni':
            # Bonferroni: multiply all p-values by number of tests
            adjusted = np.minimum(pvals * n_tests, 1.0)

        elif method == 'fdr_bh':
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(pvals)
            sorted_pvals = pvals[sorted_indices]

            # Calculate adjusted p-values
            adjusted_sorted = np.zeros(n_tests)
            for i in range(n_tests - 1, -1, -1):
                if i == n_tests - 1:
                    adjusted_sorted[i] = sorted_pvals[i]
                else:
                    adjusted_sorted[i] = min(
                        adjusted_sorted[i + 1],
                        sorted_pvals[i] * n_tests / (i + 1)
                    )

            # Reorder to original order
            adjusted = np.zeros(n_tests)
            adjusted[sorted_indices] = adjusted_sorted
            adjusted = np.minimum(adjusted, 1.0)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'fdr_bh'")

        return {name: adj_p for name, adj_p in zip(names, adjusted)}

    def _get_significance_level(self, p_value: float) -> SignificanceLevel:
        """Categorize p-value into significance level."""
        if p_value < 0.01:
            return SignificanceLevel.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            return SignificanceLevel.SIGNIFICANT
        elif p_value < 0.10:
            return SignificanceLevel.MARGINALLY_SIGNIFICANT
        else:
            return SignificanceLevel.NOT_SIGNIFICANT

    def _generate_recommendations(self, report: ValidationReport) -> None:
        """Generate actionable recommendations based on test results."""

        if report.sharpe_test and not report.sharpe_test.is_significant:
            report.recommendations.append(
                "Sharpe ratio is not statistically significant. "
                "Consider: (1) collecting more data, (2) the strategy may not have true edge."
            )

        if report.returns_test and not report.returns_test.is_significant:
            report.recommendations.append(
                "Mean return is not statistically different from zero. "
                "Results could be due to random chance."
            )

        if report.permutation_test and not report.permutation_test.is_significant:
            report.recommendations.append(
                "Signals do not show statistically significant predictive power. "
                "Random signals perform similarly."
            )

        if report.bootstrap_ci:
            sharpe_ci = report.bootstrap_ci.get('sharpe')
            if sharpe_ci and sharpe_ci[0] < 0:
                report.recommendations.append(
                    f"Sharpe ratio 95% CI includes negative values [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]. "
                    "True Sharpe could be negative."
                )

        if not report.recommendations:
            report.recommendations.append(
                "Strategy passes all statistical significance tests. "
                "Results appear statistically valid."
            )


# Convenience function for quick validation
def validate_backtest_results(
    returns: np.ndarray,
    signals: Optional[np.ndarray] = None,
    significance_level: float = 0.05
) -> ValidationReport:
    """
    Quick validation of backtest results.

    Args:
        returns: Array of strategy returns
        signals: Optional array of trading signals
        significance_level: Alpha level for tests (default 0.05)

    Returns:
        ValidationReport with all test results
    """
    validator = StatisticalValidator(significance_level=significance_level)
    return validator.validate_strategy(returns, signals)


# Export main classes
__all__ = [
    'StatisticalValidator',
    'StatisticalTestResult',
    'ValidationReport',
    'SignificanceLevel',
    'validate_backtest_results'
]
