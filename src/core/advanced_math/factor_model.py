"""
Covariance Factor Model for Risk Decomposition

Mathematical Foundation:
========================
Factor Model:
   rᵢₜ = αᵢ + Σⱼ βᵢⱼ Fⱼₜ + εᵢₜ

Where:
- rᵢₜ = return of stock i at time t
- αᵢ = stock-specific alpha
- βᵢⱼ = factor loading (sensitivity to factor j)
- Fⱼₜ = factor j return
- εᵢₜ = idiosyncratic return (stock-specific)

Covariance Decomposition:
   Σ = BΣ_F Bᵀ + Σ_ε

Where:
- B = factor loadings matrix (n_stocks x n_factors)
- Σ_F = factor covariance matrix (n_factors x n_factors)
- Σ_ε = diagonal idiosyncratic variance matrix

Risk Attribution:
   Systematic Risk = βᵢ Σ_F βᵢᵀ
   Idiosyncratic Risk = σ²ᵢ

Key Insight: Focus on stocks with high idiosyncratic risk
(less correlated with market → alpha opportunity)

References:
- Ross, S.A. (1976). The Arbitrage Theory of Capital Asset Pricing
- Fama, E.F. & French, K.R. (1993). Common Risk Factors in the Returns on Stocks and Bonds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class FactorType(Enum):
    """Types of risk factors."""
    MARKET = "market"           # Overall market factor
    SIZE = "size"               # Small vs large cap (SMB)
    VALUE = "value"             # Value vs growth (HML)
    MOMENTUM = "momentum"       # Winners vs losers (WML)
    VOLATILITY = "volatility"   # Low vs high vol
    QUALITY = "quality"         # High vs low quality
    SECTOR = "sector"           # Industry/sector
    STATISTICAL = "statistical" # PCA-derived


@dataclass
class FactorExposure:
    """Exposure of a stock to various factors."""
    stock: str
    factor_loadings: Dict[str, float]  # Factor name -> beta
    r_squared: float  # How much variance explained by factors
    alpha: float  # Unexplained return
    idiosyncratic_vol: float  # Stock-specific volatility
    systematic_risk_pct: float  # % of risk from factors


@dataclass
class FactorModelResult:
    """Results from factor model estimation."""
    # Factor returns
    factor_returns: pd.DataFrame  # (n_periods x n_factors)
    factor_covariance: np.ndarray  # (n_factors x n_factors)

    # Loadings
    loadings_matrix: pd.DataFrame  # (n_stocks x n_factors)

    # Risk decomposition per stock
    exposures: Dict[str, FactorExposure]

    # Portfolio-level metrics
    factor_contributions: Dict[str, float]  # Which factors drive portfolio risk
    total_systematic_risk: float
    total_idiosyncratic_risk: float

    # Model quality
    average_r_squared: float
    n_factors: int


@dataclass
class CovarianceDecomposition:
    """Decomposed covariance matrix."""
    # Original covariance
    total_covariance: np.ndarray

    # Decomposed components
    systematic_covariance: np.ndarray  # From factors
    idiosyncratic_covariance: np.ndarray  # Stock-specific (diagonal)

    # Eigenvalue analysis
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    effective_factors: int  # Number of significant factors

    # Quality metrics
    reconstruction_error: float
    explained_variance_ratio: float


class FactorModel:
    """
    Multi-factor model for stock return decomposition.

    Decomposes returns into:
    - Systematic (factor-driven) component
    - Idiosyncratic (stock-specific) component

    Useful for:
    - Risk attribution
    - Portfolio construction
    - Alpha signal validation
    """

    def __init__(
        self,
        n_statistical_factors: int = 5,
        use_fundamental_factors: bool = True,
        min_observations: int = 60
    ):
        self.n_statistical_factors = n_statistical_factors
        self.use_fundamental_factors = use_fundamental_factors
        self.min_observations = min_observations

        # Estimated parameters
        self.factor_returns_ = None
        self.loadings_ = None
        self.idiosyncratic_var_ = None

    def fit(self, returns: pd.DataFrame, market_returns: Optional[pd.Series] = None) -> FactorModelResult:
        """
        Fit factor model to return data.

        Args:
            returns: DataFrame of stock returns (n_periods x n_stocks)
            market_returns: Optional market benchmark returns

        Returns:
            FactorModelResult with estimated factors and loadings
        """
        n_periods, n_stocks = returns.shape

        if n_periods < self.min_observations:
            logger.warning(f"Insufficient observations ({n_periods}). Need at least {self.min_observations}.")
            return self._empty_result(returns)

        # Step 1: Extract statistical factors using PCA
        statistical_factors = self._extract_statistical_factors(returns)

        # Step 2: Create fundamental factors if requested
        if self.use_fundamental_factors and market_returns is not None:
            fundamental_factors = self._create_fundamental_factors(returns, market_returns)
            factor_returns = pd.concat([fundamental_factors, statistical_factors], axis=1)
        else:
            factor_returns = statistical_factors

        # Step 3: Estimate factor loadings for each stock
        loadings, residuals, alphas = self._estimate_loadings(returns, factor_returns)

        # Step 4: Compute idiosyncratic variances
        idiosyncratic_var = residuals.var()

        # Step 5: Compute factor covariance
        factor_cov = factor_returns.cov().values

        # Store for later use
        self.factor_returns_ = factor_returns
        self.loadings_ = loadings
        self.idiosyncratic_var_ = idiosyncratic_var

        # Step 6: Compute exposures for each stock
        exposures = {}
        for stock in returns.columns:
            stock_loadings = loadings.loc[stock].to_dict()
            stock_var = returns[stock].var()

            # Systematic variance
            B = loadings.loc[stock].values
            systematic_var = B @ factor_cov @ B.T
            idio_var = idiosyncratic_var[stock]

            # R-squared
            if stock_var > 0:
                r_squared = systematic_var / stock_var
            else:
                r_squared = 0

            exposures[stock] = FactorExposure(
                stock=stock,
                factor_loadings=stock_loadings,
                r_squared=min(1.0, r_squared),
                alpha=alphas.get(stock, 0),
                idiosyncratic_vol=np.sqrt(idio_var),
                systematic_risk_pct=min(1.0, r_squared)
            )

        # Step 7: Compute factor contributions to total risk
        total_var = returns.var().sum()
        factor_contributions = {}

        for i, factor in enumerate(factor_returns.columns):
            factor_var = factor_cov[i, i]
            factor_loading_sum = loadings[factor].abs().sum()
            factor_contribution = factor_var * factor_loading_sum ** 2 / n_stocks

            if total_var > 0:
                factor_contributions[factor] = factor_contribution / total_var
            else:
                factor_contributions[factor] = 0

        # Systematic vs idiosyncratic
        total_systematic = sum(
            exp.systematic_risk_pct * returns[stock].var()
            for stock, exp in exposures.items()
        )
        total_idio = sum(
            (1 - exp.systematic_risk_pct) * returns[stock].var()
            for stock, exp in exposures.items()
        )

        avg_r_squared = np.mean([exp.r_squared for exp in exposures.values()])

        return FactorModelResult(
            factor_returns=factor_returns,
            factor_covariance=factor_cov,
            loadings_matrix=loadings,
            exposures=exposures,
            factor_contributions=factor_contributions,
            total_systematic_risk=total_systematic,
            total_idiosyncratic_risk=total_idio,
            average_r_squared=avg_r_squared,
            n_factors=len(factor_returns.columns)
        )

    def _extract_statistical_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical factors using PCA.
        """
        # Standardize returns
        returns_std = (returns - returns.mean()) / (returns.std() + 1e-10)
        returns_std = returns_std.fillna(0)

        # Covariance matrix
        cov_matrix = returns_std.T @ returns_std / len(returns_std)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top factors
        n_factors = min(self.n_statistical_factors, len(eigenvalues))
        top_eigenvectors = eigenvectors[:, :n_factors]

        # Factor returns = standardized returns projected onto eigenvectors
        factor_returns = returns_std.values @ top_eigenvectors

        # Create DataFrame
        factor_names = [f'PC{i+1}' for i in range(n_factors)]
        factor_df = pd.DataFrame(
            factor_returns,
            index=returns.index,
            columns=factor_names
        )

        return factor_df

    def _create_fundamental_factors(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Create fundamental Fama-French style factors.
        """
        factors = pd.DataFrame(index=returns.index)

        # Market factor
        factors['MKT'] = market_returns.reindex(returns.index).fillna(0)

        # Size factor (SMB): Small minus Big
        # Proxy: Difference between equal-weighted and market-cap weighted returns
        # (Simplified since we don't have market cap)
        equal_weight = returns.mean(axis=1)
        factors['SMB'] = equal_weight - factors['MKT']

        # Momentum factor (WML): Winners minus Losers
        # Sort stocks by past 12-month return, long top quartile, short bottom
        momentum_returns = returns.rolling(252).mean()
        top_quartile = momentum_returns.apply(lambda x: x >= x.quantile(0.75), axis=1)
        bottom_quartile = momentum_returns.apply(lambda x: x <= x.quantile(0.25), axis=1)

        winners = returns.where(top_quartile).mean(axis=1).fillna(0)
        losers = returns.where(bottom_quartile).mean(axis=1).fillna(0)
        factors['MOM'] = winners - losers

        return factors

    def _estimate_loadings(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Estimate factor loadings using OLS regression.

        For each stock i:
        r_i = α_i + Σ β_ij F_j + ε_i
        """
        loadings = {}
        residuals = {}
        alphas = {}

        X = factors.values
        X_with_const = np.column_stack([np.ones(len(X)), X])

        for stock in returns.columns:
            y = returns[stock].values

            # OLS: β = (X'X)^(-1) X'y
            try:
                XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
                betas = XtX_inv @ X_with_const.T @ y

                alphas[stock] = betas[0]
                loadings[stock] = betas[1:]

                # Residuals
                y_pred = X_with_const @ betas
                residuals[stock] = y - y_pred

            except np.linalg.LinAlgError:
                alphas[stock] = 0
                loadings[stock] = np.zeros(len(factors.columns))
                residuals[stock] = y

        loadings_df = pd.DataFrame(loadings, index=factors.columns).T
        residuals_df = pd.DataFrame(residuals, index=returns.index)

        return loadings_df, residuals_df, alphas

    def _empty_result(self, returns: pd.DataFrame) -> FactorModelResult:
        """Return empty result for insufficient data."""
        return FactorModelResult(
            factor_returns=pd.DataFrame(),
            factor_covariance=np.array([[]]),
            loadings_matrix=pd.DataFrame(),
            exposures={},
            factor_contributions={},
            total_systematic_risk=0,
            total_idiosyncratic_risk=0,
            average_r_squared=0,
            n_factors=0
        )

    def get_idiosyncratic_opportunity(self, stock: str) -> Optional[float]:
        """
        Get idiosyncratic opportunity score for a stock.

        High idiosyncratic risk = less correlated with market
        = potential alpha opportunity

        Returns:
            Score 0-1 (higher = more opportunity)
        """
        if self.idiosyncratic_var_ is None or stock not in self.idiosyncratic_var_:
            return None

        idio_var = self.idiosyncratic_var_[stock]
        total_var = idio_var  # Simplified

        # Normalize against all stocks
        all_idio = self.idiosyncratic_var_.values
        if np.std(all_idio) > 0:
            z_score = (idio_var - np.mean(all_idio)) / np.std(all_idio)
            # Convert to 0-1 score using sigmoid
            opportunity = 1 / (1 + np.exp(-z_score))
        else:
            opportunity = 0.5

        return opportunity


class CovarianceDecomposer:
    """
    Decompose covariance matrix into systematic and idiosyncratic components.

    Used for:
    - Portfolio risk analysis
    - Understanding correlation structure
    - Detecting regime changes
    """

    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors

    def decompose(self, returns: pd.DataFrame) -> CovarianceDecomposition:
        """
        Decompose covariance matrix using PCA.

        Σ = BΣ_F Bᵀ + Σ_ε
        """
        # Total covariance
        total_cov = returns.cov().values

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(total_cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top factors
        n_factors = min(self.n_factors, len(eigenvalues))

        # Factor loadings (top eigenvectors scaled by sqrt of eigenvalues)
        B = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

        # Factor covariance (identity since factors are orthogonal)
        factor_cov = np.eye(n_factors)

        # Systematic covariance
        systematic_cov = B @ factor_cov @ B.T

        # Idiosyncratic covariance (diagonal: what's left)
        idio_cov = total_cov - systematic_cov
        # Make it diagonal (off-diagonal should be zero in theory)
        idio_cov = np.diag(np.diag(np.maximum(idio_cov, 0)))

        # Reconstruction error
        reconstructed = systematic_cov + idio_cov
        recon_error = np.mean((total_cov - reconstructed) ** 2)

        # Explained variance
        total_var = np.sum(eigenvalues)
        explained_var = np.sum(eigenvalues[:n_factors])
        explained_ratio = explained_var / total_var if total_var > 0 else 0

        # Effective number of factors (using Marchenko-Pastur bound)
        n_obs, n_assets = returns.shape
        q = n_obs / n_assets if n_assets > 0 else 1
        mp_upper = (1 + 1/np.sqrt(q)) ** 2 if q > 0 else 2
        effective_factors = int(np.sum(eigenvalues > mp_upper * np.mean(eigenvalues)))

        return CovarianceDecomposition(
            total_covariance=total_cov,
            systematic_covariance=systematic_cov,
            idiosyncratic_covariance=idio_cov,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            effective_factors=max(1, effective_factors),
            reconstruction_error=recon_error,
            explained_variance_ratio=explained_ratio
        )

    def get_correlation_risk(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute correlation risk metrics.

        Returns metrics useful for portfolio risk assessment.
        """
        decomp = self.decompose(returns)

        # Average correlation
        corr_matrix = returns.corr().values
        np.fill_diagonal(corr_matrix, np.nan)
        avg_correlation = np.nanmean(corr_matrix)

        # Eigenvalue concentration (higher = more systematic risk)
        total_eigenvalue = np.sum(decomp.eigenvalues)
        first_eigenvalue_pct = decomp.eigenvalues[0] / total_eigenvalue if total_eigenvalue > 0 else 0

        # Diversification ratio
        # DR = sum(individual vols) / portfolio vol
        individual_vols = np.sqrt(np.diag(decomp.total_covariance))
        portfolio_var = np.sum(decomp.total_covariance) / len(individual_vols) ** 2
        if portfolio_var > 0:
            diversification_ratio = np.sum(individual_vols) / (np.sqrt(portfolio_var) * len(individual_vols))
        else:
            diversification_ratio = 1.0

        return {
            'average_correlation': avg_correlation,
            'first_factor_risk_pct': first_eigenvalue_pct,
            'effective_factors': decomp.effective_factors,
            'diversification_ratio': diversification_ratio,
            'explained_variance': decomp.explained_variance_ratio,
            'systematic_risk_pct': decomp.explained_variance_ratio,
            'idiosyncratic_risk_pct': 1 - decomp.explained_variance_ratio
        }


def compute_factor_adjusted_probability(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    base_probability: float
) -> Tuple[float, float, float]:
    """
    Adjust prediction probability based on factor exposure.

    Returns:
        (adjusted_probability, beta, idiosyncratic_ratio)
    """
    if len(stock_returns) < 30 or len(market_returns) < 30:
        return base_probability, 1.0, 0.5

    # Align data
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()

    if len(aligned) < 30:
        return base_probability, 1.0, 0.5

    # Compute beta
    cov_matrix = aligned.cov()
    market_var = cov_matrix.loc['market', 'market']
    covariance = cov_matrix.loc['stock', 'market']

    if market_var > 0:
        beta = covariance / market_var
    else:
        beta = 1.0

    # Compute idiosyncratic ratio
    stock_var = aligned['stock'].var()
    systematic_var = beta ** 2 * market_var
    idiosyncratic_var = max(0, stock_var - systematic_var)
    idiosyncratic_ratio = idiosyncratic_var / stock_var if stock_var > 0 else 0.5

    # Adjust probability
    # High idiosyncratic = signal more reliable (not just following market)
    # Low idiosyncratic = signal may just be market exposure
    adjustment = 0.1 * (idiosyncratic_ratio - 0.5)  # Range: -0.05 to +0.05
    adjusted_prob = base_probability + adjustment

    # Clamp to valid range
    adjusted_prob = max(0.3, min(0.7, adjusted_prob))

    return adjusted_prob, beta, idiosyncratic_ratio
