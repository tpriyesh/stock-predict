"""
PCA & Eigenvalue Decomposition for Feature Reduction

Mathematical Foundation:
========================
1. Compute Covariance Matrix: Σ = (1/n) Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
2. Eigenvalue Decomposition: Σv = λv
3. Feature Projection: Z = XW where W = [v₁, v₂, ..., vₖ]

Market Application:
- PC1: Overall market direction (~40% variance)
- PC2: Volatility regime (~15% variance)
- PC3: Size factor (large vs small cap)
- PC4: Momentum vs Value rotation
- PC5+: Sector-specific factors

References:
- Jolliffe, I.T. (2002). Principal Component Analysis
- Avellaneda, M. & Lee, J.H. (2010). Statistical arbitrage in the US equities market
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class PCInterpretation(Enum):
    """Interpretation of principal components in financial context."""
    MARKET_DIRECTION = "market_direction"
    VOLATILITY_REGIME = "volatility_regime"
    SIZE_FACTOR = "size_factor"
    MOMENTUM_VALUE = "momentum_value"
    SECTOR_SPECIFIC = "sector_specific"
    UNKNOWN = "unknown"


@dataclass
class PCAResult:
    """Results from PCA decomposition."""
    # Core results
    components: np.ndarray  # Principal component matrix (n_features x n_components)
    explained_variance: np.ndarray  # Variance explained by each component
    explained_variance_ratio: np.ndarray  # Proportion of variance explained
    cumulative_variance_ratio: np.ndarray  # Cumulative variance explained

    # Transformed data
    transformed_data: np.ndarray  # Data in PC space

    # Quality metrics
    n_components_selected: int
    total_variance_explained: float
    condition_number: float

    # Interpretations
    component_interpretations: List[PCInterpretation]
    feature_loadings: pd.DataFrame  # Which features contribute to which PC

    # Reconstruction
    reconstruction_error: float  # MSE of reconstruction


@dataclass
class EigenvalueResult:
    """Results from eigenvalue analysis."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    dominant_eigenvalue: float
    spectral_gap: float  # Gap between first and second eigenvalue
    effective_rank: int  # Number of significant eigenvalues
    stability_score: float  # Based on eigenvalue distribution


class PCADecomposer:
    """
    Principal Component Analysis for stock feature reduction.

    Reduces 50+ correlated features to 5-8 uncorrelated principal components,
    eliminating multicollinearity and noise while preserving signal.
    """

    def __init__(
        self,
        variance_threshold: float = 0.90,  # Target 90% variance explained
        max_components: int = 10,
        min_components: int = 3,
        standardize: bool = True
    ):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.min_components = min_components
        self.standardize = standardize

        # Fitted parameters
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.n_components_: int = 0
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'PCADecomposer':
        """
        Fit PCA on feature matrix.

        Args:
            X: Feature matrix (n_samples x n_features)
            feature_names: Optional names for features
        """
        if X.shape[0] < X.shape[1]:
            logger.warning(f"More features ({X.shape[1]}) than samples ({X.shape[0]}). "
                          "Consider reducing features first.")

        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Standardize if requested
        if self.standardize:
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
            self.std_[self.std_ == 0] = 1  # Avoid division by zero
            X_centered = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.ones(X.shape[1])
            X_centered = X - np.nanmean(X, axis=0)

        # Handle NaN values
        X_centered = np.nan_to_num(X_centered, nan=0.0)

        # Compute covariance matrix
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors

        # Determine number of components
        total_variance = np.sum(eigenvalues)
        cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

        # Find minimum components for threshold
        n_components = np.searchsorted(cumulative_variance_ratio, self.variance_threshold) + 1
        n_components = max(self.min_components, min(self.max_components, n_components))
        self.n_components_ = min(n_components, len(eigenvalues))

        logger.info(f"PCA fitted: {self.n_components_} components explain "
                   f"{cumulative_variance_ratio[self.n_components_-1]:.1%} variance")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal component space."""
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Standardize
        X_centered = (X - self.mean_) / self.std_
        X_centered = np.nan_to_num(X_centered, nan=0.0)

        # Project onto principal components
        return np.dot(X_centered, self.components_[:, :self.n_components_])

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> PCAResult:
        """
        Fit PCA and transform data, returning comprehensive results.
        """
        self.fit(X, feature_names)
        transformed = self.transform(X)

        # Calculate metrics
        total_variance = np.sum(self.eigenvalues_)
        explained_variance = self.eigenvalues_[:self.n_components_]
        explained_variance_ratio = explained_variance / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Condition number (ratio of largest to smallest eigenvalue)
        nonzero_eigenvalues = self.eigenvalues_[self.eigenvalues_ > 1e-10]
        condition_number = nonzero_eigenvalues[0] / nonzero_eigenvalues[-1] if len(nonzero_eigenvalues) > 1 else 1.0

        # Feature loadings
        loadings = pd.DataFrame(
            self.components_[:, :self.n_components_],
            index=self.feature_names_,
            columns=[f"PC{i+1}" for i in range(self.n_components_)]
        )

        # Interpret components
        interpretations = self._interpret_components(loadings)

        # Reconstruction error
        reconstructed = self.inverse_transform(transformed)
        X_centered = (X - self.mean_) / self.std_
        X_centered = np.nan_to_num(X_centered, nan=0.0)
        reconstruction_error = np.mean((X_centered - reconstructed) ** 2)

        return PCAResult(
            components=self.components_[:, :self.n_components_],
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance_ratio=cumulative_variance_ratio,
            transformed_data=transformed,
            n_components_selected=self.n_components_,
            total_variance_explained=cumulative_variance_ratio[-1],
            condition_number=condition_number,
            component_interpretations=interpretations,
            feature_loadings=loadings,
            reconstruction_error=reconstruction_error
        )

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from principal components."""
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        return np.dot(X_transformed, self.components_[:, :self.n_components_].T)

    def _interpret_components(self, loadings: pd.DataFrame) -> List[PCInterpretation]:
        """
        Interpret principal components based on feature loadings.

        Uses domain knowledge to identify what each PC represents.
        """
        interpretations = []

        # Keywords for interpretation
        market_keywords = ['return', 'momentum', 'trend', 'sma', 'ema']
        volatility_keywords = ['volatility', 'atr', 'std', 'var', 'vol']
        volume_keywords = ['volume', 'turnover', 'liquidity']

        for col in loadings.columns:
            pc_loadings = loadings[col].abs()
            top_features = pc_loadings.nlargest(5).index.tolist()
            top_features_lower = [f.lower() for f in top_features]

            # Check for market direction
            if any(any(kw in f for kw in market_keywords) for f in top_features_lower):
                if col == 'PC1':
                    interpretations.append(PCInterpretation.MARKET_DIRECTION)
                else:
                    interpretations.append(PCInterpretation.MOMENTUM_VALUE)
            # Check for volatility
            elif any(any(kw in f for kw in volatility_keywords) for f in top_features_lower):
                interpretations.append(PCInterpretation.VOLATILITY_REGIME)
            # Check for volume/size
            elif any(any(kw in f for kw in volume_keywords) for f in top_features_lower):
                interpretations.append(PCInterpretation.SIZE_FACTOR)
            else:
                interpretations.append(PCInterpretation.SECTOR_SPECIFIC)

        return interpretations

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance across all principal components.

        Uses squared loadings weighted by variance explained.
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Squared loadings weighted by variance explained
        total_variance = np.sum(self.eigenvalues_)
        weights = self.eigenvalues_[:self.n_components_] / total_variance

        squared_loadings = self.components_[:, :self.n_components_] ** 2
        importance = np.dot(squared_loadings, weights)

        df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df


class EigenvalueAnalyzer:
    """
    Advanced eigenvalue analysis for financial time series.

    Provides insights into:
    - Market correlation structure
    - Risk factor identification
    - Stability of covariance estimates
    """

    def __init__(self):
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None

    def analyze(self, returns: pd.DataFrame) -> EigenvalueResult:
        """
        Perform eigenvalue analysis on return covariance matrix.

        Args:
            returns: DataFrame of asset returns (n_periods x n_assets)
        """
        # Compute correlation matrix (more stable than covariance)
        corr_matrix = returns.corr().values

        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        # Calculate metrics
        dominant_eigenvalue = eigenvalues[0]
        spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]

        # Effective rank (number of significant eigenvalues)
        # Using Marchenko-Pastur bound for random matrix
        n_samples, n_assets = returns.shape
        q = n_samples / n_assets
        mp_upper = (1 + 1/np.sqrt(q)) ** 2  # Upper bound of random eigenvalues
        effective_rank = np.sum(eigenvalues > mp_upper)

        # Stability score based on eigenvalue distribution
        # Higher score = more stable covariance structure
        entropy = -np.sum(eigenvalues/np.sum(eigenvalues) *
                         np.log(eigenvalues/np.sum(eigenvalues) + 1e-10))
        max_entropy = np.log(len(eigenvalues))
        stability_score = 1 - (entropy / max_entropy)  # Lower entropy = more concentrated = more stable

        return EigenvalueResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            dominant_eigenvalue=dominant_eigenvalue,
            spectral_gap=spectral_gap,
            effective_rank=effective_rank,
            stability_score=stability_score
        )

    def get_market_mode(self) -> Tuple[np.ndarray, float]:
        """
        Extract the market mode (first eigenvector) and its explanatory power.

        The first eigenvector typically represents the market factor.
        Returns:
            (market_mode_weights, variance_explained)
        """
        if self.eigenvalues_ is None:
            raise ValueError("Analysis not performed. Call analyze() first.")

        market_mode = self.eigenvectors_[:, 0]
        variance_explained = self.eigenvalues_[0] / np.sum(self.eigenvalues_)

        return market_mode, variance_explained

    def detect_correlation_breakdown(self, returns: pd.DataFrame,
                                      window: int = 60) -> pd.Series:
        """
        Detect periods where correlation structure breaks down.

        Uses rolling eigenvalue analysis to identify regime changes.
        """
        n_periods = len(returns)
        breakdown_scores = []

        baseline_eigenvalues = None

        for i in range(window, n_periods):
            window_returns = returns.iloc[i-window:i]

            try:
                corr_matrix = window_returns.corr().values
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                np.fill_diagonal(corr_matrix, 1.0)

                eigenvalues, _ = np.linalg.eigh(corr_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]

                if baseline_eigenvalues is None:
                    baseline_eigenvalues = eigenvalues
                    breakdown_scores.append(0.0)
                else:
                    # Measure distance from baseline eigenvalue distribution
                    diff = np.abs(eigenvalues - baseline_eigenvalues)
                    breakdown_score = np.sum(diff) / np.sum(baseline_eigenvalues)
                    breakdown_scores.append(breakdown_score)

                    # Update baseline with exponential decay
                    baseline_eigenvalues = 0.95 * baseline_eigenvalues + 0.05 * eigenvalues

            except Exception:
                breakdown_scores.append(0.0)

        # Pad beginning with zeros
        breakdown_scores = [0.0] * window + breakdown_scores

        return pd.Series(breakdown_scores, index=returns.index, name='correlation_breakdown')


def create_feature_matrix(price_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create comprehensive feature matrix for PCA from price data.

    Args:
        price_data: OHLCV DataFrame

    Returns:
        (feature_matrix, feature_names)
    """
    df = price_data.copy()
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    volume = df['volume'] if 'volume' in df.columns else df['Volume']

    features = {}

    # Returns at multiple horizons
    for period in [1, 2, 3, 5, 10, 20]:
        features[f'return_{period}d'] = close.pct_change(period)

    # Moving averages relative to price
    for period in [5, 10, 20, 50]:
        sma = close.rolling(period).mean()
        features[f'dist_sma_{period}'] = (close - sma) / sma

    # Momentum indicators
    features['momentum_5d'] = close.pct_change(5)
    features['momentum_10d'] = close.pct_change(10)
    features['momentum_20d'] = close.pct_change(20)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']

    # Volatility at multiple scales
    returns = close.pct_change()
    for period in [5, 10, 20, 60]:
        features[f'volatility_{period}d'] = returns.rolling(period).std()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()
    features['atr_relative'] = features['atr_14'] / close

    # Volume features
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_position'] = (close - sma20) / (2 * std20)

    # Range features
    features['daily_range'] = (high - low) / close
    features['high_low_ratio'] = high.rolling(20).max() / low.rolling(20).min() - 1

    # Create DataFrame and handle NaN
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.dropna()

    return feature_df.values, list(feature_df.columns)
