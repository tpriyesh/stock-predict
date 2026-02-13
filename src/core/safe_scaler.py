"""
SafeFeatureScaler - Prevents Data Leakage in Feature Engineering

CRITICAL FIX: Standard scaling on full dataset LEAKS FUTURE INFORMATION.

WRONG (Data Leakage):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Sees ALL data including future
    X_train, X_test = split(df_scaled)

CORRECT (No Leakage):
    X_train_raw, X_test_raw = split(df)  # Split FIRST
    scaler = SafeFeatureScaler()
    X_train = scaler.fit_transform_train(X_train_raw)  # Fit ONLY on training
    X_test = scaler.transform_test(X_test_raw)  # Apply same params to test

This module ensures you CANNOT accidentally leak future data into your features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - using numpy-based scaling")


class ScalingMethod(Enum):
    """Supported scaling methods."""
    STANDARD = "standard"       # z-score: (x - mean) / std
    ROBUST = "robust"           # Median and IQR (resistant to outliers)
    MINMAX = "minmax"           # Scale to [0, 1]
    LOG_STANDARD = "log_standard"  # Log transform then z-score
    NONE = "none"               # No scaling


@dataclass
class ScalerState:
    """Immutable record of scaler parameters from training data."""
    method: ScalingMethod
    columns: List[str]
    params: Dict[str, Dict[str, float]]  # column -> {mean, std, median, q1, q3, min, max}
    fitted_on_rows: int
    fitted_timestamp: str

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            'method': self.method.value,
            'columns': self.columns,
            'params': self.params,
            'fitted_on_rows': self.fitted_on_rows,
            'fitted_timestamp': self.fitted_timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ScalerState':
        """Deserialize from storage."""
        return cls(
            method=ScalingMethod(data['method']),
            columns=data['columns'],
            params=data['params'],
            fitted_on_rows=data['fitted_on_rows'],
            fitted_timestamp=data['fitted_timestamp']
        )


class SafeFeatureScaler:
    """
    Production-grade feature scaler that PREVENTS data leakage.

    Key features:
    1. Enforces fit on training data ONLY
    2. Raises errors if you try to transform before fitting
    3. Tracks state for reproducibility
    4. Supports multiple scaling methods
    5. Handles edge cases (constant columns, NaN, inf)

    Usage:
        scaler = SafeFeatureScaler(method='robust')

        # During walk-forward validation:
        for train_idx, test_idx in cv.split(X):
            X_train = scaler.fit_transform_train(df.iloc[train_idx])
            X_test = scaler.transform_test(df.iloc[test_idx])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
    """

    def __init__(self,
                 method: Union[str, ScalingMethod] = 'robust',
                 clip_outliers: bool = True,
                 outlier_threshold: float = 5.0,
                 handle_constant: str = 'zero'):
        """
        Initialize safe scaler.

        Args:
            method: Scaling method ('standard', 'robust', 'minmax', 'log_standard', 'none')
            clip_outliers: Whether to clip extreme values before scaling
            outlier_threshold: Number of std/IQR for outlier detection
            handle_constant: How to handle constant columns ('zero', 'drop', 'error')
        """
        if isinstance(method, str):
            self.method = ScalingMethod(method.lower())
        else:
            self.method = method

        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold
        self.handle_constant = handle_constant

        self._state: Optional[ScalerState] = None
        self._is_fitted = False
        self._fit_called_count = 0

    @property
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted

    @property
    def state(self) -> Optional[ScalerState]:
        """Get current scaler state (read-only)."""
        return self._state

    def _validate_input(self, X: pd.DataFrame, phase: str) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be pandas DataFrame, got {type(X)}")

        if X.empty:
            raise ValueError(f"Input DataFrame is empty during {phase}")

        # Check for infinite values
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in {phase} data - replacing with NaN")

    def _calculate_params(self, column: pd.Series) -> Dict[str, float]:
        """Calculate scaling parameters for a single column."""
        # Remove NaN/inf for calculation
        clean = column.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean) == 0:
            return {'mean': 0, 'std': 1, 'median': 0, 'q1': 0, 'q3': 1, 'min': 0, 'max': 1}

        return {
            'mean': float(clean.mean()),
            'std': float(clean.std()) if clean.std() > 0 else 1.0,
            'median': float(clean.median()),
            'q1': float(clean.quantile(0.25)),
            'q3': float(clean.quantile(0.75)),
            'min': float(clean.min()),
            'max': float(clean.max()),
            'count': int(len(clean))
        }

    def _apply_scaling(self, value: float, params: Dict[str, float]) -> float:
        """Apply scaling transformation to a single value."""
        if pd.isna(value) or np.isinf(value):
            return 0.0  # Default for missing/infinite

        if self.method == ScalingMethod.NONE:
            return value

        elif self.method == ScalingMethod.STANDARD:
            std = params['std'] if params['std'] > 0 else 1.0
            return (value - params['mean']) / std

        elif self.method == ScalingMethod.ROBUST:
            iqr = params['q3'] - params['q1']
            iqr = iqr if iqr > 0 else 1.0
            return (value - params['median']) / iqr

        elif self.method == ScalingMethod.MINMAX:
            range_val = params['max'] - params['min']
            range_val = range_val if range_val > 0 else 1.0
            return (value - params['min']) / range_val

        elif self.method == ScalingMethod.LOG_STANDARD:
            # Log transform (handle negative/zero)
            log_val = np.log1p(np.abs(value)) * np.sign(value) if value != 0 else 0
            std = params['std'] if params['std'] > 0 else 1.0
            return (log_val - params['mean']) / std

        return value

    def _clip_value(self, value: float, params: Dict[str, float]) -> float:
        """Clip outliers based on training data statistics."""
        if not self.clip_outliers or pd.isna(value):
            return value

        if self.method == ScalingMethod.ROBUST:
            iqr = params['q3'] - params['q1']
            lower = params['q1'] - self.outlier_threshold * iqr
            upper = params['q3'] + self.outlier_threshold * iqr
        else:
            lower = params['mean'] - self.outlier_threshold * params['std']
            upper = params['mean'] + self.outlier_threshold * params['std']

        return np.clip(value, lower, upper)

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on training data and transform it.

        This is the ONLY way to fit the scaler. You cannot fit on test data.

        Args:
            X_train: Training data (time-ordered, no future information)

        Returns:
            Scaled training data
        """
        from datetime import datetime

        self._validate_input(X_train, "fit_transform_train")
        self._fit_called_count += 1

        logger.debug(f"Fitting SafeFeatureScaler on {len(X_train)} rows, {len(X_train.columns)} columns")

        # Calculate parameters for each column
        params = {}
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        constant_cols = []
        for col in numeric_cols:
            col_params = self._calculate_params(X_train[col])
            params[col] = col_params

            # Check for constant columns
            if col_params['std'] < 1e-10:
                constant_cols.append(col)

        if constant_cols:
            if self.handle_constant == 'error':
                raise ValueError(f"Constant columns found: {constant_cols}")
            elif self.handle_constant == 'drop':
                numeric_cols = [c for c in numeric_cols if c not in constant_cols]
                logger.warning(f"Dropping constant columns: {constant_cols}")
            else:  # 'zero'
                logger.warning(f"Constant columns will be scaled to zero: {constant_cols}")

        # Store state
        self._state = ScalerState(
            method=self.method,
            columns=numeric_cols,
            params=params,
            fitted_on_rows=len(X_train),
            fitted_timestamp=datetime.now().isoformat()
        )
        self._is_fitted = True

        # Transform training data
        return self._transform(X_train, "train")

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using parameters from training.

        CRITICAL: This method can ONLY be called after fit_transform_train.
        It uses ONLY the parameters learned from training data.

        Args:
            X_test: Test data (must not overlap with training)

        Returns:
            Scaled test data
        """
        if not self._is_fitted:
            raise RuntimeError(
                "SafeFeatureScaler must be fitted on training data first! "
                "Call fit_transform_train(X_train) before transform_test(X_test)."
            )

        self._validate_input(X_test, "transform_test")

        logger.debug(f"Transforming test data: {len(X_test)} rows")

        return self._transform(X_test, "test")

    def _transform(self, X: pd.DataFrame, phase: str) -> pd.DataFrame:
        """Internal transform method."""
        result = X.copy()

        for col in self._state.columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found in {phase} data - skipping")
                continue

            params = self._state.params[col]

            # Apply transformation
            values = result[col].values.astype(float)
            scaled = np.zeros_like(values)

            for i, val in enumerate(values):
                clipped = self._clip_value(val, params)
                scaled[i] = self._apply_scaling(clipped, params)

            result[col] = scaled

        return result

    def reset(self) -> None:
        """Reset scaler state (for use in new walk-forward fold)."""
        self._state = None
        self._is_fitted = False
        logger.debug("SafeFeatureScaler reset")

    def get_feature_stats(self) -> pd.DataFrame:
        """Get statistics of features from training data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted yet")

        stats = []
        for col in self._state.columns:
            p = self._state.params[col]
            stats.append({
                'column': col,
                'mean': p['mean'],
                'std': p['std'],
                'median': p['median'],
                'q1': p['q1'],
                'q3': p['q3'],
                'min': p['min'],
                'max': p['max']
            })

        return pd.DataFrame(stats)


class WalkForwardScaler:
    """
    Scaler specifically designed for walk-forward validation.

    Creates a new SafeFeatureScaler for each fold to prevent any
    possibility of data leakage between folds.

    Usage:
        wf_scaler = WalkForwardScaler(method='robust')

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train_scaled, X_test_scaled = wf_scaler.fit_transform_fold(
                X.iloc[train_idx],
                X.iloc[test_idx],
                fold_id
            )
    """

    def __init__(self, method: str = 'robust', **scaler_kwargs):
        self.method = method
        self.scaler_kwargs = scaler_kwargs
        self.fold_scalers: Dict[int, SafeFeatureScaler] = {}
        self.current_fold: Optional[int] = None

    def fit_transform_fold(self,
                           X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           fold_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit and transform for a single fold.

        Creates a fresh scaler for each fold - no information leakage possible.

        Args:
            X_train: Training data for this fold
            X_test: Test data for this fold
            fold_id: Identifier for this fold

        Returns:
            Tuple of (scaled_train, scaled_test)
        """
        # Create fresh scaler for this fold
        scaler = SafeFeatureScaler(method=self.method, **self.scaler_kwargs)

        # Fit on training, transform both
        X_train_scaled = scaler.fit_transform_train(X_train)
        X_test_scaled = scaler.transform_test(X_test)

        # Store for potential inspection
        self.fold_scalers[fold_id] = scaler
        self.current_fold = fold_id

        logger.debug(f"Fold {fold_id}: Scaled {len(X_train)} train, {len(X_test)} test rows")

        return X_train_scaled, X_test_scaled

    def get_fold_scaler(self, fold_id: int) -> Optional[SafeFeatureScaler]:
        """Get scaler used for a specific fold."""
        return self.fold_scalers.get(fold_id)

    def get_all_feature_stats(self) -> pd.DataFrame:
        """Get feature statistics across all folds."""
        all_stats = []
        for fold_id, scaler in self.fold_scalers.items():
            stats = scaler.get_feature_stats()
            stats['fold'] = fold_id
            all_stats.append(stats)

        if not all_stats:
            return pd.DataFrame()

        return pd.concat(all_stats, ignore_index=True)


def demo():
    """Demonstrate safe scaling."""
    print("=" * 60)
    print("SafeFeatureScaler Demo - Preventing Data Leakage")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'feature_a': np.random.randn(n) * 10 + 50,
        'feature_b': np.random.randn(n) * 5 + 100,
        'feature_c': np.random.exponential(2, n),
    })

    # Split (time-based)
    split_idx = 80
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"\nOriginal data:")
    print(f"  Train: {len(train)} rows")
    print(f"  Test: {len(test)} rows")

    # Safe scaling
    scaler = SafeFeatureScaler(method='robust')

    train_scaled = scaler.fit_transform_train(train)
    test_scaled = scaler.transform_test(test)

    print(f"\nScaled data statistics:")
    print(f"  Train mean: {train_scaled.mean().mean():.4f}")
    print(f"  Test mean: {test_scaled.mean().mean():.4f}")

    print(f"\nFeature statistics from training:")
    print(scaler.get_feature_stats())

    # Demonstrate error handling
    print("\n--- Error Handling ---")
    try:
        scaler2 = SafeFeatureScaler()
        scaler2.transform_test(test)  # Should fail
    except RuntimeError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    demo()
