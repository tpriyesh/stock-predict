"""
PurgedKFoldCV - Time-Series Cross-Validation with Embargo

CRITICAL: Standard K-Fold CV LEAKS information in time-series data.

Problem with standard K-Fold:
- Training on [Jan-Mar] + [Jul-Sep], testing on [Apr-Jun]
- But features use past N days, so [Jul] training sees [Jun] info
- 5-day forward labels overlap between folds

Solution: Purged K-Fold with Embargo
1. PURGE: Remove samples from training where label overlaps with test
2. EMBARGO: Additional gap after test to prevent feature leakage
3. TIME-ORDER: Only train on past, test on future

This mimics real trading: you can ONLY use past data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Generator
from dataclasses import dataclass
from loguru import logger


@dataclass
class TimeSeriesEmbargo:
    """
    Configuration for time-series embargo periods.

    In financial ML, we must account for:
    1. Label horizon (e.g., 5-day returns = 5-day overlap)
    2. Feature lookback (e.g., 20-day MA uses 20 past days)
    3. Additional safety buffer
    """
    label_horizon: int = 5        # Days forward for label calculation
    feature_lookback: int = 20    # Max lookback for features
    safety_buffer: int = 2        # Extra buffer for safety

    @property
    def total_embargo(self) -> int:
        """Total embargo period in days."""
        return self.label_horizon + self.feature_lookback + self.safety_buffer

    @property
    def purge_period(self) -> int:
        """Samples to purge before test set."""
        return self.label_horizon


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for time-series data.

    This is the CORRECT way to do CV for financial ML.

    Standard K-Fold (WRONG):
    ├── Fold 1: Train [A B C D] Test [E]
    ├── Fold 2: Train [A B C E] Test [D]  <- D's label uses E's prices!
    └── ... data leakage everywhere

    Purged K-Fold (CORRECT):
    ├── Fold 1: Train [A B] --purge-- Test [C]
    ├── Fold 2: Train [A B C] --purge-- Test [D]
    └── Only ever train on PAST, test on FUTURE

    Usage:
        cv = PurgedKFoldCV(n_splits=5, embargo=TimeSeriesEmbargo(label_horizon=5))

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
    """

    def __init__(self,
                 n_splits: int = 5,
                 embargo: Optional[TimeSeriesEmbargo] = None,
                 test_size: Optional[int] = None,
                 expanding: bool = True):
        """
        Initialize purged K-fold CV.

        Args:
            n_splits: Number of folds
            embargo: Embargo configuration (default: 5-day labels, 20-day features)
            test_size: Fixed test size (if None, calculated from n_splits)
            expanding: If True, training window expands; if False, rolling window
        """
        self.n_splits = n_splits
        self.embargo = embargo or TimeSeriesEmbargo()
        self.test_size = test_size
        self.expanding = expanding

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with proper purging.

        Args:
            X: Feature DataFrame (must have DatetimeIndex or sequential)
            y: Labels (optional, for signature compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Calculate fold sizes
        if self.test_size:
            fold_test_size = self.test_size
        else:
            fold_test_size = n_samples // (self.n_splits + 1)

        purge_period = self.embargo.purge_period

        logger.debug(f"PurgedKFoldCV: {n_samples} samples, {self.n_splits} folds, "
                    f"purge={purge_period}, embargo={self.embargo.total_embargo}")

        for fold in range(self.n_splits):
            # Test set boundaries
            test_end = n_samples - fold * fold_test_size
            test_start = test_end - fold_test_size

            if test_start < 0:
                break

            # Training set: everything before purge point
            train_end = max(0, test_start - purge_period)

            if self.expanding:
                # Expanding window: start from beginning
                train_start = 0
            else:
                # Rolling window: fixed size
                window_size = fold_test_size * 3  # 3x test size
                train_start = max(0, train_end - window_size)

            # Skip if not enough training data
            if train_end - train_start < fold_test_size:
                logger.warning(f"Fold {fold}: Insufficient training data, skipping")
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            logger.debug(f"Fold {fold}: Train [{train_start}:{train_end}], "
                        f"Purge [{train_end}:{test_start}], "
                        f"Test [{test_start}:{test_end}]")

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation (Anchored or Rolling).

    This is the most realistic form of backtesting as it exactly
    mimics how you would trade: train on all past data, test on next period.

    Walk-Forward:
    ├── Window 1: Train [Jan-Jun] → Test [Jul]
    ├── Window 2: Train [Jan-Jul] → Test [Aug]
    ├── Window 3: Train [Jan-Aug] → Test [Sep]
    └── ... always training on ALL past, testing on NEXT period

    This is different from standard train/test split because:
    1. Multiple test periods give better estimate of future performance
    2. Model is retrained for each period (as in production)
    3. Recency bias can be detected (performance declining over time?)
    """

    def __init__(self,
                 train_size: int = 252,      # 1 year of trading days
                 test_size: int = 21,        # 1 month
                 step_size: Optional[int] = None,  # Step between windows
                 embargo: Optional[TimeSeriesEmbargo] = None,
                 anchored: bool = True):     # If True, training starts from beginning
        """
        Initialize walk-forward CV.

        Args:
            train_size: Minimum training window size
            test_size: Test window size
            step_size: Step between windows (default: test_size)
            embargo: Embargo configuration
            anchored: If True, training always starts from sample 0
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.embargo = embargo or TimeSeriesEmbargo()
        self.anchored = anchored

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test indices.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        purge = self.embargo.purge_period

        # Calculate number of possible windows
        usable_samples = n_samples - self.train_size - purge - self.test_size
        n_windows = max(1, usable_samples // self.step_size + 1)

        logger.debug(f"WalkForwardCV: {n_samples} samples, {n_windows} windows")

        for window in range(n_windows):
            # Test window
            test_start = self.train_size + purge + window * self.step_size
            test_end = min(test_start + self.test_size, n_samples)

            if test_start >= n_samples:
                break

            # Training window
            train_end = test_start - purge

            if self.anchored:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_size)

            if train_end <= train_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """Estimate number of splits for given data size."""
        n_samples = len(X)
        purge = self.embargo.purge_period
        usable = n_samples - self.train_size - purge - self.test_size
        return max(0, usable // self.step_size + 1)


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    More rigorous than standard K-Fold: instead of K folds, we test
    on MULTIPLE test sets to get better coverage.

    This is computationally expensive but gives better estimates
    of out-of-sample performance.

    Example with K=5, test_combination_size=2:
    - Test on folds [1,2], train on [3,4,5] (purged)
    - Test on folds [1,3], train on [2,4,5] (purged)
    - ... C(5,2) = 10 combinations total
    """

    def __init__(self,
                 n_splits: int = 5,
                 test_combination_size: int = 2,
                 embargo: Optional[TimeSeriesEmbargo] = None):
        self.n_splits = n_splits
        self.test_combination_size = test_combination_size
        self.embargo = embargo or TimeSeriesEmbargo()

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate combinatorial purged splits."""
        from itertools import combinations

        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        purge = self.embargo.purge_period

        # Create fold boundaries
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            folds.append(list(range(start, end)))

        # Generate all combinations of test folds
        for test_fold_ids in combinations(range(self.n_splits), self.test_combination_size):
            test_indices = []
            for fold_id in test_fold_ids:
                test_indices.extend(folds[fold_id])

            # Training: all other folds with purging
            train_indices = []
            for fold_id in range(self.n_splits):
                if fold_id not in test_fold_ids:
                    fold_indices = folds[fold_id]

                    # Check if this fold is adjacent to any test fold
                    needs_purge = False
                    for test_id in test_fold_ids:
                        if abs(fold_id - test_id) == 1:
                            needs_purge = True
                            break

                    if needs_purge:
                        # Remove indices within purge period of test
                        for idx in fold_indices:
                            min_test = min(test_indices)
                            max_test = max(test_indices)
                            if idx < min_test - purge or idx > max_test + purge:
                                train_indices.append(idx)
                    else:
                        train_indices.extend(fold_indices)

            if train_indices and test_indices:
                yield np.array(sorted(train_indices)), np.array(sorted(test_indices))


def calculate_cv_scores(model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: PurgedKFoldCV,
                        scaler=None) -> dict:
    """
    Calculate cross-validation scores with proper purging.

    Args:
        model: Sklearn-compatible model
        X: Features
        y: Labels
        cv: Cross-validation splitter
        scaler: Optional scaler (will be fitted per fold)

    Returns:
        Dictionary with scores, metrics, and diagnostics
    """
    from .safe_scaler import SafeFeatureScaler

    scores = []
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Scale with safe scaler (no leakage)
        if scaler is None:
            scaler = SafeFeatureScaler(method='robust')

        X_train_scaled = scaler.fit_transform_train(X_train)
        X_test_scaled = scaler.transform_test(X_test)
        scaler.reset()  # Reset for next fold

        # Train and evaluate
        model.fit(X_train_scaled.values, y_train.values)
        score = model.score(X_test_scaled.values, y_test.values)
        scores.append(score)

        # Predictions for detailed analysis
        y_pred = model.predict(X_test_scaled.values)
        accuracy = (y_pred == y_test.values).mean()

        fold_details.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score,
            'accuracy': accuracy,
            'train_start': train_idx[0] if len(train_idx) > 0 else None,
            'train_end': train_idx[-1] if len(train_idx) > 0 else None,
            'test_start': test_idx[0] if len(test_idx) > 0 else None,
            'test_end': test_idx[-1] if len(test_idx) > 0 else None
        })

        logger.debug(f"Fold {fold_idx}: score={score:.3f}, "
                    f"train={len(train_idx)}, test={len(test_idx)}")

    return {
        'scores': scores,
        'mean_score': np.mean(scores) if scores else 0,
        'std_score': np.std(scores) if scores else 0,
        'n_folds': len(scores),
        'fold_details': fold_details,
        'is_overfit': np.std(scores) > 0.1 if scores else True  # High variance = overfit
    }


def demo():
    """Demonstrate purged cross-validation."""
    print("=" * 60)
    print("PurgedKFoldCV Demo")
    print("=" * 60)

    # Create sample time-series data
    np.random.seed(42)
    n = 500  # 2 years of trading days

    dates = pd.date_range(start='2022-01-01', periods=n, freq='B')
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n)
    }, index=dates)

    y = pd.Series(np.random.randint(0, 2, n), index=dates)

    print(f"\nData: {n} samples, {len(X.columns)} features")

    # Standard (wrong) K-Fold
    print("\n--- Standard K-Fold (WRONG for time-series) ---")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)
    for i, (train, test) in enumerate(kf.split(X)):
        print(f"Fold {i}: Train samples include future data after test!")
        break

    # Purged K-Fold (correct)
    print("\n--- Purged K-Fold (CORRECT) ---")
    cv = PurgedKFoldCV(n_splits=5, embargo=TimeSeriesEmbargo(label_horizon=5))

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: {train_idx[0]} to {train_idx[-1]} ({len(train_idx)} samples)")
        print(f"  Test: {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} samples)")
        print(f"  Gap: {test_idx[0] - train_idx[-1]} samples (purge period)")

    # Walk-Forward CV
    print("\n--- Walk-Forward CV ---")
    wf_cv = WalkForwardCV(train_size=252, test_size=21)

    for i, (train_idx, test_idx) in enumerate(wf_cv.split(X)):
        if i >= 3:
            print(f"  ... and {wf_cv.get_n_splits(X) - 3} more windows")
            break
        print(f"Window {i}:")
        print(f"  Train: {train_idx[0]} to {train_idx[-1]}")
        print(f"  Test: {test_idx[0]} to {test_idx[-1]}")


if __name__ == "__main__":
    demo()
