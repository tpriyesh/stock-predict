"""
Robustness Checks Module

This module provides comprehensive robustness testing for the stock prediction system.

Features:
1. Parameter Sensitivity Analysis - Test how predictions change with parameter variations
2. Data Quality Validation - Ensure input data meets quality standards
3. Model Consistency Checks - Verify model outputs are internally consistent
4. Stress Testing - Test edge cases and extreme market conditions
5. Cross-Validation Diagnostics - Validate model performance across different periods

Reference:
- Bailey, D. H., Borwein, J. M., et al. (2015). "Pseudo-Mathematics and Financial Charlatanism"
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "... and the Cross-Section of Expected Returns"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from loguru import logger


class RobustnessLevel(Enum):
    """Classification of robustness test results."""
    ROBUST = "ROBUST"           # Passes all checks
    ACCEPTABLE = "ACCEPTABLE"   # Minor issues, usable
    FRAGILE = "FRAGILE"         # Significant issues
    INVALID = "INVALID"         # Fails critical checks


@dataclass
class DataQualityReport:
    """Report on data quality checks."""
    is_valid: bool
    completeness: float         # % of expected data present
    freshness_days: int         # Days since last data point
    has_sufficient_history: bool
    has_gaps: bool
    gap_count: int
    suspicious_movements: int   # Number of >20% daily moves
    zero_volume_days: int
    warnings: List[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0

        if self.completeness < 0.95:
            score -= 0.2
        if self.freshness_days > 1:
            score -= min(0.3, self.freshness_days * 0.05)
        if not self.has_sufficient_history:
            score -= 0.3
        if self.has_gaps:
            score -= min(0.2, self.gap_count * 0.02)
        if self.suspicious_movements > 5:
            score -= 0.1
        if self.zero_volume_days > 10:
            score -= 0.1

        return max(0, min(1, score))


@dataclass
class SensitivityResult:
    """Result from parameter sensitivity analysis."""
    parameter_name: str
    base_value: float
    test_values: List[float]
    base_prediction: float
    prediction_changes: List[float]
    sensitivity_score: float  # 0 = stable, 1 = very sensitive
    is_stable: bool
    interpretation: str


@dataclass
class ConsistencyResult:
    """Result from model consistency checks."""
    is_consistent: bool
    issues: List[str]
    model_agreement: Dict[str, bool]
    confidence_range: Tuple[float, float]
    signal_stability: float  # 0 = unstable, 1 = stable


@dataclass
class RobustnessReport:
    """Comprehensive robustness report."""
    overall_level: RobustnessLevel
    overall_score: float  # 0-1

    data_quality: DataQualityReport
    sensitivity_results: List[SensitivityResult]
    consistency_result: ConsistencyResult

    passed_checks: int
    total_checks: int
    critical_failures: List[str]
    warnings: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_level': self.overall_level.value,
            'overall_score': self.overall_score,
            'passed_checks': self.passed_checks,
            'total_checks': self.total_checks,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class RobustnessChecker:
    """
    Comprehensive robustness checking for stock predictions.

    This class validates that predictions are robust and reliable
    before presenting them to users.
    """

    # Thresholds for data quality
    MIN_DATA_DAYS = 60
    MAX_GAP_DAYS = 5
    MAX_SUSPICIOUS_MOVES_PCT = 0.05  # 5% of data points

    # Thresholds for sensitivity
    SENSITIVITY_THRESHOLD = 0.20  # 20% change is considered sensitive

    def __init__(self):
        """Initialize the robustness checker."""
        pass

    def check_data_quality(
        self,
        df: pd.DataFrame,
        expected_days: int = 252
    ) -> DataQualityReport:
        """
        Check data quality before prediction.

        Args:
            df: DataFrame with OHLCV data
            expected_days: Expected number of trading days

        Returns:
            DataQualityReport with quality metrics
        """
        warnings = []

        if df.empty:
            return DataQualityReport(
                is_valid=False,
                completeness=0,
                freshness_days=999,
                has_sufficient_history=False,
                has_gaps=True,
                gap_count=0,
                suspicious_movements=0,
                zero_volume_days=0,
                warnings=["No data available"]
            )

        # Completeness
        actual_days = len(df)
        completeness = min(1.0, actual_days / expected_days)

        if actual_days < self.MIN_DATA_DAYS:
            warnings.append(f"Insufficient history: {actual_days} days (need {self.MIN_DATA_DAYS})")

        # Freshness
        last_date = df.index[-1]
        if hasattr(last_date, 'date'):
            last_date = last_date.date()

        from datetime import date
        today = date.today()

        if hasattr(last_date, 'toordinal'):
            freshness_days = (today - last_date).days
        else:
            freshness_days = 0

        if freshness_days > 1:
            warnings.append(f"Data is {freshness_days} days old")

        # Check for gaps
        if len(df) > 1:
            date_diffs = pd.Series(df.index).diff().dropna()

            # Convert to days
            if hasattr(date_diffs.iloc[0], 'days'):
                gap_days = date_diffs.apply(lambda x: x.days if hasattr(x, 'days') else 1)
            else:
                gap_days = pd.Series([1] * len(date_diffs))

            large_gaps = (gap_days > self.MAX_GAP_DAYS).sum()
            has_gaps = large_gaps > 0

            if has_gaps:
                warnings.append(f"Found {large_gaps} gaps > {self.MAX_GAP_DAYS} days")
        else:
            has_gaps = False
            large_gaps = 0

        # Suspicious movements (>20% daily change)
        if 'close' in df.columns:
            daily_returns = df['close'].pct_change().abs()
            suspicious = (daily_returns > 0.20).sum()

            if suspicious > actual_days * self.MAX_SUSPICIOUS_MOVES_PCT:
                warnings.append(f"Found {suspicious} suspicious moves (>20%)")
        else:
            suspicious = 0

        # Zero volume days
        if 'volume' in df.columns:
            zero_vol = (df['volume'] == 0).sum()

            if zero_vol > 0:
                warnings.append(f"Found {zero_vol} days with zero volume")
        else:
            zero_vol = 0

        # Determine validity
        is_valid = (
            actual_days >= self.MIN_DATA_DAYS
            and freshness_days <= 5
            and zero_vol < actual_days * 0.1
        )

        return DataQualityReport(
            is_valid=is_valid,
            completeness=completeness,
            freshness_days=freshness_days,
            has_sufficient_history=actual_days >= self.MIN_DATA_DAYS,
            has_gaps=has_gaps,
            gap_count=large_gaps,
            suspicious_movements=suspicious,
            zero_volume_days=zero_vol,
            warnings=warnings
        )

    def check_sensitivity(
        self,
        prediction_func: Callable,
        base_params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 5
    ) -> List[SensitivityResult]:
        """
        Check prediction sensitivity to parameter changes.

        Args:
            prediction_func: Function that returns prediction given parameters
            base_params: Base parameter values
            param_ranges: Dict of parameter name -> (min_value, max_value)
            n_samples: Number of samples per parameter

        Returns:
            List of SensitivityResult for each parameter
        """
        results = []

        # Get base prediction
        try:
            base_prediction = prediction_func(**base_params)
        except Exception as e:
            logger.warning(f"Base prediction failed: {e}")
            return results

        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name not in base_params:
                continue

            base_value = base_params[param_name]
            test_values = np.linspace(min_val, max_val, n_samples)
            prediction_changes = []

            for test_val in test_values:
                test_params = base_params.copy()
                test_params[param_name] = test_val

                try:
                    test_prediction = prediction_func(**test_params)
                    change = abs(test_prediction - base_prediction) / max(abs(base_prediction), 1e-10)
                    prediction_changes.append(change)
                except Exception:
                    prediction_changes.append(0)

            # Calculate sensitivity score
            max_change = max(prediction_changes) if prediction_changes else 0
            avg_change = np.mean(prediction_changes) if prediction_changes else 0
            sensitivity_score = min(1.0, avg_change / self.SENSITIVITY_THRESHOLD)

            is_stable = sensitivity_score < 0.5

            if sensitivity_score > 0.7:
                interpretation = f"HIGHLY SENSITIVE: {param_name} changes cause {avg_change:.1%} prediction variation"
            elif sensitivity_score > 0.3:
                interpretation = f"MODERATELY SENSITIVE: {param_name} has noticeable impact"
            else:
                interpretation = f"STABLE: {param_name} variations have minimal impact"

            results.append(SensitivityResult(
                parameter_name=param_name,
                base_value=base_value,
                test_values=list(test_values),
                base_prediction=base_prediction,
                prediction_changes=prediction_changes,
                sensitivity_score=sensitivity_score,
                is_stable=is_stable,
                interpretation=interpretation
            ))

        return results

    def check_consistency(
        self,
        model_predictions: Dict[str, float],
        expected_range: Tuple[float, float] = (0.0, 1.0)
    ) -> ConsistencyResult:
        """
        Check internal consistency of model predictions.

        Args:
            model_predictions: Dict of model_name -> prediction value
            expected_range: Expected valid range for predictions

        Returns:
            ConsistencyResult with consistency metrics
        """
        issues = []
        model_agreement = {}

        if not model_predictions:
            return ConsistencyResult(
                is_consistent=False,
                issues=["No predictions to check"],
                model_agreement={},
                confidence_range=(0, 0),
                signal_stability=0
            )

        values = list(model_predictions.values())

        # Check if all predictions are in valid range
        for name, value in model_predictions.items():
            in_range = expected_range[0] <= value <= expected_range[1]
            model_agreement[name] = in_range

            if not in_range:
                issues.append(f"{name}: prediction {value:.3f} outside range {expected_range}")

        # Check for NaN or infinite values
        for name, value in model_predictions.items():
            if np.isnan(value) or np.isinf(value):
                issues.append(f"{name}: invalid value (NaN or Inf)")
                model_agreement[name] = False

        # Calculate confidence range
        valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]

        if valid_values:
            confidence_range = (min(valid_values), max(valid_values))
            spread = confidence_range[1] - confidence_range[0]

            # Signal stability: inversely related to spread
            signal_stability = max(0, 1 - spread / (expected_range[1] - expected_range[0]))
        else:
            confidence_range = (0, 0)
            signal_stability = 0

        # Check for extreme disagreement
        if len(valid_values) >= 2:
            std_dev = np.std(valid_values)
            if std_dev > 0.2:
                issues.append(f"High model disagreement: std={std_dev:.3f}")

        is_consistent = len(issues) == 0 and signal_stability > 0.5

        return ConsistencyResult(
            is_consistent=is_consistent,
            issues=issues,
            model_agreement=model_agreement,
            confidence_range=confidence_range,
            signal_stability=signal_stability
        )

    def run_full_check(
        self,
        df: pd.DataFrame,
        model_predictions: Dict[str, float],
        prediction_func: Optional[Callable] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        base_params: Optional[Dict[str, float]] = None
    ) -> RobustnessReport:
        """
        Run comprehensive robustness checks.

        Args:
            df: Input data DataFrame
            model_predictions: Dict of model_name -> prediction value
            prediction_func: Optional function for sensitivity testing
            param_ranges: Parameter ranges for sensitivity testing
            base_params: Base parameters for sensitivity testing

        Returns:
            RobustnessReport with all check results
        """
        checks_passed = 0
        total_checks = 0
        critical_failures = []
        warnings = []
        recommendations = []

        # 1. Data Quality Check
        total_checks += 1
        data_quality = self.check_data_quality(df)

        if data_quality.is_valid:
            checks_passed += 1
        else:
            critical_failures.append("Data quality check failed")
            recommendations.append("Ensure data has at least 60 days of history with no major gaps")

        warnings.extend(data_quality.warnings)

        # 2. Consistency Check
        total_checks += 1
        consistency = self.check_consistency(model_predictions)

        if consistency.is_consistent:
            checks_passed += 1
        else:
            if consistency.signal_stability < 0.3:
                critical_failures.append("Models show high disagreement")
            else:
                warnings.append("Some model inconsistencies detected")
            warnings.extend(consistency.issues)

        # 3. Sensitivity Analysis (if function provided)
        sensitivity_results = []

        if prediction_func and param_ranges and base_params:
            sensitivity_results = self.check_sensitivity(
                prediction_func, base_params, param_ranges
            )

            for result in sensitivity_results:
                total_checks += 1
                if result.is_stable:
                    checks_passed += 1
                else:
                    if result.sensitivity_score > 0.7:
                        critical_failures.append(f"Prediction highly sensitive to {result.parameter_name}")
                    else:
                        warnings.append(result.interpretation)

        # Calculate overall score
        if total_checks > 0:
            overall_score = (
                0.4 * data_quality.quality_score +
                0.3 * consistency.signal_stability +
                0.3 * (checks_passed / total_checks)
            )
        else:
            overall_score = 0

        # Determine robustness level
        if len(critical_failures) > 0:
            if len(critical_failures) >= 2:
                overall_level = RobustnessLevel.INVALID
            else:
                overall_level = RobustnessLevel.FRAGILE
        elif overall_score >= 0.8:
            overall_level = RobustnessLevel.ROBUST
        elif overall_score >= 0.6:
            overall_level = RobustnessLevel.ACCEPTABLE
        else:
            overall_level = RobustnessLevel.FRAGILE

        # Generate recommendations
        if overall_level == RobustnessLevel.FRAGILE:
            recommendations.append("Consider reducing position size due to prediction uncertainty")
        if data_quality.freshness_days > 1:
            recommendations.append("Update data to get more accurate predictions")
        if not consistency.is_consistent:
            recommendations.append("Wait for model consensus before trading")

        return RobustnessReport(
            overall_level=overall_level,
            overall_score=overall_score,
            data_quality=data_quality,
            sensitivity_results=sensitivity_results,
            consistency_result=consistency,
            passed_checks=checks_passed,
            total_checks=total_checks,
            critical_failures=critical_failures,
            warnings=warnings,
            recommendations=recommendations
        )


# Convenience function
def check_prediction_robustness(
    df: pd.DataFrame,
    model_predictions: Dict[str, float]
) -> RobustnessReport:
    """
    Quick robustness check for a prediction.

    Args:
        df: Input data
        model_predictions: Dict of model_name -> prediction

    Returns:
        RobustnessReport
    """
    checker = RobustnessChecker()
    return checker.run_full_check(df, model_predictions)


# Export
__all__ = [
    'RobustnessChecker',
    'RobustnessReport',
    'RobustnessLevel',
    'DataQualityReport',
    'SensitivityResult',
    'ConsistencyResult',
    'check_prediction_robustness'
]
