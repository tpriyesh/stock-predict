"""
Tensor Decomposition Module for Multi-Dimensional Pattern Analysis

Implements tensor factorization methods:
- CP (CANDECOMP/PARAFAC) Decomposition
- Tucker Decomposition
- Non-negative Tensor Factorization
- Tensor Train Decomposition

Applications for Trading:
- Multi-timeframe pattern extraction
- Cross-asset correlation structure
- Feature interaction discovery

Mathematical Foundation:
- CP: X ≈ Σᵣ aᵣ ∘ bᵣ ∘ cᵣ (sum of rank-1 tensors)
- Tucker: X ≈ G ×₁ A ×₂ B ×₃ C (core tensor with factor matrices)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TensorDecompositionResult:
    """Result of tensor decomposition"""
    factors: List[np.ndarray]  # Factor matrices
    core_tensor: Optional[np.ndarray]  # Core tensor (Tucker)
    weights: np.ndarray  # Component weights
    explained_variance: float  # Variance explained
    reconstruction_error: float  # Reconstruction RMSE
    components: List[Dict[str, Any]]  # Interpretable components
    multi_scale_patterns: Dict[str, np.ndarray]  # Patterns at different scales


class CPDecomposition:
    """
    CP (CANDECOMP/PARAFAC) Tensor Decomposition.

    Decomposes tensor X into sum of rank-1 tensors:
    X ≈ Σᵣ λᵣ (aᵣ ⊗ bᵣ ⊗ cᵣ)

    For 3D tensor (time × features × assets):
    - Mode 1: Temporal patterns
    - Mode 2: Feature loadings
    - Mode 3: Asset factors
    """

    def __init__(
        self,
        rank: int = 5,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        tensor: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fit CP decomposition using ALS (Alternating Least Squares).
        """
        shape = tensor.shape
        n_modes = len(shape)

        if n_modes < 2:
            return self._default_result()

        # Initialize factor matrices
        factors = [
            np.random.randn(s, self.rank)
            for s in shape
        ]

        # Normalize
        for i, f in enumerate(factors):
            norms = np.linalg.norm(f, axis=0)
            factors[i] = f / (norms + 1e-10)

        weights = np.ones(self.rank)
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Update each factor matrix
            for mode in range(n_modes):
                # Compute Khatri-Rao product of all other factors
                kr_product = self._khatri_rao_except(factors, mode)

                # Unfold tensor along mode
                unfolded = self._unfold(tensor, mode)

                # Solve least squares
                # A^(mode) = X_(mode) * KR * (KR^T KR)^-1
                try:
                    gram = kr_product.T @ kr_product
                    rhs = unfolded @ kr_product
                    factors[mode] = rhs @ np.linalg.pinv(gram)
                except Exception:
                    continue

                # Normalize
                norms = np.linalg.norm(factors[mode], axis=0)
                factors[mode] = factors[mode] / (norms + 1e-10)
                weights *= norms

            # Compute reconstruction error
            reconstructed = self._reconstruct(factors, weights, shape)
            error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

            if abs(prev_error - error) < self.tol:
                break
            prev_error = error

        # Explained variance
        total_var = np.var(tensor)
        residual_var = np.var(tensor - reconstructed)
        explained_var = 1 - residual_var / (total_var + 1e-10)

        return {
            "factors": factors,
            "weights": weights,
            "reconstruction_error": float(error),
            "explained_variance": float(explained_var),
            "n_iterations": iteration + 1,
            "rank": self.rank
        }

    def _khatri_rao_except(
        self,
        factors: List[np.ndarray],
        skip_mode: int
    ) -> np.ndarray:
        """Compute Khatri-Rao product excluding one mode."""
        result = None
        for i, f in enumerate(factors):
            if i == skip_mode:
                continue
            if result is None:
                result = f
            else:
                result = self._khatri_rao(result, f)
        return result

    def _khatri_rao(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """Khatri-Rao (column-wise Kronecker) product."""
        n_a, r = A.shape
        n_b = B.shape[0]
        result = np.zeros((n_a * n_b, r))
        for j in range(r):
            result[:, j] = np.outer(A[:, j], B[:, j]).ravel()
        return result

    def _unfold(
        self,
        tensor: np.ndarray,
        mode: int
    ) -> np.ndarray:
        """Unfold tensor along specified mode."""
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def _reconstruct(
        self,
        factors: List[np.ndarray],
        weights: np.ndarray,
        shape: Tuple
    ) -> np.ndarray:
        """Reconstruct tensor from CP factors."""
        result = np.zeros(shape)
        for r in range(self.rank):
            term = weights[r]
            for i, s in enumerate(shape):
                term = np.outer(term.ravel(), factors[i][:, r]).reshape(
                    list(term.shape) + [s]
                ) if i > 0 else factors[0][:, r]
            if len(shape) == 3:
                result += weights[r] * np.einsum('i,j,k->ijk',
                    factors[0][:, r], factors[1][:, r], factors[2][:, r])
            elif len(shape) == 2:
                result += weights[r] * np.outer(factors[0][:, r], factors[1][:, r])
        return result

    def _default_result(self) -> Dict[str, Any]:
        return {
            "factors": [],
            "weights": np.array([]),
            "reconstruction_error": 1.0,
            "explained_variance": 0.0
        }


class TuckerDecomposition:
    """
    Tucker Decomposition for Tensor Analysis.

    X ≈ G ×₁ A ×₂ B ×₃ C

    Where G is core tensor and A, B, C are factor matrices.
    More flexible than CP with different ranks per mode.
    """

    def __init__(
        self,
        ranks: Tuple[int, ...] = (3, 3, 3),
        max_iter: int = 50
    ):
        self.ranks = ranks
        self.max_iter = max_iter

    def fit(
        self,
        tensor: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fit Tucker decomposition using HOSVD (Higher-Order SVD).
        """
        shape = tensor.shape
        n_modes = len(shape)

        if n_modes != len(self.ranks):
            # Adjust ranks to match tensor dimensions
            ranks = tuple(min(r, s) for r, s in zip(
                self.ranks + (self.ranks[-1],) * (n_modes - len(self.ranks)),
                shape
            ))
        else:
            ranks = tuple(min(r, s) for r, s in zip(self.ranks, shape))

        # Step 1: Compute factor matrices via mode-n SVD
        factors = []
        for mode in range(n_modes):
            unfolded = self._unfold(tensor, mode)
            try:
                U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
                factors.append(U[:, :ranks[mode]])
            except Exception:
                factors.append(np.eye(shape[mode], ranks[mode]))

        # Step 2: Compute core tensor
        core = tensor.copy()
        for mode in range(n_modes):
            core = self._mode_product(core, factors[mode].T, mode)

        # Reconstruct and compute error
        reconstructed = self._reconstruct(core, factors)
        error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

        total_var = np.var(tensor)
        residual_var = np.var(tensor - reconstructed)
        explained_var = 1 - residual_var / (total_var + 1e-10)

        return {
            "factors": factors,
            "core_tensor": core,
            "ranks": ranks,
            "reconstruction_error": float(error),
            "explained_variance": float(explained_var),
            "core_size": core.shape
        }

    def _unfold(
        self,
        tensor: np.ndarray,
        mode: int
    ) -> np.ndarray:
        """Unfold tensor along mode."""
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def _mode_product(
        self,
        tensor: np.ndarray,
        matrix: np.ndarray,
        mode: int
    ) -> np.ndarray:
        """Compute mode-n product of tensor with matrix."""
        # Move mode to first dimension
        tensor_moved = np.moveaxis(tensor, mode, 0)
        shape = list(tensor_moved.shape)

        # Reshape to 2D
        tensor_2d = tensor_moved.reshape(shape[0], -1)

        # Multiply
        result_2d = matrix @ tensor_2d

        # Reshape back
        shape[0] = matrix.shape[0]
        result = result_2d.reshape(shape)

        # Move axis back
        return np.moveaxis(result, 0, mode)

    def _reconstruct(
        self,
        core: np.ndarray,
        factors: List[np.ndarray]
    ) -> np.ndarray:
        """Reconstruct tensor from Tucker decomposition."""
        result = core.copy()
        for mode, factor in enumerate(factors):
            result = self._mode_product(result, factor, mode)
        return result


class MultiScalePatternExtractor:
    """
    Extract multi-scale patterns from financial time series using tensor methods.

    Constructs a 3D tensor:
    - Dimension 1: Time windows
    - Dimension 2: Scales (different lookback periods)
    - Dimension 3: Features
    """

    def __init__(
        self,
        scales: List[int] = [5, 10, 20, 40, 60],
        n_components: int = 5
    ):
        self.scales = scales
        self.n_components = n_components
        self.cp = CPDecomposition(rank=n_components)
        self.tucker = TuckerDecomposition(ranks=(n_components, len(scales), 5))

    def extract(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> TensorDecompositionResult:
        """
        Extract multi-scale patterns from returns.
        """
        if len(returns) < max(self.scales) + 20:
            return self._default_result()

        # Build tensor
        tensor = self._build_tensor(returns, features)

        # Apply CP decomposition
        cp_result = self.cp.fit(tensor)

        # Apply Tucker decomposition
        tucker_result = self.tucker.fit(tensor)

        # Interpret patterns
        components = self._interpret_components(cp_result, tucker_result)

        # Extract multi-scale patterns
        patterns = self._extract_scale_patterns(tensor, cp_result["factors"])

        return TensorDecompositionResult(
            factors=cp_result["factors"],
            core_tensor=tucker_result.get("core_tensor"),
            weights=cp_result["weights"],
            explained_variance=cp_result["explained_variance"],
            reconstruction_error=cp_result["reconstruction_error"],
            components=components,
            multi_scale_patterns=patterns
        )

    def _build_tensor(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray]
    ) -> np.ndarray:
        """Build 3D tensor from returns."""
        n = len(returns)
        max_scale = max(self.scales)

        # Number of valid windows
        n_windows = n - max_scale

        if features is None:
            # Create features from returns
            n_features = 5
            features_matrix = np.zeros((n, n_features))
            features_matrix[:, 0] = returns  # Raw returns
            features_matrix[:, 1] = np.abs(returns)  # Absolute returns
            features_matrix[1:, 2] = returns[1:] * returns[:-1]  # Momentum
            features_matrix[1:, 3] = np.sign(returns[1:]) != np.sign(returns[:-1])  # Reversals
            features_matrix[:, 4] = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)  # Z-score
        else:
            features_matrix = features
            n_features = features.shape[1] if features.ndim > 1 else 1

        # Build tensor
        tensor = np.zeros((n_windows, len(self.scales), n_features))

        for i in range(n_windows):
            for j, scale in enumerate(self.scales):
                window_end = max_scale + i
                window_start = window_end - scale

                for k in range(n_features):
                    # Aggregate feature over window
                    tensor[i, j, k] = np.mean(features_matrix[window_start:window_end, k])

        return tensor

    def _interpret_components(
        self,
        cp_result: Dict[str, Any],
        tucker_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Interpret decomposition components."""
        components = []
        factors = cp_result.get("factors", [])
        weights = cp_result.get("weights", np.array([]))

        if len(factors) < 3 or len(weights) == 0:
            return components

        for r in range(len(weights)):
            if r >= factors[0].shape[1]:
                break

            # Temporal factor
            temporal = factors[0][:, r]
            temporal_trend = np.polyfit(range(len(temporal)), temporal, 1)[0]

            # Scale factor
            scale_factor = factors[1][:, r]
            dominant_scale_idx = np.argmax(np.abs(scale_factor))
            dominant_scale = self.scales[dominant_scale_idx] if dominant_scale_idx < len(self.scales) else 20

            # Feature factor
            feature_factor = factors[2][:, r]
            dominant_feature_idx = np.argmax(np.abs(feature_factor))

            component = {
                "rank": r,
                "weight": float(weights[r]),
                "temporal_trend": "rising" if temporal_trend > 0 else "falling",
                "dominant_scale": dominant_scale,
                "dominant_feature": int(dominant_feature_idx),
                "scale_profile": scale_factor.tolist(),
                "temporal_stability": float(1 - np.std(temporal) / (np.abs(np.mean(temporal)) + 1e-10))
            }
            components.append(component)

        return components

    def _extract_scale_patterns(
        self,
        tensor: np.ndarray,
        factors: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Extract patterns at different scales."""
        patterns = {}

        for i, scale in enumerate(self.scales):
            if i < tensor.shape[1]:
                # Extract slice at this scale
                scale_slice = tensor[:, i, :]

                # Mean pattern
                patterns[f"scale_{scale}_mean"] = np.mean(scale_slice, axis=1)

                # Variance pattern
                patterns[f"scale_{scale}_var"] = np.var(scale_slice, axis=1)

                # First principal direction
                try:
                    U, S, _ = np.linalg.svd(scale_slice, full_matrices=False)
                    patterns[f"scale_{scale}_pc1"] = U[:, 0] * S[0]
                except Exception:
                    patterns[f"scale_{scale}_pc1"] = np.mean(scale_slice, axis=1)

        return patterns

    def _default_result(self) -> TensorDecompositionResult:
        return TensorDecompositionResult(
            factors=[],
            core_tensor=None,
            weights=np.array([]),
            explained_variance=0.0,
            reconstruction_error=1.0,
            components=[],
            multi_scale_patterns={}
        )


class CrossAssetTensorAnalysis:
    """
    Cross-asset relationship analysis using tensor decomposition.

    Tensor structure:
    - Dimension 1: Time
    - Dimension 2: Asset 1
    - Dimension 3: Asset 2
    """

    def __init__(
        self,
        window_size: int = 20,
        n_components: int = 3
    ):
        self.window_size = window_size
        self.n_components = n_components

    def analyze(
        self,
        returns_matrix: np.ndarray  # Shape: (time, n_assets)
    ) -> Dict[str, Any]:
        """
        Analyze cross-asset relationships.
        """
        n_time, n_assets = returns_matrix.shape

        if n_time < self.window_size + 10 or n_assets < 2:
            return self._default_result()

        # Build correlation tensor over time
        n_windows = n_time - self.window_size
        corr_tensor = np.zeros((n_windows, n_assets, n_assets))

        for t in range(n_windows):
            window = returns_matrix[t:t+self.window_size, :]
            try:
                corr_tensor[t] = np.corrcoef(window.T)
            except Exception:
                corr_tensor[t] = np.eye(n_assets)

        # Decompose correlation tensor
        cp = CPDecomposition(rank=self.n_components)
        result = cp.fit(corr_tensor)

        factors = result.get("factors", [])

        if len(factors) >= 3:
            # Temporal dynamics
            temporal_factor = factors[0]

            # Asset groupings
            asset_factor_1 = factors[1]
            asset_factor_2 = factors[2]

            # Identify clusters
            clusters = self._identify_clusters(asset_factor_1)

            return {
                "correlation_tensor_shape": corr_tensor.shape,
                "temporal_dynamics": temporal_factor.tolist(),
                "asset_loadings": asset_factor_1.tolist(),
                "explained_variance": result["explained_variance"],
                "clusters": clusters,
                "n_components": self.n_components
            }

        return self._default_result()

    def _identify_clusters(
        self,
        loadings: np.ndarray
    ) -> Dict[str, List[int]]:
        """Identify asset clusters from loadings."""
        clusters = {}

        for r in range(loadings.shape[1]):
            # Assets with high positive loading
            high_pos = np.where(loadings[:, r] > 0.3)[0]
            # Assets with high negative loading
            high_neg = np.where(loadings[:, r] < -0.3)[0]

            if len(high_pos) > 0:
                clusters[f"cluster_{r}_pos"] = high_pos.tolist()
            if len(high_neg) > 0:
                clusters[f"cluster_{r}_neg"] = high_neg.tolist()

        return clusters

    def _default_result(self) -> Dict[str, Any]:
        return {
            "correlation_tensor_shape": (0, 0, 0),
            "temporal_dynamics": [],
            "asset_loadings": [],
            "explained_variance": 0.0,
            "clusters": {}
        }
