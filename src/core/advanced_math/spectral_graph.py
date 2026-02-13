"""
Spectral Graph Theory Module for Asset Network Analysis

Implements graph-theoretic methods:
- Laplacian Eigenmap Analysis
- Random Matrix Theory (Marchenko-Pastur)
- Graph Spectral Clustering
- Network Centrality Measures
- Community Detection

Mathematical Foundation:
- Graph Laplacian: L = D - A (degree matrix - adjacency matrix)
- Normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
- Fiedler vector: Second smallest eigenvector of L
- Marchenko-Pastur: Eigenvalue distribution of random correlation matrices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import eigsh
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpectralAnalysisResult:
    """Result of spectral graph analysis"""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fiedler_value: float  # Algebraic connectivity
    fiedler_vector: np.ndarray  # For bipartition
    n_communities: int  # Detected number of communities
    community_assignments: np.ndarray  # Community labels
    centrality_scores: Dict[str, np.ndarray]  # Various centrality measures
    spectral_gap: float  # Gap between eigenvalues
    random_matrix_signal: float  # Signal vs noise ratio
    market_mode_strength: float  # First eigenvalue strength


class LaplacianEigenmaps:
    """
    Laplacian Eigenmaps for correlation network analysis.

    Maps high-dimensional asset correlations to low-dimensional embedding
    preserving local structure.

    L = D - W (unnormalized)
    L_sym = D^(-1/2) L D^(-1/2) (symmetric normalized)
    """

    def __init__(
        self,
        n_components: int = 5,
        sigma: float = 1.0
    ):
        self.n_components = n_components
        self.sigma = sigma

    def fit_transform(
        self,
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Laplacian eigenmaps from correlation matrix.
        """
        n = correlation_matrix.shape[0]

        if n < 3:
            return self._default_result()

        # Convert correlation to distance/similarity
        # Higher correlation = higher similarity
        similarity = (correlation_matrix + 1) / 2  # Map [-1, 1] to [0, 1]

        # Apply Gaussian kernel
        W = np.exp(-(1 - similarity) ** 2 / (2 * self.sigma ** 2))
        np.fill_diagonal(W, 0)  # No self-loops

        # Degree matrix
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))

        # Normalized Laplacian
        L = D - W
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Compute eigendecomposition
        try:
            # Use smallest eigenvalues (skip the first which is 0)
            n_eig = min(self.n_components + 1, n)
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Embedding (skip first eigenvector)
            embedding = eigenvectors[:, 1:n_eig]

            # Fiedler value and vector
            fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0
            fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)

            # Spectral gap
            if len(eigenvalues) > 2:
                spectral_gap = eigenvalues[2] - eigenvalues[1]
            else:
                spectral_gap = 0

            return {
                "embedding": embedding,
                "eigenvalues": eigenvalues[:n_eig],
                "fiedler_value": float(fiedler_value),
                "fiedler_vector": fiedler_vector,
                "spectral_gap": float(spectral_gap),
                "algebraic_connectivity": float(fiedler_value)
            }

        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}")
            return self._default_result()

    def _default_result(self) -> Dict[str, Any]:
        return {
            "embedding": np.array([]),
            "eigenvalues": np.array([]),
            "fiedler_value": 0.0,
            "fiedler_vector": np.array([]),
            "spectral_gap": 0.0
        }


class RandomMatrixTheory:
    """
    Random Matrix Theory for correlation matrix analysis.

    Separates signal from noise in correlation matrices using
    Marchenko-Pastur distribution.

    For random matrix with T observations and N assets:
    λ_max = σ² (1 + √(N/T))²
    λ_min = σ² (1 - √(N/T))²
    """

    def __init__(self):
        pass

    def analyze(
        self,
        returns_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze correlation matrix using RMT.
        """
        T, N = returns_matrix.shape

        if T < N or N < 2:
            return self._default_result()

        # Compute correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = eigenvalues[::-1]  # Descending
        eigenvectors = eigenvectors[:, ::-1]

        # Marchenko-Pastur bounds
        q = N / T
        sigma = 1.0

        lambda_plus = sigma ** 2 * (1 + np.sqrt(q)) ** 2
        lambda_minus = sigma ** 2 * (1 - np.sqrt(q)) ** 2

        # Identify significant eigenvalues (above MP bound)
        significant_mask = eigenvalues > lambda_plus
        n_significant = np.sum(significant_mask)

        # Signal strength (first eigenvalue)
        market_mode_strength = eigenvalues[0] / np.sum(eigenvalues)

        # Noise ratio
        noise_eigenvalues = eigenvalues[~significant_mask]
        if len(noise_eigenvalues) > 0:
            noise_ratio = np.sum(noise_eigenvalues) / np.sum(eigenvalues)
        else:
            noise_ratio = 0

        # Effective number of factors
        effective_rank = np.sum(eigenvalues) ** 2 / (np.sum(eigenvalues ** 2) + 1e-10)

        # Denoised correlation matrix
        denoised = self._denoise_correlation(
            eigenvalues, eigenvectors, lambda_plus
        )

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "mp_upper_bound": float(lambda_plus),
            "mp_lower_bound": float(lambda_minus),
            "n_significant_factors": int(n_significant),
            "market_mode_strength": float(market_mode_strength),
            "noise_ratio": float(noise_ratio),
            "effective_rank": float(effective_rank),
            "denoised_correlation": denoised,
            "signal_to_noise": float(1 - noise_ratio) / (noise_ratio + 1e-10)
        }

    def _denoise_correlation(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Denoise correlation matrix by zeroing noise eigenvalues."""
        # Keep only significant eigenvalues
        denoised_eigenvalues = eigenvalues.copy()
        noise_mask = eigenvalues < threshold

        # Replace noise eigenvalues with their average
        if np.sum(noise_mask) > 0:
            avg_noise = np.mean(eigenvalues[noise_mask])
            denoised_eigenvalues[noise_mask] = avg_noise

        # Reconstruct
        denoised = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T

        # Normalize diagonal
        d = np.sqrt(np.diag(denoised))
        denoised = denoised / np.outer(d, d)

        return denoised

    def _default_result(self) -> Dict[str, Any]:
        return {
            "eigenvalues": np.array([]),
            "n_significant_factors": 0,
            "market_mode_strength": 0.0,
            "signal_to_noise": 0.0
        }


class SpectralClustering:
    """
    Spectral clustering for asset grouping.

    Uses eigenvectors of Laplacian to embed data,
    then applies k-means for clustering.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_components: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.n_components = n_components or n_clusters

    def fit(
        self,
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform spectral clustering on correlation matrix.
        """
        n = correlation_matrix.shape[0]

        if n < self.n_clusters:
            return self._default_result(n)

        # Get Laplacian eigenvectors
        laplacian = LaplacianEigenmaps(n_components=self.n_components)
        lap_result = laplacian.fit_transform(correlation_matrix)

        embedding = lap_result.get("embedding", np.array([]))

        if embedding.size == 0:
            return self._default_result(n)

        # K-means clustering on embedding
        labels = self._kmeans(embedding, self.n_clusters)

        # Compute cluster quality
        silhouette = self._silhouette_score(embedding, labels)

        # Cluster statistics
        cluster_stats = self._cluster_statistics(correlation_matrix, labels)

        return {
            "labels": labels,
            "n_clusters": self.n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_sizes": [int(np.sum(labels == k)) for k in range(self.n_clusters)],
            "cluster_statistics": cluster_stats,
            "embedding": embedding
        }

    def _kmeans(
        self,
        X: np.ndarray,
        k: int,
        max_iter: int = 100
    ) -> np.ndarray:
        """Simple k-means implementation."""
        n = X.shape[0]
        if n < k:
            return np.arange(n)

        # Initialize centroids randomly
        idx = np.random.choice(n, k, replace=False)
        centroids = X[idx].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign labels
            for i in range(n):
                distances = [np.linalg.norm(X[i] - c) for c in centroids]
                labels[i] = np.argmin(distances)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    new_centroids[j] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels

    def _silhouette_score(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute silhouette score."""
        n = len(labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return 0.0

        silhouettes = np.zeros(n)

        for i in range(n):
            # Same cluster
            same_cluster = labels == labels[i]
            if np.sum(same_cluster) > 1:
                a = np.mean([np.linalg.norm(X[i] - X[j])
                            for j in np.where(same_cluster)[0] if j != i])
            else:
                a = 0

            # Other clusters
            b = float('inf')
            for label in unique_labels:
                if label != labels[i]:
                    other_cluster = labels == label
                    if np.sum(other_cluster) > 0:
                        mean_dist = np.mean([np.linalg.norm(X[i] - X[j])
                                            for j in np.where(other_cluster)[0]])
                        b = min(b, mean_dist)

            if b == float('inf'):
                b = 0

            if max(a, b) > 0:
                silhouettes[i] = (b - a) / max(a, b)

        return float(np.mean(silhouettes))

    def _cluster_statistics(
        self,
        corr_matrix: np.ndarray,
        labels: np.ndarray
    ) -> List[Dict[str, float]]:
        """Compute statistics for each cluster."""
        stats = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]

            if len(indices) < 2:
                stats.append({
                    "label": int(label),
                    "size": 1,
                    "mean_internal_corr": 1.0,
                    "mean_external_corr": 0.0
                })
                continue

            # Internal correlation
            internal_corrs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    internal_corrs.append(corr_matrix[indices[i], indices[j]])

            # External correlation
            external_corrs = []
            external_indices = np.where(~mask)[0]
            for i in indices:
                for j in external_indices:
                    external_corrs.append(corr_matrix[i, j])

            stats.append({
                "label": int(label),
                "size": int(len(indices)),
                "mean_internal_corr": float(np.mean(internal_corrs)) if internal_corrs else 0.0,
                "mean_external_corr": float(np.mean(external_corrs)) if external_corrs else 0.0
            })

        return stats

    def _default_result(self, n: int) -> Dict[str, Any]:
        return {
            "labels": np.arange(n),
            "n_clusters": n,
            "silhouette_score": 0.0,
            "cluster_sizes": [1] * n
        }


class NetworkCentrality:
    """
    Network centrality measures for identifying key assets.
    """

    def __init__(self):
        pass

    def compute(
        self,
        correlation_matrix: np.ndarray,
        threshold: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Compute various centrality measures.
        """
        n = correlation_matrix.shape[0]

        if n < 2:
            return self._default_result(n)

        # Create adjacency matrix (threshold correlations)
        adjacency = (np.abs(correlation_matrix) > threshold).astype(float)
        np.fill_diagonal(adjacency, 0)

        # Weighted adjacency
        weighted = correlation_matrix.copy()
        np.fill_diagonal(weighted, 0)
        weighted = np.abs(weighted)

        # Degree centrality
        degree = np.sum(adjacency, axis=1) / (n - 1)

        # Weighted degree (strength)
        strength = np.sum(weighted, axis=1)

        # Eigenvector centrality
        try:
            eigenvalues, eigenvectors = np.linalg.eig(weighted)
            max_idx = np.argmax(np.real(eigenvalues))
            eigenvector_centrality = np.abs(np.real(eigenvectors[:, max_idx]))
            eigenvector_centrality = eigenvector_centrality / (eigenvector_centrality.sum() + 1e-10)
        except Exception:
            eigenvector_centrality = degree.copy()

        # Betweenness centrality (simplified)
        betweenness = self._approximate_betweenness(adjacency)

        # Closeness centrality
        closeness = self._closeness_centrality(adjacency)

        # PageRank
        pagerank = self._pagerank(weighted)

        return {
            "degree": degree,
            "strength": strength,
            "eigenvector": eigenvector_centrality,
            "betweenness": betweenness,
            "closeness": closeness,
            "pagerank": pagerank,
            "most_central_degree": int(np.argmax(degree)),
            "most_central_eigenvector": int(np.argmax(eigenvector_centrality)),
            "most_central_pagerank": int(np.argmax(pagerank))
        }

    def _approximate_betweenness(
        self,
        adjacency: np.ndarray
    ) -> np.ndarray:
        """Approximate betweenness centrality."""
        n = adjacency.shape[0]
        betweenness = np.zeros(n)

        # Sample some source-target pairs
        n_samples = min(n * 10, 1000)

        for _ in range(n_samples):
            source = np.random.randint(n)
            target = np.random.randint(n)
            if source != target:
                # BFS to find shortest path
                path = self._bfs_path(adjacency, source, target)
                for node in path[1:-1]:  # Exclude source and target
                    betweenness[node] += 1

        return betweenness / (n_samples + 1e-10)

    def _bfs_path(
        self,
        adjacency: np.ndarray,
        source: int,
        target: int
    ) -> List[int]:
        """Find shortest path using BFS."""
        n = adjacency.shape[0]
        visited = [False] * n
        queue = [(source, [source])]
        visited[source] = True

        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path

            for neighbor in range(n):
                if adjacency[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, path + [neighbor]))

        return []

    def _closeness_centrality(
        self,
        adjacency: np.ndarray
    ) -> np.ndarray:
        """Compute closeness centrality."""
        n = adjacency.shape[0]
        closeness = np.zeros(n)

        for i in range(n):
            # BFS for distances
            distances = self._bfs_distances(adjacency, i)
            total_distance = np.sum(distances[distances > 0])
            reachable = np.sum(distances > 0)

            if reachable > 0 and total_distance > 0:
                closeness[i] = reachable / total_distance

        return closeness

    def _bfs_distances(
        self,
        adjacency: np.ndarray,
        source: int
    ) -> np.ndarray:
        """Compute distances from source using BFS."""
        n = adjacency.shape[0]
        distances = np.full(n, -1)
        distances[source] = 0

        queue = [source]
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if adjacency[node, neighbor] > 0 and distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        return distances

    def _pagerank(
        self,
        weighted: np.ndarray,
        damping: float = 0.85,
        max_iter: int = 100
    ) -> np.ndarray:
        """Compute PageRank."""
        n = weighted.shape[0]

        # Normalize to stochastic matrix
        row_sums = np.sum(weighted, axis=1)
        transition = weighted / (row_sums[:, np.newaxis] + 1e-10)

        # Initialize
        pagerank = np.ones(n) / n

        for _ in range(max_iter):
            new_pr = (1 - damping) / n + damping * (transition.T @ pagerank)
            if np.allclose(pagerank, new_pr):
                break
            pagerank = new_pr

        return pagerank

    def _default_result(self, n: int) -> Dict[str, np.ndarray]:
        return {
            "degree": np.ones(n) / n,
            "strength": np.ones(n),
            "eigenvector": np.ones(n) / n,
            "betweenness": np.zeros(n),
            "closeness": np.ones(n),
            "pagerank": np.ones(n) / n
        }


class SpectralGraphEngine:
    """
    Unified Spectral Graph Engine for asset network analysis.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_components: int = 5
    ):
        self.laplacian = LaplacianEigenmaps(n_components=n_components)
        self.rmt = RandomMatrixTheory()
        self.clustering = SpectralClustering(n_clusters=n_clusters)
        self.centrality = NetworkCentrality()

    def analyze(
        self,
        returns_matrix: np.ndarray
    ) -> SpectralAnalysisResult:
        """
        Perform comprehensive spectral analysis.
        """
        if returns_matrix.shape[0] < 30 or returns_matrix.shape[1] < 3:
            return self._default_result(returns_matrix.shape[1])

        # Correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)

        # Random Matrix Theory analysis
        rmt_result = self.rmt.analyze(returns_matrix)

        # Laplacian eigenmaps
        lap_result = self.laplacian.fit_transform(corr_matrix)

        # Spectral clustering
        cluster_result = self.clustering.fit(corr_matrix)

        # Centrality measures
        centrality_result = self.centrality.compute(corr_matrix)

        # Extract key metrics
        eigenvalues = rmt_result.get("eigenvalues", np.array([]))
        eigenvectors = rmt_result.get("eigenvectors", np.array([]))

        fiedler_value = lap_result.get("fiedler_value", 0.0)
        fiedler_vector = lap_result.get("fiedler_vector", np.array([]))

        spectral_gap = lap_result.get("spectral_gap", 0.0)
        market_mode = rmt_result.get("market_mode_strength", 0.0)
        signal_noise = rmt_result.get("signal_to_noise", 0.0)

        return SpectralAnalysisResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            fiedler_value=fiedler_value,
            fiedler_vector=fiedler_vector,
            n_communities=cluster_result.get("n_clusters", 1),
            community_assignments=cluster_result.get("labels", np.array([])),
            centrality_scores=centrality_result,
            spectral_gap=spectral_gap,
            random_matrix_signal=signal_noise,
            market_mode_strength=market_mode
        )

    def _default_result(self, n: int) -> SpectralAnalysisResult:
        return SpectralAnalysisResult(
            eigenvalues=np.array([]),
            eigenvectors=np.array([]),
            fiedler_value=0.0,
            fiedler_vector=np.zeros(n),
            n_communities=1,
            community_assignments=np.zeros(n, dtype=int),
            centrality_scores={},
            spectral_gap=0.0,
            random_matrix_signal=0.0,
            market_mode_strength=0.0
        )
