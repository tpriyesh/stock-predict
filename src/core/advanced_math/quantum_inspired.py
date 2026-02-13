"""
Quantum-Inspired Optimization Module for Stock Prediction

Implements quantum-inspired algorithms for portfolio optimization and signal detection:
- Quantum Annealing Simulation (Simulated Quantum Annealing)
- QAOA-inspired Variational Optimization
- Quantum Tunneling for Local Minima Escape
- Grover-inspired Amplitude Amplification for Signal Detection

These are classical algorithms inspired by quantum mechanics principles,
running efficiently on classical hardware while capturing quantum-like behavior.

Mathematical Foundation:
- Energy landscape: E(s) = -Σᵢⱼ Jᵢⱼsᵢsⱼ - Σᵢ hᵢsᵢ (Ising model)
- Quantum tunneling: P(tunnel) ∝ exp(-2∫√(2m(V(x)-E))/ℏ dx)
- Adiabatic evolution: H(t) = (1-t/T)H₀ + (t/T)H₁
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuantumState(Enum):
    """Quantum-inspired state representations"""
    SUPERPOSITION = "superposition"  # Multiple states simultaneously
    COLLAPSED = "collapsed"  # Measured/decided state
    ENTANGLED = "entangled"  # Correlated with other assets


@dataclass
class QuantumPrediction:
    """Result of quantum-inspired prediction"""
    signal_strength: float  # -1 to 1 (sell to buy)
    confidence: float  # 0 to 1
    tunneling_probability: float  # Probability of regime change
    energy_landscape: Dict[str, float]  # Current energy state
    superposition_amplitudes: Dict[str, float]  # Probability amplitudes
    optimal_position: float  # Optimal position size from optimization
    coherence_time: int  # Expected stability (days)
    entanglement_scores: Dict[str, float]  # Correlation strengths


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimization for stock prediction and portfolio management.

    Uses principles from:
    1. Quantum Annealing - Find global optima in rugged landscapes
    2. QAOA - Variational optimization for combinatorial problems
    3. Quantum Tunneling - Escape local minima
    4. Amplitude Amplification - Boost weak signals
    """

    def __init__(
        self,
        n_qubits: int = 8,
        annealing_steps: int = 1000,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        tunneling_strength: float = 1.0,
        coherence_threshold: float = 0.1
    ):
        self.n_qubits = n_qubits
        self.annealing_steps = annealing_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.tunneling_strength = tunneling_strength
        self.coherence_threshold = coherence_threshold

    def analyze(
        self,
        returns: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None,
        correlations: Optional[np.ndarray] = None
    ) -> QuantumPrediction:
        """
        Perform quantum-inspired analysis on price data.

        Args:
            returns: Array of log returns
            features: Optional dictionary of additional features
            correlations: Optional correlation matrix with other assets

        Returns:
            QuantumPrediction with comprehensive analysis
        """
        if len(returns) < 20:
            return self._default_prediction()

        # Step 1: Build Ising Hamiltonian from return dynamics
        hamiltonian = self._build_hamiltonian(returns)

        # Step 2: Perform quantum annealing simulation
        ground_state, energy_history = self._quantum_anneal(hamiltonian)

        # Step 3: Calculate tunneling probability (regime change likelihood)
        tunneling_prob = self._calculate_tunneling_probability(returns, hamiltonian)

        # Step 4: Compute superposition amplitudes (probability distribution over states)
        amplitudes = self._compute_superposition_amplitudes(returns)

        # Step 5: Apply amplitude amplification to weak signals
        amplified_signal = self._amplitude_amplification(returns, amplitudes)

        # Step 6: Calculate optimal position using QAOA-inspired optimization
        optimal_position = self._qaoa_optimize_position(returns, hamiltonian)

        # Step 7: Estimate coherence time (stability duration)
        coherence_time = self._estimate_coherence_time(returns)

        # Step 8: Compute entanglement with other assets (if correlations provided)
        entanglement_scores = self._compute_entanglement(correlations)

        # Step 9: Generate energy landscape analysis
        energy_landscape = self._analyze_energy_landscape(hamiltonian, ground_state)

        # Compute confidence based on multiple quantum-inspired metrics
        confidence = self._compute_quantum_confidence(
            tunneling_prob, amplitudes, coherence_time, energy_history
        )

        return QuantumPrediction(
            signal_strength=amplified_signal,
            confidence=confidence,
            tunneling_probability=tunneling_prob,
            energy_landscape=energy_landscape,
            superposition_amplitudes=amplitudes,
            optimal_position=optimal_position,
            coherence_time=coherence_time,
            entanglement_scores=entanglement_scores
        )

    def _build_hamiltonian(self, returns: np.ndarray) -> np.ndarray:
        """
        Build Ising Hamiltonian from return dynamics.

        The Hamiltonian encodes:
        - Momentum (trend continuation preference)
        - Mean reversion (equilibrium attraction)
        - Volatility clustering (energy states)

        H = -Σᵢⱼ Jᵢⱼσᵢσⱼ - Σᵢ hᵢσᵢ

        Where:
        - Jᵢⱼ: Coupling strength between time steps (autocorrelation)
        - hᵢ: External field (trend bias)
        """
        n = min(len(returns), self.n_qubits * 8)

        # Discretize returns into n_qubits binary variables
        quantized = np.sign(returns[-n:])

        # Build coupling matrix J (autocorrelation structure)
        J = np.zeros((self.n_qubits, self.n_qubits))
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                # Coupling based on lagged autocorrelation
                lag = j - i
                if lag < len(returns):
                    autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        J[i, j] = autocorr * self.tunneling_strength
                        J[j, i] = J[i, j]  # Symmetric

        # External field h (trend bias)
        h = np.zeros(self.n_qubits)
        window = len(returns) // self.n_qubits
        for i in range(self.n_qubits):
            start = i * window
            end = min((i + 1) * window, len(returns))
            if start < len(returns):
                h[i] = np.mean(returns[start:end]) * 100  # Scale for numerical stability

        # Full Hamiltonian as matrix
        H = np.zeros((self.n_qubits, self.n_qubits))
        H = J.copy()
        np.fill_diagonal(H, h)

        return H

    def _quantum_anneal(
        self,
        hamiltonian: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Simulated quantum annealing to find ground state.

        Follows adiabatic evolution:
        H(t) = (1-s)H₀ + sH₁

        Where:
        - H₀: Transverse field (quantum fluctuations)
        - H₁: Problem Hamiltonian
        - s: Annealing schedule s(t) = t/T

        Uses Metropolis-Hastings with quantum tunneling enhancement.
        """
        n = hamiltonian.shape[0]

        # Initialize in random state (superposition-like)
        state = np.random.choice([-1, 1], size=n).astype(float)

        # Track energy history
        energy_history = []
        best_state = state.copy()
        best_energy = self._compute_ising_energy(state, hamiltonian)

        # Annealing schedule
        temperatures = np.linspace(
            self.initial_temperature,
            self.final_temperature,
            self.annealing_steps
        )

        for step, T in enumerate(temperatures):
            # Annealing parameter s
            s = step / self.annealing_steps

            # Transverse field strength (quantum fluctuations)
            gamma = (1 - s) * self.tunneling_strength

            # Pick random spin to flip
            flip_idx = np.random.randint(n)

            # Calculate energy change from flip
            proposed_state = state.copy()
            proposed_state[flip_idx] *= -1

            current_energy = self._compute_ising_energy(state, hamiltonian)
            proposed_energy = self._compute_ising_energy(proposed_state, hamiltonian)
            delta_E = proposed_energy - current_energy

            # Quantum tunneling probability enhancement
            # Allows escaping local minima more easily than classical SA
            tunneling_factor = np.exp(-gamma * abs(delta_E) / (T + 1e-10))

            # Acceptance probability (Metropolis with tunneling)
            if delta_E < 0:
                accept_prob = 1.0
            else:
                classical_prob = np.exp(-delta_E / (T + 1e-10))
                accept_prob = min(1.0, classical_prob + gamma * tunneling_factor)

            # Accept or reject
            if np.random.random() < accept_prob:
                state = proposed_state
                current_energy = proposed_energy

            # Track best state
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()

            energy_history.append(current_energy)

        return best_state, energy_history

    def _compute_ising_energy(
        self,
        state: np.ndarray,
        hamiltonian: np.ndarray
    ) -> float:
        """Compute Ising model energy: E = -sᵀHs"""
        return -float(state @ hamiltonian @ state)

    def _calculate_tunneling_probability(
        self,
        returns: np.ndarray,
        hamiltonian: np.ndarray
    ) -> float:
        """
        Calculate probability of quantum tunneling (regime change).

        Based on WKB approximation:
        P ∝ exp(-2∫√(2m(V(x)-E))/ℏ dx)

        For financial markets, this represents the probability of
        breaking out of current regime into a new one.
        """
        # Estimate current "potential well" (volatility regime)
        recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        longer_vol = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)

        # Barrier height proportional to volatility difference
        barrier = abs(recent_vol - longer_vol) / (longer_vol + 1e-10)

        # Effective "mass" (momentum/trend strength)
        if len(returns) >= 10:
            trend = abs(np.mean(returns[-10:]) / (np.std(returns[-10:]) + 1e-10))
        else:
            trend = 0.5

        # WKB tunneling probability
        # Higher barrier and higher mass = lower tunneling probability
        effective_action = barrier * (1 + trend) / self.tunneling_strength
        tunneling_prob = np.exp(-effective_action)

        # Normalize to [0, 1]
        return float(np.clip(tunneling_prob, 0, 1))

    def _compute_superposition_amplitudes(
        self,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute probability amplitudes for different market states.

        In quantum mechanics: |ψ⟩ = Σᵢ αᵢ|i⟩ where Σ|αᵢ|² = 1

        Market states:
        - STRONG_BULL: Strong uptrend with momentum
        - BULL: Moderate uptrend
        - NEUTRAL: Sideways/unclear
        - BEAR: Moderate downtrend
        - STRONG_BEAR: Strong downtrend with momentum
        """
        if len(returns) < 20:
            return {
                "strong_bull": 0.1,
                "bull": 0.2,
                "neutral": 0.4,
                "bear": 0.2,
                "strong_bear": 0.1
            }

        # Calculate state indicators
        recent_return = np.sum(returns[-5:])
        medium_return = np.sum(returns[-20:])
        momentum = recent_return - (medium_return - recent_return)
        vol = np.std(returns[-20:])

        # Normalize to get directional bias
        z_score = recent_return / (vol * np.sqrt(5) + 1e-10)
        momentum_z = momentum / (vol * np.sqrt(5) + 1e-10)

        # Compute raw amplitudes using wave function analogy
        # ψ(x) = exp(-x²/2σ²) * e^(ikx) where k is momentum
        def wave_amplitude(center: float) -> float:
            position_term = np.exp(-(z_score - center) ** 2 / 2)
            momentum_term = np.exp(1j * momentum_z * center)
            return float(abs(position_term * momentum_term))

        raw_amplitudes = {
            "strong_bull": wave_amplitude(2.0),
            "bull": wave_amplitude(1.0),
            "neutral": wave_amplitude(0.0),
            "bear": wave_amplitude(-1.0),
            "strong_bear": wave_amplitude(-2.0)
        }

        # Normalize so probabilities sum to 1
        total = sum(raw_amplitudes.values())
        if total > 0:
            amplitudes = {k: v/total for k, v in raw_amplitudes.items()}
        else:
            amplitudes = {k: 0.2 for k in raw_amplitudes}

        return amplitudes

    def _amplitude_amplification(
        self,
        returns: np.ndarray,
        amplitudes: Dict[str, float]
    ) -> float:
        """
        Grover-inspired amplitude amplification for weak signals.

        In Grover's algorithm, target state amplitude increases as:
        sin((2k+1)θ) where sin(θ) = √(M/N)

        For trading signals, we amplify directional signals that might
        otherwise be too weak to act on with confidence.
        """
        if len(returns) < 10:
            return 0.0

        # Calculate raw signal
        raw_signal = np.sum(returns[-5:]) / (np.std(returns[-20:]) * np.sqrt(5) + 1e-10)

        # Determine if signal is "marked" (worth amplifying)
        # Based on: consistency, momentum, amplitude alignment
        recent_consistency = np.mean(np.sign(returns[-5:])) if len(returns) >= 5 else 0
        amplitude_alignment = amplitudes.get("bull", 0) - amplitudes.get("bear", 0)

        # Compute "oracle" that marks good signals
        # (1 if signal should be amplified, -1 otherwise)
        signal_quality = abs(raw_signal) * (1 + abs(recent_consistency))
        is_marked = signal_quality > 0.5

        if is_marked:
            # Apply amplitude amplification
            # Number of iterations (optimal is π/(4θ) - 1/2)
            theta = np.arcsin(min(0.9, signal_quality / 3))
            optimal_iterations = max(1, int(np.pi / (4 * theta + 1e-10) - 0.5))

            # Amplified signal strength
            amplification = np.sin((2 * min(optimal_iterations, 3) + 1) * theta)
            amplified = raw_signal * (1 + amplification)
        else:
            # Don't amplify weak/unreliable signals
            amplified = raw_signal * 0.5

        # Normalize to [-1, 1]
        return float(np.clip(amplified / 3, -1, 1))

    def _qaoa_optimize_position(
        self,
        returns: np.ndarray,
        hamiltonian: np.ndarray
    ) -> float:
        """
        QAOA-inspired position sizing optimization.

        QAOA alternates between:
        - Cost layer: e^(-iγC) (problem Hamiltonian)
        - Mixer layer: e^(-iβB) (exploration)

        We simulate this to find optimal position size.
        Uses real-valued approximation for classical simulation.
        """
        n_layers = 3  # QAOA depth

        # Initialize variational parameters
        gammas = np.random.uniform(0, np.pi, n_layers)
        betas = np.random.uniform(0, np.pi/2, n_layers)

        # Position options to evaluate
        positions = np.linspace(-1, 1, 21)  # -100% to +100%

        best_position = 0.0
        best_expectation = float('inf')

        for _ in range(50):  # Variational optimization iterations
            # Compute expectation value for each position
            expectations = []
            for pos in positions:
                # State preparation (real-valued amplitude)
                state = np.ones(self.n_qubits) * pos

                # Apply QAOA layers (using real approximation)
                for layer in range(n_layers):
                    # Cost layer: bias toward profitable positions
                    expected_return = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                    cost = -(pos * expected_return * 100 - abs(pos) * np.var(returns[-20:]) * 50)

                    # Real-valued approximation of phase rotation
                    # cos(-γC) ≈ 1 - (γC)²/2 for small angles
                    phase_factor = np.cos(gammas[layer] * cost)
                    state = state * phase_factor

                    # Mixer layer: encourage exploration
                    state = state * np.cos(betas[layer]) + np.sin(betas[layer])

                # Expectation value (risk-adjusted return)
                vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
                sharpe_proxy = (pos * np.mean(returns[-10:] if len(returns) >= 10 else returns)) / (abs(pos) * vol + 1e-10)
                drawdown_risk = abs(pos) * np.min(np.cumsum(returns[-20:])) if len(returns) >= 20 else 0

                expectation = -sharpe_proxy + 0.5 * drawdown_risk
                expectations.append(expectation)

            # Find best position
            min_idx = np.argmin(expectations)
            if expectations[min_idx] < best_expectation:
                best_expectation = expectations[min_idx]
                best_position = positions[min_idx]

            # Update variational parameters (gradient-free)
            gammas += np.random.normal(0, 0.1, n_layers)
            betas += np.random.normal(0, 0.1, n_layers)
            gammas = np.clip(gammas, 0, np.pi)
            betas = np.clip(betas, 0, np.pi/2)

        return float(best_position)

    def _estimate_coherence_time(self, returns: np.ndarray) -> int:
        """
        Estimate coherence time (signal stability duration).

        In quantum systems, coherence time T₂ is how long superposition lasts.
        For markets, it's how long we expect current signal to remain valid.

        Based on:
        - Autocorrelation decay
        - Volatility clustering persistence
        - Regime stability
        """
        if len(returns) < 30:
            return 5  # Default 5 days

        # Calculate autocorrelation decay
        max_lag = min(20, len(returns) // 3)
        autocorrs = []
        for lag in range(1, max_lag + 1):
            corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(abs(corr))
            else:
                autocorrs.append(0)

        # Find coherence time (where autocorrelation drops below threshold)
        coherence_time = 1
        for i, ac in enumerate(autocorrs):
            if ac > self.coherence_threshold:
                coherence_time = i + 1
            else:
                break

        # Adjust based on volatility clustering
        vol_changes = np.abs(np.diff(np.abs(returns)))
        vol_persistence = 1 - np.mean(vol_changes) / (np.std(returns) + 1e-10)
        coherence_time = int(coherence_time * (1 + vol_persistence))

        return max(1, min(coherence_time, 20))

    def _compute_entanglement(
        self,
        correlations: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute entanglement scores (correlation strength with other assets).

        In quantum mechanics, entangled particles are correlated.
        For assets, strong correlations create "entangled" behavior.
        """
        if correlations is None or len(correlations) == 0:
            return {}

        entanglement = {}

        # Assuming correlations is with market indices/sectors
        labels = ["NIFTY", "BANK_NIFTY", "SECTOR", "PEER_1", "PEER_2"]

        for i, corr in enumerate(correlations[:5] if len(correlations) >= 5 else correlations):
            if i < len(labels):
                # Entanglement strength based on squared correlation
                # (like quantum entanglement entropy)
                if isinstance(corr, (int, float)) and not np.isnan(corr):
                    entanglement[labels[i]] = float(corr ** 2)

        return entanglement

    def _analyze_energy_landscape(
        self,
        hamiltonian: np.ndarray,
        ground_state: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze the energy landscape around current state.

        Provides:
        - Ground state energy (optimal configuration)
        - Barrier heights (resistance levels)
        - Local minima count (stability)
        """
        # Ground state energy
        ground_energy = self._compute_ising_energy(ground_state, hamiltonian)

        # Sample nearby states to understand landscape
        n_samples = 100
        energies = []
        for _ in range(n_samples):
            # Flip 1-2 spins
            sample = ground_state.copy()
            n_flips = np.random.randint(1, 3)
            flip_indices = np.random.choice(len(ground_state), n_flips, replace=False)
            sample[flip_indices] *= -1
            energies.append(self._compute_ising_energy(sample, hamiltonian))

        energies = np.array(energies)

        return {
            "ground_energy": float(ground_energy),
            "mean_barrier": float(np.mean(energies) - ground_energy),
            "max_barrier": float(np.max(energies) - ground_energy),
            "landscape_roughness": float(np.std(energies)),
            "stability_score": float(1 / (1 + np.std(energies)))
        }

    def _compute_quantum_confidence(
        self,
        tunneling_prob: float,
        amplitudes: Dict[str, float],
        coherence_time: int,
        energy_history: List[float]
    ) -> float:
        """
        Compute overall confidence from quantum-inspired metrics.
        """
        # Higher confidence if:
        # 1. Low tunneling probability (stable regime)
        regime_stability = 1 - tunneling_prob

        # 2. Clear amplitude dominance (not in superposition)
        amplitude_values = list(amplitudes.values())
        entropy = -sum(a * np.log(a + 1e-10) for a in amplitude_values)
        max_entropy = np.log(len(amplitude_values))
        amplitude_clarity = 1 - entropy / max_entropy

        # 3. Long coherence time
        coherence_score = min(1.0, coherence_time / 10)

        # 4. Smooth energy convergence
        if len(energy_history) > 10:
            convergence = 1 - np.std(energy_history[-10:]) / (np.std(energy_history) + 1e-10)
        else:
            convergence = 0.5

        # Weighted combination
        confidence = (
            0.30 * regime_stability +
            0.25 * amplitude_clarity +
            0.25 * coherence_score +
            0.20 * convergence
        )

        return float(np.clip(confidence, 0, 1))

    def _default_prediction(self) -> QuantumPrediction:
        """Return default prediction for insufficient data."""
        return QuantumPrediction(
            signal_strength=0.0,
            confidence=0.0,
            tunneling_probability=0.5,
            energy_landscape={
                "ground_energy": 0.0,
                "mean_barrier": 0.0,
                "max_barrier": 0.0,
                "landscape_roughness": 1.0,
                "stability_score": 0.5
            },
            superposition_amplitudes={
                "strong_bull": 0.1,
                "bull": 0.2,
                "neutral": 0.4,
                "bear": 0.2,
                "strong_bear": 0.1
            },
            optimal_position=0.0,
            coherence_time=5,
            entanglement_scores={}
        )


class QuantumPortfolioOptimizer:
    """
    Quantum-inspired portfolio optimization using QUBO formulation.

    Solves: min xᵀQx + cᵀx
    Where:
    - x: Binary/discrete asset allocations
    - Q: Quadratic cost matrix (covariance)
    - c: Linear cost vector (negative expected returns)
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        min_position: float = 0.0,
        max_position: float = 0.2,  # Max 20% per asset
        annealing_steps: int = 2000
    ):
        self.risk_aversion = risk_aversion
        self.min_position = min_position
        self.max_position = max_position
        self.annealing_steps = annealing_steps

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal portfolio weights using quantum-inspired annealing.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights (for turnover penalty)
            constraints: Additional constraints (sector limits, etc.)

        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(expected_returns)

        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets

        # Build QUBO matrix
        # Q = risk_aversion * Σ - μμᵀ (simplified mean-variance)
        Q = self.risk_aversion * covariance_matrix

        # Linear term (negative returns)
        c = -expected_returns

        # Add turnover penalty
        turnover_penalty = 0.01

        # Discretize to 20 levels (0%, 5%, 10%, ..., 100%)
        n_levels = 21

        # Simulated quantum annealing
        best_weights = current_weights.copy()
        best_objective = self._compute_objective(
            best_weights, Q, c, current_weights, turnover_penalty
        )

        temperature = 10.0
        for step in range(self.annealing_steps):
            # Cooling schedule
            T = temperature * (1 - step / self.annealing_steps) ** 2

            # Propose new weights
            proposed = best_weights.copy()

            # Quantum tunneling: occasionally make larger jumps
            if np.random.random() < 0.1:  # Tunneling probability
                # Large jump - resample
                proposed = np.random.dirichlet(np.ones(n_assets))
            else:
                # Small perturbation
                idx = np.random.randint(n_assets)
                change = np.random.normal(0, 0.05)
                proposed[idx] = np.clip(proposed[idx] + change, self.min_position, self.max_position)
                proposed = proposed / proposed.sum()  # Normalize

            # Compute objective
            obj = self._compute_objective(proposed, Q, c, current_weights, turnover_penalty)

            # Acceptance probability
            delta = obj - best_objective
            if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
                best_weights = proposed
                best_objective = obj

        # Compute portfolio metrics
        portfolio_return = float(np.dot(best_weights, expected_returns))
        portfolio_vol = float(np.sqrt(best_weights @ covariance_matrix @ best_weights))
        sharpe = portfolio_return / (portfolio_vol + 1e-10)

        return {
            "weights": best_weights,
            "expected_return": portfolio_return,
            "expected_volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
            "turnover": float(np.sum(np.abs(best_weights - current_weights))),
            "diversification": float(1 - np.sum(best_weights ** 2)),
            "objective": best_objective
        }

    def _compute_objective(
        self,
        weights: np.ndarray,
        Q: np.ndarray,
        c: np.ndarray,
        current_weights: np.ndarray,
        turnover_penalty: float
    ) -> float:
        """Compute mean-variance objective with turnover penalty."""
        # Quadratic risk term
        risk = float(weights @ Q @ weights)

        # Linear return term (negative since we want to maximize)
        ret = float(c @ weights)

        # Turnover penalty
        turnover = turnover_penalty * np.sum(np.abs(weights - current_weights))

        return risk + ret + turnover
