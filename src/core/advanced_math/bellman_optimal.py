"""
Bellman Equation and Dynamic Programming Module for Optimal Trading

Implements dynamic programming for optimal decision making:
- Optimal Stopping Theory (when to enter/exit positions)
- Bellman Value Functions for position management
- Hamilton-Jacobi-Bellman (HJB) equations for continuous control
- Backward Induction for multi-stage decisions

Mathematical Foundation:
- Bellman equation: V(s) = max_a [R(s,a) + γ Σₛ' P(s'|s,a)V(s')]
- Optimal stopping: V(t,x) = max[g(x), E[V(t+1,X_{t+1})|X_t=x]]
- HJB equation: 0 = ∂V/∂t + max_u[μ∂V/∂x + ½σ²∂²V/∂x² + f(x,u)]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Possible trading actions"""
    HOLD = "hold"
    BUY_SMALL = "buy_small"  # 25% position
    BUY_MEDIUM = "buy_medium"  # 50% position
    BUY_FULL = "buy_full"  # 100% position
    SELL_PARTIAL = "sell_partial"  # Reduce position
    SELL_ALL = "sell_all"  # Close position
    STOP_LOSS = "stop_loss"  # Emergency exit
    TAKE_PROFIT = "take_profit"  # Profit taking


@dataclass
class OptimalPolicy:
    """Result of optimal policy computation"""
    action: TradingAction
    value: float  # Expected value of taking action
    continuation_value: float  # Value of continuing/holding
    stopping_value: float  # Value of stopping/acting now
    optimal_horizon: int  # Optimal holding period
    confidence: float  # Confidence in decision
    action_probabilities: Dict[str, float]  # Probability distribution over actions
    expected_payoff: float  # Expected reward
    risk_adjusted_payoff: float  # Sharpe-like risk adjustment
    state_analysis: Dict[str, Any]  # Current state interpretation


class BellmanOptimalStopping:
    """
    Optimal Stopping Theory for Trade Entry/Exit Decisions.

    Solves the classical optimal stopping problem:
    V(t,x) = max[g(x), E[e^{-r}V(t+1,X_{t+1})|X_t=x]]

    Where:
    - g(x): Immediate payoff (close position now)
    - V: Continuation value (hold and act optimally later)
    - r: Discount rate (time value of money)
    """

    def __init__(
        self,
        discount_rate: float = 0.0001,  # Daily discount
        risk_free_rate: float = 0.07,  # Annual risk-free rate
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_horizon: int = 60,  # Maximum holding period
        n_price_states: int = 50,  # Discretization of price space
        n_volatility_states: int = 10  # Discretization of vol space
    ):
        self.discount_rate = discount_rate
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.max_horizon = max_horizon
        self.n_price_states = n_price_states
        self.n_volatility_states = n_volatility_states

        # Value function cache
        self._value_function: Optional[np.ndarray] = None
        self._policy: Optional[np.ndarray] = None

    def compute_optimal_policy(
        self,
        returns: np.ndarray,
        entry_price: float,
        current_price: float,
        position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OptimalPolicy:
        """
        Compute optimal trading policy using backward induction.

        Args:
            returns: Historical return series for estimating dynamics
            entry_price: Price at which position was entered
            current_price: Current market price
            position_size: Current position size (0 to 1)
            stop_loss: Optional stop loss level
            take_profit: Optional take profit level

        Returns:
            OptimalPolicy with recommended action and analysis
        """
        if len(returns) < 30:
            return self._default_policy()

        # Step 1: Estimate price dynamics (drift and volatility)
        mu, sigma = self._estimate_dynamics(returns)

        # Step 2: Build state space
        price_states, vol_states = self._build_state_space(
            current_price, sigma, returns
        )

        # Step 3: Compute transition probabilities
        transition_probs = self._compute_transitions(
            price_states, mu, sigma
        )

        # Step 4: Define payoff function
        pnl_ratio = (current_price - entry_price) / entry_price

        # Step 5: Backward induction for value function
        value_function, policy = self._backward_induction(
            price_states, vol_states, transition_probs,
            entry_price, pnl_ratio, stop_loss, take_profit
        )

        # Step 6: Find current state and extract policy
        current_state_idx = self._find_state_index(
            current_price, price_states
        )
        current_vol_idx = self._find_vol_state(returns, vol_states)

        # Step 7: Compute optimal action
        continuation_value = value_function[0, current_state_idx, current_vol_idx]
        stopping_value = self._immediate_payoff(pnl_ratio, position_size)

        # Step 8: Determine action from policy
        action, action_probs = self._determine_action(
            value_function, policy, current_state_idx, current_vol_idx,
            pnl_ratio, stop_loss, take_profit, current_price, entry_price
        )

        # Step 9: Compute optimal horizon
        optimal_horizon = self._compute_optimal_horizon(
            value_function, current_state_idx, current_vol_idx
        )

        # Step 10: Risk-adjusted analysis
        expected_payoff = continuation_value
        risk_adjusted = expected_payoff / (sigma * np.sqrt(optimal_horizon) + 1e-10)

        # Confidence based on value gap and state clarity
        confidence = self._compute_confidence(
            continuation_value, stopping_value, sigma, optimal_horizon
        )

        return OptimalPolicy(
            action=action,
            value=float(max(continuation_value, stopping_value)),
            continuation_value=float(continuation_value),
            stopping_value=float(stopping_value),
            optimal_horizon=optimal_horizon,
            confidence=confidence,
            action_probabilities=action_probs,
            expected_payoff=float(expected_payoff),
            risk_adjusted_payoff=float(risk_adjusted),
            state_analysis={
                "current_pnl_pct": float(pnl_ratio * 100),
                "estimated_drift": float(mu * 252),  # Annualized
                "estimated_vol": float(sigma * np.sqrt(252)),  # Annualized
                "value_of_waiting": float(continuation_value - stopping_value),
                "regime": self._classify_regime(mu, sigma, returns)
            }
        )

    def _estimate_dynamics(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate drift (μ) and volatility (σ) from returns.
        Uses exponentially weighted estimates for recency bias.
        """
        # Exponential weights (half-life of 20 days)
        half_life = 20
        decay = 0.5 ** (1 / half_life)
        weights = decay ** np.arange(len(returns))[::-1]
        weights = weights / weights.sum()

        # Weighted mean (drift)
        mu = np.sum(weights * returns)

        # Weighted standard deviation
        variance = np.sum(weights * (returns - mu) ** 2)
        sigma = np.sqrt(variance)

        return float(mu), float(sigma)

    def _build_state_space(
        self,
        current_price: float,
        sigma: float,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build discretized state space for price and volatility."""
        # Price states: ±3 sigma around current price
        price_range = 3 * sigma * np.sqrt(self.max_horizon)
        price_min = current_price * (1 - price_range)
        price_max = current_price * (1 + price_range)
        price_states = np.linspace(price_min, price_max, self.n_price_states)

        # Volatility states: range of observed volatilities
        rolling_vol = np.array([
            np.std(returns[max(0, i-20):i+1])
            for i in range(len(returns))
        ])
        vol_min = np.percentile(rolling_vol, 5)
        vol_max = np.percentile(rolling_vol, 95)
        vol_states = np.linspace(vol_min, vol_max, self.n_volatility_states)

        return price_states, vol_states

    def _compute_transitions(
        self,
        price_states: np.ndarray,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Compute transition probability matrix.
        Assumes log-normal price dynamics:
        log(S_{t+1}) = log(S_t) + μ - σ²/2 + σε
        """
        n_states = len(price_states)
        transitions = np.zeros((n_states, n_states))

        for i, p_current in enumerate(price_states):
            # Expected log return
            log_return_mean = mu - 0.5 * sigma ** 2

            for j, p_next in enumerate(price_states):
                # Required log return to reach state j from state i
                required_return = np.log(p_next / p_current + 1e-10)

                # Probability under normal distribution
                z = (required_return - log_return_mean) / (sigma + 1e-10)

                # Discretized probability (using state widths)
                if j == 0:
                    prob = self._norm_cdf(z + 0.5 / sigma)
                elif j == n_states - 1:
                    prob = 1 - self._norm_cdf(z - 0.5 / sigma)
                else:
                    prob = self._norm_cdf(z + 0.5 / sigma) - self._norm_cdf(z - 0.5 / sigma)

                transitions[i, j] = max(0, prob)

            # Normalize row
            row_sum = transitions[i, :].sum()
            if row_sum > 0:
                transitions[i, :] /= row_sum

        return transitions

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def _backward_induction(
        self,
        price_states: np.ndarray,
        vol_states: np.ndarray,
        transition_probs: np.ndarray,
        entry_price: float,
        current_pnl_ratio: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Bellman equation via backward induction.

        V(t,s,v) = max[g(s), e^{-r}E[V(t+1,S',V')|S=s,V=v]]
        """
        n_time = self.max_horizon
        n_price = len(price_states)
        n_vol = len(vol_states)

        # Value function: V[t, price_state, vol_state]
        V = np.zeros((n_time, n_price, n_vol))

        # Policy: 0 = continue, 1 = stop
        policy = np.zeros((n_time, n_price, n_vol), dtype=int)

        # Terminal condition: at horizon, must close
        for i, price in enumerate(price_states):
            pnl = (price - entry_price) / entry_price - self.transaction_cost
            V[-1, i, :] = pnl

        # Backward iteration
        discount = np.exp(-self.discount_rate)

        for t in range(n_time - 2, -1, -1):
            for i, price in enumerate(price_states):
                pnl_ratio = (price - entry_price) / entry_price

                # Check stop loss / take profit
                if stop_loss is not None and price <= stop_loss:
                    V[t, i, :] = pnl_ratio - self.transaction_cost
                    policy[t, i, :] = 1
                    continue

                if take_profit is not None and price >= take_profit:
                    V[t, i, :] = pnl_ratio - self.transaction_cost
                    policy[t, i, :] = 1
                    continue

                # Immediate payoff (stop now)
                stopping_value = pnl_ratio - self.transaction_cost

                for v in range(n_vol):
                    # Continuation value
                    expected_future = 0.0
                    for j in range(n_price):
                        # Simple vol state transition (mean reverting)
                        for v_next in range(n_vol):
                            vol_trans_prob = self._vol_transition_prob(v, v_next, n_vol)
                            expected_future += (
                                transition_probs[i, j] * vol_trans_prob * V[t+1, j, v_next]
                            )

                    continuation_value = discount * expected_future

                    # Bellman optimality
                    if stopping_value >= continuation_value:
                        V[t, i, v] = stopping_value
                        policy[t, i, v] = 1  # Stop
                    else:
                        V[t, i, v] = continuation_value
                        policy[t, i, v] = 0  # Continue

        self._value_function = V
        self._policy = policy

        return V, policy

    def _vol_transition_prob(
        self,
        current: int,
        next_state: int,
        n_states: int
    ) -> float:
        """Volatility transition (mean reverting to middle state)."""
        # Vol tends to mean-revert
        mean_state = n_states // 2

        # Distance from mean
        current_dist = abs(current - mean_state)
        next_dist = abs(next_state - mean_state)

        # Higher probability of moving toward mean
        if next_dist < current_dist:
            base_prob = 0.6
        elif next_dist == current_dist:
            base_prob = 0.3
        else:
            base_prob = 0.1

        # Normalize across possible transitions
        return base_prob

    def _immediate_payoff(
        self,
        pnl_ratio: float,
        position_size: float
    ) -> float:
        """Compute immediate payoff from closing position."""
        return (pnl_ratio - self.transaction_cost) * position_size

    def _find_state_index(
        self,
        price: float,
        price_states: np.ndarray
    ) -> int:
        """Find index of closest price state."""
        return int(np.argmin(np.abs(price_states - price)))

    def _find_vol_state(
        self,
        returns: np.ndarray,
        vol_states: np.ndarray
    ) -> int:
        """Find current volatility state."""
        current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        return int(np.argmin(np.abs(vol_states - current_vol)))

    def _determine_action(
        self,
        value_function: np.ndarray,
        policy: np.ndarray,
        price_idx: int,
        vol_idx: int,
        pnl_ratio: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        current_price: float,
        entry_price: float
    ) -> Tuple[TradingAction, Dict[str, float]]:
        """Determine specific trading action from policy."""

        # Check immediate conditions
        if stop_loss and current_price <= stop_loss:
            return TradingAction.STOP_LOSS, {"stop_loss": 1.0}

        if take_profit and current_price >= take_profit:
            return TradingAction.TAKE_PROFIT, {"take_profit": 1.0}

        # Get policy at current state
        optimal_stop = policy[0, price_idx, vol_idx]

        if optimal_stop == 1:  # Should stop/act
            if pnl_ratio > 0.02:  # >2% profit
                return TradingAction.TAKE_PROFIT, {
                    "take_profit": 0.7,
                    "sell_partial": 0.2,
                    "hold": 0.1
                }
            elif pnl_ratio < -0.02:  # >2% loss
                return TradingAction.STOP_LOSS, {
                    "stop_loss": 0.8,
                    "hold": 0.2
                }
            else:
                return TradingAction.SELL_ALL, {
                    "sell_all": 0.6,
                    "sell_partial": 0.3,
                    "hold": 0.1
                }
        else:  # Continue holding
            # Analyze value gradient
            continuation_val = value_function[0, price_idx, vol_idx]

            # Check if should add to position
            if continuation_val > 0.03 and pnl_ratio > 0:
                return TradingAction.BUY_SMALL, {
                    "buy_small": 0.4,
                    "hold": 0.5,
                    "buy_medium": 0.1
                }
            else:
                return TradingAction.HOLD, {
                    "hold": 0.8,
                    "sell_partial": 0.1,
                    "buy_small": 0.1
                }

    def _compute_optimal_horizon(
        self,
        value_function: np.ndarray,
        price_idx: int,
        vol_idx: int
    ) -> int:
        """Compute optimal holding period."""
        # Find when expected value peaks
        values_over_time = value_function[:, price_idx, vol_idx]

        # Optimal horizon is when value stops increasing
        for t in range(1, len(values_over_time)):
            if values_over_time[t] < values_over_time[t-1] - 0.001:
                return t

        return self.max_horizon // 2  # Default to half max

    def _compute_confidence(
        self,
        continuation: float,
        stopping: float,
        sigma: float,
        horizon: int
    ) -> float:
        """Compute confidence in the optimal decision."""
        # Value gap relative to volatility uncertainty
        value_gap = abs(continuation - stopping)
        uncertainty = sigma * np.sqrt(horizon)

        # Higher gap relative to uncertainty = higher confidence
        signal_to_noise = value_gap / (uncertainty + 1e-10)

        return float(np.clip(signal_to_noise / 2, 0, 1))

    def _classify_regime(
        self,
        mu: float,
        sigma: float,
        returns: np.ndarray
    ) -> str:
        """Classify current market regime."""
        sharpe = mu / (sigma + 1e-10) * np.sqrt(252)

        if sharpe > 1.5:
            return "strong_trend_up"
        elif sharpe > 0.5:
            return "mild_trend_up"
        elif sharpe > -0.5:
            return "sideways"
        elif sharpe > -1.5:
            return "mild_trend_down"
        else:
            return "strong_trend_down"

    def _default_policy(self) -> OptimalPolicy:
        """Return default policy for insufficient data."""
        return OptimalPolicy(
            action=TradingAction.HOLD,
            value=0.0,
            continuation_value=0.0,
            stopping_value=0.0,
            optimal_horizon=10,
            confidence=0.0,
            action_probabilities={"hold": 1.0},
            expected_payoff=0.0,
            risk_adjusted_payoff=0.0,
            state_analysis={"regime": "unknown"}
        )


class HamiltonJacobiBellman:
    """
    Hamilton-Jacobi-Bellman Equation Solver for Continuous Control.

    Solves: 0 = ∂V/∂t + max_u[f(x,u) + μ(x,u)∂V/∂x + ½σ²(x,u)∂²V/∂x²]

    For trading:
    - x: State (price, position, P&L)
    - u: Control (position change)
    - f: Running cost/reward
    - μ, σ: Drift and volatility
    """

    def __init__(
        self,
        n_space_points: int = 100,
        n_time_points: int = 50,
        risk_aversion: float = 1.0
    ):
        self.n_space = n_space_points
        self.n_time = n_time_points
        self.risk_aversion = risk_aversion

    def solve_optimal_execution(
        self,
        initial_position: float,
        target_position: float,
        time_horizon: int,
        volatility: float,
        temporary_impact: float = 0.01,
        permanent_impact: float = 0.005
    ) -> Dict[str, Any]:
        """
        Solve Almgren-Chriss optimal execution problem.

        Minimizes: E[Cost] + λ * Var[Cost]

        Where cost includes:
        - Temporary market impact (execution price slippage)
        - Permanent impact (lasting price effect)
        - Risk penalty (variance of execution cost)
        """
        position_diff = target_position - initial_position

        if abs(position_diff) < 1e-10:
            return {
                "trajectory": np.array([initial_position]),
                "execution_schedule": np.array([0.0]),
                "expected_cost": 0.0,
                "cost_std": 0.0,
                "optimal_urgency": 0.0
            }

        # Optimal urgency parameter (Almgren-Chriss formula)
        # κ² = λσ² / η where η is temporary impact
        kappa_sq = self.risk_aversion * volatility ** 2 / (temporary_impact + 1e-10)
        kappa = np.sqrt(max(kappa_sq, 0.01))

        # Time grid
        t = np.linspace(0, time_horizon, self.n_time)

        # Optimal trajectory (inventory over time)
        # X(t) = X₀ * sinh(κ(T-t)) / sinh(κT)
        sinh_kappa_T = np.sinh(kappa * time_horizon)
        if abs(sinh_kappa_T) < 1e-10:
            trajectory = np.linspace(initial_position, target_position, self.n_time)
        else:
            trajectory = initial_position + position_diff * (
                1 - np.sinh(kappa * (time_horizon - t)) / sinh_kappa_T
            )

        # Execution schedule (rate of trading)
        execution_schedule = np.diff(trajectory) / np.diff(t)
        execution_schedule = np.append(execution_schedule, execution_schedule[-1])

        # Expected cost
        # Temporary impact cost: ∫η(dX/dt)²dt
        temp_cost = temporary_impact * np.sum(execution_schedule ** 2)

        # Permanent impact cost: ½γX₀²
        perm_cost = 0.5 * permanent_impact * position_diff ** 2

        expected_cost = temp_cost + perm_cost

        # Cost variance: σ²∫X(t)²dt
        cost_variance = volatility ** 2 * np.sum(trajectory ** 2) / time_horizon
        cost_std = np.sqrt(cost_variance)

        return {
            "trajectory": trajectory,
            "execution_schedule": execution_schedule,
            "expected_cost": float(expected_cost),
            "cost_std": float(cost_std),
            "optimal_urgency": float(kappa),
            "time_grid": t
        }

    def solve_merton_problem(
        self,
        wealth: float,
        risk_free_rate: float,
        expected_return: float,
        volatility: float,
        time_horizon: float,
        risk_aversion: float
    ) -> Dict[str, Any]:
        """
        Solve Merton's portfolio problem.

        Maximize: E[U(W_T)] where U(W) = W^{1-γ}/(1-γ)

        Optimal fraction in risky asset:
        π* = (μ-r) / (γσ²)
        """
        excess_return = expected_return - risk_free_rate

        # Optimal risky asset allocation (Merton ratio)
        optimal_fraction = excess_return / (risk_aversion * volatility ** 2 + 1e-10)
        optimal_fraction = np.clip(optimal_fraction, -2, 2)  # Limit leverage

        # Value function
        # V(W,t) = exp(A(T-t)) * W^{1-γ}/(1-γ)
        # where A = (1-γ)/γ * [(μ-r)²/(2σ²) + r]
        A = ((1 - risk_aversion) / risk_aversion) * (
            excess_return ** 2 / (2 * volatility ** 2) + risk_free_rate
        )

        # Expected terminal wealth
        expected_growth = risk_free_rate + optimal_fraction * excess_return - \
                          0.5 * optimal_fraction ** 2 * volatility ** 2
        expected_terminal = wealth * np.exp(expected_growth * time_horizon)

        # Wealth volatility
        wealth_vol = abs(optimal_fraction) * volatility * np.sqrt(time_horizon)

        return {
            "optimal_risky_fraction": float(optimal_fraction),
            "optimal_cash_fraction": float(1 - optimal_fraction),
            "value_function_param": float(A),
            "expected_terminal_wealth": float(expected_terminal),
            "wealth_volatility": float(wealth_vol),
            "sharpe_ratio": float(excess_return / (volatility + 1e-10)),
            "certainty_equivalent_return": float(
                expected_growth - 0.5 * risk_aversion * wealth_vol ** 2
            )
        }


class ReinforcementLearningTrader:
    """
    Model-free reinforcement learning for trading decisions.
    Uses Q-learning with function approximation.

    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        n_features: int = 10,
        n_actions: int = 5,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-function weights (linear function approximation)
        self.weights = np.zeros((n_actions, n_features))

        # Experience replay buffer
        self.replay_buffer: List[Tuple] = []
        self.buffer_size = 10000

    def compute_features(self, returns: np.ndarray) -> np.ndarray:
        """Extract features from return series for Q-function."""
        if len(returns) < 20:
            return np.zeros(self.n_features)

        features = np.zeros(self.n_features)

        # Feature 1: Recent return (5-day)
        features[0] = np.sum(returns[-5:]) / (np.std(returns) + 1e-10)

        # Feature 2: Medium-term return (20-day)
        features[1] = np.sum(returns[-20:]) / (np.std(returns) + 1e-10)

        # Feature 3: Volatility regime
        recent_vol = np.std(returns[-10:])
        longer_vol = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)
        features[2] = recent_vol / (longer_vol + 1e-10) - 1

        # Feature 4: Trend strength (autocorrelation)
        if len(returns) >= 6:
            features[3] = np.corrcoef(returns[:-5], returns[5:])[0, 1]

        # Feature 5: Momentum (return acceleration)
        if len(returns) >= 10:
            recent = np.mean(returns[-5:])
            previous = np.mean(returns[-10:-5])
            features[4] = (recent - previous) / (np.std(returns) + 1e-10)

        # Feature 6: RSI-like overbought/oversold
        gains = np.maximum(returns[-14:], 0).sum()
        losses = -np.minimum(returns[-14:], 0).sum()
        if gains + losses > 0:
            features[5] = gains / (gains + losses) - 0.5

        # Feature 7: Drawdown from peak
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak[-1] - cumulative[-1]) / (np.std(cumulative) + 1e-10)
        features[6] = -drawdown

        # Feature 8: Days since trend change
        signs = np.sign(returns[-20:])
        changes = np.where(np.diff(signs) != 0)[0]
        if len(changes) > 0:
            features[7] = (len(returns[-20:]) - changes[-1]) / 20
        else:
            features[7] = 1.0

        # Feature 9: Volume surge (if available, use volatility proxy)
        vol_ratio = np.std(returns[-5:]) / (np.std(returns[-20:]) + 1e-10)
        features[8] = vol_ratio - 1

        # Feature 10: Mean reversion potential
        z_score = (np.mean(returns[-5:]) - np.mean(returns[-60:])) / (np.std(returns) + 1e-10)
        features[9] = -z_score  # Negative because high z suggests reversion down

        return features

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        return self.weights @ features

    def select_action(
        self,
        features: np.ndarray,
        training: bool = False
    ) -> Tuple[int, np.ndarray]:
        """Select action using epsilon-greedy policy."""
        q_values = self.get_q_values(features)

        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(q_values))

        return action, q_values

    def update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        done: bool
    ):
        """Update Q-function using TD learning."""
        # Current Q-value
        current_q = self.weights[action] @ features

        # Target Q-value
        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_features)
            target = reward + self.gamma * np.max(next_q_values)

        # TD error
        td_error = target - current_q

        # Update weights
        self.weights[action] += self.alpha * td_error * features

    def get_optimal_action(self, returns: np.ndarray) -> Dict[str, Any]:
        """Get optimal action recommendation."""
        features = self.compute_features(returns)
        action, q_values = self.select_action(features, training=False)

        # Map action to trading decision
        action_map = {
            0: TradingAction.SELL_ALL,
            1: TradingAction.SELL_PARTIAL,
            2: TradingAction.HOLD,
            3: TradingAction.BUY_SMALL,
            4: TradingAction.BUY_FULL
        }

        # Softmax for action probabilities
        exp_q = np.exp(q_values - np.max(q_values))
        action_probs = exp_q / exp_q.sum()

        return {
            "action": action_map[action],
            "action_index": action,
            "q_values": q_values.tolist(),
            "action_probabilities": {
                action_map[i].value: float(action_probs[i])
                for i in range(self.n_actions)
            },
            "confidence": float(action_probs[action]),
            "features": features.tolist()
        }
