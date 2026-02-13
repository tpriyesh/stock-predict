"""
Deep Q-Network (DQN) for Position Sizing and Timing

Mathematical Foundation:
========================
Reinforcement Learning for Trading Decisions

State Space:
   s = [price_features, momentum, volatility, regime, position, P&L]

Action Space (Discrete):
   a ∈ {Hold, Buy_25%, Buy_50%, Buy_100%, Sell_25%, Sell_50%, Sell_100%}

Reward Function:
   rₜ = returnₜ × positionₜ₋₁ - λ × |actionₜ| × transaction_cost

Bellman Equation:
   Q*(s, a) = E[r + γ max Q*(s', a')]

Deep Q-Network:
   Q(s, a; θ) ≈ Q*(s, a)

Training:
- Experience Replay: Store transitions (s, a, r, s')
- Target Network: Soft update τ = 0.001
- ε-greedy exploration: ε decays 1.0 → 0.01

References:
- Mnih et al. (2015). Human-level control through deep reinforcement learning
- Deng et al. (2016). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import random
from loguru import logger


class TradingAction(Enum):
    """Discrete trading actions."""
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6


@dataclass
class TradingState:
    """Current trading state."""
    # Market features (normalized)
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility: float
    rsi: float
    macd_signal: float
    volume_ratio: float

    # Regime indicator (one-hot encoded)
    regime: int  # 0-4

    # Position info
    current_position: float  # -1 to 1 (short to long)
    unrealized_pnl: float  # Normalized P&L

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.price_change_1d,
            self.price_change_5d,
            self.price_change_20d,
            self.volatility,
            self.rsi / 100.0,  # Normalize RSI
            self.macd_signal,
            self.volume_ratio,
            self.regime / 4.0,  # Normalize regime
            self.current_position,
            self.unrealized_pnl
        ])


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores transitions and samples random minibatches for training.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class SimpleNeuralNetwork:
    """
    Simple neural network implementation for Q-function.

    Uses numpy only (no deep learning framework dependencies).
    Architecture: Input -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Output
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros(dims[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        # Ensure 2D input
        if x.ndim == 1:
            x = x.reshape(1, -1)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            # ReLU activation for hidden layers
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)

        return x

    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all weights and biases."""
        return list(zip(self.weights, self.biases))

    def set_weights(self, weights: List[Tuple[np.ndarray, np.ndarray]]):
        """Set all weights and biases."""
        for i, (w, b) in enumerate(weights):
            self.weights[i] = w.copy()
            self.biases[i] = b.copy()

    def soft_update(self, other: 'SimpleNeuralNetwork', tau: float = 0.01):
        """Soft update weights from another network."""
        for i in range(len(self.weights)):
            self.weights[i] = tau * other.weights[i] + (1 - tau) * self.weights[i]
            self.biases[i] = tau * other.biases[i] + (1 - tau) * self.biases[i]


class DQNPositionAgent:
    """
    Deep Q-Network agent for position sizing and trade timing.

    Uses reinforcement learning to learn optimal:
    - When to enter trades
    - Position size (25%, 50%, 100%)
    - When to exit trades
    """

    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 7,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 0.01,
        buffer_size: int = 100000
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Networks
        self.q_network = SimpleNeuralNetwork(state_dim, n_actions)
        self.target_network = SimpleNeuralNetwork(state_dim, n_actions)
        self.target_network.set_weights(self.q_network.get_weights())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training tracking
        self.training_step = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state
            training: If True, use ε-greedy; if False, use greedy

        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: greedy action
            q_values = self.q_network.forward(state)
            return int(np.argmax(q_values))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step.

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare arrays
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=float)

        # Compute target Q-values
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute current Q-values
        current_q = self.q_network.forward(states)

        # Compute loss and gradients (simplified SGD)
        loss = 0.0
        for i in range(len(batch)):
            action = actions[i]
            target = targets[i]
            prediction = current_q[i, action]
            loss += (target - prediction) ** 2

            # Simple gradient update (online learning)
            error = target - prediction
            self._update_weights(states[i], action, error)

        loss /= len(batch)

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.soft_update(self.q_network, self.tau)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss

    def _update_weights(self, state: np.ndarray, action: int, error: float):
        """
        Simple gradient descent weight update.

        This is a simplified version - full backprop would be better.
        """
        # Only update output layer for simplicity
        x = state.reshape(1, -1)

        # Forward to get hidden activations
        hidden = x
        for i in range(len(self.q_network.weights) - 1):
            hidden = hidden @ self.q_network.weights[i] + self.q_network.biases[i]
            hidden = np.maximum(0, hidden)  # ReLU

        # Update last layer weights for selected action
        grad = self.learning_rate * error
        self.q_network.weights[-1][:, action] += grad * hidden.flatten()
        self.q_network.biases[-1][action] += grad

    def train_episode(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        transaction_cost: float = 0.001
    ) -> Tuple[float, List[int]]:
        """
        Train on one episode (full price history).

        Args:
            prices: Price array
            features: Feature matrix (n_periods x n_features)
            transaction_cost: Cost per trade

        Returns:
            (total_reward, action_history)
        """
        n = len(prices)
        if n < 2 or len(features) != n:
            return 0.0, []

        total_reward = 0.0
        action_history = []
        position = 0.0  # Current position (-1 to 1)
        entry_price = prices[0]

        for t in range(n - 1):
            # Create state
            state = self._create_state(features[t], position, prices[t], entry_price)

            # Select action
            action = self.select_action(state)
            action_history.append(action)

            # Execute action and get reward
            new_position, cost = self._execute_action(action, position, transaction_cost)

            # Calculate reward
            price_change = (prices[t + 1] - prices[t]) / prices[t]
            pnl = position * price_change - cost
            reward = self._calculate_reward(pnl, action, position)

            # Update entry price if position changed
            if new_position != position and new_position != 0:
                entry_price = prices[t]

            # Create next state
            next_state = self._create_state(
                features[t + 1] if t + 1 < len(features) else features[t],
                new_position,
                prices[t + 1],
                entry_price
            )

            # Store transition
            done = (t == n - 2)
            self.store_transition(state, action, reward, next_state, done)

            # Train
            self.train_step()

            # Update
            position = new_position
            total_reward += reward

        self.episode_rewards.append(total_reward)
        return total_reward, action_history

    def _create_state(
        self,
        features: np.ndarray,
        position: float,
        current_price: float,
        entry_price: float
    ) -> np.ndarray:
        """Create state array from features and position."""
        # Calculate unrealized P&L
        if position != 0 and entry_price > 0:
            pnl = position * (current_price - entry_price) / entry_price
        else:
            pnl = 0.0

        # Combine features with position info
        if len(features) >= self.state_dim - 2:
            state = np.zeros(self.state_dim)
            state[:self.state_dim - 2] = features[:self.state_dim - 2]
            state[-2] = position
            state[-1] = pnl
        else:
            state = np.zeros(self.state_dim)
            state[:len(features)] = features
            state[-2] = position
            state[-1] = pnl

        return state

    def _execute_action(
        self,
        action: int,
        current_position: float,
        transaction_cost: float
    ) -> Tuple[float, float]:
        """
        Execute trading action.

        Returns:
            (new_position, transaction_cost_incurred)
        """
        action_map = {
            TradingAction.HOLD.value: 0.0,
            TradingAction.BUY_25.value: 0.25,
            TradingAction.BUY_50.value: 0.50,
            TradingAction.BUY_100.value: 1.00,
            TradingAction.SELL_25.value: -0.25,
            TradingAction.SELL_50.value: -0.50,
            TradingAction.SELL_100.value: -1.00,
        }

        target_change = action_map.get(action, 0.0)

        if action == TradingAction.HOLD.value:
            return current_position, 0.0

        # Calculate new position
        if action in [TradingAction.BUY_25.value, TradingAction.BUY_50.value, TradingAction.BUY_100.value]:
            new_position = min(1.0, current_position + target_change)
        else:
            new_position = max(-1.0, current_position + target_change)

        # Calculate transaction cost
        position_change = abs(new_position - current_position)
        cost = position_change * transaction_cost

        return new_position, cost

    def _calculate_reward(
        self,
        pnl: float,
        action: int,
        position: float
    ) -> float:
        """
        Calculate reward from action.

        Rewards:
        - Profit: positive
        - Loss: negative
        - Holding when should trade: small negative
        - Overtrading: penalty
        """
        # Base reward is P&L
        reward = pnl * 100  # Scale up for learning

        # Small penalty for excessive trading
        if action != TradingAction.HOLD.value:
            reward -= 0.01

        # Bonus for profitable trades
        if pnl > 0.01:
            reward += 0.1

        return reward

    def get_optimal_position(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Get optimal action and Q-values for a state.

        Returns:
            (action, target_position, confidence)
        """
        q_values = self.q_network.forward(state).flatten()
        action = int(np.argmax(q_values))

        # Convert action to position
        position_map = {
            TradingAction.HOLD.value: 0.0,
            TradingAction.BUY_25.value: 0.25,
            TradingAction.BUY_50.value: 0.50,
            TradingAction.BUY_100.value: 1.00,
            TradingAction.SELL_25.value: -0.25,
            TradingAction.SELL_50.value: -0.50,
            TradingAction.SELL_100.value: -1.00,
        }

        target_position = position_map.get(action, 0.0)

        # Confidence based on Q-value difference
        sorted_q = np.sort(q_values)[::-1]
        if len(sorted_q) > 1:
            confidence = 1 / (1 + np.exp(-(sorted_q[0] - sorted_q[1])))  # Sigmoid of advantage
        else:
            confidence = 0.5

        return action, target_position, confidence


def create_trading_features(prices: pd.DataFrame) -> np.ndarray:
    """
    Create feature matrix for DQN from price data.

    Args:
        prices: OHLCV DataFrame

    Returns:
        Feature matrix (n x state_dim)
    """
    close = prices['close'] if 'close' in prices.columns else prices['Close']
    volume = prices['volume'] if 'volume' in prices.columns else prices['Volume']

    features = pd.DataFrame()

    # Price changes
    features['return_1d'] = close.pct_change()
    features['return_5d'] = close.pct_change(5)
    features['return_20d'] = close.pct_change(20)

    # Volatility
    features['volatility'] = features['return_1d'].rolling(20).std()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    features['macd_signal'] = (macd - macd.ewm(span=9).mean()) / close

    # Volume ratio
    features['volume_ratio'] = volume / volume.rolling(20).mean()

    # Regime proxy (simple: based on 20-day trend)
    trend = close.pct_change(20)
    volatility = features['volatility']

    def classify_regime(row):
        if pd.isna(row['trend']) or pd.isna(row['vol']):
            return 2  # Sideways
        if row['trend'] > 0.05:
            return 0 if row['vol'] < 0.02 else 1  # Bull low/high vol
        elif row['trend'] < -0.05:
            return 3 if row['vol'] < 0.02 else 4  # Bear low/high vol
        return 2  # Sideways

    regime_df = pd.DataFrame({'trend': trend, 'vol': volatility})
    features['regime'] = regime_df.apply(classify_regime, axis=1)

    # Fill NaN and normalize
    features = features.fillna(0)

    return features.values


def get_dqn_position_recommendation(
    prices: pd.DataFrame,
    current_position: float = 0.0,
    agent: Optional[DQNPositionAgent] = None
) -> Dict[str, Any]:
    """
    Get position recommendation from DQN agent.

    Returns:
        Dict with action, target_position, confidence, and reasoning
    """
    # Create features
    features = create_trading_features(prices)

    if len(features) < 30:
        return {
            'action': 'HOLD',
            'target_position': current_position,
            'confidence': 0.0,
            'reason': 'Insufficient data'
        }

    # Create or use provided agent
    if agent is None:
        agent = DQNPositionAgent()

    # Get latest state
    latest_features = features[-1]
    close_prices = prices['close'].values if 'close' in prices.columns else prices['Close'].values
    current_price = close_prices[-1]
    entry_price = current_price  # Assume current price as reference

    state = agent._create_state(latest_features, current_position, current_price, entry_price)

    # Get recommendation
    action, target_position, confidence = agent.get_optimal_position(state)

    # Map action to string
    action_names = {
        0: 'HOLD',
        1: 'BUY_25%',
        2: 'BUY_50%',
        3: 'BUY_100%',
        4: 'SELL_25%',
        5: 'SELL_50%',
        6: 'SELL_100%'
    }

    return {
        'action': action_names.get(action, 'HOLD'),
        'action_id': action,
        'target_position': target_position,
        'confidence': confidence,
        'current_position': current_position,
        'position_change': target_position - current_position if action != 0 else 0,
        'reason': f'DQN recommendation based on market state (confidence: {confidence:.2%})'
    }
