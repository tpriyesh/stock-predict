"""
DQN Training Scheduler

Manages periodic DQN training from real trade experiences.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
from loguru import logger

from src.rl.experience_collector import ExperienceCollector, RealTradeExperience
from src.rl.reward_shaping import RewardCalculator, RewardConfig


@dataclass
class TrainingResult:
    """Result of a training session."""
    date: date
    experiences_used: int
    epochs: int
    avg_loss: float
    epsilon: float
    avg_reward: float
    metrics: Dict[str, float] = field(default_factory=dict)


class DQNTrainingScheduler:
    """
    Manages periodic DQN training from real trade experiences.

    Usage:
        from src.core.advanced_math.dqn_position_sizing import DQNPositionAgent

        collector = ExperienceCollector()
        agent = DQNPositionAgent()

        trainer = DQNTrainingScheduler(
            agent=agent,
            experience_collector=collector,
            training_frequency="daily"
        )

        # After trade closes
        if trainer.should_train(date.today()):
            result = trainer.train_batch()
            print(f"Training loss: {result.avg_loss}")
    """

    def __init__(
        self,
        agent,  # DQNPositionAgent
        experience_collector: ExperienceCollector,
        training_frequency: str = "daily",
        min_experiences: int = 30,
        batch_size: int = 32,
        epochs_per_batch: int = 5,
        reward_config: Optional[RewardConfig] = None,
        checkpoint_dir: str = "data/rl/checkpoints"
    ):
        """
        Initialize training scheduler.

        Args:
            agent: DQNPositionAgent instance
            experience_collector: ExperienceCollector for data
            training_frequency: "daily", "weekly", or "on_trade_close"
            min_experiences: Minimum experiences before training
            batch_size: Batch size for training
            epochs_per_batch: Training epochs per batch
            reward_config: Reward calculation configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        self.agent = agent
        self.collector = experience_collector
        self.training_frequency = training_frequency
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch

        self.reward_calc = RewardCalculator(reward_config or RewardConfig())
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.last_training_date: Optional[date] = None
        self.training_history: List[TrainingResult] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load training state from checkpoint."""
        state_file = self.checkpoint_dir / "training_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.last_training_date = (
                    date.fromisoformat(state['last_training_date'])
                    if state.get('last_training_date')
                    else None
                )
                self.training_history = [
                    TrainingResult(
                        date=date.fromisoformat(r['date']),
                        experiences_used=r['experiences_used'],
                        epochs=r['epochs'],
                        avg_loss=r['avg_loss'],
                        epsilon=r['epsilon'],
                        avg_reward=r['avg_reward'],
                        metrics=r.get('metrics', {})
                    )
                    for r in state.get('training_history', [])[-50:]
                ]
                logger.info(f"Loaded training state: last trained {self.last_training_date}")
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")

    def _save_state(self) -> None:
        """Save training state to checkpoint."""
        state_file = self.checkpoint_dir / "training_state.json"
        try:
            state = {
                'last_training_date': (
                    self.last_training_date.isoformat()
                    if self.last_training_date
                    else None
                ),
                'training_history': [
                    {
                        'date': r.date.isoformat(),
                        'experiences_used': r.experiences_used,
                        'epochs': r.epochs,
                        'avg_loss': r.avg_loss,
                        'epsilon': r.epsilon,
                        'avg_reward': r.avg_reward,
                        'metrics': r.metrics,
                    }
                    for r in self.training_history[-50:]
                ],
                'last_updated': datetime.now().isoformat(),
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")

    def should_train(self, current_date: date) -> bool:
        """
        Check if training should run.

        Args:
            current_date: Current date

        Returns:
            True if training should proceed
        """
        # Check minimum experiences
        if len(self.collector.experiences) < self.min_experiences:
            logger.debug(
                f"Not enough experiences: {len(self.collector.experiences)}/{self.min_experiences}"
            )
            return False

        # First time training
        if self.last_training_date is None:
            return True

        # Check frequency
        days_since_training = (current_date - self.last_training_date).days

        if self.training_frequency == "daily":
            return days_since_training >= 1
        elif self.training_frequency == "weekly":
            return days_since_training >= 7
        elif self.training_frequency == "on_trade_close":
            return True  # Train on every call

        return days_since_training >= 1

    def train_batch(
        self,
        max_experiences: int = 200
    ) -> TrainingResult:
        """
        Perform batch training on accumulated experiences.

        Args:
            max_experiences: Maximum experiences to use

        Returns:
            TrainingResult with metrics
        """
        # Get recent experiences
        experiences = self.collector.get_recent_experiences(max_experiences)

        if not experiences:
            logger.warning("No experiences available for training")
            return TrainingResult(
                date=date.today(),
                experiences_used=0,
                epochs=0,
                avg_loss=0.0,
                epsilon=self.agent.epsilon if hasattr(self.agent, 'epsilon') else 0,
                avg_reward=0.0
            )

        logger.info(f"Training DQN on {len(experiences)} experiences")

        # Calculate rewards
        rewards = [self.reward_calc.calculate(exp) for exp in experiences]
        avg_reward = np.mean(rewards)

        # Store transitions in replay buffer
        for exp, reward in zip(experiences, rewards):
            try:
                self.agent.store_transition(
                    exp.entry_state.to_array(),
                    exp.action,
                    reward,
                    exp.exit_state.to_array(),
                    True  # done=True (each trade is complete episode)
                )
            except Exception as e:
                logger.warning(f"Failed to store transition: {e}")

        # Train multiple epochs
        total_loss = 0.0
        train_steps = 0

        for epoch in range(self.epochs_per_batch):
            # Sample batch and train
            for _ in range(max(1, len(experiences) // self.batch_size)):
                try:
                    loss = self.agent.train_step()
                    if loss is not None:
                        total_loss += loss
                        train_steps += 1
                except Exception as e:
                    logger.warning(f"Training step failed: {e}")

        avg_loss = total_loss / max(1, train_steps)

        # Get current epsilon
        epsilon = self.agent.epsilon if hasattr(self.agent, 'epsilon') else 0

        # Calculate additional metrics
        metrics = self._calculate_metrics(experiences, rewards)

        # Create result
        result = TrainingResult(
            date=date.today(),
            experiences_used=len(experiences),
            epochs=self.epochs_per_batch,
            avg_loss=avg_loss,
            epsilon=epsilon,
            avg_reward=avg_reward,
            metrics=metrics
        )

        # Update state
        self.last_training_date = date.today()
        self.training_history.append(result)

        # Save checkpoint
        self._save_state()
        self._save_agent_weights()

        logger.info(
            f"Training complete: loss={avg_loss:.4f}, "
            f"avg_reward={avg_reward:.3f}, epsilon={epsilon:.3f}"
        )

        return result

    def _calculate_metrics(
        self,
        experiences: List[RealTradeExperience],
        rewards: List[float]
    ) -> Dict[str, float]:
        """Calculate additional training metrics."""
        pnls = [e.pnl_pct for e in experiences]
        holding_days = [e.holding_days for e in experiences]

        return {
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'avg_pnl': np.mean(pnls),
            'avg_holding_days': np.mean(holding_days),
            'reward_std': np.std(rewards),
            'target_hit_rate': sum(1 for e in experiences if e.hit_target) / len(experiences),
            'stop_hit_rate': sum(1 for e in experiences if e.hit_stop) / len(experiences),
        }

    def _save_agent_weights(self) -> None:
        """Save agent weights to checkpoint."""
        weights_file = self.checkpoint_dir / "dqn_weights.npz"
        try:
            if hasattr(self.agent, 'save_weights'):
                self.agent.save_weights(str(weights_file))
                logger.debug(f"Saved agent weights to {weights_file}")
            elif hasattr(self.agent, 'q_network'):
                # Manual save for SimpleNeuralNetwork
                weights = {
                    f'layer_{i}': layer
                    for i, layer in enumerate(self.agent.q_network.weights)
                }
                np.savez(weights_file, **weights)
                logger.debug(f"Saved Q-network weights to {weights_file}")
        except Exception as e:
            logger.warning(f"Failed to save agent weights: {e}")

    def load_agent_weights(self) -> bool:
        """Load agent weights from checkpoint."""
        weights_file = self.checkpoint_dir / "dqn_weights.npz"
        if not weights_file.exists():
            return False

        try:
            if hasattr(self.agent, 'load_weights'):
                self.agent.load_weights(str(weights_file))
                logger.info(f"Loaded agent weights from {weights_file}")
                return True
            elif hasattr(self.agent, 'q_network'):
                data = np.load(weights_file)
                for i, key in enumerate(sorted(data.files)):
                    if i < len(self.agent.q_network.weights):
                        self.agent.q_network.weights[i] = data[key]
                logger.info(f"Loaded Q-network weights from {weights_file}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load agent weights: {e}")

        return False

    def get_training_summary(self) -> Dict:
        """Get summary of training history."""
        if not self.training_history:
            return {
                'total_sessions': 0,
                'last_training': None,
            }

        recent = self.training_history[-10:]

        return {
            'total_sessions': len(self.training_history),
            'last_training': self.last_training_date.isoformat() if self.last_training_date else None,
            'recent_avg_loss': np.mean([r.avg_loss for r in recent]),
            'recent_avg_reward': np.mean([r.avg_reward for r in recent]),
            'current_epsilon': recent[-1].epsilon if recent else 0,
            'total_experiences_used': sum(r.experiences_used for r in self.training_history),
        }

    def reset_training(self) -> None:
        """Reset training state (use with caution)."""
        self.last_training_date = None
        self.training_history.clear()
        self._save_state()
        logger.info("Training state reset")
