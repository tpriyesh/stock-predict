"""
Reinforcement Learning module for trade optimization.

Provides:
- Experience collection from real trades
- DQN training with reward shaping
- Feedback loop integration
"""

from .experience_collector import (
    RealTradeExperience,
    ExperienceCollector,
    TradingState,
)
from .dqn_trainer import (
    DQNTrainingScheduler,
    TrainingResult,
)
from .reward_shaping import (
    RewardConfig,
    RewardCalculator,
)

__all__ = [
    "RealTradeExperience",
    "ExperienceCollector",
    "TradingState",
    "DQNTrainingScheduler",
    "TrainingResult",
    "RewardConfig",
    "RewardCalculator",
]
