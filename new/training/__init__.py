"""Training pipeline: self-play, replay buffer, training, evaluation.

Import submodules explicitly to avoid chain-loading config:
    from new.training.dataset import ReplayBuffer, Sample
    from new.training.selfplay import generate_games
    from new.training.train import train_step
    from new.training.evaluate import evaluate
"""
from .dataset import ReplayBuffer, Sample  # noqa: F401
