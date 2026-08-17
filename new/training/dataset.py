"""Replay buffer for self-play training data.

Stores (state_features, visit_policy, winner) tuples. State features are
extracted on-the-fly during sampling so we can apply symmetry augmentation
inside the training loop.
"""
from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Sample:
    """A single training record.

    `features` is the live `Position` (cheaper to store than the 17-plane
    array, and we re-extract on the fly with random symmetries).
    `policy` is the MCTS visit-count target.
    `player` is the side that moved at this position (for value assignment).
    """

    position: object  # game.go.Position
    policy: np.ndarray  # shape (n_actions,)
    player: int  # STONE_BLACK or STONE_WHITE


class ReplayBuffer:
    """A fixed-size FIFO replay buffer of self-play samples."""

    def __init__(self, max_size: int = 200_000) -> None:
        self.max_size = max_size
        self.samples: list[Sample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.max_size:
            # Drop oldest en masse to keep memory bounded.
            self.samples = self.samples[-self.max_size :]

    def extend(self, samples: list[Sample]) -> None:
        for s in samples:
            self.add(s)

    def sample_batch(self, batch_size: int, rng: random.Random | None = None) -> list[Sample]:
        if rng is None:
            rng = random
        return rng.choices(self.samples, k=batch_size)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.samples, self.max_size), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            samples, max_size = pickle.load(f)
        self.samples = list(samples)
        self.max_size = max_size
