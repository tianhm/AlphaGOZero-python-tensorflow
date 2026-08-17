"""Training loop: policy + value loss with symmetry augmentation."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from new.game.features import apply_symmetry, extract_features
from new.model.network import AGZNet
from new.config import HPS
from .dataset import ReplayBuffer, Sample


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    mean_value: float
    lr: float


def build_optimizer(network: AGZNet) -> Adam:
    return Adam(network.parameters(), lr=HPS.lr, weight_decay=HPS.weight_decay)


def _mirror_policy(policy: np.ndarray, n: int, sym: int) -> np.ndarray:
    """Mirror a length-(n*n+1) policy by symmetry. Pass (last index) invariant."""
    board = policy[: n * n].reshape(n, n)
    flipped = np.flip(board, axis=-1) if sym >= 4 else board
    k = sym % 4
    rotated = np.rot90(flipped, k=k)
    return np.concatenate([rotated.flatten(), policy[n * n : n * n + 1]])


def train_step(
    network: AGZNet,
    optimizer: Adam,
    batch: list[Sample],
    n_planes: int,
    n_actions: int,
    board_size: int,
    device: str,
) -> TrainStats:
    """One optimization step on a batch of samples."""
    network.train()

    features = np.zeros((len(batch), n_planes, board_size, board_size), dtype=np.float32)
    policies = np.zeros((len(batch), n_actions), dtype=np.float32)
    values = np.zeros((len(batch), 1), dtype=np.float32)

    for i, sample in enumerate(batch):
        sym = random.randint(0, 7)
        feats = extract_features(sample.position, n_planes=n_planes)
        feats = apply_symmetry(feats, sym)
        features[i] = feats

        policies[i] = _mirror_policy(sample.policy, board_size, sym)

        # Value target: from the player's perspective, z = +1 if player won else -1.
        sample_winner = getattr(sample, "_winner", None)
        if sample_winner is None:
            sample_winner = sample.player
        values[i, 0] = 1.0 if sample_winner == sample.player else -1.0

    x = torch.from_numpy(features).to(device)
    p_target = torch.from_numpy(policies).to(device)
    v_target = torch.from_numpy(values).to(device)

    policy_logits, value = network(x)

    policy_loss = -torch.sum(p_target * F.log_softmax(policy_logits, dim=1)) / len(batch)
    value_loss = F.mse_loss(value, v_target)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=100.0)
    optimizer.step()

    return TrainStats(
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        mean_value=float(value.mean().item()),
        lr=optimizer.param_groups[0]["lr"],
    )


def attach_winner(samples: list[Sample], winner: int) -> None:
    """Tag every sample with the eventual game winner for the value head."""
    for s in samples:
        s._winner = winner  # noqa: SLF001


def save_checkpoint(network: AGZNet, path: str, optimizer: Adam | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": network.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(network: AGZNet, path: str, optimizer: Adam | None = None) -> None:
    state = torch.load(path, map_location="cpu")
    network.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
