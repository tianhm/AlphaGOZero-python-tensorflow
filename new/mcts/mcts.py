"""PUCT-guided Monte-Carlo Tree Search.

Tree nodes store (N, W, P, children). The search runs `n_simulations`
playouts from the root and returns a visit-count distribution over
actions, which is the policy target for training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from new.game.go import Position, opponent, PASS, STONE_EMPTY
from new.game.features import extract_features
from new.model.network import AGZNet


@dataclass
class Node:
    """A single MCTS node."""

    prior: float
    n_visits: int = 0
    total_value: float = 0.0
    is_expanded: bool = False
    children: dict[tuple[int, int], "Node"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.total_value / self.n_visits if self.n_visits > 0 else 0.0

    def expand(self, policy: np.ndarray, position: Position) -> None:
        """Add one child per legal move with priors from `policy`.

        `policy` is a length-`n_actions` vector indexed by (row * n + col),
        with the last entry reserved for the pass move.
        """
        n = position.n
        for move in position.legal_moves(include_pass=True):
            if move == PASS:
                idx = n * n
            else:
                r, c = move
                idx = r * n + c
            self.children[move] = Node(prior=float(policy[idx]))
        self.is_expanded = True

    def select(self, c_puct: float) -> tuple[tuple[int, int], "Node"]:
        """Select the child with the highest PUCT score."""
        sqrt_total = math.sqrt(max(1, self.n_visits))
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in self.children.items():
            exploit = child.q_value
            explore = c_puct * child.prior * sqrt_total / (1 + child.n_visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def backup(self, value: float) -> None:
        self.n_visits += 1
        self.total_value += value


def _add_dirichlet_noise(policy: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * noise


def run_mcts(
    position: Position,
    network: AGZNet,
    n_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Run MCTS from `position`. Returns (visit_count_policy, root_value).

    `visit_count_policy` is a length-`n_actions` vector proportional to
    the visit counts of each child of the root.
    """
    root = Node(prior=1.0)
    n = position.n
    n_actions = n * n + 1

    # Initial expansion.
    features = extract_features(position, n_planes=network.n_planes)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        policy_logits, value = network(x)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    # Mask out illegal moves.
    legal_mask = np.zeros(n_actions, dtype=np.float32)
    for move in position.legal_moves(include_pass=True):
        if move == PASS:
            legal_mask[n * n] = 1.0
        else:
            r, c = move
            legal_mask[r * n + c] = 1.0
    policy = policy * legal_mask
    s = policy.sum()
    if s > 0:
        policy /= s
    else:
        # Fallback: uniform over legal moves.
        policy = legal_mask / max(1, legal_mask.sum())

    # Add Dirichlet noise at root (only during self-play, not evaluation).
    if dirichlet_alpha > 0 and dirichlet_epsilon > 0:
        policy = _add_dirichlet_noise(policy, dirichlet_alpha, dirichlet_epsilon)
        policy = policy * legal_mask
        s = policy.sum()
        if s > 0:
            policy /= s

    root.expand(policy, position)

    for _ in range(n_simulations):
        node = root
        path = [node]
        scratch = position.copy()

        # Selection.
        while node.is_expanded and not scratch.is_terminal():
            move, node = node.select(c_puct)
            path.append(node)
            scratch.play(move)

        # Expansion at leaf.
        if not scratch.is_terminal():
            features = extract_features(scratch, n_planes=network.n_planes)
            with torch.no_grad():
                x = torch.from_numpy(features).unsqueeze(0).to(device)
                policy_logits, leaf_value = network(x)
                leaf_policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

            legal_mask = np.zeros(n_actions, dtype=np.float32)
            for m in scratch.legal_moves(include_pass=True):
                if m == PASS:
                    legal_mask[n * n] = 1.0
                else:
                    r, c = m
                    legal_mask[r * n + c] = 1.0
            leaf_policy = leaf_policy * legal_mask
            s = leaf_policy.sum()
            if s > 0:
                leaf_policy /= s
            else:
                leaf_policy = legal_mask / max(1, legal_mask.sum())

            node.expand(leaf_policy, scratch)
            value = float(leaf_value.item())
        else:
            # Terminal: assign win/loss/0 from the parent's perspective.
            winner = scratch.winner()
            value = 1.0 if winner == scratch.next_player else -1.0
            # Wait — `next_player` already advanced past the last play, so
            # the side that just moved was `opponent(scratch.next_player)`.
            last_player = opponent(scratch.next_player)
            value = 1.0 if winner == last_player else -1.0

        # Backup: value is from the perspective of the player who just moved.
        # At the root, before any move, the value is from the perspective of
        # `position.next_player`.
        for child in reversed(path):
            child.backup(value)
            value = -value

    # Compose visit-count policy.
    visits = np.zeros(n_actions, dtype=np.float32)
    for move, child in root.children.items():
        if move == PASS:
            visits[n * n] = child.n_visits
        else:
            r, c = move
            visits[r * n + c] = child.n_visits

    s = visits.sum()
    if s > 0:
        visits /= s
    return visits, root.q_value


def select_move(
    visit_policy: np.ndarray,
    position: Position,
    temperature: float,
    n: int = 19,
) -> tuple[int, int]:
    """Sample a move from the visit distribution with temperature.

    temperature == 0 → argmax (greedy). temperature == 1 → proportional.
    """
    if temperature == 0:
        idx = int(np.argmax(visit_policy))
    else:
        # Apply temperature and renormalize.
        tempered = np.power(visit_policy + 1e-12, 1.0 / temperature)
        s = tempered.sum()
        if s > 0:
            tempered /= s
        else:
            tempered = np.ones_like(tempered) / len(tempered)
        idx = int(np.random.choice(len(tempered), p=tempered))

    if idx == n * n:
        return PASS
    return idx // n, idx % n
