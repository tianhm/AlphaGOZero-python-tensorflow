"""Self-play game generator.

Plays `n_games_per_iter` games using MCTS and the current network, then
produces a flat list of (position, visit_policy, player) samples tagged
with the eventual winner.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import torch

from new.game.go import Position, STONE_BLACK, STONE_WHITE, opponent
from new.game.features import extract_features
from new.mcts.mcts import run_mcts, select_move
from new.model.network import AGZNet
from new.config import FLAGS
from .dataset import Sample


@dataclass
class GameResult:
    winner: int
    n_moves: int
    samples: list[Sample]


def play_one_game(
    network: AGZNet,
    device: str,
    n_simulations: int = FLAGS.n_simulations,
    c_puct: float = FLAGS.c_puct,
    dirichlet_alpha: float = FLAGS.dirichlet_alpha,
    dirichlet_epsilon: float = FLAGS.dirichlet_epsilon,
    max_moves: int = FLAGS.n_moves_per_game,
    temp_threshold: int = FLAGS.temp_threshold,
) -> GameResult:
    """Play one self-play game and return the resulting samples."""
    network.eval()
    pos = Position(board_size=FLAGS.board_size)
    samples: list[Sample] = []

    for move_idx in range(max_moves):
        if pos.is_terminal():
            break

        visit_policy, _root_value = run_mcts(
            pos,
            network,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            device=device,
        )

        # Select move with temperature decay.
        temperature = 1.0 if move_idx < temp_threshold else 0.0
        move = select_move(visit_policy, pos, temperature=temperature, n=pos.n)

        # Record (state, policy, current_player) for this move.
        samples.append(Sample(position=pos.copy(), policy=visit_policy.copy(), player=pos.next_player))

        pos.play(move)

    # If the game didn't terminate naturally (e.g. hit max_moves), score
    # the current position by area regardless.
    winner = pos.score() if pos.is_terminal() else pos.score()
    return GameResult(winner=winner, n_moves=len(samples), samples=samples)


def generate_games(
    network: AGZNet,
    device: str,
    n_games: int = FLAGS.n_games_per_iter,
) -> list[GameResult]:
    """Generate `n_games` self-play games. Pure-Python, single-process."""
    return [play_one_game(network, device) for _ in range(n_games)]


def avg_game_length(results: list[GameResult]) -> float:
    if not results:
        return 0.0
    return statistics.mean(r.n_moves for r in results)


def black_win_rate(results: list[GameResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.winner == STONE_BLACK) / len(results)
