"""Head-to-head evaluator: current model vs best model."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from new.game.go import Position, STONE_BLACK, STONE_WHITE
from new.mcts.mcts import run_mcts, select_move
from new.model.network import AGZNet
from new.config import FLAGS
from .selfplay import play_one_game


@dataclass
class EvalResult:
    n_games: int
    wins: int
    losses: int
    draws: int  # not used by area scoring, but reserved

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.n_games)


def evaluate(
    current: AGZNet,
    best: AGZNet,
    device: str,
    n_games: int = FLAGS.n_eval_games,
    n_simulations: int = FLAGS.n_simulations,
    c_puct: float = FLAGS.c_puct,
) -> EvalResult:
    """Play `n_games` head-to-head between current and best.

    The two networks alternate colors. Returns the win rate from the
    current network's perspective.
    """
    wins = 0
    losses = 0
    for i in range(n_games):
        if i % 2 == 0:
            player = STONE_BLACK
            result = _play_match(current, best, player, device, n_simulations, c_puct)
        else:
            player = STONE_WHITE
            result = _play_match(best, current, player, device, n_simulations, c_puct)
        # `result` is the winner from the perspective of the player who
        # was assigned `player`. If current was the assigned player, it
        # wins when result == player.
        if i % 2 == 0:
            if result == STONE_BLACK:
                wins += 1
            else:
                losses += 1
        else:
            if result == STONE_WHITE:
                wins += 1
            else:
                losses += 1
    return EvalResult(n_games=n_games, wins=wins, losses=losses, draws=0)


def _play_match(
    black_net: AGZNet,
    white_net: AGZNet,
    player_to_move: int,
    device: str,
    n_simulations: int,
    c_puct: float,
) -> int:
    """Play a single game using two networks for the two colors.

    Returns the winner (STONE_BLACK or STONE_WHITE), or 0 if no winner
    could be determined (rare).
    """
    pos = Position(board_size=FLAGS.board_size)
    net_for = {STONE_BLACK: black_net, STONE_WHITE: white_net}
    for _ in range(FLAGS.n_moves_per_game):
        if pos.is_terminal():
            break
        net = net_for[pos.next_player]
        net.eval()
        visit_policy, _ = run_mcts(
            pos,
            net,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_alpha=0.0,  # no noise during evaluation
            dirichlet_epsilon=0.0,
            device=device,
        )
        move = select_move(visit_policy, pos, temperature=0.0, n=pos.n)
        pos.play(move)
    if pos.is_terminal():
        return pos.winner()
    return 0
