"""Configuration for the PyTorch AGZ port.

Mirrors the original repo's `FLAGS` + `HPS` pattern so existing notes
still apply. New hyperparameters go here, not as module-level constants.

`FLAGS` and `HPS` are lazy: they are first parsed on attribute access,
not on import. This lets `python -m unittest new.tests.test_smoke`
import this module without consuming the unittest runner's argv.
"""
from __future__ import annotations

import sys
from collections import namedtuple
from dataclasses import dataclass


def build_parser():
    from argparse import ArgumentParser
    p = ArgumentParser(description="AlphaGo Zero — PyTorch port")

    # Network
    p.add_argument("--board_size", type=int, default=19)
    p.add_argument("--n_planes", type=int, default=17)
    p.add_argument("--n_resid_blocks", type=int, default=19)
    p.add_argument("--n_filters", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_train_steps", type=int, default=1000)
    p.add_argument("--lr_decay_steps", type=int, default=100_000)

    # Self-play
    p.add_argument("--n_games_per_iter", type=int, default=100)
    p.add_argument("--n_moves_per_game", type=int, default=250)
    p.add_argument("--n_simulations", type=int, default=800)
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--dirichlet_alpha", type=float, default=0.03)
    p.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    p.add_argument("--temp_threshold", type=int, default=30,
                   help="Move index after which temperature drops to 0.")

    # Replay / evaluation
    p.add_argument("--replay_buffer_size", type=int, default=200_000)
    p.add_argument("--n_eval_games", type=int, default=20)
    p.add_argument("--eval_win_rate_threshold", type=float, default=0.55)

    # Paths
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--save_dir", default="./new/savedmodels")
    p.add_argument("--replay_dir", default="./new/replay")
    p.add_argument("--log_dir", default="./new/logs")

    # Mode
    p.add_argument("--mode", default="selfplay",
                   choices=["selfplay", "train", "play", "evaluate"])

    # Device
    p.add_argument("--device", default="cuda" if _cuda_available() else "cpu")

    return p


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


parser = build_parser()


@dataclass
class _Parsed:
    flags: object
    hps: object


_cached: _Parsed | None = None


def parse_args(argv: list[str] | None = None) -> _Parsed:
    """Parse argv (or sys.argv[1:]) and cache the result."""
    global _cached
    if _cached is not None and argv is None:
        return _cached
    if argv is None:
        argv = sys.argv[1:]
    flags = parser.parse_args(argv)
    HParams = namedtuple("HParams", [
        "board_size", "n_planes", "n_resid_blocks", "n_filters",
        "n_actions", "lr", "weight_decay", "name",
    ])
    hps = HParams(
        board_size=flags.board_size,
        n_planes=flags.n_planes,
        n_resid_blocks=flags.n_resid_blocks,
        n_filters=flags.n_filters,
        n_actions=flags.board_size ** 2 + 1,
        lr=flags.lr,
        weight_decay=flags.weight_decay,
        name="agz",
    )
    _cached = _Parsed(flags=flags, hps=hps)
    return _cached


def __getattr__(name: str):
    """Lazy attribute access: parse_args() is triggered on first access."""
    if name == "FLAGS":
        return parse_args().flags
    if name == "HPS":
        return parse_args().hps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
