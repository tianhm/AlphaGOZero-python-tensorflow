"""Feature extraction for the AGZ network.

Encodes a Go position as a stack of `n_planes` binary feature planes
(default 17, matching the original repo).

Planes 0-7:   current player's stones over the last 8 moves
Planes 8-15:  opponent's stones over the last 8 moves
Plane 16:     player colour (1 if current player is black, else 0)

If `n_planes < 17`, only the first `n_planes` planes are returned.

Symmetry: 8 dihedral symmetries (4 rotations × 2 reflections) are exposed
via `apply_symmetry` for training augmentation.
"""
from __future__ import annotations

import numpy as np

from .go import Position, STONE_BLACK, opponent  # noqa: F401


def extract_features(position: Position, n_planes: int = 17) -> np.ndarray:
    """Extract features for a Go position.

    The current player is `position.next_player`. We replay the
    position's history on a fresh `Position` so captures are handled
    correctly, then take the last 8 board states from each player's
    perspective.
    """
    n = position.n
    features = np.zeros((17, n, n), dtype=np.float32)

    # Replay history to recover board snapshots after each move.
    replay = Position(board_size=n)
    history_boards: list[tuple[int, np.ndarray]] = [(STONE_BLACK, replay.board.copy())]
    for move in position.history:
        replay.play(move)
        history_boards.append((replay.next_player, replay.board.copy()))

    me = position.next_player
    opp = opponent(me)

    my_boards = [b for p, b in history_boards if p == me]
    opp_boards = [b for p, b in history_boards if p == opp]

    for i in range(8):
        if i < len(my_boards):
            features[i] = (my_boards[-(i + 1)] == me).astype(np.float32)
        if i < len(opp_boards):
            features[8 + i] = (opp_boards[-(i + 1)] == opp).astype(np.float32)

    features[16] = 1.0 if me == STONE_BLACK else 0.0

    return features[:n_planes]


def bulk_extract_features(positions: list[Position], n_planes: int = 17) -> np.ndarray:
    return np.stack([extract_features(p, n_planes) for p in positions], axis=0)


def apply_symmetry(planes: np.ndarray, sym: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries to a (C, H, W) feature map."""
    flipped = np.flip(planes, axis=-1) if sym >= 4 else planes
    k = sym % 4
    return np.rot90(flipped, k=k, axes=(-2, -1))


def mirror_move(move: tuple[int, int], n: int, sym: int) -> tuple[int, int]:
    """Apply the same symmetry to a (row, col) move. PASS is unchanged."""
    if move == (-1, -1):
        return move
    r, c = move
    if sym >= 4:
        c = n - 1 - c
    k = sym % 4
    for _ in range(k):
        r, c = c, n - 1 - r
    return r, c
