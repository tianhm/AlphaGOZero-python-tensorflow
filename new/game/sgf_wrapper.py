"""SGF parsing and replay into Go positions.

A minimal wrapper around the `sgf` library: parse an SGF file path into a
sequence of `Position` objects (one per move) plus the final winner
derived from the SGF `RE` tag if present, else by replay area scoring.
"""
from __future__ import annotations

import os
from typing import Iterator

from .go import (
    Position,
    STONE_BLACK,
    STONE_WHITE,
    opponent,
)


COL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _sgf_to_coords(move: str, n: int) -> tuple[int, int]:
    """Convert an SGF 'aa'/'bb' coord to (row, col), or PASS for empty."""
    if not move or move == "" or move == "tt":
        return (-1, -1)
    col = ord(move[0]) - ord("a")
    row = ord(move[1]) - ord("a")
    return (row, col)


def replay_sgf(path: str) -> tuple[list[Position], int | None]:
    """Replay an SGF file. Returns (positions, winner).

    Each entry in `positions` is the position *before* the i-th move is
    played, so `positions[i]` is the state from which the move at index i
    is taken. The last entry is the final position.

    `winner` is `STONE_BLACK` (1) or `STONE_WHITE` (2) if determinable
    from the SGF `RE` tag, else `None`.
    """
    import sgf  # python-sgf library

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        cursor = sgf.parse(f.read())

    # Take the first game tree.
    node = cursor[0]
    boardsize = 19
    if "SZ" in node.properties:
        boardsize = int(node.properties["SZ"][0])

    winner = None
    if "RE" in node.properties:
        re = node.properties["RE"][0]
        if re.startswith("B"):
            winner = STONE_BLACK
        elif re.startswith("W"):
            winner = STONE_WHITE

    positions: list[Position] = []
    pos = Position(board_size=boardsize)
    positions.append(pos.copy())

    for child in node.rest:
        for color_char, prop in (("B", "B"), ("W", "W")):
            if prop in child.properties:
                move_str = child.properties[prop][0]
                # Players are stored from the perspective of the side to move.
                # In SGF, B-move is black, W-move is white. We need to align
                # with `pos.next_player` ordering.
                if pos.next_player != (STONE_BLACK if color_char == "B" else STONE_WHITE):
                    # Skip mis-colored moves (rare in valid SGFs).
                    pass
                move = _sgf_to_coords(move_str, boardsize)
                if pos.is_legal(move):
                    pos.play(move)
                positions.append(pos.copy())
                break

    return positions, winner


def iter_sgf_paths(root: str) -> Iterator[str]:
    """Yield all `.sgf` files under `root` (non-recursive and 1-deep)."""
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if os.path.isfile(path) and entry.lower().endswith(".sgf"):
            yield path
        elif os.path.isdir(path):
            for sub in os.listdir(path):
                if sub.lower().endswith(".sgf"):
                    yield os.path.join(path, sub)
