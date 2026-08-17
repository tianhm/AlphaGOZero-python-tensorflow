"""Go board state and rules.

A minimal Go implementation: stone placement, capture, KO, pass, simple
area scoring, and position hashing. Coordinates are (row, col) with
(0, 0) top-left. Stones: 0 = empty, 1 = black, 2 = white.

This is intentionally simpler than the full Go rule set — no positional
superko (simple KO only), no eye-filling heuristics, no handicap. It is
enough for self-play training.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


STONE_EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = 2

PASS = (-1, -1)
KOMI = 7.5


def opponent(color: int) -> int:
    return STONE_WHITE if color == STONE_BLACK else STONE_BLACK


def neighbors(row: int, col: int, size: int) -> Iterator[tuple[int, int]]:
    if row > 0:
        yield row - 1, col
    if row < size - 1:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col < size - 1:
        yield row, col + 1


class Position:
    """Go position with KO, capture, and pass tracking."""

    __slots__ = ("n", "board", "next_player", "ko", "prisoners", "passes", "history")

    def __init__(
        self,
        board_size: int = 19,
        board: np.ndarray | None = None,
        next_player: int = STONE_BLACK,
    ) -> None:
        self.n = board_size
        self.board = board if board is not None else np.zeros((board_size, board_size), dtype=np.int8)
        self.next_player = next_player
        self.ko: tuple[int, int] | None = None
        self.prisoners = {STONE_BLACK: 0, STONE_WHITE: 0}
        self.passes = 0
        self.history: list[tuple[int, int]] = []

    def copy(self) -> "Position":
        new = Position(self.n, self.board.copy(), self.next_player)
        new.ko = self.ko
        new.prisoners = dict(self.prisoners)
        new.passes = self.passes
        new.history = list(self.history)
        return new

    # -------------------------------------------------------------- rules

    def is_legal(self, move: tuple[int, int]) -> bool:
        if move == PASS:
            return True
        r, c = move
        if not (0 <= r < self.n and 0 <= c < self.n):
            return False
        if self.board[r, c] != STONE_EMPTY:
            return False
        if self.ko == move:
            return False
        # Simulate to verify non-suicide.
        test = self.copy()
        return test._try_play(move)

    def _try_play(self, move: tuple[int, int]) -> bool:
        """Try move on a copy, mutating in place. Returns True if legal."""
        color = self.next_player
        opp = opponent(color)
        self.board[move] = color

        # Capture opponent neighbors with no liberties.
        captured: list[tuple[int, int]] = []
        for nr, nc in neighbors(*move, self.n):
            if self.board[nr, nc] == opp and not self._has_liberty((nr, nc)):
                captured.append((nr, nc))
        for cr, cc in captured:
            self._remove_group((cr, cc))
        self.prisoners[color] += len(captured)

        if self._has_liberty(move):
            return True

        # Suicide: undo.
        self.board[move] = STONE_EMPTY
        for cr, cc in captured:
            self._restore_group((cr, cc), opp)
        self.prisoners[color] -= len(captured)
        return False

    def play(self, move: tuple[int, int]) -> bool:
        """Play move if legal. Returns True if played, False if illegal."""
        if move == PASS:
            self.passes += 1
            self.next_player = opponent(self.next_player)
            self.ko = None
            self.history.append(move)
            return True

        r, c = move
        if not (0 <= r < self.n and 0 <= c < self.n):
            return False
        if self.board[r, c] != STONE_EMPTY:
            return False
        if self.ko == move:
            return False

        color = self.next_player
        opp = opponent(color)
        self.board[r, c] = color

        captured: list[tuple[int, int]] = []
        for nr, nc in neighbors(r, c, self.n):
            if self.board[nr, nc] == opp and not self._has_liberty((nr, nc)):
                captured.append((nr, nc))
        for cr, cc in captured:
            self._remove_group((cr, cc))
        # Normalize markers (negative values) to STONE_EMPTY after capture.
        if captured:
            self.board[self.board < 0] = STONE_EMPTY
        self.prisoners[color] += len(captured)

        if not self._has_liberty((r, c)):
            # Suicide should have been caught by is_legal; treat as no-op.
            self.board[r, c] = STONE_EMPTY
            for cr, cc in captured:
                self._restore_group((cr, cc), opp)
            self.board[self.board < 0] = STONE_EMPTY
            self.prisoners[color] -= len(captured)
            return False

        # KO: only set when exactly one stone captured and the placed stone
        # is a singleton.
        if len(captured) == 1 and self._group_size((r, c)) == 1:
            self.ko = captured[0]
        else:
            self.ko = None

        self.next_player = opp
        self.passes = 0
        self.history.append(move)
        return True

    # -------------------------------------------------------------- helpers

    def _has_liberty(self, start: tuple[int, int]) -> bool:
        color = self.board[start]
        if color == STONE_EMPTY:
            return True
        stack = [start]
        visited = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors(r, c, self.n):
                if self.board[nr, nc] == STONE_EMPTY:
                    return True
                if self.board[nr, nc] == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return False

    def _group_size(self, start: tuple[int, int]) -> int:
        color = self.board[start]
        if color == STONE_EMPTY:
            return 0
        stack = [start]
        visited = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors(r, c, self.n):
                if self.board[nr, nc] == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return len(visited)

    def _remove_group(self, start: tuple[int, int]) -> None:
        """Remove a group, marking cells with `-color` so undo can find them."""
        color = self.board[start]
        stack = [start]
        while stack:
            r, c = stack.pop()
            if self.board[r, c] != color:
                continue
            self.board[r, c] = -color  # marker (negative == empty for this group)
            for nr, nc in neighbors(r, c, self.n):
                if self.board[nr, nc] == color:
                    stack.append((nr, nc))

    def _restore_group(self, start: tuple[int, int], color: int) -> None:
        """Restore a group previously removed via `_remove_group`."""
        stack = [start]
        while stack:
            r, c = stack.pop()
            if self.board[r, c] != -color:
                continue
            self.board[r, c] = color
            for nr, nc in neighbors(r, c, self.n):
                if self.board[nr, nc] == -color:
                    stack.append((nr, nc))

    # -------------------------------------------------------------- queries

    def legal_moves(self, include_pass: bool = True) -> list[tuple[int, int]]:
        moves: list[tuple[int, int]] = []
        for r in range(self.n):
            for c in range(self.n):
                if self.is_legal((r, c)):
                    moves.append((r, c))
        if include_pass:
            moves.append(PASS)
        return moves

    def is_terminal(self) -> bool:
        if self.passes >= 2:
            return True
        if len(self.history) >= self.n * self.n * 3:  # safety cutoff
            return True
        return False

    def winner(self) -> int:
        """Area scoring + komi. Returns STONE_BLACK or STONE_WHITE.

        Raises ValueError if the game is not over. Use `score()` to force
        scoring regardless of terminal status.
        """
        if not self.is_terminal():
            raise ValueError("Game not over")
        return self.score()

    def score(self) -> int:
        """Area scoring + komi. Returns STONE_BLACK or STONE_WHITE."""
        black = int(np.sum(self.board == STONE_BLACK))
        white = int(np.sum(self.board == STONE_WHITE))
        return STONE_BLACK if black > white + KOMI else STONE_WHITE

    def hash(self) -> bytes:
        return self.board.tobytes() + bytes([self.next_player])
