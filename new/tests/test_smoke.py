"""Unit tests for the new/ PyTorch AGZ port.

Run with: python -m unittest new.tests.test_smoke
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from new.game.go import Position, STONE_BLACK, STONE_WHITE, PASS, opponent
from new.game.features import extract_features, apply_symmetry, mirror_move
from new.model.network import AGZNet, count_parameters
from new.mcts.mcts import run_mcts, select_move, Node
from new.training.dataset import ReplayBuffer, Sample


class TestGo(unittest.TestCase):
    def test_empty_board_legal_moves(self):
        pos = Position(board_size=9)
        moves = pos.legal_moves(include_pass=True)
        self.assertEqual(len(moves), 9 * 9 + 1)

    def test_alternation(self):
        pos = Position(board_size=9)
        pos.play((3, 3))
        self.assertEqual(pos.next_player, STONE_WHITE)
        pos.play((3, 4))
        self.assertEqual(pos.next_player, STONE_BLACK)

    def test_pass_terminates(self):
        pos = Position(board_size=9)
        pos.play(PASS)
        pos.play(PASS)
        self.assertTrue(pos.is_terminal())

    def test_capture(self):
        pos = Position(board_size=9)
        pos.play((4, 4))  # BLACK
        pos.play((3, 4))  # WHITE
        pos.play((0, 0))  # BLACK
        pos.play((5, 4))  # WHITE
        pos.play((0, 1))  # BLACK
        pos.play((4, 3))  # WHITE
        pos.play((0, 2))  # BLACK
        pos.play((4, 5))  # WHITE - captures (4,4)
        self.assertEqual(pos.board[4, 4], 0)
        self.assertEqual(pos.prisoners[STONE_WHITE], 1)
        self.assertEqual(int((pos.board < 0).sum()), 0)

    def test_self_capture_suicide(self):
        pos = Position(board_size=9)
        pos.play((4, 4))  # BLACK
        pos.play((3, 4))  # WHITE
        pos.play((0, 0))  # BLACK
        pos.play((5, 4))  # WHITE
        pos.play((0, 1))  # BLACK
        pos.play((4, 3))  # WHITE
        pos.play((0, 2))  # BLACK
        # (4,5) is empty; if WHITE plays there, the (4,4) BLACK has no
        # liberties and gets captured, not suicide.
        pos.play((4, 5))  # WHITE
        # After capture, BLACK at (4,4) is removed.
        self.assertEqual(pos.board[4, 4], 0)

    def test_no_marker_leak(self):
        pos = Position(board_size=9)
        for r in range(9):
            for c in range(9):
                pos.play((r, c)) if (r * 9 + c) % 2 == 0 else pos.play(PASS)
        # After many plays, no cell should be a negative marker.
        self.assertEqual(int((pos.board < 0).sum()), 0)


class TestFeatures(unittest.TestCase):
    def test_extract_shape(self):
        pos = Position(board_size=9)
        feats = extract_features(pos, n_planes=17)
        self.assertEqual(feats.shape, (17, 9, 9))
        self.assertEqual(feats.dtype, np.float32)

    def test_symmetry_inverse(self):
        pos = Position(board_size=9)
        pos.play((3, 3))
        pos.play((3, 4))
        feats = extract_features(pos, n_planes=17)
        for s in range(8):
            # Applying any dihedral symmetry 4 times returns to identity.
            roundtrip = feats
            for _ in range(4):
                roundtrip = apply_symmetry(roundtrip, s)
            np.testing.assert_allclose(feats, roundtrip, atol=1e-6)


class TestNetwork(unittest.TestCase):
    def test_forward_shape(self):
        net = AGZNet(n_planes=17, n_actions=82, n_resid_blocks=2, n_filters=32, board_size=9)
        x = torch.zeros(4, 17, 9, 9)
        net.eval()
        with torch.no_grad():
            pol, val = net(x)
        self.assertEqual(pol.shape, (4, 82))
        self.assertEqual(val.shape, (4, 1))
        self.assertGreater(count_parameters(net), 0)

    def test_does_not_nan(self):
        net = AGZNet(n_planes=17, n_actions=362, n_resid_blocks=2, n_filters=32, board_size=19)
        x = torch.randn(2, 17, 19, 19)
        net.eval()
        with torch.no_grad():
            pol, val = net(x)
        self.assertFalse(torch.isnan(pol).any())
        self.assertFalse(torch.isnan(val).any())


class TestMCTS(unittest.TestCase):
    def test_visit_policy_sums_to_one(self):
        net = AGZNet(n_planes=17, n_actions=362, n_resid_blocks=2, n_filters=32, board_size=19)
        pos = Position(board_size=19)
        visit_policy, _ = run_mcts(
            pos, net, n_simulations=10, c_puct=1.5,
            dirichlet_alpha=0.3, dirichlet_epsilon=0.25, device="cpu",
        )
        self.assertEqual(visit_policy.shape, (362,))
        self.assertAlmostEqual(visit_policy.sum(), 1.0, places=5)


class TestReplayBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        buf = ReplayBuffer(max_size=10)
        for i in range(5):
            buf.add(Sample(position=Position(9), policy=np.zeros(82), player=STONE_BLACK))
        self.assertEqual(len(buf), 5)
        batch = buf.sample_batch(3)
        self.assertEqual(len(batch), 3)

    def test_max_size_eviction(self):
        buf = ReplayBuffer(max_size=3)
        for i in range(10):
            buf.add(Sample(position=Position(9), policy=np.zeros(82), player=STONE_BLACK))
        self.assertEqual(len(buf), 3)


if __name__ == "__main__":
    unittest.main()
