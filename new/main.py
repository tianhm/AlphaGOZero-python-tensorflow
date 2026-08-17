"""Main entry point for the PyTorch AGZ port.

Modes:
    selfplay    Generate self-play games and dump to the replay buffer
    train       Train the network from the replay buffer
    play        Play against the network (interactive CLI)
    evaluate    Play current vs best model and report win rate

Typical workflow:
    python new/main.py --mode=selfplay --n_games_per_iter=50
    python new/main.py --mode=train     --n_train_steps=1000
    python new/main.py --mode=evaluate  --n_eval_games=20
"""
from __future__ import annotations

import os
import sys

# Make `new` importable whether run as `python new/main.py` (script dir on
# sys.path) or `python -m new.main` (parent dir on sys.path).
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from new.config import FLAGS, HPS, parse_args
from new.model.network import AGZNet
from new.training.dataset import ReplayBuffer
from new.training.selfplay import generate_games
from new.training.train import train_step, attach_winner, save_checkpoint, load_checkpoint, build_optimizer
from new.training.evaluate import evaluate
from new.game.go import Position, STONE_BLACK, STONE_WHITE
from new.mcts.mcts import run_mcts, select_move
from new.game.features import extract_features


def _build_network(device: str) -> AGZNet:
    net = AGZNet(
        n_planes=HPS.n_planes,
        n_actions=HPS.n_actions,
        n_resid_blocks=HPS.n_resid_blocks,
        n_filters=HPS.n_filters,
        board_size=HPS.board_size,
    ).to(device)
    return net


def _load_or_init(net: AGZNet, path: str) -> None:
    if os.path.exists(path):
        load_checkpoint(net, path)
        print(f"Loaded checkpoint from {path}")
    else:
        print(f"No checkpoint at {path}, starting from scratch")


def _replay_path() -> str:
    os.makedirs(FLAGS.replay_dir, exist_ok=True)
    return os.path.join(FLAGS.replay_dir, "replay.pkl")


def _best_model_path() -> str:
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    return os.path.join(FLAGS.save_dir, "best.pt")


def _current_model_path() -> str:
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    return os.path.join(FLAGS.save_dir, "current.pt")


def mode_selfplay(device: str) -> None:
    net = _build_network(device)
    _load_or_init(net, _current_model_path())

    buffer = ReplayBuffer(max_size=FLAGS.replay_buffer_size)
    if os.path.exists(_replay_path()):
        buffer.load(_replay_path())
        print(f"Loaded replay buffer with {len(buffer)} samples")

    games = generate_games(net, device, n_games=FLAGS.n_games_per_iter)
    for g in games:
        attach_winner(g.samples, g.winner)
        buffer.extend(g.samples)
        print(f"  game winner={g.winner} moves={g.n_moves}")

    buffer.save(_replay_path())
    print(f"Replay buffer: {len(buffer)} samples")
    save_checkpoint(net, _current_model_path())
    print(f"Saved current model to {_current_model_path()}")


def mode_train(device: str) -> None:
    net = _build_network(device)
    _load_or_init(net, _current_model_path())
    optimizer = build_optimizer(net)

    buffer = ReplayBuffer(max_size=FLAGS.replay_buffer_size)
    if os.path.exists(_replay_path()):
        buffer.load(_replay_path())
    print(f"Replay buffer: {len(buffer)} samples")

    if len(buffer) < FLAGS.batch_size:
        print("Not enough samples to train. Run --mode=selfplay first.")
        return

    for step in range(FLAGS.n_train_steps):
        batch = buffer.sample_batch(FLAGS.batch_size)
        stats = train_step(
            net,
            optimizer,
            batch,
            n_planes=HPS.n_planes,
            n_actions=HPS.n_actions,
            board_size=HPS.board_size,
            device=device,
        )
        if step % 50 == 0:
            print(
                f"step={step} policy_loss={stats.policy_loss:.4f} "
                f"value_loss={stats.value_loss:.4f} mean_value={stats.mean_value:.3f}"
            )

    save_checkpoint(net, _current_model_path(), optimizer=optimizer)
    print(f"Saved checkpoint to {_current_model_path()}")


def mode_evaluate(device: str) -> None:
    current = _build_network(device)
    best = _build_network(device)

    _load_or_init(current, _current_model_path())
    if os.path.exists(_best_model_path()):
        load_checkpoint(best, _best_model_path())
    else:
        # First iteration: current becomes best.
        save_checkpoint(current, _best_model_path())
        print("No best model yet; saving current as best. Run training first.")
        return

    result = evaluate(
        current,
        best,
        device,
        n_games=FLAGS.n_eval_games,
        n_simulations=FLAGS.n_simulations,
        c_puct=FLAGS.c_puct,
    )
    print(
        f"Eval: current {result.wins}/{result.n_games} wins "
        f"({result.win_rate:.0%}) vs best"
    )
    if result.win_rate >= FLAGS.eval_win_rate_threshold:
        save_checkpoint(current, _best_model_path())
        print("Promoted current to best.")
    else:
        print("Current did not beat best; best unchanged.")


def mode_play(device: str) -> None:
    """Simple interactive CLI: human vs network.

    The human plays black. The network plays white. Coordinates are
    typed as `row col` (e.g. `3 3`), or `pass`, or `quit`.
    """
    net = _build_network(device)
    _load_or_init(net, _current_model_path())

    pos = Position(board_size=FLAGS.board_size)
    print("You are Black. Type `row col` to play (e.g. '3 3'), or `pass` / `quit`.")
    while not pos.is_terminal():
        print(f"\nMove {len(pos.history)} — {'Black' if pos.next_player == STONE_BLACK else 'White'} to play")
        print(pos.board)
        if pos.next_player == STONE_BLACK:
            cmd = input("> ").strip().lower()
            if cmd == "quit":
                return
            if cmd == "pass":
                move = (-1, -1)
            else:
                try:
                    r, c = cmd.split()
                    move = (int(r), int(c))
                except ValueError:
                    print("Invalid. Use 'row col', 'pass', or 'quit'.")
                    continue
        else:
            visit_policy, _ = run_mcts(
                pos,
                net,
                n_simulations=FLAGS.n_simulations,
                c_puct=FLAGS.c_puct,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
                device=device,
            )
            move = select_move(visit_policy, pos, temperature=0.0, n=pos.n)
            print(f"Network plays: {move}")

        if not pos.play(move):
            print("Illegal move.")
            continue

    print(f"\nFinal board:\n{pos.board}")
    print(f"Winner: {pos.winner()}")


def main() -> None:
    device = FLAGS.device
    print(f"Mode: {FLAGS.mode} | Device: {device}")

    if FLAGS.mode == "selfplay":
        mode_selfplay(device)
    elif FLAGS.mode == "train":
        mode_train(device)
    elif FLAGS.mode == "play":
        mode_play(device)
    elif FLAGS.mode == "evaluate":
        mode_evaluate(device)
    else:
        print(f"Unknown mode: {FLAGS.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
