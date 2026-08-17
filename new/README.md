# AGZ — PyTorch port

A modern, dependency-light reimplementation of the AlphaGo Zero training pipeline (self-play + MCTS + ResNet + policy/value learning) using PyTorch 2.x.

This lives alongside the legacy TF 1.x implementation in the parent folder. The TF 1.x code is **frozen** (see `../AGENTS.md`); this folder is where new work should happen.

## What's here

```
new/
├── README.md          this file
├── AGENTS.md          repo-specific hints for future Kilo sessions
├── requirements.txt   torch>=2.0, numpy, sgf>=0.5
├── config.py          FLAGS + HPS (mirrors the original repo's pattern)
├── main.py            CLI entrypoint (modes: selfplay / train / play / evaluate)
├── game/
│   ├── go.py          Go board, rules, KO, captures, area scoring
│   ├── features.py    17-plane feature extraction + 8-symmetry augmentation
│   └── sgf_wrapper.py SGF parser (sits on top of the `sgf` library)
├── model/
│   ├── resnet.py      Pre-activation residual block
│   └── network.py     AGZNet: ResNet tower + policy head + value head
├── mcts/
│   └── mcts.py        PUCT-guided Monte-Carlo Tree Search
└── training/
    ├── dataset.py     Replay buffer (FIFO, fixed-size)
    ├── selfplay.py    Self-play game generator
    ├── train.py       Train step (policy + value loss + symmetry aug)
    ├── evaluate.py    Head-to-head evaluator
    └── __init__.py    Re-exports
```

## What is implemented

- Go board with full capture, KO, pass, area scoring + komi.
- 17-plane feature encoding (8 history planes per side + player colour).
- 8 dihedral symmetries for training augmentation.
- Pre-activation ResNet with separate policy (`n²+1` logits) and value (tanh scalar) heads.
- PUCT-guided MCTS with Dirichlet noise at the root during self-play.
- Self-play with temperature decay (tau=1 for the first `temp_threshold` moves, then 0).
- Replay buffer (FIFO, capped at `replay_buffer_size`).
- Train step: cross-entropy on policy + MSE on value, gradient clipping.
- Head-to-head evaluation: alternate colors, report win rate, promote if ≥ threshold.
- Interactive CLI play mode (human plays black).

## What is intentionally out of scope

- **Distributed / multi-GPU training.** Single-process on one device.
- **GTP / Sabaki integration.** Use `mode=play` for a CLI game.
- **SGF training data.** Self-play only. The `sgf_wrapper` module is here for future use but not wired into the main pipeline.
- **Tabula rasa from zero.** The pipeline assumes a fresh network; bring your own pretrained checkpoint if you want to start from one.
- **Distributed MCTS** (virtual losses, async batching). Pure-Python serial search.

## Run

```bash
# 1. Install
pip install -r new/requirements.txt

# 2. Generate self-play games (writes to new/replay/replay.pkl)
python new/main.py --mode=selfplay --n_games_per_iter=20

# 3. Train
python new/main.py --mode=train --n_train_steps=500

# 4. Evaluate current vs best (promotes current if win rate ≥ threshold)
python new/main.py --mode=evaluate --n_eval_games=10

# 5. Play against the network
python new/main.py --mode=play
```

## Sane defaults for a single-GPU box

If you have a laptop-class GPU, override defaults before running:

```bash
python new/main.py --mode=selfplay \
    --n_resid_blocks=4 --n_filters=64 \
    --n_simulations=50 --n_games_per_iter=5

python new/main.py --mode=train \
    --batch_size=32 --n_train_steps=200
```

The reference paper uses 19 residual blocks of 256 filters, 1600 simulations per move, and 25,000 self-play games per training iteration. That is *not* realistic on a single GPU; scale down.

## Architectural mapping to the original

| Original (TF 1.x) | This port (PyTorch) |
|---|---|
| `config.py` | `config.py` (same `FLAGS` + `HPS` pattern) |
| `main.py` | `main.py` (modes: selfplay / train / play / evaluate) |
| `Network.py` | `model/network.py` + `model/resnet.py` |
| `model/alphagozero_resnet_full_model.py` | `model/resnet.py` (full pre-activation variant) |
| `model/SelfPlayWorker.py` | `training/selfplay.py` |
| `model/APV_MCTS_tree.py` | `mcts/mcts.py` (no Cython port) |
| `utils/go.py` | `game/go.py` |
| `utils/features.py` | `game/features.py` |
| `utils/load_data_sets.py` | (split into `training/dataset.py` + `game/sgf_wrapper.py`) |
| `preprocess.py` | not used (we self-play instead of SGF-replay) |
