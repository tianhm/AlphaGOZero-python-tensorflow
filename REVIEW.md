# Closing Review — AlphaGo Zero (python-tensorflow)

A honest take for someone using this repo as their entry point into AI.

## What this repo actually is

A 2017-era research reimplementation of DeepMind's AlphaGo Zero, in TensorFlow 1.x, by Brain Lee and several contributors. It reproduces the AGZ pipeline: supervised training on human SGFs, self-play with MCTS, evaluation against a best-model pool. It is **not** a polished library, not a product, and not a teaching scaffold — it is a single-author research codebase pinned to a now-ancient toolchain.

If you came here from "I want to learn AI", this is closer to reading a senior engineer's working notebook than to following a tutorial.

## Honest assessment of the codebase

**Strengths worth reading carefully**
- The end-to-end AGZ data flow is all here, in one place: SGF → feature planes → CNN+ResNet → policy + value → MCTS → self-play → evaluation. Few modern repos give you the whole loop in 2,000 lines.
- `utils/` (Go game logic, SGF parsing, feature extraction) is a tight reimplementation of the relevant MuGo pieces. Good code to study.
- `model/SelfPlayWorker.py` and `model/APV_MCTS_tree.py` show classic MCTS + neural-network guidance in pure Python. Educational.
- `Network.py` is a clean illustration of TF 1.x graph/session plumbing — including the ugly bits (`var_to_save`, BN variable handling, the `tf.placeholder` fights). Knowing this *helps* you appreciate TF 2.x.
- The config layer (`config.py`: `FLAGS` + `HPS`) is simple and conventional.

**Weaknesses you will hit**
- **It will not run on a modern setup as-is.** The README warns TF ≥ 1.5 cannot load the shipped checkpoints. Python 3.6 is end-of-life. `protobuf==3.1.0.post1` and `six==1.10.0` are pinned because newer versions break the model files. Expect to spend hours just getting an import to work.
- **No build, no tests, no CI.** The "test" mode is "`train` on the test split", not a unit-test suite. `utils/tests/` exists but has no runner.
- **Heavy GPU + data requirements.** The reference workflow wants hundreds of GB of SGFs, a GPU, and days of training. The pretraining checkpoint download is a separate Google Drive link.
- **Legacy TF idioms everywhere.** `tf.placeholder`, `tf.Session`, `tf.global_variables_initializer()`, explicit `feed_dict` — all of which have been replaced by `tf.function` / Keras in 2019.
- **Dead/experimental code mixed with runtime code.** `model/APV_MCTS_C.pyx` (Cython) is not built and not imported; `support/` is bundled upstream reference code (MuGo, RocAlphaGo, ...) that is not part of the runtime but is committed in-tree.
- **Typos and surprises baked in.** `dest='gpt_policy'` for the `--gtp_policy` flag, `firts_time=` in `preprocess.py`, the hard-coded anaconda shebang on `main.py:1`, the README's broken `—-gtp_poliy` command. None of these are bugs you'd want to "fix" — they are part of the code you're reading.

## Is this a good place to "start AI"?

**As a hands-on environment to run: no.** You'll burn days on environment setup before you see a single training step.

**As a code-reading curriculum to learn the deep classic: yes, with structure.** Used as a textbook, this repo is one of the best single-file-window views of the AGZ algorithm you can find. The trick is to read it in a specific order, not to run it.

## Recommended reading order

If you want to actually learn from this repo without trying to run it, go in this order:

1. **`README.md`** + the original AlphaGo Zero paper (Nature, 2017). Read them side by side.
2. **`config.py`** (65 lines). All the knobs in one place.
3. **`main.py`**. The mode dispatcher (`fn = {'train': ..., 'gtp': ..., ...}` at line 225) tells you the whole project in 5 lines.
4. **`utils/`** — `go.py`, `features.py`, `sgf_wrapper.py`, `strategies.py`. Game logic + feature extraction. This is the most transferable skill in the repo.
5. **`Network.py`** — TF 1.x graph wiring. Read it as "what did TF 1 force you to write by hand?".
6. **`model/APV_MCTS_tree.py`** — pure-Python MCTS. This is the algorithmic heart.
7. **`model/SelfPlayWorker.py`** — the self-play loop. Read this once you've digested MCTS.
8. **`model/alphagozero_resnet_full_model.py`** — the architecture. Variant selection via `--model_type {original,elu,full}` is in `Network.py:70-75`.
9. **`AGENTS.md`** (the file we just wrote) for the gotchas.

Skip `support/` entirely unless you want to compare this repo to its ancestors.

## What to do instead of running this

If your goal is "get into AI" in 2026, the faster paths are:

- **PyTorch + a single modern tutorial** (Karpathy's "Zero to Hero" series, the PyTorch quickstart). TF 1.x is a dead end professionally.
- **Stable-Baselines3** or **CleanRL** for reinforcement learning with modern APIs. They replace the entire `selfplay.py` + `APV_MCTS.py` ecosystem in ~50 lines.
- **Re-implement AGZ yourself in PyTorch** as a learning exercise. You will understand the algorithm 10× better than by reading this repo, and you will have something that actually runs on a single GPU.
- **Read the code in this repo as a reference**, not as a target. The patterns here (MCTS, self-play, feature planes, dual policy+value heads) all show up in modern AlphaZero-style projects (Leela Chess, KataGo, MuZero reimplementations).

## What we did this session

- Wrote `AGENTS.md` at the repo root: a compact, repo-specific instruction file for future Kilo sessions covering stack lock-in, entrypoints, workflow, architecture, quirks, tests, and conventions. It is designed to answer "would an agent likely miss this without help?" for every line.

## TL;DR

This repo is a 2017 research notebook, not a learner-facing AI starting point. It is **excellent material to read** and **terrible material to run**. Treat it as a textbook: read it in the order above, hand-port the pieces you care about to PyTorch, and use modern frameworks for anything you actually want to train. Keep `AGENTS.md` open when you come back — it will save the next agent (or future you) hours of rediscovering the gotchas.
