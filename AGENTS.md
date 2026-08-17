# AGENTS.md

AlphaGo Zero (TF 1.x) recreation by Brain Lee et al. Single-process Python research code. There is no build system, no CI, no test runner, and no linter — every "command" is just running a script.

## Stack & compatibility

- **Python 3.6 / TensorFlow 1.4** (GPU build). README warns TF ≥ 1.5 cannot load the shipped checkpoints.
- Pinned deps in `requirements.txt`: `protobuf==3.1.0.post1`, `six==1.10.0`, `sgf==0.5`, `cython`, `numpy`, `argh`, `daiquiri`, `pygtp>=0.4`, `profilehooks`.
- Do **not** "modernize" to TF 2.x or Python 3.10+ — the model files and checkpoints will break. Treat the codebase as frozen legacy.

## Entrypoints

Two CLI scripts at repo root; both use `argh` + `argparse`:

| Script | Purpose | Subcommands |
|---|---|---|
| `main.py` | train / self-play / play / evaluate | `--mode=train\|selfplay\|gtp\|test` |
| `preprocess.py` | SGF → TF chunk datasets | `preprocess`, `tfrecord` |

Mode dispatch is in `main.py:225-232` (a dict of lambdas keyed on `FLAGS.MODE`). Default mode is `train`.

## Typical workflow

```bash
# 1. Get SGF data (kgs-4dan)
cd data/downloads && ./download.sh   # writes into data/SGFs/

# 2. Convert SGFs → gzipped feature chunks in processed_data/
python preprocess.py preprocess ./data/SGFs/kgs-*

# 3. (optional) Drop a pretrained checkpoint at savedmodels/large20/

# 4. Train or play
python main.py --mode=train
python main.py --mode=gtp --gtp_policy=greedypolicy --model_path=./savedmodels/large20
python main.py --mode=selfplay
python main.py --mode=test

# 5. Crash-safe training (auto-restart on exit code != 0)
./auto_restart.sh
```

`preprocess.py` writes a single `train0.chunk.gz` plus `test.chunk.gz` by default (see `one_big_training_chunck=True`). `main.py` then reads every `train*.chunk.gz` matching `TRAINING_CHUNK_RE` in `config.py:149`.

## Architecture

- **`Network.py`** — TF 1.x graph wrapper. Builds graph, owns `tf.Session`, holds `Saver`. Imports one of three models from `model/` (selected via `flags.model`).
- **`model/alphagozero_resnet_model.py`** — original AGZ residual block.
- **`model/alphagozero_resnet_elu_model.py`** — ELU variant.
- **`model/alphagozero_resnet_full_model.py`** — default (`--model_type=full`); full pre-activation residual net.
- **`model/SelfPlayWorker.py`** — MCTS self-play pipeline + best-model evaluation.
- **`model/APV_MCTS_tree.py`** — pure-Python MCTS (used at runtime).
- **`model/APV_MCTS_C.pyx`** — Cython MCTS. **Not compiled**; `SelfPlayWorker` imports the `.py` version. Treat the `.pyx` as dead/experimental unless explicitly asked to build it.
- **`model/resnet_model.py`** — generic ResNet building blocks shared by the three variants.
- **`utils/`** — Go game logic, SGF parsing, feature extraction, strategies, GTP wrapper. Per `utils/README.md`, this folder is a reengineered fork of MuGo.
- **`config.py`** — argparse `FLAGS` + `HPS` namedtuple. Tweak here to change defaults (batch size, residual units, self-play counts, etc.).
- **`support/`** — bundled upstream reference projects (MuGo, RocAlphaGo, mini-Alpha-Go, go-NN, resnet-tensorflow). **Not part of the runtime.** Read-only material.
- **`elo/`** — Elo rating utility (standalone; not wired into `main.py`).
- **`savedmodels/large20/`** — canonical checkpoint directory. `--model_path` defaults here.

## Quirks & gotchas

- `main.py:1` has a hard-coded shebang to the original author's anaconda env (`/home/hangyu5/anaconda2/envs/py3dl/bin/python`). Override per invocation; do not edit it as a fix.
- `config.py:24` uses `dest='gpt_policy'` (typo for `gtp_policy`). The CLI flag is still `--gtp_policy`. Don't "fix" without grepping for `gpt_policy` first.
- `Network.py:77` has a known-ugly `var_to_save` expression that double-includes BN variables. The inline comment notes TF 1.7.0 requires the second clause to be commented out.
- GPU memory is hard-capped at 40 % (`Network.py:40`); multi-GPU is configured via `--n_gpu` but actually exercised in the model files, not here.
- `Network.test` auto-saves a checkpoint whenever test accuracy > 0.4 (`Network.py:259`). Files land in `./savedmodels/large20/model-<acc>.ckpt-*`.
- `preprocess.py:44` has a typo: `firts_time=` (typo of `first_time`). Don't rename without reading the `DataSet.write` signature.
- README's "Play Against An A.I." command contains typos (`—-gtp_poliy=…`, wrong quoting). The actual invocation is shown above.
- `processed_data/`, `data/SGFs/`, `train_log/`, `test_log/`, `savedmodels/`, `result.txt` are all created **by the script** at runtime (see `main.py:212-223`). No `.gitignore` exists — large artifacts can get committed accidentally; be careful with `git add`.
- The script writes results to `./result.txt` (appended). Training logs go to TensorBoard in `./train_log` and `./test_log`.

## Tests

`utils/tests/` contains `test_datasets.py`, `test_features.py`, `test_go.py`, `test_sgf_wrapper.py`, `test_strategies.py`, `test_utils.py`. There is no `pytest`/`unittest` config and no documented runner. Run them ad hoc with `python -m unittest utils.tests.<module>` or `pytest utils/tests/` if you set up a runner — but expect failures on TF-dependent paths.

## Conventions specific to this repo

- Configuration is **always** `FLAGS` (argparse) + `HPS` (namedtuple) in `config.py`. New hyperparameters go there, not as module-level constants.
- The three model variants share a common interface (`build_graph`, `training`, `temp`, `reinforce_dir`, `use_sparse_sotfmax` [sic], `cost`, `acc`, `result_acc`, `value`, `prediction`, `train_op`, `global_step`, `norm`, `increase_global_step`, `summaries`, `lrn_rate`). To add a variant, mirror these names.
- Logging uses `daiquiri` at DEBUG level everywhere (`Network.py:9`, `main.py:15`). Don't switch to stdlib `logging`.
- Game features have 17 planes (16 board-state + 1 player-colour) on a 19×19 board; the player-colour plane (index 16) and the game-result scalar are remapped from {0,1} → {−1,+1} before being fed to the network (`Network.py:146`, `176`, `228`). Don't skip that step.