"""AlphaGo Zero — PyTorch port.

A modern, dependency-light reimplementation of the AGZ training pipeline
(self-play + MCTS + ResNet + policy/value learning) using PyTorch 2.x.

Modules:
    game/        Go board, feature extraction, SGF parsing
    model/       Pre-activation ResNet with policy + value heads
    mcts/        PUCT-guided Monte-Carlo Tree Search
    training/    Self-play, replay buffer, train loop, evaluation
    main.py      CLI entrypoint (modes: selfplay / train / play / evaluate)
    config.py    Hyperparameters (FLAGS + HPS)

Run:
    python new/main.py --mode=selfplay    # generate self-play games
    python new/main.py --mode=train       # train the network
    python new/main.py --mode=play        # human-vs-AI via GTP-like loop
    python new/main.py --mode=evaluate    # current vs best model
"""
