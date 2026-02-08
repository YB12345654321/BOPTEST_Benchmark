# -*- coding: utf-8 -*-
"""A2C 方法专用超参数（偏保守：再降 LR、多探索，利于 500 ep 内稳收敛）。
若仍不理想：可试 LR_ACTOR=1e-4, LR_CRITIC=2e-4 或 ENTROPY_DECAY_STEPS=350。"""
PARAM_PRESET = "a2c_tuned"  # "a2c_tuned" | "ppo_like"
GAMMA = 0.99
GAE_LAMBDA = 0.98
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0

if PARAM_PRESET == "ppo_like":
    LR_ACTOR = 1e-4
    LR_CRITIC = 2e-4
    ENTROPY_COEF_START = 0.15
    ENTROPY_COEF_END = 0.04
    ENTROPY_DECAY_STEPS = 350
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = True
    DROPOUT_P = 0.1
    HIDDEN_DIM = 512
else:
    # a2c_tuned：再降 LR、拉长熵衰减，提高稳定性
    LR_ACTOR = 2e-4    # 3e-4→2e-4 更稳
    LR_CRITIC = 3e-4   # 5e-4→3e-4 与 actor 同量级
    ENTROPY_COEF_START = 0.12
    ENTROPY_COEF_END = 0.04   # 多保留一点熵
    ENTROPY_DECAY_STEPS = 320  # 前 ~64% episode 保持较多探索
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = False
    DROPOUT_P = 0.0
    HIDDEN_DIM = 512

WARMUP_STEPS = 4
WARMUP_ACTION = 23
