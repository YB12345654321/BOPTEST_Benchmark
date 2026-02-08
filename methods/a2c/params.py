# -*- coding: utf-8 -*-
"""A2C 方法专用超参数（偏稳、多探索、利于 500 episode 内收敛）。
A2C 每 episode 只 1 次更新，易方差大、策略过早收敛，故：略降学习率、拉长熵衰减、提高 GAE。"""
PARAM_PRESET = "a2c_tuned"  # "a2c_tuned" | "ppo_like"
GAMMA = 0.99
GAE_LAMBDA = 0.98   # 0.95→0.98 降低 advantage 方差，更新更稳
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0

if PARAM_PRESET == "ppo_like":
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4
    ENTROPY_COEF_START = 0.15
    ENTROPY_COEF_END = 0.03
    ENTROPY_DECAY_STEPS = 350
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = True
    DROPOUT_P = 0.1
    HIDDEN_DIM = 512
else:
    # a2c_tuned：更稳、探索更长，避免过早收敛到次优
    LR_ACTOR = 3e-4    # 略降，减少策略震荡
    LR_CRITIC = 5e-4   # critic 可稍大，快速跟踪 value
    ENTROPY_COEF_START = 0.12
    ENTROPY_COEF_END = 0.03   # 保留一点熵，避免策略过早退化
    ENTROPY_DECAY_STEPS = 280  # 前 ~56% episode 保持较多探索（500 总 ep）
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = False
    DROPOUT_P = 0.0
    HIDDEN_DIM = 512

WARMUP_STEPS = 4
WARMUP_ACTION = 23
