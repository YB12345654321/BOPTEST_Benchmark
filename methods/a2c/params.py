# -*- coding: utf-8 -*-
"""A2C 方法专用超参数（对齐 DQN 表现：更大容量、更稳学习率、适度探索）。"""
PARAM_PRESET = "a2c_tuned"  # "a2c_tuned" | "ppo_like"
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0  # 与 DQN grad_clip=1.0 一致，避免梯度过早截断

if PARAM_PRESET == "ppo_like":
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4
    ENTROPY_COEF_START = 0.15
    ENTROPY_COEF_END = 0.03
    ENTROPY_DECAY_STEPS = 300
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = True
    DROPOUT_P = 0.1
    HIDDEN_DIM = 512
else:
    # a2c_tuned：与 DQN 同容量、学习率更稳、探索期更长
    LR_ACTOR = 5e-4   # 与 DQN LR=5e-4 一致
    LR_CRITIC = 5e-4  # 降低 critic 学习率，避免 value 震荡
    ENTROPY_COEF_START = 0.10
    ENTROPY_COEF_END = 0.02
    ENTROPY_DECAY_STEPS = 180  # 前 ~60% episode 保持较多探索
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = False
    DROPOUT_P = 0.0
    HIDDEN_DIM = 512  # 与 DQN 一致，提高表达能力

WARMUP_STEPS = 4
WARMUP_ACTION = 23
