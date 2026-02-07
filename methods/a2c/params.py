# -*- coding: utf-8 -*-
"""A2C 方法专用超参数（环境与 config 一致，此处仅算法相关）。"""
# 与 A2C——BOPTEST--New 一致
PARAM_PRESET = "a2c_tuned"  # "a2c_tuned" | "ppo_like"
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

if PARAM_PRESET == "ppo_like":
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    ENTROPY_COEF_START = 0.20
    ENTROPY_COEF_END = 0.05
    ENTROPY_DECAY_STEPS = 300
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = True
    DROPOUT_P = 0.1
    HIDDEN_DIM = 256
else:
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    ENTROPY_COEF_START = 0.08
    ENTROPY_COEF_END = 0.02
    ENTROPY_DECAY_STEPS = 200
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = False
    DROPOUT_P = 0.0
    HIDDEN_DIM = 256

WARMUP_STEPS = 4
WARMUP_ACTION = 23
