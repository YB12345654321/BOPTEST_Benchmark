# -*- coding: utf-8 -*-
"""A2C 方法专用超参数。若出现「只用一个 action」的坍缩：熵系数过小或衰减过快，
导致策略过早趋于确定性；此处加大熵、拉长衰减并设最小熵下限。"""
PARAM_PRESET = "a2c_tuned"  # "a2c_tuned" | "ppo_like"
GAMMA = 0.99
GAE_LAMBDA = 0.98
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0

# 防止策略坍缩到单一 action：熵系数下限，训练中 ent_coef 不会低于此值
ENTROPY_COEF_MIN = 0.06

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
    # a2c_tuned：加大熵、拉长衰减，避免只用一个 action
    LR_ACTOR = 1.5e-4   # 略降，避免过快收敛到单一动作
    LR_CRITIC = 2.5e-4
    ENTROPY_COEF_START = 0.22   # 提高初始探索
    ENTROPY_COEF_END = 0.08     # 结束时仍保留一定熵
    ENTROPY_DECAY_STEPS = 420   # 前 ~84% episode 保持较强探索
    USE_ENTROPY_DECAY = True
    USE_LAYER_NORM = True
    USE_DROPOUT = False
    DROPOUT_P = 0.0
    HIDDEN_DIM = 512

WARMUP_STEPS = 4
WARMUP_ACTION = 23
