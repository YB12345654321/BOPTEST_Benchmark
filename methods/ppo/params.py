# -*- coding: utf-8 -*-
"""PPO 方法专用超参数（偏保守：降 LR、减 epoch、稳收敛）。
若曲线仍震荡或回报不升：可再试 LEARNING_RATE_ACTOR=1e-4, PPO_EPOCHS=4。"""
PPO_EPOCHS = 6     # 8→6 减轻对单批数据的过拟合，更新更稳
PPO_CLIP = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.98
LEARNING_RATE_ACTOR = 2e-4   # 3e-4→2e-4 更稳，减少策略震荡
LEARNING_RATE_CRITIC = 3e-4  # 5e-4→3e-4 与 actor 同量级，value 更稳
ENTROPY_COEF_START = 0.18
ENTROPY_COEF_END = 0.04      # 0.03→0.04 多保留一点探索
ENTROPY_DECAY_STEPS = 380    # 500 ep 下前 ~76% 保持较多探索
USE_ENTROPY_DECAY = True
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0
MINI_BATCH_SIZE = 192
USE_LAYER_NORM = True
USE_DROPOUT = True
HIDDEN_DIM = 512
WARMUP_STEPS = 4
WARMUP_ACTION = 23
