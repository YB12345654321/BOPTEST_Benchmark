# -*- coding: utf-8 -*-
"""PPO 方法专用超参数（与 DQN/A2C 对齐：512 容量、更稳 LR、GAE、梯度裁剪）。"""
PPO_EPOCHS = 8
PPO_CLIP = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.98   # 0.95→0.98 降低 advantage 方差，与 A2C 一致
LEARNING_RATE_ACTOR = 3e-4   # 略提，加快策略学习
LEARNING_RATE_CRITIC = 5e-4  # 1e-3→5e-4 避免 value overshoot，与 A2C 一致
ENTROPY_COEF_START = 0.18
ENTROPY_COEF_END = 0.03      # 保留一点熵，避免策略过早退化
ENTROPY_DECAY_STEPS = 350    # 500 episode 下前 ~70% 保持较多探索
USE_ENTROPY_DECAY = True
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0  # 0.5→1.0 与 A2C/DQN 一致，避免梯度过早截断
MINI_BATCH_SIZE = 192
USE_LAYER_NORM = True
USE_DROPOUT = True
HIDDEN_DIM = 512    # 256→512 与 DQN/A2C 同容量
WARMUP_STEPS = 4
WARMUP_ACTION = 23
