# -*- coding: utf-8 -*-
"""DQN 方法专用超参数。若回报震荡可试 LR=3e-4 或 EPS_DECAY_STEPS=15_000。"""
GAMMA = 0.99
LR = 4e-4           # 5e-4→4e-4 略降，更稳
BATCH_SIZE = 384
BUFFER_SIZE = 12_000
MIN_REPLAY_SIZE = 7_000
TARGET_UPDATE = 50
EPS_START = 1.0
EPS_END = 0.01      # 0.001→0.01 保留极少量探索，避免完全贪心过拟合
EPS_DECAY_STEPS = 14_000  # 略拉长，探索稍多一段时间
TAU = 0.005
HIDDEN_DIM = 512
WARMUP_STEPS = 4
WARMUP_ACTION = 23
