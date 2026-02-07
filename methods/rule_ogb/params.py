# -*- coding: utf-8 -*-
"""Rule OGB 方法专用超参数（与 207OGB 一致）。需将 BOPTEST_Project 的 realkd 复制到本项目根目录。"""
GAMMA = 0.99
NUM_COL = 20
NUM_RULES = 40
REG = 0.001
MIN_REPLAY_SIZE = 7_000
BATCH_SIZE = 384
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY_STEPS = 12_000
