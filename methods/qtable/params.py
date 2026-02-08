# -*- coding: utf-8 -*-
"""Q-Table 方法专用超参数（偏保守：降 alpha、慢退火，稳收敛）。
说明：训练回报含探索，eval 回报为贪心策略，以 eval 为主看策略好坏。"""
import numpy as np
GAMMA = 0.99
# 学习率：再降一点，减少 Q 表震荡，慢衰减
ALPHA = 0.12
ALPHA_MIN = 0.02
ALPHA_DECAY = 0.99998
# 探索：稍慢退火，避免过早贪心导致陷入次优
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_STEPS = 15_000  # ~156 episode 内退火到 0.02
QT_STATE_MODE = "compact"
TEMP_TARGET_C = 22.0
if QT_STATE_MODE == "compact":
    # 室温误差 1°C、室外 5°C、时间 3h，状态数适中且对控制足够细
    BUCKETS = {
        "temp_err": np.arange(-8, 9, 1),
        "out_temp": np.arange(-10, 41, 5),
        "time_bin": np.arange(0, 24, 3),
    }
else:
    BUCKETS = {
        "room_temp": np.arange(10, 35, 2),
        "out_temp": np.arange(-10, 40, 2),
        "rel_hum": np.arange(0.0, 1.05, 0.1),
        "p_heating": np.arange(0.0, 10.5, 0.5),
        "p_cooling": np.arange(0.0, 10.5, 0.5),
        "p_fan": np.arange(0.0, 5.5, 0.5),
        "time_bin": np.arange(0, 24, 4),
    }
