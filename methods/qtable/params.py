# -*- coding: utf-8 -*-
"""Q-Table 方法专用超参数（对齐 DQN：更快收敛、更早利用、适度学习率）。"""
import numpy as np
GAMMA = 0.99
# 学习率：略降初值、提高底限、更慢衰减，便于 300 episode 内稳定收敛
ALPHA = 0.18
ALPHA_MIN = 0.03
ALPHA_DECAY = 0.99997
# 探索：更快退火、更低终点，中后期更多利用已学 Q
EPS_START = 1.0
EPS_END = 0.03
EPS_DECAY_STEPS = 18_000  # ~188 episode 内从 1.0 降到 0.03，后半程更贪心
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
