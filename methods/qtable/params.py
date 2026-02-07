# -*- coding: utf-8 -*-
"""Q-Table 方法专用超参数（与 QTable——BOPTEST 一致）。"""
import numpy as np
GAMMA = 0.99
ALPHA = 0.25
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.99995
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 40_000
QT_STATE_MODE = "compact"
TEMP_TARGET_C = 22.0
if QT_STATE_MODE == "compact":
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
