# -*- coding: utf-8 -*-
"""
统一环境配置（五方法共用，便于 benchmark 对比）
- OGB 为你的方法，DQN/PPO/A2C/QTable 为 benchmark，环境与目标完全一致。
"""
import os

# ---------- BOPTEST 服务（连接 Docker 等）----------
BOPTEST_URL = os.getenv("BOPTEST_URL", "http://127.0.0.1:80")
TESTCASE_NAME = "bestest_air"

# ---------- 训练通用 ----------
TOTAL_EPISODES = 300
STEPS_PER_EPISODE = 96   # 1 day, 15min/step
EVAL_FREQUENCY = 5
STEP_PRINT_INTERVAL = 10  # 每 N 步打印一次当前状态

# ---------- 数据与图表（各方法会存到 data/<method>/ 下）----------
DATA_ROOT = "data"
PLOT_SUBDIR = "training_output"

# ---------- Episode 起始时间：训练随机、评估固定 ----------
RANDOMIZE_START_TIME_TRAIN = True
TRAIN_START_DAY_MIN = 0
TRAIN_START_DAY_MAX = 360
EVAL_START_TIMES = [0, 90 * 86400, 180 * 86400, 270 * 86400]
EVAL_EPISODES_PER_START = 1

# ---------- 动作空间（五方法一致）----------
FAN_LEVELS = [0.4, 0.7, 1.0]
SUPPLY_TEMP_LEVELS = [288.15, 296.15]
HEAT_SETPOINT_LEVELS = [294.15, 296.15]
COOL_SETPOINT_LEVELS = [299.15, 301.15]

NUM_ACTIONS = (
    len(FAN_LEVELS) * len(SUPPLY_TEMP_LEVELS)
    * len(HEAT_SETPOINT_LEVELS) * len(COOL_SETPOINT_LEVELS)
)

# ---------- 奖励/舒适区（目标一致）----------
COMFORT_LOW = 20.0
COMFORT_HIGH = 24.0
