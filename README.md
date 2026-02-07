# BOPTEST Benchmark

五方法（207OGB、A2C、DQN、PPO、QTable）在**同一环境与目标**下的 Benchmark 项目，便于公平对比。

## 环境要求

- Python 3.8+
- BOPTEST 服务（Docker：在 `project1-boptest` 目录下 `docker-compose up`）
- 依赖：`numpy`, `requests`, `torch`, `matplotlib`, `pandas`  
- **Rule OGB** 还需 `realkd`：将 `BOPTEST_Project` 中的 `realkd` 目录复制到本项目根目录，或保证 `PYTHONPATH` 包含 `BOPTEST_Project` 路径。

## 统一配置

- 环境与奖励由根目录 `config.py`、`env.py` 统一配置（BOPTEST URL、步数、舒适区、动作空间等）。
- 各方法仅改各自 `methods/<name>/params.py` 中的超参数。

## 运行方式

### 方式一：命令行

在**项目根目录**执行（不要进入 `methods/xxx` 再执行）：

```bash
cd /Users/yu/Desktop/BOPTEST_Benchmark

# A2C
python methods/a2c/train.py

# DQN
python methods/dqn/train.py

# PPO
python methods/ppo/train.py

# Q-Table
python methods/qtable/train.py

# Rule OGB（需 realkd）
python methods/rule_ogb/train.py
```

### 方式二：Jupyter Notebook

项目根目录下提供了对应每个方法的 `.ipynb`，可在 Jupyter 中运行：

- **A2C——Benchmark.ipynb** → 运行 A2C 训练  
- **DQN——Benchmark.ipynb** → 运行 DQN 训练  
- **PPO——Benchmark.ipynb** → 运行 PPO 训练  
- **QTable——Benchmark.ipynb** → 运行 Q-Table 训练  
- **OGB——Benchmark.ipynb** → 运行 Rule OGB 训练（需 realkd）  
- **Compare——Benchmark.ipynb** → 运行五方法对比（`compare_runs.py`）

**注意：** 请先在终端 `cd` 到 **BOPTEST_Benchmark** 根目录，再执行 `jupyter notebook` 或 `jupyter lab`，然后打开对应 notebook。这样当前工作目录才是项目根，脚本才能正确找到 `methods/xxx/train.py`。

连接 BOPTEST 前请确认 Docker 已启动且 `config.py` 中 `BOPTEST_URL`（默认 `http://127.0.0.1:80`）正确。

## 数据与出图

- 各方法数据与图表保存在 **`data/<方法名>/`** 下，互不覆盖：
  - `data/a2c/`, `data/dqn/`, `data/ppo/`, `data/qtable/`, `data/rule_ogb/`
- 每个方法会生成：
  - `training_history.csv`：每 episode 的 Reward、AvgTemp、ComfortRatio、EnergyConsumption
  - `training_summary.json`：完整训练历史与最终统计
  - `training_output/`：**一张图合并、覆盖保存**（节省空间）
    - **`progress.png`**：每次做 eval 或训练结束时**覆盖**保存。图内包含：
      - **上方 1×5**：训练总览（Train Reward / Avg Temp / Comfort Ratio / Eval Reward / Entropy）
      - **下方 3×2**：当前 episode 曲线（温度、Step Rewards、动作、直方图、设定值、风机）
  - `episode_*.json`：单 episode 详细数据（可选用于细粒度对比）

## 多方法对比

**全部方法都跑完后**，在项目根目录执行：

```bash
python compare_runs.py
```

脚本会读取各方法下的 `data/<method>/training_history.csv` 和 `training_summary.json`，生成：

- `data/comparison/compare_reward.png`：训练奖励对比
- `data/comparison/compare_temp.png`：平均室温对比
- `data/comparison/compare_comfort_ratio.png`：舒适区比例对比
- `data/comparison/compare_energy.png`：能耗对比
- `data/comparison/compare_eval_reward.png`：评估奖励对比（若有）

并在终端打印各方法的汇总表（最后一轮 / 近 50 轮平均等）。

## 项目结构

```
BOPTEST_Benchmark/
├── config.py          # 统一环境与训练配置
├── env.py             # 统一 BOPTEST 环境封装
├── utils.py           # 工具函数（log, K2C, action_to_values/action_to_string）
├── monitor.py         # 统一监控、出图、数据保存（与 A2C——BOPTEST--New 一致）
├── data/              # 各方法输出目录
│   ├── a2c/
│   ├── dqn/
│   ├── ppo/
│   ├── qtable/
│   └── rule_ogb/
├── methods/
│   ├── a2c/           # params.py + train.py
│   ├── dqn/
│   ├── ppo/
│   ├── qtable/
│   └── rule_ogb/      # 依赖 realkd
└── README.md
```

每个方法的「特点」保留在各自的 `params.py` 和 `train.py` 中，仅环境与出图方式与其它方法一致。
