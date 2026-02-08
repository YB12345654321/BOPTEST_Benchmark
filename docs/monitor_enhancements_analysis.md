# Monitor 控制情况全面分析与可增强项

本文档基于当前 `monitor.py` 与 `env`/`config` 的数据能力，分析**已有监控**与**可考虑新增**的项，便于做「全面的控制情况」监控。

---

## 一、当前已有（现状概览）

### 1. 训练总览（2×3 + 汇总）
- Training Reward（含 MA10）
- Average Temperature（含舒适区 20–24°C、目标 22°C）
- Comfort Zone Ratio
- Energy Consumption（总功率·步 / kWh 在 CSV）
- Evaluation Reward
- Temperature Distribution (Last 50 Ep)

### 2. 单日 Episode 曲线（4×3 + 第三行）
- 温度曲线（室温、室外、舒适区、目标 22°C）
- Rewards over Day
- Comfort Reward Details（舒适奖励 + 是否在舒适区）
- Action Index over Day
- Action Usage Histogram
- Supply Temperature & Setpoints（供风温、加热/制冷设定值）
- Fan Control
- Energy Consumption over Day（总功率）
- Temperature Distribution（该日分布）
- Cumulative Reward Components（舒适累积、能耗累积）
- 统计文本（总奖励、平均温、舒适比例、总能耗、步数、不同动作数）
- **某日 24h 温度与功率**：左轴加热/制冷功率(W)，右轴室内/室外/设定值/舒适区，标题含 Heating/Cooling kWh

### 3. 数据持久化
- 每 episode：`episodes/episode_N.json`（actions, temps, outdoor, rewards, comfort_details, energy_details, power_heating/cooling/fan_kw, start_time_seconds, co2_ppm）
- 训练结束：`training_summary.json`（含 final_stats 的 last50 平均）、`training_history.csv`（每 episode 一行，含 Reward, AvgTemp, ComfortRatio, EnergyKWh, Heating/Cooling/FanKWh, ComfortViolationRatio, RMSE_TempC, MaxTempDeviationC, ActionSwitchCount, PeakPowerKW）

### 4. 步级打印（step_print）
- Step | 室温 | 室外 | R | 动作 | action_str
- 细分：comfort / energy / smooth

### 5. 对比与论文
- `compare_runs.py`：多方法 Reward / AvgTemp / ComfortRatio / Energy(KWh) / Eval 对比图、paper_benchmark_table.csv

---

## 二、可考虑新增（按「控制情况」分类）

### A. 控制行为与稳定性

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **动作切换频率曲线** | 每 episode 的 action_switch_count 随 episode 变化（已有 CSV 列），可增加一张「Action Switch Count / Episode」训练总览图 | ✅ CSV 有 | 低 |
| **设定值跟踪误差** | 每步 \|室温 - 目标 22°C\| 或与当前 heating/cool setpoint 的偏差，episode 级 RMSE/MAE（RMSE 已有）| ✅ 部分有 | 低 |
| **设备占空比 / 运行时间** | 加热/制冷/风机「开启」步数占比（可由 power_* > 阈值或 action 解码得到）| 可从 power_* 或 action 推 | 中 |
| **Smooth 分量曲线** | reward 的 smooth 分项随 episode 或随步的曲线，观察动作是否过于抖动 | 需在 episode 内累积或按步存 | 中 |

**建议优先**：训练总览中加一张 **Action Switch Count vs Episode**（和 PeakPowerKW 类似），便于直接看控制是否过于频繁切换。

---

### B. 舒适与约束违反

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **舒适违反时长（分钟/步数）** | 每 episode 超出 20–24°C 的步数或分钟数 | 可由 temps 算 | 低 |
| **最大连续违反步数** | 单日最长连续超标长度，评估「长时间偏离」 | 需新算 | 低 |
| **分时段舒适率** | 按 0–6/6–12/12–18/18–24 或 夜间/日间 统计舒适率，便于看白天 vs 夜间控制效果 | 有 time_h + temps | 中 |
| **超标幅度分布** | 超标时 (T-24) 或 (20-T) 的分布直方图 | 可由 temps 算 | 低 |

**建议优先**：在 episode 统计或 JSON 中增加 **comfort_violation_steps**（违反步数）和 **max_consecutive_violation**（最大连续违反），并在单日统计文本里显示。

---

### C. 能耗与功率

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **加热/制冷/风机 kWh 随 Episode 曲线** | 训练总览中三条线或堆叠面积，对应 HeatingKWh/CoolingKWh/FanKWh | ✅ CSV 有 | 低 |
| **分项能耗占比（饼图）** | 单日或 last50 平均的 heating/cooling/fan 占比 | 有 | 低 |
| **峰谷比 / 峰值需求** | 每 episode 的 PeakPowerKW 随 episode 曲线（CSV 有，图可加）| ✅ CSV 有 | 低 |
| **某日 24h 风机功率** | 在「某日 24h」视图中增加风机功率（可选第三条填充或线）| last_power_fan_kw 已有 | 低 |

**建议优先**：训练总览加 **Heating/Cooling/Fan kWh vs Episode**（或堆叠面积）；单日 24h 图可选加风机功率曲线。

---

### D. 时间与季节维度

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **Episode 起始日分布** | train_start_times 对应「第几天」，可画直方图或按季节分组 | ✅ train_start_times | 低 |
| **按起始日分组的统计** | 如按 0–90/90–180/180–270/270–360 天分组的平均 reward、舒适率、能耗 | 需按 start_time 分组 | 中 |
| **最冷/最热日自动选取** | 按室外平均温或最高/最低选 1–2 个 episode，单独出「典型日」24h 图 | 有 outdoor、start_time | 中 |

**建议优先**：在 summary 或对比表中增加「按季节分组的平均指标」；可选脚本：从 history + episode JSON 选「最冷/最热日」并保存其 24h 图。

---

### E. 室内空气品质（IAQ）

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **CO2 随步/随日曲线** | 某日 24h 的 CO2 (ppm) | ✅ episode JSON 有 co2_ppm | 低 |
| **CO2 超标比例 / 均值** | 每 episode 超过某阈值（如 1000 ppm）的步数占比、平均 CO2 | 可算 | 低 |

**建议优先**：在单日 24h 或单独子图中增加 **CO2 (ppm) vs 时间**；在 episode 统计中增加 **co2_mean / co2_violation_ratio**（若需与舒适并列展示）。

---

### F. Reward 分解与收敛

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **Comfort / Energy / Smooth 分项随 Episode** | 每 episode 对 comfort_details、energy_details 求和，smooth 需步级记录并求和 | comfort/energy 有，smooth 需传或算 | 中 |
| **Eval 时各指标** | Eval 不仅 reward，还有该 eval 的 avg_temp、comfort_ratio、energy_kwh | 需在 eval 循环里汇总 | 中 |

**建议优先**：训练总览加 **Comfort vs Episode**、**Energy (reward 分量) vs Episode**（从现有 comfort_details/energy_details 按 episode 求和即可）；eval 时写入 eval_temps、eval_comfort_ratios 等并画点或表。

---

### G. 鲁棒性与异常

| 项 | 说明 | 数据是否已有 | 实现难度 |
|----|------|--------------|----------|
| **极端温度步数** | 室温 &lt; 18 或 &gt; 28 的步数（比舒适区更严）| 可由 temps 算 | 低 |
| **单步 reward 分布** | 每 episode 内 reward 的直方图或分位数，看是否有大量负奖励步 | 有 last_rewards | 低 |
| **断连/超时计数** | 若 env 或 BOPTEST 有超时/重试，可记入 summary | 需在 env 或 train 里打点 | 视接口而定 |

---

### H. 展示与可读性

| 项 | 说明 | 实现难度 |
|----|------|----------|
| **Occupancy 示意** | 若 BOPTEST 无 occupancy 输出，可用固定时段（如 8:00–18:00）在 24h 图上画灰色带，标注「典型在室」| 低 |
| **某日标题带日期** | start_time_seconds → 转为日历日期（如 2024-01-10），显示在「某日 24h」标题中 | 低 |
| **图例/单位统一** | 功率统一 W 或 kW，温度统一 °C，图例不重叠 | 低 |
| **表格汇总单页** | 一页 PDF/图：last50 的 AvgTemp、Comfort%、Energy kWh、Peak kW、Action Switches 等表格 | 低 |

---

## 三、建议实现顺序（高价值且低成本）

1. **训练总览**  
   - 增加：**Action Switch Count vs Episode**、**Peak Power (kW) vs Episode**。  
   - 增加：**Heating / Cooling / Fan kWh vs Episode**（三条线或堆叠面积）。

2. **单日 / Episode 统计**  
   - 在 episode 统计文本或 JSON 中增加：**comfort_violation_steps**、**max_consecutive_violation**（及可选 co2_mean / co2_violation_ratio）。  
   - 在「某日 24h」标题中用 **start_time_seconds 转成日期**（如 Episode 123 | 2024-03-15 | 24h 温度与功率）。

3. **单日 24h 图**  
   - 可选：增加 **风机功率** 曲线或填充（last_power_fan_kw）。  
   - 可选：增加 **CO2 (ppm)** 右轴或单独子图。  
   - 可选：**Occupancy** 灰色带（固定 8–18h）。

4. **Reward 分解**  
   - 训练总览：**Comfort 分量 vs Episode**、**Energy 分量 vs Episode**（由现有 comfort_details/energy_details 按 episode 求和）。  
   - Eval：记录并保存 **eval 的 avg_temp、comfort_ratio、energy_kwh**，用于对比表或图。

5. **季节/典型日**  
   - 在 `compare_runs` 或单独脚本：按 **start_time** 分组统计；或选 **最冷/最热日** 自动生成 24h 图。

---

## 四、数据流检查（确保可落地）

- **已有且未画图**：ActionSwitchCount、PeakPowerKW、HeatingKWh/CoolingKWh/FanKWh、RMSE_TempC、MaxTempDeviationC、ComfortViolationRatio、train_start_times、co2_ppm（在 episode JSON）、comfort_details/energy_details（步级）。  
- **需在 train 里多传一步**：若做 reward 分项（smooth）或 eval 的 temp/comfort/energy，需在各自 train 的 eval 循环里汇总并交给 monitor（例如 `monitor.eval_temps.append(avg_temp)` 等）。  
- **纯从现有数据可算**：violation_steps、max_consecutive_violation、分时段舒适率、CO2 均值/超标率、极端温度步数、设备占空比（由 power_* > 小阈值）。

按上述顺序逐步加，即可在不大改结构的前提下，把「控制情况」做得更全面、更适合写报告和对比。
