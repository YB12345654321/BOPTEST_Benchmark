# -*- coding: utf-8 -*-
"""
统一监控与出图（与 DQN——BOPTEST 出图维度一致）
- 每 N 步打印：Step | 室温 | 室外 | R | 动作 | action_str，细分 comfort/energy/smooth
- plot: 训练总览 2×3（Reward / Avg Temp / Comfort Ratio / Energy / Eval Reward / Temp Distribution）
- plot_combined: 一张图合并「训练总览 2×3」+「当前 episode 曲线 4×3」
- plot_episode_curves: 单张 4×3 episode 曲线（温度、奖励、舒适度、动作、设定值、风机、能耗、分布、累积、统计）
- 数据保存到 save_dir，便于各方法独立保存与后续对比
"""
import os
import json
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from utils import log, K2C, action_to_values, action_to_string

# JSON 存放子目录：episode_*.json 存到 save_dir/EPISODES_JSON_SUBDIR/ 下
EPISODES_JSON_SUBDIR = "episodes"

# 单日 episode 曲线横轴：用 0–24 小时（每步 15min），不用步数 1–96
STEP_SEC_FOR_PLOT = getattr(config, "STEP_SECONDS", 900)


def _time_hours_for_episode_curve(n_steps):
    """单日 episode 的横轴：0, 0.25, ..., 23.75 (h)，与步数一一对应。"""
    return np.arange(n_steps) * (STEP_SEC_FOR_PLOT / 3600.0)


def _set_time_of_day_axis(ax):
    """统一单日曲线的横轴为 0–24h。"""
    ax.set_xlabel("Time of day (h)")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])


def step_print(step1_index, info, action, reward, extra_line=None):
    """每 N 步打印一次（与 A2C 一致）。step1_index 为 1-based 步数。"""
    rd = info.get("reward_detail", {})
    print(
        f"  Step {step1_index:3d} | "
        f"室温={info['room_temp']:5.1f}°C | "
        f"室外={info['outdoor_temp']:5.1f}°C | "
        f"R={reward:+6.2f} | "
        f"动作={action:2d} | "
        f"{action_to_string(action, config.__dict__)}",
        flush=True,
    )
    if extra_line:
        print(f"           {extra_line}", flush=True)
    print(
        f"           细分: comfort={rd.get('comfort', 0):+.2f}, energy={rd.get('energy', 0):+.2f}, smooth={rd.get('smooth', 0):+.2f}",
        flush=True,
    )


class Monitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.episodes_dir = os.path.join(save_dir, EPISODES_JSON_SUBDIR)
        os.makedirs(self.episodes_dir, exist_ok=True)
        self.train_rewards = []
        self.train_temps = []
        self.train_comfort_ratios = []
        self.train_energy_consumption = []  # 每步功率之和 (kW·步)，兼容旧图
        # 论文/benchmark 对比用
        self.train_energy_kwh = []  # 总能耗 kWh
        self.train_heating_kwh = []
        self.train_cooling_kwh = []
        self.train_fan_kwh = []
        self.train_comfort_violation_ratios = []  # 1 - comfort_ratio
        self.train_rmse_temps = []  # 室温相对目标 RMSE (°C)
        self.train_max_temp_deviations = []  # 最大温度偏差 (°C)
        self.train_action_switch_counts = []  # 动作切换次数
        self.train_peak_power_kw = []  # 步内最大功率 (kW)
        self.train_start_times = []  # episode 起始 sim_time (秒)，便于季节对比
        self.eval_rewards = []
        self.eval_episodes = []
        self.entropies = []  # 可选，A2C/PPO 用
        self.last_actions = []
        self.last_temps = []
        self.last_outdoor = []
        self.last_rewards = []
        self.last_comfort_details = []
        self.last_energy_details = []
        self.last_action_counts = Counter()

    def log_episode_curves(self, actions, temps, outdoors, rewards, comfort_details=None, energy_details=None):
        self.last_actions = list(actions)
        self.last_temps = list(temps)
        self.last_outdoor = list(outdoors)
        self.last_rewards = list(rewards)
        self.last_comfort_details = list(comfort_details or [])
        self.last_energy_details = list(energy_details or [])
        self.last_action_counts = Counter(actions)

    def save_episode_data(
        self,
        episode,
        actions,
        temps,
        outdoors,
        rewards,
        comfort_details,
        energy_details,
        power_heating_kw=None,
        power_cooling_kw=None,
        power_fan_kw=None,
        start_time_seconds=None,
        co2_ppm_list=None,
    ):
        """保存单 episode 数据。power_*_kw 为每步功率 (kW)，用于能耗分项与论文指标。"""
        to_float = lambda x: float(x) if x is not None else 0.0
        step_sec = getattr(config, "STEP_SECONDS", 900)
        kwh_factor = step_sec / 3600.0

        comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in temps])) if temps else 0.0
        violation_ratio = 1.0 - comfort_ratio
        target = getattr(config, "COMFORT_TARGET", 22.0)
        rmse_temp = float(np.sqrt(np.mean([(t - target) ** 2 for t in temps]))) if temps else 0.0
        max_dev = float(np.max([abs(t - target) for t in temps])) if temps else 0.0
        action_switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]) if len(actions) > 1 else 0
        peak_kw = float(max(energy_details)) if energy_details else 0.0

        total_energy_kwh = float(sum(energy_details) * kwh_factor) if energy_details else 0.0
        heating_kwh = cooling_kwh = fan_kwh = 0.0
        if power_heating_kw and power_cooling_kw and power_fan_kw:
            heating_kwh = float(sum(power_heating_kw) * kwh_factor)
            cooling_kwh = float(sum(power_cooling_kw) * kwh_factor)
            fan_kwh = float(sum(power_fan_kw) * kwh_factor)

        episode_data = {
            "episode": int(episode),
            "steps": len(actions),
            "actions": [int(a) for a in actions],
            "temperatures": [to_float(t) for t in temps],
            "outdoor_temps": [to_float(t) for t in outdoors],
            "rewards": [to_float(r) for r in rewards],
            "comfort_details": [to_float(c) for c in (comfort_details or [])],
            "energy_details": [to_float(e) for e in (energy_details or [])],
            "avg_temp": float(np.mean(temps)) if temps else 0.0,
            "comfort_ratio": comfort_ratio,
            "total_energy": float(sum(energy_details)) if energy_details else 0.0,
            "total_energy_kwh": total_energy_kwh,
            "total_heating_kwh": heating_kwh,
            "total_cooling_kwh": cooling_kwh,
            "total_fan_kwh": fan_kwh,
            "comfort_violation_ratio": violation_ratio,
            "rmse_temp_c": rmse_temp,
            "max_temp_deviation_c": max_dev,
            "action_switch_count": action_switches,
            "peak_power_kw": peak_kw,
            "start_time_seconds": start_time_seconds,
            "co2_ppm_mean": float(np.mean(co2_ppm_list)) if co2_ppm_list else None,
        }
        if power_heating_kw is not None:
            episode_data["power_heating_kw"] = [to_float(x) for x in power_heating_kw]
            episode_data["power_cooling_kw"] = [to_float(x) for x in power_cooling_kw]
            episode_data["power_fan_kw"] = [to_float(x) for x in power_fan_kw]
        if co2_ppm_list is not None:
            episode_data["co2_ppm"] = [to_float(x) for x in co2_ppm_list]

        self.train_comfort_violation_ratios.append(violation_ratio)
        self.train_rmse_temps.append(rmse_temp)
        self.train_max_temp_deviations.append(max_dev)
        self.train_action_switch_counts.append(action_switches)
        self.train_peak_power_kw.append(peak_kw)
        self.train_energy_kwh.append(total_energy_kwh)
        self.train_heating_kwh.append(heating_kwh)
        self.train_cooling_kwh.append(cooling_kwh)
        self.train_fan_kwh.append(fan_kwh)
        if start_time_seconds is not None:
            self.train_start_times.append(start_time_seconds)

        path = os.path.join(self.episodes_dir, f"episode_{episode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        return episode_data

    def save_training_summary(self):
        n = len(self.train_rewards)
        last50 = min(50, n) if n else 0
        def avg_last50(lst):
            if not lst or last50 == 0:
                return 0.0
            return float(np.mean(lst[-last50:]))

        summary = {
            "total_episodes": n,
            "training_history": {
                "rewards": [float(r) for r in self.train_rewards],
                "temps": [float(t) for t in self.train_temps],
                "comfort_ratios": [float(c) for c in self.train_comfort_ratios],
                "energy_consumption": [float(e) for e in self.train_energy_consumption],
                "energy_kwh": [float(e) for e in self.train_energy_kwh],
                "heating_kwh": [float(h) for h in self.train_heating_kwh],
                "cooling_kwh": [float(c) for c in self.train_cooling_kwh],
                "fan_kwh": [float(f) for f in self.train_fan_kwh],
                "comfort_violation_ratios": [float(v) for v in self.train_comfort_violation_ratios],
                "rmse_temps": [float(r) for r in self.train_rmse_temps],
                "max_temp_deviations": [float(d) for d in self.train_max_temp_deviations],
                "action_switch_counts": [int(a) for a in self.train_action_switch_counts],
                "peak_power_kw": [float(p) for p in self.train_peak_power_kw],
                "eval_episodes": [int(e) for e in self.eval_episodes],
                "eval_rewards": [float(r) for r in self.eval_rewards],
            },
            "final_stats": {
                "avg_reward": avg_last50(self.train_rewards),
                "avg_temp": avg_last50(self.train_temps),
                "avg_comfort_ratio": avg_last50(self.train_comfort_ratios),
                "avg_energy_kwh": avg_last50(self.train_energy_kwh),
                "avg_heating_kwh": avg_last50(self.train_heating_kwh),
                "avg_cooling_kwh": avg_last50(self.train_cooling_kwh),
                "avg_fan_kwh": avg_last50(self.train_fan_kwh),
                "avg_comfort_violation_ratio": avg_last50(self.train_comfort_violation_ratios),
                "avg_rmse_temp_c": avg_last50(self.train_rmse_temps),
                "avg_max_temp_deviation_c": avg_last50(self.train_max_temp_deviations),
                "avg_action_switch_count": avg_last50(self.train_action_switch_counts),
                "avg_peak_power_kw": avg_last50(self.train_peak_power_kw),
            },
        }
        path = os.path.join(self.save_dir, "training_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        csv_path = os.path.join(self.save_dir, "training_history.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Episode", "Reward", "AvgTemp", "ComfortRatio", "EnergyConsumption", "EnergyKWh",
                "HeatingKWh", "CoolingKWh", "FanKWh",
                "ComfortViolationRatio", "RMSE_TempC", "MaxTempDeviationC", "ActionSwitchCount", "PeakPowerKW",
            ])
            for i in range(n):
                w.writerow([
                    i + 1,
                    self.train_rewards[i],
                    self.train_temps[i] if i < len(self.train_temps) else 0,
                    self.train_comfort_ratios[i] if i < len(self.train_comfort_ratios) else 0,
                    self.train_energy_consumption[i] if i < len(self.train_energy_consumption) else 0,
                    self.train_energy_kwh[i] if i < len(self.train_energy_kwh) else 0,
                    self.train_heating_kwh[i] if i < len(self.train_heating_kwh) else 0,
                    self.train_cooling_kwh[i] if i < len(self.train_cooling_kwh) else 0,
                    self.train_fan_kwh[i] if i < len(self.train_fan_kwh) else 0,
                    self.train_comfort_violation_ratios[i] if i < len(self.train_comfort_violation_ratios) else 0,
                    self.train_rmse_temps[i] if i < len(self.train_rmse_temps) else 0,
                    self.train_max_temp_deviations[i] if i < len(self.train_max_temp_deviations) else 0,
                    self.train_action_switch_counts[i] if i < len(self.train_action_switch_counts) else 0,
                    self.train_peak_power_kw[i] if i < len(self.train_peak_power_kw) else 0,
                ])
        log(f"💾 训练数据已保存到 {self.save_dir}")

    def plot(self, save_path=None):
        """训练总览 2×3：Reward(MA10) / Avg Temp / Comfort Ratio / Energy / Eval Reward / Temp Distribution (last 50)."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Training Reward
        ax = axes[0, 0]
        episodes = list(range(1, len(self.train_rewards) + 1))
        ax.plot(episodes, self.train_rewards, "b-", alpha=0.4, label="Raw")
        if len(self.train_rewards) >= 10:
            ma = np.convolve(self.train_rewards, np.ones(10) / 10, mode="valid")
            ma_episodes = list(range(10, len(self.train_rewards) + 1))
            ax.plot(ma_episodes, ma, "r-", linewidth=2, label="MA(10)")
        ax.set_title("Training Reward", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Average Temperature
        ax = axes[0, 1]
        ax.plot(self.train_temps, "g-", linewidth=1.5)
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
        ax.set_title("Average Temperature", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Comfort Zone Ratio
        ax = axes[0, 2]
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Zone Ratio", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Ratio (0-1)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Energy Consumption
        ax = axes[1, 0]
        if self.train_energy_consumption:
            ax.plot(self.train_energy_consumption, "orange", linewidth=1.5)
        ax.set_title("Energy Consumption", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Energy (kW)")
        ax.grid(True, alpha=0.3)

        # 5. Evaluation Reward
        ax = axes[1, 1]
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-", markersize=6, linewidth=2)
        ax.set_title("Evaluation Reward", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval Reward")
        ax.grid(True, alpha=0.3)

        # 6. Temperature Distribution (last 50 episodes)
        ax = axes[1, 2]
        if len(self.train_temps) >= 50:
            recent_temps = self.train_temps[-50:]
            ax.hist(recent_temps, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
            ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.set_title("Temperature Distribution (Last 50 Episodes)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log(f"📊 图表已保存: {save_path}")
        plt.close()

    def plot_combined(self, save_path=None, episode_label=None):
        """
        一张图合并：上方 2×3 训练总览，下方 4×3 当前 episode 曲线（与 DQN——BOPTEST 维度一致）。
        每次保存到同一路径，覆盖旧图以节省空间。
        """
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.6], hspace=0.4)

        # ---------- 上方 2×3 训练总览 ----------
        gs_top = gs[0].subgridspec(2, 3, wspace=0.3, hspace=0.35)
        axes_top = [[fig.add_subplot(gs_top[i, j]) for j in range(3)] for i in range(2)]

        ep_list = list(range(1, len(self.train_rewards) + 1))
        ax = axes_top[0][0]
        ax.plot(ep_list, self.train_rewards, "b-", alpha=0.4)
        if len(self.train_rewards) >= 10:
            ma = np.convolve(self.train_rewards, np.ones(10) / 10, mode="valid")
            ax.plot(range(10, len(self.train_rewards) + 1), ma, "r-", linewidth=2, label="MA(10)")
        ax.set_title("Training Reward")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

        ax = axes_top[0][1]
        ax.plot(self.train_temps, "g-", linewidth=1.5)
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green")
        ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
        ax.set_title("Average Temperature")
        ax.set_xlabel("Episode")
        ax.set_ylabel("°C")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes_top[0][2]
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Zone Ratio")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes_top[1][0]
        if self.train_energy_consumption:
            ax.plot(self.train_energy_consumption, "orange", linewidth=1.5)
        ax.set_title("Energy Consumption")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Energy (kW)")
        ax.grid(True, alpha=0.3)

        ax = axes_top[1][1]
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-", markersize=4, linewidth=2)
        ax.set_title("Evaluation Reward")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

        ax = axes_top[1][2]
        if len(self.train_temps) >= 50:
            recent = self.train_temps[-50:]
            ax.hist(recent, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
            ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.set_title("Temp Distribution (Last 50 Ep)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # ---------- 下方 4×3 当前 episode 曲线 ----------
        gs_bot = gs[1].subgridspec(4, 3, wspace=0.28, hspace=0.4)
        axes_bot = [[fig.add_subplot(gs_bot[i, j]) for j in range(3)] for i in range(4)]
        ep_title = f"Episode {episode_label}" if episode_label is not None else "Latest Episode"

        if self.last_actions:
            n_steps = len(self.last_actions)
            time_h = _time_hours_for_episode_curve(n_steps)
            decoded = [action_to_values(a, config.__dict__) for a in self.last_actions]
            fan_seq = [d["fan"] for d in decoded]
            supply_seq = [K2C(d["supply_temp"]) for d in decoded]
            heat_seq = [K2C(d["heat_setpoint"]) for d in decoded]
            cool_seq = [K2C(d["cool_setpoint"]) for d in decoded]

            ax = axes_bot[0][0]
            ax.plot(time_h, self.last_temps, label="Room Temp", linewidth=2, color="blue")
            ax.plot(time_h, self.last_outdoor, label="Outdoor Temp", alpha=0.6, color="gray", linestyle="--")
            ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.15, color="green", label="Comfort Zone 20-24°C")
            ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
            ax.set_title(f"{ep_title} | Temperature Profile")
            ax.set_ylabel("Temperature (°C)")
            _set_time_of_day_axis(ax)
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[0][1]
            ax.plot(time_h, self.last_rewards, "tab:orange", linewidth=1.5)
            ax.set_title("Rewards over Day")
            ax.set_ylabel("Reward")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            _set_time_of_day_axis(ax)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[0][2]
            if self.last_comfort_details:
                ax.plot(time_h, self.last_comfort_details, "green", linewidth=1.5, label="Comfort Reward")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                comfort_zones = [1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in self.last_temps]
                ax2 = ax.twinx()
                ax2.fill_between(time_h, 0, comfort_zones, alpha=0.2, color="green", label="In Comfort Zone")
                ax2.set_ylabel("Comfort Zone (0/1)", color="green")
                ax2.set_ylim(-0.1, 1.1)
            ax.set_title("Comfort Reward Details")
            ax.set_ylabel("Comfort Reward")
            _set_time_of_day_axis(ax)
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[1][0]
            ax.scatter(time_h, self.last_actions, s=15, alpha=0.6, color="steelblue")
            ax.set_title("Action Index over Day")
            ax.set_ylabel("Action Index (0-23)")
            ax.set_ylim(-1, max(self.last_actions) + 2 if self.last_actions else 24)
            _set_time_of_day_axis(ax)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[1][1]
            items = sorted(self.last_action_counts.items(), key=lambda x: x[0])
            if items:
                labels, counts = zip(*items)
                ax.bar(labels, counts, color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_title("Action Usage Histogram")
            ax.set_xlabel("Action Index")
            ax.set_ylabel("Usage Count")
            ax.grid(True, axis="y", alpha=0.3)

            ax = axes_bot[1][2]
            ax.plot(time_h, supply_seq, label="Supply Temp", linewidth=1.5, color="red")
            ax.plot(time_h, heat_seq, label="Heat Setpoint", linewidth=1.5, color="orange")
            ax.plot(time_h, cool_seq, label="Cool Setpoint", linewidth=1.5, color="blue")
            ax.set_title("Supply Temperature & Setpoints")
            ax.set_ylabel("Temperature (°C)")
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[2][0]
            ax.plot(time_h, fan_seq, label="Fan u", color="tab:purple", linewidth=1.5)
            ax.set_title("Fan Control")
            ax.set_ylabel("Fan (0-1)")
            ax.set_ylim(0, 1.05)
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[2][1]
            if self.last_energy_details:
                ax.plot(time_h, self.last_energy_details, "orange", linewidth=1.5, label="Energy Cost")
            ax.set_title("Energy Consumption over Day")
            ax.set_ylabel("Power (kW)")
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[2][2]
            if self.last_temps:
                ax.hist(self.last_temps, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
                ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
                ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
            ax.set_title("Temperature Distribution")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

            ax = axes_bot[3][0]
            if self.last_comfort_details and self.last_energy_details:
                cum_comfort = np.cumsum(self.last_comfort_details)
                cum_energy = np.cumsum([-e for e in self.last_energy_details])
                ax.plot(time_h, cum_comfort, "g-", linewidth=2, label="Cumulative Comfort")
                ax.plot(time_h, cum_energy, "orange", linewidth=2, label="Cumulative Energy (neg)")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Cumulative Reward Components")
            ax.set_ylabel("Cumulative Reward")
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[3][1]
            ax.axis("off")
            avg_temp = float(np.mean(self.last_temps)) if self.last_temps else 0
            comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in self.last_temps])) if self.last_temps else 0
            total_energy = sum(self.last_energy_details) if self.last_energy_details else 0
            total_reward = sum(self.last_rewards) if self.last_rewards else 0
            avg_comfort = float(np.mean(self.last_comfort_details)) if self.last_comfort_details else 0
            stats = (
                f"Episode {episode_label} Statistics\n"
                + "=" * 50 + "\n"
                + f"Total Reward: {total_reward:.2f}\n"
                + f"Avg Temperature: {avg_temp:.2f}°C\n"
                + f"Comfort Zone Ratio: {comfort_ratio*100:.1f}%\n"
                + f"Total Energy: {total_energy:.2f} kW\n"
                + f"Avg Comfort Reward: {avg_comfort:.2f}\n"
                + f"Steps: {len(self.last_actions)}\n"
                + f"Unique Actions: {len(self.last_action_counts)}"
            )
            ax.text(0.1, 0.5, stats, fontsize=10, family="monospace", verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

            axes_bot[3][2].axis("off")
        else:
            for i in range(4):
                for j in range(3):
                    axes_bot[i][j].text(0.5, 0.5, "暂无 episode 曲线", ha="center", va="center", transform=axes_bot[i][j].transAxes)
                    axes_bot[i][j].set_xticks([])
                    axes_bot[i][j].set_yticks([])

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log(f"📊 图表已保存（已覆盖）: {save_path}")
        plt.close()

    def save_individual_plots(self, save_dir, method_name, episode_label=None):
        """
        在汇总看板之外，把看板里每个小图单独存为 method_图类型.png，便于后续按需提取。
        - 训练总览 6 张：training_reward, avg_temp, comfort_ratio, energy_consumption, eval_reward, temp_distribution_last50
        - 当前 episode 曲线多张：ep_temperature_profile, ep_rewards_over_day, ep_comfort_details, ...（无 episode 数据则只存训练 6 张）
        """
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{method_name}_"
        ep_list = list(range(1, len(self.train_rewards) + 1))

        # ---------- 1. Training Reward ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ep_list, self.train_rewards, "b-", alpha=0.4, label="Raw")
        if len(self.train_rewards) >= 10:
            ma = np.convolve(self.train_rewards, np.ones(10) / 10, mode="valid")
            ax.plot(range(10, len(self.train_rewards) + 1), ma, "r-", linewidth=2, label="MA(10)")
        ax.set_title("Training Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}training_reward.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- 2. Average Temperature ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(self.train_temps, "g-", linewidth=1.5)
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
        ax.set_title("Average Temperature")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}avg_temp.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- 3. Comfort Zone Ratio ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Zone Ratio")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Ratio (0-1)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}comfort_ratio.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- 4. Energy Consumption ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if self.train_energy_consumption:
            ax.plot(self.train_energy_consumption, "orange", linewidth=1.5)
        ax.set_title("Energy Consumption")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Energy (kW)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}energy_consumption.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- 5. Evaluation Reward ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-", markersize=6, linewidth=2)
        ax.set_title("Evaluation Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval Reward")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}eval_reward.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- 6. Temperature Distribution (Last 50) ----------
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if len(self.train_temps) >= 50:
            recent = self.train_temps[-50:]
            ax.hist(recent, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
            ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.set_title("Temperature Distribution (Last 50 Episodes)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}temp_distribution_last50.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---------- Episode 曲线单独图（有 last_actions 时）---------
        if self.last_actions:
            n_steps = len(self.last_actions)
            time_h = _time_hours_for_episode_curve(n_steps)
            decoded = [action_to_values(a, config.__dict__) for a in self.last_actions]
            fan_seq = [d["fan"] for d in decoded]
            supply_seq = [K2C(d["supply_temp"]) for d in decoded]
            heat_seq = [K2C(d["heat_setpoint"]) for d in decoded]
            cool_seq = [K2C(d["cool_setpoint"]) for d in decoded]

            # ep_temperature_profile
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(time_h, self.last_temps, label="Room Temp", linewidth=2, color="blue")
            ax.plot(time_h, self.last_outdoor, label="Outdoor Temp", alpha=0.6, color="gray", linestyle="--")
            ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.15, color="green", label="Comfort Zone 20-24°C")
            ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
            ax.set_title("Temperature Profile")
            ax.set_ylabel("Temperature (°C)")
            _set_time_of_day_axis(ax)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_temperature_profile.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_rewards_over_day
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(time_h, self.last_rewards, "tab:orange", linewidth=1.5)
            ax.set_title("Rewards over Day")
            ax.set_ylabel("Reward")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            _set_time_of_day_axis(ax)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_rewards_over_day.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_comfort_details
            if self.last_comfort_details:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(time_h, self.last_comfort_details, "green", linewidth=1.5, label="Comfort Reward")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                comfort_zones = [1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in self.last_temps]
                ax2 = ax.twinx()
                ax2.fill_between(time_h, 0, comfort_zones, alpha=0.2, color="green", label="In Comfort Zone")
                ax2.set_ylabel("Comfort Zone (0/1)", color="green")
                ax2.set_ylim(-0.1, 1.1)
                ax.set_title("Comfort Reward Details")
                ax.set_ylabel("Comfort Reward")
                _set_time_of_day_axis(ax)
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}ep_comfort_details.png"), dpi=150, bbox_inches="tight")
                plt.close()

            # ep_action_index
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(time_h, self.last_actions, s=15, alpha=0.6, color="steelblue")
            ax.set_title("Action Index over Day")
            ax.set_ylabel("Action Index (0-23)")
            ax.set_ylim(-1, max(self.last_actions) + 2 if self.last_actions else 24)
            _set_time_of_day_axis(ax)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_action_index.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_action_histogram
            fig, ax = plt.subplots(figsize=(6, 3.5))
            items = sorted(self.last_action_counts.items(), key=lambda x: x[0])
            if items:
                labels, counts = zip(*items)
                ax.bar(labels, counts, color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_title("Action Usage Histogram")
            ax.set_xlabel("Action Index")
            ax.set_ylabel("Usage Count")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_action_histogram.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_supply_setpoints
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(time_h, supply_seq, label="Supply Temp", linewidth=1.5, color="red")
            ax.plot(time_h, heat_seq, label="Heat Setpoint", linewidth=1.5, color="orange")
            ax.plot(time_h, cool_seq, label="Cool Setpoint", linewidth=1.5, color="blue")
            ax.set_title("Supply Temperature & Setpoints")
            ax.set_ylabel("Temperature (°C)")
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_supply_setpoints.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_fan_control
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(time_h, fan_seq, label="Fan u", color="tab:purple", linewidth=1.5)
            ax.set_title("Fan Control")
            ax.set_ylabel("Fan (0-1)")
            ax.set_ylim(0, 1.05)
            _set_time_of_day_axis(ax)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}ep_fan_control.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # ep_energy_over_day
            if self.last_energy_details:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(time_h, self.last_energy_details, "orange", linewidth=1.5, label="Energy Cost")
                ax.set_title("Energy Consumption over Day")
                ax.set_ylabel("Power (kW)")
                _set_time_of_day_axis(ax)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}ep_energy_over_day.png"), dpi=150, bbox_inches="tight")
                plt.close()

            # ep_temp_distribution
            if self.last_temps:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(self.last_temps, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
                ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
                ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
                ax.set_title("Temperature Distribution (Episode)")
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Frequency")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}ep_temp_distribution.png"), dpi=150, bbox_inches="tight")
                plt.close()

            # ep_cumulative_reward
            if self.last_comfort_details and self.last_energy_details:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                cum_comfort = np.cumsum(self.last_comfort_details)
                cum_energy = np.cumsum([-e for e in self.last_energy_details])
                ax.plot(time_h, cum_comfort, "g-", linewidth=2, label="Cumulative Comfort")
                ax.plot(time_h, cum_energy, "orange", linewidth=2, label="Cumulative Energy (neg)")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_title("Cumulative Reward Components")
                ax.set_ylabel("Cumulative Reward")
                _set_time_of_day_axis(ax)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}ep_cumulative_reward.png"), dpi=150, bbox_inches="tight")
                plt.close()

        log(f"📊 单图已保存: {save_dir} ({prefix}*.png)")

    def plot_episode_curves(self, episode_label, save_dir=None):
        """单独 4×3 episode 曲线（与 DQN——BOPTEST 一致）：温度、奖励、舒适度、动作、直方图、设定值、风机、能耗、温度分布、累积奖励、统计信息。横轴为 0–24h。"""
        if not self.last_actions:
            return
        n_steps = len(self.last_actions)
        time_h = _time_hours_for_episode_curve(n_steps)
        decoded = [action_to_values(a, config.__dict__) for a in self.last_actions]
        fan_seq = [d["fan"] for d in decoded]
        supply_seq = [K2C(d["supply_temp"]) for d in decoded]
        heat_seq = [K2C(d["heat_setpoint"]) for d in decoded]
        cool_seq = [K2C(d["cool_setpoint"]) for d in decoded]
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))

        ax = axes[0, 0]
        ax.plot(time_h, self.last_temps, label="Room Temp", linewidth=2, color="blue")
        ax.plot(time_h, self.last_outdoor, label="Outdoor Temp", alpha=0.6, color="gray", linestyle="--")
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.15, color="green", label="Comfort Zone 20-24°C")
        ax.axhline(22, color="r", linestyle="--", linewidth=1.5, label="Target 22°C")
        ax.set_title(f"Episode {episode_label} | Temperature Profile")
        ax.set_ylabel("Temperature (°C)")
        _set_time_of_day_axis(ax)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(time_h, self.last_rewards, "tab:orange", linewidth=1.5)
        ax.set_title("Rewards over Day")
        ax.set_ylabel("Reward")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        _set_time_of_day_axis(ax)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        if self.last_comfort_details:
            ax.plot(time_h, self.last_comfort_details, "green", linewidth=1.5, label="Comfort Reward")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            comfort_zones = [1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in self.last_temps]
            ax2 = ax.twinx()
            ax2.fill_between(time_h, 0, comfort_zones, alpha=0.2, color="green", label="In Comfort Zone")
            ax2.set_ylabel("Comfort Zone (0/1)", color="green")
            ax2.set_ylim(-0.1, 1.1)
        ax.set_title("Comfort Reward Details")
        ax.set_ylabel("Comfort Reward")
        _set_time_of_day_axis(ax)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.scatter(time_h, self.last_actions, s=15, alpha=0.6, color="steelblue")
        ax.set_title("Action Index over Day")
        ax.set_ylabel("Action Index (0-23)")
        ax.set_ylim(-1, max(self.last_actions) + 2 if self.last_actions else 24)
        _set_time_of_day_axis(ax)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        items = sorted(self.last_action_counts.items(), key=lambda x: x[0])
        if items:
            labels, counts = zip(*items)
            ax.bar(labels, counts, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title("Action Usage Histogram")
        ax.set_xlabel("Action Index")
        ax.set_ylabel("Usage Count")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 2]
        ax.plot(time_h, supply_seq, label="Supply Temp", linewidth=1.5, color="red")
        ax.plot(time_h, heat_seq, label="Heat Setpoint", linewidth=1.5, color="orange")
        ax.plot(time_h, cool_seq, label="Cool Setpoint", linewidth=1.5, color="blue")
        ax.set_title("Supply Temperature & Setpoints")
        ax.set_ylabel("Temperature (°C)")
        _set_time_of_day_axis(ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        ax.plot(time_h, fan_seq, label="Fan u", color="tab:purple", linewidth=1.5)
        ax.set_title("Fan Control")
        ax.set_ylabel("Fan (0-1)")
        ax.set_ylim(0, 1.05)
        _set_time_of_day_axis(ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        if self.last_energy_details:
            ax.plot(time_h, self.last_energy_details, "orange", linewidth=1.5, label="Energy Cost")
        ax.set_title("Energy Consumption over Day")
        ax.set_ylabel("Power (kW)")
        _set_time_of_day_axis(ax)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if not self.last_energy_details:
            ax.axis("off")

        ax = axes[2, 2]
        if self.last_temps:
            ax.hist(self.last_temps, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(22, color="r", linestyle="--", linewidth=2, label="Target 22°C")
            ax.axvspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Zone")
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[3, 0]
        if self.last_comfort_details and self.last_energy_details:
            cum_comfort = np.cumsum(self.last_comfort_details)
            cum_energy = np.cumsum([-e for e in self.last_energy_details])
            ax.plot(time_h, cum_comfort, "g-", linewidth=2, label="Cumulative Comfort")
            ax.plot(time_h, cum_energy, "orange", linewidth=2, label="Cumulative Energy (neg)")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Cumulative Reward Components")
        ax.set_ylabel("Cumulative Reward")
        _set_time_of_day_axis(ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[3, 1]
        ax.axis("off")
        avg_temp = float(np.mean(self.last_temps)) if self.last_temps else 0
        comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in self.last_temps])) if self.last_temps else 0
        total_energy = sum(self.last_energy_details) if self.last_energy_details else 0
        total_reward = sum(self.last_rewards) if self.last_rewards else 0
        avg_comfort = float(np.mean(self.last_comfort_details)) if self.last_comfort_details else 0
        stats = (
            f"Episode {episode_label} Statistics\n"
            + "=" * 50 + "\n"
            + f"Total Reward: {total_reward:.2f}\n"
            + f"Avg Temperature: {avg_temp:.2f}°C\n"
            + f"Comfort Zone Ratio: {comfort_ratio*100:.1f}%\n"
            + f"Total Energy: {total_energy:.2f} kW\n"
            + f"Avg Comfort Reward: {avg_comfort:.2f}\n"
            + f"Steps: {len(self.last_actions)}\n"
            + f"Unique Actions: {len(self.last_action_counts)}"
        )
        ax.text(0.1, 0.5, stats, fontsize=10, family="monospace", verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        axes[3, 2].axis("off")
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"curves_ep{episode_label}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log(f"📊 Episode 曲线已保存: {out}")
        plt.close()
