# -*- coding: utf-8 -*-
"""
统一监控与出图
- 每 N 步打印：Step | 室温 | 室外 | R | 动作 | action_str，细分 comfort/energy/smooth
- plot_combined: 一张图合并「训练总览 1×5」+「当前 episode 曲线 3×2」，覆盖保存以节省空间
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
        self.train_rewards = []
        self.train_temps = []
        self.train_comfort_ratios = []
        self.train_energy_consumption = []
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

    def save_episode_data(self, episode, actions, temps, outdoors, rewards, comfort_details, energy_details):
        to_float = lambda x: float(x) if x is not None else 0.0
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
            "comfort_ratio": float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in temps])) if temps else 0.0,
            "total_energy": float(sum(energy_details)) if energy_details else 0.0,
        }
        path = os.path.join(self.save_dir, f"episode_{episode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        return episode_data

    def save_training_summary(self):
        summary = {
            "total_episodes": len(self.train_rewards),
            "training_history": {
                "rewards": [float(r) for r in self.train_rewards],
                "temps": [float(t) for t in self.train_temps],
                "comfort_ratios": [float(c) for c in self.train_comfort_ratios],
                "energy_consumption": [float(e) for e in self.train_energy_consumption],
                "eval_episodes": [int(e) for e in self.eval_episodes],
                "eval_rewards": [float(r) for r in self.eval_rewards],
            },
            "final_stats": {
                "avg_reward": float(np.mean(self.train_rewards[-50:])) if len(self.train_rewards) >= 50 else (float(np.mean(self.train_rewards)) if self.train_rewards else 0),
                "avg_temp": float(np.mean(self.train_temps[-50:])) if len(self.train_temps) >= 50 else (float(np.mean(self.train_temps)) if self.train_temps else 0),
                "avg_comfort_ratio": float(np.mean(self.train_comfort_ratios[-50:])) if len(self.train_comfort_ratios) >= 50 else (float(np.mean(self.train_comfort_ratios)) if self.train_comfort_ratios else 0),
            },
        }
        path = os.path.join(self.save_dir, "training_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        csv_path = os.path.join(self.save_dir, "training_history.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Episode", "Reward", "AvgTemp", "ComfortRatio", "EnergyConsumption"])
            for i in range(len(self.train_rewards)):
                w.writerow([
                    i + 1,
                    self.train_rewards[i],
                    self.train_temps[i] if i < len(self.train_temps) else 0,
                    self.train_comfort_ratios[i] if i < len(self.train_comfort_ratios) else 0,
                    self.train_energy_consumption[i] if i < len(self.train_energy_consumption) else 0,
                ])
        log(f"💾 训练数据已保存到 {self.save_dir}")

    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        ax = axes[0]
        ax.plot(self.train_rewards, "b-", alpha=0.6)
        if len(self.train_rewards) >= 5:
            ma = np.convolve(self.train_rewards, np.ones(5) / 5, mode="valid")
            ax.plot(range(4, len(self.train_rewards)), ma, "r-", linewidth=2)
        ax.set_title("Train Reward")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self.train_temps, "g-")
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.1, color="green")
        ax.axhline(22, color="r", linestyle="--")
        ax.set_title("Avg Temp (°C)")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Zone Ratio")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if self.train_comfort_ratios:
            ax.legend(fontsize=8)

        ax = axes[3]
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-")
        ax.set_title("Eval Reward")
        ax.grid(True, alpha=0.3)

        ax = axes[4]
        if self.entropies:
            ax.plot(self.entropies, "purple")
        ax.set_title("Entropy")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log(f"📊 图表已保存: {save_path}")
        plt.close()

    def plot_combined(self, save_path=None, episode_label=None):
        """
        一张图合并：上方 1×5 训练总览（Train Reward / Avg Temp / Comfort / Eval / Entropy），
        下方 3×2 当前 episode 曲线（温度、奖励、动作、直方图、设定值、风机）。
        每次保存到同一路径，覆盖旧图以节省空间。
        """
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.8], hspace=0.35)

        # ---------- 上方 1×5 训练总览 ----------
        gs_top = gs[0].subgridspec(1, 5, wspace=0.3)
        axes_top = [fig.add_subplot(gs_top[0, j]) for j in range(5)]

        ax = axes_top[0]
        ax.plot(self.train_rewards, "b-", alpha=0.6)
        if len(self.train_rewards) >= 5:
            ma = np.convolve(self.train_rewards, np.ones(5) / 5, mode="valid")
            ax.plot(range(4, len(self.train_rewards)), ma, "r-", linewidth=2)
        ax.set_title("Train Reward")
        ax.grid(True, alpha=0.3)

        ax = axes_top[1]
        ax.plot(self.train_temps, "g-")
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.1, color="green")
        ax.axhline(22, color="r", linestyle="--")
        ax.set_title("Avg Temp (°C)")
        ax.grid(True, alpha=0.3)

        ax = axes_top[2]
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Ratio")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if self.train_comfort_ratios:
            ax.legend(fontsize=7)

        ax = axes_top[3]
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-", markersize=4)
        ax.set_title("Eval Reward")
        ax.grid(True, alpha=0.3)

        ax = axes_top[4]
        if self.entropies:
            ax.plot(self.entropies, "purple")
        ax.set_title("Entropy")
        ax.grid(True, alpha=0.3)

        # ---------- 下方 3×2 当前 episode 曲线 ----------
        gs_bot = gs[1].subgridspec(3, 2, wspace=0.28, hspace=0.35)
        axes_bot = [[fig.add_subplot(gs_bot[i, j]) for j in range(2)] for i in range(3)]
        ep_title = f"Episode {episode_label}" if episode_label is not None else "Latest Episode"

        if self.last_actions:
            steps = range(1, len(self.last_actions) + 1)
            decoded = [action_to_values(a, config.__dict__) for a in self.last_actions]
            fan_seq = [d["fan"] for d in decoded]
            supply_seq = [K2C(d["supply_temp"]) for d in decoded]
            heat_seq = [K2C(d["heat_setpoint"]) for d in decoded]
            cool_seq = [K2C(d["cool_setpoint"]) for d in decoded]

            ax = axes_bot[0][0]
            ax.plot(steps, self.last_temps, label="Room")
            ax.plot(steps, self.last_outdoor, label="Outdoor", alpha=0.5)
            ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.1, color="green")
            ax.axhline(22, color="r", linestyle="--", label="Target 22")
            ax.set_title(f"{ep_title} | Temperature")
            ax.set_xlabel("Step (15min)")
            ax.set_ylabel("°C")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[0][1]
            ax.plot(steps, self.last_rewards, "tab:orange")
            ax.set_title("Step Rewards")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

            ax = axes_bot[1][0]
            ax.scatter(steps, self.last_actions, s=10, alpha=0.7)
            ax.set_title("Action Index per Step")
            ax.set_xlabel("Step")
            ax.set_ylim(-1, max(self.last_actions) + 2 if self.last_actions else 24)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[1][1]
            items = sorted(self.last_action_counts.items(), key=lambda x: x[0])
            if items:
                labels, counts = zip(*items)
                ax.bar(labels, counts, color="tab:blue")
            ax.set_title("Action Histogram")
            ax.set_xlabel("Action idx")
            ax.grid(True, axis="y", alpha=0.3)

            ax = axes_bot[2][0]
            ax.plot(steps, supply_seq, label="TSupply (°C)")
            ax.plot(steps, heat_seq, label="Heat set (°C)")
            ax.plot(steps, cool_seq, label="Cool set (°C)")
            ax.set_title("Supply / Setpoints")
            ax.set_xlabel("Step")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes_bot[2][1]
            ax.plot(steps, fan_seq, label="Fan u", color="tab:purple")
            ax.set_ylim(0, 1.05)
            ax.set_title("Fan Control")
            ax.set_xlabel("Step")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            for i in range(3):
                for j in range(2):
                    axes_bot[i][j].text(0.5, 0.5, "暂无 episode 曲线", ha="center", va="center", transform=axes_bot[i][j].transAxes)
                    axes_bot[i][j].set_xticks([])
                    axes_bot[i][j].set_yticks([])

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log(f"📊 图表已保存（已覆盖）: {save_path}")
        plt.close()

    def plot(self, save_path=None):
        """保留兼容：单独训练总览 1×5。"""
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        ax = axes[0]
        ax.plot(self.train_rewards, "b-", alpha=0.6)
        if len(self.train_rewards) >= 5:
            ma = np.convolve(self.train_rewards, np.ones(5) / 5, mode="valid")
            ax.plot(range(4, len(self.train_rewards)), ma, "r-", linewidth=2)
        ax.set_title("Train Reward")
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(self.train_temps, "g-")
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.1, color="green")
        ax.axhline(22, color="r", linestyle="--")
        ax.set_title("Avg Temp (°C)")
        ax.grid(True, alpha=0.3)
        ax = axes[2]
        if self.train_comfort_ratios:
            ax.plot(self.train_comfort_ratios, "purple", linewidth=1.5)
            ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Comfort Zone Ratio")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if self.train_comfort_ratios:
            ax.legend(fontsize=8)
        ax = axes[3]
        if self.eval_rewards:
            ax.plot(self.eval_episodes, self.eval_rewards, "ro-")
        ax.set_title("Eval Reward")
        ax.grid(True, alpha=0.3)
        ax = axes[4]
        if self.entropies:
            ax.plot(self.entropies, "purple")
        ax.set_title("Entropy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log(f"📊 图表已保存: {save_path}")
        plt.close()

    def plot_episode_curves(self, episode_label, save_dir=None):
        """保留兼容：单独 3×2 episode 曲线。"""
        if not self.last_actions:
            return
        steps = range(1, len(self.last_actions) + 1)
        decoded = [action_to_values(a, config.__dict__) for a in self.last_actions]
        fan_seq = [d["fan"] for d in decoded]
        supply_seq = [K2C(d["supply_temp"]) for d in decoded]
        heat_seq = [K2C(d["heat_setpoint"]) for d in decoded]
        cool_seq = [K2C(d["cool_setpoint"]) for d in decoded]
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        ax = axes[0, 0]
        ax.plot(steps, self.last_temps, label="Room")
        ax.plot(steps, self.last_outdoor, label="Outdoor", alpha=0.5)
        ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.1, color="green")
        ax.axhline(22, color="r", linestyle="--", label="Target 22")
        ax.set_title(f"Episode {episode_label} | Temperature")
        ax.set_xlabel("Step (15min)")
        ax.set_ylabel("°C")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(steps, self.last_rewards, "tab:orange")
        ax.set_title("Step Rewards")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.scatter(steps, self.last_actions, s=12, alpha=0.7)
        ax.set_title("Action Index per Step")
        ax.set_xlabel("Step")
        ax.set_ylim(-1, max(self.last_actions) + 2 if self.last_actions else 24)
        ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        items = sorted(self.last_action_counts.items(), key=lambda x: x[0])
        if items:
            labels, counts = zip(*items)
            ax.bar(labels, counts, color="tab:blue")
        ax.set_title("Action Histogram")
        ax.set_xlabel("Action idx")
        ax.grid(True, axis="y", alpha=0.3)
        ax = axes[2, 0]
        ax.plot(steps, supply_seq, label="TSupply (°C)")
        ax.plot(steps, heat_seq, label="Heat set (°C)")
        ax.plot(steps, cool_seq, label="Cool set (°C)")
        ax.set_title("Supply / Setpoints")
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[2, 1]
        ax.plot(steps, fan_seq, label="Fan u", color="tab:purple")
        ax.set_ylim(0, 1.05)
        ax.set_title("Fan Control")
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"curves_ep{episode_label}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log(f"📊 Episode 曲线已保存: {out}")
        plt.close()
