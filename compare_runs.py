# -*- coding: utf-8 -*-
"""
五方法对比脚本：在全部方法都跑完后，在项目根目录执行
  python compare_runs.py
会读取 data/<method>/training_history.csv 和 training_summary.json，生成对比图与汇总表。
"""
import os
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

METHODS = ["a2c", "dqn", "ppo", "qtable", "rule_ogb"]
METHOD_LABELS = {"a2c": "A2C", "dqn": "DQN", "ppo": "PPO", "qtable": "Q-Table", "rule_ogb": "OGB"}


def load_history(method):
    """读取 data/<method>/training_history.csv，返回 episodes, rewards, temps, comfort_ratios, energy。"""
    path = os.path.join(config.DATA_ROOT, method, "training_history.csv")
    if not os.path.isfile(path):
        return None
    episodes, rewards, temps, comfort_ratios, energy = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            episodes.append(int(row["Episode"]))
            rewards.append(float(row["Reward"]))
            temps.append(float(row["AvgTemp"]))
            comfort_ratios.append(float(row["ComfortRatio"]))
            energy.append(float(row["EnergyConsumption"]))
    return {
        "episodes": episodes,
        "rewards": rewards,
        "temps": temps,
        "comfort_ratios": comfort_ratios,
        "energy": energy,
    }


def load_summary(method):
    """读取 data/<method>/training_summary.json，取 eval 与 final_stats。"""
    path = os.path.join(config.DATA_ROOT, method, "training_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60, flush=True)
    print("📊 BOPTEST 五方法对比", flush=True)
    print("=" * 60, flush=True)

    data = {}
    for m in METHODS:
        h = load_history(m)
        s = load_summary(m)
        if h is None and s is None:
            print(f"  ⚠ 未找到 {METHOD_LABELS.get(m, m)} 数据，跳过", flush=True)
            continue
        data[m] = {"history": h, "summary": s}
        print(f"  ✓ {METHOD_LABELS.get(m, m)}: {len(h['episodes']) if h else 0} episodes", flush=True)

    if not data:
        print("未找到任何方法的训练数据，请先运行各方法的 train.py。", flush=True)
        return

    # 对比图保存目录
    out_dir = os.path.join(config.DATA_ROOT, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ----- 1. 训练奖励对比 -----
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, d in data.items():
        h = d["history"]
        if not h:
            continue
        ep, r = h["episodes"], h["rewards"]
        ax.plot(ep, r, alpha=0.5, label=METHOD_LABELS.get(m, m))
        if len(r) >= 10:
            ma = np.convolve(r, np.ones(10) / 10, mode="valid")
            ax.plot(ep[9:], ma, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Train Reward")
    ax.set_title("Training Reward 对比（细线=原始，粗线=MA(10)）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_reward.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {out_dir}/compare_reward.png", flush=True)

    # ----- 2. 平均温度对比 -----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(config.COMFORT_LOW, config.COMFORT_HIGH, alpha=0.15, color="green", label="舒适区 20–24°C")
    for m, d in data.items():
        h = d["history"]
        if not h:
            continue
        ax.plot(h["episodes"], h["temps"], label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Temperature (°C)")
    ax.set_title("Average Room Temperature 对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_temp.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {out_dir}/compare_temp.png", flush=True)

    # ----- 3. 舒适区比例对比 -----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.7, label="目标 80%")
    for m, d in data.items():
        h = d["history"]
        if not h:
            continue
        ax.plot(h["episodes"], h["comfort_ratios"], label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Comfort Zone Ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comfort Zone Ratio 对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_comfort_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {out_dir}/compare_comfort_ratio.png", flush=True)

    # ----- 4. 能耗对比 -----
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, d in data.items():
        h = d["history"]
        if not h:
            continue
        ax.plot(h["episodes"], h["energy"], label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy Consumption (kW)")
    ax.set_title("Energy Consumption 对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_energy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {out_dir}/compare_energy.png", flush=True)

    # ----- 5. Eval Reward 对比（有 eval 数据的方法）-----
    has_eval = any(d.get("summary") and d["summary"].get("training_history", {}).get("eval_rewards") for d in data.values())
    if has_eval:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m, d in data.items():
            s = d.get("summary")
            if not s:
                continue
            th = s.get("training_history", {})
            ev_ep, ev_r = th.get("eval_episodes"), th.get("eval_rewards")
            if ev_ep and ev_r:
                ax.plot(ev_ep, ev_r, "o-", label=METHOD_LABELS.get(m, m))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval Reward")
        ax.set_title("Evaluation Reward 对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_eval_reward.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_dir}/compare_eval_reward.png", flush=True)

    # ----- 汇总表 -----
    print("\n" + "=" * 60, flush=True)
    print("汇总表（最后一轮 / 近 50 轮平均）", flush=True)
    print("=" * 60, flush=True)
    for m in METHODS:
        if m not in data:
            continue
        h, s = data[m]["history"], data[m].get("summary")
        if not h:
            continue
        r, t, c, e = h["rewards"], h["temps"], h["comfort_ratios"], h["energy"]
        n = len(r)
        last50 = min(50, n)
        print(f"\n  {METHOD_LABELS.get(m, m)}:", flush=True)
        print(f"    Episode 数: {n}", flush=True)
        print(f"    最后一轮: Reward={r[-1]:.1f}, Temp={t[-1]:.1f}°C, Comfort={c[-1]*100:.1f}%, Energy={e[-1]:.2f} kW", flush=True)
        print(f"    近{last50}轮平均: Reward={np.mean(r[-last50:]):.1f}, Temp={np.mean(t[-last50:]):.1f}°C, Comfort={np.mean(c[-last50:])*100:.1f}%", flush=True)
        if s and s.get("training_history", {}).get("eval_rewards"):
            ev = s["training_history"]["eval_rewards"]
            print(f"    最近 Eval 平均 Reward: {ev[-1]:.2f}", flush=True)

    print(f"\n📁 对比图已保存到: {os.path.abspath(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
