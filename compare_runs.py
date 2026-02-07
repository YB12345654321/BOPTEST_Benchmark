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
    """读取 data/<method>/training_history.csv，含论文指标列（可选，兼容旧 CSV）。"""
    path = os.path.join(config.DATA_ROOT, method, "training_history.csv")
    if not os.path.isfile(path):
        return None
    def safe_float(row, key, default=0.0):
        v = row.get(key, default)
        try:
            return float(v) if v != "" else default
        except (TypeError, ValueError):
            return default
    def safe_int(row, key, default=0):
        v = row.get(key, default)
        try:
            return int(float(v)) if v != "" else default
        except (TypeError, ValueError):
            return default

    episodes, rewards, temps, comfort_ratios, energy = [], [], [], [], []
    energy_kwh, heating_kwh, cooling_kwh, fan_kwh = [], [], [], []
    comfort_violation, rmse_temp, max_temp_dev, action_switches, peak_kw = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            episodes.append(int(row["Episode"]))
            rewards.append(float(row["Reward"]))
            temps.append(float(row["AvgTemp"]))
            comfort_ratios.append(float(row["ComfortRatio"]))
            energy.append(float(row["EnergyConsumption"]))
            energy_kwh.append(safe_float(row, "EnergyKWh"))
            heating_kwh.append(safe_float(row, "HeatingKWh"))
            cooling_kwh.append(safe_float(row, "CoolingKWh"))
            fan_kwh.append(safe_float(row, "FanKWh"))
            comfort_violation.append(safe_float(row, "ComfortViolationRatio"))
            rmse_temp.append(safe_float(row, "RMSE_TempC"))
            max_temp_dev.append(safe_float(row, "MaxTempDeviationC"))
            action_switches.append(safe_int(row, "ActionSwitchCount"))
            peak_kw.append(safe_float(row, "PeakPowerKW"))
    return {
        "episodes": episodes,
        "rewards": rewards,
        "temps": temps,
        "comfort_ratios": comfort_ratios,
        "energy": energy,
        "energy_kwh": energy_kwh,
        "heating_kwh": heating_kwh,
        "cooling_kwh": cooling_kwh,
        "fan_kwh": fan_kwh,
        "comfort_violation_ratios": comfort_violation,
        "rmse_temps": rmse_temp,
        "max_temp_deviations": max_temp_dev,
        "action_switch_counts": action_switches,
        "peak_power_kw": peak_kw,
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

    # ----- 4b. 能耗 (kWh) 对比（论文常用）-----
    has_kwh = any(d.get("history") and d["history"].get("energy_kwh") and any(d["history"]["energy_kwh"]) for d in data.values())
    if has_kwh:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m, d in data.items():
            h = d.get("history")
            if not h or not h.get("energy_kwh"):
                continue
            ax.plot(h["episodes"], h["energy_kwh"], label=METHOD_LABELS.get(m, m))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Energy (kWh)")
        ax.set_title("Total Energy per Episode (kWh) 对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_energy_kwh.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_dir}/compare_energy_kwh.png", flush=True)

    # ----- 4c. 能耗分项（加热/制冷/风机，近 50 轮平均，论文常用）-----
    has_breakdown = any(d.get("history") and d["history"].get("heating_kwh") for d in data.values())
    if has_breakdown:
        fig, ax = plt.subplots(figsize=(10, 5))
        methods_ok = [m for m in METHODS if m in data and data[m].get("history") and (data[m]["history"].get("heating_kwh") or data[m]["history"].get("cooling_kwh"))]
        if methods_ok:
            last50 = 50
            x = np.arange(len(methods_ok))
            w = 0.25
            heat_avg = [np.mean(data[m]["history"]["heating_kwh"][-last50:]) if len(data[m]["history"]["heating_kwh"]) >= last50 else np.mean(data[m]["history"]["heating_kwh"]) or 0 for m in methods_ok]
            cool_avg = [np.mean(data[m]["history"]["cooling_kwh"][-last50:]) if len(data[m]["history"]["cooling_kwh"]) >= last50 else np.mean(data[m]["history"]["cooling_kwh"]) or 0 for m in methods_ok]
            fan_avg = [np.mean(data[m]["history"]["fan_kwh"][-last50:]) if len(data[m]["history"]["fan_kwh"]) >= last50 else np.mean(data[m]["history"]["fan_kwh"]) or 0 for m in methods_ok]
            ax.bar(x - w, heat_avg, w, label="Heating (kWh)", color="tab:red")
            ax.bar(x, cool_avg, w, label="Cooling (kWh)", color="tab:blue")
            ax.bar(x + w, fan_avg, w, label="Fan (kWh)", color="tab:gray")
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_ok])
            ax.set_ylabel("Energy (kWh, avg last 50 ep)")
            ax.set_title("Energy Breakdown 对比（论文指标）")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_energy_breakdown.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_dir}/compare_energy_breakdown.png", flush=True)

    # ----- 4d. 室温 RMSE（相对目标，论文常用）-----
    has_rmse = any(d.get("history") and d["history"].get("rmse_temps") and any(d["history"]["rmse_temps"]) for d in data.values())
    if has_rmse:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m, d in data.items():
            h = d.get("history")
            if not h or not h.get("rmse_temps"):
                continue
            ax.plot(h["episodes"], h["rmse_temps"], label=METHOD_LABELS.get(m, m))
        ax.set_xlabel("Episode")
        ax.set_ylabel("RMSE Temperature (°C)")
        ax.set_title("Setpoint Tracking RMSE (vs 22°C) 对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_rmse_temp.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_dir}/compare_rmse_temp.png", flush=True)

    # ----- 4e. 舒适违反比例、动作切换次数、峰值功率 -----
    if has_rmse or has_kwh:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # Comfort violation
        for m, d in data.items():
            h = d.get("history")
            if not h:
                continue
            v = h.get("comfort_violation_ratios", [])
            if v:
                axes[0].plot(h["episodes"], v, label=METHOD_LABELS.get(m, m))
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Comfort Violation Ratio")
        axes[0].set_title("Comfort Violation 对比")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        # Action switch count
        for m, d in data.items():
            h = d.get("history")
            if not h:
                continue
            v = h.get("action_switch_counts", [])
            if v:
                axes[1].plot(h["episodes"], v, label=METHOD_LABELS.get(m, m))
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Action Switch Count")
        axes[1].set_title("Control Stability (动作切换次数)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        # Peak power
        for m, d in data.items():
            h = d.get("history")
            if not h:
                continue
            v = h.get("peak_power_kw", [])
            if v:
                axes[2].plot(h["episodes"], v, label=METHOD_LABELS.get(m, m))
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Peak Power (kW)")
        axes[2].set_title("Peak Power 对比")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_paper_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_dir}/compare_paper_metrics.png", flush=True)

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

    # ----- 汇总表 + 论文用表格 (CSV) -----
    print("\n" + "=" * 60, flush=True)
    print("汇总表（最后一轮 / 近 50 轮平均）", flush=True)
    print("=" * 60, flush=True)
    last50 = 50
    paper_rows = []
    for m in METHODS:
        if m not in data:
            continue
        h, s = data[m]["history"], data[m].get("summary")
        if not h:
            continue
        n = len(h["rewards"])
        L = min(last50, n)
        r, t, c, e = h["rewards"], h["temps"], h["comfort_ratios"], h["energy"]
        print(f"\n  {METHOD_LABELS.get(m, m)}:", flush=True)
        print(f"    Episode 数: {n}", flush=True)
        print(f"    最后一轮: Reward={r[-1]:.1f}, Temp={t[-1]:.1f}°C, Comfort={c[-1]*100:.1f}%, Energy={e[-1]:.2f} kW", flush=True)
        print(f"    近{L}轮平均: Reward={np.mean(r[-L:]):.1f}, Temp={np.mean(t[-L:]):.1f}°C, Comfort={np.mean(c[-L:])*100:.1f}%", flush=True)
        if s and s.get("training_history", {}).get("eval_rewards"):
            ev = s["training_history"]["eval_rewards"]
            print(f"    最近 Eval 平均 Reward: {ev[-1]:.2f}", flush=True)
        # 论文用一行（近 50 轮平均）
        fs = s.get("final_stats", {}) if s else {}
        row = {
            "method": METHOD_LABELS.get(m, m),
            "episodes": n,
            "avg_reward": fs.get("avg_reward", np.mean(r[-L:]) if r else 0),
            "avg_temp_c": fs.get("avg_temp", np.mean(t[-L:]) if t else 0),
            "avg_comfort_ratio": fs.get("avg_comfort_ratio", np.mean(c[-L:]) if c else 0),
            "avg_energy_kwh": fs.get("avg_energy_kwh", np.mean(h.get("energy_kwh", [0])[-L:]) if h.get("energy_kwh") else 0),
            "avg_heating_kwh": fs.get("avg_heating_kwh", np.mean(h.get("heating_kwh", [0])[-L:]) if h.get("heating_kwh") else 0),
            "avg_cooling_kwh": fs.get("avg_cooling_kwh", np.mean(h.get("cooling_kwh", [0])[-L:]) if h.get("cooling_kwh") else 0),
            "avg_fan_kwh": fs.get("avg_fan_kwh", np.mean(h.get("fan_kwh", [0])[-L:]) if h.get("fan_kwh") else 0),
            "avg_comfort_violation_ratio": fs.get("avg_comfort_violation_ratio", np.mean(h.get("comfort_violation_ratios", [0])[-L:]) if h.get("comfort_violation_ratios") else 0),
            "avg_rmse_temp_c": fs.get("avg_rmse_temp_c", np.mean(h.get("rmse_temps", [0])[-L:]) if h.get("rmse_temps") else 0),
            "avg_max_temp_deviation_c": fs.get("avg_max_temp_deviation_c", np.mean(h.get("max_temp_deviations", [0])[-L:]) if h.get("max_temp_deviations") else 0),
            "avg_action_switch_count": fs.get("avg_action_switch_count", np.mean(h.get("action_switch_counts", [0])[-L:]) if h.get("action_switch_counts") else 0),
            "avg_peak_power_kw": fs.get("avg_peak_power_kw", np.mean(h.get("peak_power_kw", [0])[-L:]) if h.get("peak_power_kw") else 0),
        }
        paper_rows.append(row)

    if paper_rows:
        paper_path = os.path.join(out_dir, "paper_benchmark_table.csv")
        with open(paper_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(paper_rows[0].keys()))
            w.writeheader()
            w.writerows(paper_rows)
        print(f"\n  论文用汇总表已保存: {paper_path}", flush=True)

    print(f"\n📁 对比图已保存到: {os.path.abspath(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
