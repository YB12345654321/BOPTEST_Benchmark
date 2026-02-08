# -*- coding: utf-8 -*-
"""Q-Table 训练脚本。在项目根目录执行: python methods/qtable/train.py"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import requests
import numpy as np
from collections import Counter

import config
from env import BOPTESTEnv
from utils import log, K2C
from monitor import Monitor, step_print

import importlib.util
_spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), "params.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


def discretize(obs):
    room_c = K2C(obs[0])
    out_c = K2C(obs[1])
    sin_t, cos_t = obs[9], obs[10]
    hour = (np.degrees(np.arctan2(sin_t, cos_t)) % 360) / 15

    def bucket(val, edges):
        return int(np.digitize([val], edges)[0])

    if p.QT_STATE_MODE == "compact":
        temp_err = room_c - p.TEMP_TARGET_C
        return (bucket(temp_err, p.BUCKETS["temp_err"]), bucket(out_c, p.BUCKETS["out_temp"]), bucket(hour, p.BUCKETS["time_bin"]))
    return (
        bucket(room_c, p.BUCKETS["room_temp"]), bucket(out_c, p.BUCKETS["out_temp"]),
        bucket(obs[2], p.BUCKETS["rel_hum"]), bucket(obs[6], p.BUCKETS["p_heating"]),
        bucket(obs[7], p.BUCKETS["p_cooling"]), bucket(obs[8], p.BUCKETS["p_fan"]),
        bucket(hour, p.BUCKETS["time_bin"]),
    )


class QTableAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.table = {}

    def act(self, disc_state, eps):
        key = tuple(disc_state)
        if np.random.rand() < eps or key not in self.table:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.table[key]))

    def update(self, disc_state, action, reward, disc_next, done, alpha, gamma):
        key = tuple(disc_state)
        next_key = tuple(disc_next)
        if key not in self.table:
            self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
        if next_key not in self.table:
            self.table[next_key] = np.zeros(self.n_actions, dtype=np.float32)
        q_sa = self.table[key][action]
        q_next = 0.0 if done else np.max(self.table[next_key])
        self.table[key][action] = (1 - alpha) * q_sa + alpha * (reward + gamma * q_next)


def linear_eps(step):
    return max(p.EPS_END, p.EPS_START - (p.EPS_START - p.EPS_END) * step / p.EPS_DECAY_STEPS)


def train():
    print("=" * 70, flush=True)
    print("🚀 BOPTEST Q-Table 训练 (Benchmark)", flush=True)
    print("=" * 70, flush=True)
    save_dir = os.path.join(config.DATA_ROOT, "qtable")
    plot_dir = os.path.join(save_dir, config.PLOT_SUBDIR)
    progress_fname = "qtable_progress.png"
    os.makedirs(plot_dir, exist_ok=True)
    env = BOPTESTEnv()
    monitor = Monitor(save_dir)
    try:
        r = requests.get(f"{config.BOPTEST_URL}/testcases", timeout=10)
        if r.status_code != 200:
            log("❌ 连接失败", "ERROR")
            return
        log("✅ 连接成功")
    except Exception as e:
        log(f"❌ 连接失败: {e}", "ERROR")
        return

    agent = QTableAgent(config.NUM_ACTIONS)
    global_step = 0

    for episode in range(1, config.TOTAL_EPISODES + 1):
        print(f"\n{'=' * 60}", flush=True)
        print(f"📊 Episode {episode}/{config.TOTAL_EPISODES}", flush=True)
        print(f"{'=' * 60}", flush=True)
        try:
            obs = env.reset()
        except Exception as e:
            log(f"❌ Reset 失败: {e}", "ERROR")
            env.stop()
            env.testid = None
            continue

        disc = discretize(obs)
        total_reward = 0.0
        temps, outdoor_seq, rewards_seq, comfort_details, energy_details, actions_list = [], [], [], [], [], []
        power_heating_kw, power_cooling_kw, power_fan_kw, co2_ppm_list = [], [], [], []

        for step in range(config.STEPS_PER_EPISODE):
            eps = linear_eps(global_step)
            action = agent.act(disc, eps)
            next_obs, reward, done, info = env.step(action)
            disc_next = discretize(next_obs)
            alpha = max(p.ALPHA_MIN, p.ALPHA * (p.ALPHA_DECAY ** global_step))
            agent.update(disc, action, reward, disc_next, done, alpha, p.GAMMA)
            disc = disc_next
            global_step += 1
            total_reward += reward
            temps.append(info["room_temp"])
            outdoor_seq.append(info["outdoor_temp"])
            rewards_seq.append(reward)
            rd = info.get("reward_detail", {})
            comfort_details.append(rd.get("comfort", 0))
            energy_details.append(rd.get("p_sum", 0))
            power_heating_kw.append(rd.get("p_h", 0))
            power_cooling_kw.append(rd.get("p_c", 0))
            power_fan_kw.append(rd.get("p_f", 0))
            co2_ppm_list.append(rd.get("co2_ppm", 600.0))
            actions_list.append(action)

            if (step + 1) % config.STEP_PRINT_INTERVAL == 0:
                key = tuple(disc)
                if key in agent.table:
                    qv = agent.table[key]
                    top3 = np.argsort(qv)[-3:][::-1]
                    extra = "Top3 Q: [" + ", ".join([f"A{a}:Q={qv[a]:.2f}" for a in top3]) + f"] | eps={eps:.3f}"
                else:
                    extra = f"eps={eps:.3f}"
                step_print(step + 1, info, action, reward, extra_line=extra)

            obs = next_obs
            disc = disc_next
            if done:
                break

        avg_temp = float(np.mean(temps)) if temps else 0
        comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in temps])) if temps else 0
        total_energy = float(sum(energy_details)) if energy_details else 0
        monitor.train_rewards.append(total_reward)
        monitor.train_temps.append(avg_temp)
        monitor.train_comfort_ratios.append(comfort_ratio)
        monitor.train_energy_consumption.append(total_energy)
        monitor.log_episode_curves(actions_list, temps, outdoor_seq, rewards_seq, comfort_details, energy_details)
        monitor.save_episode_data(
            episode, actions_list, temps, outdoor_seq, rewards_seq, comfort_details, energy_details,
            power_heating_kw=power_heating_kw, power_cooling_kw=power_cooling_kw, power_fan_kw=power_fan_kw,
            start_time_seconds=getattr(env, "current_start_time", None), co2_ppm_list=co2_ppm_list,
        )
        print(f"\n✅ Episode {episode} 完成 | 总奖励: {total_reward:.1f} | 平均室温: {avg_temp:.1f}°C | 舒适区: {comfort_ratio*100:.1f}%", flush=True)
        print(f"   Epsilon: {eps:.3f} | Q-states: {len(agent.table)} | 动作数: {len(Counter(actions_list))}", flush=True)

        if episode % config.EVAL_FREQUENCY == 0:
            eval_r = []
            for start_t in config.EVAL_START_TIMES:
                for _ in range(config.EVAL_EPISODES_PER_START):
                    s = env.reset(start_time=start_t)
                    ds = discretize(s)
                    er = 0
                    for _ in range(config.STEPS_PER_EPISODE):
                        a = agent.act(ds, 0.0)
                        s, r, d, _ = env.step(a)
                        ds = discretize(s)
                        er += r
                        if d:
                            break
                    eval_r.append(er)
            avg_eval = float(np.mean(eval_r)) if eval_r else 0
            monitor.eval_episodes.append(episode)
            monitor.eval_rewards.append(avg_eval)
            print(f"  🔍 Eval avg reward: {avg_eval:.2f}（贪心+固定4日）| 训练回报含探索+随机起始日", flush=True)
            monitor.plot_combined(save_path=os.path.join(plot_dir, progress_fname), episode_label=episode)

    env.stop()
    monitor.save_training_summary()
    monitor.plot_combined(save_path=os.path.join(plot_dir, progress_fname), episode_label="final")
    log("Q-Table 训练结束")


if __name__ == "__main__":
    train()
