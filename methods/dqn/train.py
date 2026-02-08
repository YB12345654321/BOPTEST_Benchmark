# -*- coding: utf-8 -*-
"""DQN 训练脚本。在项目根目录执行: python methods/dqn/train.py"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import time
import random
import shutil
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import config
from env import BOPTESTEnv
from utils import log
from monitor import Monitor, step_print

import importlib.util
_spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), "params.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01 if m is self.net[-1] else np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, done, next_obs):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        n = self.capacity if self.full else self.idx
        idxs = np.random.choice(n, batch_size, replace=False)
        return (
            torch.FloatTensor(self.obs[idxs]),
            torch.LongTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.dones[idxs]),
            torch.FloatTensor(self.next_obs[idxs]),
        )


def linear_eps(step):
    return max(p.EPS_END, p.EPS_START - (p.EPS_START - p.EPS_END) * step / p.EPS_DECAY_STEPS)


def train():
    print("=" * 70, flush=True)
    print("🚀 BOPTEST DQN 训练 (Benchmark)", flush=True)
    print("=" * 70, flush=True)
    save_dir = os.path.join(config.DATA_ROOT, "dqn")
    plot_dir = os.path.join(save_dir, config.PLOT_SUBDIR)
    progress_fname = "dqn_progress.png"
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

    q_net = QNetwork(env.obs_dim, env.action_dim, p.HIDDEN_DIM)
    target_net = QNetwork(env.obs_dim, env.action_dim, p.HIDDEN_DIM)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=p.LR)
    buffer = ReplayBuffer(p.BUFFER_SIZE, env.obs_dim)
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
            time.sleep(2)
            continue

        total_reward = 0.0
        temps, outdoor_seq, rewards_seq, comfort_details, energy_details, actions_list = [], [], [], [], [], []
        power_heating_kw, power_cooling_kw, power_fan_kw, co2_ppm_list = [], [], [], []

        for _ in range(p.WARMUP_STEPS):
            next_obs, reward, done, info = env.step(p.WARMUP_ACTION)
            buffer.add(obs, p.WARMUP_ACTION, reward, done, next_obs)
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
            actions_list.append(p.WARMUP_ACTION)
            obs = next_obs
            global_step += 1
            if done:
                break

        for step in range(config.STEPS_PER_EPISODE - len(actions_list)):
            eps = linear_eps(global_step)
            if random.random() < eps:
                action = random.randrange(env.action_dim)
            else:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(obs).unsqueeze(0))
                    action = q.argmax(dim=1).item()
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, done, next_obs)
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

            if (step + 1 + p.WARMUP_STEPS) % config.STEP_PRINT_INTERVAL == 0:
                with torch.no_grad():
                    qv = q_net(torch.FloatTensor(obs).unsqueeze(0)).squeeze()
                    top3 = torch.argsort(qv, descending=True)[:3]
                    extra = "Top3 Q: [" + ", ".join([f"A{a.item()}:{qv[a].item():.2f}" for a in top3]) + f"] | eps={eps:.3f}"
                step_print(step + 1 + p.WARMUP_STEPS, info, action, reward, extra_line=extra)

            if buffer.full or buffer.idx >= p.MIN_REPLAY_SIZE:
                ob, ac, rw, dn, no = buffer.sample(p.BATCH_SIZE)
                with torch.no_grad():
                    next_q = target_net(no).max(1)[0]
                    target = rw + p.GAMMA * (1 - dn) * next_q
                q_sa = q_net(ob).gather(1, ac.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if global_step % p.TARGET_UPDATE == 0:
                    for tp, op in zip(target_net.parameters(), q_net.parameters()):
                        tp.data.copy_(p.TAU * op.data + (1 - p.TAU) * tp.data)

            obs = next_obs
            global_step += 1
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
        print(f"\n✅ Episode {episode} 完成 | 总奖励: {total_reward:.1f} | 平均室温: {avg_temp:.1f}°C | 舒适区: {comfort_ratio*100:.1f}% | 能耗: {total_energy:.2f} kW", flush=True)
        print(f"   Epsilon: {linear_eps(global_step):.3f} | Buffer: {buffer.idx if not buffer.full else buffer.capacity}/{buffer.capacity} | 动作数: {len(Counter(actions_list))}", flush=True)

        if episode % config.EVAL_FREQUENCY == 0:
            eval_r = []
            for start_t in config.EVAL_START_TIMES:
                for _ in range(config.EVAL_EPISODES_PER_START):
                    s = env.reset(start_time=start_t)
                    er = 0
                    for _ in range(config.STEPS_PER_EPISODE):
                        with torch.no_grad():
                            a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(dim=1).item()
                        s, r, d, _ = env.step(a)
                        er += r
                        if d:
                            break
                    eval_r.append(er)
            avg_eval = float(np.mean(eval_r)) if eval_r else 0
            monitor.eval_episodes.append(episode)
            monitor.eval_rewards.append(avg_eval)
            print(f"  🔍 Eval avg reward: {avg_eval:.2f}", flush=True)
            base = progress_fname.replace(".png", "")
            combined_path = os.path.join(plot_dir, f"{base}_ep{episode}.png")
            monitor.plot_combined(save_path=combined_path, episode_label=episode)
            eval_folder = os.path.join(plot_dir, f"eval_ep{episode}")
            os.makedirs(eval_folder, exist_ok=True)
            shutil.copy(combined_path, os.path.join(eval_folder, "progress_combined.png"))
            monitor.save_individual_plots(save_dir=eval_folder, method_name="", episode_label=episode)

    env.stop()
    monitor.save_training_summary()
    monitor.plot_combined(save_path=os.path.join(plot_dir, progress_fname), episode_label="final")
    monitor.save_individual_plots(save_dir=plot_dir, method_name="dqn")
    log("DQN 训练结束")


if __name__ == "__main__":
    train()
