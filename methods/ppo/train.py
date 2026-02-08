# -*- coding: utf-8 -*-
"""PPO 训练脚本。在项目根目录执行: python methods/ppo/train.py"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import time
import shutil
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import config
from env import BOPTESTEnv
from utils import log, K2C
from monitor import Monitor, step_print

import importlib.util
_spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), "params.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, use_norm=True, use_dropout=True):
        super().__init__()
        al, cl = [], []
        for _list, name in [(al, "actor"), (cl, "critic")]:
            _list.append(nn.Linear(state_dim, hidden))
            if use_norm:
                _list.append(nn.LayerNorm(hidden))
            _list.append(nn.ReLU())
            if use_dropout:
                _list.append(nn.Dropout(0.1))
            _list.append(nn.Linear(hidden, hidden // 2))
            if use_norm:
                _list.append(nn.LayerNorm(hidden // 2))
            _list.append(nn.ReLU())
            if use_dropout:
                _list.append(nn.Dropout(0.1))
        self.actor = nn.Sequential(*al, nn.Linear(hidden // 2, action_dim))
        self.critic = nn.Sequential(*cl, nn.Linear(hidden // 2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01 if m == self.actor[-1] or m == self.critic[-1] else np.sqrt(2))
                nn.init.zeros_(m.bias)

    def get_action(self, state, deterministic=False):
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), probs

    def get_value(self, state):
        return self.critic(state).squeeze(-1)

    def evaluate_action(self, state, action):
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action), dist.entropy(), self.critic(state).squeeze(-1)


def _norm_state(s):
    room_c = K2C(s[0])
    outdoor_c = K2C(s[1])
    return np.array([
        (room_c - 22.0) / 10.0, (outdoor_c - 10.0) / 15.0, s[2], s[3], s[4] / 5.0, (s[5] - 0.6) / 0.6,
        s[6] / 5.0, s[7] / 5.0, s[8] / 5.0, s[9], s[10], s[11],
    ], dtype=np.float32)


def train():
    print("=" * 70, flush=True)
    print("🚀 BOPTEST PPO 训练 (Benchmark)", flush=True)
    print("=" * 70, flush=True)
    save_dir = os.path.join(config.DATA_ROOT, "ppo")
    plot_dir = os.path.join(save_dir, config.PLOT_SUBDIR)
    progress_fname = "ppo_progress.png"
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

    model = ActorCritic(env.obs_dim, env.action_dim, p.HIDDEN_DIM, p.USE_LAYER_NORM, p.USE_DROPOUT)
    optimizer = optim.Adam([
        {"params": model.actor.parameters(), "lr": p.LEARNING_RATE_ACTOR},
        {"params": model.critic.parameters(), "lr": p.LEARNING_RATE_CRITIC},
    ])

    for episode in range(1, config.TOTAL_EPISODES + 1):
        print(f"\n{'=' * 60}", flush=True)
        print(f"📊 Episode {episode}/{config.TOTAL_EPISODES}", flush=True)
        print(f"{'=' * 60}", flush=True)
        try:
            state = env.reset()
        except Exception as e:
            log(f"❌ Reset 失败: {e}", "ERROR")
            env.stop()
            env.testid = None
            time.sleep(2)
            continue

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        total_reward = 0.0
        temps, outdoor_seq, rewards_seq, comfort_details, energy_details = [], [], [], [], []
        power_heating_kw, power_cooling_kw, power_fan_kw, co2_ppm_list = [], [], [], []

        for step in range(config.STEPS_PER_EPISODE):
            st = torch.FloatTensor(_norm_state(state)).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, probs = model.get_action(st, False)
                value = model.get_value(st)
            next_state, reward, done, info = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

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

            if (step + 1) % config.STEP_PRINT_INTERVAL == 0:
                top3 = torch.argsort(probs.squeeze(), descending=True)[:3]
                extra = "Top3概率: [" + ", ".join([f"A{a.item()}:{probs.squeeze()[a].item():.2f}" for a in top3]) + "]"
                step_print(step + 1, info, action.item(), reward, extra_line=extra)

            state = next_state
            if done:
                break

        # GAE & returns
        with torch.no_grad():
            last_v = model.get_value(torch.FloatTensor(_norm_state(state)).unsqueeze(0)).item()
        values = values + [last_v]
        advantages, returns = [], []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + p.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + p.GAMMA * p.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        adv = torch.FloatTensor(advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.FloatTensor(returns)
        states_t = torch.FloatTensor(np.array([_norm_state(s) for s in states]))
        actions_t = torch.LongTensor(actions)
        old_lp = torch.FloatTensor(log_probs)
        ent_coef = p.ENTROPY_COEF_END + (p.ENTROPY_COEF_START - p.ENTROPY_COEF_END) * max(0, 1 - episode / p.ENTROPY_DECAY_STEPS) if p.USE_ENTROPY_DECAY else p.ENTROPY_COEF_START

        for _ in range(p.PPO_EPOCHS):
            perm = np.random.permutation(len(states))
            for start in range(0, len(states), p.MINI_BATCH_SIZE):
                idx = perm[start: start + p.MINI_BATCH_SIZE]
                new_lp, entropy, new_v = model.evaluate_action(states_t[idx], actions_t[idx])
                ratio = torch.exp(new_lp - old_lp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - p.PPO_CLIP, 1 + p.PPO_CLIP) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((new_v - ret[idx]) ** 2).mean()
                loss = policy_loss + p.VALUE_COEF * value_loss - ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), p.MAX_GRAD_NORM)
                optimizer.step()

        avg_temp = float(np.mean(temps)) if temps else 0
        comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in temps])) if temps else 0
        total_energy = float(sum(energy_details)) if energy_details else 0
        monitor.train_rewards.append(total_reward)
        monitor.train_temps.append(avg_temp)
        monitor.train_comfort_ratios.append(comfort_ratio)
        monitor.train_energy_consumption.append(total_energy)
        monitor.log_episode_curves(actions, temps, outdoor_seq, rewards_seq, comfort_details, energy_details)
        monitor.save_episode_data(
            episode, actions, temps, outdoor_seq, rewards_seq, comfort_details, energy_details,
            power_heating_kw=power_heating_kw, power_cooling_kw=power_cooling_kw, power_fan_kw=power_fan_kw,
            start_time_seconds=getattr(env, "current_start_time", None), co2_ppm_list=co2_ppm_list,
        )
        print(f"\n✅ Episode {episode} 完成 | 总奖励: {total_reward:.1f} | 平均室温: {avg_temp:.1f}°C | 舒适区: {comfort_ratio*100:.1f}%", flush=True)
        print(f"   使用了 {len(Counter(actions))} 种不同动作", flush=True)

        if episode % config.EVAL_FREQUENCY == 0:
            eval_r = []
            for start_t in config.EVAL_START_TIMES:
                for _ in range(config.EVAL_EPISODES_PER_START):
                    s = env.reset(start_time=start_t)
                    er = 0
                    for _ in range(config.STEPS_PER_EPISODE):
                        st = torch.FloatTensor(_norm_state(s)).unsqueeze(0)
                        with torch.no_grad():
                            a, _, _, _ = model.get_action(st, True)
                        s, r, d, _ = env.step(a.item())
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
    monitor.save_individual_plots(save_dir=plot_dir, method_name="ppo")
    log("PPO 训练结束")


if __name__ == "__main__":
    train()
