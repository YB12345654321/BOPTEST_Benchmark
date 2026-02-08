# -*- coding: utf-8 -*-
"""
A2C 训练脚本（可直接运行，连接 BOPTEST Docker）
在项目根目录执行: python methods/a2c/train.py
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import time
import random
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import config
from env import BOPTESTEnv, safe_json
from utils import log, action_to_string
from monitor import Monitor, step_print

# 本方法参数（同目录 params）
import importlib.util
_spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), "params.py"))
_params = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_params)
LR_ACTOR = _params.LR_ACTOR
LR_CRITIC = _params.LR_CRITIC
ENTROPY_COEF_START = _params.ENTROPY_COEF_START
ENTROPY_COEF_END = _params.ENTROPY_COEF_END
ENTROPY_DECAY_STEPS = _params.ENTROPY_DECAY_STEPS
USE_ENTROPY_DECAY = _params.USE_ENTROPY_DECAY
ENTROPY_COEF_MIN = getattr(_params, "ENTROPY_COEF_MIN", 0.04)
USE_LAYER_NORM = _params.USE_LAYER_NORM
USE_DROPOUT = _params.USE_DROPOUT
DROPOUT_P = _params.DROPOUT_P
HIDDEN_DIM = _params.HIDDEN_DIM
GAMMA = _params.GAMMA
GAE_LAMBDA = _params.GAE_LAMBDA
VALUE_COEF = _params.VALUE_COEF
MAX_GRAD_NORM = _params.MAX_GRAD_NORM
WARMUP_STEPS = _params.WARMUP_STEPS
WARMUP_ACTION = _params.WARMUP_ACTION
PARAM_PRESET = _params.PARAM_PRESET


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, use_norm=True, use_dropout=False):
        super().__init__()
        actor_layers = []
        actor_layers.append(nn.Linear(state_dim, hidden))
        if use_norm:
            actor_layers.append(nn.LayerNorm(hidden))
        actor_layers.append(nn.ReLU())
        if use_dropout:
            actor_layers.append(nn.Dropout(DROPOUT_P))
        actor_layers.append(nn.Linear(hidden, hidden // 2))
        if use_norm:
            actor_layers.append(nn.LayerNorm(hidden // 2))
        actor_layers.append(nn.ReLU())
        if use_dropout:
            actor_layers.append(nn.Dropout(DROPOUT_P))
        actor_layers.append(nn.Linear(hidden // 2, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(state_dim, hidden))
        if use_norm:
            critic_layers.append(nn.LayerNorm(hidden))
        critic_layers.append(nn.ReLU())
        if use_dropout:
            critic_layers.append(nn.Dropout(DROPOUT_P))
        critic_layers.append(nn.Linear(hidden, hidden // 2))
        if use_norm:
            critic_layers.append(nn.LayerNorm(hidden // 2))
        critic_layers.append(nn.ReLU())
        if use_dropout:
            critic_layers.append(nn.Dropout(DROPOUT_P))
        critic_layers.append(nn.Linear(hidden // 2, 1))
        self.critic = nn.Sequential(*critic_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                is_last = m is actor_layers[-1] or m is critic_layers[-1]
                nn.init.xavier_uniform_(m.weight, gain=0.01 if is_last else np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def _compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = []
    gae = 0
    values = list(values) + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def train():
    print("=" * 70, flush=True)
    print("🚀 BOPTEST A2C 训练 (Benchmark)", flush=True)
    print("=" * 70, flush=True)

    save_dir = os.path.join(config.DATA_ROOT, "a2c")
    plot_dir = os.path.join(save_dir, config.PLOT_SUBDIR)
    progress_fname = "a2c_progress.png"
    os.makedirs(plot_dir, exist_ok=True)

    env = BOPTESTEnv()
    monitor = Monitor(save_dir)

    try:
        resp = requests.get(f"{config.BOPTEST_URL}/testcases", timeout=10)
        if resp.status_code != 200:
            log(f"❌ 连接失败: {resp.status_code}", "ERROR")
            return
        log(f"✅ 连接成功")
    except Exception as e:
        log(f"❌ 连接失败: {e}", "ERROR")
        return

    model = ActorCritic(env.obs_dim, env.action_dim, hidden=HIDDEN_DIM, use_norm=USE_LAYER_NORM, use_dropout=USE_DROPOUT)
    optimizer = optim.Adam([
        {"params": model.actor.parameters(), "lr": LR_ACTOR},
        {"params": model.critic.parameters(), "lr": LR_CRITIC},
    ])
    log(f"A2C preset={PARAM_PRESET}, LR_ACTOR={LR_ACTOR}, LR_CRITIC={LR_CRITIC}")

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

        states, actions, rewards, dones, log_probs, values, entropies = [], [], [], [], [], [], []
        total_reward = 0.0
        temps, outdoor_seq, rewards_seq, comfort_details, energy_details = [], [], [], [], []
        power_heating_kw, power_cooling_kw, power_fan_kw, co2_ppm_list = [], [], [], []

        # Warmup
        for _ in range(WARMUP_STEPS):
            with torch.no_grad():
                logits, value = model(torch.FloatTensor(state).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
            next_state, reward, done, info = env.step(WARMUP_ACTION)
            states.append(state)
            actions.append(WARMUP_ACTION)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(dist.log_prob(torch.tensor([WARMUP_ACTION])).item())
            values.append(value.item())
            entropies.append(dist.entropy().item())
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
            state = next_state
            if done:
                break

        for step in range(config.STEPS_PER_EPISODE - len(states)):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits, value = model(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, info = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(dist.log_prob(action).item())
            values.append(value.item())
            entropies.append(dist.entropy().item())

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

            if (step + 1 + WARMUP_STEPS) % config.STEP_PRINT_INTERVAL == 0:
                top3 = torch.argsort(probs.squeeze(), descending=True)[:3]
                extra = "Top3概率: [" + ", ".join([f"A{a.item()}:{probs.squeeze()[a].item():.2f}" for a in top3]) + "]"
                step_print(step + 1 + WARMUP_STEPS, info, action.item(), reward, extra_line=extra)

            state = next_state
            if done:
                break

        # A2C update
        with torch.no_grad():
            last_value = model(torch.FloatTensor(state).unsqueeze(0))[1].item()
        adv, returns = _compute_gae(rewards, values, dones, last_value, GAMMA, GAE_LAMBDA)
        ent_coef = ENTROPY_COEF_END + (ENTROPY_COEF_START - ENTROPY_COEF_END) * max(0, 1 - episode / ENTROPY_DECAY_STEPS) if USE_ENTROPY_DECAY else ENTROPY_COEF_START
        ent_coef = max(ent_coef, ENTROPY_COEF_MIN)  # 防止熵过小导致策略坍缩到单一 action
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs)
        returns_t = torch.FloatTensor(returns)
        adv_t = torch.FloatTensor((adv - adv.mean()) / (adv.std() + 1e-8))
        adv_t = torch.clamp(adv_t, -10.0, 10.0)  # 避免极值导致更新震荡
        logits, v = model(states_t)
        dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
        log_probs_new = dist.log_prob(actions_t)
        policy_loss = -(log_probs_new * adv_t).mean()
        value_loss = ((v - returns_t) ** 2).mean()
        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        avg_temp = float(np.mean(temps)) if temps else 0
        comfort_ratio = float(np.mean([1 if config.COMFORT_LOW <= t <= config.COMFORT_HIGH else 0 for t in temps])) if temps else 0
        total_energy = float(sum(energy_details)) if energy_details else 0

        monitor.train_rewards.append(total_reward)
        monitor.train_temps.append(avg_temp)
        monitor.train_comfort_ratios.append(comfort_ratio)
        monitor.train_energy_consumption.append(total_energy)
        monitor.entropies.append(float(np.mean(entropies)) if entropies else 0)
        monitor.log_episode_curves(actions, temps, outdoor_seq, rewards_seq, comfort_details, energy_details)
        monitor.save_episode_data(
            episode, actions, temps, outdoor_seq, rewards_seq, comfort_details, energy_details,
            power_heating_kw=power_heating_kw, power_cooling_kw=power_cooling_kw, power_fan_kw=power_fan_kw,
            start_time_seconds=getattr(env, "current_start_time", None), co2_ppm_list=co2_ppm_list,
        )

        print(f"\n✅ Episode {episode} 完成", flush=True)
        print(f"   总奖励: {total_reward:.1f}", flush=True)
        print(f"   平均室温: {avg_temp:.1f}°C (目标 20-24°C)", flush=True)
        print(f"   舒适区时间比例: {comfort_ratio*100:.1f}%", flush=True)
        print(f"   总能耗: {total_energy:.2f} kW", flush=True)
        print(f"   Loss: {loss.item():.4f} | Entropy: {np.mean(entropies):.4f}", flush=True)
        cnt = Counter(actions)
        n_uniq = len(cnt)
        most_action, most_count = cnt.most_common(1)[0] if cnt else (0, 0)
        most_pct = 100.0 * most_count / len(actions) if actions else 0
        print(f"   动作多样性: {n_uniq} 种 | 最多: action {most_action} 占 {most_pct:.0f}%", flush=True)

        if episode % config.EVAL_FREQUENCY == 0:
            eval_rewards = []
            for start_t in config.EVAL_START_TIMES:
                for _ in range(config.EVAL_EPISODES_PER_START):
                    s = env.reset(start_time=start_t)
                    ep_r = 0
                    for _ in range(config.STEPS_PER_EPISODE):
                        with torch.no_grad():
                            logits, _ = model(torch.FloatTensor(s).unsqueeze(0))
                            a = logits.argmax(dim=-1).item()
                        s, r, d, _ = env.step(a)
                        ep_r += r
                        if d:
                            break
                    eval_rewards.append(ep_r)
            avg_eval = float(np.mean(eval_rewards)) if eval_rewards else 0
            monitor.eval_episodes.append(episode)
            monitor.eval_rewards.append(avg_eval)
            print(f"  🔍 Eval avg reward: {avg_eval:.2f}", flush=True)
            base = progress_fname.replace(".png", "")
            monitor.plot_combined(save_path=os.path.join(plot_dir, f"{base}_ep{episode}.png"), episode_label=episode)

    env.stop()
    monitor.save_training_summary()
    monitor.plot_combined(save_path=os.path.join(plot_dir, progress_fname), episode_label="final")
    monitor.save_individual_plots(save_dir=plot_dir, method_name="a2c")
    log("A2C 训练结束")


if __name__ == "__main__":
    train()
