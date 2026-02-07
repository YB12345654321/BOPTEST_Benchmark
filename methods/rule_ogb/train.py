# -*- coding: utf-8 -*-
"""Rule OGB 训练脚本。需安装 realkd：将 BOPTEST_Project 下的 realkd 目录复制到本项目的父目录或本项目根目录，或设置 PYTHONPATH。
在项目根目录执行: python methods/rule_ogb/train.py"""
import sys
import os
import copy
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
# 若未把 realkd 复制到 BOPTEST_Benchmark，则从 BOPTEST_Project 加载
_PROJECT = os.path.join(os.path.dirname(ROOT), "BOPTEST_Project")
if os.path.isdir(_PROJECT) and _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)
os.chdir(ROOT)

import time
import requests
import numpy as np
import pandas as pd
from collections import Counter

import config
from env import BOPTESTEnv
from utils import log, action_to_string
from monitor import Monitor, step_print

import importlib.util
_spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), "params.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

try:
    from realkd.boosting import (
        OrthogonalBoostingObjective,
        FullyCorrective,
        GeneralRuleBoostingEstimator,
    )
    from realkd.rules import SquaredLoss, AdditiveRuleEnsemble, Rule
except ImportError as e:
    log("请将 BOPTEST_Project 中的 realkd 目录复制到 BOPTEST_Benchmark 根目录，或确保 PYTHONPATH 包含 BOPTEST_Project", "ERROR")
    raise SystemExit(1) from e

FEATURES = ["x" + str(i) for i in range(12)]


def _merge_rules_by_query(rules):
    acc = {}
    for r in rules:
        k = str(r.q)
        if k not in acc:
            acc[k] = Rule(q=r.q, y=float(r.y))
        else:
            acc[k].y += float(r.y)
    return list(acc.values())


def _rules_to_weight_map(rules):
    return {str(r.q): float(r.y) for r in rules}


def _apply_damped_weights(old_rules, new_rules, w_star, lr):
    old_w = _rules_to_weight_map(old_rules)
    for i, r in enumerate(new_rules):
        k = str(r.q)
        w_old = old_w.get(k, 0.0)
        r.y = (1.0 - lr) * w_old + lr * float(w_star[i])


def update_rules(estimator, x, y, lr=0.005, x_eval=None, y_eval=None, tol=0.0, keep_if_equal=True):
    loss_func = estimator.loss
    if x_eval is None or y_eval is None:
        x_eval, y_eval = x, y
    old_pred = estimator.rules_(x_eval)
    old_risk = sum(loss_func(y_eval, old_pred))
    old_estimator = copy.deepcopy(estimator)
    cand = copy.deepcopy(estimator)
    old_rules_merged = _merge_rules_by_query(cand.rules_)
    if len(old_rules_merged) > 0:
        weights = np.array([r.y for r in old_rules_merged], dtype=float)
        drop_idx = int(np.argmin(np.abs(weights)))
        kept_rules = [r for j, r in enumerate(old_rules_merged) if j != drop_idx]
    else:
        kept_rules = []
    cand.rules_ = AdditiveRuleEnsemble(kept_rules) if kept_rules else AdditiveRuleEnsemble([])
    cand.fit(x, y, has_origin_rules=True)
    cand_rules_merged = _merge_rules_by_query(list(cand.rules_))
    cand.rules_ = AdditiveRuleEnsemble(cand_rules_merged)
    w_star = FullyCorrective().calc_weight(x, y, cand.rules_)
    w_star = np.asarray(w_star, dtype=float)
    _apply_damped_weights(old_rules_merged, list(cand.rules_), w_star, lr)
    new_pred = cand.rules_(x_eval)
    new_risk = sum(loss_func(y_eval, new_pred))
    improved = (new_risk < old_risk - tol) or (keep_if_equal and new_risk <= old_risk - tol)
    if improved:
        return cand
    w_star = FullyCorrective().calc_weight(x, y, old_estimator.rules_)
    w_star = np.asarray(w_star, dtype=float)
    _apply_damped_weights(old_rules_merged, list(old_estimator.rules_), w_star, lr)
    return old_estimator


def select_action(state_row, epsilon, action_dim, models_s):
    if random.random() > epsilon:
        df = pd.DataFrame(np.array(state_row).reshape(1, -1), columns=FEATURES)
        q_values = np.array([float(np.asarray(m.rules_(df)).flatten()[0]) for m in models_s])
        return int(np.argmax(q_values))
    return random.randrange(action_dim)


def linear_eps(step):
    return max(p.EPS_END, p.EPS_START - (p.EPS_START - p.EPS_END) * step / p.EPS_DECAY_STEPS)


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
        idxs = np.random.choice(n, min(batch_size, n), replace=False)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_obs[idxs],
        )


def train():
    print("=" * 70, flush=True)
    print("🚀 BOPTEST Rule OGB 训练 (Benchmark)", flush=True)
    print("=" * 70, flush=True)
    save_dir = os.path.join(config.DATA_ROOT, "rule_ogb")
    plot_dir = os.path.join(save_dir, config.PLOT_SUBDIR)
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

    models = [
        GeneralRuleBoostingEstimator(
            num_rules=p.NUM_RULES,
            objective_function=OrthogonalBoostingObjective,
            weight_update_method=FullyCorrective(),
            loss=SquaredLoss(),
            reg=p.REG,
            search="greedy",
            max_col_attr=p.NUM_COL,
            verbose=False,
        )
        for _ in range(config.NUM_ACTIONS)
    ]
    buffer = ReplayBuffer(12000, env.obs_dim)
    state_mean = np.zeros(env.obs_dim, dtype=np.float32)
    state_std = np.ones(env.obs_dim, dtype=np.float32)
    state_count = 0
    global_step = 0
    WARMUP_STEPS = 4
    WARMUP_ACTION = 23

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

        total_reward = 0.0
        temps, outdoor_seq, rewards_seq, comfort_details, energy_details, actions_list = [], [], [], [], [], []
        power_heating_kw, power_cooling_kw, power_fan_kw, co2_ppm_list = [], [], [], []

        for _ in range(WARMUP_STEPS):
            next_state, reward, done, info = env.step(WARMUP_ACTION)
            buffer.add(state, WARMUP_ACTION, reward, done, next_state)
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
            actions_list.append(WARMUP_ACTION)
            state = next_state
            global_step += 1
            if done:
                break

        for step in range(config.STEPS_PER_EPISODE - len(actions_list)):
            s_norm = (state - state_mean) / (state_std + 1e-8)
            eps = linear_eps(global_step)
            action = select_action(s_norm, eps, env.action_dim, models)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, float(done), next_state)
            state_count += 1
            alpha = 1.0 / state_count
            state_mean = (1 - alpha) * state_mean + alpha * state
            if state_count > 1:
                state_std = np.sqrt((1 - alpha) * state_std ** 2 + alpha * (state - state_mean) ** 2 + 1e-8)
            state = next_state
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
            global_step += 1

            if (step + 1 + WARMUP_STEPS) % config.STEP_PRINT_INTERVAL == 0:
                df1 = pd.DataFrame(s_norm.reshape(1, -1), columns=FEATURES)
                qv = np.array([float(np.asarray(m.rules_(df1)).flatten()[0]) for m in models])
                top3 = np.argsort(qv)[-3:][::-1]
                extra = "Top3 Q: [" + ", ".join([f"A{a}:{qv[a]:.2f}" for a in top3]) + f"] | eps={eps:.3f}"
                step_print(step + 1 + WARMUP_STEPS, info, action, reward, extra_line=extra)

            if buffer.full or buffer.idx >= p.MIN_REPLAY_SIZE:
                obs_b, act_b, rew_b, done_b, nxt_b = buffer.sample(p.BATCH_SIZE)
                obs_b_norm = (obs_b - state_mean) / (state_std + 1e-8)
                nxt_b_norm = (nxt_b - state_mean) / (state_std + 1e-8)
                batch_state = pd.DataFrame(obs_b_norm, columns=FEATURES)
                batch_next = pd.DataFrame(nxt_b_norm, columns=FEATURES)
                q_vals = np.array([m.rules_(batch_next) for m in models])
                next_q = np.max(q_vals, axis=0)
                target_q = rew_b + p.GAMMA * next_q * (1 - done_b)
                for i in range(len(models)):
                    indices = np.where(act_b == i)[0]
                    if len(indices) < 2:
                        continue
                    x_i = batch_state.iloc[indices]
                    y_i = pd.Series(target_q[indices])
                    if len(list(models[i].rules_)) < p.NUM_RULES:
                        models[i].fit(x_i, y_i)
                    else:
                        models[i] = update_rules(models[i], x_i, y_i)
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
        print(f"   Epsilon: {linear_eps(global_step):.3f} | Buffer: {buffer.idx if not buffer.full else buffer.capacity}/{buffer.capacity} | 动作数: {len(Counter(actions_list))}", flush=True)

        if episode % config.EVAL_FREQUENCY == 0:
            eval_r = []
            for start_t in config.EVAL_START_TIMES:
                for _ in range(config.EVAL_EPISODES_PER_START):
                    s = env.reset(start_time=start_t)
                    er = 0
                    for _ in range(config.STEPS_PER_EPISODE):
                        s_n = (s - state_mean) / (state_std + 1e-8)
                        a = select_action(s_n, 0.0, env.action_dim, models)
                        s, r, d, _ = env.step(a)
                        er += r
                        if d:
                            break
                    eval_r.append(er)
            avg_eval = float(np.mean(eval_r)) if eval_r else 0
            monitor.eval_episodes.append(episode)
            monitor.eval_rewards.append(avg_eval)
            print(f"  🔍 Eval avg reward: {avg_eval:.2f}", flush=True)
            monitor.plot_combined(save_path=os.path.join(plot_dir, "progress.png"), episode_label=episode)

    env.stop()
    monitor.save_training_summary()
    monitor.plot_combined(save_path=os.path.join(plot_dir, "progress.png"), episode_label="final")
    log("Rule OGB 训练结束")


if __name__ == "__main__":
    train()
