# -*- coding: utf-8 -*-
"""统一 BOPTEST 环境（五方法共用，与 config 一致）。"""
import time
import numpy as np
import requests

import config
from utils import log, K2C, action_to_values


def safe_json(resp, ctx=""):
    try:
        return resp.json()
    except Exception as e:
        log(f"JSON 解析失败({ctx}): {type(e).__name__}: {e}. status={resp.status_code}, text={resp.text[:300]}", "ERROR")
        return None


class BOPTESTEnv:
    def __init__(self):
        self.url = config.BOPTEST_URL
        self.testcase = config.TESTCASE_NAME
        self.testid = None
        self.obs_dim = 12
        self.action_dim = config.NUM_ACTIONS
        self.prev_action_norm = 0.0
        self.episode_start_time = None
        self.current_start_time = 0  # 当前 episode 起始 sim_time (秒)，供 monitor 保存
        self.steps_per_episode = config.STEPS_PER_EPISODE
        self.comfort_low = config.COMFORT_LOW
        self.comfort_high = config.COMFORT_HIGH

    def _select_testcase(self):
        log(f"选择测试案例: {self.testcase}")
        for attempt in range(3):
            try:
                log(f"尝试选择测试案例 (尝试 {attempt + 1}/3)...")
                resp = requests.post(f"{self.url}/testcases/{self.testcase}/select", timeout=120)
                if resp.status_code == 200:
                    data = safe_json(resp, "select testcase")
                    if data:
                        self.testid = data.get("testid") or data.get("testcaseid")
                        if self.testid:
                            log(f"✅ Test ID: {self.testid}")
                            time.sleep(3)
                            return True
                else:
                    log(f"❌ 选择失败，状态码={resp.status_code}, 响应={resp.text[:200]}", "ERROR")
            except Exception as e:
                log(f"❌ 选择失败: {e}", "ERROR")
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
        return False

    def reset(self, start_time=None):
        log("开始重置环境...")
        if start_time is None:
            if config.RANDOMIZE_START_TIME_TRAIN:
                day = int(np.random.randint(config.TRAIN_START_DAY_MIN, config.TRAIN_START_DAY_MAX + 1))
                start_time = int(day * 86400)
            else:
                start_time = 0
        start_time = int(start_time)

        if not self.testid:
            if not self._select_testcase():
                raise RuntimeError("无法选择测试案例")

        try:
            log(f"  → Initialize... (start_time={start_time}s)")
            resp = requests.put(
                f"{self.url}/initialize/{self.testid}",
                json={"start_time": start_time, "warmup_period": 0},
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Initialize 失败: {resp.status_code}, {resp.text[:200]}")
            log("  ✓ Initialize 成功")
            resp = requests.put(f"{self.url}/step/{self.testid}", json={"step": 900}, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Set step 失败: {resp.status_code}")
            log("  ✓ Set step 成功")
            log("  → Advance...")
            resp = requests.post(f"{self.url}/advance/{self.testid}", json={}, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Advance 失败: {resp.status_code}")
            payload = resp.json().get("payload", {})
            self.prev_action_norm = 0.0
            self.episode_start_time = payload.get("time", start_time)
            self.current_start_time = int(start_time)
            log("✅ 环境重置成功")
            return self._get_obs(payload)
        except Exception as e:
            log(f"❌ 重置失败: {e}", "ERROR")
            raise

    def step(self, action_idx):
        v = action_to_values(action_idx, config.__dict__)
        control = {
            "fcu_oveFan_u": v["fan"], "fcu_oveFan_activate": 1,
            "fcu_oveTSup_u": v["supply_temp"], "fcu_oveTSup_activate": 1,
            "con_oveTSetHea_u": v["heat_setpoint"], "con_oveTSetHea_activate": 1,
            "con_oveTSetCoo_u": v["cool_setpoint"], "con_oveTSetCoo_activate": 1,
        }
        resp = requests.post(f"{self.url}/advance/{self.testid}", json=control, timeout=60)
        payload = resp.json().get("payload", {}) or {}
        obs = self._get_obs(payload)
        reward, detail = self._calc_reward(obs)
        sim_time = payload.get("time", 0)
        if self.episode_start_time is None:
            self.episode_start_time = sim_time
        done = (sim_time - self.episode_start_time) >= self.steps_per_episode * 900
        info = {
            "sim_time": sim_time,
            "room_temp": K2C(obs[0]),
            "outdoor_temp": K2C(obs[1]),
            "reward_detail": detail,
        }
        self.prev_action_norm = action_idx / max(1, self.action_dim - 1)
        return obs, reward, done, info

    def _safe_get(self, payload, key, default):
        v = payload.get(key, default)
        return default if v is None else v

    def _get_obs(self, payload):
        if not payload:
            return np.zeros(self.obs_dim, dtype=np.float32)
        sim_time = payload.get("time", 0)
        hour = (sim_time % 86400) / 3600.0
        room_temp = self._safe_get(payload, "zon_reaTRooAir_y", 293.15)
        outdoor_temp = self._safe_get(payload, "zon_weaSta_reaWeaTDryBul_y", 283.15)
        rel_hum = self._safe_get(payload, "zon_weaSta_reaWeaRelHum_y", 0.5)
        solar = self._safe_get(payload, "zon_weaSta_reaWeaHGloHor_y", 0.0)
        wind = self._safe_get(payload, "zon_weaSta_reaWeaWinSpe_y", 0.0)
        co2 = self._safe_get(payload, "zon_reaCO2RooAir_y", 600.0)
        p_heating = self._safe_get(payload, "fcu_reaPHea_y", 0.0)
        p_cooling = self._safe_get(payload, "fcu_reaPCoo_y", 0.0)
        p_fan = self._safe_get(payload, "fcu_reaPFan_y", 0.0)
        obs = np.array([
            room_temp, outdoor_temp, rel_hum, solar / 1000.0, wind, co2 / 1000.0,
            p_heating / 1000.0, p_cooling / 1000.0, p_fan / 1000.0,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            self.prev_action_norm,
        ], dtype=np.float32)
        return obs

    def _calc_reward(self, obs):
        room_temp_c = K2C(obs[0])
        p_h, p_c, p_f = obs[6], obs[7], obs[8]
        low, high = self.comfort_low, self.comfort_high
        if low <= room_temp_c <= high:
            comfort = 5.0
        else:
            d = (low - room_temp_c) if room_temp_c < low else (room_temp_c - high)
            comfort = -2.5 * (d ** 2)
        if low <= room_temp_c <= high:
            energy = -0.03 * (p_h + p_c) - 0.007 * p_f
        else:
            energy = -0.015 * (p_h + p_c) - 0.003 * p_f
        smooth = -0.005 * abs(obs[-1] - self.prev_action_norm)
        reward = float(np.clip(comfort + energy + smooth, -60.0, 12.0))
        # 功率 kW（obs 里已是 kW），供 benchmark 与论文指标使用
        detail = {
            "comfort": comfort, "energy": energy, "smooth": smooth,
            "room_temp_c": room_temp_c, "in_comfort_zone": low <= room_temp_c <= high,
            "p_h": p_h, "p_c": p_c, "p_f": p_f, "p_sum": p_h + p_c + p_f,
            "co2_ppm": float(obs[5] * 1000.0),  # IAQ，论文可选（obs[5] 为 co2/1000）
        }
        return reward, detail

    def stop(self):
        if self.testid:
            try:
                requests.put(f"{self.url}/stop/{self.testid}", timeout=10)
                log(f"✅ 环境已停止 (testid: {self.testid})")
            except Exception as e:
                log(f"停止失败: {e}", "WARN")
            finally:
                self.testid = None
