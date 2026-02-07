# -*- coding: utf-8 -*-
"""共用工具：log、温度换算、动作编解码（依赖 config）。"""
import numpy as np
from datetime import datetime

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level:5s} | {msg}", flush=True)

def K2C(k):
    return k - 273.15

def action_to_values(action_idx, cfg):
    fl = cfg["FAN_LEVELS"]
    st = cfg["SUPPLY_TEMP_LEVELS"]
    ht = cfg["HEAT_SETPOINT_LEVELS"]
    ct = cfg["COOL_SETPOINT_LEVELS"]
    cool_idx = action_idx % len(ct)
    remaining = action_idx // len(ct)
    heat_idx = remaining % len(ht)
    remaining //= len(ht)
    supply_idx = remaining % len(st)
    fan_idx = remaining // len(st)
    v = {
        "fan": fl[fan_idx],
        "supply_temp": st[supply_idx],
        "heat_setpoint": ht[heat_idx],
        "cool_setpoint": ct[cool_idx],
    }
    if v["cool_setpoint"] <= v["heat_setpoint"] + 1.5:
        v["cool_setpoint"] = v["heat_setpoint"] + 1.5
    v["fan"] = float(np.clip(v["fan"], 0.0, 1.0))
    v["supply_temp"] = float(np.clip(v["supply_temp"], 285.15, 313.15))
    v["heat_setpoint"] = float(np.clip(v["heat_setpoint"], 278.15, 308.15))
    v["cool_setpoint"] = float(np.clip(v["cool_setpoint"], 278.15, 308.15))
    return v

def action_to_string(action_idx, cfg):
    v = action_to_values(action_idx, cfg)
    return (
        f"Fan={v['fan']:.1f}|供风={K2C(v['supply_temp']):.0f}°C|"
        f"热设={K2C(v['heat_setpoint']):.1f}°C|冷设={K2C(v['cool_setpoint']):.1f}°C"
    )
