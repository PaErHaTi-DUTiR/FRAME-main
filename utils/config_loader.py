from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_run_config(global_config_path: str | Path, experiment_config_path: str | Path) -> Dict[str, Any]:
    global_cfg = load_yaml(global_config_path)
    exp_cfg = load_yaml(experiment_config_path)
    return deep_merge(global_cfg, exp_cfg)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
