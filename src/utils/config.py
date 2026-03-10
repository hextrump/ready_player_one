"""
配置加载器 — 读取 config.yaml 并提供全局访问接口。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# 项目根目录 (ready_player_one/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_config: dict | None = None


def load_config(config_path: str | Path | None = None) -> dict:
    """
    加载 YAML 配置文件。默认读取项目根目录的 config.yaml。
    配置会被缓存，重复调用返回同一份字典。
    """
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    # 将相对路径转换为绝对路径
    _resolve_paths(_config)
    return _config


def get(key: str, default: Any = None) -> Any:
    """
    用点号分隔的路径获取配置值。
    示例: get("yolo.model_path") → "models/yolo/best.pt"
    """
    cfg = load_config()
    keys = key.split(".")
    value = cfg
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        else:
            return default
    return value


def _resolve_paths(cfg: dict) -> None:
    """将配置中的相对路径解析为绝对路径。"""
    path_keys = [
        ("yolo", "model_path"),
        ("state_bus", "db_path"),
    ]
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            p = Path(cfg[section][key])
            if not p.is_absolute():
                cfg[section][key] = str(PROJECT_ROOT / p)


def reload() -> dict:
    """强制重新加载配置。"""
    global _config
    _config = None
    return load_config()
