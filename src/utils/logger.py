"""
日志系统 — 基于 loguru，统一项目日志格式。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import PROJECT_ROOT, load_config


_initialized = False


def setup_logger() -> None:
    """初始化全局日志配置。只执行一次。"""
    global _initialized
    if _initialized:
        return

    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file", "logs/agent.log")
    rotation = log_cfg.get("rotation", "10 MB")

    # 构建日志文件绝对路径
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 移除默认 handler，重新配置
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # 文件输出（带轮转）
    logger.add(
        str(log_path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
        rotation=rotation,
        encoding="utf-8",
    )

    _initialized = True
    logger.info("日志系统初始化完成")


def get_logger(name: str = "ready_player_one"):
    """获取带有模块名称的 logger 实例。"""
    setup_logger()
    return logger.bind(name=name)
