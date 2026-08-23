"""
玩家模板加载器 (Player Profile)
==============================

按 config.yaml 的 player.active_template 字段加载对应 JSON, 给 combat_brain / patrol_mover
喂可调换的角色战斗参数。

调用方式:
    from src.utils.player_profile import get_profile
    profile = get_profile()           # 单例 (模块级缓存, 第一次调用时加载)
    profile.attack_range_x_bullet     # 直接读属性
    profile.monster_class_ids         # {1..7} 或 {5} 按模板

模板 JSON 路径: data/player/<template>.json 或 <template>/<template>.json
  - 单文件模式: data/player/player0_warrior.json
  - 目录模式:   data/player/player0_warrior/warrior.json (config 指 templates_dir 时)

切换模板 (调换角色): tools/swap_player.py --player player0_warrior

身份 (identity) 设计:
  名牌 = 角色的身份凭证。它必须**属于模板**, 不能是全局可变文件 —— 否则
  "换角色但没换名牌" 会让 bot 把别人当成自己 (定位错人 → 决策全错)。
  故 PlayerProfile.identity.nametag_path 由模板目录解析 (模板同目录 nametag.png
  或 JSON 的 identity.nametag 字段), 由 NametagHSVLocator 直接消费。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from src.utils.logger import get_logger

log = get_logger("player_profile")

# 默认模板名 (config 缺字段时兜底)
DEFAULT_TEMPLATE = "player1_gunner"


@dataclass
class CombatParams:
    """战斗相关参数 (来自模板 combat 段)。"""
    mode: str = "ranged"                 # ranged / melee
    primary_key: str = "b"               # 主攻击键
    ranged_key: str | None = "b"         # 远程键 (None=禁用远程路径)
    attack_range_x: int = 350            # 主距离 (ranged=远, melee=近)
    attack_range_y: int = 30             # 垂直容差
    attack_range_buffer: int = 45        # 滞回缓冲
    jump_attack_enabled: bool = True     # 跳发 (FLAT_MODE=True 时关闭)
    jump_attack_range_y_up: int = 120    # 跳发垂直上限
    flat_mode: bool = False              # True=关闭跳跃/爬梯 (平地地图)
    burst_interval: float = 0.65
    burst_jitter: float = 0.20
    burst_hold_min: float = 0.025
    burst_hold_max: float = 0.055
    burst_pause_every: int = 5
    burst_pause_min: float = 0.7
    burst_pause_max: float = 1.5
    burst_recheck: float = 0.40
    burst_timeout: float = 8.0
    patrol_chase_range: float = 450.0    # 巡逻方向偏置距离上限
    patrol_attack_range: float = 220.0   # 巡逻时多少 px 内才普攻
    patrol_stuck_timeout: float = 1.0
    # 巡逻: 目标驱动 (看地形 → 选最远落脚点 → 走过去)。
    # 定时换向的老做法不知道自己在哪、走了多远, 表现就是原地左摇右晃。
    patrol_goal_timeout: float = 20.0      # 单个目标最多走多久 (走不到就换)
    patrol_goal_min_gain: float = 250.0    # 目标至少要比当前位置远这么多才值得走
    patrol_goal_explore_push: float = 900.0  # 地形选不出目标时, 朝当前方向推多远继续探索
    patrol_pause_chance: float = 0.10
    patrol_pause_min: float = 1.0
    patrol_pause_max: float = 3.0
    approach_max_sec: float = 3.0
    stuck_jump_timeout: float = 1.0
    edge_jump_max_dy: int = 140
    edge_jump_max_gap: int = 60
    edge_jump_cooldown: float = 1.5
    jump_up_dy: int = 120
    jump_to_upper_dx: int = 150
    jump_down_dy: int = 30
    jump_down_dx: int = 80
    # False = 不用 ↓+Alt 下跳, 改为走出平台边缘自然落下 (默认)。
    # 下跳在最底层是空动作, 而地形层会把同一层的几像素抖动误判成"下面还有一层"
    # → 站在地面上反复下跳 = 原地抽搐。只有"正下方隔着单向平台、走不到边缘"的图才需要开。
    allow_jump_down: bool = False
    # 兜底药水 (HP/MP 自动喝 + 10 分钟兜底)
    hp_potion_interval_sec: float = 600.0
    hp_threshold: float = 0.5
    mp_threshold: float = 0.3
    # 怪识别模型 (不同模板可用不同模型: 战士→v19 单类, 火枪手→010001010 多类)
    monster_model_path: str = "models/super_brain_010001010.pt"
    monster_model_imgsz: int = 640
    monster_conf_threshold: float = 0.20      # 模型调用 conf (原始 YOLO 推理门槛)
    monster_filter_conf: float = 0.20      # 起批门槛: 新目标要拿到这个分才算数
    # 维持门槛: 已确认的目标掉到这个分仍然跟踪 (track-before-detect)。
    # 检测器在阈值附近抖动是常态 —— 用同一个门槛既起批又维持, 会让怪一帧在一帧不在,
    # 决策层就打打停停。默认取起批门槛的一半。
    monster_maintain_conf: float = 0.10
    monster_min_size: int = 25               # 怪框最小尺寸
    monster_aspect_min: float = 0.45
    monster_aspect_max: float = 2.2


@dataclass
class MonsterModelInfo:
    """怪识别模型信息 (来自模板 monster_model 段)。"""
    path: str = "models/super_brain_010001010.pt"
    imgsz: int = 640
    class_id: int | None = None     # None=全部 {1..7}; v19 单类 = 0
    class_name: str = "Monster"
    comment: str = ""
    conf_threshold: float = 0.15    # 模型调用 conf (原始)
    min_size: int = 25
    aspect_min: float = 0.45
    aspect_max: float = 2.2


@dataclass
class TargetMonster:
    """目标怪物 (来自模板 target_monster 段)。model_class_id=None → 不限制。"""
    name: str = "any"
    model_class_id: int | None = None     # None = 全部 {1..7}


@dataclass
class PlayerIdentity:
    """角色身份凭证 (名牌图) — 绑定到模板, 不是全局可变文件。

    nametag_path: 模板自带名牌裁剪图的绝对路径 (不存在则 None)。
    resolved_from: 解析来源 (日志/自检用: "identity.nametag" / "<模板目录>/nametag.png" / "缺失")。
    char_name:     角色名 (可选, 仅日志显示)。
    """
    nametag_path: str | None = None
    resolved_from: str = "缺失"
    char_name: str = ""

    @property
    def has_nametag(self) -> bool:
        return bool(self.nametag_path) and Path(self.nametag_path).exists()


@dataclass
class PlayerProfile:
    """当前激活的玩家模板 (单例)。"""
    template: str = DEFAULT_TEMPLATE
    char_class: str = "gunner"
    description: str = ""
    map_name: str = ""
    template_path: str = ""     # 模板 JSON 绝对路径 (自检/日志用)
    identity: PlayerIdentity = field(default_factory=PlayerIdentity)
    monster_model: MonsterModelInfo = field(default_factory=MonsterModelInfo)
    target: TargetMonster = field(default_factory=TargetMonster)
    combat: CombatParams = field(default_factory=CombatParams)
    monster_class_ids: Set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5, 6, 7})

    # 属性别名 (代码中读 self.profile.xxx 更直观)
    @property
    def attack_range_x_bullet(self) -> int:
        """远程 b 距离 (兼容旧字段名)。melee 模式下取 attack_range_x。"""
        if self.combat.ranged_key is None:
            # 战士无远程 → bullet 路径强制 0, 让决策层只走 melee 分支
            return 0
        return self.combat.attack_range_x if self.combat.mode == "ranged" else 0

    @property
    def attack_range_x_melee(self) -> int:
        """近战 x 距离。"""
        return self.combat.attack_range_x if self.combat.mode == "melee" else min(120, self.combat.attack_range_x)

    @property
    def engage_range_x(self) -> int:
        """【攻击范围的唯一真源】能打到怪的最大水平距离 = 本模板任一可用攻击键的射程。

        战士: melee 120, bullet 0 → 120。 火枪手: melee 120, bullet 350 → 350。
        决策层"进入攻击"与执行层"目标是否还在打得到"必须都用它, 否则
        决策说打、执行说够不着 → 0 击空转死循环 (2026-08 实测过的老 bug)。
        """
        return max(self.attack_range_x_melee, self.attack_range_x_bullet)

    @property
    def flat_mode(self) -> bool:
        return self.combat.flat_mode

    @property
    def primary_key(self) -> str:
        return self.combat.primary_key


def _load_profile_from_json(path: Path) -> PlayerProfile:
    """读模板 JSON → PlayerProfile。失败兜底默认枪手配置。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error(f"[PROFILE] 读模板失败 {path}: {e}, 兜底默认枪手")
        return PlayerProfile()

    target_data = data.get("target_monster", {}) or {}
    combat_data = data.get("combat", {}) or {}
    healer_data = data.get("healer", {}) or {}
    monster_model_data = data.get("monster_model", {}) or {}

    monster_model = MonsterModelInfo(
        path=monster_model_data.get("path", "models/super_brain_010001010.pt"),
        imgsz=monster_model_data.get("imgsz", 640),
        class_id=monster_model_data.get("class_id"),
        class_name=monster_model_data.get("class_name", "Monster"),
        comment=monster_model_data.get("comment", ""),
        conf_threshold=monster_model_data.get("conf_threshold", 0.15),  # 模型调用 conf
        min_size=monster_model_data.get("min_size", 25),
        aspect_min=monster_model_data.get("aspect_min", 0.45),
        aspect_max=monster_model_data.get("aspect_max", 2.2),
    )

    combat = CombatParams(
        mode=combat_data.get("mode", "ranged"),
        primary_key=combat_data.get("primary_key", "b"),
        ranged_key=combat_data.get("ranged_key"),
        attack_range_x=combat_data.get("attack_range_x", 350),
        attack_range_y=combat_data.get("attack_range_y", 30),
        attack_range_buffer=combat_data.get("attack_range_buffer", 45),
        jump_attack_enabled=combat_data.get("jump_attack_enabled", True),
        jump_attack_range_y_up=combat_data.get("jump_attack_range_y_up", 120),
        flat_mode=combat_data.get("flat_mode", False),
        burst_interval=combat_data.get("burst_interval", 0.65),
        burst_jitter=combat_data.get("burst_jitter", 0.20),
        burst_hold_min=combat_data.get("burst_hold_min", 0.025),
        burst_hold_max=combat_data.get("burst_hold_max", 0.055),
        burst_pause_every=combat_data.get("burst_pause_every", 5),
        burst_pause_min=combat_data.get("burst_pause_min", 0.7),
        burst_pause_max=combat_data.get("burst_pause_max", 1.5),
        burst_recheck=combat_data.get("burst_recheck", 0.40),
        burst_timeout=combat_data.get("burst_timeout", 8.0),
        patrol_chase_range=combat_data.get("patrol_chase_range", 450.0),
        patrol_attack_range=combat_data.get("patrol_attack_range", 220.0),
        patrol_goal_timeout=combat_data.get("patrol_goal_timeout", 20.0),
        patrol_goal_min_gain=combat_data.get("patrol_goal_min_gain", 250.0),
        patrol_goal_explore_push=combat_data.get("patrol_goal_explore_push", 900.0),
        patrol_pause_chance=combat_data.get("patrol_pause_chance", 0.10),
        patrol_pause_min=combat_data.get("patrol_pause_min", 1.0),
        patrol_pause_max=combat_data.get("patrol_pause_max", 3.0),
        approach_max_sec=combat_data.get("approach_max_sec", 3.0),
        stuck_jump_timeout=combat_data.get("stuck_jump_timeout", 1.0),
        edge_jump_max_dy=combat_data.get("edge_jump_max_dy", 140),
        edge_jump_max_gap=combat_data.get("edge_jump_max_gap", 60),
        edge_jump_cooldown=combat_data.get("edge_jump_cooldown", 1.5),
        jump_up_dy=combat_data.get("jump_up_dy", 120),
        jump_to_upper_dx=combat_data.get("jump_to_upper_dx", 150),
        jump_down_dy=combat_data.get("jump_down_dy", 30),
        jump_down_dx=combat_data.get("jump_down_dx", 80),
        allow_jump_down=combat_data.get("allow_jump_down", False),
        hp_potion_interval_sec=healer_data.get("hp_potion_interval_sec", 600.0),
        hp_threshold=healer_data.get("hp_threshold", 0.5),
        mp_threshold=healer_data.get("mp_threshold", 0.3),
        monster_model_path=monster_model.path,
        monster_model_imgsz=monster_model.imgsz,
        monster_conf_threshold=monster_model.conf_threshold,
        monster_filter_conf=monster_model_data.get("filter_conf", 0.20),
        monster_maintain_conf=monster_model_data.get("maintain_conf",
                                                     monster_model_data.get("filter_conf", 0.20) * 0.5),
        monster_min_size=monster_model.min_size,
        monster_aspect_min=monster_model.aspect_min,
        monster_aspect_max=monster_model.aspect_max,
    )

    target = TargetMonster(
        name=target_data.get("name", "any"),
        model_class_id=target_data.get("model_class_id", monster_model.class_id),
    )

    identity = _resolve_identity(path, data.get("identity", {}) or {})

    # 怪物类集合逻辑:
    # 1) 模板 target_monster.model_class_id 显式给出 → {cid}
    # 2) 模板 monster_model.class_id 给出 (v19 单类 0) → {cid}
    # 3) 都没给 → 全部 1-7 (010001010 多类怪)
    explicit_cid = target_data.get("model_class_id", monster_model.class_id)
    if explicit_cid is not None:
        monster_ids: Set[int] = {int(explicit_cid)}
    else:
        monster_ids = {1, 2, 3, 4, 5, 6, 7}

    return PlayerProfile(
        template=data.get("template", path.stem),
        char_class=data.get("class", "gunner"),
        description=data.get("description", ""),
        map_name=data.get("map", ""),
        template_path=str(path),
        identity=identity,
        monster_model=monster_model,
        target=target,
        combat=combat,
        monster_class_ids=monster_ids,
    )


def _resolve_identity(json_path: Path, identity_data: dict) -> PlayerIdentity:
    """解析角色身份凭证 (名牌图), 优先级:
    1) JSON 的 identity.nametag (相对模板目录, 或项目根/绝对路径)
    2) 模板同目录 nametag.png  (目录模式 data/player/<template>/nametag.png)
    3) 模板同目录 <template>_nametag.png (单文件模式 data/player/<template>.json 旁)
    都没有 → nametag_path=None, 由 NametagHSVLocator 降级 (徽章配对/几何反查) 并告警。

    注意: 这里**不**回落到 models/nametag/nametag.png —— 那个全局文件正是
    "换了角色没换名牌 → 把别人当自己" 的根源 (身份不能靠副作用传递)。
    """
    base = json_path.parent
    explicit = identity_data.get("nametag")
    if explicit:
        p = Path(explicit)
        for cand in ((p,) if p.is_absolute() else (base / p, PROJECT_ROOT_FOR_IDENTITY() / p)):
            if cand.exists():
                return PlayerIdentity(str(cand.resolve()), "identity.nametag",
                                      identity_data.get("char_name", ""))
        log.error(f"[PROFILE] 模板声明了 identity.nametag={explicit} 但文件不存在")

    for cand, tag in ((base / "nametag.png", "<模板目录>/nametag.png"),
                      (base / f"{json_path.stem}_nametag.png", "<模板名>_nametag.png")):
        if cand.exists():
            return PlayerIdentity(str(cand.resolve()), tag, identity_data.get("char_name", ""))

    return PlayerIdentity(None, "缺失", identity_data.get("char_name", ""))


def PROJECT_ROOT_FOR_IDENTITY() -> Path:
    """懒加载项目根 (避免模块级循环 import)。"""
    from src.utils.config import PROJECT_ROOT
    return Path(PROJECT_ROOT)


def _resolve_template_path(template_name: str, templates_dir: str) -> Path | None:
    """按多种约定找模板 JSON:
    1) templates_dir/<name>.json                  (单文件)
    2) templates_dir/<name>/<name>.json           (目录同名)
    3) templates_dir/<name>/<char_class>.json     (目录内按角色名)
    """
    base = Path(templates_dir)
    cands = [
        base / f"{template_name}.json",
        base / template_name / f"{template_name}.json",
        base / template_name / "warrior.json",   # 战士目录兜底
        base / template_name / "gunner.json",    # 火枪手目录兜底
        base / template_name / f"{template_name.split('_', 1)[-1]}.json",  # player0_warrior → warrior.json
    ]
    for p in cands:
        if p.exists():
            return p
    # 目录模式最后兜底: 目录里只有一个 JSON 就用它 (与 tools/swap_player.py 的扫描口径一致,
    # 避免"工具能列出、运行时找不到"的两套约定)
    d = base / template_name
    if d.is_dir():
        jsons = sorted(d.glob("*.json"))
        if len(jsons) == 1:
            return jsons[0]
    return None


# ── 单例缓存 ──
_cached_profile: PlayerProfile | None = None
_cached_template_name: str | None = None


def get_profile(reload: bool = False) -> PlayerProfile:
    """读当前激活的玩家模板 (config.yaml → player.active_template → JSON → PlayerProfile)。"""
    global _cached_profile, _cached_template_name
    if _cached_profile is not None and not reload:
        return _cached_profile

    # 懒加载 config (避免循环 import)
    from src.utils.config import load_config, PROJECT_ROOT
    cfg = load_config()
    player_section = cfg.get("player", {}) or {}
    template_name = player_section.get("active_template", DEFAULT_TEMPLATE)
    templates_dir = player_section.get("templates_dir", "data/player")

    # 相对路径 → 项目根绝对路径
    tdir = Path(templates_dir)
    if not tdir.is_absolute():
        tdir = PROJECT_ROOT / tdir

    json_path = _resolve_template_path(template_name, str(tdir))
    if json_path is None:
        log.error(f"[PROFILE] 找不到模板 '{template_name}' 在 {tdir}, 兜底默认枪手")
        profile = PlayerProfile()
    else:
        profile = _load_profile_from_json(json_path)
        log.info(f"[PROFILE] 已加载模板: {profile.template} ({profile.char_class}) 目标={profile.target.name} "
                 f"mode={profile.combat.mode} key={profile.combat.primary_key} "
                 f"range={profile.combat.attack_range_x} engage={profile.engage_range_x} "
                 f"flat={profile.combat.flat_mode} monster_ids={sorted(profile.monster_class_ids)}")
        log.info(f"[PROFILE] 模板 JSON: {json_path}")
        # 身份自检: 名牌必须属于本模板, 否则玩家定位会锁到别人 (换角色最常见的坑)
        if profile.identity.has_nametag:
            log.info(f"[PROFILE] 身份名牌: {profile.identity.nametag_path} (来源 {profile.identity.resolved_from})")
        else:
            log.warning(f"[PROFILE] ⚠ 模板 {profile.template} 没有绑定名牌图! "
                        f"玩家定位将降级为 徽章配对/几何反查 (精度下降)。"
                        f" 修法: python tools/capture_nametag.py --player {profile.template}")

    _cached_profile = profile
    _cached_template_name = template_name
    return profile


def active_template_name() -> str:
    """当前激活的模板名 (用于日志/CLI 显示)。"""
    global _cached_template_name
    if _cached_template_name is None:
        get_profile()
    return _cached_template_name or DEFAULT_TEMPLATE


def clear_cache() -> None:
    """清空单例缓存 (swap_player 切换后调用, 或测试用)。"""
    global _cached_profile, _cached_template_name
    _cached_profile = None
    _cached_template_name = None