"""
调换角色 (swap_player.py)
==========================

一键在多个玩家模板之间切换。流程:
  1. 按 --player <name> 找模板 (data/player/<name>.json 或 <name>/<name>.json)
  2. 校验模板是否自带名牌图 (身份凭证); 没有则拒绝切换, 除非 --force
  3. 更新 config.yaml 的 player.active_template = <name>
  4. 清空 player_profile 单例缓存 (combat_brain 下次启动读新模板)

名牌 (身份) 不再复制到 models/nametag/nametag.png:
  名牌属于角色, 由 PlayerProfile.identity 随模板加载。全局副本会让"换了角色、
  忘了换名牌"变成一个安静的错误 —— bot 把别人当成自己, 定位错了却什么都不报。

用法:
  python tools/swap_player.py --player player0_warrior        # 切到战士
  python tools/swap_player.py --player player1_gunner         # 切回火枪手
  python tools/swap_player.py --list                          # 列出所有可用模板
  python tools/swap_player.py --current                       # 显示当前激活模板
  python tools/swap_player.py --player player0_warrior --dry-run   # 只打印不执行

模板目录约定:
  data/player/
    player0_warrior/
      nametag.png             # 身份凭证 (随模板加载)
      warrior.json            # 战斗参数
    player1_gunner/
      nametag.png
      gunner.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK, 打 ▶/✓/中文会 UnicodeEncodeError 直接崩 (与 main.py 同处理)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_yaml(path: Path) -> dict:
    """轻量 YAML 读取 (避免与 pyyaml 版本冲突), 只支持本配置用到的标量/字典/列表。"""
    try:
        import yaml
    except ImportError:
        print("[ERR] 需要 PyYAML: pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: dict) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _template_nametag(json_path: Path, data: dict) -> Path | None:
    """解析模板的身份名牌图 —— 与 src/utils/player_profile._resolve_identity 同口径。

    优先级: identity.nametag (相对模板目录/项目根) → <模板目录>/nametag.png
            → <模板名>_nametag.png。都没有返回 None。
    """
    base = json_path.parent
    explicit = (data.get("identity") or {}).get("nametag")
    if explicit:
        p = Path(explicit)
        for cand in ((p,) if p.is_absolute() else (base / p, PROJECT_ROOT / p)):
            if cand.exists():
                return cand
    for cand in (base / "nametag.png", base / f"{json_path.stem}_nametag.png"):
        if cand.exists():
            return cand
    return None


def _list_templates(templates_dir: Path) -> list[dict]:
    """扫 templates_dir, 返回模板信息列表 [{name, json_path, nametag_path, description}]。"""
    out = []
    if not templates_dir.exists():
        return out
    # 1) 单文件模式: data/player/<name>.json
    for p in sorted(templates_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 名牌解析口径与 src/utils/player_profile._resolve_identity 完全一致,
            # 否则"工具说有名牌 / 运行时找不到"这种两套约定的坑会再来一次
            nt = _template_nametag(p, data)
            out.append({
                "name": data.get("template", p.stem),
                "char_class": data.get("class", "?"),
                "json_path": p,
                "nametag_path": nt,
                "description": data.get("description", ""),
                "map": data.get("map", ""),
            })
        except Exception as e:
            print(f"[WARN] 跳过 {p}: {e}")
    # 2) 目录模式: data/player/<name>/<name>.json 或 <name>/<char>.json + nametag.png
    for d in sorted(templates_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
            continue
        # 找模板 JSON (多种命名)
        json_cands = [d / f"{d.name}.json", d / "warrior.json", d / "gunner.json", d / "wizard.json"]
        # 也支持按目录名第一段 (player0_warrior → warrior)
        tail = d.name.split("_", 1)[-1] if "_" in d.name else d.name
        json_cands.append(d / f"{tail}.json")
        json_path = next((p for p in json_cands if p.exists()), None)
        if json_path is None:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nt = _template_nametag(json_path, data)
            out.append({
                "name": data.get("template", d.name),
                "char_class": data.get("class", "?"),
                "json_path": json_path,
                "nametag_path": nt,
                "description": data.get("description", ""),
                "map": data.get("map", ""),
            })
        except Exception as e:
            print(f"[WARN] 跳过 {d}: {e}")
    # 去重 (按 name)
    seen = set()
    uniq = []
    for t in out:
        if t["name"] in seen:
            continue
        seen.add(t["name"])
        uniq.append(t)
    return uniq


def cmd_list(args) -> int:
    cfg = _load_yaml(PROJECT_ROOT / "config.yaml")
    player_section = cfg.get("player", {}) or {}
    templates_dir = player_section.get("templates_dir", "data/player")
    tdir = PROJECT_ROOT / templates_dir
    templates = _list_templates(tdir)
    if not templates:
        print(f"[INFO] 没有在 {tdir} 找到任何模板")
        return 1
    current = player_section.get("active_template", "(none)")
    print(f"模板目录: {tdir}")
    print(f"当前激活: {current}")
    print("─" * 70)
    for t in templates:
        marker = "▶" if t["name"] == current else " "
        nt = "✓" if t["nametag_path"] else "✗"
        print(f"  {marker} {t['name']:<22} [{t['char_class']:<8}] 名牌:{nt} 地图:{t['map']}")
        if t["description"]:
            print(f"    {t['description']}")
        print(f"    json: {t['json_path'].relative_to(PROJECT_ROOT)}")
        if t["nametag_path"]:
            print(f"    nametag: {t['nametag_path'].relative_to(PROJECT_ROOT)}")
    print("─" * 70)
    print("切换: python tools/swap_player.py --player <name>")
    return 0


def cmd_current(args) -> int:
    cfg = _load_yaml(PROJECT_ROOT / "config.yaml")
    current = (cfg.get("player", {}) or {}).get("active_template", "(none)")
    print(f"当前激活模板: {current}")
    return 0


def cmd_swap(args) -> int:
    target = args.player
    cfg_path = PROJECT_ROOT / "config.yaml"
    cfg = _load_yaml(cfg_path)
    player_section = cfg.get("player", {}) or {}
    templates_dir = player_section.get("templates_dir", "data/player")
    tdir = PROJECT_ROOT / templates_dir

    templates = _list_templates(tdir)
    template = next((t for t in templates if t["name"] == target), None)
    if template is None:
        print(f"[ERR] 找不到模板 '{target}' 在 {tdir}")
        print(f"      可用模板: {', '.join(t['name'] for t in templates)}")
        return 1

    print(f"[SWAP] 切换到: {template['name']} ({template['char_class']})")
    print(f"       描述: {template['description']}")
    print(f"       模板: {template['json_path'].relative_to(PROJECT_ROOT)}")

    # 1. 名牌 = 身份凭证, 现在由模板**直接持有** (data/player/<name>/nametag.png),
    #    运行时 PlayerProfile.identity 解析后交给定位器 —— 不再复制到全局
    #    models/nametag/nametag.png。旧流程的隐患: 切到一个没有名牌的模板时,
    #    全局文件仍是上一个角色的名牌, bot 会安静地把别人当成自己。
    if template["nametag_path"] is not None:
        print(f"[OK] 身份名牌: {template['nametag_path'].relative_to(PROJECT_ROOT)} (随模板加载)")
    else:
        print(f"[WARN] ⚠ 模板 '{template['name']}' 没有名牌图!")
        print(f"       玩家定位会降级 (只能靠 v13 身体几何/徽章配对), 多人同屏时容易认错人。")
        print(f"       采集: python tools/capture_nametag.py --player {template['name']}")
        if not args.force and not args.dry_run:
            print(f"       仍要切换请加 --force")
            return 2

    # 2. 更新 config.yaml: player.active_template
    if "player" not in cfg:
        cfg["player"] = {}
    cfg["player"]["active_template"] = template["name"]
    if args.dry_run:
        print(f"[DRY] 将更新 config.yaml: player.active_template = {template['name']}")
    else:
        _save_yaml(cfg_path, cfg)
        print(f"[OK] config.yaml 已更新 (player.active_template = {template['name']})")

    # 3. 清空 player_profile 缓存 (下次启动读新模板)
    if args.dry_run:
        print(f"[DRY] 将清空 player_profile 单例缓存")
    else:
        try:
            from src.utils.player_profile import clear_cache
            clear_cache()
            print(f"[OK] player_profile 缓存已清空, 下次启动生效")
        except Exception as e:
            print(f"[WARN] 清缓存失败 (无影响, 下次启动自然重读): {e}")

    print()
    print(f"启动 bot: python main.py")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="调换玩家模板 (切换角色, 一键)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/swap_player.py --player player0_warrior
  python tools/swap_player.py --list
  python tools/swap_player.py --current
        """)
    parser.add_argument("--player", help="目标模板名 (data/player/<name>.* 或 <name>/<name>.json)")
    parser.add_argument("--list", action="store_true", help="列出所有可用模板")
    parser.add_argument("--current", action="store_true", help="显示当前激活模板")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划, 不实际改动")
    parser.add_argument("--force", action="store_true",
                        help="模板没有名牌图时也强行切换 (玩家定位会降级)")
    args = parser.parse_args()

    if args.list:
        sys.exit(cmd_list(args))
    if args.current:
        sys.exit(cmd_current(args))
    if args.player:
        sys.exit(cmd_swap(args))
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()