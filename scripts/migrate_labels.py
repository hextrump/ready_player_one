"""
一次性迁移脚本: 把 data/auto_dataset/labels/*.txt 的旧标签统一到 V13 体系 (0-3)

旧 → 新:
  0 (Player)    → 0 (Player)    [保持]
  1 (Monster)   → 1 (Monster)   [保持]
  2 (HP)        → [丢弃]
  3 (MP)        → [丢弃]
  4 (Platform)  → 2 (Platform)  [remap]
  5 (Rope)      → 3 (Rope)      [remap]

迁移后: 类别连续 0-3, 与训练 dataset.yaml 一致. 标签文件可能是空 (负样本).

用法:
  python scripts/migrate_labels.py --dry-run    # 只看, 不改
  python scripts/migrate_labels.py              # 执行迁移
"""
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "labels"

REMAP = {0: 0, 1: 1, 4: 2, 5: 3}
DROP = {2, 3}
CLASS_NAMES = {0: "Player", 1: "Monster", 2: "Platform", 3: "Rope"}


def parse_args():
    p = argparse.ArgumentParser(description="迁移 auto_dataset 标签到 V13 (0-3)")
    p.add_argument("--dry-run", action="store_true", help="只统计不修改")
    return p.parse_args()


def migrate_file(label_path: Path, dry_run: bool):
    """返回 (before_counts, after_counts, unknown_classes)"""
    before = Counter()
    after = Counter()
    unknown = set()
    kept_lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            before[cls] += 1
            if cls in DROP:
                continue
            if cls in REMAP:
                new_cls = REMAP[cls]
                kept_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                after[new_cls] += 1
            else:
                unknown.add(cls)

    if not dry_run:
        with open(label_path, "w", encoding="utf-8") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")
            # 否则留空 (负样本)

    return before, after, unknown


def main():
    args = parse_args()
    if not LABEL_DIR.is_dir():
        raise SystemExit(f"未找到 {LABEL_DIR}")

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    print(f"扫描 {LABEL_DIR}: {len(label_files)} 个标签文件")
    if args.dry_run:
        print("[DRY-RUN] 不修改任何文件\n")

    total_before = Counter()
    total_after = Counter()
    total_unknown = set()
    files_changed = 0
    files_emptied = 0

    for lbl in label_files:
        before, after, unknown = migrate_file(lbl, args.dry_run)
        total_before.update(before)
        total_after.update(after)
        total_unknown.update(unknown)
        if before != after:
            files_changed += 1
            if sum(after.values()) == 0:
                files_emptied += 1

    print(f"\n--- 旧标签统计 ---")
    for cls in sorted(total_before):
        name = {0: "Player", 1: "Monster", 2: "HP", 3: "MP", 4: "Platform", 5: "Rope"}.get(cls, f"class{cls}")
        print(f"  {cls} ({name}): {total_before[cls]}")

    print(f"\n--- 新标签统计 ---")
    for cls, name in CLASS_NAMES.items():
        print(f"  {cls} ({name}): {total_after[cls]}")

    if total_unknown:
        print(f"\n[WARN] 发现未知类别: {sorted(total_unknown)}")

    print(f"\n--- 改动文件 ---")
    print(f"  总数: {len(label_files)}")
    print(f"  改动: {files_changed}")
    print(f"  变空 (负样本): {files_emptied}")

    if args.dry_run:
        print(f"\n[DRY-RUN] 未修改任何文件. 确认后去掉 --dry-run 执行.")
    else:
        print(f"\n[OK] 迁移完成. 现在 auto_dataset 标签统一为 0-3.")


if __name__ == "__main__":
    main()
