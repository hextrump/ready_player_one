"""
从 data/auto_dataset 构建训练集 (Super Brain V13)

约定:
  data/auto_dataset/{images,labels}/ 已经是 V13 标签格式 (类别 ID 0-3 连续).
  本脚本只负责划分 train/val, 不做 remap.

类别 ID:
  0=Player, 1=Monster, 2=Platform, 3=Rope

输入:  data/auto_dataset/{images,labels}/
输出:  data/super_brain_train/{images,labels}/{train,val}/ + dataset.yaml

用法:
  python scripts/build_dataset.py [--ratio 0.8] [--seed 42]
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_IMG = PROJECT_ROOT / "data" / "auto_dataset" / "images"
SRC_LBL = PROJECT_ROOT / "data" / "auto_dataset" / "labels"

OUT = PROJECT_ROOT / "data" / "super_brain_train"
OUT_IMG = {"train": OUT / "images" / "train", "val": OUT / "images" / "val"}
OUT_LBL = {"train": OUT / "labels" / "train", "val": OUT / "labels" / "val"}

CLASS_NAMES = {0: "Player", 1: "Monster", 2: "Platform", 3: "Rope"}
VALID_CLASSES = set(CLASS_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description="Build V13 dataset from auto_dataset")
    p.add_argument("--ratio", type=float, default=0.8, help="train split ratio")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--include-empty", action="store_true",
                   help="把空标签文件 (负样本) 也划入 train/val")
    return p.parse_args()


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in list(OUT_IMG.values()) + list(OUT_LBL.values()):
        d.mkdir(parents=True, exist_ok=True)


def collect_pairs(include_empty: bool):
    pairs = []
    empty = 0
    for img_path in sorted(SRC_IMG.glob("*.jpg")):
        lbl_path = SRC_LBL / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        is_empty = lbl_path.stat().st_size == 0
        if is_empty:
            empty += 1
            if not include_empty:
                continue
        pairs.append((img_path, lbl_path, is_empty))
    return pairs, empty


def validate_label(label_path: Path):
    """校验标签, 返回 (boxes_count, invalid_classes)"""
    n = 0
    invalid = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                invalid.add("non_int")
                continue
            n += 1
            if cls not in VALID_CLASSES:
                invalid.add(cls)
    return n, invalid


def split_and_copy(pairs, train_ratio, rng, include_empty):
    rng.shuffle(pairs)
    cut = int(len(pairs) * train_ratio)
    buckets = {"train": pairs[:cut], "val": pairs[cut:]}

    stats = {
        "train": 0, "val": 0,
        "skipped": 0, "invalid_files": 0,
        "boxes": Counter(),
        "invalid_classes": set(),
    }

    for split, items in buckets.items():
        for img_path, lbl_path, is_empty in items:
            # 校验
            n_boxes, invalid = validate_label(lbl_path)
            if invalid:
                stats["invalid_files"] += 1
                stats["invalid_classes"].update(invalid)
                if "non_int" in invalid or any(isinstance(x, int) for x in invalid):
                    # 有非法类别, 跳过
                    stats["skipped"] += 1
                    continue

            dst_img = OUT_IMG[split] / f"{split}_{img_path.name}"
            dst_lbl = OUT_LBL[split] / f"{split}_{img_path.stem}.txt"
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

            stats[split] += 1
            stats["boxes"].update(_count_classes(lbl_path))

    return stats


def _count_classes(label_path: Path):
    counter = Counter()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                if cls in VALID_CLASSES:
                    counter[cls] += 1
            except ValueError:
                continue
    return counter


def write_yaml():
    yaml_path = OUT / "dataset.yaml"
    rel_out = OUT.relative_to(PROJECT_ROOT).as_posix()
    content = (
        f"path: {rel_out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
    )
    for cid, name in CLASS_NAMES.items():
        content += f"  {cid}: {name}\n"
    yaml_path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if not SRC_IMG.is_dir() or not SRC_LBL.is_dir():
        raise SystemExit(f"输入目录不存在: {SRC_IMG} 或 {SRC_LBL}")

    print(f"[1/4] 清理输出目录: {OUT}")
    reset_output()

    print(f"[2/4] 扫描 {SRC_IMG} ...")
    pairs, empty_count = collect_pairs(args.include_empty)
    print(f"      找到 {len(pairs)} 张 (空标签/负样本: {empty_count}, "
          f"include_empty={args.include_empty})")

    print(f"[3/4] 划分 (train_ratio={args.ratio}, seed={args.seed}) ...")
    stats = split_and_copy(pairs, args.ratio, rng, args.include_empty)
    print(f"      train: {stats['train']} 张")
    print(f"      val:   {stats['val']} 张")
    print(f"      skipped: {stats['skipped']}")
    print("      各类框数:")
    for cid, name in CLASS_NAMES.items():
        print(f"        {cid} ({name}): {stats['boxes'][cid]}")

    if stats["invalid_classes"]:
        print(f"\n[WARN] 发现非法类别 ID: {sorted(stats['invalid_classes'])}")
        print(f"       涉及 {stats['invalid_files']} 个文件 (已跳过)")
        sys.exit(1)

    print(f"[4/4] 写入 dataset.yaml ...")
    write_yaml()
    print(f"      {OUT / 'dataset.yaml'}")

    print("\n[OK] 数据集构建完成")


if __name__ == "__main__":
    main()
