"""
把 data/auto_dataset/images/*.jpg 统一 letterbox 到 1366x768

原因: 当前数据集混了 3 种分辨率 (2049x1152 / 2060x1115 / 1958x1058),
YOLO 训练时会做内部 letterbox, 但源分辨率不一致会导致 padding 模式不同,
影响模型收敛. 统一到 1366x768 (匹配新游戏窗口) 后:
- 训练和推理看到同样的画布
- HP monitor / minimap 等像素坐标下游有稳定参照
- 标签是 0-1 归一化, 不需要改

用法:
  python scripts/resize_dataset.py --dry-run    # 预览
  python scripts/resize_dataset.py              # 原地覆盖
"""
import argparse
import sys
from pathlib import Path
from collections import Counter
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
IMG_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "images"

CANONICAL_SIZE = (1366, 768)


def parse_args():
    p = argparse.ArgumentParser(description="统一数据集到 1366x768 letterbox")
    p.add_argument("--dry-run", action="store_true", help="只统计不修改")
    p.add_argument("--backup", action="store_true",
                   help="先备份原图到 images_orig/ 再修改")
    return p.parse_args()


def main():
    args = parse_args()

    if not IMG_DIR.is_dir():
        raise SystemExit(f"未找到 {IMG_DIR}")

    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"扫描 {IMG_DIR}: {len(files)} 张")
    print(f"目标尺寸: {CANONICAL_SIZE[0]}x{CANONICAL_SIZE[1]} (letterbox)")
    if args.dry_run:
        print("[DRY-RUN] 不修改任何文件\n")
    elif args.backup:
        backup_dir = IMG_DIR.parent / "images_orig"
        backup_dir.mkdir(exist_ok=True)
        print(f"备份原图到 {backup_dir}/")

    before_sizes = Counter()
    would_change = 0
    processed = 0
    failed = 0

    for i, img_path in enumerate(files, 1):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                before_sizes[(w, h)] += 1
                if (w, h) == CANONICAL_SIZE:
                    continue
                would_change += 1

            if args.dry_run:
                continue

            if args.backup:
                backup_path = IMG_DIR.parent / "images_orig" / img_path.name
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(img_path, backup_path)

            # 用 PIL 重新保存 (letterbox)
            from src.utils.image_utils import letterbox_resize
            letterbox_resize(str(img_path), str(img_path), CANONICAL_SIZE)
            processed += 1

            if i % 200 == 0:
                print(f"  [{i}/{len(files)}] 处理中...")

        except Exception as e:
            print(f"  [失败] {img_path.name}: {e}")
            failed += 1

    print(f"\n--- 源分辨率分布 ---")
    for (w, h), c in before_sizes.most_common():
        marker = " ← 当前规范" if (w, h) == CANONICAL_SIZE else ""
        print(f"  {w}x{h}: {c} 张{marker}")

    print(f"\n--- 改动 ---")
    print(f"  总数: {len(files)}")
    print(f"  需要改: {would_change}")
    print(f"  已改: {processed}")
    print(f"  失败: {failed}")

    if args.dry_run:
        print(f"\n[DRY-RUN] 未修改任何文件. 确认后去掉 --dry-run 执行.")
    else:
        print(f"\n[OK] 已统一到 {CANONICAL_SIZE[0]}x{CANONICAL_SIZE[1]}.")


if __name__ == "__main__":
    main()
