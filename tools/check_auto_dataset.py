"""
可视化检查 auto_dataset 标注

类别 ID (V13): 0=Player, 1=Monster, 2=Platform, 3=Rope

用法:
  python tools/check_auto_dataset.py                  # 交互检查所有
  python tools/check_auto_dataset.py --limit 50       # 只看前 50 张
  python tools/check_auto_dataset.py --save           # 导出到 check_output/
  python tools/check_auto_dataset.py --no-show --save # 批量导出 (无弹窗)
"""
import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "auto_dataset"
DEFAULT_OUTPUT = DEFAULT_DATASET / "check_output"

CLASS_NAMES = {
    0: "Player",
    1: "Monster",
    2: "Platform",
    3: "Rope",
}
COLORS = {
    0: (255, 0, 0),       # Blue - Player
    1: (0, 0, 255),       # Red - Monster
    2: (0, 255, 0),       # Green - Platform
    3: (255, 255, 0),     # Cyan - Rope
}


def yolo_to_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes):
    vis = img.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        color = COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description="可视化检查 auto_dataset 标注")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="最多检查 N 张")
    parser.add_argument("--save", action="store_true", help="保存标注可视化图")
    parser.add_argument("--no-show", action="store_true", help="不弹窗 (与 --save 配合)")
    args = parser.parse_args()

    img_dir = args.dataset / "images"
    lbl_dir = args.dataset / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"未找到图片目录: {img_dir}")

    images = sorted(p for p in img_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)

    print(f"检查 {len(images)} 张图片 (来源: {img_dir})")

    for i, img_path in enumerate(images):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i+1}] 跳过 (无法读取): {img_path.name}")
            continue

        h, w = img.shape[:2]
        boxes = yolo_to_boxes(lbl_path, w, h)
        cls_counts = {CLASS_NAMES.get(c, c): n for c, n in
                      {b[0]: sum(1 for x in boxes if x[0] == b[0]) for b in boxes}.items()}
        print(f"  [{i+1}/{len(images)}] {img_path.name} | {cls_counts}")

        if args.save:
            vis = draw_boxes(img, boxes)
            cv2.imwrite(str(args.output / img_path.name), vis)
            continue

        if not args.no_show:
            vis = draw_boxes(img, boxes)
            cv2.imshow("check (q=quit)", vis)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                print("已退出")
                break

    if args.save:
        print(f"\n可视化图已保存到: {args.output}")


if __name__ == "__main__":
    main()
