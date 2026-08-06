"""
用 terrain_v6 自动给 auto_dataset 标注 Platform (class 2) / Rope (class 3)

terrain_v6.pt 类别: {0: Platform, 1: Rope}
V13 类别映射:       0→2 (Platform), 1→3 (Rope)

行为:
- 读取 data/auto_dataset/images/*.jpg
- 用 terrain_v6 推理 (imgsz=1366, 匹配规范画布)
- 把预测框**追加**到 data/auto_dataset/labels/*.txt (保留已有的 Player/Monster 标注)
- 默认 conf=0.25 (可调)
- 默认 NMS IoU=0.45 (YOLO 默认)

可视化:
- 把带框的可视化图保存到 data/auto_dataset/sample_preview/auto_terrain/

用法:
  python scripts/auto_terrain.py --limit 2                # 试 2 张
  python scripts/auto_terrain.py --limit 100 --conf 0.3   # 100 张, 提高阈值
  python scripts/auto_terrain.py                           # 处理全部
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMG_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "images"
LBL_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "labels"
PREVIEW_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "sample_preview" / "auto_terrain"

# terrain_v6 (0=Platform, 1=Rope) -> V13 (2=Platform, 3=Rope)
MODEL_TO_V13 = {0: 2, 1: 3}
V13_NAMES = {2: "Platform", 3: "Rope"}
V13_COLORS = {2: (0, 170, 0), 3: (0, 170, 170)}


def parse_args():
    p = argparse.ArgumentParser(description="自动标注 Platform/Rope")
    p.add_argument("--limit", type=int, default=None, help="最多处理 N 张")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=1366, help="推理尺寸")
    p.add_argument("--offset", type=int, default=0, help="从第 N 张开始 (用于分批)")
    p.add_argument("--no-write", action="store_true", help="不写 label, 仅可视化")
    p.add_argument("--preview-only", action="store_true", help="只保存预览图, 不写 label")
    return p.parse_args()


def append_to_label(label_path: Path, new_lines: list[str]):
    """追加新行到标签文件 (保留已有行)"""
    with open(label_path, "a", encoding="utf-8") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")


def visualize(img_path, label_path, out_path):
    """画所有 bbox (Player/Monster/Platform/Rope) 到预览图"""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # 颜色 (BGR for cv2)
    all_colors = {
        0: (204, 102, 0),    # Player - 蓝
        1: (0, 0, 204),      # Monster - 红
        2: (0, 170, 0),      # Platform - 绿
        3: (170, 170, 0),    # Rope - 青
    }
    all_names = {0: "Player", 1: "Monster", 2: "Platform", 3: "Rope"}

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                if cls not in all_colors:
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = all_colors[cls]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, all_names[cls], (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(out_path), img)


def main():
    args = parse_args()

    from ultralytics import YOLO
    model = YOLO("models/terrain_v6.pt")
    print(f"加载 terrain_v6.pt: {model.names}")
    print(f"参数: conf={args.conf}, imgsz={args.imgsz}")

    files = sorted(IMG_DIR.glob("*.jpg"))
    if args.offset:
        files = files[args.offset:]
    if args.limit:
        files = files[: args.limit]
    print(f"处理 {len(files)} 张图片")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    write_labels = not (args.no_write or args.preview_only)
    if not write_labels:
        print("[预览模式] 不写入标签文件")

    counts = Counter()
    for i, img_path in enumerate(files, 1):
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        results = model(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]

        new_lines = []
        for box in result.boxes:
            model_cls = int(box.cls[0])
            if model_cls not in MODEL_TO_V13:
                continue
            v13_cls = MODEL_TO_V13[model_cls]
            cx, cy, bw, bh = box.xywhn[0].tolist()
            new_lines.append(f"{v13_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts[v13_cls] += 1

        if write_labels and new_lines:
            append_to_label(lbl_path, new_lines)

        # 可视化
        if i <= 30 or args.limit and i <= args.limit:  # 前 30 张或 limit 内的都画
            out_path = PREVIEW_DIR / img_path.name
            visualize(img_path, lbl_path, out_path)

        if i % 50 == 0:
            print(f"  [{i}/{len(files)}] 当前: Platform={counts[2]}, Rope={counts[3]}")

    print(f"\n--- 标注统计 ---")
    print(f"  Platform (class 2): {counts[2]} 个新框")
    print(f"  Rope (class 3):     {counts[3]} 个新框")
    print(f"  预览图保存到: {PREVIEW_DIR}/")
    if write_labels:
        print(f"  标签已追加到 {LBL_DIR}/")
    else:
        print(f"  (标签未修改)")


if __name__ == "__main__":
    main()
