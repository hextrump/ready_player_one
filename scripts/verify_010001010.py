"""
在真游戏帧上验证 010001010 模型的检测效果.

用法:
  python scripts/verify_010001010.py <image_or_dir>
"""
import sys
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 模型可能在 OmniParser 输出目录, 先找
CANDIDATES = [
    PROJECT_ROOT / "models" / "super_brain_010001010.pt",   # 本机已训练好的模型 (models/)
    PROJECT_ROOT / "runs" / "detect" / "super_brain_010001010" / "weights" / "best.pt",
    Path("C:/Users/heyas/Documents/code/OmniParser/runs/detect/runs/detect/super_brain_010001010-6/weights/best.pt"),
]

CLASS_NAMES = {
    0: "Player",
    1: "BlueSnail", 2: "Shroom", 3: "RedSnail", 4: "Stump",
    5: "Slime", 6: "OrangeMushroom", 7: "GreenMushroom",
    8: "Platform", 9: "Rope",
}
CLASS_COLORS = {
    0: (0, 200, 255),      # Player - 蓝
    1: (255, 100, 100),    # BlueSnail - 浅红
    2: (255, 150, 100),    # Shroom - 橙
    3: (100, 100, 255),    # RedSnail - 蓝
    4: (150, 100, 50),     # Stump - 棕
    5: (100, 255, 100),    # Slime - 绿
    6: (255, 200, 100),    # OrangeMushroom - 橙黄
    7: (100, 200, 100),    # GreenMushroom - 暗绿
    8: (200, 200, 200),    # Platform - 灰
    9: (100, 255, 255),    # Rope - 青
}


def find_model():
    for p in CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(f"未找到训练好的模型, 检查: {[str(p) for p in CANDIDATES]}")


def detect_one(model, image_path, conf=0.3):
    img = Image.open(image_path).convert("RGB")
    results = model(img, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf_v = float(b.conf[0])
            boxes.append((cls, x1, y1, x2, y2, conf_v))
    return img, boxes


def draw_boxes(img, boxes):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    for cls, x1, y1, x2, y2, conf in boxes:
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
        draw.text((x1 + 2, max(y1 - 12, 0)), label, fill=color)
    return vis


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/verify_010001010.py <image_or_dir> [--conf 0.3]")
        sys.exit(1)

    target = Path(sys.argv[1])
    conf = 0.3
    if "--conf" in sys.argv:
        conf = float(sys.argv[sys.argv.index("--conf") + 1])

    model_path = find_model()
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    out_dir = PROJECT_ROOT / "data" / "synthetic_010001010" / "verify_results"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 收集图片
    if target.is_dir():
        images = sorted(list(target.glob("*.jpg")) + list(target.glob("*.png")))
    elif target.is_file():
        images = [target]
    else:
        raise SystemExit(f"未找到 {target}")

    # 限制数量
    images = images[:12]
    print(f"推理 {len(images)} 张, conf={conf}")

    all_counts = Counter()
    for i, img_path in enumerate(images):
        img, boxes = detect_one(model, img_path, conf)
        cnt = Counter(b[0] for b in boxes)
        all_counts.update(cnt)
        vis = draw_boxes(img, boxes)
        out_path = out_dir / f"verify_{i:02d}_{img_path.name}"
        vis.save(out_path, "JPEG", quality=90)
        print(f"  [{i+1}] {img_path.name}: {dict(cnt)}")

    print(f"\n--- 总计 ---")
    for cid in sorted(CLASS_NAMES):
        print(f"  {cid} ({CLASS_NAMES[cid]}): {all_counts[cid]}")

    # 拼图预览
    thumbs = []
    for p in sorted(out_dir.glob("*.jpg")):
        img = Image.open(p)
        img.thumbnail((500, 500))
        thumbs.append(img)
    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        w, h = thumbs[0].size
        canvas = Image.new("RGB", (cols * w + 20, rows * h + 20), (20, 20, 20))
        for i, t in enumerate(thumbs):
            canvas.paste(t, ((i % cols) * w + 10, (i // cols) * h + 10))
        grid_path = out_dir / "_grid.jpg"
        canvas.save(grid_path, "JPEG", quality=88)
        print(f"\n拼图预览: {grid_path}")


if __name__ == "__main__":
    main()