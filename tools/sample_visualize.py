"""
可视化新数据集样本, 验证 resize 后效果.

输出:
  sample_preview/grid.jpg         - 12 张新图 (1366x768) 拼图 + bbox
  sample_preview/compare.jpg      - 3 对 原图 vs 新图 大尺寸对比 + 标注灰边
  sample_preview/showcase.jpg     - 单张高亮 letterbox 灰边
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "images"
LBL_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "labels"
ORIG_DIR = IMG_DIR.parent / "images_orig"
OUT_DIR = IMG_DIR.parent / "sample_preview"

CLASS_COLORS = {0: (0, 102, 204), 1: (204, 0, 0), 2: (0, 170, 0), 3: (0, 170, 170)}
CLASS_NAMES = {0: "Player", 1: "Monster", 2: "Platform", 3: "Rope"}


def draw_boxes(img, label_path):
    if not label_path.exists():
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls not in CLASS_COLORS:
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = CLASS_COLORS[cls]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(y1 - 14, 0)), CLASS_NAMES[cls], fill=color)
    return img


def find_letterbox_bounds(img, target_w, target_h, tol=5):
    """检测 letterbox 灰边位置 (RGB ~114)"""
    w, h = img.size
    px = img.load()
    bounds = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    def is_gray(p):
        return abs(p[0] - 114) < tol and abs(p[1] - 114) < tol and abs(p[2] - 114) < tol

    # top
    for y in range(h):
        if not all(is_gray(px[x, y]) for x in range(0, w, 20)):
            bounds["top"] = y
            break
    # bottom
    for y in range(h - 1, -1, -1):
        if not all(is_gray(px[x, y]) for x in range(0, w, 20)):
            bounds["bottom"] = h - 1 - y
            break
    # left
    for x in range(w):
        if not all(is_gray(px[x, y]) for y in range(0, h, 20)):
            bounds["left"] = x
            break
    # right
    for x in range(w - 1, -1, -1):
        if not all(is_gray(px[x, y]) for y in range(0, h, 20)):
            bounds["right"] = w - 1 - x
            break
    return bounds


def make_grid(samples, out_path, cols=4):
    if not samples:
        return
    rows = (len(samples) + cols - 1) // cols
    thumb_w = 500
    thumb_h = int(thumb_w * 768 / 1366)
    label_h = 22

    canvas = Image.new("RGB",
                       (cols * thumb_w + (cols + 1) * 4,
                        rows * (thumb_h + label_h) + (rows + 1) * 4),
                       (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for i, (img_path, lbl_path) in enumerate(samples):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img = draw_boxes(img, lbl_path)
        img.thumbnail((thumb_w, thumb_h), Image.BILINEAR)
        r, c = i // cols, i % cols
        x = 4 + c * (thumb_w + 4)
        y = 4 + r * (thumb_h + label_h + 4)
        canvas.paste(img, (x, y))
        n = sum(1 for _ in open(lbl_path)) if lbl_path.exists() else 0
        caption = f"{img_path.name}  ({n} boxes)"
        draw.text((x + 4, y + thumb_h + 4), caption, fill=(200, 200, 200))

    canvas.save(out_path, "JPEG", quality=90)
    print(f"  → {out_path}")


def make_compare(samples, out_path):
    """原图 vs 新图 大尺寸对比, 高亮 letterbox 灰边"""
    if not samples:
        return
    rows = len(samples)
    # 用大尺寸显示
    thumb_w = 900
    thumb_h = int(thumb_w * 768 / 1366)
    label_h = 30

    canvas_w = thumb_w * 2 + 80
    canvas_h = rows * (thumb_h + label_h + 60) + 40

    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, name in enumerate(samples):
        orig_path = ORIG_DIR / name
        new_path = IMG_DIR / name
        if not orig_path.exists() or not new_path.exists():
            print(f"  跳过 {name} (备份缺失)")
            continue

        orig = Image.open(orig_path).convert("RGB")
        new = Image.open(new_path).convert("RGB")
        orig_w, orig_h = orig.size

        # 检测 letterbox 灰边
        bounds = find_letterbox_bounds(new, 1366, 768)

        # 显示尺寸
        orig_show = orig.copy()
        orig_show.thumbnail((thumb_w, thumb_h), Image.BILINEAR)
        new_show = new.copy()
        new_show.thumbnail((thumb_w, thumb_h), Image.BILINEAR)

        y = 20 + i * (thumb_h + label_h + 60)
        canvas.paste(orig_show, (20, y))
        canvas.paste(new_show, (40 + thumb_w, y))

        # 标注
        draw.text((20, y + thumb_h + 6),
                  f"原图  {orig_w}x{orig_h}  (aspect {orig_w/orig_h:.3f})",
                  fill=(255, 180, 180))
        gray_desc = []
        if bounds["top"] > 0 or bounds["bottom"] > 0:
            gray_desc.append(f"上下灰边 {max(bounds['top'], bounds['bottom'])}px")
        if bounds["left"] > 0 or bounds["right"] > 0:
            gray_desc.append(f"左右灰边 {max(bounds['left'], bounds['right'])}px")
        if not gray_desc:
            gray_desc.append("无 padding (完美 16:9)")
        draw.text((40 + thumb_w, y + thumb_h + 6),
                  f"新图  1366x768 letterbox  |  {' / '.join(gray_desc)}",
                  fill=(180, 255, 180))

        # 在新图上画灰边高亮 (橙色框出 letterbox 区域)
        scale = thumb_w / 1366
        for side in ("top", "bottom", "left", "right"):
            b = bounds[side]
            if b <= 0:
                continue
            if side == "top":
                box = (40 + thumb_w, y, 40 + thumb_w + int(1366 * scale),
                       y + int(b * (thumb_h / 768)))
            elif side == "bottom":
                box = (40 + thumb_w, y + thumb_h - int(b * (thumb_h / 768)),
                       40 + thumb_w + int(1366 * scale), y + thumb_h)
            elif side == "left":
                box = (40 + thumb_w, y,
                       40 + thumb_w + int(b * scale), y + thumb_h)
            else:  # right
                box = (40 + thumb_w + int(1366 * scale) - int(b * scale), y,
                       40 + thumb_w + int(1366 * scale), y + thumb_h)
            draw.rectangle(box, outline=(255, 140, 0), width=2)

    canvas.save(out_path, "JPEG", quality=92)
    print(f"  → {out_path}")


def make_showcase(name):
    """单张高亮: 原图 + 新图 + 灰边区域标注"""
    orig_path = ORIG_DIR / name
    new_path = IMG_DIR / name
    if not orig_path.exists() or not new_path.exists():
        print(f"  {name} 备份缺失, 跳过 showcase")
        return

    orig = Image.open(orig_path).convert("RGB")
    new = Image.open(new_path).convert("RGB")
    orig_w, orig_h = orig.size
    bounds = find_letterbox_bounds(new, 1366, 768)

    # 新图按真实尺寸显示
    label_h = 40
    canvas_w = max(orig_w, 1366) + 60
    canvas_h = orig_h + 1366 + label_h * 3 + 60
    canvas = Image.new("RGB", (canvas_w, canvas_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    # 原图
    canvas.paste(orig, (30, 30))
    draw.rectangle((30, 30, 30 + orig_w, 30 + orig_h), outline=(255, 100, 100), width=2)
    draw.text((30, 30 + orig_h + 6),
              f"原图 {orig_w}x{orig_h} (aspect {orig_w/orig_h:.3f})",
              fill=(255, 180, 180))

    # 新图 (1366x768)
    y_new = 30 + orig_h + label_h * 2
    canvas.paste(new, (30, y_new))
    draw.rectangle((30, y_new, 30 + 1366, y_new + 768),
                   outline=(100, 255, 100), width=2)

    # 高亮 letterbox 灰边 (橙色)
    for side in ("top", "bottom", "left", "right"):
        b = bounds[side]
        if b <= 0:
            continue
        if side == "top":
            box = (30, y_new, 30 + 1366, y_new + b)
        elif side == "bottom":
            box = (30, y_new + 768 - b, 30 + 1366, y_new + 768)
        elif side == "left":
            box = (30, y_new, 30 + b, y_new + 768)
        else:
            box = (30 + 1366 - b, y_new, 30 + 1366, y_new + 768)
        draw.rectangle(box, outline=(255, 140, 0), width=3)
        # 标注
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        draw.text((cx - 30, cy - 8), f"{b}px", fill=(255, 200, 0))

    caption = f"→ 新图 1366x768 letterbox"
    gray_total = max(bounds["top"], bounds["bottom"]) + max(bounds["left"], bounds["right"])
    if gray_total == 0:
        caption += " (无 padding, 完美适配 16:9)"
    else:
        caption += f"  灰边: 上{bounds['top']}px 下{bounds['bottom']}px 左{bounds['left']}px 右{bounds['right']}px"
    draw.text((30, y_new + 768 + 6), caption, fill=(180, 255, 180))

    out_path = OUT_DIR / "showcase.jpg"
    canvas.save(out_path, "JPEG", quality=92)
    print(f"  → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] 当前数据拼图 (含 bbox) ...")
    all_files = sorted(IMG_DIR.glob("*.jpg"))
    labeled = [p for p in all_files
               if (LBL_DIR / (p.stem + ".txt")).exists()
               and (LBL_DIR / (p.stem + ".txt")).stat().st_size > 0]
    samples = random.sample(labeled, min(12, len(labeled)))
    make_grid([(p, LBL_DIR / (p.stem + ".txt")) for p in samples],
              OUT_DIR / "grid.jpg")

    if not ORIG_DIR.is_dir():
        print(f"\n未找到 {ORIG_DIR}, 跳过对比图")
        return

    print("\n[2/3] 原图 vs 新图 对比 (高亮灰边) ...")
    orig_files = sorted(ORIG_DIR.glob("*.jpg"))
    by_size = {}
    for f in orig_files:
        with Image.open(f) as img:
            by_size.setdefault(img.size, []).append(f.name)
    # 3 种源分辨率各选 1 个
    picks = []
    for size in sorted(by_size.keys()):
        picks.append(random.choice(by_size[size]))
    make_compare(picks, OUT_DIR / "compare.jpg")

    print("\n[3/3] 单张 showcase ...")
    # 选一个 2060x1115 (旧) 的来展示明显灰边
    target_size = (2060, 1115)
    if target_size in by_size:
        make_showcase(random.choice(by_size[target_size]))
    else:
        # fallback: 任选一张
        make_showcase(random.choice(orig_files).name)

    print(f"\n[OK] 预览图保存在 {OUT_DIR}/")


if __name__ == "__main__":
    random.seed(42)
    main()
