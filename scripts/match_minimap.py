"""
Minimap 模板匹配: 把游戏画面右上角 minimap 区域与 meowdb minimap 库比对, 找出当前地图 ID.

原理:
  1. 截游戏画面右上角 (假设 minimap 在 (1140, 20, 1366, 170))
  2. 缩放到 138x116 (匹配 meowdb minimap 尺寸)
  3. 跟 data/map_db/minimaps/*.png 做 template matching
  4. 最佳匹配 = 当前 map ID

注意: minimap 上有玩家/NPC/怪物实时标记, 跟静态库对比会有偏差.
解决: 用 cv2.TM_CCOEFF_NORMED, 并允许多个候选输出 (top-5).

用法:
  python scripts/match_minimap.py <image_path>
  python scripts/match_minimap.py data/auto_dataset/images/auto_1773638934577.jpg
"""
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINIMAP_DIR = PROJECT_ROOT / "data" / "map_db" / "minimaps"

# 游戏画面里 minimap 的位置 (相对 1366x768 画布)
# Artale (台湾私服) UI: minimap 在左上角 (X: 5-260, Y: 80-260)
DEFAULT_REGION = (5, 80, 260, 260)  # x1, y1, x2, y2

# meowdb minimap 尺寸
TEMPLATE_SIZE = (138, 116)


def extract_minimap(frame, region):
    """从游戏画面里裁剪 minimap 区域, 缩放到 template 尺寸"""
    x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)


def load_templates():
    """加载所有 minimap 模板, 返回 {map_id: template_img}"""
    templates = {}
    for p in MINIMAP_DIR.glob("*.png"):
        mid = p.stem
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            templates[mid] = img
    return templates


def match(frame, region=None, top_k=5):
    """在 frame 上找 minimap, 返回 [(map_id, score), ...]"""
    if region is None:
        region = DEFAULT_REGION

    mm = extract_minimap(frame, region)
    if mm is None:
        return []

    templates = load_templates()
    results = []
    for mid, tmpl in templates.items():
        # 缩放到相同尺寸
        tmpl_resized = cv2.resize(tmpl, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        # 转灰度
        g1 = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2GRAY)
        # 归一化模板匹配
        score = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)[0][0]
        results.append((mid, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    p = argparse.ArgumentParser(description="Minimap 模板匹配 -> map ID")
    p.add_argument("image", type=Path, help="游戏画面路径")
    p.add_argument("--region", type=int, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="minimap 区域 (默认 1140 20 1366 180)")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"未找到 {args.image}")

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"无法读取 {args.image}")
    print(f"输入: {args.image} ({frame.shape[1]}x{frame.shape[0]})")

    region = tuple(args.region) if args.region else None
    top_matches = match(frame, region, args.top_k)

    print(f"\nTop {len(top_matches)} 匹配:")
    for rank, (mid, score) in enumerate(top_matches, 1):
        bar = "█" * int(score * 30)
        print(f"  #{rank}  {mid}  score={score:.3f}  {bar}")


if __name__ == "__main__":
    main()