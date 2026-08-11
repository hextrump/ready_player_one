"""
B 方案 v2: 名牌成对检测 (白名 + 蓝徽章) 排除其他玩家干扰
===========================================================

基于实测游戏帧:
- "新手冒险家勋章" 蓝色范围: HSV H 88-118, S 40-255, V 150-230
- "叮咚大狗叫" 白底名牌: HSV H 0-40, S 0-100, V 200-255 (亮色, 低饱和)
- 两个名牌成对出现: 名字在上 (y), 徽章在下 (y+15-25)
- 玩家位置 = 徽章中心 + (0, +30)

为什么要成对检测:
- 多人场景 (其他玩家也有同名徽章) 时单检测蓝色徽章会选错
- 只有同时匹配 "叮咚大狗叫" 名字 + "新手冒险家勋章" 徽章才能确认是自己的

检测流程:
1. HSV mask 提取蓝色徽章候选
2. 对每个蓝徽章, 在其上方 ~20px 范围找对应的白色名牌
3. 成对匹配的才计入候选
4. 选最下方 (最像在地面上) 的作为我的玩家
5. 返回玩家位置

用法: 同上
"""
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==== 蓝色徽章 "新手冒险家勋章" ====
BADGE_LOWER = np.array([88, 40, 150])
BADGE_UPPER = np.array([118, 255, 230])

# ==== 白色名牌 "叮咚大狗叫" ====
# 实测: HSV (0-40, 0-100, 200-255) 是亮白/亮灰
NAME_LOWER = np.array([0, 0, 200])
NAME_UPPER = np.array([40, 100, 255])

# ==== bbox 过滤 ====
ASPECT_MIN, ASPECT_MAX = 3.0, 12.0
HEIGHT_MIN, HEIGHT_MAX = 15, 60
Y_MIN_BOX, Y_MAX_BOX = 200, 720

# ==== 配对参数 ====
# 白名牌在蓝徽章上方, 间距 5-30px
NAME_GAP_MIN, NAME_GAP_MAX = 5, 30
# 白名牌和蓝徽章中心 x 偏差不超过 30px (横向对齐)
X_OFFSET_MAX = 30
# 白名牌和蓝徽章宽度差不超过 50px (成对应当差不多宽)
WIDTH_TOLERANCE = 50


def _find_candidates(mask: np.ndarray) -> list:
    """mask 上找连通块 + bbox 过滤,返回 [(area, x, y, w, h), ...]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    ret, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, ret):
        x, y, w, h, area = stats[i]
        if area < 500:
            continue
        ar = w / max(h, 1)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX and HEIGHT_MIN <= h <= HEIGHT_MAX):
            continue
        if not (Y_MIN_BOX <= y <= Y_MAX_BOX):
            continue
        cands.append((area, x, y, w, h))
    return cands


def find_nametag_bbox(frame_bgr: np.ndarray) -> tuple | None:
    """检测玩家名牌位置. 策略:
    1. 找所有白名牌 + 蓝徽章候选
    2. 配对 (白在上, 蓝在下) -> 配对成功的视为一个完整名牌
    3. 没配上的单名牌也保留
    4. 选 y 最大的 (最靠下, 相机跟着主角走, 主角一般在地面层)
    5. 返回名牌底部 (徽章或虚拟徽章) bbox
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    badge_mask = cv2.inRange(hsv, BADGE_LOWER, BADGE_UPPER)
    name_mask = cv2.inRange(hsv, NAME_LOWER, NAME_UPPER)

    badges = _find_candidates(badge_mask)
    names = _find_candidates(name_mask)

    if not badges and not names:
        return None

    # 收集所有候选: 每个候选是一个 (y_top, bbox) 用于比较高度
    candidates = []  # [(bottom_y, x, y, w, h, source), ...]

    # 1) 已配对的 (白+蓝), 用徽章的 y 作为底部
    for b_area, bx, by, bw, bh in badges:
        bcx = bx + bw // 2
        paired = False
        for n_area, nx, ny, nw, nh in names:
            if ny >= by: continue
            gap = by - (ny + nh)
            if not (NAME_GAP_MIN <= gap <= NAME_GAP_MAX): continue
            ncx = nx + nw // 2
            if abs(bcx - ncx) > X_OFFSET_MAX: continue
            if abs(bw - nw) > WIDTH_TOLERANCE: continue
            paired = True
            break
        candidates.append((by + bh, bx, by, bw, bh, "badge"))

    # 2) 未配对的白名牌 (徽章被遮) -> 虚拟徽章框
    used_names = set()
    for b_area, bx, by, bw, bh in badges:
        bcx = bx + bw // 2
        for i, (n_area, nx, ny, nw, nh) in enumerate(names):
            if i in used_names: continue
            if ny >= by: continue
            gap = by - (ny + nh)
            if not (NAME_GAP_MIN <= gap <= NAME_GAP_MAX): continue
            ncx = nx + nw // 2
            if abs(bcx - ncx) > X_OFFSET_MAX: continue
            if abs(bw - nw) > WIDTH_TOLERANCE: continue
            used_names.add(i)

    for i, (n_area, nx, ny, nw, nh) in enumerate(names):
        if i in used_names: continue
        # 没配对的单白名牌, 虚拟一个徽章位置 (名+25~30)
        virtual_y = ny + nh + NAME_GAP_MIN
        virtual_bottom = virtual_y + 20
        candidates.append((virtual_bottom, nx, virtual_y, nw, 20, "virtual"))

    if not candidates:
        return None

    # 取 y 最大 (最靠下) 的, 那就是主角
    candidates.sort(key=lambda t: -t[0])
    _, x, y, w, h, source = candidates[0]
    return (x, y, w, h)


def nametag_to_player(nametag_bbox: tuple, head_offset_px: int = 30) -> tuple:
    """徽章中心 + 向下偏移 = 玩家中心"""
    x, y, w, h = nametag_bbox
    return (x + w // 2, y + h + head_offset_px)


def visualize(img_path: Path, bbox: tuple | None, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if bbox:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 255), width=3)
        px, py = nametag_to_player(bbox)
        draw.line([(px - 8, py), (px + 8, py)], fill=(255, 50, 50), width=2)
        draw.line([(px, py - 8), (px, py + 8)], fill=(255, 50, 50), width=2)
        draw.text((x, max(y - 15, 0)), f"Nametag ({w}x{h})", fill=(0, 200, 255))
        draw.text((px + 10, py - 8), f"Player ({px},{py})", fill=(255, 50, 50))
    else:
        draw.text((20, 20), "NOT FOUND", fill=(255, 50, 50))
    img.save(out_path, "JPEG", quality=88)


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/find_player_nametag.py <image_or_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        images = sorted(list(target.glob("*.jpg")) + list(target.glob("*.png")))
    elif target.is_file():
        images = [target]
    else:
        raise SystemExit(f"未找到 {target}")
    if not images:
        raise SystemExit(f"目录无图: {target}")

    out_dir = PROJECT_ROOT / "data" / "auto_dataset" / "nametag_verify"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"扫描 {len(images)} 张, 输出到 {out_dir}/")
    hits = 0
    for i, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [{i}/{len(images)}] {img_path.name}: 读取失败")
            continue
        bbox = find_nametag_bbox(frame)
        out_path = out_dir / f"{i:02d}_{img_path.name}"
        visualize(img_path, bbox, out_path)
        if bbox:
            x, y, w, h = bbox
            px, py = nametag_to_player(bbox)
            hits += 1
            print(f"  [{i}/{len(images)}] {img_path.name}: 名牌 @ ({x},{y}) {w}x{h} -> 玩家 @ ({px},{py})")
        else:
            print(f"  [{i}/{len(images)}] {img_path.name}: 未找到")

    print(f"\n命中率: {hits}/{len(images)} ({100*hits/len(images):.0f}%)")
    print(f"逐图结果: {out_dir}/")

    thumbs = []
    for p in sorted(out_dir.glob("*.jpg")):
        im = Image.open(p)
        im.thumbnail((500, 500))
        thumbs.append(im)
    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        w, h = thumbs[0].size
        canvas = Image.new("RGB", (cols * w + 20, rows * h + 20), (20, 20, 20))
        for j, t in enumerate(thumbs):
            canvas.paste(t, ((j % cols) * w + 10, (j // cols) * h + 10))
        grid = out_dir / "_grid.jpg"
        canvas.save(grid, "JPEG", quality=85)
        print(f"拼图预览: {grid}")


if __name__ == "__main__":
    main()