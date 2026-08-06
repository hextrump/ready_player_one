"""
合成数据生成器: 专攻地图 010001010 (Henesys Hunting Ground I)

输入:
  data/map_db/010001010_full.png     (2218x1870 全图)
  data/map_db/monsters/sprites/*.png (7 种怪物)

输出:
  data/synthetic_010001010/
    images/{idx}.jpg
    labels/{idx}.txt       (YOLO 格式)
    dataset.yaml

类别 (10 类):
  0=Player, 1=BlueSnail, 2=Shroom, 3=RedSnail, 4=Stump,
  5=Slime, 6=OrangeMushroom, 7=GreenMushroom, 8=Platform, 9=Rope

用法:
  python scripts/gen_yolo_010001010.py --count 200
"""
import argparse
import random
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_MAP = PROJECT_ROOT / "data" / "map_db" / "010001010_full.png"
SPRITES_DIR = PROJECT_ROOT / "data" / "map_db" / "monsters" / "sprites"
PLAYER_SRC = PROJECT_ROOT / "data" / "player" / "02.png"  # 用户角色 sprite 源位置

OUT = PROJECT_ROOT / "data" / "synthetic_010001010"
OUT_IMG = OUT / "images"
OUT_LBL = OUT / "labels"

CLASS_NAMES = {
    0: "Player",
    1: "BlueSnail", 2: "Shroom", 3: "RedSnail", 4: "Stump",
    5: "Slime", 6: "OrangeMushroom", 7: "GreenMushroom",
    8: "Platform", 9: "Rope",
}

# 怪物 ID -> 类别 ID (sprite 文件名: {mob_id}_{Name}.png)
# 加上 player sprite 文件: player_*.png  -> class 0
MONSTER_TO_CLASS = {
    "player": 0,   # 玩家 (从 auto_dataset 切出的多角度图)
    "3": 1,        # Blue Snail
    "4": 2,        # Shroom
    "5": 3,        # Red Snail
    "6": 4,        # Stump
    "7": 5,        # Slime
    "9": 6,        # Orange Mushroom
    "13": 7,       # Green Mushroom
}

# Platform = 8, Rope = 9 (在 generate_one 里直接赋值)

# 平台 y 坐标 (基于 2218x1870 全图, 从上到下)
# 从网格图看出来的水平平台大致高度
PLATFORM_Y_LEVELS = [350, 500, 680, 850, 1100, 1350]

# 平台水平范围 (避免平台之间 y 区间落空)
PLATFORM_X_RANGE = (100, 2100)

# 垂直结构 (rope + ladder): (x1, y1, x2, y2) - 线段
VERTICAL_STRUCTURES = [
    # 左梯子 (粗的)
    (170, 820, 170, 1110, "Rope"),
    # 右梯子
    (2050, 820, 2050, 1110, "Rope"),
    # 中央大绳 (从底到顶)
    (1110, 300, 1110, 1110, "Rope"),
    # 几个小 rope/梯子
    (430, 700, 430, 850, "Rope"),
    (1700, 700, 1700, 850, "Rope"),
    (270, 560, 270, 700, "Rope"),
    (1810, 480, 1810, 700, "Rope"),
]

# 平台列表 (用于生成 Platform 标签): (x1, y1, x2, y2)
PLATFORMS = [
    # 顶层
    (100, 320, 350, 380),     # 左上小平台
    (1900, 320, 2150, 380),   # 右上小平台
    # 中上平台 (含树和向日葵)
    (350, 470, 1850, 520),
    # 中平台
    (350, 650, 1900, 700),
    # 中下平台
    (300, 830, 1950, 880),
    # 底部平台 (草堆区)
    (100, 1090, 2150, 1130),
    # 最底层 (ground)
    (0, 1340, 2218, 1390),
    # 左侧漂浮平台
    (200, 540, 350, 580),
    (200, 690, 350, 730),
]


def parse_args():
    p = argparse.ArgumentParser(description="合成 010001010 训练数据")
    p.add_argument("--count", type=int, default=200, help="生成多少张图")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--monsters-min", type=int, default=3, help="每张图最少怪物数")
    p.add_argument("--monsters-max", type=int, default=12, help="每张图最多怪物数")
    return p.parse_args()


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)


def load_sprites():
    """加载所有怪物 sprite, 返回 {class_id: [(PIL, w, h), ...] 同 sprite 多帧备用}"""
    cache = {cid: [] for cid in MONSTER_TO_CLASS.values()}
    # 1. 优先用 data/player/02.png (用户提供, 干净透明背景)
    if PLAYER_SRC.exists():
        cache[0].append(Image.open(PLAYER_SRC).convert("RGBA"))
    # 2. 再扫 SPRITES_DIR 兼容老路径
    for sp in sorted(SPRITES_DIR.glob("*.png")):
        # 文件名: {mob_id}_{Name}.png  或  player_*.png
        name = sp.stem
        # Player sprite: 文件名以 player_ 开头
        if name.startswith("player"):
            # 已通过 PLAYER_SRC 加载, 跳过
            continue
        try:
            mob_id = name.split("_")[0]
            cid = MONSTER_TO_CLASS[mob_id]
        except (KeyError, IndexError):
            continue
        img = Image.open(sp).convert("RGBA")
        cache[cid].append(img)
    return cache


def place_monster(canvas, sprite, x, y):
    """在 canvas 的 (x, y) 位置贴 sprite, x/y 是 sprite 左下角"""
    # sprite 朝向: 脚在 (x, y)
    px = x - sprite.width // 2
    py = y - sprite.height
    # 边界检查
    if px < 0 or py < 0 or px + sprite.width > canvas.width or py + sprite.height > canvas.height:
        return None
    canvas.alpha_composite(Image.new("RGBA", canvas.size, (0, 0, 0, 0)), )
    # alpha_composite
    tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp.paste(sprite, (px, py), sprite)
    canvas.alpha_composite(tmp)


def generate_one(idx, sprites_by_class, rng):
    """生成一张合成图 + labels"""
    # 1. 随机裁剪 1366x768 窗口
    full = Image.open(FULL_MAP).convert("RGB")
    fw, fh = full.size
    crop_w, crop_h = 1366, 768
    if fw < crop_w or fh < crop_h:
        raise SystemExit(f"全图太小 {fw}x{fh}")
    max_x = fw - crop_w
    max_y = fh - crop_h
    x0 = rng.randint(0, max_x)
    y0 = rng.randint(0, max_y)
    canvas = full.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    # 转 RGBA 准备贴 sprite
    canvas = canvas.convert("RGBA")

    boxes = []  # [(cls, cx, cy, w, h), ...] 归一化坐标

    # 2. 生成 Platform 标签 (在裁剪范围内)
    for px1, py1, px2, py2 in PLATFORMS:
        # 转成裁剪后的坐标
        cx1, cy1 = px1 - x0, py1 - y0
        cx2, cy2 = px2 - x0, py2 - y0
        # 跳过不在裁剪内的平台
        if cx2 < 0 or cx1 > crop_w or cy2 < 0 or cy1 > crop_h:
            continue
        # clip 到裁剪范围
        cx1 = max(0, cx1); cy1 = max(0, cy1)
        cx2 = min(crop_w, cx2); cy2 = min(crop_h, cy2)
        if cx2 - cx1 < 30 or cy2 - cy1 < 10:
            continue
        bw = (cx2 - cx1) / crop_w
        bh = (cy2 - cy1) / crop_h
        bcx = (cx1 + cx2) / 2 / crop_w
        bcy = (cy1 + cy2) / 2 / crop_h
        boxes.append((8, bcx, bcy, bw, bh))  # class 8 = Platform

    # 3. 生成 Rope 标签
    for rx1, ry1, rx2, ry2, _ in VERTICAL_STRUCTURES:
        cx1, cy1 = rx1 - x0, ry1 - y0
        cx2, cy2 = rx2 - x0, ry2 - y0
        if cx2 < 0 or cx1 > crop_w or cy2 < 0 or cy1 > crop_h:
            continue
        # rope 很窄, 4-6 px 宽
        rope_w_px = 6
        cx1 = max(0, cx1 - rope_w_px // 2)
        cx2 = min(crop_w, cx2 + rope_w_px // 2)
        cy1 = max(0, cy1); cy2 = min(crop_h, cy2)
        bw = (cx2 - cx1) / crop_w
        bh = (cy2 - cy1) / crop_h
        if bw <= 0 or bh <= 0:
            continue
        bcx = (cx1 + cx2) / 2 / crop_w
        bcy = (cy1 + cy2) / 2 / crop_h
        boxes.append((9, bcx, bcy, bw, bh))  # class 9 = Rope

    # 4. 随机放置怪物
    # Player 数量: 0-1 个 (游戏中通常只有 1 个自己)
    n_monsters = rng.randint(args.monsters_min, args.monsters_max)
    placed = []  # (x, y, half_w, half_h)
    max_attempts = n_monsters * 5
    monster_classes = [c for c in sprites_by_class.keys() if c != 0]  # 除 Player 外的怪
    for _ in range(max_attempts):
        if len(placed) >= n_monsters:
            break
        # 随机选平台 y
        py = rng.choice(PLATFORM_Y_LEVELS) + rng.randint(-10, 10)
        # 随机选 x (裁剪范围内)
        px = rng.randint(50, crop_w - 50)
        # 转到裁剪坐标 (y 减去 y0 偏移)
        abs_y = py - y0
        if abs_y < 50 or abs_y > crop_h - 20:
            continue
        # 随机选怪物 (不放 Player, Player 单独处理)
        if not monster_classes:
            break
        cls = rng.choice(monster_classes)
        if not sprites_by_class[cls]:
            continue
        sprite = rng.choice(sprites_by_class[cls])
        # 随机缩放
        scale = rng.uniform(0.9, 1.3)
        sw = int(sprite.width * scale)
        sh = int(sprite.height * scale)
        # 边界检查
        sx = px - sw // 2
        sy = abs_y - sh
        if sx < 0 or sy < 0 or sx + sw > crop_w or sy + sh > crop_h:
            continue
        # 简单碰撞检查 (中心距离 > min_dist)
        too_close = False
        for ox, oy, ow, oh in placed:
            if abs(sx + sw // 2 - ox) < (sw + ow) // 2 and abs(sy + sh - oy) < (sh + oh):
                too_close = True
                break
        if too_close:
            continue

        # 贴 sprite
        sprite_resized = sprite.resize((sw, sh), Image.BILINEAR)
        canvas.paste(sprite_resized, (sx, sy), sprite_resized)

        # YOLO 标签
        bw = sw / crop_w
        bh = sh / crop_h
        bcx = (sx + sw / 2) / crop_w
        bcy = (sy + sh / 2) / crop_h
        boxes.append((cls, bcx, bcy, bw, bh))

        placed.append((sx + sw // 2, sy + sh, sw, sh))

    # 5. 放 1 个 Player (类似怪的方式, 但保证不超过 1 个)
    if 0 in sprites_by_class and sprites_by_class[0]:
        py = rng.choice(PLATFORM_Y_LEVELS) + rng.randint(-10, 10)
        px = rng.randint(150, crop_w - 150)
        abs_y = py - y0
        if 50 < abs_y < crop_h - 20:
            sprite = rng.choice(sprites_by_class[0])
            scale = rng.uniform(1.1, 1.4)  # Player 稍大
            sw = int(sprite.width * scale)
            sh = int(sprite.height * scale)
            sx = px - sw // 2
            sy = abs_y - sh
            if 0 <= sx and sx + sw <= crop_w and 0 <= sy and sy + sh <= crop_h:
                sprite_resized = sprite.resize((sw, sh), Image.BILINEAR)
                canvas.paste(sprite_resized, (sx, sy), sprite_resized)
                bw = sw / crop_w
                bh = sh / crop_h
                bcx = (sx + sw / 2) / crop_w
                bcy = (sy + sh / 2) / crop_h
                boxes.append((0, bcx, bcy, bw, bh))

    # 5. 数据增强 (轻微)
    if rng.random() < 0.5:
        # 亮度
        enhancer = ImageEnhance.Brightness(canvas)
        canvas = enhancer.enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:
        # 对比度
        enhancer = ImageEnhance.Contrast(canvas)
        canvas = enhancer.enhance(rng.uniform(0.85, 1.15))

    # 转回 RGB 保存
    final = canvas.convert("RGB")

    # 保存
    img_path = OUT_IMG / f"syn_{idx:04d}.jpg"
    lbl_path = OUT_LBL / f"syn_{idx:04d}.txt"
    final.save(img_path, "JPEG", quality=90)

    with open(lbl_path, "w") as f:
        for cls, cx, cy, bw, bh in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return Counter(b[0] for b in boxes)


def write_yaml():
    rel = OUT.relative_to(PROJECT_ROOT).as_posix()
    content = f"path: {rel}\ntrain: images\nval: images\n\nnames:\n"
    for cid, name in CLASS_NAMES.items():
        content += f"  {cid}: {name}\n"
    (OUT / "dataset.yaml").write_text(content, encoding="utf-8")


def main():
    global args
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not FULL_MAP.exists():
        raise SystemExit(f"未找到 {FULL_MAP}")
    reset_output()
    sprites = load_sprites()
    n_sprites = sum(len(v) for v in sprites.values())
    if n_sprites == 0:
        raise SystemExit(f"未找到任何 sprite in {SPRITES_DIR}")
    print(f"[1/3] 加载 {n_sprites} 个 sprite, {len(sprites)} 个怪物种类")

    print(f"[2/3] 生成 {args.count} 张合成图 ...")
    total_counter = Counter()
    for i in range(args.count):
        c = generate_one(i, sprites, rng)
        total_counter.update(c)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.count}]")

    print(f"[3/3] 写入 dataset.yaml")
    write_yaml()

    print(f"\n--- 各类框数 (累计 {sum(total_counter.values())}) ---")
    for cid in sorted(CLASS_NAMES):
        print(f"  {cid} ({CLASS_NAMES[cid]}): {total_counter[cid]}")
    print(f"\n[OK] 输出: {OUT}/")


if __name__ == "__main__":
    main()