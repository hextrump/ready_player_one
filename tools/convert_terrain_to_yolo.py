"""
将 Live Annotator 标注的 JSON 数据转换为 YOLO 分割/检测数据集，并进行数据增强。

类别:
  0 = Platform (平台)
  1 = Rope (绳子/梯子)

标注格式转换:
  Platform: x_left, y, x_right → YOLO bbox (cx, cy, w, h) 归一化
  Rope:     x, y_top, y_bottom → YOLO bbox (cx, cy, w, h) 归一化
"""

import os
import json
import cv2
import numpy as np
import shutil
import random
import yaml
from pathlib import Path

SNAP_DIR = Path("data/terrain/snapshots")
OUT_DIR = Path("data/yolo_terrain_dataset")

# 清除旧数据
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split in ['train', 'val']:
    (OUT_DIR / 'images' / split).mkdir(parents=True)
    (OUT_DIR / 'labels' / split).mkdir(parents=True)

# 收集所有标注组
groups = []
for f in sorted(SNAP_DIR.glob("*_labels.json")):
    prefix = f.stem.replace("_labels", "")
    raw_img = SNAP_DIR / f"{prefix}_raw.png"
    if raw_img.exists():
        with open(f, 'r', encoding='utf-8') as fp:
            labels = json.load(fp)
        # 只收集非空标注
        if labels.get("platforms") or labels.get("ropes"):
            groups.append((raw_img, labels))

print(f"找到 {len(groups)} 组有效标注")

# 平台用一个有高度的 bbox (height=20px)，绳子用 width=15px
PLATFORM_HEIGHT = 20
ROPE_WIDTH = 15

def labels_to_yolo(labels, img_w, img_h):
    """将 JSON 标注转为 YOLO 格式的行"""
    lines = []
    for p in labels.get("platforms", []):
        x_left = p["x_left"]
        x_right = p["x_right"]
        y = p["y"]
        
        w = x_right - x_left
        h = PLATFORM_HEIGHT
        cx = (x_left + x_right) / 2.0
        cy = y
        
        # 归一化
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = w / img_w
        h_n = h / img_h
        
        # 边界保护
        cx_n = max(0, min(1, cx_n))
        cy_n = max(0, min(1, cy_n))
        w_n = max(0, min(1, w_n))
        h_n = max(0, min(1, h_n))
        
        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    
    for r in labels.get("ropes", []):
        x = r["x"]
        y_top = r["y_top"]
        y_bottom = r["y_bottom"]
        
        w = ROPE_WIDTH
        h = y_bottom - y_top
        cx = x
        cy = (y_top + y_bottom) / 2.0
        
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = w / img_w
        h_n = h / img_h
        
        cx_n = max(0, min(1, cx_n))
        cy_n = max(0, min(1, cy_n))
        w_n = max(0, min(1, w_n))
        h_n = max(0, min(1, h_n))
        
        lines.append(f"1 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    
    return lines

# 数据增强函数
def augment_image(img, labels_json, img_w, img_h, aug_type):
    """对图片和标注进行同步数据增强"""
    new_labels = json.loads(json.dumps(labels_json))  # deep copy
    
    if aug_type == "brightness":
        # 随机亮度调整
        factor = random.uniform(0.6, 1.4)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] *= factor
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif aug_type == "hflip":
        # 水平翻转
        img = cv2.flip(img, 1)
        for p in new_labels.get("platforms", []):
            old_left = p["x_left"]
            old_right = p["x_right"]
            p["x_left"] = img_w - old_right
            p["x_right"] = img_w - old_left
        for r in new_labels.get("ropes", []):
            r["x"] = img_w - r["x"]
    
    elif aug_type == "noise":
        # 高斯噪声
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == "blur":
        # 轻微模糊
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    elif aug_type == "crop_shift":
        # 随机裁切平移 (上下移动画面，模拟不同滚动位置)
        shift_y = random.randint(-30, 30)
        shift_x = random.randint(-30, 30)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (img_w, img_h))
        for p in new_labels.get("platforms", []):
            p["x_left"] += shift_x
            p["x_right"] += shift_x
            p["y"] += shift_y
        for r in new_labels.get("ropes", []):
            r["x"] += shift_x
            r["y_top"] += shift_y
            r["y_bottom"] += shift_y
    
    return img, new_labels

# 生成数据集
aug_types = ["brightness", "hflip", "noise", "blur", "crop_shift"]
NUM_AUGMENTS_PER_IMAGE = 8  # 每张原图生成8个增强版本

all_samples = []  # (img, yolo_lines)

for raw_path, labels in groups:
    img = cv2.imread(str(raw_path))
    if img is None:
        continue
    img_h, img_w = img.shape[:2]
    
    # 原图
    yolo_lines = labels_to_yolo(labels, img_w, img_h)
    if yolo_lines:
        all_samples.append((img.copy(), yolo_lines))
    
    # 增强
    for _ in range(NUM_AUGMENTS_PER_IMAGE):
        aug_type = random.choice(aug_types)
        aug_img, aug_labels = augment_image(img.copy(), labels, img_w, img_h, aug_type)
        aug_yolo = labels_to_yolo(aug_labels, img_w, img_h)
        if aug_yolo:
            all_samples.append((aug_img, aug_yolo))

# 打乱并分割 train/val (80/20)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.8)

for i, (img, yolo_lines) in enumerate(all_samples):
    split = "train" if i < split_idx else "val"
    name = f"{i:04d}"
    
    cv2.imwrite(str(OUT_DIR / 'images' / split / f"{name}.jpg"), img)
    with open(OUT_DIR / 'labels' / split / f"{name}.txt", 'w') as f:
        f.write('\n'.join(yolo_lines))

# 数据集配置文件
yaml_content = {
    'path': str(OUT_DIR.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'Platform', 1: 'Rope'}
}
with open(OUT_DIR / 'dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

train_count = split_idx
val_count = len(all_samples) - split_idx
total_platforms = sum(1 for _, lines in all_samples for l in lines if l.startswith("0 "))
total_ropes = sum(1 for _, lines in all_samples for l in lines if l.startswith("1 "))

print(f"\n{'='*50}")
print(f"数据集生成完成！")
print(f"  总样本数: {len(all_samples)}")
print(f"  训练集: {train_count} | 验证集: {val_count}")
print(f"  总平台标注: {total_platforms} | 总绳子标注: {total_ropes}")
print(f"  保存位置: {OUT_DIR.absolute()}")
print(f"{'='*50}")
