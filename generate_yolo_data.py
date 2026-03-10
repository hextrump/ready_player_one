"""
YOLO Synthetic Training Data Generator
=======================================
把怪物精灵图（带透明通道）随机贴到游戏截图背景上，
自动生成 YOLOv8 格式的 (image + label) 训练数据。

输出结构:
  data/yolo_dataset/
    images/
      train/  (80%)
      val/    (20%)
    labels/
      train/
      val/
    dataset.yaml
"""
import cv2
import numpy as np
import os
import json
import random
import shutil
from pathlib import Path

# =========== CONFIG ===========
DB_DIR = "data/monster_db"
BG_DIR = "data/debug"  # game screenshots as backgrounds
OUTPUT_DIR = "data/yolo_dataset"
NUM_IMAGES = 1000      # total synthetic images to generate
MONSTERS_PER_IMG = (2, 8)  # random range of monsters per image
IMG_W, IMG_H = 1600, 900   # game resolution

# Which monsters to train on (must exist in monster_index.json)
TARGET_MONSTERS = ["Player", "Wild Boar", "Dark Axe Stump", "Pig", "Ribbon Pig"]
# ==============================

def load_sprites(target_names):
    """Load sprite PNGs with alpha channel from the database."""
    index_path = os.path.join(DB_DIR, "monster_index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    sprites = {}  # name -> list of (bgr, alpha) at original size
    class_map = {}  # name -> class_id
    
    for cls_id, name in enumerate(target_names):
        name_lower = name.lower()
        class_map[name] = cls_id
        
        for mob_id, info in index.items():
            if info['name'].lower() == name_lower:
                png_path = info.get('png')
                if not png_path or not os.path.exists(png_path):
                    continue
                
                img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[2] < 4:
                    continue
                
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                
                if name not in sprites:
                    sprites[name] = []
                sprites[name].append((bgr, alpha))
                print(f"  Loaded sprite: {name} ({bgr.shape[1]}x{bgr.shape[0]})")
                # Removed break to load all sprites for the same monster/player name
    
    return sprites, class_map

def load_backgrounds():
    """Load game screenshots to use as backgrounds."""
    bgs = []
    for f in os.listdir(BG_DIR):
        if f.endswith('.png') and 'motion' not in f and 'mask' not in f and 'edges' not in f:
            path = os.path.join(BG_DIR, f)
            img = cv2.imread(path)
            if img is not None and img.shape[0] >= 400 and img.shape[1] >= 400:
                # Resize to game resolution if needed
                if img.shape[:2] != (IMG_H, IMG_W):
                    img = cv2.resize(img, (IMG_W, IMG_H))
                bgs.append(img)
    
    if not bgs:
        # If no backgrounds, create a plain one
        print("  WARNING: No background images found, using solid color")
        bgs.append(np.full((IMG_H, IMG_W, 3), (180, 200, 210), dtype=np.uint8))
    
    print(f"  Loaded {len(bgs)} background images")
    return bgs

def paste_sprite(bg, sprite_bgr, sprite_alpha, scale, x, y):
    """Paste a sprite onto background at position (x, y) with given scale."""
    h, w = sprite_bgr.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w < 10 or new_h < 10:
        return None
    
    s_bgr = cv2.resize(sprite_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    s_alpha = cv2.resize(sprite_alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Clamp to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg.shape[1], x + new_w)
    y2 = min(bg.shape[0], y + new_h)
    
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Alpha blend
    alpha_roi = s_alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha_roi] * 3, axis=-1)
    
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
    fg_roi = s_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    
    blended = (fg_roi * alpha_3ch + bg_roi * (1 - alpha_3ch)).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended
    
    # Return YOLO format bbox (normalized)
    cx = ((x1 + x2) / 2.0) / bg.shape[1]
    cy = ((y1 + y2) / 2.0) / bg.shape[0]
    bw = (x2 - x1) / bg.shape[1]
    bh = (y2 - y1) / bg.shape[0]
    
    return (cx, cy, bw, bh)

def generate_dataset():
    print("=== YOLO Synthetic Data Generator ===")
    print(f"Target monsters: {TARGET_MONSTERS}")
    print(f"Images to generate: {NUM_IMAGES}")
    
    print("\nStep 1: Loading sprites...")
    sprites, class_map = load_sprites(TARGET_MONSTERS)
    if not sprites:
        print("ERROR: No sprites loaded!")
        return
    
    print("\nStep 2: Loading backgrounds...")
    backgrounds = load_backgrounds()
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    
    print(f"\nStep 3: Generating {NUM_IMAGES} synthetic training images...")
    
    for i in range(NUM_IMAGES):
        # Pick random background
        bg = random.choice(backgrounds).copy()
        
        # Add slight random augmentation to background
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        bg = np.clip(bg * brightness, 0, 255).astype(np.uint8)
        
        labels = []
        num_monsters = random.randint(*MONSTERS_PER_IMG)
        
        # 1. ALWAYS ADD EXACTLY ONE PLAYER
        if "Player" in sprites:
            cls_id = class_map["Player"]
            bgr, alpha = random.choice(sprites["Player"])
            scale = random.uniform(1.2, 1.8) # Player shouldn't be too artificially scaled
            x = random.randint(100, IMG_W - 100)
            y = random.randint(int(IMG_H * 0.4), IMG_H - 100)
            
            if random.random() > 0.5:
                bgr = cv2.flip(bgr, 1)
                alpha = cv2.flip(alpha, 1)
                
            bbox = paste_sprite(bg, bgr, alpha, scale, x, y)
            if bbox:
                cx, cy, bw, bh = bbox
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
        # 2. ADD MONSTERS
        monster_types = [k for k in sprites.keys() if k != "Player"]
        if monster_types:
            for _ in range(num_monsters):
                # Pick random monster type
                mon_name = random.choice(monster_types)
                cls_id = class_map[mon_name]
                bgr, alpha = random.choice(sprites[mon_name])
                
                # Random scale (game sprites are small icons, in-game they're ~1.5-2.5x bigger)
                scale = random.uniform(1.2, 2.8)
                
                # Random position (prefer lower half of screen where ground is)
                x = random.randint(0, IMG_W - 50)
                y = random.randint(int(IMG_H * 0.3), IMG_H - 30)
                
                # Random horizontal flip
                if random.random() > 0.5:
                    bgr = cv2.flip(bgr, 1)
                    alpha = cv2.flip(alpha, 1)
                
                bbox = paste_sprite(bg, bgr, alpha, scale, x, y)
                if bbox:
                    cx, cy, bw, bh = bbox
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        # Split: 80% train, 20% val
        split = 'train' if i < NUM_IMAGES * 0.8 else 'val'
        
        img_path = os.path.join(OUTPUT_DIR, 'images', split, f'syn_{i:04d}.png')
        lbl_path = os.path.join(OUTPUT_DIR, 'labels', split, f'syn_{i:04d}.txt')
        
        cv2.imwrite(img_path, bg)
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{NUM_IMAGES} images...")
    
    # Create dataset.yaml for YOLOv8
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

names:
"""
    for name, cls_id in class_map.items():
        yaml_content += f"  {cls_id}: {name}\n"
    
    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n=== DONE ===")
    print(f"Dataset: {os.path.abspath(OUTPUT_DIR)}")
    print(f"YAML: {yaml_path}")
    print(f"Classes: {class_map}")
    
    # Count
    train_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, 'images', 'train')))
    val_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, 'images', 'val')))
    print(f"Train: {train_imgs} images, Val: {val_imgs} images")
    print(f"\nNext step: Train with:")
    print(f"  yolo detect train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=640")

if __name__ == "__main__":
    generate_dataset()
