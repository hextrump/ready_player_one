import os
import cv2
import numpy as np
import random
import yaml
import shutil
from pathlib import Path

# Paths
BG_IMG = 'data/ui/game_bg.png'
HP_IMG = 'data/ui/hp_image.png'
MP_IMG = 'data/ui/mp_image.png'
OUT_DIR = Path('data/yolo_ui_dataset')

# Remove existing
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

(OUT_DIR / 'images' / 'train').mkdir(parents=True)
(OUT_DIR / 'images' / 'val').mkdir(parents=True)
(OUT_DIR / 'labels' / 'train').mkdir(parents=True)
(OUT_DIR / 'labels' / 'val').mkdir(parents=True)

# Load templates (must have alpha channel if transparent, but here we just copy directly to avoid black boxes if possible)
bg_base = cv2.imread(BG_IMG)
hp_base = cv2.imread(HP_IMG)
mp_base = cv2.imread(MP_IMG)

if bg_base is None or hp_base is None or mp_base is None:
    print("Missing UI templates!")
    exit(1)

IMG_H, IMG_W = bg_base.shape[:2]

classes = {'HP_Bar': 0, 'MP_Bar': 1}
NUM_IMGS = 300  # More images for better robustness

print("Generating YOLO UI synthetic data...")

for i in range(NUM_IMGS):
    # random crop/bg from bg_base
    bg = bg_base.copy()
    
    # randomly shift background horizontally somewhat to simulate movement
    shift_x = random.randint(-400, 400)
    shift_y = random.randint(-200, 200)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    bg = cv2.warpAffine(bg, matrix, (IMG_W, IMG_H))
    
    # Add some noise/color jitter to background
    if random.random() > 0.5:
        # random brightness
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.5, 1.5)
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype=np.uint8)
        bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    labels = []
    
    # Place HP
    # Real UI elements might not resize in game, but small scale variation (0.9 to 1.1) helps robustness
    scale_hp = random.uniform(0.85, 1.15)
    nh_h, nh_w = int(hp_base.shape[0]*scale_hp), int(hp_base.shape[1]*scale_hp)
    hp_r = cv2.resize(hp_base, (nh_w, nh_h))
    
    # Randomly place HP in the bottom half of the screen usually, but sometimes anywhere
    x1_hp = random.randint(10, IMG_W//2 - nh_w)
    y1_hp = random.randint(IMG_H//2, IMG_H - nh_h - 10)
    
    # paste HP
    bg[y1_hp:y1_hp+nh_h, x1_hp:x1_hp+nh_w] = hp_r
    
    cx_hp, cy_hp = x1_hp + nh_w/2.0, y1_hp + nh_h/2.0
    labels.append(f"0 {cx_hp/IMG_W:.6f} {cy_hp/IMG_H:.6f} {nh_w/IMG_W:.6f} {nh_h/IMG_H:.6f}")
    
    # Place MP
    scale_mp = random.uniform(0.85, 1.15)
    nm_h, nm_w = int(mp_base.shape[0]*scale_mp), int(mp_base.shape[1]*scale_mp)
    mp_r = cv2.resize(mp_base, (nm_w, nm_h))
    
    # Normally MP is to the right of HP, or around lower right. Let's make it more random for robust learning
    x1_mp = random.randint(IMG_W//2 - 100, IMG_W - nm_w - 10)
    y1_mp = random.randint(IMG_H//2, IMG_H - nm_h - 10)
    
    bg[y1_mp:y1_mp+nm_h, x1_mp:x1_mp+nm_w] = mp_r
    
    cx_mp, cy_mp = x1_mp + nm_w/2.0, y1_mp + nm_h/2.0
    labels.append(f"1 {cx_mp/IMG_W:.6f} {cy_mp/IMG_H:.6f} {nm_w/IMG_W:.6f} {nm_h/IMG_H:.6f}")
    
    # Save
    split = 'train' if i < int(NUM_IMGS * 0.8) else 'val'
    img_name = f"{i:04d}.jpg"
    
    cv2.imwrite(str(OUT_DIR / 'images' / split / img_name), bg)
    with open(OUT_DIR / 'labels' / split / img_name.replace('jpg', 'txt'), 'w') as f:
        f.write('\n'.join(labels))

yaml_content = {
    'path': str(OUT_DIR.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'HP_Bar', 1: 'MP_Bar'}
}

with open(OUT_DIR / 'dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Generated {NUM_IMGS} images successfully.")
