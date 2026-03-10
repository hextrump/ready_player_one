import os
import json
import cv2
import numpy as np
import shutil
import random
import yaml
from pathlib import Path

SNAP_DIR = Path("data/entity/snapshots")
OUT_DIR = Path("data/yolo_monster_dataset")

# OVERRIDE: ONLY extract "Monster" class
CLASS_MAP = {
    "Monster": 0
}

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split in ['train', 'val']:
    (OUT_DIR / 'images' / split).mkdir(parents=True)
    (OUT_DIR / 'labels' / split).mkdir(parents=True)

groups = []
for f in sorted(SNAP_DIR.glob("*.json")):
    prefix = f.stem
    raw_img = SNAP_DIR / f"{prefix.replace('_labels', '')}_raw.png"
    if not raw_img.exists():
        raw_img = SNAP_DIR / f"{prefix}_raw.png"
        if not raw_img.exists():
            continue
            
    with open(f, 'r', encoding='utf-8') as fp:
        labels = json.load(fp)
    
    if labels.get("boxes"):
        groups.append((raw_img, labels))

print(f"[Data] Found {len(groups)} labeled images for Monster-only dataset")

def labels_to_yolo(labels, img_w, img_h):
    lines = []
    for b in labels.get("boxes", []):
        cls_name = b["cls"]
        if cls_name not in CLASS_MAP: continue
        
        cls_id = CLASS_MAP[cls_name]
        
        x1 = max(0, min(img_w, b["x1"]))
        y1 = max(0, min(img_h, b["y1"]))
        x2 = max(0, min(img_w, b["x2"]))
        y2 = max(0, min(img_h, b["y2"]))
        
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = w / img_w
        h_n = h / img_h
        
        lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines

def augment_image(img, labels_json, img_w, img_h, aug_type):
    new_labels = json.loads(json.dumps(labels_json))
    
    if aug_type == "brightness":
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] *= factor
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    elif aug_type == "noise":
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif aug_type == "blur":
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    elif aug_type == "shift":
        shift_x = random.randint(-300, 300)
        shift_y = random.randint(-300, 300)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (img_w, img_h))
        for b in new_labels.get("boxes", []):
            b["x1"] += shift_x
            b["x2"] += shift_x
            b["y1"] += shift_y
            b["y2"] += shift_y
            
    return img, new_labels

aug_types = ["brightness", "noise", "blur", "shift", "none"]

all_samples = []
for raw_path, labels in groups:
    img = cv2.imread(str(raw_path))
    if img is None: continue
    img_h, img_w = img.shape[:2]
    
    base_yolo = labels_to_yolo(labels, img_w, img_h)
    if base_yolo:
        all_samples.append((img.copy(), base_yolo))
        
    n_augs = 20 # fixed since we don't care about ui balance
    
    for _ in range(n_augs):
        aug_type = random.choice(aug_types)
        if aug_type == "none": continue
        a_img, a_labels = augment_image(img.copy(), labels, img_w, img_h, aug_type)
        a_yolo = labels_to_yolo(a_labels, img_w, img_h)
        if a_yolo:
            all_samples.append((a_img, a_yolo))

random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.8)

stats = {0: 0}

for i, (img, yolo_lines) in enumerate(all_samples):
    split = "train" if i < split_idx else "val"
    name = f"{i:04d}"
    cv2.imwrite(str(OUT_DIR / 'images' / split / f"{name}.jpg"), img)
    with open(OUT_DIR / 'labels' / split / f"{name}.txt", 'w') as f:
        f.write('\n'.join(yolo_lines))
        
    for l in yolo_lines:
        cls_id = int(l.split(" ")[0])
        stats[cls_id] += 1

yaml_content = {
    'path': str(OUT_DIR.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'Monster'}
}
with open(OUT_DIR / 'dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print("\n--- [YOLO Monster-Only Dataset summary] ---")
print(f"Total samples: {len(all_samples)}")
for k, v in CLASS_MAP.items():
    print(f"  {k}: {stats[v]} boxes")
